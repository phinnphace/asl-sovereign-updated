"""
backend/main.py — Provenance Decoder Ring API
FastAPI wrapper around provenance_decoder.py
Deploy to HuggingFace Spaces or Railway.
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime
import json
import os
import io

from provenance_decoder import analyze_failure_mode, decode_csv

app = FastAPI(title="Provenance Decoder Ring API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten to your Vercel URL in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Config from env vars ──────────────────────────────────────────────────────
API_KEY  = os.environ.get("GOOGLE_API_KEY", "")
SHEET_ID = os.environ.get("SHEETS_ID", "")

GCP_CREDS = None
try:
    creds_json = os.environ.get("GCP_SERVICE_ACCOUNT_JSON", "")
    if creds_json:
        creds_dict = json.loads(creds_json)
        scopes = [
            "https://spreadsheets.google.com/feeds",
            "https://www.googleapis.com/auth/drive",
        ]
        GCP_CREDS = Credentials.from_service_account_info(creds_dict, scopes=scopes)
except Exception:
    pass


def log_to_sheet(user_input: str, result: dict):
    try:
        if not GCP_CREDS or not SHEET_ID:
            return
        client = gspread.authorize(GCP_CREDS)
        ws = client.open_by_key(SHEET_ID).worksheet("diagnoses")
        ws.append_row([
            datetime.utcnow().isoformat(),
            user_input[:500],
            result.get("failure_mode", ""),
            result.get("training_regime", ""),
            result.get("deployment_risk", ""),
            json.dumps(result.get("firing_pairs", [])),
            result.get("recommendation", "")[:300],
        ])
    except Exception:
        pass


def get_diag_count() -> int:
    try:
        if not GCP_CREDS or not SHEET_ID:
            return 0
        client = gspread.authorize(GCP_CREDS)
        ws = client.open_by_key(SHEET_ID).worksheet("diagnoses")
        return max(0, len(ws.get_all_values()) - 1)
    except Exception:
        return 0


# ── Routes ────────────────────────────────────────────────────────────────────

class DiagnoseRequest(BaseModel):
    text: str

@app.get("/health")
def health():
    return {"status": "ok", "api_key_set": bool(API_KEY)}

@app.get("/count")
def count():
    return {"count": get_diag_count()}

@app.post("/diagnose")
def diagnose(req: DiagnoseRequest):
    if not API_KEY:
        raise HTTPException(status_code=500, detail="GOOGLE_API_KEY not set")
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Empty input")
    result = analyze_failure_mode(req.text, API_KEY)
    result["summary"] = result.get("provenance_diagnosis", "Decoded.")[:120] + "..."
    log_to_sheet(req.text, result)
    return result

@app.post("/decode-csv")
async def decode_csv_endpoint(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="CSV files only")
    contents = await file.read()
    df = pd.read_csv(io.BytesIO(contents))
    summary = decode_csv(df)

    # Auto-diagnose if we have top errors
    result = {"type": summary.get("type"), "summary": "CSV decoded."}
    if summary.get("type") == "inference_log" and summary.get("top_errors") and API_KEY:
        acc = summary["overall_accuracy"]
        worst = sorted(summary["top_errors"].items(), key=lambda x: len(x[1]), reverse=True)[:3]
        error_desc = "; ".join(f"{l} confused with {', '.join(v)}" for l, v in worst)
        auto_input = f"Per-letter accuracy: overall {acc}%. Top errors: {error_desc}."
        diagnosis = analyze_failure_mode(auto_input, API_KEY)
        log_to_sheet(auto_input, diagnosis)
        result.update(diagnosis)
        result["summary"] = f"Overall {acc}% accuracy. {diagnosis.get('training_regime', '')} detected."
        result["per_letter"] = summary.get("per_letter", {})

    return result
