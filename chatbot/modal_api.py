import modal
import os
import json
import io
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── 1. Modal App + Volume ──────────────────────────────────────────────────────
app = modal.App("asl-decoder-cloud")
model_volume = modal.Volume.from_name("asl-model-volume")

# ── 2. Container image ────────────────────────────────────────────────────────
cuda_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.1.1-devel-ubuntu22.04",
        add_python="3.11",
    )
    .pip_install("fastapi", "pydantic", "gspread", "google-auth", "pandas")
    .run_commands(
        "pip install llama-cpp-python --extra-index-url "
        "https://abetlen.github.io/llama-cpp-python/whl/cu121"
    )
)

# ── 3. ASGI app (lives outside the Modal function so CORS is always applied) ──
web_app = FastAPI(title="ASL Provenance Decoder API")

web_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class DiagnosisRequest(BaseModel):
    user_text: str


# ── 4. Serverless GPU function ─────────────────────────────────────────────────
@app.function(
    image=cuda_image,
    gpu="L4",
    volumes={"/data": model_volume},
    secrets=[modal.Secret.from_name("google-sheets-auth")],  # injects GCP_SERVICE_ACCOUNT_JSON + SHEETS_ID
    min_containers=0,   # sleeps when idle, wakes on request
    timeout=120,
)
@modal.asgi_app()
def fastapi_app():
    from llama_cpp import Llama
    import gspread
    from google.oauth2.service_account import Credentials

    # ── Load model ────────────────────────────────────────────────────────────
    print("Loading Gemma2 GGUF model into VRAM…")
    llm = Llama(
        model_path="/data/gemma2_asl_finetuned-unsloth.Q4_K_M.gguf",
        n_gpu_layers=-1,
        n_ctx=2048,
        verbose=False,
    )

    # ── Google Sheets (optional — app still works if creds missing) ───────────
    SHEET_ID = os.environ.get("SHEETS_ID", "1DAgXOA3pJIWJqoKcJApG9wg9E8f5qcj1VZCuK4i0mks")
    GCP_CREDS = None
    try:
        scopes = [
            "https://spreadsheets.google.com/feeds",
            "https://www.googleapis.com/auth/drive",
        ]
        GCP_CREDS = Credentials.from_service_account_file("/data/credentials.json", scopes=scopes)
    except Exception as e:
        print(f"Sheets auth failed: {e}")

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
        except Exception as e:
            print(f"Sheet log failed: {e}")

    def get_diag_count() -> int:
        try:
            if not GCP_CREDS or not SHEET_ID:
                return 0
            client = gspread.authorize(GCP_CREDS)
            ws = client.open_by_key(SHEET_ID).worksheet("diagnoses")
            return max(0, len(ws.get_all_values()) - 1)
        except Exception:
            return 0

    INSTRUCTION = "Diagnose the training data provenance based on the following user complaint or matrix topology."

    def run_inference(user_text: str) -> dict:
        prompt = (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{INSTRUCTION}\n\n"
            f"### Input:\n{user_text}\n\n"
            "### Response:\n"
        )
        output = llm(
            prompt,
            max_tokens=512,
            stop=["### Instruction:", "### Input:", "\n\n\n"],
            repeat_penalty=1.3,
            temperature=0.7,
            top_p=0.9,
        )
        text = output["choices"][0]["text"].strip()

        # Try to parse as JSON
        try:
            parsed = json.loads(text)
        except Exception:
            # Extract JSON block if wrapped in text
            import re
            match = re.search(r'\{.*\}', text, re.DOTALL)
            parsed = json.loads(match.group()) if match else {}

        return {
            "response": parsed.get("provenance_diagnosis", text),
            "failure_mode": parsed.get("failure_mode", ""),
            "training_regime": parsed.get("failure_mode", ""),
            "deployment_risk": parsed.get("compute_saved", ""),
            "firing_pairs": [],
            "recommendation": parsed.get("recommendation", ""),
        }
        text = output["choices"][0]["text"].strip()

        # Best-effort structured extraction
        failure_mode = ""
        training_regime = ""
        deployment_risk = ""
        if "landmark" in text.lower():
            failure_mode = "landmark-only"
            training_regime = "MediaPipe 21-point landmark"
            deployment_risk = "High — body-anchored and two-hand signs will fail"
        elif "lab" in text.lower() or "controlled" in text.lower():
            failure_mode = "lab-curated"
            training_regime = "Controlled studio dataset"
            deployment_risk = "Medium — real-world signer diversity not represented"

        return {
            "response": text,
            "failure_mode": failure_mode,
            "training_regime": training_regime,
            "deployment_risk": deployment_risk,
            "firing_pairs": [],
            "recommendation": text[:300],
        }

    # ── Routes ────────────────────────────────────────────────────────────────

    @web_app.get("/api/health")
    def health():
        return {"status": "ok", "model": "gemma2_asl_finetuned", "sheets": bool(GCP_CREDS)}

    @web_app.get("/api/count")
    def count():
        return {"count": get_diag_count()}

    @web_app.post("/api/diagnose")
    def diagnose(req: DiagnosisRequest):
        if not req.user_text.strip():
            raise HTTPException(status_code=400, detail="Empty input")
        result = run_inference(req.user_text)
        log_to_sheet(req.user_text, result)
        return result

    return web_app
