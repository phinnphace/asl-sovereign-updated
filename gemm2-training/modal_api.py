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
    modal.Image.debian_slim(python_version="3.11")
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
    min_containers=0,   # sleeps when idle, wakes on request
    timeout=120,
)
@modal.asgi_app()
def fastapi_app():
    from llama_cpp import Llama
    import gspread

    # ── Load model ────────────────────────────────────────────────────────────
    print("Loading Gemma2 GGUF model into VRAM…")
    llm = Llama(
        model_path="/data/gemma2_asl_finetuned-unsloth.Q4_K_M.gguf",
        n_gpu_layers=-1,
        n_ctx=2048,
        verbose=False,
    )

    # ── Google Sheets (optional — app still works if creds are missing) ───────
    sheet = None
    try:
        creds_raw = os.environ.get("GOOGLE_CREDS", "")
        if creds_raw:
            creds_dict = json.loads(creds_raw)
            gc = gspread.service_account_from_dict(creds_dict)
            sheet = gc.open("decode").sheet1
            print("Google Sheets connected.")
        else:
            print("GOOGLE_CREDS not set — sheet logging disabled.")
    except Exception as e:
        print(f"Google Sheets setup failed: {e} — logging disabled.")

    def _log(user_input: str, diagnosis: str):
        try:
            if sheet:
                sheet.append_row([
                    datetime.utcnow().isoformat(),
                    user_input[:500],
                    diagnosis[:500],
                ])
        except Exception as e:
            print(f"Sheet append failed: {e}")

    def _count() -> int:
        try:
            if sheet:
                return max(0, len(sheet.get_all_values()) - 1)
        except Exception:
            pass
        return 0

    # ── Routes ────────────────────────────────────────────────────────────────

    @web_app.get("/api/health")
    def health():
        return {"status": "ok", "model": "gemma2-asl-finetuned"}

    @web_app.get("/api/count")
    def count():
        return {"count": _count()}

    @web_app.post("/api/diagnose")
    async def diagnose(request: DiagnosisRequest):
        if not request.user_text.strip():
            raise HTTPException(status_code=400, detail="Empty input")

        prompt = (
            "<start_of_turn>user\n"
            "You are the ASL Provenance Decoder Ring. "
            "A researcher describes what their sign language model gets wrong. "
            "Identify the training data fingerprint.\n\n"
            f"Description: {request.user_text}\n\n"
            "Reply with:\n"
            "training_regime: LANDMARK-ONLY | FULL LANDMARKS | HYBRID\n"
            "firing_pairs: the specific letter pairs that are confused (e.g. K→V, M/N/T)\n"
            "provenance_diagnosis: 2–3 plain-English sentences explaining the fingerprint\n"
            "recommendation: what this means for how the model should be used\n"
            "<end_of_turn>\n"
            "<start_of_turn>model\n"
        )

        response = llm(prompt, max_tokens=512, stop=["<end_of_turn>"])
        output = response["choices"][0]["text"].strip()

        # Pull structured fields out of free text where possible
        regime = None
        for label in ("LANDMARK-ONLY", "FULL LANDMARKS", "HYBRID"):
            if label in output.upper():
                regime = label
                break

        result = {
            "diagnosis": output,
            "training_regime": regime,
            "firing_pairs": [],
            "provenance_diagnosis": output,
            "recommendation": "See diagnosis above.",
        }

        _log(request.user_text, output)
        return result

    @web_app.post("/api/decode-csv")
    async def decode_csv(file: UploadFile = File(...)):
        import pandas as pd

        if not file.filename.endswith(".csv"):
            raise HTTPException(status_code=400, detail="CSV files only")

        contents = await file.read()
        try:
            df = pd.read_csv(io.BytesIO(contents))
        except Exception:
            raise HTTPException(status_code=400, detail="Could not parse CSV")

        # Build a plain-language summary to pass to the model
        cols = df.columns.tolist()
        auto_input = f"CSV with {len(df)} rows. Columns: {', '.join(cols[:10])}."

        # If it looks like an inference log, pull top errors
        if {"predicted", "actual"}.issubset({c.lower() for c in cols}):
            pred_col = next(c for c in cols if c.lower() == "predicted")
            act_col  = next(c for c in cols if c.lower() == "actual")
            wrong = df[df[pred_col] != df[act_col]]
            if not wrong.empty:
                pairs = (
                    wrong.groupby([act_col, pred_col])
                    .size()
                    .sort_values(ascending=False)
                    .head(5)
                )
                pair_str = ", ".join(f"{a}→{p}" for (a, p), _ in pairs.items())
                auto_input = f"Top confusion pairs from CSV: {pair_str}."

        # Run through the model
        fake_req = DiagnosisRequest(user_text=auto_input)
        result = await diagnose(fake_req)
        return result

    return web_app
