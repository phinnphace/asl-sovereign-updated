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

    # ── Google Sheets (optional — app still w