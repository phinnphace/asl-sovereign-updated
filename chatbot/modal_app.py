import modal
import os
import json
import re
import time
from datetime import datetime
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException
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
    .pip_install("fastapi", "pydantic", "gspread", "google-auth")
    .run_commands(
        "pip install llama-cpp-python==0.3.6 --extra-index-url "
        "https://abetlen.github.io/llama-cpp-python/whl/cu121"
    )
)

# ── 3. ASGI app (module level — CORS always applied) ───────────────────────────
web_app = FastAPI(title="Decoder Ring Chat API")

web_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── 4. Request/Response Models ────────────────────────────────────────────────

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage] = []

class ChatResponse(BaseModel):
    reply: str
    evidence_grid: Optional[Dict[str, Any]] = None
    conversation_complete: bool = False
    turn_count: int = 0


# ── 5. Serverless GPU function ─────────────────────────────────────────────────
@app.function(
    image=cuda_image,
    gpu="L4",
    volumes={"data": model_volume},
    secrets=[
        model.Secret.from_name("google-sheets-auth"),
        model.Secret.from_name("System-prompt"),   # ← Fixed name
    ],
    min_containers=0,
    timeout=120,
)
@modal.asgi_app()
def fastapi_app():
    from llama_cpp import Llama
    import gspread
    from google.oauth2.service_account import Credentials

    # ── Load system prompt from secret ─────────────────────────────────────
systemprompt = os.environ.get("systemprompt", "")
if not systemprompt:
    print("WARNING: systemprompt secret is empty or missing")

    # ── Load model ─────────────────────────────────────────────────────────
    MODEL_PATH = "/data/Qwen2.5-7B-Instruct-Q4_K_M.gguf"
    print(f"Loading model from {MODEL_PATH}…")
    llm = Llama(
        model_path=MODEL_PATH,
        n_gpu_layers=-1,
        n_ctx=8192,
        verbose=False,
    )
    print("Model loaded.")

    # ── Google Sheets ──────────────────────────────────────────────────────
SHEET_ID = os.environ.get("SHEETS_ID", "")
GCP_CREDS = None

try:
    import json
    from google.oauth2.service_account import Credentials
    
    # Get the credentials JSON from the secret
    gcp_json = os.environ.get("CREDENTIALS_JSON", "")
    
    if gcp_json:
        creds_dict = json.loads(gcp_json)
        scopes = [
            "https://spreadsheets.google.com/feeds",
            "https://www.googleapis.com/auth/drive",
        ]
        GCP_CREDS = Credentials.from_service_account_info(creds_dict, scopes=scopes)
        print("Sheets auth: OK")
    else:
        print("Sheets auth skipped: CREDENTIALS_JSON not set")
        
except Exception as e:
    print(f"Sheets auth error: {e}")
    # ── Evidence Grid Parser ───────────────────────────────────────────────

    def parse_evidence_grid(text: str) -> Optional[Dict[str, Any]]:
        start_marker = "[EVIDENCE_GRID]"
        end_marker = "[/EVIDENCE_GRID]"
        start_idx = text.find(start_marker)
        end_idx = text.find(end_marker)
        if start_idx == -1 or end_idx == -1 or end_idx <= start_idx:
            return None

        block = text[start_idx + len(start_marker):end_idx].strip()
        fields = {}
        for line in block.split("\n"):
            line = line.strip()
            if ":" not in line:
                continue
            key, _, value = line.partition(":")
            fields[key.strip()] = value.strip()

        try:
            def b(k):
                return fields.get(k, "false").lower() == "true"

            def arr(k):
                v = fields.get(k, "null").lower()
                if v == "null":
                    return None
                cleaned = v.strip("[]")
                if not cleaned:
                    return None
                items = [i.strip().strip('"').strip("'") for i in cleaned.split(",")]
                items = [i for i in items if i]
                return items if items else None

            def opt(k, allowed):
                v = fields.get(k, "null").lower()
                return v if v in allowed else None

            def s(k):
                v = fields.get(k, "")
                if v.startswith('"') and v.endswith('"'):
                    v = v[1:-1]
                return v

            grid = {
                "A_specific_letters_named": b("A_specific_letters_named"),
                "A_letter_details": arr("A_letter_details"),
                "B_timing_complaint": b("B_timing_complaint"),
                "B_timing_details": opt("B_timing_details", [
                    "slowing_down_helps", "transitions_fail",
                    "speed_dependent", "JZ_fail"
                ]),
                "C_face_body_ignored": b("C_face_body_ignored"),
                "D_signer_exclusion": b("D_signer_exclusion"),
                "D_exclusion_details": opt("D_exclusion_details", [
                    "regional", "skin_tone", "handedness", "camera",
                    "age", "disability", "fluency_path", "style_rejected"
                ]),
                "E_randomness_inconsistency": b("E_randomness_inconsistency"),
                "F_emotion_only_no_technical_detail": b("F_emotion_only_no_technical_detail"),
                "user_verbatim_quote": s("user_verbatim_quote"),
            }

            # Cross-field constraints
            if not grid["A_specific_letters_named"]:
                grid["A_letter_details"] = None
            if not grid["B_timing_complaint"]:
                grid["B_timing_details"] = None
            if not grid["D_signer_exclusion"]:
                grid["D_exclusion_details"] = None

            return grid
        except Exception as e:
            print(f"Grid parse error: {e}")
            return None

    # ── Conversation Runner ─────────────────────────────────────────────────

    def run_chat(messages: List[Dict[str, str]]) -> Dict[str, Any]:
        full_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        full_messages.extend(messages)

        user_turns = sum(1 for m in messages if m.get("role") == "user")

        start = time.time()
        output = llm.create_chat_completion(
            messages=full_messages,
            max_tokens=512,
            temperature=0.7,
            top_p=0.9,
            repeat_penalty=1.1,
        )
        elapsed = time.time() - start

        reply = output["choices"][0]["message"]["content"].strip()
        finish = output["choices"][0].get("finish_reason", "?")
        print(f"Turn {user_turns} ({elapsed:.1f}s, {finish}): {reply[:120]}...")

        grid = parse_evidence_grid(reply)
        complete = grid is not None

        if complete and grid:
            log_to_sheet(messages, grid)

        return {
            "reply": reply,
            "evidence_grid": grid,
            "conversation_complete": complete,
            "turn_count": user_turns,
        }

    # ── Sheets Logging ─────────────────────────────────────────────────────

    def log_to_sheet(messages: List[Dict[str, str]], grid: Dict[str, Any]):
        try:
            if not GCP_CREDS or not SHEET_ID:
                return
            user_texts = [m["content"] for m in messages if m.get("role") == "user"]
            client = gspread.authorize(GCP_CREDS)
            ws = client.open_by_key(SHEET_ID).worksheet("diagnoses")
            ws.append_row([
                datetime.utcnow().isoformat(),
                " | ".join(user_texts)[:1000],
                json.dumps(grid.get("A_letter_details") or []),
                grid.get("user_verbatim_quote", ""),
                grid.get("A_specific_letters_named", False),
                grid.get("B_timing_complaint", False),
                grid.get("B_timing_details", ""),
                grid.get("C_face_body_ignored", False),
                grid.get("D_signer_exclusion", False),
                grid.get("D_exclusion_details", ""),
                grid.get("E_randomness_inconsistency", False),
                grid.get("F_emotion_only_no_technical_detail", False),
            ])
            print("✓ Sheet logged")
        except Exception as e:
            print(f"Sheet error: {e}")

    # ── Routes ────────────────────────────────────────────────────────────

    @web_app.get("/api/health")
    def health():
        return {
            "status": "ok",
            "model": "Qwen2.5-7B-Instruct",
            "sheets": bool(GCP_CREDS),
            "prompt_loaded": bool(SYSTEM_PROMPT),
            "prompt_length": len(SYSTEM_PROMPT),
        }

    @web_app.post("/api/chat")
    def chat(req: ChatRequest):
        messages = [{"role": m.role, "content": m.content} for m in req.messages]
        result = run_chat(messages)
        return ChatResponse(
            reply=result["reply"],
            evidence_grid=result["evidence_grid"],
            conversation_complete=result["conversation_complete"],
            turn_count=result["turn_count"],
        )

    return web_app