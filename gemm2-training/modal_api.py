import os
import json
import gspread
from fastapi import FastAPI
from pydantic import BaseModel

# 1. Define the App and connect your Cloud Drive
app = modal.App("asl-decoder-cloud")
model_volume = modal.Volume.from_name("asl-model-volume")

# 2. Build the Environment
cuda_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("fastapi", "pydantic", "gspread", "google-auth")
    # We drop the C++ compilers and just grab the pre-built CUDA package directly
    .run_commands("pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121")
)

web_app = FastAPI()

# Defeat CORS immediately so Vercel can talk to it
web_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class DiagnosisRequest(BaseModel):
    user_text: str

# 3. Define the Serverless GPU Endpoint
@app.function(
    image=cuda_image,
    gpu="L4", 
    volumes={"/data": model_volume}, 
    min_containers=0, # Replaced keep_warm=0
    timeout=120
)
@modal.asgi_app()
def fastapi_app():
    # We import Llama and Sheets tools *inside* the container
    from llama_cpp import Llama
    import gspread

    # Load the brain into the cloud GPU
    print("Loading 5GB GGUF Model into VRAM...")
    llm = Llama(
        model_path="/data/gemma2_asl_finetuned-unsloth.Q4_K_M.gguf",
        n_gpu_layers=-1, 
        n_ctx=2048,
        verbose=False
    )
    
    import os
import json
import gspread
from fastapi import FastAPI
from pydantic import BaseModel

# --- [Your other setup code and Pydantic models stay here] ---

# Connect to Google Sheets securely using Modal's encrypted vault
# We pull the raw JSON string from the environment, convert it to a Python dictionary, 
# and pass it directly to gspread. No physical files on the hard drive.
creds_json_string = os.environ["GOOGLE_CREDS"]
creds_dict = json.loads(creds_json_string)
gc = gspread.service_account_from_dict(creds_dict)

# Replace with your actual Google Sheet exact name!
sheet = gc.open("decode").sheet1 

@web_app.post("/api/diagnose")
async def diagnose_asl(request: DiagnosisRequest):
    # The rest of your route logic stays exactly the same