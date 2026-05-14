import modal
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
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
    
    # Connect to Google Sheets securely
    gc = gspread.service_account(filename='/data/credentials.json')
    # Replace with your actual Google Sheet exact name!
    sheet = gc.open("ASL_Diagnosis_Log").sheet1 

    @web_app.post("/api/diagnose")
    async def diagnose_asl(request: DiagnosisRequest):
        try:
            # Recreate your Gemma 2 Instruct template manually
            prompt = f"<start_of_turn>user\n{request.user_text}<end_of_turn>\n<start_of_turn>model\n"
            
            # Run the inference
            output = llm(
                prompt,
                max_tokens=500,
                stop=["<end_of_turn>"],
                echo=False
            )
            
            diagnosis_result = output["choices"][0]["text"].strip()
            
            # Log it to Google Sheets
            sheet.append_row([request.user_text, diagnosis_result])
            
            return {"diagnosis": diagnosis_result}
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
            
    return web_app