import os
import base64
import requests
import pandas as pd
from PIL import Image
from io import BytesIO
import random

# --- CONFIGURATION ---
OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_ID = "gemma4:e4b"
DATA_PATH = r"C:\ASL_Project\raw_data\train"
OUTPUT_FILE = "asl_prompt_ab_test.csv"
TEST_LETTER = "A"
SAMPLE_SIZE = 10  # Run 10 images per prompt variant

# --- PROMPTS ---
PROMPT_A = (
    "Identify this ASL letter. Describe the finger positions in detail, "
    "including palm orientation and thumb placement, then provide the final letter."
)

PROMPT_B = "What letter of the ASL alphabet does this hand show? Answer with just the letter."

PROMPT_C = "What ASL letter is this?"

PROMPT_D = "Look at this ASL hand image carefully. Think through what you see, then answer with just the single letter it represents."

PROMPTS = {
    "elaborate": PROMPT_A,
    "simple": PROMPT_B,
    "minimal": PROMPT_C,
    "cot_simple": PROMPT_D
}

# --- HELPER ---
def image_to_base64(img: Image.Image) -> str:
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

def query_ollama(image_b64: str, prompt_text: str, system: str = None) -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({
        "role": "user",
        "content": prompt_text,
        "images": [image_b64]
    })

    payload = {
        "model": MODEL_ID,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": 0.1,  # Low temp for classification
            "top_p": 0.95,
            "top_k": 64
        }
    }

    response = requests.post(OLLAMA_URL, json=payload, timeout=300)
    response.raise_for_status()
    return response.json()["message"]["content"]

# --- LOAD IMAGES ---
folder_path = os.path.join(DATA_PATH, TEST_LETTER)
all_images = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
sample_images = random.sample(all_images, min(SAMPLE_SIZE, len(all_images)))

print(f"Testing {len(sample_images)} images of letter '{TEST_LETTER}' across {len(PROMPTS)} prompt variants\n")

# --- RUN TEST ---
results = []

for img_name in sample_images:
    img = Image.open(os.path.join(folder_path, img_name)).convert("RGB")
    image_b64 = image_to_base64(img)

    for prompt_name, prompt_text in PROMPTS.items():
        try:
            response = query_ollama(image_b64, prompt_text)
        except Exception as e:
            response = f"ERROR: {str(e)}"

        results.append({
            "actual_letter": TEST_LETTER,
            "image_file": img_name,
            "prompt_variant": prompt_name,
            "model_output": response
        })
        print(f"[{TEST_LETTER}] [{prompt_name}] {img_name[:30]}... → {response[:80]}")

# --- SAVE ---
df = pd.DataFrame(results)
df.to_csv(OUTPUT_FILE, index=False)
print(f"\nA/B test complete. Results saved to {OUTPUT_FILE}")
