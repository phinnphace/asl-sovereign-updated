import torch
import os
import random
import pandas as pd
from PIL import Image
from transformers import AutoProcessor, AutoModelForMultimodalLM, BitsAndBytesConfig

# --- 1. CONFIGURATION ---
MODEL_ID = "google/gemma-4-E4B-it"
DATA_PATH = r"C:\ASL_Project\raw_data\train"
OUTPUT_FILE = "asl_hallucination_baseline.csv"

# --- 2. HARDWARE SETUP (Optimized for RTX 3070) ---
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)

# --- 3. THE HEAVY LOAD ---
print("Loading Gemma-4 E4B into RTX 3070 VRAM...")
model = AutoModelForMultimodalLM.from_pretrained(
    MODEL_ID,
    quantization_config=quant_config,
    device_map="cuda:0"
)
processor = AutoProcessor.from_pretrained(MODEL_ID)

# --- 4. DATA DISCOVERY ---
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Check your path: {DATA_PATH}")

folders = [f for f in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, f))]

# --- 5. THE AUDIT LOOP ---
audit_data = []
print(f"Starting Audit of {len(folders)} letters...")

for letter in sorted(folders):
    folder_full_path = os.path.join(DATA_PATH, letter)
    images = [f for f in os.listdir(folder_full_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not images:
        continue
    
    sample_img_name = random.choice(images)
    
    # FIX 1: Strip alpha channels and force 3-channel RGB to prevent silent failures.
    img = Image.open(os.path.join(folder_full_path, sample_img_name)).convert("RGB")

    # RIGOROUS PROMPT WRAPPER
    messages = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": "Identify this ASL letter. Describe the finger positions in detail (<|think|>), then provide the final letter."}
        ]}
    ]
    
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    
    # FIX 2: The Sanity Check. Force the token if the chat template fails to map the dictionary.
    if "<image>" not in prompt and "<|image|>" not in prompt:
        prompt = "<image>\nIdentify this ASL letter. Describe the finger positions in detail (<|think|>), then provide the final letter."
    
    # FIX 3: Explicitly pass the image as a list.
    inputs = processor(text=prompt, images=[img], return_tensors="pt")
    
    # FIX 4: Ensure vision tensors strictly match the 4-bit compute dtype (bfloat16).
    inputs = {
        k: v.to(model.device, dtype=torch.bfloat16) if k == "pixel_values" else v.to(model.device) 
        for k, v in inputs.items() 
        if isinstance(v, torch.Tensor)
    }

    # Generate the documentation
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=600)
    
    # Clean the output (removing the prompt tokens)
    response = processor.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    audit_data.append({"actual_letter": letter, "model_output": response})
    print(f"Captured: {letter}")
    
    # --- DRY RUN HALT ---
    # Stops the loop immediately after processing the first valid image.
    break

# --- 6. SAVE RESULTS ---
df = pd.DataFrame(audit_data)
df.to_csv(OUTPUT_FILE, index=False)
print(f"\nAudit Complete. Results saved to: {OUTPUT_FILE}")