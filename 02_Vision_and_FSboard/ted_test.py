import torch
from PIL import Image
from transformers import Gemma4Processor, AutoModelForImageTextToText, BitsAndBytesConfig

# --- CONFIGURATION ---
MODEL_ID = "google/gemma-4-E4B-it"
IMAGE_PATH = r"C:\ASL_Project\ted.jpg"

# --- HARDWARE SETUP ---
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    llm_int8_skip_modules=["vision_tower", "vision_model", "image_encoder"]
)
# --- MODEL LOAD ---
print("Loading processor...")
processor = Gemma4Processor.from_pretrained(MODEL_ID)

print("Loading model...")

model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID,
    quantization_config=quant_config,
    device_map="cuda:0",
    dtype=torch.bfloat16
)
# --- TEST ---
img = Image.open(IMAGE_PATH).convert("RGB")
print(f"Image size: {img.size} | Mode: {img.mode}")

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image"
            },
            {
                "type": "text",
                "text": "Describe what you see in this image in detail."
            }
        ]
    }
]

prompt = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False
)

inputs = processor(text=prompt, images=img, return_tensors="pt")

inputs = {
    k: (v.to("cuda:0", dtype=torch.bfloat16) if k == "pixel_values" else v.to("cuda:0"))
    for k, v in inputs.items()
    if isinstance(v, torch.Tensor)
}

input_len = inputs["input_ids"].shape[-1]

with torch.inference_mode():
    output = model.generate(
        **inputs,
        max_new_tokens=500,
        temperature=1.0,
        top_p=0.95,
        top_k=64,
        do_sample=True
    )

response = processor.decode(output[0][input_len:], skip_special_tokens=True)
print("\n--- MODEL OUTPUT ---")
print(response)
