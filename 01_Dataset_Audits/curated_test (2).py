import torch
import os
from PIL import Image
from transformers import Gemma4Processor, AutoModelForImageTextToText, BitsAndBytesConfig

# --- CONFIGURATION ---
MODEL_ID = "google/gemma-4-E4B-it"
CURATED_DIR = r"C:\ASL_Project\curated_standard"
TEST_LETTERS = ['A', 'B', 'C']

# --- HARDWARE SETUP ---
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)

# --- MODEL LOAD ---
print("Loading processor...")
processor = Gemma4Processor.from_pretrained(MODEL_ID)

print("Loading model...")
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID,
    quantization_config=quant_config,
    device_map="cuda:0"
)
model.eval()

# --- TEST LOOP ---
for letter in TEST_LETTERS:
    img_path = os.path.join(CURATED_DIR, f"{letter}.png")

    if not os.path.exists(img_path):
        print(f"\n[{letter}] MISSING: {img_path}")
        continue

    img = Image.open(img_path).convert("RGB")
    print(f"\n[{letter}] Image size: {img.size}")

    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are an equity-centered research scientist specialized in ASL linguistics. Your task is to provide a rigorous, objective analysis of provided handshapes to establish a performance baseline. Maintain high fidelity to ASL grammatical standards."
                }
            ]
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image"
                },
                {
                    "type": "text",
                    "text": "Identify this ASL letter. Describe the finger positions in detail, including palm orientation and thumb placement, then provide the final letter."
                }
            ]
        }
    ]

    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True
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

    response = processor.decode(output[0][input_len:], skip_special_tokens=False)
    parsed = processor.parse_response(response)

    print(f"[{letter}] Model output:")
    print(parsed)
    print("-" * 60)

print("\nComparison test complete.")
