from transformers import Gemma4Processor
from PIL import Image
import torch

print("Loading Gemma4Processor explicitly...")
processor = Gemma4Processor.from_pretrained("google/gemma-4-E4B-it")

dummy_img = Image.new('RGB', (224, 224), color='red')

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "Diagnostic test."}
        ]
    }
]

# Test 1: What does apply_chat_template produce?
prompt = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
print("\n--- CHAT TEMPLATE OUTPUT ---")
print(repr(prompt))

# Test 2: Count image tokens in the template output
image_token_count = prompt.count("<|image|>")
print(f"\nImage token placeholders in template: {image_token_count}")
print("(Should be exactly 1)")

# Test 3: Pass through processor with image and check input_ids shape
inputs = processor(text=prompt, images=dummy_img, return_tensors="pt")
print("\n--- INPUT SHAPES ---")
for k, v in inputs.items():
    if isinstance(v, torch.Tensor):
        print(f"  {k}: {v.shape}")

# Test 4: Decode and inspect
decoded = processor.decode(inputs["input_ids"][0])
# Count soft image tokens in decoded output
soft_count = decoded.count("<|image|>")
print(f"\nDecoded image tokens: {soft_count}")

# Test 5: Check pixel_values dtype and range
if "pixel_values" in inputs:
    pv = inputs["pixel_values"]
    print(f"\npixel_values dtype: {pv.dtype}")
    print(f"pixel_values min/max: {pv.min().item():.3f} / {pv.max().item():.3f}")
    print("(Non-zero range means image was actually processed)")
else:
    print("\nWARNING: pixel_values not in inputs — image was dropped!")