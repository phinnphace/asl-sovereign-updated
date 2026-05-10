from transformers import AutoProcessor
from PIL import Image
import torch

print("Loading processor natively...")
processor = AutoProcessor.from_pretrained("google/gemma-4-E4B-it")

# Force the processor to compile a raw multimodal call
try:
    dummy_img = Image.new('RGB', (224, 224), color='red')
    
    messages = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": "Diagnostic test."}
        ]}
    ]
    
    # See what apply_chat_template actually outputs under the hood
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    print("\n--- CHAT TEMPLATE OUTPUT ---")
    print(repr(prompt))
    
    # See how the processor natively maps the input IDs
    inputs = processor(text=prompt, images=dummy_img, return_tensors="pt")
    decoded_prompt = processor.decode(inputs["input_ids"][0])
    print("\n--- NATIVE PROCESSOR INJECTION ---")
    print(repr(decoded_prompt))
    
except Exception as e:
    print(f"Diagnostic failed during processing: {e}")