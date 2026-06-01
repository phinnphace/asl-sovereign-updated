"""
train_local.py — Local fine-tuning script for ASL Decoder Ring
Run from the chatbot/ directory:
    python train_local.py

Requirements (run once):
    pip install "unsloth[cu121-torch240] @ git+https://github.com/unslothai/unsloth.git"
    pip install --no-deps trl peft accelerate bitsandbytes xformers
"""

from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
import torch
import os

# ── Config ────────────────────────────────────────────────────────────────────
DATA_FILE   = "asl_training_data_v2.jsonl"
OUTPUT_DIR  = "asl_finetuned_gguf"
LORA_DIR    = "asl_provenance_lora"
MAX_STEPS   = 200
MAX_SEQ_LEN = 2048

# ── Load model ────────────────────────────────────────────────────────────────
print("Loading Gemma 2 9B base model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/gemma-2-9b-it-bnb-4bit",
    max_seq_length=MAX_SEQ_LEN,
    dtype=None,
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

# ── Load dataset ──────────────────────────────────────────────────────────────
print(f"Loading training data from {DATA_FILE}...")
dataset = load_dataset("json", data_files={"train": DATA_FILE}, split="train")

prompt_template = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

def format_prompts(examples):
    texts = []
    for instruction, input_text, output in zip(
        examples["instruction"], examples["input"], examples["output"]
    ):
        text = prompt_template.format(instruction, input_text, output) + tokenizer.eos_token
        texts.append(text)
    return {"text": texts}

dataset = dataset.map(format_prompts, batched=True)
print(f"Dataset loaded: {len(dataset)} examples")

# ── Train ─────────────────────────────────────────────────────────────────────
print("Starting training...")
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LEN,
    dataset_num_proc=2,
    packing=False,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=10,
        max_steps=MAX_STEPS,
        learning_rate=2e-4,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
    ),
)

trainer_stats = trainer.train()
print(f"Training complete! Loss: {trainer_stats.training_loss:.4f}")

# ── Save LoRA adapters ────────────────────────────────────────────────────────
print(f"Saving LoRA adapters to {LORA_DIR}...")
model.save_pretrained(LORA_DIR)
tokenizer.save_pretrained(LORA_DIR)

# ── Export GGUF ───────────────────────────────────────────────────────────────
print(f"Exporting GGUF to {OUTPUT_DIR}...")
model.save_pretrained_gguf(OUTPUT_DIR, tokenizer, quantization_method="q4_k_m")

print(f"\nDone! Upload the .gguf file from {OUTPUT_DIR}/ to Modal:")
print(f"  modal volume put asl-model-volume {OUTPUT_DIR}/unsloth.Q4_K_M.gguf gemma2_asl_finetuned-unsloth.Q4_K_M.gguf")
