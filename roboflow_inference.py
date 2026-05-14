"""
Roboflow ASL Inference Rerun
Gemma 4 via Ollama — simple prompt, temp=0.1
Saves predictions row-by-row to CSV — safe to interrupt and resume.

Run in asl_gemma4 conda environment:
    conda activate asl_gemma4
    python roboflow_inference.py
"""

import os
import base64
import csv
import json
import random
import requests
from pathlib import Path
from PIL import Image
import io
import time

# ── CONFIGURATION ─────────────────────────────────────────────────────────────

IMAGE_DIR   = r"C:\ASL_Project\raw_data\train"
OUTPUT_CSV  = r"C:\ASL_Project\roboflow_predictions.csv"
OLLAMA_URL  = "http://localhost:11434/api/chat"
MODEL       = "gemma4:e4b"
TEMPERATURE = 0.1
SAMPLES_PER_LETTER = 40
RANDOM_SEED = 42

SKIP_LETTERS = {'J', 'Z'}  # motion signs

PROMPT = "What letter of the ASL alphabet does this hand show? Answer with just the letter."

# ── HELPERS ───────────────────────────────────────────────────────────────────

def encode_image(image_path):
    """Load image and encode as base64 string."""
    with Image.open(image_path) as img:
        if img.mode != 'RGB':
            img = img.convert('RGB')
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=95)
        return base64.b64encode(buffer.getvalue()).decode('utf-8')


def query_ollama(image_b64):
    """POST image to Ollama, return raw model output string."""
    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": PROMPT,
                "images": [image_b64]
            }
        ],
        "options": {
            "temperature": TEMPERATURE
        },
        "stream": False
    }
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        return data["message"]["content"].strip()
    except requests.exceptions.Timeout:
        return "TIMEOUT"
    except Exception as e:
        return f"ERROR: {str(e)[:80]}"


def already_processed(output_csv):
    """Return set of (actual_letter, image_file) already in output CSV."""
    done = set()
    if not os.path.exists(output_csv):
        return done
    with open(output_csv, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            done.add((row['actual_letter'], row['image_file']))
    return done


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    random.seed(RANDOM_SEED)
    image_root = Path(IMAGE_DIR)

    # Get letter folders, sorted, skip J and Z
    letter_dirs = sorted([
        d for d in image_root.iterdir()
        if d.is_dir() and d.name.upper() not in SKIP_LETTERS
        and d.name.upper().isalpha() and len(d.name) == 1
    ])

    print(f"Found {len(letter_dirs)} letter folders: "
          f"{[d.name for d in letter_dirs]}")
    print(f"Output CSV: {OUTPUT_CSV}")
    print(f"Samples per letter: {SAMPLES_PER_LETTER}")
    print(f"Model: {MODEL} | Temp: {TEMPERATURE}")
    print()

    # Check resume state
    done = already_processed(OUTPUT_CSV)
    if done:
        print(f"Resuming — {len(done)} predictions already in CSV, skipping.")
    else:
        print("Fresh run — writing header.")

    # Open CSV in append mode
    write_header = not os.path.exists(OUTPUT_CSV) or os.path.getsize(OUTPUT_CSV) == 0
    csv_file = open(OUTPUT_CSV, 'a', newline='', encoding='utf-8')
    writer = csv.DictWriter(csv_file,
                            fieldnames=['actual_letter', 'image_file', 'model_output'])
    if write_header:
        writer.writeheader()
        csv_file.flush()

    total_done = len(done)
    grand_total = len(letter_dirs) * SAMPLES_PER_LETTER

    for letter_dir in letter_dirs:
        letter = letter_dir.name.upper()

        # Collect image files
        images = sorted([
            f for f in letter_dir.iterdir()
            if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        ])

        if not images:
            print(f"  [{letter}] No images found, skipping.")
            continue

        # Sample up to SAMPLES_PER_LETTER
        if len(images) > SAMPLES_PER_LETTER:
            images = random.sample(images, SAMPLES_PER_LETTER)
        images = sorted(images)  # consistent order after sampling

        letter_done = 0
        letter_skipped = 0

        for img_path in images:
            img_name = img_path.name

            # Resume check
            if (letter, img_name) in done:
                letter_skipped += 1
                continue

            # Encode and query
            try:
                img_b64 = encode_image(img_path)
            except Exception as e:
                print(f"    [{letter}] Failed to load {img_name}: {e}")
                continue

            output = query_ollama(img_b64)

            # Write immediately
            writer.writerow({
                'actual_letter': letter,
                'image_file':    img_name,
                'model_output':  output
            })
            csv_file.flush()

            total_done += 1
            letter_done += 1

            # Progress
            pred_display = output if len(output) <= 30 else output[:27] + '...'
            print(f"  [{letter}] {letter_done}/{len(images)-letter_skipped} "
                  f"| {img_name} → '{pred_display}' "
                  f"| total {total_done}/{grand_total}")

        print(f"  [{letter}] complete — {letter_done} new, "
              f"{letter_skipped} skipped (already done)\n")

    csv_file.close()
    print("="*60)
    print(f"Run complete. {total_done} predictions written to:")
    print(f"  {OUTPUT_CSV}")

    # Quick summary
    import pandas as pd
    df = pd.read_csv(OUTPUT_CSV)
    df['actual_letter'] = df['actual_letter'].str.strip()
    df['model_output'] = df['model_output'].str.strip()
    LABELS = set('ABCDEFGHIKLMNOPQRSTUVWXY')
    df['valid'] = df['model_output'].isin(LABELS)
    print(f"\nTotal rows: {len(df)}")
    print(f"Valid predictions: {df['valid'].sum()} "
          f"({100*df['valid'].mean():.1f}%)")
    print(f"Refusals/errors: {(~df['valid']).sum()}")
    print(f"\nPer-letter valid count:")
    print(df[df['valid']]['actual_letter'].value_counts().sort_index().to_string())


if __name__ == '__main__':
    main()
