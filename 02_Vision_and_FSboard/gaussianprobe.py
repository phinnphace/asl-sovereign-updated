"""
Gaussian Noise Probe — ASL Confusion Matrix Decoder Ring
Applies Gaussian noise at incremental sigma levels to images,
runs Gemma 4 inference via Ollama, saves predictions row-by-row.

Run in asl_gemma4 conda environment:
    conda activate asl_gemma4
    python gaussian_noise_probe.py

Safe to interrupt and resume — appends row by row.
"""

import os
import base64
import csv
import random
import requests
import time
import numpy as np
from pathlib import Path
from PIL import Image
import io

# ── CONFIGURATION ─────────────────────────────────────────────────────────────

OLLAMA_URL  = "http://localhost:11434/api/chat"
MODEL       = "gemma4:e4b"
TEMPERATURE = 0.1
TIMEOUT_SEC = 30
SAMPLES_PER_LETTER = 40
RANDOM_SEED = 42

SIGMA_LEVELS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

SKIP_LETTERS = {'J', 'Z'}
LABELS = set('ABCDEFGHIKLMNOPQRSTUVWXY')

PROMPT = "What letter of the ASL alphabet does this hand show? Answer with just the letter."

DATASETS = [
    ('ISL',      r"C:\ASL_Project\ISL custom Data"),
    ('Roboflow', r"C:\ASL_Project\raw_data\train"),
]

OUTPUT_CSV = r"C:\ASL_Project\gaussian_noise_results.csv"

# ── IMAGE HELPERS ─────────────────────────────────────────────────────────────

def add_gaussian_noise(image_path, sigma):
    with Image.open(image_path) as img:
        if img.mode != 'RGB':
            img = img.convert('RGB')
        arr = np.array(img, dtype=np.float32) / 255.0
        if sigma > 0:
            noise = np.random.normal(0, sigma, arr.shape).astype(np.float32)
            arr = np.clip(arr + noise, 0.0, 1.0)
        noisy_img = Image.fromarray((arr * 255).astype(np.uint8))
        buffer = io.BytesIO()
        noisy_img.save(buffer, format='JPEG', quality=95)
        return base64.b64encode(buffer.getvalue()).decode('utf-8')


def query_ollama(image_b64):
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": PROMPT, "images": [image_b64]}],
        "options": {"temperature": TEMPERATURE},
        "stream": False
    }
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=TIMEOUT_SEC)
        resp.raise_for_status()
        return resp.json()["message"]["content"].strip()
    except requests.exceptions.Timeout:
        return "TIMEOUT"
    except Exception as e:
        return f"ERROR: {str(e)[:80]}"


def already_processed(output_csv):
    done = set()
    if not os.path.exists(output_csv):
        return done
    with open(output_csv, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            done.add((row['dataset'], float(row['sigma']),
                      row['actual_letter'], row['image_file']))
    return done


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    done = already_processed(OUTPUT_CSV)
    if done:
        print(f"Resuming — {len(done)} predictions already logged.")

    write_header = (not os.path.exists(OUTPUT_CSV) or
                    os.path.getsize(OUTPUT_CSV) == 0)
    csv_file = open(OUTPUT_CSV, 'a', newline='', encoding='utf-8')
    writer = csv.DictWriter(csv_file, fieldnames=[
        'dataset', 'sigma', 'actual_letter', 'image_file', 'model_output'])
    if write_header:
        writer.writeheader()
        csv_file.flush()

    # Count total calls for progress
    total_calls = 0
    dataset_images = {}
    for ds_name, image_dir in DATASETS:
        image_root = Path(image_dir)
        letter_dirs = sorted([
            d for d in image_root.iterdir()
            if d.is_dir()
            and d.name.upper() not in SKIP_LETTERS
            and d.name.upper().isalpha()
            and len(d.name) == 1
        ])
        letter_images = {}
        for letter_dir in letter_dirs:
            letter = letter_dir.name.upper()
            images = sorted([
                f for f in letter_dir.iterdir()
                if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
            ])
            if not images:
                continue
            if len(images) > SAMPLES_PER_LETTER:
                images = random.sample(images, SAMPLES_PER_LETTER)
            letter_images[letter] = sorted(images)
        dataset_images[ds_name] = letter_images
        total_calls += len(SIGMA_LEVELS) * sum(len(v) for v in letter_images.values())

    print(f"Total inference calls planned: {total_calls}")
    print(f"Output: {OUTPUT_CSV}")
    print()

    completed = len(done)
    t_start = time.time()

    for ds_name, _ in DATASETS:
        letter_images = dataset_images[ds_name]
        print(f"\n{'='*60}")
        print(f"Dataset: {ds_name}")
        print(f"{'='*60}")

        for sigma in SIGMA_LEVELS:
            print(f"\n  Sigma = {sigma}")

            for letter, images in sorted(letter_images.items()):
                for img_path in images:
                    img_name = img_path.name

                    if (ds_name, sigma, letter, img_name) in done:
                        continue

                    try:
                        img_b64 = add_gaussian_noise(img_path, sigma)
                    except Exception as e:
                        print(f"    [{letter}] Failed to load {img_name}: {e}")
                        continue

                    output = query_ollama(img_b64)

                    writer.writerow({
                        'dataset':       ds_name,
                        'sigma':         sigma,
                        'actual_letter': letter,
                        'image_file':    img_name,
                        'model_output':  output
                    })
                    csv_file.flush()

                    completed += 1
                    elapsed = time.time() - t_start
                    rate = completed / elapsed if elapsed > 0 else 0
                    remaining = (total_calls - completed) / rate if rate > 0 else 0

                    pred = output if len(output) <= 20 else output[:17] + '...'
                    print(f"    [{ds_name}][{letter}] σ={sigma} | "
                          f"{img_name} → '{pred}' | "
                          f"{completed}/{total_calls} | "
                          f"~{remaining/60:.0f}m left")

    csv_file.close()
    elapsed_total = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"Complete in {elapsed_total/60:.1f} minutes")
    print(f"Output: {OUTPUT_CSV}")

    # Quick summary
    import pandas as pd
    df = pd.read_csv(OUTPUT_CSV)
    df['valid'] = df['model_output'].isin(LABELS)
    print(f"\nTotal rows: {len(df)}")
    print(f"\nValid predictions per sigma:")
    print(df.groupby('sigma')['valid'].agg(['sum','count','mean']).round(3))


if __name__ == '__main__':
    main()