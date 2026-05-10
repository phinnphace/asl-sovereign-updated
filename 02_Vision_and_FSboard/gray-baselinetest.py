"""
Zero-Input Gray Image Baseline Test
Sends pure gray images at three resolutions to Gemma 4 via Ollama.
Tests whether structured confusion patterns emerge from contentless input.

If K->V or other decoder ring pairs appear on gray input:
the confusion structure is a prior in the output layer, not image-driven.

Run in asl_gemma4 conda environment:
    conda activate asl_gemma4
    python gray_baseline_test.py

Safe to interrupt and resume.
"""

import os
import base64
import csv
import requests
import time
import numpy as np
from PIL import Image
import io

# ── CONFIGURATION ─────────────────────────────────────────────────────────────

OLLAMA_URL  = "http://localhost:11434/api/chat"
MODEL       = "gemma4:e4b"
TEMPERATURE = 0.1
TIMEOUT_SEC = 30
TRIALS_PER_LETTER = 20
GRAY_VALUE  = 128  # pure mid-gray

# Three sizes matching actual dataset images
GRAY_SIZES = [
    (300, 248, 'roboflow_small'),
    (500, 360, 'roboflow_large'),
    (300, 400, 'isl'),
]

# 23 classes — J and Z excluded
LABELS = ['A','B','C','D','E','F','G','H','I','K','L','M',
          'N','O','P','Q','R','S','T','U','V','W','X','Y']

PROMPT = "What letter of the ASL alphabet does this hand show? Answer with just the letter."

OUTPUT_CSV = r"C:\ASL_Project\gray_baseline_results.csv"

# ── HELPERS ───────────────────────────────────────────────────────────────────

def make_gray_image_b64(width, height, gray_value=128):
    img = Image.new('RGB', (width, height), color=(gray_value, gray_value, gray_value))
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG', quality=95)
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
            done.add((row['image_size'], row['actual_letter'], int(row['trial'])))
    return done


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    label_set = set(LABELS)

    done = already_processed(OUTPUT_CSV)
    if done:
        print(f"Resuming — {len(done)} trials already logged.")

    write_header = (not os.path.exists(OUTPUT_CSV) or
                    os.path.getsize(OUTPUT_CSV) == 0)
    csv_file = open(OUTPUT_CSV, 'a', newline='', encoding='utf-8')
    writer = csv.DictWriter(csv_file, fieldnames=[
        'image_size', 'width', 'height', 'actual_letter',
        'trial', 'model_output', 'is_valid_letter', 'is_correct'])
    if write_header:
        writer.writeheader()
        csv_file.flush()

    total_calls = len(GRAY_SIZES) * len(LABELS) * TRIALS_PER_LETTER
    completed = len(done)
    t_start = time.time()

    print(f"Zero-input gray baseline test")
    print(f"Gray sizes: {[(w,h,n) for w,h,n in GRAY_SIZES]}")
    print(f"Letters: {len(LABELS)} | Trials per letter: {TRIALS_PER_LETTER}")
    print(f"Total calls: {total_calls}")
    print(f"Output: {OUTPUT_CSV}")
    print()

    for width, height, size_name in GRAY_SIZES:
        print(f"\n{'='*60}")
        print(f"Gray image: {width}x{height} ({size_name})")
        print(f"{'='*60}")

        # Pre-encode gray image once per size
        img_b64 = make_gray_image_b64(width, height, GRAY_VALUE)

        letter_valid_counts = {}
        letter_outputs = {}

        for letter in LABELS:
            letter_valid_counts[letter] = 0
            letter_outputs[letter] = []

            for trial in range(TRIALS_PER_LETTER):
                if (size_name, letter, trial) in done:
                    continue

                output = query_ollama(img_b64)
                is_valid = output in label_set
                is_correct = output == letter  # should always be False on gray

                writer.writerow({
                    'image_size':      size_name,
                    'width':           width,
                    'height':          height,
                    'actual_letter':   letter,
                    'trial':           trial,
                    'model_output':    output,
                    'is_valid_letter': is_valid,
                    'is_correct':      is_correct
                })
                csv_file.flush()

                completed += 1
                elapsed = time.time() - t_start
                rate = completed / elapsed if elapsed > 0 else 0
                remaining = (total_calls - completed) / rate if rate > 0 else 0

                if is_valid:
                    letter_valid_counts[letter] += 1
                    letter_outputs[letter].append(output)

                status = f"→ '{output}'" if is_valid else "→ REFUSAL/TIMEOUT"
                print(f"  [{size_name}][{letter}] trial {trial:02d} {status} | "
                      f"{completed}/{total_calls} | ~{remaining/60:.0f}m left")

            # Per-letter summary
            valid_count = letter_valid_counts[letter]
            if valid_count > 0:
                from collections import Counter
                dist = Counter(letter_outputs[letter])
                print(f"  [{letter}] {valid_count}/{TRIALS_PER_LETTER} valid predictions: {dict(dist)}")
            else:
                print(f"  [{letter}] 0/{TRIALS_PER_LETTER} valid — all refusals (expected)")

        # Per-size summary
        total_valid = sum(letter_valid_counts.values())
        total_trials = len(LABELS) * TRIALS_PER_LETTER
        print(f"\n  {size_name} summary: {total_valid}/{total_trials} valid predictions "
              f"({100*total_valid/total_trials:.1f}%)")
        if total_valid > 0:
            print(f"  *** ALERT: Structured predictions on gray input detected ***")
            print(f"  This suggests output layer prior, not image-driven classification")

    csv_file.close()

    elapsed_total = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"Complete in {elapsed_total/60:.1f} minutes")
    print(f"Output: {OUTPUT_CSV}")

    # Summary analysis
    import pandas as pd
    from collections import Counter

    df = pd.read_csv(OUTPUT_CSV)
    df['is_valid_letter'] = df['is_valid_letter'].astype(str).str.upper() == 'TRUE'

    print(f"\nTotal rows: {len(df)}")
    print(f"\nValid letter predictions per image size:")
    print(df.groupby('image_size')['is_valid_letter'].agg(['sum','count','mean']).round(3))

    valid = df[df['is_valid_letter']]
    if len(valid) > 0:
        print(f"\nDistribution of predictions on gray input (all sizes):")
        print(valid['model_output'].value_counts())
        print(f"\nDecoder ring pairs detected on gray input:")
        ring_pairs = [('K','V'),('D','I'),('G','D'),('E','W'),('P','D')]
        for true, pred in ring_pairs:
            hits = valid[(valid['actual_letter']==true) & (valid['model_output']==pred)]
            if len(hits) > 0:
                print(f"  {true}→{pred}: {len(hits)} hits *** PRIOR DETECTED ***")
    else:
        print(f"\nNo valid letter predictions on gray input — confusion structure is image-driven")


if __name__ == '__main__':
    main()