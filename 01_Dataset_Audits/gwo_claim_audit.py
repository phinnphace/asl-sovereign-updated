"""
gwo_claim_audit.py
------------------
Audits the ASL MNIST accuracy claims from:
"A hybrid CNN-transformer framework optimized by Grey Wolf Algorithm
for accurate sign language recognition" (Hashi et al., 2025)

Their claim: 98.07% accuracy on ASL MNIST (34,627 images, 26 classes,
28x28 px, J and Z excluded due to movement).

We test the same dataset with our Gemma 4 / Ollama pipeline
(simple prompt, temp=0.1) — the same setup that produced our
best results on Roboflow (19.6%) and ISL (19.2%).

This is not a fair fight. That's the point.
A model claiming 98.07% on this benchmark should be benchmarked
against what a zero-shot VLM actually sees.

Usage:
    conda activate asl_gemma4
    python gwo_claim_audit.py

Output:
    gwo_audit_results.csv   — per-image predictions
    gwo_audit_summary.txt   — accuracy breakdown + paper comparison
"""

import requests
import base64
import csv
import os
import json
import time
import numpy as np
from PIL import Image
import io
from datetime import datetime

# ── Config ────────────────────────────────────────────────────────────────────

DATA_PATH = r"C:\ASL_Project\sign_mnist_test.csv"
OUTPUT_CSV = r"C:\ASL_Project\gwo_audit_results.csv"
OUTPUT_SUMMARY = r"C:\ASL_Project\gwo_audit_summary.txt"

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "gemma3:12b"  # same model as baseline_audit.py
TEMPERATURE = 0.1

# Their claimed performance
PAPER_ACCURACY = 98.07
PAPER_MODEL = "GWO-CTransNet (Hashi et al., 2025)"

# sign_mnist_test.csv uses numeric labels 1-24
# mapping: 1=A, 2=B, ... skipping 9(J) and 25(Z) as they require motion
# label 9 maps to I, label 10 maps to K, etc. — standard Kaggle ASL MNIST encoding
LABEL_TO_LETTER = {i: chr(64 + i) for i in range(1, 26)}  # 1->A, 2->B ... 25->Y

# How many images to test — set to None for full dataset
# Paper used 15% of 34,627 = ~5,194 for test set
# We'll use the same approximate count for a fair comparison
SAMPLE_SIZE = 500  # start with 500 for a quick sanity check; set None for full run

RANDOM_SEED = 42

# ── Prompt (same simple prompt that won our A/B test) ─────────────────────────

PROMPT_TEMPLATE = """This image shows a hand forming an American Sign Language (ASL) letter.
Which letter A-Z is being signed? Reply with a single uppercase letter only."""

# ── Image reconstruction from CSV ─────────────────────────────────────────────

def csv_row_to_image_b64(pixel_values: list) -> str:
    """
    ASL MNIST rows are 784 pixel values (28x28 grayscale, 0-255).
    Reconstruct as PNG and return base64.
    """
    arr = np.array([int(p) for p in pixel_values], dtype=np.uint8).reshape(28, 28)
    # Upscale to 112x112 — 28x28 is genuinely tiny for a VLM
    img = Image.fromarray(arr, mode='L').resize((112, 112), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode('utf-8')


def query_ollama(image_b64: str) -> str:
    """Send image to Ollama, return raw response text."""
    payload = {
        "model": MODEL,
        "prompt": PROMPT_TEMPLATE,
        "images": [image_b64],
        "stream": False,
        "options": {"temperature": TEMPERATURE}
    }
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=30)
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except Exception as e:
        return f"ERROR: {e}"


def parse_letter(raw: str) -> str:
    """Extract single letter from model response."""
    raw = raw.strip().upper()
    if not raw:
        return "?"
    # Accept first uppercase letter found
    for ch in raw:
        if ch.isalpha():
            return ch
    return "?"


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"GWO-CTransNet Claim Audit — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"Paper claim: {PAPER_ACCURACY}% on ASL MNIST")
    print(f"Our pipeline: Gemma 4 via Ollama, simple prompt, temp={TEMPERATURE}")
    print("-" * 60)

    # Load CSV
    print(f"Loading data from {DATA_PATH}...")
    rows = []
    with open(DATA_PATH, newline='', encoding='utf-8', errors='replace') as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            rows.append(row)

    print(f"Loaded {len(rows)} rows. Header: {header[:3]}...")

    # Identify label column and pixel columns
    # ASL MNIST format: label, pixel1, pixel2, ..., pixel784
    label_col = 0
    pixel_start = 1

    # Sample
    if SAMPLE_SIZE and SAMPLE_SIZE < len(rows):
        rng = np.random.default_rng(RANDOM_SEED)
        indices = rng.choice(len(rows), size=SAMPLE_SIZE, replace=False)
        rows = [rows[i] for i in sorted(indices)]
        print(f"Sampled {SAMPLE_SIZE} rows (seed={RANDOM_SEED})")
    else:
        print(f"Running full dataset ({len(rows)} rows)")

    # Run audit
    results = []
    correct = 0
    errors = 0

    for i, row in enumerate(rows):
        true_label_num = int(row[label_col])
        true_letter = LABEL_TO_LETTER.get(true_label_num, "?")
        pixels = row[pixel_start:]

        if len(pixels) != 784:
            print(f"  Row {i}: unexpected pixel count {len(pixels)}, skipping")
            errors += 1
            continue

        image_b64 = csv_row_to_image_b64(pixels)
        raw_response = query_ollama(image_b64)
        predicted_letter = parse_letter(raw_response)
        is_correct = (predicted_letter == true_letter)

        if is_correct:
            correct += 1

        results.append({
            "index": i,
            "true_label_num": true_label_num,
            "true_letter": true_letter,
            "predicted": predicted_letter,
            "correct": is_correct,
            "raw_response": raw_response
        })

        # Progress
        if (i + 1) % 50 == 0:
            running_acc = correct / (i + 1 - errors) * 100
            print(f"  [{i+1}/{len(rows)}] Running accuracy: {running_acc:.1f}%")

    # ── Results ───────────────────────────────────────────────────────────────

    total_valid = len(results) - errors
    accuracy = correct / total_valid * 100 if total_valid > 0 else 0

    # Per-letter breakdown
    letter_stats = {}
    for r in results:
        letter = r["true_letter"]
        if letter not in letter_stats:
            letter_stats[letter] = {"correct": 0, "total": 0}
        letter_stats[letter]["total"] += 1
        if r["correct"]:
            letter_stats[letter]["correct"] += 1

    # Save CSV
    with open(OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults saved to {OUTPUT_CSV}")

    # Save summary
    summary_lines = [
        "=" * 60,
        "GWO-CTransNet Claim Audit — ASL MNIST",
        f"Run date: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "=" * 60,
        "",
        "PAPER CLAIM:",
        f"  Model: {PAPER_MODEL}",
        f"  Accuracy: {PAPER_ACCURACY}%",
        f"  Dataset: ASL MNIST, 34,627 images, 26 classes",
        f"  Excludes J and Z (movement gestures)",
        "",
        "OUR RESULT:",
        f"  Model: Gemma 4 via Ollama (simple prompt, temp={TEMPERATURE})",
        f"  Sample size: {total_valid} images",
        f"  Correct: {correct}",
        f"  Accuracy: {accuracy:.2f}%",
        f"  Gap vs. paper claim: {PAPER_ACCURACY - accuracy:.2f} percentage points",
        "",
        "PER-LETTER BREAKDOWN:",
    ]

    for letter in sorted(letter_stats.keys()):
        s = letter_stats[letter]
        pct = s["correct"] / s["total"] * 100 if s["total"] > 0 else 0
        summary_lines.append(f"  {letter}: {s['correct']}/{s['total']} ({pct:.1f}%)")

    summary_lines += [
        "",
        "NOTE:",
        "  This is a zero-shot VLM vs. a fully trained+tuned classifier.",
        "  The gap is expected. What matters is whether their numbers",
        "  are internally consistent with what 28x28 grayscale ASL images",
        "  actually look like to any model — trained or not.",
        "  See: lessons__1_.md — 'ISL/Roboflow convergence is the headline",
        "  finding' — performance is determined by geometric distinctiveness",
        "  of the handshape, not model sophistication alone.",
    ]

    with open(OUTPUT_SUMMARY, 'w') as f:
        f.write('\n'.join(summary_lines))

    # Print summary
    print('\n'.join(summary_lines))


if __name__ == "__main__":
    main()
