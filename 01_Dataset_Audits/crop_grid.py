from PIL import Image
import os

# --- CONFIGURATION ---
INPUT_IMAGE = r"C:\ASL_Project\amer_sign2 .png"  # Update this to your actual filename
OUTPUT_DIR = r"C:\ASL_Project\curated_standard"

# Grid layout
COLS = 4
ROWS = 6

# ASL letters A-Y (no Z), left to right, top to bottom
# Adjust this order if your grid is arranged differently
LETTERS = [
    'A', 'B', 'C', 'D',
    'E', 'F', 'G', 'H',
    'I', 'J', 'K', 'L',
    'M', 'N', 'O', 'P',
    'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X',
    'Y'
]

# --- SETUP ---
os.makedirs(OUTPUT_DIR, exist_ok=True)
img = Image.open(INPUT_IMAGE).convert("RGB")
width, height = img.size
print(f"Image size: {width}x{height}")

cell_w = width // COLS
cell_h = height // ROWS
print(f"Cell size: {cell_w}x{cell_h}")

# --- CROP LOOP ---
letter_idx = 0
for row in range(ROWS):
    for col in range(COLS):
        if letter_idx >= len(LETTERS):
            break

        left = col * cell_w
        top = row * cell_h
        right = left + cell_w
        bottom = top + cell_h

        cell = img.crop((left, top, right, bottom))
        letter = LETTERS[letter_idx]
        out_path = os.path.join(OUTPUT_DIR, f"{letter}.png")
        cell.save(out_path)
        print(f"Saved: {letter} → {out_path}")
        letter_idx += 1

print(f"\nDone. {letter_idx} letters cropped to {OUTPUT_DIR}")
print("Open a few in Windows Photos to verify cell alignment before running the audit.")
