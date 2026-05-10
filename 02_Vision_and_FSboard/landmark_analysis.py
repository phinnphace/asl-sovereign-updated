import os
import numpy as np
import pandas as pd
import cv2
import mediapipe as mp
from itertools import combinations

# --- CONFIGURATION ---
DATA_PATH = r"C:\ASL_Project\raw_data\train"
SAMPLE_PER_LETTER = 30
OUTPUT_FILE = "landmark_distances_roboflow.csv"

# --- MEDIAPIPE SETUP ---
hands = mp.solutions.hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.1
)

# --- COLLECT LANDMARKS ---
letter_landmarks = {}

folders = sorted([f for f in os.listdir(DATA_PATH)
                  if os.path.isdir(os.path.join(DATA_PATH, f))])

print(f"Processing {len(folders)} letters...\n")

for letter in folders:
    folder_path = os.path.join(DATA_PATH, letter)
    images = [f for f in os.listdir(folder_path)
              if f.lower().endswith(('.jpg', '.jpeg', '.png'))][:SAMPLE_PER_LETTER]

    vectors = []
    detected = 0

    for img_name in images:
        img = cv2.imread(os.path.join(folder_path, img_name))
        if img is None:
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(img_rgb)

        if result.multi_hand_landmarks:
            lm = result.multi_hand_landmarks[0].landmark
            vec = np.array([[l.x, l.y, l.z] for l in lm]).flatten()
            vectors.append(vec)
            detected += 1

    if vectors:
        letter_landmarks[letter] = np.array(vectors)
        print(f"[{letter}] Detected hands in {detected}/{len(images)} images")
    else:
        print(f"[{letter}] No hands detected -- skipping")

hands.close()

# --- MEAN LANDMARK VECTOR PER LETTER ---
print("\nComputing mean landmark vectors...")
mean_landmarks = {}
for letter, vecs in letter_landmarks.items():
    mean_landmarks[letter] = vecs.mean(axis=0)

letters = sorted(mean_landmarks.keys())

# --- PAIRWISE DISTANCES ---
print("Computing pairwise distances...")
distances = []
for l1, l2 in combinations(letters, 2):
    v1 = mean_landmarks[l1]
    v2 = mean_landmarks[l2]
    dist = np.linalg.norm(v1 - v2)
    distances.append({'letter_1': l1, 'letter_2': l2, 'distance': dist})

dist_df = pd.DataFrame(distances).sort_values('distance')
dist_df.to_csv(OUTPUT_FILE, index=False)

print(f"\nDone. Distances saved to {OUTPUT_FILE}")
print("\n--- 20 MOST SIMILAR LETTER PAIRS ---")
print(dist_df.head(20).to_string(index=False))

print("\n--- 10 MOST DISTINCT LETTER PAIRS ---")
print(dist_df.tail(10).to_string(index=False))

# --- CONFUSION MATRIX CROSS REFERENCE ---
confusion_pairs = [
    ('K', 'V'), ('D', 'I'), ('T', 'I'), ('X', 'I'),
    ('W', 'V'), ('Y', 'V'), ('U', 'V'), ('R', 'V'),
    ('L', 'I'), ('P', 'I'), ('G', 'I')
]

print("\n--- CONFUSION PAIRS vs LANDMARK DISTANCE ---")
print(f"{'Pair':<10} {'Distance':>12} {'Rank':>10}")
print("-" * 35)
for l1, l2 in confusion_pairs:
    l1, l2 = min(l1, l2), max(l1, l2)
    row = dist_df[(dist_df['letter_1'] == l1) & (dist_df['letter_2'] == l2)]
    if not row.empty:
        d = row['distance'].values[0]
        rank = dist_df[dist_df['distance'] <= d].shape[0]
        print(f"{l1}-{l2:<7} {d:>12.4f} {rank:>10}")
    else:
        print(f"{l1}-{l2:<7} {'not found':>12}")
