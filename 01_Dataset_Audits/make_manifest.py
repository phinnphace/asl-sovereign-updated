import os
import json

# The "Explainable AI" Map for your professor
asl_logic = {
    "A": "Closed fist, thumb positioned vertically against the lateral side of the index finger.",
    "B": "Flat palm, fingers extended and touching, thumb folded across the palm.",
    "C": "Fingers and thumb curved to form a semi-circle or 'C' shape.",
    "D": "Index finger extended upward, thumb and other fingers touching to form a loop.",
    "E": "Fingers folded tightly toward the palm, thumb tucked horizontally across the knuckles.",
    "F": "Index finger and thumb touching to form a circle, other three fingers extended upward.",
    "G": "Index finger and thumb extended parallel, pointing horizontally with a slight gap.",
    "H": "Index and middle fingers extended horizontally and together, thumb tucked.",
    "I": "Pinky finger extended upward, all other fingers and thumb folded into the palm.",
    "J": "Pinky finger extended and curved (final position of the 'J' hook motion).",
    "K": "Index and middle fingers extended in a 'V' shape, thumb resting between them.",
    "L": "Index finger pointing up, thumb extended horizontally to form an 'L' shape.",
    "M": "Three fingers folded over the thumb, which is tucked between the ring and pinky.",
    "N": "Two fingers folded over the thumb, which is tucked between the middle and ring.",
    "O": "All fingers and thumb curved and touching to form a circle or 'O' shape.",
    "P": "Similar to 'K' but pointing downward; index and middle extended, thumb between.",
    "Q": "Similar to 'G' but pointing downward; thumb and index finger in a pinching shape.",
    "R": "Index and middle fingers extended and crossed over each other.",
    "S": "Closed fist with the thumb folded across the front of the fingers.",
    "T": "Fist with the thumb tucked between the index and middle fingers.",
    "U": "Index and middle fingers extended upward and touching.",
    "V": "Index and middle fingers extended upward in a 'V' shape.",
    "W": "Index, middle, and ring fingers extended upward and spread apart.",
    "X": "Index finger bent into a hook shape, other fingers folded, thumb tucked.",
    "Y": "Thumb and pinky finger extended, middle three fingers folded into the palm."
}

def build_manifest(image_root, output_file):
    manifest = []
    # Ensure we use the 'train' folder we found in the audit
    for letter in sorted(asl_logic.keys()):
        folder_path = os.path.join(image_root, letter)
        if not os.path.isdir(folder_path):
            continue
            
        reasoning = asl_logic[letter]
        
        for img_name in os.listdir(folder_path):
            # We use a relative path that we will mirror on the OSU Supercomputer
            manifest.append({
                "image": f"train/{letter}/{img_name}",
                "prefix": "classify asl letter and explain",
                "suffix": f"{letter}. Reasoning: {reasoning}"
            })
            
    with open(output_file, 'w') as f:
        for entry in manifest:
            f.write(json.dumps(entry) + "\n")
    
    print(f"Success! {len(manifest)} entries written to {output_file}")

if __name__ == "__main__":
    build_manifest(r"C:\ASL_Project\raw_data\train", r"C:\ASL_Project\dataset.jsonl")