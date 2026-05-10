from PIL import Image
from pathlib import Path
import random

# Check a sample of Roboflow images
root = Path(r"C:\ASL_Project\raw_data\train")
sizes = set()
for letter_dir in list(root.iterdir())[:5]:
    for img_path in list(letter_dir.iterdir())[:3]:
        if img_path.suffix.lower() in {'.jpg','.jpeg','.png'}:
            with Image.open(img_path) as img:
                sizes.add(img.size)
print("Roboflow sizes:", sizes)

# Check ISL
root2 = Path(r"C:\ASL_Project\ISL custom Data")
sizes2 = set()
for letter_dir in list(root2.iterdir())[:5]:
    for img_path in list(letter_dir.iterdir())[:3]:
        if img_path.suffix.lower() in {'.jpg','.jpeg','.png'}:
            with Image.open(img_path) as img:
                sizes2.add(img.size)
print("ISL sizes:", sizes2) 
