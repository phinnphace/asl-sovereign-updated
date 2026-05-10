import pandas as pd
import json

# Read the mapping file
with open(r"C:\ASL_Project\character_to_prediction_index.json") as f:
    char_map = json.load(f)
print("Character mapping:")
print(char_map)

# Peek at the parquet file
df = pd.read_parquet(r"C:\ASL_Project\111123288.parquet")
print(f"\nShape: {df.shape}")
print(f"\nColumns (first 20): {list(df.columns[:20])}")
print(f"\nFirst 3 rows (first 5 cols):")
print(df.iloc[:3, :5])