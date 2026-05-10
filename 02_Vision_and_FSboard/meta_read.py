import pandas as pd

df = pd.read_csv(r"C:\ASL_Project\supplemental_metadata.csv")
print(f"Shape: {df.shape}")
print(f"\nColumns: {list(df.columns)}")
print(f"\nFirst 5 rows:")
print(df.head())
print(f"\nUnique phrases sample: {df['phrase'].unique()[:20]}")