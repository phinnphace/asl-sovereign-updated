import pandas as pd

df = pd.read_parquet(r"C:\ASL_Project\111123288.parquet")
hand_cols = [c for c in df.columns if 'right_hand' in c or 'left_hand' in c]
print(f"Hand columns: {len(hand_cols)}")
print(hand_cols[:10])