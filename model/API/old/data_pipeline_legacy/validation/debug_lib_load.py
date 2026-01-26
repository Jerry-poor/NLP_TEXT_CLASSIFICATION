
import pandas as pd
import os

csv_path = r"f:\Git\NLP_TEXT_CLASSIFICATION\model\API\datasets\lib_unified.csv"
df = pd.read_csv(csv_path)

print("Columns:", df.columns.tolist())
print("\nFirst 5 rows:")
print(df.head(5))

print("\nData Types:")
print(df.dtypes)

print("\nDDC L1 unique samples (first 10):")
print(df["ddc_l1"].unique()[:10])

print("\nCheck for NaNs in ddc_l1:")
print(df["ddc_l1"].isna().sum())

# Simulate the extraction logic
print("\nSimulating extraction for the first row:")
row = df.iloc[0]
val = row["ddc_l1"]
print(f"Raw value: {val}, Type: {type(val)}")
zfilled = str(val).strip().zfill(3)
print(f"Zfilled: '{zfilled}'")
