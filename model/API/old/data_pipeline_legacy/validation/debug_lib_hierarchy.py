
import pandas as pd

# Load dataset
input_csv = r"f:\Git\NLP_TEXT_CLASSIFICATION\model\API\datasets\lib_unified.csv"
df = pd.read_csv(input_csv)

print(f"Total rows: {len(df)}")

# Check non-null counts
print("\nNon-null counts:")
print(f"ddc_l1: {df['ddc_l1'].notna().sum()}")
print(f"ddc_l2: {df['ddc_l2'].notna().sum()}")
print(f"ddc_l3: {df['ddc_l3'].notna().sum()}")

print("\n--- Overlap Analysis ---")
# Rows that have L3 but NOT L1? (Should be impossibility if hierarchy is strict)
l3_no_l1 = df[df['ddc_l3'].notna() & df['ddc_l1'].isna()]
print(f"Rows with L3 but missing L1: {len(l3_no_l1)}")

# Rows that have L2 but NOT L1?
l2_no_l1 = df[df['ddc_l2'].notna() & df['ddc_l1'].isna()]
print(f"Rows with L2 but missing L1: {len(l2_no_l1)}")

# Rows that ONLY have L1 (no L2, no L3)
only_l1 = df[df['ddc_l1'].notna() & df['ddc_l2'].isna() & df['ddc_l3'].isna()]
print(f"Rows with ONLY L1 (Leaf L1): {len(only_l1)}")
if len(only_l1) > 0:
    print("Example Only-L1 rows:")
    print(only_l1[['ddc_l1', 'title']].head())
