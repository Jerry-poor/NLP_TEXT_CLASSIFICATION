
import pandas as pd
import numpy as np

input_path = r"f:\Git\NLP_TEXT_CLASSIFICATION\model\API\datasets\lib_unified.csv"
output_path = r"f:\Git\NLP_TEXT_CLASSIFICATION\model\API\datasets\lib_unified_clean.csv"

def clean_and_infer_hierarchy():
    print("Loading Lib dataset...")
    df = pd.read_csv(input_path, dtype={"ddc_l1": str, "ddc_l2": str, "ddc_l3": str})
    
    initial_len = len(df)
    print(f"Original rows: {initial_len}")
    
    # 1. Keep ONLY rows where L3 is present (not null)
    df_clean = df[df["ddc_l3"].notna()].copy()
    print(f"Rows after dropping missing L3: {len(df_clean)} (Removed {initial_len - len(df_clean)})")
    
    # helper to zfill
    def to_code(val):
        try:
            return str(int(float(val))).zfill(3)
        except:
            return str(val).zfill(3)

    # 2. Infer L2 and L1 from L3
    # Logic: L3='531' -> L2='530', L1='500'
    print("Inferring L1 and L2 from L3...")
    
    l3_codes = df_clean["ddc_l3"].apply(to_code)
    
    # Infer L2: Take first 2 digits + '0'
    df_clean["ddc_l2"] = l3_codes.str[:2] + "0"
    
    # Infer L1: Take first 1 digit + '00'
    df_clean["ddc_l1"] = l3_codes.str[:1] + "00"
    
    # Update L3 to standard zfilled format just in case
    df_clean["ddc_l3"] = l3_codes

    # Save
    print(f"Saving cleaned dataset to {output_path}...")
    df_clean.to_csv(output_path, index=False)
    print("Done.")
    
    # Verification
    print("\nVerification (First 5 rows):")
    print(df_clean[["ddc_l1", "ddc_l2", "ddc_l3"]].head())

if __name__ == "__main__":
    clean_and_infer_hierarchy()
