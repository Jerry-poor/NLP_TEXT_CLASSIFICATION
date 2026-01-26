
import pandas as pd
import os

datasets = {
    "AG_News": r"f:\Git\NLP_TEXT_CLASSIFICATION\model\API\datasets\ag_news_unified.csv",
    "WOS": r"f:\Git\NLP_TEXT_CLASSIFICATION\model\API\datasets\wos_unified.csv",
    "Lib": r"f:\Git\NLP_TEXT_CLASSIFICATION\model\API\datasets\lib_unified.csv"
}

def analyze_hierarchy(name, path):
    print(f"\n{'='*20} Analyzing {name} {'='*20}")
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return

    df = pd.read_csv(path)
    print(f"Total rows: {len(df)}")
    
    # Check if columns exist
    cols = [c for c in ['ddc_l1', 'ddc_l2', 'ddc_l3'] if c in df.columns]
    print(f"Available DDC columns: {cols}")
    
    # 1. Check Rootless L2 (Has L2 but missing L1)
    if 'ddc_l2' in df.columns and 'ddc_l1' in df.columns:
        l2_no_l1 = df[df['ddc_l2'].notna() & df['ddc_l1'].isna()]
        print(f"Rows with L2 but MISSING L1: {len(l2_no_l1)} ({len(l2_no_l1)/len(df):.2%})")
    
    # 2. Check Rootless L3 (Has L3 but missing L1 or L2)
    if 'ddc_l3' in df.columns:
        if 'ddc_l1' in df.columns:
            l3_no_l1 = df[df['ddc_l3'].notna() & df['ddc_l1'].isna()]
            print(f"Rows with L3 but MISSING L1: {len(l3_no_l1)} ({len(l3_no_l1)/len(df):.2%})")
        
        if 'ddc_l2' in df.columns:
            l3_no_l2 = df[df['ddc_l3'].notna() & df['ddc_l2'].isna()]
            print(f"Rows with L3 but MISSING L2: {len(l3_no_l2)} ({len(l3_no_l2)/len(df):.2%})")

for name, path in datasets.items():
    analyze_hierarchy(name, path)
