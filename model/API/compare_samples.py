import pandas as pd
import os

# Define paths
file_ds = r"F:\Git\NLP_TEXT_CLASSIFICATION\model\API\result\deepseek-chat\lib_unified_level2_sample500_zero_deepseek-chat.csv"
file_gpt = r"F:\Git\NLP_TEXT_CLASSIFICATION\model\API\result\gpt-52-2025-12-11\lib_unified_level2_sample500_zero_gpt-5.2-2025-12-11.csv"

# Function to peek at data
def analyze_file(filepath, name):
    print(f"\n--- Analyzing {name} ---")
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return None
        
    try:
        df = pd.read_csv(filepath, on_bad_lines='skip')
        print(f"Total Rows: {len(df)}")
        if 'index' in df.columns:
            print(f"Indices (First 5): {df['index'].head(5).tolist()}")
            print(f"Indices (Last 5): {df['index'].tail(5).tolist()}")
            return df
        else:
            print("Column 'index' not found.")
            print(f"Columns: {df.columns.tolist()}")
            return df
    except Exception as e:
        print(f"Error reading {name}: {e}")
        return None

df_ds = analyze_file(file_ds, "DeepSeek")
df_gpt = analyze_file(file_gpt, "GPT-5.2")

if df_ds is not None and df_gpt is not None:
    print("\n--- Comparison ---")
    
    # 1. Compare Indices if they exist
    if 'index' in df_ds.columns and 'index' in df_gpt.columns:
        set_ds = set(df_ds['index'])
        set_gpt = set(df_gpt['index'])
        
        common = set_ds.intersection(set_gpt)
        print(f"DeepSeek Unique Indices: {len(set_ds)}")
        print(f"GPT-5.2 Unique Indices: {len(set_gpt)}")
        print(f"Common Indices: {len(common)}")
        
        if len(common) == len(set_ds) and len(common) == len(set_gpt):
            print("RESULT: EXACTLY THE SAME SAMPLES used.")
        else:
            print("RESULT: DIFFERENT SAMPLES used.")
            
    # 2. Check for 'Contaminated' content keywords in DeepSeek too
    # We look for typical sports/news words in the 'title' column
    print("\n--- DeepSeek Contamination Check ---")
    suspicious_keywords = ["Arrested", "win", "Coach", "vs.", "Cup", "League", "Police"]
    
    if 'title' in df_ds.columns:
        suspicious_rows = df_ds[df_ds['title'].str.contains('|'.join(suspicious_keywords), case=False, na=False)]
        print(f"DeepSeek Suspicious Rows Count: {len(suspicious_rows)}")
        if len(suspicious_rows) > 0:
            print("Examples of suspicious titles in DeepSeek:")
            print(suspicious_rows['title'].head(3).tolist())
    
