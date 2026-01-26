import os
import pandas as pd
import time

target_dir = r"F:\Git\NLP_TEXT_CLASSIFICATION\model\API\result\qwen3-235b-a22b-instruct-2507"
target_csv = "wos_unified_level2_sample500_zero_qwen3-235b-a22b-instruct-2507.csv"
target_md = "wos_unified_level2_sample500_zero_qwen3-235b-a22b-instruct-2507.md"

csv_path = os.path.join(target_dir, target_csv)
md_path = os.path.join(target_dir, target_md)

print(f"Checking updates for: {target_csv}\n")

if os.path.exists(csv_path):
    mtime = os.path.getmtime(csv_path)
    print(f"CSV Last Modified: {time.ctime(mtime)}")
    
    df = pd.read_csv(csv_path, on_bad_lines='skip')
    total = len(df)
    
    # Check for remaining failures
    unknowns = df[df['predicted_code'].astype(str).str.upper() == 'UNKNOWN']
    print(f"Total Rows: {total}")
    print(f"Remaining Unknowns/Failures: {len(unknowns)}")
    
    if len(unknowns) > 0:
        print("Sample failing rows:")
        print(unknowns[['index', 'response']].head(3))
else:
    print("CSV File not found!")

if os.path.exists(md_path):
    mtime_md = os.path.getmtime(md_path)
    print(f"\nMD Last Modified: {time.ctime(mtime_md)}")
    with open(md_path, 'r', encoding='utf-8') as f:
        content = f.read()[:500] # Read header
        print("MD Header Content:")
        print(content)
else:
    print("MD File not found!")
