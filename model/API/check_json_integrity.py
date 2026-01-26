import os
import pandas as pd
import glob
import json
import ast

base_dir = r"F:\Git\NLP_TEXT_CLASSIFICATION\model\API\result"
files = glob.glob(os.path.join(base_dir, "**", "*_sample500_zero_*.csv"), recursive=True)

print(f"{'File':<60} | {'Total':<6} | {'Bad JSON':<8} | {'Empty Preds':<11} | {'Status'}")
print("-" * 110)

for file_path in files:
    try:
        df = pd.read_csv(file_path, on_bad_lines='skip')
        total = len(df)
        
        if total == 0:
            continue
            
        bad_json_count = 0
        empty_preds_count = 0
        
        # Check integrity
        for _, row in df.iterrows():
            # 1. Check Raw Response JSON
            raw_resp = str(row.get('response', ''))
            is_valid_json = False
            try:
                # Try parsing standard JSON
                parsed = json.loads(raw_resp)
                is_valid_json = True
                
                # Optional: Check structure inside if valid
                if isinstance(parsed, dict):
                    # Expecting 'categories' or 'predictions'
                    if not ('categories' in parsed or 'predictions' in parsed):
                        # Some models might output plain dictionary without root key, strict check depends on prompt
                        pass 
            except:
                # Fallback: sometimes response is wrapped in markdown ```json ... ```
                # Simple cleanup check
                clean_resp = raw_resp.strip()
                if clean_resp.startswith("```json"):
                    clean_resp = clean_resp[7:].split("```")[0].strip()
                    try:
                        json.loads(clean_resp)
                        is_valid_json = True
                    except:
                        pass
            
            if not is_valid_json:
                bad_json_count += 1
                
            # 2. Check Prediction Result (Post-processing)
            preds_str = str(row.get('predicted_codes', ''))
            try:
                preds_list = ast.literal_eval(preds_str) if preds_str.startswith('[') else []
                if not isinstance(preds_list, list) or len(preds_list) == 0:
                     empty_preds_count += 1
            except:
                empty_preds_count += 1

        filename = os.path.basename(file_path)
        if len(filename) > 58:
            filename = filename[:55] + "..."
            
        status = "OK"
        if bad_json_count > 0 or empty_preds_count > 0:
            status = "ISSUES FOUND"
            
        # Highlight high failure rates
        if bad_json_count / total > 0.1:
            status += " (HIGH JSON FAIL)"
            
        print(f"{filename:<60} | {total:<6} | {bad_json_count:<8} | {empty_preds_count:<11} | {status}")

    except Exception as e:
        print(f"Error reading {os.path.basename(file_path)}: {e}")
