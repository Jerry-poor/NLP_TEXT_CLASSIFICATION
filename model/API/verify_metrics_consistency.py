import pandas as pd
import os
import re
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

TARGETS = [
    {
        "name": "GPT-5.2 @ Lib Level 2",
        "csv": r"F:\Git\NLP_TEXT_CLASSIFICATION\model\API\result\gpt-52-2025-12-11\lib_unified_level2_sample500_zero_gpt-5.2-2025-12-11.csv",
        "md": r"F:\Git\NLP_TEXT_CLASSIFICATION\model\API\result\gpt-52-2025-12-11\lib_unified_level2_sample500_zero_gpt-5.2-2025-12-11.md"
    },
    {
        "name": "Qwen @ WOS Level 2",
        "csv": r"F:\Git\NLP_TEXT_CLASSIFICATION\model\API\result\qwen3-235b-a22b-instruct-2507\wos_unified_level2_sample500_zero_qwen3-235b-a22b-instruct-2507.csv",
        "md": r"F:\Git\NLP_TEXT_CLASSIFICATION\model\API\result\qwen3-235b-a22b-instruct-2507\wos_unified_level2_sample500_zero_qwen3-235b-a22b-instruct-2507.md"
    }
]

def parse_md_metrics(md_path):
    if not os.path.exists(md_path):
        return None
    with open(md_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    metrics = {}
    
    # Regex to find metrics in MD tables
    patterns = {
        "Top-1 Acc": r"\|\s*\*\*Top-1\*\*\s*\|\s*([\d\.]+)%\s*\|",
        "Top-5 Acc": r"\|\s*\*\*Top-5\*\*\s*\|\s*([\d\.]+)%\s*\|",
        "Ref: Accuracy": r"\|\s*\*\*Accuracy\*\*\s*\|\s*([\d\.]+)\s*\|", # Weighted table
        "Precision": r"\|\s*\*\*Precision \(Weighted\)\*\*\s*\|\s*([\d\.]+)\s*\|",
        "Recall": r"\|\s*\*\*Recall \(Weighted\)\*\*\s*\|\s*([\d\.]+)\s*\|",
        "F1": r"\|\s*\*\*F1 Score \(Weighted\)\*\*\s*\|\s*([\d\.]+)\s*\|"
    }
    
    for key, pat in patterns.items():
        match = re.search(pat, content)
        if match:
            val = float(match.group(1))
            # Convert percentage to decimal for consistency
            if "%" in pat: 
                metrics[key] = val / 100.0
            else:
                metrics[key] = val
    return metrics

def normalize_code(code):
    """Normalize DDC code to 3 digits string."""
    s = str(code).strip()
    # Remove decimal points like 790.0 or 790.
    if '.' in s:
        s = s.split('.')[0]
    # Extract digits only just in case
    s = ''.join(filter(str.isdigit, s))
    if not s: 
        return "UNKNOWN"
    return s.zfill(3)

def compute_csv_metrics(csv_path):
    if not os.path.exists(csv_path):
        return None
        
    df = pd.read_csv(csv_path, on_bad_lines='skip')
    if df.empty: return None
    
    # Ground Truth
    y_true = df['ddc_code'].apply(normalize_code).tolist()
    
    # Predictions
    # 1. Top-1 for standard metrics
    y_pred_top1 = df['predicted_code'].apply(normalize_code).tolist()
    
    # 2. Top-5 for Top-5 Acc
    # Expect predicted_codes to be a list string like ['790', '700']
    # We need to parse it safely
    y_pred_lists = []
    for x in df['predicted_codes']:
        try:
            # simple eval for list string
            lst = eval(x) if isinstance(x, str) and x.startswith('[') else []
            # normalize each
            y_pred_lists.append([normalize_code(c) for c in lst])
        except:
            y_pred_lists.append([])
            
    # Calculate
    acc = accuracy_score(y_true, y_pred_top1)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred_top1, average='weighted', zero_division=0)
    
    # Top-5 Accuracy Logic
    top5_hits = 0
    for true, preds in zip(y_true, y_pred_lists):
        # We check if TRUE label is in the top 5 preds
        # Standard implementation checks first 5 elements
        if true in preds[:5]:
            top5_hits += 1
    top5_acc = top5_hits / len(y_true)
    
    return {
        "Top-1 Acc": acc, 
        "Top-5 Acc": top5_acc,
        "Ref: Accuracy": acc,
        "Precision": p,
        "Recall": r,
        "F1": f1
    }

print(f"{'Metric':<20} | {'MD (Report)':<12} | {'CSV (Re-calc)':<12} | {'Diff':<8} | {'Status'}")
print("-" * 80)

for target in TARGETS:
    print(f"\nTarget: {target['name']}")
    md_met = parse_md_metrics(target['md'])
    csv_met = compute_csv_metrics(target['csv'])
    
    if not md_met or not csv_met:
        print("  Error: Could not load MD or CSV.")
        continue
        
    for k in ["Top-1 Acc", "Top-5 Acc", "Precision", "Recall", "F1"]:
        v_md = md_met.get(k, 0.0)
        v_csv = csv_met.get(k, 0.0)
        diff = abs(v_md - v_csv)
        
        status = "MATCH" if diff < 0.0001 else "MISMATCH"
        print(f"{k:<20} | {v_md:.4f}       | {v_csv:.4f}       | {diff:.4f}   | {status}")
