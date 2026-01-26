
import pandas as pd
import numpy as np
import os
import sys

# Paths to CSVs
level1_path = r"f:\Git\NLP_TEXT_CLASSIFICATION\model\API\classification_results\wos_ddc_level1_sample500_deepseek_chat_aligned.csv"
level2_path = r"f:\Git\NLP_TEXT_CLASSIFICATION\model\API\classification_results\wos_ddc_level2_sample500_deepseek_chat_aligned.csv"

def analyze_file(path, level):
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return

    print(f"\n{'='*30}\nAnalyzing Level {level}\n{'='*30}")
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"Error reading csv: {e}")
        return

    print(f"Columns: {df.columns.tolist()}")
    
    # Identify One-vs-All or standard columns
    # We expect 'true_levelX_code' and 'predicted_code'
    # Check if we have consistent columns
    
    true_col = f"true_level{level}_code"
    pred_col = "predicted_code"

    # Some aligned files might have different naming conventions
    if true_col not in df.columns:
        print(f"WARNING: column '{true_col}' not found.")
        candidates = [c for c in df.columns if "true" in c or "label" in c]
        print(f"Potential candidates: {candidates}")
        if candidates:
            true_col = candidates[0]
            print(f"Using '{true_col}' as ground truth.")
        else:
            return

    if pred_col not in df.columns:
        print(f"WARNING: column '{pred_col}' not found.")
        candidates = [c for c in df.columns if "pred" in c]
        print(f"Potential candidates: {candidates}")
        if candidates:
            pred_col = candidates[0]
            print(f"Using '{pred_col}' as prediction.")
        else:
            return

    # Filter valid rows
    df_clean = df.dropna(subset=[true_col, pred_col]).copy()
    print(f"Total rows: {len(df)}, Valid rows: {len(df_clean)}")

    # Ensure string and normalization
    # Level 1 codes are often '00x' or '0xx'. Level 2 are '000', '010'.
    # For Level 1, we often deal with broad categories: 000, 100, 200...
    # But usually represented as '000', '100', etc.
    # The evaluation script uses '_zfill3'.
    
    def normalize(val):
        s = str(val).split('.')[0].strip() # remove decimals if floats
        return s.zfill(3)

    df_clean['y_true'] = df_clean[true_col].apply(normalize)
    df_clean['y_pred'] = df_clean[pred_col].apply(normalize)
    
    # For Level 1, sometimes we only care about the first digit?
    # But the eval script logic:
    # if level == 1: return code[0] -> used for bucketing support
    # But for accuracy, does it check full code match?
    # In `ag_news_multi_level_eval.py`:
    # candidates = build_candidates_for_code(level, true_code, maps)
    # The candidates likely are the full codes for that level (e.g. 000, 100, 200).
    # So we compare exact string match of 3-digit codes.
    
    y_true = df_clean['y_true']
    y_pred = df_clean['y_pred']

    correct = (y_true == y_pred).sum()
    accuracy = correct / len(df_clean) if len(df_clean) > 0 else 0
    print(f"Accuracy: {accuracy:.4f} ({correct}/{len(df_clean)})")
    
    # Confusion Analysis
    df_clean['match'] = y_true == y_pred
    errors = df_clean[~df_clean['match']]
    
    print("\nTop 10 Errors (True -> Pred):")
    error_counts = errors.apply(lambda row: f"{row['y_true']} -> {row['y_pred']}", axis=1).value_counts().head(10)
    print(error_counts)
    
    # Check if errors are systematically shifting to a specific class
    print("\nMost Common Predicted Labels in Errors:")
    print(errors['y_pred'].value_counts().head(5))

    print("\nMost Common True Labels in Errors:")
    print(errors['y_true'].value_counts().head(5))

if __name__ == "__main__":
    analyze_file(level1_path, 1)
    analyze_file(level2_path, 2)
