
import pandas as pd
import numpy as np
import os
import sys

level1_path = r"f:\Git\NLP_TEXT_CLASSIFICATION\model\API\classification_results\wos_ddc_level1_sample500_deepseek_chat_aligned.csv"

def analyze_level1():
    print(f"\n{'='*30}\nAnalyzing Level 1 ONLY\n{'='*30}")
    if not os.path.exists(level1_path):
        print("File not found.")
        return

    df = pd.read_csv(level1_path)
    
    true_col = "true_level1_code"
    pred_col = "predicted_code"

    if true_col not in df.columns:
        # Fallback search
        cols = df.columns
        match = [c for c in cols if "true" in c and "level1" in c]
        if match: true_col = match[0]
    
    print(f"Using True Col: {true_col}, Pred Col: {pred_col}")

    df_clean = df.dropna(subset=[true_col, pred_col]).copy()
    
    # Normalize
    df_clean['y_true'] = df_clean[true_col].astype(str).str.strip().str.zfill(3)
    df_clean['y_pred'] = df_clean[pred_col].astype(str).str.strip().str.zfill(3)
    
    # Check if True labels are proper Level 1 codes (ending in 00)
    def is_valid_level1(code):
        return code.endswith("00")
    
    df_clean['true_is_valid_l1'] = df_clean['y_true'].apply(is_valid_level1)
    df_clean['pred_is_valid_l1'] = df_clean['y_pred'].apply(is_valid_level1)
    
    invalid_true_count = (~df_clean['true_is_valid_l1']).sum()
    invalid_pred_count = (~df_clean['pred_is_valid_l1']).sum()
    
    print(f"Total Rows: {len(df_clean)}")
    print(f"Rows with True Code NOT ending in '00': {invalid_true_count} ({invalid_true_count/len(df_clean):.2%})")
    print(f"Rows with Pred Code NOT ending in '00': {invalid_pred_count} ({invalid_pred_count/len(df_clean):.2%})")

    # Accuracy with strict match
    strict_acc = (df_clean['y_true'] == df_clean['y_pred']).mean()
    print(f"Strict Accuracy: {strict_acc:.4f}")

    # Accuracy with normalized True labels
    # Normalize True to hundreds
    # E.g. 540 -> 500
    df_clean['y_true_normalized'] = df_clean['y_true'].apply(lambda x: x[0] + "00")
    
    norm_acc = (df_clean['y_true_normalized'] == df_clean['y_pred']).mean()
    print(f"Normalized Accuracy (True label converted to Level 1): {norm_acc:.4f}")

    # Show some mismatches where normalization helps
    print("\nExamples where Strict Match failed but Normalized Match succeeds:")
    fixed = df_clean[(df_clean['y_true'] != df_clean['y_pred']) & (df_clean['y_true_normalized'] == df_clean['y_pred'])]
    
    if not fixed.empty:
        print(fixed[[true_col, pred_col, 'y_true_normalized']].head(10))
    else:
        print("None found.")

if __name__ == "__main__":
    analyze_level1()
