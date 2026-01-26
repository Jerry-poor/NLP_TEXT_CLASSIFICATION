
import pandas as pd
import os

level2_path = r"f:\Git\NLP_TEXT_CLASSIFICATION\model\API\classification_results\wos_ddc_level2_sample500_deepseek_chat_aligned.csv"

def analyze_level2():
    print(f"Analyzing Level 2: {level2_path}")
    if not os.path.exists(level2_path):
        print("File not found")
        return
        
    df = pd.read_csv(level2_path)
    true_col = "true_level2_code"
    pred_col = "predicted_code"
    
    df = df.dropna(subset=[true_col, pred_col])
    
    # Normalize 
    y_true = df[true_col].astype(str).str.strip().str.zfill(3)
    y_pred = df[pred_col].astype(str).str.strip().str.zfill(3)
    
    acc = (y_true == y_pred).mean()
    print(f"Level 2 Accuracy: {acc:.4f} ({ (y_true == y_pred).sum() }/{len(df)})")

if __name__ == "__main__":
    analyze_level2()
