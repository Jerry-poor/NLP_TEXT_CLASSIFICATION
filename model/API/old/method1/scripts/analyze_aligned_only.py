
import pandas as pd
import os

file2_path = r"f:\Git\NLP_TEXT_CLASSIFICATION\model\API\classification_results\wos_ddc_level1_sample500_deepseek_chat_aligned.csv"

def normalize_code(val):
    s = str(val).split('.')[0].strip()
    if s.isdigit():
        return s.zfill(3)
    return s

def analyze_aligned():
    if not os.path.exists(file2_path):
        print("File not found")
        return

    df = pd.read_csv(file2_path)
    
    df['true_code'] = df['true_level1_code'].apply(normalize_code)
    df['pred_code'] = df['predicted_code'].apply(normalize_code)
    
    valid_df = df.dropna(subset=['true_code', 'pred_code'])
    total = len(valid_df)
    
    print(f"\n{'='*20}\nANALYZING: Aligned\n{'='*20}")
    print(f"Total: {total}")

    print(f"\n[Ground Truth Distribution]")
    vc = valid_df['true_code'].value_counts()
    for code in ['000', '500', '600']:
        count = vc.get(code, 0)
        print(f"DDC {code}: {count} ({count/total:.2%})")

    errors = valid_df[valid_df['true_code'] != valid_df['pred_code']]
    print(f"\n[Error Analysis]")
    print(f"Total Errors: {len(errors)} (Accuracy: {1 - len(errors)/total:.2%})")
    
    if len(errors) > 0:
        print("\nTop 5 Wrongly Predicted Labels (What did it predict instead?):")
        print(errors['pred_code'].value_counts().head(5))
        
        print("\nTop 5 Confusions (True -> Pred):")
        errors['confusion'] = errors['true_code'] + " -> " + errors['pred_code']
        print(errors['confusion'].value_counts().head(5))

if __name__ == "__main__":
    analyze_aligned()
