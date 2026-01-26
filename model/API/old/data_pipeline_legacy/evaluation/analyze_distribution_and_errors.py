
import pandas as pd
import ast
import os
import re

# Paths
file1_path = r"f:\Git\NLP_TEXT_CLASSIFICATION\model\API\classification_results\wos_ddc_level1_sample500_deepseek_chat.csv"
file2_path = r"f:\Git\NLP_TEXT_CLASSIFICATION\model\API\classification_results\wos_ddc_level1_sample500_deepseek_chat_aligned.csv"

def map_text_to_code(text):
    text = str(text).lower()
    if "computer science" in text or "information" in text: return "000"
    if "technology" in text: return "600"
    if "science" in text and "computer" not in text: return "500"
    if "philosophy" in text or "psychology" in text: return "100"
    if "religion" in text: return "200"
    if "social sciences" in text: return "300"
    if "language" in text: return "400"
    if "arts" in text: return "700"
    if "literature" in text: return "800"
    if "history" in text: return "900"
    return "Unknown"

def parse_prediction_file1(pred_str):
    if not isinstance(pred_str, str): return None
    try:
        # Try AST
        data = ast.literal_eval(pred_str)
        if isinstance(data, list) and len(data) > 0:
            item = data[0]
            # Case 1: List of Dicts [{'category': 'Tech...'}]
            if isinstance(item, dict):
                return map_text_to_code(item.get('category', ''))
            # Case 2: List of Tuples [('Tech', 0.9, '600')]
            if isinstance(item, tuple) or isinstance(item, list):
                # If code is present (3rd element)
                if len(item) >= 3:
                    return str(item[2]).zfill(3)
                # If only name is present
                if len(item) >= 1:
                    return map_text_to_code(item[0])
    except:
        pass

    # Fallback Regex for [{'category': 'Technology'
    m = re.search(r"'category':\s*['\"]([^'\"]+)['\"]", pred_str)
    if m:
        return map_text_to_code(m.group(1))
    
    return None

def normalize_code(val):
    s = str(val).split('.')[0].strip()
    if s.isdigit():
        return s.zfill(3)
    return s

def analyze_dataset(name, df, true_extractor, pred_extractor):
    print(f"\n{'='*20}\nANALYZING: {name}\n{'='*20}")
    
    df['true_code'] = df.apply(true_extractor, axis=1)
    df['pred_code'] = df.apply(pred_extractor, axis=1)
    
    valid_df = df.dropna(subset=['true_code', 'pred_code'])
    valid_df = valid_df[valid_df['true_code'] != "Unknown"]
    
    total = len(valid_df)
    print(f"Total Valid Samples: {total} / {len(df)}")
    
    if total == 0:
        return

    # 1. Distribution of 000, 500, 600 in Ground Truth
    print(f"\n[Ground Truth Distribution]")
    vc = valid_df['true_code'].value_counts()
    for code in ['000', '500', '600']:
        count = vc.get(code, 0)
        print(f"DDC {code}: {count} ({count/total:.2%})")
    
    # 2. Error Analysis
    errors = valid_df[valid_df['true_code'] != valid_df['pred_code']]
    print(f"\n[Error Analysis]")
    print(f"Total Errors: {len(errors)} (Accuracy: {1 - len(errors)/total:.2%})")
    
    if len(errors) > 0:
        print("\nDistribution of PREDICTED LABELS in Error cases (What did it wrongly predict?):")
        print(errors['pred_code'].value_counts().head(10))
        
        print("\nTop 5 specific confusions (True -> Pred):")
        errors['confusion'] = errors['true_code'] + " -> " + errors['pred_code']
        print(errors['confusion'].value_counts().head(5))

def main():
    # File 1: Original
    if os.path.exists(file1_path):
        try:
            df1 = pd.read_csv(file1_path, on_bad_lines='skip')
            def get_true_1(row):
                return map_text_to_code(row.get('true_category', ''))
            def get_pred_1(row):
                return parse_prediction_file1(row.get('predicted_categories', ''))
            analyze_dataset("Original (Sample500 Deepseek)", df1, get_true_1, get_pred_1)
        except Exception as e:
            print(f"Error File 1: {e}")

    # File 2: Aligned
    if os.path.exists(file2_path):
        try:
            df2 = pd.read_csv(file2_path)
            def get_true_2(row):
                return normalize_code(row.get('true_level1_code', ''))
            def get_pred_2(row):
                return normalize_code(row.get('predicted_code', ''))
            analyze_dataset("Aligned (Sample500 Deepseek Aligned)", df2, get_true_2, get_pred_2)
        except Exception as e:
            print(f"Error File 2: {e}")

if __name__ == "__main__":
    main()
