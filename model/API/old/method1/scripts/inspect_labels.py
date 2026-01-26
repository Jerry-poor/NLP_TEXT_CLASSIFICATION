
import pandas as pd
import os

file1 = r"f:\Git\NLP_TEXT_CLASSIFICATION\model\API\classification_results\wos_ddc_level1_sample500_deepseek_chat.csv"

def inspect_labels():
    if not os.path.exists(file1):
        print("File not found.")
        return

    try:
        df = pd.read_csv(file1)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    print("Unique values in 'true_category':")
    print(df['true_category'].unique())
    
    print("\nValue counts:")
    print(df['true_category'].value_counts())

if __name__ == "__main__":
    inspect_labels()
