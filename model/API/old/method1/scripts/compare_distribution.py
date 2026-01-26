
import pandas as pd
import os

file1 = r"f:\Git\NLP_TEXT_CLASSIFICATION\model\API\classification_results\wos_ddc_level1_sample500_deepseek_chat.csv"
file2 = r"f:\Git\NLP_TEXT_CLASSIFICATION\model\API\classification_results\wos_ddc_level1_sample500_deepseek_chat_aligned.csv"

def analyze(path, name):
    print(f"\nAnalyzing: {name}")
    if not os.path.exists(path):
        print("File not found.")
        return

    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # Find the ground truth column
    true_col = None
    possible_cols = ['true_level1_code', 'true_category', 'true_code', 'DDC_Level1', 'correct_code']
    
    for col in df.columns:
        if col in possible_cols:
            true_col = col
            break
    
    if not true_col:
        # Fuzzy match
        for col in df.columns:
            if "true" in col.lower() and "pred" not in col.lower():
                true_col = col
                break
    
    if not true_col:
        print(f"Could not find ground truth column. Columns: {df.columns.tolist()}")
        return

    print(f"Using column: '{true_col}'")
    
    # Normalize to string and just take the numeric part if possible for safety, though keeping it simple is better
    # The user asks for DDC500 and DDC600.
    # Level 1 codes in this dataset seem to be 3 digits like '500', '600'.
    
    # Clean data
    df[true_col] = df[true_col].astype(str).str.strip().str.replace(r'\.0$', '', regex=True)
    
    # Count 500 and 600
    # Also we might check if there are 5xx vs 6xx if exact matches are rare
    
    counts = df[true_col].value_counts()
    
    count_500 = counts.get('500', 0)
    count_600 = counts.get('600', 0)
    
    print(f"Count of '500': {count_500}")
    print(f"Count of '600': {count_600}")
    
    # Also show general distribution of starting digits just in case
    df['first_digit'] = df[true_col].apply(lambda x: x[0] if len(x)>0 else '')
    print("Distribution by first digit (Century):")
    print(df['first_digit'].value_counts().sort_index())
    
    # Calculate percentages
    total = len(df)
    print(f"Total rows: {total}")
    print(f"Percentage '500': {count_500/total:.2%}")
    print(f"Percentage '600': {count_600/total:.2%}")

analyze(file1, "Dataset 1 (Original)")
analyze(file2, "Dataset 2 (Aligned)")
