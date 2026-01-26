
import pandas as pd
import sys

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 1000)

level1_path = r"f:\Git\NLP_TEXT_CLASSIFICATION\model\API\classification_results\wos_ddc_level1_sample500_deepseek_chat_aligned.csv"

df = pd.read_csv(level1_path)
print(df[['true_level1_code', 'predicted_code']].head(30))

print("\nValue Counts for true_level1_code:")
print(df['true_level1_code'].value_counts().head(20))

print("\nValue Counts for predicted_code:")
print(df['predicted_code'].value_counts().head(20))
