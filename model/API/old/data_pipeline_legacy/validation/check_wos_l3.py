
import pandas as pd
df = pd.read_csv('f:/Git/NLP_TEXT_CLASSIFICATION/model/API/datasets/wos_unified.csv')
print(f'Total rows: {len(df)}')
print(f'L3 non-null count: {df["ddc_l3"].notna().sum()}')
if df["ddc_l3"].notna().sum() > 0:
    print("Example L3 values:")
    print(df[df["ddc_l3"].notna()]["ddc_l3"].head())
