import pandas as pd
import os

base_dir = r'f:\Git\NLP_TEXT_CLASSIFICATION\model\API\datasets'
output_dir = os.path.join(base_dir, 'unified')
os.makedirs(output_dir, exist_ok=True)

def zfill3(val):
    if pd.isna(val): return ""
    try:
        return str(int(float(val))).zfill(3)
    except:
        return str(val).strip().zfill(3)

# 1. Process AG News
print("Processing AG News...")
ag_path = os.path.join(base_dir, 'AG_news', 'AG_news_test_with_DDC_Hierarchy_mod_26Nov25_english_mapped.csv')
ag_df = pd.read_csv(ag_path)
ag_unified = pd.DataFrame({
    'title': ag_df['Title'],
    'abstract': ag_df['Abstract'],
    'ddc_l1': ag_df['DDC_Level1'].apply(zfill3),
    'ddc_l2': ag_df['DDC_Level2'].apply(zfill3),
    'ddc_l3': ag_df['DDC_Level3'].apply(zfill3)
})
ag_unified.to_csv(os.path.join(output_dir, 'ag_news_unified.csv'), index=False, encoding='utf-8')

# 2. Process WOS
print("Processing WOS...")
wos_path = os.path.join(base_dir, 'WOS', 'WOS_46985_Dataset_rev_26Nov25_ddc12_mapped.csv')
wos_df = pd.read_csv(wos_path)
# WOS uses keywords as title equivalent in many scripts
wos_unified = pd.DataFrame({
    'title': wos_df['keywords'],
    'abstract': wos_df['Abstract'],
    'ddc_l1': wos_df['DDC_L1_code'].apply(zfill3),
    'ddc_l2': wos_df['DDC_L2_code'].apply(zfill3),
    'ddc_l3': "" # WOS in this file doesn't have L3
})
wos_unified.to_csv(os.path.join(output_dir, 'wos_unified.csv'), index=False, encoding='utf-8')

# 3. Process Lib (Merge 3 levels)
print("Processing Lib...")
lib_l1 = pd.read_excel(os.path.join(base_dir, 'Lib', 'Lib_Dataset_Level1_30Nov25_final.xlsx'))
lib_l2 = pd.read_excel(os.path.join(base_dir, 'Lib', 'Lib_Dataset_Level2_27Nov25_final.xlsx'))
lib_l3 = pd.read_excel(os.path.join(base_dir, 'Lib', 'Lib_Dataset_Level3_26Nov25_final.xlsx'))

# Combine them. Assuming Title/Abstract can be used to join or just concatenate. 
# Since they are labeled per level, we'll try to align them if they share titles, 
# otherwise we keep them as separate rows but standard columns.
# Actually, the user likely wants a single dataset where each row has the best available hierarchy.
# For now, let's treat them as a combined pool of samples.

def process_lib_df(df, l_num):
    col = f'DDC-L{l_num}'
    # Normalize column names to title, abstract and the specific ddc level
    res = pd.DataFrame({
        'title': df['Title'],
        'abstract': df['Abstract'],
        'ddc_l1': df[col].apply(zfill3) if l_num == 1 else "",
        'ddc_l2': df[col].apply(zfill3) if l_num == 2 else "",
        'ddc_l3': df[col].apply(zfill3) if l_num == 3 else ""
    })
    return res

lib_unified = pd.concat([
    process_lib_df(lib_l1, 1),
    process_lib_df(lib_l2, 2),
    process_lib_df(lib_l3, 3)
], ignore_index=True)

lib_unified.to_csv(os.path.join(output_dir, 'lib_unified.csv'), index=False, encoding='utf-8')

print(f"Unified datasets saved to {output_dir}")
