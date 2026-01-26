import pandas as pd
import os
import re
import hashlib

# Configuration
BASE_DIR = r'f:\Git\NLP_TEXT_CLASSIFICATION\model\API\datasets'
OUTPUT_DIR = os.path.join(BASE_DIR, 'final')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def clean_text(text):
    if not isinstance(text, str):
        return ""
    # Normalize whitespace and convert to lower case for fingerprinting
    return re.sub(r'\s+', ' ', text).strip().lower()

def get_fingerprint(text):
    # MD5 fingerprint of cleaned text for deduplication
    cleaned = clean_text(text)
    if not cleaned:
        return None
    return hashlib.md5(cleaned.encode('utf-8')).hexdigest()

def zfill_code(val, width=3):
    if pd.isna(val):
        return ""
    s = str(val).strip()
    if s == "" or s.lower() == "nan":
        return ""
    try:
        # Handle cases like "500.0" -> "500"
        return str(int(float(s))).zfill(width)
    except:
        return s.zfill(width)

def save_dataset(df, name):
    output_path = os.path.join(OUTPUT_DIR, f"{name}_final.csv")
    print(f"[{name}] Saving {len(df)} rows to {output_path}...")
    df.to_csv(output_path, index=False, encoding='utf-8')

# --- 1. Process Library Dataset (Lib) ---
print("\n=== Processing Library Dataset (Lib) ===")
lib_files = {
    1: os.path.join(BASE_DIR, 'Lib', 'Lib_Dataset_Level1_30Nov25_final.xlsx'),
    2: os.path.join(BASE_DIR, 'Lib', 'Lib_Dataset_Level2_27Nov25_final.xlsx'),
    3: os.path.join(BASE_DIR, 'Lib', 'Lib_Dataset_Level3_26Nov25_final.xlsx')
}

lib_chunks = []

for level, path in lib_files.items():
    if not os.path.exists(path):
        print(f"Warning: File not found {path}")
        continue
    
    print(f"Reading Level {level}: {os.path.basename(path)}")
    df = pd.read_excel(path)
    
    # Map columns based on level
    chunk = pd.DataFrame()
    chunk['title'] = df['Title']
    chunk['abstract'] = df['Abstract']
    
    # Handle DDC columns
    col_name = f'DDC-L{level}'
    chunk['primary_code'] = df[col_name].apply(lambda x: zfill_code(x, 3))
    
    # Derive other levels based on primary code
    # This assumes standard DDC hierarchy: L1=Hundreds, L2=Tens, L3=Units
    if level == 3:
        chunk['ddc_l3'] = chunk['primary_code']
        chunk['ddc_l2'] = chunk['primary_code'].apply(lambda x: x[:2] + '0' if len(x)>=2 else "")
        chunk['ddc_l1'] = chunk['primary_code'].apply(lambda x: x[:1] + '00' if len(x)>=1 else "")
        chunk['depth'] = 3
    elif level == 2:
        chunk['ddc_l3'] = ""
        chunk['ddc_l2'] = chunk['primary_code']
        chunk['ddc_l1'] = chunk['primary_code'].apply(lambda x: x[:1] + '00' if len(x)>=1 else "")
        chunk['depth'] = 2
    elif level == 1:
        chunk['ddc_l3'] = ""
        chunk['ddc_l2'] = ""
        chunk['ddc_l1'] = chunk['primary_code']
        chunk['depth'] = 1
        
    lib_chunks.append(chunk)

# Merge Lib
if lib_chunks:
    lib_raw = pd.concat(lib_chunks, ignore_index=True)
    total_raw = len(lib_raw)
    
    # Generate fingerprint
    lib_raw['fingerprint'] = lib_raw['abstract'].apply(get_fingerprint)
    
    # Sort by depth descending (so deeper levels come first)
    lib_raw.sort_values('depth', ascending=False, inplace=True)
    
    # Deduplicate keeping first (deepest) match
    lib_final = lib_raw.drop_duplicates(subset=['fingerprint'], keep='first')
    
    # Clean up aux columns
    lib_final = lib_final[['title', 'abstract', 'ddc_l1', 'ddc_l2', 'ddc_l3']].copy()
    
    print(f"Lib Deduplication: {total_raw} -> {len(lib_final)} (Removed {total_raw - len(lib_final)} duplicates)")
    save_dataset(lib_final, "Lib")

# --- 2. Process AG News ---
print("\n=== Processing AG News ===")
ag_path = os.path.join(BASE_DIR, 'AG_news', 'AG_news_test_with_DDC_Hierarchy_mod_26Nov25_english_mapped.csv')
if os.path.exists(ag_path):
    df_ag = pd.read_csv(ag_path)
    
    ag_final = pd.DataFrame()
    ag_final['title'] = df_ag['Title']
    ag_final['abstract'] = df_ag['Abstract']
    ag_final['ddc_l1'] = df_ag['DDC_Level1'].apply(lambda x: zfill_code(x, 3))
    ag_final['ddc_l2'] = df_ag['DDC_Level2'].apply(lambda x: zfill_code(x, 3))
    ag_final['ddc_l3'] = df_ag['DDC_Level3'].apply(lambda x: zfill_code(x, 3))
    
    raw_len = len(ag_final)
    ag_final['fingerprint'] = ag_final['abstract'].apply(get_fingerprint)
    ag_final.drop_duplicates(subset=['fingerprint'], keep='first', inplace=True)
    
    ag_final = ag_final[['title', 'abstract', 'ddc_l1', 'ddc_l2', 'ddc_l3']]
    
    print(f"AG News Deduplication: {raw_len} -> {len(ag_final)}")
    save_dataset(ag_final, "AG_News")
else:
    print(f"skip AG News: {ag_path} not found")

# --- 3. Process WOS ---
print("\n=== Processing WOS ===")
wos_path = os.path.join(BASE_DIR, 'WOS', 'WOS_46985_Dataset_rev_26Nov25_ddc12_mapped.csv')
if os.path.exists(wos_path):
    df_wos = pd.read_csv(wos_path)
    
    wos_final = pd.DataFrame()
    # WOS uses keywords often as pseudo-title or signal, we map keywords to title for structure consistency
    wos_final['title'] = df_wos['keywords'] 
    wos_final['abstract'] = df_wos['Abstract']
    wos_final['ddc_l1'] = df_wos['DDC_L1_code'].apply(lambda x: zfill_code(x, 3))
    wos_final['ddc_l2'] = df_wos['DDC_L2_code'].apply(lambda x: zfill_code(x, 3))
    wos_final['ddc_l3'] = "" # WOS typically L1/L2 only in this file
    
    raw_len = len(wos_final)
    wos_final['fingerprint'] = wos_final['abstract'].apply(get_fingerprint)
    wos_final.drop_duplicates(subset=['fingerprint'], keep='first', inplace=True)
    
    wos_final = wos_final[['title', 'abstract', 'ddc_l1', 'ddc_l2', 'ddc_l3']]
    
    print(f"WOS Deduplication: {raw_len} -> {len(wos_final)}")
    save_dataset(wos_final, "WOS")
else:
    print(f"skip WOS: {wos_path} not found")

print("\nDone.")
