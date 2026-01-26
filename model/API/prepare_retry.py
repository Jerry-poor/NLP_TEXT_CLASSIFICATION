import pandas as pd
import os
import shutil

# Target configurations to fix
TARGETS = [
    {
        "model_dir": "gpt-52-2025-12-11",
        "csv_name": "lib_unified_level2_sample500_zero_gpt-5.2-2025-12-11.csv",
        "dataset_name": "lib_unified",
        "level": 2
    },
    {
        "model_dir": "qwen3-235b-a22b-instruct-2507",
        "csv_name": "wos_unified_level2_sample500_zero_qwen3-235b-a22b-instruct-2507.csv",
        "dataset_name": "wos_unified",
        "level": 2
    }
]

BASE_DIR = r"F:\Git\NLP_TEXT_CLASSIFICATION\model\API\result"

def is_failed(row):
    # Check 1: UNKNOWN code
    if str(row.get('predicted_code', '')).upper() == 'UNKNOWN':
        return True
    
    # Check 2: Empty prediction list
    pred_list_str = str(row.get('predicted_codes', ''))
    if pred_list_str in ['[]', '']:
        return True
        
    # Check 3: Explicit API Error in response text
    response_text = str(row.get('response', '')).lower()
    error_keywords = ["timeout", "api failure", "error:", "exception", "max retries exceeded"]
    if any(k in response_text for k in error_keywords):
        return True
        
    return False

def prepare_retry():
    print("Preparing checkpoints for retry...")
    
    for target in TARGETS:
        csv_path = os.path.join(BASE_DIR, target['model_dir'], target['csv_name'])
        
        if not os.path.exists(csv_path):
            print(f"Skipping {target['csv_name']}, file not found.")
            continue
            
        print(f"\nProcessing {target['csv_name']}...")
        
        # Load existing results
        df = pd.read_csv(csv_path, on_bad_lines='skip')
        original_count = len(df)
        
        # Filter for SUCCESSFUL rows only
        # We KEEP the rows that did NOT fail, so the orchestrator skips them.
        success_df = df[~df.apply(is_failed, axis=1)]
        success_count = len(success_df)
        failed_count = original_count - success_count
        
        print(f"  Total: {original_count}, Success: {success_count}, Failed (to retry): {failed_count}")
        
        if failed_count == 0:
            print("  No failed rows found, nothing to retry.")
            continue
            
        # Construct Checkpoint Path
        # Format: .checkpoint_{dataset_name}_level{level}.csv inside the model dir
        # Note: The model_name part of the directory is 'gpt-52-2025-12-11'
        checkpoint_filename = f".checkpoint_{target['dataset_name']}_level{target['level']}.csv"
        checkpoint_path = os.path.join(BASE_DIR, target['model_dir'], checkpoint_filename)
        
        # Save valid rows to checkpoint
        success_df.to_csv(checkpoint_path, index=False)
        print(f"  Created checkpoint at: {checkpoint_path}")
        print(f"  Now you can re-run main_evaluation.py for this model/dataset/level.")
        
        # Optional: Backup original file just in case
        backup_path = csv_path + ".bak"
        shutil.copy2(csv_path, backup_path)
        print(f"  Backed up original result to: {os.path.basename(backup_path)}")

if __name__ == "__main__":
    prepare_retry()
