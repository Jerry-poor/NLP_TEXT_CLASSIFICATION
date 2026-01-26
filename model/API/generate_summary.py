
import os
import re
import pandas as pd
import traceback

# Constants
BASE_DIR = r"f:\Git\NLP_TEXT_CLASSIFICATION\model\API\result"

# Model Mapping: Directory Name -> File Suffix
MODELS = {
    "qwen3-235b-a22b-instruct-2507": "qwen3-235b-a22b-instruct-2507",
    "gpt-52-2025-12-11": "gpt-5.2-2025-12-11",
    "deepseek-chat": "deepseek-chat"
}

# Dataset Configurations
DATASETS = [
    {
        "name": "lib",
        "file_prefix": "lib_unified",
        "levels": [1, 2, 3]
    },
    {
        "name": "ag_news",
        "file_prefix": "ag_news_unified",
        "levels": [1, 2, 3]
    },
    {
        "name": "wos",
        "file_prefix": "wos_unified",
        "levels": [1, 2]
    }
]

def parse_metrics(file_path):
    """Parses Accuracy (Top-1), Recall (Weighted), and F1 Score (Weighted) from the markdown report."""
    metrics = {
        "accuracy": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0
    }
    
    if not os.path.exists(file_path):
        print(f"Warning: File not found: {file_path}")
        return metrics
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # 1. Parse Top-1 Accuracy (from Accuracy Metrics table)
        # Matches: | **Top-1** | 49.80% |
        match_acc = re.search(r"\|\s*\*\*Top-1\*\*\s*\|\s*([\d\.]+)%\s*\|", content)
        if match_acc:
            metrics["accuracy"] = float(match_acc.group(1)) / 100.0
        else:
            # Fallback to Weighted Metrics table
            match_acc_w = re.search(r"\|\s*\*\*Accuracy\*\*\s*\|\s*([\d\.]+)\s*\|", content)
            if match_acc_w:
                 metrics["accuracy"] = float(match_acc_w.group(1))

        # 2. Parse Precision (Weighted)
        # Matches: | **Precision (Weighted)** | 0.8215 |
        match_prec = re.search(r"\|\s*\*\*Precision \(Weighted\)\*\*\s*\|\s*([\d\.]+)\s*\|", content)
        if match_prec:
            metrics["precision"] = float(match_prec.group(1))

        # 3. Parse Recall (Weighted)
        # Matches: | **Recall (Weighted)** | 0.4980 |
        match_rec = re.search(r"\|\s*\*\*Recall \(Weighted\)\*\*\s*\|\s*([\d\.]+)\s*\|", content)
        if match_rec:
            metrics["recall"] = float(match_rec.group(1))

        # 4. Parse F1 Score (Weighted)
        # Matches: | **F1 Score (Weighted)** | 0.5323 |
        match_f1 = re.search(r"\|\s*\*\*F1 Score \(Weighted\)\*\*\s*\|\s*([\d\.]+)\s*\|", content)
        if match_f1:
            metrics["f1"] = float(match_f1.group(1))
            
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        
    return metrics

def main():
    try:
        results = []

        for dir_name, file_suffix in MODELS.items():
            print(f"Processing model: {file_suffix}")
            row = {"Model": file_suffix}
            
            # Iterate datasets
            for ds in DATASETS:
                ds_name = ds['name']
                file_prefix = ds['file_prefix']
                levels = ds['levels']
                
                combined_acc = 1.0
                
                for lvl in levels:
                    # Construct filename
                    filename = f"{file_prefix}_level{lvl}_sample500_zero_{file_suffix}.md"
                    file_path = os.path.join(BASE_DIR, dir_name, filename)
                    
                    m = parse_metrics(file_path)
                    
                    # Store metrics
                    row[f"{ds_name}_level{lvl}_accuracy"] = m["accuracy"]
                    row[f"{ds_name}_level{lvl}_precision"] = m["precision"]
                    row[f"{ds_name}_level{lvl}_recall"] = m["recall"]
                    row[f"{ds_name}_level{lvl}_f1"] = m["f1"]
                    
                    combined_acc *= m["accuracy"]
                
                # Add combined metric
                row[f"{ds_name}_overall_hit_rate"] = combined_acc
                
            results.append(row)

        print("Creating DataFrame...")
        # Direct DataFrame creation
        df = pd.DataFrame(results)
        
        # Reorder columns
        cols = ["Model"]
        
        for ds in DATASETS:
            ds_name = ds['name']
            for lvl in ds['levels']:
                cols.append(f"{ds_name}_level{lvl}_accuracy")
                cols.append(f"{ds_name}_level{lvl}_precision")
                cols.append(f"{ds_name}_level{lvl}_recall")
                cols.append(f"{ds_name}_level{lvl}_f1")
            cols.append(f"{ds_name}_overall_hit_rate")
            
        # Ensure all columns exist (fill 0 if missing potentially)
        for c in cols:
            if c not in df.columns:
                df[c] = 0.0
                
        df = df[cols]
        df = df.round(4)
        
        output_path = os.path.join(BASE_DIR, "model_comparison_summary_v2.csv")
        print(f"Saving to {output_path}...")
        df.to_csv(output_path, index=False)
        print(f"Summary saved successfully.")
        print(df.to_markdown(index=False))
        
    except Exception as e:
        print("CRITICAL ERROR IN MAIN:")
        traceback.print_exc()

if __name__ == "__main__":
    main()
