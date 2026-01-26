import os
import pandas as pd
import glob

# Base directory
base_dir = r"F:\Git\NLP_TEXT_CLASSIFICATION\model\API\result"

# Pattern to match result CSVs (excluding summaries)
files = glob.glob(os.path.join(base_dir, "**", "*_sample500_zero_*.csv"), recursive=True)

print(f"{'File':<60} | {'Total':<6} | {'Errors':<6} | {'Rate':<7} | {'Primary Error Types'}")
print("-" * 120)

for file_path in files:
    try:
        # Read CSV
        df = pd.read_csv(file_path, on_bad_lines='skip')
        
        total = len(df)
        if total == 0:
            print(f"{os.path.basename(file_path):<60} | {0:<6} | {0:<6} | {0.0:<7.1%} | Empty File")
            continue

        # Initialize error flags
        # 1. Prediction is UNKNOWN
        unknown_mask = df['predicted_code'].astype(str).str.upper() == 'UNKNOWN'
        
        # 2. Predicted codes is empty list or error string
        # Often '[]' implies a failure to parse or no prediction found.
        # Also check for explicit error messages in predicted_codes if it happens to be populated with them
        empty_pred_mask = df['predicted_codes'].astype(str).isin(['[]', ''])
        
        # 3. Explicit error keywords in response (if available)
        # Common error strings mentioned: "Timeout", "API Failure", "Error:"
        error_keywords = ["timeout", "api failure", "error:", "exception", "max retries exceeded"]
        response_error_mask = pd.Series([False] * total)
        
        if 'response' in df.columns:
            response_error_mask = df['response'].astype(str).str.lower().apply(
                lambda x: any(k in x for k in error_keywords)
            )
            
        # Combined error mask
        # We consider it an error if it's UNKNOWN OR (Empty codes) OR (Explicit Error in Response)
        # Note: Sometimes empty codes might be valid if the model really thinks none apply, but for this task we expect 5.
        # So empty is likely an error.
        
        failed_rows = df[unknown_mask | empty_pred_mask | response_error_mask]
        error_count = len(failed_rows)
        error_rate = error_count / total if total > 0 else 0
        
        # Analyze error types for this file
        error_types = []
        
        # Check for Network/Timeout specifically
        if 'response' in df.columns:
            network_errors = failed_rows['response'].astype(str).str.lower().apply(
                lambda x: 'timeout' in x or 'connection' in x or 'api failure' in x
            ).sum()
            if network_errors > 0:
                error_types.append(f"Network/Timeout ({network_errors})")
                
            json_errors = failed_rows['response'].astype(str).str.lower().apply(
                lambda x: 'json' in x or 'format' in x
            ).sum()
            if json_errors > 0:
                error_types.append(f"JSON/Format ({json_errors})")

        # Fallback if we just have UNKNOWN without clear response message
        if not error_types and error_count > 0:
             error_types.append("Unknown/Generic")

        error_summary = ", ".join(error_types)
        
        # Shorten filename for display
        filename = os.path.basename(file_path)
        # If filename too long, truncate
        if len(filename) > 58:
            filename = filename[:55] + "..."
            
        print(f"{filename:<60} | {total:<6} | {error_count:<6} | {error_rate:<7.1%} | {error_summary}")

    except Exception as e:
        print(f"Error processing {os.path.basename(file_path)}: {str(e)}")
