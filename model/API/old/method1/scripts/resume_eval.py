#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Resume evaluation for AG News / Level classification by retrying failed/empty rows.
This script reads an existing results CSV, identifies rows with missing predictions (or API errors),
re-runs classification for them, and updates the CSV.
"""

import argparse
import os
import sys
import pandas as pd
from typing import Optional, Dict

# Add script directory to path to import local modules
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SCRIPT_DIR)

from llm_classifier_utils import get_client, save_results, load_results
from ag_news_multi_level_eval import (
    load_mapping, 
    build_candidates_for_code, 
    create_prompt, 
    parse_code, 
    compute_topk, 
    build_support_map, 
    bucket_key,
    _zfill3,
    normalize_level3_code
)

from concurrent.futures import ThreadPoolExecutor, as_completed

def resume_eval(
    level: int,
    input_csv: str,  # Original dataset
    result_csv: str, # Partially completed results
    mapping_csv: str,
    output_csv: str, # Where to save updated results (can be same as result_csv)
    provider: str = "qwen",
    model: Optional[str] = None,
    max_workers: int = 5,
):
    print(f"Loading original data from {input_csv}...")
    df_orig = pd.read_csv(input_csv)
    
    print(f"Loading existing results from {result_csv}...")
    # Load results effectively
    try:
        results = load_results(result_csv)
        print(f"Loaded {len(results)} rows from results file.")
    except Exception as e:
        print(f"Error loading results: {e}")
        return

    # Create a map of index -> result_row for quick lookup
    # Assuming 'index' column exists in results and corresponds to df_orig index
    # However, sample_size might have been used. 
    # Logic: The 'index' in results should map back to df_orig index.
    
    result_map = {int(r['index']): r for r in results if 'index' in r and str(r['index']).isdigit()}
    
    # Identify failed rows
    # Criteria 1: Rows missing explicitly in results (empty prediction)
    failed_indices = []
    
    # Check loaded results for failure
    for idx, res in result_map.items():
        preds = res.get('predicted_categories', [])
        if not preds or preds == "[]":
            failed_indices.append(idx)
            
    print(f"Found {len(failed_indices)} rows with empty/failed predictions in results.")

    # Criteria 2: Rows COMPLETELY MISSING from results
    # We need to know which rows were supposed to be run.
    # In ag_news_multi_level_eval.py, sampling was done:
    #   eval_df = df.drop(index=used_idx).copy()
    #   eval_df = eval_df.sample(n=min(sample_size, len(eval_df)), random_state=random_seed)...
    # Since we can't easily reproduce the exacting sampling without code duplication,
    # we assume result_map keys are the ground truth of what WAS attempted (or at least partially).
    # BUT if rows dropped entirely, we might miss them.
    #
    # HEURISTIC: If we have a 'sample_size' parameter we might guess, but here we only have
    # the input CSV and the result CSV.
    #
    # If the user says "only 493 samples" but we expect 500, we simply verify against the set of indices
    # present in the result file versus what we MIGHT expect.
    #
    # However, without exact sampling logic reproduction, safely identifying "missing" rows is hard
    # UNLESS we assume the inputs to this script (input_csv) was ALREADY the sampled dataset?
    #
    # No, input_csv is the FULL dataset.
    #
    # To fix specific missing rows in Level 3 case (493 vs 500), we can re-run the sampling logic to find expected indices.
    #
    # Let's import the sampling logic from ag_news_multi_level_eval.
    
    from ag_news_multi_level_eval import build_support_map
    
    # Re-simulate sampling to find target indices
    # We need sample_size and random_seed defaults from the eval script (500, 42)
    # Adding arguments for them would be best, but let's default to standard values.
    
    sample_size = 500
    random_seed = 42
    
    support_map, used_idx = build_support_map(df_orig, level)
    eval_df = df_orig.drop(index=used_idx).copy()
    eval_df = eval_df.sample(n=min(sample_size, len(eval_df)), random_state=random_seed).reset_index(drop=True)
    
    # In ag_news_multi_level_eval.py:
    # futures = {executor.submit(classify_one, item): item[0] for item in eval_df.reset_index().iterrows()}
    # Here item[0] is the index AFTER reset_index (0 to 499), NOT the original DataFrame index.
    # The 'index' field recorded in results comes from 'idx' in classify_one, which is this 0-499 index.
    
    expected_indices = set(eval_df.index) 
    
    # Debug info
    print(f"Debug: Expected index range: {min(expected_indices)}-{max(expected_indices)}, Count: {len(expected_indices)}")
    if result_map:
        res_indices = list(result_map.keys())
        print(f"Debug: Result map keys range: {min(res_indices)}-{max(res_indices)}, Count: {len(res_indices)}")

    
    existing_indices = set(result_map.keys())
    missing_indices = expected_indices - existing_indices
    
    if missing_indices:
        print(f"Found {len(missing_indices)} rows completely missing from results (expected {len(expected_indices)}, found {len(existing_indices)}).")
        failed_indices.extend(list(missing_indices))
    
    # Deduplicate
    failed_indices = list(set(failed_indices))
    
    print(f"Total rows to retry: {len(failed_indices)}")
    
    if not failed_indices:
        print("No failed or missing rows found. Exiting.")
        return

    # Load resources
    maps = load_mapping(mapping_csv)
    client = get_client(provider, model=model)
    
    # We need to rebuild support map to consistent with original run logic?
    # Ideally yes. Or just re-build fresh supports.
    # ag_news_multi_level_eval uses build_support_map which consumes some rows.
    # We should exclude rows used as support from being evaluated?
    # But here we are ONLY retrying rows that were ALREADY ATTEMPTED (so they are not support rows).
    
    support_map, used_idx = build_support_map(df_orig, level)
    default_support = next(iter(support_map.values()))
    
    code_col = {1: "DDC_Level1", 2: "DDC_Level2", 3: "DDC_Level3"}[level]

    def retry_one(idx):
        # idx is the 0-499 index from eval_df
        try:
            row = eval_df.loc[idx]
        except KeyError:
            print(f"Error: Index {idx} not found in eval_df")
            return None
            
        if level == 3:
            true_code = normalize_level3_code(row[code_col])
        else:
            true_code = _zfill3(row[code_col])
            
        if not true_code:
            return None
            
        candidates = build_candidates_for_code(level, true_code, maps)
        if not candidates:
            return None
            
        b = bucket_key(level, true_code)
        support = support_map.get(b, default_support)
        prompt = create_prompt(row["Title"], row["Abstract"], candidates, [support])

        # API Call
        try:
            pred = client.classify_text_with_confidence(prompt)
        except Exception as e:
            # If API call fails, print error but don't crash
            print(f"API Error for sample {idx}: {e}")
            return None
        
        # If result is empty or None
        if not pred:
            return None 
            
        pred_with_codes = []
        pred_code = None
        for cat, conf in pred:
            code = parse_code(cat, candidates)
            pred_with_codes.append((cat, conf, code))
            if pred_code is None and code:
                pred_code = code

        return {
            "index": idx,
            "title": row["Title"],
            "abstract": row["Abstract"],
            f"true_level{level}_code": true_code,
            "predicted_code": pred_code,
            "predicted_name": candidates.get(pred_code, "") if pred_code else "",
            "predicted_categories": pred,
            "predicted_categories_with_codes": pred_with_codes,
            "candidate_count": len(candidates),
            "provider": provider,
            "model": model or "",
        }

    # Retry with ThreadPool
    print(f"Retrying {len(failed_indices)} tasks with concurrency {max_workers}...")
    
    new_results_map = {}
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {executor.submit(retry_one, idx): idx for idx in failed_indices}
        
        processed = 0
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            processed += 1
            try:
                res = future.result()
                if res:
                    new_results_map[idx] = res
                    print(f"[{processed}/{len(failed_indices)}] Fixed sample index {idx}")
                else:
                    print(f"[{processed}/{len(failed_indices)}] Failed again sample index {idx}")
            except Exception as exc:
                print(f"Sample {idx} raised exc: {exc}")

    # Merge results
    success_count = 0
    for idx, new_res in new_results_map.items():
        # Update original result map
        result_map[idx] = new_res
        success_count += 1
        
    print(f"Successfully fixed {success_count} rows.")
    
    # Convert map back to list
    final_results = list(result_map.values())
    final_results.sort(key=lambda x: int(x.get('index', 0)))
    
    # Save
    save_results(final_results, output_csv)
    
    # Compute metrics
    metrics = compute_topk(final_results, level)
    counts = metrics["counts"]
    total = counts["total"] or 1
    
    md_path = output_csv.replace(".csv", ".md")
    md_lines = [
        f"# AG_news Level-{level} Summary (Resumed)",
        f"- Samples: {counts['total']}",
        f"- Top-1: {metrics['top1']:.2%} ({counts['top1']}/{total})",
        f"- Top-3: {metrics['top3']:.2%} ({counts['top3']}/{total})",
        f"- Top-5: {metrics['top5']:.2%} ({counts['top5']}/{total})",
        f"- Output CSV: {output_csv}",
    ]
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
        
    print(f"Updated summary written to {md_path}")

def main():
    parser = argparse.ArgumentParser(description="Resume failed AG News evaluations")
    parser.add_argument("--level", type=int, required=True)
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--result_csv", type=str, required=True, help="Path to incomplete results csv")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--mapping_csv", type=str, default=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "dewey_decimal_unique.csv"))
    parser.add_argument("--provider", type=str, default="qwen")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--max_workers", type=int, default=5)
    
    args = parser.parse_args()
    
    resume_eval(
        level=args.level,
        input_csv=args.input,
        result_csv=args.result_csv,
        mapping_csv=args.mapping_csv,
        output_csv=args.output,
        provider=args.provider,
        model=args.model,
        max_workers=args.max_workers
    )

if __name__ == "__main__":
    main()
