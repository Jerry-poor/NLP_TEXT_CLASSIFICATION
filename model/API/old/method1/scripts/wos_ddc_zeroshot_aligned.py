
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WOS evaluation (Zero-shot) aligned with `wos_ddc_fewshot_eval.py` logic:
- Same dataset and filtering (aligned sampling)
- Same metrics (compute_topk)
- Uses bucketed sampling logic just to SKIP the support rows (to keep eval set identical to few-shot)
- BUT prompts are Zero-shot (no examples)
"""

import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
import sys

sys.path.append(SCRIPT_DIR)

from llm_classifier_utils import get_client, save_results  # type: ignore
from multi_level_few_shot import (  # type: ignore
    load_mapping,
    build_candidates_for_code,
    parse_code,
    compute_topk,
)


def _zfill3(val: str) -> str:
    s = str(val).strip()
    if not s or s.lower() == "nan":
        return ""
    try:
        s = str(int(float(s)))
    except Exception:
        pass
    return s.zfill(3)


def bucket_key(level: int, code: str) -> Optional[Tuple[str, int]]:
    code = _zfill3(code)
    if not code:
        return None
    if level in (1, 2):
        return (code[0], level)  # 0-9 buckets
    return None


def build_support_map_dummy(
    df: pd.DataFrame, level: int, code_col: str
) -> Tuple[Dict[Tuple[str, int], Dict[str, str]], List[int]]:
    """
    Simulate building support map to identify which indices WOULD be used as support,
    so we can exclude them and ensure the eval set is identical to the few-shot version.
    """
    support_map: Dict[Tuple[str, int], Dict[str, str]] = {}
    used: List[int] = []
    for idx, row in df.iterrows():
        code = _zfill3(row.get(code_col, ""))
        b = bucket_key(level, code)
        if not b or b in support_map:
            continue
        support_map[b] = {"dummy": "dummy"}
        used.append(idx)
    return support_map, used


def create_wos_prompt_zeroshot(keywords: str, abstract: str, candidates: Dict[str, Dict[str, str]]) -> str:
    cand_text_list = []
    for info in candidates.values():
        label = info.get("label", "")
        defn = info.get("definition", "")
        if defn:
            cand_text_list.append(f"- {label}: {defn}")
        else:
            cand_text_list.append(f"- {label}")
    cand_text = "\n".join(cand_text_list)
    return f"""You are a Dewey Decimal classifier. Choose the single best category NAME from the allowed list.

Allowed categories:
{cand_text}

Now classify the following text. Return up to 5 category names (from the allowed list) with confidence (e.g., "Science: 72%"), one per line.
Keywords: {keywords}
Abstract: {abstract}
Answer with names only (with percentages); do NOT output numeric codes."""


def run_eval(
    input_csv: str,
    level: int,
    mapping_csv: str,
    output_csv: str,
    sample_size: int = 500,
    random_seed: int = 42,
    provider: str = "deepseek-chat",
    model: Optional[str] = None,
    max_workers: int = 20,
) -> Tuple[List[Dict], Dict[str, float]]:
    df = pd.read_csv(input_csv)
    required = {"keywords", "Abstract"}
    if not required.issubset(df.columns):
        raise ValueError(f"Input missing required columns: {required}")

    maps = load_mapping(mapping_csv)
    client = get_client(provider, model=model)

    # determine code column
    code_col = {1: "DDC_L1_code", 2: "DDC_L2_code"}[level]
    if code_col not in df.columns:
        fallback = {1: "DDC L1", 2: "DDC L2"}[level]
        if fallback not in df.columns:
            raise ValueError(f"Input missing code columns: {code_col} (or fallback {fallback})")
        code_col = fallback

    # Identifying support indices to EXCLUDE them (to align with few-shot eval set)
    _, used_idx = build_support_map_dummy(df, level, code_col)
    
    eval_df = df.drop(index=used_idx).copy()
    eval_df = eval_df.sample(n=min(sample_size, len(eval_df)), random_state=random_seed).reset_index(drop=True)
    print(f"Evaluating {len(eval_df)} samples (same seed/supports as few-shot).")

    def classify_one(idx_row):
        idx, row = idx_row
        true_code = _zfill3(row.get(code_col, ""))
        if not true_code:
            return None
        candidates = build_candidates_for_code(level, true_code, maps)
        if not candidates:
            return None
        
        # Zero-shot prompt
        prompt = create_wos_prompt_zeroshot(str(row.get("keywords", "")), str(row.get("Abstract", "")), candidates)

        pred = client.classify_text_with_confidence(prompt)
        pred_with_codes = []
        pred_code = None
        for cat, conf in pred:
            code = parse_code(cat, candidates)
            pred_with_codes.append((cat, conf, code))
            if pred_code is None and code:
                pred_code = code

        return {
            "index": idx,
            "keywords": row.get("keywords", ""),
            "abstract": row.get("Abstract", ""),
            f"true_level{level}_code": true_code,
            "predicted_code": pred_code,
            "predicted_name": candidates.get(pred_code, "") if pred_code else "",
            "predicted_categories": pred,
            "predicted_categories_with_codes": pred_with_codes,
            "candidate_count": len(candidates),
            "provider": provider,
            "model": model or "",
        }

    results: List[Dict] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(classify_one, item): item[0] for item in eval_df.reset_index().iterrows()}
        for fut in as_completed(futures):
            res = fut.result()
            if res:
                results.append(res)
                idx = res["index"]
                print(f"[{idx+1}/{len(eval_df)}] predicted {res.get('predicted_code') or 'N/A'} for WOS level{level}")

    save_results(results, output_csv)
    metrics = compute_topk(results, level)
    return results, metrics


def main():
    default_input = os.path.join(os.path.dirname(os.path.dirname(__file__)), "datasets", "WOS", "WOS_46985_Dataset_rev_26Nov25_ddc12_mapped.csv")
    default_input = os.path.join(os.path.dirname(os.path.dirname(__file__)), "datasets", "WOS", "WOS_46985_Dataset_rev_26Nov25_ddc12_mapped.csv")
    default_mapping = os.path.join(os.path.dirname(os.path.dirname(__file__)), "DDClabel_deepseek_hierarchical.csv")

    parser = argparse.ArgumentParser(description="WOS DDC Zero-shot evaluation (Aligned with Few-shot set).")
    parser.add_argument("--level", type=int, choices=[1, 2], required=True)
    parser.add_argument("--input", type=str, default=default_input)
    parser.add_argument("--mapping_csv", type=str, default=default_mapping)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--sample_size", type=int, default=500)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--provider", type=str, default="deepseek-chat")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--max_workers", type=int, default=20)
    args = parser.parse_args()

    input_path = args.input if os.path.isabs(args.input) else os.path.abspath(args.input)
    output_path = args.output if os.path.isabs(args.output) else os.path.abspath(args.output)

    _, metrics = run_eval(
        input_csv=input_path,
        level=args.level,
        mapping_csv=args.mapping_csv,
        output_csv=output_path,
        sample_size=args.sample_size,
        random_seed=args.random_seed,
        provider=args.provider,
        model=args.model,
        max_workers=args.max_workers,
    )
    counts = metrics["counts"]
    total = counts["total"] or 1
    md_path = output_path.replace(".csv", ".md")
    md_lines = [
        f"# WOS DDC Level-{args.level} Zero-shot Summary",
        f"- Samples: {counts['total']}",
        f"- Top-1: {metrics['top1']:.2%} ({counts['top1']}/{total})",
        f"- Top-3: {metrics['top3']:.2%} ({counts['top3']}/{total})",
        f"- Top-5: {metrics['top5']:.2%} ({counts['top5']}/{total})",
        f"- Output CSV: {output_path}",
    ]
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    print("Summary written to", md_path)


if __name__ == "__main__":
    main()
