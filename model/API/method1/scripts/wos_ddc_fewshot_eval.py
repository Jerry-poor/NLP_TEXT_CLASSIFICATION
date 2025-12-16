#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WOS evaluation aligned with `ag_news_multi_level_eval.py` logic:
- bucketed one-shot support (1 per bucket, first occurrence)
- drop support rows from eval
- random sampling with seed
- constrained candidate set (<=10) derived from oracle true-code bucket
- metrics computed with `compute_topk` (skips missing/empty predictions)

Input dataset should contain:
- `keywords`, `Abstract`
- DDC code columns: preferred `DDC_L1_code` / `DDC_L2_code` (3-digit strings).
  If absent, falls back to `DDC L1` / `DDC L2` and zfill to 3 digits.
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


def bucket_key(level: int, code: str) -> Optional[str]:
    code = _zfill3(code)
    if not code:
        return None
    if level in (1, 2):
        return code[0]  # 0-9 buckets
    return None


def build_support_map(
    df: pd.DataFrame, level: int, code_col: str
) -> Tuple[Dict[str, Dict[str, str]], List[int]]:
    support_map: Dict[str, Dict[str, str]] = {}
    used: List[int] = []
    for idx, row in df.iterrows():
        code = _zfill3(row.get(code_col, ""))
        b = bucket_key(level, code)
        if not b or b in support_map:
            continue
        support_map[b] = {
            "keywords": str(row.get("keywords", "")),
            "abstract": str(row.get("Abstract", "")),
            "code": code,
        }
        used.append(idx)
    return support_map, used


def create_wos_prompt(keywords: str, abstract: str, candidates: Dict[str, str], support: Dict[str, str]) -> str:
    cand_text = "\n".join([f"- {name}" for name in candidates.values()])
    support_label = candidates.get(support["code"], "")
    return f"""You are a Dewey Decimal classifier. Choose the single best category NAME from the allowed list.

Allowed categories:
{cand_text}

Support example (one-shot):
Keywords: {support.get('keywords','')}
Abstract: {support.get('abstract','')}
Label: ### {support_label} ###

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

    support_map, used_idx = build_support_map(df, level, code_col)
    if not support_map:
        raise ValueError("No support samples could be constructed.")
    default_support = next(iter(support_map.values()))

    eval_df = df.drop(index=used_idx).copy()
    eval_df = eval_df.sample(n=min(sample_size, len(eval_df)), random_state=random_seed).reset_index(drop=True)

    def classify_one(idx_row):
        idx, row = idx_row
        true_code = _zfill3(row.get(code_col, ""))
        if not true_code:
            return None
        candidates = build_candidates_for_code(level, true_code, maps)
        if not candidates:
            return None
        b = bucket_key(level, true_code)
        support = support_map.get(b, default_support)
        prompt = create_wos_prompt(str(row.get("keywords", "")), str(row.get("Abstract", "")), candidates, support)

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
    default_input = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dataset", "WOS_46985_Dataset_rev_26Nov25_ddc12_mapped.csv")
    default_mapping = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dewey_decimal_unique.csv")

    parser = argparse.ArgumentParser(description="WOS DDC evaluation aligned with ag_news_multi_level_eval.py (level1/level2).")
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
        f"# WOS DDC Level-{args.level} Summary",
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

