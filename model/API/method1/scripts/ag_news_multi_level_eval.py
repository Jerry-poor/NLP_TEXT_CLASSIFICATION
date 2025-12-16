#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate LLM classification on AG_news DDC dataset for level 1/2/3 with bucketed one-shot support."""

import argparse
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import pandas as pd

# Import from method1 package directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
import sys

sys.path.append(SCRIPT_DIR)

from llm_classifier_utils import get_client, save_results  # type: ignore
from multi_level_few_shot import (  # type: ignore
    load_mapping,
    build_candidates_for_code,
    create_prompt,
    parse_code,
    compute_topk,
)


def _zfill3(val: str) -> str:
    return str(val).strip().zfill(3)


def normalize_level3_code(val: str) -> str:
    """Normalize dataset Level3 value like '570-579' or '004-006' or '330' to a 3-digit code (first number)."""
    s = str(val).strip()
    m = re.search(r"(\d{1,3})", s)
    if not m:
        return ""
    return _zfill3(m.group(1))


def bucket_key(level: int, code: str) -> Optional[str]:
    code = _zfill3(code)
    if level == 1:
        return code[0]
    if level == 2:
        return code[0]
    if level == 3:
        return code[:2]
    return None


def build_support_map(df: pd.DataFrame, level: int) -> Tuple[Dict[str, Dict[str, str]], List[int]]:
    """Pick 1 support sample per bucket; return mapping bucket->support and row indices used."""
    code_col = {1: "DDC_Level1", 2: "DDC_Level2", 3: "DDC_Level3"}[level]
    support_map: Dict[str, Dict[str, str]] = {}
    used: List[int] = []
    for idx, row in df.iterrows():
        if level == 3:
            code = normalize_level3_code(row[code_col])
        else:
            code = _zfill3(row[code_col])
        b = bucket_key(level, code)
        if not b or b in support_map:
            continue
        support_map[b] = {"title": row["Title"], "abstract": row["Abstract"], "code": code}
        used.append(idx)
    return support_map, used


def run_eval(
    input_csv: str,
    level: int,
    mapping_csv: str,
    output_csv: str,
    sample_size: int = 500,
    random_seed: int = 42,
    provider: str = "chatgpt",
    model: Optional[str] = None,
    max_workers: int = 20,
) -> Tuple[List[Dict], Dict[str, float]]:
    df = pd.read_csv(input_csv)
    required = {"Title", "Abstract", "DDC_Level1", "DDC_Level2", "DDC_Level3"}
    if not required.issubset(df.columns):
        raise ValueError(f"Input missing required columns: {required}")

    maps = load_mapping(mapping_csv)
    client = get_client(provider, model=model)

    support_map, used_idx = build_support_map(df, level)
    if not support_map:
        raise ValueError("No support samples could be constructed.")
    default_support = next(iter(support_map.values()))

    eval_df = df.drop(index=used_idx).copy()
    eval_df = eval_df.sample(n=min(sample_size, len(eval_df)), random_state=random_seed).reset_index(drop=True)

    code_col = {1: "DDC_Level1", 2: "DDC_Level2", 3: "DDC_Level3"}[level]

    def classify_one(idx_row):
        idx, row = idx_row
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

    results: List[Dict] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(classify_one, item): item[0] for item in eval_df.reset_index().iterrows()}
        for fut in as_completed(futures):
            res = fut.result()
            if res:
                results.append(res)

    save_results(results, output_csv)
    metrics = compute_topk(results, level)
    return results, metrics


def main():
    default_input = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "datasets",
        "AG_news_test_with_DDC_Hierarchy_mod_26Nov25_english_mapped.csv",
    )
    default_mapping = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "dewey_decimal_unique.csv")

    parser = argparse.ArgumentParser(description="AG_news DDC evaluation for level 1/2/3 with bucketed one-shot.")
    parser.add_argument("--level", type=int, choices=[1, 2, 3], required=True)
    parser.add_argument("--input", type=str, default=default_input)
    parser.add_argument("--mapping_csv", type=str, default=default_mapping)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--sample_size", type=int, default=500)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--provider", type=str, default="chatgpt")
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
        f"# AG_news Level-{args.level} Summary",
        f"- Samples: {counts['total']}",
        f"- Top-1: {metrics['top1']:.2%} ({counts['top1']}/{total})",
        f"- Top-3: {metrics['top3']:.2%} ({counts['top3']}/{total})",
        f"- Top-5: {metrics['top5']:.2%} ({counts['top5']}/{total})",
        f"- Output CSV: {output_path}",
        "",
        "_Note: Level3 ground-truth values like '570-579' are normalized to the first 3-digit code (e.g., 570) for bucketing and scoring._",
    ]
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    print("Summary written to", md_path)


if __name__ == "__main__":
    main()

