#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Few-shot classifier for separate Level1/2/3 datasets with fixed 3-shot supports."""

import argparse
import os
import re
from typing import Dict, List, Optional, Tuple

import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.append(os.path.join(os.path.dirname(SCRIPT_DIR), "utils"))

from llm_classifier_utils import get_client, save_results


def load_mapping(dewey_csv: str) -> Tuple[Dict[str, Dict[str, str]], Dict[str, Dict[str, str]], Dict[str, Dict[str, str]]]:
    df = pd.read_csv(dewey_csv)
    # Check if this is the new hierarchical format with "Sequence" and "definition"
    if "Sequence" in df.columns and "Label" in df.columns:
        df["code"] = df["Sequence"].astype(str).str.zfill(3)
        # Create rich mapping: code -> {label, definition}
        # Filter by Level
        lvl1_df = df[df["Level"] == 1]
        lvl2_df = df[df["Level"] == 2]
        lvl3_df = df[df["Level"] == 3]

        def make_map(sub_df):
            return {
                row["code"]: {
                    "label": str(row["Label"]).strip(),
                    "definition": str(row["definition"]).strip() if pd.notna(row.get("definition")) else ""
                }
                for _, row in sub_df.iterrows()
            }

        lvl1 = make_map(lvl1_df)
        lvl2 = make_map(lvl2_df)
        lvl3 = make_map(lvl3_df)
        return lvl1, lvl2, lvl3
    
    # Fallback to old format
    df["code"] = df["ddc_number"].astype(str).str.zfill(3)
    # Wrap old string names in dict structure for compatibility
    lvl1 = {k: {"label": v, "definition": ""} for k, v in zip(df["code"], df["level1"])}
    lvl2 = {k: {"label": v, "definition": ""} for k, v in zip(df["code"], df["level2"])}
    lvl3 = {k: {"label": v, "definition": ""} for k, v in zip(df["code"], df["level3"])}
    return lvl1, lvl2, lvl3


def zfill_code(val: str, width: int = 3) -> str:
    return str(val).strip().zfill(width)


def pick_candidates(codes: List[str], name_map: Dict[str, Dict[str, str]]) -> Dict[str, Dict[str, str]]:
    uniq = sorted({zfill_code(c) for c in codes})
    return {c: name_map.get(c, {"label": f"Unknown code {c}", "definition": ""}) for c in uniq}


def select_support(df: pd.DataFrame, code_col: str, k: int = 3, bucket_prefix: Optional[str] = None) -> List[Dict[str, str]]:
    supports = []
    subset = df
    if bucket_prefix is not None:
        subset = df[df[code_col].astype(str).str.zfill(3).str.startswith(bucket_prefix)]
        if subset.empty:
            subset = df
    for _, row in subset.head(k).iterrows():
        supports.append(
            {
                "title": row["title"],
                "abstract": row["abstract"],
                "code": zfill_code(row[code_col]),
            }
        )
    return supports


def get_bucket_key(level: int, code: str) -> Optional[Tuple[str, int]]:
    code = zfill_code(code)
    if level == 1:
        return (code[0], level)  # 0-9 buckets
    if level == 2:
        return (code[0], level)  # 0-9 hundreds bucket
    if level == 3:
        return (code[:2], level)  # e.g., 43*
    return None


def build_support_map(df: pd.DataFrame, code_col: str, level: int) -> Tuple[Dict[Tuple[str, int], Dict[str, str]], List[int]]:
    support_map: Dict[Tuple[str, int], Dict[str, str]] = {}
    used_indices: List[int] = []
    for idx, row in df.iterrows():
        code = zfill_code(row[code_col])
        bucket = get_bucket_key(level, code)
        if bucket is None or bucket in support_map:
            continue
        support_map[bucket] = {"title": row["title"], "abstract": row["abstract"], "code": code}
        used_indices.append(idx)
    return support_map, used_indices


def create_prompt(
    title: str,
    abstract: str,
    candidates: Dict[str, Dict[str, str]],
    supports: List[Dict[str, str]],
) -> str:
    # candidates is code -> {label, definition}
    cand_text_list = []
    for info in candidates.values():
        label = info.get("label", "")
        defn = info.get("definition", "")
        if defn:
            cand_text_list.append(f"- {label}: {defn}")
        else:
            cand_text_list.append(f"- {label}")
            
    cand_text_bullets = "\n".join(cand_text_list)
    support_text = ""
    for i, s in enumerate(supports, 1):
        support_text += (
            f"\nExample {i}:\n"
            f"Title: {s['title']}\n"
            f"Abstract: {s['abstract']}\n"
            f"Abstract: {s['abstract']}\n"
            f"Label: ### {candidates.get(s['code'], {}).get('label', '')} ###\n"
        )
    prompt = f"""You are a Dewey Decimal classifier. Choose the single best category NAME from the allowed list.

Allowed categories:
{cand_text_bullets}

Support example (one-shot):
{support_text}

Now classify the following text. Return up to 5 category names (from the allowed list) with confidence (e.g., "Science: 72%"), one per line. If fewer than 5 options make sense, return fewer (at least 1).
Text title: {title}
Text abstract: {abstract}
Answer with names only (with percentages); do NOT output numeric codes."""
    return prompt


def parse_code(text: str, candidates: Dict[str, str]) -> Optional[str]:
    cleaned = text.strip().lower()
    # drop any confidence suffix like ": 72%" or "- 50%"
    cleaned = re.sub(r"[:\\-]\\s*\\d+(?:\\.\\d+)?\\s*%?", "", cleaned)
    cleaned = cleaned.strip()
    for code, info in candidates.items():
        name = info.get("label", "")
        name_l = name.lower().strip()
        if cleaned == name_l or cleaned.startswith(name_l) or name_l in cleaned:
            return code
    return None


def build_candidates_for_code(level: int, code: str, maps: Tuple[Dict, Dict, Dict]) -> Dict[str, Dict[str, str]]:
    code = zfill_code(code)
    lvl1_map, lvl2_map, lvl3_map = maps
    result: Dict[str, str] = {}
    if level == 1:
        codes = ["000"] + [str(i * 100).zfill(3) for i in range(1, 10)]  # include 000 plus 100-900
        for c in codes:
            entry = lvl1_map.get(c)
            if entry and not entry["label"].lower().startswith("unknown"):
                result[c] = entry
        return result
    if level == 2:
        prefix = code[0]
        codes = [f"{prefix}{i}0" for i in range(0, 10)]  # 00-90 within prefix
        for c in codes:
            entry = lvl2_map.get(c)
            if entry and not entry["label"].lower().startswith("unknown"):
                result[c] = entry
        return result
    if level == 3:
        prefix = code[:2]
        codes = [f"{prefix}{i}" for i in range(0, 10)]  # 0-9 within prefix
        for c in codes:
            entry = lvl3_map.get(c)
            if entry and not entry["label"].lower().startswith("unknown"):
                result[c] = entry
        return result
    return result


def compute_topk(results: List[Dict], level: int) -> Dict[str, float]:
    counts = {"total": 0, "top1": 0, "top3": 0, "top5": 0}
    for r in results:
        true_code = zfill_code(r.get(f"true_level{level}_code", ""))
        preds = r.get("predicted_categories_with_codes") or []
        pred_codes = [c for _, _, c in preds if c]
        if not true_code or not pred_codes:
            continue
        counts["total"] += 1
        if pred_codes[0] == true_code:
            counts["top1"] += 1
        if true_code in pred_codes[:3]:
            counts["top3"] += 1
        if true_code in pred_codes[:5]:
            counts["top5"] += 1
    total = counts["total"] or 1
    return {
        "top1": counts["top1"] / total,
        "top3": counts["top3"] / total,
        "top5": counts["top5"] / total,
        "counts": counts,
    }


def run_level(
    data_path: str,
    level: int,
    mapping_csv: str,
    sample_size: int,
    provider: str,
    model: Optional[str],
    output_path: str,
    max_workers: int = 5,
) -> Tuple[List[Dict], Dict[str, float]]:
    df = pd.read_excel(data_path)
    code_col = {1: "ddc_l1", 2: "ddc_l2", 3: "ddc_l3"}[level]
    required = {code_col, "title", "abstract"}
    if not required.issubset(df.columns):
        raise ValueError(f"Dataset {data_path} missing columns {required}")
    # load maps once
    maps = load_mapping(mapping_csv)

    support_map, used_idx = build_support_map(df, code_col, level)
    if not support_map:
        raise ValueError("No support samples available for any bucket.")
    default_support = next(iter(support_map.values()))
    eval_df = df.drop(index=used_idx).copy()
    if sample_size > 0:
        eval_df = eval_df.sample(n=min(sample_size, len(eval_df)), random_state=42)

    client = get_client(provider, model=model)

    lvl_maps = {1: maps[0], 2: maps[1], 3: maps[2]}

    def classify_one(idx_row):
        idx, row = idx_row
        true_code = zfill_code(row[code_col])
        candidates = build_candidates_for_code(level, true_code, maps)
        if not candidates:
            return None
        bucket_key = get_bucket_key(level, true_code)
        support = support_map.get(bucket_key, default_support)
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
            f"true_level{level}_name": lvl_maps[level].get(true_code, ""),
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
                print(f"[{idx+1}/{len(eval_df)}] predicted {res.get('predicted_code') or 'N/A'} for level{level}")

    save_results(results, output_path)
    metrics = compute_topk(results, level)
    md_path = output_path.replace(".csv", ".md")
    counts = metrics["counts"]
    total = counts["total"] or 1
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(
            "\n".join(
                [
                    f"# Level-{level} Few-shot Summary",
                    f"- Samples: {counts['total']}",
                    f"- Top-1: {metrics['top1']:.2%} ({counts['top1']}/{total})",
                    f"- Top-3: {metrics['top3']:.2%} ({counts['top3']}/{total})",
                    f"- Top-5: {metrics['top5']:.2%} ({counts['top5']}/{total})",
                    f"- Output CSV: {output_path}",
                ]
            )
        )
    print("Summary written to", md_path)
    return results, metrics


def main():
    parser = argparse.ArgumentParser(description="Few-shot classifier for Level1/2/3 datasets with fixed 3-shot supports")
    parser.add_argument("--level", type=int, choices=[1, 2, 3], required=True, help="DDC level to classify")
    parser.add_argument("--input", type=str, required=True, help="Input dataset path (xlsx)")
    parser.add_argument("--mapping_csv", type=str, default=os.path.join(os.path.dirname(os.path.dirname(__file__)), "dewey_decimal_unique.csv"))
    parser.add_argument("--output", type=str, required=True, help="Output CSV path")
    parser.add_argument("--sample_size", type=int, default=500, help="Max eval samples (after removing 3-shot supports)")
    parser.add_argument("--provider", type=str, default="deepseek-chat")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--max_workers", type=int, default=5, help="Concurrency for classification requests")
    args = parser.parse_args()

    input_path = args.input if os.path.isabs(args.input) else os.path.abspath(args.input)
    output_path = args.output if os.path.isabs(args.output) else os.path.abspath(args.output)
    run_level(
        data_path=input_path,
        level=args.level,
        mapping_csv=args.mapping_csv,
        sample_size=args.sample_size,
        provider=args.provider,
        model=args.model,
        output_path=output_path,
        max_workers=args.max_workers,
    )


if __name__ == "__main__":
    main()
