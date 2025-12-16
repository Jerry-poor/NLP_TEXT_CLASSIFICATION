#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""3-shot level-3 DDC classification with constrained candidates per parent Level2."""

import argparse
import os
import sys
import re
from typing import Dict, List, Tuple, Optional

import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from llm_classifier_utils import (  # type: ignore
    get_client,
    save_results,
)


def load_dewey_mapping(csv_path: str) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, str]]:
    """Load ddc_number -> (level1, level2, level3) names."""
    df = pd.read_csv(csv_path)
    if not {"ddc_number", "level1", "level2", "level3"}.issubset(df.columns):
        raise ValueError("dewey mapping must contain ddc_number, level1, level2, level3 columns")
    df["code"] = df["ddc_number"].astype(str).str.zfill(3)
    lvl1 = dict(zip(df["code"], df["level1"]))
    lvl2 = dict(zip(df["code"], df["level2"]))
    lvl3 = dict(zip(df["code"], df["level3"]))
    return lvl1, lvl2, lvl3


def build_candidates(level2_code: str, level3_map: Dict[str, str]) -> Dict[str, str]:
    """Return candidate level3 codes/names within the same Level2 parent (e.g., 33* for 330)."""
    parent = str(level2_code).zfill(3)
    # use first two digits to cover 00x/33x buckets
    prefix = parent[:2]
    candidates = {c: n for c, n in level3_map.items() if c.startswith(prefix)}
    if candidates:
        return candidates
    # fallback: use first digit (Level1 bucket)
    prefix = parent[:1]
    candidates = {c: n for c, n in level3_map.items() if c.startswith(prefix)}
    return candidates


def select_support(df: pd.DataFrame, parent_code: str, k: int = 3) -> List[Dict[str, str]]:
    """Select up to k support examples from the same Level2 parent."""
    parent = str(parent_code)
    subset = df[df["DDC_Level2"].astype(str) == parent]
    supports = []
    for _, row in subset.head(k).iterrows():
        supports.append(
            {
                "title": row["Title"],
                "abstract": row["Abstract"],
                "code": str(row["DDC_Level3"]).zfill(3),
            }
        )
    return supports


def create_prompt(
    title: str,
    abstract: str,
    candidates: Dict[str, str],
    supports: List[Dict[str, str]],
) -> str:
    """Build few-shot prompt: list candidates and 3 support examples."""
    cand_text = "\n".join([f"- code: {c}, name: {n}" for c, n in candidates.items()])
    support_text = ""
    for i, s in enumerate(supports, 1):
        name = candidates.get(s["code"], "")
        support_text += (
            f"\nExample {i}:\n"
            f"Title: {s['title']}\n"
            f"Abstract: {s['abstract']}\n"
            f"True code: {s['code']} ({name})\n"
        )

    prompt = f"""You are a Dewey Decimal Classification (DDC) level-3 classifier.
Pick the single best level-3 code from the allowed list below.

Allowed codes (level-3 within the same parent):
{cand_text}

Support examples (3-shot):
{support_text}

Now classify the following text. Return ONLY the most likely code with a confidence percentage (e.g., "331: 82%"). Optionally provide up to 3 codes total, sorted by confidence, one per line."""
    return prompt


def parse_predicted_code(category: str, candidates: Dict[str, str]) -> Optional[str]:
    """Extract code from model output token."""
    # if category is like "331" or "331: 80"
    m = re.search(r"(\d{3})", category)
    if m:
        code = m.group(1)
        if code in candidates:
            return code
    # try exact match on candidate names
    for code, name in candidates.items():
        if name.lower() == category.lower():
            return code
    return None


def classify_dataset(
    input_file: str,
    output_file: str,
    dewey_csv: str,
    provider: str = "deepseek-chat",
    model: Optional[str] = None,
) -> List[Dict]:
    """Run constrained 3-shot level3 classification."""
    df = pd.read_csv(input_file)
    required = {"DDC_Level1", "DDC_Level2", "DDC_Level3", "Title", "Abstract"}
    if not required.issubset(df.columns):
        raise ValueError(f"Input missing required columns: {required}")

    lvl1_map, lvl2_map, lvl3_map = load_dewey_mapping(dewey_csv)
    client = get_client(provider, model=model)

    results: List[Dict] = []
    for idx, row in df.iterrows():
        parent = str(row["DDC_Level2"])
        candidates = build_candidates(parent, lvl3_map)
        if not candidates:
            continue
        supports = select_support(df, parent, k=3)
        prompt = create_prompt(row["Title"], row["Abstract"], candidates, supports)
        predicted_categories = client.classify_text_with_confidence(prompt)

        # map predicted to codes
        predicted_with_codes = []
        predicted_code = None
        for cat, conf in predicted_categories:
            code = parse_predicted_code(cat, candidates)
            predicted_with_codes.append((cat, conf, code))
            if predicted_code is None and code:
                predicted_code = code

        true_c = str(row["DDC_Level3"]).zfill(3)
        result = {
            "index": idx,
            "title": row["Title"],
            "abstract": row["Abstract"],
            "true_level1_code": str(row["DDC_Level1"]),
            "true_level2_code": parent,
            "true_level3_code": true_c,
            "true_level1_name": lvl1_map.get(str(row["DDC_Level1"]).zfill(3), ""),
            "true_level2_name": lvl2_map.get(parent.zfill(3), ""),
            "true_level3_name": lvl3_map.get(true_c, ""),
            "predicted_code": predicted_code,
            "predicted_name": candidates.get(predicted_code, "") if predicted_code else "",
            "predicted_categories": predicted_categories,
            "predicted_categories_with_codes": predicted_with_codes,
            "candidate_count": len(candidates),
            "provider": provider,
            "model": model or "",
        }
        results.append(result)

        print(f"\n[{idx+1}/{len(df)}] Parent {parent}: predicted {predicted_code or 'N/A'}")

    save_results(results, output_file)
    return results


def compute_topk(results: List[Dict]) -> Dict[str, Dict[str, int]]:
    """Compute top-1/3/5 hit counts for level3 codes."""
    counts = {"total": 0, "top1_correct": 0, "top3_correct": 0, "top5_correct": 0}
    for r in results:
        true_code = str(r.get("true_level3_code") or "").zfill(3)
        preds = r.get("predicted_categories_with_codes") or []
        pred_codes = [c for _, _, c in preds if c]
        if not true_code or not pred_codes:
            continue
        counts["total"] += 1
        if pred_codes[0] == true_code:
            counts["top1_correct"] += 1
        if true_code in pred_codes[:3]:
            counts["top3_correct"] += 1
        if true_code in pred_codes[:5]:
            counts["top5_correct"] += 1
    return counts


def main():
    parser = argparse.ArgumentParser(description="3-shot level-3 DDC classification with constrained candidates")
    parser.add_argument("--input", type=str, required=True, help="Input CSV (sampled dataset)")
    parser.add_argument("--dewey_csv", type=str, default=os.path.join(os.path.dirname(os.path.dirname(__file__)), "dewey_decimal_unique.csv"))
    parser.add_argument("--output", type=str, default="level3_fewshot_results.csv", help="Output CSV")
    parser.add_argument("--provider", type=str, default="deepseek-chat")
    parser.add_argument("--model", type=str, default=None)
    args = parser.parse_args()

    input_path = args.input if os.path.isabs(args.input) else os.path.abspath(args.input)
    output_path = args.output if os.path.isabs(args.output) else os.path.abspath(args.output)
    results = classify_dataset(input_path, output_path, args.dewey_csv, provider=args.provider, model=args.model)

    counts = compute_topk(results)
    total = counts["total"] or 1
    summary_lines = [
        "# Level-3 Few-shot Summary",
        f"- Samples evaluated: {counts['total']}",
        f"- Top-1: {counts['top1_correct']}/{total} ({counts['top1_correct']/total:.2%})",
        f"- Top-3: {counts['top3_correct']}/{total} ({counts['top3_correct']/total:.2%})",
        f"- Top-5: {counts['top5_correct']}/{total} ({counts['top5_correct']/total:.2%})",
        f"- Output CSV: {output_path}",
    ]
    md_path = output_path.replace(".csv", ".md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))
    print("\nSummary written to:", md_path)


if __name__ == "__main__":
    main()
