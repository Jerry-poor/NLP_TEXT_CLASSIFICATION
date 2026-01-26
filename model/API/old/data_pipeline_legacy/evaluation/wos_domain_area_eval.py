#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate LLM classification on WOS dataset using:
1) Domain classification (top-level)
2) area classification (second-level, bucketed by oracle Domain)

This follows the "rules + LLM" setup:
- For area evaluation, Domain is treated as correct (oracle) and used to constrain candidates.
- Prompts expose only TEXT labels (Domain/area strings), no numeric codes.
"""

import argparse
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

# Import from method1 package directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
import sys

sys.path.append(SCRIPT_DIR)
# Add utils directory based on new structure
sys.path.append(os.path.join(os.path.dirname(SCRIPT_DIR), "utils"))

from llm_classifier_utils import get_client, save_results  # type: ignore


def _norm_text(s: str) -> str:
    s = str(s or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s


def _clean_model_line(s: str) -> str:
    s = _norm_text(s)
    s = re.sub(r"[:\-]\s*\d+(?:\.\d+)?\s*%?", "", s)
    s = s.strip().strip('"').strip("'").strip()
    return s


def build_candidates(values: Sequence[str]) -> List[str]:
    uniq: List[str] = []
    seen = set()
    for v in values:
        t = _norm_text(v)
        if not t:
            continue
        if t.lower().startswith("unknown"):
            continue
        if t in seen:
            continue
        seen.add(t)
        uniq.append(t)
    return uniq


def pick_support_per_bucket(
    df: pd.DataFrame, bucket_col: str, require_cols: Optional[Sequence[str]] = None
) -> Tuple[Dict[str, Dict[str, str]], List[int]]:
    """
    Pick 1 support row per bucket (first occurrence in file order).
    Return bucket->support row dict and used row indices.
    """
    support_map: Dict[str, Dict[str, str]] = {}
    used: List[int] = []
    for idx, row in df.iterrows():
        b = _norm_text(row.get(bucket_col, ""))
        if require_cols:
            if any(not _norm_text(row.get(col, "")) for col in require_cols):
                continue
        if not b or b in support_map:
            continue
        support_map[b] = {
            "domain": _norm_text(row.get("Domain", "")),
            "area": _norm_text(row.get("area", "")),
            "keywords": _norm_text(row.get("keywords", "")),
            "abstract": _norm_text(row.get("Abstract", "")),
        }
        used.append(idx)
    return support_map, used


def create_prompt(
    keywords: str,
    abstract: str,
    candidates: List[str],
    support: Dict[str, str],
    target: str,
) -> str:
    cand_lines = "\n".join([f"- {c}" for c in candidates])
    support_label = support["domain"] if target == "domain" else support["area"]
    prompt = f"""You are a text classification assistant.
You must choose category NAME(s) strictly from the allowed list below.

Allowed categories:
{cand_lines}

One support example:
Keywords: {support['keywords']}
Abstract: {support['abstract']}
True label: ### {support_label} ###

Now classify the following text.
Return up to 5 category names (from the allowed list) with confidence as percentages, one per line.
Format strictly as: Category Name: 72%
Do NOT output anything else.

Keywords: {keywords}
Abstract: {abstract}
"""
    return prompt


def match_to_allowed(pred_label: str, allowed: List[str]) -> Optional[str]:
    """
    Map a model-returned label string to one of allowed labels.
    Uses strict equality first, then containment.
    """
    cleaned = _clean_model_line(pred_label).lower()
    if not cleaned:
        return None
    allowed_l = [(a, a.lower()) for a in allowed]
    for a, al in allowed_l:
        if cleaned == al:
            return a
    for a, al in allowed_l:
        if cleaned.startswith(al) or al in cleaned:
            return a
    return None


def compute_topk(results: List[Dict], true_key: str, pred_key: str) -> Dict[str, float]:
    counts = {"total": 0, "top1": 0, "top3": 0, "top5": 0}
    for r in results:
        true_val = _norm_text(r.get(true_key, ""))
        preds = r.get(pred_key) or []
        pred_vals = [p for p in preds if p]
        if not true_val or not pred_vals:
            continue
        counts["total"] += 1
        if pred_vals[0] == true_val:
            counts["top1"] += 1
        if true_val in pred_vals[:3]:
            counts["top3"] += 1
        if true_val in pred_vals[:5]:
            counts["top5"] += 1
    total = counts["total"] or 1
    return {
        "top1": counts["top1"] / total,
        "top3": counts["top3"] / total,
        "top5": counts["top5"] / total,
        "counts": counts,
    }


def run_eval(
    input_csv: str,
    output_csv: str,
    target: str,
    sample_size: int = 500,
    random_seed: int = 42,
    provider: str = "deepseek-chat",
    model: Optional[str] = None,
    max_workers: int = 20,
    max_area_candidates: Optional[int] = None,
) -> Tuple[List[Dict], Dict[str, float]]:
    df = pd.read_csv(input_csv)
    required = {"Domain", "area", "keywords", "Abstract"}
    if not required.issubset(df.columns):
        raise ValueError(f"Input missing required columns: {required}")
    if target not in ("domain", "area"):
        raise ValueError("target must be 'domain' or 'area'")

    domains = build_candidates(df["Domain"].tolist())
    if not domains:
        raise ValueError("No Domain labels found.")

    support_bucket_col = "Domain"
    # For area evaluation, ensure the one-shot label (area) is present.
    support_requires = ("area",) if target == "area" else None
    support_map, used_idx = pick_support_per_bucket(df, support_bucket_col, require_cols=support_requires)
    if not support_map:
        raise ValueError("No support samples could be constructed.")
    default_support = next(iter(support_map.values()))

    eval_df = df.drop(index=used_idx).copy()
    eval_df = eval_df.sample(n=min(sample_size, len(eval_df)), random_state=random_seed).reset_index(drop=True)

    client = get_client(provider, model=model)

    def classify_one(idx_row):
        idx, row = idx_row
        true_domain = _norm_text(row.get("Domain", ""))
        true_area = _norm_text(row.get("area", ""))
        keywords = _norm_text(row.get("keywords", ""))
        abstract = _norm_text(row.get("Abstract", ""))

        if target == "domain":
            candidates = domains
            true_val = true_domain
            bucket = true_domain
        else:
            # oracle Domain used to constrain area candidates
            bucket = true_domain
            if not bucket:
                return None
            areas_in_domain = build_candidates(df.loc[df["Domain"] == bucket, "area"].tolist())
            if not areas_in_domain:
                return None
            if max_area_candidates and max_area_candidates > 0 and len(areas_in_domain) > max_area_candidates:
                # Keep the true label (if present) and fill the remainder by frequency within domain.
                vc = df.loc[df["Domain"] == bucket, "area"].value_counts()
                ordered = [a for a in vc.index.tolist() if _norm_text(a)]
                ordered = [a for a in ordered if not _norm_text(a).lower().startswith("unknown")]
                kept: List[str] = []
                if true_area:
                    kept.append(true_area)
                for a in ordered:
                    a = _norm_text(a)
                    if a and a not in kept:
                        kept.append(a)
                    if len(kept) >= max_area_candidates:
                        break
                candidates = kept
            else:
                candidates = areas_in_domain
            true_val = true_area

        if not true_val:
            return None

        support = support_map.get(bucket, default_support)
        prompt = create_prompt(keywords, abstract, candidates, support, target=target)
        pred = client.classify_text_with_confidence(prompt)

        allowed = candidates
        matched: List[Tuple[str, float, Optional[str]]] = []
        matched_labels: List[str] = []
        for label, conf in pred:
            m = match_to_allowed(label, allowed)
            matched.append((label, conf, m))
            if m and m not in matched_labels:
                matched_labels.append(m)

        return {
            "index": idx,
            "true_domain": true_domain,
            "true_area": true_area,
            "target": target,
            "true_label": true_val,
            "predicted_label": matched_labels[0] if matched_labels else None,
            "predicted_labels": matched_labels,
            "predicted_raw": pred,
            "predicted_raw_with_match": matched,
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
                print(f"[{idx+1}/{len(eval_df)}] predicted {res.get('predicted_label') or 'N/A'} for {target}")

    save_results(results, output_csv)
    metrics = compute_topk(results, true_key="true_label", pred_key="predicted_labels")
    return results, metrics


def main():
    default_input = os.path.join(os.path.dirname(os.path.dirname(__file__)), "datasets", "WOS", "WOS_46985_Dataset_rev_26Nov25.csv")
    parser = argparse.ArgumentParser(description="WOS Domain/area evaluation with bucketed one-shot support.")
    parser.add_argument("--target", type=str, choices=["domain", "area"], required=True, help="Classification target")
    parser.add_argument("--input", type=str, default=default_input)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--sample_size", type=int, default=500)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--provider", type=str, default="deepseek-chat")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--max_workers", type=int, default=20)
    parser.add_argument(
        "--max_area_candidates",
        type=int,
        default=None,
        help="Optional cap for area candidates per domain (keeps true label + frequent others). Default: no cap.",
    )
    args = parser.parse_args()

    input_path = args.input if os.path.isabs(args.input) else os.path.abspath(args.input)
    output_path = args.output if os.path.isabs(args.output) else os.path.abspath(args.output)

    _, metrics = run_eval(
        input_csv=input_path,
        output_csv=output_path,
        target=args.target,
        sample_size=args.sample_size,
        random_seed=args.random_seed,
        provider=args.provider,
        model=args.model,
        max_workers=args.max_workers,
        max_area_candidates=args.max_area_candidates,
    )

    counts = metrics["counts"]
    total = counts["total"] or 1
    md_path = output_path.replace(".csv", ".md")
    md_lines = [
        f"# WOS {args.target} Summary",
        f"- Samples: {counts['total']}",
        f"- Top-1: {metrics['top1']:.2%} ({counts['top1']}/{total})",
        f"- Top-3: {metrics['top3']:.2%} ({counts['top3']}/{total})",
        f"- Top-5: {metrics['top5']:.2%} ({counts['top5']}/{total})",
        f"- Output CSV: {output_path}",
        "",
        "_Note: For area evaluation, Domain is treated as oracle-correct and used to constrain the candidate set._",
    ]
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    print("Summary written to", md_path)


if __name__ == "__main__":
    main()
