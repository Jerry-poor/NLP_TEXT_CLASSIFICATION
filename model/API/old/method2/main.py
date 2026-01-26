#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Method2 entrypoint: same dataset/flow intent as method1, but prompt returns only category; confidence from token logprobs."""

import argparse
import os
import sys
from typing import Optional, List, Dict

import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "method1")))

from zero_shot_classifier import (  # type: ignore
    zero_shot_classify_dataset,
    one_shot_classify_dataset,
    few_shot_classify_dataset,
)
from llm_classifier_utils import (
    load_dataset,
    get_ddc_categories,
    calculate_top_k_accuracy,
)  # type: ignore


def infer_provider(model_name: Optional[str], explicit_provider: Optional[str]) -> str:
    if explicit_provider:
        return explicit_provider
    if not model_name:
        return "deepseek-chat"
    lowered = model_name.lower()
    if "gemini" in lowered:
        return "gemini"
    if "gpt" in lowered or "chatgpt" in lowered or "o3" in lowered or "gpt-" in lowered:
        return "chatgpt"
    if "deepseek" in lowered:
        return "deepseek-chat"
    return "deepseek-chat"


def main():
    default_input = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "datasets", "AG_news_test_with_DDC_Hierarchy_mod_26Nov25_english_mapped.csv")
    )
    default_output_dir = os.path.join(os.path.dirname(__file__), "classification_results")

    parser = argparse.ArgumentParser(description="Method2 text classification (logprob-based confidence)")
    parser.add_argument("--input", type=str, default=default_input, help="Input file path")
    parser.add_argument("--output_dir", type=str, default=default_output_dir, help="Base output directory path")
    parser.add_argument("--sample_size", type=int, default=100, help="Per-class sample size (after exclusions)")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed (parity with method1)")
    parser.add_argument("--exclude_first_n", type=int, default=5, help="Exclude first N rows for support reuse (parity)")
    parser.add_argument("--methods", type=str, default="zero", help="Methods to run: zero,one,few (comma separated)")
    parser.add_argument("--provider", type=str, default=None, help="LLM provider: deepseek, chatgpt, gemini")
    parser.add_argument("--model", type=str, default=None, help="Optional model name override for provider")
    parser.add_argument("--model_name", type=str, default=None, help="Model name used for folder naming")

    args = parser.parse_args()

    model_name = args.model_name or args.model or "deepseek-chat"
    provider = infer_provider(model_name, args.provider)

    input_path = args.input if os.path.isabs(args.input) else os.path.abspath(args.input)

    output_base_dir = args.output_dir
    if not os.path.isabs(output_base_dir):
        output_base_dir = os.path.join(os.path.dirname(__file__), output_base_dir)
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), model_name)
    os.makedirs(output_dir, exist_ok=True)

    # Sampling flow (aligned with method1 intent)
    df_full = load_dataset(input_path)
    required_columns = ["Title", "Abstract", "DDC_Level1"]
    for col in required_columns:
        if col not in df_full.columns:
            raise ValueError(f"Dataset missing required column: {col}")

    # Reserve first N rows for potential supports (parity), then per-class head sampling
    def get_sampled_data(df: pd.DataFrame) -> pd.DataFrame:
        available = df.iloc[args.exclude_first_n :].copy()
        parts: List[pd.DataFrame] = []
        for cls, group in available.groupby("DDC_Level1"):
            parts.append(group.head(args.sample_size))
        if not parts:
            raise ValueError("No data available for sampling after exclusions.")
        sampled = pd.concat(parts).sample(frac=1.0, random_state=args.random_seed).reset_index(drop=True)
        return sampled

    shared_sample = get_sampled_data(df_full)

    methods_to_run = [m.strip().lower() for m in args.methods.split(",") if m.strip()]
    support_pool = df_full.iloc[: args.exclude_first_n].copy()
    summaries = []

    if "zero" in methods_to_run:
        result_csv = os.path.join(output_dir, f"zero_shot_method2_seed{args.random_seed}_size{len(shared_sample)}.csv")
        results = zero_shot_classify_dataset(
            input_path,
            result_csv,
            sample_size=0,
            provider=provider,
            model=model_name,
            sampled_data=shared_sample,
            max_workers=10,
        )
        summaries.append(("zero", result_csv, results))

    if "one" in methods_to_run and not support_pool.empty:
        example = support_pool.iloc[0]
        support_example = {
            "title": example["Title"],
            "abstract": example["Abstract"],
            "category": get_ddc_categories().get(str(example["DDC_Level1"]), "Unknown"),
            "code": str(example["DDC_Level1"]),
        }
        result_csv = os.path.join(output_dir, f"one_shot_method2_seed{args.random_seed}_size{len(shared_sample)}.csv")
        results = one_shot_classify_dataset(
            shared_sample,
            support_example,
            provider=provider,
            model=model_name,
            max_workers=10,
        )
        save_results(results, result_csv)
        summaries.append(("one", result_csv, results))

    if "few" in methods_to_run and not support_pool.empty:
        num_examples = min(3, len(support_pool))
        support_examples = []
        for _, row in support_pool.head(num_examples).iterrows():
            support_examples.append(
                {
                    "title": row["Title"],
                    "abstract": row["Abstract"],
                    "category": get_ddc_categories().get(str(row["DDC_Level1"]), "Unknown"),
                    "code": str(row["DDC_Level1"]),
                }
            )
        result_csv = os.path.join(output_dir, f"few_shot_method2_seed{args.random_seed}_size{len(shared_sample)}.csv")
        results = few_shot_classify_dataset(
            shared_sample,
            support_examples,
            provider=provider,
            model=model_name,
            max_workers=10,
        )
        save_results(results, result_csv)
        summaries.append(("few", result_csv, results))

    for method, csv_path, res in summaries:
        accuracy_metrics = calculate_top_k_accuracy(
            [
                {
                    "true_ddc_name": r.get("true_ddc_name"),
                    # keep the full sorted list of (name, prob) so Top-3/5 use real probabilities
                    "predicted_categories": r.get("predicted_categories") or [],
                }
                for r in res
            ]
        )
        counts = accuracy_metrics.get("counts", {})
        total = counts.get("total", len(res))
        md_lines = [
            f"# Method2 {method}-shot Summary ({model_name})",
            "",
            f"- Provider: {provider}",
            f"- Model: {model_name}",
            f"- Samples: {len(res)}",
            f"- Random seed: {args.random_seed}",
            "",
            "## Accuracy",
            f"- Top-1: {accuracy_metrics['top1']:.2%} ({counts.get('top1_correct', 0)}/{total})",
            f"- Top-3: {accuracy_metrics['top3']:.2%} ({counts.get('top3_correct', 0)}/{total})",
            f"- Top-5: {accuracy_metrics['top5']:.2%} ({counts.get('top5_correct', 0)}/{total})",
            "",
            "_Note: Method2 uses the model-returned scores for all 10 categories; Top-3/5 derive from that full probability list._",
            "",
            "## Files",
            f"- Results CSV: `{csv_path}`",
        ]
        md_path = csv_path.replace(".csv", ".md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(md_lines))
        print(f"\nMethod2 {method}-shot completed.")
        print(f"Results saved under: {output_dir}")
        print(f"Summary markdown: {md_path}")


if __name__ == "__main__":
    main()
