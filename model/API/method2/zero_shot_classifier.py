#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Zero/One/Few-shot classifiers for method2: model returns per-class scores; confidence from softmax."""

import os
import sys
import json
import math
from typing import List, Tuple, Optional, Dict, Any, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "method1")))

from llm_classifier_utils import (  # type: ignore
    get_client,
    load_dataset,
    save_results,
    get_ddc_categories,
    calculate_top_k_accuracy,
)


def create_zero_shot_prompt(title: str, abstract: str) -> str:
    """Prompt asking for raw scores per DDC category; we will softmax locally."""
    ddc_categories = get_ddc_categories()
    categories_list = "\n".join([f'- code: {code}, name: "{name}"' for code, name in ddc_categories.items()])

    prompt = f"""You are a text classifier. For each Dewey Decimal Classification (DDC) first-level category below, produce an UNNORMALIZED score (any real number; do NOT convert to percent or probability). We will apply softmax ourselves.

Categories:
{categories_list}

Text title: {title}
Text abstract: {abstract}

Return JSON array, e.g.:
[
  {{"code": "500", "name": "Science", "score": 3.2}},
  ...
]
Include ALL 10 categories in the response, exactly one entry per category code."""
    return prompt


def softmax_from_scores(scores: Dict[str, float]) -> Dict[str, float]:
    max_s = max(scores.values()) if scores else 0.0
    exp_vals = {k: math.exp(v - max_s) for k, v in scores.items()}
    total = sum(exp_vals.values()) or 1.0
    return {k: v / total for k, v in exp_vals.items()}


def parse_scores(text: str) -> Dict[str, float]:
    """Parse JSON-ish output into {code: score}."""
    try:
        data = json.loads(text)
        scores = {}
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    code = str(item.get("code") or "").strip()
                    score = item.get("score")
                    if code and isinstance(score, (int, float)):
                        scores[code] = float(score)
        return scores
    except Exception:
        pass

    # fallback: regex for code/score pairs like 500: 3.2
    import re

    pattern = r'\"?(\d{1,3})\"?\s*[:=]\s*(-?\d+(?:\.\d+)?)'
    scores: Dict[str, float] = {}
    for code, val in re.findall(pattern, text):
        scores[str(code)] = float(val)
    return scores


def _classify_dataset_with_builder(
    df: pd.DataFrame,
    provider: str,
    model: Optional[str],
    prompt_builder: Callable[[pd.Series], str],
    max_workers: int = 10,
) -> List[Dict[str, Any]]:
    """Shared classification loop with thread pool."""
    ddc_categories = get_ddc_categories()
    results: List[Dict[str, Any]] = []

    def classify_row(idx_row):
        idx, row = idx_row
        prompt = prompt_builder(row)
        client = get_client(provider, model=model)
        label, _ = client.classify_text_with_logprobs(prompt, max_tokens=512)

        raw_scores = parse_scores(label) or {}
        for code in ddc_categories.keys():
            raw_scores.setdefault(code, 0.0)
        probs = softmax_from_scores(raw_scores) if raw_scores else {}

        best_code = max(probs, key=probs.get) if probs else None
        best_name = ddc_categories.get(best_code, "") if best_code else ""

        sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        predicted_categories = [(ddc_categories.get(code, code), p) for code, p in sorted_probs]

        result = {
            "index": idx,
            "title": row["Title"],
            "abstract": row["Abstract"],
            "predicted_category": best_name or label,
            "predicted_code": best_code,
            "predicted_prob": probs.get(best_code) if best_code else None,
            "predicted_categories": predicted_categories,
            "all_categories_with_probs": predicted_categories,
            "provider": provider,
            "model": model or "",
        }
        if "DDC_Level1" in row:
            true_category_code = str(row["DDC_Level1"])
            true_category_name = ddc_categories.get(true_category_code, "Unknown")
            result["true_ddc_code"] = true_category_code
            result["true_ddc_name"] = true_category_name
        return result

    printed_queue: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(classify_row, item): item[0] for item in df.reset_index().iterrows()}
        for fut in as_completed(futures):
            res = fut.result()
            results.append(res)
            printed_queue.append(res)

    for res in sorted(printed_queue, key=lambda x: x["index"]):
        prob_txt = f"{res.get('predicted_prob', 0):.3f}" if res.get("predicted_prob") is not None else "N/A"
        print(f"\nProcessing sample {res['index'] + 1}/{len(df)}:")
        print(f"Title: {res['title'][:100]}...")
        print(f"Predicted: {res['predicted_category']} (prob ~ {prob_txt})")

    return results


def zero_shot_classify_dataset(
    input_file: str,
    output_file: str,
    sample_size: int = 10,
    provider: str = "deepseek-chat",
    model: Optional[str] = None,
    sampled_data: Optional[pd.DataFrame] = None,
    max_workers: int = 10,
) -> List[dict]:
    """Zero-shot classification using method2 (logprob-based confidence)."""
    if sampled_data is None:
        print("Loading dataset...")
        df = load_dataset(input_file)
    else:
        df = sampled_data.copy()

    required_columns = ["Title", "Abstract"]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Dataset missing required column: {col}")

    if sample_size > 0 and sampled_data is None:
        df = df.head(sample_size)

    print(f"Starting zero-shot classification for {len(df)} samples (method2)...")

    def builder(row: pd.Series) -> str:
        return create_zero_shot_prompt(row["Title"], row["Abstract"])

    results = _classify_dataset_with_builder(df, provider, model, builder, max_workers=max_workers)
    save_results(results, output_file)
    return results


def create_one_shot_prompt(example: Dict[str, str], title: str, abstract: str) -> str:
    ddc_categories = get_ddc_categories()
    categories_list = "\n".join([f'- code: {code}, name: "{name}"' for code, name in ddc_categories.items()])
    prompt = f"""You are a text classifier. Given one labeled example, assign scores to EACH DDC first-level category below (unnormalized).

Reference example:
Title: {example['title']}
Abstract: {example['abstract']}
True category: {example['category']} (code: {example['code']})

Categories:
{categories_list}

Now score ALL 10 categories for the following text:
Title: {title}
Abstract: {abstract}

Return JSON array with ALL categories and a numeric "score" field per item."""
    return prompt


def one_shot_classify_dataset(
    df_eval: pd.DataFrame,
    support_example: Dict[str, str],
    provider: str,
    model: Optional[str],
    max_workers: int = 10,
) -> List[dict]:
    def builder(row: pd.Series) -> str:
        return create_one_shot_prompt(support_example, row["Title"], row["Abstract"])

    return _classify_dataset_with_builder(df_eval, provider, model, builder, max_workers=max_workers)


def create_few_shot_prompt(examples: List[Dict[str, str]], title: str, abstract: str) -> str:
    ddc_categories = get_ddc_categories()
    categories_list = "\n".join([f'- code: {code}, name: "{name}"' for code, name in ddc_categories.items()])
    examples_text = ""
    for i, ex in enumerate(examples, 1):
        examples_text += f"\nExample {i}:\nTitle: {ex['title']}\nAbstract: {ex['abstract']}\nCategory: {ex['category']} (code: {ex['code']})\n"
    prompt = f"""You are a text classifier. Given several labeled examples, assign scores to EACH DDC first-level category below (unnormalized).

Support examples:
{examples_text}

Categories:
{categories_list}

Now score ALL 10 categories for the following text:
Title: {title}
Abstract: {abstract}

Return JSON array with ALL categories and a numeric "score" field per item."""
    return prompt


def few_shot_classify_dataset(
    df_eval: pd.DataFrame,
    support_examples: List[Dict[str, str]],
    provider: str,
    model: Optional[str],
    max_workers: int = 10,
) -> List[dict]:
    def builder(row: pd.Series) -> str:
        return create_few_shot_prompt(support_examples, row["Title"], row["Abstract"])

    return _classify_dataset_with_builder(df_eval, provider, model, builder, max_workers=max_workers)


if __name__ == "__main__":
    input_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "datasets", "AG_news_test_with_DDC_Hierarchy_mod_26Nov25_english_mapped.csv"))
    output_file = "zero_shot_method2_results.csv"
    zero_shot_classify_dataset(input_file, output_file, sample_size=10)
