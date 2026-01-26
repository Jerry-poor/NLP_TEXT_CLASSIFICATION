#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
One-shot text classification script (with confidence)
Use Deepseek API for one-shot text classification, output 5 most likely categories and confidence scores
"""

import sys
import os
import random
from typing import Optional
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from llm_classifier_utils import (
    get_client,
    load_dataset,
    save_results,
    get_ddc_categories,
    calculate_top_k_accuracy,
)

def create_one_shot_prompt_with_confidence(title: str, abstract: str, example_title: str, example_abstract: str, example_category: str) -> str:
    """Create one-shot classification prompt (with confidence)"""
    
    ddc_categories = get_ddc_categories()
    categories_list = "\n".join([f"{name}" for code, name in ddc_categories.items()])
    
    prompt = f"""Please classify the following text content into the most appropriate Dewey Decimal Classification (DDC) first-level category.

Available DDC first-level categories:
{categories_list}

First, here is a classification example:
Text title: {example_title}
Text abstract: {example_abstract}
Classification result: {example_category}

Now, please classify the following text:
Text title: {title}
Text abstract: {abstract}

Please return the 5 most likely DDC category English names and their corresponding confidence scores (percentage).
Please output in the following format:
Category1: Confidence1%
Category2: Confidence2%
Category3: Confidence3%
Category4: Confidence4%
Category5: Confidence5%

Make sure the total confidence does not exceed 100%, and sort by confidence from high to low.

Classification result:"""
    
    return prompt

def get_example_sample(df, target_category_code: str = None):
    """Get an example sample from the dataset"""
    
    if target_category_code is not None:
        # Randomly select from specified category
        category_samples = df[df['DDC_Level1'] == target_category_code]
        if len(category_samples) > 0:
            example = category_samples.sample(1).iloc[0]
        else:
            # If no samples in specified category, random selection
            example = df.sample(1).iloc[0]
    else:
        # Random selection
        example = df.sample(1).iloc[0]
    
    return example

from concurrent.futures import ThreadPoolExecutor, as_completed

def one_shot_classify_dataset_with_confidence(
    input_file: str,
    output_file: str,
    sample_size: int = 10,
    example_category: str = None,
    example_support: dict = None,
    provider: str = "deepseek-chat",
    model: Optional[str] = None,
    concurrency: int = 1,
):
    """Perform one-shot classification on dataset (with confidence)."""
    
    print("Loading dataset...")
    df = load_dataset(input_file)
    
    # Check if required columns exist
    required_columns = ['Title', 'Abstract', 'DDC_Level1']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Dataset missing required column: {col}")
    
    # Limit sample size (optional)
    if sample_size > 0:
        df = df.head(sample_size)
    
    print(f"Starting one-shot classification (with confidence) for {len(df)} samples...")
    print(f"Concurrency level: {concurrency}")
    
    # Get example sample
    ddc_categories = get_ddc_categories()

    if example_support is not None:
        # Use provided support example (e.g., from reserved rows) and keep all eval rows
        example_category_code = example_support['code']
        example_category_name = ddc_categories.get(str(example_category_code), 'Unknown')
        example_sample = {
            'Title': example_support['title'],
            'Abstract': example_support['abstract']
        }
        df_eval = df.reset_index(drop=True)
        print(f"Example category (provided): {example_category_name} ({example_category_code})")
        print(f"Example title: {example_sample['Title'][:100]}...")
        print(f"Evaluating {len(df_eval)} samples (no exclusion).")
    else:
        if example_category is None:
            # Randomly select a category as example
            available_categories = df['DDC_Level1'].unique()
            example_category_code = random.choice(available_categories)
        else:
            # Use specified category
            example_category_code = example_category
        
        example_sample = get_example_sample(df, example_category_code)
        example_category_name = ddc_categories.get(str(example_category_code), 'Unknown')
        
        print(f"Example category: {example_category_name} ({example_category_code})")
        print(f"Example title: {example_sample['Title'][:100]}...")
        
        # Exclude the example sample from evaluation to avoid leakage
        df_eval = df.drop(index=example_sample.name, errors='ignore').reset_index(drop=True)
        if len(df_eval) < len(df):
            print(f"Excluded 1 example from evaluation, evaluating {len(df_eval)} samples.")
        else:
            print(f"Evaluating {len(df_eval)} samples.")
    
    client = get_client(provider, model=model)
    results = []

    def process_row(args):
        idx, row = args
        title = row['Title']
        abstract = row['Abstract']
        
        # print(f"\nProcessing sample {idx + 1}/{len(df_eval)} (Thread)...")
        
        prompt = create_one_shot_prompt_with_confidence(
            title, abstract, 
            example_sample['Title'], example_sample['Abstract'], example_category_name
        )
        predicted_categories = client.classify_text_with_confidence(prompt)
        
        # Reverse lookup codes
        predicted_with_codes = []
        for category, confidence in predicted_categories:
            code = None
            for c, name in ddc_categories.items():
                if name.lower() == category.lower():
                    code = c
                    break
            predicted_with_codes.append((category, confidence, code))
        
        true_category_code = str(row['DDC_Level1'])
        true_category_name = ddc_categories.get(true_category_code, 'Unknown')
        
        result = {
            'index': idx,
            'title': title,
            'abstract': abstract,
            'true_ddc_code': true_category_code,
            'true_ddc_name': true_category_name,
            'predicted_categories': predicted_categories,
            'predicted_categories_with_codes': predicted_with_codes,
            'example_category': example_category_name,
            'example_title': example_sample['Title'],
            'example_abstract': example_sample['Abstract'],
            'provider': provider,
            'model': model or '',
        }
        return result

    # Prepare arguments
    tasks = [(idx, row) for idx, row in df_eval.iterrows()]
    
    if concurrency > 1:
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            future_to_idx = {executor.submit(process_row, task): task[0] for task in tasks}
            processed_count = 0
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                processed_count += 1
                try:
                    res = future.result()
                    results.append(res)
                    print(f"[{processed_count}/{len(df_eval)}] Finished sample {idx + 1}")
                except Exception as exc:
                    print(f"Sample {idx + 1} generated an exception: {exc}")
    else:
        for task in tasks:
            print(f"Processing sample {task[0] + 1}/{len(df_eval)}:")
            results.append(process_row(task))
            
    # Sort results by index to maintain order
    results.sort(key=lambda x: x['index'])
    
    # Save results
    save_results(results, output_file)
    
    # Calculate top-k accuracy
    accuracy_metrics = calculate_top_k_accuracy(results)
    print(f"\n=== Accuracy Statistics ===")
    print(f"Example category: {example_category_name}")
    print(f"Top-1 Accuracy: {accuracy_metrics['top1']:.2%}")
    print(f"Top-3 Accuracy: {accuracy_metrics['top3']:.2%}")
    print(f"Top-5 Accuracy: {accuracy_metrics['top5']:.2%}")
    
    return results

if __name__ == "__main__":
    # Default parameters
    input_file = "AG_news_test_with_DDC_Hierarchy_mod_26Nov25_english_mapped.csv"
    output_file = "one_shot_classification_with_confidence_results.csv"
    sample_size = 10  # Test sample size, set to 0 for all data
    example_category = None  # Optional: specify example category code, e.g., '300' for Social sciences
    
    print("=== One-shot Text Classification (with Confidence) ===")
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    print(f"Sample size: {sample_size if sample_size > 0 else 'all'}")
    
    try:
        results = one_shot_classify_dataset_with_confidence(input_file, output_file, sample_size, example_category)
        print("\n✅ One-shot classification (with confidence) completed!")
    except Exception as e:
        print(f"❌ Classification failed: {e}")
        sys.exit(1)
