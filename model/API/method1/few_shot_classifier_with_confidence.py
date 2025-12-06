#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Few-shot text classification script (with confidence)
Use Deepseek API for few-shot text classification, output 5 most likely categories and confidence scores
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

def create_few_shot_prompt_with_confidence(title: str, abstract: str, examples: list) -> str:
    """Create few-shot classification prompt (with confidence)"""
    
    ddc_categories = get_ddc_categories()
    categories_list = "\n".join([f"{name}" for code, name in ddc_categories.items()])
    
    # Build examples section
    examples_text = ""
    for i, example in enumerate(examples, 1):
        examples_text += f"\nExample {i}:"
        examples_text += f"\nText title: {example['title']}"
        examples_text += f"\nText abstract: {example['abstract']}"
        examples_text += f"\nClassification result: {example['category']}\n"
    
    prompt = f"""Please classify the following text content into the most appropriate Dewey Decimal Classification (DDC) first-level category.

Available DDC first-level categories:
{categories_list}

First, here are some classification examples:{examples_text}

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

def get_few_shot_examples(df, num_examples: int = 3):
    """Get multiple example samples from the dataset"""
    
    ddc_categories = get_ddc_categories()
    
    # Get samples from different categories
    available_categories = df['DDC_Level1'].unique()
    
    # If insufficient categories, reuse some categories
    if len(available_categories) < num_examples:
        selected_categories = list(available_categories) * (num_examples // len(available_categories))
        selected_categories += list(available_categories)[:num_examples % len(available_categories)]
    else:
        selected_categories = random.sample(list(available_categories), num_examples)
    
    examples = []
    for category_code in selected_categories:
        category_samples = df[df['DDC_Level1'] == category_code]
        if len(category_samples) > 0:
            example = category_samples.sample(1).iloc[0]
            category_name = ddc_categories.get(str(category_code), 'Unknown')
            examples.append({
                'title': example['Title'],
                'abstract': example['Abstract'],
                'category': category_name,
                'code': category_code,
                'row_index': example.name
            })
    
    return examples

def few_shot_classify_dataset_with_confidence(
    input_file: str,
    output_file: str,
    sample_size: int = 10,
    num_examples: int = 3,
    support_examples: list = None,
    provider: str = "deepseek-chat",
    model: Optional[str] = None,
):
    """Perform few-shot classification on dataset (with confidence)."""
    
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
    
    print(f"Starting few-shot classification (with confidence) for {len(df)} samples...")
    
    # Get example samples (use provided support examples to keep eval size unchanged)
    if support_examples is not None:
        examples = support_examples[:num_examples]
        df_eval = df.reset_index(drop=True)
        print(f"Using provided {len(examples)} support examples, evaluating {len(df_eval)} samples.")
    else:
        examples = get_few_shot_examples(df, num_examples)
        # Remove support examples from evaluation set to avoid leakage
        support_indices = [ex['row_index'] for ex in examples if 'row_index' in ex]
        df_eval = df.drop(index=support_indices, errors='ignore').reset_index(drop=True)
        if len(df_eval) < len(df):
            print(f"Excluded {len(df) - len(df_eval)} support examples from evaluation, evaluating {len(df_eval)} samples.")
        else:
            print(f"Evaluating {len(df_eval)} samples.")
    
    print(f"Using {len(examples)} training examples:")
    for i, example in enumerate(examples, 1):
        print(f"  Example {i}: {example['category']} - {example['title'][:50]}...")
    
    client = get_client(provider, model=model)
    results = []
    
    for idx, row in df_eval.iterrows():
        title = row['Title']
        abstract = row['Abstract']
        
        print(f"\nProcessing sample {idx + 1}/{len(df_eval)}:")
        print(f"Title: {title[:100]}...")
        
        prompt = create_few_shot_prompt_with_confidence(title, abstract, examples)
        predicted_categories = client.classify_text_with_confidence(prompt)
        
        # Reverse lookup codes
        predicted_with_codes = []
        ddc_categories = get_ddc_categories()
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
            'num_examples': num_examples,
            'examples': examples,
            'provider': provider,
            'model': model or '',
        }
        
        results.append(result)
        
        # Print prediction results
        print("Prediction results (top 5 most likely categories):")
        for i, (category, confidence) in enumerate(predicted_categories, 1):
            print(f"  {i}. {category}: {confidence:.1%}")
    
    # Save results
    save_results(results, output_file)
    
    # Calculate top-k accuracy
    accuracy_metrics = calculate_top_k_accuracy(results)
    print(f"\n=== Accuracy Statistics ===")
    print(f"Training examples: {num_examples}")
    print(f"Top-1 Accuracy: {accuracy_metrics['top1']:.2%}")
    print(f"Top-3 Accuracy: {accuracy_metrics['top3']:.2%}")
    print(f"Top-5 Accuracy: {accuracy_metrics['top5']:.2%}")
    
    return results

if __name__ == "__main__":
    # Default parameters
    input_file = "AG_news_test_with_DDC_Hierarchy_mod_26Nov25_english_mapped.csv"
    output_file = "few_shot_classification_with_confidence_results.csv"
    sample_size = 10  # Test sample size, set to 0 for all data
    num_examples = 3  # Training examples count
    
    print("=== Few-shot Text Classification (with Confidence) ===")
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    print(f"Sample size: {sample_size if sample_size > 0 else 'all'}")
    print(f"Training examples: {num_examples}")
    
    try:
        results = few_shot_classify_dataset_with_confidence(input_file, output_file, sample_size, num_examples)
        print("\n✅ Few-shot classification (with confidence) completed!")
    except Exception as e:
        print(f"❌ Classification failed: {e}")
        sys.exit(1)
