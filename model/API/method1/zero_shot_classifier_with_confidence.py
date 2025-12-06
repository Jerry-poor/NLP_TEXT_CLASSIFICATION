#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Zero-shot text classification script (with confidence)
Use Deepseek API for zero-shot text classification, output 5 most likely categories and confidence scores
"""

import sys
import os
from typing import Optional
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from llm_classifier_utils import (
    get_client,
    load_dataset,
    save_results,
    get_ddc_categories,
    calculate_top_k_accuracy,
)

def create_zero_shot_prompt_with_confidence(title: str, abstract: str) -> str:
    """Create zero-shot classification prompt (with confidence)"""
    
    ddc_categories = get_ddc_categories()
    categories_list = "\n".join([f"{name}" for code, name in ddc_categories.items()])
    
    prompt = f"""Please classify the following text content into the most appropriate Dewey Decimal Classification (DDC) first-level category.

Available DDC first-level categories:
{categories_list}

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

def zero_shot_classify_dataset_with_confidence(
    input_file: str,
    output_file: str,
    sample_size: int = 10,
    provider: str = "deepseek-chat",
    model: Optional[str] = None,
):
    """Perform zero-shot classification on dataset (with confidence)."""
    
    print("Loading dataset...")
    df = load_dataset(input_file)
    
    # Check if required columns exist
    required_columns = ['Title', 'Abstract']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Dataset missing required column: {col}")
    
    # Limit sample size (optional)
    if sample_size > 0:
        df = df.head(sample_size)
    
    print(f"Starting zero-shot classification (with confidence) for {len(df)} samples...")

    client = get_client(provider, model=model)
    results = []
    
    for idx, row in df.iterrows():
        title = row['Title']
        abstract = row['Abstract']
        
        print(f"\nProcessing sample {idx + 1}/{len(df)}:")
        print(f"Title: {title[:100]}...")
        
        prompt = create_zero_shot_prompt_with_confidence(title, abstract)
        predicted_categories = client.classify_text_with_confidence(prompt)
        
        # Reverse lookup codes
        predicted_with_codes = []
        for category, confidence in predicted_categories:
            code = None
            for c, name in get_ddc_categories().items():
                if name.lower() == category.lower():
                    code = c
                    break
            predicted_with_codes.append((category, confidence, code))
        
        result = {
            'index': idx,
            'title': title,
            'abstract': abstract,
            'predicted_categories': predicted_categories,
            'predicted_categories_with_codes': predicted_with_codes,
            'provider': provider,
            'model': model or '',
        }
        
        # If original data has DDC labels, save for comparison
        if 'DDC_Level1' in df.columns:
            true_category_code = str(row['DDC_Level1'])
            true_category_name = get_ddc_categories().get(true_category_code, 'Unknown')
            result['true_ddc_code'] = true_category_code
            result['true_ddc_name'] = true_category_name
        
        results.append(result)
        
        # Print prediction results
        print("Prediction results (top 5 most likely categories):")
        for i, (category, confidence) in enumerate(predicted_categories, 1):
            print(f"  {i}. {category}: {confidence:.1%}")
    
    # Save results
    save_results(results, output_file)
    
    # Calculate top-k accuracy (if true labels exist)
    if 'true_ddc_code' in results[0]:
        accuracy_metrics = calculate_top_k_accuracy(results)
        print(f"\n=== Accuracy Statistics ===")
        print(f"Top-1 Accuracy: {accuracy_metrics['top1']:.2%}")
        print(f"Top-3 Accuracy: {accuracy_metrics['top3']:.2%}")
        print(f"Top-5 Accuracy: {accuracy_metrics['top5']:.2%}")
    
    return results

if __name__ == "__main__":
    # Default parameters
    input_file = "AG_news_test_with_DDC_Hierarchy_mod_26Nov25_english_mapped.csv"
    output_file = "zero_shot_classification_with_confidence_results.csv"
    sample_size = 10  # Test sample size, set to 0 for all data
    
    print("=== Zero-shot Text Classification (with Confidence) ===")
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    print(f"Sample size: {sample_size if sample_size > 0 else 'all'}")
    
    try:
        results = zero_shot_classify_dataset_with_confidence(input_file, output_file, sample_size)
        print("\n✅ Zero-shot classification (with confidence) completed!")
    except Exception as e:
        print(f"❌ Classification failed: {e}")
        sys.exit(1)
