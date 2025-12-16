#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main classification script - Reuse three classification script modules
Support random sampling, resume from breakpoint, and result summary
"""

import sys
import os
import pandas as pd
import random
import json
import argparse
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Import functions from three classification scripts
from zero_shot_classifier_with_confidence import zero_shot_classify_dataset_with_confidence
from one_shot_classifier_with_confidence import one_shot_classify_dataset_with_confidence
from few_shot_classifier_with_confidence import few_shot_classify_dataset_with_confidence
from llm_classifier_utils import (
    load_dataset,
    get_ddc_categories,
    calculate_top_k_accuracy,
    calculate_top_k_accuracy_by_class,
    load_results,
)

class ClassificationManager:
    """Classification manager class, responsible for coordinating three classification scripts"""
    
    def __init__(self, input_file: str, output_dir: str = "classification_results", provider: str = "deepseek-chat", model: str = None):
        self.input_file = input_file
        self.provider = provider
        self.model = model
        self.output_dir = output_dir
        self.resume_file = os.path.join(output_dir, "resume_status.csv")
        self.summary_file = os.path.join(output_dir, "classification_summary.json")
        self.sample_cache_file: Optional[str] = None
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load dataset
        self.df = load_dataset(input_file)
        
        # Check required columns
        required_columns = ['Title', 'Abstract', 'DDC_Level1']
        for col in required_columns:
            if col not in self.df.columns:
                raise ValueError(f"Dataset missing required column: {col}")
        
        # Load DDC categories
        self.ddc_categories = get_ddc_categories()
        
        # Example samples (use first 5 rows of dataset)
        self.example_samples = self.df.head(5).to_dict('records')
        
        # Initialize status
        self.resume_status = self._load_resume_status()

    def get_or_create_sample(self, sample_size: int, random_seed: int = 42, exclude_first_n: int = 5,
                             cache_file: Optional[str] = None) -> pd.DataFrame:
        """Load cached sampled data if present; otherwise stratified sample per class, cache, and return."""
        self.sample_cache_file = cache_file

        if cache_file and os.path.exists(cache_file):
            print(f"Loading cached sampled data from: {cache_file}")
            cached_df = load_dataset(cache_file)
            return cached_df

        sampled_df = self.get_sampled_data(sample_size, random_seed, exclude_first_n)

        # Attach English label name for clarity/checkpoint
        sampled_df = sampled_df.copy()
        sampled_df['DDC_Level1_Name'] = sampled_df['DDC_Level1'].astype(str).map(self.ddc_categories)

        if cache_file:
            sampled_df.to_csv(cache_file, index=False, encoding='utf-8')
            print(f"Cached sampled data saved to: {cache_file}")

        return sampled_df
    
    def _load_resume_status(self) -> List[Dict]:
        """Load resume from breakpoint status"""
        if os.path.exists(self.resume_file):
            try:
                df_status = pd.read_csv(self.resume_file)
                return df_status.to_dict('records')
            except Exception as e:
                print(f"Failed to load resume status: {e}")
                return []
        return []
    
    def _save_resume_status(self, status_data: List[Dict]):
        """Save resume from breakpoint status"""
        if status_data:
            try:
                df_status = pd.DataFrame(status_data)
                df_status.to_csv(self.resume_file, index=False)
                print(f"Resume status saved to: {self.resume_file}")
            except Exception as e:
                print(f"Failed to save resume status: {e}")
    
    def get_sampled_data(self, sample_size: int, random_seed: int = 42, 
                        exclude_first_n: int = 5) -> pd.DataFrame:
        """Get stratified sampled data.
        - Reserve first `exclude_first_n` rows for shot examples.
        - For each class, take the first `sample_size` rows from the remaining data (deterministic).
        - Concatenate all class slices, then shuffle once with the given random seed.
        """
        
        # Data after excluding first N rows (kept for shot examples)
        available_data = self.df.iloc[exclude_first_n:].copy()
        
        per_class_target = sample_size
        sampled_parts = []
        class_counts = {}
        
        for cls, group in available_data.groupby('DDC_Level1'):
            take = min(len(group), per_class_target)
            class_counts[str(cls)] = take
            if take == 0:
                continue
            # Deterministic slice per class, no random sampling inside class
            sampled_parts.append(group.head(take))
        
        if not sampled_parts:
            raise ValueError("No data available for sampling after exclusions.")
        
        # Shuffle once after concatenation for randomness across classes
        sampled_data = pd.concat(sampled_parts).sample(frac=1.0, random_state=random_seed).reset_index(drop=True)
        
        print(f"Sampling settings:")
        print(f"  - Random seed for shuffle: {random_seed}")
        print(f"  - Exclude first {exclude_first_n} rows (reserved for shot examples)")
        print(f"  - Per-class sample size (head): {per_class_target}")
        print(f"  - Available data: {len(available_data)} rows")
        print(f"  - Actual sampled per class: {class_counts}")
        print(f"  - Total sampled: {len(sampled_data)}")
        
        return sampled_data
    
    def run_zero_shot_classification(self, sample_size: int = 10, random_seed: int = 42,
                                    exclude_first_n: int = 5, sampled_data: Optional[pd.DataFrame] = None) -> Dict:
        """Run zero-shot classification"""
        
        print("\n=== Starting Zero-shot Classification ===")
        
        # Get sampled data (reuse if provided to keep control variables constant)
        if sampled_data is None:
            sampled_data = self.get_sampled_data(sample_size, random_seed, exclude_first_n)
        sampled_data = sampled_data.copy().reset_index(drop=True)
        sample_size = len(sampled_data)
        
        # Create output file
        output_file = os.path.join(self.output_dir, f"zero_shot_results_seed{random_seed}_size{sample_size}.csv")
        
        # Create temporary file for classification
        temp_input_file = os.path.join(self.output_dir, "temp_zero_shot_input.csv")
        sampled_data.to_csv(temp_input_file, index=False)
        
        # Run classification
        results = zero_shot_classify_dataset_with_confidence(
            temp_input_file, output_file, sample_size=0, provider=self.provider, model=self.model
        )
        
        # Calculate accuracy
        accuracy_metrics = calculate_top_k_accuracy(results)
        per_class_accuracy = calculate_top_k_accuracy_by_class(results)
        
        # Save status
        status_entry = {
            'timestamp': datetime.now().isoformat(),
            'method': 'zero_shot',
            'sample_size': sample_size,
            'random_seed': random_seed,
            'top1_accuracy': accuracy_metrics['top1'],
            'top3_accuracy': accuracy_metrics['top3'],
            'top5_accuracy': accuracy_metrics['top5'],
            'per_class_accuracy': per_class_accuracy,
            'results_file': output_file,
            'provider': self.provider,
            'model': self.model or '',
        }
        
        self.resume_status.append(status_entry)
        self._save_resume_status(self.resume_status)
        
        # Clean up temporary file
        if os.path.exists(temp_input_file):
            os.remove(temp_input_file)
        
        return {
            'method': 'zero_shot',
            'accuracy': accuracy_metrics,
            'per_class_accuracy': per_class_accuracy,
            'results': results,
            'sample_size': sample_size
        }
    
    def run_one_shot_classification(self, sample_size: int = 10, random_seed: int = 42,
                                   exclude_first_n: int = 5, sampled_data: Optional[pd.DataFrame] = None) -> Dict:
        """Run one-shot classification"""
        
        print("\n=== Starting One-shot Classification ===")
        
        # Get sampled data (reuse if provided to keep control variables constant)
        if sampled_data is None:
            sampled_data = self.get_sampled_data(sample_size, random_seed, exclude_first_n)
        sampled_data = sampled_data.copy().reset_index(drop=True)
        sample_size = len(sampled_data)
        
        # Select an example from the reserved first N rows (keeps eval size intact)
        example_sample = self.example_samples[0]
        example_category_code = str(example_sample['DDC_Level1'])
        example_category_name = self.ddc_categories.get(example_category_code, 'Unknown')
        example_support = {
            'title': example_sample['Title'],
            'abstract': example_sample['Abstract'],
            'code': example_category_code
        }
        
        print(f"Example category: {example_category_name}")
        
        # Create output file
        output_file = os.path.join(self.output_dir, f"one_shot_results_seed{random_seed}_size{sample_size}.csv")
        
        # Create temporary file for classification
        temp_input_file = os.path.join(self.output_dir, "temp_one_shot_input.csv")
        sampled_data.to_csv(temp_input_file, index=False)
        
        # Run classification
        results = one_shot_classify_dataset_with_confidence(
            temp_input_file, output_file, sample_size=0,
            example_category=example_category_code, example_support=example_support,
            provider=self.provider, model=self.model
        )
        
        # Calculate accuracy
        accuracy_metrics = calculate_top_k_accuracy(results)
        per_class_accuracy = calculate_top_k_accuracy_by_class(results)
        
        # Save status
        status_entry = {
            'timestamp': datetime.now().isoformat(),
            'method': 'one_shot',
            'sample_size': sample_size,
            'random_seed': random_seed,
            'example_category': example_category_name,
            'top1_accuracy': accuracy_metrics['top1'],
            'top3_accuracy': accuracy_metrics['top3'],
            'top5_accuracy': accuracy_metrics['top5'],
            'per_class_accuracy': per_class_accuracy,
            'results_file': output_file,
            'provider': self.provider,
            'model': self.model or '',
        }
        
        self.resume_status.append(status_entry)
        self._save_resume_status(self.resume_status)
        
        # Clean up temporary file
        if os.path.exists(temp_input_file):
            os.remove(temp_input_file)
        
        return {
            'method': 'one_shot',
            'accuracy': accuracy_metrics,
            'per_class_accuracy': per_class_accuracy,
            'results': results,
            'sample_size': sample_size,
            'example_category': example_category_name
        }
    
    def run_few_shot_classification(self, sample_size: int = 10, random_seed: int = 42, 
                                  num_examples: int = 5, exclude_first_n: int = 5,
                                  sampled_data: Optional[pd.DataFrame] = None) -> Dict:
        """Run few-shot classification"""
        
        print("\n=== Starting Few-shot Classification ===")
        
        # Get sampled data (reuse if provided to keep control variables constant)
        if sampled_data is None:
            sampled_data = self.get_sampled_data(sample_size, random_seed, exclude_first_n)
        sampled_data = sampled_data.copy().reset_index(drop=True)
        sample_size = len(sampled_data)

        # Build support examples from the reserved first N rows to avoid shrinking eval set
        support_examples = []
        for row in self.example_samples[:num_examples]:
            code = str(row['DDC_Level1'])
            support_examples.append({
                'title': row['Title'],
                'abstract': row['Abstract'],
                'category': self.ddc_categories.get(code, 'Unknown'),
                'code': code
            })
        
        # Create output file
        output_file = os.path.join(self.output_dir, f"few_shot_results_seed{random_seed}_size{sample_size}.csv")
        
        # Create temporary file for classification
        temp_input_file = os.path.join(self.output_dir, "temp_few_shot_input.csv")
        sampled_data.to_csv(temp_input_file, index=False)
        
        # Run classification
        results = few_shot_classify_dataset_with_confidence(
            temp_input_file, output_file, sample_size=0, num_examples=num_examples,
            support_examples=support_examples, provider=self.provider, model=self.model
        )
        
        # Calculate accuracy
        accuracy_metrics = calculate_top_k_accuracy(results)
        per_class_accuracy = calculate_top_k_accuracy_by_class(results)
        
        # Save status
        status_entry = {
            'timestamp': datetime.now().isoformat(),
            'method': 'few_shot',
            'sample_size': sample_size,
            'random_seed': random_seed,
            'num_examples': num_examples,
            'top1_accuracy': accuracy_metrics['top1'],
            'top3_accuracy': accuracy_metrics['top3'],
            'top5_accuracy': accuracy_metrics['top5'],
            'per_class_accuracy': per_class_accuracy,
            'results_file': output_file,
            'provider': self.provider,
            'model': self.model or '',
        }
        
        self.resume_status.append(status_entry)
        self._save_resume_status(self.resume_status)
        
        # Clean up temporary file
        if os.path.exists(temp_input_file):
            os.remove(temp_input_file)
        
        return {
            'method': 'few_shot',
            'accuracy': accuracy_metrics,
            'per_class_accuracy': per_class_accuracy,
            'results': results,
            'sample_size': sample_size,
            'num_examples': num_examples
        }
    
    def generate_summary(self, results: List[Dict]) -> Dict:
        """Generate classification result summary"""
        
        summary = {
            'generated_at': datetime.now().isoformat(),
            'input_file': self.input_file,
            'total_samples': len(self.df),
            'classification_results': {},
            'comparison': {}
        }
        
        for result in results:
            method = result['method']
            summary['classification_results'][method] = {
                'sample_size': result['sample_size'],
                'accuracy': result['accuracy'],
                'per_class_accuracy': result.get('per_class_accuracy', {}),
                'additional_info': {k: v for k, v in result.items() 
                                  if k not in ['method', 'accuracy', 'results', 'sample_size']}
            }
        
        # Compare performance of different methods
        if len(results) > 1:
            comparison = {}
            for metric in ['top1', 'top3', 'top5']:
                comparison[metric] = {}
                for result in results:
                    method = result['method']
                    comparison[metric][method] = result['accuracy'][metric]
            
            summary['comparison'] = comparison
        
        # Save summary to file
        with open(self.summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        return summary
    
    def print_summary(self, summary: Dict):
        """Print classification result summary"""
        
        print("\n" + "="*60)
        print("Classification Result Summary")
        print("="*60)
        
        print(f"Input file: {summary['input_file']}")
        print(f"Total samples: {summary['total_samples']}")
        print(f"Generated at: {summary['generated_at']}")
        
        print("\nAccuracy by Method:")
        for method, info in summary['classification_results'].items():
            print(f"\n{method.upper()}:")
            print(f"  Sample size: {info['sample_size']}")
            accuracy = info['accuracy']
            counts = accuracy.get('counts', {})
            total = counts.get('total', 0)
            print(f"  Top-1 Accuracy: {accuracy['top1']:.2%} ({counts.get('top1_correct', 0)}/{total})")
            print(f"  Top-3 Accuracy: {accuracy['top3']:.2%} ({counts.get('top3_correct', 0)}/{total})")
            print(f"  Top-5 Accuracy: {accuracy['top5']:.2%} ({counts.get('top5_correct', 0)}/{total})")

            # Print per-class accuracy if available
            per_class = info.get('per_class_accuracy', {})
            if per_class:
                print(f"  Per-class accuracy (support=#samples):")
                for cls_name, metrics in per_class.items():
                    support = metrics.get('support', 0)
                    top1_c = metrics.get('top1_correct', 0)
                    top3_c = metrics.get('top3_correct', 0)
                    top5_c = metrics.get('top5_correct', 0)
                    print(f"    {cls_name} (n={support}): "
                          f"Top1 {metrics.get('top1', 0):.2%} ({top1_c}/{support}), "
                          f"Top3 {metrics.get('top3', 0):.2%} ({top3_c}/{support}), "
                          f"Top5 {metrics.get('top5', 0):.2%} ({top5_c}/{support})")
                
                # Print additional information
                if 'example_category' in info['additional_info']:
                    print(f"  Example category: {info['additional_info']['example_category']}")
                if 'num_examples' in info['additional_info']:
                    print(f"  Training examples: {info['additional_info']['num_examples']}")
        
        # Print comparison results
        if summary['comparison']:
            print("\nMethod Comparison:")
            for metric, methods in summary['comparison'].items():
                print(f"  {metric.upper()}:")
                for method, accuracy in methods.items():
                    print(f"    {method}: {accuracy:.2%}")
        
        print(f"\nDetailed results saved to: {self.summary_file}")
        print("="*60)

def main():
    """Main function"""
    
    # Set command line arguments
    default_input = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', 'datasets', 'AG_news_test_with_DDC_Hierarchy_mod_26Nov25_english_mapped.csv'))
    default_output_dir = os.path.join(SCRIPT_DIR, '..', 'results')
    parser = argparse.ArgumentParser(description='Text classification main script')
    parser.add_argument('--input', type=str, default=default_input,
                       help='Input file path')
    parser.add_argument('--output_dir', type=str, default=default_output_dir,
                       help='Output directory path')
    parser.add_argument('--sample_size', type=int, default=100,
                       help='Per-class sample size (after excluding first N rows)')
    parser.add_argument('--random_seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--exclude_first_n', type=int, default=5,
                       help='Exclude first N rows')
    parser.add_argument('--methods', type=str, default='zero,one,few',
                       help='Methods to run, comma separated: zero,one,few')
    parser.add_argument('--provider', type=str, default=None,
                       help='LLM provider: deepseek, chatgpt, gemini (auto-inferred from model_name when omitted)')
    parser.add_argument('--model', type=str, default=None,
                       help='Optional model name override for the provider')
    parser.add_argument('--model_name', type=str, default=None,
                       help='Model name used to name the output folder (e.g., deepseek-chat, chatgpt-5.1)')
    
    args = parser.parse_args()

    # Normalize paths so outputs/input live under model/API/Deepseek by default
    input_path = args.input
    if not os.path.isabs(input_path):
        input_path = os.path.join(SCRIPT_DIR, input_path)
    # Infer provider from model_name if provider not explicitly set
    model_name_for_folder = args.model_name or args.provider or "deepseek-chat"
    provider = args.provider
    if provider is None:
        lowered = model_name_for_folder.lower()
        if "gemini" in lowered:
            provider = "gemini"
        elif "gpt" in lowered or "chatgpt" in lowered or "o3" in lowered or "gpt-" in lowered:
            provider = "chatgpt"
        elif "deepseek" in lowered:
            provider = "deepseek-chat"
        else:
            provider = "deepseek-chat"

    output_base_dir = args.output_dir
    if not os.path.isabs(output_base_dir):
        output_base_dir = os.path.join(SCRIPT_DIR, output_base_dir)
    # ensure model-specific folder
    output_dir = os.path.join(output_base_dir, model_name_for_folder)
    
    print("=== Text Classification Main Script ===")
    print("Supports Zero-shot, One-shot, Few-shot Classification")
    print("="*50)
    
    # Create classification manager
    manager = ClassificationManager(input_path, output_dir, provider=provider, model=args.model or args.model_name)
    
    # Parse methods to run
    methods_to_run = [method.strip().lower() for method in args.methods.split(',')]
    
    # Sample once (exclude first N rows) to keep control variables fixed across methods.
    # Cache the sampled set for restart/reevaluation; if the cache exists, reuse it.
    sample_cache_file = os.path.join(
        output_dir,
        f"sample_seed{args.random_seed}_perclass{args.sample_size}_skip{args.exclude_first_n}.csv"
    )
    shared_sample = manager.get_or_create_sample(
        args.sample_size, args.random_seed, args.exclude_first_n, cache_file=sample_cache_file
    )
    
    # Run specified classification methods on the same sampled data
    all_results = []
    
    try:
        if 'zero' in methods_to_run:
            zero_shot_result = manager.run_zero_shot_classification(
                args.sample_size, args.random_seed, args.exclude_first_n, shared_sample
            )
            all_results.append(zero_shot_result)
        
        if 'one' in methods_to_run:
            one_shot_result = manager.run_one_shot_classification(
                args.sample_size, args.random_seed, args.exclude_first_n, shared_sample
            )
            all_results.append(one_shot_result)
        
        if 'few' in methods_to_run:
            few_shot_result = manager.run_few_shot_classification(
                args.sample_size, args.random_seed, num_examples=3,
                exclude_first_n=args.exclude_first_n, sampled_data=shared_sample
            )
            all_results.append(few_shot_result)
        
        # Generate summary
        if all_results:
            summary = manager.generate_summary(all_results)
            manager.print_summary(summary)
            print("\n✅ All classification tasks completed!")
        else:
            print("⚠️ No classification methods were run")
        
    except Exception as e:
        print(f"❌ Error during classification: {e}")
        # Save current status
        if all_results:
            summary = manager.generate_summary(all_results)
            manager.print_summary(summary)
        sys.exit(1)

if __name__ == "__main__":
    main()
