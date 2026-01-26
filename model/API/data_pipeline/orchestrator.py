
import os
import sys
import pandas as pd
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from .config import ConfigLoader, WORKERS, RANDOM_SEED
from .data_loader import DataLoader
from .generator import PromptGenerator
from .inference import InferenceEngine
from .validator import ResponseValidator
from .metrics import MetricsCalculator
from .reporting import Reporter
import random
import numpy as np

def get_checkpoint_path(config) -> str:
    """Generate checkpoint file path based on config."""
    dataset_name = os.path.splitext(os.path.basename(config.dataset_path))[0]
    model_name = "".join([c for c in config.deepseek_model if c.isalnum() or c in ('-', '_')]).strip()
    checkpoint_dir = os.path.join(config.output_dir, model_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    return os.path.join(checkpoint_dir, f".checkpoint_{dataset_name}_level{config.level}.csv")

def load_checkpoint(checkpoint_path: str) -> set:
    """Load completed indices from checkpoint file."""
    if os.path.exists(checkpoint_path):
        try:
            df = pd.read_csv(checkpoint_path)
            completed = set(df['index'].tolist())
            print(f"Resuming from checkpoint: {len(completed)} samples already completed")
            return completed
        except Exception as e:
            print(f"Warning: Could not load checkpoint ({e}), starting fresh")
    return set()

def save_checkpoint(results: list, checkpoint_path: str):
    """Save results to checkpoint file (append mode for incremental save)."""
    if not results:
        return
    df = pd.DataFrame(results)
    # Append if exists, else create
    if os.path.exists(checkpoint_path):
        df.to_csv(checkpoint_path, mode='a', header=False, index=False)
    else:
        df.to_csv(checkpoint_path, index=False)

def main():
    # Set Global Seeds
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # State 1: Initialization
    config_loader = ConfigLoader(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    config = config_loader.parse_args()
    
    print(f"Starting Pipeline: Level={config.level}, Shot={config.shot_type}, Dataset={config.dataset_path}")

    # State 2: Data Loading
    loader = DataLoader(config)
    df = loader.load_and_clean()
    
    # State 3: Support Set
    support_map, used_indices = loader.select_support_set(df)
    
    # State 4: Test Sampling
    test_df = loader.sample_test_set(df, used_indices)
    print(f"Test Set Size: {len(test_df)}")
    
    # Checkpoint: Load completed indices
    checkpoint_path = get_checkpoint_path(config)
    completed_indices = load_checkpoint(checkpoint_path)
    
    # Filter out already completed samples
    if completed_indices:
        original_size = len(test_df)
        test_df = test_df[~test_df.index.isin(completed_indices)]
        print(f"Remaining samples to process: {len(test_df)} (skipped {original_size - len(test_df)})")
    
    # Components
    generator = PromptGenerator(config)
    engine = InferenceEngine(config)
    validator = ResponseValidator()
    
    # Execution (Parallel) with incremental checkpoint
    results = []
    batch_size = 50  # Save checkpoint every N results
    
    def process_row(index, row):
        try:
            # 1. Candidate Generation (State 5)
            parent_code = row['parent_code']
            candidates = generator.get_candidates(parent_code)
            
            if not candidates:
                return None
            
            # 2. Prompt Construction (State 6)
            support_sample = support_map.get(parent_code) 
            prompt = generator.construct_prompt(row, support_sample, candidates)
            
            # 3. Model Inference (State 7)
            start_t = time.time()
            raw_response = engine.call_model(prompt)
            duration = time.time() - start_t
            
            # 4. Parsing (State 8) - Now returns list of (code, conf) tuples
            predictions = validator.parse_and_validate(raw_response, candidates)
            
            # Extract codes list for Top-K evaluation
            predicted_codes = [p[0] for p in predictions]
            top1_code = predicted_codes[0] if predicted_codes else "UNKNOWN"
            top1_conf = predictions[0][1] if predictions else 0.0
            
            return {
                "index": index,
                "title": row.get('title', ''),
                "ddc_code": row['ddc_code'],
                "predicted_code": top1_code,  # For backward compatibility
                "predicted_codes": predicted_codes,  # List of up to 5 codes
                "predictions": predictions,  # Full (code, conf) list
                "confidence": top1_conf,
                "duration": duration,
                "candidate_count": len(candidates),
                "prompt": prompt,
                "response": raw_response
            }
        except Exception as e:
            print(f"Error processing row {index}: {e}")
            return None

    print("Running Inference...")
    pending_save = []
    
    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        futures = {executor.submit(process_row, idx, row): idx for idx, row in test_df.iterrows()}
        
        for future in tqdm(as_completed(futures), total=len(futures)):
            res = future.result()
            if res:
                results.append(res)
                pending_save.append(res)
                
                # Incremental checkpoint save
                if len(pending_save) >= batch_size:
                    save_checkpoint(pending_save, checkpoint_path)
                    pending_save = []
    
    # Save remaining results
    if pending_save:
        save_checkpoint(pending_save, checkpoint_path)
    
    # Load all results from checkpoint for final metrics
    if os.path.exists(checkpoint_path):
        results_df = pd.read_csv(checkpoint_path)
        # Parse predicted_codes if stored as string
        if 'predicted_codes' in results_df.columns:
            results_df['predicted_codes'] = results_df['predicted_codes'].apply(
                lambda x: eval(x) if isinstance(x, str) and x.startswith('[') else []
            )
    else:
        results_df = pd.DataFrame(results)
    
    # State 9: Metrics
    if not results_df.empty:
        # Prepare data for Top-K metrics
        # CRITICAL FIX: Ensure Ground Truth is 3-digit string (handle '0' -> '000' issue)
        y_true = results_df['ddc_code'].astype(str).str.strip().str.extract(r'(\d+)')[0].fillna('0').str.zfill(3).tolist()
        
        y_pred_top5 = results_df['predicted_codes'].apply(
             lambda x: [str(c).zfill(3) for c in x] if isinstance(x, list) else []
        ).tolist()
        
        durations = results_df['duration'].tolist() if 'duration' in results_df.columns else [0] * len(y_true)
        
        metrics = MetricsCalculator.compute(
            y_true=y_true,
            y_pred_top5=y_pred_top5,
            durations=durations
        )
        print(f"Metrics: Top-1={metrics['top1_accuracy']:.2%}, Top-3={metrics['top3_accuracy']:.2%}, Top-5={metrics['top5_accuracy']:.2%}")
        
        # State 10: Reporting
        reporter = Reporter(config)
        reporter.generate_report(metrics, results_df)
        
        # Clean up checkpoint after successful completion
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
            print("Checkpoint file cleaned up after successful completion.")
    else:
        print("No results generated.")

if __name__ == "__main__":
    main()
