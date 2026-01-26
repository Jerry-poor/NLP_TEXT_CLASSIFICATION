
import argparse
import subprocess
import os
import sys
import time

def run_command(cmd, env=None):
    print(f"Executing: {' '.join(cmd)}")
    start_time = time.time()
    # Pass env to subprocess to ensures changes like BASE_URL propagate
    result = subprocess.run(cmd, capture_output=False, text=True, env=env)
    duration = time.time() - start_time
    if result.returncode != 0:
        print(f"Error executing command: {' '.join(cmd)}")
        print(f"Exit Code: {result.returncode}")
    else:
        print(f"Completed in {duration:.2f}s")
    return result.returncode == 0

def main():
    parser = argparse.ArgumentParser(description="Master Evaluation Script for DDC Classification")
    parser.add_argument('--model', type=str, default=None, help='Model name to evaluate (default: from .env)')
    parser.add_argument('--base_url', type=str, default=None, help='Base URL for API')
    parser.add_argument('--api_key', type=str, default=None, help='API Key override')
    parser.add_argument('--sample_size', type=int, default=500, help='Number of samples per dataset/level')
    parser.add_argument('--shot_type', type=str, default='zero', choices=['zero', 'one', 'few'], help='Shot type')
    parser.add_argument('--provider', type=str, default=None, help='Explicit provider override')
    
    args = parser.parse_args()
    
    # Dataset Configuration
    datasets = [
        {
            "name": "wos_unified.csv",
            "levels": [1, 2], # WOS has no Level 3
        },
        {
            "name": "ag_news_unified.csv",
            "levels": [1, 2, 3], # Full hierarchy
        },
        {
            "name": "lib_unified.csv",
            "levels": [1, 2, 3], # Full hierarchy
        }
    ]
    
    # Environment Setup
    # Allows overriding model/URL via main script args to propagate to orchestrator
    env = os.environ.copy()
    if args.model:
        env['DEEPSEEK_MODEL'] = args.model
    if args.base_url:
        env['DEEPSEEK_BASE_URL'] = args.base_url
    if args.api_key:
        env['DEEPSEEK_API_KEY'] = args.api_key
        
    print(f"==================================================")
    print(f"Starting Batch Evaluation")
    print(f"Model: {args.model if args.model else 'Default (.env)'}")
    print(f"Sample Size: {args.sample_size}")
    print(f"Shot Type: {args.shot_type}")
    print(f"==================================================\n")
    
    success_count = 0
    total_count = 0
    
    for ds in datasets:
        filename = ds['name']
        levels = ds['levels']
        
        for lvl in levels:
            total_count += 1
            print(f">>> Task {total_count}: {filename} Level {lvl} <<<")
            
            cmd = [
                sys.executable, "-m", "data_pipeline.orchestrator",
                "--dataset", filename,
                "--level", str(lvl),
                "--sample_size", str(args.sample_size),
                "--shot_type", args.shot_type
            ]
            
            if args.model:
                cmd.extend(["--model", args.model])
            if args.provider:
                cmd.extend(["--provider", args.provider])
            
            # Run with modified environment
            if run_command(cmd, env=env):
                success_count += 1
            print("\n")

    print(f"==================================================")
    print(f"Batch Evaluation Complete")
    print(f"Successful Tasks: {success_count}/{total_count}")
    
    # Determine result directory for display
    model_dir = args.model if args.model else os.getenv("DEEPSEEK_MODEL", "default")
    safe_model_dir = "".join([c for c in model_dir if c.isalnum() or c in ('-', '_')]).strip()
    print(f"Results are saved in: model/API/result/{safe_model_dir}/")
    print(f"==================================================")

if __name__ == "__main__":
    main()
