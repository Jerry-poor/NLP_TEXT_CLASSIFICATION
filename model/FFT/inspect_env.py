#!/usr/bin/env python3
#SBATCH --job-name=inspect_env
#SBATCH --output=logs/inspect_env_%j.out
#SBATCH --error=logs/inspect_env_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --partition=compute

import os
import subprocess
import sys
import traceback

def print_flush(msg):
    print(msg)
    sys.stdout.flush()

def main():
    print_flush("===== BEGIN ENV INSPECTION =====")

    # 原始环境
    print_flush("\n-- BEFORE sourcing ~/.my_envs --")
    for k,v in sorted(os.environ.items()):
        print_flush(f"{k}={v}")

    # 读取 ~/.my_envs
    path = os.path.expanduser("~/.my_envs")
    print_flush(f"\n-- CONTENT OF {path} --")
    try:
        print(open(path).read())
    except FileNotFoundError:
        print_flush(f"{path} not found!")

    # source ~/.my_envs via login shell
    print_flush("\n-- AFTER sourcing ~/.my_envs --")
    result = subprocess.run(
        ["bash", "-lc", "source ~/.my_envs && env"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    print_flush(result.stdout)
    if result.stderr:
        print_flush("STDERR:\n" + result.stderr)

    # GPU check
    print_flush("\n-- CUDA_VISIBLE_DEVICES --")
    print_flush(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    try:
        out = subprocess.check_output(["nvidia-smi"], stderr=subprocess.STDOUT).decode()
        print_flush(out)
    except Exception as e:
        print_flush(f"nvidia-smi error: {e}")

    print_flush("===== END ENV INSPECTION =====")

if __name__ == "__main__":
    main()
    sys.exit(0)
