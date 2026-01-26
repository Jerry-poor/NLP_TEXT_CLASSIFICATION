#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unified API entrypoint. Dispatches to method1 (and future method2) with model-based output foldering."""

import argparse
import os
import sys
from subprocess import call


def parse_args():
    parser = argparse.ArgumentParser(description="Unified API entrypoint")
    parser.add_argument("--method_id", type=int, choices=[1, 2], default=1, help="Method version to run (1 or 2)")
    parser.add_argument("--model_name", type=str, required=True, help="Model name (e.g., deepseek-chat, chatgpt-5.1, gemini-1.5)")
    parser.add_argument("--provider", type=str, default=None, help="LLM provider (auto-inferred from model_name if omitted)")
    parser.add_argument("--methods", type=str, default="zero,one,few", help="Shot strategies to run: zero,one,few")
    parser.add_argument("--input", type=str, default=None, help="Optional input dataset path override")
    parser.add_argument("--output_dir", type=str, default=None, help="Optional base output dir (defaults to method dir)")
    parser.add_argument("--sample_size", type=int, default=100, help="Per-class sample size")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed")
    parser.add_argument("--exclude_first_n", type=int, default=5, help="Exclude first N rows for supports")
    parser.add_argument("--num_examples", type=int, default=3, help="Few-shot support examples (if applicable)")
    return parser.parse_args()


def dispatch_method1(args):
    method_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "method1")
    target = os.path.join(method_dir, "main.py")
    if not os.path.exists(target):
        raise FileNotFoundError(f"method1 main not found at {target}")

    forwarded = [
        "--methods", args.methods,
        "--provider", args.provider or args.model_name,
        "--model_name", args.model_name,
        "--sample_size", str(args.sample_size),
        "--random_seed", str(args.random_seed),
        "--exclude_first_n", str(args.exclude_first_n),
    ]

    if args.input:
        forwarded += ["--input", args.input]
    if args.output_dir:
        forwarded += ["--output_dir", args.output_dir]
    if args.num_examples is not None:
        forwarded += ["--num_examples", str(args.num_examples)]
    # pass through explicit model override as model
    forwarded += ["--model", args.model_name]

    cmd = [sys.executable, target] + forwarded
    return call(cmd)


def dispatch_method2(args):
    method_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "method2")
    target = os.path.join(method_dir, "main.py")
    if not os.path.exists(target):
        raise FileNotFoundError(f"method2 main not found at {target}")

    forwarded = [
        "--methods", args.methods,
        "--model_name", args.model_name,
        "--sample_size", str(args.sample_size),
        "--random_seed", str(args.random_seed),
        "--exclude_first_n", str(args.exclude_first_n),
    ]

    if args.provider:
        forwarded += ["--provider", args.provider]
    if args.input:
        forwarded += ["--input", args.input]
    if args.output_dir:
        forwarded += ["--output_dir", args.output_dir]
    if args.model_name:
        forwarded += ["--model", args.model_name]

    cmd = [sys.executable, target] + forwarded
    return call(cmd)


def main():
    args = parse_args()
    method_id = args.method_id

    if method_id == 1:
        sys.exit(dispatch_method1(args))
    elif method_id == 2:
        sys.exit(dispatch_method2(args))
    else:
        raise ValueError(f"Unsupported method_id: {method_id}")


if __name__ == "__main__":
    main()
