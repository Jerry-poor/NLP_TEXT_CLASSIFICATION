#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Utility: print token IDs for DDC codes (for logit_bias tuning)."""

import argparse
import sys
from typing import List


def parse_codes(codes_str: str) -> List[str]:
    return [c.strip() for c in codes_str.split(",") if c.strip()]


def main():
    parser = argparse.ArgumentParser(description="Print token IDs for given codes using tiktoken encoding.")
    parser.add_argument(
        "--codes",
        type=str,
        default="0,100,200,300,400,500,600,700,800,900",
        help="Comma-separated list of codes to tokenize.",
    )
    parser.add_argument(
        "--encoding",
        type=str,
        default="o200k_base",
        help="tiktoken encoding name (try o200k_base, cl100k_base, p50k_base).",
    )
    args = parser.parse_args()

    try:
        import tiktoken
    except Exception as exc:  # pragma: no cover
        print("tiktoken is required to run this script. Install it and retry.", file=sys.stderr)
        print(f"Import error: {exc}", file=sys.stderr)
        sys.exit(1)

    try:
        enc = tiktoken.get_encoding(args.encoding)
    except Exception as exc:
        print(f"Failed to load encoding '{args.encoding}': {exc}", file=sys.stderr)
        sys.exit(1)

    codes = parse_codes(args.codes)
    print(f"Encoding: {args.encoding}")
    for code in codes:
        token_ids = enc.encode(code)
        print(f"{code}: {token_ids}")
    single_token = [code for code in codes if len(enc.encode(code)) == 1]
    multi_token = [code for code in codes if len(enc.encode(code)) != 1]
    print("\nSummary:")
    print(f"Single-token codes ({len(single_token)}): {single_token}")
    if multi_token:
        print(f"Multi-token codes ({len(multi_token)}): {multi_token}")


if __name__ == "__main__":
    main()
