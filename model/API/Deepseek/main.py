#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compatibility wrapper that delegates to the unified API/method1 entrypoint."""

import os
import sys
from subprocess import call


def _inject_default_provider(args):
    """Ensure deepseek provider is set when calling method1/main."""
    if "--provider" in args:
        return args
    return args + ["--provider", "deepseek-chat"]


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    method1_dir = os.path.abspath(os.path.join(script_dir, "..", "method1"))
    target = os.path.join(method1_dir, "main.py")

    if not os.path.exists(target):
        raise FileNotFoundError(f"method1 entrypoint not found at {target}")

    args = _inject_default_provider(sys.argv[1:])
    cmd = [sys.executable, target] + args
    sys.exit(call(cmd))


if __name__ == "__main__":
    main()
