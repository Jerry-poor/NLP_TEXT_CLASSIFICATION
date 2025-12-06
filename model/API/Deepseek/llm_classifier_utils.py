#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compatibility shim: delegates to API/method1/llm_classifier_utils."""

import os
import sys

METHOD1_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "method1"))
sys.path.insert(0, METHOD1_DIR)

from llm_classifier_utils import *  # noqa: F401,F403
