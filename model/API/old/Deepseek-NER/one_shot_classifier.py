#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compatibility shim: delegates to API/method1/one_shot_classifier_with_confidence."""

import os
import sys

METHOD1_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "method1"))
sys.path.insert(0, METHOD1_DIR)

from one_shot_classifier_with_confidence import *  # noqa: F401,F403
