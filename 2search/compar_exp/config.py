# -*- coding: utf-8 -*-
"""
Shared configuration for the L, LLM, F and R experiments.
============================================
Paths
    - Raw test series: 0dataset/test/{univariate,multivariate}/*.npy
    - Test meta-features: 0dataset/testmetrics/{univariate,multivariate}/<dataset>/meta_features.csv
    - Law table: connect/document_summary.csv. Module baselines use pooled law_search/inputs candidates.
    - Training pipeline: reuse training and evaluation from 1experiment/.

Training defaults: TIME_LAG=12, split 0.6/0.2/0.2, early-stopping patience=15.
Seeds [2026, 2027, 2028]: L repeats one architecture; LLM requests an architecture for each seed.
"""

import os
from pathlib import Path

# Root paths
REPO_ROOT = Path(__file__).resolve().parents[2]
SEARCH_ROOT = Path(__file__).resolve().parent.parent
TEST_DATA_ROOT = REPO_ROOT / "0dataset" / "test"
TEST_FEAT_ROOT = REPO_ROOT / "0dataset" / "testmetrics"
AI_TYPES = ("deepseek", "chatgpt", "gemini")
LAW_TABLE = Path(os.environ.get("TSF_LAW_TABLE") or SEARCH_ROOT / "connect/document_summary.csv").resolve()
POOLED_LAW_TABLE = Path(os.environ.get("TSF_POOLED_LAW_TABLE") or
                        SEARCH_ROOT / "connect/04_pooled_all_4734/outputs/pooled_all_law_table.csv").resolve()
LAW_XLSX = LAW_TABLE  # Retain the existing name. LAW_TABLE also accepts the original manually selected XLSX.
LAW_INPUT_DIR = SEARCH_ROOT / "law_search/inputs"
OVERVIEW_CSV = os.environ.get("TSF_OVERVIEW_CSV")
RUN_PIPELINE_DIR = REPO_ROOT / "1experiment"
# L and R denote law-guided and random selection. Write both to compar_result.
RESULT_ROOT = Path(os.environ.get("TSF_RESULT_ROOT") or
                   (SEARCH_ROOT / "compar_result")).expanduser().resolve()

# main.py provides a shared entry point for the existing experiment arguments.
EXPERIMENTS = ("L", "F", "R", "LLM")
DATASETS = "all"
DATASET_LIMIT = 0
DATASET_RANGE = (0, 0)
RUN_SUMMARY = True

# Task and training defaults
TIME_LAG = 12
HORIZONS = [3, 12]                # Evaluate configurations 12_3 and 12_12.
BATCH_SIZE = 16
EPOCHS = 300
LR = 1e-3
WEIGHT_DECAY = 0.0
EARLY_STOP_PATIENCE = 15
TRAIN_RATIO = 0.6
VAL_RATIO = 0.2
SEEDS = [2026, 2027, 2028]        # Seeds used for stability runs.

# Architecture sources
ARCH_LAW = "law"        # L: law-guided
ARCH_LLM = "llm"        # LLM: direct model design
ARCH_DEFAULT = "default"  # F: most frequent motifs by stage (highest p_base)
ARCH_RANDOM = "random"    # R: random architecture

# LLM settings. Read keys from environment variables.
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
LLM_BASE_URL = "https://api.deepseek.com"
LLM_MODEL = "deepseek-flash"  # Model identifier used for V4.1-Flash-1017 on 2026-11-09.
LLM_TEMPERATURE = 0.2     # Lower temperature reduces variability in generated JSON.
LLM_MAX_TOKENS = 6000     # Allow more tokens for model code than for a JSON specification.

# LLM capacity guidance
# The LLM generates model code freely. The prompt uses the L parameter count times
# [LLM_PARAM_LO, LLM_PARAM_HI] as a soft reference; results record whether the count falls in range.
LLM_PARAM_LO, LLM_PARAM_HI = 0.5, 2.0
