# -*- coding: utf-8 -*-
"""Discovery configuration and paths, using the original statistical defaults."""
from pathlib import Path

SEARCH_ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = Path(__file__).resolve().parent / "inputs"
OUTPUT_DIR = SEARCH_ROOT / "law_result"
AI_TYPES = ("deepseek", "chatgpt", "gemini")
INCLUDE_COMBINED = True  # Discover laws separately by provider, then pool candidates across all three providers.
VERBOSE = False

# Tree and candidate-rule settings from the original discover_rules.py.
MAX_DEPTH = 3
MIN_LEAF = 60
MIN_REL_GAIN = 0.01
MIN_RULE_LIFT = 1.15
SEED = 1234
THRESHOLD_ROUND = 3
MIN_RULE_N = 60
MIN_DISCOVERY_DATASETS = 15
PURITY_MIN_SUB = 15
PURITY_MIN_FRAC = 0.10
PURITY_LIFT_MAX = 0.90
PURITY_MAX_SPLIT = 2
CANONICALIZE = True
MIN_PURITY_DATASETS = 12

# Original full-data permutation, bootstrap and grading settings, without a confirmation split.
B_PERM = 1000
PERM_SEED = 2026
FDR_ALPHA = 0.10
N_BOOT = 1000
BOOT_SEED = 1234
CELLS = [("uni", "12_3"), ("uni", "12_12"),
         ("multi", "12_3"), ("multi", "12_12")]
MIN_CELL_DATASETS = 5
FDR_STRONG = 0.05
FDR_EXPLORATORY = 0.10
REPRO_MIN = 0.80
N_CELLS_TOTAL = 4
MIN_STRONG_COVERAGE = 0.10
MIN_STRONG_CELL_DATASETS = 10
MIN_STABLE_DATASET_FRACTION = 0.08
MIN_CANDIDATE_DATASETS = 12
