"""
Unified configuration file.

Usage steps:
1. Modify PROJECT_ROOT, data paths, and task hyperparameters.
2. Fill the selected AI profile or its API key environment variable.
3. Run run_discovery.py --ai deepseek (or chatgpt / gemini).
"""
import os
from datetime import datetime

AI_TYPES = ("chatgpt", "deepseek", "gemini")
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "deepseek")
_LLM_PROFILES = {'chatgpt': {'api_key': '',
             'model': 'gpt-5.6-luna',
             'base_url': 'https://api.openai.com/v1',
             'api_key_env': 'OPENAI_API_KEY'},
 'deepseek': {'api_key': '',
              'model': 'deepseek-flash',
              'base_url': 'https://api.deepseek.com',
              'api_key_env': 'DEEPSEEK_API_KEY'},
 'gemini': {'api_key': '',
            'model': 'gemini-3.5-flash-lite',
            'base_url': 'https://generativelanguage.googleapis.com/v1beta/openai/',
            'api_key_env': 'GEMINI_API_KEY'}}

# Univariate/Multivariate toggle: DATASET_NPY_DIR, UNIVARIATE
# ===== 1. Project Root Directory =====
# _INTERNAL_DIR is the directory where this config file lives (i.e., _internal/).
# All paths default relative to it for portability.
_INTERNAL_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = _INTERNAL_DIR

# ===== 2. Data Input =====
# ── npy scan directory: run_iterative_loop.py automatically scans all .npy files in this directory,
#    running one complete experiment per file ──
DATASET_NPY_DIR = os.path.join(os.path.dirname(PROJECT_ROOT), "0dataset", "train")
# Shared data split name, e.g. "train" or "test".
DATASET_CATEGORY = os.path.basename(DATASET_NPY_DIR)

# ── The following three paths are defaults; at runtime they are dynamically overridden
#     by run_iterative_loop.py via cfg_overrides ──
NPY_PATH = os.path.join(DATASET_NPY_DIR, "0639_australia_maxtemp_T1_4254_1.npy")
RESULT_ROOT = os.environ.get("RESULT_ROOT", os.path.join(PROJECT_ROOT, "result"))
ORIGINAL_STATS_CSV_PATH = os.path.join(RESULT_ROOT, "raw_indicators.csv")
ABSTRACT_STATS_CSV_PATH = os.path.join(RESULT_ROOT, "meta_features.csv")

# ── Seasonal period used by compute_metrics; set to 0 for automatic estimation when unknown ──
SEASONAL_PERIOD = int(os.environ.get("SEASONAL_PERIOD", "0"))

# When SEASONAL_PERIOD=0, whether to automatically estimate the seasonal period from the spectrum.
AUTO_ESTIMATE_SEASONAL_PERIOD = True
# Automatic period estimation range (min_period ~ n * max_period_ratio)
MIN_AUTO_SEASONAL_PERIOD = 2
MAX_AUTO_SEASONAL_PERIOD_RATIO = 1.0 / 3.0

# Statistical tests: True = statsmodels must be installed; False = skip ADF/KPSS when unavailable
REQUIRE_STAT_TESTS = True

# ── compute_metrics window/threshold parameters ──
# All window parameters act as FLOORS (minimums). get_effective_lags() scales them
# up with series length n, using n-based fractions as ceilings (capped at n//4).
# This ensures reasonable values from n=800 to n=50000+.
#
# Scaling rules (when SEASONAL_PERIOD=0):
#   trend_win    = min(n//4, max(config, n//10))   -> ~10% of n for long series
#   roll_var_win = min(n//4, max(config, n//20))   ->  ~5% of n for long series
#   cp_win       = min(n//4, max(config, n//20))   ->  ~5% of n for long series
#   cross_lag    = min(48,   max(config, n//40))   -> capped at 48
#   acf_lag      = min(n//4, max(config, n//10))   -> ~10% of n, capped at 2000
#
# ACF maximum lag; floor value, scales with n via get_effective_lags
MAX_ACF_LAG = 48
# Maximum lag for multivariate cross-correlation; floor value, absolute cap at 48
MAX_CROSS_LAG = 24
# Correlation network threshold: absolute correlation above this value is treated as an edge
CORR_THRESHOLD = 0.30
# Multivariate pair sampling cap
MAX_CROSS_PAIRS = 100

# Sample entropy / permutation entropy window parameters (fixed window size;
# entropy is estimated locally within each window, not scaled with n)
ENTROPY_WINDOW = 200               # Window length for entropy (fixed, ~n/4 at n=800)
MAX_ENTROPY_WINDOWS = 8            # Number of evenly-spaced windows to sample
MAX_ENTROPY_POINTS = 144           # Max points subsampled within one entropy window

# Spectrum high-frequency threshold: periods <= this value are considered high-frequency
HIGH_FREQ_MAX_PERIOD = 8

# Rolling windows — floor values, scale up with n via get_effective_lags
TREND_WINDOW = 101                 # Trend rolling-mean window floor (~n/8 at n=800)
ROLL_VAR_WINDOW = 51               # Rolling variance window floor (~n/16 at n=800)
CHANGEPOINT_WINDOW = 51            # Changepoint detection window floor

# ===== 3. Task Hyperparameters (env vars override defaults) =====
TIME_LAG = int(os.environ.get("TIME_LAG", "12"))
PRE_LEN = int(os.environ.get("PRE_LEN", "3"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "16"))
EPOCHS = int(os.environ.get("EPOCHS", "1000"))
LR = float(os.environ.get("LR", "0.001"))
WEIGHT_DECAY = float(os.environ.get("WEIGHT_DECAY", "0"))
EARLY_STOP_PATIENCE = int(os.environ.get("EARLY_STOP_PATIENCE", "15"))
#  UNIVARIATE = True → univariate mode. The model uses only the 0th variable
UNIVARIATE = os.environ.get("UNIVARIATE", "1") == "1"
TRAIN_RATIO = float(os.environ.get("TRAIN_RATIO", "0.6"))
VAL_RATIO = float(os.environ.get("VAL_RATIO", "0.2"))
TEST_RATIO = float(os.environ.get("TEST_RATIO", "0.2"))

# Supported types: 'mse_mae' or 'huber'
TRAIN_LOSS_TYPE = "mse_mae"
MSE_LOSS_WEIGHT = 0.2
MAE_LOSS_WEIGHT = 0.8
HUBER_BETA = 0.5

# ===== 4. Result Output Directory =====
# Each npy's experimental results are stored independently under result/<npy_basename>/.
# BASE_RESULT_DIR / RESULT_DIR / RUN_ID are dynamically injected by run_iterative_loop.py
# via cfg_overrides. The following are static defaults (used when manually running
# LLM_with_RAG_iterative.py as a single run).
_NPY_BASENAME_DEFAULT = os.path.splitext(os.path.basename(NPY_PATH))[0]
BASE_RESULT_DIR = os.path.join(RESULT_ROOT, _NPY_BASENAME_DEFAULT)
RUN_ID = os.getenv("EXPERIMENT_RUN_ID")
if not RUN_ID:
    RUN_ID = datetime.now().strftime("run_%Y%m%d_%H%M%S")
RESULT_DIR = os.path.join(BASE_RESULT_DIR, RUN_ID)

# Each iteration is saved separately under RESULT_DIR/iter_001, iter_002, ...
# run_iterative_loop.py places LLM input, LLM output, training, testing, and feedback into
# subdirectories, avoiding repeated copying of global runtime/latest files.
ITERATION_DIR_TEMPLATE = "iter_{:03d}"
LLM_INPUT_DIR_NAME = "llm_input"
LLM_OUTPUT_DIR_NAME = "llm_output"
FEEDBACK_DIR_NAME = "feedback"
TRAIN_DIR_NAME = "train"
TEST_DIR_NAME = "test"

# RUNTIME_DIR is reserved for manual single-run usage;
# during formal iteration, run_iterative_loop.py overrides it to iter_XXX/llm_input.
RUNTIME_DIR = os.path.join(RESULT_DIR, "_manual_runtime")
CANDIDATE_MODEL_PATH = os.path.join(RUNTIME_DIR, "candidate_model.py")
SELECTION_REASON_SAVE_PATH = os.path.join(RUNTIME_DIR, "module_selection_reason.md")
RESET_LATEST_FEEDBACK_ON_START = True
OVERWRITE_EXISTING_ITER_DIR = True
AUTO_REPAIR_CANDIDATE_SYNTAX = True

# ===== 5. Per-Iteration Result Filenames =====
ITERATION_HISTORY_FILE = "iteration_history.json"
GENERATED_MODEL_FILE = "candidate_model.py"
MODULE_REASON_FILE = "module_selection_reason.md"
LLM_PROMPT_FILE = "llm_prompt.txt"
LLM_RAW_OUTPUT_FILE = "llm_raw_output.txt"
LLM_INPUT_MANIFEST_FILE = "llm_input_manifest.json"
FEEDBACK_SUMMARY_FILE = "feedback_summary.json"
STOP_JUDGEMENT_FILE = "stop_judgement.json"
BEST_MODEL_FILE = "best_model_weights.pth"
TRAIN_STATS_FILE = "train_standardization_stats.npz"
TRAIN_HISTORY_FILE = "training_loss.csv"
TEST_METRICS_FILE = "test_metrics.json"
FINAL_PREDS_FILE = "averaged_final_predictions.npy"
FINAL_TARGETS_FILE = "averaged_final_targets.npy"
PREDICTION_PLOT_PREFIX = "prediction_vs_truth_feature"
ITERATION_HISTORY_PATH = os.path.join(RESULT_DIR, ITERATION_HISTORY_FILE)
LATEST_TEST_METRICS_PATH = os.path.join(RUNTIME_DIR, "latest_feedback_test_metrics.json")
LATEST_TRAIN_HISTORY_PATH = os.path.join(RUNTIME_DIR, "latest_feedback_training_loss.csv")
LATEST_BEST_MODEL_PATH = os.path.join(RUNTIME_DIR, "latest_feedback_best_model_weights.pth")
LATEST_TRAIN_STATS_PATH = os.path.join(RUNTIME_DIR, "latest_feedback_train_standardization_stats.npz")

# ===== 6. Testing, Evaluation Metrics, and Plotting Configuration =====
EVALUATE_ON_NORMALIZED_DATA = False
TEST_PLOT_POINTS = 576
TEST_PLOT_MAX_FEATURES = 5
SAVE_PREDICTION_PLOTS = True
TEST_PLOT_DPI = 300

# ===== 7. Module Library =====
# Paths relative to _INTERNAL_DIR (defined above) for portability
MODULE_LIBRARY_PATH = os.path.join(_INTERNAL_DIR, "module_set", "module_library_candidates.json")
SEARCH_GRAMMAR_PATH = os.path.join(_INTERNAL_DIR, "module_set", "search_grammar_constraints.json")

# ===== 8. Iteration Control (env vars override defaults) =====
MAX_NPY_FILES = int(os.environ.get("MAX_NPY_FILES", "0"))
MAX_ITERATIONS = int(os.environ.get("MAX_ITERATIONS", "100"))
MIN_ITERATIONS_BEFORE_STOP = int(os.environ.get("MIN_ITERATIONS_BEFORE_STOP", "10"))
ITERATION_PATIENCE = int(os.environ.get("ITERATION_PATIENCE", "8"))

# ── Early-stopping primary metric ──
# Early stopping is based on a SINGLE primary metric (scientific standard: the search
# should continue as long as the primary objective is improving). The composite score
# (COMPOSITE_SCORE_WEIGHTS) is computed and logged for reference only; it does NOT
# gate the stop decision. Model ranking uses TOP_K_SORT_METRIC instead.
# Valid keys: "val_rmse", "val_mae", "val_loss" (validation-driven selection),
# or "overall_mae"/"overall_rmse"/"overall_wmape" (test-driven, NOT recommended —
# causes test-set selection bias). Default uses validation per Nature rigor.
STOP_PRIMARY_METRIC = "val_rmse"

# Minimum relative improvement on the primary metric to count as "better".
# e.g. 1e-4 means at least 0.01% improvement over the current best.
STOP_PRIMARY_MIN_REL_IMPROVEMENT = 1e-4

# ── Composite score weights (for reporting / analysis only, not for early stopping) ──
# NOTE: raw weights need NOT sum to 1 — compute_composite_score() normalizes by the
# sum of active weights (weighted_score / total_weight), so the composite is scale-free.
COMPOSITE_SCORE_WEIGHTS = {
    "overall_mae": 1.0,
    "overall_rmse": 0.0,
    "overall_wmape": 0.3,
    "overall_smape": 0.0,
    "bias_abs": 0.3,
    "residual_std": 0.0,
    "lag1_autocorr_abs": 0.2,
    "residual_target_corr_abs": 0.2,
}

# ===== 9. Top-K Candidate Model Recording =====
TOP_K = int(os.environ.get("TOP_K", "5"))
TOP_K_SORT_METRIC = "val_rmse"      # Selection metric: val_rmse / val_mae / val_loss (validation-driven)
TOP_K_SORT_ASCENDING = True         # True = ascending (lower is better)
TOP_K_CSV_FILENAME = "topk_candidates.csv"
ALL_CANDIDATES_CSV_FILENAME = "all_candidates.csv"

# Dedup Top-K by architecture signature (the A/B/C/D/E module composition).
# Only the best candidate (by TOP_K_SORT_METRIC) per signature is kept, so the
# Top-K list reflects diverse architectures instead of near-duplicate variants
# of one architecture that differ only in hyperparameters.
TOP_K_DEDUP_BY_ARCH = True

# Inject the list of already-tried architecture signatures into the LLM prompt
# and forbid it from proposing a candidate with a matching signature (steers
# generation away from near-duplicate architectures at the source).
LLM_AVOID_DUPLICATE_SIGNATURE = True

# ===== 9.1 Two-Phase Experiment: Search Seed + Stability Re-run =====
# Phase 1 search stage uses a fixed random seed (affects model initialization, DataLoader shuffle)
SEARCH_SEED = int(os.environ.get("SEARCH_SEED", "2026"))
# Phase 2 stability re-run seed list (same model retrained with different seeds)
STABILITY_SEEDS = [2026, 2027, 2028]
# Phase 2 re-run the Top-K candidate models
STABILITY_TOP_K = 3
# Phase 3 stability results CSV filename
STABILITY_CSV_FILENAME = "stability_topk_results.csv"

# ===== 9.2 Per-Step Output Directories (set dynamically by run_iterative_loop.py via cfg_overrides) =====
# Training step output directory; defaults to a manual_run path when not overridden
TRAIN_OUTPUT_DIR = os.path.join(RESULT_DIR, "manual_run", "train")
# Testing step output directory; defaults to a manual_run path when not overridden
TEST_OUTPUT_DIR = os.path.join(RESULT_DIR, "manual_run", "test")
# Path to trained best model weights for the test step to load
MODEL_WEIGHTS_PATH = os.path.join(TRAIN_OUTPUT_DIR, BEST_MODEL_FILE)
# Path to training standardization stats for the test step to load
TRAIN_STATS_NPZ_PATH = os.path.join(TRAIN_OUTPUT_DIR, TRAIN_STATS_FILE)

META_FEATURE_NAMES_ORDERED = [
    "distribution_shape_score",
    "trend_strength_score",
    "stationarity_score",
    "periodicity_score",
    "temporal_dependence_score",
    "noise_volatility_score",
    "multivariate_structure_score",
]

# ===== 10. Apply the selected AI profile =====
def configure_llm(provider):
    """Select model, endpoint and credentials together; openai aliases chatgpt."""
    global LLM_PROVIDER, LLM_API_KEY, LLM_BASE_URL, LLM_MODEL
    provider = "chatgpt" if provider == "openai" else provider
    if provider not in _LLM_PROFILES:
        raise ValueError(f"Unknown AI: {provider}; choices: {', '.join(AI_TYPES)}")
    profile = _LLM_PROFILES[provider]
    LLM_PROVIDER = provider
    LLM_API_KEY = profile["api_key"].strip() or os.environ.get(profile["api_key_env"], "").strip()
    LLM_BASE_URL = profile["base_url"]
    LLM_MODEL = profile["model"]
    # Preserve the original DeepSeek environment variables without applying them to other AIs.
    if provider == "deepseek":
        LLM_API_KEY = LLM_API_KEY or os.environ.get("LLM_API_KEY", "").strip()
        LLM_BASE_URL = os.environ.get("LLM_BASE_URL", LLM_BASE_URL)
        LLM_MODEL = os.environ.get("LLM_MODEL", LLM_MODEL)


configure_llm(LLM_PROVIDER)
LLM_REASONING_EFFORT = os.environ.get("LLM_REASONING_EFFORT", "medium")
LLM_TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", "0.4"))
LLM_STREAMING = os.environ.get("LLM_STREAMING", "1") == "1"
LLM_JSON_RETRY_MAX = int(os.environ.get("LLM_JSON_RETRY_MAX", "3"))
HTTP_PROXY = os.environ.get("HTTP_PROXY", "")
HTTPS_PROXY = os.environ.get("HTTPS_PROXY", "")


def apply_proxy_settings():
    """Apply proxy settings to the current Python process; if the configuration is empty, clear any existing proxies."""
    proxy_keys = [
        "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
        "http_proxy", "https_proxy", "all_proxy",
    ]

    if HTTP_PROXY or HTTPS_PROXY:
        if HTTP_PROXY:
            os.environ["HTTP_PROXY"] = os.environ["http_proxy"] = HTTP_PROXY
        if HTTPS_PROXY:
            os.environ["HTTPS_PROXY"] = os.environ["https_proxy"] = HTTPS_PROXY
    else:
        for key in proxy_keys:
            os.environ.pop(key, None)
        os.environ["NO_PROXY"] = os.environ["no_proxy"] = "*"
