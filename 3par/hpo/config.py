"""Load tuning settings, input tasks, and run paths while keeping API keys in memory."""

import copy
import os
import re
from pathlib import Path
from urllib.parse import urlsplit

from .artifacts import read_json
from .losses import DEFAULT_LOSS_PROFILE, LOSS_PROFILES

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
FEATURES_DIR = PROJECT_ROOT / "0dataset" / "testmetrics"
MAX_TRAIN_EPOCHS = 300
EARLY_STOP_PATIENCE = 15


def feature_path(dataset, directory=FEATURES_DIR):
    directory = Path(directory)
    flat = directory / dataset / "meta_features.csv"
    if flat.is_file():
        return flat
    matches = sorted(directory.glob(f"*/{dataset}/meta_features.csv"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one feature file for {dataset} under {directory}; found {len(matches)}")
    return matches[0]


def load_settings(path=None, smoke=False):
    settings = read_json(path or ROOT / "settings.json")
    if settings["time_lag"] != 12 or settings["horizons"] != [3, 12]:
        raise ValueError("The default input window must be 12 and prediction horizons must remain [3,12]")
    if "time_lag" in settings["space"]:
        windows = settings["space"]["time_lag"]
        if windows != [12, 32, 64] or any(type(value) is not int for value in windows):
            raise ValueError("space.time_lag must be [12,32,64]; the default input window remains 12")
        if settings["search"]["probe_values"].get("time_lag") != [32, 64]:
            raise ValueError("Input-window probe values must be [32,64]")
    if type(settings["space"].get("enforce_param_ratio", True)) is not bool:
        raise ValueError("space.enforce_param_ratio must be a boolean")
    if not 1 <= settings["search"]["trials"] <= 32:
        raise ValueError("Screening search allows at most 32 slots")
    two_seed_screening = "screening_seeds" in settings["search"]
    if two_seed_screening:
        if settings["search"]["screening_seeds"] != [2026, 2027]:
            raise ValueError("Screening seeds must be 2026 and 2027")
        if settings["search"]["confirmation_seeds"] != [2028]:
            raise ValueError("Confirmation seed must be 2028 after two-seed screening")
        if settings["search"].get("confirmation_top_k") != 4:
            raise ValueError("Two-seed screening confirms the Top 4 plus the default configuration")
    elif settings["search"]["confirmation_seeds"] != [2027, 2028]:
        raise ValueError("Legacy single-seed runs require confirmation seeds 2027 and 2028")
    if settings["training"]["search_train_seed"] != 2026:
        raise ValueError("The first screening seed for the baseline and all tuning slots must be 2026")
    if settings["training"]["final_seeds"] != [2026, 2027, 2028]:
        raise ValueError("Final evaluation seeds must be 2026, 2027, and 2028")
    if (settings["training"]["short_epochs"], settings["training"]["short_patience"]) != (MAX_TRAIN_EPOCHS, EARLY_STOP_PATIENCE):
        raise ValueError("Formal runs require 300 maximum epochs and patience 15; neither is tunable")
    if (settings["training"]["mse_weight"], settings["training"]["mae_weight"]) != (0.2, 0.8):
        raise ValueError("Default training loss weights must remain 0.2*MSE + 0.8*MAE")
    losses = settings["space"].get("loss_profile")
    if "loss_profile" in settings["space"]:
        if (not isinstance(losses, list) or not losses
                or any(not isinstance(value, str) or value not in LOSS_PROFILES for value in losses)
                or len(losses) != len(set(losses)) or DEFAULT_LOSS_PROFILE not in losses):
            raise ValueError("space.loss_profile must contain unique supported profiles and include mae_heavy")
        probes = settings["search"]["probe_values"].get("loss_profile", [])
        if (not isinstance(probes, list) or any(value not in losses for value in probes)
                or len(probes) != len(set(probes))):
            raise ValueError("loss_profile probes must be unique values from space.loss_profile")
    if (settings["train_ratio"], settings["val_ratio"]) != (0.6, 0.2):
        raise ValueError("This protocol requires a 60%/20%/20% chronological split")
    settings["smoke"] = bool(smoke)
    if smoke:
        settings = copy.deepcopy(settings)
        settings["training"].update(short_epochs=1, short_patience=1, device="cpu")
        settings["smoke_prefix_points"] = 256
    agent = settings["agent"]
    # Read the endpoint and model only from settings to avoid legacy environment overrides.
    options = agent["request_options"]
    allowed = {"temperature", "max_tokens", "max_completion_tokens", "response_format"}
    if set(options) - allowed:
        raise ValueError("request_options may contain only temperature, output limits, and JSON format")
    if agent["max_attempts"] not in (1, 2) or agent["max_changed_parameters"] != 2:
        raise ValueError("This protocol permits at most two response-validation attempts and two changed parameters")
    retries = agent.get("max_retries", 3)
    if type(retries) is not int or not 0 <= retries <= 3:
        raise ValueError("agent.max_retries must be an integer between 0 and 3")
    if agent["base_url"]:
        parsed = urlsplit(agent["base_url"])
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("base_url must not contain credentials, query parameters, or fragments")
        if parsed.scheme != "https" and not (parsed.scheme == "http" and parsed.hostname in {"localhost", "127.0.0.1"}):
            raise ValueError("A remote LLM endpoint must use HTTPS")
    return settings


def require_llm(settings, mock=False, api_key=None):
    """Validate and return an in-memory API key without adding it to saved settings."""
    if mock:
        if not settings["smoke"]:
            raise ValueError("The mock agent is allowed only with --smoke")
        return
    if not settings["agent"]["model"] or not settings["agent"]["base_url"]:
        raise ValueError("Check agent.model and agent.base_url in settings.json; training has not started")
    key = (api_key if api_key is not None else os.environ.get("DEEPSEEK_API_KEY", "")).strip()
    if not key:
        raise ValueError("Provide --api-key or set DEEPSEEK_API_KEY; training has not started")
    return key


def tasks_from_manifest(root=ROOT):
    return read_json(Path(root) / "tasks.json")["tasks"]


def select_tasks(tasks, datasets="first1", horizons=None):
    names = sorted({task["dataset"] for task in tasks})
    if ":" in datasets:
        match = re.fullmatch(r"([0-9]+):([0-9]+)", datasets)
        if match is None:
            raise ValueError("Write dataset ranges as start:end, for example --datasets 3:10")
        start, end = map(int, match.groups())
        if not 1 <= start <= end <= len(names):
            raise ValueError(f"Dataset indices must be within 1..{len(names)}, with start <= end")
        selected = set(names[start - 1:end])
    elif datasets.startswith("first") and datasets[5:].isdigit():
        selected = set(names[:int(datasets[5:])])
    elif datasets == "all":
        selected = set(names)
    else:
        selected = set(datasets.split(","))
        if selected - set(names):
            raise ValueError(f"Unknown datasets: {sorted(selected - set(names))}")
    chosen = [task for task in tasks if task["dataset"] in selected
              and (horizons is None or task["horizon"] in horizons)]
    if not chosen:
        raise ValueError("No matching final L model tasks")
    return chosen


def run_folder(name, root=ROOT):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
        raise ValueError("Run names may contain only ASCII letters, digits, underscores, and hyphens")
    if name.lower() == "checks":
        raise ValueError("The reserved name 'checks' cannot be used for an experiment run")
    return Path(root) / "outputs" / name
