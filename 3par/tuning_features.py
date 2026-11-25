"""Read seven user-provided meta-features without recomputing or relabeling them."""

import csv
import math
from pathlib import Path

FEATURE_NAMES = (
    "distribution_shape", "trend_strength", "stationarity", "periodicity",
    "temporal_dependence", "noise_volatility", "multivariate_structure",
)


def load_features(path):
    path = Path(path).resolve()
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows, fields = list(reader), set(reader.fieldnames or [])
    if {"indicator_name", "value"} <= fields:
        pairs = [(row["indicator_name"], row["value"]) for row in rows]
    elif len(rows) == 1:
        pairs = list(rows[0].items())
    else:
        raise ValueError(f"Meta-feature CSV must be a long indicator_name/value table or one wide row: {path}")
    values = {}
    for name, value in pairs:
        name = name.strip().removesuffix("_score")
        if name not in FEATURE_NAMES or name in values:
            raise ValueError(f"Unknown or duplicate meta-feature name: {name}")
        number = float(value)
        if not math.isfinite(number) or not 0 <= number <= 1:
            raise ValueError(f"Meta-feature must be finite and within [0, 1]: {name}")
        values[name] = number
    if set(values) != set(FEATURE_NAMES):
        raise ValueError(f"Missing meta-feature fields: {sorted(set(FEATURE_NAMES) - set(values))}")
    return {"values": {name: values[name] for name in FEATURE_NAMES},
            "source": str(path),
            "scope": "user_provided_train_only_unverified",
            "note": "User-provided dataset descriptors. Training-only provenance is unverified; they may include validation/test-period information. This run is exploratory, not a certified leakage-free external evaluation."}
