# -*- coding: utf-8 -*-
"""
Shared infrastructure for the L, F, R and LLM experiments.
==========================================
Discover datasets, read meta-features, select tasks and resume recorded runs.
Each entry point defines"how architectures are generated"and reuses this module for dataset discovery and result recording.
"""

import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_TOP = _HERE.parent
for _p in (_TOP, _HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import config  # noqa: E402


EXCLUDED_DATASETS = {
    "0253_birminghamparking_BHMBCCPST01_occupancy_3332_1",
}


# ---------------------------------------------------------------------------
# Dataset discovery and meta-features
# ---------------------------------------------------------------------------

def discover_datasets():
    """Return a list of dictionaries with name, npy_path, data_type, station_num and meta_csv.
    Scan the test directory, including flat NPY files and nested domain/type directories.
    Include only datasets with a matching meta_features.csv."""
    rows = []
    flat_npys = sorted(config.TEST_DATA_ROOT.glob("*.npy"))
    for npy in flat_npys:
        if npy.stem not in EXCLUDED_DATASETS:
            rows.append({"name": npy.stem, "npy_path": str(npy),
                         "data_type": None})
    # Current layout: test/univariate/*.npy and test/multivariate/*.npy.
    category_dirs = []
    for mode_name, data_type in (("univariate", "uni"), ("multivariate", "multi"),
                                 ("\u5355\u53d8\u91cf", "uni"), ("\u591a\u53d8\u91cf", "multi")):
        mode = config.TEST_DATA_ROOT / mode_name
        if mode.is_dir():
            category_dirs.append(mode)
            for npy in sorted(mode.glob("*.npy")):
                if npy.stem not in EXCLUDED_DATASETS:
                    rows.append({"name": npy.stem, "npy_path": str(npy),
                                 "data_type": data_type})
    for domain in sorted(config.TEST_DATA_ROOT.iterdir()):
        if not domain.is_dir() or not domain.name[0].isdigit():
            continue
        # Preserve the original traversal order, with univariate series first.
        for mode in sorted(domain.iterdir(), key=lambda p: (
                not ("\u5355" in p.name or p.name.lower() == "univariate"), p.name)):
            if not mode.is_dir():
                continue
            data_type = "uni" if ("\u5355" in mode.name or mode.name.lower() == "univariate") else "multi"
            for npy in sorted(mode.glob("*.npy")):
                if npy.stem in EXCLUDED_DATASETS:
                    continue
                rows.append({"name": npy.stem, "npy_path": str(npy),
                             "data_type": data_type})
    matched = []
    for r in rows:
        mf = config.TEST_FEAT_ROOT / r["name"] / "meta_features.csv"
        if not mf.exists():
            relative_parent = Path(r["npy_path"]).parent.relative_to(config.TEST_DATA_ROOT)
            mf = config.TEST_FEAT_ROOT / relative_parent / r["name"] / "meta_features.csv"
        if not mf.exists():
            continue
        a = np.load(r["npy_path"])
        station_num = a.shape[1] if a.ndim == 2 else 1
        if r["data_type"] is None:
            r["data_type"] = "uni" if station_num == 1 else "multi"
        r["station_num"] = int(station_num)
        r["meta_csv"] = str(mf)
        matched.append(r)
    if flat_npys or category_dirs:
        # The first two digits identify the domain. Preserve domain, series type and filename order.
        matched.sort(key=lambda r: (r["name"][:2], r["data_type"] != "uni", r["name"]))
    return matched


def select_datasets(all_ds, spec="first10", limit=0, range_slice=(0, 0)):
    """
    Select a subset of datasets. Accepted specifications:
        "first<N>"  : first N names in sorted order (default "first10")
        "all"       : all datasets
        "a,b,c"     : comma-separated dataset names
    Apply limit and range_slice after selection for parallel shards.
    """
    if spec and spec.startswith("first"):
        try:
            n = int(spec[len("first"):])
        except ValueError:
            n = 10
        ds = sorted(all_ds, key=lambda d: d["name"])[:n]
    elif spec == "all":
        ds = list(all_ds)
    else:
        names = {x.strip() for x in (spec or "").split(",") if x.strip()}
        ds = [d for d in all_ds if d["name"] in names]
        ds = sorted(ds, key=lambda d: d["name"])
    if limit:
        uni = [d for d in ds if d["data_type"] == "uni"]
        multi = [d for d in ds if d["data_type"] == "multi"]
        n_uni = min(len(uni), int(np.ceil(limit * 0.7)))
        n_multi = limit - n_uni
        ds = uni[:n_uni] + multi[:n_multi]
        ds = sorted(ds, key=lambda d: d["name"])
    if range_slice[1] > 0:
        ds = ds[range_slice[0]:range_slice[1]]
    return ds


def load_meta(meta_csv):
    df = pd_read_csv(meta_csv)
    return dict(zip(df["indicator_name"], df["value"]))


def pd_read_csv(path):
    import pandas as pd
    return pd.read_csv(path, encoding="utf-8-sig")


def ds_base(ds):
    return {"dataset": ds["name"], "data_type": ds["data_type"],
            "station_num": ds["station_num"]}


# ---------------------------------------------------------------------------
# Result recording and resumption
# ---------------------------------------------------------------------------

COLUMNS = ["dataset", "data_type", "station_num", "horizon", "arch",
           "seed", "module_A", "module_B", "module_C", "module_E",
           "rmse", "mae", "wmape", "params"]


def spec_modules(spec):
    """Extract the selected motifs at each stage for the results CSV.
    Represent an empty preprocessing stage A as "Identity"to match the explicit no-op in the motif library.
    This gives empty preprocessing and Identity the same frequency count.
    """
    a = spec.get("A") or []
    b = spec.get("B") or []
    return {
        "module_A": "+".join(a) if a else "Identity",
        "module_B": "+".join(b),
        "module_C": spec.get("C") or "",
        "module_E": spec.get("E") or "",
    }


def append_result(csv_path, row):
    import pandas as pd
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    new = pd.DataFrame([row])
    if csv_path.exists():
        new.to_csv(csv_path, mode="a", header=False, index=False,
                   encoding="utf-8-sig")
    else:
        new.to_csv(csv_path, index=False, encoding="utf-8-sig")


def already_done(run_dir, tag):
    """Check whether training and testing completed for this dataset, horizon, architecture and seed."""
    return (Path(run_dir) / tag / "test" / "test_metrics.json").exists()


def l_param_target(dataset, horizon):
    """Read the selected L parameter count as the capacity target for F and R.
    As in exp_LLM._l_param_target, each dataset and horizon must have a unique L parameter count.
    Return None if L has yet to record the task so a parallel worker can skip it for now."""
    path = config.RESULT_ROOT / "exp_L.csv"
    if not path.exists():
        raise FileNotFoundError("Run L first: exp_L.csv is missing")
    df = pd_read_csv(path)
    rows = df[(df["dataset"] == dataset) & (df["horizon"] == horizon)]
    vals = rows["params"].dropna().unique()
    if len(vals) == 0:
        return None
    if len(vals) != 1:
        raise ValueError(f"{dataset} h{horizon} has inconsistent L parameter counts: {vals.tolist()}")
    return int(vals[0])


def common_argparser(desc):
    """Shared arguments for datasets, horizons, seeds, limits and parallel ranges."""
    import argparse
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("--datasets", type=str, default="first10",
                        help="first<N> selects the first N names (default first10); all selects every dataset;"
                             "or supply comma-separated dataset names")
    parser.add_argument("--horizons", type=int, nargs="+", default=None,
                        help="Forecast horizons (default config.HORIZONS=[3,12])")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="Random seeds (default [2026,2027,2028])")
    parser.add_argument("--limit", type=int, default=0,
                        help="Select at most N datasets, balancing univariate and multivariate series where possible")
    parser.add_argument("--range", type=int, nargs=2, default=[0, 0],
                        help="Process selected indices [K, M) with --limit for parallel shards")
    return parser
