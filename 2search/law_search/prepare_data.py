# -*- coding: utf-8 -*-
"""Data preparation and shared encoding. Run the complete pipeline with main.py.

Preserve analysis units, performance weights, motif baselines and feature encodings from stages 00-02.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import config as law_config


RESULTS_DIR = law_config.INPUT_DIR
RAW_CSV_NAMES = [
    "12_3_u.csv", "12_3_m.csv", "12_12_u.csv", "12_12_m.csv",
]
RAW_STAGE_COLS = {
    "A": "A_Input_Preprocessing",
    "B": "B_Backbone_Modeling",
    "C": "C_Head_Projection",
    "E": "E_Combination_Paradigm",
}
WEIGHT_MODES = {
    "perf": {"col": "mean_RMSE", "r_col": "regret_perf", "w_col": "w_perf", "log": False},
}
W_COL = "w_perf"
FEATURES = [
    "distribution_shape_score",
    "trend_strength_score",
    "stationarity_score",
    "periodicity_score",
    "temporal_dependence_score",
    "noise_volatility_score",
    "multivariate_structure_score",
]
STAGE_COLS = {
    "A": "module_A",
    "B": "module_B",
    "C": "module_C",
    "E": "module_E",
}
STAGE_MODULES = {
    "A": ["Identity", "Z-score", "RevIN", "Differencing",
          "Series Decomposition", "Patch", "Mask"],
    "B": ["Linear Block", "MLP Block", "CNN Block", "TCN Block", "GRU Block",
          "LSTM Block", "State Space Model Block", "Transformer Encoder Block", "Basis Block"],
    "C": ["Linear Head", "MLP Head", "Basis Projection"],
    "E": ["Single Block", "Sequential Composition", "Parallel Additive Branches",
          "Residual Modeling", "Gated Fusion"],
}
FEATURE_CN = {
    "distribution_shape_score": "Distribution shape",
    "trend_strength_score": "Trend",
    "stationarity_score": "Stationarity",
    "periodicity_score": "Periodicity",
    "temporal_dependence_score": "Temporal dependence",
    "noise_volatility_score": "Noise",
    "multivariate_structure_score": "Multivariate structure",
}
DESIGN_FEATURES = ["data_type", "config"]


def build_binary(units, stage, module, target):
    """
    Return a binary motif indicator for each row of units.
    stage   : "A"/"B"/"C"/"E"
    module: motif name, for example "Mask"
    target  :
        - "contains" : 1 if the composition contains the motif ("Mask, Patch" counts as 1 for both Mask and Patch).
    """
    col = STAGE_COLS[stage]                       # Motif column, e.g. module_B
    values = units[col].fillna("").astype(str)    # Fill missing values before converting to strings.
    return values.str.contains(module, regex=False).astype(int)


def build_feature_matrix(units):
    """
    Return the nine-column model feature DataFrame aligned to units:
      - First seven columns: original meta-feature scores in [0, 1].
      - data_type: univariate=0, multivariate=1.
      - config: 12_3=0 / 12_12=1.
    """
    X = units[FEATURES].astype(float).copy()                # Original seven scores
    X["data_type"] = (units["data_type"] == "multi").astype(float)   # 0/1
    X["config"] = (units["config"] == "12_12").astype(float)         # 0/1
    return X


def rule_mask(units, conditions, X=None):
    """
    Return a Boolean mask for rows satisfying the condition list.
    conditions = [(feature name, direction (">"or"≤"), threshold), ...]
    Pass a precomputed X to avoid rebuilding the feature matrix.
    """
    if X is None:
        X = build_feature_matrix(units)
    mask = np.ones(len(units), dtype=bool)
    for feat, direction, thr in conditions:
        x = X[feat].values
        mask &= (x > thr) if direction == ">" else (x <= thr)
    return mask


def parse_conditions(s):
    """Parse the serialized stage 03 conditions "feat:direction:threshold|..." into [(feat, direction, threshold)]."""
    if not isinstance(s, str) or not s:
        return []
    return [(f, d, float(t)) for part in s.split("|")
            for f, d, t in [part.split(":")]]


def serialize_conditions(conditions):
    """[(feat, direction, threshold)] -> "feat:direction:threshold|..." string for stage 04 and later."""
    return "|".join(f"{f}:{d}:{t}" for f, d, t in conditions)


def _parse_name(csv_name):
    """
    Parse (config, data_type) from a CSV filename.
    Example:"12_3_u.csv"  -> ("12_3", "uni")
        "12_12_m.csv" -> ("12_12", "multi")
    """
    base = csv_name.replace(".csv", "")
    prefix, mode_char = base.rsplit("_", 1)
    data_type = "uni" if mode_char == "u" else "multi"
    return prefix, data_type


def load_raw_csvs(ai_type=None, input_dir=None):
    """
    Read four source CSVs for one provider or return standardized filenames for a pooled analysis.
    Derive the config and data_type columns from each filename.
    Keep source_ai in pooled data and preserve Dataset to avoid double-counting datasets.
    """
    ai_type = ai_type or law_config.AI_TYPES[0]
    input_dir = Path(input_dir) if input_dir is not None else RESULTS_DIR
    loaded = {}
    for name in RAW_CSV_NAMES:
        cfg, data_type = _parse_name(name)
        suffix = "univariate" if data_type == "uni" else "multivariate"
        if ai_type == "combined":
            parts = []
            for source_ai in law_config.AI_TYPES:
                path = input_dir / f"{source_ai}_{cfg}_{suffix}.csv"
                part = pd.read_csv(path, encoding="utf-8-sig")
                part["source_ai"] = source_ai
                parts.append(part)
            df = pd.concat(parts, ignore_index=True)
            # Keep nine rows per dataset and configuration contiguous for block permutations.
            df = df.sort_values("Dataset", kind="stable", ignore_index=True)
            # Preserve each experiment's features. Provider identity is retained only for provenance.
            grouped = df.groupby("Dataset")[FEATURES]
            spread = grouped.max() - grouped.min()
            n_mismatch = int(spread.gt(1e-12).any(axis=1).sum())
            if n_mismatch:
                print(f"  [WARN] {name}: {n_mismatch} datasets have feature differences across providers;"
                      "keeping each provider's original feature values.")
        else:
            path = input_dir / f"{ai_type}_{cfg}_{suffix}.csv"
            # utf-8-sig strips the initial BOM from the first column name.
            df = pd.read_csv(path, encoding="utf-8-sig")
        cfg, data_type = _parse_name(name)
        df["config"] = cfg
        df["data_type"] = data_type
        loaded[name] = df
        print(f"  {name}: {len(df)} rows ({df['Dataset'].nunique()} datasets, config={cfg}, data_type={data_type})")
    return loaded


def check_feature_consistency(csvs):
    """
    The seven features of a dataset should agree between the 12_3 and 12_12 CSVs
    because they describe the dataset independently of forecast horizon.
    Report the mismatch count for inspection; zero indicates agreement.
    """
    mismatch = 0
    for data_type in ("uni", "multi"):
        a_name = f"12_3_{'u' if data_type == 'uni' else 'm'}.csv"
        b_name = f"12_12_{'u' if data_type == 'uni' else 'm'}.csv"
        # Each dataset has three ranked rows with identical meta-features.
        # Keep the first row per dataset so indexing returns scalar feature values.
        a = csvs[a_name].drop_duplicates(subset="Dataset").set_index("Dataset")
        b = csvs[b_name].drop_duplicates(subset="Dataset").set_index("Dataset")
        for ds in a.index.intersection(b.index):
            for feat in FEATURES:
                va, vb = a.loc[ds, feat], b.loc[ds, feat]
                if pd.isna(va) or pd.isna(vb) or not abs(float(va) - float(vb)) < 1e-12:
                    mismatch += 1
                    print(f"    [Warning] Feature mismatch for dataset={ds} feature={feat} {va} vs {vb}")
    if mismatch == 0:
        print("  [OK] Meta-features agree between configurations for every dataset")
    else:
        print(f"  [Warning] Found {mismatch} feature mismatches across configurations. Inspect the source data.")
    return mismatch


def build_units(csvs):
    """
    Build 1,578 analysis rows from four CSVs (263 datasets x two configurations x top three).
    unit_id groups Dataset and config into 526 units.

    Retain every original CSV column:
      - Structure: Dataset, Stability_Rank, TIME_LAG, PRE_LEN, Candidate_Model_Dir, ...
      - Seven meta-features shared by the three rows of a unit
      - Metrics: Selection_val_RMSE/MAE/loss, mean and SD of RMSE/MAE/WMAPE,
        RMSE/MAE/WMAPE by seed 2026-2028, Params and FLOPs
      - Rename A/B/C/E columns to module_<stage>; retain the original D column.
    Add data_type, config, rank and unit_id.
    """
    rows = []
    for name, df in csvs.items():
        cfg = df["config"].iloc[0]          # CSV configuration: 12_3 or 12_12
        data_type = df["data_type"].iloc[0]  # CSV series type: uni or multi
        sub = df[df["Stability_Rank"] <= 3]  # Keep all three top-ranked rows.

        for _, r in sub.iterrows():
            row = dict(r)                       # Copy all original CSV columns, including evaluation metrics.
            row["data_type"] = data_type        # Series type
            row["config"] = cfg                 # Forecast configuration
            row["rank"] = int(r["Stability_Rank"])   # Integer rank
            row["unit_id"] = f"{r['Dataset']}|{cfg}" # Cluster identifier shared by all three rows
            # Rename motif columns. D follows A automatically and is excluded from the analysis.
            for stage, col in RAW_STAGE_COLS.items():
                row[f"module_{stage}"] = row.pop(col)
            rows.append(row)

    return pd.DataFrame(rows)


def report_module_none(units):
    """Count units whose motif column is "none"(no active motif) in each stage."""
    for stage in RAW_STAGE_COLS:
        col = f"module_{stage}"
        n = (units[col].fillna("").astype(str) == "none").sum()
        print(f"    Stage {stage}: {n} units")


def add_weights(units):
    """
    Within each dataset/configuration unit, add performance"regret and smooth weight"columns.
      regret measures the gap from the smallest metric value within the unit. Zero is best.
               Linear relative regret is (v-best)/best.
      w is the smooth weight 1/(1+regret).
               The best row has w=1. A poor third-ranked candidate (e.g. regret over 1,000%) receives
               a weight approaching zero, which handles cases where"the first two candidates are close and the third performs poorly"within a unit.

    RMSE and FLOPs can vary by orders of magnitude across datasets.
    Normalize within units to compare which alternatives are"near-optimal").

    If the best metric is nonpositive (e.g. all zeros), treat the unit as tied and assign every row weight 1.
    """
    for mode, spec in WEIGHT_MODES.items():
        col, r_col, w_col, use_log = spec["col"], spec["r_col"], spec["w_col"], spec["log"]
        v = units[col].astype(float)
        best = units.groupby("unit_id")[col].transform("min").astype(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            if use_log:
                regret = np.log2(v / best)
            else:
                regret = (v - best) / best
        regret = regret.where(best > 0, 0.0).fillna(0.0)   # best<=0: tie with w=1
        units[r_col] = regret
        units[w_col] = 1.0 / (1.0 + regret)
    return units


def build_unit_table(work_dir, ai_type=None, input_dir=None):
    OUTPUT_DIR = Path(work_dir)
    UNITS_CSV = OUTPUT_DIR / "00_unit_table.csv"
    print("=" * 72)
    print("00_build_units: build analysis units from the original experiment CSVs")
    print("=" * 72)

    # 1. Read original CSVs.
    print("\n[1] Reading original experiment CSVs:")
    csvs = load_raw_csvs(ai_type, input_dir)

    # 2. Check feature consistency across configurations.
    print("\n[2] Cross-configuration feature consistency:")
    check_feature_consistency(csvs)

    # 3. Build the analysis-unit table.
    print("\n[3] Building units (Stability_Rank <= 3; keep all top-three rows):")
    units = build_units(csvs)
    print(f"    Unit table:{len(units)} rows x {units.shape[1]} columns")
    print(f"    (all original features, metrics and structural columns retained)")
    print(f"    Unique datasets: {units['Dataset'].nunique()} | "
          f"Clusters (unit_id): {units['unit_id'].nunique()} total")
    print(f"    Univariate (uni): {int((units['data_type'] == 'uni').sum())} rows / "
          f"multivariate (multi): {int((units['data_type'] == 'multi').sum())} rows")
    print(f"    Configuration 12_3: {int((units['config'] == '12_3').sum())} rows / "
          f"12_12 {int((units['config'] == '12_12').sum())} rows")
    print("    Units with no active motif (none) by stage:")
    report_module_none(units)

    # 3.5. Compute performance regret and smooth weights.
    print("\n[3.5] Unit-level regret and smooth weight w=1/(1+regret):")
    units = add_weights(units)
    for mode, spec in WEIGHT_MODES.items():
        wcol, rcol = spec["w_col"], spec["r_col"]
        print(f"    [{mode:6s}] based on {spec['col']}({'log2 relative' if spec['log'] else 'linear relative'}regret):"
              f"w∈[{units[wcol].min():.3f}, {units[wcol].max():.3f}] median {units[wcol].median():.3f}, "
              f"median regret {units[rcol].median():.2%}, "
              f"effective sample size {units[wcol].sum():.0f}/{len(units)}")

    # 4. Save the analysis-unit table.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    units.to_csv(UNITS_CSV, index=False, encoding="utf-8-sig")
    print(f"\n[Save] Analysis units: {UNITS_CSV}")


def build_overview(work_dir):
    OUTPUT_DIR = Path(work_dir)
    UNITS_CSV = OUTPUT_DIR / "00_unit_table.csv"
    # 1. Read the data and inspect dimensions.
    units = pd.read_csv(UNITS_CSV, encoding="utf-8-sig")
    print("=" * 72)
    print("01_explore_data: dataset overview")
    print("=" * 72)
    print(f"\n[1] Data dimensions:{len(units)} rows x {units.shape[1]} columns")
    print(f"    Unique datasets: {units['Dataset'].nunique()} | "
          f"Clusters (unit_id): {units['unit_id'].nunique()} total")
    print(f"    Univariate (uni): {int((units['data_type'] == 'uni').sum())} rows / "
          f"multivariate (multi): {int((units['data_type'] == 'multi').sum())} rows")
    print(f"    Configuration 12_3: {int((units['config'] == '12_3').sum())} rows / "
          f"12_12 {int((units['config'] == '12_12').sum())} rows")
    print(f"    Rank values within each dataset and configuration:{sorted(units['rank'].unique())}")

    # 2. Inspect columns and missing values.
    print("\n[2] Columns:")
    print("    Dataset: dataset name | data_type: uni/multi | config: forecast configuration | rank: top rank")
    print("    unit_id: cluster | seven features on [0,1] | module_A/B/C/E: motif combinations")
    print("\n    Missing values (expected zero in every column):")
    print(units.isna().sum().rename("Missing count").to_string())

    # 3. Distribution of the seven meta-features
    print("\n[3] Feature distributions (min/q25/median/q75/max):")
    feat_stats = units[FEATURES].describe().loc[["min", "25%", "50%", "75%", "max"]].T
    feat_stats = feat_stats.rename(index={k: FEATURE_CN.get(k, k) for k in FEATURES})
    print(feat_stats.to_string(float_format=lambda v: f"{v:.3f}"))

    # 4. Baseline selection probability p_base for each motif
    # For every stage, motif and target, calculate the"selection rate" = p_base
    # using the performance weights in W_COL from stage 0. Poor rank-three candidates receive near-zero weight.
    #        The baseline estimates the"frequency of a motif among performance-weighted architectures".
    print(f"\n[4] Motif baseline p_base (performance-weighted overall selection fraction):")
    w = units[W_COL].values
    rows = []
    for stage, modules in STAGE_MODULES.items():
        targets = ["contains"]
        for module in modules:
            for target in targets:
                y = build_binary(units, stage, module, target)
                p = float((y.values * w).sum() / w.sum())
                rows.append({"stage": stage, "module": module,
                             "target": target, "p_base": round(p, 4),
                             "n_select": int(y.sum())})
    baseline = pd.DataFrame(rows)
    # Print one table for each stage.
    for stage in STAGE_MODULES:
        sub = baseline[baseline["stage"] == stage].sort_values("p_base", ascending=False)
        print(f"\n  -- Stage {stage} --")
        print(sub[["module", "target", "p_base", "n_select"]].to_string(index=False))

    # 5. Save the baseline table.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "01_data_overview.csv"
    baseline.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n[Save] Baseline probabilities: {out_path}")



def build_target_features(work_dir):
    OUTPUT_DIR = Path(work_dir)
    UNITS_CSV = OUTPUT_DIR / "00_unit_table.csv"
    units = pd.read_csv(UNITS_CSV, encoding="utf-8-sig")
    print("=" * 72)
    print("02_build_target_features: construct target y and feature matrix X")
    print("=" * 72)

    # 1. Compare"strings and binary labels"in the examples.
    print("\n[1] Example target encoding (stage B, LSTM Block)")
    demo = units[["module_B"]].copy()
    demo["contains (membership)"] = build_binary(units, "B", "LSTM Block", "contains")
    print("  Original motif combination".ljust(30), "-> binary label")
    print(demo.head(8).to_string(index=False))
    print("  Note:'CNN Block, LSTM Block' is 1 for LSTM Block when the composition contains it.")

    # 2. Inspect the feature matrix.
    X = build_feature_matrix(units)
    print("\n[2] Feature matrix X (nine columns, first six rows):")
    preview = X.copy()
    preview.insert(0, "Original data_type", units["data_type"].values)
    preview.insert(1, "Original config", units["config"].values)
    print(preview.head(6).to_string(float_format=lambda v: f"{v:.3f}"))
    print("  data_type/config: 0=univariate/12_3 and 1=multivariate/12_12")

    # 3. Examine"seven features x data_type"correlations to check whether features encode the series type.
    print("\n[3] Feature correlations with the multivariate indicator (1=multi):")
    corr = X[FEATURES].corrwith(X["data_type"])
    corr.index = [FEATURE_CN.get(k, k) for k in FEATURES]
    print(corr.to_string(float_format=lambda v: f"{v:.3f}"))
    print("  multivariate_structure is strongly correlated with the multivariate indicator "
          "and therefore captures differences between univariate and multivariate datasets.")

    # 4. Save the feature matrix.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "02_feature_matrix.csv"
    X.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n[Save] Feature matrix: {out_path}")



def prepare_data(work_dir, ai_type=None, input_dir=None):
    """Prepare intermediate analysis tables and motif frequencies in the original order."""
    build_unit_table(work_dir, ai_type, input_dir)
    build_overview(work_dir)
    build_target_features(work_dir)
