# -*- coding: utf-8 -*-
"""Run the full discovery pipeline and retain the two CSVs required for architecture selection.

Usage: python 2search/law_search/main.py
Read and write intermediate tables in a temporary directory to preserve serialization and calculation order.
"""
import shutil
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from prepare_data import (
    DESIGN_FEATURES, FEATURES, FEATURE_CN, W_COL, build_binary,
    build_feature_matrix, parse_conditions, prepare_data, rule_mask,
)
from discover_rules import discover


import config as law_config
from config import (
    B_PERM, PERM_SEED, FDR_ALPHA, N_BOOT, BOOT_SEED, CELLS,
    MIN_CELL_DATASETS, THRESHOLD_ROUND, FDR_STRONG, FDR_EXPLORATORY,
    REPRO_MIN, N_CELLS_TOTAL, MIN_STRONG_COVERAGE, MIN_STRONG_CELL_DATASETS,
    MIN_STABLE_DATASET_FRACTION, MIN_CANDIDATE_DATASETS,
    PURITY_MIN_SUB, PURITY_MIN_FRAC, PURITY_LIFT_MAX,
)


def permutation_rule_mask(units, conditions):
    """Convert conditions to a Boolean mask of matching rows."""
    mask = np.ones(len(units), dtype=bool)
    for feat, direction, thr in conditions:
        x = units[FEATURES].astype(float)[feat].values if feat in FEATURES\
            else ((units["data_type"] == "multi").astype(float).values
                  if feat == "data_type" else
                  (units["config"] == "12_12").astype(float).values)
        mask &= (x > thr) if direction == ">" else (x <= thr)
    return mask


def _weighted_rate(y, weights, mask):
    """Return the performance-weighted fraction; return NaN when no valid weight is available."""
    denom = float(weights[mask].sum())
    if denom <= 0:
        return np.nan
    return float((y.values[mask] * weights[mask]).sum() / denom)


def _unit_permutation(units, rng):
    """
    Build a function"that permutes y in blocks by unit_id"for the permutation test.
    The three rows per dataset and configuration are pseudo-replicates. Move them together in rank order.
    Shuffling individual rows would inflate the effective sample size threefold.
    Return permute(y_vals), producing a permuted array of the original length.
    """
    unit_codes, uids = pd.factorize(units["unit_id"], sort=False)
    n_units = len(uids)
    rank_in_unit = units.groupby("unit_id", sort=False).cumcount().to_numpy()
    first_idx = -np.ones(n_units, dtype=int)
    for i, c in enumerate(unit_codes):
        if first_idx[c] < 0:
            first_idx[c] = i

    def permute(y_vals):
        perm = rng.permutation(n_units)
        row_map = first_idx[perm[unit_codes]] + rank_in_unit
        return y_vals[row_map]

    return permute


def rule_permutation_stats(units, stage, module, target, conditions, permute,
                           n_perm=B_PERM):
    """
    Test fixed rules with unit-level permutations over all data and return validation statistics.

    Statistic: weighted selection rate inside the rule minus the rate outside (risk difference).
    The lift ratio has a compressed null distribution for rules with"high coverage"because the variance collapses
    when the matching subset nearly equals the full set and p_rule and p_base are highly correlated.
    The difference retains two-sample variance. The one-sided p-value is P(rd_perm >= rd_obs) with direction fixed at discovery.

    These are in-sample permutation tests. Selecting rules on the same data makes their p-values optimistic.
      The test asks"whether the fixed condition is informative"and does not adjust for"searching many tree splits across 24 motifs"during discovery.

    Returned columns retain the stage 05 schema for full-data permutation statistics:
      n_rule_confirm / n_dataset_confirm / n_dataset_nontrigger_confirm /
      coverage_dataset_confirm / p_rule_confirm / p_base_confirm / lift_confirm /
      risk_diff_confirm (observed difference in selected rates inside versus outside the rule) /
      p_rule_wald (permutation p-value; retain the downstream column name)
    """
    y = build_binary(units, stage, module, target)
    mask = permutation_rule_mask(units, conditions)
    w = units[W_COL].astype(float).values
    n_rule = int(mask.sum())
    n_dataset_rule = int(units.loc[mask, "Dataset"].nunique())
    n_dataset_nonrule = int(units.loc[~mask, "Dataset"].nunique())
    p_base = _weighted_rate(y, w, np.ones(len(units), dtype=bool))
    p_rule = _weighted_rate(y, w, mask)
    p_nonrule = _weighted_rate(y, w, ~mask)
    lift = (p_rule / p_base
            if not np.isnan(p_rule) and not np.isnan(p_base) and p_base > 0
            else np.nan)
    risk_diff = (p_rule - p_nonrule
                 if not np.isnan(p_rule) and not np.isnan(p_nonrule) else np.nan)
    result = {
        "n_rule_confirm": n_rule,
        "n_dataset_confirm": n_dataset_rule,
        "n_dataset_nontrigger_confirm": n_dataset_nonrule,
        "coverage_dataset_confirm": n_dataset_rule / units["Dataset"].nunique(),
        "p_rule_confirm": p_rule,
        "p_base_confirm": p_base,
        "lift_confirm": lift,
        "risk_diff_confirm": risk_diff,
        "p_rule_wald": np.nan,   # Permutation p-value under the historical column name.
    }
    if n_rule == 0 or np.isnan(risk_diff):
        return result

    # Keep mask and weights fixed. Shuffle y by block to obtain the null distribution of the difference.
    y_vals = y.values
    w_mask = w[mask]
    w_non = w[~mask]
    denom_mask = float(w_mask.sum())
    denom_non = float(w_non.sum())
    null = np.empty(n_perm)
    for b in range(n_perm):
        yp = permute(y_vals)
        pr = (float((yp[mask] * w_mask).sum() / denom_mask)
              if denom_mask > 0 else np.nan)
        pn = (float((yp[~mask] * w_non).sum() / denom_non)
              if denom_non > 0 else np.nan)
        null[b] = pr - pn

    valid = null[~np.isnan(null)]
    if len(valid) == 0:
        return result
    result["p_rule_wald"] = float((1 + np.sum(valid >= risk_diff)) / (len(valid) + 1))
    return result


def bh_fdr(p_values):
    """
    Benjamini-Hochberg FDR correction; return q-values in input order.
    Sort p-values, compute q_i = p_i * n / i and enforce monotonicity by cumulative minima.
    Skip NaNs and retain their positions.
    """
    p = np.asarray(p_values, dtype=float)
    valid = ~np.isnan(p)
    q = np.full(len(p), np.nan)
    idx = np.where(valid)[0]
    pv = p[valid]
    if len(pv) == 0:
        return q
    n = len(pv)
    order = np.argsort(pv)
    sorted_p = pv[order]
    sorted_q = sorted_p * n / np.arange(1, len(pv) + 1)      # BH formula
    sorted_q = np.minimum.accumulate(sorted_q[::-1])[::-1]   # Monotone cumulative minima
    sorted_q = np.minimum(sorted_q, 1.0)
    q[idx[order]] = sorted_q
    return q


def compute_significance(work_dir):
    OUTPUT_DIR = Path(work_dir)
    UNITS_CSV = OUTPUT_DIR / "00_unit_table.csv"
    PREV_CSV = OUTPUT_DIR / "04_pure_rules.csv"
    all_units = pd.read_csv(UNITS_CSV, encoding="utf-8-sig")
    units = all_units                                  # Full dataset without a confirmation split
    prev = pd.read_csv(PREV_CSV, encoding="utf-8-sig")
    print("=" * 72)
    print(f"05_significance: full-data unit-level permutation tests with BH-FDR correction")
    print("=" * 72)
    print(f"[Read] Stage 04 pure rules: {len(prev)} rules; full data: "
          f"{units['Dataset'].nunique()} datasets / {len(units)} rows\n")

    rng = np.random.default_rng(PERM_SEED)
    permute = _unit_permutation(units, rng)

    # Run unit-level permutation tests for each rule.
    stats_list, conds_list = [], []
    for _, row in prev.iterrows():
        conds_list.append(parse_conditions(row["conditions_str"]))
        stats_list.append(rule_permutation_stats(
            units, row["stage"], row["module"], row["target"],
            conds_list[-1], permute))

    stats_df = pd.DataFrame(stats_list)
    p_arr = stats_df["p_rule_wald"].to_numpy(dtype=float)
    q_arr = bh_fdr(p_arr)

    n_sig = int(np.sum((~np.isnan(q_arr)) & (q_arr < FDR_ALPHA)))
    n_nan = int(np.isnan(p_arr).sum())
    print(f"[Result] Significant rules at q<{FDR_ALPHA}: {n_sig} out of {len(p_arr)} records"
          f"(undefined lift:{n_nan} rules)\n")

    # Merge and save the statistics.
    # Save p/q at full precision in scientific notation.
    # Strong rules can give p-values near 1e-15; rounding to four decimals would display zero.
    df = prev.copy()
    for col in stats_df.columns:
        df[col] = stats_df[col].values
    df["q_rule"] = q_arr
    out_path = OUTPUT_DIR / "05_rules_significant.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")

    # Show significant rules in ascending q order.
    show = df.sort_values("q_rule", na_position="last").head(20)
    print("Top 20 rules by ascending q:")
    print(show[["stage", "module", "target", "condition",
                "n_dataset_confirm", "lift_confirm", "q_rule"]].to_string(index=False))

    print(f"\n[Save] Rule significance: {out_path}")


def bootstrap_repro(units, all_rules, n_boot=N_BOOT):
    """
    Compute reproducibility for each rule in all_rules.
    Resample univariate and multivariate datasets separately with replacement, retaining their counts.
    Recompute each rule's lift and count lift>1 as a"replication". Return four arrays (repro, lift_med, lift_lo, lift_hi).
    repro is directional support, lift_med the median lift, and lift_lo/lift_hi the 95% bootstrap limits.

    Build the feature matrix once per draw and the binary labels once per stage, motif and target.
    """
    # Group by dataset and list univariate and multivariate datasets separately.
    ds_groups = {ds: g for ds, g in units.groupby("Dataset")}
    uni_list = sorted(ds for ds, g in ds_groups.items()
                      if g["data_type"].iloc[0] == "uni")
    multi_list = sorted(ds for ds, g in ds_groups.items()
                        if g["data_type"].iloc[0] == "multi")

    lifts = {i: [] for i in range(len(all_rules))}
    rng = np.random.default_rng(BOOT_SEED)

    for _ in range(n_boot):
        # Sample the same counts with replacement in each stratum and combine the rows.
        uni_tbl = pd.concat(
            [ds_groups[ds] for ds in rng.choice(uni_list, size=len(uni_list), replace=True)],
            ignore_index=True)
        multi_tbl = pd.concat(
            [ds_groups[ds] for ds in rng.choice(multi_list, size=len(multi_list), replace=True)],
            ignore_index=True)
        tbl = pd.concat([uni_tbl, multi_tbl], ignore_index=True)

        X_boot = build_feature_matrix(tbl)      # Build one feature matrix per draw.
        y_cache = {}

        w_boot = tbl[W_COL].values              # Keep row performance weights with the resampled datasets.
        for i, rule in enumerate(all_rules):
            key = (rule["stage"], rule["module"], rule["target"])
            if key not in y_cache:              # Compute labels once per target.
                y_cache[key] = build_binary(tbl, *key)
            y = y_cache[key]
            mask = rule_mask(tbl, rule["conditions"], X=X_boot)
            p_base = float((y.values * w_boot).sum() / w_boot.sum())   # Weighted baseline
            n_rule = int(mask.sum())
            if p_base == 0 or n_rule == 0:
                lifts[i].append(np.nan)
            else:
                pr = float((y.values[mask] * w_boot[mask]).sum()
                           / w_boot[mask].sum())                        # Weighted selection rate inside the rule
                lifts[i].append(pr / p_base)

    repro = np.full(len(all_rules), np.nan)
    lift_med = np.full(len(all_rules), np.nan)
    lift_lo = np.full(len(all_rules), np.nan)
    lift_hi = np.full(len(all_rules), np.nan)
    n_boot_valid = np.zeros(len(all_rules), dtype=int)
    for i in range(len(all_rules)):
        arr = np.array(lifts[i], dtype=float)
        arr = arr[~np.isnan(arr)]
        n_boot_valid[i] = len(arr)
        if len(arr) > 0:
            repro[i] = float(np.mean(arr > 1))       # Directional support is the fraction of draws with lift>1.
            lift_med[i] = float(np.median(arr))      # Median lift
            # Use the 2.5th and 97.5th percentiles of the same 1,000 bootstrap draws.
            lift_lo[i] = float(np.percentile(arr, 2.5))
            lift_hi[i] = float(np.percentile(arr, 97.5))
    return repro, lift_med, lift_lo, lift_hi, n_boot_valid


def compute_reproducibility(work_dir):
    OUTPUT_DIR = Path(work_dir)
    UNITS_CSV = OUTPUT_DIR / "00_unit_table.csv"
    PREV_CSV = OUTPUT_DIR / "05_rules_significant.csv"
    all_units = pd.read_csv(UNITS_CSV, encoding="utf-8-sig")
    units = all_units                                  # Full dataset without a confirmation split
    prev = pd.read_csv(PREV_CSV, encoding="utf-8-sig")
    print("=" * 72)
    print("06_reproducibility: full-data dataset-bootstrap directional support")
    print("=" * 72)
    print(f"[Read] Stage 05 rules: {len(prev)} rules; full data: "
          f"{units['Dataset'].nunique()} datasets / {len(units)} rows\n")

    # Reconstruct rule dictionaries with condition lists for the bootstrap.
    all_rules = []
    for _, row in prev.iterrows():
        all_rules.append({
            "stage": row["stage"], "module": row["module"], "target": row["target"],
            "conditions": parse_conditions(row["conditions_str"]),
        })

    print(f"[Bootstrap] Dataset resamples: {N_BOOT} draws, stratified by series type with replacement (about two minutes)...")
    repro, lift_med, lift_lo, lift_hi, n_boot_valid = bootstrap_repro(
        units, all_rules, N_BOOT)

    df = prev.copy()
    df["bootstrap_directional_support"] = np.round(repro, 3)
    df["repro"] = df["bootstrap_directional_support"]  # Retain compatibility with downstream scripts and historical plots.
    df["n_boot_valid"] = n_boot_valid
    df["lift_med_boot"] = np.round(lift_med, 2)
    df["lift_ci_lo"] = np.round(lift_lo, 2)      # 95% CI lower bound
    df["lift_ci_hi"] = np.round(lift_hi, 2)      # 95% CI upper bound

    n_repro = int(np.sum((~np.isnan(repro)) & (repro >= 0.80)))
    print(f"\n[Result] Rules with directional support >= 0.80:{n_repro} / {len(df)} records\n")

    # Show the 15 rules with strongest directional support.
    show = df.sort_values("repro", ascending=False).head(15)
    print("Top 15 rules by bootstrap directional support:")
    print(show[["stage", "module", "target", "condition",
                "lift_confirm", "repro", "lift_med_boot"]].to_string(index=False))

    out_path = OUTPUT_DIR / "06_rules_repro.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n[Save] Rule reproducibility: {out_path}")


def _cell_lift(units, y, mask, mode, cfg):
    """
    Compute fixed-condition lift within a series-type/configuration cell using current branch weights.
    Return lift and the number of distinct matching datasets. Lift is NaN for empty matches or a zero baseline.
    """
    sub = units[(units["data_type"] == mode) & (units["config"] == cfg)]
    if sub.empty:
        return np.nan, 0
    idx = sub.index
    y_s = y[idx].values          # y is a Series indexed like units.
    mask_s = mask[idx]           # mask is a NumPy array; use positional indexing.
    w_s = units.loc[idx, W_COL].values      # Row-aligned performance weights
    n_s = int(mask_s.sum())
    n_dataset_s = int(sub.loc[mask_s, "Dataset"].nunique())
    pb_s = float((y_s * w_s).sum() / w_s.sum())
    if n_s == 0 or pb_s == 0:
        return np.nan, n_dataset_s
    pr_s = float((y_s[mask_s] * w_s[mask_s]).sum() / w_s[mask_s].sum())
    return pr_s / pb_s, n_dataset_s


def cell_stability(units, stage, module, target, conditions):
    """
    Return per-cell lifts, distinct dataset counts, supported cell count and effective cell count.
    Cell lift keys include {"lift_uni_12_3": 2.1, ...}; 
    A supported cell needs MIN_CELL_DATASETS matching datasets; an effective cell also needs lift>1.
    """
    y = build_binary(units, stage, module, target)
    mask = rule_mask(units, conditions)
    lifts = {}
    supports = {}
    for mode, cfg in CELLS:
        lift, n_dataset = _cell_lift(units, y, mask, mode, cfg)
        lifts[f"lift_{mode}_{cfg}"] = lift
        supports[f"n_dataset_{mode}_{cfg}"] = n_dataset
    n_supported = int(sum(v >= MIN_CELL_DATASETS for v in supports.values()))
    n_eff = int(sum(
        supports[f"n_dataset_{mode}_{cfg}"] >= MIN_CELL_DATASETS
        and not np.isnan(lifts[f"lift_{mode}_{cfg}"])
        and lifts[f"lift_{mode}_{cfg}"] > 1
        for mode, cfg in CELLS))
    return lifts, supports, n_supported, n_eff


def compute_stability(work_dir):
    OUTPUT_DIR = Path(work_dir)
    UNITS_CSV = OUTPUT_DIR / "00_unit_table.csv"
    PREV_CSV = OUTPUT_DIR / "06_rules_repro.csv"
    all_units = pd.read_csv(UNITS_CSV, encoding="utf-8-sig")
    units = all_units                                  # Full dataset without a confirmation split
    prev = pd.read_csv(PREV_CSV, encoding="utf-8-sig")
    print("=" * 72)
    print(f"07_stability: four-cell stability by series type and configuration, performance-weighted over all data")
    print("=" * 72)
    print(f"[Read] Stage 06 rules: {len(prev)} rules; full data: "
          f"{units['Dataset'].nunique()} datasets / {len(units)} rows;"
          f"each cell requires at least {MIN_CELL_DATASETS} matching datasets\n")

    # Compute all four cell lifts per rule.
    lift_cols = [f"lift_{mode}_{cfg}" for mode, cfg in CELLS]
    support_cols = [f"n_dataset_{mode}_{cfg}" for mode, cfg in CELLS]
    n_eff_list = []
    n_supported_list = []
    lift_records = {c: [] for c in lift_cols}
    support_records = {c: [] for c in support_cols}
    for _, row in prev.iterrows():
        lifts, supports, n_supported, n_eff = cell_stability(
            units, row["stage"], row["module"], row["target"],
            parse_conditions(row["conditions_str"]))
        for c in lift_cols:
            lift_records[c].append(lifts[c])
        for c in support_cols:
            support_records[c].append(supports[c])
        n_supported_list.append(n_supported)
        n_eff_list.append(n_eff)

    df = prev.copy()
    for c in lift_cols:
        df[c] = np.round(lift_records[c], 2)
    for c in support_cols:
        df[c] = support_records[c]
    df["n_cells_supported"] = n_supported_list
    df["n_cells_effective"] = n_eff_list

    n_univ = int((df["n_cells_effective"] == 4).sum())
    print(f"[Result] Rules valid in all four cells:{n_univ} / {len(df)} rules;"
          f"others are specific to a series type or configuration\n")

    # Show rules supported in all four cells.
    show = df[df["n_cells_effective"] == 4].sort_values(
        "lift_confirm", ascending=False)
    print("Rules valid in all four cells:")
    if show.empty:
        print("  (none)")
    else:
        print(show[["stage", "module", "target", "condition", "lift_confirm",
                    *lift_cols, *support_cols, "n_cells_effective"]].to_string(index=False))

    out_path = OUTPUT_DIR / "07_rules_stability.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n[Save] Rule stability: {out_path}")


def qualitative_label(conditions, units):
    """
    Express numeric thresholds as qualitative Low/Medium/High labels, for example
        trend_strength_score > 0.478  → "Trend=High"(threshold in the upper third).
    Use discovery-set feature tertiles (33%, 67%) to define the three display bands:
      - Single boundary "feature > thr"(upper side): High for thresholds at or above the lower tertile, otherwise Medium.
      - Single boundary "feature ≤ thr"(lower side): Low for thresholds at or below the upper tertile, otherwise Medium.
      - Interval "lo < feature ≤ hi": Low below the lower band, High above the upper band, otherwise Medium.
    """
    X = build_feature_matrix(units)
    from collections import defaultdict
    lows = defaultdict(lambda: -np.inf)
    highs = defaultdict(lambda: np.inf)
    for feat, direction, thr in conditions:
        if direction == ">":
            lows[feat] = max(lows[feat], thr)
        else:
            highs[feat] = min(highs[feat], thr)

    parts = []
    for feat in dict.fromkeys(f for f, _, _ in conditions):
        lo, hi = lows[feat], highs[feat]

        # Translate binary design dimensions directly.
        if feat in DESIGN_FEATURES:
            if feat == "data_type":
                parts.append("Data=" + ("multivariate" if lo != -np.inf and lo < 1 else "univariate"))
            else:
                parts.append("Configuration=" + ("12_12" if lo != -np.inf and lo < 1 else "12_3"))
            continue

        x = X[feat].values
        t33, t67 = np.percentile(x, [33, 67])
        if lo != -np.inf and hi != np.inf:          # Interval
            if hi <= t67:
                lv = "Low"
            elif lo >= t33:
                lv = "High"
            else:
                lv = "Medium"
        elif lo != -np.inf:                          # feature > thr: upper side
            lv = "High" if lo >= t33 else "Medium"
        else:                                        # feature <= thr: lower side
            lv = "Low" if hi <= t67 else "Medium"
        parts.append(f"{FEATURE_CN.get(feat, feat)}={lv}")
    return ", ".join(parts)


def rule_purity(units, stage, module, target, conditions, y=None, mask=None):
    """
    Audit a rule for"counter-preference regions".
    Return (is_pure, pockets) with feat/dir/thr/n/lift for each region.
    """
    if y is None:
        y = build_binary(units, stage, module, target)
    if mask is None:
        mask = rule_mask(units, conditions)
    X = build_feature_matrix(units)
    w_all = units[W_COL].values
    base = float((y.values * w_all).sum() / w_all.sum())   # Weighted baseline
    n_rule = int(mask.sum())
    if n_rule < PURITY_MIN_SUB or base == 0:
        return True, []

    x_all, y_all = X.values, y.values
    w_rule = w_all[mask]
    pockets = []
    for ci, feat in enumerate(X.columns):
        xs, ys = x_all[:, ci][mask], y_all[mask]
        cand = np.unique(np.percentile(xs, np.linspace(5, 95, 19)))
        if len(cand) < 2:
            continue
        for thr in cand:
            if thr <= xs.min() or thr >= xs.max():
                continue
            side = xs <= thr
            n_side = int(side.sum())
            if n_side < PURITY_MIN_SUB or (n_rule - n_side) < PURITY_MIN_SUB:
                continue
            for dir_, s in (("≤", side), (">", ~side)):
                n_s = int(s.sum())
                p_s = (float((ys[s] * w_rule[s]).sum() / w_rule[s].sum())
                       if n_s else np.nan)
                if (n_s >= PURITY_MIN_SUB
                        and n_s / n_rule >= PURITY_MIN_FRAC
                        and p_s / base < PURITY_LIFT_MAX):
                    pockets.append({"feat": feat, "dir": dir_,
                                    "thr": round(float(thr), THRESHOLD_ROUND),
                                    "n_sub": n_s, "n_rule": n_rule,
                                    "lift": round(float(p_s / base), 2)})
                    break
    return (len(pockets) == 0), pockets


def assign_grade(q_rule, repro, lift_ci_lo, n_cells_eff, coverage,
                 n_dataset, supports, is_pure, min_stable_datasets):
    """Grade jointly by evidence, lower effect bound, scope and purity."""
    finite_ci = not np.isnan(lift_ci_lo)
    stable = (not np.isnan(repro)) and repro >= REPRO_MIN
    strong_stat = (not np.isnan(q_rule)) and q_rule < FDR_STRONG
    exploratory_stat = (not np.isnan(q_rule)) and q_rule < FDR_EXPLORATORY
    ci_positive = finite_ci and lift_ci_lo > 1.0
    all_cells_well_supported = all(v >= MIN_STRONG_CELL_DATASETS for v in supports)

    if (strong_stat and stable and ci_positive and is_pure
            and n_cells_eff == N_CELLS_TOTAL
            and all_cells_well_supported and coverage >= MIN_STRONG_COVERAGE):
        return "S"
    if (strong_stat and stable and ci_positive and is_pure
            and n_cells_eff >= 2 and n_dataset >= min_stable_datasets):
        return "A"
    if exploratory_stat and stable and ci_positive and n_dataset >= MIN_CANDIDATE_DATASETS:
        return "B"
    if exploratory_stat and n_dataset >= MIN_CANDIDATE_DATASETS:
        return "C"
    return "D"


def rule_scope(n_cells_eff):
    """Report statistical strength separately from scope so a restricted rule can still be valid."""
    if n_cells_eff == N_CELLS_TOTAL:
        return "Generalizes across cells"
    if n_cells_eff >= 2:
        return "Stable in some cells"
    return "Condition- or cell-specific"


def operational_evidence(grade):
    """Convert grade to deployment weight. Evidence is assessed within the same distribution."""
    labels = {
        "S": ("Strong general evidence within the same distribution", 1.0),
        "A": ("Robust conditional evidence within the same distribution", 0.8),
        "B": ("Reproducible candidate", 0.4),
        "C": ("Exploratory evidence", 0.2),
        "D": ("Exclude from recommendations", 0.0),
    }
    return labels[grade]


def select_deployment_rules(law):
    """Select all deployable S/A rules and flag motif representatives.

    The evidence thresholds determine the count. Keep B/C/D in the audit table
    outside the deployable set.
    """
    pool = law[law["grade"].isin(["S", "A"])].copy()
    pool["_grade_rank"] = pool["grade"].map({"S": 0, "A": 1})
    pool = pool.sort_values(
        ["_grade_rank", "lift_ci_lo", "coverage", "lift", "n_features"],
        ascending=[True, False, False, False, True],
    )
    representatives = pool.drop_duplicates(["stage", "module", "target"], keep="first")
    return list(pool.index), set(representatives.index)


def export_law_table(work_dir):
    OUTPUT_DIR = Path(work_dir)
    UNITS_CSV = OUTPUT_DIR / "00_unit_table.csv"
    PREV_CSV = OUTPUT_DIR / "07_rules_stability.csv"
    all_units = pd.read_csv(UNITS_CSV, encoding="utf-8-sig")
    units = all_units                                  # Full dataset without a confirmation split
    prev = pd.read_csv(PREV_CSV, encoding="utf-8-sig")
    print("=" * 72)
    print(f"08_build_cookbook: grade and assemble the law table with full-data permutation q-values and performance weights")
    print("=" * 72)
    print(f"[Read] Stage 07 rules: {len(prev)} rules; compute qualitative labels and purity over all "
          f"{units['Dataset'].nunique()} datasets / {len(units)} rows\n")
    min_stable_datasets = int(np.ceil(
        units["Dataset"].nunique() * MIN_STABLE_DATASET_FRACTION))
    print(f"[Grade A support] minimum fraction of all datasets: "
          f"{MIN_STABLE_DATASET_FRACTION:.0%}(≥{min_stable_datasets} datasets)\n")

    lift_cols = [f"lift_{mode}_{cfg}" for mode, cfg in
                 [("uni", "12_3"), ("uni", "12_12"),
                  ("multi", "12_3"), ("multi", "12_12")]]
    support_cols = [f"n_dataset_{mode}_{cfg}" for mode, cfg in
                    [("uni", "12_3"), ("uni", "12_12"),
                     ("multi", "12_3"), ("multi", "12_12")]]

    # Add qualitative conditions, purity and grades per rule.
    rows = []
    for _, r in prev.iterrows():
        conds = parse_conditions(r["conditions_str"])
        is_pure, pockets = rule_purity(units, r["stage"], r["module"],
                                       r["target"], conds)
        row = {
            "stage": r["stage"],
            "module": r["module"],
            "target": r["target"],
            "rule_origin": r.get("rule_origin", "leaf"),
            "n_features": int(r["n_features"]),
            "condition": r["condition"],
            "condition_qual": qualitative_label(conds, units),   # Qualitative columns
            # Discovery and validation effects here use the same full dataset.
            "n_rule": int(r["n_rule_confirm"]),
            "n_dataset_confirm": int(r["n_dataset_confirm"]),
            "n_dataset_nontrigger_confirm": int(r["n_dataset_nontrigger_confirm"]),
            "coverage": r["coverage_dataset_confirm"],
            "p_rule": r["p_rule_confirm"],
            "p_base": r["p_base_confirm"],
            "lift": r["lift_confirm"],
            "risk_diff_confirm": r["risk_diff_confirm"],
            "p_rule_wald": r["p_rule_wald"],
            "q_rule": r["q_rule"],
            "repro": r["repro"],
            "bootstrap_directional_support": r["bootstrap_directional_support"],
            "n_boot_valid": int(r["n_boot_valid"]),
            "lift_med_boot": r["lift_med_boot"],
            "lift_ci_lo": r["lift_ci_lo"],      # Lower 95% bootstrap CI for forest plot error bars
            "lift_ci_hi": r["lift_ci_hi"],      # Upper 95% bootstrap CI
            "n_rule_discovery": int(r["n_rule"]),
            "n_dataset_discovery": int(r["n_dataset_rule"]),
            "p_rule_discovery": r["p_rule"],
            "p_base_discovery": r["p_base"],
            "lift_discovery": r["lift"],
        }
        for c in lift_cols:
            row[c] = r[c]
        for c in support_cols:
            row[c] = int(r[c])
        row["n_cells_supported"] = int(r["n_cells_supported"])
        row["n_cells_effective"] = int(r["n_cells_effective"])
        row["scope"] = rule_scope(int(r["n_cells_effective"]))
        row["purity"] = "Pure" if is_pure else "Contains a counter-preference region"
        row["impure_detail"] = "" if is_pure or not pockets else " ; ".join(
            f"{p['feat']}{p['dir']}{p['thr']}: "
            f"n={p['n_sub']}, lift={p['lift']}x" for p in pockets[:3])
        supports = [int(r[c]) for c in support_cols]
        row["grade"] = assign_grade(
            float(r["q_rule"]), float(r["repro"]), float(r["lift_ci_lo"]),
            int(r["n_cells_effective"]), float(row["coverage"]),
            int(r["n_dataset_confirm"]), supports, is_pure, min_stable_datasets)
        row["evidence_tier"], row["rule_weight"] = operational_evidence(row["grade"])
        rows.append(row)

    law = pd.DataFrame(rows)

    # Sort by stage, evidence grade and descending lift.
    order = {"S": 0, "A": 1, "B": 2, "C": 3, "D": 4}
    law["_o"] = law["grade"].map(order)
    law = law.sort_values(["stage", "_o", "lift"],
                          ascending=[True, True, False]).drop(columns="_o")
    law = law.reset_index(drop=True)

    # Main table: all deployable S/A rules. Save the full audit separately.
    law["is_representative"] = 0
    selected_idx, representative_idx = select_deployment_rules(law)
    law["selected_for_deployment"] = 0
    law.loc[selected_idx, "selected_for_deployment"] = 1
    law.loc[list(representative_idx), "is_representative"] = 1
    cert = law[law["selected_for_deployment"] == 1].copy()
    law["group"] = law["grade"].map({
        "S": "Main table", "A": "Main table", "B": "candidate", "C": "Exploratory", "D": "Excluded",
    })
    law.loc[(law["grade"].isin(["S", "A"]))
            & (law["selected_for_deployment"] == 0), "group"] = "Reserve"
    certified_df = law[law["group"] == "Main table"]\
        .sort_values(["stage", "lift"], ascending=[True, False]).reset_index(drop=True)
    candidate_df = law[law["group"] == "candidate"]\
        .sort_values(["stage", "lift"], ascending=[True, False]).reset_index(drop=True)

    # Save deployable rules and the full audit separately.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "08_cookbook_law_table_deploy.csv"
    audit_path = OUTPUT_DIR / "08_cookbook_law_table_audit.csv"
    cert.to_csv(out_path, index=False, encoding="utf-8-sig")
    law.to_csv(audit_path, index=False, encoding="utf-8-sig")
    print(f"[Save] Deployable S/A rules: {out_path}")
    print(f"[Save] Full rule audit: {audit_path}\n")

    # Console output
    print("[S/A/B/C/D grade counts]:")
    print(law["grade"].value_counts().to_string())

    print("\n[Strong general S rules] Stable across four cells, CI lower bound >1, sufficient coverage:")
    strong = law[law["grade"] == "S"]
    if strong.empty:
        print("  (none)")
    else:
        print(strong[["stage", "module", "condition_qual", "condition",
                      "lift", "lift_ci_lo", "q_rule", "repro", "n_cells_effective"]]
              .to_string(index=False))

    print(f"\n[Main table] Deployable S/A rules ( {len(certified_df)} rules):")
    show_main = certified_df[["stage", "module", "target", "condition_qual", "condition",
                              "lift", "q_rule", "repro", "scope", "grade"]]
    print(show_main.to_string(index=False))

    print(f"\n[Candidates] Reproducible grade B rules ( {len(candidate_df)} rules):")
    print(candidate_df[["stage", "module", "condition_qual", "condition",
                        "lift", "q_rule", "repro", "scope"]].to_string(index=False))

    print("\n[R soft constraints] Evidence tiers and weights for all rules:")
    print(law[["evidence_tier", "rule_weight"]].value_counts().to_string())

    print("\n[All rules ordered by stage, grade and lift]:")
    show = law[["stage", "module", "condition_qual", "condition", "n_features",
                "lift", "q_rule", "repro", "n_cells_effective", "purity", "grade"]]
    print(show.to_string(index=False))

    print("\n" + "=" * 72)
    print("Cookbook exported.")
    print("=" * 72)


def run_pipeline(output_dir, ai_type=None, input_dir=None):
    """Save motif baselines and deployable rules after successful computation. Temporary tables are cleaned up automatically."""
    output_dir = Path(output_dir)
    ai_type = ai_type or law_config.AI_TYPES[0]
    with TemporaryDirectory(prefix="law-search-") as temp_dir:
        work_dir = Path(temp_dir)
        prepare_data(work_dir, ai_type, input_dir)
        discover(work_dir)
        compute_significance(work_dir)
        compute_reproducibility(work_dir)
        compute_stability(work_dir)
        export_law_table(work_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        outputs = {
            "01_data_overview.csv": output_dir / f"{ai_type}_module_baseline.csv",
            "08_cookbook_law_table_deploy.csv": output_dir / f"{ai_type}_law_table.csv",
        }
        for name, destination in outputs.items():
            shutil.copyfile(work_dir / name, destination)
        rules = pd.read_csv(outputs["08_cookbook_law_table_deploy.csv"], encoding="utf-8-sig")
    return {"ai_type": ai_type, "n_rules": len(rules),
            "grades": rules["grade"].value_counts().to_dict(),
            "law_table": outputs["08_cookbook_law_table_deploy.csv"],
            "module_baseline": outputs["01_data_overview.csv"]}


def main():
    from main import main as run_main
    run_main()


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    main()
