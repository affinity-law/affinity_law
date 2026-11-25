# -*- coding: utf-8 -*-
"""Candidate discovery, purity splitting and deduplication from stages 03-04."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from prepare_data import (
    DESIGN_FEATURES, STAGE_MODULES, W_COL, build_binary,
    build_feature_matrix, parse_conditions, rule_mask, serialize_conditions,
)


from config import (
    MAX_DEPTH, MIN_LEAF, MIN_REL_GAIN, MIN_RULE_LIFT, SEED, THRESHOLD_ROUND,
    MIN_RULE_N, MIN_DISCOVERY_DATASETS, PURITY_MIN_SUB, PURITY_MIN_FRAC,
    PURITY_LIFT_MAX, PURITY_MAX_SPLIT, CANONICALIZE, MIN_PURITY_DATASETS,
)


def _relative_gain(tree, node):
    """
    Relative node gain = (parent impurity - weighted child impurity) / parent impurity.
    It measures"the proportional reduction in impurity at this split"using Gini impurity.
    """
    imp = tree.impurity[node]
    if imp == 0:                     # Already pure.
        return 1.0
    # Impurities are weighted, so use weighted sample counts here as well.
    n = tree.weighted_n_node_samples[node]
    if n <= 0:
        return 0.0
    L, R = tree.children_left[node], tree.children_right[node]
    nL, nR = tree.weighted_n_node_samples[L], tree.weighted_n_node_samples[R]
    weighted = (nL * tree.impurity[L] + nR * tree.impurity[R]) / n
    return (imp - weighted) / imp


def discover_rules(units, stage, module, target):
    """
    Fit a gain-gated tree per stage, motif and target to identify conditions that"favor the motif"as candidate rules.

    Fit the tree with performance weights via sample_weight so that"better-performing architectures on each dataset"
    drive splits. Poor third-ranked candidates receive near-zero weight.

    Return a list of dictionaries with fields:
      conditions: [(feature, direction, threshold), ...] on the root-to-node path
      depth: number of conditions on that path
      n_leaf: row count in the leaf
      frac: weighted fraction selecting the motif (normalized by sklearn for sample_weight)
    """
    y = build_binary(units, stage, module, target)
    if y.nunique() < 2:
        return []
    X = build_feature_matrix(units)
    w = units[W_COL].values
    p_base = float((y.values * w).sum() / w.sum())     # Performance-weighted baseline

    # Shallow performance-weighted tree; default depth limit 3 and minimum leaf size 60.
    clf = DecisionTreeClassifier(max_depth=MAX_DEPTH,
                                 min_samples_leaf=MIN_LEAF,
                                 random_state=SEED)
    clf.fit(X, y, sample_weight=w)
    tree = clf.tree_
    pos_idx = int(np.where(clf.classes_ == 1)[0][0])   # "selected"class index in tree.value

    rules = []

    def positive_fraction(node):
        """Handle sklearn versions returning either weighted counts or normalized values."""
        values = tree.value[node][0]
        total = float(values.sum())
        return float(values[pos_idx] / total) if total > 0 else 0.0

    def emit(node, conditions, depth, origin):
        """Consider favored nodes as candidates; the unconditional root is ineligible."""
        if not conditions:
            return
        frac = positive_fraction(node)
        if frac > p_base * MIN_RULE_LIFT:
            rules.append({"conditions": list(conditions),
                          "depth": depth,
                          "n_leaf": int(tree.n_node_samples[node]),
                          "frac": frac,
                          "rule_origin": origin})

    def dfs(node, conditions, depth):
        """Traverse depth-first, retaining favorable intermediate nodes and root-to-leaf paths."""
        is_leaf = tree.children_left[node] == -1
        if is_leaf:
            emit(node, conditions, depth, "leaf")
            return

        # Shallow nodes may yield simpler rules with broader coverage than the deepest leaves.
        emit(node, conditions, depth, "ancestor")

        # From depth 2 onward, stop at nodes with relative gain below the noise floor.
        if depth >= 1 and _relative_gain(tree, node) < MIN_REL_GAIN:
            return

        feat = X.columns[tree.feature[node]]
        # Preserve original thresholds in computations and serialization; round only display conditions.
        thr = float(tree.threshold[node])
        dfs(tree.children_left[node], conditions + [(feat, "≤", thr)], depth + 1)
        dfs(tree.children_right[node], conditions + [(feat, ">", thr)], depth + 1)

    dfs(0, [], 0)
    return rules


def _translate(feat, direction, thr):
    """Render one condition, using human-readable labels for design features."""
    if feat == "data_type":
        side = "multivariate" if (direction == ">" and thr < 1) else "univariate"
        return f"data_type={side}"
    if feat == "config":
        side = "12_12" if (direction == ">" and thr < 1) else "12_3"
        return f"config={side}"
    return f"{feat} {direction} {thr}"


def format_discovery_conditions(conditions):
    """Convert (feature, direction, threshold) tuples to text, merging same-feature conditions into intervals."""
    from collections import defaultdict
    lows = defaultdict(lambda: -np.inf)   # Lower bound for each feature ('>')
    highs = defaultdict(lambda: np.inf)   # Upper bound for each feature ('≤')
    for feat, direction, thr in conditions:
        if direction == ">":
            lows[feat] = max(lows[feat], thr)
        else:
            highs[feat] = min(highs[feat], thr)

    parts, seen = [], set()
    for feat, direction, thr in conditions:
        if feat in seen:
            continue
        seen.add(feat)
        lo, hi = lows[feat], highs[feat]

        if feat in DESIGN_FEATURES:      # Translate binary design features directly.
            parts.append(_translate(feat, ">", lo) if lo != -np.inf
                         else _translate(feat, "≤", hi))
            continue

        if lo != -np.inf and hi != np.inf:
            parts.append(f"{lo:.{THRESHOLD_ROUND}f} < {feat} ≤ {hi:.{THRESHOLD_ROUND}f}")
        elif lo != -np.inf:
            parts.append(f"{feat} > {lo:.{THRESHOLD_ROUND}f}")
        elif hi != np.inf:
            parts.append(f"{feat} ≤ {hi:.{THRESHOLD_ROUND}f}")
        else:
            parts.append(feat)
    return ", ".join(parts)


def _rule_stats(units, stage, module, target, conditions):
    """Compute performance-weighted rule statistics on the given subset."""
    y = build_binary(units, stage, module, target)
    w = units[W_COL].values
    mask = rule_mask(units, conditions)
    n_rule = int(mask.sum())
    n_dataset_rule = int(units.loc[mask, "Dataset"].nunique())
    p_base = float((y.values * w).sum() / w.sum())
    if n_rule == 0:
        return {"n_rule": 0, "n_dataset_rule": 0,
                "p_rule": np.nan, "p_base": p_base, "lift": np.nan}
    p_rule = float((y.values[mask] * w[mask]).sum() / w[mask].sum())
    lift = p_rule / p_base if p_base > 0 else np.nan
    return {"n_rule": n_rule, "n_dataset_rule": n_dataset_rule,
            "p_rule": p_rule, "p_base": p_base, "lift": lift}


def discover_candidates(work_dir):
    OUTPUT_DIR = Path(work_dir)
    UNITS_CSV = OUTPUT_DIR / "00_unit_table.csv"
    units = pd.read_csv(UNITS_CSV, encoding="utf-8-sig")
    print("=" * 72)
    print("03_tree_discovery: discover decision-tree rules on the entire dataset")
    print("=" * 72)

    # Build all stage/motif pairs using contains semantics.
    groups = []
    for stage, modules in STAGE_MODULES.items():
        targets = ["contains"]
        for module in modules:
            for target in targets:
                groups.append((stage, module, target))
    print(f"[Discovery] Fit trees for {len(groups)} (stage, motif, target) groups over all "
          f"{units['Dataset'].nunique()} datasets...\n")

    # Discover rules for each group on the full dataset.
    all_rules = []
    for stage, module, target in groups:
        for r in discover_rules(units, stage, module, target):
            ds = _rule_stats(units, stage, module, target, r["conditions"])
            eligible_discovery = (ds["n_rule"] >= MIN_RULE_N
                                  and ds["n_dataset_rule"] >= MIN_DISCOVERY_DATASETS)
            r.update({
                "stage": stage, "module": module, "target": target,
                "condition": format_discovery_conditions(r["conditions"]),
                "conditions_str": serialize_conditions(r["conditions"]),
                "rule_origin": r["rule_origin"],
                "n_rule": ds["n_rule"], "n_dataset_rule": ds["n_dataset_rule"],
                "p_rule": ds["p_rule"],
                "p_base": ds["p_base"], "lift": ds["lift"],
                "eligible_discovery": eligible_discovery,
            })
            all_rules.append(r)

    df = pd.DataFrame(all_rules)
    print(f"[Discovery] Raw candidate rules: {len(df)} records")
    kept = df[df["eligible_discovery"]].copy()
    print(f"[Filter] Require at least{MIN_RULE_N} rows and{MIN_DISCOVERY_DATASETS} datasets:"
          f"retained {len(kept)} rules,"
          f"excluded {len(df)-len(kept)} narrow rules\n")

    # Show the top 15 retained rules by descending lift.
    show = kept.sort_values("lift", ascending=False).head(15)
    print("Top 15 retained rules by full-data lift:")
    print(show[["stage", "module", "target", "condition",
                "n_rule", "n_dataset_rule", "lift"]].to_string(index=False))

    # Save results.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "03_candidate_rules.csv"
    kept.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n[Save] Retained rules: {out_path}")



def format_conditions(conditions):
    """Format feature/direction/threshold tuples as in stage 03."""
    from collections import defaultdict
    lows, highs = defaultdict(lambda: -np.inf), defaultdict(lambda: np.inf)
    for feat, direction, thr in conditions:
        if direction == ">":
            lows[feat] = max(lows[feat], thr)
        else:
            highs[feat] = min(highs[feat], thr)
    parts, seen = [], set()
    for feat, direction, thr in conditions:
        if feat in seen:
            continue
        seen.add(feat)
        lo, hi = lows[feat], highs[feat]
        if feat in DESIGN_FEATURES:
            parts.append("data_type=" + ("multivariate" if lo != -np.inf and lo < 1 else "univariate")
                         if feat == "data_type" else
                         "config=" + ("12_12" if lo != -np.inf and lo < 1 else "12_3"))
            continue
        if lo != -np.inf and hi != np.inf:
            parts.append(f"{lo:.{THRESHOLD_ROUND}f} < {feat} ≤ {hi:.{THRESHOLD_ROUND}f}")
        elif lo != -np.inf:
            parts.append(f"{feat} > {lo:.{THRESHOLD_ROUND}f}")
        elif hi != np.inf:
            parts.append(f"{feat} ≤ {hi:.{THRESHOLD_ROUND}f}")
        else:
            parts.append(feat)
    return ", ".join(parts)


def rule_purity(units, stage, module, target, conditions, y=None, mask=None):
    """
    Audit a rule for"counter-preference regions".
    Within the matching rows, scan feature quantiles for subregions satisfying
        lift < PURITY_LIFT_MAX and sample count >= PURITY_MIN_SUB
       and fraction of matching rows >= PURITY_MIN_FRAC.
    If any such region exists, the rule is"impure"because it contains a region where the motif is disfavored.

    Return (is_pure, pockets), where pockets lists counter-preference regions
        as dictionaries with feat, dir, thr, n_sub, n_rule and lift.
    """
    if y is None:
        y = build_binary(units, stage, module, target)
    if mask is None:
        mask = rule_mask(units, conditions)
    X = build_feature_matrix(units)
    w_all = units[W_COL].values                   # Performance weights from stage 0
    base = float((y.values * w_all).sum() / w_all.sum())   # Weighted baseline
    n_rule = int(mask.sum())
    if n_rule < PURITY_MIN_SUB or base == 0:
        return True, []

    x_all, y_all = X.values, y.values
    w_rule = w_all[mask]                          # Weights inside the rule
    pockets = []
    for ci, feat in enumerate(X.columns):
        xs, ys = x_all[:, ci][mask], y_all[mask]
        cand = np.unique(np.percentile(xs, np.linspace(5, 95, 19)))   # Candidate quantile thresholds
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
                       if n_s else np.nan)       # Weighted subregion selection rate
                if (n_s >= PURITY_MIN_SUB
                        and n_s / n_rule >= PURITY_MIN_FRAC
                        and p_s / base < PURITY_LIFT_MAX):
                    pockets.append({"feat": feat, "dir": dir_,
                                    # Use unrounded thresholds when splitting; round only human-readable text.
                                    "thr": float(thr),
                                    "n_sub": n_s, "n_rule": n_rule,
                                    "lift": round(float(p_s / base), 2)})
                    break
    return (len(pockets) == 0), pockets


def purity_split(units, rule, depth=0):
    """
    Recursively split rules containing counter-preference regions into pure subrules.
    rule includes stage/module/target/conditions/depth and basic statistics.
    Split at the threshold of the lowest-lift counter-preference region,
    then recurse only on"still-favored"subrules with selection probability above baseline and sufficient support.
    Return pure subrules in place of the parent.
    """
    if depth >= PURITY_MAX_SPLIT:
        return [rule]

    pure, pockets = rule_purity(units, rule["stage"], rule["module"],
                                rule["target"], rule["conditions"])
    if pure or not pockets:
        return [rule]

    best = min(pockets, key=lambda p: p["lift"])      # Lowest-lift counter-preference region
    out = []
    w = units[W_COL].values
    y_base = build_binary(units, rule["stage"], rule["module"], rule["target"])
    base = rule.get("p_base") or float((y_base.values * w).sum() / w.sum())
    for new_dir in ("≤", ">"):
        new_conds = list(rule["conditions"]) + [(best["feat"], new_dir, best["thr"])]
        y = build_binary(units, rule["stage"], rule["module"], rule["target"])
        m = rule_mask(units, new_conds)
        n = int(m.sum())
        if n < PURITY_MIN_SUB:
            continue
        p_rule = float((y.values[m] * w[m]).sum() / w[m].sum())   # Weighted selection rate
        if p_rule <= base:                            # Discard the disfavored side.
            continue
        child = dict(rule, conditions=new_conds, depth=rule["depth"] + 1,
                     n_leaf=n, frac=p_rule, p_rule=p_rule, p_base=base)
        # Remove inherited stale statistics and recompute condition, n_rule and lift.
        for k in ("condition", "n_rule", "lift"):
            child.pop(k, None)
        out += purity_split(units, child, depth + 1)
    return out


def _recompute_stats(units, rule):
    """Recompute n_rule/p_rule/p_base/lift/depth with the current branch weights."""
    w = units[W_COL].values
    y = build_binary(units, rule["stage"], rule["module"], rule["target"])
    mask = rule_mask(units, rule["conditions"])
    n_rule = int(mask.sum())
    n_dataset_rule = int(units.loc[mask, "Dataset"].nunique())
    p_base = float((y.values * w).sum() / w.sum())
    p_rule = (float((y.values[mask] * w[mask]).sum() / w[mask].sum())
              if n_rule else np.nan)
    rule.update({
        "condition": format_conditions(rule["conditions"]),
        "n_rule": n_rule, "n_dataset_rule": n_dataset_rule,
        "p_rule": p_rule, "p_base": p_base,
        "lift": (p_rule / p_base) if (p_base > 0 and n_rule) else np.nan,
        "depth": len(rule["conditions"]),
        "n_leaf": n_rule, "frac": p_rule,
    })
    return rule


def _mergeable(r1, r2):
    """
    Check whether two complementary rules can be merged.
    Let common be the intersection of the condition sets. Each rule must add exactly one condition
    on the same feature and threshold in opposite directions (f>a versus f<=a).
    """
    si, sj = set(r1["conditions"]), set(r2["conditions"])
    common = si & sj
    di, dj = si - common, sj - common
    if len(di) != 1 or len(dj) != 1:
        return False
    (f1, d1, a1), = di
    (f2, d2, a2), = dj
    return (f1 == f2 and abs(a1 - a2) < 1e-9 and {d1, d2} == {"≤", ">"})


def _merge_two(units, r1, r2):
    """Merge using common conditions and recompute statistics. Return None if the merge fails thresholds."""
    common = set(r1["conditions"]) & set(r2["conditions"])
    merged = dict(r1, conditions=sorted(common), rule_origin="merged")
    _recompute_stats(units, merged)
    if merged["n_rule"] == 0 or merged["lift"] < MIN_RULE_LIFT:
        return None
    return merged


def merge_complementary(units, rules):
    """
    Within each (stage, module, target), repeatedly merge favored rules differing only in
    complementary conditions spanning the whole domain of a feature; remove that condition.
    Recompute statistics over the union of the two rules.
    """
    from collections import defaultdict
    grouped = defaultdict(list)
    for r in rules:
        grouped[(r["stage"], r["module"], r["target"])].append(r)

    out = []
    for group in grouped.values():
        changed = True
        while changed:
            changed = False
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    if not _mergeable(group[i], group[j]):
                        continue
                    merged = _merge_two(units, group[i], group[j])
                    if merged is None:
                        continue
                    group = [g for k, g in enumerate(group) if k not in (i, j)]
                    group.append(merged)
                    changed = True
                    break
                if changed:
                    break
        out += group
    return out


def prune_dominated(rules):
    """
    Within a group, prune a rule R if a retained K has a subset of its conditions
    and lift(K) >= lift(R). R adds no useful distinction.
    Order each group by"ascending condition count, then descending lift"and retain rules greedily.
    """
    from collections import defaultdict
    grouped = defaultdict(list)
    for r in rules:
        grouped[(r["stage"], r["module"], r["target"])].append(r)

    out = []
    for group in grouped.values():
        group.sort(key=lambda r: (len(r["conditions"]), -r["lift"]))
        kept = []
        for r in group:
            conds_r = set(r["conditions"])
            dominated = any(set(k["conditions"]) <= conds_r and k["lift"] >= r["lift"]
                            for k in kept)
            if not dominated:
                kept.append(r)
        out += kept
    return out


def deduplicate_equivalent(units, rules):
    """Remove near-duplicate thresholds selecting identical rows for the same motif.

    Neighboring tree splits can yield different thresholds selecting the same observed rows.
    Keep a simpler, stronger representative in the cookbook.
    """
    from collections import defaultdict
    grouped = defaultdict(list)
    for r in rules:
        grouped[(r["stage"], r["module"], r["target"])].append(r)

    out = []
    for group in grouped.values():
        representatives = {}
        for r in group:
            key = rule_mask(units, r["conditions"]).tobytes()
            old = representatives.get(key)
            if old is None or (len(r["conditions"]), -r["lift"]) < (len(old["conditions"]), -old["lift"]):
                representatives[key] = r
        out.extend(representatives.values())
    return out


def canonicalize_rules(units, rules):
    """Merge complements to a fixed point, then prune nested specializations. Pruning adds no rules."""
    rules = deduplicate_equivalent(units, rules)
    rules = merge_complementary(units, rules)
    rules = prune_dominated(rules)
    return rules


def purify_rules(work_dir):
    OUTPUT_DIR = Path(work_dir)
    UNITS_CSV = OUTPUT_DIR / "00_unit_table.csv"
    PREV_CSV = OUTPUT_DIR / "03_candidate_rules.csv"
    all_units = pd.read_csv(UNITS_CSV, encoding="utf-8-sig")
    units = all_units                                  # Full dataset without a confirmation split
    print("=" * 72)
    print("04_purity_split: audit and split rules using the full dataset")
    print("=" * 72)

    prev = pd.read_csv(PREV_CSV, encoding="utf-8-sig")
    print(f"[Read] Stage 03 candidates: {len(prev)} rules;"
          f"Audit purity across {units['Dataset'].nunique()} datasets\n")

    # Reconstruct rule dictionaries.
    rules = []
    for _, row in prev.iterrows():
        rules.append({
            "stage": row["stage"], "module": row["module"], "target": row["target"],
            "conditions": parse_conditions(row["conditions_str"]),
            "depth": int(row["depth"]),
            "n_leaf": int(row["n_leaf"]),
            "frac": float(row["frac"]),
            "p_base": float(row["p_base"]),
            "p_rule": float(row["p_rule"]),
            "condition": row["condition"],
            "n_rule": int(row["n_rule"]),
            "lift": float(row["lift"]),
            "rule_origin": row.get("rule_origin", "leaf"),
        })

    # Split impure rules.
    before = len(rules)
    pure_rules = []
    n_impure = 0
    for r in rules:
        is_pure, pockets = rule_purity(units, r["stage"], r["module"],
                                       r["target"], r["conditions"])
        if is_pure:
            pure_rules.append(r)
        else:
            n_impure += 1
            pure_rules += purity_split(units, r)

    # Recompute all rules on the discovery set so original and split rules use identical samples.
    for r in pure_rules:
        _recompute_stats(units, r)
    before_support = len(pure_rules)
    pure_rules = [r for r in pure_rules
                  if r["n_dataset_rule"] >= MIN_PURITY_DATASETS]
    print(f"[Support] Require at least{MIN_PURITY_DATASETS} distinct datasets after splitting:"
          f"retained {len(pure_rules)} / {before_support} records")

    print(f"[Result] Purity audit:{before} rules, of which {n_impure} contain counter-preference regions;"
          f"after splitting: {len(pure_rules)} rules (pure subrules replace misleading parents)")

    # Merge complementary rules and prune nested specializations.
    if CANONICALIZE:
        n_before = len(pure_rules)
        pure_rules = canonicalize_rules(units, pure_rules)
        print(f"[Deduplication] Canonicalized {n_before} → {len(pure_rules)} records"
              f"(complementary merges and nested pruning leave distinct preference regions)")
    else:
        print("[Deduplication] Skipped because CANONICALIZE=False")

    # Recompute counts, selection rates and lifts from the discovery set after merging.
    for r in pure_rules:
        _recompute_stats(units, r)

    df = pd.DataFrame(pure_rules)
    df["conditions_str"] = df["conditions"].apply(serialize_conditions)
    df["n_features"] = df["depth"]

    print(f"\n[Final] Canonical rules: {len(df)} records\n")

    # Show the top 15 by descending lift.
    show = df.sort_values("lift", ascending=False).head(15)
    print("Top 15 canonical rules by lift:")
    print(show[["stage", "module", "target", "condition",
                "n_rule", "n_dataset_rule", "p_rule", "p_base", "lift"]]
          .to_string(index=False))

    # Save results.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cols = ["stage", "module", "target", "rule_origin", "n_features", "condition", "conditions_str",
            "n_leaf", "frac", "n_rule", "n_dataset_rule",
            "p_rule", "p_base", "lift"]
    out_path = OUTPUT_DIR / "04_pure_rules.csv"
    df[cols].to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n[Save] Pure rules: {out_path}")



def discover(work_dir):
    """Discover and audit candidate rules from the analysis-unit table."""
    discover_candidates(work_dir)
    purify_rules(work_dir)
