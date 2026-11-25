# -*- coding: utf-8 -*-
"""
Architecture selection for law-guided recommendations.
============================================
L: rule-guided selection
    Read each test dataset's seven meta-features and apply deployable S/A laws.
      - Rules specific to config=12_3 or 12_12 apply only to that horizon.
      - Within each stage, choose the"highest grade, breaking ties by the robust effect size (lift_ci_lo)"of each matching motif.
      - Unmatched A/B/C stages use their most frequent training motif.
      - Unmatched E uses Single Block with one B.
      - A matched non-single composition uses two B blocks at most.
    For B, count a motif wherever it occurs in the composition (contains).

F: frequency baseline
    Choose the highest-p_base motifs for A/B/C/E and the top two B motifs when composition requires two.
    L uses default_arch() for fallback, with E fixed to Single Block.

Specification format{"A": [motif names...], "B": [motif names...], "C": motif name, "E": composition name}
    Pass the specification to model_assembler.build_model_code().
"""

import re
from functools import lru_cache
from pathlib import Path

import pandas as pd

import config


def load_law():
    """Read the published laws; trigger on full-precision thresholds and rank by pooled confidence bounds."""
    path = Path(config.LAW_TABLE)
    if path.suffix.lower() == ".csv":
        law = pd.read_csv(path, encoding="utf-8-sig")
        if 'conditions_str_exact' in law.columns:
            required = {'rule_id', 'stage', 'module', 'target', 'condition', 'conditions_str_exact',
                        'pooled_grade', 'pooled_lift', 'source_csv_row'}
            if required - set(law):
                raise ValueError(f'Missing columns in published rule table:{sorted(required - set(law))}')
            if law.rule_id.duplicated().any():
                raise ValueError('Published rule table contains duplicate rule IDs')
            pooled = pd.read_csv(config.POOLED_LAW_TABLE, encoding='utf-8-sig')
            ci = []
            for _, row in law.iterrows():
                source = pooled.iloc[int(row.source_csv_row) - 2]
                if any(row[key] != source[key] for key in ('stage', 'module', 'target', 'condition')):
                    raise ValueError(f'{row.rule_id} does not match pooled discovery statistics')
                if row.pooled_grade != source.grade or abs(row.pooled_lift - source.lift) > 1e-12:
                    raise ValueError(f'{row.rule_id} has a different pooled grade or effect size')
                ci.append(source.lift_ci_lo)
                rule_conditions(row)  # Fail before training if exact conditions are malformed.
            law['grade'] = law['pooled_grade']
            law['lift'] = law['pooled_lift']
            law['lift_ci_lo'] = ci
        required = {"stage", "module", "target", "condition", "grade", "lift", "lift_ci_lo"}
        missing = required - set(law.columns)
        if missing:
            raise ValueError(f"Missing columns in the law table:{sorted(missing)}")
    else:
        law = pd.read_excel(path, sheet_name="Affinity laws", engine="openpyxl")
        expected_ids = ({f"A{i:02d}" for i in range(1, 13)} |
                        {f"B{i:02d}" for i in range(1, 6)} |
                        {"C01", "C02", "E01", "E02", "E03"})
        if len(law) != 22 or set(law["law_id"]) != expected_ids:
            raise ValueError("09.xlsx must contain the 22 manuscript rules (A01-E03)")
    if not law["grade"].isin(["S", "A"]).all():
        raise ValueError("The deployable law table may contain only S/A rules")
    return law


@lru_cache(maxsize=1)
def load_overview():
    """Compute defaults from the same 4,734 discovery candidates, independently of the 63 test datasets."""
    if config.OVERVIEW_CSV:
        return pd.read_csv(config.OVERVIEW_CSV, encoding='utf-8-sig')
    parts = []
    for ai in config.AI_TYPES:
        for horizon in (3, 12):
            for mode in ('univariate', 'multivariate'):
                frame = pd.read_csv(config.LAW_INPUT_DIR / f'{ai}_12_{horizon}_{mode}.csv', encoding='utf-8-sig')
                frame['config'] = f'12_{horizon}'
                parts.append(frame)
    units = pd.concat(parts, ignore_index=True)
    if len(units) != 4734 or units.Dataset.nunique() != 263:
        raise ValueError('The motif baseline requires 4,734 discovery candidates from the original 263 datasets')
    values = units['mean_RMSE'].astype(float)
    best = units.groupby(['Dataset', 'config'])['mean_RMSE'].transform('min').astype(float)
    weights = 1 / (1 + (values / best - 1).where(best > 0, 0).fillna(0))
    columns = {'A': 'A_Input_Preprocessing', 'B': 'B_Backbone_Modeling',
               'C': 'C_Head_Projection', 'E': 'E_Combination_Paradigm'}
    rows = []
    for stage, column in columns.items():
        series = units[column].fillna('').astype(str)
        modules = sorted({item.strip() for value in series for item in value.split(',')
                          if item.strip() and item.strip().lower() != 'none'})
        for module in modules:
            selected = series.str.contains(module, regex=False).astype(int)
            rows.append(dict(stage=stage, module=module, target='contains',
                             p_base=round(float((selected * weights).sum() / weights.sum()), 4),
                             n_select=int(selected.sum())))
    return pd.DataFrame(rows)


def rule_conditions(row):
    """Use conditions_str_exact for triggers. Labels and rounded conditions are for display."""
    exact = row.get('conditions_str_exact')
    if not isinstance(exact, str) or not exact:
        if 'conditions_str_exact' in row:
            raise ValueError('Full-precision conditions are empty')
        return parse_condition(row['condition'])
    conditions = []
    for part in exact.split('|'):
        feature, direction, threshold = part.split(':')
        if direction not in ('>', '≤'):
            raise ValueError(f'Unsupported condition direction:{direction}')
        conditions.append((feature, '>' if direction == '>' else '<=', float(threshold)))
    return conditions


# ---------------------------------------------------------------------------
# Condition parsing and trigger evaluation
# ---------------------------------------------------------------------------

def parse_condition(s):
    """
    'trend_strength_score > 0.478, 0.420 < periodicity_score ≤ 0.519, config=12_12'
      → [('trend_strength_score','>',0.478), ('periodicity_score','in',0.420,0.519), ('config','=','12_12')]
    """
    if not isinstance(s, str) or not s:
        return []
    conds = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "≤" in part and "<" in part:
            m = re.match(r"([\d.eE+-]+)\s*<\s*(\w+)\s*≤\s*([\d.eE+-]+)", part)
            if m:
                conds.append((m.group(2), "in", float(m.group(1)), float(m.group(3))))
                continue
        if "=" in part and "<" not in part and ">" not in part:
            k, v = part.split("=", 1)
            conds.append((k.strip(), "=", v.strip()))
            continue
        m = re.match(r"(\w+)\s*>\s*([\d.eE+-]+)", part)
        if m:
            conds.append((m.group(1), ">", float(m.group(2))))
            continue
        m = re.match(r"(\w+)\s*≤\s*([\d.eE+-]+)", part)
        if m:
            conds.append((m.group(1), "<=", float(m.group(2))))
            continue
        m = re.match(r"(\w+)\s*<\s*([\d.eE+-]+)", part)
        if m:
            conds.append((m.group(1), "<", float(m.group(2))))
            continue
    return conds


def rule_triggers(conds, meta, horizon):
    """meta: dict{seven feature names: values}; horizon: 3 or 12. Filter config conditions by forecast horizon."""
    for item in conds:
        feat, op = item[0], item[1]
        if feat == "config":
            if op == "=" and item[2] != f"12_{horizon}":
                return False
            continue
        if feat == "data_type":
            return False  # The current table has no data-dimension condition. Treat such conditions as unmatched.
        x = meta.get(feat)
        if x is None:
            return False
        if op == ">":
            if not x > item[2]:
                return False
        elif op == "<=":
            if not x <= item[2]:
                return False
        elif op == "<":
            if not x < item[2]:
                return False
        elif op == "in":
            if not (item[2] < x <= item[3]):
                return False
    return True


# ---------------------------------------------------------------------------
# L: law-guided recommendation
# ---------------------------------------------------------------------------

_GRADE_RANK = {"S": 0, "A": 1}


def _robust_score(row):
    """Use the bootstrap lower bound (lift_ci_lo) as robust effect size and fall back to lift if missing.
    Ranking by lift_ci_lo prevents"narrow, high-lift rules"
    from outranking"broader rules with more robust evidence".
    """
    v = row.get("lift_ci_lo")
    if v is None or (isinstance(v, float) and pd.isna(v)):
        v = row.get("lift")
    return float(v) if v is not None else 1.0


def _best_rule(rules):
    """For (row, grade_rank, lift) tuples, select by grade and then by robust effect size."""
    return min(rules, key=lambda t: (t[1], -_robust_score(t[0])))


def _rule_corners(row):
    """Parse a rule's counter-preference region (impure_detail, for example 'stationarity≤0.058: n=60, lift=0.83x').
    A region inside the rule has lift<1, indicating lower preference for the motif there.
    Exclude such regions from triggers. Return [(feature, op, threshold)].
    """
    s = row.get("impure_detail", "")
    if not isinstance(s, str) or not s.strip():
        return []
    corners = []
    for seg in re.split(r"[\s;]+", s):
        if not seg or ":" not in seg:
            continue
        cond = seg.split(":")[0]
        m = re.match(r"^([a-zA-Z_]+)\s*([>≤])\s*(-?[\d.]+)$", cond)
        if m:
            corners.append((m.group(1), m.group(2), float(m.group(3))))
    return corners


def _in_corner(row, meta):
    """Check whether the dataset lies in an unreliable region where the rule should not trigger."""
    for feat, op, thr in _rule_corners(row):
        v = meta.get(feat)
        if v is None:
            continue
        if op == ">" and v > thr:
            return True
        if op == "≤" and v <= thr:
            return True
    return False


def _fit_b_to_e(preferred, e_name, fallback):
    """Fill B to match the composition using training frequencies only."""
    pool = []
    for m in list(preferred) + list(fallback):
        if m and m not in pool:
            pool.append(m)
    need = 1 if e_name == "Single Block" else 2
    if len(pool) < need:
        raise ValueError(f"{e_name} Expected {need} distinct B motifs; found only {pool}")
    return pool[:need]


def _ranked_stage_modules(overview, stage, target="contains"):
    """List unique motifs for a stage in descending training p_base order."""
    sub = overview[(overview["stage"] == stage) & (overview["target"] == target)]
    if sub.empty:
        sub = overview[overview["stage"] == stage]
    if sub.empty:
        raise ValueError(f"No overview entries for stage={stage} of each matching motif.")
    return (sub.sort_values("p_base", ascending=False)["module"]
               .drop_duplicates().tolist())


def law_recommended(meta, horizon, law=None):
    """meta: dict{seven feature names: values}; horizon: 3/12. Return a specification.

    Use the strongest rule by grade and effect size in matched stages.
    For unmatched stages use the most frequent A/B/C motifs and Single Block for E.
    """
    law = law if law is not None else load_law()
    sig = law[law["grade"].isin(["S", "A"])].copy()
    sig["_g"] = sig["grade"].map(_GRADE_RANK)

    triggers = {"A": [], "B": [], "C": [], "E": []}
    for _, r in sig.iterrows():
        conds = rule_conditions(r)
        if rule_triggers(conds, meta, horizon) and not _in_corner(r, meta):
            triggers[r["stage"]].append((r, r["_g"], r["lift"]))

    overview = load_overview()
    default = default_arch(overview)
    fallback_b = _ranked_stage_modules(overview, "B")

    spec = {}
    # A: choose the best match or use the F default for A.
    spec["A"] = [_best_rule(triggers["A"])[0]["module"]] if triggers["A"] else list(default["A"])

    # B: use contains semantics, ranking by grade and robust effect size.
    preferred_b = ([_best_rule(triggers["B"])[0]["module"]]
                   if triggers["B"] else list(default["B"]))

    # C / E
    spec["C"] = _best_rule(triggers["C"])[0]["module"] if triggers["C"] else default["C"]
    spec["E"] = (_best_rule(triggers["E"])[0]["module"]
                 if triggers["E"] else "Single Block")
    spec["B"] = _fit_b_to_e(preferred_b, spec["E"], fallback_b)
    return spec


def law_candidates(meta, horizon, law=None, max_candidates=6):
    """Return variations of the rule-based specification for validation selection.

    Rank matched motifs per stage by robust effect size (lift_ci_lo). Use the top N
    or the default if unmatched. Candidates include the base and each stage's"runner-up motif" +
    and paired A preprocessing. Deduplicate to obtain a small set of
    "rule-supported"candidates for validation selection. LLM uses a single design.
    """
    law = law if law is not None else load_law()
    sig = law[law["grade"].isin(["S", "A"])].copy()

    hit = {"A": [], "B": [], "C": [], "E": []}
    seen_mod = {"A": set(), "B": set(), "C": set(), "E": set()}
    for _, r in sig.iterrows():
        st = r["stage"]
        conds = rule_conditions(r)
        if rule_triggers(conds, meta, horizon) and not _in_corner(r, meta):
            if r["module"] not in seen_mod[st]:
                seen_mod[st].add(r["module"])
                hit[st].append(r)

    overview = load_overview()
    default = default_arch(overview)
    fallback_b = _ranked_stage_modules(overview, "B")
    # Stage candidates: matches ranked by grade and lift_ci_lo plus the fallback.
    stage_mods = {}
    for st, dflt in [("A", list(default["A"])), ("B", list(default["B"])),
                     ("C", [default["C"]]), ("E", [default["E"]])]:
        hit[st].sort(key=lambda r: (_GRADE_RANK[r["grade"]], -_robust_score(r)))
        mods = [r["module"] for r in hit[st]]
        for m in dflt:
            if m not in mods:
                mods.append(m)
        stage_mods[st] = mods

    a_mods, b_mods = stage_mods["A"], stage_mods["B"]
    c_mods, e_mods = stage_mods["C"], stage_mods["E"]

    cands = []
    base = {"A": a_mods[:1], "B": _fit_b_to_e(b_mods, e_mods[0], fallback_b),
            "C": c_mods[0], "E": e_mods[0]}
    cands.append(dict(base))
    # Replace each stage with its"runner-up motif"if it has at least two candidate motifs.
    for st, mods in [("A", a_mods), ("B", b_mods), ("C", c_mods), ("E", e_mods)]:
        if len(mods) >= 2:
            s = dict(base)
            if st == "A":
                s["A"] = [mods[1]]
            elif st == "B":
                alt_b = [mods[1]] + [m for m in b_mods if m != mods[1]]
                s["B"] = _fit_b_to_e(alt_b, s["E"], fallback_b)
            elif st == "C":
                s["C"] = mods[1]
            else:
                s["E"] = mods[1]
                s["B"] = _fit_b_to_e(s["B"] + b_mods, s["E"], fallback_b)
            cands.append(s)
    # Combine the top two preprocessing motifs in A.
    if len(a_mods) >= 2:
        s = dict(base)
        s["A"] = a_mods[:2]
        cands.append(s)

    # Remove duplicates.
    seen = set()
    uniq = []
    for s in cands:
        key = (tuple(s["A"]), tuple(s["B"]), s["C"], s["E"])
        if key not in seen:
            seen.add(key)
            uniq.append(s)
    return uniq[:max_candidates]


# ---------------------------------------------------------------------------
# F: frequency defaults
# ---------------------------------------------------------------------------

def default_arch(overview=None):
    """L fallback: most frequent A/B/C motifs, Single Block for E and a single B."""
    overview = overview if overview is not None else load_overview()
    ranked = {stage: _ranked_stage_modules(overview, stage)
              for stage in ["A", "B", "C"]}
    return {"A": [ranked["A"][0]],
            "B": [ranked["B"][0]],
            "C": ranked["C"][0], "E": "Single Block"}


def frequency_default_arch(overview=None):
    """F baseline: highest training p_base in each A/B/C/E stage.
    Apply the"most frequent motif at every stage"criterion literally. E is Gated Fusion when it has the highest p_base and
    requires two B blocks, so take the top two, consistent with _fit_b_to_e.
    L's default_arch() instead fixes E to Single Block."""
    overview = overview if overview is not None else load_overview()
    ranked = {stage: _ranked_stage_modules(overview, stage)
              for stage in ["A", "B", "C", "E"]}
    return {"A": [ranked["A"][0]],
            "B": ranked["B"][:2],
            "C": ranked["C"][0],
            "E": ranked["E"][0]}


# ---------------------------------------------------------------------------
# Demo: apply L to the first N datasets in the configured meta-feature directory.
# ---------------------------------------------------------------------------

def demo():
    from collections import Counter

    law = load_law()
    default = default_arch()
    print("L defaults:", default)
    print()

    a_count = Counter()
    a_fallback = 0
    rows = []
    from common import discover_datasets
    for dataset in discover_datasets()[:10]:
        f = Path(dataset["meta_csv"])
        ds = f.parent.name
        meta = dict(zip(*pd.read_csv(f, encoding="utf-8-sig")
                        [["indicator_name", "value"]].values.T))
        for horizon in config.HORIZONS:
            spec = law_recommended(meta, horizon, law=law)
            rows.append((ds, horizon, spec))
            for m in spec["A"]:
                a_count[m] += 1
            if spec["A"] == list(default["A"]):
                a_fallback += 1
    print(f"First 10 datasets across horizons: {len(rows)} recommendations; stage A distribution: {dict(a_count)}; "
          f"A used the L fallback motif {a_fallback} times")
    print("\nFirst six recommendations:")
    for ds, h, spec in rows[:6]:
        print(f"  {ds} @{h}: A={spec['A']} B={spec['B']} C={spec['C']} E={spec['E']}")


if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    demo()
