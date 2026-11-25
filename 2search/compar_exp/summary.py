# -*- coding: utf-8 -*-
"""
Pairwise L/LLM/F/R summaries using both means and medians.
============================================================================
Read the available exp_L.csv, exp_LLM.csv, exp_F.csv and exp_R.csv files in ../compar_result/.
Pair by dataset and horizon. Compute RMSE mean, median, variance and SD across three seeds.
Count wins, ties (rtol=1e-9) and losses separately for both mean and median comparisons.
Report a one-sided sign test, series-type subgroups, three-seed stability and parameter counts.

Outputs:
    Console: net win rate, p-value, subgroup results, mean SD and parameter counts per comparison.
    ../compar_result/summary.csv: one row per pair of methods, with mean and median comparisons.
    ../compar_result/summary_<A>_vs_<B>_pairs.csv: task-level statistics, outcomes, motifs and parameters.
"""

import itertools
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_TOP = _HERE.parent
for _p in (_TOP, _HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import config  # noqa: E402

# Experiment CSV to architecture name; include existing files only.
ARCH_CSVS = [("exp_L.csv", "L"), ("exp_LLM.csv", "LLM"),
             ("exp_F.csv", "F"), ("exp_R.csv", "R")]
# Compare L against each baseline, then compare baselines pairwise.
MAIN_PAIRS = [("L", "LLM"), ("L", "F"), ("L", "R")]
CROSS_PAIRS = [("LLM", "F"), ("LLM", "R"), ("F", "R")]
TIE_RTOL = 1e-9
TIE_ATOL = 0.0


def _binom_p(wins, n):
    """One-sided sign-test p-value: P(X >= wins | Bin(n, 0.5)) for a greater-than alternative."""
    p = 0.0
    for k in range(wins, n + 1):
        p += math.comb(n, k) * 0.5 ** n
    return p


def agg(df):
    """Aggregate by dataset, horizon and architecture, retaining seed statistics and motif information."""
    g = df.groupby(["dataset", "horizon", "arch"], as_index=False)
    out = g.agg(
        data_type=("data_type", "first"),
        rmse_mean=("rmse", "mean"),
        rmse_std=("rmse", "std"),
        rmse_var=("rmse", "var"),
        rmse_median=("rmse", "median"),
        rmse_n=("rmse", "count"),
        mae_mean=("mae", "mean"),
        wmape_mean=("wmape", "mean"),
        params_mean=("params", "mean"),
    )
    # List distinct motifs because designs may vary across seeds for LLM.
    mods = (df.dropna(subset=["rmse"])
              .groupby(["dataset", "horizon", "arch"])
              .agg(mod_A=("module_A", lambda s: "|".join(sorted(set(s)))),
                   mod_B=("module_B", lambda s: "|".join(sorted(set(s)))),
                   mod_C=("module_C", lambda s: "|".join(sorted(set(s)))),
                   mod_E=("module_E", lambda s: "|".join(sorted(set(s)))))
              .reset_index())
    return out.merge(mods, on=["dataset", "horizon", "arch"], how="left")


def _pair_winrate(ra, xa, name_r, name_x, col="rmse_mean"):
    """Return (piv, wins, draws, losses, eff, net, p) for the chosen statistic."""
    rcol = ra[["dataset", "horizon", col]].rename(columns={col: name_r})
    xcol = xa[["dataset", "horizon", col, "data_type"]].rename(columns={col: name_x})
    piv = rcol.merge(xcol, on=["dataset", "horizon"], how="inner")
    piv = piv.dropna(subset=[name_r, name_x])
    draw_mask = np.isclose(
        piv[name_r], piv[name_x], rtol=TIE_RTOL, atol=TIE_ATOL
    )
    wins = int(((piv[name_r] < piv[name_x]) & ~draw_mask).sum())
    draws = int(draw_mask.sum())
    losses = int(((piv[name_r] > piv[name_x]) & ~draw_mask).sum())
    eff = wins + losses
    net = wins / eff if eff else float("nan")
    p = _binom_p(wins, eff) if eff else 1.0
    return piv, wins, draws, losses, eff, net, p


def compare(ra, xa, name_r, name_x, out_prefix):
    """Compare two architectures and return a summary dictionary."""
    print(f"\n===== {name_r} vs {name_x}(paired by dataset and horizon) =====")

    # -- Mean comparison (primary) --
    piv, wins, draws, losses, eff, net, p = _pair_winrate(ra, xa, name_r, name_x)
    print(f"  Mean comparison: pairs {len(piv)}  (wins/ties/losses = {wins}/{draws}/{losses})  "
          f"net win rate = {net:.1%}  ({wins}/{eff}) sign-test p = {p:.4g}")

    # -- Median comparison (robustness check) --
    m_piv, mwins, mdraws, mlosses, meff, mnet, mp = _pair_winrate(ra, xa, name_r, name_x, "rmse_median")
    print(f"  Median comparison: pairs {len(m_piv)}  (wins/ties/losses = {mwins}/{mdraws}/{mlosses})  "
          f"net win rate = {mnet:.1%}  ({mwins}/{meff})  p = {mp:.4g}")

    # -- Univariate/multivariate subgroups (mean) --
    for dt in ["uni", "multi"]:
        sub = piv[piv.data_type == dt]
        if sub.empty:
            continue
        draw_mask = np.isclose(
            sub[name_r], sub[name_x], rtol=TIE_RTOL, atol=TIE_ATOL
        )
        w = int(((sub[name_r] < sub[name_x]) & ~draw_mask).sum())
        d = int(draw_mask.sum())
        e = len(sub) - d
        print(f"  [{dt}] pairs={len(sub)} wins={w} ties={d}  net win rate={w / e if e else float('nan'):.1%}")

    # -- Stability (SD across three seeds) --
    pair_std = ra[["dataset", "horizon", "rmse_std"]].rename(
        columns={"rmse_std": f"{name_r}_std"}).merge(
        xa[["dataset", "horizon", "rmse_std"]].rename(
            columns={"rmse_std": f"{name_x}_std"}), on=["dataset", "horizon"])
    pair_std = pair_std.dropna()
    std_ratio = float("nan")
    if len(pair_std):
        a_s, b_s = pair_std[f"{name_r}_std"].median(), pair_std[f"{name_x}_std"].median()
        std_ratio = float((pair_std[f"{name_r}_std"] < pair_std[f"{name_x}_std"]).mean())
        print(f"  median SD: {name_r}={a_s:.4f}  {name_x}={b_s:.4f}  "
              f"{name_r} fraction of more stable pairs = {std_ratio:.1%}")

    # -- Parameter counts (capacity comparison) --
    pair_params = ra[["dataset", "horizon", "params_mean"]].rename(
        columns={"params_mean": f"{name_r}_params"}).merge(
        xa[["dataset", "horizon", "params_mean"]].rename(
            columns={"params_mean": f"{name_x}_params"}), on=["dataset", "horizon"])
    pair_params = pair_params.dropna()
    ratio_p = float("nan")
    if len(pair_params):
        rp, xp = pair_params[f"{name_r}_params"].mean(), pair_params[f"{name_x}_params"].mean()
        ratio_p = float(xp / max(rp, 1))
        print(f"  Mean parameter count: {name_r}={rp:.0f}  {name_x}={xp:.0f}  "
              f"{name_x}/{name_r} = {ratio_p:.2f}x")

    # -- Task-level comparisons --
    draw_mask = np.isclose(
        piv[name_r], piv[name_x], rtol=TIE_RTOL, atol=TIE_ATOL
    )
    piv["win"] = np.where(
        draw_mask, "draw", np.where(piv[name_r] < piv[name_x], name_r, name_x)
    )
    piv["rel_improve"] = (piv[name_x] - piv[name_r]) / piv[name_x]
    for label, col in [(f"{name_r}_std", "rmse_std"), (f"{name_r}_median", "rmse_median"),
                       (f"{name_r}_params", "params_mean")]:
        piv = piv.merge(ra[["dataset", "horizon", col]].rename(columns={col: label}),
                        on=["dataset", "horizon"], how="left")
    for label, col in [(f"{name_x}_std", "rmse_std"), (f"{name_x}_median", "rmse_median"),
                       (f"{name_x}_params", "params_mean")]:
        piv = piv.merge(xa[["dataset", "horizon", col]].rename(columns={col: label}),
                        on=["dataset", "horizon"], how="left")
    piv = piv.merge(ra[["dataset", "horizon", "mod_A", "mod_B", "mod_C", "mod_E"]].rename(
        columns={f"mod_A": f"{name_r}_mod_A", f"mod_B": f"{name_r}_mod_B",
                 f"mod_C": f"{name_r}_mod_C", f"mod_E": f"{name_r}_mod_E"}), on=["dataset", "horizon"], how="left")
    piv = piv.merge(xa[["dataset", "horizon", "mod_A", "mod_B", "mod_C", "mod_E"]].rename(
        columns={f"mod_A": f"{name_x}_mod_A", f"mod_B": f"{name_x}_mod_B",
                 f"mod_C": f"{name_x}_mod_C", f"mod_E": f"{name_x}_mod_E"}), on=["dataset", "horizon"], how="left")
    piv = piv.sort_values(["dataset", "horizon"]).reset_index(drop=True)
    out_pairs = config.RESULT_ROOT / f"{out_prefix}_pairs.csv"
    piv.to_csv(out_pairs, index=False, encoding="utf-8-sig")
    print(f"  Task-level comparisons: {out_pairs}")

    return {
        "pair": f"{name_r}_vs_{name_x}",
        "n_pairs": len(piv), "wins": wins, "draws": draws, "losses": losses,
        "net_win_rate": net, "sign_p": p,
        "median_n_pairs": len(m_piv), "median_wins": mwins, "median_draws": mdraws,
        "median_losses": mlosses, "median_net_win_rate": mnet, "median_sign_p": mp,
        "a_std_median": float(pair_std[f"{name_r}_std"].median()) if len(pair_std) else None,
        "b_std_median": float(pair_std[f"{name_x}_std"].median()) if len(pair_std) else None,
        "a_more_stable_ratio": std_ratio,
        "a_params_mean": float(pair_params[f"{name_r}_params"].mean()) if len(pair_params) else None,
        "b_params_mean": float(pair_params[f"{name_x}_params"].mean()) if len(pair_params) else None,
        "param_ratio": ratio_p,
    }


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    loaded = {}
    for csv, name in ARCH_CSVS:
        path = config.RESULT_ROOT / csv
        if path.exists():
            loaded[name] = agg(pd.read_csv(path, encoding="utf-8-sig"))
            print(f"Loaded {csv} → {name}({len(loaded[name])} aggregated rows)")
        else:
            print(f"[Skip] {csv} is missing (experiment has not run)")

    if "L" not in loaded:
        print("Missing exp_L.csv. Complete L before comparing methods.")
        return

    rows = []
    for a, b in MAIN_PAIRS + CROSS_PAIRS:
        if a in loaded and b in loaded:
            rows.append(compare(loaded[a], loaded[b], a, b, f"summary_{a}_vs_{b}"))

    if rows:
        summary = pd.DataFrame(rows)
        out_sum = config.RESULT_ROOT / "summary.csv"
        summary.to_csv(out_sum, index=False, encoding="utf-8-sig")
        print(f"\nSummary: {out_sum}({len(rows)} method pairs)")


if __name__ == "__main__":
    main()
