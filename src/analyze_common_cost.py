#!/usr/bin/env python3
"""
Analysis of the common-cost (multi-objective) tuning check, seeds 701-900.

Usage:
    python analyze_common_cost.py results_701_900.csv [--train train_101_150.csv]

Input CSV (one row per seed), columns named like the earlier gen_*.csv files:
    seed,
    <POLICY>_coverage_pct, <POLICY>_avg_relay_delay_steps, <POLICY>_avg_relay_energy_pct
for POLICY in: LARS, LARS_MO, T_MO, V_MO, AUC_MO, C, D, B, J, T_tuned, V_tuned
(BR optional). Missing optional policies are skipped.

Cost (fixed in the plan BEFORE any run):
    cost = (100 - coverage) + 0.5 * delay + 0.5 * energy        [lower is better]

Families (BH within each family, alpha = 0.05):
    F19  primary   : cost, LARS vs {LARS_MO, T_MO, V_MO, AUC_MO}          4 tests (Wilcoxon)
    F19b primary   : cost TOST, same 4 comparators, margin +-COST_MARGIN   4 tests
    F20  secondary : coverage, delay, energy, same 4 comparators         12 tests
    F20b secondary : coverage TOST (+-2.0), same 4 comparators            4 tests
    F21  exploratory: cost/cov/delay/energy, LARS vs the other policies   4 x n_other tests
Sign convention: Delta = LARS minus comparator (negative cost/delay/energy favors LARS).
"""
import sys, argparse
import numpy as np, pandas as pd
from scipy import stats

W_COV, W_DLY, W_ENG = 1.0, 0.5, 0.5     # fixed in the plan
COV_MARGIN = 2.0                         # coverage points (as in the paper)
COST_MARGIN = 1.0                        # cost units: CONFIRM THIS IN THE PLAN BEFORE RUNNING
MET = {"cov": "coverage_pct", "dly": "avg_relay_delay_steps", "eng": "avg_relay_energy_pct"}
PRIMARY = ["LARS_MO", "T_MO", "V_MO", "AUC_MO"]
EXPLORATORY = ["C", "D", "B", "J", "T_tuned", "V_tuned", "BR"]


def bh(p):
    p = np.asarray(p, float)
    n = len(p)
    if n == 0:
        return p
    o = np.argsort(p)
    c = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    q = np.empty(n)
    q[o] = np.minimum(c, 1.0)
    return q


def cost(df, pol, w=(W_COV, W_DLY, W_ENG)):
    return (w[0] * (100 - df[f"{pol}_{MET['cov']}"])
            + w[1] * df[f"{pol}_{MET['dly']}"]
            + w[2] * df[f"{pol}_{MET['eng']}"])


def series(df, pol, m):
    return cost(df, pol) if m == "cost" else df[f"{pol}_{MET[m]}"]


def tost_p(d, margin):
    n = len(d)
    sd = d.std(ddof=1)
    if sd == 0:
        return 0.0 if abs(d.mean()) < margin else 1.0
    se = sd / np.sqrt(n)
    return max(stats.t.sf((d.mean() + margin) / se, n - 1),
               stats.t.cdf((d.mean() - margin) / se, n - 1))


def compare(df, pol, m, focal="LARS"):
    d = (series(df, focal, m) - series(df, pol, m)).to_numpy(float)
    n = len(d)
    sd = d.std(ddof=1)
    se = sd / np.sqrt(n) if sd > 0 else 0.0
    try:
        p = stats.wilcoxon(d).pvalue
    except ValueError:
        p = 1.0
    t95, t90 = stats.t.ppf(0.975, n - 1), stats.t.ppf(0.95, n - 1)
    nz = d[d != 0]
    ranks = stats.rankdata(np.abs(nz))
    rb = (ranks[nz > 0].sum() - ranks[nz < 0].sum()) / ranks.sum() if len(nz) else 0.0
    return dict(comp=pol, metric=m, n=n,
                lars=series(df, focal, m).mean(), comp_mean=series(df, pol, m).mean(),
                diff=d.mean(), lo95=d.mean() - t95 * se, hi95=d.mean() + t95 * se,
                lo90=d.mean() - t90 * se, hi90=d.mean() + t90 * se, p=p, r=rb, d_arr=d)


def run_family(df, comps, metrics, name, tost_metric=None, tost_margin=None):
    rows = []
    for c in comps:
        for m in metrics:
            r = compare(df, c, m)
            if tost_metric == m:
                r["ptost"] = tost_p(r.pop("d_arr"), tost_margin)
            else:
                r.pop("d_arr")
            rows.append(r)
    R = pd.DataFrame(rows)
    R["family"] = name
    R["q"] = bh(R["p"])
    if "ptost" in R:
        R["qtost"] = bh(R["ptost"].fillna(1.0))
    return R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--train", help="training-block CSV (seeds 101-150) to report training costs")
    ap.add_argument("--out", default="common_cost_results.csv")
    a = ap.parse_args()

    df = pd.read_csv(a.results).sort_values("seed").reset_index(drop=True)
    assert df.seed.is_unique, "duplicate seeds"
    print(f"seeds {df.seed.min()}-{df.seed.max()}  n={len(df)}")
    have = lambda p: f"{p}_{MET['cov']}" in df.columns
    prim = [p for p in PRIMARY if have(p)]
    expl = [p for p in EXPLORATORY if have(p)]
    missing = [p for p in PRIMARY if not have(p)]
    if missing:
        print("WARNING: missing primary policies:", missing)

    allp = ["LARS"] + prim + expl
    print("\nMean outcomes (cost uses 1.0*(100-cov) + 0.5*delay + 0.5*energy)")
    tab = pd.DataFrame({
        p: dict(cov=df[f"{p}_{MET['cov']}"].mean(), delay=df[f"{p}_{MET['dly']}"].mean(),
                energy=df[f"{p}_{MET['eng']}"].mean(), cost=cost(df, p).mean(),
                cost_sem=cost(df, p).std(ddof=1) / np.sqrt(len(df)))
        for p in allp}).T
    print(tab.round(2).to_string())

    # F19 (primary): cost
    F19 = run_family(df, prim, ["cost"], "F19", tost_metric=None)
    # F19b: cost TOST
    rows = []
    for c in prim:
        d = (cost(df, "LARS") - cost(df, c)).to_numpy(float)
        rows.append(dict(comp=c, diff=d.mean(), ptost=tost_p(d, COST_MARGIN)))
    F19b = pd.DataFrame(rows)
    F19b["qtost"] = bh(F19b.ptost)
    # F20 secondary
    F20 = run_family(df, prim, ["cov", "dly", "eng"], "F20")
    pt = [tost_p((series(df, "LARS", "cov") - series(df, c, "cov")).to_numpy(float), COV_MARGIN) for c in prim]
    F20b = pd.DataFrame(dict(comp=prim, ptost=pt, qtost=bh(pt)))
    # F21 exploratory
    F21 = run_family(df, expl, ["cost", "cov", "dly", "eng"], "F21") if expl else pd.DataFrame()

    pd.set_option("display.width", 220)
    cols = ["comp", "metric", "lars", "comp_mean", "diff", "lo95", "hi95", "p", "q", "r"]
    print("\nF19  PRIMARY  cost, LARS minus comparator (negative = LARS cheaper)")
    print(F19[cols].round(4).to_string(index=False))
    print(f"\nF19b PRIMARY  cost TOST, margin +-{COST_MARGIN} (equivalent if qtost < 0.05)")
    print(F19b.round(4).to_string(index=False))
    print("\nF20  SECONDARY  coverage / delay / energy")
    print(F20[cols].round(4).to_string(index=False))
    print(f"\nF20b SECONDARY  coverage TOST, margin +-{COV_MARGIN}")
    print(F20b.round(4).to_string(index=False))
    if len(F21):
        print("\nF21  EXPLORATORY")
        print(F21[cols].round(4).to_string(index=False))

    # sensitivity (exploratory): other cost weightings, no re-tuning
    print("\nSensitivity of mean cost to the weights (exploratory; weights were tuned at 1/0.5/0.5 only)")
    for w in [(1, 0.5, 0.5), (1, 1, 1), (1, 0.25, 0.25), (0, 1, 1)]:
        print(w, {p: round(cost(df, p, w).mean(), 2) for p in ["LARS"] + prim})

    # training costs
    if a.train:
        tr = pd.read_csv(a.train)
        print("\nTraining-block cost (seeds 101-150)")
        print({p: round(cost(tr, p).mean(), 3) for p in allp if f"{p}_{MET['cov']}" in tr.columns})

    out = pd.concat([F19, F20, F21], ignore_index=True)
    out = out.merge(pd.concat([F19b.assign(metric="cost"), F20b.assign(metric="cov")]),
                    on=["comp", "metric"], how="left", suffixes=("", "_t"))
    out.to_csv(a.out, index=False)
    print(f"\nwritten {a.out}")


if __name__ == "__main__":
    main()
