# -*- coding: utf-8 -*-
"""
run_tuned_test_lite.py (v2)

Compares Policy J with the tuned TOPSIS (T) and VIKOR (V) baselines.

Usage:
    python -u run_tuned_test_lite.py dev      # seeds 1-50   (exploratory block)
    python -u run_tuned_test_lite.py fresh    # seeds 201-400 (use this one for the paper)

Reads tuned_T_lite.json and tuned_V_lite.json (written by tune_baselines.py).
Reports paired mean difference with 95% CI, Wilcoxon p, rank-biserial r, and
Benjamini-Hochberg q over the 6 tests (2 baselines x 3 metrics).
Per-seed results are saved to tuned_test_<block>.csv.
"""

import sys
import json
import csv
import numpy as np
from scipy.stats import wilcoxon, rankdata, t as t_dist

import a2e_relay_fixed_final_51 as relay
from policy_j_full import select_best_relay_j

CRITERIA = ("distance", "battery_margin", "coverage_loss", "relay_delay")
METRICS = ("coverage_pct", "avg_relay_delay_steps", "avg_relay_energy_pct")
BLOCKS = {"dev": range(1, 51), "fresh": range(201, 401)}

J_WEIGHTS = {
    "W_TRAVEL_ENERGY": 0.283556320123423,
    "W_COVERAGE_LOSS": 0.023813783478481596,
    "W_RELAY_DELAY": 0.28445380353708727,
    "W_BATTERY_MARGIN": 0.40817609286100814,
}


def rank_biserial(x, y):
    d = np.asarray(x, float) - np.asarray(y, float)
    nz = d[d != 0]
    if len(nz) == 0:
        return 0.0
    ranks = rankdata(np.abs(nz))
    pos, neg = ranks[nz > 0].sum(), ranks[nz < 0].sum()
    return float((pos - neg) / (pos + neg))


def wilcoxon_p(x, y):
    d = np.asarray(x, float) - np.asarray(y, float)
    return float(wilcoxon(x, y).pvalue) if np.any(d != 0) else 1.0


def paired_ci(x, y, level=0.95):
    d = np.asarray(x, float) - np.asarray(y, float)
    n = len(d)
    m = d.mean()
    se = d.std(ddof=1) / np.sqrt(n)
    h = t_dist.ppf(0.5 + level / 2, n - 1) * se
    return float(m), float(m - h), float(m + h)


def bh_adjust(pvals):
    p = np.asarray(pvals, float)
    n = len(p)
    order = np.argsort(p)
    adj = np.empty(n)
    prev = 1.0
    for rank_from_end, idx in enumerate(order[::-1]):
        rank = n - rank_from_end
        prev = min(prev, p[idx] * n / rank)
        adj[idx] = prev
    return adj


def load_weights(key):
    with open(f"tuned_{key}_lite.json") as f:
        return json.load(f)["weights"]


def main():
    block = sys.argv[1] if len(sys.argv) > 1 else "dev"
    if block not in BLOCKS:
        raise SystemExit("block must be 'dev' or 'fresh'")
    seeds = list(BLOCKS[block])

    wT, wV = load_weights("T"), load_weights("V")
    print("T weights:", wT)
    print("V weights:", wV)

    def j_sel(n, d, g, de, s):
        return select_best_relay_j(n, d, g, de, s, weights=J_WEIGHTS, theta=0.5)

    def t_sel(n, d, g, de, s):
        return relay.select_topsis_relay(n, d, g, de, s, weights=wT)

    def v_sel(n, d, g, de, s):
        return relay.select_vikor_relay(n, d, g, de, s, weights=wV, v=0.5)

    selectors = {"J": j_sel, "T_tuned": t_sel, "V_tuned": v_sel}

    rows = []
    print(f"Running {len(seeds)} seeds ({block} block) ...")
    for seed in seeds:
        row = {"seed": seed}
        for label, sel in selectors.items():
            m = relay.simulate_fleet(seed, relay_selector=sel, max_steps=500)
            for k in METRICS:
                row[f"{label}_{k}"] = m[k]
        rows.append(row)

    with open(f"tuned_test_{block}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    results = []
    for base in ("T_tuned", "V_tuned"):
        for m in METRICS:
            j = np.array([r[f"J_{m}"] for r in rows], float)
            b = np.array([r[f"{base}_{m}"] for r in rows], float)
            diff, lo, hi = paired_ci(j, b)
            results.append((base, m, j.mean(), b.mean(), diff, lo, hi,
                            wilcoxon_p(j, b), rank_biserial(j, b)))

    q = bh_adjust([r[7] for r in results])

    print(f"\n{'=' * 100}\nJ vs tuned T/V -- {block} block, n={len(seeds)}  (diff = J - baseline)\n{'=' * 100}")
    print(f"{'vs':<9}{'metric':<24}{'J':>8}{'base':>9}{'diff [95% CI]':>26}{'p':>10}{'q_BH':>9}{'r':>8}")
    for (base, m, jm, bm, diff, lo, hi, p, r), qq in zip(results, q):
        print(f"{base:<9}{m:<24}{jm:>8.3f}{bm:>9.3f}"
              f"{diff:>+10.2f} [{lo:+.2f},{hi:+.2f}]{p:>10.4f}{qq:>9.4f}{r:>8.2f}")
    print("\nFor delay and energy a negative diff favors J; for coverage a positive diff favors J.")


if __name__ == "__main__":
    main()