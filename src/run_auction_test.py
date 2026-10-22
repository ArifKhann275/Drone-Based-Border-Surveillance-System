# -*- coding: utf-8 -*-
"""
run_auction_test.py

Compares Policy J with the auction baseline (AUC), nearest-feasible (B) and the
tuned TOPSIS / VIKOR policies (T_tuned, V_tuned) on a seed block.

Usage:
    python -u run_auction_test.py fresh              # seeds 201-400 (paper)
    python -u run_auction_test.py dev                # seeds 1-50
    python -u run_auction_test.py fresh --no-reuse   # re-run J/T/V too (needed
                                                     # to get THEIR reserve share)
    python -u run_auction_test.py fresh --lam=400    # override the AUC price (e.g. to
                                                     # report the original grid-edge value);
                                                     # output goes to auction_test_<block>_lam400.csv

Needs: tuned_AUC_lite.json, tuned_T_lite.json, tuned_V_lite.json.
By default J / T_tuned / V_tuned per-seed results are re-used from
tuned_test_<block>.csv (written by run_tuned_test_lite.py) when that file
covers the same seeds; the simulator is deterministic per seed.

Reports, for every policy vs J (diff = J - policy):
  paired mean difference with 95% CI, Wilcoxon p, rank-biserial r,
  BH-adjusted q over all difference tests, TOST coverage equivalence
  (margin +-2.0 points, BH over the TOST tests), and the reserve share
  (% of relay decisions served by an idle reserve) for newly run policies.
Per-seed results -> auction_test_<block>[_lam<value>].csv
"""

import sys
import os
import csv
import json
import numpy as np
from scipy.stats import wilcoxon, rankdata, t as t_dist

import a2e_relay_fixed_final_51 as relay
from policy_j_full import select_best_relay_j
from auction_baseline import make_auction_selector

METRICS = ("coverage_pct", "avg_relay_delay_steps", "avg_relay_energy_pct")
BLOCKS = {"dev": range(1, 51), "fresh": range(201, 401)}
MARGIN = 2.0

J_WEIGHTS = {
    "W_TRAVEL_ENERGY": 0.283556320123423,
    "W_COVERAGE_LOSS": 0.023813783478481596,
    "W_RELAY_DELAY": 0.28445380353708727,
    "W_BATTERY_MARGIN": 0.40817609286100814,
}


# ---------------------------------------------------------------- statistics
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


def tost_p(x, y, margin=MARGIN):
    """Paired TOST (two one-sided t tests); returns the larger of the two p-values."""
    d = np.asarray(x, float) - np.asarray(y, float)
    n = len(d)
    se = d.std(ddof=1) / np.sqrt(n)
    if se == 0:
        return 0.0 if abs(d.mean()) < margin else 1.0
    p_low = 1.0 - t_dist.cdf((d.mean() + margin) / se, n - 1)   # H0: diff <= -margin
    p_high = t_dist.cdf((d.mean() - margin) / se, n - 1)        # H0: diff >= +margin
    return float(max(p_low, p_high))


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


# ------------------------------------------------------------------- helpers
class Tracked:
    """Wraps a selector and counts how often the chosen candidate is an idle reserve."""

    def __init__(self, sel):
        self.sel, self.n, self.res = sel, 0, 0

    def __call__(self, *a, **k):
        chosen, diag = self.sel(*a, **k)
        if chosen is not None:
            self.n += 1
            self.res += int(chosen.get("role") == "idle_reserve")
        return chosen, diag

    def reset(self):
        self.n = self.res = 0

    def share(self):
        return 100.0 * self.res / self.n if self.n else float("nan")


def load_json(path):
    with open(path) as f:
        return json.load(f)


def load_cache(block, seeds):
    path = f"tuned_test_{block}.csv"
    if not os.path.exists(path):
        return None
    with open(path) as f:
        rows = {int(r["seed"]): r for r in csv.DictReader(f)}
    if not all(s in rows for s in seeds):
        return None
    return rows


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    block = args[0] if args else "fresh"
    reuse = "--no-reuse" not in sys.argv
    if block not in BLOCKS:
        raise SystemExit("block must be 'dev' or 'fresh'")
    seeds = list(BLOCKS[block])

    lam_override = next((float(a.split("=", 1)[1]) for a in sys.argv if a.startswith("--lam=")), None)
    lam = lam_override if lam_override is not None else load_json("tuned_AUC_lite.json")["lam"]
    suffix = f"_lam{int(lam)}" if lam_override is not None else ""
    wT = load_json("tuned_T_lite.json")["weights"]
    wV = load_json("tuned_V_lite.json")["weights"]
    print(f"AUC lam = {lam}" + ("  (command-line override)" if lam_override is not None else ""))

    selectors = {
        "J": lambda n, d, g, de, s: select_best_relay_j(n, d, g, de, s, weights=J_WEIGHTS, theta=0.5),
        "B": relay.select_nearest_relay,
        "AUC": make_auction_selector(lam),
        "T_tuned": lambda n, d, g, de, s: relay.select_topsis_relay(n, d, g, de, s, weights=wT),
        "V_tuned": lambda n, d, g, de, s: relay.select_vikor_relay(n, d, g, de, s, weights=wV, v=0.5),
    }

    cache = load_cache(block, seeds) if reuse else None
    cached_labels = {"J", "T_tuned", "V_tuned"} if cache else set()
    if cache:
        print("Re-using J / T_tuned / V_tuned per-seed results from "
              f"tuned_test_{block}.csv (use --no-reuse to re-run them).")
    run_labels = [l for l in selectors if l not in cached_labels]
    tracked = {l: Tracked(selectors[l]) for l in run_labels}
    shares = {l: [] for l in run_labels}

    rows = []
    print(f"Running {len(seeds)} seeds ({block} block) for: {', '.join(run_labels)}")
    for seed in seeds:
        row = {"seed": seed}
        for label in selectors:
            if label in cached_labels:
                for k in METRICS:
                    row[f"{label}_{k}"] = float(cache[seed][f"{label}_{k}"])
            else:
                tr = tracked[label]
                tr.reset()
                m = relay.simulate_fleet(seed, relay_selector=tr, max_steps=500)
                for k in METRICS:
                    row[f"{label}_{k}"] = m[k]
                row[f"{label}_reserve_share"] = tr.share()
                shares[label].append(tr.share())
        rows.append(row)

    all_keys = []
    for r in rows:
        for k in r:
            if k not in all_keys:
                all_keys.append(k)
    out_csv = f"auction_test_{block}{suffix}.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=all_keys, restval="")
        w.writeheader()
        w.writerows(rows)

    others = [l for l in selectors if l != "J"]
    results = []
    for base in others:
        for m in METRICS:
            j = np.array([r[f"J_{m}"] for r in rows], float)
            b = np.array([r[f"{base}_{m}"] for r in rows], float)
            diff, lo, hi = paired_ci(j, b)
            results.append(dict(base=base, metric=m, jm=j.mean(), bm=b.mean(), diff=diff,
                                lo=lo, hi=hi, p=wilcoxon_p(j, b), r=rank_biserial(j, b),
                                tost=tost_p(j, b) if m == "coverage_pct" else None))

    q = bh_adjust([x["p"] for x in results])
    tost_idx = [i for i, x in enumerate(results) if x["tost"] is not None]
    q_tost = bh_adjust([results[i]["tost"] for i in tost_idx])
    for x, qq in zip(results, q):
        x["q"] = qq
    for i, qq in zip(tost_idx, q_tost):
        results[i]["q_tost"] = qq

    print(f"\n{'=' * 118}\nJ vs other policies -- {block} block, n={len(seeds)}   (diff = J - policy)\n{'=' * 118}")
    print(f"{'vs':<9}{'metric':<24}{'J':>8}{'policy':>9}{'diff [95% CI]':>26}{'p':>9}{'q_BH':>9}{'r':>7}"
          f"{'q_TOST(+-2)':>13}")
    for x in results:
        eq = ""
        if x["tost"] is not None:
            eq = f"{x['q_tost']:>10.4f} {'EQUIV' if x['q_tost'] < 0.05 else 'not equiv'}"
        print(f"{x['base']:<9}{x['metric']:<24}{x['jm']:>8.3f}{x['bm']:>9.3f}"
              f"{x['diff']:>+10.2f} [{x['lo']:+.2f},{x['hi']:+.2f}]{x['p']:>9.4f}{x['q']:>9.4f}{x['r']:>7.2f}  {eq}")

    print("\nFor delay and energy a negative diff favors J; for coverage a positive diff favors J.")
    print("\nReserve share (% of relay decisions served by an idle reserve), mean over seeds:")
    for l, v in shares.items():
        if v:
            print(f"  {l:<8}{np.nanmean(v):6.1f}%")
    if cached_labels:
        print("  (J / T_tuned / V_tuned not measured in this run; use --no-reuse)")


if __name__ == "__main__":
    main()