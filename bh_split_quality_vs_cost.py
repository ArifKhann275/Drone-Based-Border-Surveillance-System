# -*- coding: utf-8 -*-
"""
bh_split_quality_vs_cost.py
══════════════════════════════════════════════════════════════
Recomputes BH correction from the already-saved circularity_check_with_j.csv
(raw per-seed data -- no re-simulation needed), but splits the test family
in two ways that make the correction fairer and more interpretable:

  1. QUALITY metrics (coverage_pct, delay, energy, staleness, mission
     completion, etc.) are corrected as their OWN family, separate from
     COST metrics (selector_time -- trivially significant everywhere
     since the FIS is slower by construction, and not the paper's
     actual research question). Mixing them inflates the quality
     family's correction for no scientific reason.

  2. D_indep is DROPPED from all comparisons. It is documented in
     policy_d_true_fis.py (RULE_PROFILES["indep"]'s own comment) to
     collapse onto D in reserve-dominated fleets -- all three of its
     vetoes reference coverage_loss, which an idle reserve candidate
     never triggers, so "indep" and "expert" (D) agree on ~77/81 cells
     in this regime. Confirmed here: D and D_indep gave IDENTICAL
     coverage_pct on all 50 seeds. Testing against it adds a redundant,
     duplicate hypothesis to the family and does nothing but weaken
     correction power for the comparisons that DO carry independent
     information (vs D, vs D_indep2).

USAGE
-----
    python bh_split_quality_vs_cost.py --csv circularity_check_with_j.csv
"""
import argparse
import csv as csvmod
from collections import defaultdict

import numpy as np
from scipy.stats import wilcoxon

COST_METRICS = {"avg_selector_time_ms", "median_selector_time_ms",
                "trimmed_mean_selector_time_ms", "total_selector_time_ms"}

# Every other numeric metric present in the CSV is treated as "quality"
# automatically (see main()).

POLICIES_A = ("C", "I", "J")
POLICIES_B = ("D", "D_indep2")   # D_indep intentionally dropped -- see docstring


def safe_wilcoxon(a, b):
    a, b = np.array(a, dtype=float), np.array(b, dtype=float)
    if np.all(a - b == 0):
        return 1.0
    return wilcoxon(a, b).pvalue


def bh_correct(pvals):
    indexed = [(i, p) for i, p in enumerate(pvals) if p is not None]
    n = len(indexed)
    if n == 0:
        return [None] * len(pvals)
    indexed.sort(key=lambda x: x[1])
    q = [None] * len(pvals)
    prev_q = 1.0
    for rank in range(n, 0, -1):
        i, p = indexed[rank - 1]
        q_val = min(prev_q, p * n / rank)
        q[i] = q_val
        prev_q = q_val
    return q


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="circularity_check_with_j.csv")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--out-prefix", default="circularity_split")
    args = p.parse_args()

    with open(args.csv, newline="") as fh:
        rows = list(csvmod.DictReader(fh))
    print(f"Loaded {len(rows)} seed-rows from {args.csv}")

    # discover metrics present for all 5 policies (C, D, D_indep2, I, J)
    sample = rows[0]
    metrics = sorted({k.split("_", 1)[1] for k in sample if k.startswith("C_")
                      if f"D_{k.split('_',1)[1]}" in sample
                      and f"D_indep2_{k.split('_',1)[1]}" in sample
                      and f"I_{k.split('_',1)[1]}" in sample
                      and f"J_{k.split('_',1)[1]}" in sample})

    def build_family(metric_filter):
        records = []
        for metric in metrics:
            if not metric_filter(metric):
                continue
            for a in POLICIES_A:
                for b in POLICIES_B:
                    a_key, b_key = f"{a}_{metric}", f"{b}_{metric}"
                    try:
                        a_vals = [float(r[a_key]) for r in rows]
                        b_vals = [float(r[b_key]) for r in rows]
                    except (KeyError, ValueError):
                        continue
                    p_val = safe_wilcoxon(a_vals, b_vals)
                    records.append({
                        "metric": metric, "a": a, "b": b,
                        "a_mean": round(float(np.mean(a_vals)), 4),
                        "b_mean": round(float(np.mean(b_vals)), 4),
                        "p": p_val,
                    })
        pvals = [r["p"] for r in records]
        qvals = bh_correct(pvals)
        for r, q in zip(records, qvals):
            r["q_bh"] = round(q, 6) if q is not None else None
            r["survives_bh"] = (q is not None and q < args.alpha)
        return records

    quality_records = build_family(lambda m: m not in COST_METRICS)
    cost_records = build_family(lambda m: m in COST_METRICS)

    for name, records in (("QUALITY", quality_records), ("COST", cost_records)):
        out_path = f"{args.out_prefix}_{name.lower()}.csv"
        with open(out_path, "w", newline="") as fh:
            w = csvmod.DictWriter(fh, fieldnames=["metric", "a", "b", "a_mean", "b_mean",
                                                  "p", "q_bh", "survives_bh"])
            w.writeheader()
            w.writerows(records)

        n = len(records)
        n_survive = sum(1 for r in records if r["survives_bh"])
        print(f"\n{'='*100}\n{name} metrics family -- {n} tests, D_indep dropped "
              f"(redundant, see docstring)\n{'='*100}")
        for r in sorted(records, key=lambda x: x["p"]):
            flag = "  <-- SURVIVES BH" if r["survives_bh"] else ""
            print(f"  {r['metric']:<26} {r['a']:>4} vs {r['b']:<10} "
                  f"{r['a_mean']:>12} vs {r['b_mean']:<12} p={r['p']:.5f} "
                  f"q_bh={r['q_bh']:.5f}{flag}")
        print(f"\n{n_survive}/{n} {name.lower()} tests survive BH at alpha={args.alpha}.")
        print(f"Written: {out_path}")

    print(f"\n{'='*100}\nHEADLINE: coverage_pct, J vs D_indep2 (the circularity question)\n{'='*100}")
    hl = [r for r in quality_records if r["metric"] == "coverage_pct" and r["a"] == "J" and r["b"] == "D_indep2"]
    if hl:
        r = hl[0]
        print(f"  J={r['a_mean']}%  D_indep2={r['b_mean']}%  p={r['p']:.5f}  "
              f"q_bh={r['q_bh']:.5f}  survives_bh={r['survives_bh']}")


if __name__ == "__main__":
    main()
