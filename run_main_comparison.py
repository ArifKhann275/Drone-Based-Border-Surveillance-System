# -*- coding: utf-8 -*-
"""
run_main_comparison.py
══════════════════════════════════════════════════════════════
The paper's MAIN comparison on one block of seeds, in one run:

    Policy J   vs   D (true FIS), T (TOPSIS), V (VIKOR), D2 (surrogate FIS),
                    B (nearest feasible), BR (nearest idle reserve)

Policies C and I (and H, if you add it) are NOT competitors here.  They are
run last, in a separate ABLATION block that only answers "does J's added
machinery help over its own building blocks?" (skip it with --no-ablation).

Why this script exists: the fresh-block run (seeds 201-400) contained
J, D, D-indep*, C, I, B, BR but NOT T, V, D2; T and V had the highest
coverage among the multi-attribute policies on seeds 1-50, so the paper's
main comparison is incomplete until they are run on a fresh block too.

USAGE (fresh confirmatory block; 200 seeds)
    python -u run_main_comparison.py --seed-start 201 --seeds 200 --steps 500 \\
        --w-energy 0.283556320123423 --w-coverage 0.023813783478481596 \\
        --w-delay 0.28445380353708727 --w-battery 0.40817609286100814 --theta 0.5

USAGE (sanity run on the legacy block; must reproduce the earlier numbers)
    python -u run_main_comparison.py --seed-start 1 --seeds 50 ... (same weights)
    -> the SANITY block compares J, D, T, V, D2, B, BR, C, I with the earlier runs.

CALL ORDER (fixed, do not reorder; the project is known to be call-order
sensitive): J, D, T, V, D2, B, BR, then C, I (ablation) last.

OUTPUT
    main_comparison.csv   every scalar field of simulate_fleet for every
                          policy + idle-reserve share per policy
    console               means, J-vs-X tests (paired Wilcoxon, rank-based r,
                          paired-t 95% CI, BH over the main family), a plain
                          "who is better" matrix, equivalence (TOST) for
                          coverage, and the ablation block.

Sign convention: difference = J minus the other policy.  Coverage: positive
favours J.  Delay and energy: negative favours J.

EQUIVALENCE (TOST) -- the margin must be justified BEFORE looking at results
(default 2.0 coverage points; change with --margin and report the change).
"""

import argparse

import numpy as np
from scipy.stats import t as tdist

import a2e_relay_fixed_final_51 as relay
import run_hybrid as rh

HEADLINE = rh.HEADLINE
SHORT = rh.SHORT
MAIN = ["D", "T", "V", "D2", "B", "BR"]
ABLATION = ["C", "I"]

KNOWN = {   # means on seeds 1-50 from the earlier runs (cov, delay, energy)
    "J": (90.568, 13.457, 9.876), "D": (90.732, 14.520, 11.837),
    "T": (91.316, 15.798, 12.567), "V": (91.116, 16.808, 13.396),
    "D2": (90.136, 14.545, 11.907), "B": (88.846, 11.620, 10.018),
    "BR": (93.772, 19.491, 16.156), "C": (88.794, 14.217, 10.453),
    "I": (87.926, 13.455, 9.030),
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--w-energy", type=float, default=None)
    p.add_argument("--w-coverage", type=float, default=None)
    p.add_argument("--w-delay", type=float, default=None)
    p.add_argument("--w-battery", type=float, default=None)
    p.add_argument("--theta", type=float, default=0.5)
    p.add_argument("--seed-start", type=int, default=201)
    p.add_argument("--seeds", type=int, default=200)
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--num-drones", type=int, default=None)
    p.add_argument("--num-reserves", type=int, default=None)
    p.add_argument("--margin", type=float, default=2.0, help="TOST equivalence margin, coverage points")
    p.add_argument("--no-ablation", action="store_true")
    p.add_argument("--out-csv", default="main_comparison.csv")
    return p.parse_args()


def tost_p(x, y, margin):
    """Paired two one-sided t-tests; returns (p_TOST, mean diff, 90% CI)."""
    d = np.asarray(x, float) - np.asarray(y, float)
    n = len(d)
    m = d.mean()
    se = d.std(ddof=1) / np.sqrt(n)
    p_lo = 1.0 - tdist.cdf((m + margin) / se, n - 1)     # H0: true diff <= -margin
    p_hi = tdist.cdf((m - margin) / se, n - 1)           # H0: true diff >= +margin
    tc = tdist.ppf(0.95, n - 1)
    return max(p_lo, p_hi), m, m - tc * se, m + tc * se


def run_tests(rows, refs):
    tests = []
    for r in refs:
        for m in HEADLINE:
            x, y = rh.col(rows, "J", m), rh.col(rows, r, m)
            d, lo, hi = rh.paired_ci(x, y)
            tests.append(dict(ref=r, m=m, d=d, lo=lo, hi=hi,
                              p=rh.wtest(x, y), rb=rh.rank_biserial(x, y)))
    for t, q in zip(tests, rh.bh([t["p"] for t in tests])):
        t["q"] = q
    return tests


def verdict(t):
    better = (t["d"] > 0) if t["m"] == "coverage_pct" else (t["d"] < 0)
    if t["q"] >= 0.05:
        return "no significant difference"
    return "J significantly BETTER" if better else "J significantly WORSE"


def print_tests(tests, title):
    print(f"\n{'=' * 100}\n{title}\n{'=' * 100}")
    cur = None
    for t in tests:
        if t["ref"] != cur:
            cur = t["ref"]
            print(f"\n--- J vs {cur} ---")
            print(f"{'metric':<24}{'diff':>9}{'95% CI':>22}{'p':>10}{'q_BH':>10}{'r':>8}")
        star = " *" if t["q"] < 0.05 else ""
        print(f"{t['m']:<24}{t['d']:>+9.3f}  [{t['lo']:>+7.3f},{t['hi']:>+7.3f}]"
              f"{t['p']:>10.4f}{t['q']:>10.4f}{t['rb']:>8.3f}{star}")
    print("\n  * = q_BH < 0.05")


def main():
    args = parse_args()
    labels = ["J"] + MAIN + ([] if args.no_ablation else ABLATION)
    selectors, jw = rh.build_selectors(args, [])
    for lab in ("T", "V", "D2"):
        selectors[lab] = relay.POLICY_SELECTORS[lab]

    lo, hi = args.seed_start, args.seed_start + args.seeds - 1
    print(f"J weights: {jw or rh.NORMALIZED_WEIGHTS_J_DEFAULT} theta={args.theta}")
    print(f"Block: seeds {lo}..{hi} (n={args.seeds}), steps={args.steps}")
    if lo <= 150 and hi >= 101:
        print("!! WARNING: overlaps the TRAINING seeds 101-150; not confirmatory.")
    if lo <= 50:
        print("!! NOTE: seeds 1-50 were reused many times; treat results on them as descriptive.")
    print(f"Per-seed order (fixed): {labels}\n")

    rows = rh.simulate(args, labels, selectors, args.out_csv)

    if args.seed_start == 1 and args.seeds == 50:
        print(f"\n{'=' * 100}\nSANITY CHECKS (seeds 1-50) -- must reproduce the earlier runs\n{'=' * 100}")
        for lab in labels:
            for m, k in zip(HEADLINE, KNOWN[lab]):
                mean = rh.mean_of(rows, lab, m)
                flag = "OK   " if abs(mean - k) <= 0.01 else "DRIFT"
                print(f"  [{flag}] {lab:<3} {m:<24} this run {mean:8.3f}  expected {k:8.3f}  diff {mean-k:+.3f}")

    print(f"\n{'=' * 100}\nMEANS (n={len(rows)})\n{'=' * 100}")
    print(f"{'policy':<8}{'cov':>9}{'delay':>9}{'energy':>9}{'reserve%':>10}")
    for lab in labels:
        v = [rh.mean_of(rows, lab, m) for m in HEADLINE]
        print(f"{lab:<8}{v[0]:>9.2f}{v[1]:>9.2f}{v[2]:>9.2f}{100*rh.reserve_share(rows, lab):>10.1f}")

    main_tests = run_tests(rows, MAIN)
    print_tests(main_tests, f"MAIN COMPARISON -- J vs {', '.join(MAIN)}  ({len(main_tests)}-test BH family)")

    print(f"\n{'=' * 100}\nWHO IS BETTER? (J compared with each policy, BH-corrected)\n{'=' * 100}")
    print(f"{'vs':<6}" + "".join(f"{SHORT[m]:<34}" for m in HEADLINE))
    for r in MAIN:
        cells = [verdict(next(t for t in main_tests if t["ref"] == r and t["m"] == m)) for m in HEADLINE]
        print(f"{r:<6}" + "".join(f"{c:<34}" for c in cells))

    print(f"\n{'=' * 100}\nCOVERAGE EQUIVALENCE (TOST), margin = +/-{args.margin} points\n{'=' * 100}")
    eq = []
    for r in MAIN:
        p, d, l90, h90 = tost_p(rh.col(rows, "J", "coverage_pct"), rh.col(rows, r, "coverage_pct"), args.margin)
        eq.append((r, p, d, l90, h90))
    for (r, p, d, l90, h90), q in zip(eq, rh.bh([e[1] for e in eq])):
        print(f"  J vs {r:<3} diff {d:+.2f}  90% CI [{l90:+.2f},{h90:+.2f}]  p_TOST={p:.4f}  q={q:.4f}  "
              f"{'EQUIVALENT within margin' if q < 0.05 else 'equivalence NOT shown'}")
    print("  (BR is expected to fail: it is genuinely different. Equivalence is only meaningful where "
          "the difference is small.)")

    if not args.no_ablation:
        abl = run_tests(rows, ABLATION)
        print_tests(abl, "ABLATION (separate family; C and I are J's building blocks, not competitors)")

    print("\nReport the result either way. Do not change the margin, the block or the policy set after "
          "seeing this output.")


if __name__ == "__main__":
    main()
