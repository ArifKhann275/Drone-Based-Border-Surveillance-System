# -*- coding: utf-8 -*-
"""
run_heuristics_vs_j.py
══════════════════════════════════════════════════════════════
Adds four simple heuristic baselines to the held-out evaluation so the
paper can answer "is the MADM machinery (FIS / VIKOR / weight tuning)
actually needed?":

    B    nearest feasible candidate
    BR   nearest feasible idle reserve (patrol drone only as fallback)
    HB   highest current battery among feasible candidates
    RND  uniformly random feasible candidate (stateless, reproducible)

CALL ORDER (read before editing) -- same rule as run_c_i_vs_multi_indep.py
--------------------------------------------------------------------------
The validated per-seed sequence is  J, D, Dindep, Dindep2, Dindep3,
Dindep4, C, I.  Results have been sensitive to call order for reasons the
project never fully isolated, so this script runs that ENTIRE validated
prefix unchanged and APPENDS B, BR, HB, RND at the end, seed by seed.
Do not reorder, do not batch per policy, and do not drop the prefix to
"save time": the prefix is what lets Sanity Checks 1 and 2 certify that
appending the heuristics did not perturb anything.

USAGE
-----
    python run_heuristics_vs_j.py --seeds 50 --steps 500 \\
        --w-energy 0.283556320123423 --w-coverage 0.023813783478481596 \\
        --w-delay 0.28445380353708727 --w-battery 0.40817609286100814 \\
        --theta 0.5

OUTPUT
------
    heuristics_j_heldout.csv   raw per-seed data, every scalar field of
                               simulate_fleet, every policy label
    console report             means for all policies (headline metrics),
                               then J and D vs each heuristic: mean diff,
                               paired-t 95% CI, Wilcoxon p, rank-based r,
                               and BH-adjusted q over the whole
                               (2 policies x 4 heuristics x 3 metrics =
                               24-test) family.
Sign convention: difference = first-named minus second-named.  For delay
and energy a negative value favours the first-named policy; for coverage
a negative value means the first-named policy has LOWER coverage.
"""

import argparse
import csv

import numpy as np
from scipy.stats import wilcoxon, rankdata, t as tdist

import a2e_relay_fixed_final_51 as relay
import policy_heuristics as heur
from policy_j_full import select_best_relay_j, NORMALIZED_WEIGHTS_J_DEFAULT
from policy_d_true_fis import (
    select_fuzzy_relay_true_fis_indep_weights,
    select_fuzzy_relay_true_fis_indep2_weights,
    select_fuzzy_relay_true_fis_indep3_weights,
    select_fuzzy_relay_true_fis_indep4_weights,
)

HEADLINE = ("coverage_pct", "avg_relay_delay_steps", "avg_relay_energy_pct")

FIS_BASELINES = {
    "D":       lambda needer, drones, g, detected, step: relay.POLICY_SELECTORS["D"](needer, drones, g, detected, step),
    "Dindep":  select_fuzzy_relay_true_fis_indep_weights,
    "Dindep2": select_fuzzy_relay_true_fis_indep2_weights,
    "Dindep3": select_fuzzy_relay_true_fis_indep3_weights,
    "Dindep4": select_fuzzy_relay_true_fis_indep4_weights,
}

HEURISTIC_LABELS = ["B", "BR", "HB", "RND"]
REPORT_POLICIES = ["J", "D"]

# Ground truth for the sanity checks (same numbers as run_c_i_vs_multi_indep.py)
KNOWN = {
    "J": {"coverage_pct": 90.568, "avg_relay_delay_steps": 13.457, "avg_relay_energy_pct": 9.876},
    "C": {"coverage_pct": 88.79,  "avg_relay_delay_steps": 14.22,  "avg_relay_energy_pct": 10.45},
    "I": {"coverage_pct": 87.93,  "avg_relay_delay_steps": 13.46,  "avg_relay_energy_pct": 9.03},
}
TOL = {"J": 0.01, "C": 0.02, "I": 0.02}   # absolute tolerance on each mean


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--w-energy", type=float, default=None)
    p.add_argument("--w-coverage", type=float, default=None)
    p.add_argument("--w-delay", type=float, default=None)
    p.add_argument("--w-battery", type=float, default=None)
    p.add_argument("--theta", type=float, default=0.5)
    p.add_argument("--seed-start", type=int, default=1)
    p.add_argument("--seeds", type=int, default=50,
                   help="keep this the SAME held-out block used everywhere else "
                        "in the paper (seeds 1-50)")
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--num-drones", type=int, default=None)
    p.add_argument("--num-reserves", type=int, default=None)
    p.add_argument("--random-salt", type=int, default=0,
                   help="changes the (reproducible) random policy draw")
    p.add_argument("--out-csv", default="heuristics_j_heldout.csv")
    args = p.parse_args()
    ws = [args.w_energy, args.w_coverage, args.w_delay, args.w_battery]
    if any(w is not None for w in ws) and any(w is None for w in ws):
        p.error("give all four --w-* weights or none of them")
    return args


def _j_weights(args):
    if args.w_energy is None:
        return None
    total = args.w_energy + args.w_coverage + args.w_delay + args.w_battery
    return {
        "W_TRAVEL_ENERGY": args.w_energy / total,
        "W_COVERAGE_LOSS": args.w_coverage / total,
        "W_RELAY_DELAY": args.w_delay / total,
        "W_BATTERY_MARGIN": args.w_battery / total,
    }


# ── statistics (identical definitions to run_c_i_vs_multi_indep.py) ──────
def _rank_biserial(x, y):
    d = np.asarray(x, float) - np.asarray(y, float)
    nz = d[d != 0]
    if len(nz) == 0:
        return 0.0
    ranks = rankdata(np.abs(nz))
    rp, rn = ranks[nz > 0].sum(), ranks[nz < 0].sum()
    return float((rp - rn) / (rp + rn))


def _wtest(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    return wilcoxon(x, y).pvalue if np.any(x != y) else 1.0


def _paired_ci95(x, y):
    d = np.asarray(x, float) - np.asarray(y, float)
    n = len(d)
    m = d.mean()
    se = d.std(ddof=1) / np.sqrt(n) if n > 1 else 0.0
    tc = tdist.ppf(0.975, n - 1) if n > 1 else 0.0
    return m, m - tc * se, m + tc * se


def _bh(pvals):
    p = np.asarray(pvals, float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    q_sorted = np.minimum.accumulate(ranked[::-1])[::-1]
    q = np.empty(n)
    q[order] = np.minimum(q_sorted, 1.0)
    return q


# ── simulation ────────────────────────────────────────────────────────────
def run(seeds, steps, num_drones, num_reserves, j_weights, theta):
    def j_selector(needer, drones, g, detected, step):
        return select_best_relay_j(needer, drones, g, detected, step,
                                   weights=j_weights, theta=theta)

    # Validated prefix UNCHANGED; heuristics appended at the very end.
    prefix = ["J"] + list(FIS_BASELINES.keys()) + ["C", "I"]
    ordered_labels = prefix + HEURISTIC_LABELS
    selectors = {
        "J": j_selector,
        **FIS_BASELINES,
        "C": relay.select_best_relay,
        "I": relay.select_best_relay_coverage_aware,
        **heur.HEURISTIC_SELECTORS,
    }

    rows = []
    for seed in seeds:
        row = {"seed": seed}
        for label in ordered_labels:
            m = relay.simulate_fleet(seed, relay_selector=selectors[label],
                                     num_drones=num_drones,
                                     num_reserves=num_reserves,
                                     max_steps=steps)
            for k, v in m.items():
                if isinstance(v, (dict, list, tuple, set)):
                    continue
                row[f"{label}_{k}"] = float(v) if isinstance(v, bool) else v
        rows.append(row)
        print(f"  seed {seed} done")

    sample = rows[0]
    common = sorted({
        k.split("_", 1)[1] for k in sample if k.startswith("J_")
        if all(f"{lab}_{k.split('_', 1)[1]}" in sample for lab in ordered_labels)
    })
    print(f"\nDiscovered {len(common)} metric fields present for every policy: {common}")
    return rows, ordered_labels


# ── report ────────────────────────────────────────────────────────────────
def _col(rows, label, metric):
    return np.array([r[f"{label}_{metric}"] for r in rows], dtype=float)


def report(rows, labels):
    n = len(rows)
    print(f"\n{'=' * 100}\nSANITY CHECKS -- the validated prefix must reproduce known numbers "
          f"(appending the heuristics must not perturb it)\n{'=' * 100}")
    ok = True
    for lab, known in KNOWN.items():
        for m in HEADLINE:
            mean = _col(rows, lab, m).mean()
            diff = mean - known[m]
            flag = "OK " if abs(diff) <= TOL[lab] else "DRIFT"
            ok &= (flag == "OK ")
            print(f"  [{flag}] {lab:<2} {m:<24} this run {mean:8.3f}   expected {known[m]:8.3f}   diff {diff:+.3f}")
    if not ok:
        print("\n  !! At least one sanity check drifted. Do NOT use the heuristic "
              "numbers below until this is resolved (see module docstring).")

    print(f"\n{'=' * 100}\nMEANS -- {n} held-out seeds\n{'=' * 100}")
    print(f"{'policy':<10}" + "".join(f"{m:>26}" for m in HEADLINE))
    for lab in labels:
        if lab == "Dindep":
            continue            # bit-identical to D on outcome fields
        names = heur.HEURISTIC_DISPLAY_NAMES.get(lab, lab)
        print(f"{lab:<10}" + "".join(f"{_col(rows, lab, m).mean():>26.3f}" for m in HEADLINE)
              + (f"   {names}" if lab in heur.HEURISTIC_DISPLAY_NAMES else ""))

    tests = []
    for pol in REPORT_POLICIES:
        for hl in HEURISTIC_LABELS:
            for m in HEADLINE:
                x, y = _col(rows, pol, m), _col(rows, hl, m)
                mean, lo, hi = _paired_ci95(x, y)
                tests.append(dict(pol=pol, h=hl, m=m, xm=x.mean(), ym=y.mean(),
                                  d=mean, lo=lo, hi=hi, p=_wtest(x, y),
                                  r=_rank_biserial(x, y)))
    q = _bh([t["p"] for t in tests])
    for t, qi in zip(tests, q):
        t["q"] = qi

    print(f"\n{'=' * 100}\nPOLICY vs HEURISTIC -- {len(tests)}-test BH family "
          f"(difference = policy minus heuristic)\n{'=' * 100}")
    for pol in REPORT_POLICIES:
        for hl in HEURISTIC_LABELS:
            print(f"\n--- {pol} vs {hl} ({heur.HEURISTIC_DISPLAY_NAMES[hl]}) ---")
            print(f"{'metric':<26}{pol:>10}{hl:>10}{'diff':>10}{'95% CI':>22}{'p':>10}{'q_BH':>10}{'r':>8}")
            print("-" * 106)
            for t in tests:
                if t["pol"] == pol and t["h"] == hl:
                    star = " *" if t["q"] < 0.05 else ""
                    print(f"{t['m']:<26}{t['xm']:>10.3f}{t['ym']:>10.3f}{t['d']:>+10.3f}"
                          f"  [{t['lo']:>+7.3f},{t['hi']:>+7.3f}]{t['p']:>10.4f}{t['q']:>10.4f}{t['r']:>8.3f}{star}")
    print("\n  * = q_BH < 0.05")

    # The question this experiment exists to answer, stated mechanically.
    print(f"\n{'=' * 100}\nREADING GUIDE\n{'=' * 100}")
    j_cov = _col(rows, "J", "coverage_pct").mean()
    for hl in HEURISTIC_LABELS:
        h_cov = _col(rows, hl, "coverage_pct").mean()
        print(f"  {hl:<4} coverage {h_cov:6.2f}  vs J {j_cov:6.2f}  (J - {hl} = {j_cov - h_cov:+.2f} points)")
    print("  If a heuristic's coverage is within ~1-2 points of J AND J's delay/energy "
          "advantage over it is small or insignificant, say so in the paper: the "
          "contribution then rests on the delay/energy trade-off, not on coverage.")


def main():
    args = _parse_args()
    heur.RANDOM_SALT = args.random_salt
    seeds = list(range(args.seed_start, args.seed_start + args.seeds))
    num_drones = args.num_drones if args.num_drones is not None else relay.NUM_DRONES
    num_reserves = args.num_reserves if args.num_reserves is not None else relay.RESERVE_POOL_SIZE
    j_weights = _j_weights(args)

    print(f"J weights: {j_weights or NORMALIZED_WEIGHTS_J_DEFAULT}  theta={args.theta}")
    print(f"Held-out block: seeds {seeds[0]}..{seeds[-1]}, max_steps={args.steps}, "
          f"Nf={num_drones + num_reserves}, random_salt={args.random_salt}")
    print("Per-seed call order: J, D, Dindep, Dindep2, Dindep3, Dindep4, C, I, "
          "B, BR, HB, RND (fixed -- do not reorder)\n")

    rows, labels = run(seeds, args.steps, num_drones, num_reserves, j_weights, args.theta)

    with open(args.out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWritten: {args.out_csv}")

    report(rows, labels)


if __name__ == "__main__":
    main()
