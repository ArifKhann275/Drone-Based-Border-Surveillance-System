# -*- coding: utf-8 -*-
"""
run_c_i_vs_multi_indep.py
══════════════════════════════════════════════════════════════
Extends run_j_vs_multi_indep.py to also run Policies C and I against
all five FIS baselines (D, D-indep, D-indep2, D-indep3, D-indep4),
so the extended independence check (paper Section VII-E) can be
folded into a single, complete family alongside J -- currently C and
I have only been tested against D and D-indep2 (Section VII-D), not
against D-indep3/D-indep4.

WHY THE CALL ORDER MATTERS HERE (read before editing)
--------------------------------------------------------
run_j_vs_multi_indep.py's per-seed call sequence -- J, then D,
D-indep, D-indep2, D-indep3, D-indep4, in that order -- is EMPIRICALLY
VALIDATED: the J-vs-D numbers it produced matched the paper's own
tab:heldout report closely (90.568 vs. 90.570 coverage, etc.). An
earlier script that used a *different*, shorter call sequence (just
J then D, omitting other policies) produced numbers that did NOT
match, for reasons that were never fully isolated to a single root
cause -- see the project's own theta-sensitivity debugging history.
Because the mechanism isn't fully understood, the safest change here
is to APPEND the new C and I calls at the END of the already-working
per-seed sequence (after D-indep4), rather than interleaving them
earlier. This means nothing that happens before D-indep4 in the
sequence changes, so J's and the five baselines' own numbers should
reproduce run_j_vs_multi_indep.py's results unchanged (verified by
Sanity Check 1 below) -- only C and I are new, and their own numbers
are checked against a DIFFERENT, independently-generated ground
truth: circularity_check_with_j.csv's C-vs-D and I-vs-D figures
(Sanity Check 2).

Do not reorder the sequence, and do not run this "per-policy, batched
across all seeds" (all seeds of C first, then all seeds of I, etc.)
-- always seed-by-seed, full sequence, exactly as written below.

USAGE
-----
    python run_c_i_vs_multi_indep.py --seeds 50 --steps 500 \\
        --w-energy 0.283556320123423 --w-coverage 0.023813783478481596 \\
        --w-delay 0.28445380353708727 --w-battery 0.40817609286100814 \\
        --theta 0.5

OUTPUT
------
    c_i_j_vs_multi_indep_heldout.csv -- raw per-seed data, every
    policy (J, C, I, D, D-indep, D-indep2, D-indep3, D-indep4), with
    EVERY scalar field simulate_fleet returns (not just the 6-field
    METRICS tuple), so this file can support a full family matching
    circularity_check_with_j.csv's ~21-metric scope (Section VII-D),
    not just the 5-6 headline metrics. The printed report still only
    walks coverage_pct/avg_relay_delay_steps/avg_relay_energy_pct for
    console readability; a separate family-builder script (not this
    one) should consume the CSV directly for the full-metric BH
    correction. Console output also prints the full list of metric
    fields discovered as common across every policy, to confirm what
    the CSV actually contains before building on it.

    Printed report: mean, paired Wilcoxon p, matched-pairs
    rank-biserial r (RANK-based, not the sign-count approximation
    used by the original run_j_vs_multi_indep.py's report -- see
    _rank_biserial below), and a paired-t 95% CI on the mean
    difference, for coverage/delay/energy, one block per (policy,
    baseline) pair.
"""

import argparse
import csv

import numpy as np
from scipy.stats import wilcoxon, rankdata, t as tdist

import a2e_relay_fixed_final_51 as relay
from policy_j_full import select_best_relay_j, METRICS, NORMALIZED_WEIGHTS_J_DEFAULT
from policy_d_true_fis import (
    select_fuzzy_relay_true_fis_indep_weights,
    select_fuzzy_relay_true_fis_indep2_weights,
    select_fuzzy_relay_true_fis_indep3_weights,
    select_fuzzy_relay_true_fis_indep4_weights,
)

FIS_BASELINES = {
    "D":       lambda needer, drones, g, detected, step: relay.POLICY_SELECTORS["D"](needer, drones, g, detected, step),
    "Dindep":  select_fuzzy_relay_true_fis_indep_weights,
    "Dindep2": select_fuzzy_relay_true_fis_indep2_weights,
    "Dindep3": select_fuzzy_relay_true_fis_indep3_weights,
    "Dindep4": select_fuzzy_relay_true_fis_indep4_weights,
}

# Ground truth for Sanity Check 2 (C, I) -- from circularity_check_with_j.csv,
# already verified elsewhere against tab:heldout.
KNOWN_C_VS_D = {"coverage_pct": 88.79, "avg_relay_delay_steps": 14.22, "avg_relay_energy_pct": 10.45}
KNOWN_I_VS_D = {"coverage_pct": 87.93, "avg_relay_delay_steps": 13.46, "avg_relay_energy_pct": 9.03}
# Ground truth for Sanity Check 1 (J) -- from run_j_vs_multi_indep_heldout.csv / tab:heldout.
KNOWN_J = {"coverage_pct": 90.568, "avg_relay_delay_steps": 13.457, "avg_relay_energy_pct": 9.876}


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--w-energy", type=float, default=None)
    p.add_argument("--w-coverage", type=float, default=None)
    p.add_argument("--w-delay", type=float, default=None)
    p.add_argument("--w-battery", type=float, default=None)
    p.add_argument("--theta", type=float, default=0.5)
    p.add_argument("--seed-start", type=int, default=1)
    p.add_argument("--seeds", type=int, default=50,
                   help="keep this the SAME held-out block used everywhere "
                       "else in the paper (seeds 1-50); do not cherry-pick "
                       "after seeing a result")
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--num-drones", type=int, default=None)
    p.add_argument("--num-reserves", type=int, default=None)
    p.add_argument("--out-csv", default="c_i_j_vs_multi_indep_heldout.csv")
    return p.parse_args()


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


def _rank_biserial(x, y):
    """Matched-pairs rank-biserial correlation, r = (R+ - R-)/(R+ + R-),
    where R+/R- are RANK sums of |paired differences| -- consistent
    with the effect sizes used throughout the rest of the paper. (The
    original run_j_vs_multi_indep.py's _rank_biserial only counted how
    often x>y vs. x<y, ignoring magnitude -- a different, cruder
    statistic that produced numerically different, non-comparable
    values; do not reuse that version.)"""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    d = x - y
    nz = d[d != 0]
    if len(nz) == 0:
        return 0.0
    ranks = rankdata(np.abs(nz))
    r_pos = ranks[nz > 0].sum()
    r_neg = ranks[nz < 0].sum()
    return float((r_pos - r_neg) / (r_pos + r_neg))


def _wtest(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    d = x - y
    return wilcoxon(x, y).pvalue if np.any(d != 0) else 1.0


def _paired_ci95(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    d = x - y
    n = len(d)
    mean = d.mean()
    se = d.std(ddof=1) / np.sqrt(n) if n > 1 else 0.0
    tcrit = tdist.ppf(0.975, n - 1) if n > 1 else 0.0
    return mean, mean - tcrit * se, mean + tcrit * se


def run(seeds, steps, num_drones, num_reserves, j_weights, theta):
    def j_selector(needer, drones, g, detected, step):
        return select_best_relay_j(needer, drones, g, detected, step,
                                   weights=j_weights, theta=theta)

    # Validated prefix (J, then the five FIS baselines) UNCHANGED,
    # C and I appended at the end -- see module docstring.
    ordered_labels = ["J"] + list(FIS_BASELINES.keys()) + ["C", "I"]
    selectors = {
        "J": j_selector,
        **FIS_BASELINES,
        "C": relay.select_best_relay,
        "I": relay.select_best_relay_coverage_aware,
    }

    rows = []
    for seed in seeds:
        row = {"seed": seed}
        for label in ordered_labels:
            m = relay.simulate_fleet(seed, relay_selector=selectors[label],
                                     num_drones=num_drones, num_reserves=num_reserves,
                                     max_steps=steps)
            # Log EVERY field simulate_fleet returns, not just the 6-field
            # METRICS tuple -- circularity_check_with_j.csv (the source of
            # Section VII-D's 126-test family) has ~21 fields per policy,
            # and building a single, consistent family across Section VII-D
            # and this extended check requires the same field set. Skip
            # fields that clearly aren't scalar (nested dicts/lists); a
            # later analysis step can further filter to whichever fields
            # are numeric across every policy.
            for k, v in m.items():
                if isinstance(v, (dict, list, tuple, set)):
                    continue
                row[f"{label}_{k}"] = float(v) if isinstance(v, bool) else v
        rows.append(row)
        print(f"  seed {seed} done")

    sample = rows[0]
    common_metrics = sorted({
        key.split("_", 1)[1] for key in sample if key.startswith("J_")
        if all(f"{label}_{key.split('_', 1)[1]}" in sample for label in ordered_labels)
    })
    print(f"\nDiscovered {len(common_metrics)} metric fields present for every "
          f"policy (candidates for the family builder): {common_metrics}")

    return rows, ordered_labels


def report(rows):
    metrics = ("coverage_pct", "avg_relay_delay_steps", "avg_relay_energy_pct")
    baselines = ["D", "Dindep", "Dindep2", "Dindep3", "Dindep4"]
    policies = ["J", "C", "I"]

    print(f"\n{'='*100}\nC, I, J vs FIVE FIS BASELINES -- {len(rows)} seeds "
          f"(held-out)\n{'='*100}")

    for policy in policies:
        for baseline in baselines:
            print(f"\n--- {policy} vs {baseline} ---")
            print(f"{'metric':<26}{policy:>10}{baseline:>10}{'diff':>10}"
                  f"{'95% CI':>22}{'p':>10}{'r':>8}")
            print("-" * 96)
            for m in metrics:
                x = np.array([r[f"{policy}_{m}"] for r in rows], dtype=float)
                y = np.array([r[f"{baseline}_{m}"] for r in rows], dtype=float)
                pv = _wtest(x, y)
                r_eff = _rank_biserial(x, y)
                mean, lo, hi = _paired_ci95(x, y)
                print(f"{m:<26}{x.mean():>10.3f}{y.mean():>10.3f}{mean:>+10.3f}"
                      f"  [{lo:>+7.3f},{hi:>+7.3f}]{pv:>10.4f}{r_eff:>8.3f}")

    print(f"\n{'='*100}\nSANITY CHECK 1 -- J vs D/Dindep2 should match the "
          f"validated run_j_vs_multi_indep_heldout.csv numbers closely "
          f"(J vs D coverage/delay/energy ≈ 90.568/13.457/9.876). If these "
          f"drift, appending C and I at the end of the sequence still "
          f"perturbed something -- do not trust the C/I numbers below "
          f"until this is resolved.\n{'='*100}")
    for m in metrics:
        j = np.array([r[f"J_{m}"] for r in rows], dtype=float)
        print(f"  {m:<24} this run: {j.mean():.3f}   expected: {KNOWN_J[m]:.3f}   "
              f"diff: {j.mean()-KNOWN_J[m]:+.3f}")

    print(f"\n{'='*100}\nSANITY CHECK 2 -- C vs D and I vs D should match "
          f"circularity_check_with_j.csv's independently-generated figures "
          f"(C: 88.79/14.22/10.45; I: 87.93/13.46/9.03). This is C/I's own "
          f"ground truth, generated by a different script entirely, so "
          f"agreement here is meaningful evidence the new C/I calls are "
          f"behaving correctly.\n{'='*100}")
    for label, known in (("C", KNOWN_C_VS_D), ("I", KNOWN_I_VS_D)):
        for m in metrics:
            x = np.array([r[f"{label}_{m}"] for r in rows], dtype=float)
            print(f"  {label} {m:<22} this run: {x.mean():.3f}   "
                  f"expected: {known[m]:.3f}   diff: {x.mean()-known[m]:+.3f}")

    print(
        "\nNext step once both sanity checks pass: this CSV now carries every "
        "scalar metric field simulate_fleet returns (see the 'Discovered N "
        "metric fields' line printed during the run), matching "
        "circularity_check_with_j.csv's scope. Feed it into a family-builder "
        "script (extend bh_split_quality_vs_cost.py's POLICIES_B from "
        "(D, D-indep2) to (D, D-indep2, D-indep3, D-indep4) -- D-indep stays "
        "dropped as redundant, per its own bit-identical-to-D finding; "
        "D-indep3 is NOT dropped yet, since its possible redundancy is only "
        "a hypothesis pending the decision-level shadow-evaluation check) "
        "to produce a single, consistent, full-metric BH correction covering "
        "C, I, and J against all four independent baselines."
    )


def main():
    args = _parse_args()
    seeds = list(range(args.seed_start, args.seed_start + args.seeds))
    num_drones = args.num_drones if args.num_drones is not None else relay.NUM_DRONES
    num_reserves = args.num_reserves if args.num_reserves is not None else relay.RESERVE_POOL_SIZE
    j_weights = _j_weights(args)

    print(f"J weights: {j_weights or NORMALIZED_WEIGHTS_J_DEFAULT}  theta={args.theta}")
    print(f"Held-out block: seeds {seeds[0]}..{seeds[-1]}, max_steps={args.steps}, "
          f"Nf={num_drones + num_reserves}")
    print("Per-seed call order: J, D, Dindep, Dindep2, Dindep3, Dindep4, C, I "
          "(fixed -- see module docstring; do not reorder)\n")

    rows, labels = run(seeds, args.steps, num_drones, num_reserves, j_weights, args.theta)

    with open(args.out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWritten: {args.out_csv}")

    report(rows)


if __name__ == "__main__":
    main()
