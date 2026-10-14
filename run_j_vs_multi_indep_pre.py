# -*- coding: utf-8 -*-
"""
run_j_vs_multi_indep.py  --  the multi-independent-baseline circularity
test: Policy J against D (the co-designed/"expert" FIS) AND four
independently-authored FIS rule bases (D-indep, D-indep2, D-indep3,
D-indep4), all on the SAME held-out test block.

WHY THIS TEST, AND WHAT WOULD MAKE IT CONVINCING
--------------------------------------------------
A single independent baseline (D-indep) inviting the objection "the
proposed method doesn't match an independently designed FIS" is weak
evidence either way -- one data point. This script runs J against FOUR
independently-authored, individually-verified-non-compensatory rule
bases (indep/indep2/indep3/indep4; see apply_indep3_indep4_patch.py for
how indep3/indep4 were chosen -- BEFORE any Policy J result was
inspected). The strongest possible outcome is

    J ~ D-indep  ~  D-indep2  ~  D-indep3  ~  D-indep4

(J is not significantly different from any of them, or the direction/
magnitude of any difference is small and inconsistent across the four)
-- that would show J's behavior is not an artifact of matching one
particular FIS rule base, because it also (fails to) match FOUR
differently-authored ones in a similar way. The weakest outcome is
J differing from the CO-DESIGNED FIS (D) specifically while tracking
every independent one closely, which would suggest D was tuned to be
beatable. Whatever the actual pattern is, report it as found -- do not
select which independent baselines to keep after seeing this table.

USAGE
-----
    # requires: apply_indep3_indep4_patch.py already run (adds indep3/indep4)
    python run_j_vs_multi_indep.py --seeds 50 --steps 500

    # supply the DE-converged weights/theta from policy_j_full.py's output
    # (tuned_weights_j_full.json) instead of Policy J's built-in default:
    python run_j_vs_multi_indep.py --w-energy 0.28 --w-coverage 0.24 \\
        --w-delay 0.28 --w-battery 0.20 --theta 0.5

Output: j_vs_multi_indep_heldout.csv (raw per-seed, every policy) and a
printed report: mean, paired Wilcoxon p vs J, and matched-pairs
rank-biserial effect size, for coverage/delay/energy, one block per
FIS baseline.
"""

import argparse
import csv

import numpy as np
from scipy.stats import wilcoxon

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


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--w-energy", type=float, default=None)
    p.add_argument("--w-coverage", type=float, default=None)
    p.add_argument("--w-delay", type=float, default=None)
    p.add_argument("--w-battery", type=float, default=None)
    p.add_argument("--theta", type=float, default=0.5)
    p.add_argument("--seed-start", type=int, default=1)
    p.add_argument("--seeds", type=int, default=50,
                   help="the paper's own held-out test block is seeds 1-50; "
                       "keep this the SAME block used everywhere else, do not "
                       "cherry-pick a seed range after seeing a result")
    p.add_argument("--steps", type=int, default=500,
                   help="paper's Table III mission length, NOT a20.MAX_STEPS")
    p.add_argument("--num-drones", type=int, default=None)
    p.add_argument("--num-reserves", type=int, default=None)
    p.add_argument("--out-csv", default="j_vs_multi_indep_heldout.csv")
    return p.parse_args()


def _j_weights(args):
    if args.w_energy is None:
        return None   # select_best_relay_j falls back to NORMALIZED_WEIGHTS_J_DEFAULT
    total = args.w_energy + args.w_coverage + args.w_delay + args.w_battery
    return {
        "W_TRAVEL_ENERGY": args.w_energy / total,
        "W_COVERAGE_LOSS": args.w_coverage / total,
        "W_RELAY_DELAY": args.w_delay / total,
        "W_BATTERY_MARGIN": args.w_battery / total,
    }


def _rank_biserial(x, y):
    d = x - y
    d = d[d != 0]
    if len(d) == 0:
        return 0.0
    return float((np.sum(d > 0) - np.sum(d < 0)) / len(d))


def _wtest(x, y):
    d = x - y
    return wilcoxon(x, y).pvalue if np.any(d != 0) else 1.0


def run(seeds, steps, num_drones, num_reserves, j_weights, theta):
    def j_selector(needer, drones, g, detected, step):
        return select_best_relay_j(needer, drones, g, detected, step,
                                   weights=j_weights, theta=theta)

    labels = ["J"] + list(FIS_BASELINES.keys())
    selectors = {"J": j_selector, **FIS_BASELINES}

    rows = []
    for seed in seeds:
        row = {"seed": seed}
        for label in labels:
            m = relay.simulate_fleet(seed, relay_selector=selectors[label],
                                     num_drones=num_drones, num_reserves=num_reserves,
                                     max_steps=steps)
            for k in METRICS:
                v = m.get(k)
                row[f"{label}_{k}"] = float(v) if isinstance(v, bool) else v
        rows.append(row)
        print(f"  seed {seed} done")
    return rows, labels


def report(rows, labels):
    metrics = ("coverage_pct", "avg_relay_delay_steps", "avg_relay_energy_pct")
    print(f"\n{'='*100}\nJ vs FIVE FIS BASELINES -- {len(rows)} seeds "
          f"(held-out, never used to choose J's weights or any indep profile)\n{'='*100}")

    for baseline in [l for l in labels if l != "J"]:
        print(f"\n--- J vs {baseline} ---")
        print(f"{'metric':<26}{'J':>10}{baseline:>10}{'diff%':>9}{'p':>10}{'r':>8}")
        print("-" * 73)
        for m in metrics:
            j = np.array([r[f"J_{m}"] for r in rows], dtype=float)
            x = np.array([r[f"{baseline}_{m}"] for r in rows], dtype=float)
            pv = _wtest(j, x)
            r_eff = _rank_biserial(j, x)
            pct = 100 * (j.mean() - x.mean()) / x.mean() if x.mean() else float("nan")
            print(f"{m:<26}{j.mean():>10.3f}{x.mean():>10.3f}{pct:>+8.2f}%{pv:>10.4f}{r_eff:>8.3f}")

    print("\n" + "=" * 100)
    print("READ THIS TABLE ACROSS ROWS, NOT DOWN ONE COLUMN:")
    print("  If J's p-value and effect size against Dindep/Dindep2/Dindep3/Dindep4")
    print("  are all small and inconsistent in sign, J's behavior does not depend")
    print("  on matching any ONE particular FIS rule base -- report exactly that.")
    print("  If J tracks D-indep2..4 closely but differs sharply from D (the")
    print("  co-designed FIS), say so plainly rather than emphasizing the")
    print("  independent-baseline agreement alone.")


def main():
    args = _parse_args()
    seeds = list(range(args.seed_start, args.seed_start + args.seeds))
    num_drones = args.num_drones if args.num_drones is not None else relay.NUM_DRONES
    num_reserves = args.num_reserves if args.num_reserves is not None else relay.RESERVE_POOL_SIZE
    j_weights = _j_weights(args)

    print(f"J weights: {j_weights or NORMALIZED_WEIGHTS_J_DEFAULT}  theta={args.theta}")
    print(f"Held-out block: seeds {seeds[0]}..{seeds[-1]}, max_steps={args.steps}, "
          f"Nf={num_drones + num_reserves}\n")

    rows, labels = run(seeds, args.steps, num_drones, num_reserves, j_weights, args.theta)

    with open(args.out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWritten: {args.out_csv}")

    report(rows, labels)


if __name__ == "__main__":
    main()