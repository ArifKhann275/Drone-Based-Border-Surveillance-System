# -*- coding: utf-8 -*-
"""
compare_weight_stability.py
══════════════════════════════════════════════════════════════
Loads two or more tuned_weights_j*.json files (from separate
policy_j_full.py runs -- different --seed-rng, possibly different
--popsize/--maxiter budgets) and reports how much the discovered
weights actually move around. This is the evidence a Q2 reviewer wants
for "did you check the optimizer didn't just get lucky once."

Note: comparing a full-budget run against smaller quick-check runs is
fine and standard practice -- the question isn't "are these bit-for-bit
identical" (they won't be), it's "do they land in the same REGION of
weight-space" (e.g. battery_margin always dominant, coverage_loss
always small). Report which one is true.

USAGE
-----
    python compare_weight_stability.py tuned_weights_j_full.json \\
        tuned_weights_j_seed1.json tuned_weights_j_seed2.json
"""
import argparse
import json

import numpy as np

KEYS = ("W_TRAVEL_ENERGY", "W_COVERAGE_LOSS", "W_RELAY_DELAY", "W_BATTERY_MARGIN")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("json_paths", nargs="+", help="two or more tuned_weights_j*.json files")
    args = p.parse_args()

    if len(args.json_paths) < 2:
        raise SystemExit("Need at least 2 files to compare stability.")

    runs = []
    for path in args.json_paths:
        with open(path) as fh:
            payload = json.load(fh)
        runs.append({
            "path": path,
            "weights": payload["tuned_weights"],
            "coverage": payload.get("tuning_objective_value_mean_coverage_pct"),
            "seed_rng": payload.get("optimizer", {}).get("seed"),
            "evaluations": payload.get("optimizer", {}).get("evaluations"),
            "elapsed_s": payload.get("optimizer", {}).get("elapsed_s"),
        })

    print("=" * 100)
    print("POLICY J -- WEIGHT STABILITY ACROSS DE RUNS")
    print("=" * 100)
    header = f"{'run':<28}{'seed':>6}{'evals':>8}{'elapsed_h':>11}{'tune_cov%':>11}"
    for k in KEYS:
        header += f"{k.replace('W_', ''):>16}"
    print(header)
    print("-" * len(header))

    matrix = []
    for r in runs:
        row_vals = [r["weights"][k] for k in KEYS]
        matrix.append(row_vals)
        line = (f"{r['path']:<28}{str(r['seed_rng']):>6}{str(r['evaluations']):>8}"
                f"{(r['elapsed_s'] or 0)/3600:>11.2f}{(r['coverage'] or 0):>11.3f}")
        for v in row_vals:
            line += f"{v:>16.4f}"
        print(line)

    matrix = np.array(matrix)
    print("-" * len(header))
    std_line = f"{'std across runs':<28}{'':>6}{'':>8}{'':>11}{'':>11}"
    for j in range(len(KEYS)):
        std_line += f"{matrix[:, j].std():>16.4f}"
    print(std_line)
    mean_line = f"{'mean across runs':<28}{'':>6}{'':>8}{'':>11}{'':>11}"
    for j in range(len(KEYS)):
        mean_line += f"{matrix[:, j].mean():>16.4f}"
    print(mean_line)

    print(f"\n{'='*100}")
    ranking_consistent = True
    orders = [tuple(sorted(range(len(KEYS)), key=lambda j: -row[j])) for row in matrix]
    if len(set(orders)) > 1:
        ranking_consistent = False

    if ranking_consistent:
        order_names = [KEYS[i].replace("W_", "") for i in orders[0]]
        print(f"STABLE: every run ranks the four attributes in the SAME order "
              f"of importance: {' > '.join(order_names)}.")
        print("This is the claim worth making in the paper: 'while exact weight "
              "values varied slightly across DE runs with different RNG seeds, "
              "the relative priority ordering (battery_margin and relay_delay "
              "dominant, coverage_loss consistently minor) was stable,' rather "
              "than claiming the exact tuned values are unique/optimal.")
    else:
        print("NOT STABLE: different runs disagree on which attribute matters "
              "most. Report this honestly -- it means the tuning objective's "
              "landscape is flat/multi-modal in this region, and the specific "
              "tuned_weights.json values should be presented as 'a' solution "
              "found, not 'the' solution, with the coverage-parity / delay-energy "
              "results framed as holding across this whole flat region rather "
              "than depending on one exact weight vector.")

    max_std = matrix.std(axis=0).max()
    print(f"\nMax per-attribute std across runs: {max_std:.4f} "
          f"({'small relative to weight scale [0,1] -- reasonably stable' if max_std < 0.08 else 'large -- treat exact weights with caution, lean on the ranking-order claim instead'})")


if __name__ == "__main__":
    main()
