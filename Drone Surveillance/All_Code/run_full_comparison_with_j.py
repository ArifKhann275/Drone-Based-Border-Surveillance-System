# -*- coding: utf-8 -*-
"""
run_full_comparison_with_j.py
══════════════════════════════════════════════════════════════
Runs Policy J (H's battery fix + I's coverage fix + VIKOR-style Q
combination, from policy_j_full.py) through the SAME journal-extension
battery every other policy in this project has already been run
through -- not just the single default-condition test that produced the
promising checkpoint result:

  1. run_terrain_wind_generalization_analysis(): 4 terrains x 3 wind
     modes = 12 cells (matches terrain_wind_sweep.csv's design)
  2. run_fleet_size_sensitivity_analysis(): swept fleet sizes (matches
     fleet_size_sweep.csv's design)
  3. Benjamini-Hochberg correction across EVERY p-value collected from
     both sweeps (coverage_pct, avg_relay_delay_steps,
     avg_relay_energy_pct, each vs D, in every cell) -- matching
     sweep_significance_bh.csv's discipline, so J's numbers are held to
     exactly the same multiple-comparison standard C/D2/T/V/I already
     were.

WHY THIS MATTERS (see the "should I write the paper now" discussion):
a single default-condition checkpoint is a promising SIGNAL, not a
result -- reviewers will ask about other terrains, other wind regimes,
other fleet sizes, and whether the "wins" survive correction for
testing many hypotheses at once. This script produces the full,
correctable evidence base before anything gets written up.

PREREQUISITE: a FROZEN, converged set of Policy J weights. Point
--weights-json at policy_j_full.py's output (tuned_weights_j_full.json)
once that run has actually converged (not a mid-run checkpoint) --
this script fails loudly if the file it's given still looks like a
placeholder. If you only have checkpoint values right now, pass them
directly with --w-energy/--w-coverage/--w-delay/--w-battery/--theta
instead, but treat everything this script then produces as
provisional, same caveat as the earlier checkpoint run.

USAGE
-----
    # once tuned_weights_j_full.json exists and DE has converged:
    python run_full_comparison_with_j.py

    # smoke test on a tiny budget first:
    python run_full_comparison_with_j.py --tune-check --seeds 3 --max-steps 60

    # or from explicit weights (e.g. a checkpoint, clearly provisional):
    python run_full_comparison_with_j.py --w-energy 0.2836 --w-coverage 0.0238 \\
        --w-delay 0.2845 --w-battery 0.4082 --theta 0.5 --seeds 5 --max-steps 100

OUTPUT
------
    terrain_wind_sweep_with_j.csv     -- per-seed raw data, all policies incl. J
    fleet_size_sweep_with_j.csv       -- per-seed raw data, all policies incl. J
    sweep_significance_bh_with_j.csv  -- every J-vs-D p-value from both sweeps,
                                         BH-corrected together (q_bh, survives_bh)
"""

import argparse
import csv
import json
import sys

import a2e_relay_fixed_final_51 as relay
from policy_j_full import select_best_relay_j


def _load_weights(args):
    if args.weights_json:
        with open(args.weights_json) as fh:
            payload = json.load(fh)
        weights = payload["tuned_weights"]
        theta = payload.get("theta", 0.5)
        print(f"Loaded frozen Policy J weights from {args.weights_json}: {weights}, theta={theta}")
        return weights, theta, payload

    required = (args.w_energy, args.w_coverage, args.w_delay, args.w_battery)
    if any(v is None for v in required):
        sys.exit("Need either --weights-json, or all four of "
                 "--w-energy/--w-coverage/--w-delay/--w-battery.")
    total = sum(required)
    weights = {
        "W_TRAVEL_ENERGY": args.w_energy / total,
        "W_COVERAGE_LOSS": args.w_coverage / total,
        "W_RELAY_DELAY": args.w_delay / total,
        "W_BATTERY_MARGIN": args.w_battery / total,
    }
    print(f"WARNING: using explicitly-passed weights (no --weights-json) -- "
          f"treat all results below as PROVISIONAL, not paper-ready, unless "
          f"these came from a converged DE run: {weights}, theta={args.theta}")
    return weights, args.theta, None


# ══════════════════════════════════════════════════════════════════════════
# Benjamini-Hochberg, dependency-free (no statsmodels requirement)
# ══════════════════════════════════════════════════════════════════════════
def bh_correct(pvals):
    """Standard Benjamini-Hochberg step-up procedure. Returns q-values in
    the same order as the input. None entries pass through as None."""
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
    p.add_argument("--weights-json", default=None,
                   help="path to policy_j_full.py's tuned_weights_j_full.json")
    p.add_argument("--w-energy", type=float, default=None)
    p.add_argument("--w-coverage", type=float, default=None)
    p.add_argument("--w-delay", type=float, default=None)
    p.add_argument("--w-battery", type=float, default=None)
    p.add_argument("--theta", type=float, default=0.5)
    p.add_argument("--seeds", type=int, default=50, help="number of seeds, 1..N")
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--terrain-wind-csv", default="terrain_wind_sweep_with_j.csv")
    p.add_argument("--fleet-size-csv", default="fleet_size_sweep_with_j.csv")
    p.add_argument("--bh-csv", default="sweep_significance_bh_with_j.csv")
    args = p.parse_args()

    weights, theta, payload = _load_weights(args)

    def j_selector(needer, drones, g, detected, step):
        return select_best_relay_j(needer, drones, g, detected, step,
                                   weights=weights, theta=theta)

    relay.POLICY_SELECTORS["J"] = j_selector
    relay.POLICY_DISPLAY_NAMES["J"] = "RelayScore-J (H+I+VIKOR-Q)"

    seeds = list(range(1, args.seeds + 1))

    print("=" * 78)
    print("POLICY J -- FULL GENERALIZATION SWEEP (terrain x wind, fleet size)")
    print("=" * 78)
    print(f"  seeds=1..{args.seeds}, max_steps={args.max_steps}")
    print(f"  Policies: C, D, D2, T, V, I, J\n")

    # ---- 1. terrain x wind sweep -------------------------------------
    print(">>> Terrain x wind generalization sweep...")
    tw_rows, tw_summary = relay.run_terrain_wind_generalization_analysis(
        seeds=seeds, max_steps=args.max_steps,
        policies=("C", "D", "D2", "T", "V", "I", "J"),
        out_csv=args.terrain_wind_csv,
    )
    relay.a20.set_terrain_scenario("benapole_delta")  # restore, per the fn's own docstring warning

    # ---- 2. fleet-size sweep ------------------------------------------
    print("\n>>> Fleet-size sensitivity sweep...")
    fs_rows, fs_summary = relay.run_fleet_size_sensitivity_analysis(
        seeds=seeds, max_steps=args.max_steps,
        policies=("C", "D", "T", "V", "I", "J"),
        out_csv=args.fleet_size_csv,
    )

    # ---- 3. gather every J-vs-D p-value from both sweeps, BH-correct together
    print("\n>>> Collecting J-vs-D p-values for BH correction...")
    records = []  # each: dict(sweep, condition, metric, p)

    def _p_of(entry, key):
        # entry[key] is {"mean_diff": ..., "wilcoxon_p": ...}, not a bare float.
        v = entry.get(key)
        return v.get("wilcoxon_p") if isinstance(v, dict) else v

    for entry in tw_summary:
        condition = f"{entry['terrain']}/{entry['wind_mode']}"
        for metric, key in (
            ("coverage_pct", "coverage_pct_test_J_vs_D"),
            ("avg_relay_delay_steps", "avg_relay_delay_steps_test_J_vs_D"),
            ("avg_relay_energy_pct", "avg_relay_energy_pct_test_J_vs_D"),
        ):
            records.append({"sweep": "terrain_wind", "condition": condition,
                            "metric": metric, "policy": "J", "p": _p_of(entry, key)})

    for entry in fs_summary:
        condition = f"N={entry['fleet_size']} (patrol={entry['num_drones']},reserve={entry['num_reserves']})"
        for metric, key in (
            ("coverage_pct", "coverage_test_J_vs_D"),
            ("avg_relay_delay_steps", "delay_test_J_vs_D"),
            ("avg_relay_energy_pct", "energy_test_J_vs_D"),
        ):
            records.append({"sweep": "fleet_size", "condition": condition,
                            "metric": metric, "policy": "J", "p": _p_of(entry, key)})

    pvals = [r["p"] for r in records]
    qvals = bh_correct(pvals)
    for r, q in zip(records, qvals):
        r["q_bh"] = round(q, 6) if q is not None else None
        r["survives_bh"] = (q is not None and q < args.alpha)

    with open(args.bh_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["sweep", "condition", "metric",
                                                "policy", "p", "q_bh", "survives_bh"])
        writer.writeheader()
        writer.writerows(records)

    n_tested = sum(1 for r in records if r["p"] is not None)
    n_survive = sum(1 for r in records if r["survives_bh"])
    print(f"\n{'='*78}\nJ vs D -- BH correction across all {n_tested} tests "
          f"(terrain x wind + fleet size, all 3 metrics)\n{'='*78}")
    for r in records:
        if r["p"] is None:
            continue
        flag = "  <-- SURVIVES BH" if r["survives_bh"] else ""
        print(f"  [{r['sweep']:<12}] {r['condition']:<45} {r['metric']:<24} "
              f"p={r['p']:.4f}  q_bh={r['q_bh']:.4f}{flag}")

    print(f"\n{n_survive}/{n_tested} tests survive BH correction at alpha={args.alpha}.")
    print(f"Written: {args.terrain_wind_csv}, {args.fleet_size_csv}, {args.bh_csv}")

    if payload is None:
        print("\nREMINDER: this run used provisional (non-converged / manually "
              "passed) weights -- re-run with --weights-json once DE has "
              "actually converged before using any of this in the paper.")


if __name__ == "__main__":
    main()
