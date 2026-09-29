# -*- coding: utf-8 -*-
"""
run_heldout_only.py
Skips differential_evolution (already done, frozen in tuned_weights.json)
and runs ONLY weight_tuning_heldout.py's evaluate_heldout() + report()
on the held-out test seeds. Needs a2e_relay_fixed_final_51.py,
weight_tuning_heldout.py, and tuned_weights.json in the same folder.

Usage:
    python run_heldout_only.py
    python run_heldout_only.py --test-steps 500 --test-seed-start 1 --test-seeds 50
"""
import argparse
import json

import weight_tuning_heldout as wth


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights-json", default="tuned_weights.json")
    p.add_argument("--test-seed-start", type=int, default=None)
    p.add_argument("--test-seeds", type=int, default=None)
    p.add_argument("--test-steps", type=int, default=None)
    args = p.parse_args()

    with open(args.weights_json) as fh:
        payload = json.load(fh)

    tuned_weights = payload["tuned_weights"]
    fleet = payload.get("fleet", {})
    num_drones = fleet.get("num_drones")
    num_reserves = fleet.get("num_reserves")

    test_start = args.test_seed_start or payload["test_seeds"][0]
    test_end = args.test_seed_start is None and payload["test_seeds"][1] or None
    n_test = args.test_seeds or (payload["test_seeds"][1] - payload["test_seeds"][0] + 1)
    test_steps = args.test_steps or payload.get("tune_steps", 500)

    test_seeds = list(range(test_start, test_start + n_test))

    print("=" * 78)
    print("HELD-OUT EVALUATION ONLY (optimizer skipped -- using frozen weights)")
    print("=" * 78)
    print(f"  Weights      : {tuned_weights}")
    print(f"  Test block   : seeds {test_seeds[0]}..{test_seeds[-1]} "
          f"({len(test_seeds)} seeds), max_steps={test_steps}")
    print(f"  Fleet        : num_drones={num_drones}, num_reserves={num_reserves}\n")

    results = wth.evaluate_heldout(tuned_weights, test_seeds, test_steps,
                                    num_drones, num_reserves)
    wth.report(results, test_seeds)


if __name__ == "__main__":
    main()
