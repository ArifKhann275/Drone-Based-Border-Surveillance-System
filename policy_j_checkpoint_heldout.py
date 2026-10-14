# -*- coding: utf-8 -*-
"""
policy_j_checkpoint_heldout.py
══════════════════════════════════════════════════════════════
Runs the held-out evaluation using a MANUALLY SUPPLIED set of weights --
i.e. whatever the DE run's terminal log last printed as "best" -- instead
of waiting for the full 25-generation run (policy_j_full.py) to finish.

This does NOT touch, kill, or interfere with an already-running
policy_j_full.py process -- it's a separate, independent evaluation.
If you run this WHILE policy_j_full.py is still running in another
terminal, both will compete for CPU and both will slow down; if you
want a clean fast read, pause/stop the DE run first (Ctrl+C in its
terminal is safe -- this script does not depend on its output files).

USAGE (edit the four weight values / theta below to match whatever your
terminal most recently printed as "best_mean_coverage=...weights=..."):

    python policy_j_checkpoint_heldout.py \\
        --w-energy 0.2836 --w-coverage 0.0238 --w-delay 0.2845 --w-battery 0.4082 \\
        --theta 0.5
"""
import argparse

import numpy as np
from scipy.stats import wilcoxon

import a2e_relay_fixed_final_51 as relay
from policy_j_full import select_best_relay_j, METRICS


def _p(a, b):
    a, b = np.array(a, dtype=float), np.array(b, dtype=float)
    return wilcoxon(a, b).pvalue if np.any(a - b != 0) else 1.0


def evaluate_heldout(weights, theta, seeds, max_steps, num_drones, num_reserves):
    def j_selector(needer, drones, g, detected, step):
        return select_best_relay_j(needer, drones, g, detected, step,
                                   weights=weights, theta=theta)

    results = {"J": {m: [] for m in METRICS},
               "H": {m: [] for m in METRICS},
               "I": {m: [] for m in METRICS},
               "C": {m: [] for m in METRICS},
               "D": {m: [] for m in METRICS}}
    for seed in seeds:
        m_j = relay.simulate_fleet(seed, relay_selector=j_selector,
                                   num_drones=num_drones, num_reserves=num_reserves,
                                   max_steps=max_steps)
        m_h = relay.simulate_fleet(seed, relay_selector=relay.select_best_relay_battery_aware_capped,
                                   num_drones=num_drones, num_reserves=num_reserves,
                                   max_steps=max_steps)
        m_i = relay.simulate_fleet(seed, relay_selector=relay.select_best_relay_coverage_aware,
                                   num_drones=num_drones, num_reserves=num_reserves,
                                   max_steps=max_steps)
        m_c = relay.simulate_fleet(seed, relay_selector=relay.select_best_relay,
                                   num_drones=num_drones, num_reserves=num_reserves,
                                   max_steps=max_steps)
        m_d = relay.simulate_fleet(seed, relay_selector=relay.POLICY_SELECTORS["D"],
                                   num_drones=num_drones, num_reserves=num_reserves,
                                   max_steps=max_steps)
        for label, m in (("J", m_j), ("H", m_h), ("I", m_i), ("C", m_c), ("D", m_d)):
            for k in METRICS:
                v = m.get(k)
                if isinstance(v, bool):
                    v = float(v)
                results[label][k].append(v)
    return results


def report(results, seeds, note):
    print(f"\n{'='*112}\nCHECKPOINT HELD-OUT REPORT -- seeds {seeds[0]}..{seeds[-1]} "
          f"({note})\n{'='*112}\n")
    print(f"{'metric':<26}{'J':>9}{'H':>9}{'I':>9}{'C':>9}{'D':>9}"
          f"{'p(J-D)':>9}{'p(J-H)':>9}{'p(J-I)':>9}{'p(J-C)':>9}")
    print("-" * 106)

    for m in METRICS:
        j, h, i, c, d = (results[k][m] for k in ("J", "H", "I", "C", "D"))
        p_jd, p_jh, p_ji, p_jc = _p(j, d), _p(j, h), _p(j, i), _p(j, c)
        print(f"{m:<26}{np.mean(j):>9.3f}{np.mean(h):>9.3f}{np.mean(i):>9.3f}"
              f"{np.mean(c):>9.3f}{np.mean(d):>9.3f}{p_jd:>9.4f}{p_jh:>9.4f}"
              f"{p_ji:>9.4f}{p_jc:>9.4f}")

    print(f"\nNOTE: these weights are a mid-run checkpoint, not the DE run's "
          f"converged output -- treat this as a directional read, not the "
          f"final number to report in the paper.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--w-energy", type=float, required=True)
    p.add_argument("--w-coverage", type=float, required=True)
    p.add_argument("--w-delay", type=float, required=True)
    p.add_argument("--w-battery", type=float, required=True)
    p.add_argument("--theta", type=float, default=0.5)
    p.add_argument("--test-seed-start", type=int, default=1)
    p.add_argument("--test-seeds", type=int, default=50)
    p.add_argument("--test-steps", type=int, default=500)
    p.add_argument("--num-drones", type=int, default=None)
    p.add_argument("--num-reserves", type=int, default=None)
    args = p.parse_args()

    total = args.w_energy + args.w_coverage + args.w_delay + args.w_battery
    weights = {
        "W_TRAVEL_ENERGY": args.w_energy / total,
        "W_COVERAGE_LOSS": args.w_coverage / total,
        "W_RELAY_DELAY": args.w_delay / total,
        "W_BATTERY_MARGIN": args.w_battery / total,
    }
    num_drones = args.num_drones if args.num_drones is not None else relay.NUM_DRONES
    num_reserves = args.num_reserves if args.num_reserves is not None else relay.RESERVE_POOL_SIZE
    test_seeds = list(range(args.test_seed_start, args.test_seed_start + args.test_seeds))

    print(f"Checkpoint weights (renormalized): {weights}")
    print(f"theta: {args.theta}")
    print(f"Test block: seeds {test_seeds[0]}..{test_seeds[-1]}, "
          f"max_steps={args.test_steps}\n")

    results = evaluate_heldout(weights, args.theta, test_seeds, args.test_steps,
                               num_drones, num_reserves)
    report(results, test_seeds, note="MID-RUN CHECKPOINT weights, DE not yet converged")


if __name__ == "__main__":
    main()
