# -*- coding: utf-8 -*-
"""
weight_tuning_heldout.py  --  train/test split for Policy I's weight vector
with Multi-Objective Fitness (Coverage, Delay, Energy).

WHAT THIS DOES
-----------------------------------------------------------
This script tunes Policy I (absolute-scale coverage loss fix) using 
a multi-objective cost function:
Cost = alpha*(100 - coverage_pct)/100 + beta*(delay/30.0) + gamma*(energy/20.0)
where alpha=0.5, beta=0.25, gamma=0.25.

It ensures that we get an "apples-to-apples" optimized version of Policy I 
to fairly compare against Policy J.
"""

import argparse
import csv
import json
import time

import numpy as np
from scipy.optimize import differential_evolution
from scipy.stats import wilcoxon

import a2e_relay_fixed_final_51 as relay

METRICS = ("coverage_pct", "avg_zone_staleness_steps",
           "avg_relay_delay_steps", "avg_relay_energy_pct",
           "total_energy_consumed", "mission_completed")


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tune-seed-start", type=int, default=101)
    p.add_argument("--tune-seeds", type=int, default=50)
    p.add_argument("--tune-steps", type=int, default=500)
    p.add_argument("--test-seed-start", type=int, default=1)
    p.add_argument("--test-seeds", type=int, default=50)
    p.add_argument("--test-steps", type=int, default=500)
    p.add_argument("--num-drones", type=int, default=None)
    p.add_argument("--num-reserves", type=int, default=None)
    p.add_argument("--popsize", type=int, default=12)
    p.add_argument("--maxiter", type=int, default=25)
    p.add_argument("--seed-rng", type=int, default=0)
    p.add_argument("--out-json", default="tuned_weights_policy_i.json")
    p.add_argument("--out-history", default="weight_tuning_history_policy_i.csv")
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════
# Weight parameterization (2-simplex mapping for 3 parameters)
# ══════════════════════════════════════════════════════════════════════════
def params_to_weights(theta):
    a, b = theta
    a = min(max(a, 0.0), 1.0)
    b = min(max(b, 0.0), 1.0)
    w_energy = a
    w_coverage = (1.0 - a) * b
    w_delay = (1.0 - a) * (1.0 - b)
    return {
        "W_TRAVEL_ENERGY": float(w_energy),
        "W_COVERAGE_LOSS": float(w_coverage),
        "W_RELAY_DELAY": float(w_delay),
    }


def weights_to_tuple(w):
    return (round(w["W_TRAVEL_ENERGY"], 4), round(w["W_COVERAGE_LOSS"], 4),
            round(w["W_RELAY_DELAY"], 4))


# ══════════════════════════════════════════════════════════════════════════
# Multi-Objective Function for Policy I
# ══════════════════════════════════════════════════════════════════════════
class Objective:
    def __init__(self, seeds, max_steps, num_drones, num_reserves, history_path):
        self.seeds = seeds
        self.max_steps = max_steps
        self.num_drones = num_drones
        self.num_reserves = num_reserves
        self.calls = 0
        self.t0 = time.perf_counter()
        self.history_path = history_path
        self._hist_rows = []
        # best stores: (cost, weights_tuple, coverage, delay, energy)
        self._best = (float("inf"), None, 0.0, 0.0, 0.0)
        self._popsize_hint = 1

    def __call__(self, theta):
        w = params_to_weights(theta)

        def selector(needer, drones, g, detected, step):
            # NEW: Using Policy I (absolute coverage fix) instead of Base C
            return relay.select_best_relay_coverage_aware(needer, drones, g, detected, step,
                                           weights=w)

        covs, delays, energies = [], [], []
        for seed in self.seeds:
            m = relay.simulate_fleet(seed, relay_selector=selector,
                                     num_drones=self.num_drones,
                                     num_reserves=self.num_reserves,
                                     max_steps=self.max_steps)
            covs.append(m["coverage_pct"])
            delays.append(m["avg_relay_delay_steps"])
            energies.append(m["avg_relay_energy_pct"])

        mean_cov = float(np.mean(covs))
        mean_delay = float(np.mean(delays))
        mean_energy = float(np.mean(energies))

        # MULTI-OBJECTIVE CALCULATIONS
        alpha, beta, gamma = 0.5, 0.25, 0.25
        
        norm_coverage_loss = (100.0 - mean_cov) / 100.0
        norm_delay = mean_delay / 30.0    # Approx max delay 30 steps
        norm_energy = mean_energy / 20.0  # Approx max energy 20%
        
        # We minimize this cost
        cost = (alpha * norm_coverage_loss) + (beta * norm_energy) + (gamma * norm_delay)
        
        self.calls += 1

        if cost < self._best[0]:
            self._best = (cost, weights_to_tuple(w), mean_cov, mean_delay, mean_energy)

        self._hist_rows.append({
            "call": self.calls, "w_energy": w["W_TRAVEL_ENERGY"],
            "w_coverage": w["W_COVERAGE_LOSS"], "w_delay": w["W_RELAY_DELAY"],
            "mean_coverage_pct": mean_cov, 
            "mean_delay": mean_delay, 
            "mean_energy": mean_energy,
            "cost": cost,
            "elapsed_s": round(time.perf_counter() - self.t0, 1),
        })
        return cost

    def flush_history(self):
        if not self._hist_rows:
            return
        with open(self.history_path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(self._hist_rows[0].keys()))
            w.writeheader()
            w.writerows(self._hist_rows)


def de_callback(objective):
    def _cb(xk, convergence):
        gen = objective.calls // max(objective._popsize_hint, 1)
        best_cost = objective._best[0]
        best_cov = objective._best[2]
        best_delay = objective._best[3]
        best_energy = objective._best[4]
        print(f"  [gen ~{gen:>3}] calls={objective.calls:<5} "
              f"best_cost={best_cost:6.4f} (cov={best_cov:5.2f}%, dly={best_delay:5.2f}, nrg={best_energy:5.2f}) "
              f"weights(E,C,D)={objective._best[1]}  convergence={convergence:.4f}")
    return _cb


# ══════════════════════════════════════════════════════════════════════════
# Held-out evaluation
# ══════════════════════════════════════════════════════════════════════════
def evaluate_heldout(weights, seeds, max_steps, num_drones, num_reserves):
    def tuned_selector(needer, drones, g, detected, step):
        return relay.select_best_relay_coverage_aware(needer, drones, g, detected, step,
                                       weights=weights)

    results = {"I2": {m: [] for m in METRICS},  # I2 = Tuned Policy I
               "I": {m: [] for m in METRICS},   # I = Untuned Policy I
               "D": {m: [] for m in METRICS}}
    for seed in seeds:
        m_i2 = relay.simulate_fleet(seed, relay_selector=tuned_selector,
                                    num_drones=num_drones, num_reserves=num_reserves,
                                    max_steps=max_steps)
        m_i = relay.simulate_fleet(seed, relay_selector=relay.select_best_relay_coverage_aware,
                                   num_drones=num_drones, num_reserves=num_reserves,
                                   max_steps=max_steps)
        m_d = relay.simulate_fleet(seed, relay_selector=relay.POLICY_SELECTORS["D"],
                                   num_drones=num_drones, num_reserves=num_reserves,
                                   max_steps=max_steps)
        for label, m in (("I2", m_i2), ("I", m_i), ("D", m_d)):
            for k in METRICS:
                v = m.get(k)
                if isinstance(v, bool):
                    v = float(v)
                results[label][k].append(v)
    return results


def report(results, seeds):
    print(f"\n{'='*78}\nHELD-OUT REPORT -- seeds {seeds[0]}..{seeds[-1]} "
          f"(never touched by the optimizer)\n{'='*78}\n")
    print(f"{'metric':<28}{'I2 (tuned)':>12}{'I (base)':>12}{'D':>12}"
          f"{'p(I2 vs D)':>12}{'p(I2 vs I)':>12}")
    print("-" * 92)
    verdict_bad = []
    for m in METRICS:
        i2 = np.array(results["I2"][m], dtype=float)
        i = np.array(results["I"][m], dtype=float)
        d = np.array(results["D"][m], dtype=float)
        p_d = wilcoxon(i2, d).pvalue if np.any(i2 - d != 0) else 1.0
        p_i = wilcoxon(i2, i).pvalue if np.any(i2 - i != 0) else 1.0
        print(f"{m:<28}{i2.mean():>12.3f}{i.mean():>12.3f}{d.mean():>12.3f}"
              f"{p_d:>12.4f}{p_i:>12.4f}")

        lower_is_better = m in ("avg_zone_staleness_steps", "avg_relay_delay_steps",
                                "avg_relay_energy_pct", "total_energy_consumed")
        if m in ("avg_relay_delay_steps", "avg_relay_energy_pct") and p_i < 0.05:
            worse = (i2.mean() > i.mean()) if lower_is_better else (i2.mean() < i.mean())
            if worse:
                verdict_bad.append(m)

    print()
    if verdict_bad:
        print(f"HONEST FLAG: the tuned weights significantly REGRESS {verdict_bad} "
              f"relative to base Policy I on the held-out set.")
    else:
        print("No significant delay/energy regression vs base Policy I "
              "detected on the held-out set.")

    cov_i2 = np.array(results["I2"]["coverage_pct"], dtype=float)
    cov_d = np.array(results["D"]["coverage_pct"], dtype=float)
    p = wilcoxon(cov_i2, cov_d).pvalue if np.any(cov_i2 - cov_d != 0) else 1.0
    if p >= 0.05:
        print(f"\nCoverage gap vs D on the held-out set: NOT significant (p={p:.4f})")
    else:
        print(f"\nCoverage gap vs D on the held-out set: STILL significant (p={p:.4f}).")


def main():
    args = _parse_args()

    num_drones = args.num_drones if args.num_drones is not None else relay.NUM_DRONES
    num_reserves = args.num_reserves if args.num_reserves is not None else relay.RESERVE_POOL_SIZE

    tune_seeds = list(range(args.tune_seed_start, args.tune_seed_start + args.tune_seeds))
    test_seeds = list(range(args.test_seed_start, args.test_seed_start + args.test_seeds))

    overlap = set(tune_seeds) & set(test_seeds)
    if overlap:
        raise SystemExit(f"Tuning and test seed blocks overlap.")

    print("=" * 78)
    print("HELD-OUT WEIGHT TUNING (Policy I with Multi-Objective Fitness)")
    print("=" * 78)
    print(f"  Tuning block : seeds {tune_seeds[0]}..{tune_seeds[-1]}")
    print(f"  Test block   : seeds {test_seeds[0]}..{test_seeds[-1]}")
    print(f"  Fleet        : Nf={num_drones + num_reserves}")
    print(f"  Objective    : MINIMIZE Multi-Objective Cost (0.5*CovLoss + 0.25*Energy + 0.25*Delay)")
    print(f"  Optimizer    : scipy differential_evolution, popsize={args.popsize}, maxiter={args.maxiter}")

    objective = Objective(tune_seeds, args.tune_steps, num_drones, num_reserves, args.out_history)
    objective._popsize_hint = args.popsize * 2

    t0 = time.perf_counter()
    res = differential_evolution(
        objective, bounds=[(0.0, 1.0), (0.0, 1.0)],
        popsize=args.popsize, maxiter=args.maxiter,
        seed=args.seed_rng, polish=True, tol=1e-3,
        callback=de_callback(objective), disp=False, workers=1,
    )
    objective.flush_history()
    elapsed = time.perf_counter() - t0

    tuned_weights = params_to_weights(res.x)
    print(f"\nOptimization done in {elapsed:.1f}s, {objective.calls} evaluations.")
    print(f"Tuned weights: {tuned_weights}")
    print(f"Tuning-set COST at these weights: {res.fun:.4f}")
    print(f"Tuning-set metrics: Cov={objective._best[2]:.2f}%, Delay={objective._best[3]:.2f}, Energy={objective._best[4]:.2f}%")

    base = relay.NORMALIZED_WEIGHTS
    print(f"\nCompare to base weights: {base}")

    with open(args.out_json, "w") as fh:
        json.dump({
            "policy": "Policy I (Multi-Objective Tuned)",
            "tuned_weights": tuned_weights,
            "base_weights": base,
            "tune_seeds": [tune_seeds[0], tune_seeds[-1]],
            "test_seeds": [test_seeds[0], test_seeds[-1]],
            "tuning_cost": res.fun,
            "tuning_metrics": {
                "coverage": objective._best[2],
                "delay": objective._best[3],
                "energy": objective._best[4]
            }
        }, fh, indent=2)

    print("\nEvaluating tuned weights on the HELD-OUT test block...")
    results = evaluate_heldout(tuned_weights, test_seeds, args.test_steps, num_drones, num_reserves)
    report(results, test_seeds)


if __name__ == "__main__":
    main()