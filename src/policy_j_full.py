# -*- coding: utf-8 -*-
"""
policy_j_full.py
══════════════════════════════════════════════════════════════
POLICY J (full version) -- tuned using a MULTI-OBJECTIVE FITNESS
function (Coverage, Delay, Energy) for fair apples-to-apples comparison.

Combines THREE separately-validated fixes into one selector:
  1. Policy H's battery_margin fix (Absolute scale, ranked).
  2. Policy I's coverage_loss fix (Absolute scale).
  3. VIKOR-style compromise combination (S/R/Q) replacing weighted-SUM.
"""

import argparse
import csv
import json
import time
import tie_tracker  # <<< NEW: tie tracker import

import numpy as np
from scipy.optimize import differential_evolution
from scipy.stats import wilcoxon

import a2e_relay_fixed_final_51 as relay

METRICS = ("coverage_pct", "avg_zone_staleness_steps",
           "avg_relay_delay_steps", "avg_relay_energy_pct",
           "total_energy_consumed", "mission_completed")

# Starting point
NORMALIZED_WEIGHTS_J_DEFAULT = {
    "W_TRAVEL_ENERGY": 0.30,
    "W_COVERAGE_LOSS": 0.30,
    "W_RELAY_DELAY": 0.20,
    "W_BATTERY_MARGIN": 0.20,
}
VIKOR_THETA_DEFAULT = 0.5   # VIKOR's textbook v; 1.0 = pure weighted-sum


# ══════════════════════════════════════════════════════════════════════════
# Attribute-level gaps
# ══════════════════════════════════════════════════════════════════════════
def _battery_gap_absolute(battery_margin):
    span = max(relay.BATTERY_MARGIN_SATURATION_PCT - relay.RELAY_SAFETY_FLOOR, 1e-9)
    membership = max(0.0, min(1.0, (battery_margin - relay.RELAY_SAFETY_FLOOR) / span))
    return 1.0 - membership

def _coverage_gap_absolute(coverage_loss):
    span = max(relay.COVERAGE_LOSS_SATURATION_CEILING, 1e-9)
    return max(0.0, min(1.0, coverage_loss / span))

def _pool_gap(values, higher_is_better):
    membership = relay._fuzzy_normalize(values, higher_is_better=higher_is_better)
    return [1.0 - m for m in membership]


# ══════════════════════════════════════════════════════════════════════════
# Policy J core: H's battery fix + I's coverage fix + VIKOR-style Q
# ══════════════════════════════════════════════════════════════════════════
def _relay_utility_pool_j(feasible, needer, g, detected, step, weights=None,
                          fleet=None, theta=VIKOR_THETA_DEFAULT):
    if weights is None:
        weights = NORMALIZED_WEIGHTS_J_DEFAULT

    rows = []
    for cand, travel_energy, battery_margin in feasible:
        dist = relay._manhattan(cand["r"], cand["c"], needer["r"], needer["c"])
        relay_delay = dist
        coverage_loss, raw_marginal_loss, current_tgap, cov_meta = relay._coverage_loss_for_candidate(
            cand, needer, g, detected, step, relay_delay, fleet=fleet
        )
        rows.append({
            "cand": cand, "distance": dist, "travel_energy": travel_energy,
            "battery_margin": battery_margin, "coverage_loss": coverage_loss,
            "raw_marginal_loss": raw_marginal_loss, "current_tgap": current_tgap,
            "cov_meta": cov_meta, "relay_delay": relay_delay,
        })

    if len(rows) == 1:
        r = rows[0]
        diag = {
            "candidate_id": r["cand"]["id"], "role": r["cand"]["role"],
            "distance": r["distance"], "travel_energy": round(r["travel_energy"], 3),
            "battery_margin": round(r["battery_margin"], 2),
            "coverage_loss": round(r["coverage_loss"], 6),
            "relay_delay": r["relay_delay"],
            "vikor_S": 0.0, "vikor_R": 0.0, "vikor_q": 0.0,
            "note": "only feasible candidate -- no ranking needed",
            "coverage_metadata": r["cov_meta"],
        }
        return [(r["cand"], 0.0, diag)]

    energy_gap = _pool_gap([r["travel_energy"] for r in rows], higher_is_better=False)
    delay_gap = _pool_gap([r["relay_delay"] for r in rows], higher_is_better=False)
    cov_gap = [_coverage_gap_absolute(r["coverage_loss"]) for r in rows]
    batt_gap = [_battery_gap_absolute(r["battery_margin"]) for r in rows]

    w = weights
    S_list, R_list, term_rows = [], [], []
    for i in range(len(rows)):
        terms = {
            "energy": w["W_TRAVEL_ENERGY"] * energy_gap[i],
            "coverage": w["W_COVERAGE_LOSS"] * cov_gap[i],
            "delay": w["W_RELAY_DELAY"] * delay_gap[i],
            "battery": w["W_BATTERY_MARGIN"] * batt_gap[i],
        }
        term_rows.append(terms)
        S_list.append(sum(terms.values()))
        R_list.append(max(terms.values()))

    S_star, S_worst = min(S_list), max(S_list)
    R_star, R_worst = min(R_list), max(R_list)
    s_spread = (S_worst - S_star) if (S_worst - S_star) > 1e-12 else 1.0
    r_spread = (R_worst - R_star) if (R_worst - R_star) > 1e-12 else 1.0

    scored = []
    for i, r in enumerate(rows):
        Q_i = (theta * (S_list[i] - S_star) / s_spread
               + (1 - theta) * (R_list[i] - R_star) / r_spread)
        diag = {
            "candidate_id": r["cand"]["id"], "role": r["cand"]["role"],
            "distance": r["distance"], "travel_energy": round(r["travel_energy"], 3),
            "battery_margin": round(r["battery_margin"], 2),
            "coverage_loss": round(r["coverage_loss"], 6),
            "relay_delay": r["relay_delay"],
            "gap_energy": round(energy_gap[i], 3), "gap_coverage": round(cov_gap[i], 3),
            "gap_delay": round(delay_gap[i], 3), "gap_battery": round(batt_gap[i], 3),
            "vikor_S": round(S_list[i], 4), "vikor_R": round(R_list[i], 4),
            "vikor_q": round(Q_i, 4),
            "coverage_metadata": r["cov_meta"],
        }
        scored.append((r["cand"], Q_i, diag))
        
    tie_tracker.log_ties("J", scored, higher_is_better=False)

    scored.sort(key=lambda x: x[1])   # LOWER Q wins
    return scored

def select_best_relay_j(needer, drones, g, detected, step, weights=None,
                        theta=VIKOR_THETA_DEFAULT):
    feasible = relay._feasible_candidate_pool(needer, drones)
    if not feasible:
        return None, []
    scored = _relay_utility_pool_j(feasible, needer, g, detected, step,
                                   weights, fleet=drones, theta=theta)
    best_cand = scored[0][0]
    return best_cand, [d for _, _, d in scored]


# ══════════════════════════════════════════════════════════════════════════
# Weight parameterization: 3 free params -> 4 non-negative weights
# ══════════════════════════════════════════════════════════════════════════
def params_to_weights4(theta_vec):
    a, b, c = [min(max(x, 0.0), 1.0) for x in theta_vec[:3]]
    w_energy = a
    w_coverage = (1.0 - a) * b
    w_delay = (1.0 - a) * (1.0 - b) * c
    w_battery = (1.0 - a) * (1.0 - b) * (1.0 - c)
    return {
        "W_TRAVEL_ENERGY": float(w_energy),
        "W_COVERAGE_LOSS": float(w_coverage),
        "W_RELAY_DELAY": float(w_delay),
        "W_BATTERY_MARGIN": float(w_battery),
    }

def weights_to_tuple4(w):
    return tuple(round(w[k], 4) for k in
                 ("W_TRAVEL_ENERGY", "W_COVERAGE_LOSS", "W_RELAY_DELAY", "W_BATTERY_MARGIN"))

# ══════════════════════════════════════════════════════════════════════════
# Multi-Objective Function for Policy J
# ══════════════════════════════════════════════════════════════════════════
class Objective:
    def __init__(self, seeds, max_steps, num_drones, num_reserves, history_path,
                tune_theta):
        self.seeds = seeds
        self.max_steps = max_steps
        self.num_drones = num_drones
        self.num_reserves = num_reserves
        self.tune_theta = tune_theta
        self.calls = 0
        self.t0 = time.perf_counter()
        self.history_path = history_path
        self._hist_rows = []
        # best stores: (cost, weights_tuple, theta, coverage, delay, energy)
        self._best = (float("inf"), None, None, 0.0, 0.0, 0.0)
        self._popsize_hint = 1

    def __call__(self, theta_vec):
        w = params_to_weights4(theta_vec)
        theta = min(max(theta_vec[3], 0.0), 1.0) if self.tune_theta else VIKOR_THETA_DEFAULT

        def selector(needer, drones, g, detected, step):
            return select_best_relay_j(needer, drones, g, detected, step,
                                       weights=w, theta=theta)

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
        norm_delay = mean_delay / 30.0    # Approx max delay
        norm_energy = mean_energy / 20.0  # Approx max energy
        
        # Minimize this cost
        cost = (alpha * norm_coverage_loss) + (beta * norm_energy) + (gamma * norm_delay)

        self.calls += 1

        if cost < self._best[0]:
            self._best = (cost, weights_to_tuple4(w), round(theta, 3), mean_cov, mean_delay, mean_energy)

        self._hist_rows.append({
            "call": self.calls, "w_energy": w["W_TRAVEL_ENERGY"],
            "w_coverage": w["W_COVERAGE_LOSS"], "w_delay": w["W_RELAY_DELAY"],
            "w_battery": w["W_BATTERY_MARGIN"], "theta": theta,
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
        best_cov = objective._best[3]
        best_delay = objective._best[4]
        best_energy = objective._best[5]
        print(f"  [gen ~{gen:>3}] calls={objective.calls:<5} "
              f"best_cost={best_cost:6.4f} (cov={best_cov:5.2f}%, dly={best_delay:5.2f}, nrg={best_energy:5.2f}) "
              f"weights={objective._best[1]}  theta={objective._best[2]}  convergence={convergence:.4f}")
    return _cb


# ══════════════════════════════════════════════════════════════════════════
# Held-out evaluation
# ══════════════════════════════════════════════════════════════════════════
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
                                   num_drones=num_drones, num_reserves=num_reserves, max_steps=max_steps)
        m_h = relay.simulate_fleet(seed, relay_selector=relay.select_best_relay_battery_aware_capped,
                                   num_drones=num_drones, num_reserves=num_reserves, max_steps=max_steps)
        m_i = relay.simulate_fleet(seed, relay_selector=relay.select_best_relay_coverage_aware,
                                   num_drones=num_drones, num_reserves=num_reserves, max_steps=max_steps)
        m_c = relay.simulate_fleet(seed, relay_selector=relay.select_best_relay,
                                   num_drones=num_drones, num_reserves=num_reserves, max_steps=max_steps)
        m_d = relay.simulate_fleet(seed, relay_selector=relay.POLICY_SELECTORS["D"],
                                   num_drones=num_drones, num_reserves=num_reserves, max_steps=max_steps)
        for label, m in (("J", m_j), ("H", m_h), ("I", m_i), ("C", m_c), ("D", m_d)):
            for k in METRICS:
                v = m.get(k)
                if isinstance(v, bool):
                    v = float(v)
                results[label][k].append(v)
    return results

def _p(a, b):
    a, b = np.array(a, dtype=float), np.array(b, dtype=float)
    return wilcoxon(a, b).pvalue if np.any(a - b != 0) else 1.0

def report(results, seeds):
    print(f"\n{'='*112}\nHELD-OUT REPORT (Policy J, full) -- seeds {seeds[0]}..{seeds[-1]}\n{'='*112}\n")
    print(f"{'metric':<26}{'J':>9}{'H':>9}{'I':>9}{'C':>9}{'D':>9}"
          f"{'p(J-D)':>9}{'p(J-H)':>9}{'p(J-I)':>9}{'p(J-C)':>9}")
    print("-" * 106)

    verdict_bad = []
    j_beats_d = {}
    for m in METRICS:
        j, h, i, c, d = (results[k][m] for k in ("J", "H", "I", "C", "D"))
        p_jd, p_jh, p_ji, p_jc = _p(j, d), _p(j, h), _p(j, i), _p(j, c)
        print(f"{m:<26}{np.mean(j):>9.3f}{np.mean(h):>9.3f}{np.mean(i):>9.3f}"
              f"{np.mean(c):>9.3f}{np.mean(d):>9.3f}{p_jd:>9.4f}{p_jh:>9.4f}"
              f"{p_ji:>9.4f}{p_jc:>9.4f}")

        lower_is_better = m in ("avg_zone_staleness_steps", "avg_relay_delay_steps",
                                "avg_relay_energy_pct", "total_energy_consumed")
        if m in ("avg_relay_delay_steps", "avg_relay_energy_pct") and p_jc < 0.05:
            worse = (np.mean(j) > np.mean(c)) if lower_is_better else (np.mean(j) < np.mean(c))
            if worse:
                verdict_bad.append(m)

        if m == "coverage_pct":
            j_beats_d = {"p": p_jd, "j_mean": np.mean(j), "d_mean": np.mean(d)}

    print()
    if verdict_bad:
        print(f"HONEST FLAG: J significantly REGRESSES {verdict_bad} relative to C.")
    else:
        print("No significant delay/energy regression vs base RelayScore (C).")

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
    p.add_argument("--tune-theta", action="store_true")
    p.add_argument("--out-json", default="tuned_weights_j_full.json")
    p.add_argument("--out-history", default="weight_tuning_history_j_full.csv")
    return p.parse_args()

def main():
    args = _parse_args()

    num_drones = args.num_drones if args.num_drones is not None else relay.NUM_DRONES
    num_reserves = args.num_reserves if args.num_reserves is not None else relay.RESERVE_POOL_SIZE

    tune_seeds = list(range(args.tune_seed_start, args.tune_seed_start + args.tune_seeds))
    test_seeds = list(range(args.test_seed_start, args.test_seed_start + args.test_seeds))
    overlap = set(tune_seeds) & set(test_seeds)
    if overlap:
        raise SystemExit(f"Tuning and test seed blocks overlap.")

    bounds = [(0.0, 1.0)] * (4 if args.tune_theta else 3)

    print("=" * 78)
    print("POLICY J (full) with MULTI-OBJECTIVE FITNESS")
    print("=" * 78)
    print(f"  Tuning block : seeds {tune_seeds[0]}..{tune_seeds[-1]}")
    print(f"  Test block   : seeds {test_seeds[0]}..{test_seeds[-1]}  <-- untouched")
    print(f"  Objective    : MINIMIZE Cost (0.5*CovLoss + 0.25*Energy + 0.25*Delay)")
    print(f"  Optimizer    : scipy differential_evolution, popsize={args.popsize}, maxiter={args.maxiter}\n")

    objective = Objective(tune_seeds, args.tune_steps, num_drones, num_reserves,
                          args.out_history, args.tune_theta)
    objective._popsize_hint = args.popsize * len(bounds)

    t0 = time.perf_counter()
    res = differential_evolution(
        objective, bounds=bounds,
        popsize=args.popsize, maxiter=args.maxiter,
        seed=args.seed_rng, polish=True, tol=1e-3,
        callback=de_callback(objective), disp=False, workers=1,
    )
    objective.flush_history()
    elapsed = time.perf_counter() - t0

    tuned_weights = params_to_weights4(res.x)
    tuned_theta = min(max(res.x[3], 0.0), 1.0) if args.tune_theta else VIKOR_THETA_DEFAULT

    print(f"\nOptimization done in {elapsed:.1f}s, {objective.calls} evaluations.")
    print(f"Tuned weights: {tuned_weights}")
    print(f"Tuned theta: {tuned_theta}")
    print(f"Tuning-set COST at these weights: {res.fun:.4f}")
    print(f"Tuning-set metrics: Cov={objective._best[3]:.2f}%, Delay={objective._best[4]:.2f}, Energy={objective._best[5]:.2f}%")

    with open(args.out_json, "w") as fh:
        json.dump({
            "policy": "J (full) = H battery fix + I coverage fix + VIKOR-style Q (Multi-Objective Tuned)",
            "tuned_weights": tuned_weights,
            "theta": tuned_theta,
            "theta_tuned": args.tune_theta,
            "tuning_cost": res.fun,
            "tuning_metrics": {
                "coverage": objective._best[3],
                "delay": objective._best[4],
                "energy": objective._best[5]
            },
            "tune_seeds": [tune_seeds[0], tune_seeds[-1]],
            "test_seeds": [test_seeds[0], test_seeds[-1]],
            "optimizer": {"method": "scipy.optimize.differential_evolution",
                          "popsize": args.popsize, "maxiter": args.maxiter,
                          "evaluations": objective.calls, "elapsed_s": round(elapsed, 1),
                          "seed": args.seed_rng}
        }, fh, indent=2)
    print(f"Written: {args.out_json}, {args.out_history}")

    print("\nEvaluating J on the HELD-OUT test block...")
    results = evaluate_heldout(tuned_weights, tuned_theta, test_seeds, args.test_steps,
                               num_drones, num_reserves)
    report(results, test_seeds)


if __name__ == "__main__":
    main()