# -*- coding: utf-8 -*-
import json
import csv
import numpy as np
from scipy.stats import wilcoxon, rankdata

import a2e_relay_fixed_final_51 as relay
from policy_j_full import select_best_relay_j

def _rank_biserial(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    d = x - y
    nz = d[d != 0]
    if len(nz) == 0: return 0.0
    ranks = rankdata(np.abs(nz))
    return float((ranks[nz > 0].sum() - ranks[nz < 0].sum()) / (ranks[nz > 0].sum() + ranks[nz < 0].sum()))

def _wtest(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    d = x - y
    return wilcoxon(x, y).pvalue if np.any(d != 0) else 1.0

def main():
    with open("tuned_baselines.json", "r") as f:
        tuned_w = json.load(f)

    # Policy J tuned weights (from paper)
    j_weights = {
        "W_TRAVEL_ENERGY": 0.283556320123423,
        "W_COVERAGE_LOSS": 0.023813783478481596,
        "W_RELAY_DELAY": 0.28445380353708727,
        "W_BATTERY_MARGIN": 0.40817609286100814,
    }

    def j_selector(needer, drones, g, detected, step):
        return select_best_relay_j(needer, drones, g, detected, step, weights=j_weights, theta=0.5)

    def c_selector(needer, drones, g, detected, step):
        relay.W_ENERGY_C = tuned_w["C"]["w_energy"]
        relay.W_COVERAGE_C = tuned_w["C"]["w_coverage"]
        relay.W_DELAY_C = tuned_w["C"]["w_delay"]
        return relay.select_best_relay(needer, drones, g, detected, step)

    def t_selector(needer, drones, g, detected, step):
        relay.WEIGHTS_D = (tuned_w["T"]["w_distance"], tuned_w["T"]["w_battery"], tuned_w["T"]["w_coverage"], tuned_w["T"]["w_delay"])
        return relay.POLICY_SELECTORS["T"](needer, drones, g, detected, step)

    def v_selector(needer, drones, g, detected, step):
        relay.WEIGHTS_D = (tuned_w["V"]["w_distance"], tuned_w["V"]["w_battery"], tuned_w["V"]["w_coverage"], tuned_w["V"]["w_delay"])
        return relay.POLICY_SELECTORS["V"](needer, drones, g, detected, step)

    selectors = {"J": j_selector, "C_tuned": c_selector, "T_tuned": t_selector, "V_tuned": v_selector}
    labels = ["J", "C_tuned", "T_tuned", "V_tuned"]
    
    rows = []
    seeds = list(range(1, 51))
    print("Running simulations on held-out seeds 1-50...")
    for seed in seeds:
        row = {"seed": seed}
        for label in labels:
            m = relay.simulate_fleet(seed, relay_selector=selectors[label], max_steps=500)
            for k in ["coverage_pct", "avg_relay_delay_steps", "avg_relay_energy_pct"]:
                row[f"{label}_{k}"] = m[k]
        rows.append(row)
        print(f"Seed {seed} done.")

    metrics = ("coverage_pct", "avg_relay_delay_steps", "avg_relay_energy_pct")
    print(f"\n{'='*80}\nPolicy J vs TUNED Baselines (C, T, V) -- Held-out 50 seeds\n{'='*80}")
    
    for baseline in ["C_tuned", "T_tuned", "V_tuned"]:
        print(f"\n--- J vs {baseline} ---")
        print(f"{'Metric':<26}{'J Mean':>10}{baseline:>12}{'p-value':>12}{'r':>10}")
        print("-" * 70)
        for m in metrics:
            j_vals = np.array([r[f"J_{m}"] for r in rows], dtype=float)
            b_vals = np.array([r[f"{baseline}_{m}"] for r in rows], dtype=float)
            p_val = _wtest(j_vals, b_vals)
            r_val = _rank_biserial(j_vals, b_vals)
            print(f"{m:<26}{j_vals.mean():>10.3f}{b_vals.mean():>12.3f}{p_val:>12.4f}{r_val:>10.3f}")

if __name__ == "__main__":
    main()