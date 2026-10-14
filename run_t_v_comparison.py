# -*- coding: utf-8 -*-
import argparse
import csv
import numpy as np
from scipy.stats import wilcoxon, rankdata

import a2e_relay_fixed_final_51 as relay
from policy_j_full import select_best_relay_j, METRICS

def _rank_biserial(x, y):
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

def main():
    seeds = list(range(1, 51))
    steps = 500
    theta = 0.5
    
    # Policy J tuned weights
    j_weights = {
        "W_TRAVEL_ENERGY": 0.283556320123423,
        "W_COVERAGE_LOSS": 0.023813783478481596,
        "W_RELAY_DELAY": 0.28445380353708727,
        "W_BATTERY_MARGIN": 0.40817609286100814,
    }

    def j_selector(needer, drones, g, detected, step):
        return select_best_relay_j(needer, drones, g, detected, step, weights=j_weights, theta=theta)

    # Dictionary of policies to test
    selectors = {
        "J": j_selector,
        "T": relay.POLICY_SELECTORS["T"], # Assuming 'T' is the key for TOPSIS in your simulator
        "V": relay.POLICY_SELECTORS["V"], # Assuming 'V' is the key for VIKOR in your simulator
        "D2": relay.POLICY_SELECTORS.get("D2", relay.POLICY_SELECTORS["D"]) # Fallback if D2 isn't in dict
    }
    
    labels = ["J", "T", "V", "D2"]
    rows = []
    
    print("Running simulations for J vs T, V, D2...")
    for seed in seeds:
        row = {"seed": seed}
        for label in labels:
            m = relay.simulate_fleet(seed, relay_selector=selectors[label], max_steps=steps)
            for k in ["coverage_pct", "avg_relay_delay_steps", "avg_relay_energy_pct"]:
                v = m.get(k)
                row[f"{label}_{k}"] = float(v) if isinstance(v, bool) else v
        rows.append(row)
        print(f"Seed {seed} done.")

    # Generate Report
    metrics = ("coverage_pct", "avg_relay_delay_steps", "avg_relay_energy_pct")
    baselines = ["T", "V", "D2"]
    
    print(f"\n{'='*80}\nJ vs TOPSIS(T), VIKOR(V), Surrogate FIS(D2) -- Held-out 50 seeds\n{'='*80}")
    for baseline in baselines:
        print(f"\n--- J vs {baseline} ---")
        print(f"{'Metric':<26}{'J Mean':>10}{baseline+' Mean':>12}{'p-value':>12}{'r (effect)':>12}")
        print("-" * 72)
        for m in metrics:
            j_vals = np.array([r[f"J_{m}"] for r in rows], dtype=float)
            b_vals = np.array([r[f"{baseline}_{m}"] for r in rows], dtype=float)
            
            p_val = _wtest(j_vals, b_vals)
            r_val = _rank_biserial(j_vals, b_vals)
            
            print(f"{m:<26}{j_vals.mean():>10.3f}{b_vals.mean():>12.3f}{p_val:>12.4f}{r_val:>12.3f}")

if __name__ == "__main__":
    main()