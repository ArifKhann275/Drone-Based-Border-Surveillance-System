# -*- coding: utf-8 -*-
"""
run_indep3_4_check.py
Usage: python run_indep3_4_check.py
"""

import numpy as np
from scipy.stats import wilcoxon, rankdata, t
import a2e_relay_fixed_final_51 as relay
import policy_j_full
import policy_d_true_fis

METRICS = ("coverage_pct", "avg_relay_delay_steps", "avg_relay_energy_pct")

# Policy J-এর ফুল প্রিসিশন ওয়েট (Sanity Check 2-এ ভেরিফাইড)
J_WEIGHTS = {
    "W_TRAVEL_ENERGY": 0.283556320123423,
    "W_COVERAGE_LOSS": 0.023813783478481596,
    "W_RELAY_DELAY": 0.28445380353708727,
    "W_BATTERY_MARGIN": 0.40817609286100814
}
THETA = 0.5
SEEDS = list(range(1, 51))
MAX_STEPS = 500

def calc_stats(j_vals, d_vals):
    j_vals, d_vals = np.array(j_vals, dtype=float), np.array(d_vals, dtype=float)
    diff = j_vals - d_vals
    mean_diff = np.mean(diff)
    
    # 95% CI
    n = len(diff)
    se = np.std(diff, ddof=1) / np.sqrt(n)
    if se == 0:
        ci = (mean_diff, mean_diff)
    else:
        tcrit = t.ppf(0.975, n - 1)
        ci = (mean_diff - tcrit * se, mean_diff + tcrit * se)
        
    # Wilcoxon p-value and Rank-biserial r
    if np.all(diff == 0):
        return mean_diff, ci, 1.0, 0.0
    
    p_val = wilcoxon(j_vals, d_vals).pvalue
    
    nz = diff[diff != 0]
    if len(nz) == 0:
        return mean_diff, ci, p_val, 0.0
        
    ranks = rankdata(np.abs(nz))
    r_pos = ranks[nz > 0].sum()
    r_neg = ranks[nz < 0].sum()
    rb = (r_pos - r_neg) / (r_pos + r_neg)
    
    return mean_diff, ci, p_val, rb

def run_condition(selector):
    out = {m: [] for m in METRICS}
    for seed in SEEDS:
        m = relay.simulate_fleet(seed, relay_selector=selector, max_steps=MAX_STEPS)
        for k in METRICS:
            out[k].append(float(m[k]))
    return out

def main():
    print("=" * 85)
    print("INDEPENDENT RULE BASE CHECK (J vs D-indep3 & D-indep4)")
    print("=" * 85)
    
    # 1. Run Policy J
    print("Running Policy J (Full precision weights, theta=0.5)...")
    def j_selector(needer, drones, g, detected, step):
        return policy_j_full.select_best_relay_j(
            needer, drones, g, detected, step, weights=J_WEIGHTS, theta=THETA
        )
    j_results = run_condition(j_selector)

    # 2. Run D-indep3
    print("Running Policy D-indep3 (Expert-Oriented, Energy/Battery Emphasis)...")
    indep3_results = run_condition(policy_d_true_fis.select_fuzzy_relay_true_fis_indep3_weights)

    # 3. Run D-indep4
    print("Running Policy D-indep4 (Balanced Distance/Battery/Delay)...")
    indep4_results = run_condition(policy_d_true_fis.select_fuzzy_relay_true_fis_indep4_weights)

    # Report Generation
    for label, d_results in [("D-indep3", indep3_results), ("D-indep4", indep4_results)]:
        print("\n" + "=" * 85)
        print(f"Policy J vs {label}")
        print("=" * 85)
        print(f"{'Metric':<25} | {'J Mean':<8} | {label+' Mean':<13} | {'Diff':<7} | {'95% CI':<18} | {'p-value':<8} | {'r'}")
        print("-" * 85)
        
        for metric in METRICS:
            j_vals = j_results[metric]
            d_vals = d_results[metric]
            mean_j = np.mean(j_vals)
            mean_d = np.mean(d_vals)
            diff, ci, p_val, r = calc_stats(j_vals, d_vals)
            
            ci_str = f"[{ci[0]:+5.2f}, {ci[1]:+5.2f}]"
            print(f"{metric:<25} | {mean_j:<8.2f} | {mean_d:<13.2f} | {diff:+7.2f} | {ci_str:<18} | {p_val:<8.4f} | {r:+5.2f}")

if __name__ == "__main__":
    main()