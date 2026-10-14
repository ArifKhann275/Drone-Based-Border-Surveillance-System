# -*- coding: utf-8 -*-
"""
run_theta_sensitivity.py
Usage: python run_theta_sensitivity.py --test-seeds 50 --w-energy 0.284 --w-coverage 0.024 --w-delay 0.284 --w-battery 0.408
"""

import argparse
import numpy as np
from scipy.stats import wilcoxon, t, rankdata
import a2e_relay_fixed_final_51 as relay
import policy_j_full

METRICS = ("coverage_pct", "avg_relay_delay_steps", "avg_relay_energy_pct")

# গ্লোবাল ইনস্ট্রুমেন্টেশন ট্র্যাকার
INSTRUMENT = {
    "total_decisions": 0,
    "nc_2": 0,
    "s_r_disagree": 0,
    "ties": 0
}

def instrumented_j_selector(needer, drones, g, detected, step, weights, theta):
    feasible = relay._feasible_candidate_pool(needer, drones)
    if not feasible:
        return None, []
    
    scored = policy_j_full._relay_utility_pool_j(
        feasible, needer, g, detected, step, weights, fleet=drones, theta=theta
    )
    
    # ইনস্ট্রুমেন্টেশন লগিং
    INSTRUMENT["total_decisions"] += 1
    if len(scored) == 2:
        INSTRUMENT["nc_2"] += 1
        
        s0, r0 = scored[0][2]["vikor_S"], scored[0][2]["vikor_R"]
        s1, r1 = scored[1][2]["vikor_S"], scored[1][2]["vikor_R"]
        
        # S এবং R এর মধ্যে দ্বিমত (S একজনকে ভালো বলছে, R অন্যজনকে)
        if (s0 < s1 and r0 > r1) or (s0 > s1 and r0 < r1):
            INSTRUMENT["s_r_disagree"] += 1
            
        # Q স্কোরে টাই
        if abs(scored[0][1] - scored[1][1]) < 1e-9:
            INSTRUMENT["ties"] += 1

    best_cand = scored[0][0]
    return best_cand, [d for _, _, d in scored]

def calc_stats(j_vals, d_vals):
    diff = np.array(j_vals) - np.array(d_vals)
    mean_diff = np.mean(diff)
    
    # 95% CI
    se = np.std(diff, ddof=1) / np.sqrt(len(diff))
    if se == 0:
        ci = (mean_diff, mean_diff)
    else:
        ci = t.interval(0.95, df=len(diff)-1, loc=mean_diff, scale=se)
        
    # Wilcoxon p-value
    if np.all(diff == 0):
        return mean_diff, ci, 1.0, 0.0
    
    p_val = wilcoxon(j_vals, d_vals).pvalue
    
    # Rank-biserial effect size (r)
    diff_nz = diff[diff != 0]
    if len(diff_nz) == 0:
        return mean_diff, ci, p_val, 0.0
        
    ranks = rankdata(np.abs(diff_nz))
    r_plus = np.sum(ranks[diff_nz > 0])
    r_minus = np.sum(ranks[diff_nz < 0])
    rb = (r_plus - r_minus) / (r_plus + r_minus)
    
    return mean_diff, ci, p_val, rb

def run_evaluation():
    p = argparse.ArgumentParser()
    p.add_argument("--w-energy", type=float, default=0.284)
    p.add_argument("--w-coverage", type=float, default=0.024)
    p.add_argument("--w-delay", type=float, default=0.284)
    p.add_argument("--w-battery", type=float, default=0.408)
    p.add_argument("--test-seeds", type=int, default=50)
    args = p.parse_args()

    weights = {
        "W_TRAVEL_ENERGY": args.w_energy,
        "W_COVERAGE_LOSS": args.w_coverage,
        "W_RELAY_DELAY": args.w_delay,
        "W_BATTERY_MARGIN": args.w_battery,
    }
    
    seeds = list(range(1, args.test_seeds + 1))
    thetas = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
    
    print("Running baseline Policy D...")
    d_results = {m: [] for m in METRICS}
    for seed in seeds:
        res = relay.simulate_fleet(seed, relay_selector=relay.POLICY_SELECTORS["D"], max_steps=500)
        for m in METRICS: d_results[m].append(res[m])

    all_j_results = {}
    instrument_logs = {}
    
    for theta in thetas:
        print(f"Running Policy J with theta = {theta}...")
        
        # Reset instrumentation for this theta
        for k in INSTRUMENT: INSTRUMENT[k] = 0
            
        def j_selector(needer, drones, g, detected, step):
            return instrumented_j_selector(needer, drones, g, detected, step, weights, theta)
            
        j_res = {m: [] for m in METRICS}
        for seed in seeds:
            res = relay.simulate_fleet(seed, relay_selector=j_selector, max_steps=500)
            for m in METRICS: j_res[m].append(res[m])
            
        all_j_results[theta] = j_res
        instrument_logs[theta] = INSTRUMENT.copy()

    # Reporting
    print("\n" + "="*90)
    print("THETA SENSITIVITY ANALYSIS (vs Policy D)")
    print("="*90)
    
    for metric in METRICS:
        print(f"\n--- Metric: {metric} ---")
        print(f"{'Theta':<6} | {'J Mean':<8} | {'Diff(J-D)':<10} | {'95% CI':<18} | {'p-value':<8} | {'Effect (r)':<8}")
        print("-" * 70)
        
        d_vals = d_results[metric]
        for theta in thetas:
            j_vals = all_j_results[theta][metric]
            mean_j = np.mean(j_vals)
            diff, ci, p_val, r = calc_stats(j_vals, d_vals)
            
            ci_str = f"[{ci[0]:+5.2f}, {ci[1]:+5.2f}]"
            print(f"{theta:<6.2f} | {mean_j:<8.2f} | {diff:+10.2f} | {ci_str:<18} | {p_val:<8.4f} | {r:+8.2f}")

    print("\n" + "="*90)
    print("INSTRUMENTATION LOG (Structural behavior of the candidate pool)")
    print("="*90)
    print(f"{'Theta':<6} | {'Total Dec.':<10} | {'Nc=2 (%)':<10} | {'S vs R Disagree (%)':<20} | {'Ties (%)':<10}")
    print("-" * 70)
    for theta in thetas:
        log = instrument_logs[theta]
        tot = log["total_decisions"]
        nc2_pct = (log["nc_2"] / tot * 100) if tot else 0
        disagree_pct = (log["s_r_disagree"] / log["nc_2"] * 100) if log["nc_2"] else 0
        ties_pct = (log["ties"] / log["nc_2"] * 100) if log["nc_2"] else 0
        
        print(f"{theta:<6.2f} | {tot:<10} | {nc2_pct:<9.1f}% | {disagree_pct:<19.1f}% | {ties_pct:<9.1f}%")

if __name__ == "__main__":
    run_evaluation()