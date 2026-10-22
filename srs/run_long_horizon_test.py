import pandas as pd
import numpy as np
from scipy import stats
import a2e_relay_fixed_final_51 as sim

# Benjamini-Hochberg Correction Function
def bh_correction(p_values):
    p = np.asarray(p_values, float)
    n = len(p)
    if n == 0: return p
    o = np.argsort(p)
    c = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    q = np.empty(n)
    q[o] = np.minimum(c, 1.0)
    return q

def main():
    seeds = list(range(801, 1001)) # 200 seeds
    max_steps = 800 # Long horizon
    
    print(f"Preparing Standard Long Horizon Stress Test for {len(seeds)} seeds (max_steps={max_steps})...")
    
    # Load default standard policies
    policies = {
        "LARS": sim.POLICY_SELECTORS.get("LARS") or sim.select_best_relay_coverage_aware,
        "C": sim.POLICY_SELECTORS.get("C") or sim.select_best_relay,
        "D": sim.POLICY_SELECTORS.get("D") or sim.select_fis_relay,
        "T": sim.POLICY_SELECTORS.get("T"),
        "V": sim.POLICY_SELECTORS.get("V"),
    }
    
    # Add AUC and J if they exist in your registry
    if sim.POLICY_SELECTORS.get("AUC"):
        policies["AUC"] = sim.POLICY_SELECTORS.get("AUC")
    if sim.POLICY_SELECTORS.get("J"):
        policies["J"] = sim.POLICY_SELECTORS.get("J")
        
    # Ensure all policies were found
    policies = {k: v for k, v in policies.items() if v is not None}

    print(f"Policies to be tested: {list(policies.keys())}\n")
    
    # RUN SIMULATION
    rows = []
    for i, seed in enumerate(seeds):
        if i % 10 == 0:
            print(f"Processing seed {seed} ({i}/{len(seeds)})...")
            
        row = {"seed": seed}
        for pol_name, selector in policies.items():
            m = sim.simulate_fleet(seed, relay_selector=selector, max_steps=max_steps)
            row[f"{pol_name}_coverage_pct"] = m.get("coverage_pct")
            row[f"{pol_name}_avg_relay_delay_steps"] = m.get("avg_relay_delay_steps")
            row[f"{pol_name}_avg_relay_energy_pct"] = m.get("avg_relay_energy_pct")
        rows.append(row)
        
    df = pd.DataFrame(rows)
    df.to_csv("long_horizon_standard_800_steps.csv", index=False)
    print("\n✅ Simulation complete! Data saved to long_horizon_standard_800_steps.csv\n")
    
    # STATISTICAL ANALYSIS
    print("="*65)
    print(" SIGNIFICANCE TEST (vs LARS) - 800 STEPS (n=200) ")
    print("="*65)
    
    metrics = ["coverage_pct", "avg_relay_delay_steps", "avg_relay_energy_pct"]
    comparators = [p for p in policies.keys() if p != "LARS"]
    
    for metric in metrics:
        print(f"\n--- Metric: {metric} ---")
        lars_mean = df['LARS_'+metric].mean()
        print(f"LARS Mean: {lars_mean:.2f}")
        
        p_values = []
        valid_comps = []
        
        # Calculate p-values
        for comp in comparators:
            diff = df[f"{comp}_{metric}"] - df[f"LARS_{metric}"]
            comp_mean = df[f"{comp}_{metric}"].mean()
            try:
                p = stats.wilcoxon(diff).pvalue
                p_values.append(p)
                valid_comps.append((comp, comp_mean, diff.mean(), diff))
            except ValueError:
                p_values.append(1.0)
                valid_comps.append((comp, comp_mean, 0.0, diff))
                
        # BH Correction
        q_values = bh_correction(p_values)
        
        # Print Results
        for (comp, mean_val, mean_diff, diff_series), p, q in zip(valid_comps, p_values, q_values):
            sig = "*" if q < 0.05 else ""
            print(f"{comp:5s}: Mean={mean_val:>6.2f} | Diff={mean_diff:>6.2f} | p={p:.4e} | q(BH)={q:.4e} {sig}")
            
            # TOST for Coverage
            if metric == "coverage_pct":
                margin = 2.0
                n = len(diff_series)
                se = diff_series.std(ddof=1) / np.sqrt(n) if diff_series.std(ddof=1) > 0 else 1e-9
                ptost = max(stats.t.sf((mean_diff + margin) / se, n - 1), 
                            stats.t.cdf((mean_diff - margin) / se, n - 1))
                equiv = "Yes" if ptost < 0.05 else "No"
                print(f"       -> TOST Equiv (±{margin}): p_tost={ptost:.4e} [{equiv}]")

if __name__ == "__main__":
    main()