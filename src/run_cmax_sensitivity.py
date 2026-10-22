import a2e_relay_fixed_final_51 as sim
import pandas as pd

def run_cmax_sweep():
    # Use a small test block (e.g., 20 seeds) for quick sensitivity checking
    seeds = list(range(101, 121))
    cmax_values = [0.3, 0.6, 1.0]
    
    results = []
    print("Running C_max Sensitivity Analysis...")
    
    for cmax in cmax_values:
        print(f"\nTesting C_max = {cmax}...")
        # Temporarily override the simulator's global constant
        sim.COVERAGE_LOSS_SATURATION_CEILING = cmax
        
        covs, delays, energies = [], [], []
        for seed in seeds:
            metrics = sim.simulate_fleet(seed, relay_selector=sim.select_best_relay_coverage_aware, max_steps=500)
            covs.append(metrics["coverage_pct"])
            delays.append(metrics["avg_relay_delay_steps"])
            energies.append(metrics["avg_relay_energy_pct"])
            
        mean_cov = sum(covs)/len(covs)
        mean_delay = sum(d for d in delays if d is not None)/len(delays)
        mean_energy = sum(e for e in energies if e is not None)/len(energies)
        
        print(f"Result -> Cov: {mean_cov:.2f}%, Delay: {mean_delay:.2f}, Energy: {mean_energy:.2f}%")
        results.append({"C_max": cmax, "Coverage": mean_cov, "Delay": mean_delay, "Energy": mean_energy})
        
    df = pd.DataFrame(results)
    print("\n--- C_max Sensitivity Summary ---")
    print(df.to_string(index=False))

if __name__ == "__main__":
    run_cmax_sweep()