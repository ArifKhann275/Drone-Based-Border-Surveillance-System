import pandas as pd
import numpy as np
from scipy import stats
import a2e_relay_fixed_final_51 as sim

def _relay_utility_pool_all_abs(feasible, needer, g, detected, step, weights=None, fleet=None):
    if weights is None:
        weights = sim.NORMALIZED_WEIGHTS_I

    rows = []
    for cand, travel_energy, battery_margin in feasible:
        dist = sim._manhattan(cand["r"], cand["c"], needer["r"], needer["c"])
        relay_delay = dist
        coverage_loss, raw_marginal_loss, current_tgap, cov_meta = sim._coverage_loss_for_candidate(
            cand, needer, g, detected, step, relay_delay, fleet=fleet
        )
        rows.append({
            "cand": cand, "distance": dist, "travel_energy": travel_energy,
            "battery_margin": battery_margin, "coverage_loss": coverage_loss,
            "raw_marginal_loss": raw_marginal_loss, "current_tgap": current_tgap,
            "cov_meta": cov_meta, "relay_delay": relay_delay,
        })

    if len(rows) == 1:
        return sim._relay_utility_pool_i(feasible, needer, g, detected, step, weights, fleet=fleet)

    # --- THE ABLATION TEST: ABSOLUTE NORMALIZATION FOR ALL 3 ATTRIBUTES ---
    ENERGY_MAX = 30.0  # Assumed absolute ceiling for travel energy (%)
    DELAY_MAX = 100.0  # Assumed absolute ceiling for delay (steps)
    COV_MAX = sim.COVERAGE_LOSS_SATURATION_CEILING  # Default 0.6

    # Credit inverted: 0 cost = 1.0 membership, MAX cost = 0.0 membership
    energy_r = [max(0.0, min(1.0, 1.0 - (r["travel_energy"] / ENERGY_MAX))) for r in rows]
    cov_r = [max(0.0, min(1.0, 1.0 - (r["coverage_loss"] / COV_MAX))) for r in rows]
    delay_r = [max(0.0, min(1.0, 1.0 - (r["relay_delay"] / DELAY_MAX))) for r in rows]

    w = weights
    scored = []
    for i, r in enumerate(rows):
        utility = (w["W_TRAVEL_ENERGY"] * energy_r[i]
                   + w["W_COVERAGE_LOSS"] * cov_r[i]
                   + w["W_RELAY_DELAY"] * delay_r[i])

        diag = {
            "candidate_id": r["cand"]["id"], 
            "role": r["cand"]["role"],
            "distance": r["distance"],
            "travel_energy": round(r["travel_energy"], 3),
            "battery_margin": round(r["battery_margin"], 2),
            "coverage_loss": round(r["coverage_loss"], 6),
            "relay_delay": r["relay_delay"],
            "score": round(utility, 4),
        }
        scored.append((r["cand"], utility, diag))

    scored.sort(key=lambda x: -x[1])
    return scored

def select_lars_all_abs(needer, drones, g, detected, step, weights=None):
    feasible = sim._feasible_candidate_pool(needer, drones)
    if not feasible:
        return None, []
    scored = _relay_utility_pool_all_abs(feasible, needer, g, detected, step, weights, fleet=drones)
    best_cand, _best_score, _best_diag = scored[0]
    return best_cand, [diag for _, _, diag in scored]

def main():
    # 200 Seeds from the Confirmatory Block!
    seeds = list(range(401, 601)) 
    print(f"Running Ablation Test: LARS vs LARS_All_Abs on {len(seeds)} Confirmatory Seeds (401-600)...")
    
    sim.POLICY_SELECTORS["LARS_All_Abs"] = select_lars_all_abs
    
    rows = []
    for i, seed in enumerate(seeds):
        if i % 20 == 0: 
            print(f"Processing seed {seed} ({i}/{len(seeds)})...")
        row = {"seed": seed}
        
        m_lars = sim.simulate_fleet(seed, relay_selector=sim.select_best_relay_coverage_aware, max_steps=500)
        row["LARS_cov"] = m_lars["coverage_pct"]
        row["LARS_delay"] = m_lars["avg_relay_delay_steps"]
        row["LARS_energy"] = m_lars["avg_relay_energy_pct"]
        
        m_abs = sim.simulate_fleet(seed, relay_selector=select_lars_all_abs, max_steps=500)
        row["All_Abs_cov"] = m_abs["coverage_pct"]
        row["All_Abs_delay"] = m_abs["avg_relay_delay_steps"]
        row["All_Abs_energy"] = m_abs["avg_relay_energy_pct"]
        
        rows.append(row)
        
    df = pd.DataFrame(rows)
    print("\n" + "="*60)
    print(" ABLATION RESULTS (LARS vs LARS_All_Abs) - Confirmatory Block")
    print("="*60)
    print(f"LARS          -> Cov: {df['LARS_cov'].mean():.2f}%, Delay: {df['LARS_delay'].mean():.2f}, Energy: {df['LARS_energy'].mean():.2f}%")
    print(f"LARS_All_Abs  -> Cov: {df['All_Abs_cov'].mean():.2f}%, Delay: {df['All_Abs_delay'].mean():.2f}, Energy: {df['All_Abs_energy'].mean():.2f}%")
    
    print("\n--- Wilcoxon Significance (LARS_All_Abs minus LARS) ---")
    for metric in ["cov", "delay", "energy"]:
        diff = df[f"All_Abs_{metric}"] - df[f"LARS_{metric}"]
        try:
            p = stats.wilcoxon(diff).pvalue
        except ValueError:
            p = 1.0
        print(f"{metric.capitalize():<6s} Diff: {diff.mean():+.2f} | p = {p:.4e}")

if __name__ == "__main__":
    main()