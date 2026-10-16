import pandas as pd
import numpy as np
import a2e_relay_fixed_final_51 as sim

def analyze_decision_pool(candidates_diag):
    """
    Reconstructs exact RelayScore (C) and LARS utilities on the same candidate pool
    to see raw spread, scale inflation, and exact ranking flips.
    """
    n_c = len(candidates_diag)
    if n_c < 2:
        return None

    c_vals = [d["coverage_loss"] for d in candidates_diag]
    e_vals = [d["travel_energy"] for d in candidates_diag]
    l_vals = [d["relay_delay"] for d in candidates_diag]

    raw_c_spread = max(c_vals) - min(c_vals)
    raw_e_spread = max(e_vals) - min(e_vals)
    raw_l_spread = max(l_vals) - min(l_vals)

    C_MAX = sim.COVERAGE_LOSS_SATURATION_CEILING  # 0.6
    lars_c_spread = min(1.0, max(c_vals) / C_MAX) - min(1.0, min(c_vals) / C_MAX)

    # Pool-relative normalization has spread = 1.0 whenever raw_c_spread > 1e-12
    # Inflation factor = 1.0 / lars_c_spread (how much min-max exaggerates the gap vs LARS)
    inflation_factor = (1.0 / lars_c_spread) if lars_c_spread > 1e-9 else 0.0

    # Compute RelayScore (C) and LARS scores on this exact pool
    w_e = sim.NORMALIZED_WEIGHTS["W_TRAVEL_ENERGY"]   # 0.35
    w_c = sim.NORMALIZED_WEIGHTS["W_COVERAGE_LOSS"]   # 0.40
    w_l = sim.NORMALIZED_WEIGHTS["W_RELAY_DELAY"]     # 0.25

    mu_e = sim._fuzzy_normalize(e_vals, higher_is_better=False)
    mu_l = sim._fuzzy_normalize(l_vals, higher_is_better=False)
    mu_c_rel = sim._fuzzy_normalize(c_vals, higher_is_better=False)
    mu_c_abs = [max(0.0, min(1.0, 1.0 - (c / C_MAX))) for c in c_vals]

    scores_c = [w_e * mu_e[i] + w_c * mu_c_rel[i] + w_l * mu_l[i] for i in range(n_c)]
    scores_lars = [w_e * mu_e[i] + w_c * mu_c_abs[i] + w_l * mu_l[i] for i in range(n_c)]

    # Winner indices (highest score wins, tie-broken by first index)
    win_c_idx = int(np.argmax(scores_c))
    win_lars_idx = int(np.argmax(scores_lars))

    winner_flipped = (win_c_idx != win_lars_idx)

    win_c = candidates_diag[win_c_idx]
    win_lars = candidates_diag[win_lars_idx]

    return {
        "N_c": n_c,
        "raw_c_spread": raw_c_spread,
        "lars_c_spread": lars_c_spread,
        "inflation_factor": inflation_factor,
        "raw_e_spread": raw_e_spread,
        "raw_l_spread": raw_l_spread,
        "winner_flipped": int(winner_flipped),
        "c_winner_role": win_c.get("role", ""),
        "lars_winner_role": win_lars.get("role", ""),
        "c_winner_cov_loss": win_c["coverage_loss"],
        "lars_winner_cov_loss": win_lars["coverage_loss"],
        "c_winner_delay": win_c["relay_delay"],
        "lars_winner_delay": win_lars["relay_delay"],
        "c_winner_energy": win_c["travel_energy"],
        "lars_winner_energy": win_lars["travel_energy"],
        # Savings when LARS picks its winner instead of C's winner on this decision:
        "delay_saved_by_lars": win_c["relay_delay"] - win_lars["relay_delay"],
        "energy_saved_by_lars": win_c["travel_energy"] - win_lars["travel_energy"],
        "cov_loss_added_by_lars": win_lars["coverage_loss"] - win_c["coverage_loss"],
    }

def main():
    # Run on Confirmatory Block seeds (401-600, n=200)
    seeds = list(range(401, 601))
    print(f"Running Mechanism Instrumentation on {len(seeds)} Confirmatory Seeds (401-600)...")

    decision_rows = []
    for i, seed in enumerate(seeds):
        if i % 25 == 0:
            print(f"  Processing seed {seed} ({i}/{len(seeds)})...")
        # Run LARS trajectory and inspect every decision pool encountered
        m = sim.simulate_fleet(seed, relay_selector=sim.select_best_relay_coverage_aware, max_steps=500)
        for entry in m.get("decision_log", []):
            res = analyze_decision_pool(entry.get("candidates_diag", []))
            if res is not None:
                res["seed"] = seed
                res["step"] = entry.get("step")
                decision_rows.append(res)

    df = pd.DataFrame(decision_rows)
    df.to_csv("mechanism_spread_and_flips.csv", index=False)
    print(f"\n✅ Saved {len(df)} decision records to mechanism_spread_and_flips.csv\n")

    print("=" * 80)
    print(" 1. RAW COVERAGE-LOSS SPREAD & INFLATION BY POOL SIZE (N_c)")
    print("=" * 80)
    summary_rows = []
    for nc, grp in df.groupby("N_c"):
        n_dec = len(grp)
        pct_dec = 100.0 * n_dec / len(df)
        mean_raw = grp["raw_c_spread"].mean()
        med_raw = grp["raw_c_spread"].median()
        p90_raw = np.percentile(grp["raw_c_spread"], 90)
        mean_lars = grp["lars_c_spread"].mean()
        # Filter positive spreads for inflation factor
        pos_inf = grp[grp["lars_c_spread"] > 1e-9]["inflation_factor"]
        med_inf = pos_inf.median() if len(pos_inf) > 0 else 0.0
        flip_rate = 100.0 * grp["winner_flipped"].mean()

        print(
            f"N_c={nc} | Decisions={n_dec:4d} ({pct_dec:5.1f}%) | "
            f"Raw Spread: Mean={mean_raw:.4f}, Med={med_raw:.4f}, P90={p90_raw:.4f} | "
            f"LARS Spread Mean={mean_lars:.4f} (vs C=1.00) | Med Inflation={med_inf:5.1f}x | "
            f"Rank Flip={flip_rate:5.1f}%"
        )
        summary_rows.append({
            "N_c": nc, "decisions": n_dec, "share_pct": round(pct_dec, 2),
            "raw_spread_mean": round(mean_raw, 4), "raw_spread_median": round(med_raw, 4),
            "raw_spread_p90": round(p90_raw, 4), "lars_spread_mean": round(mean_lars, 4),
            "median_inflation_x": round(med_inf, 1), "rank_flip_pct": round(flip_rate, 2)
        })

    pd.DataFrame(summary_rows).to_csv("mechanism_summary_by_Nc.csv", index=False)

    print("\n" + "=" * 80)
    print(" 2. OVERALL MECHANISM & RANKING FLIP ANALYSIS (ALL N_c >= 2)")
    print("=" * 80)
    flips = df[df["winner_flipped"] == 1]
    print(f"Total decisions (N_c >= 2): {len(df)}")
    print(f"Overall Mean Raw Coverage-Loss Spread:   {df['raw_c_spread'].mean():.4f} (Median: {df['raw_c_spread'].median():.4f})")
    print(f"Overall Mean LARS Normalized Spread:     {df['lars_c_spread'].mean():.4f} (whereas RelayScore forces 1.0000)")
    print(f"Decisions where RelayScore & LARS disagree (Winner Flipped): {len(flips)} / {len(df)} ({100.0*len(flips)/len(df):.2f}%)")

    if len(flips) > 0:
        print("\n--- Inside the Flipped Decisions (When Pool-Relative vs Absolute Disagree) ---")
        print(f"Mean Raw Coverage-Loss Difference (LARS winner - C winner): {flips['cov_loss_added_by_lars'].mean():+.4f} (out of C_max=0.60)")
        print(f"Mean Relay Delay Saved by LARS per flipped decision:        {flips['delay_saved_by_lars'].mean():+.2f} steps")
        print(f"Mean Relay Energy Saved by LARS per flipped decision:       {flips['energy_saved_by_lars'].mean():+.2f} %")
        c_res_pct = 100.0 * (flips["c_winner_role"] == "idle_reserve").mean()
        lars_res_pct = 100.0 * (flips["lars_winner_role"] == "idle_reserve").mean()
        print(f"Idle Reserve chosen in flipped decisions -> RelayScore (C): {c_res_pct:.1f}% | LARS: {lars_res_pct:.1f}%")

if __name__ == "__main__":
    main()