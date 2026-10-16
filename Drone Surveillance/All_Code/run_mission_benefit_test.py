import pandas as pd
import numpy as np
from scipy import stats
import a2e_relay_fixed_final_51 as sim

def bh_correction(p_values):
    p = np.asarray(p_values, float)
    n = len(p)
    if n == 0:
        return p
    o = np.argsort(p)
    c = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    q = np.empty(n)
    q[o] = np.minimum(c, 1.0)
    return q

def get_physical_constants():
    # Extract cell size in meters and drone speed from a20 if available
    cell_m = getattr(sim.a20, "CELL_SIZE_M", getattr(sim.a20, "cell_size_m", 499.6))
    speed_mps = (
        getattr(sim.a20, "CRUISE_SPEED_MPS", None)
        or getattr(sim.a20, "DRONE_SPEED_MPS", None)
        or getattr(sim.a20, "V_CRUISE", None)
        or 15.0  # fallback standard multirotor cruise speed (15 m/s)
    )
    step_sec = cell_m / float(speed_mps)
    return float(cell_m), float(speed_mps), float(step_sec)

def extract_mission_metrics(m, cell_m, step_sec):
    # Calculate exact cumulative unattended zone-steps from decision_log
    chosen_delays = []
    for entry in m.get("decision_log", []):
        chosen_id = entry.get("chosen_candidate_id")
        for diag in entry.get("candidates_diag", []):
            if diag.get("candidate_id") == chosen_id:
                chosen_delays.append(diag.get("relay_delay", 0))
                break

    avg_delay = m.get("avg_relay_delay_steps") or 0.0
    fulfilled = m.get("relay_fulfilled") or 0

    # Unattended zone-steps: sum of relay delays across all handoffs in the mission
    unattended_zone_steps = sum(chosen_delays) if chosen_delays else (avg_delay * fulfilled)
    unattended_dist_km = (unattended_zone_steps * cell_m) / 1000.0
    unattended_time_min = (unattended_zone_steps * step_sec) / 60.0
    delay_m = avg_delay * cell_m
    delay_s = avg_delay * step_sec

    return {
        "coverage_pct": m.get("coverage_pct"),
        "avg_relay_delay_steps": avg_delay,
        "avg_relay_delay_m": delay_m,
        "avg_relay_delay_s": delay_s,
        "relay_fulfilled": fulfilled,
        "unattended_zone_steps": unattended_zone_steps,
        "unattended_dist_km": unattended_dist_km,
        "unattended_time_min": unattended_time_min,
        "avg_zone_staleness_steps": m.get("avg_zone_staleness_steps"),
        "avg_relay_energy_pct": m.get("avg_relay_energy_pct"),
        "total_energy_consumed": m.get("total_energy_consumed"),
        "detection_rate": m.get("detection_rate"),
    }

def main():
    cell_m, speed_mps, step_sec = get_physical_constants()
    print(f"Physical Constants -> Cell Size: {cell_m:.1f} m | Assumed/Model Speed: {speed_mps:.1f} m/s | 1 Step = {step_sec:.1f} s")

    # Using 100 seeds from the confirmatory block (401-500) for fast, rigorous testing
    # (You can change 501 to 601 if you want all 200 seeds)
    seeds = list(range(401, 501))
    max_steps = 500

    # Settings where coverage is < 90%:
    # N=6 (3 patrol, 3 reserve) and N=7 (4 patrol, 3 reserve)
    configs = [
        ("N=6 (3P+3R)", 3, 3),
        ("N=7 (4P+3R)", 4, 3),
    ]

    policies = {
        "LARS": sim.select_best_relay_coverage_aware,
        "C": sim.select_best_relay,
        "D": sim.POLICY_SELECTORS["D"],
    }

    all_rows = []
    for cfg_name, n_patrol, n_reserve in configs:
        print(f"\n=======================================================")
        print(f" Running {cfg_name} on {len(seeds)} seeds (max_steps={max_steps})...")
        print(f"=======================================================")

        cfg_rows = []
        for i, seed in enumerate(seeds):
            if i % 20 == 0:
                print(f"  [{cfg_name}] Processing seed {seed} ({i}/{len(seeds)})...")
            row = {"config": cfg_name, "seed": seed, "N": n_patrol + n_reserve}
            for pol_name, selector in policies.items():
                m = sim.simulate_fleet(
                    seed,
                    relay_selector=selector,
                    num_drones=n_patrol,
                    num_reserves=n_reserve,
                    max_steps=max_steps,
                )
                ext = extract_mission_metrics(m, cell_m, step_sec)
                for k, v in ext.items():
                    row[f"{pol_name}_{k}"] = v
            cfg_rows.append(row)
            all_rows.append(row)

        df_cfg = pd.DataFrame(cfg_rows)
        print_summary(df_cfg, cfg_name, cell_m, step_sec)

    pd.DataFrame(all_rows).to_csv("mission_benefit_stressed_N6_N7.csv", index=False)
    print("\n✅ Saved full results to mission_benefit_stressed_N6_N7.csv")

def print_summary(df, cfg_name, cell_m, step_sec):
    print(f"\n--- RESULTS FOR {cfg_name} (n={len(df)}) ---")
    metrics = [
        ("coverage_pct", "Coverage (%)"),
        ("avg_relay_delay_steps", "Relay Delay (steps)"),
        ("avg_relay_delay_m", "Relay Delay (meters)"),
        ("unattended_zone_steps", "Unattended Zone-Steps / Mission"),
        ("unattended_dist_km", "Total Relay Travel (km / Mission)"),
        ("avg_zone_staleness_steps", "Zone Staleness (steps)"),
        ("avg_relay_energy_pct", "Relay Energy (%)"),
        ("total_energy_consumed", "Total Energy Consumed"),
        ("detection_rate", "Detection Rate"),
    ]

    for m_key, m_label in metrics:
        lars_m = df[f"LARS_{m_key}"].mean()
        c_m = df[f"C_{m_key}"].mean()
        d_m = df[f"D_{m_key}"].mean()

        diff_c = df[f"LARS_{m_key}"] - df[f"C_{m_key}"]
        diff_d = df[f"LARS_{m_key}"] - df[f"D_{m_key}"]

        try:
            p_c = stats.wilcoxon(diff_c).pvalue
        except ValueError:
            p_c = 1.0
        try:
            p_d = stats.wilcoxon(diff_d).pvalue
        except ValueError:
            p_d = 1.0

        print(
            f"{m_label:34s} | LARS: {lars_m:7.2f} | C: {c_m:7.2f} (Diff={diff_c.mean():+6.2f}, p={p_c:.2e}) "
            f"| D: {d_m:7.2f} (Diff={diff_d.mean():+6.2f}, p={p_d:.2e})"
        )

if __name__ == "__main__":
    main()