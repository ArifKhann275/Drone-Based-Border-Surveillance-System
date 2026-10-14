# run_tie_check.py
import tie_tracker
import a2e_relay_fixed_final_51 as relay
from policy_j_full import select_best_relay_j, NORMALIZED_WEIGHTS_J_DEFAULT

# J's tuned weights from your previous runs
j_weights = {
    "W_TRAVEL_ENERGY": 0.283556320123423,
    "W_COVERAGE_LOSS": 0.023813783478481596,
    "W_RELAY_DELAY": 0.28445380353708727,
    "W_BATTERY_MARGIN": 0.40817609286100814,
}

def j_selector(needer, drones, g, detected, step):
    return select_best_relay_j(needer, drones, g, detected, step, weights=j_weights, theta=0.5)

print("Running Tie Check on seeds 201-250 (50 seeds)...")

for seed in range(201, 251):
    # Run Policy I
    relay.simulate_fleet(seed, relay_selector=relay.select_best_relay_coverage_aware, max_steps=500)
    # Run Policy J
    relay.simulate_fleet(seed, relay_selector=j_selector, max_steps=500)
    print(f"Seed {seed} done.")

print("\n=== TIE TRACKER REPORT ===")
tie_tracker.report()