# -*- coding: utf-8 -*-
"""
fleet_relay_optimization.py
════════════════════════════════════════════════════════════════════════
PATH B — Cooperative Multi-Drone Fleet with Optimization-Based Relay
Selection (extends a20.py from "nearest-station handoff" to a genuine
N-drone RelayScore_i = Benefit_i − Cost_i utility-function selection).
════════════════════════════════════════════════════════════════════════
"""

import math
import random
import statistics
import csv
import json
from copy import deepcopy
from functools import partial  # Added for Sensitivity Analysis dynamic weights

import a20  # noqa: reuse the existing physics/grid/zone engine as-is

from scipy.stats import wilcoxon, ttest_rel


# ══════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════
NUM_DRONES = 4                  # simultaneously-patrolling drones (fixed, matches a20.STATIONS)
RESERVE_POOL_SIZE = 4           # extra idle spare drones (one pre-positioned per station)
RELAY_SAFETY_FLOOR = 8.0       # % battery a candidate must keep in reserve after the trip
COVERAGE_STALENESS_CAP_STEPS = 200   # same normalization a20.zone_info() uses (tgap)

# Utility-function weights (Moved to a dictionary for Sensitivity Analysis)
DEFAULT_WEIGHTS = {
    "W_TRAVEL_ENERGY": 1.0,           # already in %battery units -> weight 1.0 is "as-is"
    "W_COVERAGE_LOSS": 25.0,          # scales the 0..1 staleness fraction into %battery-comparable units
    "W_RELAY_DELAY": 0.4,             # small tie-breaker weight on ETA (steps)
    "COVERAGE_MAINTAINED_BONUS": 12.0 # benefit bonus ONLY for the "fresh station reserve" candidate
}


# ══════════════════════════════════════════════════════════════
# FLEET INITIALIZATION
# ══════════════════════════════════════════════════════════════
def _new_drone(idx, seed):
    r, c = a20.STATIONS[idx % len(a20.STATIONS)]
    start_r, start_c = (1, 1) if (r, c) == (0, 0) else \
                        (1, a20.COLS - 2) if (r, c) == (0, a20.COLS - 1) else \
                        (a20.ROWS - 2, 1) if (r, c) == (a20.ROWS - 1, 0) else \
                        (a20.ROWS - 2, a20.COLS - 2)
    return {
        "id": idx,
        "home_station": (r, c),
        "r": start_r, "c": start_c, "pr": None, "pc": None,
        "battery": 100.0,
        "target_zone": None, "target_cell": None,
        "zone_rng": random.Random(seed * 97 + idx * 31 + 17),
        "role": "patrol",          # patrol | relay_incoming | returning
        "rendezvous_id": None,     # which drone id this one is relaying toward (if relay_incoming)
        "awaiting_relay": False,   # True once this drone has flagged low battery and a relay is inbound
        "return_target": None,    # station coords this drone heads to when role == returning
    }


def init_fleet(seed, num_drones=NUM_DRONES, num_reserves=RESERVE_POOL_SIZE):
    g = a20.make_grid(seed)
    a20.place_threats(seed, [g])
    drones = [_new_drone(i, seed) for i in range(num_drones)]
    for d in drones:
        d["role"] = "patrol"

    for j in range(num_reserves):
        idx = num_drones + j
        st = a20.STATIONS[j % len(a20.STATIONS)]
        rd = _new_drone(idx, seed)
        rd["r"], rd["c"] = st       
        rd["role"] = "idle_reserve"
        rd["home_station"] = st
        drones.append(rd)

    return {
        "g": g,
        "detected": set(),
        "drones": drones,
        "step": 0,
        "RC": 0,          
        "target_patrol_count": num_drones,   
        "log": [],
    }


# ══════════════════════════════════════════════════════════════
# RELAYSCORE — the 5-factor utility function
# ══════════════════════════════════════════════════════════════
def _manhattan(r1, c1, r2, c2):
    return abs(r1 - r2) + abs(c1 - c2)


def relay_score(candidate, needer, g, detected, step, weights=None):
    if weights is None:
        weights = DEFAULT_WEIGHTS

    dist = _manhattan(candidate["r"], candidate["c"], needer["r"], needer["c"])
    travel_energy = a20.energy_to_travel(candidate["r"], candidate["c"],
                                          needer["r"], needer["c"])
    battery_margin = candidate["battery"] - travel_energy - RELAY_SAFETY_FLOOR
    
    if battery_margin < 0:
        return None, {"reason": "insufficient_battery", "battery_margin": round(battery_margin, 2)}

    if candidate["role"] == "idle_reserve":
        coverage_loss = 0.0
        coverage_maintained = weights["COVERAGE_MAINTAINED_BONUS"]
    else:
        if candidate["target_zone"] is not None:
            info = a20.zone_info(candidate["target_zone"][0], candidate["target_zone"][1],
                                  0, 0, g, detected, step)
            coverage_loss = info["tgap"] if info else 0.0
        else:
            coverage_loss = 0.0
        coverage_maintained = 0.0

    relay_delay = dist

    benefit = coverage_maintained + battery_margin
    cost = (weights["W_TRAVEL_ENERGY"] * travel_energy
            + weights["W_COVERAGE_LOSS"] * coverage_loss
            + weights["W_RELAY_DELAY"] * relay_delay)
    score = benefit - cost

    return score, {
        "distance": dist, "travel_energy": round(travel_energy, 3),
        "battery_margin": round(battery_margin, 2),
        "coverage_loss": round(coverage_loss, 3),
        "coverage_maintained": coverage_maintained,
        "relay_delay": relay_delay, "score": round(score, 3),
    }


def _candidate_pool(needer, drones):
    return [d for d in drones
            if d["id"] != needer["id"] and not d["awaiting_relay"]
            and d["role"] in ("patrol", "idle_reserve")]


def select_best_relay(needer, drones, g, detected, step, weights=None):
    candidates = _candidate_pool(needer, drones)
    scored = []
    for cand in candidates:
        s, diag = relay_score(cand, needer, g, detected, step, weights=weights)
        if s is not None:
            scored.append((cand, s, diag))

    if not scored:
        return None, []

    scored.sort(key=lambda x: -x[1])
    best_cand, best_score, best_diag = scored[0]
    return best_cand, [d for _, _, d in scored]


# ══════════════════════════════════════════════════════════════
# BASELINE (generalized nearest-only, for fair ablation comparison)
# ══════════════════════════════════════════════════════════════
def select_nearest_relay(needer, drones, g, detected, step):
    candidates = _candidate_pool(needer, drones)
    if not candidates:
        return None, []
    best = min(candidates, key=lambda d: _manhattan(d["r"], d["c"], needer["r"], needer["c"]))
    return best, []


# ══════════════════════════════════════════════════════════════
# ONE SIMULATION STEP FOR THE WHOLE FLEET
# ══════════════════════════════════════════════════════════════
def step_fleet(state, relay_selector=select_best_relay):
    g, detected, drones, step = state["g"], state["detected"], state["drones"], state["step"]
    by_id = {d["id"]: d for d in drones}

    # ---- 1. advance drones that are RETURNING to recharge ----
    for d in drones:
        if d["role"] != "returning":
            continue
        tr, tc = d["return_target"]
        if (d["r"], d["c"]) == (tr, tc):
            d["battery"] = 100.0
            d["target_zone"] = None; d["target_cell"] = None
            d["pr"] = d["pc"] = None
            state["RC"] += 1
            active_now = sum(1 for x in drones
                              if x["id"] != d["id"] and x["role"] in ("patrol", "relay_incoming"))
            if active_now < state["target_patrol_count"]:
                d["role"] = "patrol"
            else:
                d["role"] = "idle_reserve"
        else:
            old_r, old_c = d["r"], d["c"]
            nr, nc = a20.move_toward(d["r"], d["c"], tr, tc)
            d["r"], d["c"] = nr, nc
            cost = a20.calculate_actual_step_cost(old_r, old_c, nr, nc, None, None)
            d["battery"] = max(0.0, d["battery"] - cost)
            a20.sensor_sweep(nr, nc, g, detected, step=step)

    # ---- 2. advance drones that are RELAY_INCOMING toward a needer ----
    for d in drones:
        if d["role"] != "relay_incoming":
            continue
        needer = by_id.get(d["rendezvous_id"])
        if needer is None:
            d["role"] = "patrol"; continue
        tr, tc = needer["r"], needer["c"]
        dist_now = _manhattan(d["r"], d["c"], tr, tc)
        if dist_now <= 1:
            st = a20.nearest_st(needer["r"], needer["c"])
            needer["role"] = "returning"
            needer["return_target"] = st
            needer["awaiting_relay"] = False
            d["role"] = "patrol"
            d["rendezvous_id"] = None
            d["target_zone"] = None; d["target_cell"] = None
            d["pr"] = d["pc"] = None
        else:
            old_r, old_c = d["r"], d["c"]
            nr, nc = a20.move_toward(d["r"], d["c"], tr, tc)
            d["r"], d["c"] = nr, nc
            cost = a20.calculate_actual_step_cost(old_r, old_c, nr, nc, None, None)
            d["battery"] = max(0.0, d["battery"] - cost)
            a20.sensor_sweep(nr, nc, g, detected, step=step)

    # ---- 3. handle drones that are on PATROL ----
    for d in drones:
        if d["role"] != "patrol":
            continue

        if not d["awaiting_relay"]:
            nh, _st = a20.needs_handoff_now(d["r"], d["c"], d["battery"])
            if nh:
                chosen, _diag = relay_selector(d, drones, g, detected, step)
                if chosen is not None:
                    d["awaiting_relay"] = True
                    real = by_id[chosen["id"]]
                    real["role"] = "relay_incoming"
                    real["rendezvous_id"] = d["id"]

        mrc, e_ret, st2 = a20.must_recharge_now(d["r"], d["c"], d["battery"])
        if mrc:
            cost = a20.calculate_actual_step_cost(d["r"], d["c"], d["r"], d["c"], d["pr"], d["pc"])
            d["battery"] = max(0.0, d["battery"] - cost)
            a20.sensor_sweep(d["r"], d["c"], g, detected, step=step)
            continue

        if d["target_zone"]:
            zr, zc = d["target_zone"]
            if a20.get_zone_id(d["r"], d["c"]) == (zr, zc) and a20.find_uncovered_in_zone(zr, zc, g) is None:
                d["target_zone"] = None; d["target_cell"] = None
        if not d["target_zone"]:
            rk = a20.rank_zones(d["r"], d["c"], g, detected, step)
            d["target_zone"], d["target_cell"] = a20.select_zone_mixed_strategy(
                rk, g, d["r"], d["c"], d["battery"], d["zone_rng"], step)

        old_r, old_c = d["r"], d["c"]
        nr, nc, _bd = a20.smart_move(d["r"], d["c"], d["pr"], d["pc"], g, detected, d["target_zone"])
        d["r"], d["c"] = nr, nc
        cost = a20.calculate_actual_step_cost(old_r, old_c, nr, nc, old_r if d["pr"] is None else d["pr"],
                                               old_c if d["pc"] is None else d["pc"])
        d["battery"] = max(0.0, d["battery"] - cost)
        a20.sensor_sweep(nr, nc, g, detected, step=step)
        d["pr"], d["pc"] = old_r, old_c

        if d["target_zone"]:
            d["target_cell"] = a20.find_uncovered_in_zone(*d["target_zone"], g)

    state["step"] += 1
    return state


# ══════════════════════════════════════════════════════════════
# METRICS
# ══════════════════════════════════════════════════════════════
def collect_metrics(state):
    g, detected, drones = state["g"], state["detected"], state["drones"]
    active_drones = [d for d in drones]
    avg_staleness = statistics.mean(
        (state["step"] - c["last_visited_step"]) for c in g.values() if not c["is_station"]
    )
    return {
        "coverage_pct": a20.coverage_pct(g),
        "threats_detected": len(detected),
        "threats_total": a20.NUM_THREATS,
        "RC": state["RC"],
        "avg_final_battery": round(statistics.mean(d["battery"] for d in active_drones), 2),
        "avg_zone_staleness_steps": round(avg_staleness, 2),
        "final_drone_count": len(active_drones),
    }


def simulate_fleet(seed, relay_selector=select_best_relay, num_drones=NUM_DRONES, max_steps=None):
    max_steps = max_steps or a20.MAX_STEPS
    state = init_fleet(seed, num_drones)
    for _ in range(max_steps):
        step_fleet(state, relay_selector=relay_selector)
    return collect_metrics(state)


# ══════════════════════════════════════════════════════════════
# BATCH COMPARISON: baseline (nearest-only) vs optimized (RelayScore)
# ══════════════════════════════════════════════════════════════
def run_comparison(seeds, max_steps=None, out_csv=None):
    rows = []
    for seed in seeds:
        base = simulate_fleet(seed, relay_selector=select_nearest_relay, max_steps=max_steps)
        opt = simulate_fleet(seed, relay_selector=select_best_relay, max_steps=max_steps)
        row = {"seed": seed}
        for k, v in base.items():
            row[f"baseline_{k}"] = v
        for k, v in opt.items():
            row[f"optimized_{k}"] = v
        rows.append(row)
        print(f"[seed {seed}] baseline cov={base['coverage_pct']}% RC={base['RC']} "
              f"staleness={base['avg_zone_staleness_steps']}  |  "
              f"optimized cov={opt['coverage_pct']}% RC={opt['RC']} "
              f"staleness={opt['avg_zone_staleness_steps']}")

    if out_csv:
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    summary = {}
    for metric in ["coverage_pct", "RC", "avg_zone_staleness_steps", "avg_final_battery"]:
        base_vals = [r[f"baseline_{metric}"] for r in rows]
        opt_vals = [r[f"optimized_{metric}"] for r in rows]
        entry = {
            "baseline_mean": round(statistics.mean(base_vals), 3),
            "optimized_mean": round(statistics.mean(opt_vals), 3),
        }
        if len(seeds) >= 2 and any(b != o for b, o in zip(base_vals, opt_vals)):
            try:
                stat, p = wilcoxon(base_vals, opt_vals)
                entry["wilcoxon_p"] = round(p, 5)
            except ValueError as e:
                entry["wilcoxon_p"] = f"n/a ({e})"
        summary[metric] = entry

    return rows, summary

# ══════════════════════════════════════════════════════════════
# SENSITIVITY ANALYSIS
# ══════════════════════════════════════════════════════════════
def run_sensitivity_analysis(seeds, max_steps=None, out_csv="sensitivity_analysis_coverage.csv"):
    coverage_loss_values = [10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0]
    rows = []
    
    for val in coverage_loss_values:
        print(f"\n--- Running Sensitivity Analysis for W_COVERAGE_LOSS = {val} ---")
        
        test_weights = DEFAULT_WEIGHTS.copy()
        test_weights["W_COVERAGE_LOSS"] = val
        
        custom_selector = partial(select_best_relay, weights=test_weights)
        
        for seed in seeds:
            opt = simulate_fleet(seed, relay_selector=custom_selector, max_steps=max_steps)
            row = {
                "param_tested": "W_COVERAGE_LOSS",
                "param_value": val,
                "seed": seed,
                "coverage_pct": opt["coverage_pct"],
                "RC": opt["RC"],
                "avg_zone_staleness_steps": opt["avg_zone_staleness_steps"],
                "avg_final_battery": opt["avg_final_battery"]
            }
            rows.append(row)
            print(f"[seed {seed}] val={val} -> cov={opt['coverage_pct']}% | RC={opt['RC']} | staleness={opt['avg_zone_staleness_steps']}")

    if out_csv:
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
            
    print(f"\n✅ Sensitivity analysis saved to {out_csv}")
    return rows


if __name__ == "__main__":
    import sys
    seeds = [1, 2, 3, 4, 5]
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 300   # quick smoke-test default
    
    # Run the original batch comparison
    rows, summary = run_comparison(seeds, max_steps=steps, out_csv="fleet_relay_comparison.csv")
    print("\n=== SUMMARY (baseline nearest-only vs optimized RelayScore) ===")
    print(json.dumps(summary, indent=2))
    
    # Run the new Sensitivity Analysis
    run_sensitivity_analysis(seeds, max_steps=steps, out_csv="sensitivity_analysis_coverage.csv")
