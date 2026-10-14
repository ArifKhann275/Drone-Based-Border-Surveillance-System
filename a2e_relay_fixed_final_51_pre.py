# -*- coding: utf-8 -*-

import math
import random
import statistics
import csv
import json
import time  # NEW: wall-clock timing instrumentation for relay_selector calls
import tie_tracker
from copy import deepcopy
from functools import partial

# Plotting libraries for decision breakdown
try:
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")  # headless-safe: only ever savefig(), never show()
    import matplotlib.pyplot as plt
except ImportError:
    print("Warning: pandas or matplotlib not installed. Plotting will be skipped.")

import a20  # noqa: reuse the existing physics/grid/zone engine as-is

from scipy.stats import wilcoxon, ttest_rel, rankdata


# ══════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════
NUM_DRONES = 4                  # simultaneously-patrolling drones (fixed, matches a20.STATIONS)
RESERVE_POOL_SIZE = 4           # extra idle spare drones (one pre-positioned per station)
RELAY_SAFETY_FLOOR = 8.0       # % battery a candidate must keep in reserve after the trip
COVERAGE_STALENESS_CAP_STEPS = 200   # same normalization a20.zone_info() uses (tgap)
THREAT_ZONE_PENALTY = 2.0      # penalty multiplier if a threat zone is left unattended
PRIORITY_ZONE_WEIGHT = 1.0     # weight for border priority

# ══════════════════════════════════════════════════════════════
# RESERVE-POOL OPPORTUNITY COST
# ══════════════════════════════════════════════════════════════
RESERVE_OPPORTUNITY_ENABLED = True
RESERVE_OPPORTUNITY_LAMBDA = 0.030
RESERVE_OPPORTUNITY_GAMMA = 2.0
RESERVE_OPPORTUNITY_APPLY_TO_UNASSIGNED = False

# Utility-function weights
DEFAULT_WEIGHTS = {
    "W_TRAVEL_ENERGY": 1.0,      # travel energy from a20 OWT-A* model, in battery-% equivalent
    "W_COVERAGE_LOSS": 25.0,     # marginal surveillance-staleness cost, normalized to [0,1]
    "W_RELAY_DELAY": 0.4,        # small delay penalty, measured in simulation steps
}

FUZZY_ATTR_WEIGHTS = {
    "distance": 0.30,        
    "battery_margin": 0.30,  
    "coverage_loss": 0.25,   
    "relay_delay": 0.15,     
}

NORMALIZED_WEIGHTS = {
    "W_TRAVEL_ENERGY": 0.35,
    "W_COVERAGE_LOSS": 0.40,
    "W_RELAY_DELAY": 0.25,
}

NORMALIZED_WEIGHTS_G = {
    "W_TRAVEL_ENERGY": 0.30,
    "W_COVERAGE_LOSS": 0.30,
    "W_RELAY_DELAY": 0.20,
    "W_BATTERY_MARGIN": 0.20,
}

BATTERY_MARGIN_SATURATION_PCT = 25.0

NORMALIZED_WEIGHTS_H = dict(NORMALIZED_WEIGHTS_G) 


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
        "cycles": 0,                
        "target_zone": None, "target_cell": None,
        "zone_rng": random.Random(seed * 97 + idx * 31 + 17),
        "role": "patrol",          
        "rendezvous_id": None,     
        "awaiting_relay": False,   
        "return_target": None,    
        "relay_episode_counted": False,  
    }


def init_fleet(seed, num_drones=NUM_DRONES, num_reserves=RESERVE_POOL_SIZE):
    g = a20.make_grid(seed)
    a20.place_threats(seed, [g])
    drones = [_new_drone(i, seed) for i in range(num_drones)]
    for d in drones:
        d["role"] = "patrol"
        d["reserve_slot"] = False

    for j in range(num_reserves):
        idx = num_drones + j
        st = a20.STATIONS[j % len(a20.STATIONS)]
        rd = _new_drone(idx, seed)
        rd["r"], rd["c"] = st       
        rd["role"] = "idle_reserve"
        rd["reserve_slot"] = True      
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
        "first_all_threats_step": None,   
        "first_full_coverage_step": None, 
        "relay_requested": 0,   
        "relay_fulfilled": 0,   
        "total_energy_consumed": 0.0,   
        "stranded_events": 0,   
        "selector_time_total_s": 0.0,
        "selector_calls": 0,
        "selector_time_samples_s": [],
    }



# ══════════════════════════════════════════════════════════════
# RELAYSCORE — the 5-factor utility function
# ══════════════════════════════════════════════════════════════
def _manhattan(r1, c1, r2, c2):
    return abs(r1 - r2) + abs(c1 - c2)


def _safe_wilcoxon(x, y):
    pairs = [(int(a) if isinstance(a, bool) else a, int(b) if isinstance(b, bool) else b)
             for a, b in zip(x, y)
             if isinstance(a, (int, float, bool)) and isinstance(b, (int, float, bool))]
    if len(pairs) < 2:
        return {"mean_diff": None, "wilcoxon_p": None}
    diffs = [b - a for a, b in pairs]
    mean_diff = statistics.mean(diffs)
    if all(abs(d) < 1e-12 for d in diffs):
        return {"mean_diff": mean_diff, "wilcoxon_p": 1.0}
    try:
        _stat, p = wilcoxon([a for a, _ in pairs], [b for _, b in pairs])
        return {"mean_diff": mean_diff, "wilcoxon_p": round(float(p), 5)}
    except Exception as e:
        try:
            _stat2, p2 = ttest_rel([a for a, _ in pairs], [b for _, b in pairs])
            return {"mean_diff": mean_diff, "wilcoxon_p": round(float(p2), 5),
                     "test_used": "ttest_rel_fallback",
                     "wilcoxon_error": f"{type(e).__name__}: {e}"}
        except Exception as e2:
            return {"mean_diff": mean_diff, "wilcoxon_p": None,
                     "wilcoxon_error": f"{type(e).__name__}: {e}",
                     "ttest_error": f"{type(e2).__name__}: {e2}"}


def _rank_biserial_paired(x, y):
    pairs = [(int(a) if isinstance(a, bool) else a, int(b) if isinstance(b, bool) else b)
             for a, b in zip(x, y)
             if isinstance(a, (int, float, bool)) and isinstance(b, (int, float, bool))]
    diffs = [a - b for a, b in pairs if a != b]
    if not diffs:
        return None
    abs_diffs = [abs(d) for d in diffs]
    ranks = rankdata(abs_diffs)
    pos = sum(r for r, d in zip(ranks, diffs) if d > 0)
    neg = sum(r for r, d in zip(ranks, diffs) if d < 0)
    total = pos + neg
    if total == 0:
        return 0.0
    return round(float((pos - neg) / total), 4)


def _feasibility(candidate, needer):
    travel_energy = a20.energy_to_travel(candidate["r"], candidate["c"],
                                          needer["r"], needer["c"])
    battery_margin = candidate["battery"] - travel_energy - RELAY_SAFETY_FLOOR
    return travel_energy, battery_margin, battery_margin >= 0


def _candidate_pool(needer, drones):
    if needer is None or drones is None:
        return []

    pool = []

    for d in drones:
        if not isinstance(d, dict):
            continue
        if "id" not in d or "role" not in d:
            continue
        if d["id"] == needer.get("id"):
            continue
        if d["role"] not in ("patrol", "idle_reserve"):
            continue
        if d.get("awaiting_relay", False):
            continue

        pool.append(d)
    return pool


def _feasible_candidate_pool(needer, drones):
    if needer is None or drones is None:
        return []

    pool = []
    for d in _candidate_pool(needer, drones):
        try:
            travel_energy, battery_margin, feasible = _feasibility(d, needer)
        except (KeyError, TypeError, ValueError):
            continue

        if feasible:
            pool.append((d, travel_energy, battery_margin))
    return pool


def _reserve_pool_state(fleet):
    if not fleet:
        return 0, 0
    available = 0
    capacity = 0
    tagged = False
    for d in fleet:
        if d.get("reserve_slot") is not None:
            tagged = True
            if d.get("reserve_slot"):
                capacity += 1
        if d.get("role") == "idle_reserve" and not d.get("awaiting_relay", False):
            available += 1
    if not tagged or capacity == 0:
        capacity = max(available, 1)
    return available, capacity


def _reserve_opportunity_cost(candidate, fleet,
                              lam=None, gamma=None):
    if not RESERVE_OPPORTUNITY_ENABLED or not fleet:
        return 0.0

    role = candidate.get("role")
    draws_from_pool = (role == "idle_reserve")
    if not draws_from_pool and RESERVE_OPPORTUNITY_APPLY_TO_UNASSIGNED:
        draws_from_pool = candidate.get("target_zone") is None
    if not draws_from_pool:
        return 0.0

    available, capacity = _reserve_pool_state(fleet)
    if capacity <= 0:
        return 0.0

    remaining_after = max(available - 1, 0)
    scarcity = 1.0 - (remaining_after / float(capacity))
    scarcity = min(max(scarcity, 0.0), 1.0)

    lam = RESERVE_OPPORTUNITY_LAMBDA if lam is None else lam
    gamma = RESERVE_OPPORTUNITY_GAMMA if gamma is None else gamma
    return float(lam) * (scarcity ** float(gamma))


def _coverage_loss_for_candidate(candidate, needer, g, detected, step, relay_delay,
                                 fleet=None):
    reserve_cost = _reserve_opportunity_cost(candidate, fleet)

    if candidate["role"] == "idle_reserve":
        return reserve_cost, 0.0, 0.0, ({"reserve_opportunity_cost": round(reserve_cost, 6)}
                                        if reserve_cost else None)

    if candidate["target_zone"] is None:
        return reserve_cost, 0.0, 0.0, ({"reserve_opportunity_cost": round(reserve_cost, 6)}
                                        if reserve_cost else None)

    info = a20.zone_info(
        candidate["target_zone"][0],
        candidate["target_zone"][1],
        0, 0, g, detected, step
    )
    if not info:
        return 0.0, 0.0, 0.0, None

    cap = max(float(COVERAGE_STALENESS_CAP_STEPS), 1.0)
    current_tgap = min(info["raw_gap"] / cap, 1.0)
    projected_tgap = min((info["raw_gap"] + relay_delay) / cap, 1.0)
    marginal_loss = max(0.0, projected_tgap - current_tgap)

    has_threat = bool(info.get("has_threat", False))
    border_pr = float(info.get("border_pr", 0.0))
    penalty_mult = 1.0
    if has_threat:
        penalty_mult += THREAT_ZONE_PENALTY
    penalty_mult += PRIORITY_ZONE_WEIGHT * border_pr

    weighted_loss = marginal_loss * penalty_mult + reserve_cost
    return weighted_loss, marginal_loss, current_tgap, {
        "has_threat": has_threat,
        "border_priority": border_pr,
        "penalty_multiplier": round(penalty_mult, 3),
        "current_tgap": round(current_tgap, 3),
        "projected_tgap": round(projected_tgap, 3),
        "reserve_opportunity_cost": round(reserve_cost, 6),
    }


def relay_score(candidate, needer, g, detected, step, weights=None, fleet=None):
    if weights is None:
        weights = DEFAULT_WEIGHTS

    dist = _manhattan(candidate["r"], candidate["c"], needer["r"], needer["c"])
    travel_energy, battery_margin, feasible = _feasibility(candidate, needer)

    if not feasible:
        return None, {
            "candidate_id": candidate["id"],
            "reason": "insufficient_battery",
            "battery_margin": round(battery_margin, 2),
            "score": None,
        }

    relay_delay = dist

    coverage_loss, raw_marginal_loss, current_tgap, cov_meta = _coverage_loss_for_candidate(
        candidate, needer, g, detected, step, relay_delay, fleet=fleet
    )

    energy_cost = weights["W_TRAVEL_ENERGY"] * travel_energy
    coverage_cost = weights["W_COVERAGE_LOSS"] * coverage_loss
    delay_cost = weights["W_RELAY_DELAY"] * relay_delay
    total_cost = energy_cost + coverage_cost + delay_cost

    score = -total_cost

    diag = {
        "candidate_id": candidate["id"],
        "role": candidate["role"],
        "distance": dist,
        "travel_energy": round(travel_energy, 3),
        "battery_margin": round(battery_margin, 2),  
        "battery_feasible": True,
        "coverage_loss": round(coverage_loss, 6),
        "raw_marginal_coverage_loss": round(raw_marginal_loss, 6),
        "current_tgap": round(current_tgap, 6),
        "relay_delay": relay_delay,
        "energy_cost": round(energy_cost, 3),
        "coverage_cost": round(coverage_cost, 3),
        "delay_cost": round(delay_cost, 3),
        "score": round(score, 3),
        "coverage_metadata": cov_meta,
    }
    return score, diag


def _relay_utility_pool(feasible, needer, g, detected, step, weights=None, fleet=None):
    if weights is None:
        weights = NORMALIZED_WEIGHTS

    rows = []
    for cand, travel_energy, battery_margin in feasible:
        dist = _manhattan(cand["r"], cand["c"], needer["r"], needer["c"])
        relay_delay = dist
        coverage_loss, raw_marginal_loss, current_tgap, cov_meta = _coverage_loss_for_candidate(
            cand, needer, g, detected, step, relay_delay, fleet=fleet
        )
        rows.append({
            "cand": cand, "distance": dist, "travel_energy": travel_energy,
            "battery_margin": battery_margin, "coverage_loss": coverage_loss,
            "raw_marginal_loss": raw_marginal_loss, "current_tgap": current_tgap,
            "cov_meta": cov_meta, "relay_delay": relay_delay,
        })

    if len(rows) == 1:
        r = rows[0]
        diag = {
            "candidate_id": r["cand"]["id"], "role": r["cand"]["role"],
            "distance": r["distance"],
            "travel_energy": round(r["travel_energy"], 3),
            "battery_margin": round(r["battery_margin"], 2),
            "coverage_loss": round(r["coverage_loss"], 6),
            "raw_marginal_coverage_loss": round(r["raw_marginal_loss"], 6),
            "current_tgap": round(r["current_tgap"], 6),
            "relay_delay": r["relay_delay"],
            "membership_energy": 1.0, "membership_coverage": 1.0, "membership_delay": 1.0,
            "score": 1.0,
            "note": "only feasible candidate -- no ranking needed",
            "coverage_metadata": r["cov_meta"],
        }
        return [(r["cand"], 1.0, diag)]

    energy_r = _fuzzy_normalize([r["travel_energy"] for r in rows], higher_is_better=False)
    cov_r = _fuzzy_normalize([r["coverage_loss"] for r in rows], higher_is_better=False)
    delay_r = _fuzzy_normalize([r["relay_delay"] for r in rows], higher_is_better=False)

    w = weights
    scored = []
    for i, r in enumerate(rows):
        utility = (w["W_TRAVEL_ENERGY"] * energy_r[i]
                   + w["W_COVERAGE_LOSS"] * cov_r[i]
                   + w["W_RELAY_DELAY"] * delay_r[i])

        diag = {
            "candidate_id": r["cand"]["id"], "role": r["cand"]["role"],
            "distance": r["distance"],
            "travel_energy": round(r["travel_energy"], 3),
            "battery_margin": round(r["battery_margin"], 2),
            "coverage_loss": round(r["coverage_loss"], 6),
            "raw_marginal_coverage_loss": round(r["raw_marginal_loss"], 6),
            "current_tgap": round(r["current_tgap"], 6),
            "relay_delay": r["relay_delay"],
            "membership_energy": round(energy_r[i], 3),
            "membership_coverage": round(cov_r[i], 3),
            "membership_delay": round(delay_r[i], 3),
            "score": round(utility, 4),
            "coverage_metadata": r["cov_meta"],
        }
        scored.append((r["cand"], utility, diag))

    scored.sort(key=lambda x: -x[1])
    
    # === NEW: MECHANISM LOGGING (Policy C / RelayScore) ===
    try:
        import csv
        import os
        log_file = "mechanism_log_C.csv"
        file_exists = os.path.isfile(log_file)
        with open(log_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["step", "needer_id", "pool_size", "candidate_id", "role", 
                                 "distance", "raw_coverage_loss", "membership_coverage", 
                                 "final_score", "is_winner"])
            if scored:
                winner_id = scored[0][0]["id"]
                for _cand, _util, _diag in scored:
                    writer.writerow([
                        step, needer.get("id", -1), len(scored), _diag.get("candidate_id", -1), 
                        _diag.get("role", ""), _diag.get("distance", 0),
                        _diag.get("raw_marginal_coverage_loss", 0.0), 
                        _diag.get("membership_coverage", 0.0), _diag.get("score", 0.0), 
                        1 if _diag.get("candidate_id") == winner_id else 0
                    ])
    except Exception as e:
        pass
    # ========================================================
    
    return scored


def select_best_relay(needer, drones, g, detected, step, weights=None):
    feasible = _feasible_candidate_pool(needer, drones)
    if not feasible:
        return None, []

    scored = _relay_utility_pool(feasible, needer, g, detected, step, weights, fleet=drones)
    best_cand, _best_score, _best_diag = scored[0]
    return best_cand, [diag for _, _, diag in scored]


# ══════════════════════════════════════════════════════════════
# POLICY G — BATTERY-MARGIN-AWARE RELAYSCORE
# ══════════════════════════════════════════════════════════════
def _relay_utility_pool_g(feasible, needer, g, detected, step, weights=None, fleet=None):
    if weights is None:
        weights = NORMALIZED_WEIGHTS_G

    rows = []
    for cand, travel_energy, battery_margin in feasible:
        dist = _manhattan(cand["r"], cand["c"], needer["r"], needer["c"])
        relay_delay = dist
        coverage_loss, raw_marginal_loss, current_tgap, cov_meta = _coverage_loss_for_candidate(
            cand, needer, g, detected, step, relay_delay, fleet=fleet
        )
        rows.append({
            "cand": cand, "distance": dist, "travel_energy": travel_energy,
            "battery_margin": battery_margin, "coverage_loss": coverage_loss,
            "raw_marginal_loss": raw_marginal_loss, "current_tgap": current_tgap,
            "cov_meta": cov_meta, "relay_delay": relay_delay,
        })

    if len(rows) == 1:
        r = rows[0]
        diag = {
            "candidate_id": r["cand"]["id"], "role": r["cand"]["role"],
            "distance": r["distance"],
            "travel_energy": round(r["travel_energy"], 3),
            "battery_margin": round(r["battery_margin"], 2),
            "coverage_loss": round(r["coverage_loss"], 6),
            "raw_marginal_coverage_loss": round(r["raw_marginal_loss"], 6),
            "current_tgap": round(r["current_tgap"], 6),
            "relay_delay": r["relay_delay"],
            "membership_energy": 1.0, "membership_coverage": 1.0,
            "membership_delay": 1.0, "membership_battery": 1.0,
            "score": 1.0,
            "note": "only feasible candidate -- no ranking needed",
            "coverage_metadata": r["cov_meta"],
        }
        return [(r["cand"], 1.0, diag)]

    energy_r = _fuzzy_normalize([r["travel_energy"] for r in rows], higher_is_better=False)
    cov_r = _fuzzy_normalize([r["coverage_loss"] for r in rows], higher_is_better=False)
    delay_r = _fuzzy_normalize([r["relay_delay"] for r in rows], higher_is_better=False)
    batt_r = _fuzzy_normalize([r["battery_margin"] for r in rows], higher_is_better=True)

    w = weights
    scored = []
    for i, r in enumerate(rows):
        utility = (w["W_TRAVEL_ENERGY"] * energy_r[i]
                   + w["W_COVERAGE_LOSS"] * cov_r[i]
                   + w["W_RELAY_DELAY"] * delay_r[i]
                   + w["W_BATTERY_MARGIN"] * batt_r[i])

        diag = {
            "candidate_id": r["cand"]["id"], "role": r["cand"]["role"],
            "distance": r["distance"],
            "travel_energy": round(r["travel_energy"], 3),
            "battery_margin": round(r["battery_margin"], 2),
            "coverage_loss": round(r["coverage_loss"], 6),
            "raw_marginal_coverage_loss": round(r["raw_marginal_loss"], 6),
            "current_tgap": round(r["current_tgap"], 6),
            "relay_delay": r["relay_delay"],
            "membership_energy": round(energy_r[i], 3),
            "membership_coverage": round(cov_r[i], 3),
            "membership_delay": round(delay_r[i], 3),
            "membership_battery": round(batt_r[i], 3),
            "score": round(utility, 4),
            "coverage_metadata": r["cov_meta"],
        }
        scored.append((r["cand"], utility, diag))

    scored.sort(key=lambda x: -x[1])
    return scored


def select_best_relay_battery_aware(needer, drones, g, detected, step, weights=None):
    feasible = _feasible_candidate_pool(needer, drones)
    if not feasible:
        return None, []

    scored = _relay_utility_pool_g(feasible, needer, g, detected, step, weights, fleet=drones)
    best_cand, _best_score, _best_diag = scored[0]
    return best_cand, [diag for _, _, diag in scored]


# ══════════════════════════════════════════════════════════════
# POLICY H — BATTERY-MARGIN-AWARE RELAYSCORE, SATURATION-CAPPED
# ══════════════════════════════════════════════════════════════
def _relay_utility_pool_h(feasible, needer, g, detected, step, weights=None, fleet=None):
    if weights is None:
        weights = NORMALIZED_WEIGHTS_H

    rows = []
    for cand, travel_energy, battery_margin in feasible:
        dist = _manhattan(cand["r"], cand["c"], needer["r"], needer["c"])
        relay_delay = dist
        coverage_loss, raw_marginal_loss, current_tgap, cov_meta = _coverage_loss_for_candidate(
            cand, needer, g, detected, step, relay_delay, fleet=fleet
        )
        rows.append({
            "cand": cand, "distance": dist, "travel_energy": travel_energy,
            "battery_margin": battery_margin, "coverage_loss": coverage_loss,
            "raw_marginal_loss": raw_marginal_loss, "current_tgap": current_tgap,
            "cov_meta": cov_meta, "relay_delay": relay_delay,
        })

    if len(rows) == 1:
        r = rows[0]
        diag = {
            "candidate_id": r["cand"]["id"], "role": r["cand"]["role"],
            "distance": r["distance"],
            "travel_energy": round(r["travel_energy"], 3),
            "battery_margin": round(r["battery_margin"], 2),
            "coverage_loss": round(r["coverage_loss"], 6),
            "raw_marginal_coverage_loss": round(r["raw_marginal_loss"], 6),
            "current_tgap": round(r["current_tgap"], 6),
            "relay_delay": r["relay_delay"],
            "membership_energy": 1.0, "membership_coverage": 1.0,
            "membership_delay": 1.0, "membership_battery": 1.0,
            "score": 1.0,
            "note": "only feasible candidate -- no ranking needed",
            "coverage_metadata": r["cov_meta"],
        }
        return [(r["cand"], 1.0, diag)]

    energy_r = _fuzzy_normalize([r["travel_energy"] for r in rows], higher_is_better=False)
    cov_r = _fuzzy_normalize([r["coverage_loss"] for r in rows], higher_is_better=False)
    delay_r = _fuzzy_normalize([r["relay_delay"] for r in rows], higher_is_better=False)

    batt_span = max(BATTERY_MARGIN_SATURATION_PCT - RELAY_SAFETY_FLOOR, 1e-9)
    batt_r = [
        max(0.0, min(1.0, (r["battery_margin"] - RELAY_SAFETY_FLOOR) / batt_span))
        for r in rows
    ]

    w = weights
    scored = []
    for i, r in enumerate(rows):
        utility = (w["W_TRAVEL_ENERGY"] * energy_r[i]
                   + w["W_COVERAGE_LOSS"] * cov_r[i]
                   + w["W_RELAY_DELAY"] * delay_r[i]
                   + w["W_BATTERY_MARGIN"] * batt_r[i])

        diag = {
            "candidate_id": r["cand"]["id"], "role": r["cand"]["role"],
            "distance": r["distance"],
            "travel_energy": round(r["travel_energy"], 3),
            "battery_margin": round(r["battery_margin"], 2),
            "battery_margin_capped": round(min(r["battery_margin"], BATTERY_MARGIN_SATURATION_PCT), 2),
            "coverage_loss": round(r["coverage_loss"], 6),
            "raw_marginal_coverage_loss": round(r["raw_marginal_loss"], 6),
            "current_tgap": round(r["current_tgap"], 6),
            "relay_delay": r["relay_delay"],
            "membership_energy": round(energy_r[i], 3),
            "membership_coverage": round(cov_r[i], 3),
            "membership_delay": round(delay_r[i], 3),
            "membership_battery": round(batt_r[i], 3),
            "score": round(utility, 4),
            "coverage_metadata": r["cov_meta"],
        }
        scored.append((r["cand"], utility, diag))

    scored.sort(key=lambda x: -x[1])
    return scored


def select_best_relay_battery_aware_capped(needer, drones, g, detected, step, weights=None):
    feasible = _feasible_candidate_pool(needer, drones)
    if not feasible:
        return None, []

    scored = _relay_utility_pool_h(feasible, needer, g, detected, step, weights, fleet=drones)
    best_cand, _best_score, _best_diag = scored[0]
    return best_cand, [diag for _, _, diag in scored]


# ══════════════════════════════════════════════════════════════
# POLICY I — COVERAGE-MARGIN-AWARE RELAYSCORE (LARS)
# ══════════════════════════════════════════════════════════════
COVERAGE_LOSS_SATURATION_CEILING = 0.6
NORMALIZED_WEIGHTS_I = dict(NORMALIZED_WEIGHTS)  


def _relay_utility_pool_i(feasible, needer, g, detected, step, weights=None, fleet=None):
    if weights is None:
        weights = NORMALIZED_WEIGHTS_I

    rows = []
    for cand, travel_energy, battery_margin in feasible:
        dist = _manhattan(cand["r"], cand["c"], needer["r"], needer["c"])
        relay_delay = dist
        coverage_loss, raw_marginal_loss, current_tgap, cov_meta = _coverage_loss_for_candidate(
            cand, needer, g, detected, step, relay_delay, fleet=fleet
        )
        rows.append({
            "cand": cand, "distance": dist, "travel_energy": travel_energy,
            "battery_margin": battery_margin, "coverage_loss": coverage_loss,
            "raw_marginal_loss": raw_marginal_loss, "current_tgap": current_tgap,
            "cov_meta": cov_meta, "relay_delay": relay_delay,
        })

    if len(rows) == 1:
        r = rows[0]
        diag = {
            "candidate_id": r["cand"]["id"], "role": r["cand"]["role"],
            "distance": r["distance"],
            "travel_energy": round(r["travel_energy"], 3),
            "battery_margin": round(r["battery_margin"], 2),
            "coverage_loss": round(r["coverage_loss"], 6),
            "raw_marginal_coverage_loss": round(r["raw_marginal_loss"], 6),
            "current_tgap": round(r["current_tgap"], 6),
            "relay_delay": r["relay_delay"],
            "membership_energy": 1.0, "membership_coverage": 1.0, "membership_delay": 1.0,
            "score": 1.0,
            "note": "only feasible candidate -- no ranking needed",
            "coverage_metadata": r["cov_meta"],
        }
        return [(r["cand"], 1.0, diag)]

    energy_r = _fuzzy_normalize([r["travel_energy"] for r in rows], higher_is_better=False)
    delay_r = _fuzzy_normalize([r["relay_delay"] for r in rows], higher_is_better=False)
    
    cov_span = max(COVERAGE_LOSS_SATURATION_CEILING, 1e-9)
    cov_r = [
        max(0.0, min(1.0, 1.0 - (r["coverage_loss"] / cov_span)))
        for r in rows
    ]

    w = weights
    scored = []
    for i, r in enumerate(rows):
        utility = (w["W_TRAVEL_ENERGY"] * energy_r[i]
                   + w["W_COVERAGE_LOSS"] * cov_r[i]
                   + w["W_RELAY_DELAY"] * delay_r[i])

        diag = {
            "candidate_id": r["cand"]["id"], "role": r["cand"]["role"],
            "distance": r["distance"],
            "travel_energy": round(r["travel_energy"], 3),
            "battery_margin": round(r["battery_margin"], 2),
            "coverage_loss": round(r["coverage_loss"], 6),
            "coverage_loss_capped": round(min(r["coverage_loss"], COVERAGE_LOSS_SATURATION_CEILING), 6),
            "raw_marginal_coverage_loss": round(r["raw_marginal_loss"], 6),
            "current_tgap": round(r["current_tgap"], 6),
            "relay_delay": r["relay_delay"],
            "membership_energy": round(energy_r[i], 3),
            "membership_coverage": round(cov_r[i], 3),
            "membership_delay": round(delay_r[i], 3),
            "score": round(utility, 4),
            "coverage_metadata": r["cov_meta"],
        }
        scored.append((r["cand"], utility, diag))

    # ==========================================
    # NEW: Log ties for Policy I before sorting
    # ==========================================
    tie_tracker.log_ties("I", scored, higher_is_better=True)

    scored.sort(key=lambda x: -x[1])
    
    # === NEW: MECHANISM LOGGING (Policy LARS / I) ===
    try:
        import csv
        import os
        log_file = "mechanism_log_LARS.csv"
        file_exists = os.path.isfile(log_file)
        with open(log_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["step", "needer_id", "pool_size", "candidate_id", "role", 
                                 "distance", "raw_coverage_loss", "membership_coverage", 
                                 "final_score", "is_winner"])
            if scored:
                winner_id = scored[0][0]["id"]
                for _cand, _util, _diag in scored:
                    writer.writerow([
                        step, needer.get("id", -1), len(scored), _diag.get("candidate_id", -1), 
                        _diag.get("role", ""), _diag.get("distance", 0),
                        _diag.get("raw_marginal_coverage_loss", 0.0), 
                        _diag.get("membership_coverage", 0.0), _diag.get("score", 0.0), 
                        1 if _diag.get("candidate_id") == winner_id else 0
                    ])
    except Exception as e:
        pass
    # ========================================================
    
    return scored


def select_best_relay_coverage_aware(needer, drones, g, detected, step, weights=None):
    feasible = _feasible_candidate_pool(needer, drones)
    if not feasible:
        return None, []

    scored = _relay_utility_pool_i(feasible, needer, g, detected, step, weights, fleet=drones)
    best_cand, _best_score, _best_diag = scored[0]
    return best_cand, [diag for _, _, diag in scored]


# ══════════════════════════════════════════════════════════════
# POLICY E — DYNAMIC-WEIGHT RELAYSCORE
# ══════════════════════════════════════════════════════════════
COVERAGE_PRESSURE_THRESHOLD = 0.5   
COVERAGE_PRESSURE_MAX_BOOST = 2.0    
BATTERY_CRITICAL_MARGIN = 5.0        
BATTERY_CRITICAL_MAX_BOOST = 2.5     


def _dynamic_weights(needer, g, step, base_weights=None,
                      coverage_pressure_threshold=COVERAGE_PRESSURE_THRESHOLD,
                      coverage_pressure_max_boost=COVERAGE_PRESSURE_MAX_BOOST,
                      battery_critical_margin=BATTERY_CRITICAL_MARGIN,
                      battery_critical_max_boost=BATTERY_CRITICAL_MAX_BOOST):
    w = dict(base_weights or NORMALIZED_WEIGHTS)

    staleness_values = [step - c["last_visited_step"] for c in g.values()
                         if not c.get("is_station")]
    if staleness_values:
        cap = max(float(COVERAGE_STALENESS_CAP_STEPS), 1.0)
        coverage_pressure = min(statistics.mean(staleness_values) / cap, 1.0)
    else:
        coverage_pressure = 0.0

    if coverage_pressure > coverage_pressure_threshold:
        span = max(1.0 - coverage_pressure_threshold, 1e-9)
        ramp = (coverage_pressure - coverage_pressure_threshold) / span
        boost = 1.0 + (coverage_pressure_max_boost - 1.0) * ramp
        w["W_COVERAGE_LOSS"] *= boost

    battery_headroom = needer.get("battery", 100.0) - RELAY_SAFETY_FLOOR
    if battery_headroom < battery_critical_margin:
        deficit = max(0.0, battery_critical_margin - battery_headroom)
        ramp = min(deficit / battery_critical_margin, 1.0)
        boost = 1.0 + (battery_critical_max_boost - 1.0) * ramp
        w["W_TRAVEL_ENERGY"] *= boost

    total = sum(w.values())
    if total > 0:
        w = {k: v / total for k, v in w.items()}
    return w


def select_best_relay_dynamic(needer, drones, g, detected, step, weights=None,
                               coverage_pressure_threshold=COVERAGE_PRESSURE_THRESHOLD,
                               coverage_pressure_max_boost=COVERAGE_PRESSURE_MAX_BOOST,
                               battery_critical_margin=BATTERY_CRITICAL_MARGIN,
                               battery_critical_max_boost=BATTERY_CRITICAL_MAX_BOOST):
    feasible = _feasible_candidate_pool(needer, drones)
    if not feasible:
        return None, []

    dyn_weights = _dynamic_weights(
        needer, g, step, base_weights=weights,
        coverage_pressure_threshold=coverage_pressure_threshold,
        coverage_pressure_max_boost=coverage_pressure_max_boost,
        battery_critical_margin=battery_critical_margin,
        battery_critical_max_boost=battery_critical_max_boost,
    )
    scored = _relay_utility_pool(feasible, needer, g, detected, step, dyn_weights, fleet=drones)
    best_cand, _best_score, _best_diag = scored[0]
    diags = [diag for _, _, diag in scored]
    for diag in diags:
        diag["dynamic_weights_used"] = {k: round(v, 3) for k, v in dyn_weights.items()}
    return best_cand, diags


# ══════════════════════════════════════════════════════════════
# POLICY F — PREDICTIVE RELAYSCORE
# ══════════════════════════════════════════════════════════════
PREDICTIVE_COVERAGE_HORIZON = 30   


def _predicted_travel_energy(candidate, needer, step):
    dist = _manhattan(candidate["r"], candidate["c"], needer["r"], needer["c"])
    saved_step = a20._SIM_CLOCK["step"]
    try:
        a20._SIM_CLOCK["step"] = step + dist
        energy = a20.energy_to_travel(candidate["r"], candidate["c"], needer["r"], needer["c"])
    finally:
        a20._SIM_CLOCK["step"] = saved_step   
    return energy


def _predicted_coverage_loss_for_candidate(candidate, needer, g, detected, step, relay_delay,
                                            fleet=None,
                                             horizon=PREDICTIVE_COVERAGE_HORIZON):
    reserve_cost = _reserve_opportunity_cost(candidate, fleet)
    if candidate["role"] == "idle_reserve" or candidate["target_zone"] is None:
        if reserve_cost:
            return reserve_cost, 0.0, 0.0, {"reserve_opportunity_cost": round(reserve_cost, 6)}
        return 0.0, 0.0, 0.0, None

    info = a20.zone_info(candidate["target_zone"][0], candidate["target_zone"][1],
                          0, 0, g, detected, step)
    if not info:
        return 0.0, 0.0, 0.0, None

    cap = max(float(COVERAGE_STALENESS_CAP_STEPS), 1.0)
    current_tgap = min(info["raw_gap"] / cap, 1.0)
    projected_tgap = min((info["raw_gap"] + relay_delay + horizon) / cap, 1.0)
    marginal_loss = max(0.0, projected_tgap - current_tgap)

    has_threat = bool(info.get("has_threat", False))
    border_pr = float(info.get("border_pr", 0.0))
    penalty_mult = 1.0
    if has_threat:
        penalty_mult += THREAT_ZONE_PENALTY
    penalty_mult += PRIORITY_ZONE_WEIGHT * border_pr

    weighted_loss = marginal_loss * penalty_mult
    return weighted_loss, marginal_loss, current_tgap, {
        "has_threat": has_threat, "border_priority": border_pr,
        "penalty_multiplier": round(penalty_mult, 3),
        "current_tgap": round(current_tgap, 3),
        "projected_tgap": round(projected_tgap, 3),
        "predictive_horizon": horizon,
    }


def _relay_utility_pool_predictive(feasible, needer, g, detected, step, weights=None,
                                    fleet=None,
                                     horizon=PREDICTIVE_COVERAGE_HORIZON):
    if weights is None:
        weights = NORMALIZED_WEIGHTS

    rows = []
    for cand, _travel_energy_current, battery_margin in feasible:
        dist = _manhattan(cand["r"], cand["c"], needer["r"], needer["c"])
        relay_delay = dist
        pred_energy = _predicted_travel_energy(cand, needer, step)
        coverage_loss, raw_marginal_loss, current_tgap, cov_meta = _predicted_coverage_loss_for_candidate(
            cand, needer, g, detected, step, relay_delay, fleet=fleet, horizon=horizon
        )
        rows.append({
            "cand": cand, "distance": dist, "travel_energy": pred_energy,
            "battery_margin": battery_margin, "coverage_loss": coverage_loss,
            "raw_marginal_loss": raw_marginal_loss, "current_tgap": current_tgap,
            "cov_meta": cov_meta, "relay_delay": relay_delay,
        })

    if len(rows) == 1:
        r = rows[0]
        diag = {
            "candidate_id": r["cand"]["id"], "role": r["cand"]["role"],
            "distance": r["distance"], "travel_energy": round(r["travel_energy"], 3),
            "battery_margin": round(r["battery_margin"], 2),
            "coverage_loss": round(r["coverage_loss"], 6),
            "relay_delay": r["relay_delay"],
            "membership_energy": 1.0, "membership_coverage": 1.0, "membership_delay": 1.0,
            "score": 1.0, "note": "only feasible candidate -- no ranking needed",
            "coverage_metadata": r["cov_meta"],
        }
        return [(r["cand"], 1.0, diag)]

    energy_r = _fuzzy_normalize([r["travel_energy"] for r in rows], higher_is_better=False)
    cov_r = _fuzzy_normalize([r["coverage_loss"] for r in rows], higher_is_better=False)
    delay_r = _fuzzy_normalize([r["relay_delay"] for r in rows], higher_is_better=False)

    w = weights
    scored = []
    for i, r in enumerate(rows):
        utility = (w["W_TRAVEL_ENERGY"] * energy_r[i]
                   + w["W_COVERAGE_LOSS"] * cov_r[i]
                   + w["W_RELAY_DELAY"] * delay_r[i])
        diag = {
            "candidate_id": r["cand"]["id"], "role": r["cand"]["role"],
            "distance": r["distance"], "travel_energy": round(r["travel_energy"], 3),
            "battery_margin": round(r["battery_margin"], 2),
            "coverage_loss": round(r["coverage_loss"], 6),
            "relay_delay": r["relay_delay"],
            "membership_energy": round(energy_r[i], 3),
            "membership_coverage": round(cov_r[i], 3),
            "membership_delay": round(delay_r[i], 3),
            "score": round(utility, 4),
            "coverage_metadata": r["cov_meta"],
        }
        scored.append((r["cand"], utility, diag))

    scored.sort(key=lambda x: -x[1])
    return scored


def select_best_relay_predictive(needer, drones, g, detected, step, weights=None,
                                   horizon=PREDICTIVE_COVERAGE_HORIZON):
    feasible = _feasible_candidate_pool(needer, drones)
    if not feasible:
        return None, []

    scored = _relay_utility_pool_predictive(feasible, needer, g, detected, step, weights, fleet=drones, horizon=horizon)
    best_cand, _best_score, _best_diag = scored[0]
    return best_cand, [diag for _, _, diag in scored]


def tune_predictive_horizon(train_seeds, horizon_grid, max_steps=None, metric="coverage_pct"):
    max_steps = max_steps or a20.MAX_STEPS
    results = []
    for h in horizon_grid:
        selector = lambda needer, drones, g, detected, step, weights=None, _h=h: \
            select_best_relay_predictive(needer, drones, g, detected, step, weights=weights, horizon=_h)
        vals = []
        for seed in train_seeds:
            m = simulate_fleet(seed, relay_selector=selector, max_steps=max_steps)
            vals.append(m.get(metric))
        clean = [v for v in vals if isinstance(v, (int, float))]
        mean_v = round(statistics.mean(clean), 4) if clean else None
        results.append({"horizon": h, f"mean_{metric}": mean_v})
        print(f"  [PREDICTIVE_COVERAGE_HORIZON={h}] mean_{metric}={mean_v}")
    results.sort(key=lambda r: (r[f"mean_{metric}"] is None, -(r[f"mean_{metric}"] or 0)))
    return results


# ══════════════════════════════════════════════════════════════
# BASELINE (generalized nearest-only)
# ══════════════════════════════════════════════════════════════
def select_nearest_relay(needer, drones, g, detected, step, weights=None):
    feasible = _feasible_candidate_pool(needer, drones)
    if not feasible:
        return None, []

    best = min(
        feasible,
        key=lambda x: _manhattan(
            x[0]["r"], x[0]["c"], needer["r"], needer["c"]
        )
    )[0]

    _score, best_diag = relay_score(
        best, needer, g, detected, step, weights=weights, fleet=drones
    )
    return best, ([best_diag] if best_diag is not None else [])


# ══════════════════════════════════════════════════════════════
# POLICY D — FUZZY MULTI-ATTRIBUTE RELAY SELECTION
# ══════════════════════════════════════════════════════════════
def _fuzzy_normalize(values, higher_is_better):
    lo, hi = min(values), max(values)
    if hi - lo < 1e-12:
        return [1.0] * len(values)
    if higher_is_better:
        return [(v - lo) / (hi - lo) for v in values]
    return [(hi - v) / (hi - lo) for v in values]


def select_fuzzy_relay(needer, drones, g, detected, step, weights=None, p=2):
    if weights is None:
        weights = FUZZY_ATTR_WEIGHTS

    feasible = _feasible_candidate_pool(needer, drones)
    if not feasible:
        return None, []

    rows = []
    for cand, travel_energy, battery_margin in feasible:
        dist = _manhattan(cand["r"], cand["c"], needer["r"], needer["c"])
        relay_delay = dist
        coverage_loss, _raw, _tgap, _meta = _coverage_loss_for_candidate(
            cand, needer, g, detected, step, relay_delay, fleet=drones
        )
        rows.append({
            "cand": cand,
            "distance": dist,
            "travel_energy": travel_energy, 
            "battery_margin": battery_margin,
            "coverage_loss": coverage_loss,
            "relay_delay": relay_delay,
        })

    if len(rows) == 1:
        only = rows[0]
        diag = {
            "candidate_id": only["cand"]["id"],
            "distance": only["distance"],
            "travel_energy": round(only["travel_energy"], 3), 
            "battery_margin": round(only["battery_margin"], 2),
            "coverage_loss": round(only["coverage_loss"], 6),
            "relay_delay": only["relay_delay"],
            "fuzzy_closeness": 1.0,
            "note": "only feasible candidate -- no ranking needed",
        }
        return only["cand"], [diag]

    dist_r = _fuzzy_normalize([r["distance"] for r in rows], higher_is_better=False)
    batt_r = _fuzzy_normalize([r["battery_margin"] for r in rows], higher_is_better=True)
    cov_r = _fuzzy_normalize([r["coverage_loss"] for r in rows], higher_is_better=False)
    delay_r = _fuzzy_normalize([r["relay_delay"] for r in rows], higher_is_better=False)

    w = weights
    scored = []
    for i, r in enumerate(rows):
        terms = [
            w["distance"] * (1 - dist_r[i]) ** p,
            w["battery_margin"] * (1 - batt_r[i]) ** p,
            w["coverage_loss"] * (1 - cov_r[i]) ** p,
            w["relay_delay"] * (1 - delay_r[i]) ** p,
        ]
        dist_to_ideal = sum(terms) ** (1.0 / p)
        closeness = 1.0 - dist_to_ideal   

        diag = {
            "candidate_id": r["cand"]["id"],
            "role": r["cand"]["role"],
            "distance": r["distance"],
            "travel_energy": round(r["travel_energy"], 3), 
            "battery_margin": round(r["battery_margin"], 2),
            "coverage_loss": round(r["coverage_loss"], 6),
            "relay_delay": r["relay_delay"],
            "membership_distance": round(dist_r[i], 3),
            "membership_battery": round(batt_r[i], 3),
            "membership_coverage": round(cov_r[i], 3),
            "membership_delay": round(delay_r[i], 3),
            "fuzzy_closeness": round(closeness, 4),
        }
        scored.append((r["cand"], closeness, diag))

    scored.sort(key=lambda x: -x[1])
    best_cand = scored[0][0]
    return best_cand, [d for _, _, d in scored]


# ══════════════════════════════════════════════════════════════
# POLICY T -- TOPSIS & POLICY V -- VIKOR
# ══════════════════════════════════════════════════════════════
TOPSIS_ATTR_WEIGHTS = dict(FUZZY_ATTR_WEIGHTS)
VIKOR_ATTR_WEIGHTS = dict(FUZZY_ATTR_WEIGHTS)

_MADM_CRITERIA = ("distance", "battery_margin", "coverage_loss", "relay_delay")
_MADM_BENEFIT = {
    "distance": False,
    "battery_margin": True,
    "coverage_loss": False,
    "relay_delay": False,
}

def _madm_build_rows(needer, drones, g, detected, step):  
    feasible = _feasible_candidate_pool(needer, drones)
    if not feasible:
        return []
    rows = []
    for cand, travel_energy, battery_margin in feasible:
        dist = _manhattan(cand["r"], cand["c"], needer["r"], needer["c"])
        relay_delay = dist
        coverage_loss, _raw, _tgap, _meta = _coverage_loss_for_candidate(
            cand, needer, g, detected, step, relay_delay, fleet=drones
        )
        rows.append({
            "cand": cand,
            "distance": dist,
            "travel_energy": travel_energy,
            "battery_margin": battery_margin,
            "coverage_loss": coverage_loss,
            "relay_delay": relay_delay,
        })
    return rows


def _madm_only_candidate_result(rows, extra_key, extra_val):
    only = rows[0]
    diag = {
        "candidate_id": only["cand"]["id"],
        "distance": only["distance"],
        "travel_energy": round(only["travel_energy"], 3),
        "battery_margin": round(only["battery_margin"], 2),
        "coverage_loss": round(only["coverage_loss"], 6),
        "relay_delay": only["relay_delay"],
        extra_key: extra_val,
        "note": "only feasible candidate -- no ranking needed",
    }
    return only["cand"], [diag]


def select_topsis_relay(needer, drones, g, detected, step, weights=None):
    if weights is None:
        weights = TOPSIS_ATTR_WEIGHTS

    rows = _madm_build_rows(needer, drones, g, detected, step)
    if not rows:
        return None, []
    if len(rows) == 1:
        return _madm_only_candidate_result(rows, "topsis_closeness", 1.0)

    norms = {}
    for crit in _MADM_CRITERIA:
        vals = [r[crit] for r in rows]
        denom = math.sqrt(sum(v * v for v in vals))
        norms[crit] = denom if denom > 1e-12 else 1.0

    weighted = [
        {crit: weights[crit] * (r[crit] / norms[crit]) for crit in _MADM_CRITERIA}
        for r in rows
    ]

    ideal_pos, ideal_neg = {}, {}
    for crit in _MADM_CRITERIA:
        col = [w[crit] for w in weighted]
        if _MADM_BENEFIT[crit]:
            ideal_pos[crit], ideal_neg[crit] = max(col), min(col)
        else:
            ideal_pos[crit], ideal_neg[crit] = min(col), max(col)

    scored = []
    for r, w in zip(rows, weighted):
        s_pos = math.sqrt(sum((w[c] - ideal_pos[c]) ** 2 for c in _MADM_CRITERIA))
        s_neg = math.sqrt(sum((w[c] - ideal_neg[c]) ** 2 for c in _MADM_CRITERIA))
        closeness = s_neg / (s_pos + s_neg) if (s_pos + s_neg) > 1e-12 else 0.5
        diag = {
            "candidate_id": r["cand"]["id"],
            "role": r["cand"]["role"],
            "distance": r["distance"],
            "travel_energy": round(r["travel_energy"], 3),
            "battery_margin": round(r["battery_margin"], 2),
            "coverage_loss": round(r["coverage_loss"], 6),
            "relay_delay": r["relay_delay"],
            "dist_to_positive_ideal": round(s_pos, 4),
            "dist_to_negative_ideal": round(s_neg, 4),
            "topsis_closeness": round(closeness, 4),
        }
        scored.append((r["cand"], closeness, diag))

    scored.sort(key=lambda x: -x[1])  
    best_cand = scored[0][0]
    return best_cand, [d for _, _, d in scored]


def select_vikor_relay(needer, drones, g, detected, step, weights=None, v=0.5):
    if weights is None:
        weights = VIKOR_ATTR_WEIGHTS

    rows = _madm_build_rows(needer, drones, g, detected, step)
    if not rows:
        return None, []
    if len(rows) == 1:
        return _madm_only_candidate_result(rows, "vikor_q", 0.0)

    best_val, worst_val = {}, {}
    for crit in _MADM_CRITERIA:
        vals = [r[crit] for r in rows]
        if _MADM_BENEFIT[crit]:
            best_val[crit], worst_val[crit] = max(vals), min(vals)
        else:
            best_val[crit], worst_val[crit] = min(vals), max(vals)

    S_list, R_list = [], []
    for r in rows:
        terms = []
        for crit in _MADM_CRITERIA:
            b, w_ = best_val[crit], worst_val[crit]
            denom = abs(b - w_)
            if denom < 1e-12:
                terms.append(0.0)  
            elif _MADM_BENEFIT[crit]:
                terms.append(weights[crit] * (b - r[crit]) / denom)
            else:
                terms.append(weights[crit] * (r[crit] - b) / denom)
        S_list.append(sum(terms))
        R_list.append(max(terms))

    S_star, S_worst = min(S_list), max(S_list)
    R_star, R_worst = min(R_list), max(R_list)
    s_spread = (S_worst - S_star) if (S_worst - S_star) > 1e-12 else 1.0
    r_spread = (R_worst - R_star) if (R_worst - R_star) > 1e-12 else 1.0

    scored = []
    for r, S_i, R_i in zip(rows, S_list, R_list):
        Q_i = v * (S_i - S_star) / s_spread + (1 - v) * (R_i - R_star) / r_spread
        diag = {
            "candidate_id": r["cand"]["id"],
            "role": r["cand"]["role"],
            "distance": r["distance"],
            "travel_energy": round(r["travel_energy"], 3),
            "battery_margin": round(r["battery_margin"], 2),
            "coverage_loss": round(r["coverage_loss"], 6),
            "relay_delay": r["relay_delay"],
            "vikor_S": round(S_i, 4),
            "vikor_R": round(R_i, 4),
            "vikor_q": round(Q_i, 4),
        }
        scored.append((r["cand"], Q_i, diag))

    scored.sort(key=lambda x: x[1])  
    best_cand = scored[0][0]
    return best_cand, [d for _, _, d in scored]


def _vikor_selector(*args, **kwargs):
    return select_vikor_relay(*args, v=0.5, **kwargs)
from surrogate_fis import select_fuzzy_relay_surrogate_fis

POLICY_SELECTORS = {
    "C": select_best_relay,
    "D": None, 
    "T": select_topsis_relay,
    "D2": select_fuzzy_relay_surrogate_fis,  
    "V": _vikor_selector,
    "I": select_best_relay_coverage_aware,
}
POLICY_DISPLAY_NAMES = {
    "C": "RelayScore", "D": "true FIS", "T": "TOPSIS","D2": "Surrogate FIS", "V": "VIKOR",
    "I": "RelayScore-CovFix",
}

from policy_d_true_fis import (
    select_fuzzy_relay_true_fis,
    select_fuzzy_relay_true_fis_indep_weights,
    select_fuzzy_relay_true_fis_indep2_weights,
    select_fuzzy_relay_true_fis_indep3_weights,
    select_fuzzy_relay_true_fis_indep4_weights,
    DEFAULT_RULE_WEIGHTS,
    INDEPENDENT_RULE_WEIGHTS,
)
POLICY_SELECTORS["D"] = select_fuzzy_relay_true_fis

# ══════════════════════════════════════════════════════════════
# ENERGY ACCUMULATION HELPER
# ══════════════════════════════════════════════════════════════
def _track_energy(state, cost):
    if cost == float('inf'):
        state["stranded_events"] += 1
    else:
        state["total_energy_consumed"] += cost


# ══════════════════════════════════════════════════════════════
# ONE SIMULATION STEP FOR THE WHOLE FLEET
# ══════════════════════════════════════════════════════════════
def step_fleet(state, relay_selector=select_best_relay):
    g, detected, drones, step = state["g"], state["detected"], state["drones"], state["step"]
    a20.advance_sim_clock(step)   
    by_id = {d["id"]: d for d in drones}

    for d in drones:
        if d["role"] != "returning":
            continue
        tr, tc = d["return_target"]
        if (d["r"], d["c"]) == (tr, tc):
            d["battery"] = 100.0
            d["cycles"] += 1  
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
            cost = a20.calculate_actual_step_cost(
                old_r, old_c, nr, nc, None, None,
                current_soc_pct=(d["battery"] if a20.USE_NONLINEAR_BATTERY else None),
                cycle_count=(d["cycles"] if a20.USE_SOH_AGING else 0))
            d["battery"] = max(0.0, d["battery"] - cost)
            _track_energy(state, cost)
            a20.sensor_sweep(nr, nc, g, detected, step=step)

    for d in drones:
        if d["role"] != "relay_incoming":
            continue
        needer = by_id.get(d["rendezvous_id"])
        if needer is None:
            d["role"] = "patrol"; continue
        tr, tc = needer["r"], needer["c"]
        dist_now = _manhattan(d["r"], d["c"], tr, tc)
        if dist_now <= 1:
            state["relay_fulfilled"] += 1
            st = a20.nearest_st(needer["r"], needer["c"])
            needer["role"] = "returning"
            needer["return_target"] = st
            needer["awaiting_relay"] = False
            needer["relay_episode_counted"] = False   
            d["role"] = "patrol"
            d["rendezvous_id"] = None
            d["target_zone"] = None; d["target_cell"] = None
            d["pr"] = d["pc"] = None
        else:
            old_r, old_c = d["r"], d["c"]
            nr, nc = a20.move_toward(d["r"], d["c"], tr, tc)
            d["r"], d["c"] = nr, nc
            cost = a20.calculate_actual_step_cost(
                old_r, old_c, nr, nc, None, None,
                current_soc_pct=(d["battery"] if a20.USE_NONLINEAR_BATTERY else None),
                cycle_count=(d["cycles"] if a20.USE_SOH_AGING else 0))
            d["battery"] = max(0.0, d["battery"] - cost)
            _track_energy(state, cost)
            a20.sensor_sweep(nr, nc, g, detected, step=step)

    for d in drones:
        if d["role"] != "patrol":
            continue

        if not d["awaiting_relay"]:
            nh, _st = a20.needs_handoff_now(d["r"], d["c"], d["battery"])
            if nh:
                if not d["relay_episode_counted"]:
                    state["relay_requested"] += 1
                    d["relay_episode_counted"] = True
                
                _t0 = time.perf_counter()
                chosen, _diag = relay_selector(d, drones, g, detected, step)
                _t1 = time.perf_counter()
                _dt = _t1 - _t0
                state["selector_time_total_s"] += _dt
                state["selector_time_samples_s"].append(_dt)
                state["selector_calls"] += 1
                if chosen is not None:
                    d["awaiting_relay"] = True
                    real = by_id[chosen["id"]]
                    real["role"] = "relay_incoming"
                    real["rendezvous_id"] = d["id"]
                    
                    state["log"].append({
                        "step": step,
                        "needer_id": d["id"],
                        "chosen_candidate_id": chosen["id"],
                        "candidates_diag": deepcopy(_diag)
                    })

        mrc, e_ret, st2 = a20.must_recharge_now(d["r"], d["c"], d["battery"])
        if mrc:
            cost = a20.calculate_actual_step_cost(
                d["r"], d["c"], d["r"], d["c"], d["pr"], d["pc"],
                current_soc_pct=(d["battery"] if a20.USE_NONLINEAR_BATTERY else None),
                cycle_count=(d["cycles"] if a20.USE_SOH_AGING else 0))
            d["battery"] = max(0.0, d["battery"] - cost)
            _track_energy(state, cost)
            a20.sensor_sweep(d["r"], d["c"], g, detected, step=step)
            continue

        if d["target_zone"]:
            zr, zc = d["target_zone"]
            if a20.get_zone_id(d["r"], d["c"]) == (zr, zc) and a20.find_uncovered_in_zone(zr, zc, g) is None:
                d["target_zone"] = None; d["target_cell"] = None
        if not d["target_zone"]:
            rk = a20.rank_zones(d["r"], d["c"], g, detected, step)

            claimed = {x["target_zone"] for x in drones
                       if x["id"] != d["id"] and x["role"] == "patrol" and x["target_zone"] is not None}
            rk_avail = [entry for entry in rk if entry["zone"] not in claimed]
            if not rk_avail:
                rk_avail = rk

            d["target_zone"], d["target_cell"] = a20.select_zone_mixed_strategy(
                rk_avail, g, d["r"], d["c"], d["battery"], d["zone_rng"], step)

        old_r, old_c = d["r"], d["c"]
        nr, nc, _bd = a20.smart_move(d["r"], d["c"], d["pr"], d["pc"], g, detected, d["target_zone"])
        d["r"], d["c"] = nr, nc
        cost = a20.calculate_actual_step_cost(
            old_r, old_c, nr, nc, old_r if d["pr"] is None else d["pr"],
            old_c if d["pc"] is None else d["pc"],
            current_soc_pct=(d["battery"] if a20.USE_NONLINEAR_BATTERY else None),
            cycle_count=(d["cycles"] if a20.USE_SOH_AGING else 0))
        d["battery"] = max(0.0, d["battery"] - cost)
        _track_energy(state, cost)
        a20.sensor_sweep(nr, nc, g, detected, step=step)
        d["pr"], d["pc"] = old_r, old_c

        if d["target_zone"]:
            d["target_cell"] = a20.find_uncovered_in_zone(*d["target_zone"], g)

    state["step"] += 1
    return state


# ══════════════════════════════════════════════════════════════
# METRICS & PLOTTING
# ══════════════════════════════════════════════════════════════
def _extract_baseline_a_metrics(seed, max_steps):
    sim = a20.DroneSimHeadless(seed)
    sim.run(max_steps)

    g, detected, step = sim.gs, sim.s_detected, sim.step
    staleness_values = [step - c["last_visited_step"] for c in g.values() if not c["is_station"]]
    avg_staleness = statistics.mean(staleness_values) if staleness_values else 0.0
    max_staleness = max(staleness_values) if staleness_values else 0.0

    requested = sim.sR if sim.sR > 0 else (1 if sim.s_handoff_mode else 0)
    fulfilled = sim.sR   

    return {
        "coverage_pct": a20.coverage_pct(g),
        "avg_zone_staleness_steps": round(avg_staleness, 2),
        "max_zone_staleness_steps": max_staleness,
        "detection_rate": round(len(detected) / a20.NUM_THREATS, 4) if a20.NUM_THREATS else None,
        "threats_detected": len(detected),
        "threats_total": a20.NUM_THREATS,
        "relay_success_rate": round(fulfilled / requested, 4) if requested > 0 else None,
        "avg_relay_delay_steps": None,     
        "avg_relay_energy_pct": None,
        "avg_relay_coverage_gap": None,
        "mission_completed": sim.s_first_all is not None,
        "first_all_threats_step": sim.s_first_all,
        "first_full_coverage_step": sim.s_full_cov_step,
        "avg_final_battery": round(sim.s_active["b"], 2),
        "avg_patrol_battery": round(sim.s_active["b"], 2),   
        "avg_reserve_battery": None,                          
        "total_energy_consumed": round(sim.s_energy, 2),      
        "relay_requested": requested,
        "relay_fulfilled": fulfilled,
        
        "avg_selector_time_ms": None,
        "median_selector_time_ms": None,
        "trimmed_mean_selector_time_ms": None,
        "max_selector_time_ms": None,
        "total_selector_time_ms": None,
        "selector_calls": None,

        "RC": sim.sR,
        "final_drone_count": 1,
    }


def collect_metrics(state):
    g, detected, drones = state["g"], state["detected"], state["drones"]
    active_drones = [d for d in drones]

    patrol_pool = [d for d in active_drones if d["role"] != "idle_reserve"]
    reserve_pool = [d for d in active_drones if d["role"] == "idle_reserve"]
    avg_patrol_battery = round(statistics.mean(d["battery"] for d in patrol_pool), 2) if patrol_pool else None
    avg_reserve_battery = round(statistics.mean(d["battery"] for d in reserve_pool), 2) if reserve_pool else None

    staleness_values = [state["step"] - c["last_visited_step"] for c in g.values() if not c["is_station"]]
    avg_staleness = statistics.mean(staleness_values)
    max_staleness = max(staleness_values)   

    requested = state["relay_requested"]
    fulfilled = state["relay_fulfilled"]
    success_rate = round(fulfilled / requested, 4) if requested > 0 else None

    chosen_diags = []
    for entry in state["log"]:
        for diag in entry["candidates_diag"]:
            if diag.get("candidate_id") == entry["chosen_candidate_id"]:
                chosen_diags.append(diag)
                break

    avg_relay_delay = round(statistics.mean(d.get("relay_delay", 0) for d in chosen_diags), 3) if chosen_diags else None
    avg_relay_energy = round(statistics.mean(d.get("travel_energy", 0) for d in chosen_diags), 3) if chosen_diags else None
    avg_coverage_gap = round(statistics.mean(d.get("coverage_loss", 0) for d in chosen_diags), 3) if chosen_diags else None

    detection_rate = round(len(detected) / a20.NUM_THREATS, 4) if a20.NUM_THREATS else None

    selector_calls = state.get("selector_calls", 0)
    selector_time_total_s = state.get("selector_time_total_s", 0.0)
    avg_selector_time_ms = (
        round(1000.0 * selector_time_total_s / selector_calls, 4)
        if selector_calls > 0 else None
    )

    _samples_ms = sorted(1000.0 * s for s in state.get("selector_time_samples_s", []))
    if _samples_ms:
        median_selector_time_ms = round(statistics.median(_samples_ms), 4)
        _trim = int(len(_samples_ms) * 0.10)
        _trimmed = _samples_ms[_trim: len(_samples_ms) - _trim] if _trim > 0 else _samples_ms
        trimmed_mean_selector_time_ms = round(statistics.mean(_trimmed), 4) if _trimmed else None
        max_selector_time_ms = round(_samples_ms[-1], 4)
    else:
        median_selector_time_ms = None
        trimmed_mean_selector_time_ms = None
        max_selector_time_ms = None

    return {
        "coverage_pct": a20.coverage_pct(g),                       
        "avg_zone_staleness_steps": round(avg_staleness, 2),       
        "max_zone_staleness_steps": max_staleness,                 
        "detection_rate": detection_rate,                          
        "threats_detected": len(detected),
        "threats_total": a20.NUM_THREATS,
        "relay_success_rate": success_rate,
        "avg_relay_delay_steps": avg_relay_delay,
        "avg_relay_energy_pct": avg_relay_energy,
        "avg_relay_coverage_gap": avg_coverage_gap,
        "mission_completed": state["first_all_threats_step"] is not None,   
        "first_all_threats_step": state["first_all_threats_step"],
        "first_full_coverage_step": state["first_full_coverage_step"],
        "avg_final_battery": avg_patrol_battery,        
        "avg_patrol_battery": avg_patrol_battery,        
        "avg_reserve_battery": avg_reserve_battery,      
        "total_energy_consumed": round(state["total_energy_consumed"], 2),  
        "stranded_events": state["stranded_events"],   
        "relay_requested": requested,                              
        "relay_fulfilled": fulfilled,
        "avg_selector_time_ms": avg_selector_time_ms,   
        "median_selector_time_ms": median_selector_time_ms,
        "trimmed_mean_selector_time_ms": trimmed_mean_selector_time_ms,
        "max_selector_time_ms": max_selector_time_ms,
        "total_selector_time_ms": round(1000.0 * selector_time_total_s, 3),
        "selector_calls": selector_calls,
        "RC": state["RC"],
        "final_drone_count": len(active_drones),
        "decision_log": state["log"]
    }

PRIMARY_METRICS = ["coverage_pct", "avg_zone_staleness_steps", "max_zone_staleness_steps"]
SECONDARY_METRICS = ["detection_rate", "relay_success_rate", "avg_relay_delay_steps",
                      "avg_relay_energy_pct", "mission_completed", "avg_final_battery",
                      "avg_patrol_battery", "avg_reserve_battery", "total_energy_consumed",
                      "relay_requested", "avg_selector_time_ms", "median_selector_time_ms",
                      "trimmed_mean_selector_time_ms", "total_selector_time_ms"]

def simulate_fleet(seed, relay_selector=select_best_relay, num_drones=NUM_DRONES,
                    num_reserves=RESERVE_POOL_SIZE, max_steps=None):
    max_steps = max_steps or a20.MAX_STEPS
    state = init_fleet(seed, num_drones, num_reserves)
    for _ in range(max_steps):
        step_fleet(state, relay_selector=relay_selector)
        if state["first_all_threats_step"] is None and len(state["detected"]) >= a20.NUM_THREATS:
            state["first_all_threats_step"] = state["step"]
        if state["first_full_coverage_step"] is None and a20.coverage_pct(state["g"]) >= 100:
            state["first_full_coverage_step"] = state["step"]
    return collect_metrics(state)

def plot_decision_breakdown(log_entry, weights=DEFAULT_WEIGHTS,
                            save_path="decision_breakdown.png"):
    if 'pd' not in globals() or 'plt' not in globals():
        print("❌ Cannot plot: pandas or matplotlib is missing.")
        return

    data = []
    for diag in log_entry["candidates_diag"]:
        if diag.get("score") is None:
            continue

        cand_id = diag["candidate_id"]
        energy_cost = -(weights["W_TRAVEL_ENERGY"] * diag.get("travel_energy", 0))
        cov_loss_cost = -(weights["W_COVERAGE_LOSS"] * diag.get("coverage_loss", 0))
        delay_cost = -(weights["W_RELAY_DELAY"] * diag.get("relay_delay", 0))

        is_winner = cand_id == log_entry["chosen_candidate_id"]
        label = f"Drone {cand_id}\n(Winner)" if is_winner else f"Drone {cand_id}"

        data.append({
            "Candidate": label,
            "Feasibility headroom": diag.get("battery_margin", 0),
            "Energy Cost (-)": energy_cost,
            "Coverage Cost (-)": cov_loss_cost,
            "Delay Cost (-)": delay_cost,
            "Net Utility": energy_cost + cov_loss_cost + delay_cost,
        })

    if not data:
        print("ℹ️ No feasible candidate diagnostics available for plotting.")
        return

    df = pd.DataFrame(data).set_index("Candidate")

    fig, ax = plt.subplots(figsize=(10, 7))
    df[["Energy Cost (-)", "Coverage Cost (-)", "Delay Cost (-)"]].plot(
        kind="bar", stacked=True, ax=ax, width=0.6
    )
    ax.plot(
        range(len(df)),
        df["Net Utility"],
        marker='D',
        linestyle='',
        markersize=8,
        label="Net Utility"
    )
    ax.axhline(0, linewidth=1.2)
    ax.set_title(
        f"Mission-Aware Relay Decision (Step {log_entry['step']} - "
        f"Needer: Drone {log_entry['needer_id']})",
        fontsize=12, fontweight='bold'
    )
    ax.set_ylabel("Weighted utility / cost units")
    ax.set_xticklabels(df.index, rotation=0)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"✅ Generated and saved decision breakdown chart to {save_path}")

def run_sensitivity_analysis(weight_name, sweep_values, seeds, max_steps=None,
                              base_weights=None, out_csv=None):
    base_weights = dict(base_weights or NORMALIZED_WEIGHTS)
    if weight_name not in base_weights:
        raise ValueError(
            f"Unknown weight '{weight_name}'. Choose from {list(base_weights)}"
        )

    results = []
    for val in sweep_values:
        trial_weights = dict(base_weights)
        trial_weights[weight_name] = val
        selector = partial(select_best_relay, weights=trial_weights)

        per_seed = [
            simulate_fleet(seed, relay_selector=selector, max_steps=max_steps)
            for seed in seeds
        ]

        def mean_metric(name):
            vals = [
                r[name] for r in per_seed
                if isinstance(r.get(name), (int, float))
            ]
            return round(statistics.mean(vals), 3) if vals else None

        row = {
            weight_name: val,
            "coverage_pct": mean_metric("coverage_pct"),
            "avg_zone_staleness_steps": mean_metric("avg_zone_staleness_steps"),
            "avg_final_battery": mean_metric("avg_final_battery"),
            "total_energy_consumed": mean_metric("total_energy_consumed"),
            "relay_success_rate": mean_metric("relay_success_rate"),
        }
        results.append(row)

        print(
            f"[{weight_name}={val}] "
            f"coverage={row['coverage_pct']}% "
            f"staleness={row['avg_zone_staleness_steps']} "
            f"energy={row['total_energy_consumed']}"
        )

    if out_csv and results:
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)

    return results

def plot_sensitivity(results, weight_name, metric="coverage_pct",
                      save_path="sensitivity_analysis.png"):
    if 'pd' not in globals() or 'plt' not in globals():
        print("❌ Cannot plot: pandas or matplotlib is missing.")
        return

    df = pd.DataFrame(results)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(df[weight_name], df[metric], marker='o', color="#1f77b4")
    ax.set_xlabel(weight_name)
    ax.set_ylabel(metric)
    ax.set_title(f"Sensitivity of {metric} to {weight_name}", fontweight='bold')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"✅ Generated and saved sensitivity plot to {save_path}")

def run_comparison(seeds, max_steps=None, out_csv=None, weights=None):
    selector_c = partial(select_best_relay, weights=weights) if weights else select_best_relay

    rows = []
    for seed in seeds:
        base = simulate_fleet(seed, relay_selector=select_nearest_relay,
                              max_steps=max_steps)
        opt = simulate_fleet(seed, relay_selector=selector_c,
                             max_steps=max_steps)

        row = {"seed": seed}
        for k, v in base.items():
            if k != "decision_log":
                row[f"baseline_{k}"] = v
        for k, v in opt.items():
            if k != "decision_log":
                row[f"optimized_{k}"] = v
        rows.append(row)

        print(
            f"[seed {seed}] "
            f"baseline cov={base['coverage_pct']}% "
            f"staleness={base['avg_zone_staleness_steps']} "
            f"relay_success={base['relay_success_rate']} | "
            f"optimized cov={opt['coverage_pct']}% "
            f"staleness={opt['avg_zone_staleness_steps']} "
            f"relay_success={opt['relay_success_rate']}"
        )

    if out_csv and rows:
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    metrics = [
        "coverage_pct",
        "avg_zone_staleness_steps",
        "max_zone_staleness_steps",
        "detection_rate",
        "avg_final_battery",
        "total_energy_consumed",
        "relay_success_rate",
        "avg_relay_delay_steps",
        "avg_relay_energy_pct",
        "avg_relay_coverage_gap",
    ]

    summary = {}
    for metric in metrics:
        base_vals = [r[f"baseline_{metric}"] for r in rows]
        opt_vals = [r[f"optimized_{metric}"] for r in rows]
        stats = _safe_wilcoxon(base_vals, opt_vals)

        summary[metric] = {
            "baseline_mean": round(
                statistics.mean(
                    [x for x in base_vals if isinstance(x, (int, float))]
                ), 3
            ) if any(isinstance(x, (int, float)) for x in base_vals) else None,
            "optimized_mean": round(
                statistics.mean(
                    [x for x in opt_vals if isinstance(x, (int, float))]
                ), 3
            ) if any(isinstance(x, (int, float)) for x in opt_vals) else None,
            **stats,
        }

    return rows, summary

def run_three_way_comparison(seeds, max_steps=None, num_drones=NUM_DRONES,
                              num_reserves=RESERVE_POOL_SIZE, out_csv=None):
    max_steps = max_steps or a20.MAX_STEPS
    rows = []
    for seed in seeds:
        m_a = _extract_baseline_a_metrics(seed, max_steps)
        m_b = simulate_fleet(seed, relay_selector=select_nearest_relay,
                              num_drones=num_drones, num_reserves=num_reserves, max_steps=max_steps)
        m_c = simulate_fleet(seed, relay_selector=select_best_relay,
                              num_drones=num_drones, num_reserves=num_reserves, max_steps=max_steps)
        row = {"seed": seed}
        for label, m in (("A", m_a), ("B", m_b), ("C", m_c)):
            for k, v in m.items():
                if k in ("decision_log",):
                    continue
                row[f"{label}_{k}"] = v
        rows.append(row)
        print(f"[seed {seed}] "
              f"A(orig): cov={m_a['coverage_pct']}% staleness(avg/max)={m_a['avg_zone_staleness_steps']}/{m_a['max_zone_staleness_steps']}  |  "
              f"B(8-drone,nearest): cov={m_b['coverage_pct']}% staleness(avg/max)={m_b['avg_zone_staleness_steps']}/{m_b['max_zone_staleness_steps']}  |  "
              f"C(8-drone,RelayScore): cov={m_c['coverage_pct']}% staleness(avg/max)={m_c['avg_zone_staleness_steps']}/{m_c['max_zone_staleness_steps']}")

    if out_csv:
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    summary = {}
    for metric in PRIMARY_METRICS + SECONDARY_METRICS:
        a_vals = [r[f"A_{metric}"] for r in rows]
        b_vals = [r[f"B_{metric}"] for r in rows]
        c_vals = [r[f"C_{metric}"] for r in rows]
        entry = {}
        for label, vals in (("A", a_vals), ("B", b_vals), ("C", c_vals)):
            clean = [v for v in vals if isinstance(v, (int, float))]
            entry[f"{label}_mean"] = round(statistics.mean(clean), 3) if clean else None
        fleet_effect = _safe_wilcoxon(a_vals, b_vals)  
        relay_effect = _safe_wilcoxon(b_vals, c_vals)  
        entry["fleet_effect_A_vs_B"] = fleet_effect
        entry["relay_intel_effect_B_vs_C"] = relay_effect
        summary[metric] = entry
    print("\n=== A vs B vs C SUMMARY (fleet-size effect: A->B | relay-intelligence effect: B->C) ===")
    print(json.dumps(summary, indent=2, default=str))
    return rows, summary
def _print_significance_summary(summary, comparisons, n=None, alpha=0.05):
    print(f"\n--- Statistical significance readout (alpha={alpha}, n={n} seeds) ---")
    for metric, entry in summary.items():
        for key, label_a, label_b in comparisons:
            stat = entry.get(key)
            if not stat or stat.get("wilcoxon_p") is None:
                continue
            p = stat["wilcoxon_p"]
            mean_a = entry.get(f"{label_a}_mean")
            mean_b = entry.get(f"{label_b}_mean")
            tag = "SIGNIFICANT" if p < alpha else "not significant"
            marker = "✅" if p < alpha else "—"
            test_note = " [t-test fallback]" if stat.get("test_used") == "ttest_rel_fallback" else ""
            print(f"{marker} [{metric}] {label_a}_mean={mean_a}  {label_b}_mean={mean_b}  "
                  f"diff({label_b}-{label_a})={stat['mean_diff']}  p={p}{test_note}  ({tag})")
            
def run_four_way_comparison(seeds, max_steps=None, num_drones=NUM_DRONES,
                             num_reserves=RESERVE_POOL_SIZE, out_csv=None):
    max_steps = max_steps or a20.MAX_STEPS
    rows = []
    for seed in seeds:
        m_a = _extract_baseline_a_metrics(seed, max_steps)
        m_b = simulate_fleet(seed, relay_selector=select_nearest_relay,
                              num_drones=num_drones, num_reserves=num_reserves, max_steps=max_steps)
        m_c = simulate_fleet(seed, relay_selector=select_best_relay,
                              num_drones=num_drones, num_reserves=num_reserves, max_steps=max_steps)
        m_d = simulate_fleet(seed, relay_selector=select_fuzzy_relay_true_fis,
                              num_drones=num_drones, num_reserves=num_reserves, max_steps=max_steps)
        row = {"seed": seed}
        for label, m in (("A", m_a), ("B", m_b), ("C", m_c), ("D", m_d)):
            for k, v in m.items():
                if k in ("decision_log",):
                    continue
                row[f"{label}_{k}"] = v
        rows.append(row)
        print(f"[seed {seed}] "
              f"A(orig): cov={m_a['coverage_pct']}%  |  "
              f"B(nearest): cov={m_b['coverage_pct']}%  |  "
              f"C(RelayScore): cov={m_c['coverage_pct']}%  |  "
              f"D(fuzzy-MADM): cov={m_d['coverage_pct']}%")

    if out_csv:
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    summary = {}
    for metric in PRIMARY_METRICS + SECONDARY_METRICS:
        vals = {}
        for label in ("A", "B", "C", "D"):
            vals[label] = [r.get(f"{label}_{metric}") for r in rows]
        entry = {}
        for label in ("A", "B", "C", "D"):
            clean = [v for v in vals[label] if isinstance(v, (int, float))]
            entry[f"{label}_mean"] = round(statistics.mean(clean), 3) if clean else None
        entry["fleet_effect_A_vs_B"] = _safe_wilcoxon(vals["A"], vals["B"])
        entry["relay_intel_effect_B_vs_C"] = _safe_wilcoxon(vals["B"], vals["C"])
        entry["relay_intel_effect_B_vs_D"] = _safe_wilcoxon(vals["B"], vals["D"])
        entry["ranking_rule_effect_C_vs_D"] = _safe_wilcoxon(vals["C"], vals["D"])
        summary[metric] = entry

    print("\n=== A vs B vs C vs D SUMMARY ===")
    print("(fleet-size: A->B | RelayScore vs nearest: B->C | "
          "fuzzy-MADM vs nearest: B->D | RelayScore vs fuzzy-MADM: C->D)")
    print(json.dumps(summary, indent=2, default=str))

    _print_significance_summary(
        summary,
        comparisons=[
            ("fleet_effect_A_vs_B", "A", "B"),
            ("relay_intel_effect_B_vs_C", "B", "C"),
            ("relay_intel_effect_B_vs_D", "B", "D"),
            ("ranking_rule_effect_C_vs_D", "C", "D"),
        ],
        n=len(seeds),
    )
    return rows, summary


def run_topsis_vikor_comparison(seeds, max_steps=None, num_drones=NUM_DRONES,
                                 num_reserves=RESERVE_POOL_SIZE, out_csv=None,
                                 fuzzy_selector=None, vikor_v=0.5,
                                 policies=("C", "D", "T", "V")):
    max_steps = max_steps or a20.MAX_STEPS
    if fuzzy_selector is not None:
        selectors = dict(POLICY_SELECTORS)
        selectors["D"] = fuzzy_selector
    else:
        selectors = POLICY_SELECTORS
    if vikor_v != 0.5:
        selectors = dict(selectors)
        selectors["V"] = lambda *a, **kw: select_vikor_relay(*a, v=vikor_v, **kw)

    non_d_policies = [p for p in policies if p != "D"]

    rows = []
    for seed in seeds:
        metrics_by_policy = {}
        for label in policies:
            metrics_by_policy[label] = simulate_fleet(
                seed, relay_selector=selectors[label],
                num_drones=num_drones, num_reserves=num_reserves, max_steps=max_steps,
            )
        row = {"seed": seed}
        for label, m in metrics_by_policy.items():
            for k, val in m.items():
                if k in ("decision_log",):
                    continue
                row[f"{label}_{k}"] = val
        rows.append(row)

        summary_bits = "  |  ".join(
            f"{label}({POLICY_DISPLAY_NAMES.get(label, label)}): "
            f"cov={metrics_by_policy[label]['coverage_pct']}%"
            for label in policies
        )
        print(f"[seed {seed}] {summary_bits}")

    if out_csv:
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    summary = {}
    for metric in PRIMARY_METRICS + SECONDARY_METRICS:
        vals = {label: [r.get(f"{label}_{metric}") for r in rows] for label in policies}
        entry = {}
        for label in policies:
            clean = [x for x in vals[label] if isinstance(x, (int, float))]
            entry[f"{label}_mean"] = round(statistics.mean(clean), 3) if clean else None

        comparisons = []
        if "D" in policies:
            for label in non_d_policies:
                key = f"{label}_vs_D"
                entry[key] = _safe_wilcoxon(vals[label], vals["D"])
                entry[f"{key}_effect"] = _rank_biserial_paired(vals[label], vals["D"])
                comparisons.append((key, label, "D"))
        for i, label_a in enumerate(non_d_policies):
            for label_b in non_d_policies[i + 1:]:
                key = f"{label_a}_vs_{label_b}"
                entry[key] = _safe_wilcoxon(vals[label_a], vals[label_b])
                entry[f"{key}_effect"] = _rank_biserial_paired(vals[label_a], vals[label_b])
                comparisons.append((key, label_a, label_b))
        entry["_comparisons"] = comparisons
        summary[metric] = entry

    print(f"\n=== {' vs '.join(policies)} SUMMARY ===")
    print(json.dumps({m: {k: v for k, v in e.items() if k != "_comparisons"} for m, e in summary.items()},
                      indent=2, default=str))

    all_comparisons = summary[PRIMARY_METRICS[0]]["_comparisons"] if PRIMARY_METRICS else []
    _print_significance_summary(
        {m: {k: v for k, v in e.items() if k != "_comparisons"} for m, e in summary.items()},
        comparisons=[(key, a, b) for key, a, b in all_comparisons],
        n=len(seeds),
    )

    print("\n--- Headline: selector time (median) & speedup vs D ---")
    time_metric = "median_selector_time_ms"
    d_time = summary.get(time_metric, {}).get("D_mean")
    for label in policies:
        t = summary.get(time_metric, {}).get(f"{label}_mean")
        if label == "D" or t is None:
            print(f"  {label} ({POLICY_DISPLAY_NAMES.get(label, label)}): {t} ms")
            continue
        speedup = round(d_time / t, 1) if (d_time and t and t > 0) else None
        p = summary.get(time_metric, {}).get(f"{label}_vs_D")
        pval = (p or {}).get("wilcoxon_p")
        print(f"  {label} ({POLICY_DISPLAY_NAMES.get(label, label)}): {t} ms  "
              f"| speedup over D = {speedup}x  (p={pval})")

    return rows, summary


def run_policy_i_comparison(seeds, max_steps=None, num_drones=NUM_DRONES,
                             num_reserves=RESERVE_POOL_SIZE, out_csv=None,
                             fuzzy_selector=None):
    max_steps = max_steps or a20.MAX_STEPS
    fuzzy_selector = fuzzy_selector or select_fuzzy_relay_true_fis
    rows = []
    for seed in seeds:
        m_c = simulate_fleet(seed, relay_selector=select_best_relay,
                              num_drones=num_drones, num_reserves=num_reserves, max_steps=max_steps)
        m_i = simulate_fleet(seed, relay_selector=select_best_relay_coverage_aware,
                              num_drones=num_drones, num_reserves=num_reserves, max_steps=max_steps)
        m_d = simulate_fleet(seed, relay_selector=fuzzy_selector,
                              num_drones=num_drones, num_reserves=num_reserves, max_steps=max_steps)
        row = {"seed": seed}
        for label, m in (("C", m_c), ("I", m_i), ("D", m_d)):
            for k, val in m.items():
                if k in ("decision_log",):
                    continue
                row[f"{label}_{k}"] = val
        rows.append(row)
        print(f"[seed {seed}] "
              f"C(RelayScore): cov={m_c['coverage_pct']}% staleness={m_c['avg_zone_staleness_steps']}  |  "
              f"I(RelayScore-CovFix): cov={m_i['coverage_pct']}% staleness={m_i['avg_zone_staleness_steps']}  |  "
              f"D(fuzzy-MADM): cov={m_d['coverage_pct']}% staleness={m_d['avg_zone_staleness_steps']}")

    if out_csv:
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    summary = {}
    for metric in PRIMARY_METRICS + SECONDARY_METRICS:
        vals = {}
        for label in ("C", "I", "D"):
            vals[label] = [r.get(f"{label}_{metric}") for r in rows]
        entry = {}
        for label in ("C", "I", "D"):
            clean = [x for x in vals[label] if isinstance(x, (int, float))]
            entry[f"{label}_mean"] = round(statistics.mean(clean), 3) if clean else None
        entry["fix_effect_I_vs_C"] = _safe_wilcoxon(vals["I"], vals["C"])
        entry["fix_effect_I_vs_C_rank_biserial"] = _rank_biserial_paired(vals["I"], vals["C"])
        entry["gap_closed_I_vs_D"] = _safe_wilcoxon(vals["I"], vals["D"])
        entry["gap_closed_I_vs_D_rank_biserial"] = _rank_biserial_paired(vals["I"], vals["D"])
        entry["original_gap_C_vs_D"] = _safe_wilcoxon(vals["C"], vals["D"])
        entry["original_gap_C_vs_D_rank_biserial"] = _rank_biserial_paired(vals["C"], vals["D"])
        summary[metric] = entry

    print("\n=== C vs I (RelayScore-CovFix) vs D SUMMARY ===")
    print("(fix's own effect: I->C | does the gap close: I->D | original gap for reference: C->D)")
    print(json.dumps(summary, indent=2, default=str))

    _print_significance_summary(
        summary,
        comparisons=[
            ("fix_effect_I_vs_C", "I", "C"),
            ("gap_closed_I_vs_D", "I", "D"),
            ("original_gap_C_vs_D", "C", "D"),
        ],
        n=len(seeds),
    )

    print("\n--- Headline readout: did the fix work? ---")
    cov = summary.get("coverage_pct", {})
    stale = summary.get("avg_zone_staleness_steps", {})
    print(f"coverage_pct:              C={cov.get('C_mean')}%  I={cov.get('I_mean')}%  D={cov.get('D_mean')}%  "
          f"| I-vs-D p={ (cov.get('gap_closed_I_vs_D') or {}).get('wilcoxon_p') }  "
          f"(was C-vs-D p={ (cov.get('original_gap_C_vs_D') or {}).get('wilcoxon_p') })")
    print(f"avg_zone_staleness_steps:  C={stale.get('C_mean')}  I={stale.get('I_mean')}  D={stale.get('D_mean')}  "
          f"| I-vs-D p={ (stale.get('gap_closed_I_vs_D') or {}).get('wilcoxon_p') }  "
          f"(was C-vs-D p={ (stale.get('original_gap_C_vs_D') or {}).get('wilcoxon_p') })")
    for metric in ("avg_relay_delay_steps", "avg_relay_energy_pct", "avg_selector_time_ms"):
        m = summary.get(metric, {})
        print(f"{metric} (should stay ~flat I vs C): C={m.get('C_mean')}  I={m.get('I_mean')}  "
              f"| I-vs-C p={ (m.get('fix_effect_I_vs_C') or {}).get('wilcoxon_p') }")

    return rows, summary


def run_five_way_comparison(seeds, max_steps=None, num_drones=NUM_DRONES,
                             num_reserves=RESERVE_POOL_SIZE, out_csv=None):
    max_steps = max_steps or a20.MAX_STEPS
    rows = []
    for seed in seeds:
        m_a = _extract_baseline_a_metrics(seed, max_steps)
        m_b = simulate_fleet(seed, relay_selector=select_nearest_relay,
                              num_drones=num_drones, num_reserves=num_reserves, max_steps=max_steps)
        m_c = simulate_fleet(seed, relay_selector=select_best_relay,
                              num_drones=num_drones, num_reserves=num_reserves, max_steps=max_steps)
        m_d = simulate_fleet(seed, relay_selector=select_fuzzy_relay_true_fis,
                              num_drones=num_drones, num_reserves=num_reserves, max_steps=max_steps)
        m_e = simulate_fleet(seed, relay_selector=select_best_relay_dynamic,
                              num_drones=num_drones, num_reserves=num_reserves, max_steps=max_steps)
        row = {"seed": seed}
        for label, m in (("A", m_a), ("B", m_b), ("C", m_c), ("D", m_d), ("E", m_e)):
            for k, v in m.items():
                if k in ("decision_log",):
                    continue
                row[f"{label}_{k}"] = v
        rows.append(row)
        print(f"[seed {seed}] "
              f"B(nearest): cov={m_b['coverage_pct']}%  |  "
              f"C(RelayScore-static): cov={m_c['coverage_pct']}%  |  "
              f"D(fuzzy-MADM): cov={m_d['coverage_pct']}%  |  "
              f"E(RelayScore-dynamic): cov={m_e['coverage_pct']}%")

    if out_csv:
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    summary = {}
    for metric in PRIMARY_METRICS + SECONDARY_METRICS:
        vals = {}
        for label in ("A", "B", "C", "D", "E"):
            vals[label] = [r[f"{label}_{metric}"] for r in rows]
        entry = {}
        for label in ("A", "B", "C", "D", "E"):
            clean = [v for v in vals[label] if isinstance(v, (int, float))]
            entry[f"{label}_mean"] = round(statistics.mean(clean), 3) if clean else None
        entry["fleet_effect_A_vs_B"] = _safe_wilcoxon(vals["A"], vals["B"])
        entry["relay_intel_effect_B_vs_C"] = _safe_wilcoxon(vals["B"], vals["C"])
        entry["relay_intel_effect_B_vs_D"] = _safe_wilcoxon(vals["B"], vals["D"])
        entry["ranking_rule_effect_C_vs_D"] = _safe_wilcoxon(vals["C"], vals["D"])
        entry["dynamic_weighting_effect_C_vs_E"] = _safe_wilcoxon(vals["C"], vals["E"])
        entry["fully_refined_effect_D_vs_E"] = _safe_wilcoxon(vals["D"], vals["E"])
        summary[metric] = entry

    print("\n=== A vs B vs C vs D vs E SUMMARY ===")
    print("(RelayScore vs fuzzy-MADM: C->D | dynamic layer's own effect: C->E | "
          "fully-refined RelayScore vs fuzzy-MADM: D->E)")
    print(json.dumps(summary, indent=2, default=str))

    _print_significance_summary(
        summary,
        comparisons=[
            ("fleet_effect_A_vs_B", "A", "B"),
            ("relay_intel_effect_B_vs_C", "B", "C"),
            ("relay_intel_effect_B_vs_D", "B", "D"),
            ("ranking_rule_effect_C_vs_D", "C", "D"),
            ("dynamic_weighting_effect_C_vs_E", "C", "E"),
            ("fully_refined_effect_D_vs_E", "D", "E"),
        ],
        n=len(seeds),
    )
    return rows, summary

def run_six_way_comparison(seeds, max_steps=None, num_drones=NUM_DRONES,
                             num_reserves=RESERVE_POOL_SIZE, out_csv=None,
                             predictive_horizon=PREDICTIVE_COVERAGE_HORIZON):
    max_steps = max_steps or a20.MAX_STEPS
    rows = []
    predictive_selector = lambda needer, drones, g, detected, step, weights=None: \
        select_best_relay_predictive(needer, drones, g, detected, step, weights=weights,
                                       horizon=predictive_horizon)
    for seed in seeds:
        m_a = _extract_baseline_a_metrics(seed, max_steps)
        m_b = simulate_fleet(seed, relay_selector=select_nearest_relay,
                              num_drones=num_drones, num_reserves=num_reserves, max_steps=max_steps)
        m_c = simulate_fleet(seed, relay_selector=select_best_relay,
                              num_drones=num_drones, num_reserves=num_reserves, max_steps=max_steps)
        m_d = simulate_fleet(seed, relay_selector=select_fuzzy_relay_true_fis,
                              num_drones=num_drones, num_reserves=num_reserves, max_steps=max_steps)
        m_e = simulate_fleet(seed, relay_selector=select_best_relay_dynamic,
                              num_drones=num_drones, num_reserves=num_reserves, max_steps=max_steps)
        m_f = simulate_fleet(seed, relay_selector=predictive_selector,
                              num_drones=num_drones, num_reserves=num_reserves, max_steps=max_steps)
        row = {"seed": seed}
        for label, m in (("A", m_a), ("B", m_b), ("C", m_c), ("D", m_d), ("E", m_e), ("F", m_f)):
            for k, v in m.items():
                if k in ("decision_log",):
                    continue
                row[f"{label}_{k}"] = v
        rows.append(row)
        print(f"[seed {seed}] "
              f"C(static): cov={m_c['coverage_pct']}%  |  "
              f"D(fuzzy): cov={m_d['coverage_pct']}%  |  "
              f"E(dynamic-weight): cov={m_e['coverage_pct']}%  |  "
              f"F(predictive): cov={m_f['coverage_pct']}%")

    if out_csv:
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    summary = {}
    for metric in PRIMARY_METRICS + SECONDARY_METRICS:
        vals = {}
        for label in ("A", "B", "C", "D", "E", "F"):
            vals[label] = [r[f"{label}_{metric}"] for r in rows]
        entry = {}
        for label in ("A", "B", "C", "D", "E", "F"):
            clean = [v for v in vals[label] if isinstance(v, (int, float))]
            entry[f"{label}_mean"] = round(statistics.mean(clean), 3) if clean else None
        entry["fleet_effect_A_vs_B"] = _safe_wilcoxon(vals["A"], vals["B"])
        entry["relay_intel_effect_B_vs_C"] = _safe_wilcoxon(vals["B"], vals["C"])
        entry["ranking_rule_effect_C_vs_D"] = _safe_wilcoxon(vals["C"], vals["D"])
        entry["dynamic_weighting_effect_C_vs_E"] = _safe_wilcoxon(vals["C"], vals["E"])
        entry["predictive_effect_C_vs_F"] = _safe_wilcoxon(vals["C"], vals["F"])
        entry["reweight_vs_predict_E_vs_F"] = _safe_wilcoxon(vals["E"], vals["F"])
        entry["predictive_vs_fuzzy_D_vs_F"] = _safe_wilcoxon(vals["D"], vals["F"])
        summary[metric] = entry

    print("\n=== A vs B vs C vs D vs E vs F SUMMARY ===")
    print(f"(predictive_horizon={predictive_horizon})")
    print("(predictive effect: C->F | reweight-vs-predict: E->F | predictive vs fuzzy: D->F)")
    print(json.dumps(summary, indent=2, default=str))

    _print_significance_summary(
        summary,
        comparisons=[
            ("fleet_effect_A_vs_B", "A", "B"),
            ("relay_intel_effect_B_vs_C", "B", "C"),
            ("ranking_rule_effect_C_vs_D", "C", "D"),
            ("dynamic_weighting_effect_C_vs_E", "C", "E"),
            ("predictive_effect_C_vs_F", "C", "F"),
            ("reweight_vs_predict_E_vs_F", "E", "F"),
            ("predictive_vs_fuzzy_D_vs_F", "D", "F"),
        ],
        n=len(seeds),
    )
    return rows, summary

def run_seven_way_comparison(seeds, max_steps=None, num_drones=NUM_DRONES,
                               num_reserves=RESERVE_POOL_SIZE, out_csv=None,
                               predictive_horizon=PREDICTIVE_COVERAGE_HORIZON):
    max_steps = max_steps or a20.MAX_STEPS
    rows = []
    predictive_selector = lambda needer, drones, g, detected, step, weights=None: \
        select_best_relay_predictive(needer, drones, g, detected, step, weights=weights,
                                       horizon=predictive_horizon)
    for seed in seeds:
        m_a = _extract_baseline_a_metrics(seed, max_steps)
        m_b = simulate_fleet(seed, relay_selector=select_nearest_relay,
                              num_drones=num_drones, num_reserves=num_reserves, max_steps=max_steps)
        m_c = simulate_fleet(seed, relay_selector=select_best_relay,
                              num_drones=num_drones, num_reserves=num_reserves, max_steps=max_steps)
        m_d = simulate_fleet(seed, relay_selector=select_fuzzy_relay_true_fis,
                              num_drones=num_drones, num_reserves=num_reserves, max_steps=max_steps)
        m_e = simulate_fleet(seed, relay_selector=select_best_relay_dynamic,
                              num_drones=num_drones, num_reserves=num_reserves, max_steps=max_steps)
        m_f = simulate_fleet(seed, relay_selector=predictive_selector,
                              num_drones=num_drones, num_reserves=num_reserves, max_steps=max_steps)
        m_g = simulate_fleet(seed, relay_selector=select_best_relay_battery_aware,
                              num_drones=num_drones, num_reserves=num_reserves, max_steps=max_steps)
        row = {"seed": seed}
        for label, m in (("A", m_a), ("B", m_b), ("C", m_c), ("D", m_d),
                          ("E", m_e), ("F", m_f), ("G", m_g)):
            for k, v in m.items():
                if k in ("decision_log",):
                    continue
                row[f"{label}_{k}"] = v
        rows.append(row)
        print(f"[seed {seed}] "
              f"C(static): cov={m_c['coverage_pct']}%  |  "
              f"D(fuzzy): cov={m_d['coverage_pct']}%  |  "
              f"F(predictive): cov={m_f['coverage_pct']}%  |  "
              f"G(battery-aware): cov={m_g['coverage_pct']}%  |  "
              f"G mission_completed={m_g['mission_completed']}")

    if out_csv:
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    summary = {}
    for metric in PRIMARY_METRICS + SECONDARY_METRICS:
        vals = {}
        for label in ("A", "B", "C", "D", "E", "F", "G"):
            vals[label] = [r[f"{label}_{metric}"] for r in rows]
        entry = {}
        for label in ("A", "B", "C", "D", "E", "F", "G"):
            clean = [v for v in vals[label] if isinstance(v, (int, float))]
            entry[f"{label}_mean"] = round(statistics.mean(clean), 3) if clean else None
        entry["fleet_effect_A_vs_B"] = _safe_wilcoxon(vals["A"], vals["B"])
        entry["relay_intel_effect_B_vs_C"] = _safe_wilcoxon(vals["B"], vals["C"])
        entry["ranking_rule_effect_C_vs_D"] = _safe_wilcoxon(vals["C"], vals["D"])
        entry["dynamic_weighting_effect_C_vs_E"] = _safe_wilcoxon(vals["C"], vals["E"])
        entry["predictive_effect_C_vs_F"] = _safe_wilcoxon(vals["C"], vals["F"])
        entry["battery_aware_effect_C_vs_G"] = _safe_wilcoxon(vals["C"], vals["G"])
        entry["battery_vs_fuzzy_D_vs_G"] = _safe_wilcoxon(vals["D"], vals["G"])
        entry["battery_vs_predictive_G_vs_F"] = _safe_wilcoxon(vals["G"], vals["F"])
        summary[metric] = entry

    print("\n=== A vs B vs C vs D vs E vs F vs G SUMMARY ===")
    print(f"(predictive_horizon={predictive_horizon})")
    print("(battery-aware effect: C->G | battery vs fuzzy: D->G | battery vs predictive: G->F)")
    print(json.dumps(summary, indent=2, default=str))

    _print_significance_summary(
        summary,
        comparisons=[
            ("fleet_effect_A_vs_B", "A", "B"),
            ("relay_intel_effect_B_vs_C", "B", "C"),
            ("ranking_rule_effect_C_vs_D", "C", "D"),
            ("dynamic_weighting_effect_C_vs_E", "C", "E"),
            ("predictive_effect_C_vs_F", "C", "F"),
            ("battery_aware_effect_C_vs_G", "C", "G"),
            ("battery_vs_fuzzy_D_vs_G", "D", "G"),
            ("battery_vs_predictive_G_vs_F", "G", "F"),
        ],
        n=len(seeds),
    )
    return rows, summary


def run_policy_d_weight_robustness_check(seeds, max_steps=None, num_drones=NUM_DRONES,
                                          num_reserves=RESERVE_POOL_SIZE, out_csv=None):
    max_steps = max_steps or a20.MAX_STEPS
    rows = []
    for seed in seeds:
        m_c = simulate_fleet(seed, relay_selector=select_best_relay,
                              num_drones=num_drones, num_reserves=num_reserves, max_steps=max_steps)
        m_d = simulate_fleet(seed, relay_selector=select_fuzzy_relay_true_fis,
                              num_drones=num_drones, num_reserves=num_reserves, max_steps=max_steps)
        m_d_indep = simulate_fleet(seed, relay_selector=select_fuzzy_relay_true_fis_indep_weights,
                                    num_drones=num_drones, num_reserves=num_reserves, max_steps=max_steps)
        row = {"seed": seed}
        for label, m in (("C", m_c), ("D", m_d), ("D_indep", m_d_indep)):
            for k, v in m.items():
                if k in ("decision_log",):
                    continue
                row[f"{label}_{k}"] = v
        rows.append(row)
        print(f"[seed {seed}] "
              f"C(RelayScore): cov={m_c['coverage_pct']}%  |  "
              f"D(fuzzy-MADM, weights shared with C): cov={m_d['coverage_pct']}%  |  "
              f"D_indep(fuzzy-MADM, equal/independent weights): cov={m_d_indep['coverage_pct']}%")

    if out_csv:
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    summary = {}
    for metric in PRIMARY_METRICS + SECONDARY_METRICS:
        vals = {}
        for label in ("C", "D", "D_indep"):
            vals[label] = [r[f"{label}_{metric}"] for r in rows]
        entry = {}
        for label in ("C", "D", "D_indep"):
            clean = [v for v in vals[label] if isinstance(v, (int, float))]
            entry[f"{label}_mean"] = round(statistics.mean(clean), 3) if clean else None
        entry["C_vs_D_shared_weights"] = _safe_wilcoxon(vals["C"], vals["D"])
        entry["C_vs_D_indep_weights"] = _safe_wilcoxon(vals["C"], vals["D_indep"])
        entry["D_vs_D_indep_rule_base_sensitivity"] = _safe_wilcoxon(vals["D"], vals["D_indep"])
        summary[metric] = entry

    print("\n=== RULE-BASE-CIRCULARITY ROBUSTNESS CHECK: C vs D (shared weights) vs D-indep (equal weights) ===")
    print(json.dumps(summary, indent=2, default=str))

    _print_significance_summary(
        summary,
        comparisons=[
            ("C_vs_D_shared_weights", "C", "D"),
            ("C_vs_D_indep_weights", "C", "D_indep"),
            ("D_vs_D_indep_rule_base_sensitivity", "D", "D_indep"),
        ],
        n=len(seeds),
    )
    return rows, summary


# ══════════════════════════════════════════════════════════════
# INDEPENDENT RULE-BASE PANEL
# ══════════════════════════════════════════════════════════════
INDEP_PANEL_SELECTORS = {
    "D_indep":  select_fuzzy_relay_true_fis_indep_weights,
    "D_indep2": select_fuzzy_relay_true_fis_indep2_weights,
    "D_indep3": select_fuzzy_relay_true_fis_indep3_weights,
    "D_indep4": select_fuzzy_relay_true_fis_indep4_weights,
}


def _holm_adjust(pvals):
    idx = [i for i, p in enumerate(pvals) if p is not None]
    order = sorted(idx, key=lambda i: pvals[i])
    m = len(order)
    adj = [None] * len(pvals)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        adj[i] = round(running, 5)
    return adj


def _paired_diff_ci(x, y, n_boot=10000, seed=12345, level=0.95):
    import numpy as _np
    d = _np.array([float(b) - float(a) for a, b in zip(x, y)
                   if isinstance(a, (int, float, bool)) and isinstance(b, (int, float, bool))])
    if d.size < 2:
        return None
    rng = _np.random.default_rng(seed)
    means = rng.choice(d, size=(n_boot, d.size), replace=True).mean(axis=1)
    lo, hi = _np.percentile(means, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return [round(float(lo), 4), round(float(hi), 4)]


def run_indep_rulebase_panel_check(seeds, panel=("D_indep", "D_indep2", "D_indep3", "D_indep4"),
                                   max_steps=None, num_drones=NUM_DRONES,
                                   num_reserves=RESERVE_POOL_SIZE, out_csv=None, alpha=0.05):
    max_steps = max_steps or a20.MAX_STEPS
    unknown = [p for p in panel if p not in INDEP_PANEL_SELECTORS]
    if unknown:
        raise ValueError(f"unknown panel entries {unknown}; known: {sorted(INDEP_PANEL_SELECTORS)}")
    labels = ("C", "D") + tuple(panel)
    selectors = {"C": select_best_relay, "D": select_fuzzy_relay_true_fis}
    selectors.update({p: INDEP_PANEL_SELECTORS[p] for p in panel})

    rows = []
    for seed in seeds:
        row = {"seed": seed}
        covs = []
        for label in labels:
            m = simulate_fleet(seed, relay_selector=selectors[label], num_drones=num_drones,
                               num_reserves=num_reserves, max_steps=max_steps)
            for k, v in m.items():
                if k != "decision_log":
                    row[f"{label}_{k}"] = v
            covs.append(f"{label}={m['coverage_pct']}%")
        rows.append(row)
        print(f"[seed {seed}] cov: " + "  |  ".join(covs))

    if out_csv:
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    summary = {}
    for metric in PRIMARY_METRICS + SECONDARY_METRICS:
        vals = {lab: [r.get(f"{lab}_{metric}") for r in rows] for lab in labels}
        entry = {}
        for lab in labels:
            clean = [v for v in vals[lab] if isinstance(v, (int, float))]
            entry[f"{lab}_mean"] = round(statistics.mean(clean), 3) if clean else None
        raw_p = []
        for p in panel:
            st = _safe_wilcoxon(vals["C"], vals[p])
            st["diff_ci95"] = _paired_diff_ci(vals["C"], vals[p])
            entry[f"C_vs_{p}"] = st
            raw_p.append(st.get("wilcoxon_p"))
        for p, adj in zip(panel, _holm_adjust(raw_p)):
            entry[f"C_vs_{p}"]["holm_p"] = adj
        summary[metric] = entry

    print("\n=== INDEPENDENT RULE-BASE PANEL: C (RelayScore) vs " + ", ".join(panel) + " ===")
    print(json.dumps(summary, indent=2, default=str))

    print(f"\n--- coverage_pct readout (alpha={alpha}, n={len(seeds)}, Holm over {len(panel)} tests) ---")
    cov = summary["coverage_pct"]
    for p in panel:
        st = cov[f"C_vs_{p}"]
        hp = st.get("holm_p")
        tag = ("SIGNIFICANT" if hp is not None and hp < alpha else "not significant")
        print(f"C_mean={cov['C_mean']}  {p}_mean={cov[f'{p}_mean']}  "
              f"diff={st['mean_diff']}  CI95={st['diff_ci95']}  "
              f"p={st['wilcoxon_p']}  holm_p={hp}  ({tag})")
    return rows, summary


def run_fleet_size_sensitivity_analysis(seeds, fleet_sizes=((1, 1), (2, 2), (4, 4), (8, 8)),
                                         max_steps=None, out_csv=None,
                                         policies=("C", "D", "T", "V")):
    max_steps = max_steps or a20.MAX_STEPS
    all_rows = []
    scaling_summary = []
    non_d_policies = [p for p in policies if p != "D"]

    def _mean(xs):
        clean = [x for x in xs if isinstance(x, (int, float))]
        return round(statistics.mean(clean), 4) if clean else None

    for num_drones, num_reserves in fleet_sizes:
        total_n = num_drones + num_reserves
        rows = []
        for seed in seeds:
            metrics_by_policy = {}
            for label in policies:
                metrics_by_policy[label] = simulate_fleet(
                    seed, relay_selector=POLICY_SELECTORS[label],
                    num_drones=num_drones, num_reserves=num_reserves, max_steps=max_steps,
                )
            row = {"fleet_size": total_n, "num_drones": num_drones,
                   "num_reserves": num_reserves, "seed": seed}
            for label, m in metrics_by_policy.items():
                for k, v in m.items():
                    if k == "decision_log":
                        continue
                    row[f"{label}_{k}"] = v
            rows.append(row)
            all_rows.append(row)

        metric_names = ("coverage_pct", "avg_relay_delay_steps",
                         "avg_relay_energy_pct", "avg_selector_time_ms",
                         "median_selector_time_ms", "total_selector_time_ms")
        vals = {
            label: {metric: [r[f"{label}_{metric}"] for r in rows] for metric in metric_names}
            for label in policies
        }

        entry = {"fleet_size": total_n, "num_drones": num_drones, "num_reserves": num_reserves}
        d_time_robust = None
        for label in policies:
            entry[f"{label}_coverage_pct"] = _mean(vals[label]["coverage_pct"])
            entry[f"{label}_avg_relay_delay_steps"] = _mean(vals[label]["avg_relay_delay_steps"])
            entry[f"{label}_avg_relay_energy_pct"] = _mean(vals[label]["avg_relay_energy_pct"])
            entry[f"{label}_avg_selector_time_ms"] = _mean(vals[label]["avg_selector_time_ms"])
            med_times = [x for x in vals[label]["median_selector_time_ms"] if x is not None]
            entry[f"{label}_median_selector_time_ms"] = round(statistics.median(med_times), 4) if med_times else None
            if label == "D":
                d_time_robust = entry[f"{label}_median_selector_time_ms"]

        for label in non_d_policies:
            if "D" not in policies:
                continue
            entry[f"coverage_test_{label}_vs_D"] = _safe_wilcoxon(vals[label]["coverage_pct"], vals["D"]["coverage_pct"])
            entry[f"coverage_effect_{label}_vs_D"] = _rank_biserial_paired(vals[label]["coverage_pct"], vals["D"]["coverage_pct"])
            entry[f"delay_test_{label}_vs_D"] = _safe_wilcoxon(vals[label]["avg_relay_delay_steps"], vals["D"]["avg_relay_delay_steps"])
            entry[f"delay_effect_{label}_vs_D"] = _rank_biserial_paired(vals[label]["avg_relay_delay_steps"], vals["D"]["avg_relay_delay_steps"])
            entry[f"energy_test_{label}_vs_D"] = _safe_wilcoxon(vals[label]["avg_relay_energy_pct"], vals["D"]["avg_relay_energy_pct"])
            entry[f"energy_effect_{label}_vs_D"] = _rank_biserial_paired(vals[label]["avg_relay_energy_pct"], vals["D"]["avg_relay_energy_pct"])
            entry[f"selector_time_test_{label}_vs_D"] = _safe_wilcoxon(vals[label]["median_selector_time_ms"], vals["D"]["median_selector_time_ms"])
            label_time_robust = entry.get(f"{label}_median_selector_time_ms")
            entry[f"speedup_D_over_{label}"] = round(d_time_robust / label_time_robust, 2) \
                if (label_time_robust and d_time_robust and label_time_robust > 0) else None

        if "C" in policies and "D" in policies:
            entry["coverage_test"] = entry.get("coverage_test_C_vs_D")
            entry["delay_test"] = entry.get("delay_test_C_vs_D")
            entry["energy_test"] = entry.get("energy_test_C_vs_D")
            entry["selector_time_test"] = entry.get("selector_time_test_C_vs_D")
            entry["speedup_D_over_C"] = entry.get("speedup_D_over_C")  
            c_time = entry.get("C_avg_selector_time_ms")
            d_time = entry.get("D_avg_selector_time_ms")
            entry["speedup_D_over_C_mean_based"] = round(d_time / c_time, 2) if (c_time and d_time and c_time > 0) else None

        scaling_summary.append(entry)

        summary_bits = "  |  ".join(
            f"{label}({POLICY_DISPLAY_NAMES[label]}) cov={entry[f'{label}_coverage_pct']}% "
            f"delay={entry[f'{label}_avg_relay_delay_steps']} "
            f"selector(median)={entry[f'{label}_median_selector_time_ms']}ms"
            for label in policies
        )
        print(f"\n[fleet_size={total_n} (patrol={num_drones}, reserve={num_reserves})]  {summary_bits}")

    if out_csv and all_rows:
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)

    print(f"\n=== FLEET-SIZE SCALING SENSITIVITY: {', '.join(policies)} across N ===")
    print(json.dumps(scaling_summary, indent=2, default=str))
    return all_rows, scaling_summary


def plot_fleet_size_scaling(scaling_summary, save_path="fleet_size_scaling.png"):
    if 'plt' not in globals():
        print("❌ Cannot plot: matplotlib is missing.")
        return

    sizes = [e["fleet_size"] for e in scaling_summary]
    colors = {"C": "tab:blue", "D": "tab:orange", "T": "tab:green", "V": "tab:purple"}
    present = [label for label in ("C", "D", "T", "V")
               if all(f"{label}_median_selector_time_ms" in e for e in scaling_summary)]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for label in present:
        times = [e[f"{label}_median_selector_time_ms"] for e in scaling_summary]
        name = POLICY_DISPLAY_NAMES.get(label, label)
        axes[0].plot(sizes, times, marker="o", label=f"{name} ({label})", color=colors.get(label))
    axes[0].set_xlabel("Fleet size (N drones)")
    axes[0].set_ylabel("Median selector time (ms)")
    axes[0].set_title("Per-decision runtime vs. fleet size")
    axes[0].legend()

    for label in present:
        if label == "D":
            continue
        key = "speedup_D_over_C" if label == "C" else f"speedup_D_over_{label}"
        if not all(key in e for e in scaling_summary):
            continue
        speedup = [e[key] for e in scaling_summary]
        name = POLICY_DISPLAY_NAMES.get(label, label)
        axes[1].plot(sizes, speedup, marker="o", label=f"{name} ({label})", color=colors.get(label))
    axes[1].set_xlabel("Fleet size (N drones)")
    axes[1].set_ylabel("Speedup (D / policy, median-based)")
    axes[1].set_title("Speedup over true FIS vs. fleet size")
    if len(present) > 2:
        axes[1].legend()

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"✅ Saved fleet-size scaling plot to {save_path}")


def run_terrain_wind_generalization_analysis(seeds, terrain_scenarios=None, wind_modes=("static", "dynamic", "stochastic"),
                                               max_steps=None, num_drones=NUM_DRONES,
                                               num_reserves=RESERVE_POOL_SIZE, out_csv=None,
                                               policies=("C", "D", "T", "V")):
    max_steps = max_steps or a20.MAX_STEPS
    terrain_scenarios = terrain_scenarios or list(a20.TERRAIN_SCENARIOS.keys())
    non_d_policies = [p for p in policies if p != "D"]

    def _mean(xs):
        clean = [x for x in xs if isinstance(x, (int, float))]
        return round(statistics.mean(clean), 4) if clean else None

    all_rows = []
    scenario_summary = []

    try:
        for terrain in terrain_scenarios:
            a20.set_terrain_scenario(terrain)

            for wind_mode in wind_modes:
                rows = []
                for seed in seeds:
                    if wind_mode == "stochastic":
                        a20.set_wind_mode("stochastic", seed=seed, n_steps=max_steps + 1)
                    else:
                        a20.set_wind_mode(wind_mode)

                    metrics_by_policy = {}
                    for label in policies:
                        metrics_by_policy[label] = simulate_fleet(
                            seed, relay_selector=POLICY_SELECTORS[label],
                            num_drones=num_drones, num_reserves=num_reserves, max_steps=max_steps,
                        )

                    row = {"terrain": terrain, "wind_mode": wind_mode, "seed": seed}
                    for label, m in metrics_by_policy.items():
                        for k, v in m.items():
                            if k == "decision_log":
                                continue
                            row[f"{label}_{k}"] = v
                    rows.append(row)
                    all_rows.append(row)

                metric_names = ("coverage_pct", "avg_relay_delay_steps", "avg_relay_energy_pct")
                vals = {
                    label: {metric: [r[f"{label}_{metric}"] for r in rows] for metric in metric_names}
                    for label in policies
                }
                med_times = {}
                for label in policies:
                    times = [r[f"{label}_median_selector_time_ms"] for r in rows
                             if r.get(f"{label}_median_selector_time_ms") is not None]
                    med_times[label] = round(statistics.median(times), 4) if times else None

                entry = {"terrain": terrain, "wind_mode": wind_mode}
                for label in policies:
                    for metric in metric_names:
                        entry[f"{label}_{metric}"] = _mean(vals[label][metric])
                    entry[f"{label}_median_selector_time_ms"] = med_times[label]

                d_time_robust = med_times.get("D")
                for label in non_d_policies:
                    if "D" not in policies:
                        continue
                    for metric in metric_names:
                        entry[f"{metric}_test_{label}_vs_D"] = _safe_wilcoxon(vals[label][metric], vals["D"][metric])
                        entry[f"{metric}_effect_{label}_vs_D"] = _rank_biserial_paired(vals[label][metric], vals["D"][metric])
                    label_time_robust = med_times.get(label)
                    entry[f"speedup_D_over_{label}_median_based"] = round(d_time_robust / label_time_robust, 1) \
                        if (label_time_robust and d_time_robust and label_time_robust > 0) else None

                if "C" in policies and "D" in policies:
                    for metric in metric_names:
                        entry[f"{metric}_test"] = entry.get(f"{metric}_test_C_vs_D")
                    entry["speedup_D_over_C_median_based_legacy"] = entry.get("speedup_D_over_C_median_based")

                scenario_summary.append(entry)

                summary_bits = "  |  ".join(
                    f"{label} cov={entry[f'{label}_coverage_pct']}% "
                    f"delay={entry[f'{label}_avg_relay_delay_steps']} "
                    f"energy={entry[f'{label}_avg_relay_energy_pct']} "
                    f"selector(median)={entry[f'{label}_median_selector_time_ms']}ms"
                    for label in policies
                )
                print(f"\n[terrain={terrain} wind={wind_mode}]  {summary_bits}")
    finally:
        a20.set_wind_mode("static")   
        a20.set_terrain_scenario("benapole_delta")

    if out_csv and all_rows:
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)

    print(f"\n=== TERRAIN x WIND GENERALIZATION SWEEP: {', '.join(policies)} ===")
    print(json.dumps(scenario_summary, indent=2, default=str))
    return all_rows, scenario_summary


if __name__ == "__main__":
    import sys

    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    num_seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    seeds = list(range(1, num_seeds + 1))
    quick_seeds = seeds[:5]

    print("\n=== B vs C: FEASIBLE-NEAREST vs CORRECTED RELAYSCORE ===")
    rows, summary = run_comparison(
        seeds, max_steps=steps, out_csv="fleet_relay_comparison_fixed.csv"
    )
    print(json.dumps(summary, indent=2, default=str))

    print("\n=== GENERATING DECISION BREAKDOWN CHART ===")
    state_metrics = simulate_fleet(1, select_best_relay, max_steps=steps)
    d_log = state_metrics.get("decision_log", [])
    if d_log:
        interesting_logs = [x for x in d_log if len(x["candidates_diag"]) > 1]
        target_log = interesting_logs[0] if interesting_logs else d_log[0]
        plot_decision_breakdown(
            target_log,
            save_path="decision_breakdown_fixed.png"
        )
    else:
        print("ℹ️ No relay decisions occurred in this run to plot.")

    print("\n=== SENSITIVITY: W_COVERAGE_LOSS ===")
    sens_results = run_sensitivity_analysis(
        "W_COVERAGE_LOSS",
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        seeds=quick_seeds,
        max_steps=steps,
        out_csv="sensitivity_coverage_weight_fixed.csv",
    )
    plot_sensitivity(
        sens_results,
        "W_COVERAGE_LOSS",
        metric="coverage_pct",
        save_path="sensitivity_coverage_weight_fixed.png",
    )

    print("\n=== THREE-WAY: A original | B feasible-nearest | C corrected RelayScore ===")
    abc_rows, abc_summary = run_three_way_comparison(
        seeds,
        max_steps=steps,
        out_csv="abc_comparison_fixed.csv",
    )
    print(json.dumps(abc_summary, indent=2, default=str))

    print("\n=== FOUR-WAY: A original | B nearest | C RelayScore | D fuzzy-MADM (paper-adapted) ===")
    abcd_rows, abcd_summary = run_four_way_comparison(
        seeds,
        max_steps=steps,
        out_csv="abcd_comparison_with_fuzzy.csv",
    )

    print("\n=== RULE-BASE-CIRCULARITY ROBUSTNESS CHECK: C vs D (shared weights) vs D-indep (independent weights) ===")
    d_robustness_rows, d_robustness_summary = run_policy_d_weight_robustness_check(
        seeds,
        max_steps=steps,
        out_csv="policy_d_weight_robustness_check.csv",
    )

    print("\n=== INDEPENDENT RULE-BASE PANEL: C vs D-indep / indep2 / indep3 / indep4 ===")
    indep_panel_rows, indep_panel_summary = run_indep_rulebase_panel_check(
        seeds,
        max_steps=steps,
        out_csv="indep_rulebase_panel.csv",
    )

    print("\n=== FLEET-SIZE SENSITIVITY: RelayScore (C) vs true FIS (D) across N = 2,4,8,16 drones ===")
    fleet_rows, fleet_scaling_summary = run_fleet_size_sensitivity_analysis(
        quick_seeds,
        fleet_sizes=((1, 1), (2, 2), (4, 4), (8, 8)),
        max_steps=steps,
        out_csv="fleet_size_sensitivity.csv",
    )
    plot_fleet_size_scaling(fleet_scaling_summary, save_path="fleet_size_scaling.png")

    print("\n=== TERRAIN x WIND GENERALIZATION: RelayScore (C) vs true FIS (D) across "
          "4 terrains x {static, dynamic, stochastic} wind ===")
    terrain_wind_rows, terrain_wind_summary = run_terrain_wind_generalization_analysis(
        seeds,
        max_steps=steps,
        out_csv="terrain_wind_generalization.csv",
    )
    a20.set_terrain_scenario("benapole_delta")  
    print(json.dumps(abcd_summary, indent=2, default=str))