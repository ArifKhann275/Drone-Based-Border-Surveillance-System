"""
policy_d_true_fis.py
A GENUINE rule-based Fuzzy Inference System (Mamdani-type FIS) for Policy D.
(Optimized with ControlSystemSimulation Caching to prevent computational overhead)
"""

import itertools
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

FUZZY_ATTR_WEIGHTS = {
    "distance": 0.30,
    "battery_margin": 0.30,
    "coverage_loss": 0.25,
    "relay_delay": 0.15,
}

# ── FIXED: Made the independent weights significantly different to ensure 
# real rule variation, not just an identical mathematical overlap.
INDEPENDENT_RULE_WEIGHTS = {
    "distance": 0.15,
    "battery_margin": 0.20,
    "coverage_loss": 0.50,    # Prioritizing coverage heavily
    "relay_delay": 0.15,
}

BATTERY_MARGIN_UNIVERSE = np.linspace(0, 60, 121)     
DISTANCE_UNIVERSE       = np.linspace(0, 40, 81)       
COVERAGE_LOSS_UNIVERSE  = np.linspace(0, 0.06, 121)    
RELAY_DELAY_UNIVERSE    = np.linspace(0, 40, 81)       
SUITABILITY_UNIVERSE    = np.linspace(0, 100, 101)     

_GOODNESS = {
    "battery_margin": {"LOW": 0, "MEDIUM": 1, "HIGH": 2},
    "distance":       {"FAR": 0, "MEDIUM": 1, "NEAR": 2},
    "coverage_loss":  {"HIGH": 0, "MEDIUM": 1, "LOW": 2},
    "relay_delay":    {"LONG": 0, "MEDIUM": 1, "SHORT": 2},
}

DEFAULT_RULE_WEIGHTS = dict(FUZZY_ATTR_WEIGHTS)

def _rule_consequent(batt_lbl, dist_lbl, cov_lbl, delay_lbl, weights):
    score = (
        weights["battery_margin"] * _GOODNESS["battery_margin"][batt_lbl]
        + weights["distance"] * _GOODNESS["distance"][dist_lbl]
        + weights["coverage_loss"] * _GOODNESS["coverage_loss"][cov_lbl]
        + weights["relay_delay"] * _GOODNESS["relay_delay"][delay_lbl]
    )
    max_score = 2.0 * sum(weights.values())
    frac = score / max_score if max_score > 0 else 0.0
    if frac < 1 / 3: return "LOW"
    if frac < 2 / 3: return "MEDIUM"
    return "HIGH"

def _weights_key(weights):
    return tuple(sorted((k, round(float(v), 6)) for k, v in weights.items()))

_CONTROL_SYSTEM_CACHE = {}

def _get_control_system(weights):
    key = _weights_key(weights)
    if key in _CONTROL_SYSTEM_CACHE:
        return _CONTROL_SYSTEM_CACHE[key]

    # Create fresh independent variables for THIS specific rule base
    battery_margin = ctrl.Antecedent(BATTERY_MARGIN_UNIVERSE, f"battery_margin_{id(weights)}")
    distance       = ctrl.Antecedent(DISTANCE_UNIVERSE, f"distance_{id(weights)}")
    coverage_loss  = ctrl.Antecedent(COVERAGE_LOSS_UNIVERSE, f"coverage_loss_{id(weights)}")
    relay_delay    = ctrl.Antecedent(RELAY_DELAY_UNIVERSE, f"relay_delay_{id(weights)}")
    suitability    = ctrl.Consequent(SUITABILITY_UNIVERSE, f"suitability_{id(weights)}")

    for var in [battery_margin, coverage_loss]:
        lo, hi = var.universe[0], var.universe[-1]
        mid = (lo + hi) / 2.0
        var["LOW"] = fuzz.trimf(var.universe, [lo, lo, mid])
        var["MEDIUM"] = fuzz.trimf(var.universe, [lo, mid, hi])
        var["HIGH"] = fuzz.trimf(var.universe, [mid, hi, hi])

    for var in [distance]:
        lo, hi = var.universe[0], var.universe[-1]
        mid = (lo + hi) / 2.0
        var["NEAR"] = fuzz.trimf(var.universe, [lo, lo, mid])
        var["MEDIUM"] = fuzz.trimf(var.universe, [lo, mid, hi])
        var["FAR"] = fuzz.trimf(var.universe, [mid, hi, hi])

    for var in [relay_delay]:
        lo, hi = var.universe[0], var.universe[-1]
        mid = (lo + hi) / 2.0
        var["SHORT"] = fuzz.trimf(var.universe, [lo, lo, mid])
        var["MEDIUM"] = fuzz.trimf(var.universe, [lo, mid, hi])
        var["LONG"] = fuzz.trimf(var.universe, [mid, hi, hi])

    suitability["LOW"]    = fuzz.trimf(suitability.universe, [0, 0, 50])
    suitability["MEDIUM"] = fuzz.trimf(suitability.universe, [0, 50, 100])
    suitability["HIGH"]   = fuzz.trimf(suitability.universe, [50, 100, 100])

    rules = []
    for b, d, c, dl in itertools.product(["LOW", "MEDIUM", "HIGH"], ["NEAR", "MEDIUM", "FAR"], ["LOW", "MEDIUM", "HIGH"], ["SHORT", "MEDIUM", "LONG"]):
        consequent_label = _rule_consequent(b, d, c, dl, weights)
        antecedent = battery_margin[b] & distance[d] & coverage_loss[c] & relay_delay[dl]
        rules.append(ctrl.Rule(antecedent, suitability[consequent_label]))

    system = ctrl.ControlSystem(rules)
    _CONTROL_SYSTEM_CACHE[key] = (system, battery_margin, distance, coverage_loss, relay_delay)
    return _CONTROL_SYSTEM_CACHE[key]


# --- OPTIMIZATION FIX: Cache the simulation object to prevent building 
# the computational graph on every single candidate evaluation ---
_SIM_CACHE = {}

def _run_fis(batt_val, dist_val, cov_val, delay_val, weights=None):
    if weights is None:
        weights = FUZZY_ATTR_WEIGHTS
        
    key = _weights_key(weights)
    
    # Create the ControlSystemSimulation ONLY ONCE and cache it
    if key not in _SIM_CACHE:
        system, b_var, d_var, c_var, dl_var = _get_control_system(weights)
        sim = ctrl.ControlSystemSimulation(system)
        out_label = list(system.consequents)[0].label
        _SIM_CACHE[key] = (sim, b_var, d_var, c_var, dl_var, out_label)

    # Retrieve the cached simulation object
    sim, b_var, d_var, c_var, dl_var, out_label = _SIM_CACHE[key]
    
    # Just feed inputs and compute (Microsecond level operation)
    sim.input[b_var.label] = float(np.clip(batt_val, BATTERY_MARGIN_UNIVERSE[0], BATTERY_MARGIN_UNIVERSE[-1]))
    sim.input[d_var.label] = float(np.clip(dist_val, DISTANCE_UNIVERSE[0], DISTANCE_UNIVERSE[-1]))
    sim.input[c_var.label] = float(np.clip(cov_val, COVERAGE_LOSS_UNIVERSE[0], COVERAGE_LOSS_UNIVERSE[-1]))
    sim.input[dl_var.label] = float(np.clip(delay_val, RELAY_DELAY_UNIVERSE[0], RELAY_DELAY_UNIVERSE[-1]))
    
    sim.compute()
    return float(sim.output[out_label])


def select_fuzzy_relay_true_fis(needer, drones, g, detected, step, weights=None, resolution=101):
    from a2e_relay_fixed_final_51 import _feasible_candidate_pool, _manhattan, _coverage_loss_for_candidate

    feasible = _feasible_candidate_pool(needer, drones)
    if not feasible:
        return None, []

    rows = []
    for cand, travel_energy, battery_margin_val in feasible:
        dist = _manhattan(cand["r"], cand["c"], needer["r"], needer["c"])
        relay_delay_val = dist
        coverage_loss_val, _raw, _tgap, _meta = _coverage_loss_for_candidate(
            cand, needer, g, detected, step, relay_delay_val
        )
        rows.append({
            "cand": cand, "distance": dist, "travel_energy": travel_energy,
            "battery_margin": battery_margin_val, "coverage_loss": coverage_loss_val,
            "relay_delay": relay_delay_val,
        })

    if len(rows) == 1:
        only = rows[0]
        suitability_score = _run_fis(
            only["battery_margin"], only["distance"], only["coverage_loss"], only["relay_delay"], weights=weights
        )
        diag = {
            "candidate_id": only["cand"]["id"], "distance": only["distance"],
            "travel_energy": round(only["travel_energy"], 3), "battery_margin": round(only["battery_margin"], 2),
            "coverage_loss": round(only["coverage_loss"], 6), "relay_delay": only["relay_delay"],
            "fis_suitability": round(suitability_score, 2), "note": "only feasible candidate",
        }
        return only["cand"], [diag]

    scored = []
    for r in rows:
        suitability_score = _run_fis(
            r["battery_margin"], r["distance"], r["coverage_loss"], r["relay_delay"], weights=weights
        )
        diag = {
            "candidate_id": r["cand"]["id"], "role": r["cand"]["role"], "distance": r["distance"],
            "travel_energy": round(r["travel_energy"], 3), "battery_margin": round(r["battery_margin"], 2),
            "coverage_loss": round(r["coverage_loss"], 6), "relay_delay": r["relay_delay"],
            "fis_suitability": round(suitability_score, 2),
        }
        scored.append((r["cand"], suitability_score, diag))

    scored.sort(key=lambda x: -x[1])
    return scored[0][0], [d for _, _, d in scored]

def select_fuzzy_relay_true_fis_indep_weights(needer, drones, g, detected, step, resolution=101):
    return select_fuzzy_relay_true_fis(
        needer, drones, g, detected, step,
        weights=INDEPENDENT_RULE_WEIGHTS, resolution=resolution,
    )