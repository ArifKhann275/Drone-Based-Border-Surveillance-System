# -*- coding: utf-8 -*-
"""
tune_multiobjective.py

Tunes T, V, and LARS (Policy I) using a multi-objective cost function:
Cost = W_COV * (100 - coverage) + W_DLY * delay + W_ENG * energy
This addresses reviewer concerns about tuning asymmetry (tuning baselines 
only for coverage vs comparing their delay/energy).

Usage:
    python -u tune_multiobjective.py T
    python -u tune_multiobjective.py V
    python -u tune_multiobjective.py LARS
"""

import sys
import json
import time
import numpy as np
from scipy.optimize import differential_evolution

import a2e_relay_fixed_final_51 as relay

TRAIN_SEEDS = list(range(101, 151))
MAX_STEPS = 500

# =============================================================================
# MULTI-OBJECTIVE WEIGHTS (Adjust these beforehand if needed)
# Currently: 1% coverage loss penalty == 2 steps delay penalty == 2% energy penalty
# =============================================================================
W_COV = 1.0
W_DLY = 0.5
W_ENG = 0.5

POPSIZE = 8      
MAXITER = 12     
DE_SEED = 42

def to_weight_dict_tv(x):
    x = np.asarray(x, dtype=float)
    s = x.sum()
    if s <= 1e-12:
        x, s = np.ones(4), 4.0
    return {k: float(v / s) for k, v in zip(["distance", "battery_margin", "coverage_loss", "relay_delay"], x)}

def to_weight_dict_lars(x):
    x = np.asarray(x, dtype=float)
    s = x.sum()
    if s <= 1e-12:
        x, s = np.ones(3), 3.0
    return {
        "W_TRAVEL_ENERGY": float(x[0]/s),
        "W_COVERAGE_LOSS": float(x[1]/s),
        "W_RELAY_DELAY": float(x[2]/s)
    }

def make_selector(policy_key, wd):
    if policy_key == "T":
        return lambda n, d, g, de, s: relay.select_topsis_relay(n, d, g, de, s, weights=wd)
    if policy_key == "V":
        return lambda n, d, g, de, s: relay.select_vikor_relay(n, d, g, de, s, weights=wd, v=0.5)
    if policy_key == "LARS":
        return lambda n, d, g, de, s: relay.select_best_relay_coverage_aware(n, d, g, de, s, weights=wd)
    raise ValueError(policy_key)

def evaluate_fleet(policy_key, wd, seeds):
    sel = make_selector(policy_key, wd)
    covs, dlys, engs = [], [], []
    for s in seeds:
        m = relay.simulate_fleet(s, relay_selector=sel, max_steps=MAX_STEPS)
        covs.append(m["coverage_pct"])
        dlys.append(m["avg_relay_delay_steps"])
        engs.append(m["avg_relay_energy_pct"])
    return float(np.mean(covs)), float(np.mean(dlys)), float(np.mean(engs))

def objective(x, policy_key):
    if policy_key in ["T", "V"]:
        wd = to_weight_dict_tv(x)
    else:
        wd = to_weight_dict_lars(x)
        
    cov, dly, eng = evaluate_fleet(policy_key, wd, TRAIN_SEEDS)
    
    # Minimize this multi-objective cost
    cost = (100.0 - cov) * W_COV + (dly * W_DLY) + (eng * W_ENG)
    return cost

def tune(policy_key):
    print(f"\n=== Multi-Objective Tuning {policy_key} (popsize={POPSIZE}, maxiter={MAXITER}) ===")
    print(f"Cost = {W_COV}*(100-Cov) + {W_DLY}*(Delay) + {W_ENG}*(Energy)")
    
    t0 = time.time()
    num_params = 3 if policy_key == "LARS" else 4
    
    res = differential_evolution(
        objective, [(0.0, 1.0)] * num_params, args=(policy_key,),
        maxiter=MAXITER, popsize=POPSIZE, seed=DE_SEED,
        polish=False, tol=0, init="latinhypercube", disp=True,
    )
    
    wd = to_weight_dict_lars(res.x) if policy_key == "LARS" else to_weight_dict_tv(res.x)
    best_cov, best_dly, best_eng = evaluate_fleet(policy_key, wd, TRAIN_SEEDS)
    
    out = {
        "policy": policy_key,
        "weights": wd,
        "train_cov": best_cov,
        "train_dly": best_dly,
        "train_eng": best_eng,
        "multi_obj_cost": float(res.fun),
        "nfev": int(res.nfev),
        "seconds": round(time.time() - t0, 1),
    }
    
    fname = f"tuned_{policy_key}_multiobj.json"
    with open(fname, "w") as f:
        json.dump(out, f, indent=4)
        
    print(f"\nBest {policy_key} weights: {wd}")
    print(f"Train Cov: {best_cov:.2f}%, Delay: {best_dly:.2f}, Energy: {best_eng:.2f}%")
    print(f"Saved -> {fname}")

if __name__ == "__main__":
    keys = [a.upper() for a in sys.argv[1:]]
    if not keys:
        print("Usage: python tune_multiobjective.py T V LARS")
    for k in keys:
        tune(k)