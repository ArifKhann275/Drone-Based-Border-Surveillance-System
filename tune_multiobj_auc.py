# -*- coding: utf-8 -*-
"""
tune_multiobj_auc.py
Tunes the AUC baseline using the same multi-objective cost function 
used for T, V, and LARS. Uses the predefined lambda grid.
"""

import json
import time
import numpy as np
import a2e_relay_fixed_final_51 as relay
from auction_baseline import make_auction_selector

TRAIN_SEEDS = list(range(101, 151))
MAX_STEPS = 500

# কস্ট ফাংশনের ওজন (T, V, LARS এর মতই)
W_COV = 1.0
W_DLY = 0.5
W_ENG = 0.5

# পেপারে উল্লেখিত ল্যাম্বডা গ্রিড
GRID = [0, 5, 10, 25, 50, 100, 200, 400, 800, 1600, 3200, 6400, 1000000]

def evaluate_fleet_auc(lam, seeds):
    sel = make_auction_selector(lam)
    covs, dlys, engs = [], [], []
    for s in seeds:
        m = relay.simulate_fleet(s, relay_selector=sel, max_steps=MAX_STEPS)
        covs.append(m["coverage_pct"])
        dlys.append(m["avg_relay_delay_steps"])
        engs.append(m["avg_relay_energy_pct"])
    return float(np.mean(covs)), float(np.mean(dlys)), float(np.mean(engs))

def tune_auc():
    print(f"=== Multi-Objective Tuning AUC ===")
    print(f"Cost = {W_COV}*(100-Cov) + {W_DLY}*(Delay) + {W_ENG}*(Energy)")
    
    best_cost = float('inf')
    best_lam = None
    results = {}
    
    t0 = time.time()
    for lam in GRID:
        cov, dly, eng = evaluate_fleet_auc(lam, TRAIN_SEEDS)
        cost = (100.0 - cov) * W_COV + (dly * W_DLY) + (eng * W_ENG)
        print(f"Lambda {lam:<7}: Cov={cov:.2f}%, Dly={dly:.2f}, Eng={eng:.2f}% -> Cost={cost:.4f}")
        
        if cost < best_cost:
            best_cost = cost
            best_lam = lam
            results = {
                "policy": "AUC",
                "lam": best_lam,
                "train_cov": cov,
                "train_dly": dly,
                "train_eng": eng,
                "multi_obj_cost": best_cost,
                "seconds": round(time.time() - t0, 1)
            }
            
    with open("tuned_AUC_multiobj.json", "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nSaved best AUC Lambda: {best_lam} with Cost: {best_cost:.4f} to tuned_AUC_multiobj.json")

if __name__ == "__main__":
    tune_auc()