# -*- coding: utf-8 -*-
"""
tune_auction_lambda.py

Tunes the single price parameter lam of the marginal-cost auction baseline
(auction_baseline.py) on the TRAINING block (seeds 101-150) ONLY.

Because there is just one parameter, a grid is used instead of DE; the whole
grid doubles as a coverage/delay/energy trade-off curve you can show.

PRE-COMMITTED SELECTION RULE (written before any dev/fresh result exists):
    choose the lam with the highest mean TRAINING coverage
    (same objective as Policy J, T and V); ties -> smaller lam.
The full grid is saved too, so a different operating point can be discussed
openly, but the rule above decides which lam is evaluated on the fresh block.

Usage:   python -u tune_auction_lambda.py
Output:  tuned_AUC_lite.json
"""

import json
import time
import numpy as np

import a2e_relay_fixed_final_51 as relay
from auction_baseline import make_auction_selector

TRAIN_SEEDS = list(range(101, 151))
MAX_STEPS = 500
LAM_GRID = [0.0, 5.0, 10.0, 25.0, 50.0, 100.0, 200.0, 400.0]


def run_block(selector, seeds):
    covs, dels, ens = [], [], []
    for s in seeds:
        m = relay.simulate_fleet(s, relay_selector=selector, max_steps=MAX_STEPS)
        covs.append(m["coverage_pct"])
        dels.append(m["avg_relay_delay_steps"])
        ens.append(m["avg_relay_energy_pct"])
    return float(np.mean(covs)), float(np.mean(dels)), float(np.mean(ens))


def sanity_check():
    """lam must change behaviour, and lam=0 should be close to nearest-feasible B."""
    seeds = TRAIN_SEEDS[:4]
    a = run_block(make_auction_selector(0.0), seeds)
    b = run_block(make_auction_selector(400.0), seeds)
    n = run_block(relay.select_nearest_relay, seeds)
    print(f"[sanity] lam=0   -> cov {a[0]:.3f}, delay {a[1]:.2f}, energy {a[2]:.2f}")
    print(f"[sanity] lam=400 -> cov {b[0]:.3f}, delay {b[1]:.2f}, energy {b[2]:.2f}")
    print(f"[sanity] B (nearest) -> cov {n[0]:.3f}, delay {n[1]:.2f}, energy {n[2]:.2f}")
    if all(abs(a[i] - b[i]) < 1e-9 for i in range(3)):
        raise SystemExit("ABORT: lam has no effect on the outcome.")


def main():
    sanity_check()
    rows = []
    print(f"\n{'lam':>8}{'train cov':>12}{'delay':>9}{'energy':>9}{'minutes':>9}")
    for lam in LAM_GRID:
        t0 = time.time()
        cov, dl, en = run_block(make_auction_selector(lam), TRAIN_SEEDS)
        rows.append({"lam": lam, "train_cov": cov, "train_delay": dl, "train_energy": en})
        print(f"{lam:>8.1f}{cov:>12.3f}{dl:>9.2f}{en:>9.2f}{(time.time() - t0) / 60:>9.1f}", flush=True)

    best = max(rows, key=lambda r: (round(r["train_cov"], 9), -r["lam"]))
    out = {
        "policy": "AUC",
        "lam": best["lam"],
        "selection_rule": "max mean training coverage over the grid; ties -> smaller lam",
        "train_seeds": [TRAIN_SEEDS[0], TRAIN_SEEDS[-1]],
        "grid": rows,
    }
    with open("tuned_AUC_lite.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSelected lam = {best['lam']} (train coverage {best['train_cov']:.3f}). "
          f"Saved -> tuned_AUC_lite.json")


if __name__ == "__main__":
    main()
