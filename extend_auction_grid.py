# -*- coding: utf-8 -*-
"""
extend_auction_grid.py

The first lam grid (0 ... 400) ended with the best training coverage at its
upper edge (400), and coverage was still rising. This script extends the grid
upward on the TRAINING block (seeds 101-150) only, re-using the saved results
for the old values, and applies the SAME pre-committed selection rule:

    choose the lam with the highest mean training coverage; ties -> smaller lam.

Nothing from the dev or fresh blocks is used. The old file is backed up as
tuned_AUC_lite_grid400_backup.json. Results are saved after every lam, so an
interrupted run loses nothing; re-running skips values that are already done.

lam = 1e6 stands for the "coverage-loss first, energy second" limit.

Usage:   python -u extend_auction_grid.py
"""

import os
import json
import shutil
import time

import a2e_relay_fixed_final_51 as relay  # noqa: F401  (loads terrain once)
from auction_baseline import make_auction_selector
from tune_auction_lambda import run_block, TRAIN_SEEDS

NEW_LAMS = [800.0, 1600.0, 3200.0, 6400.0, 1000000.0]
FILE = "tuned_AUC_lite.json"
BACKUP = "tuned_AUC_lite_grid400_backup.json"


def pick_best(grid):
    return max(grid, key=lambda r: (round(r["train_cov"], 9), -r["lam"]))


def save(data):
    data["grid"] = sorted(data["grid"], key=lambda r: r["lam"])
    best = pick_best(data["grid"])
    data["lam"] = best["lam"]
    with open(FILE, "w") as f:
        json.dump(data, f, indent=2)
    return best


def main():
    with open(FILE) as f:
        data = json.load(f)
    if not os.path.exists(BACKUP):
        shutil.copy(FILE, BACKUP)
    data["grid_extension_note"] = (
        "Original grid ended at lam=400 with coverage still rising; extended on the "
        "training block only (seeds 101-150) with the same selection rule."
    )
    done = {r["lam"] for r in data["grid"]}

    print(f"{'lam':>10}{'train cov':>12}{'delay':>9}{'energy':>9}{'minutes':>9}")
    for r in sorted(data["grid"], key=lambda r: r["lam"]):
        print(f"{r['lam']:>10.0f}{r['train_cov']:>12.3f}{r['train_delay']:>9.2f}"
              f"{r['train_energy']:>9.2f}{'(saved)':>9}")

    for lam in NEW_LAMS:
        if lam in done:
            continue
        t0 = time.time()
        cov, dl, en = run_block(make_auction_selector(lam), TRAIN_SEEDS)
        data["grid"].append({"lam": lam, "train_cov": cov, "train_delay": dl, "train_energy": en})
        save(data)
        print(f"{lam:>10.0f}{cov:>12.3f}{dl:>9.2f}{en:>9.2f}{(time.time() - t0) / 60:>9.1f}", flush=True)

    best = save(data)
    print(f"\nSelected lam = {best['lam']:.0f} (train coverage {best['train_cov']:.3f}); saved -> {FILE}")
    if best["lam"] == max(r["lam"] for r in data["grid"]):
        print("NOTE: the best value is again the largest in the grid, i.e. the coverage-first limit; "
              "report it as such.")


if __name__ == "__main__":
    main()
