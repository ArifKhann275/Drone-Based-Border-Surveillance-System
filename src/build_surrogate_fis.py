# -*- coding: utf-8 -*-
"""
build_surrogate_fis.py  --  offline LUT builder for Policy D2.

Run this whenever policy_d_true_fis.py's rule base, universes, or membership
functions change. Every .npz it writes carries the rule-base fingerprint it
was built from; surrogate_fis.py refuses to load a LUT whose fingerprint does
not match the live rule base, so a stale table can never silently corrupt a
D2-vs-D comparison.

Usage
-----
    python build_surrogate_fis.py                 # both profiles, the resolution
                                                  # sweep of the paper's Table XVIII
    python build_surrogate_fis.py 9               # only Rgrid=9
    python build_surrogate_fis.py 2 3 4 5 7 9 11 13

Note on build time: with the vectorized Mamdani backend a 9^4 grid takes well
under a second, not the ~6 minutes reported in the old Table XVIII. Those
build times were scikit-fuzzy's per-call overhead, not the cost of evaluating
the rule base -- if you keep that table, regenerate it with this script and
say which backend produced it.
"""

import os
import sys
import time

import numpy as np

import policy_d_true_fis as fis

LUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "luts")

# profile name -> file label (must match surrogate_fis._LABEL_FOR_PROFILE)
WEIGHT_SETS = {
    "shared": fis.FUZZY_ATTR_WEIGHTS,
    "indep": fis.INDEPENDENT_RULE_WEIGHTS,
}

DEFAULT_RESOLUTIONS = (2, 3, 4, 5, 7, 9, 11, 13)


def build_lut(weights, resolution, label, out_dir=LUT_DIR, backend="vectorized",
              verbose=True):
    """Evaluate the true FIS on a uniform 4-D grid and save it with its
    rule-base fingerprint."""
    os.makedirs(out_dir, exist_ok=True)

    grid_b = np.linspace(fis.BATTERY_MARGIN_UNIVERSE[0],
                         fis.BATTERY_MARGIN_UNIVERSE[-1], resolution)
    grid_d = np.linspace(fis.DISTANCE_UNIVERSE[0],
                         fis.DISTANCE_UNIVERSE[-1], resolution)
    grid_c = np.linspace(fis.COVERAGE_LOSS_UNIVERSE[0],
                         fis.COVERAGE_LOSS_UNIVERSE[-1], resolution)
    grid_dl = np.linspace(fis.RELAY_DELAY_UNIVERSE[0],
                          fis.RELAY_DELAY_UNIVERSE[-1], resolution)

    out = np.empty((resolution,) * 4, dtype=float)
    t0 = time.perf_counter()
    for i, b in enumerate(grid_b):
        for j, d in enumerate(grid_d):
            for k, c in enumerate(grid_c):
                for m, dl in enumerate(grid_dl):
                    out[i, j, k, m] = fis._run_fis(b, d, c, dl, weights=weights,
                                                   backend=backend)
    build_time = time.perf_counter() - t0

    fp = fis.rule_base_fingerprint(weights)
    prof = fis._resolve_profile(weights)
    path = os.path.join(out_dir, f"lut_{label}_res{resolution}.npz")
    np.savez_compressed(
        path,
        grid_b=grid_b, grid_d=grid_d, grid_c=grid_c, grid_dl=grid_dl,
        output=out,
        resolution=np.array(resolution),
        build_time_sec=np.array(build_time),
        rule_base_fingerprint=np.array(fp),
        profile=np.array(prof["name"]),
        backend=np.array(backend),
    )
    if verbose:
        size_kb = os.path.getsize(path) / 1024.0
        print(f"  [{label}] Rgrid={resolution:<3d} "
              f"{resolution**4:>7d} nodes  {build_time:7.2f} s  "
              f"{size_kb:7.1f} KB  -> {os.path.basename(path)}")
    return path


def fidelity(weights, resolution, label, n=2000, seed=12345, backend="vectorized"):
    """Score fidelity at randomly sampled points that are NOT grid nodes."""
    from surrogate_fis import _run_fis_surrogate

    rng = np.random.default_rng(seed)
    pts = np.column_stack([
        rng.uniform(fis.BATTERY_MARGIN_UNIVERSE[0], fis.BATTERY_MARGIN_UNIVERSE[-1], n),
        rng.uniform(fis.DISTANCE_UNIVERSE[0], fis.DISTANCE_UNIVERSE[-1], n),
        rng.uniform(fis.COVERAGE_LOSS_UNIVERSE[0], fis.COVERAGE_LOSS_UNIVERSE[-1], n),
        rng.uniform(fis.RELAY_DELAY_UNIVERSE[0], fis.RELAY_DELAY_UNIVERSE[-1], n),
    ])
    true = np.array([fis._run_fis(*p, weights=weights, backend=backend) for p in pts])
    approx = np.array([_run_fis_surrogate(*p, weights=weights, resolution=resolution)
                       for p in pts])
    err = approx - true
    return {
        "resolution": resolution,
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mae": float(np.mean(np.abs(err))),
        "max_err": float(np.max(np.abs(err))),
    }


def decision_agreement(weights, resolution, k=6, trials=2000, seed=999,
                       backend="vectorized"):
    """Top-1 agreement and Kendall tau over synthetic candidate pools."""
    from scipy.stats import kendalltau
    from surrogate_fis import _run_fis_surrogate

    rng = np.random.default_rng(seed)
    top1, taus = 0, []
    for _ in range(trials):
        pool = np.column_stack([
            rng.uniform(fis.BATTERY_MARGIN_UNIVERSE[0], fis.BATTERY_MARGIN_UNIVERSE[-1], k),
            rng.uniform(fis.DISTANCE_UNIVERSE[0], fis.DISTANCE_UNIVERSE[-1], k),
            rng.uniform(fis.COVERAGE_LOSS_UNIVERSE[0], fis.COVERAGE_LOSS_UNIVERSE[-1], k),
            rng.uniform(fis.RELAY_DELAY_UNIVERSE[0], fis.RELAY_DELAY_UNIVERSE[-1], k),
        ])
        t = [fis._run_fis(*p, weights=weights, backend=backend) for p in pool]
        a = [_run_fis_surrogate(*p, weights=weights, resolution=resolution) for p in pool]
        if int(np.argmax(t)) == int(np.argmax(a)):
            top1 += 1
        tau = kendalltau(t, a).correlation
        if tau == tau:                      # skip NaN on degenerate pools
            taus.append(tau)
    return {"resolution": resolution, "top1_agreement": top1 / trials,
            "kendall_tau": float(np.mean(taus)) if taus else float("nan")}


def main(argv):
    resolutions = tuple(int(a) for a in argv[1:]) or DEFAULT_RESOLUTIONS
    print(f"LUT output dir : {LUT_DIR}")
    print(f"FIS backend    : vectorized (policy_d_true_fis)")
    for label, weights in WEIGHT_SETS.items():
        prof = fis._resolve_profile(weights)
        print(f"\nProfile '{prof['name']}'  fingerprint={fis.rule_base_fingerprint(weights)}")
        for res in resolutions:
            build_lut(weights, res, label)

    print("\nFidelity / decision agreement for the 'shared' (operating) profile:")
    print(f"{'Rgrid':>6} {'RMSE':>9} {'MAE':>9} {'MaxErr':>9} {'Top-1':>8} {'tau':>8}")
    for res in resolutions:
        f = fidelity(fis.FUZZY_ATTR_WEIGHTS, res, "shared", n=1500)
        a = decision_agreement(fis.FUZZY_ATTR_WEIGHTS, res, trials=800)
        print(f"{res:>6} {f['rmse']:>9.4f} {f['mae']:>9.4f} {f['max_err']:>9.4f} "
              f"{a['top1_agreement']*100:>7.2f}% {a['kendall_tau']:>8.4f}")
    print("\nDone. These numbers regenerate the paper's LUT-resolution table.")


if __name__ == "__main__":
    main(sys.argv)
