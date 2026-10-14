# -*- coding: utf-8 -*-
"""
tune_baselines.py (v3)

Tunes attribute weights of TOPSIS (T), VIKOR (V), and Policy I with
differential evolution on the TRAINING block (seeds 101-150), maximizing
mean coverage -- the SAME objective, seed block, and budget convention
used for Policy J (coverage-only; popsize=8, maxiter=12, seed=42; i.e.
the "lite" budget already used for T/V, not J's full 940-eval budget).

Usage:
    python -u tune_baselines.py T      # tune TOPSIS only
    python -u tune_baselines.py V      # tune VIKOR only
    python -u tune_baselines.py I      # tune Policy I only (NEW)
    python -u tune_baselines.py        # tune all three, one after another

Output: tuned_T_lite.json / tuned_V_lite.json / tuned_I_lite.json
(weights + budget actually used).

Policy I (RelayScore with Fix 1: absolute-scale coverage-loss
normalization) has only THREE attributes -- travel energy, coverage
loss, relay delay -- not four: battery margin only gates feasibility
in Policy I, it is never a ranked, weighted attribute (unlike Policy J,
where Fix 2 promotes it to a fourth ranked attribute). The DE search
space below is therefore 3-dimensional for I, same as it already was
for T and V's 4-dimensional case minus one -- NOT 4-dimensional; do not
add a battery weight for I, there is nothing in Policy I's own scoring
rule for a battery weight to multiply.

Changes vs. the earlier (T/V-only) version:
  * added Policy I as a third tunable policy, 3-parameter search space,
    using relay.select_best_relay_coverage_aware (the existing Policy I
    selector) with its own weight dict keys (W_TRAVEL_ENERGY,
    W_COVERAGE_LOSS, W_RELAY_DELAY -- same naming convention as Policy
    J's weight dict, just without W_BATTERY_MARGIN);
  * CRITERIA is now looked up per policy (4 keys for T/V, 3 for I)
    rather than being a single module-level constant, so to_weight_dict
    and the DE bounds automatically match each policy's own attribute
    count;
  * everything else (sanity check, polish=False, tol=0, nfev tracking,
    output JSON schema) is unchanged from v2, so tuned_I_lite.json is
    directly comparable to tuned_T_lite.json / tuned_V_lite.json.
"""

import sys
import json
import time
import numpy as np
from scipy.optimize import differential_evolution

import a2e_relay_fixed_final_51 as relay

TRAIN_SEEDS = list(range(101, 151))
MAX_STEPS = 500

# Criteria (and therefore DE dimensionality) per policy. T and V rank
# four attributes; Policy I ranks only three (no battery weight -- see
# module docstring).
CRITERIA = {
    "T": ("distance", "battery_margin", "coverage_loss", "relay_delay"),
    "V": ("distance", "battery_margin", "coverage_loss", "relay_delay"),
    "I": ("energy", "coverage_loss", "relay_delay"),
}

# Policy I's selector expects this weight-dict key naming (same
# convention as Policy J's own weight dict, minus W_BATTERY_MARGIN).
I_KEY_MAP = {
    "energy": "W_TRAVEL_ENERGY",
    "coverage_loss": "W_COVERAGE_LOSS",
    "relay_delay": "W_RELAY_DELAY",
}

POPSIZE = 8      # scipy multiplies this by the number of parameters -> 32/gen for T,V; 24/gen for I
MAXITER = 12
DE_SEED = 42


def to_weight_dict(policy_key, x):
    """Raw DE vector -> normalized dict with the keys the selector expects."""
    x = np.asarray(x, dtype=float)
    s = x.sum()
    n = len(CRITERIA[policy_key])
    if s <= 1e-12:
        x, s = np.ones(n), float(n)
    raw = {k: float(v / s) for k, v in zip(CRITERIA[policy_key], x)}
    if policy_key == "I":
        return {I_KEY_MAP[k]: v for k, v in raw.items()}
    return raw


def make_selector(policy_key, wd):
    if policy_key == "T":
        return lambda n, d, g, de, s: relay.select_topsis_relay(n, d, g, de, s, weights=wd)
    if policy_key == "V":
        return lambda n, d, g, de, s: relay.select_vikor_relay(n, d, g, de, s, weights=wd, v=0.5)
    if policy_key == "I":
        return lambda n, d, g, de, s: relay.select_best_relay_coverage_aware(n, d, g, de, s, weights=wd)
    raise ValueError(policy_key)


def mean_coverage(policy_key, wd, seeds):
    sel = make_selector(policy_key, wd)
    covs = [relay.simulate_fleet(s, relay_selector=sel, max_steps=MAX_STEPS)["coverage_pct"]
            for s in seeds]
    return float(np.mean(covs))


def objective(x, policy_key):
    return -mean_coverage(policy_key, to_weight_dict(policy_key, x), TRAIN_SEEDS)


def sanity_check(policy_key):
    """Abort if different weight vectors give identical results (weights ignored)."""
    seeds = TRAIN_SEEDS[:4]
    n = len(CRITERIA[policy_key])
    probes = []
    # one balanced-ish probe, then one probe per dimension pushed to ~0.9
    probes.append(tuple(round(1.0 / n, 4) for _ in range(n)))
    for i in range(n):
        p = [0.03] * n
        p[i] = 1.0 - 0.03 * (n - 1)
        probes.append(tuple(p))
    vals = [mean_coverage(policy_key, to_weight_dict(policy_key, p), seeds) for p in probes]
    print(f"[sanity {policy_key}] coverage under {len(probes)} probe weight vectors: "
          + ", ".join(f"{v:.3f}" for v in vals))
    if len({round(v, 6) for v in vals}) < 2:
        raise SystemExit(f"ABORT: {policy_key} output does not depend on the weights.")


def tune(policy_key):
    n = len(CRITERIA[policy_key])
    print(f"\n=== Tuning {policy_key} ({n} weights: {CRITERIA[policy_key]}; "
          f"popsize={POPSIZE}, maxiter={MAXITER}, seed={DE_SEED}) ===")
    sanity_check(policy_key)
    t0 = time.time()
    res = differential_evolution(
        objective, [(0.0, 1.0)] * n, args=(policy_key,),
        maxiter=MAXITER, popsize=POPSIZE, seed=DE_SEED,
        polish=False, tol=0, init="latinhypercube", disp=True,
    )
    wd = to_weight_dict(policy_key, res.x)
    out = {
        "policy": policy_key,
        "weights": wd,
        "train_cov": float(-res.fun),
        "nfev": int(res.nfev),
        "popsize": POPSIZE, "maxiter": MAXITER, "de_seed": DE_SEED,
        "train_seeds": [TRAIN_SEEDS[0], TRAIN_SEEDS[-1]],
        "seconds": round(time.time() - t0, 1),
    }
    fname = f"tuned_{policy_key}_lite.json"
    with open(fname, "w") as f:
        json.dump(out, f, indent=4)
    print(f"Best {policy_key} weights: {wd}")
    print(f"Train coverage {out['train_cov']:.3f}%, nfev={out['nfev']}, "
          f"elapsed {out['seconds']}s, saved -> {fname}")


if __name__ == "__main__":
    keys = [a.upper() for a in sys.argv[1:]] or ["T", "V", "I"]
    for k in keys:
        if k not in CRITERIA:
            raise SystemExit(f"unknown policy key {k!r}; choose from {sorted(CRITERIA)}")
        tune(k)