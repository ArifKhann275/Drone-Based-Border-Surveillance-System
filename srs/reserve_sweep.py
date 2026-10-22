# -*- coding: utf-8 -*-
"""
reserve_sweep.py  --  choose and justify the reserve opportunity-cost weight.

    python reserve_sweep.py --pool                 # does the pool ever run dry?
    python reserve_sweep.py --calibrate            # measure the real L_i scale
    python reserve_sweep.py                        # sweep lambda, default grid
    python reserve_sweep.py 0 0.1 0.2 0.3 0.5      # sweep a custom grid
    python reserve_sweep.py --seeds 30 --steps 500 # sample size / mission length

WHY A SWEEP AND NOT A TUNED CONSTANT
------------------------------------
lambda is a modelling parameter, not a fitted one. If you report a single
hand-picked value, a reviewer reads it as tuning until RelayScore wins. A
sweep answers the honest question instead -- "is the conclusion stable across
the plausible range of this parameter, or does it hinge on one value?" -- and
costs you one extra table.

lambda = 0 reproduces the pre-patch model exactly, so the first row of the
sweep is always your old result. That makes the effect of the term visible
rather than baked in.

Output: reserve_lambda_sweep.csv plus a printed summary.
"""

import csv
import sys

import numpy as np

import a2e_relay_fixed_final_51 as relay

DEFAULT_LAMBDAS = (0.0, 0.05, 0.10, 0.20, 0.35, 0.50)
POLICIES = ("C", "D", "T", "V")


def _arg(flag, default, cast=int):
    if flag in sys.argv:
        return cast(sys.argv[sys.argv.index(flag) + 1])
    return default


# ══════════════════════════════════════════════════════════════════════════
# Calibration: what scale does L_i actually live on?
# ══════════════════════════════════════════════════════════════════════════
def calibrate(seeds=5, steps=500):
    """Measure the in-situ distribution of the marginal coverage loss.

    Use this to justify COVERAGE_LOSS_UNIVERSE in policy_d_true_fis.py and to
    place lambda on the same scale. Run it again whenever you change the
    staleness cap, the threat/border penalties, the grid, or the fleet size.
    """
    rec = {"patrol": [], "reserve": []}
    original = relay._coverage_loss_for_candidate
    prev_flag = relay.RESERVE_OPPORTUNITY_ENABLED
    relay.RESERVE_OPPORTUNITY_ENABLED = False      # measure the base term only

    def spy(cand, needer, g, detected, step, relay_delay, fleet=None):
        out = original(cand, needer, g, detected, step, relay_delay, fleet=fleet)
        rec["reserve" if cand.get("role") == "idle_reserve" else "patrol"].append(out[0])
        return out

    relay._coverage_loss_for_candidate = spy
    try:
        for seed in range(1, seeds + 1):
            relay.simulate_fleet(seed, relay_selector=relay.select_best_relay,
                                 max_steps=steps)
    finally:
        relay._coverage_loss_for_candidate = original
        relay.RESERVE_OPPORTUNITY_ENABLED = prev_flag

    print(f"\nMarginal coverage loss L_i, {seeds} seeds x {steps} steps, "
          f"reserve term OFF")
    print(f"{'group':<10}{'n':>7}{'zero%':>8}{'p50+':>9}{'p75+':>9}"
          f"{'p95+':>9}{'p99+':>9}{'max':>9}")
    print("-" * 70)
    out = {}
    for key, vals in rec.items():
        a = np.array(vals, dtype=float)
        if a.size == 0:
            continue
        nz = a[a > 0]
        q = (lambda p: float(np.percentile(nz, p)) if nz.size else 0.0)
        print(f"{key:<10}{a.size:>7}{np.mean(a == 0) * 100:>7.1f}%"
              f"{q(50):>9.4f}{q(75):>9.4f}{q(95):>9.4f}{q(99):>9.4f}{a.max():>9.4f}")
        out[key] = {"n": int(a.size), "zero_frac": float(np.mean(a == 0)),
                    "p50": q(50), "p95": q(95), "p99": q(99), "max": float(a.max())}

    if "patrol" in out:
        p99 = out["patrol"]["p99"]
        print(f"\n  -> set COVERAGE_LOSS_UNIVERSE to span 0 .. {p99:.2f} "
              f"(empirical p99 of non-zero patrol losses)")
        print(f"  -> a reasonable lambda is around the median non-zero loss, "
              f"{out['patrol']['p50']:.2f}")
        print("     (lambda is the cost of the LAST reserve; earlier ones cost")
        print("      lambda * scarcity**gamma, which is far less)")
    return out


# ══════════════════════════════════════════════════════════════════════════
# Sweep
# ══════════════════════════════════════════════════════════════════════════
METRICS = ("coverage_pct", "avg_zone_staleness_steps", "avg_relay_delay_steps",
           "avg_relay_energy_pct", "mission_completed")


def _as_float(v):
    if isinstance(v, bool):
        return float(v)
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def pool_occupancy(seeds=6, steps=500, every=25):
    """Does the idle-reserve pool ever actually run dry?

    The opportunity-cost term is only meaningful if it does. Run this before
    trusting any lambda > 0 -- on the default fleet the pool sits full most of
    the mission, which makes the whole term a solution without a problem.
    """
    prev = relay.RESERVE_OPPORTUNITY_ENABLED
    relay.RESERVE_OPPORTUNITY_ENABLED = False
    samples = []
    try:
        for seed in range(1, seeds + 1):
            st = relay.init_fleet(seed)
            for t in range(steps):
                relay.step_fleet(st)
                if t % every == 0:
                    samples.append(relay._reserve_pool_state(st["drones"]))
    finally:
        relay.RESERVE_OPPORTUNITY_ENABLED = prev

    a = np.array(samples, dtype=float)
    cap = int(a[:, 1].max()) if a.size else 0
    print(f"\nIdle-reserve pool occupancy, {seeds} seeds x {steps} steps "
          f"(sampled every {every})")
    print(f"  pool capacity at mission start : {cap}")
    print(f"  available: mean={a[:, 0].mean():.2f}  min={a[:, 0].min():.0f}  "
          f"max={a[:, 0].max():.0f}")
    for k in range(cap + 1):
        print(f"    exactly {k} free : {np.mean(a[:, 0] == k) * 100:5.1f}% of samples")
    empty = float(np.mean(a[:, 0] == 0) * 100)
    print(f"  pool EMPTY {empty:.1f}% of the time")
    if empty < 2.0:
        print("\n  -> The pool essentially never runs dry, so a depletion cost has")
        print("     nothing to price. Leave RESERVE_OPPORTUNITY_ENABLED = False and")
        print("     report this measurement instead: it rules the reserve pool OUT")
        print("     as the mechanism behind any coverage difference.")
    else:
        print("\n  -> The pool does run dry often enough for the term to matter.")
        print("     Sweep lambda and report the sensitivity curve.")
    return {"capacity": cap, "empty_pct": empty, "mean_available": float(a[:, 0].mean())}


def run_sweep(lambdas, seeds, steps, policies=POLICIES):
    rows = []
    for lam in lambdas:
        relay.RESERVE_OPPORTUNITY_LAMBDA = float(lam)
        relay.RESERVE_OPPORTUNITY_ENABLED = (lam > 0)
        per_policy = {p: {m: [] for m in METRICS} for p in policies}
        for seed in range(1, seeds + 1):
            for p in policies:
                m = relay.simulate_fleet(seed, relay_selector=relay.POLICY_SELECTORS[p],
                                         max_steps=steps)
                for k in METRICS:
                    per_policy[p][k].append(_as_float(m.get(k)))
        for p in policies:
            row = {"lambda": lam, "policy": p}
            for k in METRICS:
                row[k] = float(np.nanmean(per_policy[p][k]))
            rows.append(row)
        print(f"  lambda={lam:<5} done  " + "  ".join(
            f"{p}:cov={np.nanmean(per_policy[p]['coverage_pct']):.2f}" for p in policies))
    return rows


def main():
    if "--pool" in sys.argv:
        pool_occupancy(seeds=_arg("--seeds", 6), steps=_arg("--steps", 500))
        return

    if "--calibrate" in sys.argv:
        calibrate(seeds=_arg("--seeds", 5), steps=_arg("--steps", 500))
        return

    seeds = _arg("--seeds", 20)
    steps = _arg("--steps", 500)
    # positional args are the lambda grid; skip anything that is a flag or the
    # value belonging to a flag (--seeds 20 must not become lambda=20)
    argv = sys.argv[1:]
    skip = set()
    for i, a in enumerate(argv):
        if a.startswith("--"):
            skip.add(i)
            if a in ("--seeds", "--steps") and i + 1 < len(argv):
                skip.add(i + 1)
    grid = [float(a) for i, a in enumerate(argv)
            if i not in skip and a.replace(".", "", 1).replace("-", "", 1).isdigit()]
    lambdas = grid or list(DEFAULT_LAMBDAS)

    print(f"Reserve opportunity-cost sweep: lambda in {lambdas}")
    print(f"  seeds=1..{seeds}, max_steps={steps}, "
          f"gamma={relay.RESERVE_OPPORTUNITY_GAMMA}, policies={POLICIES}")
    print("  lambda=0 is the pre-patch model (term disabled).\n")

    rows = run_sweep(lambdas, seeds, steps)

    with open("reserve_lambda_sweep.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["lambda", "policy"] + list(METRICS))
        w.writeheader()
        w.writerows(rows)

    print(f"\n{'lambda':>7} {'policy':>7} " + " ".join(f"{m[:14]:>15}" for m in METRICS))
    print("-" * 95)
    for r in rows:
        print(f"{r['lambda']:>7.2f} {r['policy']:>7} "
              + " ".join(f"{r[m]:>15.3f}" for m in METRICS))

    print("\nRead the C-vs-D coverage column down the sweep. If the gap closes")
    print("monotonically and stays closed, the reserve term explains it. If it")
    print("only closes at one lambda, say so -- that is a weaker claim, and")
    print("reporting it weak is better than having a reviewer find it.")
    print("\nWritten: reserve_lambda_sweep.csv")


if __name__ == "__main__":
    main()
