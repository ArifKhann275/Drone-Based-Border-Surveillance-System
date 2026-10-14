# -*- coding: utf-8 -*-
"""
run_sweeps.py  --  re-run the terrain x wind generalization sweep and the
fleet-size sensitivity sweep with the CORRECTED FIS (D), the CORRECTED
surrogate (D2), TOPSIS (T) and VIKOR (V) -- everything in
a2e_relay_fixed_final_51.py's own POLICY_SELECTORS table.

Both sweeps already exist in a2e_relay_fixed_final_51.py
(run_terrain_wind_generalization_analysis, run_fleet_size_sensitivity_analysis)
and already accept a `policies=` argument. This script is the thing that:

  1. makes sure real SRTM terrain is actually being used (not the synthetic
     fallback -- every number from a synthetic-terrain run is NOT comparable
     to the paper's Tables IX/X, and the fallback is easy to miss in a long
     log; this script checks and refuses to silently proceed on it)
  2. runs BOTH sweeps with policies=("C","D","D2","T","V")
  3. adds a Benjamini-Hochberg correction ACROSS THE WHOLE COMBINED FAMILY of
     both sweeps at once -- exactly what your paper's Section VII/Table
     methodology needs, and what neither built-in sweep function does on its
     own (each reports only its own per-cell uncorrected p-values)
  4. writes three files: terrain_wind_sweep.csv, fleet_size_sweep.csv,
     sweep_significance_bh.csv, plus the fleet-size scaling figure
  5. prints a compact summary table you can paste straight into the paper

USAGE
-----
    python run_sweeps.py                        # full sweep, 50 seeds
    python run_sweeps.py --seeds 10              # quick check, 10 seeds
    python run_sweeps.py --steps 300 --seeds 10  # faster still
    python run_sweeps.py --skip-terrain          # fleet-size sweep only
    python run_sweeps.py --skip-fleet            # terrain x wind sweep only
    python run_sweeps.py --allow-synthetic       # proceed even without SRTM
                                                 # (NOT valid for the paper --
                                                 #  only for a code smoke test)

Runtime: with D vectorized (~2 ms/decision) and D2 (~1.3 ms/decision), the
full sweep at 50 seeds is dominated by simulation time, not selector time --
budget the same wall-clock time your previous D-only sweeps needed.
"""

import argparse
import csv

import numpy as np

import a20
import a2e_relay_fixed_final_51 as relay
import policy_d_true_fis as fis

POLICIES = ("C", "D", "D2", "T", "V", "I")
# "I" = RelayScore with coverage_loss scored on an absolute scale instead of
# pool-relative min-max (a2e's own Policy I, COVERAGE_LOSS_SATURATION_CEILING
# just corrected from 0.06 to 0.6 -- see a2e_relay_fixed_final_51.py). Include
# it because a 20-seed smoke test shows it may close C's remaining
# coverage/staleness gap against D without any weight tuning; confirm at
# full n=50 before reporting.

# The paper's Table III mission length is 500 steps (700 only for the
# separate RelayScore-BA extended-budget sub-study of Section V-G).
# a20.MAX_STEPS is a generic module ceiling (2000) that no experiment in the
# paper actually runs at -- it existed only as a safety cap, not a default to
# inherit. An earlier version of this script defaulted to a20.MAX_STEPS,
# which silently ran every sweep at 4x the paper's mission length. Anchor
# the default here instead, and require an explicit --steps to override it.
PAPER_MAX_STEPS = 500

FLEET_SIZES = ((1, 1), (2, 2), (4, 4), (8, 8))     # -> Nf = 2, 4, 8, 16
WIND_MODES = ("static", "dynamic", "stochastic")
TERRAIN_METRICS = ("coverage_pct", "avg_relay_delay_steps", "avg_relay_energy_pct")
FLEET_METRICS = ("coverage_pct", "avg_relay_delay_steps", "avg_relay_energy_pct")


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=50)
    p.add_argument("--steps", type=int, default=PAPER_MAX_STEPS,
                   help=f"default: {PAPER_MAX_STEPS} (the paper's Table III "
                       "mission length -- NOT a20.MAX_STEPS, which is a "
                       "generic module ceiling of 2000 unrelated to any "
                       "experiment in the paper, and would silently run "
                       "missions 4x longer than every prior result)")
    p.add_argument("--skip-terrain", action="store_true")
    p.add_argument("--skip-fleet", action="store_true")
    p.add_argument("--allow-synthetic", action="store_true",
                   help="proceed even if real SRTM terrain is unavailable "
                       "(code smoke test only -- invalid for the paper)")
    p.add_argument("--terrain-out", default="terrain_wind_sweep.csv")
    p.add_argument("--fleet-out", default="fleet_size_sweep.csv")
    p.add_argument("--bh-out", default="sweep_significance_bh.csv")
    p.add_argument("--plot-out", default="fleet_size_scaling.png")
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════
# Pre-flight: make sure the corrected FIS is actually active, and that real
# SRTM terrain is available before spending an hour on a sweep that would be
# thrown away.
# ══════════════════════════════════════════════════════════════════════════
def preflight(allow_synthetic):
    print("=" * 74)
    print("PRE-FLIGHT CHECKS")
    print("=" * 74)

    ok = True

    # 1. rule base / backend sanity -- catches a stale checkout or a LUT
    #    built from the pre-fix rule base before an hour of compute is spent.
    try:
        fis.validate_membership_orientation()
        print("  [ok] membership-function orientation")
    except AssertionError as exc:
        print(f"  [FAIL] membership orientation: {exc}")
        ok = False

    v = fis.validate_backends(n=200)
    if v["max_abs_err"] < 1e-6:
        print(f"  [ok] backend agreement: max|diff|={v['max_abs_err']:.2e}")
    else:
        print(f"  [FAIL] backend disagreement: max|diff|={v['max_abs_err']:.4g}")
        ok = False

    for p in POLICIES:
        if p not in relay.POLICY_SELECTORS:
            print(f"  [FAIL] policy {p!r} missing from POLICY_SELECTORS")
            ok = False
    if ok:
        print(f"  [ok] policies present: {POLICIES}")

    # 2. LUT freshness -- surrogate_fis raises StaleLutError on mismatch, but
    #    surface that clearly here rather than mid-sweep after seed 1 of 50.
    try:
        import surrogate_fis as S
        S._load_interpolator("expert")
        S._load_interpolator("indep")
        print("  [ok] D2 LUTs match the current rule-base fingerprint")
    except Exception as exc:                                    # noqa: BLE001
        print(f"  [FAIL] D2 LUT problem: {exc}")
        print("         Fix:  rm -rf luts && python build_surrogate_fis.py")
        ok = False

    # 3. real SRTM vs synthetic fallback. init_fleet()/set_terrain_scenario()
    #    print a loud warning on fallback; we additionally check the grid
    #    directly so this is enforced, not just logged.
    real_srtm = _check_real_terrain()
    if real_srtm:
        print("  [ok] real SRTM terrain is active")
    else:
        print("  [WARN] SYNTHETIC terrain fallback is active -- every number")
        print("         this sweep produces will NOT be comparable to a real-")
        print("         terrain run, and the terrain x wind sweep specifically")
        print("         tests climb-cost effects that synthetic terrain does")
        print("         not reproduce correctly.")
        if not allow_synthetic:
            print("\n  Refusing to run the real sweep on synthetic terrain.")
            print("  Fix: run a20.prime_srtm_cache() with a working internet")
            print("  connection once (caches permanently), or pass")
            print("  --allow-synthetic to proceed anyway for a CODE smoke")
            print("  test only -- do not use that output in the paper.")
            ok = False

    print()
    if not ok:
        raise SystemExit("Pre-flight failed -- see messages above.")
    print("Pre-flight OK.\n")


def _check_real_terrain():
    """Checks a20.TERRAIN_SOURCE, the flag get_elevation_grid() sets on every
    call ("real_srtm" | "real_srtm_export" | "synthetic_fallback" |
    "synthetic_forced"). Only the first two are valid for the paper.

    On a fresh import this is often still None (nothing has built a grid
    yet), in which case we force one build via a20.set_terrain_scenario() on
    the paper's own default sector and read the flag it leaves behind --
    this is the same call init_fleet()/simulate_fleet() make internally, so
    it does not change what the sweep itself will see.
    """
    src = getattr(a20, "TERRAIN_SOURCE", None)
    if src is None:
        try:
            a20.set_terrain_scenario("benapole_delta", use_real_srtm=True)
            src = getattr(a20, "TERRAIN_SOURCE", None)
        except Exception:                                        # noqa: BLE001
            src = None

    if src in ("real_srtm", "real_srtm_export"):
        print(f"      TERRAIN_SOURCE = {src!r}")
        return True

    print(f"      TERRAIN_SOURCE = {src!r}")
    if src == "synthetic_fallback":
        print("      (network SRTM download failed; trying prime_srtm_cache()...)")
        ok = False
        try:
            ok = bool(a20.prime_srtm_cache())
        except Exception as exc:                                 # noqa: BLE001
            print(f"      prime_srtm_cache() raised: {exc}")
        if ok:
            a20.set_terrain_scenario("benapole_delta", use_real_srtm=True)
            src = getattr(a20, "TERRAIN_SOURCE", None)
            print(f"      TERRAIN_SOURCE after cache prime = {src!r}")
            return src in ("real_srtm", "real_srtm_export")
        print("      prime_srtm_cache() could not reach the network either.")
        print("      If this environment can never reach the internet (e.g. a")
        print("      sandboxed CI runner), build ELEVATION_GRID once on a machine")
        print("      that can, then run a20.export_elevation_grid() there and copy")
        print("      the resulting elevation_grid_export.json next to a20.py here --")
        print("      get_elevation_grid() loads that with zero network calls.")
    return False


# ══════════════════════════════════════════════════════════════════════════
# BH correction across BOTH sweeps combined
# ══════════════════════════════════════════════════════════════════════════
def _bh_adjust(pvals):
    """Benjamini-Hochberg step-up. Returns q-values, same order as input.
    None/NaN entries pass through unadjusted (they carry no test)."""
    idx = [i for i, p in enumerate(pvals) if p is not None and not np.isnan(p)]
    m = len(idx)
    q = [None] * len(pvals)
    if m == 0:
        return q
    order = sorted(idx, key=lambda i: pvals[i])
    prev = 1.0
    for rank, i in enumerate(reversed(order), start=1):
        real_rank = m - rank + 1
        val = pvals[i] * m / real_rank
        prev = min(prev, val)
        q[i] = prev
    return q


def collect_bh_family(terrain_summary, fleet_summary, policies=POLICIES):
    """Pulls every {metric}_test_{P}_vs_D p-value out of both sweep summaries
    and BH-corrects them together, exactly like the paper's single 180-test
    family in Section VII -- just recomputed for the corrected FIS/D2."""
    non_d = [p for p in policies if p != "D"]
    rows = []

    for entry in terrain_summary:
        for metric in TERRAIN_METRICS:
            for p in non_d:
                key = f"{metric}_test_{p}_vs_D"
                if key not in entry:
                    continue
                test = entry[key] or {}
                pv = test.get("wilcoxon_p") if isinstance(test, dict) else test
                rows.append({
                    "sweep": "terrain_wind",
                    "condition": f"{entry['terrain']}/{entry['wind_mode']}",
                    "metric": metric, "policy": p, "p": pv,
                })

    # The fleet-size sweep names its keys differently from the terrain sweep
    # (coverage_test_{p}_vs_D / delay_test_{p}_vs_D / energy_test_{p}_vs_D,
    # per run_fleet_size_sensitivity_analysis's own scheme) -- map explicitly
    # rather than guessing a shared pattern.
    fleet_key_map = {"coverage_pct": "coverage", "avg_relay_delay_steps": "delay",
                     "avg_relay_energy_pct": "energy"}
    for entry in fleet_summary:
        for metric, short in fleet_key_map.items():
            for p in non_d:
                key = f"{short}_test_{p}_vs_D"
                if key not in entry:
                    continue
                test = entry[key] or {}
                pv = test.get("wilcoxon_p") if isinstance(test, dict) else test
                rows.append({
                    "sweep": "fleet_size", "condition": f"Nf={entry['fleet_size']}",
                    "metric": metric, "policy": p, "p": pv,
                })

    pvals = [r["p"] for r in rows]
    q = _bh_adjust(pvals)
    for r, qq in zip(rows, q):
        r["q_bh"] = qq
        r["survives_bh"] = bool(qq is not None and qq < 0.05)
    return rows


def write_bh_csv(rows, path):
    if not rows:
        return
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Written: {path}  ({len(rows)} tests)")


def print_bh_summary(rows):
    survivors = [r for r in rows if r["survives_bh"]]
    print(f"\n{len(rows)} tests total, {len(survivors)} survive BH correction "
          f"(q < 0.05):\n")
    if not survivors:
        print("  (none)")
        return
    print(f"{'sweep':<14}{'condition':<28}{'metric':<26}{'policy':>7}{'p':>10}{'q':>10}")
    print("-" * 96)
    for r in sorted(survivors, key=lambda r: r["q_bh"]):
        print(f"{r['sweep']:<14}{r['condition']:<28}{r['metric']:<26}{r['policy']:>7}"
              f"{r['p']:>10.3g}{r['q_bh']:>10.3g}")


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════
def main():
    args = _parse_args()
    seeds = list(range(1, args.seeds + 1))
    max_steps = args.steps

    if max_steps != PAPER_MAX_STEPS:
        print(f"NOTE: running at max_steps={max_steps}, NOT the paper's "
              f"Table III value of {PAPER_MAX_STEPS}. Every metric from this "
              f"run is fine for its own internal comparisons but is NOT "
              f"directly comparable to any {PAPER_MAX_STEPS}-step result "
              f"(more relay decisions accumulate, coverage/staleness partly "
              f"re-saturate, per-mission selector-time totals scale up).\n")

    preflight(args.allow_synthetic)

    terrain_summary, fleet_summary = [], []

    if not args.skip_terrain:
        print("=" * 74)
        print(f"TERRAIN x WIND SWEEP -- policies={POLICIES}, "
              f"seeds=1..{args.seeds}, max_steps={max_steps}")
        print("=" * 74)
        _, terrain_summary = relay.run_terrain_wind_generalization_analysis(
            seeds=seeds, wind_modes=WIND_MODES, max_steps=max_steps,
            out_csv=args.terrain_out, policies=POLICIES,
        )
        a20.set_terrain_scenario("benapole_delta")   # restore baseline terrain
        print(f"\nWritten: {args.terrain_out}")

    if not args.skip_fleet:
        print("\n" + "=" * 74)
        print(f"FLEET-SIZE SWEEP -- policies={POLICIES}, "
              f"seeds=1..{args.seeds}, max_steps={max_steps}")
        print("=" * 74)
        _, fleet_summary = relay.run_fleet_size_sensitivity_analysis(
            seeds=seeds, fleet_sizes=FLEET_SIZES, max_steps=max_steps,
            out_csv=args.fleet_out, policies=POLICIES,
        )
        print(f"\nWritten: {args.fleet_out}")
        try:
            relay.plot_fleet_size_scaling(fleet_summary, save_path=args.plot_out)
        except Exception as exc:                                 # noqa: BLE001
            print(f"(plot skipped: {exc})")

    if terrain_summary or fleet_summary:
        print("\n" + "=" * 74)
        print("COMBINED BENJAMINI-HOCHBERG CORRECTION (both sweeps, one family)")
        print("=" * 74)
        rows = collect_bh_family(terrain_summary, fleet_summary)
        write_bh_csv(rows, args.bh_out)
        print_bh_summary(rows)

    print("\nDone. Compare this run's D/D2 numbers against your earlier")
    print("(pre-fix) sweep -- D's delay/energy should have improved sharply")
    print("(the FIS was picking far, slow, saturated-attribute candidates")
    print("before both fixes), and D2 should track D closely everywhere,")
    print("the same pattern as the single-condition run you already checked.")


if __name__ == "__main__":
    main()
