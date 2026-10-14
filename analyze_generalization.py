# -*- coding: utf-8 -*-
"""
analyze_generalization.py  (confirmatory analysis, seeds 801-900, v3)

Implements section 5 of PREREGISTRATION_confirmatory_seeds801-900.txt.
  P1/P2  delay and energy: paired Wilcoxon, LARS minus comparator; BH over the 14 P-tests
  S1     coverage equivalence: paired TOST, margin +/-2.0; BH over the 7 S1-tests
  S2     mission non-inferiority: Tango (1998) score CI for the paired risk difference
         (LARS minus comparator). Pass if the lower limit > -0.05. No adjustment.
  EXPL   B, BR, T_tuned, V_tuned, LARS_tuned: descriptive only; BH within this family.

The script refuses data that do not match the plan, and refuses to overwrite its output,
so the analysis can be run once.

Usage:
  python analyze_generalization.py --tango-selftest              (checks the CI code only)
  python analyze_generalization.py --csv confirm_801_900.csv --out confirmatory_801_900_results.csv
"""

import argparse
import csv
import hashlib
import os
import sys

import numpy as np
from scipy.optimize import brentq, minimize_scalar
from scipy.stats import norm

from compare_focal_vs_all import paired_ci, wilcoxon_p, tost, bh_adjust, rank_biserial

# ---- constants fixed by the plan (keep identical to run_generalization_sweep.py)
FOCAL = "LARS"
P_COMPS = ["C", "D", "D2", "T", "V", "AUC", "J"]
EXPL_COMPS = ["B", "BR", "T_tuned", "V_tuned", "LARS_tuned"]
COV, DLY, ENG, MISS = "coverage_pct", "avg_relay_delay_steps", "avg_relay_energy_pct", "mission_completed"
CONF_SEEDS = list(range(801, 901))
TERRAIN, WIND, NP, NR = "benapole_delta", "static", 4, 4
PLAN_CELL_SIZE = 499.6
PLAN_FLAGS = {
    "USE_NONLINEAR_BATTERY": True,
    "USE_SOH_AGING": True,
    "PATROL_MODE": "sse_gap",
    "SSE_TERRAIN_AWARE": False,
    "HY_SSE_W": 1.0,
    "TERRAIN_COST_ENABLED": True,
}
MARGIN_COV = 2.0
NI_MARGIN = -0.05
ALPHA = 0.05


def to_float(x):
    try:
        v = float(x)
        return v if np.isfinite(v) else np.nan
    except (TypeError, ValueError):
        return np.nan


# ============================================================ Tango score CI
def _restricted_t(delta, n10, n01, m):
    """Restricted MLE under p10 - p01 = delta. Free parameter t = p01, so
    p10 = t + delta, p01 = t, and p11 + p00 = 1 - 2t - delta, shared by the m concordant cells."""
    lo = max(0.0, -delta)
    hi = (1.0 - delta) / 2.0
    if m == 0:
        return hi                     # the discordant cells must carry all the mass
    if hi - lo < 1e-14:
        return lo

    def negll(t):
        p10, p01, rest = t + delta, t, 1.0 - 2.0 * t - delta
        if (n10 and p10 <= 0) or (n01 and p01 <= 0) or rest <= 0:
            return np.inf
        v = -m * np.log(rest)
        if n10:
            v -= n10 * np.log(p10)
        if n01:
            v -= n01 * np.log(p01)
        return v

    return minimize_scalar(negll, bounds=(lo, hi), method="bounded", options={"xatol": 1e-12}).x


def _score_var(delta, n10, n01, m, n):
    """Score variance of the paired difference under the restricted MLE:
    V(delta) = [p10 + p01 - (p10 - p01)^2] / n, with p10 - p01 = delta."""
    t = _restricted_t(delta, n10, n01, m)
    p10, p01 = t + delta, t
    return max((p10 + p01 - delta * delta) / n, 0.0)


def tango_ci(x, y, alpha=ALPHA):
    """Tango (1998) score interval for the paired risk difference p_x - p_y.
    x, y: 0/1 arrays, seed-matched (x = LARS, y = comparator).
    Returns (estimate, lower, upper, note). Lower/upper come from the two-sided
    (1 - alpha) interval; its lower limit is the one-sided 97.5% bound used for non-inferiority.
    With no discordant pairs the score interval is undefined; note is set and no bounds are returned."""
    x = np.asarray(x, dtype=int)
    y = np.asarray(y, dtype=int)
    n = len(x)
    n10 = int(np.sum((x == 1) & (y == 0)))
    n01 = int(np.sum((x == 0) & (y == 1)))
    m = int(np.sum(x == y))
    theta = (n10 - n01) / n
    if n10 + n01 == 0:
        return theta, None, None, "no discordant pairs"
    z = norm.ppf(1 - alpha / 2)

    def V(d):
        return _score_var(d, n10, n01, m, n)

    def f_lo(d):
        return (theta - d) - z * np.sqrt(V(d))

    def f_hi(d):
        return (d - theta) - z * np.sqrt(V(d))

    eps = 1e-9
    a, b = -1.0 + eps, 1.0 - eps
    lower = brentq(f_lo, a, theta) if f_lo(a) > 0 else -1.0
    upper = brentq(f_hi, theta, b) if f_hi(b) > 0 else 1.0
    return theta, lower, upper, ""


def _make_table(n10, n01, n11, n00):
    x = [1] * n10 + [0] * n01 + [1] * n11 + [0] * n00
    y = [0] * n10 + [1] * n01 + [1] * n11 + [0] * n00
    return np.array(x), np.array(y)


def tango_selftest():
    """Internal consistency checks of the CI code. These do NOT replace a comparison with a
    published reference (see the note in the paper plan); they check the algebra and symmetry."""
    z = norm.ppf(0.975)
    tables = [(3, 1, 10, 6), (0, 0, 20, 5), (5, 5, 3, 7), (1, 0, 40, 9), (12, 2, 30, 10)]
    for (n10, n01, n11, n00) in tables:
        x, y = _make_table(n10, n01, n11, n00)
        n = len(x)
        theta, lo, hi, note = tango_ci(x, y)
        if note:
            assert n10 + n01 == 0
            continue
        assert -1 <= lo <= theta <= hi <= 1, (n10, n01, lo, theta, hi)
        # the endpoints must sit on the score boundary (skip if the bound is at +/-1)
        for d, label in ((lo, "lower"), (hi, "upper")):
            if abs(d) < 1 - 1e-6:
                m = n11 + n00
                s = (theta - d) ** 2 / max(_score_var(d, n10, n01, m, n), 1e-300)
                assert abs(s - z * z) < 1e-3 * z * z, (n10, n01, label, s, z * z)
        # swapping x and y must mirror the interval
        t2, lo2, hi2, _ = tango_ci(y, x)
        assert abs(t2 + theta) < 1e-12
        assert abs(lo2 + hi) < 1e-6 and abs(hi2 + lo) < 1e-6, (n10, n01, (lo, hi), (lo2, hi2))
    # balanced table: estimate 0, interval symmetric
    x, y = _make_table(4, 4, 20, 10)
    theta, lo, hi, _ = tango_ci(x, y)
    assert abs(theta) < 1e-12 and abs(lo + hi) < 1e-6
    print("Tango self-test passed (algebra, boundary, symmetry). Compare with a reference "
          "implementation before the paper is submitted.")


# ============================================================ data checks
def check_inputs(rows, header):
    errs = []
    need = [f"{p}_{m}" for p in [FOCAL] + P_COMPS + EXPL_COMPS for m in (COV, DLY, ENG, MISS)]
    missing = [c for c in need if c not in header]
    if missing:
        errs.append(f"missing columns: {missing[:6]}{' ...' if len(missing) > 6 else ''}")
    if not rows:
        return ["no rows"]

    def vals(col):
        return {r.get(col, "") for r in rows}

    if vals("mode") != {"confirmatory"}:
        errs.append(f"mode values {vals('mode')}; expected only 'confirmatory'")
    if vals("terrain") != {TERRAIN}:
        errs.append(f"terrain values {vals('terrain')}; expected {TERRAIN}")
    if vals("wind") != {WIND}:
        errs.append(f"wind values {vals('wind')}; expected {WIND}")
    if {int(r["num_drones"]) for r in rows} != {NP} or {int(r["num_reserves"]) for r in rows} != {NR}:
        errs.append(f"fleet is not {NP}x{NR}")
    seeds = sorted(int(r["seed"]) for r in rows)
    if seeds != CONF_SEEDS:
        errs.append(f"seeds: {len(seeds)} rows; need exactly 801-900 once each")
    if vals("git_dirty") != {"0"}:
        errs.append(f"git_dirty values {vals('git_dirty')}; every row must be 0")
    if vals("terrain_source") != {"real_srtm"}:
        errs.append(f"terrain_source values {vals('terrain_source')}; need real_srtm")
    if any(abs(float(r["cell_size_m"]) - PLAN_CELL_SIZE) > 0.05 for r in rows):
        errs.append(f"cell_size_m differs from {PLAN_CELL_SIZE}")
    for k, v in PLAN_FLAGS.items():
        if vals(f"flag_{k}") != {str(v)}:
            errs.append(f"flag_{k} values {vals(f'flag_{k}')}; plan says {v!r}")
    codes = vals("code_last_commit")
    if len(codes) != 1 or "NO_GIT" in codes:
        errs.append(f"code_last_commit not a single valid commit: {codes}")
    for c in need:
        bad = [r["seed"] for r in rows if to_float(r.get(c, "")) != to_float(r.get(c, ""))]
        if bad:
            errs.append(f"{c}: {len(bad)} missing/non-numeric values")
    for c in [f"{p}_{MISS}" for p in [FOCAL] + P_COMPS + EXPL_COMPS]:
        if any(r[c] not in ("0", "1") for r in rows):
            errs.append(f"{c}: values other than 0/1")
    return errs


def arr(rows, pol, metric):
    return np.array([to_float(r[f"{pol}_{metric}"]) for r in rows], dtype=float)


# ============================================================ tests
def _wp(d):
    return 1.0 if np.all(d == 0) else wilcoxon_p(d)


def run_P(rows):
    out = []
    for metric, hyp in ((DLY, "P1"), (ENG, "P2")):
        for c in P_COMPS:
            d = arr(rows, FOCAL, metric) - arr(rows, c, metric)
            mean, lo, hi = paired_ci(d)
            out.append(dict(section="P", hyp=hyp, comparator=c, metric=metric, n=len(d),
                            mean=mean, lo=lo, hi=hi, p=_wp(d), r=rank_biserial(d)))
    q = bh_adjust([o["p"] for o in out])
    for o, qq in zip(out, q):
        o["q"] = float(qq)
        o["pass"] = bool(o["mean"] < 0 and o["q"] < ALPHA)
    return out


def run_S1(rows):
    out = []
    for c in P_COMPS:
        d = arr(rows, FOCAL, COV) - arr(rows, c, COV)
        p, ci90 = tost(d, MARGIN_COV)
        out.append(dict(section="S1", hyp="S1", comparator=c, metric=COV, n=len(d),
                        mean=float(d.mean()), lo=ci90[0], hi=ci90[1], p=p, r=None))
    q = bh_adjust([o["p"] for o in out])
    for o, qq in zip(out, q):
        o["q"] = float(qq)
        o["pass"] = bool(o["q"] < ALPHA)
    return out


def run_S2(rows, comps, section):
    out = []
    x = arr(rows, FOCAL, MISS).astype(int)
    for c in comps:
        y = arr(rows, c, MISS).astype(int)
        theta, lo, hi, note = tango_ci(x, y)
        passed = True if note else bool(lo > NI_MARGIN)
        out.append(dict(section=section, hyp="S2" if section == "S2" else "EXPL", comparator=c,
                        metric=MISS, n=len(x), mean=theta, lo=lo, hi=hi, p=None, q=None,
                        r=None, pass_=passed, note=note))
    return out


def run_expl(rows):
    out = []
    for c in EXPL_COMPS:
        for metric in (COV, DLY, ENG):
            d = arr(rows, FOCAL, metric) - arr(rows, c, metric)
            mean, lo, hi = paired_ci(d)
            out.append(dict(section="EXPL", hyp="EXPL", comparator=c, metric=metric, n=len(d),
                            mean=mean, lo=lo, hi=hi, p=_wp(d), r=rank_biserial(d)))
    q = bh_adjust([o["p"] for o in out])
    for o, qq in zip(out, q):
        o["q"] = float(qq)
    return out


# ============================================================ output
def fmt(o):
    lo = "NA" if o["lo"] is None else f"{o['lo']:+.3f}"
    hi = "NA" if o["hi"] is None else f"{o['hi']:+.3f}"
    return f"{o['mean']:+.3f} [{lo}, {hi}]"


def print_block(title, items):
    print("\n" + title)
    print(f"{'hyp':<5}{'vs':<10}{'metric':<24}{'diff [CI]':<34}{'p':>10}{'q':>9}{'pass':>7}")
    for o in items:
        p = "" if o["p"] is None else f"{o['p']:.4g}"
        q = "" if o.get("q") is None else f"{o['q']:.4g}"
        passed = o.get("pass_", o.get("pass", ""))
        print(f"{o['hyp']:<5}{o['comparator']:<10}{o['metric']:<24}{fmt(o):<34}{p:>10}{q:>9}{str(passed):>7}")
        if o.get("note"):
            print(f"      note: {o['note']}")


def verdict(label, items, key):
    fails = [o["comparator"] for o in items if not o[key]]
    status = "CONFIRMED" if not fails else "NOT CONFIRMED"
    extra = "" if not fails else f"  failing: {', '.join(fails)}"
    print(f"{label}: {status}{extra}")
    return not fails


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv")
    ap.add_argument("--out", default="confirmatory_801_900_results.csv")
    ap.add_argument("--tango-selftest", action="store_true")
    args = ap.parse_args()

    if args.tango_selftest:
        tango_selftest()
        return
    if not args.csv:
        raise SystemExit("give --csv (or --tango-selftest)")
    if os.path.exists(args.out):
        raise SystemExit(f"{args.out} exists. The confirmatory analysis is run once; "
                         f"do not overwrite it. Move it aside only with a written reason.")

    tango_selftest()

    with open(args.csv, newline="") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        rows = list(reader)
    errs = check_inputs(rows, header)
    if errs:
        print("STOP: the data do not match the plan:")
        for e in errs:
            print("  -", e)
        raise SystemExit(1)

    with open(args.csv, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    code = rows[0]["code_last_commit"]
    print(f"input: {args.csv}  sha256={digest}")
    print(f"code commit: {code}   run commit: {rows[0]['git_commit']}")
    print(f"rows: {len(rows)} (seeds 801-900)")

    P = run_P(rows)
    S1 = run_S1(rows)
    S2 = run_S2(rows, P_COMPS, "S2")
    E = run_expl(rows)
    E_mis = run_S2(rows, EXPL_COMPS, "EXPL")

    print_block("P-TESTS (LARS minus comparator; BH over 14)", P)
    print_block("S1 (coverage TOST, margin +/-2.0; 90% CI; BH over 7)", S1)
    print_block("S2 (mission Tango 95% CI; pass if lower > -0.05; no adjustment)", S2)
    print_block("EXPLORATORY (descriptive, not used in any decision)", E + E_mis)

    print("\nVERDICTS (section 6 of the plan)")
    verdict("P1 delay", [o for o in P if o["metric"] == DLY], "pass")
    verdict("P2 energy", [o for o in P if o["metric"] == ENG], "pass")
    verdict("S1 coverage equivalence", S1, "pass")
    verdict("S2 mission non-inferiority", S2, "pass_")

    fields = ["section", "hyp", "comparator", "metric", "n", "mean", "lo", "hi", "p", "q", "r", "pass", "note"]
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for o in P + S1 + S2 + E + E_mis:
            o = dict(o)
            o["pass"] = o.pop("pass_", o.get("pass", ""))
            w.writerow(o)
        f.write(f"\n# input sha256={digest}\n")
    print(f"\nWritten: {args.out}. Report every line in the paper, whatever the outcome.")


if __name__ == "__main__":
    main()