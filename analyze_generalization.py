# -*- coding: utf-8 -*-
"""
analyze_generalization.py

Evaluates the hypotheses G1-G3 of PREREGISTRATION_generalization_seeds601-700.txt on the
CSV files written by run_generalization_sweep.py, and prints every cell in full.

    G1  coverage of LARS equivalent (TOST +/-2.0) to C and to D in every terrain x wind cell
        (24 TOST tests, BH over the 24)
    G2  delay and energy of LARS significantly lower than C and D in every cell
        (48 tests, BH over the 48)
    G3  the same two statements at every fleet size
        (10 TOST tests and 20 difference tests, BH within each)
    Exploratory: LARS vs B and T_tuned (all three metrics, BH over that family, descriptive).

Usage:
    python analyze_generalization.py --csv gen_benapole.csv --csv gen_sylhet.csv \
        --csv gen_coxs.csv --csv gen_chittagong.csv --csv gen_fleet.csv
Writes generalization_results.csv (one row per cell, comparator and metric).

Needs compare_focal_vs_all.py in the same folder. Report every line, whatever the outcome.
"""

import argparse
import csv
from collections import defaultdict

import numpy as np

from compare_focal_vs_all import paired_ci, wilcoxon_p, tost, bh_adjust, rank_biserial

FOCAL = "LARS"
PRIMARY = ["C", "D"]
EXPLORATORY = ["B", "T_tuned"]
COV, DLY, ENG = "coverage_pct", "avg_relay_delay_steps", "avg_relay_energy_pct"
MARGIN = 2.0


def to_float(x):
    try:
        v = float(x)
        return v if np.isfinite(v) else np.nan
    except (TypeError, ValueError):
        return np.nan


def load(paths):
    rows = []
    for p in paths:
        with open(p, newline="") as f:
            rows.extend(csv.DictReader(f))
    if not rows:
        raise SystemExit("No rows found.")
    return rows


def group_cells(rows):
    cells = defaultdict(list)
    for r in rows:
        key = (r["part"], r["terrain"], r["wind"], int(r["num_drones"]), int(r["num_reserves"]))
        cells[key].append(r)
    return cells


def vec(rows, pol, metric):
    return np.array([to_float(r.get(f"{pol}_{metric}", "")) for r in rows], float)


def paired_diff(rows, a, b, metric):
    x, y = vec(rows, a, metric), vec(rows, b, metric)
    ok = np.isfinite(x) & np.isfinite(y)
    return (x - y)[ok], x[ok], y[ok]


def label(key):
    part, terrain, wind, nd, nr = key
    if part == "terrain":
        return f"{terrain}/{wind}"
    return f"N={nd + nr} ({nd}x{nr})"


def analyse_part(part, cells, policies):
    keys = sorted(k for k in cells if k[0] == part)
    if not keys:
        return None
    results = []          # dicts
    # ---- primary families
    tost_rows, diff_rows = [], []
    for k in keys:
        rws = cells[k]
        for comp in PRIMARY:
            if comp not in policies:
                continue
            d, a, b = paired_diff(rws, FOCAL, comp, COV)
            if len(d) >= 3:
                p, ci90 = tost(d, MARGIN)
                tost_rows.append(dict(key=k, comp=comp, n=len(d), mean=float(d.mean()), ci90=ci90, p=p))
            for m in (DLY, ENG):
                d, a, b = paired_diff(rws, FOCAL, comp, m)
                if len(d) >= 3:
                    mean, lo, hi = paired_ci(d)
                    diff_rows.append(dict(key=k, comp=comp, metric=m, n=len(d), fa=float(a.mean()),
                                          fb=float(b.mean()), mean=mean, lo=lo, hi=hi, p=wilcoxon_p(d),
                                          r=rank_biserial(d)))
    q_t = bh_adjust([t["p"] for t in tost_rows])
    for t, q in zip(tost_rows, q_t):
        t["q"] = float(q)
        t["pass"] = bool(q < 0.05)
    q_d = bh_adjust([x["p"] for x in diff_rows])
    for x, q in zip(diff_rows, q_d):
        x["q"] = float(q)
        x["pass"] = bool(x["mean"] < 0 and q < 0.05)
        x["worse"] = bool(x["mean"] > 0 and q < 0.05)
    # ---- exploratory family
    expl = []
    for k in keys:
        rws = cells[k]
        for comp in EXPLORATORY:
            if comp not in policies:
                continue
            for m in (COV, DLY, ENG):
                d, a, b = paired_diff(rws, FOCAL, comp, m)
                if len(d) >= 3:
                    mean, lo, hi = paired_ci(d)
                    expl.append(dict(key=k, comp=comp, metric=m, n=len(d), mean=mean, lo=lo, hi=hi,
                                     p=wilcoxon_p(d)))
    q_e = bh_adjust([e["p"] for e in expl])
    for e, q in zip(expl, q_e):
        e["q"] = float(q)
    return keys, tost_rows, diff_rows, expl


def print_part(part, cells, policies, res, out_rows):
    keys, tost_rows, diff_rows, expl = res
    title = "TERRAIN x WIND" if part == "terrain" else "FLEET SIZE"
    gname = "G1/G2" if part == "terrain" else "G3"
    print("=" * 110)
    print(f"{title}  ({len(keys)} cells; hypotheses {gname})")
    print("=" * 110)
    # means per cell
    print(f"\nMeans per cell (coverage / delay / energy), n = number of paired seeds")
    head = f"{'cell':<34}{'n':>4}" + "".join(f"{p:>22}" for p in policies)
    print(head)
    for k in keys:
        rws = cells[k]
        line = f"{label(k):<34}{len(rws):>4}"
        for p in policies:
            c, d, e = (np.nanmean(vec(rws, p, m)) for m in (COV, DLY, ENG))
            line += f"{c:>8.2f}/{d:>5.2f}/{e:>5.2f} "
        print(line)

    print(f"\nPRIMARY: {FOCAL} minus comparator (negative delay/energy favors {FOCAL})")
    print(f"{'cell':<34}{'vs':<4}{'coverage diff [90% CI]':>28}{'q_TOST':>9}{'equiv':>7}"
          f"{'delay diff [95% CI]':>26}{'q':>8}{'energy diff [95% CI]':>26}{'q':>8}")
    for k in keys:
        for comp in PRIMARY:
            t = next((x for x in tost_rows if x["key"] == k and x["comp"] == comp), None)
            dd = next((x for x in diff_rows if x["key"] == k and x["comp"] == comp and x["metric"] == DLY), None)
            ee = next((x for x in diff_rows if x["key"] == k and x["comp"] == comp and x["metric"] == ENG), None)
            if not (t and dd and ee):
                continue
            tc = f"{t['mean']:+.2f} [{t['ci90'][0]:+.2f},{t['ci90'][1]:+.2f}]"
            dc = f"{dd['mean']:+.2f} [{dd['lo']:+.2f},{dd['hi']:+.2f}]"
            ec = f"{ee['mean']:+.2f} [{ee['lo']:+.2f},{ee['hi']:+.2f}]"
            flag_d = "" if dd["pass"] else (" WORSE" if dd["worse"] else " ns")
            flag_e = "" if ee["pass"] else (" WORSE" if ee["worse"] else " ns")
            print(f"{label(k):<34}{comp:<4}{tc:>28}{t['q']:>9.4f}{'yes' if t['pass'] else 'NO':>7}"
                  f"{dc:>26}{dd['q']:>8.4f}{flag_d:<6}{ec:>26}{ee['q']:>8.4f}{flag_e:<6}")
            out_rows.append(dict(part=part, cell=label(k), comparator=comp, metric=COV, n=t["n"],
                                 mean=t["mean"], lo=t["ci90"][0], hi=t["ci90"][1], q=t["q"], passed=t["pass"],
                                 family="primary_tost"))
            for x in (dd, ee):
                out_rows.append(dict(part=part, cell=label(k), comparator=comp, metric=x["metric"], n=x["n"],
                                     mean=x["mean"], lo=x["lo"], hi=x["hi"], q=x["q"], passed=x["pass"],
                                     family="primary_diff"))

    # verdicts
    n_t, n_d = len(tost_rows), len(diff_rows)
    pass_t = [t for t in tost_rows if t["pass"]]
    pass_d = [x for x in diff_rows if x["pass"]]
    g_cov = "G1" if part == "terrain" else "G3 (coverage)"
    g_eff = "G2" if part == "terrain" else "G3 (delay/energy)"
    print(f"\nVERDICT {g_cov}: coverage equivalence to C and D: {len(pass_t)} of {n_t} tests pass "
          f"-> {'CONFIRMED' if len(pass_t) == n_t and n_t else 'NOT CONFIRMED'}")
    for t in tost_rows:
        if not t["pass"]:
            print(f"   FAIL  {label(t['key'])} vs {t['comp']}: diff {t['mean']:+.2f}, "
                  f"90% CI [{t['ci90'][0]:+.2f},{t['ci90'][1]:+.2f}], q_TOST {t['q']:.4f}")
    print(f"VERDICT {g_eff}: delay and energy significantly lower than C and D: "
          f"{len(pass_d)} of {n_d} tests pass -> {'CONFIRMED' if len(pass_d) == n_d and n_d else 'NOT CONFIRMED'}")
    for x in diff_rows:
        if not x["pass"]:
            tag = "COUNTER-EXAMPLE (LARS significantly worse)" if x["worse"] else "not significant"
            print(f"   FAIL  {label(x['key'])} vs {x['comp']} {x['metric']}: diff {x['mean']:+.2f} "
                  f"[{x['lo']:+.2f},{x['hi']:+.2f}], q {x['q']:.4f}  <- {tag}")

    # exploratory
    print(f"\nEXPLORATORY (descriptive, BH over {len(expl)} tests): {FOCAL} minus B / T_tuned")
    print(f"{'cell':<34}{'vs':<9}{'coverage':>26}{'delay':>26}{'energy':>26}")
    for k in keys:
        for comp in EXPLORATORY:
            es = {e["metric"]: e for e in expl if e["key"] == k and e["comp"] == comp}
            if len(es) < 3:
                continue
            def f(e):
                star = "*" if e["q"] < 0.05 else " "
                return f"{e['mean']:+.2f} [{e['lo']:+.2f},{e['hi']:+.2f}]{star}"
            print(f"{label(k):<34}{comp:<9}{f(es[COV]):>26}{f(es[DLY]):>26}{f(es[ENG]):>26}")
            for m in (COV, DLY, ENG):
                e = es[m]
                out_rows.append(dict(part=part, cell=label(k), comparator=comp, metric=m, n=e["n"],
                                     mean=e["mean"], lo=e["lo"], hi=e["hi"], q=e["q"], passed="",
                                     family="exploratory"))
    print("   (* = q < 0.05 within the exploratory family)\n")
    return len(pass_t) == n_t and n_t > 0, len(pass_d) == n_d and n_d > 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", action="append", required=True)
    ap.add_argument("--out", default="generalization_results.csv")
    args = ap.parse_args()
    rows = load(args.csv)
    cells = group_cells(rows)
    sample = rows[0]
    policies = [c[: -len("_coverage_pct")] for c in sample if c.endswith("_coverage_pct")]
    if FOCAL not in policies:
        raise SystemExit(f"{FOCAL} not in the data; policies present: {policies}")
    print(f"Policies in the data: {policies}\nCells: {len(cells)}; rows: {len(rows)}\n")

    out_rows = []
    verdicts = {}
    for part in ("terrain", "fleet"):
        res = analyse_part(part, cells, policies)
        if res is None:
            print(f"(no {part} data in the files)\n")
            continue
        verdicts[part] = print_part(part, cells, policies, res, out_rows)

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["part", "cell", "comparator", "metric", "n", "mean", "lo", "hi", "q",
                                          "passed", "family"])
        w.writeheader()
        w.writerows(out_rows)
    print("=" * 110)
    if "terrain" in verdicts:
        print(f"SUMMARY terrain x wind: G1 {'confirmed' if verdicts['terrain'][0] else 'NOT confirmed'}, "
              f"G2 {'confirmed' if verdicts['terrain'][1] else 'NOT confirmed'}")
    if "fleet" in verdicts:
        print(f"SUMMARY fleet size:     G3 coverage {'confirmed' if verdicts['fleet'][0] else 'NOT confirmed'}, "
              f"G3 delay/energy {'confirmed' if verdicts['fleet'][1] else 'NOT confirmed'}")
    print(f"Per-test results written to {args.out}. Report every line in the paper (see the plan, section 5).")


if __name__ == "__main__":
    main()
