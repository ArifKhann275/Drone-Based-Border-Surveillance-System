# -*- coding: utf-8 -*-
"""
analyze_battery_safety.py

Paired comparison of two policies (default: J vs I) on the battery-safety and
reliability fields that the simulator saves per seed, with BH correction over
the informative safety fields.

WRITE THE HYPOTHESIS DOWN BEFORE YOU RUN IT (suggested wording):
  H1: J (Fix 2: battery margin ranked) ends missions with a higher mean patrol
      battery and/or fewer stranded events than I, at equal coverage.
The family below is fixed in advance (all non-constant, non-duplicate safety
fields); report the result whichever way it goes.

INPUT FORMATS (pick one)
  1) wide CSV with columns "<policy>_<field>" and a "seed" column, e.g. the file
     written by collect_battery_fields.py, or auction_test_fresh.csv:
         python analyze_battery_safety.py --wide battery_fresh_static.csv --a J --b I
  2) two CSV files, each with a "seed" column and the plain field names
     (optionally the same prefix in both with --prefix-a / --prefix-b):
         python analyze_battery_safety.py --file-a J_fresh.csv --file-b I_fresh.csv
  Use --fields to choose fields yourself (comma separated).

WHAT THE SIMULATOR'S FIELDS CAN AND CANNOT SHOW (check before you hope):
  * stranded_events counts steps whose required airspeed exceeded the hardware
    cap; per the simulator this is only possible under STOCHASTIC wind. Under
    static wind it is 0 for every seed, so it is reported as "constant".
  * avg_reserve_battery is constant if reserves are always fully charged.
  * avg_final_battery and avg_patrol_battery are the same quantity in the
    simulator; the duplicate is detected and skipped.
  So under the default (static-wind) condition the informative safety fields
  are essentially avg_patrol_battery and relay_success_rate. To test stranded
  events, collect data under stochastic wind (collect_battery_fields.py --wind
  stochastic).

Output per field: mean A, mean B, A-B, paired-t 95% CI, Wilcoxon p, rank-biserial
r, win/tie/loss counts, BH q (within the informative safety family).
"""

import argparse
import csv
import sys
import numpy as np
from scipy.stats import wilcoxon, rankdata, t as t_dist

SAFETY_FIELDS = [
    "avg_patrol_battery", "avg_final_battery", "avg_reserve_battery",
    "stranded_events", "relay_success_rate", "relay_requested", "relay_fulfilled",
]
CONTEXT_FIELDS = [
    "coverage_pct", "avg_relay_delay_steps", "avg_relay_energy_pct", "total_energy_consumed",
]
HIGHER_BETTER = {"avg_patrol_battery", "avg_final_battery", "avg_reserve_battery",
                 "relay_success_rate", "relay_fulfilled", "coverage_pct"}
LOWER_BETTER = {"stranded_events", "avg_relay_delay_steps", "avg_relay_energy_pct",
                "total_energy_consumed"}


def rank_biserial(d):
    nz = d[d != 0]
    if len(nz) == 0:
        return 0.0
    ranks = rankdata(np.abs(nz))
    pos, neg = ranks[nz > 0].sum(), ranks[nz < 0].sum()
    return float((pos - neg) / (pos + neg))


def wilcoxon_p(d):
    return float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0


def paired_ci(d, level=0.95):
    n = len(d)
    m = d.mean()
    if n < 2 or d.std(ddof=1) == 0:
        return float(m), float(m), float(m)
    se = d.std(ddof=1) / np.sqrt(n)
    h = t_dist.ppf(0.5 + level / 2, n - 1) * se
    return float(m), float(m - h), float(m + h)


def bh_adjust(p):
    p = np.asarray(p, float)
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    adj = np.empty(n)
    prev = 1.0
    for k, idx in enumerate(order[::-1]):
        rank = n - k
        prev = min(prev, p[idx] * n / rank)
        adj[idx] = prev
    return adj


def read_csv(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"{path}: empty file")
    return rows


def to_float(x):
    try:
        v = float(x)
        return v if np.isfinite(v) else np.nan
    except (TypeError, ValueError):
        # booleans written as True/False
        if str(x).strip().lower() == "true":
            return 1.0
        if str(x).strip().lower() == "false":
            return 0.0
        return np.nan


def column(rows, name):
    return np.array([to_float(r.get(name, "")) for r in rows], float)


def load_pairs(args):
    """Return (seeds, {field: (a_vector, b_vector)}, label_a, label_b)."""
    if args.wide:
        rows = read_csv(args.wide)
        seeds = np.array([to_float(r["seed"]) for r in rows])
        cols = rows[0].keys()
        fields = {}
        pa, pb = f"{args.a}_", f"{args.b}_"
        for c in cols:
            if c.startswith(pa):
                f = c[len(pa):]
                if f"{pb}{f}" in cols:
                    fields[f] = (column(rows, c), column(rows, f"{pb}{f}"))
        if not fields:
            raise SystemExit(f"No columns '{pa}<field>' / '{pb}<field>' found in {args.wide}.\n"
                             f"Columns present: {list(cols)[:12]} ...")
        return seeds, fields, args.a, args.b
    if args.file_a and args.file_b:
        ra, rb = read_csv(args.file_a), read_csv(args.file_b)
        da = {int(to_float(r["seed"])): r for r in ra}
        db = {int(to_float(r["seed"])): r for r in rb}
        common = sorted(set(da) & set(db))
        if not common:
            raise SystemExit("The two files share no seeds.")
        ra = [da[s] for s in common]
        rb = [db[s] for s in common]
        fields = {}
        for c in ra[0].keys():
            if c == "seed":
                continue
            ca = c
            cb = c
            if args.prefix_a and c.startswith(args.prefix_a):
                cb = args.prefix_b + c[len(args.prefix_a):] if args.prefix_b else c
            if cb in rb[0]:
                f = c[len(args.prefix_a):] if args.prefix_a and c.startswith(args.prefix_a) else c
                fields[f] = (column(ra, ca), column(rb, cb))
        return np.array(common, float), fields, args.file_a, args.file_b
    raise SystemExit("Give either --wide FILE --a X --b Y, or --file-a FILE --file-b FILE.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wide")
    ap.add_argument("--a", default="J")
    ap.add_argument("--b", default="I")
    ap.add_argument("--file-a")
    ap.add_argument("--file-b")
    ap.add_argument("--prefix-a", default="")
    ap.add_argument("--prefix-b", default="")
    ap.add_argument("--fields", default="")
    args = ap.parse_args()

    seeds, fields, la, lb = load_pairs(args)
    n_all = len(seeds)
    wanted = [f.strip() for f in args.fields.split(",") if f.strip()] or (SAFETY_FIELDS + CONTEXT_FIELDS)
    missing = [f for f in wanted if f not in fields]
    wanted = [f for f in wanted if f in fields]
    print(f"Comparing A = {la}  vs  B = {lb}   (n = {n_all} paired seeds in file)")
    if missing:
        print(f"Fields not present in the data (skipped): {', '.join(missing)}")

    tested, skipped = [], []
    seen_vectors = {}
    for f in wanted:
        a, b = fields[f]
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() < 3:
            skipped.append((f, "fewer than 3 valid paired seeds"))
            continue
        a, b = a[ok], b[ok]
        d = a - b
        if np.all(a == a[0]) and np.all(b == a[0]):
            skipped.append((f, f"constant (= {a[0]:g} for both policies on every seed)"))
            continue
        key = (tuple(np.round(a, 9)), tuple(np.round(b, 9)))
        if key in seen_vectors:
            skipped.append((f, f"identical to field '{seen_vectors[key]}'"))
            continue
        seen_vectors[key] = f
        m, lo, hi = paired_ci(d)
        tested.append(dict(field=f, n=int(ok.sum()), ma=float(a.mean()), mb=float(b.mean()),
                           diff=m, lo=lo, hi=hi, p=wilcoxon_p(d), r=rank_biserial(d),
                           win=int((d > 0).sum()), tie=int((d == 0).sum()), loss=int((d < 0).sum())))

    safety = [t for t in tested if t["field"] in SAFETY_FIELDS]
    context = [t for t in tested if t["field"] not in SAFETY_FIELDS]
    for group in (safety, context):
        q = bh_adjust([t["p"] for t in group])
        for t, qq in zip(group, q):
            t["q"] = float(qq)

    def hint(f):
        if f in HIGHER_BETTER:
            return "higher=better"
        if f in LOWER_BETTER:
            return "lower=better"
        return "direction not set"

    def show(title, group):
        print(f"\n{title}  (BH within this group, m = {len(group)})")
        if not group:
            print("  (no informative field)")
            return
        print(f"{'field':<24}{'n':>5}{'A':>10}{'B':>10}{'A-B [95% CI]':>28}{'p':>9}{'q_BH':>9}{'r':>7}"
              f"{'A>B/=/<':>12}  direction")
        for t in group:
            ci = f"{t['diff']:+.3f} [{t['lo']:+.3f},{t['hi']:+.3f}]"
            wtl = f"{t['win']}/{t['tie']}/{t['loss']}"
            print(f"{t['field']:<24}{t['n']:>5}{t['ma']:>10.3f}{t['mb']:>10.3f}{ci:>28}"
                  f"{t['p']:>9.4f}{t['q']:>9.4f}{t['r']:>7.2f}{wtl:>12}  {hint(t['field'])}")

    show("SAFETY / RELIABILITY FIELDS", safety)
    show("CONTEXT (coverage, delay, energy)", context)

    if skipped:
        print("\nSkipped (not informative):")
        for f, why in skipped:
            print(f"  {f:<24} {why}")
    print("\nSign convention: A - B. For 'higher=better' fields a positive difference favors A; "
          "for 'lower=better' fields a negative difference favors A.")
    if not safety:
        print("\nNo informative safety field in this data. Under static wind stranded_events is 0 and the "
              "reserve battery is constant; collect data under stochastic wind to test stranded events.")


if __name__ == "__main__":
    main()
