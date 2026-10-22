# -*- coding: utf-8 -*-
"""
compare_focal_vs_all.py

Compares ONE focal policy (e.g. I or I_lite) with every other policy that is
present in the per-seed CSV files, on coverage, delay and energy, with paired-t
95% CI, Wilcoxon p, rank-biserial r, BH q over all difference tests, and a
coverage TOST (margin +/-2.0 points, BH over the TOST tests).

NO NEW SIMULATION IS NEEDED if the per-seed files already exist:
  battery_201-400_static.csv   (from collect_battery_fields.py: J, I, I_lite, ...)
  auction_test_fresh.csv       (J, B, AUC, T_tuned, V_tuned)
  tuned_test_fresh.csv         (J, T_tuned, V_tuned)
Policies are merged by seed; the first file that contains a policy wins (they
are deterministic per seed, so duplicates agree).

Why this exists: if Policy I (absolute-scale RelayScore) becomes the headline
contribution, it needs the same comparison set that J has. In particular
"I vs C" is the DIRECT test of the paper's claim that absolute-scale
normalization helps; the paper so far only tested J vs C and J vs I.
Collect the missing policies, e.g.:
    python -u collect_battery_fields.py --seeds 201-400 --wind static --policies C,D,D2,T,V,B,I,I_lite
(about 10-15 minutes per policy per 200 seeds)

Usage:
    python compare_focal_vs_all.py --focal I \
        --wide battery_201-400_static.csv --wide auction_test_fresh.csv
    python compare_focal_vs_all.py --focal I_lite --wide ... --wide ...
Options: --margin 2.0   --metrics coverage_pct,avg_relay_delay_steps,avg_relay_energy_pct
"""

import argparse
import csv
import numpy as np
from scipy.stats import wilcoxon, rankdata, t as t_dist

DEFAULT_METRICS = ["coverage_pct", "avg_relay_delay_steps", "avg_relay_energy_pct"]
KNOWN_METRICS = set(DEFAULT_METRICS) | {"total_energy_consumed", "reserve_share"}


def to_float(x):
    try:
        v = float(x)
        return v if np.isfinite(v) else np.nan
    except (TypeError, ValueError):
        return np.nan


def read_wide(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def collect_policies(files, metrics):
    """-> {policy: {metric: {seed: value}}}"""
    data = {}
    for path in files:
        rows = read_wide(path)
        if not rows:
            continue
        cols = list(rows[0].keys())
        for c in cols:
            if c == "seed":
                continue
            for m in sorted(KNOWN_METRICS | set(metrics), key=len, reverse=True):
                if c.endswith("_" + m):
                    pol = c[: -(len(m) + 1)]
                    if pol in data and m in data[pol]:
                        break   # first file wins
                    data.setdefault(pol, {}).setdefault(m, {})
                    for r in rows:
                        data[pol][m][int(to_float(r["seed"]))] = to_float(r[c])
                    break
    return data


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
    se = d.std(ddof=1) / np.sqrt(n)
    h = t_dist.ppf(0.5 + level / 2, n - 1) * se
    return float(m), float(m - h), float(m + h)


def tost(d, margin):
    n = len(d)
    se = d.std(ddof=1) / np.sqrt(n)
    if se == 0:
        return 0.0 if abs(d.mean()) < margin else 1.0, (float(d.mean()),) * 2
    p_low = 1.0 - t_dist.cdf((d.mean() + margin) / se, n - 1)
    p_high = t_dist.cdf((d.mean() - margin) / se, n - 1)
    h90 = t_dist.ppf(0.95, n - 1) * se
    return float(max(p_low, p_high)), (float(d.mean() - h90), float(d.mean() + h90))


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--focal", required=True)
    ap.add_argument("--wide", action="append", required=True)
    ap.add_argument("--metrics", default=",".join(DEFAULT_METRICS))
    ap.add_argument("--margin", type=float, default=2.0)
    ap.add_argument("--only", default="", help="comma list of comparators (default: all found)")
    args = ap.parse_args()
    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]

    data = collect_policies(args.wide, metrics)
    if args.focal not in data:
        raise SystemExit(f"Focal policy '{args.focal}' not found. Policies present: {sorted(data)}")
    others = [p for p in sorted(data) if p != args.focal]
    if args.only:
        keep = [x.strip() for x in args.only.split(",")]
        others = [p for p in others if p in keep]
    print(f"Focal policy: {args.focal}   comparators found: {', '.join(others)}")
    print("Difference = focal - comparator. For delay/energy a negative value favors the focal policy.\n")

    results = []
    for o in others:
        for m in metrics:
            if m not in data[args.focal] or m not in data[o]:
                continue
            seeds = sorted(set(data[args.focal][m]) & set(data[o][m]))
            a = np.array([data[args.focal][m][s] for s in seeds])
            b = np.array([data[o][m][s] for s in seeds])
            ok = np.isfinite(a) & np.isfinite(b)
            a, b = a[ok], b[ok]
            if len(a) < 3:
                continue
            d = a - b
            mean, lo, hi = paired_ci(d)
            r = dict(o=o, m=m, n=len(a), fa=float(a.mean()), fb=float(b.mean()), diff=mean, lo=lo, hi=hi,
                     p=wilcoxon_p(d), r=rank_biserial(d))
            if m == "coverage_pct":
                r["p_tost"], r["ci90"] = tost(d, args.margin)
            results.append(r)

    q = bh_adjust([r["p"] for r in results])
    tidx = [i for i, r in enumerate(results) if "p_tost" in r]
    qt = bh_adjust([results[i]["p_tost"] for i in tidx])
    for r, x in zip(results, q):
        r["q"] = float(x)
    for i, x in zip(tidx, qt):
        results[i]["q_tost"] = float(x)

    print(f"{'vs':<10}{'metric':<24}{'n':>5}{'focal':>9}{'other':>9}{'diff [95% CI]':>26}{'p':>9}{'q_BH':>9}{'r':>7}"
          f"   TOST(+/-{args.margin:g}) q, 90% CI")
    for r in results:
        t = ""
        if "q_tost" in r:
            t = (f"{r['q_tost']:.4f} {'EQUIV' if r['q_tost'] < 0.05 else 'not equiv':<9} "
                 f"[{r['ci90'][0]:+.2f},{r['ci90'][1]:+.2f}]")
        print(f"{r['o']:<10}{r['m']:<24}{r['n']:>5}{r['fa']:>9.3f}{r['fb']:>9.3f}"
              f"{r['diff']:>+10.2f} [{r['lo']:+.2f},{r['hi']:+.2f}]{r['p']:>9.4f}{r['q']:>9.4f}{r['r']:>7.2f}   {t}")
    print(f"\nBH family: {len(results)} difference tests; TOST family: {len(tidx)} tests.")
    print("If you also report J-vs-X tests from earlier analyses, state that this is a separate family.")


if __name__ == "__main__":
    main()