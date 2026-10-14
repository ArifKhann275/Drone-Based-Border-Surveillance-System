# -*- coding: utf-8 -*-
"""
bh_from_saved_csvs.py
══════════════════════════════════════════════════════════════
Recovers from run_full_comparison_with_j.py's bh_correct() crash
WITHOUT re-running any simulation. That crash happened after both
sweeps had already finished and their raw per-seed CSVs were already
written -- this script reads those CSVs back and recomputes the paired
Wilcoxon tests + BH correction itself, in seconds.

USAGE
-----
    python bh_from_saved_csvs.py \\
        --terrain-wind-csv terrain_wind_sweep_with_j.csv \\
        --fleet-size-csv fleet_size_sweep_with_j.csv \\
        --out-csv sweep_significance_bh_with_j.csv
"""
import argparse
import csv
from collections import defaultdict

import numpy as np
from scipy.stats import wilcoxon


def _safe_wilcoxon(a, b):
    a, b = np.array(a, dtype=float), np.array(b, dtype=float)
    if np.all(a - b == 0):
        return 1.0
    return wilcoxon(a, b).pvalue


def bh_correct(pvals):
    indexed = [(i, p) for i, p in enumerate(pvals) if p is not None]
    n = len(indexed)
    if n == 0:
        return [None] * len(pvals)
    indexed.sort(key=lambda x: x[1])
    q = [None] * len(pvals)
    prev_q = 1.0
    for rank in range(n, 0, -1):
        i, p = indexed[rank - 1]
        q_val = min(prev_q, p * n / rank)
        q[i] = q_val
        prev_q = q_val
    return q


def _read_csv(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


METRICS = ("coverage_pct", "avg_relay_delay_steps", "avg_relay_energy_pct")


def collect_terrain_wind(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[(r["terrain"], r["wind_mode"])].append(r)

    records = []
    for (terrain, wind_mode), grp in groups.items():
        for metric in METRICS:
            j_key, d_key = f"J_{metric}", f"D_{metric}"
            if j_key not in grp[0] or d_key not in grp[0]:
                continue
            j_vals = [float(r[j_key]) for r in grp]
            d_vals = [float(r[d_key]) for r in grp]
            p = _safe_wilcoxon(j_vals, d_vals)
            records.append({
                "sweep": "terrain_wind", "condition": f"{terrain}/{wind_mode}",
                "metric": metric, "policy": "J",
                "j_mean": round(float(np.mean(j_vals)), 4),
                "d_mean": round(float(np.mean(d_vals)), 4),
                "p": p,
            })
    return records


def collect_fleet_size(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[r["fleet_size"]].append(r)

    records = []
    for fleet_size, grp in groups.items():
        num_drones = grp[0].get("num_drones", "?")
        num_reserves = grp[0].get("num_reserves", "?")
        for metric in METRICS:
            j_key, d_key = f"J_{metric}", f"D_{metric}"
            if j_key not in grp[0] or d_key not in grp[0]:
                continue
            j_vals = [float(r[j_key]) for r in grp]
            d_vals = [float(r[d_key]) for r in grp]
            p = _safe_wilcoxon(j_vals, d_vals)
            records.append({
                "sweep": "fleet_size",
                "condition": f"N={fleet_size} (patrol={num_drones},reserve={num_reserves})",
                "metric": metric, "policy": "J",
                "j_mean": round(float(np.mean(j_vals)), 4),
                "d_mean": round(float(np.mean(d_vals)), 4),
                "p": p,
            })
    return records


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--terrain-wind-csv", default="terrain_wind_sweep_with_j.csv")
    p.add_argument("--fleet-size-csv", default="fleet_size_sweep_with_j.csv")
    p.add_argument("--out-csv", default="sweep_significance_bh_with_j.csv")
    p.add_argument("--alpha", type=float, default=0.05)
    args = p.parse_args()

    records = []
    try:
        tw_rows = _read_csv(args.terrain_wind_csv)
        records += collect_terrain_wind(tw_rows)
        print(f"Loaded {len(tw_rows)} rows from {args.terrain_wind_csv}")
    except FileNotFoundError:
        print(f"WARNING: {args.terrain_wind_csv} not found, skipping terrain/wind sweep.")

    try:
        fs_rows = _read_csv(args.fleet_size_csv)
        records += collect_fleet_size(fs_rows)
        print(f"Loaded {len(fs_rows)} rows from {args.fleet_size_csv}")
    except FileNotFoundError:
        print(f"WARNING: {args.fleet_size_csv} not found, skipping fleet-size sweep.")

    if not records:
        raise SystemExit("No data found -- check the CSV paths.")

    pvals = [r["p"] for r in records]
    qvals = bh_correct(pvals)
    for r, q in zip(records, qvals):
        r["q_bh"] = round(q, 6) if q is not None else None
        r["survives_bh"] = (q is not None and q < args.alpha)

    with open(args.out_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["sweep", "condition", "metric", "policy",
                                                "j_mean", "d_mean", "p", "q_bh", "survives_bh"])
        writer.writeheader()
        writer.writerows(records)

    n = len(records)
    n_survive = sum(1 for r in records if r["survives_bh"])
    print(f"\n{'='*100}\nJ vs D -- BH correction across all {n} tests "
          f"(terrain x wind + fleet size, all 3 metrics)\n{'='*100}")
    for r in sorted(records, key=lambda x: x["p"]):
        flag = "  <-- SURVIVES BH" if r["survives_bh"] else ""
        print(f"  [{r['sweep']:<12}] {r['condition']:<45} {r['metric']:<24} "
              f"J={r['j_mean']:<10} D={r['d_mean']:<10} p={r['p']:.5f}  "
              f"q_bh={r['q_bh']:.5f}{flag}")

    print(f"\n{n_survive}/{n} tests survive BH correction at alpha={args.alpha}.")
    print(f"Written: {args.out_csv}")


if __name__ == "__main__":
    main()
