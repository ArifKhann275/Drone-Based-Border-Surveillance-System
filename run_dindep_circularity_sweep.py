# -*- coding: utf-8 -*-
"""
run_dindep_circularity_sweep.py  --  re-runs the rule-base circularity
check (Section V-D / rule_base_circularity) across the FULL 12-cell
terrain x wind sweep and the 4-size fleet sweep, using the CORRECTED
FIS -- not the single-condition, pre-audit numbers currently sitting in
the paper draft's Table VIII / D-indep paragraph.

WHY THIS IS URGENT, NOT OPTIONAL
---------------------------------
select_fuzzy_relay_true_fis_indep_weights already routes through the
fixed policy_d_true_fis.py engine (corrected membership orientation,
corrected coverage-loss universe, proven non-compensatory rule base --
Section rule_base_audit's eps*=-0.25 for the 'indep' profile). So this
script does NOT need any new fix -- it only needs to be RUN, at the
same scale as every other sweep in this paper, before the paper cites
D-indep results at all. Until it is run, the circularity section is
citing numbers computed under the OLD, buggy FIS, which is exactly the
kind of stale-baseline problem this paper spends two sections warning
about elsewhere.

WHAT IT PRODUCES
-----------------
- dindep_terrain_wind_sweep.csv   (raw per-seed, 12 cells x policies C/D/Dindep)
- dindep_fleet_size_sweep.csv     (raw per-seed, 4 fleet sizes x policies C/D/Dindep)
- dindep_circularity_bh.csv       (every C-vs-D, C-vs-Dindep, D-vs-Dindep
                                   test across both sweeps, BH-corrected
                                   together as its own family -- this is
                                   a DIFFERENT family from the main
                                   240-test one in run_sweeps.py, because
                                   D-indep was never part of that family)
- printed summary table, ready to replace the stale paragraph in
  Section rule_base_circularity

USAGE
-----
    python run_dindep_circularity_sweep.py --seeds 50   # full paper scale
    python run_dindep_circularity_sweep.py --seeds 10   # quick check first
"""

import argparse
import csv

import numpy as np
from scipy.stats import wilcoxon

import a20
import a2e_relay_fixed_final_51 as relay
from policy_d_true_fis import (select_fuzzy_relay_true_fis_indep_weights,
                               select_fuzzy_relay_true_fis_indep2_weights)

TERRAINS = ("benapole_delta", "sylhet_hills", "coxs_bazar", "chittagong_hill_tracts")
WINDS = ("static", "dynamic", "stochastic")
FLEET_SIZES = ((1, 1), (2, 2), (4, 4), (8, 8))
METRICS = ("coverage_pct", "avg_relay_delay_steps", "avg_relay_energy_pct")
POLICY_LABELS = ("C", "D", "Dindep", "Dindep2")
# NOTE: "indep" alone is a weak circularity test -- its three vetoes all
# reference coverage_loss, so they never fire for an idle-reserve candidate
# (coverage_loss=0=best by construction) and it collapses onto "expert" on
# 77/81 cells. "indep2" (see policy_d_true_fis.INDEPENDENT2_RULE_WEIGHTS) is
# a second, independently-authored profile whose vetoes/gates never
# reference coverage_loss, so it stays genuinely distinct across the
# reserve-dominated decision space. Both are reported; do not drop Dindep,
# since the fact that it collapses onto D is itself a reportable finding.
PAIRS = (("C", "D"), ("C", "Dindep"), ("D", "Dindep"),
         ("C", "Dindep2"), ("D", "Dindep2"))


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=50)
    p.add_argument("--steps", type=int, default=500,
                   help="paper's Table III mission length (NOT a20.MAX_STEPS)")
    p.add_argument("--allow-synthetic", action="store_true")
    p.add_argument("--terrain-out", default="dindep_terrain_wind_sweep.csv")
    p.add_argument("--fleet-out", default="dindep_fleet_size_sweep.csv")
    p.add_argument("--bh-out", default="dindep_circularity_bh.csv")
    return p.parse_args()


def _check_real_terrain():
    src = getattr(a20, "TERRAIN_SOURCE", None)
    if src is None:
        try:
            a20.set_terrain_scenario("benapole_delta", use_real_srtm=True)
            src = getattr(a20, "TERRAIN_SOURCE", None)
        except Exception:                                        # noqa: BLE001
            src = None
    return src in ("real_srtm", "real_srtm_export"), src


def preflight(allow_synthetic):
    ok, src = _check_real_terrain()
    print(f"TERRAIN_SOURCE = {src!r}")
    if not ok:
        print("SYNTHETIC terrain fallback detected. Every number this sweep")
        print("produces would NOT match the paper's real-SRTM tables.")
        if not allow_synthetic:
            raise SystemExit("Refusing to run on synthetic terrain "
                             "(pass --allow-synthetic for a code smoke test only).")
    print("Preflight OK.\n")


def run_selector(label):
    if label == "C":
        return relay.select_best_relay
    if label == "D":
        return relay.POLICY_SELECTORS["D"]
    if label == "Dindep":
        return select_fuzzy_relay_true_fis_indep_weights
    if label == "Dindep2":
        return select_fuzzy_relay_true_fis_indep2_weights
    raise ValueError(label)


def collect_terrain_wind(seeds, steps):
    rows = []
    for terrain in TERRAINS:
        for wind in WINDS:
            a20.set_terrain_scenario(terrain, use_real_srtm=True)
            for seed in seeds:
                a20.set_wind_mode(wind, seed=seed)   # stochastic mode requires a seed
                row = {"terrain": terrain, "wind_mode": wind, "seed": seed}
                for label in POLICY_LABELS:
                    m = relay.simulate_fleet(seed, relay_selector=run_selector(label),
                                             max_steps=steps)
                    for k in METRICS:
                        row[f"{label}_{k}"] = m.get(k)
                rows.append(row)
            print(f"  [{terrain}/{wind}] done ({len(seeds)} seeds)")
    a20.set_terrain_scenario("benapole_delta", use_real_srtm=True)
    return rows


def collect_fleet_size(seeds, steps):
    rows = []
    for nd, nr in FLEET_SIZES:
        for seed in seeds:
            row = {"num_drones": nd, "num_reserves": nr, "seed": seed}
            for label in POLICY_LABELS:
                m = relay.simulate_fleet(seed, relay_selector=run_selector(label),
                                         num_drones=nd, num_reserves=nr,
                                         max_steps=steps)
                for k in METRICS:
                    row[f"{label}_{k}"] = m.get(k)
            rows.append(row)
        print(f"  [Nf={nd+nr}] done ({len(seeds)} seeds)")
    return rows


def _write_csv(rows, path):
    if not rows:
        return
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Written: {path}")


def _wtest(x, y):
    d = x - y
    return wilcoxon(x, y).pvalue if np.any(d != 0) else 1.0


def build_bh_family(tw_rows, fs_rows):
    import pandas as pd
    tw = pd.DataFrame(tw_rows)
    fs = pd.DataFrame(fs_rows)

    tests = []
    for terrain in TERRAINS:
        for wind in WINDS:
            sub = tw[(tw.terrain == terrain) & (tw.wind_mode == wind)]
            for m in METRICS:
                for a, b in PAIRS:
                    x = sub[f"{a}_{m}"].astype(float)
                    y = sub[f"{b}_{m}"].astype(float)
                    pv = _wtest(x, y)
                    tests.append({"sweep": "terrain_wind", "condition": f"{terrain}/{wind}",
                                 "metric": m, "pair": f"{a}_vs_{b}", "p": pv,
                                 "mean_a": x.mean(), "mean_b": y.mean()})
    for nd, nr in FLEET_SIZES:
        sub = fs[(fs.num_drones == nd) & (fs.num_reserves == nr)]
        for m in METRICS:
            for a, b in PAIRS:
                x = sub[f"{a}_{m}"].astype(float)
                y = sub[f"{b}_{m}"].astype(float)
                pv = _wtest(x, y)
                tests.append({"sweep": "fleet_size", "condition": f"Nf={nd+nr}",
                             "metric": m, "pair": f"{a}_vs_{b}", "p": pv,
                             "mean_a": x.mean(), "mean_b": y.mean()})

    pvals = np.array([t["p"] for t in tests])
    order = np.argsort(pvals)
    n = len(pvals)
    q = np.empty(n)
    prev = 1.0
    for rank, idx in enumerate(order[::-1]):
        real_rank = n - rank
        val = pvals[idx] * n / real_rank
        prev = min(prev, val)
        q[idx] = prev
    for t, qq in zip(tests, q):
        t["q_bh"] = qq
        t["survives_bh"] = bool(qq < 0.05)
    return tests


def main():
    args = _parse_args()
    seeds = list(range(1, args.seeds + 1))

    preflight(args.allow_synthetic)

    print("Collecting terrain x wind sweep (C, D, Dindep)...")
    tw_rows = collect_terrain_wind(seeds, args.steps)
    _write_csv(tw_rows, args.terrain_out)

    print("\nCollecting fleet-size sweep (C, D, Dindep)...")
    fs_rows = collect_fleet_size(seeds, args.steps)
    _write_csv(fs_rows, args.fleet_out)

    n_tests = len(TERRAINS) * len(WINDS) * len(METRICS) * len(PAIRS) \
        + len(FLEET_SIZES) * len(METRICS) * len(PAIRS)
    print(f"\nBH-correcting the D-indep circularity family ({n_tests} tests)...")
    tests = build_bh_family(tw_rows, fs_rows)
    _write_csv(tests, args.bh_out)

    survivors = [t for t in tests if t["survives_bh"]]
    print(f"\n{len(tests)} tests total, {len(survivors)} survive BH (q<0.05):\n")
    print(f"{'sweep':<14}{'condition':<28}{'metric':<26}{'pair':<14}{'p':>10}{'q':>10}")
    print("-" * 100)
    for t in sorted(survivors, key=lambda t: t["q_bh"]):
        print(f"{t['sweep']:<14}{t['condition']:<28}{t['metric']:<26}{t['pair']:<14}"
              f"{t['p']:>10.3g}{t['q_bh']:>10.3g}")

    for pair_label in ("D_vs_Dindep", "D_vs_Dindep2"):
        d_vs_x = [t for t in tests if t["pair"] == pair_label]
        if not d_vs_x:
            continue
        x_sig = [t for t in d_vs_x if t["p"] < 0.05]
        print(f"\n{pair_label.replace('_', ' ')}: {len(x_sig)}/{len(d_vs_x)} individually "
              f"significant, {sum(t['survives_bh'] for t in d_vs_x)} survive BH.")
    print("\nDindep's vetoes/gates all reference coverage_loss, so it never")
    print("diverges from D for reserve-dominated decisions (coverage_loss is")
    print("always best for a reserve candidate) -- expect it to look near-")
    print("identical to D. Dindep2's vetoes/gates never reference")
    print("coverage_loss, so it stays active across that subspace and is the")
    print("more informative circularity check; report both, and explain the")
    print("Dindep collapse as a finding rather than dropping it.")


if __name__ == "__main__":
    main()
