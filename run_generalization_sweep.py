# -*- coding: utf-8 -*-
"""
run_generalization_sweep.py

Collects per-seed results of LARS, C, D, B and T_tuned in
  Part A: 4 terrains x 3 wind modes (12 cells), default fleet 4 patrol + 4 reserve
  Part B: fleet sizes 2x2, 3x3, 4x4, 6x6, 8x8 (N = 4, 6, 8, 12, 16) on Benapole Delta, static wind
for the seeds written in PREREGISTRATION_generalization_seeds601-700.txt.

READ THE PLAN FIRST AND COMMIT IT (git) BEFORE YOU RUN THIS FILE.

Usage (Git Bash, from the simulator folder). Four terrain jobs in parallel + one fleet job:
    python -u run_generalization_sweep.py --part terrain --terrains benapole_delta     --out gen_benapole.csv
    python -u run_generalization_sweep.py --part terrain --terrains sylhet_hills       --out gen_sylhet.csv
    python -u run_generalization_sweep.py --part terrain --terrains coxs_bazar         --out gen_coxs.csv
    python -u run_generalization_sweep.py --part terrain --terrains chittagong_hill_tracts --out gen_chittagong.csv
    python -u run_generalization_sweep.py --part fleet                                 --out gen_fleet.csv

Quick self-test (2 seeds, one cell, prints every policy's numbers; do this first):
    python -u run_generalization_sweep.py --part terrain --terrains sylhet_hills --winds static --seeds 601-602 --out selftest.csv

Options:
    --seeds 601-700       seed range (default: the plan's 601-700)
    --policies LARS,C,D,B,T_tuned
    --winds static,dynamic,stochastic
    --fleets 2x2,3x3,4x4,6x6,8x8         (patrol x reserve), used by --part fleet
    --max-steps 500
    --list                print the terrain names the simulator knows, then exit

The file is RESUMABLE: rows are appended after every seed, and a re-run with the same
--out skips (terrain, wind, fleet, seed) combinations that are already in the file.
The first call of a terrain downloads SRTM tiles (internet needed once).
Rough time (estimates from earlier runs, ~3 s per policy and seed): about 1.5-2 h per
terrain job and about 3 h for the fleet job at 100 seeds.
"""

import argparse
import csv
import os
import sys
import time

import a2e_relay_fixed_final_51 as relay
from collect_battery_fields import build_selectors, Tracked

DISPLAY_TO_INTERNAL = {"LARS": "I", "LARS_tuned": "I_lite"}
METRICS = ["coverage_pct", "avg_relay_delay_steps", "avg_relay_energy_pct",
           "total_energy_consumed", "relay_success_rate", "stranded_events", "mission_completed"]
KEYS = ["part", "terrain", "wind", "num_drones", "num_reserves", "seed"]


def parse_seeds(s):
    if "-" in s:
        a, b = s.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(",")]


def parse_fleets(s):
    out = []
    for tok in s.split(","):
        a, b = tok.lower().split("x")
        out.append((int(a), int(b)))
    return out


def load_done(path):
    done = set()
    if os.path.exists(path):
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                done.add((r["part"], r["terrain"], r["wind"], int(r["num_drones"]),
                          int(r["num_reserves"]), int(r["seed"])))
    return done


def set_wind(mode, seed, max_steps):
    if mode == "stochastic":
        relay.a20.set_wind_mode("stochastic", seed=seed, n_steps=max_steps + 1)
    else:
        relay.a20.set_wind_mode(mode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["terrain", "fleet"], default="terrain")
    ap.add_argument("--terrains", default="benapole_delta,sylhet_hills,coxs_bazar,chittagong_hill_tracts")
    ap.add_argument("--winds", default="static,dynamic,stochastic")
    ap.add_argument("--fleets", default="2x2,3x3,4x4,6x6,8x8")
    ap.add_argument("--seeds", default="601-700")
    ap.add_argument("--policies", default="LARS,C,D,B,T_tuned")
    ap.add_argument("--max-steps", type=int, default=500)
    ap.add_argument("--out", default="")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list:
        print("terrain scenarios known to the simulator:", list(relay.a20.TERRAIN_SCENARIOS.keys()))
        return

    seeds = parse_seeds(args.seeds)
    display = [p.strip() for p in args.policies.split(",") if p.strip()]
    internal = [DISPLAY_TO_INTERNAL.get(p, p) for p in display]
    sels = build_selectors(internal)
    tracked = {d: Tracked(sels[i]) for d, i in zip(display, internal)}

    if args.part == "terrain":
        terrains = [t.strip() for t in args.terrains.split(",") if t.strip()]
        winds = [w.strip() for w in args.winds.split(",") if w.strip()]
        fleet = (relay.NUM_DRONES, relay.RESERVE_POOL_SIZE)
        cells = [(t, w, fleet[0], fleet[1]) for t in terrains for w in winds]
    else:
        cells = [("benapole_delta", "static", a, b) for a, b in parse_fleets(args.fleets)]

    out = args.out or f"gen_{args.part}.csv"
    fieldnames = KEYS + [f"{p}_{m}" for p in display for m in METRICS + ["reserve_share"]]
    done = load_done(out)
    new_file = not os.path.exists(out) or os.path.getsize(out) == 0
    print(f"Part {args.part}: {len(cells)} cells x {len(seeds)} seeds x policies {display} -> {out}")
    print(f"Already done in the file: {len(done)} (cell, seed) rows")

    cur_terrain = None
    t_start = time.time()
    try:
        with open(out, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fieldnames)
            if new_file:
                w.writeheader()
                fh.flush()
            for (terrain, wind, nd, nr) in cells:
                todo = [s for s in seeds if (args.part, terrain, wind, nd, nr, s) not in done]
                if not todo:
                    print(f"[skip] {terrain}/{wind}/{nd}x{nr}: all seeds already done")
                    continue
                if terrain != cur_terrain:
                    relay.a20.set_terrain_scenario(terrain)
                    cur_terrain = terrain
                t_cell = time.time()
                for k, seed in enumerate(todo, 1):
                    set_wind(wind, seed, args.max_steps)
                    row = {"part": args.part, "terrain": terrain, "wind": wind,
                           "num_drones": nd, "num_reserves": nr, "seed": seed}
                    for d in display:
                        tr = tracked[d]
                        tr.reset()
                        m = relay.simulate_fleet(seed, relay_selector=tr, num_drones=nd,
                                                 num_reserves=nr, max_steps=args.max_steps)
                        for key in METRICS:
                            v = m.get(key)
                            row[f"{d}_{key}"] = int(v) if isinstance(v, bool) else v
                        row[f"{d}_reserve_share"] = tr.share()
                    w.writerow(row)
                    fh.flush()
                    if k % 10 == 0 or k == len(todo):
                        print(f"  {terrain}/{wind}/{nd}x{nr}: {k}/{len(todo)} seeds "
                              f"({(time.time() - t_cell) / 60:.1f} min)", flush=True)
                print(f"[done] {terrain}/{wind}/{nd}x{nr} in {(time.time() - t_cell) / 60:.1f} min "
                      f"(total {(time.time() - t_start) / 60:.1f} min)", flush=True)
    finally:
        try:
            relay.a20.set_wind_mode("static")
            relay.a20.set_terrain_scenario("benapole_delta")
        except Exception as e:  # restoring defaults must never hide the real error
            print("could not restore default terrain/wind:", e)

    print(f"\nFinished. Analyse with:\n  python analyze_generalization.py --csv {out} [--csv other.csv ...]")


if __name__ == "__main__":
    main()
