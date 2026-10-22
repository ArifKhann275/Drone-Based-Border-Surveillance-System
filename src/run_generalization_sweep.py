# -*- coding: utf-8 -*-
"""
run_generalization_sweep.py  (v3, locked to PREREGISTRATION_confirmatory_seeds801-900.txt)

Modes
  --mode confirmatory  Benapole Delta, static wind, Np=4, Nr=4, seeds 801-900, policies and
                       flags exactly as in the plan. No other setting is accepted.
  --mode pipeline      seeds 9001-9002 only. Checks that the program runs and writes every
                       column. Its output must never be analysed as a result.
  --mode free          any terrain / wind / fleet / seeds. Exploratory only.

Before any row is written the script stops if: git reports uncommitted changes to watched
files, elevation_grid_export.json exists, a model flag differs from the plan, the terrain
source is not real_srtm, or the cell size differs from 499.6 m (confirmatory and pipeline).

Usage:
  python -u run_generalization_sweep.py --mode pipeline --out pipeline_check_9001_9002.csv
  python -u run_generalization_sweep.py --mode confirmatory --out confirm_801_900.csv 2>&1 | tee log_confirm_801_900.txt
"""

import argparse
import csv
import datetime
import os
import subprocess
import time
import traceback

import a2e_relay_fixed_final_51 as relay
from collect_battery_fields import build_selectors, Tracked

WATCHED = ["a20.py", "a2e_relay_fixed_final_51.py", "collect_battery_fields.py",
           "run_generalization_sweep.py", "analyze_generalization.py", "compare_focal_vs_all.py"]

FOCAL = "LARS"
PLAN_POLICIES = ["LARS", "C", "D", "D2", "T", "V", "AUC", "J"]
EXPLORATORY = ["B", "BR", "T_tuned", "V_tuned", "LARS_tuned"]
DEFAULT_POLICIES = ",".join(PLAN_POLICIES + EXPLORATORY)
DISPLAY_TO_INTERNAL = {"LARS": "I", "LARS_tuned": "I_lite"}   # all other names pass through

TERRAIN, WIND, NP, NR = "benapole_delta", "static", 4, 4
CONF_SEEDS = list(range(801, 901))
PIPE_SEEDS = [9001, 9002]
PLAN_CELL_SIZE = 499.6
PLAN_FLAGS = {
    "USE_NONLINEAR_BATTERY": True,
    "USE_SOH_AGING": True,
    "PATROL_MODE": "sse_gap",
    "SSE_TERRAIN_AWARE": False,
    "HY_SSE_W": 1.0,
    "TERRAIN_COST_ENABLED": True,
}

METRICS = ["coverage_pct", "avg_relay_delay_steps", "avg_relay_energy_pct",
           "total_energy_consumed", "relay_success_rate", "stranded_events", "mission_completed"]
KEYS = ["mode", "terrain", "wind", "num_drones", "num_reserves", "seed"]
PROV_KEYS = (["git_commit", "code_last_commit", "git_dirty", "terrain_source", "cell_size_m",
              "energy_per_cell", "num_drones_default", "reserve_default"]
             + [f"flag_{k}" for k in PLAN_FLAGS])


# ---------------------------------------------------------------- provenance
def git(*args):
    try:
        r = subprocess.run(["git", *args], capture_output=True, text=True, check=True)
        return r.stdout.strip()
    except Exception:
        return None


def git_dirty():
    out = git("status", "--porcelain", "--", *WATCHED)
    return True if out is None else out != ""


def provenance():
    a = relay.a20
    prov = {
        "git_commit": git("rev-parse", "HEAD") or "NO_GIT",
        "code_last_commit": git("log", "-1", "--format=%H", "--", *WATCHED) or "NO_GIT",
        "git_dirty": int(git_dirty()),
        "terrain_source": a.TERRAIN_SOURCE,
        "cell_size_m": round(a.OWT_CFG.cell_size_m, 3),
        "energy_per_cell": a.ENERGY_PER_CELL,
        "num_drones_default": relay.NUM_DRONES,
        "reserve_default": relay.RESERVE_POOL_SIZE,
    }
    for k in PLAN_FLAGS:
        prov[f"flag_{k}"] = getattr(a, k)
    return prov


def flag_problems():
    return [f"{k} = {getattr(relay.a20, k)!r}, plan says {v!r}"
            for k, v in PLAN_FLAGS.items() if getattr(relay.a20, k) != v]


def preflight(mode):
    errs = []
    if git_dirty():
        errs.append("git reports uncommitted changes to watched files: commit them first")
    if os.path.exists("elevation_grid_export.json"):
        errs.append("elevation_grid_export.json exists in this folder: move it away first")
    errs += flag_problems()
    if mode != "free" and (relay.NUM_DRONES, relay.RESERVE_POOL_SIZE) != (NP, NR):
        errs.append(f"fleet is {relay.NUM_DRONES}x{relay.RESERVE_POOL_SIZE}, plan needs {NP}x{NR}")
    return errs


# ---------------------------------------------------------------- helpers
def parse_seeds(s):
    if "-" in s:
        a, b = s.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(",")]


def parse_fleets(s):
    out = []
    for tok in [t for t in s.split(",") if t.strip()]:
        a, b = tok.lower().split("x")
        out.append((int(a), int(b)))
    return out


def load_done(path):
    done = set()
    if os.path.exists(path) and os.path.getsize(path) > 0:
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                done.add((r["mode"], r["terrain"], r["wind"], int(r["num_drones"]),
                          int(r["num_reserves"]), int(r["seed"])))
    return done


def check_resume(path, prov):
    """A resumed file must come from the same code commit and the same plan commit."""
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if r["git_commit"] != prov["git_commit"] or r["code_last_commit"] != prov["code_last_commit"]:
                raise SystemExit(f"{path} was written at commit {r['git_commit'][:8]} "
                                 f"(code {r['code_last_commit'][:8]}); current is "
                                 f"{prov['git_commit'][:8]} (code {prov['code_last_commit'][:8]}). "
                                 f"Archive that file and start a new --out.")


def set_wind(mode, seed, max_steps):
    if mode == "stochastic":
        relay.a20.set_wind_mode("stochastic", seed=seed, n_steps=max_steps + 1)
    else:
        relay.a20.set_wind_mode(mode)


def make_trackers(names):
    internal = [DISPLAY_TO_INTERNAL.get(n, n) for n in names]
    try:
        sels = build_selectors(internal)
    except Exception as e:
        raise SystemExit(f"build_selectors failed for {internal}: {e}")
    missing = [i for i in internal if i not in sels]
    if missing:
        raise SystemExit(f"build_selectors has no selector for: {missing}")
    return {n: Tracked(sels[i]) for n, i in zip(names, internal)}


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["confirmatory", "pipeline", "free"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--resume", action="store_true",
                    help="continue an interrupted run of the SAME commit (never after viewing results)")
    ap.add_argument("--seeds", default=None, help="free mode only (default 601-700)")
    ap.add_argument("--terrains", default="benapole_delta,sylhet_hills,coxs_bazar,chittagong_hill_tracts",
                    help="free mode only")
    ap.add_argument("--winds", default="static,dynamic,stochastic", help="free mode only")
    ap.add_argument("--fleets", default="", help="free mode only, benapole static, e.g. 2x2,4x4")
    ap.add_argument("--policies", default=DEFAULT_POLICIES, help="free mode only")
    ap.add_argument("--max-steps", type=int, default=500)
    args = ap.parse_args()

    # ---- fix the design from the mode; refuse anything the plan does not allow
    if args.mode == "free":
        seeds = parse_seeds(args.seeds or "601-700")
        policies = [p.strip() for p in args.policies.split(",") if p.strip()]
        terrains = [t.strip() for t in args.terrains.split(",") if t.strip()]
        winds = [w.strip() for w in args.winds.split(",") if w.strip()]
        cells = [(t, w, relay.NUM_DRONES, relay.RESERVE_POOL_SIZE) for t in terrains for w in winds]
        cells += [(TERRAIN, WIND, a, b) for a, b in parse_fleets(args.fleets)]
    else:
        if args.seeds is not None or args.fleets or args.policies != DEFAULT_POLICIES:
            raise SystemExit(f"--mode {args.mode} is locked: --seeds, --fleets and --policies "
                             f"are not accepted. Use --mode free for other designs.")
        seeds = CONF_SEEDS if args.mode == "confirmatory" else PIPE_SEEDS
        policies = PLAN_POLICIES + EXPLORATORY
        cells = [(TERRAIN, WIND, NP, NR)]

    # ---- preflight: stop before writing anything
    errs = preflight(args.mode)
    if errs:
        print("STOP (nothing written):")
        for e in errs:
            print("  -", e)
        raise SystemExit(1)

    prov0 = provenance()
    if os.path.exists(args.out) and os.path.getsize(args.out) > 0:
        if not args.resume:
            raise SystemExit(f"{args.out} already exists. Use a new --out name, or --resume "
                             f"only if this is the same interrupted run.")
        check_resume(args.out, prov0)

    trackers = make_trackers(policies)
    fieldnames = KEYS + PROV_KEYS + [f"{p}_{m}" for p in policies for m in METRICS + ["reserve_share"]]
    done = load_done(args.out)
    new_file = not os.path.exists(args.out) or os.path.getsize(args.out) == 0

    print(f"mode={args.mode}  started={datetime.datetime.now().isoformat(timespec='seconds')}")
    print(f"cells={cells}  seeds={seeds[0]}-{seeds[-1]} ({len(seeds)})  policies={policies}")
    print(f"already in file: {len(done)} rows")
    print("provenance:", prov0, flush=True)

    cur_terrain, last = None, None
    t0 = time.time()
    try:
        with open(args.out, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fieldnames)
            if new_file:
                w.writeheader()
                fh.flush()
            for (terrain, wind, nd, nr) in cells:
                if terrain != cur_terrain:
                    relay.a20.set_terrain_scenario(terrain)
                    cur_terrain = terrain
                    prov = provenance()
                    print(f"[provenance] {terrain}: TERRAIN_SOURCE={prov['terrain_source']} "
                          f"cell={prov['cell_size_m']} ENERGY_PER_CELL={prov['energy_per_cell']} "
                          f"commit={prov['git_commit'][:8]} dirty={prov['git_dirty']}", flush=True)
                    if args.mode != "free":
                        if prov["terrain_source"] != "real_srtm":
                            raise SystemExit(f"TERRAIN_SOURCE={prov['terrain_source']}; plan needs real_srtm. Stopped.")
                        if abs(prov["cell_size_m"] - PLAN_CELL_SIZE) > 0.05:
                            raise SystemExit(f"cell_size_m={prov['cell_size_m']}; plan needs {PLAN_CELL_SIZE}. Stopped.")
                todo = [s for s in seeds if (args.mode, terrain, wind, nd, nr, s) not in done]
                if not todo:
                    print(f"[skip] {terrain}/{wind}/{nd}x{nr}: all seeds already done")
                    continue
                t_cell = time.time()
                for k, seed in enumerate(todo, 1):
                    last = (terrain, wind, nd, nr, seed)
                    set_wind(wind, seed, args.max_steps)
                    row = {"mode": args.mode, "terrain": terrain, "wind": wind,
                           "num_drones": nd, "num_reserves": nr, "seed": seed}
                    row.update(prov)
                    for p in policies:
                        tr = trackers[p]
                        tr.reset()
                        m = relay.simulate_fleet(seed, relay_selector=tr, num_drones=nd,
                                                 num_reserves=nr, max_steps=args.max_steps)
                        for key in METRICS:
                            v = m.get(key)
                            row[f"{p}_{key}"] = int(v) if isinstance(v, bool) else v
                        row[f"{p}_reserve_share"] = tr.share()
                    w.writerow(row)
                    fh.flush()
                    if k % 10 == 0 or k == len(todo):
                        print(f"  {terrain}/{wind}/{nd}x{nr}: {k}/{len(todo)} seeds "
                              f"({(time.time() - t_cell) / 60:.1f} min)", flush=True)
                print(f"[done] {terrain}/{wind}/{nd}x{nr} ({(time.time() - t_cell) / 60:.1f} min)", flush=True)
    except BaseException as e:
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"crash_{args.mode}_{stamp}.txt"
        with open(name, "w", encoding="utf-8") as f:
            f.write(f"crash at {datetime.datetime.now().isoformat()}\n")
            f.write(f"last cell/seed started: {last}\n")
            f.write(f"commit: {prov0['git_commit']}\n\n")
            f.write(traceback.format_exc())
        print(f"\nCRASH. Log written to {name}. Archive the partial CSV under archive_crashed_runs/ "
              f"before any re-run. Fix only if no row of this cell has been viewed.")
        raise
    finally:
        try:
            relay.a20.set_wind_mode("static")
            relay.a20.set_terrain_scenario("benapole_delta")
        except Exception as e:
            print("could not restore default terrain/wind:", e)

    print(f"\nFinished in {(time.time() - t0) / 60:.1f} min. Output: {args.out}")
    if args.mode == "confirmatory":
        print("Do not open the CSV. Run analyze_generalization.py once, after all 100 seeds.")


if __name__ == "__main__":
    main()