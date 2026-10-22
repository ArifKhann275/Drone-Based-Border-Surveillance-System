# -*- coding: utf-8 -*-
"""
probe_pool_size.py   (EXPLORATORY DIAGNOSTIC, not part of the pre-specified hypotheses)

Question: in the hilly terrains all policies behave almost identically and ~90% of relays
are served by idle reserves. Is it because the feasible candidate pool is almost always
tiny or reserve-only, so that the ranking rule rarely matters?

For each terrain (static wind by default) this script runs LARS on a few seeds and, at
every relay decision, records
  * the size of the feasible pool (and the share of decisions with exactly 1 candidate),
  * whether the pool contains at least one PATROL candidate (not an idle reserve),
  * for pools with >= 2 candidates: whether RelayScore (C) would have chosen the same UAV
    as LARS (agreement rate),
  * the share of LARS's choices that are idle reserves.
The simulation itself follows LARS's choices (C is only queried, not executed).

Usage (about 1 minute per terrain per 5 seeds):
    python -u probe_pool_size.py --seeds 601-620
    python -u probe_pool_size.py --seeds 601-620 --terrains sylhet_hills,benapole_delta --wind stochastic

Use seeds that are ALREADY part of the sweep (601-700): this is a diagnostic of the same
runs, not a new test. Report it as exploratory.
"""

import argparse
from collections import Counter

import numpy as np

import a2e_relay_fixed_final_51 as relay


def parse_seeds(s):
    if "-" in s:
        a, b = s.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(",")]


def make_probe(lars_sel, c_sel, st):
    def sel(needer, drones, g, detected, step):
        feasible = relay._feasible_candidate_pool(needer, drones)
        n = len(feasible)
        st["decisions"] += 1
        st["hist"][min(n, 8)] += 1
        st["pool_sum"] += n
        n_patrol = sum(1 for c, _, _ in feasible if c["role"] != "idle_reserve")
        st["with_patrol"] += int(n_patrol > 0)
        chosen, diag = lars_sel(needer, drones, g, detected, step)
        if chosen is not None:
            st["chosen"] += 1
            st["chosen_reserve"] += int(chosen.get("role") == "idle_reserve")
        if n >= 2:
            other, _ = c_sel(needer, drones, g, detected, step)
            st["multi"] += 1
            if chosen is not None and other is not None and other["id"] == chosen["id"]:
                st["agree"] += 1
        return chosen, diag
    return sel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--terrains", default="benapole_delta,sylhet_hills,coxs_bazar,chittagong_hill_tracts")
    ap.add_argument("--wind", default="static", choices=["static", "dynamic", "stochastic"])
    ap.add_argument("--seeds", default="601-620")
    ap.add_argument("--max-steps", type=int, default=500)
    args = ap.parse_args()
    seeds = parse_seeds(args.seeds)
    lars_sel = relay.select_best_relay_coverage_aware          # LARS = Policy I with default weights
    c_sel = relay.POLICY_SELECTORS["C"]                         # RelayScore

    print(f"{'terrain':<26}{'decisions':>10}{'mean pool':>10}{'pool=1 %':>9}{'pool>=2 %':>10}"
          f"{'has patrol %':>13}{'LARS=C %*':>10}{'reserve %':>10}")
    try:
        for terrain in [t.strip() for t in args.terrains.split(",") if t.strip()]:
            relay.a20.set_terrain_scenario(terrain)
            st = dict(decisions=0, hist=Counter(), pool_sum=0, with_patrol=0, multi=0, agree=0,
                      chosen=0, chosen_reserve=0)
            probe = make_probe(lars_sel, c_sel, st)
            for s in seeds:
                if args.wind == "stochastic":
                    relay.a20.set_wind_mode("stochastic", seed=s, n_steps=args.max_steps + 1)
                else:
                    relay.a20.set_wind_mode(args.wind)
                relay.simulate_fleet(s, relay_selector=probe, max_steps=args.max_steps)
            d = max(st["decisions"], 1)
            print(f"{terrain:<26}{st['decisions']:>10}{st['pool_sum'] / d:>10.2f}"
                  f"{100 * st['hist'][1] / d:>9.1f}{100 * st['multi'] / d:>10.1f}"
                  f"{100 * st['with_patrol'] / d:>13.1f}"
                  f"{100 * st['agree'] / max(st['multi'], 1):>10.1f}"
                  f"{100 * st['chosen_reserve'] / max(st['chosen'], 1):>10.1f}", flush=True)
    finally:
        relay.a20.set_wind_mode("static")
        relay.a20.set_terrain_scenario("benapole_delta")
    print("\n* agreement among decisions whose feasible pool has at least 2 candidates.")
    print("Pool = 1 means the rule has no choice; 'has patrol' = at least one patrol UAV is feasible.")


if __name__ == "__main__":
    main()
