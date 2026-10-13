# -*- coding: utf-8 -*-
"""
collect_battery_fields.py

Runs Policy J, Policy I (default weights) and the tuned Policy I from
tuned_I_lite.json on a block of seeds and saves EVERY per-seed field the
simulator returns (coverage, delay, energy, battery fields, stranded events,
relay success, ...) plus the reserve share, in one wide CSV with columns
"<policy>_<field>". Feed the file to analyze_battery_safety.py.

Usage (Git Bash / PowerShell, from the folder with the simulator):
    python -u collect_battery_fields.py --seeds 201-400 --wind static
    python -u collect_battery_fields.py --seeds 201-400 --wind stochastic
Options:
    --policies J,I,I_lite        any of: J, I, I_lite (tuned_I_lite.json), I2 (earlier tuned I),
                                 B (nearest feasible), BR (nearest idle reserve first), C (RelayScore),
                                 D (true FIS), D2, T, V (untuned, D's weights),
                                 T_tuned, V_tuned (tuned_T_lite.json / tuned_V_lite.json),
                                 AUC (lambda from tuned_AUC_lite.json), AUC400 (lambda = 400)
    --set confirm                the 13 policies of the confirmatory block:
                                 J,I,I_lite,C,D,D2,T,V,B,BR,AUC,T_tuned,V_tuned
    --out FILE                   default battery_<block>_<wind>.csv
    --max-steps 500

Why two winds: stranded_events can only be non-zero under stochastic wind, so it
is only informative there. Run the static condition for the headline setting and
the stochastic condition for stranded events. Estimated time: roughly 10 min per
policy per 200 seeds (about 2.5 min per 50 seeds, from earlier runs).

Seed block: if you use seeds 401-450 (never used before) the results are clean
of any earlier selection; seeds 201-400 are the "fresh" block already seen for
several analyses.
"""

import argparse
import csv
import json
import sys

import a2e_relay_fixed_final_51 as relay
from policy_j_full import select_best_relay_j

J_WEIGHTS = {
    "W_TRAVEL_ENERGY": 0.283556320123423,
    "W_COVERAGE_LOSS": 0.023813783478481596,
    "W_RELAY_DELAY": 0.28445380353708727,
    "W_BATTERY_MARGIN": 0.40817609286100814,
}
I2_WEIGHTS = {  # the earlier tuned I (267 evaluations), from the held-out tuning log
    "W_TRAVEL_ENERGY": 0.2386430204827954,
    "W_COVERAGE_LOSS": 0.13254292305288234,
    "W_RELAY_DELAY": 0.6288140564643223,
}

FIELDS = [
    "coverage_pct", "avg_zone_staleness_steps", "max_zone_staleness_steps",
    "detection_rate", "threats_detected", "relay_success_rate",
    "avg_relay_delay_steps", "avg_relay_energy_pct", "avg_relay_coverage_gap",
    "mission_completed", "avg_final_battery", "avg_patrol_battery",
    "avg_reserve_battery", "total_energy_consumed", "stranded_events",
    "relay_requested", "relay_fulfilled", "RC",
]


def select_nearest_reserve_first(needer, drones, g, detected, step, weights=None):
    """BR: nearest feasible idle reserve; if no reserve is feasible, nearest feasible patrol UAV.
    (Re-implemented here from the paper's definition; validate it once against the paper's
    BR fresh-block means 92.93 / 19.39 / 16.03 with --policies BR --seeds 201-400.
    Coverage 92.93 already matched; v2 fixes the delay/energy logging.)"""
    feasible = relay._feasible_candidate_pool(needer, drones)
    if not feasible:
        return None, []
    reserves = [x for x in feasible if x[0]["role"] == "idle_reserve"]
    pool = reserves if reserves else feasible
    best, best_d = None, float("inf")
    for cand, travel_energy, battery_margin in pool:
        d = relay._manhattan(cand["r"], cand["c"], needer["r"], needer["c"])
        if d < best_d:                      # strict: ties keep the first candidate in pool order
            best, best_d = cand, d
    # Diagnostics MUST come from the simulator's own relay_score(), exactly as B does it
    # (select_nearest_relay): the simulator reads relay delay and relay energy from this
    # record. A hand-made dict leaves them at 0 (that was the bug in the first version).
    _score, best_diag = relay.relay_score(
        best, needer, g, detected, step, weights=weights, fleet=drones
    )
    return best, ([best_diag] if best_diag is not None else [])


class Tracked:
    """Counts how often the chosen candidate is an idle reserve."""

    def __init__(self, sel):
        self.sel, self.n, self.res = sel, 0, 0

    def __call__(self, *a, **k):
        chosen, diag = self.sel(*a, **k)
        if chosen is not None:
            self.n += 1
            self.res += int(chosen.get("role") == "idle_reserve")
        return chosen, diag

    def reset(self):
        self.n = self.res = 0

    def share(self):
        return 100.0 * self.res / self.n if self.n else float("nan")


def parse_seeds(s):
    if "-" in s:
        a, b = s.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(",")]


def build_selectors(names):
    sel = {}
    for n in names:
        if n == "J":
            sel[n] = lambda nd, d, g, de, s: select_best_relay_j(nd, d, g, de, s, weights=J_WEIGHTS, theta=0.5)
        elif n == "I":
            sel[n] = relay.select_best_relay_coverage_aware
        elif n == "I_lite":
            with open("tuned_I_lite.json") as f:
                w = json.load(f)["weights"]
            sel[n] = (lambda ww: (lambda nd, d, g, de, s: relay.select_best_relay_coverage_aware(
                nd, d, g, de, s, weights=ww)))(w)
        elif n == "I2":
            sel[n] = (lambda ww: (lambda nd, d, g, de, s: relay.select_best_relay_coverage_aware(
                nd, d, g, de, s, weights=ww)))(I2_WEIGHTS)
        elif n == "B":
            sel[n] = relay.select_nearest_relay
        elif n == "BR":
            sel[n] = select_nearest_reserve_first
        elif n == "T_tuned":
            with open("tuned_T_lite.json") as f:
                w = json.load(f)["weights"]
            sel[n] = (lambda ww: (lambda nd, d, g, de, s: relay.select_topsis_relay(
                nd, d, g, de, s, weights=ww)))(w)
        elif n == "V_tuned":
            with open("tuned_V_lite.json") as f:
                w = json.load(f)["weights"]
            sel[n] = (lambda ww: (lambda nd, d, g, de, s: relay.select_vikor_relay(
                nd, d, g, de, s, weights=ww, v=0.5)))(w)
        elif n in ("AUC", "AUC400"):
            from auction_baseline import make_auction_selector
            if n == "AUC400":
                lam = 400.0
            else:
                with open("tuned_AUC_lite.json") as f:
                    lam = float(json.load(f)["lam"])
            sel[n] = make_auction_selector(lam)
        elif n in ("C", "D", "D2", "T", "V"):
            # the simulator's own registry: C = RelayScore, D = true FIS, D2 = surrogate FIS,
            # T/V = TOPSIS/VIKOR with D's (untuned) weights
            sel[n] = relay.POLICY_SELECTORS[n]
        else:
            raise SystemExit(f"unknown policy {n}")
    return sel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="201-400")
    ap.add_argument("--wind", default="static", choices=["static", "dynamic", "stochastic"])
    ap.add_argument("--policies", default="J,I,I_lite")
    ap.add_argument("--set", default="", choices=["", "confirm"])
    ap.add_argument("--out", default="")
    ap.add_argument("--max-steps", type=int, default=500)
    args = ap.parse_args()

    seeds = parse_seeds(args.seeds)
    if args.set == "confirm":
        args.policies = "J,I,I_lite,C,D,D2,T,V,B,BR,AUC,T_tuned,V_tuned"
    names = [p.strip() for p in args.policies.split(",") if p.strip()]
    selectors = {n: Tracked(s) for n, s in build_selectors(names).items()}
    out = args.out or f"battery_{seeds[0]}-{seeds[-1]}_{args.wind}.csv"

    rows = []
    try:
        for k, seed in enumerate(seeds, 1):
            if args.wind == "stochastic":
                relay.a20.set_wind_mode("stochastic", seed=seed, n_steps=args.max_steps + 1)
            else:
                relay.a20.set_wind_mode(args.wind)
            row = {"seed": seed}
            for n, tr in selectors.items():
                tr.reset()
                m = relay.simulate_fleet(seed, relay_selector=tr, max_steps=args.max_steps)
                for f in FIELDS:
                    v = m.get(f)
                    row[f"{n}_{f}"] = int(v) if isinstance(v, bool) else v
                row[f"{n}_reserve_share"] = tr.share()
            rows.append(row)
            if k % 25 == 0 or k == len(seeds):
                with open(out, "w", newline="") as fh:       # checkpoint
                    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
                    w.writeheader()
                    w.writerows(rows)
                print(f"  {k}/{len(seeds)} seeds done, saved -> {out}", flush=True)
    finally:
        relay.a20.set_wind_mode("static")   # restore the default wind mode

    print(f"\nDone. Now run, e.g.:\n  python analyze_battery_safety.py --wide {out} --a J --b I\n"
          f"  python analyze_battery_safety.py --wide {out} --a J --b I_lite")


if __name__ == "__main__":
    main()