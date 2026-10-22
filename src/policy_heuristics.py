# -*- coding: utf-8 -*-
"""
policy_heuristics.py
══════════════════════════════════════════════════════════════
Simple, non-MADM relay-selection baselines for the "is all this
machinery actually needed?" question a reviewer will ask.

All selectors share EXACTLY the same feasible candidate pool as
Policies C / D / I / J (relay._feasible_candidate_pool: role-eligible
AND battery - travel_energy >= RELAY_SAFETY_FLOOR), so the only thing
that differs is the ranking rule.

Labels used by run_heuristics_vs_j.py
    B    nearest feasible candidate (already in the codebase; re-exported)
    BR   nearest feasible IDLE RESERVE; falls back to nearest feasible
         patrol drone only if no reserve is feasible
    HB   feasible candidate with the highest CURRENT battery
         (ties: nearer, then lower id)
    RND  uniformly random feasible candidate

Interface (same as every other selector in a2e_relay_fixed_final_51):
    selector(needer, drones, g, detected, step, weights=None)
        -> (chosen_candidate_or_None, [diag_of_chosen])

WHY EACH SELECTOR RETURNS relay_score()'s diag FOR THE CHOSEN CANDIDATE
-----------------------------------------------------------------------
collect_metrics() computes avg_relay_delay_steps, avg_relay_energy_pct
and avg_relay_coverage_gap from the *diagnostics of the chosen candidate*
(state["log"][i]["candidates_diag"]).  A selector that returns [] would
make those three metrics None.  select_nearest_relay does the same thing
(calls relay_score() only to build the diagnostics; the score is NOT used
for the choice).

CAVEAT FOR THE SELECTOR-TIME FIELDS
-----------------------------------
Because every heuristic still calls relay_score() once to build the
diagnostics, its measured selector time includes that call.  Do NOT quote
these timings as the cost of the heuristic itself; they are an upper bound.

RANDOM SELECTOR IS STATELESS (call-order safe)
----------------------------------------------
The project's own notes say results have been sensitive to call order
for reasons not fully isolated.  So RND does NOT touch the global `random`
state or any module-level generator.  It builds a private
random.Random from a deterministic string key (needer id, step, needer
position/battery, sorted candidate ids, optional salt).  String seeding
in CPython uses SHA-512, so it is identical across processes and
unaffected by PYTHONHASHSEED.  Change RANDOM_SALT to draw a different
(but still reproducible) random policy.
"""

import random

import a2e_relay_fixed_final_51 as relay

RANDOM_SALT = 0  # set from run_heuristics_vs_j.py (--random-salt)


def _diag_for(best, needer, drones, g, detected, step, weights):
    """Diagnostics for the chosen candidate, built exactly the way
    relay.select_nearest_relay builds them (score itself is not used)."""
    _score, best_diag = relay.relay_score(
        best, needer, g, detected, step, weights=weights, fleet=drones
    )
    return [best_diag] if best_diag is not None else []


def _dist(cand, needer):
    return relay._manhattan(cand["r"], cand["c"], needer["r"], needer["c"])


# ── B: nearest feasible (re-export of the existing baseline) ─────────────
select_nearest_feasible = relay.select_nearest_relay


# ── BR: nearest feasible idle reserve, patrol drone only as fallback ─────
def select_nearest_reserve(needer, drones, g, detected, step, weights=None):
    feasible = relay._feasible_candidate_pool(needer, drones)
    if not feasible:
        return None, []
    reserves = [x for x in feasible if x[0].get("role") == "idle_reserve"]
    pool = reserves if reserves else feasible
    best = min(pool, key=lambda x: (_dist(x[0], needer), x[0]["id"]))[0]
    return best, _diag_for(best, needer, drones, g, detected, step, weights)


# ── HB: highest current battery ──────────────────────────────────────────
def select_highest_battery(needer, drones, g, detected, step, weights=None):
    feasible = relay._feasible_candidate_pool(needer, drones)
    if not feasible:
        return None, []
    best = min(
        feasible,
        key=lambda x: (-x[0]["battery"], _dist(x[0], needer), x[0]["id"]),
    )[0]
    return best, _diag_for(best, needer, drones, g, detected, step, weights)


# ── RND: uniformly random feasible candidate (stateless) ─────────────────
def select_random_feasible(needer, drones, g, detected, step, weights=None):
    feasible = relay._feasible_candidate_pool(needer, drones)
    if not feasible:
        return None, []
    feasible = sorted(feasible, key=lambda x: x[0]["id"])
    ids = [x[0]["id"] for x in feasible]
    key = (f"rnd|salt={RANDOM_SALT}|needer={needer['id']}|step={step}"
           f"|pos={needer['r']},{needer['c']}|bat={round(needer['battery'], 3)}"
           f"|cands={ids}")
    rng = random.Random(key)          # private generator; global state untouched
    best = feasible[rng.randrange(len(feasible))][0]
    return best, _diag_for(best, needer, drones, g, detected, step, weights)


HEURISTIC_SELECTORS = {
    "B":   select_nearest_feasible,
    "BR":  select_nearest_reserve,
    "HB":  select_highest_battery,
    "RND": select_random_feasible,
}

HEURISTIC_DISPLAY_NAMES = {
    "B":   "Nearest feasible",
    "BR":  "Nearest idle reserve",
    "HB":  "Highest battery",
    "RND": "Random feasible",
}
