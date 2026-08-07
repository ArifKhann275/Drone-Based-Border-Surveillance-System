# -*- coding: utf-8 -*-
"""
fleet_relay_optimization.py
════════════════════════════════════════════════════════════════════════
PATH B — Cooperative Multi-Drone Fleet with Optimization-Based Relay
Selection (extends a20.py from nearest-station handoff to a fixed-size
N-drone relay-selection problem).

CONCEPTUAL FIX IN THIS VERSION
------------------------------
Battery is treated as a HARD feasibility constraint:
    candidate_battery - travel_energy >= RELAY_SAFETY_FLOOR

The RelayScore does NOT add battery margin as a soft benefit, because that
would double-count travel energy (battery margin already subtracts the same
travel energy). Once a candidate is physically feasible, the utility is
based on:
    1) physics-based travel energy,
    2) marginal surveillance staleness caused by removing the candidate, and
    3) relay arrival delay.

Both the nearest baseline and the proposed RelayScore use the EXACT SAME
feasible candidate pool, so B-vs-C isolates the ranking/decision rule rather
than candidate availability.
════════════════════════════════════════════════════════════════════════
"""

import math
import random
import statistics
import csv
import json
from copy import deepcopy
from functools import partial

# Plotting libraries for decision breakdown
try:
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")  # headless-safe: only ever savefig(), never show()
    import matplotlib.pyplot as plt
except ImportError:
    print("Warning: pandas or matplotlib not installed. Plotting will be skipped.")

import a20  # noqa: reuse the existing physics/grid/zone engine as-is

from scipy.stats import wilcoxon, ttest_rel


# ══════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════
NUM_DRONES = 4                  # simultaneously-patrolling drones (fixed, matches a20.STATIONS)
RESERVE_POOL_SIZE = 4           # extra idle spare drones (one pre-positioned per station)
RELAY_SAFETY_FLOOR = 8.0       # % battery a candidate must keep in reserve after the trip
COVERAGE_STALENESS_CAP_STEPS = 200   # same normalization a20.zone_info() uses (tgap)
THREAT_ZONE_PENALTY = 2.0      # penalty multiplier if a threat zone is left unattended
PRIORITY_ZONE_WEIGHT = 1.0     # weight for border priority

# Utility-function weights (Moved to a dictionary for Sensitivity Analysis)
DEFAULT_WEIGHTS = {
    # All three terms are costs to the mission. Battery is NOT a soft utility
    # benefit anymore: feasibility is enforced as a hard constraint in
    # _feasibility(), which prevents travel energy from being double-counted
    # through both "battery margin" and "energy cost".
    "W_TRAVEL_ENERGY": 1.0,      # travel energy from a20 OWT-A* model, in battery-% equivalent
    "W_COVERAGE_LOSS": 25.0,     # marginal surveillance-staleness cost, normalized to [0,1]
    "W_RELAY_DELAY": 0.4,        # small delay penalty, measured in simulation steps
}

# ── Policy D: fuzzy multi-attribute relay selection ─────────────────────
# Attribute weights for select_fuzzy_relay(). These are NOT the same kind
# of object as DEFAULT_WEIGHTS (raw cost multipliers) -- they are relative
# IMPORTANCE weights (should sum to ~1.0) applied AFTER each attribute has
# already been fuzzy-normalized onto a common [0,1] membership scale, per
# the fuzzy-optimum-selection method the paper cites. See select_fuzzy_relay()
# docstring for the fidelity caveat versus the original paper's own 4
# UAV-alternative attributes (which require a continuous-space RF/
# k-connectivity model this codebase does not implement).
FUZZY_ATTR_WEIGHTS = {
    "distance": 0.30,        # analogue of the paper's link/hop-distance attribute
    "battery_margin": 0.30,  # analogue of the paper's energy/endurance attribute
    "coverage_loss": 0.25,   # analogue of the paper's mission-payoff attribute
    "relay_delay": 0.15,     # analogue of the paper's latency attribute
}

# ── Policy C: RelayScore weights, APPLIED TO POOL-NORMALIZED ATTRIBUTES ──
# BUG FIX (Aug 2026): DEFAULT_WEIGHTS above multiplies RAW, physically-unit-ed
# costs together (travel_energy in battery-%, coverage_loss in [0, ~0.03],
# relay_delay in steps). Empirically, travel_energy sits in the 5-15 range
# while 25*coverage_loss only reaches ~0.1-0.75 -- so coverage_loss ended up
# contributing under ~5% of total_cost even with W_COVERAGE_LOSS=25, and
# RelayScore's ranking collapsed to "cheapest energy + shortest delay",
# almost completely ignoring the coverage term its own docstring claims to
# optimize. This was confirmed on real run data: RelayScore's chosen
# candidates had ~6x HIGHER avg_relay_coverage_gap than fuzzy MADM's, despite
# fuzzy MADM giving coverage only 25% importance vs RelayScore's intended
# dominant weighting.
#
# FIX: normalize travel_energy / coverage_loss / relay_delay onto a common
# [0,1] "how good is this candidate on this attribute, relative to the other
# feasible candidates right now" scale -- the SAME normalization step
# select_fuzzy_relay() (Policy D) already uses via _fuzzy_normalize() --
# BEFORE applying importance weights. See _relay_utility_pool(). These
# weights now mean RELATIVE IMPORTANCE (should sum to ~1.0), exactly like
# FUZZY_ATTR_WEIGHTS, NOT raw cost multipliers.
NORMALIZED_WEIGHTS = {
    "W_TRAVEL_ENERGY": 0.35,
    "W_COVERAGE_LOSS": 0.40,
    "W_RELAY_DELAY": 0.25,
}


# ══════════════════════════════════════════════════════════════
# FLEET INITIALIZATION
# ══════════════════════════════════════════════════════════════
def _new_drone(idx, seed):
    r, c = a20.STATIONS[idx % len(a20.STATIONS)]
    start_r, start_c = (1, 1) if (r, c) == (0, 0) else \
                        (1, a20.COLS - 2) if (r, c) == (0, a20.COLS - 1) else \
                        (a20.ROWS - 2, 1) if (r, c) == (a20.ROWS - 1, 0) else \
                        (a20.ROWS - 2, a20.COLS - 2)
    return {
        "id": idx,
        "home_station": (r, c),
        "r": start_r, "c": start_c, "pr": None, "pc": None,
        "battery": 100.0,
        "cycles": 0,                # completed full-recharge count, feeds SOH aging model
                                     # (a20.USE_SOH_AGING) -- per-DRONE, not fleet-wide, so a
                                     # drone that has cycled through many relief trips ages
                                     # independently of its fleet-mates
        "target_zone": None, "target_cell": None,
        "zone_rng": random.Random(seed * 97 + idx * 31 + 17),
        "role": "patrol",          # patrol | relay_incoming | returning
        "rendezvous_id": None,     # which drone id this one is relaying toward (if relay_incoming)
        "awaiting_relay": False,   # True once this drone has flagged low battery and a relay is inbound
        "return_target": None,    # station coords this drone heads to when role == returning
        "relay_episode_counted": False,  # True once this low-battery episode has already
                                          # incremented relay_requested (see step_fleet());
                                          # reset when the episode is resolved so the NEXT
                                          # low-battery episode can be counted separately.
    }


def init_fleet(seed, num_drones=NUM_DRONES, num_reserves=RESERVE_POOL_SIZE):
    g = a20.make_grid(seed)
    a20.place_threats(seed, [g])
    drones = [_new_drone(i, seed) for i in range(num_drones)]
    for d in drones:
        d["role"] = "patrol"

    for j in range(num_reserves):
        idx = num_drones + j
        st = a20.STATIONS[j % len(a20.STATIONS)]
        rd = _new_drone(idx, seed)
        rd["r"], rd["c"] = st       
        rd["role"] = "idle_reserve"
        rd["home_station"] = st
        drones.append(rd)

    return {
        "g": g,
        "detected": set(),
        "drones": drones,
        "step": 0,
        "RC": 0,          
        "target_patrol_count": num_drones,   
        "log": [],
        "first_all_threats_step": None,   # step at which len(detected) first reached NUM_THREATS
        "first_full_coverage_step": None, # step at which coverage_pct(g) first reached 100
        "relay_requested": 0,   # incremented ONCE per low-battery episode (see step_fleet())
        "relay_fulfilled": 0,   # incremented every time a feasible candidate was actually found
        "total_energy_consumed": 0.0,   # running sum of EVERY step-cost across ALL drones,
                                         # NEVER reset by recharge -- a true total-work metric,
                                         # mirrors a20.DroneSimHeadless's own self.s_energy pattern.
    }


# ══════════════════════════════════════════════════════════════
# RELAYSCORE — the 5-factor utility function
# ══════════════════════════════════════════════════════════════
def _manhattan(r1, c1, r2, c2):
    return abs(r1 - r2) + abs(c1 - c2)


def _safe_wilcoxon(x, y):
    """Paired Wilcoxon helper that avoids scipy warnings on all-zero differences
    and returns a stable, JSON-friendly result.

    Catches broadly (not just ValueError): environments with a mismatched
    numpy/scipy pair (e.g. numpy>=2.0 with an old scipy build) can raise
    AttributeError/TypeError deep inside scipy's wilcoxon() rather than a
    clean ValueError. A single metric failing to get a p-value should not
    crash a 30-seed thesis run -- it should just report wilcoxon_p=None
    for that metric and let everything else continue. If you see
    "wilcoxon_p": None everywhere, run `pip install --upgrade scipy numpy`
    to fix the real underlying cause rather than relying on this fallback.

    Booleans (e.g. mission_completed: True/False) are coerced to 0/1
    before any subtraction/stats -- numpy refuses to subtract two boolean
    arrays directly ("use bitwise_xor instead"), which would otherwise
    crash BOTH wilcoxon() and the ttest_rel() fallback on this metric.
    """
    pairs = [(int(a) if isinstance(a, bool) else a, int(b) if isinstance(b, bool) else b)
             for a, b in zip(x, y)
             if isinstance(a, (int, float, bool)) and isinstance(b, (int, float, bool))]
    if len(pairs) < 2:
        return {"mean_diff": None, "wilcoxon_p": None}
    diffs = [b - a for a, b in pairs]
    mean_diff = statistics.mean(diffs)
    if all(abs(d) < 1e-12 for d in diffs):
        return {"mean_diff": mean_diff, "wilcoxon_p": 1.0}
    try:
        _stat, p = wilcoxon([a for a, _ in pairs], [b for _, b in pairs])
        return {"mean_diff": mean_diff, "wilcoxon_p": round(float(p), 5)}
    except Exception as e:
        # Common cause: numpy>=2.0 paired with a scipy build too old to
        # support it -- wilcoxon()'s internal error-handling path itself
        # crashes (AttributeError on np.AxisError) before it can even
        # return a real statistical result. ttest_rel() runs through a
        # different scipy code path and is usually unaffected, so fall
        # back to a paired t-test rather than silently losing the
        # significance test entirely. Flagged via "test_used" so it's
        # clear in the output which test actually produced the p-value.
        try:
            _stat2, p2 = ttest_rel([a for a, _ in pairs], [b for _, b in pairs])
            return {"mean_diff": mean_diff, "wilcoxon_p": round(float(p2), 5),
                     "test_used": "ttest_rel_fallback",
                     "wilcoxon_error": f"{type(e).__name__}: {e}"}
        except Exception as e2:
            return {"mean_diff": mean_diff, "wilcoxon_p": None,
                     "wilcoxon_error": f"{type(e).__name__}: {e}",
                     "ttest_error": f"{type(e2).__name__}: {e2}"}


def _feasibility(candidate, needer):
    """Weight-INDEPENDENT feasibility check, shared by both B (nearest) and
    C (RelayScore) so a candidate that can't physically make the trip is
    excluded from BOTH policies' candidate pools identically. Returns
    (travel_energy, battery_margin, is_feasible)."""
    travel_energy = a20.energy_to_travel(candidate["r"], candidate["c"],
                                          needer["r"], needer["c"])
    battery_margin = candidate["battery"] - travel_energy - RELAY_SAFETY_FLOOR
    return travel_energy, battery_margin, battery_margin >= 0


def _candidate_pool(needer, drones):
    """
    Return all role-eligible relay candidates.

    Eligible:
      - active patrol drones
      - idle reserve drones

    Excluded:
      - the drone requesting the relay
      - drones already assigned to another relay
      - drones already returning to a station
      - malformed entries that do not contain required fleet fields
    """
    if needer is None or drones is None:
        return []

    pool = []

    for d in drones:
        if not isinstance(d, dict):
            continue

        if "id" not in d or "role" not in d:
            continue

        # A drone cannot relay for itself.
        if d["id"] == needer.get("id"):
            continue

        # Only patrol or idle-reserve drones can be dispatched.
        if d["role"] not in ("patrol", "idle_reserve"):
            continue

        # Defensive guard: never reuse a drone already committed to another relay.
        if d.get("awaiting_relay", False):
            continue

        pool.append(d)

    return pool


def _feasible_candidate_pool(needer, drones):
    """
    Shared feasible candidate pool for both B and C.

    B = nearest feasible candidate.
    C = highest RelayScore among the exact same feasible candidates.

    This isolates the relay-selection/ranking rule in the B-vs-C comparison.
    """
    if needer is None or drones is None:
        return []

    pool = []

    for d in _candidate_pool(needer, drones):
        try:
            travel_energy, battery_margin, feasible = _feasibility(d, needer)
        except (KeyError, TypeError, ValueError):
            # Ignore malformed candidate state instead of crashing the whole run.
            continue

        if feasible:
            pool.append((d, travel_energy, battery_margin))

    return pool


def _coverage_loss_for_candidate(candidate, needer, g, detected, step, relay_delay):
    """Estimate the candidate-induced marginal surveillance-staleness cost.

    This is intentionally a conservative lower-bound estimate:
      projected_staleness - current_staleness
    over the relay-arrival interval only.

    It does NOT assume what happens after the relay arrives, nor does it
    simulate the counterfactual patrol trajectory of the candidate. Those
    assumptions are documented as a modeling limitation in the thesis.
    """
    if candidate["role"] == "idle_reserve":
        return 0.0, 0.0, 0.0, None

    if candidate["target_zone"] is None:
        return 0.0, 0.0, 0.0, None

    info = a20.zone_info(
        candidate["target_zone"][0],
        candidate["target_zone"][1],
        0, 0, g, detected, step
    )
    if not info:
        return 0.0, 0.0, 0.0, None

    cap = max(float(COVERAGE_STALENESS_CAP_STEPS), 1.0)
    current_tgap = min(info["raw_gap"] / cap, 1.0)
    projected_tgap = min((info["raw_gap"] + relay_delay) / cap, 1.0)
    marginal_loss = max(0.0, projected_tgap - current_tgap)

    # Threat / border priority can increase the mission impact of leaving a
    # strategically important zone unattended. These fields already exist in
    # a20.zone_info(), so no extra "ground truth" state is invented here.
    has_threat = bool(info.get("has_threat", False))
    border_pr = float(info.get("border_pr", 0.0))
    penalty_mult = 1.0
    if has_threat:
        penalty_mult += THREAT_ZONE_PENALTY
    penalty_mult += PRIORITY_ZONE_WEIGHT * border_pr

    weighted_loss = marginal_loss * penalty_mult
    return weighted_loss, marginal_loss, current_tgap, {
        "has_threat": has_threat,
        "border_priority": border_pr,
        "penalty_multiplier": round(penalty_mult, 3),
        "current_tgap": round(current_tgap, 3),
        "projected_tgap": round(projected_tgap, 3),
    }


def relay_score(candidate, needer, g, detected, step, weights=None):
    """Compute mission utility for one feasible relay candidate.

    CONCEPTUAL FIX:
      Battery is a HARD FEASIBILITY CONSTRAINT:
          battery - travel_energy >= RELAY_SAFETY_FLOOR

      Once feasible, battery surplus is NOT added as a soft benefit because
      that would partially count travel energy twice through:
          (battery - travel_energy) + travel_energy_cost.

    Therefore the utility is:
        RS_i = -(w_E * E_i + w_C * L_i + w_D * D_i)

    where E_i is physics-based travel energy, L_i is marginal surveillance
    staleness cost, and D_i is relay arrival delay.

    Larger score is better (less negative cost).
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS

    dist = _manhattan(candidate["r"], candidate["c"], needer["r"], needer["c"])
    travel_energy, battery_margin, feasible = _feasibility(candidate, needer)

    if not feasible:
        return None, {
            "candidate_id": candidate["id"],
            "reason": "insufficient_battery",
            "battery_margin": round(battery_margin, 2),
            "score": None,
        }

    relay_delay = dist

    coverage_loss, raw_marginal_loss, current_tgap, cov_meta = _coverage_loss_for_candidate(
        candidate, needer, g, detected, step, relay_delay
    )

    energy_cost = weights["W_TRAVEL_ENERGY"] * travel_energy
    coverage_cost = weights["W_COVERAGE_LOSS"] * coverage_loss
    delay_cost = weights["W_RELAY_DELAY"] * relay_delay
    total_cost = energy_cost + coverage_cost + delay_cost

    # Higher is better. Zero is the theoretical cost of a perfect
    # zero-energy/zero-delay/zero-impact relay, which is unattainable.
    score = -total_cost

    diag = {
        "candidate_id": candidate["id"],
        "role": candidate["role"],
        "distance": dist,
        "travel_energy": round(travel_energy, 3),
        "battery_margin": round(battery_margin, 2),  # feasibility/headroom only
        "battery_feasible": True,
        "coverage_loss": round(coverage_loss, 6),
        "raw_marginal_coverage_loss": round(raw_marginal_loss, 6),
        "current_tgap": round(current_tgap, 6),
        "relay_delay": relay_delay,
        "energy_cost": round(energy_cost, 3),
        "coverage_cost": round(coverage_cost, 3),
        "delay_cost": round(delay_cost, 3),
        "score": round(score, 3),
        "coverage_metadata": cov_meta,
    }
    return score, diag


def _relay_utility_pool(feasible, needer, g, detected, step, weights=None):
    """Rank ALL feasible candidates together using POOL-NORMALIZED attributes,
    so no single term's raw physical scale (e.g. travel_energy's battery-%
    units vs coverage_loss's [0,1] fraction) can silently dominate the
    ranking regardless of the configured weight -- see NORMALIZED_WEIGHTS
    docstring above for the bug this fixes.

    Mirrors select_fuzzy_relay()'s normalization step (_fuzzy_normalize),
    but keeps RelayScore's own weighted-SUM utility form (as opposed to
    fuzzy's weighted distance-to-ideal-point), so Policy C and Policy D
    remain a fair, like-for-like comparison: same candidate pool, same
    normalized attributes, different combination rule.

    Returns a list of (candidate, utility, diag) sorted best-first.
    Higher utility is better (1.0 = candidate is best-on-every-attribute).
    """
    if weights is None:
        weights = NORMALIZED_WEIGHTS

    rows = []
    for cand, travel_energy, battery_margin in feasible:
        dist = _manhattan(cand["r"], cand["c"], needer["r"], needer["c"])
        relay_delay = dist
        coverage_loss, raw_marginal_loss, current_tgap, cov_meta = _coverage_loss_for_candidate(
            cand, needer, g, detected, step, relay_delay
        )
        rows.append({
            "cand": cand, "distance": dist, "travel_energy": travel_energy,
            "battery_margin": battery_margin, "coverage_loss": coverage_loss,
            "raw_marginal_loss": raw_marginal_loss, "current_tgap": current_tgap,
            "cov_meta": cov_meta, "relay_delay": relay_delay,
        })

    if len(rows) == 1:
        r = rows[0]
        diag = {
            "candidate_id": r["cand"]["id"], "role": r["cand"]["role"],
            "distance": r["distance"],
            "travel_energy": round(r["travel_energy"], 3),
            "battery_margin": round(r["battery_margin"], 2),
            "coverage_loss": round(r["coverage_loss"], 6),
            "raw_marginal_coverage_loss": round(r["raw_marginal_loss"], 6),
            "current_tgap": round(r["current_tgap"], 6),
            "relay_delay": r["relay_delay"],
            "membership_energy": 1.0, "membership_coverage": 1.0, "membership_delay": 1.0,
            "score": 1.0,
            "note": "only feasible candidate -- no ranking needed",
            "coverage_metadata": r["cov_meta"],
        }
        return [(r["cand"], 1.0, diag)]

    # Lower raw value = better on every one of these three attributes.
    energy_r = _fuzzy_normalize([r["travel_energy"] for r in rows], higher_is_better=False)
    cov_r = _fuzzy_normalize([r["coverage_loss"] for r in rows], higher_is_better=False)
    delay_r = _fuzzy_normalize([r["relay_delay"] for r in rows], higher_is_better=False)

    w = weights
    scored = []
    for i, r in enumerate(rows):
        # Each membership term is already "1.0 = best candidate on this
        # attribute", so a plain weighted SUM is the utility -- higher is
        # better, same "bigger score wins" convention the old relay_score()
        # used, just no longer vulnerable to raw-unit scale mismatch.
        utility = (w["W_TRAVEL_ENERGY"] * energy_r[i]
                   + w["W_COVERAGE_LOSS"] * cov_r[i]
                   + w["W_RELAY_DELAY"] * delay_r[i])

        diag = {
            "candidate_id": r["cand"]["id"], "role": r["cand"]["role"],
            "distance": r["distance"],
            "travel_energy": round(r["travel_energy"], 3),
            "battery_margin": round(r["battery_margin"], 2),
            "coverage_loss": round(r["coverage_loss"], 6),
            "raw_marginal_coverage_loss": round(r["raw_marginal_loss"], 6),
            "current_tgap": round(r["current_tgap"], 6),
            "relay_delay": r["relay_delay"],
            "membership_energy": round(energy_r[i], 3),
            "membership_coverage": round(cov_r[i], 3),
            "membership_delay": round(delay_r[i], 3),
            "score": round(utility, 4),
            "coverage_metadata": r["cov_meta"],
        }
        scored.append((r["cand"], utility, diag))

    scored.sort(key=lambda x: -x[1])
    return scored


def select_best_relay(needer, drones, g, detected, step, weights=None):
    """Proposed policy: highest mission-utility candidate from the SAME
    feasible candidate pool used by the nearest baseline.

    FIXED (Aug 2026): ranking now uses POOL-NORMALIZED attributes via
    _relay_utility_pool() instead of raw physical-unit costs, so
    travel_energy's larger raw magnitude can no longer silently swamp
    coverage_loss's contribution to the ranking. Pass weights=DEFAULT_WEIGHTS
    (or any dict with the old raw-cost semantics) only if you specifically
    want the old, scale-mismatched behavior for a regression comparison --
    normal use should leave weights=None (-> NORMALIZED_WEIGHTS).
    """
    feasible = _feasible_candidate_pool(needer, drones)
    if not feasible:
        return None, []

    scored = _relay_utility_pool(feasible, needer, g, detected, step, weights)
    best_cand, _best_score, _best_diag = scored[0]
    return best_cand, [diag for _, _, diag in scored]


# ══════════════════════════════════════════════════════════════
# POLICY E — DYNAMIC-WEIGHT RELAYSCORE (context-aware refinement on top
# of the pool-normalization fix, NOT a replacement for it)
# ══════════════════════════════════════════════════════════════
# Tunable pressure thresholds. Kept as module-level constants (like
# RELAY_SAFETY_FLOOR etc. above) so they show up in one obvious place for
# sensitivity analysis, rather than being buried as magic numbers.
COVERAGE_PRESSURE_THRESHOLD = 0.5   # mean zone staleness / cap, above which
                                      # W_COVERAGE_LOSS starts getting boosted
COVERAGE_PRESSURE_MAX_BOOST = 2.0    # multiplier applied at pressure == 1.0
BATTERY_CRITICAL_MARGIN = 5.0        # % headroom above RELAY_SAFETY_FLOOR,
                                      # below which W_TRAVEL_ENERGY starts
                                      # getting boosted
BATTERY_CRITICAL_MAX_BOOST = 2.5     # multiplier applied at zero headroom


def _dynamic_weights(needer, g, step, base_weights=None,
                      coverage_pressure_threshold=COVERAGE_PRESSURE_THRESHOLD,
                      coverage_pressure_max_boost=COVERAGE_PRESSURE_MAX_BOOST,
                      battery_critical_margin=BATTERY_CRITICAL_MARGIN,
                      battery_critical_max_boost=BATTERY_CRITICAL_MAX_BOOST):
    """Compute a FRESH set of normalized-importance weights for one decision,
    based on current mission pressure:

      - COVERAGE PRESSURE: if the whole map's mean zone staleness (relative
        to the same COVERAGE_STALENESS_CAP_STEPS cap _coverage_loss_for_
        candidate() uses) is already high, boost W_COVERAGE_LOSS -- the
        fleet is falling behind on surveillance overall, so this decision
        should weigh coverage more heavily than usual.

      - BATTERY PRESSURE: if the NEEDER's own battery headroom above
        RELAY_SAFETY_FLOOR is small, boost W_TRAVEL_ENERGY -- a
        battery-critical needer can't afford an energy-expensive relay
        choice, regardless of what that choice would do for coverage.

    Both boosts are linear ramps from 1.0x (at the threshold) up to their
    max at the extreme (pressure=1.0 / headroom=0), so the effect is
    continuous rather than a hard on/off switch. Weights are renormalized
    to sum to 1.0 afterward, so they stay directly comparable in scale to
    NORMALIZED_WEIGHTS and to each other.

    NOTE: this is deliberately a SEPARATE function from _relay_utility_pool's
    normalization step. Attribute normalization (fixing the scale-mismatch
    bug) and importance re-weighting (this function, an optional refinement)
    are two different concerns -- keeping them separate means either one can
    be turned off/ablated independently when writing up the comparison.
    """
    w = dict(base_weights or NORMALIZED_WEIGHTS)

    # ── Coverage pressure: mean zone staleness across the WHOLE map, not
    # just this one candidate's marginal effect -- a global signal that
    # _relay_utility_pool()'s per-decision normalization has no visibility
    # into, since that only compares candidates against each other.
    staleness_values = [step - c["last_visited_step"] for c in g.values()
                         if not c.get("is_station")]
    if staleness_values:
        cap = max(float(COVERAGE_STALENESS_CAP_STEPS), 1.0)
        coverage_pressure = min(statistics.mean(staleness_values) / cap, 1.0)
    else:
        coverage_pressure = 0.0

    if coverage_pressure > coverage_pressure_threshold:
        span = max(1.0 - coverage_pressure_threshold, 1e-9)
        ramp = (coverage_pressure - coverage_pressure_threshold) / span
        boost = 1.0 + (coverage_pressure_max_boost - 1.0) * ramp
        w["W_COVERAGE_LOSS"] *= boost

    # ── Battery pressure: how close THIS needer is to the hard feasibility
    # floor right now, independent of which candidate ends up chosen.
    battery_headroom = needer.get("battery", 100.0) - RELAY_SAFETY_FLOOR
    if battery_headroom < battery_critical_margin:
        deficit = max(0.0, battery_critical_margin - battery_headroom)
        ramp = min(deficit / battery_critical_margin, 1.0)
        boost = 1.0 + (battery_critical_max_boost - 1.0) * ramp
        w["W_TRAVEL_ENERGY"] *= boost

    total = sum(w.values())
    if total > 0:
        w = {k: v / total for k, v in w.items()}
    return w


def select_best_relay_dynamic(needer, drones, g, detected, step, weights=None,
                               coverage_pressure_threshold=COVERAGE_PRESSURE_THRESHOLD,
                               coverage_pressure_max_boost=COVERAGE_PRESSURE_MAX_BOOST,
                               battery_critical_margin=BATTERY_CRITICAL_MARGIN,
                               battery_critical_max_boost=BATTERY_CRITICAL_MAX_BOOST):
    """Policy E: RelayScore with dynamic (state-dependent) importance
    weights, computed fresh for every decision via _dynamic_weights().

    This is Policy C PLUS one more layer -- it still uses the SAME feasible
    candidate pool and the SAME pool-normalization step (_relay_utility_pool)
    that fixed the original scale-mismatch bug. Only the importance weights
    handed to that normalization step change, based on current coverage
    pressure and the needer's battery state. Compare against Policy C
    (static-normalized weights) to isolate the marginal value of the
    dynamic layer specifically.
    """
    feasible = _feasible_candidate_pool(needer, drones)
    if not feasible:
        return None, []

    dyn_weights = _dynamic_weights(
        needer, g, step, base_weights=weights,
        coverage_pressure_threshold=coverage_pressure_threshold,
        coverage_pressure_max_boost=coverage_pressure_max_boost,
        battery_critical_margin=battery_critical_margin,
        battery_critical_max_boost=battery_critical_max_boost,
    )
    scored = _relay_utility_pool(feasible, needer, g, detected, step, dyn_weights)
    best_cand, _best_score, _best_diag = scored[0]
    diags = [diag for _, _, diag in scored]
    for diag in diags:
        diag["dynamic_weights_used"] = {k: round(v, 3) for k, v in dyn_weights.items()}
    return best_cand, diags


# ══════════════════════════════════════════════════════════════
# BASELINE (generalized nearest-only, for fair ablation comparison)
# ══════════════════════════════════════════════════════════════
def select_nearest_relay(needer, drones, g, detected, step, weights=None):
    """Nearest-feasible baseline.

    B and C use EXACTLY the same feasible candidate pool. B differs only in
    the ranking rule: minimum Manhattan distance.
    """
    feasible = _feasible_candidate_pool(needer, drones)
    if not feasible:
        return None, []

    best = min(
        feasible,
        key=lambda x: _manhattan(
            x[0]["r"], x[0]["c"], needer["r"], needer["c"]
        )
    )[0]

    # Build diagnostics using the same component definitions as C, but do
    # NOT use the RelayScore to select the baseline.
    _score, best_diag = relay_score(
        best, needer, g, detected, step, weights=weights
    )
    return best, ([best_diag] if best_diag is not None else [])


# ══════════════════════════════════════════════════════════════
# POLICY D — FUZZY MULTI-ATTRIBUTE RELAY SELECTION
# (paper-comparison baseline, adapted from Zhu, Zhou & Zhang, 2017,
#  Appl. Sci. 7(1):8, "Connectivity Maintenance Based on Multiple Relay
#  UAVs Selection Scheme in Cooperative Surveillance")
# ══════════════════════════════════════════════════════════════
def _fuzzy_normalize(values, higher_is_better):
    """Relative membership degree r_i in [0,1] for one attribute across a
    candidate set: 1.0 = best candidate on this attribute, 0.0 = worst.
    This is the standard fuzzy-optimum-selection normalization step (each
    attribute is rescaled onto a common membership scale before the
    weighted distance-to-ideal-solution is computed). If every candidate
    ties on this attribute, everyone gets full membership (no information
    to discriminate on, so it should not penalize anyone)."""
    lo, hi = min(values), max(values)
    if hi - lo < 1e-12:
        return [1.0] * len(values)
    if higher_is_better:
        return [(v - lo) / (hi - lo) for v in values]
    return [(hi - v) / (hi - lo) for v in values]


def select_fuzzy_relay(needer, drones, g, detected, step, weights=None, p=2):
    """Policy D: fuzzy multi-attribute-decision-making relay selection."""
    if weights is None:
        weights = FUZZY_ATTR_WEIGHTS

    feasible = _feasible_candidate_pool(needer, drones)
    if not feasible:
        return None, []

    rows = []
    # FIX: Rename _travel_energy to travel_energy and capture it
    for cand, travel_energy, battery_margin in feasible:
        dist = _manhattan(cand["r"], cand["c"], needer["r"], needer["c"])
        relay_delay = dist
        coverage_loss, _raw, _tgap, _meta = _coverage_loss_for_candidate(
            cand, needer, g, detected, step, relay_delay
        )
        rows.append({
            "cand": cand,
            "distance": dist,
            "travel_energy": travel_energy,  # FIX: Store travel energy
            "battery_margin": battery_margin,
            "coverage_loss": coverage_loss,
            "relay_delay": relay_delay,
        })

    if len(rows) == 1:
        only = rows[0]
        diag = {
            "candidate_id": only["cand"]["id"],
            "distance": only["distance"],
            "travel_energy": round(only["travel_energy"], 3), # FIX: Output to diag
            "battery_margin": round(only["battery_margin"], 2),
            "coverage_loss": round(only["coverage_loss"], 6),
            "relay_delay": only["relay_delay"],
            "fuzzy_closeness": 1.0,
            "note": "only feasible candidate -- no ranking needed",
        }
        return only["cand"], [diag]

    dist_r = _fuzzy_normalize([r["distance"] for r in rows], higher_is_better=False)
    batt_r = _fuzzy_normalize([r["battery_margin"] for r in rows], higher_is_better=True)
    cov_r = _fuzzy_normalize([r["coverage_loss"] for r in rows], higher_is_better=False)
    delay_r = _fuzzy_normalize([r["relay_delay"] for r in rows], higher_is_better=False)

    w = weights
    scored = []
    for i, r in enumerate(rows):
        terms = [
            w["distance"] * (1 - dist_r[i]) ** p,
            w["battery_margin"] * (1 - batt_r[i]) ** p,
            w["coverage_loss"] * (1 - cov_r[i]) ** p,
            w["relay_delay"] * (1 - delay_r[i]) ** p,
        ]
        dist_to_ideal = sum(terms) ** (1.0 / p)
        closeness = 1.0 - dist_to_ideal   

        diag = {
            "candidate_id": r["cand"]["id"],
            "role": r["cand"]["role"],
            "distance": r["distance"],
            "travel_energy": round(r["travel_energy"], 3), # FIX: Output to diag
            "battery_margin": round(r["battery_margin"], 2),
            "coverage_loss": round(r["coverage_loss"], 6),
            "relay_delay": r["relay_delay"],
            "membership_distance": round(dist_r[i], 3),
            "membership_battery": round(batt_r[i], 3),
            "membership_coverage": round(cov_r[i], 3),
            "membership_delay": round(delay_r[i], 3),
            "fuzzy_closeness": round(closeness, 4),
        }
        scored.append((r["cand"], closeness, diag))

    scored.sort(key=lambda x: -x[1])
    best_cand = scored[0][0]
    return best_cand, [d for _, _, d in scored]


# ══════════════════════════════════════════════════════════════
# ONE SIMULATION STEP FOR THE WHOLE FLEET
# ══════════════════════════════════════════════════════════════
def step_fleet(state, relay_selector=select_best_relay):
    g, detected, drones, step = state["g"], state["detected"], state["drones"], state["step"]
    a20.advance_sim_clock(step)   # keep dynamic-wind clock in sync (no-op if WIND_DYNAMIC_ENABLED=False)
    by_id = {d["id"]: d for d in drones}

    for d in drones:
        if d["role"] != "returning":
            continue
        tr, tc = d["return_target"]
        if (d["r"], d["c"]) == (tr, tc):
            d["battery"] = 100.0
            d["cycles"] += 1  # one more full recharge cycle completed -> pack ages (SOH model)
            d["target_zone"] = None; d["target_cell"] = None
            d["pr"] = d["pc"] = None
            state["RC"] += 1
            active_now = sum(1 for x in drones
                              if x["id"] != d["id"] and x["role"] in ("patrol", "relay_incoming"))
            if active_now < state["target_patrol_count"]:
                d["role"] = "patrol"
            else:
                d["role"] = "idle_reserve"
        else:
            old_r, old_c = d["r"], d["c"]
            nr, nc = a20.move_toward(d["r"], d["c"], tr, tc)
            d["r"], d["c"] = nr, nc
            cost = a20.calculate_actual_step_cost(
                old_r, old_c, nr, nc, None, None,
                current_soc_pct=(d["battery"] if a20.USE_NONLINEAR_BATTERY else None),
                cycle_count=(d["cycles"] if a20.USE_SOH_AGING else 0))
            d["battery"] = max(0.0, d["battery"] - cost)
            state["total_energy_consumed"] += cost
            a20.sensor_sweep(nr, nc, g, detected, step=step)

    for d in drones:
        if d["role"] != "relay_incoming":
            continue
        needer = by_id.get(d["rendezvous_id"])
        if needer is None:
            d["role"] = "patrol"; continue
        tr, tc = needer["r"], needer["c"]
        dist_now = _manhattan(d["r"], d["c"], tr, tc)
        if dist_now <= 1:
            # ✅ RELAY FULFILLED — counted HERE (handoff actually completed),
            # not at dispatch time. Before this fix, relay_fulfilled was
            # incremented the moment a candidate was CHOSEN, which measures
            # "did the algorithm find someone", not "did the mission-critical
            # handoff actually happen". Under current physics a dispatched
            # relay_incoming drone always eventually reaches the needer
            # (movement is never hard-gated on battery), so this rarely
            # changes the final count today -- but it is the methodologically
            # correct place to count it, and stays correct if a future
            # failure mode (e.g. battery hits 0 mid-transit and the drone
            # can no longer move) is ever added.
            state["relay_fulfilled"] += 1
            st = a20.nearest_st(needer["r"], needer["c"])
            needer["role"] = "returning"
            needer["return_target"] = st
            needer["awaiting_relay"] = False
            needer["relay_episode_counted"] = False   # episode resolved -> next low-battery
                                                        # spell is a NEW episode
            d["role"] = "patrol"
            d["rendezvous_id"] = None
            d["target_zone"] = None; d["target_cell"] = None
            d["pr"] = d["pc"] = None
        else:
            old_r, old_c = d["r"], d["c"]
            nr, nc = a20.move_toward(d["r"], d["c"], tr, tc)
            d["r"], d["c"] = nr, nc
            cost = a20.calculate_actual_step_cost(
                old_r, old_c, nr, nc, None, None,
                current_soc_pct=(d["battery"] if a20.USE_NONLINEAR_BATTERY else None),
                cycle_count=(d["cycles"] if a20.USE_SOH_AGING else 0))
            d["battery"] = max(0.0, d["battery"] - cost)
            state["total_energy_consumed"] += cost
            a20.sensor_sweep(nr, nc, g, detected, step=step)

    for d in drones:
        if d["role"] != "patrol":
            continue

        if not d["awaiting_relay"]:
            nh, _st = a20.needs_handoff_now(d["r"], d["c"], d["battery"])
            if nh:
                # ✅ RELAY REQUEST COUNTER — counts EPISODES, not steps.
                # One continuous low-battery spell (needs_handoff_now stays
                # True across possibly several consecutive steps while no
                # feasible candidate is found) increments relay_requested
                # exactly ONCE, on its first step, via relay_episode_counted.
                # This makes relay_success_rate = fulfilled/requested a real
                # per-episode fulfillment rate instead of being diluted by
                # however many retry-steps a hard episode happened to need.
                # The flag resets (see step_fleet(), relay_incoming handling
                # above) once the handoff actually completes, so the next
                # distinct low-battery episode is counted fresh.
                if not d["relay_episode_counted"]:
                    state["relay_requested"] += 1
                    d["relay_episode_counted"] = True
                chosen, _diag = relay_selector(d, drones, g, detected, step)
                if chosen is not None:
                    d["awaiting_relay"] = True
                    real = by_id[chosen["id"]]
                    real["role"] = "relay_incoming"
                    real["rendezvous_id"] = d["id"]
                    
                    # ✅ DECISION LOGGING
                    state["log"].append({
                        "step": step,
                        "needer_id": d["id"],
                        "chosen_candidate_id": chosen["id"],
                        "candidates_diag": deepcopy(_diag)
                    })

        mrc, e_ret, st2 = a20.must_recharge_now(d["r"], d["c"], d["battery"])
        if mrc:
            cost = a20.calculate_actual_step_cost(
                d["r"], d["c"], d["r"], d["c"], d["pr"], d["pc"],
                current_soc_pct=(d["battery"] if a20.USE_NONLINEAR_BATTERY else None),
                cycle_count=(d["cycles"] if a20.USE_SOH_AGING else 0))
            d["battery"] = max(0.0, d["battery"] - cost)
            state["total_energy_consumed"] += cost
            a20.sensor_sweep(d["r"], d["c"], g, detected, step=step)
            continue

        if d["target_zone"]:
            zr, zc = d["target_zone"]
            if a20.get_zone_id(d["r"], d["c"]) == (zr, zc) and a20.find_uncovered_in_zone(zr, zc, g) is None:
                d["target_zone"] = None; d["target_cell"] = None
        if not d["target_zone"]:
            rk = a20.rank_zones(d["r"], d["c"], g, detected, step)

            # ── ZONE-CLAIMING COORDINATION FIX ──────────────────────────
            # Previously every patrol drone picked independently, with zero
            # awareness of what any other drone was already doing. Direct
            # diagnostic check (500-step, 3 seeds): 2+ drones were targeting
            # the SAME zone in ~50% of steps. That plausibly explains why the
            # 8-drone fleet's detection_rate/mission_completed came out WORSE
            # than a single drone's at 500 steps (86.9%/93.3% single-drone
            # vs ~89%/27% fleet) despite near-identical raw coverage_pct --
            # redundant effort on an already-claimed zone doesn't help find
            # threats sitting in a zone nobody has reached yet.
            #
            # Fix: exclude any zone another currently-patrolling drone has
            # already claimed as ITS target_zone from this drone's ranked
            # candidate list, before the existing tier/mixed-strategy
            # selection runs. Nothing about the ranking/selection logic
            # itself changes -- this only shrinks the candidate POOL, the
            # same way the relay-selection feasibility gate shrinks a
            # candidate pool without changing how candidates are scored.
            claimed = {x["target_zone"] for x in drones
                       if x["id"] != d["id"] and x["role"] == "patrol" and x["target_zone"] is not None}
            rk_avail = [entry for entry in rk if entry["zone"] not in claimed]
            if not rk_avail:
                # Every zone already claimed -- can't happen in practice with
                # 36 zones and at most NUM_DRONES=4 simultaneous patrol
                # drones, but fall back to the unfiltered list rather than
                # leaving this drone with no target at all.
                rk_avail = rk

            d["target_zone"], d["target_cell"] = a20.select_zone_mixed_strategy(
                rk_avail, g, d["r"], d["c"], d["battery"], d["zone_rng"], step)

        old_r, old_c = d["r"], d["c"]
        nr, nc, _bd = a20.smart_move(d["r"], d["c"], d["pr"], d["pc"], g, detected, d["target_zone"])
        d["r"], d["c"] = nr, nc
        cost = a20.calculate_actual_step_cost(
            old_r, old_c, nr, nc, old_r if d["pr"] is None else d["pr"],
            old_c if d["pc"] is None else d["pc"],
            current_soc_pct=(d["battery"] if a20.USE_NONLINEAR_BATTERY else None),
            cycle_count=(d["cycles"] if a20.USE_SOH_AGING else 0))
        d["battery"] = max(0.0, d["battery"] - cost)
        state["total_energy_consumed"] += cost
        a20.sensor_sweep(nr, nc, g, detected, step=step)
        d["pr"], d["pc"] = old_r, old_c

        if d["target_zone"]:
            d["target_cell"] = a20.find_uncovered_in_zone(*d["target_zone"], g)

    state["step"] += 1
    return state


# ══════════════════════════════════════════════════════════════
# METRICS & PLOTTING
# ══════════════════════════════════════════════════════════════
def collect_metrics(state):
    g, detected, drones = state["g"], state["detected"], state["drones"]
    active_drones = [d for d in drones]

    # ✅ avg_final_battery previously averaged ALL 8 drones together,
    # including the idle reserves that sit at 100% doing nothing -- with
    # NUM_DRONES=4 patrol + RESERVE_POOL_SIZE=4 reserve, that's a 50%
    # dilution pulling the number up regardless of how hard the patrol
    # drones actually worked. Split into the two pools that mean different
    # things; keep "avg_final_battery" as an alias for avg_patrol_battery
    # (the one every existing PRIMARY/SECONDARY_METRICS list and the
    # plotting code already reference) so nothing downstream breaks.
    patrol_pool = [d for d in active_drones if d["role"] != "idle_reserve"]
    reserve_pool = [d for d in active_drones if d["role"] == "idle_reserve"]
    avg_patrol_battery = round(statistics.mean(d["battery"] for d in patrol_pool), 2) if patrol_pool else None
    avg_reserve_battery = round(statistics.mean(d["battery"] for d in reserve_pool), 2) if reserve_pool else None

    staleness_values = [state["step"] - c["last_visited_step"] for c in g.values() if not c["is_station"]]
    avg_staleness = statistics.mean(staleness_values)
    max_staleness = max(staleness_values)   # S_max = max_z(t - t_last,z) -- persistent-surveillance metric

    requested = state["relay_requested"]
    fulfilled = state["relay_fulfilled"]
    success_rate = round(fulfilled / requested, 4) if requested > 0 else None

    # Pull the CHOSEN candidate's own diag out of each log entry (not every
    # candidate that was scored) so the averages below reflect the relays
    # that actually happened, not every option that was merely considered.
    # select_nearest_relay (baseline) never populates candidates_diag, so
    # these three will be None for baseline runs -- that's expected, not a bug.
    chosen_diags = []
    for entry in state["log"]:
        for diag in entry["candidates_diag"]:
            if diag.get("candidate_id") == entry["chosen_candidate_id"]:
                chosen_diags.append(diag)
                break

    # Defensive get() approach here as a fallback precaution
    avg_relay_delay = round(statistics.mean(d.get("relay_delay", 0) for d in chosen_diags), 3) if chosen_diags else None
    avg_relay_energy = round(statistics.mean(d.get("travel_energy", 0) for d in chosen_diags), 3) if chosen_diags else None
    avg_coverage_gap = round(statistics.mean(d.get("coverage_loss", 0) for d in chosen_diags), 3) if chosen_diags else None

    detection_rate = round(len(detected) / a20.NUM_THREATS, 4) if a20.NUM_THREATS else None

    return {
        # ── PRIMARY METRICS ──────────────────────────────────────────
        "coverage_pct": a20.coverage_pct(g),                       # 1. Coverage %
        "avg_zone_staleness_steps": round(avg_staleness, 2),       # 2. Average surveillance staleness
        "max_zone_staleness_steps": max_staleness,                 # 3. Maximum surveillance staleness (S_max)

        # ── SECONDARY METRICS ────────────────────────────────────────
        "detection_rate": detection_rate,                          # threats_detected / threats_total
        "threats_detected": len(detected),
        "threats_total": a20.NUM_THREATS,
        "relay_success_rate": success_rate,
        "avg_relay_delay_steps": avg_relay_delay,
        "avg_relay_energy_pct": avg_relay_energy,
        "avg_relay_coverage_gap": avg_coverage_gap,
        "mission_completed": state["first_all_threats_step"] is not None,   # all threats found within max_steps?
        "first_all_threats_step": state["first_all_threats_step"],
        "first_full_coverage_step": state["first_full_coverage_step"],
        "avg_final_battery": avg_patrol_battery,        # ALIAS: patrol-only, NOT diluted by idle reserves
        "avg_patrol_battery": avg_patrol_battery,        # explicit name, same value as above
        "avg_reserve_battery": avg_reserve_battery,      # sanity-check number, should sit near 100%
        "total_energy_consumed": round(state["total_energy_consumed"], 2),  # cumulative, survives recharges
        "relay_requested": requested,                              # number of relay events (episodes)
        "relay_fulfilled": fulfilled,

        # ── bookkeeping / plotting ────────────────────────────────────
        "RC": state["RC"],
        "final_drone_count": len(active_drones),
        "decision_log": state["log"]
    }

def simulate_fleet(seed, relay_selector=select_best_relay, num_drones=NUM_DRONES,
                    num_reserves=RESERVE_POOL_SIZE, max_steps=None):
    max_steps = max_steps or a20.MAX_STEPS
    state = init_fleet(seed, num_drones, num_reserves)
    for _ in range(max_steps):
        step_fleet(state, relay_selector=relay_selector)
        if state["first_all_threats_step"] is None and len(state["detected"]) >= a20.NUM_THREATS:
            state["first_all_threats_step"] = state["step"]
        if state["first_full_coverage_step"] is None and a20.coverage_pct(state["g"]) >= 100:
            state["first_full_coverage_step"] = state["step"]
    return collect_metrics(state)

def plot_decision_breakdown(log_entry, weights=DEFAULT_WEIGHTS,
                            save_path="decision_breakdown.png"):
    """Save a decision breakdown showing the three utility costs.

    Battery margin is shown only as feasibility/headroom information; it is
    NOT part of the RelayScore. This makes the plot consistent with the
    corrected constrained-utility formulation.
    """
    if 'pd' not in globals() or 'plt' not in globals():
        print("❌ Cannot plot: pandas or matplotlib is missing.")
        return

    data = []
    for diag in log_entry["candidates_diag"]:
        if diag.get("score") is None:
            continue

        cand_id = diag["candidate_id"]
        energy_cost = -(weights["W_TRAVEL_ENERGY"] * diag.get("travel_energy", 0))
        cov_loss_cost = -(weights["W_COVERAGE_LOSS"] * diag.get("coverage_loss", 0))
        delay_cost = -(weights["W_RELAY_DELAY"] * diag.get("relay_delay", 0))

        is_winner = cand_id == log_entry["chosen_candidate_id"]
        label = f"Drone {cand_id}\n(Winner)" if is_winner else f"Drone {cand_id}"

        data.append({
            "Candidate": label,
            "Feasibility headroom": diag.get("battery_margin", 0),
            "Energy Cost (-)": energy_cost,
            "Coverage Cost (-)": cov_loss_cost,
            "Delay Cost (-)": delay_cost,
            # NOTE: deliberately the SUM of the three raw-unit bars above,
            # not diag["score"] -- diag["score"] is now the ranking-time
            # utility, which for Policy C is pool-normalized to [0,1] and
            # therefore lives on a different scale than these raw-unit cost
            # bars (which recompute from raw travel_energy/coverage_loss/
            # relay_delay using whatever `weights` this function was passed).
            # Using the raw sum keeps the marker visually consistent with
            # the bars for BOTH old-format (Policy B) and new-format
            # (Policy C) diagnostics.
            "Net Utility": energy_cost + cov_loss_cost + delay_cost,
        })

    if not data:
        print("ℹ️ No feasible candidate diagnostics available for plotting.")
        return

    df = pd.DataFrame(data).set_index("Candidate")

    fig, ax = plt.subplots(figsize=(10, 7))
    df[["Energy Cost (-)", "Coverage Cost (-)", "Delay Cost (-)"]].plot(
        kind="bar", stacked=True, ax=ax, width=0.6
    )
    ax.plot(
        range(len(df)),
        df["Net Utility"],
        marker='D',
        linestyle='',
        markersize=8,
        label="Net Utility"
    )
    ax.axhline(0, linewidth=1.2)
    ax.set_title(
        f"Mission-Aware Relay Decision (Step {log_entry['step']} - "
        f"Needer: Drone {log_entry['needer_id']})",
        fontsize=12, fontweight='bold'
    )
    ax.set_ylabel("Weighted utility / cost units")
    ax.set_xticklabels(df.index, rotation=0)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"✅ Generated and saved decision breakdown chart to {save_path}")




def run_sensitivity_analysis(weight_name, sweep_values, seeds, max_steps=None,
                              base_weights=None, out_csv=None):
    """One-factor sensitivity analysis for the corrected constrained utility.

    The default model has only three utility weights:
      W_TRAVEL_ENERGY
      W_COVERAGE_LOSS
      W_RELAY_DELAY
    Battery is a feasibility constraint, not a soft utility term.

    NOTE (Aug 2026): defaults to NORMALIZED_WEIGHTS, not DEFAULT_WEIGHTS,
    because select_best_relay() now ranks on pool-normalized [0,1]
    attributes (see NORMALIZED_WEIGHTS docstring). Sweep values should be
    relative-importance fractions (e.g. 0.1-0.8), not raw multipliers like
    25 -- pass base_weights=DEFAULT_WEIGHTS explicitly only if you are
    deliberately reproducing the old scale-mismatched behavior.
    """
    base_weights = dict(base_weights or NORMALIZED_WEIGHTS)
    if weight_name not in base_weights:
        raise ValueError(
            f"Unknown weight '{weight_name}'. Choose from {list(base_weights)}"
        )

    results = []
    for val in sweep_values:
        trial_weights = dict(base_weights)
        trial_weights[weight_name] = val
        selector = partial(select_best_relay, weights=trial_weights)

        per_seed = [
            simulate_fleet(seed, relay_selector=selector, max_steps=max_steps)
            for seed in seeds
        ]

        def mean_metric(name):
            vals = [
                r[name] for r in per_seed
                if isinstance(r.get(name), (int, float))
            ]
            return round(statistics.mean(vals), 3) if vals else None

        row = {
            weight_name: val,
            "coverage_pct": mean_metric("coverage_pct"),
            "avg_zone_staleness_steps": mean_metric("avg_zone_staleness_steps"),
            "avg_final_battery": mean_metric("avg_final_battery"),
            "total_energy_consumed": mean_metric("total_energy_consumed"),
            "relay_success_rate": mean_metric("relay_success_rate"),
        }
        results.append(row)

        print(
            f"[{weight_name}={val}] "
            f"coverage={row['coverage_pct']}% "
            f"staleness={row['avg_zone_staleness_steps']} "
            f"energy={row['total_energy_consumed']}"
        )

    if out_csv and results:
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)

    return results




def plot_sensitivity(results, weight_name, metric="coverage_pct",
                      save_path="sensitivity_analysis.png"):
    """Line plot of `metric` vs the swept weight value, from
    run_sensitivity_analysis()'s output."""
    if 'pd' not in globals() or 'plt' not in globals():
        print("❌ Cannot plot: pandas or matplotlib is missing.")
        return

    df = pd.DataFrame(results)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(df[weight_name], df[metric], marker='o', color="#1f77b4")
    ax.set_xlabel(weight_name)
    ax.set_ylabel(metric)
    ax.set_title(f"Sensitivity of {metric} to {weight_name}", fontweight='bold')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"✅ Generated and saved sensitivity plot to {save_path}")


# ══════════════════════════════════════════════════════════════
# RUN EXECUTIONS
# ══════════════════════════════════════════════════════════════
def run_comparison(seeds, max_steps=None, out_csv=None, weights=None):
    """Paired B-vs-C comparison: same 8-drone fleet, different selector."""
    selector_c = partial(select_best_relay, weights=weights) if weights else select_best_relay

    rows = []
    for seed in seeds:
        base = simulate_fleet(seed, relay_selector=select_nearest_relay,
                              max_steps=max_steps)
        opt = simulate_fleet(seed, relay_selector=selector_c,
                             max_steps=max_steps)

        row = {"seed": seed}
        for k, v in base.items():
            if k != "decision_log":
                row[f"baseline_{k}"] = v
        for k, v in opt.items():
            if k != "decision_log":
                row[f"optimized_{k}"] = v
        rows.append(row)

        print(
            f"[seed {seed}] "
            f"baseline cov={base['coverage_pct']}% "
            f"staleness={base['avg_zone_staleness_steps']} "
            f"relay_success={base['relay_success_rate']} | "
            f"optimized cov={opt['coverage_pct']}% "
            f"staleness={opt['avg_zone_staleness_steps']} "
            f"relay_success={opt['relay_success_rate']}"
        )

    if out_csv and rows:
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    metrics = [
        "coverage_pct",
        "avg_zone_staleness_steps",
        "max_zone_staleness_steps",
        "detection_rate",
        "avg_final_battery",
        "total_energy_consumed",
        "relay_success_rate",
        "avg_relay_delay_steps",
        "avg_relay_energy_pct",
        "avg_relay_coverage_gap",
    ]

    summary = {}
    for metric in metrics:
        base_vals = [r[f"baseline_{metric}"] for r in rows]
        opt_vals = [r[f"optimized_{metric}"] for r in rows]
        stats = _safe_wilcoxon(base_vals, opt_vals)

        summary[metric] = {
            "baseline_mean": round(
                statistics.mean(
                    [x for x in base_vals if isinstance(x, (int, float))]
                ), 3
            ) if any(isinstance(x, (int, float)) for x in base_vals) else None,
            "optimized_mean": round(
                statistics.mean(
                    [x for x in opt_vals if isinstance(x, (int, float))]
                ), 3
            ) if any(isinstance(x, (int, float)) for x in opt_vals) else None,
            **stats,
        }

    return rows, summary




# ══════════════════════════════════════════════════════════════
# THREE-WAY COMPARISON: A (original single-drone + nearest handoff)
#                        vs B (fixed 8-drone fleet + nearest relay)
#                        vs C (fixed 8-drone fleet + RelayScore)
#
# A vs B isolates the FLEET-SIZE effect (1 active drone + serial
# station-launched backup  ->  8 parallel drones with an idle reserve pool).
# B vs C isolates the RELAY-INTELLIGENCE effect (same 8-drone fleet, only
# the candidate-selection rule changes: nearest-distance vs RelayScore).
# This is exactly the ablation a reviewer needs to see that C's improvement
# isn't just "more drones" in disguise.
#
# NOTE on "fixed 8-drone fleet": NUM_DRONES(4) + RESERVE_POOL_SIZE(4) already
# equals 8 in this file's defaults, so B and C below use init_fleet()'s
# existing defaults unchanged -- no config change was needed to get "8".
# ══════════════════════════════════════════════════════════════
def _extract_baseline_a_metrics(seed, max_steps):
    """Runs a20.py's ORIGINAL architecture unchanged: exactly one active
    drone at a time, backup launched from the nearest station on
    needs_handoff_now() (a20._advance_relay_policy / DroneSimHeadless "s_"
    fields), using SMRS zone-tier movement. Then reshapes its state into
    the SAME metric keys collect_metrics() produces, so Baseline A lines up
    directly against Baseline B / Proposed C in one table.

    LIMITATION (report in thesis): this architecture has no feasibility
    check and no competing candidates -- a backup is simply assumed
    available at the nearest station every time. So relay_success_rate is
    trivially ~1.0 here and avg_relay_delay/energy/coverage_gap are not
    computed the same way RelayScore computes them (marked None). This is
    an apples-to-oranges caveat inherent to comparing against the
    original single-drone design, not a bug in the metric extraction.
    """
    sim = a20.DroneSimHeadless(seed)
    sim.run(max_steps)

    g, detected, step = sim.gs, sim.s_detected, sim.step
    staleness_values = [step - c["last_visited_step"] for c in g.values() if not c["is_station"]]
    avg_staleness = statistics.mean(staleness_values) if staleness_values else 0.0
    max_staleness = max(staleness_values) if staleness_values else 0.0

    requested = sim.sR if sim.sR > 0 else (1 if sim.s_handoff_mode else 0)
    fulfilled = sim.sR   # every handoff in this architecture succeeds by construction

    return {
        "coverage_pct": a20.coverage_pct(g),
        "avg_zone_staleness_steps": round(avg_staleness, 2),
        "max_zone_staleness_steps": max_staleness,
        "detection_rate": round(len(detected) / a20.NUM_THREATS, 4) if a20.NUM_THREATS else None,
        "threats_detected": len(detected),
        "threats_total": a20.NUM_THREATS,
        "relay_success_rate": round(fulfilled / requested, 4) if requested > 0 else None,
        "avg_relay_delay_steps": None,     # not computed in the original architecture (see docstring)
        "avg_relay_energy_pct": None,
        "avg_relay_coverage_gap": None,
        "mission_completed": sim.s_first_all is not None,
        "first_all_threats_step": sim.s_first_all,
        "first_full_coverage_step": sim.s_full_cov_step,
        "avg_final_battery": round(sim.s_active["b"], 2),
        "avg_patrol_battery": round(sim.s_active["b"], 2),   # A only ever has 1 active drone
        "avg_reserve_battery": None,                          # A has no idle-reserve concept -- N/A
        "total_energy_consumed": round(sim.s_energy, 2),      # a20's own running total (see DroneSimHeadless),
                                                                # NOT reset by recharge, same semantics as B/C's
                                                                # total_energy_consumed -- genuinely comparable
        "relay_requested": requested,
        "relay_fulfilled": fulfilled,
        "RC": sim.sR,
        "final_drone_count": 1,
    }


PRIMARY_METRICS = ["coverage_pct", "avg_zone_staleness_steps", "max_zone_staleness_steps"]
SECONDARY_METRICS = ["detection_rate", "relay_success_rate", "avg_relay_delay_steps",
                      "avg_relay_energy_pct", "mission_completed", "avg_final_battery",
                      "avg_patrol_battery", "avg_reserve_battery", "total_energy_consumed",
                      "relay_requested"]


def run_three_way_comparison(seeds, max_steps=None, num_drones=NUM_DRONES,
                              num_reserves=RESERVE_POOL_SIZE, out_csv=None):
    """Runs Baseline A / Baseline B / Proposed C on the SAME seeds and
    reports the 3 primary + secondary metrics for each, plus paired
    significance tests for the two comparisons that matter:
      A vs B -> fleet-size effect
      B vs C -> relay-intelligence effect (RelayScore over nearest, same fleet)
    """
    max_steps = max_steps or a20.MAX_STEPS
    rows = []
    for seed in seeds:
        m_a = _extract_baseline_a_metrics(seed, max_steps)
        m_b = simulate_fleet(seed, relay_selector=select_nearest_relay,
                              num_drones=num_drones, num_reserves=num_reserves, max_steps=max_steps)
        m_c = simulate_fleet(seed, relay_selector=select_best_relay,
                              num_drones=num_drones, num_reserves=num_reserves, max_steps=max_steps)
        row = {"seed": seed}
        for label, m in (("A", m_a), ("B", m_b), ("C", m_c)):
            for k, v in m.items():
                if k in ("decision_log",):
                    continue
                row[f"{label}_{k}"] = v
        rows.append(row)
        print(f"[seed {seed}] "
              f"A(orig): cov={m_a['coverage_pct']}% staleness(avg/max)={m_a['avg_zone_staleness_steps']}/{m_a['max_zone_staleness_steps']}  |  "
              f"B(8-drone,nearest): cov={m_b['coverage_pct']}% staleness(avg/max)={m_b['avg_zone_staleness_steps']}/{m_b['max_zone_staleness_steps']}  |  "
              f"C(8-drone,RelayScore): cov={m_c['coverage_pct']}% staleness(avg/max)={m_c['avg_zone_staleness_steps']}/{m_c['max_zone_staleness_steps']}")

    if out_csv:
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    summary = {}
    for metric in PRIMARY_METRICS + SECONDARY_METRICS:
        a_vals = [r[f"A_{metric}"] for r in rows]
        b_vals = [r[f"B_{metric}"] for r in rows]
        c_vals = [r[f"C_{metric}"] for r in rows]
        entry = {}
        for label, vals in (("A", a_vals), ("B", b_vals), ("C", c_vals)):
            clean = [v for v in vals if isinstance(v, (int, float))]
            entry[f"{label}_mean"] = round(statistics.mean(clean), 3) if clean else None
        fleet_effect = _safe_wilcoxon(a_vals, b_vals)  # A vs B
        relay_effect = _safe_wilcoxon(b_vals, c_vals)  # B vs C
        entry["fleet_effect_A_vs_B"] = fleet_effect
        entry["relay_intel_effect_B_vs_C"] = relay_effect
        summary[metric] = entry
    print("\n=== A vs B vs C SUMMARY (fleet-size effect: A->B | relay-intelligence effect: B->C) ===")
    print(json.dumps(summary, indent=2, default=str))
    return rows, summary


def _print_significance_summary(summary, comparisons, n=None, alpha=0.05):
    """Human-readable readout of the paired Wilcoxon tests already stored
    in `summary` (from _safe_wilcoxon). `comparisons` is a list of
    (summary_key, label_left, label_right) tuples, e.g.
    ("ranking_rule_effect_C_vs_D", "C", "D").

    With only a handful of seeds, MOST rows correctly print
    "not significant" -- that is not a bug, it is the honest finding:
    a coverage_pct difference you saw on 5 seeds is very likely just
    seed-to-seed noise, not a real effect of C vs D. Only trust a
    "SIGNIFICANT" row, and even then prefer to see it hold up as you
    add more seeds rather than treating one run as final.
    """
    print(f"\n--- Statistical significance readout (alpha={alpha}, n={n} seeds) ---")
    for metric, entry in summary.items():
        for key, label_a, label_b in comparisons:
            stat = entry.get(key)
            if not stat or stat.get("wilcoxon_p") is None:
                continue
            p = stat["wilcoxon_p"]
            mean_a = entry.get(f"{label_a}_mean")
            mean_b = entry.get(f"{label_b}_mean")
            tag = "SIGNIFICANT" if p < alpha else "not significant"
            marker = "✅" if p < alpha else "—"
            test_note = " [t-test fallback]" if stat.get("test_used") == "ttest_rel_fallback" else ""
            print(f"{marker} [{metric}] {label_a}_mean={mean_a}  {label_b}_mean={mean_b}  "
                  f"diff({label_b}-{label_a})={stat['mean_diff']}  p={p}{test_note}  ({tag})")


def run_four_way_comparison(seeds, max_steps=None, num_drones=NUM_DRONES,
                             num_reserves=RESERVE_POOL_SIZE, out_csv=None):
    """Runs Baseline A / Baseline B / Proposed C / Policy D (fuzzy MADM,
    adapted from Zhu, Zhou & Zhang 2017) on the SAME seeds and the SAME
    8-drone fleet (B/C/D), reporting paired significance tests for:
      A vs B -> fleet-size effect
      B vs C -> relay-intelligence effect (RelayScore utility over nearest)
      B vs D -> relay-intelligence effect (fuzzy MADM over nearest)
      C vs D -> RelayScore utility function vs fuzzy MADM ranking rule,
                same feasible candidate pool, same 8-drone fleet -- this
                is the comparison that isolates "your ranking rule" vs
                "the paper's-style ranking rule" specifically.

    See select_fuzzy_relay() docstring for the fidelity caveat: D is a
    methodological analogue of the cited paper's approach, not a literal
    reproduction of its own attribute formulas.
    """
    max_steps = max_steps or a20.MAX_STEPS
    rows = []
    for seed in seeds:
        m_a = _extract_baseline_a_metrics(seed, max_steps)
        m_b = simulate_fleet(seed, relay_selector=select_nearest_relay,
                              num_drones=num_drones, num_reserves=num_reserves, max_steps=max_steps)
        m_c = simulate_fleet(seed, relay_selector=select_best_relay,
                              num_drones=num_drones, num_reserves=num_reserves, max_steps=max_steps)
        m_d = simulate_fleet(seed, relay_selector=select_fuzzy_relay,
                              num_drones=num_drones, num_reserves=num_reserves, max_steps=max_steps)
        row = {"seed": seed}
        for label, m in (("A", m_a), ("B", m_b), ("C", m_c), ("D", m_d)):
            for k, v in m.items():
                if k in ("decision_log",):
                    continue
                row[f"{label}_{k}"] = v
        rows.append(row)
        print(f"[seed {seed}] "
              f"A(orig): cov={m_a['coverage_pct']}%  |  "
              f"B(nearest): cov={m_b['coverage_pct']}%  |  "
              f"C(RelayScore): cov={m_c['coverage_pct']}%  |  "
              f"D(fuzzy-MADM): cov={m_d['coverage_pct']}%")

    if out_csv:
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    summary = {}
    for metric in PRIMARY_METRICS + SECONDARY_METRICS:
        vals = {}
        for label in ("A", "B", "C", "D"):
            vals[label] = [r[f"{label}_{metric}"] for r in rows]
        entry = {}
        for label in ("A", "B", "C", "D"):
            clean = [v for v in vals[label] if isinstance(v, (int, float))]
            entry[f"{label}_mean"] = round(statistics.mean(clean), 3) if clean else None
        entry["fleet_effect_A_vs_B"] = _safe_wilcoxon(vals["A"], vals["B"])
        entry["relay_intel_effect_B_vs_C"] = _safe_wilcoxon(vals["B"], vals["C"])
        entry["relay_intel_effect_B_vs_D"] = _safe_wilcoxon(vals["B"], vals["D"])
        entry["ranking_rule_effect_C_vs_D"] = _safe_wilcoxon(vals["C"], vals["D"])
        summary[metric] = entry

    print("\n=== A vs B vs C vs D SUMMARY ===")
    print("(fleet-size: A->B | RelayScore vs nearest: B->C | "
          "fuzzy-MADM vs nearest: B->D | RelayScore vs fuzzy-MADM: C->D)")
    print(json.dumps(summary, indent=2, default=str))

    _print_significance_summary(
        summary,
        comparisons=[
            ("fleet_effect_A_vs_B", "A", "B"),
            ("relay_intel_effect_B_vs_C", "B", "C"),
            ("relay_intel_effect_B_vs_D", "B", "D"),
            ("ranking_rule_effect_C_vs_D", "C", "D"),
        ],
        n=len(seeds),
    )
    return rows, summary


def run_five_way_comparison(seeds, max_steps=None, num_drones=NUM_DRONES,
                             num_reserves=RESERVE_POOL_SIZE, out_csv=None):
    """Runs A / B / C / D / E on the SAME seeds and the SAME 8-drone fleet
    (B/C/D/E), where E = Policy C plus the dynamic-weighting layer
    (select_best_relay_dynamic). This isolates the dynamic layer's OWN
    marginal contribution, on top of the pool-normalization fix C already
    has:
      C vs E -> does dynamic (state-dependent) weighting help RelayScore
                beyond static pool-normalized weighting alone?
      D vs E -> does the fully-refined RelayScore now beat fuzzy MADM?

    Use this after run_four_way_comparison() has shown C is close to but
    not clearly beating D (e.g. under dynamic wind) -- that is exactly the
    situation E is meant to close.
    """
    max_steps = max_steps or a20.MAX_STEPS
    rows = []
    for seed in seeds:
        m_a = _extract_baseline_a_metrics(seed, max_steps)
        m_b = simulate_fleet(seed, relay_selector=select_nearest_relay,
                              num_drones=num_drones, num_reserves=num_reserves, max_steps=max_steps)
        m_c = simulate_fleet(seed, relay_selector=select_best_relay,
                              num_drones=num_drones, num_reserves=num_reserves, max_steps=max_steps)
        m_d = simulate_fleet(seed, relay_selector=select_fuzzy_relay,
                              num_drones=num_drones, num_reserves=num_reserves, max_steps=max_steps)
        m_e = simulate_fleet(seed, relay_selector=select_best_relay_dynamic,
                              num_drones=num_drones, num_reserves=num_reserves, max_steps=max_steps)
        row = {"seed": seed}
        for label, m in (("A", m_a), ("B", m_b), ("C", m_c), ("D", m_d), ("E", m_e)):
            for k, v in m.items():
                if k in ("decision_log",):
                    continue
                row[f"{label}_{k}"] = v
        rows.append(row)
        print(f"[seed {seed}] "
              f"B(nearest): cov={m_b['coverage_pct']}%  |  "
              f"C(RelayScore-static): cov={m_c['coverage_pct']}%  |  "
              f"D(fuzzy-MADM): cov={m_d['coverage_pct']}%  |  "
              f"E(RelayScore-dynamic): cov={m_e['coverage_pct']}%")

    if out_csv:
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    summary = {}
    for metric in PRIMARY_METRICS + SECONDARY_METRICS:
        vals = {}
        for label in ("A", "B", "C", "D", "E"):
            vals[label] = [r[f"{label}_{metric}"] for r in rows]
        entry = {}
        for label in ("A", "B", "C", "D", "E"):
            clean = [v for v in vals[label] if isinstance(v, (int, float))]
            entry[f"{label}_mean"] = round(statistics.mean(clean), 3) if clean else None
        entry["fleet_effect_A_vs_B"] = _safe_wilcoxon(vals["A"], vals["B"])
        entry["relay_intel_effect_B_vs_C"] = _safe_wilcoxon(vals["B"], vals["C"])
        entry["relay_intel_effect_B_vs_D"] = _safe_wilcoxon(vals["B"], vals["D"])
        entry["ranking_rule_effect_C_vs_D"] = _safe_wilcoxon(vals["C"], vals["D"])
        entry["dynamic_weighting_effect_C_vs_E"] = _safe_wilcoxon(vals["C"], vals["E"])
        entry["fully_refined_effect_D_vs_E"] = _safe_wilcoxon(vals["D"], vals["E"])
        summary[metric] = entry

    print("\n=== A vs B vs C vs D vs E SUMMARY ===")
    print("(RelayScore vs fuzzy-MADM: C->D | dynamic layer's own effect: C->E | "
          "fully-refined RelayScore vs fuzzy-MADM: D->E)")
    print(json.dumps(summary, indent=2, default=str))

    _print_significance_summary(
        summary,
        comparisons=[
            ("fleet_effect_A_vs_B", "A", "B"),
            ("relay_intel_effect_B_vs_C", "B", "C"),
            ("relay_intel_effect_B_vs_D", "B", "D"),
            ("ranking_rule_effect_C_vs_D", "C", "D"),
            ("dynamic_weighting_effect_C_vs_E", "C", "E"),
            ("fully_refined_effect_D_vs_E", "D", "E"),
        ],
        n=len(seeds),
    )
    return rows, summary
if __name__ == "__main__":
    import sys

    # Usage: python a2e_relay_fixed_final.py [max_steps] [num_seeds]
    # Defaults now favor a THESIS-GRADE run (30 seeds) over a quick smoke
    # test, because 5 seeds is not enough to trust any A/B/C/D difference
    # (see the significance readout after the four-way comparison below).
    # For a fast sanity check while iterating on code, run e.g.:
    #   python a2e_relay_fixed_final.py 300 5
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    num_seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    seeds = list(range(1, num_seeds + 1))
    # Lighter seed set for the earlier B-vs-C-only sections and the
    # sensitivity sweep, so those don't balloon runtime unnecessarily --
    # trim/expand this independently of the main ABCD seed set above.
    quick_seeds = seeds[:5]

    print("\n=== B vs C: FEASIBLE-NEAREST vs CORRECTED RELAYSCORE ===")
    rows, summary = run_comparison(
        seeds, max_steps=steps, out_csv="fleet_relay_comparison_fixed.csv"
    )
    print(json.dumps(summary, indent=2, default=str))

    print("\n=== GENERATING DECISION BREAKDOWN CHART ===")
    state_metrics = simulate_fleet(1, select_best_relay, max_steps=steps)
    d_log = state_metrics.get("decision_log", [])
    if d_log:
        interesting_logs = [x for x in d_log if len(x["candidates_diag"]) > 1]
        target_log = interesting_logs[0] if interesting_logs else d_log[0]
        plot_decision_breakdown(
            target_log,
            save_path="decision_breakdown_fixed.png"
        )
    else:
        print("ℹ️ No relay decisions occurred in this run to plot.")

    print("\n=== SENSITIVITY: W_COVERAGE_LOSS ===")
    # Uses quick_seeds (not the full ABCD seed set) -- 8 weight values x 30
    # seeds would be a lot of sims just for a sensitivity sweep. Widen this
    # to `seeds` too once you're ready to report it in the thesis.
    #
    # NOTE (Aug 2026): sweep values are now relative-importance FRACTIONS
    # (matching NORMALIZED_WEIGHTS' scale), not raw multipliers -- the old
    # [5, 10, ..., 50] range was calibrated for the pre-fix raw-cost model
    # and would be meaningless here. 0.1 = coverage barely matters,
    # 0.8 = coverage dominates almost everything else in the ranking.
    sens_results = run_sensitivity_analysis(
        "W_COVERAGE_LOSS",
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        seeds=quick_seeds,
        max_steps=steps,
        out_csv="sensitivity_coverage_weight_fixed.csv",
    )
    plot_sensitivity(
        sens_results,
        "W_COVERAGE_LOSS",
        metric="coverage_pct",
        save_path="sensitivity_coverage_weight_fixed.png",
    )

    print("\n=== THREE-WAY: A original | B feasible-nearest | C corrected RelayScore ===")
    abc_rows, abc_summary = run_three_way_comparison(
        seeds,
        max_steps=steps,
        out_csv="abc_comparison_fixed.csv",
    )
    print(json.dumps(abc_summary, indent=2, default=str))

    print("\n=== FOUR-WAY: A original | B nearest | C RelayScore | D fuzzy-MADM (paper-adapted) ===")
    abcd_rows, abcd_summary = run_four_way_comparison(
        seeds,
        max_steps=steps,
        out_csv="abcd_comparison_with_fuzzy.csv",
    )
    print(json.dumps(abcd_summary, indent=2, default=str))