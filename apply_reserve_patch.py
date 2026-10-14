# -*- coding: utf-8 -*-
"""
apply_reserve_patch.py  --  adds the reserve-pool opportunity-cost term to
a2e_relay_fixed_final_51.py.

WHAT IT FIXES
-------------
The coverage-loss model prices only the zone a candidate abandons. An
idle-reserve candidate abandons nothing, so it scores L_i = 0 -- a perfect
coverage attribute, for free, every time. Every policy therefore prefers
reserves, the reserve pool drains early, and the later relay requests have to
be served by patrolling drones, which is exactly when coverage collapses. The
cost of consuming a reserve is real but deferred, and the model never charged
for it. Your own Section VII names this limitation.

This patch charges it:

    L_i  +=  lambda * scarcity ** gamma        (idle-reserve candidates only)
    scarcity = 1 - (reserves_available - 1) / reserves_at_mission_start

A full pool costs almost nothing to draw from; the last reserve costs the full
lambda. The term is applied inside _coverage_loss_for_candidate(), so EVERY
policy (C, D, Dref, D2, T, V, I, ...) sees it identically -- it is a change to
the mission model, not to any one selector. That is what makes it reportable
as a modelling contribution rather than as tuning in RelayScore's favour.

USAGE
-----
    python apply_reserve_patch.py                # patch in place (makes .bak)
    python apply_reserve_patch.py --check        # report status, change nothing
    python apply_reserve_patch.py --restore      # undo, from the .bak

The patch is idempotent: running it twice is safe.
"""

import os
import shutil
import sys

TARGET = "a2e_relay_fixed_final_51.py"
BACKUP = TARGET + ".prereserve.bak"
MARKER = "RESERVE_OPPORTUNITY_LAMBDA"


# ── (anchor, replacement, expected_count) ────────────────────────────────────
EDITS = []

# 1. constants -------------------------------------------------------------
EDITS.append((
    'PRIORITY_ZONE_WEIGHT = 1.0     # weight for border priority',
    '''PRIORITY_ZONE_WEIGHT = 1.0     # weight for border priority

# ══════════════════════════════════════════════════════════════
# RESERVE-POOL OPPORTUNITY COST
# ══════════════════════════════════════════════════════════════
# Marginal coverage loss L_i prices the zone a candidate stops watching.
# An idle reserve watches no zone, so L_i = 0 and it looks free on the
# coverage attribute no matter how scarce reserves have become. Consuming a
# reserve IS costly -- it just pays later, when no reserve is left and a
# patrolling drone has to be pulled off its zone instead. This term prices
# that deferred cost at decision time:
#
#     L_i += RESERVE_OPPORTUNITY_LAMBDA * scarcity ** RESERVE_OPPORTUNITY_GAMMA
#     scarcity = 1 - (available_after_this_one) / pool_at_mission_start
#
# scarcity -> 1/N when the pool is untouched, -> 1 for the last reserve.
# GAMMA > 1 makes the pool cheap until it starts running out.
#
# Scale note: L_i lives on the same scale as COVERAGE_LOSS_UNIVERSE in
# policy_d_true_fis.py (0 .. 0.06), so LAMBDA must be read on that scale.
# 0.030 makes an exhausted pool cost about as much as a mid-range zone
# abandonment. Sweep it -- the sensitivity curve is a results subsection.
RESERVE_OPPORTUNITY_ENABLED = True
RESERVE_OPPORTUNITY_LAMBDA = 0.030
RESERVE_OPPORTUNITY_GAMMA = 2.0
# Also charge patrolling candidates that currently hold no zone assignment?
# They do not consume a reserve, so the default is False.
RESERVE_OPPORTUNITY_APPLY_TO_UNASSIGNED = False''',
    1,
))

# 2. tag reserve slots at fleet construction -------------------------------
EDITS.append((
    '''        rd["role"] = "idle_reserve"
        rd["home_station"] = st''',
    '''        rd["role"] = "idle_reserve"
        rd["reserve_slot"] = True      # permanent tag: counts toward the pool
                                        # size even after this drone is used
        rd["home_station"] = st''',
    1,
))
EDITS.append((
    '''    drones = [_new_drone(i, seed) for i in range(num_drones)]
    for d in drones:
        d["role"] = "patrol"''',
    '''    drones = [_new_drone(i, seed) for i in range(num_drones)]
    for d in drones:
        d["role"] = "patrol"
        d["reserve_slot"] = False''',
    1,
))

# 3. helper functions ------------------------------------------------------
EDITS.append((
    '''def _coverage_loss_for_candidate(candidate, needer, g, detected, step, relay_delay):''',
    '''def _reserve_pool_state(fleet):
    """(available_now, pool_at_mission_start) for the idle-reserve pool.

    `available_now` counts drones that are idle reserves right now and not
    already committed to another relay. `pool_at_mission_start` counts the
    reserve slots the fleet was built with, so the denominator does not shrink
    as reserves get consumed. Falls back to the live count on fleets built
    before this patch (no reserve_slot tag), which keeps old pickles working.
    """
    if not fleet:
        return 0, 0
    available = 0
    capacity = 0
    tagged = False
    for d in fleet:
        if d.get("reserve_slot") is not None:
            tagged = True
            if d.get("reserve_slot"):
                capacity += 1
        if d.get("role") == "idle_reserve" and not d.get("awaiting_relay", False):
            available += 1
    if not tagged or capacity == 0:
        capacity = max(available, 1)
    return available, capacity


def _reserve_opportunity_cost(candidate, fleet,
                              lam=None, gamma=None):
    """Deferred cost of consuming one drone from the idle-reserve pool.

    Returns 0.0 for candidates that are not drawn from the pool, and for every
    candidate when the term is disabled -- so switching
    RESERVE_OPPORTUNITY_ENABLED off reproduces the pre-patch behaviour exactly.
    """
    if not RESERVE_OPPORTUNITY_ENABLED or not fleet:
        return 0.0

    role = candidate.get("role")
    draws_from_pool = (role == "idle_reserve")
    if not draws_from_pool and RESERVE_OPPORTUNITY_APPLY_TO_UNASSIGNED:
        draws_from_pool = candidate.get("target_zone") is None
    if not draws_from_pool:
        return 0.0

    available, capacity = _reserve_pool_state(fleet)
    if capacity <= 0:
        return 0.0

    remaining_after = max(available - 1, 0)
    scarcity = 1.0 - (remaining_after / float(capacity))
    scarcity = min(max(scarcity, 0.0), 1.0)

    lam = RESERVE_OPPORTUNITY_LAMBDA if lam is None else lam
    gamma = RESERVE_OPPORTUNITY_GAMMA if gamma is None else gamma
    return float(lam) * (scarcity ** float(gamma))


def _coverage_loss_for_candidate(candidate, needer, g, detected, step, relay_delay,
                                 fleet=None):''',
    1,
))

# 4. body of _coverage_loss_for_candidate ----------------------------------
EDITS.append((
    '''    if candidate["role"] == "idle_reserve":
        return 0.0, 0.0, 0.0, None

    if candidate["target_zone"] is None:
        return 0.0, 0.0, 0.0, None''',
    '''    reserve_cost = _reserve_opportunity_cost(candidate, fleet)

    if candidate["role"] == "idle_reserve":
        # No zone is abandoned, but the reserve pool shrinks by one.
        return reserve_cost, 0.0, 0.0, ({"reserve_opportunity_cost": round(reserve_cost, 6)}
                                        if reserve_cost else None)

    if candidate["target_zone"] is None:
        return reserve_cost, 0.0, 0.0, ({"reserve_opportunity_cost": round(reserve_cost, 6)}
                                        if reserve_cost else None)''',
    1,
))
EDITS.append((
    '''    weighted_loss = marginal_loss * penalty_mult
    return weighted_loss, marginal_loss, current_tgap, {
        "has_threat": has_threat,
        "border_priority": border_pr,
        "penalty_multiplier": round(penalty_mult, 3),
        "current_tgap": round(current_tgap, 3),
        "projected_tgap": round(projected_tgap, 3),
    }''',
    '''    weighted_loss = marginal_loss * penalty_mult + reserve_cost
    return weighted_loss, marginal_loss, current_tgap, {
        "has_threat": has_threat,
        "border_priority": border_pr,
        "penalty_multiplier": round(penalty_mult, 3),
        "current_tgap": round(current_tgap, 3),
        "projected_tgap": round(projected_tgap, 3),
        "reserve_opportunity_cost": round(reserve_cost, 6),
    }''',
    1,
))

# 5. predictive variant (Policy J) -----------------------------------------
EDITS.append((
    '''    if candidate["role"] == "idle_reserve" or candidate["target_zone"] is None:''',
    '''    reserve_cost = _reserve_opportunity_cost(candidate, fleet)
    if candidate["role"] == "idle_reserve" or candidate["target_zone"] is None:
        if reserve_cost:
            return reserve_cost, 0.0, 0.0, {"reserve_opportunity_cost": round(reserve_cost, 6)}''',
    1,
))

# 6. thread `fleet` through every scoring helper ---------------------------
SIGNATURE_EDITS = [
    ('def relay_score(candidate, needer, g, detected, step, weights=None):',
     'def relay_score(candidate, needer, g, detected, step, weights=None, fleet=None):'),
    ('def _relay_utility_pool(feasible, needer, g, detected, step, weights=None):',
     'def _relay_utility_pool(feasible, needer, g, detected, step, weights=None, fleet=None):'),
    ('def _relay_utility_pool_g(feasible, needer, g, detected, step, weights=None):',
     'def _relay_utility_pool_g(feasible, needer, g, detected, step, weights=None, fleet=None):'),
    ('def _relay_utility_pool_h(feasible, needer, g, detected, step, weights=None):',
     'def _relay_utility_pool_h(feasible, needer, g, detected, step, weights=None, fleet=None):'),
    ('def _relay_utility_pool_i(feasible, needer, g, detected, step, weights=None):',
     'def _relay_utility_pool_i(feasible, needer, g, detected, step, weights=None, fleet=None):'),
    ('def _madm_build_rows(needer, drones, g, detected, step):',
     'def _madm_build_rows(needer, drones, g, detected, step):  # fleet == drones'),
]
for a, b in SIGNATURE_EDITS:
    EDITS.append((a, b, 1))

EDITS.append((
    '''def _predicted_coverage_loss_for_candidate(candidate, needer, g, detected, step, relay_delay,''',
    '''def _predicted_coverage_loss_for_candidate(candidate, needer, g, detected, step, relay_delay,
                                            fleet=None,''',
    1,
))
EDITS.append((
    '''def _relay_utility_pool_predictive(feasible, needer, g, detected, step, weights=None,''',
    '''def _relay_utility_pool_predictive(feasible, needer, g, detected, step, weights=None,
                                    fleet=None,''',
    1,
))

# callers of the pool helpers
EDITS.append(('scored = _relay_utility_pool(feasible, needer, g, detected, step, weights)',
              'scored = _relay_utility_pool(feasible, needer, g, detected, step, weights, fleet=drones)', 1))
EDITS.append(('scored = _relay_utility_pool_g(feasible, needer, g, detected, step, weights)',
              'scored = _relay_utility_pool_g(feasible, needer, g, detected, step, weights, fleet=drones)', 1))
EDITS.append(('scored = _relay_utility_pool_h(feasible, needer, g, detected, step, weights)',
              'scored = _relay_utility_pool_h(feasible, needer, g, detected, step, weights, fleet=drones)', 1))
EDITS.append(('scored = _relay_utility_pool_i(feasible, needer, g, detected, step, weights)',
              'scored = _relay_utility_pool_i(feasible, needer, g, detected, step, weights, fleet=drones)', 1))
EDITS.append(('scored = _relay_utility_pool(feasible, needer, g, detected, step, dyn_weights)',
              'scored = _relay_utility_pool(feasible, needer, g, detected, step, dyn_weights, fleet=drones)', 1))
EDITS.append((
    'scored = _relay_utility_pool_predictive(feasible, needer, g, detected, step, weights, horizon=horizon)',
    'scored = _relay_utility_pool_predictive(feasible, needer, g, detected, step, weights, '
    'fleet=drones, horizon=horizon)', 1))
EDITS.append((
    '''    _score, best_diag = relay_score(
        best, needer, g, detected, step, weights=weights
    )''',
    '''    _score, best_diag = relay_score(
        best, needer, g, detected, step, weights=weights, fleet=drones
    )''',
    1,
))

# every _coverage_loss_for_candidate(...) call site
EDITS.append((
    '''    coverage_loss, raw_marginal_loss, current_tgap, cov_meta = _coverage_loss_for_candidate(
        candidate, needer, g, detected, step, relay_delay
    )''',
    '''    coverage_loss, raw_marginal_loss, current_tgap, cov_meta = _coverage_loss_for_candidate(
        candidate, needer, g, detected, step, relay_delay, fleet=fleet
    )''',
    1,
))
EDITS.append((
    '''        coverage_loss, raw_marginal_loss, current_tgap, cov_meta = _coverage_loss_for_candidate(
            cand, needer, g, detected, step, relay_delay
        )''',
    '''        coverage_loss, raw_marginal_loss, current_tgap, cov_meta = _coverage_loss_for_candidate(
            cand, needer, g, detected, step, relay_delay, fleet=fleet
        )''',
    4,   # pools C, G, H, I
))
EDITS.append((
    '''        coverage_loss, raw_marginal_loss, current_tgap, cov_meta = _predicted_coverage_loss_for_candidate(
            cand, needer, g, detected, step, relay_delay, horizon=horizon
        )''',
    '''        coverage_loss, raw_marginal_loss, current_tgap, cov_meta = _predicted_coverage_loss_for_candidate(
            cand, needer, g, detected, step, relay_delay, fleet=fleet, horizon=horizon
        )''',
    1,
))
EDITS.append((
    '''        coverage_loss, _raw, _tgap, _meta = _coverage_loss_for_candidate(
            cand, needer, g, detected, step, relay_delay
        )''',
    '''        coverage_loss, _raw, _tgap, _meta = _coverage_loss_for_candidate(
            cand, needer, g, detected, step, relay_delay, fleet=drones
        )''',
    2,   # select_fuzzy_relay + _madm_build_rows (TOPSIS / VIKOR)
))


def apply(src):
    for old, new, count in EDITS:
        found = src.count(old)
        if found != count:
            raise SystemExit(
                f"PATCH ABORTED: expected {count} occurrence(s) of:\n"
                f"----\n{old[:300]}\n----\nbut found {found}. "
                "Your a2e_relay_fixed_final_51.py differs from the version this "
                "patch was written against; apply that hunk by hand."
            )
        src = src.replace(old, new)
    return src


def main():
    if not os.path.exists(TARGET):
        raise SystemExit(f"{TARGET} not found. Run this from your project folder.")
    src = open(TARGET, encoding="utf-8").read()

    if "--check" in sys.argv:
        print("PATCHED" if MARKER in src else "NOT PATCHED")
        return

    if "--restore" in sys.argv:
        if not os.path.exists(BACKUP):
            raise SystemExit(f"No backup at {BACKUP}")
        shutil.copy2(BACKUP, TARGET)
        print(f"restored {TARGET} from {BACKUP}")
        return

    if MARKER in src:
        print("Already patched -- nothing to do.")
        return

    if not os.path.exists(BACKUP):
        shutil.copy2(TARGET, BACKUP)
        print(f"backup written: {BACKUP}")

    out = apply(src)
    compile(out, TARGET, "exec")          # syntax check before writing
    open(TARGET, "w", encoding="utf-8").write(out)
    print(f"patched {TARGET}  ({len(EDITS)} hunks)")
    print("\nNext:")
    print("  rm -rf luts && python build_surrogate_fis.py")
    print("  python reserve_sweep.py            # pick lambda")
    print("  python run_full_comparison.py 500 50")


if __name__ == "__main__":
    main()
