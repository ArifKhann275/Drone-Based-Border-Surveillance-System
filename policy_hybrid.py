# -*- coding: utf-8 -*-
"""
policy_hybrid.py
══════════════════════════════════════════════════════════════
Hybrid relay-selection policies built on the two simple heuristics that
bracket Policy J in the held-out results:

    B   nearest feasible candidate      -> lowest delay, lowest coverage
    BR  nearest feasible idle reserve   -> highest coverage, highest delay/energy

Each hybrid has ONE tunable parameter and interpolates between those two
extremes (or between J and BR).  All selectors use exactly the same feasible
pool as every other policy (relay._feasible_candidate_pool), return
(chosen, [diag_of_chosen]), and are STATELESS (no global RNG, no counters),
so they are as safe with respect to the project's call-order sensitivity as
the heuristics in policy_heuristics.py.

FAMILIES (label prefix -> parameter)
------------------------------------
HYBB(tau)   "nearest, but protect thin zones".
            p = nearest feasible candidate.  If p is an idle reserve -> p.
            If p is a patrol drone whose marginal coverage_loss <= tau -> p.
            Otherwise -> nearest feasible idle reserve (p if none exists).
              tau = +inf  ==  B          tau = -1  ==  BR

HYBD(dmax)  "reserve-first, but not from too far away".
            r = nearest feasible idle reserve, p = nearest feasible candidate.
            If r exists and its distance <= dmax -> r, else -> p.
              dmax = +inf ==  BR         dmax = -1 ==  B

HYBJ(tau)   "Policy J, but protect thin zones".
            c = J's choice.  If c is an idle reserve -> c.  If c is a patrol
            drone with coverage_loss <= tau -> c.  Otherwise -> nearest
            feasible idle reserve (c if none exists).
              tau = +inf  ==  J

The ==-identities above are checked on every run by run_hybrid.py
(per-seed equality of the three headline metrics); a mismatch means hidden
state is leaking between runs and the numbers should not be trusted.

RoleRecorder wraps ANY selector and counts how often the chosen candidate was
an idle reserve.  It never changes the choice.  The role is read BEFORE the
simulator re-labels the chosen drone as "relay_incoming".
"""

import a2e_relay_fixed_final_51 as relay

RESERVE = "idle_reserve"
BIG = 1e8          # tau/dmax >= BIG is treated as +infinity


def _dist(cand, needer):
    return relay._manhattan(cand["r"], cand["c"], needer["r"], needer["c"])


def _nearest(feasible, needer, role=None):
    pool = [x for x in feasible if role is None or x[0].get("role") == role]
    if not pool:
        return None
    return min(pool, key=lambda x: (_dist(x[0], needer), x[0]["id"]))[0]


def _score_diag(cand, needer, drones, g, detected, step, weights):
    """(coverage_loss_or_None, diag_dict_or_None) for one feasible candidate;
    relay_score's score is not used for any choice here."""
    _s, diag = relay.relay_score(cand, needer, g, detected, step,
                                 weights=weights, fleet=drones)
    cl = diag.get("coverage_loss") if diag else None
    return cl, diag


def _as_list(diag):
    return [diag] if diag is not None else []


# ── HYBB: nearest feasible, override to nearest reserve if it would thin a zone
def make_hybrid_b(tau):
    def select(needer, drones, g, detected, step, weights=None):
        feasible = relay._feasible_candidate_pool(needer, drones)
        if not feasible:
            return None, []
        p = _nearest(feasible, needer)
        cl, diag_p = _score_diag(p, needer, drones, g, detected, step, weights)
        if p.get("role") == RESERVE or tau >= BIG:
            return p, _as_list(diag_p)
        if cl is not None and cl <= tau:
            return p, _as_list(diag_p)
        r = _nearest(feasible, needer, RESERVE)
        if r is None:
            return p, _as_list(diag_p)
        _cl, diag_r = _score_diag(r, needer, drones, g, detected, step, weights)
        return r, _as_list(diag_r)
    return select


# ── HYBD: reserve-first with a distance (delay) cap
def make_hybrid_dcap(dmax):
    def select(needer, drones, g, detected, step, weights=None):
        feasible = relay._feasible_candidate_pool(needer, drones)
        if not feasible:
            return None, []
        r = _nearest(feasible, needer, RESERVE)
        if r is not None and (dmax >= BIG or _dist(r, needer) <= dmax):
            best = r
        else:
            best = _nearest(feasible, needer)
        _cl, diag = _score_diag(best, needer, drones, g, detected, step, weights)
        return best, _as_list(diag)
    return select


# ── HYBJ: Policy J, override to nearest reserve if J's patrol pick would thin a zone
def make_hybrid_j(j_selector, tau):
    """j_selector must have the plain 5-argument selector signature, i.e. the
    same closure the run_*.py scripts build around select_best_relay_j."""
    def select(needer, drones, g, detected, step, weights=None):
        best, diag = j_selector(needer, drones, g, detected, step)
        if best is None or tau >= BIG or best.get("role") == RESERVE:
            return best, diag
        cl, _d = _score_diag(best, needer, drones, g, detected, step, weights)
        if cl is not None and cl <= tau:
            return best, diag
        feasible = relay._feasible_candidate_pool(needer, drones)
        r = _nearest(feasible, needer, RESERVE)
        if r is None:
            return best, diag
        _cl, diag_r = _score_diag(r, needer, drones, g, detected, step, weights)
        return r, _as_list(diag_r)
    return select


class RoleRecorder:
    """Transparent wrapper: counts decisions and idle-reserve choices."""

    def __init__(self, selector):
        self.selector = selector
        self.n_decisions = 0
        self.n_reserve = 0

    def __call__(self, needer, drones, g, detected, step):
        best, diag = self.selector(needer, drones, g, detected, step)
        if best is not None:
            self.n_decisions += 1
            if best.get("role") == RESERVE:
                self.n_reserve += 1
        return best, diag
