# -*- coding: utf-8 -*-
"""
auction_baseline.py

Marginal-cost auction relay selection ("AUC"), adapted from the market-based /
auction family of multi-robot task allocation (Gerkey & Mataric, "Sold!:
Auction Methods for Multirobot Coordination", IEEE Trans. Robotics and
Automation 18(5):758-768, 2002; taxonomy: Int. J. Robotics Research
23(9):939-954, 2004).

ADAPTATION (state this wording in the paper: "adapted from", not "reproduced"):
  * One relay request = one task, one winner. In that single-task,
    single-robot, instantaneous-assignment setting an auction reduces to
    "the lowest bid wins".
  * Every feasible candidate i computes its own bid from LOCAL information only
        bid_i = travel_energy_i  +  lam * coverage_loss_i
    i.e. the marginal cost of taking the job: the energy it spends getting
    there plus a price for the staleness it causes by abandoning its own zone.
    Raw physical units, NO pool-relative normalization (this is the key
    difference from RelayScore / Policy C).
  * lam is the single price parameter; it is tuned on the TRAINING block only
    (see tune_auction_lambda.py).
  * Candidate pool = exactly the shared feasible pool of B, C, D, J.
  * Ties: first candidate in pool order.

lam = 0 reduces to "cheapest travel energy", which is close to the
nearest-feasible heuristic B (travel energy grows with distance).
"""

import a2e_relay_fixed_final_51 as relay


def select_auction_relay(needer, drones, g, detected, step, weights=None, lam=50.0):
    """Return (winner, [diagnostics]) like the other selectors."""
    feasible = relay._feasible_candidate_pool(needer, drones)
    if not feasible:
        return None, []

    best, best_bid, best_diag = None, float("inf"), None
    for cand, travel_energy, battery_margin in feasible:
        dist = relay._manhattan(cand["r"], cand["c"], needer["r"], needer["c"])
        cov_loss, raw_loss, _tgap, _meta = relay._coverage_loss_for_candidate(
            cand, needer, g, detected, step, dist, fleet=drones
        )
        bid = float(travel_energy) + float(lam) * float(cov_loss)
        if bid < best_bid:  # strict: ties keep the first candidate in pool order
            best, best_bid = cand, bid
            best_diag = {
                "candidate_id": cand["id"],
                "role": cand["role"],
                "distance": dist,
                "travel_energy": round(float(travel_energy), 3),
                "coverage_loss": round(float(cov_loss), 6),
                "relay_delay": dist,
                "bid": round(bid, 4),
                "lam": lam,
            }
    return best, ([best_diag] if best_diag is not None else [])


def make_auction_selector(lam):
    return lambda needer, drones, g, detected, step: select_auction_relay(
        needer, drones, g, detected, step, lam=lam
    )
