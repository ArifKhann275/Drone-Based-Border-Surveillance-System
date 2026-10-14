# -*- coding: utf-8 -*-
"""
run_nc_distribution_check.py
══════════════════════════════════════════════════════════════
Checks two things run_theta_sensitivity.py's instrumentation could not,
because it only logged when the feasible pool had exactly 2 candidates:

  1. The FULL distribution of Nc (feasible-pool size) across every relay
     decision -- to verify or correct the paper's claim that "the
     feasible pool almost always contains exactly two candidates."

  2. S-vs-R disagreement (does VIKOR's regret term R ever rank a
     different candidate best than the plain group-utility term S?)
     at EVERY Nc, not just Nc=2 -- because with Nc>2 there are more
     ways for a single worst-attribute candidate to be caught by R
     but not S. If disagreement is still ~0% even including Nc>2
     decisions, Fix 3's compromise mechanism is inert in this system
     as tested, and Section V-C's justification needs to say so
     honestly.

USAGE
-----
    python run_nc_distribution_check.py --test-seeds 50 \\
        --w-energy 0.284 --w-coverage 0.024 --w-delay 0.284 --w-battery 0.408 \\
        --theta 0.5
"""
import argparse
from collections import Counter

import a2e_relay_fixed_final_51 as relay
import policy_j_full

NC_COUNTS = Counter()
DISAGREE_BY_NC = Counter()
TIES_BY_NC = Counter()
TOTAL_BY_NC = Counter()


def instrumented_j_selector(needer, drones, g, detected, step, weights, theta):
    feasible = relay._feasible_candidate_pool(needer, drones)
    if not feasible:
        return None, []

    nc = len(feasible)
    NC_COUNTS[nc] += 1

    scored = policy_j_full._relay_utility_pool_j(
        feasible, needer, g, detected, step, weights, fleet=drones, theta=theta
    )

    if nc >= 2:
        TOTAL_BY_NC[nc] += 1
        s_vals = [(cand, diag["vikor_S"]) for cand, _, diag in scored]
        r_vals = [(cand, diag["vikor_R"]) for cand, _, diag in scored]
        best_by_s = min(s_vals, key=lambda x: x[1])[0]
        best_by_r = min(r_vals, key=lambda x: x[1])[0]
        id_s = best_by_s.get("id") if isinstance(best_by_s, dict) else best_by_s
        id_r = best_by_r.get("id") if isinstance(best_by_r, dict) else best_by_r
        if id_s != id_r:
            DISAGREE_BY_NC[nc] += 1
        q_vals = sorted(diag["vikor_q"] for _, _, diag in scored)
        if len(q_vals) >= 2 and abs(q_vals[0] - q_vals[1]) < 1e-9:
            TIES_BY_NC[nc] += 1

    best_cand = scored[0][0]
    return best_cand, [d for _, _, d in scored]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--w-energy", type=float, default=0.284)
    p.add_argument("--w-coverage", type=float, default=0.024)
    p.add_argument("--w-delay", type=float, default=0.284)
    p.add_argument("--w-battery", type=float, default=0.408)
    p.add_argument("--theta", type=float, default=0.5)
    p.add_argument("--test-seeds", type=int, default=50)
    p.add_argument("--max-steps", type=int, default=500)
    args = p.parse_args()

    weights = {
        "W_TRAVEL_ENERGY": args.w_energy,
        "W_COVERAGE_LOSS": args.w_coverage,
        "W_RELAY_DELAY": args.w_delay,
        "W_BATTERY_MARGIN": args.w_battery,
    }
    seeds = list(range(1, args.test_seeds + 1))

    def j_selector(needer, drones, g, detected, step):
        return instrumented_j_selector(needer, drones, g, detected, step, weights, args.theta)

    print(f"Running {args.test_seeds} seeds, theta={args.theta}, weights={weights}...")
    for seed in seeds:
        relay.simulate_fleet(seed, relay_selector=j_selector, max_steps=args.max_steps)

    total_decisions = sum(NC_COUNTS.values())
    print("\n" + "=" * 78)
    print("FULL Nc DISTRIBUTION (every relay decision, all seeds)")
    print("=" * 78)
    print(f"{'Nc':<6}{'count':<10}{'% of all decisions':<20}")
    for nc in sorted(NC_COUNTS):
        cnt = NC_COUNTS[nc]
        print(f"{nc:<6}{cnt:<10}{100*cnt/total_decisions:<20.2f}")
    print(f"\nTotal relay decisions: {total_decisions}")

    print("\n" + "=" * 78)
    print("S-vs-R DISAGREEMENT AND Q-TIES, BY Nc (Nc>=2 only; Nc=1 has no ranking)")
    print("=" * 78)
    print(f"{'Nc':<6}{'n (Nc>=2 decisions)':<22}{'S-vs-R disagree %':<20}{'Q-ties %':<12}")
    for nc in sorted(TOTAL_BY_NC):
        tot = TOTAL_BY_NC[nc]
        dis = DISAGREE_BY_NC[nc]
        tie = TIES_BY_NC[nc]
        print(f"{nc:<6}{tot:<22}{100*dis/tot:<20.2f}{100*tie/tot:<12.2f}")

    n_ge2 = sum(TOTAL_BY_NC.values())
    dis_total = sum(DISAGREE_BY_NC.values())
    pct = (100 * dis_total / n_ge2) if n_ge2 else 0
    print(f"\nOverall (all Nc>=2 decisions pooled): "
          f"{dis_total}/{n_ge2} disagree ({pct:.2f}%)")

    print(
        "\nInterpretation guide:\n"
        "  - If Nc=2 is far below 'almost always' (e.g. under ~50%), the\n"
        "    paper's Section IV diagnosis text needs to say what the actual\n"
        "    modal Nc is instead (check the Nc=1 row -- likely dominant if so).\n"
        "  - If S-vs-R disagreement stays near 0% even pooling all Nc>=2,\n"
        "    Fix 3 (VIKOR-style compromise ranking) is not actually changing\n"
        "    any decisions relative to a plain weighted-sum on S alone in\n"
        "    this system as tested -- Section V-C's contribution claim needs\n"
        "    to be revised to say so honestly, and J's advantage should be\n"
        "    attributed to Fixes 1+2 (absolute-scale normalization) rather\n"
        "    than the compromise-ranking mechanism itself.\n"
        "  - If disagreement is meaningfully >0% once Nc>2 cases are\n"
        "    included (even if it was 0% for Nc=2 alone), that's an\n"
        "    important correction to report: Fix 3 activates specifically\n"
        "    when more than 2 candidates compete, not in the 2-candidate\n"
        "    case the paper currently frames as central."
    )


if __name__ == "__main__":
    main()
