# -*- coding: utf-8 -*-
"""
run_theta_sensitivity_full.py
══════════════════════════════════════════════════════════════
Combines two things that were previously done by separate, inconsistent
scripts:

  1. Theta sensitivity: Policy J vs Policy D on coverage_pct,
     avg_relay_delay_steps, avg_relay_energy_pct, at each theta in
     --thetas, on the SAME held-out test-seed block, with Wilcoxon p,
     matched-pairs rank-biserial r, and a paired-t 95% CI on the mean
     difference.

  2. Structural instrumentation (Nc distribution, S-vs-R disagreement,
     Q-ties), computed CORRECTLY -- pooled across every Nc>=2 decision,
     not just Nc=2. This fixes a suspected bug in an earlier
     theta-sweep script whose "S vs R Disagree" column stayed flat at
     0.0% across every theta: that is exactly what you get if the
     check only inspects Nc=2 decisions (where disagreement really is
     0.00%, per run_nc_distribution_check.py's own Nc=2 row), silently
     ignoring the Nc=3..7 decisions where disagreement was actually
     2-6% at theta=0.5 (3.91% pooled). If a corrected run still shows
     0.0% pooled disagreement, that is a real finding; if it instead
     reproduces ~3.9% pooled at theta=0.5, the earlier sweep's
     instrumentation had this Nc-pooling bug.

Both are computed from the SAME simulation runs (one pass per theta),
so the reported disagreement/tie rates and the reported effect sizes
are guaranteed to come from the same underlying decisions -- no risk
of the two halves silently drifting apart across separate scripts.

USAGE
-----
    python run_theta_sensitivity_full.py \\
        --w-energy 0.284 --w-coverage 0.024 --w-delay 0.284 --w-battery 0.408 \\
        --thetas 0.0 0.1 0.25 0.5 0.75 0.9 1.0 \\
        --test-seeds 50

OUTPUT
------
    theta_sensitivity_full.csv -- one row per (theta, metric), with
    j_mean, d_mean, mean_diff, ci_lo, ci_hi, p, r, plus the pooled
    structural instrumentation columns for that theta.
"""
import argparse
import csv
from collections import Counter

import numpy as np
from scipy.stats import wilcoxon, rankdata
from scipy.stats import t as tdist

import a2e_relay_fixed_final_51 as relay
import policy_j_full

METRICS = ("coverage_pct", "avg_relay_delay_steps", "avg_relay_energy_pct")


def rank_biserial(a, b):
    a, b = np.array(a, dtype=float), np.array(b, dtype=float)
    d = a - b
    nz = d[d != 0]
    if len(nz) == 0:
        return 0.0
    ranks = rankdata(np.abs(nz))
    r_pos = ranks[nz > 0].sum()
    r_neg = ranks[nz < 0].sum()
    return (r_pos - r_neg) / (r_pos + r_neg)


def paired_ci95(a, b):
    d = np.array(a, dtype=float) - np.array(b, dtype=float)
    n = len(d)
    mean = d.mean()
    se = d.std(ddof=1) / np.sqrt(n)
    tcrit = tdist.ppf(0.975, n - 1)
    return mean, mean - tcrit * se, mean + tcrit * se


def wilcoxon_p(a, b):
    a, b = np.array(a, dtype=float), np.array(b, dtype=float)
    if np.all(a - b == 0):
        return 1.0
    return wilcoxon(a, b).pvalue


def make_instrumented_j_selector(weights, theta, nc_counts, disagree_by_nc,
                                 ties_by_nc, total_by_nc):
    """Wraps policy_j_full._relay_utility_pool_j directly (not through
    select_best_relay_j) so we get the per-candidate diag dicts needed
    for S-vs-R disagreement, pooled across EVERY Nc>=2 decision."""
    def selector(needer, drones, g, detected, step):
        feasible = relay._feasible_candidate_pool(needer, drones)
        if not feasible:
            return None, []
        nc = len(feasible)
        nc_counts[nc] += 1

        scored = policy_j_full._relay_utility_pool_j(
            feasible, needer, g, detected, step, weights,
            fleet=drones, theta=theta,
        )

        if nc >= 2:
            total_by_nc[nc] += 1
            s_best_id = min(scored, key=lambda x: x[2]["vikor_S"])[2]["candidate_id"]
            r_best_id = min(scored, key=lambda x: x[2]["vikor_R"])[2]["candidate_id"]
            if s_best_id != r_best_id:
                disagree_by_nc[nc] += 1
            q_sorted = sorted(d["vikor_q"] for _, _, d in scored)
            if len(q_sorted) >= 2 and abs(q_sorted[0] - q_sorted[1]) < 1e-9:
                ties_by_nc[nc] += 1

        best_cand = scored[0][0]
        return best_cand, [d for _, _, d in scored]

    return selector


def run_theta_condition_interleaved(weights, theta, seeds, max_steps,
                                    num_drones, num_reserves,
                                    nc_counts, disagree_by_nc, ties_by_nc, total_by_nc):
    """Runs J (instrumented) and D for the SAME seed back-to-back, one
    seed at a time, mirroring policy_j_full.py's own evaluate_heldout()
    call order exactly. This avoids any cross-run RNG-state drift that
    batching 'all seeds of D, then all seeds of J' could introduce if
    the simulator does not fully reseed all randomness at the top of
    simulate_fleet(seed=...). Returns (j_out, d_out)."""
    j_out = {m: [] for m in METRICS}
    d_out = {m: [] for m in METRICS}
    selector = make_instrumented_j_selector(
        weights, theta, nc_counts, disagree_by_nc, ties_by_nc, total_by_nc
    )
    for seed in seeds:
        m_j = relay.simulate_fleet(seed, relay_selector=selector,
                                   num_drones=num_drones, num_reserves=num_reserves,
                                   max_steps=max_steps)
        m_d = relay.simulate_fleet(seed, relay_selector=relay.POLICY_SELECTORS["D"],
                                   num_drones=num_drones, num_reserves=num_reserves,
                                   max_steps=max_steps)
        for k in METRICS:
            j_out[k].append(float(m_j[k]))
            d_out[k].append(float(m_d[k]))
    return j_out, d_out


def run_condition(selector, seeds, max_steps, num_drones, num_reserves):
    out = {m: [] for m in METRICS}
    for seed in seeds:
        m = relay.simulate_fleet(
            seed, relay_selector=selector,
            num_drones=num_drones, num_reserves=num_reserves,
            max_steps=max_steps,
        )
        for k in METRICS:
            out[k].append(float(m[k]))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--w-energy", type=float, required=True)
    p.add_argument("--w-coverage", type=float, required=True)
    p.add_argument("--w-delay", type=float, required=True)
    p.add_argument("--w-battery", type=float, required=True)
    p.add_argument("--thetas", type=float, nargs="+",
                   default=[0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
    p.add_argument("--test-seed-start", type=int, default=1)
    p.add_argument("--test-seeds", type=int, default=50)
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--num-drones", type=int, default=None)
    p.add_argument("--num-reserves", type=int, default=None)
    p.add_argument("--out-csv", default="theta_sensitivity_full.csv")
    args = p.parse_args()

    weights = {
        "W_TRAVEL_ENERGY": args.w_energy,
        "W_COVERAGE_LOSS": args.w_coverage,
        "W_RELAY_DELAY": args.w_delay,
        "W_BATTERY_MARGIN": args.w_battery,
    }
    num_drones = args.num_drones if args.num_drones is not None else relay.NUM_DRONES
    num_reserves = args.num_reserves if args.num_reserves is not None else relay.RESERVE_POOL_SIZE
    seeds = list(range(args.test_seed_start, args.test_seed_start + args.test_seeds))

    print("=" * 90)
    print("THETA SENSITIVITY (vs Policy D) + CORRECTLY-POOLED STRUCTURAL INSTRUMENTATION")
    print("=" * 90)
    print(f"weights={weights}")
    print(f"test seeds {seeds[0]}..{seeds[-1]}, max_steps={args.max_steps}")
    print("D is re-run interleaved with J at every theta (not cached from a "
          "single upfront pass), to avoid any cross-run RNG-state drift.\n")

    rows_out = []
    d_means_seen = []
    for theta in args.thetas:
        print(f"\nRunning Policy J with theta={theta} (instrumented, D interleaved per-seed)...")
        nc_counts = Counter()
        disagree_by_nc = Counter()
        ties_by_nc = Counter()
        total_by_nc = Counter()

        j_out, d_out = run_theta_condition_interleaved(
            weights, theta, seeds, args.max_steps, num_drones, num_reserves,
            nc_counts, disagree_by_nc, ties_by_nc, total_by_nc,
        )
        d_means_seen.append({m: round(float(np.mean(d_out[m])), 4) for m in METRICS})

        total_dec = sum(nc_counts.values())
        n_ge2 = sum(total_by_nc.values())
        dis_total = sum(disagree_by_nc.values())
        tie_total = sum(ties_by_nc.values())
        disagree_pct = 100 * dis_total / n_ge2 if n_ge2 else 0.0
        tie_pct = 100 * tie_total / n_ge2 if n_ge2 else 0.0
        nc2_pct = 100 * nc_counts.get(2, 0) / total_dec if total_dec else 0.0

        print(f"  total_decisions={total_dec}  Nc=2%={nc2_pct:.2f}  "
              f"S-vs-R-disagree% (ALL Nc>=2 pooled)={disagree_pct:.2f}  "
              f"ties%={tie_pct:.2f}")
        print(f"  by-Nc breakdown: " +
              ", ".join(f"Nc={nc}:n={total_by_nc[nc]},dis%={100*disagree_by_nc[nc]/total_by_nc[nc]:.2f}"
                       for nc in sorted(total_by_nc)))

        for metric in METRICS:
            j_vals = j_out[metric]
            d_vals = d_out[metric]
            p_val = wilcoxon_p(j_vals, d_vals)
            r_val = rank_biserial(j_vals, d_vals)
            mean_diff, lo, hi = paired_ci95(j_vals, d_vals)
            rows_out.append({
                "theta": theta,
                "metric": metric,
                "j_mean": round(float(np.mean(j_vals)), 4),
                "d_mean": round(float(np.mean(d_vals)), 4),
                "mean_diff": round(mean_diff, 4),
                "ci_lo": round(lo, 4),
                "ci_hi": round(hi, 4),
                "p": round(p_val, 6),
                "r": round(r_val, 4),
                "total_decisions": total_dec,
                "nc2_pct": round(nc2_pct, 2),
                "disagree_pct_all_nc": round(disagree_pct, 3),
                "tie_pct_all_nc": round(tie_pct, 2),
            })

    with open(args.out_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows_out[0].keys()))
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"\n{'='*90}\nWritten: {args.out_csv}\n{'='*90}")

    print("\nSANITY CHECK 1 -- D-mean stability across theta (should be "
          "near-identical every time, since D does not depend on theta; "
          "if these drift, simulate_fleet(seed=...) is not fully "
          "reseeding all randomness at the top of the call, and results "
          "here should not be trusted until that is fixed):")
    for theta, dm in zip(args.thetas, d_means_seen):
        print(f"  theta={theta}: {dm}")

    KNOWN_HELDOUT_J = {"coverage_pct": 90.57, "avg_relay_delay_steps": 13.46,
                       "avg_relay_energy_pct": 9.88}
    if 0.5 in args.thetas:
        idx = args.thetas.index(0.5)
        print("\nSANITY CHECK 2 -- theta=0.5 J-means vs. the original "
              "tab:heldout report (circularity_check_with_j.csv), same "
              "weights/theta/seeds. These SHOULD match closely; any "
              "material difference means this script's execution order "
              "(or something else) still does not reproduce the original "
              "run, and nothing here should be written into the paper "
              "until that is resolved:")
        for m in METRICS:
            row = [r for r in rows_out if r["theta"] == 0.5 and r["metric"] == m][0]
            known = KNOWN_HELDOUT_J[m]
            print(f"  {m:<24} this run: {row['j_mean']:.3f}   "
                  f"tab:heldout: {known:.3f}   diff: {row['j_mean']-known:+.3f}")

    print("\nSanity check 3: compare the disagree_pct_all_nc column at theta=0.5 "
          "against run_nc_distribution_check.py's earlier pooled figure "
          "(3.91% at the same weights/theta). If they now match, the earlier "
          "theta-sweep script's flat 0.0% was the Nc=2-only pooling bug "
          "described in this script's docstring. If they still disagree, "
          "something else is different (check weights/seeds/max_steps match "
          "exactly) and should be investigated before trusting either number.")


if __name__ == "__main__":
    main()