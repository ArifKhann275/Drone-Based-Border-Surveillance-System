# -*- coding: utf-8 -*-
"""
run_circularity_check_with_j.py
══════════════════════════════════════════════════════════════
Extends the existing rule-base-circularity robustness check
(a2e_relay_fixed_final_51.py's run_policy_d_weight_robustness_check,
which only compared C vs D vs D_indep) to also include I and J.

REVIEWER CONCERN THIS ANSWERS: Policy D's rule base shares its
attribute-priority family with RelayScore's own weights. If J's
"coverage parity with D" finding depends on that shared priority
rather than a genuinely independent fuzzy-MADM philosophy, the paper
should say so. D-indep (equal 0.25/0.25/0.25/0.25 weights) and D-indep2
(a second independently-authored rule base whose gates never reference
coverage_loss) are both NOT tuned on these seeds and NOT related to
RelayScore's own weights -- if J still holds up against them, that is
much stronger evidence than the C-vs-D result alone.

Reuses a2e_relay_fixed_final_51.py's own PRIMARY_METRICS/SECONDARY_METRICS
and _safe_wilcoxon so results are directly comparable to the earlier
dindep_*.csv files' methodology.

USAGE
-----
    python run_circularity_check_with_j.py --weights-json tuned_weights_j_full.json \\
        --seeds 50 --max-steps 500

OUTPUT
------
    circularity_check_with_j.csv       -- per-seed raw data
    circularity_check_with_j_bh.csv    -- BH-corrected significance table
"""
import argparse
import csv
import json
import statistics

import a2e_relay_fixed_final_51 as relay
from policy_j_full import select_best_relay_j
from policy_d_true_fis import (
    select_fuzzy_relay_true_fis_indep_weights,
    select_fuzzy_relay_true_fis_indep2_weights,
)

POLICIES = ("C", "D", "D_indep", "D_indep2", "I", "J")


def bh_correct(pvals):
    indexed = [(i, p) for i, p in enumerate(pvals) if p is not None]
    n = len(indexed)
    if n == 0:
        return [None] * len(pvals)
    indexed.sort(key=lambda x: x[1])
    q = [None] * len(pvals)
    prev_q = 1.0
    for rank in range(n, 0, -1):
        i, p = indexed[rank - 1]
        q_val = min(prev_q, p * n / rank)
        q[i] = q_val
        prev_q = q_val
    return q


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights-json", required=True,
                   help="tuned_weights_j_full.json (Policy J's frozen weights)")
    p.add_argument("--seeds", type=int, default=50)
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--num-drones", type=int, default=None)
    p.add_argument("--num-reserves", type=int, default=None)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--out-csv", default="circularity_check_with_j.csv")
    p.add_argument("--out-bh-csv", default="circularity_check_with_j_bh.csv")
    args = p.parse_args()

    with open(args.weights_json) as fh:
        payload = json.load(fh)
    j_weights = payload["tuned_weights"]
    j_theta = payload.get("theta", 0.5)
    print(f"Loaded Policy J weights: {j_weights}, theta={j_theta}")

    def j_selector(needer, drones, g, detected, step):
        return select_best_relay_j(needer, drones, g, detected, step,
                                   weights=j_weights, theta=j_theta)

    num_drones = args.num_drones if args.num_drones is not None else relay.NUM_DRONES
    num_reserves = args.num_reserves if args.num_reserves is not None else relay.RESERVE_POOL_SIZE
    seeds = list(range(1, args.seeds + 1))

    selectors = {
        "C": relay.select_best_relay,
        "D": relay.POLICY_SELECTORS["D"],
        "D_indep": select_fuzzy_relay_true_fis_indep_weights,
        "D_indep2": select_fuzzy_relay_true_fis_indep2_weights,
        "I": relay.select_best_relay_coverage_aware,
        "J": j_selector,
    }

    print("=" * 78)
    print("RULE-BASE-CIRCULARITY ROBUSTNESS CHECK (extended: + I, J)")
    print("=" * 78)
    print(f"  Policies: {POLICIES}")
    print(f"  seeds=1..{args.seeds}, max_steps={args.max_steps}\n")

    rows = []
    for seed in seeds:
        row = {"seed": seed}
        results = {}
        for label in POLICIES:
            m = relay.simulate_fleet(seed, relay_selector=selectors[label],
                                     num_drones=num_drones, num_reserves=num_reserves,
                                     max_steps=args.max_steps)
            results[label] = m
            for k, v in m.items():
                if k == "decision_log":
                    continue
                row[f"{label}_{k}"] = v
        rows.append(row)
        print(f"[seed {seed}] " + "  |  ".join(
            f"{label}: cov={results[label]['coverage_pct']}%" for label in POLICIES))

    with open(args.out_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWritten raw data: {args.out_csv}")

    # ---- BH-corrected significance: J and I vs each of D, D_indep, D_indep2,
    # plus C vs the same (for direct comparison to the original check) ----
    metrics = relay.PRIMARY_METRICS + relay.SECONDARY_METRICS
    comparisons = [("C", "D"), ("C", "D_indep"), ("C", "D_indep2"),
                   ("I", "D"), ("I", "D_indep"), ("I", "D_indep2"),
                   ("J", "D"), ("J", "D_indep"), ("J", "D_indep2")]

    records = []
    for metric in metrics:
        for a, b in comparisons:
            a_vals = [r.get(f"{a}_{metric}") for r in rows]
            b_vals = [r.get(f"{b}_{metric}") for r in rows]
            a_vals = [v for v in a_vals if isinstance(v, (int, float))]
            b_vals = [v for v in b_vals if isinstance(v, (int, float))]
            if len(a_vals) != len(rows) or len(b_vals) != len(rows):
                continue
            test = relay._safe_wilcoxon(a_vals, b_vals)
            p = (test or {}).get("wilcoxon_p") if isinstance(test, dict) else test
            records.append({
                "metric": metric, "a": a, "b": b,
                "a_mean": round(statistics.mean(a_vals), 4),
                "b_mean": round(statistics.mean(b_vals), 4),
                "p": p,
            })

    pvals = [r["p"] for r in records]
    qvals = bh_correct(pvals)
    for r, q in zip(records, qvals):
        r["q_bh"] = round(q, 6) if q is not None else None
        r["survives_bh"] = (q is not None and q < args.alpha)

    with open(args.out_bh_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["metric", "a", "b", "a_mean", "b_mean",
                                                "p", "q_bh", "survives_bh"])
        writer.writeheader()
        writer.writerows(records)

    print(f"\n{'='*100}\nBH-corrected circularity check (all {len(records)} tests together)\n{'='*100}")
    for r in sorted(records, key=lambda x: (x["metric"] != "coverage_pct", x["p"] if x["p"] is not None else 1)):
        flag = "  <-- SURVIVES BH" if r["survives_bh"] else ""
        print(f"  {r['metric']:<26} {r['a']:>8} vs {r['b']:<10} "
              f"{r['a_mean']:>10} vs {r['b_mean']:<10} p={r['p']:.5f} q_bh={r['q_bh']:.5f}{flag}")

    print(f"\nWritten: {args.out_bh_csv}")
    print(
        "\nInterpretation guide:\n"
        "  - Look at 'coverage_pct  J vs D_indep' and 'J vs D_indep2' rows.\n"
        "  - If J is NOT significantly behind D_indep/D_indep2 (survives_bh=False,\n"
        "    or J_mean >= indep_mean), the coverage-parity finding does NOT depend\n"
        "    on D's rule base sharing a priority family with J/RelayScore -- report\n"
        "    this as a strong robustness result.\n"
        "  - If J IS significantly behind D_indep2 in particular (matching your\n"
        "    earlier finding that C lost to D_indep2 by ~4pp), that means the\n"
        "    circularity concern is NOT fully resolved by J -- state this plainly\n"
        "    as a limitation rather than only comparing J to D throughout the paper."
    )


if __name__ == "__main__":
    main()
