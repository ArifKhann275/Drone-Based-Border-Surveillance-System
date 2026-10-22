# -*- coding: utf-8 -*-
"""
confirmatory_report.py

Evaluates the hypotheses written in PREREGISTRATION_seeds401-600.txt on the
per-seed files of the confirmatory block, with the decision rules fixed there.

Usage (after collecting the data):
    python confirmatory_report.py --wide conf_A.csv --wide conf_B.csv --wide conf_C.csv

H1  coverage of I equivalent (TOST +/-2.0, BH q < 0.05) to C, D, D2, T, V, AUC, B, J
H2  delay AND energy of I significantly lower (BH q < 0.05, negative difference)
    than C, D, D2, T, V, AUC
H3  delay and energy of I NOT significantly higher than J
Also printed (expected limitations, not part of the confirmation): I vs T_tuned,
V_tuned, B (delay), BR (coverage), and I_lite.

Needs compare_focal_vs_all.py in the same folder.
"""

import argparse
import numpy as np

from compare_focal_vs_all import (collect_policies, paired_ci, wilcoxon_p, tost, bh_adjust,
                                  DEFAULT_METRICS)

COV, DLY, ENG = DEFAULT_METRICS
H1_SET = ["C", "D", "D2", "T", "V", "AUC", "B", "J"]
H2_SET = ["C", "D", "D2", "T", "V", "AUC"]
EXPECTED = ["T_tuned", "V_tuned", "B", "BR", "I_lite"]


def paired(data, focal, other, m):
    seeds = sorted(set(data[focal][m]) & set(data[other][m]))
    a = np.array([data[focal][m][s] for s in seeds], float)
    b = np.array([data[other][m][s] for s in seeds], float)
    ok = np.isfinite(a) & np.isfinite(b)
    return a[ok] - b[ok]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wide", action="append", required=True)
    ap.add_argument("--focal", default="I")
    ap.add_argument("--margin", type=float, default=2.0)
    args = ap.parse_args()

    data = collect_policies(args.wide, DEFAULT_METRICS)
    f = args.focal
    need = set(H1_SET) | set(H2_SET) | {f}
    missing = [p for p in sorted(need) if p not in data]
    if missing:
        raise SystemExit(f"Missing policies in the data: {missing}. Present: {sorted(data)}")
    n = len(data[f][COV])
    print(f"Confirmatory report: focal = {f}; seeds in file: {n}; margin = +/-{args.margin:g}\n")

    # ---- H1: coverage equivalence
    p_t, rows1 = [], []
    for o in H1_SET:
        d = paired(data, f, o, COV)
        p, ci90 = tost(d, args.margin)
        mean = d.mean()
        p_t.append(p)
        rows1.append((o, mean, ci90))
    q1 = bh_adjust(p_t)
    print("H1  coverage equivalence (TOST, BH over the 8 tests)")
    h1_pass = True
    for (o, mean, ci90), q in zip(rows1, q1):
        ok = q < 0.05
        h1_pass &= ok
        print(f"   I vs {o:<4} diff {mean:+.2f}  90% CI [{ci90[0]:+.2f},{ci90[1]:+.2f}]  q_TOST {q:.4f}  "
              f"{'PASS' if ok else 'FAIL'}")
    print(f"   => H1 {'CONFIRMED' if h1_pass else 'NOT CONFIRMED'}\n")

    # ---- H2: lower delay and energy than C, D, D2, T, V, AUC
    p2, meta2 = [], []
    for o in H2_SET:
        for m in (DLY, ENG):
            d = paired(data, f, o, m)
            mean, lo, hi = paired_ci(d)
            p2.append(wilcoxon_p(d))
            meta2.append((o, m, mean, lo, hi))
    q2 = bh_adjust(p2)
    print("H2  delay and energy lower than C, D, D2, T, V, AUC (BH over the 12 tests)")
    h2_pass = True
    for (o, m, mean, lo, hi), q in zip(meta2, q2):
        ok = (mean < 0) and (q < 0.05)
        h2_pass &= ok
        print(f"   I vs {o:<4} {m:<24} diff {mean:+.2f} [{lo:+.2f},{hi:+.2f}]  q {q:.4f}  "
              f"{'PASS' if ok else 'FAIL'}")
    print(f"   => H2 {'CONFIRMED' if h2_pass else 'NOT CONFIRMED'}\n")

    # ---- H3: not significantly higher than J
    p3, meta3 = [], []
    for m in (DLY, ENG):
        d = paired(data, f, "J", m)
        mean, lo, hi = paired_ci(d)
        p3.append(wilcoxon_p(d))
        meta3.append((m, mean, lo, hi))
    q3 = bh_adjust(p3)
    print("H3  delay and energy of I not significantly higher than J (BH over 2 tests)")
    h3_pass = True
    for (m, mean, lo, hi), q in zip(meta3, q3):
        ok = not (mean > 0 and q < 0.05)
        h3_pass &= ok
        print(f"   I vs J    {m:<24} diff {mean:+.2f} [{lo:+.2f},{hi:+.2f}]  q {q:.4f}  "
              f"{'PASS' if ok else 'FAIL'}")
    print(f"   => H3 {'CONFIRMED' if h3_pass else 'NOT CONFIRMED'}\n")

    # ---- expected limitations (descriptive)
    print("Expected limitations / exploratory (no pass-fail):")
    for o in EXPECTED:
        if o not in data:
            print(f"   {o}: not in the data")
            continue
        parts = []
        for m, label in ((COV, "cov"), (DLY, "delay"), (ENG, "energy")):
            if m in data[o]:
                d = paired(data, f, o, m)
                mean, lo, hi = paired_ci(d)
                parts.append(f"{label} {mean:+.2f} [{lo:+.2f},{hi:+.2f}]")
        print(f"   I vs {o:<8} " + "   ".join(parts))

    print("\nSUMMARY: H1 {}, H2 {}, H3 {}".format(
        "confirmed" if h1_pass else "NOT confirmed",
        "confirmed" if h2_pass else "NOT confirmed",
        "confirmed" if h3_pass else "NOT confirmed"))
    print("Report every line above in the paper, whatever the outcome (see section 6 of the plan).")


if __name__ == "__main__":
    main()
