# -*- coding: utf-8 -*-
"""
run_resolution_study.py
══════════════════════════════════════════════════════════════
D2 (surrogate FIS)-কে থিসিসের জন্য আরও শক্ত করার মূল script। একটাই
রিপোর্টে চারটা প্রশ্নের উত্তর দেয়, যেগুলো একটা "convergence study"
appendix-এ থাকা উচিত:

  1. Score-level fidelity: LUT resolution বাড়ালে RMSE/MAE কমে কতটা?
  2. Decision-agreement: candidate-ranking-এ agreement% কত ভালো হয়?
  3. Memory/build-cost: resolution বাড়ালে LUT ফাইলের সাইজ আর বানাতে
     কত সময় লাগে (এইটা trade-off-এর "cost" দিকটা)?
  4. **সবচেয়ে গুরুত্বপূর্ণ** -- Simulation-level validation: ছোট
     resolution দিয়েও কি আসল simulation-এ D-এর মতোই coverage/staleness
     পাওয়া যায়, নাকি একটা ন্যূনতম resolution দরকার? এটাই thesis-এর
     আসল দাবি ("resolution=9-ই যথেষ্ট, তার কম হলে quality নষ্ট হয়")
     সরাসরি প্রমাণ করে -- শুধু abstract RMSE না, বাস্তব সিমুলেশন ফলাফল।

আগে থেকে যা লাগবে (একই ফোল্ডারে):
    a20.py, a2e_relay_fixed_final_51.py, policy_d_true_fis.py,
    surrogate_fis.py, build_surrogate_fis.py, run_resolution_study.py

চালানোর নিয়ম:
    python run_resolution_study.py

    ডিফল্ট resolutions=[5,7,9,11,13], n_score=5000, n_decision=2000,
    simulation-check এ seeds=50, max_steps=500। এটা লম্বা সময় নেবে
    (নিচে "WARNING" অংশ দেখো) -- প্রথমে ছোট করে টেস্ট করতে:

    python run_resolution_study.py --resolutions 5 7 --sim-seeds 5 --sim-max-steps 60

ফলাফল:
    - resolution_study.csv -- প্রতিটা resolution-এর সব মেট্রিক এক জায়গায়
    - টার্মিনালে সরাসরি একটা সারণি + "কোন resolution যথেষ্ট" সুপারিশ

⚠️ WARNING -- এটা ব্যয়বহুল। প্রতিটা resolution R বানাতে R^4টা আসল
skfuzzy FIS কল লাগে (~০.৩ সেকেন্ড/কল)। ডিফল্ট [5,7,9,11,13] একসাথে
বানালে মোট সময় লাগবে মোটামুটি:
    5^4 + 7^4 + 9^4 + 11^4 + 13^4 = 625+2401+6561+14641+28561 ≈ 52,789 কল
    ≈ 52,789 × 0.3s ≈ 4.4 ঘণ্টা (শুধু LUT বানাতেই)
যেই resolution-এর LUT আগে থেকেই luts/ ফোল্ডারে আছে, সেটা আবার বানাবে
না (স্বয়ংক্রিয়ভাবে skip করে) -- তাই তোমার আগের res=9 বা অন্য LUT থাকলে সেটা
পুনরায় বানাবে না, শুধু বাকিগুলো বানাবে।
"""
import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from policy_d_true_fis import FUZZY_ATTR_WEIGHTS
from build_surrogate_fis import build_lut, _LUT_DIR_DEFAULT
from validate_surrogate import score_fidelity, decision_agreement, runtime_benchmark


def _lut_exists(label, resolution, lut_dir):
    return os.path.exists(os.path.join(lut_dir, f"lut_{label}_res{resolution}.npz"))


def simulation_level_check(resolution, label, lut_dir, seeds, max_steps):
    """The strongest evidence: does the FULL simulation (not just
    isolated FIS calls) give the same coverage/staleness as the real
    Mamdani FIS (D) when D2 is pointed at THIS resolution's LUT?
    Returns (mean_coverage_D2, mean_coverage_D, coverage_p,
             mean_staleness_D2, mean_staleness_D, staleness_p)."""
    import a2e_relay_fixed_final_51 as relay
    from surrogate_fis import _INTERP_CACHE, select_fuzzy_relay_surrogate_fis

    _INTERP_CACHE.clear()  # force this resolution to actually be (re)loaded

    def d2_selector(*a, **kw):
        return select_fuzzy_relay_surrogate_fis(*a, lut_dir=lut_dir, resolution=resolution, **kw)

    cov_d2, cov_d, stale_d2, stale_d = [], [], [], []
    for seed in seeds:
        m_d2 = relay.simulate_fleet(seed, relay_selector=d2_selector, max_steps=max_steps)
        m_d = relay.simulate_fleet(seed, relay_selector=relay.select_fuzzy_relay_true_fis, max_steps=max_steps)
        cov_d2.append(m_d2["coverage_pct"])
        cov_d.append(m_d["coverage_pct"])
        stale_d2.append(m_d2["avg_zone_staleness_steps"])
        stale_d.append(m_d["avg_zone_staleness_steps"])

    from scipy.stats import wilcoxon

    def safe_p(a, b):
        a, b = np.array(a, dtype=float), np.array(b, dtype=float)
        if np.allclose(a - b, 0):
            return 1.0
        try:
            return float(wilcoxon(a, b).pvalue)
        except Exception:
            return None

    return {
        "sim_coverage_D2_mean": round(float(np.mean(cov_d2)), 3),
        "sim_coverage_D_mean": round(float(np.mean(cov_d)), 3),
        "sim_coverage_p": safe_p(cov_d2, cov_d),
        "sim_staleness_D2_mean": round(float(np.mean(stale_d2)), 3),
        "sim_staleness_D_mean": round(float(np.mean(stale_d)), 3),
        "sim_staleness_p": safe_p(stale_d2, stale_d),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", choices=["shared", "indep"], default="shared")
    ap.add_argument("--lut-dir", default=_LUT_DIR_DEFAULT)
    ap.add_argument("--resolutions", type=int, nargs="+", default=[5, 7, 9, 11, 13])
    
    # ── UPDATED: Increased sample sizes to remove statistical noise ──
    ap.add_argument("--n-score", type=int, default=5000)
    ap.add_argument("--n-decision", type=int, default=2000)
    ap.add_argument("--k-candidates", type=int, default=6)
    ap.add_argument("--n-runtime", type=int, default=100)
    ap.add_argument("--sim-seeds", type=int, default=50, help="how many seeds for the simulation-level check")
    # ─────────────────────────────────────────────────────────────────
    
    ap.add_argument("--sim-max-steps", type=int, default=500)
    ap.add_argument("--skip-sim-check", action="store_true",
                     help="skip the (expensive) full-simulation check, just do FIS-level fidelity/agreement/runtime")
    ap.add_argument("--out-csv", default="resolution_study.csv")
    args = ap.parse_args()

    weights = FUZZY_ATTR_WEIGHTS if args.weights == "shared" else None
    if weights is None:
        from policy_d_true_fis import INDEPENDENT_RULE_WEIGHTS
        weights = INDEPENDENT_RULE_WEIGHTS
    sim_seeds = list(range(1, args.sim_seeds + 1))

    print(f"=== Resolution sensitivity study (weights='{args.weights}') ===")
    print(f"resolutions to test: {args.resolutions}\n")

    rows = []
    for res in args.resolutions:
        print(f"\n{'='*70}\nRESOLUTION = {res}\n{'='*70}")

        if _lut_exists(args.weights, res, args.lut_dir):
            print(f"  LUT already exists -- skipping build, reusing it.")
            path = os.path.join(args.lut_dir, f"lut_{args.weights}_res{res}.npz")
            build_time = float(np.load(path)["build_time_sec"])
            file_size = os.path.getsize(path)
        else:
            path, build_time, file_size = build_lut(weights, res, args.weights, lut_dir=args.lut_dir)

        print(f"  [1/4] score fidelity...")
        fid = score_fidelity(weights, args.n_score, resolution=res)

        print(f"  [2/4] decision agreement...")
        agree = decision_agreement(weights, args.n_decision, args.k_candidates, resolution=res)

        print(f"  [3/4] runtime benchmark...")
        bench = runtime_benchmark(weights, args.n_runtime, resolution=res)

        row = {
            "resolution": res,
            "build_time_sec": round(build_time, 1),
            "file_size_kb": round(file_size / 1024, 1),
            "rmse": round(fid["rmse"], 4),
            "mae": round(fid["mae"], 4),
            "max_abs_err": round(fid["max_abs_err"], 4),
            "top1_agreement_pct": round(100 * agree["top1_agreement_rate"], 2),
            "mean_kendall_tau": round(agree["mean_kendall_tau"], 4),
            "true_fis_ms_per_call": round(bench["true_fis_ms_per_call"], 3),
            "surrogate_ms_per_call": round(bench["surrogate_ms_per_call"], 4),
            "speedup_x": round(bench["speedup_x"], 1),
        }

        if not args.skip_sim_check:
            print(f"  [4/4] FULL SIMULATION check ({args.sim_seeds} seeds, max_steps={args.sim_max_steps})...")
            sim = simulation_level_check(res, args.weights, args.lut_dir, sim_seeds, args.sim_max_steps)
            row.update(sim)
        else:
            print(f"  [4/4] skipped (--skip-sim-check)")

        rows.append(row)
        print(f"  -> RMSE={row['rmse']}  top1_agreement={row['top1_agreement_pct']}%  "
              f"speedup={row['speedup_x']}x  file_size={row['file_size_kb']}KB")

    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n\n{'='*90}\nSUMMARY TABLE  (saved to {args.out_csv})\n{'='*90}")
    header = list(rows[0].keys())
    print("  ".join(f"{h:>16s}" for h in header))
    for row in rows:
        print("  ".join(f"{str(row.get(h)):>16s}" for h in header))

    if not args.skip_sim_check:
        print("\n--- Recommendation: smallest resolution with sim_coverage_p and sim_staleness_p both > 0.05 ---")
        ok = [r for r in rows if (r.get("sim_coverage_p") or 0) > 0.05 and (r.get("sim_staleness_p") or 0) > 0.05]
        if ok:
            best = min(ok, key=lambda r: r["resolution"])
            print(f"  resolution={best['resolution']}  "
                  f"(file_size={best['file_size_kb']}KB, speedup={best['speedup_x']}x, "
                  f"sim_coverage_p={best['sim_coverage_p']}, sim_staleness_p={best['sim_staleness_p']})")
        else:
            print("  কোনো resolution-ই দুটো p>0.05 শর্ত একসাথে মেটায়নি -- আরও বড় resolution try করো, "
                  "বা --sim-seeds বাড়িয়ে আবার চালাও (sample ছোট হলে noise বেশি থাকে)।")


if __name__ == "__main__":
    main()