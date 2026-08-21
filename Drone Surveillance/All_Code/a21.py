# -*- coding: utf-8 -*-


import csv
import json
import statistics

import a20  

from scipy.stats import wilcoxon


_PLOTTING_AVAILABLE = False
_HAS_GUI = False
try:
    import matplotlib
    import matplotlib.pyplot as plt
    for _backend in ("QtAgg", "TkAgg", "Qt5Agg", "MacOSX"):
        try:
            matplotlib.use(_backend, force=True)
            _test_fig = plt.figure()
            plt.close(_test_fig)
            _HAS_GUI = True
            break
        except Exception:
            continue
    if not _HAS_GUI:
        matplotlib.use("Agg", force=True)
        print("⚠️  No interactive GUI backend found -- charts will be saved "
              "to disk as PNG but won't pop up in a window.")
    _PLOTTING_AVAILABLE = True
except ImportError:
    print("Warning: matplotlib not installed. Plotting will be skipped.")


# ══════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════
CONDITIONS = {
    "SSE_only": "sse_only",
    "SSE_Gap":  "sse_gap",   
    "SSE_PPS":  "sse_pps",   
}
BASELINE = "SSE_only"
HEAD_TO_HEAD = ("SSE_Gap", "SSE_PPS")

METRIC_KEYS = ("cv", "gini", "max_age", "mean_age", "never_visited", "detect_step", "full_cov_step")
METRIC_LABELS = {
    "cv": "CV of zone-visit distribution (↓ = more uniform)",
    "gini": "Gini coefficient of visits (↓ = less inequality)",
    "max_age": "Max zone age, steps (↓ = less worst-case staleness)",
    "mean_age": "Mean zone age, steps (↓ = less staleness overall)",
    "never_visited": "# zones never visited (↓ better)",
    "detect_step": "Step all threats detected (↓ better; efficiency trade-off check)",
    "full_cov_step": "Step 100% coverage reached (↓ better; efficiency trade-off check)",
}

# ══════════════════════════════════════════════════════════════
# METRICS FUNCTIONS
# ══════════════════════════════════════════════════════════════
def _all_zone_keys():
    return list(a20.get_all_zones())

def coefficient_of_variation(visit_counts_by_zone):
    zones = _all_zone_keys()
    vals = [visit_counts_by_zone.get(z, 0) for z in zones]
    mean = statistics.mean(vals)
    if mean == 0:
        return 0.0
    std = statistics.pstdev(vals)
    return std / mean

def gini_coefficient(visit_counts_by_zone):
    zones = _all_zone_keys()
    vals = sorted(visit_counts_by_zone.get(z, 0) for z in zones)
    n = len(vals)
    total = sum(vals)
    if n == 0 or total == 0:
        return 0.0
    numerator = 2 * sum((i + 1) * v for i, v in enumerate(vals))
    return (numerator / (n * total)) - (n + 1) / n

def max_zone_age(zone_max_age):
    zones = _all_zone_keys()
    vals = [zone_max_age.get(z, 0) for z in zones]
    return max(vals) if vals else 0

def mean_zone_age(zone_max_age):
    zones = _all_zone_keys()
    vals = [zone_max_age.get(z, 0) for z in zones]
    return statistics.mean(vals) if vals else 0.0

def zones_never_visited(visit_counts_by_zone):
    zones = _all_zone_keys()
    return sum(1 for z in zones if visit_counts_by_zone.get(z, 0) == 0)

# ══════════════════════════════════════════════════════════════
# EXPERIMENT RUNNER
# ══════════════════════════════════════════════════════════════
def run_pps_comparison(seeds, max_steps=None, out_csv="pps_full_evaluation.csv"):
    if max_steps is None:
        max_steps = a20.MAX_STEPS

    rows = []
    totals_by_cond = {cond: {} for cond in CONDITIONS}
    per_metric = {cond: {m: [] for m in METRIC_KEYS} for cond in CONDITIONS}

    original_mode = a20.PATROL_MODE
    try:
        for seed in seeds:
            for cond_name, mode_value in CONDITIONS.items():
                a20.PATROL_MODE = mode_value
                sim = a20.DroneSimHeadless(seed)
                result = sim.run(max_steps=max_steps)

                presence = result["zone_step_presence"]
                ages = result["zone_max_age"]

                vals = {
                    "cv": coefficient_of_variation(presence),
                    "gini": gini_coefficient(presence),
                    "max_age": max_zone_age(ages),
                    "mean_age": mean_zone_age(ages),
                    "never_visited": zones_never_visited(presence),
                    "detect_step": result["s"],
                    "full_cov_step": result["s_full_cov"],
                }
                for m in METRIC_KEYS:
                    per_metric[cond_name][m].append(vals[m])

                for z, v in presence.items():
                    totals_by_cond[cond_name][z] = totals_by_cond[cond_name].get(z, 0) + v

                rows.append({
                    "seed": seed,
                    "condition": cond_name,
                    "coverage_pct": round(a20.coverage_pct(sim.gs), 2),
                    "detect_all_threats_step": vals["detect_step"],
                    "full_coverage_step": vals["full_cov_step"],
                    "cv_visit_distribution": round(vals["cv"], 4),
                    "gini_visit_distribution": round(vals["gini"], 4),
                    "max_zone_age_steps": vals["max_age"],
                    "mean_zone_age_steps": round(vals["mean_age"], 2),
                    "zones_never_visited": vals["never_visited"],
                })
    finally:
        a20.PATROL_MODE = original_mode

    if out_csv:
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"💾 Saved per-seed results to {out_csv}")

    summary = {
        "vs_baseline": {
            cond: _paired_stats(per_metric[BASELINE], per_metric[cond])
            for cond in CONDITIONS if cond != BASELINE
        },
        "head_to_head": _paired_stats(per_metric[HEAD_TO_HEAD[0]], per_metric[HEAD_TO_HEAD[1]]),
    }
    return rows, per_metric, summary, totals_by_cond

def _paired_stats(metric_dict_a, metric_dict_b):
    out = {}
    for metric in METRIC_KEYS:
        a_raw = metric_dict_a[metric]
        b_raw = metric_dict_b[metric]
        pairs = [(a, b) for a, b in zip(a_raw, b_raw) if a is not None and b is not None]
        n_usable = len(pairs)
        a_vals = [p[0] for p in pairs]
        b_vals = [p[1] for p in pairs]
        mean_a = statistics.mean(a_vals) if a_vals else None
        mean_b = statistics.mean(b_vals) if b_vals else None

        if n_usable < 2:
            out[metric] = {"a_mean": mean_a, "b_mean": mean_b, "n_pairs": n_usable, "wilcoxon_p": None}
            continue

        diffs = [b - a for a, b in zip(a_vals, b_vals)]
        zero_diff = all(abs(d) < 1e-12 for d in diffs)
        if zero_diff:
            p = 1.0
        else:
            try:
                _, p = wilcoxon(b_vals, a_vals)
            except Exception:
                p = None

        out[metric] = {"a_mean": mean_a, "b_mean": mean_b, "n_pairs": n_usable, "wilcoxon_p": p}
    return out

def print_summary(summary):
    print("\n" + "=" * 82)
    print("PPS FULL COMPARISON  (SSE-only  vs  SSE+Gap  vs  SSE+PPS [Mission Urgency])")
    print("=" * 82)

    for cond in CONDITIONS:
        if cond == BASELINE:
            continue
        print(f"\n--- {BASELINE} (baseline)  vs  {cond} ---")
        stats = summary["vs_baseline"][cond]
        for metric in METRIC_KEYS:
            s = stats[metric]
            p = s["wilcoxon_p"]
            p_str = f"{p:.4g}" if p is not None else "N/A"
            sig = "✅" if (p is not None and p < 0.05) else "❌"
            a_str = f"{s['a_mean']:.4f}" if s["a_mean"] is not None else "N/A"
            b_str = f"{s['b_mean']:.4f}" if s["b_mean"] is not None else "N/A"
            print(f"  {METRIC_LABELS[metric]}")
            print(f"      {BASELINE}: {a_str}   {cond}: {b_str}   "
                  f"p={p_str} {sig}  (n={s['n_pairs']})")

    print(f"\n--- HEAD-TO-HEAD: {HEAD_TO_HEAD[0]}  vs  {HEAD_TO_HEAD[1]}  "
          f"(Is the full PPS formula significantly better than the simple gap term?) ---")
    stats = summary["head_to_head"]
    for metric in METRIC_KEYS:
        s = stats[metric]
        p = s["wilcoxon_p"]
        p_str = f"{p:.4g}" if p is not None else "N/A"
        sig = "✅" if (p is not None and p < 0.05) else "❌"
        a_str = f"{s['a_mean']:.4f}" if s["a_mean"] is not None else "N/A"
        b_str = f"{s['b_mean']:.4f}" if s["b_mean"] is not None else "N/A"
        print(f"  {METRIC_LABELS[metric]}")
        print(f"      {HEAD_TO_HEAD[0]}: {a_str}   {HEAD_TO_HEAD[1]}: {b_str}   "
              f"p={p_str} {sig}  (n={s['n_pairs']})")
    print("=" * 82)

# ══════════════════════════════════════════════════════════════
# MOTIVATION / RESULT FIGURE — 3-panel zone-visit distribution
# ══════════════════════════════════════════════════════════════
_COND_COLORS = {"SSE_only": "#9FB0C3", "SSE_Gap": "#2255EE", "SSE_PPS": "#E67E22"}
_COND_LABELS = {
    "SSE_only": "SSE-only (baseline)",
    "SSE_Gap": "SSE + Gap term (Ablation)",
    "SSE_PPS": "SSE + PPS: Mission Urgency = Threat × Age (Proposed)",
}

def plot_zone_visit_distribution(totals_by_cond, save_path="pps_full_zone_distribution.png", show=True):
    if not _PLOTTING_AVAILABLE:
        print("❌ Cannot plot: matplotlib is missing.")
        return

    zones = _all_zone_keys()
    baseline_totals = totals_by_cond[BASELINE]
    zones_sorted = sorted(zones, key=lambda z: -baseline_totals.get(z, 0))
    x_labels = [f"{z}" for z in zones_sorted]

    conds = list(CONDITIONS.keys())
    fig, axes = plt.subplots(len(conds), 1, figsize=(max(14, len(zones) * 0.35), 14), sharex=True)
    fig.suptitle(
        "Zone-Visit Distribution: SSE-only vs SSE+Gap vs SSE+PPS (Mission Urgency)\n"
        "Summed over all seeds, zones sorted by SSE-only visit count",
        fontsize=14, fontweight="bold"
    )

    for ax, cond in zip(axes, conds):
        vals = [totals_by_cond[cond].get(z, 0) for z in zones_sorted]
        ax.bar(range(len(zones_sorted)), vals, color=_COND_COLORS[cond], width=0.8, zorder=3)
        ax.set_title(_COND_LABELS[cond], fontsize=11, fontweight="bold")
        ax.set_ylabel("Total visit-steps")
        ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[-1].set_xticks(range(len(zones_sorted)))
    axes[-1].set_xticklabels(x_labels, rotation=90, fontsize=6)
    axes[-1].set_xlabel("Zone (sorted by SSE-only visit count, descending)")

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"📊 Saved zone-visit-distribution chart to {save_path}")

    if show and _HAS_GUI:
        plt.show()
    else:
        if show and not _HAS_GUI:
            print("ℹ️  Skipping popup window (no GUI backend available) -- see the saved PNG instead.")
        plt.close(fig)

# ══════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import sys

    # Usage: python a21.py [max_steps] [num_seeds] [urgency_weight]
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else a20.MAX_STEPS
    num_seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    if len(sys.argv) > 3:
        a20.HY_URGENCY_W = float(sys.argv[3])
    seeds = list(range(1, num_seeds + 1))

    print(f"\n=== PPS FULL COMPARISON: SSE-only vs SSE+Gap vs SSE+PPS "
          f"({num_seeds} seeds, {steps} steps, HY_URGENCY_W={a20.HY_URGENCY_W}) ===")
    rows, per_metric, summary, totals_by_cond = run_pps_comparison(
        seeds, max_steps=steps, out_csv="pps_full_evaluation.csv"
    )
    print_summary(summary)
    print("\n" + json.dumps(summary, indent=2, default=str))

    print("\n=== GENERATING ZONE-VISIT DISTRIBUTION CHART ===")
    plot_zone_visit_distribution(totals_by_cond, save_path="pps_full_zone_distribution.png")
