# -*- coding: utf-8 -*-
"""
generate_patrol_figures.py
════════════════════════════════════════════════════════════════════════
Contribution 2 (Persistent Patrol Scheduling) — automatic figure +
statistics generator, mirroring generate_relay_figures.py's role for
Contribution 1.

WHY A SEPARATE FILE?
Does NOT modify a21.py's validated run_pps_comparison() logic. It reuses
a21's pure metric functions (coefficient_of_variation, gini_coefficient,
etc. -- so the math is never duplicated/forked) but runs its OWN sweep
over seeds x conditions, because the figures requested here need data
a21.run_pps_comparison() doesn't keep around after computing its scalar
summaries:
    - per-zone age values (for the boxplot/CDF -- a21 only keeps the
      per-seed MAX/MEAN across zones, not the per-zone values themselves)
    - the coverage-over-time trace (for the coverage_curve figure -- new
      instrumentation added to a20.py's DroneSimHeadless.run() alongside
      the existing zone_step_presence/zone_max_age fields)

REQUIRES
    a20.py (with the coverage_curve + PATROL_MODE instrumentation) and
    a21.py in the same folder as this script.

HOW TO RUN
    python generate_patrol_figures.py [max_steps] [num_seeds] [after_condition]

    after_condition is one of SSE_Gap (default -- current production
    default, the "gap term only" condition) or SSE_PPS (the full Mission-
    Urgency formula). This choice only controls which condition is used
    as "after" in the two-panel before/after figures (heatmap, histogram,
    boxplot, CDF) -- the CSV and the coverage-curve/cv_gini_bar figures
    always report all three conditions (SSE_only, SSE_Gap, SSE_PPS).

    Example (thesis-grade):
        python generate_patrol_figures.py 2000 30

OUTPUT (results/persistence/)
    heatmap_before.png       -- zone-visit heatmap, SSE-only
    heatmap_after.png        -- zone-visit heatmap, chosen "after" condition
    zone_visit_histogram.png -- distribution of per-zone visit totals
    zone_age_boxplot.png     -- per-zone mean-age spread, before vs after
    zone_age_cdf.png         -- CDF of the same per-zone age values
    coverage_curve.png       -- mean coverage % over time, all 3 conditions
    cv_gini_bar.png          -- CV & Gini bar chart with paired p-values
    persistence_statistics.csv -- full stats table, all 3 conditions
    persistence_raw.csv      -- per-seed-per-condition raw metrics (appendix)
════════════════════════════════════════════════════════════════════════
"""

import os
import sys
import csv
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import a20            # noqa: shared physics/grid/zone engine + instrumentation
import a21             # noqa: reuse a21's validated pure metric functions

# ══════════════════════════════════════════════════════════════
# TERRAIN-SOURCE SAFETY CHECK (same guard used for Contribution 1 --
# see generate_relay_figures.py for the full explanation)
# ══════════════════════════════════════════════════════════════
EXPECTED_TERRAIN_SOURCES = {"real_srtm", "real_srtm_export"}

def _check_terrain_consistency():
    actual = a20.TERRAIN_SOURCE
    if actual not in EXPECTED_TERRAIN_SOURCES:
        print("=" * 70)
        print(f"⚠️  TERRAIN MISMATCH: expected one of {sorted(EXPECTED_TERRAIN_SOURCES)}, "
              f"got '{actual}'.")
        print("    This run's results will NOT be comparable to any Contribution-1")
        print("    figures (or any earlier Contribution-2 run) made with a different")
        print("    terrain source. Fix: python3 -c \"import a20; a20.prime_srtm_cache()\"")
        print("    (run from THIS folder, with internet), then re-run.")
        print("=" * 70)
    else:
        print(f"✅ Terrain source confirmed: '{actual}' (matches expected set "
              f"{sorted(EXPECTED_TERRAIN_SOURCES)})")


_check_terrain_consistency()

OUT_DIR = os.path.join("results", "persistence")
os.makedirs(OUT_DIR, exist_ok=True)


def _p(filename):
    return os.path.join(OUT_DIR, filename)


CONDITIONS = {"SSE_only": "sse_only", "SSE_Gap": "sse_gap", "SSE_PPS": "sse_pps"}
BEFORE_COND = "SSE_only"
METRIC_KEYS = ("cv", "gini", "max_age", "mean_age", "never_visited",
               "detect_step", "full_cov_step")
_COND_COLORS = {"SSE_only": "#9FB0C3", "SSE_Gap": "#2255EE", "SSE_PPS": "#E67E22"}
_COND_LABELS = {
    "SSE_only": "SSE-only (baseline)",
    "SSE_Gap": "SSE + Gap term",
    "SSE_PPS": "SSE + PPS (Mission Urgency)",
}


# ══════════════════════════════════════════════════════════════
# ONE SWEEP OVER seeds x conditions -- collects everything every
# figure below needs, so each seed/condition is only simulated once.
# ══════════════════════════════════════════════════════════════
def run_full_sweep(seeds, max_steps):
    zones = list(a20.get_all_zones())
    totals_by_cond = {c: {z: 0 for z in zones} for c in CONDITIONS}
    ages_by_cond = {c: {z: [] for z in zones} for c in CONDITIONS}   # per-zone, one value per seed
    coverage_traces = {c: [] for c in CONDITIONS}                     # list of per-seed [(step, pct), ...]
    per_metric = {c: {m: [] for m in METRIC_KEYS} for c in CONDITIONS}
    rows = []

    original_mode = a20.PATROL_MODE
    try:
        for seed in seeds:
            for cname, mode in CONDITIONS.items():
                a20.PATROL_MODE = mode
                sim = a20.DroneSimHeadless(seed)
                result = sim.run(max_steps=max_steps)

                presence = result["zone_step_presence"]
                ages = result["zone_max_age"]

                for z in zones:
                    totals_by_cond[cname][z] += presence.get(z, 0)
                    ages_by_cond[cname][z].append(ages.get(z, 0))

                coverage_traces[cname].append(result["coverage_curve"])

                vals = {
                    "cv": a21.coefficient_of_variation(presence),
                    "gini": a21.gini_coefficient(presence),
                    "max_age": a21.max_zone_age(ages),
                    "mean_age": a21.mean_zone_age(ages),
                    "never_visited": a21.zones_never_visited(presence),
                    "detect_step": result["s"],
                    "full_cov_step": result["s_full_cov"],
                }
                for m in METRIC_KEYS:
                    per_metric[cname][m].append(vals[m])

                rows.append({
                    "seed": seed, "condition": cname,
                    "coverage_pct": round(a20.coverage_pct(sim.gs), 2),
                    **{f"{m}": vals[m] for m in METRIC_KEYS},
                })
    finally:
        a20.PATROL_MODE = original_mode

    return rows, per_metric, totals_by_cond, ages_by_cond, coverage_traces


# ══════════════════════════════════════════════════════════════
# PAIRED STATS (reuses a21's own paired-Wilcoxon convention)
# ══════════════════════════════════════════════════════════════
def paired_stats(a_vals_raw, b_vals_raw):
    pairs = [(a, b) for a, b in zip(a_vals_raw, b_vals_raw) if a is not None and b is not None]
    n = len(pairs)
    a_vals = [p[0] for p in pairs]
    b_vals = [p[1] for p in pairs]
    mean_a = statistics.mean(a_vals) if a_vals else None
    mean_b = statistics.mean(b_vals) if b_vals else None
    if n < 2:
        return {"a_mean": mean_a, "b_mean": mean_b, "n_pairs": n, "wilcoxon_p": None}
    diffs = [b - a for a, b in zip(a_vals, b_vals)]
    if all(abs(d) < 1e-12 for d in diffs):
        p = 1.0
    else:
        try:
            from scipy.stats import wilcoxon
            _, p = wilcoxon(b_vals, a_vals)
        except Exception:
            p = None
    return {"a_mean": mean_a, "b_mean": mean_b, "n_pairs": n, "wilcoxon_p": p}


def _sig_marker(p):
    if p is None:
        return "n/a"
    if p < 0.001:
        return "p < 0.001 ***"
    if p < 0.01:
        return f"p = {p:.4f} **"
    if p < 0.05:
        return f"p = {p:.4f} *"
    return f"p = {p:.4f} (n.s.)"


def _annotate_bracket(ax, x1, x2, y, text):
    h = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.03
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.2, color="black")
    ax.text((x1 + x2) / 2, y + h, text, ha="center", va="bottom", fontsize=9)


# ══════════════════════════════════════════════════════════════
# FIGURE 1/2 — HEATMAPS (before / after)
# ══════════════════════════════════════════════════════════════
def _zone_grid_shape():
    zones = list(a20.get_all_zones())
    max_zr = max(z[0] for z in zones)
    max_zc = max(z[1] for z in zones)
    return max_zr + 1, max_zc + 1


def plot_heatmap(totals, cond_name, filename, vmin=None, vmax=None):
    n_zr, n_zc = _zone_grid_shape()
    grid = np.zeros((n_zr, n_zc))
    for (zr, zc), v in totals.items():
        grid[zr, zc] = v

    fig, ax = plt.subplots(figsize=(6, 5.5))
    im = ax.imshow(grid, cmap="YlOrRd", aspect="equal", vmin=vmin, vmax=vmax)
    for zr in range(n_zr):
        for zc in range(n_zc):
            ax.text(zc, zr, int(grid[zr, zc]), ha="center", va="center",
                     fontsize=7, color="black")
    ax.set_title(f"Zone Visit Heatmap -- {_COND_LABELS[cond_name]}\n"
                  "(total steps physically present, summed over all seeds)",
                  fontsize=11, fontweight="bold")
    ax.set_xlabel("Zone column")
    ax.set_ylabel("Zone row")
    fig.colorbar(im, ax=ax, label="Total visit-steps (all seeds)")
    plt.tight_layout()
    plt.savefig(_p(filename), dpi=300)
    plt.close(fig)
    print(f"✅ Saved {filename}")


# ══════════════════════════════════════════════════════════════
# FIGURE 3 — ZONE VISIT HISTOGRAM (before vs after, overlaid)
# ══════════════════════════════════════════════════════════════
def plot_zone_visit_histogram(totals_before, totals_after, before_label, after_label, filename):
    vals_before = list(totals_before.values())
    vals_after = list(totals_after.values())

    fig, ax = plt.subplots(figsize=(7, 5))
    bins = np.linspace(0, max(vals_before + vals_after) * 1.05, 15)
    ax.hist(vals_before, bins=bins, alpha=0.6, color="#9FB0C3", label=before_label, edgecolor="black")
    ax.hist(vals_after, bins=bins, alpha=0.6, color="#2255EE", label=after_label, edgecolor="black")
    ax.set_xlabel("Total visit-steps per zone (summed over all seeds)")
    ax.set_ylabel("Number of zones")
    ax.set_title("Zone-Visit Distribution: Before vs After\n"
                  "(a tight/right-skewed histogram = starvation; a wide, centered one = uniform coverage)",
                  fontsize=11, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(_p(filename), dpi=300)
    plt.close(fig)
    print(f"✅ Saved {filename}")


# ══════════════════════════════════════════════════════════════
# FIGURE 4 — ZONE AGE BOXPLOT (per-zone mean age across seeds)
# ══════════════════════════════════════════════════════════════
def plot_zone_age_boxplot(ages_before, ages_after, before_label, after_label, filename):
    zone_means_before = [statistics.mean(v) for v in ages_before.values()]
    zone_means_after = [statistics.mean(v) for v in ages_after.values()]

    fig, ax = plt.subplots(figsize=(6, 5.5))
    bp = ax.boxplot([zone_means_before, zone_means_after],
                     tick_labels=[before_label, after_label],
                     patch_artist=True, widths=0.5, showmeans=True)
    for patch, color in zip(bp["boxes"], ["#9FB0C3", "#2255EE"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)

    stat = paired_stats(zone_means_before, zone_means_after)
    y_top = max(zone_means_before + zone_means_after)
    _annotate_bracket(ax, 1, 2, y_top * 1.05, _sig_marker(stat["wilcoxon_p"]))

    ax.set_ylabel("Mean zone age across seeds (steps)")
    ax.set_title("Per-Zone Staleness: Before vs After\n"
                  "(each point = one zone's average worst-case age)",
                  fontsize=11, fontweight="bold")
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(_p(filename), dpi=300)
    plt.close(fig)
    print(f"✅ Saved {filename}  (zone-level paired test: {_sig_marker(stat['wilcoxon_p'])})")


# ══════════════════════════════════════════════════════════════
# FIGURE 5 — ZONE AGE CDF
# ══════════════════════════════════════════════════════════════
def plot_zone_age_cdf(ages_before, ages_after, before_label, after_label, filename):
    zone_means_before = sorted(statistics.mean(v) for v in ages_before.values())
    zone_means_after = sorted(statistics.mean(v) for v in ages_after.values())

    fig, ax = plt.subplots(figsize=(7, 5))
    for vals, label, color in ((zone_means_before, before_label, "#9FB0C3"),
                                (zone_means_after, after_label, "#2255EE")):
        y = np.arange(1, len(vals) + 1) / len(vals)
        ax.step(vals, y, where="post", color=color, lw=2, label=label)

    ax.set_xlabel("Mean zone age (steps)")
    ax.set_ylabel("Cumulative fraction of zones")
    ax.set_title("CDF of Zone Staleness: Before vs After\n"
                  "(a curve further LEFT = zones stay fresher, more often)",
                  fontsize=11, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(_p(filename), dpi=300)
    plt.close(fig)
    print(f"✅ Saved {filename}")


# ══════════════════════════════════════════════════════════════
# FIGURE 6 — COVERAGE CURVE (mean coverage % over time, all 3 conditions)
# ══════════════════════════════════════════════════════════════
def plot_coverage_curve(coverage_traces, max_steps, filename):
    step_grid = np.arange(0, max_steps + 1, a20.COVERAGE_CURVE_SAMPLE_EVERY)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    _last_informative_step = []
    for cname in CONDITIONS:
        traces = coverage_traces[cname]
        # Forward-fill each seed's trace onto the common step grid -- a run
        # can stop early (once all 4 policies have found every threat), so
        # traces have different lengths. After a trace ends we hold its
        # last known value flat (a simplification: the actual policy would
        # keep patrolling and coverage would keep evolving past that point,
        # but we simply have no recorded data there).
        filled = np.zeros((len(traces), len(step_grid)))
        for i, trace in enumerate(traces):
            steps_arr = np.array([t[0] for t in trace]) if trace else np.array([0])
            vals_arr = np.array([t[1] for t in trace]) if trace else np.array([0.0])
            filled[i] = np.interp(step_grid, steps_arr, vals_arr,
                                   left=0.0, right=vals_arr[-1])
        mean_curve = filled.mean(axis=0)
        std_curve = filled.std(axis=0)
        ax.plot(step_grid, mean_curve, color=_COND_COLORS[cname], lw=2, label=_COND_LABELS[cname])
        ax.fill_between(step_grid, mean_curve - std_curve, mean_curve + std_curve,
                         color=_COND_COLORS[cname], alpha=0.15)
        # track the last step where this condition's mean was still < 99.5%,
        # so the x-axis can zoom to the informative region across all conditions
        below_ceiling = np.where(mean_curve < 99.5)[0]
        _last_informative_step.append(step_grid[below_ceiling[-1]] if len(below_ceiling) else step_grid[0])

    zoom_to = min(max(_last_informative_step) * 1.15, max_steps)  # small margin past the last real change
    ax.set_xlim(0, zoom_to)

    ax.set_xlabel("Simulation step")
    ax.set_ylabel("Coverage (%)")
    ax.set_title("Coverage-Over-Time (mean ± std across seeds)", fontsize=12, fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.25)
    ax.set_ylim(0, 105)
    plt.tight_layout()
    plt.savefig(_p(filename), dpi=300)
    plt.close(fig)
    print(f"✅ Saved {filename}")


# ══════════════════════════════════════════════════════════════
# FIGURE 7 — CV & GINI BAR CHART (all 3 conditions, paired p-values vs baseline)
# ══════════════════════════════════════════════════════════════
def plot_cv_gini_bar(per_metric, filename):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
    conds = list(CONDITIONS.keys())

    for ax, metric, title in zip(axes, ("cv", "gini"),
                                  ("CV of zone-visit distribution (↓ better)",
                                   "Gini coefficient of visits (↓ better)")):
        means = [statistics.mean(per_metric[c][metric]) for c in conds]
        colors = [_COND_COLORS[c] for c in conds]
        bars = ax.bar([_COND_LABELS[c] for c in conds], means, color=colors, width=0.55)
        for bar, m in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                     f"{m:.3f}", ha="center", va="bottom", fontsize=9)

        y_top = max(means)
        for i, c in enumerate(conds):
            if c == BEFORE_COND:
                continue
            stat = paired_stats(per_metric[BEFORE_COND][metric], per_metric[c][metric])
            _annotate_bracket(ax, 0, i, y_top * (1.08 + 0.12 * (i - 1)), _sig_marker(stat["wilcoxon_p"]))

        ax.set_title(title, fontsize=10)
        ax.tick_params(axis="x", labelsize=8)
        ax.grid(axis="y", alpha=0.25)

    plt.suptitle("Persistence Metrics: SSE-only vs SSE+Gap vs SSE+PPS", fontweight="bold")
    plt.tight_layout()
    plt.savefig(_p(filename), dpi=300)
    plt.close(fig)
    print(f"✅ Saved {filename}")


# ══════════════════════════════════════════════════════════════
# STATS CSV
# ══════════════════════════════════════════════════════════════
def write_statistics_csv(per_metric, filename):
    rows_out = []
    for cond in CONDITIONS:
        if cond == BEFORE_COND:
            continue
        for metric in METRIC_KEYS:
            stat = paired_stats(per_metric[BEFORE_COND][metric], per_metric[cond][metric])
            rows_out.append({
                "comparison": f"{BEFORE_COND}_vs_{cond}",
                "metric": metric,
                "SSE_only_mean": stat["a_mean"],
                f"{cond}_mean": stat["b_mean"],
                "n_pairs": stat["n_pairs"],
                "wilcoxon_p": stat["wilcoxon_p"],
            })
    # head-to-head: SSE_Gap vs SSE_PPS
    for metric in METRIC_KEYS:
        stat = paired_stats(per_metric["SSE_Gap"][metric], per_metric["SSE_PPS"][metric])
        rows_out.append({
            "comparison": "SSE_Gap_vs_SSE_PPS",
            "metric": metric,
            "SSE_only_mean": None,
            "SSE_Gap_mean": stat["a_mean"],
            "SSE_PPS_mean": stat["b_mean"],
            "n_pairs": stat["n_pairs"],
            "wilcoxon_p": stat["wilcoxon_p"],
        })

    fieldnames = sorted({k for r in rows_out for k in r.keys()})
    with open(_p(filename), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["comparison", "metric"] +
                                 [k for k in fieldnames if k not in ("comparison", "metric")])
        writer.writeheader()
        prov = {k: "" for k in writer.fieldnames}
        prov["comparison"] = "RUN_METADATA"
        prov["metric"] = "terrain_source"
        writer.writerow(prov)
        writer.writerows(rows_out)
    print(f"✅ Saved {filename}  ({len(rows_out)} rows, terrain_source={a20.TERRAIN_SOURCE})")


def write_raw_csv(rows, filename):
    with open(_p(filename), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"✅ Saved {filename}  ({len(rows)} rows)")


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════
def main():
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else a20.MAX_STEPS
    num_seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    after_cond = sys.argv[3] if len(sys.argv) > 3 else "SSE_Gap"
    if after_cond not in CONDITIONS or after_cond == BEFORE_COND:
        print(f"⚠️  Invalid after_condition '{after_cond}', defaulting to 'SSE_Gap'.")
        after_cond = "SSE_Gap"
    seeds = list(range(1, num_seeds + 1))

    print(f"Running with max_steps={steps}, num_seeds={num_seeds}, "
          f"before='{BEFORE_COND}', after='{after_cond}'\n")

    print("=== [1/8] Running full sweep (SSE-only / SSE+Gap / SSE+PPS) ===")
    rows, per_metric, totals_by_cond, ages_by_cond, coverage_traces = run_full_sweep(seeds, steps)
    write_raw_csv(rows, "persistence_raw.csv")

    before_label, after_label = _COND_LABELS[BEFORE_COND], _COND_LABELS[after_cond]

    print("\n=== [2/8] Heatmaps ===")
    _shared_vmin = 0
    _shared_vmax = max(max(totals_by_cond[BEFORE_COND].values()),
                        max(totals_by_cond[after_cond].values()))
    plot_heatmap(totals_by_cond[BEFORE_COND], BEFORE_COND, "heatmap_before.png",
                 vmin=_shared_vmin, vmax=_shared_vmax)
    plot_heatmap(totals_by_cond[after_cond], after_cond, "heatmap_after.png",
                 vmin=_shared_vmin, vmax=_shared_vmax)

    print("\n=== [3/8] Zone-visit histogram ===")
    plot_zone_visit_histogram(totals_by_cond[BEFORE_COND], totals_by_cond[after_cond],
                               before_label, after_label, "zone_visit_histogram.png")

    print("\n=== [4/8] Zone-age boxplot ===")
    plot_zone_age_boxplot(ages_by_cond[BEFORE_COND], ages_by_cond[after_cond],
                           before_label, after_label, "zone_age_boxplot.png")

    print("\n=== [5/8] Zone-age CDF ===")
    plot_zone_age_cdf(ages_by_cond[BEFORE_COND], ages_by_cond[after_cond],
                       before_label, after_label, "zone_age_cdf.png")

    print("\n=== [6/8] Coverage curve ===")
    plot_coverage_curve(coverage_traces, steps, "coverage_curve.png")

    print("\n=== [7/8] CV / Gini bar chart ===")
    plot_cv_gini_bar(per_metric, "cv_gini_bar.png")

    print("\n=== [8/8] Statistics CSV ===")
    write_statistics_csv(per_metric, "persistence_statistics.csv")

    print(f"\n🎉 All Contribution-2 figures saved under: {os.path.abspath(OUT_DIR)}")


if __name__ == "__main__":
    main()
