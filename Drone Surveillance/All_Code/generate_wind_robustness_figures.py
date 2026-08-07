# -*- coding: utf-8 -*-
"""
generate_wind_robustness_figures.py
════════════════════════════════════════════════════════════════════════
Answers the reviewer comment:
  "বাতাসের গতি (WIND_SPEED) এবং দিক স্থির ধরা হয়েছে... সময়ের সাথে সাথে
   পরিবর্তনশীল আবহাওয়া যুক্ত করলে এটি আরও বাস্তবসম্মত হবে।"

This script does four things:
  1. Plots the new time-varying wind field itself (speed + direction over
     one full run) -- a direct, visual answer to the reviewer comment.
  2. Runs the SAME four-way ablation (A: single drone+nearest,
     B: 8-drone fleet+nearest, C: 8-drone fleet+RelayScore (proposed),
     D: 8-drone fleet+fuzzy MADM, adapted from Zhu, Zhou & Zhang 2017)
     TWICE -- once under the old static wind, once under the new dynamic
     wind -- so the robustness check now includes the fuzzy-MADM baseline,
     not just the plain-nearest baseline.
  3. Produces a dedicated "RelayScore vs fuzzy MADM" figure (C vs D,
     paired boxplot + Wilcoxon p) under BOTH wind conditions side by
     side. This is the comparison that matters most for the paper's
     core claim: RelayScore's utility-function ranking should still
     beat the fuzzy-MADM ranking rule even once the wind stops being a
     conveniently fixed constant.
  4. If step 3 shows a residual gap under dynamic wind, runs the five-way
     comparison (adds Policy E = RelayScore + dynamic importance
     weighting, see a2e_relay_fixed_final_51.py's select_best_relay_dynamic)
     under dynamic wind ONLY, and plots C vs D vs E to check whether the
     dynamic-weighting layer closes that residual gap.

REQUIRES
    a20.py and a2e_relay_fixed_final_51.py (WITH the dynamic-wind patch
    AND the Policy E dynamic-weighting layer already applied) in the
    same folder as this script.

HOW TO RUN
    python generate_wind_robustness_figures.py [max_steps] [num_seeds]

OUTPUT (results/relay/)
    wind_profile.png                    -- the dynamic wind field itself
    ablation_static_wind.png            -- A/B/C/D bars under static wind
    ablation_dynamic_wind.png           -- A/B/C/D bars under dynamic wind
    relayscore_vs_fuzzy_madm_robustness.png
                                         -- C vs D boxplots, static | dynamic
    cde_dynamic_gap.png                 -- C vs D vs E bars, dynamic wind only
                                            (does the dynamic layer close the gap?)
    wind_robustness_statistics.csv      -- full numbers behind the A/B/C/D figures
    abcd_static_raw.csv / abcd_dynamic_raw.csv
                                         -- per-seed raw data (A/B/C/D), for appendix
    abcde_dynamic_raw.csv               -- per-seed raw data (A/B/C/D/E, dynamic wind)
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

import a2e_relay_fixed_final_51 as relay
import a20

# ══════════════════════════════════════════════════════════════
# TERRAIN-SOURCE SAFETY CHECK (same guard as generate_relay_figures.py --
# see that file for the full explanation)
# ══════════════════════════════════════════════════════════════
EXPECTED_TERRAIN_SOURCES = {"real_srtm", "real_srtm_export"}  # both are genuine
        # real terrain -- see generate_relay_figures.py for the full note.

def _check_terrain_consistency():
    actual = a20.TERRAIN_SOURCE
    if actual not in EXPECTED_TERRAIN_SOURCES:
        print("=" * 70)
        print(f"⚠️  TERRAIN MISMATCH: expected one of {EXPECTED_TERRAIN_SOURCES}, "
              f"got '{actual}'.")
        print("    Fix: python3 -c \"import a20; a20.prime_srtm_cache()\" "
              "(run from THIS folder, with internet), then re-run.")
        print("=" * 70)
    else:
        print(f"✅ Terrain source confirmed: '{actual}' (real terrain, "
              f"matches expected set: {EXPECTED_TERRAIN_SOURCES})")


_check_terrain_consistency()

OUT_DIR = os.path.join("results", "relay")
os.makedirs(OUT_DIR, exist_ok=True)


def _p(filename):
    return os.path.join(OUT_DIR, filename)


# ══════════════════════════════════════════════════════════════
# SIGNIFICANCE ANNOTATION HELPERS (same convention as generate_relay_figures.py,
# duplicated here so this script has no import-order dependency on that file)
# ══════════════════════════════════════════════════════════════
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


def _annotate_bracket(ax, x1, x2, y, text, fontsize=9):
    h = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.03
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.2, color="black")
    ax.text((x1 + x2) / 2, y + h, text, ha="center", va="bottom", fontsize=fontsize)


# ══════════════════════════════════════════════════════════════
# 1) WIND PROFILE FIGURE — direct visual answer to the reviewer
# ══════════════════════════════════════════════════════════════
def plot_wind_profile(max_steps, filename="wind_profile.png"):
    steps = np.arange(0, max_steps)

    # Temporarily force dynamic mode ON just to compute the profile values,
    # then restore whatever the caller had set -- this function should
    # never have a side effect on the rest of the run.
    prev = a20.WIND_DYNAMIC_ENABLED
    a20.WIND_DYNAMIC_ENABLED = True
    speeds, dirs = zip(*(a20.get_current_wind(int(s)) for s in steps))
    a20.WIND_DYNAMIC_ENABLED = prev

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)

    ax1.plot(steps, speeds, color="#2b8cbe", lw=1.5)
    ax1.axhline(a20.WIND_BASE_SPEED, color="gray", ls="--", lw=1,
                label=f"Old static value ({a20.WIND_BASE_SPEED} m/s)")
    ax1.set_ylabel("Wind speed (m/s)")
    ax1.set_title("Dynamic Wind Field Used in the Simulation", fontweight="bold")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(alpha=0.25)

    ax2.plot(steps, dirs, color="#fdae61", lw=1.5)
    ax2.axhline(a20.WIND_BASE_DIR, color="gray", ls="--", lw=1,
                label=f"Old static value ({a20.WIND_BASE_DIR}°)")
    ax2.set_ylabel("Wind direction (°)")
    ax2.set_xlabel("Simulation step")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(alpha=0.25)

    plt.tight_layout()
    plt.savefig(_p(filename), dpi=300)
    plt.close(fig)
    print(f"✅ Saved {filename}")


# ══════════════════════════════════════════════════════════════
# 2) FOUR-WAY (A/B/C/D, D = fuzzy MADM) RUN UNDER ONE WIND CONDITION
# ══════════════════════════════════════════════════════════════
def _run_abcd_under_condition(seeds, max_steps, dynamic, out_csv):
    a20.WIND_DYNAMIC_ENABLED = dynamic
    rows, summary = relay.run_four_way_comparison(
        seeds, max_steps=max_steps, out_csv=out_csv
    )
    a20.WIND_DYNAMIC_ENABLED = False  # always leave the module in the safe default state
    return rows, summary


def _run_abcde_dynamic(seeds, max_steps, out_csv):
    """Five-way (A/B/C/D/E) comparison, DYNAMIC wind only -- this is the
    condition where run_four_way_comparison showed C had not clearly
    closed the gap with D yet (see the session notes this script answers).
    E = Policy C + the dynamic-weighting layer (select_best_relay_dynamic);
    isolates whether that layer closes the remaining gap. Not run under
    static wind too -- E's dynamic-weighting mechanism only has something
    to react to (coverage/battery pressure spikes) under the harder,
    non-stationary dynamic-wind condition; under static wind C already
    has no significant gap with D (see ablation_static_wind.png), so a
    static A/B/C/D/E run would just reproduce that with an extra column."""
    a20.WIND_DYNAMIC_ENABLED = True
    rows, summary = relay.run_five_way_comparison(
        seeds, max_steps=max_steps, out_csv=out_csv
    )
    a20.WIND_DYNAMIC_ENABLED = False  # always leave the module in the safe default state
    return rows, summary


# ══════════════════════════════════════════════════════════════
# 3) ABLATION BAR CHART (A/B/C/D) FOR ONE WIND CONDITION
#    (same visual convention as generate_relay_figures.py's ablation_bar,
#    reused per-condition so static and dynamic figures line up 1:1)
# ══════════════════════════════════════════════════════════════
def ablation_bar_one_condition(summary, metric, condition_label, filename):
    entry = summary.get(metric)
    if not entry:
        print(f"⚠️  Skipping {filename}: metric '{metric}' not in summary.")
        return

    labels = ["A\n(Single drone\n+ nearest)",
              "B\n(8-drone fleet\n+ nearest)",
              "C\n(8-drone fleet\n+ RelayScore)",
              "D\n(8-drone fleet\n+ fuzzy MADM)"]
    means = [entry.get(f"{l}_mean") for l in ("A", "B", "C", "D")]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    bars = ax.bar(labels, means, color=["#bdbdbd", "#a6bddb", "#2b8cbe", "#fdae61"], width=0.55)
    for bar, m in zip(bars, means):
        if m is not None:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                     f"{m:.1f}", ha="center", va="bottom", fontsize=10)

    valid_means = [v for v in means if v is not None]
    if not valid_means:
        plt.close(fig)
        print(f"⚠️  Skipping {filename}: no numeric means for metric '{metric}'.")
        return
    y_top = max(valid_means)

    relay_p = entry.get("relay_intel_effect_B_vs_C", {}).get("wilcoxon_p")
    fuzzy_p = entry.get("relay_intel_effect_B_vs_D", {}).get("wilcoxon_p")
    rank_p = entry.get("ranking_rule_effect_C_vs_D", {}).get("wilcoxon_p")
    _annotate_bracket(ax, 1, 2, y_top * 1.03, f"RelayScore vs nearest\n{_sig_marker(relay_p)}")
    _annotate_bracket(ax, 1, 3, y_top * 1.18, f"fuzzy-MADM vs nearest\n{_sig_marker(fuzzy_p)}")
    _annotate_bracket(ax, 2, 3, y_top * 1.33, f"RelayScore vs fuzzy-MADM\n{_sig_marker(rank_p)}")

    ax.set_ylabel(metric)
    ax.set_title(f"Ablation ({condition_label}): {metric} across A / B / C / D",
                  fontsize=12, fontweight="bold")
    ax.set_ylim(top=y_top * 1.5)
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(_p(filename), dpi=300)
    plt.close(fig)
    print(f"✅ Saved {filename}  ({condition_label} A/B/C/D means: {means})")


# ══════════════════════════════════════════════════════════════
# 4) HEADLINE ROBUSTNESS FIGURE: RelayScore (C) vs fuzzy MADM (D),
#    static wind vs dynamic wind, side by side, paired boxplot + p-value.
#    This is the figure that directly answers "does RelayScore's edge
#    over fuzzy MADM survive non-stationary weather?"
# ══════════════════════════════════════════════════════════════
def plot_relayscore_vs_fuzzy_robustness(static_rows, dynamic_rows, metric,
                                         ylabel, filename):
    conditions = [("Static wind", static_rows), ("Dynamic wind", dynamic_rows)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5), sharey=True)

    import random as _r

    for ax, (cond_label, rows) in zip(axes, conditions):
        c_vals = [r.get(f"C_{metric}") for r in rows]
        d_vals = [r.get(f"D_{metric}") for r in rows]
        c_clean = [v for v in c_vals if isinstance(v, (int, float))]
        d_clean = [v for v in d_vals if isinstance(v, (int, float))]

        if not c_clean or not d_clean:
            ax.set_title(f"{cond_label}\n(no data)")
            continue

        stats = relay._safe_wilcoxon(c_vals, d_vals)
        p = stats.get("wilcoxon_p")

        bp = ax.boxplot(
            [c_clean, d_clean],
            labels=["C\n(RelayScore)", "D\n(fuzzy MADM)"],
            patch_artist=True, widths=0.5, showmeans=True,
        )
        colors = ["#2b8cbe", "#fdae61"]
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.8)

        _r.seed(0)
        for xpos, vals in ((1, c_clean), (2, d_clean)):
            jitter = [xpos + _r.uniform(-0.06, 0.06) for _ in vals]
            ax.scatter(jitter, vals, s=14, color="black", alpha=0.35, zorder=3)

        y_top = max(c_clean + d_clean)
        _annotate_bracket(ax, 1, 2, y_top * 1.03, _sig_marker(p))
        ax.set_ylim(top=y_top * 1.15)
        ax.set_title(cond_label, fontsize=12, fontweight="bold")
        ax.grid(axis="y", alpha=0.25)

        print(f"   {cond_label}: RelayScore(C) mean={round(statistics.mean(c_clean), 3)}, "
              f"fuzzy-MADM(D) mean={round(statistics.mean(d_clean), 3)}, {_sig_marker(p)}")

    axes[0].set_ylabel(ylabel)
    plt.suptitle("RelayScore vs Fuzzy MADM: Robustness to Non-Stationary Wind",
                  fontweight="bold")
    plt.tight_layout()
    plt.savefig(_p(filename), dpi=300)
    plt.close(fig)
    print(f"✅ Saved {filename}")


# ══════════════════════════════════════════════════════════════
# 4b) DOES THE DYNAMIC LAYER CLOSE THE GAP? C vs D vs E, dynamic wind
#     only -- direct follow-up to the headline C-vs-D figure above.
# ══════════════════════════════════════════════════════════════
def plot_cde_dynamic_gap(summary, metric, ylabel, filename):
    entry = summary.get(metric)
    if not entry:
        print(f"⚠️  Skipping {filename}: metric '{metric}' not in summary.")
        return

    labels = ["C\n(RelayScore,\nstatic weights)",
              "D\n(fuzzy MADM)",
              "E\n(RelayScore,\ndynamic weights)"]
    means = [entry.get(f"{l}_mean") for l in ("C", "D", "E")]
    valid_means = [v for v in means if v is not None]
    if not valid_means:
        print(f"⚠️  Skipping {filename}: no numeric means for metric '{metric}'.")
        return
    y_top = max(valid_means)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    bars = ax.bar(labels, means, color=["#2b8cbe", "#fdae61", "#31a354"], width=0.55)
    for bar, m in zip(bars, means):
        if m is not None:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                     f"{m:.1f}", ha="center", va="bottom", fontsize=10)

    rank_p = entry.get("ranking_rule_effect_C_vs_D", {}).get("wilcoxon_p")
    dyn_p = entry.get("dynamic_weighting_effect_C_vs_E", {}).get("wilcoxon_p")
    refined_p = entry.get("fully_refined_effect_D_vs_E", {}).get("wilcoxon_p")
    _annotate_bracket(ax, 0, 1, y_top * 1.03, f"C vs D (existing gap)\n{_sig_marker(rank_p)}")
    _annotate_bracket(ax, 0, 2, y_top * 1.20, f"dynamic layer's own effect (C vs E)\n{_sig_marker(dyn_p)}")
    _annotate_bracket(ax, 1, 2, y_top * 1.37, f"fully-refined vs fuzzy-MADM (D vs E)\n{_sig_marker(refined_p)}")

    ax.set_ylabel(ylabel)
    ax.set_title(f"Does the dynamic-weighting layer close the gap?\n{metric}, dynamic wind only",
                  fontsize=12, fontweight="bold")
    ax.set_ylim(top=y_top * 1.55)
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(_p(filename), dpi=300)
    plt.close(fig)
    print(f"✅ Saved {filename}  (dynamic-wind C/D/E means: {means})")


# ══════════════════════════════════════════════════════════════
# 5) COMBINED STATISTICS TABLE
# ══════════════════════════════════════════════════════════════
def write_csv(static_summary, dynamic_summary, filename):
    comparisons = [
        ("fleet_effect_A_vs_B", "A", "B"),
        ("relay_intel_effect_B_vs_C", "B", "C"),
        ("relay_intel_effect_B_vs_D", "B", "D"),
        ("ranking_rule_effect_C_vs_D", "C", "D"),
    ]

    rows = []
    all_metrics = sorted(set(static_summary) | set(dynamic_summary))
    for metric in all_metrics:
        s = static_summary.get(metric, {})
        d = dynamic_summary.get(metric, {})
        row = {
            "metric": metric,
            "static_A_mean": s.get("A_mean"), "dynamic_A_mean": d.get("A_mean"),
            "static_B_mean": s.get("B_mean"), "dynamic_B_mean": d.get("B_mean"),
            "static_C_mean": s.get("C_mean"), "dynamic_C_mean": d.get("C_mean"),
            "static_D_mean": s.get("D_mean"), "dynamic_D_mean": d.get("D_mean"),
        }
        for key, la, lb in comparisons:
            row[f"static_{key}_p"] = s.get(key, {}).get("wilcoxon_p")
            row[f"dynamic_{key}_p"] = d.get(key, {}).get("wilcoxon_p")
        rows.append(row)

    if not rows:
        print("⚠️  No statistics to write.")
        return

    fieldnames = list(rows[0].keys())
    with open(_p(filename), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        prov_row = {k: "" for k in fieldnames}
        prov_row["metric"] = "terrain_source"
        prov_row["static_A_mean"] = a20.TERRAIN_SOURCE
        writer.writerow(prov_row)
        writer.writerows(rows)
    print(f"✅ Saved {filename}  ({len(rows)} rows, terrain_source={a20.TERRAIN_SOURCE})")


def main():
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    num_seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    seeds = list(range(1, num_seeds + 1))

    print(f"Running with max_steps={steps}, num_seeds={num_seeds}\n")

    print("=== [1/6] Wind profile figure ===")
    plot_wind_profile(steps)

    print("\n=== [2/6] A/B/C/D ablation under STATIC wind (regression baseline) ===")
    static_rows, static_summary = _run_abcd_under_condition(
        seeds, steps, dynamic=False, out_csv=_p("abcd_static_raw.csv")
    )

    print("\n=== [3/6] A/B/C/D ablation under DYNAMIC wind (robustness check) ===")
    dynamic_rows, dynamic_summary = _run_abcd_under_condition(
        seeds, steps, dynamic=True, out_csv=_p("abcd_dynamic_raw.csv")
    )

    print("\n=== [4/6] Ablation bar charts (static + dynamic) ===")
    ablation_bar_one_condition(static_summary, "coverage_pct", "Static wind",
                                "ablation_static_wind.png")
    ablation_bar_one_condition(dynamic_summary, "coverage_pct", "Dynamic wind",
                                "ablation_dynamic_wind.png")

    print("\n=== [5/6] Headline figure: RelayScore vs fuzzy MADM robustness ===")
    plot_relayscore_vs_fuzzy_robustness(
        static_rows, dynamic_rows, "coverage_pct", "Coverage (%)",
        "relayscore_vs_fuzzy_madm_robustness.png",
    )

    write_csv(static_summary, dynamic_summary, "wind_robustness_statistics.csv")

    print("\n=== [6/6] Does the dynamic-weighting layer (Policy E) close the "
          "dynamic-wind gap? (C vs D vs E, dynamic wind only) ===")
    _, cde_summary = _run_abcde_dynamic(
        seeds, steps, out_csv=_p("abcde_dynamic_raw.csv")
    )
    plot_cde_dynamic_gap(cde_summary, "coverage_pct", "Coverage (%)",
                          "cde_dynamic_gap.png")

    print(f"\n🎉 Wind + robustness figures saved under: {os.path.abspath(OUT_DIR)}")


if __name__ == "__main__":
    main()

    