# -*- coding: utf-8 -*-


import os
import sys
import csv
import json
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata

import a2e_relay_fixed_final_51 as relay
import a20

# Selectors used directly (not via a run_X_way_comparison wrapper) so this
# script can build the C/D/E/F/H refinement comparison without needing to
# touch a2e_relay_fixed_final_51.py -- run_seven_way_comparison() there
# uses the buggy Policy G (select_best_relay_battery_aware); we want its
# bug-fixed sibling H (select_best_relay_battery_aware_capped) instead,
# see that file's own "Policy G's failure mode, and the fix (Policy H)"
# comment block for why.
from a2e_relay_fixed_final_51 import (
    select_best_relay,                    # C
    select_fuzzy_relay_true_fis,          # D (NEW)
    select_best_relay_dynamic,            # E
    select_best_relay_predictive,         # F
    select_best_relay_battery_aware_capped,  # H
    simulate_fleet,
    PRIMARY_METRICS,
    SECONDARY_METRICS,
)

# Surrogate (distilled) FIS -- Policy D2. Requires build_surrogate_fis.py to
# have been run first (produces luts/lut_shared_res9.npz next to this file).
# Imported directly here too (not just via a2e_relay_fixed_final_51's own
# try/except registration) so run_surrogate_comparison() below can call it
# by name, matching how select_fuzzy_relay_true_fis (D) is imported above.
from surrogate_fis import select_fuzzy_relay_surrogate_fis  # D2

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
# FILE-COUNT REDUCTION HELPERS
#
# The pipeline used to leave ~26 separate PNG/CSV files in OUT_DIR
# because every wind condition, every metric (coverage / delay /
# energy), and every robustness check got its own file. Nothing about
# WHAT gets computed or plotted changes below -- every individual
# plotting/statistics function above is untouched. These two helpers
# just glue closely-related outputs that were already saved to disk
# into ONE combined file afterward, then delete the now-redundant
# originals, so the thesis folder only has the files actually worth
# citing individually.
#
# Set KEEP_INDIVIDUAL_FILES = True to go back to the old behaviour
# (every plot/CSV kept as its own separate file, nothing combined or
# deleted) -- e.g. if a reviewer wants the delay figure on its own, or
# for an appendix that lists every raw file.
# ══════════════════════════════════════════════════════════════
KEEP_INDIVIDUAL_FILES = False


def _combine_pngs(paths, out_path, ncols=1, panel_w=6.5, panel_h=5.0):
    """Stack several already-saved PNGs into one combined PNG (a grid
    with `ncols` columns), then delete the originals. Paths that don't
    exist (e.g. a conditional plot such as refinement_dynamic_gap.png,
    only produced when the C-vs-D gap is significant) are silently
    skipped so this never crashes the rest of main()."""
    if KEEP_INDIVIDUAL_FILES:
        return
    imgs = [(p, plt.imread(p)) for p in paths if p and os.path.exists(p)]
    if not imgs:
        print(f"⚠️  Skipping {os.path.basename(out_path)}: nothing to combine.")
        return
    n = len(imgs)
    ncols = max(1, min(ncols, n))
    nrows = -(-n // ncols)  # ceiling division, no extra import needed
    fig, axes = plt.subplots(nrows, ncols, figsize=(panel_w * ncols, panel_h * nrows))
    axes = np.atleast_1d(axes).flatten()
    for ax, (_, img) in zip(axes, imgs):
        ax.imshow(img)
        ax.axis("off")
    for ax in axes[n:]:
        ax.axis("off")
    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    for p, _ in imgs:
        if os.path.abspath(p) != os.path.abspath(out_path):
            try:
                os.remove(p)
            except OSError:
                pass
    print(f"✅ Combined {n} figure(s) -> {os.path.basename(out_path)}")


def _combine_csvs(path_label_pairs, out_path, label_col="condition"):
    """Merge several CSVs that only differ by a run condition (e.g.
    static vs dynamic wind) into ONE CSV with an extra label column,
    instead of leaving several near-duplicate files on disk.
    path_label_pairs: list of (path, label) tuples; missing paths are
    skipped (same reasoning as _combine_pngs above). Column set is the
    union across all input files, in first-seen order, with the label
    column placed first."""
    if KEEP_INDIVIDUAL_FILES:
        return
    fieldnames, all_rows, existing_paths = [], [], []
    seen = set()
    for path, label in path_label_pairs:
        if not path or not os.path.exists(path):
            continue
        existing_paths.append(path)
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for col in reader.fieldnames or []:
                if col not in seen:
                    seen.add(col)
                    fieldnames.append(col)
            for row in reader:
                row[label_col] = label
                all_rows.append(row)
    if not all_rows:
        print(f"⚠️  Skipping {os.path.basename(out_path)}: nothing to combine.")
        return
    fieldnames = [label_col] + fieldnames
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    for path in existing_paths:
        if os.path.abspath(path) != os.path.abspath(out_path):
            try:
                os.remove(path)
            except OSError:
                pass
    print(f"✅ Combined {len(existing_paths)} CSV(s) -> {os.path.basename(out_path)} "
          f"({len(all_rows)} rows)")


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


def _sig_stars(p):
    """Compact star-only marker (no p-value text) for dense multi-panel
    figures where _sig_marker's full string would be too wide."""
    if p is None:
        return "n/a"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def _annotate_bracket(ax, x1, x2, y, text, fontsize=9):
    h = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.03
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.2, color="black")
    ax.text((x1 + x2) / 2, y + h, text, ha="center", va="bottom", fontsize=fontsize)


# ══════════════════════════════════════════════════════════════
# EFFECT SIZE + CONFIDENCE INTERVAL (addresses reviewer concern: a
# non-significant Wilcoxon p-value alone doesn't tell a reader whether
# there is "no effect" or just "not enough seeds to detect it" -- these
# three numbers together do:
#   - mean_diff + 95% CI  -> the PRACTICAL size of the difference, with
#     uncertainty. A CI that straddles 0 is the honest picture behind a
#     non-significant p; a CI that excludes 0 despite p>0.05 would be a
#     red flag worth investigating (shouldn't happen with a correct
#     Wilcoxon/bootstrap pair, but is a useful sanity check).
#   - rank_biserial        -> matched-pairs rank-biserial correlation,
#     the standard non-parametric effect size that PAIRS WITH a Wilcoxon
#     signed-rank test. NOTE: Cliff's delta is the equivalent effect size
#     for INDEPENDENT/unpaired samples -- every comparison in this script
#     is paired on seed (same seed run under both policies), so
#     rank-biserial correlation is the statistically correct choice here,
#     not Cliff's delta (the two coincide for the unpaired 2-sample case,
#     but this isn't that case).
# ══════════════════════════════════════════════════════════════
def _rank_biserial_paired(xs, ys):
    """Matched-pairs rank-biserial correlation for two paired samples.
    r = (sum of ranks favoring x - sum of ranks favoring y) / total rank
    sum, computed on |difference| ranks (ties in the differences get
    average ranks, standard treatment). Returns None if every pair is
    tied (no information). Range: [-1, 1]; sign matches whether x tends
    to exceed y (+) or fall short of y (-)."""
    diffs = [a - b for a, b in zip(xs, ys) if a != b]
    if not diffs:
        return None
    abs_diffs = [abs(d) for d in diffs]
    ranks = rankdata(abs_diffs)
    pos = sum(r for r, d in zip(ranks, diffs) if d > 0)
    neg = sum(r for r, d in zip(ranks, diffs) if d < 0)
    total = pos + neg
    if total == 0:
        return 0.0
    return round((pos - neg) / total, 4)


def _effect_size_label(rb):
    if rb is None:
        return "n/a"
    m = abs(rb)
    if m < 0.10:
        return "negligible"
    if m < 0.30:
        return "small"
    if m < 0.50:
        return "medium"
    return "large"


def _bootstrap_ci_mean_diff(xs, ys, n_boot=10000, alpha=0.05, seed=12345):
    """Percentile bootstrap 95% CI on the mean PAIRED difference (x - y).
    Resamples the per-seed differences with replacement (not x and y
    separately -- that would break the pairing) n_boot times."""
    diffs = np.asarray(xs, dtype=float) - np.asarray(ys, dtype=float)
    n = len(diffs)
    if n < 2:
        return None, None
    rng = np.random.default_rng(seed)
    boot_means = rng.choice(diffs, size=(n_boot, n), replace=True).mean(axis=1)
    lo, hi = np.percentile(boot_means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return round(float(lo), 4), round(float(hi), 4)


def _paired_effect_stats(x, y, n_boot=10000, alpha=0.05):
    """Full paired-comparison report for one metric: n, mean_diff (x-y),
    95% bootstrap CI on mean_diff, Wilcoxon signed-rank p, and matched-
    pairs rank-biserial correlation (+ a plain-language size label).
    Drops any (a, b) pair where either side is None/non-numeric (e.g. A's
    avg_relay_delay_steps, which is always None -- see
    _extract_baseline_a_metrics's docstring). Booleans (mission_completed)
    are coerced to 0/1, same convention as a2e_relay_fixed_final_51.py's
    _safe_wilcoxon."""
    xs, ys = [], []
    for a, b in zip(x, y):
        if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
            continue
        xs.append(int(a) if isinstance(a, bool) else a)
        ys.append(int(b) if isinstance(b, bool) else b)

    if len(xs) < 2:
        return {"n": len(xs), "mean_diff": None, "ci_lo": None, "ci_hi": None,
                 "wilcoxon_p": None, "rank_biserial": None, "effect_size": "n/a"}

    mean_diff = round(statistics.mean(a - b for a, b in zip(xs, ys)), 4)
    ci_lo, ci_hi = _bootstrap_ci_mean_diff(xs, ys, n_boot=n_boot, alpha=alpha)
    wilcoxon_p = relay._safe_wilcoxon(xs, ys).get("wilcoxon_p")
    rb = _rank_biserial_paired(xs, ys)

    return {
        "n": len(xs),
        "mean_diff": mean_diff,
        "ci_lo": ci_lo, "ci_hi": ci_hi,
        "wilcoxon_p": wilcoxon_p,
        "rank_biserial": rb,
        "effect_size": _effect_size_label(rb),
    }


def _format_effect_stats(stats, unit=""):
    """One-line human-readable rendering of _paired_effect_stats()'s output,
    for print statements and figure annotations."""
    if stats.get("mean_diff") is None:
        return "n/a (insufficient paired data)"
    md, lo, hi = stats["mean_diff"], stats["ci_lo"], stats["ci_hi"]
    p = stats["wilcoxon_p"]
    rb = stats["rank_biserial"]
    p_str = "n/a" if p is None else (f"{p:.4f}" if p >= 0.001 else "<0.001")
    rb_str = "n/a" if rb is None else f"{rb:+.3f}"
    ci_str = "n/a" if lo is None else f"[{lo:+.3f}, {hi:+.3f}]{unit}"
    return (f"mean diff = {md:+.3f}{unit}, 95% CI = {ci_str}, "
            f"Wilcoxon p = {p_str}, rank-biserial r = {rb_str} ({stats['effect_size']})")


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


def _run_d_indep_robustness_under_condition(seeds, max_steps, dynamic, out_csv):
    """RULE-BASE-CIRCULARITY ROBUSTNESS CHECK, wired into the actual pipeline.

    Runs relay.run_policy_d_weight_robustness_check() (C vs original D vs
    D-indep, see that function's docstring in a2e_relay_fixed_final_51.py
    for the full reviewer-concern writeup) under ONE wind condition, using
    the same a20.WIND_DYNAMIC_ENABLED toggle convention as
    _run_abcd_under_condition() / _run_refinements_dynamic() above.

    NOTE: this was previously only reachable via
    a2e_relay_fixed_final_51.py's own `if __name__ == "__main__":` block,
    which never runs when that file is imported as a module (as it is
    here) -- so the D-indep comparison never actually executed as part of
    this pipeline. This wrapper is what fixes that.
    """
    a20.WIND_DYNAMIC_ENABLED = dynamic
    try:
        rows, summary = relay.run_policy_d_weight_robustness_check(
            seeds, max_steps=max_steps, out_csv=out_csv
        )
    finally:
        a20.WIND_DYNAMIC_ENABLED = False  # always leave the module in the safe default state
    return rows, summary


def _run_terrain_wind_sweep(seeds, max_steps, terrain_scenarios=None,
                             wind_modes=("static", "dynamic", "stochastic"), out_csv=None,
                             policies=("C", "D", "T", "V")):
    """MULTIPLE TERRAIN x WIND GENERALIZATION SWEEP, wired into the actual
    pipeline (journal-extension experiment -- see Section VI, "Single
    simulated environment" limitation in the paper).

    Thin wrapper around relay.run_terrain_wind_generalization_analysis()
    (C, D, T, V across every named a20.TERRAIN_SCENARIOS preset crossed
    with static / dynamic / stochastic wind -- see that function's
    docstring for the full reviewer-concern writeup and
    a20.set_terrain_scenario()/set_wind_mode()'s docstrings for how each
    axis is switched). `policies` defaults to all four -- pass
    policies=("C","D") for the original, cheaper C-vs-D-only sweep if you
    don't want the added T/V runtime cost.

    Restores a20's terrain to the paper's original 'benapole_delta' sector
    when done (the underlying relay function already restores wind mode
    to 'static' on its own), so nothing else in this pipeline silently
    keeps running against a different terrain afterward."""
    try:
        rows, summary = relay.run_terrain_wind_generalization_analysis(
            seeds, terrain_scenarios=terrain_scenarios, wind_modes=wind_modes,
            max_steps=max_steps, out_csv=out_csv, policies=policies,
        )
    finally:
        a20.set_terrain_scenario("benapole_delta")
    return rows, summary


def _run_refinements_dynamic(seeds, max_steps, out_csv, dynamic=True):
    """C / D / E / F / H comparison, run under EITHER wind condition
    (dynamic=True keeps the original dynamic-wind behaviour; dynamic=False
    runs the identical C/D/E/F/H comparison under static wind instead).
    Runs THREE different refinements of C side by side on the SAME
    seeds/fleet so their marginal value is directly comparable:
        E = Policy C + dynamic (state-dependent) importance reweighting
        F = Policy C + predictive (arrival-time) wind & coverage-loss --
            per a2e_relay_fixed_final_51.py's own module comments, this
            is the stronger of the two reweighting/prediction refinements
        H = Policy C + a genuine, BOUNDED battery-margin ranking term --
            the bug-fixed version of Policy G (G's uncapped battery term
            gives idle-reserve candidates an unbeatable, unbounded credit
            and was diagnosed to cause "relay starvation" in ~13% of
            seeds; H fixes this by capping credit on a fixed absolute
            scale). This is used here instead of G for that reason.
    Built directly on simulate_fleet() (not a run_X_way_comparison()
    wrapper) so H can be substituted for the buggy G without needing to
    add a new wrapper function to a2e_relay_fixed_final_51.py.

    Originally only run under dynamic wind (E/F/H's extra machinery only
    has something to react to -- wind swings, coverage/battery pressure
    spikes -- under the harder, non-stationary condition; under static
    wind C already had no significant gap with D, see
    ablation_static_wind.png). Now also called with dynamic=False so the
    Static column in metric_summary_grid.png shows the same C/E/D
    three-way breakdown as Dynamic, for symmetry -- see main().
    """
    a20.WIND_DYNAMIC_ENABLED = dynamic
    try:
        rows = []
        for seed in seeds:
            m_c = simulate_fleet(seed, relay_selector=select_best_relay, max_steps=max_steps)
            m_d = simulate_fleet(seed, relay_selector=select_fuzzy_relay_true_fis, max_steps=max_steps)
            m_e = simulate_fleet(seed, relay_selector=select_best_relay_dynamic, max_steps=max_steps)
            m_f = simulate_fleet(seed, relay_selector=select_best_relay_predictive, max_steps=max_steps)
            m_h = simulate_fleet(seed, relay_selector=select_best_relay_battery_aware_capped, max_steps=max_steps)
            row = {"seed": seed}
            for label, m in (("C", m_c), ("D", m_d), ("E", m_e), ("F", m_f), ("H", m_h)):
                for k, v in m.items():
                    if k == "decision_log":
                        continue
                    row[f"{label}_{k}"] = v
            rows.append(row)
            print(f"[seed {seed}] C={m_c['coverage_pct']}%  D(fuzzy)={m_d['coverage_pct']}%  "
                  f"E(dyn-weight)={m_e['coverage_pct']}%  F(predictive)={m_f['coverage_pct']}%  "
                  f"H(batt-capped)={m_h['coverage_pct']}%")

        if out_csv and rows:
            with open(out_csv, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)

        summary = {}
        for metric in PRIMARY_METRICS + SECONDARY_METRICS:
            vals = {}
            for label in ("C", "D", "E", "F", "H"):
                vals[label] = [r[f"{label}_{metric}"] for r in rows]
            entry = {}
            for label in ("C", "D", "E", "F", "H"):
                clean = [v for v in vals[label] if isinstance(v, (int, float))]
                entry[f"{label}_mean"] = round(statistics.mean(clean), 3) if clean else None
            entry["ranking_rule_effect_C_vs_D"] = relay._safe_wilcoxon(vals["C"], vals["D"])
            entry["dynamic_weighting_effect_C_vs_E"] = relay._safe_wilcoxon(vals["C"], vals["E"])
            entry["predictive_effect_C_vs_F"] = relay._safe_wilcoxon(vals["C"], vals["F"])
            entry["battery_aware_effect_C_vs_H"] = relay._safe_wilcoxon(vals["C"], vals["H"])
            entry["dynamic_weighting_vs_fuzzy_D_vs_E"] = relay._safe_wilcoxon(vals["D"], vals["E"])
            entry["predictive_vs_fuzzy_D_vs_F"] = relay._safe_wilcoxon(vals["D"], vals["F"])
            entry["battery_aware_vs_fuzzy_D_vs_H"] = relay._safe_wilcoxon(vals["D"], vals["H"])
            summary[metric] = entry

        return rows, summary
    finally:
        a20.WIND_DYNAMIC_ENABLED = False  # always leave the module in the safe default state


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


def plot_d_indep_robustness(summary, metric, ylabel, condition_label, filename):
    """C vs original-D (shared FUZZY_ATTR_WEIGHTS) vs D-indep (independent
    equal-weight rule base) bar chart, same visual convention as
    ablation_bar_one_condition() above -- see
    relay.run_policy_d_weight_robustness_check()'s docstring for what the
    three comparison keys (C_vs_D_shared_weights, C_vs_D_indep_weights,
    D_vs_D_indep_rule_base_sensitivity) mean."""
    entry = summary.get(metric)
    if not entry:
        print(f"⚠️  Skipping {filename}: metric '{metric}' not in summary.")
        return

    labels = ["C\n(RelayScore)",
              "D\n(fuzzy MADM,\nweights shared with C)",
              "D-indep\n(fuzzy MADM,\nindependent weights)"]
    means = [entry.get(f"{l}_mean") for l in ("C", "D", "D_indep")]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    bars = ax.bar(labels, means, color=["#2b8cbe", "#fdae61", "#66c2a5"], width=0.55)
    for bar, m in zip(bars, means):
        if m is not None:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                     f"{m:.2f}", ha="center", va="bottom", fontsize=10)

    valid_means = [v for v in means if v is not None]
    if not valid_means:
        plt.close(fig)
        print(f"⚠️  Skipping {filename}: no numeric means for metric '{metric}'.")
        return
    y_top = max(valid_means)

    shared_p = (entry.get("C_vs_D_shared_weights") or {}).get("wilcoxon_p")
    indep_p = (entry.get("C_vs_D_indep_weights") or {}).get("wilcoxon_p")
    sens_p = (entry.get("D_vs_D_indep_rule_base_sensitivity") or {}).get("wilcoxon_p")
    _annotate_bracket(ax, 0, 1, y_top * 1.03, f"C vs D (shared weights)\n{_sig_marker(shared_p)}")
    _annotate_bracket(ax, 0, 2, y_top * 1.18, f"C vs D-indep (independent weights)\n{_sig_marker(indep_p)}")
    _annotate_bracket(ax, 1, 2, y_top * 1.33, f"D vs D-indep (rule-base sensitivity)\n{_sig_marker(sens_p)}")

    ax.set_ylabel(ylabel)
    ax.set_title(f"Rule-base-circularity check ({condition_label}): {ylabel}",
                  fontsize=12, fontweight="bold")
    ax.set_ylim(top=y_top * 1.5)
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(_p(filename), dpi=300)
    plt.close(fig)
    print(f"✅ Saved {filename}  ({condition_label} C/D/D-indep means: {means})")


# ══════════════════════════════════════════════════════════════
# 3b) TERRAIN x WIND GENERALIZATION HEATMAP (journal-extension figure):
#     C-vs-D coverage gap across every terrain scenario x wind mode
#     combination, in one grid -- the "does the paper's finding
#     generalize?" figure.
# ══════════════════════════════════════════════════════════════
def plot_terrain_wind_generalization(summary, metric="coverage_pct", ylabel="Coverage (%)",
                                      filename="terrain_wind_generalization.png",
                                      policies=("C", "D", "T", "V")):
    """summary: the scenario_summary list returned by
    relay.run_terrain_wind_generalization_analysis() (same structure
    _run_terrain_wind_sweep() passes through). One row per
    (terrain, wind_mode) combination; this renders a terrain-x-wind grid
    of small panels, each a bar per requested policy (2 bars for the
    original policies=("C","D"), or 4 bars -- C, D, T, V -- with the new
    default), annotated with the paired Wilcoxon significance of each
    non-D policy against D -- so a reader sees at a glance whether the
    paper's "X matches D on coverage" (or delay/energy) finding holds
    across every terrain/wind combination tested, for every policy, not
    just C on the one originally reported.

    `policies` only draws bars for labels actually present in `summary`
    (checked per-entry, not just the first one), so this still renders
    correctly on OLD C-vs-D-only summaries without needing the caller to
    pass policies=("C","D") explicitly."""
    if not summary:
        print(f"⚠️  Skipping {filename}: empty terrain/wind summary.")
        return

    colors = {"C": "#2b8cbe", "D": "#fdae61", "T": "#66c2a5", "V": "#9e7bb5"}
    labels_display = {"C": "C\n(RelayScore)", "D": "D\n(fuzzy MADM)",
                       "T": "T\n(TOPSIS)", "V": "V\n(VIKOR)"}

    terrains = sorted({e["terrain"] for e in summary})
    wind_modes = sorted({e["wind_mode"] for e in summary},
                         key=lambda w: {"static": 0, "dynamic": 1, "stochastic": 2}.get(w, 99))
    by_key = {(e["terrain"], e["wind_mode"]): e for e in summary}

    fig, axes = plt.subplots(len(terrains), len(wind_modes),
                              figsize=(3.6 * len(wind_modes), 3.4 * len(terrains)),
                              squeeze=False)

    for i, terrain in enumerate(terrains):
        for j, wind_mode in enumerate(wind_modes):
            ax = axes[i][j]
            entry = by_key.get((terrain, wind_mode))
            present = [p for p in policies if entry and entry.get(f"{p}_{metric}") is not None]
            if not entry or not present:
                ax.axis("off")
                ax.set_title(f"{terrain}\n{wind_mode}\n(no data)", fontsize=9)
                continue

            means = [entry.get(f"{p}_{metric}") for p in present]
            bars = ax.bar([labels_display.get(p, p) for p in present], means,
                           color=[colors.get(p, "gray") for p in present], width=0.6)
            for bar, m in zip(bars, means):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                         f"{m:.1f}", ha="center", va="bottom", fontsize=8)

            # Significance stars: prefer the new "{metric}_test_<label>_vs_D"
            # keys (works for any non-D policy); fall back to the legacy
            # "{metric}_test" key (always meant C-vs-D) on old summaries.
            sig_bits = []
            for p in present:
                if p == "D":
                    continue
                test = entry.get(f"{metric}_test_{p}_vs_D")
                if test is None and p == "C":
                    test = entry.get(f"{metric}_test")  # legacy fallback
                pval = (test or {}).get("wilcoxon_p")
                if pval is not None:
                    sig_bits.append(f"{p}:{_sig_stars(pval)}")
            sig_str = "  ".join(sig_bits) if sig_bits else ""

            ax.set_title(f"{terrain}\n{wind_mode}  {sig_str}", fontsize=9)
            ax.tick_params(axis="x", labelsize=7)
            ax.grid(axis="y", alpha=0.2)
            if j == 0:
                ax.set_ylabel(ylabel, fontsize=8)

    fig.suptitle(f"Terrain x Wind Generalization: {ylabel}  (vs.\\ true FIS D, per policy)",
                  fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(_p(filename), dpi=300)
    plt.close(fig)
    print(f"✅ Saved {filename}")




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

    # First pass: clean the data and compute stats per condition, and find
    # the GLOBAL y_top across BOTH conditions. This matters because
    # sharey=True links the two axes' y-limits -- calling ax.set_ylim()
    # once per condition inside a single loop means whichever condition is
    # processed LAST silently overrides the other's headroom. If one
    # condition has a taller outlier (as static wind did here, for
    # avg_relay_energy_pct), its bracket annotation ends up positioned for
    # a tall axis that then gets squashed down to the shorter condition's
    # range, pushing the text out of the panel and up near the suptitle.
    # Using one global_top for both the bracket position AND the shared
    # ylim (set once, after the loop) fixes this.
    prepared = []
    global_top = 0.0
    for cond_label, rows in conditions:
        c_vals = [r.get(f"C_{metric}") for r in rows]
        d_vals = [r.get(f"D_{metric}") for r in rows]
        c_clean = [v for v in c_vals if isinstance(v, (int, float))]
        d_clean = [v for v in d_vals if isinstance(v, (int, float))]
        if not c_clean or not d_clean:
            prepared.append((cond_label, None))
            continue
        stats = _paired_effect_stats(c_vals, d_vals)
        global_top = max(global_top, max(c_clean + d_clean))
        prepared.append((cond_label, (c_clean, d_clean, stats)))

    for ax, (cond_label, data) in zip(axes, prepared):
        if data is None:
            ax.set_title(f"{cond_label}\n(no data)")
            continue
        c_clean, d_clean, stats = data
        p = stats.get("wilcoxon_p")

        try:
            bp = ax.boxplot(
                [c_clean, d_clean],
                tick_labels=["C\n(RelayScore)", "D\n(fuzzy MADM)"],
                patch_artist=True, widths=0.5, showmeans=True,
            )
        except TypeError:  # matplotlib < 3.9 doesn't know tick_labels
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

        rb = stats.get("rank_biserial")
        rb_str = "n/a" if rb is None else f"{rb:+.2f} ({stats['effect_size']})"
        ci_str = ("n/a" if stats.get("ci_lo") is None
                   else f"[{stats['ci_lo']:+.2f}, {stats['ci_hi']:+.2f}]")
        bracket_text = (f"{_sig_marker(p)}\n"
                          f"Δ={stats['mean_diff']:+.2f}%  95% CI={ci_str}\n"
                          f"rank-biserial r={rb_str}")
        _annotate_bracket(ax, 1, 2, global_top * 1.03, bracket_text, fontsize=8)
        ax.set_title(cond_label, fontsize=12, fontweight="bold")
        ax.grid(axis="y", alpha=0.25)

        print(f"   {cond_label}: RelayScore(C) mean={round(statistics.mean(c_clean), 3)}, "
              f"fuzzy-MADM(D) mean={round(statistics.mean(d_clean), 3)}")
        print(f"      {_format_effect_stats(stats, unit='%')}")

    axes[0].set_ylim(top=global_top * 1.28)  # shared axis -- set ONCE, after the loop
    axes[0].set_ylabel(ylabel)
    plt.suptitle("RelayScore vs Fuzzy MADM: Robustness to Non-Stationary Wind",
                  fontweight="bold")
    plt.tight_layout()
    plt.savefig(_p(filename), dpi=300)
    plt.close(fig)
    print(f"✅ Saved {filename}")


# ══════════════════════════════════════════════════════════════
# 4b) DOES ANY REFINEMENT CLOSE THE GAP? C vs D vs E vs F vs H,
#     dynamic wind only -- direct follow-up to the headline C-vs-D
#     figure above. Three independent refinements of C are shown
#     side by side so their marginal value against fuzzy MADM (D) is
#     directly comparable, not just each refinement's effect over C.
# ══════════════════════════════════════════════════════════════
def plot_refinement_dynamic_gap(summary, metric, ylabel, filename):
    entry = summary.get(metric)
    if not entry:
        print(f"⚠️  Skipping {filename}: metric '{metric}' not in summary.")
        return

    order = ["C", "D", "E", "F", "H"]
    labels = ["C\n(RelayScore,\nstatic weights)",
              "D\n(fuzzy MADM)",
              "E\n(RelayScore,\ndynamic weights)",
              "F\n(RelayScore,\npredictive)",
              "H\n(RelayScore,\nbattery-capped)"]
    means = [entry.get(f"{l}_mean") for l in order]
    valid_means = [v for v in means if v is not None]
    if not valid_means:
        print(f"⚠️  Skipping {filename}: no numeric means for metric '{metric}'.")
        return
    y_top = max(valid_means)

    fig, ax = plt.subplots(figsize=(9.5, 6))
    colors = ["#2b8cbe", "#fdae61", "#31a354", "#756bb1", "#e6550d"]
    bars = ax.bar(labels, means, color=colors, width=0.6)
    for bar, m in zip(bars, means):
        if m is not None:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                     f"{m:.1f}", ha="center", va="bottom", fontsize=10)

    rank_p = entry.get("ranking_rule_effect_C_vs_D", {}).get("wilcoxon_p")
    dyn_vs_fuzzy_p = entry.get("dynamic_weighting_vs_fuzzy_D_vs_E", {}).get("wilcoxon_p")
    pred_vs_fuzzy_p = entry.get("predictive_vs_fuzzy_D_vs_F", {}).get("wilcoxon_p")
    batt_vs_fuzzy_p = entry.get("battery_aware_vs_fuzzy_D_vs_H", {}).get("wilcoxon_p")

    _annotate_bracket(ax, 0, 1, y_top * 1.03, f"C vs D (existing gap)\n{_sig_marker(rank_p)}")
    _annotate_bracket(ax, 1, 2, y_top * 1.20, f"E vs fuzzy-MADM\n{_sig_marker(dyn_vs_fuzzy_p)}")
    _annotate_bracket(ax, 1, 3, y_top * 1.37, f"F vs fuzzy-MADM\n{_sig_marker(pred_vs_fuzzy_p)}")
    _annotate_bracket(ax, 1, 4, y_top * 1.54, f"H vs fuzzy-MADM\n{_sig_marker(batt_vs_fuzzy_p)}")

    ax.set_ylabel(ylabel)
    ax.set_title(f"Which refinement closes the gap with fuzzy MADM?\n{metric}, dynamic wind only",
                  fontsize=12, fontweight="bold")
    ax.set_ylim(top=y_top * 1.72)
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(_p(filename), dpi=300)
    plt.close(fig)
    print(f"✅ Saved {filename}  (dynamic-wind C/D/E/F/H means: {means})")


# ══════════════════════════════════════════════════════════════
# 5) COMBINED STATISTICS TABLE
# ══════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════
# 5c) ONE-GLANCE SUMMARY: a grid of small grouped-bar panels, one per
# key metric, C (RelayScore) vs D (fuzzy MADM), static wind vs dynamic
# wind side by side within each panel. This is the "which system wins
# where" overview figure for a supervisor/paper skim -- the three
# detailed boxplots above (coverage/delay/energy) remain the figures to
# cite for exact effect sizes; this one is for the at-a-glance summary
# slide/section.
# ══════════════════════════════════════════════════════════════
_COND_COLOR = {"C": "#2b8cbe", "E": "#2ca02c", "D": "#fdae61"}
_COND_LABEL = {"C": "C (RelayScore)", "E": "E (RelayScore + dynamic weights)",
               "D": "D (fuzzy MADM)"}


def plot_metric_summary_grid(static_rows, dynamic_rows, dynamic_e_rows=None,
                              static_e_rows=None,
                              filename="metric_summary_grid.png"):
    """Same 2x3 "who wins where" grid as before, with one addition: if
    dynamic_e_rows / static_e_rows is given (E = RelayScore + dynamic
    weights, from _run_refinements_dynamic run under the matching wind
    condition), that column shows THREE bars (C, E, D) instead of two,
    with all three pairwise Wilcoxon comparisons (C-E, C-D, E-D) stacked
    above the cluster. A column falls back to C-vs-D only if its E rows
    were not supplied.
    """
    # (metric key, display label, True = higher is better / False = lower
    # is better) -- curated to the metrics that matter for the paper's
    # "comparable effectiveness, lower cost" claim, not every column in
    # the CSV appendix.
    metrics = [
        ("coverage_pct", "Coverage (%)", True),
        ("detection_rate", "Detection rate", True),
        ("relay_success_rate", "Relay success rate", True),
        ("avg_relay_delay_steps", "Avg relay delay (steps)", False),
        ("avg_relay_energy_pct", "Avg relay energy (% battery)", False),
        ("total_energy_consumed", "Total energy consumed", False),
    ]
    # cluster_x: horizontal center of each wind-condition cluster. Dynamic
    # gets extra room (1.35 vs 1.0 spacing) because it may hold 3 bars.
    clusters = []
    if static_e_rows is not None:
        clusters.append(("Static", static_rows, ("C", "E", "D"), static_e_rows))
    else:
        clusters.append(("Static", static_rows, ("C", "D")))
    if dynamic_e_rows is not None:
        clusters.append(("Dynamic", dynamic_rows, ("C", "E", "D"), dynamic_e_rows))
    else:
        clusters.append(("Dynamic", dynamic_rows, ("C", "D")))
    cluster_x = [0.0, 1.35]
    bar_width = 0.28

    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    for ax, (metric, ylabel, higher_is_better) in zip(axes.flat, metrics):
        has_data = False
        all_means = []
        cluster_pair_texts = []  # (center_x, [line, line, ...])

        for center, cluster in zip(cluster_x, clusters):
            cname, rows = cluster[0], cluster[1]
            conds = cluster[2]
            e_rows = cluster[3] if len(cluster) > 3 else None

            # When E is present, pull C/D/E all from the SAME refinement
            # run (e_rows) rather than mixing in the separate ablation
            # run's C/D -- same rows means same seed on every line, so
            # the three-way paired Wilcoxon comparisons below are
            # comparing like-for-like, not just numerically similar reruns.
            source_rows = e_rows if e_rows is not None else rows
            vals = {c: [r.get(f"{c}_{metric}") for r in source_rows] for c in conds}
            clean = {c: [v for v in vals[c] if isinstance(v, (int, float))] for c in conds}
            if any(not clean[c] for c in conds):
                continue
            has_data = True

            n = len(conds)
            for i, c in enumerate(conds):
                xpos = center + bar_width * (i - (n - 1) / 2)
                mean = statistics.mean(clean[c])
                all_means.append(mean)
                ax.bar(xpos, mean, bar_width * 0.92, color=_COND_COLOR[c])
                ax.text(xpos, mean, f"{mean:.2f}", ha="center", va="bottom", fontsize=7.5)

            pair_lines = []
            for i in range(n):
                for j in range(i + 1, n):
                    a, b = conds[i], conds[j]
                    stats = _paired_effect_stats(vals[a], vals[b])
                    sig = _sig_stars(stats["wilcoxon_p"])
                    pair_lines.append((f"{a} vs {b}: {sig}", sig not in ("n.s.", "n/a")))
            cluster_pair_texts.append((center, pair_lines))

        if not has_data:
            ax.set_title(f"{ylabel}\n(no data)", fontsize=10)
            ax.axis("off")
            continue

        y_top = max(all_means) if all_means else 1.0
        line_h = y_top * 0.09
        for center, pair_lines in cluster_pair_texts:
            for k, (text, is_sig) in enumerate(pair_lines):
                ax.text(center, y_top * 1.08 + k * line_h, text, ha="center", va="bottom",
                        fontsize=7.5, fontweight="bold" if is_sig else "normal")

        arrow = "↑ higher is better" if higher_is_better else "↓ lower is better"
        ax.set_xticks(cluster_x[:len(clusters)])
        ax.set_xticklabels([c[0] for c in clusters], fontsize=9)
        ax.set_title(f"{ylabel}  ({arrow})", fontsize=9.5, fontweight="bold")
        max_lines = max((len(pl) for _, pl in cluster_pair_texts), default=1)
        ax.set_ylim(top=y_top * (1.12 + 0.11 * max_lines))
        ax.set_xlim(cluster_x[0] - 0.9, cluster_x[-1] + 0.9)
        ax.grid(axis="y", alpha=0.25)
        # NOTE: no per-panel ax.legend() here -- six repeated legends (one
        # per subplot, all "upper right") is both visual clutter and the
        # actual cause of a real bug: the legend box for the "Dynamic"
        # group (the right-hand bars) sat directly on top of that group's
        # bars and significance-star text, making both unreadable. A
        # single shared figure-level legend (added once, below) removes
        # the collision entirely and de-clutters the whole grid.

    any_e = dynamic_e_rows is not None or static_e_rows is not None
    used_conds = ["C", "E", "D"] if any_e else ["C", "D"]
    patches = [plt.matplotlib.patches.Patch(color=_COND_COLOR[c], label=_COND_LABEL[c])
               for c in used_conds]
    fig.legend(handles=patches, loc="upper center",
               ncol=len(patches), fontsize=11, frameon=False, bbox_to_anchor=(0.5, 0.965))

    if static_e_rows is not None and dynamic_e_rows is not None:
        subtitle_extra = ("\nBoth columns add E (RelayScore + dynamic weights), "
                           "run separately under each wind condition.")
    elif dynamic_e_rows is not None:
        subtitle_extra = ("\nDynamic column adds E (RelayScore + dynamic weights); "
                           "Static shows C vs D only -- E was not tested under static wind.")
    elif static_e_rows is not None:
        subtitle_extra = ("\nStatic column adds E (RelayScore + dynamic weights); "
                           "Dynamic shows C vs D only -- E was not tested under dynamic wind.")
    else:
        subtitle_extra = ""
    fig.suptitle(
        "RelayScore (C) vs Fuzzy MADM (D): Metric-by-Metric Summary" +
        (" vs RelayScore+dynamic-weights (E)" if any_e else "") + "\n"
        "Wilcoxon signed-rank, paired by seed, n=30: n.s. = not significant, "
        "* p<0.05, ** p<0.01, *** p<0.001" + subtitle_extra,
        fontsize=12, fontweight="bold", y=1.04,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.86])
    plt.savefig(_p(filename), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Saved {filename}")


def write_csv(static_rows, dynamic_rows, static_summary, dynamic_summary, filename):
    """Full effect-size table: for every metric x every A/B/C/D comparison
    x both wind conditions, reports n, mean_diff, 95% bootstrap CI,
    Wilcoxon p, and matched-pairs rank-biserial correlation (+ size
    label) -- not just a p-value. See _paired_effect_stats() docstring
    for why rank-biserial (not Cliff's delta) is the correct effect size
    here (every comparison is paired on seed)."""
    comparisons = [
        ("fleet_effect_A_vs_B", "A", "B"),
        ("relay_intel_effect_B_vs_C", "B", "C"),
        ("relay_intel_effect_B_vs_D", "B", "D"),
        ("ranking_rule_effect_C_vs_D", "C", "D"),
    ]
    conditions = [("static", static_rows), ("dynamic", dynamic_rows)]

    rows = []
    all_metrics = sorted(set(static_summary) | set(dynamic_summary))
    for metric in all_metrics:
        s, d = static_summary.get(metric, {}), dynamic_summary.get(metric, {})
        row = {
            "metric": metric,
            "static_A_mean": s.get("A_mean"), "dynamic_A_mean": d.get("A_mean"),
            "static_B_mean": s.get("B_mean"), "dynamic_B_mean": d.get("B_mean"),
            "static_C_mean": s.get("C_mean"), "dynamic_C_mean": d.get("C_mean"),
            "static_D_mean": s.get("D_mean"), "dynamic_D_mean": d.get("D_mean"),
        }
        for key, la, lb in comparisons:
            for cond_label, cond_rows in conditions:
                x = [r.get(f"{la}_{metric}") for r in cond_rows]
                y = [r.get(f"{lb}_{metric}") for r in cond_rows]
                stats = _paired_effect_stats(x, y)
                prefix = f"{cond_label}_{key}"
                row[f"{prefix}_n"] = stats["n"]
                row[f"{prefix}_mean_diff"] = stats["mean_diff"]
                row[f"{prefix}_ci_lo"] = stats["ci_lo"]
                row[f"{prefix}_ci_hi"] = stats["ci_hi"]
                row[f"{prefix}_p"] = stats["wilcoxon_p"]
                row[f"{prefix}_rank_biserial"] = stats["rank_biserial"]
                row[f"{prefix}_effect_size"] = stats["effect_size"]
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
    print(f"✅ Saved {filename}  ({len(rows)} rows, terrain_source={a20.TERRAIN_SOURCE}, "
          f"includes mean_diff/95%-CI/rank-biserial for every comparison)")


def main():
    # ── FIXED: Added missing baseline_a keys directly to the function inside Relay_System.py ──
    def _extract_baseline_a_metrics(seed, max_steps):
        sim = a20.DroneSimHeadless(seed)
        sim.run(max_steps)

        g, detected, step = sim.gs, sim.s_detected, sim.step
        staleness_values = [step - c["last_visited_step"] for c in g.values() if not c["is_station"]]
        avg_staleness = statistics.mean(staleness_values) if staleness_values else 0.0
        max_staleness = max(staleness_values) if staleness_values else 0.0

        requested = sim.sR if sim.sR > 0 else (1 if sim.s_handoff_mode else 0)
        fulfilled = sim.sR   

        return {
            "coverage_pct": a20.coverage_pct(g),
            "avg_zone_staleness_steps": round(avg_staleness, 2),
            "max_zone_staleness_steps": max_staleness,
            "detection_rate": round(len(detected) / a20.NUM_THREATS, 4) if a20.NUM_THREATS else None,
            "threats_detected": len(detected),
            "threats_total": a20.NUM_THREATS,
            "relay_success_rate": round(fulfilled / requested, 4) if requested > 0 else None,
            "avg_relay_delay_steps": None,     
            "avg_relay_energy_pct": None,
            "avg_relay_coverage_gap": None,
            "mission_completed": sim.s_first_all is not None,
            "first_all_threats_step": sim.s_first_all,
            "first_full_coverage_step": sim.s_full_cov_step,
            "avg_final_battery": round(sim.s_active["b"], 2),
            "avg_patrol_battery": round(sim.s_active["b"], 2),   
            "avg_reserve_battery": None,                          
            "total_energy_consumed": round(sim.s_energy, 2),      
            "relay_requested": requested,
            "relay_fulfilled": fulfilled,
            # ── NEW TIMING KEYS TO PREVENT KeyError ──
            "avg_selector_time_ms": None,
            "total_selector_time_ms": None,
            "selector_calls": None,
            # ─────────────────────────────────────────
            "RC": sim.sR,
            "final_drone_count": 1,
        }

    # Override the imported one with the fixed one locally
    global relay
    relay._extract_baseline_a_metrics = _extract_baseline_a_metrics

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
    _combine_csvs(
        [(_p("abcd_static_raw.csv"), "static"), (_p("abcd_dynamic_raw.csv"), "dynamic")],
        _p("abcd_raw_combined.csv"),
    )

    print("\n=== [4/6] Ablation bar charts (static + dynamic) ===")
    ablation_bar_one_condition(static_summary, "coverage_pct", "Static wind",
                                "ablation_static_wind.png")
    ablation_bar_one_condition(dynamic_summary, "coverage_pct", "Dynamic wind",
                                "ablation_dynamic_wind.png")
    _combine_pngs(
        [_p("ablation_static_wind.png"), _p("ablation_dynamic_wind.png")],
        _p("ablation_combined.png"), ncols=2,
    )

    print("\n=== [5/6] Headline figure: RelayScore vs fuzzy MADM robustness ===")
    plot_relayscore_vs_fuzzy_robustness(
        static_rows, dynamic_rows, "coverage_pct", "Coverage (%)",
        "relayscore_vs_fuzzy_madm_robustness.png",
    )

    print("\n--- Efficiency companions: relay delay & relay energy, same C-vs-D format ---")
    plot_relayscore_vs_fuzzy_robustness(
        static_rows, dynamic_rows, "avg_relay_delay_steps", "Avg relay delay (steps)",
        "relayscore_vs_fuzzy_madm_delay_robustness.png",
    )
    plot_relayscore_vs_fuzzy_robustness(
        static_rows, dynamic_rows, "avg_relay_energy_pct", "Avg relay energy (% battery)",
        "relayscore_vs_fuzzy_madm_energy_robustness.png",
    )
    _combine_pngs(
        [_p("relayscore_vs_fuzzy_madm_robustness.png"),
         _p("relayscore_vs_fuzzy_madm_delay_robustness.png"),
         _p("relayscore_vs_fuzzy_madm_energy_robustness.png")],
        _p("relayscore_vs_fuzzy_madm_robustness_combined.png"), ncols=1,
    )

    dyn_c = [r.get("C_coverage_pct") for r in dynamic_rows]
    dyn_d = [r.get("D_coverage_pct") for r in dynamic_rows]
    dyn_cd_stats = _paired_effect_stats(dyn_c, dyn_d)
    print("\n--- Effect-size report: coverage_pct, C (RelayScore) vs D (fuzzy MADM), "
          "DYNAMIC wind ---")
    print(f"   n = {dyn_cd_stats['n']} seeds")
    print(f"   {_format_effect_stats(dyn_cd_stats, unit='%')}")
    if dyn_cd_stats["wilcoxon_p"] is not None and dyn_cd_stats["wilcoxon_p"] >= 0.05:
        print("   Interpretation: not statistically significant at alpha=0.05 -- "
              "the CI above shows the range of plausible true differences given "
              f"only {dyn_cd_stats['n']} seeds; a CI straddling 0 with a "
              f"{dyn_cd_stats['effect_size']} rank-biserial effect means coverage_pct "
              "is not where C and D differ, not that they are proven identical.")

    write_csv(static_rows, dynamic_rows, static_summary, dynamic_summary,
              "wind_robustness_statistics.csv")

    dyn_gap_p = dyn_cd_stats["wilcoxon_p"]
    gap_is_significant = dyn_gap_p is not None and dyn_gap_p < 0.05

    refinement_rows = None
    if gap_is_significant:
        print(f"\n=== [6/6] Residual C-vs-D gap under dynamic wind is significant "
              f"(p={dyn_gap_p:.4f}, rank-biserial r={dyn_cd_stats['rank_biserial']}, "
              f"{dyn_cd_stats['effect_size']}) -> testing refinements E/F/H ===")
        refinement_rows, refinement_summary = _run_refinements_dynamic(
            seeds, steps, out_csv=_p("abcdefh_dynamic_raw.csv"), dynamic=True
        )
        plot_refinement_dynamic_gap(refinement_summary, "coverage_pct", "Coverage (%)",
                                     "refinement_dynamic_gap.png")
    else:
        print(f"\n=== [6/6] No significant residual C-vs-D gap under dynamic wind "
              f"(p={dyn_gap_p}, rank-biserial r={dyn_cd_stats['rank_biserial']}, "
              f"{dyn_cd_stats['effect_size']} effect) -> skipping refinement "
              f"comparison (E/F/H), per the module docstring's decision rule ===")

    print("\n=== [6b] Refinements E/F/H under STATIC wind too "
          "(for symmetry with the Dynamic column in metric_summary_grid.png) ===")
    static_refinement_rows, static_refinement_summary = _run_refinements_dynamic(
        seeds, steps, out_csv=_p("abcdefh_static_raw.csv"), dynamic=False
    )
    _combine_csvs(
        [(_p("abcdefh_dynamic_raw.csv"), "dynamic"), (_p("abcdefh_static_raw.csv"), "static")],
        _p("abcdefh_raw_combined.csv"),
    )

    plot_metric_summary_grid(static_rows, dynamic_rows,
                              dynamic_e_rows=refinement_rows,
                              static_e_rows=static_refinement_rows,
                              filename="metric_summary_grid.png")

    print("\n=== [7/7] Policy D rule-base-circularity robustness check "
          "(C vs original D vs D-indep, independent equal-weight rule base) ===")
    d_indep_static_rows, d_indep_static_summary = _run_d_indep_robustness_under_condition(
        seeds, steps, dynamic=False, out_csv=_p("policy_d_weight_robustness_static.csv")
    )
    d_indep_dynamic_rows, d_indep_dynamic_summary = _run_d_indep_robustness_under_condition(
        seeds, steps, dynamic=True, out_csv=_p("policy_d_weight_robustness_dynamic.csv")
    )
    plot_d_indep_robustness(d_indep_static_summary, "coverage_pct", "Coverage (%)",
                             "Static wind", "policy_d_weight_robustness_static.png")
    plot_d_indep_robustness(d_indep_dynamic_summary, "coverage_pct", "Coverage (%)",
                             "Dynamic wind", "policy_d_weight_robustness_dynamic.png")

    print("\n--- Efficiency companions: same rule-base-circularity check for "
          "relay delay & relay energy (D-indep turned out to be far MORE "
          "sensitive to its weight source on these two than on coverage_pct "
          "-- see interpretation below) ---")
    plot_d_indep_robustness(d_indep_static_summary, "avg_relay_delay_steps",
                             "Avg relay delay (steps)", "Static wind",
                             "policy_d_weight_robustness_delay_static.png")
    plot_d_indep_robustness(d_indep_dynamic_summary, "avg_relay_delay_steps",
                             "Avg relay delay (steps)", "Dynamic wind",
                             "policy_d_weight_robustness_delay_dynamic.png")
    plot_d_indep_robustness(d_indep_static_summary, "avg_relay_energy_pct",
                             "Avg relay energy (% battery)", "Static wind",
                             "policy_d_weight_robustness_energy_static.png")
    plot_d_indep_robustness(d_indep_dynamic_summary, "avg_relay_energy_pct",
                             "Avg relay energy (% battery)", "Dynamic wind",
                             "policy_d_weight_robustness_energy_dynamic.png")
    _combine_csvs(
        [(_p("policy_d_weight_robustness_static.csv"), "static"),
         (_p("policy_d_weight_robustness_dynamic.csv"), "dynamic")],
        _p("policy_d_weight_robustness_combined.csv"),
    )
    _combine_pngs(
        [_p("policy_d_weight_robustness_static.png"), _p("policy_d_weight_robustness_dynamic.png"),
         _p("policy_d_weight_robustness_delay_static.png"), _p("policy_d_weight_robustness_delay_dynamic.png"),
         _p("policy_d_weight_robustness_energy_static.png"), _p("policy_d_weight_robustness_energy_dynamic.png")],
        _p("policy_d_weight_robustness_combined.png"), ncols=2,
    )

    for cond_label, summ in (("static", d_indep_static_summary), ("dynamic", d_indep_dynamic_summary)):
        indep_p = (summ.get("coverage_pct", {}).get("C_vs_D_indep_weights") or {}).get("wilcoxon_p")
        if indep_p is not None and indep_p >= 0.05:
            print(f"   [{cond_label}] Interpretation: C_vs_D_indep_weights on coverage_pct is "
                  "NOT significant -- the 'no coverage cost' finding is robust to the "
                  "rule-base-circularity concern here; it does not depend on Policy D's "
                  "rule base sharing FUZZY_ATTR_WEIGHTS with RelayScore's own weights.")
        else:
            print(f"   [{cond_label}] Interpretation: C_vs_D_indep_weights on coverage_pct IS "
                  "significant -- part of the original 'no coverage cost' result under "
                  f"{cond_label} wind appears to depend on Policy D's rule base sharing "
                  "FUZZY_ATTR_WEIGHTS with RelayScore. Report this as a scope limitation "
                  "on Policy D rather than retrofitting a defense.")

        for metric, metric_label in (("avg_relay_delay_steps", "relay delay"),
                                      ("avg_relay_energy_pct", "relay energy")):
            sens_p = (summ.get(metric, {}).get("D_vs_D_indep_rule_base_sensitivity") or {}).get("wilcoxon_p")
            if sens_p is not None and sens_p < 0.05:
                print(f"   [{cond_label}] Interpretation: D_vs_D_indep_rule_base_sensitivity "
                      f"on {metric_label} IS significant -- unlike coverage, Policy D's "
                      f"{metric_label} output moves substantially when its rule base is "
                      "rebuilt from an independent equal-weight prior instead of "
                      "FUZZY_ATTR_WEIGHTS. Report this as a metric-specific scope "
                      "limitation: the rule-base-circularity concern is answered for "
                      "coverage_pct, but not fully answered for efficiency metrics.")

    print("\n=== [7b/8] Fleet-size sensitivity sweep "
          "(RelayScore, true FIS, TOPSIS, VIKOR across N = 2, 4, 8, 16 drones) ===")
    # NOTE: this was previously only wired into a2e_relay_fixed_final_51.py's own
    # __main__ block, which never runs when that module is imported as `relay`
    # here -- so it silently never produced fleet_size_sensitivity.csv /
    # fleet_size_scaling.png when running this file. Calling it explicitly here
    # fixes that. Full n=30 paired seeds, matching the terrain/wind sweep below;
    # shrink to seeds[:10] first for a quick sanity check.
    # As of this revision, run_fleet_size_sensitivity_analysis() defaults to
    # policies=("C","D","T","V") -- previously TOPSIS/VIKOR were only ever
    # tested at the single, fixed 8-drone fleet used elsewhere in this paper
    # (see Section V-I), leaving their own O(N)-vs-O(NR) scaling claim
    # untested at any other fleet size. This is now 4 fleet sizes x
    # len(seeds) x 4 policies simulate_fleet() calls (roughly double the
    # previous 2-policy runtime, dominated by D's FIS cost either way).
    fleet_rows, fleet_scaling_summary = relay.run_fleet_size_sensitivity_analysis(
        seeds,
        fleet_sizes=((1, 1), (2, 2), (4, 4), (8, 8)),
        max_steps=steps,
        out_csv=_p("fleet_size_sensitivity.csv"),
    )
    relay.plot_fleet_size_scaling(fleet_scaling_summary, save_path=_p("fleet_size_scaling.png"))

    print("\n=== [8/8] Terrain x Wind generalization sweep "
          "(RelayScore, true FIS, TOPSIS, VIKOR across multiple terrains x wind models) ===")
    # Full n=30 paired seeds (matches the paper's main experiments -- this is
    # the paper-grade run, not a quick sanity check). As of this revision,
    # _run_terrain_wind_sweep() defaults to policies=("C","D","T","V") -- this
    # is now 4 terrains x 3 wind modes x len(seeds) x 4 policies
    # simulate_fleet() calls (previously 2 policies; TOPSIS/VIKOR were only
    # ever tested on the single default terrain under static wind before this
    # revision). Still the slowest step in main() because of Policy D's FIS
    # cost -- shrink to seeds[:10] here first if you just want a fast sanity
    # check before committing to the full run.
    terrain_wind_seeds = seeds
    terrain_wind_rows, terrain_wind_summary = _run_terrain_wind_sweep(
        terrain_wind_seeds, steps, out_csv=_p("terrain_wind_generalization.csv"),
    )
    plot_terrain_wind_generalization(terrain_wind_summary, "coverage_pct", "Coverage (%)",
                                      "terrain_wind_generalization_coverage.png")
    plot_terrain_wind_generalization(terrain_wind_summary, "avg_relay_delay_steps",
                                      "Avg relay delay (steps)",
                                      "terrain_wind_generalization_delay.png")
    plot_terrain_wind_generalization(terrain_wind_summary, "avg_relay_energy_pct",
                                      "Avg relay energy (% battery)",
                                      "terrain_wind_generalization_energy.png")
    _combine_pngs(
        [_p("terrain_wind_generalization_coverage.png"), _p("terrain_wind_generalization_delay.png"),
         _p("terrain_wind_generalization_energy.png")],
        _p("terrain_wind_generalization_combined.png"), ncols=1,
    )

    print("\n=== [9/9] TOPSIS vs VIKOR vs fuzzy MADM (Policy D) vs RelayScore (C) ===")
    # Same seeds/steps as everything above -- fair, matched comparison.
    # T (TOPSIS) and V (VIKOR) are defined directly inside
    # a2e_relay_fixed_final_51.py, next to select_fuzzy_relay() (Policy D),
    # using the SAME 4 attributes and weights as D, so this isolates the
    # ranking METHOD rather than a different weighting choice.
    topsis_vikor_rows, topsis_vikor_summary = relay.run_topsis_vikor_comparison(
        seeds, max_steps=steps, out_csv=_p("topsis_vikor_comparison.csv"),
    )

    final_files = sorted(os.listdir(OUT_DIR))
    print(f"\n🎉 Wind + robustness figures saved under: {os.path.abspath(OUT_DIR)}")
    print(f"   {len(final_files)} output file(s) (KEEP_INDIVIDUAL_FILES={KEEP_INDIVIDUAL_FILES}):")
    for fn in final_files:
        print(f"     - {fn}")


# ══════════════════════════════════════════════════════════════
# SURROGATE VALIDATION -- True Mamdani FIS (D) vs. Surrogate LUT FIS (D2)
#
# Goal: show coverage/staleness/energy differences between D and D2 are
# "n.s." (not significant) -- i.e. the surrogate reproduces D's decision
# quality -- while running at RelayScore's (C's) O(N) speed. Mirrors the
# same pattern as relay.run_topsis_vikor_comparison() / run_four_way_
# comparison() (same simulate_fleet() calls, same CSV export, same
# _safe_wilcoxon + _print_significance_summary reporting), just with
# C/D/D2 instead of C/D/T/V.
#
# Requires build_surrogate_fis.py to have been run first, so that
# luts/lut_shared_res9.npz (next to surrogate_fis.py) exists.
# ══════════════════════════════════════════════════════════════
def _decision_log_agreement(d_log, d2_log):
    """Per-decision comparison of two decision_log lists (same seed, same
    starting state, D vs D2) -- the ground-truth complement to the seed-
    level metric comparison above.

    Matches log entries by (step, needer_id) rather than by list position:
    as long as D and D2 haven't diverged yet, the same drone requests a
    relay at the same step in both runs, so this key is stable. Once a
    decision actually differs, the two simulations' fleets start
    occupying different positions/battery levels, so later needer/step
    pairs stop lining up on their own -- that's not a bug in the
    comparison, it's the real downstream consequence of the earlier
    divergence, and is exactly what `first_divergence_step` is meant to
    surface.
    """
    d2_by_key = {}
    for e in d2_log:
        d2_by_key.setdefault((e["step"], e["needer_id"]), e["chosen_candidate_id"])

    matched = 0
    agreed = 0
    first_divergence_step = None
    for e in d_log:
        key = (e["step"], e["needer_id"])
        if key in d2_by_key:
            matched += 1
            if d2_by_key[key] == e["chosen_candidate_id"]:
                agreed += 1
            elif first_divergence_step is None:
                first_divergence_step = e["step"]
        elif first_divergence_step is None:
            # D made a relay decision at (step, needer_id) that D2's
            # trajectory never reached -- itself a symptom of an earlier
            # divergence (or, if this is the very first entry, an
            # immediate one).
            first_divergence_step = e["step"]

    return {
        "d_decisions": len(d_log),
        "d2_decisions": len(d2_log),
        "matched_pairs": matched,
        "agreements": agreed,
        "agreement_rate": round(agreed / matched, 4) if matched else None,
        "first_divergence_step": first_divergence_step,
        "fully_identical": (len(d_log) == len(d2_log) == matched == agreed),
    }


def run_surrogate_comparison(seeds, max_steps=None, num_drones=relay.NUM_DRONES,
                              num_reserves=relay.RESERVE_POOL_SIZE, out_csv=None,
                              decision_log_csv=None):
    """
    Compares the True Mamdani FIS (D) against the Surrogate LUT FIS (D2).
    Goal: Prove that coverage/staleness differences are "n.s." (not significant),
    while keeping the O(N) runtime speedup.

    Also computes true per-decision agreement between D and D2 (not just
    seed-level mission-outcome agreement) -- see _decision_log_agreement().
    Pass `decision_log_csv` to save the per-seed breakdown; the aggregate
    is always printed regardless.
    """
    max_steps = max_steps or a20.MAX_STEPS
    rows = []
    decision_rows = []

    print("\n=== RUNNING SURROGATE vs TRUE FIS VALIDATION (D vs D2) ===")
    for seed in seeds:
        m_c = simulate_fleet(seed, relay_selector=select_best_relay,
                              num_drones=num_drones, num_reserves=num_reserves, max_steps=max_steps)
        m_d = simulate_fleet(seed, relay_selector=select_fuzzy_relay_true_fis,
                              num_drones=num_drones, num_reserves=num_reserves, max_steps=max_steps)
        m_d2 = simulate_fleet(seed, relay_selector=select_fuzzy_relay_surrogate_fis,
                               num_drones=num_drones, num_reserves=num_reserves, max_steps=max_steps)

        row = {"seed": seed}
        for label, m in (("C", m_c), ("D", m_d), ("D2", m_d2)):
            for k, v in m.items():
                if k != "decision_log":
                    row[f"{label}_{k}"] = v
        rows.append(row)

        d_log = m_d.get("decision_log", [])
        d2_log = m_d2.get("decision_log", [])
        agree = _decision_log_agreement(d_log, d2_log)
        decision_rows.append({"seed": seed, **agree})

        agree_pct = f"{100*agree['agreement_rate']:.1f}%" if agree["agreement_rate"] is not None else "n/a"
        print(f"[seed {seed}] "
              f"C(RelayScore)={m_c['coverage_pct']}% | "
              f"D(True FIS)={m_d['coverage_pct']}% | "
              f"D2(Surrogate FIS)={m_d2['coverage_pct']}% | "
              f"decision-agreement={agree_pct} "
              f"({agree['agreements']}/{agree['matched_pairs']} matched, "
              f"first divergence @ step {agree['first_divergence_step']})")

    if out_csv:
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    if decision_log_csv and decision_rows:
        with open(decision_log_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(decision_rows[0].keys()))
            writer.writeheader()
            writer.writerows(decision_rows)

    # Aggregate decision-level agreement across every seed -- this is the
    # ground-truth number the seed-level Wilcoxon "n.s." result upstream
    # is a downstream consequence of.
    total_matched = sum(r["matched_pairs"] for r in decision_rows)
    total_agreed = sum(r["agreements"] for r in decision_rows)
    total_d_decisions = sum(r["d_decisions"] for r in decision_rows)
    n_fully_identical = sum(1 for r in decision_rows if r["fully_identical"])
    overall_rate = (total_agreed / total_matched) if total_matched else None

    print("\n=== DECISION-LEVEL AGREEMENT (D vs D2, ground truth) ===")
    print(f"  seeds with a fully identical decision trace: {n_fully_identical}/{len(seeds)}")
    print(f"  total individual relay decisions (D):        {total_d_decisions}")
    print(f"  matched (step, needer) pairs compared:        {total_matched}")
    print(f"  agreements among matched pairs:                {total_agreed}")
    if overall_rate is not None:
        print(f"  overall per-decision agreement rate:           {100*overall_rate:.2f}%")
    diverged_seeds = [r["seed"] for r in decision_rows if not r["fully_identical"]]
    if diverged_seeds:
        print(f"  seeds with >=1 divergence: {diverged_seeds}")
        for r in decision_rows:
            if not r["fully_identical"]:
                print(f"    seed {r['seed']}: first divergence @ step {r['first_divergence_step']}, "
                      f"{r['agreements']}/{r['matched_pairs']} matched pairs still agreed")

    summary = {}
    for metric in PRIMARY_METRICS + SECONDARY_METRICS:
        vals = {"C": [r.get(f"C_{metric}") for r in rows],
                "D": [r.get(f"D_{metric}") for r in rows],
                "D2": [r.get(f"D2_{metric}") for r in rows]}
        entry = {}
        for label in ("C", "D", "D2"):
            clean = [v for v in vals[label] if isinstance(v, (int, float))]
            entry[f"{label}_mean"] = round(statistics.mean(clean), 3) if clean else None

        entry["true_vs_surrogate_D_vs_D2"] = relay._safe_wilcoxon(vals["D"], vals["D2"])
        entry["relayscore_vs_surrogate_C_vs_D2"] = relay._safe_wilcoxon(vals["C"], vals["D2"])
        summary[metric] = entry

    summary["_decision_level_agreement"] = {
        "seeds_fully_identical": n_fully_identical,
        "n_seeds": len(seeds),
        "total_matched_pairs": total_matched,
        "total_agreements": total_agreed,
        "overall_agreement_rate": round(overall_rate, 4) if overall_rate is not None else None,
    }

    print("\n=== D vs D2 SUMMARY ===")
    print(json.dumps(summary, indent=2, default=str))

    relay._print_significance_summary(
        summary,
        comparisons=[
            ("true_vs_surrogate_D_vs_D2", "D", "D2"),
            ("relayscore_vs_surrogate_C_vs_D2", "C", "D2"),
        ],
        n=len(seeds),
    )
    return rows, summary, decision_rows


if __name__ == "__main__":
    # `python Relay_System.py`                       -> full pipeline (main()), unchanged.
    # `python Relay_System.py surrogate [steps] [seeds]` -> ONLY the D-vs-D2
    #     surrogate-FIS validation (fast, doesn't touch the rest of the
    #     figures/CSVs main() produces). Kept as an opt-in mode rather than
    #     replacing main() outright, since main() is the full thesis pipeline
    #     (sections [1/9]-[9/9], every figure/CSV this codebase produces) --
    #     silently skipping it here would remove everything except D vs D2.
    if len(sys.argv) > 1 and sys.argv[1] == "surrogate":
        steps = int(sys.argv[2]) if len(sys.argv) > 2 else 500
        num_seeds = int(sys.argv[3]) if len(sys.argv) > 3 else 30
        seeds = list(range(1, num_seeds + 1))
        run_surrogate_comparison(seeds, max_steps=steps, out_csv=_p("surrogate_validation.csv"),
                                  decision_log_csv=_p("surrogate_decision_agreement.csv"))
    else:
        main()