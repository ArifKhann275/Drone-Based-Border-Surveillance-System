# -*- coding: utf-8 -*-

import os
import sys
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import a2e_relay_fixed_final_51 as relay
import a20

OUT_DIR = os.path.join("results", "relay")
os.makedirs(OUT_DIR, exist_ok=True)


def _p(filename):
    return os.path.join(OUT_DIR, filename)


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
# 2) STATIC vs DYNAMIC ROBUSTNESS COMPARISON
# ══════════════════════════════════════════════════════════════
def _run_bc_under_condition(seeds, max_steps, dynamic):
    a20.WIND_DYNAMIC_ENABLED = dynamic
    rows, summary = relay.run_comparison(seeds, max_steps=max_steps)
    a20.WIND_DYNAMIC_ENABLED = False  # always leave the module in the safe default state
    return rows, summary


def plot_robustness(static_summary, dynamic_summary, metrics, filename):
    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 5))
    if len(metrics) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        s = static_summary.get(metric, {})
        d = dynamic_summary.get(metric, {})
        labels = ["Static wind\n(Baseline)", "Static wind\n(RelayScore)",
                  "Dynamic wind\n(Baseline)", "Dynamic wind\n(RelayScore)"]
        values = [s.get("baseline_mean"), s.get("optimized_mean"),
                  d.get("baseline_mean"), d.get("optimized_mean")]
        colors = ["#a6bddb", "#2b8cbe", "#fdd0a2", "#e6550d"]

        bars = ax.bar(labels, values, color=colors, width=0.6)
        for bar, v in zip(bars, values):
            if v is not None:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                         f"{v:.2f}", ha="center", va="bottom", fontsize=9)

        p_static = s.get("wilcoxon_p")
        p_dynamic = d.get("wilcoxon_p")
        subtitle = (f"static p={p_static}" if p_static is not None else "static p=n/a") + \
                   "  |  " + \
                   (f"dynamic p={p_dynamic}" if p_dynamic is not None else "dynamic p=n/a")
        ax.set_title(f"{metric}\n({subtitle})", fontsize=10)
        ax.tick_params(axis="x", labelsize=8)
        ax.grid(axis="y", alpha=0.25)

    plt.suptitle("RelayScore Advantage: Static vs Dynamic Wind (Robustness Check)",
                  fontweight="bold")
    plt.tight_layout()
    plt.savefig(_p(filename), dpi=300)
    plt.close(fig)
    print(f"✅ Saved {filename}")


def write_csv(static_summary, dynamic_summary, filename):
    rows = []
    all_metrics = set(static_summary) | set(dynamic_summary)
    for metric in sorted(all_metrics):
        s = static_summary.get(metric, {})
        d = dynamic_summary.get(metric, {})
        rows.append({
            "metric": metric,
            "static_baseline_mean": s.get("baseline_mean"),
            "static_optimized_mean": s.get("optimized_mean"),
            "static_wilcoxon_p": s.get("wilcoxon_p"),
            "dynamic_baseline_mean": d.get("baseline_mean"),
            "dynamic_optimized_mean": d.get("optimized_mean"),
            "dynamic_wilcoxon_p": d.get("wilcoxon_p"),
        })
    with open(_p(filename), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"✅ Saved {filename}  ({len(rows)} rows)")


def main():
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    num_seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    seeds = list(range(1, num_seeds + 1))

    print(f"Running with max_steps={steps}, num_seeds={num_seeds}\n")

    print("=== [1/3] Wind profile figure ===")
    plot_wind_profile(steps)

    print("\n=== [2/3] B vs C under STATIC wind (regression baseline) ===")
    _, static_summary = _run_bc_under_condition(seeds, steps, dynamic=False)

    print("\n=== [3/3] B vs C under DYNAMIC wind (robustness check) ===")
    _, dynamic_summary = _run_bc_under_condition(seeds, steps, dynamic=True)

    key_metrics = ["coverage_pct", "total_energy_consumed", "avg_relay_energy_pct"]
    plot_robustness(static_summary, dynamic_summary, key_metrics,
                     "robustness_static_vs_dynamic.png")
    write_csv(static_summary, dynamic_summary, "wind_robustness_statistics.csv")

    print(f"\n🎉 Wind + robustness figures saved under: {os.path.abspath(OUT_DIR)}")


if __name__ == "__main__":
    main()
