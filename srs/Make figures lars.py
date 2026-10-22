# -*- coding: utf-8 -*-
"""
make_figures_lars.py

Draws the five LARS-centred figures of the revised paper from the numbers printed
in the paper's tables (means and paired-t 95% CIs).

    pareto_fresh_lars.pdf     coverage vs delay / energy, fresh block (201-400), 13 policies
    pareto_confirm_lars.pdf   same, confirmatory block (401-600)
    reserve_share_lars.pdf    idle-reserve share vs coverage / delay / energy (fresh block)
    forest_confirm_lars.pdf   LARS minus comparator, confirmatory block, 95% CIs
    forest_fresh_lars.pdf     LARS minus comparator, fresh block, 95% CIs

Means carry no error bars here (per-seed standard errors are in your CSV files).
Usage:  python make_figures_lars.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

GROUPS = {
    "LARS": ("LARS", "#1b7f3b"),
    "J": ("J (extension)", "#0f8b8d"),
    "LARS-tuned": ("LARS-tuned", "#66a61e"),
    "D": ("FIS/TOPSIS/VIKOR, D's weights", "#2b6cb0"),
    "D2": ("FIS/TOPSIS/VIKOR, D's weights", "#2b6cb0"),
    "T": ("FIS/TOPSIS/VIKOR, D's weights", "#2b6cb0"),
    "V": ("FIS/TOPSIS/VIKOR, D's weights", "#2b6cb0"),
    "T-tuned": ("tuned T, V and AUC", "#7b3fa0"),
    "V-tuned": ("tuned T, V and AUC", "#7b3fa0"),
    "AUC": ("tuned T, V and AUC", "#7b3fa0"),
    "C": ("RelayScore (C)", "#e08a00"),
    "B": ("heuristics (B, BR)", "#c0262d"),
    "BR": ("heuristics (B, BR)", "#c0262d"),
}
LEGEND = ["LARS", "J (extension)", "LARS-tuned", "FIS/TOPSIS/VIKOR, D's weights",
          "tuned T, V and AUC", "RelayScore (C)", "heuristics (B, BR)"]
COLORS = {GROUPS[k][0]: GROUPS[k][1] for k in GROUPS}

# block means: coverage %, delay (steps), energy %
FRESH = {
    "LARS": (90.04, 12.83, 8.92), "J": (90.33, 13.43, 9.80), "C": (89.44, 14.50, 10.77),
    "D": (90.34, 15.07, 12.63), "D2": (90.24, 15.00, 12.61), "T": (90.53, 15.47, 12.20),
    "V": (90.40, 16.45, 13.09), "T-tuned": (92.73, 17.55, 13.85), "V-tuned": (92.30, 18.48, 14.72),
    "AUC": (89.70, 17.00, 11.94), "LARS-tuned": (89.87, 12.28, 10.10), "B": (89.95, 11.14, 9.54),
    "BR": (92.93, 19.39, 16.03)}
CONF = {
    "LARS": (90.17, 12.77, 8.76), "J": (89.72, 13.49, 9.73), "C": (90.29, 14.53, 10.79),
    "D": (89.83, 15.12, 12.53), "D2": (89.93, 15.08, 12.53), "T": (90.53, 15.39, 12.21),
    "V": (90.35, 16.24, 13.03), "T-tuned": (92.39, 17.37, 13.73), "V-tuned": (92.96, 18.49, 14.55),
    "AUC": (89.84, 16.71, 11.78), "LARS-tuned": (90.88, 12.34, 10.12), "B": (89.69, 11.30, 9.67),
    "BR": (93.10, 19.10, 15.70)}
RESERVE = {"LARS": 42.2, "J": 49.6, "C": 40.3, "D": 66.2, "D2": 65.7, "T": 55.9, "V": 68.0,
           "T-tuned": 89.1, "V-tuned": 92.6, "AUC": 58.1, "B": 44.2, "BR": 100.0}

# LARS minus comparator: (delta, lo, hi) for coverage, delay, energy
FOREST_CONF = {
    "C":          [(-0.13, -1.05, 0.80), (-1.76, -2.12, -1.40), (-2.03, -2.38, -1.68)],
    "D":          [(0.34, -0.65, 1.33), (-2.35, -2.73, -1.97), (-3.77, -4.14, -3.40)],
    "D2":         [(0.24, -0.75, 1.23), (-2.30, -2.67, -1.93), (-3.77, -4.13, -3.40)],
    "T":          [(-0.36, -1.31, 0.59), (-2.62, -2.99, -2.24), (-3.45, -3.81, -3.09)],
    "V":          [(-0.19, -1.21, 0.84), (-3.47, -3.86, -3.08), (-4.27, -4.62, -3.92)],
    "AUC":        [(0.33, -0.64, 1.30), (-3.93, -4.38, -3.49), (-3.02, -3.44, -2.61)],
    "B":          [(0.48, -0.40, 1.35), (1.48, 1.15, 1.80), (-0.91, -1.21, -0.61)],
    "J":          [(0.44, -0.42, 1.31), (-0.71, -1.09, -0.34), (-0.97, -1.32, -0.62)],
    "LARS-tuned": [(-0.71, -1.59, 0.16), (0.43, 0.11, 0.75), (-1.36, -1.67, -1.04)],
    "T-tuned":    [(-2.22, -3.04, -1.41), (-4.60, -4.96, -4.24), (-4.97, -5.32, -4.62)],
    "V-tuned":    [(-2.79, -3.67, -1.91), (-5.71, -6.09, -5.34), (-5.79, -6.14, -5.43)],
    "BR":         [(-2.93, -3.88, -1.98), (-6.32, -6.73, -5.92), (-6.94, -7.30, -6.58)],
}
FOREST_FRESH = {
    "C":          [(0.59, -0.49, 1.68), (-1.67, -2.02, -1.32), (-1.85, -2.18, -1.52)],
    "D":          [(-0.30, -1.45, 0.84), (-2.24, -2.59, -1.89), (-3.71, -4.05, -3.38)],
    "D2":         [(-0.20, -1.31, 0.91), (-2.17, -2.53, -1.82), (-3.69, -4.03, -3.35)],
    "T":          [(-0.50, -1.51, 0.51), (-2.64, -3.04, -2.25), (-3.28, -3.65, -2.91)],
    "V":          [(-0.36, -1.31, 0.59), (-3.62, -3.98, -3.26), (-4.18, -4.53, -3.82)],
    "AUC":        [(0.34, -0.76, 1.43), (-4.18, -4.60, -3.76), (-3.03, -3.45, -2.60)],
    "B":          [(0.09, -0.95, 1.14), (1.68, 1.35, 2.01), (-0.63, -0.96, -0.29)],
    "J":          [(-0.30, -1.24, 0.65), (-0.61, -0.98, -0.24), (-0.88, -1.23, -0.54)],
    "LARS-tuned": [(0.17, -0.90, 1.23), (0.55, 0.20, 0.89), (-1.18, -1.50, -0.85)],
    "T-tuned":    [(-2.69, -3.59, -1.79), (-4.73, -5.10, -4.35), (-4.93, -5.31, -4.56)],
    "V-tuned":    [(-2.27, -3.18, -1.35), (-5.66, -6.03, -5.28), (-5.80, -6.18, -5.42)],
}


def dominated(P, a, b):
    ca, da, ea = P[a]
    cb, db, eb = P[b]
    return ca >= cb and da <= db and ea <= eb and (ca > cb or da < db or ea < eb)


def pareto(P, fname, title):
    nondom = {n for n in P if not any(dominated(P, m, n) for m in P if m != n)}
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.0))
    for xi, ax, xl in ((1, axs[0], "Relay delay (steps)  \u2193"), (2, axs[1], "Relay energy (%)  \u2193")):
        for n, v in P.items():
            col = COLORS[GROUPS[n][0]]
            filled = n in nondom
            big = 110 if n == "LARS" else 62
            ax.scatter(v[xi], v[0], s=big, zorder=3, facecolor=col if filled else "white",
                       edgecolor=col, linewidth=1.6)
            ax.annotate(n, (v[xi], v[0]), xytext=(v[xi] + 0.12, v[0] + 0.05), fontsize=8)
        ax.set_xlabel(xl)
        ax.set_ylabel("Coverage (%)")
        ax.grid(alpha=0.3)
    axs[0].set_title(f"(a) Coverage vs delay, {title}", fontsize=10)
    axs[1].set_title(f"(b) Coverage vs energy, {title}", fontsize=10)
    handles = [plt.Line2D([], [], marker="o", ls="", markerfacecolor=COLORS[g], markeredgecolor=COLORS[g],
                          label=g) for g in LEGEND]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8, frameon=False)
    fig.text(0.5, 0.135, "Filled = not dominated on means (coverage, delay, energy); hollow = dominated.",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.18, 1, 1))
    fig.savefig(fname)
    plt.close(fig)
    return sorted(nondom)


def reserve_figure(fname):
    names = list(RESERVE)
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.5))
    labels = ["Coverage (%)", "Relay delay (steps)", "Relay energy (%)"]
    out = []
    for k, ax in enumerate(axs):
        xs = np.array([RESERVE[n] for n in names])
        ys = np.array([FRESH[n][k] for n in names])
        rho, p = spearmanr(xs, ys)
        out.append((round(rho, 2), p))
        for n in names:
            col = COLORS[GROUPS[n][0]]
            ax.scatter(RESERVE[n], FRESH[n][k], s=90 if n == "LARS" else 55, color=col, zorder=3)
            ax.annotate(n, (RESERVE[n], FRESH[n][k]), xytext=(RESERVE[n] + 1.0, FRESH[n][k]), fontsize=7.5)
        ax.set_xlabel("Relays served by an idle reserve (%)")
        ax.set_ylabel(labels[k])
        ptxt = "p<0.0001" if p < 1e-4 else f"p={p:.4f}"
        ax.set_title(f"({'abc'[k]}) Spearman \u03c1={rho:.2f}, {ptxt}", fontsize=9)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fname)
    plt.close(fig)
    return out


def forest(F, fname, note):
    order = list(F)[::-1]
    fig, axs = plt.subplots(1, 3, figsize=(11, 4.4), sharey=True)
    titles = ("Coverage (pts)  LARS higher = better", "Relay delay (steps)  LARS lower = better",
              "Relay energy (pts)  LARS lower = better")
    for k, ax in enumerate(axs):
        for i, n in enumerate(order):
            d, lo, hi = F[n][k]
            excl = lo > 0 or hi < 0
            ax.errorbar(d, i, xerr=[[d - lo], [hi - d]], fmt="o", capsize=3,
                        color="#1b7f3b" if excl else "#555555")
        ax.axvline(0, color="k", lw=0.8)
        if k == 0:
            ax.axvline(-2, color="#c0262d", ls="--", lw=1)
            ax.axvline(2, color="#c0262d", ls="--", lw=1)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([f"LARS \u2212 {n}" for n in order])
        ax.set_title(titles[k], fontsize=9)
        ax.grid(alpha=0.3, axis="x")
    fig.text(0.5, 0.005, note, ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(fname)
    plt.close(fig)


if __name__ == "__main__":
    print("fresh non-dominated:", pareto(FRESH, "pareto_fresh_lars.pdf", "fresh block"))
    print("confirmatory non-dominated:", pareto(CONF, "pareto_confirm_lars.pdf", "confirmatory block"))
    print("spearman (fresh, 12 policies):", reserve_figure("reserve_share_lars.pdf"))
    base = ("Mean paired difference with 95% CI (paired-t), n=200 seeds. Dashed: \u00b12.0-pt equivalence "
            "margin. Green: CI excludes 0. AUC at \u03bb=1600.")
    forest(FOREST_CONF, "forest_confirm_lars.pdf", "Confirmatory block (seeds 401-600). " + base)
    forest(FOREST_FRESH, "forest_fresh_lars.pdf", "Fresh block (seeds 201-400). " + base)
    print("written: pareto_fresh_lars.pdf, pareto_confirm_lars.pdf, reserve_share_lars.pdf, "
          "forest_confirm_lars.pdf, forest_fresh_lars.pdf")