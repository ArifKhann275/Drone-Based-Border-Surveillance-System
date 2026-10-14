"""
Pareto plot (coverage vs delay, coverage vs energy) with Error Bars (SEM) 
on the fresh block using per-seed data from CSV.
Run: python pareto_plot.py -> pareto_fresh_with_errorbars.pdf / .png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# CSV ফাইল থেকে ডেটা লোড করা (আপনার ফ্রেশ ব্লকের ফাইল নাম এখানে দিন, যেমন 'main_comparison.csv')
CSV_FILE = "main_comparison.csv" 
COV_TOL = 0.0  # কভারেজ টলারেন্স (প্রয়োজন হলে ০.৫ বা অন্য কিছু দিতে পারেন)

try:
    df = pd.read_csv(CSV_FILE)
except FileNotFoundError:
    print(f"❌ Error: {CSV_FILE} পাওয়া যায়নি। দয়া করে সঠিক CSV ফাইলের নাম দিন।")
    exit()

# পলিসি এবং তাদের লেবেল ম্যাপিং (আপনার রান করা লেবেল অনুযায়ী এডজাস্ট করে নিতে পারেন)
# যেমন নথিতে লেবেলগুলো ছিল: J, I, C, D, D2, T, V, B, BR
POLICIES = {
    "J": ("proposed", "#1b7a3d"),
    "I": ("ablation", "#e08a00"),
    "C": ("ablation", "#e08a00"),
    "D": ("madm", "#2b6cb0"),
    "D2": ("madm", "#2b6cb0"),
    "T": ("madm", "#2b6cb0"),
    "V": ("madm", "#2b6cb0"),
    "B": ("heuristic", "#c53030"),
    "BR": ("heuristic", "#c53030"),
}

summary_data = {}

for n in POLICIES.keys():
    # CSV-তে কলামের নাম সাধারণত label_metric এভাবে থাকে (যেমন: J_coverage_pct)
    cov_col = f"{n}_coverage_pct"
    delay_col = f"{n}_avg_relay_delay_steps"
    energy_col = f"{n}_avg_relay_energy_pct"
    
    if cov_col in df.columns:
        covs = df[cov_col].dropna()
        delays = df[delay_col].dropna() if delay_col in df.columns else [0]
        energies = df[energy_col].dropna() if energy_col in df.columns else [0]
        
        # Mean এবং SEM (Standard Error of the Mean) হিসাব
        cov_mean, cov_sem = covs.mean(), covs.sem()
        delay_mean, delay_sem = delays.mean(), delays.sem()
        energy_mean, energy_sem = energies.mean(), energies.sem()
        
        summary_data[n] = {
            "cov": (cov_mean, cov_sem),
            "delay": (delay_mean, delay_sem),
            "energy": (energy_mean, energy_sem),
            "group": POLICIES[n][0],
            "color": POLICIES[n][1]
        }

# প্যারেটো ডমিন্যান্স চেক (Means এর ওপর ভিত্তি করে)
def dominated(a, b, tol=COV_TOL):
    ca, da, ea = a["cov"][0], a["delay"][0], a["energy"][0]
    cb, db, eb = b["cov"][0], b["delay"][0], b["energy"][0]
    ge = cb >= ca - tol and db <= da and eb <= ea
    strict = (cb > ca + tol) or db < da or eb < ea
    return ge and strict

nd = {n for n, a in summary_data.items() 
      if not any(dominated(a, b) for m, b in summary_data.items() if m != n)}

print("Non-dominated (tol=%.2f):" % COV_TOL, sorted(nd))

# প্লটিং শুরু
fig, axes = plt.subplots(1, 2, figsize=(8, 3.8))

for ax, (metric_key, xl) in zip(axes, [("delay", "Relay delay (steps) ↓"), ("energy", "Relay energy (%) ↓")]):
    for n, data in summary_data.items():
        filled = n in nd
        x_m, x_s = data[metric_key]
        y_m, y_s = data["cov"]
        color = data["color"]
        
        # Error bar সহ স্ক্যাটার প্লট
        ax.errorbar(x_m, y_m, xerr=x_s, yerr=y_s, fmt='o', color=color,
                    mfc=color if filled else "none", mec=color, mew=1.5,
                    ms=6, capsize=3, elinewidth=1, zorder=3)
        
        ax.annotate(n, (x_m, y_m), xytext=(4, 3), textcoords="offset points", fontsize=8)
        
    ax.set_xlabel(xl)
    ax.set_ylabel("Coverage (%) ↑")
    ax.grid(alpha=.3)

axes[0].set_title("(a) Coverage vs delay (with SEM)", fontsize=9)
axes[1].set_title("(b) Coverage vs energy (with SEM)", fontsize=9)

fig.text(0.5, -0.04, "Error bars represent Standard Error of the Mean (SEM, n=200). Filled = non-dominated on means.",
         ha="center", fontsize=7.5)
fig.tight_layout()
fig.savefig("pareto_fresh_with_errorbars.pdf", bbox_inches="tight")
fig.savefig("pareto_fresh_with_errorbars.png", dpi=300, bbox_inches="tight")
print("✅ নতুন গ্রাফ সফলভাবে সেভ হয়েছে: pareto_fresh_with_errorbars.pdf / .png")