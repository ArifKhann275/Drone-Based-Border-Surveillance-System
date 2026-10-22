# LARS: A Lightweight Absolute-Scale Relay Selection Policy for Multi-UAV Surveillance Coverage

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Reproducibility](https://img.shields.io/badge/Reproducibility-Verified-brightgreen.svg)]()

Official simulation framework, benchmark datasets, and reproducibility suite for the paper:  
**"LARS: A Lightweight Absolute-Scale Relay Selection Policy for Multi-UAV Surveillance Coverage"**

---

## 📌 Overview

Continuous multi-UAV border and perimeter surveillance requires seamless **relay handoffs** when patrolling drones deplete their batteries. Existing lightweight multi-attribute decision-making (MADM) selectors—such as **RelayScore (Policy C)**, **TOPSIS (Policy T)**, and **VIKOR (Policy V)**—rely on *pool-relative min–max normalization* across the candidate pool at each handoff decision.

In small candidate pools ($N_c \ge 2$), when all feasible candidates exhibit similarly low coverage loss, pool-relative normalization stretches negligible differences across the full $[0, 1]$ range (**scale inflation**), violating the *Independence of Irrelevant Alternatives (IIA)* and unnecessarily dispatching distant idle reserves.

**LARS (Lightweight Absolute-Scale Relay Selection)** resolves this pathology with a single structural change: replacing pool-relative min–max normalization on coverage loss with a **piecewise-linear absolute scale** anchored to a physical saturation ceiling ($C_{\max} = 0.60$), while retaining parameter-free relative scaling for travel energy and relay delay.

### Key Highlights
* **Zero Coverage Loss Penalty:** Achieves statistically equivalent surveillance coverage ($\pm 2.0\%$ TOST margin) to Mamdani FIS (**D**), Lookup-Table Surrogate FIS (**D2**), **TOPSIS**, **VIKOR**, and Marginal-Cost Auction (**AUC**).
* **Substantial Efficiency Gains:** Reduces relay handoff delay and relay travel energy significantly across both the **Fresh Block** ($n=200$, seeds 201–400) and **Confirmatory Block** ($n=200$, seeds 401–600).
* **Sub-Millisecond Runtime:** Operates in $\mathcal{O}(N_c)$ time independent of rule-base size, avoiding the $\mathcal{O}(N_c \cdot R)$ scaling bottleneck of fuzzy inference systems.
* **Environmental Generalization:** Validated across 4 real SRTM terrain sectors (*Benapole Delta, Cox's Bazar Coastal, Sylhet Haor/Hills, Chittagong Hill Tracts*), 3 wind regimes (*Static, Dynamic, Stochastic*), and multiple fleet sizes ($N \in \{4, 6, 7, 8, 12, 16\}$).

---

## 📂 Repository Structure

```text
├── src/                                     # Core simulation engine, policies, and experiment runners
│   ├── a20.py                               # Base UAV kinematics, battery model, and SRTM terrain loader
│   ├── a2e_relay_fixed_final_51.py          # Core multi-UAV relay handoff simulator & policy registry
│   ├── policy_d_true_fis.py                 # Policy D: Vectorized Mamdani Fuzzy Inference System (FIS) & independent rule bases
│   ├── surrogate_fis.py                     # Policy D2: Fast 4-D multilinear lookup-table (LUT) surrogate FIS
│   ├── build_surrogate_fis.py               # Builder script for Policy D2 LUT (.npz) with rule-base fingerprint guard
│   ├── auction_baseline.py                  # Policy AUC: Marginal-cost auction baseline selector
│   ├── policy_heuristics.py                 # Heuristic baselines (Nearest Feasible 'B', Nearest Reserve 'BR', etc.)
│   ├── policy_j_full.py                     # Policy J: 4-attribute VIKOR-style compromise extension
│   ├── run_mechanism_test.py                # Mechanism instrumentation: raw spread, inflation factor, and rank flips
│   ├── run_generalization_sweep.py          # Terrain × Wind and Fleet-Size generalization sweeps
│   ├── run_mission_benefit_test.py          # Stressed fleet (N=6, N=7) mission-level handoff gap evaluation
│   ├── run_long_horizon_test.py             # Extended 800-step mission horizon stress test (n=200 seeds)
│   ├── tune_baselines.py                    # Differential Evolution (DE) coverage tuning for T, V, and LARS
│   ├── tune_auction_lambda.py               # Grid calibration for auction price parameter (lambda)
│   ├── tune_multiobj_energy.py              # Multi-objective DE tuning (Common Cost J_MO) for T, V, and LARS
│   ├── tune_multiobj_auc.py                 # Multi-objective grid tuning for Auction baseline
│   └── Make figures lars.py                 # Publication figure generator (Pareto fronts, Forest plots, Reserve share)
│
├── results/                                 # Pre-computed raw per-seed CSV logs, tuned JSON weights, and PDF/PNG figures
│   ├── main_comparison.csv                  # Fresh block (seeds 201–400) evaluation across all policies
│   ├── mechanism_spread_and_flips.csv       # Per-decision pool diagnostics and rank-reversal records
│   ├── mechanism_summary_by_Nc.csv          # Aggregated inflation factor and flip rates by pool size N_c
│   ├── terrain_wind_sweep.csv               # 4 Terrains × 3 Wind modes generalization dataset
│   ├── fleet_size_sweep.csv                 # Fleet-size scaling dataset
│   ├── mission_benefit_stressed_N6_N7.csv   # Reduced-fleet (N=6, N=7) cumulative unattended zone-steps
│   ├── long_horizon_standard_800_steps.csv  # 800-step extended horizon results (seeds 801–1000)
│   ├── multiobj_comparison_701_900.csv      # Multi-objective tuned policies evaluation (seeds 701–900)
│   ├── dindep_circularity_bh.csv            # Benjamini-Hochberg corrected circularity test across independent FIS rule bases
│   ├── forest_confirm_lars.pdf              # Forest plot of paired differences on Confirmatory Block (seeds 401–600)
│   ├── forest_fresh_lars.pdf                # Forest plot of paired differences on Fresh Block (seeds 201–400)
│   └── reserve_share_lars.pdf               # Spearman correlation: Idle reserve usage share vs. Coverage/Delay/Energy
│
└── papers/                                  # Manuscript and Supplementary Material
