# -*- coding: utf-8 -*-
"""run_indep_panel.py -- run ONLY the independent rule-base panel (fast path).

    python run_indep_panel.py                      # 300 steps, 30 seeds, indep..indep4
    python run_indep_panel.py 300 30 D_indep3 D_indep4   # only the new two
"""
import sys
import a2e_relay_fixed_final_51 as m

steps = int(sys.argv[1]) if len(sys.argv) > 1 else 300
n_seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 30
panel = tuple(sys.argv[3:]) or ("D_indep", "D_indep2", "D_indep3", "D_indep4")

m.run_indep_rulebase_panel_check(
    list(range(1, n_seeds + 1)), panel=panel, max_steps=steps,
    out_csv="indep_rulebase_panel.csv",
)
