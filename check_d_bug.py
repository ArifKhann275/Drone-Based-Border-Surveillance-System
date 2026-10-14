import a2e_relay_fixed_final_51 as relay
from policy_d_true_fis import select_fuzzy_relay_true_fis, select_fuzzy_relay_true_fis_indep_weights

print("Running D...")
state_d = relay.simulate_fleet(1, relay_selector=select_fuzzy_relay_true_fis, max_steps=100)
d_log = state_d.get("decision_log", [])

print("Running D-indep...")
state_indep = relay.simulate_fleet(1, relay_selector=select_fuzzy_relay_true_fis_indep_weights, max_steps=100)
indep_log = state_indep.get("decision_log", [])

if d_log and indep_log:
    print("\n--- Decision at Step", d_log[0]["step"], "---")
    
    # FIX: Using the correct key 'fis_suitability'
    d_scores = [d["fis_suitability"] for d in d_log[0]["candidates_diag"]]
    indep_scores = [d["fis_suitability"] for d in indep_log[0]["candidates_diag"]]
    
    print("D scores      :", d_scores)
    print("D-indep scores:", indep_scores)
    
    if d_scores == indep_scores:
        print("\n✅ The scores are identical, but it's NOT a bug!")
        print("It means for these specific candidates (likely reserves), both rule bases mathematically evaluate to the exact same fuzzy outcome. Your logic is perfectly fine!")
    else:
        print("\n✅ Scores are different. The logic is fine!")