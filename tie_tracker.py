# tie_tracker.py
from collections import defaultdict

TOL = 1e-12  # Algorithm 1-এর tolerance
STATS = defaultdict(lambda: {"decisions": 0, "ties": 0, "role_ties": 0})

def log_ties(policy_name, scored_list, higher_is_better=True):
    if len(scored_list) < 2:
        return  # N_c = 1 গোনা হবে না
        
    st = STATS[policy_name]
    st["decisions"] += 1
    
    # FIX 1: Unpack 3 values (cand, score, diag) instead of 2
    scores = [s for _, s, _ in scored_list]
    best = max(scores) if higher_is_better else min(scores)
    
    # FIX 2: Unpack 3 values here too
    tied = [c for c, s, _ in scored_list if abs(s - best) <= TOL]
    
    if len(tied) >= 2:
        st["ties"] += 1
        # FIX 3: cand is a dict, so use c.get("role") instead of getattr
        roles = {c.get("role") for c in tied if isinstance(c, dict)}
        if len(roles) >= 2:
            st["role_ties"] += 1

def report():
    for name, st in STATS.items():
        n = st["decisions"]
        if n == 0:
            print(f"{name}: কোনো decision নেই")
            continue
            
        print(f"{name}: decisions(Nc>=2)={n}, "
              f"ties={st['ties']} ({100*st['ties']/n:.2f}%), "
              f"role-differing ties={st['role_ties']} ({100*st['role_ties']/n:.2f}%)")