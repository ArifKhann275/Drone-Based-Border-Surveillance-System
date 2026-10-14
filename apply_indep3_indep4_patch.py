# -*- coding: utf-8 -*-
"""
apply_indep3_indep4_patch.py  --  adds two more independently-authored,
non-compensatory FIS rule profiles ("indep3", "indep4") to
policy_d_true_fis.py, for the multi-independent-baseline circularity
test (J vs D, D-indep, D-indep2, D-indep3, D-indep4).

DESIGN PROCESS (done BEFORE looking at any Policy J result)
-------------------------------------------------------------
Both profiles were chosen by an automated search over vetoes/gates/
thresholds restricted to {battery, distance, delay} only (never
referencing coverage_loss in a veto or gate, for the same reason
"indep2" was introduced: a veto/gate on coverage_loss never fires for
an idle-reserve candidate, whose coverage_loss is 0/best by
definition, so such a profile silently collapses onto "expert" in the
reserve-dominated part of the decision space -- see indep2's own
docstring in policy_d_true_fis.py). The search selected, among
candidates meeting ALL of the following criteria simultaneously, the
one maximizing disagreement with "expert" specifically within the
coverage_loss=LOW slice (the slice real candidates actually occupy):

  - non-compensatory with a comfortable safety margin (eps* <= -0.10,
    not merely eps* < 0 -- a hairline-negative margin is fragile to
    reproduce exactly and is a weaker "independent" witness)
  - balanced class counts (>=10 cells in each of LOW/MEDIUM/HIGH)
  - disagrees with EVERY existing profile (expert, indep, indep2, and,
    for indep4, indep3) on at least 8-10 of the 81 cells overall -- not
    just a relabeling of an existing profile with different weights
  - indep3 and indep4 additionally differ from each other in veto/gate
    CARDINALITY (indep3: one veto pair, one-attribute gate; indep4: two
    veto pairs, two-attribute gate), so the four independent profiles
    are not four cosmetic variations of the same rule-authoring
    template.

"indep3" theme: energy/battery emphasis (heavy battery_margin and
distance weight, since RelayScore/Policy J score travel energy
directly and distance is the FIS's only proxy for it).

"indep4" theme: balanced distance/battery_margin/relay_delay (roughly
equal three-way weight, small coverage_loss weight), with a
structurally distinct (2-veto, 2-gate) tier configuration from every
prior profile.

USAGE
-----
    python apply_indep3_indep4_patch.py            # patch in place, makes .bak
    python apply_indep3_indep4_patch.py --check     # report status only
    python apply_indep3_indep4_patch.py --restore   # undo from .bak

Idempotent: running it twice is safe.
"""

import os
import shutil
import sys

TARGET = "policy_d_true_fis.py"
BACKUP = TARGET + ".preindep34.bak"
MARKER = "INDEPENDENT3_RULE_WEIGHTS"

EDITS = []

# 1. weight vectors, right after INDEPENDENT2_RULE_WEIGHTS -----------------
EDITS.append((
    'DEFAULT_RULE_WEIGHTS = dict(FUZZY_ATTR_WEIGHTS)',
    '''# "indep3": a THIRD independently-authored profile (energy/battery
# emphasis). Its vetoes/gates reference only battery_margin, distance and
# relay_delay -- never coverage_loss -- so, like "indep2", it stays active
# in the reserve-dominated part of the decision space instead of silently
# collapsing onto "expert" there. Chosen by a search maximizing disagreement
# with "expert" inside the coverage_loss=LOW slice specifically, subject to
# a comfortable non-compensatory margin (eps*<=-0.10) and reasonable class
# balance; the search ran before any Policy J result was inspected.
# eps*=-0.2308; differs from expert on 24/81 cells, indep on 26/81,
# indep2 on 18/81; 8/27 within the coverage_loss=LOW slice.
INDEPENDENT3_RULE_WEIGHTS = {
    "distance": 0.30,
    "battery_margin": 0.45,
    "coverage_loss": 0.10,
    "relay_delay": 0.15,
}

# "indep4": a FOURTH independently-authored profile (balanced
# distance/battery_margin/relay_delay). Structurally distinct from
# "indep3" in veto/gate cardinality (two veto pairs and a two-attribute
# gate here, vs. one and one for indep3), so the four independent profiles
# are not four cosmetic variations of the same authoring template.
# eps*=-0.1333; differs from expert on 24/81, indep on 26/81, indep2 on
# 15/81, indep3 on 15/81; 10/27 within the coverage_loss=LOW slice.
INDEPENDENT4_RULE_WEIGHTS = {
    "distance": 0.31,
    "battery_margin": 0.31,
    "coverage_loss": 0.06,
    "relay_delay": 0.32,
}

DEFAULT_RULE_WEIGHTS = dict(FUZZY_ATTR_WEIGHTS)''',
    1,
))

# 2. RULE_PROFILES entries -- insert right after the indep2 profile block --
EDITS.append((
    '''        "lo_cut": 0.50,
        "weights": INDEPENDENT2_RULE_WEIGHTS,
    },
}''',
    '''        "lo_cut": 0.50,
        "weights": INDEPENDENT2_RULE_WEIGHTS,
    },

    # -- Profile "indep3": ENERGY/BATTERY-FIRST, authored independently a
    #    third time. See INDEPENDENT3_RULE_WEIGHTS above for the design
    #    process. Vetoes/gates reference only b/d/l, never c.
    "indep3": {
        "name": "indep3",
        "description": (
            "Energy/battery-first relay doctrine authored independently of "
            "'expert', 'indep', and 'indep2'. A candidate that is both "
            "marginal on battery and far is unusable regardless of "
            "coverage or delay. Top suitability demands both a comfortable "
            "battery margin and at least two best-in-class attributes."
        ),
        "vetoes": [
            ("b", "d"),
        ],
        "excellence_gates": ["b"],
        "n_best_required": 2,
        "n_worst_low": 3,
        "lo_cut": 0.50,
        "weights": INDEPENDENT3_RULE_WEIGHTS,
    },

    # -- Profile "indep4": BALANCED distance/battery/delay, authored
    #    independently a fourth time, structurally distinct from indep3
    #    (two veto pairs, a two-attribute gate) rather than a cosmetic
    #    reweighting of the same tier template.
    "indep4": {
        "name": "indep4",
        "description": (
            "Balanced distance/battery/delay relay doctrine authored "
            "independently of every other profile in this module. A "
            "candidate that is marginal on battery and either far or slow "
            "is unusable; top suitability demands both a comfortable "
            "battery margin and a nearby trip."
        ),
        "vetoes": [
            ("b", "d"),
            ("b", "l"),
        ],
        "excellence_gates": ["b", "d"],
        "n_best_required": 2,
        "n_worst_low": 2,
        "lo_cut": 0.55,
        "weights": INDEPENDENT4_RULE_WEIGHTS,
    },
}''',
    1,
))

# 3. selector functions -- right after select_fuzzy_relay_true_fis_indep2_weights
EDITS.append((
    '''def select_fuzzy_relay_true_fis_skfuzzy(needer, drones, g, detected, step,''',
    '''def select_fuzzy_relay_true_fis_indep3_weights(needer, drones, g, detected, step,
                                               resolution=101, backend=None):
    """Policy D-indep3: a third independently-authored rule base
    (energy/battery emphasis; see INDEPENDENT3_RULE_WEIGHTS)."""
    return select_fuzzy_relay_true_fis(
        needer, drones, g, detected, step,
        weights=INDEPENDENT3_RULE_WEIGHTS, resolution=resolution, backend=backend,
    )


def select_fuzzy_relay_true_fis_indep4_weights(needer, drones, g, detected, step,
                                               resolution=101, backend=None):
    """Policy D-indep4: a fourth independently-authored rule base
    (balanced distance/battery/delay; see INDEPENDENT4_RULE_WEIGHTS)."""
    return select_fuzzy_relay_true_fis(
        needer, drones, g, detected, step,
        weights=INDEPENDENT4_RULE_WEIGHTS, resolution=resolution, backend=backend,
    )


def select_fuzzy_relay_true_fis_skfuzzy(needer, drones, g, detected, step,''',
    1,
))

# 4. include indep3/indep4 in the built-in __main__ audit printout ---------
EDITS.append((
    '''    for name in ("expert", "indep", "indep2"):''',
    '''    for name in ("expert", "indep", "indep2", "indep3", "indep4"):''',
    1,
))


def apply(src):
    for old, new, count in EDITS:
        found = src.count(old)
        if found != count:
            raise SystemExit(
                f"PATCH ABORTED: expected {count} occurrence(s) of:\n"
                f"----\n{old[:200]}\n----\nbut found {found}. Your "
                f"{TARGET} differs from the version this patch was written "
                f"against; apply that hunk by hand."
            )
        src = src.replace(old, new)
    return src


def main():
    if not os.path.exists(TARGET):
        raise SystemExit(f"{TARGET} not found. Run this from your project folder.")
    src = open(TARGET, encoding="utf-8").read()

    if "--check" in sys.argv:
        print("PATCHED" if MARKER in src else "NOT PATCHED")
        return

    if "--restore" in sys.argv:
        if not os.path.exists(BACKUP):
            raise SystemExit(f"No backup at {BACKUP}")
        shutil.copy2(BACKUP, TARGET)
        print(f"restored {TARGET} from {BACKUP}")
        return

    if MARKER in src:
        print("Already patched -- nothing to do.")
        return

    if not os.path.exists(BACKUP):
        shutil.copy2(TARGET, BACKUP)
        print(f"backup written: {BACKUP}")

    out = apply(src)
    compile(out, TARGET, "exec")
    open(TARGET, "w", encoding="utf-8").write(out)
    print(f"patched {TARGET}  ({len(EDITS)} hunks)")
    print("\nVerify before running anything else:")
    print("  python -c \"import policy_d_true_fis as f; "
          "[print(n, f.audit_non_compensatory(n)['separability_margin'], "
          "f.audit_non_compensatory(n)['is_weighted_sum']) "
          "for n in ('expert','indep','indep2','indep3','indep4')]\"")
    print("\nThen: python run_j_vs_multi_indep.py --seeds 50")


if __name__ == "__main__":
    main()
