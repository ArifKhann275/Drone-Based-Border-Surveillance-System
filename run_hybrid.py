# -*- coding: utf-8 -*-
"""
run_hybrid.py
══════════════════════════════════════════════════════════════
Looks for a relay policy that is "effective on all sides": coverage at least
as good as Policy J (ideally better) WITHOUT the delay/energy cost of the
reserve-first heuristic.  Two sub-commands:

  tune  : sweep each hybrid's single parameter on the TRAINING block
          (seeds 101-150, the block Policy J's weights were tuned on),
          print means, reserve-selection share, Pareto flags, the
          consistency checks, and pick a parameter by a rule declared
          BEFORE looking at any confirmatory data.  Writes a JSON file.

  eval  : run J, D, B, BR (+ the chosen hybrids) on a block of seeds and test
          every hybrid against J, D, B and BR (paired Wilcoxon, rank-based r,
          paired-t CI, BH over all tests), plus an explicit verdict against
          the "all-round" criteria below.

HOW TO USE (in this order)
--------------------------
  1) python run_hybrid.py tune --seed-start 101 --seeds 50 --steps 500 ^
         --w-energy 0.283556320123423 --w-coverage 0.023813783478481596 ^
         --w-delay 0.28445380353708727 --w-battery 0.40817609286100814 --theta 0.5
  2) python run_hybrid.py eval --seed-start 201 --seeds 200 --steps 500 ^
         --w-energy ... (same weights) --tau-file tau_selection.json
     (201-400 is a FRESH block: nothing in the project has used it.)
  3) Optional, for comparability with the paper's tables only:
     python run_hybrid.py eval --seed-start 1 --seeds 50 ...  (descriptive;
     seeds 1-50 have been reused many times and are NOT confirmatory.)

ALL-ROUND CRITERIA (fixed here, before any confirmatory run)
------------------------------------------------------------
A hybrid is called "all-round" only if, on the confirmatory block and after
BH correction (q < 0.05):
   vs J : coverage significantly HIGHER, and delay and energy each either not
          significantly different or significantly LOWER;
   vs BR: delay AND energy significantly LOWER.
(vs B and vs D are reported but not part of the criterion.)  If no hybrid
meets this, the script says so; do not move the goalposts afterwards.

PARAMETER SELECTION RULE (tune; fixed in advance, change only via CLI flags
and report the change)
------------------------------------------------------------------------
Within a family, take the parameter values whose TRAINING delay and energy are
each <= (1 + cost_margin) x Policy J's training delay/energy (default 10%),
and among them pick the one with the highest training coverage.  Select it
only if it beats J's training coverage by at least min_gain points (default
0.5); otherwise the family is dropped (null in the JSON).

CALL ORDER / SANITY CHECKS
--------------------------
Results have been sensitive to call order for reasons never isolated, so the
per-seed order is fixed:  [prefix]  then the hybrids in the order given.
  --prefix full    J, D, Dindep..Dindep4, C, I, B, BR  (the validated prefix;
                   default for eval)
  --prefix minimal J, (D in eval), B, BR                (cheaper; default for
                   tune, gated by the J training-mean check below)
Checks printed on every run:
  * hybrid identities: HYBB@inf==B, HYBB@neg==BR, HYBD@inf==BR, HYBD@neg==B,
    HYBJ@inf==J, per seed on all three headline metrics;
  * tune on seeds 101-150: J coverage must be ~92.27 (the training coverage
    reported for the main weight vector);
  * eval on seeds 1-50: J, D, B, BR (and C, I with --prefix full) must
    reproduce the numbers from the earlier held-out runs.
Do not parallelise across seeds and do not reorder the sequence.

Tau/dmax tokens: numbers, "inf" (never override / no cap), "neg" (= -1,
always override / always cap).  Use '=' for lists:  --tau-grid-b=neg,0,0.05,inf
"""

import argparse
import csv
import json
import os
import time

import numpy as np
from scipy.stats import wilcoxon, rankdata, t as tdist

import a2e_relay_fixed_final_51 as relay
import policy_heuristics as heur
import policy_hybrid as hyb
from policy_j_full import select_best_relay_j, NORMALIZED_WEIGHTS_J_DEFAULT
from policy_d_true_fis import (
    select_fuzzy_relay_true_fis_indep_weights,
    select_fuzzy_relay_true_fis_indep2_weights,
    select_fuzzy_relay_true_fis_indep3_weights,
    select_fuzzy_relay_true_fis_indep4_weights,
)

HEADLINE = ("coverage_pct", "avg_relay_delay_steps", "avg_relay_energy_pct")
SHORT = {"coverage_pct": "cov", "avg_relay_delay_steps": "delay", "avg_relay_energy_pct": "energy"}
INF = 1e9

DEFAULT_GRID = {
    "B": "neg,0,0.01,0.02,0.05,0.1,0.2,0.4,inf",
    "D": "neg,4,8,12,16,20,25,inf",
    "J": "neg,0,0.01,0.02,0.05,0.1,0.2,0.4,inf",
}
FAMILY_NAME = {"B": "HYBB", "D": "HYBD", "J": "HYBJ"}
FAMILY_DESC = {
    "B": "nearest feasible; override to nearest reserve if the patrol pick's coverage_loss > tau",
    "D": "reserve-first; fall back to nearest feasible if the nearest reserve is farther than dmax",
    "J": "Policy J; override to nearest reserve if J's patrol pick has coverage_loss > tau",
}

KNOWN_TRAIN_J_COV = 92.27     # training-block coverage of the main weight vector
KNOWN_HELDOUT = {              # means on seeds 1-50 from the earlier runs
    "J": (90.568, 13.457, 9.876), "D": (90.732, 14.520, 11.837),
    "B": (88.846, 11.620, 10.018), "BR": (93.772, 19.491, 16.156),
    "C": (88.794, 14.217, 10.453), "I": (87.926, 13.455, 9.030),
}


# ── helpers ───────────────────────────────────────────────────────────────
def parse_tokens(spec):
    out = []
    for tok in spec.split(","):
        tok = tok.strip().lower()
        if not tok:
            continue
        out.append(-1.0 if tok == "neg" else (INF if tok in ("inf", "+inf") else float(tok)))
    return out


def tau_str(v):
    return "inf" if v >= hyb.BIG else ("neg" if v == -1.0 else f"{v:g}")


def tag(fam, v):
    return f"{FAMILY_NAME[fam]}@{tau_str(v)}"


def rank_biserial(x, y):
    d = np.asarray(x, float) - np.asarray(y, float)
    nz = d[d != 0]
    if len(nz) == 0:
        return 0.0
    rk = rankdata(np.abs(nz))
    rp, rn = rk[nz > 0].sum(), rk[nz < 0].sum()
    return float((rp - rn) / (rp + rn))


def wtest(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    return wilcoxon(x, y).pvalue if np.any(x != y) else 1.0


def paired_ci(x, y):
    d = np.asarray(x, float) - np.asarray(y, float)
    n = len(d)
    m = d.mean()
    se = d.std(ddof=1) / np.sqrt(n) if n > 1 else 0.0
    tc = tdist.ppf(0.975, n - 1) if n > 1 else 0.0
    return m, m - tc * se, m + tc * se


def bh(pvals):
    p = np.asarray(pvals, float)
    n = len(p)
    order = np.argsort(p)
    q_sorted = np.minimum.accumulate((p[order] * n / np.arange(1, n + 1))[::-1])[::-1]
    q = np.empty(n)
    q[order] = np.minimum(q_sorted, 1.0)
    return q


def col(rows, label, metric):
    return np.array([r.get(f"{label}_{metric}") for r in rows], dtype=float)


def mean_of(rows, label, metric):
    return float(np.nanmean(col(rows, label, metric)))


def reserve_share(rows, label):
    n = sum(r.get(f"{label}_n_decisions", 0) or 0 for r in rows)
    k = sum(r.get(f"{label}_n_reserve", 0) or 0 for r in rows)
    return (k / n) if n else float("nan")


def dominates(a, b, eps=1e-9):
    """a, b = (cov, delay, energy). True if a is at least as good everywhere and better somewhere."""
    ge = a[0] >= b[0] - eps and a[1] <= b[1] + eps and a[2] <= b[2] + eps
    gt = a[0] > b[0] + eps or a[1] < b[1] - eps or a[2] < b[2] - eps
    return ge and gt


def j_weights_from(args):
    ws = [args.w_energy, args.w_coverage, args.w_delay, args.w_battery]
    if all(w is None for w in ws):
        return None
    if any(w is None for w in ws):
        raise SystemExit("give all four --w-* weights or none of them")
    tot = sum(ws)
    return {"W_TRAVEL_ENERGY": ws[0] / tot, "W_COVERAGE_LOSS": ws[1] / tot,
            "W_RELAY_DELAY": ws[2] / tot, "W_BATTERY_MARGIN": ws[3] / tot}


# ── simulation ────────────────────────────────────────────────────────────
def build_selectors(args, specs):
    jw = j_weights_from(args)

    def j_sel(needer, drones, g, detected, step):
        return select_best_relay_j(needer, drones, g, detected, step,
                                   weights=jw, theta=args.theta)

    base = {
        "J": j_sel,
        "D": lambda n, d, g, det, s: relay.POLICY_SELECTORS["D"](n, d, g, det, s),
        "Dindep": select_fuzzy_relay_true_fis_indep_weights,
        "Dindep2": select_fuzzy_relay_true_fis_indep2_weights,
        "Dindep3": select_fuzzy_relay_true_fis_indep3_weights,
        "Dindep4": select_fuzzy_relay_true_fis_indep4_weights,
        "C": relay.select_best_relay,
        "I": relay.select_best_relay_coverage_aware,
        "B": heur.select_nearest_feasible,
        "BR": heur.select_nearest_reserve,
    }
    sel = dict(base)
    for fam, v in specs:
        if fam == "B":
            sel[tag(fam, v)] = hyb.make_hybrid_b(v)
        elif fam == "D":
            sel[tag(fam, v)] = hyb.make_hybrid_dcap(v)
        else:
            sel[tag(fam, v)] = hyb.make_hybrid_j(j_sel, v)
    return sel, jw


def prefix_labels(kind, need_d):
    if kind == "full":
        return ["J", "D", "Dindep", "Dindep2", "Dindep3", "Dindep4", "C", "I", "B", "BR"]
    return ["J"] + (["D"] if need_d else []) + ["B", "BR"]


def simulate(args, labels, selectors, csv_path):
    seeds = list(range(args.seed_start, args.seed_start + args.seeds))
    nd = args.num_drones if args.num_drones is not None else relay.NUM_DRONES
    nr = args.num_reserves if args.num_reserves is not None else relay.RESERVE_POOL_SIZE
    rows, t0 = [], time.time()
    for i, seed in enumerate(seeds, 1):
        row = {"seed": seed}
        for lab in labels:
            rec = hyb.RoleRecorder(selectors[lab])
            m = relay.simulate_fleet(seed, relay_selector=rec, num_drones=nd,
                                     num_reserves=nr, max_steps=args.steps)
            for k, v in m.items():
                if isinstance(v, (dict, list, tuple, set)):
                    continue
                row[f"{lab}_{k}"] = float(v) if isinstance(v, bool) else v
            row[f"{lab}_n_decisions"] = rec.n_decisions
            row[f"{lab}_n_reserve"] = rec.n_reserve
        rows.append(row)
        el = time.time() - t0
        print(f"  seed {seed} done  ({i}/{len(seeds)}, elapsed {el/60:.1f} min, "
              f"ETA {el/i*(len(seeds)-i)/60:.1f} min)", flush=True)
        with open(csv_path, "w", newline="") as fh:       # checkpoint every seed
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    print(f"\nWritten: {csv_path}")
    return rows


# ── shared checks ─────────────────────────────────────────────────────────
def identity_checks(rows, labels):
    print(f"\n{'=' * 100}\nCONSISTENCY CHECKS -- hybrid identities (per-seed equality)\n{'=' * 100}")
    pairs = [("HYBB@inf", "B"), ("HYBB@neg", "BR"), ("HYBD@inf", "BR"),
             ("HYBD@neg", "B"), ("HYBJ@inf", "J")]
    ok = True
    any_run = False
    for a, b in pairs:
        if a in labels and b in labels:
            any_run = True
            same = all(np.allclose(col(rows, a, m), col(rows, b, m), equal_nan=True) for m in HEADLINE)
            ok &= same
            print(f"  [{'OK ' if same else 'MISMATCH'}] {a} == {b}")
    if not any_run:
        print("  (no extreme-parameter hybrids in this run, nothing to check)")
    if not ok:
        print("\n  !! A mismatch means hidden state leaks between runs (the project's known "
              "call-order sensitivity). Do not trust any number below until it is resolved.")
    return ok


def sanity_heldout(rows, labels, args):
    if not (args.seed_start == 1 and args.seeds == 50):
        return
    print(f"\n{'=' * 100}\nSANITY CHECKS (seeds 1-50) -- must reproduce the earlier held-out runs\n{'=' * 100}")
    for lab, known in KNOWN_HELDOUT.items():
        if lab not in labels:
            continue
        for m, k in zip(HEADLINE, known):
            mean = mean_of(rows, lab, m)
            flag = "OK   " if abs(mean - k) <= 0.01 else "DRIFT"
            print(f"  [{flag}] {lab:<3} {m:<24} this run {mean:8.3f}  expected {k:8.3f}  diff {mean-k:+.3f}")


# ── tune ──────────────────────────────────────────────────────────────────
def cmd_tune(args):
    fams = [f.strip().upper() for f in args.families.split(",") if f.strip()]
    grids = {f: parse_tokens(getattr(args, f"tau_grid_{f.lower()}") or DEFAULT_GRID[f]) for f in fams}
    specs = [(f, v) for f in fams for v in grids[f]]
    labels = prefix_labels(args.prefix or "minimal", need_d=False) + [tag(f, v) for f, v in specs]
    selectors, jw = build_selectors(args, specs)

    print(f"J weights: {jw or NORMALIZED_WEIGHTS_J_DEFAULT} theta={args.theta}")
    print(f"TUNE block: seeds {args.seed_start}..{args.seed_start + args.seeds - 1}, steps={args.steps}, "
          f"prefix={args.prefix or 'minimal'}, families={fams}")
    for f in fams:
        print(f"  {FAMILY_NAME[f]}: {FAMILY_DESC[f]};  grid = {[tau_str(v) for v in grids[f]]}")
    print(f"Per-seed order: {labels}\nSimulations per seed: {len(labels)}\n")

    rows = simulate(args, labels, selectors, args.out_csv)
    identity_checks(rows, labels)

    jm = tuple(mean_of(rows, "J", m) for m in HEADLINE)
    if args.seed_start == 101 and args.seeds == 50:
        flag = "OK   " if abs(jm[0] - KNOWN_TRAIN_J_COV) <= 0.05 else "DRIFT"
        print(f"\n  [{flag}] J training coverage {jm[0]:.3f}  (expected ~{KNOWN_TRAIN_J_COV})"
              + ("" if flag.strip() == "OK" else "   <-- the shorter call order changed J; rerun with --prefix full"))

    refs = {l: tuple(mean_of(rows, l, m) for m in HEADLINE) for l in ("J", "B", "BR")}
    print(f"\n{'=' * 100}\nREFERENCES (training block)\n{'=' * 100}")
    print(f"{'policy':<12}{'cov':>9}{'delay':>9}{'energy':>9}{'reserve%':>10}")
    for l, v in refs.items():
        print(f"{l:<12}{v[0]:>9.2f}{v[1]:>9.2f}{v[2]:>9.2f}{100*reserve_share(rows, l):>10.1f}")

    selection = {"rule": {"cost_margin": args.cost_margin, "min_gain": args.min_gain,
                          "reference": "J training means"},
                 "training": {"seed_start": args.seed_start, "seeds": args.seeds,
                              "J": dict(zip(SHORT.values(), jm))},
                 "tau_b": None, "tau_d": None, "tau_j": None}
    for f in fams:
        print(f"\n{'=' * 100}\n{FAMILY_NAME[f]} sweep -- {FAMILY_DESC[f]}\n{'=' * 100}")
        print(f"{'param':<8}{'cov':>9}{'delay':>9}{'energy':>9}{'reserve%':>10}  flags")
        table = []
        for v in grids[f]:
            l = tag(f, v)
            table.append((v, tuple(mean_of(rows, l, m) for m in HEADLINE), reserve_share(rows, l)))
        pool = list(refs.values()) + [t[1] for t in table]
        for v, pt, rs in table:
            flags = []
            if not any(dominates(o, pt) for o in pool if o is not pt):
                flags.append("Pareto")
            if (pt[1] <= jm[1] * (1 + args.cost_margin) and pt[2] <= jm[2] * (1 + args.cost_margin)):
                flags.append("in-budget")
            print(f"{tau_str(v):<8}{pt[0]:>9.2f}{pt[1]:>9.2f}{pt[2]:>9.2f}{100*rs:>10.1f}  {' '.join(flags)}")
        # the extremes are identical to B / BR / J themselves, so they are not 'hybrids'
        extreme = {"B": {"inf", "neg"}, "D": {"inf", "neg"}, "J": {"inf"}}[f]
        ok = [(v, pt) for v, pt, _ in table
              if tau_str(v) not in extreme
              and pt[1] <= jm[1] * (1 + args.cost_margin) and pt[2] <= jm[2] * (1 + args.cost_margin)]
        best = max(ok, key=lambda t: (t[1][0], t[0])) if ok else None
        key = {"B": "tau_b", "D": "tau_d", "J": "tau_j"}[f]
        if best and best[1][0] - jm[0] >= args.min_gain:
            selection[key] = best[0]
            print(f"  -> SELECTED {tau_str(best[0])}  (training coverage {best[1][0]:.2f} vs J {jm[0]:.2f}, "
                  f"delay {best[1][1]:.2f} vs {jm[1]:.2f}, energy {best[1][2]:.2f} vs {jm[2]:.2f})")
        else:
            print(f"  -> NO value of this family beats J's training coverage by >= {args.min_gain} points "
                  f"within the {100*args.cost_margin:.0f}% delay/energy budget; family dropped.")
    with open(args.out_json, "w") as fh:
        json.dump(selection, fh, indent=2)
    print(f"\nWritten: {args.out_json}  ->  {({k: v for k, v in selection.items() if k.startswith('tau_')})}")
    print("\nNEXT: run the 'eval' sub-command on the fresh block 201-400 with --tau-file "
          f"{args.out_json}. Do not re-tune after looking at it.")


# ── eval ──────────────────────────────────────────────────────────────────
def cmd_eval(args):
    taus = {"B": None, "D": None, "J": None}
    if args.tau_file and os.path.exists(args.tau_file):
        with open(args.tau_file) as fh:
            js = json.load(fh)
        taus = {"B": js.get("tau_b"), "D": js.get("tau_d"), "J": js.get("tau_j")}
    for f, a in (("B", args.tau_b), ("D", args.dmax), ("J", args.tau_j)):
        if a is not None:
            taus[f] = parse_tokens(a)[0]
    specs = [(f, v) for f, v in taus.items() if v is not None]
    hyb_labels = [tag(f, v) for f, v in specs]

    lo, hi = args.seed_start, args.seed_start + args.seeds - 1
    if lo <= 150 and hi >= 101:
        print("!! WARNING: this block overlaps the TRAINING seeds 101-150. Not confirmatory.")
    if lo <= 50:
        print("!! NOTE: seeds 1-50 have been used for many analyses in the project; results on them are "
              "descriptive, not confirmatory.")
    if not specs:
        print("No hybrid parameter selected (all families dropped or none supplied); "
              "running references only.")

    labels = prefix_labels(args.prefix or "full", need_d=True) + hyb_labels
    labels = list(dict.fromkeys(labels))
    selectors, jw = build_selectors(args, specs)
    print(f"J weights: {jw or NORMALIZED_WEIGHTS_J_DEFAULT} theta={args.theta}")
    print(f"EVAL block: seeds {lo}..{hi} (n={args.seeds}), steps={args.steps}, "
          f"prefix={args.prefix or 'full'}; hybrids: {hyb_labels or 'none'}")
    print(f"Per-seed order: {labels}\n")

    rows = simulate(args, labels, selectors, args.out_csv)
    identity_checks(rows, labels)
    sanity_heldout(rows, labels, args)

    show = [l for l in labels if l not in ("Dindep", "Dindep2", "Dindep3", "Dindep4")]
    print(f"\n{'=' * 100}\nMEANS (n={len(rows)})\n{'=' * 100}")
    print(f"{'policy':<12}{'cov':>9}{'delay':>9}{'energy':>9}{'reserve%':>10}")
    pts = {}
    for l in show:
        pts[l] = tuple(mean_of(rows, l, m) for m in HEADLINE)
        print(f"{l:<12}{pts[l][0]:>9.2f}{pts[l][1]:>9.2f}{pts[l][2]:>9.2f}{100*reserve_share(rows, l):>10.1f}")
    nd = [l for l in show if not any(dominates(pts[o], pts[l]) for o in show if o != l)]
    print(f"\n  Non-dominated in mean values (cov up, delay down, energy down): {nd}")

    if not hyb_labels:
        return
    refs = [r for r in ("J", "D", "B", "BR") if r in labels]
    tests = []
    for h in hyb_labels:
        for r in refs:
            for m in HEADLINE:
                x, y = col(rows, h, m), col(rows, r, m)
                mean, lo_, hi_ = paired_ci(x, y)
                tests.append(dict(h=h, r=r, m=m, d=mean, lo=lo_, hi=hi_,
                                  p=wtest(x, y), rb=rank_biserial(x, y)))
    for t, q in zip(tests, bh([t["p"] for t in tests])):
        t["q"] = q

    print(f"\n{'=' * 100}\nHYBRID vs REFERENCE -- {len(tests)}-test BH family (difference = hybrid minus reference)\n{'=' * 100}")
    for h in hyb_labels:
        for r in refs:
            print(f"\n--- {h} vs {r} ---")
            print(f"{'metric':<24}{'diff':>9}{'95% CI':>22}{'p':>10}{'q_BH':>10}{'r':>8}")
            for t in tests:
                if t["h"] == h and t["r"] == r:
                    st = " *" if t["q"] < 0.05 else ""
                    print(f"{t['m']:<24}{t['d']:>+9.3f}  [{t['lo']:>+7.3f},{t['hi']:>+7.3f}]"
                          f"{t['p']:>10.4f}{t['q']:>10.4f}{t['rb']:>8.3f}{st}")
    print("\n  * = q_BH < 0.05   (delay/energy: negative diff = hybrid better; coverage: positive = hybrid better)")

    def get(h, r, m):
        return next(t for t in tests if t["h"] == h and t["r"] == r and t["m"] == m)

    print(f"\n{'=' * 100}\nVERDICT against the pre-declared all-round criteria\n{'=' * 100}")
    for h in hyb_labels:
        if "J" not in refs or "BR" not in refs:
            print(f"  {h}: J or BR missing from the run, cannot evaluate.")
            continue
        c = get(h, "J", "coverage_pct")
        dl = get(h, "J", "avg_relay_delay_steps")
        en = get(h, "J", "avg_relay_energy_pct")
        c1 = c["d"] > 0 and c["q"] < 0.05
        c2 = not (dl["d"] > 0 and dl["q"] < 0.05)
        c3 = not (en["d"] > 0 and en["q"] < 0.05)
        bd = get(h, "BR", "avg_relay_delay_steps")
        be = get(h, "BR", "avg_relay_energy_pct")
        c4 = bd["d"] < 0 and bd["q"] < 0.05
        c5 = be["d"] < 0 and be["q"] < 0.05
        print(f"  {h}:")
        print(f"     coverage significantly higher than J ........ {'YES' if c1 else 'no '} (diff {c['d']:+.2f}, q={c['q']:.4f})")
        print(f"     delay  not significantly worse than J ........ {'YES' if c2 else 'no '} (diff {dl['d']:+.2f}, q={dl['q']:.4f})")
        print(f"     energy not significantly worse than J ........ {'YES' if c3 else 'no '} (diff {en['d']:+.2f}, q={en['q']:.4f})")
        print(f"     delay  significantly lower than BR ........... {'YES' if c4 else 'no '} (diff {bd['d']:+.2f}, q={bd['q']:.4f})")
        print(f"     energy significantly lower than BR ........... {'YES' if be['d'] < 0 and be['q'] < 0.05 else 'no '} (diff {be['d']:+.2f}, q={be['q']:.4f})")
        print(f"     => ALL-ROUND CRITERIA {'MET' if all([c1, c2, c3, c4, c5]) else 'NOT MET'}")
    print("\n  Report the result either way. Do not change the criteria, the parameter or the "
          "block after seeing this output.")


# ── CLI ───────────────────────────────────────────────────────────────────
def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--w-energy", type=float, default=None)
        sp.add_argument("--w-coverage", type=float, default=None)
        sp.add_argument("--w-delay", type=float, default=None)
        sp.add_argument("--w-battery", type=float, default=None)
        sp.add_argument("--theta", type=float, default=0.5)
        sp.add_argument("--steps", type=int, default=500)
        sp.add_argument("--num-drones", type=int, default=None)
        sp.add_argument("--num-reserves", type=int, default=None)
        sp.add_argument("--prefix", choices=["full", "minimal"], default=None)

    t = sub.add_parser("tune", help="sweep hybrid parameters on the training block")
    common(t)
    t.add_argument("--seed-start", type=int, default=101)
    t.add_argument("--seeds", type=int, default=50)
    t.add_argument("--families", default="B,D,J", help="comma list from B,D,J")
    t.add_argument("--tau-grid-b", default=None)
    t.add_argument("--tau-grid-d", default=None)
    t.add_argument("--tau-grid-j", default=None)
    t.add_argument("--cost-margin", type=float, default=0.10)
    t.add_argument("--min-gain", type=float, default=0.5)
    t.add_argument("--out-csv", default="hybrid_tune.csv")
    t.add_argument("--out-json", default="tau_selection.json")
    t.set_defaults(func=cmd_tune)

    e = sub.add_parser("eval", help="evaluate selected hybrids on a block of seeds")
    common(e)
    e.add_argument("--seed-start", type=int, default=201)
    e.add_argument("--seeds", type=int, default=200)
    e.add_argument("--tau-file", default="tau_selection.json")
    e.add_argument("--tau-b", default=None, help="override HYBB parameter")
    e.add_argument("--dmax", default=None, help="override HYBD parameter")
    e.add_argument("--tau-j", default=None, help="override HYBJ parameter")
    e.add_argument("--out-csv", default="hybrid_eval.csv")
    e.set_defaults(func=cmd_eval)
    return p


if __name__ == "__main__":
    a = build_parser().parse_args()
    a.func(a)
