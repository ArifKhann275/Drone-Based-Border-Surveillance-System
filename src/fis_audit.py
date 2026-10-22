# -*- coding: utf-8 -*-
"""
fis_audit.py  --  regenerates the evidence for both fixes, in one run.

    python fis_audit.py            # everything (a few minutes; skfuzzy is slow)
    python fis_audit.py --quick    # skip the slow scikit-fuzzy timings

Sections
--------
A. Is the rule base actually non-compensatory?
   LP separability margin eps* and hand-checkable arithmetic certificates,
   for the new rule base AND for the old auto-generated one, side by side.
   -> This is the table that answers "your FIS is a strawman".

B. Are the two backends the same decision function?
   Vectorized NumPy Mamdani vs scikit-fuzzy, on random points.
   -> Lets you state that the fast FIS is the FIS, not an approximation.

C. What does a Mamdani inference actually cost?
   Per-call microbenchmark: scikit-fuzzy, vectorized FIS, D2 LUT, and a
   RelayScore-style weighted sum, on this machine.
   -> Replaces the "484x" headline, which measured library overhead.

D. Does cost scale with rule count R?
   The same vectorized engine at R = 16, 81, 256, 625, 1296.
   -> This is the part of the runtime claim that survives a reviewer
      reimplementing your FIS properly, so it should carry the argument.

Everything prints as plain text you can paste into a results table, and is
also written to fis_audit_report.json.
"""

import json
import itertools
import sys
import time

import numpy as np

import policy_d_true_fis as fis

QUICK = "--quick" in sys.argv


def _hr(title):
    print("\n" + "=" * 74)
    print(title)
    print("=" * 74)


def _rand_points(n, seed=0):
    rng = np.random.default_rng(seed)
    return np.column_stack([
        rng.uniform(fis.BATTERY_MARGIN_UNIVERSE[0], fis.BATTERY_MARGIN_UNIVERSE[-1], n),
        rng.uniform(fis.DISTANCE_UNIVERSE[0], fis.DISTANCE_UNIVERSE[-1], n),
        rng.uniform(fis.COVERAGE_LOSS_UNIVERSE[0], fis.COVERAGE_LOSS_UNIVERSE[-1], n),
        rng.uniform(fis.RELAY_DELAY_UNIVERSE[0], fis.RELAY_DELAY_UNIVERSE[-1], n),
    ])


def _timeit(fn, pts, warmup=3):
    for p in pts[:warmup]:
        fn(*p)
    t0 = time.perf_counter()
    for p in pts:
        fn(*p)
    return (time.perf_counter() - t0) / len(pts)


# ══════════════════════════════════════════════════════════════════════════
# A. NON-COMPENSATORY AUDIT
# ══════════════════════════════════════════════════════════════════════════
def section_a():
    _hr("A. IS THE RULE BASE NON-COMPENSATORY?")
    print("eps* = largest margin with which SOME non-negative weighted sum")
    print("       w.g plus two cut points reproduces the whole rule table.")
    print("  eps* >  0 : the table IS a linear scalarization  (compensatory)")
    print("  eps* <= 0 : NO weighted sum reproduces it        (non-compensatory)\n")

    rows = []
    legacy = fis.legacy_compensatory_table()
    eps_legacy = fis.linear_separability_margin(table=legacy)
    counts = {t: sum(1 for v in legacy.values() if v == t) for t in fis.CONSEQUENT_TERMS}
    rows.append(("OLD v1 auto-generated", counts, eps_legacy, eps_legacy > 1e-9))
    print(f"{'rule base':<26} {'LOW':>5} {'MED':>5} {'HIGH':>5} {'eps*':>9}  verdict")
    print("-" * 74)
    print(f"{'OLD v1 auto-generated':<26} {counts['LOW']:>5} {counts['MEDIUM']:>5} "
          f"{counts['HIGH']:>5} {eps_legacy:>+9.4f}  IS a weighted sum (strawman)")

    report = {"legacy": {"counts": counts, "eps": eps_legacy}}
    for profile in ("expert", "indep"):
        rep = fis.audit_non_compensatory(profile)
        c = rep["class_counts"]
        verdict = ("IS a weighted sum" if rep["is_weighted_sum"]
                   else "NOT any weighted sum")
        print(f"{'NEW ' + profile:<26} {c['LOW']:>5} {c['MEDIUM']:>5} {c['HIGH']:>5} "
              f"{rep['separability_margin']:>+9.4f}  {verdict}")
        report[profile] = {
            "counts": c, "eps": rep["separability_margin"],
            "fingerprint": rep["fingerprint"],
            "is_weighted_sum": rep["is_weighted_sum"],
        }

    print("\nHand-checkable certificate for the operating rule base ('expert'):")
    certs = fis.non_compensatory_certificates("expert", limit=1)
    if certs:
        why, A, ca, B, cb, C, cc = certs[0]
        gof = lambda k: (fis._GOODNESS["battery_margin"][k[0]],
                         fis._GOODNESS["distance"][k[1]],
                         fis._GOODNESS["coverage_loss"][k[2]],
                         fis._GOODNESS["relay_delay"][k[3]])
        print(f"  {why}")
        print(f"    A = {A}  g={gof(A)}  -> {ca}")
        print(f"    B = {B}  g={gof(B)}  -> {cb}")
        print(f"    C = {C}  g={gof(C)}  -> {cc}")
        print("  g(C) = (g(A)+g(B))/2, so for ANY weights w >= 0:")
        print("      s(C) = (s(A)+s(B))/2 < theta2,  yet C is rated HIGH.")
        print("  No linear scalarization can do that. QED.")
        report["certificate"] = {"why": why, "A": list(A), "cA": ca,
                                 "B": list(B), "cB": cb, "C": list(C), "cC": cc}
    return report


# ══════════════════════════════════════════════════════════════════════════
# B. BACKEND EQUIVALENCE
# ══════════════════════════════════════════════════════════════════════════
def section_b():
    _hr("B. IS THE FIS WIRED UP CORRECTLY?")
    out = {}

    # B0 -- orientation. Two backends agreeing proves nothing if both share
    # the same wrong table; this compares each label to its universe instead.
    print("  B0. membership-function orientation")
    try:
        rows = fis.validate_membership_orientation()
        for attr, lbl, peak, want, ok in rows:
            print(f"      {attr:<15} {lbl:<7} peak={peak:<9.4g} expected at {want}"
                  f"   {'OK' if ok else 'WRONG'}")
        out["orientation"] = "ok"
    except AssertionError as exc:
        print(f"      FAILED: {exc}")
        out["orientation"] = str(exc)

    # B1 -- monotonicity of the decision surface
    print("\n  B1. monotonicity (non-compensatory is fine; backwards is not)")
    mono = fis.validate_monotonicity(n=150)
    for attr, r in mono.items():
        print(f"      suitability {r['direction']:<10} in {attr:<15} "
              f"max violation {r['max_violation']:.3f} / 100")
    out["monotonicity"] = mono
    print("      Small violations are expected: the excellence gate and the")
    print("      balance floor are deliberately not additive. A violation of")
    print("      tens of points would mean something is inverted.")

    print("\n  B2. vectorized NumPy vs scikit-fuzzy, same rule base")
    if QUICK:
        print("      skipped (--quick)")
        return out
    for profile in ("expert", "indep"):
        v = fis.validate_backends(n=600, weights=profile, seed=1)
        out[profile] = v
        print(f"      {profile:<8} n={v['n']}  max|diff| = {v['max_abs_err']:.3e}   "
              f"RMSE = {v['rmse']:.3e}")
    print("\n      The vectorized engine is the SAME decision function,")
    print("      reproduced to floating-point noise -- not a surrogate.")
    return out


# ══════════════════════════════════════════════════════════════════════════
# C. PER-CALL COST
# ══════════════════════════════════════════════════════════════════════════
def _relayscore_like(b, d, c, dl):
    """A RelayScore-style weighted sum, for a like-for-like cost reference."""
    return 0.35 * d + 0.40 * c + 0.25 * dl + 0.0 * b


def section_c():
    _hr("C. PER-CALL COST OF ONE CANDIDATE EVALUATION")
    pts = _rand_points(2000, seed=2)
    res = {}

    t_vec = _timeit(lambda *p: fis._run_fis(*p, backend="vectorized"), pts)
    res["fis_vectorized_us"] = t_vec * 1e6

    t_lin = _timeit(_relayscore_like, pts)
    res["weighted_sum_us"] = t_lin * 1e6

    t_lut = None
    try:
        from surrogate_fis import _run_fis_surrogate
        t_lut = _timeit(lambda *p: _run_fis_surrogate(*p, resolution=9), pts)
        res["d2_lut_us"] = t_lut * 1e6
    except Exception as exc:                                   # noqa: BLE001
        print(f"  (D2 skipped: {exc})")

    t_skf = None
    if not QUICK:
        t_skf = _timeit(lambda *p: fis._run_fis(*p, backend="skfuzzy"), pts[:60])
        res["fis_skfuzzy_us"] = t_skf * 1e6

    print(f"{'implementation':<34} {'us/call':>12} {'vs skfuzzy':>12}")
    print("-" * 74)
    if t_skf:
        print(f"{'Mamdani FIS, scikit-fuzzy 0.5.0':<34} {t_skf*1e6:>12.1f} {1.0:>11.1f}x")
    def _row(lbl, t):
        ratio = f"{t_skf/t:>11.1f}x" if t_skf else f"{'-':>12}"
        print(f"{lbl:<34} {t*1e6:>12.1f} {ratio}")
    _row("Mamdani FIS, vectorized NumPy", t_vec)
    if t_lut:
        _row("D2 distilled LUT (Rgrid=9)", t_lut)
    _row("RelayScore-style weighted sum", t_lin)

    print("\n  Read this as: almost the entire gap you previously reported was")
    print("  scikit-fuzzy's per-call rule-graph rebuild, not Mamdani inference.")
    if t_lut:
        print(f"  Against a properly implemented FIS, D2 is ~{t_vec/t_lut:.1f}x faster")
        print("  at R=81 -- not two orders of magnitude. Its structural advantage")
        print("  is independence from R (Section D), so lead with that.")
    return res


# ══════════════════════════════════════════════════════════════════════════
# D. SCALING WITH RULE COUNT R
# ══════════════════════════════════════════════════════════════════════════
class _GenericMamdani:
    """Vectorized Mamdani over 4 antecedents with `m` triangular terms each,
    so R = m^4. Same inference semantics as the production engine; used only
    to measure how cost grows with rule count."""

    def __init__(self, m, seed=0):
        self.m = m
        self.R = m ** 4
        self.unis = [np.linspace(0, 1, 101) for _ in range(4)]
        self.mfs = []
        for u in self.unis:
            centres = np.linspace(0, 1, m)
            w = 1.0 / max(m - 1, 1)
            self.mfs.append(np.vstack([
                np.clip(1.0 - np.abs(u - c) / w, 0, 1) for c in centres]))
        su = np.linspace(0, 100, 101)
        self.su = su
        self.cons_mf = np.vstack([
            np.clip(1.0 - np.abs(su - c) / 50.0, 0, 1) for c in (0.0, 50.0, 100.0)])
        rng = np.random.default_rng(seed)
        self.idx = rng.integers(0, 3, size=(m, m, m, m))
        self.masks = [self.idx == k for k in range(3)]

    def run(self, *xs):
        mu = []
        for k in range(4):
            u, mf = self.unis[k], self.mfs[k]
            x = float(np.clip(xs[k], 0.0, 1.0))
            mu.append(np.array([np.interp(x, u, mf[j]) for j in range(self.m)]))
        m = self.m
        alpha = np.minimum(
            np.minimum(mu[0].reshape(m, 1, 1, 1), mu[1].reshape(1, m, 1, 1)),
            np.minimum(mu[2].reshape(1, 1, m, 1), mu[3].reshape(1, 1, 1, m)))
        s = [float(alpha[self.masks[k]].max()) if self.masks[k].any() else 0.0
             for k in range(3)]
        agg = np.zeros_like(self.su)
        for k in range(3):
            np.maximum(agg, np.minimum(s[k], self.cons_mf[k]), out=agg)
        area = agg.sum()
        return float((agg * self.su).sum() / area) if area > 0 else 50.0


def section_d():
    _hr("D. DOES COST SCALE WITH RULE COUNT R?")
    rng = np.random.default_rng(5)
    pts = rng.uniform(0, 1, size=(1200, 4))
    print(f"{'terms/input':>12} {'R':>8} {'us/call':>10}")
    print("-" * 74)
    out = []
    for m in (2, 3, 4, 5, 6, 8, 10):
        eng = _GenericMamdani(m)
        n = 1200 if m <= 5 else (400 if m <= 8 else 200)
        t = _timeit(eng.run, pts[:n])
        print(f"{m:>12} {eng.R:>8} {t*1e6:>10.1f}")
        out.append({"terms": m, "R": eng.R, "us_per_call": t * 1e6})

    Rs = np.array([o["R"] for o in out], float)
    ts = np.array([o["us_per_call"] for o in out], float)
    slope, intercept = np.polyfit(Rs, ts, 1)
    print(f"\n  least-squares fit:  cost ~ {intercept:.1f} us + {slope*1e3:.3f} us per 1000 rules")
    print("\n  Honest reading: once the inference is vectorized, the O(R) term is")
    print("  small at R=81 -- a fixed fuzzify/defuzzify overhead dominates. The")
    print("  rule-count dependence is real and is what a LUT removes entirely,")
    print("  but at this rule-base size it is NOT worth hundreds of milliseconds.")
    print("  Say this explicitly rather than letting a reviewer discover it.")
    out.append({"fit_intercept_us": float(intercept), "fit_slope_us_per_rule": float(slope)})
    return out


def main():
    print("FIS AUDIT  --  rule base + runtime, regenerated from source")
    print(f"numpy {np.__version__}")
    report = {"section_a": section_a(), "section_b": section_b(),
              "section_c": section_c(), "section_d": section_d()}
    with open("fis_audit_report.json", "w") as fh:
        json.dump(report, fh, indent=2, default=str)
    print("\nWritten: fis_audit_report.json")


if __name__ == "__main__":
    main()
