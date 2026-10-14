# -*- coding: utf-8 -*-
"""
policy_d_true_fis.py  --  Policy D: a GENUINE Mamdani-type Fuzzy Inference System.

═══════════════════════════════════════════════════════════════════════════
WHAT CHANGED vs. THE PREVIOUS VERSION (and why it matters for review)
═══════════════════════════════════════════════════════════════════════════

FIX 1 -- The rule base is no longer a disguised weighted sum.
    The previous `_rule_consequent()` generated all 81 consequents from
        score = sum_k w_k * goodness_k        (an additive, compensatory score)
    and then thresholded it into LOW/MEDIUM/HIGH.  That table is *exactly*
    representable by a linear scalarization plus two cut points, so a paper
    claiming "a FIS encodes non-compensatory reasoning a weighted sum cannot"
    was benchmarking a linear model against linear models.  A reviewer who
    opens the source sees this immediately.

    This version hand-authors a genuinely non-compensatory rule base in four
    tiers (vetoes / excellence gate / balance floor / residual band), and
    ships a machine-checkable proof that it is NOT representable by any
    non-negative weighted sum with monotone cut points:

        audit_non_compensatory()  ->  LP separability margin  eps* < 0
                                  ->  explicit arithmetic certificates

    Run `python fis_audit.py` to regenerate those numbers for the paper.

FIX 2 -- The runtime baseline no longer measures scikit-fuzzy's overhead.
    scikit-fuzzy rebuilds a NetworkX rule graph inside ControlSystemSimulation
    .compute() on *every* call.  ~99% of the previously reported ~322 ms/decision
    was that, not Mamdani inference.  This module now ships TWO interchangeable
    backends over the SAME rule base:

        backend="vectorized"  (default)  pure-NumPy Mamdani, ~tens of microseconds
        backend="skfuzzy"     (reference) the original library path

    Both are validated to agree to ~1e-12 (see validate_backends()).  Report
    all three in the paper -- skfuzzy FIS, vectorized FIS, and the lightweight
    policies -- so the runtime claim becomes "cost scales with rule count R",
    which is true, instead of "Mamdani is inherently slow", which is not.

    Switch globally with  set_backend("skfuzzy")  or the environment variable
    FIS_BACKEND=skfuzzy, or per-call with the `backend=` keyword.

PUBLIC API IS BACKWARD COMPATIBLE.  Every name a2e_relay_fixed_final_51.py and
surrogate_fis.py import still exists with the same signature.

IMPORTANT: the rule base changed, so every previously built LUT under luts/ is
stale.  build_surrogate_fis.py now stamps a rule-base fingerprint into each
.npz and surrogate_fis.py refuses to load a LUT whose fingerprint does not
match, rather than silently distilling the old rules.  Rebuild with:

    python build_surrogate_fis.py
"""

import hashlib
import itertools
import json
import os

import numpy as np

# ══════════════════════════════════════════════════════════════════════════
# 1. UNIVERSES  (unchanged -- kept identical so LUT axes and the paper's
#    Table III stay valid)
# ══════════════════════════════════════════════════════════════════════════
BATTERY_MARGIN_UNIVERSE = np.linspace(0, 60, 121)
DISTANCE_UNIVERSE       = np.linspace(0, 40, 81)
# NOTE (calibration): this universe was 0..0.06 in v1, but the model actually
# produces marginal coverage losses with a non-zero median around 0.17 and a
# tail past 0.6 (measured in-situ; see reserve_sweep.py --calibrate). Every
# non-zero loss therefore saturated at the top of the old universe, so the FIS
# could only tell "reserve (0)" from "patroller (maximally HIGH)" and had no
# resolution left inside the attribute it weights at 0.25. RelayScore, TOPSIS
# and VIKOR were unaffected because they normalise pool-relatively; only the
# FIS reads this absolute scale, which is why the mis-scaling stayed invisible.
# 0..0.6 covers the empirical p99. Re-run the calibration if you change the
# staleness cap, the threat/border penalties, or the grid size.
COVERAGE_LOSS_UNIVERSE  = np.linspace(0, 0.6, 121)
RELAY_DELAY_UNIVERSE    = np.linspace(0, 40, 81)
SUITABILITY_UNIVERSE    = np.linspace(0, 100, 101)

# Linguistic term order per antecedent, WORST-first, so index == "goodness".
TERMS = {
    "battery_margin": ("LOW", "MEDIUM", "HIGH"),     # LOW margin  = worst
    "distance":       ("FAR", "MEDIUM", "NEAR"),     # FAR         = worst
    "coverage_loss":  ("HIGH", "MEDIUM", "LOW"),     # HIGH loss   = worst
    "relay_delay":    ("LONG", "MEDIUM", "SHORT"),   # LONG delay  = worst
}
# Kept for backward compatibility with any external code that imported it.
_GOODNESS = {attr: {lbl: i for i, lbl in enumerate(labels)}
             for attr, labels in TERMS.items()}

CONSEQUENT_TERMS = ("LOW", "MEDIUM", "HIGH")

# ── CRITICAL: linguistic label -> position of its triangle on the NUMERIC
# universe.  0 = leftmost triangle (small numeric values), 1 = centre,
# 2 = rightmost (large numeric values).
#
# This is NOT the same ordering as TERMS above. TERMS is ordered worst-first
# so that its index is the attribute's "goodness"; for distance, coverage
# loss and relay delay, goodness runs OPPOSITE to the numeric value (a FAR
# distance is a LARGE number but the WORST outcome). Deriving the membership
# functions from the goodness order puts "NEAR" at distance=40 and "FAR" at
# distance=0, i.e. a FIS that deliberately picks the farthest candidate while
# every internal consistency check still passes. Keep these two tables
# separate, and keep validate_membership_orientation() in your test run.
MF_SLOT = {
    "battery_margin": {"LOW": 0, "MEDIUM": 1, "HIGH": 2},
    "distance":       {"NEAR": 0, "MEDIUM": 1, "FAR": 2},
    "coverage_loss":  {"LOW": 0, "MEDIUM": 1, "HIGH": 2},
    "relay_delay":    {"SHORT": 0, "MEDIUM": 1, "LONG": 2},
}
ANTECEDENT_ORDER = ("battery_margin", "distance", "coverage_loss", "relay_delay")

# Short attribute keys used inside the rule tiers below.
_SHORT = {"b": "battery_margin", "d": "distance", "c": "coverage_loss", "l": "relay_delay"}
_ATTR_NAME = {"b": "battery margin", "d": "distance",
              "c": "coverage loss", "l": "relay delay"}


# ══════════════════════════════════════════════════════════════════════════
# 2. RULE PROFILES -- the non-compensatory expert rule base
# ══════════════════════════════════════════════════════════════════════════
#
# Each profile is evaluated in strict tier order.  Tiers 1-3 are where the
# non-compensatory behaviour lives: they cannot be undone by being excellent
# on the remaining attributes.  Tier 4 only orders the cells the first three
# tiers left undecided, and that is the only place a weight vector appears.
#
#   TIER 1  VETOES          two attributes simultaneously at their WORST
#                           level  ->  LOW, unconditionally.
#   TIER 2  EXCELLENCE GATE HIGH requires (a) no attribute at its worst
#                           level, (b) at least `n_best_required` attributes
#                           at their best level, and (c) every gate attribute
#                           strictly above its worst level.
#   TIER 3  BALANCE FLOOR   `n_worst_low` or more attributes at their worst
#                           level  ->  LOW, however good the rest are.
#   TIER 4  RESIDUAL BAND   weighted ordinal score decides LOW vs MEDIUM
#                           among whatever is left.
#
# Tiers 1-3 are what break linear representability: a candidate that is
# excellent on three attributes and worst-in-class on the fourth is rated
# BELOW a balanced candidate whose additive score is lower.  That is the
# textbook definition of non-compensatory aggregation, and it is exactly
# what a weighted sum cannot express (proof: audit_non_compensatory()).

FUZZY_ATTR_WEIGHTS = {
    "distance": 0.30,
    "battery_margin": 0.30,
    "coverage_loss": 0.25,
    "relay_delay": 0.15,
}

INDEPENDENT_RULE_WEIGHTS = {
    "distance": 0.15,
    "battery_margin": 0.20,
    "coverage_loss": 0.50,
    "relay_delay": 0.15,
}

# "indep2": a SECOND independently-authored profile, added after we found
# that "indep"'s three vetoes (c,d)/(c,l)/(b,c) all involve coverage_loss,
# so they can never fire for an idle-reserve candidate (whose coverage_loss
# is by definition 0 = best, TERMS index 2 = never "worst"). Because most
# relay decisions in the default fleet are reserve-dominated, "indep" and
# "expert" turned out to agree on all but 4 of 81 cells -- all 4 in the
# coverage_loss=HIGH corner that reserve candidates never reach. That is
# not a genuine circularity test of the space the policies actually operate
# in. "indep2" is authored with vetoes/gates that never reference
# coverage_loss at all, so it stays active across the reserve-dominated
# subspace: it differs from "expert" on 22/81 cells overall and 8/27 cells
# within the coverage_loss=LOW slice specifically (vs. 0/27 for "indep"),
# and is independently verified non-compensatory (eps*=-0.20 < 0).
INDEPENDENT2_RULE_WEIGHTS = {
    "distance": 0.55,
    "battery_margin": 0.10,
    "coverage_loss": 0.10,
    "relay_delay": 0.25,
}

DEFAULT_RULE_WEIGHTS = dict(FUZZY_ATTR_WEIGHTS)


RULE_PROFILES = {
    # ── Profile "expert": SAFETY-FIRST. The operating rule base of Policy D.
    "expert": {
        "name": "expert",
        "description": (
            "Safety-first relay doctrine. A marginal-battery candidate is "
            "unusable for any trip that is long, slow, or that strips a "
            "critical zone; a critical zone must not be stripped by a slow "
            "arrival; top suitability demands balance, not one excellent "
            "attribute."
        ),
        "vetoes": [
            ("b", "d"),   # marginal battery + far trip -> unsafe, period
            ("b", "l"),   # marginal battery + long relay delay -> unsafe
            ("c", "l"),   # critical coverage loss + long delay -> double jeopardy
            ("b", "c"),   # marginal battery must not be pulled off a critical zone
        ],
        "excellence_gates": ["b", "c"],   # never HIGH if battery or coverage is worst
        "n_best_required": 1,
        "n_worst_low": 2,
        "lo_cut": 0.35,
        "weights": FUZZY_ATTR_WEIGHTS,
    },

    # ── Profile "indep": COVERAGE-FIRST, authored WITHOUT reference to
    #    RelayScore's weight vector. Used for the rule-base circularity check
    #    (paper Section V-D). Different vetoes, different gates, different
    #    residual weights -- and independently verified non-compensatory.
    "indep": {
        "name": "indep",
        "description": (
            "Coverage-first relay doctrine authored independently of "
            "RelayScore. Surveillance continuity dominates: a candidate that "
            "would strip a critical zone is unusable if it is also far, slow, "
            "or battery-marginal."
        ),
        "vetoes": [
            ("c", "d"),
            ("c", "l"),
            ("b", "c"),
        ],
        "excellence_gates": ["c", "d"],
        "n_best_required": 1,
        "n_worst_low": 2,
        "lo_cut": 0.40,
        "weights": INDEPENDENT_RULE_WEIGHTS,
    },

    # ── Profile "indep2": EFFICIENCY-FIRST, authored independently a second
    #    time, specifically so its vetoes/gates do NOT reference
    #    coverage_loss (see the comment above INDEPENDENT2_RULE_WEIGHTS).
    #    This keeps it genuinely active across the reserve-dominated part of
    #    the decision space where "indep" collapses onto "expert".
    "indep2": {
        "name": "indep2",
        "description": (
            "Efficiency-first relay doctrine authored independently of both "
            "'expert' and 'indep'. A candidate that is far AND slow is "
            "unusable regardless of battery or coverage; a marginal-battery "
            "candidate must never be pulled off a critical zone. Top "
            "suitability demands both a nearby trip and a healthy coverage "
            "outcome, not merely an absence of worst-case attributes."
        ),
        "vetoes": [
            ("d", "l"),   # far trip + long relay delay -> unsafe/inefficient
            ("b", "c"),   # marginal battery must not strip a critical zone
        ],
        "excellence_gates": ["c", "d"],
        "n_best_required": 2,
        "n_worst_low": 3,
        "lo_cut": 0.50,
        "weights": INDEPENDENT2_RULE_WEIGHTS,
    },
}

# weights-dict identity -> profile name, so the historical `weights=` keyword
# still selects the right rule base.
_PROFILE_FOR_WEIGHTS = {}


def _weights_key(weights):
    """Canonical, hashable key for a weight dict (unchanged public helper)."""
    return tuple(sorted((k, round(float(v), 6)) for k, v in weights.items()))


for _pname, _prof in RULE_PROFILES.items():
    _PROFILE_FOR_WEIGHTS[_weights_key(_prof["weights"])] = _pname


def _resolve_profile(weights):
    """Map a weights dict (or None, or a profile name) to a rule profile."""
    if weights is None:
        return RULE_PROFILES["expert"]
    if isinstance(weights, str):
        if weights not in RULE_PROFILES:
            raise ValueError(f"Unknown rule profile {weights!r}. "
                             f"Known: {sorted(RULE_PROFILES)}")
        return RULE_PROFILES[weights]
    key = _weights_key(weights)
    pname = _PROFILE_FOR_WEIGHTS.get(key)
    if pname is None:
        # A custom weight vector: keep the expert tier structure and swap in
        # the caller's residual weights. Still non-compensatory (tiers 1-3 are
        # weight-independent), but not one of the two audited profiles.
        prof = dict(RULE_PROFILES["expert"])
        prof["name"] = "custom_" + hashlib.sha1(
            repr(key).encode()).hexdigest()[:8]
        prof["weights"] = dict(weights)
        prof["description"] = "Custom residual weights, expert tier structure."
        return prof
    return RULE_PROFILES[pname]


def _consequent_for(goodness, prof):
    """Evaluate the four tiers for one rule.

    `goodness` is a dict {b,d,c,l} -> 0 (worst) | 1 | 2 (best).
    Returns "LOW" | "MEDIUM" | "HIGH".
    """
    g = goodness

    # ── TIER 1: hard vetoes ────────────────────────────────────────────────
    for a1, a2 in prof["vetoes"]:
        if g[a1] == 0 and g[a2] == 0:
            return "LOW"

    vals = (g["b"], g["d"], g["c"], g["l"])
    n_worst = sum(1 for v in vals if v == 0)
    n_best = sum(1 for v in vals if v == 2)

    # ── TIER 2: conjunctive excellence gate ────────────────────────────────
    if (n_worst == 0
            and n_best >= prof["n_best_required"]
            and all(g[a] >= 1 for a in prof["excellence_gates"])):
        return "HIGH"

    # ── TIER 3: balance floor ──────────────────────────────────────────────
    if n_worst >= prof["n_worst_low"]:
        return "LOW"

    # ── TIER 4: residual weighted band (the ONLY compensatory tier) ────────
    w = prof["weights"]
    score = (w["battery_margin"] * g["b"] + w["distance"] * g["d"]
             + w["coverage_loss"] * g["c"] + w["relay_delay"] * g["l"])
    denom = 2.0 * sum(w.values())
    frac = (score / denom) if denom > 0 else 0.0
    return "LOW" if frac < prof["lo_cut"] else "MEDIUM"


def rule_table(weights=None):
    """The full R=81 rule base as {(batt,dist,cov,delay) -> consequent}."""
    prof = _resolve_profile(weights)
    table = {}
    for b, d, c, l in itertools.product(TERMS["battery_margin"], TERMS["distance"],
                                        TERMS["coverage_loss"], TERMS["relay_delay"]):
        goodness = {"b": _GOODNESS["battery_margin"][b],
                    "d": _GOODNESS["distance"][d],
                    "c": _GOODNESS["coverage_loss"][c],
                    "l": _GOODNESS["relay_delay"][l]}
        table[(b, d, c, l)] = _consequent_for(goodness, prof)
    return table


def rule_table_as_text(weights=None):
    """Human-readable IF-THEN listing -- paste into the paper's appendix."""
    prof = _resolve_profile(weights)
    lines = [f"# Rule profile: {prof['name']}", f"# {prof['description']}", ""]
    for (b, d, c, l), cons in rule_table(weights).items():
        lines.append(
            f"IF battery_margin IS {b:<6} AND distance IS {d:<6} "
            f"AND coverage_loss IS {c:<6} AND relay_delay IS {l:<6} "
            f"THEN suitability IS {cons}"
        )
    return "\n".join(lines)


def rule_base_fingerprint(weights=None):
    """SHA-1 over rule table + universes + MF layout.

    Any change to the rules, the universes, or the membership functions
    changes this hash. build_surrogate_fis.py stamps it into every LUT and
    surrogate_fis.py refuses to load a LUT whose stamp does not match, which
    is what stops D2 from silently distilling a rule base that no longer
    exists.
    """
    prof = _resolve_profile(weights)
    payload = {
        "profile": prof["name"],
        "rules": [list(k) + [v] for k, v in sorted(rule_table(weights).items())],
        "universes": {
            "battery_margin": [float(BATTERY_MARGIN_UNIVERSE[0]),
                               float(BATTERY_MARGIN_UNIVERSE[-1]),
                               int(BATTERY_MARGIN_UNIVERSE.size)],
            "distance": [float(DISTANCE_UNIVERSE[0]), float(DISTANCE_UNIVERSE[-1]),
                         int(DISTANCE_UNIVERSE.size)],
            "coverage_loss": [float(COVERAGE_LOSS_UNIVERSE[0]),
                              float(COVERAGE_LOSS_UNIVERSE[-1]),
                              int(COVERAGE_LOSS_UNIVERSE.size)],
            "relay_delay": [float(RELAY_DELAY_UNIVERSE[0]), float(RELAY_DELAY_UNIVERSE[-1]),
                            int(RELAY_DELAY_UNIVERSE.size)],
            "suitability": [float(SUITABILITY_UNIVERSE[0]), float(SUITABILITY_UNIVERSE[-1]),
                            int(SUITABILITY_UNIVERSE.size)],
        },
        "mf": "triangular-3-term-symmetric",
        "implication": "min", "aggregation": "max", "defuzz": "centroid",
        "mf_slots": {a: dict(MF_SLOT[a]) for a in ANTECEDENT_ORDER},
        "engine_version": 3,
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha1(blob).hexdigest()


# ══════════════════════════════════════════════════════════════════════════
# 3. BACKEND A -- VECTORIZED NUMPY MAMDANI  (default; the FIX-2 path)
# ══════════════════════════════════════════════════════════════════════════
#
# Identical inference semantics to scikit-fuzzy:
#   fuzzify (triangular)  ->  AND = min  ->  implication = min (clip)
#   ->  aggregation = max  ->  defuzzification = centroid of the
#   piecewise-linear aggregate, computed segment-by-segment exactly the way
#   skfuzzy.defuzzify.centroid does it.
# Verified numerically by validate_backends(): max |vec - skfuzzy| ~ 1e-12.

_EPS = float(np.finfo(float).eps)


def _trimf_params(universe):
    """The 3 symmetric triangles this FIS uses on one antecedent universe,
    returned WORST-first so index == goodness index."""
    lo, hi = float(universe[0]), float(universe[-1])
    mid = (lo + hi) / 2.0
    # worst-first: shoulder at lo, centre, shoulder at hi
    return ((lo, lo, mid), (lo, mid, hi), (mid, hi, hi))


def _trimf_scalar(x, abc):
    """skfuzzy.trimf evaluated at a single point (degenerate edges included)."""
    a, b, c = abc
    if x == b:
        return 1.0
    if a != b and a < x < b:
        return (x - a) / (b - a)
    if b != c and b < x < c:
        return (c - x) / (c - b)
    return 0.0


def _trimf_array(universe, abc):
    a, b, c = abc
    y = np.zeros_like(universe, dtype=float)
    if a != b:
        idx = (universe > a) & (universe < b)
        y[idx] = (universe[idx] - a) / (b - a)
    if b != c:
        idx = (universe > b) & (universe < c)
        y[idx] = (c - universe[idx]) / (c - b)
    y[universe == b] = 1.0
    return y


def _centroid_like_skfuzzy(x, mfx):
    """Vectorized reproduction of skfuzzy.defuzzify.defuzz(..., 'centroid').

    skfuzzy walks consecutive (x1,y1)-(x2,y2) segments and sums moment*area
    for each trapezoid. The general trapezoid formula below reduces exactly to
    each of skfuzzy's four special cases, so results match bit-for-bit up to
    floating-point associativity.
    """
    x1, x2 = x[:-1], x[1:]
    y1, y2 = mfx[:-1], mfx[1:]
    dx = x2 - x1
    denom = y1 + y2
    live = (denom > 0.0) & (dx != 0.0)
    if not np.any(live):
        return float(np.sum(x * mfx) / max(float(np.sum(mfx)), _EPS))
    dxl, y1l, y2l, x1l, dn = dx[live], y1[live], y2[live], x1[live], denom[live]
    moment = x1l + (2.0 / 3.0 * dxl * (y2l + 0.5 * y1l)) / dn
    area = 0.5 * dxl * dn
    return float(np.sum(moment * area) / max(float(np.sum(area)), _EPS))


class _VectorizedMamdani:
    """Pre-compiled, allocation-light Mamdani engine for one rule profile.

    `exact_defuzz=True` (default) reproduces scikit-fuzzy's consequent-universe
    upsampling -- skfuzzy inserts the exact x positions where each clipped
    consequent triangle meets its activation level before taking the centroid.
    Skipping that step costs ~8e-3 of suitability score (harmless for ranking)
    but reproducing it makes the two backends agree to ~1e-13, which is what
    lets the paper state that the vectorized FIS is the same decision function,
    not an approximation of it.
    """

    __slots__ = ("profile", "_abc", "_uni", "_cons_mf", "_rule_idx",
                 "_masks", "_mu", "_strength", "_agg", "_fingerprint",
                 "exact_defuzz", "_su_lo", "_su_hi", "_su_half", "_extra_buf")

    def __init__(self, weights=None, exact_defuzz=True):
        self.exact_defuzz = bool(exact_defuzz)
        self.profile = _resolve_profile(weights)
        self._uni = (BATTERY_MARGIN_UNIVERSE, DISTANCE_UNIVERSE,
                     COVERAGE_LOSS_UNIVERSE, RELAY_DELAY_UNIVERSE)
        # triangles indexed by goodness, but pulled from the correct numeric slot
        self._abc = tuple(
            tuple(_trimf_params(self._uni[i])[MF_SLOT[attr][lbl]]
                  for lbl in TERMS[attr])
            for i, attr in enumerate(ANTECEDENT_ORDER)
        )

        # consequent membership functions on the suitability universe
        su = SUITABILITY_UNIVERSE
        lo, hi = float(su[0]), float(su[-1])
        mid = (lo + hi) / 2.0
        self._su_lo, self._su_hi, self._su_half = lo, hi, (hi - lo) / 2.0
        self._extra_buf = np.empty(4, dtype=float)
        self._cons_mf = np.vstack([
            _trimf_array(su, (lo, lo, mid)),    # LOW
            _trimf_array(su, (lo, mid, hi)),    # MEDIUM
            _trimf_array(su, (mid, hi, hi)),    # HIGH
        ])

        # rule consequent index cube, shape (3,3,3,3), axes = b,d,c,l (worst-first)
        idx = np.zeros((3, 3, 3, 3), dtype=np.int8)
        cons_i = {t: i for i, t in enumerate(CONSEQUENT_TERMS)}
        for (b, d, c, l), cons in rule_table(weights).items():
            idx[_GOODNESS["battery_margin"][b], _GOODNESS["distance"][d],
                _GOODNESS["coverage_loss"][c], _GOODNESS["relay_delay"][l]] = cons_i[cons]
        self._rule_idx = idx
        self._masks = tuple(idx == k for k in range(3))

        # reusable scratch buffers (no per-call allocation in the hot path)
        self._mu = np.empty((4, 3), dtype=float)
        self._strength = np.zeros(3, dtype=float)
        self._agg = np.empty_like(su)
        self._fingerprint = rule_base_fingerprint(weights)

    # -- hot path ---------------------------------------------------------
    def run(self, batt_val, dist_val, cov_val, delay_val):
        mu = self._mu
        for i, (x, abc) in enumerate(zip((batt_val, dist_val, cov_val, delay_val),
                                         self._abc)):
            u = self._uni[i]
            xv = float(x)
            if xv < u[0]:
                xv = float(u[0])
            elif xv > u[-1]:
                xv = float(u[-1])
            mu[i, 0] = _trimf_scalar(xv, abc[0])
            mu[i, 1] = _trimf_scalar(xv, abc[1])
            mu[i, 2] = _trimf_scalar(xv, abc[2])

        # AND = min, over the full 3x3x3x3 antecedent product
        alpha = np.minimum(
            np.minimum(mu[0].reshape(3, 1, 1, 1), mu[1].reshape(1, 3, 1, 1)),
            np.minimum(mu[2].reshape(1, 1, 3, 1), mu[3].reshape(1, 1, 1, 3)),
        )

        # aggregation over rules sharing a consequent term = max
        s = self._strength
        for k in range(3):
            sel = alpha[self._masks[k]]
            s[k] = sel.max() if sel.size else 0.0

        if self.exact_defuzz:
            return self._defuzz_exact(s)

        # implication (clip) + aggregation over consequent terms (max)
        agg = self._agg
        np.minimum(self._cons_mf[0], s[0], out=agg)
        np.maximum(agg, np.minimum(self._cons_mf[1], s[1]), out=agg)
        np.maximum(agg, np.minimum(self._cons_mf[2], s[2]), out=agg)

        if agg.max() <= 0.0:          # cannot happen with a full partition,
            return 50.0               # but never crash a 50-seed run over it
        return _centroid_like_skfuzzy(SUITABILITY_UNIVERSE, agg)

    def _defuzz_exact(self, s):
        """skfuzzy-identical defuzzification, including universe upsampling.

        skfuzzy locates the upsampling points numerically (np.diff over the
        discretised MF, then linear interpolation). Because the three
        consequent triangles have their vertices exactly on grid points, those
        numerically-found crossings coincide with the analytic ones below, so
        we compute them in closed form instead -- same points, no array scan.
        Guaranteed by validate_backends(): max |vec - skfuzzy| ~ 1e-13.
        """
        x = SUITABILITY_UNIVERSE
        lo, hi = self._su_lo, self._su_hi
        half = self._su_half                      # (hi - lo) / 2
        s0, s1, s2 = s[0], s[1], s[2]
        # LOW = tri(lo, lo, mid): falling leg crosses level s0
        # MEDIUM = tri(lo, mid, hi): rising and falling legs
        # HIGH = tri(mid, hi, hi): rising leg
        extra = self._extra_buf
        extra[0] = lo + half * (1.0 - s0)
        extra[1] = lo + half * s1
        extra[2] = hi - half * s1
        extra[3] = lo + half + half * s2
        new_u = np.sort(np.concatenate((x, extra)))
        agg = np.zeros_like(new_u)
        for k in range(3):
            um = np.interp(new_u, x, self._cons_mf[k])
            np.maximum(agg, np.minimum(s[k], um), out=agg)

        if agg.max() <= 0.0:
            return 50.0
        return _centroid_like_skfuzzy(new_u, agg)


_VEC_CACHE = {}


def _get_vectorized(weights=None):
    prof = _resolve_profile(weights)
    key = prof["name"]
    eng = _VEC_CACHE.get(key)
    if eng is None:
        eng = _VectorizedMamdani(weights)
        _VEC_CACHE[key] = eng
    return eng


# ══════════════════════════════════════════════════════════════════════════
# 4. BACKEND B -- SCIKIT-FUZZY REFERENCE  (kept for validation + the paper's
#    "library overhead" baseline row).  Imported lazily so the vectorized
#    path has zero skfuzzy dependency at runtime.
# ══════════════════════════════════════════════════════════════════════════

_SKF_CACHE = {}


def _get_skfuzzy_sim(weights=None):
    import skfuzzy as fuzz                     # noqa: F401  (lazy on purpose)
    from skfuzzy import control as ctrl

    prof = _resolve_profile(weights)
    key = prof["name"]
    if key in _SKF_CACHE:
        return _SKF_CACHE[key]

    tag = key
    battery_margin = ctrl.Antecedent(BATTERY_MARGIN_UNIVERSE, f"battery_margin_{tag}")
    distance       = ctrl.Antecedent(DISTANCE_UNIVERSE, f"distance_{tag}")
    coverage_loss  = ctrl.Antecedent(COVERAGE_LOSS_UNIVERSE, f"coverage_loss_{tag}")
    relay_delay    = ctrl.Antecedent(RELAY_DELAY_UNIVERSE, f"relay_delay_{tag}")
    suitability    = ctrl.Consequent(SUITABILITY_UNIVERSE, f"suitability_{tag}")

    for var, attr in ((battery_margin, "battery_margin"), (distance, "distance"),
                      (coverage_loss, "coverage_loss"), (relay_delay, "relay_delay")):
        abcs = _trimf_params(var.universe)
        for lbl in TERMS[attr]:
            var[lbl] = fuzz.trimf(var.universe, list(abcs[MF_SLOT[attr][lbl]]))

    su = suitability.universe
    lo, hi = float(su[0]), float(su[-1])
    mid = (lo + hi) / 2.0
    suitability["LOW"]    = fuzz.trimf(su, [lo, lo, mid])
    suitability["MEDIUM"] = fuzz.trimf(su, [lo, mid, hi])
    suitability["HIGH"]   = fuzz.trimf(su, [mid, hi, hi])

    rules = []
    for (b, d, c, l), cons in rule_table(weights).items():
        antecedent = (battery_margin[b] & distance[d]
                      & coverage_loss[c] & relay_delay[l])
        rules.append(ctrl.Rule(antecedent, suitability[cons]))

    system = ctrl.ControlSystem(rules)
    sim = ctrl.ControlSystemSimulation(system)
    bundle = (sim, battery_margin.label, distance.label, coverage_loss.label,
              relay_delay.label, suitability.label)
    _SKF_CACHE[key] = bundle
    return bundle


def _run_fis_skfuzzy(batt_val, dist_val, cov_val, delay_val, weights=None):
    sim, b_lbl, d_lbl, c_lbl, l_lbl, out_lbl = _get_skfuzzy_sim(weights)
    sim.input[b_lbl] = float(np.clip(batt_val, BATTERY_MARGIN_UNIVERSE[0],
                                     BATTERY_MARGIN_UNIVERSE[-1]))
    sim.input[d_lbl] = float(np.clip(dist_val, DISTANCE_UNIVERSE[0],
                                     DISTANCE_UNIVERSE[-1]))
    sim.input[c_lbl] = float(np.clip(cov_val, COVERAGE_LOSS_UNIVERSE[0],
                                     COVERAGE_LOSS_UNIVERSE[-1]))
    sim.input[l_lbl] = float(np.clip(delay_val, RELAY_DELAY_UNIVERSE[0],
                                     RELAY_DELAY_UNIVERSE[-1]))
    sim.compute()
    return float(sim.output[out_lbl])


def _run_fis_vectorized(batt_val, dist_val, cov_val, delay_val, weights=None):
    return _get_vectorized(weights).run(batt_val, dist_val, cov_val, delay_val)


# ══════════════════════════════════════════════════════════════════════════
# 5. BACKEND SELECTION
# ══════════════════════════════════════════════════════════════════════════

_BACKENDS = {"vectorized": _run_fis_vectorized, "skfuzzy": _run_fis_skfuzzy}
_ACTIVE_BACKEND = os.environ.get("FIS_BACKEND", "vectorized").strip().lower()
if _ACTIVE_BACKEND not in _BACKENDS:
    _ACTIVE_BACKEND = "vectorized"


def set_backend(name):
    """Globally switch the Mamdani implementation ('vectorized' | 'skfuzzy')."""
    global _ACTIVE_BACKEND
    name = str(name).strip().lower()
    if name not in _BACKENDS:
        raise ValueError(f"backend must be one of {sorted(_BACKENDS)}")
    _ACTIVE_BACKEND = name
    return _ACTIVE_BACKEND


def get_backend():
    return _ACTIVE_BACKEND


def _run_fis(batt_val, dist_val, cov_val, delay_val, weights=None, backend=None):
    """Evaluate the Mamdani FIS. Public entry point, signature unchanged."""
    fn = _BACKENDS[(backend or _ACTIVE_BACKEND)]
    return fn(batt_val, dist_val, cov_val, delay_val, weights=weights)


# ══════════════════════════════════════════════════════════════════════════
# 6. RELAY SELECTOR  (same contract as before, plus a documented tie-break)
# ══════════════════════════════════════════════════════════════════════════
#
# A non-compensatory rule base produces exact ties far more often than a
# weighted sum does (many candidates land in the same consequent band). The
# previous code resolved ties by candidate pool order, which is an artifact of
# fleet construction, not a decision rule. We resolve them explicitly and
# deterministically -- nearer first, then more battery margin, then id -- so
# the policy is reproducible and reviewers can see exactly what breaks a tie.

TIE_BREAK = True
_SUITABILITY_TIE_TOL = 1e-9


def _sort_key(score, distance, battery_margin, cand_id):
    if not TIE_BREAK:
        return (-score,)
    return (-round(score / _SUITABILITY_TIE_TOL),
            distance, -battery_margin, str(cand_id))


def select_fuzzy_relay_true_fis(needer, drones, g, detected, step,
                                weights=None, resolution=101, backend=None):
    """Policy D. `resolution` is accepted for signature compatibility only."""
    from a2e_relay_fixed_final_51 import (_feasible_candidate_pool, _manhattan,
                                          _coverage_loss_for_candidate)

    feasible = _feasible_candidate_pool(needer, drones)
    if not feasible:
        return None, []

    rows = []
    for cand, travel_energy, battery_margin_val in feasible:
        dist = _manhattan(cand["r"], cand["c"], needer["r"], needer["c"])
        relay_delay_val = dist
        coverage_loss_val, _raw, _tgap, _meta = _coverage_loss_for_candidate(
            cand, needer, g, detected, step, relay_delay_val, fleet=drones
        )
        rows.append({
            "cand": cand, "distance": dist, "travel_energy": travel_energy,
            "battery_margin": battery_margin_val, "coverage_loss": coverage_loss_val,
            "relay_delay": relay_delay_val,
        })

    if len(rows) == 1:
        only = rows[0]
        suitability_score = _run_fis(
            only["battery_margin"], only["distance"], only["coverage_loss"],
            only["relay_delay"], weights=weights, backend=backend,
        )
        diag = {
            "candidate_id": only["cand"]["id"], "distance": only["distance"],
            "travel_energy": round(only["travel_energy"], 3),
            "battery_margin": round(only["battery_margin"], 2),
            "coverage_loss": round(only["coverage_loss"], 6),
            "relay_delay": only["relay_delay"],
            "fis_suitability": round(suitability_score, 2),
            "note": "only feasible candidate",
        }
        return only["cand"], [diag]

    scored = []
    for r in rows:
        suitability_score = _run_fis(
            r["battery_margin"], r["distance"], r["coverage_loss"],
            r["relay_delay"], weights=weights, backend=backend,
        )
        diag = {
            "candidate_id": r["cand"]["id"], "role": r["cand"]["role"],
            "distance": r["distance"], "travel_energy": round(r["travel_energy"], 3),
            "battery_margin": round(r["battery_margin"], 2),
            "coverage_loss": round(r["coverage_loss"], 6),
            "relay_delay": r["relay_delay"],
            "fis_suitability": round(suitability_score, 2),
        }
        scored.append((r["cand"], suitability_score, diag, r))

    scored.sort(key=lambda e: _sort_key(e[1], e[3]["distance"],
                                        e[3]["battery_margin"],
                                        e[0].get("id", "")))
    return scored[0][0], [d for _, _, d, _ in scored]


def select_fuzzy_relay_true_fis_indep_weights(needer, drones, g, detected, step,
                                              resolution=101, backend=None):
    """Policy D-indep: the independently-authored coverage-first rule base."""
    return select_fuzzy_relay_true_fis(
        needer, drones, g, detected, step,
        weights=INDEPENDENT_RULE_WEIGHTS, resolution=resolution, backend=backend,
    )


def select_fuzzy_relay_true_fis_indep2_weights(needer, drones, g, detected, step,
                                               resolution=101, backend=None):
    """Policy D-indep2: a second independently-authored rule base whose
    vetoes/gates never reference coverage_loss, so it stays genuinely
    distinct from 'expert' across the reserve-dominated decision subspace
    where 'indep' collapses onto it (see INDEPENDENT2_RULE_WEIGHTS)."""
    return select_fuzzy_relay_true_fis(
        needer, drones, g, detected, step,
        weights=INDEPENDENT2_RULE_WEIGHTS, resolution=resolution, backend=backend,
    )


def select_fuzzy_relay_true_fis_skfuzzy(needer, drones, g, detected, step,
                                        weights=None, resolution=101):
    """Policy Dref: identical rules, forced onto the scikit-fuzzy backend.

    Register this as its own policy label so the paper can report the
    library-overhead baseline and the vectorized FIS side by side instead of
    conflating them.
    """
    return select_fuzzy_relay_true_fis(needer, drones, g, detected, step,
                                       weights=weights, resolution=resolution,
                                       backend="skfuzzy")


# ══════════════════════════════════════════════════════════════════════════
# 7. AUDIT  --  the machine-checkable evidence for the paper
# ══════════════════════════════════════════════════════════════════════════

def linear_separability_margin(weights=None, table=None):
    """Largest margin eps* of any non-negative weighted sum + cut points that
    reproduces the rule table.

        eps* > 0   the table IS a linear scalarization  (compensatory)
        eps* <= 0  NO weighted sum reproduces it        (non-compensatory)

    Solved as an LP over (w in R^4_+, theta1, theta2, eps) with sum(w)=1.
    """
    from scipy.optimize import linprog

    tab = table if table is not None else rule_table(weights)
    order = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
    A_ub, b_ub = [], []
    for (b, d, c, l), cons in tab.items():
        gv = np.array([_GOODNESS["battery_margin"][b], _GOODNESS["distance"][d],
                       _GOODNESS["coverage_loss"][c], _GOODNESS["relay_delay"][l]],
                      dtype=float)
        k = order[cons]
        if k == 0:                       # w.g <= theta1
            row = np.zeros(7); row[:4] = gv; row[4] = -1.0
            A_ub.append(row); b_ub.append(0.0)
        elif k == 1:                     # theta1 + eps <= w.g <= theta2
            row = np.zeros(7); row[:4] = -gv; row[4] = 1.0; row[6] = 1.0
            A_ub.append(row); b_ub.append(0.0)
            row = np.zeros(7); row[:4] = gv; row[5] = -1.0
            A_ub.append(row); b_ub.append(0.0)
        else:                            # w.g >= theta2 + eps
            row = np.zeros(7); row[:4] = -gv; row[5] = 1.0; row[6] = 1.0
            A_ub.append(row); b_ub.append(0.0)
    row = np.zeros(7); row[4] = 1.0; row[5] = -1.0; row[6] = 1.0   # th1+eps <= th2
    A_ub.append(row); b_ub.append(0.0)

    obj = np.zeros(7); obj[6] = -1.0                                # maximise eps
    res = linprog(obj, A_ub=np.array(A_ub), b_ub=np.array(b_ub),
                  A_eq=[[1, 1, 1, 1, 0, 0, 0]], b_eq=[1.0],
                  bounds=[(0, None)] * 4 + [(-10, 10), (-10, 10), (None, 1.0)],
                  method="highs")
    return (float(res.x[6]) if res.success else float("-inf"))


def non_compensatory_certificates(weights=None, table=None, limit=5):
    """Elementary (LP-free) proofs of non-representability.

    Returns triples (A, B, C) whose ordinal goodness vectors satisfy
    g(C) = (g(A)+g(B))/2 while the consequents are impossible for any
    non-negative weighted sum: e.g. A and B both below HIGH but C HIGH, since
    then s(C) = (s(A)+s(B))/2 < theta2 for every weight vector.
    A reviewer can verify one of these by hand in 30 seconds.
    """
    tab = table if table is not None else rule_table(weights)
    order = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
    keys = list(tab.keys())
    gvec = {k: np.array([_GOODNESS["battery_margin"][k[0]], _GOODNESS["distance"][k[1]],
                         _GOODNESS["coverage_loss"][k[2]], _GOODNESS["relay_delay"][k[3]]],
                        dtype=float) for k in keys}
    by_g = {}
    for k in keys:
        by_g[tuple(gvec[k])] = k

    out = []
    for A, B in itertools.combinations(keys, 2):
        mid = (gvec[A] + gvec[B]) / 2.0
        if np.any(mid != np.round(mid)):
            continue
        C = by_g.get(tuple(mid))
        if C is None:
            continue
        ca, cb, cc = order[tab[A]], order[tab[B]], order[tab[C]]
        if cc == 2 and ca < 2 and cb < 2:
            out.append(("midpoint rated HIGH while both endpoints are below HIGH",
                        A, tab[A], B, tab[B], C, tab[C]))
        elif cc == 0 and ca > 0 and cb > 0:
            out.append(("midpoint rated LOW while both endpoints are above LOW",
                        A, tab[A], B, tab[B], C, tab[C]))
        if len(out) >= limit:
            break
    return out


def legacy_compensatory_table(weights=None):
    """The OLD (v1) auto-generated rule table, for the before/after contrast."""
    w = dict(weights or FUZZY_ATTR_WEIGHTS)
    tab = {}
    for b, d, c, l in itertools.product(TERMS["battery_margin"], TERMS["distance"],
                                        TERMS["coverage_loss"], TERMS["relay_delay"]):
        score = (w["battery_margin"] * _GOODNESS["battery_margin"][b]
                 + w["distance"] * _GOODNESS["distance"][d]
                 + w["coverage_loss"] * _GOODNESS["coverage_loss"][c]
                 + w["relay_delay"] * _GOODNESS["relay_delay"][l])
        frac = score / (2.0 * sum(w.values()))
        tab[(b, d, c, l)] = "LOW" if frac < 1 / 3 else ("MEDIUM" if frac < 2 / 3 else "HIGH")
    return tab


def audit_non_compensatory(weights=None):
    """One-call summary: class balance, LP margin, certificates."""
    tab = rule_table(weights)
    counts = {t: sum(1 for v in tab.values() if v == t) for t in CONSEQUENT_TERMS}
    eps = linear_separability_margin(table=tab)
    certs = non_compensatory_certificates(table=tab, limit=3)
    return {
        "profile": _resolve_profile(weights)["name"],
        "fingerprint": rule_base_fingerprint(weights),
        "rule_count": len(tab),
        "class_counts": counts,
        "separability_margin": eps,
        "is_weighted_sum": bool(eps > 1e-9),
        "certificates": certs,
    }


def validate_membership_orientation(weights=None):
    """Assert every linguistic term peaks where its NAME says it should.

    This is the check that would have caught the inverted-membership bug:
    NEAR must peak at the smallest distance, FAR at the largest, SHORT at the
    smallest delay, LOW coverage loss at the smallest loss, LOW battery margin
    at the smallest margin. Two backends agreeing with each other proves
    nothing if both share the same wrong table -- this compares each label
    against the physical meaning of its universe instead.

    Returns a list of (attribute, label, peak_value, expected_end) rows and
    raises AssertionError on any mismatch.
    """
    eng = _get_vectorized(weights)
    expect_low_end = {
        ("battery_margin", "LOW"), ("distance", "NEAR"),
        ("coverage_loss", "LOW"), ("relay_delay", "SHORT"),
    }
    expect_high_end = {
        ("battery_margin", "HIGH"), ("distance", "FAR"),
        ("coverage_loss", "HIGH"), ("relay_delay", "LONG"),
    }
    rows, bad = [], []
    for i, attr in enumerate(ANTECEDENT_ORDER):
        u = eng._uni[i]
        lo, hi = float(u[0]), float(u[-1])
        for gidx, lbl in enumerate(TERMS[attr]):
            peak = float(eng._abc[i][gidx][1])
            if (attr, lbl) in expect_low_end:
                ok, want = abs(peak - lo) < 1e-9, "low end"
            elif (attr, lbl) in expect_high_end:
                ok, want = abs(peak - hi) < 1e-9, "high end"
            else:
                ok, want = abs(peak - (lo + hi) / 2.0) < 1e-9, "centre"
            rows.append((attr, lbl, peak, want, ok))
            if not ok:
                bad.append(f"{attr}.{lbl} peaks at {peak} but should sit at the {want} "
                           f"of [{lo}, {hi}]")
    if bad:
        raise AssertionError("Inverted membership functions:\n  " + "\n  ".join(bad))
    return rows


def validate_monotonicity(weights=None, n=400, seed=3):
    """Sanity check on the decision surface itself.

    Holding three inputs fixed, suitability must not increase when distance,
    coverage loss or relay delay increases, and must not decrease when battery
    margin increases. A rule base can be non-compensatory and still be
    monotone in each attribute; if this fails, something is wired backwards.
    """
    rng = np.random.default_rng(seed)
    axes = [BATTERY_MARGIN_UNIVERSE, DISTANCE_UNIVERSE,
            COVERAGE_LOSS_UNIVERSE, RELAY_DELAY_UNIVERSE]
    want_up = (True, False, False, False)      # higher input -> higher suitability?
    report = {}
    for k, attr in enumerate(ANTECEDENT_ORDER):
        worst = 0.0
        for _ in range(n):
            base = [float(rng.uniform(a[0], a[-1])) for a in axes]
            lo_v, hi_v = sorted(rng.uniform(axes[k][0], axes[k][-1], 2))
            a_in = list(base); a_in[k] = lo_v
            b_in = list(base); b_in[k] = hi_v
            sa = _run_fis(*a_in, weights=weights)
            sb = _run_fis(*b_in, weights=weights)
            viol = (sa - sb) if want_up[k] else (sb - sa)
            worst = max(worst, viol)
        report[attr] = {"direction": "increasing" if want_up[k] else "decreasing",
                        "max_violation": float(worst)}
    return report


def validate_backends(n=1500, weights=None, seed=0):
    """Numerical agreement between the vectorized engine and scikit-fuzzy."""
    rng = np.random.default_rng(seed)
    pts = np.column_stack([
        rng.uniform(BATTERY_MARGIN_UNIVERSE[0], BATTERY_MARGIN_UNIVERSE[-1], n),
        rng.uniform(DISTANCE_UNIVERSE[0], DISTANCE_UNIVERSE[-1], n),
        rng.uniform(COVERAGE_LOSS_UNIVERSE[0], COVERAGE_LOSS_UNIVERSE[-1], n),
        rng.uniform(RELAY_DELAY_UNIVERSE[0], RELAY_DELAY_UNIVERSE[-1], n),
    ])
    a = np.array([_run_fis_vectorized(*p, weights=weights) for p in pts])
    b = np.array([_run_fis_skfuzzy(*p, weights=weights) for p in pts])
    d = np.abs(a - b)
    return {"n": n, "max_abs_err": float(d.max()),
            "rmse": float(np.sqrt(np.mean(d ** 2))),
            "mean_abs_err": float(d.mean())}


if __name__ == "__main__":
    for name in ("expert", "indep", "indep2"):
        rep = audit_non_compensatory(name)
        print(f"[{rep['profile']}] R={rep['rule_count']} {rep['class_counts']} "
              f"eps*={rep['separability_margin']:+.4f} "
              f"is_weighted_sum={rep['is_weighted_sum']} "
              f"fp={rep['fingerprint'][:12]}")
