# -*- coding: utf-8 -*-
"""
surrogate_fis.py  --  Policy D2: the distilled lookup-table surrogate.

═══════════════════════════════════════════════════════════════════════════
WHAT CHANGED, AND WHY YOUR D2 STORY HAS TO CHANGE WITH IT
═══════════════════════════════════════════════════════════════════════════

1. STALE-LUT GUARD.  policy_d_true_fis.py's rule base is no longer the old
   auto-generated weighted-sum table, so every .npz under luts/ built before
   that change encodes a rule base that no longer exists. Loading one would
   make D2 "distil" a FIS that is not the FIS being compared against -- a
   silent, invisible experimental error. Every LUT now carries the rule-base
   fingerprint it was built from, and _load_interpolator() refuses to run if
   it does not match. Rebuild with:  python build_surrogate_fis.py

2. FAST LOOKUP.  scipy's RegularGridInterpolator costs ~100 us per scalar
   query, almost all of it Python-level argument validation. That is the same
   order as the vectorized Mamdani FIS itself, which would leave D2 with no
   runtime advantage at all to report. The runtime path is now a hand-rolled
   4-D multilinear interpolation on a uniform grid: 16 corner lookups and one
   dot product, ~5 us. RegularGridInterpolator is retained as a reference
   backend and the two are checked against each other in validate_backends().

3. HONEST FRAMING FOR THE PAPER.  Once the true FIS is implemented properly
   (tens of microseconds instead of tens of milliseconds), D2 is no longer
   "two orders of magnitude faster than the FIS". Its defensible advantage
   becomes the one that is actually structural:

       the true FIS costs O(Nc * R); D2 costs O(Nc), independent of R.

   At R=81 the gap is modest. Report the rule-count scaling curve from
   fis_audit.py (R = 81, 243, 729, 2187) instead of a single ratio -- that is
   the claim that survives a reviewer reimplementing your FIS efficiently.

Drop-in API (unchanged):
    _run_fis_surrogate(batt, dist, cov, delay, weights=None) -> float
    select_fuzzy_relay_surrogate_fis(needer, drones, g, detected, step, ...)
    select_fuzzy_relay_surrogate_fis_indep_weights(...)
"""

import glob
import os

import numpy as np

from policy_d_true_fis import (
    _weights_key,
    _resolve_profile,
    rule_base_fingerprint,
    FUZZY_ATTR_WEIGHTS,
    INDEPENDENT_RULE_WEIGHTS,
    BATTERY_MARGIN_UNIVERSE,
    DISTANCE_UNIVERSE,
    COVERAGE_LOSS_UNIVERSE,
    RELAY_DELAY_UNIVERSE,
    TIE_BREAK,
    _SUITABILITY_TIE_TOL,
)

_LUT_DIR_DEFAULT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "luts")

# profile name -> LUT file label (build_surrogate_fis.py uses the same map)
_LABEL_FOR_PROFILE = {"expert": "shared", "indep": "indep"}

# Back-compat: the old module exposed this weights-keyed map.
_LABEL_FOR_WEIGHTS = {
    _weights_key(FUZZY_ATTR_WEIGHTS): "shared",
    _weights_key(INDEPENDENT_RULE_WEIGHTS): "indep",
}

# If True, a fingerprint mismatch raises instead of warning. Keep it True --
# a warning in a 50-seed sweep scrolls past unnoticed and poisons the results.
STRICT_FINGERPRINT = True

_LUT_CACHE = {}        # (profile, dir, res) -> _FastMultilinear
_SCIPY_CACHE = {}      # same key -> RegularGridInterpolator


class StaleLutError(RuntimeError):
    """Raised when a LUT was built from a different rule base."""


# ══════════════════════════════════════════════════════════════════════════
# Fast uniform-grid multilinear interpolation (the runtime path)
# ══════════════════════════════════════════════════════════════════════════
class _FastMultilinear:
    """4-D piecewise-multilinear interpolation on a uniform grid.

    Mathematically identical to
    RegularGridInterpolator(method="linear", bounds_error=False), for queries
    inside the grid; outside, the query is clamped to the grid (the universes
    are clipped upstream anyway, so this never fires in the simulation).
    """

    __slots__ = ("axes", "los", "steps", "ns", "flat", "offsets", "meta",
                 "_w", "_pairs")

    def __init__(self, axes, values, meta=None):
        self.axes = [np.ascontiguousarray(a, dtype=float) for a in axes]
        for a in self.axes:
            if a.size < 2:
                raise ValueError("every LUT axis needs at least 2 nodes")
            d = np.diff(a)
            if not np.allclose(d, d[0], rtol=1e-9, atol=1e-12):
                raise ValueError("_FastMultilinear requires uniformly spaced axes")
        self.los = [float(a[0]) for a in self.axes]
        self.steps = [float(a[1] - a[0]) for a in self.axes]
        self.ns = [int(a.size) for a in self.axes]

        values = np.ascontiguousarray(values, dtype=float)
        if values.shape != tuple(self.ns):
            raise ValueError(f"LUT value shape {values.shape} != axes {tuple(self.ns)}")
        self.flat = values.ravel()

        n0, n1, n2, n3 = self.ns
        strides = (n1 * n2 * n3, n2 * n3, n3, 1)
        self.offsets = np.array(
            [b0 * strides[0] + b1 * strides[1] + b2 * strides[2] + b3 * strides[3]
             for b0 in (0, 1) for b1 in (0, 1) for b2 in (0, 1) for b3 in (0, 1)],
            dtype=np.intp,
        )
        self._w = np.empty(16, dtype=float)
        self._pairs = np.empty((4, 2), dtype=float)
        self.meta = meta or {}

    def __call__(self, b, d, c, dl):
        pairs = self._pairs
        cell = [0, 0, 0, 0]
        for k, x in enumerate((b, d, c, dl)):
            lo, step, n = self.los[k], self.steps[k], self.ns[k]
            pos = (float(x) - lo) / step
            i = int(pos)
            if i < 0:
                i, t = 0, 0.0
            elif i > n - 2:
                i, t = n - 2, 1.0
            else:
                t = pos - i
            cell[k] = i
            pairs[k, 0] = 1.0 - t
            pairs[k, 1] = t

        n0, n1, n2, n3 = self.ns
        base = ((cell[0] * n1 + cell[1]) * n2 + cell[2]) * n3 + cell[3]

        w = self._w
        p0, p1, p2, p3 = pairs[0], pairs[1], pairs[2], pairs[3]
        m = 0
        for a in (0, 1):
            wa = p0[a]
            for bb in (0, 1):
                wab = wa * p1[bb]
                for cc in (0, 1):
                    wabc = wab * p2[cc]
                    w[m] = wabc * p3[0]
                    w[m + 1] = wabc * p3[1]
                    m += 2
        return float(np.dot(self.flat[base + self.offsets], w))


# ══════════════════════════════════════════════════════════════════════════
# LUT discovery / loading
# ══════════════════════════════════════════════════════════════════════════
def _label_for(weights):
    prof = _resolve_profile(weights)
    label = _LABEL_FOR_PROFILE.get(prof["name"])
    if label is None:
        raise ValueError(
            f"No LUT label registered for rule profile {prof['name']!r}. "
            "surrogate_fis ships LUTs for the two audited profiles only "
            "(FUZZY_ATTR_WEIGHTS -> 'shared', INDEPENDENT_RULE_WEIGHTS -> "
            "'indep'). Build one with build_surrogate_fis.build_lut(...)."
        )
    return label


def _find_lut_file(label, lut_dir, resolution=None):
    if resolution is not None:
        path = os.path.join(lut_dir, f"lut_{label}_res{resolution}.npz")
        if os.path.exists(path):
            return path
        raise FileNotFoundError(f"No LUT found at {path}")
    candidates = glob.glob(os.path.join(lut_dir, f"lut_{label}_res*.npz"))
    if not candidates:
        raise FileNotFoundError(
            f"No LUT found for label='{label}' in {lut_dir}. "
            f"Run: python build_surrogate_fis.py"
        )

    def _res_of(p):
        return int(os.path.basename(p).split("_res")[1].split(".npz")[0])

    return max(candidates, key=_res_of)


def _check_fingerprint(data, path, weights):
    expected = rule_base_fingerprint(weights)
    stored = data["rule_base_fingerprint"].item() if "rule_base_fingerprint" in data.files else None
    if isinstance(stored, bytes):
        stored = stored.decode()
    if stored == expected:
        return
    msg = (
        "\n" + "=" * 72 +
        f"\nSTALE LUT: {path}"
        f"\n  LUT was built from rule base : {stored or '<none recorded: pre-fingerprint file>'}"
        f"\n  policy_d_true_fis.py now has : {expected}"
        "\nThe rule base changed, so this LUT no longer distils the FIS that"
        "\nPolicy D actually runs. Any D2-vs-D comparison using it is invalid."
        "\nFix:   rm -rf luts && python build_surrogate_fis.py"
        "\n" + "=" * 72
    )
    if STRICT_FINGERPRINT:
        raise StaleLutError(msg)
    import warnings
    warnings.warn(msg, RuntimeWarning, stacklevel=2)


def _load_interpolator(weights, lut_dir=_LUT_DIR_DEFAULT, resolution=None,
                       backend="fast"):
    prof = _resolve_profile(weights)
    cache = _LUT_CACHE if backend == "fast" else _SCIPY_CACHE
    ck = (prof["name"], lut_dir, resolution)
    if ck in cache:
        return cache[ck]

    label = _label_for(weights)
    path = _find_lut_file(label, lut_dir, resolution)
    data = np.load(path, allow_pickle=False)
    _check_fingerprint(data, path, weights)

    axes = (data["grid_b"], data["grid_d"], data["grid_c"], data["grid_dl"])
    values = data["output"]
    meta = {
        "path": path,
        "resolution": int(data["resolution"]),
        "build_time_sec": float(data["build_time_sec"]),
        "profile": prof["name"],
        "fingerprint": rule_base_fingerprint(weights),
        "backend": backend,
    }

    if backend == "fast":
        obj = _FastMultilinear(axes, values, meta)
    else:
        from scipy.interpolate import RegularGridInterpolator
        rgi = RegularGridInterpolator(axes, values, method="linear",
                                      bounds_error=False, fill_value=None)
        obj = rgi

    cache[ck] = (obj, meta)
    return obj, meta


# ══════════════════════════════════════════════════════════════════════════
# Query
# ══════════════════════════════════════════════════════════════════════════
def _run_fis_surrogate(batt_val, dist_val, cov_val, delay_val, weights=None,
                       lut_dir=_LUT_DIR_DEFAULT, resolution=None, backend="fast"):
    interp, _meta = _load_interpolator(weights, lut_dir=lut_dir,
                                       resolution=resolution, backend=backend)

    b = float(np.clip(batt_val, BATTERY_MARGIN_UNIVERSE[0], BATTERY_MARGIN_UNIVERSE[-1]))
    d = float(np.clip(dist_val, DISTANCE_UNIVERSE[0], DISTANCE_UNIVERSE[-1]))
    c = float(np.clip(cov_val, COVERAGE_LOSS_UNIVERSE[0], COVERAGE_LOSS_UNIVERSE[-1]))
    dl = float(np.clip(delay_val, RELAY_DELAY_UNIVERSE[0], RELAY_DELAY_UNIVERSE[-1]))

    if backend == "fast":
        return interp(b, d, c, dl)
    return float(interp([[b, d, c, dl]])[0])


# ══════════════════════════════════════════════════════════════════════════
# Selector -- structural mirror of policy_d_true_fis.select_fuzzy_relay_true_fis
# (identical tie-break, so D2-vs-D differences can only come from the LUT)
# ══════════════════════════════════════════════════════════════════════════
def _sort_key(score, distance, battery_margin, cand_id):
    if not TIE_BREAK:
        return (-score,)
    return (-round(score / _SUITABILITY_TIE_TOL), distance, -battery_margin, str(cand_id))


def select_fuzzy_relay_surrogate_fis(needer, drones, g, detected, step, weights=None,
                                     lut_dir=_LUT_DIR_DEFAULT, resolution=None,
                                     backend="fast"):
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
        suitability_score = _run_fis_surrogate(
            only["battery_margin"], only["distance"], only["coverage_loss"],
            only["relay_delay"], weights=weights, lut_dir=lut_dir,
            resolution=resolution, backend=backend,
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
        suitability_score = _run_fis_surrogate(
            r["battery_margin"], r["distance"], r["coverage_loss"],
            r["relay_delay"], weights=weights, lut_dir=lut_dir,
            resolution=resolution, backend=backend,
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


def select_fuzzy_relay_surrogate_fis_indep_weights(needer, drones, g, detected, step,
                                                   lut_dir=_LUT_DIR_DEFAULT,
                                                   resolution=None, backend="fast"):
    return select_fuzzy_relay_surrogate_fis(
        needer, drones, g, detected, step, weights=INDEPENDENT_RULE_WEIGHTS,
        lut_dir=lut_dir, resolution=resolution, backend=backend,
    )


# ══════════════════════════════════════════════════════════════════════════
# Validation helper
# ══════════════════════════════════════════════════════════════════════════
def validate_backends(n=1000, weights=None, lut_dir=_LUT_DIR_DEFAULT,
                      resolution=None, seed=0):
    """Fast multilinear vs scipy RegularGridInterpolator on random points."""
    rng = np.random.default_rng(seed)
    pts = np.column_stack([
        rng.uniform(BATTERY_MARGIN_UNIVERSE[0], BATTERY_MARGIN_UNIVERSE[-1], n),
        rng.uniform(DISTANCE_UNIVERSE[0], DISTANCE_UNIVERSE[-1], n),
        rng.uniform(COVERAGE_LOSS_UNIVERSE[0], COVERAGE_LOSS_UNIVERSE[-1], n),
        rng.uniform(RELAY_DELAY_UNIVERSE[0], RELAY_DELAY_UNIVERSE[-1], n),
    ])
    a = np.array([_run_fis_surrogate(*p, weights=weights, lut_dir=lut_dir,
                                     resolution=resolution, backend="fast") for p in pts])
    b = np.array([_run_fis_surrogate(*p, weights=weights, lut_dir=lut_dir,
                                     resolution=resolution, backend="scipy") for p in pts])
    d = np.abs(a - b)
    return {"n": n, "max_abs_err": float(d.max()), "rmse": float(np.sqrt(np.mean(d ** 2)))}
