import math, random, json, statistics, threading, os, gzip, time
import tkinter as tk
from tkinter import messagebox
import numpy as np
from scipy.optimize import linprog, minimize, milp, LinearConstraint, Bounds
from scipy.stats import wilcoxon, ttest_rel

# ══════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════
#  OWT-A* PHYSICALLY-DERIVED ENERGY MODEL  (inlined from owt_physics.py)
# ══════════════════════════════════════════════════════════════
#  Grid model: 2.5D — the underlying (row, col) grid stays exactly as-is.
#  Elevation is stored as a per-cell ATTRIBUTE (from real SRTM data), not
#  a third movement axis. This lets every existing SMRS/return-to-base/
#  adversary-model function keep working unchanged; only the ENERGY
#  computation inside a step is replaced with the derived physics below.
#
#  Three physically-derived terms, replacing the old heuristic versions:
#    1. Wind cost  — vector-triangle required-airspeed + cubic drag power law
#    2. Turn cost  — momentum-change + actuator disk momentum theory
#    3. Climb cost — gravitational potential energy + induced climb power
# ══════════════════════════════════════════════════════════════

class OWTPhysicsConfig:
    """UAV physical configuration—anchored to DJI Matrice 30 Series Enterprise(https://enterprise.dji.com/matrice-30)."""
    def __init__(self):
        # --- Mass / airframe ---
        # Ref: DJI Matrice 30 Series Enterprise Datasheet
        self.drone_mass       = 3.77     # kg (Takeoff weight including two TB30 batteries)
        self.payload_mass     = 0.20     # kg (Typical extra payload/camera margin)
        self.g                = 9.81     # m/s^2

        # --- Flight speed ---
        # Ref: DJI Matrice 30 specs (Cruise speed ~15 m/s)
        self.cruise_speed     = 15.0     # m/s 
        self.max_airspeed_cap = 23.0     # m/s hardware limit (Max speed in sport mode)

        # --- Aerodynamics ---
        self.rho_standard     = 1.225    # kg/m^3
        # Ref: Drag coefficient for enterprise quadrotors
        self.drag_coefficient = 0.54     # Cd (Typical for sleek enterprise drones)
        self.frontal_area     = 0.10     # m^2 (Estimated cross-section from dimensions)
        self.rotor_disk_area  = 0.25     # m^2 (Swept area of 16-inch propellers)

        # --- Turn kinematics (Restored from original) ---
        self.turn_duration_estimate = 1.5   # seconds, typical maneuver time
        self.turn_threshold   = 0.05        # radians (~3deg), below this: no turn penalty

        # --- Grid-to-real-world scaling (Restored from original) ---
        self.cell_size_m      = 100.0    # metres per grid cell (synthetic-terrain default)

        # --- Battery capacity ---
        # Ref: DJI TB30 Intelligent Flight Battery specs (2x 131.6 Wh)
        self.battery_capacity_wh = 263.2 
        self.battery_capacity_joules = self.battery_capacity_wh * 3600.0

        # Ref: UAV Battery management systems survey (Lithium-ion max DOD)
        self.usable_capacity_fraction = 0.85 # 85% depth of discharge to prevent battery damage
        self.usable_capacity_joules = self.battery_capacity_joules * self.usable_capacity_fraction

        # --- State of Health (SOH) / capacity-aging model ---
        # Ref: general Li-ion cycle-aging literature (e.g. NREL/Sandia
        # cycle-life studies, UAV battery-management surveys). Real packs
        # lose capacity roughly linearly with full-equivalent-cycle count
        # over the useful part of their life, then are typically retired
        # once they hit an "end-of-life" capacity threshold -- widely
        # cited as ~80% of rated capacity for Li-ion. These numbers are
        # NOT TB30-specific (DJI doesn't publish a cycle-fade curve); they
        # are representative mid-range literature estimates used so the
        # model has the right qualitative shape and order of magnitude.
        self.soh_fade_per_cycle = 0.0005     # ~0.05%-of-capacity lost per full recharge cycle
        self.soh_floor_fraction = 0.80       # capacity fade saturates at 80% (typical Li-ion EOL)

        # --- Propulsion efficiency ---
        # Ref: Energy consumption models for UAVs
        self.propulsion_efficiency = 0.75   # Typical combined motor+ESC+prop efficiency
        self.avionics_power   = 15.0     # Watts (Enterprise sensors, RTK, and compute board)
        self.battery_internal_efficiency = 0.95

        # --- Non-linear Li-ion SOC model (voltage sag + Peukert effect) ---
        # Ref (verified): TB30 is a 6S Li-ion pack, 5880 mAh, 26.1V max charge
        # voltage, 22.8V nominal (DJI TB30 datasheet / multiple retailer specs).
        self.batt_v_full    = 26.1   # V, fully-charged pack voltage (verified spec)
        self.batt_v_nominal = 22.8   # V, nominal/plateau voltage (verified spec)
        # NOT a published DJI spec -- typical 6S Li-ion low-voltage cutoff
        # (~3.3 V/cell), used only as the knee/floor for the sag curve below.
        self.batt_v_cutoff  = 19.8   # V, engineering estimate (typical 6S Li-ion cutoff)
        self.batt_soc_knee_pct = 30.0   # SOC% below which voltage sag steepens sharply
        # Ref: Peukert's law for Li-ion cells typically k=1.05-1.15; mid-range
        # estimate used here (not TB30-specific, DJI doesn't publish this).
        self.peukert_k = 1.10
        # 1C reference current: the "rated" discharge current used as the
        # Peukert-law baseline, derived from nameplate capacity (2x TB30 in
        # parallel => combined Ah at nominal voltage).
        self.batt_capacity_ah = self.battery_capacity_wh / self.batt_v_nominal
        self.batt_rated_current_a = self.batt_capacity_ah   # 1C reference (A)


OWT_CFG = OWTPhysicsConfig()


def owt_voltage_at_soc(soc_pct):
    """Piecewise-linear approximation of a Li-ion discharge voltage curve:
    a near-flat 'plateau' from 100% down to batt_soc_knee_pct, then a much
    steeper drop from the knee down to the cutoff voltage at 0%. This is
    the standard qualitative shape of a real Li-ion discharge curve (flat
    plateau, then a knee near empty) -- NOT a per-cell electrochemical
    simulation, but far closer to reality than assuming constant voltage."""
    soc_pct = max(0.0, min(100.0, soc_pct))
    knee = OWT_CFG.batt_soc_knee_pct
    if soc_pct >= knee:
        # Plateau: v_full at 100% -> v_nominal at the knee
        frac = (soc_pct - knee) / (100.0 - knee)
        return OWT_CFG.batt_v_nominal + frac * (OWT_CFG.batt_v_full - OWT_CFG.batt_v_nominal)
    else:
        # Knee -> cutoff: v_nominal at the knee -> v_cutoff at 0%
        frac = soc_pct / knee
        return OWT_CFG.batt_v_cutoff + frac * (OWT_CFG.batt_v_nominal - OWT_CFG.batt_v_cutoff)


def owt_peukert_derating(current_a):
    """Peukert-law capacity derating factor (>= 1.0): how much MORE
    effective capacity a given discharge current consumes, relative to
    the 1C rated baseline. factor=1.0 at/below the 1C rated current
    (Peukert effect negligible near the reference C-rate); grows for
    higher current, per C_p = I^k * t."""
    ratio = current_a / max(OWT_CFG.batt_rated_current_a, 1e-9)
    if ratio <= 1.0:
        return 1.0
    return ratio ** (OWT_CFG.peukert_k - 1.0)


def owt_soh_capacity_fraction(cycle_count):
    """State-of-Health: fraction (0-1) of usable capacity REMAINING after
    `cycle_count` full recharge cycles. Linear fade from 1.0 at cycle 0
    down to soh_floor_fraction, then flat (real packs are retired at/near
    that point rather than continuing to fade toward zero -- so the floor
    also acts as a stand-in "replace the battery" bound instead of letting
    aging silently make the mission impossible).

    cycle_count=0 -> returns exactly 1.0 (no aging), so this is a pure
    no-op unless a non-zero cycle count is deliberately passed in."""
    if cycle_count is None or cycle_count <= 0:
        return 1.0
    fraction = 1.0 - (cycle_count * OWT_CFG.soh_fade_per_cycle)
    return max(OWT_CFG.soh_floor_fraction, fraction)


def owt_usable_capacity_joules_with_soh(cycle_count):
    """Usable battery capacity in Joules AFTER applying SOH capacity fade
    for the given number of completed recharge cycles. cycle_count=0 (or
    None) reproduces OWT_CFG.usable_capacity_joules exactly -- the fresh-
    battery baseline used everywhere else in the file."""
    return OWT_CFG.usable_capacity_joules * owt_soh_capacity_fraction(cycle_count)


def owt_set_cell_size_from_bbox(rows, cols, lon_min, lat_min, lon_max, lat_max):
    """FIX (#1 — cell_size_m / real-SRTM scale mismatch):

    OWT_CFG.cell_size_m defaulted to a flat 100m regardless of what
    geographic area the grid actually represents. When USE_REAL_SRTM=True,
    a 30x30 grid over a 1deg x 1deg bounding box (e.g. lon 91.5-92.5,
    lat 24.5-25.5) actually spans ~3.3-3.7 km PER CELL, not 100m — a
    ~33-37x error that silently corrupts every distance/time/climb-slope
    calculation in wind_cost/climb_cost/owt_step_cost.

    This computes the real meters-per-cell from the bounding box using a
    standard equirectangular approximation (111.32 km/deg latitude,
    111.32*cos(mid_latitude) km/deg longitude — accurate to well under 1%
    over a 1deg box, which is more than enough for this grid-cell scale)
    and overwrites OWT_CFG.cell_size_m so every physics term downstream
    (wind_cost, climb_cost, owt_step_cost, owt_energy_to_travel) uses the
    correct real-world distance. Must be called AFTER OWT_CFG exists and
    BEFORE any energy calibration or simulation runs.
    """
    mid_lat = (lat_min + lat_max) / 2.0
    km_per_deg_lat = 111.32
    km_per_deg_lon = 111.32 * math.cos(math.radians(mid_lat))

    lat_span_km = (lat_max - lat_min) * km_per_deg_lat
    lon_span_km = (lon_max - lon_min) * km_per_deg_lon

    m_per_row = (lat_span_km * 1000.0) / rows
    m_per_col = (lon_span_km * 1000.0) / cols

    # Grid distance math elsewhere (np.hypot(dr,dc), Manhattan dist_cells)
    # assumes SQUARE cells with one scalar cell_size_m, so use the average
    # of the row/col spacing (they're close — a few % apart at these
    # latitudes/box sizes — this is a reasonable approximation, not exact).
    cell_size_m = (m_per_row + m_per_col) / 2.0
    OWT_CFG.cell_size_m = cell_size_m
    return cell_size_m


# ── 1. WIND COST — verified derivation ──────────────────────────────────
def owt_required_airspeed(cruise_speed, wind_vector, heading_hat, max_cap):
    """Vector-subtraction method (verified correct for a body that keeps
    its nose/camera pointed along the ground track — appropriate for a
    surveillance UAV that must keep its sensor forward-facing)."""
    v_ground_vec = cruise_speed * heading_hat
    v_air_vec = v_ground_vec - wind_vector
    air_speed = np.linalg.norm(v_air_vec)
    if air_speed > max_cap:
        return None
    return air_speed


def owt_wind_cost(r_old, c_old, r_new, c_new, wind_speed, wind_dir_deg, elevation_m=0.0):
    """Step energy (Joules) for the wind/drag component.
    Physically derived:
      - required airspeed via vector triangle
      - drag power ~ airspeed^3 (real drag equation P = 0.5*rho*v^3*Cd*A)
      - air density corrected for altitude via barometric formula
        (connects this term to elevation, same as climb cost)"""
    dr, dc = r_new - r_old, c_new - c_old
    distance = np.hypot(dr, dc) * OWT_CFG.cell_size_m
    if distance == 0.0:
        return 0.0

    heading_hat = np.array([dc, dr], dtype=float)
    heading_hat /= np.linalg.norm(heading_hat)

    wind_rad = np.radians(wind_dir_deg)
    wind_vector = wind_speed * np.array([np.cos(wind_rad), np.sin(wind_rad)])

    a = owt_required_airspeed(OWT_CFG.cruise_speed, wind_vector, heading_hat, OWT_CFG.max_airspeed_cap)
    if a is None:
        return float('inf')   # exceeds hardware airspeed capability -> impassable

    # Barometric air density at this cell's elevation
    rho_current = OWT_CFG.rho_standard * np.exp(-elevation_m / 8500.0)

    drag_power = 0.5 * rho_current * (a ** 3) * OWT_CFG.drag_coefficient * OWT_CFG.frontal_area

    time = distance / OWT_CFG.cruise_speed
    return drag_power * time


# ── 2. TURN COST — momentum theory, verified derivation ─────────────────
def owt_turn_cost(r_old, c_old, r_mid, c_mid, r_new, c_new):
    """Marginal energy (Joules) for a heading change, derived from:
      delta_v = 2*v*sin(theta/2)          [exact vector geometry]
      Force   = mass * delta_v / duration [Newton's 2nd law]
      Power   = Force^1.5 / sqrt(2*rho*A) [actuator disk momentum theory]"""
    if r_old is None or c_old is None:
        return 0.0

    h1 = np.arctan2(r_mid - r_old, c_mid - c_old)
    h2 = np.arctan2(r_new - r_mid, c_new - c_mid)
    delta = h1 - h2
    angle_diff = abs(np.arctan2(np.sin(delta), np.cos(delta)))

    if angle_diff < OWT_CFG.turn_threshold:
        return 0.0

    v_g = OWT_CFG.cruise_speed
    delta_v = 2 * v_g * np.sin(angle_diff / 2.0)

    mass = OWT_CFG.drone_mass + OWT_CFG.payload_mass
    required_force = mass * (delta_v / OWT_CFG.turn_duration_estimate)

    power_constant = 1.0 / np.sqrt(2 * OWT_CFG.rho_standard * OWT_CFG.rotor_disk_area)
    marginal_power = power_constant * (required_force ** 1.5)

    return marginal_power * OWT_CFG.turn_duration_estimate


# ── 3. CLIMB COST — gravitational PE + induced climb power ──────────────
def owt_climb_cost(elev_old_m, elev_new_m, distance_m):
    """Energy (Joules) to change altitude between two cells.
    Two components:
      - Gravitational potential energy: m*g*delta_h (exact, always correct)
      - Induced power penalty for climb RATE, via actuator disk theory,
        same rho/A constants used in turn_cost for consistency."""
    delta_h = elev_new_m - elev_old_m
    mass = OWT_CFG.drone_mass + OWT_CFG.payload_mass

    if delta_h <= 0:
        # Descending: potential energy is recovered, not spent.
        # Model a small recovery fraction (not 100% - real drones still
        # burn some power to control descent rate, not a free glide).
        return 0.3 * abs(mass * OWT_CFG.g * delta_h)

    # Climbing: full gravitational PE cost
    e_gravity = mass * OWT_CFG.g * delta_h

    # Extra induced power from needing a vertical climb-rate component,
    # using the same momentum-theory constant as turn_cost for consistency.
    time = max(distance_m / OWT_CFG.cruise_speed, 1e-6)
    climb_rate = delta_h / time
    power_constant = 1.0 / np.sqrt(2 * OWT_CFG.rho_standard * OWT_CFG.rotor_disk_area)
    induced_force = mass * climb_rate / max(OWT_CFG.turn_duration_estimate, 1e-6)
    e_induced = power_constant * (induced_force ** 1.5) * time

    return e_gravity + e_induced


def _owt_joules_to_battery_pct(joules, current_soc_pct=None, time_s=1.0, cycle_count=0):
    """Bridges real energy (Joules) to this file's 0-100 battery
    percentage scale. This is the critical unit conversion that was
    initially missing — without it, real Joule costs (hundreds per step)
    instantly zero a 0-100 percentage battery in a fraction of one step.

    FIX #3 (battery realism): previously this divided by the full
    NAMEPLATE capacity (battery_capacity_joules) as if 100% of it were
    usable and every Joule requested at the propeller/avionics arrived
    there loss-free from the cells. Two corrections apply:
      (a) joules / battery_internal_efficiency — extra energy the battery
          must actually supply to cover its own internal-resistance (I^2R)
          heating losses.
      (b) usable_capacity_joules (== nameplate * usable_capacity_fraction)
          instead of the full nameplate, since BMS/flight-controller
          cutoffs mean not all nameplate capacity is actually usable in
          the field.

    FIX #4 (non-linear Li-ion SOC model, OPT-IN): the plain version above
    is still linear in Joules -- fine near a full battery, but real Li-ion
    packs drain FASTER per Joule at high discharge current (Peukert effect)
    and their usable energy shrinks further as SOC drops and pack voltage
    sags (especially below ~30% SOC, see owt_voltage_at_soc()). Passing
    current_soc_pct (and optionally time_s, the real duration in seconds
    this energy was drawn over -- needed to recover average power/current)
    switches on this non-linear correction:
        1. average electrical power P = joules_from_battery / time_s
        2. pack voltage V = owt_voltage_at_soc(current_soc_pct)
        3. discharge current I = P / V
        4. Peukert derating factor >= 1.0 from owt_peukert_derating(I)
        5. effective joules = joules_from_battery * derating_factor,
           i.e. the SAME real energy costs MORE "effective capacity" at
           high current / low SOC than the linear model assumes.

    BACKWARD COMPATIBILITY: current_soc_pct=None (the default) skips all
    of the above and returns EXACTLY the old linear result -- every
    existing call site in this file that doesn't pass current_soc_pct is
    completely unaffected. This backward-compatibility was verified
    during development.

    FIX #5 (State-of-Health / capacity aging, OPT-IN): cycle_count (the
    number of full recharge cycles this pack has already been through --
    e.g. the drone's own recharge counter) fades the usable-capacity
    denominator via owt_soh_capacity_fraction(). A drone with a worn pack
    therefore loses MORE %-points for the exact same Joules than a fresh
    one. cycle_count=0 (the default) reproduces the fresh-battery
    denominator exactly, so this is a no-op unless the caller passes a
    real cycle count."""
    joules_from_battery = joules / OWT_CFG.battery_internal_efficiency
    usable_capacity_joules = owt_usable_capacity_joules_with_soh(cycle_count)

    if current_soc_pct is None:
        return (joules_from_battery / usable_capacity_joules) * 100.0

    power_w = joules_from_battery / max(time_s, 1e-6)
    voltage_v = owt_voltage_at_soc(current_soc_pct)
    current_a = power_w / voltage_v
    derating = owt_peukert_derating(current_a)
    effective_joules = joules_from_battery * derating

    return (effective_joules / usable_capacity_joules) * 100.0


def owt_step_cost(r_old, c_old, r_new, c_new, pr, pc,
                   elevation_grid, wind_speed, wind_dir_deg, current_soc_pct=None,
                   cycle_count=0):
    """Signature-compatible replacement for calculate_actual_step_cost().
    Same call pattern: (r_old, c_old, r_new, c_new, pr, pc)
    elevation_grid: dict {(r,c): elevation_in_metres}, populated from SRTM
                     (see OWT_load_real_srtm/OWT_generate_synthetic_terrain
                     below) or synthetic terrain.

    current_soc_pct: OPTIONAL. Pass the drone's CURRENT battery percentage
    (e.g. self.s_active["b"]) to switch on the non-linear voltage-sag +
    Peukert model in _owt_joules_to_battery_pct() instead of the plain
    linear one. Default None = unchanged old behavior (every existing
    caller in this file that doesn't pass this argument is unaffected).

    cycle_count: OPTIONAL. Pass the drone's completed-recharge count
    (e.g. self.sR/self.greR/self.gR/self.aR) to switch on the SOH
    capacity-fade model -- the SAME step now costs slightly MORE %-points
    on a pack that's been recharged many times. Default 0 = fresh-battery
    behavior, unchanged from before this parameter existed."""
    if (r_old, c_old) == (r_new, c_new):
        # Hover in place
        elev = elevation_grid.get((r_old, c_old), 0.0)
        rho_current = OWT_CFG.rho_standard * np.exp(-elev / 8500.0)
        mass = OWT_CFG.drone_mass + OWT_CFG.payload_mass
        thrust = mass * OWT_CFG.g
        power_constant = 1.0 / np.sqrt(2 * rho_current * OWT_CFG.rotor_disk_area)
        hover_power = power_constant * (thrust ** 1.5)
        # FIX #2: hover_power is MECHANICAL (actuator-disk) power; divide by
        # propulsion_efficiency to get real ELECTRICAL power drawn from the
        # battery. avionics_power is already electrical — not divided.
        e_hover_mech = hover_power * 1.0 / OWT_CFG.propulsion_efficiency   # 1 second hover step
        joules = e_hover_mech + OWT_CFG.avionics_power
        return round(_owt_joules_to_battery_pct(joules, current_soc_pct=current_soc_pct, time_s=1.0, cycle_count=cycle_count), 3)

    elev_old = elevation_grid.get((r_old, c_old), 0.0)
    elev_new = elevation_grid.get((r_new, c_new), 0.0)

    e_wind = owt_wind_cost(r_old, c_old, r_new, c_new, wind_speed, wind_dir_deg, elev_new)
    if e_wind == float('inf'):
        return float('inf')

    e_turn = owt_turn_cost(pr, pc, r_old, c_old, r_new, c_new) if pr is not None else 0.0

    distance_m = np.hypot(r_new - r_old, c_new - c_old) * OWT_CFG.cell_size_m
    e_climb = owt_climb_cost(elev_old, elev_new, distance_m)

    # BUGFIX: avionics_power is a POWER (Watts), not an energy (Joules).
    # It must be multiplied by the step's flight time to get Joules —
    # exactly like owt_energy_to_travel() already does below. Adding it
    # directly (old code: `+ OWT_CFG.avionics_power`) silently mixed units
    # and made this function's cost inconsistent with owt_energy_to_travel(),
    # which is used by must_recharge_now()/two_hop_check() for safety
    # margins — those two MUST agree on the same physics.
    time = distance_m / OWT_CFG.cruise_speed
    e_avionics = OWT_CFG.avionics_power * time

    # FIX #2: e_wind/e_turn/e_climb are all MECHANICAL energy (aerodynamic
    # drag power, actuator-disk turn/climb power, gravitational PE) — divide
    # by propulsion_efficiency to get the real ELECTRICAL energy drawn from
    # the battery. e_avionics is already electrical (sensors/compute), so it
    # is added afterward, undivided.
    e_mechanical = (e_wind + e_turn + e_climb) / OWT_CFG.propulsion_efficiency

    total_joules = e_mechanical + e_avionics
    return round(_owt_joules_to_battery_pct(total_joules, current_soc_pct=current_soc_pct, time_s=time, cycle_count=cycle_count), 3)


def owt_energy_to_travel(r1, c1, r2, c2, elevation_grid, wind_speed, wind_dir_deg):
    """Signature-compatible replacement for energy_to_travel().
    Used by E_ret / two_hop_check / must_recharge_now for return-to-base
    safety calculations — MUST use the same physics as owt_step_cost,
    otherwise safety margins are computed against the wrong energy model.
    Uses Manhattan-decomposed straight-line estimate (fast safety-check
    estimate, not the actual planned route)."""
    dr, dc = r2 - r1, c2 - c1
    dist_cells = abs(dr) + abs(dc)
    if dist_cells == 0:
        return 0.0

    elev1 = elevation_grid.get((r1, c1), 0.0)
    elev2 = elevation_grid.get((r2, c2), 0.0)

    # Estimate heading toward target for wind calc
    heading_hat = np.array([dc, dr], dtype=float)
    norm = np.linalg.norm(heading_hat)
    if norm > 0:
        heading_hat /= norm

    wind_rad = np.radians(wind_dir_deg)
    wind_vector = wind_speed * np.array([np.cos(wind_rad), np.sin(wind_rad)])

    a = owt_required_airspeed(OWT_CFG.cruise_speed, wind_vector, heading_hat, OWT_CFG.max_airspeed_cap)
    if a is None:
        return float('inf')

    rho_current = OWT_CFG.rho_standard * np.exp(-elev2 / 8500.0)
    drag_power = 0.5 * rho_current * (a ** 3) * OWT_CFG.drag_coefficient * OWT_CFG.frontal_area

    distance_m = dist_cells * OWT_CFG.cell_size_m
    time = distance_m / OWT_CFG.cruise_speed

    e_wind_total = drag_power * time
    e_climb_total = owt_climb_cost(elev1, elev2, distance_m)
    e_avionics_total = OWT_CFG.avionics_power * time

    # FIX #2: same propulsion_efficiency correction as owt_step_cost — MUST
    # stay consistent between the two functions (see docstring above), or
    # the RTB safety-margin estimate (this function) and the real per-step
    # drain (owt_step_cost) disagree again, exactly the bug fixed earlier.
    e_mechanical_total = (e_wind_total + e_climb_total) / OWT_CFG.propulsion_efficiency

    total_joules = e_mechanical_total + e_avionics_total
    return round(_owt_joules_to_battery_pct(total_joules), 2)


# ══════════════════════════════════════════════════════════════
#  SRTM ELEVATION LOADER  (inlined from srtm_loader.py)
# ══════════════════════════════════════════════════════════════
#  Loads real NASA SRTM elevation data and maps it onto this file's
#  ROWSxCOLS (row, col) grid as a per-cell elevation attribute.
#
#  WINDOWS NOTE: downloads SRTM .hgt tiles DIRECTLY over HTTPS and parses
#  them with pure numpy — no make, no GDAL CLI, no external dependencies
#  beyond `requests` and `numpy`/`scipy`. Works natively on Windows, Mac,
#  and Linux.  pip install requests numpy scipy
# ══════════════════════════════════════════════════════════════

OWT_SRTM1_SIZE = 3601   # samples per side for SRTM1 (30m resolution) .hgt tiles

# FIX (cache-path fragility -> reproducibility bug): this used to default to
# the *relative* path '.srtm_cache', which points at a different folder
# depending on the current working directory the script happens to be
# launched from (terminal vs IDE vs double-click). If a previous run's
# cache lived in one cwd's '.srtm_cache' but this run's cwd is different,
# the tile looks "missing" even though it was already downloaded once --
# triggering a fresh network call, and a SILENT fallback to synthetic
# terrain if that call fails (see get_elevation_grid()). Anchoring the
# cache to the script's own directory makes "already cached" mean the same
# thing every time, from every launch location -- required for the real
# vs synthetic terrain choice to stop depending on which folder you were
# sitting in when you ran python.
OWT_SRTM_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.srtm_cache')


def _owt_srtm_tile_name(lat_deg, lon_deg):
    ns = 'N' if lat_deg >= 0 else 'S'
    ew = 'E' if lon_deg >= 0 else 'W'
    return ns, ew, abs(int(lat_deg)), abs(int(lon_deg))


def _owt_download_hgt_tile(lat_deg, lon_deg, cache_dir=OWT_SRTM_CACHE_DIR):
    """Downloads and decompresses a single 1-degree SRTM .hgt tile directly
    over HTTPS. Caches to disk so repeated runs don't re-download."""
    import requests

    os.makedirs(cache_dir, exist_ok=True)
    ns, ew, lat_i, lon_i = _owt_srtm_tile_name(lat_deg, lon_deg)
    tile_id = f"{ns}{lat_i:02d}{ew}{lon_i:03d}"
    cache_path = os.path.join(cache_dir, f"{tile_id}.hgt")

    if os.path.exists(cache_path):
        return cache_path

    url = f"https://s3.amazonaws.com/elevation-tiles-prod/skadi/{ns}{lat_i:02d}/{tile_id}.hgt.gz"

    print(f"  Downloading tile {tile_id} ...")
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    raw = gzip.decompress(resp.content)
    with open(cache_path, 'wb') as f:
        f.write(raw)
    return cache_path


def _owt_read_hgt(path, size=OWT_SRTM1_SIZE):
    """Parses a raw .hgt file: a square grid of 16-bit signed big-endian
    integers, row-major from the north-west corner. No GDAL needed —
    this is just numpy reading a known binary layout."""
    data = np.fromfile(path, dtype='>i2')
    expected = size * size
    if data.size != expected:
        alt_size = int(round(np.sqrt(data.size)))
        if alt_size * alt_size == data.size:
            size = alt_size
        else:
            raise ValueError(f"Unexpected .hgt file size: {data.size} samples")
    return data.reshape(size, size).astype(float)


def owt_load_real_srtm(rows, cols, lon_min, lat_min, lon_max, lat_max,
                        cache_dir=OWT_SRTM_CACHE_DIR):
    """Downloads real NASA SRTM elevation data for a bounding box and resizes
    it to match this file's grid dimensions. Pure Python/numpy — no make,
    no GDAL CLI required. Works on native Windows.
    pip install requests numpy scipy

    Example regions for a Bangladesh border thesis:
        Sylhet hills (India border):   lon 91.5-92.5, lat 24.5-25.5
        Cox's Bazar (Myanmar border):  lon 92.0-92.5, lat 20.8-21.5
        Chittagong Hill Tracts:        lon 92.0-92.6, lat 22.0-23.0
        Benapole (Jessore, flat delta): lon 88.82-89.07, lat 22.97-23.11

    Returns: dict {(r,c): elevation_metres}"""
    from scipy.ndimage import zoom

    lat_tiles = range(int(np.floor(lat_min)), int(np.ceil(lat_max)))
    lon_tiles = range(int(np.floor(lon_min)), int(np.ceil(lon_max)))

    print(f"Downloading SRTM tiles for bounds ({lon_min},{lat_min})-({lon_max},{lat_max})...")

    tile_grids = {}
    for lat_deg in lat_tiles:
        for lon_deg in lon_tiles:
            path = _owt_download_hgt_tile(lat_deg, lon_deg, cache_dir)
            tile_grids[(lat_deg, lon_deg)] = _owt_read_hgt(path)

    lat_list = sorted(lat_tiles, reverse=True)
    lon_list = sorted(lon_tiles)

    row_blocks = []
    for lat_deg in lat_list:
        col_blocks = [tile_grids[(lat_deg, lon_deg)] for lon_deg in lon_list]
        row_blocks.append(np.hstack(col_blocks))
    mosaic = np.vstack(row_blocks)

    mosaic[mosaic < -1000] = 0   # SRTM voids are typically -32768; clean up

    # BUGFIX (cropping): downloaded tiles are whole 1deg x 1deg blocks
    # (SRTM granularity), which almost never line up exactly with the
    # requested lon_min/lat_min/lon_max/lat_max box — e.g. a 15km Benapole
    # box (~0.14deg) sits INSIDE a much bigger 1deg (or 2deg, if it
    # straddles a tile boundary) mosaic. The old code resized the ENTIRE
    # downloaded mosaic straight to rows x cols, silently stretching a much
    # larger area than requested into the grid — inconsistent with
    # owt_set_cell_size_from_bbox(), which computes cell_size_m from the
    # actual requested box. Now the mosaic is cropped to the exact
    # requested lon/lat box (in pixel space, 3600 samples/degree for
    # SRTM1) BEFORE resizing, so the returned grid really is the box asked
    # for, matching the cell_size_m calculation exactly.
    px_per_deg = OWT_SRTM1_SIZE - 1   # 3600 samples per degree (3601 incl. shared edge)
    top_lat  = max(lat_list) + 1      # geographic latitude of mosaic's top (row 0) edge
    left_lon = min(lon_list)          # geographic longitude of mosaic's left (col 0) edge

    row_start = int(round((top_lat - lat_max) * px_per_deg))
    row_end   = int(round((top_lat - lat_min) * px_per_deg))
    col_start = int(round((lon_min - left_lon) * px_per_deg))
    col_end   = int(round((lon_max - left_lon) * px_per_deg))

    row_start = max(0, min(row_start, mosaic.shape[0] - 1))
    row_end   = max(row_start + 1, min(row_end, mosaic.shape[0]))
    col_start = max(0, min(col_start, mosaic.shape[1] - 1))
    col_end   = max(col_start + 1, min(col_end, mosaic.shape[1]))

    mosaic = mosaic[row_start:row_end, col_start:col_end]

    resized = zoom(mosaic, (rows / mosaic.shape[0], cols / mosaic.shape[1]))
    resized = np.clip(resized, 0, None)

    elevation_grid = {}
    for r in range(rows):
        for c in range(cols):
            elevation_grid[(r, c)] = float(resized[r, c])

    print(f"Loaded real SRTM terrain: max={resized.max():.0f}m, "
          f"min={resized.min():.0f}m, mean={resized.mean():.0f}m")
    return elevation_grid


def owt_generate_synthetic_terrain(rows, cols, hills=None, seed=42):
    """Generates realistic-looking Gaussian-hill terrain with the SAME
    interface as owt_load_real_srtm() — a drop-in stand-in for testing
    and development before you run the real SRTM loader."""
    rng = np.random.RandomState(seed)

    if hills is None:
        hills = [
            (8,  22, 180, 5),
            (18, 12, 120, 4),
            (25, 20,  90, 3),
            (5,   5,  40, 3),
        ]

    terrain = np.full((rows, cols), 15.0)

    for cr, cc, height, spread in hills:
        for r in range(rows):
            for c in range(cols):
                dist_sq = (r - cr) ** 2 + (c - cc) ** 2
                terrain[r, c] += height * np.exp(-dist_sq / (2 * spread ** 2))

    terrain += rng.normal(0, 2.0, size=(rows, cols))
    terrain = np.clip(terrain, 0, None)

    elevation_grid = {}
    for r in range(rows):
        for c in range(cols):
            elevation_grid[(r, c)] = float(terrain[r, c])

    print(f"Synthetic terrain generated: max={terrain.max():.0f}m, "
          f"min={terrain.min():.0f}m, mean={terrain.mean():.0f}m")
    return elevation_grid


# Audit trail for reproducibility: which terrain a given process actually
# ended up using. Set by get_elevation_grid() below; read by
# so every report states
# up front whether it was generated on real or synthetic terrain -- so two
# result sets that look like "the same experiment" but actually used
# different terrain are never silently mistaken for each other again.
TERRAIN_SOURCE = None   # "real_srtm" | "real_srtm_export" | "synthetic_fallback" | "synthetic_forced"

# ══════════════════════════════════════════════════════════════
# PORTABLE ELEVATION EXPORT/IMPORT — fixes the terrain-source mismatch
# permanently for network-isolated environments (e.g. Claude's sandbox)
# ══════════════════════════════════════════════════════════════
# prime_srtm_cache() solves this for YOUR machine (disk-cached .hgt tiles,
# no repeat downloads). It does NOT help an environment that can never
# reach s3.amazonaws.com at all (host blocked at the network layer, not
# just slow/flaky) — no amount of local caching helps if the tile was
# never downloadable there in the first place.
#
# The fix: export the FINAL, already-cropped-and-resized 30x30
# ELEVATION_GRID (a few KB of JSON) instead of the raw ~26MB .hgt tile.
# Run this ONCE, after a successful real_srtm run, on a machine with
# working internet:
#     python3 -c "import a10; a10.export_elevation_grid()"
# This writes elevation_grid_export.json next to this script. Copy/upload
# that file anywhere else you want this script to run (including a
# network-isolated sandbox) — get_elevation_grid() below checks for it
# FIRST, before attempting any network call, and if present, loads the
# EXACT SAME real elevation values with zero network dependency. This
# guarantees both sides of any future comparison use identical real
# terrain, permanently — no more alpha-identical-but-numerically-different
# runs caused by one side quietly landing on synthetic fallback.
OWT_ELEVATION_EXPORT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'elevation_grid_export.json')

def export_elevation_grid(grid=None, path=OWT_ELEVATION_EXPORT_PATH):
    """Save a computed real ELEVATION_GRID to a small portable JSON file.
    Defaults to exporting the module's own ELEVATION_GRID (only makes
    sense to call this if TERRAIN_SOURCE == "real_srtm" at the time)."""
    if grid is None:
        grid = globals().get("ELEVATION_GRID")
    if grid is None:
        print("No ELEVATION_GRID available to export.")
        return
    if TERRAIN_SOURCE not in ("real_srtm", "real_srtm_export"):
        print(f"WARNING: TERRAIN_SOURCE is '{TERRAIN_SOURCE}', not 'real_srtm' — "
              f"exporting this would just spread the synthetic-fallback problem "
              f"instead of fixing it. Re-run with a working internet connection first.")
        return
    payload = {f"{r},{c}": v for (r, c), v in grid.items()}
    with open(path, 'w') as f:
        json.dump(payload, f)
    size_kb = os.path.getsize(path) / 1024
    print(f"Exported {len(payload)}-cell real ELEVATION_GRID to {path} "
          f"({size_kb:.1f} KB). Upload/copy this file anywhere you want this "
          f"script to see IDENTICAL real terrain with no network call needed.")

def _load_elevation_grid_export(path=OWT_ELEVATION_EXPORT_PATH):
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            payload = json.load(f)
        grid = {}
        for key, v in payload.items():
            r, c = key.split(',')
            grid[(int(r), int(c))] = float(v)
        return grid
    except Exception as e:
        print(f"WARNING: found {path} but couldn't parse it ({e}) — ignoring, "
              f"falling back to the normal real_srtm/synthetic path.")
        return None

def get_elevation_grid(rows, cols, use_real_srtm=False,
                        lon_min=91.5, lat_min=24.5, lon_max=92.5, lat_max=25.5,
                        fallback_hills=None):
    """Single entry point. Set use_real_srtm=True to download real terrain
    (needs internet + `pip install requests scipy`) -- or, once
    prime_srtm_cache() has been run successfully, to load it from the
    local on-disk cache with NO network call at all. Falls back to
    synthetic terrain automatically if real terrain can't be obtained for
    any reason — `fallback_hills` lets the caller pick a fallback profile
    that actually resembles the target region (e.g. flat/None for a delta
    area like Benapole) instead of always defaulting to the generic hilly
    profile, which would be misleading for a real flat-terrain sector.

    IMPORTANT for reproducibility: a fallback here silently swaps every
    distance/elevation-dependent number in the whole simulation for a
    different (synthetic) dataset. Two runs that both "worked" but one hit
    the network and one didn't are NOT the same experiment, even with
    identical alpha/seeds -- see TERRAIN_SOURCE and prime_srtm_cache()."""
    global TERRAIN_SOURCE
    if use_real_srtm:
        exported = _load_elevation_grid_export()
        if exported is not None and len(exported) == rows * cols:
            TERRAIN_SOURCE = "real_srtm_export"
            print(f"Loaded ELEVATION_GRID from portable export ({OWT_ELEVATION_EXPORT_PATH}) "
                  f"-- identical real terrain to whichever machine created it, no network call.")
            return exported
        try:
            grid = owt_load_real_srtm(rows, cols, lon_min, lat_min, lon_max, lat_max)
            TERRAIN_SOURCE = "real_srtm"
            return grid
        except Exception as e:
            TERRAIN_SOURCE = "synthetic_fallback"
            print("!"*70)
            print("WARNING: real SRTM terrain unavailable (no cached tile + download")
            print(f"  failed: {e})")
            print("  -> FALLING BACK TO SYNTHETIC TERRAIN. Results from this run are")
            print("     NOT comparable to a run that used real SRTM terrain, even with")
            print("     identical alpha/seeds -- the whole travel/elevation cost model")
            print("     changes underneath the experiment.")
            print("  -> Run prime_srtm_cache() once with a working internet connection")
            print("     to permanently fix this (downloads once, cached forever after).")
            print("!"*70)
            return owt_generate_synthetic_terrain(rows, cols, hills=fallback_hills)
    else:
        TERRAIN_SOURCE = "synthetic_forced"
        return owt_generate_synthetic_terrain(rows, cols)

def prime_srtm_cache():
    """Run this ONCE, by itself, with a working internet connection:
        python3 -c "import a7; a7.prime_srtm_cache()"
    Downloads and permanently caches the SRTM .hgt tile(s) this project's
    ELEVATION_GRID needs (the Benapole bounding box — SRTM_LON_MIN/MAX,
    SRTM_LAT_MIN/MAX below) into OWT_SRTM_CACHE_DIR, which is an absolute
    path next to this script (NOT dependent on your current working
    directory — see the note on OWT_SRTM_CACHE_DIR above).

    After this succeeds, every future run/import of this script reads the
    tile straight from that local cache (see _owt_download_hgt_tile's
    early-return when the cache file already exists) — no network call,
    no dependence on whether S3 happens to answer that day, and therefore
    identical real-terrain results on every run from here on.

    Safe to re-run any time; already-cached tiles are skipped instantly."""
    print(f"Cache directory: {OWT_SRTM_CACHE_DIR}")
    lat_tiles = range(int(np.floor(SRTM_LAT_MIN)), int(np.ceil(SRTM_LAT_MAX)))
    lon_tiles = range(int(np.floor(SRTM_LON_MIN)), int(np.ceil(SRTM_LON_MAX)))
    all_ok = True
    for lat_deg in lat_tiles:
        for lon_deg in lon_tiles:
            try:
                path = _owt_download_hgt_tile(lat_deg, lon_deg, OWT_SRTM_CACHE_DIR)
                print(f"  OK: {path}")
            except Exception as e:
                all_ok = False
                print(f"  FAILED: tile at lat={lat_deg}, lon={lon_deg} -- {e}")
    if all_ok:
        print("SUCCESS: all required SRTM tile(s) are now cached in "
              f"{OWT_SRTM_CACHE_DIR}. Re-run/re-import this script -- ELEVATION_GRID "
              "will load from the cache from now on, every time, with no network call.")
    else:
        print("INCOMPLETE: at least one tile failed to download. Re-run "
              "prime_srtm_cache() with a working internet connection before trusting "
              "any 'real terrain' result -- until every tile above says OK, imports "
              "will keep silently falling back to synthetic terrain.")
    return all_ok

# ══════════════════════════════════════════════════════════════
#  END OF INLINED OWT-A* PHYSICS + SRTM MODULES
# ══════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════
# CONFIGURATION — 30×30 Grid
# ══════════════════════════════════════════════════════════════
ROWS, COLS   = 30, 30
STATIONS     = [(0,0),(0,29),(29,0),(29,29)]
ZONE_SIZE    = 5          
MAX_STEPS    = 2000       
NUM_THREATS  = 20

# ── Sensor / detection radius (Feature #1) ──────────────────────
DETECTION_RADIUS = 1
BATCH_SEEDS  = list(range(42, 72))   # 30 seeds (42-71) — আগের ১০টা (42-51) subset হিসেবে থেকে গেছে,
                                       # তুলনাযোগ্যতা বজায় রেখে statistical power বাড়ানো হলো (n=10 → n=30)

# ══════════════════════════════════════════════════════════════
# REALISTIC UAV ENERGY MODEL PARAMETERS
# ══════════════════════════════════════════════════════════════
# BUGFIX: these used to be hardcoded to old generic-drone values (2.0kg /
# 0.12kg / 10.0 m/s) independently of OWTPhysicsConfig, so when OWT_CFG
# was anchored to the real DJI Matrice 30 datasheet (drone_mass=3.77kg,
# payload_mass=0.20kg, cruise_speed=15 m/s), these three silently stayed
# on the old numbers — used by needs_handoff_now()'s early_margin, so the
# handoff safety threshold was quietly computed against the wrong drone.
# Now derived directly from OWT_CFG so there is exactly ONE source of
# truth for the vehicle spec; they can never drift out of sync again.
UAV_MASS       = OWT_CFG.drone_mass     # kg (ড্রোনের নিজস্ব ওজন — datasheet-anchored)
PAYLOAD_MASS   = OWT_CFG.payload_mass   # kg (payload — datasheet-anchored)
UAV_SPEED      = OWT_CFG.cruise_speed   # m/s (ক্রুজ স্পিড — datasheet-anchored)

WIND_SPEED     = 4.0    # m/s (বাতাসের গতিবেগ) -- BASE/mean value, unchanged meaning.
WIND_DIR       = 45.0   # ডিগ্রিতে (বাতাস উত্তর-পূর্ব থেকে আসছে) -- BASE/mean value, unchanged meaning.

# ══════════════════════════════════════════════════════════════
# DYNAMIC WIND MODEL (time-varying wind speed/direction)
# ══════════════════════════════════════════════════════════════
# WHY: WIND_SPEED/WIND_DIR above were previously hardcoded constants used
# for the ENTIRE simulation -- physically unrealistic (real wind gusts and
# shifts direction over time). This section adds an OPTIONAL time-varying
# wind field, driven purely by the simulation's own step counter (NOT by
# per-seed randomness), so that:
#   (a) it is fully deterministic and reproducible -- same step -> same
#       wind, every run, every seed;
#   (b) every relay-selection policy (A/B/C/D, baseline/optimized) that is
#       compared on the SAME seed still experiences the EXACT SAME wind
#       trajectory at the same step, so paired significance tests (the
#       Wilcoxon comparisons already used everywhere in this codebase)
#       remain valid -- only the DECISION policy differs, not the weather;
#   (c) it is OFF by default (WIND_DYNAMIC_ENABLED = False), so every
#       existing experiment, CSV, and figure you have already produced
#       remains EXACTLY reproducible without any code change on your
#       part. Only runs that explicitly opt in (see bottom of this file /
#       generate_relay_figures.py) use the new dynamic-wind physics --
#       old and new results are therefore never silently mixed.
#
# MODEL: base wind ± a slow sinusoidal "gust cycle" in speed, plus a slow
# linear directional drift with a secondary faster wobble (captures both
# a gradual weather-front-style direction change and shorter-period
# gustiness) -- a standard lightweight synthetic-wind approach used when a
# real historical wind time series isn't available.
WIND_DYNAMIC_ENABLED     = False   # ← the ONLY switch you need to flip on/off

WIND_BASE_SPEED          = WIND_SPEED   # m/s, center of the oscillation
WIND_SPEED_AMPLITUDE     = 2.0          # m/s, +/- swing around the base
WIND_SPEED_PERIOD        = 400          # steps for one full gust cycle

WIND_BASE_DIR            = WIND_DIR     # degrees, center of the drift
WIND_DIR_DRIFT_RATE      = 0.15         # deg/step, slow one-directional drift
WIND_DIR_NOISE_AMPLITUDE = 20.0         # deg, secondary oscillation (gustiness)
WIND_DIR_NOISE_PERIOD    = 137          # steps (deliberately not a clean divisor
                                         # of WIND_SPEED_PERIOD, so the two cycles
                                         # don't resonate into a repeating pattern)

# Simulation clock: a tiny piece of module-level state that the outer
# step-loop updates ONCE per step (see advance_sim_clock() call sites in
# DroneSimHeadless.do_step(), DroneSimGUI.tick(), and
# a2e_relay_fixed_final_51.step_fleet()). Every wind-dependent function
# below just reads the current step from here -- this avoids having to
# thread a new `step` parameter through the ~40 existing call sites of
# calculate_actual_step_cost()/energy_to_travel() across this file.
_SIM_CLOCK = {"step": 0}

def advance_sim_clock(step):
    """Call once per simulation step, BEFORE any cost/energy function runs
    for that step. No-op cost if you forget -- wind just stays at whatever
    step it was last set to -- but call it every step for correctness."""
    _SIM_CLOCK["step"] = step

def get_current_wind(step=None):
    """Returns (wind_speed_m_s, wind_dir_deg) for the given step (defaults
    to the last value set via advance_sim_clock()). When
    WIND_DYNAMIC_ENABLED is False (the default), always returns the
    original static (WIND_SPEED, WIND_DIR) -- byte-identical to the old
    behavior."""
    if not WIND_DYNAMIC_ENABLED:
        return WIND_SPEED, WIND_DIR

    if step is None:
        step = _SIM_CLOCK["step"]

    speed = WIND_BASE_SPEED + WIND_SPEED_AMPLITUDE * math.sin(2 * math.pi * step / WIND_SPEED_PERIOD)
    speed = max(0.0, speed)   # wind speed can't go negative

    direction = (
        WIND_BASE_DIR
        + WIND_DIR_DRIFT_RATE * step
        + WIND_DIR_NOISE_AMPLITUDE * math.sin(2 * math.pi * step / WIND_DIR_NOISE_PERIOD)
    ) % 360.0

    return speed, direction

# ── Elevation grid (SRTM real terrain, with synthetic fallback) ─────────
# use_real_srtm=True downloads real NASA SRTM tiles for the given lon/lat
# bounding box (needs `pip install requests scipy` + internet). Set to
# False to use the synthetic Gaussian-hill terrain, which needs no
# network access and is good enough for algorithm development.
#
# Benapole (Jessore District) is Bangladesh's busiest land port on the
# India border — real coordinates 23.042 deg N, 88.896 deg E (Wikipedia /
# Bangladesh Railway). Box below is a ~15km x 15km sector centered on it,
# matching BORDER_SECTOR_SIDE_KM further down. NOTE: Benapole sits on the
# flat Ganges-Brahmaputra delta — real elevation here is only ~3-11m
# above sea level (source: Benapole railway station elevation record),
# NOT hilly like the Sylhet/Chittagong examples below. Expect climb_cost
# to be almost negligible for this specific sector — that's realistic,
# not a bug.
USE_REAL_SRTM  = True
SRTM_LON_MIN, SRTM_LAT_MIN = 88.822, 22.975   # Benapole, Jessore (India border)
SRTM_LON_MAX, SRTM_LAT_MAX = 88.969, 23.109
# Other candidate sectors (uncomment to use instead):
#   Sylhet hills (India border):    88.822 -> 91.5, 92.5 / 24.5, 25.5
#   Cox's Bazar (Myanmar border):   92.0, 92.5 / 20.8, 21.5
#   Chittagong Hill Tracts:         92.0, 92.6 / 22.0, 23.0
ELEVATION_GRID = get_elevation_grid(
    ROWS, COLS, use_real_srtm=USE_REAL_SRTM,
    lon_min=SRTM_LON_MIN, lat_min=SRTM_LAT_MIN,
    lon_max=SRTM_LON_MAX, lat_max=SRTM_LAT_MAX,
    # If the real SRTM download can't complete (no internet / blocked host),
    # fall back to a FLAT profile (hills=[]) instead of the generic hilly
    # default — Benapole's real recorded elevation is only ~11m (Benapole
    # railway station), so a flat ~15m+-noise fallback is honestly close to
    # the real terrain, unlike the old default's fictitious 190m hills.
    fallback_hills=[]
)

# ══════════════════════════════════════════════════════════════
# TERRAIN-COST TOGGLE (flat-vs-terrain-aware SSE ablation)
# ══════════════════════════════════════════════════════════════
# REAL toggle -- calculate_actual_step_cost()/energy_to_travel() below
# actually read this. (Earlier ablation-script versions set fictitious
# module attributes named TERRAIN_AWARE / SSE_TERRAIN_AWARE that nothing
# in this file ever read, so "flat" and "terrain" runs silently executed
# identical code against the same ELEVATION_GRID -- every seed produced
# byte-identical flat/terrain results. This flag is the actual switch.)
TERRAIN_COST_ENABLED = True

# .get((r, c), 0.0) on an empty dict always returns 0.0 -- i.e. every
# cell reads as sea-level, which is exactly "flat terrain" for
# owt_step_cost()/owt_energy_to_travel()'s elevation-difference climb-cost
# term. Reusing the exact same physics functions with this grid instead
# of ELEVATION_GRID is what makes the flat/terrain comparison apples-to-
# apples (same wind/turn model, only the climb term changes).
_FLAT_ELEVATION_GRID = {}

# FIX #1 (cell_size_m / real-SRTM scale mismatch): if real SRTM terrain is
# active, the grid's true meters-per-cell is derived from the actual
# bounding box (≈3.3-3.7 km/cell for the box above on a 30x30 grid) —
# NOT the synthetic-terrain placeholder of 100m. Must run before any
# calibration/simulation so every distance-dependent physics term
# (wind_cost, climb_cost, owt_step_cost, owt_energy_to_travel) is correct.
if USE_REAL_SRTM:
    _cell_m = owt_set_cell_size_from_bbox(
        ROWS, COLS, SRTM_LON_MIN, SRTM_LAT_MIN, SRTM_LON_MAX, SRTM_LAT_MAX
    )
    print(f"[cell_size_m fix] real-SRTM grid -> {_cell_m:.1f} m/cell "
          f"(was hardcoded 100.0 m before the fix)")
# else: synthetic terrain keeps the OWTPhysicsConfig default (100.0 m) —
# there's no real bounding box to derive a true scale from.

# ══════════════════════════════════════════════════════════════
# FIX (synthetic-terrain scale realism): with the drone now anchored to a
# real DJI Matrice 30 datasheet, its per-charge range is a REAL, FIXED
# physical quantity (~29 km at this cruise speed/mass/battery — verified
# empirically: (100/ENERGY_PER_CELL)*cell_size_m). At the old placeholder
# cell_size_m=100m, the 30x30 grid represents only a 3km x 3km patch —
# so of course a drone that can fly 29km on one charge covers it in ~2
# recharges. That's not an energy-model bug; the GRID was simply far
# smaller than the vehicle it now represents.
#
# The correct fix is NOT to inflate wind/motor constants to force more
# recharges — that would silently break the DJI-datasheet anchoring just
# fixed (wind_speed=4 m/s and the M30's real drag/motor specs are already
# realistic; changing them to hit a target recharge count would be
# reverse-engineering the physics, not modeling it).
#
# Instead, cell_size_m is set here so the 30x30 grid represents a
# realistically-sized single-fleet border patrol sector — 15km x 15km is
# a reasonable assignment for a 4-station relay fleet (adjust
# BORDER_SECTOR_SIDE_KM to whatever sector size your thesis scenario
# actually claims). This makes recharge count scale with real coverage
# distance instead of an arbitrary small grid, while leaving every
# datasheet-anchored physics constant untouched.
BORDER_SECTOR_SIDE_KM = 15.0   # <-- tune this to your thesis's claimed patrol-sector size
if not USE_REAL_SRTM:
    OWT_CFG.cell_size_m = (BORDER_SECTOR_SIDE_KM * 1000.0) / ROWS
    print(f"[cell_size_m realism fix] synthetic grid -> {OWT_CFG.cell_size_m:.1f} m/cell "
          f"({BORDER_SECTOR_SIDE_KM:.0f}km x {BORDER_SECTOR_SIDE_KM:.0f}km sector, "
          f"was hardcoded 100.0 m before the fix)")

# ══════════════════════════════════════════════════════════════
# ENERGY THRESHOLD CALIBRATION  (FIXED — was hardcoded to the OLD
# heuristic model's scale, ~1.0%/step. The physics-based owt_step_cost()
# runs at a DIFFERENT scale — measured empirically here to be roughly
# 0.35-0.40%/step on this terrain/wind config, ~3x smaller than before.
# ENERGY_PER_CELL is auto-calibrated below by sampling owt_step_cost()
# directly on the actual ELEVATION_GRID/WIND_SPEED/WIND_DIR in use, so
# must_recharge_now()/two_hop_check()/needs_handoff_now() thresholds
# always track whatever terrain/wind config is active — instead of a
# stale magic number left over from the old (now-removed) heuristic
# model (E_AVIONICS/E_HOVER/E_TURN, which are no longer used anywhere).
# ══════════════════════════════════════════════════════════════
def _calibrate_energy_per_cell(sample_steps=400, seed=7):
    """Empirically measures the mean per-step battery drain (%) of
    owt_step_cost() on the CURRENT elevation grid / wind config, via a
    short deterministic random walk. Used as the 'cost of ~1 cell of
    travel' safety-margin unit in the RTB/handoff threshold formulas —
    replaces the old hardcoded ENERGY_PER_CELL = 1.0."""
    rng = random.Random(seed)
    r, c = STATIONS[0]
    pr, pc = None, None
    total, n = 0.0, 0
    for _ in range(sample_steps):
        nr = min(ROWS - 1, max(0, r + rng.choice([-1, 0, 1])))
        nc = min(COLS - 1, max(0, c + rng.choice([-1, 0, 1])))
        if (nr, nc) == (r, c):
            continue
        cost = owt_step_cost(r, c, nr, nc, pr, pc, ELEVATION_GRID, WIND_SPEED, WIND_DIR)
        if cost != float('inf'):
            total += cost
            n += 1
        pr, pc = r, c
        r, c = nr, nc
    return (total / n) if n else 1.0

ENERGY_PER_CELL = round(_calibrate_energy_per_cell(), 4)   # e.g. ≈0.36-0.40 on default config
SAFETY_BUFFER   = 1.2

# ── Cost Function Weights ─────────────────────────────────────────
ALPHA         = 0.5
GAMMA         = 0.2
EPSILON       = 0.3
ETA           = 0.05
LAMBDA        = 8.0
MU            = 1.5
K_TURN        = 0.1
VISIT_PENALTY = 3.0
BACKTRACK_PENALTY = 5.0

HY_T1_GAP_W    = 1.0
HY_T1_DIST_W   = 0.05
HY_T2_INCOMPL_W= 1.5
HY_T2_GAP_W    = 0.8
HY_T2_DIST_W   = 0.1
HY_T2_BORDER_W = 0.4    
HY_T3_BORDER_W = 1.2    
HY_T3_DIST_W   = 0.05

# ── PATROL-SCORING MODE: three-way ablation for Contribution 2 (PPS) ──
# Controls what persistence/urgency signal (if any) the routine
# (tier2/tier3) zone-scoring formula in rank_zones() uses, AND whether
# the hard whole-grid "most neglected cell" override in
# select_zone_mixed_strategy() is active (that override is itself a
# persistence mechanism -- see the note at its call site -- so it must
# be gated the same way, or it silently masks every mode below it).
#
#   "sse_only" -- pure-SSE baseline. No persistence signal of any kind
#                 (soft gap term OR hard neglect-cap override). This is
#                 the clean baseline used for all three comparisons.
#   "sse_gap"  -- current production behavior (unchanged): adds the
#                 simple additive staleness/gap term (HY_T2_GAP_W*tgap,
#                 and tgap directly in tier3). This is what the first
#                 ablation experiment already validated (CV/Gini/max-age
#                 all significantly reduced vs sse_only, p<1e-5, with a
#                 bonus ~20% faster full-coverage, no detection-speed
#                 cost).
#   "sse_pps"  -- full Persistent Patrol Scheduling (PPS) formula: same
#                 gap term as "sse_gap", PLUS an explicit Mission
#                 Urgency interaction term = Threat x Age, where Age is
#                 the same normalized tgap signal and Threat is the
#                 zone's SSE-derived security value (sse_zone_utility --
#                 the only "how important/risky is this zone" signal
#                 available for tier2/3 zones, which by definition have
#                 no CONFIRMED threat; confirmed threats are tier1 and
#                 always handled separately, see score_t1). Unlike the
#                 flat additive gap term, this interaction term makes
#                 staleness matter MORE for zones the game already
#                 considers important, and matter LESS for low-value
#                 zones -- i.e. persistence is prioritized, not uniform.
PATROL_MODE  = "sse_gap"     # "sse_only" | "sse_gap" | "sse_pps"
COVERAGE_CURVE_SAMPLE_EVERY = 5   # steps between coverage-over-time samples
                                   # (Contribution 2 persistence figures)
HY_URGENCY_W = 1.5           # weight for the Threat x Age interaction term, "sse_pps" mode only
URGENCY_FLOOR = 0.4          # min Threat-proxy value after rescaling (see rank_zones' _threat_proxy);
                              # keeps zero/low-SSE zones from losing urgency-priority entirely as
                              # HY_URGENCY_W grows -- see the bug note at _threat_proxy's definition

# ══════════════════════════════════════════════════════════════
# HYBRID GAME-THEORETIC PRIOR (Option C — SSE as a nudge, not a replacement)
# ══════════════════════════════════════════════════════════════
# Honest gap this closes: previously the Stackelberg SSE/DOBSS solvers
# were used ONLY as an offline benchmark to measure SMRS's "regret" against
# the theoretical optimum — SMRS's actual live zone-selection (rank_zones()
# below) never consulted them. That's a fair question a supervisor/
# committee can raise: "if you compute the game-theoretic optimum, why
# doesn't your controller use it?"
#
# This adds the SSE-optimal per-zone coverage PROBABILITY as a small
# additive term in tier2/tier3 zone scoring (see rank_zones()) — SMRS's
# real-time heuristic (battery/relay/threat/gap-driven) remains the
# PRIMARY driver (deliberately: SSE is a static-game solution and can't
# react to battery/relay state, so it shouldn't override real-time safety
# logic), but zones the game theory says are strategically more valuable
# now get a small game-theoretically-justified boost. HY_SSE_W controls
# how much influence this prior has relative to the existing heuristic
# terms — 0.0 fully disables it (reverts to the old pure-heuristic
# behavior) without touching any other code.
HY_SSE_W       = 1.0

# ══════════════════════════════════════════════════════════════
# TERRAIN-AWARE SSE (Model B) — energy folded INTO the strategic
# zone-selection utility, not just the movement-cost physics
# ══════════════════════════════════════════════════════════════
# Gap this closes: TERRAIN_COST_ENABLED only changes calculate_actual_
# step_cost()/energy_to_travel() (movement PHYSICS) -- it never touches
# build_zone_payoffs()/solve_sse() (the STRATEGIC game-theoretic layer),
# which scores zones using ONLY zone_border_priority(). So "terrain-aware
# planning" so far has only been "terrain-aware energy accounting"; the
# actual SSE zone-selection decision has never seen terrain at all.
#
# U_z = SSE_z − λ·Ê_z  (see get_zone_energy_hat() below for Ê_z, and
# sse_zone_utility() below for how this replaces the plain sse_cov.get(z)
# term inside rank_zones()). SSE_TERRAIN_AWARE=False reproduces
# the EXACT old behavior (Model A, U_z = SSE_z) -- this is an additive,
# opt-in toggle, not a rewrite of the existing SSE prior.
SSE_TERRAIN_AWARE = False   # Model A (off) vs Model B/C (on)

# SSE_LAMBDA — NEEDS EMPIRICAL TUNING, not a hand-picked constant.
# Same "why 12?" trap flagged earlier for COVERAGE_MAINTAINED_BONUS: pick
# this via a train/test seed split (see run_terrain_aware_sse_experiment.py
# run_lambda_sensitivity()), never just eyeball one value and report it.
# Starting point of 1.0 makes Ê_z (already normalized to [0,1], same scale
# as sse_cov which solve_sse() also returns in [0,1]) comparable in
# magnitude to the SSE term it's being subtracted from -- a deliberate
# choice so the initial default isn't ALSO an arbitrary unjustified number,
# but it is only a **starting point** for the sensitivity sweep.
SSE_LAMBDA = 1.0

# ══════════════════════════════════════════════════════════════
# ADAPTIVE LEARNING (online, experience-based — contrast with the SSE
# prior above, which is static/precomputed from game theory)
# ══════════════════════════════════════════════════════════════
# Answers a different question than HY_SSE_W: "if the drone learns a
# per-zone threat-risk estimate purely from what it observes DURING this
# run (no game-theoretic structure, no precomputed prior), does zone
# scoring informed by that online estimate detect MORE crossings than the
# static game-theoretic (SSE) prior?" This is deliberately a separate,
# independently-toggleable mechanism (USE_ADAPTIVE_LEARNING) so the two
# can be compared head-to-head.
#
# Mechanism: a simple online exponential-moving-average per zone,
# LEARNED_ZONE_RISK[zone] in [0,1] — nudged toward 1 whenever a threat is
# found in that zone during patrol, toward 0 otherwise (see
# update_learned_risk()). This is intentionally simple/transparent (not a
# neural net or RL agent) so its effect on detection% is easy to reason
# about and defend — a from-scratch deep-RL agent is a much larger,
# riskier undertaking than this thesis's remaining timeline can safely
# absorb; this captures the core idea (environment-driven adaptation vs
# static game-theoretic prior) at low implementation risk.
USE_ADAPTIVE_LEARNING = False   # off by default — zero effect on existing behavior/results
HY_ADAPT_W     = 2.0            # influence of the learned-risk nudge (same role as HY_SSE_W)
ADAPT_PRIOR    = 0.15           # neutral starting estimate per zone (~ base threat rate); see make_grid/NUM_THREATS
ADAPT_ALPHA    = 0.15           # EMA learning rate — higher = adapts faster but noisier
LEARNED_ZONE_RISK = {}          # {zone: risk_estimate}, reset per simulation run — see reset_learned_risk()

# Step 4: replaces score_t2()'s hand-tuned/auto-tuned linear formula with
# get_q_value(z, i, Q_WEIGHTS, sse_cov) -- same features, weights learned
# online via TD-update instead of fixed/offline-searched. Off by default;
# tier1 (active threats) is untouched either way, for the reason given
# above. Q_WEIGHTS itself is defined further down, right after
# init_q_weights() -- see the LINEAR Q-LEARNING FOUNDATION section.
USE_Q_LEARNING = False   # off by default — zero effect on existing behavior/results

# Non-linear Li-ion SOC model (voltage sag + Peukert effect, see
# owt_voltage_at_soc()/owt_peukert_derating() near the top of the file).
# Off by default — zero effect on existing behavior/results/reproducibility
# of every experiment run earlier in this project. When True,
# _advance_relay_policy() passes each drone's CURRENT battery %% into
# calculate_actual_step_cost(), switching every step-cost calculation from
# the plain linear model to the non-linear one.
USE_NONLINEAR_BATTERY = True   # ENABLED — relay simulations now use voltage-sag + Peukert SOC model

# State-of-Health (SOH) / capacity-aging model (see owt_soh_capacity_fraction()
# / owt_usable_capacity_joules_with_soh() near the top of the file, and
# FIX #5 in _owt_joules_to_battery_pct()).
# Off by default — zero effect on existing behavior/results/reproducibility
# of every experiment run earlier in this project. When True, each drone's
# OWN recharge counter (self.sR/self.greR/self.gR/self.aR, or state["RC"]
# inside _advance_relay_policy) is passed into calculate_actual_step_cost()
# as cycle_count, so a pack that has already been recharged many times
# loses more %-points per step than a fresh one -- i.e. the SAME physical
# work costs progressively more battery percentage as the pack ages over
# the run. This is fully independent of USE_NONLINEAR_BATTERY above: either
# can be on/off in any combination.
USE_SOH_AGING = True   # ENABLED — relay simulations now fade usable capacity with recharge-cycle count

# ══════════════════════════════════════════════════════════════
# BIASED ENVIRONMENT (unknown-to-defender persistent zone bias)
# ══════════════════════════════════════════════════════════════
# Honest gap this closes: the old threat-placement (random.shuffle(non_st)
# + take first NUM_THREATS) is uniform i.i.d. over all non-station cells,
# every single run. That has ZERO exploitable structure — there is nothing
# for USE_ADAPTIVE_LEARNING's per-zone EMA to actually learn, because the
# long-run threat rate is identical in every zone by construction. So a
# "does adaptive learning help?" test run against this environment can
# never show a real effect either way — any measured difference is just
# noise, and a null result is meaningless (not evidence adaptive learning
# doesn't work, just evidence the environment gave it nothing to find).
#
# USE_BIASED_ENVIRONMENT introduces a genuinely learnable — but still
# hidden — pattern: for a given seed, a subset of zones ("hot" zones) is
# picked and gets a persistently higher share of threats every run with
# that seed, while all other zones ("cold" zones) share the remainder.
# Crucially, which zones are hot is NEVER exposed to any defender-facing
# code (rank_zones/SSE/build_zone_payoffs never read it) — the only way
# the defender could ever exploit this bias is by actually noticing it
# from its own sensor detections, which is exactly what
# USE_ADAPTIVE_LEARNING's online EMA is supposed to do. This makes it
# possible to honestly test whether USE_ADAPTIVE_LEARNING works: compare
# adaptive-ON vs adaptive-OFF detection% under USE_BIASED_ENVIRONMENT=True
# (a real, learnable pattern exists) versus under the old uniform
# environment (no pattern exists — expected null result, useful as a
# negative control).
USE_BIASED_ENVIRONMENT      = False   # off by default — zero effect on existing behavior/results
BIASED_HOT_ZONE_FRACTION    = 0.25    # fraction of all zones designated "hot" (persistent, per-seed)
BIASED_HOT_ZONE_THREAT_SHARE= 0.70    # share of NUM_THREATS placed inside hot zones (rest -> cold zones)

def _select_hot_zones(seed):
    """Deterministically (per seed) picks a hidden subset of zones as
    'hot' for this run — used only when USE_BIASED_ENVIRONMENT is True.
    Uses an INDEPENDENT random.Random stream (not the shared `random`
    module) keyed off `seed` with large prime multipliers, so it never
    consumes/perturbs the global `random` state that other code
    (make_grid, zone_rng, etc.) depends on for reproducibility. No
    defender-facing function ever reads this — it exists purely to shape
    the *environment*, not to give the defender a free prior."""
    zones = get_all_zones()
    rng = random.Random(seed * 104729 + 7919)
    n_hot = max(1, min(len(zones), round(len(zones) * BIASED_HOT_ZONE_FRACTION)))
    return set(rng.sample(zones, n_hot))

def place_threats(seed, grids, reseed=True):
    """Central threat-placement routine, used at every simulation entry
    point (replaces the previously-duplicated 'shuffle non-station cells,
    take first NUM_THREATS' snippet that appeared at each call site).

    When USE_BIASED_ENVIRONMENT is False (default): behavior is IDENTICAL
    to the original code — `random.seed(seed)` (if reseed=True, matching
    what each original call site did) then `random.shuffle(non_st)` on
    the shared `random` module, taking the first NUM_THREATS cells. This
    guarantees zero change to any existing experiment/result while the
    flag stays off.

    When True: BIASED_HOT_ZONE_THREAT_SHARE of NUM_THREATS threats are
    placed uniformly at random among the cells of this seed's hidden
    'hot' zones (see _select_hot_zones), the remainder placed uniformly
    among all other ('cold') zone cells — using an independent RNG stream
    so it never disturbs the shared `random` module's state either.

    grids: list of grid dicts to stamp identically with `threat=True`
    (mirrors the multi-grid stamping — self.gs/self.ggre/self.ggps/self.gaco
    — done at each existing call site, so SMRS/greedy/GPS/ACO all see the
    exact same threat placement for a fair comparison).

    Returns the list of threat positions placed.
    """
    if not USE_BIASED_ENVIRONMENT:
        if reseed:
            random.seed(seed)
        non_st = [(r, c) for r in range(ROWS) for c in range(COLS) if not is_st(r, c)]
        random.shuffle(non_st)
        threats = non_st[:NUM_THREATS]
    else:
        hot_zones = _select_hot_zones(seed)
        rng = random.Random(seed * 7919 + 104729)
        hot_cells, cold_cells = [], []
        for r in range(ROWS):
            for c in range(COLS):
                if is_st(r, c):
                    continue
                (hot_cells if get_zone_id(r, c) in hot_zones else cold_cells).append((r, c))
        rng.shuffle(hot_cells)
        rng.shuffle(cold_cells)
        n_hot = min(round(NUM_THREATS * BIASED_HOT_ZONE_THREAT_SHARE), len(hot_cells))
        n_cold = min(NUM_THREATS - n_hot, len(cold_cells))
        threats = hot_cells[:n_hot] + cold_cells[:n_cold]
        shortfall = NUM_THREATS - len(threats)
        if shortfall > 0:   # edge case: one pool too small — top up from the other's leftovers
            leftover = hot_cells[n_hot:] + cold_cells[n_cold:]
            threats += leftover[:shortfall]

    for t in threats:
        for g in grids:
            g[t]["threat"] = True
    return threats

def reset_learned_risk():
    """Call at the start of a run that will use adaptive learning, so each
    run's learning starts fresh (no leakage between seeds/runs)."""
    global LEARNED_ZONE_RISK
    LEARNED_ZONE_RISK = {z: ADAPT_PRIOR for z in get_all_zones()}

def update_learned_risk(zone, threat_found_here):
    """Online EMA update — called once per SMRS step (see do_step() hooks
    below), only when USE_ADAPTIVE_LEARNING is on."""
    global LEARNED_ZONE_RISK
    old = LEARNED_ZONE_RISK.get(zone, ADAPT_PRIOR)
    LEARNED_ZONE_RISK[zone] = (1 - ADAPT_ALPHA) * old + ADAPT_ALPHA * (1.0 if threat_found_here else 0.0)

def _adaptive_learning_step_hook(r, c, g):
    """Called right after the SMRS active drone's sensor_sweep each step
    (both DroneSimHeadless and DroneSimGUI) — a complete no-op unless
    USE_ADAPTIVE_LEARNING is on, so this never affects any existing
    result/behavior when the feature isn't explicitly being tested."""
    if not USE_ADAPTIVE_LEARNING:
        return
    zr, zc = get_zone_id(r, c)
    found = any(cell["threat"] and cell["threat_detected"] for cell in
                (g[pos] for pos in get_zone_cells(zr, zc) if pos in g))
    update_learned_risk((zr, zc), found)

# ══════════════════════════════════════════════════════════════
# GRID SETUP
# ══════════════════════════════════════════════════════════════
def make_grid(seed=42):
    random.seed(seed)
    g = {}
    for r in range(ROWS):
        for c in range(COLS):
            edge = min(r, c, ROWS-1-r, COLS-1-c)
            pri  = 3 if edge == 0 else (2 if edge == 1 else 1)
            g[(r,c)] = {
                "covered": 0, "priority": pri,
                "risk": round(random.uniform(0.1, 0.9), 2),
                "threat": False, "is_station": False,
                "visits": 0, "threat_detected": False,
                "last_visited_step": 0
            }
    for s in STATIONS:
        g[s]["is_station"] = True
        g[s]["priority"]   = 0
    return g

def is_st(r,c): return (r,c) in STATIONS
def nearest_st(r,c): return min(STATIONS, key=lambda s: abs(r-s[0])+abs(c-s[1]))
def get_zone_id(r,c): return (r//ZONE_SIZE, c//ZONE_SIZE)

def get_zone_cells(zr,zc):
    out = []
    for r in range(zr*ZONE_SIZE, min((zr+1)*ZONE_SIZE, ROWS)):
        for c in range(zc*ZONE_SIZE, min((zc+1)*ZONE_SIZE, COLS)):
            if not is_st(r,c):
                out.append((r,c))
    return out

def get_all_zones():
    return [(zr,zc) for zr in range(math.ceil(ROWS/ZONE_SIZE)) for zc in range(math.ceil(COLS/ZONE_SIZE))]

def nbrs(r,c,g):
    out = []
    for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
        nr,nc = r+dr, c+dc
        if 0<=nr<ROWS and 0<=nc<COLS and not g[(nr,nc)]["is_station"]:
            out.append((nr,nc))
    return out

def coverage_pct(g):
    non_station = [k for k,v in g.items() if not v["is_station"]]
    if not non_station: return 0.0
    covered = sum(1 for k in non_station if g[k]["covered"] >= 100)
    return round(covered/len(non_station)*100, 1)

def sensor_sweep(r, c, g, detected=None, step=None, pr=None, pc=None, radius=DETECTION_RADIUS):
    cells = [(r, c)]

    if pr is None or pc is None or (pr, pc) == (r, c):
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            cells.append((r + dr, c + dc))
    else:
        hr, hc = r - pr, c - pc 
        if hr != 0:
            for k in range(1, radius + 1):
                cells.append((r, c - k))
                cells.append((r, c + k))
        elif hc != 0:
            for k in range(1, radius + 1):
                cells.append((r - k, c))
                cells.append((r + k, c))

    for pos in cells:
        cell = g.get(pos)
        if not cell or cell["is_station"]: continue
        cell["covered"] = 100
        if step is not None:
            cell["last_visited_step"] = step
        if detected is not None and cell["threat"] and pos not in detected:
            detected.add(pos)
            cell["threat_detected"] = True

# ══════════════════════════════════════════════════════════════
# REALISTIC PHYSICS & ENERGY MODEL
# ══════════════════════════════════════════════════════════════

def calculate_actual_step_cost(r_old, c_old, r_new, c_new, pr, pc, current_soc_pct=None, cycle_count=0):
    """Holistic Energy Model — now backed by owt_physics.py's physically-derived
    OWT-A* model (vector-triangle wind cost, momentum-theory turn cost,
    gravitational + induced-power climb cost) instead of the old heuristic
    wind_factor/mass_factor formula. Signature unchanged so every call site
    below keeps working as-is. Elevation comes from ELEVATION_GRID (real
    SRTM or synthetic fallback, see srtm_loader.py) UNLESS
    TERRAIN_COST_ENABLED is False, in which case _FLAT_ELEVATION_GRID
    (all-zero elevation) is used instead -- the real flat/terrain-aware
    ablation toggle (see TERRAIN_COST_ENABLED above).

    current_soc_pct: OPTIONAL passthrough to owt_step_cost()'s non-linear
    (voltage-sag + Peukert) battery model. Default None = old linear
    behavior, unaffected. See USE_NONLINEAR_BATTERY below for the toggle
    that actually turns this on inside _advance_relay_policy().

    cycle_count: OPTIONAL passthrough to owt_step_cost()'s SOH capacity-
    fade model. Default 0 = fresh-battery behavior, unaffected. See
    USE_SOH_AGING below for the toggle that turns this on."""
    grid = ELEVATION_GRID if TERRAIN_COST_ENABLED else _FLAT_ELEVATION_GRID
    wind_speed_now, wind_dir_now = get_current_wind()
    return owt_step_cost(r_old, c_old, r_new, c_new, pr, pc,
                          grid, wind_speed_now, wind_dir_now,
                          current_soc_pct=current_soc_pct, cycle_count=cycle_count)

def energy_to_travel(r1, c1, r2, c2):
    """Predictive Oracle — now backed by owt_physics.py's owt_energy_to_travel(),
    the same physically-derived model as calculate_actual_step_cost() above
    (Manhattan-decomposed straight-line estimate, wind/turn/climb all
    physics-consistent with the actual step cost), so must_recharge_now()/
    two_hop_check()/E_ret() stay accurate against the real per-step drain.
    Also respects TERRAIN_COST_ENABLED -- MUST stay consistent with
    calculate_actual_step_cost() above, or RTB safety margins get computed
    against a different terrain model than the real per-step drain."""
    grid = ELEVATION_GRID if TERRAIN_COST_ENABLED else _FLAT_ELEVATION_GRID
    wind_speed_now, wind_dir_now = get_current_wind()
    return owt_energy_to_travel(r1, c1, r2, c2, grid, wind_speed_now, wind_dir_now)

def D_eff(r1,c1,r2,c2,pr,pc):
    D    = math.sqrt((r2-r1)**2 + (c2-c1)**2)
    turn = 0 if pr is None else (0 if (r2-r1==r1-pr and c2-c1==c1-pc) else 1)
    return round(D*(1+K_TURN*turn), 3), turn

def W_eff(r1,c1,r2,c2,ws=None,wd=None):
    # ws/wd default to None (not WIND_SPEED/WIND_DIR directly) because
    # Python evaluates default arguments ONCE at function-definition time --
    # binding them to the static constants here would silently freeze this
    # function's wind reading at import time and make it ignore
    # WIND_DYNAMIC_ENABLED entirely, even though it feeds cost_fn()'s
    # per-step path-cost heuristic. Resolving via get_current_wind() here
    # instead keeps this consistent with calculate_actual_step_cost().
    if ws is None or wd is None:
        ws_now, wd_now = get_current_wind()
        ws = ws_now if ws is None else ws
        wd = wd_now if wd is None else wd
    h    = math.degrees(math.atan2(c2-c1, r2-r1)) % 360
    diff = abs(wd-h) % 360
    if diff > 180: diff = 360-diff
    return round(ws*(diff/180)*(1+50/200), 3)

def E_ret(r,c):
    # BUGFIX: this used to compute `dist` to the NEAREST station but then
    # call energy_to_travel() to the fixed STATIONS[0] corner (0,0)
    # regardless of which station was actually nearest — `dist` was dead
    # and cost_fn()'s ETA*er term was biased against cells far from (0,0)
    # even when a closer station existed. Now uses the real nearest station.
    st = nearest_st(r, c)
    return energy_to_travel(r, c, st[0], st[1]) * SAFETY_BUFFER

def two_hop_check(dr,dc,tr,tc,battery):
    e1 = energy_to_travel(dr,dc,tr,tc)
    st = nearest_st(tr,tc)
    e2 = energy_to_travel(tr,tc,st[0],st[1])
    eni = (e1+e2)*SAFETY_BUFFER + ENERGY_PER_CELL
    return battery>=eni, round(e1,1), round(e2,1), round(eni,1)

def must_recharge_now(r,c,battery):
    st    = nearest_st(r,c)
    e_ret = energy_to_travel(r,c,st[0],st[1])
    threshold = e_ret*SAFETY_BUFFER + ENERGY_PER_CELL
    return battery<=threshold, round(e_ret,1), st

def needs_handoff_now(r, c, battery):
    st   = nearest_st(r, c)
    # BUGFIX: E_AVIONICS/E_HOVER/E_TURN were leftovers from the OLD
    # heuristic energy model (different units/scale) and are no longer
    # used anywhere else in the file now that owt_step_cost() (physics
    # model) drives calculate_actual_step_cost(). ENERGY_PER_CELL is now
    # auto-calibrated to the real physics scale (see _calibrate_energy_per_cell
    # above), so it alone is the correct "1 cell of travel" unit here.
    early_margin = 10 * (ENERGY_PER_CELL * ((UAV_MASS + PAYLOAD_MASS) / UAV_MASS) ** 1.5)
    hard_threshold = energy_to_travel(r, c, st[0], st[1]) * SAFETY_BUFFER + ENERGY_PER_CELL 
    return battery <= (hard_threshold + early_margin), st

# ══════════════════════════════════════════════════════════════
# SMRS — Zone System & Pathing
# ══════════════════════════════════════════════════════════════
def zone_info(zr,zc,dr,dc,g,detected,step):
    cells = get_zone_cells(zr,zc)
    if not cells: return None
    avg_cov    = sum(g[c]["covered"] for c in cells) / len(cells)
    incompl    = (100-avg_cov)/100
    has_threat = any(g[c]["threat"] and c not in detected for c in cells)
    ls         = min((g[c]["last_visited_step"] for c in cells), default=0)
    raw_gap    = step - ls                      
    tgap       = min(raw_gap/200, 1.0)
    zrc, zcc   = zr*ZONE_SIZE + ZONE_SIZE//2, zc*ZONE_SIZE + ZONE_SIZE//2
    travel     = abs(dr-zrc)+abs(dc-zcc)
    border_pr  = (sum(g[c]["priority"] for c in cells) / len(cells)) / 3.0
    return {"has_threat": has_threat, "incompl": incompl, "tgap": tgap, "raw_gap": raw_gap,
            "travel": travel, "avg_cov": avg_cov, "border_pr": border_pr}

def rank_zones(dr,dc,g,detected,step):
    sse_cov = get_sse_zone_coverage()   # cached after first call; {} if unavailable (safe no-op)
    energy_hat = get_zone_energy_hat() if SSE_TERRAIN_AWARE else {}   # only computed when actually needed

    all_info = []
    for (zr,zc) in get_all_zones():
        info = zone_info(zr,zc,dr,dc,g,detected,step)
        if info is None: continue
        all_info.append(((zr,zc), info))

    tier1 = [(z,i) for z,i in all_info if i["has_threat"]]
    tier2 = [(z,i) for z,i in all_info if (not i["has_threat"]) and i["incompl"] > 0]
    tier3 = [(z,i) for z,i in all_info if (not i["has_threat"]) and i["incompl"] == 0]

    # Tier1 (active-threat zones) deliberately does NOT get the SSE nudge —
    # a confirmed threat is real-time, urgent information the static SSE
    # game can't see, and should never be diluted by a long-run statistical
    # prior. Tier2/tier3 (routine patrol scheduling) are exactly the kind
    # of decision the SSE prior is meant to inform. The adaptive-learning
    # term (see ADAPTIVE LEARNING section below) is the same idea but
    # ONLINE-learned from this run's own experience instead of a static
    # game-theoretic prior — also excluded from tier1 for the same reason,
    # and a complete no-op (LEARNED_ZONE_RISK.get(...) never called with
    # nonzero weight) unless USE_ADAPTIVE_LEARNING is explicitly turned on.
    def adapt_bonus(z):
        return HY_ADAPT_W * LEARNED_ZONE_RISK.get(z, ADAPT_PRIOR) if USE_ADAPTIVE_LEARNING else 0.0
    # Tier1 (confirmed-threat zones) keeps its gap term regardless of the
    # patrol mode -- that urgency signal isn't part of the routine
    # SSE-vs-persistence patrol-scheduling question PATROL_MODE targets,
    # and a CONFIRMED threat should never be diluted by a probabilistic
    # importance term the way tier2/3's Mission Urgency term is.
    def score_t1(i): return HY_T1_GAP_W*i["tgap"] - HY_T1_DIST_W*i["travel"]

    # PPS mode only: rescale the per-zone SSE value into a Threat proxy in
    # [URGENCY_FLOOR, 1.0] instead of using the raw sse_zone_utility value
    # directly. BUG FOUND EMPIRICALLY (30-seed sweep, HY_URGENCY_W 1.5 ->
    # 4.0): sse_zone_utility takes only 3 distinct values across the 36
    # zones here, and is exactly 0.0 for 16/36 of them. With the raw value
    # as the Threat multiplier, those 16 zones get URGENCY = 0 no matter
    # how large HY_URGENCY_W is, while the other 20 zones' urgency keeps
    # growing with the weight -- so increasing the weight didn't sharpen
    # prioritization, it starved the zero-SSE zones' relative priority as
    # they aged, which is the OPPOSITE of what a persistence term should
    # do. Measured effect: raising HY_URGENCY_W 1.5->4.0 made every
    # persistence metric (CV, mean/max age, full-coverage speed) worse,
    # not better. Rescaling to a floor >0 guarantees every zone still
    # gets a meaningful, growing urgency contribution as it goes stale,
    # regardless of its static SSE value.
    _sse_vals = [sse_zone_utility(zz, sse_cov, energy_hat) for zz, _ in tier2 + tier3]
    _sse_lo = min(_sse_vals) if _sse_vals else 0.0
    _sse_hi = max(_sse_vals) if _sse_vals else 0.0
    _sse_span = (_sse_hi - _sse_lo) or 1.0  # avoid /0 when all zones tie

    def _threat_proxy(z):
        raw = sse_zone_utility(z, sse_cov, energy_hat)
        normalized = (raw - _sse_lo) / _sse_span          # -> [0, 1]
        return URGENCY_FLOOR + (1.0 - URGENCY_FLOOR) * normalized  # -> [URGENCY_FLOOR, 1]

    def _persistence_terms(z, i):
        """Returns (gap_term, urgency_term) for tier2/tier3 scoring,
        per PATROL_MODE. urgency_term is the Mission Urgency = Threat x
        Age interaction (Threat = _threat_proxy(z), a floor-rescaled SSE
        value -- see note above; Age = the same normalized tgap the
        gap_term already uses, no separate Age counter needed)."""
        if PATROL_MODE == "sse_only":
            return 0.0, 0.0
        gap_term = i["tgap"]
        if PATROL_MODE == "sse_pps":
            urgency_term = HY_URGENCY_W * _threat_proxy(z) * i["tgap"]
        else:  # "sse_gap"
            urgency_term = 0.0
        return gap_term, urgency_term

    def score_t2(z,i):
        if USE_Q_LEARNING:
            return get_q_value(z, i, Q_WEIGHTS, sse_cov)
        gap_term, urgency_term = _persistence_terms(z, i)
        return (HY_T2_INCOMPL_W*i["incompl"] + HY_T2_GAP_W*gap_term + urgency_term + HY_T2_BORDER_W*i["border_pr"]
                + HY_SSE_W*sse_zone_utility(z, sse_cov, energy_hat) + adapt_bonus(z) - HY_T2_DIST_W*i["travel"])
    def score_t3(z,i):
        gap_term, urgency_term = _persistence_terms(z, i)
        return gap_term + urgency_term + HY_T3_BORDER_W*i["border_pr"] + HY_SSE_W*sse_zone_utility(z, sse_cov, energy_hat) + adapt_bonus(z) - HY_T3_DIST_W*i["travel"]

    tier1.sort(key=lambda zi: -score_t1(zi[1]))
    tier2.sort(key=lambda zi: -score_t2(zi[0], zi[1]))
    tier3.sort(key=lambda zi: -score_t3(zi[0], zi[1]))

    scored = []
    for z,i in tier1: scored.append({"zone":z, "score":round(score_t1(i),4), "cov":round(i["avg_cov"],1), "tier":1, "raw_gap":i["raw_gap"]})
    for z,i in tier2: scored.append({"zone":z, "score":round(score_t2(z,i),4), "cov":round(i["avg_cov"],1), "tier":2, "raw_gap":i["raw_gap"]})
    for z,i in tier3: scored.append({"zone":z, "score":round(score_t3(z,i),4), "cov":round(i["avg_cov"],1), "tier":3, "raw_gap":i["raw_gap"]})
    return scored

def calc_C(cov): return 0.0 if cov >= 100 else (2.0 if cov == 0 else 1.0)

def cost_fn(r1,c1,r2,c2,pr,pc,g,detected,target_zone=None):
    d,turn = D_eff(r1,c1,r2,c2,pr,pc)
    w      = W_eff(r1,c1,r2,c2)
    R      = g[(r2,c2)]["risk"]
    er     = E_ret(r2,c2)
    C      = calc_C(g[(r2,c2)]["covered"])
    P      = g[(r2,c2)]["priority"]
    dp     = 20.0 if g[(r2,c2)]["covered"] >= 100 else 0
    V      = VISIT_PENALTY*g[(r2,c2)]["visits"] + dp
    tc     = 10.0 if (g[(r2,c2)]["threat"] and (r2,c2) in detected) else (-4.0 if g[(r2,c2)]["threat"] else 0)
    zb     = -3.0 if (target_zone and get_zone_id(r2,c2)==target_zone) else 0
    bt     = BACKTRACK_PENALTY if (pr,pc)==(r2,c2) else 0
    cost   = ALPHA*d+GAMMA*w+EPSILON*R+ETA*er+V+tc+zb+bt-LAMBDA*C-MU*P
    return round(cost,4), {"cost":round(cost,4)}

def smart_move(r,c,pr,pc,g,detected,target_zone=None):
    ns = nbrs(r,c,g)
    if not ns: return r,c,{}
    best,bpos,bbd = float('inf'),(r,c),{}
    for nr,nc in ns:
        cv,bd = cost_fn(r,c,nr,nc,pr,pc,g,detected,target_zone)
        if cv < best: best=cv; bpos=(nr,nc); bbd=bd
    g[bpos]["visits"] += 1
    return bpos[0],bpos[1],bbd

def move_toward(r,c,tr,tc_):
    nr,nc = r,c
    if r!=tr: nr = r+(1 if tr>r else -1)
    elif c!=tc_: nc = c+(1 if tc_>c else -1)
    if is_st(nr,nc) and (nr,nc)!=(tr,tc_): return r,c
    return nr,nc

def bfs_next_step(r, c, tr, tc, g):
    from collections import deque
    if (r, c) == (tr, tc): return r, c
    start, goal, visited, parent, q = (r, c), (tr, tc), {(r,c)}, {}, deque([(r,c)])
    found = False
    while q:
        cur = q.popleft()
        if cur == goal:
            found = True; break
        for nb in nbrs(cur[0], cur[1], g):
            if nb not in visited:
                visited.add(nb); parent[nb] = cur; q.append(nb)
    if not found: return r, c
    step = goal
    while parent.get(step) != start and step in parent: step = parent[step]
    return step

def find_uncovered_in_zone(zr,zc,g):
    cells   = get_zone_cells(zr,zc)
    empties = [c for c in cells if g[c]["covered"]==0]
    if empties: return empties[0]
    partials = [c for c in cells if 0<g[c]["covered"]<100]
    return partials[0] if partials else None

def pick_feasible_zone(ranked, g, r, c, battery):
    fallback_zone, fallback_cell = None, None
    for entry in ranked:
        zone = entry["zone"]
        cell = find_uncovered_in_zone(*zone, g)
        if cell is None: continue
        if fallback_zone is None: fallback_zone, fallback_cell = zone, cell
        can, *_ = two_hop_check(r, c, cell[0], cell[1], battery)
        if can: return zone, cell
    return fallback_zone, fallback_cell

RANDOMIZATION_TAU  = 6.0     
NEGLECT_CAP_STEPS  = 400     

def find_most_neglected_cell(g, step):
    worst_cell, worst_gap = None, -1
    for pos, cell in g.items():
        if cell["is_station"]: continue
        gap = step - cell["last_visited_step"]
        if gap > worst_gap:
            worst_gap = gap
            worst_cell = pos
    return worst_cell, worst_gap

def select_zone_mixed_strategy(ranked, g, r, c, battery, rng, step, tau=None, neglect_cap=None):
    if tau is None: tau = RANDOMIZATION_TAU
    if neglect_cap is None: neglect_cap = NEGLECT_CAP_STEPS
    if not ranked: return None, None

    top_tier = ranked[0]["tier"]

    # IMPORTANT: this whole-grid "most neglected cell" override is itself a
    # hard, deterministic persistence mechanism -- completely independent of
    # (and, empirically, far more influential than) the soft additive gap
    # term inside rank_zones()'s score. Verified experimentally: with this
    # left unconditional, changing the score-level persistence term produced
    # a byte-identical zone-visit sequence, because this override was firing
    # on almost every non-tier1 decision once any cell crossed
    # NEGLECT_CAP_STEPS, masking the score-level ablation entirely. Gating
    # it on PATROL_MODE too makes "sse_only" a genuinely clean baseline with
    # NO persistence mechanism of any kind (soft or hard).
    if top_tier != 1 and PATROL_MODE != "sse_only":
        worst_cell, worst_gap = find_most_neglected_cell(g, step)
        if worst_cell is not None and worst_gap >= neglect_cap:
            can, *_ = two_hop_check(r, c, worst_cell[0], worst_cell[1], battery)
            if can: return get_zone_id(*worst_cell), worst_cell

    tier_entries = [e for e in ranked if e["tier"] == top_tier]
    feasible = []
    for e in tier_entries:
        cell = find_uncovered_in_zone(*e["zone"], g)
        if cell is None: continue
        can, *_ = two_hop_check(r, c, cell[0], cell[1], battery)
        if can: feasible.append((e, cell))

    if not feasible:
        return pick_feasible_zone(ranked, g, r, c, battery)

    scores = [e["score"] for e, _ in feasible]
    m = max(scores)
    weights = [math.exp((s - m) / max(tau, 1e-6)) for s in scores]
    total = sum(weights)
    probs = [w/total for w in weights]

    pick = rng.random()
    cum = 0.0
    for (e, cell), p in zip(feasible, probs):
        cum += p
        if pick <= cum: return e["zone"], cell
    return feasible[-1][0]["zone"], feasible[-1][1]   

# ══════════════════════════════════════════════════════════════
# BASELINES
# ══════════════════════════════════════════════════════════════
def greedy_move(r,c,pr,pc,g,detected):
    ns = nbrs(r,c,g)
    if not ns: return r,c
    best_s,best_p = -float('inf'),(r,c)
    for nr,nc in ns:
        cell  = g[(nr,nc)]
        score = ((100-cell["covered"])+(50 if cell["threat"] and (nr,nc) not in detected else 0)+cell["priority"]*10-cell["visits"]*15)
        if (nr,nc) == (pr,pc): score -= 8
        if score > best_s: best_s=score; best_p=(nr,nc)
    g[best_p]["visits"] += 1
    return best_p[0],best_p[1]

def gps_snake(r,c,d):
    nc,nr = c+d, r
    if nc<0 or nc>=COLS or is_st(r,nc): nr=(r+1)%ROWS; d=-d; nc=c+d
    if 0<=nr<ROWS and 0<=nc<COLS and is_st(nr,nc): nc+=d
    if not(0<=nc<COLS): nc=c
    return nr,nc,d

# ══════════════════════════════════════════════════════════════
# 4th POLICY — ACO (Ant Colony Optimization) coverage scheduling
# ══════════════════════════════════════════════════════════════
# SOTA metaheuristic baseline: zone-visit ORDER precompute করা হয় ACO
# দিয়ে (একটা open-path/Hamiltonian-path TSP-এর মতো সমস্যা — শুরুর
# অবস্থান থেকে সব zone একবার করে ভ্রমণ করে, মোট distance minimize
# করা)। এরপর single drone সেই precomputed route অনুসরণ করে (GPS-Snake
# baseline-এর মতোই single-drone, শুধু movement pattern deterministic
# lawnmower না, বরং ACO-optimized zone-visiting order)।

def _manhattan(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

def aco_solve_route(zones, start_rc, n_ants=20, iterations=40,
                     alpha=1.0, beta=3.0, evaporation=0.5, Q=100.0, seed=7):
    """Ant Colony Optimization দিয়ে zone-visit order (route) বের করে।
    Standard ACO-for-TSP formulation: pheromone(tau) ও heuristic(eta=1/distance)
    দিয়ে প্রতিটা ant সম্ভাব্যতা-ভিত্তিকভাবে next-zone বেছে সম্পূর্ণ route
    তৈরি করে; সবচেয়ে ছোট route-এ বেশি pheromone জমা হয় (reinforcement),
    আর প্রতি iteration-এ পুরনো pheromone evaporate হয়।"""
    rng = random.Random(seed)
    centers = [(zr*ZONE_SIZE+ZONE_SIZE//2, zc*ZONE_SIZE+ZONE_SIZE//2) for (zr, zc) in zones]
    n = len(zones)
    pher = [[1.0]*n for _ in range(n)]
    start_pher = [1.0]*n
    best_route, best_len = list(range(n)), float('inf')

    def weighted_pick(candidates, weights):
        total = sum(weights)
        if total <= 0: return candidates[0]
        r = rng.random()*total
        cum = 0.0
        for j, w in zip(candidates, weights):
            cum += w
            if r <= cum: return j
        return candidates[-1]

    for _ in range(iterations):
        routes = []
        for _ant in range(n_ants):
            unvisited = set(range(n))
            cands = list(unvisited)
            weights = [(start_pher[j]**alpha) * ((1.0/(_manhattan(start_rc, centers[j])+1e-6))**beta) for j in cands]
            first = weighted_pick(cands, weights)
            route = [first]; unvisited.remove(first); cur = first
            while unvisited:
                cands = list(unvisited)
                weights = [(pher[cur][j]**alpha) * ((1.0/(_manhattan(centers[cur], centers[j])+1e-6))**beta) for j in cands]
                nxt = weighted_pick(cands, weights)
                route.append(nxt); unvisited.remove(nxt); cur = nxt
            length = _manhattan(start_rc, centers[route[0]])
            for k in range(len(route)-1):
                length += _manhattan(centers[route[k]], centers[route[k+1]])
            routes.append((route, length))
            if length < best_len:
                best_len, best_route = length, route[:]

        for i in range(n):
            start_pher[i] *= (1-evaporation)
            for j in range(n): pher[i][j] *= (1-evaporation)
        for route, length in routes:
            deposit = Q/max(length, 1e-6)
            start_pher[route[0]] += deposit
            for k in range(len(route)-1):
                a, b = route[k], route[k+1]
                pher[a][b] += deposit; pher[b][a] += deposit

    return [zones[i] for i in best_route]

def aco_move(r, c, route, idx, g):
    """ACO-precomputed route ধরে drone-কে এক ধাপ move করায়; বর্তমান
    target zone সম্পূর্ণ covered হয়ে গেলে route-এর পরের zone-এ চলে যায়
    (route শেষ হলে আবার শুরু থেকে loop করে — দীর্ঘমেয়াদি patrol)।"""
    n = len(route)
    tries = 0
    zone = route[idx % n]
    target = find_uncovered_in_zone(*zone, g)
    while target is None and tries < n:
        idx += 1; tries += 1
        zone = route[idx % n]
        target = find_uncovered_in_zone(*zone, g)
    if target is None:
        zr, zc = zone
        target = (zr*ZONE_SIZE+ZONE_SIZE//2, zc*ZONE_SIZE+ZONE_SIZE//2)
    tr, tc = target
    nr, nc = move_toward(r, c, tr, tc)
    if (nr, nc) == (r, c):
        idx += 1
    return nr, nc, idx

# ══════════════════════════════════════════════════════════════
# HEADLESS SIM (Batch-Average)
# ══════════════════════════════════════════════════════════════
class DroneSimHeadless:
    def __init__(self, seed):
        self.seed = seed
        self.zone_rng = random.Random(seed*31 + 17)   
        self.gs, self.ggre, self.ggps, self.gaco = make_grid(seed), make_grid(seed), make_grid(seed), make_grid(seed)
        self.threats = place_threats(seed, [self.gs, self.ggre, self.ggps, self.gaco])

        self.s_active = {"r": 1, "c": 1, "b": 100, "pr": None, "pc": None}
        self.s_returning = []
        self.s_incoming = None
        self.s_handoff_mode = False
        self.sR = 0; self.s_detected = set()
        self.target_zone = None; self.target_cell = None

        self.grer,self.grec,self.greb = 1,1,100
        self.gre_pr=self.gre_pc=None; self.greR=0
        self.gre_detected=set(); self.gre_going=False; self.gre_station=None

        self.gr,self.gc,self.gb = 1,1,100
        self.g_pr, self.g_pc = None, None  
        self.gR=0; self.dr=1; self.gps_going=False; self.gps_station=None
        self.gps_returning=False; self.gps_resume=None
        self.gps_detected=set()   

        self.aco_route = aco_solve_route(get_all_zones(), (1,1), seed=seed)
        self._aco_idx = 0
        self.ar,self.ac,self.ab = 1,1,100
        self.aco_pr,self.aco_pc = None,None
        self.aR=0; self.aco_going=False; self.aco_station=None
        self.aco_returning=False; self.aco_resume=None
        self.aco_detected=set()

        self.step=0
        self.s_first_all=None; self.gre_first_all=None; self.g_first_all=None; self.aco_first_all=None
        self.s_full_cov_step=None

        # --- Total Energy Consumption + Mission Failure Rate trackers ---
        # Energy: running sum of every real step-cost (battery-% units,
        # from owt_step_cost via calculate_actual_step_cost) across the
        # whole run, INCLUDING SMRS's returning/incoming fleet drones --
        # unaffected by recharge resets, so it's a true total-work metric.
        # Stranded: counts steps where owt_wind_cost() returned inf
        # (required airspeed exceeded OWT_CFG.max_airspeed_cap) -- an
        # infeasible-wind event that forces battery to 0 via the existing
        # max(0, b-cost) pattern. A stranding OR failing to find all
        # threats within max_steps both count as "mission failed" (see run()).
        self.s_energy=0.0; self.gre_energy=0.0; self.g_energy=0.0; self.aco_energy=0.0
        self.s_stranded=0; self.gre_stranded=0; self.g_stranded=0; self.aco_stranded=0

        # --- Zone-visit / persistence instrumentation (SMRS grid only) ---
        # zone_step_presence: {(zr,zc): #steps the SMRS drone physically
        #   occupied a cell inside that zone} -- this IS the "zone visit
        #   distribution" (histogram) used to demonstrate/measure the
        #   patrol-starvation problem (e.g. Zone A revisited constantly
        #   while Zone B/C go stale) and to compare it before/after a
        #   persistence-aware scoring change.
        # zone_max_age: {(zr,zc): worst-case steps-since-last-visited ever
        #   reached during this run} -- a running max, updated cheaply
        #   in-place (no growing per-step trace kept), used for the
        #   "max zone age" / staleness metric.
        self.zone_step_presence = {}
        self.zone_max_age = {}

        # Coverage-over-time trace (Contribution 2: "does the proposed
        # scoring reach full coverage FASTER, not just more evenly?").
        # Downsampled (see COVERAGE_CURVE_SAMPLE_EVERY) so a 2000-step run
        # doesn't produce an unwieldy 2000-point trace per seed.
        self.coverage_curve = []

    def _track_energy(self, cost, energy_attr, stranded_attr):
        """Observes a step cost without altering existing movement/battery
        behavior. inf cost (impassable wind step) -> stranded counter;
        otherwise accumulated into the running total-energy counter."""
        if cost == float('inf'):
            setattr(self, stranded_attr, getattr(self, stranded_attr) + 1)
        else:
            setattr(self, energy_attr, getattr(self, energy_attr) + cost)

    def _update_zone_max_age(self):
        """Cheap per-zone staleness snapshot for the SMRS grid (self.gs):
        raw_gap = steps since a zone's cells were last visited/covered
        (same definition zone_info() uses), updated as a running max per
        zone so we don't need to keep a growing per-step trace. This is
        the source for the "max zone age" persistence/starvation metric."""
        for (zr, zc) in get_all_zones():
            cells = get_zone_cells(zr, zc)
            if not cells:
                continue
            ls = min(self.gs[c]["last_visited_step"] for c in cells)
            gap = self.step - ls
            zkey = (zr, zc)
            if gap > self.zone_max_age.get(zkey, 0):
                self.zone_max_age[zkey] = gap

    def do_step(self):
        self.step += 1
        advance_sim_clock(self.step)   # keep dynamic-wind clock in sync (no-op if WIND_DYNAMIC_ENABLED=False)

        # --- SMRS MULTI-DRONE LOGIC ---
        for rd in self.s_returning[:]:
            if (rd["r"], rd["c"]) == rd["st"]:
                self.sR += 1
                self.s_returning.remove(rd)
            else:
                old_r, old_c = rd["r"], rd["c"]
                nr, nc = move_toward(rd["r"], rd["c"], rd["st"][0], rd["st"][1])
                rd["r"], rd["c"] = nr, nc
                cost = calculate_actual_step_cost(old_r, old_c, nr, nc, None, None, current_soc_pct=(rd["b"] if USE_NONLINEAR_BATTERY else None), cycle_count=(self.sR if USE_SOH_AGING else 0))
                self._track_energy(cost, "s_energy", "s_stranded")
                rd["b"] = max(0, rd["b"] - cost)
                sensor_sweep(nr, nc, self.gs, self.s_detected, step=self.step)

        if self.s_handoff_mode and self.s_incoming:
            tr, tc = self.s_active["r"], self.s_active["c"]
            dist_ai = abs(self.s_incoming["r"]-tr) + abs(self.s_incoming["c"]-tc)
            if dist_ai <= 1:
                old = self.s_active.copy()
                old["st"] = nearest_st(tr, tc)
                self.s_returning.append(old)
                self.s_active = self.s_incoming
                self.s_active["pr"] = self.s_active["pc"] = None
                self.s_handoff_mode = False; self.s_incoming = None
                self.target_zone = None; self.target_cell = None
            else:
                old_r, old_c = self.s_incoming["r"], self.s_incoming["c"]
                nr, nc = move_toward(self.s_incoming["r"], self.s_incoming["c"], tr, tc)
                self.s_incoming["r"], self.s_incoming["c"] = nr, nc
                cost = calculate_actual_step_cost(old_r, old_c, nr, nc, None, None, current_soc_pct=(self.s_incoming["b"] if USE_NONLINEAR_BATTERY else None), cycle_count=(self.sR if USE_SOH_AGING else 0))
                self._track_energy(cost, "s_energy", "s_stranded")
                self.s_incoming["b"] = max(0, self.s_incoming["b"] - cost)
                sensor_sweep(nr, nc, self.gs, self.s_detected, step=self.step)

        nh, st = needs_handoff_now(self.s_active["r"], self.s_active["c"], self.s_active["b"])
        if nh and not self.s_handoff_mode:
            self.s_handoff_mode = True
            self.s_incoming = {"r": st[0], "c": st[1], "b": 100}

        mrc, e_ret, st = must_recharge_now(self.s_active["r"], self.s_active["c"], self.s_active["b"])
        if mrc:
            old = self.s_active.copy()
            old["st"] = st
            self.s_returning.append(old)
            if self.s_incoming:
                self.s_active = self.s_incoming
                self.s_active["pr"] = self.s_active["pc"] = None
                self.s_handoff_mode = False; self.s_incoming = None
            else:
                self.s_active = {"r": st[0], "c": st[1], "b": 100, "pr": None, "pc": None}
            self.target_zone = None; self.target_cell = None
        else:
            old_r, old_c = self.s_active["r"], self.s_active["c"]
            pr, pc = self.s_active["pr"], self.s_active["pc"]

            if self.target_zone:
                zr,zc=self.target_zone
                if all(self.gs[cl]["covered"]>=100 for cl in get_zone_cells(zr,zc)):
                    rk=rank_zones(self.s_active["r"],self.s_active["c"],self.gs,self.s_detected,self.step)
                    self.target_zone,self.target_cell=select_zone_mixed_strategy(rk,self.gs,self.s_active["r"],self.s_active["c"],self.s_active["b"],self.zone_rng,self.step)
            if not self.target_zone:
                rk=rank_zones(self.s_active["r"],self.s_active["c"],self.gs,self.s_detected,self.step)
                self.target_zone,self.target_cell=select_zone_mixed_strategy(rk,self.gs,self.s_active["r"],self.s_active["c"],self.s_active["b"],self.zone_rng,self.step)
            
            if self.target_cell:
                tr,tc_=self.target_cell
                can_go, e1, e2, eni = two_hop_check(
                    self.s_active["r"], self.s_active["c"], tr, tc_, self.s_active["b"])
                if can_go:
                    nr, nc = bfs_next_step(self.s_active["r"],self.s_active["c"],tr,tc_,self.gs)
                    self.s_active["r"], self.s_active["c"] = nr, nc
                else:
                    if self.gs[(self.s_active["r"], self.s_active["c"])]["is_station"]:
                        self.s_active["b"] = 100
                        self.target_zone = None; self.target_cell = None
                    else:
                        st_ = nearest_st(self.s_active["r"], self.s_active["c"])
                        nr, nc = bfs_next_step(self.s_active["r"], self.s_active["c"], st_[0], st_[1], self.gs)
                        self.s_active["r"], self.s_active["c"] = nr, nc
                        self.target_zone = None; self.target_cell = None
            else:
                self.s_active["r"],self.s_active["c"],_=smart_move(
                    self.s_active["r"],self.s_active["c"],self.s_active["pr"],self.s_active["pc"],
                    self.gs,self.s_detected,self.target_zone)

            new_r, new_c = self.s_active["r"], self.s_active["c"]
            cell=self.gs[(new_r, new_c)]
            if not cell["is_station"]:
                cost = calculate_actual_step_cost(old_r, old_c, new_r, new_c, pr, pc, current_soc_pct=(self.s_active["b"] if USE_NONLINEAR_BATTERY else None), cycle_count=(self.sR if USE_SOH_AGING else 0))
                self._track_energy(cost, "s_energy", "s_stranded")
                self.s_active["b"]=max(0,self.s_active["b"]-cost)
                sensor_sweep(new_r, new_c, self.gs, self.s_detected, step=self.step)
                _adaptive_learning_step_hook(new_r, new_c, self.gs)
                zkey = (new_r // ZONE_SIZE, new_c // ZONE_SIZE)
                self.zone_step_presence[zkey] = self.zone_step_presence.get(zkey, 0) + 1

            if self.target_zone:
                self.target_cell=find_uncovered_in_zone(*self.target_zone,self.gs)

        self.s_active["pr"],self.s_active["pc"] = self.s_active["r"],self.s_active["c"]
        if self.s_first_all is None and len(self.s_detected) >= NUM_THREATS: self.s_first_all=self.step
        if self.s_full_cov_step is None and coverage_pct(self.gs) >= 100: self.s_full_cov_step=self.step
        self._update_zone_max_age()
        if self.step == 1 or self.step % COVERAGE_CURVE_SAMPLE_EVERY == 0:
            self.coverage_curve.append((self.step, coverage_pct(self.gs)))

        # --- GREEDY ---
        if self.gre_going:
            if (self.grer,self.grec)==self.gre_station:
                self.greb=100; self.greR+=1; self.gre_going=False
            else:
                old_r, old_c = self.grer, self.grec
                nr,nc=move_toward(self.grer,self.grec,*self.gre_station)
                if (nr,nc)==(self.grer,self.grec): nr,nc=self.gre_station
                self.grer,self.grec=nr,nc
                cost = calculate_actual_step_cost(old_r, old_c, nr, nc, self.gre_pr, self.gre_pc, current_soc_pct=(self.greb if USE_NONLINEAR_BATTERY else None), cycle_count=(self.greR if USE_SOH_AGING else 0))
                self._track_energy(cost, "gre_energy", "gre_stranded")
                self.greb=max(0,self.greb-cost)
                sensor_sweep(self.grer, self.grec, self.ggre, self.gre_detected, step=self.step)
        else:
            mrc_gre,e_ret_gre,st_gre = must_recharge_now(self.grer,self.grec,self.greb)
            if mrc_gre:
                self.gre_going=True; self.gre_station=st_gre
                old_r, old_c = self.grer, self.grec
                nr,nc=move_toward(self.grer,self.grec,*st_gre)
                if (nr,nc)==(self.grer,self.grec): nr,nc=st_gre
                self.grer,self.grec=nr,nc
                cost = calculate_actual_step_cost(old_r, old_c, nr, nc, self.gre_pr, self.gre_pc, current_soc_pct=(self.greb if USE_NONLINEAR_BATTERY else None), cycle_count=(self.greR if USE_SOH_AGING else 0))
                self._track_energy(cost, "gre_energy", "gre_stranded")
                self.greb=max(0,self.greb-cost)
                sensor_sweep(self.grer, self.grec, self.ggre, self.gre_detected, step=self.step)
            else:
                old_r, old_c = self.grer, self.grec
                self.grer,self.grec=greedy_move(self.grer,self.grec,self.gre_pr,self.gre_pc,self.ggre,self.gre_detected)
                cost = calculate_actual_step_cost(old_r, old_c, self.grer, self.grec, self.gre_pr, self.gre_pc, current_soc_pct=(self.greb if USE_NONLINEAR_BATTERY else None), cycle_count=(self.greR if USE_SOH_AGING else 0))
                self._track_energy(cost, "gre_energy", "gre_stranded")
                self.greb=max(0,self.greb-cost)
                sensor_sweep(self.grer, self.grec, self.ggre, self.gre_detected, step=self.step)
        self.gre_pr,self.gre_pc=self.grer,self.grec
        if self.gre_first_all is None and len(self.gre_detected) >= NUM_THREATS: self.gre_first_all=self.step

        # --- GPS ---
        if self.gps_going:
            if (self.gr,self.gc)==self.gps_station:
                self.gb=100; self.gR+=1; self.gps_going=False
                if self.gps_resume and (self.gr,self.gc)!=self.gps_resume[:2]: self.gps_returning=True
                elif self.gps_resume: self.dr=self.gps_resume[2]; self.gps_resume=None
            else:
                old_r, old_c = self.gr, self.gc
                nr,nc=move_toward(self.gr,self.gc,*self.gps_station)
                if (nr,nc)==(self.gr,self.gc): nr,nc=self.gps_station
                self.gr,self.gc=nr,nc
                cost = calculate_actual_step_cost(old_r, old_c, nr, nc, self.g_pr, self.g_pc, current_soc_pct=(self.gb if USE_NONLINEAR_BATTERY else None), cycle_count=(self.gR if USE_SOH_AGING else 0))
                self._track_energy(cost, "g_energy", "g_stranded")
                self.gb=max(0,self.gb-cost)
                sensor_sweep(self.gr, self.gc, self.ggps, self.gps_detected, step=self.step)
        elif self.gps_returning:
            tr,tc_=self.gps_resume[:2]
            if (self.gr,self.gc)==(tr,tc_): self.dr=self.gps_resume[2]; self.gps_returning=False; self.gps_resume=None
            else:
                old_r, old_c = self.gr, self.gc
                nr,nc=move_toward(self.gr,self.gc,tr,tc_)
                if (nr,nc)==(self.gr,self.gc): nr,nc=(tr,tc_)
                self.gr,self.gc=nr,nc
                cost = calculate_actual_step_cost(old_r, old_c, nr, nc, self.g_pr, self.g_pc, current_soc_pct=(self.gb if USE_NONLINEAR_BATTERY else None), cycle_count=(self.gR if USE_SOH_AGING else 0))
                self._track_energy(cost, "g_energy", "g_stranded")
                self.gb=max(0,self.gb-cost)
                sensor_sweep(self.gr, self.gc, self.ggps, self.gps_detected, step=self.step)
        else:
            gps_min_bat=(min(abs(self.gr-s[0])+abs(self.gc-s[1]) for s in STATIONS)+1)*ENERGY_PER_CELL*SAFETY_BUFFER
            if self.gb <= max(15, gps_min_bat):
                self.gps_resume=(self.gr,self.gc,self.dr)
                self.gps_going=True; self.gps_station=nearest_st(self.gr,self.gc)
                old_r, old_c = self.gr, self.gc
                nr,nc=move_toward(self.gr,self.gc,*self.gps_station)
                if (nr,nc)==(self.gr,self.gc): nr,nc=self.gps_station
                self.gr,self.gc=nr,nc
                cost = calculate_actual_step_cost(old_r, old_c, nr, nc, self.g_pr, self.g_pc, current_soc_pct=(self.gb if USE_NONLINEAR_BATTERY else None), cycle_count=(self.gR if USE_SOH_AGING else 0))
                self._track_energy(cost, "g_energy", "g_stranded")
                self.gb=max(0,self.gb-cost)
                sensor_sweep(self.gr, self.gc, self.ggps, self.gps_detected, step=self.step)
            else:
                old_r, old_c = self.gr, self.gc
                self.gr,self.gc,self.dr=gps_snake(self.gr,self.gc,self.dr)
                cost = calculate_actual_step_cost(old_r, old_c, self.gr, self.gc, self.g_pr, self.g_pc, current_soc_pct=(self.gb if USE_NONLINEAR_BATTERY else None), cycle_count=(self.gR if USE_SOH_AGING else 0))
                self._track_energy(cost, "g_energy", "g_stranded")
                self.gb=max(0,self.gb-cost)
                sensor_sweep(self.gr, self.gc, self.ggps, self.gps_detected, step=self.step)

        self.g_pr, self.g_pc = self.gr, self.gc
        if self.g_first_all is None and len(self.gps_detected) >= NUM_THREATS: self.g_first_all=self.step

        # --- ACO (4th policy) ---
        if self.aco_going:
            if (self.ar,self.ac)==self.aco_station:
                self.ab=100; self.aR+=1; self.aco_going=False
                if self.aco_resume and (self.ar,self.ac)!=self.aco_resume[:2]: self.aco_returning=True
                elif self.aco_resume: self._aco_idx=self.aco_resume[2]; self.aco_resume=None
            else:
                old_r, old_c = self.ar, self.ac
                nr,nc=move_toward(self.ar,self.ac,*self.aco_station)
                if (nr,nc)==(self.ar,self.ac): nr,nc=self.aco_station
                self.ar,self.ac=nr,nc
                cost = calculate_actual_step_cost(old_r, old_c, nr, nc, self.aco_pr, self.aco_pc, current_soc_pct=(self.ab if USE_NONLINEAR_BATTERY else None), cycle_count=(self.aR if USE_SOH_AGING else 0))
                self._track_energy(cost, "aco_energy", "aco_stranded")
                self.ab=max(0,self.ab-cost)
                sensor_sweep(self.ar, self.ac, self.gaco, self.aco_detected, step=self.step)
        elif self.aco_returning:
            tr,tc_=self.aco_resume[:2]
            if (self.ar,self.ac)==(tr,tc_): self._aco_idx=self.aco_resume[2]; self.aco_returning=False; self.aco_resume=None
            else:
                old_r, old_c = self.ar, self.ac
                nr,nc=move_toward(self.ar,self.ac,tr,tc_)
                if (nr,nc)==(self.ar,self.ac): nr,nc=(tr,tc_)
                self.ar,self.ac=nr,nc
                cost = calculate_actual_step_cost(old_r, old_c, nr, nc, self.aco_pr, self.aco_pc, current_soc_pct=(self.ab if USE_NONLINEAR_BATTERY else None), cycle_count=(self.aR if USE_SOH_AGING else 0))
                self._track_energy(cost, "aco_energy", "aco_stranded")
                self.ab=max(0,self.ab-cost)
                sensor_sweep(self.ar, self.ac, self.gaco, self.aco_detected, step=self.step)
        else:
            aco_min_bat=(min(abs(self.ar-s[0])+abs(self.ac-s[1]) for s in STATIONS)+1)*ENERGY_PER_CELL*SAFETY_BUFFER
            if self.ab <= max(15, aco_min_bat):
                self.aco_resume=(self.ar,self.ac,self._aco_idx)
                self.aco_going=True; self.aco_station=nearest_st(self.ar,self.ac)
                old_r, old_c = self.ar, self.ac
                nr,nc=move_toward(self.ar,self.ac,*self.aco_station)
                if (nr,nc)==(self.ar,self.ac): nr,nc=self.aco_station
                self.ar,self.ac=nr,nc
                cost = calculate_actual_step_cost(old_r, old_c, nr, nc, self.aco_pr, self.aco_pc, current_soc_pct=(self.ab if USE_NONLINEAR_BATTERY else None), cycle_count=(self.aR if USE_SOH_AGING else 0))
                self._track_energy(cost, "aco_energy", "aco_stranded")
                self.ab=max(0,self.ab-cost)
                sensor_sweep(self.ar, self.ac, self.gaco, self.aco_detected, step=self.step)
            else:
                old_r, old_c = self.ar, self.ac
                self.ar,self.ac,self._aco_idx=aco_move(self.ar,self.ac,self.aco_route,self._aco_idx,self.gaco)
                cost = calculate_actual_step_cost(old_r, old_c, self.ar, self.ac, self.aco_pr, self.aco_pc, current_soc_pct=(self.ab if USE_NONLINEAR_BATTERY else None), cycle_count=(self.aR if USE_SOH_AGING else 0))
                self._track_energy(cost, "aco_energy", "aco_stranded")
                self.ab=max(0,self.ab-cost)
                sensor_sweep(self.ar, self.ac, self.gaco, self.aco_detected, step=self.step)

        self.aco_pr, self.aco_pc = self.ar, self.ac
        if self.aco_first_all is None and len(self.aco_detected) >= NUM_THREATS: self.aco_first_all=self.step

    def run(self, max_steps=MAX_STEPS):
        for _ in range(max_steps):
            self.do_step()
            if (self.s_first_all is not None and self.gre_first_all is not None
                    and self.g_first_all is not None and self.aco_first_all is not None): break
        # Mission failure := stranded at least once (inf-cost/impassable-wind
        # event forced battery to 0) OR failed to find all threats within
        # max_steps. Two independent, honest failure modes -- neither one
        # is defined in terms of "coverage %" so this can't just relabel
        # the same coverage number under a new name.
        return {"seed": self.seed, "s": self.s_first_all, "gre": self.gre_first_all,
                "g": self.g_first_all, "aco": self.aco_first_all,
                "s_full_cov": self.s_full_cov_step,
                "s_energy": round(self.s_energy, 1), "gre_energy": round(self.gre_energy, 1),
                "g_energy": round(self.g_energy, 1), "aco_energy": round(self.aco_energy, 1),
                "s_stranded": self.s_stranded, "gre_stranded": self.gre_stranded,
                "g_stranded": self.g_stranded, "aco_stranded": self.aco_stranded,
                "s_failed": bool(self.s_stranded > 0 or self.s_first_all is None),
                "gre_failed": bool(self.gre_stranded > 0 or self.gre_first_all is None),
                "g_failed": bool(self.g_stranded > 0 or self.g_first_all is None),
                "aco_failed": bool(self.aco_stranded > 0 or self.aco_first_all is None),
                # Persistence/staleness instrumentation for the SMRS (policy-
                # under-test) grid -- see _update_zone_max_age() and the
                # sensor_sweep hook in do_step() for how these are built.
                "zone_step_presence": dict(self.zone_step_presence),
                "zone_max_age": dict(self.zone_max_age),
                "coverage_curve": list(self.coverage_curve)}

def fmt_mean_std(values):
    clean=[v for v in values if v is not None]
    if not clean: return "None ± None"
    mean=statistics.mean(clean); std=statistics.pstdev(clean) if len(clean)>1 else 0.0
    if len(clean)<len(values): return f"{mean:.1f} ± {std:.2f}  (partial: {len(clean)}/{len(values)} seeds)"
    return f"{mean:.1f} ± {std:.2f}"

# ══════════════════════════════════════════════════════════════
# STATISTICAL SIGNIFICANCE (paired t-test + Wilcoxon signed-rank)
#
# আগে শুধু mean±std রিপোর্ট হতো — ১০টা random seed-এর পার্থক্য noise
# না signal সেটার কোনো প্রমাণ ছিল না। এখন প্রতিটা seed-এ SMRS আর অন্য
# policy একই grid/threat-layout-এ (একই seed) চলে বলে এটা একটা natural
# PAIRED design — তাই paired t-test (normality ধরে) আর Wilcoxon
# signed-rank test (distribution-free, right-skewed detection-time
# ডেটার জন্য বেশি নির্ভরযোগ্য) — দুটোই রিপোর্ট করা হচ্ছে।
#
# শুধু সেই seed-জোড়া ব্যবহার হয় যেখানে দুটো policy-ই সম্পন্ন হয়েছে
# (কোনোটা None হলে paired test-এ ব্যবহারযোগ্য না) — কতগুলো জোড়া আসলে
# ব্যবহৃত হয়েছে সেটাও স্পষ্টভাবে রিপোর্ট করা হয়, যাতে partial-completion
# থাকলে সেটা লুকানো না থাকে।
# ══════════════════════════════════════════════════════════════
def paired_significance_test(baseline_vals, other_vals):
    """baseline_vals (SMRS) বনাম other_vals (Greedy/GPS/ACO)-এর same-seed
    paired তুলনা। রিটার্ন করে dict: n (ব্যবহৃত pair সংখ্যা), mean_diff
    (other-baseline, ঋণাত্মক মানে SMRS দ্রুত), t_p (paired t-test p-value),
    w_p (Wilcoxon signed-rank p-value)।"""
    pairs = [(b, o) for b, o in zip(baseline_vals, other_vals) if b is not None and o is not None]
    n = len(pairs)
    if n < 2:
        return {"n": n, "t_p": None, "w_p": None, "mean_diff": None}

    b_arr = [p[0] for p in pairs]
    o_arr = [p[1] for p in pairs]
    mean_diff = statistics.mean(o_arr) - statistics.mean(b_arr)

    # সব paired difference ঠিক শূন্য হলে (দুই মডেল প্রতিটা seed-এ একই মান
    # দিয়েছে) t-test/wilcoxon উভয়েরই variance শূন্য হয়ে যায় -- scipy তখন
    # exception ছোঁড়ে না, বরং চুপচাপ NaN রিটার্ন করে (RuntimeWarning সহ),
    # তাই নিচের except ব্লক এটা ধরতে পারে না। এই কেসে সঠিক উত্তর হলো
    # p=1.0 (কোনো পার্থক্য পাওয়া যায়নি, নিশ্চিতভাবে), NaN না।
    diffs = [o - b for o, b in zip(o_arr, b_arr)]
    zero_diff = all(abs(d) < 1e-12 for d in diffs)

    if zero_diff:
        t_p = 1.0
    else:
        try:
            from scipy.stats import ttest_rel
            _, t_p = ttest_rel(o_arr, b_arr)
        except Exception:
            t_p = None

    if zero_diff:
        w_p = 1.0
    else:
        try:
            _, w_p = wilcoxon(o_arr, b_arr)
        except Exception:
            w_p = None   # e.g. খুব কম non-zero difference থাকলে wilcoxon ব্যর্থ হতে পারে

    return {"n": n, "t_p": t_p, "w_p": w_p, "mean_diff": mean_diff}

def format_significance_block(baseline_name, baseline_vals, others, alpha=0.05):
    """others = {"Greedy": vals, "GPS": vals, ...} — SMRS-এর সাথে প্রতিটার
    paired significance test রিপোর্ট করে, readable লাইন হিসেবে।"""
    lines = [f"STATISTICAL SIGNIFICANCE ({baseline_name} বনাম প্রতিটা baseline, paired by seed)", "-"*66]
    for name, vals in others.items():
        res = paired_significance_test(baseline_vals, vals)
        if res["n"] < 2:
            lines.append(f"  {baseline_name} vs {name:8s}: paired তুলনার জন্য যথেষ্ট common-completed seed নেই (n={res['n']})")
            continue
        t_p, w_p = res["t_p"], res["w_p"]
        t_str = f"{t_p:.4g}" if t_p is not None else "N/A"
        w_str = f"{w_p:.4g}" if w_p is not None else "N/A"
        t_sig = "✅" if (t_p is not None and t_p < alpha) else "❌"
        w_sig = "✅" if (w_p is not None and w_p < alpha) else "❌"
        lines.append(
            f"  {baseline_name} vs {name:8s}: n={res['n']:2d} pairs | mean Δ={res['mean_diff']:+8.1f} steps | "
            f"paired t-test p={t_str} {t_sig} | Wilcoxon p={w_str} {w_sig}  (α={alpha})"
        )
    lines.append("  (✅ = p < α, অর্থাৎ পার্থক্যটা পরিসংখ্যানগতভাবে তাৎপর্যপূর্ণ, শুধু random noise না)")
    return "\n".join(lines)

def run_batch_and_format(seeds=BATCH_SEEDS, max_steps=MAX_STEPS):
    results=[DroneSimHeadless(s).run(max_steps) for s in seeds]
    s_vals=[r["s"] for r in results]; gre_vals=[r["gre"] for r in results]
    g_vals=[r["g"] for r in results]; aco_vals=[r["aco"] for r in results]
    lines = [f"FIRST ALL THREATS DETECTION STEP (lower is better)  —  {len(seeds)} seeds", "-"*52]
    lines.append(f"SMRS   : {fmt_mean_std(s_vals)}")
    lines.append(f"Greedy : {fmt_mean_std(gre_vals)}")
    lines.append(f"GPS    : {fmt_mean_std(g_vals)}")
    lines.append(f"ACO    : {fmt_mean_std(aco_vals)}")
    lines.append("-" * 52)
    lines.append(format_significance_block("SMRS", s_vals, {"Greedy": gre_vals, "GPS": g_vals, "ACO": aco_vals}))
    lines.append("-" * 52); lines.append("Per seed values:")
    for r in results:
        lines.append(f"Seed {r['seed']} | SMRS={r['s'] if r['s'] else 'None'} | Greedy={r['gre'] if r['gre'] else 'None'} | GPS={r['g'] if r['g'] else 'None'} | ACO={r['aco'] if r['aco'] else 'None'}")
    return "\n".join(lines)

# ══════════════════════════════════════════════════════════════
# TOTAL ENERGY CONSUMPTION + MISSION FAILURE RATE
#
# Energy: প্রতি seed-এ প্রতি policy-র পুরো run জুড়ে সব step-cost-এর যোগফল
# (battery-% একক, recharge reset-এ প্রভাবিত হয় না) — কম মানে বেশি
# energy-efficient। Coverage/detection-rate-এর independent dimension।
#
# Mission Failure: দুটো স্বতন্ত্র, honest failure mode-এর OR —
#   (１) stranded: owt_wind_cost()-এ required airspeed hardware cap
#       ছাড়িয়ে গিয়ে battery জোর করে 0 হওয়া (আসল physics-driven ঘটনা)
#   (２) detection failure: max_steps-এর মধ্যে সব threat না পাওয়া
# কোনোটাই coverage %-এর পুনঃলেবেল না — আলাদা, স্বাধীনভাবে পরিমাপযোগ্য ঘটনা।
#
# Failure একটা binary (0/1) outcome বলে wilcoxon/ttest_rel (continuous-data
# assumption) এখানে statistically অবৈধ — paired binary outcome-এর সঠিক
# test হলো McNemar's exact test (discordant pairs-এর উপর binomial test)।
# ══════════════════════════════════════════════════════════════

def _greedy_move_wrapper(r, c, pr, pc, g, detected, mstate):
    nr, nc = greedy_move(r, c, pr, pc, g, detected)
    return nr, nc, mstate

def _gps_move_wrapper(r, c, pr, pc, g, detected, mstate):
    nr, nc, nd = gps_snake(r, c, mstate)
    return nr, nc, nd

def _make_aco_move_wrapper(route):
    def wrapper(r, c, pr, pc, g, detected, mstate):
        nr, nc, nidx = aco_move(r, c, route, mstate, g)
        return nr, nc, nidx
    return wrapper

def _advance_relay_policy(state, g, detected, move_fn, step):
    """Advances ONE policy's relay/handoff fleet by exactly one time step.

    `state` is a dict: {"active": {r,c,b,pr,pc}, "returning": [...],
    "incoming": {...}|None, "handoff_mode": bool, "RC": int, "mstate": any}.
    `move_fn(r,c,pr,pc,g,detected,mstate) -> (nr,nc,mstate)` is the
    underlying single-drone movement policy (greedy_move/gps_snake/aco_move,
    wrapped — see _greedy_move_wrapper/_gps_move_wrapper/_make_aco_move_wrapper).

    This is the exact relay/handoff logic SMRS already uses (backup drone
    launches from the nearest station when needs_handoff_now() fires, meets
    the active drone, they swap — active retreats to recharge instead of
    coverage simply pausing) — refactored out so it can drive ANY policy
    (Greedy/GPS-Snake/ACO), not just SMRS. Used by both
    simulate_policy_with_relay() (headless batch loop below) and
    DroneSimGUI.do_step() (live per-frame simulation), so the two stay
    numerically identical.
    """
    active = state["active"]; returning = state["returning"]
    incoming = state["incoming"]; handoff_mode = state["handoff_mode"]

    for rd in returning[:]:
        if (rd["r"], rd["c"]) == rd["st"]:
            state["RC"] += 1; returning.remove(rd)
        else:
            old_r, old_c = rd["r"], rd["c"]
            nr, nc = move_toward(rd["r"], rd["c"], rd["st"][0], rd["st"][1])
            rd["r"], rd["c"] = nr, nc
            cost = calculate_actual_step_cost(old_r, old_c, nr, nc, None, None,
                                               current_soc_pct=(rd["b"] if USE_NONLINEAR_BATTERY else None),
                                               cycle_count=(state["RC"] if USE_SOH_AGING else 0))
            rd["b"] = max(0, rd["b"] - cost)
            sensor_sweep(nr, nc, g, detected, step=step)

    if handoff_mode and incoming:
        tr, tc = active["r"], active["c"]
        dist_ai = abs(incoming["r"]-tr) + abs(incoming["c"]-tc)
        if dist_ai <= 1:
            old = active.copy(); old["st"] = nearest_st(tr, tc)
            returning.append(old)
            active = incoming
            active["pr"] = active["pc"] = None
            handoff_mode = False; incoming = None
        else:
            old_r, old_c = incoming["r"], incoming["c"]
            nr, nc = move_toward(incoming["r"], incoming["c"], tr, tc)
            incoming["r"], incoming["c"] = nr, nc
            cost = calculate_actual_step_cost(old_r, old_c, nr, nc, None, None,
                                               current_soc_pct=(incoming["b"] if USE_NONLINEAR_BATTERY else None),
                                               cycle_count=(state["RC"] if USE_SOH_AGING else 0))
            incoming["b"] = max(0, incoming["b"] - cost)
            sensor_sweep(nr, nc, g, detected, step=step)

    nh, st = needs_handoff_now(active["r"], active["c"], active["b"])
    if nh and not handoff_mode:
        handoff_mode = True
        incoming = {"r": st[0], "c": st[1], "b": 100}

    mrc, e_ret, st2 = must_recharge_now(active["r"], active["c"], active["b"])
    if mrc:
        if incoming:
            # backup ইতিমধ্যে পথে আছে — active এখানেই hover করে অপেক্ষা করে
            # (position জোর করে backup-এর মাঝপথের অবস্থানে না সরিয়ে), যাতে
            # rendezvous (dist<=1) বাস্তবে ঘটার পরই handoff হয় — এতে
            # GPS-Snake/Greedy-এর মতো position-নির্ভর policy-র continuity
            # ভেঙে অসীম লুপে আটকে যায় না।
            cost = calculate_actual_step_cost(active["r"], active["c"], active["r"], active["c"], active["pr"], active["pc"],
                                               cycle_count=(state["RC"] if USE_SOH_AGING else 0))
            active["b"] = max(0, active["b"] - cost)
            sensor_sweep(active["r"], active["c"], g, detected, step=step)
        else:
            # backup কখনো launch হয়নি (rare edge-case) — teleport না করে
            # ধাপে-ধাপে physically station-এর দিকে move করে, তারপর recharge
            old_r, old_c = active["r"], active["c"]
            nr, nc = move_toward(active["r"], active["c"], *st2)
            if (nr, nc) == (active["r"], active["c"]): nr, nc = st2
            active["r"], active["c"] = nr, nc
            cost = calculate_actual_step_cost(old_r, old_c, nr, nc, active["pr"], active["pc"],
                                               cycle_count=(state["RC"] if USE_SOH_AGING else 0))
            active["b"] = max(0, active["b"] - cost)
            sensor_sweep(nr, nc, g, detected, step=step)
            if (active["r"], active["c"]) == st2:
                active["b"] = 100; state["RC"] += 1
    else:
        old_r, old_c = active["r"], active["c"]
        nr, nc, state["mstate"] = move_fn(active["r"], active["c"], active["pr"], active["pc"], g, detected, state["mstate"])
        active["r"], active["c"] = nr, nc
        cost = calculate_actual_step_cost(old_r, old_c, nr, nc, active["pr"], active["pc"],
                                           cycle_count=(state["RC"] if USE_SOH_AGING else 0))
        active["b"] = max(0, active["b"] - cost)
        sensor_sweep(nr, nc, g, detected, step=step)

    active["pr"], active["pc"] = active["r"], active["c"]
    state["active"] = active; state["returning"] = returning
    state["incoming"] = incoming; state["handoff_mode"] = handoff_mode
    return state


def simulate_policy_with_relay(seed, move_fn, mstate_init, max_steps=MAX_STEPS):
    """যেকোনো single-step move_fn(r,c,pr,pc,g,detected,mstate)->(nr,nc,mstate)
    কে SMRS-এর মতোই multi-drone relay/handoff ক্ষমতা দিয়ে চালায় — অর্থাৎ
    resource (drone/relay) সবার জন্য সমান, শুধু movement-policy আলাদা।"""
    g = make_grid(seed)
    place_threats(seed, [g])

    state = {
        "active": {"r": 1, "c": 1, "b": 100, "pr": None, "pc": None},
        "returning": [], "incoming": None, "handoff_mode": False,
        "RC": 0, "mstate": mstate_init,
    }
    detected = set()
    first_all = None
    full_cov_step = None

    for step in range(1, max_steps+1):
        state = _advance_relay_policy(state, g, detected, move_fn, step)

        if first_all is None and len(detected) >= NUM_THREATS: first_all = step
        if full_cov_step is None and coverage_pct(g) >= 100: full_cov_step = step
        if first_all is not None and full_cov_step is not None: break

    return {"seed": seed, "first_all": first_all, "full_cov": full_cov_step, "RC": state["RC"]}

def run_fair_batch_and_format(seeds=BATCH_SEEDS, max_steps=MAX_STEPS):
    """Matched-resource comparison: SMRS বনাম Greedy+Relay বনাম GPS+Relay
    বনাম ACO+Relay — সবাই একই multi-drone relay সুবিধা পাচ্ছে, শুধু
    target-selection logic আলাদা। এটাই প্রমাণ করে SMRS-এর সুবিধা আসলে
    কি বেশি ড্রোন থেকে না scheduling logic থেকে আসছে।"""
    smrs_results = [DroneSimHeadless(s).run(max_steps) for s in seeds]
    smrs_vals = [r["s"] for r in smrs_results]

    gre_vals, gps_vals, aco_vals = [], [], []
    gre_RC, gps_RC, aco_RC = [], [], []
    for s in seeds:
        r_gre = simulate_policy_with_relay(s, _greedy_move_wrapper, None, max_steps)
        r_gps = simulate_policy_with_relay(s, _gps_move_wrapper, 1, max_steps)
        route = aco_solve_route(get_all_zones(), (1,1), seed=s)
        r_aco = simulate_policy_with_relay(s, _make_aco_move_wrapper(route), 0, max_steps)
        gre_vals.append(r_gre["first_all"]); gre_RC.append(r_gre["RC"])
        gps_vals.append(r_gps["first_all"]); gps_RC.append(r_gps["RC"])
        aco_vals.append(r_aco["first_all"]); aco_RC.append(r_aco["RC"])

    lines = []
    lines.append(f"FAIR / MATCHED-RESOURCE COMPARISON  (সবাইকে multi-drone relay সুবিধা দিয়ে)  —  {len(seeds)} seeds")
    lines.append("=" * 66)
    lines.append("FIRST ALL THREATS DETECTION STEP (lower is better)")
    lines.append("-" * 66)
    lines.append(f"SMRS (zone-tier + mixed-strategy) : {fmt_mean_std(smrs_vals)}")
    lines.append(f"Greedy + Relay                    : {fmt_mean_std(gre_vals)}")
    lines.append(f"GPS-Snake + Relay                 : {fmt_mean_std(gps_vals)}")
    lines.append(f"ACO + Relay                       : {fmt_mean_std(aco_vals)}")
    lines.append("-" * 66)
    lines.append(format_significance_block("SMRS", smrs_vals,
                  {"Greedy+Relay": gre_vals, "GPS+Relay": gps_vals, "ACO+Relay": aco_vals}))
    lines.append("-" * 66)
    lines.append(f"গড় recharge count -> Greedy+Relay: {statistics.mean(gre_RC):.1f} | "
                 f"GPS+Relay: {statistics.mean(gps_RC):.1f} | ACO+Relay: {statistics.mean(aco_RC):.1f}")
    lines.append("-" * 66)
    lines.append("এখানে resource (drone-সংখ্যা/relay) সবার জন্য সমান — তাই SMRS")
    lines.append("এখনো ভালো ফলাফল দিলে সেটা প্রমাণ করে যে এর zone-tier +")
    lines.append("mixed-strategy scheduling logic-ই আসল উন্নতির কারণ, শুধু বেশি")
    lines.append("ড্রোন থাকাটা না — এটাই Q1/Q2-level rigor-এর জন্য জরুরি প্রমাণ।")
    lines.append("উপরের paired significance test সেই দাবিটাকে সংখ্যাভিত্তিক ভিত্তি দেয়।")
    return "\n".join(lines)


TUNABLE_WEIGHT_NAMES = [
    "ALPHA","GAMMA","EPSILON","ETA","LAMBDA","MU","K_TURN",
    "VISIT_PENALTY","BACKTRACK_PENALTY",
    "HY_T1_GAP_W","HY_T1_DIST_W",
    "HY_T2_INCOMPL_W","HY_T2_GAP_W","HY_T2_DIST_W","HY_T2_BORDER_W",
    "HY_T3_BORDER_W","HY_T3_DIST_W",
]

def init_q_weights():
    """Starting weights = the CURRENT hand-tuned/auto-tuned HY_* values,
    so get_q_value() reproduces score_t2() exactly before any TD-update
    ever runs. That exact match is the sanity check for this module --
    """
    return {
        "incompl":      HY_T2_INCOMPL_W,
        "tgap":         HY_T2_GAP_W,
        "border_pr":    HY_T2_BORDER_W,
        "sse_cov":      HY_SSE_W,
        "learned_risk": HY_ADAPT_W,
        "travel":       -HY_T2_DIST_W,   # stored negative so Q is a pure dot
                                          # product, matching the "- ...*travel"
                                          # term in score_t2() with no special-casing
    }

# The live weight vector score_t2() actually reads when USE_Q_LEARNING is
# True. Starts at the hand-tuned point; only an external TD-update loop
# should ever mutate this in place.
Q_WEIGHTS = init_q_weights()

def extract_q_features(z, info, sse_cov):
    """Raw phi(s,a) values for one candidate zone -- pure data, no
    weights. info comes from zone_info(); sse_cov from get_sse_zone_coverage()."""
    return {
        "incompl":      info["incompl"],
        "tgap":         info["tgap"],
        "border_pr":    info["border_pr"],
        "sse_cov":      sse_cov.get(z, 0.0),
        "learned_risk": (LEARNED_ZONE_RISK.get(z, ADAPT_PRIOR) if USE_ADAPTIVE_LEARNING else 0.0),
        "travel":       info["travel"],
    }

def get_q_value(z, info, weights, sse_cov):
    """Q(s,a) = w . phi(s,a) -- linear function approximation.
    Intended to eventually replace score_t2()/score_t3() once validated;
    until then, call this alongside them for comparison only -- never
    wire it into an actual patrol decision yet."""
    phi = extract_q_features(z, info, sse_cov)
    return sum(weights[name] * phi[name] for name in Q_FEATURE_NAMES)

def solve_sse(Ud_c, Ud_u, Ua_c, Ua_u, K):
    """n-target Stackelberg Security Game সলভ করে SSE বের করে।
    Ud_c/Ud_u: defender payoff (covered/uncovered অবস্থায় attacked)
    Ua_c/Ua_u: attacker payoff (একই দুই অবস্থায়)
    K        : sum(c_i) <= K রিসোর্স constraint
    """
    n = len(Ud_c)
    best_val, best_c, best_t, best_aval = -float('inf'), None, None, None

    for t in range(n):
        obj = np.zeros(n)
        obj[t] = Ud_c[t] - Ud_u[t]
        const = Ud_u[t]

        A_ub, b_ub = [], []
        A_ub.append(np.ones(n)); b_ub.append(K)   # resource constraint

        for i in range(n):                         # attacker IC constraints
            if i == t: continue
            row = np.zeros(n)
            row[t] = -(Ua_c[t] - Ua_u[t])
            row[i] = (Ua_c[i] - Ua_u[i])
            A_ub.append(row)
            b_ub.append(Ua_u[t] - Ua_u[i])

        bounds = [(0, 1)] * n
        res = linprog(c=-obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
        if res.success:
            def_val = -res.fun + const
            if def_val > best_val:
                c = res.x
                a_val = Ua_u[t] + c[t] * (Ua_c[t] - Ua_u[t])
                best_val, best_c, best_t, best_aval = def_val, c.copy(), t, a_val

    if best_c is None:
        raise RuntimeError("কোনো candidate target-এর জন্য feasible LP পাওয়া যায়নি।")
    return {"attacked_target": best_t, "coverage": best_c,
            "defender_value": best_val, "attacker_value": best_aval}


def zone_border_priority(zr, zc, g):
    """একটা zone-এর গড় border-priority (0..1 স্কেলে), আপনার
    priority ফিল্ড (0/1/2/3) থেকেই নেওয়া — zone_info()-এর border_pr
    এর সাথে সামঞ্জস্যপূর্ণ, শুধু dr/dc/step এর ওপর নির্ভর করে না।"""
    cells = get_zone_cells(zr, zc)
    if not cells: return 0.0
    return (sum(g[c]["priority"] for c in cells) / len(cells)) / 3.0


def build_zone_payoffs(g, breach_penalty=10.0, capture_reward=2.0, capture_penalty=-5.0):
    """সব zone-এর জন্য payoff টেবিল বানায়, border-priority অনুযায়ী স্কেল
    করে। এই constant গুলো (breach_penalty ইত্যাদি) domain-assumption —
    থিসিসে এগুলো justify করতে হবে (real incident severity/expert weight)।"""
    zones = get_all_zones()
    pr = [zone_border_priority(zr, zc, g) for (zr, zc) in zones]
    Ud_u = [-breach_penalty * p for p in pr]
    Ud_c = [capture_reward] * len(zones)
    Ua_u = [breach_penalty * p for p in pr]
    Ua_c = [capture_penalty] * len(zones)
    return zones, Ud_c, Ud_u, Ua_c, Ua_u


def observe_zone_coverage_profile(seed, steps):
    """SMRS প্রকৃতপক্ষে
    দীর্ঘমেয়াদে প্রতিটা zone কতটা সময় কভার রাখে তার empirical frequency।"""
    sim = DroneSimHeadless(seed)
    zones = get_all_zones()
    hits = {z: 0 for z in zones}
    for _ in range(steps):
        sim.do_step()
        t = sim.step
        for z in zones:
            cells = get_zone_cells(*z)
            if any(sim.gs[c]["last_visited_step"] == t for c in cells):
                hits[z] += 1
    return {z: hits[z] / steps for z in zones}


_CACHED_ZONE_ENERGY_HAT = None   # {(zr,zc): float in [0,1]}, cached until reset

def reset_energy_aware_sse_cache():
    """Clears ONLY the Ê_z (zone energy-hat) cache. Called automatically
    by reset_sse_zone_coverage_cache() below -- kept as its own function
    so a caller that only needs to invalidate Ê_z (e.g. after toggling
    TERRAIN_COST_ENABLED without touching the SSE prior itself) can do
    so without recalibrating the whole SSE game."""
    global _CACHED_ZONE_ENERGY_HAT
    _CACHED_ZONE_ENERGY_HAT = None


def get_zone_energy_hat():
    """Ê_z ∈ [0,1] for every zone: the REPRESENTATIVE energy cost to reach
    that zone, normalized (min-max) across all 36 zones so it's on the
    same [0,1] scale as sse_cov (solve_sse()'s coverage probabilities),
    which is what makes `SSE_z − λ·Ê_z` a well-posed subtraction instead
    of mixing incompatible units.

    "Representative cost" = MIN over the 4 corner STATIONS of
    energy_to_travel(station -> zone centroid) -- i.e. the cost from
    whichever station is closest/cheapest to dispatch a drone from,
    mirroring how a real drone would actually be launched toward that
    zone (not an arbitrary average over all 4, which would overstate the
    true cost for zones near a station).

    Reads OWT-A*'s energy_to_travel(), which itself respects
    TERRAIN_COST_ENABLED (flat vs real elevation) -- so Ê_z automatically
    reflects whichever physics mode is currently active. Cached until
    reset_energy_aware_sse_cache() / reset_sse_zone_coverage_cache() is
    called (needed whenever TERRAIN_COST_ENABLED changes, exactly like
    the SSE coverage cache -- see that reset function's docstring)."""
    global _CACHED_ZONE_ENERGY_HAT
    if _CACHED_ZONE_ENERGY_HAT is not None:
        return _CACHED_ZONE_ENERGY_HAT

    zones = get_all_zones()
    raw = {}
    for (zr, zc) in zones:
        zrc = zr * ZONE_SIZE + ZONE_SIZE // 2
        zcc = zc * ZONE_SIZE + ZONE_SIZE // 2
        raw[(zr, zc)] = min(energy_to_travel(st[0], st[1], zrc, zcc) for st in STATIONS)

    lo, hi = min(raw.values()), max(raw.values())
    span = (hi - lo) if hi > lo else 1.0   # guard against a degenerate all-equal grid
    _CACHED_ZONE_ENERGY_HAT = {z: (v - lo) / span for z, v in raw.items()}
    return _CACHED_ZONE_ENERGY_HAT


def sse_zone_utility(z, sse_cov, energy_hat):
    """U_z = SSE_z − λ·Ê_z  when SSE_TERRAIN_AWARE (Model B/C),
    else plain U_z = SSE_z (Model A, byte-identical to the original
    behavior). This is the ONLY function rank_zones() calls to get the
    game-theoretic term -- so Model A vs B is a single global-flag flip,
    nothing else in rank_zones() changes between them."""
    base = sse_cov.get(z, 0.0)
    if not SSE_TERRAIN_AWARE:
        return base
    return base - SSE_LAMBDA * energy_hat.get(z, 0.0)


def compute_spe(g, total_energy_consumed):
    """Security-per-Energy = (SSE-weighted protected coverage) / (total
    energy consumed). Reported as its own metric because coverage_pct and
    energy_consumed can each look "the same" between Model A/B/C while
    trading off very differently: SPE is what actually captures whether
    the terrain-aware strategy bought you more game-theoretically-valuable
    protection per unit of energy spent, which plain coverage_pct alone
    can't.

    SSE-weighted protected coverage = Σ_z sse_cov(z) * fraction_of_zone_z_covered
    (fraction covered = mean cell "covered" value in that zone, 0..1).
    This weights each zone's coverage credit by how strategically
    important the static game says it is -- fully covering a
    high-SSE-priority zone counts for more than fully covering a
    low-priority one, which plain coverage_pct treats identically.

    Returns None if total_energy_consumed is ~0 (can't divide), so
    callers don't need a separate guard."""
    if total_energy_consumed is None or total_energy_consumed < 1e-9:
        return None
    sse_cov = get_sse_zone_coverage()
    if not sse_cov:
        return None
    weighted_coverage = 0.0
    for (zr, zc) in get_all_zones():
        cells = get_zone_cells(zr, zc)
        if not cells:
            continue
        frac_covered = sum(g[c]["covered"] for c in cells) / (100.0 * len(cells))
        weighted_coverage += sse_cov.get((zr, zc), 0.0) * frac_covered
    return round(weighted_coverage / total_energy_consumed, 6)


_CACHED_SSE_ZONE_COVERAGE = None
_SSE_CALIBRATION_IN_PROGRESS = False

def get_sse_zone_coverage(breach_penalty=10.0, capture_reward=2.0, capture_penalty=-5.0,
                           calibration_seed=101, calibration_steps=400):
    """SSE-optimal per-zone coverage probability, used as a small prior in
    rank_zones() (Option C hybrid integration — see HY_SSE_W above).
    Computed ONCE and cached globally:
      - Zone priority (which drives the SSE payoff table via
        zone_border_priority()) is deterministic and seed-independent (see
        make_grid()'s edge-based priority formula) — so the payoff table
        itself doesn't vary run to run, no need to recompute per-seed.
      - K (resource budget for the LP) follows the SAME empirical
        calibration convention used consistently across the game-theory code
        elsewhere in this file (short probe simulation -> sum of empirical
        zone-visit frequency), rather than an arbitrary hand-picked
        constant — keeps this consistent with the rest of the game-theory
        code instead of introducing a new unjustified number.
    Gracefully returns {} (no prior, rank_zones() falls back to its
    original pure-heuristic behavior) if anything fails — this is an
    additive nudge, never a hard dependency.

    BUGFIX (recursion guard): the K-calibration probe below
    (observe_zone_coverage_profile) runs a full DroneSimHeadless simulation,
    which itself calls rank_zones() every step — which calls this very
    function. Without a guard, the first-ever call would recurse into
    itself infinitely (each nested call sees the cache still empty and
    re-triggers calibration). _SSE_CALIBRATION_IN_PROGRESS makes any such
    nested call return {} (no nudge) instead of recursing — the probe run
    used for K-calibration thus reflects SMRS's plain heuristic behavior
    (consistent with how this K convention is used elsewhere in the file),
    and the REAL cached prior becomes active for every call afterward."""
    global _CACHED_SSE_ZONE_COVERAGE, _SSE_CALIBRATION_IN_PROGRESS
    if _CACHED_SSE_ZONE_COVERAGE is not None:
        return _CACHED_SSE_ZONE_COVERAGE
    if _SSE_CALIBRATION_IN_PROGRESS:
        return {}
    _SSE_CALIBRATION_IN_PROGRESS = True
    try:
        probe = DroneSimHeadless(calibration_seed)
        zones, Ud_c, Ud_u, Ua_c, Ua_u = build_zone_payoffs(probe.gs, breach_penalty, capture_reward, capture_penalty)
        freq = observe_zone_coverage_profile(calibration_seed, calibration_steps)
        K = float(sum(freq[z] for z in zones))
        sse = solve_sse(Ud_c, Ud_u, Ua_c, Ua_u, K)
        _CACHED_SSE_ZONE_COVERAGE = {z: float(p) for z, p in zip(zones, sse["coverage"])}
        print(f"✅ SSE zone-coverage prior ক্যালিব্রেট হয়েছে (K={K:.2f}, {len(zones)} zones) — "
              f"SMRS zone-selection এখন এই game-theoretic prior দ্বারা আংশিকভাবে informed।")
    except Exception as e:
        print(f"⚠️ SSE zone-coverage prior compute ব্যর্থ হয়েছে ({e}) — SMRS zone-selection "
              f"স্বাভাবিক heuristic-ই চলবে, কোনো SSE nudge ছাড়া।")
        _CACHED_SSE_ZONE_COVERAGE = {}
    finally:
        _SSE_CALIBRATION_IN_PROGRESS = False
    return _CACHED_SSE_ZONE_COVERAGE


def reset_sse_zone_coverage_cache():
    """Clears BOTH sse-related caches (raw SSE coverage AND the derived
    terrain-aware energy estimate) -- needed whenever TERRAIN_COST_ENABLED
    changes, since energy_to_travel() (which Ê_z depends on) reads that
    flag; a stale Ê_z cache would otherwise silently keep using energy
    numbers computed under whichever terrain setting happened to run
    first, exactly the kind of silent-no-op bug the flat-vs-terrain
    ablation is designed to catch."""
    global _CACHED_SSE_ZONE_COVERAGE, _SSE_CALIBRATION_IN_PROGRESS
    _CACHED_SSE_ZONE_COVERAGE = None
    _SSE_CALIBRATION_IN_PROGRESS = False
    reset_energy_aware_sse_cache()


class DroneSimGUI:
    def __init__(self,root):
        self.root=root
        self.root.title("Multi-Drone Relay Handoff (SMRS) vs Greedy vs GPS-Snake vs ACO")
        self.root.configure(bg="#1a1a2e")
        self.cell = 9
        self.reset_state()
        self.create_widgets()
        self.draw_grids()

    def reset_state(self):
        seed = random.randint(1,999)
        self.zone_rng = random.Random(seed*31 + 17)   
        self.gs, self.ggre, self.ggps, self.gaco = make_grid(seed), make_grid(seed), make_grid(seed), make_grid(seed)
        # reseed=False: original code never called random.seed(seed) here (seed
        # itself came from random.randint(1,999) using whatever global state
        # already existed), so this keeps that exact behavior when the biased
        # environment is off.
        self.threats = place_threats(seed, [self.gs, self.ggre, self.ggps, self.gaco], reseed=False)

        self.s_active = {"r": 1, "c": 1, "b": 100, "pr": None, "pc": None}
        self.s_returning = []
        self.s_incoming = None
        self.s_handoff_mode = False
        
        self.sR = 0; self.s_detected = set()
        self.target_zone = None; self.target_cell = None

        self.grer,self.grec,self.greb = 1,1,100
        self.gre_pr=self.gre_pc=None; self.greR=0
        self.gre_detected=set()
        self.gre_returning=[]; self.gre_incoming=None; self.gre_handoff_mode=False

        self.gr,self.gc,self.gb = 1,1,100
        self.g_pr, self.g_pc = None, None
        self.gR=0; self.dr=1
        self.gps_detected=set()
        self.gps_returning=[]; self.gps_incoming=None; self.gps_handoff_mode=False

        # --- ACO (4th policy) ---
        self.aco_route = aco_solve_route(get_all_zones(), (1,1), seed=seed)
        self._aco_idx = 0
        self.ar,self.ac,self.ab = 1,1,100
        self.aco_pr,self.aco_pc = None,None
        self.aR=0
        self.aco_detected=set()
        self.aco_returning=[]; self.aco_incoming=None; self.aco_handoff_mode=False

        self.step=0; self.log=[]; self.running=False; self.speed=50

    def create_widgets(self):
        top=tk.Frame(self.root,bg="#16213e",pady=5); top.pack(fill="x")
        self.step_lbl=tk.Label(top,text=f"Step:0/{MAX_STEPS}", font=("Consolas",11,"bold"),fg="#e2e2e2",bg="#16213e")
        self.step_lbl.pack(side="left",padx=10)
        self.info_lbl=tk.Label(top,text="—", font=("Consolas",9),fg="#ffdd57",bg="#16213e")
        self.info_lbl.pack(side="left",padx=8)

        mf=tk.Frame(self.root,bg="#1a1a2e"); mf.pack(padx=8,pady=4)

        sf=tk.Frame(mf,bg="#0f3460",bd=2,relief="groove"); sf.pack(side="left",padx=3)
        tk.Label(sf,text="🧠 SMRS (Relay Fleet)", font=("Arial",9,"bold"),fg="#00d2ff",bg="#0f3460").pack(pady=2)
        self.sc_=tk.Canvas(sf,width=COLS*self.cell,height=ROWS*self.cell, bg="#0a0a1a",highlightthickness=0); self.sc_.pack()
        self.s_stat=tk.Label(sf,text="Bat:100% | RC:0 | Cov:0%", font=("Consolas",8),fg="#aaffaa",bg="#0f3460"); self.s_stat.pack(pady=2)

        grf=tk.Frame(mf,bg="#4a235a",bd=2,relief="groove"); grf.pack(side="left",padx=3)
        tk.Label(grf,text="🎯 Greedy (Single Drone)", font=("Arial",9,"bold"),fg="#d7bde2",bg="#4a235a").pack(pady=2)
        self.gc2=tk.Canvas(grf,width=COLS*self.cell,height=ROWS*self.cell, bg="#0a0a1a",highlightthickness=0); self.gc2.pack()
        self.g_stat2=tk.Label(grf,text="Bat:100% | RC:0 | Cov:0%", font=("Consolas",8),fg="#ebdef0",bg="#4a235a"); self.g_stat2.pack(pady=2)

        gpf=tk.Frame(mf,bg="#3d0000",bd=2,relief="groove"); gpf.pack(side="left",padx=3)
        tk.Label(gpf,text="🛰️ GPS Snake (Single)", font=("Arial",9,"bold"),fg="#ff9999",bg="#3d0000").pack(pady=2)
        self.gpc=tk.Canvas(gpf,width=COLS*self.cell,height=ROWS*self.cell, bg="#0a0a1a",highlightthickness=0); self.gpc.pack()
        self.g_stat3=tk.Label(gpf,text="Bat:100% | RC:0 | Cov:0%", font=("Consolas",8),fg="#ffaaaa",bg="#3d0000"); self.g_stat3.pack(pady=2)

        acf=tk.Frame(mf,bg="#1a4d2e",bd=2,relief="groove"); acf.pack(side="left",padx=3)
        tk.Label(acf,text="🐜 ACO (Single Drone)", font=("Arial",9,"bold"),fg="#7dffb3",bg="#1a4d2e").pack(pady=2)
        self.acc=tk.Canvas(acf,width=COLS*self.cell,height=ROWS*self.cell, bg="#0a0a1a",highlightthickness=0); self.acc.pack()
        self.a_stat=tk.Label(acf,text="Bat:100% | RC:0 | Cov:0%", font=("Consolas",8),fg="#c8ffdf",bg="#1a4d2e"); self.a_stat.pack(pady=2)

        self.hop_lbl=tk.Label(self.root, text="SMRS Backup Status: Waiting...",font=("Consolas",8),fg="#888",bg="#1a1a2e")
        self.hop_lbl.pack(pady=1)

        leg=tk.Frame(self.root,bg="#1a1a2e"); leg.pack(pady=2)
        for sym,lbl in [("🟨","Active"),("🟧","Returning"),("🟦","Incoming"),("🟩","Covered"),("🟥","Threat")]:
            tk.Label(leg,text=f"{sym}{lbl}",font=("Arial",8), fg="#ccc",bg="#1a1a2e").pack(side="left",padx=4)

        # ----------------------------------------------------
        # বাটন বার — শুধু core simulation control + battery/coverage comparison
        # ----------------------------------------------------
        cf=tk.Frame(self.root,bg="#1a1a2e",pady=4); cf.pack()
        row1 = tk.Frame(cf, bg="#1a1a2e")
        row1.pack(pady=2)

        self.btn_run=tk.Button(row1,text="▶ Start",font=("Arial",10,"bold"), bg="#27ae60",fg="white",width=9,command=self.toggle_run)
        self.btn_run.pack(side="left",padx=4)
        tk.Button(row1,text="⏭ Step",font=("Arial",9),bg="#2980b9", fg="white",width=7,command=self.do_step).pack(side="left",padx=4)
        tk.Button(row1,text="↺ Reset",font=("Arial",9),bg="#7f8c8d", fg="white",width=7,command=self.reset).pack(side="left",padx=4)

        self.btn_batch=tk.Button(row1,text="📊 Batch Avg",font=("Arial",9,"bold"), bg="#8e44ad",fg="white",width=12,command=self.run_batch_avg)
        self.btn_batch.pack(side="left",padx=4)
        self.btn_fair=tk.Button(row1,text="Fair Batch",font=("Arial",9,"bold"), bg="#5b2c6f",fg="white",width=12,command=self.run_fair_batch)
        self.btn_fair.pack(side="left",padx=4)

        tk.Label(row1,text="Speed:",fg="#ccc",bg="#1a1a2e", font=("Arial",9)).pack(side="left",padx=(10,3))
        self.speed_var=tk.IntVar(value=20)
        tk.Scale(row1,from_=5,to=500,orient="horizontal", variable=self.speed_var,bg="#1a1a2e",fg="white", troughcolor="#333",length=120,highlightthickness=0).pack(side="left")

    def run_batch_avg(self):
        self.btn_batch.config(state="disabled", text="⏳ Running...")
        def worker():
            table_text = run_batch_and_format()
            self.root.after(0, lambda: self.show_batch_result(table_text))
        threading.Thread(target=worker, daemon=True).start()

    def show_batch_result(self, table_text):
        self.btn_batch.config(state="normal", text="📊 Batch Avg")
        messagebox.showinfo(f"Batch Average", table_text)

    def run_fair_batch(self):
        self.btn_fair.config(state="disabled", text="⏳ Running (matched-resource)...")
        def worker():
            table_text = run_fair_batch_and_format()
            self.root.after(0, lambda: self.show_fair_batch_result(table_text))
        threading.Thread(target=worker, daemon=True).start()

    def show_fair_batch_result(self, table_text):
        self.btn_fair.config(state="normal", text="⚖️ Fair Batch")
        messagebox.showinfo("Fair / Matched-Resource Comparison", table_text)

    def draw_canvas(self,canvas,grid,active_d,drone_col,draw_zones=False, returning=[], incoming=None):
        canvas.delete("all")
        cs = self.cell
        tz_cells = set(get_zone_cells(*self.target_zone)) if (draw_zones and self.target_zone) else set()

        for r in range(ROWS):
            for c in range(COLS):
                x1,y1 = c*cs,r*cs
                z = grid[(r,c)]
                if z["is_station"]: col="#f39c12"
                elif draw_zones and (r,c) in tz_cells:
                    col="#003a6e" if z["covered"]==0 else("#0a4a2a" if z["covered"]>=100 else "#004488")
                elif z["threat"] and z.get("threat_detected",False): col="#8e44ad"
                elif z["threat"]: col="#c0392b"
                elif z["covered"]>=100: col="#1abc9c" if draw_zones else "#2980b9"
                elif z["covered"]>0: col="#148f77" if draw_zones else "#1a5276"
                else: col="#0d1b2a"
                canvas.create_rectangle(x1,y1,x1+cs,y1+cs,fill=col,outline="",width=0)

                if z["is_station"]: canvas.create_text(x1+cs//2,y1+cs//2,text="⚡",font=("Arial",7))
                elif z["threat"] and z.get("threat_detected",False): canvas.create_text(x1+cs//2,y1+cs//2,text="✅",font=("Arial",6))
                elif z["threat"]: canvas.create_text(x1+cs//2,y1+cs//2,text="🚨",font=("Arial",6))

        if draw_zones:
            for zr in range(math.ceil(ROWS/ZONE_SIZE)+1): canvas.create_line(0,zr*ZONE_SIZE*cs,COLS*cs, zr*ZONE_SIZE*cs,fill="#2a3a4a",dash=(2,3))
            for zc in range(math.ceil(COLS/ZONE_SIZE)+1): canvas.create_line(zc*ZONE_SIZE*cs,0,zc*ZONE_SIZE*cs, ROWS*cs,fill="#2a3a4a",dash=(2,3))
            if self.target_zone:
                tzr,tzc = self.target_zone
                zx1,zy1 = tzc*ZONE_SIZE*cs, tzr*ZONE_SIZE*cs
                canvas.create_rectangle(zx1,zy1, zx1+ZONE_SIZE*cs,zy1+ZONE_SIZE*cs, outline="#00aaff",width=2)

        for rd in returning:
            px,py = rd["c"]*cs+cs//2, rd["r"]*cs+cs//2
            canvas.create_oval(px-cs//2+2,py-cs//2+2,px+cs//2-2,py+cs//2-2, fill="#e67e22",outline="")

        if incoming:
            px,py = incoming["c"]*cs+cs//2, incoming["r"]*cs+cs//2
            r_small = max(3, cs//2 - 3)
            ox, oy = -cs//4, -cs//4
            canvas.create_oval(px+ox-r_small,py+oy-r_small,px+ox+r_small,py+oy+r_small, outline="#00ffff", width=2)

        if active_d:
            px,py = active_d["c"]*cs+cs//2, active_d["r"]*cs+cs//2
            canvas.create_oval(px-cs//2,py-cs//2,px+cs//2,py+cs//2, fill=drone_col,outline="#fff",width=1)

    def draw_grids(self):
        self.draw_canvas(self.sc_, self.gs, self.s_active, "#f0c040", draw_zones=True, returning=self.s_returning, incoming=self.s_incoming)
        self.draw_canvas(self.gc2, self.ggre, {"r":self.grer,"c":self.grec}, "#d7bde2", returning=self.gre_returning, incoming=self.gre_incoming)
        self.draw_canvas(self.gpc, self.ggps, {"r":self.gr,"c":self.gc}, "#e74c3c", returning=self.gps_returning, incoming=self.gps_incoming)
        self.draw_canvas(self.acc, self.gaco, {"r":self.ar,"c":self.ac}, "#2ecc71", returning=self.aco_returning, incoming=self.aco_incoming)

    def do_step(self):
        if self.step>=MAX_STEPS: self.finish(); return
        self.step+=1
        advance_sim_clock(self.step)   # keep dynamic-wind clock in sync (no-op if WIND_DYNAMIC_ENABLED=False)

        # ==============================================================
        # 1. SMRS: RELAY HANDOFF LOGIC
        # ==============================================================
        for rd in self.s_returning[:]:
            if (rd["r"], rd["c"]) == rd["st"]:
                self.sR += 1
                self.s_returning.remove(rd)
            else:
                old_r, old_c = rd["r"], rd["c"]
                nr, nc = move_toward(rd["r"], rd["c"], rd["st"][0], rd["st"][1])
                rd["r"], rd["c"] = nr, nc
                cost = calculate_actual_step_cost(old_r, old_c, nr, nc, None, None, cycle_count=(self.sR if USE_SOH_AGING else 0))
                rd["b"] = max(0, rd["b"] - cost)
                sensor_sweep(nr, nc, self.gs, self.s_detected, step=self.step)

        if self.s_handoff_mode and self.s_incoming:
            tr, tc = self.s_active["r"], self.s_active["c"]
            dist_ai = abs(self.s_incoming["r"]-tr) + abs(self.s_incoming["c"]-tc)
            if dist_ai <= 1:
                old_active = self.s_active.copy()
                old_active["st"] = nearest_st(tr, tc)
                self.s_returning.append(old_active)

                self.s_active = self.s_incoming
                self.s_active["pr"] = self.s_active["pc"] = None
                self.s_handoff_mode = False
                self.s_incoming = None
                self.target_zone = None; self.target_cell = None
                self.info_lbl.config(text=f"🔄 Handoff Complete near {tr},{tc}", fg="#00d2ff")
                self.hop_lbl.config(text="SMRS Backup Status: Idle", fg="#888")
            else:
                old_r, old_c = self.s_incoming["r"], self.s_incoming["c"]
                nr, nc = move_toward(self.s_incoming["r"], self.s_incoming["c"], tr, tc)
                self.s_incoming["r"], self.s_incoming["c"] = nr, nc
                cost = calculate_actual_step_cost(old_r, old_c, nr, nc, None, None, cycle_count=(self.sR if USE_SOH_AGING else 0))
                self.s_incoming["b"] = max(0, self.s_incoming["b"] - cost)
                sensor_sweep(nr, nc, self.gs, self.s_detected, step=self.step)

        nh, st = needs_handoff_now(self.s_active["r"], self.s_active["c"], self.s_active["b"])
        if nh and not self.s_handoff_mode:
            self.s_handoff_mode = True
            self.s_incoming = {"r": st[0], "c": st[1], "b": 100}
            self.info_lbl.config(text=f"🚀 Backup launched from {st}", fg="#ff9900")
            self.hop_lbl.config(text=f"SMRS Backup: INCOMING from {st}", fg="#00ffff")

        mrc, e_ret, st = must_recharge_now(self.s_active["r"], self.s_active["c"], self.s_active["b"])

        if mrc:
            old_active = self.s_active.copy()
            old_active["st"] = st
            self.s_returning.append(old_active)

            if self.s_incoming:
                self.s_active = self.s_incoming
                self.s_active["pr"] = self.s_active["pc"] = None
                self.s_handoff_mode = False
                self.s_incoming = None
                self.info_lbl.config(text=f"⚠️ Emergency Swap!", fg="#ff4444")
            else:
                self.s_active = {"r": st[0], "c": st[1], "b": 100, "pr": None, "pc": None}
            self.target_zone = None; self.target_cell = None

        elif self.s_handoff_mode and self.s_incoming:
            old_r, old_c = self.s_active["r"], self.s_active["c"]
            pr, pc = self.s_active["pr"], self.s_active["pc"]
            if self.target_zone:
                zr, zc = self.target_zone
                cell_here = find_uncovered_in_zone(zr, zc, self.gs)
                if cell_here:
                    tr, tc_ = cell_here
                    nr, nc = bfs_next_step(self.s_active["r"], self.s_active["c"], tr, tc_, self.gs)
                else:
                    zrc = zr*ZONE_SIZE + ZONE_SIZE//2
                    zcc = zc*ZONE_SIZE + ZONE_SIZE//2
                    nr, nc = move_toward(self.s_active["r"], self.s_active["c"], zrc, zcc)
            else:
                nr, nc = self.s_active["r"], self.s_active["c"]

            self.s_active["r"], self.s_active["c"] = nr, nc
            cell = self.gs[(nr, nc)]
            if not cell["is_station"]:
                cost = calculate_actual_step_cost(old_r, old_c, nr, nc, pr, pc, cycle_count=(self.sR if USE_SOH_AGING else 0))
                self.s_active["b"] = max(0, self.s_active["b"] - cost)
                sensor_sweep(nr, nc, self.gs, self.s_detected, step=self.step)
        else:
            old_r, old_c = self.s_active["r"], self.s_active["c"]
            pr, pc = self.s_active["pr"], self.s_active["pc"]

            if self.target_zone:
                zr,zc=self.target_zone
                if all(self.gs[cl]["covered"]>=100 for cl in get_zone_cells(zr,zc)):
                    rk=rank_zones(self.s_active["r"],self.s_active["c"],self.gs,self.s_detected,self.step)
                    self.target_zone,self.target_cell=select_zone_mixed_strategy(rk,self.gs,self.s_active["r"],self.s_active["c"],self.s_active["b"],self.zone_rng,self.step)
            if not self.target_zone:
                rk=rank_zones(self.s_active["r"],self.s_active["c"],self.gs,self.s_detected,self.step)
                self.target_zone,self.target_cell=select_zone_mixed_strategy(rk,self.gs,self.s_active["r"],self.s_active["c"],self.s_active["b"],self.zone_rng,self.step)

            if self.target_cell:
                tr,tc_=self.target_cell
                can_go, e1, e2, eni = two_hop_check(
                    self.s_active["r"], self.s_active["c"], tr, tc_, self.s_active["b"])
                if can_go:
                    nr, nc = bfs_next_step(self.s_active["r"],self.s_active["c"],tr,tc_,self.gs)
                    self.s_active["r"], self.s_active["c"] = nr, nc
                else:
                    if self.gs[(self.s_active["r"], self.s_active["c"])]["is_station"]:
                        self.s_active["b"] = 100
                        self.target_zone = None; self.target_cell = None
                    else:
                        st_ = nearest_st(self.s_active["r"], self.s_active["c"])
                        nr, nc = bfs_next_step(self.s_active["r"], self.s_active["c"], st_[0], st_[1], self.gs)
                        self.s_active["r"], self.s_active["c"] = nr, nc
                        self.target_zone = None; self.target_cell = None
            else:
                self.s_active["r"],self.s_active["c"],_=smart_move(
                    self.s_active["r"],self.s_active["c"],self.s_active["pr"],self.s_active["pc"],
                    self.gs,self.s_detected,self.target_zone)

            new_r, new_c = self.s_active["r"], self.s_active["c"]
            cell=self.gs[(new_r, new_c)]
            if not cell["is_station"]:
                cost = calculate_actual_step_cost(old_r, old_c, new_r, new_c, pr, pc, cycle_count=(self.sR if USE_SOH_AGING else 0))
                self.s_active["b"]=max(0,self.s_active["b"]-cost)
                sensor_sweep(new_r, new_c, self.gs, self.s_detected, step=self.step)
                _adaptive_learning_step_hook(new_r, new_c, self.gs)

            if self.target_zone:
                self.target_cell=find_uncovered_in_zone(*self.target_zone,self.gs)

        self.s_active["pr"], self.s_active["pc"] = self.s_active["r"], self.s_active["c"]

        # ==============================================================
        # 2. Greedy (Relay-Handoff — matched with SMRS, FIX per user request)
        # ==============================================================
        gre_state = {
            "active": {"r": self.grer, "c": self.grec, "b": self.greb,
                       "pr": self.gre_pr, "pc": self.gre_pc},
            "returning": self.gre_returning, "incoming": self.gre_incoming,
            "handoff_mode": self.gre_handoff_mode, "RC": self.greR, "mstate": None,
        }
        gre_state = _advance_relay_policy(gre_state, self.ggre, self.gre_detected,
                                           _greedy_move_wrapper, self.step)
        a = gre_state["active"]
        self.grer, self.grec, self.greb = a["r"], a["c"], a["b"]
        self.gre_pr, self.gre_pc = a["pr"], a["pc"]
        self.gre_returning = gre_state["returning"]; self.gre_incoming = gre_state["incoming"]
        self.gre_handoff_mode = gre_state["handoff_mode"]; self.greR = gre_state["RC"]

        # ==============================================================
        # 3. GPS-Snake (Relay-Handoff — matched with SMRS, FIX per user request)
        # ==============================================================
        gps_state = {
            "active": {"r": self.gr, "c": self.gc, "b": self.gb,
                       "pr": self.g_pr, "pc": self.g_pc},
            "returning": self.gps_returning, "incoming": self.gps_incoming,
            "handoff_mode": self.gps_handoff_mode, "RC": self.gR, "mstate": self.dr,
        }
        gps_state = _advance_relay_policy(gps_state, self.ggps, self.gps_detected,
                                           _gps_move_wrapper, self.step)
        a = gps_state["active"]
        self.gr, self.gc, self.gb = a["r"], a["c"], a["b"]
        self.g_pr, self.g_pc = a["pr"], a["pc"]
        self.gps_returning = gps_state["returning"]; self.gps_incoming = gps_state["incoming"]
        self.gps_handoff_mode = gps_state["handoff_mode"]; self.gR = gps_state["RC"]
        self.dr = gps_state["mstate"]   # snake direction carries across handoffs

        # ==============================================================
        # 4. ACO (Relay-Handoff — matched with SMRS, FIX per user request)
        #    route precomputed via Ant Colony Optimization
        # ==============================================================
        aco_state = {
            "active": {"r": self.ar, "c": self.ac, "b": self.ab,
                       "pr": self.aco_pr, "pc": self.aco_pc},
            "returning": self.aco_returning, "incoming": self.aco_incoming,
            "handoff_mode": self.aco_handoff_mode, "RC": self.aR, "mstate": self._aco_idx,
        }
        aco_state = _advance_relay_policy(aco_state, self.gaco, self.aco_detected,
                                           _make_aco_move_wrapper(self.aco_route), self.step)
        a = aco_state["active"]
        self.ar, self.ac, self.ab = a["r"], a["c"], a["b"]
        self.aco_pr, self.aco_pc = a["pr"], a["pc"]
        self.aco_returning = aco_state["returning"]; self.aco_incoming = aco_state["incoming"]
        self.aco_handoff_mode = aco_state["handoff_mode"]; self.aR = aco_state["RC"]
        self._aco_idx = aco_state["mstate"]   # route index carries across handoffs

        nt=len(self.threats)
        sc_  = coverage_pct(self.gs)
        grc_ = coverage_pct(self.ggre)
        gpc_ = coverage_pct(self.ggps)
        gac_ = coverage_pct(self.gaco)
        st_=len(self.s_detected)
        grt_=len(self.gre_detected)
        gpt_=len(self.gps_detected)
        gat_=len(self.aco_detected)

        self.log.append({
            "step":self.step, "s_cov":sc_,"gre_cov":grc_,"g_cov":gpc_,"aco_cov":gac_,
            "s_bat":round(self.s_active["b"],1),"gre_bat":round(self.greb,1),"g_bat":round(self.gb,1),"aco_bat":round(self.ab,1),
            "s_thr":st_,"gre_thr":grt_,"g_thr":gpt_,"aco_thr":gat_,
            "s_RC":self.sR,"gre_RC":self.greR,"g_RC":self.gR,"aco_RC":self.aR
        })

        self.step_lbl.config(text=f"Step:{self.step}/{MAX_STEPS}")
        self.s_stat.config(text=f"Bat:{self.s_active['b']:.1f}% | RC:{self.sR} | Cov:{sc_}% | Thr:{st_}/{nt}")
        self.g_stat2.config(text=f"Bat:{self.greb:.1f}% | RC:{self.greR} | Cov:{grc_}% | Thr:{grt_}/{nt}")
        self.g_stat3.config(text=f"Bat:{self.gb:.1f}% | RC:{self.gR} | Cov:{gpc_}% | Thr:{gpt_}/{nt}")
        self.a_stat.config(text=f"Bat:{self.ab:.1f}% | RC:{self.aR} | Cov:{gac_}% | Thr:{gat_}/{nt}")
        self.draw_grids()

    def toggle_run(self):
        self.running=not self.running
        self.btn_run.config(text="⏸ Pause" if self.running else "▶ Start", bg="#e67e22" if self.running else "#27ae60")
        if self.running: self.auto_run()

    def auto_run(self):
        if self.running and self.step<MAX_STEPS:
            self.do_step()
            self.root.after(self.speed_var.get(),self.auto_run)
        elif self.step>=MAX_STEPS:
            self.running=False
            self.btn_run.config(text="▶ Start",bg="#27ae60")
            self.finish()

    def reset(self):
        self.running=False
        self.btn_run.config(text="▶ Start",bg="#27ae60")
        self.reset_state(); self.info_lbl.config(text="—")
        self.hop_lbl.config(text="SMRS Backup Status: Idle")
        self.draw_grids()

    def finish(self):
        self.running=False
        last=self.log[-1] if self.log else {}
        nt=len(self.threats)
        msg=(
            f"{'─'*52}\n  FINAL RESULTS — 30×30 Grid ({MAX_STEPS} steps)\n{'─'*52}\n"
            f"🧠 SMRS(Fleet) → Cov:{last.get('s_cov',0)}%  Thr:{last.get('s_thr',0)}/{nt}  RC:{self.sR}\n"
            f"🎯 Greedy      → Cov:{last.get('gre_cov',0)}%  Thr:{last.get('gre_thr',0)}/{nt}  RC:{self.greR}\n"
            f"🛰️  GPS       → Cov:{last.get('g_cov',0)}%  Thr:{last.get('g_thr',0)}/{nt}  RC:{self.gR}\n"
            f"🐜 ACO        → Cov:{last.get('aco_cov',0)}%  Thr:{last.get('aco_thr',0)}/{nt}  RC:{self.aR}\n"
            f"{'─'*52}\n"
        )
        messagebox.showinfo("Simulation Complete",msg)

if __name__=="__main__":
    root=tk.Tk()
    app=DroneSimGUI(root)
    root.mainloop()
