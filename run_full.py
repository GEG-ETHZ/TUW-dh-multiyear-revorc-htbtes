"""
Southern Munich (GHP) case — FULL 8760-h resolution run.

Difference vs. run_rd.py (representative days):
  * Full chronological 8760 h per investment period (no clustering).
  * Geothermal modelled as a UNIT-COMMITMENT plant that the optimizer can
    SHUT DOWN, with a 4-week (672 h) cold-start: after a restart the well
    produces nothing for 672 h, and once off it stays off >= 672 h
    (min-down-time). When running it respects the 65% min-load.
  * No curtailment — unused geothermal heat is REINJECTED hot into the well
    (free slack `reinjection` in the energy balance). Reported as reinjection.

The 4-week restart constraint needs a continuous calendar, which only the full
chronological model provides (the rep-day model has no 672-h window).

Usage:
    python run_full.py
"""

import sys, os, time
import numpy as np
import pandas as pd
from datetime import datetime
from loguru import logger
from pathlib import Path
import pyomo.environ as pyo


# ── Graceful stop ────────────────────────────────────────────────────────────
# Registered at import time so it is active during model build as well as during
# the solve. SIGTERM/SIGINT become KeyboardInterrupt: the solve wrapper catches it
# and the run continues to the reporting step instead of dying, so `kill <pid>`
# (NOT kill -9) stops a run early and still writes results. Gurobi additionally
# dumps every new incumbent to <run>/incumbent_N.sol via the SolFiles parameter,
# so the best solution found so far survives even a hard kill.
import signal as _signal


def _graceful_stop(signum, _frame):
    """Stop the SOLVE early, never the save.

    While the solver is running this raises KeyboardInterrupt, which the solve
    wrapper catches so the run proceeds to reporting with the best incumbent.
    Once the solve has returned, the signal is ignored: interrupting the write-up
    is the one thing that actually loses results, which is what this exists to
    prevent. Send it twice more to force a real exit.
    """
    global _STOP_REQUESTS
    _STOP_REQUESTS += 1
    if _IN_SOLVE:
        logger.warning(f"signal {signum} received — stopping the solve, the best "
                       f"incumbent will still be reported")
        raise KeyboardInterrupt
    if _STOP_REQUESTS >= 3:
        logger.error("third stop request — exiting without finishing the save")
        raise SystemExit(130)
    logger.warning(f"signal {signum} received while saving results — finishing the "
                   f"write-up first (send twice more to force an exit)")


_IN_SOLVE = False
_STOP_REQUESTS = 0


for _sig in (_signal.SIGINT, _signal.SIGTERM):
    try:
        _signal.signal(_sig, _graceful_stop)
    except (ValueError, OSError):
        pass

GRB_THREADS = int(os.environ.get('GRB_THREADS', 40))
try:
    os.sched_setaffinity(0, set(range(1, 41)))
except (AttributeError, OSError):
    pass

_CASE_DIR     = Path(__file__).parent.resolve()
sys.path.insert(0, str(_CASE_DIR / "src"))
sys.path.insert(0, str(_CASE_DIR))

from data_loader import load_all_munich_data, build_uc_params
from models.full_model import DistrictHeatingModelFull
from utils.solver import get_solver_name
from run_rd import (derive_saphea_capacities, derive_hthp_capex,
                    GEO_TECH, HTHP_TECH, REVORC_TECH, _inject_revorc_tech)
from orc_surrogate_maps import load_revorc_config, build_revorc_maps

# run_rd registers its own SIGINT/SIGTERM handler at import time, which would replace
# ours and read run_rd's _IN_SOLVE flag (never set here). Re-install ours.
for _sig in (_signal.SIGINT, _signal.SIGTERM):
    _signal.signal(_sig, _graceful_stop)

# ── Settings ──────────────────────────────────────────────────────────────────
GEO_COLD_START_H   = 672    # 4 weeks: after a restart, geo produces 0 for 672 h
GEO_MIN_DOWN_H     = 672    # once shut down, geo stays off >= 672 h
# Default OFF (must-run, u=1 fixed) to mirror run_rd's GEO_MUST_RUN_ALWAYS —
# required for an apples-to-apples RD vs full validation. GEO_SHUTDOWN=1 restores
# the 4-week shutdown/cold-start study of the 2026-06-26 run.
ALLOW_GEO_SHUTDOWN = os.environ.get('GEO_SHUTDOWN', '0') == '1'
SOLVE_MIPGAP       = float(os.environ.get('MIPGAP', '0.0025'))  # match run_rd
SOLVE_TIMELIMIT_S  = int(os.environ.get('TIMELIMIT_S', 48 * 3600))   # 2 days


def _saphea_cop_profile(demand, resource):
    """Per-hour SAPHEA standard COP for the HTHP (supply-temp driven).

    Computed before model build so the precomputed electricity-cost coefficients
    use the dynamic COP. COP_std = η·(T_sup+273.15)/(T_sup−T_inject_min) + 1.
    """
    eta = resource["carnot_eta"]; T_min = resource["T_inject_min"]
    supply = demand['supply_temp'].values
    cop = np.empty(len(supply))
    for h in range(len(supply)):
        T_s = float(supply[h])
        cop[h] = eta * (T_s + 273.15) / max(T_s - T_min, 1.0) + 1.0
    return cop


def _build_saphea_coupling_full(model, demand, resource):
    """SAPHEA 3-mode geothermal+HTHP coupling at full 8760-h resolution.

    Port of run_rd._build_saphea_coupling from (REP_DAYS, HOURS) to (HOURS),
    using the raw measured supply/return-temperature profiles directly.
    """
    m = model.model
    if HTHP_TECH not in model.technologies:
        logger.warning(f"'{HTHP_TECH}' not in technologies — SAPHEA skipped")
        return

    mdot = resource["brine_mass_flow"]; cp = resource["brine_cp"]
    T_prod = resource["T_prod"]; T_min = resource["T_inject_min"]
    PINCH_K = resource["direct_hx_pinch"]; eta = resource["carnot_eta"]
    design = derive_saphea_capacities(resource)
    Q_geo_du_design = design["Q_geo_du"]; Q_evap_ti = design["Q_evap_ti"]

    def _q_brine(dT):
        return mdot * cp * max(dT, 0.0) / 1000.0

    supply = demand['supply_temp'].values
    ret    = demand['return_temp'].values
    T = len(supply)

    coupling_ci, hp_max_ti, q_geo_du, geo_delta, ti_req = {}, {}, {}, {}, {}
    ci_hp_ceiling = {}; cop_log = []
    for h in range(T):
        T_s = float(supply[h]); T_ret = float(ret[h]); T_woHP = T_ret + PINCH_K
        q_du = min(_q_brine(T_prod - T_ret), Q_geo_du_design)
        q_ci = _q_brine(T_prod - T_woHP)
        q_geo_du[h] = q_du; geo_delta[h] = q_du - q_ci
        dT_lift = max(T_s - T_min, 1.0)
        cop_saphea = eta * (T_s + 273.15) / dT_lift
        cop_std = cop_saphea + 1.0
        dT_evap_ci = T_woHP - T_min; dT_direct = T_prod - T_woHP
        coupling_ci[h] = (dT_evap_ci / dT_direct) * (1.0 + 1.0 / cop_saphea)
        hp_max_ti[h] = Q_evap_ti * (1.0 + 1.0 / cop_saphea)
        ci_hp_ceiling[h] = q_ci * coupling_ci[h]
        ti_req[h] = int(T_s > T_prod)
        m.efficiency[HTHP_TECH, h].set_value(cop_std)
        cop_log.append(cop_std)

    m.hthp_coupling_ratio = pyo.Param(m.HOURS, initialize=coupling_ci)
    m.hthp_hp_max_ti      = pyo.Param(m.HOURS, initialize=hp_max_ti)
    m.q_geo_du_cap        = pyo.Param(m.HOURS, initialize=q_geo_du)
    m.geo_delta_ci        = pyo.Param(m.HOURS, initialize=geo_delta)
    m.ti_required         = pyo.Param(m.HOURS, initialize=ti_req, within=pyo.Binary)

    _q_hthp_max = float(model.technologies[HTHP_TECH]['max_output_MW'])

    # Mode binaries per (period, hour). For Munich ti_required = 0 everywhere
    # (T_prod 106 > supply), so y_ti is fixed to 0 and only y_ci (DU vs CI) is free.
    m.y_ci = pyo.Var(m.PERIODS, m.HOURS, domain=pyo.Binary)
    m.y_ti = pyo.Var(m.PERIODS, m.HOURS, domain=pyo.Binary)

    m.mode_mutex = pyo.Constraint(m.PERIODS, m.HOURS,
        rule=lambda m, p, h: m.y_ci[p, h] + m.y_ti[p, h] <= 1)
    m.ti_force = pyo.Constraint(m.PERIODS, m.HOURS,
        rule=lambda m, p, h: m.y_ti[p, h] >= m.ti_required[h])
    m.ti_block = pyo.Constraint(m.PERIODS, m.HOURS,
        rule=lambda m, p, h: m.y_ti[p, h] <= m.ti_required[h])
    m.ci_block_ti = pyo.Constraint(m.PERIODS, m.HOURS,
        rule=lambda m, p, h: m.y_ci[p, h] <= 1 - m.ti_required[h])
    m.geo_mode_cap = pyo.Constraint(m.PERIODS, m.HOURS,
        rule=lambda m, p, h: m.gen[GEO_TECH, p, h]
            <= m.q_geo_du_cap[h] * (1 - m.y_ti[p, h]) - m.geo_delta_ci[h] * m.y_ci[p, h])
    m.hthp_mode_off = pyo.Constraint(m.PERIODS, m.HOURS,
        rule=lambda m, p, h: m.gen[HTHP_TECH, p, h]
            <= _q_hthp_max * (m.y_ci[p, h] + m.y_ti[p, h]))
    m.hthp_ci_coupling = pyo.Constraint(m.PERIODS, m.HOURS,
        rule=lambda m, p, h: m.gen[HTHP_TECH, p, h]
            <= m.gen[GEO_TECH, p, h] * m.hthp_coupling_ratio[h] + _q_hthp_max * (1 - m.y_ci[p, h]))
    m.hthp_ti_cap = pyo.Constraint(m.PERIODS, m.HOURS,
        rule=lambda m, p, h: m.gen[HTHP_TECH, p, h]
            <= m.hthp_hp_max_ti[h] * m.y_ti[p, h] + _q_hthp_max * (1 - m.y_ti[p, h]))

    cop_arr = np.array(cop_log)
    logger.info(
        f"SAPHEA-full built — COP_std {cop_arr.min():.2f}–{cop_arr.max():.2f} "
        f"(mean {cop_arr.mean():.2f}); TI slots {sum(ti_req.values())}/{T}; "
        f"geo DU {min(q_geo_du.values()):.2f}–{max(q_geo_du.values()):.2f} MW"
    )
    model._saphea_design = design


def _build_revorc_full(model, demand, resource, cfg):
    """Series revORC at full 8760-h resolution (port of run_rd._build_revorc).

    Identical physics to the RD version: ORC parallel at the wellhead (FLEXGEO
    surrogate maps, feasibility-gated), HP SERIES on the SAPHEA CI slice at the
    series-trained FLEXGEO HP COP (injected via efficiency_profiles at
    construction — see main()), one shared fleet with the FlexGeo dead-hour
    switching rule — here chronological over the full year instead of cyclic
    within a rep day. Surrogates are evaluated on the raw hourly profiles
    reshaped to (365, 24) pseudo-rep-days.
    """
    m = model.model
    if REVORC_TECH not in model.technologies:
        logger.warning(f"'{REVORC_TECH}' not in model technologies — revORC skipped")
        return
    if "ambient_temp" not in demand.columns:
        raise ValueError("revORC needs the 'ambient_temp' column in Munich_DH_data.xlsx")

    T = 8760
    rep_amb = demand["ambient_temp"].values[:T].reshape(365, 24)
    rep_sup = demand["supply_temp"].values[:T].reshape(365, 24)
    rep_ret = demand["return_temp"].values[:T].reshape(365, 24)
    maps = build_revorc_maps(rep_amb, rep_sup, rep_ret, resource, cfg)
    q_in  = {d * 24 + hh: v for (d, hh), v in maps["orc_q_in_MW"].items()}
    p_net = {d * 24 + hh: v for (d, hh), v in maps["orc_p_net_MW"].items()}
    mod_MW = cfg["module_hp_MWth"]

    m.orc_share = pyo.Var(m.PERIODS, m.HOURS, domain=pyo.NonNegativeReals)
    m.hp_share  = pyo.Var(m.PERIODS, m.HOURS, domain=pyo.NonNegativeReals)

    m.revorc_fleet = pyo.Constraint(m.PERIODS, m.HOURS,
        rule=lambda m, p, h: m.orc_share[p, h] + m.hp_share[p, h]
            <= m.cap_total[REVORC_TECH, p] / mod_MW)

    # Dead-hour switching rule, chronological (cyclic over the year boundary).
    dead_hours = int(cfg.get("dead_hours", 1))
    if dead_hours > 0:
        m.revorc_dead = pyo.ConstraintList()
        for p in m.PERIODS:
            fleet = m.cap_total[REVORC_TECH, p] / mod_MW
            for h in range(T):
                for k in range(1, dead_hours + 1):
                    hk = (h - k) % T
                    m.revorc_dead.add(m.orc_share[p, h] + m.hp_share[p, hk] <= fleet)
                    m.revorc_dead.add(m.hp_share[p, h] + m.orc_share[p, hk] <= fleet)
        logger.info(f"revORC dead-hour rule: {dead_hours} h OFF between modes "
                    f"({len(m.revorc_dead)} constraints)")

    m.revorc_hp_cap = pyo.Constraint(m.PERIODS, m.HOURS,
        rule=lambda m, p, h: m.gen[REVORC_TECH, p, h] <= m.hp_share[p, h] * mod_MW)

    # ORC parallel at wellhead: competes with the direct HX; reinjected must-run
    # surplus is available brine flow (full model always has m.reinjection).
    # geo_delta_ci·y_ci: in CI mode the high-grade band shrinks by the pinch
    # delta — the "freed" budget sits at ~65 °C, outside the ORC's inlet domain,
    # so it must NOT be ORC-usable (prevents phantom winter ORC operation).
    m.revorc_brine = pyo.Constraint(m.PERIODS, m.HOURS,
        rule=lambda m, p, h: m.gen[GEO_TECH, p, h] - m.reinjection[p, h]
            + m.orc_share[p, h] * q_in[h]
            <= m.q_geo_du_cap[h] - m.geo_delta_ci[h] * m.y_ci[p, h])

    # Series-HP CI coupling: shares the low-grade slice with the HTHP.
    M_rev = cfg["max_modules"] * mod_MW
    _q_hthp_max = float(model.technologies[HTHP_TECH]["max_output_MW"])
    m.revorc_ci_gate = pyo.Constraint(m.PERIODS, m.HOURS,
        rule=lambda m, p, h: m.gen[REVORC_TECH, p, h] <= M_rev * m.y_ci[p, h])
    m.revorc_ci_slice = pyo.Constraint(m.PERIODS, m.HOURS,
        rule=lambda m, p, h: m.gen[REVORC_TECH, p, h] + m.gen[HTHP_TECH, p, h]
            <= m.gen[GEO_TECH, p, h] * m.hthp_coupling_ratio[h]
               + (M_rev + _q_hthp_max) * (1 - m.y_ci[p, h]))

    # REVORC_EXCLUSIVE overrides the sheet, as in run_rd (N6 sets it to 0 so both
    # machines may be built). With the HTHP excluded the binary is redundant, and
    # presolve then removes it, so skip it rather than leave it unset.
    _excl = os.environ.get("REVORC_EXCLUSIVE")
    _excl = cfg.get("exclusive_with_hthp", 0) if _excl is None else float(_excl)
    if _excl:
        m.y_revorc = pyo.Var(domain=pyo.Binary)
        m.revorc_excl_rev = pyo.Constraint(m.PERIODS,
            rule=lambda m, p: m.cap_total[REVORC_TECH, p] <= M_rev * m.y_revorc)
        m.revorc_excl_hthp = pyo.Constraint(m.PERIODS,
            rule=lambda m, p: m.cap_total[HTHP_TECH, p] <= _q_hthp_max * (1 - m.y_revorc))
        logger.info("revORC↔HTHP investment exclusivity active")

    k_pump = float(cfg.get("brine_pump_el_per_MWth", 0.0))
    if k_pump > 0:
        m.orc_pump_extra = pyo.Var(m.PERIODS, m.HOURS, domain=pyo.NonNegativeReals)
        m.orc_pump_def = pyo.Constraint(m.PERIODS, m.HOURS,
            rule=lambda m, p, h: m.orc_pump_extra[p, h]
                >= m.orc_share[p, h] * q_in[h] - m.reinjection[p, h])

    # ORC revenue (+ pump cost) in the objective, same discounting as the model.
    _EPS = 1e-4
    # ORC-mode variable O&M is charged on generated electricity, so it lowers the
    # net revenue per MWh_el produced (run_rd parity).
    vom_orc = float(cfg.get("opex_var_orc_eur_per_MWh_el", 0.0))
    if vom_orc:
        logger.info(f"revORC ORC-mode variable O&M: {vom_orc:.2f} EUR/MWh_el on generation")
    base_yr = min(model.investment_years)
    plist = sorted(model.investment_years)
    extra = 0
    for i, p in enumerate(plist):
        yip = plist[i + 1] - p if i + 1 < len(plist) else 5
        r = model.discount_rate_by_period[p]
        pv = sum(1.0 / (1.0 + r) ** (p + y - base_yr) for y in range(yip))
        sell = model._sell_arr_by_period[p]
        buy  = model._elec_arr_by_period[p]
        for h in range(T):
            extra = extra + m.orc_share[p, h] * (
                (_EPS - p_net[h] * (float(sell[h]) - vom_orc)) * pv)
            if k_pump > 0:
                # Floor the purchase price at zero: orc_pump_extra is bounded from
                # below only, so a negative price would make extra pumping a reward
                # and the problem unbounded. This bites at full resolution and not
                # in the RD model, because clustering averages the 573 negative
                # hours away before the optimiser ever sees them.
                extra = extra + m.orc_pump_extra[p, h] * (
                    k_pump * max(float(buy[h]), 0.0) * pv)
    m.total_cost.expr = m.total_cost.expr + extra

    model._revorc = {"q_in": q_in, "p_net": p_net, "cfg": cfg}
    logger.info(
        f"revORC-full built (parallel ORC + SERIES HP, series-trained COP) — "
        f"≤ {cfg['max_modules']:.0f} × {mod_MW:.1f} MW_th modules, "
        f"ORC P_net {min(p_net.values())*1e3:.0f}–{max(p_net.values())*1e3:.0f} kW/module"
    )


def _read_grid_capacity_fee(tech_file):
    """Grid capacity fee [EUR/kW_el/a] from the RevORC sheet (as run_rd reads it).

    GRID_CAP_FEE overrides the sheet value; 0 disables the capacity charge while
    leaving the energy-based grid fee, which sits in the purchase price series.
    """
    fee = 0.0
    try:
        _rv = pd.read_excel(tech_file, sheet_name='RevORC')
        _hit = _rv[_rv['parameter'].astype(str).str.strip()
                   == 'grid_capacity_fee_eur_per_kW_yr']
        if len(_hit):
            fee = float(_hit['value'].iloc[0])
    except Exception as e:
        logger.warning(f"grid capacity fee not read from the RevORC sheet: {e}")
    if os.environ.get('GRID_CAP_FEE') is not None:
        _override = float(os.environ['GRID_CAP_FEE'])
        logger.warning(f"GRID_CAP_FEE override: {fee:.0f} -> {_override:.0f} EUR/kW_el/a")
        fee = _override
    return fee


def _add_grid_capacity_charge_full(model, fee_eur_per_kW_yr, k_pump=0.0):
    """Leistungspreis Netzkosten on the PEAK electricity draw of each period.

    Port of run_rd._add_grid_capacity_charge from (PERIODS, REP_DAYS, HOURS) to
    (PERIODS, HOURS). A capacity charge is levied on the highest power a site
    draws, not on its energy, so it is modelled with one peak variable per
    investment period:

        elec_peak[p] >= sum_t gen[t,p,h] / COP[t,h] + k_pump * pump draw   for all h

    and the objective carries elec_peak[p] * 1000 * fee, discounted over the
    period. Unlike a charge on installed capacity this lets the optimiser shave
    its peak, which is what a real site would do.
    """
    if fee_eur_per_kW_yr <= 0:
        return
    m = model.model
    elec_techs = [t for t in m.TECH
                  if str(pyo.value(m.tech_carrier[t])).strip().lower().startswith(
                      ("electricity", "ambient heat", "waste heat"))]
    if not elec_techs:
        logger.info("grid capacity charge skipped: no electricity-drawing technology")
        return

    m.elec_peak = pyo.Var(m.PERIODS, domain=pyo.NonNegativeReals)   # MW_el
    has_pump = hasattr(m, "orc_pump_extra") and k_pump > 0

    def _peak_rule(mm, p_, h_):
        draw = sum(mm.gen[t, p_, h_] / float(pyo.value(mm.efficiency[t, h_]))
                   for t in elec_techs
                   if float(pyo.value(mm.efficiency[t, h_])) > 1e-9)
        if has_pump:
            draw = draw + k_pump * mm.orc_pump_extra[p_, h_]
        return mm.elec_peak[p_] >= draw
    m.elec_peak_def = pyo.Constraint(m.PERIODS, m.HOURS, rule=_peak_rule)

    plist = sorted(model.investment_years)
    base_yr = min(plist)
    charge = 0.0
    for i, p_ in enumerate(plist):
        yrs = plist[i + 1] - p_ if i + 1 < len(plist) else 5
        r_ = model.discount_rate_by_period[p_]
        pv = sum(1.0 / (1.0 + r_) ** (p_ + y - base_yr) for y in range(yrs))
        charge = charge + m.elec_peak[p_] * 1000.0 * fee_eur_per_kW_yr * pv
    m.total_cost.expr = m.total_cost.expr + charge
    logger.info(
        f"Grid capacity charge active: {fee_eur_per_kW_yr:.0f} EUR/kW_el/a on the peak "
        f"draw of {elec_techs}{' + brine pump' if has_pump else ''} "
        f"({len(m.PERIODS) * 8760} constraints)")


def main():
    root = _CASE_DIR
    ts   = datetime.now().strftime('%Y%m%d_%H%M%S')
    # Scenario selector, mirroring run_rd: S1 = no storage, S2 = HT-BTES
    # (hourly ARX chain via add_btes_to_model — exact, not the daily RD
    # aggregation), S3 = pit/PTES.
    SCENARIO = os.environ.get('SCENARIO', 'S1').upper()
    # RESULTS_SUBDIR redirects output under results/<subdir>/ (default 'full'),
    # so screening sweeps do not clutter the main run folder. RUN_TAG is appended
    # to the folder name.
    _sub = os.environ.get('RESULTS_SUBDIR', 'full')
    _tag = os.environ.get('RUN_TAG', '')
    _em = float(os.environ.get('ELEC_MULT', '1') or '1')
    if _em != 1.0:
        _tag += f"_E{round(_em*100)}"
    results_dir = root / 'results' / _sub / f'MILP_{ts}_{SCENARIO}{_tag}'
    results_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Munich FULL 8760-h model [{SCENARIO}] — output folder: {results_dir}")

    # Input workbooks come from the environment, exactly as in run_rd, so the two
    # resolutions can be pointed at the same data set.
    tech_file = root / 'data' / os.environ.get('TECH_DATA', 'Munich_tech_data.xlsx')
    demand_file = root / 'data' / os.environ.get('DH_DATA', 'Munich_DH_data.xlsx')
    if os.environ.get('TECH_DATA') or os.environ.get('DH_DATA'):
        logger.warning(f"Input override: tech={tech_file.name}, demand={demand_file.name}")
    techs, stor, prices, demand, extra, hp = load_all_munich_data(
        tech_file=tech_file, demand_file=demand_file)

    # Network-temperature sensitivity (env NET_TEMP_SHIFT_K): shift the measured hourly
    # supply and return temperature series by -K. Affects the geothermal direct-use
    # ceiling, HP COPs and the BTES discharge rule alike (the BTES surrogate itself is
    # left as fitted at 55/80 C; only the dispatch-side temperatures move).
    _shift = float(os.environ.get('NET_TEMP_SHIFT_K', 0) or 0)
    if _shift:
        for _c in ('supply_temp', 'return_temp', 'supply_temp_fixed', 'return_temp_fixed'):
            if _c in demand.columns:
                demand[_c] = demand[_c] - _shift
        logger.warning(f"NET_TEMP_SHIFT_K={_shift:g}: supply now "
                       f"{demand['supply_temp'].min():.0f}-{demand['supply_temp'].max():.0f} C, return "
                       f"{demand['return_temp'].min():.0f}-{demand['return_temp'].max():.0f} C")

    # Electricity-price sensitivity (env ELEC_MULT): scale the whole buy AND sell
    # trajectory by a factor, preserving the hourly shape and the sell/buy ratio,
    # so the ORC and the heat pump still see a real price signal. Same flag and
    # same meaning as in run_rd.py.
    ELEC_MULT = float(os.environ.get('ELEC_MULT', '1') or '1')
    if ELEC_MULT != 1.0:
        _cols = [c for c in demand.columns
                 if c.startswith('electricity_price_')
                 or c.startswith('sell_electricity_price_')]
        for c in _cols:
            demand[c] = demand[c] * ELEC_MULT
        logger.warning(
            f"ELEC_MULT={ELEC_MULT:g}: buy and sell electricity prices scaled "
            f"({len(_cols)} columns); hourly shape and sell/buy ratio preserved")

    BTES_BUILD_PERIOD = None
    EXCLUDE_HTHP = False
    if SCENARIO == 'S1':                      # legacy suite: storage only varies
        stor = {}
    elif SCENARIO == 'S2':
        stor = {}
        BTES_BUILD_PERIOD = 2025
    elif SCENARIO in ('N1', 'N2', 'N3', 'N4', 'N5', 'N6'):
        # New suite (paper v2), identical gating to run_rd: every scenario shares
        # geothermal + gas boiler; one flexibility option is added at a time.
        #   N1 base | N2 +revORC | N3 +HTHP | N4 +HT-BTES | N5 +PTES | N6 all
        # The short-term buffer is not part of the case (KEEP_BUFFER=1 restores it).
        _keep_buffer = os.environ.get('KEEP_BUFFER') == '1'
        stor = {k: v for k, v in stor.items()
                if ('buffer' not in k.lower() or _keep_buffer)
                and ('buffer' in k.lower() or SCENARIO in ('N5', 'N6'))}
        BTES_BUILD_PERIOD = 2025 if SCENARIO in ('N4', 'N6') else None
        # BTES_OFF=1 gives the no-field member of the design enumeration,
        # matching run_rd. Without it the default parameter file is loaded
        # and the run silently builds a 200-borehole field.
        if os.environ.get('BTES_OFF') == '1':
            BTES_BUILD_PERIOD = None
            logger.info('BTES_OFF=1 - BTES disabled for this run')
        if SCENARIO not in ('N2', 'N6'):
            os.environ['REVORC'] = '0'
        if SCENARIO == 'N6':
            os.environ.setdefault('REVORC_EXCLUSIVE', '0')
        EXCLUDE_HTHP = SCENARIO in ('N1', 'N2', 'N4', 'N5')
        if EXCLUDE_HTHP:
            # nothing to be exclusive against: the binary would be redundant and
            # presolve drops it, leaving it unset at reporting time
            os.environ['REVORC_EXCLUSIVE'] = '0'
    logger.info(f"Scenario {SCENARIO}: BTES={BTES_BUILD_PERIOD}, storages={list(stor) or 'none'}, "
                f"revORC={'off' if os.environ.get('REVORC', '1') == '0' else 'on'}, "
                f"HTHP={'excluded' if EXCLUDE_HTHP else 'investable'}")
    INVESTMENT_YEARS = extra['investment_years']
    logger.info(f"Investment years: {INVESTMENT_YEARS}")

    # ── revORC (reversible HP/ORC), same config sheet as run_rd ──
    revorc_cfg = None
    if os.environ.get('REVORC', '1') != '0':
        revorc_cfg = load_revorc_config(tech_file)
    if revorc_cfg:
        _inject_revorc_tech(techs, revorc_cfg)

    # ── SAPHEA resource-driven capacities (override tech-sheet ratings) ──
    resource = extra['geo_resource']
    design   = derive_saphea_capacities(resource)
    techs[GEO_TECH]['max_output_MW']         = design['Q_geo_du']
    techs[GEO_TECH]['installed_capacity_MW'] = design['Q_geo_du']
    techs[HTHP_TECH]['max_output_MW']        = design['Q_hthp_max']
    capex_curve = extra.get('hthp_capex_curve') or []
    if capex_curve and os.environ.get('HTHP_CAPEX_EXCEL') == '1':
        logger.info(f"  HTHP_CAPEX_EXCEL=1: keeping the tech-sheet CAPEX "
                    f"{float(techs[HTHP_TECH]['capex_EUR_per_MW']):,.0f} EUR/MW_th")
        capex_curve = []
    if capex_curve:
        techs[HTHP_TECH]['capex_EUR_per_MW'] = derive_hthp_capex(
            capex_curve, extra.get('hthp_capex_scale', 0.8),
            design['Q_evap'], design['Q_hthp_max'])

    # ── Geothermal unit-commitment: shutdown allowed, 4-week restart ──
    if ALLOW_GEO_SHUTDOWN:
        techs[GEO_TECH]['cold_start_time_h'] = GEO_COLD_START_H
        techs[GEO_TECH]['min_down_h']        = GEO_MIN_DOWN_H
        logger.info(f"Geothermal shutdown ENABLED: cold-start {GEO_COLD_START_H} h, "
                    f"min-down {GEO_MIN_DOWN_H} h, min-load "
                    f"{techs[GEO_TECH].get('min_load_factor', 0):.2f}")

    uc = build_uc_params(techs)
    logger.info(f"MILP/UC mode: {'ENABLED' if uc else 'DISABLED (LP)'}")

    # Dynamic HTHP COP (SAPHEA, supply-temp driven) injected before build so the
    # electricity-cost coefficients use it. The revORC-HP gets its OWN hourly COP
    # from the series-trained FLEXGEO surrogate (below) — both are then charged
    # hourly at buy_price/COP by the precomputed coefficients.
    cop_profile = _saphea_cop_profile(demand, resource) if HTHP_TECH in techs else None
    eff_profiles = {}
    if cop_profile is not None:
        eff_profiles[HTHP_TECH] = cop_profile
        if revorc_cfg:
            # The revORC HP mode runs on the SERIES-trained FLEXGEO surrogate COP,
            # exactly as run_rd does — not the SAPHEA HTHP COP, which belongs to a
            # machine on a different brine slice (mean 4.40 vs 3.15, i.e. 30 % less
            # electricity per MWh_th). It has to be known HERE and not in
            # _build_revorc_full, because full_model.build() precomputes the
            # variable-cost coefficients numerically: a post-build set_value() on
            # m.efficiency would never reach the objective. The maps are disk-cached
            # on a hash of the input profiles, so this second call is free.
            _hp_maps = build_revorc_maps(
                demand['ambient_temp'].values[:8760].reshape(365, 24),
                demand['supply_temp'].values[:8760].reshape(365, 24),
                demand['return_temp'].values[:8760].reshape(365, 24),
                resource, revorc_cfg)
            eff_profiles[REVORC_TECH] = np.array(
                [_hp_maps['hp_cop'][(d, h)] for d in range(365) for h in range(24)])
            logger.info(
                f"revORC HP mode uses the series-trained surrogate COP "
                f"(hourly mean {eff_profiles[REVORC_TECH].mean():.2f}, range "
                f"{eff_profiles[REVORC_TECH].min():.2f}-{eff_profiles[REVORC_TECH].max():.2f}); "
                f"the SAPHEA HTHP COP is not substituted")

    btes_json = root / 'data' / 'btes' / 'btes_hourly_params.json'
    # BTES_DESIGN_N picks one field from the design library, matching run_rd's
    # single-design route (the full model has no endogenous design selection).
    _dn = os.environ.get('BTES_DESIGN_N')
    if BTES_BUILD_PERIOD and _dn:
        _cand = root / 'data' / 'btes' / f"btes_hourly_params{os.environ.get('BTES_LIB_SUFFIX', '')}_n{_dn}.json"
        if _cand.exists():
            btes_json = _cand
            logger.info(f"BTES design {_dn}: using {_cand.name}")
        else:
            raise SystemExit(f"BTES_DESIGN_N={_dn}: {_cand.name} not found")
    model = DistrictHeatingModelFull(
        demand_data=demand, technologies=techs, storage=stor, prices=prices,
        investment_years=INVESTMENT_YEARS, extra_params=extra,
        hp_params=hp if hp else None, uc_params=uc,
        efficiency_profiles=eff_profiles or None,
        btes_build_period=BTES_BUILD_PERIOD,
        btes_arx_json=btes_json if BTES_BUILD_PERIOD else None,
        name='MunichDH_Full',
    )
    logger.info("Building full model (this constructs 8760 h × periods constraints)…")
    t_build = time.perf_counter()
    model.build()

    # ── Reinjection is brine returned to the well, so it cannot exceed the heat the
    #    doublet actually produced. Without this bound any generation may be dumped,
    #    which turns negative-price hours into a reward for running electric units
    #    and reinjecting their output (and an inflated reinjection also cancels the
    #    ORC brine-pump charge). Same constraint as run_rd / run_rh. ──
    _m = model.model
    if GEO_TECH in set(_m.TECH):
        def _reinject_cap_rule(mm, p_, h_):
            return mm.reinjection[p_, h_] <= mm.gen[GEO_TECH, p_, h_]
        _m.reinject_le_geo = pyo.Constraint(
            _m.PERIODS, _m.HOURS, rule=_reinject_cap_rule)
        logger.info("Reinjection bounded by geothermal generation "
                    f"({len(_m.PERIODS) * 8760} constraints)")

    _build_saphea_coupling_full(model, demand, resource)
    if revorc_cfg:
        _build_revorc_full(model, demand, resource, revorc_cfg)

    # Leistungspreis Netzkosten on the peak electrical draw. Called after the
    # SAPHEA and revORC blocks so the hourly COPs are final.
    _add_grid_capacity_charge_full(
        model, _read_grid_capacity_fee(tech_file),
        float((revorc_cfg or {}).get('brine_pump_el_per_MWth', 0.0)))
    logger.info(f"Model built in {time.perf_counter() - t_build:.1f} s")

    if EXCLUDE_HTHP:
        for _p in model.investment_years:
            model.model.cap_add[HTHP_TECH, _p].fix(0)
        logger.info(f"HTHP investment fixed to 0 (scenario {SCENARIO})")

    if not ALLOW_GEO_SHUTDOWN:
        m = model.model
        if hasattr(m, 'u') and GEO_TECH in set(m.UC_TECH):
            for p in m.PERIODS:
                for h in m.HOURS:
                    m.u[GEO_TECH, p, h].fix(1)
            logger.info("Geothermal forced always-on (must-run).")

    # ── MIP start from the matching RD run (WARM_START=<results_rd.xlsx or run dir>) ──
    # N6 is primal-hard: in a 3-h run the bound moved 0.2 % while the heuristics
    # never found anything better than 5x the true cost. Seeding the capacities
    # from the aggregated solution gives Gurobi a feasible point to improve on
    # instead of hunting for one in a 56k-binary space.
    _ws = os.environ.get('WARM_START')
    warmstart = False
    if _ws:
        _wp = Path(_ws)
        if _wp.is_dir():
            _wp = _wp / 'results_rd.xlsx'
        if not _wp.exists():
            raise SystemExit(f"WARM_START: {_wp} not found")
        _cap = pd.read_excel(_wp, sheet_name='capacity').set_index('Year')
        m = model.model
        _n = 0
        for _t in m.TECH:
            if _t not in _cap.columns:
                continue
            _prev = 0.0
            for _p in sorted(model.investment_years):
                _tot = float(_cap.at[_p, _t]) if _p in _cap.index else _prev
                m.cap_total[_t, _p].set_value(_tot, skip_validation=True)
                if not m.cap_add[_t, _p].fixed:
                    m.cap_add[_t, _p].set_value(max(_tot - _prev, 0.0), skip_validation=True)
                _prev = _tot
                _n += 1
        warmstart = True
        logger.warning(f"WARM_START from {_wp.name}: seeded {_n} capacity values "
                       f"({', '.join(c for c in _cap.columns if c in set(m.TECH))})")

    # ── BTES MIP start (BTES_MIPSTART=1): the trivially feasible "field built, never used"
    # point — Qch = Qdis = 0, gate binaries 0, store at ground temperature. Under the
    # temperature rule Gurobi's heuristics take minutes to find any BTES-feasible incumbent;
    # this gives it one at t=0 and it improves from there. Composes with WARM_START.
    if os.environ.get('BTES_MIPSTART') and getattr(model, 'btes_enabled', False):
        m = model.model
        _pp = model._btes_period_params
        _first = min(p for p, v in _pp.items() if v.get('active'))
        _Tg = float(_pp[_first]['T_init'])
        _n = 0
        for _var in (m.btes_Qch, m.btes_Qdis):
            for _k in _var:
                _var[_k].set_value(0.0, skip_validation=True); _n += 1
        if hasattr(m, 'btes_dir'):
            for _k in m.btes_dir:
                m.btes_dir[_k].set_value(0, skip_validation=True); _n += 1
        if hasattr(m, 'btes_lvl'):
            for _k in m.btes_lvl:
                m.btes_lvl[_k].set_value(0, skip_validation=True); _n += 1
        for _k in m.btes_T:
            m.btes_T[_k].set_value(_Tg, skip_validation=True); _n += 1
        warmstart = True
        logger.warning(f"BTES_MIPSTART: seeded {_n} BTES variables with the never-discharge point (T = {_Tg} C)")

    _solfiles = str(results_dir / 'incumbent')

    t0 = time.perf_counter()
    try:
        globals()["_IN_SOLVE"] = True
        results_raw = model.solve(
            solver_name=get_solver_name(), tee=True, warmstart=warmstart,
            MIPGap=SOLVE_MIPGAP, MIPFocus=1, Heuristics=0.20, Cuts=2,
            NoRelHeurTime=300, SolFiles=_solfiles, LogFile=_solfiles.rsplit('/', 1)[0] + '/gurobi.log',
            TimeLimit=SOLVE_TIMELIMIT_S, Threads=GRB_THREADS,
        )
    except KeyboardInterrupt:
        logger.warning("Interrupted — saving best incumbent …")
        results_raw = None
    # solve window closed: from here on a stop signal only logs,
    # because interrupting the write-up is what actually loses results
    globals()["_IN_SOLVE"] = False
    wall = time.perf_counter() - t0
    logger.info(f"Solve wall time: {wall:.0f} s")

    if results_raw is None:
        _sols = sorted(results_dir.glob('incumbent*.sol'))
        logger.warning(
            f"No solution object was returned. {len(_sols)} incumbent file(s) were "
            f"written by Gurobi during the solve"
            + (f"; the last is {_sols[-1].name}" if _sols else "")
            + ". The reported values below are only meaningful if a solution was "
              "loaded — check them before use.")


    # If the solve was stopped before Gurobi found any feasible point there is
    # nothing to report: the variables are uninitialised and every downstream
    # export would raise. Say so and stop cleanly rather than traceback.
    try:
        pyo.value(next(iter(model.model.component_data_objects(pyo.Var))))
    except Exception:
        _sols = sorted(results_dir.glob('incumbent*.sol'))
        logger.error(
            "Stopped before any feasible solution existed — nothing to report. "
            + (f"{len(_sols)} incumbent file(s) are on disk ({_sols[-1].name})."
               if _sols else "Gurobi wrote no incumbent files either.")
            + " Re-run with a longer time limit, or stop later.")
        return

    _save_full_results(model, results_dir, wall)

    # NO_REPORT=1 stops here: summary.csv (objective, bound, gap, wall) and the
    # capacity/geo CSVs are written, but the plot and Excel pass is skipped. Used
    # by the BTES design screening, where only the objective is compared.
    if os.environ.get('NO_REPORT') == '1':
        logger.info(f"NO_REPORT=1 — skipping plots/Excel. Results: {results_dir}")
        return

    # ── Full RD-style reporting via the FullAsRD adapter: every plot/CSV/xlsx
    #    that run_rd produces is generated here too, from the SAME code —
    #    the full model is treated as 365 rep-days of weight 1. ──
    import run_rd as rd
    from full_reporting import FullAsRD
    rd.INVESTMENT_YEARS = list(INVESTMENT_YEARS)
    rd.N_TYPICAL_DAYS = 365
    rd.BTES_BUILD_PERIOD = BTES_BUILD_PERIOD
    adapter = FullAsRD(model, demand)
    res = model.get_results()

    if revorc_cfg:
        rd._report_revorc(adapter, results_dir)
        rd.save_revorc_price_analysis(adapter, results_dir)

    logger.info("Saving RD-style plots and exports (via adapter)…")
    rd.save_dispatch_plots(adapter, results_dir)
    rd.save_winter_dispatch_and_csv(adapter, results_dir)
    rd.save_rep_dispatch(adapter, results_dir)
    rd.save_combined_dispatch_plot(adapter, results_dir)
    rd.save_soc_plots(adapter, results_dir)
    rd.save_btes_plots(adapter, results_dir)
    try:                                  # the webinar-style operation figure, for every HT-BTES run
        rd.save_btes_timeseries(adapter, results_dir)
        rd.save_btes_operation_plot(adapter, results_dir)
    except Exception as exc:
        logger.warning('BTES operation figure skipped: %s', exc)
    rd.save_capacity_plot(res, results_dir)
    rd.save_generation_plot(res, results_dir)
    rd.save_mode_plots(adapter, results_dir)
    comp_params = {}
    if results_raw is not None:
        try:
            rd.save_computational_params(adapter, results_raw, wall, results_dir)
            comp_params = rd.collect_computational_params(adapter, results_raw, wall)
        except Exception as e:
            logger.warning(f"computational params skipped: {e}")
    rd.export_excel(adapter, res, results_dir, comp_params)
    try:
        model.compute_lcoh().to_csv(results_dir / 'lcoh_by_period.csv')
        logger.info("LCOH saved")
    except Exception as e:
        logger.warning(f"LCOH skipped: {e}")

    logger.info(f"Done. Results saved to: {results_dir}")


def _save_full_results(model, results_dir, wall):
    """Extract NPV + geothermal-operation diagnostics (on/off, reinjection)."""
    m = model.model
    try:
        npv = float(pyo.value(m.total_cost))
    except Exception as e:
        logger.error(f"Could not read objective: {e}")
        return
    logger.info(f"Objective (NPV): EUR {npv:,.0f}")

    years = list(model.investment_years)
    rows = []
    has_u = hasattr(m, 'u') and GEO_TECH in set(getattr(m, 'UC_TECH', []))
    for p in years:
        geo = np.array([pyo.value(m.gen[GEO_TECH, p, h]) for h in m.HOURS])
        gas = np.array([pyo.value(m.gen['Gas Boiler', p, h]) for h in m.HOURS]) \
              if 'Gas Boiler' in model.technologies else np.zeros(len(geo))
        hthp = np.array([pyo.value(m.gen[HTHP_TECH, p, h]) for h in m.HOURS]) \
               if HTHP_TECH in model.technologies else np.zeros(len(geo))
        reinj = np.array([pyo.value(m.reinjection[p, h]) for h in m.HOURS])
        if has_u:
            u = np.array([round(pyo.value(m.u[GEO_TECH, p, h])) for h in m.HOURS])
            vstop = np.array([round(pyo.value(m.v_stop[GEO_TECH, p, h])) for h in m.HOURS])
        else:
            u = np.ones(len(geo)); vstop = np.zeros(len(geo))
        row = {
            'year': p,
            'geo_gen_GWh': geo.sum() / 1e3,
            'gas_gen_GWh': gas.sum() / 1e3,
            'hthp_gen_GWh': hthp.sum() / 1e3,
            'reinjection_GWh': reinj.sum() / 1e3,
            'geo_on_hours': int(u.sum()),
            'geo_off_hours': int(len(u) - u.sum()),
            'geo_shutdown_events': int(vstop.sum()),
            'high_reinjection_hours': int((reinj > 1.0).sum()),
        }
        if hasattr(model, '_revorc'):
            p_net = model._revorc['p_net']
            q_in  = model._revorc['q_in']
            mod_MW = model._revorc['cfg']['module_hp_MWth']
            share = np.array([pyo.value(m.orc_share[p, h]) for h in m.HOURS])
            pn    = np.array([p_net[h] for h in m.HOURS])
            qi    = np.array([q_in[h] for h in m.HOURS])
            sell  = np.asarray(model._sell_arr_by_period[p], dtype=float)
            rev_gen = np.array([pyo.value(m.gen[REVORC_TECH, p, h]) for h in m.HOURS])
            row.update({
                'revorc_modules': round(float(pyo.value(m.cap_total[REVORC_TECH, p])) / mod_MW, 2),
                'orc_elec_GWh': float((share * pn).sum()) / 1e3,
                'orc_brine_GWh': float((share * qi).sum()) / 1e3,
                'orc_revenue_kEUR': float((share * pn * sell).sum()) / 1e3,
                'orc_hours': int((share * pn > 1e-6).sum()),
                'hp_heat_GWh': rev_gen.sum() / 1e3,
            })
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(results_dir / 'geo_operation_summary.csv', index=False)
    pick = None
    if hasattr(m, 'y_revorc') and m.y_revorc.value is not None:
        pick = 'revORC' if pyo.value(m.y_revorc) > 0.5 else 'HTHP'
        logger.info(f"revORC↔HTHP exclusivity: optimizer picked {pick}")
    pd.DataFrame([{'objective_NPV_EUR': npv, 'wall_time_s': round(wall),
                   'geo_shutdown_allowed': ALLOW_GEO_SHUTDOWN,
                   'scenario': os.environ.get('SCENARIO', 'S1').upper(),
                   'mipgap': SOLVE_MIPGAP, 'excl_pick': pick}]).to_csv(
        results_dir / 'summary.csv', index=False)
    logger.info("\n" + df.to_string(index=False))

    # Save the 2045 dispatch for inspection
    p = years[-1]
    disp = {'Hour': list(m.HOURS),
            'Demand_MW': [pyo.value(m.demand[p, h]) for h in m.HOURS],
            'Geothermal Heat': [pyo.value(m.gen[GEO_TECH, p, h]) for h in m.HOURS],
            'Reinjection_MW': [pyo.value(m.reinjection[p, h]) for h in m.HOURS]}
    if 'Gas Boiler' in model.technologies:
        disp['Gas Boiler'] = [pyo.value(m.gen['Gas Boiler', p, h]) for h in m.HOURS]
    if has_u:
        disp['geo_on'] = [round(pyo.value(m.u[GEO_TECH, p, h])) for h in m.HOURS]
    pd.DataFrame(disp).to_csv(results_dir / f'dispatch_{p}.csv', index=False)


if __name__ == '__main__':
    main()
