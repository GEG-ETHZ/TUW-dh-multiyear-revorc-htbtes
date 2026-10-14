"""SAPHEA geothermal coupling and the reversible HP/ORC, at RH resolution.

Ports the two Munich-specific mechanisms from run_rd.py's (period, rep_day, hour)
index space to the representative-hour space (period, rep_hour). The physics is
unchanged — the same SAPHEA formulas and the same FLEXGEO surrogates — only the
slot index and the time weights differ.

What a representative hour changes:
  * weights. A rep day carries `day_weight` days of 24 h; a rep hour carries
    `weight` hours. Every annual sum uses w[i] directly.
  * adjacency. The 24 hours of a rep day ARE consecutive real hours, so the
    revORC dead-hour rule is well defined there. Representative hours are not
    adjacent in real time, so the rule is only applied under SEGMENTED
    clustering, where interval j-1 and j are contiguous blocks. In free RH it
    is skipped with a warning rather than silently approximated.

    from models.saphea_revorc_rh import build_saphea_rh, build_revorc_rh
"""

from typing import Dict, List

import numpy as np
import pyomo.environ as pyo
from loguru import logger


def rep_hour_profile(values: np.ndarray, sigma: List[int], n_rep: int) -> np.ndarray:
    """Average an 8760-h array onto the representative-hour index."""
    sums = np.zeros(n_rep)
    cnt = np.zeros(n_rep)
    for t, i in enumerate(sigma):
        sums[i] += values[t]
        cnt[i] += 1
    return sums / np.maximum(cnt, 1)


def build_saphea_rh(model, demand, resource, geo_tech, hthp_tech, design):
    """SAPHEA three-mode geothermal/HTHP coupling on (period, rep_hour).

    Mirrors run_rd._build_saphea_coupling: DU / CI / TI selected by y_ci, y_ti,
    with the geothermal ceiling and the HTHP coupling ratio both driven by the
    hourly supply and return temperatures.
    """
    m = model.model
    years = sorted(model.investment_years)
    I = len(m.REP_HOURS)

    T_prod = float(resource["T_prod"])
    T_min = float(resource["T_inject_min"])
    PINCH = float(resource["direct_hx_pinch"])
    eta = float(resource.get("carnot_eta", 0.5))
    mdot = float(resource["brine_mass_flow"])
    cp = float(resource.get("brine_cp", 4.2))

    def _q_brine(dT):
        return mdot * cp * dT / 1000.0            # MW

    Q_geo_du_design = float(design["Q_geo_du"])
    Q_evap_ti = _q_brine(T_prod - T_min)

    sup = demand["supply_temp"].to_numpy()[:8760]
    ret = demand["return_temp"].to_numpy()[:8760]

    coupling, q_du_map, delta_map, ti_req, hp_ti = {}, {}, {}, {}, {}
    cops = []
    for p in years:
        sigma = model.rh_by_period[p]["sigma"]
        s_i = rep_hour_profile(sup, sigma, I)
        r_i = rep_hour_profile(ret, sigma, I)
        for i in range(I):
            T_s, T_ret = float(s_i[i]), float(r_i[i])
            T_woHP = T_ret + PINCH
            q_du = min(_q_brine(T_prod - T_ret), Q_geo_du_design)
            q_ci = _q_brine(T_prod - T_woHP)
            q_du_map[(p, i)] = q_du
            delta_map[(p, i)] = q_du - q_ci
            cop_s = eta * (T_s + 273.15) / max(T_s - T_min, 1.0)
            cop_std = cop_s + 1.0
            coupling[(p, i)] = ((T_woHP - T_min) / (T_prod - T_woHP)) * (1.0 + 1.0 / cop_s)
            hp_ti[(p, i)] = Q_evap_ti * (1.0 + 1.0 / cop_s)
            ti_req[(p, i)] = 1 if T_s > T_prod else 0
            if hthp_tech in model.technologies:
                m.efficiency[hthp_tech, p, i].set_value(cop_std)
            cops.append(cop_std)

    m.q_geo_du_cap = pyo.Param(m.PERIODS, m.REP_HOURS, initialize=q_du_map)
    m.geo_delta_ci = pyo.Param(m.PERIODS, m.REP_HOURS, initialize=delta_map)
    m.hthp_coupling_ratio = pyo.Param(m.PERIODS, m.REP_HOURS, initialize=coupling)
    m.hthp_hp_max_ti = pyo.Param(m.PERIODS, m.REP_HOURS, initialize=hp_ti)
    m.ti_required = pyo.Param(m.PERIODS, m.REP_HOURS, initialize=ti_req, within=pyo.Binary)

    m.y_ci = pyo.Var(m.PERIODS, m.REP_HOURS, domain=pyo.Binary)
    m.y_ti = pyo.Var(m.PERIODS, m.REP_HOURS, domain=pyo.Binary)

    m.mode_mutex = pyo.Constraint(m.PERIODS, m.REP_HOURS,
        rule=lambda m, p, i: m.y_ci[p, i] + m.y_ti[p, i] <= 1)
    m.ti_force = pyo.Constraint(m.PERIODS, m.REP_HOURS,
        rule=lambda m, p, i: m.y_ti[p, i] >= m.ti_required[p, i])
    m.ti_block = pyo.Constraint(m.PERIODS, m.REP_HOURS,
        rule=lambda m, p, i: m.y_ti[p, i] <= m.ti_required[p, i])
    m.ci_block_ti = pyo.Constraint(m.PERIODS, m.REP_HOURS,
        rule=lambda m, p, i: m.y_ci[p, i] <= 1 - m.ti_required[p, i])

    # geothermal ceiling: DU cap, less the pinch-delta budget in CI, zero in TI
    m.geo_mode_cap = pyo.Constraint(m.PERIODS, m.REP_HOURS,
        rule=lambda m, p, i: m.gen[geo_tech, p, i]
        <= m.q_geo_du_cap[p, i] * (1 - m.y_ti[p, i])
           - m.geo_delta_ci[p, i] * m.y_ci[p, i])

    if hthp_tech in model.technologies:
        q_hthp_max = float(model.technologies[hthp_tech]["max_output_MW"])
        if q_hthp_max > 0:
            m.hthp_mode_off = pyo.Constraint(m.PERIODS, m.REP_HOURS,
                rule=lambda m, p, i: m.gen[hthp_tech, p, i]
                <= q_hthp_max * (m.y_ci[p, i] + m.y_ti[p, i]))
            m.hthp_ci_couple = pyo.Constraint(m.PERIODS, m.REP_HOURS,
                rule=lambda m, p, i: m.gen[hthp_tech, p, i]
                <= m.gen[geo_tech, p, i] * m.hthp_coupling_ratio[p, i]
                   + q_hthp_max * (1 - m.y_ci[p, i]))
    # run_rd's mode/Excel reporters read model._saphea_design
    model._saphea_design = design
    logger.info(f"SAPHEA (RH): DU/CI/TI on {len(years)}x{I} slots, "
                f"HTHP COP {min(cops):.2f}-{max(cops):.2f}")


def build_revorc_rh(model, maps_by_p, cfg, geo_tech, hthp_tech, revorc_tech,
                    segmented: bool):
    """Reversible HP/ORC on (period, rep_hour).

    Same fleet formulation as run_rd._build_revorc:
        orc_share + hp_share <= n_mod          (one machine, one mode)
        gen[revORC]          <= hp_share*mod   (HP-mode heat)
        gen[revORC]          <= M*y_ci         (series HP implies CI mode)
        geo - reinject + orc_share*q_in <= geo ceiling   (ORC brine draw)
    with the ORC's revenue and the brine-pump cost added to the objective,
    weighted by the hours each representative hour stands for.
    """
    m = model.model
    years = sorted(model.investment_years)
    I = len(m.REP_HOURS)
    mod_MW = float(cfg["module_hp_MWth"])
    n_max = float(cfg["max_modules"])

    q_in_p = {p: maps_by_p[p]["orc_q_in_MW"] for p in years}
    p_net_p = {p: maps_by_p[p]["orc_p_net_MW"] for p in years}

    m.orc_share = pyo.Var(m.PERIODS, m.REP_HOURS, domain=pyo.NonNegativeReals)
    m.hp_share = pyo.Var(m.PERIODS, m.REP_HOURS, domain=pyo.NonNegativeReals)

    # one fleet, two modes
    m.revorc_fleet = pyo.Constraint(m.PERIODS, m.REP_HOURS,
        rule=lambda m, p, i: m.orc_share[p, i] + m.hp_share[p, i]
        <= m.cap_total[revorc_tech, p] / mod_MW)
    m.revorc_hp_cap = pyo.Constraint(m.PERIODS, m.REP_HOURS,
        rule=lambda m, p, i: m.gen[revorc_tech, p, i] <= m.hp_share[p, i] * mod_MW)

    # hourly HP-mode COP from the surrogate
    for p in years:
        for i in range(I):
            cop = maps_by_p[p]["hp_cop"].get((i, 0), 3.0)
            m.efficiency[revorc_tech, p, i].set_value(max(float(cop), 1e-3))

    # ORC brine draw sits under the geothermal ceiling, alongside direct use
    has_reinject = hasattr(m, "reinject")

    def _brine(mm, p, i):
        served = mm.gen[geo_tech, p, i] - (mm.reinject[p, i] if has_reinject else 0)
        return (served + mm.orc_share[p, i] * q_in_p[p].get((i, 0), 0.0)
                <= mm.q_geo_du_cap[p, i] - mm.geo_delta_ci[p, i] * mm.y_ci[p, i])
    m.revorc_brine = pyo.Constraint(m.PERIODS, m.REP_HOURS, rule=_brine)

    # series HP implies capacity-increase mode
    M_rev = n_max * mod_MW
    m.revorc_ci_gate = pyo.Constraint(m.PERIODS, m.REP_HOURS,
        rule=lambda m, p, i: m.gen[revorc_tech, p, i] <= M_rev * m.y_ci[p, i])

    if hthp_tech in model.technologies:
        q_hthp = float(model.technologies[hthp_tech]["max_output_MW"])
        m.revorc_ci_slice = pyo.Constraint(m.PERIODS, m.REP_HOURS,
            rule=lambda m, p, i: m.gen[revorc_tech, p, i] + m.gen[hthp_tech, p, i]
            <= m.gen[geo_tech, p, i] * m.hthp_coupling_ratio[p, i]
               + (M_rev + q_hthp) * (1 - m.y_ci[p, i]))

    # dead-hour switching rule: only meaningful when rep hours are contiguous
    dead = int(cfg.get("dead_hours", 1))
    if dead > 0 and segmented:
        m.revorc_dead = pyo.ConstraintList()
        for p in years:
            fleet = m.cap_total[revorc_tech, p] / mod_MW
            for i in range(I):
                for k in range(1, dead + 1):
                    j = (i - k) % I
                    m.revorc_dead.add(m.orc_share[p, i] + m.hp_share[p, j] <= fleet)
                    m.revorc_dead.add(m.hp_share[p, i] + m.orc_share[p, j] <= fleet)
        logger.info(f"revORC dead-hour rule applied across {dead} adjacent interval(s)")
    elif dead > 0:
        logger.warning(
            "revORC dead-hour rule SKIPPED: representative hours are not adjacent "
            "in real time under free clustering. Use RH_SEGMENTED=1 to enforce it "
            "(the model otherwise over-states mode-switching flexibility).")

    # brine-pump electricity for ORC draw beyond the already-pumped reinjected heat
    k_pump = float(cfg.get("brine_pump_el_per_MWth", 0.0))
    if k_pump > 0:
        m.orc_pump_extra = pyo.Var(m.PERIODS, m.REP_HOURS, domain=pyo.NonNegativeReals)

        def _pump(mm, p, i):
            draw = mm.orc_share[p, i] * q_in_p[p].get((i, 0), 0.0)
            if has_reinject:
                return mm.orc_pump_extra[p, i] >= draw - mm.reinject[p, i]
            return mm.orc_pump_extra[p, i] >= draw
        m.orc_pump_def = pyo.Constraint(m.PERIODS, m.REP_HOURS, rule=_pump)

    # ── objective: ORC revenue and pump cost, weighted by hours represented ──
    _EPS = 1e-4
    vom_orc = float(cfg.get("opex_var_orc_eur_per_MWh_el", 0.0))
    base_yr = min(years)
    extra = 0
    for n, p in enumerate(years):
        yrs_in_p = years[n + 1] - p if n + 1 < len(years) else 5
        r = model.discount_rate_by_period[p]
        pv = sum(1.0 / (1.0 + r) ** (p + y - base_yr) for y in range(yrs_in_p))
        w = model.rh_by_period[p]["weights"]
        for i in range(I):
            sell = float(pyo.value(m.electricity_sell_price[p, i]))
            pn = p_net_p[p].get((i, 0), 0.0)
            extra += m.orc_share[p, i] * ((_EPS - pn * (sell - vom_orc)) * w[i] * pv)
            if k_pump > 0:
                buy = max(float(pyo.value(m.electricity_price[p, i])), 0.0)
                extra += m.orc_pump_extra[p, i] * (k_pump * buy * w[i] * pv)
    m.total_cost.expr = m.total_cost.expr + extra

    model._revorc = {"maps_by_p": maps_by_p, "cfg": cfg}
    pn_all = [v for p in years for v in p_net_p[p].values()]
    logger.info(f"revORC (RH) built: <= {n_max:.0f} modules x {mod_MW:.1f} MW_th, "
                f"ORC P_net {min(pn_all)*1e3:.0f}-{max(pn_all)*1e3:.0f} kW/module")
