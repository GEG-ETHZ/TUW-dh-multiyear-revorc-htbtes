"""
BTES operational constraints for the multi-year district heating MILP.

Investment is exogenous: the caller specifies which period BTES is built in.
The model only optimises charge/discharge dispatch and tracks borehole temperature.

Three variants — one per model resolution:
  add_btes_to_model()      → full 8760-hour model
  add_btes_to_rd_model()   → representative-day (Kotzur) model
  add_btes_to_rh_model()   → representative-hour model

Physics (all variants):
    T[t+1] = a_h * T[t] + b_ch_h * Qch[t] + b_dis_h * Qdis[t] + c_h

Full model: hourly chain — exact for any a_h.

RD model: 24 hourly steps collapsed to one daily update —
    T[d+1] = a_h^24 · T[d]
           + Σ_h  a_h^(23−h) · (b_ch·Qch[rd,h] + b_dis·Qdis[rd,h])
           + c_h · Σ_{k=0}^{23} a_h^k
    When a_h=1 this simplifies to T[d] + b_ch·ΣQch + b_dis·ΣQdis + 24·c_h.

RH model: dur[j] hourly steps collapsed to one interval update —
    T[j+1] = a_h^dur · T[j]
           + (b_ch·Qch[i] + b_dis·Qdis[i] + c_h) · Σ_{k=0}^{dur−1} a_h^k
    Geometric sum  Σ a_h^k = (1−a_h^dur)/(1−a_h)  for a_h≠1, else = dur.
    When a_h=1 this simplifies to T[j] + (b_ch·Qch + b_dis·Qdis + c_h)·dur.

Both aggregated forms are mathematically exact (not approximations) for
any constant a_h — they are derived by unrolling the hourly chain.

Seasonal windows (0-indexed hours):
    charging   : h in [2880, 6551]   (May 1 – Sep 30)
    discharging: h in [0, 2159] ∪ [7296, 8759]  (Nov – Mar)
    idle       : h in [2160, 2879] ∪ [6552, 7295]  (April and October)

Year-by-year temperature tracking
-----------------------------------
Dispatch (Qch, Qdis) is optimised at PERIOD resolution (one representative
year per 5-year investment period).  Borehole temperature is tracked at
individual YEAR resolution so the warm-up dynamics across years 2026–2029
are captured correctly.

Within a period, all individual years share the same dispatch profile.
Temperature links consecutively:

    T[year+1, 0] = T[year, last_timestep]

This means the Tmax constraint applied within each individual year's ARX
chain naturally limits over-charging in later warm-up years (when the field
is already warm), even though the same representative dispatch is applied.

Only the very first active year is pinned to ground temperature T_g (~11 °C).
Tmin (80.8 °C) is enforced for all years except those in the warm-up phase
(first period_length years after build), where the field is still cold.
"""

from __future__ import annotations
from collections import defaultdict
from typing import Dict, Any, List, Tuple
import pyomo.environ as pyo
from loguru import logger

_CH_START    = 2880   # May 1  (h 0-indexed)
_CH_END      = 6551   # Sep 30
_DIS_END1    = 2159   # Mar 31
_IDLE2_START = 6552   # Oct 1
_IDLE2_END   = 7295   # Oct 31
_DIS_START2  = 7296   # Nov 1


def _hour_season(h: int) -> str:
    if _CH_START <= h <= _CH_END:
        return 'charge'
    if _IDLE2_START <= h <= _IDLE2_END:
        return 'idle'
    if h <= _DIS_END1 or h >= _DIS_START2:
        return 'discharge'
    return 'idle'


def _day_season(d: int) -> str:
    """d is 0-indexed (doy = d+1)."""
    doy = d + 1
    if 121 <= doy <= 273:
        return 'charge'
    if (91 <= doy <= 120) or (274 <= doy <= 304):
        return 'idle'
    return 'discharge'


def _majority(items):
    return max(set(items), key=items.count)


def _build_year_structures(
    periods: List[int],
    active: Dict[int, bool],
    period_length: int,
) -> Tuple[List[int], Dict[int, int], set, List[Tuple[int, int]]]:
    """
    Returns:
      btes_years      : all individual years in which BTES is operational
      year_to_period  : maps each individual year → its investment period
      warmup_years    : set of individual years belonging to the first period
                        (the warm-up phase — Tmin skipped here)
      year_pairs      : consecutive (y, y+1) pairs within btes_years
    """
    first_active = min(p for p in periods if active[p])

    btes_years: List[int] = []
    year_to_period: Dict[int, int] = {}
    for p in periods:
        if active[p]:
            for y in range(p, p + period_length):
                btes_years.append(y)
                year_to_period[y] = p

    warmup_years = set(range(first_active, first_active + period_length))
    year_set = set(btes_years)
    year_pairs = [(y, y + 1) for y in btes_years if y + 1 in year_set]

    return btes_years, year_to_period, warmup_years, year_pairs


# ─────────────────────────────────────────────────────────────────────────────
#  FULL MODEL  (8760 real hours)
# ─────────────────────────────────────────────────────────────────────────────

def add_btes_to_model(
    m: pyo.ConcreteModel,
    investment_years: List[int],
    T: int,
    period_params: Dict[int, Dict[str, float]],
    arx: Dict[str, Any],
    T_sup=None,
    T_ret=None,
) -> None:
    """Add BTES variables and constraints to the full 8760-hour model.

    Temperature rule (Fiorentini, Heer & Baldini, CLIMA 2022), active when the ARX file
    carries UA_dis_kW_per_K and hourly network temperatures are given (env
    BTES_TEMP_RULE=0 disables it):
      (1) direct discharge only if the store is hotter than the network supply:
          delta[p,t] in {0,1};  T[y,t] >= T_sup[t] - M(1-delta);  Qdis[p,t] <= Qdis_max*delta
      (2) discharge power fades with the store temperature:
          Qdis[p,t] <= UA_dis * (T[y,t] - T_ret[t])          for every year y of period p
    It replaces the fixed Tmin floor and the warm-up binaries, which were proxies for (1).
    """
    import os
    periods = sorted(investment_years)
    UA_dis = arx.get("UA_dis_kW_per_K")
    temp_rule = (UA_dis is not None and T_sup is not None and T_ret is not None
                 and os.environ.get("BTES_TEMP_RULE", "1") != "0")
    period_length = periods[1] - periods[0] if len(periods) > 1 else 5

    a_h   = float(arx["a_h"])
    c_h   = float(arx["c_h"])
    Tmax  = float(arx["Tmax"])
    Tmin  = float(arx["Tmin"])

    # Age-block-specific ARX coefficients (fall back to arx globals if not in period_params)
    b_ch_p  = {p: float(period_params[p].get("b_ch_h",  arx["b_ch_h"])) for p in periods}
    b_dis_p = {p: float(period_params[p].get("b_dis_h", arx["b_dis_h"])) for p in periods}

    qch_cap = {p: period_params[p]["Qch_max_kW"]  for p in periods}
    qdis_cap= {p: period_params[p]["Qdis_max_kW"] for p in periods}
    # discrete discharge-flow levels (env BTES_FLOW_LEVELS=1), as in the RD model
    _lv_raw = arx.get("UA_dis_levels") or {}
    _lv_on = (os.environ.get("BTES_FLOW_LEVELS", "0") == "1" or os.environ.get("BTES_FLOW_CONTINUOUS", "0") == "1") and len(_lv_raw) > 1 and UA_dis
    _ua_ref = float(arx.get("UA_dis_cap_ref_kW_per_K") or (UA_dis or 1.0))
    _ua_top = max(float(v["UA_kW_per_K"]) for v in _lv_raw.values()) if _lv_on else float(UA_dis or _ua_ref)
    qdis_hard = {p: qdis_cap[p] * _ua_top / _ua_ref for p in periods}
    k_wall = float(arx.get("k_wall_K_per_kW") or 0.0)
    active  = {p: period_params[p]["active"]        for p in periods}

    first_active = min(p for p in periods if active[p])
    T_g = period_params[first_active]["T_init"]   # ground temperature ~11.25 °C

    btes_years, year_to_period, warmup_years, year_pairs = _build_year_structures(
        periods, active, period_length
    )

    # ------------------------------------------------------------------
    # Sets and variables
    # ------------------------------------------------------------------
    m.BTES_YEARS = pyo.Set(initialize=btes_years, ordered=True)

    # Dispatch at PERIOD resolution (decision variables)
    m.btes_Qch  = pyo.Var(m.PERIODS, m.HOURS, domain=pyo.NonNegativeReals)
    m.btes_Qdis = pyo.Var(m.PERIODS, m.HOURS, domain=pyo.NonNegativeReals)

    # Temperature at individual YEAR resolution (state variables)
    m.btes_T    = pyo.Var(m.BTES_YEARS, m.HOURS, domain=pyo.NonNegativeReals)

    # ------------------------------------------------------------------
    # C1/C2: Seasonal capacity windows  (period-indexed dispatch)
    # ------------------------------------------------------------------
    def btes_charge_cap_rule(m, p, t):
        if not active[p]:
            return m.btes_Qch[p, t] == 0
        if _CH_START <= t <= _CH_END:
            return m.btes_Qch[p, t] <= qch_cap[p]
        return m.btes_Qch[p, t] == 0
    m.btes_charge_cap = pyo.Constraint(m.PERIODS, m.HOURS, rule=btes_charge_cap_rule)

    def btes_discharge_cap_rule(m, p, t):
        if not active[p]:
            return m.btes_Qdis[p, t] == 0
        if t <= _DIS_END1 or t >= _DIS_START2:
            return m.btes_Qdis[p, t] <= qdis_hard[p]
        return m.btes_Qdis[p, t] == 0
    m.btes_discharge_cap = pyo.Constraint(m.PERIODS, m.HOURS, rule=btes_discharge_cap_rule)

    # ------------------------------------------------------------------
    # C2.5: Warm-up discharge guard (MILP)
    #   btes_warm[y, t] = 1  iff  T[y, t] >= Tmin  (field warm enough to discharge)
    #   Qdis[period(y), t] <= Qdis_max * btes_warm[y, t]  for ALL warmup years y.
    #   The most restrictive warmup year (coldest T) limits the shared dispatch.
    #   Big-M = Tmax - T_g covers the full reachable temperature range.
    # ------------------------------------------------------------------
    warmup_dis_pairs = [
        (y, t) for y in sorted(warmup_years)
        for t in range(T)
        if t <= _DIS_END1 or t >= _DIS_START2
    ]
    if temp_rule:
        warmup_dis_pairs = []          # superseded by the hourly temperature rule
    if warmup_dis_pairs:
        M_btes = Tmax - T_g + 1.0
        m.BTES_WARMUP_DIS = pyo.Set(initialize=warmup_dis_pairs, dimen=2, ordered=True)
        m.btes_warm = pyo.Var(m.BTES_WARMUP_DIS, domain=pyo.Binary)

        def btes_warm_T_lb_rule(m, y, t):
            return m.btes_T[y, t] >= Tmin - M_btes * (1 - m.btes_warm[y, t])
        m.btes_warm_T_lb = pyo.Constraint(m.BTES_WARMUP_DIS, rule=btes_warm_T_lb_rule)

        def btes_warm_T_ub_rule(m, y, t):
            return m.btes_T[y, t] <= Tmin + M_btes * m.btes_warm[y, t]
        m.btes_warm_T_ub = pyo.Constraint(m.BTES_WARMUP_DIS, rule=btes_warm_T_ub_rule)

        def btes_dis_warm_guard_rule(m, y, t):
            p = year_to_period[y]
            return m.btes_Qdis[p, t] <= qdis_cap[p] * m.btes_warm[y, t]
        m.btes_dis_warm_guard = pyo.Constraint(m.BTES_WARMUP_DIS, rule=btes_dis_warm_guard_rule)

    # ------------------------------------------------------------------
    # C3: Ground temperature for the first active year only
    # ------------------------------------------------------------------
    def btes_T_init_rule(m, y):
        if y != first_active:
            return pyo.Constraint.Skip
        return m.btes_T[y, 0] == T_g
    m.btes_T_init = pyo.Constraint(m.BTES_YEARS, rule=btes_T_init_rule)

    # ------------------------------------------------------------------
    # C4: Year-to-year temperature linking
    #   T[y+1, 0] = T[y, last_hour]
    # ------------------------------------------------------------------
    if year_pairs:
        m.BTES_YEAR_LINKS = pyo.Set(initialize=year_pairs, dimen=2)

        def btes_year_link_rule(m, y_curr, y_next):
            return m.btes_T[y_next, 0] == m.btes_T[y_curr, T - 1]
        m.btes_year_link = pyo.Constraint(m.BTES_YEAR_LINKS, rule=btes_year_link_rule)

    # ------------------------------------------------------------------
    # C5: Hourly ARX — temperature uses year index, dispatch uses period index
    # ------------------------------------------------------------------
    def btes_arx_rule(m, y, t):
        if t == T - 1:
            return pyo.Constraint.Skip
        p = year_to_period[y]
        return m.btes_T[y, t + 1] == (
            a_h * m.btes_T[y, t]
            + b_ch_p[p]  * m.btes_Qch[p, t]
            + b_dis_p[p] * m.btes_Qdis[p, t]
            + c_h
        )
    m.btes_arx = pyo.Constraint(m.BTES_YEARS, m.HOURS, rule=btes_arx_rule)

    # ------------------------------------------------------------------
    # C6: Temperature bounds
    #   Tmax everywhere; Tmin for all years outside the warm-up phase.
    # ------------------------------------------------------------------
    def btes_T_ub_rule(m, y, t):
        return m.btes_T[y, t] <= Tmax
    m.btes_T_ub = pyo.Constraint(m.BTES_YEARS, m.HOURS, rule=btes_T_ub_rule)

    def btes_T_lb_rule(m, y, t):
        if temp_rule or y in warmup_years:
            return pyo.Constraint.Skip
        return m.btes_T[y, t] >= Tmin
    m.btes_T_lb = pyo.Constraint(m.BTES_YEARS, m.HOURS, rule=btes_T_lb_rule)

    # ------------------------------------------------------------------
    # C6b: Hourly temperature rule (direct use, no heat pump)
    # ------------------------------------------------------------------
    _continuous = temp_rule and os.environ.get("BTES_FLOW_CONTINUOUS", "0") == "1" and len(_lv_raw) > 1
    if _continuous:
        # Continuous discharge flow with the flow eliminated analytically (same as the RD rule):
        # Q <= s_j*(T_wall - T_sup) + c_j*dT_net per concave-hull segment, Q <= C_max*dT_net,
        # one on/off binary per hour, T_wall = T - k_wall*Q.
        T_sup_arr = [float(v) for v in T_sup][:T]
        T_ret_arr = [float(v) for v in T_ret][:T]
        years_of = {p: [y for y in btes_years if year_to_period[y] == p] for p in periods}
        dis_pt = [(p, t) for p in periods if active[p] and years_of[p]
                  for t in range(T) if t <= _DIS_END1 or t >= _DIS_START2]
        m.BTES_DIS_PT = pyo.Set(initialize=dis_pt, dimen=2, ordered=True)
        m.btes_dir = pyo.Var(m.BTES_DIS_PT, domain=pyo.Binary)
        pts = sorted(((float(v["C_kW_per_K"]) / float(v["UA_kW_per_K"]) - 1.0, float(v["C_kW_per_K"])) for v in _lv_raw.values()), key=lambda q: q[0])
        pts = [(0.0, 0.0)] + pts
        hull = [pts[0]]
        while hull[-1] != pts[-1]:
            cur = hull[-1]; cand = [q for q in pts if q[0] > cur[0]]
            hull.append(max(cand, key=lambda q: (q[1] - cur[1]) / (q[0] - cur[0])))
        segs = [((b[1] - a[1]) / (b[0] - a[0]), a[1] - (b[1] - a[1]) / (b[0] - a[0]) * a[0]) for a, b in zip(hull[:-1], hull[1:])]
        C_max = hull[-1][1]
        T_floor = T_g - 5.0
        T_floor_p = {p: T_floor for p in periods}
        _tlb = os.environ.get("BTES_TIGHT_M")
        if _tlb:
            _tlb = float(_tlb)
            for p in periods:
                if p != first_active:
                    T_floor_p[p] = _tlb
            for y in btes_years:
                if y not in warmup_years:
                    for t in range(T):
                        m.btes_T[y, t].setlb(_tlb)
            logger.warning(f"BTES_TIGHT_M (full): state lower bound {_tlb} C after the warm-up period")
        pytj = [(p, y, t, j) for (p, t) in dis_pt for y in years_of[p] for j in range(len(segs))]
        m.BTES_DIS_PYTJ = pyo.Set(initialize=pytj, dimen=4, ordered=True)

        def btes_cont_seg_rule(m, p, y, t, j):
            dT = T_sup_arr[t] - T_ret_arr[t]
            if dT <= 0.0:
                return m.btes_Qdis[p, t] <= 0.0
            s_j, c_j = segs[j]
            M_j = s_j * max(T_sup_arr[t] - T_floor_p[p], 0.0) + 1.0
            return (m.btes_Qdis[p, t] * (1.0 + s_j * k_wall)
                    <= s_j * (m.btes_T[y, t] - T_sup_arr[t]) + c_j * dT + M_j * (1 - m.btes_dir[p, t]))
        m.btes_cont_seg = pyo.Constraint(m.BTES_DIS_PYTJ, rule=btes_cont_seg_rule)

        def btes_cont_ceiling_rule(m, p, t):
            dT = max(T_sup_arr[t] - T_ret_arr[t], 0.0)
            return m.btes_Qdis[p, t] <= min(C_max * dT, qdis_hard[p]) * m.btes_dir[p, t]
        m.btes_cont_ceiling = pyo.Constraint(m.BTES_DIS_PT, rule=btes_cont_ceiling_rule)
        logger.info(f"BTES continuous-flow rule ON (full): hull {[(round(a,3), round(b,1)) for a,b in hull]}, segments {[(round(a,0), round(b,1)) for a,b in segs]}, "
                    f"C_max {C_max:.1f} kW/K, k_wall {k_wall:.2e} K/kW, {len(pytj)} rows, {len(dis_pt)} on/off binaries")
    elif temp_rule:
        UA = float(UA_dis)
        T_sup_arr = [float(v) for v in T_sup][:T]
        T_ret_arr = [float(v) for v in T_ret][:T]
        years_of = {p: [y for y in btes_years if year_to_period[y] == p] for p in periods}
        dis_pt = [(p, t) for p in periods if active[p] and years_of[p]
                  for t in range(T) if t <= _DIS_END1 or t >= _DIS_START2]
        m.BTES_DIS_PT = pyo.Set(initialize=dis_pt, dimen=2, ordered=True)
        m.btes_dir = pyo.Var(m.BTES_DIS_PT, domain=pyo.Binary)
        yt = [(p, y, t) for (p, t) in dis_pt for y in years_of[p]]
        m.BTES_DIS_PYT = pyo.Set(initialize=yt, dimen=3, ordered=True)
        T_floor = T_g - 5.0

        lv = []
        if _lv_on:
            base_key = min(_lv_raw, key=float)
            for k, v in sorted(_lv_raw.items(), key=lambda kv: float(kv[0])):
                if k != base_key:
                    lv.append({"m": float(k), "UA": float(v["UA_kW_per_K"]), "C": float(v["C_kW_per_K"])})
        K = list(range(len(lv)))
        cap_k = {(p, k): qdis_cap[p] * lv[k]["UA"] / _ua_ref for p in periods for k in K}
        cap_base = {p: qdis_cap[p] * UA / _ua_ref for p in periods}

        def _Twall(m, p, y, t):                       # wall seen by the fluid
            return m.btes_T[y, t] - k_wall * m.btes_Qdis[p, t]

        def btes_dir_gate_rule(m, p, y, t):           # (1) store hotter than supply
            M_s = T_sup_arr[t] - T_floor + k_wall * qdis_hard[p] + 1.0
            return _Twall(m, p, y, t) >= T_sup_arr[t] - M_s * (1 - m.btes_dir[p, t])
        m.btes_dir_gate = pyo.Constraint(m.BTES_DIS_PYT, rule=btes_dir_gate_rule)

        if K:
            m.BTES_DIS_PTK = pyo.Set(initialize=[(p, t, k) for (p, t) in dis_pt for k in K], dimen=3, ordered=True)
            m.btes_lvl = pyo.Var(m.BTES_DIS_PTK, domain=pyo.Binary)
            m.BTES_DIS_PYTK = pyo.Set(initialize=[(p, y, t, k) for (p, y, t) in yt for k in K], dimen=4, ordered=True)

            def btes_lvl_one_rule(m, p, t):
                return m.btes_dir[p, t] + pyo.quicksum(m.btes_lvl[p, t, k] for k in K) <= 1
            m.btes_lvl_one = pyo.Constraint(m.BTES_DIS_PT, rule=btes_lvl_one_rule)

            def btes_lvl_gate_rule(m, p, y, t, k):
                r = lv[k]["C"] / lv[k]["UA"]
                need = T_ret_arr[t] + r * (T_sup_arr[t] - T_ret_arr[t])
                M_k = need - T_floor + k_wall * qdis_hard[p] + 1.0
                return _Twall(m, p, y, t) >= need - M_k * (1 - m.btes_lvl[p, t, k])
            m.btes_lvl_gate = pyo.Constraint(m.BTES_DIS_PYTK, rule=btes_lvl_gate_rule)

            def btes_lvl_ua_rule(m, p, y, t, k):
                UA_k = lv[k]["UA"]
                M_q = qdis_hard[p] * (1.0 + UA_k * k_wall) + UA_k * (T_ret_arr[t] - T_floor) + 1.0
                return (m.btes_Qdis[p, t] <= UA_k * (_Twall(m, p, y, t) - T_ret_arr[t])
                        + M_q * (1 - m.btes_lvl[p, t, k]))
            m.btes_lvl_ua = pyo.Constraint(m.BTES_DIS_PYTK, rule=btes_lvl_ua_rule)

            def btes_lvl_outlet_rule(m, p, t, k):      # outlet must reach supply at flow k
                C_k = lv[k]["C"]; dT = max(T_sup_arr[t] - T_ret_arr[t], 0.0)
                return m.btes_Qdis[p, t] >= C_k * dT - C_k * dT * (1 - m.btes_lvl[p, t, k])
            m.btes_lvl_outlet = pyo.Constraint(m.BTES_DIS_PTK, rule=btes_lvl_outlet_rule)

        def btes_dir_cap_rule(m, p, t):
            return (m.btes_Qdis[p, t] <= cap_base[p] * m.btes_dir[p, t]
                    + pyo.quicksum(cap_k[p, k] * m.btes_lvl[p, t, k] for k in K))
        m.btes_dir_cap = pyo.Constraint(m.BTES_DIS_PT, rule=btes_dir_cap_rule)

        def btes_dis_ua_rule(m, p, y, t):             # (2) power fades with store temperature
            M_base = qdis_hard[p] * (1.0 + UA * k_wall) + UA * (T_ret_arr[t] - T_floor) + 1.0
            return (m.btes_Qdis[p, t] <= UA * (_Twall(m, p, y, t) - T_ret_arr[t])
                    + M_base * (1 - m.btes_dir[p, t]))
        m.btes_dis_ua = pyo.Constraint(m.BTES_DIS_PYT, rule=btes_dis_ua_rule)
        logger.info(f"BTES temperature rule ON: UA_dis {UA:.2f} kW/K (cap ref {_ua_ref:.1f}), k_wall {k_wall:.2e} K/kW, "
                    f"hourly T_sup {min(T_sup_arr):.0f}-{max(T_sup_arr):.0f} C, T_ret "
                    f"{min(T_ret_arr):.0f}-{max(T_ret_arr):.0f} C; {len(dis_pt)} gate binaries"
                    + (f"; flow levels {[l['m'] for l in lv]} kg/s/bh, {len(dis_pt)*len(K)} level binaries" if K else ""))
    else:
        logger.info("BTES temperature rule OFF: fixed Tmin floor + warm-up binaries")

    # ------------------------------------------------------------------
    # C7: Annual energy balance guard  (per period, on representative year)
    # ------------------------------------------------------------------
    def btes_energy_balance_rule(m, p):
        if not active[p]:
            return pyo.Constraint.Skip
        return (
            pyo.quicksum(m.btes_Qdis[p, t] for t in m.HOURS)
            <= pyo.quicksum(m.btes_Qch[p, t]  for t in m.HOURS)
        )
    m.btes_energy_balance = pyo.Constraint(m.PERIODS, rule=btes_energy_balance_rule)


# ─────────────────────────────────────────────────────────────────────────────
#  RD MODEL  (representative days × 24 hours, Kotzur inter-day coupling)
# ─────────────────────────────────────────────────────────────────────────────

def add_btes_to_rd_model(
    m: pyo.ConcreteModel,
    investment_years: List[int],
    n_days: int,
    day_to_rep: List[int],
    period_params: Dict[int, Dict[str, float]],
    arx: Dict[str, Any],
    T_sup=None,
    T_ret=None,
) -> None:
    """
    Add BTES to the RD (representative-day) model.

    Temperature rule (as in add_btes_to_model), at RD resolution: the gate delta[p,rd,h]
    compares the CLUSTER-MEAN store temperature of the calendar days behind rep day rd
    (for every year of the period) with the cluster-mean hourly supply temperature;
    discharge power is limited by UA_dis*(T_mean - T_ret,mean). Env BTES_TEMP_RULE=0 disables.

    Dispatch is indexed on REP_DAYS × HOURS (period resolution).
    Temperature state is indexed on BTES_YEARS × DAYS (individual year resolution).

    Since a_h = 1.0, the 24-hourly updates collapse to a single daily sum:
        T[y, d+1] = T[y, d] + b_ch * sum_h(Qch[period(y), rep(d), h])
                              + b_dis * sum_h(Qdis[period(y), rep(d), h])

    Year-to-year linking:
        T[y+1, 0] = T[y, n_days-1]
    """
    import os
    periods = sorted(investment_years)
    period_length = periods[1] - periods[0] if len(periods) > 1 else 5
    UA_dis = arx.get("UA_dis_kW_per_K")
    # Fixed discharge flow other than the library base (env BTES_BASE_FLOW=0.10|0.20|0.30, RD only):
    # use that level's conductance and capacity rate as the single fixed flow, no level binaries.
    _C_fixed = None
    _bf = os.environ.get("BTES_BASE_FLOW")
    if _bf:
        _lvr0 = arx.get("UA_dis_levels") or {}
        _key = next((k for k in _lvr0 if abs(float(k) - float(_bf)) < 1e-6), None)
        if _key is None:
            raise ValueError(f"BTES_BASE_FLOW={_bf} not in library levels {list(_lvr0)}")
        UA_dis = float(_lvr0[_key]["UA_kW_per_K"]); _C_fixed = float(_lvr0[_key]["C_kW_per_K"])
        logger.info(f"BTES fixed discharge flow override: {_bf} kg/s per bh -> UA_dis {UA_dis:.2f} kW/K, C {_C_fixed:.2f} kW/K")
    temp_rule = (UA_dis is not None and T_sup is not None and T_ret is not None
                 and os.environ.get("BTES_TEMP_RULE", "1") != "0")

    a_h   = float(arx.get("a_h", 1.0))
    c_h   = float(arx.get("c_h", 0.0))
    Tmax  = float(arx["Tmax"])
    Tmin  = float(arx["Tmin"])

    # Age-block-specific ARX coefficients (fall back to arx globals if not in period_params)
    b_ch_p  = {p: float(period_params[p].get("b_ch_h",  arx["b_ch_h"])) for p in periods}
    b_dis_p = {p: float(period_params[p].get("b_dis_h", arx["b_dis_h"])) for p in periods}

    qch_cap = {p: period_params[p]["Qch_max_kW"]  for p in periods}
    qdis_cap= {p: period_params[p]["Qdis_max_kW"] for p in periods}
    # hard per-hour discharge ceiling: base cap, scaled up when discrete flow levels are on
    _lv_raw = arx.get("UA_dis_levels") or {}
    _lv_on = (os.environ.get("BTES_FLOW_LEVELS", "0") == "1" or os.environ.get("BTES_FLOW_CONTINUOUS", "0") == "1") and len(_lv_raw) > 1 and UA_dis
    _ua_ref = float(arx.get("UA_dis_cap_ref_kW_per_K") or (UA_dis or 1.0))
    _ua_top = max(float(v["UA_kW_per_K"]) for v in _lv_raw.values()) if _lv_on else float(UA_dis or _ua_ref)
    _lv_scale = _ua_top / _ua_ref          # block caps were produced at UA_ref; hard ceiling at the top level
    qdis_hard = {p: qdis_cap[p] * _lv_scale for p in periods}
    k_wall = float(arx.get("k_wall_K_per_kW") or 0.0)   # wall = bulk state - k_wall * Qdis
    active  = {p: period_params[p]["active"]        for p in periods}

    first_active = min(p for p in periods if active[p])
    T_g = period_params[first_active]["T_init"]

    btes_years, year_to_period, warmup_years, year_pairs = _build_year_structures(
        periods, active, period_length
    )

    # Precompute daily ARX scalars (pure Python, outside Pyomo rules).
    # Expand 24 hourly steps: T[d+1] = a_d·T[d] + Σ_h w_h·(b·Q[h]) + c_d
    # w_h = a_h^(23-h): weight of hour-h dispatch on end-of-day temperature.
    # When a_h=1: a_d=1, all w_h=1, c_d=24·c_h → reduces to simple daily sum.
    _a_d  = a_h ** 24
    _hw   = [a_h ** (23 - h) for h in range(24)]
    _c_d  = c_h * sum(_hw)   # c_h · Σ_{k=0}^{23} a_h^k

    # Classify each representative day by the majority season of its real days
    rd_season: Dict[int, str] = {}
    rep_to_days: Dict[int, List[int]] = defaultdict(list)
    for d, rd in enumerate(day_to_rep):
        rep_to_days[rd].append(d)
    for rd, days in rep_to_days.items():
        rd_season[rd] = _majority([_day_season(d) for d in days])

    # ------------------------------------------------------------------
    # Sets and variables
    # ------------------------------------------------------------------
    m.BTES_YEARS = pyo.Set(initialize=btes_years, ordered=True)

    m.btes_Qch  = pyo.Var(m.PERIODS, m.REP_DAYS, m.HOURS, domain=pyo.NonNegativeReals)
    m.btes_Qdis = pyo.Var(m.PERIODS, m.REP_DAYS, m.HOURS, domain=pyo.NonNegativeReals)
    m.btes_T    = pyo.Var(m.BTES_YEARS, m.DAYS,           domain=pyo.NonNegativeReals)

    # ------------------------------------------------------------------
    # C1/C2: Seasonal capacity windows
    # ------------------------------------------------------------------
    def btes_charge_cap_rule(m, p, rd, h):
        if not active[p] or rd_season[rd] != 'charge':
            return m.btes_Qch[p, rd, h] == 0
        return m.btes_Qch[p, rd, h] <= qch_cap[p]
    m.btes_charge_cap = pyo.Constraint(m.PERIODS, m.REP_DAYS, m.HOURS, rule=btes_charge_cap_rule)

    def btes_discharge_cap_rule(m, p, rd, h):
        if not active[p] or rd_season[rd] != 'discharge':
            return m.btes_Qdis[p, rd, h] == 0
        return m.btes_Qdis[p, rd, h] <= qdis_hard[p]
    m.btes_discharge_cap = pyo.Constraint(m.PERIODS, m.REP_DAYS, m.HOURS, rule=btes_discharge_cap_rule)

    # ------------------------------------------------------------------
    # C2.5: Warm-up discharge guard (MILP, daily resolution)
    #   btes_warm[y, d] = 1 iff T[y, d] >= Tmin.
    #   Qdis[period(y), rd, h] <= Qdis_max * btes_warm[y, d(rd)]  for ALL warmup years.
    # ------------------------------------------------------------------
    discharge_rep_days = [rd for rd, s in rd_season.items() if s == 'discharge']
    # Map rep_day → one representative calendar day (for big-M pairing with btes_T on DAYS)
    rep_to_day = {rd: rep_to_days[rd][0] for rd in discharge_rep_days}

    warmup_dis_pairs = [
        (y, rep_to_day[rd]) for y in sorted(warmup_years) for rd in discharge_rep_days
    ]
    if temp_rule:
        warmup_dis_pairs = []          # superseded by the temperature rule
    if warmup_dis_pairs:
        M_btes = Tmax - T_g + 1.0
        m.BTES_WARMUP_DIS = pyo.Set(initialize=warmup_dis_pairs, dimen=2, ordered=True)
        m.btes_warm = pyo.Var(m.BTES_WARMUP_DIS, domain=pyo.Binary)

        def btes_warm_T_lb_rule(m, y, d):
            return m.btes_T[y, d] >= Tmin - M_btes * (1 - m.btes_warm[y, d])
        m.btes_warm_T_lb = pyo.Constraint(m.BTES_WARMUP_DIS, rule=btes_warm_T_lb_rule)

        def btes_warm_T_ub_rule(m, y, d):
            return m.btes_T[y, d] <= Tmin + M_btes * m.btes_warm[y, d]
        m.btes_warm_T_ub = pyo.Constraint(m.BTES_WARMUP_DIS, rule=btes_warm_T_ub_rule)

        # Qdis on (period, rep_day, hour); guard is indexed on (year, calendar_day)
        # For each warmup year and each discharge rep_day, limit ALL hours of that rep_day
        warmup_rd_pairs = [(y, rd) for y in sorted(warmup_years) for rd in discharge_rep_days]
        m.BTES_WARMUP_RD = pyo.Set(initialize=warmup_rd_pairs, dimen=2, ordered=True)

        def btes_dis_warm_guard_rule(m, y, rd):
            p   = year_to_period[y]
            d   = rep_to_day[rd]
            return (
                pyo.quicksum(m.btes_Qdis[p, rd, h] for h in m.HOURS)
                <= qdis_cap[p] * 24 * m.btes_warm[y, d]
            )
        m.btes_dis_warm_guard = pyo.Constraint(m.BTES_WARMUP_RD, rule=btes_dis_warm_guard_rule)

    # ------------------------------------------------------------------
    # C3: Ground temperature for the first active year
    # ------------------------------------------------------------------
    def btes_T_init_rule(m, y):
        if y != first_active:
            return pyo.Constraint.Skip
        return m.btes_T[y, 0] == T_g
    m.btes_T_init = pyo.Constraint(m.BTES_YEARS, rule=btes_T_init_rule)

    # ------------------------------------------------------------------
    # C4: Year-to-year linking
    # ------------------------------------------------------------------
    if year_pairs:
        m.BTES_YEAR_LINKS = pyo.Set(initialize=year_pairs, dimen=2)

        def btes_year_link_rule(m, y_curr, y_next):
            return m.btes_T[y_next, 0] == m.btes_T[y_curr, n_days - 1]
        m.btes_year_link = pyo.Constraint(m.BTES_YEAR_LINKS, rule=btes_year_link_rule)

    # ------------------------------------------------------------------
    # C5: Inter-day ARX — exact for any a_h (see module docstring).
    # ------------------------------------------------------------------
    def btes_arx_rule(m, y, d):
        if d == n_days - 1:
            return pyo.Constraint.Skip
        p  = year_to_period[y]
        rd = day_to_rep[d]
        weighted_Qch  = pyo.quicksum(_hw[h] * m.btes_Qch[p, rd, h]  for h in m.HOURS)
        weighted_Qdis = pyo.quicksum(_hw[h] * m.btes_Qdis[p, rd, h] for h in m.HOURS)
        return m.btes_T[y, d + 1] == (
            _a_d * m.btes_T[y, d]
            + b_ch_p[p]  * weighted_Qch
            + b_dis_p[p] * weighted_Qdis
            + _c_d
        )
    m.btes_arx = pyo.Constraint(m.BTES_YEARS, m.DAYS, rule=btes_arx_rule)

    # ------------------------------------------------------------------
    # C6: Temperature bounds
    # ------------------------------------------------------------------
    def btes_T_ub_rule(m, y, d):
        return m.btes_T[y, d] <= Tmax
    m.btes_T_ub = pyo.Constraint(m.BTES_YEARS, m.DAYS, rule=btes_T_ub_rule)

    def btes_T_lb_rule(m, y, d):
        if temp_rule or y in warmup_years:
            return pyo.Constraint.Skip
        return m.btes_T[y, d] >= Tmin
    m.btes_T_lb = pyo.Constraint(m.BTES_YEARS, m.DAYS, rule=btes_T_lb_rule)

    # ------------------------------------------------------------------
    # C6b: Temperature rule at RD resolution (direct use, no heat pump)
    # ------------------------------------------------------------------
    _continuous = temp_rule and os.environ.get("BTES_FLOW_CONTINUOUS", "0") == "1" and len(arx.get("UA_dis_levels") or {}) > 1
    if _continuous:
        # ------------------------------------------------------------------
        # Continuous discharge flow, flow eliminated analytically (no binaries).
        # Outlet pinned at the supply temperature: at flow m the power is Q = C(m)*dT_net and
        # the wall must exceed the supply by e(m) = (C/UA - 1)*dT_net. Inverting, the admissible
        # power is a concave piecewise-linear function of the excess x = T_wall - T_sup through
        # the library's flow points (concave hull), flat at the pump ceiling:
        #     Q <= s_j * x + c_j (each hull segment),  Q <= C_max * dT_net,
        # with T_wall = T_mean - k_wall*Q. A store colder than the supply gets Q <= 0 automatically.
        # ------------------------------------------------------------------
        Ts = [float(v) for v in T_sup]; Tr = [float(v) for v in T_ret]
        years_of = {p: [y for y in btes_years if year_to_period[y] == p] for p in periods}
        Ts_rd = {rd: [sum(Ts[d*24+h] for d in days)/len(days) for h in range(24)] for rd, days in rep_to_days.items()}
        Tr_rd = {rd: [sum(Tr[d*24+h] for d in days)/len(days) for h in range(24)] for rd, days in rep_to_days.items()}
        pts = sorted(((float(v["C_kW_per_K"]) / float(v["UA_kW_per_K"]) - 1.0, float(v["C_kW_per_K"]))
                      for v in arx["UA_dis_levels"].values()), key=lambda t: t[0])   # (r-1, C) per flow point
        pts = [(0.0, 0.0)] + pts
        hull = [pts[0]]                                             # concave hull from the origin
        while hull[-1] != pts[-1]:
            cur = hull[-1]; cand = [q for q in pts if q[0] > cur[0]]
            hull.append(max(cand, key=lambda q: (q[1] - cur[1]) / (q[0] - cur[0])))
        segs = [((b[1] - a[1]) / (b[0] - a[0]), a[1] - (b[1] - a[1]) / (b[0] - a[0]) * a[0]) for a, b in zip(hull[:-1], hull[1:])]
        C_max = hull[-1][1]
        prh = [(p, rd, h) for p in periods if active[p] and years_of[p] for rd in discharge_rep_days for h in range(24)]
        pyrhj = [(p, y, rd, h, j) for (p, rd, h) in prh for y in years_of[p] for j in range(len(segs))]
        m.BTES_DIS_PYRHJ = pyo.Set(initialize=pyrhj, dimen=5, ordered=True)
        m.BTES_DIS_PRH = pyo.Set(initialize=prh, dimen=3, ordered=True)
        # one on/off binary per hour: a store colder than the supply cannot deliver (the admissible-power
        # curve is zero for a negative excess, which no linear row can express on its own)
        m.btes_dir = pyo.Var(m.BTES_DIS_PRH, domain=pyo.Binary)
        T_floor = T_g - 5.0
        T_floor_p = {p: T_floor for p in periods}
        _tlb = os.environ.get("BTES_TIGHT_M")
        if _tlb:
            _tlb = float(_tlb)
            for p in periods:
                if p != first_active:
                    T_floor_p[p] = _tlb
            for y in btes_years:
                if y not in warmup_years:
                    for d in m.DAYS:
                        m.btes_T[y, d].setlb(_tlb)
            logger.warning(f"BTES_TIGHT_M: state lower bound {_tlb} C after the warm-up period; big-M floors {T_floor_p}")

        def _Tmean_c(m, y, rd):
            days = rep_to_days[rd]
            return pyo.quicksum(m.btes_T[y, d] for d in days) / len(days)

        def btes_cont_seg_rule(m, p, y, rd, h, j):
            dT = Ts_rd[rd][h] - Tr_rd[rd][h]
            if dT <= 0.0:
                return m.btes_Qdis[p, rd, h] <= 0.0
            s_j, c_j = segs[j]                                      # slope in kW/K, intercept in kW per K of lift
            M_j = s_j * max(Ts_rd[rd][h] - T_floor_p[p], 0.0) + 1.0  # covers the most negative excess when off
            return (m.btes_Qdis[p, rd, h] * (1.0 + s_j * k_wall)
                    <= s_j * (_Tmean_c(m, y, rd) - Ts_rd[rd][h]) + c_j * dT + M_j * (1 - m.btes_dir[p, rd, h]))
        m.btes_cont_seg = pyo.Constraint(m.BTES_DIS_PYRHJ, rule=btes_cont_seg_rule)

        def btes_cont_ceiling_rule(m, p, rd, h):                    # pump ceiling, outlet at supply; zero when off
            dT = max(Ts_rd[rd][h] - Tr_rd[rd][h], 0.0)
            return m.btes_Qdis[p, rd, h] <= min(C_max * dT, qdis_hard[p]) * m.btes_dir[p, rd, h]
        m.btes_cont_ceiling = pyo.Constraint(m.BTES_DIS_PRH, rule=btes_cont_ceiling_rule)
        logger.info(f"BTES continuous-flow rule ON (RD): hull points (r-1, C kW/K) {[(round(a,3), round(b,1)) for a,b in hull]}, "
                    f"segments (slope kW/K, intercept kW per K lift) {[(round(a,0), round(b,1)) for a,b in segs]}, "
                    f"ceiling C_max {C_max:.1f} kW/K, k_wall {k_wall:.2e} K/kW, {len(pyrhj)} linear rows, {len(prh)} on/off binaries, no level binaries")
    elif temp_rule:
        UA = float(UA_dis)
        Ts = [float(v) for v in T_sup]; Tr = [float(v) for v in T_ret]
        years_of = {p: [y for y in btes_years if year_to_period[y] == p] for p in periods}
        # cluster-mean hourly network temperatures per rep day
        Ts_rd = {rd: [sum(Ts[d*24+h] for d in days)/len(days) for h in range(24)]
                 for rd, days in rep_to_days.items()}
        Tr_rd = {rd: [sum(Tr[d*24+h] for d in days)/len(days) for h in range(24)]
                 for rd, days in rep_to_days.items()}
        prh = [(p, rd, h) for p in periods if active[p] and years_of[p]
               for rd in discharge_rep_days for h in range(24)]
        m.BTES_DIS_PRH = pyo.Set(initialize=prh, dimen=3, ordered=True)
        m.btes_dir = pyo.Var(m.BTES_DIS_PRH, domain=pyo.Binary)
        M_sup = max(Ts) - T_g + 1.0
        pyrh = [(p, y, rd, h) for (p, rd, h) in prh for y in years_of[p]]
        m.BTES_DIS_PYRH = pyo.Set(initialize=pyrh, dimen=4, ordered=True)

        def _Tmean(m, y, rd):
            days = rep_to_days[rd]
            return pyo.quicksum(m.btes_T[y, d] for d in days) / len(days)

        def _Twall(m, p, y, rd, h):                    # wall seen by the fluid
            return _Tmean(m, y, rd) - k_wall * m.btes_Qdis[p, rd, h]

        hard_max = max(qdis_hard[p] for p in periods if active[p])
        T_floor = T_g - 5.0                            # the state never falls below the ground temperature
        # Speed-up A (env BTES_TIGHT_M=<T_lb>): after the warm-up period the state is assumed to stay
        # above T_lb (a modelling assumption, logged); the big-M constants of the gate/cap rows then
        # use that floor instead of the ground temperature, and the state variable gets the bound.
        _tlb = os.environ.get("BTES_TIGHT_M")
        T_floor_p = {p: T_floor for p in periods}
        if _tlb:
            _tlb = float(_tlb)
            for p in periods:
                if p != first_active:
                    T_floor_p[p] = _tlb
            for y in btes_years:
                if y not in warmup_years:
                    for d in m.DAYS:
                        m.btes_T[y, d].setlb(_tlb)
            logger.warning(f"BTES_TIGHT_M: state lower bound {_tlb} C after the warm-up period; big-M floors {T_floor_p}")

        def _need_base(rd, h):                         # wall needed for the outlet to reach supply at the fixed flow
            if _C_fixed is None:
                return Ts_rd[rd][h]
            return Tr_rd[rd][h] + (_C_fixed / UA) * (Ts_rd[rd][h] - Tr_rd[rd][h])

        def btes_dir_gate_rule(m, p, y, rd, h):        # (1) store hotter than supply
            need = _need_base(rd, h)
            M_s = need - T_floor_p[p] + k_wall * qdis_hard[p] + 1.0
            return _Twall(m, p, y, rd, h) >= need - M_s * (1 - m.btes_dir[p, rd, h])
        m.btes_dir_gate = pyo.Constraint(m.BTES_DIS_PYRH, rule=btes_dir_gate_rule)

        # Discrete discharge-flow levels (env BTES_FLOW_LEVELS=1): besides the base
        # flow (modulated freely below its UA limit, as before) the model may pick a
        # higher pump flow k with its own conductance UA_k and capacity rate C_k.
        # At level k the outlet T_ret + Q/C_k must reach the supply temperature, which
        # gives a LOWER bound on Q and, combined with Q <= UA_k (T - T_ret), a stricter
        # gate T >= T_ret + (C_k/UA_k)(T_sup - T_ret). Caps scale with UA_k/UA_base.
        levels_raw = arx.get("UA_dis_levels") or {}
        use_levels = os.environ.get("BTES_FLOW_LEVELS", "0") == "1" and len(levels_raw) > 1
        lv = []
        if use_levels:
            base_key = min(levels_raw, key=float)
            for k, v in sorted(levels_raw.items(), key=lambda kv: float(kv[0])):
                if k == base_key:
                    continue
                lv.append({"m": float(k), "UA": float(v["UA_kW_per_K"]), "C": float(v["C_kW_per_K"])})
        K = list(range(len(lv)))
        max_cap = hard_max
        cap_k = {(p, k): qdis_cap[p] * lv[k]["UA"] / _ua_ref for p in periods for k in K}
        cap_base = {p: qdis_cap[p] * UA / _ua_ref for p in periods}
        Tr_max = max(Tr); Ts_max = max(Ts); dTsr_max = max(a - b for a, b in zip(Ts, Tr))
        if use_levels:
            m.BTES_DIS_PRHK = pyo.Set(initialize=[(p, rd, h, k) for (p, rd, h) in prh for k in K],
                                      dimen=4, ordered=True)
            m.btes_lvl = pyo.Var(m.BTES_DIS_PRHK, domain=pyo.Binary)

            def btes_lvl_one_rule(m, p, rd, h):
                return m.btes_dir[p, rd, h] + pyo.quicksum(m.btes_lvl[p, rd, h, k] for k in K) <= 1
            m.btes_lvl_one = pyo.Constraint(m.BTES_DIS_PRH, rule=btes_lvl_one_rule)

            m.BTES_DIS_PYRHK = pyo.Set(initialize=[(p, y, rd, h, k) for (p, y, rd, h) in pyrh for k in K],
                                       dimen=5, ordered=True)

            # Speed-up B (env BTES_PRUNE=1): a level whose gate temperature exceeds Tmax, or whose
            # minimum power exceeds its cap, can never be used in that hour -> fix the binary to 0
            # and skip its rows (valid for any solution).
            _prune = os.environ.get("BTES_PRUNE", "0") == "1"
            _dead = set()
            if _prune:
                for (p, rd, h, k) in m.BTES_DIS_PRHK:
                    r = lv[k]["C"] / lv[k]["UA"]
                    need = Tr_rd[rd][h] + r * (Ts_rd[rd][h] - Tr_rd[rd][h])
                    qmin = lv[k]["C"] * max(Ts_rd[rd][h] - Tr_rd[rd][h], 0.0)
                    if need > Tmax or qmin > cap_k[p, k]:
                        m.btes_lvl[p, rd, h, k].fix(0); _dead.add((p, rd, h, k))
                logger.warning(f"BTES_PRUNE: fixed {len(_dead)} of {len(m.BTES_DIS_PRHK)} level binaries to 0 (never admissible)")

            def btes_lvl_gate_rule(m, p, y, rd, h, k):
                if (p, rd, h, k) in _dead:
                    return pyo.Constraint.Skip
                r = lv[k]["C"] / lv[k]["UA"]
                need = Tr_rd[rd][h] + r * (Ts_rd[rd][h] - Tr_rd[rd][h])
                M_k = need - T_floor_p[p] + k_wall * qdis_hard[p] + 1.0
                return _Twall(m, p, y, rd, h) >= need - M_k * (1 - m.btes_lvl[p, rd, h, k])
            m.btes_lvl_gate = pyo.Constraint(m.BTES_DIS_PYRHK, rule=btes_lvl_gate_rule)

            def btes_lvl_ua_rule(m, p, y, rd, h, k):
                if (p, rd, h, k) in _dead:
                    return pyo.Constraint.Skip
                UA_k = lv[k]["UA"]
                M_q = qdis_hard[p] * (1.0 + UA_k * k_wall) + UA_k * (Tr_rd[rd][h] - T_floor_p[p]) + 1.0
                return (m.btes_Qdis[p, rd, h] <= UA_k * (_Twall(m, p, y, rd, h) - Tr_rd[rd][h])
                        + M_q * (1 - m.btes_lvl[p, rd, h, k]))
            m.btes_lvl_ua = pyo.Constraint(m.BTES_DIS_PYRHK, rule=btes_lvl_ua_rule)

            def btes_lvl_outlet_rule(m, p, rd, h, k):      # outlet must reach supply at flow k
                if (p, rd, h, k) in _dead:
                    return pyo.Constraint.Skip
                C_k = lv[k]["C"]
                dT = max(Ts_rd[rd][h] - Tr_rd[rd][h], 0.0)
                return (m.btes_Qdis[p, rd, h] >= C_k * dT
                        - C_k * dT * (1 - m.btes_lvl[p, rd, h, k]))
            m.btes_lvl_outlet = pyo.Constraint(m.BTES_DIS_PRHK, rule=btes_lvl_outlet_rule)

            # Speed-up C (env BTES_PUMP_COST=<EUR per kWh_el>): circulation-pump electricity per level,
            # EUR per year per period; tiny against gas but breaks the ties between levels and hours.
            _pc = os.environ.get("BTES_PUMP_COST")
            if _pc:
                _pc = float(_pc)
                _W_per_bh = {0.0517: 0.1, 0.10: 0.8, 0.20: 6.0, 0.30: 20.0}   # Darcy estimate, 200 m U-tube
                _nb = int(arx.get("n_boreholes") or os.environ.get("BTES_DESIGN_N") or 0) or int(round(float(arx.get("total_length_m", 0)) / 200.0)) or 1
                _pk = {k: _nb * min(_W_per_bh.items(), key=lambda kv: abs(kv[0] - lv[k]["m"]))[1] / 1000.0 for k in K}  # kW
                def _pump_rule(m, p):
                    if not active[p]:
                        return 0.0
                    return pyo.quicksum(m.btes_lvl[p, rd, h, k] * _pk[k] * _pc * m.day_weight[p, rd]
                                        for (pp, rd, h, k) in m.BTES_DIS_PRHK if pp == p)
                m.btes_pump_cost = pyo.Expression(m.PERIODS, rule=_pump_rule)
                logger.warning(f"BTES_PUMP_COST: {_pc} EUR/kWh_el, pump power per level {_pk} kW ({_nb} bh)")

        # debug: BTES_FLOW_FORCE="period:rep_day:level_idx:h0-h1" fixes those level binaries to 1
        _force = os.environ.get("BTES_FLOW_FORCE")
        if use_levels and _force:
            fp, frd, fk, fh = _force.split(":"); h0, h1 = [int(x) for x in fh.split("-")]
            for h in range(h0, h1 + 1):
                m.btes_lvl[int(fp), int(frd), h, int(fk)].fix(1)
            logger.warning(f"BTES_FLOW_FORCE: fixed level {fk} on period {fp}, rep day {frd}, hours {h0}-{h1}")

        def btes_dir_cap_rule(m, p, rd, h):
            return (m.btes_Qdis[p, rd, h] <= cap_base[p] * m.btes_dir[p, rd, h]
                    + pyo.quicksum(cap_k[p, k] * m.btes_lvl[p, rd, h, k] for k in K))
        m.btes_dir_cap = pyo.Constraint(m.BTES_DIS_PRH, rule=btes_dir_cap_rule)

        def btes_dis_ua_rule(m, p, y, rd, h):          # (2) power fades with store temperature
            M_base = qdis_hard[p] * (1.0 + UA * k_wall) + UA * (Tr_rd[rd][h] - T_floor_p[p]) + 1.0
            return (m.btes_Qdis[p, rd, h] <= UA * (_Twall(m, p, y, rd, h) - Tr_rd[rd][h])
                    + M_base * (1 - m.btes_dir[p, rd, h]))
        m.btes_dis_ua = pyo.Constraint(m.BTES_DIS_PYRH, rule=btes_dis_ua_rule)
        logger.info(f"BTES temperature rule ON (RD): UA_dis {UA:.2f} kW/K (cap ref {_ua_ref:.1f}), "
                    f"k_wall {k_wall:.2e} K/kW, {len(prh)} gate binaries, "
                    f"{len(discharge_rep_days)} discharge rep days"
                    + (f"; flow levels {[l['m'] for l in lv]} kg/s/bh, UA {[round(l['UA'],1) for l in lv]} kW/K, "
                       f"{len(prh)*len(K)} level binaries" if use_levels else ""))
    else:
        logger.info("BTES temperature rule OFF (RD): fixed Tmin floor + warm-up binaries")

    # ------------------------------------------------------------------
    # C7: Annual energy balance guard
    # ------------------------------------------------------------------
    def btes_energy_balance_rule(m, p):
        if not active[p]:
            return pyo.Constraint.Skip
        # Weight each representative day by the number of calendar days it stands
        # for: charging and discharging rep days carry different weights, so an
        # unweighted comparison is not the annual balance it is meant to be (it
        # rescales the two sides by their season-mean weights, and whether that
        # is conservative or permissive is an accident of the clustering).
        return (
            pyo.quicksum(m.btes_Qdis[p, rd, h] * m.day_weight[p, rd]
                         for rd in m.REP_DAYS for h in m.HOURS)
            <= pyo.quicksum(m.btes_Qch[p, rd, h] * m.day_weight[p, rd]
                            for rd in m.REP_DAYS for h in m.HOURS)
        )
    m.btes_energy_balance = pyo.Constraint(m.PERIODS, rule=btes_energy_balance_rule)


# ─────────────────────────────────────────────────────────────────────────────
#  RH MODEL  (representative hours, interval-based storage levels)
# ─────────────────────────────────────────────────────────────────────────────

def add_btes_to_rh_model(
    m: pyo.ConcreteModel,
    investment_years: List[int],
    sigma: List[int],
    intervals: List[Dict[str, Any]],
    period_params: Dict[int, Dict[str, float]],
    arx: Dict[str, Any],
) -> None:
    """
    Add BTES to the RH (representative-hour) model.

    Dispatch is indexed on REP_HOURS (period resolution).
    Temperature state is indexed on BTES_YEARS × STOR_LEVELS (individual year resolution).

    ARX per interval j of duration dur[j] hours, with rep_hour rh[j]:
        T[y, j+1] = T[y, j] + b_ch * Qch[period(y), rh[j]] * dur[j]
                              + b_dis * Qdis[period(y), rh[j]] * dur[j]

    Year-to-year linking:
        T[y+1, 0] = T[y, J-1]
    """
    periods = sorted(investment_years)
    period_length = periods[1] - periods[0] if len(periods) > 1 else 5

    a_h   = float(arx.get("a_h", 1.0))
    c_h   = float(arx.get("c_h", 0.0))
    Tmax  = float(arx["Tmax"])
    Tmin  = float(arx["Tmin"])
    J     = len(intervals)

    # Age-block-specific ARX coefficients (fall back to arx globals if not in period_params)
    b_ch_p  = {p: float(period_params[p].get("b_ch_h",  arx["b_ch_h"])) for p in periods}
    b_dis_p = {p: float(period_params[p].get("b_dis_h", arx["b_dis_h"])) for p in periods}

    qch_cap = {p: period_params[p]["Qch_max_kW"]  for p in periods}
    qdis_cap= {p: period_params[p]["Qdis_max_kW"] for p in periods}
    active  = {p: period_params[p]["active"]        for p in periods}

    first_active = min(p for p in periods if active[p])
    T_g = period_params[first_active]["T_init"]

    btes_years, year_to_period, warmup_years, year_pairs = _build_year_structures(
        periods, active, period_length
    )

    # Map actual hours to their rep hour, classify by majority season
    sigma_slots: Dict[int, List[int]] = defaultdict(list)
    for actual_h, rep_i in enumerate(sigma):
        sigma_slots[rep_i].append(actual_h)
    rh_season: Dict[int, str] = {
        i: _majority([_hour_season(h) for h in hours])
        for i, hours in sigma_slots.items()
    }

    rep_h_of = {j: intervals[j]["rep_hour"] for j in range(J)}
    dur_of   = {j: intervals[j]["duration"] for j in range(J)}

    # Precompute per-interval ARX scalars (pure Python, outside Pyomo rules).
    # Expand dur[j] hourly steps: T[j+1] = decay[j]·T[j] + geo[j]·(b·Q + c_h)
    # decay[j] = a_h^dur[j]
    # geo[j]   = Σ_{k=0}^{dur-1} a_h^k = (1-a_h^dur)/(1-a_h) for a_h≠1, else dur
    # When a_h=1: decay=1, geo=dur → reduces to T[j]+(b·Q+c_h)·dur (old form).
    _decay = {}
    _geo   = {}
    for j in range(J):
        dur = dur_of[j]
        _decay[j] = a_h ** dur
        _geo[j]   = float(dur) if abs(a_h - 1.0) < 1e-10 else (1.0 - a_h ** dur) / (1.0 - a_h)

    # ------------------------------------------------------------------
    # Sets and variables
    # ------------------------------------------------------------------
    m.BTES_YEARS = pyo.Set(initialize=btes_years, ordered=True)

    m.btes_Qch  = pyo.Var(m.PERIODS, m.REP_HOURS,  domain=pyo.NonNegativeReals)
    m.btes_Qdis = pyo.Var(m.PERIODS, m.REP_HOURS,  domain=pyo.NonNegativeReals)
    m.btes_T    = pyo.Var(m.BTES_YEARS, m.STOR_LEVELS, domain=pyo.NonNegativeReals)

    # ------------------------------------------------------------------
    # C1/C2: Seasonal capacity windows
    # ------------------------------------------------------------------
    def btes_charge_cap_rule(m, p, i):
        if not active[p] or rh_season.get(i, 'idle') != 'charge':
            return m.btes_Qch[p, i] == 0
        return m.btes_Qch[p, i] <= qch_cap[p]
    m.btes_charge_cap = pyo.Constraint(m.PERIODS, m.REP_HOURS, rule=btes_charge_cap_rule)

    def btes_discharge_cap_rule(m, p, i):
        if not active[p] or rh_season.get(i, 'idle') != 'discharge':
            return m.btes_Qdis[p, i] == 0
        return m.btes_Qdis[p, i] <= qdis_cap[p]
    m.btes_discharge_cap = pyo.Constraint(m.PERIODS, m.REP_HOURS, rule=btes_discharge_cap_rule)

    # ------------------------------------------------------------------
    # C2.5: Warm-up discharge guard (MILP, storage-level resolution)
    #   btes_warm[y, j] = 1 iff T[y, j] >= Tmin.
    #   Qdis[period(y), rep_h_of[j]] <= Qdis_max * btes_warm[y, j] for ALL warmup years.
    # ------------------------------------------------------------------
    discharge_j = [j for j in range(J) if rh_season.get(rep_h_of[j], 'idle') == 'discharge']
    warmup_dis_pairs = [(y, j) for y in sorted(warmup_years) for j in discharge_j]
    if warmup_dis_pairs:
        M_btes = Tmax - T_g + 1.0
        m.BTES_WARMUP_DIS = pyo.Set(initialize=warmup_dis_pairs, dimen=2, ordered=True)
        m.btes_warm = pyo.Var(m.BTES_WARMUP_DIS, domain=pyo.Binary)

        def btes_warm_T_lb_rule(m, y, j):
            return m.btes_T[y, j] >= Tmin - M_btes * (1 - m.btes_warm[y, j])
        m.btes_warm_T_lb = pyo.Constraint(m.BTES_WARMUP_DIS, rule=btes_warm_T_lb_rule)

        def btes_warm_T_ub_rule(m, y, j):
            return m.btes_T[y, j] <= Tmin + M_btes * m.btes_warm[y, j]
        m.btes_warm_T_ub = pyo.Constraint(m.BTES_WARMUP_DIS, rule=btes_warm_T_ub_rule)

        def btes_dis_warm_guard_rule(m, y, j):
            p = year_to_period[y]
            i = rep_h_of[j]
            return m.btes_Qdis[p, i] <= qdis_cap[p] * m.btes_warm[y, j]
        m.btes_dis_warm_guard = pyo.Constraint(m.BTES_WARMUP_DIS, rule=btes_dis_warm_guard_rule)

    # ------------------------------------------------------------------
    # C3: Ground temperature for the first active year
    # ------------------------------------------------------------------
    def btes_T_init_rule(m, y):
        if y != first_active:
            return pyo.Constraint.Skip
        return m.btes_T[y, 0] == T_g
    m.btes_T_init = pyo.Constraint(m.BTES_YEARS, rule=btes_T_init_rule)

    # ------------------------------------------------------------------
    # C4: Year-to-year linking
    # ------------------------------------------------------------------
    if year_pairs:
        m.BTES_YEAR_LINKS = pyo.Set(initialize=year_pairs, dimen=2)

        def btes_year_link_rule(m, y_curr, y_next):
            return m.btes_T[y_next, 0] == m.btes_T[y_curr, J - 1]
        m.btes_year_link = pyo.Constraint(m.BTES_YEAR_LINKS, rule=btes_year_link_rule)

    # ------------------------------------------------------------------
    # C5: ARX per storage-level interval — exact for any a_h (see module docstring).
    # ------------------------------------------------------------------
    def btes_arx_rule(m, y, j):
        if j == J - 1:
            return pyo.Constraint.Skip
        p = year_to_period[y]
        i = rep_h_of[j]
        return m.btes_T[y, j + 1] == (
            _decay[j] * m.btes_T[y, j]
            + b_ch_p[p]  * m.btes_Qch[p, i] * _geo[j]
            + b_dis_p[p] * m.btes_Qdis[p, i] * _geo[j]
            + c_h * _geo[j]
        )
    m.btes_arx = pyo.Constraint(m.BTES_YEARS, m.STOR_LEVELS, rule=btes_arx_rule)

    # ------------------------------------------------------------------
    # C6: Temperature bounds
    # ------------------------------------------------------------------
    def btes_T_ub_rule(m, y, j):
        return m.btes_T[y, j] <= Tmax
    m.btes_T_ub = pyo.Constraint(m.BTES_YEARS, m.STOR_LEVELS, rule=btes_T_ub_rule)

    def btes_T_lb_rule(m, y, j):
        if y in warmup_years:
            return pyo.Constraint.Skip
        return m.btes_T[y, j] >= Tmin
    m.btes_T_lb = pyo.Constraint(m.BTES_YEARS, m.STOR_LEVELS, rule=btes_T_lb_rule)

    # ------------------------------------------------------------------
    # C7: Annual energy balance guard (weighted by occurrence count)
    # ------------------------------------------------------------------
    def btes_energy_balance_rule(m, p):
        if not active[p]:
            return pyo.Constraint.Skip
        occ = {i: len(sigma_slots[i]) for i in sigma_slots}
        return (
            pyo.quicksum(m.btes_Qdis[p, i] * occ.get(i, 1) for i in m.REP_HOURS)
            <= pyo.quicksum(m.btes_Qch[p, i]  * occ.get(i, 1) for i in m.REP_HOURS)
        )
    m.btes_energy_balance = pyo.Constraint(m.PERIODS, rule=btes_energy_balance_rule)


# ─────────────────────────────────────────────────────────────────────────────
#  RD MODEL, ENDOGENOUS FIELD DESIGN
#  One ARX parameter set per candidate field size; a binary picks at most one.
# ─────────────────────────────────────────────────────────────────────────────

def add_btes_designs_to_rd_model(
    m: pyo.ConcreteModel,
    investment_years: List[int],
    n_days: int,
    day_to_rep: List[int],
    designs: Dict[str, Dict[str, Any]],
    period_params_by_design: Dict[str, Dict[int, Dict[str, Any]]],
    build_period: int,
) -> None:
    """
    Add HT-BTES with the field design as a decision.

    For every candidate design k the model carries its own charge/discharge
    flows and its own borehole-temperature trajectory, governed by the ARX
    coefficients fitted to that design's simulation. A binary y_k selects at
    most one design; the flows of every unselected design are forced to zero,
    so its temperature state is inert and its cost is not incurred.

    The aggregate flows m.btes_Qch / m.btes_Qdis are kept (as the sum over
    designs) so the heat balance, the objective and the reporting code work
    unchanged.
    """
    periods = sorted(investment_years)
    period_length = periods[1] - periods[0] if len(periods) > 1 else 5
    keys = list(designs)

    m.BTES_DESIGNS = pyo.Set(initialize=keys, ordered=True)
    m.btes_y = pyo.Var(m.BTES_DESIGNS, domain=pyo.Binary)

    # at most one field is built
    m.btes_one_design = pyo.Constraint(
        expr=pyo.quicksum(m.btes_y[k] for k in m.BTES_DESIGNS) <= 1
    )

    # season of each representative day (identical for all designs)
    rep_to_days: Dict[int, List[int]] = defaultdict(list)
    for d, rd in enumerate(day_to_rep):
        rep_to_days[rd].append(d)
    rd_season = {rd: _majority([_day_season(d) for d in days])
                 for rd, days in rep_to_days.items()}
    discharge_rep_days = [rd for rd, s in rd_season.items() if s == 'discharge']
    rep_to_day = {rd: rep_to_days[rd][0] for rd in discharge_rep_days}

    active = {k: {p: period_params_by_design[k][p]["active"] for p in periods}
              for k in keys}
    first_active = min(p for p in periods if active[keys[0]][p])
    btes_years, year_to_period, warmup_years, year_pairs = _build_year_structures(
        periods, active[keys[0]], period_length
    )
    m.BTES_YEARS = pyo.Set(initialize=btes_years, ordered=True)

    # per-design flows and temperature state
    m.btes_qch_k = pyo.Var(m.BTES_DESIGNS, m.PERIODS, m.REP_DAYS, m.HOURS,
                           domain=pyo.NonNegativeReals)
    m.btes_qdis_k = pyo.Var(m.BTES_DESIGNS, m.PERIODS, m.REP_DAYS, m.HOURS,
                            domain=pyo.NonNegativeReals)
    m.btes_T_k = pyo.Var(m.BTES_DESIGNS, m.BTES_YEARS, m.DAYS,
                         domain=pyo.NonNegativeReals)

    # aggregates used by the heat balance and the reporting code
    m.btes_Qch = pyo.Var(m.PERIODS, m.REP_DAYS, m.HOURS, domain=pyo.NonNegativeReals)
    m.btes_Qdis = pyo.Var(m.PERIODS, m.REP_DAYS, m.HOURS, domain=pyo.NonNegativeReals)

    def agg_ch_rule(m, p, rd, h):
        return m.btes_Qch[p, rd, h] == pyo.quicksum(
            m.btes_qch_k[k, p, rd, h] for k in m.BTES_DESIGNS)
    m.btes_agg_ch = pyo.Constraint(m.PERIODS, m.REP_DAYS, m.HOURS, rule=agg_ch_rule)

    def agg_dis_rule(m, p, rd, h):
        return m.btes_Qdis[p, rd, h] == pyo.quicksum(
            m.btes_qdis_k[k, p, rd, h] for k in m.BTES_DESIGNS)
    m.btes_agg_dis = pyo.Constraint(m.PERIODS, m.REP_DAYS, m.HOURS, rule=agg_dis_rule)

    # ── seasonal windows and capacities, gated by the design binary ──────────
    def ch_cap_rule(m, k, p, rd, h):
        pp = period_params_by_design[k][p]
        if not pp["active"] or rd_season[rd] != 'charge':
            return m.btes_qch_k[k, p, rd, h] == 0
        return m.btes_qch_k[k, p, rd, h] <= pp["Qch_max_kW"] * m.btes_y[k]
    m.btes_ch_cap_k = pyo.Constraint(m.BTES_DESIGNS, m.PERIODS, m.REP_DAYS,
                                     m.HOURS, rule=ch_cap_rule)

    def dis_cap_rule(m, k, p, rd, h):
        pp = period_params_by_design[k][p]
        if not pp["active"] or rd_season[rd] != 'discharge':
            return m.btes_qdis_k[k, p, rd, h] == 0
        return m.btes_qdis_k[k, p, rd, h] <= pp["Qdis_max_kW"] * m.btes_y[k]
    m.btes_dis_cap_k = pyo.Constraint(m.BTES_DESIGNS, m.PERIODS, m.REP_DAYS,
                                      m.HOURS, rule=dis_cap_rule)

    # ── ARX temperature trajectory, one per design ───────────────────────────
    def arx_rule(m, k, y, d):
        if d == n_days - 1:
            return pyo.Constraint.Skip
        p = year_to_period[y]
        rd = day_to_rep[d]
        a_h = float(designs[k]["a_h"])
        hw = [a_h ** (23 - h) for h in range(24)]
        pp = period_params_by_design[k][p]
        return m.btes_T_k[k, y, d + 1] == (
            a_h ** 24 * m.btes_T_k[k, y, d]
            + pp["b_ch_h"] * pyo.quicksum(hw[h] * m.btes_qch_k[k, p, rd, h]
                                          for h in m.HOURS)
            + pp["b_dis_h"] * pyo.quicksum(hw[h] * m.btes_qdis_k[k, p, rd, h]
                                           for h in m.HOURS)
            + float(designs[k]["c_h"]) * sum(hw)
        )
    m.btes_arx_k = pyo.Constraint(m.BTES_DESIGNS, m.BTES_YEARS, m.DAYS, rule=arx_rule)

    def t_init_rule(m, k, y):
        if y != first_active:
            return pyo.Constraint.Skip
        return m.btes_T_k[k, y, 0] == float(designs[k]["T_g"])
    m.btes_T_init_k = pyo.Constraint(m.BTES_DESIGNS, m.BTES_YEARS, rule=t_init_rule)

    if year_pairs:
        m.BTES_YEAR_LINKS = pyo.Set(initialize=year_pairs, dimen=2)

        def link_rule(m, k, y_curr, y_next):
            return m.btes_T_k[k, y_next, 0] == m.btes_T_k[k, y_curr, n_days - 1]
        m.btes_year_link_k = pyo.Constraint(m.BTES_DESIGNS, m.BTES_YEAR_LINKS,
                                            rule=link_rule)

    # ── temperature bounds: only binding for the selected design ─────────────
    def t_ub_rule(m, k, y, d):
        return m.btes_T_k[k, y, d] <= float(designs[k]["Tmax"])
    m.btes_T_ub_k = pyo.Constraint(m.BTES_DESIGNS, m.BTES_YEARS, m.DAYS, rule=t_ub_rule)

    def t_lb_rule(m, k, y, d):
        if y in warmup_years:
            return pyo.Constraint.Skip
        Tmin = float(designs[k]["Tmin"])
        return m.btes_T_k[k, y, d] >= Tmin - Tmin * (1 - m.btes_y[k])
    m.btes_T_lb_k = pyo.Constraint(m.BTES_DESIGNS, m.BTES_YEARS, m.DAYS, rule=t_lb_rule)

    # ── warm-up discharge gate, per design ───────────────────────────────────
    warm_pairs = [(k, y, rep_to_day[rd]) for k in keys
                  for y in sorted(warmup_years) for rd in discharge_rep_days]
    if warm_pairs:
        m.BTES_WARM_K = pyo.Set(initialize=sorted(set(warm_pairs)), dimen=3,
                                ordered=True)
        m.btes_warm_k = pyo.Var(m.BTES_WARM_K, domain=pyo.Binary)

        def warm_lb(m, k, y, d):
            Tmin = float(designs[k]["Tmin"])
            M = float(designs[k]["Tmax"]) - float(designs[k]["T_g"]) + 1.0
            return m.btes_T_k[k, y, d] >= Tmin - M * (1 - m.btes_warm_k[k, y, d])
        m.btes_warm_lb_k = pyo.Constraint(m.BTES_WARM_K, rule=warm_lb)

        def warm_ub(m, k, y, d):
            Tmin = float(designs[k]["Tmin"])
            M = float(designs[k]["Tmax"]) - float(designs[k]["T_g"]) + 1.0
            return m.btes_T_k[k, y, d] <= Tmin + M * m.btes_warm_k[k, y, d]
        m.btes_warm_ub_k = pyo.Constraint(m.BTES_WARM_K, rule=warm_ub)

        guard = [(k, y, rd) for k in keys for y in sorted(warmup_years)
                 for rd in discharge_rep_days]
        m.BTES_WARM_RD_K = pyo.Set(initialize=guard, dimen=3, ordered=True)

        def warm_guard(m, k, y, rd):
            p = year_to_period[y]
            d = rep_to_day[rd]
            cap = period_params_by_design[k][p]["Qdis_max_kW"]
            return (pyo.quicksum(m.btes_qdis_k[k, p, rd, h] for h in m.HOURS)
                    <= cap * 24 * m.btes_warm_k[k, y, d])
        m.btes_dis_warm_guard_k = pyo.Constraint(m.BTES_WARM_RD_K, rule=warm_guard)

    # ── seasonal energy balance guard, weighted, per design ──────────────────
    def balance_rule(m, k, p):
        if not period_params_by_design[k][p]["active"]:
            return pyo.Constraint.Skip
        return (pyo.quicksum(m.btes_qdis_k[k, p, rd, h] * m.day_weight[p, rd]
                             for rd in m.REP_DAYS for h in m.HOURS)
                <= pyo.quicksum(m.btes_qch_k[k, p, rd, h] * m.day_weight[p, rd]
                                for rd in m.REP_DAYS for h in m.HOURS))
    m.btes_energy_balance_k = pyo.Constraint(m.BTES_DESIGNS, m.PERIODS,
                                             rule=balance_rule)
