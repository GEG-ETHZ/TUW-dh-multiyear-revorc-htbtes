"""Generate a BTES design-library entry for an arbitrary field geometry.

The shipped library is 10 designs that all share H = 200 m and vary only the
borehole count. This reruns the same physics for any (n_x, n_y, H, B) so a
different depth can be evaluated:

  1. pygfunction g-function for the field
  2. daily forward simulation, 20 years, same seasons/temperatures as the
     original archive/btes_standalone/BTES/01_btes_simulation script
  3. ARX fit  T[d+1] = a_d*T[d] + b_ch*Qch_d + b_dis*Qdis_d + c_d,
     converted to hourly (a_h = a_d^(1/24), c_h = c_d/24), with the
     b coefficients refitted per 5-year age block

    python make_btes_design.py -N 150 --H 200 --label n150
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pygfunction as gt
from scipy.interpolate import interp1d

# ── ground / fluid / pipe: identical to the original simulation ──
LAMBDA_S, RHO_CP_S, T_G = 3.70, 2.537e6, 11.25
R_B_RADIUS, D_BURIAL = 0.075, 1.0
R_OUT, R_IN, D_S, K_P, K_G = 0.02108, 0.01725, 0.06216 / 2, 0.42, 2.0
RHO_F, CP_F, K_F, MU_F = 1034.9, 3827.4, 0.51, 0.00281
RHO_F_CF = RHO_F * CP_F
M_CH_PER_BH, M_DIS_PER_BH = 0.30, 0.0517          # kg/s per borehole
# Variable discharge flow (--variable-flow): flow raised per day, from M_DIS_PER_BH up
# to M_DIS_MAX, until the outlet just meets the setpoint (daily supply temperature
# when a network profile is given, else T_USE_MIN). Pump power not modelled.
VARIABLE_FLOW, M_DIS_MAX = False, 0.30
NET_SUP_DAILY = NET_RET_DAILY = None                # daily-mean network profiles (365,)
NET_RESID_DAILY = None                              # daily-mean residual demand above the geo ceiling (W)
# T_IN_CH / T_B_MAX default to the wellhead brine temperature of the Munich case
# (Kaufmann et al., ECM 2026: Malm aquifer, 106 C). A borehole store cannot be
# heated above the fluid charging it, so the wall cap equals the charge inlet.
# Override with --t-charge to explore a boosted charge stream.
T_IN_CH, T_B_MAX, T_DH_RET, T_USE_MIN = 106.0, 106.0, 55.0, 80.0
CH_START, CH_END, DIS_S1, DIS_E2 = 121, 273, 305, 90
N_YEARS = 20
CAPEX_EUR_PER_M = 70.0


def disc_field(N_b, B, H, D=None, r_b=None):
    """Filled cylindrical field: hex-packed boreholes clipped to a circle.

    The literature consensus (Skarphagen et al. 2019; IEA SHC Task 45) is that a
    cylindrical array minimises the surface-area-to-volume ratio and therefore
    the conductive loss from the store. Hexagonal packing at pitch B gives the
    densest arrangement; taking the N_b boreholes nearest the centre yields the
    most compact — i.e. most nearly circular — field for that count.
    """
    D = D_BURIAL if D is None else D
    r_b = R_B_RADIUS if r_b is None else r_b
    pts = []
    R = int(np.ceil(np.sqrt(N_b))) + 4
    for j in range(-R, R + 1):
        for i in range(-R, R + 1):
            x = B * (i + 0.5 * (j % 2))
            y = B * j * np.sqrt(3) / 2.0
            pts.append((np.hypot(x, y), x, y))
    pts.sort()
    return [gt.boreholes.Borehole(H, D, r_b, x, y) for _, x, y in pts[:N_b]]


def hx_coefficients(N_b_target, H, B=3.0, layout="disc", m_dis_per_bh=None):
    """Field + lumped heat-exchange coefficients (W/K) for charge and discharge.

    dis_coeff is the UA that the simulation itself uses: Q_dis = dis_coeff*(T_wall - T_return).
    Exposed so the dispatch model can apply the same temperature-dependent discharge limit.
    """
    if layout == "rect":
        n = int(np.ceil(np.sqrt(N_b_target)))
        boreholes = gt.boreholes.rectangle_field(n, N_b_target // n, B, B, H,
                                                 D_BURIAL, R_B_RADIUS)
    else:
        boreholes = disc_field(N_b_target, B, H)
    N_b = len(boreholes); L_total = N_b * H
    m_dis_bh = M_DIS_PER_BH if m_dis_per_bh is None else float(m_dis_per_bh)
    v_ch, v_dis = M_CH_PER_BH * N_b / RHO_F, m_dis_bh * N_b / RHO_F
    bh = gt.boreholes.Borehole(H, D_BURIAL, R_B_RADIUS, 0.0, 0.0)
    pos = [(-D_S, 0.0), (D_S, 0.0)]
    def R_b(m_per_bh):
        h_f = gt.pipes.convective_heat_transfer_coefficient_circular_pipe(
            m_per_bh, R_IN, MU_F, RHO_F, K_F, CP_F, 1e-6)
        R_fp = 1.0 / (h_f * 2 * np.pi * R_IN) + \
            gt.pipes.conduction_thermal_resistance_circular_pipe(R_IN, R_OUT, K_P)
        p = gt.pipes.SingleUTube(pos, R_IN, R_OUT, bh, LAMBDA_S, K_G, R_fp)
        return gt.pipes.borehole_thermal_resistance(p, m_per_bh, CP_F)
    R_b_ch, R_b_dis = R_b(M_CH_PER_BH), R_b(m_dis_bh)
    ch_coeff = v_ch * RHO_F_CF / (0.5 + v_ch * RHO_F_CF * R_b_ch / L_total)
    dis_coeff = v_dis * RHO_F_CF / (0.5 + v_dis * RHO_F_CF * R_b_dis / L_total)
    return boreholes, N_b, L_total, ch_coeff, dis_coeff


def wall_step_coefficient(N_b_target, H, B=3.0, layout="disc"):
    """K per kW of daily-mean extraction: 1-day step g-function / (2 pi lambda) / L_total."""
    boreholes = (disc_field(N_b_target, B, H) if layout == "disc" else
                 gt.boreholes.rectangle_field(int(np.ceil(np.sqrt(N_b_target))),
                                              N_b_target // int(np.ceil(np.sqrt(N_b_target))),
                                              B, B, H, D_BURIAL, R_B_RADIUS))
    g1 = gt.gfunction.uniform_temperature(boreholes, np.array([86400.0]), LAMBDA_S / RHO_CP_S,
                                          nSegments=1, disp=False)
    return float(g1[0]) / (2 * np.pi * LAMBDA_S) * 1000.0 / (len(boreholes) * H)


def simulate(N_b_target, H, B=3.0, layout="disc"):
    """Simulate one field. layout='disc' (cylindrical, default) or 'rect'."""
    if layout == "rect":
        n = int(np.ceil(np.sqrt(N_b_target)))
        boreholes = gt.boreholes.rectangle_field(n, N_b_target // n, B, B, H,
                                                 D_BURIAL, R_B_RADIUS)
    else:
        boreholes = disc_field(N_b_target, B, H)
    N_b = len(boreholes)
    L_total = N_b * H
    m_ch, m_dis = M_CH_PER_BH * N_b, M_DIS_PER_BH * N_b
    v_ch, v_dis = m_ch / RHO_F, m_dis / RHO_F

    bh = gt.boreholes.Borehole(H, D_BURIAL, R_B_RADIUS, 0.0, 0.0)
    pos = [(-D_S, 0.0), (D_S, 0.0)]

    def R_b(m_per_bh):
        h_f = gt.pipes.convective_heat_transfer_coefficient_circular_pipe(
            m_per_bh, R_IN, MU_F, RHO_F, K_F, CP_F, 1e-6)
        R_fp = 1.0 / (h_f * 2 * np.pi * R_IN) + \
            gt.pipes.conduction_thermal_resistance_circular_pipe(R_IN, R_OUT, K_P)
        p = gt.pipes.SingleUTube(pos, R_IN, R_OUT, bh, LAMBDA_S, K_G, R_fp)
        return gt.pipes.borehole_thermal_resistance(p, m_per_bh, CP_F)

    R_b_ch, R_b_dis = R_b(M_CH_PER_BH), R_b(M_DIS_PER_BH)
    ch_coeff = v_ch * RHO_F_CF / (0.5 + v_ch * RHO_F_CF * R_b_ch / L_total)
    dis_coeff = v_dis * RHO_F_CF / (0.5 + v_dis * RHO_F_CF * R_b_dis / L_total)

    # Variable discharge flow: table of (m per bh, UA_field [W/K], C_field [W/K]) from
    # the base flow up to M_DIS_MAX; per day the largest flow whose outlet still
    # meets the setpoint is used. Outlet temperature falls monotonically with flow.
    if VARIABLE_FLOW:
        m_grid = np.geomspace(M_DIS_PER_BH, M_DIS_MAX, 60)
        ua_grid = np.array([(m * N_b / RHO_F) * RHO_F_CF /
                            (0.5 + (m * N_b / RHO_F) * RHO_F_CF * R_b(m) / L_total) for m in m_grid])
        c_grid = m_grid * N_b * CP_F

    def _discharge(Tb, Tret, Tset, resid=None):
        """Return (Q_W extracted >= 0, flow per bh) for one discharge day."""
        if not VARIABLE_FLOW:
            Q_av = dis_coeff * (Tb - Tret)
            T_out = Tret + Q_av / (v_dis * RHO_F_CF) if Q_av > 0 else Tb
            return (Q_av, M_DIS_PER_BH) if (T_out >= Tset and Q_av > 0) else (0.0, 0.0)
        if Tb <= Tret:
            return 0.0, 0.0
        Q_g = ua_grid * (Tb - Tret)
        T_out = Tret + Q_g / c_grid
        ok = np.where(T_out >= Tset)[0]
        if len(ok) == 0:
            return 0.0, 0.0
        i = ok[-1]                       # largest admissible flow on the grid
        if i + 1 < len(m_grid):          # refine between grid points i and i+1
            w = (T_out[i] - Tset) / (T_out[i] - T_out[i + 1])
            Q, mflow = Q_g[i] + w * (Q_g[i + 1] - Q_g[i]), m_grid[i] + w * (m_grid[i + 1] - m_grid[i])
        else:
            Q, mflow = Q_g[i], m_grid[i]
        if resid is not None and Q > resid:      # demand-driven: never more than the day needs
            if resid <= 0:
                return 0.0, 0.0
            j = int(np.searchsorted(Q_g, resid))  # smallest flow that delivers the residual
            return resid, float(m_grid[min(j, len(m_grid) - 1)])
        return Q, mflow

    dt_s, N_days = 86400.0, N_YEARS * 365
    _x = np.array([b.x for b in boreholes]); _y = np.array([b.y for b in boreholes])
    _d = 2 * (np.hypot(_x - _x.mean(), _y - _y.mean()).max() + B / 2)
    print(f"  {layout} field, {N_b} boreholes, H={H} m, total {L_total:,.0f} m, "
          f"diameter {_d:.1f} m, aspect H/D {H/_d:.2f}")
    print(f"  computing g-function …")
    t_log = np.geomspace(dt_s, N_days * dt_s, 200)
    g_log = gt.gfunction.uniform_temperature(boreholes, t_log, LAMBDA_S / RHO_CP_S,
                                             nSegments=1, disp=False)
    h_step = interp1d(t_log, g_log, kind="linear", fill_value="extrapolate")(
        np.arange(1, N_days + 1) * dt_s) / (2 * np.pi * LAMBDA_S)

    T_b = np.full(N_days + 1, T_G)
    q = np.zeros(N_days)
    Q_W = np.zeros(N_days)
    m_used = np.zeros(N_days)
    doys = (np.arange(N_days) % 365) + 1
    for d in range(N_days):
        Tb, doy = T_b[d], doys[d]
        if CH_START <= doy <= CH_END:
            Q = max(ch_coeff * (T_IN_CH - Tb), 0.0) if Tb < T_B_MAX else 0.0
        elif doy >= DIS_S1 or doy <= DIS_E2:
            Tret = NET_RET_DAILY[doy - 1] if NET_RET_DAILY is not None else T_DH_RET
            Tset = NET_SUP_DAILY[doy - 1] if NET_SUP_DAILY is not None else T_USE_MIN
            resid = NET_RESID_DAILY[doy - 1] if NET_RESID_DAILY is not None else None
            Q_av, m_used[d] = _discharge(Tb, Tret, Tset, resid)
            Q = -Q_av
        else:
            Q = 0.0
        Q_W[d] = Q
        q[d] = Q / L_total
        dq = np.diff(np.concatenate(([0.0], q[:d + 1])))
        T_b[d + 1] = T_G + float(np.dot(dq, h_step[d::-1][:d + 1]))
    # short-term wall response: 1-day step g-function -> K per kW of daily-mean extraction
    k_wall = float(h_step[0]) * 1000.0 / L_total
    return pd.DataFrame({"day": np.arange(N_days), "T_b": T_b[:-1],
                         "T_b_next": T_b[1:], "Q_kWh_d": Q_W * 24 / 1000.0,
                         "m_dis_kg_s_per_bh": m_used}), N_b, L_total, k_wall


def _fit_b(s, a_d, c_d, eta=None):
    """Charge/discharge sensitivities on the temperature residual.

    With a_d = 1 and c_d = 0 the recursion is a pure accumulator, so a cycle
    returning to its starting temperature satisfies

        sum(Qdis) / sum(Qch) = b_ch / |b_dis|

    i.e. b_ch/|b_dis| IS the round-trip efficiency the MILP can realise. Fitting
    b_ch and b_dis independently leaves that ratio unconstrained: it comes out
    above 100 % for the larger fields, which hands the optimiser free energy, and
    well above the simulated efficiency for the rest.

    Passing `eta` pins b_dis = -b_ch/eta and fits the single remaining parameter,
    so the surrogate's achievable round trip equals the simulation's by
    construction. This restores the energy-conservation property that a
    state-of-charge storage formulation has structurally, while keeping the
    temperature state that the 80 degC usability threshold needs.
    """
    r = s.T_b_next.values - a_d * s.T_b.values - c_d
    X = np.column_stack([s.Qch.values, s.Qdis.values])
    if eta is None or not (0.0 < eta <= 1.0):
        b_ch, b_dis = np.linalg.lstsq(X, r, rcond=None)[0]
        if b_ch > 0 and abs(b_dis) < b_ch:          # never better than lossless
            z = X[:, 0] - X[:, 1]
            b_ch = float(np.dot(z, r) / np.dot(z, z))
            b_dis = -b_ch
        return float(b_ch), float(b_dis)
    z = X[:, 0] - X[:, 1] / eta
    b_ch = float(np.dot(z, r) / np.dot(z, z))
    return b_ch, -b_ch / eta


def _replay(s, a_d, c_d, b_ch, b_dis, k_wall=0.0):
    """Drive the fitted ARX with the simulation's own loads; return wall-temperature RMSE.

    The state is the bulk temperature T_s; the wall seen by the fluid is
    T_b = T_s - k_wall * Qdis_kW (short-term ground response, 1-day step)."""
    T = float(s.T_b.iloc[0]) + k_wall * float(s.Qdis.iloc[0]) / 24.0; err = []
    for Tb, qc, qd in zip(s.T_b.values, s.Qch.values, s.Qdis.values):
        err.append(T - k_wall * qd / 24.0 - Tb)
        T = a_d * T + c_d + b_ch * qc + b_dis * qd
    return float(np.sqrt(np.mean(np.square(err))))


def fit(sim, k_wall=0.0):
    """ARX fit, then daily -> hourly, then per-age-block b.

    a_d and c_d are CONSTRAINED to 1 and 0 --- the field is a pure accumulator
    with no idle dynamics. Fitting them freely on the transition months reads the
    thermal rebound of a warm field as attraction toward c_d/(1-a_d), giving an
    idle equilibrium of 80-115 degC: a store that heats itself to usable
    temperature with no heat input at all. The original archive fit applied this
    same constraint for the same stated reason ("a_d=1.0, thermal rebound in
    transition"); it was dropped in a later regeneration.
    """
    sim = sim.copy()
    sim["Qch"] = sim.Q_kWh_d.clip(lower=0)
    sim["Qdis"] = (-sim.Q_kWh_d).clip(lower=0)
    if k_wall:                       # identify the accumulator on the bulk state, not the wall
        sim["T_b"] = sim.T_b + k_wall * sim.Qdis / 24.0
        sim["T_b_next"] = sim.T_b_next + k_wall * sim.Qdis.shift(-1).fillna(0.0) / 24.0
    a_d, c_d = 1.0, 0.0
    _eta_all = (sim.Qdis.sum() / sim.Qch.sum()) if sim.Qch.sum() > 0 else None
    b_ch_d, b_dis_d = _fit_b(sim, a_d, c_d, _eta_all)
    out = {"a_h": float(a_d ** (1 / 24)), "c_h": float(c_d / 24),
           "b_ch_h": float(b_ch_d), "b_dis_h": float(b_dis_d)}
    blocks = {}
    for blk in range(4):
        lo, hi = blk * 5 * 365, min((blk + 1) * 5 * 365, len(sim))
        s = sim.iloc[lo:hi]
        ch, dis = s.Qch.sum(), s.Qdis.sum()
        _eta = (dis / ch) if ch > 0 else None
        bb = _fit_b(s, a_d, c_d, _eta)
        blocks[str(blk)] = {"b_ch_h": float(bb[0]), "b_dis_h": float(bb[1]),
                            "T_init": float(s.T_b.iloc[0]),
                            "Qch_max_kW": float(s.Qch.max() / 24),
                            "Qdis_max_kW": float(s.Qdis.max() / 24),
                            "sim_RTE_pct": float(100 * dis / ch) if ch > 0 else 0.0}
    out["blocks"] = blocks
    out["Tmin"] = float(T_USE_MIN + 0.8)
    out["Tmax"] = float(T_B_MAX)
    out["T_g"] = float(T_G)
    out["k_wall_K_per_kW"] = float(k_wall)
    return out


def main():
    global T_IN_CH, T_B_MAX, T_DH_RET, T_USE_MIN
    ap = argparse.ArgumentParser()
    ap.add_argument("-N", "--n-boreholes", type=int, required=True)
    ap.add_argument("--H", type=float, required=True)
    ap.add_argument("--B", type=float, default=3.0)
    ap.add_argument("--layout", choices=("disc", "rect"), default="disc")
    ap.add_argument("--label", required=True)
    ap.add_argument("--t-charge", type=float, default=T_IN_CH,
                    help="charging inlet temperature and borehole-wall cap (degC)")
    ap.add_argument("--t-return", type=float, default=T_DH_RET,
                    help="network return temperature entering the store on discharge (degC)")
    ap.add_argument("--t-use-min", type=float, default=T_USE_MIN,
                    help="minimum useful outlet temperature on discharge (degC)")
    ap.add_argument("--m-dis", type=float, default=None, help="fixed discharge flow per borehole, kg/s (default 0.0517)")
    ap.add_argument("--variable-flow", action="store_true",
                    help="raise the discharge flow per day until the outlet just meets the setpoint")
    ap.add_argument("--m-max", type=float, default=0.30, help="discharge flow ceiling, kg/s per borehole")
    ap.add_argument("--net-profile", default=None,
                    help="DH data xlsx with hourly supply_temp/return_temp; daily means become the "
                         "discharge setpoint (supply) and return; overrides --t-return/--t-use-min")
    ap.add_argument("--net-shift", type=float, default=0.0, help="K subtracted from the profile temperatures")
    ap.add_argument("--demand-cap", action="store_true",
                    help="cap daily discharge at the daily-mean residual demand above --geo-ceiling "
                         "(column --demand-col of the profile); demand-driven instead of greedy")
    ap.add_argument("--geo-ceiling", type=float, default=17.378, help="MW covered before the store is needed")
    ap.add_argument("--demand-col", default="corrected_demand_2045")
    a = ap.parse_args()
    global VARIABLE_FLOW, M_DIS_MAX, NET_SUP_DAILY, NET_RET_DAILY, NET_RESID_DAILY, M_DIS_PER_BH
    if a.m_dis:
        M_DIS_PER_BH = float(a.m_dis)
    T_IN_CH = T_B_MAX = float(a.t_charge)
    T_DH_RET, T_USE_MIN = float(a.t_return), float(a.t_use_min)
    VARIABLE_FLOW, M_DIS_MAX = bool(a.variable_flow), float(a.m_max)
    net_txt = ""
    if a.net_profile:
        dh = pd.read_excel(a.net_profile)
        sup = dh["supply_temp"].to_numpy(float)[:8760] - a.net_shift
        ret = dh["return_temp"].to_numpy(float)[:8760] - a.net_shift
        NET_SUP_DAILY = sup.reshape(365, 24).mean(axis=1)
        NET_RET_DAILY = ret.reshape(365, 24).mean(axis=1)
        T_DH_RET, T_USE_MIN = float(NET_RET_DAILY.mean()), float(NET_SUP_DAILY.mean())
        if a.demand_cap:
            dem = dh[a.demand_col].to_numpy(float)[:8760]
            NET_RESID_DAILY = np.clip(dem - a.geo_ceiling, 0.0, None).reshape(365, 24).mean(axis=1) * 1e6  # W
        net_txt = (f", network profile {Path(a.net_profile).name} shifted -{a.net_shift:.0f} K "
                   f"(daily supply {NET_SUP_DAILY.min():.0f}-{NET_SUP_DAILY.max():.0f} C, "
                   f"return {NET_RET_DAILY.min():.0f}-{NET_RET_DAILY.max():.0f} C)")

    print(f"Generating BTES design '{a.label}'  (charge / wall cap {T_IN_CH:.1f} C, "
          f"return {T_DH_RET:.0f} C, min outlet {T_USE_MIN:.0f} C"
          + (f", variable discharge flow {M_DIS_PER_BH}-{M_DIS_MAX} kg/s/bh" if VARIABLE_FLOW else "")
          + net_txt + ")")
    sim, N_b, L, k_wall = simulate(a.n_boreholes, a.H, a.B, a.layout)
    p = fit(sim, k_wall)
    print(f"  wall feedthrough k_wall = {k_wall:.3e} K/kW  (1-day g-function step; "
          f"{k_wall*1e3:.1f} K per MW of daily-mean extraction)")
    entry = {"n_boreholes": N_b, "H_m": a.H, "total_length_m": L,
             "capex_EUR": CAPEX_EUR_PER_M * L,
             "capex_basis": f"drilling {CAPEX_EUR_PER_M:.0f} EUR/m x {N_b} boreholes x {a.H:.0f} m",
             "layout": a.layout, "B_m": a.B,
             "variable_flow": VARIABLE_FLOW, "m_dis_max_kg_s_per_bh": (M_DIS_MAX if VARIABLE_FLOW else M_DIS_PER_BH),
             "demand_cap": bool(a.demand_cap),
             "net_profile": (Path(a.net_profile).name if a.net_profile else None), "net_shift_K": a.net_shift,
             "lifetime_a": 50, **{k: v for k, v in p.items() if k != "blocks"},
             "blocks": {int(k): v for k, v in p["blocks"].items()}}
    here = Path(__file__).parent
    sim.to_csv(here / f"sim_{a.label}.csv", index=False)
    (here / f"design_{a.label}.json").write_text(json.dumps(entry, indent=2))
    print(f"\n  a_h={entry['a_h']:.8f}  c_h={entry['c_h']:.6f}")
    print(f"  b_ch_h={entry['b_ch_h']:.4e}  b_dis_h={entry['b_dis_h']:.4e}")
    for k, v in entry["blocks"].items():
        lo, hi = k * 5 * 365, min((k + 1) * 5 * 365, len(sim))
        s = sim.iloc[lo:hi].assign(Qch=sim.Q_kWh_d.clip(lower=0).iloc[lo:hi],
                                   Qdis=(-sim.Q_kWh_d).clip(lower=0).iloc[lo:hi])
        rmse = _replay(s, 1.0, 0.0, v["b_ch_h"], v["b_dis_h"], k_wall)
        ratio = 100 * v["b_ch_h"] / abs(v["b_dis_h"])
        print(f"    block {k}: b_ch {v['b_ch_h']:.4e}  ARX ratio {ratio:5.1f} %  "
              f"sim RTE {v['sim_RTE_pct']:5.1f} %  replay RMSE {rmse:5.2f} K")
    print(f"  wrote design_{a.label}.json")


if __name__ == "__main__":
    main()
