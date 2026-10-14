#!/usr/bin/env python3
"""
plot_btes_assumptions.py — one assumptions table per HT-BTES design.

Regenerates the assumptions summary for EVERY member of the design library,
so each borehole count gets its own figure. Unlike the earlier single-field
version, the charge/discharge power limits are shown PER AGE BLOCK, because
they are simulation outputs that evolve as the ground warms and they differ
between designs.

All values are read from:
  data/btes/design_library.json     per-design, per-age-block parameters
  data/btes/gen/make_btes_design.py the simulation constants
  pygfunction                       borehole thermal resistance

    python data/btes/gen/plot_btes_assumptions.py            # all designs
    python data/btes/gen/plot_btes_assumptions.py 150 200    # selected designs

Output: data/btes/gen/assumptions/btes_assumptions_n<N>.png
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import pygfunction as gt

HERE = Path(__file__).resolve().parent
LIB = json.loads((HERE.parent / "design_library.json").read_text())
_CSV = HERE.parent / "design_library.csv"
GRID = {}
if _CSV.exists():
    import csv
    with _CSV.open() as fh:
        for r in csv.DictReader(fh):
            GRID[int(float(r["n_boreholes"]))] = (
                f'{r["layout"]}, D={float(r["diameter_m"]):.1f} m, H/D={float(r["aspect_H_over_D"]):.2f}')
OUT = HERE / "assumptions"
OUT.mkdir(exist_ok=True)

# ── simulation constants (mirror make_btes_design.py) ───────────────────────
LAMBDA_S, RHO_CP_S, T_G = 3.70, 2.537e6, 11.25
R_B_RADIUS, D_BURIAL = 0.075, 1.0
R_OUT, R_IN, D_S, K_P, K_G = 0.02108, 0.01725, 0.06216 / 2, 0.42, 2.0
RHO_F, CP_F, K_F, MU_F = 1034.9, 3827.4, 0.51, 0.00281
M_CH_PER_BH, M_DIS_PER_BH = 0.30, 0.0517
# single source of truth: the simulation constants live in make_btes_design.py
sys.path.insert(0, str(HERE))
from make_btes_design import T_IN_CH, T_B_MAX, T_DH_RET, T_USE_MIN  # noqa: E402
N_YEARS, CAPEX_EUR_PER_M = 20, 70.0


def borehole_resistance(m_per_bh, H):
    bh = gt.boreholes.Borehole(H, D_BURIAL, R_B_RADIUS, 0.0, 0.0)
    pos = [(-D_S, 0.0), (D_S, 0.0)]
    h_f = gt.pipes.convective_heat_transfer_coefficient_circular_pipe(
        m_per_bh, R_IN, MU_F, RHO_F, K_F, CP_F, 1e-6)
    R_fp = (1.0 / (h_f * 2 * np.pi * R_IN)
            + gt.pipes.conduction_thermal_resistance_circular_pipe(R_IN, R_OUT, K_P))
    p = gt.pipes.SingleUTube(pos, R_IN, R_OUT, bh, LAMBDA_S, K_G, R_fp)
    Re = 4 * m_per_bh / (np.pi * 2 * R_IN * MU_F)
    return gt.pipes.borehole_thermal_resistance(p, m_per_bh, CP_F), Re


CAT_COLORS = {
    "Field Geometry": "#DCE9F5", "Ground": "#DEEDE2", "Fluid": "#FBEBD9",
    "Operation Limits": "#F7DDE4", "Seasonal Schedule": "#E4DDEF",
    "Simulation": "#D8ECEF", "ARX Surrogate": "#F7F2CE",
    "ARX per Age Block": "#EFECD5", "Dispatch Bounds": "#EADDEF",
    "Economics": "#DEE0EA",
}


def build_rows(N, e):
    H = e["H_m"]; L = e["total_length_m"]; blocks = e["blocks"]
    grid = GRID.get(N, f"{N} boreholes")
    Rb_ch, Re_ch = borehole_resistance(M_CH_PER_BH, H)
    Rb_di, Re_di = borehole_resistance(M_DIS_PER_BH, H)
    m_ch, m_dis = M_CH_PER_BH * N, M_DIS_PER_BH * N
    R = []

    def add(cat, p, v, n=""):
        R.append((cat, p, v, n))

    add("Field Geometry", "Configuration", grid, f"{N} boreholes total")
    add("Field Geometry", "Borehole depth  H", f"{H:.0f}", "m")
    add("Field Geometry", "Borehole spacing  B", "3.0", "m")
    add("Field Geometry", "Burial depth  D", f"{D_BURIAL:.1f}", "m")
    add("Field Geometry", "Borehole radius  r_b", f"{R_B_RADIUS:.3f}", "m")
    add("Field Geometry", "Total borehole length", f"{L:,.0f}", f"m  ({N} x {H:.0f} m)")

    add("Ground", "Thermal conductivity  lambda_s", f"{LAMBDA_S:.2f}", "W/(m.K)")
    add("Ground", "Volumetric heat capacity  rho_cp", f"{RHO_CP_S:.3e}", "J/(m3.K)")
    add("Ground", "Thermal diffusivity  alpha_s", f"{LAMBDA_S/RHO_CP_S:.2e}", "m2/s")
    add("Ground", "Undisturbed temperature  T_g", f"{T_G:.2f}", "degC")
    add("Ground", "R_b (charge)", f"{Rb_ch:.4f}",
        f"m.K/W  ({M_CH_PER_BH:.3f} kg/s/bh, Re ~ {Re_ch:.0f}, turbulent)")
    add("Ground", "R_b (discharge)", f"{Rb_di:.4f}",
        f"m.K/W  ({M_DIS_PER_BH:.4f} kg/s/bh, Re ~ {Re_di:.0f}, laminar)")

    add("Fluid", "Type", "EG 20%", "ethylene glycol 20 %")
    add("Fluid", "Volumetric heat cap.  rho_Cf", f"{RHO_F*CP_F:.3e}", "J/(m3.K)")
    add("Fluid", "Charge flow (whole field)", f"{m_ch:.1f} kg/s",
        f"{M_CH_PER_BH:.3f} kg/s per borehole")
    add("Fluid", "Discharge flow (whole field)", f"{m_dis:.2f} kg/s",
        f"{M_DIS_PER_BH:.4f} kg/s per borehole")

    add("Operation Limits", "Charging inlet temp  T_in", f"{T_IN_CH:.0f}",
        "degC  (fixed, high-temperature)")
    add("Operation Limits", "Min discharge  T_out", f"{T_USE_MIN:.0f}",
        "degC  (direct DH supply cutoff)")
    add("Operation Limits", "Max borehole temp  T_b_max", f"{T_B_MAX:.0f}",
        "degC  (stop charging above)")
    for b in range(4):
        q = blocks[str(b)]["Qch_max_kW"]
        add("Operation Limits", f"Max charge power  Q_ch  block {b}",
            f"{q:,.0f}", f"kW  ({1000*q/L:.1f} W/m x {L:,.0f} m)  yrs {b*5+1}-{b*5+5}")
    for b in range(4):
        q = blocks[str(b)]["Qdis_max_kW"]
        add("Operation Limits", f"Max discharge power  Q_dis  block {b}",
            f"{q:,.1f}", f"kW  ({1000*q/L:.1f} W/m x {L:,.0f} m)  yrs {b*5+1}-{b*5+5}")

    add("Seasonal Schedule", "Charging season", "May 1 - Sep 30", "DOY 121-273  (153 days)")
    add("Seasonal Schedule", "Discharging season", "Nov 1 - Mar 31", "DOY 305-365 + 1-90  (151 days)")
    add("Seasonal Schedule", "Transition (idle)", "Apr  and  Oct", "DOY 91-120, 274-304  (61 days)")

    add("Simulation", "Method", "Eskilson g-function", "pygfunction, UBWT BC, 1 segment/bh")
    add("Simulation", "Simulation horizon", f"{N_YEARS}", f"years  ({N_YEARS*365:,} days)")
    add("Simulation", "Time step", "daily", "hourly ARX derived analytically")
    add("Simulation", "Round-trip eff. year 20", f"{blocks['3']['sim_RTE_pct']:.1f}", "%  (energy ratio)")

    add("ARX Surrogate", "Model form", "hourly, kWh",
        "T(h+1) = a_h*T(h) + b_ch*Qch(h) + b_dis*Qdis(h) + c_h")
    add("ARX Surrogate", "Decay factor  a_h", f"{e['a_h']:.7f}", "/hr")
    add("ARX Surrogate", "Drift  c_h", f"{e['c_h']:.5f}", "degC/hr")
    add("ARX Surrogate", "Fit procedure", "constrained",
        "a=1, c=0 fixed (pure accumulator); b_dis pinned to -b_ch/RTE_sim per block")

    for b in range(4):
        bl = blocks[str(b)]
        add("ARX per Age Block", f"Block {b}  (op. years {b*5+1}-{b*5+5})",
            f"b_ch={bl['b_ch_h']:.3e}  b_dis={bl['b_dis_h']:.3e}",
            f"degC/kWh   simulated RTE = {bl['sim_RTE_pct']:.1f} %")

    add("Dispatch Bounds", "Min borehole temp  T_min", f"{e['Tmin']:.1f}",
        "degC  (discharge cutoff; T_out >= 80 degC)")
    add("Dispatch Bounds", "Max borehole temp  T_max", f"{e['Tmax']:.0f}", "degC  (charge cutoff)")
    add("Dispatch Bounds", "Initial temperature  T_init", f"{T_G:.2f}",
        "degC  (undisturbed ground; field starts cold)")

    add("Economics", "Investment cost  IK", f"{e['capex_EUR']:,.0f}",
        f"EUR  ({N} BHE x {H:.0f} m x {CAPEX_EUR_PER_M:.0f} EUR/m, drilling only)")
    add("Economics", "Specific cost", f"{CAPEX_EUR_PER_M:.0f}", "EUR/m of borehole")
    add("Economics", "Lifetime", f"{e['lifetime_a']:.0f}", "years")
    _mp = HERE.parent / f"btes_hourly_params_n{N}.json"
    _m = json.loads(_mp.read_text()) if _mp.exists() else {}
    add("Economics", "Fixed O&M", f"{_m.get('opex_fix_EUR_per_MWh_cap', 0):.1f}",
        "EUR per MWh of usable capacity per year")
    add("Economics", "Variable O&M", f"{_m.get('cost_discharge', 0):.2f} / {_m.get('cost_charge', 0):.2f}",
        "EUR/MWh discharged / charged")
    return R


def draw(N, e, rows):
    row_h_in = 0.235
    fig_h = 1.05 + row_h_in * (len(rows) + 3)
    fig, ax = plt.subplots(figsize=(15.0, fig_h))
    ax.axis("off")
    ax.set_title(f"HT-BTES Model Assumptions  —  {N} boreholes\n"
                 f"{e['total_length_m']:,.0f} m total  |  H={e['H_m']:.0f} m  |  "
                 f"T_in={T_IN_CH:.0f} degC  |  direct DH supply (T_out >= {T_USE_MIN:.0f} degC)  |  "
                 f"hourly ARX surrogate, 4 age blocks",
                 fontsize=11.5, fontweight="bold", linespacing=1.9, pad=10)

    col_labels = ["Category", "Parameter", "Value", "Unit / Note"]
    col_w = [0.155, 0.275, 0.185, 0.385]
    col_x = [sum(col_w[:i]) for i in range(len(col_w))]
    row_h = 1.0 / (len(rows) + 2.0)
    hdr_y = 1.0

    for xi, (lbl, cw) in enumerate(zip(col_labels, col_w)):
        ax.add_patch(FancyBboxPatch((col_x[xi], hdr_y - row_h), cw - 0.002, row_h,
                                    boxstyle="round,pad=0.001", facecolor="#263238",
                                    edgecolor="white", lw=0.5, transform=ax.transAxes,
                                    clip_on=False))
        ax.text(col_x[xi] + cw / 2, hdr_y - row_h / 2, lbl, ha="center", va="center",
                fontsize=9, fontweight="bold", color="white", transform=ax.transAxes)

    for i, (cat, param, val, note) in enumerate(rows):
        y_top = hdr_y - (i + 2) * row_h
        for xi, (txt, cw) in enumerate(zip(["", param, val, note], col_w)):
            ax.add_patch(FancyBboxPatch((col_x[xi], y_top, ), cw - 0.002, row_h - 0.0008,
                                        boxstyle="round,pad=0.0008",
                                        facecolor=CAT_COLORS[cat], edgecolor="#CFD8DC",
                                        lw=0.4, transform=ax.transAxes, clip_on=False))
            ha = "center" if xi == 2 else "left"
            xp = col_x[xi] + (cw / 2 if xi == 2 else 0.007)
            ax.text(xp, y_top + row_h / 2, txt, ha=ha, va="center", fontsize=8.1,
                    color="#212121", transform=ax.transAxes)

    groups, cur, start = [], None, 0
    for i, r in enumerate(rows):
        if r[0] != cur:
            if cur is not None:
                groups.append((cur, start, i - 1))
            cur, start = r[0], i
    groups.append((cur, start, len(rows) - 1))
    for cat, a, b in groups:
        y_top = hdr_y - (a + 2) * row_h
        y_bot = hdr_y - (b + 3) * row_h
        h = (b - a + 1) * row_h
        base = plt.matplotlib.colors.to_rgb(CAT_COLORS[cat])
        ax.add_patch(FancyBboxPatch((col_x[0], y_bot + row_h), col_w[0] - 0.002, h - 0.001,
                                    boxstyle="round,pad=0.001",
                                    facecolor=[max(0, c - 0.08) for c in base],
                                    edgecolor="#90A4AE", lw=0.8,
                                    transform=ax.transAxes, clip_on=False))
        ax.text(col_x[0] + col_w[0] / 2, (y_top + y_bot + row_h) / 2 + row_h / 2, cat,
                ha="center", va="center", fontsize=8.2, fontweight="bold",
                color="#1A237E", transform=ax.transAxes)

    ax.text(0.0, hdr_y - (len(rows) + 2.4) * row_h,
            "Q_ch / Q_dis are simulation OUTPUTS (peak daily energy of the age block / 24 h), "
            "not imposed limits; they are applied as fixed hourly caps in the MILP.   "
            "g-function: pygfunction (Cimmino 2018).   ARX: two-step constrained OLS.",
            ha="left", va="top", fontsize=7.6, color="#546E7A",
            transform=ax.transAxes, style="italic")
    ax.set_xlim(0, 1); ax.set_ylim(-0.03, 1.0)
    p = OUT / f"btes_assumptions_n{N}.png"
    fig.savefig(p, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return p


def main():
    want = sys.argv[1:] or sorted(LIB, key=lambda x: int(x))
    for k in want:
        e = LIB[str(k)]
        N = int(e["n_boreholes"])
        p = draw(N, e, build_rows(N, e))
        print(f"  wrote {p.name}  ({len(build_rows(N, e))} rows)")


if __name__ == "__main__":
    main()
