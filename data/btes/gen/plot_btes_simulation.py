#!/usr/bin/env python3
"""Four-panel 20-year simulation figure per HT-BTES design.

Reads sim_<label>_n<N>.csv + design_<label>_n<N>.json written by
make_btes_design.py and reproduces: charge/discharge power, borehole wall
temperature, fluid outlet temperature, annual energy + round-trip efficiency.
Outlet temperatures are recomputed from the same flow constants the
simulation used (M_CH_PER_BH, M_DIS_PER_BH, RHO_F_CF), so nothing is fitted.

    python plot_btes_simulation.py --label t106            # all designs
    python plot_btes_simulation.py --label t106 150 200    # selected
Output: simulation/btes_sim_<label>_n<N>.png
"""
import argparse, json, re, sys
from pathlib import Path
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from make_btes_design import (M_CH_PER_BH, M_DIS_PER_BH, RHO_F, RHO_F_CF, T_DH_RET,
                              T_USE_MIN, T_G, CH_START, CH_END, DIS_S1, DIS_E2)

ap = argparse.ArgumentParser()
ap.add_argument("--label", required=True)
ap.add_argument("designs", nargs="*", type=int)
a = ap.parse_args()
OUT = HERE / "simulation"; OUT.mkdir(exist_ok=True)

files = sorted(HERE.glob(f"design_{a.label}_n*.json"),
               key=lambda p: int(re.search(r"_n(\d+)\.json$", p.name).group(1)))
for f in files:
    d = json.loads(f.read_text()); N = int(d["n_boreholes"])
    if a.designs and N not in a.designs: continue
    s = pd.read_csv(HERE / f"sim_{a.label}_n{N}.csv")
    yrs = s.day.values / 365.0
    P_kW = s.Q_kWh_d.values / 24.0                     # daily mean power
    ch = np.clip(P_kW, 0, None); dis = np.clip(-P_kW, 0, None)
    v_ch, v_dis = M_CH_PER_BH * N / RHO_F, M_DIS_PER_BH * N / RHO_F
    T_in = d["Tmax"]
    doy = (s.day.values % 365) + 1
    in_ch = (doy >= CH_START) & (doy <= CH_END) & (ch > 0)
    in_dis = ((doy >= DIS_S1) | (doy <= DIS_E2)) & (dis > 0)
    T_out_ch = T_in - ch * 1e3 / (v_ch * RHO_F_CF)      # warm return during charge
    T_out_dis = T_DH_RET + dis * 1e3 / (v_dis * RHO_F_CF)  # DH supply during discharge
    yr = (s.day.values // 365) + 1
    ann = pd.DataFrame({"yr": yr, "ch": ch * 24 / 1e3, "dis": dis * 24 / 1e3}).groupby("yr").sum()
    rte = 100 * ann.dis / ann.ch
    blocks = d["blocks"]; qch_max = max(v["Qch_max_kW"] for v in blocks.values())
    qch_mature = blocks[max(blocks, key=int)]["Qch_max_kW"]

    fig, ax = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f"HT-BTES 20-Year Simulation — g-function Method (pygfunction)\n"
                 f"{N} boreholes (hex-packed disc), H={d['H_m']:.0f} m, B={d['B_m']} m, "
                 f"T_in_charge={T_in:.0f} degC, direct DH supply (T_out >= {T_USE_MIN:.0f} degC)",
                 fontsize=15)
    a1, a2, a3, a4 = ax[0, 0], ax[0, 1], ax[1, 0], ax[1, 1]
    a1.fill_between(yrs, 0, ch, color="#e45756", alpha=0.75, label="Charging power (kW)")
    a1.fill_between(yrs, 0, dis, color="#4c9be8", alpha=0.75, label="Discharge power (kW)")
    a1.axhline(qch_mature, color="#c1121f", ls=":", lw=1.2)
    a1.text(0.3, qch_mature * 1.03, f"Qch_max mature ({qch_mature:,.0f} kW)", color="#c1121f", fontsize=10)
    a1.set_title("Charging & Discharging Power"); a1.set_ylabel("Heat rate [kW]"); a1.legend(loc="upper right")
    a2.plot(yrs, s.T_b, color="#2ca02c", lw=0.9, label="T_borehole")
    a2.axhline(T_in, color="#c1121f", ls="--", lw=1, label=f"T_b max ({T_in:.0f} degC)")
    a2.axhline(T_G, color="#6c6cd8", ls="--", lw=1, label=f"T_ground ({T_G} degC)")
    a2.axhline(T_USE_MIN + 0.8, color="#333", ls=":", lw=1, label="T_min dispatch (80.8 degC)")
    a2.set_title("Borehole Wall Temperature"); a2.set_ylabel("Temperature [degC]"); a2.legend(loc="lower right")
    a3.scatter(yrs[in_ch], T_out_ch[in_ch], s=2, color="#f28e2b", label="T_out charging (warm return)")
    a3.scatter(yrs[in_dis], T_out_dis[in_dis], s=2, color="#4c9be8", label="T_out discharge (DH supply)")
    a3.axhline(T_USE_MIN, color="#7b1fa2", ls=":", lw=1.2, label=f"T_min discharge ({T_USE_MIN:.0f} degC)")
    a3.set_title("Fluid Outlet Temperature T_out"); a3.set_ylabel("Temperature [degC]"); a3.legend(loc="lower right")
    w = 0.38
    a4.bar(ann.index - w / 2, ann.ch, w, color="#e45756", alpha=0.8, label="Annual charge (MWh)")
    a4.bar(ann.index + w / 2, ann.dis, w, color="#4c9be8", alpha=0.8, label="Annual discharge (MWh)")
    a4.set_ylabel("Annual energy [MWh]"); a4.set_title("Annual Energy & Round-Trip Efficiency")
    a4b = a4.twinx(); a4b.plot(rte.index, rte, "-o", color="k", ms=4, label="Round-trip efficiency")
    a4b.set_ylabel("Round-trip efficiency [%]"); a4b.set_ylim(0, max(60, rte.max() * 1.1))
    h1, l1 = a4.get_legend_handles_labels(); h2, l2 = a4b.get_legend_handles_labels()
    a4.legend(h1 + h2, l1 + l2, loc="center right")
    for x in (a1, a2, a3): x.set_xlim(0, 20); x.set_xlabel("Year"); x.grid(alpha=0.3)
    a4.set_xlabel("Year"); a4.grid(alpha=0.3, axis="y")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    p = OUT / f"btes_sim_{a.label}_n{N}.png"; fig.savefig(p, dpi=150, facecolor="white"); plt.close(fig)
    print(f"  n{N:<4} peak T_b {s.T_b.max():5.1f}  yr20 charge {ann.ch.iloc[-1]:7.0f} MWh  "
          f"discharge {ann.dis.iloc[-1]:6.0f} MWh  RTE {rte.iloc[-1]:4.1f}%  -> {p.name}")
