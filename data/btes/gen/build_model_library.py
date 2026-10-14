"""Build the model-side BTES library from generator output.

  design_<label>_n<N>.json + sim_<label>_n<N>.csv  ->  design_library.json (all N)
                                                    ->  btes_hourly_params_n<N>.json (per N)

Per-block T_init / Qch_max / Qdis_max travel via design_library.json (read by
src/models/btes_params.py); the ARX coefficients, bounds and costs via the
btes_hourly_params files. Usage:  python build_model_library.py --label t106
"""
import argparse, json, re
from pathlib import Path
import pandas as pd

here = Path(__file__).parent
import sys; sys.path.insert(0, str(here))
from make_btes_design import hx_coefficients, wall_step_coefficient, CP_F, M_DIS_PER_BH   # noqa: E402
ap = argparse.ArgumentParser()
ap.add_argument("--label", required=True, help="generator label prefix, e.g. t106")
ap.add_argument("--cost-discharge", type=float, default=2.03)
ap.add_argument("--cost-charge", type=float, default=0.0)
ap.add_argument("--opex-fix", type=float, default=4.0, help="EUR per MWh capacity per year")
ap.add_argument("--flow-levels", default="0.10,0.20,0.30",
                help="extra discharge flows per borehole [kg/s] for the discrete-flow-level rule (base flow always included)")
ap.add_argument("--k-wall", action="store_true",
                help="add the physical wall feedthrough (1-day g-function step) when the design lacks it")
ap.add_argument("--out-suffix", default="", help="suffix for output files, e.g. _s10 -> btes_hourly_params_s10_n<N>.json")
a = ap.parse_args()

files = sorted(here.glob(f"design_{a.label}_n*.json"),
               key=lambda p: int(re.search(r"_n(\d+)\.json$", p.name).group(1)))
library = {}
for f in files:
    d = json.loads(f.read_text())
    N = int(d["n_boreholes"])
    _, _, _, ua_ch, ua_dis = hx_coefficients(N, d["H_m"], d["B_m"], d.get("layout", "disc"))
    d["UA_ch_kW_per_K"], d["UA_dis_kW_per_K"] = round(ua_ch / 1e3, 3), round(ua_dis / 1e3, 3)
    # discrete discharge-flow levels: UA and fluid capacity rate C = m*cp per level
    levels = {}
    for m_bh in [M_DIS_PER_BH] + [float(x) for x in a.flow_levels.split(",") if x.strip()]:
        _, nb, _, _, ua_k = hx_coefficients(N, d["H_m"], d["B_m"], d.get("layout", "disc"), m_dis_per_bh=m_bh)
        levels[f"{m_bh:.4f}"] = {"UA_kW_per_K": round(ua_k / 1e3, 3), "C_kW_per_K": round(m_bh * nb * CP_F / 1e3, 3)}
    d["UA_dis_levels"] = levels
    # conductance at the flow that produced the block caps (base flow, or the ceiling for variable-flow sims)
    _, _, _, _, ua_ref = hx_coefficients(N, d["H_m"], d["B_m"], d.get("layout", "disc"),
                                         m_dis_per_bh=d.get("m_dis_max_kg_s_per_bh", M_DIS_PER_BH))
    d["UA_dis_cap_ref_kW_per_K"] = round(ua_ref / 1e3, 3)
    if a.k_wall and not d.get("k_wall_K_per_kW"):
        d["k_wall_K_per_kW"] = wall_step_coefficient(N, d["H_m"], d["B_m"], d.get("layout", "disc"))
    library[str(N)] = d
    sim = pd.read_csv(here / f"sim_{a.label}_n{N}.csv")
    last = sim.iloc[15 * 365:]
    blocks = {str(k): v for k, v in sorted(d["blocks"].items(), key=lambda kv: int(kv[0]))}
    lastblk = blocks[max(blocks, key=int)]
    years = {"0": "1-5", "1": "6-10", "2": "11-15", "3": "16-20"}
    out = {
        "description": (f"HT-BTES surrogate for dispatch model. Cylindrical (hex-packed disc) "
                        f"field, H={d['H_m']:.0f}m, B={d['B_m']}m ({N} bh, {d['total_length_m']:,.0f}m "
                        f"total). Charges at {d['Tmax']:.0f}C (wellhead brine, Kaufmann et al. ECM 2026), "
                        f"discharges directly to DH network when T_out >= 80C. ARX fitted to "
                        f"pygfunction 20-yr simulation; per-age-block coefficients."),
        "model_form": "T_state(h+1) = a_h*T_state(h) + b_ch_h*Qch(h) + b_dis_h*Qdis(h) + c_h  [hourly, kWh units]",
        "resolution": "hourly",
        "fit": {"a_h": d["a_h"], "b_ch_h_per_kWh": d["b_ch_h"], "b_dis_h_per_kWh": d["b_dis_h"],
                "c_h_degC": d["c_h"],
                "source": ("a_d=1, c_d=0 (pure accumulator, no idle dynamics); b_dis pinned to "
                           "-b_ch/eta_sim per age block so the surrogate round trip equals the "
                           "simulated energy ratio by construction"),
                "daily_to_hourly_conversion": "a_h = a_d^(1/24), b unchanged (kWh units same), c_h = c_d/24"},
        "bounds": {"Qdis_max_kWh_per_hour": round(lastblk["Qdis_max_kW"], 1),
                   "Qch_max_kWh_per_hour": round(lastblk["Qch_max_kW"], 1),
                   "Tmin_degC": d["Tmin"], "Tmax_degC": d["Tmax"],
                   "T_init_degC": round(float(last.T_b.mean()), 1),
                   "UA_dis_kW_per_K": d["UA_dis_kW_per_K"], "UA_ch_kW_per_K": d["UA_ch_kW_per_K"],
                   "UA_dis_levels": d["UA_dis_levels"],
                   "UA_dis_cap_ref_kW_per_K": d["UA_dis_cap_ref_kW_per_K"],
                   "k_wall_K_per_kW": d.get("k_wall_K_per_kW", 0.0)},
        "invest": {"enabled": False, "IK_EUR": int(round(d["capex_EUR"])), "lt_yr": 50, "ir": 0.05,
                   "warmup_cost_annual_EUR": 0,
                   "notes": f"IK_EUR: {d['capex_basis']} = {d['capex_EUR']:,.0f} EUR (drilling only). "
                            f"Ancillaries separate. Warm-up cost set to 0."},
        "cost_charge": a.cost_charge, "cost_discharge": a.cost_discharge,
        "age_blocks": {k: {"years": years[k], "b_ch_h_per_kWh": v["b_ch_h"],
                           "b_dis_h_per_kWh": v["b_dis_h"], "rte_pct": round(v["sim_RTE_pct"], 1),
                           "T_init": round(v["T_init"], 2),
                           "Qch_max_kW": round(v["Qch_max_kW"], 1),
                           "Qdis_max_kW": round(v["Qdis_max_kW"], 1)}
                       for k, v in blocks.items()},
        "n_boreholes": N,
        "opex_fix_EUR_per_MWh_cap": a.opex_fix,
    }
    (here.parent / f"btes_hourly_params{a.out_suffix}_n{N}.json").write_text(json.dumps(out, indent=2))
    print(f"  n{N:<4} Tmax {d['Tmax']:.0f}C  band {d['Tmax']-d['Tmin']:.1f}K  "
          f"Qch {lastblk['Qch_max_kW']/1e3:5.2f} MW  Qdis {lastblk['Qdis_max_kW']/1e3:4.2f} MW  "
          f"RTE " + "->".join(f"{v['sim_RTE_pct']:.0f}" for v in blocks.values()) + "%")
(here.parent / f"design_library{a.out_suffix}.json").write_text(json.dumps(library, indent=2))
print(f"wrote design_library{a.out_suffix}.json ({len(library)} designs) and {len(files)} btes_hourly_params{a.out_suffix}_n*.json")
