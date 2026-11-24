#!/usr/bin/env python3
"""Munich case at representative-hour (RH) resolution.

Third resolution alongside run_rd.py (representative days, Kotzur superposition)
and run_full.py (exact 8760 h). Scenario gating, input workbooks and env flags
mirror run_rd exactly so the three are directly comparable.

Scope: the RH model core carries the generic DH plant plus the BTES ARX chain.
The SAPHEA three-mode geothermal/HTHP coupling and the reversible HP/ORC are NOT
yet written in the RH index space, so only the scenarios that do not need them
can be run here:

    N1  base (geothermal + gas)          ✓
    N4  + HT-BTES                        ✓
    N5  + PTES                           ✓
    N2 / N3 / N6                         ✗ need revORC / HTHP

    SCENARIO=N1 python run_rh.py
    RH_SEGMENTED=1 N_REP_HOURS=300 SCENARIO=N5 python run_rh.py

Env: SCENARIO, TECH_DATA, DH_DATA, N_REP_HOURS, RH_SEGMENTED, RH_SEASON_PURE,
     TSA_PRICE_WEIGHT, BTES_DESIGN_N, GRB_THREADS, MIPGAP, TIMELIMIT_S,
     RESULTS_ROOT.
"""

import os
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import pyomo.environ as pyo
from loguru import logger

ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from data_loader import load_all_munich_data, build_uc_params      # noqa: E402
from models.rh_model import DistrictHeatingModelRH                 # noqa: E402
from utils.solver import get_solver_name                           # noqa: E402
from run_rd import (derive_saphea_capacities, _inject_revorc_tech,
                    GEO_TECH, HTHP_TECH, REVORC_TECH)               # noqa: E402
from orc_surrogate_maps import load_revorc_config, build_revorc_maps  # noqa: E402
from models.saphea_revorc_rh import (build_saphea_rh, build_revorc_rh,
                                     rep_hour_profile)              # noqa: E402

GRB_THREADS = int(os.environ.get("GRB_THREADS", 40))
MIPGAP = float(os.environ.get("MIPGAP", 0.0025))
TIMELIMIT_S = int(os.environ.get("TIMELIMIT_S", 48 * 3600))
N_REP_HOURS = int(os.environ.get("N_REP_HOURS", 300))
SEGMENTED = os.environ.get("RH_SEGMENTED", "0") == "1"

SUPPORTED = {"N1", "N2", "N3", "N4", "N5", "N6"}


def main():
    t_start = time.perf_counter()
    scenario = os.environ.get("SCENARIO", "N1").upper()
    if scenario not in SUPPORTED:
        raise SystemExit(f"Unknown SCENARIO={scenario} (use {sorted(SUPPORTED)})")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = Path(os.environ.get("RESULTS_ROOT", ROOT))
    # the clustering mode goes in the folder name: a free and a segmented run of
    # the same scenario are different experiments, not two copies of one
    tag = "seg" if SEGMENTED else "free"
    results_dir = root / "results" / "rh" / f"MILP_{ts}_{scenario}_{tag}{N_REP_HOURS}"
    results_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Munich RH model [{scenario}] — {N_REP_HOURS} rep hours, "
                f"{'segmented' if SEGMENTED else 'free'} — output: {results_dir}")

    tech_file = ROOT / "data" / os.environ.get("TECH_DATA", "Munich_tech_data.xlsx")
    demand_file = ROOT / "data" / os.environ.get("DH_DATA", "Munich_DH_data.xlsx")
    logger.warning(f"Inputs: tech={tech_file.name}, demand={demand_file.name}")
    techs, stor, prices, demand, extra, hp = load_all_munich_data(
        tech_file=tech_file, demand_file=demand_file)

    # ── scenario gating, identical to run_rd's N-suite ──
    keep_buffer = os.environ.get("KEEP_BUFFER") == "1"
    stor = {k: v for k, v in stor.items()
            if ("buffer" not in k.lower() or keep_buffer)
            and ("buffer" in k.lower() or scenario in ("N5", "N6"))}
    btes_build_period = 2025 if scenario in ("N4", "N6") else None
    if scenario not in ("N2", "N6"):
        os.environ["REVORC"] = "0"
    exclude_hthp = scenario in ("N1", "N2", "N4", "N5")
    logger.info(f"Scenario {scenario}: BTES={btes_build_period}, "
                f"storages={list(stor) or 'none'}, "
                f"revORC={'off' if os.environ.get('REVORC') == '0' else 'on'}, "
                f"HTHP={'excluded' if exclude_hthp else 'investable'}")

    # ── resource-driven geothermal capacity (same derivation as run_rd) ──
    resource = extra["geo_resource"]
    design = derive_saphea_capacities(resource)
    techs[GEO_TECH]["max_output_MW"] = design["Q_geo_du"]
    techs[GEO_TECH]["installed_capacity_MW"] = design["Q_geo_du"]
    techs[HTHP_TECH]["max_output_MW"] = design["Q_hthp_max"]
    techs[HTHP_TECH]["installed_capacity_MW"] = 0.0

    revorc_cfg = None
    if os.environ.get("REVORC", "1") != "0":
        revorc_cfg = load_revorc_config(tech_file)
        if revorc_cfg:
            _inject_revorc_tech(techs, revorc_cfg)

    investment_years = extra["investment_years"]
    uc = build_uc_params(techs)

    btes_json = ROOT / "data" / "btes" / "btes_hourly_params.json"
    dn = os.environ.get("BTES_DESIGN_N")
    if btes_build_period and dn:
        cand = ROOT / "data" / "btes" / f"btes_hourly_params_n{dn}.json"
        if not cand.exists():
            raise SystemExit(f"BTES_DESIGN_N={dn}: {cand.name} not found")
        btes_json = cand
        logger.info(f"BTES design {dn}: {cand.name}")

    model = DistrictHeatingModelRH(
        demand_data=demand, technologies=techs, storage=stor, prices=prices,
        investment_years=investment_years, n_rep_hours=N_REP_HOURS,
        extra_params=extra, hp_params=hp or None, uc_params=uc,
        btes_build_period=btes_build_period,
        btes_arx_json=btes_json if btes_build_period else None,
        name=f"MunichDH_RH_{scenario}", segmented=SEGMENTED,
        clustering_mode=os.environ.get("TSA_MODE", "per_period"),
        allow_reinjection=True,          # must-run well, surplus goes back down
    )
    t_build = time.perf_counter()
    model.build()
    logger.info(f"Model built in {time.perf_counter() - t_build:.1f} s")

    m = model.model

    # ── SAPHEA three-mode coupling and the reversible machine (RH index) ──
    build_saphea_rh(model, demand, resource, GEO_TECH, HTHP_TECH, design)

    if revorc_cfg:
        I = len(m.REP_HOURS)
        amb = demand["ambient_temp"].to_numpy()[:8760]
        sup = demand["supply_temp"].to_numpy()[:8760]
        ret = demand["return_temp"].to_numpy()[:8760]
        maps_by_p = {}
        for p_ in investment_years:
            sig = model.rh_by_period[p_]["sigma"]
            # (I, 1): each representative hour is its own single-slot "day"
            maps_by_p[p_] = build_revorc_maps(
                rep_hour_profile(amb, sig, I).reshape(I, 1),
                rep_hour_profile(sup, sig, I).reshape(I, 1),
                rep_hour_profile(ret, sig, I).reshape(I, 1),
                resource, revorc_cfg)
        build_revorc_rh(model, maps_by_p, revorc_cfg, GEO_TECH, HTHP_TECH,
                        REVORC_TECH, segmented=SEGMENTED)

    if exclude_hthp:
        for p_ in investment_years:
            m.cap_add[HTHP_TECH, p_].fix(0)
        logger.info(f"HTHP investment fixed to 0 (scenario {scenario})")

    # geothermal must-run, as in run_rd / run_full
    if hasattr(m, "u") and GEO_TECH in set(m.UC_TECH):
        n = 0
        for p in m.PERIODS:
            for i in m.REP_HOURS:
                m.u[GEO_TECH, p, i].fix(1)
                n += 1
        logger.info(f"Geothermal forced always-on (must-run): {n} slots")

    # reinjection cannot exceed what the well produces
    def _reinj_cap(mm, p, i):
        return mm.reinject[p, i] <= mm.gen[GEO_TECH, p, i]
    m.reinject_le_geo = pyo.Constraint(m.PERIODS, m.REP_HOURS, rule=_reinj_cap)
    logger.info("Reinjection bounded by geothermal generation")

    results_raw = model.solve(solver_name=get_solver_name(), tee=True, MIPGap=MIPGAP,
                              TimeLimit=TIMELIMIT_S, Threads=GRB_THREADS)
    res = model.get_results(export_dir=str(results_dir))
    npv_early = float(pyo.value(model.model.total_cost))
    logger.info(f"Objective (NPV): EUR {npv_early:,.0f}")

    # ── RD-style reporting through the RHAsRD adapter: every plot, CSV and the
    #    Excel workbook that run_rd and run_full produce, on the RH solution ──
    import run_rd as rd
    from rh_reporting import RHAsRD
    # run_rd's reporters read these module-level globals (they are set inside
    # run_rd.main(), which we never call) — same handshake run_full uses.
    rd.INVESTMENT_YEARS = list(investment_years)
    rd.N_TYPICAL_DAYS = 365
    rd.SCENARIO = scenario
    adapter = RHAsRD(model, demand)
    logger.info("Saving RD-style plots and exports (via adapter)…")
    if revorc_cfg:
        try:
            rd._report_revorc(adapter, results_dir)
            rd.save_revorc_price_analysis(adapter, results_dir)
        except Exception as e:                      # reporting must not kill a run
            logger.warning(f"revORC report skipped: {type(e).__name__}: {e}")
    for fn in (rd.save_dispatch_plots, rd.save_winter_dispatch_and_csv,
               rd.save_rep_dispatch, rd.save_combined_dispatch_plot,
               rd.save_soc_plots, rd.save_btes_plots, rd.save_mode_plots):
        try:
            fn(adapter, results_dir)
        except Exception as e:
            logger.warning(f"{fn.__name__} skipped: {type(e).__name__}: {e}")
    for fn in (rd.save_capacity_plot, rd.save_generation_plot):
        try:
            fn(res, results_dir)
        except Exception as e:
            logger.warning(f"{fn.__name__} skipped: {type(e).__name__}: {e}")
    comp_params = None
    try:
        comp_params = rd.save_computational_params(
            adapter, results_raw, time.perf_counter() - t_start, results_dir)
    except Exception as e:
        logger.warning(f"computational params skipped: {type(e).__name__}: {e}")
    try:
        rd.export_excel(adapter, res, results_dir, comp_params)
    except Exception as e:
        logger.warning(f"Excel export skipped: {type(e).__name__}: {e}")

    npv = float(pyo.value(m.total_cost))
    wall = time.perf_counter() - t_start
    logger.info(f"Objective (NPV): EUR {npv:,.0f}")
    pd.DataFrame([{
        "scenario": scenario, "resolution": "RH",
        "n_rep_hours": N_REP_HOURS, "segmented": SEGMENTED,
        "n_storage_levels": model.rh_by_period[investment_years[0]]["n_storage_levels"],
        "objective_NPV_EUR": npv, "wall_time_s": round(wall, 1),
    }]).to_csv(results_dir / "summary.csv", index=False)
    logger.info(f"Done in {wall:.1f} s → {results_dir}")


if __name__ == "__main__":
    main()
