"""
Southern Munich (GHP) case — RD (representative-days) model run script.

System (Kaufmann et al., ECM 2026):
  Geothermal brine at 106°C (17.4 MW max) → direct DHN supply
  Gas boiler for backup

Usage:
    python run_rd.py
"""

import sys
import os
import time
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams.update({
    'font.size': 20, 'axes.labelsize': 21, 'axes.titlesize': 21,
    'xtick.labelsize': 18, 'ytick.labelsize': 18,
    'legend.fontsize': 18, 'figure.titlesize': 21,
    'axes.spines.top': False, 'axes.spines.right': False,
})

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

# Multi-year TSA mode (rd_model clustering_mode): 'per_period' (default) clusters
# each investment milestone on its OWN demand and price series; 'joint' pools the
# milestones into one shared clustering. With the current data the two coincide
# exactly -- demand is a scalar multiple across milestones (x1.02..x1.08) and the
# price series an affine transform, and min-max scaling makes the clustering
# invariant to both -- but per-period is the robust default for when a milestone
# carries its own demand or price SHAPE. Override with TSA_MODE=joint.
TSA_MODE = os.environ.get('TSA_MODE', 'per_period')

try:
    os.sched_setaffinity(0, set(range(1, 41)))
    logger.info("CPU affinity set to cores 1–40")
except (AttributeError, OSError) as e:
    logger.warning(f"Could not set CPU affinity: {e}")

# ── Paths ─────────────────────────────────────────────────────────────────────
_CASE_DIR     = Path(__file__).parent.resolve()          # paper2_munich/
sys.path.insert(0, str(_CASE_DIR / "src"))           # generic models/utils
sys.path.insert(0, str(_CASE_DIR))                       # data_loader, colors

from data_loader import load_all_munich_data, build_uc_params
from models.rd_model import DistrictHeatingModel
from colors import TECH_COLORS, get_color
from orc_surrogate_maps import load_revorc_config, build_revorc_maps

# ── Settings ──────────────────────────────────────────────────────────────────
# INVESTMENT_YEARS and N_TYPICAL_DAYS are loaded from the 'extra' sheet of
# Munich_tech_data.xlsx and populated in main() (the year columns of that sheet are
# the investment milestones; the 'n_typical_days' row sets the clustering resolution).
INVESTMENT_YEARS: list = []
N_TYPICAL_DAYS:   int  = 0
FORCE_LP          = False
BTES_BUILD_PERIOD = None # toggled
GEO_MUST_RUN_ALWAYS = True # Doublet cannot be cycled off: force commitment on every hour (gen >= min-load all year)

# ── SAPHEA capacity & COP derivation (Munich) ─────────────────────────────────
# Source: Saphea_LSGEOHP-tool.html (calculateDirectUse / calculateCapacityIncrease /
# calculateTemperatureIncrease + calculateCopVeryHigh). All capacities and the COP
# are DERIVED from the geothermal brine resource (the 'Geothermal Resource' sheet of
# Munich_tech_data.xlsx) rather than hard-coded — see SAPHEA_modes_and_integration.md.
#
# COP_SAPHEA  = carnot_eta × (T_supply + 273.15) / (T_supply − T_inject_min)
# W_elec      = Q_evap / COP_SAPHEA
# Q_HP        = Q_evap + W_elec   →   COP_standard = COP_SAPHEA + 1
# coupling_ratio[d,h] = (dT_evap / dT_direct) × (1 + 1/COP_SAPHEA[d,h])
GEO_TECH  = "Geothermal Heat"
HTHP_TECH = "HTHP"

# ── revORC (FLEXGEO reversible HP/ORC, 'RevORC' sheet of Munich_tech_data.xlsx) ──
# A SEPARATE modular technology, additive to the untouched SAPHEA geo+HTHP system.
# Each module takes the high-grade brine slice (T_prod → T_brine_out_min ≈ 65 °C)
# and either generates electricity (ORC mode, heat rejected to air) or delivers
# heat to the DHN (HP mode). Hourly per-module performance comes from the FLEXGEO
# surrogates, evaluated offline in orc_surrogate_maps.py. Costs: Kaufmann et al.
# ECM 2026 (Tables 3.1/3.3, 50 % ACC optimum) + VDI 2067 O&M.
REVORC_TECH = "RevORC HP"


def derive_saphea_capacities(resource):
    """
    Derive the SAPHEA design capacities & COP from the brine resource inputs.

    ``resource`` is the dict from the 'Geothermal Resource' sheet
    (brine_mass_flow, brine_cp, T_prod, T_inject_min, T_return_design,
    direct_hx_pinch, carnot_eta, T_supply_design).

    Returns a dict of design-point values (MW unless noted)::

        Q_geo_du    Direct-Use geo ceiling     = ṁ·cp·(T_prod − T_return) / 1000
        Q_geo_ci    Capacity-Increase direct   = ṁ·cp·(T_prod − T_woHP)   / 1000
        Q_evap      HP evaporator (CI source)  = ṁ·cp·(T_woHP − T_inj_min)/ 1000
        Q_evap_ti   HP evaporator (TI, full)   = ṁ·cp·(T_prod − T_inj_min)/ 1000
        Q_hthp_max  HP design output rating    = Q_evap · (1 + 1/COP_SAPHEA)
        cop_std     design COP_standard        = COP_SAPHEA + 1
        T_woHP      CI direct-HX brine exit    = T_return + pinch
    """
    mdot  = resource["brine_mass_flow"]
    cp    = resource["brine_cp"]
    T_prod = resource["T_prod"]
    T_min  = resource["T_inject_min"]
    T_ret  = resource["T_return_design"]
    pinch  = resource["direct_hx_pinch"]
    eta    = resource["carnot_eta"]
    T_sup  = resource["T_supply_design"]

    def q(dT):
        return mdot * cp * max(dT, 0.0) / 1000.0

    T_woHP     = T_ret + pinch
    cop_saphea = eta * (T_sup + 273.15) / (T_sup - T_min)
    Q_evap     = q(T_woHP - T_min)
    return {
        "Q_geo_du":   q(T_prod - T_ret),
        "Q_geo_ci":   q(T_prod - T_woHP),
        "Q_evap":     Q_evap,
        "Q_evap_ti":  q(T_prod - T_min),
        "Q_hthp_max": Q_evap * (1.0 + 1.0 / cop_saphea),
        "cop_std":    cop_saphea + 1.0,
        "T_woHP":     T_woHP,
    }


def _interp_capex_curve(curve, size):
    """Linear interpolation of specific cost (€/MW) at HP `size` (MW), clamped to the
    curve endpoints outside its range. ``curve`` = sorted [(size_MW, eur_per_MW), …]."""
    sizes = [p[0] for p in curve]
    costs = [p[1] for p in curve]
    if size <= sizes[0]:
        return costs[0]
    if size >= sizes[-1]:
        return costs[-1]
    for i in range(1, len(sizes)):
        if sizes[i] >= size:
            x0, x1, y0, y1 = sizes[i - 1], sizes[i], costs[i - 1], costs[i]
            return y0 + (y1 - y0) * (size - x0) / (x1 - x0)
    return costs[-1]


def derive_hthp_capex(curve, scale, q_evap, q_hthp_max):
    """HTHP CAPEX (€/MW of HP output) derived from the SAPHEA cost curve.

    SAPHEA picks the specific cost by interpolating the curve at the HP output rating
    (``q_hthp_max``), applies the ``scale`` factor (≈0.8), and charges the investment
    on the evaporator/source MW (``q_evap``). The model charges CAPEX on the installed
    HTHP capacity (condenser output), so we express the same total per condenser-MW:

        capex_per_MW = q_evap · interp(curve, q_hthp_max) · scale / q_hthp_max
    """
    specific = _interp_capex_curve(curve, q_hthp_max) * scale
    return q_evap * specific / q_hthp_max


# ── Helper: month labels for hour axis ────────────────────────────────────────
def _soc_hour_label(h):
    MONTHS = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    d = h // 24
    for mi, md in enumerate([31,28,31,30,31,30,31,31,30,31,30,31]):
        if d < md:
            return f"{MONTHS[mi]} {d+1}"
        d -= md
    return "Dec 31"


def _demand_for(model, year):
    """The 8760-h demand series of one milestone.

    The exports used to take model.demand_data['heat_demand'] once, outside the
    year loop, so every dispatch plot and CSV carried the BASE-year demand while
    the supply grew milestone by milestone — the curves stopped matching. Each
    milestone has its own 'corrected_demand_<year>' column; use it.
    """
    col = f'corrected_demand_{year}'
    if col in model.demand_data.columns:
        return model.demand_data[col].values
    return model.demand_data['heat_demand'].values


# ── Reconstruct full 8760h arrays from representative days ────────────────────
def _reconstruct_8760(model, year):
    m = model.model; tsa = model.tsa_by_period[year]
    techs = [t for t in model.technologies if pyo.value(m.cap_total[t, year]) > 1e-2]
    gen_data = {t: [] for t in techs}
    stor_discharge, stor_charge = {}, {}
    if hasattr(m, 'charge'):
        for s in m.STOR:
            stor_discharge[f'{s} (discharge)'] = []
            stor_charge[f'{s} (charge)'] = []
    if model.btes_enabled and hasattr(m, 'btes_Qdis'):
        stor_discharge['BTES (discharge)'] = []
        stor_charge['BTES (charge)'] = []
    for h in range(8760):
        d_orig, hr = h // 24, h % 24
        d_rep = tsa['day_to_rep'][d_orig]
        for t in techs:
            gen_data[t].append(pyo.value(m.gen[t, year, d_rep, hr]))
        if hasattr(m, 'charge'):
            for s in m.STOR:
                stor_discharge[f'{s} (discharge)'].append(pyo.value(m.discharge[s, year, d_rep, hr]))
                stor_charge[f'{s} (charge)'].append(-pyo.value(m.charge[s, year, d_rep, hr]))
        if model.btes_enabled and hasattr(m, 'btes_Qdis'):
            stor_discharge['BTES (discharge)'].append(pyo.value(m.btes_Qdis[year, d_rep, hr]) / 1000.0)
            stor_charge['BTES (charge)'].append(-pyo.value(m.btes_Qch[year, d_rep, hr]) / 1000.0)
    return pd.DataFrame(gen_data), stor_discharge, stor_charge


def save_rep_dispatch(model, results_dir):
    """Export the CONSISTENT representative-day dispatch (no 8760 reconstruction).

    At the rep-day level the heat balance is exact — supply == rep_demand + reinject —
    so peak-coverage / winter-dispatch figures built from this are artefact-free, unlike
    anything mapped back onto real 8760-h demand. One row per (rep_day, hour) per year;
    `weight_days` = how many real days the rep-day represents (for duration weighting)."""
    m = model.model
    n_rep = model.tsa['n_rep_days']
    for year in INVESTMENT_YEARS:
        rep_dem = model.tsa_by_period[year]['rep_data']['heat_demand'].values
        techs = [t for t in model.technologies if pyo.value(m.cap_total[t, year]) > 1e-2]
        rows = []
        for d in range(n_rep):
            w = float(pyo.value(m.day_weight[year, d])) if hasattr(m, 'day_weight') else 1.0
            for h in range(24):
                r = {'rep_day': d, 'hour': h,
                     'rep_demand_MW': float(rep_dem[d * 24 + h]), 'weight_days': w}
                for t in techs:
                    r[t] = pyo.value(m.gen[t, year, d, h])
                if hasattr(m, 'reinject'):
                    r['Reinjection_MW'] = pyo.value(m.reinject[year, d, h])
                if hasattr(m, 'charge'):
                    for s in m.STOR:
                        r[f'{s} (discharge)'] = pyo.value(m.discharge[s, year, d, h])
                        r[f'{s} (charge)'] = -pyo.value(m.charge[s, year, d, h])
                if model.btes_enabled and hasattr(m, 'btes_Qdis'):
                    r['BTES (discharge)'] = pyo.value(m.btes_Qdis[year, d, h]) / 1000.0
                    r['BTES (charge)'] = -pyo.value(m.btes_Qch[year, d, h]) / 1000.0
                if hasattr(m, 'orc_share') and hasattr(model, '_revorc'):
                    p_net = model._revorc['maps_by_p'][year]['orc_p_net_MW']
                    r['ORC_electricity_MW'] = pyo.value(m.orc_share[year, d, h]) * p_net[(d, h)]
                rows.append(r)
        pd.DataFrame(rows).to_csv(results_dir / f'rep_dispatch_{year}.csv', index=False)
    logger.info("  Rep-level (consistent) dispatch CSVs saved")


def _reconstruct_soc_8760(model, stor_name, year):
    m = model.model; tsa = model.tsa_by_period[year]
    n_days    = tsa['n_days']
    soc_inter = [pyo.value(m.soc_inter[stor_name, year, d]) for d in range(n_days)]
    soc_intra = {(d, h): pyo.value(m.soc_intra[stor_name, year, d, h])
                 for d in range(tsa['n_rep_days']) for h in range(24)}
    result = []
    for t in range(8760):
        day, hr  = t // 24, t % 24
        prev_day = (day - 1) % n_days
        d_rep    = tsa['day_to_rep'][day]
        result.append(soc_inter[prev_day] + soc_intra[d_rep, hr])
    return result


# ── Dispatch plots ─────────────────────────────────────────────────────────────
def save_dispatch_plots(model, results_dir):
    _MONTH_TICKS  = [0, 744, 1416, 2160, 2880, 3624, 4344, 5088, 5832, 6552, 7296, 8016]
    _MONTH_LABELS = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    _ZOOMS = [('Winter Week (Jan 15–22)', 336, 504), ('Summer Week (Jul 1–8)', 4344, 4512)]
    hours  = np.arange(8760)

    for year in INVESTMENT_YEARS:
        demand = _demand_for(model, year)
        df, stor_discharge, stor_charge = _reconstruct_8760(model, year)
        if df.empty:
            continue
        stor_discharge = {k: v for k, v in stor_discharge.items() if any(x > 1e-4 for x in v)}
        stor_charge    = {k: v for k, v in stor_charge.items()    if any(x < -1e-4 for x in v)}
        pos_cols = list(df.columns) + list(stor_discharge.keys())
        pos_data = np.vstack([df.values.T, np.array(list(stor_discharge.values()))]
                             if stor_discharge else [df.values.T])
        colors     = [get_color(c) for c in pos_cols]
        chg_cols   = list(stor_charge.keys())
        chg_data   = np.array(list(stor_charge.values())) if chg_cols else None
        chg_colors = [get_color(c) for c in chg_cols]

        fig, axes = plt.subplots(3, 1, figsize=(18, 14))

        ax = axes[0]
        ax.stackplot(hours, pos_data, labels=pos_cols, colors=colors, alpha=0.85)
        if chg_data is not None:
            ax.stackplot(hours, chg_data, labels=chg_cols, colors=chg_colors, alpha=0.75)
        ax.plot(hours, demand, 'k--', lw=1.4, label='Heat Demand', zorder=10)
        ax.axhline(0, color='black', lw=0.6, zorder=5)
        ax.set_ylabel('Thermal Power [MW]')
        ax.set_xlim(0, 8760)
        ax.set_xticks(_MONTH_TICKS); ax.set_xticklabels(_MONTH_LABELS)
        ax.grid(alpha=0.2)
        ax.legend(loc='upper right', ncol=min(len(pos_cols)+1, 5), fontsize=15, framealpha=0.9)

        for ax, (title, h0, h1) in zip(axes[1:], _ZOOMS):
            sl = slice(h0, h1); hrs = hours[sl]
            ax.stackplot(hrs, pos_data[:, sl], colors=colors, alpha=0.85)
            if chg_data is not None:
                ax.stackplot(hrs, chg_data[:, sl], colors=chg_colors, alpha=0.75)
            ax.plot(hrs, demand[sl], 'k--', lw=1.4, zorder=10)
            ax.axhline(0, color='black', lw=0.6, zorder=5)
            ax.set_ylabel('Thermal Power [MW]')
            ax.set_xlim(hrs[0], hrs[-1])
            ax.set_xticks(hrs[::24])
            ax.set_xticklabels([_soc_hour_label(h) for h in hrs[::24]],
                               rotation=30, ha='right', fontsize=13)
            ax.grid(alpha=0.2)

        plt.tight_layout()
        plt.savefig(results_dir / f'dispatch_{year}.png', dpi=150, bbox_inches='tight')
        plt.close()

        # Clean single full-year panel for slides — no title, large fonts
        figy, axy = plt.subplots(figsize=(16, 5.2))
        axy.stackplot(hours, pos_data, labels=pos_cols, colors=colors, alpha=0.9)
        if chg_data is not None:
            axy.stackplot(hours, chg_data, labels=chg_cols, colors=chg_colors, alpha=0.8)
        axy.plot(hours, demand, 'k--', lw=1.8, label='Heat Demand', zorder=10)
        axy.axhline(0, color='black', lw=0.8, zorder=5)
        axy.set_ylabel('Thermal Power [MW]')
        axy.set_xlim(0, 8760)
        axy.set_xticks(_MONTH_TICKS); axy.set_xticklabels(_MONTH_LABELS)
        axy.set_ylim(top=float(np.nanmax([pos_data.sum(axis=0).max(), demand.max()])) * 1.08)
        axy.grid(alpha=0.2)
        axy.legend(loc='lower center', bbox_to_anchor=(0.5, 1.01),
                   ncol=min(len(pos_cols)+1, 5), fontsize=15, frameon=False)
        figy.tight_layout()
        figy.savefig(results_dir / f'dispatch_year_{year}.png', dpi=150, bbox_inches='tight')
        plt.close(figy)
        logger.info(f"  Dispatch plot saved: {year}")


def _orc_elec_8760(model, year):
    """Hourly ORC electricity output [MW] for the year (0 if no revORC)."""
    m = model.model; tsa = model.tsa_by_period[year]
    if not (hasattr(m, "orc_share") and hasattr(model, "_revorc")):
        return None
    p_net = model._revorc["maps_by_p"][year]["orc_p_net_MW"]
    out = []
    for h in range(8760):
        d_rep = tsa["day_to_rep"][h // 24]; hr = h % 24
        out.append(pyo.value(m.orc_share[year, d_rep, hr]) * p_net[(d_rep, hr)])
    return np.array(out)


def save_winter_dispatch_and_csv(model, results_dir):
    """Clean standalone winter-week heat-dispatch panel + hourly per-tech CSV.

    Winter window Jan 15–22 (hours 336–504). The stacked area is heat to demand
    (all techs with capacity, incl. RevORC HP); the ORC electricity output is
    overlaid as a dashed line (right axis) since it is power, not heat.
    """
    W0, W1 = 336, 504
    hours = np.arange(8760)
    for year in INVESTMENT_YEARS:
        demand = _demand_for(model, year)
        df, stor_discharge, stor_charge = _reconstruct_8760(model, year)
        if df.empty:
            continue
        stor_discharge = {k: v for k, v in stor_discharge.items() if any(x > 1e-4 for x in v)}
        stor_charge = {k: v for k, v in stor_charge.items() if any(x < -1e-4 for x in v)}
        pos_cols = list(df.columns) + list(stor_discharge.keys())
        pos_data = np.vstack([df.values.T, np.array(list(stor_discharge.values()))]
                             if stor_discharge else [df.values.T])
        colors = [get_color(c) for c in pos_cols]
        orc_el = _orc_elec_8760(model, year)

        # ── hourly CSV (per-tech heat + storage + ORC electricity) ──
        out = {'Hour': hours, 'Demand_MW': demand}
        for c in df.columns:
            out[c] = df[c].values
        for k, v in {**stor_discharge, **stor_charge}.items():
            out[k] = np.array(v)
        if orc_el is not None:
            out['ORC_electricity_MW'] = orc_el
        pd.DataFrame(out).to_csv(results_dir / f'dispatch_hourly_{year}.csv', index=False)

        # ── clean winter panel ──
        sl = slice(W0, W1); hrs = hours[sl]
        fig, ax = plt.subplots(figsize=(11, 5.0))
        ax.stackplot(hrs, pos_data[:, sl], labels=pos_cols, colors=colors, alpha=0.9)
        ax.plot(hrs, demand[sl], 'k--', lw=2.0, label='Heat demand', zorder=10)
        ax.set_ylabel('Thermal power [MW]')
        ax.set_xlim(hrs[0], hrs[-1])
        ax.set_xticks(hrs[::24])
        ax.set_xticklabels([_soc_hour_label(h) for h in hrs[::24]], rotation=30, ha='right')
        ax.grid(alpha=0.2)
        handles, labels_ = ax.get_legend_handles_labels()
        if orc_el is not None and np.nanmax(orc_el[sl]) > 1e-3:
            ax2 = ax.twinx()
            l2, = ax2.plot(hrs, orc_el[sl], color='#9467bd', lw=2.2, ls=':',
                           label='ORC electricity')
            ax2.set_ylabel('ORC electricity [MW]', color='#9467bd')
            ax2.tick_params(axis='y', labelcolor='#9467bd')
            ax2.set_ylim(bottom=0)
            handles.append(l2); labels_.append('ORC electricity')
        ax.legend(handles, labels_, loc='lower center', bbox_to_anchor=(0.5, 1.01),
                  ncol=min(len(labels_), 4), fontsize=12, frameon=False)
        fig.tight_layout()
        fig.savefig(results_dir / f'dispatch_winter_{year}.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
    logger.info("  Winter dispatch panels + hourly CSVs saved")


def save_combined_dispatch_plot(model, results_dir):
    _MONTH_TICKS  = [0, 744, 1416, 2160, 2880, 3624, 4344, 5088, 5832, 6552, 7296, 8016]
    _MONTH_LABELS = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    hours  = np.arange(8760)
    year_data = {}
    all_pos_cols, all_chg_cols = [], []
    for year in INVESTMENT_YEARS:
        demand = _demand_for(model, year)
        df, stor_dis, stor_chg = _reconstruct_8760(model, year)
        if df.empty:
            continue
        stor_dis = {k: v for k, v in stor_dis.items() if any(x > 1e-4 for x in v)}
        stor_chg = {k: v for k, v in stor_chg.items() if any(x < -1e-4 for x in v)}
        pos_cols = list(df.columns) + list(stor_dis.keys())
        chg_cols = list(stor_chg.keys())
        pos_data = np.vstack([df.values.T, np.array(list(stor_dis.values()))] if stor_dis else [df.values.T])
        chg_data = np.array(list(stor_chg.values())) if stor_chg else None
        year_data[year] = (pos_cols, pos_data, chg_cols, chg_data)
        for c in pos_cols:
            if c not in all_pos_cols: all_pos_cols.append(c)
        for c in chg_cols:
            if c not in all_chg_cols: all_chg_cols.append(c)
    if not year_data:
        return

    n = len(year_data)
    fig, axes = plt.subplots(n, 1, figsize=(20, 4*n), sharex=True)
    if n == 1: axes = [axes]
    fig.suptitle('Hourly Thermal Dispatch — Munich, All Investment Years', y=0.98)

    legend_handles = {}
    for ax, year in zip(axes, sorted(year_data)):
        pos_cols, pos_data, chg_cols, chg_data = year_data[year]
        colors     = [get_color(c) for c in pos_cols]
        chg_colors = [get_color(c) for c in chg_cols]
        polys = ax.stackplot(hours, pos_data, labels=pos_cols, colors=colors, alpha=0.85)
        for poly, label in zip(polys, pos_cols):
            if label not in legend_handles: legend_handles[label] = poly
        if chg_data is not None:
            polys_c = ax.stackplot(hours, chg_data, labels=chg_cols, colors=chg_colors, alpha=0.75)
            for poly, label in zip(polys_c, chg_cols):
                if label not in legend_handles: legend_handles[label] = poly
        line, = ax.plot(hours, demand, 'k--', lw=1.2, zorder=10, label='Heat Demand')
        if 'Heat Demand' not in legend_handles: legend_handles['Heat Demand'] = line
        ax.axhline(0, color='black', lw=0.6, zorder=5)
        ax.set_ylabel('Power [MW]', fontsize=16)
        ax.set_title(f'{year}', fontsize=18, pad=4)
        ax.set_xlim(0, 8760)
        ax.set_xticks(_MONTH_TICKS)
        ax.set_xticklabels(_MONTH_LABELS if ax is axes[-1] else ['']*len(_MONTH_TICKS), fontsize=16)
        ax.grid(axis='y', alpha=0.2)

    fig.legend(handles=list(legend_handles.values()), labels=list(legend_handles.keys()),
               loc='lower center', ncol=min(len(legend_handles), 6),
               fontsize=14, framealpha=0.9, bbox_to_anchor=(0.5, 0.0))
    plt.tight_layout(rect=[0, 0.06, 1, 0.97])
    fig.savefig(results_dir / 'dispatch_combined.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("  Saved dispatch_combined.png")


# ── SOC plots ─────────────────────────────────────────────────────────────────
def save_soc_plots(model, results_dir):
    m = model.model
    if not hasattr(m, 'soc_inter'):
        return
    _MONTH_TICKS  = [0, 744, 1416, 2160, 2880, 3624, 4344, 5088, 5832, 6552, 7296, 8016]
    _MONTH_LABELS = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    stor_names = list(m.STOR)
    n_stor     = len(stor_names)
    hours      = np.arange(8760)

    for year in INVESTMENT_YEARS:
        soc_data = {s: np.array(_reconstruct_soc_8760(model, s, year)) for s in stor_names}

        fig, axes = plt.subplots(n_stor, 1, figsize=(18, 5.2*n_stor), sharex=True)
        if n_stor == 1: axes = [axes]
        for ax, s in zip(axes, stor_names):
            soc = soc_data[s]; color = get_color(s + ' (discharge)')
            ax.fill_between(hours, soc, alpha=0.25, color=color)
            ax.plot(hours, soc, color=color, lw=2.0)
            ax.set_ylabel('Energy [MWh]')
            ax.set_xlim(0, 8760); ax.grid(alpha=0.3)
            ax.text(0.99, 0.97, f'peak {soc.max():.0f} MWh',
                    transform=ax.transAxes, va='top', ha='right', fontsize=16,
                    bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.75))
        axes[-1].set_xticks(_MONTH_TICKS); axes[-1].set_xticklabels(_MONTH_LABELS)
        plt.tight_layout()
        plt.savefig(results_dir / f'soc_{year}.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"  SOC plot saved: {year}")

        fig2, axes2 = plt.subplots(n_stor, 1, figsize=(14, 4*n_stor))
        if n_stor == 1: axes2 = [axes2]
        fig2.suptitle(f'Storage SOC — 1-week zoom around peak, Munich {year}',
                      fontsize=14, fontweight='bold')
        for ax, s in zip(axes2, stor_names):
            soc = soc_data[s]; color = get_color(s + ' (discharge)')
            peak_h  = int(np.argmax(soc))
            w_start = max(0, min(peak_h - 84, 8760 - 168))
            ticks   = list(range(0, 169, 24))
            ax.fill_between(range(168), soc[w_start:w_start+168], alpha=0.25, color=color)
            ax.plot(soc[w_start:w_start+168], color=color, lw=1.5)
            ax.set_xticks(ticks)
            ax.set_xticklabels([_soc_hour_label(w_start+t) for t in ticks],
                               rotation=30, ha='right', fontsize=9)
            ax.set_ylabel('Energy [MWh]')
            ax.set_title(f'{s}  (peak {soc[peak_h]:.1f} MWh @ {_soc_hour_label(peak_h)})', fontsize=11)
            ax.set_xlim(0, 168); ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(results_dir / f'soc_{year}_zoom.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"  SOC zoom plot saved: {year}")


# ── BTES plots ────────────────────────────────────────────────────────────────
def save_btes_plots(model, results_dir):
    if not model.btes_enabled or not hasattr(model.model, 'btes_Qch'):
        return
    m = model.model; tsa = model.tsa; build_period = model.btes_build_period
    btes_years = sorted(m.BTES_YEARS)
    Tmin = model._btes_arx.get("Tmin", 80.8)
    # With an endogenous field design the temperature state is per design; report
    # the one the model selected.
    sel_design = None
    if hasattr(m, 'btes_y'):
        chosen = [k for k in m.BTES_DESIGNS if pyo.value(m.btes_y[k]) > 0.5]
        if not chosen:
            logger.info("BTES: no field design selected — skipping BTES plots")
            return
        sel_design = chosen[0]
        Tmin = float(model._btes_designs[sel_design]["Tmin"])
        logger.info(f"BTES: selected field design = {sel_design} boreholes "
                    f"(CAPEX {model._btes_designs[sel_design]['capex_EUR']/1e6:.2f} MEUR)")

    def _T(year, d):
        return (pyo.value(m.btes_T_k[sel_design, year, d]) if sel_design
                else pyo.value(m.btes_T[year, d]))
    period_length = INVESTMENT_YEARS[1] - INVESTMENT_YEARS[0] if len(INVESTMENT_YEARS) > 1 else 5
    year_to_period = {}
    for y in btes_years:
        for p in sorted(INVESTMENT_YEARS):
            if p <= y < p + period_length:
                year_to_period[y] = p; break
    active_periods = sorted({year_to_period[y] for y in btes_years})
    hours = np.arange(8760)

    T_by_year = {}
    for year in btes_years:
        p = year_to_period[year]
        d2r = model.tsa_by_period[p]['day_to_rep']
        T_daily = np.array([_T(year, d) for d in range(365)])
        T_prof  = np.repeat(T_daily, 24)
        T_by_year[year] = T_prof
        Qch  = np.array([pyo.value(m.btes_Qch[p,  d2r[h//24], h%24]) for h in range(8760)])
        Qdis = np.array([pyo.value(m.btes_Qdis[p, d2r[h//24], h%24]) for h in range(8760)])
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 8), sharex=True)
        ax1.plot(T_prof, color='#e74c3c', lw=1.2, label='Borehole temperature')
        ax1.axhline(Tmin, color='grey', ls='--', lw=0.8, label=f'T_min discharge ({Tmin} °C)')
        ax1.set_ylabel('Temperature [°C]'); ax1.legend(loc='upper right')
        ax1.set_title(f'BTES (built {build_period}) — year {year} (period {p}) [RD — Munich]')
        ax1.grid(alpha=0.3)
        ax2.fill_between(hours,  Qch/1000, label='Charge [MW]',     color='#3498db', alpha=0.7)
        ax2.fill_between(hours, -Qdis/1000, label='Discharge [MW]', color='#e67e22', alpha=0.7)
        ax2.set_ylabel('Power [MW]'); ax2.set_xlabel('Hour of year')
        ax2.legend(loc='upper right'); ax2.grid(alpha=0.3)
        ax2.text(0.01, 0.97,
                 f'Total charge: {Qch.sum()/1e6:.1f} GWh\nTotal discharge: {Qdis.sum()/1e6:.1f} GWh',
                 transform=ax2.transAxes, va='top', ha='left', fontsize=10,
                 bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.75))
        plt.tight_layout()
        plt.savefig(results_dir / f'btes_{year}.png', dpi=150, bbox_inches='tight')
        plt.close()

    all_T = np.concatenate([T_by_year[y] for y in btes_years])
    ymax  = float(all_T.max()) * 1.08 if all_T.max() > 0 else 100.0
    fig, ax = plt.subplots(figsize=(22, 6))
    ax.plot(np.arange(len(all_T)), all_T, color='#c0392b', lw=1.0)
    ax.axhline(Tmin, color='#888', ls='--', lw=1.2, label=f'T_min {Tmin} °C')
    for i, y in enumerate(btes_years):
        is_p = y in active_periods
        ax.axvline(i*8760, color='#222' if is_p else '#aaa', ls='-' if is_p else '--',
                   lw=1.2 if is_p else 0.5, alpha=0.9 if is_p else 0.5)
        ax.text(i*8760+80, ymax*0.96, str(y), fontsize=15 if is_p else 11,
                color='#111' if is_p else '#555', va='top')
    ax.set_xlim(0, len(all_T)); ax.set_ylim(bottom=0)
    ax.set_xlabel('Hour of planning horizon'); ax.set_ylabel('Borehole temperature [°C]')
    ax.legend(loc='upper right'); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(results_dir / 'btes_horizon.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("  BTES plots saved")


def save_btes_operation_plot(model, results_dir):
    """Storage operation over the year (temperature, charge/discharge power, discharge flow).

    Same figure as the webinar deck, written for every run that builds a field. Drawn from the
    files the run has just written, by tools/make_btes_run_operation.py.
    """
    if not getattr(model, "btes_enabled", False) or not hasattr(model.model, "btes_Qch"):
        return
    try:
        import importlib.util
        root = Path(__file__).resolve().parent
        spec = importlib.util.spec_from_file_location(
            "btes_run_operation", root / "tools" / "make_btes_run_operation.py")
        mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        arx = getattr(model, "_btes_arx", {}) or {}
        n_bh = arx.get("n_boreholes")
        if hasattr(model.model, "btes_y"):                     # endogenous design: report the chosen one
            chosen = [k for k in model.model.BTES_DESIGNS if pyo.value(model.model.btes_y[k]) > 0.5]
            if chosen:
                n_bh = model._btes_designs[chosen[0]].get("n_boreholes", chosen[0])
        if not n_bh:
            n_bh = int(os.environ.get("BTES_DESIGN_N") or 0) or None
        levels = arx.get("UA_dis_levels") or {}
        m_max = max((float(k) for k in levels), default=0.30)
        for p in mod.plot_run(results_dir, n_bh=int(n_bh) if n_bh else None, m_max=m_max):
            logger.info(f"  BTES operation plot: {Path(p).name}")
    except Exception as exc:                                    # never let a plot kill a run
        logger.warning(f"BTES operation plot skipped: {exc}")


def save_btes_timeseries(model, results_dir):
    """Export the BTES charge / discharge / temperature evolution to Excel.

    The plots in save_btes_plots() draw this same data but never wrote it out.
    Temperature is a DAILY state in the ARX surrogate (btes_T[year, day]); charge
    and discharge are hourly, reconstructed from the representative-day dispatch.
    Sheets: annual summary, daily temperature, and the full hourly series.
    """
    m = model.model
    if not model.btes_enabled or not hasattr(m, 'btes_Qch'):
        return
    sel_design, design = None, {}
    if hasattr(m, 'btes_y'):
        chosen = [k for k in m.BTES_DESIGNS if pyo.value(m.btes_y[k]) > 0.5]
        if not chosen:
            return
        sel_design = chosen[0]
        design = dict(model._btes_designs[sel_design])
    else:
        design = dict(getattr(model, '_btes_arx', {}))
        # with the design fixed via BTES_DESIGN_N there is no selection binary,
        # so the geometry lives only in the library file
        _dn = os.environ.get('BTES_DESIGN_N')
        if _dn:
            import json as _json
            _lib = Path(__file__).parent / 'data' / 'btes' / 'design_library.json'
            if _lib.exists():
                design.update(_json.loads(_lib.read_text()).get(str(_dn), {}))

    def _T(year, d):
        return (pyo.value(m.btes_T_k[sel_design, year, d]) if sel_design
                else pyo.value(m.btes_T[year, d]))

    btes_years = sorted(m.BTES_YEARS)
    plen = (INVESTMENT_YEARS[1] - INVESTMENT_YEARS[0]) if len(INVESTMENT_YEARS) > 1 else 5
    y2p = {}
    for y in btes_years:
        for p_ in sorted(INVESTMENT_YEARS):
            if p_ <= y < p_ + plen:
                y2p[y] = p_
                break

    hourly, daily, annual = [], [], []
    for year in btes_years:
        p_ = y2p[year]
        d2r = model.tsa_by_period[p_]['day_to_rep']
        T_daily = np.array([_T(year, d) for d in range(365)])
        qch = np.array([pyo.value(m.btes_Qch[p_, d2r[h // 24], h % 24]) for h in range(8760)]) / 1000.0
        qdis = np.array([pyo.value(m.btes_Qdis[p_, d2r[h // 24], h % 24]) for h in range(8760)]) / 1000.0
        for h in range(8760):
            hourly.append({'year': year, 'hour_of_year': h, 'day_of_year': h // 24 + 1,
                           'charge_MW': qch[h], 'discharge_MW': qdis[h],
                           'T_borehole_C': T_daily[h // 24]})
        for d in range(365):
            sl = slice(d * 24, d * 24 + 24)
            daily.append({'year': year, 'day_of_year': d + 1, 'T_borehole_C': T_daily[d],
                          'charge_MWh': qch[sl].sum(), 'discharge_MWh': qdis[sl].sum()})
        ch, dis = qch.sum(), qdis.sum()
        annual.append({'year': year, 'period': p_,
                       'charge_GWh': ch / 1000, 'discharge_GWh': dis / 1000,
                       'efficiency_pct': 100 * dis / ch if ch > 0 else np.nan,
                       'T_min_C': T_daily.min(), 'T_max_C': T_daily.max(),
                       'T_mean_C': T_daily.mean(), 'T_year_end_C': T_daily[-1],
                       'charge_peak_MW': qch.max(), 'discharge_peak_MW': qdis.max()})

    meta = [{'item': 'scenario', 'value': os.environ.get('SCENARIO', 'n/a')},
            {'item': 'field design [boreholes]', 'value': sel_design or design.get('n_boreholes', 'n/a')},
            {'item': 'borehole depth [m]', 'value': design.get('H_m', 'n/a')},
            {'item': 'total borehole length [m]', 'value': design.get('total_length_m', 'n/a')},
            {'item': 'CAPEX [EUR]', 'value': design.get('capex_EUR', 'n/a')},
            {'item': 'T_min for discharge [C]', 'value': design.get('Tmin', 'n/a')},
            {'item': 'T_max [C]', 'value': design.get('Tmax', 'n/a')},
            {'item': 'undisturbed ground T [C]', 'value': design.get('T_g', 'n/a')},
            {'item': 'built in period', 'value': model.btes_build_period},
            {'item': 'temperature resolution', 'value': 'daily (ARX state)'},
            {'item': 'charge/discharge resolution', 'value': 'hourly, from representative days'}]

    out = Path(results_dir) / 'btes_timeseries.xlsx'
    with pd.ExcelWriter(out, engine='openpyxl') as w:
        pd.DataFrame(meta).to_excel(w, sheet_name='README', index=False)
        pd.DataFrame(annual).to_excel(w, sheet_name='annual', index=False)
        pd.DataFrame(daily).to_excel(w, sheet_name='daily', index=False)
        pd.DataFrame(hourly).to_excel(w, sheet_name='hourly', index=False)
    logger.info(f"  BTES time series exported: {out.name} "
                f"({len(hourly):,} hourly rows, {len(btes_years)} years)")


# ── Capacity & generation plots ────────────────────────────────────────────────
def save_capacity_plot(res, results_dir):
    cap  = res['capacity_total']
    df   = pd.DataFrame(list(cap.items()), columns=['keys', 'Value'])
    df[['tech', 'year']] = pd.DataFrame(df['keys'].tolist(), index=df.index)
    pivot = df.pivot(index='year', columns='tech', values='Value').fillna(0)
    fig, ax = plt.subplots(figsize=(12, 7))
    pivot.plot(kind='bar', stacked=True, ax=ax, color=[get_color(c) for c in pivot.columns],
               edgecolor='white', width=0.8)
    ax.set_ylabel('Power [MW]')
    ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
    ax.set_title('Capacity Evolution — Munich')
    plt.tight_layout()
    plt.savefig(results_dir / 'capacity_evolution.png', dpi=300)
    plt.close()


def save_generation_plot(res, results_dir):
    gen = res['annual_generation']
    df  = pd.DataFrame(list(gen.items()), columns=['keys', 'Generation_MWh'])
    df[['Technology', 'Year']] = pd.DataFrame(df['keys'].tolist(), index=df.index)
    pivot = df.drop(columns='keys').pivot(
        index='Year', columns='Technology', values='Generation_MWh'
    ).fillna(0) / 1000  # GWh
    pivot = pivot.loc[INVESTMENT_YEARS, (pivot.loc[INVESTMENT_YEARS] > 0.01).any()]
    fig, ax = plt.subplots(figsize=(12, 7))
    pivot.plot(kind='bar', stacked=True, ax=ax, color=[get_color(c) for c in pivot.columns],
               alpha=0.9, edgecolor='white', width=0.7)
    ax.set_ylabel('Generation [GWh]'); ax.set_xlabel('Investment Period')
    ax.set_xticklabels([str(y) for y in INVESTMENT_YEARS], rotation=0)
    ax.set_title('Generation Mix — Munich')
    ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(results_dir / 'generation_mix.png', dpi=300)
    plt.close()
    logger.info("  Generation mix plot saved")


# ── TSA analysis ───────────────────────────────────────────────────────────────
def save_tsa_analysis(model, demand, results_dir):
    # NOTE: diagnostic of the FIRST period's clustering only; under
    # TSA_MODE=per_period the other milestones have their own (similar) mappings.
    tsa        = model.tsa
    n_rep      = tsa['n_rep_days']
    weights    = tsa['weights']
    day_to_rep = tsa['day_to_rep']
    rep_data   = tsa['rep_data']
    demand_full = demand['heat_demand'].values

    ncols = min(4, n_rep); nrows = (n_rep + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 3.5*nrows), sharey=True)
    for d in range(n_rep):
        ax = np.array(axes).flatten()[d]
        profile = rep_data['heat_demand'].values[d*24:(d+1)*24]
        ax.plot(range(24), profile, lw=2, color='steelblue')
        ax.fill_between(range(24), profile, alpha=0.25, color='steelblue')
        ax.set_xlabel(f'RD{d} ×{weights[d]}d', labelpad=2); ax.set_ylabel('MW'); ax.grid(alpha=0.3)
    for ax in np.array(axes).flatten()[n_rep:]:
        ax.set_visible(False)
    plt.tight_layout()
    plt.savefig(results_dir / 'tsa_rep_day_profiles.png', dpi=300); plt.close()

    reconstructed = np.array([rep_data['heat_demand'].values[day_to_rep[h//24]*24 + h%24]
                               for h in range(8760)])
    residuals = demand_full - reconstructed
    rmse  = float(np.sqrt(np.mean(residuals**2)))
    mae   = float(np.mean(np.abs(residuals)))
    nrmse = rmse / (demand_full.max() - demand_full.min()) * 100
    logger.info(f"  RMSE: {rmse:.3f} MW | MAE: {mae:.3f} MW | NRMSE: {nrmse:.2f}%"
                f" | bias: {reconstructed.mean()-demand_full.mean():+.3f} MW")

    fig, axes = plt.subplots(2, 1, figsize=(20, 8))
    axes[0].plot(demand_full, lw=0.8, alpha=0.8, label='Original', color='steelblue')
    axes[0].plot(reconstructed, lw=0.8, alpha=0.8, label='Reconstructed', color='orange')
    axes[0].set_ylabel('MW'); axes[0].legend(); axes[0].grid(alpha=0.2)
    axes[1].plot(residuals, lw=0.6, color='crimson', alpha=0.7)
    axes[1].axhline(0, color='black', lw=0.8)
    axes[1].set_ylabel('Error [MW]'); axes[1].set_xlabel('Hour of Year'); axes[1].grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(results_dir / 'tsa_error_profile.png', dpi=300); plt.close()

    pd.DataFrame({'Metric': ['n_rep_days','rmse_MW','mae_MW','nrmse_pct','mean_bias_MW'],
                  'Value':  [n_rep, round(rmse,3), round(mae,3), round(nrmse,3),
                             round(float(reconstructed.mean()-demand_full.mean()),3)]})\
      .to_csv(results_dir / 'tsa_summary.csv', index=False)
    logger.info(f"  TSA analysis saved ({n_rep} rep days, NRMSE={nrmse:.2f}%)")


# ── Excel export ───────────────────────────────────────────────────────────────
def _compute_cost_co2_breakdown(model):
    """
    Compute per-(Technology, Year) breakdown of costs and CO2 emissions.

    Returns a long-form DataFrame with columns:
        Technology, Year, Carrier,
        Capacity_added_MW, Capacity_total_MW, Annual_generation_MWh,
        Fuel_consumption_MWh,   ← MWh of primary fuel (gas) or electricity input
        CO2_emissions_tCO2,     ← physical, undiscounted
        CAPEX_EUR,              ← undiscounted investment cost (cap_add × capex/MW)
        OPEX_fix_EUR,           ← annual fixed OPEX (cap_total × opex_fix/MW/yr)
        OPEX_var_EUR,           ← annual variable OPEX (gen × opex_var/MWh)
        Fuel_cost_EUR,          ← annual fuel / electricity cost
        Carbon_cost_EUR,        ← annual carbon cost (CO2 × carbon_price)
        Total_running_cost_EUR  ← OPEX_fix + OPEX_var + Fuel_cost + Carbon_cost
    """
    m  = model.model
    _v = lambda x: float(pyo.value(x))

    records = []
    for t in m.TECH:
        carrier = model.technologies[t].get('carrier', '')
        for p in m.PERIODS:
            annual_gen     = 0.0
            fuel_kwh       = 0.0   # MWh of primary fuel or electricity consumed
            elec_cost      = 0.0   # electricity cost from hourly prices (elec carrier only)

            for d in m.REP_DAYS:
                w = _v(m.day_weight[p, d])
                for h in m.HOURS:
                    gen_val = _v(m.gen[t, p, d, h])
                    eff_val = _v(m.efficiency[t, p, d, h])
                    annual_gen += gen_val * w
                    if eff_val > 1e-9:
                        fuel = gen_val / eff_val * w
                        fuel_kwh += fuel
                        if carrier == 'electricity':
                            ep = _v(m.electricity_price[p, d, h])
                            elec_cost += fuel * ep

            ef      = _v(m.emission_factor[t, p])
            fp      = _v(m.fuel_price[t, p])
            cp      = _v(m.carbon_price[p])
            cap_add = _v(m.cap_add[t, p])
            cap_tot = _v(m.cap_total[t, p])

            co2 = fuel_kwh * ef

            if carrier == 'electricity':
                fuel_cost = elec_cost
            else:
                fuel_cost = fuel_kwh * fp

            capex_eur    = cap_add * _v(m.capex[t])
            opex_fix_eur = cap_tot * _v(m.opex_fix[t])
            opex_var_eur = annual_gen * _v(m.opex_var[t])
            carbon_eur   = co2 * cp
            running_eur  = opex_fix_eur + opex_var_eur + fuel_cost + carbon_eur

            records.append({
                'Technology':           t,
                'Year':                 p,
                'Carrier':              carrier,
                'Capacity_added_MW':    round(cap_add, 4),
                'Capacity_total_MW':    round(cap_tot, 4),
                'Annual_generation_MWh': round(annual_gen, 1),
                'Fuel_consumption_MWh': round(fuel_kwh, 1),
                'CO2_emissions_tCO2':   round(co2, 1),
                'CAPEX_EUR':            round(capex_eur, 0),
                'OPEX_fix_EUR':         round(opex_fix_eur, 0),
                'OPEX_var_EUR':         round(opex_var_eur, 0),
                'Fuel_cost_EUR':        round(fuel_cost, 0),
                'Carbon_cost_EUR':      round(carbon_eur, 0),
                'Total_running_cost_EUR': round(running_eur, 0),
            })

    return pd.DataFrame(records)


def export_excel(model, res, results_dir, comp_params=None):
    m    = model.model
    path = results_dir / 'results_rd.xlsx'
    with pd.ExcelWriter(path, engine='openpyxl') as writer:
        pd.DataFrame([{'model': f'RD ({N_TYPICAL_DAYS} typical days) — Munich',
                       'objective_EUR': res['objective_value'],
                       'variables': m.nvariables(), 'constraints': m.nconstraints(),
                       'milp': model.uc_enabled}]).to_excel(writer, sheet_name='summary', index=False)
        if comp_params:
            pd.DataFrame(list(comp_params.items()), columns=['Parameter','Value'])\
                .to_excel(writer, sheet_name='computational', index=False)

        # ── Capacity ──────────────────────────────────────────────────────────
        cap = res['capacity_total']
        df_cap = pd.DataFrame(list(cap.items()), columns=['keys','Capacity_MW'])
        df_cap[['Technology','Year']] = pd.DataFrame(df_cap['keys'].tolist(), index=df_cap.index)
        df_cap.drop(columns='keys').pivot(index='Year', columns='Technology',
                                          values='Capacity_MW').fillna(0)\
            .to_excel(writer, sheet_name='capacity')

        # ── Generation ────────────────────────────────────────────────────────
        gen = res['annual_generation']
        df_gen = pd.DataFrame(list(gen.items()), columns=['keys','Generation_MWh'])
        df_gen[['Technology','Year']] = pd.DataFrame(df_gen['keys'].tolist(), index=df_gen.index)
        df_gen.drop(columns='keys').pivot(index='Year', columns='Technology',
                                          values='Generation_MWh').fillna(0)\
            .to_excel(writer, sheet_name='generation')

        # ── Cost & CO2 breakdown ──────────────────────────────────────────────
        df_bk = _compute_cost_co2_breakdown(model)
        df_bk.to_excel(writer, sheet_name='cost_co2_breakdown', index=False)

        # Pivot summaries
        for col, sheet in [
            ('CO2_emissions_tCO2',    'co2_by_tech'),
            ('Total_running_cost_EUR', 'running_cost_by_tech'),
            ('CAPEX_EUR',             'capex_by_tech'),
            ('Fuel_cost_EUR',         'fuel_cost_by_tech'),
        ]:
            df_bk.pivot(index='Year', columns='Technology', values=col).fillna(0)\
                .to_excel(writer, sheet_name=sheet)

        # ── SOC (storage) ─────────────────────────────────────────────────────
        if hasattr(m, 'soc_inter'):
            for year in INVESTMENT_YEARS:
                soc_data = {'Hour': range(8760)}
                for s in m.STOR:
                    soc_data[s] = _reconstruct_soc_8760(model, s, year)
                pd.DataFrame(soc_data).to_excel(writer, sheet_name=f'soc_{year}', index=False)

        # ── Hourly dispatch per year ──────────────────────────────────────────
        demand_mw = model.demand_data['heat_demand'].values
        for year in INVESTMENT_YEARS:
            df_d, stor_dis, stor_chg = _reconstruct_8760(model, year)
            if df_d.empty: continue
            row = {'Hour': list(range(8760)), 'Demand_MW': list(demand_mw)}
            for t in df_d.columns: row[t] = df_d[t].tolist()
            for k, v in stor_dis.items(): row[k] = v
            for k, v in stor_chg.items(): row[k] = v
            if hasattr(m, 'reinject'):
                d2r = model.tsa_by_period[year]['day_to_rep']
                reinject_8760 = [
                    pyo.value(m.reinject[year, d2r[h // 24], h % 24])
                    for h in range(8760)
                ]
                row['Reinjection_MW'] = reinject_8760
            pd.DataFrame(row).to_excel(writer, sheet_name=f'dispatch_{year}', index=False)

        # ── Mode analysis ─────────────────────────────────────────────────────
        if hasattr(m, 'y_ci'):
            df_mode = _compute_mode_stats(model)
            df_mode.to_excel(writer, sheet_name='mode_stats', index=False)
            df_mode.pivot(index='Year', columns='Mode', values='Hours_hyr').fillna(0)\
                .to_excel(writer, sheet_name='mode_hours')
            df_mode.pivot(index='Year', columns='Mode', values='Total_heat_MWh').fillna(0)\
                .to_excel(writer, sheet_name='mode_heat_gen')
            _compute_economics_table(model).to_excel(writer, sheet_name='hthp_economics', index=False)
            demand_mw = model.demand_data['heat_demand'].values
            for year in INVESTMENT_YEARS:
                du, ci_arr, ti = _reconstruct_mode_8760(model, year)
                pd.DataFrame({
                    'Hour':      range(8760),
                    'Demand_MW': demand_mw,
                    'Mode':      ['DU' if d else ('CI' if c else 'TI')
                                  for d, c, t in zip(du, ci_arr, ti)],
                    'Mode_DU':   du.tolist(),
                    'Mode_CI':   ci_arr.tolist(),
                    'Mode_TI':   ti.tolist(),
                }).to_excel(writer, sheet_name=f'mode_dispatch_{year}', index=False)
    logger.info(f"Excel exported: {path}")


def collect_computational_params(model, results_raw, solve_wall_time):
    m = model.model
    params = {'model_type': f'RD ({N_TYPICAL_DAYS} typical days) — Munich',
              'uc_enabled': model.uc_enabled, 'n_periods': len(model.investment_years),
              'n_technologies': len(model.technologies), 'n_storages': len(model.storage),
              'n_rep_days': model.tsa['n_rep_days'],
              'pyomo_variables': m.nvariables(), 'pyomo_constraints': m.nconstraints(),
              'wall_time_s': round(solve_wall_time, 2)}
    try:
        grb = m._solver_model
        params.update({'gurobi_status': grb.Status, 'gurobi_runtime_s': round(grb.Runtime, 2),
                       'obj_value': grb.ObjVal, 'obj_bound': grb.ObjBound,
                       'mip_gap_pct': round(grb.MIPGap*100, 4), 'node_count': int(grb.NodeCount),
                       'simplex_iterations': int(grb.IterCount), 'num_vars': int(grb.NumVars),
                       'num_constraints': int(grb.NumConstrs), 'num_binary_vars': int(grb.NumBinVars),
                       'num_nonzeros': int(grb.NumNZs)})
    except Exception:
        pass
    return params


def save_computational_params(model, results_raw, solve_wall_time, results_dir):
    params = collect_computational_params(model, results_raw, solve_wall_time)
    pd.DataFrame(list(params.items()), columns=['Parameter','Value'])\
        .to_csv(results_dir / 'computational_params.csv', index=False)
    logger.info("── Computational Summary ─────────────────────────────────")
    for k, v in params.items():
        logger.info(f"  {k:<25} {v}")
    logger.info("──────────────────────────────────────────────────────────")


# ── SAPHEA three-mode coupling ────────────────────────────────────────────────
def _build_saphea_coupling(model, demand, resource):
    """
    Post-build: SAPHEA three-mode dispatch for geothermal + HTHP.

    All capacities and the COP are DERIVED from ``resource`` (the 'Geothermal
    Resource' sheet): brine_mass_flow, brine_cp, T_prod, T_inject_min,
    T_return_design, direct_hx_pinch, carnot_eta, T_supply_design.

    Two binary variables select the operating mode per (period, d, h):
      y_ci=0, y_ti=0  →  Direct Use (DU):
                          gen[Geo] ≤ Q_GEO_MAX_DIRECT,  gen[HTHP] = 0
      y_ci=1, y_ti=0  →  Capacity Increase (CI):
                          gen[Geo] ≤ Q_GEO_MAX_CI,       gen[HTHP] ≤ gen[Geo]×ratio[d,h]
      y_ci=0, y_ti=1  →  Temperature Increase (TI):
                          gen[Geo] = 0,                  gen[HTHP] ≤ Q_HP_MAX_TI[d,h]
      y_ci + y_ti ≤ 1    (mutual exclusivity)

    SAPHEA formulas (exact from Saphea_LSGEOHP-tool.html):
      COP_SAPHEA[d,h]   = η × (T_supply[d,h] + 273.15) / (T_supply[d,h] − T_inject_min)
      COP_standard[d,h] = COP_SAPHEA + 1          ← used for electricity cost in MILP
      CI coupling_ratio  = (dT_evap_ci / dT_direct) × (1 + 1/COP_SAPHEA)
        dT_evap_ci  = T_inject_no_hp − T_inject_min   (HTHP evap range in CI mode)
        dT_direct   = T_prod − T_inject_no_hp          (direct HX range)
      TI max HP output  = Q_EVAP_MAX_TI × (1 + 1/COP_SAPHEA)
        Q_EVAP_MAX_TI   = ṁ × cp × (T_prod − T_inject_min)   (full brine, TI mode)

    Note: TI mode is not applicable for Munich (T_prod=106°C > T_supply always),
    but the constraints are included so this function generalises to other case
    studies where T_prod < T_supply.
    """
    m   = model.model

    if HTHP_TECH not in model.technologies:
        logger.warning(f"'{HTHP_TECH}' not in model technologies — SAPHEA skipped")
        return

    n_rep = model.tsa['n_rep_days']
    years = sorted(model.investment_years)

    # ── Resource inputs & design capacities (from the 'Geothermal Resource' sheet) ──
    mdot    = resource["brine_mass_flow"]
    cp      = resource["brine_cp"]
    T_prod  = resource["T_prod"]
    T_min   = resource["T_inject_min"]
    PINCH_K = resource["direct_hx_pinch"]
    eta     = resource["carnot_eta"]
    design  = derive_saphea_capacities(resource)
    Q_geo_du_design = design["Q_geo_du"]      # plant rating cap for DU
    Q_evap_ti       = design["Q_evap_ti"]     # full-brine TI evaporator

    def _q_brine(dT):                                  # brine heat for a ΔT → MW
        return mdot * cp * max(dT, 0.0) / 1000.0

    # ── Average supply & return temperature per (period, rep_day, hour) from the
    #    measured 8760h profiles. SUPPLY temp drives the COP & coupling ratio; RETURN
    #    temp sets how deeply the brine can be cooled, hence the geo & HTHP
    #    capacities. Under per-period TSA (clustering_mode='per_period') each
    #    milestone has its OWN day→rep mapping, so all derived quantities are
    #    period-indexed; in 'joint' mode the periods share one mapping and the
    #    values simply repeat. ──
    supply_arr = demand['supply_temp'].values    # measured T_VL_FW_C, °C
    return_arr = demand['return_temp'].values     # measured T_RL_FW_C, °C
    rep_supply, rep_return = {}, {}              # period → (n_rep, 24) arrays
    for p_ in years:
        d2r = model.tsa_by_period[p_]['day_to_rep']
        sum_s = np.zeros((n_rep, 24)); sum_r = np.zeros((n_rep, 24))
        cnt   = np.zeros((n_rep, 24))
        for h in range(8760):
            d  = d2r[h // 24]
            hr = h % 24
            sum_s[d, hr] += supply_arr[h]
            sum_r[d, hr] += return_arr[h]
            cnt[d, hr]   += 1
        rep_supply[p_] = sum_s / cnt    # °C, shape (n_rep, 24)
        rep_return[p_] = sum_r / cnt    # °C, shape (n_rep, 24)

    # ── Per-(period,d,h) hourly quantities (all return/supply-temp dependent) ──
    #   DU : brine cooled 106 °C → T_return                → q_geo_du
    #        (capped at the design plant rating Q_GEO_MAX_DIRECT)
    #   CI : direct HX 106 °C → T_return+pinch              → q_geo_ci (= du − delta)
    #        HP evaporator (T_return+pinch) → T_inject_min   → coupling ratio
    #   COP: supply-temp driven, evaporator cold end = injection floor
    coupling_ci   = {}
    hp_max_ti     = {}
    q_geo_du      = {}      # MW, DU geo ceiling per (p,d,h)
    geo_delta     = {}      # MW, geo lost when switching DU → CI per (p,d,h)
    ci_hp_ceiling = {}      # MW, max HTHP output the brine can support per (p,d,h)
    cop_standard_log = []
    for p_ in years:
        for d in range(n_rep):
            for h in range(24):
                T_s    = float(rep_supply[p_][d, h])
                T_ret  = float(rep_return[p_][d, h])
                T_woHP = T_ret + PINCH_K                   # CI direct-HX brine exit

                # Hourly geo heat (return-temp dependent); DU capped at plant rating
                q_du = min(_q_brine(T_prod - T_ret), Q_geo_du_design)
                q_ci = _q_brine(T_prod - T_woHP)
                q_geo_du[(p_, d, h)]  = q_du
                geo_delta[(p_, d, h)] = q_du - q_ci

                # COP (supply-temp driven; evaporator cold end = geological injection floor)
                dT_lift      = max(T_s - T_min, 1.0)
                cop_saphea   = eta * (T_s + 273.15) / dT_lift
                cop_standard = cop_saphea + 1.0

                # CI coupling ratio = (evap ΔT / direct ΔT) × (1 + 1/COP), all hourly
                dT_evap_ci = T_woHP - T_min
                dT_direct  = T_prod - T_woHP
                ratio_ci   = (dT_evap_ci / dT_direct) * (1.0 + 1.0 / cop_saphea)
                q_hp_ti    = Q_evap_ti * (1.0 + 1.0 / cop_saphea)   # full-brine TI ceiling

                m.efficiency[HTHP_TECH, p_, d, h].set_value(cop_standard)
                coupling_ci[(p_, d, h)]   = ratio_ci
                hp_max_ti[(p_, d, h)]     = q_hp_ti
                ci_hp_ceiling[(p_, d, h)] = q_ci * ratio_ci    # = q_evap × (1 + 1/COP)
                cop_standard_log.append(cop_standard)

    m.hthp_coupling_ratio = pyo.Param(m.PERIODS, m.REP_DAYS, m.HOURS, initialize=coupling_ci, mutable=False)
    m.hthp_hp_max_ti      = pyo.Param(m.PERIODS, m.REP_DAYS, m.HOURS, initialize=hp_max_ti,   mutable=False)
    m.q_geo_du_cap        = pyo.Param(m.PERIODS, m.REP_DAYS, m.HOURS, initialize=q_geo_du,    mutable=False)
    m.geo_delta_ci        = pyo.Param(m.PERIODS, m.REP_DAYS, m.HOURS, initialize=geo_delta,   mutable=False)

    # ── Physical feasibility: TI required iff T_supply > T_prod ──
    # SAPHEA TI mode is the ONLY valid mode when brine is too cold for direct DHN supply.
    # ti_required = 1 → T_supply > T_prod → force y_ti=1, block DU and CI.
    # ti_required = 0 → T_supply ≤ T_prod → block y_ti, let optimizer choose DU/CI.
    ti_req = {
        (p_, d, h): int(float(rep_supply[p_][d, h]) > T_prod)
        for p_ in years for d in range(n_rep) for h in range(24)
    }
    m.ti_required = pyo.Param(m.PERIODS, m.REP_DAYS, m.HOURS, initialize=ti_req, within=pyo.Binary)

    n_ti_slots = sum(ti_req.values())
    logger.info(
        f"  TI required in {n_ti_slots}/{len(ti_req)} rep (day,hour) slots "
        f"(T_supply > T_prod={T_prod:.0f}°C)"
    )

    # ── Brine-physical maxima (derived) vs the installed capacities ──────────────
    # Capacities are now DERIVED from the resource and written into the technology
    # dict in main() before build, so the installed values should already equal the
    # derived design ratings. We re-derive the over-the-profile ceiling here and warn
    # if the installed HTHP rating exceeds what the brine can ever dispatch (which
    # would mean paying CAPEX for capacity that can never run).
    q_geo_installed  = float(model.technologies[GEO_TECH]['max_output_MW'])
    q_hthp_installed = float(model.technologies[HTHP_TECH]['max_output_MW'])

    ci_ceiling = max(ci_hp_ceiling.values())     # max q_evap × (1 + 1/COP) over all hours
    ti_ceiling = max((hp_max_ti[k] for k, req in ti_req.items() if req), default=0.0)
    phys_max_hthp = max(ci_ceiling, ti_ceiling)

    if q_hthp_installed > phys_max_hthp * (1.0 + 1e-3):
        if os.environ.get('NET_TEMP_SHIFT_K'):
            # network-temperature sensitivity: the shifted return leaves less low-grade brine for the
            # HTHP, so clamp its rating to the brine-physical maximum instead of aborting
            logger.warning(f"NET_TEMP_SHIFT_K set: HTHP rating {q_hthp_installed:.2f} MW clamped to brine-physical "
                           f"maximum {phys_max_hthp:.2f} MW")
            model.technologies[HTHP_TECH]['max_output_MW'] = float(phys_max_hthp)
            q_hthp_installed = float(phys_max_hthp)
        else:
            raise ValueError(
                f"HTHP 'Maximal Output Power' = {q_hthp_installed:.2f} MW exceeds the "
                f"brine-physical maximum of {phys_max_hthp:.2f} MW.\n"
                f"  The HTHP is coupled to the geothermal brine (flow {mdot} kg/s, "
                f"{T_prod:.0f}°C wellhead, {T_min:.0f}°C injection floor).\n"
                f"  Max dispatchable HTHP (brine-limited, CI mode, over the return-temp "
                f"profile) = {ci_ceiling:.2f} MW"
                + (f", or {ti_ceiling:.2f} MW in TI mode" if ti_ceiling > 0 else "") + ".\n"
                f"  Reduce the brine resource / HTHP rating accordingly."
            )

    logger.info(
        f"  Capacities derived from resource — geo {q_geo_installed:.2f} MW "
        f"(design {Q_geo_du_design:.2f} MW), HTHP {q_hthp_installed:.2f} MW "
        f"(design {design['Q_hthp_max']:.2f} MW, brine-profile max {phys_max_hthp:.2f} MW)"
    )

    # Stash design values for downstream reporting (economics table, plots).
    model._saphea_design = design

    # ── Mode binary variables ──
    m.y_ci = pyo.Var(m.PERIODS, m.REP_DAYS, m.HOURS, domain=pyo.Binary)   # Capacity Increase
    m.y_ti = pyo.Var(m.PERIODS, m.REP_DAYS, m.HOURS, domain=pyo.Binary)   # Temperature Increase

    # ── Mutual exclusivity: at most one mode active per slot ──
    def mode_mutex_rule(m, period, d, h):
        return m.y_ci[period, d, h] + m.y_ti[period, d, h] <= 1
    m.mode_mutex = pyo.Constraint(m.PERIODS, m.REP_DAYS, m.HOURS, rule=mode_mutex_rule)

    # ── Physical mode lock: enforce y_ti when T_supply > T_prod; block it otherwise ──
    # y_ti ≥ ti_required  → forces TI when brine is too cold (T_supply > T_prod)
    # y_ti ≤ ti_required  → blocks TI when brine is hot enough (T_supply ≤ T_prod)
    # Together: y_ti = ti_required (fully fixed by temperature physics, period-independent)
    def ti_force_rule(m, period, d, h):
        return m.y_ti[period, d, h] >= m.ti_required[period, d, h]
    m.ti_force = pyo.Constraint(m.PERIODS, m.REP_DAYS, m.HOURS, rule=ti_force_rule)

    def ti_block_rule(m, period, d, h):
        return m.y_ti[period, d, h] <= m.ti_required[period, d, h]
    m.ti_block = pyo.Constraint(m.PERIODS, m.REP_DAYS, m.HOURS, rule=ti_block_rule)

    # y_ci ≤ 1 - ti_required  → blocks CI (and implicitly DU stays) when TI is required
    def ci_block_ti_rule(m, period, d, h):
        return m.y_ci[period, d, h] <= 1 - m.ti_required[period, d, h]
    m.ci_block_ti = pyo.Constraint(m.PERIODS, m.REP_DAYS, m.HOURS, rule=ci_block_ti_rule)

    # ── Geo capacity ceiling (mode-dependent, hourly via return-temp profile) ──
    # DU  (y_ci=0, y_ti=0): gen[Geo] ≤ q_geo_du_cap[d,h]
    # CI  (y_ci=1, y_ti=0): gen[Geo] ≤ q_geo_du_cap[d,h] − geo_delta_ci[d,h]  (= q_geo_ci)
    # TI  (y_ci=0, y_ti=1): gen[Geo] ≤ 0               (all brine to HTHP evap in TI mode)
    def geo_mode_cap_rule(m, period, d, h):
        return (
            m.gen[GEO_TECH, period, d, h]
            <= m.q_geo_du_cap[period, d, h] * (1 - m.y_ti[period, d, h])
               - m.geo_delta_ci[period, d, h] * m.y_ci[period, d, h]
        )
    m.geo_mode_cap = pyo.Constraint(m.PERIODS, m.REP_DAYS, m.HOURS, rule=geo_mode_cap_rule)

    _q_hthp_max = model.technologies[HTHP_TECH]['max_output_MW']   # = 8.82 MW

    # ── HTHP = 0 in Direct Use mode (y_ci = y_ti = 0) ──
    def hthp_mode_off_rule(m, period, d, h):
        return (
            m.gen[HTHP_TECH, period, d, h]
            <= _q_hthp_max * (m.y_ci[period, d, h] + m.y_ti[period, d, h])
        )
    m.hthp_mode_off = pyo.Constraint(m.PERIODS, m.REP_DAYS, m.HOURS, rule=hthp_mode_off_rule)

    # ── CI brine coupling: gen[HTHP] ≤ gen[Geo] × ratio[d,h] when y_ci=1 ──
    # Big-M deactivates when y_ci=0 (HTHP already zeroed by mode_off)
    def hthp_ci_coupling_rule(m, period, d, h):
        return (
            m.gen[HTHP_TECH, period, d, h]
            <= m.gen[GEO_TECH, period, d, h] * m.hthp_coupling_ratio[period, d, h]
               + _q_hthp_max * (1 - m.y_ci[period, d, h])
        )
    m.hthp_ci_coupling = pyo.Constraint(m.PERIODS, m.REP_DAYS, m.HOURS, rule=hthp_ci_coupling_rule)

    # ── TI brine capacity: gen[HTHP] ≤ Q_HP_MAX_TI[d,h] when y_ti=1 ──
    # Bounds HTHP by available brine heat in TI mode; Big-M deactivates when y_ti=0
    def hthp_ti_cap_rule(m, period, d, h):
        return (
            m.gen[HTHP_TECH, period, d, h]
            <= m.hthp_hp_max_ti[period, d, h] * m.y_ti[period, d, h]
               + _q_hthp_max * (1 - m.y_ti[period, d, h])
        )
    m.hthp_ti_cap = pyo.Constraint(m.PERIODS, m.REP_DAYS, m.HOURS, rule=hthp_ti_cap_rule)

    cop_arr = np.array(cop_standard_log)
    cr_vals = list(coupling_ci.values())
    ti_vals = list(hp_max_ti.values())
    du_vals = list(q_geo_du.values())
    ci_vals = [q_geo_du[k] - geo_delta[k] for k in q_geo_du]
    logger.info(
        f"SAPHEA 3-mode built — COP_std {cop_arr.min():.2f}–{cop_arr.max():.2f} "
        f"(mean {cop_arr.mean():.2f})"
    )
    logger.info(
        f"  DU : geo ≤ {min(du_vals):.2f}–{max(du_vals):.2f} MW  "
        f"(return-temp dependent, capped at {Q_geo_du_design:.2f} MW),  HTHP = 0"
    )
    logger.info(
        f"  CI : geo ≤ {min(ci_vals):.2f}–{max(ci_vals):.2f} MW,  HTHP ≤ geo×ratio  "
        f"(ratio {min(cr_vals):.3f}–{max(cr_vals):.3f})"
    )
    logger.info(
        f"  TI : geo = 0,  HTHP ≤ Q_HP_MAX_TI  "
        f"(brine-limited {min(ti_vals):.1f}–{max(ti_vals):.1f} MW; "
        f"capped at {_q_hthp_max:.2f} MW by installed capacity)"
    )
    if T_prod >= max(demand['supply_temp'].values):
        logger.info(
            f"  Note: TI inactive for this case "
            f"(T_prod={T_prod:.0f}°C ≥ max T_supply={max(demand['supply_temp'].values):.1f}°C)"
        )


# ── revORC (reversible HP/ORC) ────────────────────────────────────────────────

def _inject_revorc_tech(techs, cfg):
    """Add the revORC as a greenfield technology to the tech dict (before build).

    Capacity is expressed in MW_th of HP-mode output (= the paper's CAPEX basis);
    the ORC capability of the same modules comes at no extra cost. The hourly COP
    is a placeholder here and is overwritten per (d, h) in _build_revorc."""
    techs[REVORC_TECH] = {
        "installed_capacity_MW": 0.0,
        "efficiency_th":         4.0,          # placeholder, set hourly post-build
        "efficiency_el":         0.0,
        "capex_EUR_per_MW":      cfg["capex_eur_per_MWth"],
        # fixed OPEX = VDI 2067 repair + servicing. The grid capacity fee is added
        # as an objective term in _build_revorc, where the HP-mode COP is known.
        "opex_fix_EUR_per_MWa":  cfg["opex_fix_frac"] * cfg["capex_eur_per_MWth"],
        # HP-mode variable O&M, charged on delivered heat. The ORC-mode variable
        # O&M is charged on generated electricity in _build_revorc.
        "opex_var_EUR_per_MWh":  cfg.get("opex_var_hp_eur_per_MWh_th", 0.0),
        "energy_carrier":        "electricity",
        "coldstart_cost_EUR":    0.0,
        "cold_start_time_h":     0,
        "ramp_cost_EUR_per_MW":  0.0,
        "min_load_factor":       0.0,
        "lifetime_a":            int(cfg["lifetime_a"]),
        "max_potential_MWh":     0.0,
        "max_output_MW":         cfg["max_modules"] * cfg["module_hp_MWth"],
        "installation_year":     None,
    }
    logger.info(
        f"Injected '{REVORC_TECH}' (greenfield): max "
        f"{techs[REVORC_TECH]['max_output_MW']:.1f} MW_th, "
        f"CAPEX {cfg['capex_eur_per_MWth']:,.0f} €/MW_th, "
        f"OPEX fix {techs[REVORC_TECH]['opex_fix_EUR_per_MWa']:,.0f} €/MW/a, "
        f"OPEX var {techs[REVORC_TECH]['opex_var_EUR_per_MWh']:.2f} €/MWh_th (HP mode)"
    )


def _rep_profile(values, day_to_rep, n_rep):
    """Average an 8760-h array onto the (rep_day, hour) grid (as in SAPHEA)."""
    sums = np.zeros((n_rep, 24))
    cnt = np.zeros((n_rep, 24))
    for t in range(8760):
        d, hr = day_to_rep[t // 24], t % 24
        sums[d, hr] += values[t]
        cnt[d, hr] += 1
    return sums / cnt




def _add_grid_capacity_charge(model, fee_eur_per_kW_yr: float, k_pump: float = 0.0):
    """Leistungspreis Netzkosten on the PEAK electricity draw of each period.

    A capacity charge is levied on the highest power a site draws, not on its
    energy, so it is modelled with one peak variable per investment period:

        elec_peak[p] >= sum_t gen[t,p,d,h] / COP[t,p,d,h] + pump draw   for all d,h

    and the objective carries elec_peak[p] * 1000 * fee, discounted over the
    period. Unlike a charge on installed capacity this lets the optimiser shave
    its peak -- running the heat pump at part load, or spreading its operation --
    to reduce the fee, which is what a real site would do.

    Called after the SAPHEA and revORC blocks so the hourly COPs are final; the
    coefficients are the numeric COP values at that point.
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

    def _peak_rule(mm, p_, d_, h_):
        draw = sum(mm.gen[t, p_, d_, h_] / float(pyo.value(mm.efficiency[t, p_, d_, h_]))
                   for t in elec_techs
                   if float(pyo.value(mm.efficiency[t, p_, d_, h_])) > 1e-9)
        if has_pump:
            draw = draw + k_pump * mm.orc_pump_extra[p_, d_, h_]
        return mm.elec_peak[p_] >= draw
    m.elec_peak_def = pyo.Constraint(m.PERIODS, m.REP_DAYS, m.HOURS, rule=_peak_rule)

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
        f"({len(m.PERIODS) * len(m.REP_DAYS) * 24} constraints)")


def _build_revorc(model, demand, resource, cfg):
    """
    Post-build: revORC dispatch, brine competition and ORC revenue.

    Topology (Kaufmann et al. ECM 2026 Fig. 2.1 — SERIES HP / PARALLEL ORC,
    confirmed 2026-07-02):
      ORC mode — PARALLEL: surplus brine at wellhead temperature feeds the ORC
        evaporator, competing with the direct HX for the high-grade slice.
      HP mode — SERIES: the evaporator sits DOWNSTREAM of the DHN heat exchanger
        on the same low-grade slice as the SAPHEA HTHP (T_woHP → T_inject_min).
        It shares the CI coupling ratio with the SAPHEA HTHP, but carries its own
        HTHP (the FLEXGEO HP surrogate was trained on wellhead-temperature source,
        95–115 °C in / 60–75 °C outlet floor, and would extrapolate at series
        conditions — open question with the FLEXGEO partner) and genuinely ADDS
        heat beyond the geothermal direct-use capacity, exactly like the HTHP.

    Per (period, d, h), with n_mod = cap_total[RevORC]/module_MW modules installed:
      orc_share[p,d,h]  continuous number of modules running in ORC mode
      hp_share[p,d,h]   continuous number of modules running in HP mode
      gen[RevORC,p,d,h] HP-mode heat output (regular tech → heat balance, elec cost)

      fleet:    orc_share + hp_share             ≤ n_mod  (machine mode exclusivity)
      HP cap:   gen[RevORC]                      ≤ hp_share·module_MW
      brine:    gen[Geo] − reinject + orc_share·q_in[d,h]
                ≤ q_geo_du_cap[d,h] − geo_delta_ci[d,h]·y_ci
                (ORC, parallel; reinjected must-run heat is available brine flow —
                 the geo min-load is a flow constraint, so the ORC ceiling is
                 effectively cap − net demand served, not cap − min-load. The
                 y_ci term removes the pinch-delta budget in CI mode: that heat
                 sits at ~65 °C, outside the ORC's wellhead inlet domain)
      CI gate:  gen[RevORC]                      ≤ M·y_ci  (series HP ⇒ CI mode, so
                the geo direct-HX cap correctly drops by the pinch delta)
      CI slice: gen[RevORC] + gen[HTHP]
                ≤ gen[Geo]·coupling_ratio[d,h] + M(1−y_ci)   (shared with the HTHP)

    Objective gains −Σ orc_share·P_net·sell_price (discounted like all other
    annual flows) plus a 1e-4 €/module-h anti-degeneracy term. HP-mode brine is
    already pumped for the DHN (series) → no extra pump charge; the ORC pump
    term below is unchanged.

    exclusive_with_hthp=1 (RevORC sheet): one horizon-wide binary y_revorc makes
    revORC and the SAPHEA HTHP mutually exclusive INVESTMENTS — per Kaufmann et
    al. ECM 2026, where RHP / HTHP / boiler are alternative peak-load systems.
    The optimizer still picks which one endogenously.

    brine_pump_el_per_MWth (RevORC sheet): the ORC surrogate's P_net excludes the
    geothermal brine pump, which the paper shows consumes >half the gross output.
    ORC brine draw beyond the already-pumped reinjected heat (m.reinject) incurs
    pump electricity at the hourly buy price:
        orc_pump_extra ≥ orc_share·q_in − reinject   (≥ 0)
        cost += k_pump · orc_pump_extra · elec_price

    dead_hours (RevORC sheet, default 1): FlexGeo dead-hour switching rule —
    fleet-level constraints orc_share[h] + hp_share[h−k] ≤ n_mod for
    k = 1..dead_hours (cyclic within the rep day, both directions) force
    modules to sit OFF between HP and ORC operation.

    Feasibility gates live in orc_surrogate_maps.py: ORC hours outside the
    surrogate training bounds (or below orc_amb_min_C) get q_in = P_net = 0.
    The maps' HP-surrogate outputs (hp_out/hp_cop, incl. the part-load sweep)
    feed the series HP mode directly (2026-07-30 series-trained bundle).
    """
    m = model.model
    tsa = model.tsa
    if REVORC_TECH not in model.technologies:
        logger.warning(f"'{REVORC_TECH}' not in model technologies — revORC skipped")
        return
    if "ambient_temp" not in demand.columns:
        raise ValueError(
            "revORC needs the 'ambient_temp' column in Munich_DH_data.xlsx "
            "(ERA5 profile; see orc_surrogate_maps.py)"
        )

    n_rep = tsa["n_rep_days"]
    years = sorted(model.investment_years)
    mod_MW = cfg["module_hp_MWth"]

    # Per-period surrogate maps: under per-period TSA each milestone has its own
    # day→rep mapping, hence its own rep ambient/supply/return profiles and its
    # own (d,h) maps. Deduped by TSA object identity so 'joint' mode evaluates
    # (and disk-caches) only once.
    maps_by_p, _seen = {}, {}
    for p_ in years:
        tsa_p = model.tsa_by_period[p_]
        if id(tsa_p) not in _seen:
            d2r = tsa_p["day_to_rep"]
            rep_amb = _rep_profile(demand["ambient_temp"].values, d2r, n_rep)
            rep_sup = _rep_profile(demand["supply_temp"].values, d2r, n_rep)
            rep_ret = _rep_profile(demand["return_temp"].values, d2r, n_rep)
            _seen[id(tsa_p)] = build_revorc_maps(rep_amb, rep_sup, rep_ret, resource, cfg)
        maps_by_p[p_] = _seen[id(tsa_p)]
    q_in_p  = {p_: maps_by_p[p_]["orc_q_in_MW"]  for p_ in years}
    p_net_p = {p_: maps_by_p[p_]["orc_p_net_MW"] for p_ in years}

    if HTHP_TECH not in model.technologies:
        raise ValueError(
            "Series revORC-HP needs the HTHP technology in the model (its SAPHEA "
            "hourly COP and CI coupling ratio are reused for the revORC HP mode)"
        )

    # ── HP mode is SERIES (Kaufmann Fig. 2.1): the revORC-HP evaporator sits
    #    DOWNSTREAM of the DHN heat exchanger on the low-grade slice
    #    (T_woHP → T_inject_min) — the same slice as the SAPHEA HTHP — so it
    #    genuinely ADDS heat beyond the geothermal direct-use capacity.
    #    The 2026-07-30 FLEXGEO HP bundle is trained on exactly this slice
    #    (brine in 55–75 °C, outlet floor 35–45 °C, sink supply 75–100 °C), so the
    #    machine's own hourly COP is used. Earlier runs had to substitute the
    #    SAPHEA HTHP COP because the previous bundle was trained on wellhead
    #    source (95–115 °C) and extrapolated badly here. ──
    hp_cop_p = {p_: maps_by_p[p_]["hp_cop"] for p_ in years}
    _cops = [hp_cop_p[years[0]][(d, h)] for d in range(n_rep) for h in range(24)]
    logger.info(
        f"revORC HP mode uses the series-trained surrogate COP "
        f"(rep-hour mean {np.mean(_cops):.2f}, range {min(_cops):.2f}–{max(_cops):.2f}); "
        f"SAPHEA HTHP COP is no longer substituted"
    )
    for p_ in years:
        for d in range(n_rep):
            for h in range(24):
                m.efficiency[REVORC_TECH, p_, d, h].set_value(hp_cop_p[p_][(d, h)])

    m.orc_share = pyo.Var(m.PERIODS, m.REP_DAYS, m.HOURS, domain=pyo.NonNegativeReals)
    m.hp_share = pyo.Var(m.PERIODS, m.REP_DAYS, m.HOURS, domain=pyo.NonNegativeReals)

    # One fleet, two modes: modules in ORC mode + modules in HP mode ≤ installed.
    def revorc_fleet_rule(m, p, d, h):
        return (
            m.orc_share[p, d, h] + m.hp_share[p, d, h]
            <= m.cap_total[REVORC_TECH, p] / mod_MW
        )
    m.revorc_fleet = pyo.Constraint(m.PERIODS, m.REP_DAYS, m.HOURS, rule=revorc_fleet_rule)

    # ── FlexGeo dead-hour switching rule: a module changing mode must sit OFF
    #    for `dead_hours` in between (RevORC sheet, default 1 h). Fleet-level
    #    linear form: modules in ORC at h + modules in HP at h−k ≤ fleet size
    #    for k = 1..dead_hours (and symmetrically), cyclic within the rep day —
    #    by pigeonhole no module can have been in the other mode inside the dead
    #    window. k = 0 is revorc_fleet itself. ──
    dead_hours = int(cfg.get("dead_hours", 1))
    if dead_hours > 0:
        m.revorc_dead = pyo.ConstraintList()
        for p in m.PERIODS:
            for d in m.REP_DAYS:
                fleet = m.cap_total[REVORC_TECH, p] / mod_MW
                for h in m.HOURS:
                    for k in range(1, dead_hours + 1):
                        hk = (h - k) % 24
                        m.revorc_dead.add(m.orc_share[p, d, h] + m.hp_share[p, d, hk] <= fleet)
                        m.revorc_dead.add(m.hp_share[p, d, h] + m.orc_share[p, d, hk] <= fleet)
        logger.info(
            f"revORC dead-hour switching rule: {dead_hours} h OFF required "
            f"between HP and ORC operation ({len(m.revorc_dead)} constraints)"
        )

    # HP-mode heat ≤ nominal module heat output × modules running as HP.
    def revorc_hp_cap_rule(m, p, d, h):
        return m.gen[REVORC_TECH, p, d, h] <= m.hp_share[p, d, h] * mod_MW
    m.revorc_hp_cap = pyo.Constraint(m.PERIODS, m.REP_DAYS, m.HOURS, rule=revorc_hp_cap_rule)

    # ORC (parallel): competes with the direct HX for the high-grade slice only.
    # Reinjected (dumped) must-run heat is hot brine flow the ORC may consume —
    # the geo min-load is a FLOW constraint (53 kg/s, well protection), not a
    # heat-use constraint, so the network's true claim is gen[Geo] − reinject
    # (Kaufmann Fig. 2.1(b): surplus brine feeds the ORC at wellhead temperature).
    # The series HP does NOT appear here: its evaporator sits behind the DHN HX
    # on the low-grade slice (CI coupling below), not on the wellhead brine.
    # In CI mode the high-grade band shrinks by the pinch delta (brine only
    # cooled to T_return+pinch before the HP evaporator) — without the
    # geo_delta_ci·y_ci term the ORC could draw "freed" budget that physically
    # sits at ~65 °C, far below its 95–115 °C inlet domain (phantom winter ORC).
    _has_reinject = hasattr(m, "reinject")

    def revorc_brine_rule(m, p, d, h):
        net_geo = m.gen[GEO_TECH, p, d, h]
        if _has_reinject:
            net_geo = net_geo - m.reinject[p, d, h]
        return (
            net_geo + m.orc_share[p, d, h] * q_in_p[p][(d, h)]
            <= m.q_geo_du_cap[p, d, h]
               - m.geo_delta_ci[p, d, h] * m.y_ci[p, d, h]
        )
    m.revorc_brine = pyo.Constraint(m.PERIODS, m.REP_DAYS, m.HOURS, rule=revorc_brine_rule)

    # ── Series-HP CI coupling (same low-grade slice as the SAPHEA HTHP) ──
    M_rev = cfg["max_modules"] * mod_MW
    _q_hthp_max_rev = float(model.technologies[HTHP_TECH]["max_output_MW"])

    # Series HP ⇒ CI mode: running the downstream evaporator drops the geo
    # direct-HX cap by the pinch delta (geo_mode_cap handles that via y_ci).
    def revorc_ci_gate_rule(m, p, d, h):
        return m.gen[REVORC_TECH, p, d, h] <= M_rev * m.y_ci[p, d, h]
    m.revorc_ci_gate = pyo.Constraint(m.PERIODS, m.REP_DAYS, m.HOURS, rule=revorc_ci_gate_rule)

    # revORC-HP and HTHP share the finite low-grade evaporator slice:
    # combined output ≤ gen[Geo] × coupling_ratio when CI is active.
    def revorc_ci_slice_rule(m, p, d, h):
        return (
            m.gen[REVORC_TECH, p, d, h] + m.gen[HTHP_TECH, p, d, h]
            <= m.gen[GEO_TECH, p, d, h] * m.hthp_coupling_ratio[p, d, h]
               + (M_rev + _q_hthp_max_rev) * (1 - m.y_ci[p, d, h])
        )
    m.revorc_ci_slice = pyo.Constraint(m.PERIODS, m.REP_DAYS, m.HOURS, rule=revorc_ci_slice_rule)

    # ── revORC ↔ HTHP investment exclusivity (Kaufmann: alternative peak techs) ──
    _excl = os.environ.get("REVORC_EXCLUSIVE")
    _excl = cfg.get("exclusive_with_hthp", 0) if _excl is None else float(_excl)
    if _excl and HTHP_TECH in model.technologies:
        hthp_existing = float(model.technologies[HTHP_TECH]["installed_capacity_MW"])
        if hthp_existing > 0:
            logger.warning(
                f"exclusive_with_hthp=1 but HTHP has existing capacity "
                f"({hthp_existing:.2f} MW) — y_revorc will be forced to 0 (HTHP kept)"
            )
        M_hthp = float(model.technologies[HTHP_TECH]["max_output_MW"])
        m.y_revorc = pyo.Var(domain=pyo.Binary)

        def revorc_excl_rev_rule(m, p):
            return m.cap_total[REVORC_TECH, p] <= M_rev * m.y_revorc
        m.revorc_excl_rev = pyo.Constraint(m.PERIODS, rule=revorc_excl_rev_rule)

        def revorc_excl_hthp_rule(m, p):
            return m.cap_total[HTHP_TECH, p] <= M_hthp * (1 - m.y_revorc)
        m.revorc_excl_hthp = pyo.Constraint(m.PERIODS, rule=revorc_excl_hthp_rule)
        logger.info(
            f"revORC↔HTHP exclusivity active: y_revorc=1 → revORC only (≤{M_rev:.1f} MW), "
            f"y_revorc=0 → HTHP only (≤{M_hthp:.2f} MW)"
        )

    # ── Brine-pump electricity for ORC draw beyond already-pumped reinjected heat ──
    k_pump = float(cfg.get("brine_pump_el_per_MWth", 0.0))
    if k_pump > 0:
        m.orc_pump_extra = pyo.Var(m.PERIODS, m.REP_DAYS, m.HOURS, domain=pyo.NonNegativeReals)
        has_reinject = hasattr(m, "reinject")

        # Series-HP brine is already pumped for the DHN → only the ORC draw
        # (beyond reinjected, already-pumped heat) incurs pump electricity.
        def orc_pump_extra_rule(m, p, d, h):
            draw = m.orc_share[p, d, h] * q_in_p[p][(d, h)]
            if has_reinject:
                return m.orc_pump_extra[p, d, h] >= draw - m.reinject[p, d, h]
            return m.orc_pump_extra[p, d, h] >= draw
        m.orc_pump_def = pyo.Constraint(m.PERIODS, m.REP_DAYS, m.HOURS, rule=orc_pump_extra_rule)
        logger.info(
            f"revORC brine-pump cost active: {k_pump:.3f} MWh_el/MWh_th on ORC draw "
            f"{'beyond reinjected heat' if has_reinject else '(full draw — no reinject var)'}"
        )

    # ── ORC revenue in the objective (mirror of the rd_model discounting) ──
    _EPS = 1e-4  # €/module-h anti-degeneracy
    vom_orc = float(cfg.get("opex_var_orc_eur_per_MWh_el", 0.0))
    if vom_orc:
        logger.info(f"revORC ORC-mode variable O&M: {vom_orc:.2f} EUR/MWh_el on generation")
    base_yr = min(model.investment_years)
    plist = sorted(model.investment_years)
    extra = 0
    for i, p in enumerate(plist):
        years_in_period = plist[i + 1] - p if i + 1 < len(plist) else 5
        r = model.discount_rate_by_period[p]
        pv_annual = sum(1.0 / (1.0 + r) ** (p + y - base_yr) for y in range(years_in_period))
        for d in range(n_rep):
            w = float(pyo.value(m.day_weight[p, d]))
            for h in range(24):
                sell = float(pyo.value(m.electricity_sell_price[p, d, h]))
                # ORC-mode variable O&M is charged on generated electricity, so
                # it lowers the net revenue per MWh_el produced.
                coeff = (_EPS - p_net_p[p][(d, h)] * (sell - vom_orc)) * w * pv_annual
                extra = extra + m.orc_share[p, d, h] * coeff
                if k_pump > 0:
                    # floor the price at zero: orc_pump_extra is bounded from below
                    # only, so a negative purchase price would make pumping a
                    # reward and the problem unbounded. Extra pumping must never
                    # earn money.
                    buy = max(float(pyo.value(m.electricity_price[p, d, h])), 0.0)
                    extra = extra + m.orc_pump_extra[p, d, h] * (k_pump * buy * w * pv_annual)
    m.total_cost.expr = m.total_cost.expr + extra

    model._revorc = {"maps_by_p": maps_by_p, "cfg": cfg}
    _cops = np.array([
        pyo.value(m.efficiency[REVORC_TECH, p_, d, h])
        for p_ in years for d in range(n_rep) for h in range(24)
    ])
    _pn = [v for p_ in years for v in p_net_p[p_].values()]
    logger.info(
        f"revORC built (parallel ORC + SERIES HP, series-trained COP) — ≤ {cfg['max_modules']:.0f} "
        f"modules × {mod_MW:.1f} MW_th, ORC P_net {min(_pn)*1e3:.0f}–"
        f"{max(_pn)*1e3:.0f} kW/module, HP COP (FLEXGEO series bundle) "
        f"{_cops.min():.2f}–{_cops.max():.2f} (mean {_cops.mean():.2f}), "
        f"HP ≤ {mod_MW:.1f} MW/module on the shared CI slice"
    )


def _report_revorc(model, results_dir):
    """Post-solve: per-period revORC summary (modules, ORC MWh, revenue, HP heat)."""
    if not hasattr(model, "_revorc"):
        return
    m = model.model
    maps_by_p, cfg = model._revorc["maps_by_p"], model._revorc["cfg"]
    mod_MW = cfg["module_hp_MWth"]
    n_rep = model.tsa["n_rep_days"]

    if hasattr(m, "y_revorc"):
        pick = "revORC" if pyo.value(m.y_revorc) > 0.5 else "HTHP"
        logger.info(f"  revORC↔HTHP exclusivity: optimizer picked {pick} (y_revorc={pyo.value(m.y_revorc):.0f})")

    rows = []
    for p in sorted(model.investment_years):
        cap = float(pyo.value(m.cap_total[REVORC_TECH, p]))
        hthp_cap = (float(pyo.value(m.cap_total[HTHP_TECH, p]))
                    if HTHP_TECH in model.technologies else 0.0)
        q_in, p_net = maps_by_p[p]["orc_q_in_MW"], maps_by_p[p]["orc_p_net_MW"]
        orc_mwh = orc_rev = orc_qin = hp_mwh = pump_mwh = 0.0
        for d in range(n_rep):
            w = float(pyo.value(m.day_weight[p, d]))
            for h in range(24):
                share = float(pyo.value(m.orc_share[p, d, h]))
                sell = float(pyo.value(m.electricity_sell_price[p, d, h]))
                orc_mwh += share * p_net[(d, h)] * w
                orc_qin += share * q_in[(d, h)] * w
                orc_rev += share * p_net[(d, h)] * sell * w
                hp_mwh += float(pyo.value(m.gen[REVORC_TECH, p, d, h])) * w
                if hasattr(m, "orc_pump_extra"):
                    pump_mwh += (float(pyo.value(m.orc_pump_extra[p, d, h]))
                                 * float(cfg.get("brine_pump_el_per_MWth", 0.0)) * w)
        rows.append({
            "period": p,
            "modules": round(cap / mod_MW, 2),
            "cap_MWth": round(cap, 2),
            "hthp_cap_MWth": round(hthp_cap, 2),
            "orc_elec_MWh_a": round(orc_mwh, 1),
            "orc_brine_heat_MWh_a": round(orc_qin, 1),
            "orc_revenue_EUR_a": round(orc_rev),
            "brine_pump_el_MWh_a": round(pump_mwh, 1),
            "hp_heat_MWh_a": round(hp_mwh, 1),
        })
        logger.info(
            f"  revORC {p}: {rows[-1]['modules']:.1f} modules ({cap:.1f} MW_th) | "
            f"HTHP {hthp_cap:.2f} MW | ORC {orc_mwh:,.0f} MWh_el/a ({orc_rev:,.0f} €/a, "
            f"pump {pump_mwh:,.0f} MWh_el/a) | HP {hp_mwh:,.0f} MWh_th/a"
        )
    df = pd.DataFrame(rows)
    df.to_csv(Path(results_dir) / "revorc_summary.csv", index=False)
    return df


def save_revorc_price_analysis(model, results_dir):
    """Analyse revORC operation against the hourly price signal.

    Produces per analysed period (first & last investment year):
      revorc_price_analysis_<year>.png — 4 panels:
        (a) sell price vs ORC output scatter (dispatch threshold visible)
        (b) price duration curve, ORC operating hours highlighted
        (c) monthly ORC generation / HP heat vs monthly mean sell price
        (d) mean diurnal profile (ORC output & price), winter vs summer
      revorc_hourly_<year>.csv — reconstructed 8760-h operation table
      revorc_price_stats.csv — operating hours, realized price, value factor,
        dispatch threshold, negative-price operation (should be 0)

    Scatter/threshold use the rep-day prices the OPTIMIZER saw
    (m.electricity_sell_price); monthly/diurnal panels use the 8760-h mapping.
    """
    if not hasattr(model, "_revorc"):
        return
    m = model.model
    maps_by_p = model._revorc["maps_by_p"]
    n_rep = model.tsa["n_rep_days"]
    periods = sorted(model.investment_years)
    results_dir = Path(results_dir)

    month_starts = np.cumsum([0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]) * 24
    month_labels = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']

    stats_rows = []
    plot_jobs = []
    for p in periods:
        # ── rep-space operation & price (what the optimizer saw) ──
        p_net = maps_by_p[p]["orc_p_net_MW"]
        day_to_rep = model.tsa_by_period[p]["day_to_rep"]
        share_r = np.zeros((n_rep, 24))
        price_r = np.zeros((n_rep, 24))
        for d in range(n_rep):
            for h in range(24):
                share_r[d, h] = float(pyo.value(m.orc_share[p, d, h]))
                price_r[d, h] = float(pyo.value(m.electricity_sell_price[p, d, h]))
        pnet_r = np.array([[p_net[(d, h)] for h in range(24)] for d in range(n_rep)])
        w = np.array([float(pyo.value(m.day_weight[p, d])) for d in range(n_rep)])

        orc_el_r = share_r * pnet_r                      # MW_el per rep slot
        run = orc_el_r > 1e-6
        hours_run = float((run * w[:, None]).sum())
        mwh = float((orc_el_r * w[:, None]).sum())
        rev = float((orc_el_r * price_r * w[:, None]).sum())
        realized = rev / mwh if mwh > 0 else np.nan
        mean_price = float(np.average(price_r, weights=np.repeat(w[:, None], 24, 1)))
        thr_run = float(price_r[run].min()) if run.any() else np.nan
        idle_prices = price_r[~run & (pnet_r > 1e-6)]
        thr_idle = float(idle_prices.max()) if idle_prices.size else np.nan
        neg_run_hours = float(((run & (price_r < 0)) * w[:, None]).sum())

        stats_rows.append({
            "period": p, "orc_hours": round(hours_run), "orc_MWh": round(mwh, 1),
            "realized_price_EUR_MWh": round(realized, 1),
            "mean_sell_price_EUR_MWh": round(mean_price, 1),
            "value_factor": round(realized / mean_price, 3) if mean_price else np.nan,
            "min_price_running": round(thr_run, 1),
            "max_price_idle": round(thr_idle, 1),
            "neg_price_run_hours": round(neg_run_hours),
        })

        # ── 8760-h reconstruction (for seasonal/diurnal views + CSV) ──
        # Exported for EVERY milestone: the mode analysis is per-period.
        # The per-period figures below stay on the first and last only.
        sell_col = f"sell_electricity_price_{p}"
        sell_8760 = (model.demand_data[sell_col].values
                     if sell_col in model.demand_data.columns else None)
        orc_el = np.zeros(8760); hp_th = np.zeros(8760); reinj = np.zeros(8760)
        for t in range(8760):
            d, h = day_to_rep[t // 24], t % 24
            orc_el[t] = orc_el_r[d, h]
            hp_th[t] = float(pyo.value(m.gen[REVORC_TECH, p, d, h]))
            if hasattr(m, "reinject"):
                reinj[t] = float(pyo.value(m.reinject[p, d, h]))
        pd.DataFrame({
            "hour": np.arange(8760),
            "sell_price_EUR_MWh": sell_8760 if sell_8760 is not None else np.nan,
            "orc_el_MW": orc_el, "hp_heat_MW": hp_th, "reinject_MW": reinj,
        }).to_csv(results_dir / f"revorc_hourly_{p}.csv", index=False)

        if p not in (periods[0], periods[-1]):
            continue
        plot_jobs.append({
            "p": p, "price_r": price_r, "run": run, "orc_el_r": orc_el_r,
            "thr_run": thr_run, "w": w, "sell_8760": sell_8760,
            "orc_el": orc_el, "hp_th": hp_th,
        })

    # ── Render period figures with SHARED axis limits (comparability) ──
    c_orc = get_color(REVORC_TECH)
    if plot_jobs:
        price_max = max(float(j["price_r"].max()) for j in plot_jobs)
        price_min = min(0.0, min(float(j["price_r"].min()) for j in plot_jobs))
        out_max = max(float(j["orc_el_r"].max()) for j in plot_jobs)
        mo_max = 0.0
        for j in plot_jobs:
            for i in range(12):
                s, e = month_starts[i], month_starts[i + 1]
                mo_max = max(mo_max, j["orc_el"][s:e].sum(), j["hp_th"][s:e].sum())
        price_lim = (price_min - 20, price_max * 1.05)
        out_lim = (-0.02, out_max * 1.1)
        mo_lim = (0, mo_max * 1.1)

    for j in plot_jobs:
        p, price_r, run = j["p"], j["price_r"], j["run"]
        orc_el_r, thr_run, w = j["orc_el_r"], j["thr_run"], j["w"]
        sell_8760, orc_el, hp_th = j["sell_8760"], j["orc_el"], j["hp_th"]

        fig, axes = plt.subplots(2, 2, figsize=(20, 13))

        ax = axes[0, 0]   # (a) scatter, rep-space
        ax.scatter(price_r[~run].ravel(), orc_el_r[~run].ravel(), s=18, alpha=0.4,
                   color='#95A5A6', label='idle')
        ax.scatter(price_r[run].ravel(), orc_el_r[run].ravel(), s=18, alpha=0.6,
                   color=c_orc, label='ORC running')
        if not np.isnan(thr_run):
            ax.axvline(thr_run, ls='--', color='k', lw=1.2)
            ax.annotate(f'{thr_run:.0f} €/MWh', (thr_run, out_lim[1] * 0.85),
                        fontsize=15, rotation=90, va='top', ha='right')
        ax.set_xlim(*price_lim); ax.set_ylim(*out_lim)
        ax.set_xlabel('Sell price (€/MWh)'); ax.set_ylabel('ORC output (MW$_{el}$)')
        ax.set_title(f'(a) Dispatch vs price — {p}'); ax.legend()

        ax = axes[0, 1]   # (b) price duration curve, rep-space expanded by weights
        price_flat, run_flat, wt_flat = price_r.ravel(), run.ravel(), np.repeat(w, 24)
        order = np.argsort(price_flat)[::-1]
        cum = np.cumsum(wt_flat[order])
        ax.fill_between(cum, price_flat[order], step='post', alpha=0.15, color='#7F8C8D')
        ax.plot(cum, price_flat[order], color='#2C3E50', lw=1.5, label='price duration')
        run_mask = run_flat[order]
        ax.scatter(cum[run_mask], price_flat[order][run_mask], s=8, color=c_orc,
                   label='ORC running', zorder=3)
        ax.axhline(0, color='k', lw=0.8)
        ax.set_ylim(*price_lim)
        ax.set_xlabel('Hours per year'); ax.set_ylabel('Sell price (€/MWh)')
        ax.set_title(f'(b) Price duration & ORC operation — {p}'); ax.legend()

        ax = axes[1, 0]   # (c) monthly energy vs price
        mo_orc = [orc_el[month_starts[i]:month_starts[i + 1]].sum() for i in range(12)]
        mo_hp = [hp_th[month_starts[i]:month_starts[i + 1]].sum() for i in range(12)]
        x = np.arange(12)
        ax.bar(x - 0.2, mo_orc, 0.4, color=c_orc, label='ORC (MWh$_{el}$)')
        ax.bar(x + 0.2, mo_hp, 0.4, color='#8E44AD', label='HP heat (MWh$_{th}$)')
        ax.set_xticks(x); ax.set_xticklabels(month_labels)
        ax.set_ylim(*mo_lim)
        ax.set_ylabel('Monthly energy (MWh)')
        if sell_8760 is not None:
            ax2 = ax.twinx()
            mo_pr = [np.mean(sell_8760[month_starts[i]:month_starts[i + 1]]) for i in range(12)]
            ax2.plot(x, mo_pr, 'o-', color='#2C3E50', label='mean sell price')
            ax2.set_ylim(*price_lim)
            ax2.set_ylabel('€/MWh')
            ax2.spines['right'].set_visible(True)
        ax.set_title(f'(c) Seasonal operation — {p}'); ax.legend(loc='upper center')

        ax = axes[1, 1]   # (d) diurnal profiles winter vs summer
        win = np.r_[0:month_starts[3], month_starts[10]:8760]
        summ = np.r_[month_starts[5]:month_starts[9]]
        for sel, lab, ls in [(win, 'winter (Nov–Mar)', '-'), (summ, 'summer (Jun–Sep)', '--')]:
            prof = [orc_el[sel][(sel % 24) == h].mean() for h in range(24)]
            ax.plot(range(24), prof, ls, color=c_orc, lw=2.5, label=f'ORC {lab}')
        ax.set_ylim(*out_lim)
        ax.set_xlabel('Hour of day'); ax.set_ylabel('Mean ORC output (MW$_{el}$)')
        if sell_8760 is not None:
            ax2 = ax.twinx()
            for sel, lab, ls in [(win, 'winter', '-'), (summ, 'summer', '--')]:
                prof = [sell_8760[sel][(sel % 24) == h].mean() for h in range(24)]
                ax2.plot(range(24), prof, ls, color='#2C3E50', lw=1.2, alpha=0.7)
            ax2.set_ylim(*price_lim)
            ax2.set_ylabel('Sell price (€/MWh)')
            ax2.spines['right'].set_visible(True)
        ax.set_title(f'(d) Diurnal ORC dispatch vs price — {p}'); ax.legend(fontsize=15)

        fig.tight_layout()
        fig.savefig(results_dir / f'revorc_price_analysis_{p}.png', dpi=150,
                    bbox_inches='tight')
        plt.close(fig)

    df = pd.DataFrame(stats_rows)
    df.to_csv(results_dir / "revorc_price_stats.csv", index=False)
    for r in stats_rows:
        logger.info(
            f"  revORC price response {r['period']}: {r['orc_hours']} h/a, "
            f"realized {r['realized_price_EUR_MWh']} vs mean {r['mean_sell_price_EUR_MWh']} €/MWh "
            f"(VF {r['value_factor']}), threshold ≈ {r['min_price_running']} €/MWh, "
            f"neg-price operation {r['neg_price_run_hours']} h"
        )
    return df


# ── SAPHEA mode analysis ───────────────────────────────────────────────────────
_MODE_COLORS = {'DU': '#2ca02c', 'CI': '#ff7f0e', 'TI': '#d62728'}


def _reconstruct_mode_8760(model, year):
    """Return (du, ci, ti) integer arrays of length 8760 based on actual dispatch.

    CI is derived from HTHP generation > 0 (not y_ci) to avoid the MILP artefact
    where the solver sets y_ci=1 for idle hours because Big-M makes it cost-free.
    TI remains binary-based because y_ti is physically forced by temperature.
    """
    m   = model.model
    d2r = model.tsa_by_period[year]['day_to_rep']

    # TI: physically forced by temperature — binary is reliable
    if hasattr(m, 'y_ti'):
        ti = np.array([round(pyo.value(m.y_ti[year, d2r[h // 24], h % 24]))
                       for h in range(8760)], dtype=int)
    else:
        ti = np.zeros(8760, int)

    # CI: HTHP is actually generating heat (dispatch-based, avoids solver artefact)
    if HTHP_TECH in model.technologies:
        hthp_gen = np.array([
            pyo.value(m.gen[HTHP_TECH, year, d2r[h // 24], h % 24])
            for h in range(8760)
        ])
    else:
        hthp_gen = np.zeros(8760)

    # The revORC's series heat pump ALSO runs in capacity-increase mode (its CI
    # gate is gen[RevORC] <= M*y_ci), so CI must count it too. Deriving CI from
    # the HTHP alone reported 0 CI hours in every scenario where the HTHP is not
    # investable — including runs that delivered several GWh of revORC HP heat.
    if REVORC_TECH in model.technologies:
        rev_gen = np.array([
            pyo.value(m.gen[REVORC_TECH, year, d2r[h // 24], h % 24])
            for h in range(8760)
        ])
    else:
        rev_gen = np.zeros(8760)

    ci = (((hthp_gen > 0.01) | (rev_gen > 0.01)) & (ti == 0)).astype(int)

    return 1 - ci - ti, ci, ti


def _compute_mode_stats(model):
    """Return DataFrame with per-year × per-mode statistics (hours, gen, COP, demand)."""
    m           = model.model
    tsa         = model.tsa
    # NB: per-year demand is selected inside the year loop below; this is only a
    # fallback for cases with no corrected_demand_<year> column.
    demand_8760 = model.demand_data['heat_demand'].values
    supply_8760 = (model.demand_data['supply_temp'].values
                   if 'supply_temp' in model.demand_data else np.full(8760, np.nan))

    records = []
    for year in INVESTMENT_YEARS:
        demand_8760 = _demand_for(model, year)
        du, ci_arr, ti = _reconstruct_mode_8760(model, year)
        d2r = model.tsa_by_period[year]['day_to_rep']
        geo_8760  = np.array([
            pyo.value(m.gen[GEO_TECH, year, d2r[h // 24], h % 24])
            for h in range(8760)
        ])
        has_hthp  = HTHP_TECH in model.technologies
        hthp_8760 = np.array([
            pyo.value(m.gen[HTHP_TECH, year, d2r[h // 24], h % 24])
            for h in range(8760)
        ]) if has_hthp else np.zeros(8760)
        cop_8760  = np.array([
            pyo.value(m.efficiency[HTHP_TECH, year, d2r[h // 24], h % 24])
            for h in range(8760)
        ]) if has_hthp else np.ones(8760)

        for mode_name, mask in [('DU', du.astype(bool)), ('CI', ci_arr.astype(bool)), ('TI', ti.astype(bool))]:
            n = int(mask.sum())
            records.append({
                'Year':              year,
                'Mode':              mode_name,
                'Hours_hyr':         n,
                'Hours_pct':         round(n / 8760 * 100, 1),
                'Geo_gen_MWh':       round(float(geo_8760[mask].sum()),  0) if n else 0,
                'HTHP_gen_MWh':      round(float(hthp_8760[mask].sum()), 0) if n else 0,
                'Total_heat_MWh':    round(float((geo_8760 + hthp_8760)[mask].sum()), 0) if n else 0,
                'Avg_demand_MW':     round(float(demand_8760[mask].mean()), 2) if n else 0,
                'Peak_demand_MW':    round(float(demand_8760[mask].max()),  2) if n else 0,
                'Avg_supply_temp_C': round(float(np.nanmean(supply_8760[mask])), 1) if n else 0,
                'Avg_COP':           round(float(cop_8760[mask].mean()), 3) if (n and mode_name != 'DU') else 0,
                'Avg_geo_MW':        round(float(geo_8760[mask].mean()),  2) if n else 0,
                'Avg_HTHP_MW':       round(float(hthp_8760[mask].mean()), 2) if n else 0,
            })
    return pd.DataFrame(records)


def _compute_economics_table(model):
    """
    Per-year table explaining why HTHP becomes more prominent for peak over time.

    Compares marginal variable cost of covering one extra MWh of peak heat via:
      - DU + Gas Boiler  (geo at 17.4 MW, boiler covers demand gap)
      - CI + HTHP        (geo at 14 MW,   HTHP covers demand gap using brine)

    The breakeven demand is the level above which CI+HTHP is cheaper than DU+boiler.
    """
    m           = model.model
    tsa         = model.tsa
    heat_8760   = model.demand_data['heat_demand'].values
    BOILER      = 'Gas Boiler'

    # Design capacities derived from the brine resource (set in _build_saphea_coupling)
    design     = model._saphea_design
    Q_geo_du   = design['Q_geo_du']
    Q_geo_ci   = design['Q_geo_ci']
    geo_delta  = Q_geo_du - Q_geo_ci

    var_geo  = float(pyo.value(m.opex_var[GEO_TECH]))
    var_hthp = float(pyo.value(m.opex_var[HTHP_TECH]))
    var_boil = float(pyo.value(m.opex_var[BOILER]))
    eta_boil = float(pyo.value(m.efficiency[BOILER, INVESTMENT_YEARS[0], 0, 0]))

    rows = []
    for year in INVESTMENT_YEARS:
        d2r = model.tsa_by_period[year]['day_to_rep']
        cop_8760 = np.array([
            pyo.value(m.efficiency[HTHP_TECH, year, d2r[h // 24], h % 24])
            for h in range(8760)
        ])
        el_price  = float(pyo.value(m.electricity_price[year, 0, 0]))
        gas_price = float(pyo.value(m.fuel_price[BOILER, year]))

        du, ci, ti    = _reconstruct_mode_8760(model, year)
        ci_mask       = ci.astype(bool)
        avg_cop       = float(cop_8760[ci_mask].mean()) if ci_mask.sum() > 0 else float(cop_8760.mean())

        # Variable cost of delivering 1 MWh of heat
        c_hthp = el_price / avg_cop + var_hthp   # €/MWh_heat via HTHP
        c_boil = gas_price / eta_boil + var_boil  # €/MWh_heat via boiler

        # Breakeven demand: CI+HTHP cheaper than DU+boiler above this level
        # From: Q_ci*var_geo + (D-Q_ci)*c_hthp = Q_du*var_geo + (D-Q_du)*c_boil
        denom = c_boil - c_hthp
        if abs(denom) > 1e-6:
            d_be = (-geo_delta * var_geo + Q_geo_du * c_boil - Q_geo_ci * c_hthp) / denom
        else:
            d_be = float('inf')

        boiler_cap  = float(pyo.value(m.cap_total[BOILER, year]))
        du_max      = Q_geo_du + boiler_cap
        ci_total_max = Q_geo_ci + model.technologies[HTHP_TECH]['max_output_MW']

        forced_h = int((heat_8760 > du_max).sum())
        econ_h   = int((heat_8760 > max(d_be, Q_geo_du)).sum()) if d_be < 30 else 0
        actual_h = int(ci_mask.sum())

        hthp_gen = sum(
            pyo.value(m.gen[HTHP_TECH, year, d2r[h // 24], h % 24])
            for h in range(8760)
        )

        rows.append({
            'Year':                      year,
            'Electricity price (€/MWh)': round(el_price, 0),
            'Gas price (€/MWh)':         round(gas_price, 1),
            'Avg COP (CI hours)':         round(avg_cop, 2),
            'HTHP var cost (€/MWh_th)':  round(c_hthp, 1),
            'Boiler var cost (€/MWh_th)': round(c_boil, 1),
            'HTHP saving vs boiler (€/MWh)': round(c_boil - c_hthp, 1),
            'DU+boiler max (MW)':         round(du_max, 1),
            'CI breakeven demand (MW)':   round(d_be, 1) if d_be < 40 else '>peak',
            'Hours demand>breakeven':     econ_h,
            'Hours forced CI (demand>DU+boiler cap)': forced_h,
            'CI hours (model)':           actual_h,
            'HTHP generation (MWh/yr)':  round(hthp_gen, 0),
        })
    return pd.DataFrame(rows)


def save_mode_plots(model, results_dir):
    """Per-year mode timeline + demand duration curve, and a cross-year summary."""
    from matplotlib.patches import Patch

    _MT  = [0, 744, 1416, 2160, 2880, 3624, 4344, 5088, 5832, 6552, 7296, 8016]
    _ML  = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    hours  = np.arange(8760)
    m      = model.model
    tsa    = model.tsa

    design   = model._saphea_design
    Q_geo_du = design['Q_geo_du']
    Q_geo_ci = design['Q_geo_ci']
    hthp_max = model.technologies.get(HTHP_TECH, {}).get('max_output_MW', 0)
    ci_max   = Q_geo_ci + hthp_max

    # ── Per-year: dispatch + mode strip + duration curve ──────────────────────
    for year in INVESTMENT_YEARS:
        demand = _demand_for(model, year)
        du, ci_arr, ti = _reconstruct_mode_8760(model, year)
        mode_str = np.where(ti == 1, 'TI', np.where(ci_arr == 1, 'CI', 'DU'))

        d2r_y   = model.tsa_by_period[year]['day_to_rep']
        techs   = [t for t in model.technologies if pyo.value(m.cap_total[t, year]) > 1e-2]
        gen_arr = {t: np.array([
            pyo.value(m.gen[t, year, d2r_y[h // 24], h % 24])
            for h in range(8760)
        ]) for t in techs}

        fig = plt.figure(figsize=(20, 13), constrained_layout=True)
        gs  = fig.add_gridspec(3, 1, height_ratios=[5, 0.8, 3.5], hspace=0.07)
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])
        ax3 = fig.add_subplot(gs[2])
        fig.suptitle(f'SAPHEA Mode Analysis — Munich {year}', fontsize=16, fontweight='bold')

        # Dispatch stackplot
        ax1.stackplot(hours, [gen_arr[t] for t in techs],
                      labels=techs, colors=[get_color(t) for t in techs], alpha=0.85)
        ax1.plot(hours, demand, 'k--', lw=1.5, label='Heat Demand', zorder=10)
        ax1.axhline(Q_geo_du, color='#2ca02c', ls=':', lw=1.5, alpha=0.8,
                    label=f'DU geo cap ({Q_geo_du:.1f} MW)')
        ax1.axhline(ci_max, color='#ff7f0e', ls=':', lw=1.5, alpha=0.8,
                    label=f'CI total cap ({ci_max:.1f} MW)')
        ax1.set_ylabel('Thermal Power [MW]')
        ax1.set_xlim(0, 8760); ax1.set_xticks(_MT); ax1.set_xticklabels(['']*12)
        ax1.grid(alpha=0.2)
        ax1.legend(loc='upper right', ncol=4, fontsize=10, framealpha=0.9)

        # Mode color strip (contiguous run-length encoding for efficiency)
        prev, start = mode_str[0], 0
        for h in range(1, 8761):
            cur = mode_str[h] if h < 8760 else None
            if cur != prev:
                ax2.axvspan(start, h, color=_MODE_COLORS[prev], alpha=1.0, lw=0)
                start, prev = h, cur
        ax2.legend(handles=[
            Patch(color=_MODE_COLORS['DU'], label=f"DU  {du.sum():4d} h  ({du.sum()/87.6:.1f}%)"),
            Patch(color=_MODE_COLORS['CI'], label=f"CI  {ci_arr.sum():4d} h  ({ci_arr.sum()/87.6:.1f}%)"),
            Patch(color=_MODE_COLORS['TI'], label=f"TI  {ti.sum():4d} h  ({ti.sum()/87.6:.1f}%)"),
        ], loc='upper right', ncol=3, fontsize=10, framealpha=0.9)
        ax2.set_xlim(0, 8760); ax2.set_xticks(_MT); ax2.set_xticklabels(['']*12)
        ax2.set_yticks([]); ax2.set_ylabel('Mode', fontsize=11)

        # Demand duration curve coloured by mode
        sort_idx = np.argsort(demand)[::-1]
        s_dem    = demand[sort_idx]
        s_mode   = mode_str[sort_idx]
        d_hrs    = np.arange(8760)
        for key in ['DU', 'CI', 'TI']:
            mask = s_mode == key
            if mask.sum():
                ax3.scatter(d_hrs[mask], s_dem[mask], c=_MODE_COLORS[key],
                            s=2, alpha=0.8, label=f'{key} ({mask.sum()} h)', rasterized=True)
        ax3.axhline(Q_geo_du, color='#2ca02c', ls=':', lw=1.5,
                    label=f'DU geo cap {Q_geo_du:.1f} MW')
        ax3.axhline(ci_max, color='#ff7f0e', ls=':', lw=1.5,
                    label=f'CI total cap {ci_max:.1f} MW')
        ax3.set_xlabel('Hours (sorted by demand, descending)')
        ax3.set_ylabel('Heat Demand [MW]')
        ax3.set_title('Demand Duration Curve — colour = operating mode')
        ax3.set_xlim(0, 8760); ax3.grid(alpha=0.2)
        ax3.legend(loc='upper right', ncol=3, fontsize=10)
        ax3.set_xticks(_MT); ax3.set_xticklabels(_ML)

        fig.savefig(results_dir / f'mode_analysis_{year}.png', dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"  Mode analysis plot saved: {year}")

    # ── Cross-year summary bar charts ─────────────────────────────────────────
    df_st = _compute_mode_stats(model)
    years = list(INVESTMENT_YEARS)
    x     = np.arange(len(years))

    fig, axes = plt.subplots(1, 3, figsize=(21, 7))

    # Panel 1: hours stacked by mode
    ax = axes[0]
    bot = np.zeros(len(years))
    for mo in ['DU', 'CI', 'TI']:
        vals = np.array([
            df_st[(df_st.Year == y) & (df_st.Mode == mo)]['Hours_hyr'].values[0]
            for y in years
        ], float)
        ax.bar(x, vals, bottom=bot, color=_MODE_COLORS[mo], label=mo, width=0.55,
               edgecolor='white', lw=0.5)
        bot += vals
    ax.set_xticks(x); ax.set_xticklabels(years)
    ax.set_ylabel('Operating hours by mode [h/yr]')
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False); ax.grid(axis='y', alpha=0.3)

    # Panel 2: heat generation stacked by mode
    ax = axes[1]
    bot = np.zeros(len(years))
    for mo in ['DU', 'CI', 'TI']:
        vals = np.array([
            df_st[(df_st.Year == y) & (df_st.Mode == mo)]['Total_heat_MWh'].values[0]
            for y in years
        ], float)
        ax.bar(x, vals, bottom=bot, color=_MODE_COLORS[mo], label=mo, width=0.55,
               edgecolor='white', lw=0.5)
        bot += vals
    ax.set_xticks(x); ax.set_xticklabels(years)
    ax.set_ylabel('Heat generation by mode [MWh/yr]')
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False); ax.grid(axis='y', alpha=0.3)

    # Panel 3: CI mode Geo vs HTHP split
    ax = axes[2]
    ci_geo  = np.array([
        df_st[(df_st.Year == y) & (df_st.Mode == 'CI')]['Geo_gen_MWh'].values[0]
        for y in years
    ], float)
    ci_hthp = np.array([
        df_st[(df_st.Year == y) & (df_st.Mode == 'CI')]['HTHP_gen_MWh'].values[0]
        for y in years
    ], float)
    ax.bar(x, ci_geo,  color='#1f77b4', label='Geo in CI',  width=0.55, edgecolor='white', lw=0.5)
    ax.bar(x, ci_hthp, bottom=ci_geo, color='#ff7f0e', label='HTHP in CI', width=0.55,
           edgecolor='white', lw=0.5)
    ax.set_xticks(x); ax.set_xticklabels(years)
    ax.set_ylabel('CI mode — Geo vs HTHP [MWh/yr]')
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False); ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    fig.savefig(results_dir / 'mode_summary.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("  Mode summary plot saved")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    root = _CASE_DIR
    ts   = datetime.now().strftime('%Y%m%d_%H%M%S')
    # Scenario selector (env SCENARIO): S1 = no storage, S2 = HT-BTES only,
    # S3 = pit/PTES only. Controls BTES_BUILD_PERIOD and whether the Heat-Storage
    # (PTES) row is kept, so all three are reproducible from one command.
    global BTES_BUILD_PERIOD
    SCENARIO = os.environ.get('SCENARIO', 'S3').upper()
    # Injection-temperature sensitivity (env T_INJECT_MIN, °C): overrides the
    # Geothermal Resource sheet's T_inject_min. Lower T → larger HP evaporator
    # range / higher HTHP heat ceiling (slightly lower COP). Folder is tagged _T<deg>.
    T_INJECT_OVERRIDE = os.environ.get('T_INJECT_MIN')
    _tsuf = f"_T{int(float(T_INJECT_OVERRIDE))}" if T_INJECT_OVERRIDE else ""
    # CO2-price sensitivity (env CO2_MULT): scales the carbon-price trajectory.
    # Tests whether a higher CO2 price makes the electric HTHP economic vs gas.
    CO2_MULT = float(os.environ.get('CO2_MULT', '1') or '1')
    _csuf = f"_C{CO2_MULT:g}" if CO2_MULT != 1.0 else ""
    # Gas- and electricity-price sensitivity (env GAS_PRICE / ELEC_PRICE, EUR/MWh):
    # replace the whole trajectory by a flat level, so the (gas, electricity) plane
    # can be swept the way Kaufmann et al. do for a static plant design. The
    # electricity level scales buy and sell columns alike, keeping the sales-to-
    # purchase ratio at 1.0 and preserving the hourly shape of the sell price.
    # Flat EUA level (env CO2_PRICE, EUR/tCO2): holds the carbon price constant
    # over the horizon instead of the rising EU-ETS trajectory, which is the
    # assumption behind the favourability maps of Kaufmann et al.
    CO2_LEVEL  = os.environ.get('CO2_PRICE')
    _psuf2 = f"_P{float(CO2_LEVEL):g}" if CO2_LEVEL else ""
    GAS_LEVEL  = os.environ.get('GAS_PRICE')
    ELEC_LEVEL = os.environ.get('ELEC_PRICE')
    # Electricity-price sensitivity (env ELEC_MULT): scale the whole buy AND sell
    # trajectory by a factor, preserving the hourly shape and the sell/buy ratio.
    # Unlike ELEC_PRICE (which flattens to a level) this keeps the price signal the
    # ORC and the heat pump actually respond to. Folder is tagged _E<pct>.
    ELEC_MULT = float(os.environ.get('ELEC_MULT', '1') or '1')
    _emsuf = f"_E{round(ELEC_MULT*100)}" if ELEC_MULT != 1.0 else ""
    _gsuf = f"_G{float(GAS_LEVEL):g}" if GAS_LEVEL else ""
    _esuf = f"_E{float(ELEC_LEVEL):g}" if ELEC_LEVEL else ""
    # Price-flexibility counterfactual (env SELL_PRICE=flat): hourly sell-price
    # columns are replaced by their annual means AFTER loading — the TSA is
    # untouched (clustering uses the flat BUY columns), so the rep days are
    # bit-identical to the hourly-price runs and the delta isolates the value
    # of electricity-price variability. Folder tagged _flatpx (regex-excluded
    # from the newest-run discovery of the figure scripts).
    SELL_PRICE_MODE = os.environ.get('SELL_PRICE', 'hourly').lower()
    _psuf = "_flatpx" if SELL_PRICE_MODE == 'flat' else ""
    # revORC dead-hour sensitivity (env REVORC_DEAD_HOURS): folder tagged _DH<n>
    _dhsuf = (f"_DH{float(os.environ['REVORC_DEAD_HOURS']):g}"
              if os.environ.get('REVORC_DEAD_HOURS') else "")
    results_dir = root / 'results' / 'rd' / f'MILP_{ts}_{SCENARIO}{_tsuf}{_csuf}{_psuf}{_psuf2}{_gsuf}{_esuf}{_emsuf}{_dhsuf}'
    results_dir.mkdir(parents=True, exist_ok=True)
    # keep a full log next to the results, so a run can be inspected afterwards
    logger.add(results_dir / "run.log", level="INFO", enqueue=True,
               format="{time:HH:mm:ss} | {level:<7} | {message}")
    logger.info(f"Munich RD model [{SCENARIO}] — output folder: {results_dir}")

    # TECH_DATA / DH_DATA override the input workbooks (file name or full path).
    # Used for diagnostic runs on alternative input sets; unset = the case data.
    tech_file   = root / 'data' / os.environ.get('TECH_DATA', 'Munich_tech_data.xlsx')
    demand_file = root / 'data' / os.environ.get('DH_DATA', 'Munich_DH_data.xlsx')
    if tech_file.name != 'Munich_tech_data.xlsx' or demand_file.name != 'Munich_DH_data.xlsx':
        logger.warning(f"Input override: tech={tech_file.name}, demand={demand_file.name}")
    techs, stor, prices, demand, extra, hp = load_all_munich_data(
        tech_file   = tech_file,
        demand_file = demand_file,
    )

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

    if SELL_PRICE_MODE == 'flat':
        _scols = [c for c in demand.columns if c.startswith('sell_electricity_price_')]
        for c in _scols:
            demand[c] = float(demand[c].mean())
        logger.info(
            f"SELL_PRICE=flat — {len(_scols)} hourly sell-price columns replaced by "
            "their annual means (233→155 €/MWh trajectory kept; TSA unchanged)"
        )

    # ── Apply scenario storage configuration ──
    if SCENARIO == 'S1':        # no storage at all
        BTES_BUILD_PERIOD = None; stor = {}
    elif SCENARIO == 'S2':      # HT-BTES only (drop PTES)
        BTES_BUILD_PERIOD = 2025; stor = {}
    elif SCENARIO == 'S3':      # pit/PTES only (keep storage sheet, no BTES)
        BTES_BUILD_PERIOD = None
    elif SCENARIO == 'S4':      # HT-BTES, field size chosen endogenously
        BTES_BUILD_PERIOD = 2025; stor = {}
    elif SCENARIO == 'S5':      # everything on the table: HT-BTES (endogenous
        BTES_BUILD_PERIOD = 2025            # design) + PTES + revORC/HTHP + boiler
    elif SCENARIO in ('N1', 'N2', 'N3', 'N4', 'N5', 'N6'):
        # ── New suite (paper v2): every scenario shares the base system
        #    geothermal + gas boiler; one flexibility option is added at a time,
        #    all of them together in N6.
        #      N1 base | N2 +revORC | N3 +HTHP | N4 +HT-BTES (endogenous design)
        #      N5 +PTES | N6 all options
        #    The short-term buffer store is NOT part of the case: the reference
        #    plant of Kaufmann et al. has no thermal store, and with an inter-day
        #    state variable the optimiser used a 25-h tank as a multi-week store
        #    (8-10 equivalent full cycles per year). Set KEEP_BUFFER=1 to restore
        #    it for a diagnostic run. ──
        _keep_buffer = os.environ.get('KEEP_BUFFER') == '1'
        stor = {k: v for k, v in stor.items()
                if ('buffer' not in k.lower() or _keep_buffer)
                and ('buffer' in k.lower() or SCENARIO in ('N5', 'N6'))}
        BTES_BUILD_PERIOD = 2025 if SCENARIO in ('N4', 'N6') else None
        if SCENARIO not in ('N2', 'N6'):        # revORC only in N2/N6
            os.environ['REVORC'] = '0'
        if SCENARIO == 'N6':                    # all options: both machines allowed
            os.environ.setdefault('REVORC_EXCLUSIVE', '0')
    else:
        raise SystemExit(f"Unknown SCENARIO '{SCENARIO}' (use S1-S5 or N1-N6)")
    # HTHP is investable only in N3/N6 within the new suite (its parameters stay
    # in the model, since the revORC series-HP mode reuses the CI coupling and
    # coupling; investment is fixed to zero after build for the other scenarios).
    EXCLUDE_HTHP = SCENARIO in ('N1', 'N2', 'N4', 'N5')
    logger.info(f"Scenario {SCENARIO}: BTES={BTES_BUILD_PERIOD}, "
                f"storages={list(stor)}, "
                f"revORC={'on' if os.environ.get('REVORC', '1') != '0' else 'off'}, "
                f"HTHP={'excluded' if EXCLUDE_HTHP else 'investable'}")

    # Planning horizon & clustering resolution come from the 'extra' sheet.
    global INVESTMENT_YEARS, N_TYPICAL_DAYS
    INVESTMENT_YEARS = extra['investment_years']
    N_TYPICAL_DAYS   = extra['n_typical_days']
    # N_TYPICAL_DAYS env override: clustering-resolution sweeps
    if os.environ.get('N_TYPICAL_DAYS'):
        _nd = int(os.environ['N_TYPICAL_DAYS'])
        logger.warning(f"N_TYPICAL_DAYS override: {N_TYPICAL_DAYS} -> {_nd}")
        N_TYPICAL_DAYS = _nd
        extra['n_typical_days'] = _nd
    logger.info(f"Investment years: {INVESTMENT_YEARS}  |  N typical days: {N_TYPICAL_DAYS}")

    # ── Derive geo & HTHP capacities from the brine resource (SAPHEA), overriding
    #    the tech-sheet 'Maximal Output Power' so capacity is resource-driven, not a
    #    free input. The geothermal doublet is an EXISTING asset (installed = design
    #    DU rating); the HTHP is greenfield (max investable = design output rating). ──
    if CO2_MULT != 1.0:
        extra['carbon_price_by_period'] = {
            y: v * CO2_MULT for y, v in extra['carbon_price_by_period'].items()}
        logger.info(f"CO2 price ×{CO2_MULT:g} → {extra['carbon_price_by_period']}")

    if CO2_LEVEL:
        _c = float(CO2_LEVEL)
        extra['carbon_price_by_period'] = {
            y: _c for y in extra['carbon_price_by_period']}
        logger.info(f"Carbon price OVERRIDE → flat {_c:g} EUR/tCO2 (all periods)")

    if GAS_LEVEL:
        _g = float(GAS_LEVEL)
        prices['natural gas']['price_by_period'] = {
            y: _g for y in prices['natural gas']['price_by_period']}
        logger.info(f"Gas price OVERRIDE → flat {_g:g} EUR/MWh (all periods)")
    if ELEC_LEVEL:
        _e = float(ELEC_LEVEL)
        _buy = [c for c in demand.columns if c.startswith('electricity_price_')]
        _sell = [c for c in demand.columns if c.startswith('sell_electricity_price_')]
        for c in _buy:
            demand[c] = _e
        for c in _sell:            # keep the hourly shape, rescale to the new mean
            demand[c] = demand[c] * (_e / demand[c].mean())
        logger.info(
            f"Electricity price OVERRIDE → flat buy {_e:g} EUR/MWh, sell rescaled "
            f"to the same mean ({len(_buy)} buy + {len(_sell)} sell columns)")
    if ELEC_MULT != 1.0:
        _cols = [c for c in demand.columns
                 if c.startswith('electricity_price_')
                 or c.startswith('sell_electricity_price_')]
        for c in _cols:
            demand[c] = demand[c] * ELEC_MULT
        logger.warning(
            f"ELEC_MULT={ELEC_MULT:g}: buy and sell electricity prices scaled "
            f"({len(_cols)} columns); hourly shape and sell/buy ratio preserved")

    resource = extra['geo_resource']
    if T_INJECT_OVERRIDE:
        resource['T_inject_min'] = float(T_INJECT_OVERRIDE)
        logger.info(f"T_inject_min OVERRIDE → {resource['T_inject_min']:.0f} °C (sensitivity)")
    design   = derive_saphea_capacities(resource)
    for tech, key, val in [
        (GEO_TECH,  'max_output_MW',          design['Q_geo_du']),
        (GEO_TECH,  'installed_capacity_MW',  design['Q_geo_du']),
        (HTHP_TECH, 'max_output_MW',          design['Q_hthp_max']),
    ]:
        old = techs[tech].get(key)
        if old is not None and abs(float(old) - val) > 0.05:
            logger.info(
                f"  Override {tech}.{key}: {float(old):.2f} → {val:.2f} MW (resource-derived)"
            )
        techs[tech][key] = val

    # ── Grid capacity fee (Leistungspreis Netzkosten): a charge on PEAK electrical
    #    draw, so it scales with installed capacity rather than with generation.
    #    Converted from EUR/kW_el/a to EUR/MW_th/a through the design COP and added
    #    to fixed OPEX. Not part of the DEA fixed-O&M definition (administration,
    #    staff, insurance, service agreements, property tax) nor of the VDI 2067
    #    repair-and-servicing basis used for the revORC, so this is additive. ──
    _grid_fee = 0.0
    try:
        _rv = pd.read_excel(tech_file, sheet_name='RevORC')
        _hit = _rv[_rv['parameter'].astype(str).str.strip()
                   == 'grid_capacity_fee_eur_per_kW_yr']
        if len(_hit):
            _grid_fee = float(_hit['value'].iloc[0])
    except Exception as _e:
        logger.warning(f"grid capacity fee not read from the RevORC sheet: {_e}")
    # GRID_CAP_FEE overrides the sheet value (0 disables the capacity charge while
    # leaving the energy-based grid fee, which sits in the purchase price series)
    if os.environ.get('GRID_CAP_FEE') is not None:
        _override = float(os.environ['GRID_CAP_FEE'])
        logger.warning(f"GRID_CAP_FEE override: {_grid_fee:.0f} -> {_override:.0f} EUR/kW_el/a")
        _grid_fee = _override
    if _grid_fee > 0:
        logger.info(f"  Grid capacity fee {_grid_fee:.0f} EUR/kW_el/a charged on the "
                    f"modelled PEAK electricity draw (see _add_grid_capacity_charge)")

    # HTHP CAPEX from the SAPHEA cost curve (if the 'HTHP CAPEX Curve' sheet exists);
    # otherwise the tech-sheet 'investment costs' value is kept.
    capex_curve = extra.get('hthp_capex_curve') or []
    # HTHP_CAPEX_EXCEL=1 keeps the tech-sheet value instead of the SAPHEA curve.
    # Use it when the sheet carries a catalogue cost on the MW_th-output basis
    # (e.g. DEA '40 Comp. hp, excess heat 10 MW'); the curve is charged on the
    # evaporator basis and is not comparable term-by-term.
    if capex_curve and os.environ.get('HTHP_CAPEX_EXCEL') == '1':
        logger.info(
            f"  HTHP_CAPEX_EXCEL=1: keeping the tech-sheet CAPEX "
            f"{float(techs[HTHP_TECH]['capex_EUR_per_MW']):,.0f} EUR/MW_th "
            f"(SAPHEA curve skipped)")
        capex_curve = []
    if capex_curve:
        scale      = extra.get('hthp_capex_scale', 0.8)
        hthp_capex = derive_hthp_capex(
            capex_curve, scale, design['Q_evap'], design['Q_hthp_max']
        )
        old_capex = techs[HTHP_TECH].get('capex_EUR_per_MW')
        logger.info(
            f"  HTHP CAPEX from curve: {float(old_capex):,.0f} → {hthp_capex:,.0f} €/MW "
            f"(interp@{design['Q_hthp_max']:.2f} MW ×{scale}, charged on "
            f"{design['Q_evap']:.2f} MW evaporator)"
        )
        techs[HTHP_TECH]['capex_EUR_per_MW'] = hthp_capex

    # ── revORC (reversible HP/ORC): inject as a separate greenfield technology if
    #    the 'RevORC' sheet is present and enabled. env REVORC=0 disables it. ──
    revorc_cfg = None
    if os.environ.get('REVORC', '1') != '0':
        revorc_cfg = load_revorc_config(tech_file)
    if revorc_cfg:
        _inject_revorc_tech(techs, revorc_cfg)

    uc = None if FORCE_LP else build_uc_params(techs)
    logger.info(f"MILP mode: {'ENABLED' if uc else 'DISABLED (pure LP)'}")
    logger.info(f"BTES mode: {'built in ' + str(BTES_BUILD_PERIOD) if BTES_BUILD_PERIOD else 'DISABLED'}")

    btes_json = root / 'data' / 'btes' / 'btes_hourly_params.json'
    btes_lib  = root / 'data' / 'btes' / 'design_library.json'
    # the suffixed library (BTES_LIB_SUFFIX, e.g. _vf/_vfa) carries the same designs plus any extra sizes (400 bh)
    _lib_suf = root / 'data' / 'btes' / f"design_library{os.environ.get('BTES_LIB_SUFFIX', '')}.json"
    if os.environ.get('BTES_LIB_SUFFIX') and _lib_suf.exists():
        btes_lib = _lib_suf
    # ── Endogenous-design ENUMERATION mode: the 5-design monolith is
    #    numerically unreliable (25-yr temperature chains x 5 designs -> wrong
    #    "optimal" results), so the design choice is made by enumeration:
    #    BTES_DESIGN_N=<n> solves the stable single-design model for one
    #    library entry; BTES_OFF=1 gives the no-BTES member of the enumeration.
    BTES_DESIGN_N = os.environ.get('BTES_DESIGN_N')
    if os.environ.get('BTES_OFF'):
        BTES_BUILD_PERIOD = None
        logger.info("BTES_OFF=1 — BTES disabled for this run")
    if BTES_DESIGN_N and BTES_BUILD_PERIOD:
        btes_json = (root / 'data' / 'btes' /
                     f"btes_hourly_params{os.environ.get('BTES_LIB_SUFFIX', '')}_n{BTES_DESIGN_N}.json")
        logger.info(f"BTES single-design run: {BTES_DESIGN_N} boreholes ({btes_json.name})")

    model = DistrictHeatingModel(
        demand_data=demand, technologies=techs, storage=stor, prices=prices,
        investment_years=INVESTMENT_YEARS, n_typical_days=N_TYPICAL_DAYS,
        extra_params=extra, hp_params=hp if hp else None, uc_params=uc,
        btes_build_period=BTES_BUILD_PERIOD,
        btes_arx_json=btes_json if BTES_BUILD_PERIOD else None,
        btes_design_library=(btes_lib if (BTES_BUILD_PERIOD and btes_lib.exists()
                                          and not BTES_DESIGN_N
                                          and SCENARIO in ('S4', 'S5', 'N4', 'N6')) else None),
        btes_force_design=os.environ.get('BTES_FORCE_DESIGN'),
        stratified_tsa=True,
        clustering_mode=TSA_MODE,   # 'joint' (default) or 'per_period', see top of file
        allow_reinjection=True,   # geo must-run: reinject excess hot when demand < min-load
    )
    logger.info(f"Multi-year TSA clustering mode: {TSA_MODE}")
    if BTES_DESIGN_N and model.btes_enabled and not getattr(model, '_btes_designs', None):
        import json as _json
        _libd = _json.loads(btes_lib.read_text())[str(BTES_DESIGN_N)]
        model._btes_arx['capex_EUR'] = float(_libd['capex_EUR'])
        model._btes_arx['lifetime_a'] = int(_libd['lifetime_a'])
        logger.info(f"BTES design {BTES_DESIGN_N}: CAPEX override "
                    f"{_libd['capex_EUR']/1e6:.2f} MEUR, lifetime {_libd['lifetime_a']} a")
    model.build()
    # ── BTES_DEBUG=fam1,fam2 deactivates named constraint families of the
    #    endogenous-design block (diagnostic only) ──
    _dbg = os.environ.get('BTES_DEBUG', '')
    if _dbg:
        _m = model.model
        fams = {
            'warm': ['btes_warm_lb_k', 'btes_warm_ub_k', 'btes_dis_warm_guard_k'],
            'warmpair': ['btes_warm_lb_k', 'btes_warm_ub_k'],
            'warmguard': ['btes_dis_warm_guard_k'],
            'tlb':  ['btes_T_lb_k'],
            'tub':  ['btes_T_ub_k'],
            'arx':  ['btes_arx_k', 'btes_year_link_k', 'btes_T_init_k'],
            'bal':  ['btes_energy_balance_k'],
        }
        if 'fixy0' in _dbg:
            for _k in _m.BTES_DESIGNS:
                _m.btes_y[_k].fix(0)
            logger.warning("BTES_DEBUG: all btes_y fixed to 0")
        for f in _dbg.split(','):
            for cname in fams.get(f.strip(), []):
                c = getattr(_m, cname, None)
                if c is not None:
                    c.deactivate()
                    logger.warning(f"BTES_DEBUG: deactivated {cname}")
    if os.environ.get('BTES_REPORT'):
        model._btes_report = True

    # FORCE_REVORC_MW=<MW>: build that much reversible capacity in the first
    # milestone and allow no further additions. Diagnostic only — used to test
    # whether the optimiser's own capacity choice is optimal in its own terms.
    _force_rev = os.environ.get('FORCE_REVORC_MW')
    if _force_rev and REVORC_TECH in model.technologies:
        _mw = float(_force_rev)
        _yrs = sorted(model.investment_years)
        model.model.cap_add[REVORC_TECH, _yrs[0]].fix(_mw)
        for _p in _yrs[1:]:
            model.model.cap_add[REVORC_TECH, _p].fix(0)
        logger.warning(f"FORCE_REVORC_MW={_mw:g}: revORC fixed at {_mw:g} MW from {_yrs[0]}")

    if EXCLUDE_HTHP:
        for _p in model.investment_years:
            model.model.cap_add[HTHP_TECH, _p].fix(0)
        logger.info(f"HTHP investment fixed to 0 (scenario {SCENARIO})")
    # FORCE_HTHP_MW=<MW>: build that much HTHP in the first milestone and allow no
    # further additions. Diagnostic only — used to price the option the optimizer
    # declines, not part of any reported scenario.
    _force_hthp = os.environ.get('FORCE_HTHP_MW')
    if _force_hthp and not EXCLUDE_HTHP:
        _mw = float(_force_hthp)
        for _i, _p in enumerate(model.investment_years):
            model.model.cap_add[HTHP_TECH, _p].fix(_mw if _i == 0 else 0.0)
        logger.info(f"HTHP investment FORCED to {_mw:.2f} MW in "
                    f"{model.investment_years[0]} (diagnostic run)")
    # ── Reinjection is brine returned to the well, so it cannot exceed the heat the
    #    doublet actually produced. Without this bound any generation may be dumped,
    #    which turns negative-price hours into a reward for running electric units
    #    and reinjecting their output. ──
    _m = model.model
    if hasattr(_m, 'reinject') and GEO_TECH in set(_m.TECH):
        def _reinject_cap_rule(mm, p_, d_, h_):
            return mm.reinject[p_, d_, h_] <= mm.gen[GEO_TECH, p_, d_, h_]
        _m.reinject_le_geo = pyo.Constraint(
            _m.PERIODS, _m.REP_DAYS, _m.HOURS, rule=_reinject_cap_rule)
        logger.info("Reinjection bounded by geothermal generation "
                    f"({len(_m.PERIODS)*len(_m.REP_DAYS)*24} constraints)")

    _build_saphea_coupling(model, demand, resource)
    if revorc_cfg:
        _build_revorc(model, demand, resource, revorc_cfg)

    # FORCE_CI_ABOVE=<MW>: fix y_ci=1 in every slot whose demand exceeds the
    # threshold. Diagnostic: separates "the model cannot see the value of HP
    # mode" from "the CI penalty makes HP mode genuinely unattractive". If HP
    # heat stays zero once CI is free of charge, the objection is economic; if
    # it turns positive, the mode choice was the obstacle.
    _force_ci = os.environ.get('FORCE_CI_ABOVE')
    if _force_ci and hasattr(model.model, 'y_ci'):
        _thr = float(_force_ci)
        _m = model.model
        _n = 0
        for _p in model.investment_years:
            for _d in _m.REP_DAYS:
                for _h in _m.HOURS:
                    if float(pyo.value(_m.demand[_p, _d, _h])) > _thr:
                        _m.y_ci[_p, _d, _h].fix(1)
                        _n += 1
        logger.warning(f"FORCE_CI_ABOVE={_thr:g} MW: y_ci fixed to 1 in {_n} slots")

    _add_grid_capacity_charge(
        model, _grid_fee,
        float((revorc_cfg or {}).get('brine_pump_el_per_MWth', 0.0)))

    # ── Geothermal true must-run: the doublet cannot be cycled off. Fix its
    #    unit-commitment binary on for every hour so it always runs >= min-load
    #    (0.65 × installed). Without this the solver may shut it down in summer
    #    (seen in the pit-storage scenario). ──
    m = model.model
    if (GEO_MUST_RUN_ALWAYS and not FORCE_LP
            and hasattr(m, "u") and GEO_TECH in set(m.UC_TECH)):
        n = 0
        for p in m.PERIODS:
            for d in m.REP_DAYS:
                for h in m.HOURS:
                    m.u[GEO_TECH, p, d, h].fix(1)
                    n += 1
        logger.info(f"Geothermal forced always-on (must-run): u[{GEO_TECH}]=1 fixed on {n} slots")

    save_tsa_analysis(model, demand, results_dir)

    if os.environ.get('BTES_IIS'):
        # cap the objective below the pathological optimum -> infeasible ->
        # Gurobi's IIS names the conflicting constraints. Placed AFTER the
        # SAPHEA coupling / must-run fixes so the exported model is complete.
        _m = model.model
        _m.dbg_costcap = pyo.Constraint(expr=_m.total_cost.expr <= 2.0e7)
        _lp = str(results_dir / 'n4_debug.lp')
        _m.write(_lp, io_options={'symbolic_solver_labels': True})
        import subprocess
        r = subprocess.run(['gurobi_cl', 'TimeLimit=900',
                            f'ResultFile={results_dir}/n4_debug.ilp', _lp],
                           capture_output=True, text=True)
        print(r.stdout[-1800:])
        raise SystemExit('IIS probe done')

    t0 = time.perf_counter()
    try:

        globals()["_IN_SOLVE"] = True
        results_raw = model.solve(
            solver_name='gurobi', tee=True,
            MIPGap=float(os.environ.get('GRB_MIP_GAP', 0.005)),
            MIPFocus=2, Heuristics=0.05, Cuts=2,
            # GRB_NUMERIC_FOCUS>0 trades speed for numerical care. The endogenous
            # BTES design block spans ~11 orders of magnitude (ARX coefficients
            # ~1e-6 against kW-scale flows), which can make Gurobi prune feasible
            # nodes at default tolerances.
            NumericFocus=int(os.environ.get('GRB_NUMERIC_FOCUS', 0)),
            # 30-min cap: hard instances return their incumbent + reported gap
            # instead of grinding for hours; the gurobi.log in the results
            # folder makes solve progress observable while running.
            TimeLimit=int(os.environ.get('GRB_TIME_LIMIT', 1800)),
            Threads=GRB_THREADS,
            LogFile=str(results_dir / 'gurobi.log'),
            # Gurobi dumps every new incumbent here as it finds it, so the best
            # solution so far survives even if the process is killed. See the
            # graceful-stop note in run_full.py.
            SolFiles=str(results_dir / 'incumbent'),
        )
    except KeyboardInterrupt:
        logger.warning("Interrupted — saving results at current MIP gap …")
        results_raw = None
    # solve window closed: from here on a stop signal only logs,
    # because interrupting the write-up is what actually loses results
    globals()["_IN_SOLVE"] = False
    wall_time = time.perf_counter() - t0

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

    res = model.get_results()

    if revorc_cfg:
        _report_revorc(model, results_dir)
        save_revorc_price_analysis(model, results_dir)

    logger.info("Saving plots...")
    save_dispatch_plots(model, results_dir)
    save_winter_dispatch_and_csv(model, results_dir)
    save_rep_dispatch(model, results_dir)
    save_combined_dispatch_plot(model, results_dir)
    save_soc_plots(model, results_dir)
    save_btes_plots(model, results_dir)
    save_btes_timeseries(model, results_dir)
    save_btes_operation_plot(model, results_dir)
    save_capacity_plot(res, results_dir)
    save_generation_plot(res, results_dir)
    save_mode_plots(model, results_dir)
    if results_raw is not None:
        save_computational_params(model, results_raw, wall_time, results_dir)
    comp_params = collect_computational_params(model, results_raw, wall_time) if results_raw is not None else {}
    export_excel(model, res, results_dir, comp_params)

    logger.info(f"Objective (NPV): EUR {res['objective_value']:,.0f}")

    if os.environ.get('BTES_REPORT') and hasattr(model.model, 'btes_T_k'):
        m_ = model.model
        for k_ in m_.BTES_DESIGNS:
            yv = pyo.value(m_.btes_y[k_])
            Ts = [pyo.value(m_.btes_T_k[k_, y_, d_]) for y_ in (2025, 2027, 2029)
                  for d_ in (0, 180, 364)]
            logger.info(f"BTES_REPORT k={k_} y={yv:.0f} "
                        f"T[2025/27/29 @ d0,180,364]=" +
                        ",".join(f"{t:.1f}" for t in Ts))
        n_warm1 = sum(1 for i in m_.BTES_WARM_K if pyo.value(m_.btes_warm_k[i]) > 0.5)
        logger.info(f"BTES_REPORT warm=1 on {n_warm1}/{len(m_.BTES_WARM_K)} triples")
        # per-design flow totals and the temperature envelope actually reached
        for k_ in m_.BTES_DESIGNS:
            ch = sum(pyo.value(m_.btes_qch_k[k_, p_, d_, h_])
                     for p_ in m_.PERIODS for d_ in m_.REP_DAYS for h_ in m_.HOURS)
            di = sum(pyo.value(m_.btes_qdis_k[k_, p_, d_, h_])
                     for p_ in m_.PERIODS for d_ in m_.REP_DAYS for h_ in m_.HOURS)
            Tv = [pyo.value(m_.btes_T_k[k_, y_, d_])
                  for y_ in m_.BTES_YEARS for d_ in m_.DAYS]
            logger.info(f"BTES_REPORT k={k_} qch_sum={ch/1000:.1f} MWh_rep "
                        f"qdis_sum={di/1000:.1f} MWh_rep  T range "
                        f"{min(Tv):.1f}–{max(Tv):.1f} °C")
    try:
        lcoh_df = model.compute_lcoh()
        lcoh_df.to_csv(results_dir / 'lcoh_by_period.csv')
        logger.info("LCOH saved")
    except Exception as e:
        logger.warning(f"LCOH skipped: {e}")
    logger.info(f"Done. Results saved to: {results_dir}")


if __name__ == '__main__':
    main()
