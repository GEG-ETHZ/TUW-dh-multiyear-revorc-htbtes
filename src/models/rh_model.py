"""
Multi-year district heating model — Representative Hours (RH) variant.

Uses the *reduced RH storage representation* from arXiv 2512.00892,
Section 3.3.  Key differences from the RD (representative days) model:

- Dispatch is indexed by representative hours I (not days × hours)
- Storage level is tracked on the reduced index J (groups of adjacent
  hours sharing the same representative hour)
- This avoids the Kotzur superposition assumption entirely and
  preserves full chronological storage dynamics at a fraction of the
  cost of tracking all 8760 hours

Same technology / investment / fuel / emission logic as complete_model.py.
"""

import os
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from loguru import logger
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from preprocessing.time_aggregation import create_representative_hours_with_mapping
from models.btes_params import period_params_for_build, load_arx_params
from models.btes_constraints import add_btes_to_rh_model
from utils.economic_calculations import (
    planning_horizon_end_year,
    linear_salvage_fraction_of_capex,
    discount_factor_end_of_calendar_year,
)


def _has_retirement_data(items: Dict[str, Dict[str, Any]]) -> bool:
    """True if at least one item has a non-None installation_year."""
    return any(
        v.get("installation_year") is not None
        for v in items.values()
        if v.get("installed_capacity_MW", v.get("capacity_MWh", 0)) > 0
    )


class DistrictHeatingModelRH:
    """
    Multi-year capacity expansion model using Representative Hours (RH).

    Same interface as DistrictHeatingModel (RD) but with RH-based
    dispatch and the reduced storage representation from the paper.
    """

    def __init__(
        self,
        demand_data: pd.DataFrame,
        technologies: Dict[str, Dict[str, Any]],
        storage: Dict[str, Dict[str, Any]],
        prices: Dict[str, Dict[str, float]],
        investment_years: List[int] = None,
        n_rep_hours: int = 1000,
        extra_params: Dict[str, float] = None,
        hp_params: Dict[str, Dict[str, Any]] = None,
        uc_params: Dict[str, Dict[str, float]] = None,
        btes_build_period: Optional[int] = None,
        btes_arx_json: Optional[Path] = None,
        name: str = "KranjDH_RH",
        segmented: bool = False,
        clustering_mode: str = "per_period",
        allow_reinjection: bool = False,
    ):
        if investment_years is None:
            investment_years = [2025, 2030, 2035, 2040, 2045]
            
        if extra_params is None:
            extra_params = {}

        self.demand_data = demand_data
        self.technologies = technologies
        self.storage = storage
        self.prices = prices
        self.hp_params = hp_params or {}
        self.investment_years = sorted(investment_years)
        self.n_rep_hours = n_rep_hours
        self.discount_rate_by_period = extra_params["discount_rate_by_period"]
        self.carbon_price_by_period = extra_params.get(
            "carbon_price_by_period",
            {yr: 25.0 for yr in investment_years},
        )
        self.name = name
        self.segmented = segmented
        self.clustering_mode = clustering_mode  # "per_period" or "joint"
        # Must-run geothermal: when demand falls below the well's minimum load the
        # unused high-temperature heat is reinjected hot, not curtailed. Without
        # this slack the Munich case is infeasible every summer hour.
        self.allow_reinjection = allow_reinjection
        self.model = pyo.ConcreteModel(name=name)

        # Unit commitment (MILP extension) — gated; None keeps model as pure LP.
        # uc_params: dict mapping tech_name → {c_startup [€/event],
        #                                       c_ramp     [€/MW ramped],
        #                                       min_load_factor [0–1]}
        self.uc_params = uc_params or {}
        self.uc_enabled = bool(self.uc_params)
        self._uc_tech_list = [
            t for t, v in self.uc_params.items()
            if v.get("c_startup", 0) > 0 or v.get("c_ramp", 0) > 0
            or v.get("min_load_factor", 0) > 0 or v.get("cold_start_hours", 0) > 0
        ] if self.uc_enabled else []
        self.btes_enabled = btes_build_period is not None
        self.btes_build_period = btes_build_period
        if self.btes_enabled:
            self._btes_arx = load_arx_params(btes_arx_json)
            self._btes_period_params = period_params_for_build(
                sorted(investment_years), btes_build_period, arx=self._btes_arx
            )
        else:
            self._btes_arx = None
            self._btes_period_params = None

        self.retirement_enabled = _has_retirement_data(technologies)
        self.stor_retirement_enabled = _has_retirement_data(storage)

        if self.retirement_enabled:
            self._precompute_tech_retirement()
        if self.stor_retirement_enabled:
            self._precompute_stor_retirement()

        # ── RH time-series aggregation (PER investment period) ───────
        # Each milestone is clustered INDEPENDENTLY on its OWN demand column
        # (corrected_demand_{p}) and its OWN electricity price
        # (electricity_price_{p}); temperature is shared for now.  This yields,
        # per period, its own representative hours, weights, sigma mapping and
        # reduced storage index J.  The rep-hour count I is fixed across
        # periods; |J| MAY differ per period (especially non-segmented).
        # weightDict normalises attributes to unit std (non-segmented only);
        # segmented Ward handles normalisation internally.  Column names are
        # preserved so rep_profiles["heat_demand"] works downstream.
        _all_price_cols = [f"electricity_price_{p}" for p in self.investment_years
                           if f"electricity_price_{p}" in demand_data.columns]
        _elec_mean = (demand_data[_all_price_cols].mean(axis=1)
                      if _all_price_cols else None)

        def _run_rh_tsa(dem_series, price_series, label):
            tsa_df = pd.DataFrame(
                {"heat_demand": dem_series.values}, index=demand_data.index
            )
            tsam_kwargs: dict = {}
            if price_series is not None:
                tsa_df["elec_price_mean"] = price_series.values
                if not self.segmented:
                    # tsam min-max scales each column and applies weightDict only
                    # AFTER, so 1/std in raw units does not equalise anything: it
                    # left demand ~30x the influence of price in the RD model.
                    # Equal weights on the scaled columns is the balanced choice.
                    _pw = float(os.environ.get("TSA_PRICE_WEIGHT", 1.0))
                    tsam_kwargs["weightDict"] = {
                        "heat_demand": 1.0, "elec_price_mean": _pw,
                    }
            # Season-pure clustering keeps every representative hour inside one
            # BTES seasonal window. Without it the majority-vote season in
            # add_btes_to_rh_model() misassigns 12 % of the year (0.3 % when
            # segmented). RH_SEASON_PURE=0 disables.
            season_pure = (self.btes_enabled
                           and os.environ.get("RH_SEASON_PURE", "1") == "1")
            logger.info(f"Running RH-TSA ({label}) on {list(tsa_df.columns)}"
                        f"{', season-pure' if season_pure else ''} …")
            return create_representative_hours_with_mapping(
                time_series_data=tsa_df, n_rep_hours=n_rep_hours,
                method="hierarchical", segmented=self.segmented,
                season_pure=season_pure, **tsam_kwargs,
            )

        self.rh_by_period: Dict[int, dict] = {}
        if self.clustering_mode == "joint":
            # ONE clustering on the pooled (mean-across-milestones) series, shared
            # by all periods; per-year values aggregated onto it downstream.
            _dem_cols = [f"corrected_demand_{p}" for p in self.investment_years
                         if f"corrected_demand_{p}" in demand_data.columns]
            dem_series = (demand_data[_dem_cols].mean(axis=1) if _dem_cols
                          else demand_data["heat_demand"])
            _joint = _run_rh_tsa(dem_series, _elec_mean, "JOINT, pooled across milestones")
            self.rh_by_period = {p: _joint for p in self.investment_years}
        else:
            for p in self.investment_years:
                dem_col = f"corrected_demand_{p}"
                dem_series = (demand_data[dem_col] if dem_col in demand_data.columns
                              else demand_data["heat_demand"])
                price_col = f"electricity_price_{p}"
                if price_col in demand_data.columns:
                    price_series = demand_data[price_col]
                elif _elec_mean is not None:
                    price_series = _elec_mean
                else:
                    price_series = None
                self.rh_by_period[p] = _run_rh_tsa(dem_series, price_series, f"period {p}")

        # Model size is fixed: every period must return the same rep-hour count.
        _n_reps = {p: self.rh_by_period[p]["n_rep_hours"]
                   for p in self.investment_years}
        assert len(set(_n_reps.values())) == 1, (
            f"Per-period clustering returned differing rep-hour counts: {_n_reps}. "
            "All periods must yield the same number of representative hours."
        )
        # Legacy single-clustering alias (first period) for BTES / older code.
        self.rh = self.rh_by_period[self.investment_years[0]]

        ret_tag = " [retirement ON]" if self.retirement_enabled else ""
        uc_tag  = " [UC/MILP ON]"    if self.uc_enabled        else ""
        logger.info(
            f"Initialised {name}: {len(investment_years)} investment periods, "
            f"I={n_rep_hours} rep hours, |J|={self.rh['n_storage_levels']} "
            f"storage levels, {len(storage)} storages, {len(technologies)} techs.{ret_tag}{uc_tag}"
        )

    def _precompute_tech_retirement(self):
        """Pre-compute how much existing capacity survives into each period."""
        self.existing_remaining = {}
        for t_name, t_data in self.technologies.items():
            inst_yr = t_data.get("installation_year")
            lifetime = t_data["lifetime_a"]
            existing = t_data["installed_capacity_MW"]
            for period in self.investment_years:
                if inst_yr is not None and existing > 0:
                    self.existing_remaining[(t_name, period)] = (
                        existing if period < inst_yr + lifetime else 0.0
                    )
                else:
                    self.existing_remaining[(t_name, period)] = existing

        retired = {t: yr for t, yr in (
            (t_name, t_data.get("installation_year", 0) + t_data["lifetime_a"])
            for t_name, t_data in self.technologies.items()
            if t_data.get("installation_year") is not None
            and t_data["installed_capacity_MW"] > 0
        ) if yr <= max(self.investment_years)}
        for t_name, ret_yr in retired.items():
            logger.info(f"  Retirement: {t_name} retires in {ret_yr}")

    def _precompute_stor_retirement(self):
        """Pre-compute how much existing storage capacity survives into each period."""
        self.stor_existing_remaining = {}
        for s_name, s_data in self.storage.items():
            inst_yr = s_data.get("installation_year")
            lifetime = s_data["lifetime_a"]
            existing = s_data["capacity_MWh"]
            for period in self.investment_years:
                if inst_yr is not None and existing > 0:
                    self.stor_existing_remaining[(s_name, period)] = (
                        existing if period < inst_yr + lifetime else 0.0
                    )
                else:
                    self.stor_existing_remaining[(s_name, period)] = existing

    # ──────────────────────────────────────────────────────────────────
    #  BUILD
    # ──────────────────────────────────────────────────────────────────
    def build(self):
        logger.info("Building RH model …")
        m = self.model
        rh = self.rh
        techs = self.technologies
        stor = self.storage
        prices = self.prices

        tech_names = list(techs.keys())
        stor_names = list(stor.keys())
        I = rh["n_rep_hours"]

        # ── Per-period RH structures (independent clustering) ──────────
        # Each milestone has its OWN sigma mapping, rep-hour weights, reduced
        # storage index J (|J| may differ per period), cluster members, and
        # chronological hour-transition counts for UC continuity.  I (rep-hour
        # count) is shared; everything else is period-specific.  Transitions
        # count how many times each rep-hour-type transition occurs in that
        # period's chronological 8760-hour sequence (including year wrap).
        sigma_by_period: Dict[int, list] = {}
        weights_by_period: Dict[int, list] = {}
        intervals_by_period: Dict[int, list] = {}
        J_by_period: Dict[int, int] = {}
        members_by_period: Dict[int, dict] = {}
        hour_transitions_by_period: Dict[int, dict] = {}
        for p in self.investment_years:
            rh_p = self.rh_by_period[p]
            sig = rh_p["sigma"]
            sigma_by_period[p] = sig
            weights_by_period[p] = rh_p["weights"]
            intervals_by_period[p] = rh_p["storage_intervals"]
            J_by_period[p] = rh_p["n_storage_levels"]
            mem = {i: [] for i in range(I)}
            for t, i in enumerate(sig):
                mem[i].append(t)
            members_by_period[p] = mem
            trans: dict = {}
            for t in range(len(sig) - 1):
                pair = (sig[t], sig[t + 1])
                trans[pair] = trans.get(pair, 0) + 1
            pair = (sig[-1], sig[0])   # year wrap
            trans[pair] = trans.get(pair, 0) + 1
            hour_transitions_by_period[p] = trans

        self._hour_transitions_by_period = hour_transitions_by_period  # get_results()
        # Legacy first-period aliases (BTES uses first period's sigma/intervals).
        first_p = self.investment_years[0]
        sigma = sigma_by_period[first_p]
        intervals = intervals_by_period[first_p]
        self._hour_transitions = hour_transitions_by_period[first_p]

        # ════════════════════════════════════════════════════════════
        #  SETS
        # ════════════════════════════════════════════════════════════
        m.TECH = pyo.Set(initialize=tech_names)
        m.STOR = pyo.Set(initialize=stor_names)
        m.PERIODS = pyo.Set(initialize=self.investment_years, ordered=True)
        m.REP_HOURS = pyo.RangeSet(0, I - 1)
        # Reduced storage index J differs per period → sparse (period, j) set.
        m.PERIOD_J = pyo.Set(
            dimen=2,
            initialize=[(p, j) for p in self.investment_years
                        for j in range(J_by_period[p])],
            ordered=True,
        )

        # ════════════════════════════════════════════════════════════
        #  PARAMETERS
        # ════════════════════════════════════════════════════════════

        # Demand per (period, representative hour)
        # -- demand per (period, rep_hour) --
        # Aggregate each period's ACTUAL demand on the cluster assignment
        # (mean over the original hours mapped to each representative hour),
        # consistent with the plain-mean representative-profile construction.
        # No scaling: each milestone uses its own demand series, so per-year
        # changes in demand shape are represented, not just magnitude.
        demand_init = {}
        for period in self.investment_years:
            col = f"corrected_demand_{period}"
            arr = (self.demand_data[col].values if col in self.demand_data.columns
                   else self.demand_data["heat_demand"].values)
            mem_p = members_by_period[period]
            for i in range(I):
                mem = mem_p[i]
                demand_init[(period, i)] = float(np.mean([arr[t] for t in mem])) if mem else 0.0
        m.demand = pyo.Param(m.PERIODS, m.REP_HOURS, initialize=demand_init)

        # Electricity price per (period, representative hour)
        # Average actual hourly prices over all hours mapped to each rep_hour
        # via that period's OWN sigma (members_by_period).
        # Buy price (electricity_price) → HP fuel cost.
        # Sell price (sell_electricity_price) → CHP revenue.
        def _build_price_dict_rh(col_prefix):
            price_dict = {}
            for period in self.investment_years:
                col = f"{col_prefix}_{period}"
                if col in self.demand_data.columns:
                    arr = self.demand_data[col].values
                else:
                    arr = next(
                        (self.demand_data[f"{col_prefix}_{yr}"].values
                         for yr in sorted(self.investment_years)
                         if f"{col_prefix}_{yr}" in self.demand_data.columns),
                        None,
                    )
                    if arr is None:
                        arr = np.zeros(8760)
                    logger.warning(f"No {col} column found, using fallback.")
                mem_p = members_by_period[period]
                for rep_h in range(I):
                    orig_hrs = mem_p[rep_h]
                    price_dict[period, rep_h] = (
                        float(np.mean([arr[h] for h in orig_hrs])) if orig_hrs else 0.0
                    )
            return price_dict

        elec_price_dict = _build_price_dict_rh("electricity_price")
        m.electricity_price = pyo.Param(m.PERIODS, m.REP_HOURS, initialize=elec_price_dict)

        sell_price_dict = _build_price_dict_rh("sell_electricity_price")
        if not any(f"sell_electricity_price_{p}" in self.demand_data.columns
                   for p in self.investment_years):
            sell_price_dict = elec_price_dict
        m.electricity_sell_price = pyo.Param(
            m.PERIODS, m.REP_HOURS, initialize=sell_price_dict
        )

        # Weight per (period, representative hour)
        m.weight = pyo.Param(
            m.PERIODS, m.REP_HOURS,
            initialize={
                (period, i): float(weights_by_period[period][i])
                for period in self.investment_years
                for i in range(I)
            },
        )

        # Storage interval parameters per (period, reduced level j)
        m.j_rep = pyo.Param(
            m.PERIOD_J,
            initialize={
                (period, j): intervals_by_period[period][j]["rep_hour"]
                for period in self.investment_years
                for j in range(J_by_period[period])
            },
        )
        m.j_dur = pyo.Param(
            m.PERIOD_J,
            initialize={
                (period, j): intervals_by_period[period][j]["duration"]
                for period in self.investment_years
                for j in range(J_by_period[period])
            },
        )

        # Technology parameters
        m.existing_cap = pyo.Param(m.TECH, initialize={t: techs[t]["installed_capacity_MW"] for t in tech_names})
        m.max_new_cap = pyo.Param(m.TECH, initialize={t: techs[t]["max_output_MW"] for t in tech_names})
        m.capex = pyo.Param(m.TECH, initialize={t: techs[t]["capex_EUR_per_MW"] for t in tech_names})
        m.opex_fix = pyo.Param(m.TECH, initialize={t: techs[t]["opex_fix_EUR_per_MWa"] for t in tech_names})
        m.opex_var = pyo.Param(m.TECH, initialize={t: techs[t]["opex_var_EUR_per_MWh"] for t in tech_names})
        m.lifetime = pyo.Param(m.TECH, initialize={t: techs[t]["lifetime_a"] for t in tech_names})
        m.efficiency_el = pyo.Param(m.TECH, initialize={t: techs[t]["efficiency_el"] for t in tech_names})
        m.max_potential = pyo.Param(m.TECH, initialize={t: techs[t]["max_potential_MWh"] for t in tech_names})
        # Big-M for UC linearisation: 2× system peak demand >> any feasible cap_total
        _bigM = float(self.demand_data["heat_demand"].max()) * 2.0
        m.uc_bigM = pyo.Param(initialize=_bigM)

        # Dynamic Efficiency and Restriction Factors
        temp_amb = (self.demand_data["temperature"].values
                    if "temperature" in self.demand_data.columns
                    else np.zeros(8760))
        temp_flow = self.demand_data["temp_supply"].values if "temp_supply" in self.demand_data.columns else np.full(8760, 70.0)
        temp_ret = self.demand_data["temp_return"].values if "temp_return" in self.demand_data.columns else np.full(8760, 35.0)

        # Rep-hour membership now differs per period, so temperature-based COP /
        # restriction factors are indexed by period as well: each period picks
        # its representative hour from its OWN cluster members.
        eff_dict = {}
        rf_dict = {}
        for period in self.investment_years:
            mem_p = members_by_period[period]
            for t_name in tech_names:
                carrier = techs[t_name].get("energy_carrier", "")
                if t_name in self.hp_params:
                    # Air/Water source HP — COP varies with rep-hour outdoor temperature
                    h = self.hp_params[t_name]
                    for i in range(I):
                        t_orig = mem_p[i][0] if mem_p[i] else 0
                        cop = (
                            (temp_amb[t_orig] - h["nom_source_temp"]) * h["sens_source"] +
                            (temp_ret[t_orig] - h["nom_return_temp"]) * h["sens_return"] +
                            (temp_flow[t_orig] - h["nom_flow_temp"]) * h["sens_flow"] +
                            techs[t_name]["efficiency_th"]
                        )
                        eff_dict[(t_name, period, i)] = max(cop, 1e-3)
                        rf = 1.0
                        if temp_amb[t_orig] < h["min_source_temp"]: rf = 0.0
                        elif temp_amb[t_orig] < h["fade_out_temp"]:
                            dx = h["fade_out_temp"] - h["min_source_temp"]
                            dy = temp_amb[t_orig] - h["min_source_temp"]
                            rf = dy / dx
                        if rf <= h["power_min_factor"]: rf = 0.0
                        rf_dict[(t_name, period, i)] = rf
                elif carrier.startswith("waste heat") and "Waste HP" in self.hp_params:
                    # Waste heat HP — COP uses fixed source temperature from wasteheat column
                    h = self.hp_params["Waste HP"]
                    col = carrier.replace("waste heat ", "wasteheat_")
                    if col not in self.demand_data.columns:
                        raise ValueError(
                            f"Waste-heat source-temperature column '{col}' required by "
                            f"generator '{t_name}' (carrier '{carrier}') is missing from the "
                            f"time-series input. Add a '{col}' column (hourly source "
                            f"temperature in °C) to the DH data."
                        )
                    src_arr = self.demand_data[col].values
                    for i in range(I):
                        t_orig = mem_p[i][0] if mem_p[i] else 0
                        t_source = float(src_arr[t_orig])
                        cop = (
                            (t_source - h["nom_source_temp"]) * h["sens_source"] +
                            (temp_ret[t_orig] - h["nom_return_temp"]) * h["sens_return"] +
                            (temp_flow[t_orig] - h["nom_flow_temp"]) * h["sens_flow"] +
                            techs[t_name]["efficiency_th"]
                        )
                        eff_dict[(t_name, period, i)] = max(cop, 1e-3)
                        rf_dict[(t_name, period, i)] = 1.0  # waste heat source always available
                else:
                    for i in range(I):
                        eff_dict[(t_name, period, i)] = techs[t_name]["efficiency_th"]
                        rf_dict[(t_name, period, i)] = 1.0

        # mutable: the SAPHEA coupling and the revORC overwrite the COP per
        # (tech, period, rep_hour) after build, as they do in the RD model
        m.efficiency = pyo.Param(m.TECH, m.PERIODS, m.REP_HOURS,
                                 initialize=eff_dict, mutable=True)
        m.restriction_factor = pyo.Param(m.TECH, m.PERIODS, m.REP_HOURS, initialize=rf_dict)

        # Plain Python dict for cold_start_hours — safe to use inside Pyomo rules
        cold_start_dict = {
            t: int(self.uc_params.get(t, {}).get("cold_start_hours", 0))
            for t in tech_names
        }

        # Unit commitment parameters (only built when uc_enabled)
        if self.uc_enabled:
            m.c_startup = pyo.Param(
                m.TECH,
                initialize={t: self.uc_params.get(t, {}).get("c_startup", 0.0) for t in tech_names},
            )
            m.c_ramp = pyo.Param(
                m.TECH,
                initialize={t: self.uc_params.get(t, {}).get("c_ramp", 0.0) for t in tech_names},
            )
            m.min_load_factor = pyo.Param(
                m.TECH,
                initialize={t: self.uc_params.get(t, {}).get("min_load_factor", 0.0) for t in tech_names},
            )


        # Fuel prices & emission factors per (tech, period)
        fuel_price_dict = {}
        emission_dict = {}
        carrier_dict = {}
        for t in tech_names:
            carrier = techs[t].get("energy_carrier")
            if not carrier or str(carrier).lower() == "nan":
                raise ValueError(
                    f"Technology '{t}' has no 'energy carrier' set in Excel. "
                    "Please fill in the 'energy carrier' column in Kranj_tech_data.xlsx."
                )
            if carrier not in prices:
                raise ValueError(
                    f"Technology '{t}' has energy carrier '{carrier}' which is not "
                    f"found in the prices sheet. Available carriers: {list(prices.keys())}"
                )
            carrier_dict[t] = carrier
            for period in self.investment_years:
                if carrier == "ambient heat" or carrier.startswith("waste heat"):
                    fuel_price_dict[t, period] = 0.0
                    emission_dict[t, period] = prices["electricity"]["emission_by_period"].get(period, 0.0)
                else:
                    fuel_price_dict[t, period] = prices[carrier]["price_by_period"].get(period, 0.0)
                    emission_dict[t, period] = prices[carrier]["emission_by_period"].get(period, 0.0)

        m.fuel_price = pyo.Param(m.TECH, m.PERIODS, initialize=fuel_price_dict)
        m.emission_factor = pyo.Param(m.TECH, m.PERIODS, initialize=emission_dict)
        m.tech_carrier = pyo.Param(m.TECH, initialize=carrier_dict, within=pyo.Any)

        # Carbon price — directly from input data per period
        carbon_prices = {
            period: self.carbon_price_by_period.get(period, 0.0)
            for period in self.investment_years
        }
        m.carbon_price = pyo.Param(m.PERIODS, initialize=carbon_prices)

        # Storage parameters
        if stor_names:
            m.stor_cap_existing = pyo.Param(
                m.STOR, initialize={s: stor[s]["capacity_MWh"] for s in stor_names}
            )
            m.stor_loss = pyo.Param(
                m.STOR, initialize={s: stor[s]["hourly_loss_frac"] for s in stor_names}
            )
            m.stor_eff_ch = pyo.Param(
                m.STOR, initialize={s: stor[s]["charge_eff"] for s in stor_names}
            )
            m.stor_eff_dis = pyo.Param(
                m.STOR, initialize={s: stor[s]["discharge_eff"] for s in stor_names}
            )
            m.tau_ch = pyo.Param(
                m.STOR, initialize={s: stor[s]["tau_charge_h"] for s in stor_names}
            )
            m.tau_dis = pyo.Param(
                m.STOR, initialize={s: stor[s]["tau_discharge_h"] for s in stor_names}
            )
            m.stor_capex = pyo.Param(
                m.STOR, initialize={s: stor[s]["capex_EUR_per_MWh"] for s in stor_names}
            )
            m.stor_opex_fix = pyo.Param(
                m.STOR, initialize={s: stor[s]["opex_fix_EUR_per_MWh"] for s in stor_names}
            )
            m.stor_lifetime = pyo.Param(
                m.STOR, initialize={s: stor[s]["lifetime_a"] for s in stor_names}
            )

        # ════════════════════════════════════════════════════════════
        #  VARIABLES
        # ════════════════════════════════════════════════════════════

        # Capacity
        m.cap_add = pyo.Var(m.TECH, m.PERIODS, domain=pyo.NonNegativeReals)
        m.cap_total = pyo.Var(m.TECH, m.PERIODS, domain=pyo.NonNegativeReals)

        # Dispatch — indexed by representative hours, not day×hour
        m.gen = pyo.Var(m.TECH, m.PERIODS, m.REP_HOURS, domain=pyo.NonNegativeReals)

        # Unit commitment binary variables (MILP, only for UC techs)
        if self.uc_enabled and self._uc_tech_list:
            m.UC_TECH = pyo.Set(initialize=self._uc_tech_list, within=m.TECH)
            m.u       = pyo.Var(m.UC_TECH, m.PERIODS, m.REP_HOURS, domain=pyo.Binary)
            m.ramp_up = pyo.Var(m.UC_TECH, m.PERIODS, m.REP_HOURS, domain=pyo.NonNegativeReals)
            # Per-period chronological transition set and occurrence counts.
            # Each period has its own clustering, hence its own set of (i1, i2)
            # rep-hour transitions.  Stored as a sparse (period, i1, i2) set.
            period_htrans_list = [
                (p, i1, i2)
                for p in self.investment_years
                for (i1, i2) in hour_transitions_by_period[p]
            ]
            m.PERIOD_HOUR_TRANS = pyo.Set(dimen=3, initialize=period_htrans_list)
            m.hour_trans_count = pyo.Param(
                m.PERIOD_HOUR_TRANS,
                initialize={
                    (p, i1, i2): hour_transitions_by_period[p][(i1, i2)]
                    for p in self.investment_years
                    for (i1, i2) in hour_transitions_by_period[p]
                },
            )
            # v_cross_h[tech, period, i1, i2]: 1 if a cold start occurs at the
            # actual-hour boundary from rep-hour i1 to rep-hour i2 (that period).
            m.v_cross_h = pyo.Var(
                m.UC_TECH, m.PERIOD_HOUR_TRANS, domain=pyo.Binary
            )

        # Storage — dispatch on representative hours, SOC on reduced index J
        if stor_names:
            m.stor_cap_add = pyo.Var(m.STOR, m.PERIODS, domain=pyo.NonNegativeReals)
            m.stor_cap_total = pyo.Var(m.STOR, m.PERIODS, domain=pyo.NonNegativeReals)
            m.charge = pyo.Var(
                m.STOR, m.PERIODS, m.REP_HOURS, domain=pyo.NonNegativeReals
            )
            m.discharge = pyo.Var(
                m.STOR, m.PERIODS, m.REP_HOURS, domain=pyo.NonNegativeReals
            )
            # Storage level on reduced index J (sparse per-period (period, j))
            m.soc = pyo.Var(
                m.STOR, m.PERIOD_J, domain=pyo.NonNegativeReals
            )

        # BTES variables + constraints (before objective)
        if self.btes_enabled:
            # add_btes_to_rh_model indexes the temperature state on
            # BTES_YEARS x STOR_LEVELS. The RH model itself uses the sparse
            # (period, j) set PERIOD_J because |J| can differ per period; the
            # BTES chain is built on the FIRST period's interval list, so give
            # it a dense level set of that length.
            if not hasattr(m, "STOR_LEVELS"):
                m.STOR_LEVELS = pyo.RangeSet(0, len(intervals) - 1)
            add_btes_to_rh_model(
                m,
                investment_years=self.investment_years,
                sigma=sigma,
                intervals=intervals,
                period_params=self._btes_period_params,
                arx=self._btes_arx,
            )
            logger.info("BTES constraints added to RH model.")

        # ════════════════════════════════════════════════════════════
        #  OBJECTIVE
        # ════════════════════════════════════════════════════════════
        base_yr_obj = min(self.investment_years)
        periods_list_obj = sorted(self.investment_years)
        horizon_end = planning_horizon_end_year(periods_list_obj, 5)
        df_salv = discount_factor_end_of_calendar_year(
            horizon_end, base_yr_obj, self.discount_rate_by_period[periods_list_obj[-1]]
        )

        # Tech and storage CAPEX both use lump-sum approach (see objective).

        # ── BTES CAPEX (lump-sum PV) and salvage (scalar, computed once) ─────
        btes_capex_lump  = 0.0
        btes_salv_credit = 0.0
        if self.btes_enabled:
            _cx  = self._btes_arx["capex_EUR"]
            _lt  = int(self._btes_arx["lifetime_a"])
            _df_build = 1.0 / (1.0 + self.discount_rate_by_period[self.btes_build_period]) ** (
                self.btes_build_period - base_yr_obj
            )
            btes_capex_lump = _cx * _df_build
            _frac = linear_salvage_fraction_of_capex(
                self.btes_build_period, _lt, horizon_end
            )
            btes_salv_credit = _cx * _frac * df_salv if _frac > 0 else 0.0
            logger.info(
                f"BTES CAPEX (lump-sum PV): {btes_capex_lump:,.0f} EUR | "
                f"salvage_credit (PV)={btes_salv_credit:,.0f} EUR "
                f"(remaining life {max(0, self.btes_build_period + _lt - horizon_end)} yr)"
            )

        def objective_rule(m):
            total = 0.0
            base_yr = base_yr_obj
            periods_list = periods_list_obj

            for p_idx, period in enumerate(periods_list):
                if p_idx + 1 < len(periods_list):
                    years_in_period = periods_list[p_idx + 1] - period
                else:
                    years_in_period = 5

                _r_p = self.discount_rate_by_period[period]
                pv_annual = sum(
                    1 / (1 + _r_p) ** (period + y - base_yr)
                    for y in range(years_in_period)
                )

                # ── Fixed OPEX storage (CAPEX lump-sum outside loop) ─────────
                if stor_names:
                    for s in m.STOR:
                        total += m.stor_cap_total[s, period] * m.stor_opex_fix[s] * pv_annual

                # Fixed OPEX
                for tech in m.TECH:
                    total += m.cap_total[tech, period] * m.opex_fix[tech] * pv_annual

                # Variable OPEX + fuel + CO2 - CHP revenue
                # Weight-based scaling: each rep hour i has weight w_i
                for tech in m.TECH:
                    for i in m.REP_HOURS:
                        weight = m.weight[period, i]
                        gen_val = m.gen[tech, period, i]
                        eff_val = m.efficiency[tech, period, i]
                        carrier = m.tech_carrier[tech]
                        if carrier == "electricity" or carrier == "ambient heat" or carrier.startswith("waste heat"):
                            p_fuel = m.electricity_price[period, i]
                        else:
                            p_fuel = m.fuel_price[tech, period]

                        fuel_cost = gen_val / eff_val * p_fuel
                        var_cost = gen_val * m.opex_var[tech]
                        co2_cost = (
                            gen_val
                            / eff_val
                            * m.emission_factor[tech, period]
                            * m.carbon_price[period]
                        )

                        elec_revenue = 0
                        if m.efficiency_el[tech] > 0:
                            elec_out = gen_val * m.efficiency_el[tech] / eff_val
                            elec_revenue = elec_out * m.electricity_sell_price[period, i]

                        total += (fuel_cost + var_cost + co2_cost - elec_revenue) * weight * pv_annual

                # UC costs — startup via sigma-based cross-hour transitions,
                # weighted by actual occurrence count; ramp via rep-hour sequence.
                if self.uc_enabled and self._uc_tech_list:
                    for tech in self._uc_tech_list:
                        # Cross-hour startup costs (chronologically correct)
                        for (i1, i2) in hour_transitions_by_period[period]:
                            total += (
                                m.v_cross_h[tech, period, i1, i2]
                                * m.c_startup[tech]
                                * m.hour_trans_count[period, i1, i2]
                                * pv_annual
                            )
                        # Ramp costs (rep-hour sequence proxy)
                        for i in m.REP_HOURS:
                            total += m.ramp_up[tech, period, i] * m.c_ramp[tech] * pv_annual

            # ── CAPEX: lump-sum discounted to base year (tech + storage + BTES) ──
            for pp in periods_list:
                df_pp = 1.0 / (1.0 + self.discount_rate_by_period[pp]) ** (pp - base_yr)
                for tech in m.TECH:
                    total += m.cap_add[tech, pp] * m.capex[tech] * df_pp
                if stor_names:
                    for s in m.STOR:
                        total += m.stor_cap_add[s, pp] * m.stor_capex[s] * df_pp

            if self.btes_enabled:
                total += btes_capex_lump
                # TODO: add BTES fixed OPEX (€/yr × ξ_p for each active period)
                # TODO: add BTES variable OPEX / pumping energy (proportional to Qch + Qdis)

            for p in periods_list:
                for tech in m.TECH:
                    lt = int(m.lifetime[tech])
                    frac = linear_salvage_fraction_of_capex(p, lt, horizon_end)
                    if frac > 0:
                        total -= (
                            m.cap_add[tech, p]
                            * m.capex[tech]
                            * frac
                            * df_salv
                        )
                if stor_names:
                    for s in m.STOR:
                        lt = int(m.stor_lifetime[s])
                        frac = linear_salvage_fraction_of_capex(p, lt, horizon_end)
                        if frac > 0:
                            total -= (
                                m.stor_cap_add[s, p]
                                * m.stor_capex[s]
                                * frac
                                * df_salv
                            )

            # ── BTES salvage terminal credit ───────────────────────────
            total -= btes_salv_credit

            return total

        m.total_cost = pyo.Objective(rule=objective_rule, sense=pyo.minimize)

        # ════════════════════════════════════════════════════════════
        #  CONSTRAINTS
        # ════════════════════════════════════════════════════════════

        # 1. Capacity balance (with optional retirement)
        def cap_balance_rule(m, tech, period):
            p_list = sorted(self.investment_years)
            lt = int(m.lifetime[tech])
            if self.retirement_enabled:
                surviving_existing = self.existing_remaining[(tech, period)]
                surviving_additions = sum(
                    m.cap_add[tech, pp]
                    for pp in p_list if pp <= period and period < pp + lt
                )
                return m.cap_total[tech, period] == surviving_existing + surviving_additions
            else:
                p_idx = p_list.index(period)
                if p_idx == 0:
                    return m.cap_total[tech, period] == (
                        m.existing_cap[tech] + m.cap_add[tech, period]
                    )
                else:
                    prev = p_list[p_idx - 1]
                    return m.cap_total[tech, period] == (
                        m.cap_total[tech, prev] + m.cap_add[tech, period]
                    )

        m.cap_balance = pyo.Constraint(m.TECH, m.PERIODS, rule=cap_balance_rule)

        # 2. Max capacity
        def max_cap_rule(m, tech, period):
            limit = m.max_new_cap[tech]
            if limit <= 0:
                # 0 means unlimited potential
                return pyo.Constraint.Skip
            return m.cap_total[tech, period] <= limit

        m.max_cap = pyo.Constraint(m.TECH, m.PERIODS, rule=max_cap_rule)

        # 3. Energy balance per representative hour
        if self.allow_reinjection:
            m.reinject = pyo.Var(m.PERIODS, m.REP_HOURS, domain=pyo.NonNegativeReals)

        def energy_balance_rule(m, period, i):
            gen_total = sum(m.gen[tech, period, i] for tech in m.TECH)
            if stor_names:
                for s in m.STOR:
                    gen_total += m.discharge[s, period, i]
                    gen_total -= m.charge[s, period, i]
            if self.btes_enabled:
                gen_total += (m.btes_Qdis[period, i] - m.btes_Qch[period, i]) / 1000.0
            if self.allow_reinjection:
                # gen = demand + reinject  ->  unused must-run heat goes back down the well
                return gen_total == m.demand[period, i] + m.reinject[period, i]
            return gen_total == m.demand[period, i]

        m.energy_balance = pyo.Constraint(
            m.PERIODS, m.REP_HOURS, rule=energy_balance_rule
        )

        # 4. Generation ≤ capacity (linear; Big-M on/off added in UC block below)
        def gen_limit_rule(m, tech, period, i):
            return m.gen[tech, period, i] <= m.cap_total[tech, period] * m.restriction_factor[tech, period, i]
        m.gen_limit = pyo.Constraint(m.TECH, m.PERIODS, m.REP_HOURS, rule=gen_limit_rule)

        if self.uc_enabled and self._uc_tech_list:
            # Big-M on/off: gen ≤ uc_bigM * rf * u  → forces gen=0 when u=0
            # gen_limit tightens this to cap_total*rf when u=1; fully linear.
            def gen_uc_bigm_rule(m, tech, period, i):
                return m.gen[tech, period, i] <= (
                    m.uc_bigM * m.restriction_factor[tech, period, i] * m.u[tech, period, i]
                )
            m.gen_uc_bigm = pyo.Constraint(m.UC_TECH, m.PERIODS, m.REP_HOURS, rule=gen_uc_bigm_rule)

        # Build predecessor lookup per period: for each rep hour i2, which i1s
        # transition to it?  Used in min_load relaxation to avoid infeasibility
        # with multiple predecessors.  Per-period because clustering differs.
        predecessors_by_period: dict = {}   # period → {i2: [i1, …]}
        for p in self.investment_years:
            pred_p: dict = {}
            for (i1, i2) in hour_transitions_by_period[p]:
                pred_p.setdefault(i2, []).append(i1)
            predecessors_by_period[p] = pred_p

        # Unit commitment constraints (MILP only, UC techs only)
        if self.uc_enabled and self._uc_tech_list:

            # ── cross-hour startup: v_cross_h[i1,i2] ≥ u[i2] − u[i1] ──────
            # Fires whenever the unit is OFF at a rep-hour of type i1 but ON
            # at the immediately following rep-hour of type i2 in actual time.
            def cross_startup_h_rule(m, tech, period, i1, i2):
                return m.v_cross_h[tech, period, i1, i2] >= (
                    m.u[tech, period, i2] - m.u[tech, period, i1]
                )
            m.cross_startup_h = pyo.Constraint(
                m.UC_TECH, m.PERIOD_HOUR_TRANS, rule=cross_startup_h_rule
            )

            # startup upper bounds — tighten LP relaxation
            def cross_startup_ub_u_rule(m, tech, period, i1, i2):
                return m.v_cross_h[tech, period, i1, i2] <= m.u[tech, period, i2]
            m.cross_startup_ub_u = pyo.Constraint(
                m.UC_TECH, m.PERIOD_HOUR_TRANS, rule=cross_startup_ub_u_rule
            )

            def cross_startup_ub_uprev_rule(m, tech, period, i1, i2):
                return m.v_cross_h[tech, period, i1, i2] <= 1 - m.u[tech, period, i1]
            m.cross_startup_ub_uprev = pyo.Constraint(
                m.UC_TECH, m.PERIOD_HOUR_TRANS, rule=cross_startup_ub_uprev_rule
            )

            # ── cold-start output: gen = 0 during warm-up hour ───────────────
            # If v_cross_h[i1,i2] = 1, all actual-hour transitions of type
            # (i1→i2) start cold: suppress output at rep-hour i2.
            # Multiple predecessor constraints per i2 are each valid upper bounds;
            # the tightest (v_cross_h=1) dominates without causing infeasibility.
            def cross_cold_start_h_rule(m, tech, period, i1, i2):
                tau = cold_start_dict.get(tech, 0)
                if tau <= 0:
                    return pyo.Constraint.Skip
                # Linearised: gen ≤ uc_bigM*(u - v_cross_h); gen_limit tightens to cap*rf.
                return m.gen[tech, period, i2] <= m.uc_bigM * (
                    m.u[tech, period, i2] - m.v_cross_h[tech, period, i1, i2]
                )
            m.cross_cold_start_h = pyo.Constraint(
                m.UC_TECH, m.PERIOD_HOUR_TRANS, rule=cross_cold_start_h_rule
            )

            # ── min-load: relaxed when any predecessor triggers a cold start ──
            # Sum of all v_cross_h into i relaxes the lower bound so min-load
            # and the cold-start upper bound remain jointly feasible.
            def min_load_rule(m, tech, period, i):
                mlf = m.min_load_factor[tech]
                if mlf <= 0:
                    return pyo.Constraint.Skip
                tau = cold_start_dict.get(tech, 0)
                cap = m.cap_total[tech, period] * m.restriction_factor[tech, period, i]
                M = m.uc_bigM
                pred_p = predecessors_by_period[period]
                # Linearised Big-M: gen >= mlf*cap - mlf*M*(1 - u + pred_sum)
                if tau > 0 and i in pred_p:
                    pred_sum = sum(
                        m.v_cross_h[tech, period, i1, i]
                        for i1 in pred_p[i]
                    )
                    return m.gen[tech, period, i] >= mlf * (
                        cap - M * (1 - m.u[tech, period, i] + pred_sum)
                    )
                return m.gen[tech, period, i] >= mlf * (
                    cap - M * (1 - m.u[tech, period, i])
                )
            m.min_load = pyo.Constraint(m.UC_TECH, m.PERIODS, m.REP_HOURS, rule=min_load_rule)

            def ramp_up_rule(m, tech, period, i):
                # Wrap-around: i=0 uses gen[I-1] as previous (circular year).
                i_prev = I - 1 if i == 0 else i - 1
                return m.ramp_up[tech, period, i] >= m.gen[tech, period, i] - m.gen[tech, period, i_prev]
            m.ramp_up_def = pyo.Constraint(m.UC_TECH, m.PERIODS, m.REP_HOURS, rule=ramp_up_rule)

        # 5. Annual energy potential
        def annual_potential_rule(m, tech, period):
            potential = m.max_potential[tech]
            # 0 or NaN means unlimited annual energy potential
            if potential <= 0 or np.isnan(float(potential)):
                return pyo.Constraint.Skip
            annual_gen = sum(
                m.gen[tech, period, i] * m.weight[period, i] for i in m.REP_HOURS
            )
            return annual_gen <= potential

        m.annual_potential = pyo.Constraint(
            m.TECH, m.PERIODS, rule=annual_potential_rule
        )

        # ════════════════════════════════════════════════════════════
        #  STORAGE — Reduced RH representation (paper Eq. 7–11)
        # ════════════════════════════════════════════════════════════
        if stor_names:
            # Storage capacity balance (with optional retirement)
            def stor_cap_balance_rule(m, s, period):
                p_list = sorted(self.investment_years)
                lt = int(m.stor_lifetime[s])
                if self.stor_retirement_enabled:
                    surviving_existing = self.stor_existing_remaining[(s, period)]
                    surviving_additions = sum(
                        m.stor_cap_add[s, pp]
                        for pp in p_list if pp <= period and period < pp + lt
                    )
                    return m.stor_cap_total[s, period] == surviving_existing + surviving_additions
                else:
                    p_idx = p_list.index(period)
                    if p_idx == 0:
                        return m.stor_cap_total[s, period] == (
                            m.stor_cap_existing[s] + m.stor_cap_add[s, period]
                        )
                    else:
                        prev = p_list[p_idx - 1]
                        return m.stor_cap_total[s, period] == (
                            m.stor_cap_total[s, prev] + m.stor_cap_add[s, period]
                        )

            m.stor_cap_balance = pyo.Constraint(m.STOR, m.PERIODS, rule=stor_cap_balance_rule)

            # Maximum total storage capacity (0 = unlimited)
            m.stor_max_cap_param = pyo.Param(
                m.STOR, initialize={s: stor[s].get("max_capacity_MWh", 0.0) for s in stor_names}
            )
            def stor_max_cap_rule(m, s, period):
                limit = m.stor_max_cap_param[s]
                if limit <= 0:
                    return pyo.Constraint.Skip
                return m.stor_cap_total[s, period] <= limit
            m.stor_max_cap_con = pyo.Constraint(m.STOR, m.PERIODS, rule=stor_max_cap_rule)

            # Charge / discharge power limits
            def charge_limit_rule(m, s, period, i):
                return m.charge[s, period, i] <= m.stor_cap_total[s, period] / m.tau_ch[s]

            def discharge_limit_rule(m, s, period, i):
                return m.discharge[s, period, i] <= m.stor_cap_total[s, period] / m.tau_dis[s]

            m.charge_limit = pyo.Constraint(
                m.STOR, m.PERIODS, m.REP_HOURS, rule=charge_limit_rule
            )
            m.discharge_limit = pyo.Constraint(
                m.STOR, m.PERIODS, m.REP_HOURS, rule=discharge_limit_rule
            )

            # Storage dynamics on reduced index J
            # soc[j] = soc[j-1] * (1-φ)^dur[j]
            #         + ΔH[rep(j)] * Σ_{k=0..dur[j]-1} (1-φ)^k
            #
            # The geometric sum Σ (1-φ)^k from k=0..d-1 = (1-(1-φ)^d)/φ
            def stor_dyn_rule(m, s, period, j):
                phi = m.stor_loss[s]
                rep_i = m.j_rep[period, j]
                dur = m.j_dur[period, j]

                # Net charging flow
                delta_h = (
                    m.charge[s, period, rep_i] * m.stor_eff_ch[s]
                    - m.discharge[s, period, rep_i] / m.stor_eff_dis[s]
                )

                # Geometric sum for self-discharge within the interval
                if phi > 1e-10:
                    geo_sum = (1 - (1 - phi) ** dur) / phi
                else:
                    geo_sum = dur  # No losses → just multiply by duration

                decay = (1 - phi) ** dur

                if j == 0:
                    # Periodicity: link to last storage level of THIS period
                    prev_soc = m.soc[s, period, J_by_period[period] - 1]
                else:
                    prev_soc = m.soc[s, period, j - 1]

                return m.soc[s, period, j] == prev_soc * decay + delta_h * geo_sum

            m.stor_dyn = pyo.Constraint(
                m.STOR, m.PERIOD_J, rule=stor_dyn_rule
            )

            # SOC capacity limits (only per J — the key reduction!)
            def soc_ub_rule(m, s, period, j):
                return m.soc[s, period, j] <= m.stor_cap_total[s, period]

            m.soc_ub = pyo.Constraint(
                m.STOR, m.PERIOD_J, rule=soc_ub_rule
            )

        model_type = "MILP" if self.uc_enabled else "LP"
        logger.info(
            f"RH {model_type} model built: {m.nvariables()} variables, "
            f"{m.nconstraints()} constraints"
        )

    # ──────────────────────────────────────────────────────────────────
    #  SOLVE
    # ──────────────────────────────────────────────────────────────────
    def solve(self, solver_name="gurobi", tee=True, **solver_options):
        logger.info(f"Solving RH model with {solver_name} …")
        solver = SolverFactory(solver_name)
        solver.options["MIPGap"] = solver_options.get("MIPGap", 0.01)
        for k, v in solver_options.items():
            solver.options[k] = v
        results = solver.solve(self.model, tee=tee)
        status = results.solver.termination_condition
        logger.info(f"RH solver finished: {status}")
        return results

    # ──────────────────────────────────────────────────────────────────
    #  RESULTS
    # ──────────────────────────────────────────────────────────────────
    def get_results(self, export_dir: str = None) -> Dict:
        """Extract results into a plain dictionary and optionally export to CSV."""
        m = self.model
        import pandas as pd
        from pathlib import Path

        try:
            obj_val = float(pyo.value(m.total_cost))
        except (ValueError, TypeError, ArithmeticError):
            obj_val = float("nan")

        results: Dict[str, Any] = {
            "objective_value": obj_val,
            "capacity_added": {},
            "capacity_total": {},
            "annual_generation": {},
        }

        cap_records = []
        gen_records = []

        def _v(expr, default=0.0):
            try:
                return float(pyo.value(expr))
            except (ValueError, TypeError, ArithmeticError):
                return default

        for tech in m.TECH:
            for period in m.PERIODS:
                cap_add = _v(m.cap_add[tech, period])
                cap_tot = _v(m.cap_total[tech, period])
                results["capacity_added"][(tech, period)] = cap_add
                results["capacity_total"][(tech, period)] = cap_tot
                
                cap_records.append({
                    "technology": tech, "period": period,
                    "capacity_added_MW": cap_add, "capacity_total_MW": cap_tot
                })

                annual = sum(
                    _v(m.gen[tech, period, i]) * _v(m.weight[period, i])
                    for i in m.REP_HOURS
                )
                annual = max(0.0, annual)  # clip solver numerical noise
                results["annual_generation"][(tech, period)] = annual
                gen_records.append({
                    "technology": tech, "period": period,
                    "annual_generation_MWh": annual
                })

        stor_records = []
        if list(m.STOR):
            results["storage_capacity"] = {}
            for s in m.STOR:
                for period in m.PERIODS:
                    cap = _v(m.stor_cap_total[s, period])
                    results["storage_capacity"][(s, period)] = cap
                    stor_records.append({"storage": s, "period": period, "capacity_MWh": cap})

        if self.uc_enabled and hasattr(m, "v_cross_h"):
            results["annual_startups"] = {}
            for tech in m.TECH:
                for period in m.PERIODS:
                    if tech in self._uc_tech_list:
                        results["annual_startups"][(tech, period)] = int(round(
                            sum(
                                _v(m.v_cross_h[tech, period, i1, i2])
                                * _v(m.hour_trans_count[period, i1, i2])
                                for (i1, i2) in self._hour_transitions_by_period[period]
                            )
                        ))
                    else:
                        results["annual_startups"][(tech, period)] = 0

        if export_dir:
            out_path = Path(export_dir)
            out_path.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(cap_records).to_csv(out_path / "capacity_results.csv", index=False)
            pd.DataFrame(gen_records).to_csv(out_path / "generation_results.csv", index=False)
            if stor_records:
                pd.DataFrame(stor_records).to_csv(out_path / "storage_results.csv", index=False)

            with open(out_path / "summary.txt", "w") as f:
                f.write(f"Objective Value (NPV EUR): {results['objective_value']:,.0f}\n")
            logger.info(f"Exported CSV results to {export_dir}")

        return results

    def compute_lcoh(self) -> "pd.DataFrame":
        """
        Levelized Cost of Heat (EUR/MWh) for each investment period.

        Numerator   = annualised CAPEX of ALL installed capacity (existing + new)
                    + fixed OPEX of all installed capacity
                    + variable OPEX + fuel cost + CO2 cost (weight-scaled annual)
                    − CHP electricity revenue
        Denominator = annual heat demand (MWh)

        Each representative hour i contributes m.weight[i] actual hours per year.
        CAPEX is annualised over each asset's full lifetime using CRF(lt, r).
        """
        import pandas as pd
        from utils.economic_calculations import capital_recovery_factor

        m = self.model
        periods = sorted(self.investment_years)
        stor_names = list(m.STOR) if hasattr(m, "STOR") and list(m.STOR) else []

        records = []
        for p in periods:
            r = self.discount_rate_by_period[p]
            # ── Annual heat demand ────────────────────────────────────────
            annual_demand = sum(
                float(pyo.value(m.demand[p, i])) * float(pyo.value(m.weight[p, i]))
                for i in m.REP_HOURS
            )

            # ── Annualised CAPEX of ALL installed capacity in this period ─
            # cap_total includes existing + new investments, minus retired.
            capex_annual = 0.0
            for tech in m.TECH:
                lt = int(pyo.value(m.lifetime[tech]))
                cap = float(pyo.value(m.cap_total[tech, p]))
                if cap > 1e-6:
                    capex_annual += cap * float(pyo.value(m.capex[tech])) * capital_recovery_factor(lt, r)
            for s in stor_names:
                lt = int(pyo.value(m.stor_lifetime[s]))
                cap = float(pyo.value(m.stor_cap_total[s, p]))
                if cap > 1e-6:
                    capex_annual += cap * float(pyo.value(m.stor_capex[s])) * capital_recovery_factor(lt, r)

            # BTES annualised CAPEX (active from build period onwards)
            if self.btes_enabled and p >= self.btes_build_period:
                btes_lt = int(self._btes_arx["lifetime_a"])
                capex_annual += self._btes_arx["capex_EUR"] * capital_recovery_factor(btes_lt, r)

            # ── Fixed OPEX of all installed capacity ──────────────────────
            opex_fix = sum(
                float(pyo.value(m.cap_total[tech, p])) * float(pyo.value(m.opex_fix[tech]))
                for tech in m.TECH
            )
            for s in stor_names:
                opex_fix += (
                    float(pyo.value(m.stor_cap_total[s, p]))
                    * float(pyo.value(m.stor_opex_fix[s]))
                )

            # ── Variable OPEX + fuel + CO2 − CHP revenue (annual) ────────
            var_fuel_co2 = 0.0
            chp_revenue  = 0.0
            for tech in m.TECH:
                carrier = str(pyo.value(m.tech_carrier[tech]))
                eff_el  = float(pyo.value(m.efficiency_el[tech]))
                fp      = float(pyo.value(m.fuel_price[tech, p]))
                ef      = float(pyo.value(m.emission_factor[tech, p]))
                cp      = float(pyo.value(m.carbon_price[p]))
                ov      = float(pyo.value(m.opex_var[tech]))
                for i in m.REP_HOURS:
                    gen_val = float(pyo.value(m.gen[tech, p, i]))
                    if gen_val < 1e-9:
                        continue
                    weight  = float(pyo.value(m.weight[p, i]))
                    eff     = float(pyo.value(m.efficiency[tech, p, i]))
                    if carrier in ("electricity", "ambient heat") or carrier.startswith("waste heat"):
                        p_fuel = float(pyo.value(m.electricity_price[p, i]))
                    else:
                        p_fuel = fp
                    var_fuel_co2 += weight * (
                        gen_val * ov
                        + gen_val / eff * p_fuel
                        + gen_val / eff * ef * cp
                    )
                    if eff_el > 0:
                        chp_revenue += (
                            weight * gen_val * eff_el / eff
                            * float(pyo.value(m.electricity_sell_price[p, i]))
                        )

            total_annual = capex_annual + opex_fix + var_fuel_co2 - chp_revenue
            lcoh = total_annual / annual_demand if annual_demand > 1e-6 else float("nan")

            records.append({
                "period":                p,
                "annual_demand_MWh":     round(annual_demand),
                "capex_annual_EUR":      round(capex_annual),
                "opex_fix_EUR":          round(opex_fix),
                "var_fuel_co2_EUR":      round(var_fuel_co2),
                "chp_revenue_EUR":       round(chp_revenue),
                "total_annual_cost_EUR": round(total_annual),
                "lcoh_EUR_per_MWh":      round(lcoh, 2),
            })

        return pd.DataFrame(records).set_index("period")
