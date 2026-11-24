"""
Multi-year district heating capacity expansion model (Kranj case).

Uses representative days with Kotzur inter/intra-day superposition
storage method, informed by arXiv 2512.00892.

Key features
------------
- Reads technology / storage / price data from Kranj Excel files
- Existing capacity carried forward; new capacity investment decisions
- Representative-day dispatch with weighted annual cost scaling
- Kotzur superposition storage: intra-day SOC within each rep-day,
  inter-day SOC tracking across the full 365-day sequence
- Fuel costs based on energy carrier prices from Excel
- CHP electricity revenue offset
- CO₂ emission costs
"""

import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from loguru import logger
import os
from pathlib import Path

from dh_multiyear.preprocessing.time_aggregation import (
    create_representative_days_with_mapping,
    create_representative_days_stratified,
)
from dh_multiyear.models.btes_params import (period_params_for_build, load_arx_params,
                               load_design_library, period_params_for_design)
from dh_multiyear.models.btes_constraints import (add_btes_to_rd_model,
                                     add_btes_designs_to_rd_model)
from dh_multiyear.utils.solver import get_solver_name, is_gurobi, translate_options
from dh_multiyear.utils.economic_calculations import (
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


class DistrictHeatingModel:
    """
    Multi-year district heating capacity expansion model for Kranj DH system.

    Parameters
    ----------
    demand_data : pd.DataFrame
        8760-row DataFrame (datetime index) with at least a ``heat_demand`` column (MW).
    technologies : dict
        Output of ``load_kranj_technologies()``.
    storage : dict
        Output of ``load_kranj_storage()``.
    prices : dict
        Output of ``load_energy_prices()``.
    investment_years : list[int]
        Investment milestone years, e.g. [2025, 2030, 2035, 2040, 2045].
    n_typical_days : int
        Number of representative days for TSA.
    discount_rate : float
    carbon_price_base : float
        EUR / tonne CO₂ in the base year.
    carbon_price_escalation : float
        Annual escalation rate for the carbon price.
    extra_params may also set ``electricity_price_escalation``,
    ``natural_gas_price_escalation``, ``biomass_price_escalation`` (applies to
    ``wood chips`` / biomass carriers), and ``electricity_emission_deescalation``
    (annual relative decay of grid electricity EF for HP / electric boiler).
    """

    def __init__(
        self,
        demand_data: pd.DataFrame,
        technologies: Dict[str, Dict[str, Any]],
        storage: Dict[str, Dict[str, Any]],
        prices: Dict[str, Dict[str, float]],
        investment_years: List[int] = None,
        n_typical_days: int = 12,
        extra_params: Dict[str, float] = None,
        hp_params: Dict[str, Dict[str, Any]] = None,
        uc_params: Dict[str, Dict[str, float]] = None,
        btes_build_period: Optional[int] = None,
        btes_arx_json: Optional[Path] = None,
        btes_design_library: Optional[Path] = None,
        btes_force_design: Optional[str] = None,
        stratified_tsa: bool = False,
        cluster_method: str = "hierarchical",
        name: str = "KranjDH",
        allow_reinjection: bool = False,
        clustering_mode: str = "per_period",
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
        self.uc_params = uc_params or {}
        self.uc_enabled = bool(self.uc_params)
        self._uc_tech_list = [
            t for t, v in self.uc_params.items()
            if v.get("c_startup", 0) > 0 or v.get("c_ramp", 0) > 0
            or v.get("min_load_factor", 0) > 0 or v.get("cold_start_hours", 0) > 0
        ] if self.uc_enabled else []
        self.btes_enabled = btes_build_period is not None
        self.btes_build_period = btes_build_period
        self._btes_force_design = btes_force_design
        self._btes_designs = (load_design_library(btes_design_library)
                              if (btes_design_library and btes_build_period) else None)
        if self.btes_enabled and self._btes_designs:
            self._btes_arx = load_arx_params(btes_arx_json)
            self._btes_design_pp = {
                k: period_params_for_design(d, sorted(investment_years),
                                            btes_build_period)
                for k, d in self._btes_designs.items()}
            self._btes_period_params = None
            logger.info(
                f"BTES field design is endogenous: {len(self._btes_designs)} "
                f"candidates ({', '.join(self._btes_designs)} boreholes)")
        elif self.btes_enabled:
            self._btes_arx = load_arx_params(btes_arx_json)
            self._btes_period_params = period_params_for_build(
                sorted(investment_years), btes_build_period, arx=self._btes_arx
            )
        else:
            self._btes_arx = None
            self._btes_period_params = None
        self.investment_years = sorted(investment_years)
        self.n_typical_days = n_typical_days
        self.stratified_tsa = stratified_tsa
        self.cluster_method = cluster_method
        self.clustering_mode = clustering_mode  # "per_period" or "joint"
        self.discount_rate_by_period = extra_params["discount_rate_by_period"]
        self.carbon_price_by_period = extra_params.get(
            "carbon_price_by_period",
            {yr: 25.0 for yr in investment_years},
        )
        self.name = name
        self.allow_reinjection = allow_reinjection
        self.model = pyo.ConcreteModel(name=name)

        self.retirement_enabled = _has_retirement_data(technologies)
        self.stor_retirement_enabled = _has_retirement_data(storage)

        if self.retirement_enabled:
            self._precompute_tech_retirement()
        if self.stor_retirement_enabled:
            self._precompute_stor_retirement()

        # ── time-series aggregation (PER investment period) ──────────
        # Each milestone is clustered INDEPENDENTLY on its OWN demand column
        # (corrected_demand_{p}) and its OWN electricity price (electricity_price_{p}),
        # temperature being shared for now.  This yields, per period, its own
        # representative days, day weights, and day→rep mapping.  Model size is
        # unchanged: still n_typical_days rep-days per period.
        # weightDict equalises the influence of demand and electricity price on
        # the clustering distance (see _norm_weight below).  Column names are
        # kept as-is so rep_data["heat_demand"] still works downstream.
        tsa_fn = (
            create_representative_days_stratified if self.stratified_tsa
            else create_representative_days_with_mapping
        )
        tsa_mode = "stratified" if self.stratified_tsa else "standard"

        # Fallback mean price series (used only when a period lacks its own column)
        _all_price_cols = [f"electricity_price_{p}" for p in self.investment_years
                           if f"electricity_price_{p}" in demand_data.columns]
        _elec_mean = (demand_data[_all_price_cols].mean(axis=1)
                      if _all_price_cols else None)

        def _run_tsa(attrs, label):
            """Cluster on an arbitrary set of profiles.

            ``attrs`` maps column name -> 8760-h series. Every attribute is
            weighted equally: tsam min-max scales each column to [0, 1] and only
            then applies weightDict, so equal weights mean equal influence on the
            clustering distance. That is the balanced choice -- favouring one
            profile necessarily degrades the other (measured on this case at 24
            rep-days: demand/price NRMSE 4.14/3.64 % at equal weight, against
            3.31/5.84 % when demand dominates and 6.67/2.69 % when price does).

            Extra profiles (a second price series, ambient temperature, a
            per-milestone price shape) can simply be added to ``attrs``.
            """
            attrs = {k: v for k, v in attrs.items() if v is not None}
            tsa_df = pd.DataFrame(
                {k: np.asarray(v).ravel() for k, v in attrs.items()},
                index=demand_data.index,
            )
            _pw = float(os.environ.get("TSA_PRICE_WEIGHT", 1.0))
            tsam_kwargs: dict = {
                "weightDict": {c: (_pw if c.startswith("elec_price") else 1.0)
                               for c in tsa_df.columns},
            }
            # Peak retention: without it the per-cluster averaging clips the
            # annual peak (~8 % here) and undersizes the peak plant. TSA_PEAK=0
            # disables it for sensitivity runs.
            tsam_kwargs["extreme_periods"] = os.environ.get("TSA_PEAK", "1") == "1"

            logger.info(
                f"Running {tsa_mode} TSA ({label}) on {list(tsa_df.columns)} "
                f"with weights {tsam_kwargs['weightDict']} …")
            return tsa_fn(
                time_series_data=tsa_df, n_periods=n_typical_days,
                hours_per_period=24, method=cluster_method, **tsam_kwargs,
            )

        self.tsa_by_period: Dict[int, dict] = {}
        if self.clustering_mode == "joint":
            # ONE clustering on the pooled (mean-across-milestones) series,
            # SHARED by all periods.  Per-year demand/price values are still
            # aggregated onto this shared structure downstream.
            _dem_cols = [f"corrected_demand_{p}" for p in self.investment_years
                         if f"corrected_demand_{p}" in demand_data.columns]
            dem_series = (demand_data[_dem_cols].mean(axis=1) if _dem_cols
                          else demand_data["heat_demand"])
            _joint = _run_tsa({"heat_demand": dem_series, "elec_price_mean": _elec_mean},
                              "JOINT, pooled across milestones")
            self.tsa_by_period = {p: _joint for p in self.investment_years}
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
                self.tsa_by_period[p] = _run_tsa(
                    {"heat_demand": dem_series, "elec_price_mean": price_series},
                    f"period {p}")

        # Model size is fixed: every period must return the same rep-day count.
        _n_reps = {p: self.tsa_by_period[p]["n_rep_days"]
                   for p in self.investment_years}
        if len(set(_n_reps.values())) != 1:
            raise ValueError(
                f"Per-period clustering returned differing rep-day counts: {_n_reps}. "
                "The model is built on one shared REP_DAYS set, so every period "
                "must yield the same number. This can happen when peak retention "
                "adds a cluster in some milestones but not others -- rerun with "
                "TSA_PEAK=0, or align the peak day across milestones."
            )
        # Legacy single-clustering reference (first period) for any older code.
        self.tsa = self.tsa_by_period[self.investment_years[0]]

        ret_tag  = " [retirement ON]" if self.retirement_enabled else ""
        uc_tag   = " [UC/MILP ON]"    if self.uc_enabled        else ""
        btes_tag = " [BTES ON]"       if self.btes_enabled      else ""
        logger.info(
            f"Initialised {name}: {len(investment_years)} investment periods, "
            f"{n_typical_days} typical days, {len(storage)} storage technologies, "
            f"{len(technologies)} heat generators.{ret_tag}{uc_tag}{btes_tag}"
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
        """Build the Pyomo optimisation model."""
        logger.info("Building model …")
        m = self.model
        tsa = self.tsa
        techs = self.technologies
        stor = self.storage
        prices = self.prices

        tech_names = list(techs.keys())
        stor_names = list(stor.keys())
        n_rep = tsa["n_rep_days"]
        n_days = tsa["n_days"]  # 365

        # ── Per-period rep-day structures (independent clustering) ──
        # For each milestone: day→rep mapping, rep-day weights, cluster members,
        # and cross-day UC transition counts.  Transitions count how many times
        # each rep-day-type transition occurs in that period's chronological
        # 365-day sequence (including year wrap-around).
        day_to_rep_by_period: Dict[int, list] = {}
        weights_by_period: Dict[int, list] = {}
        members_by_period: Dict[int, dict] = {}
        transitions_by_period: Dict[int, dict] = {}
        for p in self.investment_years:
            tsa_p = self.tsa_by_period[p]
            d2r = tsa_p["day_to_rep"]  # list[int], length 365
            day_to_rep_by_period[p] = d2r
            weights_by_period[p] = tsa_p["weights"]
            members_by_period[p] = {
                d_rep: [d for d, r in enumerate(d2r) if r == d_rep]
                for d_rep in range(n_rep)
            }
            trans: dict = {}
            for day in range(n_days - 1):
                pair = (d2r[day], d2r[day + 1])
                trans[pair] = trans.get(pair, 0) + 1
            pair = (d2r[n_days - 1], d2r[0])  # year wrap
            trans[pair] = trans.get(pair, 0) + 1
            transitions_by_period[p] = trans

        self._transitions_by_period = transitions_by_period  # for get_results()
        # Legacy single-period aliases (first period): only for BTES day_to_rep
        # (per-period BTES is out of scope) and older references.
        day_to_rep = day_to_rep_by_period[self.investment_years[0]]
        self._transitions = transitions_by_period[self.investment_years[0]]

        # ════════════════════════════════════════════════════════════
        #  SETS
        # ════════════════════════════════════════════════════════════
        m.TECH = pyo.Set(initialize=tech_names)
        m.STOR = pyo.Set(initialize=stor_names)
        m.PERIODS = pyo.Set(initialize=self.investment_years, ordered=True)
        m.REP_DAYS = pyo.RangeSet(0, n_rep - 1)
        m.HOURS = pyo.RangeSet(0, 23)
        m.DAYS = pyo.RangeSet(0, n_days - 1)  # 0..364 for inter-day storage

        # ════════════════════════════════════════════════════════════
        #  PARAMETERS
        # ════════════════════════════════════════════════════════════

        # -- demand per representative day / hour (MW) --
        # -- demand per (period, rep_day, hour) --
        # Aggregate each period's ACTUAL demand column on the cluster
        # assignment (mean over the days mapped to each rep day), exactly as
        # electricity price and ambient temperature are aggregated below.
        # No scaling assumption: each milestone uses its own demand series, so
        # per-year changes in demand shape are represented, not just magnitude.
        demand_dict = {}
        for period in self.investment_years:
            col = f"corrected_demand_{period}"
            arr = (self.demand_data[col].values if col in self.demand_data.columns
                   else self.demand_data["heat_demand"].values)
            members_p = members_by_period[period]
            for d_rep in range(n_rep):
                orig_days = members_p[d_rep]
                for h in range(24):
                    vals = [arr[od * 24 + h] for od in orig_days]
                    demand_dict[period, d_rep, h] = float(np.mean(vals)) if vals else 0.0
        m.demand = pyo.Param(m.PERIODS, m.REP_DAYS, m.HOURS, initialize=demand_dict)

        # -- electricity price per (period, rep_day, hour) --
        # Average actual hourly prices over all days mapped to each rep_day.
        # Buy price (electricity_price) → HP fuel cost.
        # Sell price (sell_electricity_price) → CHP revenue.
        def _build_price_dict(col_prefix):
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
                        arr = np.zeros(n_days * 24)
                    logger.warning(f"No {col} column found, using fallback.")
                members_p = members_by_period[period]
                for d_rep in range(n_rep):
                    orig_days = members_p[d_rep]
                    for h in range(24):
                        vals = [float(arr[d * 24 + h]) for d in orig_days]
                        price_dict[period, d_rep, h] = float(np.mean(vals)) if vals else 0.0
            return price_dict

        elec_price_dict = _build_price_dict("electricity_price")
        m.electricity_price = pyo.Param(m.PERIODS, m.REP_DAYS, m.HOURS, initialize=elec_price_dict)

        # Sell price: fall back to buy price if sell columns not present
        sell_price_dict = _build_price_dict("sell_electricity_price")
        if not any(f"sell_electricity_price_{p}" in self.demand_data.columns
                   for p in self.investment_years):
            sell_price_dict = elec_price_dict
        m.electricity_sell_price = pyo.Param(
            m.PERIODS, m.REP_DAYS, m.HOURS, initialize=sell_price_dict
        )

        # -- day weights (occurrences) per (period, rep_day) --
        m.day_weight = pyo.Param(
            m.PERIODS, m.REP_DAYS,
            initialize={
                (period, d): float(weights_by_period[period][d])
                for period in self.investment_years
                for d in range(n_rep)
            },
        )

        # -- day_to_rep mapping (original day → rep day) per period --
        m.day_to_rep = pyo.Param(
            m.PERIODS, m.DAYS,
            initialize={
                (period, d): day_to_rep_by_period[period][d]
                for period in self.investment_years
                for d in range(n_days)
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
        # Big-M for UC linearisation: must exceed any feasible cap_total.
        # Use 2× system peak demand — no single tech can ever supply more than this.
        _bigM = float(self.demand_data["heat_demand"].max()) * 2.0
        m.uc_bigM = pyo.Param(initialize=_bigM)

        # Dynamic Efficiency and Restriction Factors
        temp_amb = (self.demand_data["temperature"].values
                    if "temperature" in self.demand_data.columns
                    else np.zeros(8760))
        temp_flow = self.demand_data["temp_supply"].values if "temp_supply" in self.demand_data.columns else np.full(8760, 70.0)
        temp_ret = self.demand_data["temp_return"].values if "temp_return" in self.demand_data.columns else np.full(8760, 35.0)

        # Rep-day membership now differs per period, so the temperature-based
        # COP / restriction factors must be indexed by period as well: each
        # period picks its representative day from its OWN cluster members.
        eff_dict = {}
        rf_dict = {}
        for period in self.investment_years:
            members_p = members_by_period[period]
            for t_name in tech_names:
                carrier = techs[t_name].get("energy_carrier", "")
                if t_name in self.hp_params:
                    # Air/Water source HP — COP varies with hourly outdoor temperature
                    h = self.hp_params[t_name]
                    for d_rep in range(n_rep):
                        orig_days = members_p[d_rep]
                        for hour in range(24):
                            orig_t = orig_days[0] * 24 + hour
                            cop = (
                                (temp_amb[orig_t] - h["nom_source_temp"]) * h["sens_source"] +
                                (temp_ret[orig_t] - h["nom_return_temp"]) * h["sens_return"] +
                                (temp_flow[orig_t] - h["nom_flow_temp"]) * h["sens_flow"] +
                                techs[t_name]["efficiency_th"]
                            )
                            eff_dict[(t_name, period, d_rep, hour)] = max(cop, 1e-3)
                            rf = 1.0
                            if temp_amb[orig_t] < h["min_source_temp"]: rf = 0.0
                            elif temp_amb[orig_t] < h["fade_out_temp"]:
                                dx = h["fade_out_temp"] - h["min_source_temp"]
                                dy = temp_amb[orig_t] - h["min_source_temp"]
                                rf = dy / dx
                            if rf <= h["power_min_factor"]: rf = 0.0
                            rf_dict[(t_name, period, d_rep, hour)] = rf
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
                    for d_rep in range(n_rep):
                        orig_days = members_p[d_rep]
                        for hour in range(24):
                            orig_t = orig_days[0] * 24 + hour
                            t_source = float(src_arr[orig_t])
                            cop = (
                                (t_source - h["nom_source_temp"]) * h["sens_source"] +
                                (temp_ret[orig_t] - h["nom_return_temp"]) * h["sens_return"] +
                                (temp_flow[orig_t] - h["nom_flow_temp"]) * h["sens_flow"] +
                                techs[t_name]["efficiency_th"]
                            )
                            eff_dict[(t_name, period, d_rep, hour)] = max(cop, 1e-3)
                            rf_dict[(t_name, period, d_rep, hour)] = 1.0  # waste heat source always available
                else:
                    for d_rep in range(n_rep):
                        for hour in range(24):
                            eff_dict[(t_name, period, d_rep, hour)] = techs[t_name]["efficiency_th"]
                            rf_dict[(t_name, period, d_rep, hour)] = 1.0

        m.efficiency = pyo.Param(m.TECH, m.PERIODS, m.REP_DAYS, m.HOURS, initialize=eff_dict, mutable=True)
        m.restriction_factor = pyo.Param(m.TECH, m.PERIODS, m.REP_DAYS, m.HOURS, initialize=rf_dict)

        # -- fuel prices & emission factors per (tech, period) --
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
                    # HP fuel cost comes from hourly electricity price in objective;
                    # emission factor uses electricity emission factor per period
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

        # -- storage params --
        if stor_names:
            m.stor_cap_existing = pyo.Param(
                m.STOR,
                initialize={s: stor[s]["capacity_MWh"] for s in stor_names},
            )
            m.stor_loss = pyo.Param(
                m.STOR,
                initialize={s: stor[s]["hourly_loss_frac"] for s in stor_names},
            )
            m.stor_eff_ch = pyo.Param(
                m.STOR,
                initialize={s: stor[s]["charge_eff"] for s in stor_names},
            )
            m.stor_eff_dis = pyo.Param(
                m.STOR,
                initialize={s: stor[s]["discharge_eff"] for s in stor_names},
            )
            m.tau_ch = pyo.Param(
                m.STOR,
                initialize={s: stor[s]["tau_charge_h"] for s in stor_names},
            )
            m.tau_dis = pyo.Param(
                m.STOR,
                initialize={s: stor[s]["tau_discharge_h"] for s in stor_names},
            )
            m.stor_capex = pyo.Param(
                m.STOR,
                initialize={s: stor[s]["capex_EUR_per_MWh"] for s in stor_names},
            )
            m.stor_opex_fix = pyo.Param(
                m.STOR,
                initialize={s: stor[s]["opex_fix_EUR_per_MWh"] for s in stor_names},
            )
            # Variable OPEX on throughput (€/MWh charged + discharged). A small
            # value breaks the simultaneous charge/discharge degeneracy that the
            # free curtailable must-run surplus would otherwise allow.
            m.stor_opex_var = pyo.Param(
                m.STOR,
                initialize={s: stor[s].get("opex_var_EUR_per_MWh", 0.0) for s in stor_names},
            )
            m.stor_lifetime = pyo.Param(
                m.STOR,
                initialize={s: stor[s]["lifetime_a"] for s in stor_names},
            )

        # -- UC parameters (only added if MILP enabled) --
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

        # ════════════════════════════════════════════════════════════
        #  VARIABLES
        # ════════════════════════════════════════════════════════════

        # Capacity investment
        m.cap_add = pyo.Var(m.TECH, m.PERIODS, domain=pyo.NonNegativeReals)
        m.cap_total = pyo.Var(m.TECH, m.PERIODS, domain=pyo.NonNegativeReals)

        # Dispatch (rep-day resolution)
        m.gen = pyo.Var(
            m.TECH, m.PERIODS, m.REP_DAYS, m.HOURS, domain=pyo.NonNegativeReals
        )

        # Reinjection slack (must-run geothermal: when demand < min-load, the unused
        # high-temperature heat is reinjected hot into the well — not curtailed/wasted).
        if self.allow_reinjection:
            m.reinject = pyo.Var(m.PERIODS, m.REP_DAYS, m.HOURS, domain=pyo.NonNegativeReals)

        # Plain Python dict for cold_start_hours — safe to use inside Pyomo rules
        cold_start_dict = {
            t: int(self.uc_params.get(t, {}).get("cold_start_hours", 0))
            for t in tech_names
        }

        # UC binary variables (only if MILP enabled, only for UC techs)
        if self.uc_enabled and self._uc_tech_list:
            m.UC_TECH = pyo.Set(initialize=self._uc_tech_list, within=m.TECH)
            m.u = pyo.Var(
                m.UC_TECH, m.PERIODS, m.REP_DAYS, m.HOURS, domain=pyo.Binary
            )
            m.v_start = pyo.Var(
                m.UC_TECH, m.PERIODS, m.REP_DAYS, m.HOURS, domain=pyo.Binary
            )
            m.ramp_up = pyo.Var(
                m.UC_TECH, m.PERIODS, m.REP_DAYS, m.HOURS, domain=pyo.NonNegativeReals
            )
            # Cross-day transition set and count parameter — PER PERIOD.
            # Each period has its own clustering, hence its own set of
            # (r1, r2) day-type transitions and occurrence counts.  Stored as a
            # sparse 3-index set of (period, r1, r2) triples.
            period_trans_list = [
                (p, r1, r2)
                for p in self.investment_years
                for (r1, r2) in transitions_by_period[p]
            ]
            m.PERIOD_TRANS = pyo.Set(dimen=3, initialize=period_trans_list)
            m.trans_count = pyo.Param(
                m.PERIOD_TRANS,
                initialize={
                    (p, r1, r2): transitions_by_period[p][(r1, r2)]
                    for p in self.investment_years
                    for (r1, r2) in transitions_by_period[p]
                },
            )
            # v_cross[tech, period, r1, r2]: 1 if a cold start occurs at the
            # boundary from rep-day r1 to rep-day r2 (i.e. unit was OFF at end
            # of r1 and turns ON at start of r2), for that period's transitions.
            m.v_cross = pyo.Var(
                m.UC_TECH, m.PERIOD_TRANS, domain=pyo.Binary
            )

        # Storage
        if stor_names:
            m.stor_cap_add = pyo.Var(m.STOR, m.PERIODS, domain=pyo.NonNegativeReals)
            m.stor_cap_total = pyo.Var(m.STOR, m.PERIODS, domain=pyo.NonNegativeReals)
            m.charge = pyo.Var(
                m.STOR, m.PERIODS, m.REP_DAYS, m.HOURS, domain=pyo.NonNegativeReals
            )
            m.discharge = pyo.Var(
                m.STOR, m.PERIODS, m.REP_DAYS, m.HOURS, domain=pyo.NonNegativeReals
            )
            # Intra-day SOC (within each representative day)
            # Must be Reals (not NonNegativeReals): soc_intra is the deviation from
            # the start-of-day level and goes negative when storage discharges.
            # Non-negativity of the TOTAL SOC is enforced by the combined_lb constraint:
            #   soc_inter[day-1] + soc_intra_min[rep(day)] >= 0
            m.soc_intra = pyo.Var(
                m.STOR, m.PERIODS, m.REP_DAYS, m.HOURS, domain=pyo.Reals
            )
            # Inter-day SOC (across 365 days, Kotzur method)
            m.soc_inter = pyo.Var(
                m.STOR, m.PERIODS, m.DAYS, domain=pyo.NonNegativeReals
            )

        # BTES variables + constraints (before objective so variables exist in it)
        if self.btes_enabled and self._btes_designs:
            add_btes_designs_to_rd_model(
                m,
                investment_years=self.investment_years,
                n_days=n_days,
                day_to_rep=day_to_rep,
                designs=self._btes_designs,
                period_params_by_design=self._btes_design_pp,
                build_period=self.btes_build_period,
            )
            logger.info("BTES constraints added to RD model (endogenous design).")
            if self._btes_force_design:
                k = str(self._btes_force_design)
                m.btes_y[k].fix(1)
                logger.info(f"BTES design FORCED to {k} boreholes (validation run)")
        elif self.btes_enabled:
            add_btes_to_rd_model(
                m,
                investment_years=self.investment_years,
                n_days=n_days,
                day_to_rep=day_to_rep,
                period_params=self._btes_period_params,
                arx=self._btes_arx,
                T_sup=(self.demand_data["supply_temp"].values[:8760]
                       if "supply_temp" in self.demand_data.columns else None),
                T_ret=(self.demand_data["return_temp"].values[:8760]
                       if "return_temp" in self.demand_data.columns else None),
            )
            logger.info("BTES constraints added to RD model.")

        # ════════════════════════════════════════════════════════════
        #  OBJECTIVE — minimise NPV of total system cost
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
        self._btes_cost_by_design = None
        if self.btes_enabled and self._btes_designs:
            _df_build = 1.0 / (1.0 + self.discount_rate_by_period[self.btes_build_period]) ** (
                self.btes_build_period - base_yr_obj)
            self._btes_cost_by_design = {}
            for k, d in self._btes_designs.items():
                cx, lt = float(d["capex_EUR"]), int(d["lifetime_a"])
                frac = linear_salvage_fraction_of_capex(self.btes_build_period, lt,
                                                        horizon_end)
                self._btes_cost_by_design[k] = cx * _df_build - (cx * frac * df_salv
                                                                if frac > 0 else 0.0)
                logger.info(
                    f"  BTES design {k} boreholes: CAPEX {cx/1e6:.2f} MEUR -> "
                    f"net PV {self._btes_cost_by_design[k]/1e6:.2f} MEUR")
        elif self.btes_enabled:
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
                # years this period covers
                if p_idx + 1 < len(periods_list):
                    years_in_period = periods_list[p_idx + 1] - period
                else:
                    years_in_period = 5  # last period covers 5 years

                _r_p = self.discount_rate_by_period[period]

                # present-value factor for annual costs over the period
                pv_annual = sum(
                    1 / (1 + _r_p) ** (period + y - base_yr)
                    for y in range(years_in_period)
                )

                # ── Fixed OPEX storage ────────────────────────────────────
                # Storage CAPEX is lump-sum (outside the period loop).
                if stor_names:
                    for s in m.STOR:
                        total += m.stor_cap_total[s, period] * m.stor_opex_fix[s] * pv_annual
                        # Variable OPEX on throughput (charge + discharge): small
                        # cost that prevents non-physical simultaneous cycling.
                        for d_idx in m.REP_DAYS:
                            w_d = m.day_weight[period, d_idx]
                            for h_idx in m.HOURS:
                                total += m.stor_opex_var[s] * (
                                    m.charge[s, period, d_idx, h_idx]
                                    + m.discharge[s, period, d_idx, h_idx]
                                ) * w_d * pv_annual

                # ── Fixed OPEX ───────────────────────────────────────
                for tech in m.TECH:
                    total += m.cap_total[tech, period] * m.opex_fix[tech] * pv_annual

                # Variable OPEX + fuel + CO2 - CHP revenue
                for tech in m.TECH:
                    for d_idx in m.REP_DAYS:
                        weight = m.day_weight[period, d_idx]
                        for h_idx in m.HOURS:
                            gen_val = m.gen[tech, period, d_idx, h_idx]
                            eff_val = m.efficiency[tech, period, d_idx, h_idx]
                            carrier = m.tech_carrier[tech]
                            if carrier == "electricity" or carrier == "ambient heat" or carrier.startswith("waste heat"):
                                p_fuel = m.electricity_price[period, d_idx, h_idx]
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
                                elec_revenue = elec_out * m.electricity_sell_price[period, d_idx, h_idx]

                            total += (fuel_cost + var_cost + co2_cost - elec_revenue) * weight * pv_annual

                # ── UC startup / ramp costs ───────────────────────────
                if self.uc_enabled and self._uc_tech_list:
                    for tech in self._uc_tech_list:
                        # Within-day startups (h > 0 only; h=0 handled by v_cross)
                        for d_idx in m.REP_DAYS:
                            for h_idx in m.HOURS:
                                if h_idx == 0:
                                    continue
                                total += (
                                    m.v_start[tech, period, d_idx, h_idx] * m.c_startup[tech]
                                    + m.ramp_up[tech, period, d_idx, h_idx] * m.c_ramp[tech]
                                ) * m.day_weight[period, d_idx] * pv_annual
                        # Cross-day startups: weighted by number of transitions
                        for (r1, r2) in transitions_by_period[period]:
                            total += (
                                m.v_cross[tech, period, r1, r2] * m.c_startup[tech]
                                * m.trans_count[period, r1, r2] * pv_annual
                            )

                # ── Reinjection penalty (anti-degeneracy: 1 mEUR/MWh) ────
                # Ensures the optimizer reinjects only as much heat as physically required.
                if self.allow_reinjection:
                    _REINJECT_PENALTY = 1e-3  # EUR/MWh
                    for d_idx in m.REP_DAYS:
                        for h_idx in m.HOURS:
                            total += (
                                m.reinject[period, d_idx, h_idx]
                                * _REINJECT_PENALTY
                                * m.day_weight[period, d_idx]
                                * pv_annual
                            )

                # ── BTES variable O&M (EUR/MWh charged and discharged) ────
                # Without this the store delivers heat at zero marginal cost and
                # undercuts the geothermal plant (2.02 EUR/MWh), so it displaces
                # free-flowing geothermal heat as readily as gas.
                if self.btes_enabled:
                    _c_ch = float((self._btes_arx or {}).get("cost_charge", 0.0))
                    _c_di = float((self._btes_arx or {}).get("cost_discharge", 0.0))
                    if _c_ch or _c_di:
                        for d_idx in m.REP_DAYS:
                            for h_idx in m.HOURS:
                                total += (
                                    m.btes_Qch[period, d_idx, h_idx] * _c_ch
                                    + m.btes_Qdis[period, d_idx, h_idx] * _c_di
                                ) / 1000.0 * m.day_weight[period, d_idx] * pv_annual
                    _fx = float((self._btes_arx or {}).get("opex_fix_EUR_per_yr", 0.0))
                    if _fx and (self._btes_period_params or {}).get(period, {}).get("active", True):
                        total += _fx * pv_annual
                    if hasattr(m, "btes_pump_cost"):          # BTES_PUMP_COST tie-breaker (EUR/yr)
                        total += m.btes_pump_cost[period] * pv_annual

                # ── Annual energy potential constraint cost (soft) ────
                # (Handled as hard constraint below, not in objective)

            # ── CAPEX: lump-sum discounted to base year (tech + storage + BTES) ──
            # df_pp = 1/(1+r)^(pp-base) = full-lifetime annuity PV at pp.
            # Eliminates end-of-horizon artifact for all asset types.
            for pp in periods_list:
                df_pp = 1.0 / (1.0 + self.discount_rate_by_period[pp]) ** (pp - base_yr)
                for tech in m.TECH:
                    total += m.cap_add[tech, pp] * m.capex[tech] * df_pp
                if stor_names:
                    for s in m.STOR:
                        total += m.stor_cap_add[s, pp] * m.stor_capex[s] * df_pp

            if self.btes_enabled and self._btes_cost_by_design:
                total += pyo.quicksum(c * m.btes_y[k]
                                      for k, c in self._btes_cost_by_design.items())
            elif self.btes_enabled:
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

        # ── 1. Capacity balance (existing + cumulative additions, with optional retirement) ──
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

        # ── 2. Maximum capacity limit ─────────────────────────────
        def max_cap_rule(m, tech, period):
            limit = m.max_new_cap[tech]
            if limit <= 0:
                # 0 means unlimited potential
                return pyo.Constraint.Skip
            return m.cap_total[tech, period] <= limit

        m.max_cap = pyo.Constraint(m.TECH, m.PERIODS, rule=max_cap_rule)

        # ── 3. Energy balance (per period, rep-day, hour) ──────────
        def energy_balance_rule(m, period, d, h):
            gen_total = sum(m.gen[tech, period, d, h] for tech in m.TECH)
            if stor_names:
                for s in m.STOR:
                    gen_total += m.discharge[s, period, d, h]
                    gen_total -= m.charge[s, period, d, h]
            if self.btes_enabled:
                gen_total += (m.btes_Qdis[period, d, h] - m.btes_Qch[period, d, h]) / 1000.0
            if self.allow_reinjection:
                # gen_total = demand + reinject  →  unused must-run geo heat reinjected hot
                return gen_total == m.demand[period, d, h] + m.reinject[period, d, h]
            return gen_total == m.demand[period, d, h]

        m.energy_balance = pyo.Constraint(
            m.PERIODS, m.REP_DAYS, m.HOURS, rule=energy_balance_rule
        )

        # ── 4. Generation ≤ installed capacity ─────────────────────
        # Linear capacity bound (applies to all techs)
        def gen_limit_rule(m, tech, period, d, h):
            return m.gen[tech, period, d, h] <= m.cap_total[tech, period] * m.restriction_factor[tech, period, d, h]
        m.gen_limit = pyo.Constraint(m.TECH, m.PERIODS, m.REP_DAYS, m.HOURS, rule=gen_limit_rule)

        # ── 4a. UC constraints (only if MILP enabled, UC techs only) ──
        if self.uc_enabled and self._uc_tech_list:

            # ── Big-M on/off: gen = 0 when u = 0 ────────────────────
            # gen ≤ cap_total * rf  (gen_limit, linear)
            # gen ≤ uc_bigM * rf * u  (here, forces gen=0 when u=0, linear)
            # uc_bigM = 2 × peak demand >> any feasible cap_total, so never binding when u=1.
            def gen_uc_bigm_rule(m, tech, period, d, h):
                return m.gen[tech, period, d, h] <= (
                    m.uc_bigM * m.restriction_factor[tech, period, d, h] * m.u[tech, period, d, h]
                )
            m.gen_uc_bigm = pyo.Constraint(m.UC_TECH, m.PERIODS, m.REP_DAYS, m.HOURS, rule=gen_uc_bigm_rule)

            # ── min-load: relaxed during cold-start warm-up ──────────
            # During the tau warm-up hours after a within-day startup,
            # (u - v_start) = 0 so the lower bound drops to 0 automatically,
            # consistent with the cold_start_within constraint below.
            def min_load_rule(m, tech, period, d, h):
                mlf = m.min_load_factor[tech]
                if mlf <= 0:
                    return pyo.Constraint.Skip
                tau = cold_start_dict.get(tech, 0)
                cap = m.cap_total[tech, period]
                M = m.uc_bigM
                # Linearised Big-M: gen >= mlf*cap - mlf*M*(1 - u + warm_sum)
                # Equivalent to gen >= mlf*cap*(u - warm_sum) but fully linear.
                if tau > 0 and h > 0:
                    warm_sum = sum(
                        m.v_start[tech, period, d, hk]
                        for hk in range(max(1, h - tau + 1), h + 1)
                    )
                    return m.gen[tech, period, d, h] >= mlf * (
                        cap - M * (1 - m.u[tech, period, d, h] + warm_sum)
                    )
                return m.gen[tech, period, d, h] >= mlf * (
                    cap - M * (1 - m.u[tech, period, d, h])
                )
            m.min_load = pyo.Constraint(m.UC_TECH, m.PERIODS, m.REP_DAYS, m.HOURS, rule=min_load_rule)

            # ── within-day startup definition (h > 0) ────────────────
            # h = 0 is handled by cross_startup_rule below; skipping it
            # here removes the incorrect cyclic same-day wrap.
            def startup_def_rule(m, tech, period, d, h):
                if h == 0:
                    return pyo.Constraint.Skip
                return m.v_start[tech, period, d, h] >= (
                    m.u[tech, period, d, h] - m.u[tech, period, d, h - 1]
                )
            m.startup_def = pyo.Constraint(m.UC_TECH, m.PERIODS, m.REP_DAYS, m.HOURS, rule=startup_def_rule)

            # within-day startup upper bounds (h > 0 only; h=0 is cross-day)
            def startup_ub_u_rule(m, tech, period, d, h):
                if h == 0:
                    return pyo.Constraint.Skip
                return m.v_start[tech, period, d, h] <= m.u[tech, period, d, h]
            m.startup_ub_u = pyo.Constraint(m.UC_TECH, m.PERIODS, m.REP_DAYS, m.HOURS, rule=startup_ub_u_rule)

            def startup_ub_uprev_rule(m, tech, period, d, h):
                if h == 0:
                    return pyo.Constraint.Skip
                return m.v_start[tech, period, d, h] <= 1 - m.u[tech, period, d, h - 1]
            m.startup_ub_uprev = pyo.Constraint(m.UC_TECH, m.PERIODS, m.REP_DAYS, m.HOURS, rule=startup_ub_uprev_rule)

            # ── cross-day startup: v_cross[tech,period,r1,r2] ≥ u[r2,0] − u[r1,23] ──
            # Fires whenever the unit is OFF at the end of rep-day r1 but ON
            # at the start of rep-day r2, correctly accounting for continuity
            # across ALL chronological day-type transitions.
            def cross_startup_rule(m, tech, period, r1, r2):
                return m.v_cross[tech, period, r1, r2] >= (
                    m.u[tech, period, r2, 0] - m.u[tech, period, r1, 23]
                )
            m.cross_startup = pyo.Constraint(
                m.UC_TECH, m.PERIOD_TRANS, rule=cross_startup_rule
            )

            # cross-day startup upper bounds
            def cross_startup_ub_u_rule(m, tech, period, r1, r2):
                return m.v_cross[tech, period, r1, r2] <= m.u[tech, period, r2, 0]
            m.cross_startup_ub_u = pyo.Constraint(
                m.UC_TECH, m.PERIOD_TRANS, rule=cross_startup_ub_u_rule
            )

            def cross_startup_ub_uprev_rule(m, tech, period, r1, r2):
                return m.v_cross[tech, period, r1, r2] <= 1 - m.u[tech, period, r1, 23]
            m.cross_startup_ub_uprev = pyo.Constraint(
                m.UC_TECH, m.PERIOD_TRANS, rule=cross_startup_ub_uprev_rule
            )

            # ── within-day cold-start output constraint ───────────────
            # For tau warm-up hours after a startup, generation is forced
            # to zero (unit is on but not yet producing useful heat).
            # Only applied for h > 0 (cross-day h=0 is below).
            def cold_start_within_rule(m, tech, period, d, h):
                tau = cold_start_dict.get(tech, 0)
                if tau <= 0 or h == 0:
                    return pyo.Constraint.Skip
                warm_sum = sum(
                    m.v_start[tech, period, d, hk]
                    for hk in range(max(1, h - tau + 1), h + 1)
                )
                # Linearised: gen ≤ M*(u - warm_sum); gen_limit tightens to cap*rf.
                return m.gen[tech, period, d, h] <= m.uc_bigM * (
                    m.u[tech, period, d, h] - warm_sum
                )
            m.cold_start_within = pyo.Constraint(
                m.UC_TECH, m.PERIODS, m.REP_DAYS, m.HOURS, rule=cold_start_within_rule
            )

            # ── cross-day cold-start output constraint (h = 0) ───────
            # If v_cross[r1,r2] = 1, all occurrences of this transition
            # start cold: suppress output at h = 0..tau-1 of rep-day r2.
            # Since all occurrences of (r1→r2) share the same profile,
            # this constraint is consistent across the year.
            def cross_cold_start_rule(m, tech, period, r1, r2):
                tau = cold_start_dict.get(tech, 0)
                if tau <= 0:
                    return pyo.Constraint.Skip
                # h = 0 only (tau > 1 warm-up on later hours handled within-day)
                # Linearised: gen ≤ M*(u - v_cross); gen_limit tightens to cap*rf.
                return m.gen[tech, period, r2, 0] <= m.uc_bigM * (
                    m.u[tech, period, r2, 0] - m.v_cross[tech, period, r1, r2]
                )
            m.cross_cold_start = pyo.Constraint(
                m.UC_TECH, m.PERIOD_TRANS, rule=cross_cold_start_rule
            )

            def ramp_up_def_rule(m, tech, period, d, h):
                prev_gen = m.gen[tech, period, d, h - 1] if h > 0 else m.gen[tech, period, d, 23]
                return m.ramp_up[tech, period, d, h] >= m.gen[tech, period, d, h] - prev_gen
            m.ramp_up_def = pyo.Constraint(m.UC_TECH, m.PERIODS, m.REP_DAYS, m.HOURS, rule=ramp_up_def_rule)
        
        # ── 5. Annual energy potential constraint (CHP / waste heat)
        def annual_potential_rule(m, tech, period):
            potential = m.max_potential[tech]
            # 0 or NaN means unlimited annual energy potential
            if potential <= 0 or np.isnan(float(potential)):
                return pyo.Constraint.Skip
            annual_gen = sum(
                m.gen[tech, period, d, h] * m.day_weight[period, d]
                for d in m.REP_DAYS
                for h in m.HOURS
            )
            return annual_gen <= potential

        m.annual_potential = pyo.Constraint(
            m.TECH, m.PERIODS, rule=annual_potential_rule
        )

        # ════════════════════════════════════════════════════════════
        #  STORAGE CONSTRAINTS — Kotzur superposition method
        # ════════════════════════════════════════════════════════════
        if stor_names:
            # -- Storage capacity balance (with optional retirement)
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
                if not (limit > 0):  # skips NaN, 0, and negatives
                    return pyo.Constraint.Skip
                return m.stor_cap_total[s, period] <= limit
            m.stor_max_cap_con = pyo.Constraint(m.STOR, m.PERIODS, rule=stor_max_cap_rule)

            # -- Charge / discharge power limits --
            def charge_limit_rule(m, s, period, d, h):
                return m.charge[s, period, d, h] <= m.stor_cap_total[s, period] / m.tau_ch[s]

            def discharge_limit_rule(m, s, period, d, h):
                return m.discharge[s, period, d, h] <= m.stor_cap_total[s, period] / m.tau_dis[s]

            m.charge_limit = pyo.Constraint(
                m.STOR, m.PERIODS, m.REP_DAYS, m.HOURS, rule=charge_limit_rule
            )
            m.discharge_limit = pyo.Constraint(
                m.STOR, m.PERIODS, m.REP_DAYS, m.HOURS, rule=discharge_limit_rule
            )

            # -- Intra-day SOC dynamics --
            # soc_intra[s,p,d,h] is the SOC deviation from the start-of-day level
            # at the END of hour h (can be negative when discharging):
            #   soc_intra[h=0] = net[h=0]
            #   soc_intra[h]   = soc_intra[h-1]*(1-φ) + net[h]   for h > 0
            # where net = charge*η_ch − discharge/η_dis
            def intra_soc_rule(m, s, period, d, h):
                phi = m.stor_loss[s]
                net = (
                    m.charge[s, period, d, h] * m.stor_eff_ch[s]
                    - m.discharge[s, period, d, h] / m.stor_eff_dis[s]
                )
                if h == 0:
                    return m.soc_intra[s, period, d, h] == net
                else:
                    return m.soc_intra[s, period, d, h] == (
                        m.soc_intra[s, period, d, h - 1] * (1 - phi) + net
                    )

            m.intra_soc = pyo.Constraint(
                m.STOR, m.PERIODS, m.REP_DAYS, m.HOURS, rule=intra_soc_rule
            )

            # -- Inter-day SOC dynamics (Kotzur, Eq. 23-25) --
            # soc_inter[s,p,day] = soc_inter[s,p,day-1] * (1-φ)^24
            #   + Σ_h [ charge[rep(day),h]*η_ch - discharge[rep(day),h]/η_dis ]
            #     * (1-φ)^(23-h)      ← self-discharge within the day
            def inter_soc_rule(m, s, period, day):
                phi = m.stor_loss[s]
                rd = m.day_to_rep[period, day]  # representative day index
                # net energy added during the day (accounting for intra-day losses)
                daily_net = sum(
                    (
                        m.charge[s, period, rd, h] * m.stor_eff_ch[s]
                        - m.discharge[s, period, rd, h] / m.stor_eff_dis[s]
                    )
                    * (1 - phi) ** (23 - h)
                    for h in m.HOURS
                )
                if day == 0:
                    # periodicity: link to last day
                    prev_soc = m.soc_inter[s, period, n_days - 1]
                else:
                    prev_soc = m.soc_inter[s, period, day - 1]
                return m.soc_inter[s, period, day] == (
                    prev_soc * (1 - phi) ** 24 + daily_net
                )

            m.inter_soc = pyo.Constraint(
                m.STOR, m.PERIODS, m.DAYS, rule=inter_soc_rule
            )

            # -- Combined SOC capacity limit (Kotzur, Eq. 17 / 26) --
            # S_actual(n, h) = soc_inter[n-1] + soc_intra[k(n), h]
            # soc_inter is END-of-day, so the baseline ENTERING day n is soc_inter[day-1].
            # soc_inter[day-1] + soc_intra[rep(day), h] ≤ E_s   ∀ day, h
            # soc_inter[day-1] + soc_intra[rep(day), h] ≥ 0     ∀ day, h
            # We use the reduced MinMax form to keep constraint count manageable.

            # First compute max and min of intra-day SOC per representative day
            # as auxiliary variables.
            m.soc_intra_max = pyo.Var(
                m.STOR, m.PERIODS, m.REP_DAYS, domain=pyo.NonNegativeReals
            )
            m.soc_intra_min = pyo.Var(m.STOR, m.PERIODS, m.REP_DAYS, domain=pyo.Reals)

            def intra_max_rule(m, s, period, d, h):
                return m.soc_intra_max[s, period, d] >= m.soc_intra[s, period, d, h]

            def intra_min_rule(m, s, period, d, h):
                return m.soc_intra_min[s, period, d] <= m.soc_intra[s, period, d, h]

            m.intra_max_con = pyo.Constraint(
                m.STOR, m.PERIODS, m.REP_DAYS, m.HOURS, rule=intra_max_rule
            )
            m.intra_min_con = pyo.Constraint(
                m.STOR, m.PERIODS, m.REP_DAYS, m.HOURS, rule=intra_min_rule
            )

            # Upper bound: soc_inter[day-1] + max_intra[rep(day)] ≤ E_s
            def combined_ub_rule(m, s, period, day):
                rd = m.day_to_rep[period, day]
                prev_day = n_days - 1 if day == 0 else day - 1  # start-of-day baseline
                return (
                    m.soc_inter[s, period, prev_day] + m.soc_intra_max[s, period, rd]
                    <= m.stor_cap_total[s, period]
                )

            m.combined_ub = pyo.Constraint(
                m.STOR, m.PERIODS, m.DAYS, rule=combined_ub_rule
            )

            # Lower bound: soc_inter[day-1] + min_intra[rep(day)] ≥ 0
            def combined_lb_rule(m, s, period, day):
                rd = m.day_to_rep[period, day]
                prev_day = n_days - 1 if day == 0 else day - 1  # start-of-day baseline
                return m.soc_inter[s, period, prev_day] + m.soc_intra_min[s, period, rd] >= 0

            m.combined_lb = pyo.Constraint(
                m.STOR, m.PERIODS, m.DAYS, rule=combined_lb_rule
            )

        logger.info(
            f"Model built: {m.nvariables()} variables, {m.nconstraints()} constraints"
        )

    # ──────────────────────────────────────────────────────────────────
    #  SOLVE
    # ──────────────────────────────────────────────────────────────────
    def solve(self, solver_name=None, tee=True, **solver_options):
        """Solve the optimisation model.

        solver_name defaults to the SOLVER environment variable (see
        utils/solver.py). Options are given in Gurobi names; for other solvers
        only the MIP gap and the time limit are passed on.
        """
        solver_name = solver_name or get_solver_name()
        logger.info(f"Solving with {solver_name} …")
        solver = SolverFactory(solver_name)

        def _set_options(opts):
            for k, v in translate_options(solver_name, opts).items():
                solver.options[k] = v

        _set_options({"MIPGap": solver_options.get("MIPGap", 0.01), **solver_options})
        results = None
        # Two-phase warm start for the discrete discharge-flow levels (env
        # BTES_LEVEL_WARMSTART=1): solve the fixed-flow model first (all level
        # binaries fixed at 0), then release them and continue from that solution
        # as a MIP start. The fixed-flow optimum is feasible for the level model,
        # so the second phase begins with a good incumbent.
        import os as _os
        warm = False
        if _os.environ.get("BTES_LEVEL_WARMSTART", "0") == "1" and hasattr(self.model, "btes_lvl"):
            lvl = self.model.btes_lvl
            for idx in lvl:
                lvl[idx].fix(0)
            ph1 = {"MIPGap": solver_options.get("MIPGap", 0.01), **solver_options}
            ph1["TimeLimit"] = int(_os.environ.get("BTES_WARMSTART_TIMELIMIT", 900))
            ph1["MIPGap"] = float(_os.environ.get("BTES_WARMSTART_GAP", 0.001))
            _set_options(ph1)
            logger.info("Warm-start phase 1: levels fixed at 0 (fixed-flow model)")
            try:
                r1 = solver.solve(self.model, tee=tee, load_solutions=False)
                if len(r1.solution) > 0:
                    self.model.solutions.load_from(r1)
                    warm = True
                    logger.info("Warm-start phase 1 solution loaded; releasing level binaries")
            except KeyboardInterrupt:
                logger.warning("Phase 1 interrupted; continuing to the full solve without a start")
            for idx in lvl:
                lvl[idx].unfix()
                if lvl[idx].value is None:
                    lvl[idx].value = 0
            _set_options({**solver_options, "MIPGap": solver_options.get("MIPGap", 0.01)})
        # warmstart is a keyword not every Pyomo solver interface accepts; Gurobi
        # always gets it as before, other solvers only when there is a start.
        warm_kw = {"warmstart": warm} if (is_gurobi(solver_name) or warm) else {}
        try:
            results = solver.solve(self.model, tee=tee, load_solutions=False, **warm_kw)
        except KeyboardInterrupt:
            logger.warning("Keyboard interrupt received during the solve.")
        # An interrupted solve comes back with status "aborted", and load_from()
        # raises on that even when Gurobi did return an incumbent. Load defensively
        # so a stopped run still reports whatever was found instead of dying here.
        if results is not None:
            try:
                if len(results.solution) > 0:
                    self.model.solutions.load_from(results)
                    logger.info("Solution loaded.")
                else:
                    logger.warning("Solver returned no solution — nothing to load. "
                                   "Reported values will be uninitialised.")
            except Exception as exc:
                logger.warning(f"Could not load the solution ({exc}). Any incumbent "
                               f"Gurobi wrote via SolFiles is still on disk.")
            status = results.solver.termination_condition
            logger.info(f"Solver finished: {status}")
        else:
            logger.warning("No results object returned — the solve was interrupted "
                           "before Gurobi reported back.")
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

        if self.uc_enabled and hasattr(m, "v_start"):
            results["annual_startups"] = {
                (tech, period): int(round(
                    # Within-day startups (h > 0), weighted by day occurrence
                    sum(
                        _v(m.v_start[tech, period, d, h]) * _v(m.day_weight[period, d])
                        for d in m.REP_DAYS
                        for h in m.HOURS
                        if h > 0
                    )
                    # Cross-day startups, weighted by transition occurrence count
                    + sum(
                        _v(m.v_cross[tech, period, r1, r2]) * _v(m.trans_count[period, r1, r2])
                        for (r1, r2) in self._transitions_by_period[period]
                    )
                )) if tech in self._uc_tech_list else 0
                for tech in m.TECH
                for period in m.PERIODS
            }

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

                # Annual generation (MWh) = Σ_d Σ_h gen[d,h]*weight[d]
                annual = sum(
                    _v(m.gen[tech, period, d, h]) * _v(m.day_weight[period, d])
                    for d in m.REP_DAYS
                    for h in m.HOURS
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

    def print_results(self):
        """Pretty-print key results."""
        r = self.get_results()
        print("=" * 70)
        ov = r["objective_value"]
        ov_str = f"{ov:,.0f}" if ov == ov else "n/a (solve failed)"
        print(f"OBJECTIVE (NPV total cost): EUR {ov_str}")
        print("=" * 70)

        periods = sorted(self.investment_years)
        techs = list(self.technologies.keys())

        print("\nCapacity evolution (MW):")
        header = f"{'Period':<8}" + "".join(f"{t[:18]:>20}" for t in techs)
        print(header)
        print("-" * len(header))
        for p in periods:
            row = f"{p:<8}"
            for t in techs:
                cap = r["capacity_total"].get((t, p), 0)
                row += f"{cap:>20.2f}"
            print(row)

        print("\nAnnual generation (MWh):")
        header = f"{'Period':<8}" + "".join(f"{t[:18]:>20}" for t in techs)
        print(header)
        print("-" * len(header))
        for p in periods:
            row = f"{p:<8}"
            for t in techs:
                gen = r["annual_generation"].get((t, p), 0)
                row += f"{gen:>20.0f}"
            print(row)

        if "storage_capacity" in r:
            print("\nStorage capacity (MWh):")
            for s in self.storage:
                for p in periods:
                    cap = r["storage_capacity"].get((s, p), 0)
                    print(f"  {s} @ {p}: {cap:.1f} MWh")

        print("=" * 70)

    def compute_lcoh(self) -> "pd.DataFrame":
        """
        Levelized Cost of Heat (EUR/MWh) for each investment period.

        Numerator   = annualised CAPEX of all active investments (new only)
                    + fixed OPEX of all installed capacity
                    + variable OPEX + fuel cost + CO2 cost (annual weighted)
                    − CHP electricity revenue
        Denominator = annual heat demand (MWh)

        CAPEX is annualised over each asset's full lifetime using CRF(lt, r).
        This is independent of the lump-sum representation in the objective —
        CRF annualisation is the standard LCOH convention in energy planning.
        """
        import pandas as pd
        from dh_multiyear.utils.economic_calculations import capital_recovery_factor

        m = self.model
        periods = sorted(self.investment_years)
        stor_names = list(m.STOR) if hasattr(m, "STOR") and list(m.STOR) else []

        records = []
        for p in periods:
            r = self.discount_rate_by_period[p]
            # ── Annual heat demand ────────────────────────────────────────
            annual_demand = sum(
                float(pyo.value(m.demand[p, d, h])) * float(pyo.value(m.day_weight[p, d]))
                for d in m.REP_DAYS
                for h in m.HOURS
            )

            # ── Annualised CAPEX of ALL installed capacity in this period ─
            # Uses cap_total (existing + new investments, minus retired) so
            # existing pre-horizon assets are included at replacement cost.
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
                for d in m.REP_DAYS:
                    weight = float(pyo.value(m.day_weight[p, d]))
                    for h in m.HOURS:
                        gen_val = float(pyo.value(m.gen[tech, p, d, h]))
                        if gen_val < 1e-9:
                            continue
                        eff = float(pyo.value(m.efficiency[tech, p, d, h]))
                        if carrier in ("electricity", "ambient heat") or carrier.startswith("waste heat"):
                            p_fuel = float(pyo.value(m.electricity_price[p, d, h]))
                        else:
                            p_fuel = float(pyo.value(m.fuel_price[tech, p]))
                        var_fuel_co2 += weight * (
                            gen_val * float(pyo.value(m.opex_var[tech]))
                            + gen_val / eff * p_fuel
                            + gen_val / eff
                            * float(pyo.value(m.emission_factor[tech, p]))
                            * float(pyo.value(m.carbon_price[p]))
                        )
                        if eff_el > 0:
                            chp_revenue += (
                                weight * gen_val * eff_el / eff
                                * float(pyo.value(m.electricity_sell_price[p, d, h]))
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
