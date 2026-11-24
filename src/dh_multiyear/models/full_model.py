"""
Multi-year district heating model — Full 8760-hour resolution.

No temporal aggregation. Tracks all 8760 hours of the year for each
investment period. Provides the most accurate (ground truth) results
for storage dynamics and dispatch.
"""

import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from loguru import logger
from pathlib import Path

from dh_multiyear.utils.solver import get_solver_name, is_gurobi, translate_options
from dh_multiyear.utils.economic_calculations import (
    planning_horizon_end_year,
    linear_salvage_fraction_of_capex,
    discount_factor_end_of_calendar_year,
)
from dh_multiyear.models.btes_params import period_params_for_build, load_arx_params
from dh_multiyear.models.btes_constraints import add_btes_to_model

def _has_retirement_data(items: Dict[str, Dict[str, Any]]) -> bool:
    """True if at least one item has a non-None installation_year."""
    return any(
        v.get("installation_year") is not None
        for v in items.values()
        if v.get("installed_capacity_MW", v.get("capacity_MWh", 0)) > 0
    )


class DistrictHeatingModelFull:
    """
    Multi-year capacity expansion model using full 8760-hour resolution.
    """

    def __init__(
        self,
        demand_data: pd.DataFrame,
        technologies: Dict[str, Dict[str, Any]],
        storage: Dict[str, Dict[str, Any]],
        prices: Dict[str, Dict[str, float]],
        investment_years: List[int] = None,
        extra_params: Dict[str, float] = None,
        hp_params: Dict[str, Dict[str, Any]] = None,
        uc_params: Dict[str, Dict[str, float]] = None,
        btes_build_period: Optional[int] = None,
        btes_arx_json: Optional[Path] = None,
        efficiency_profiles: Optional[Dict[str, Any]] = None,
        name: str = "KranjDH_Full",
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
        # Optional per-hour efficiency/COP override per tech: {tech_name: array(T)}.
        # Used to inject a dynamic (e.g. SAPHEA temperature-dependent) HTHP COP so the
        # precomputed electricity-cost coefficients reflect it.
        self.efficiency_profiles = efficiency_profiles or {}
        self.investment_years = sorted(investment_years)
        self.discount_rate_by_period = extra_params["discount_rate_by_period"]
        self.carbon_price_by_period = extra_params.get(
            "carbon_price_by_period",
            {yr: 25.0 for yr in investment_years},
        )
        self.name = name
        self.model = pyo.ConcreteModel(name=name)

        # Unit commitment (MILP extension) — gated; None keeps model as pure LP.
        # uc_params: dict mapping tech_name → {c_startup [€/event],
        #                                       c_ramp     [€/MW ramped up],
        #                                       min_load_factor [0–1 fraction of cap]}
        self.uc_params = uc_params or {}
        self.uc_enabled = bool(self.uc_params)
        # Only techs with at least one non-zero UC parameter need binary variables
        self._uc_tech_list = [
            t for t, v in self.uc_params.items()
            if v.get("c_startup", 0) > 0 or v.get("c_ramp", 0) > 0
            or v.get("min_load_factor", 0) > 0 or v.get("cold_start_hours", 0) > 0
            or v.get("min_down_hours", 0) > 0
        ] if self.uc_enabled else []

        self.btes_enabled = btes_build_period is not None
        self.btes_build_period = btes_build_period
        if self.btes_enabled:
            self._btes_arx = load_arx_params(btes_arx_json)
            self._btes_period_params = period_params_for_build(
                sorted(investment_years or [2025, 2030, 2035, 2040, 2045]),
                btes_build_period,
                arx=self._btes_arx,
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

        ret_tag  = " [retirement ON]" if self.retirement_enabled else ""
        uc_tag   = " [UC/MILP ON]"    if self.uc_enabled        else ""
        btes_tag = " [BTES ON]"       if self.btes_enabled      else ""
        logger.info(
            f"Initialised {name}: {len(investment_years)} investment periods, "
            f"8760 hours/year, {len(storage)} storages, {len(technologies)} techs."
            f"{ret_tag}{uc_tag}{btes_tag}"
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

    def build(self):
        logger.info("Building full 8760 model …")
        m = self.model
        techs = self.technologies
        stor = self.storage
        prices = self.prices

        tech_names = list(techs.keys())
        stor_names = list(stor.keys())
        T = 8760

        # ════════════════════════════════════════════════════════════
        #  SETS
        # ════════════════════════════════════════════════════════════
        m.TECH = pyo.Set(initialize=tech_names)
        m.STOR = pyo.Set(initialize=stor_names)
        m.PERIODS = pyo.Set(initialize=self.investment_years, ordered=True)
        m.HOURS = pyo.RangeSet(0, T - 1)

        # ════════════════════════════════════════════════════════════
        #  PARAMETERS
        # ════════════════════════════════════════════════════════════

        # Demand per (period, hour) — uses corrected_demand_{yr} if available
        demand_init = {}
        for period in self.investment_years:
            col = f"corrected_demand_{period}"
            if col in self.demand_data.columns:
                arr = self.demand_data[col].values
            else:
                arr = self.demand_data["heat_demand"].values
            for t in range(T):
                demand_init[period, t] = float(arr[t])
        m.demand = pyo.Param(m.PERIODS, m.HOURS, initialize=demand_init)

        # Electricity price per (period, hour)
        elec_price_dict = {}
        for period in self.investment_years:
            col = f"electricity_price_{period}"
            if col in self.demand_data.columns:
                ep_arr = self.demand_data[col].values
            else:
                ep_arr = next(
                    (self.demand_data[f"electricity_price_{yr}"].values
                     for yr in sorted(self.investment_years)
                     if f"electricity_price_{yr}" in self.demand_data.columns),
                    None,
                )
                if ep_arr is None:
                    ep_arr = np.zeros(T)
                logger.warning(f"No electricity_price_{period} column found, using fallback.")
            for h in range(T):
                elec_price_dict[period, h] = float(ep_arr[h])
        m.electricity_price = pyo.Param(m.PERIODS, m.HOURS, initialize=elec_price_dict)

        # Technology parameters
        m.existing_cap = pyo.Param(
            m.TECH, initialize={t: techs[t]["installed_capacity_MW"] for t in tech_names}
        )
        m.max_new_cap = pyo.Param(
            m.TECH, initialize={t: techs[t]["max_output_MW"] for t in tech_names}
        )
        m.max_potential = pyo.Param(
            m.TECH, initialize={t: techs[t]["max_potential_MWh"] for t in tech_names}
        )
        # Big-M for UC linearisation: 2× system peak demand >> any feasible cap_total
        _bigM = float(self.demand_data["heat_demand"].max()) * 2.0
        m.uc_bigM = pyo.Param(initialize=_bigM)
        m.capex = pyo.Param(
            m.TECH, initialize={t: techs[t]["capex_EUR_per_MW"] for t in tech_names}
        )
        m.opex_fix = pyo.Param(
            m.TECH, initialize={t: techs[t]["opex_fix_EUR_per_MWa"] for t in tech_names}
        )
        m.opex_var = pyo.Param(
            m.TECH, initialize={t: techs[t]["opex_var_EUR_per_MWh"] for t in tech_names}
        )
        m.lifetime = pyo.Param(
            m.TECH, initialize={t: techs[t]["lifetime_a"] for t in tech_names}
        )
        m.efficiency_el = pyo.Param(
            m.TECH, initialize={t: techs[t]["efficiency_el"] for t in tech_names}
        )

        # Ambient and supply/return temperatures
        temp_amb = (self.demand_data["temperature"].values
                    if "temperature" in self.demand_data.columns
                    else np.zeros(8760))
        temp_flow = self.demand_data["temp_supply"].values if "temp_supply" in self.demand_data.columns else np.full(T, 70.0)
        temp_ret = self.demand_data["temp_return"].values if "temp_return" in self.demand_data.columns else np.full(T, 35.0)

        # Dynamic Efficiency and Restriction Factors
        eff_dict = {}
        rf_dict = {}
        for t_name in tech_names:
            carrier = techs[t_name].get("energy_carrier", "")
            if t_name in self.hp_params:
                # Air/Water source HP — COP varies with hourly outdoor temperature
                h = self.hp_params[t_name]
                for t_idx in range(T):
                    cop = (
                        (temp_amb[t_idx] - h["nom_source_temp"]) * h["sens_source"] +
                        (temp_ret[t_idx] - h["nom_return_temp"]) * h["sens_return"] +
                        (temp_flow[t_idx] - h["nom_flow_temp"]) * h["sens_flow"] +
                        techs[t_name]["efficiency_th"]
                    )
                    eff_dict[(t_name, t_idx)] = max(cop, 1e-3)
                    rf = 1.0
                    if temp_amb[t_idx] < h["min_source_temp"]:
                        rf = 0.0
                    elif temp_amb[t_idx] < h["fade_out_temp"]:
                        dx = h["fade_out_temp"] - h["min_source_temp"]
                        dy = temp_amb[t_idx] - h["min_source_temp"]
                        rf = dy / dx
                    if rf <= h["power_min_factor"]:
                        rf = 0.0
                    rf_dict[(t_name, t_idx)] = rf
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
                for t_idx in range(T):
                    t_source = float(src_arr[t_idx])
                    cop = (
                        (t_source - h["nom_source_temp"]) * h["sens_source"] +
                        (temp_ret[t_idx] - h["nom_return_temp"]) * h["sens_return"] +
                        (temp_flow[t_idx] - h["nom_flow_temp"]) * h["sens_flow"] +
                        techs[t_name]["efficiency_th"]
                    )
                    eff_dict[(t_name, t_idx)] = max(cop, 1e-3)
                    rf_dict[(t_name, t_idx)] = 1.0  # waste heat source is always available
            else:
                for t_idx in range(T):
                    eff_dict[(t_name, t_idx)] = techs[t_name]["efficiency_th"]
                    rf_dict[(t_name, t_idx)] = 1.0

        # Per-hour efficiency/COP overrides (e.g. SAPHEA temperature-dependent HTHP COP).
        # Applied before the variable-cost coefficients are precomputed so the electricity
        # cost (elec_price / COP) reflects the dynamic profile.
        for _t, _prof in self.efficiency_profiles.items():
            if _t in tech_names:
                for _h in range(T):
                    eff_dict[(_t, _h)] = float(_prof[_h])
        m.efficiency = pyo.Param(m.TECH, m.HOURS, initialize=eff_dict, mutable=True)
        m.restriction_factor = pyo.Param(m.TECH, m.HOURS, initialize=rf_dict)

        # Plain Python dicts for UC time parameters — safe to use inside Pyomo rules
        cold_start_dict = {
            t: int(self.uc_params.get(t, {}).get("cold_start_hours", 0))
            for t in tech_names
        }
        min_up_dict = {
            t: int(self.uc_params.get(t, {}).get("min_up_hours", 0))
            for t in tech_names
        }
        min_down_dict = {
            t: int(self.uc_params.get(t, {}).get("min_down_hours", 0))
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

        # Fuel prices & emissions
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
            # Variable OPEX on throughput (€/MWh charged+discharged): small value
            # prevents non-physical simultaneous charge/discharge cycling.
            m.stor_opex_var = pyo.Param(
                m.STOR, initialize={s: stor[s].get("opex_var_EUR_per_MWh", 0.0) for s in stor_names}
            )
            m.stor_lifetime = pyo.Param(
                m.STOR, initialize={s: stor[s]["lifetime_a"] for s in stor_names}
            )

        # ════════════════════════════════════════════════════════════
        #  VARIABLES
        # ════════════════════════════════════════════════════════════
        m.cap_add = pyo.Var(m.TECH, m.PERIODS, domain=pyo.NonNegativeReals)
        m.cap_total = pyo.Var(m.TECH, m.PERIODS, domain=pyo.NonNegativeReals)
        m.gen = pyo.Var(m.TECH, m.PERIODS, m.HOURS, domain=pyo.NonNegativeReals)
        # Reinjection slack: must-run geothermal heat not taken by demand/storage is
        # reinjected hot into the well (free). Lets the energy balance hold when the
        # geothermal min-load exceeds the load. (Reported as "reinjection", not waste.)
        m.reinjection = pyo.Var(m.PERIODS, m.HOURS, domain=pyo.NonNegativeReals)

        # Unit commitment binary variables (MILP, only when uc_enabled)
        if self.uc_enabled and self._uc_tech_list:
            m.UC_TECH = pyo.Set(initialize=self._uc_tech_list, within=m.TECH)
            m.u       = pyo.Var(m.UC_TECH, m.PERIODS, m.HOURS, domain=pyo.Binary)           # ON/OFF status
            m.v_start = pyo.Var(m.UC_TECH, m.PERIODS, m.HOURS, domain=pyo.Binary)           # startup event
            m.v_stop  = pyo.Var(m.UC_TECH, m.PERIODS, m.HOURS, domain=pyo.Binary)           # shutdown event
            m.ramp_up = pyo.Var(m.UC_TECH, m.PERIODS, m.HOURS, domain=pyo.NonNegativeReals) # ramp-up MW

        if stor_names:
            m.stor_cap_add = pyo.Var(m.STOR, m.PERIODS, domain=pyo.NonNegativeReals)
            m.stor_cap_total = pyo.Var(m.STOR, m.PERIODS, domain=pyo.NonNegativeReals)
            m.charge = pyo.Var(m.STOR, m.PERIODS, m.HOURS, domain=pyo.NonNegativeReals)
            m.discharge = pyo.Var(m.STOR, m.PERIODS, m.HOURS, domain=pyo.NonNegativeReals)
            m.soc = pyo.Var(m.STOR, m.PERIODS, m.HOURS, domain=pyo.NonNegativeReals)

        # BTES variables + constraints added here so btes_build is available in the objective
        if self.btes_enabled:
            add_btes_to_model(
                m,
                investment_years=self.investment_years,
                T=T,
                period_params=self._btes_period_params,
                arx=self._btes_arx,
                T_sup=(self.demand_data["supply_temp"].values[:T]
                       if "supply_temp" in self.demand_data.columns else None),
                T_ret=(self.demand_data["return_temp"].values[:T]
                       if "return_temp" in self.demand_data.columns else None),
            )
            logger.info("BTES constraints added (Option-B, per-period investment).")

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

        # Precompute per-(tech, period, hour) variable-cost coefficient × pv_annual.
        # Replaces per-hour Pyomo arithmetic in the objective with scalar multiplications,
        # speeding up model build and producing a cleaner coefficient matrix for Gurobi.

        # Period-specific electricity price arrays.
        # Buy price (_elec_arr): HP fuel cost. Sell price (_sell_arr): CHP revenue.
        _elec_arr_by_period = {}
        _sell_arr_by_period = {}
        for _period in self.investment_years:
            _buy_col  = f"electricity_price_{_period}"
            _sell_col = f"sell_electricity_price_{_period}"
            _sorted_yrs = sorted(self.investment_years)
            if _buy_col in self.demand_data.columns:
                _elec_arr_by_period[_period] = self.demand_data[_buy_col].values.astype(float)
            else:
                _elec_arr_by_period[_period] = next(
                    (self.demand_data[f"electricity_price_{yr}"].values.astype(float)
                     for yr in _sorted_yrs
                     if f"electricity_price_{yr}" in self.demand_data.columns),
                    np.zeros(T),
                )
            if _sell_col in self.demand_data.columns:
                _sell_arr_by_period[_period] = self.demand_data[_sell_col].values.astype(float)
            else:
                # Fall back to buy price if no separate sell price column exists
                _sell_arr_by_period[_period] = _elec_arr_by_period[_period]

        # Store for post-solve LCOH computation (not added to Pyomo model)
        self._sell_arr_by_period = _sell_arr_by_period
        self._elec_arr_by_period = _elec_arr_by_period

        _pv = {}   # pv_annual discount sum per period
        _vc = {}   # variable cost coeff per (tech, period, hour), already × pv_annual

        for _p_idx, _period in enumerate(periods_list_obj):
            _yip = periods_list_obj[_p_idx + 1] - _period if _p_idx + 1 < len(periods_list_obj) else 5
            _r_p = self.discount_rate_by_period[_period]
            _pv[_period] = sum(
                1.0 / (1.0 + _r_p) ** (_period + y - base_yr_obj) for y in range(_yip)
            )
            _elec_arr = _elec_arr_by_period[_period]
            _sell_arr = _sell_arr_by_period[_period]
            for _tech in tech_names:
                _carrier = carrier_dict[_tech]
                _fp      = float(fuel_price_dict[(_tech, _period)])
                _ov      = float(techs[_tech]["opex_var_EUR_per_MWh"])
                _ef      = float(emission_dict[(_tech, _period)])
                _cp      = float(carbon_prices[_period])
                _eff_el  = float(techs[_tech]["efficiency_el"])
                for _h in range(T):
                    _eff = float(eff_dict[(_tech, _h)])
                    if _carrier == "electricity" or _carrier == "ambient heat" or _carrier.startswith("waste heat"):
                        _fuel_c = _elec_arr[_h] / _eff
                    else:
                        _fuel_c = _fp / _eff
                    _co2_c    = _ef * _cp / _eff
                    _elec_rev = (_eff_el / _eff * _sell_arr[_h]) if _eff_el > 0.0 else 0.0
                    _vc[(_tech, _period, _h)] = (_fuel_c + _ov + _co2_c - _elec_rev) * _pv[_period]

        def objective_rule(m):
            total = 0.0
            base_yr = base_yr_obj
            periods_list = periods_list_obj

            for p_idx, period in enumerate(periods_list):
                pv_annual = _pv[period]

                # ── Fixed OPEX storage (CAPEX lump-sum outside loop) ─────────
                if stor_names:
                    for s in m.STOR:
                        total += m.stor_cap_total[s, period] * m.stor_opex_fix[s] * pv_annual
                        # Variable OPEX on throughput (charge + discharge): prevents
                        # non-physical simultaneous cycling.
                        total += m.stor_opex_var[s] * pyo.quicksum(
                            m.charge[s, period, h] + m.discharge[s, period, h]
                            for h in range(T)
                        ) * pv_annual

                # Fixed OPEX
                for tech in m.TECH:
                    total += m.cap_total[tech, period] * m.opex_fix[tech] * pv_annual

                # Variable OPEX + fuel + CO2 - CHP revenue (precomputed scalar coefficients)
                for tech in tech_names:
                    total += pyo.quicksum(
                        _vc[tech, period, h] * m.gen[tech, period, h] for h in range(T)
                    )

                # Startup and ramp costs (MILP only, UC techs only)
                if self.uc_enabled and self._uc_tech_list:
                    for tech in self._uc_tech_list:
                        total += pyo.quicksum(
                            m.v_start[tech, period, t] * m.c_startup[tech]
                            + m.ramp_up[tech, period, t] * m.c_ramp[tech]
                            for t in m.HOURS
                        ) * pv_annual

            # ── CAPEX: lump-sum discounted to base year (tech + storage + BTES) ──
            for pp in periods_list:
                df_pp = 1.0 / (1.0 + self.discount_rate_by_period[pp]) ** (pp - base_yr)
                for tech in m.TECH:
                    total += m.cap_add[tech, pp] * m.capex[tech] * df_pp
                if stor_names:
                    for s in m.STOR:
                        total += m.stor_cap_add[s, pp] * m.stor_capex[s] * df_pp

            # ── BTES variable O&M (EUR/MWh charged and discharged) ────
            # Mirrors rd_model: without it the store delivers at zero marginal cost
            # and undercuts geothermal (2.02 EUR/MWh).
            if self.btes_enabled:
                _c_ch = float((self._btes_arx or {}).get("cost_charge", 0.0))
                _c_di = float((self._btes_arx or {}).get("cost_discharge", 0.0))
                if _c_ch or _c_di:
                    for period in periods_list:
                        for t_ in m.HOURS:
                            total += (m.btes_Qch[period, t_] * _c_ch
                                      + m.btes_Qdis[period, t_] * _c_di) / 1000.0 * _pv[period]
                _fx = float((self._btes_arx or {}).get("opex_fix_EUR_per_yr", 0.0))
                if _fx:
                    for period in periods_list:
                        if (self._btes_period_params or {}).get(period, {}).get("active", True):
                            total += _fx * _pv[period]

            if self.btes_enabled:
                total += btes_capex_lump
                # TODO: add BTES fixed OPEX (€/yr × ξ_p for each active period)
                # TODO: add BTES variable OPEX / pumping energy (proportional to Qch + Qdis)

            # Salvage (terminal credit)
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

            # ── BTES salvage terminal credit ───────────────────────────────
            total -= btes_salv_credit

            return total

        m.total_cost = pyo.Objective(rule=objective_rule, sense=pyo.minimize)

        # ════════════════════════════════════════════════════════════
        #  CONSTRAINTS
        # ════════════════════════════════════════════════════════════
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
                    return m.cap_total[tech, period] == (m.existing_cap[tech] + m.cap_add[tech, period])
                else:
                    prev = p_list[p_idx - 1]
                    return m.cap_total[tech, period] == (m.cap_total[tech, prev] + m.cap_add[tech, period])

        m.cap_balance = pyo.Constraint(m.TECH, m.PERIODS, rule=cap_balance_rule)

        def max_cap_rule(m, tech, period):
            limit = m.max_new_cap[tech]
            if limit <= 0: return pyo.Constraint.Skip
            return m.cap_total[tech, period] <= limit
        m.max_cap = pyo.Constraint(m.TECH, m.PERIODS, rule=max_cap_rule)

        def energy_balance_rule(m, period, t):
            gen_total = pyo.quicksum(m.gen[tech, period, t] for tech in m.TECH)
            if stor_names:
                for s in m.STOR:
                    gen_total += m.discharge[s, period, t] - m.charge[s, period, t]
            if self.btes_enabled:
                # kW → MW: discharge adds heat, charge absorbs heat
                gen_total += (m.btes_Qdis[period, t] - m.btes_Qch[period, t]) / 1000.0
            # gen = demand + reinjection (geothermal heat not used is reinjected hot)
            return gen_total == m.demand[period, t] + m.reinjection[period, t]
        m.energy_balance = pyo.Constraint(m.PERIODS, m.HOURS, rule=energy_balance_rule)

        def gen_limit_rule(m, tech, period, t):
            return m.gen[tech, period, t] <= m.cap_total[tech, period] * m.restriction_factor[tech, t]
        m.gen_limit = pyo.Constraint(m.TECH, m.PERIODS, m.HOURS, rule=gen_limit_rule)

        # Unit commitment constraints (MILP only)
        if self.uc_enabled and self._uc_tech_list:

            # Big-M on/off: gen ≤ uc_bigM*rf*u → forces gen=0 when u=0; linear.
            # gen_limit tightens to cap_total*rf when u=1.
            def gen_uc_bigm_rule(m, tech, period, t):
                return m.gen[tech, period, t] <= (
                    m.uc_bigM * m.restriction_factor[tech, t] * m.u[tech, period, t]
                )
            m.gen_uc_bigm = pyo.Constraint(m.UC_TECH, m.PERIODS, m.HOURS, rule=gen_uc_bigm_rule)

            # ── min-load: relaxed during cold-start warm-up ──────────────────
            def min_load_rule(m, tech, period, t):
                mlf = m.min_load_factor[tech]
                if mlf <= 0:
                    return pyo.Constraint.Skip
                tau = cold_start_dict.get(tech, 0)
                cap = m.cap_total[tech, period] * m.restriction_factor[tech, t]
                M = m.uc_bigM
                # Linearised Big-M: gen >= mlf*cap - mlf*M*(1 - u + warm_sum)
                if tau > 0:
                    warm_sum = sum(
                        m.v_start[tech, period, tk]
                        for tk in range(max(0, t - tau + 1), t + 1)
                    )
                    return m.gen[tech, period, t] >= mlf * (
                        cap - M * (1 - m.u[tech, period, t] + warm_sum)
                    )
                return m.gen[tech, period, t] >= mlf * (
                    cap - M * (1 - m.u[tech, period, t])
                )
            m.min_load = pyo.Constraint(m.UC_TECH, m.PERIODS, m.HOURS, rule=min_load_rule)

            def startup_rule(m, tech, period, t):
                # v_start[t] ≥ u[t] - u[t-1]  →  1 when unit turns ON.
                # Wrap-around: t=0 uses u[T-1] as the previous state so the year
                # is treated as circular and no spurious cold start is charged.
                t_prev = T - 1 if t == 0 else t - 1
                return m.v_start[tech, period, t] >= m.u[tech, period, t] - m.u[tech, period, t_prev]
            m.startup_def = pyo.Constraint(m.UC_TECH, m.PERIODS, m.HOURS, rule=startup_rule)

            # ── cold-start output constraint ──────────────────────────────────
            # For tau hours after a startup, generation is forced to zero.
            # Wrap-around: for t < tau-1 include end-of-year startups so a
            # startup at t=T-1 still suppresses output at t=0 (tau=2 case).
            def cold_start_rule(m, tech, period, t):
                tau = cold_start_dict.get(tech, 0)
                if tau <= 0:
                    return pyo.Constraint.Skip
                # Circular index: collect tau slots ending at t
                warm_sum = sum(
                    m.v_start[tech, period, (t - tau + 1 + k) % T]
                    for k in range(tau)
                )
                return m.gen[tech, period, t] <= m.uc_bigM * (
                    m.u[tech, period, t] - warm_sum
                )
            m.cold_start = pyo.Constraint(m.UC_TECH, m.PERIODS, m.HOURS, rule=cold_start_rule)

            def ramp_up_rule(m, tech, period, t):
                # ramp_up[t] ≥ gen[t] - gen[t-1]  →  captures positive output changes.
                # Wrap-around: t=0 uses gen[T-1] as the previous value.
                t_prev = T - 1 if t == 0 else t - 1
                return m.ramp_up[tech, period, t] >= m.gen[tech, period, t] - m.gen[tech, period, t_prev]
            m.ramp_up_def = pyo.Constraint(m.UC_TECH, m.PERIODS, m.HOURS, rule=ramp_up_rule)

            # ── startup upper bounds (tighten LP relaxation) ──────────────────
            # Full 3-constraint formulation: lower bound already in startup_def.
            def startup_ub_u_rule(m, tech, period, t):
                return m.v_start[tech, period, t] <= m.u[tech, period, t]
            m.startup_ub_u = pyo.Constraint(m.UC_TECH, m.PERIODS, m.HOURS, rule=startup_ub_u_rule)

            def startup_ub_uprev_rule(m, tech, period, t):
                t_prev = T - 1 if t == 0 else t - 1
                return m.v_start[tech, period, t] <= 1 - m.u[tech, period, t_prev]
            m.startup_ub_uprev = pyo.Constraint(m.UC_TECH, m.PERIODS, m.HOURS, rule=startup_ub_uprev_rule)

            # ── shutdown event definition ─────────────────────────────────────
            def stop_rule(m, tech, period, t):
                t_prev = T - 1 if t == 0 else t - 1
                return m.v_stop[tech, period, t] >= m.u[tech, period, t_prev] - m.u[tech, period, t]
            m.stop_def = pyo.Constraint(m.UC_TECH, m.PERIODS, m.HOURS, rule=stop_rule)

            def stop_ub_uprev_rule(m, tech, period, t):
                t_prev = T - 1 if t == 0 else t - 1
                return m.v_stop[tech, period, t] <= m.u[tech, period, t_prev]
            m.stop_ub_uprev = pyo.Constraint(m.UC_TECH, m.PERIODS, m.HOURS, rule=stop_ub_uprev_rule)

            def stop_ub_u_rule(m, tech, period, t):
                return m.v_stop[tech, period, t] <= 1 - m.u[tech, period, t]
            m.stop_ub_u = pyo.Constraint(m.UC_TECH, m.PERIODS, m.HOURS, rule=stop_ub_u_rule)

            # ── minimum up-time ───────────────────────────────────────────────
            # Any startup in the window [t-MUT+1, t] forces u[t]=1.
            def min_up_time_rule(m, tech, period, t):
                mut = min_up_dict.get(tech, 0)
                if mut <= 1:
                    return pyo.Constraint.Skip
                return pyo.quicksum(
                    m.v_start[tech, period, (t - mut + 1 + k) % T]
                    for k in range(mut)
                ) <= m.u[tech, period, t]
            m.min_up_time = pyo.Constraint(m.UC_TECH, m.PERIODS, m.HOURS, rule=min_up_time_rule)

            # ── minimum down-time ─────────────────────────────────────────────
            # Any shutdown in the window [t-MDT+1, t] forces u[t]=0.
            def min_down_time_rule(m, tech, period, t):
                mdt = min_down_dict.get(tech, 0)
                if mdt <= 1:
                    return pyo.Constraint.Skip
                return pyo.quicksum(
                    m.v_stop[tech, period, (t - mdt + 1 + k) % T]
                    for k in range(mdt)
                ) <= 1 - m.u[tech, period, t]
            m.min_down_time = pyo.Constraint(m.UC_TECH, m.PERIODS, m.HOURS, rule=min_down_time_rule)

        def annual_potential_rule(m, tech, period):
            potential = m.max_potential[tech]
            if potential <= 0 or np.isnan(float(potential)): return pyo.Constraint.Skip
            return pyo.quicksum(m.gen[tech, period, t] for t in m.HOURS) <= potential
        m.annual_potential = pyo.Constraint(m.TECH, m.PERIODS, rule=annual_potential_rule)

        if stor_names:
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

            def charge_limit_rule(m, s, period, t):
                return m.charge[s, period, t] <= m.stor_cap_total[s, period] / m.tau_ch[s]
            m.charge_limit = pyo.Constraint(m.STOR, m.PERIODS, m.HOURS, rule=charge_limit_rule)

            def discharge_limit_rule(m, s, period, t):
                return m.discharge[s, period, t] <= m.stor_cap_total[s, period] / m.tau_dis[s]
            m.discharge_limit = pyo.Constraint(m.STOR, m.PERIODS, m.HOURS, rule=discharge_limit_rule)

            def stor_dyn_rule(m, s, period, t):
                phi = m.stor_loss[s]
                net = m.charge[s, period, t] * m.stor_eff_ch[s] - m.discharge[s, period, t] / m.stor_eff_dis[s]
                if t == 0:
                    return m.soc[s, period, t] == m.soc[s, period, T-1] * (1-phi) + net
                else:
                    return m.soc[s, period, t] == m.soc[s, period, t-1] * (1-phi) + net
            m.stor_dyn = pyo.Constraint(m.STOR, m.PERIODS, m.HOURS, rule=stor_dyn_rule)

            def soc_ub_rule(m, s, period, t):
                return m.soc[s, period, t] <= m.stor_cap_total[s, period]
            m.soc_ub = pyo.Constraint(m.STOR, m.PERIODS, m.HOURS, rule=soc_ub_rule)

        model_type = "MILP" if (self.uc_enabled or self.btes_enabled) else "LP"
        logger.info(f"Full {model_type} model built: {m.nvariables()} variables, {m.nconstraints()} constraints")

    def solve(self, solver_name=None, tee=True, warmstart=False, **solver_options):
        """Solve; solver_name defaults to the SOLVER environment variable.

        Options are given in Gurobi names; for other solvers only the MIP gap and
        the time limit are passed on (see utils/solver.py).
        """
        solver_name = solver_name or get_solver_name()
        logger.info(f"Solving Full model with {solver_name} …")
        solver = SolverFactory(solver_name)
        for k, v in translate_options(solver_name, solver_options).items():
            solver.options[k] = v
        # warmstart is a keyword not every Pyomo solver interface accepts; Gurobi
        # always gets it as before, other solvers only when there is a start.
        warm_kw = {"warmstart": warmstart} if (is_gurobi(solver_name) or warmstart) else {}
        results = None
        try:
            results = solver.solve(self.model, tee=tee, load_solutions=False, **warm_kw)
        except KeyboardInterrupt:
            logger.warning("Keyboard interrupt received.")
        if results is not None:
            condition = results.solver.termination_condition
            logger.info(f"Full solver finished: {condition}")
            if len(results.solution) > 0:
                try:
                    self.model.solutions.load_from(results)
                except Exception as exc:
                    # interrupted solves come back as "aborted"; load_from raises on
                    # that even when an incumbent exists
                    logger.warning(f"Could not load the solution ({exc}). Any "
                                   f"incumbent written via SolFiles is on disk.")
                try:
                    gap = float(results.solution(0).gap)
                    if gap is not None:
                        logger.info(f"Best incumbent MIP gap: {gap:.4%}")
                except Exception:
                    pass
                logger.info("Best incumbent solution loaded.")
            else:
                logger.warning(
                    f"No feasible solution available (status: {condition}). "
                    "Try increasing TimeLimit or relaxing MIPGap."
                )
        else:
            logger.warning("No results object — Gurobi was interrupted before returning. "
                           "Variable values may be zero/uninitialized.")
        return results

    def get_results(self, export_dir: str = None) -> Dict:
        m = self.model

        def _v(expr, default=0.0):
            try:
                return float(pyo.value(expr))
            except (ValueError, TypeError, ArithmeticError):
                return default

        try:
            obj_val = float(pyo.value(m.total_cost))
        except (ValueError, TypeError, ArithmeticError):
            obj_val = float("nan")

        results: Dict[str, Any] = {
            "objective_value": obj_val,
            "capacity_total": {},
            "annual_generation": {},
        }
        for tech in m.TECH:
            for period in m.PERIODS:
                results["capacity_total"][(tech, period)] = _v(m.cap_total[tech, period])
                results["annual_generation"][(tech, period)] = max(0.0, sum(
                    _v(m.gen[tech, period, t]) for t in m.HOURS
                ))
        if hasattr(m, "STOR"):
            results["storage_capacity"] = {}
            for s in m.STOR:
                for period in m.PERIODS:
                    results["storage_capacity"][(s, period)] = _v(m.stor_cap_total[s, period])

        if self.uc_enabled and hasattr(m, "v_start"):
            results["annual_startups"] = {}
            for tech in m.TECH:
                for period in m.PERIODS:
                    if self._uc_tech_list and tech in self._uc_tech_list:
                        results["annual_startups"][(tech, period)] = int(
                            round(sum(_v(m.v_start[tech, period, t]) for t in m.HOURS))
                        )
                    else:
                        results["annual_startups"][(tech, period)] = 0

        return results

    def compute_lcoh(self) -> "pd.DataFrame":
        """
        Levelized Cost of Heat (EUR/MWh) for each investment period.

        Numerator   = annualised CAPEX of all active investments (new only)
                    + fixed OPEX of all installed capacity
                    + variable OPEX + fuel cost + CO2 cost (annual sum)
                    − CHP electricity revenue
        Denominator = annual heat demand (MWh, 8760-hour sum)

        CAPEX is annualised over each asset's full lifetime using CRF(lt, r).
        """
        import pandas as pd
        from dh_multiyear.utils.economic_calculations import capital_recovery_factor

        m = self.model
        periods = sorted(self.investment_years)
        stor_names = list(m.STOR) if hasattr(m, "STOR") and list(m.STOR) else []
        T = int(m.HOURS.last()) + 1  # 8760

        records = []
        for p in periods:
            r = self.discount_rate_by_period[p]

            # ── Annual heat demand ────────────────────────────────────────
            annual_demand = sum(float(pyo.value(m.demand[p, h])) for h in m.HOURS)

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
            sell_arr = self._sell_arr_by_period.get(p, self._elec_arr_by_period.get(p, np.zeros(T)))
            var_fuel_co2 = 0.0
            chp_revenue  = 0.0
            for tech in m.TECH:
                carrier = str(pyo.value(m.tech_carrier[tech]))
                eff_el  = float(pyo.value(m.efficiency_el[tech]))
                fp      = float(pyo.value(m.fuel_price[tech, p]))
                ef      = float(pyo.value(m.emission_factor[tech, p]))
                cp      = float(pyo.value(m.carbon_price[p]))
                ov      = float(pyo.value(m.opex_var[tech]))
                for h in m.HOURS:
                    gen_val = float(pyo.value(m.gen[tech, p, h]))
                    if gen_val < 1e-9:
                        continue
                    eff = float(pyo.value(m.efficiency[tech, h]))
                    if carrier in ("electricity", "ambient heat") or carrier.startswith("waste heat"):
                        p_fuel = float(pyo.value(m.electricity_price[p, h]))
                    else:
                        p_fuel = fp
                    var_fuel_co2 += (
                        gen_val * ov
                        + gen_val / eff * p_fuel
                        + gen_val / eff * ef * cp
                    )
                    if eff_el > 0:
                        chp_revenue += gen_val * eff_el / eff * sell_arr[h]

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
