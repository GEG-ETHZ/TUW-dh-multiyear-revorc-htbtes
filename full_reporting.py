"""
Adapter: present a solved DistrictHeatingModelFull as an RD-shaped model.

The full 8760-h model IS an RD model with 365 "representative days" of weight 1
and day_to_rep = identity. This module wraps the solved full model so that ALL
run_rd reporting functions (dispatch/SOC/mode plots, winter CSVs, revORC
summary + price analysis, Excel export, computational params) run UNCHANGED on
the full-resolution solution — no reporting code is duplicated.

Mechanics: `_FakeM` proxies attribute access to the real Pyomo model and only
overrides the components whose RD index carries a (rep_day, hour) pair,
translating (…, d, h) → (…, d*24 + h) and returning plain floats (pyo.value on
a float is the identity, so the reporting code never notices).

Usage (run_full.py):
    adapter = FullAsRD(model, demand)
    run_rd.save_dispatch_plots(adapter, results_dir)   # etc.
"""

from __future__ import annotations

import pandas as pd
import pyomo.environ as pyo


class _As4D:
    """(…head[n], d, h) → float(real[(…head, d*24 + h)])."""

    def __init__(self, comp, nlead: int):
        self._c, self._n = comp, nlead

    def __getitem__(self, key):
        head, d, h = key[:self._n], key[self._n], key[self._n + 1]
        return float(pyo.value(self._c[(*head, d * 24 + h)]))


class _EffAs4D:
    """RD efficiency is (tech, period, d, h); full is (tech, hour) — period dropped."""

    def __init__(self, comp):
        self._c = comp

    def __getitem__(self, key):
        t, _p, d, h = key
        return float(pyo.value(self._c[t, d * 24 + h]))


class _Const:
    def __init__(self, v: float):
        self._v = v

    def __getitem__(self, key):
        return self._v


class _SellPrice:
    """RD electricity_sell_price[(p, d, h)] from the full model's stored arrays."""

    def __init__(self, arrs_by_period):
        self._a = arrs_by_period

    def __getitem__(self, key):
        p, d, h = key
        return float(self._a[p][d * 24 + h])


class _BtesTDaily:
    """RD btes_T[year, day] (daily state) from the full model's hourly state."""

    def __init__(self, comp):
        self._c = comp

    def __getitem__(self, key):
        y, d = key
        return float(pyo.value(self._c[y, d * 24]))


class _FakeM:
    def __init__(self, real, full_model):
        self._real = real
        self.PERIODS = list(full_model.investment_years)
        self.REP_DAYS = range(365)
        self.HOURS = range(24)
        self.day_weight = _Const(1.0)

        self.gen = _As4D(real.gen, 2)
        self.efficiency = _EffAs4D(real.efficiency)
        self.electricity_price = _As4D(real.electricity_price, 1)
        self.electricity_sell_price = _SellPrice(full_model._sell_arr_by_period)

        if hasattr(real, "reinjection"):
            self.reinject = _As4D(real.reinjection, 1)
        for name in ("orc_share", "hp_share", "orc_pump_extra"):
            if hasattr(real, name):
                setattr(self, name, _As4D(getattr(real, name), 1))
        for name in ("y_ci", "y_ti"):
            if hasattr(real, name):
                setattr(self, name, _As4D(getattr(real, name), 1))
        if hasattr(real, "u"):
            self.u = _As4D(real.u, 2)
        if hasattr(real, "charge"):
            self.charge = _As4D(real.charge, 2)
            self.discharge = _As4D(real.discharge, 2)
        if hasattr(real, "soc") and list(getattr(real, "STOR", [])):
            # RD splits SOC into inter-day + intra-day (Kotzur); the full model
            # tracks one chronological SOC — expose it as intra with inter = 0.
            self.soc_intra = _As4D(real.soc, 2)
            self.soc_inter = _Const(0.0)
        if hasattr(real, "btes_Qch"):
            # BTES dispatch is period-indexed hourly in both variants; the RD
            # temperature state is DAILY (year, d) while the full state is
            # hourly (year, h) — report the start-of-day value.
            self.btes_Qch = _As4D(real.btes_Qch, 1)
            self.btes_Qdis = _As4D(real.btes_Qdis, 1)
            self.btes_T = _BtesTDaily(real.btes_T)

    def nvariables(self):
        return self._real.nvariables()

    def nconstraints(self):
        return self._real.nconstraints()

    def __getattr__(self, name):
        # cap_total/cap_add/capex/opex_*/fuel_price/emission_factor/carbon_price/
        # STOR/UC_TECH/total_cost/y_revorc/… all have identical shapes → real.
        return getattr(self._real, name)


class FullAsRD:
    """RD-shaped view of a solved full model for the run_rd reporting suite."""

    def __init__(self, full_model, demand: pd.DataFrame):
        self.model = _FakeM(full_model.model, full_model)
        self.technologies = full_model.technologies
        self.investment_years = full_model.investment_years
        self.storage = full_model.storage
        self.uc_enabled = full_model.uc_enabled
        self.btes_enabled = full_model.btes_enabled
        self.btes_build_period = full_model.btes_build_period
        self._btes_arx = getattr(full_model, "_btes_arx", None)
        self.demand_data = full_model.demand_data
        self.discount_rate_by_period = full_model.discount_rate_by_period
        self._saphea_design = getattr(full_model, "_saphea_design", None)

        identity = list(range(365))

        def _tsa_for(p):
            col = f"corrected_demand_{p}"
            dem = (demand[col].values if col in demand.columns
                   else demand["heat_demand"].values)
            return {
                "n_rep_days": 365, "n_days": 365,
                "day_to_rep": identity, "weights": [1.0] * 365,
                "rep_data": pd.DataFrame({"heat_demand": dem[:8760]}),
            }

        self.tsa_by_period = {p: _tsa_for(p) for p in self.investment_years}
        self.tsa = self.tsa_by_period[min(self.investment_years)]

        if hasattr(full_model, "_revorc"):
            q_in = full_model._revorc["q_in"]
            p_net = full_model._revorc["p_net"]
            maps = {
                "orc_q_in_MW": {(h // 24, h % 24): v for h, v in q_in.items()},
                "orc_p_net_MW": {(h // 24, h % 24): v for h, v in p_net.items()},
            }
            self._revorc = {
                "maps_by_p": {p: maps for p in self.investment_years},
                "cfg": full_model._revorc["cfg"],
            }
