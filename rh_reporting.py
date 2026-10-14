"""RD-shaped view of a solved RH model, so run_rd's reporting suite works on it.

Mirrors full_reporting.FullAsRD. The only structural difference is the index
translation: the full model maps (rep_day d, hour h) -> the real hour d*24+h,
whereas the RH model maps it through the clustering, (d, h) -> sigma[d*24+h],
i.e. the representative hour that real hour belongs to.

Everything downstream — dispatch plots, SOC, BTES, capacity, generation, the
Excel workbook, the revORC reports — then works unchanged, and the 8760-h
reconstruction is the model's own mapping rather than an interpolation.

    from rh_reporting import RHAsRD
    adapter = RHAsRD(model, demand)
    rd.save_dispatch_plots(adapter, results_dir)
"""

from typing import Dict, List

import pandas as pd
import pyomo.environ as pyo


class _Sigma:
    """(…head[n], d, h) → float(real[(…head, sigma[period][d*24 + h])]).

    The clustering is PER MILESTONE, so the period must select which sigma to
    use. Taking the first period's mapping for every period (as an earlier
    version did) hands each hour a dispatch value that belongs to a different
    hour — the reconstruction then looks like hour-to-hour noise in every
    milestone except the first.
    """

    def __init__(self, comp, nlead: int, sigma_by_period: Dict[int, List[int]],
                 period_pos: int = 1):
        self._c, self._n, self._s, self._pp = comp, nlead, sigma_by_period, period_pos

    def __getitem__(self, key):
        head, d, h = key[:self._n], key[self._n], key[self._n + 1]
        sig = self._s[head[self._pp]]
        return float(pyo.value(self._c[(*head, sig[d * 24 + h])]))


class _SigmaPerPeriod:
    """Same, but sigma differs per period (per-milestone clustering)."""

    def __init__(self, comp, sigma_by_period: Dict[int, List[int]]):
        self._c, self._s = comp, sigma_by_period

    def __getitem__(self, key):
        p, d, h = key
        return float(pyo.value(self._c[p, self._s[p][d * 24 + h]]))


class _EffSigma:
    """RD efficiency is (tech, period, d, h); RH is (tech, period, rep_hour)."""

    def __init__(self, comp, sigma_by_period):
        self._c, self._s = comp, sigma_by_period

    def __getitem__(self, key):
        t, p, d, h = key
        return float(pyo.value(self._c[t, p, self._s[p][d * 24 + h]]))


class _Const:
    def __init__(self, v: float):
        self._v = v

    def __getitem__(self, key):
        return self._v


class _SellSigma:
    def __init__(self, comp, sigma_by_period):
        self._c, self._s = comp, sigma_by_period

    def __getitem__(self, key):
        p, d, h = key
        return float(pyo.value(self._c[p, self._s[p][d * 24 + h]]))


class _BtesTDaily:
    """RD btes_T[year, day]; RH tracks it on the reduced storage index J."""

    def __init__(self, comp, n_levels: int):
        self._c, self._n = comp, n_levels

    def __getitem__(self, key):
        y, d = key
        return float(pyo.value(self._c[y, min(d, self._n - 1)]))


class _FakeM:
    def __init__(self, real, rh_model):
        self._real = real
        self.PERIODS = list(rh_model.investment_years)
        self.REP_DAYS = range(365)
        self.HOURS = range(24)
        self.day_weight = _Const(1.0)

        sig = {p: rh_model.rh_by_period[p]["sigma"] for p in self.PERIODS}
        self._sig = sig

        self.gen = _Sigma(real.gen, 2, sig)
        self.efficiency = _EffSigma(real.efficiency, sig)
        self.electricity_price = _SigmaPerPeriod(real.electricity_price, sig)
        self.electricity_sell_price = _SellSigma(real.electricity_sell_price, sig)

        if hasattr(real, "reinject"):
            self.reinject = _SigmaPerPeriod(real.reinject, sig)
        for name in ("orc_share", "hp_share", "orc_pump_extra", "y_ci", "y_ti"):
            if hasattr(real, name):
                setattr(self, name, _SigmaPerPeriod(getattr(real, name), sig))
        if hasattr(real, "u"):
            self.u = _Sigma(real.u, 2, sig)
        if hasattr(real, "charge"):
            self.charge = _Sigma(real.charge, 2, sig)
            self.discharge = _Sigma(real.discharge, 2, sig)
        if hasattr(real, "soc") and list(getattr(real, "STOR", [])):
            # RH tracks one SOC on the reduced index; expose it as the RD
            # intra-day component with inter-day zero, as the full adapter does.
            self.soc_intra = _Sigma(real.soc, 2, sig)
            self.soc_inter = _Const(0.0)
        if hasattr(real, "btes_Qch"):
            self.btes_Qch = _SigmaPerPeriod(real.btes_Qch, sig)
            self.btes_Qdis = _SigmaPerPeriod(real.btes_Qdis, sig)
            n_lv = len(real.STOR_LEVELS) if hasattr(real, "STOR_LEVELS") else 365
            self.btes_T = _BtesTDaily(real.btes_T, n_lv)

    def nvariables(self):
        return self._real.nvariables()

    def nconstraints(self):
        return self._real.nconstraints()

    def __getattr__(self, name):
        return getattr(self._real, name)


class RHAsRD:
    """RD-shaped view of a solved RH model for the run_rd reporting suite."""

    def __init__(self, rh_model, demand: pd.DataFrame):
        self.model = _FakeM(rh_model.model, rh_model)
        self.technologies = rh_model.technologies
        self.investment_years = rh_model.investment_years
        self.storage = rh_model.storage
        self.uc_enabled = getattr(rh_model, "uc_enabled", False)
        self.btes_enabled = rh_model.btes_enabled
        self.btes_build_period = rh_model.btes_build_period
        self._btes_arx = getattr(rh_model, "_btes_arx", None)
        self.demand_data = rh_model.demand_data
        self.discount_rate_by_period = rh_model.discount_rate_by_period
        self._saphea_design = getattr(rh_model, "_saphea_design", None)

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

        # revORC maps are keyed (i, 0) in RH; the RD reporting wants (d, h)
        if hasattr(rh_model, "_revorc"):
            sig = self.model._sig
            src = rh_model._revorc["maps_by_p"]
            maps_by_p = {}
            for p in self.investment_years:
                mp = src[p]
                maps_by_p[p] = {
                    k: {(d, h): mp[k].get((sig[p][d * 24 + h], 0), 0.0)
                        for d in range(365) for h in range(24)}
                    for k in ("orc_q_in_MW", "orc_p_net_MW", "hp_cop")
                    if k in mp
                }
            self._revorc = {"maps_by_p": maps_by_p, "cfg": rh_model._revorc["cfg"]}
