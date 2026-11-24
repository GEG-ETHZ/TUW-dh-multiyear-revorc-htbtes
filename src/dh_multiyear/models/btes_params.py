"""
BTES parameter tables for multi-year Option-B investment.

Period-specific parameters (T_init, Qch_max, Qdis_max) are indexed by
(build_period, op_period).  ARX coefficients are constant across all periods —
only the borehole boundary conditions vary as the rock warms up.

Age-block mapping (5-year periods since construction):
  age_block 0  → years  1-5  of operation  (warm-up)
  age_block 1  → years  6-10
  age_block 2  → years 11-15
  age_block 3  → years 16-20+ (capped at steady-state)
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List
import json
import re

# ---------------------------------------------------------------------------
# Simulation-derived parameters per age block
# Source: BTES/01_btes_simulation/output/hourly_btes_simulation.csv
#   T_init  = T_borehole_C at hour 0 of the first year in the block
#   Qch_max = peak charging power (kW) observed in that block
#   Qdis_max= peak discharge power (kW, absolute) observed in that block
# ---------------------------------------------------------------------------
_AGE_BLOCK_PARAMS: Dict[int, Dict[str, float]] = {
    # T_init, Qch_max_kW, Qdis_max_kW from simulation.
    # b_ch_h / b_dis_h are loaded per-block from btes_hourly_params.json (age_blocks section).
    0: {"T_init": 11.25,  "Qch_max_kW": 19902.0, "Qdis_max_kW": 2141.0},
    1: {"T_init": 85.52,  "Qch_max_kW":  7377.0, "Qdis_max_kW": 2172.0},
    2: {"T_init": 87.11,  "Qch_max_kW":  7193.0, "Qdis_max_kW": 2186.0},
    3: {"T_init": 87.87,  "Qch_max_kW":  7036.0, "Qdis_max_kW": 2192.0},
}
_MAX_AGE_BLOCK = max(_AGE_BLOCK_PARAMS)


_DESIGN_LIBRARY_CACHE: Dict[str, Any] | None = None


def _design_library() -> Dict[str, Any]:
    """The per-design BTES parameter library, loaded once."""
    global _DESIGN_LIBRARY_CACHE
    if _DESIGN_LIBRARY_CACHE is None:
        import os
        _sfx = os.environ.get("BTES_LIB_SUFFIX", "")
        from dh_multiyear import PROJECT_ROOT
        lib = PROJECT_ROOT / "data" / "btes" / f"design_library{_sfx}.json"
        try:
            _DESIGN_LIBRARY_CACHE = json.loads(lib.read_text())
        except Exception:
            _DESIGN_LIBRARY_CACHE = {}
    return _DESIGN_LIBRARY_CACHE


def _boreholes_of(raw: Dict[str, Any], path: Path) -> int | None:
    """Borehole count of an ARX parameter file: explicit key, description, or filename."""
    if "n_boreholes" in raw:
        return int(raw["n_boreholes"])
    m = re.search(r"\((\d+)\s*bh", str(raw.get("description", "")))
    if m:
        return int(m.group(1))
    m = re.search(r"_n(\d+)\.json$", path.name)
    if m:
        return int(m.group(1))
    return 200 if path.name == "btes_hourly_params.json" else None


def _age_block(build_year: int, op_year: int, period_length: int = 5) -> int:
    """Return the age-block index for an operating period given when BTES was built."""
    if op_year < build_year:
        return -1  # not yet operational
    elapsed_periods = (op_year - build_year) // period_length
    return min(elapsed_periods, _MAX_AGE_BLOCK)


def build_period_params(
    investment_years: List[int],
    period_length: int = 5,
    arx: Dict[str, Any] | None = None,
) -> Dict[int, Dict[int, Dict[str, float]]]:
    """
    Returns a 2-level dict:  params[build_period][op_period] → {T_init, Qch_max_kW, Qdis_max_kW, b_ch_h, b_dis_h}

    Only entries where op_period >= build_period are populated.
    arx: output of load_arx_params(); used to pull per-age-block b_ch_h / b_dis_h.
    """
    age_blocks = (arx or {}).get("age_blocks", {})
    b_ch_global  = (arx or {}).get("b_ch_h",  5.387e-6)
    b_dis_global = (arx or {}).get("b_dis_h", -1.0355e-5)

    params: Dict[int, Dict[int, Dict[str, float]]] = {}
    for bp in investment_years:
        params[bp] = {}
        for op in investment_years:
            ab = _age_block(bp, op, period_length)
            if ab < 0:
                continue
            entry = dict(_AGE_BLOCK_PARAMS[ab])
            blk = age_blocks.get(ab, {})
            entry["b_ch_h"]  = blk.get("b_ch_h",  b_ch_global)
            entry["b_dis_h"] = blk.get("b_dis_h", b_dis_global)
            for _k in ("Qch_max_kW", "Qdis_max_kW", "T_init"):
                if _k in blk:
                    entry[_k] = blk[_k]
            params[bp][op] = entry
    return params


def period_params_for_build(
    investment_years: List[int],
    btes_build_period: int,
    period_length: int = 5,
    arx: Dict[str, Any] | None = None,
) -> Dict[int, Dict[str, Any]]:
    """
    Return a flat dict  period_year → {T_init, Qch_max_kW, Qdis_max_kW, b_ch_h, b_dis_h, active}
    for a given exogenous build decision.

    Periods before btes_build_period have active=False and zero capacities.
    arx: output of load_arx_params(); used to pull per-age-block b_ch_h / b_dis_h.
    """
    age_blocks = (arx or {}).get("age_blocks", {})
    b_ch_global  = (arx or {}).get("b_ch_h",  5.387e-6)
    b_dis_global = (arx or {}).get("b_dis_h", -1.0355e-5)

    result: Dict[int, Dict[str, Any]] = {}
    for op in investment_years:
        ab = _age_block(btes_build_period, op, period_length)
        if ab < 0:
            result[op] = {
                "active": False,
                "T_init": 0.0,
                "Qch_max_kW": 0.0,
                "Qdis_max_kW": 0.0,
                "b_ch_h": 0.0,
                "b_dis_h": 0.0,
            }
        else:
            entry = dict(_AGE_BLOCK_PARAMS[ab])
            blk = age_blocks.get(ab, {})
            entry["b_ch_h"]  = blk.get("b_ch_h",  b_ch_global)
            entry["b_dis_h"] = blk.get("b_dis_h", b_dis_global)
            for _k in ("Qch_max_kW", "Qdis_max_kW", "T_init"):
                if _k in blk:
                    entry[_k] = blk[_k]
            entry["active"] = True
            result[op] = entry
    return result


def load_arx_params(json_path: Path | str | None = None) -> Dict[str, Any]:
    """
    Load ARX surrogate parameters from JSON.  Falls back to hard-coded defaults
    if json_path is None or file not found.

    Returns a dict that includes an 'age_blocks' key:
        {0: {b_ch_h, b_dis_h}, 1: ..., 2: ..., 3: ...}
    sourced from the JSON 'age_blocks' section written by fit_btes_hourly.py.
    Falls back to the global b_ch_h/b_dis_h for all blocks if section is absent.
    """
    _defaults = {
        "a_h":        1.0,
        "b_ch_h":     5.387e-6,    # °C per kWh charged  (steady-state fallback)
        "b_dis_h":   -1.0355e-5,   # °C per kWh discharged
        "c_h":        0.0,
        "Tmin":      80.8,
        "Tmax":     120.0,
        "capex_EUR": 2_800_000.0,
        "lifetime_a": 25,
        "cost_charge":    0.0,
        "cost_discharge": 0.0,
        "opex_fix_EUR_per_yr": 0.0,
        "age_blocks": {},           # populated from JSON; empty = use global fallback
    }
    if json_path is None:
        return _defaults

    path = Path(json_path)
    if not path.exists():
        return _defaults

    raw = json.loads(path.read_text())
    fit    = raw.get("fit", {})
    bounds = raw.get("bounds", {})
    invest = raw.get("invest", {})

    b_ch_h_global  = fit.get("b_ch_h_per_kWh", _defaults["b_ch_h"])
    b_dis_h_global = fit.get("b_dis_h_per_kWh", _defaults["b_dis_h"])

    # Parse per-age-block coefficients written by fit_btes_hourly.py
    age_blocks: Dict[int, Dict[str, float]] = {}
    for k, v in raw.get("age_blocks", {}).items():
        age_blocks[int(k)] = {
            "b_ch_h":  v.get("b_ch_h_per_kWh",  b_ch_h_global),
            "b_dis_h": v.get("b_dis_h_per_kWh", b_dis_h_global),
        }

    # Per-age-block POWER caps and initial temperatures for THIS field size.
    # Without this the caps fall back to _AGE_BLOCK_PARAMS, which are the
    # 200-borehole reference values and do not scale with the design — a
    # 100-borehole field would then be allowed the discharge power of a
    # 200-borehole one. The design library carries them per block per design.
    n_bh = _boreholes_of(raw, path)
    if n_bh is not None:
        lib_entry = _design_library().get(str(n_bh), {})
        for k, v in lib_entry.get("blocks", {}).items():
            blk = age_blocks.setdefault(int(k), {
                "b_ch_h": b_ch_h_global, "b_dis_h": b_dis_h_global})
            if "Qch_max_kW" in v:
                blk["Qch_max_kW"] = float(v["Qch_max_kW"])
            if "Qdis_max_kW" in v:
                blk["Qdis_max_kW"] = float(v["Qdis_max_kW"])
            if "T_init" in v:
                blk["T_init"] = float(v["T_init"])

    return {
        "a_h":        fit.get("a_h",            _defaults["a_h"]),
        "b_ch_h":     b_ch_h_global,
        "b_dis_h":    b_dis_h_global,
        "c_h":        fit.get("c_h_degC",        _defaults["c_h"]),
        "Tmin":       bounds.get("Tmin_degC",    _defaults["Tmin"]),
        "Tmax":       bounds.get("Tmax_degC",    _defaults["Tmax"]),
        "capex_EUR":  invest.get("IK_EUR",       _defaults["capex_EUR"]),
        "lifetime_a": invest.get("lt_yr",        _defaults["lifetime_a"]),
        "cost_charge":    raw.get("cost_charge",     _defaults["cost_charge"]),
        "cost_discharge": raw.get("cost_discharge",  _defaults["cost_discharge"]),
        "opex_fix_EUR_per_yr": _fixed_opex_per_yr(raw, bounds, age_blocks),
        "age_blocks": age_blocks,
        # lumped discharge HX coefficient (kW/K) for the temperature-dependent discharge rule
        "UA_dis_kW_per_K": bounds.get("UA_dis_kW_per_K"),
        "UA_dis_levels": bounds.get("UA_dis_levels"),
        "UA_dis_cap_ref_kW_per_K": bounds.get("UA_dis_cap_ref_kW_per_K"),
        "k_wall_K_per_kW": bounds.get("k_wall_K_per_kW", 0.0),
    }


def _fixed_opex_per_yr(raw, bounds, age_blocks) -> float:
    """Annual fixed O&M, on the DEA pit-storage basis (EUR per MWh of capacity).

    No published fixed-O&M figure exists for BTES (IEA SHC Task 45 and the RSER
    2023 review give CAPEX only), so the rate is taken from the Danish Energy
    Agency pit-storage datasheet -- 0.0037 EUR/kWh/a, i.e. the same 4 EUR/MWh/a
    the pit store in this model already carries. Capacity is the recoverable
    energy of the mature field: (Tmax - Tmin) / |b_dis| of the last age block.
    """
    rate = float(raw.get("opex_fix_EUR_per_MWh_cap", 0.0))
    if rate <= 0 or not age_blocks:
        return 0.0
    b_dis = abs(age_blocks[max(age_blocks)].get("b_dis_h", 0.0))
    if b_dis <= 0:
        return 0.0
    Tmin = float(bounds.get("Tmin_degC", 80.8)); Tmax = float(bounds.get("Tmax_degC", 120.0))
    cap_MWh = (Tmax - Tmin) / b_dis / 1000.0
    return rate * cap_MWh

# ---------------------------------------------------------------------------
#  Design library: one ARX parameter set per candidate borehole-field size
# ---------------------------------------------------------------------------

def load_design_library(json_path: Path | str) -> Dict[str, Dict[str, Any]]:
    """Load the per-design parameter library written by cases/kranj/btes.

    Each entry carries the geometry, the CAPEX, the ARX scalars (a_h, c_h,
    Tmin, Tmax) and the per-age-block coefficients and capacities measured from
    that design's own 20-year simulation. Selecting a design in the MILP means
    selecting one of these parameter sets.
    """
    return json.loads(Path(json_path).read_text())


def period_params_for_design(design: Dict[str, Any],
                             investment_years: List[int],
                             build_period: int,
                             period_length: int = 5) -> Dict[int, Dict[str, Any]]:
    """Flat dict  period -> {T_init, Qch_max_kW, Qdis_max_kW, b_ch_h, b_dis_h,
    active} for one design, using the age block reached in each period."""
    blocks = design["blocks"]
    out: Dict[int, Dict[str, Any]] = {}
    for op in investment_years:
        ab = _age_block(build_period, op, period_length)
        if ab < 0:
            out[op] = {"active": False, "T_init": 0.0, "Qch_max_kW": 0.0,
                       "Qdis_max_kW": 0.0, "b_ch_h": 0.0, "b_dis_h": 0.0}
            continue
        blk = blocks[str(min(ab, max(int(k) for k in blocks)))]
        out[op] = {
            "active": True,
            "T_init": float(blk["T_init"]),
            "Qch_max_kW": float(blk["Qch_max_kW"]),
            "Qdis_max_kW": float(blk["Qdis_max_kW"]),
            "b_ch_h": float(blk["b_ch_h"]),
            "b_dis_h": float(blk["b_dis_h"]),
        }
    return out
