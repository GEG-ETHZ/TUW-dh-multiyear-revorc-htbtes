"""
revORC surrogate maps — hourly performance parameters for the reversible HP/ORC.

Evaluates the FLEXGEO surrogate models (2026-07-30 revision, in
<case>/revers_surrogates/) OFFLINE on the representative-day profiles and returns
per-(rep_day, hour) linear coefficients for the MILP, exactly like the hourly
SAPHEA COP treatment in run_rd.py — the optimization itself stays linear and
never touches the joblib bundles.

Per module (basis: `module_brine_kg_s` brine at T_prod, see 'RevORC' sheet of
Munich_tech_data.xlsx):

    orc_q_in_MW[d,h]   brine heat draw in ORC mode        (orc_Q_in_W)
    orc_p_net_MW[d,h]  net electric output in ORC mode    (orc_P_net_W, net of
                       working-fluid pump, ACC fans and auxiliaries)
    hp_out_MW[d,h]     max heat output in HP mode         (hp_Q_delivered_W)
    hp_cop[d,h]        heating COP in HP mode             (Q_delivered/W_comp)

Feasibility gates (FlexGeo manual: infeasible hours are handled through the
applied-mode/output logic, i.e. produce zero output):

  ORC out-of-bounds gate: hours whose features fall outside the ORC bundle's
    `input_bounds_filtered` (training data was filtered to feasible ORC
    operation, e.g. brine−ambient ≥ 63 K, T_amb ≤ 34.8 °C) get orc_q_in =
    orc_p_net = 0 instead of silent extrapolation. A 0.5 %-of-range tolerance
    absorbs edge cases (module_brine_kg_s = 8.0 vs training max 7.9998).
  ORC ambient gate: optional `orc_amb_min_C` (RevORC sheet) mirrors the FlexGeo
    mode-profile rule "ORC only when T_amb ≥ orc_amb_min_C".
  HP-mode sanity gate: the HP bundle still ships without `input_bounds_filtered`,
    so hours whose predicted full-load COP falls outside the training filter band
    [1, 12] are disabled (hp_out=0); if most hours fail, HP mode is disabled
    entirely. With the 2026-07-30 series-trained bundle this gate should no longer
    bite: it is evaluated on the cooled-brine slice it was trained for (inlet
    T_return + pinch ~65 °C, outlet floor = T_inject_min 45 °C, positive lift),
    not on wellhead brine.

HP part-load: the FlexGeo HP branch is demand-following and COP varies with
load (`load_fraction_nominal` is a trained feature). The MILP applies one
linear COP per hour, so the surrogate is evaluated at load fractions
{hp_min_load_frac (default 0.25), 0.5, 1.0} and the MINIMUM COP across them is
used — conservative for any dispatch point the optimizer may choose.

Results are cached as CSV in data/revorc_maps_cache/ keyed by a hash of all
inputs, so reruns skip the (2 GB) bundle loading.
"""

from __future__ import annotations

import hashlib
import os
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

_CASE_DIR = Path(__file__).parent.resolve()
_ORC_DIR = _CASE_DIR / "revers_surrogates"
_CACHE_DIR = _CASE_DIR / "data" / "revorc_maps_cache"

# ORC: the one-stage expander bundle (byte-identical to the 2026-06-30 delivery).
# The 2026-07-30 revision also ships a two-stage bundle for cold-climate designs;
# Munich's design pressure ratio is 6.5 (< the 8.0 staging threshold), so the
# one-stage machine is the one being modelled here.
ORC_MODEL = _ORC_DIR / "models" / "orc_synthetic_year_surrogate_filtered_variants_vopt6.joblib"
# HP: the 2026-07-30 series-trained bundle (brine inlet 55-75 °C, outlet floor
# 35-45 °C, sink supply 75-100 °C) — the arrangement the revORC actually has.
HP_MODEL = (_ORC_DIR / "models" /
            "hp_synthetic_year_surrogate_1mw_filtered_variants_vopt6"
            "_hp55_75_out35_45_sourcebalanced.joblib")

# Training filter band of the HP synthetic-year dataset (metrics JSON: cop_min/cop_max)
HP_COP_VALID = (1.0, 12.0)


def load_revorc_config(tech_file: str | Path) -> dict | None:
    """Read the 'RevORC' sheet (parameter/value rows). None if absent or disabled."""
    tech_file = Path(tech_file)
    if "RevORC" not in pd.ExcelFile(tech_file).sheet_names:
        return None
    df = pd.read_excel(tech_file, sheet_name="RevORC")
    cfg = {
        str(row["parameter"]).strip(): float(row["value"])
        for _, row in df.iterrows()
        if pd.notna(row.get("parameter")) and pd.notna(row.get("value"))
    }
    # A capex given per kW_el (the convention the ORC literature reports) is
    # converted onto the model's MW_th-of-HP-output capacity basis using the
    # module's rated ORC electric output. The ratio is large here (~13.8): the
    # same hardware delivers 1 MW_th in HP mode but only ~73 kW_el from the
    # 106 °C wellhead brine in ORC mode.
    if "capex_eur_per_kW_el" in cfg:
        _need = {"module_orc_MW_el", "module_hp_MWth"} - set(cfg)
        if _need:
            raise ValueError(
                f"capex_eur_per_kW_el needs {sorted(_need)} on the 'RevORC' sheet")
        _ratio = cfg["module_hp_MWth"] / cfg["module_orc_MW_el"]   # MW_th per MW_el
        cfg["capex_eur_per_MWth"] = cfg["capex_eur_per_kW_el"] * 1e3 / _ratio
        logger.info(
            f"RevORC capex on the electric basis: {cfg['capex_eur_per_kW_el']:,.0f} "
            f"EUR/kW_el -> {cfg['capex_eur_per_MWth']:,.0f} EUR/MW_th (rated "
            f"{cfg['module_orc_MW_el']*1e3:.1f} kW_el per {cfg['module_hp_MWth']:.1f} "
            f"MW_th module, ratio {_ratio:.1f})")

    # REVORC_CAPEX_SCALE scales the specific investment cost for sensitivity runs
    _scale = os.environ.get("REVORC_CAPEX_SCALE")
    if _scale and "capex_eur_per_MWth" in cfg:
        _f = float(_scale)
        _old = cfg["capex_eur_per_MWth"]
        cfg["capex_eur_per_MWth"] = _old * _f
        logger.warning(
            f"REVORC_CAPEX_SCALE={_f:g}: revORC CAPEX {_old:,.0f} -> "
            f"{cfg['capex_eur_per_MWth']:,.0f} EUR/MW_th")
    # REVORC_DEAD_HOURS overrides the FlexGeo dead-hour switching rule for
    # sensitivity runs (0 = a module may change mode with no idle hour between).
    _dh = os.environ.get("REVORC_DEAD_HOURS")
    if _dh is not None and _dh != "":
        _old_dh = cfg.get("dead_hours", 1.0)
        cfg["dead_hours"] = float(_dh)
        logger.warning(
            f"REVORC_DEAD_HOURS={_dh}: dead-hour switching rule "
            f"{_old_dh:g} -> {cfg['dead_hours']:g} h")
    if not cfg.get("enabled", 0):
        logger.info("RevORC sheet present but enabled=0 — revORC disabled")
        return None
    required = {"module_brine_kg_s", "module_hp_MWth", "T_brine_out_min_C",
                "capex_eur_per_MWth", "opex_fix_frac", "lifetime_a", "max_modules"}
    missing = required - set(cfg)
    if missing:
        raise ValueError(f"'RevORC' sheet is missing parameters: {sorted(missing)}")
    logger.info(
        f"Loaded RevORC config: {cfg['max_modules']:.0f} × {cfg['module_hp_MWth']:.1f} MW_th "
        f"modules ({cfg['module_brine_kg_s']:.0f} kg/s each), "
        f"CAPEX {cfg['capex_eur_per_MWth']:,.0f} €/MW_th"
    )
    return cfg


def _cache_key(rep_ambient, rep_supply, rep_return, resource, cfg) -> str:
    payload = {
        "ver": 3,  # 3: series-trained HP bundle evaluated on the cooled-brine slice
        "ambient": np.round(rep_ambient, 2).tolist(),
        "supply": np.round(rep_supply, 2).tolist(),
        "return": np.round(rep_return, 2).tolist(),
        "resource": {k: round(float(v), 4) for k, v in sorted(resource.items())},
        "cfg": {k: round(float(v), 6) for k, v in sorted(cfg.items())},
        "orc_mtime": ORC_MODEL.stat().st_mtime if ORC_MODEL.exists() else 0,
        "hp_mtime": HP_MODEL.stat().st_mtime if HP_MODEL.exists() else 0,
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def _predict_orc(rows: pd.DataFrame) -> pd.DataFrame:
    """Batch-evaluate the ORC off-design surrogate (selected variant)."""
    import joblib

    sys.path.insert(0, str(_ORC_DIR))
    from orc_offdesign_surrogate_filtered_variants_vopt6 import (
        add_derived_features, inverse_y, outputs_from_targets,
    )

    bundle = joblib.load(ORC_MODEL)
    variant = bundle.get("selected_variant", "direct_power")
    feats = add_derived_features(rows)
    X = feats[list(bundle["input_columns"])].astype(float).to_numpy()
    spec = bundle["variant_specs"][variant]
    out = outputs_from_targets(inverse_y(bundle["models"][variant].predict(X), spec), variant)

    # Per-row out-of-bounds mask against the training-data bounds. The training
    # set was filtered to feasible ORC operation, so OOB hours are treated as
    # ORC-infeasible (zeroed by the caller), not extrapolated. 0.5 % of each
    # bound's range as tolerance absorbs sampling edges of the training LHS.
    bounds = bundle.get("input_bounds_filtered", {})
    oob = np.zeros(len(feats), dtype=bool)
    for col, lim in bounds.items():
        if col in feats.columns:
            v = feats[col].astype(float).to_numpy()
            lo, hi = float(lim.get("min", -np.inf)), float(lim.get("max", np.inf))
            tol = 0.005 * (hi - lo) if np.isfinite(hi - lo) else 0.0
            oob |= (v < lo - tol) | (v > hi + tol)
    out["oob"] = oob
    return out


def _predict_hp(rows: pd.DataFrame, module_hp_MWth: float) -> pd.DataFrame:
    """Batch-evaluate the HP off-design surrogate (vectorised build_feature_frame)."""
    import joblib

    bundle = joblib.load(HP_MODEL)
    approach = float(bundle.get("sink_approach_K", 5.0))
    nominal = float(bundle.get("nominal_load_W", 1.0e6))
    cp = float(bundle.get("cp_proxy_J_kgK", 4180.0))

    f = rows.copy()
    f["T_sink_supply_C"] = f["T_brine_in_C"] + f["temperature_lift_K"]
    f["T_sink_return_C"] = f["T_sink_supply_C"] - f["sink_deltaT_K"]
    f["T_cond_HP_target_C"] = f["T_sink_supply_C"] + approach
    f["load_fraction_nominal"] = f["heating_load_W"] / max(nominal, 1.0)
    f["source_deltaT_available_K"] = f["T_brine_in_C"] - f["T_brine_out_min_C"]
    f["geo_capacity_proxy_W"] = f["m_brine_max_kg_s"] * cp * f["source_deltaT_available_K"]
    f["sink_supply_minus_brine_out_min_K"] = f["T_sink_supply_C"] - f["T_brine_out_min_C"]
    f["cond_target_minus_brine_in_K"] = f["T_cond_HP_target_C"] - f["T_brine_in_C"]
    f["cond_target_minus_brine_out_min_K"] = f["T_cond_HP_target_C"] - f["T_brine_out_min_C"]
    f["sink_deltaT_over_source_deltaT"] = f["sink_deltaT_K"] / f["source_deltaT_available_K"]

    X = f[list(bundle["input_columns"])].astype(float).to_numpy()
    q_pred = np.asarray(bundle["q_model"].predict(X), dtype=float)
    q_cols = list(bundle["q_output_columns"])
    out = pd.DataFrame(q_pred, columns=q_cols, index=rows.index)

    variant = bundle.get("selected_variant", "inverse_cop")
    cfg_v = bundle["variant_configs"][variant]
    cols = cfg_v.get("cols") or cfg_v.get("target_columns")
    log_cols = set(cfg_v.get("log_cols", [])) | set(cfg_v.get("log_columns", []))
    raw = np.asarray(bundle["variant_models"][variant].predict(X), dtype=float)
    if raw.ndim == 1:
        raw = raw.reshape(-1, 1)
    for j, c in enumerate(cols):
        v = raw[:, j]
        out[c] = np.expm1(v) if c in log_cols else v

    Q = out["hp_Q_delivered_W"].clip(lower=0.0)
    if variant == "inverse_cop":
        inv = out["hp_inverse_COP"].clip(lower=1e-12)
        out["hp_COP"] = 1.0 / inv
    elif variant == "direct_w":
        out["hp_COP"] = Q / out["hp_W_comp_W"].clip(lower=1e-12)
    else:
        raise ValueError(f"Unsupported HP variant for batch evaluation: {variant}")
    out["hp_Q_delivered_MW"] = np.minimum(Q / 1e6, module_hp_MWth)
    return out


def build_revorc_maps(rep_ambient, rep_supply, rep_return, resource, cfg) -> dict:
    """
    Evaluate both surrogates on the (n_rep, n_h) representative profiles.

    Shape-agnostic: the representative-DAY model passes (n_rep_days, 24) and gets
    (d, h) keys; the representative-HOUR model passes (n_rep_hours, 1) and gets
    (i, 0) keys. Nothing here depends on the slots being a day.

    Returns dict with (d, h)-keyed float dicts: orc_q_in_MW, orc_p_net_MW,
    hp_out_MW, hp_cop, plus 'hp_enabled' and 'stats'.
    """
    n_rep = rep_ambient.shape[0]
    n_h = rep_ambient.shape[1]
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = _cache_key(rep_ambient, rep_supply, rep_return, resource, cfg)
    cache_file = _CACHE_DIR / f"revorc_maps_{key}.csv"

    if "p_brine_bar" not in resource:
        raise ValueError(
            "revORC needs 'p_brine_bar' in the 'Geothermal Resource' sheet of "
            "Munich_tech_data.xlsx (surrogate training range 12–24 bar)"
        )

    if cache_file.exists():
        df = pd.read_csv(cache_file)
        logger.info(f"revORC maps loaded from cache ({cache_file.name})")
    else:
        logger.info("Evaluating revORC surrogates on rep-day profiles (no cache hit)…")
        idx = [(d, h) for d in range(n_rep) for h in range(n_h)]
        base = pd.DataFrame({
            "T_amb_C": [float(rep_ambient[d, h]) for d, h in idx],
            "T_brine_in_C": resource["T_prod"],
            "p_brine_bar": resource["p_brine_bar"],
            "T_brine_out_min_C": cfg["T_brine_out_min_C"],
            "m_brine_max_kg_s": cfg["module_brine_kg_s"],
        })
        orc = _predict_orc(base.copy())

        # HP mode runs on the SERIES slice, not on wellhead brine: the machine
        # takes the brine that has already given up its heat in the direct HX
        # (T_woHP = T_return + pinch, ~65 °C) and cools it to the injection floor.
        # The 2026-07-30 bundle is trained exactly there (55-75 °C in, 35-45 °C
        # outlet floor), so the lift is positive and the COP is the machine's own.
        t_hp_in = np.array([float(rep_return[d, h]) + resource["direct_hx_pinch"]
                            for d, h in idx])
        hp_rows = base.drop(columns=["T_amb_C"]).copy()
        hp_rows["T_brine_in_C"] = t_hp_in
        hp_rows["T_brine_out_min_C"] = resource["T_inject_min"]
        hp_rows["temperature_lift_K"] = [
            float(rep_supply[d, h]) - t for (d, h), t in zip(idx, t_hp_in)
        ]
        hp_rows["sink_deltaT_K"] = [
            max(float(rep_supply[d, h]) - float(rep_return[d, h]), 1.0) for d, h in idx
        ]
        # Part-load sweep: COP varies with load in the FlexGeo HP branch; take
        # the minimum over the dispatchable load range (conservative, keeps the
        # MILP linear). hp_out and the sanity-gate COP come from full load.
        fracs = sorted({float(cfg.get("hp_min_load_frac", 0.25)), 0.5, 1.0})
        cop_by_frac = {}
        hp_full = None
        for fr in fracs:
            rows_f = hp_rows.copy()
            rows_f["heating_load_W"] = fr * cfg["module_hp_MWth"] * 1e6
            res = _predict_hp(rows_f, cfg["module_hp_MWth"])
            cop_by_frac[fr] = res["hp_COP"].to_numpy()
            if fr == 1.0:
                hp_full = res
        cop_min = np.min(np.column_stack(list(cop_by_frac.values())), axis=1)
        logger.info(
            f"  HP part-load COP sweep {fracs}: full-load mean "
            f"{cop_by_frac[1.0].mean():.2f}, min-over-fractions mean {cop_min.mean():.2f}"
        )

        df = pd.DataFrame({
            "d": [d for d, _ in idx],
            "h": [h for _, h in idx],
            "orc_q_in_MW": orc["orc_Q_in_W"].values / 1e6,
            "orc_p_net_MW": orc["orc_P_net_W"].clip(lower=0.0).values / 1e6,
            "orc_oob": orc["oob"].astype(int).values,
            "hp_out_MW": hp_full["hp_Q_delivered_MW"].values,
            "hp_cop": cop_min,
            "hp_cop_full": cop_by_frac[1.0],
        })
        df.to_csv(cache_file, index=False)
        logger.info(f"revORC maps cached → {cache_file.name}")

    # ── ORC feasibility gates (manual: infeasible hours → zero output) ──
    amb = np.array([float(rep_ambient[d, h]) for d in range(n_rep) for h in range(n_h)])
    orc_ok = df["orc_oob"].to_numpy() == 0
    n_oob = int((~orc_ok).sum())
    if n_oob:
        logger.warning(
            f"revORC ORC mode: {n_oob}/{len(df)} rep hours outside the surrogate "
            "training bounds — gated off (zero ORC output) instead of extrapolated"
        )
    amb_min = cfg.get("orc_amb_min_C")
    if amb_min is not None:
        amb_ok = amb >= float(amb_min)
        n_cold = int((orc_ok & ~amb_ok).sum())
        if n_cold:
            logger.info(
                f"revORC ORC mode: {n_cold} additional rep hours below "
                f"orc_amb_min_C = {float(amb_min):.1f} °C gated off"
            )
        orc_ok &= amb_ok
    df.loc[~orc_ok, ["orc_q_in_MW", "orc_p_net_MW"]] = 0.0

    # ── HP sanity gate (training COP band; bundle has no stored input bounds).
    #    Gated on the full-load COP; the dispatched (min-over-fractions) COP is
    #    clipped to the band. ──
    cop_ok = (df["hp_cop_full"] >= HP_COP_VALID[0]) & (df["hp_cop_full"] <= HP_COP_VALID[1])
    n_bad = int((~cop_ok).sum())
    hp_enabled = n_bad <= 0.5 * len(df)
    if not hp_enabled:
        logger.warning(
            f"revORC HP mode DISABLED: {n_bad}/{len(df)} rep hours predict COP outside "
            f"{HP_COP_VALID} — Munich conditions (negative temperature lift) are outside "
            "the surrogate's training domain. ORC mode is unaffected."
        )
    elif n_bad:
        logger.warning(f"revORC HP mode: {n_bad}/{len(df)} rep hours gated off (COP outside {HP_COP_VALID})")

    hp_out = df["hp_out_MW"].where(cop_ok, 0.0) if hp_enabled else pd.Series(0.0, index=df.index)
    hp_cop = df["hp_cop"].clip(*HP_COP_VALID)

    maps = {
        "orc_q_in_MW": {(int(r.d), int(r.h)): float(r.orc_q_in_MW) for r in df.itertuples()},
        "orc_p_net_MW": {(int(r.d), int(r.h)): float(r.orc_p_net_MW) for r in df.itertuples()},
        "hp_out_MW": {(int(d), int(h)): float(v) for d, h, v in zip(df["d"], df["h"], hp_out)},
        "hp_cop": {(int(d), int(h)): float(v) for d, h, v in zip(df["d"], df["h"], hp_cop)},
        "hp_enabled": hp_enabled,
        "stats": {
            "orc_p_net_min_MW": float(df["orc_p_net_MW"].min()),
            "orc_p_net_max_MW": float(df["orc_p_net_MW"].max()),
            "orc_eta_net_mean": float(
                (df.loc[orc_ok, "orc_p_net_MW"]
                 / df.loc[orc_ok, "orc_q_in_MW"].clip(lower=1e-9)).mean()
            ) if orc_ok.any() else 0.0,
            "orc_hours_gated": int((~orc_ok).sum()),
            "hp_cop_mean": float(df["hp_cop"].mean()),
            "hp_cop_full_mean": float(df["hp_cop_full"].mean()),
            "hp_hours_gated": n_bad,
        },
    }
    logger.info(
        f"revORC maps: ORC P_net {maps['stats']['orc_p_net_min_MW']*1e3:.0f}–"
        f"{maps['stats']['orc_p_net_max_MW']*1e3:.0f} kW/module "
        f"(η_net ≈ {maps['stats']['orc_eta_net_mean']*100:.1f} %), "
        f"HP mode {'ON (COP mean %.2f)' % maps['stats']['hp_cop_mean'] if hp_enabled else 'OFF'}"
    )
    return maps
