#!/usr/bin/env python3
from __future__ import annotations

"""Prediction wrapper for the source-balanced 1 MW HP off-design surrogate."""

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parent
    / "models"
    / (
        "hp_synthetic_year_surrogate_1mw_filtered_variants_vopt6_"
        "hp55_75_out35_45_sourcebalanced.joblib"
    )
)


def eval_eta_poly(poly: dict[str, Any], pr: np.ndarray) -> np.ndarray:
    eta = np.polyval(np.asarray(poly["coeffs_desc"], dtype=float), np.asarray(pr, dtype=float))
    return np.clip(eta, float(poly["eta_clip_min"]), float(poly["eta_clip_max"]))


def undo_log(y: np.ndarray) -> np.ndarray:
    return np.expm1(np.asarray(y, dtype=float))


def build_feature_frame(
    *,
    T_brine_in_C: float,
    p_brine_bar: float,
    T_brine_out_min_C: float,
    m_brine_max_kg_s: float,
    heating_load_W: float,
    temperature_lift_K: float,
    sink_deltaT_K: float,
    sink_approach_K: float,
    nominal_load_W: float,
    cp_proxy_J_kgK: float,
    input_columns: list[str],
) -> pd.DataFrame:
    row = {
        "T_brine_in_C": float(T_brine_in_C),
        "p_brine_bar": float(p_brine_bar),
        "T_brine_out_min_C": float(T_brine_out_min_C),
        "m_brine_max_kg_s": float(m_brine_max_kg_s),
        "heating_load_W": float(heating_load_W),
        "temperature_lift_K": float(temperature_lift_K),
        "sink_deltaT_K": float(sink_deltaT_K),
    }
    row["T_sink_supply_C"] = row["T_brine_in_C"] + row["temperature_lift_K"]
    row["T_sink_return_C"] = row["T_sink_supply_C"] - row["sink_deltaT_K"]
    row["T_cond_HP_target_C"] = row["T_sink_supply_C"] + float(sink_approach_K)
    row["load_fraction_nominal"] = row["heating_load_W"] / max(float(nominal_load_W), 1.0)
    row["source_deltaT_available_K"] = row["T_brine_in_C"] - row["T_brine_out_min_C"]
    row["geo_capacity_proxy_W"] = row["m_brine_max_kg_s"] * float(cp_proxy_J_kgK) * row["source_deltaT_available_K"]
    row["sink_supply_minus_brine_out_min_K"] = row["T_sink_supply_C"] - row["T_brine_out_min_C"]
    row["cond_target_minus_brine_in_K"] = row["T_cond_HP_target_C"] - row["T_brine_in_C"]
    row["cond_target_minus_brine_out_min_K"] = row["T_cond_HP_target_C"] - row["T_brine_out_min_C"]
    row["sink_deltaT_over_source_deltaT"] = row["sink_deltaT_K"] / row["source_deltaT_available_K"] if abs(row["source_deltaT_available_K"]) > 1e-12 else np.nan
    return pd.DataFrame([{c: row[c] for c in input_columns}])


def compute_variant(bundle: dict[str, Any], X: np.ndarray, q_pred: np.ndarray, variant: str) -> dict[str, float]:
    q_cols = bundle["q_output_columns"]
    cfg = bundle["variant_configs"][variant]
    cols = cfg["cols"]
    log_cols = set(cfg.get("log_cols", []))
    raw = np.asarray(bundle["variant_models"][variant].predict(X), dtype=float)
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)

    out: dict[str, float] = {}
    for i, c in enumerate(q_cols):
        out[c] = float(q_pred[0, i])
    for j, c in enumerate(cols):
        val = raw[0, j]
        if c in log_cols:
            val = undo_log(np.array([val]))[0]
        out[c] = float(val)

    Q = max(float(out["hp_Q_delivered_W"]), 0.0)
    Q_source_surrogate = max(float(out.get("hp_Q_source_W", 0.0)), 0.0)

    if variant == "direct_w":
        W = max(float(out["hp_W_comp_W"]), 1e-9)
    elif variant == "inverse_cop":
        inv = max(float(out["hp_inverse_COP"]), 1e-9)
        W = Q * inv
    elif variant == "pressure_hybrid":
        pcond = max(float(out["hp_p_cond_bar"]), 1e-9)
        pevap = max(float(out["hp_p_evap_bar"]), 1e-9)
        pr = pcond / pevap
        eta = float(eval_eta_poly(bundle["eta_poly"], np.array([pr]))[0])
        Wis = max(float(out["hp_W_is_proxy_W"]), 1e-9)
        W = Wis / max(eta, 1e-9)
        out["hp_pressure_ratio"] = pr
        out["hp_eta_is_fit"] = eta
    else:
        raise ValueError(f"Unknown variant {variant}")

    if Q > 0.0:
        W = float(np.clip(W, 1e-9, Q))
        out["hp_W_comp_W"] = W
        out["hp_inverse_COP"] = W / Q
        out["hp_COP"] = Q / W
        out["hp_Q_source_W"] = Q - W
    else:
        W = 0.0
        out["hp_W_comp_W"] = 0.0
        out["hp_inverse_COP"] = 0.0
        out["hp_COP"] = 0.0
        out["hp_Q_source_W"] = 0.0

    out["hp_Q_source_surrogate_W"] = Q_source_surrogate
    out["hp_energy_balance_residual_W"] = (
        Q - out["hp_Q_source_W"] - out["hp_W_comp_W"]
    )
    return out


def predict_1mw_hp_performance(
    *,
    model_path: str | Path,
    T_brine_in_C: float,
    p_brine_bar: float,
    T_brine_out_min_C: float,
    m_brine_max_kg_s: float,
    heating_load_W: float,
    temperature_lift_K: float,
    sink_deltaT_K: float,
    variant: str | None = None,
    return_all_variants: bool = True,
) -> dict[str, Any]:
    bundle = joblib.load(Path(model_path))
    Xdf = build_feature_frame(
        T_brine_in_C=T_brine_in_C,
        p_brine_bar=p_brine_bar,
        T_brine_out_min_C=T_brine_out_min_C,
        m_brine_max_kg_s=m_brine_max_kg_s,
        heating_load_W=heating_load_W,
        temperature_lift_K=temperature_lift_K,
        sink_deltaT_K=sink_deltaT_K,
        sink_approach_K=float(bundle.get("sink_approach_K", 5.0)),
        nominal_load_W=float(bundle.get("nominal_load_W", 1.0e6)),
        cp_proxy_J_kgK=float(bundle.get("cp_proxy_J_kgK", 4180.0)),
        input_columns=bundle["input_columns"],
    )
    X = Xdf[bundle["input_columns"]].to_numpy(dtype=float)
    q_pred = np.asarray(bundle["q_model"].predict(X), dtype=float)
    if q_pred.ndim == 1:
        q_pred = q_pred.reshape(1, -1)
    selected = variant or bundle.get("selected_variant", "inverse_cop")
    result = {
        "selected_variant": selected,
        "inputs": Xdf.iloc[0].to_dict(),
        "hp": compute_variant(bundle, X, q_pred, selected),
        "warnings": [],
    }
    if return_all_variants:
        result["variants"] = {v: compute_variant(bundle, X, q_pred, v) for v in bundle["variant_models"]}
    return result


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Predict HP off-design performance using filtered/smoothed variant surrogate.")
    p.add_argument("--model", default=str(DEFAULT_MODEL_PATH))
    p.add_argument("--T-brine-in-c", type=float, required=True)
    p.add_argument("--p-brine-bar", type=float, required=True)
    p.add_argument("--T-brine-out-min-c", type=float, required=True)
    p.add_argument("--m-brine-max-kg-s", type=float, required=True)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--heating-load-W", type=float)
    group.add_argument("--heating-load-mw", type=float)
    p.add_argument("--temperature-lift-K", type=float, required=True)
    p.add_argument("--sink-deltaT-K", type=float, required=True)
    p.add_argument("--variant", choices=["direct_w", "inverse_cop", "pressure_hybrid"], default=None)
    p.add_argument("--selected-only", action="store_true")
    return p


def main() -> None:
    args = build_parser().parse_args()
    heating_load_W = args.heating_load_W if args.heating_load_W is not None else float(args.heating_load_mw) * 1e6
    result = predict_1mw_hp_performance(
        model_path=args.model,
        T_brine_in_C=args.T_brine_in_c,
        p_brine_bar=args.p_brine_bar,
        T_brine_out_min_C=args.T_brine_out_min_c,
        m_brine_max_kg_s=args.m_brine_max_kg_s,
        heating_load_W=heating_load_W,
        temperature_lift_K=args.temperature_lift_K,
        sink_deltaT_K=args.sink_deltaT_K,
        variant=args.variant,
        return_all_variants=not bool(args.selected_only),
    )
    print(json.dumps(result, indent=2, default=float))


if __name__ == "__main__":
    main()
