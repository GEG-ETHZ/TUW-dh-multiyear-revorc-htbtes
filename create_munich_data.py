"""
One-time script: generate Munich_tech_data.xlsx and Munich_DH_data.xlsx
from DhnData.csv + parameters from Kaufmann et al. (ECM 2026).

DhnData.csv  — Southern Munich GHP DHN measurements (Apr 2022 – Mar 2023).
               Provides: heat demand, supply/return temperatures (measured + fixed).

Kaufmann et al., Energy Conversion and Management (2026)
               Provides: system capacity, CAPEX, OPEX, fuel prices, emission factors.

Run once from the project root:
    python create_munich_data.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"


# ── Re-sort DhnData.csv (Apr-Mar) to calendar year (Jan-Dec) ─────────────────
def _resort_jan_dec(series: np.ndarray) -> np.ndarray:
    """
    DhnData.csv starts 01-Apr-2022.  Re-orders any 8760-length column to
    Jan-Dec by moving Jan-Mar 2023 (tail) to the front.
    """
    # Month lengths in DhnData order (Apr 2022 … Mar 2023)
    month_hours = [720, 744, 720, 744, 744, 720, 744, 720, 744, 744, 672, 744]
    # months:      Apr  May  Jun  Jul  Aug  Sep  Oct  Nov  Dec  Jan  Feb  Mar
    boundaries = np.concatenate([[0], np.cumsum(month_hours)])
    # Calendar order: Jan=9, Feb=10, Mar=11, Apr=0, …, Dec=8  (0-indexed)
    cal_order = [9, 10, 11, 0, 1, 2, 3, 4, 5, 6, 7, 8]
    chunks = [series[boundaries[i]: boundaries[i + 1]] for i in cal_order]
    result = np.concatenate(chunks)
    assert len(result) == 8760
    return result


def _load_dhn_columns() -> dict:
    """Load DhnData.csv and re-sort all relevant columns to Jan-Dec."""
    df = pd.read_csv(DATA_DIR / "DhnData.csv")
    return {
        "demand_mw":     _resort_jan_dec(df["Heat_demand_in_kW"].values) / 1000.0,
        "supply_temp":   _resort_jan_dec(df["T_VL_FW_C"].values),         # measured °C
        "return_temp":   _resort_jan_dec(df["T_RL_FW_C"].values),         # measured °C
        "supply_fixed":  _resort_jan_dec(df["T_VL_FW_C_fixed"].values),   # design 90°C
        "return_fixed":  _resort_jan_dec(df["T_RL_FW_C_fixed"].values),   # design 55°C
    }


# ── Munich_DH_data.xlsx ────────────────────────────────────────────────────────
def create_munich_dh_data():
    """
    Build Munich_DH_data.xlsx from DhnData.csv (Kaufmann et al., ECM 2026).
    Columns (8760 rows, Jan-Dec order):
      day, hour                    — datetime index
      corrected_demand_YYYY        — heat demand [MW], with gentle growth
      electricity_price_YYYY       — flat buy price [€/MWh] per period
      sell_electricity_price_YYYY  — flat sell price [€/MWh] per period
      supply_temp                  — measured DHN supply temp T_VL_FW_C [°C]
      return_temp                  — measured DHN return temp T_RL_FW_C [°C]
      supply_temp_fixed            — design supply temp T_VL_FW_C_fixed [°C] (=90)
      return_temp_fixed            — design return temp T_RL_FW_C_fixed [°C] (=55)
    """
    dhn = _load_dhn_columns()
    demand_2025 = dhn["demand_mw"]

    # Gentle demand growth per period (no published projection → conservative 2%/5yr)
    growth = {2025: 1.00, 2030: 1.02, 2035: 1.04, 2040: 1.06, 2045: 1.08}

    # Electricity prices: Kaufmann §2.2.1 industrial price 232.8 €/MWh (2024), declining
    el_buy  = {2025: 233.0, 2030: 215.0, 2035: 195.0, 2040: 175.0, 2045: 155.0}
    el_sell = el_buy.copy()

    n = 8760
    jan1  = pd.Timestamp("2025-01-01")
    dates = pd.date_range(jan1, periods=n, freq="h")

    rows = {
        "day":  [d.strftime("%d.%m.%Y") for d in dates],
        "hour": [d.strftime("%H:%M:%S") for d in dates],
    }
    for yr, g in growth.items():
        rows[f"corrected_demand_{yr}"] = demand_2025 * g
    for yr, p in el_buy.items():
        rows[f"electricity_price_{yr}"] = np.full(n, p)
    for yr, p in el_sell.items():
        rows[f"sell_electricity_price_{yr}"] = np.full(n, p)
    # Supply/return temperatures from DhnData.csv (Kaufmann measurements)
    rows["supply_temp"]       = dhn["supply_temp"]
    rows["return_temp"]       = dhn["return_temp"]
    rows["supply_temp_fixed"] = dhn["supply_fixed"]
    rows["return_temp_fixed"] = dhn["return_fixed"]

    df   = pd.DataFrame(rows)
    path = DATA_DIR / "Munich_DH_data.xlsx"
    df.to_excel(path, index=False, sheet_name="List1")
    print(f"Created {path}  ({n} rows)")
    print(f"  Peak demand 2025:      {demand_2025.max():.2f} MW")
    print(f"  Mean demand 2025:      {demand_2025.mean():.2f} MW")
    print(f"  Supply temp: {dhn['supply_temp'].min():.1f}–{dhn['supply_temp'].max():.1f} °C "
          f"(mean {dhn['supply_temp'].mean():.1f} °C)")
    print(f"  Return temp: {dhn['return_temp'].min():.1f}–{dhn['return_temp'].max():.1f} °C "
          f"(mean {dhn['return_temp'].mean():.1f} °C)")


# ── Munich_tech_data.xlsx ──────────────────────────────────────────────────────
def create_munich_tech_data():
    """
    Build Munich_tech_data.xlsx with DEA Technology Catalogue (May 2025) costs.
    Sheet structure mirrors Kranj_tech_data.xlsx (same column order).
    """
    path = DATA_DIR / "Munich_tech_data.xlsx"

    # ── Heat Generators ────────────────────────────────────────────────────────
    # All capacity and cost parameters from DEA Technology Catalogue (May 2025).
    gen_header = [
        "name", "energy carrier", "installed capacity (MW_th)",
        "efficiency th", "efficiency el",
        "investment costs (EUR/MW_th)", "OPEX fix (EUR/MWa)", "OPEX var (EUR/MWh)",
        "Start up time (h)", "Coldstart costs (EUR/Start)", "Ramping costs [EUR/MWh]",
        "Min Load factor", "Installation Year", "life time",
        "Maximal Available Potential [MWh]", "Maximal Output Power [MW]",
    ]
    gen_data = [
        # ── Geothermal Heat — existing asset (Kaufmann §2.1.1) ────────────────
        # Brine: 81.52 kg/s at 106°C
        # Direct Use mode (no HTHP): brine 106→55°C → Q = 81.52×4.18×51/1000 = 17.4 MW
        # Capacity Increase mode (HTHP on): brine 106→65°C (direct HX) + 65→45°C (HTHP evap)
        #   → Q_geo_direct = 81.52×4.18×41/1000 = 14.0 MW (remaining 3.4 MW extracted by HTHP)
        # Min brine flow: 53 kg/s (§2.1.3) = 65% of 81.52 kg/s
        #   DU mode min: 53×4.18×51/1000 = 11.3 MW | CI mode min: 53×4.18×41/1000 = 9.1 MW
        # installed capacity = DU-mode max; CI-mode cap reduction is handled by y_ci binary in run_rd.py
        #
        # DEA Technology Catalogue (May 2025), sheet 45.4 "Geothermal DH, large scale":
        #   CAPEX wells (0.65 MEUR/MW) + surface (0.39 MEUR/MW) = 1.04 MEUR/MW
        #   OPEX_fix prorated: 18,255 × (1.04/1.30) = 14,604 EUR/MW/yr
        #   OPEX_var: 2.02 EUR/MWh | Lifetime: 25 yr
        ["Geothermal Heat", "geothermal", 17.4, 1.0, 0,
         1_040_000, 14_604, 2.02, 0, 0, 0, 0.65, 2010, 25, 0, 17.4],

        # ── HTHP — High-Temperature Heat Pump (existing, SAPHEA Capacity Increase) ──
        # Coupled to geothermal well: brine 65→45°C through HTHP evaporator
        # SAPHEA model (Saphea_LSGEOHP-tool.html):
        #   COP_SAPHEA  = η × T_supply_K / (T_supply − T_inject_min)  (η = 0.4212)
        #   W_elec      = Q_evap / COP_SAPHEA
        #   Q_HP        = Q_evap + W_elec   →   COP_standard = COP_SAPHEA + 1
        # Design point (T_supply=90°C, T_inject_min=45°C):
        #   COP_SAPHEA = 0.4212×363.15/45 = 3.40  →  COP_standard = 4.40
        #   Q_evap = 81.52×4.18×(65−45)/1000 = 6.82 MW
        #   Q_HP   = 6.82×(1+1/3.40) = 8.82 MW
        # CAPEX from SAPHEA cost regression (×0.8 factor):
        #   specific_cost(8.82 MW) = interp(8.82, {3:910k,10:710k}) × 0.8 ≈ 595,000 €/MW_HP
        #   investment = Q_evap × specific_cost = 6.82×595,000 = 4,058,000 €
        #   CAPEX per MW_HP = 4,058,000 / 8.82 ≈ 460,000 €/MW_HP
        # efficiency_th = nominal COP_standard at design point (patched per (d,h) in run_rd.py)
        ["HTHP", "electricity", 8.82, 4.40, 0,
         460_000, 4_600, 0, 0, 0, 0, 0, 2010, 25, 0, 8.82],

        # ── Gas Boiler — district-heating only ────────────────────────────────
        # DEA Technology Catalogue (May 2025), sheet 44 "Natural Gas DH Only":
        #   CAPEX: 63,800 EUR/MW | OPEX_fix: 2,074 EUR/MW/yr | OPEX_var: 1.06 EUR/MWh
        #   Efficiency: 1.03 (condensing, LHV basis) | Lifetime: 25 yr
        ["Gas Boiler", "natural gas", 0, 1.03, 0,
         63_800, 2_074, 1.06, 0, 0, 0, 0, None, 25, 0, 30],

        # ── Electric Boiler — large-scale ─────────────────────────────────────
        # DEA Technology Catalogue (May 2025), sheet 41 "Electric boiler, large":
        #   CAPEX: 74,436 EUR/MW | OPEX_fix: 1,138 EUR/MW/yr | OPEX_var: 0.53 EUR/MWh
        #   Efficiency: 0.99 | Lifetime: 25 yr
        ["Electric Boiler", "electricity", 0, 0.99, 0,
         74_436, 1_138, 0.53, 0, 0, 0, 0, None, 25, 0, 30],
    ]
    # Build the exact two-row header structure that Kranj uses (row-0 = NaN + header names)
    # Kranj raw: row 0 = [NaN, "name", "energy carrier", ...], rows 1-N = [idx, values]
    header_row = pd.DataFrame([[None] + gen_header], columns=range(len(gen_header) + 1))
    data_rows  = pd.DataFrame(
        [[i] + gen_data[i] for i in range(len(gen_data))],
        columns=range(len(gen_header) + 1),
    )
    df_gen = pd.concat([header_row, data_rows], ignore_index=True)

    # ── Heat Storage ───────────────────────────────────────────────────────────
    # The Kaufmann (ECM 2026) GHP system has NO short-term thermal storage.
    # Empty sheet — no storage in this case.
    stor_header = [
        "name", "Storage Capacity [MWh]", "Hourly Stoarge Losses [%]",
        "Minimum Charge Time [h]", "Minimum Discharge Time [h]",
        "loading efficiency", "unloading efficiency",
        "Invesment costs for additional storage capacity  [€/MWh]",
        "OPEX fix [€/MWh]", "Installation Year", "Life Time [a]",
        "Maximum Storage Capacity [MWh]",
    ]
    stor_data = []  # no storage in Kaufmann GHP system
    stor_header_row = pd.DataFrame([[None] + stor_header], columns=range(len(stor_header) + 1))
    stor_data_rows  = pd.DataFrame(columns=range(len(stor_header) + 1))
    df_stor = pd.concat([stor_header_row, stor_data_rows], ignore_index=True)

    # ── Prices and emission factors ────────────────────────────────────────────
    # Gas price: Kaufmann §2.2.1: 67.8 €/MWh (2024 industrial), escalating ~3%/yr
    # Gas emission factor: 0.202 tCO₂/MWh (Kaufmann §2.2.2)
    # Electricity EF: German grid, decarbonising over planning horizon
    price_header = [
        "energy carrier",
        "prices_2025(EUR/MWh)", "prices_2030(EUR/MWh)",
        "prices_2035(EUR/MWh)", "prices_2040(EUR/MWh)", "prices_2045(EUR/MWh)",
        "emission factor_2025 [tCO2/MWh]", "emission factor_2030 [tCO2/MWh]",
        "emission factor_2035 [tCO2/MWh]", "emission factor_2040 [tCO2/MWh]",
        "emission factor_2045 [tCO2/MWh]",
    ]
    price_data = [
        # Geothermal: free, zero emissions
        ["geothermal",   0,    0,    0,    0,    0,   0,    0,    0,    0,    0],
        # Electricity: price comes from DH_data (hourly), EF declines with grid greening
        ["electricity",  0,    0,    0,    0,    0, 0.233, 0.20, 0.17, 0.14, 0.08],
        # Natural gas: Kaufmann 67.8 €/MWh 2024 → escalating ~3%/yr
        ["natural gas", 67.8, 72.0, 76.5, 81.4, 86.7, 0.202, 0.202, 0.202, 0.202, 0.202],
    ]
    price_header_row = pd.DataFrame([[None] + price_header], columns=range(len(price_header) + 1))
    price_data_rows  = pd.DataFrame(
        [[i] + price_data[i] for i in range(len(price_data))],
        columns=range(len(price_header) + 1),
    )
    df_price = pd.concat([price_header_row, price_data_rows], ignore_index=True)

    # ── Extra parameters ───────────────────────────────────────────────────────
    # Discount rate: Kaufmann Table A.4 / §A.4:
    #   ECB average interest rate 1.31% (ref [83]), Eurozone inflation 2.13% (ref [82])
    #   Nominal ≈ 1.31% + 2.13% = 3.44% → rounded to 3.5% for planning model.
    # Carbon price: EUA 70 €/tCO₂ (Kaufmann §2.2.2, late 2024); escalating per EU ETS trajectory.
    extra_data = {
        "name":          ["discount rate", "carbon price"],
        2025:            [0.035,   70.0],
        2030:            [0.035,  120.0],
        2035:            [0.035,  200.0],
        2040:            [0.035,  300.0],
        2045:            [0.035,  400.0],
    }
    df_extra = pd.DataFrame(extra_data)

    # ── HP COP parameters ──────────────────────────────────────────────────────
    # No heat pumps in this model — empty sheet.
    df_hp = pd.DataFrame(columns=[
        "type", "COP source temperature sensitivity (1/°C)",
        "COP flow temperature sensitivity (1/°C)",
        "COP return temperature sensitivity (1/°C)",
        "min shutoff source temperature (°C)",
        "fade out temp where power reduction starts (°C)",
        "Source Temperature (°C)", "Nominal Inlet Temperature (°C)",
        "Nominal Return Temperature (°C)", "Refrigerants",
    ])

    # ── Write all sheets ───────────────────────────────────────────────────────
    with pd.ExcelWriter(path, engine="openpyxl") as w:
        df_gen.to_excel(w,   sheet_name="Heat Generators",              index=False, header=False)
        df_stor.to_excel(w,  sheet_name="Heat Storage",                 index=False, header=False)
        df_price.to_excel(w, sheet_name="prices and emmision factors",  index=False, header=False)
        df_extra.to_excel(w, sheet_name="extra",                        index=False)
        df_hp.to_excel(w,    sheet_name="HP COP",                       index=False)

    print(f"Created {path}")
    print("  Technologies: Geothermal Heat (existing 17.4 MW DU / 14.0 MW CI), "
          "HTHP (existing 8.82 MW, SAPHEA), Gas Boiler, Electric Boiler")
    print("  Storage:      none (Kaufmann GHP system has no STTS)")


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Creating Munich data files …")
    create_munich_tech_data()
    create_munich_dh_data()
    print("Done.")
