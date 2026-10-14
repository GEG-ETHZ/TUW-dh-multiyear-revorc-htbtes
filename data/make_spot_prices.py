"""Rebuild the electricity price columns from real day-ahead spot prices.

Usage: python make_spot_prices.py [year]   (default 2025)

Selling price  = DE-LU day-ahead spot, hourly shape of the reference year, with
                 the annual mean scaled onto a published projection per milestone
Purchase price = the same spot price + the energy-based grid fee (Arbeitspreis
                 Netzkosten, 10 EUR/MWh). The capacity-based grid fee
                 (Leistungspreis, 60 EUR/kW/yr) is NOT a price adder -- it is a
                 charge on peak electrical draw and is handled in the model.

Source: Bundesnetzagentur / SMARD via api.energy-charts.info, CC BY 4.0.
Writes data/Munich_DH_data_spot2025.xlsx; the original workbook is untouched.

    python make_spot2025_prices.py
"""
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent.resolve()
YEAR = int(sys.argv[1]) if len(sys.argv) > 1 else 2025
SPOT = HERE / "spot_prices" / f"spot_DE_LU_{YEAR}.csv"
SRC = HERE / "Munich_DH_data.xlsx"
DST = HERE / f"Munich_DH_data_spot{YEAR}.xlsx"
YEARS = [2025, 2030, 2035, 2040, 2045]
GRID_FEE_ENERGY = 10.0          # EUR/MWh, Arbeitspreis Netzkosten

# Wholesale price trajectory, EUR/MWh annual mean. The hourly shape of the
# reference year is kept and only its mean is scaled, so the intra-year
# volatility the ORC trades against stays realistic.
#   2025  the reference year itself
#   2030+ Ariadne (Kopernikus/BMBF) "Szenarien zur Klimaneutralitaet 2045",
#         March 2025: ~82 EUR/MWh in 2030, stabilising at 70-80 long term.
MEAN_TRAJECTORY = {2025: 89.0, 2030: 82.0, 2035: 78.0, 2040: 75.0, 2045: 75.0}


def main():
    raw = pd.read_csv(SPOT, parse_dates=[0], index_col=0)["price_EUR_MWh"]
    if len(raw) == 8784:                      # leap year: drop 29 February
        raw = raw[~((raw.index.month == 2) & (raw.index.day == 29))]
        print(f"  {YEAR} is a leap year - 29 February dropped to align with 8760 hours")
    spot = raw.to_numpy()
    assert len(spot) == 8760, f"expected 8760 hourly prices, got {len(spot)}"
    dh = pd.read_excel(SRC, sheet_name="List1")
    assert len(dh) == 8760

    base_mean = spot.mean()
    for y in YEARS:
        target = MEAN_TRAJECTORY[y]
        series = spot * (target / base_mean)          # scale the level, keep the shape
        dh[f"sell_electricity_price_{y}"] = series.round(2)
        dh[f"electricity_price_{y}"] = (series + GRID_FEE_ENERGY).round(2)

    with pd.ExcelWriter(DST, engine="openpyxl") as w:
        dh.to_excel(w, sheet_name="List1", index=False)

    print(f"wrote {DST.name}")
    print(f"  reference year {YEAR}: spot mean {spot.mean():.2f} EUR/MWh, "
          f"scaled onto {MEAN_TRAJECTORY}")
    for y in YEARS:
        a = dh[f"sell_electricity_price_{y}"]; b = dh[f"electricity_price_{y}"]
        print(f"    {y}: sale mean {a.mean():6.2f}  purchase mean {b.mean():6.2f}  "
              f"negative sale hours {(a < 0).sum():4d}")
    s = pd.Series(dh[f"sell_electricity_price_{YEARS[0]}"])
    print(f"  sale (spot):     mean {s.mean():6.2f}  std {s.std():5.1f}  "
          f"min {s.min():7.2f}  max {s.max():6.1f}  negative hours {(s < 0).sum()}")
    b = s + GRID_FEE_ENERGY
    print(f"  purchase (+{GRID_FEE_ENERGY:.0f}): mean {b.mean():6.2f}  "
          f"min {b.min():7.2f}  max {b.max():6.1f}  hours below zero {(b < 0).sum()}")
    print(f"  hours where selling beats buying: {(s > b).sum()} of 8760 (spread is "
          f"always {GRID_FEE_ENERGY:.0f} EUR/MWh)")
    print(f"  monthly mean sale: {[round(v) for v in s.groupby(pd.Series(range(8760))//730).mean()[:12]]}")


if __name__ == "__main__":
    main()
