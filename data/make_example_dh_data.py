"""Build a synthetic hourly input workbook so the model can run without the measured Munich data.

The measured network data of the Munich plant cannot be published. This script writes
``data/example_DH_data.xlsx`` with the same sheet and column layout, filled with generic,
reproducible profiles that are NOT derived from the measured series:

  ambient_temp              seasonal + daily sine with seeded noise (Central European range)
  corrected_demand_<year>   heating-curve demand: base load + slope x (T_heating_limit - T_amb),
                            with a weekday/night modulation; grown 2 % per milestone
  supply_temp               network heating curve, 75 C at mild weather rising to 105 C at -12 C
  return_temp               55 C with a small seasonal swing
  supply_temp_fixed / return_temp_fixed   design values 90 / 55 C
  electricity_price_<year>, sell_electricity_price_<year>
                            public DE-LU day-ahead prices of a reference year (data/spot_prices/),
                            level scaled per milestone, buy = sell + 10 EUR/MWh grid energy fee

Usage:  python data/make_example_dh_data.py [spot_year]     (default 2025)
Then:   DH_DATA=example_DH_data.xlsx python run_rd.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent.resolve()
YEAR_SPOT = int(sys.argv[1]) if len(sys.argv) > 1 else 2025
MILESTONES = [2025, 2030, 2035, 2040, 2045]
MEAN_TRAJECTORY = {2025: 89.0, 2030: 82.0, 2035: 78.0, 2040: 75.0, 2045: 75.0}  # EUR/MWh
GRID_FEE_ENERGY = 10.0                                                            # EUR/MWh
DEMAND_GROWTH = 0.02                                                              # per milestone
PEAK_MW, BASE_MW = 25.0, 2.0                                                      # target scale
T_HEAT_LIMIT = 15.0                                                               # C

rng = np.random.default_rng(42)
idx = pd.date_range("2025-01-01", periods=8760, freq="h")
h = np.arange(8760)
doy = idx.dayofyear.to_numpy()
hod = idx.hour.to_numpy()

# ── ambient temperature ─────────────────────────────────────────────────────
t_season = 10.0 - 12.0 * np.cos(2 * np.pi * (doy - 15) / 365.0)        # -2 C mid-Jan, 22 C mid-Jul
t_daily = 5.0 * np.sin(2 * np.pi * (hod - 9) / 24.0)                   # warmest ~15:00
t_noise = pd.Series(rng.normal(0, 2.5, 8760)).rolling(24, min_periods=1, center=True).mean().to_numpy()
t_amb = np.round(t_season + t_daily + t_noise, 2)

# ── heat demand: heating curve + domestic hot water base + daily modulation ──
deg = np.clip(T_HEAT_LIMIT - t_amb, 0, None)
day_mod = 1.0 + 0.12 * np.sin(2 * np.pi * (hod - 7) / 24.0) + 0.06 * (hod >= 6) * (hod <= 9)
raw = BASE_MW + deg * day_mod
demand = BASE_MW + (raw - BASE_MW) * (PEAK_MW - BASE_MW) / (raw.max() - BASE_MW)
demand = np.round(demand, 3)

# ── network temperatures ────────────────────────────────────────────────────
supply = np.clip(75.0 + (105.0 - 75.0) * (T_HEAT_LIMIT - t_amb) / (T_HEAT_LIMIT + 12.0), 75.0, 105.0)
supply = np.round(supply, 2)
ret = np.round(55.0 + 2.0 * np.cos(2 * np.pi * (doy - 15) / 365.0), 2)

# ── prices from the public day-ahead series ─────────────────────────────────
spot_file = HERE / "spot_prices" / f"spot_DE_LU_{YEAR_SPOT}.csv"
spot = pd.read_csv(spot_file, parse_dates=[0], index_col=0)["price_EUR_MWh"]
if len(spot) == 8784:
    spot = spot[~((spot.index.month == 2) & (spot.index.day == 29))]
spot = spot.to_numpy()[:8760]
assert len(spot) == 8760, f"expected 8760 prices in {spot_file.name}, got {len(spot)}"

# ── assemble the workbook ───────────────────────────────────────────────────
out = pd.DataFrame({"day": idx.strftime("%d.%m.%Y"), "hour": idx.strftime("%H:%M:%S")})
for i, y in enumerate(MILESTONES):
    out[f"corrected_demand_{y}"] = np.round(demand * (1 + DEMAND_GROWTH * i), 3)
for y in MILESTONES:
    sell = spot * (MEAN_TRAJECTORY[y] / spot.mean())
    out[f"electricity_price_{y}"] = np.round(sell + GRID_FEE_ENERGY, 2)
for y in MILESTONES:
    out[f"sell_electricity_price_{y}"] = np.round(spot * (MEAN_TRAJECTORY[y] / spot.mean()), 2)
out["supply_temp"] = supply
out["return_temp"] = ret
out["supply_temp_fixed"] = 90.0
out["return_temp_fixed"] = 55.0
out["ambient_temp"] = t_amb

dst = HERE / "example_DH_data.xlsx"
with pd.ExcelWriter(dst, engine="openpyxl") as w:
    out.to_excel(w, sheet_name="List1", index=False)
print(f"wrote {dst.name}: demand {demand.min():.1f}-{demand.max():.1f} MW (mean {demand.mean():.1f}), "
      f"supply {supply.min():.0f}-{supply.max():.0f} C, ambient {t_amb.min():.1f}-{t_amb.max():.1f} C, "
      f"spot {YEAR_SPOT} mean {spot.mean():.1f} EUR/MWh")
