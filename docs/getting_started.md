# Getting Started

This guide takes you from installation to a first run with the example data, explains the
inputs and outputs, and then shows how the paper runs are reproduced and how to set the model up
for your own system.

* [1. What the model does](#1-what-the-model-does)
* [2. Installation](#2-installation)
* [3. A first run](#3-a-first-run)
* [4. Scenarios](#4-scenarios)
* [5. Input data](#5-input-data)
* [6. Environment variables](#6-environment-variables)
* [7. Output](#7-output)
* [8. Reproducing the paper runs](#8-reproducing-the-paper-runs)
* [9. The revORC component](#9-the-revorc-component)
* [10. The HT-BTES component](#10-the-ht-btes-component)
* [11. Setting the model up for your own system](#11-setting-the-model-up-for-your-own-system)
* [12. Troubleshooting](#12-troubleshooting)

---

## 1. What the model does

The model plans a district-heating (DH) supply system over a multi-decade horizon. At each
investment milestone (the year columns of the `extra` sheet, 2025–2045 in 5-year steps for the
Munich case) it may add capacity of any technology or storage in the input workbook; between
milestones the installed capacity is a non-decreasing stock. Within every milestone year the
plant is dispatched hour by hour against the demand, network temperatures and prices of that
year. All decisions are taken together in one Mixed-Integer Linear Programme (MILP) that
minimises the **total discounted system cost** over the horizon: investment (lump sum, with a
salvage credit for remaining life at the end of the horizon), fixed and variable O&M, fuel,
purchased electricity, CO2, minus revenue from sold electricity.

The base system of the Munich case is a geothermal doublet (must-run, brine at 106 °C) with a
direct heat exchanger, and a gas boiler. The geothermal plant has three operating modes, direct
use, heat-pump assisted and combined, documented in
[SAPHEA_modes_and_integration.md](SAPHEA_modes_and_integration.md). On top of this the model
can add:

| Option | Model element |
|---|---|
| revORC | modular reversible ORC / heat pump on the brine; surrogate-based hourly maps, mutually exclusive ORC / HP mode per hour, dead hours when switching, grid capacity fee on peak electrical draw |
| HTHP | high-temperature heat pump on the brine tail (kept in the code, not part of the paper) |
| HT-BTES | high-temperature borehole storage with a temperature state, age-dependent coefficients, charge at wellhead temperature, direct-use discharge rule |
| PTES | pit thermal energy storage as an energy-balance storage with losses and a fixed energy-to-power ratio |
| buffer tank | short-term storage (switched off in the Munich scenarios, `KEEP_BUFFER=1` restores it) |

Three temporal formulations share the same constraints:

| Script | Model | Time structure | Use |
|---|---|---|---|
| `run_full.py` | `src/models/full_model.py` | all 8,760 h of every milestone year | reported results |
| `run_rd.py` | `src/models/rd_model.py` | representative days (tsam), storage by Kotzur superposition of intra-day and inter-day states | design screening, field-size enumeration, sensitivities |
| `run_rh.py` | `src/models/rh_model.py` | representative hours with a reduced storage index | experimental |

## 2. Installation

1. Choose a solver with the environment variable `SOLVER`. The default, `appsi_highs`, is the
   open-source HiGHS solver (`highspy`, installed with the environment, no licence). For
   `SOLVER=gurobi` install [Gurobi](https://www.gurobi.com/downloads/) (`pip install gurobipy`)
   and a licence; academic users get a free licence with `grbgetkey`. Gurobi is what the paper
   runs used and what the large HT-BTES and full 8,760 h runs need in practice; with HiGHS only
   the MIP gap and the time limit are passed on, not the Gurobi tuning parameters.
2. Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html).
3. Download or clone the repository and, in its folder:

   ```bash
   conda env create -f environment.yml
   conda activate dh-multiyear
   python -c "import gurobipy; print(gurobipy.gurobi.version())"   # only for SOLVER=gurobi
   ```

   If the last command complains about the licence, set
   `export GRB_LICENSE_FILE=/path/to/gurobi.lic` (on Windows: `set GRB_LICENSE_FILE=...`).

4. The hourly example workbook `data/example_DH_data.xlsx` provides demand, network and
   ambient temperatures and prices with synthetic profiles (section 5.2) so that every scenario
   can be run and the workflow learned. `python data/make_example_dh_data.py` regenerates it.

5. For the revORC scenarios copy the four surrogate bundles into
   `revers_surrogates/models/` (names and layout in
   `revers_surrogates/README_reversible_ORC_HP_wrappers.txt`). They are available on request.

Tested with Python 3.14, Pyomo 6.10, Gurobi 13.0, pandas 3.0, numpy 2.4, scikit-learn 1.8,
tsam 3.2, pygfunction 2.3 on Linux. The code needs Python ≥ 3.10.

## 3. A first run

```bash
export DH_DATA=example_DH_data.xlsx
SCENARIO=N1 GRB_THREADS=8 python run_rd.py
```

This solves the base system (geothermal + gas boiler) on representative days. The log shows the
data loading (with a warning that the default hourly workbook was overridden, which is expected),
the clustering summary, the model size, the Gurobi log and the reporting steps, and ends with

```
Objective (NPV): EUR xx,xxx,xxx
Done. Results saved to: .../results/rd/MILP_<timestamp>_N1
```

(the label "NPV" in the log means the total discounted system cost). The run folder is
described in section 7. On a shared machine start long runs with `./run_nice.sh python
run_full.py ...` so they take the lowest CPU priority.

## 4. Scenarios

`SCENARIO` selects which options are offered to the optimiser. Every scenario shares the base
system; one option is added at a time and all of them together in N6.

| `SCENARIO` | Options offered | Paper label | Notes |
|---|---|---|---|
| `N1` | none | S1 | base system |
| `N2` | revORC | S2 | needs the surrogate bundles |
| `N3` | HTHP | – | not part of the paper |
| `N4` | HT-BTES | S3 | field size fixed with `BTES_DESIGN_N=<boreholes>`; `BTES_OFF=1` gives the no-field reference; built in the first milestone |
| `N5` | PTES | S4 | |
| `N6` | revORC + HT-BTES + PTES (+ HTHP investable) | S5 | HTHP is built at 0 MW in the paper runs |

`S1`–`S5` are an older scenario suite kept for backward compatibility (`S1` no storage, `S2`
HT-BTES only, `S3` PTES only, `S4` HT-BTES with endogenous field choice, `S5` everything).

The HT-BTES field size is **enumerated, not co-optimised**: one solve per borehole count from
the design library, and the design curve (cost vs. field size) is read from the set of runs.
This keeps the MILP small and gives the full curve rather than one point.

## 5. Input data

Two workbooks in `data/` describe the case completely. The files the scripts read are chosen
with `TECH_DATA` and `DH_DATA` (defaults `Munich_tech_data.xlsx`, `Munich_DH_data.xlsx`). The
paper uses `Munich_tech_data_v3.xlsx` and `Munich_DH_data_spot2025.xlsx`; the example workbook
`example_DH_data.xlsx` stands in for the hourly file (section 5.2).

### 5.1 Technology workbook (`Munich_tech_data*.xlsx`)

**Sheet `Heat Generators`** — one row per technology.

| Column | Meaning |
|---|---|
| `name` | technology name used everywhere (results, plots) |
| `energy carrier` | must match a row of the prices sheet (`geothermal`, `electricity`, `natural gas`, …) |
| `installed capacity (MW_th)` | existing capacity at the start (0 = greenfield) |
| `efficiency th`, `efficiency el` | thermal efficiency (COP for heat pumps), electrical efficiency (CHP) |
| `investment costs (EUR/MW_th)` | specific CAPEX |
| `OPEX fix (EUR/MWa)`, `OPEX var (EUR/MWh)` | fixed O&M per MW and year, variable O&M per MWh heat |
| `Start up time (h)`, `Coldstart costs (EUR/Start)`, `Ramping costs [EUR/MWh]`, `Min Load factor` | unit-commitment parameters; all zero switches unit commitment off |
| `life time` | technical lifetime in years (annuity, salvage) |
| `Maximal Available Potential [MWh]` | annual generation cap (0 = none) |
| `Maximal Output Power [MW]` | capacity cap (0 = none); for the geothermal and HTHP rows this is overridden by the resource-derived capacities |
| `Installation Year` | optional; year an existing unit was built |

**Sheet `Heat Storage`** — one row per storage type (energy-balance storages).

| Column | Meaning |
|---|---|
| `name` | storage name; rows containing "buffer" are the short-term tank |
| `Storage Capacity [MWh]` | existing capacity |
| `Hourly Stoarge Losses [%]` | self-discharge per hour |
| `Minimum Charge Time [h]`, `Minimum Discharge Time [h]` | energy-to-power ratio τ: charge/discharge power ≤ capacity / τ |
| `loading efficiency`, `unloading efficiency` | charge / discharge efficiency |
| `Invesment costs for additional storage capacity  [€/MWh]` | specific CAPEX |
| `OPEX fix [€/MWh]`, `OPEX var [€/MWh]` | fixed O&M per MWh capacity and year, variable per MWh throughput |
| `Life Time [a]`, `Maximum Storage Capacity [MWh]`, `Installation Year` | lifetime, capacity cap, optional build year |

**Sheet `prices and emmision factors`** — one row per energy carrier with
`prices_<year>(EUR/MWh)` and `emission factor_<year> [tCO2/MWh]` per milestone year. The
electricity price is **not** taken from here but from the hourly columns of the hourly workbook;
the electricity emission factor is.

**Sheet `extra`** — rows `discount rate` and `carbon price` (EUR/tCO2) per milestone year,
`n_typical_days` (representative days per milestone for `run_rd.py`) and `hthp_capex_scale`.
**The year columns of this sheet define the investment milestones.**

**Sheet `Geothermal Resource`** — `brine_mass_flow` (kg/s), `brine_cp`, `T_prod` (wellhead
°C), `T_inject_min`, `T_return_design`, `direct_hx_pinch`, `carnot_eta`, `T_supply_design`,
`p_brine_bar`. The geothermal and HTHP capacities and the HTHP COP are derived from these, not
typed in.

**Sheet `HTHP CAPEX Curve`** — size-dependent specific cost of the HTHP (`size_MW`,
`specific_cost_EUR_per_MW`).

**Sheet `RevORC`** — parameter / value / unit / source rows:

| Parameter | Meaning |
|---|---|
| `enabled` | 0 disables the revORC entirely |
| `module_brine_kg_s`, `module_hp_MWth`, `module_orc_MW_el` | module basis: brine flow, HP heat output, rated ORC output per module |
| `T_brine_out_min_C` | brine outlet floor of the module |
| `capex_eur_per_kW_el` | CAPEX on the electrical basis; converted to EUR/MW_th with `module_orc_MW_el / module_hp_MWth`. If absent, `capex_eur_per_MWth` is used |
| `opex_fix_frac`, `opex_var_orc_eur_per_MWh_el`, `opex_var_hp_eur_per_MWh_th` | fixed O&M as fraction of CAPEX, variable O&M per mode |
| `grid_capacity_fee_eur_per_kW_yr` | capacity-based grid charge on the peak electrical draw |
| `lifetime_a`, `max_modules`, `exclusive_with_hthp` | lifetime, module cap, either/or with the HTHP |
| `brine_pump_el_per_MWth`, `dead_hours`, `orc_amb_min_C`, `hp_min_load_frac` | pumping electricity, switching dead time, ORC ambient floor, lowest HP part load |

### 5.2 Hourly workbook (sheet `List1`)

8,760 rows with `day` (`DD.MM.YYYY`) and `hour` (`HH:MM:SS`) and the columns

| Column | Meaning |
|---|---|
| `corrected_demand_<year>` | heat demand in MW per milestone year |
| `electricity_price_<year>` | purchase price EUR/MWh (hourly) |
| `sell_electricity_price_<year>` | selling price EUR/MWh (hourly) |
| `supply_temp`, `return_temp` | network supply / return temperature °C |
| `supply_temp_fixed`, `return_temp_fixed` | design values, constant |
| `ambient_temp` | ambient temperature °C (required by the revORC ORC mode) |

**Building a workbook from measurements.** `create_munich_data.py` shows how such a workbook
is built from a year of measured network data (demand, supply and return temperature, re-sorted
to a calendar year), and `data/make_spot_prices.py` how the price columns are filled with the
public DE-LU day-ahead prices of a reference year (`data/spot_prices/`), hourly shape kept and
the annual mean scaled onto a projection per milestone.

**Example workbook (included).** `data/example_DH_data.xlsx`, written by
`data/make_example_dh_data.py`, holds synthetic profiles that do not derive from the
measured data: a sinusoidal ambient temperature with seeded noise, a heating-curve demand
(base load plus degree-hours with a daily modulation, scaled to a 25 MW peak and grown 2 % per
milestone), a network heating curve from 75 °C in mild weather to 105 °C at −12 °C, a 55 °C
return, and the public spot prices scaled per milestone with a 10 EUR/MWh energy grid fee on
the purchase side. Select it with `DH_DATA=example_DH_data.xlsx`.

### 5.3 HT-BTES libraries (`data/btes/`)

| File | Content |
|---|---|
| `design_library<suffix>.json` | per borehole count and age block: initial temperature, charge and discharge power limits, CAPEX, geometry |
| `btes_hourly_params<suffix>_n<N>.json` | per borehole count: hourly coefficients per age block, bounds, costs, optional wall feedthrough and discharge-flow levels |

Suffixes: none = measured network, fixed discharge flow; `_s10` = network 10 K colder;
`_vfa` / `_vf` = measured / colder network with the borehole-wall feedthrough (used in the
paper). `BTES_LIB_SUFFIX` selects the set; `BTES_DESIGN_N` the borehole count (50 … 500).

## 6. Environment variables

Case and solver settings:

| Variable | Default | Meaning |
|---|---|---|
| `SCENARIO` | `S3` (rd) / `S1` (full) | see section 4 |
| `TECH_DATA`, `DH_DATA` | `Munich_tech_data.xlsx`, `Munich_DH_data.xlsx` | workbooks in `data/`; set `DH_DATA=example_DH_data.xlsx` unless you have the Munich data |
| `NET_TEMP_SHIFT_K` | 0 | shift measured supply and return temperatures down by this many K |
| `SOLVER` | `appsi_highs` | Pyomo solver name: `appsi_highs` (HiGHS) or `gurobi`; see section 2 |
| `GRB_THREADS` | 40 | solver threads (Gurobi only) |
| `GRB_MIP_GAP` (rd) / `MIPGAP` (full) | 0.005 / 0.0025 | relative MIP gap |
| `GRB_TIME_LIMIT` (rd) / `TIMELIMIT_S` (full) | 1800 / 172800 | time limit in seconds |
| `WARM_START` | – | RD run folder (or its `results_rd.xlsx`) whose capacities seed the full model |
| `RESULTS_SUBDIR` (full), `RUN_TAG` | `full`, `` | output folder under `results/`, tag appended to the folder name |
| `RESULTS_ROOT` (rd, rh) | repository folder | folder in which `results/` is created |
| `NO_REPORT` | – | `1` skips plots and workbook |

Temporal aggregation (`run_rd.py`, `run_rh.py`):

| Variable | Default | Meaning |
|---|---|---|
| `N_TYPICAL_DAYS` | `extra` sheet | representative days per milestone |
| `TSA_MODE` | `per_period` | cluster each milestone on its own series, or `joint` |
| `TSA_PEAK`, `TSA_PRICE_WEIGHT` | 1, 1 | keep the peak day (`0` drops it), weight of the price series in the clustering |
| `N_REP_HOURS`, `RH_SEGMENTED`, `RH_SEASON_PURE` | – | representative-hour settings |

revORC:

| Variable | Default | Meaning |
|---|---|---|
| `REVORC` | sheet `enabled` | `0` switches the revORC off |
| `REVORC_EXCLUSIVE` | 1 | revORC and HTHP cannot both be built |
| `REVORC_DEAD_HOURS`, `REVORC_CAPEX_SCALE` | sheet | override dead hours, scale the CAPEX |
| `GRID_CAP_FEE` | sheet | capacity grid charge EUR/kW_el/a (`0` disables) |
| `FORCE_REVORC_MW`, `FORCE_HTHP_MW`, `FORCE_CI_ABOVE` | – | fix capacities / modes for diagnostics |

HT-BTES:

| Variable | Default | Meaning |
|---|---|---|
| `BTES_DESIGN_N` | – | borehole count from the library |
| `BTES_OFF` | – | `1`: no field in an N4/N6 run |
| `BTES_LIB_SUFFIX` | `` | library set, see 5.3 |
| `BTES_TEMP_RULE` | off | `1`: direct-use discharge rule (storage wall above network supply + approach) |
| `BTES_FLOW_CONTINUOUS` | off | `1`: continuous discharge flow (convex hull of the flow-UA relation) |
| `BTES_TIGHT_M` | – | state lower bound (°C) after warm-up used to tighten the rule's big-M |
| `BTES_FLOW_LEVELS` | off | `1`: discrete discharge-flow levels with binaries (RD only) |
| `BTES_MIPSTART`, `BTES_LEVEL_WARMSTART`, `BTES_WARMSTART_GAP`, `BTES_WARMSTART_TIMELIMIT` | – | solver warm-start helpers |
| `BTES_PUMP_COST`, `BTES_BASE_FLOW`, `BTES_PRUNE`, `BTES_FORCE_DESIGN`, `BTES_DEBUG`, `BTES_IIS`, `BTES_REPORT` | – | sensitivities and diagnostics |

Prices and other sensitivities: `CO2_MULT`, `CO2_PRICE`, `ELEC_MULT`, `ELEC_PRICE`,
`SELL_PRICE`, `GAS_PRICE`, `T_INJECT_MIN`, `GEO_SHUTDOWN`, `HTHP_CAPEX_EXCEL`, `KEEP_BUFFER`.

## 7. Output

Each run writes `results/<rd|full|rh|RESULTS_SUBDIR>/MILP_<timestamp>_<SCENARIO><tags>/`:

| File | Content |
|---|---|
| `results_rd.xlsx` | sheets `summary` (objective, model size), `computational`, `capacity`, `generation`, `cost_co2_breakdown`, `co2_by_tech`, `running_cost_by_tech`, `capex_by_tech`, `fuel_cost_by_tech`, `dispatch_<year>`, `mode_stats`, `mode_hours`, `mode_heat_gen`, `hthp_economics`, `mode_dispatch_<year>` |
| `dispatch_hourly_<year>.csv` | hourly dispatch per technology, storage charge/discharge, reinjected heat, revORC mode |
| `rep_dispatch_<year>.csv` | (rd) dispatch on the representative days |
| `lcoh_by_period.csv` | levelised cost of heat per milestone |
| `computational_params.csv`, `tsa_summary.csv` | model size and solve time, clustering error |
| `gurobi.log`, `run.log`, `incumbent_<k>.sol` | solver log, run log, incumbents (every new incumbent is saved, so a stopped run still has its best solution) |
| `*.png` | capacity evolution, generation mix, dispatch per year and winter week, geothermal mode analysis, clustering profiles |
| with a field: `btes_timeseries.xlsx`, `btes_<year>.png`, `btes_horizon.png`, `btes_operation_<year>.png` | storage state, charge and discharge over every year of the horizon |

`run.log` is the complete log of the run, including every input override and derived capacity.

## 8. Reproducing the paper runs

The paper uses the DEA-based technology workbook `Munich_tech_data_v3.xlsx` (included) with
the hourly workbook of the case study `Munich_DH_data_spot2025.xlsx`, and solves every scenario
for two networks: the network temperatures as measured and the same network 10 K colder. With
the example workbook the same commands run, but the numbers differ from the paper.

```bash
export TECH_DATA=Munich_tech_data_v3.xlsx DH_DATA=Munich_DH_data_spot2025.xlsx
export BTES_TEMP_RULE=1 BTES_FLOW_CONTINUOUS=1 BTES_TIGHT_M=65     # HT-BTES method of the paper
export SOLVER=gurobi GRB_THREADS=20                                   # paper runs used Gurobi

# ── measured network ──────────────────────────────────────────────────────
export BTES_LIB_SUFFIX=_vfa; unset NET_TEMP_SHIFT_K
# 1. HT-BTES field-size screening on representative days, one solve per size
for N in 100 150 200 250 300 400 500; do SCENARIO=N4 BTES_DESIGN_N=$N python run_rd.py; done
SCENARIO=N4 BTES_OFF=1 python run_rd.py                              # no-field reference
# 2. full resolution for the reported table (no field pays on this network)
for SC in N1 N2 N5 N6; do SCENARIO=$SC BTES_OFF=1 python run_full.py; done

# ── 10 K colder network ───────────────────────────────────────────────────
export NET_TEMP_SHIFT_K=10 BTES_LIB_SUFFIX=_vf
for N in 100 150 200 250 300 400 500; do SCENARIO=N4 BTES_DESIGN_N=$N python run_rd.py; done
SCENARIO=N4 BTES_OFF=1 python run_rd.py
for SC in N1 N2 N5 N6; do SCENARIO=$SC BTES_OFF=1 python run_full.py; done
SCENARIO=N4 BTES_DESIGN_N=200 python run_full.py                    # best size of step 1
```

Solver settings of the paper: MIP gap 0.25 % (`GRB_MIP_GAP=0.0025` / `MIPGAP=0.0025`), time
limits 600 s for the screening, 4 h for the full-resolution runs. Runs that hit the time limit
are reported with their gap.

## 9. The revORC component

`orc_surrogate_maps.py` reads the `RevORC` sheet, assembles the hourly operating conditions of
the representative or full year (brine temperature, pressure and flow per module, ambient
temperature, network supply and return temperatures), evaluates the ORC and heat-pump
off-design surrogates in `revers_surrogates/` once, and hands the MILP hourly maps of

- net electric power per module in ORC mode (hours outside the surrogate's training domain
  are ORC-infeasible, not extrapolated), and
- COP and heat output per module in HP mode, at full load and at the part-load fractions of the
  `hp_min_load_frac` sweep.

The MILP then decides per hour whether the installed modules run as ORC or as HP (never both),
with `dead_hours` off between mode changes. Maps are cached in `data/revorc_maps_cache/` under a
hash of all inputs and model files, so the 2 GB bundle is loaded once per input set.

The surrogates are extra-trees regressors trained on a detailed thermodynamic model of a
modular reversible unit (one brine heat exchanger and one air-cooled condenser shared by both
modes). Inputs, training ranges and the Python API of the wrappers are described in
`revers_surrogates/README_reversible_ORC_HP_wrappers.txt`.

## 10. The HT-BTES component

Pipeline, all in `data/btes/`:

1. **Field simulation** — `gen/make_btes_design.py -N <boreholes> --H <depth> --B <spacing>
   --label <label>`. g-function model of a hex-packed cylindrical field (pygfunction), daily
   simulation over 20 years with a fixed charge temperature of 106 °C in summer and direct
   discharge to the network in winter, then a fit of the hourly storage coefficients per 5-year
   age block. Writes `gen/design_<label>_n<N>.json` and `gen/sim_<label>_n<N>.csv`.
   Options: `--layout disc|rect`, `--t-charge`, `--t-return`, `--t-use-min` (temperatures),
   `--m-dis` (fixed discharge flow per borehole), `--variable-flow` with `--m-max`,
   `--net-profile` / `--net-shift` (network temperature profile from an hourly workbook),
   `--demand-cap` with `--geo-ceiling` and `--demand-col`.
2. **Library build** — `gen/build_model_library.py --label <label> [--out-suffix _x]
   [--k-wall] [--flow-levels 0.10,0.20,0.30]`. Collects all designs of a label into
   `design_library<suffix>.json` and one `btes_hourly_params<suffix>_n<N>.json` per size.
3. **MILP** — `src/models/btes_params.py` loads the library; `src/models/btes_constraints.py`
   adds the storage to the full, RD and RH models. The temperature state is tracked for every
   individual year of the horizon (the warm-up years included), the daily and interval updates
   of the aggregated models are exact unrollings of the hourly chain, and discharge is gated by
   the direct-use rule: the storage wall temperature must exceed the network supply temperature
   plus the heat-exchanger approach, which depends on the discharge flow.
4. **Figures and checks** — `gen/plot_btes_simulation.py --label <label>` (20-year
   simulation per design), `gen/plot_btes_assumptions.py` (assumption table per design),
   `validation/validate_field.py` (replay of a monitored field with the same storage model;
   `validation/emmaboda/` holds the Emmaboda replay).

Shipped generator outputs: labels `t106` (measured network) and `t106s10` (10 K colder
network), from which the `_vfa` and `_vf` libraries were built with `--k-wall`. These files
contain the storage state and daily storage energy of the simulation.

## 11. Setting the model up for your own system

1. Copy `data/Munich_tech_data_v3.xlsx` and the generated `data/example_DH_data.xlsx` under
   new names and fill them with your technologies, storages, prices, milestone years and hourly
   profiles. Keep the sheet names and column headers of section 5. Start without the
   `RevORC` sheet (or set `enabled` to 0) and without HT-BTES to get a plain multi-year model
   running first.
2. Point the scripts at your files: `TECH_DATA=my_tech.xlsx DH_DATA=my_profiles.xlsx`.
3. The geothermal plant logic (modes, resource-derived capacities) is Munich-specific and is
   driven by the `Geothermal Resource` sheet. For a system without geothermal supply, set the
   geothermal row's capacity caps to 0 and remove the geothermal and HTHP rows; the mode
   constraints are then inactive.
4. For an HT-BTES of your own: run `gen/make_btes_design.py` with your ground properties (the
   constants at the top of the file), network temperatures and field sizes, then
   `gen/build_model_library.py` with a new `--out-suffix`, and select it with
   `BTES_LIB_SUFFIX`.
5. For a revORC at another site: the surrogates are valid only inside their training ranges
   (brine 95–115 °C, 12–24 bar, module 3–8 kg/s, ambient −12 … 35 °C, see the wrapper README).
   Outside these ranges the unit needs new surrogates.
6. Check `run.log` of the first run: every override and every derived capacity is logged.

## 12. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `File not found: .../Munich_DH_data.xlsx` | set `DH_DATA=example_DH_data.xlsx`, or point `DH_DATA` at your own workbook |
| `No HostID specified in license file` or `GurobiError: License expired` | set `GRB_LICENSE_FILE` to the licence file; check `gurobi_cl --license` |
| `revORC needs the 'ambient_temp' column` | the hourly workbook has no ambient temperature; add it or set `REVORC=0` |
| `FileNotFoundError: ... .joblib` in an N2/N6 run | surrogate bundles missing in `revers_surrogates/models/` |
| scikit-learn `InconsistentVersionWarning` when loading the bundles | bundles were pickled with another scikit-learn version; usually harmless, otherwise install the version named in the warning |
| `btes_hourly_params<suffix>_n<N>.json not found` | borehole count not in the library for that suffix; `ls data/btes/` shows the available sizes |
| run stops at the time limit with a large gap | HT-BTES runs with the rule are hard MILPs; use `BTES_TIGHT_M`, raise `GRB_TIME_LIMIT`, or seed with `WARM_START` |
| very slow first revORC run | the 2 GB ORC bundle is loaded and evaluated once; later runs with the same inputs hit the cache |
| results differ from the paper | check `DH_DATA` and `TECH_DATA` (section 8), `BTES_LIB_SUFFIX` and the three BTES method flags |

Stopping a run with `kill <pid>` (not `kill -9`) or Ctrl-C interrupts the solve only: the best
incumbent is reported and the run folder is written completely.
