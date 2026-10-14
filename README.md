# Multi-year district-heating optimisation with a reversible ORC/heat pump and a high-temperature borehole storage

Code and input data for the Munich case study: a multi-year capacity-expansion and dispatch
MILP (Pyomo + Gurobi) for a geothermal district-heating (DH) system, extended with two
flexibility technologies that are represented through surrogate models inside the MILP:

- **revORC** — a modular reversible ORC / heat-pump unit on the geothermal brine. In summer
  it runs as an ORC and sells electricity from surplus brine heat; in winter it runs as a
  heat pump and lifts the brine's low-temperature tail into the network. Hourly performance
  comes from machine-learned (extra-trees) surrogates of a detailed component model.
- **HT-BTES** — a high-temperature borehole thermal energy storage charged at wellhead
  temperature in summer and discharged directly into the network in winter. The field is
  simulated with g-functions (pygfunction) over 20 years and reduced to an age-dependent
  linear storage model with a temperature state and a direct-use discharge rule.

The planning horizon runs 2025–2045 in 5-year investment milestones. The objective is the
total discounted system cost over the horizon (investment, fixed and variable O&M, fuel,
electricity, CO2, minus electricity revenue). Three temporal formulations share one model
core: a full 8,760-hour model, a representative-day model (Kotzur storage superposition,
`tsam` clustering), and a representative-hour model.

## Repository layout

```
run_rd.py                     representative-days run (design screening, field-size enumeration)
run_full.py                   full 8,760-h run (reported results)
run_rh.py                     representative-hours run
data_loader.py                reads the two Munich Excel workbooks
orc_surrogate_maps.py         evaluates the revORC surrogates offline -> hourly ORC/HP maps for the MILP
full_reporting.py, rh_reporting.py, colors.py      reporting and figures
create_munich_data.py         builds the Munich workbooks from the measured network data
run_nice.sh                   runs any command at the lowest CPU priority

src/models/full_model.py      hourly MILP (class DistrictHeatingModelFull)
src/models/rd_model.py        representative-day MILP (class DistrictHeatingModel)
src/models/rh_model.py        representative-hour MILP (class DistrictHeatingModelRH)
src/models/saphea_revorc_rh.py  geothermal / revORC blocks for the RH model
src/models/btes_params.py     HT-BTES parameter tables (age blocks, design library)
src/models/btes_constraints.py  HT-BTES variables and constraints for all three models
src/preprocessing/time_aggregation.py   tsam clustering, reduced storage index
src/utils/economic_calculations.py      discounting, annuities, salvage, escalation
src/utils/heat_pump_cop.py              Carnot / Hotmaps COP models

revers_surrogates/            revORC surrogate wrappers (design ANN, ORC off-design, HP off-design)
revers_surrogates/models/     *.joblib bundles  -- NOT in git, see "Large files"

data/Munich_tech_data*.xlsx   technologies, storage, prices, geothermal resource, RevORC sheet
data/Munich_DH_data*.xlsx     hourly demand, network temperatures, ambient temperature, prices
data/DhnData.csv              measured network data the workbooks are generated from
data/spot_prices/             DE-LU day-ahead prices 2022-2025 (+ make_spot_prices.py)
data/btes/                    HT-BTES surrogate libraries read by the model
data/btes/gen/                field simulation and library builder (pygfunction)
data/btes/validation/         replay of the monitored Emmaboda storage

tools/make_btes_run_operation.py   storage-operation figure written after each run with a field
docs/SAPHEA_modes_and_integration.md  geothermal plant operating modes and how they enter the MILP
results/                      run output (git-ignored)
```

## Setup

```bash
conda env create -f environment.yml && conda activate dh-multiyear
# or: pip install -r requirements.txt
```

A **Gurobi licence** is required (free academic licences via `grbgetkey`). If Gurobi reports
`No HostID specified in license file`, point it at the file explicitly:

```bash
export GRB_LICENSE_FILE=$HOME/gurobi.lic/gurobi.lic
```

The revORC scenarios also need the surrogate bundles in `revers_surrogates/models/`
(see "Large files"). Everything else runs from the repository as is.

## Running

```bash
python run_rd.py        # representative days   -> results/rd/MILP_<timestamp>_<SCENARIO>/
python run_full.py      # full 8,760 h          -> results/full/MILP_<timestamp>_<SCENARIO>/
python run_rh.py        # representative hours  -> results/rh/
```

All settings are environment variables, so a scenario is one command line. Long runs should
be started with `./run_nice.sh python run_full.py` on a shared machine.

### Scenarios

Every scenario shares the base system of the reference plant: a must-run geothermal doublet
with direct heat exchange, plus a gas boiler. One flexibility option is added at a time:

| `SCENARIO` | Option added | Paper label |
|---|---|---|
| `N1` | none (base system) | S1 |
| `N2` | revORC | S2 |
| `N3` | high-temperature heat pump on the brine tail (not in the paper) | – |
| `N4` | HT-BTES, field size via `BTES_DESIGN_N` (or `BTES_OFF=1` for no field) | S3 |
| `N5` | pit thermal energy storage (PTES) | S4 |
| `N6` | all options together | S5 |

`S1`–`S5` are an older scenario suite kept for backward compatibility.

### Reproducing the paper runs

The paper uses the DEA-based technology workbook and the 2025 spot-price workbook, not the
defaults, and solves each scenario for two networks: the network temperatures as measured,
and the same network operated 10 K colder.

```bash
export TECH_DATA=Munich_tech_data_v3.xlsx DH_DATA=Munich_DH_data_spot2025.xlsx
export BTES_TEMP_RULE=1 BTES_FLOW_CONTINUOUS=1 BTES_TIGHT_M=65     # HT-BTES method of the paper

# measured network, HT-BTES library with wall feedthrough
export BTES_LIB_SUFFIX=_vfa; unset NET_TEMP_SHIFT_K
# 10 K colder network and its refitted library
# export NET_TEMP_SHIFT_K=10 BTES_LIB_SUFFIX=_vf

# 1. field-size screening with representative days (one solve per size)
for N in 100 150 200 250 300 400 500; do SCENARIO=N4 BTES_DESIGN_N=$N python run_rd.py; done
SCENARIO=N4 BTES_OFF=1 python run_rd.py                              # no-field reference

# 2. full-resolution runs of the five scenarios with the field size chosen in step 1
for SC in N1 N2 N5; do SCENARIO=$SC BTES_OFF=1 python run_full.py; done
SCENARIO=N4 BTES_DESIGN_N=200 python run_full.py                    # colder network: 200 boreholes
SCENARIO=N6 BTES_OFF=1 python run_full.py
```

Solver controls differ between the two scripts: `run_rd.py` reads `GRB_THREADS`,
`GRB_MIP_GAP`, `GRB_TIME_LIMIT`; `run_full.py` reads `GRB_THREADS`, `MIPGAP`, `TIMELIMIT_S`
and `RESULTS_SUBDIR`.

### Main environment flags

| Variable | Default | Meaning |
|---|---|---|
| `SCENARIO` | `S3` (rd) / `S1` (full) | scenario, see table above |
| `TECH_DATA`, `DH_DATA` | `Munich_tech_data.xlsx`, `Munich_DH_data.xlsx` | workbooks in `data/` |
| `NET_TEMP_SHIFT_K` | 0 | shift the measured supply and return temperatures by −K (network sensitivity) |
| `GRB_THREADS` | 40 | solver threads |
| `GRB_MIP_GAP` / `MIPGAP` | 0.005 / 0.0025 | relative MIP gap (rd / full) |
| `GRB_TIME_LIMIT` / `TIMELIMIT_S` | 1800 / 172800 | time limit in seconds (rd / full) |
| `N_TYPICAL_DAYS` | `extra` sheet value | representative days per milestone (`run_rd.py`) |
| `TSA_MODE` | `per_period` | cluster each milestone separately or `joint` |
| `REVORC` | from the `RevORC` sheet | `0` disables the revORC |
| `REVORC_EXCLUSIVE` | 1 | revORC and HTHP mutually exclusive investments |
| `GRID_CAP_FEE` | sheet value | capacity-based grid charge in EUR/kW_el/a (`0` disables) |
| `BTES_DESIGN_N` | – | fix the field to this borehole count (50 … 500 from the library) |
| `BTES_OFF` | – | `1`: no field in an N4/N6 run |
| `BTES_LIB_SUFFIX` | `` | library variant: `` 106 °C fixed flow, `_s10` colder network, `_vf` / `_vfa` with wall feedthrough (colder / measured network) |
| `BTES_TEMP_RULE`, `BTES_FLOW_CONTINUOUS`, `BTES_TIGHT_M` | off | direct-use discharge rule, continuous discharge flow, big-M for the rule |
| `BTES_FLOW_LEVELS` | off | discrete discharge-flow levels (0.10/0.20/0.30 kg/s per borehole), RD only |
| `KEEP_BUFFER` | off | `1` restores the short-term buffer tank for a diagnostic run |
| `CO2_MULT`, `ELEC_MULT`, `GAS_PRICE`, `CO2_PRICE`, `ELEC_PRICE`, `SELL_PRICE` | – | price sensitivities |
| `T_INJECT_MIN` | sheet value | minimum reinjection temperature in °C |
| `RESULTS_SUBDIR`, `RUN_TAG` | `full`, `` | output folder name and tag |

## The two components

### revORC

`orc_surrogate_maps.py` reads the `RevORC` sheet of the technology workbook, builds the
hourly operating conditions of the representative or full year (brine temperature and flow,
ambient temperature, network supply and return temperature), evaluates the ORC and HP
off-design surrogates in `revers_surrogates/` once, and hands the MILP hourly maps of
electric power per module (ORC mode) and COP / heat output per module (HP mode). The two
modes are mutually exclusive per hour. Maps are cached in `data/revorc_maps_cache/` under a
hash of all inputs and model files.

Module cost is entered per kW_el and converted to the model's MW_th capacity unit with the
surrogate's rated electrical output per module (`module_orc_MW_el` in the sheet). The
surrogates and their wrappers were provided by project partners; see
`revers_surrogates/README_reversible_ORC_HP_wrappers.txt` for inputs, training ranges and usage.

### HT-BTES

Pipeline (all in `data/btes/`):

1. `gen/make_btes_design.py` — g-function field model for a hex-packed cylindrical field
   (`-N` boreholes, depth `--H`, spacing `--B`), 20-year daily simulation with a fixed
   charge temperature of 106 °C and direct discharge to the network, then an ARX fit per
   5-year age block. Writes `design_<label>_n<N>.json` and `sim_<label>_n<N>.csv`.
2. `gen/build_model_library.py --label <label> [--out-suffix _x] [--k-wall]` — collects the
   designs into `design_library<suffix>.json` (per-design, per-age-block initial temperature
   and power limits) and one `btes_hourly_params<suffix>_n<N>.json` per size (coefficients,
   bounds, costs, optional wall feedthrough and discharge-flow levels).
3. `src/models/btes_params.py` loads these; `src/models/btes_constraints.py` adds the
   storage to the full, RD and RH models. The temperature state is tracked year by year
   across the warm-up, the daily and interval updates are exact unrollings of the hourly
   chain, and discharge is gated by the direct-use rule (storage wall temperature above the
   network supply temperature plus the heat-exchanger approach).

Shipped generator outputs: labels `t106` (measured network) and `t106s10` (10 K colder
network). `gen/plot_btes_simulation.py` and `gen/plot_btes_assumptions.py` draw the
simulation and assumption figures per design. `validation/validate_field.py` replays a
monitored field with the same storage model; `validation/emmaboda/` holds the replay of the
Emmaboda storage against published measurements.

## Outputs

Each run folder contains the results workbook (capacities, generation, costs, CO2 per
milestone), hourly dispatch CSVs per milestone year, a geothermal operation summary, the
Gurobi log, and plots. Runs with a field also get storage time series and the operation
figure from `tools/make_btes_run_operation.py`.

## Large files

| Path | Size | How to obtain |
|---|---|---|
| `revers_surrogates/models/*.joblib` | 2.4 GB (largest file 2.0 GB) | not in git (above GitHub's file limit); request from the authors or download from the data archive linked in the paper |
| `results/` | tens of GB | regenerate with the commands above |

The folder layout expected for the bundles is given in
`revers_surrogates/README_reversible_ORC_HP_wrappers.txt`. The wrapper code is included so
the model code can be read without the bundles.

## Data provenance

- `data/DhnData.csv` — measured heat demand and network temperatures of the reference
  plant (April 2022 – March 2023), re-sorted to a calendar year by `create_munich_data.py`.
- Technology and cost assumptions — Danish Energy Agency technology catalogue and the
  reference-plant publication, cited in the paper; sources are noted per row in the workbooks.
- `data/spot_prices/` — Bundesnetzagentur / SMARD day-ahead prices via api.energy-charts.info,
  CC BY 4.0. `make_spot_prices.py` builds the hourly price columns of `Munich_DH_data_spot2025.xlsx`.
- `data/btes/validation/emmaboda/` — field geometry, loads and temperatures digitised from
  the published Emmaboda monitoring reports named in its README.

## Citation

A citation will be added when the paper is published.
