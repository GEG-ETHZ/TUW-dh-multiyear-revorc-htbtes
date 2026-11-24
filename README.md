[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11--3.13-blue)](pyproject.toml)
[![Solver](https://img.shields.io/badge/solver-HiGHS%20%7C%20Gurobi-red)](#choosing-the-solver)

---
# Multi-year district-heating optimisation with revORC and HT-BTES

*A multi-year capacity-expansion and dispatch model for geothermal district heating with a
reversible ORC / heat pump and a high-temperature borehole thermal energy storage* |
[GitHub](https://github.com/nirav0906/dh-multiyear-revorc-htbtes)

---

## Content

* [About](#about)
* [How to install](#how-to-install)
* [Quick start](#quick-start)
* [Documentation](#documentation)
* [Repository layout](#repository-layout)
* [Surrogate data](#surrogate-data)
* [Contributing](#contributing)
* [What's new](#whats-new)
* [Citing](#citing)
* [License](#license)
* [Remove / uninstall](#remove--uninstall)
* [Attribution](#attribution)
---

## About

This repository holds the code and input data behind the Munich case study of a multi-year
district-heating (DH) planning model. The model is a Mixed-Integer Linear Programme written in
[Pyomo](https://pyomo.readthedocs.io/) and solved with [HiGHS](https://highs.dev/) (default) or
[Gurobi](https://www.gurobi.com/). It
decides, jointly, which capacities to build at each 5-year investment milestone of a 2025–2045
horizon and how to dispatch the plant hour by hour, minimising the total discounted system cost
(investment, fixed and variable O&M, fuel, electricity, CO2, minus electricity revenue).

The reference system is a geothermal doublet with direct heat exchange and a gas boiler. Two
flexibility technologies are added to it and are represented inside the MILP through surrogate
models:

- **revORC** — a modular reversible ORC / heat-pump unit on the geothermal brine. In summer it
  runs as an ORC and sells electricity made from surplus brine heat; in winter it runs as a heat
  pump and lifts the low-temperature tail of the brine into the network. Hourly performance in
  both modes comes from machine-learned (extra-trees) surrogates of a detailed component model.
- **HT-BTES** — a high-temperature borehole thermal energy storage, charged at wellhead
  temperature in summer and discharged directly into the network in winter. The field is
  simulated with g-functions ([pygfunction](https://pygfunction.readthedocs.io/)) over 20 years
  and reduced to an age-dependent linear storage model with a temperature state and a
  direct-use discharge rule.

Three temporal formulations share one model core and one set of inputs: a full 8,760-hour
model, a representative-day model (Kotzur storage superposition, clustering with
[tsam](https://tsam.readthedocs.io/)) and a representative-hour model. All inputs are two Excel
workbooks, so the model can be moved to another DH system without touching the code. Results
are written as an Excel workbook, CSV files and PNG figures per run.

## How to install

1. Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html).

2. Download this [repository](https://github.com/nirav0906/dh-multiyear-revorc-htbtes/archive/main.zip)
   (or `git clone` it) and unzip it to the location of your choice.

3. Open a terminal (conda prompt) in that folder and run

   a. `conda env create -f environment.yml` (this also installs the model code itself as the
      Python package `dh_multiyear`, in editable mode, so changes to `src/` take effect directly)

   b. `conda activate dh-multiyear`

   c. Only if you want to solve with Gurobi (see [Choosing the solver](#choosing-the-solver)):
      `pip install -e ".[gurobi]"`, then
      `python -c "import gurobipy; print(gurobipy.gurobi.version())"` to check that Gurobi
      finds its licence. If it reports `No HostID specified in license file`, point it at the
      file: `export GRB_LICENSE_FILE=/path/to/gurobi.lic`.

4. The hourly example workbook `data/example_DH_data.xlsx` (included) provides demand,
   network temperatures, ambient temperature and prices for a first run; it is regenerated with
   `python data/make_example_dh_data.py`.

5. **Only for the revORC scenarios:** place the surrogate bundles in
   `revers_surrogates/models/` (see [Surrogate data](#surrogate-data)). Every other scenario
   runs without them.

Alternative without conda: `pip install -r requirements.txt` into a Python 3.11–3.13
environment, run in the repository folder. It installs the package and its dependencies, which
are listed in `pyproject.toml`.

To use the model from other Python code, install the repository into that environment
(`pip install -e path/to/dh-multiyear-revorc-htbtes`) and import from the package, e.g.
`from dh_multiyear.models.rd_model import DistrictHeatingModel`. Importing the package or the
run scripts has no side effects; the input data stay in `data/` of the repository.

### Choosing the solver

The solver is chosen with the environment variable `SOLVER`. The default, `appsi_highs`, is the
open-source [HiGHS](https://highs.dev/) solver (installed with `highspy`, no licence needed);
it solves the small scenarios such as N1 on representative days in seconds. `SOLVER=gurobi`
uses Gurobi with the same settings as before (install `gurobipy` and a licence); it is
recommended for the large HT-BTES and full 8,760 h runs. With HiGHS only the MIP gap and the time
limit (`GRB_MIP_GAP`/`MIPGAP`, `GRB_TIME_LIMIT`/`TIMELIMIT_S`) are passed on; the Gurobi
tuning parameters, the `gurobi.log` and the incumbent files exist only with Gurobi.

## Quick start

Activate the environment and run the base scenario on representative days with the example data:

```bash
conda activate dh-multiyear
cd path/to/dh-multiyear-revorc-htbtes
export DH_DATA=example_DH_data.xlsx
SCENARIO=N1 python run_rd.py
```

The run folder appears under `results/rd/MILP_<timestamp>_N1/` with the results workbook,
hourly dispatch CSVs, the solver log (Gurobi only) and figures. Every setting is an environment variable, so a
scenario is one command line:

```bash
SCENARIO=N2 python run_rd.py                                   # + revORC (needs the bundles)
SCENARIO=N4 BTES_DESIGN_N=150 BTES_LIB_SUFFIX=_vfa \
  BTES_TEMP_RULE=1 BTES_FLOW_CONTINUOUS=1 BTES_TIGHT_M=65 python run_rd.py   # + HT-BTES, 150 boreholes
SCENARIO=N5 python run_rd.py                                   # + pit storage
SCENARIO=N6 python run_full.py                                 # all options, full 8,760 h
```

The [Getting Started guide](docs/getting_started.md) explains the scenarios, every input sheet
and column, all environment variables, the output files, how the paper runs are reproduced and
how to set the model up for your own DH system.

## Documentation

- [Getting Started guide](docs/getting_started.md) — installation, inputs, running, outputs,
  adapting the model to another case, troubleshooting.
- [Geothermal plant operating modes](docs/SAPHEA_modes_and_integration.md) — how the direct-use,
  heat-pump and combined modes of the geothermal plant enter the MILP.
- [revORC surrogate wrappers](revers_surrogates/README_reversible_ORC_HP_wrappers.txt) —
  inputs, training ranges and usage of the ORC and heat-pump surrogates.
- [HT-BTES validation](data/btes/validation/emmaboda/README.txt) — replay of the monitored
  Emmaboda storage with the storage model used here.

## Repository layout

```
run_rd.py                     representative-days run (design screening, field-size enumeration)
run_full.py                   full 8,760-h run (reported results)
run_rh.py                     representative-hours run
full_reporting.py, rh_reporting.py      reporting of run_full / run_rh through the run_rd figures
create_munich_data.py         builds an hourly workbook from measured network data (bring your own measurements)
run_nice.sh                   runs any command at the lowest CPU priority

pyproject.toml                package definition (name dh-multiyear, dependencies, extras)
src/dh_multiyear/             the model code, importable as the package dh_multiyear:
  data_loader.py              reads the two Excel workbooks
  orc_surrogate_maps.py       evaluates the revORC surrogates offline -> hourly ORC/HP maps for the MILP
  colors.py                   technology colours of the figures
  models/full_model.py        hourly MILP (class DistrictHeatingModelFull)
  models/rd_model.py          representative-day MILP (class DistrictHeatingModel)
  models/rh_model.py          representative-hour MILP (class DistrictHeatingModelRH)
  models/saphea_revorc_rh.py  geothermal / revORC blocks for the RH model
  models/btes_params.py       HT-BTES parameter tables (age blocks, design library)
  models/btes_constraints.py  HT-BTES variables and constraints for all three models
  preprocessing/time_aggregation.py  tsam clustering, reduced storage index
  utils/economic_calculations.py     discounting, annuities, salvage, escalation
  utils/heat_pump_cop.py             Carnot / Hotmaps COP models
  utils/solver.py                    solver choice (SOLVER) and option translation
tests/                        regression tests (see tests/README.md)

revers_surrogates/            revORC surrogate wrappers (design ANN, ORC off-design, HP off-design)
revers_surrogates/models/     *.joblib surrogate bundles (available on request)

data/Munich_tech_data*.xlsx   technologies, storage, prices, geothermal resource, RevORC sheet
data/example_DH_data.xlsx     synthetic hourly demand, temperatures, prices (+ make_example_dh_data.py)
data/spot_prices/             DE-LU day-ahead prices 2022-2025 (+ make_spot_prices.py)
data/btes/                    HT-BTES surrogate libraries read by the model
data/btes/gen/                field simulation and library builder (pygfunction)
data/btes/validation/         replay of the monitored Emmaboda storage

tools/make_btes_run_operation.py   storage-operation figure written after each run with a field
tools/remove_outputs.sh            deletes results, caches and byte-code
docs/                         Getting Started guide, geothermal operating modes
results/                      run output (git-ignored)
```

## Surrogate data

The trained revORC surrogate bundles (`revers_surrogates/models/*.joblib`, about 2.4 GB) are
not included because of their size. They can be requested from the authors; the expected file
names and folder layout are given in `revers_surrogates/README_reversible_ORC_HP_wrappers.txt`.
All scenarios without the revORC run without them.

## Contributing

To contribute changes:

1. Fork the project on GitHub
2. Create a feature branch to work on in your fork (`git checkout -b new-feature`)
3. Add your name to the `AUTHORS.md` file
4. Commit your changes to the feature branch
5. Push the branch to GitHub (`git push origin new-feature`)
6. On GitHub, create a new pull request from the feature branch

See the [contribution guidelines](CONTRIBUTING.md) for more information, or open an issue to
ask a question or report a problem.

## What's new

See changes made in recent versions in the [changelog](changelog.md).

## Citing

If you use this model for academic work, please cite the paper (reference to be added on
publication) and the software via [`CITATION.cff`](CITATION.cff):

Patel, N. Multi-year district-heating optimisation with a reversible ORC/heat pump and a
high-temperature borehole storage. *Software, GitHub.*

## License

Copyright 2026 Nirav Patel

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file
except in compliance with the License. You may obtain a copy of the License at

<http://www.apache.org/licenses/LICENSE-2.0>

Unless required by applicable law or agreed to in writing, software distributed under the
License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
either express or implied. See the License for the specific language governing permissions and
limitations under the License.

The revORC surrogate models and their wrapper code were provided by project partners; the
`.joblib` bundles are not part of this repository. Third-party input data keep their own
licences: the DE-LU day-ahead prices are CC BY 4.0 (Bundesnetzagentur / SMARD via
energy-charts.info).

## Remove / uninstall

Delete the folder and remove the conda environment:

```bash
conda remove --name dh-multiyear --all
```

To delete only run outputs and caches while keeping code and inputs: `./tools/remove_outputs.sh`.

## Attribution

The layout of this document follows the [hotmapsDispatch](https://github.com/tuw-eeg/hotmapsDispatch)
README, which in turn is based on the [calliope-project](https://github.com/calliope-project/calliope)
README.
