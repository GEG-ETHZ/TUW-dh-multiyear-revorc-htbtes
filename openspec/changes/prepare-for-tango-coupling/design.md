# Design

## Context

What we found reading the code at `efc3bea` and running it on the synthetic example data (`example_DH_data.xlsx`, `Munich_tech_data.xlsx`) with Python 3.12 and HiGHS (`appsi_highs`):

| Case | Model size | HiGHS solve | Objective |
|---|---|---|---|
| N1, 4 representative days | 4,400 variables | 0.6 s, optimal | EUR 19,403,870 |
| N1, 24 representative days | 27,455 variables | 1.2 s, optimal | EUR 22,305,869 |
| N4 (HT-BTES, 150 boreholes), 4 days | 13,645 variables | about 20 min, gap 0.43% | EUR 19,767,077 |
| N2 (revORC) | | does not run | the `.joblib` bundles are not in the repository |

- Entry points `run_rd.py`, `run_full.py`, `run_rh.py`, each a `main()` configured by environment variables; `run_rd.py` is the core the others import from.
- `src/models/rd_model.py` `DistrictHeatingModel` (`build`, `solve`, `get_results`, `compute_lcoh`); the geothermal (SAPHEA) and revORC blocks are added after `build()` by functions in `run_rd.py`.
- Brine inputs come from the `Geothermal Resource` sheet as single values (`data_loader.py` `load_geothermal_resource`); hourly supply, return and ambient temperatures are one series for all years.
- Results are written as files under `results/rd/`; `get_results()` returns only the objective, capacities and annual generation.

## Decisions

### D1: Small pull requests, one concern each

Each mandatory, proposed change is a separate pull request, stacked in this order so each can be reviewed alone: solver and dependencies; tests and CI; packaging and import side effects; reporting fixes. The first three do not change any result, and the tests show it. The reporting fixes change reported values only, never the objective, and say which values change.

### D2: HiGHS by default, Gurobi still available

A free default solver lets anyone, including CI and TANGO users, run the model. Gurobi stays selectable for the large cases where it is faster; its options are passed only to Gurobi. We have not compared HiGHS with Gurobi results, because we have no Gurobi licence; HiGHS solved the small cases to optimality and the HT-BTES case to a 0.43% gap.

### D3: Tests freeze today's results on small cases

The tests run N1 at 4 and 24 representative days with HiGHS and compare the objective, capacities and representative-day dispatch with stored values. They take seconds. The HT-BTES case takes about 20 minutes with HiGHS and is not part of CI.

### D4: One revORC model for both of us

TANGO couples the revORC through NTUA's RevORCHP-SM package. If this model keeps its own copy of older scripts, the two models disagree about the same plant. We propose depending on NTUA's released package instead, once its packaged release exists; until then this is listed, not drafted.

### D5: A library interface comes after agreement

A clean interface (build a case, solve, read results as data) is the largest change and shapes how TANGO calls the model, so we draft it only after we agree on it and on the requested items 8 to 12.

## Who does what

To agree. We have drafted items 1 to 4.

| Item | Kind | Proposed | Agreed |
|---|---|---|---|
| 1. Solver setting, tsam pin | Mandatory, proposed | TANGO team (drafted) | |
| 2. Regression tests and CI | Mandatory, proposed | TANGO team (drafted) | |
| 3. Installable package, no import side effects | Mandatory, proposed | TANGO team (drafted) | |
| 4. Reporting fixes | Mandatory, proposed | TANGO team (drafted) | |
| 5. Library interface | Mandatory, proposed | TANGO team, after agreement | |
| 6. NTUA's revORC package | Mandatory, proposed | open | |
| 7. Tagged releases | Mandatory, proposed | author | |
| 8–13. Requested items | Mandatory, requested | author | |

## Reporting bugs found

- The cost breakdown reads a technology's `carrier`, but the data loader stores `energy_carrier`; the Carrier column is empty and electricity-driven technologies are costed at the flat fuel price instead of the hourly electricity price in `Fuel_cost_EUR`. The objective is not affected.
- The Excel dispatch sheets use the base-year demand for every year (for 2045: 79,815 MWh in the sheet against 86,201 MWh in `dispatch_hourly_2045.csv`).
- The user guide says `dispatch_hourly_<yr>.csv` contains reinjected heat and the revORC mode; it does not.
- Not fixed, only noted: the hourly dispatch rebuilt from representative days differs from the hourly demand by up to 9 MW; `rep_dispatch_<yr>.csv` balances.

## Open Questions

1. How should the reinjection temperature be computed per hour and mode (direct use, ORC in parallel, heat pump in series, BTES charging, reinjected heat)? Is `T_inject_min` a hard floor or a target?
2. What is the "small change" for reading network temperatures per milestone year or per hour from TANGO? Per-year columns, like `corrected_demand_<yr>`?
3. Should brine flow, production temperature, minimum injection temperature and pressure vary per milestone year (reservoir decline) or per hour? The doublet's capacity is derived once from them today.
4. Will you switch to NTUA's RevORCHP-SM 1.0.2 package? Does it cover the series-trained heat-pump bundle (`…hp55_75_out35_45_sourcebalanced.joblib`, "2026-07-30") your code uses?
5. Which formulation should TANGO use: representative days (`run_rd.py`) or the full year (`run_full.py`)? Should representative hours (experimental) be left out?
6. How should TANGO read hourly dispatch from the representative-day model, given that the rebuilt hourly dispatch does not balance demand exactly?
7. Is HiGHS acceptable as the default, with Gurobi optional? Does anything rely on Gurobi-specific behaviour (for example `NumericFocus` for the BTES block)?
8. Is the HT-BTES library valid if TANGO supplies a different production temperature or network temperatures, or must it be regenerated?
9. Would you like to keep the `openspec/` folder (these documents) in the repository, or keep this discussion in a pull request or an issue?
