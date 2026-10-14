# Contributing

Thank you for considering a contribution. The model is research code; the aim of this repository
is that the published results can be reproduced and that the model can be adapted to other
district-heating systems.

## Reporting problems

Open an issue on GitHub with:

- the command line you ran (including the environment variables you set),
- the `run.log` of the run folder under `results/`, or the traceback,
- your Python, Pyomo and Gurobi versions.

## Proposing changes

1. Fork the project on GitHub.
2. Create a feature branch in your fork (`git checkout -b my-change`).
3. Make the change. Keep the three temporal formulations (`full_model.py`, `rd_model.py`,
   `rh_model.py`) consistent: a change to the shared constraints has to be made in each.
4. Run a short smoke test before you push, e.g.

       export DH_DATA=example_DH_data.xlsx
       SCENARIO=N1 GRB_TIME_LIMIT=60 python run_rd.py
       SCENARIO=N4 BTES_DESIGN_N=150 BTES_LIB_SUFFIX=_vfa BTES_TEMP_RULE=1 BTES_FLOW_CONTINUOUS=1 BTES_TIGHT_M=65 GRB_TIME_LIMIT=60 python run_rd.py

   and check that the run folder is written completely (`results_rd.xlsx`, dispatch CSVs, plots).
5. Add yourself to `AUTHORS.md`, describe the change in `changelog.md`.
6. Push the branch and open a pull request against `main`.

## Style

- Python 3.10+, PEP 8, no new hard-coded paths: everything case-specific comes from the Excel
  workbooks in `data/` or from environment variables.
- Log through `loguru`, not `print`.
- Do not commit solver output, results, the surrogate `.joblib` bundles or any measured network
  data (see `.gitignore`).
