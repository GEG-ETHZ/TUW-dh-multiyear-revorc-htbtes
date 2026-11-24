# Tests

Regression tests that check that the model still gives the same results.

```bash
pip install -r requirements.txt -r requirements-dev.txt
python -m pytest tests
```

The tests need no Gurobi licence and take about 20 s. They also run on GitHub
(`.github/workflows/tests.yml`) for Python 3.11, 3.12 and 3.13.

## What is tested

`test_regression_n1.py` runs `run_rd.py` twice, scenario N1 (geothermal + gas boiler)
on the synthetic example data `data/example_DH_data.xlsx` with 4 and with 24
representative days. It solves with HiGHS to a MIP gap of 0 and writes the run folder
into a temporary directory (`RESULTS_ROOT`), not into `results/`. It then compares
with the stored values in `data/reference_<case>.json`:

- the objective (total discounted system cost), relative tolerance 1e-6;
- capacity added and total per technology and milestone, 1e-3 MW;
- the whole content of every `rep_dispatch_<year>.csv`, 1e-4 MW.

The docstring of `test_regression_n1.py` explains the tolerances.

## When a test fails

A failure means the results changed. If that is not intended, the change you made
has a side effect on the model. If it is intended, regenerate the reference values
and commit them together with the change:

```bash
python tests/generate_reference.py
```

Read the docstring of `generate_reference.py` first: regenerating is only right
when you know why the numbers changed.
