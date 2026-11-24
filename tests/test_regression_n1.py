"""Regression tests: scenario N1 on representative days with the example data.

Each case runs run_rd.py once (HiGHS, MIP gap 0, output in a temporary folder) and
compares the result with the reference in tests/data/reference_<case>.json.

Tolerances. The solve is run to a proven optimum (gap 0), not to the 0.5 % gap of
the normal runs, so a different solution inside the gap cannot make the tests
fail. What is left are floating-point and solver tolerances (HiGHS feasibility
tolerance 1e-7):

- objective: relative 1e-6, i.e. about EUR 20 on EUR 2e7;
- capacities: absolute 1e-3 MW (run_rd.py writes them rounded to 1e-4 MW);
- representative-day dispatch: absolute 1e-4 MW plus relative 1e-6. N1 (geothermal
  must-run plus a gas boiler, no storage) has a unique optimal dispatch: the gas
  boiler covers exactly what the geothermal plant cannot, and the capacities follow
  from the peak. This was checked by solving with other HiGHS random seeds and with
  presolve off, which gave the same dispatch.
"""

import math

import pytest

from reference_cases import CASES, load_reference, run_case

OBJ_RTOL = 1e-6
CAP_ATOL = 1e-3
DISPATCH_ATOL = 1e-4
DISPATCH_RTOL = 1e-6


@pytest.fixture(scope="module", params=sorted(CASES))
def case(request, tmp_path_factory):
    name = request.param
    out_root = tmp_path_factory.mktemp(name)
    return name, run_case(name, out_root), load_reference(name)


def test_objective(case):
    name, got, ref = case
    assert math.isclose(got["objective_EUR"], ref["objective_EUR"], rel_tol=OBJ_RTOL), (
        f"{name}: objective {got['objective_EUR']:,.2f} != reference {ref['objective_EUR']:,.2f}")


def test_capacity(case):
    name, got, ref = case
    assert sorted(got["capacity"]) == sorted(ref["capacity"]), f"{name}: technologies differ"
    for tech, by_year in ref["capacity"].items():
        assert sorted(got["capacity"][tech]) == sorted(by_year), f"{name}: {tech} years differ"
        for year, ref_vals in by_year.items():
            for key, ref_v in ref_vals.items():
                got_v = got["capacity"][tech][year][key]
                assert abs(got_v - ref_v) <= CAP_ATOL, (
                    f"{name}: {tech} {year} {key} = {got_v} != reference {ref_v}")


def test_rep_dispatch(case):
    name, got, ref = case
    assert sorted(got["rep_dispatch"]) == sorted(ref["rep_dispatch"]), f"{name}: years differ"
    for year, ref_d in ref["rep_dispatch"].items():
        got_d = got["rep_dispatch"][year]
        assert got_d["columns"] == ref_d["columns"], f"{name} {year}: columns differ"
        assert len(got_d["rows"]) == len(ref_d["rows"]), f"{name} {year}: row count differs"
        for i, (g_row, r_row) in enumerate(zip(got_d["rows"], ref_d["rows"])):
            for col, g, r in zip(ref_d["columns"], g_row, r_row):
                assert math.isclose(g, r, rel_tol=DISPATCH_RTOL, abs_tol=DISPATCH_ATOL), (
                    f"{name} {year} row {i} {col}: {g} != reference {r}")
