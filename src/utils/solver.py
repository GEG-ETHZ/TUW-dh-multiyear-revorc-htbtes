"""Solver choice (env SOLVER) and translation of the solver options.

The run scripts pass Gurobi parameter names (MIPGap, TimeLimit, MIPFocus, ...).
SOLVER selects the Pyomo solver:

    SOLVER=appsi_highs   HiGHS, open source, `pip install highspy` (default)
    SOLVER=gurobi        Gurobi, needs a licence; every option is passed as before

For a solver other than Gurobi only the MIP gap and the time limit are
translated; the Gurobi tuning parameters (MIPFocus, Heuristics, Cuts,
NumericFocus, Threads, LogFile, SolFiles, ...) have no direct equivalent and are
left out.
"""

import os

from loguru import logger

DEFAULT_SOLVER = "appsi_highs"

# Gurobi parameter name -> HiGHS option name
_HIGHS_OPTIONS = {"MIPGap": "mip_rel_gap", "TimeLimit": "time_limit"}


def get_solver_name() -> str:
    """Solver selected with the SOLVER environment variable."""
    return os.environ.get("SOLVER", "").strip() or DEFAULT_SOLVER


def is_gurobi(solver_name: str) -> bool:
    return solver_name.lower().startswith("gurobi")


def translate_options(solver_name: str, options: dict) -> dict:
    """Return the options to hand to `solver_name`, given in Gurobi names."""
    if is_gurobi(solver_name):
        return dict(options)
    names = _HIGHS_OPTIONS if "highs" in solver_name.lower() else {}
    out = {names[k]: v for k, v in options.items() if k in names}
    skipped = sorted(k for k in options if k not in names)
    if skipped:
        logger.info(f"{solver_name}: Gurobi-only options not passed: {', '.join(skipped)}")
    return out
