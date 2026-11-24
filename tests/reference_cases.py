"""Shared code of the regression tests and tests/generate_reference.py.

A case is one `run_rd.py` run on the synthetic example data, solved with HiGHS
to a MIP gap of 0, written into a temporary folder (RESULTS_ROOT). From the run
folder we keep the numbers the tests compare:

- objective     total discounted system cost (results_rd.xlsx, sheet summary)
- capacity      added and total MW per technology and milestone
                (results_rd.xlsx, sheet cost_co2_breakdown)
- rep_dispatch  the full content of rep_dispatch_<year>.csv per milestone
"""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
DATA_DIR = Path(__file__).resolve().parent / "data"

# Settings shared by every case. GRB_MIP_GAP=0 makes HiGHS prove optimality, so the
# reference is the optimum and not one of many solutions inside a gap. The small
# N1 cases still solve in about a second this way.
COMMON_ENV = {
    "DH_DATA": "example_DH_data.xlsx",
    "TECH_DATA": "Munich_tech_data.xlsx",
    "SOLVER": "appsi_highs",
    "GRB_MIP_GAP": "0",
    "GRB_TIME_LIMIT": "300",
    "MPLBACKEND": "Agg",
}

CASES = {
    "n1_4days": {"SCENARIO": "N1", "N_TYPICAL_DAYS": "4"},
    "n1_24days": {"SCENARIO": "N1", "N_TYPICAL_DAYS": "24"},
}

# Model settings a developer may have exported in the shell; they are removed so
# that only the settings above define the run.
_MODEL_ENV_PREFIXES = (
    "SCENARIO", "TECH_DATA", "DH_DATA", "SOLVER", "GRB_", "MIPGAP", "TIMELIMIT_S",
    "N_TYPICAL_DAYS", "N_REP_HOURS", "RH_", "TSA_", "NET_TEMP_SHIFT_K", "BTES_",
    "REVORC", "GRID_CAP_FEE", "FORCE_", "CO2_", "ELEC_", "GAS_PRICE", "SELL_PRICE",
    "T_INJECT_MIN", "GEO_", "HTHP_", "KEEP_BUFFER", "WARM_START", "RESULTS_",
    "RUN_TAG", "NO_REPORT",
)


def run_case(name: str, out_root: Path) -> dict:
    """Run case `name` with run_rd.py into `out_root` and return its numbers."""
    env = {k: v for k, v in os.environ.items() if not k.startswith(_MODEL_ENV_PREFIXES)}
    env.update(COMMON_ENV)
    env.update(CASES[name])
    env["RESULTS_ROOT"] = str(out_root)
    proc = subprocess.run([sys.executable, str(REPO / "run_rd.py")], cwd=REPO, env=env,
                          capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"run_rd.py failed for {name}:\n{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}")
    run_dirs = sorted((out_root / "results" / "rd").glob("MILP_*"))
    if len(run_dirs) != 1:
        raise RuntimeError(f"expected one run folder in {out_root}, found {run_dirs}")
    return extract(run_dirs[0])


def extract(run_dir: Path) -> dict:
    """Read the compared numbers from a run folder."""
    xlsx = run_dir / "results_rd.xlsx"
    summary = pd.read_excel(xlsx, sheet_name="summary")
    breakdown = pd.read_excel(xlsx, sheet_name="cost_co2_breakdown")
    capacity = {}
    for row in breakdown.itertuples():
        capacity.setdefault(row.Technology, {})[str(row.Year)] = {
            "added_MW": float(row.Capacity_added_MW),
            "total_MW": float(row.Capacity_total_MW),
        }
    rep_dispatch = {}
    for csv in sorted(run_dir.glob("rep_dispatch_*.csv")):
        df = pd.read_csv(csv)
        rep_dispatch[csv.stem.removeprefix("rep_dispatch_")] = {
            "columns": list(df.columns),
            "rows": df.astype(float).values.tolist(),
        }
    return {
        "objective_EUR": float(summary.loc[0, "objective_EUR"]),
        "capacity": capacity,
        "rep_dispatch": rep_dispatch,
    }


def reference_path(name: str) -> Path:
    return DATA_DIR / f"reference_{name}.json"


def load_reference(name: str) -> dict:
    return json.loads(reference_path(name).read_text())


def write_reference(name: str, values: dict) -> None:
    """Write the reference JSON; dispatch rounded to 1e-6 MW (far below the tolerance)."""
    out = dict(values)
    out["rep_dispatch"] = {
        year: {"columns": d["columns"],
               "rows": [[round(v, 6) for v in row] for row in d["rows"]]}
        for year, d in values["rep_dispatch"].items()
    }
    out = {"case": name, "settings": {**COMMON_ENV, **CASES[name]}, **out}
    DATA_DIR.mkdir(exist_ok=True)
    text = json.dumps(out, indent=1)
    # one dispatch row per line: collapse the innermost arrays of numbers
    text = re.sub(r"\[[^\[\]{}\"]*\]", lambda m: re.sub(r"\s+", "", m.group(0)).replace(",", ", "), text)
    reference_path(name).write_text(text + "\n")
