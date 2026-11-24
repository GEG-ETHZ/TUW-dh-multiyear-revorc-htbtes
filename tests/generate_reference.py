"""Regenerate the reference values in tests/data/ for the regression tests.

    python tests/generate_reference.py              # all cases
    python tests/generate_reference.py n1_4days     # one case

The reference files record what the model computes today, so the tests catch any
change of the results. Regenerating them makes a failing test pass again, so only
do it when the change of the results is intended and understood:

- you changed the model on purpose (a new constraint, a corrected cost term, a
  bug fix) and checked that the new numbers are right;
- the input data in data/ (Munich_tech_data.xlsx, example_DH_data.xlsx) was changed
  on purpose;
- a reported value the tests store was wrong and has been fixed.

Do NOT regenerate to make a failure go away that you do not understand, or after
an upgrade of a library (pandas, tsam, HiGHS, ...) without first finding out why
the numbers moved. Commit the new JSON files together with the change that caused
them and say in the commit message which values changed and why.
"""

import sys
import tempfile
from pathlib import Path

from reference_cases import CASES, reference_path, run_case, write_reference


def main(names):
    for name in names or CASES:
        with tempfile.TemporaryDirectory() as tmp:
            values = run_case(name, Path(tmp))
        write_reference(name, values)
        print(f"{name}: objective EUR {values['objective_EUR']:,.2f} -> {reference_path(name)}")


if __name__ == "__main__":
    main(sys.argv[1:])
