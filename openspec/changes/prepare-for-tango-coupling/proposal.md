# Proposal: prepare the multi-year district-heating model for coupling with TANGO

## Why

In FlexGEO, TANGO (task T4.5) couples the subsurface, the surface plant and the economics of geothermal systems. This model is the district-heating side of that picture: given the brine a doublet delivers and the network's temperatures and demand, it decides each hour how the heat is used (direct use, revORC in ORC or heat-pump mode, HT-BTES storage, boiler), and which capacities to build over the years. TANGO will decide the revORC's operating mode hour by hour, informed by this model's dispatch, so the two models need to exchange data in both directions.

The model is open source on GitHub, documented in its README and user guide, and runs: with the synthetic example data and the free HiGHS solver, the base case (N1) solves in about a second. A few things make it hard to call from another program, and a few would give TANGO results it cannot use:

- It is a set of scripts, not an installable package: modules are found by adding `src/` to the import path, and names such as `models` and `utils` would clash with other packages.
- A run is one `main()` function configured by environment variables, which writes Excel and CSV files and returns nothing. Importing it also installs signal handlers, pins CPU cores and changes matplotlib settings for the whole program.
- The solver is fixed to Gurobi, which needs a licence; other TANGO users and automated tests cannot run it.
- There are no automated tests, so neither of us can show that a change leaves the results unchanged.
- `tsam>=2.3` now installs tsam 4, which breaks the time aggregation with a misleading "tsam is not installed" message.
- The brine conditions are single numbers for the whole horizon and the network temperatures one series for all milestone years, while TANGO's wells deliver brine that changes over the years.
- The model does not compute the brine's reinjection temperature, which TANGO needs to close the loop back to the injection well.
- `revers_surrogates/` holds an older copy (October 2026) of NTUA's RevORCHP-SM scripts. NTUA's corrected release 1.0.2 (10 November 2026) replaced those models. TANGO couples the revORC through NTUA's package, so both models should use the same, current version.

This document proposes what to change, so that we agree before anything is merged. Each change is marked:

- **Mandatory, proposed**: TANGO needs it, and we have drafted the change as a pull request for you to review.
- **Mandatory, requested**: TANGO needs it, but the right change needs your knowledge of the model; we describe what is needed and you decide how.
- **Nice to have**: welcome, but TANGO does not depend on it.

## What changes

Mandatory, proposed (drafted as pull requests in our fork, none changing the optimisation model):

1. A solver setting with HiGHS as the default and Gurobi still available; `tsam>=3,<4`.
2. Regression tests that freeze the current results of small cases, run by GitHub Actions on every pull request.
3. An installable package (`pyproject.toml`), importable from any directory, with no side effects on import.
4. Fixes to three reporting bugs found while reading the code (see `design.md`).

Mandatory, proposed, later (after we agree on them):

5. A small library interface: build a case from inputs, solve it, read the results as data, instead of only files.
6. Use NTUA's RevORCHP-SM package for the revORC surrogates instead of the copy in `revers_surrogates/`.
7. Tagged releases, so TANGO can depend on a fixed version.

Mandatory, requested (your decision how):

8. A reinjection temperature per hour.
9. Brine conditions per milestone year, and network temperatures per milestone year.
10. A documented output contract: units, signs and indexing of the results TANGO reads.
11. When the HT-BTES library stays valid if TANGO changes the brine or network temperatures.
12. How TANGO takes over the revORC's ORC/heat-pump decision.
13. Docstrings with units and ranges on the functions of the library interface (5).

Nice to have: type hints throughout, validity bounds outside the pickles, HT-BTES solve time with HiGHS, removing legacy scenarios and diagnostic settings, milestone years read from the data.

### Not part of this proposal

- Any change to the optimisation model, its constraints or its data.
- How TANGO uses the model; that is TANGO's own proposal in the TANGO repository.

## Capabilities

### New Capabilities

- `tango-interface`: what the model provides so that a program like TANGO can install it, run it with a free solver, pass it time-varying brine and network conditions, and read its results.

## Impact

- New: `tests/`, `.github/workflows/tests.yml`, `pyproject.toml`, `openspec/`.
- Moved: `src/models`, `src/preprocessing`, `src/utils` and the modules they need into one package; the run scripts keep working as before.
- Changed: `requirements.txt` and `environment.yml` (solver and tsam), the solver calls, three reporting functions.
- Unchanged: the optimisation model, its data and its results.
