# Tasks

Each item is a separate pull request. Owners are agreed in `design.md`.

## 0. Agreement

- [ ] 0.1 Go through this proposal and the open questions in `design.md` with the author; record the answers and who does each item.

## 1. Drafted (mandatory, proposed)

- [ ] 1.1 Solver setting (HiGHS default, Gurobi selectable) and `tsam>=3,<4`. Verify the example cases give the same objectives.
- [ ] 1.2 Regression tests and CI. Verify they pass on unchanged code and fail when a stored value is changed.
- [ ] 1.3 Installable package, no side effects on import. Verify the tests pass installed and from another directory.
- [ ] 1.4 Reporting fixes. Verify the objective is unchanged and state which reported values change.

## 2. After agreement (mandatory, proposed)

- [ ] 2.1 Library interface: build a case, solve, read results as data.
- [ ] 2.2 Use NTUA's RevORCHP-SM package for the revORC surrogates.
- [ ] 2.3 Tag a release.

## 3. Requested (mandatory, the author's decision how)

- [ ] 3.1 Reinjection temperature per hour.
- [ ] 3.2 Brine and network conditions per milestone year.
- [ ] 3.3 Output contract: units, signs and indexing.
- [ ] 3.4 Validity of the HT-BTES library when temperatures change.
- [ ] 3.5 Hand-off of the revORC mode decision to TANGO.
- [ ] 3.6 Docstrings with units and ranges on the library interface.

## 4. Nice to have

- [ ] 4.1 Type hints throughout.
- [ ] 4.2 Validity bounds outside the pickled models.
- [ ] 4.3 HT-BTES solve time with HiGHS.
- [ ] 4.4 Remove legacy scenarios and diagnostic settings.
- [ ] 4.5 Milestone years read from the data.
