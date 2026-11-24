# Spec Delta

## Purpose

What the multi-year district-heating model provides so that a program such as TANGO can install it, run it with a free solver, pass it time-varying brine and network conditions, and read its results, without changing the optimisation model.

## ADDED Requirements

### Requirement: Runs with a free solver

The model SHALL run with HiGHS by default, and SHALL keep Gurobi selectable.

#### Scenario: No Gurobi licence

- **WHEN** a user without a Gurobi licence runs the base case on the example data
- **THEN** it solves with HiGHS and writes the same results as before

### Requirement: Results are protected by tests

The repository SHALL contain tests that run small cases with HiGHS and compare the objective, capacities and representative-day dispatch with stored reference values, and CI SHALL run them for every pull request.

#### Scenario: A change alters a result

- **WHEN** a change makes a small case give a different objective or dispatch
- **THEN** the tests fail, naming the case

### Requirement: Installable package without side effects

The model SHALL be installable with `pip install` and importable from any directory, and importing it SHALL NOT install signal handlers, pin CPU cores, change plotting settings or add log handlers.

#### Scenario: Imported by another program

- **WHEN** another program imports the package
- **THEN** nothing global changes until it runs a case

### Requirement: Time-varying brine and network conditions

The model SHALL accept brine flow, production temperature, minimum injection temperature and pressure, and network supply and return temperatures, per milestone year.

#### Scenario: Declining production temperature

- **WHEN** the production temperature is lower in later milestone years
- **THEN** each year's dispatch uses that year's temperature

### Requirement: Reinjection temperature

The results SHALL include the brine reinjection temperature per hour.

#### Scenario: Reading the reinjection temperature

- **WHEN** a case has been solved
- **THEN** the reinjection temperature is available per milestone year and hour, with its unit

### Requirement: Results as data

A solved case SHALL return its status, objective and hourly dispatch per technology as data, with documented units and signs, in addition to the files it writes.

#### Scenario: Reading dispatch without files

- **WHEN** a program solves a case
- **THEN** it reads the dispatch per technology and hour without opening an output file
