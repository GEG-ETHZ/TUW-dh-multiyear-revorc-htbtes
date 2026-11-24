"""
District Heating Optimization Package

Multi-year capacity planning and dispatch optimization for district heating systems.
"""

__version__ = "0.1.0"
__author__ = "Your Name"

from pathlib import Path

# Repository folder (holds data/, revers_surrogates/, results/). This file is
# src/dh_multiyear/__init__.py, so the root is two levels up. It is only valid when
# the package is used from a checkout of the repository (also `pip install -e .`).
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
CONFIG_DIR = DATA_DIR / "config"

# Results directory
RESULTS_DIR = PROJECT_ROOT / "results"
