"""
District Heating Optimization Package

Multi-year capacity planning and dispatch optimization for district heating systems.
"""

__version__ = "0.1.0"
__author__ = "Your Name"

from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
CONFIG_DIR = DATA_DIR / "config"

# Results directory
RESULTS_DIR = PROJECT_ROOT / "results"
