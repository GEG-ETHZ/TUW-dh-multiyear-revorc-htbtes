"""Preprocessing utilities for district heating optimization."""

from .time_aggregation import (
    create_representative_periods,
    create_representative_days_with_mapping,
    create_representative_hours_with_mapping,
    create_representative_periods_per_year,
    calculate_period_durations,
    validate_representative_periods
)

__all__ = [
    'create_representative_periods',
    'create_representative_days_with_mapping',
    'create_representative_hours_with_mapping',
    'create_representative_periods_per_year',
    'calculate_period_durations',
    'validate_representative_periods'
]
