"""Forecasttools helpers exposed through the cfa.stf.forecasttools namespace."""

from importlib import import_module

from .aggregate_to_weekly import (
    ceiling_isoweek,
    ceiling_mmwr_epiweek,
    ceiling_week,
    daily_to_weekly,
    floor_isoweek,
    floor_mmwr_epiweek,
    floor_week,
)
from .location_table import LOCATION_LIST
from .prop_data import (
    augment_samples_with_observations,
    create_proportions,
)
from .utils import coalesce_common_columns


def __getattr__(name):
    if name == "arviz":
        return import_module(".arviz_helpers", __name__)
    if name == "get_us_loc_pop_tbl":
        return import_module(".location_table", __name__).get_us_loc_pop_tbl
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "coalesce_common_columns",
    "augment_samples_with_observations",
    "create_proportions",
    "get_us_loc_pop_tbl",
    "LOCATION_LIST",
    "arviz",
    "daily_to_weekly",
    "floor_week",
    "ceiling_week",
    "floor_isoweek",
    "ceiling_isoweek",
    "floor_mmwr_epiweek",
    "ceiling_mmwr_epiweek",
]
