import datetime

import polars as pl
import pytest

import cfa.stf.forecasttools as ft


def test_augment_samples_with_observations_adds_training_draws():
    forecast_date = datetime.date(2026, 1, 8)
    observation_dates = [datetime.date(2026, 1, 6), datetime.date(2026, 1, 7)]
    samples = pl.DataFrame(
        {
            "date": [forecast_date, forecast_date],
            "location": ["US", "US"],
            "resolution": ["daily", "daily"],
            "data_type": ["forecast", "forecast"],
            ".draw": [0, 1],
            "cases": [20.0, 30.0],
        }
    )
    observations = pl.DataFrame(
        {
            "date": observation_dates,
            "location": ["US", "US"],
            "resolution": ["daily", "daily"],
            "cases": [10.0, 20.0],
        }
    )

    result = ft.augment_samples_with_observations(samples, observations).sort(
        "date", ".draw"
    )

    assert result.select("date", ".draw", "data_type", "cases").rows() == [
        (observation_dates[0], 0, "train", 10.0),
        (observation_dates[0], 1, "train", 10.0),
        (observation_dates[1], 0, "train", 20.0),
        (observation_dates[1], 1, "train", 20.0),
        (forecast_date, 0, "forecast", 20.0),
        (forecast_date, 1, "forecast", 30.0),
    ]


def test_augment_samples_with_observations_rejects_mismatched_resolution():
    samples = pl.DataFrame(
        {
            "date": [datetime.date(2026, 1, 8)],
            "resolution": ["daily"],
            ".draw": [0],
            "cases": [1.0],
        }
    )
    observations = pl.DataFrame(
        {
            "date": [datetime.date(2026, 1, 7)],
            "resolution": ["epiweekly"],
            "cases": [1.0],
        }
    )

    with pytest.raises(ValueError, match="same resolution"):
        ft.augment_samples_with_observations(samples, observations)


def test_create_proportions_joins_shared_identifiers():
    dates = [datetime.date(2026, 1, 7), datetime.date(2026, 1, 8)]
    numerator = pl.DataFrame(
        {
            "date": dates,
            "location": ["US", "US"],
            ".draw": [0, 0],
            "observed_ed_visits": [20.0, 30.0],
        }
    )
    other = pl.DataFrame(
        {
            "date": dates,
            "location": ["US", "US"],
            ".draw": [0, 0],
            "other_ed_visits": [80.0, 70.0],
        }
    )

    result = ft.create_proportions(
        numerator,
        other,
        "observed_ed_visits",
        "other_ed_visits",
    )

    assert result.get_column(".value").to_list() == [0.2, 0.3]
    assert result.get_column(".variable").to_list() == [
        "prop_disease_ed_visits",
        "prop_disease_ed_visits",
    ]
