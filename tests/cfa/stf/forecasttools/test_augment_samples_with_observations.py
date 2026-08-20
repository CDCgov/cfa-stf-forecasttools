import datetime

import polars as pl
import polars.testing as plt

import cfa.stf.forecasttools as ft


def test_augment_samples_with_observations_adds_observations_to_each_draw():
    forecast_date = datetime.date(2026, 1, 8)
    second_forecast_date = datetime.date(2026, 1, 9)
    observation_dates = [
        datetime.date(2026, 1, 6),
        datetime.date(2026, 1, 7),
        forecast_date,
        second_forecast_date,
    ]
    samples = pl.DataFrame(
        {
            "date": [
                forecast_date,
                forecast_date,
                second_forecast_date,
                second_forecast_date,
            ],
            "location": ["US", "US", "US", "US"],
            "resolution": ["daily", "daily", "daily", "daily"],
            "data_type": ["forecast", "forecast", "forecast", "forecast"],
            ".draw": [0, 1, 0, 1],
            "cases": [20.0, 30.0, 40.0, 50.0],
        }
    )
    observations = pl.DataFrame(
        {
            "date": observation_dates,
            "location": ["US", "US", "US", "US"],
            "resolution": ["daily", "daily", "daily", "daily"],
            "cases": [10.0, 20.0, 999.0, 888.0],
        }
    )

    result = ft.augment_samples_with_observations(samples, observations).sort(
        "date", ".draw"
    )

    expected = pl.DataFrame(
        {
            "date": [
                observation_dates[0],
                observation_dates[0],
                observation_dates[1],
                observation_dates[1],
                forecast_date,
                forecast_date,
                second_forecast_date,
                second_forecast_date,
            ],
            ".draw": [0, 1, 0, 1, 0, 1, 0, 1],
            "cases": [10.0, 10.0, 20.0, 20.0, 20.0, 30.0, 40.0, 50.0],
        }
    )
    plt.assert_frame_equal(result.select("date", ".draw", "cases"), expected)


def test_augment_samples_with_observations_accepts_custom_column_names():
    observation_date = datetime.date(2026, 1, 7)
    forecast_date = datetime.date(2026, 1, 8)
    samples = pl.DataFrame(
        {
            "target_date": [forecast_date],
            "sample_id": [4],
            "type": ["forecast"],
            "cases": [20.0],
        }
    )
    observations = pl.DataFrame({"target_date": [observation_date], "cases": [10.0]})

    result = ft.augment_samples_with_observations(
        samples,
        observations,
        date_col="target_date",
        draw_col="sample_id",
    ).sort("target_date")

    assert result.select("target_date", "sample_id", "cases").rows() == [
        (observation_date, 4, 10.0),
        (forecast_date, 4, 20.0),
    ]
