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


def test_create_proportions_joins_shared_identifiers():
    dates = [datetime.date(2026, 1, 7), datetime.date(2026, 1, 8)]
    numerator_df = pl.DataFrame(
        {
            "date": dates,
            "location": ["US", "US"],
            ".draw": [0, 0],
            "observed_ed_visits": [20.0, 30.0],
        }
    )
    other_df = pl.DataFrame(
        {
            "date": dates,
            "location": ["US", "US"],
            ".draw": [0, 0],
            "other_ed_visits": [80.0, 70.0],
        }
    )

    result = ft.create_proportions(
        numerator_df=numerator_df,
        other_df=other_df,
        num_val_col="observed_ed_visits",
        other_val_col="other_ed_visits",
    )

    assert result.get_column(".value").to_list() == [0.2, 0.3]
    assert result.get_column(".variable").to_list() == [
        "prop_disease_ed_visits",
        "prop_disease_ed_visits",
    ]


def test_create_proportions_does_not_join_mismatched_shared_identifiers():
    numerator_df = pl.DataFrame(
        {
            "date": [datetime.date(2026, 1, 8)],
            "location": ["US"],
            "source": ["numerator"],
            "observed_ed_visits": [20.0],
        }
    )
    other_df = pl.DataFrame(
        {
            "date": [datetime.date(2026, 1, 8)],
            "location": ["US"],
            "source": ["other"],
            "other_ed_visits": [80.0],
        }
    )

    result = ft.create_proportions(
        numerator_df=numerator_df,
        other_df=other_df,
        num_val_col="observed_ed_visits",
        other_val_col="other_ed_visits",
    )

    assert result.is_empty()


def test_create_proportions_supports_same_value_column_name():
    dates = [datetime.date(2026, 1, 7), datetime.date(2026, 1, 8)]
    numerator_df = pl.DataFrame(
        {"date": dates, "location": ["US", "US"], "ed_visits": [20.0, 30.0]}
    )
    other_df = pl.DataFrame(
        {"date": dates, "location": ["US", "US"], "ed_visits": [80.0, 70.0]}
    )

    result = ft.create_proportions(
        numerator_df=numerator_df,
        other_df=other_df,
        num_val_col="ed_visits",
        other_val_col="ed_visits",
        prop_var="ed_visit_proportion",
    )

    assert result.get_column(".value").to_list() == [0.2, 0.3]
    assert result.get_column(".variable").to_list() == [
        "ed_visit_proportion",
        "ed_visit_proportion",
    ]
