import datetime

import polars as pl

import cfa.stf.forecasttools as ft


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
