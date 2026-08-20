import polars as pl
import polars.testing as plt
import pytest

import cfa.stf.forecasttools as ft


def test_append_prop_data_appends_proportion_rows():
    data = pl.DataFrame(
        {
            "date": [
                "2024-01-02",
                "2024-01-01",
                "2024-01-01",
                "2024-01-02",
                "2024-01-01",
            ],
            "location": ["US", "US", "US", "US", "US"],
            ".variable": [
                "other_ed_visits",
                "observed_ed_visits",
                "other_ed_visits",
                "observed_ed_visits",
                "some_other_variable",
            ],
            ".value": [70, 20, 80, 30, 999],
        }
    )

    result = ft.append_prop_data(data)

    expected = pl.DataFrame(
        {
            "date": [
                "2024-01-01",
                "2024-01-01",
                "2024-01-01",
                "2024-01-01",
                "2024-01-02",
                "2024-01-02",
                "2024-01-02",
            ],
            "location": ["US", "US", "US", "US", "US", "US", "US"],
            ".variable": [
                "observed_ed_visits",
                "other_ed_visits",
                "prop_disease_ed_visits",
                "some_other_variable",
                "observed_ed_visits",
                "other_ed_visits",
                "prop_disease_ed_visits",
            ],
            ".value": [20.0, 80.0, 0.2, 999.0, 30.0, 70.0, 0.3],
        }
    )
    plt.assert_frame_equal(result, expected)


def test_append_prop_data_preserves_additional_identifier_columns():
    data = pl.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-01", "2024-01-01", "2024-01-01"],
            "location": ["CA", "CA", "NY", "NY"],
            "age_group": ["all", "all", "all", "all"],
            ".variable": [
                "observed_ed_visits",
                "other_ed_visits",
                "observed_ed_visits",
                "other_ed_visits",
            ],
            ".value": [1, 3, 4, 6],
        }
    )

    result = ft.append_prop_data(data).filter(
        pl.col(".variable") == "prop_disease_ed_visits"
    )

    expected = pl.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-01"],
            "location": ["CA", "NY"],
            "age_group": ["all", "all"],
            ".variable": ["prop_disease_ed_visits", "prop_disease_ed_visits"],
            ".value": [0.25, 0.4],
        }
    )
    plt.assert_frame_equal(result, expected)


def test_append_prop_data_uses_all_identifiers_to_match_inputs():
    data = pl.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-01"],
            "location": ["US", "US"],
            "source": ["numerator", "other"],
            ".variable": ["observed_ed_visits", "other_ed_visits"],
            ".value": [2, 8],
        }
    )

    result = ft.append_prop_data(data)

    assert not result.get_column(".variable").eq("prop_disease_ed_visits").any()


def test_append_prop_data_allows_custom_variable_names():
    data = pl.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-01"],
            "location": ["US", "US"],
            ".variable": ["num_visits", "denom_other_visits"],
            ".value": [3, 7],
        }
    )

    result = ft.append_prop_data(
        data,
        observed_var="num_visits",
        other_var="denom_other_visits",
        prop_var="prop_num_visits",
    )

    prop_row = result.filter(pl.col(".variable") == "prop_num_visits")
    assert prop_row.item(0, ".value") == 0.3


def test_append_prop_data_matches_null_identifiers():
    data = pl.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-01"],
            "all_null_id": [None, None],
            ".variable": ["observed_ed_visits", "other_ed_visits"],
            ".value": [1, 1],
        }
    )

    result = ft.append_prop_data(data)
    prop_row = result.filter(pl.col(".variable") == "prop_disease_ed_visits")

    assert prop_row.item(0, ".value") == 0.5
    assert prop_row.item(0, "all_null_id") is None


def test_append_prop_data_errors_when_required_column_is_missing():
    data = pl.DataFrame(
        {
            "date": ["2024-01-01"],
            ".variable": ["observed_ed_visits"],
        }
    )

    with pytest.raises(ValueError, match=r"\.value"):
        ft.append_prop_data(data)
