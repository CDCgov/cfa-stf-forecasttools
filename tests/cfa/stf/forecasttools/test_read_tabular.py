from datetime import date, datetime
from zoneinfo import ZoneInfo

import polars as pl
import polars.testing as plt
import pytest

import cfa.stf.forecasttools as ft


@pytest.fixture
def tabular_data():
    return pl.DataFrame(
        {
            "location": ["US", "01"],
            "value": [1.5, 2.5],
        }
    )


@pytest.mark.parametrize("file_format", ["csv", "tsv", "parquet"])
def test_read_tabular_reads_supported_formats(tmp_path, tabular_data, file_format):
    path = tmp_path / f"data.{file_format}"
    ft.write_tabular(tabular_data, path)

    result = ft.read_tabular(path)

    plt.assert_frame_equal(result, tabular_data)


def test_read_tabular_matches_extension_case_insensitively(tmp_path, tabular_data):
    path = tmp_path / "data.CSV"
    ft.write_tabular(tabular_data, path)

    result = ft.read_tabular(path)

    plt.assert_frame_equal(result, tabular_data)


def test_read_tabular_forwards_reader_options(tmp_path):
    path = tmp_path / "data.csv"
    ft.write_tabular(pl.DataFrame({"value": [1, 2, 3]}), path)

    result = ft.read_tabular(path, n_rows=2)

    assert result.get_column("value").to_list() == [1, 2]


@pytest.mark.parametrize("file_format", ["csv", "tsv"])
def test_read_tabular_parses_dates_by_default(tmp_path, file_format):
    path = tmp_path / f"dates.{file_format}"
    data = pl.DataFrame({"date": [date(2026, 1, 15)], "value": [1]})
    ft.write_tabular(data, path)

    result = ft.read_tabular(path)

    plt.assert_frame_equal(result, data)


def test_read_tabular_allows_disabling_date_parsing(tmp_path):
    path = tmp_path / "dates.csv"
    ft.write_tabular(pl.DataFrame({"date": [date(2026, 1, 15)], "value": [1]}), path)

    result = ft.read_tabular(path, try_parse_dates=False)

    assert result.get_column("date").to_list() == ["2026-01-15"]


def test_read_tabular_corrects_timezone_naive_parquet_timestamps(tmp_path):
    path = tmp_path / "timestamps.parquet"
    tokyo = ZoneInfo("Asia/Tokyo")
    data = pl.DataFrame(
        {
            "timestamp_without_timezone": [datetime(2026, 1, 15, 12, 30)],
            "timestamp_with_timezone": [datetime(2026, 1, 15, 12, 30, tzinfo=tokyo)],
        },
        schema={
            "timestamp_without_timezone": pl.Datetime("us"),
            "timestamp_with_timezone": pl.Datetime("us", "Asia/Tokyo"),
        },
    )
    ft.write_tabular(data, path)

    result = ft.read_tabular(path)

    assert result.schema == pl.Schema(
        {
            "timestamp_without_timezone": pl.Datetime("us", "UTC"),
            "timestamp_with_timezone": pl.Datetime("us", "Asia/Tokyo"),
        }
    )
    assert result.select(pl.all().to_physical()).row(0) == data.select(
        pl.all().to_physical()
    ).row(0)


@pytest.mark.parametrize("filename", ["data.json", "data"])
def test_read_tabular_rejects_unsupported_extensions(tmp_path, filename):
    path = tmp_path / filename

    with pytest.raises(ValueError, match="Unsupported file extension"):
        ft.read_tabular(path)


@pytest.mark.parametrize("filename", ["data.json", "data"])
def test_write_tabular_rejects_unsupported_extensions(tmp_path, filename):
    path = tmp_path / filename

    with pytest.raises(ValueError, match="Unsupported file extension"):
        ft.write_tabular(pl.DataFrame(), path)
