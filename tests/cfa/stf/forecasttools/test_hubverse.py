"""Tests for forecast-sample to Hubverse conversion."""

import datetime
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal, cast

import polars as pl
import pytest
from polars.testing import assert_frame_equal

import cfa.stf.forecasttools as ft
from cfa.stf.forecasttools.hubverse import (
    _build_task_rows,
    _finalize_output,
    _normalize_mappings,
    _normalize_quantile_levels,
    _normalize_reference_date,
    _normalize_samples,
    _validate_horizon_unit,
    _validate_sample_coverage,
)

_TEST_DATA_DIR = Path(__file__).parent / "test_data"
_HUBVERSE_OUTPUT_COLUMNS = (
    "reference_date",
    "target",
    "horizon",
    "target_end_date",
    "location",
    "output_type",
    "output_type_id",
    "value",
)
_R_HUB_QUANTILES = [
    0.01,
    0.025,
    *[level / 20 for level in range(1, 20)],
    0.975,
    0.99,
]


def _r_parity_samples() -> pl.DataFrame:
    """Load the R parity input and normalize its weekly resolution name."""
    return pl.read_csv(
        _TEST_DATA_DIR / "r_hubverse_parity_input.csv",
        try_parse_dates=True,
    ).with_columns(
        pl.col("resolution").replace_strict({"daily": "daily", "epiweekly": "weekly"})
    )


def _r_parity_expected(
    filename: str,
    *,
    output_type: Literal["sample", "quantile"],
) -> pl.DataFrame:
    """Load an R golden output using the Python output schema and ordering."""
    expected = pl.read_csv(
        _TEST_DATA_DIR / filename,
        try_parse_dates=True,
    ).with_columns(
        pl.col("reference_date").cast(pl.Date),
        pl.col("target").cast(pl.String),
        pl.col("horizon").cast(pl.Int64),
        pl.col("target_end_date").cast(pl.Date),
        pl.col("location").cast(pl.String),
        pl.col("output_type").cast(pl.String),
        pl.col("value").cast(pl.Float64),
    )
    if output_type == "sample":
        expected = expected.with_columns(pl.col("output_type_id").cast(pl.Int64))
    else:
        expected = expected.with_columns(pl.col("output_type_id").cast(pl.Float64))
    return expected.select(_HUBVERSE_OUTPUT_COLUMNS).sort(_HUBVERSE_OUTPUT_COLUMNS[:-1])


def _valid_samples() -> pl.DataFrame:
    """Return a valid canonical sample table."""
    first_date = datetime.date(2026, 1, 1)
    return pl.DataFrame(
        {
            "draw": [0, 0, 1, 1],
            "date": [
                first_date,
                first_date + datetime.timedelta(days=1),
            ]
            * 2,
            "location": ["US"] * 4,
            "variable": ["admissions"] * 4,
            "value": [1.0, 2.0, 3.0, 4.0],
            "resolution": ["daily"] * 4,
        },
        schema_overrides={"draw": pl.UInt32},
    )


def _normalize(samples: pl.DataFrame) -> pl.DataFrame:
    """Normalize samples using the canonical source-column names."""
    return _normalize_samples(
        samples,
        draw_col="draw",
        date_col="date",
        location_col="location",
        variable_col="variable",
        value_col="value",
        resolution_col="resolution",
    )


def _series_samples(
    dates: list[datetime.date],
    *,
    location: str = "US",
    variable: str = "admissions",
    resolution: str = "daily",
) -> pl.DataFrame:
    """Return two draws covering the supplied dates for one series."""
    row_count = len(dates) * 2
    return pl.DataFrame(
        {
            "draw": [0] * len(dates) + [1] * len(dates),
            "date": dates * 2,
            "location": [location] * row_count,
            "variable": [variable] * row_count,
            "value": [float(value) for value in range(row_count)],
            "resolution": [resolution] * row_count,
        }
    )


def _representative_samples() -> pl.DataFrame:
    """Return samples spanning draws, dates, resolutions, and locations."""
    reference_date = datetime.date(2026, 1, 1)
    rows: list[tuple[int, datetime.date, str, str, float, str]] = []
    locations = [("US", 0.0), ("CA", 100.0)]
    series = [
        ("daily", [reference_date, reference_date + datetime.timedelta(days=1)], 0.0),
        (
            "weekly",
            [
                reference_date + datetime.timedelta(days=2),
                reference_date + datetime.timedelta(days=9),
            ],
            20.0,
        ),
    ]
    for location, location_offset in locations:
        for resolution, dates, resolution_offset in series:
            for draw in (0, 1):
                for date_index, date in enumerate(dates):
                    value = (
                        location_offset
                        + resolution_offset
                        + date_index * 4.0
                        + draw * 2.0
                    )
                    rows.append((draw, date, location, "admissions", value, resolution))
    return pl.DataFrame(
        rows,
        schema={
            "draw": pl.Int64,
            "date": pl.Date,
            "location": pl.String,
            "variable": pl.String,
            "value": pl.Float64,
            "resolution": pl.String,
        },
        orient="row",
    )


def test_normalize_samples_uses_canonical_names_and_types() -> None:
    """Canonical input is selected in order and integer draw IDs become Int64."""
    result = _normalize(_valid_samples().select(reversed(_valid_samples().columns)))

    assert result.columns == [
        "draw",
        "date",
        "location",
        "variable",
        "value",
        "resolution",
    ]
    assert result.schema["draw"] == pl.Int64
    assert result.drop("draw").equals(_valid_samples().drop("draw"))


def test_normalize_samples_supports_source_column_aliases() -> None:
    """Configured aliases are renamed to the canonical internal names."""
    aliases = {
        "draw": ".draw",
        "location": "geo_value",
        "variable": ".variable",
        "value": ".value",
    }
    samples = _valid_samples().rename(aliases)

    result = _normalize_samples(
        samples,
        draw_col=".draw",
        date_col="date",
        location_col="geo_value",
        variable_col=".variable",
        value_col=".value",
        resolution_col="resolution",
    )

    expected = _valid_samples().with_columns(pl.col("draw").cast(pl.Int64))
    assert result.equals(expected)


def test_normalize_samples_requires_polars_dataframe() -> None:
    """Non-Polars sample tables are rejected."""
    with pytest.raises(TypeError, match="Polars DataFrame"):
        _normalize(cast(pl.DataFrame, {"draw": [0]}))


@pytest.mark.parametrize("column_name", [1, None])
def test_normalize_samples_requires_string_column_names(
    column_name: object,
) -> None:
    """Source-column parameters must be strings."""
    with pytest.raises(TypeError, match="draw_col must be a string"):
        _normalize_samples(
            _valid_samples(),
            draw_col=cast(str, column_name),
            date_col="date",
            location_col="location",
            variable_col="variable",
            value_col="value",
            resolution_col="resolution",
        )


def test_normalize_samples_requires_nonempty_column_names() -> None:
    """Whitespace-only source-column names are rejected."""
    with pytest.raises(ValueError, match="draw_col must be a nonempty string"):
        _normalize_samples(
            _valid_samples(),
            draw_col=" ",
            date_col="date",
            location_col="location",
            variable_col="variable",
            value_col="value",
            resolution_col="resolution",
        )


def test_normalize_samples_requires_distinct_column_names() -> None:
    """One source column cannot serve two canonical fields."""
    with pytest.raises(ValueError, match="source column names must be distinct"):
        _normalize_samples(
            _valid_samples(),
            draw_col="draw",
            date_col="draw",
            location_col="location",
            variable_col="variable",
            value_col="value",
            resolution_col="resolution",
        )


def test_normalize_samples_requires_source_columns() -> None:
    """Missing required source columns are reported."""
    with pytest.raises(ValueError, match="missing.*\\['value'\\]"):
        _normalize(_valid_samples().drop("value"))


def test_normalize_samples_ignores_extra_columns() -> None:
    """Columns outside the predictive-draw contract are ignored."""
    samples = _valid_samples().with_columns(pl.lit("model-a").alias("model"))

    result = _normalize(samples)

    assert result.equals(_normalize(_valid_samples()))


def test_normalize_samples_rejects_empty_table() -> None:
    """A sample table must contain at least one row."""
    with pytest.raises(ValueError, match="must not be empty"):
        _normalize(_valid_samples().head(0))


@pytest.mark.parametrize(
    "column",
    ["draw", "date", "location", "variable", "value", "resolution"],
)
def test_normalize_samples_rejects_nulls(column: str) -> None:
    """Nulls are rejected in every required source column."""
    samples = _valid_samples()
    values = samples.get_column(column).to_list()
    values[0] = None
    samples = samples.with_columns(
        pl.Series(column, values, dtype=samples.schema[column])
    )

    with pytest.raises(ValueError, match=column):
        _normalize(samples)


@pytest.mark.parametrize(
    ("column", "dtype", "message"),
    [
        ("draw", pl.Float64, "draw column must have an integer or String type"),
        ("date", pl.Datetime, "date column must have Polars Date type"),
        ("location", pl.Categorical, "location column must have Polars String type"),
        ("variable", pl.Categorical, "variable column must have Polars String type"),
        (
            "resolution",
            pl.Categorical,
            "resolution column must have Polars String type",
        ),
        ("value", pl.String, "value column must have a numeric type"),
    ],
)
def test_normalize_samples_rejects_invalid_column_types(
    column: str,
    dtype: pl.DataType,
    message: str,
) -> None:
    """Each canonical field requires its documented Polars type."""
    samples = _valid_samples().with_columns(pl.col(column).cast(dtype))

    with pytest.raises(TypeError, match=message):
        _normalize(samples)


def test_normalize_samples_rejects_draw_above_int64_range() -> None:
    """Draw IDs must fit in the output Int64 type."""
    samples = (
        _valid_samples()
        .head(1)
        .with_columns(pl.lit(2**63, dtype=pl.UInt64).alias("draw"))
    )

    with pytest.raises(ValueError, match="signed 64-bit integer"):
        _normalize(samples)


def test_normalize_samples_preserves_negative_draw() -> None:
    """Integer draw IDs are identifiers and may be negative."""
    samples = _valid_samples().with_columns(
        pl.when(pl.int_range(pl.len()) == 0)
        .then(-1)
        .otherwise(pl.col("draw"))
        .alias("draw")
    )

    assert _normalize(samples).get_column("draw").to_list()[0] == -1


@pytest.mark.parametrize("value", ["", "  "])
def test_normalize_samples_rejects_empty_string_draw(value: str) -> None:
    """String draw identifiers must contain visible text."""
    samples = _valid_samples().with_columns(pl.lit(value).alias("draw"))

    with pytest.raises(ValueError, match="string draw values must be nonempty"):
        _normalize(samples)


@pytest.mark.parametrize("column", ["location", "variable"])
@pytest.mark.parametrize("value", ["", "  "])
def test_normalize_samples_rejects_empty_strings(
    column: str,
    value: str,
) -> None:
    """Location and variable identifiers must contain visible text."""
    samples = _valid_samples().with_columns(pl.lit(value).alias(column))

    with pytest.raises(ValueError, match=f"{column} values must be nonempty"):
        _normalize(samples)


def test_normalize_samples_rejects_unsupported_resolution() -> None:
    """Only runner-supported resolution labels are accepted."""
    samples = _valid_samples().with_columns(pl.lit("epiweekly").alias("resolution"))

    with pytest.raises(ValueError, match="'epiweekly'"):
        _normalize(samples)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_normalize_samples_rejects_nonfinite_value(value: float) -> None:
    """Forecast values must be finite."""
    samples = _valid_samples().with_columns(
        pl.when(pl.int_range(pl.len()) == 0)
        .then(value)
        .otherwise(pl.col("value"))
        .alias("value")
    )

    with pytest.raises(ValueError, match="only finite values"):
        _normalize(samples)


def test_normalize_samples_rejects_duplicate_identity() -> None:
    """Rows cannot repeat a draw and forecast-task identity."""
    samples = pl.concat([_valid_samples(), _valid_samples().head(1)])

    with pytest.raises(ValueError, match="duplicate.*rows"):
        _normalize(samples)


def test_public_converters_apply_source_validation() -> None:
    """Both public converters normalize input before later conversion steps."""
    with pytest.raises(ValueError, match="missing"):
        ft.samples_to_hubverse(
            _valid_samples().drop("value"),
            reference_date=datetime.date(2026, 1, 1),
            target_map={("admissions", "daily"): "inc admissions"},
            location_map={"US": "US"},
            horizon_unit="days",
        )
    with pytest.raises(ValueError, match="missing"):
        ft.samples_to_hubverse_quantiles(
            _valid_samples().drop("value"),
            reference_date=datetime.date(2026, 1, 1),
            target_map={("admissions", "daily"): "inc admissions"},
            location_map={"US": "US"},
            horizon_unit="days",
            quantile_levels=[0.5],
        )


def test_validate_sample_coverage_accepts_independent_series_dates() -> None:
    """Series may have different valid date sequences and resolutions."""
    daily_start = datetime.date(2026, 1, 1)
    weekly_start = datetime.date(2026, 1, 3)
    samples = pl.concat(
        [
            _series_samples([daily_start, daily_start + datetime.timedelta(days=1)]),
            _series_samples(
                [weekly_start, weekly_start + datetime.timedelta(days=7)],
                variable="hospitalizations",
                resolution="weekly",
            ),
        ]
    )

    _validate_sample_coverage(_normalize(samples))


def test_validate_sample_coverage_requires_shared_draw_identities() -> None:
    """Every forecast series must contain the same draw IDs."""
    start = datetime.date(2026, 1, 1)
    dates = [start, start + datetime.timedelta(days=1)]
    first_series = _series_samples(dates)
    incomplete_series = _series_samples(
        dates,
        variable="hospitalizations",
    ).filter(pl.col("draw") == 0)
    samples = pl.concat([first_series, incomplete_series])

    with pytest.raises(ValueError, match="same draw identities"):
        _validate_sample_coverage(_normalize(samples))


def test_validate_sample_coverage_requires_shared_dates_within_series() -> None:
    """Draws in one series must cover identical dates."""
    start = datetime.date(2026, 1, 1)
    samples = _series_samples([start, start + datetime.timedelta(days=1)])
    samples = samples.with_columns(
        pl.Series(
            "date",
            [
                start,
                start + datetime.timedelta(days=1),
                start,
                start + datetime.timedelta(days=2),
            ],
            dtype=pl.Date,
        )
    )

    with pytest.raises(ValueError, match="same dates"):
        _validate_sample_coverage(_normalize(samples))


def test_validate_sample_coverage_requires_consecutive_daily_dates() -> None:
    """Daily series cannot contain gaps."""
    start = datetime.date(2026, 1, 1)
    samples = _series_samples([start, start + datetime.timedelta(days=2)])

    with pytest.raises(ValueError, match="daily.*consecutive"):
        _validate_sample_coverage(_normalize(samples))


def test_validate_sample_coverage_accepts_weekly_spacing() -> None:
    """Weekly dates exactly seven days apart are valid."""
    start = datetime.date(2026, 1, 3)
    samples = _series_samples(
        [start, start + datetime.timedelta(days=7)],
        resolution="weekly",
    )

    _validate_sample_coverage(_normalize(samples))


def test_validate_sample_coverage_requires_weekly_spacing() -> None:
    """Weekly series must use seven-day spacing."""
    start = datetime.date(2026, 1, 3)
    samples = _series_samples(
        [start, start + datetime.timedelta(days=8)],
        resolution="weekly",
    )

    with pytest.raises(ValueError, match="weekly.*seven days apart"):
        _validate_sample_coverage(_normalize(samples))


def test_public_converters_apply_coverage_validation() -> None:
    """Both public converters reject internally incomplete trajectories."""
    start = datetime.date(2026, 1, 1)
    dates = [start, start + datetime.timedelta(days=1)]
    samples = pl.concat(
        [
            _series_samples(dates),
            _series_samples(dates, variable="hospitalizations").filter(
                pl.col("draw") == 0
            ),
        ]
    )

    with pytest.raises(ValueError, match="same draw identities"):
        ft.samples_to_hubverse(
            samples,
            reference_date=start,
            target_map={},
            location_map={},
            horizon_unit="days",
        )
    with pytest.raises(ValueError, match="same draw identities"):
        ft.samples_to_hubverse_quantiles(
            samples,
            reference_date=start,
            target_map={},
            location_map={},
            horizon_unit="days",
            quantile_levels=[0.5],
        )


def test_normalize_reference_date_accepts_date_and_datetime() -> None:
    """Reference datetimes are reduced to their calendar date."""
    date = datetime.date(2026, 1, 2)
    date_time = datetime.datetime(2026, 1, 2, 15, 30)

    assert _normalize_reference_date(date) == date
    assert _normalize_reference_date(date_time) == date


def test_normalize_reference_date_rejects_other_types() -> None:
    """Reference dates must use a datetime date type."""
    with pytest.raises(TypeError, match="reference_date"):
        _normalize_reference_date(cast(datetime.date, "2026-01-02"))


@pytest.mark.parametrize("horizon_unit", ["days", "weeks"])
def test_validate_horizon_unit_accepts_supported_values(
    horizon_unit: Literal["days", "weeks"],
) -> None:
    """Both documented horizon units are accepted."""
    _validate_horizon_unit(horizon_unit)


def test_validate_horizon_unit_rejects_unknown_value() -> None:
    """Unknown string horizon units are rejected."""
    with pytest.raises(ValueError, match="days.*weeks"):
        _validate_horizon_unit(cast(Literal["days", "weeks"], "months"))


def test_validate_horizon_unit_rejects_nonstring_value() -> None:
    """Horizon units must be strings."""
    with pytest.raises(TypeError, match="must be a string"):
        _validate_horizon_unit(cast(Literal["days", "weeks"], 7))


def test_normalize_mappings_accepts_unused_entries() -> None:
    """Valid unused entries can remain in shared caller mappings."""
    targets = {
        ("admissions", "daily"): "inc admissions",
        ("hospitalizations", "weekly"): "wk inc hospitalizations",
    }
    locations = {"US": "US", "CA": "06"}

    result_targets, result_locations = _normalize_mappings(
        _normalize(_valid_samples()),
        target_map=targets,
        location_map=locations,
    )

    assert result_targets == targets
    assert result_locations == locations


def test_normalize_mappings_reports_all_missing_observed_keys() -> None:
    """One error reports every missing target and location key."""
    start = datetime.date(2026, 1, 1)
    dates = [start, start + datetime.timedelta(days=1)]
    samples = pl.concat(
        [
            _series_samples(dates),
            _series_samples(
                dates,
                location="CA",
                variable="hospitalizations",
            ),
        ]
    )

    with pytest.raises(ValueError) as error:
        _normalize_mappings(
            _normalize(samples),
            target_map={},
            location_map={},
        )

    message = str(error.value)
    assert "('admissions', 'daily')" in message
    assert "('hospitalizations', 'daily')" in message
    assert "'CA'" in message
    assert "'US'" in message


@pytest.mark.parametrize("mapping_name", ["target_map", "location_map"])
def test_normalize_mappings_requires_mapping_objects(mapping_name: str) -> None:
    """Target and location mappings must implement Mapping."""
    target_map: Mapping[tuple[str, str], str] = {
        ("admissions", "daily"): "inc admissions"
    }
    location_map: Mapping[str, str] = {"US": "US"}
    if mapping_name == "target_map":
        target_map = cast(Mapping[tuple[str, str], str], [])
    else:
        location_map = cast(Mapping[str, str], [])

    with pytest.raises(TypeError, match=mapping_name):
        _normalize_mappings(
            _normalize(_valid_samples()),
            target_map=target_map,
            location_map=location_map,
        )


@pytest.mark.parametrize(
    "key",
    [
        "admissions",
        ("admissions",),
        ("admissions", 1),
        ("", "daily"),
    ],
)
def test_normalize_mappings_rejects_invalid_target_keys(key: object) -> None:
    """Target keys must be nonempty string pairs."""
    target_map = {
        cast(tuple[str, str], key): "inc admissions",
        ("admissions", "daily"): "inc admissions",
    }

    with pytest.raises(TypeError, match="target_map keys"):
        _normalize_mappings(
            _normalize(_valid_samples()),
            target_map=target_map,
            location_map={"US": "US"},
        )


@pytest.mark.parametrize(
    ("value", "error_type", "message"),
    [
        (1, TypeError, "values must be strings"),
        ("", ValueError, "values must be nonempty"),
        ("  ", ValueError, "values must be nonempty"),
    ],
)
def test_normalize_mappings_rejects_invalid_target_values(
    value: object,
    error_type: type[Exception],
    message: str,
) -> None:
    """Target mapping values must be nonempty strings."""
    with pytest.raises(error_type, match=message):
        _normalize_mappings(
            _normalize(_valid_samples()),
            target_map={("admissions", "daily"): cast(str, value)},
            location_map={"US": "US"},
        )


@pytest.mark.parametrize(
    ("key", "error_type", "message"),
    [
        (1, TypeError, "keys must be strings"),
        ("", ValueError, "keys must be nonempty"),
        ("  ", ValueError, "keys must be nonempty"),
    ],
)
def test_normalize_mappings_rejects_invalid_location_keys(
    key: object,
    error_type: type[Exception],
    message: str,
) -> None:
    """Location mapping keys must be nonempty strings."""
    location_map = {
        cast(str, key): "US",
        "US": "US",
    }

    with pytest.raises(error_type, match=message):
        _normalize_mappings(
            _normalize(_valid_samples()),
            target_map={("admissions", "daily"): "inc admissions"},
            location_map=location_map,
        )


@pytest.mark.parametrize(
    ("value", "error_type", "message"),
    [
        (1, TypeError, "values must be strings"),
        ("", ValueError, "values must be nonempty"),
        ("  ", ValueError, "values must be nonempty"),
    ],
)
def test_normalize_mappings_rejects_invalid_location_values(
    value: object,
    error_type: type[Exception],
    message: str,
) -> None:
    """Location mapping values must be nonempty strings."""
    with pytest.raises(error_type, match=message):
        _normalize_mappings(
            _normalize(_valid_samples()),
            target_map={("admissions", "daily"): "inc admissions"},
            location_map={"US": cast(str, value)},
        )


def test_normalize_quantile_levels_sorts_numeric_levels() -> None:
    """Valid integer and float quantile levels become sorted floats."""
    assert _normalize_quantile_levels([1, 0.25, 0]) == [0.0, 0.25, 1.0]


@pytest.mark.parametrize("levels", [None, "0.5", {0.5}])
def test_normalize_quantile_levels_requires_sequence(levels: object) -> None:
    """Quantile levels must be supplied as a non-string sequence."""
    with pytest.raises(TypeError, match="sequence"):
        _normalize_quantile_levels(cast(Sequence[float], levels))


def test_normalize_quantile_levels_rejects_empty_sequence() -> None:
    """At least one quantile level is required."""
    with pytest.raises(ValueError, match="must not be empty"):
        _normalize_quantile_levels([])


@pytest.mark.parametrize("level", [True, "0.5", None])
def test_normalize_quantile_levels_rejects_nonnumeric_values(
    level: object,
) -> None:
    """Booleans and nonnumeric quantile levels are rejected."""
    with pytest.raises(TypeError, match="numeric and not boolean"):
        _normalize_quantile_levels([cast(float, level)])


@pytest.mark.parametrize("level", [float("nan"), float("inf"), float("-inf")])
def test_normalize_quantile_levels_rejects_nonfinite_values(level: float) -> None:
    """Quantile levels must be finite."""
    with pytest.raises(ValueError, match="must be finite"):
        _normalize_quantile_levels([level])


@pytest.mark.parametrize("level", [-0.1, 1.1])
def test_normalize_quantile_levels_rejects_out_of_range_values(
    level: float,
) -> None:
    """Quantile levels must be between zero and one."""
    with pytest.raises(ValueError, match="between 0 and 1"):
        _normalize_quantile_levels([level])


@pytest.mark.parametrize("levels", [[0.5, 0.5], [0.0, -0.0]])
def test_normalize_quantile_levels_rejects_duplicates(
    levels: list[float],
) -> None:
    """Numerically equivalent quantile levels cannot be repeated."""
    with pytest.raises(ValueError, match="duplicates"):
        _normalize_quantile_levels(levels)


def test_public_sample_converter_applies_configuration_validation() -> None:
    """The sample converter validates caller configuration after samples."""
    with pytest.raises(ValueError, match="horizon_unit"):
        ft.samples_to_hubverse(
            _valid_samples(),
            reference_date=datetime.date(2026, 1, 1),
            target_map={("admissions", "daily"): "inc admissions"},
            location_map={"US": "US"},
            horizon_unit=cast(Literal["days", "weeks"], "months"),
        )


def test_public_quantile_converter_applies_level_validation() -> None:
    """The quantile converter validates caller-supplied probabilities."""
    with pytest.raises(ValueError, match="duplicates"):
        ft.samples_to_hubverse_quantiles(
            _valid_samples(),
            reference_date=datetime.date(2026, 1, 1),
            target_map={("admissions", "daily"): "inc admissions"},
            location_map={"US": "US"},
            horizon_unit="days",
            quantile_levels=[0.5, 0.5],
        )


def test_build_task_rows_maps_columns_and_day_horizons() -> None:
    """Task rows contain mapped identifiers and signed day horizons."""
    reference_date = datetime.date(2026, 1, 2)
    dates = [
        reference_date - datetime.timedelta(days=1),
        reference_date,
        reference_date + datetime.timedelta(days=1),
    ]
    samples = _normalize(_series_samples(dates))

    result = _build_task_rows(
        samples,
        reference_date=reference_date,
        target_map={("admissions", "daily"): "inc admissions"},
        location_map={"US": "59"},
        horizon_unit="days",
    )

    assert result.columns == [
        "reference_date",
        "target",
        "horizon",
        "target_end_date",
        "location",
        "draw",
        "value",
    ]
    assert result.schema == pl.Schema(
        {
            "reference_date": pl.Date,
            "target": pl.String,
            "horizon": pl.Int64,
            "target_end_date": pl.Date,
            "location": pl.String,
            "draw": pl.Int64,
            "value": pl.Float64,
        }
    )
    assert result.select("target_end_date", "horizon").unique().sort(
        "target_end_date"
    ).rows() == [
        (dates[0], -1),
        (dates[1], 0),
        (dates[2], 1),
    ]
    assert result.get_column("reference_date").unique().to_list() == [reference_date]
    assert result.get_column("target").unique().to_list() == ["inc admissions"]
    assert result.get_column("location").unique().to_list() == ["59"]
    assert result.get_column("draw").unique().sort().to_list() == [0, 1]
    assert result.get_column("value").sort().to_list() == [
        0.0,
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
    ]


def test_build_task_rows_maps_variable_and_resolution_pairs() -> None:
    """One variable can map to distinct daily and weekly targets."""
    reference_date = datetime.date(2026, 1, 1)
    samples = pl.concat(
        [
            _series_samples([reference_date]),
            _series_samples(
                [reference_date],
                resolution="weekly",
            ),
        ]
    )

    result = _build_task_rows(
        _normalize(samples),
        reference_date=reference_date,
        target_map={
            ("admissions", "daily"): "inc admissions",
            ("admissions", "weekly"): "wk inc admissions",
        },
        location_map={"US": "US"},
        horizon_unit="days",
    )

    assert sorted(result.get_column("target").unique()) == [
        "inc admissions",
        "wk inc admissions",
    ]


def test_build_task_rows_maps_multiple_locations_without_losing_rows() -> None:
    """Each source location is mapped while preserving every sample row."""
    reference_date = datetime.date(2026, 1, 1)
    samples = pl.concat(
        [
            _series_samples([reference_date]),
            _series_samples([reference_date], location="CA"),
        ]
    )

    result = _build_task_rows(
        _normalize(samples),
        reference_date=reference_date,
        target_map={("admissions", "daily"): "inc admissions"},
        location_map={"US": "59", "CA": "06"},
        horizon_unit="days",
    )

    assert result.height == samples.height
    assert sorted(result.get_column("location").unique()) == ["06", "59"]


def test_build_task_rows_calculates_exact_week_horizons() -> None:
    """Whole-week differences produce signed integer week horizons."""
    reference_date = datetime.date(2026, 1, 8)
    dates = [
        reference_date - datetime.timedelta(days=7),
        reference_date,
        reference_date + datetime.timedelta(days=7),
    ]
    samples = _normalize(_series_samples(dates, resolution="weekly"))

    result = _build_task_rows(
        samples,
        reference_date=reference_date,
        target_map={("admissions", "weekly"): "wk inc admissions"},
        location_map={"US": "US"},
        horizon_unit="weeks",
    )

    assert result.select("target_end_date", "horizon").unique().sort(
        "target_end_date"
    ).rows() == [
        (dates[0], -1),
        (dates[1], 0),
        (dates[2], 1),
    ]


def test_build_task_rows_rejects_fractional_week_horizon() -> None:
    """Week horizons require target dates an exact number of weeks away."""
    reference_date = datetime.date(2026, 1, 1)
    target_date = reference_date + datetime.timedelta(days=1)
    samples = _normalize(_series_samples([target_date], resolution="weekly"))

    with pytest.raises(ValueError, match="whole number of weeks"):
        _build_task_rows(
            samples,
            reference_date=reference_date,
            target_map={("admissions", "weekly"): "wk inc admissions"},
            location_map={"US": "US"},
            horizon_unit="weeks",
        )


def test_public_converters_build_horizons() -> None:
    """Both public converters reject fractional weeks during task construction."""
    reference_date = datetime.date(2026, 1, 1)
    target_date = reference_date + datetime.timedelta(days=1)
    samples = _series_samples([target_date], resolution="weekly")
    target_map = {("admissions", "weekly"): "wk inc admissions"}
    location_map = {"US": "US"}

    with pytest.raises(ValueError, match="whole number of weeks"):
        ft.samples_to_hubverse(
            samples,
            reference_date=reference_date,
            target_map=target_map,
            location_map=location_map,
            horizon_unit="weeks",
        )
    with pytest.raises(ValueError, match="whole number of weeks"):
        ft.samples_to_hubverse_quantiles(
            samples,
            reference_date=reference_date,
            target_map=target_map,
            location_map=location_map,
            horizon_unit="weeks",
            quantile_levels=[0.5],
        )


def test_samples_to_hubverse_returns_sample_output() -> None:
    """The public sample converter returns complete Hubverse sample rows."""
    reference_date = datetime.date(2026, 1, 1)

    result = ft.samples_to_hubverse(
        _valid_samples(),
        reference_date=reference_date,
        target_map={("admissions", "daily"): "inc admissions"},
        location_map={"US": "59"},
        horizon_unit="days",
    ).sort("output_type_id", "target_end_date")

    assert result.schema == pl.Schema(
        {
            "reference_date": pl.Date,
            "target": pl.String,
            "horizon": pl.Int64,
            "target_end_date": pl.Date,
            "location": pl.String,
            "output_type": pl.String,
            "output_type_id": pl.Int64,
            "value": pl.Float64,
        }
    )
    assert result.rows() == [
        (
            reference_date,
            "inc admissions",
            0,
            datetime.date(2026, 1, 1),
            "59",
            "sample",
            0,
            1.0,
        ),
        (
            reference_date,
            "inc admissions",
            1,
            datetime.date(2026, 1, 2),
            "59",
            "sample",
            0,
            2.0,
        ),
        (
            reference_date,
            "inc admissions",
            0,
            datetime.date(2026, 1, 1),
            "59",
            "sample",
            1,
            3.0,
        ),
        (
            reference_date,
            "inc admissions",
            1,
            datetime.date(2026, 1, 2),
            "59",
            "sample",
            1,
            4.0,
        ),
    ]


def test_samples_to_hubverse_supports_aliases_in_public_api() -> None:
    """The public sample converter accepts explicitly named source columns."""
    aliases = {
        "draw": ".draw",
        "location": "geo_value",
        "variable": ".variable",
        "value": ".value",
    }
    aliased_samples = _valid_samples().rename(aliases)

    result = ft.samples_to_hubverse(
        aliased_samples,
        reference_date=datetime.date(2026, 1, 1),
        target_map={("admissions", "daily"): "inc admissions"},
        location_map={"US": "59"},
        horizon_unit="days",
        draw_col=".draw",
        location_col="geo_value",
        variable_col=".variable",
        value_col=".value",
    )
    expected = ft.samples_to_hubverse(
        _valid_samples(),
        reference_date=datetime.date(2026, 1, 1),
        target_map={("admissions", "daily"): "inc admissions"},
        location_map={"US": "59"},
        horizon_unit="days",
    )

    assert result.equals(expected)


def test_samples_to_hubverse_preserves_noncontiguous_draw_ids() -> None:
    """Integer source draw IDs are preserved without rebasing."""
    samples = _valid_samples().with_columns(pl.col("draw").replace({0: 2, 1: 7}))

    result = ft.samples_to_hubverse(
        samples,
        reference_date=datetime.date(2026, 1, 1),
        target_map={("admissions", "daily"): "inc admissions"},
        location_map={"US": "59"},
        horizon_unit="days",
    )

    assert result.get_column("output_type_id").unique().sort().to_list() == [2, 7]


def test_samples_to_hubverse_preserves_string_draw_ids() -> None:
    """String source draw IDs become unchanged Hubverse sample IDs."""
    samples = _valid_samples().with_columns(
        pl.col("draw").replace_strict({0: "sample-a", 1: "sample-b"})
    )

    result = ft.samples_to_hubverse(
        samples,
        reference_date=datetime.date(2026, 1, 1),
        target_map={("admissions", "daily"): "inc admissions"},
        location_map={"US": "59"},
        horizon_unit="days",
    )

    assert result.schema["output_type_id"] == pl.String
    assert result.get_column("output_type_id").unique().sort().to_list() == [
        "sample-a",
        "sample-b",
    ]


def test_samples_to_hubverse_normalizes_datetime_reference() -> None:
    """A reference datetime is represented as its calendar date."""
    reference_datetime = datetime.datetime(2026, 1, 1, 15, 30)

    result = ft.samples_to_hubverse(
        _valid_samples(),
        reference_date=reference_datetime,
        target_map={("admissions", "daily"): "inc admissions"},
        location_map={"US": "US"},
        horizon_unit="days",
    )

    assert result.get_column("reference_date").unique().to_list() == [
        reference_datetime.date()
    ]


def test_samples_to_hubverse_quantiles_uses_linear_interpolation() -> None:
    """The public quantile converter matches R type-7 interpolation."""
    reference_date = datetime.date(2026, 1, 1)
    samples = pl.DataFrame(
        {
            "draw": [0, 1, 2, 3],
            "date": [reference_date] * 4,
            "location": ["US"] * 4,
            "variable": ["admissions"] * 4,
            "value": [0.0, 10.0, 20.0, 30.0],
            "resolution": ["daily"] * 4,
        }
    )

    result = ft.samples_to_hubverse_quantiles(
        samples,
        reference_date=reference_date,
        target_map={("admissions", "daily"): "inc admissions"},
        location_map={"US": "59"},
        horizon_unit="days",
        quantile_levels=[1, 0.25, 0, 0.5],
    ).sort("output_type_id")

    assert result.schema == pl.Schema(
        {
            "reference_date": pl.Date,
            "target": pl.String,
            "horizon": pl.Int64,
            "target_end_date": pl.Date,
            "location": pl.String,
            "output_type": pl.String,
            "output_type_id": pl.Float64,
            "value": pl.Float64,
        }
    )
    assert result.select("output_type_id", "value").rows() == [
        (0.0, 0.0),
        (0.25, 7.5),
        (0.5, 15.0),
        (1.0, 30.0),
    ]
    assert result.get_column("output_type").unique().to_list() == ["quantile"]


def test_samples_to_hubverse_quantiles_groups_by_task() -> None:
    """Quantiles are calculated independently for each Hubverse task row."""
    reference_date = datetime.date(2026, 1, 1)
    us_samples = _series_samples([reference_date])
    ca_samples = _series_samples([reference_date], location="CA").with_columns(
        (pl.col("value") + 100.0).alias("value")
    )
    samples = pl.concat([us_samples, ca_samples])

    result = ft.samples_to_hubverse_quantiles(
        samples,
        reference_date=reference_date,
        target_map={("admissions", "daily"): "inc admissions"},
        location_map={"US": "59", "CA": "06"},
        horizon_unit="days",
        quantile_levels=[0.5],
    ).sort("location")

    assert result.select("location", "value").rows() == [
        ("06", 100.5),
        ("59", 0.5),
    ]


def test_samples_to_hubverse_sorts_by_output_key() -> None:
    """Sample output order is deterministic regardless of source row order."""
    result = ft.samples_to_hubverse(
        _valid_samples().reverse(),
        reference_date=datetime.date(2026, 1, 1),
        target_map={("admissions", "daily"): "inc admissions"},
        location_map={"US": "US"},
        horizon_unit="days",
    )

    assert result.select("horizon", "output_type_id").rows() == [
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
    ]


def test_samples_to_hubverse_quantiles_sorts_by_output_key() -> None:
    """Quantile output is sorted by task columns and quantile level."""
    result = ft.samples_to_hubverse_quantiles(
        _valid_samples().reverse(),
        reference_date=datetime.date(2026, 1, 1),
        target_map={("admissions", "daily"): "inc admissions"},
        location_map={"US": "US"},
        horizon_unit="days",
        quantile_levels=[0.75, 0.25],
    )

    assert result.select("horizon", "output_type_id", "value").rows() == [
        (0, 0.25, 1.5),
        (0, 0.75, 2.5),
        (1, 0.25, 2.5),
        (1, 0.75, 3.5),
    ]


def test_public_converters_reject_target_mapping_collisions() -> None:
    """Distinct source series cannot map onto the same output keys."""
    reference_date = datetime.date(2026, 1, 1)
    dates = [reference_date, reference_date + datetime.timedelta(days=1)]
    samples = pl.concat(
        [
            _series_samples(dates),
            _series_samples(dates, variable="hospitalizations"),
        ]
    )
    target_map = {
        ("admissions", "daily"): "inc admissions",
        ("hospitalizations", "daily"): "inc admissions",
    }

    with pytest.raises(ValueError, match="duplicate Hubverse output keys"):
        ft.samples_to_hubverse(
            samples,
            reference_date=reference_date,
            target_map=target_map,
            location_map={"US": "US"},
            horizon_unit="days",
        )
    with pytest.raises(ValueError, match="duplicate Hubverse output keys"):
        ft.samples_to_hubverse_quantiles(
            samples,
            reference_date=reference_date,
            target_map=target_map,
            location_map={"US": "US"},
            horizon_unit="days",
            quantile_levels=[0.5],
        )


def test_public_converters_reject_location_mapping_collisions() -> None:
    """Distinct source locations cannot map onto the same output keys."""
    reference_date = datetime.date(2026, 1, 1)
    samples = pl.concat(
        [
            _series_samples([reference_date]),
            _series_samples([reference_date], location="CA"),
        ]
    )

    with pytest.raises(ValueError, match="duplicate Hubverse output keys"):
        ft.samples_to_hubverse(
            samples,
            reference_date=reference_date,
            target_map={("admissions", "daily"): "inc admissions"},
            location_map={"US": "US", "CA": "US"},
            horizon_unit="days",
        )
    with pytest.raises(ValueError, match="duplicate Hubverse output keys"):
        ft.samples_to_hubverse_quantiles(
            samples,
            reference_date=reference_date,
            target_map={("admissions", "daily"): "inc admissions"},
            location_map={"US": "US", "CA": "US"},
            horizon_unit="days",
            quantile_levels=[0.5],
        )


def test_finalize_output_rejects_duplicate_keys() -> None:
    """Final output validation catches repeated seven-column keys."""
    output = ft.samples_to_hubverse(
        _valid_samples(),
        reference_date=datetime.date(2026, 1, 1),
        target_map={("admissions", "daily"): "inc admissions"},
        location_map={"US": "US"},
        horizon_unit="days",
    )
    output_with_duplicate = pl.concat([output, output.head(1)])

    with pytest.raises(ValueError, match="duplicate output keys"):
        _finalize_output(
            output_with_duplicate,
            output_type_id_dtype=pl.Int64,
        )


def test_sample_converter_handles_representative_multitask_table() -> None:
    """Sample conversion preserves every row across representative tasks."""
    reference_date = datetime.date(2026, 1, 1)

    result = ft.samples_to_hubverse(
        _representative_samples().reverse(),
        reference_date=reference_date,
        target_map={
            ("admissions", "daily"): "inc admissions",
            ("admissions", "weekly"): "wk inc admissions",
        },
        location_map={"US": "59", "CA": "06"},
        horizon_unit="days",
    )

    assert result.height == 16
    assert result.select(
        "target", "horizon", "target_end_date", "location"
    ).unique().sort("target", "horizon", "location").rows() == [
        ("inc admissions", 0, reference_date, "06"),
        ("inc admissions", 0, reference_date, "59"),
        (
            "inc admissions",
            1,
            reference_date + datetime.timedelta(days=1),
            "06",
        ),
        (
            "inc admissions",
            1,
            reference_date + datetime.timedelta(days=1),
            "59",
        ),
        (
            "wk inc admissions",
            2,
            reference_date + datetime.timedelta(days=2),
            "06",
        ),
        (
            "wk inc admissions",
            2,
            reference_date + datetime.timedelta(days=2),
            "59",
        ),
        (
            "wk inc admissions",
            9,
            reference_date + datetime.timedelta(days=9),
            "06",
        ),
        (
            "wk inc admissions",
            9,
            reference_date + datetime.timedelta(days=9),
            "59",
        ),
    ]
    assert (
        result.group_by(
            "target",
            "horizon",
            "target_end_date",
            "location",
        )
        .agg(pl.col("output_type_id").sort())
        .get_column("output_type_id")
        .to_list()
        == [[0, 1]] * 8
    )
    assert result.get_column("value").sort().to_list() == [
        0.0,
        2.0,
        4.0,
        6.0,
        20.0,
        22.0,
        24.0,
        26.0,
        100.0,
        102.0,
        104.0,
        106.0,
        120.0,
        122.0,
        124.0,
        126.0,
    ]


def test_quantile_converter_handles_representative_multitask_table() -> None:
    """Quantile conversion aggregates each representative task separately."""
    result = ft.samples_to_hubverse_quantiles(
        _representative_samples().reverse(),
        reference_date=datetime.date(2026, 1, 1),
        target_map={
            ("admissions", "daily"): "inc admissions",
            ("admissions", "weekly"): "wk inc admissions",
        },
        location_map={"US": "59", "CA": "06"},
        horizon_unit="days",
        quantile_levels=[0.75, 0.25, 0.5],
    )

    assert result.height == 24
    assert (
        result.group_by(
            "target",
            "horizon",
            "target_end_date",
            "location",
        )
        .agg(pl.col("output_type_id").sort())
        .get_column("output_type_id")
        .to_list()
        == [[0.25, 0.5, 0.75]] * 8
    )
    assert result.filter(
        (pl.col("target") == "wk inc admissions")
        & (pl.col("horizon") == 9)
        & (pl.col("location") == "06")
    ).select("output_type_id", "value").rows() == [
        (0.25, 124.5),
        (0.5, 125.0),
        (0.75, 125.5),
    ]


def test_sample_converter_handles_generated_pyrenew_output() -> None:
    """Sample conversion accepts a production-shaped PyRenew HEW artifact."""
    samples = pl.read_parquet(_TEST_DATA_DIR / "pyrenew_he_samples_3_draws.parquet")

    result = ft.samples_to_hubverse(
        samples,
        reference_date=datetime.date(2025, 1, 28),
        target_map={
            ("ed_visits", "daily"): "inc COVID-19 ed visits",
            ("hospital_admissions", "weekly"): "wk inc COVID-19 hosp",
        },
        location_map={"CA": "06"},
        horizon_unit="days",
    )

    assert samples.shape == (48, 6)
    assert result.shape == (48, 8)
    assert result.get_column("output_type_id").unique().sort().to_list() == [0, 1, 2]
    assert result.select("target", "horizon").unique().group_by("target").agg(
        pl.col("horizon").sort()
    ).sort("target").rows() == [
        ("inc COVID-19 ed visits", list(range(1, 15))),
        ("wk inc COVID-19 hosp", [4, 11]),
    ]
    assert result.filter(
        (pl.col("target") == "inc COVID-19 ed visits") & (pl.col("horizon") == 1)
    ).select("output_type_id", "value").rows() == [
        (0, 455.0),
        (1, 587.0),
        (2, 443.0),
    ]


def test_quantile_converter_handles_generated_pyrenew_output() -> None:
    """Quantile conversion accepts a production-shaped PyRenew HEW artifact."""
    samples = pl.read_parquet(_TEST_DATA_DIR / "pyrenew_he_samples_3_draws.parquet")

    result = ft.samples_to_hubverse_quantiles(
        samples,
        reference_date=datetime.date(2025, 1, 28),
        target_map={
            ("ed_visits", "daily"): "inc COVID-19 ed visits",
            ("hospital_admissions", "weekly"): "wk inc COVID-19 hosp",
        },
        location_map={"CA": "06"},
        horizon_unit="days",
        quantile_levels=[0.25, 0.5, 0.75],
    )

    assert result.shape == (48, 8)
    assert result.filter(
        (pl.col("target") == "wk inc COVID-19 hosp") & (pl.col("horizon") == 4)
    ).select("output_type_id", "value").rows() == [
        (0.25, 550.5),
        (0.5, 897.0),
        (0.75, 2429.5),
    ]


def test_sample_conversion_matches_r_golden_output() -> None:
    """Sample conversion matches the static R reference output."""
    actual = ft.samples_to_hubverse(
        _r_parity_samples(),
        reference_date=datetime.date(2025, 1, 28),
        target_map={
            ("observed_ed_visits", "daily"): "inc covid ed visits",
            ("observed_hospital_admissions", "weekly"): "wk inc covid hosp",
        },
        location_map={"CA": "CA"},
        horizon_unit="days",
        draw_col=".draw",
        location_col="geo_value",
        variable_col=".variable",
        value_col=".value",
    )
    expected = _r_parity_expected(
        "r_hubverse_sample_expected.csv",
        output_type="sample",
    )

    assert_frame_equal(actual, expected)


def test_quantile_conversion_matches_r_golden_output() -> None:
    """Quantile conversion matches static R type-7 quantiles."""
    actual = ft.samples_to_hubverse_quantiles(
        _r_parity_samples(),
        reference_date=datetime.date(2025, 1, 28),
        target_map={
            ("observed_ed_visits", "daily"): "inc covid ed visits",
            ("observed_hospital_admissions", "weekly"): "wk inc covid hosp",
        },
        location_map={"CA": "CA"},
        horizon_unit="days",
        quantile_levels=_R_HUB_QUANTILES,
        draw_col=".draw",
        location_col="geo_value",
        variable_col=".variable",
        value_col=".value",
    )
    expected = _r_parity_expected(
        "r_hubverse_quantile_expected.csv",
        output_type="quantile",
    )

    assert_frame_equal(
        actual,
        expected,
        check_exact=False,
        abs_tol=1e-12,
        rel_tol=1e-12,
    )
