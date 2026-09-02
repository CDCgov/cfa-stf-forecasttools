"""Convert forecast samples to Hubverse model-output tables."""

import datetime
import math
from collections.abc import Mapping, Sequence
from numbers import Real
from typing import TYPE_CHECKING, Literal

import polars as pl

if TYPE_CHECKING:
    from polars._typing import PolarsDataType

_IDENTITY_COLUMNS = ("draw", "date", "location", "variable", "resolution")
_TASK_COLUMNS = (
    "reference_date",
    "target",
    "horizon",
    "target_end_date",
    "location",
)
_OUTPUT_KEY_COLUMNS = (*_TASK_COLUMNS, "output_type", "output_type_id")


def _normalize_samples(
    samples: pl.DataFrame,
    *,
    draw_col: str,
    date_col: str,
    location_col: str,
    variable_col: str,
    value_col: str,
    resolution_col: str,
) -> pl.DataFrame:
    """Validate source samples and return canonical column names and order."""
    if not isinstance(samples, pl.DataFrame):
        raise TypeError("samples must be a Polars DataFrame")

    source_columns = {
        "draw": draw_col,
        "date": date_col,
        "location": location_col,
        "variable": variable_col,
        "value": value_col,
        "resolution": resolution_col,
    }
    for parameter, column in source_columns.items():
        if not isinstance(column, str):
            raise TypeError(f"{parameter}_col must be a string")
        if not column.strip():
            raise ValueError(f"{parameter}_col must be a nonempty string")
    if len(set(source_columns.values())) != len(source_columns):
        raise ValueError("source column names must be distinct")

    expected = set(source_columns.values())
    actual = set(samples.columns)
    if actual != expected:
        details = []
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        if missing:
            details.append(f"missing: {missing}")
        if extra:
            details.append(f"extra: {extra}")
        raise ValueError(
            "samples must contain exactly the configured source columns "
            f"({'; '.join(details)})"
        )

    normalized = samples.select(
        pl.col(source).alias(canonical) for canonical, source in source_columns.items()
    )
    if normalized.is_empty():
        raise ValueError("samples must not be empty")

    null_counts = normalized.null_count().row(0, named=True)
    columns_with_nulls = sorted(
        column for column, count in null_counts.items() if count > 0
    )
    if columns_with_nulls:
        raise ValueError(
            "samples contain null values in required columns: "
            + ", ".join(columns_with_nulls)
        )

    schema = normalized.schema
    if not schema["draw"].is_integer():
        raise TypeError("draw column must have an integer type")
    try:
        normalized = normalized.with_columns(pl.col("draw").cast(pl.Int64))
    except pl.exceptions.InvalidOperationError as error:
        raise ValueError("draw values must fit in a signed 64-bit integer") from error
    if normalized.select((pl.col("draw") < 0).any()).item():
        raise ValueError("draw values must be nonnegative")

    if schema["date"] != pl.Date:
        raise TypeError("date column must have Polars Date type")
    for column in ("location", "variable", "resolution"):
        if schema[column] != pl.String:
            raise TypeError(f"{column} column must have Polars String type")
    for column in ("location", "variable"):
        if normalized.select(pl.col(column).str.strip_chars().eq("").any()).item():
            raise ValueError(f"{column} values must be nonempty strings")

    unsupported_resolutions = sorted(
        set(normalized.get_column("resolution").unique()) - {"daily", "weekly"}
    )
    if unsupported_resolutions:
        raise ValueError(
            "resolution contains unsupported values: "
            + ", ".join(repr(value) for value in unsupported_resolutions)
        )

    if not schema["value"].is_numeric():
        raise TypeError("value column must have a numeric type")
    if not normalized.select(pl.col("value").is_finite().all()).item():
        raise ValueError("value column must contain only finite values")

    duplicates = normalized.group_by(_IDENTITY_COLUMNS).len().filter(pl.col("len") > 1)
    if not duplicates.is_empty():
        raise ValueError(
            "samples contain duplicate "
            "(draw, date, location, variable, resolution) rows"
        )

    return normalized


def _validate_date_spacing(
    dates: list[datetime.date],
    *,
    resolution: str,
) -> None:
    """Require consecutive daily dates or seven-day weekly spacing."""
    expected_days = 1 if resolution == "daily" else 7
    if any(
        (right - left).days != expected_days for left, right in zip(dates, dates[1:])
    ):
        spacing = "consecutive" if resolution == "daily" else "seven days apart"
        raise ValueError(f"{resolution} forecast dates must be {spacing}")


def _validate_sample_coverage(samples: pl.DataFrame) -> None:
    """Validate coherent draw identities and dates across forecast series."""
    expected_draws: list[int] | None = None
    for series in samples.partition_by(
        ["location", "variable", "resolution"],
        maintain_order=False,
    ):
        draws: list[int] = sorted(series.get_column("draw").unique().to_list())
        if expected_draws is None:
            expected_draws = draws
        elif draws != expected_draws:
            raise ValueError(
                "all forecast series must contain the same draw identities"
            )

        expected_dates: list[datetime.date] | None = None
        for draw_samples in series.partition_by("draw", maintain_order=False):
            dates: list[datetime.date] = sorted(
                draw_samples.get_column("date").to_list()
            )
            if expected_dates is None:
                expected_dates = dates
            elif dates != expected_dates:
                raise ValueError(
                    "all draws within a forecast series must contain the same dates"
                )

        if expected_dates is None:
            raise ValueError("forecast series must not be empty")
        resolution: str = series.get_column("resolution").item(0)
        _validate_date_spacing(expected_dates, resolution=resolution)


def _normalize_sample_ids(
    samples: pl.DataFrame,
    *,
    source_draw_id_base: Literal[0, 1],
) -> pl.DataFrame:
    """Validate source draw IDs and return one-based Hubverse sample IDs."""
    if isinstance(source_draw_id_base, bool) or not isinstance(
        source_draw_id_base, int
    ):
        raise TypeError("source_draw_id_base must be an integer")
    if source_draw_id_base not in (0, 1):
        raise ValueError("source_draw_id_base must be either 0 or 1")

    draw_ids: list[int] = sorted(samples.get_column("draw").unique().to_list())
    expected_draw_ids = list(
        range(source_draw_id_base, source_draw_id_base + len(draw_ids))
    )
    if draw_ids != expected_draw_ids:
        raise ValueError(
            "draw values must be contiguous and start at source_draw_id_base"
        )

    return samples.with_columns(
        (pl.col("draw") - source_draw_id_base + 1).alias("draw")
    )


def _normalize_reference_date(reference_date: datetime.date) -> datetime.date:
    """Validate a reference date and remove any time component."""
    if not isinstance(reference_date, datetime.date):
        raise TypeError("reference_date must be a datetime.date")
    if isinstance(reference_date, datetime.datetime):
        return reference_date.date()
    return reference_date


def _validate_horizon_unit(
    horizon_unit: Literal["days", "weeks"],
) -> None:
    """Require a supported horizon unit."""
    if not isinstance(horizon_unit, str):
        raise TypeError("horizon_unit must be a string")
    if horizon_unit not in ("days", "weeks"):
        raise ValueError('horizon_unit must be either "days" or "weeks"')


def _normalize_mappings(
    samples: pl.DataFrame,
    *,
    target_map: Mapping[tuple[str, str], str],
    location_map: Mapping[str, str],
) -> tuple[dict[tuple[str, str], str], dict[str, str]]:
    """Validate mappings and return plain dictionaries."""
    if not isinstance(target_map, Mapping):
        raise TypeError("target_map must be a mapping")
    if not isinstance(location_map, Mapping):
        raise TypeError("location_map must be a mapping")

    normalized_targets: dict[tuple[str, str], str] = {}
    for target_key, target_value in target_map.items():
        if (
            not isinstance(target_key, tuple)
            or len(target_key) != 2
            or not all(isinstance(part, str) and part.strip() for part in target_key)
        ):
            raise TypeError(
                "target_map keys must be nonempty (variable, resolution) string tuples"
            )
        if not isinstance(target_value, str):
            raise TypeError("target_map values must be strings")
        if not target_value.strip():
            raise ValueError("target_map values must be nonempty strings")
        normalized_targets[target_key] = target_value

    normalized_locations: dict[str, str] = {}
    for location_key, location_value in location_map.items():
        if not isinstance(location_key, str):
            raise TypeError("location_map keys must be strings")
        if not location_key.strip():
            raise ValueError("location_map keys must be nonempty strings")
        if not isinstance(location_value, str):
            raise TypeError("location_map values must be strings")
        if not location_value.strip():
            raise ValueError("location_map values must be nonempty strings")
        normalized_locations[location_key] = location_value

    observed_targets = {
        (variable, resolution)
        for variable, resolution in samples.select("variable", "resolution").iter_rows()
    }
    observed_locations = set(samples.get_column("location"))
    missing_targets = sorted(observed_targets - normalized_targets.keys())
    missing_locations = sorted(observed_locations - normalized_locations.keys())
    if missing_targets or missing_locations:
        details = []
        if missing_targets:
            details.append(f"target_map missing keys: {missing_targets}")
        if missing_locations:
            details.append(f"location_map missing keys: {missing_locations}")
        raise ValueError("; ".join(details))

    return normalized_targets, normalized_locations


def _normalize_quantile_levels(
    quantile_levels: Sequence[float],
) -> list[float]:
    """Validate quantile probabilities and return them sorted."""
    if isinstance(quantile_levels, (str, bytes)) or not isinstance(
        quantile_levels, Sequence
    ):
        raise TypeError("quantile_levels must be a sequence of numbers")
    if not quantile_levels:
        raise ValueError("quantile_levels must not be empty")

    normalized: list[float] = []
    for level in quantile_levels:
        if isinstance(level, bool) or not isinstance(level, Real):
            raise TypeError("quantile levels must be numeric and not boolean")
        normalized_level = float(level)
        if not math.isfinite(normalized_level):
            raise ValueError("quantile levels must be finite")
        if not 0.0 <= normalized_level <= 1.0:
            raise ValueError("quantile levels must be between 0 and 1")
        normalized.append(normalized_level)

    if len(set(normalized)) != len(normalized):
        raise ValueError("quantile levels must not contain duplicates")
    return sorted(normalized)


def _build_task_rows(
    samples: pl.DataFrame,
    *,
    reference_date: datetime.date,
    target_map: Mapping[tuple[str, str], str],
    location_map: Mapping[str, str],
    horizon_unit: Literal["days", "weeks"],
) -> pl.DataFrame:
    """Add mapped Hubverse task columns to normalized forecast samples."""
    target_items = sorted(target_map.items())
    target_lookup = pl.DataFrame(
        {
            "variable": [key[0] for key, _ in target_items],
            "resolution": [key[1] for key, _ in target_items],
            "target": [target for _, target in target_items],
        },
        schema={
            "variable": pl.String,
            "resolution": pl.String,
            "target": pl.String,
        },
    )
    location_items = sorted(location_map.items())
    location_lookup = pl.DataFrame(
        {
            "location": [location for location, _ in location_items],
            "_hubverse_location": [mapped for _, mapped in location_items],
        },
        schema={
            "location": pl.String,
            "_hubverse_location": pl.String,
        },
    )

    task_rows = (
        samples.join(
            target_lookup,
            on=["variable", "resolution"],
            how="left",
            validate="m:1",
        )
        .join(
            location_lookup,
            on="location",
            how="left",
            validate="m:1",
        )
        .with_columns(
            pl.lit(reference_date, dtype=pl.Date).alias("reference_date"),
            (pl.col("date") - pl.lit(reference_date, dtype=pl.Date))
            .dt.total_days()
            .cast(pl.Int64)
            .alias("_day_horizon"),
        )
    )

    if horizon_unit == "weeks":
        has_fractional_week = task_rows.select(
            (pl.col("_day_horizon") % 7 != 0).any()
        ).item()
        if has_fractional_week:
            raise ValueError(
                "weekly horizons require every target date to differ from "
                "reference_date by a whole number of weeks"
            )
        horizon = (pl.col("_day_horizon") // 7).alias("horizon")
    else:
        horizon = pl.col("_day_horizon").alias("horizon")

    return task_rows.with_columns(horizon).select(
        "reference_date",
        "target",
        "horizon",
        pl.col("date").alias("target_end_date"),
        pl.col("_hubverse_location").alias("location"),
        "draw",
        "value",
    )


def _validate_task_row_keys(task_rows: pl.DataFrame) -> None:
    """Reject source mappings that collapse distinct rows onto one task."""
    task_draw_columns = (*_TASK_COLUMNS, "draw")
    duplicates = task_rows.group_by(task_draw_columns).len().filter(pl.col("len") > 1)
    if not duplicates.is_empty():
        raise ValueError(
            "target_map and location_map create duplicate Hubverse output keys"
        )


def _build_sample_output(task_rows: pl.DataFrame) -> pl.DataFrame:
    """Format mapped task rows as Hubverse sample output."""
    return task_rows.select(
        *_TASK_COLUMNS,
        pl.lit("sample").alias("output_type"),
        pl.col("draw").cast(pl.Int64).alias("output_type_id"),
        pl.col("value").cast(pl.Float64),
    )


def _build_quantile_output(
    task_rows: pl.DataFrame,
    *,
    quantile_levels: Sequence[float],
) -> pl.DataFrame:
    """Aggregate mapped task rows into Hubverse quantile output."""
    quantile_rows = pl.concat(
        [
            task_rows.group_by(_TASK_COLUMNS)
            .agg(
                pl.col("value").quantile(level, interpolation="linear").cast(pl.Float64)
            )
            .with_columns(
                pl.lit("quantile").alias("output_type"),
                pl.lit(level, dtype=pl.Float64).alias("output_type_id"),
            )
            for level in quantile_levels
        ]
    )
    return quantile_rows.select(
        *_TASK_COLUMNS,
        "output_type",
        "output_type_id",
        "value",
    )


def _finalize_output(
    output: pl.DataFrame,
    *,
    output_type_id_dtype: "PolarsDataType",
) -> pl.DataFrame:
    """Cast, validate, and deterministically sort Hubverse output rows."""
    schema: dict[str, PolarsDataType] = {
        "reference_date": pl.Date,
        "target": pl.String,
        "horizon": pl.Int64,
        "target_end_date": pl.Date,
        "location": pl.String,
        "output_type": pl.String,
        "output_type_id": output_type_id_dtype,
        "value": pl.Float64,
    }
    finalized = output.select(
        pl.col(column).cast(dtype) for column, dtype in schema.items()
    )
    duplicates = finalized.group_by(_OUTPUT_KEY_COLUMNS).len().filter(pl.col("len") > 1)
    if not duplicates.is_empty():
        raise ValueError("Hubverse output contains duplicate output keys")
    return finalized.sort(_OUTPUT_KEY_COLUMNS)


def samples_to_hubverse(
    samples: pl.DataFrame,
    *,
    reference_date: datetime.date,
    target_map: Mapping[tuple[str, str], str],
    location_map: Mapping[str, str],
    horizon_unit: Literal["days", "weeks"],
    source_draw_id_base: Literal[0, 1] = 0,
    draw_col: str = "draw",
    date_col: str = "date",
    location_col: str = "location",
    variable_col: str = "variable",
    value_col: str = "value",
    resolution_col: str = "resolution",
) -> pl.DataFrame:
    """Convert forecast draws to Hubverse sample rows.

    Parameters
    ----------
    samples
        Forecast draws with columns for draw, date, location, variable, value,
        and resolution.
    reference_date
        Date from which forecast horizons are calculated.
    target_map
        Mapping from (variable, resolution) pairs to Hubverse targets.
    location_map
        Mapping from source locations to Hubverse locations.
    horizon_unit
        Unit used to calculate horizons. Must be "days" or "weeks". Week
        horizons require every target date to be an exact multiple of seven
        days from the reference date.
    source_draw_id_base
        First source draw ID. Source IDs must be contiguous and start at zero
        or one. Hubverse sample IDs are always returned as one through N;
        zero-based source IDs are shifted by one.
    draw_col
        Name of the source draw column.
    date_col
        Name of the source forecast-date column.
    location_col
        Name of the source location column.
    variable_col
        Name of the source variable column.
    value_col
        Name of the source forecast-value column.
    resolution_col
        Name of the source resolution column.

    Returns
    -------
    pl.DataFrame
        Hubverse sample rows with task columns followed by output_type,
        output_type_id, and value.

    Raises
    ------
    TypeError
        If an input has an unsupported type.
    ValueError
        If the samples or caller configuration are invalid.
    """
    normalized = _normalize_samples(
        samples,
        draw_col=draw_col,
        date_col=date_col,
        location_col=location_col,
        variable_col=variable_col,
        value_col=value_col,
        resolution_col=resolution_col,
    )
    _validate_sample_coverage(normalized)
    normalized = _normalize_sample_ids(
        normalized,
        source_draw_id_base=source_draw_id_base,
    )
    normalized_reference_date = _normalize_reference_date(reference_date)
    _validate_horizon_unit(horizon_unit)
    normalized_targets, normalized_locations = _normalize_mappings(
        normalized,
        target_map=target_map,
        location_map=location_map,
    )
    task_rows = _build_task_rows(
        normalized,
        reference_date=normalized_reference_date,
        target_map=normalized_targets,
        location_map=normalized_locations,
        horizon_unit=horizon_unit,
    )
    _validate_task_row_keys(task_rows)
    return _finalize_output(
        _build_sample_output(task_rows),
        output_type_id_dtype=pl.Int64,
    )


def samples_to_hubverse_quantiles(
    samples: pl.DataFrame,
    *,
    reference_date: datetime.date,
    target_map: Mapping[tuple[str, str], str],
    location_map: Mapping[str, str],
    horizon_unit: Literal["days", "weeks"],
    quantile_levels: Sequence[float],
    draw_col: str = "draw",
    date_col: str = "date",
    location_col: str = "location",
    variable_col: str = "variable",
    value_col: str = "value",
    resolution_col: str = "resolution",
) -> pl.DataFrame:
    """Convert forecast draws to Hubverse quantile rows.

    Parameters
    ----------
    samples
        Forecast draws with columns for draw, date, location, variable, value,
        and resolution.
    reference_date
        Date from which forecast horizons are calculated.
    target_map
        Mapping from (variable, resolution) pairs to Hubverse targets.
    location_map
        Mapping from source locations to Hubverse locations.
    horizon_unit
        Unit used to calculate horizons. Must be "days" or "weeks". Week
        horizons require every target date to be an exact multiple of seven
        days from the reference date.
    quantile_levels
        Quantile probabilities between zero and one. Quantiles use linear
        interpolation, equivalent to R type 7.
    draw_col
        Name of the source draw column.
    date_col
        Name of the source forecast-date column.
    location_col
        Name of the source location column.
    variable_col
        Name of the source variable column.
    value_col
        Name of the source forecast-value column.
    resolution_col
        Name of the source resolution column.

    Returns
    -------
    pl.DataFrame
        Hubverse quantile rows with task columns followed by output_type,
        output_type_id, and value.

    Raises
    ------
    TypeError
        If an input has an unsupported type.
    ValueError
        If the samples, quantile levels, or caller configuration are invalid.
    """
    normalized = _normalize_samples(
        samples,
        draw_col=draw_col,
        date_col=date_col,
        location_col=location_col,
        variable_col=variable_col,
        value_col=value_col,
        resolution_col=resolution_col,
    )
    _validate_sample_coverage(normalized)
    normalized_reference_date = _normalize_reference_date(reference_date)
    _validate_horizon_unit(horizon_unit)
    normalized_targets, normalized_locations = _normalize_mappings(
        normalized,
        target_map=target_map,
        location_map=location_map,
    )
    normalized_quantile_levels = _normalize_quantile_levels(quantile_levels)
    task_rows = _build_task_rows(
        normalized,
        reference_date=normalized_reference_date,
        target_map=normalized_targets,
        location_map=normalized_locations,
        horizon_unit=horizon_unit,
    )
    _validate_task_row_keys(task_rows)
    return _finalize_output(
        _build_quantile_output(
            task_rows,
            quantile_levels=normalized_quantile_levels,
        ),
        output_type_id_dtype=pl.Float64,
    )
