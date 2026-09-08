from collections.abc import Sequence
from typing import Literal

import numpy as np
import polars as pl
from numpy.typing import NDArray

_OUTPUT_COLUMNS = frozenset({"scale", "crps"})


def _convert_to_float64(values: pl.Series) -> NDArray[np.float64]:
    """Convert a numeric series to Float64 without changing its values."""
    converted = values.cast(pl.Float64)
    converted_values = converted.to_numpy()
    if not np.isfinite(converted_values).all():
        raise ValueError(f"{values.name!r} must contain only finite values")

    restored = converted.cast(values.dtype, strict=False)
    if restored.null_count() > 0 or not restored.equals(values):
        raise ValueError(
            f"{values.name!r} contains values that cannot be represented safely "
            "as Float64"
        )

    return converted_values


def _validate_column_names(
    forecasts: pl.DataFrame,
    forecast_unit: Sequence[str],
    sample_id_col: str,
    prediction_col: str,
    observation_col: str,
) -> list[str]:
    """Validate column-name arguments and return the forecast-unit columns."""
    if isinstance(forecast_unit, (str, bytes)) or not isinstance(
        forecast_unit, Sequence
    ):
        raise TypeError("forecast_unit must be a sequence of column-name strings")

    unit_columns = list(forecast_unit)
    if not unit_columns:
        raise ValueError("forecast_unit must contain at least one column name")
    if not all(isinstance(column, str) for column in unit_columns):
        raise TypeError("forecast_unit must contain only column-name strings")
    if any(not column for column in unit_columns):
        raise ValueError("forecast_unit column names must be nonempty")
    if len(unit_columns) != len(set(unit_columns)):
        raise ValueError("forecast_unit column names must be distinct")

    value_columns = {
        "sample_id_col": sample_id_col,
        "prediction_col": prediction_col,
        "observation_col": observation_col,
    }
    for argument, column in value_columns.items():
        if not isinstance(column, str):
            raise TypeError(f"{argument} must be a column-name string")
        if not column:
            raise ValueError(f"{argument} must be nonempty")

    if len(set(value_columns.values())) != len(value_columns):
        raise ValueError(
            "sample_id_col, prediction_col, and observation_col must be distinct"
        )

    overlapping_columns = set(unit_columns).intersection(value_columns.values())
    if overlapping_columns:
        overlap = ", ".join(sorted(overlapping_columns))
        raise ValueError(
            f"forecast_unit must not contain sample or value column(s): {overlap}"
        )

    reserved_columns = set(unit_columns).intersection(_OUTPUT_COLUMNS)
    if reserved_columns:
        reserved = ", ".join(sorted(reserved_columns))
        raise ValueError(
            f"forecast_unit must not contain reserved output column(s): {reserved}"
        )

    requested_columns = [*unit_columns, *value_columns.values()]
    missing_columns = [
        column for column in requested_columns if column not in forecasts.columns
    ]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"forecasts is missing required column(s): {missing}")

    return unit_columns


def _validate_values(
    forecasts: pl.DataFrame,
    unit_columns: list[str],
    scale: Literal["natural", "log1p"],
    sample_id_col: str,
    prediction_col: str,
    observation_col: str,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Validate sample and value columns and return exact Float64 arrays."""
    for column in (prediction_col, observation_col):
        if not forecasts.schema[column].is_numeric():
            raise TypeError(f"{column!r} must be a numeric, non-boolean column")
        if forecasts.get_column(column).null_count() > 0:
            raise ValueError(f"{column!r} must not contain null values")

    if forecasts.get_column(sample_id_col).null_count() > 0:
        raise ValueError(f"{sample_id_col!r} must not contain null values")

    prediction_values = _convert_to_float64(forecasts.get_column(prediction_col))
    observation_values = _convert_to_float64(forecasts.get_column(observation_col))

    if scale == "log1p" and (
        np.any(prediction_values <= -1.0) or np.any(observation_values <= -1.0)
    ):
        raise ValueError(
            "prediction and observation values must be greater than -1 for log1p scoring"
        )

    identity_columns = forecasts.select([*unit_columns, sample_id_col])
    if identity_columns.is_duplicated().any():
        raise ValueError("sample identifiers must be unique within each forecast unit")

    observation_counts = forecasts.group_by(unit_columns).agg(
        pl.col(observation_col).n_unique()
    )
    if observation_counts.filter(pl.col(observation_col) != 1).height > 0:
        raise ValueError("each forecast unit must have exactly one observation")

    return prediction_values, observation_values


def _empirical_crps(samples: NDArray[np.float64], observation: float) -> float:
    """Calculate empirical CRPS for one observation and its predictive samples."""
    sorted_samples = np.sort(samples)
    sample_count = sorted_samples.size
    probability_midpoints = (
        np.arange(sample_count, dtype=np.float64) + 0.5
    ) / sample_count
    above_observation = (observation < sorted_samples).astype(np.float64)
    score = 2.0 * np.mean(
        (above_observation - probability_midpoints) * (sorted_samples - observation)
    )
    return float(score)


def score_sample_crps(
    forecasts: pl.DataFrame,
    *,
    forecast_unit: Sequence[str],
    scale: Literal["natural", "log1p"],
    sample_id_col: str = "sample_id",
    prediction_col: str = "predicted",
    observation_col: str = "observed",
) -> pl.DataFrame:
    """Score predictive samples independently with empirical CRPS.

    ``forecasts`` must contain one row per predictive sample. The columns in
    ``forecast_unit`` jointly identify one predictive distribution, and the
    observation must be repeated consistently across that unit's sample rows.
    Additional columns are ignored unless included in ``forecast_unit``.

    Select ``scale="natural"`` to score values as supplied or ``scale="log1p"``
    to transform predictions and observations with ``numpy.log1p`` before
    scoring. Values at or below -1 are invalid on the log1p scale, and scores
    calculated on different scales should not be combined or directly compared.

    The result has one row per forecast unit, sorted by the forecast-unit
    columns. It contains those columns in the requested order, followed by the
    string column ``scale`` and the Float64 column ``crps``. Lower CRPS values
    are better, and zero is optimal. This function does not aggregate scores
    across forecast units.

    Raises
    ------
    TypeError
        If the input or named value columns have invalid types.
    ValueError
        If required data are missing, duplicated, inconsistent, non-finite,
        not safely representable as Float64, or outside the selected scale's
        domain.
    """
    if not isinstance(forecasts, pl.DataFrame):
        raise TypeError("forecasts must be a Polars DataFrame")
    if forecasts.is_empty():
        raise ValueError("forecasts must contain at least one row")
    if scale not in ("natural", "log1p"):
        raise ValueError("scale must be either 'natural' or 'log1p'")

    unit_columns = _validate_column_names(
        forecasts,
        forecast_unit,
        sample_id_col,
        prediction_col,
        observation_col,
    )
    prediction_values, observation_values = _validate_values(
        forecasts,
        unit_columns,
        scale,
        sample_id_col,
        prediction_col,
        observation_col,
    )

    if scale == "log1p":
        prediction_values = np.log1p(prediction_values)
        observation_values = np.log1p(observation_values)

    prepared = forecasts.with_columns(
        pl.Series(prediction_col, prediction_values, dtype=pl.Float64),
        pl.Series(observation_col, observation_values, dtype=pl.Float64),
    )

    prediction_group_col = "__forecasttools_predictions"
    while prediction_group_col in unit_columns:
        prediction_group_col = f"_{prediction_group_col}"
    observation_group_col = "__forecasttools_observation"
    while observation_group_col in [*unit_columns, prediction_group_col]:
        observation_group_col = f"_{observation_group_col}"

    grouped = (
        prepared.group_by(unit_columns)
        .agg(
            pl.col(prediction_col).alias(prediction_group_col),
            pl.col(observation_col).first().alias(observation_group_col),
        )
        .sort(unit_columns)
    )
    scores = [
        _empirical_crps(np.asarray(samples, dtype=np.float64), float(observation))
        for samples, observation in grouped.select(
            prediction_group_col, observation_group_col
        ).iter_rows()
    ]

    return grouped.select(unit_columns).with_columns(
        pl.lit(scale, dtype=pl.String).alias("scale"),
        pl.Series("crps", scores, dtype=pl.Float64),
    )
