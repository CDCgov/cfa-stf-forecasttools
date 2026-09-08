from collections.abc import Callable, Sequence

import numpy as np
import polars as pl
import scoringrules as sr
from numpy.typing import NDArray

_OUTPUT_COLUMNS = frozenset({"scale", "crps"})
_ValueTransform = Callable[[NDArray[np.float64]], NDArray[np.float64]]


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

    identity_columns = forecasts.select([*unit_columns, sample_id_col])
    if identity_columns.is_duplicated().any():
        raise ValueError("sample identifiers must be unique within each forecast unit")

    observation_counts = forecasts.group_by(unit_columns).agg(
        pl.col(observation_col).n_unique()
    )
    if observation_counts.filter(pl.col(observation_col) != 1).height > 0:
        raise ValueError("each forecast unit must have exactly one observation")

    return prediction_values, observation_values


def _apply_transform(
    values: NDArray[np.float64],
    transform: _ValueTransform | None,
    column_name: str,
) -> NDArray[np.float64]:
    """Apply a scale transform and validate its output."""
    if transform is None:
        return values

    with np.errstate(all="ignore"):
        transformed = np.asarray(transform(values))

    if transformed.shape != values.shape:
        raise ValueError("transform must preserve the shape of its input")
    if not (
        np.issubdtype(transformed.dtype, np.integer)
        or np.issubdtype(transformed.dtype, np.floating)
    ):
        raise TypeError("transform must return real numeric values")

    converted = transformed.astype(np.float64, copy=False)
    if not np.isfinite(converted).all():
        raise ValueError(
            f"transform must return only finite values for {column_name!r}"
        )
    return converted


def _empirical_crps(samples: NDArray[np.float64], observation: float) -> float:
    """Calculate empirical CRPS for one observation and its predictive samples."""
    score = sr.crps_ensemble(
        observation,
        samples,
        estimator="qd",
        nan_policy="raise",
        backend="numpy",
    )
    return float(np.asarray(score, dtype=np.float64).item())


def score_sample_crps(
    forecasts: pl.DataFrame,
    *,
    forecast_unit: Sequence[str],
    transform: _ValueTransform | None,
    scale_name: str,
    sample_id_col: str = "sample_id",
    prediction_col: str = "predicted",
    observation_col: str = "observed",
) -> pl.DataFrame:
    """Score predictive samples independently with empirical CRPS.

    ``forecasts`` must contain one row per predictive sample. The columns in
    ``forecast_unit`` jointly identify one predictive distribution, and the
    observation must be repeated consistently across that unit's sample rows.
    Additional columns are ignored unless included in ``forecast_unit``.

    Set ``transform=None`` to score values as supplied. To score on another
    scale, provide a function that accepts and returns a same-shaped NumPy
    Float64 array. The transform is applied to both predictions and observations
    before scoring. ``scale_name`` labels the scores in the returned table;
    scores calculated on different scales should not be combined or directly
    compared.

    The result has one row per forecast unit, sorted by the forecast-unit
    columns. It contains those columns in the requested order, followed by the
    string column ``scale`` and the Float64 column ``crps``. Lower CRPS values
    are better, and zero is optimal. This function does not aggregate scores
    across forecast units.

    Raises
    ------
    TypeError
        If the input, transform, scale name, or named value columns have invalid
        types.
    ValueError
        If required data are missing, duplicated, inconsistent, non-finite,
        not safely representable as Float64, or invalid after transformation.
    """
    if not isinstance(forecasts, pl.DataFrame):
        raise TypeError("forecasts must be a Polars DataFrame")
    if forecasts.is_empty():
        raise ValueError("forecasts must contain at least one row")
    if transform is not None and not callable(transform):
        raise TypeError("transform must be callable or None")
    if not isinstance(scale_name, str):
        raise TypeError("scale_name must be a string")
    if not scale_name:
        raise ValueError("scale_name must be nonempty")

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
        sample_id_col,
        prediction_col,
        observation_col,
    )
    prediction_values = _apply_transform(prediction_values, transform, prediction_col)
    observation_values = _apply_transform(
        observation_values, transform, observation_col
    )

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
        pl.lit(scale_name, dtype=pl.String).alias("scale"),
        pl.Series("crps", scores, dtype=pl.Float64),
    )
