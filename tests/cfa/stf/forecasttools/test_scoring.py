import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import polars as pl
import polars.testing as plt
import pytest

import cfa.stf.forecasttools as ft

SCORING_FIXTURE_DIR = (
    Path(__file__).resolve().parent / "test_data" / "scoringutils_crps"
)
FORECAST_UNIT = [
    "model",
    "reference_date",
    "target_end_date",
    "location",
    "target",
    "horizon",
]


def _base_forecasts() -> pl.DataFrame:
    """Return a valid sample forecast table for validation tests."""
    return pl.DataFrame(
        {
            "unit": ["a", "a"],
            "sample_id": [0, 1],
            "predicted": [0.0, 2.0],
            "observed": [1.0, 1.0],
        }
    )


def test_one_sample_crps_equals_absolute_error() -> None:
    """A one-sample empirical distribution reduces to absolute error."""
    forecasts = pl.DataFrame(
        {
            "unit": ["a"],
            "sample_id": [7],
            "predicted": [4.0],
            "observed": [1.0],
        }
    )

    result = ft.score_sample_crps(forecasts, forecast_unit=["unit"], scale="natural")

    assert result.item(0, "crps") == 3.0


def test_multi_sample_crps_matches_hand_calculation() -> None:
    """The scorer matches a hand-calculated empirical CRPS."""
    result = ft.score_sample_crps(
        _base_forecasts(), forecast_unit=["unit"], scale="natural"
    )

    assert result.item(0, "crps") == 0.5


def test_crps_is_stable_under_large_common_offsets() -> None:
    """Large common offsets do not introduce cancellation error."""
    unshifted = pl.DataFrame(
        {
            "unit": ["a", "a", "a"],
            "sample_id": [0, 1, 2],
            "predicted": [0.0, 2.0, 4.0],
            "observed": [1.0, 1.0, 1.0],
        }
    )
    offset = 1.0e12
    shifted = unshifted.with_columns(
        (pl.col("predicted") + offset).alias("predicted"),
        (pl.col("observed") + offset).alias("observed"),
    )

    unshifted_score = ft.score_sample_crps(
        unshifted, forecast_unit=["unit"], scale="natural"
    ).item(0, "crps")
    shifted_score = ft.score_sample_crps(
        shifted, forecast_unit=["unit"], scale="natural"
    ).item(0, "crps")

    assert shifted_score == pytest.approx(unshifted_score)

    large_value = 1.0e100
    identical = pl.DataFrame(
        {
            "unit": ["a"] * 100,
            "sample_id": list(range(100)),
            "predicted": [large_value] * 100,
            "observed": [large_value] * 100,
        }
    )
    identical_score = ft.score_sample_crps(
        identical, forecast_unit=["unit"], scale="natural"
    ).item(0, "crps")
    assert identical_score == 0.0


def test_repeated_predicted_values_are_accepted() -> None:
    """Equal values from distinct samples remain valid samples."""
    forecasts = pl.DataFrame(
        {
            "unit": ["a"] * 4,
            "sample_id": [0, 1, 2, 3],
            "predicted": [0.0, 0.0, 2.0, 2.0],
            "observed": [1.0] * 4,
        }
    )

    result = ft.score_sample_crps(forecasts, forecast_unit=["unit"], scale="natural")

    assert result.item(0, "crps") == 0.5


def test_multiple_models_are_scored_as_separate_forecast_units() -> None:
    """Only the declared forecast-unit columns determine grouping."""
    forecasts = pl.DataFrame(
        {
            "model": ["worse", "better", "worse", "better"],
            "location": ["US"] * 4,
            "ignored": [1, 2, 3, 4],
            "sample_id": [10, 20, 11, 21],
            "predicted": [0.0, 1.0, 4.0, 1.0],
            "observed": [1.0] * 4,
        }
    )

    result = ft.score_sample_crps(
        forecasts,
        forecast_unit=["model", "location"],
        scale="natural",
    )

    assert result.rows(named=True) == [
        {"model": "better", "location": "US", "scale": "natural", "crps": 0.0},
        {"model": "worse", "location": "US", "scale": "natural", "crps": 1.0},
    ]


def test_units_may_have_different_sample_ids_and_counts() -> None:
    """Sample identities and counts need not be coherent across units."""
    forecasts = pl.DataFrame(
        {
            "unit": ["a", "b", "b"],
            "sample_id": [100, 3, 4],
            "predicted": [2.0, 0.0, 2.0],
            "observed": [1.0, 1.0, 1.0],
        }
    )

    result = ft.score_sample_crps(forecasts, forecast_unit=["unit"], scale="natural")

    assert result.get_column("crps").to_list() == [1.0, 0.5]


def test_custom_column_names_are_supported() -> None:
    """Callers may name sample, prediction, and observation columns."""
    forecasts = pl.DataFrame(
        {
            "unit": ["a", "a"],
            "draw": [0, 1],
            "forecast": [0.0, 2.0],
            "truth": [1.0, 1.0],
        }
    )

    result = ft.score_sample_crps(
        forecasts,
        forecast_unit=["unit"],
        scale="natural",
        sample_id_col="draw",
        prediction_col="forecast",
        observation_col="truth",
    )

    assert result.item(0, "crps") == 0.5


def test_log1p_scale_transforms_values_before_scoring() -> None:
    """Natural and log1p scores are calculated independently."""
    forecasts = pl.DataFrame(
        {
            "unit": ["a", "a"],
            "sample_id": [0, 1],
            "predicted": [0.0, 3.0],
            "observed": [1.0, 1.0],
        }
    )

    natural = ft.score_sample_crps(forecasts, forecast_unit=["unit"], scale="natural")
    log1p = ft.score_sample_crps(forecasts, forecast_unit=["unit"], scale="log1p")

    assert natural.item(0, "crps") == 0.75
    assert log1p.item(0, "crps") == pytest.approx(np.log(2.0) / 2.0)
    assert natural.item(0, "scale") == "natural"
    assert log1p.item(0, "scale") == "log1p"


def test_better_forecast_has_lower_crps() -> None:
    """A forecast concentrated on the observation receives a lower score."""
    forecasts = pl.DataFrame(
        {
            "unit": ["better", "better", "worse", "worse"],
            "sample_id": [0, 1, 0, 1],
            "predicted": [1.0, 1.0, 0.0, 4.0],
            "observed": [1.0] * 4,
        }
    )

    result = ft.score_sample_crps(forecasts, forecast_unit=["unit"], scale="natural")

    scores = dict(result.select("unit", "crps").iter_rows())
    assert scores["better"] < scores["worse"]


def test_output_schema_and_order_are_deterministic() -> None:
    """Output preserves unit types and follows the documented order."""
    forecasts = pl.DataFrame(
        {
            "reference_date": [
                datetime.date(2026, 1, 2),
                datetime.date(2026, 1, 1),
            ],
            "model": pl.Series(["b", "a"], dtype=pl.Categorical),
            "sample_id": [0, 0],
            "predicted": [2, 1],
            "observed": [1, 1],
        }
    )

    result = ft.score_sample_crps(
        forecasts,
        forecast_unit=["reference_date", "model"],
        scale="natural",
    )

    assert result.columns == ["reference_date", "model", "scale", "crps"]
    assert result.schema["reference_date"] == pl.Date
    assert result.schema["model"] == forecasts.schema["model"]
    assert result.schema["scale"] == pl.String
    assert result.schema["crps"] == pl.Float64
    assert result.get_column("reference_date").to_list() == [
        datetime.date(2026, 1, 1),
        datetime.date(2026, 1, 2),
    ]


@pytest.mark.parametrize(
    "prediction_dtype,observation_dtype",
    [
        (pl.Int32, pl.UInt16),
        (pl.Float32, pl.Float64),
        (pl.Decimal(10, 2), pl.Decimal(10, 2)),
    ],
)
def test_numeric_columns_are_safely_converted_to_float64(
    prediction_dtype: pl.DataType | type[pl.DataType],
    observation_dtype: pl.DataType | type[pl.DataType],
) -> None:
    """Accepted Polars numeric types are converted before scoring."""
    predicted: list[int | float | Decimal]
    observed: list[int | float | Decimal]
    if prediction_dtype == pl.Decimal(10, 2):
        predicted = [Decimal("0.00"), Decimal("2.00")]
        observed = [Decimal("1.00"), Decimal("1.00")]
    else:
        predicted = [0, 2]
        observed = [1, 1]
    forecasts = pl.DataFrame(
        {
            "unit": ["a", "a"],
            "sample_id": [0, 1],
            "predicted": pl.Series(predicted, dtype=prediction_dtype),
            "observed": pl.Series(observed, dtype=observation_dtype),
        }
    )

    result = ft.score_sample_crps(forecasts, forecast_unit=["unit"], scale="natural")

    assert result.item(0, "crps") == 0.5
    assert result.schema["crps"] == pl.Float64


@pytest.mark.parametrize(
    "scale,expected_filename",
    [
        ("natural", "expected_natural.csv"),
        ("log1p", "expected_log1p.csv"),
    ],
)
def test_crps_matches_pinned_scoringutils_fixture(
    scale: Literal["natural", "log1p"], expected_filename: str
) -> None:
    """Python scores match results from pinned R package versions."""
    forecasts = pl.read_csv(SCORING_FIXTURE_DIR / "input.csv", try_parse_dates=True)
    expected = pl.read_csv(
        SCORING_FIXTURE_DIR / expected_filename, try_parse_dates=True
    )

    result = ft.score_sample_crps(forecasts, forecast_unit=FORECAST_UNIT, scale=scale)

    plt.assert_frame_equal(result, expected, check_exact=False, abs_tol=1e-12)


def test_empty_input_is_rejected() -> None:
    """A scorer call requires at least one sample row."""
    with pytest.raises(ValueError, match="at least one row"):
        ft.score_sample_crps(pl.DataFrame(), forecast_unit=["unit"], scale="natural")


def test_non_dataframe_input_is_rejected() -> None:
    """The input must be a Polars DataFrame."""
    with pytest.raises(TypeError, match="Polars DataFrame"):
        ft.score_sample_crps(
            cast(Any, {"unit": ["a"]}),
            forecast_unit=["unit"],
            scale="natural",
        )


@pytest.mark.parametrize(
    "missing_column", ["unit", "sample_id", "predicted", "observed"]
)
def test_missing_columns_are_rejected(missing_column: str) -> None:
    """Every configured input column must be present."""
    forecasts = _base_forecasts().drop(missing_column)

    with pytest.raises(ValueError, match=missing_column):
        ft.score_sample_crps(forecasts, forecast_unit=["unit"], scale="natural")


@pytest.mark.parametrize(
    "forecast_unit,error_type,error_match",
    [
        ([], ValueError, "at least one"),
        ("unit", TypeError, "sequence"),
        (["unit", "unit"], ValueError, "distinct"),
        ([""], ValueError, "nonempty"),
        ([1], TypeError, "strings"),
        (["scale"], ValueError, "reserved"),
        (["crps"], ValueError, "reserved"),
        (["sample_id"], ValueError, "must not contain"),
        (["predicted"], ValueError, "must not contain"),
        (["observed"], ValueError, "must not contain"),
    ],
)
def test_invalid_forecast_unit_is_rejected(
    forecast_unit: Any, error_type: type[Exception], error_match: str
) -> None:
    """Forecast-unit names must be valid, distinct, and non-overlapping."""
    forecasts = _base_forecasts().with_columns(
        pl.lit("input scale").alias("scale"),
        pl.lit(10.0).alias("crps"),
    )

    with pytest.raises(error_type, match=error_match):
        ft.score_sample_crps(
            forecasts,
            forecast_unit=cast(Any, forecast_unit),
            scale="natural",
        )


@pytest.mark.parametrize(
    "overrides,error_type,error_match",
    [
        ({"sample_id_col": ""}, ValueError, "sample_id_col"),
        ({"prediction_col": ""}, ValueError, "prediction_col"),
        ({"observation_col": ""}, ValueError, "observation_col"),
        ({"sample_id_col": 1}, TypeError, "sample_id_col"),
        ({"prediction_col": 1}, TypeError, "prediction_col"),
        ({"observation_col": 1}, TypeError, "observation_col"),
        ({"sample_id_col": "predicted"}, ValueError, "must be distinct"),
    ],
)
def test_invalid_value_column_names_are_rejected(
    overrides: dict[str, Any], error_type: type[Exception], error_match: str
) -> None:
    """Configured value-column names must be strings and remain distinct."""
    with pytest.raises(error_type, match=error_match):
        ft.score_sample_crps(
            _base_forecasts(),
            forecast_unit=["unit"],
            scale="natural",
            **overrides,
        )


@pytest.mark.parametrize("column", ["predicted", "observed"])
@pytest.mark.parametrize("dtype", [pl.String, pl.Boolean])
def test_nonnumeric_value_columns_are_rejected(
    column: str, dtype: pl.DataType | type[pl.DataType]
) -> None:
    """Prediction and observation columns must be numeric and non-boolean."""
    forecasts = _base_forecasts().with_columns(pl.col(column).cast(dtype))

    with pytest.raises(TypeError, match="numeric, non-boolean"):
        ft.score_sample_crps(forecasts, forecast_unit=["unit"], scale="natural")


@pytest.mark.parametrize("column", ["predicted", "observed"])
@pytest.mark.parametrize("invalid_value", [None, np.nan, np.inf, -np.inf])
def test_invalid_value_data_are_rejected(
    column: str, invalid_value: float | None
) -> None:
    """Prediction and observation values must be complete and finite."""
    forecasts = _base_forecasts().with_columns(
        pl.Series(column, [0.0, invalid_value], dtype=pl.Float64)
    )
    error_match = "null" if invalid_value is None else "finite"

    with pytest.raises(ValueError, match=error_match):
        ft.score_sample_crps(forecasts, forecast_unit=["unit"], scale="natural")


def test_null_sample_identifiers_are_rejected() -> None:
    """Every predictive sample must have an identifier."""
    forecasts = _base_forecasts().with_columns(
        pl.Series("sample_id", [0, None], dtype=pl.Int64)
    )

    with pytest.raises(ValueError, match="must not contain null"):
        ft.score_sample_crps(forecasts, forecast_unit=["unit"], scale="natural")


def test_duplicate_sample_identifiers_within_unit_are_rejected() -> None:
    """A sample identifier may occur only once within a forecast unit."""
    forecasts = _base_forecasts().with_columns(pl.lit(0).alias("sample_id"))

    with pytest.raises(ValueError, match="unique within each forecast unit"):
        ft.score_sample_crps(forecasts, forecast_unit=["unit"], scale="natural")


def test_same_sample_identifier_across_units_is_accepted() -> None:
    """Different forecast units may reuse sample identifiers."""
    forecasts = pl.DataFrame(
        {
            "unit": ["a", "b"],
            "sample_id": [0, 0],
            "predicted": [1.0, 2.0],
            "observed": [1.0, 1.0],
        }
    )

    result = ft.score_sample_crps(forecasts, forecast_unit=["unit"], scale="natural")

    assert result.height == 2


def test_inconsistent_observations_are_rejected() -> None:
    """A forecast unit must be paired with exactly one observation."""
    forecasts = _base_forecasts().with_columns(pl.Series("observed", [1.0, 2.0]))

    with pytest.raises(ValueError, match="exactly one observation"):
        ft.score_sample_crps(forecasts, forecast_unit=["unit"], scale="natural")


def test_invalid_scale_is_rejected() -> None:
    """Only the documented natural and log1p scales are accepted."""
    with pytest.raises(ValueError, match="scale must be"):
        ft.score_sample_crps(
            _base_forecasts(),
            forecast_unit=["unit"],
            scale=cast(Any, "log"),
        )


@pytest.mark.parametrize("column", ["predicted", "observed"])
@pytest.mark.parametrize("invalid_value", [-1.0, -2.0])
def test_log1p_rejects_values_outside_domain(column: str, invalid_value: float) -> None:
    """Log1p scoring requires every scored value to be greater than -1."""
    forecasts = _base_forecasts().with_columns(pl.Series(column, [0.0, invalid_value]))

    with pytest.raises(ValueError, match="greater than -1"):
        ft.score_sample_crps(forecasts, forecast_unit=["unit"], scale="log1p")
