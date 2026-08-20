import polars as pl


def _get_only_value(data: pl.DataFrame, column: str) -> object:
    if column not in data.columns:
        raise ValueError(f"data is missing required column: {column}")

    values = data.get_column(column).unique().to_list()
    if len(values) != 1:
        raise ValueError(f"data must contain exactly one {column!r} value")
    return values[0]


def augment_samples_with_observations(
    samples: pl.DataFrame, observations: pl.DataFrame
) -> pl.DataFrame:
    """Prepend observations before the forecast date for every sample draw."""
    if "date" not in samples.columns or "date" not in observations.columns:
        raise ValueError("samples and observations must contain a 'date' column")
    if ".draw" not in samples.columns:
        raise ValueError("samples is missing required column: .draw")
    if samples.is_empty():
        raise ValueError("samples must contain at least one row")

    sample_resolution = _get_only_value(samples, "resolution")
    observation_resolution = _get_only_value(observations, "resolution")
    if sample_resolution != observation_resolution:
        raise ValueError("samples and observations must use the same resolution")

    first_forecast_date = samples.get_column("date").min()
    target_draws = samples.select(".draw").unique().sort(".draw")
    training_samples = (
        observations.filter(pl.col("date") < first_forecast_date)
        .join(target_draws, how="cross")
        .with_columns(pl.lit("train").alias("data_type"))
    )

    return pl.concat([training_samples, samples], how="diagonal_relaxed").with_columns(
        pl.lit(sample_resolution).alias("resolution")
    )


def create_proportions(
    numerator: pl.DataFrame,
    other: pl.DataFrame,
    numerator_col: str,
    other_col: str,
    proportion_var: str = "prop_disease_ed_visits",
) -> pl.DataFrame:
    """Create long-format proportions from two wide value DataFrames."""
    join_columns = [
        column
        for column in numerator.columns
        if column != numerator_col and column in other.columns
    ]
    if not join_columns:
        raise ValueError(
            "numerator and other data have no identifier columns in common"
        )

    return (
        numerator.join(other, on=join_columns, how="inner", nulls_equal=True)
        .with_columns(
            (pl.col(numerator_col) / (pl.col(numerator_col) + pl.col(other_col))).alias(
                ".value"
            ),
            pl.lit(proportion_var).alias(".variable"),
        )
        .drop(numerator_col, other_col)
    )
