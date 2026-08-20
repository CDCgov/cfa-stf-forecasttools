import polars as pl


def augment_samples_with_observations(
    samples: pl.DataFrame,
    observations: pl.DataFrame,
    date_col: str = "date",
    draw_col: str = ".draw",
) -> pl.DataFrame:
    """Prepend observations before the forecast date for every sample draw."""
    if date_col not in samples.columns or date_col not in observations.columns:
        raise ValueError(f"samples and observations must contain a {date_col!r} column")
    if draw_col not in samples.columns:
        raise ValueError(f"samples is missing required column: {draw_col}")
    if samples.is_empty():
        raise ValueError("samples must contain at least one row")

    first_forecast_date = samples.get_column(date_col).min()
    target_draws = samples.select(draw_col).unique().sort(draw_col)
    obs_as_samples = observations.filter(pl.col(date_col) < first_forecast_date).join(
        target_draws, how="cross"
    )

    return pl.concat([obs_as_samples, samples], how="diagonal_relaxed")


def create_proportions(
    numerator_df: pl.DataFrame,
    other_df: pl.DataFrame,
    num_val_col: str,
    other_val_col: str,
    prop_var: str = "prop_disease_ed_visits",
) -> pl.DataFrame:
    """Create long-format proportions from two wide value DataFrames."""
    join_columns = sorted(
        set(numerator_df.columns).intersection(other_df.columns)
        - {num_val_col, other_val_col}
    )
    if not join_columns:
        raise ValueError(
            "numerator and other data have no identifier columns in common"
        )

    suffix = "_other"
    other_value_col = (
        f"{other_val_col}{suffix}"
        if other_val_col in numerator_df.columns
        else other_val_col
    )

    return (
        numerator_df.join(
            other_df,
            on=join_columns,
            how="inner",
            suffix=suffix,
            nulls_equal=True,
        )
        .with_columns(
            (
                pl.col(num_val_col) / (pl.col(num_val_col) + pl.col(other_value_col))
            ).alias(".value"),
            pl.lit(prop_var).alias(".variable"),
        )
        .drop(num_val_col, other_value_col)
    )
