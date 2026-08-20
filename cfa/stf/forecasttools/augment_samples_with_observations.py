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
