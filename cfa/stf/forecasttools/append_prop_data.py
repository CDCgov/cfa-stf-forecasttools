import polars as pl

from .create_proportions import create_proportions


def append_prop_data(
    data: pl.DataFrame,
    observed_var: str = "observed_ed_visits",
    other_var: str = "other_ed_visits",
    prop_var: str = "prop_disease_ed_visits",
) -> pl.DataFrame:
    """Append proportion rows derived from two variables in long-format data."""
    required_columns = {"date", ".variable", ".value"}
    if missing_columns := required_columns.difference(data.columns):
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"data is missing required column(s): {missing}")

    numerator_df = (
        data.filter(pl.col(".variable") == observed_var)
        .drop(".variable")
        .rename({".value": observed_var})
    )
    other_df = (
        data.filter(pl.col(".variable") == other_var)
        .drop(".variable")
        .rename({".value": other_var})
    )
    prop_data = create_proportions(
        numerator_df=numerator_df,
        other_df=other_df,
        num_val_col=observed_var,
        other_val_col=other_var,
        prop_var=prop_var,
    )

    return pl.concat([data, prop_data], how="diagonal_relaxed").sort(
        "date", ".variable", maintain_order=True
    )
