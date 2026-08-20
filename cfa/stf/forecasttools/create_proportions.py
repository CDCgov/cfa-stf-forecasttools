import polars as pl


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
