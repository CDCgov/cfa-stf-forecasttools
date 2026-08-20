from pathlib import Path
from typing import Any

import polars as pl
import polars.selectors as cs


def _read_parquet_with_timezone_correction(
    path_to_file: str | Path, **kwargs: Any
) -> pl.DataFrame:
    """Read Parquet data, interpreting timezone-naive timestamps as UTC."""
    df = pl.read_parquet(path_to_file, **kwargs)
    timestamp_columns_without_timezone = [
        name
        for name, dtype in df.schema.items()
        if isinstance(dtype, pl.Datetime) and dtype.time_zone is None
    ]

    return df.with_columns(
        pl.col(timestamp_columns_without_timezone).dt.convert_time_zone("UTC")
    )


def read_tabular(path_to_file: str | Path, **kwargs: Any) -> pl.DataFrame:
    """Read a tabular file, inferring its format from the file extension.

    Parameters
    ----------
    path_to_file
        Path to a ``.csv``, ``.tsv``, or ``.parquet`` file. The extension is
        matched case-insensitively.
    **kwargs
        Additional keyword arguments passed to :func:`polars.read_csv` or
        :func:`polars.read_parquet`, depending on the inferred format.

    Returns
    -------
    pl.DataFrame
        The contents of the file.

    Raises
    ------
    ValueError
        If the path does not have a supported file extension.
    """
    file_format = Path(path_to_file).suffix.removeprefix(".").lower()

    if file_format in {"csv", "tsv"}:
        kwargs.setdefault("try_parse_dates", True)

    if file_format == "csv":
        return pl.read_csv(path_to_file, **kwargs)
    if file_format == "tsv":
        kwargs.setdefault("separator", "\t")
        return pl.read_csv(path_to_file, **kwargs)
    if file_format == "parquet":
        return _read_parquet_with_timezone_correction(path_to_file, **kwargs)

    supported_formats = ".csv, .tsv, and .parquet"
    raise ValueError(
        f"Unsupported file extension {Path(path_to_file).suffix!r}; "
        f"expected one of {supported_formats}."
    )


def coalesce_common_columns(
    df: pl.DataFrame, suffix: str, new_colname: str | None = None
) -> pl.DataFrame:
    """
    Coalesce multiple columns with a common suffix into a single column.
    This function finds all columns in the DataFrame that end with the specified
    suffix, coalesces them (takes the first non-null value across the columns),
    and creates a new column with the coalesced values. The original columns
    with the suffix are then removed from the DataFrame.

    Parameters
    ----------
    df : pl.DataFrame
        The input Polars DataFrame to process.
    suffix : str
        The suffix to match column names for coalescing.
    new_colname : str | None, optional
        The name for the new coalesced column.
        If None, defaults to the suffix with leading underscores stripped.

    Returns:
        pl.DataFrame: A new DataFrame with the coalesced column and original suffix columns removed.
    """
    if new_colname is None:
        new_colname = suffix.lstrip("_")
    coalesced_df = df.with_columns(
        pl.coalesce(cs.ends_with(suffix)).alias(new_colname)
    ).select(cs.exclude(cs.ends_with(suffix)))

    return coalesced_df
