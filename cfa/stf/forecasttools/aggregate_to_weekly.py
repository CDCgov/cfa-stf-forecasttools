import datetime
from typing import Literal

import epiweeks
import polars as pl

WeekStandard = Literal["MMWR", "USA", "ISO"]


def _normalize_week_standard(standard: WeekStandard | str) -> Literal["cdc", "iso"]:
    """Normalize week standard labels."""
    normalized = standard.upper()
    if normalized in ["MMWR", "USA"]:
        return "cdc"
    if normalized in ["ISO"]:
        return "iso"

    raise ValueError(
        "standard must be one of {'MMWR', 'USA', 'ISO'} (not case-sensitive)"
    )


def _calculate_week_and_year(
    date_input: datetime.date | datetime.datetime,
    standard: WeekStandard | str = "MMWR",
) -> dict[str, int]:
    """Convert a date to week and weekyear for the requested weekly standard."""
    if not isinstance(date_input, datetime.date):
        raise TypeError(
            f"date_input must be either a datetime.date or datetime.datetime, got {type(date_input).__name__}"
        )
    epiweek_standard = _normalize_week_standard(standard)

    epiweek = epiweeks.Week.fromdate(date_input, system=epiweek_standard)
    return {
        "week": epiweek.week,
        "weekyear": epiweek.year,
    }


def _calculate_week_startdate(
    year: int,
    week: int,
    standard: WeekStandard | str = "MMWR",
) -> datetime.date:
    """Given a week and year, return the week start date."""
    epiweek_standard = _normalize_week_standard(standard)
    return epiweeks.Week(year, week, system=epiweek_standard).startdate()


def _calculate_week_enddate(
    year: int,
    week: int,
    standard: WeekStandard | str = "MMWR",
) -> datetime.date:
    """Given a week and year, return the week end date."""
    epiweek_standard = _normalize_week_standard(standard)
    return epiweeks.Week(year, week, system=epiweek_standard).enddate()


def floor_week(
    date: datetime.date | datetime.datetime,
    standard: WeekStandard | str = "MMWR",
) -> datetime.date:
    """Get the first date in an epidemiological week.

    Given any date from an epidemiological week, return
    the date of the first day of the epiweek,
    according to the given epidemiological week standard.

    Parameters
    ----------
    date
        A date.
    standard
        One of ``"USA"`` or ``"MMWR"``
        (USA / MMWR epiweek, starts on Sunday) and ``"ISO"``
        (ISO week, starts on Monday). Not case-sensitive.

    Returns
    -------
    datetime.date
        The date that starts the epidemiological week.
    """
    week_info = _calculate_week_and_year(date, standard=standard)
    return _calculate_week_startdate(
        year=week_info["weekyear"],
        week=week_info["week"],
        standard=standard,
    )


def ceiling_week(
    date: datetime.date | datetime.datetime,
    standard: WeekStandard | str = "MMWR",
) -> datetime.date:
    """Get the last date in an epidemiological week.

    Given any date from an epidemiological week, return
    the date of the final day of the epiweek,
    according to the given epidemiological week standard.

    Parameters
    ----------
    date
        A date.
    standard
        One of ``"USA"`` or ``"MMWR"``
        (USA / MMWR epiweek, starts on Sunday) and ``"ISO"``
        (ISO week, starts on Monday). Not case-sensitive.

    Returns
    -------
    datetime.date
        The date that ends the epidemiological week.

    """
    week_info = _calculate_week_and_year(date, standard=standard)
    return _calculate_week_enddate(
        year=week_info["weekyear"],
        week=week_info["week"],
        standard=standard,
    )


def floor_isoweek(
    date: datetime.date | datetime.datetime,
) -> datetime.date:
    """Get the first date (Monday) in an ISO week."""
    return floor_week(date, standard="ISO")


def ceiling_isoweek(
    date: datetime.date | datetime.datetime,
) -> datetime.date:
    """Get the last date (Sunday) in an ISO week."""
    return ceiling_week(date, standard="ISO")


def floor_mmwr_epiweek(
    date: datetime.date | datetime.datetime,
) -> datetime.date:
    """Get the first date (Sunday) in an MMWR epidemiological week."""
    return floor_week(date, standard="MMWR")


def ceiling_mmwr_epiweek(
    date: datetime.date | datetime.datetime,
) -> datetime.date:
    """Get the last date (Saturday) in an MMWR epidemiological week."""
    return ceiling_week(date, standard="MMWR")


def daily_to_weekly(
    df: pl.DataFrame,
    value_col: str = "value",
    date_col: str = "date",
    id_cols: str | list[str] = ".draw",
    weekly_value_name: str = "weekly_value",
    standard: WeekStandard | str = "MMWR",
    with_week_start_date: bool = False,
    with_week_end_date: bool = False,
    week_start_date_name: str = "week_start_date",
    week_end_date_name: str = "week_end_date",
    strict: bool = True,
) -> pl.DataFrame:
    """
    Aggregate a dataframe of daily values to weekly totals.

    Parameters
    ----------
    df
        Tidy data frame of daily values to aggregate.
    value_col
        Name of the column containing daily values. Defaults to ``"value"``.
    date_col
        Name of the date column. Defaults to ``"date"``.
    id_cols
        Column name(s) that uniquely identify a single timeseries.
    weekly_value_name
        Name for the output weekly value column.
        Defaults to ``"weekly_value"``.
    standard
        Weekly standard used to compute week and weekyear.
        One of ``"MMWR"`` or ``"USA"`` (Sunday start) and ``"ISO"``
        (Monday start). Not case-sensitive.
        Defaults to ``"MMWR"``.
    with_week_start_date
        Whether to annotate output with week start date.
    with_week_end_date
        Whether to annotate output with week end date.
    week_start_date_name
        Name for the output week start-date column.
        Defaults to ``"week_start_date"``.
    week_end_date_name
        Name for the output week end-date column.
        Defaults to ``"week_end_date"``.
    strict
        If ``True``, only aggregate weeks with exactly seven dates.

    Returns
    -------
    pl.DataFrame
        Dataframe with weekly aggregated values and optionally week metadata.
    """
    if value_col not in df.columns:
        raise ValueError(
            f"Specified value column '{value_col}' is missing from the dataframe"
        )
    if date_col not in df.columns:
        raise ValueError(
            f"Specified date column '{date_col}' is missing from the dataframe"
        )

    if isinstance(id_cols, str):
        id_cols = [id_cols]
    else:
        id_cols = list(id_cols)
    missing_id_cols = [col for col in id_cols if col not in df.columns]
    if missing_id_cols:
        raise ValueError(
            f"Specified trajectory id column(s) {missing_id_cols} are missing from the dataframe"
        )

    df = (
        df.drop_nulls(value_col)
        .drop_nans(value_col)
        .with_columns(
            pl.col(date_col)
            .map_elements(
                lambda elt: _calculate_week_and_year(elt, standard=standard),
                return_dtype=pl.Struct(
                    [pl.Field("week", pl.Int64), pl.Field("weekyear", pl.Int64)]
                ),
            )
            .alias("week_struct")
        )
        .unnest("week_struct")
    )

    group_cols = ["week", "weekyear"] + id_cols
    grouped_df = df.group_by(group_cols).agg(
        pl.col(value_col).sum().alias(weekly_value_name),
        pl.len().alias("n_dates"),
        pl.col(date_col).dt.date().n_unique().alias("n_unique_dates"),
    )

    duplicate_date_trajectories = grouped_df.filter(
        pl.col("n_dates") != pl.col("n_unique_dates")
    )
    if not duplicate_date_trajectories.is_empty():
        message = (
            "At least one trajectory has repeated dates within a given "
            f"week.\n"
            f"Trajectories with repeated dates: {duplicate_date_trajectories}"
        )
        raise ValueError(message)

    if strict:
        grouped_df = grouped_df.filter(pl.col("n_unique_dates") == 7)

    grouped_df = grouped_df.drop(["n_dates", "n_unique_dates"]).sort(
        [*id_cols, "weekyear", "week"]
    )

    if with_week_start_date:
        grouped_df = grouped_df.with_columns(
            pl.struct(["weekyear", "week"])
            .map_elements(
                lambda elt: _calculate_week_startdate(
                    year=elt["weekyear"],
                    week=elt["week"],
                    standard=standard,
                ),
                return_dtype=pl.Date,
            )
            .alias(week_start_date_name)
        )

    if with_week_end_date:
        grouped_df = grouped_df.with_columns(
            pl.struct(["weekyear", "week"])
            .map_elements(
                lambda elt: _calculate_week_enddate(
                    year=elt["weekyear"],
                    week=elt["week"],
                    standard=standard,
                ),
                return_dtype=pl.Date,
            )
            .alias(week_end_date_name)
        )

    return grouped_df
