"""
cutoff_policy.py
================
Stateless policy engine enforcing the Day 133 Cutoff Law.

The Day 133 boundary is the single most important data-hygiene constraint in
this pipeline.  DayNum 133 is the start of the NCAA Tournament play-in round;
any game result at or beyond that day must never cross the pre-tournament
feature boundary.

This module provides:
  - CutoffViolationError  — raised when a specific (season, day_num) pair
                             violates the cutoff.
  - Day133CutoffPolicy    — stateless enforcer; two methods:
        filter_dataframe   drops rows that violate the cutoff.
        assert_permitted   hard-checks a single (season, day_num) pair.

Design notes
------------
* The policy is intentionally stateless.  Its sole input at call time is the
  FoldContext that carries the cutoff value.
* There is no override path.  There is no "relaxed mode".
* Inclusive/exclusive semantics are unambiguous:
    permitted:   DayNum < 133
    forbidden:   DayNum >= 133
"""

from __future__ import annotations

import pandas as pd

from ncaa_pipeline.context.fold_context import FoldContext


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class CutoffViolationError(Exception):
    """
    Raised when a (season, day_num) pair violates the Day 133 cutoff.

    Attributes
    ----------
    season   : int  — season year of the offending observation
    day_num  : int  — DayNum value that triggered the violation
    fold_id  : str  — fold_id from the active FoldContext
    cutoff   : int  — the active cutoff (always 133)
    """

    def __init__(
        self,
        season: int,
        day_num: int,
        fold_id: str,
        cutoff: int,
    ) -> None:
        self.season = season
        self.day_num = day_num
        self.fold_id = fold_id
        self.cutoff = cutoff
        super().__init__(
            f"Cutoff violation in fold {fold_id!r}: "
            f"DayNum {day_num} in season {season} is >= cutoff {cutoff}.  "
            f"Permitted range: DayNum < {cutoff}."
        )


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------

class Day133CutoffPolicy:
    """
    Stateless enforcer of the Day 133 pre-tournament information boundary.

    No constructor state is required.  Both public methods accept a
    FoldContext, which carries the authoritative cutoff value (always 133).

    Usage
    -----
    policy = Day133CutoffPolicy()
    clean_df = policy.filter_dataframe(raw_df, context)
    policy.assert_permitted(season=2022, day_num=100, context=context)
    """

    # ------------------------------------------------------------------
    # Schema validation helper
    # ------------------------------------------------------------------

    def validate_dataframe_schema(self, df: pd.DataFrame) -> None:
        """
        Verify that *df* contains the required 'DayNum' column.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to inspect.

        Raises
        ------
        ValueError
            If 'DayNum' is absent from the column index.
        """
        if "DayNum" not in df.columns:
            raise ValueError(
                "Day133CutoffPolicy requires a 'DayNum' column; "
                f"found columns: {list(df.columns)}."
            )

    # ------------------------------------------------------------------
    # filter_dataframe
    # ------------------------------------------------------------------

    def filter_dataframe(
        self,
        df: pd.DataFrame,
        context: FoldContext,
    ) -> pd.DataFrame:
        """
        Return a new DataFrame containing only rows with DayNum < cutoff.

        Rows with DayNum >= context.day_cutoff are dropped.  The input
        DataFrame is never mutated.  Row order and the original integer index
        are preserved for surviving rows.

        Parameters
        ----------
        df : pd.DataFrame
            Source DataFrame.  Must contain a 'DayNum' column with numeric
            values.
        context : FoldContext
            Active fold context supplying the authoritative cutoff.

        Returns
        -------
        pd.DataFrame
            Filtered copy; same columns as *df*, subset of rows.

        Raises
        ------
        ValueError
            If 'DayNum' column is absent.
        """
        self.validate_dataframe_schema(df)
        mask: pd.Series = df["DayNum"] < context.day_cutoff  # type: ignore[assignment]
        return df.loc[mask].copy()

    # ------------------------------------------------------------------
    # assert_permitted
    # ------------------------------------------------------------------

    def assert_permitted(
        self,
        season: int,
        day_num: int,
        context: FoldContext,
    ) -> None:
        """
        Assert that a single (season, day_num) observation is within the
        permitted pre-tournament window.

        Parameters
        ----------
        season : int
            Season year of the observation (used only for the error message).
        day_num : int
            DayNum value to check.
        context : FoldContext
            Active fold context supplying the authoritative cutoff.

        Raises
        ------
        CutoffViolationError
            Immediately if day_num >= context.day_cutoff.
        """
        if day_num >= context.day_cutoff:
            raise CutoffViolationError(
                season=season,
                day_num=day_num,
                fold_id=context.fold_id,
                cutoff=context.day_cutoff,
            )
