"""
massey_extractor.py
===================
Safe pregame ordinal rank extraction from Massey-style ordinals data.

Enforces the Day 133 Cutoff Law: ``safe_snapshot()`` never returns a ranking
with ``RankingDayNum >= cutoff_day``.

Last Available Day (LAD) logic: among all rankings for a given
(season, team_id, system_name) that are strictly below the cutoff, the one
with the highest ``RankingDayNum`` is chosen.

Staleness is classified explicitly:
  OK       — ranking_day >= cutoff_day - stale_warning_days
  WARNING  — stale_warning_days < staleness <= stale_critical_days
  CRITICAL — staleness > stale_critical_days, or no ranking exists at all
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass

import pandas as pd


# ---------------------------------------------------------------------------
# Staleness classification
# ---------------------------------------------------------------------------

class OrdinalStatus(enum.Enum):
    """Staleness classification for an ordinal snapshot."""
    OK = "ok"
    WARNING = "warning"
    CRITICAL = "critical"


# ---------------------------------------------------------------------------
# Snapshot result type
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OrdinalSnapshot:
    """
    Result of a single safe ordinal rank lookup.

    Attributes
    ----------
    season : int
    team_id : int
    system_name : str
    ordinal_rank : float
        The ordinal rank value, or ``float('nan')`` if no ranking was found.
    ranking_day : int
        The ``RankingDayNum`` of the snapshot actually used, or ``-1`` if no
        valid ranking existed.
    cutoff_day : int
        The cutoff that was enforced (exclusive upper bound on RankingDayNum).
    status : OrdinalStatus
        Staleness classification.
    """

    season: int
    team_id: int
    system_name: str
    ordinal_rank: float
    ranking_day: int
    cutoff_day: int
    status: OrdinalStatus

    @property
    def is_available(self) -> bool:
        """True if a valid ranking was found (ordinal_rank is not NaN)."""
        return not math.isnan(self.ordinal_rank)

    @property
    def staleness_days(self) -> int | None:
        """
        Days between the snapshot day and the cutoff.

        Returns ``None`` if no valid ranking existed (ranking_day == -1).
        """
        if self.ranking_day < 0:
            return None
        return self.cutoff_day - self.ranking_day


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------

class MasseyOrdinalExtractor:
    """
    Safe extractor for pregame ordinal rankings.

    Enforces the Day 133 Cutoff Law: ``safe_snapshot()`` never returns a
    ranking at or beyond ``cutoff_day``.  LAD logic picks the latest
    available ranking strictly below the cutoff.

    Parameters
    ----------
    ordinals_df : pd.DataFrame
        Massey ordinals data.  Required columns:
        ``Season``, ``RankingDayNum``, ``SystemName``, ``TeamID``,
        ``OrdinalRank``.
    stale_warning_days : int
        Staleness (``cutoff_day - ranking_day``) at or below which a snapshot
        is classified as OK.  Default: 14.
    stale_critical_days : int
        Staleness above which a snapshot is classified as CRITICAL.
        Must be strictly greater than ``stale_warning_days``.  Default: 28.
    """

    _REQUIRED_COLS: frozenset[str] = frozenset({
        "Season", "RankingDayNum", "SystemName", "TeamID", "OrdinalRank",
    })

    _DEFAULT_WARNING_DAYS: int = 14
    _DEFAULT_CRITICAL_DAYS: int = 28

    def __init__(
        self,
        ordinals_df: pd.DataFrame,
        stale_warning_days: int = _DEFAULT_WARNING_DAYS,
        stale_critical_days: int = _DEFAULT_CRITICAL_DAYS,
    ) -> None:
        self._validate_schema(ordinals_df)
        if stale_warning_days >= stale_critical_days:
            raise ValueError(
                f"stale_warning_days ({stale_warning_days}) must be less than "
                f"stale_critical_days ({stale_critical_days})."
            )
        self._df = ordinals_df.copy()
        self._stale_warning_days = stale_warning_days
        self._stale_critical_days = stale_critical_days

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def safe_snapshot(
        self,
        season: int,
        team_id: int,
        cutoff_day: int,
        system_name: str,
    ) -> OrdinalSnapshot:
        """
        Return the Last Available Day (LAD) ordinal rank for a team before
        the cutoff.

        Among all rankings for (season, team_id, system_name) with
        ``RankingDayNum < cutoff_day``, the row with the highest
        ``RankingDayNum`` is chosen.  A ranking at or beyond ``cutoff_day``
        is never returned.

        Parameters
        ----------
        season : int
        team_id : int
        cutoff_day : int
            Exclusive upper bound.  Rankings at ``RankingDayNum >= cutoff_day``
            are forbidden.
        system_name : str
            Exact system identifier (e.g. ``"MOR"``, ``"SAG"``).

        Returns
        -------
        OrdinalSnapshot
            Status is CRITICAL (with NaN rank and ranking_day = -1) if no
            valid ranking exists before the cutoff.

        Raises
        ------
        ValueError
            If ``cutoff_day < 1``.
        """
        if cutoff_day < 1:
            raise ValueError(
                f"cutoff_day must be >= 1; got {cutoff_day}."
            )

        # Filter: correct season + team + system, strictly below cutoff
        mask = (
            (self._df["Season"] == season)
            & (self._df["TeamID"] == team_id)
            & (self._df["SystemName"] == system_name)
            & (self._df["RankingDayNum"] < cutoff_day)
        )
        candidates = self._df.loc[mask]

        if candidates.empty:
            return OrdinalSnapshot(
                season=season,
                team_id=team_id,
                system_name=system_name,
                ordinal_rank=float("nan"),
                ranking_day=-1,
                cutoff_day=cutoff_day,
                status=OrdinalStatus.CRITICAL,
            )

        # LAD: row with the highest RankingDayNum below the cutoff
        best_idx = candidates["RankingDayNum"].idxmax()
        best_row = candidates.loc[best_idx]
        ranking_day = int(best_row["RankingDayNum"])
        ordinal_rank = float(best_row["OrdinalRank"])

        staleness = cutoff_day - ranking_day
        if staleness <= self._stale_warning_days:
            status = OrdinalStatus.OK
        elif staleness <= self._stale_critical_days:
            status = OrdinalStatus.WARNING
        else:
            status = OrdinalStatus.CRITICAL

        return OrdinalSnapshot(
            season=season,
            team_id=team_id,
            system_name=system_name,
            ordinal_rank=ordinal_rank,
            ranking_day=ranking_day,
            cutoff_day=cutoff_day,
            status=status,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _validate_schema(self, df: pd.DataFrame) -> None:
        missing = self._REQUIRED_COLS - set(df.columns)
        if missing:
            raise ValueError(
                f"MasseyOrdinalExtractor: missing required columns: {sorted(missing)}.  "
                f"Present columns: {sorted(df.columns)}."
            )
