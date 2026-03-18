"""
rolling_store.py
================
Rolling/season-summary team feature computation.

Computes pre-tournament-eligible features from regular-season detailed game results.

Architectural laws enforced here:
1. Day 133 Cutoff Law — ``Day133CutoffPolicy.filter_dataframe()`` is called as
   the first meaningful data operation in ``materialize()``.  No aggregation
   is ever performed on unfiltered data.
2. No winner/loser-coded leakage — game rows are pivoted to a neutral
   team-perspective representation *before* any per-team aggregation.
3. Freeze law — once frozen, the store rejects all mutation and rebuild.
4. No dataset objects — outputs are DataFrames and plain dicts, never
   ``TrainDataset``, ``CalDataset``, ``EvalDataset``, or ``EvalLabels``.
"""

from __future__ import annotations

from typing import Iterator

import numpy as np
import pandas as pd

from ncaa_pipeline.context.fold_context import FoldContext
from ncaa_pipeline.policies.cutoff_policy import Day133CutoffPolicy


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class FrozenStoreError(Exception):
    """Raised when a frozen RollingFeatureStore is asked to mutate or rebuild."""


# ---------------------------------------------------------------------------
# RollingFeatureStore
# ---------------------------------------------------------------------------

class RollingFeatureStore:
    """
    Per-team season-summary feature store.

    Computes a fixed set of team-level features from raw regular-season
    detailed results.  All aggregation is performed on data that has already
    passed through ``Day133CutoffPolicy.filter_dataframe()``.

    Parameters
    ----------
    context : FoldContext
        Active fold context; supplies the authoritative ``day_cutoff``.
    recent_form_n : int
        Number of most-recent games (by DayNum) used to compute
        ``recent_form_win_pct``.  Default: 10.
    """

    _REQUIRED_COLS: frozenset[str] = frozenset({
        "Season", "DayNum",
        "WTeamID", "LTeamID", "WScore", "LScore",
        "WFGM", "WFGA", "WFGM3", "WFGA3", "WFTM", "WFTA",
        "WOR", "WDR", "WTO",
        "LFGM", "LFGA", "LFGM3", "LFGA3", "LFTM", "LFTA",
        "LOR", "LDR", "LTO",
    })

    # All feature names produced by materialize().  Downstream code must not
    # hard-code column names; iterate over this tuple instead.
    FEATURE_NAMES: tuple[str, ...] = (
        "games_played",
        "wins",
        "win_pct",
        "points_scored_mean",
        "points_allowed_mean",
        "scoring_margin_mean",
        "efg_pct",
        "tov_rate",
        "orb_rate",
        "ftr",
        "recent_form_win_pct",
        "sos_opp_win_pct",
    )

    def __init__(
        self,
        context: FoldContext,
        recent_form_n: int = 10,
    ) -> None:
        self._context = context
        self._policy = Day133CutoffPolicy()
        self._recent_form_n = recent_form_n
        self._features: pd.DataFrame | None = None
        self._frozen: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def is_frozen(self) -> bool:
        """True after freeze() has been called."""
        return self._frozen

    def materialize(self, games_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute team-level season features from raw game results.

        Applies the Day 133 cutoff *before* any aggregation.  Winner/loser-coded
        rows are converted to neutral team-perspective rows before grouping.

        Parameters
        ----------
        games_df : pd.DataFrame
            Raw regular-season detailed results.  Must include all columns in
            ``_REQUIRED_COLS``.

        Returns
        -------
        pd.DataFrame
            One row per (Season, TeamID).  Columns: ``Season``, ``TeamID``,
            and all names in ``FEATURE_NAMES``.

        Raises
        ------
        FrozenStoreError
            If the store has already been frozen.
        ValueError
            If required columns are absent.
        """
        if self._frozen:
            raise FrozenStoreError(
                "RollingFeatureStore is frozen; materialize() is prohibited.  "
                "Create a new store instance to recompute features."
            )

        self._validate_schema(games_df)

        # ----------------------------------------------------------------
        # LAW: cutoff filter BEFORE any aggregation (Day 133 Cutoff Law)
        # ----------------------------------------------------------------
        filtered = self._policy.filter_dataframe(games_df, self._context)

        # Convert winner/loser-coded rows to team-perspective rows
        team_rows = self._pivot_to_team_perspective(filtered)

        # Aggregate per (Season, TeamID)
        features = self._aggregate(team_rows)

        # Compute recent-form win percentage
        recent_form = self._compute_recent_form(team_rows, self._recent_form_n)
        features = features.merge(recent_form, on=["Season", "TeamID"], how="left")

        # Compute first-order strength of schedule
        features = self._compute_sos(team_rows, features)

        self._features = features.reset_index(drop=True)
        return self._features.copy()

    def get_team_features(self, season: int, team_id: int) -> dict[str, float]:
        """
        Return the feature dict for a specific (season, team_id).

        Returns an empty dict if the team/season is not present in the
        materialized data (e.g. team did not play in that season).

        Raises
        ------
        RuntimeError
            If materialize() has not been called yet.
        """
        if self._features is None:
            raise RuntimeError(
                "RollingFeatureStore has not been materialized.  "
                "Call materialize(games_df) before get_team_features()."
            )

        mask = (
            (self._features["Season"] == season)
            & (self._features["TeamID"] == team_id)
        )
        rows = self._features.loc[mask]
        if rows.empty:
            return {}

        row = rows.iloc[0]
        return {
            name: float(row[name])
            for name in self.FEATURE_NAMES
            if name in self._features.columns
        }

    def freeze(self) -> None:
        """
        Permanently freeze this store.

        After ``freeze()``, any call to ``materialize()`` raises
        ``FrozenStoreError``.  Calling ``freeze()`` multiple times is safe
        (idempotent).
        """
        self._frozen = True

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _validate_schema(self, df: pd.DataFrame) -> None:
        missing = self._REQUIRED_COLS - set(df.columns)
        if missing:
            raise ValueError(
                f"RollingFeatureStore: missing required columns: {sorted(missing)}.  "
                f"Present columns: {sorted(df.columns)}."
            )

    def _pivot_to_team_perspective(self, filtered_df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert winner/loser-coded game rows into neutral team-perspective rows.

        Each input row becomes two output rows: one for the winner and one for
        the loser.  This eliminates all outcome-semantic encoding from the
        row structure before any per-team aggregation.
        """
        base = {
            "Season": filtered_df["Season"].values,
            "DayNum": filtered_df["DayNum"].values,
        }

        winner_rows = pd.DataFrame({
            **base,
            "TeamID": filtered_df["WTeamID"].values,
            "OppID": filtered_df["LTeamID"].values,
            "win": 1,
            "pts_scored": filtered_df["WScore"].values,
            "pts_allowed": filtered_df["LScore"].values,
            "margin": (filtered_df["WScore"] - filtered_df["LScore"]).values,
            "fgm": filtered_df["WFGM"].values,
            "fga": filtered_df["WFGA"].values,
            "fgm3": filtered_df["WFGM3"].values,
            "fta": filtered_df["WFTA"].values,
            "oreb": filtered_df["WOR"].values,
            "opp_dreb": filtered_df["LDR"].values,
            "tov": filtered_df["WTO"].values,
        })

        loser_rows = pd.DataFrame({
            **base,
            "TeamID": filtered_df["LTeamID"].values,
            "OppID": filtered_df["WTeamID"].values,
            "win": 0,
            "pts_scored": filtered_df["LScore"].values,
            "pts_allowed": filtered_df["WScore"].values,
            "margin": (filtered_df["LScore"] - filtered_df["WScore"]).values,
            "fgm": filtered_df["LFGM"].values,
            "fga": filtered_df["LFGA"].values,
            "fgm3": filtered_df["LFGM3"].values,
            "fta": filtered_df["LFTA"].values,
            "oreb": filtered_df["LOR"].values,
            "opp_dreb": filtered_df["WDR"].values,
            "tov": filtered_df["LTO"].values,
        })

        team_rows = pd.concat([winner_rows, loser_rows], ignore_index=True)
        return team_rows.sort_values(
            ["Season", "TeamID", "DayNum"]
        ).reset_index(drop=True)

    def _aggregate(self, team_rows: pd.DataFrame) -> pd.DataFrame:
        """Aggregate team-perspective rows into per-team season-summary stats."""
        grp = team_rows.groupby(["Season", "TeamID"], sort=True)

        sums = grp.agg(
            games_played=("win", "count"),
            wins=("win", "sum"),
            pts_scored_sum=("pts_scored", "sum"),
            pts_allowed_sum=("pts_allowed", "sum"),
            scoring_margin_sum=("margin", "sum"),
            fgm_sum=("fgm", "sum"),
            fga_sum=("fga", "sum"),
            fgm3_sum=("fgm3", "sum"),
            fta_sum=("fta", "sum"),
            oreb_sum=("oreb", "sum"),
            opp_dreb_sum=("opp_dreb", "sum"),
            tov_sum=("tov", "sum"),
        ).reset_index()

        n = sums["games_played"]
        fga = sums["fga_sum"]

        sums["win_pct"] = sums["wins"] / n
        sums["points_scored_mean"] = sums["pts_scored_sum"] / n
        sums["points_allowed_mean"] = sums["pts_allowed_sum"] / n
        sums["scoring_margin_mean"] = sums["scoring_margin_sum"] / n

        sums["efg_pct"] = np.where(
            fga > 0,
            (sums["fgm_sum"] + 0.5 * sums["fgm3_sum"]) / fga,
            np.nan,
        )

        tov_denom = fga + 0.44 * sums["fta_sum"] + sums["tov_sum"]
        sums["tov_rate"] = np.where(
            tov_denom > 0,
            sums["tov_sum"] / tov_denom,
            np.nan,
        )

        orb_denom = sums["oreb_sum"] + sums["opp_dreb_sum"]
        sums["orb_rate"] = np.where(
            orb_denom > 0,
            sums["oreb_sum"] / orb_denom,
            np.nan,
        )

        sums["ftr"] = np.where(
            fga > 0,
            sums["fta_sum"] / fga,
            np.nan,
        )

        drop_cols = [
            "pts_scored_sum", "pts_allowed_sum", "scoring_margin_sum",
            "fgm_sum", "fga_sum", "fgm3_sum", "fta_sum",
            "oreb_sum", "opp_dreb_sum", "tov_sum",
        ]
        return sums.drop(columns=drop_cols)

    def _compute_recent_form(
        self,
        team_rows: pd.DataFrame,
        n: int,
    ) -> pd.DataFrame:
        """
        Compute last-N-games win percentage per (Season, TeamID).

        Games are ordered by DayNum ascending; ``tail(n)`` yields the n most
        recent games before the cutoff.
        """
        sorted_rows = team_rows.sort_values(["Season", "TeamID", "DayNum"])
        rows_out: list[dict] = []

        for (season, team_id), grp in sorted_rows.groupby(
            ["Season", "TeamID"], sort=False
        ):
            recent = grp.tail(n)
            rows_out.append({
                "Season": season,
                "TeamID": team_id,
                "recent_form_win_pct": float(recent["win"].mean()),
            })

        return pd.DataFrame(rows_out) if rows_out else pd.DataFrame(
            columns=["Season", "TeamID", "recent_form_win_pct"]
        )

    def _compute_sos(
        self,
        team_rows: pd.DataFrame,
        base_features: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Compute first-order strength of schedule as average opponent win_pct.

        For each (Season, TeamID), averages the win_pct of all opponents faced
        in the filtered game data.  Opponents with no win_pct in the base
        features (should not occur with complete data) are excluded from the
        average; a team with all-missing opponents gets NaN.
        """
        wp_map: dict[tuple[int, int], float] = {}
        for _, row in base_features[["Season", "TeamID", "win_pct"]].iterrows():
            wp_map[(int(row["Season"]), int(row["TeamID"]))] = float(row["win_pct"])

        sos_rows: list[dict] = []
        for (season, team_id), grp in team_rows.groupby(["Season", "TeamID"]):
            opp_wps = [
                wp_map.get((int(season), int(opp)), float("nan"))
                for opp in grp["OppID"].values
            ]
            valid = [x for x in opp_wps if not np.isnan(x)]
            sos_rows.append({
                "Season": season,
                "TeamID": team_id,
                "sos_opp_win_pct": float(np.mean(valid)) if valid else float("nan"),
            })

        sos_df = pd.DataFrame(sos_rows) if sos_rows else pd.DataFrame(
            columns=["Season", "TeamID", "sos_opp_win_pct"]
        )
        return base_features.merge(sos_df, on=["Season", "TeamID"], how="left")
