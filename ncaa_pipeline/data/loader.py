"""
loader.py
=========
Raw table loading and schema validation.

Loads the four CSV tables required by the NCAA pipeline.  Each method
reads a CSV, validates that required columns are present, and returns
the DataFrame unchanged — no transformation, no feature engineering.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ncaa_pipeline.features.massey_extractor import MasseyOrdinalExtractor
from ncaa_pipeline.features.rolling_store import RollingFeatureStore


class RawTableLoader:
    """
    Stateless loader for the four raw NCAA CSV tables.

    Each load method:
      1. Reads the CSV at the given path.
      2. Validates that all required columns are present.
      3. Returns the DataFrame unchanged.

    No caching.  No transformation.  No feature engineering.
    """

    REGULAR_SEASON_DETAILED_REQUIRED: frozenset[str] = RollingFeatureStore._REQUIRED_COLS
    MASSEY_REQUIRED: frozenset[str] = MasseyOrdinalExtractor._REQUIRED_COLS
    TOURNEY_SEEDS_REQUIRED: frozenset[str] = frozenset({"Season", "Seed", "TeamID"})
    LABELED_GAMES_REQUIRED: frozenset[str] = frozenset({"Season", "DayNum", "WTeamID", "LTeamID"})

    def load_regular_season_detailed(self, path: Path | str) -> pd.DataFrame:
        """Load MRegularSeasonDetailedResults.csv."""
        return self._load_and_validate(
            path, self.REGULAR_SEASON_DETAILED_REQUIRED, "load_regular_season_detailed"
        )

    def load_massey_ordinals(self, path: Path | str) -> pd.DataFrame:
        """Load MMasseyOrdinals.csv."""
        return self._load_and_validate(
            path, self.MASSEY_REQUIRED, "load_massey_ordinals"
        )

    def load_tournament_seeds(self, path: Path | str) -> pd.DataFrame:
        """Load MNCAATourneySeeds.csv."""
        return self._load_and_validate(
            path, self.TOURNEY_SEEDS_REQUIRED, "load_tournament_seeds"
        )

    def load_labeled_games(self, path: Path | str) -> pd.DataFrame:
        """Load a labeled-games CSV (e.g. MNCAATourneyCompactResults.csv)."""
        return self._load_and_validate(
            path, self.LABELED_GAMES_REQUIRED, "load_labeled_games"
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_and_validate(
        self,
        path: Path | str,
        required_cols: frozenset[str],
        method_name: str,
    ) -> pd.DataFrame:
        df = pd.read_csv(path)
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(
                f"RawTableLoader.{method_name}: missing required columns: "
                f"{sorted(missing)}.  Present columns: {sorted(df.columns)}."
            )
        return df
