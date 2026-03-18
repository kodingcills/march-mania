"""
test_materializer.py
====================
Acceptance tests for Step 4: RawTableLoader and DatasetMaterializer.

All DatasetMaterializer tests use inline DataFrames — no filesystem access.
RawTableLoader tests use pytest tmp_path for file I/O.
"""

from __future__ import annotations

import itertools
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ncaa_pipeline.context.fold_context import FoldContext
from ncaa_pipeline.data.datasets import CalDataset, EvalDataset, EvalLabels, TrainDataset
from ncaa_pipeline.data.loader import RawTableLoader
from ncaa_pipeline.data.materializer import DatasetMaterializer


# ---------------------------------------------------------------------------
# Shared test fixtures / helpers
# ---------------------------------------------------------------------------

# Fold used across most materializer tests:
#   train: 2018, cal: 2019, eval: 2021  (2020 is anomaly season — skipped)
_FOLD = FoldContext(
    fold_id="test_fold_v1",
    train_seasons=(2018,),
    cal_season=2019,
    eval_season=2021,
)

# Team IDs used across tests
_TEAMS_4 = [1001, 1002, 1003, 1004]
_TEAMS_5 = [1001, 1002, 1003, 1004, 1005]
_TEAMS_6 = [1001, 1002, 1003, 1004, 1005, 1006]


def _game_row(season: int, w_id: int, l_id: int, day: int = 50) -> dict:
    """Single regular-season game row with valid stats (no div-by-zero)."""
    return {
        "Season": season,
        "DayNum": day,
        "WTeamID": w_id,
        "LTeamID": l_id,
        "WScore": 70,
        "LScore": 60,
        "WFGM": 25,
        "WFGA": 50,
        "WFGM3": 5,
        "WFGA3": 15,
        "WFTM": 10,
        "WFTA": 12,
        "WOR": 5,
        "WDR": 20,
        "WTO": 10,
        "LFGM": 20,
        "LFGA": 50,
        "LFGM3": 3,
        "LFGA3": 15,
        "LFTM": 8,
        "LFTA": 10,
        "LOR": 4,
        "LDR": 22,
        "LTO": 12,
    }


def _make_regular_season_df(teams: list[int], seasons: list[int]) -> pd.DataFrame:
    """Round-robin games between all team pairs for each season. DayNum=50 (< 133)."""
    rows = []
    for season in seasons:
        for t1, t2 in itertools.combinations(sorted(teams), 2):
            rows.append(_game_row(season, w_id=t1, l_id=t2))
    return pd.DataFrame(rows)


def _make_seeds_df(teams: list[int], seasons: list[int]) -> pd.DataFrame:
    """Assign seeds W01, W02, ... to teams in sorted order for each season."""
    regions = ["W", "X", "Y", "Z"]
    rows = []
    for season in seasons:
        for i, team_id in enumerate(sorted(teams)):
            region = regions[i % 4]
            seed_num = i // 4 + 1
            rows.append(
                {"Season": season, "TeamID": team_id, "Seed": f"{region}{seed_num:02d}"}
            )
    return pd.DataFrame(rows)


def _make_labeled_games_df(games_by_season: dict[int, list[tuple[int, int]]]) -> pd.DataFrame:
    """games_by_season: {season: [(winner_id, loser_id), ...]}"""
    rows = []
    for season, games in games_by_season.items():
        for w_id, l_id in games:
            rows.append({"Season": season, "DayNum": 136, "WTeamID": w_id, "LTeamID": l_id})
    return pd.DataFrame(rows)


def _make_default_inputs(
    teams: list[int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, FoldContext]:
    """Return (regular_season_df, seeds_df, labeled_games_df, fold_context)."""
    if teams is None:
        teams = _TEAMS_4
    seasons = [2018, 2019, 2021]
    reg_df = _make_regular_season_df(teams, seasons)
    seeds_df = _make_seeds_df(teams, seasons)
    labeled_df = _make_labeled_games_df(
        {
            2018: [(teams[0], teams[1]), (teams[2], teams[3])],
            2019: [(teams[0], teams[1]), (teams[2], teams[3])],
            2021: [(teams[0], teams[1]), (teams[2], teams[3])],
        }
    )
    return reg_df, seeds_df, labeled_df, _FOLD


# ---------------------------------------------------------------------------
# TestRawTableLoader
# ---------------------------------------------------------------------------


class TestRawTableLoader:
    def _write_csv(self, tmp_path: Path, name: str, df: pd.DataFrame) -> Path:
        p = tmp_path / name
        df.to_csv(p, index=False)
        return p

    def test_load_regular_season_detailed_roundtrip(self, tmp_path: Path) -> None:
        loader = RawTableLoader()
        cols = list(RawTableLoader.REGULAR_SEASON_DETAILED_REQUIRED)
        df = pd.DataFrame([[1] * len(cols)] * 3, columns=cols)
        path = self._write_csv(tmp_path, "reg.csv", df)
        result = loader.load_regular_season_detailed(path)
        assert list(result.columns) == list(df.columns)
        assert len(result) == 3

    def test_load_regular_season_detailed_accepts_str_path(self, tmp_path: Path) -> None:
        loader = RawTableLoader()
        cols = list(RawTableLoader.REGULAR_SEASON_DETAILED_REQUIRED)
        df = pd.DataFrame([[1] * len(cols)] * 2, columns=cols)
        path = self._write_csv(tmp_path, "reg.csv", df)
        result = loader.load_regular_season_detailed(str(path))  # str, not Path
        assert len(result) == 2

    def test_load_regular_season_detailed_missing_col_raises(self, tmp_path: Path) -> None:
        loader = RawTableLoader()
        cols = list(RawTableLoader.REGULAR_SEASON_DETAILED_REQUIRED)
        df = pd.DataFrame([[1] * len(cols)] * 2, columns=cols)
        df = df.drop(columns=[cols[0]])
        path = self._write_csv(tmp_path, "reg_bad.csv", df)
        with pytest.raises(ValueError, match="missing required columns"):
            loader.load_regular_season_detailed(path)

    def test_load_massey_ordinals_roundtrip(self, tmp_path: Path) -> None:
        loader = RawTableLoader()
        cols = list(RawTableLoader.MASSEY_REQUIRED)
        df = pd.DataFrame([[1] * len(cols)] * 4, columns=cols)
        path = self._write_csv(tmp_path, "massey.csv", df)
        result = loader.load_massey_ordinals(path)
        assert set(result.columns) >= RawTableLoader.MASSEY_REQUIRED
        assert len(result) == 4

    def test_load_massey_ordinals_missing_col_raises(self, tmp_path: Path) -> None:
        loader = RawTableLoader()
        cols = list(RawTableLoader.MASSEY_REQUIRED)
        df = pd.DataFrame([[1] * len(cols)] * 2, columns=cols)
        df = df.drop(columns=[cols[0]])
        path = self._write_csv(tmp_path, "massey_bad.csv", df)
        with pytest.raises(ValueError, match="missing required columns"):
            loader.load_massey_ordinals(path)

    def test_load_tournament_seeds_roundtrip(self, tmp_path: Path) -> None:
        loader = RawTableLoader()
        df = pd.DataFrame(
            {"Season": [2024, 2024], "Seed": ["W01", "X01"], "TeamID": [1001, 1002]}
        )
        path = self._write_csv(tmp_path, "seeds.csv", df)
        result = loader.load_tournament_seeds(path)
        assert set(result.columns) >= {"Season", "Seed", "TeamID"}
        assert len(result) == 2

    def test_load_tournament_seeds_missing_col_raises(self, tmp_path: Path) -> None:
        loader = RawTableLoader()
        df = pd.DataFrame({"Season": [2024], "Seed": ["W01"]})  # missing TeamID
        path = self._write_csv(tmp_path, "seeds_bad.csv", df)
        with pytest.raises(ValueError, match="missing required columns"):
            loader.load_tournament_seeds(path)

    def test_load_labeled_games_roundtrip(self, tmp_path: Path) -> None:
        loader = RawTableLoader()
        df = pd.DataFrame(
            {
                "Season": [2024, 2024],
                "DayNum": [136, 138],
                "WTeamID": [1001, 1003],
                "LTeamID": [1002, 1004],
            }
        )
        path = self._write_csv(tmp_path, "games.csv", df)
        result = loader.load_labeled_games(path)
        assert len(result) == 2

    def test_load_labeled_games_missing_col_raises(self, tmp_path: Path) -> None:
        loader = RawTableLoader()
        df = pd.DataFrame(
            {"Season": [2024], "DayNum": [136], "WTeamID": [1001]}
        )  # missing LTeamID
        path = self._write_csv(tmp_path, "games_bad.csv", df)
        with pytest.raises(ValueError, match="missing required columns"):
            loader.load_labeled_games(path)

    def test_load_labeled_games_accepts_path_and_str(self, tmp_path: Path) -> None:
        loader = RawTableLoader()
        df = pd.DataFrame(
            {"Season": [2024], "DayNum": [136], "WTeamID": [1001], "LTeamID": [1002]}
        )
        path = self._write_csv(tmp_path, "games.csv", df)
        r1 = loader.load_labeled_games(path)
        r2 = loader.load_labeled_games(str(path))
        assert len(r1) == len(r2) == 1

    def test_returned_dataframe_is_not_modified(self, tmp_path: Path) -> None:
        loader = RawTableLoader()
        df = pd.DataFrame(
            {
                "Season": [2024, 2023],
                "DayNum": [136, 134],
                "WTeamID": [1001, 1003],
                "LTeamID": [1002, 1004],
            }
        )
        path = self._write_csv(tmp_path, "games.csv", df)
        result = loader.load_labeled_games(path)
        # Verify the values are unchanged
        assert result["WTeamID"].tolist() == [1001, 1003]
        assert result["LTeamID"].tolist() == [1002, 1004]


# ---------------------------------------------------------------------------
# TestDatasetMaterializerZones
# ---------------------------------------------------------------------------


class TestDatasetMaterializerZones:
    def test_materialize_fold_returns_3_tuple(self) -> None:
        m = DatasetMaterializer()
        reg, seeds, labeled, fold = _make_default_inputs()
        result = m.materialize_fold(reg, seeds, labeled, fold)
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_materialize_fold_return_types(self) -> None:
        m = DatasetMaterializer()
        reg, seeds, labeled, fold = _make_default_inputs()
        train, cal, eval_ds = m.materialize_fold(reg, seeds, labeled, fold)
        assert isinstance(train, TrainDataset)
        assert isinstance(cal, CalDataset)
        assert isinstance(eval_ds, EvalDataset)

    def test_materialize_fold_does_not_return_eval_labels(self) -> None:
        m = DatasetMaterializer()
        reg, seeds, labeled, fold = _make_default_inputs()
        result = m.materialize_fold(reg, seeds, labeled, fold)
        for item in result:
            assert not isinstance(item, EvalLabels), (
                "materialize_fold must never return EvalLabels"
            )

    def test_eval_dataset_has_no_y_attribute(self) -> None:
        m = DatasetMaterializer()
        reg, seeds, labeled, fold = _make_default_inputs()
        _, _, eval_ds = m.materialize_fold(reg, seeds, labeled, fold)
        with pytest.raises(AttributeError):
            _ = eval_ds.y  # type: ignore[attr-defined]

    def test_eval_labels_only_from_materialize_eval_labels(self) -> None:
        m = DatasetMaterializer()
        _, _, labeled, fold = _make_default_inputs()
        result = m.materialize_eval_labels(labeled, fold)
        assert isinstance(result, EvalLabels)

    def test_all_zones_carry_fold_id(self) -> None:
        m = DatasetMaterializer()
        reg, seeds, labeled, fold = _make_default_inputs()
        train, cal, eval_ds = m.materialize_fold(reg, seeds, labeled, fold)
        assert train.fold_id == fold.fold_id
        assert cal.fold_id == fold.fold_id
        assert eval_ds.fold_id == fold.fold_id

    def test_eval_labels_carries_fold_id(self) -> None:
        m = DatasetMaterializer()
        _, _, labeled, fold = _make_default_inputs()
        labels = m.materialize_eval_labels(labeled, fold)
        assert labels.fold_id == fold.fold_id

    def test_zone_a_matchup_ids_from_train_seasons_only(self) -> None:
        m = DatasetMaterializer()
        reg, seeds, labeled, fold = _make_default_inputs()
        train, _, _ = m.materialize_fold(reg, seeds, labeled, fold)
        for mid in train.matchup_ids:
            season = int(mid.split("_")[0])
            assert season in fold.train_seasons, (
                f"Zone A matchup_id {mid!r} references season {season} "
                f"not in train_seasons {fold.train_seasons}"
            )

    def test_zone_b_matchup_ids_from_cal_season_only(self) -> None:
        m = DatasetMaterializer()
        reg, seeds, labeled, fold = _make_default_inputs()
        _, cal, _ = m.materialize_fold(reg, seeds, labeled, fold)
        for mid in cal.matchup_ids:
            season = int(mid.split("_")[0])
            assert season == fold.cal_season, (
                f"Zone B matchup_id {mid!r} references season {season}, "
                f"expected cal_season={fold.cal_season}"
            )

    def test_zone_c_matchup_ids_reference_eval_season(self) -> None:
        m = DatasetMaterializer()
        reg, seeds, labeled, fold = _make_default_inputs()
        _, _, eval_ds = m.materialize_fold(reg, seeds, labeled, fold)
        for mid in eval_ds.matchup_ids:
            season = int(mid.split("_")[0])
            assert season == fold.eval_season


# ---------------------------------------------------------------------------
# TestZoneCAllPairs
# ---------------------------------------------------------------------------


class TestZoneCAllPairs:
    @pytest.mark.parametrize("k", [4, 5, 6])
    def test_zone_c_generates_all_pairs(self, k: int) -> None:
        teams = list(range(1001, 1001 + k))
        expected_pairs = k * (k - 1) // 2  # C(k, 2)

        seasons = [2018, 2019, 2021]
        reg_df = _make_regular_season_df(teams, seasons)
        seeds_df = _make_seeds_df(teams, seasons)
        labeled_df = _make_labeled_games_df(
            {
                2018: [(teams[0], teams[1])],
                2019: [(teams[0], teams[1])],
                2021: [(teams[0], teams[1])],
            }
        )

        m = DatasetMaterializer()
        _, _, eval_ds = m.materialize_fold(reg_df, seeds_df, labeled_df, _FOLD)
        assert len(eval_ds.matchup_ids) == expected_pairs, (
            f"Expected C({k},2)={expected_pairs} pairs, got {len(eval_ds.matchup_ids)}"
        )

    def test_zone_c_all_pairs_have_canonical_ordering(self) -> None:
        reg, seeds, labeled, fold = _make_default_inputs()
        m = DatasetMaterializer()
        _, _, eval_ds = m.materialize_fold(reg, seeds, labeled, fold)
        for mid in eval_ds.matchup_ids:
            parts = mid.split("_")
            t1, t2 = int(parts[1]), int(parts[2])
            assert t1 < t2, f"Pair {mid!r} violates canonical ordering (team1_id < team2_id)"

    def test_zone_c_not_restricted_to_realized_games(self) -> None:
        """Zone C must include all C(K,2) pairs even when only a few games were realized."""
        k = 5
        teams = list(range(1001, 1001 + k))
        expected_pairs = k * (k - 1) // 2  # 10

        seasons = [2018, 2019, 2021]
        reg_df = _make_regular_season_df(teams, seasons)
        seeds_df = _make_seeds_df(teams, seasons)

        # Only 1 realized game in the eval season — Zone C must still have C(5,2)=10 rows
        labeled_df = _make_labeled_games_df(
            {
                2018: [(teams[0], teams[1])],
                2019: [(teams[0], teams[1])],
                2021: [(teams[0], teams[1])],  # only 1 realized game, not 10
            }
        )

        m = DatasetMaterializer()
        _, _, eval_ds = m.materialize_fold(reg_df, seeds_df, labeled_df, _FOLD)
        assert len(eval_ds.matchup_ids) == expected_pairs

    def test_zone_c_row_order_is_deterministic(self) -> None:
        reg, seeds, labeled, fold = _make_default_inputs(_TEAMS_5)
        m = DatasetMaterializer()
        _, _, eval_ds1 = m.materialize_fold(reg, seeds, labeled, fold)
        _, _, eval_ds2 = m.materialize_fold(reg, seeds, labeled, fold)
        np.testing.assert_array_equal(eval_ds1.matchup_ids, eval_ds2.matchup_ids)

    def test_zone_c_sorted_by_team_ids(self) -> None:
        reg, seeds, labeled, fold = _make_default_inputs()
        m = DatasetMaterializer()
        _, _, eval_ds = m.materialize_fold(reg, seeds, labeled, fold)
        pairs = [
            (int(mid.split("_")[1]), int(mid.split("_")[2]))
            for mid in eval_ds.matchup_ids
        ]
        assert pairs == sorted(pairs), "Zone C rows must be sorted by (team1_id, team2_id)"


# ---------------------------------------------------------------------------
# TestFeatureNameConsistency
# ---------------------------------------------------------------------------


class TestFeatureNameConsistency:
    def test_feature_names_identical_across_all_zones(self) -> None:
        reg, seeds, labeled, fold = _make_default_inputs()
        m = DatasetMaterializer()
        train, cal, eval_ds = m.materialize_fold(reg, seeds, labeled, fold)
        assert train.feature_names == cal.feature_names
        assert train.feature_names == eval_ds.feature_names

    def test_feature_names_consistent_without_massey(self) -> None:
        reg, seeds, labeled, fold = _make_default_inputs()
        m = DatasetMaterializer()
        train, cal, eval_ds = m.materialize_fold(
            reg, seeds, labeled, fold, massey_df=None, massey_systems=None
        )
        assert train.feature_names == cal.feature_names == eval_ds.feature_names
        # seed_diff should be present (seeds_df was provided)
        assert "seed_diff" in train.feature_names

    def test_feature_names_include_seed_diff_when_seeds_present(self) -> None:
        reg, seeds, labeled, fold = _make_default_inputs()
        m = DatasetMaterializer()
        train, _, _ = m.materialize_fold(reg, seeds, labeled, fold)
        assert "seed_diff" in train.feature_names

    def test_feature_matrix_width_matches_feature_names(self) -> None:
        reg, seeds, labeled, fold = _make_default_inputs()
        m = DatasetMaterializer()
        train, cal, eval_ds = m.materialize_fold(reg, seeds, labeled, fold)
        assert train.X.shape[1] == len(train.feature_names)
        assert cal.X.shape[1] == len(cal.feature_names)
        assert eval_ds.X.shape[1] == len(eval_ds.feature_names)


# ---------------------------------------------------------------------------
# TestDeterminism
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_repeated_calls_produce_identical_outputs(self) -> None:
        reg, seeds, labeled, fold = _make_default_inputs()
        m = DatasetMaterializer()

        train1, cal1, eval1 = m.materialize_fold(reg, seeds, labeled, fold)
        train2, cal2, eval2 = m.materialize_fold(reg, seeds, labeled, fold)

        np.testing.assert_array_equal(train1.X, train2.X)
        np.testing.assert_array_equal(train1.y, train2.y)
        np.testing.assert_array_equal(train1.matchup_ids, train2.matchup_ids)

        np.testing.assert_array_equal(cal1.X, cal2.X)
        np.testing.assert_array_equal(cal1.y, cal2.y)
        np.testing.assert_array_equal(cal1.matchup_ids, cal2.matchup_ids)

        np.testing.assert_array_equal(eval1.X, eval2.X)
        np.testing.assert_array_equal(eval1.matchup_ids, eval2.matchup_ids)

    def test_matchup_ids_sorted_ascending(self) -> None:
        reg, seeds, labeled, fold = _make_default_inputs()
        m = DatasetMaterializer()
        train, cal, _ = m.materialize_fold(reg, seeds, labeled, fold)
        # Sorted as strings
        assert list(train.matchup_ids) == sorted(train.matchup_ids.tolist())
        assert list(cal.matchup_ids) == sorted(cal.matchup_ids.tolist())


# ---------------------------------------------------------------------------
# TestNoArchitectureCreep
# ---------------------------------------------------------------------------


class TestNoArchitectureCreep:
    def test_materializer_does_not_import_ml_libraries(self) -> None:
        import sys

        import ncaa_pipeline.data.materializer  # noqa: F401

        forbidden = {"lightgbm", "sklearn", "mlflow", "aim", "torch", "tensorflow"}
        imported = set(sys.modules.keys())
        violations = forbidden & imported
        assert not violations, (
            f"materializer.py pulled in forbidden ML libraries: {sorted(violations)}"
        )

    def test_loader_does_not_import_ml_libraries(self) -> None:
        import sys

        import ncaa_pipeline.data.loader  # noqa: F401

        forbidden = {"lightgbm", "sklearn", "mlflow", "aim", "torch", "tensorflow"}
        imported = set(sys.modules.keys())
        violations = forbidden & imported
        assert not violations, (
            f"loader.py pulled in forbidden ML libraries: {sorted(violations)}"
        )
