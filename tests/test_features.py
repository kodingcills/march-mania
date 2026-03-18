"""
test_features.py
================
Step 3 feature-layer tests: prove cutoff laws, symmetry, LAD logic, and
freeze semantics hold before advancing to Step 4.

All tests are:
  - deterministic
  - isolated (no shared mutable state)
  - self-contained (inline DataFrames only, no external files)
  - free of labels, dataset contracts, and model/calibration logic

Acceptance criteria verified here (per PHASE_PLAN Step 3):
  1. materialize() applies cutoff filter before aggregation
  2. Post-cutoff rows do not influence pre-cutoff features
  3. freeze() blocks rebuild/mutation
  4. Winner/loser-coded rows are correctly pivoted to team-perspective
  5. safe_snapshot() never returns RankingDayNum >= cutoff_day
  6. LAD fallback picks the latest available ranking below cutoff
  7. Stale systems are surfaced explicitly via OrdinalStatus
  8. FeatureAssembler enforces team1_id < team2_id
  9. Additive (sum_*) features are invariant under input team-order reversal
  10. Differential (diff_*) features are consistent with canonical order
  11. No Step 3 code attaches labels or returns Step 2 dataset contract objects
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from ncaa_pipeline.context.fold_context import FoldContext
from ncaa_pipeline.data.datasets import CalDataset, EvalDataset, EvalLabels, TrainDataset
from ncaa_pipeline.features.assembler import AssembledFeatures, FeatureAssembler
from ncaa_pipeline.features.massey_extractor import (
    MasseyOrdinalExtractor,
    OrdinalSnapshot,
    OrdinalStatus,
)
from ncaa_pipeline.features.rolling_store import FrozenStoreError, RollingFeatureStore


# ===========================================================================
# Shared helpers
# ===========================================================================

def _make_context(
    fold_id: str = "test-step3",
    train_seasons: tuple[int, ...] = (2018, 2019, 2021),
    cal_season: int = 2022,
    eval_season: int = 2023,
) -> FoldContext:
    return FoldContext(
        fold_id=fold_id,
        train_seasons=train_seasons,
        cal_season=cal_season,
        eval_season=eval_season,
    )


def _game(
    season: int,
    day_num: int,
    w_id: int,
    l_id: int,
    w_score: int = 80,
    l_score: int = 70,
    wfgm: int = 30,
    wfga: int = 60,
    wfgm3: int = 5,
    wfga3: int = 15,
    wftm: int = 10,
    wfta: int = 14,
    wor: int = 10,
    wdr: int = 25,
    wto: int = 12,
    lfgm: int = 25,
    lfga: int = 55,
    lfgm3: int = 3,
    lfga3: int = 12,
    lftm: int = 12,
    lfta: int = 16,
    lor: int = 8,
    ldr: int = 22,
    lto: int = 14,
) -> dict:
    """Return a minimal valid detailed results game row dict."""
    return {
        "Season": season, "DayNum": day_num,
        "WTeamID": w_id, "LTeamID": l_id,
        "WScore": w_score, "LScore": l_score,
        "WFGM": wfgm, "WFGA": wfga, "WFGM3": wfgm3, "WFGA3": wfga3,
        "WFTM": wftm, "WFTA": wfta, "WOR": wor, "WDR": wdr, "WTO": wto,
        "LFGM": lfgm, "LFGA": lfga, "LFGM3": lfgm3, "LFGA3": lfga3,
        "LFTM": lftm, "LFTA": lfta, "LOR": lor, "LDR": ldr, "LTO": lto,
    }


def _games_df(*rows: dict) -> pd.DataFrame:
    return pd.DataFrame(list(rows))


def _massey_row(
    season: int,
    ranking_day: int,
    system: str,
    team_id: int,
    rank: float,
) -> dict:
    return {
        "Season": season,
        "RankingDayNum": ranking_day,
        "SystemName": system,
        "TeamID": team_id,
        "OrdinalRank": rank,
    }


# ===========================================================================
# 1-3: RollingFeatureStore — cutoff enforcement and freeze
# ===========================================================================

class TestRollingStoreCutoff:
    """Cutoff law: post-Day-133 rows must never influence aggregated features."""

    def _store_and_context(self) -> tuple[RollingFeatureStore, FoldContext]:
        ctx = _make_context()
        store = RollingFeatureStore(ctx)
        return store, ctx

    def test_post_cutoff_row_excluded_from_games_played(self) -> None:
        """
        A game at DayNum=140 must not appear in games_played.

        Team 1001 plays 3 pre-cutoff games (all wins) and 1 post-cutoff game
        (a loss).  After materialization games_played must be 3, not 4.
        """
        store, _ = self._store_and_context()
        games = _games_df(
            _game(season=2022, day_num=50, w_id=1001, l_id=1002),
            _game(season=2022, day_num=60, w_id=1001, l_id=1003),
            _game(season=2022, day_num=70, w_id=1001, l_id=1004),
            # post-cutoff loss — must be excluded
            _game(season=2022, day_num=140, w_id=1002, l_id=1001),
        )
        features = store.materialize(games)
        row = features.loc[
            (features["Season"] == 2022) & (features["TeamID"] == 1001)
        ].iloc[0]

        assert int(row["games_played"]) == 3
        assert float(row["wins"]) == 3.0
        assert float(row["win_pct"]) == pytest.approx(1.0)

    def test_post_cutoff_row_does_not_change_pre_cutoff_features(self) -> None:
        """
        Materializing with vs without post-cutoff rows must yield identical
        pre-cutoff feature values for every team.
        """
        ctx = _make_context()
        base_games = _games_df(
            _game(season=2022, day_num=50, w_id=1001, l_id=1002),
            _game(season=2022, day_num=60, w_id=1002, l_id=1003),
        )
        contaminated_games = _games_df(
            _game(season=2022, day_num=50, w_id=1001, l_id=1002),
            _game(season=2022, day_num=60, w_id=1002, l_id=1003),
            # post-cutoff injections
            _game(season=2022, day_num=133, w_id=1001, l_id=1003),
            _game(season=2022, day_num=150, w_id=1002, l_id=1001),
        )

        store_base = RollingFeatureStore(ctx)
        feats_base = store_base.materialize(base_games).set_index(["Season", "TeamID"])

        store_cont = RollingFeatureStore(ctx)
        feats_cont = store_cont.materialize(contaminated_games).set_index(["Season", "TeamID"])

        for team_id in (1001, 1002, 1003):
            for col in RollingFeatureStore.FEATURE_NAMES:
                v_base = feats_base.loc[(2022, team_id), col]
                v_cont = feats_cont.loc[(2022, team_id), col]
                # Both may be NaN (e.g. for teams with 1 game and no SOS)
                if math.isnan(float(v_base)) and math.isnan(float(v_cont)):
                    continue
                assert float(v_base) == pytest.approx(float(v_cont)), (
                    f"Feature '{col}' for team {team_id} differs: "
                    f"base={v_base}, contaminated={v_cont}"
                )

    def test_day_133_row_is_excluded(self) -> None:
        """DayNum==133 is at or beyond the cutoff and must be excluded."""
        ctx = _make_context()
        store = RollingFeatureStore(ctx)
        games = _games_df(
            _game(season=2022, day_num=130, w_id=1001, l_id=1002),
            _game(season=2022, day_num=133, w_id=1001, l_id=1002),  # at cutoff
        )
        features = store.materialize(games)
        row_1001 = features.loc[
            (features["Season"] == 2022) & (features["TeamID"] == 1001)
        ].iloc[0]
        assert int(row_1001["games_played"]) == 1


class TestRollingStoreFreeze:
    """Freeze law: frozen stores must reject materialize()."""

    def test_freeze_blocks_materialize(self) -> None:
        ctx = _make_context()
        store = RollingFeatureStore(ctx)
        games = _games_df(_game(season=2022, day_num=50, w_id=1001, l_id=1002))
        store.materialize(games)
        store.freeze()

        with pytest.raises(FrozenStoreError):
            store.materialize(games)

    def test_freeze_is_idempotent(self) -> None:
        ctx = _make_context()
        store = RollingFeatureStore(ctx)
        store.freeze()
        store.freeze()  # must not raise
        assert store.is_frozen

    def test_not_frozen_by_default(self) -> None:
        ctx = _make_context()
        store = RollingFeatureStore(ctx)
        assert not store.is_frozen

    def test_freeze_blocks_rebuild_after_first_materialize(self) -> None:
        """Even after a successful materialize, freeze must prevent re-compute."""
        ctx = _make_context()
        store = RollingFeatureStore(ctx)
        games = _games_df(_game(season=2022, day_num=50, w_id=1001, l_id=1002))
        store.materialize(games)
        store.freeze()

        extra_games = _games_df(
            _game(season=2022, day_num=50, w_id=1001, l_id=1002),
            _game(season=2022, day_num=60, w_id=1001, l_id=1003),
        )
        with pytest.raises(FrozenStoreError):
            store.materialize(extra_games)


# ===========================================================================
# 4: Winner/loser conversion correctness
# ===========================================================================

class TestWinnerLoserConversion:
    """Winner/loser-coded rows must become correct team-perspective records."""

    def _materialize_single_game(
        self,
        w_score: int = 80,
        l_score: int = 70,
    ) -> pd.DataFrame:
        ctx = _make_context()
        store = RollingFeatureStore(ctx)
        games = _games_df(
            _game(
                season=2022, day_num=50,
                w_id=1001, l_id=1002,
                w_score=w_score, l_score=l_score,
            )
        )
        return store.materialize(games)

    def test_winner_gets_win_count_1(self) -> None:
        feats = self._materialize_single_game()
        row = feats.loc[(feats["Season"] == 2022) & (feats["TeamID"] == 1001)].iloc[0]
        assert float(row["wins"]) == 1.0

    def test_loser_gets_win_count_0(self) -> None:
        feats = self._materialize_single_game()
        row = feats.loc[(feats["Season"] == 2022) & (feats["TeamID"] == 1002)].iloc[0]
        assert float(row["wins"]) == 0.0

    def test_winner_points_scored_equals_winner_score(self) -> None:
        feats = self._materialize_single_game(w_score=85, l_score=72)
        row = feats.loc[(feats["Season"] == 2022) & (feats["TeamID"] == 1001)].iloc[0]
        assert float(row["points_scored_mean"]) == pytest.approx(85.0)
        assert float(row["points_allowed_mean"]) == pytest.approx(72.0)

    def test_loser_points_scored_equals_loser_score(self) -> None:
        feats = self._materialize_single_game(w_score=85, l_score=72)
        row = feats.loc[(feats["Season"] == 2022) & (feats["TeamID"] == 1002)].iloc[0]
        assert float(row["points_scored_mean"]) == pytest.approx(72.0)
        assert float(row["points_allowed_mean"]) == pytest.approx(85.0)

    def test_both_teams_appear_in_output(self) -> None:
        feats = self._materialize_single_game()
        season_feats = feats.loc[feats["Season"] == 2022]
        team_ids = set(season_feats["TeamID"].tolist())
        assert 1001 in team_ids
        assert 1002 in team_ids

    def test_games_played_both_teams_is_1(self) -> None:
        feats = self._materialize_single_game()
        for team_id in (1001, 1002):
            row = feats.loc[
                (feats["Season"] == 2022) & (feats["TeamID"] == team_id)
            ].iloc[0]
            assert int(row["games_played"]) == 1

    def test_winner_scoring_margin_is_positive(self) -> None:
        feats = self._materialize_single_game(w_score=80, l_score=70)
        row_w = feats.loc[(feats["Season"] == 2022) & (feats["TeamID"] == 1001)].iloc[0]
        row_l = feats.loc[(feats["Season"] == 2022) & (feats["TeamID"] == 1002)].iloc[0]
        assert float(row_w["scoring_margin_mean"]) > 0.0
        assert float(row_l["scoring_margin_mean"]) < 0.0


# ===========================================================================
# 5-7: MasseyOrdinalExtractor — LAD, cutoff, staleness
# ===========================================================================

class TestMasseyExtractorCutoff:
    """safe_snapshot() must never return a ranking at or beyond cutoff_day."""

    def test_snapshot_at_cutoff_day_is_excluded(self) -> None:
        """A ranking on DayNum == cutoff_day must be excluded."""
        df = pd.DataFrame([
            _massey_row(2022, 130, "MOR", 1001, rank=10.0),
            _massey_row(2022, 133, "MOR", 1001, rank=5.0),  # at cutoff — forbidden
        ])
        extractor = MasseyOrdinalExtractor(df)
        snap = extractor.safe_snapshot(2022, 1001, 133, "MOR")

        assert snap.ranking_day < 133
        assert snap.ranking_day == 130
        assert snap.ordinal_rank == pytest.approx(10.0)

    def test_snapshot_beyond_cutoff_day_is_excluded(self) -> None:
        """A ranking on DayNum > cutoff_day must be excluded."""
        df = pd.DataFrame([
            _massey_row(2022, 128, "MOR", 1001, rank=15.0),
            _massey_row(2022, 140, "MOR", 1001, rank=3.0),  # well beyond cutoff
        ])
        extractor = MasseyOrdinalExtractor(df)
        snap = extractor.safe_snapshot(2022, 1001, 133, "MOR")

        assert snap.ranking_day == 128
        assert snap.ordinal_rank == pytest.approx(15.0)


class TestMasseyExtractorLAD:
    """LAD logic: picks the latest available ranking strictly below cutoff."""

    def test_lad_picks_highest_valid_day(self) -> None:
        """Among days 100, 115, 125, LAD must pick 125."""
        df = pd.DataFrame([
            _massey_row(2022, 100, "MOR", 1001, rank=20.0),
            _massey_row(2022, 115, "MOR", 1001, rank=15.0),
            _massey_row(2022, 125, "MOR", 1001, rank=10.0),
        ])
        extractor = MasseyOrdinalExtractor(df)
        snap = extractor.safe_snapshot(2022, 1001, 133, "MOR")

        assert snap.ranking_day == 125
        assert snap.ordinal_rank == pytest.approx(10.0)

    def test_lad_ignores_post_cutoff_entry(self) -> None:
        """When a later entry exists at DayNum=135, LAD must still pick 125."""
        df = pd.DataFrame([
            _massey_row(2022, 100, "MOR", 1001, rank=20.0),
            _massey_row(2022, 125, "MOR", 1001, rank=10.0),
            _massey_row(2022, 135, "MOR", 1001, rank=2.0),  # beyond cutoff
        ])
        extractor = MasseyOrdinalExtractor(df)
        snap = extractor.safe_snapshot(2022, 1001, 133, "MOR")

        assert snap.ranking_day == 125
        assert snap.ordinal_rank == pytest.approx(10.0)

    def test_missing_team_returns_critical(self) -> None:
        """No data for team → CRITICAL status, NaN rank, ranking_day == -1."""
        df = pd.DataFrame([
            _massey_row(2022, 120, "MOR", 9999, rank=50.0),
        ])
        extractor = MasseyOrdinalExtractor(df)
        snap = extractor.safe_snapshot(2022, 1001, 133, "MOR")

        assert snap.status == OrdinalStatus.CRITICAL
        assert math.isnan(snap.ordinal_rank)
        assert snap.ranking_day == -1
        assert not snap.is_available

    def test_missing_system_returns_critical(self) -> None:
        """No data for the requested system → CRITICAL."""
        df = pd.DataFrame([
            _massey_row(2022, 120, "SAG", 1001, rank=5.0),
        ])
        extractor = MasseyOrdinalExtractor(df)
        snap = extractor.safe_snapshot(2022, 1001, 133, "MOR")

        assert snap.status == OrdinalStatus.CRITICAL

    def test_snapshot_day_is_strictly_less_than_cutoff(self) -> None:
        """ranking_day must be < cutoff_day in all non-missing snapshots."""
        df = pd.DataFrame([
            _massey_row(2022, day, "MOR", 1001, rank=float(i))
            for i, day in enumerate([50, 80, 100, 120, 130])
        ])
        extractor = MasseyOrdinalExtractor(df)
        snap = extractor.safe_snapshot(2022, 1001, 133, "MOR")

        assert snap.ranking_day < 133
        assert snap.is_available


class TestMasseyExtractorStaleness:
    """Stale systems must be classified with WARNING or CRITICAL status."""

    def test_ok_status_for_fresh_snapshot(self) -> None:
        """Staleness within warning threshold → OK."""
        # default warning=14: day 125 → staleness = 133-125 = 8 ≤ 14 → OK
        df = pd.DataFrame([_massey_row(2022, 125, "MOR", 1001, rank=10.0)])
        extractor = MasseyOrdinalExtractor(df)
        snap = extractor.safe_snapshot(2022, 1001, 133, "MOR")

        assert snap.status == OrdinalStatus.OK
        assert snap.staleness_days == 8

    def test_warning_status_for_moderately_stale_snapshot(self) -> None:
        """Staleness between warning and critical thresholds → WARNING."""
        # default warning=14, critical=28: day 112 → staleness = 21, 14<21≤28 → WARNING
        df = pd.DataFrame([_massey_row(2022, 112, "MOR", 1001, rank=10.0)])
        extractor = MasseyOrdinalExtractor(df)
        snap = extractor.safe_snapshot(2022, 1001, 133, "MOR")

        assert snap.status == OrdinalStatus.WARNING
        assert snap.staleness_days == 21

    def test_critical_status_for_very_stale_snapshot(self) -> None:
        """Staleness beyond critical threshold → CRITICAL."""
        # day 90 → staleness = 43 > 28 → CRITICAL
        df = pd.DataFrame([_massey_row(2022, 90, "MOR", 1001, rank=10.0)])
        extractor = MasseyOrdinalExtractor(df)
        snap = extractor.safe_snapshot(2022, 1001, 133, "MOR")

        assert snap.status == OrdinalStatus.CRITICAL
        assert snap.staleness_days == 43

    def test_custom_thresholds_applied(self) -> None:
        """Custom warning/critical day thresholds must be respected."""
        df = pd.DataFrame([_massey_row(2022, 120, "MOR", 1001, rank=10.0)])
        extractor = MasseyOrdinalExtractor(df, stale_warning_days=5, stale_critical_days=20)
        snap = extractor.safe_snapshot(2022, 1001, 133, "MOR")

        # staleness = 13; 5 < 13 ≤ 20 → WARNING
        assert snap.status == OrdinalStatus.WARNING

    def test_snapshot_attributes_are_consistent(self) -> None:
        """Snapshot attributes must be self-consistent (cutoff, day, staleness)."""
        df = pd.DataFrame([_massey_row(2022, 128, "SAG", 1001, rank=7.0)])
        extractor = MasseyOrdinalExtractor(df)
        snap = extractor.safe_snapshot(2022, 1001, 133, "SAG")

        assert snap.cutoff_day == 133
        assert snap.ranking_day == 128
        assert snap.staleness_days == 5
        assert snap.season == 2022
        assert snap.team_id == 1001
        assert snap.system_name == "SAG"


# ===========================================================================
# 8-10: FeatureAssembler — canonical ordering and symmetry
# ===========================================================================

def _build_assembler_with_store(
    context: FoldContext | None = None,
) -> tuple[FeatureAssembler, RollingFeatureStore, FoldContext]:
    """
    Build a FeatureAssembler backed by a materialized RollingFeatureStore.

    Game setup (season 2022, all DayNum < 133):
      1001 beats 1002 (day 50)
      1001 beats 1003 (day 60)
      1001 beats 1004 (day 70)
      1002 beats 1003 (day 55)
      1004 beats 1002 (day 65)

    Resulting win_pct:
      1001: 3/3 = 1.000
      1002: 1/3 = 0.333
      1003: 0/2 = 0.000
      1004: 1/2 = 0.500
    """
    ctx = context or _make_context()
    store = RollingFeatureStore(ctx)
    games = _games_df(
        _game(season=2022, day_num=50, w_id=1001, l_id=1002),
        _game(season=2022, day_num=60, w_id=1001, l_id=1003),
        _game(season=2022, day_num=70, w_id=1001, l_id=1004),
        _game(season=2022, day_num=55, w_id=1002, l_id=1003),
        _game(season=2022, day_num=65, w_id=1004, l_id=1002),
    )
    store.materialize(games)
    assembler = FeatureAssembler(ctx, store)
    return assembler, store, ctx


class TestFeatureAssemblerOrdering:
    """FeatureAssembler must enforce team1_id < team2_id."""

    def test_canonical_ordering_when_input_is_reversed(self) -> None:
        """assemble(1002, 1001) must produce team1=1001, team2=1002."""
        assembler, _, _ = _build_assembler_with_store()
        result = assembler.assemble(team_a_id=1002, team_b_id=1001, season=2022)

        assert result.team1_id == 1001
        assert result.team2_id == 1002

    def test_canonical_ordering_when_input_is_already_sorted(self) -> None:
        """assemble(1001, 1002) must also give team1=1001, team2=1002."""
        assembler, _, _ = _build_assembler_with_store()
        result = assembler.assemble(team_a_id=1001, team_b_id=1002, season=2022)

        assert result.team1_id == 1001
        assert result.team2_id == 1002

    def test_same_team_raises(self) -> None:
        """assemble(1001, 1001) must raise ValueError."""
        assembler, _, _ = _build_assembler_with_store()
        with pytest.raises(ValueError):
            assembler.assemble(1001, 1001, season=2022)

    def test_fold_id_is_stamped(self) -> None:
        """Assembled features must carry the fold_id from the active context."""
        ctx = _make_context(fold_id="provenance-check-fold")
        assembler, _, _ = _build_assembler_with_store(context=ctx)
        result = assembler.assemble(1001, 1002, season=2022)

        assert result.fold_id == "provenance-check-fold"


class TestFeatureAssemblerSymmetry:
    """Sum features invariant; diff features deterministic under canonical order."""

    def test_sum_features_invariant_under_team_swap(self) -> None:
        """
        sum_win_pct must be the same whether we call assemble(A, B) or
        assemble(B, A).
        """
        assembler, _, _ = _build_assembler_with_store()
        r_ab = assembler.assemble(1001, 1002, season=2022)
        r_ba = assembler.assemble(1002, 1001, season=2022)

        assert r_ab.features["sum_win_pct"] == pytest.approx(r_ba.features["sum_win_pct"])
        assert r_ab.features["sum_games_played"] == pytest.approx(
            r_ba.features["sum_games_played"]
        )

    def test_diff_features_same_under_team_swap(self) -> None:
        """
        diff_win_pct must be identical regardless of input order because
        canonical ordering pins team1=1001.
        """
        assembler, _, _ = _build_assembler_with_store()
        r_ab = assembler.assemble(1001, 1002, season=2022)
        r_ba = assembler.assemble(1002, 1001, season=2022)

        assert r_ab.features["diff_win_pct"] == pytest.approx(r_ba.features["diff_win_pct"])

    def test_diff_features_are_team1_minus_team2(self) -> None:
        """diff_win_pct == team1_win_pct - team2_win_pct."""
        assembler, store, _ = _build_assembler_with_store()
        result = assembler.assemble(1001, 1002, season=2022)

        t1_wp = result.features["team1_win_pct"]
        t2_wp = result.features["team2_win_pct"]
        expected_diff = t1_wp - t2_wp

        assert result.features["diff_win_pct"] == pytest.approx(expected_diff)

    def test_sum_features_equal_raw_sum(self) -> None:
        """sum_win_pct == team1_win_pct + team2_win_pct."""
        assembler, _, _ = _build_assembler_with_store()
        result = assembler.assemble(1001, 1002, season=2022)

        t1_wp = result.features["team1_win_pct"]
        t2_wp = result.features["team2_win_pct"]
        expected_sum = t1_wp + t2_wp

        assert result.features["sum_win_pct"] == pytest.approx(expected_sum)

    def test_all_sum_features_invariant_across_all_feature_names(self) -> None:
        """All sum_* features must be invariant under swap for this matchup."""
        assembler, store, _ = _build_assembler_with_store()
        r_ab = assembler.assemble(1001, 1002, season=2022)
        r_ba = assembler.assemble(1002, 1001, season=2022)

        for name in store.FEATURE_NAMES:
            v_ab = r_ab.features[f"sum_{name}"]
            v_ba = r_ba.features[f"sum_{name}"]
            if math.isnan(v_ab) and math.isnan(v_ba):
                continue
            assert v_ab == pytest.approx(v_ba, rel=1e-9), (
                f"sum_{name}: {v_ab} != {v_ba} under team-order swap"
            )

    def test_all_diff_features_consistent_under_swap(self) -> None:
        """All diff_* features must be equal under canonical-order pinning."""
        assembler, store, _ = _build_assembler_with_store()
        r_ab = assembler.assemble(1001, 1002, season=2022)
        r_ba = assembler.assemble(1002, 1001, season=2022)

        for name in store.FEATURE_NAMES:
            v_ab = r_ab.features[f"diff_{name}"]
            v_ba = r_ba.features[f"diff_{name}"]
            if math.isnan(v_ab) and math.isnan(v_ba):
                continue
            assert v_ab == pytest.approx(v_ba, rel=1e-9), (
                f"diff_{name}: {v_ab} != {v_ba} under team-order swap"
            )

    def test_seed_diff_is_canonical(self) -> None:
        """seed_diff must be seed1 - seed2, consistent under input swap."""
        assembler, _, _ = _build_assembler_with_store()
        # 1001 gets seed 3, 1002 gets seed 14
        r_ab = assembler.assemble(1001, 1002, season=2022, seed_a=3, seed_b=14)
        r_ba = assembler.assemble(1002, 1001, season=2022, seed_a=14, seed_b=3)

        assert r_ab.features["seed_diff"] == pytest.approx(-11.0)  # seed1(1001)=3 - seed2(1002)=14
        assert r_ba.features["seed_diff"] == pytest.approx(-11.0)


# ===========================================================================
# 11: No dataset contracts returned by Step 3 code
# ===========================================================================

class TestNoDatasetContractLeakage:
    """Step 3 must not produce or return Step 2 dataset contract objects."""

    _DATASET_TYPES = (TrainDataset, CalDataset, EvalDataset, EvalLabels)

    def test_materialize_returns_dataframe_not_dataset(self) -> None:
        ctx = _make_context()
        store = RollingFeatureStore(ctx)
        games = _games_df(_game(season=2022, day_num=50, w_id=1001, l_id=1002))
        result = store.materialize(games)

        assert isinstance(result, pd.DataFrame)
        assert not isinstance(result, self._DATASET_TYPES)

    def test_get_team_features_returns_dict_not_dataset(self) -> None:
        ctx = _make_context()
        store = RollingFeatureStore(ctx)
        games = _games_df(_game(season=2022, day_num=50, w_id=1001, l_id=1002))
        store.materialize(games)
        result = store.get_team_features(2022, 1001)

        assert isinstance(result, dict)
        assert not isinstance(result, self._DATASET_TYPES)

    def test_safe_snapshot_returns_ordinal_snapshot_not_dataset(self) -> None:
        df = pd.DataFrame([_massey_row(2022, 125, "MOR", 1001, rank=10.0)])
        extractor = MasseyOrdinalExtractor(df)
        result = extractor.safe_snapshot(2022, 1001, 133, "MOR")

        assert isinstance(result, OrdinalSnapshot)
        assert not isinstance(result, self._DATASET_TYPES)

    def test_assemble_returns_assembled_features_not_dataset(self) -> None:
        assembler, _, _ = _build_assembler_with_store()
        result = assembler.assemble(1001, 1002, season=2022)

        assert isinstance(result, AssembledFeatures)
        assert not isinstance(result, self._DATASET_TYPES)

    def test_assembled_features_has_no_y_attribute(self) -> None:
        """AssembledFeatures must not carry any label field."""
        assembler, _, _ = _build_assembler_with_store()
        result = assembler.assemble(1001, 1002, season=2022)

        assert not hasattr(result, "y")
        assert not hasattr(result, "labels")


# ===========================================================================
# Schema validation
# ===========================================================================

class TestSchemaValidation:
    """Missing required columns must raise ValueError immediately."""

    def test_rolling_store_raises_on_missing_columns(self) -> None:
        ctx = _make_context()
        store = RollingFeatureStore(ctx)
        incomplete = pd.DataFrame({"Season": [2022], "DayNum": [50]})

        with pytest.raises(ValueError, match="missing required columns"):
            store.materialize(incomplete)

    def test_massey_extractor_raises_on_missing_columns(self) -> None:
        incomplete = pd.DataFrame({"Season": [2022], "TeamID": [1001]})

        with pytest.raises(ValueError, match="missing required columns"):
            MasseyOrdinalExtractor(incomplete)

    def test_materialize_requires_daynm_for_cutoff(self) -> None:
        """filter_dataframe will raise if DayNum is absent."""
        ctx = _make_context()
        store = RollingFeatureStore(ctx)
        # Build a DataFrame with all required cols except DayNum
        row = _game(season=2022, day_num=50, w_id=1001, l_id=1002)
        df = pd.DataFrame([row]).drop(columns=["DayNum"])

        with pytest.raises(ValueError):
            store.materialize(df)
