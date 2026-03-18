"""
test_tier1_foundation.py
========================
Tier-1 foundation tests: prove that immutability, cutoff, and leakage laws hold.

These tests are the executable specification of the pipeline's first-line
guarantees.  They must pass unconditionally before any later stage may build
on top of this module.

All tests are:
  - deterministic
  - isolated (no shared mutable state, no external files)
  - self-contained (inline DataFrames only)
"""

from __future__ import annotations

import dataclasses

import pandas as pd
import pytest

from ncaa_pipeline.context.fold_context import FoldContext
from ncaa_pipeline.policies.cutoff_policy import CutoffViolationError, Day133CutoffPolicy
from ncaa_pipeline.policies.leakage_guard import (
    FutureDataViolationError,
    LabelLeakageError,
    LeakageGuard,
    ProvenanceError,
)


# ---------------------------------------------------------------------------
# Helpers / shared fixtures
# ---------------------------------------------------------------------------

def _make_valid_context(
    fold_id: str = "test-fold-001",
    train_seasons: tuple[int, ...] = (2018, 2019, 2021),
    cal_season: int = 2022,
    eval_season: int = 2023,
) -> FoldContext:
    """Return a minimal valid FoldContext for use in tests."""
    return FoldContext(
        fold_id=fold_id,
        train_seasons=train_seasons,
        cal_season=cal_season,
        eval_season=eval_season,
    )


# ===========================================================================
# FoldContext — immutability
# ===========================================================================

class TestFoldContextFrozen:
    def test_fold_context_frozen(self) -> None:
        """FoldContext must be immutable; any attribute assignment must fail."""
        ctx = _make_valid_context()
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            ctx.fold_id = "mutated"  # type: ignore[misc]

    def test_fold_context_frozen_train_seasons(self) -> None:
        """Attempting to replace train_seasons must also fail."""
        ctx = _make_valid_context()
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            ctx.train_seasons = (2015, 2016)  # type: ignore[misc]

    def test_fold_context_frozen_cal_season(self) -> None:
        """Attempting to replace cal_season must also fail."""
        ctx = _make_valid_context()
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            ctx.cal_season = 2025  # type: ignore[misc]


# ===========================================================================
# FoldContext — construction validation
# ===========================================================================

class TestFoldContextValidation:

    def test_fold_context_rejects_empty_train_seasons(self) -> None:
        """Empty train_seasons must be rejected at construction time."""
        with pytest.raises(ValueError, match="non-empty"):
            FoldContext(
                fold_id="f",
                train_seasons=(),
                cal_season=2022,
                eval_season=2023,
            )

    def test_fold_context_rejects_duplicate_train_seasons(self) -> None:
        """Duplicate entries in train_seasons must be rejected."""
        with pytest.raises(ValueError, match="[Uu]nique|duplicate"):
            FoldContext(
                fold_id="f",
                train_seasons=(2018, 2018, 2019),
                cal_season=2022,
                eval_season=2023,
            )

    def test_fold_context_rejects_overlapping_seasons_cal_in_train(self) -> None:
        """cal_season that also appears in train_seasons must be rejected."""
        with pytest.raises(ValueError):
            FoldContext(
                fold_id="f",
                train_seasons=(2018, 2019, 2022),
                cal_season=2022,
                eval_season=2023,
            )

    def test_fold_context_rejects_overlapping_seasons_eval_in_train(self) -> None:
        """eval_season that also appears in train_seasons must be rejected."""
        with pytest.raises(ValueError):
            FoldContext(
                fold_id="f",
                train_seasons=(2018, 2019, 2023),
                cal_season=2022,
                eval_season=2023,
            )

    def test_fold_context_rejects_overlapping_seasons_cal_equals_eval(self) -> None:
        """cal_season == eval_season must be rejected."""
        with pytest.raises(ValueError):
            FoldContext(
                fold_id="f",
                train_seasons=(2018, 2019),
                cal_season=2022,
                eval_season=2022,
            )

    def test_fold_context_rejects_cal_not_after_max_train(self) -> None:
        """cal_season <= max(train_seasons) must be rejected."""
        with pytest.raises(ValueError, match="[Ss]trictly greater|cal_season"):
            FoldContext(
                fold_id="f",
                train_seasons=(2018, 2019, 2021),
                cal_season=2019,   # not after max(train)=2021
                eval_season=2023,
            )

    def test_fold_context_rejects_eval_not_after_cal(self) -> None:
        """eval_season <= cal_season must be rejected."""
        with pytest.raises(ValueError, match="[Ss]trictly greater|eval_season"):
            FoldContext(
                fold_id="f",
                train_seasons=(2018, 2019),
                cal_season=2022,
                eval_season=2021,  # before cal
            )

    def test_fold_context_rejects_empty_fold_id(self) -> None:
        """Empty or whitespace-only fold_id must be rejected."""
        for bad_id in ("", "   ", "\t"):
            with pytest.raises(ValueError, match="fold_id"):
                FoldContext(
                    fold_id=bad_id,
                    train_seasons=(2018, 2019),
                    cal_season=2021,
                    eval_season=2022,
                )

    def test_fold_context_valid_construction(self) -> None:
        """A fully valid FoldContext must be constructable without error."""
        ctx = _make_valid_context()
        assert ctx.fold_id == "test-fold-001"
        assert ctx.cal_season == 2022
        assert ctx.eval_season == 2023
        assert ctx.day_cutoff == 133


# ===========================================================================
# FoldContext — 2020 anomaly rejection
# ===========================================================================

class TestFoldContext2020Anomaly:

    def test_2020_anomaly_rejection_cal(self) -> None:
        """cal_season == 2020 must be rejected (no tournament data)."""
        with pytest.raises(ValueError, match="[Aa]nomaly|2020"):
            FoldContext(
                fold_id="f",
                train_seasons=(2017, 2018, 2019),
                cal_season=2020,
                eval_season=2021,
            )

    def test_2020_anomaly_rejection_eval(self) -> None:
        """eval_season == 2020 must be rejected (no tournament data)."""
        with pytest.raises(ValueError, match="[Aa]nomaly|2020"):
            FoldContext(
                fold_id="f",
                train_seasons=(2017, 2018),
                cal_season=2019,
                eval_season=2020,
            )


# ===========================================================================
# FoldContext — day_cutoff immutability
# ===========================================================================

class TestFoldContextDayCutoff:

    def test_day_cutoff_cannot_be_relaxed_134(self) -> None:
        """day_cutoff=134 must be rejected."""
        with pytest.raises(ValueError, match="133|cutoff"):
            FoldContext(
                fold_id="f",
                train_seasons=(2018, 2019),
                cal_season=2021,
                eval_season=2022,
                day_cutoff=134,
            )

    def test_day_cutoff_cannot_be_relaxed_200(self) -> None:
        """day_cutoff=200 must also be rejected."""
        with pytest.raises(ValueError, match="133|cutoff"):
            FoldContext(
                fold_id="f",
                train_seasons=(2018, 2019),
                cal_season=2021,
                eval_season=2022,
                day_cutoff=200,
            )

    def test_day_cutoff_cannot_be_tightened_132(self) -> None:
        """day_cutoff=132 (stricter but wrong) must also be rejected."""
        with pytest.raises(ValueError, match="133|cutoff"):
            FoldContext(
                fold_id="f",
                train_seasons=(2018, 2019),
                cal_season=2021,
                eval_season=2022,
                day_cutoff=132,
            )

    def test_day_cutoff_accepts_133(self) -> None:
        """day_cutoff=133 (the only legal value) must be accepted."""
        ctx = FoldContext(
            fold_id="f",
            train_seasons=(2018, 2019),
            cal_season=2021,
            eval_season=2022,
            day_cutoff=133,
        )
        assert ctx.day_cutoff == 133


# ===========================================================================
# FoldContext — accessor methods
# ===========================================================================

class TestFoldContextAccessors:

    def test_max_train_season(self) -> None:
        ctx = _make_valid_context(train_seasons=(2015, 2018, 2017))
        assert ctx.max_train_season() == 2018

    def test_all_seasons_sorted(self) -> None:
        ctx = _make_valid_context(
            train_seasons=(2019, 2017, 2018),
            cal_season=2021,
            eval_season=2023,
        )
        assert ctx.all_seasons() == (2017, 2018, 2019, 2021, 2023)

    def test_describe_returns_string(self) -> None:
        ctx = _make_valid_context()
        desc = ctx.describe()
        assert isinstance(desc, str)
        assert "test-fold-001" in desc
        assert "2022" in desc
        assert "2023" in desc


# ===========================================================================
# Day133CutoffPolicy — filter_dataframe
# ===========================================================================

class TestCutoffPolicyFilterDataframe:

    def _policy(self) -> Day133CutoffPolicy:
        return Day133CutoffPolicy()

    def test_cutoff_filters_day134(self) -> None:
        """DayNum 134 must be removed; DayNum 132 must remain."""
        ctx = _make_valid_context()
        policy = self._policy()
        df = pd.DataFrame({
            "Season": [2022, 2022, 2022],
            "DayNum": [132, 133, 134],
            "Value":  [10,  20,   30],
        })
        result = policy.filter_dataframe(df, ctx)
        assert list(result["DayNum"]) == [132], (
            f"Expected only DayNum=132 to survive; got {list(result['DayNum'])}"
        )

    def test_cutoff_removes_exactly_133(self) -> None:
        """DayNum == 133 is forbidden (boundary is exclusive at 133)."""
        ctx = _make_valid_context()
        policy = self._policy()
        df = pd.DataFrame({"DayNum": [133]})
        result = policy.filter_dataframe(df, ctx)
        assert len(result) == 0

    def test_cutoff_allows_132(self) -> None:
        """DayNum == 132 must survive."""
        ctx = _make_valid_context()
        policy = self._policy()
        df = pd.DataFrame({"DayNum": [132]})
        result = policy.filter_dataframe(df, ctx)
        assert len(result) == 1

    def test_filter_does_not_mutate_input(self) -> None:
        """Input DataFrame must not be modified after filtering."""
        ctx = _make_valid_context()
        policy = self._policy()
        df = pd.DataFrame({"DayNum": [100, 133, 150]})
        original_len = len(df)
        original_values = list(df["DayNum"])
        policy.filter_dataframe(df, ctx)
        assert len(df) == original_len
        assert list(df["DayNum"]) == original_values

    def test_filter_preserves_original_index(self) -> None:
        """Surviving rows must retain their original integer index values."""
        ctx = _make_valid_context()
        policy = self._policy()
        df = pd.DataFrame({"DayNum": [100, 133, 50]})
        result = policy.filter_dataframe(df, ctx)
        # row 0 (DayNum=100) and row 2 (DayNum=50) survive; row 1 is dropped
        assert list(result.index) == [0, 2]

    def test_filter_preserves_row_order(self) -> None:
        """Row order among surviving rows must match input order."""
        ctx = _make_valid_context()
        policy = self._policy()
        df = pd.DataFrame({"DayNum": [10, 50, 130, 133, 1]})
        result = policy.filter_dataframe(df, ctx)
        assert list(result["DayNum"]) == [10, 50, 130, 1]

    def test_filter_raises_on_missing_daynum_column(self) -> None:
        """Missing DayNum column must raise ValueError with clear message."""
        ctx = _make_valid_context()
        policy = self._policy()
        df = pd.DataFrame({"Season": [2022], "Score": [70]})
        with pytest.raises(ValueError, match="DayNum"):
            policy.filter_dataframe(df, ctx)


# ===========================================================================
# Day133CutoffPolicy — assert_permitted
# ===========================================================================

class TestCutoffPolicyAssertPermitted:

    def _policy(self) -> Day133CutoffPolicy:
        return Day133CutoffPolicy()

    def test_cutoff_assert_permitted_raises_on_133(self) -> None:
        """assert_permitted(day_num=133) must raise CutoffViolationError."""
        ctx = _make_valid_context()
        policy = self._policy()
        with pytest.raises(CutoffViolationError):
            policy.assert_permitted(season=2022, day_num=133, context=ctx)

    def test_cutoff_assert_permitted_raises_on_134(self) -> None:
        """assert_permitted(day_num=134) must raise CutoffViolationError."""
        ctx = _make_valid_context()
        policy = self._policy()
        with pytest.raises(CutoffViolationError):
            policy.assert_permitted(season=2022, day_num=134, context=ctx)

    def test_cutoff_assert_permitted_raises_error_contains_metadata(self) -> None:
        """CutoffViolationError message must include season, day_num, fold_id, cutoff."""
        ctx = _make_valid_context(fold_id="fold-xyz")
        policy = self._policy()
        with pytest.raises(CutoffViolationError) as exc_info:
            policy.assert_permitted(season=2022, day_num=155, context=ctx)
        err = str(exc_info.value)
        assert "fold-xyz" in err
        assert "155" in err
        assert "133" in err

    def test_cutoff_assert_permitted_passes_on_132(self) -> None:
        """assert_permitted(day_num=132) must return None (no exception)."""
        ctx = _make_valid_context()
        policy = self._policy()
        result = policy.assert_permitted(season=2022, day_num=132, context=ctx)
        assert result is None

    def test_cutoff_assert_permitted_passes_on_0(self) -> None:
        """assert_permitted(day_num=0) must also pass."""
        ctx = _make_valid_context()
        policy = self._policy()
        policy.assert_permitted(season=2022, day_num=0, context=ctx)


# ===========================================================================
# LeakageGuard — assert_no_future_games
# ===========================================================================

class TestLeakageGuardFutureGames:

    def _guard(self) -> LeakageGuard:
        return LeakageGuard()

    def test_leakage_guard_detects_future_season(self) -> None:
        """Row with Season > max_season must raise FutureDataViolationError."""
        guard = self._guard()
        df = pd.DataFrame({
            "Season": [2021, 2023],   # 2023 > max_season=2022
            "DayNum": [100, 50],
        })
        with pytest.raises(FutureDataViolationError):
            guard.assert_no_future_games(df, max_season=2022, max_daynum=132)

    def test_leakage_guard_detects_future_day_in_same_season(self) -> None:
        """Row with Season==max_season but DayNum>max_daynum must raise."""
        guard = self._guard()
        df = pd.DataFrame({
            "Season": [2022, 2022],
            "DayNum": [100, 200],   # 200 > max_daynum=132
        })
        with pytest.raises(FutureDataViolationError):
            guard.assert_no_future_games(df, max_season=2022, max_daynum=132)

    def test_leakage_guard_allows_past_seasons_large_daynum(self) -> None:
        """Earlier seasons are always OK regardless of DayNum."""
        guard = self._guard()
        df = pd.DataFrame({
            "Season": [2019, 2020, 2021],
            "DayNum": [999, 999, 999],  # absurdly large, but seasons < max
        })
        # must not raise
        guard.assert_no_future_games(df, max_season=2022, max_daynum=132)

    def test_leakage_guard_allows_max_season_at_max_daynum(self) -> None:
        """Row exactly at (max_season, max_daynum) is within the frontier."""
        guard = self._guard()
        df = pd.DataFrame({
            "Season": [2022],
            "DayNum": [132],
        })
        guard.assert_no_future_games(df, max_season=2022, max_daynum=132)

    def test_leakage_guard_raises_on_missing_season_column(self) -> None:
        """Missing Season column must raise ValueError."""
        guard = self._guard()
        df = pd.DataFrame({"DayNum": [100]})
        with pytest.raises(ValueError, match="Season"):
            guard.assert_no_future_games(df, max_season=2022, max_daynum=132)

    def test_leakage_guard_raises_on_missing_daynum_column(self) -> None:
        """Missing DayNum column must raise ValueError."""
        guard = self._guard()
        df = pd.DataFrame({"Season": [2022]})
        with pytest.raises(ValueError, match="DayNum"):
            guard.assert_no_future_games(df, max_season=2022, max_daynum=132)

    def test_leakage_guard_error_contains_violating_row_count(self) -> None:
        """FutureDataViolationError message must mention the number of offenders."""
        guard = self._guard()
        df = pd.DataFrame({
            "Season": [2023, 2024],
            "DayNum": [10,   10],
        })
        with pytest.raises(FutureDataViolationError) as exc_info:
            guard.assert_no_future_games(df, max_season=2022, max_daynum=132)
        assert exc_info.value.max_season == 2022
        assert len(exc_info.value.violating_rows) == 2


# ===========================================================================
# LeakageGuard — assert_no_eval_labels_in_memory
# ===========================================================================

class TestLeakageGuardEvalLabels:

    def _guard(self) -> LeakageGuard:
        return LeakageGuard()

    def test_leakage_guard_detects_eval_labels_by_type_name(self) -> None:
        """Object whose class name is 'EvalLabels' must be detected."""

        class EvalLabels:  # name matches the registry
            pass

        state = {"features": {"data": EvalLabels()}}
        guard = self._guard()
        with pytest.raises(LabelLeakageError, match="EvalLabels"):
            guard.assert_no_eval_labels_in_memory(state)

    def test_leakage_guard_detects_eval_labels_nested_in_list(self) -> None:
        """EvalLabels inside a list must also be detected."""

        class EvalLabels:
            pass

        state = {"cache": [1, 2, EvalLabels()]}
        guard = self._guard()
        with pytest.raises(LabelLeakageError):
            guard.assert_no_eval_labels_in_memory(state)

    def test_leakage_guard_detects_eval_labels_by_key_name(self) -> None:
        """Dict key 'eval_labels' must trigger LabelLeakageError."""
        state = {"pipeline_output": {"eval_labels": [1, 0, 1]}}
        guard = self._guard()
        with pytest.raises(LabelLeakageError, match="eval_labels"):
            guard.assert_no_eval_labels_in_memory(state)

    def test_leakage_guard_detects_tourney_results_key(self) -> None:
        """Dict key 'tourney_results' must trigger LabelLeakageError."""
        state = {"tourney_results": {"W": [1, 0]}}
        guard = self._guard()
        with pytest.raises(LabelLeakageError):
            guard.assert_no_eval_labels_in_memory(state)

    def test_leakage_guard_detects_tournament_results_key(self) -> None:
        """Dict key 'tournament_results' must trigger LabelLeakageError."""
        state = {"tournament_results": {}}
        guard = self._guard()
        with pytest.raises(LabelLeakageError):
            guard.assert_no_eval_labels_in_memory(state)

    def test_leakage_guard_detects_actual_results_key(self) -> None:
        """Dict key 'actual_results' must trigger LabelLeakageError."""
        state = {"actual_results": [1, 0, 1]}
        guard = self._guard()
        with pytest.raises(LabelLeakageError):
            guard.assert_no_eval_labels_in_memory(state)

    def test_leakage_guard_detects_zone_c_labels_key(self) -> None:
        """Dict key 'zone_c_labels' must trigger LabelLeakageError."""
        state = {"zone_c_labels": pd.Series([1, 0])}
        guard = self._guard()
        with pytest.raises(LabelLeakageError):
            guard.assert_no_eval_labels_in_memory(state)

    def test_leakage_guard_allows_clean_state(self) -> None:
        """A state dict with no suspicious content must pass without error."""
        state = {
            "features": {"home_seed": [1, 4], "away_seed": [2, 3]},
            "model_config": {"lr": 0.01},
            "metadata": {"version": "1.0"},
        }
        guard = self._guard()
        guard.assert_no_eval_labels_in_memory(state)  # must not raise

    def test_leakage_guard_detects_path_in_error(self) -> None:
        """LabelLeakageError must include the dot-path where leakage was found."""
        state = {"outer": {"inner": {"eval_labels": []}}}
        guard = self._guard()
        with pytest.raises(LabelLeakageError) as exc_info:
            guard.assert_no_eval_labels_in_memory(state)
        assert "eval_labels" in exc_info.value.detection_path


# ===========================================================================
# LeakageGuard — assert_artifact_provenance
# ===========================================================================

class TestLeakageGuardProvenance:

    def _guard(self) -> LeakageGuard:
        return LeakageGuard()

    def test_provenance_guard_detects_missing_fold_id(self) -> None:
        """artifact_metadata without 'fold_id' must raise ProvenanceError."""
        ctx = _make_valid_context(fold_id="fold-A")
        guard = self._guard()
        with pytest.raises(ProvenanceError):
            guard.assert_artifact_provenance(
                artifact_metadata={"created_at": "2026-01-01"},
                context=ctx,
            )

    def test_provenance_guard_detects_mismatched_fold_id(self) -> None:
        """Mismatched fold_id must raise ProvenanceError."""
        ctx = _make_valid_context(fold_id="fold-A")
        guard = self._guard()
        with pytest.raises(ProvenanceError) as exc_info:
            guard.assert_artifact_provenance(
                artifact_metadata={"fold_id": "fold-B"},
                context=ctx,
            )
        err = exc_info.value
        assert err.expected_fold_id == "fold-A"
        assert err.actual_fold_id == "fold-B"

    def test_provenance_guard_passes_matching_fold_id(self) -> None:
        """Matching fold_id must pass without error."""
        ctx = _make_valid_context(fold_id="fold-A")
        guard = self._guard()
        guard.assert_artifact_provenance(
            artifact_metadata={"fold_id": "fold-A", "created_at": "2026-01-01"},
            context=ctx,
        )  # must not raise

    def test_provenance_error_contains_both_fold_ids(self) -> None:
        """ProvenanceError message must include both expected and actual fold_id."""
        ctx = _make_valid_context(fold_id="expected-fold")
        guard = self._guard()
        with pytest.raises(ProvenanceError) as exc_info:
            guard.assert_artifact_provenance(
                artifact_metadata={"fold_id": "wrong-fold"},
                context=ctx,
            )
        msg = str(exc_info.value)
        assert "expected-fold" in msg
        assert "wrong-fold" in msg
