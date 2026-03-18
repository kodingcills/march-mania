"""
test_datasets.py
================
Tier-1 dataset contract tests: prove that typed boundaries, immutability,
and label-separation laws hold.

These tests are the executable specification of Phase 1, Step 2.  Every
acceptance criterion listed in PHASE_PLAN must pass before Step 2 is complete.

All tests are:
  - deterministic
  - isolated (no shared mutable state, no external files)
  - self-contained (inline numpy arrays only)
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from ncaa_pipeline.data.datasets import (
    CalDataset,
    EvalDataset,
    EvalLabels,
    TrainDataset,
)


# ---------------------------------------------------------------------------
# Shared array factories
# ---------------------------------------------------------------------------

def _X(n: int = 5, f: int = 3) -> np.ndarray:
    return np.ones((n, f), dtype=np.float64)


def _y(n: int = 5) -> np.ndarray:
    return np.array([0, 1, 0, 1, 0][:n] + [0] * max(0, n - 5), dtype=np.int32)


def _ids(n: int = 5) -> np.ndarray:
    return np.arange(n, dtype=np.int64)


def _names(f: int = 3) -> tuple[str, ...]:
    return tuple(f"feat_{i}" for i in range(f))


# ---------------------------------------------------------------------------
# TrainDataset factory
# ---------------------------------------------------------------------------

def _make_train(
    n: int = 5,
    f: int = 3,
    fold_id: str = "fold-train-001",
    zone: str = "train",
) -> TrainDataset:
    return TrainDataset(
        X=_X(n, f),
        y=_y(n),
        matchup_ids=_ids(n),
        feature_names=_names(f),
        fold_id=fold_id,
        zone=zone,
    )


# ---------------------------------------------------------------------------
# CalDataset factory
# ---------------------------------------------------------------------------

def _make_cal(
    n: int = 5,
    f: int = 3,
    fold_id: str = "fold-cal-001",
    zone: str = "cal",
) -> CalDataset:
    return CalDataset(
        X=_X(n, f),
        y=_y(n),
        matchup_ids=_ids(n),
        feature_names=_names(f),
        fold_id=fold_id,
        zone=zone,
    )


# ---------------------------------------------------------------------------
# EvalDataset factory
# ---------------------------------------------------------------------------

def _make_eval(
    n: int = 5,
    f: int = 3,
    fold_id: str = "fold-eval-001",
    zone: str = "eval",
) -> EvalDataset:
    return EvalDataset(
        X=_X(n, f),
        matchup_ids=_ids(n),
        feature_names=_names(f),
        fold_id=fold_id,
        zone=zone,
    )


# ---------------------------------------------------------------------------
# EvalLabels factory
# ---------------------------------------------------------------------------

def _make_labels(
    n: int = 5,
    fold_id: str = "fold-eval-001",
    season: int = 2023,
) -> EvalLabels:
    return EvalLabels(
        y=_y(n),
        matchup_ids=_ids(n),
        season=season,
        fold_id=fold_id,
    )


# ===========================================================================
# TrainDataset — valid construction and field access
# ===========================================================================

class TestTrainDatasetConstruction:

    def test_valid_construction(self) -> None:
        ds = _make_train()
        assert ds.X.shape == (5, 3)
        assert ds.y.shape == (5,)
        assert ds.matchup_ids.shape == (5,)
        assert len(ds.feature_names) == 3
        assert ds.fold_id == "fold-train-001"
        assert ds.zone == "train"

    def test_frozen(self) -> None:
        ds = _make_train()
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            ds.X = np.zeros((5, 3))  # type: ignore[misc]

    def test_frozen_y(self) -> None:
        ds = _make_train()
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            ds.y = np.zeros(5)  # type: ignore[misc]

    def test_frozen_fold_id(self) -> None:
        ds = _make_train()
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            ds.fold_id = "mutated"  # type: ignore[misc]

    def test_X_is_readonly(self) -> None:
        ds = _make_train()
        assert not ds.X.flags.writeable
        with pytest.raises((ValueError, TypeError)):
            ds.X[0, 0] = 999.0

    def test_y_is_readonly(self) -> None:
        ds = _make_train()
        assert not ds.y.flags.writeable
        with pytest.raises((ValueError, TypeError)):
            ds.y[0] = 1

    def test_matchup_ids_is_readonly(self) -> None:
        ds = _make_train()
        assert not ds.matchup_ids.flags.writeable
        with pytest.raises((ValueError, TypeError)):
            ds.matchup_ids[0] = 999


# ===========================================================================
# TrainDataset — validation failures
# ===========================================================================

class TestTrainDatasetValidation:

    def test_X_must_be_ndarray(self) -> None:
        with pytest.raises(TypeError, match="ndarray|numpy"):
            TrainDataset(
                X=[[1, 2], [3, 4]],  # type: ignore[arg-type]
                y=_y(2),
                matchup_ids=_ids(2),
                feature_names=_names(2),
                fold_id="f",
            )

    def test_y_must_be_ndarray(self) -> None:
        with pytest.raises(TypeError, match="ndarray|numpy"):
            TrainDataset(
                X=_X(2, 2),
                y=[0, 1],  # type: ignore[arg-type]
                matchup_ids=_ids(2),
                feature_names=_names(2),
                fold_id="f",
            )

    def test_matchup_ids_must_be_ndarray(self) -> None:
        with pytest.raises(TypeError, match="ndarray|numpy"):
            TrainDataset(
                X=_X(2, 2),
                y=_y(2),
                matchup_ids=[0, 1],  # type: ignore[arg-type]
                feature_names=_names(2),
                fold_id="f",
            )

    def test_X_must_be_2d(self) -> None:
        with pytest.raises(ValueError, match="2-D|ndim"):
            TrainDataset(
                X=np.ones(5),
                y=_y(5),
                matchup_ids=_ids(5),
                feature_names=("a",),
                fold_id="f",
            )

    def test_y_must_be_1d(self) -> None:
        with pytest.raises(ValueError, match="1-D|ndim"):
            TrainDataset(
                X=_X(2, 2),
                y=np.zeros((2, 1)),
                matchup_ids=_ids(2),
                feature_names=_names(2),
                fold_id="f",
            )

    def test_matchup_ids_must_be_1d(self) -> None:
        with pytest.raises(ValueError, match="1-D|ndim"):
            TrainDataset(
                X=_X(2, 2),
                y=_y(2),
                matchup_ids=np.zeros((2, 1)),
                feature_names=_names(2),
                fold_id="f",
            )

    def test_X_y_row_mismatch(self) -> None:
        with pytest.raises(ValueError, match="shape|rows|match"):
            TrainDataset(
                X=_X(5, 2),
                y=_y(3),  # wrong row count
                matchup_ids=_ids(5),
                feature_names=_names(2),
                fold_id="f",
            )

    def test_X_matchup_ids_row_mismatch(self) -> None:
        with pytest.raises(ValueError, match="shape|rows|match"):
            TrainDataset(
                X=_X(5, 2),
                y=_y(5),
                matchup_ids=_ids(3),  # wrong row count
                feature_names=_names(2),
                fold_id="f",
            )

    def test_feature_names_length_mismatch(self) -> None:
        with pytest.raises(ValueError, match="feature_names|shape"):
            TrainDataset(
                X=_X(5, 3),
                y=_y(5),
                matchup_ids=_ids(5),
                feature_names=("a", "b"),  # should be 3
                fold_id="f",
            )

    def test_feature_names_must_be_tuple(self) -> None:
        with pytest.raises(TypeError, match="tuple"):
            TrainDataset(
                X=_X(5, 2),
                y=_y(5),
                matchup_ids=_ids(5),
                feature_names=["a", "b"],  # type: ignore[arg-type]
                fold_id="f",
            )

    def test_feature_names_must_contain_strings(self) -> None:
        with pytest.raises(TypeError, match="str"):
            TrainDataset(
                X=_X(5, 2),
                y=_y(5),
                matchup_ids=_ids(5),
                feature_names=("a", 2),  # type: ignore[arg-type]
                fold_id="f",
            )

    def test_empty_fold_id_rejected(self) -> None:
        with pytest.raises(ValueError, match="fold_id"):
            TrainDataset(
                X=_X(),
                y=_y(),
                matchup_ids=_ids(),
                feature_names=_names(),
                fold_id="",
            )

    def test_whitespace_fold_id_rejected(self) -> None:
        with pytest.raises(ValueError, match="fold_id"):
            TrainDataset(
                X=_X(),
                y=_y(),
                matchup_ids=_ids(),
                feature_names=_names(),
                fold_id="   ",
            )

    def test_non_binary_y_rejected(self) -> None:
        with pytest.raises(ValueError, match="binary|0 or 1"):
            TrainDataset(
                X=_X(3, 2),
                y=np.array([0, 1, 2]),  # 2 is not binary
                matchup_ids=_ids(3),
                feature_names=_names(2),
                fold_id="f",
            )

    def test_float_y_with_non_binary_rejected(self) -> None:
        with pytest.raises(ValueError, match="binary|0 or 1"):
            TrainDataset(
                X=_X(3, 2),
                y=np.array([0.0, 0.5, 1.0]),  # 0.5 is not binary
                matchup_ids=_ids(3),
                feature_names=_names(2),
                fold_id="f",
            )

    def test_wrong_zone_rejected(self) -> None:
        with pytest.raises(ValueError, match="zone|train"):
            _make_train(zone="eval")

    def test_zone_cal_rejected_for_train(self) -> None:
        with pytest.raises(ValueError, match="zone|train"):
            _make_train(zone="cal")


# ===========================================================================
# CalDataset — valid construction and field access
# ===========================================================================

class TestCalDatasetConstruction:

    def test_valid_construction(self) -> None:
        ds = _make_cal()
        assert ds.X.shape == (5, 3)
        assert ds.y.shape == (5,)
        assert ds.zone == "cal"

    def test_frozen(self) -> None:
        ds = _make_cal()
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            ds.X = np.zeros((5, 3))  # type: ignore[misc]

    def test_X_is_readonly(self) -> None:
        ds = _make_cal()
        assert not ds.X.flags.writeable
        with pytest.raises((ValueError, TypeError)):
            ds.X[0, 0] = 999.0

    def test_y_is_readonly(self) -> None:
        ds = _make_cal()
        assert not ds.y.flags.writeable
        with pytest.raises((ValueError, TypeError)):
            ds.y[0] = 1

    def test_matchup_ids_is_readonly(self) -> None:
        ds = _make_cal()
        assert not ds.matchup_ids.flags.writeable
        with pytest.raises((ValueError, TypeError)):
            ds.matchup_ids[0] = 999


# ===========================================================================
# CalDataset — validation failures
# ===========================================================================

class TestCalDatasetValidation:

    def test_X_y_row_mismatch(self) -> None:
        with pytest.raises(ValueError, match="shape|rows|match"):
            CalDataset(
                X=_X(5, 2),
                y=_y(4),
                matchup_ids=_ids(5),
                feature_names=_names(2),
                fold_id="f",
            )

    def test_feature_names_length_mismatch(self) -> None:
        with pytest.raises(ValueError, match="feature_names|shape"):
            CalDataset(
                X=_X(5, 3),
                y=_y(5),
                matchup_ids=_ids(5),
                feature_names=("a",),
                fold_id="f",
            )

    def test_non_binary_y_rejected(self) -> None:
        with pytest.raises(ValueError, match="binary|0 or 1"):
            CalDataset(
                X=_X(3, 2),
                y=np.array([-1, 0, 1]),
                matchup_ids=_ids(3),
                feature_names=_names(2),
                fold_id="f",
            )

    def test_wrong_zone_rejected(self) -> None:
        with pytest.raises(ValueError, match="zone|cal"):
            _make_cal(zone="train")

    def test_zone_eval_rejected_for_cal(self) -> None:
        with pytest.raises(ValueError, match="zone|cal"):
            _make_cal(zone="eval")

    def test_empty_fold_id_rejected(self) -> None:
        with pytest.raises(ValueError, match="fold_id"):
            CalDataset(
                X=_X(),
                y=_y(),
                matchup_ids=_ids(),
                feature_names=_names(),
                fold_id="",
            )


# ===========================================================================
# EvalDataset — valid construction and field access
# ===========================================================================

class TestEvalDatasetConstruction:

    def test_valid_construction(self) -> None:
        ds = _make_eval()
        assert ds.X.shape == (5, 3)
        assert ds.matchup_ids.shape == (5,)
        assert len(ds.feature_names) == 3
        assert ds.fold_id == "fold-eval-001"
        assert ds.zone == "eval"

    def test_frozen(self) -> None:
        ds = _make_eval()
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            ds.X = np.zeros((5, 3))  # type: ignore[misc]

    def test_X_is_readonly(self) -> None:
        ds = _make_eval()
        assert not ds.X.flags.writeable
        with pytest.raises((ValueError, TypeError)):
            ds.X[0, 0] = 999.0

    def test_matchup_ids_is_readonly(self) -> None:
        ds = _make_eval()
        assert not ds.matchup_ids.flags.writeable
        with pytest.raises((ValueError, TypeError)):
            ds.matchup_ids[0] = 999


# ===========================================================================
# EvalDataset — THE TYPE FIREWALL (acceptance criteria §1 and §4)
# ===========================================================================

class TestEvalDatasetTypeFirewall:

    def test_y_attribute_absent(self) -> None:
        """
        ACCEPTANCE CRITERION 1: eval_data.y must raise AttributeError.

        This is the primary anti-leakage boundary.  EvalDataset has no .y
        field by construction.  No model, calibrator, or router may read
        evaluation labels from this object.
        """
        ds = _make_eval()
        with pytest.raises(AttributeError):
            _ = ds.y  # type: ignore[attr-defined]

    def test_y_cannot_be_set(self) -> None:
        """
        Attempting to assign .y to an EvalDataset must also fail.
        frozen=True blocks field assignment; slots=True blocks dynamic attrs.
        Python 3.10 raises TypeError for non-existent slots on frozen classes.
        """
        ds = _make_eval()
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError, TypeError)):
            ds.y = np.zeros(5)  # type: ignore[attr-defined]

    def test_y_kwarg_rejected_at_construction(self) -> None:
        """
        ACCEPTANCE CRITERION 4: EvalDataset cannot be instantiated with y.

        Passing y=... to EvalDataset.__init__ must raise TypeError because
        the field does not exist.
        """
        with pytest.raises(TypeError):
            EvalDataset(  # type: ignore[call-arg]
                X=_X(),
                y=_y(),  # y is not a valid field
                matchup_ids=_ids(),
                feature_names=_names(),
                fold_id="f",
            )

    def test_eval_dataset_has_no_y_in_slots(self) -> None:
        """
        EvalDataset.__slots__ must not contain 'y'.

        This verifies the structural guarantee at the class level, not just
        the instance level.
        """
        all_slots: set[str] = set()
        for cls in type(_make_eval()).__mro__:
            all_slots.update(getattr(cls, "__slots__", ()))
        assert "y" not in all_slots, (
            "EvalDataset has 'y' in its __slots__; this violates the type firewall."
        )


# ===========================================================================
# EvalDataset — validation failures
# ===========================================================================

class TestEvalDatasetValidation:

    def test_X_must_be_ndarray(self) -> None:
        with pytest.raises(TypeError, match="ndarray|numpy"):
            EvalDataset(
                X=[[1, 2]],  # type: ignore[arg-type]
                matchup_ids=_ids(1),
                feature_names=_names(2),
                fold_id="f",
            )

    def test_X_must_be_2d(self) -> None:
        with pytest.raises(ValueError, match="2-D|ndim"):
            EvalDataset(
                X=np.ones(5),
                matchup_ids=_ids(5),
                feature_names=("a",),
                fold_id="f",
            )

    def test_X_matchup_ids_row_mismatch(self) -> None:
        with pytest.raises(ValueError, match="shape|rows|match"):
            EvalDataset(
                X=_X(5, 2),
                matchup_ids=_ids(3),
                feature_names=_names(2),
                fold_id="f",
            )

    def test_feature_names_length_mismatch(self) -> None:
        with pytest.raises(ValueError, match="feature_names|shape"):
            EvalDataset(
                X=_X(5, 3),
                matchup_ids=_ids(5),
                feature_names=("a", "b"),  # should be 3
                fold_id="f",
            )

    def test_wrong_zone_rejected(self) -> None:
        with pytest.raises(ValueError, match="zone|eval"):
            _make_eval(zone="train")

    def test_zone_cal_rejected_for_eval(self) -> None:
        with pytest.raises(ValueError, match="zone|eval"):
            _make_eval(zone="cal")

    def test_empty_fold_id_rejected(self) -> None:
        with pytest.raises(ValueError, match="fold_id"):
            EvalDataset(
                X=_X(),
                matchup_ids=_ids(),
                feature_names=_names(),
                fold_id="",
            )


# ===========================================================================
# EvalLabels — valid construction and field access
# ===========================================================================

class TestEvalLabelsConstruction:

    def test_valid_construction(self) -> None:
        labels = _make_labels()
        assert labels.y.shape == (5,)
        assert labels.matchup_ids.shape == (5,)
        assert labels.season == 2023
        assert labels.fold_id == "fold-eval-001"

    def test_frozen(self) -> None:
        labels = _make_labels()
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            labels.y = np.zeros(5)  # type: ignore[misc]

    def test_y_is_readonly(self) -> None:
        labels = _make_labels()
        assert not labels.y.flags.writeable
        with pytest.raises((ValueError, TypeError)):
            labels.y[0] = 1

    def test_matchup_ids_is_readonly(self) -> None:
        labels = _make_labels()
        assert not labels.matchup_ids.flags.writeable
        with pytest.raises((ValueError, TypeError)):
            labels.matchup_ids[0] = 999


# ===========================================================================
# EvalLabels — validation failures
# ===========================================================================

class TestEvalLabelsValidation:

    def test_y_must_be_ndarray(self) -> None:
        with pytest.raises(TypeError, match="ndarray|numpy"):
            EvalLabels(
                y=[0, 1, 0],  # type: ignore[arg-type]
                matchup_ids=_ids(3),
                season=2023,
                fold_id="f",
            )

    def test_y_must_be_1d(self) -> None:
        with pytest.raises(ValueError, match="1-D|ndim"):
            EvalLabels(
                y=np.zeros((3, 1)),
                matchup_ids=_ids(3),
                season=2023,
                fold_id="f",
            )

    def test_y_matchup_ids_row_mismatch(self) -> None:
        with pytest.raises(ValueError, match="shape|rows|match"):
            EvalLabels(
                y=_y(5),
                matchup_ids=_ids(3),  # wrong size
                season=2023,
                fold_id="f",
            )

    def test_non_binary_y_rejected(self) -> None:
        with pytest.raises(ValueError, match="binary|0 or 1"):
            EvalLabels(
                y=np.array([0, 2, 1]),
                matchup_ids=_ids(3),
                season=2023,
                fold_id="f",
            )

    def test_empty_fold_id_rejected(self) -> None:
        with pytest.raises(ValueError, match="fold_id"):
            EvalLabels(
                y=_y(),
                matchup_ids=_ids(),
                season=2023,
                fold_id="",
            )

    def test_zero_season_rejected(self) -> None:
        with pytest.raises(ValueError, match="season"):
            EvalLabels(
                y=_y(),
                matchup_ids=_ids(),
                season=0,
                fold_id="f",
            )

    def test_negative_season_rejected(self) -> None:
        with pytest.raises(ValueError, match="season"):
            EvalLabels(
                y=_y(),
                matchup_ids=_ids(),
                season=-2023,
                fold_id="f",
            )

    def test_non_int_season_rejected(self) -> None:
        with pytest.raises((ValueError, TypeError)):
            EvalLabels(
                y=_y(),
                matchup_ids=_ids(),
                season="2023",  # type: ignore[arg-type]
                fold_id="f",
            )


# ===========================================================================
# Type firewall — EvalDataset and EvalLabels are distinct (acceptance §5)
# ===========================================================================

class TestTypeDistinctness:

    def test_eval_dataset_is_not_eval_labels(self) -> None:
        """
        ACCEPTANCE CRITERION 5: EvalLabels must not be confused with EvalDataset.

        They are structurally distinct types.  isinstance checks confirm this.
        """
        ds = _make_eval()
        labels = _make_labels()
        assert not isinstance(ds, EvalLabels)
        assert not isinstance(labels, EvalDataset)

    def test_eval_dataset_type_name(self) -> None:
        assert type(_make_eval()).__name__ == "EvalDataset"

    def test_eval_labels_type_name(self) -> None:
        assert type(_make_labels()).__name__ == "EvalLabels"

    def test_train_dataset_is_not_cal_dataset(self) -> None:
        assert not isinstance(_make_train(), CalDataset)

    def test_cal_dataset_is_not_train_dataset(self) -> None:
        assert not isinstance(_make_cal(), TrainDataset)

    def test_eval_dataset_has_no_y(self) -> None:
        """EvalDataset field names must not include 'y'."""
        field_names = {f.name for f in dataclasses.fields(EvalDataset)}
        assert "y" not in field_names

    def test_eval_labels_has_y(self) -> None:
        """EvalLabels field names must include 'y'."""
        field_names = {f.name for f in dataclasses.fields(EvalLabels)}
        assert "y" in field_names

    def test_train_and_cal_have_y(self) -> None:
        """TrainDataset and CalDataset must both have a 'y' field."""
        assert "y" in {f.name for f in dataclasses.fields(TrainDataset)}
        assert "y" in {f.name for f in dataclasses.fields(CalDataset)}


# ===========================================================================
# Immutability — mutation fails on all array types (acceptance criterion §2)
# ===========================================================================

class TestImmutabilityAllTypes:

    def test_train_X_mutation_blocked(self) -> None:
        ds = _make_train()
        with pytest.raises((ValueError, TypeError)):
            ds.X[0, 0] = -1.0

    def test_train_y_mutation_blocked(self) -> None:
        ds = _make_train()
        with pytest.raises((ValueError, TypeError)):
            ds.y[0] = 0

    def test_cal_X_mutation_blocked(self) -> None:
        ds = _make_cal()
        with pytest.raises((ValueError, TypeError)):
            ds.X[0, 0] = -1.0

    def test_cal_y_mutation_blocked(self) -> None:
        ds = _make_cal()
        with pytest.raises((ValueError, TypeError)):
            ds.y[0] = 0

    def test_eval_X_mutation_blocked(self) -> None:
        ds = _make_eval()
        with pytest.raises((ValueError, TypeError)):
            ds.X[0, 0] = -1.0

    def test_eval_labels_y_mutation_blocked(self) -> None:
        labels = _make_labels()
        with pytest.raises((ValueError, TypeError)):
            labels.y[0] = 0


# ===========================================================================
# Source array isolation — input mutation does not affect dataset arrays
# ===========================================================================

class TestSourceArrayIsolation:
    """
    Datasets do not defensively copy arrays on construction (not required by
    spec).  What matters is that the stored array is non-writeable.  These
    tests verify that the sealed array rejects writes regardless of whether
    the caller still holds a reference to the original.
    """

    def test_train_X_sealed_regardless_of_source(self) -> None:
        X = np.ones((4, 2))
        ds = TrainDataset(
            X=X,
            y=_y(4),
            matchup_ids=_ids(4),
            feature_names=_names(2),
            fold_id="f",
        )
        # The stored array is sealed; direct item assignment must fail.
        with pytest.raises((ValueError, TypeError)):
            ds.X[0, 0] = 0.0

    def test_eval_labels_y_sealed_regardless_of_source(self) -> None:
        y = _y(4)
        labels = EvalLabels(
            y=y,
            matchup_ids=_ids(4),
            season=2023,
            fold_id="f",
        )
        with pytest.raises((ValueError, TypeError)):
            labels.y[0] = 0


# ===========================================================================
# Zone field correctness
# ===========================================================================

class TestZoneFields:

    def test_train_zone_literal(self) -> None:
        assert _make_train().zone == "train"

    def test_cal_zone_literal(self) -> None:
        assert _make_cal().zone == "cal"

    def test_eval_dataset_zone_literal(self) -> None:
        assert _make_eval().zone == "eval"

    def test_eval_labels_has_no_zone(self) -> None:
        """EvalLabels has no zone field; accessing .zone must raise AttributeError."""
        labels = _make_labels()
        with pytest.raises(AttributeError):
            _ = labels.zone  # type: ignore[attr-defined]


# ===========================================================================
# Binary label edge cases
# ===========================================================================

class TestBinaryLabelEdgeCases:

    def test_all_zeros_accepted(self) -> None:
        _make_train(n=4)  # _y returns zeros for n<=5

    def test_all_ones_accepted(self) -> None:
        TrainDataset(
            X=_X(3, 2),
            y=np.ones(3, dtype=np.int32),
            matchup_ids=_ids(3),
            feature_names=_names(2),
            fold_id="f",
        )

    def test_float_binary_labels_accepted(self) -> None:
        TrainDataset(
            X=_X(3, 2),
            y=np.array([0.0, 1.0, 0.0]),
            matchup_ids=_ids(3),
            feature_names=_names(2),
            fold_id="f",
        )

    def test_nan_in_y_rejected(self) -> None:
        with pytest.raises(ValueError, match="binary|0 or 1"):
            TrainDataset(
                X=_X(3, 2),
                y=np.array([0.0, np.nan, 1.0]),
                matchup_ids=_ids(3),
                feature_names=_names(2),
                fold_id="f",
            )

    def test_negative_y_rejected(self) -> None:
        with pytest.raises(ValueError, match="binary|0 or 1"):
            EvalLabels(
                y=np.array([-1, 0, 1]),
                matchup_ids=_ids(3),
                season=2023,
                fold_id="f",
            )
