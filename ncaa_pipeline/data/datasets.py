"""
datasets.py
===========
Typed, immutable data contracts for the three pipeline zones.

This module implements the primary type-level anti-leakage boundary for the
NCAA March Mania 2026 pipeline.  Four distinct types cover the three temporal
zones and the evaluation label lifecycle:

    TrainDataset   — Zone A inputs + labels (fit only)
    CalDataset     — Zone B inputs + labels (calibration fit + router fit)
    EvalDataset    — Zone C inputs ONLY (no labels, ever)
    EvalLabels     — Zone C labels ONLY (loaded at Stage 13 only)

The structural rule that matters most:

    EvalDataset has no .y attribute.
    EvalLabels is a physically separate type.

This is not a documentation preference.  Accessing .y on an EvalDataset raises
AttributeError immediately.  Passing EvalLabels through a stage boundary
triggers LeakageGuard.assert_no_eval_labels_in_memory().

Design constraints (PHASE_PLAN Step 2, MASTER_ARCHITECTURE §5):
  - frozen dataclasses with slots=True (consistent with FoldContext)
  - all NumPy arrays set to writeable=False in __post_init__
  - shape and type validation is strict; failures raise immediately
  - no inheritance hierarchy
  - no optional label fields
  - no generic BaseDataset abstraction
  - zone string field is validated against the expected literal per type
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# Module-private validation helpers
# ---------------------------------------------------------------------------

def _require_ndarray(value: object, name: str) -> None:
    """Raise TypeError if *value* is not a numpy.ndarray."""
    if not isinstance(value, np.ndarray):
        raise TypeError(
            f"{name} must be a numpy.ndarray; "
            f"got {type(value).__name__!r}."
        )


def _require_2d(arr: np.ndarray, name: str) -> None:
    """Raise ValueError if *arr* is not 2-dimensional."""
    if arr.ndim != 2:
        raise ValueError(
            f"{name} must be a 2-D array with shape (n_samples, n_features); "
            f"got ndim={arr.ndim}, shape={arr.shape}."
        )


def _require_1d(arr: np.ndarray, name: str) -> None:
    """Raise ValueError if *arr* is not 1-dimensional."""
    if arr.ndim != 1:
        raise ValueError(
            f"{name} must be a 1-D array; "
            f"got ndim={arr.ndim}, shape={arr.shape}."
        )


def _require_row_match(n_ref: int, ref_name: str, arr: np.ndarray, arr_name: str) -> None:
    """Raise ValueError if arr.shape[0] does not equal n_ref."""
    if arr.shape[0] != n_ref:
        raise ValueError(
            f"{arr_name}.shape[0] = {arr.shape[0]} does not match "
            f"{ref_name}.shape[0] = {n_ref}; "
            f"all arrays must share the same number of rows."
        )


def _require_feature_names(feature_names: object, n_features: int) -> None:
    """
    Raise TypeError if feature_names is not a tuple of str, or ValueError if
    its length does not match n_features.
    """
    if not isinstance(feature_names, tuple):
        raise TypeError(
            f"feature_names must be a tuple[str, ...]; "
            f"got {type(feature_names).__name__!r}."
        )
    for i, item in enumerate(feature_names):
        if not isinstance(item, str):
            raise TypeError(
                f"feature_names[{i}] must be a str; "
                f"got {type(item).__name__!r}."
            )
    if len(feature_names) != n_features:
        raise ValueError(
            f"len(feature_names) = {len(feature_names)} does not match "
            f"X.shape[1] = {n_features}; one name per feature column is required."
        )


def _require_fold_id(fold_id: object) -> None:
    """Raise ValueError if fold_id is not a non-empty, non-whitespace string."""
    if not isinstance(fold_id, str) or not fold_id.strip():
        raise ValueError(
            f"fold_id must be a non-empty string; got {fold_id!r}."
        )


def _require_binary_labels(y: np.ndarray, name: str) -> None:
    """
    Raise ValueError if *y* contains any value outside {0, 1}.

    Handles integer, float, and boolean dtypes.  NaN values are caught because
    NaN == 0 and NaN == 1 both evaluate to False in numpy.
    """
    if not np.all((y == 0) | (y == 1)):
        bad_mask = ~((y == 0) | (y == 1))
        bad_sample = np.unique(y[bad_mask])[:5].tolist()
        raise ValueError(
            f"{name} must contain only binary values (0 or 1); "
            f"found non-binary values (up to 5 unique): {bad_sample}."
        )


def _require_zone(zone: str, expected: str, class_name: str) -> None:
    """Raise ValueError if zone does not equal the expected literal."""
    if zone != expected:
        raise ValueError(
            f"{class_name}.zone must be {expected!r}; got {zone!r}."
        )


def _seal(arr: np.ndarray) -> None:
    """Set arr.flags.writeable = False in place."""
    arr.flags.writeable = False


# ---------------------------------------------------------------------------
# TrainDataset — Zone A
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class TrainDataset:
    """
    Immutable container for Zone A (training) inputs and labels.

    Read by base model .fit() methods only.  Must not be passed to
    calibrators, the router, or MetricEngine.

    Parameters
    ----------
    X :
        Feature matrix, shape (n_samples, n_features).  Set to
        writeable=False after construction.
    y :
        Binary outcome labels, shape (n_samples,), values in {0, 1}.
        Set to writeable=False after construction.
    matchup_ids :
        Row-level provenance identifiers, shape (n_samples,).
        Set to writeable=False after construction.
    feature_names :
        Column names for X; len must equal X.shape[1].
    fold_id :
        Non-empty fold identifier; must match the active FoldContext.fold_id.
    zone :
        Must be exactly "train".  Validated at construction time.
    """

    X: np.ndarray
    y: np.ndarray
    matchup_ids: np.ndarray
    feature_names: tuple[str, ...]
    fold_id: str
    zone: str = "train"

    def __post_init__(self) -> None:
        # --- type checks -----------------------------------------------
        _require_ndarray(self.X, "X")
        _require_ndarray(self.y, "y")
        _require_ndarray(self.matchup_ids, "matchup_ids")
        _require_fold_id(self.fold_id)

        # --- dimension checks ------------------------------------------
        _require_2d(self.X, "X")
        _require_1d(self.y, "y")
        _require_1d(self.matchup_ids, "matchup_ids")

        # --- row count consistency -------------------------------------
        n = self.X.shape[0]
        _require_row_match(n, "X", self.y, "y")
        _require_row_match(n, "X", self.matchup_ids, "matchup_ids")

        # --- feature names consistency ---------------------------------
        _require_feature_names(self.feature_names, self.X.shape[1])

        # --- label validity -------------------------------------------
        _require_binary_labels(self.y, "y")

        # --- zone guard -----------------------------------------------
        _require_zone(self.zone, "train", "TrainDataset")

        # --- immutability seal ----------------------------------------
        _seal(self.X)
        _seal(self.y)
        _seal(self.matchup_ids)


# ---------------------------------------------------------------------------
# CalDataset — Zone B
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class CalDataset:
    """
    Immutable container for Zone B (calibration) inputs and labels.

    Calibrators and the router may read .X and .y.  MetricEngine must not
    use CalDataset for final evaluation metrics.

    Parameters
    ----------
    X :
        Feature matrix, shape (n_samples, n_features).  Writeable=False.
    y :
        Binary outcome labels, shape (n_samples,), values in {0, 1}.
        Writeable=False.
    matchup_ids :
        Row-level provenance identifiers, shape (n_samples,).
        Writeable=False.
    feature_names :
        Column names for X; len must equal X.shape[1].
    fold_id :
        Non-empty fold identifier.
    zone :
        Must be exactly "cal".  Validated at construction time.
    """

    X: np.ndarray
    y: np.ndarray
    matchup_ids: np.ndarray
    feature_names: tuple[str, ...]
    fold_id: str
    zone: str = "cal"

    def __post_init__(self) -> None:
        _require_ndarray(self.X, "X")
        _require_ndarray(self.y, "y")
        _require_ndarray(self.matchup_ids, "matchup_ids")
        _require_fold_id(self.fold_id)

        _require_2d(self.X, "X")
        _require_1d(self.y, "y")
        _require_1d(self.matchup_ids, "matchup_ids")

        n = self.X.shape[0]
        _require_row_match(n, "X", self.y, "y")
        _require_row_match(n, "X", self.matchup_ids, "matchup_ids")

        _require_feature_names(self.feature_names, self.X.shape[1])
        _require_binary_labels(self.y, "y")
        _require_zone(self.zone, "cal", "CalDataset")

        _seal(self.X)
        _seal(self.y)
        _seal(self.matchup_ids)


# ---------------------------------------------------------------------------
# EvalDataset — Zone C inputs ONLY (NO LABELS)
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class EvalDataset:
    """
    Immutable container for Zone C (evaluation) INPUTS ONLY.

    This type is the primary type-level anti-leakage firewall.

    EvalDataset deliberately has NO .y attribute.  Any attempt to access .y
    on an EvalDataset instance raises AttributeError immediately — not None,
    not a sentinel value, not a silent failure.

    Evaluation labels are carried exclusively by EvalLabels, which is a
    physically distinct type loaded only at Stage 13 (MetricEngine.evaluate).
    LeakageGuard.assert_no_eval_labels_in_memory() enforces this lifecycle.

    Parameters
    ----------
    X :
        Feature matrix, shape (n_samples, n_features).  Writeable=False.
    matchup_ids :
        Row-level provenance identifiers, shape (n_samples,).
        Writeable=False.
    feature_names :
        Column names for X; len must equal X.shape[1].
    fold_id :
        Non-empty fold identifier.
    zone :
        Must be exactly "eval".  Validated at construction time.
    """

    X: np.ndarray
    matchup_ids: np.ndarray
    feature_names: tuple[str, ...]
    fold_id: str
    zone: str = "eval"

    # .y is NOT defined here.  This is intentional and non-negotiable.
    # Adding .y would violate the Type Firewall Law.

    def __post_init__(self) -> None:
        _require_ndarray(self.X, "X")
        _require_ndarray(self.matchup_ids, "matchup_ids")
        _require_fold_id(self.fold_id)

        _require_2d(self.X, "X")
        _require_1d(self.matchup_ids, "matchup_ids")

        n = self.X.shape[0]
        _require_row_match(n, "X", self.matchup_ids, "matchup_ids")

        _require_feature_names(self.feature_names, self.X.shape[1])
        _require_zone(self.zone, "eval", "EvalDataset")

        _seal(self.X)
        _seal(self.matchup_ids)


# ---------------------------------------------------------------------------
# EvalLabels — Zone C labels (Stage 13 only)
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class EvalLabels:
    """
    Immutable container for Zone C (evaluation) LABELS ONLY.

    EvalLabels is physically separate from EvalDataset.  The two types exist
    independently and must never be merged into a single object.  They share
    matchup_ids as a join key for metric computation.

    Lifecycle:
      - Not loaded into memory until Stage 13 (MetricEngine.evaluate()).
      - LeakageGuard detects EvalLabels by class name; any EvalLabels instance
        crossing a pre-Stage-13 boundary triggers LabelLeakageError.

    Parameters
    ----------
    y :
        Binary outcome labels, shape (n_samples,), values in {0, 1}.
        Writeable=False.
    matchup_ids :
        Row-level provenance identifiers, shape (n_samples,).  Must match
        EvalDataset.matchup_ids for the same fold.  Writeable=False.
    season :
        Evaluation season year (Zone C season).  Must be a positive integer.
    fold_id :
        Non-empty fold identifier.
    """

    y: np.ndarray
    matchup_ids: np.ndarray
    season: int
    fold_id: str

    def __post_init__(self) -> None:
        _require_ndarray(self.y, "y")
        _require_ndarray(self.matchup_ids, "matchup_ids")

        _require_1d(self.y, "y")
        _require_1d(self.matchup_ids, "matchup_ids")

        n = self.y.shape[0]
        _require_row_match(n, "y", self.matchup_ids, "matchup_ids")

        _require_binary_labels(self.y, "y")
        _require_fold_id(self.fold_id)

        if not isinstance(self.season, int) or self.season <= 0:
            raise ValueError(
                f"season must be a positive int; got {self.season!r}."
            )

        _seal(self.y)
        _seal(self.matchup_ids)
