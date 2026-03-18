"""
leakage_guard.py
================
Runtime assertion checker fired at stage boundaries to catch data leakage.

This module is the second line of defense (after typed contracts and the
cutoff policy).  Its guards are called at explicit stage handoff points to
catch structural violations before they silently contaminate downstream
computations.

The three guard classes provided here are:
  - FutureDataViolationError  — future game rows crossed the stage boundary.
  - LabelLeakageError         — eval labels are present in intermediate state.
  - ProvenanceError           — artifact was produced by a different fold.

And the single guard class:
  - LeakageGuard              — three assert_* methods, one per error class.

Design principles
-----------------
* Stateless: no caches, no constructor config, no hidden state.
* Conservative: prefer false positives over false negatives.
* Deterministic: no randomness, no heuristics, no ML.
* Loud: all failures raise immediately with human-readable messages.
* Auditable: logic is short and reviewable.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from ncaa_pipeline.context.fold_context import FoldContext


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class FutureDataViolationError(Exception):
    """
    Raised when rows from beyond the permitted temporal frontier are found
    in a DataFrame being passed across a stage boundary.

    Attributes
    ----------
    violating_rows : pd.DataFrame
        Compact summary (Season, DayNum) of the offending rows.
    max_season     : int  — inclusive frontier season
    max_daynum     : int  — inclusive frontier DayNum within max_season
    """

    def __init__(
        self,
        violating_rows: pd.DataFrame,
        max_season: int,
        max_daynum: int,
    ) -> None:
        self.violating_rows = violating_rows
        self.max_season = max_season
        self.max_daynum = max_daynum
        n = len(violating_rows)
        sample = violating_rows.head(5).to_dict(orient="records")
        super().__init__(
            f"FutureDataViolation: {n} row(s) exceed the temporal frontier "
            f"(max_season={max_season}, max_daynum={max_daynum}).  "
            f"First up-to-5 offenders: {sample}."
        )


class LabelLeakageError(Exception):
    """
    Raised when evaluation labels or structurally suspicious objects are
    detected inside a state dictionary crossing a stage boundary.

    Attributes
    ----------
    detection_path : str  — dot-separated key path where leakage was found
    detail         : str  — human-readable description of what was detected
    """

    def __init__(self, detection_path: str, detail: str) -> None:
        self.detection_path = detection_path
        self.detail = detail
        super().__init__(
            f"LabelLeakage detected at path {detection_path!r}: {detail}"
        )


class ProvenanceError(Exception):
    """
    Raised when an artifact's recorded fold_id does not match the active
    FoldContext, indicating the artifact was produced by a different fold.

    Attributes
    ----------
    expected_fold_id : str  — fold_id from the active FoldContext
    actual_fold_id   : str  — fold_id found in the artifact metadata
    """

    def __init__(self, expected_fold_id: str, actual_fold_id: str) -> None:
        self.expected_fold_id = expected_fold_id
        self.actual_fold_id = actual_fold_id
        super().__init__(
            f"ProvenanceError: artifact fold_id {actual_fold_id!r} does not "
            f"match active fold {expected_fold_id!r}.  The artifact was "
            f"produced by a different fold and must not be used here."
        )


# ---------------------------------------------------------------------------
# Suspicious key registry
# ---------------------------------------------------------------------------

_SUSPICIOUS_KEY_NAMES: frozenset[str] = frozenset({
    "eval_labels",
    "evaluation_labels",
    "zone_c_labels",
    "tourney_results",
    "tournament_results",
    "actual_results",
})
"""
Normalized dict key names that indicate the presence of evaluation labels.

Normalization: strip whitespace, lower-case, replace hyphens/spaces with _.
New entries must be added here explicitly; no fuzzy matching is performed.
"""

_LEAKING_CLASS_NAMES: frozenset[str] = frozenset({
    "EvalLabels",
})
"""
Exact class names (type.__name__) that are never permitted in stage state.
"""


def _normalize_key(key: str) -> str:
    """Normalize a dict key for suspicious-name comparison."""
    return key.strip().lower().replace("-", "_").replace(" ", "_")


# ---------------------------------------------------------------------------
# LeakageGuard
# ---------------------------------------------------------------------------

class LeakageGuard:
    """
    Stateless runtime assertion checker for stage-boundary data leakage.

    Each public method is an assertion: it returns None on success and raises
    an appropriate exception on failure.  Methods never return booleans.

    Usage
    -----
    guard = LeakageGuard()
    guard.assert_no_future_games(df, max_season=2022, max_daynum=132)
    guard.assert_no_eval_labels_in_memory(state)
    guard.assert_artifact_provenance(metadata, context)
    """

    # ------------------------------------------------------------------
    # assert_no_future_games
    # ------------------------------------------------------------------

    def assert_no_future_games(
        self,
        df: pd.DataFrame,
        max_season: int,
        max_daynum: int,
    ) -> None:
        """
        Assert that *df* contains no rows beyond the temporal frontier.

        The frontier is an inclusive (max_season, max_daynum) pair:
          - Rows with Season > max_season                       → violation
          - Rows with Season == max_season and DayNum > max_daynum → violation
          - Rows with Season < max_season                       → always OK

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to inspect.  Must contain 'Season' and 'DayNum' columns.
        max_season : int
            The most recent season that is permitted.
        max_daynum : int
            The most recent DayNum permitted within max_season.

        Raises
        ------
        ValueError
            If 'Season' or 'DayNum' columns are absent.
        FutureDataViolationError
            If any row exceeds the frontier.
        """
        for col in ("Season", "DayNum"):
            if col not in df.columns:
                raise ValueError(
                    f"assert_no_future_games requires a '{col}' column; "
                    f"found columns: {list(df.columns)}."
                )

        future_season_mask: pd.Series = df["Season"] > max_season
        same_season_future_day_mask: pd.Series = (
            (df["Season"] == max_season) & (df["DayNum"] > max_daynum)
        )
        violation_mask: pd.Series = future_season_mask | same_season_future_day_mask

        if violation_mask.any():
            violating = df.loc[violation_mask, ["Season", "DayNum"]].copy()
            raise FutureDataViolationError(
                violating_rows=violating,
                max_season=max_season,
                max_daynum=max_daynum,
            )

    # ------------------------------------------------------------------
    # assert_no_eval_labels_in_memory
    # ------------------------------------------------------------------

    def assert_no_eval_labels_in_memory(self, state_dict: dict) -> None:
        """
        Assert that *state_dict* contains no evaluation labels.

        Performs a conservative structural scan:
          1. Any object whose ``type.__name__`` is in _LEAKING_CLASS_NAMES
             (e.g. ``"EvalLabels"``) → violation.
          2. Any dict key whose normalized form is in _SUSPICIOUS_KEY_NAMES
             → violation.

        The scan is recursive through dict, list, tuple, and set containers.
        It is deterministic and non-probabilistic.  Fuzzy / ML-style detection
        is explicitly excluded.

        Parameters
        ----------
        state_dict : dict
            Arbitrary stage-boundary state dictionary to inspect.

        Raises
        ------
        LabelLeakageError
            If any suspicious object or key is found; the error includes the
            dot-separated path where leakage was detected.
        """
        self._scan_value(state_dict, path="state")

    def _scan_value(self, value: Any, path: str) -> None:
        """Recursively scan *value*, reporting leakage at *path*."""
        # --- class-name check -------------------------------------------
        class_name = type(value).__name__
        if class_name in _LEAKING_CLASS_NAMES:
            raise LabelLeakageError(
                detection_path=path,
                detail=(
                    f"object of type {class_name!r} is not permitted in "
                    f"stage state (registered leaking class name)."
                ),
            )

        # --- container recursion ----------------------------------------
        if isinstance(value, dict):
            self._scan_dict(value, path)
        elif isinstance(value, (list, tuple)):
            for idx, item in enumerate(value):
                self._scan_value(item, path=f"{path}[{idx}]")
        elif isinstance(value, (set, frozenset)):
            for item in value:
                self._scan_value(item, path=f"{path}{{item}}")

    def _scan_dict(self, d: dict, path: str) -> None:
        """Scan a dict: check each key name then recurse into each value."""
        for key, val in d.items():
            child_path = f"{path}[{key!r}]"

            # key-name check (string keys only)
            if isinstance(key, str):
                normalized = _normalize_key(key)
                if normalized in _SUSPICIOUS_KEY_NAMES:
                    raise LabelLeakageError(
                        detection_path=child_path,
                        detail=(
                            f"dict key {key!r} (normalized: {normalized!r}) "
                            f"matches the suspicious-key registry."
                        ),
                    )

            # recurse into the value
            self._scan_value(val, path=child_path)

    # ------------------------------------------------------------------
    # assert_artifact_provenance
    # ------------------------------------------------------------------

    def assert_artifact_provenance(
        self,
        artifact_metadata: dict,
        context: FoldContext,
    ) -> None:
        """
        Assert that *artifact_metadata* was produced by *context*.

        Verifies that ``artifact_metadata["fold_id"]`` matches
        ``context.fold_id``.  Artifacts that carry no fold_id or a mismatched
        fold_id must never be used in the current fold's pipeline stages.

        Parameters
        ----------
        artifact_metadata : dict
            Metadata dict attached to an artifact.  Must contain 'fold_id'.
        context : FoldContext
            Active fold context against which provenance is checked.

        Raises
        ------
        ProvenanceError
            If 'fold_id' is absent from *artifact_metadata*, or if the
            recorded fold_id differs from *context.fold_id*.
        """
        if "fold_id" not in artifact_metadata:
            raise ProvenanceError(
                expected_fold_id=context.fold_id,
                actual_fold_id="<missing>",
            )
        recorded: str = artifact_metadata["fold_id"]
        if recorded != context.fold_id:
            raise ProvenanceError(
                expected_fold_id=context.fold_id,
                actual_fold_id=recorded,
            )
