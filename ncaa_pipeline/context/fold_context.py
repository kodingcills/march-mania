"""
fold_context.py
===============
Immutable temporal contract governing a single pipeline fold.

This module defines the FoldContext dataclass — the single source of truth for
fold temporal boundaries.  Every downstream stage that needs to know *which*
seasons are training / calibration / evaluation data, or *what* the hard
day-cutoff is, must read those values from a FoldContext instance.

No downstream stage may alter a FoldContext once it has been constructed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

# ---------------------------------------------------------------------------
# Module-level invariants (never relax these)
# ---------------------------------------------------------------------------

_REQUIRED_DAY_CUTOFF: Final[int] = 133
"""
The tournament-safe information boundary.

DayNum 133 is the first day of the NCAA Tournament play-in round.  Any game
result at or beyond this day must not cross the pre-tournament feature
boundary.  The value is sealed here as an immutable constant so that no call
site can accidentally weaken it.
"""

_ANOMALY_SEASONS: Final[frozenset[int]] = frozenset({2020})
"""
Seasons excluded from use as calibration or evaluation targets.

2020: COVID-19 cancellation left no tournament bracket — predictions cannot
be evaluated and calibration cannot be grounded.  This set is frozen to
prevent accidental modification at runtime.
"""


# ---------------------------------------------------------------------------
# FoldContext
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class FoldContext:
    """
    Immutable temporal contract for one pipeline fold.

    A fold describes a three-zone temporal split:

        train_seasons  →  historical data used to fit models
        cal_season     →  held-out season used for calibration / hyper-tuning
        eval_season    →  final held-out season used for evaluation only

    The zones must be strictly ordered and mutually disjoint.  The global
    day cutoff (133) is bound to the fold so every stage can read it from
    a single authoritative source.

    Construction raises ValueError immediately if any invariant is violated.
    Mutation is prohibited by the frozen dataclass contract.

    Parameters
    ----------
    fold_id:
        Human-readable, non-empty identifier for this fold.  Used in
        provenance checks to ensure artifacts are matched to the correct fold.
    train_seasons:
        Tuple of integer season years used for training.  Must be non-empty
        and contain no duplicates.
    cal_season:
        Integer season year used for calibration.  Must be strictly greater
        than max(train_seasons) and must not be an anomaly season (e.g. 2020).
    eval_season:
        Integer season year used for evaluation.  Must be strictly greater
        than cal_season and must not be an anomaly season.
    day_cutoff:
        The pre-tournament information boundary.  Must be exactly 133.
        Any other value is rejected to prevent accidental policy relaxation.
    random_seed:
        Integer seed for reproducible stochastic operations.  Default is 0.
    """

    fold_id: str
    train_seasons: tuple[int, ...]
    cal_season: int
    eval_season: int
    day_cutoff: int = 133
    random_seed: int = 0

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        self._validate_fold_id()
        self._validate_random_seed()
        self._validate_train_seasons()
        self._validate_season_disjointness()
        self._validate_season_ordering()
        self._validate_anomaly_seasons()
        self._validate_day_cutoff()

    def _validate_fold_id(self) -> None:
        if not isinstance(self.fold_id, str) or not self.fold_id.strip():
            raise ValueError(
                f"fold_id must be a non-empty string; got {self.fold_id!r}."
            )

    def _validate_random_seed(self) -> None:
        if not isinstance(self.random_seed, int):
            raise ValueError(
                f"random_seed must be an int; "
                f"got {type(self.random_seed).__name__!r}."
            )

    def _validate_train_seasons(self) -> None:
        if not self.train_seasons:
            raise ValueError(
                "train_seasons must be a non-empty tuple of season years."
            )
        if len(self.train_seasons) != len(set(self.train_seasons)):
            counts: dict[int, int] = {}
            for s in self.train_seasons:
                counts[s] = counts.get(s, 0) + 1
            duplicates = sorted(k for k, v in counts.items() if v > 1)
            raise ValueError(
                f"train_seasons must contain unique season years; "
                f"duplicate(s) found: {duplicates}."
            )

    def _validate_season_disjointness(self) -> None:
        train_set = set(self.train_seasons)
        if self.cal_season in train_set:
            raise ValueError(
                f"cal_season {self.cal_season} appears in train_seasons "
                f"{sorted(train_set)}; the three zones must be disjoint."
            )
        if self.eval_season in train_set:
            raise ValueError(
                f"eval_season {self.eval_season} appears in train_seasons "
                f"{sorted(train_set)}; the three zones must be disjoint."
            )
        if self.cal_season == self.eval_season:
            raise ValueError(
                f"cal_season and eval_season must be different; "
                f"both equal {self.cal_season}."
            )

    def _validate_season_ordering(self) -> None:
        max_train = max(self.train_seasons)
        if self.cal_season <= max_train:
            raise ValueError(
                f"cal_season {self.cal_season} must be strictly greater than "
                f"max(train_seasons) = {max_train}."
            )
        if self.eval_season <= self.cal_season:
            raise ValueError(
                f"eval_season {self.eval_season} must be strictly greater than "
                f"cal_season {self.cal_season}."
            )

    def _validate_anomaly_seasons(self) -> None:
        if self.cal_season in _ANOMALY_SEASONS:
            raise ValueError(
                f"cal_season {self.cal_season} is a known anomaly season "
                f"(no tournament data available) and may not be used for "
                f"calibration.  Anomaly seasons: {sorted(_ANOMALY_SEASONS)}."
            )
        if self.eval_season in _ANOMALY_SEASONS:
            raise ValueError(
                f"eval_season {self.eval_season} is a known anomaly season "
                f"(no tournament data available) and may not be used for "
                f"evaluation.  Anomaly seasons: {sorted(_ANOMALY_SEASONS)}."
            )

    def _validate_day_cutoff(self) -> None:
        if self.day_cutoff != _REQUIRED_DAY_CUTOFF:
            raise ValueError(
                f"day_cutoff must be exactly {_REQUIRED_DAY_CUTOFF} to "
                f"preserve the tournament-safe information boundary.  "
                f"Received {self.day_cutoff}.  "
                f"Relaxing this value is explicitly prohibited."
            )

    # ------------------------------------------------------------------
    # Read-only derived accessors
    # ------------------------------------------------------------------

    def max_train_season(self) -> int:
        """Return the most recent season in train_seasons."""
        return max(self.train_seasons)

    def all_seasons(self) -> tuple[int, ...]:
        """Return all fold seasons (train + cal + eval) sorted ascending."""
        combined = set(self.train_seasons) | {self.cal_season, self.eval_season}
        return tuple(sorted(combined))

    def describe(self) -> str:
        """Return a single-line human-readable summary of this fold."""
        return (
            f"FoldContext("
            f"fold_id={self.fold_id!r}, "
            f"train={sorted(self.train_seasons)}, "
            f"cal={self.cal_season}, "
            f"eval={self.eval_season}, "
            f"day_cutoff={self.day_cutoff}, "
            f"seed={self.random_seed})"
        )
