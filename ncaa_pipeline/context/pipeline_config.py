"""
pipeline_config.py
==================
Immutable, hashable, canonical pipeline configuration contract.

This module defines PipelineConfig — the single frozen identity object that
governs all tunable behavior of the pipeline:

    - experimental path enablement
    - LightGBM hyperparameters (baseline path)
    - TabPFN hyperparameters (experimental path)
    - calibration bounds and optimization settings
    - router / ensemble settings
    - evaluation and bootstrap CI settings
    - Massey ordinal feature extraction settings
    - global random seed

Every pipeline run is identified by a deterministic SHA256 config hash
computed from the full logical state of this object.

Design constraints (PHASE_PLAN Step 1, MASTER_ARCHITECTURE_V2 §Reproducibility):
  - @dataclass(frozen=True, slots=True) throughout; no mutable nested containers
  - no runtime state (no file handles, DataFrames, model objects)
  - no environment-variable-dependent defaults
  - config_hash() uses sorted-key JSON → SHA256, never Python's built-in hash()
  - no imports from ML, tracking, or manifest libraries
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from typing import Final

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

_APPROVED_SUBSAMPLE_STRATEGIES: Final[frozenset[str]] = frozenset(
    {"recency_weighted", "uniform"}
)


# ---------------------------------------------------------------------------
# Nested frozen config groups
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ExperimentalFlags:
    """
    Explicit opt-in flags for experimental pipeline paths.

    All flags default to False.  Experimental components must not activate
    unless their corresponding flag is explicitly set to True.  These flags
    participate in config_hash() so that experimental runs are identifiable
    and not confused with default-path runs.

    Parameters
    ----------
    enable_tabpfn :
        If True, the TabPFN model is included in the ensemble.
        TabPFN remains experimental until subsampling is validated on
        >= 3 folds with the accepted acceptance criterion.
    enable_graph_features :
        If True, graph-derived scalar features are appended to the feature
        matrix.  Experimental until validated on >= 3 folds.
    enable_weighted_router :
        If True, the RSC performance-weighted router is used instead of
        the default simple-average router.
    """

    enable_tabpfn: bool = False
    enable_graph_features: bool = False
    enable_weighted_router: bool = False


@dataclass(frozen=True, slots=True)
class LightGBMConfig:
    """
    Hyperparameters for the baseline LightGBM binary model.

    These values represent the canonical defaults approved in
    MASTER_ARCHITECTURE_V2 Revision R1.  ``max_delta_step`` provides gradient
    clipping in place of the rejected Huber objective.  ``deterministic=True``
    and ``num_threads=1`` enforce serial, reproducible execution.

    Parameters
    ----------
    objective :
        LightGBM loss objective.  Must be ``"binary"`` for the pipeline's
        calibration-compatible default path.
    metric :
        Evaluation metric reported during training.
    num_leaves :
        Maximum number of leaves per tree.  Must be >= 1.
    learning_rate :
        Gradient descent step size.  Must be > 0.
    n_estimators :
        Number of boosting rounds.  Must be >= 1.
    max_delta_step :
        Gradient clipping bound.  Replaces Huber loss; controls tail
        behavior.  Must be > 0.
    deterministic :
        If True, LightGBM uses deterministic algorithms for reproducibility.
    num_threads :
        Number of CPU threads.  1 enforces serial execution for
        reproducibility.  Must be >= 1.
    """

    objective: str = "binary"
    metric: str = "binary_logloss"
    num_leaves: int = 31
    learning_rate: float = 0.05
    n_estimators: int = 500
    max_delta_step: float = 1.0
    deterministic: bool = True
    num_threads: int = 1

    def __post_init__(self) -> None:
        if self.num_leaves < 1:
            raise ValueError(
                f"num_leaves must be >= 1; got {self.num_leaves}."
            )
        if self.n_estimators < 1:
            raise ValueError(
                f"n_estimators must be >= 1; got {self.n_estimators}."
            )
        if self.num_threads < 1:
            raise ValueError(
                f"num_threads must be >= 1; got {self.num_threads}."
            )
        if self.learning_rate <= 0.0:
            raise ValueError(
                f"learning_rate must be > 0; got {self.learning_rate}."
            )
        if self.max_delta_step <= 0.0:
            raise ValueError(
                f"max_delta_step must be > 0; got {self.max_delta_step}."
            )


@dataclass(frozen=True, slots=True)
class TabPFNConfig:
    """
    Hyperparameters for the experimental TabPFN path.

    These settings are active only when ``ExperimentalFlags.enable_tabpfn``
    is True.  They are stored in ``PipelineConfig`` regardless so that the
    full config state is captured in the hash even on the default path — a
    no-op TabPFN config is still part of run identity.

    Parameters
    ----------
    max_rows :
        Maximum number of training rows TabPFN will accept.  Rows beyond
        this limit are subsampled according to ``subsample_strategy``.
        Must be >= 1.
    subsample_strategy :
        Strategy for reducing training rows when ``max_rows`` is exceeded.
        Supported values: ``"recency_weighted"``, ``"uniform"``.
    n_ensemble :
        Number of TabPFN ensemble members.  Must be >= 1.
    """

    max_rows: int = 10_000
    subsample_strategy: str = "recency_weighted"
    n_ensemble: int = 16

    def __post_init__(self) -> None:
        if self.max_rows < 1:
            raise ValueError(
                f"max_rows must be >= 1; got {self.max_rows}."
            )
        if self.subsample_strategy not in _APPROVED_SUBSAMPLE_STRATEGIES:
            raise ValueError(
                f"subsample_strategy must be one of "
                f"{sorted(_APPROVED_SUBSAMPLE_STRATEGIES)}; "
                f"got {self.subsample_strategy!r}."
            )
        if self.n_ensemble < 1:
            raise ValueError(
                f"n_ensemble must be >= 1; got {self.n_ensemble}."
            )


@dataclass(frozen=True, slots=True)
class CalibrationConfig:
    """
    Bounds and optimization settings for per-model Platt calibration.

    The calibration family is a 2-parameter affine-logit transformation
    parameterized by (a, b).  These bounds constrain the optimizer's search
    space during calibration fitting.

    Note: this config stores calibration settings only.
    The NelderMeadCalibrator implementation is Phase 2 Step 2+ scope.

    Parameters
    ----------
    bounds_a :
        ``(min, max)`` bounds for the scaling parameter a.  Must satisfy
        ``min < max``.
    bounds_b :
        ``(min, max)`` bounds for the shift parameter b.  Must satisfy
        ``min < max``.
    lambda_reg :
        Regularization strength applied during calibration optimization.
        Must be >= 0.
    n_restarts :
        Number of Nelder-Mead restarts to mitigate local minima.
        Must be >= 1.
    c :
        Kaggle Logistic Brier Score scaling constant.  Must be > 0.
    """

    bounds_a: tuple[float, float] = (0.3, 3.0)
    bounds_b: tuple[float, float] = (-2.0, 2.0)
    lambda_reg: float = 0.1
    n_restarts: int = 5
    c: float = 7.0

    def __post_init__(self) -> None:
        if self.bounds_a[0] >= self.bounds_a[1]:
            raise ValueError(
                f"bounds_a must be ordered (min < max); got {self.bounds_a}."
            )
        if self.bounds_b[0] >= self.bounds_b[1]:
            raise ValueError(
                f"bounds_b must be ordered (min < max); got {self.bounds_b}."
            )
        if self.lambda_reg < 0.0:
            raise ValueError(
                f"lambda_reg must be >= 0; got {self.lambda_reg}."
            )
        if self.n_restarts < 1:
            raise ValueError(
                f"n_restarts must be >= 1; got {self.n_restarts}."
            )
        if self.c <= 0.0:
            raise ValueError(
                f"c must be > 0; got {self.c}."
            )


@dataclass(frozen=True, slots=True)
class RouterConfig:
    """
    Settings for the router / ensemble weighting layer.

    Parameters
    ----------
    tau :
        Temperature parameter for the performance-weighted router.
        ``tau=1.0`` corresponds to standard softmax weighting.
        Only active when ``ExperimentalFlags.enable_weighted_router`` is True.
        Must be > 0.
    """

    tau: float = 1.0

    def __post_init__(self) -> None:
        if self.tau <= 0.0:
            raise ValueError(
                f"tau must be > 0; got {self.tau}."
            )


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    """
    Settings for cross-fold evaluation and bootstrap confidence intervals.

    Parameters
    ----------
    bootstrap_n :
        Number of bootstrap resamples for CI computation.  Must be >= 1.
    bootstrap_ci :
        Target confidence level.  Must be strictly between 0 and 1.
    """

    bootstrap_n: int = 10_000
    bootstrap_ci: float = 0.95

    def __post_init__(self) -> None:
        if self.bootstrap_n < 1:
            raise ValueError(
                f"bootstrap_n must be >= 1; got {self.bootstrap_n}."
            )
        if not (0.0 < self.bootstrap_ci < 1.0):
            raise ValueError(
                f"bootstrap_ci must be strictly between 0 and 1; "
                f"got {self.bootstrap_ci}."
            )


@dataclass(frozen=True, slots=True)
class MasseyConfig:
    """
    Settings for Massey ordinal feature extraction.

    Parameters
    ----------
    system_allowlist :
        If not None, only Massey rating systems in this tuple are used as
        features.  If None, all available systems are used.  Each entry
        must be a non-empty string.
    stale_day_threshold :
        Massey systems whose latest ranking day is strictly below this
        threshold are treated as stale and excluded from the feature matrix.
        Must be >= 0.
    """

    system_allowlist: tuple[str, ...] | None = None
    stale_day_threshold: int = 128

    def __post_init__(self) -> None:
        if self.stale_day_threshold < 0:
            raise ValueError(
                f"stale_day_threshold must be >= 0; "
                f"got {self.stale_day_threshold}."
            )
        if self.system_allowlist is not None:
            if not isinstance(self.system_allowlist, tuple):
                raise TypeError(
                    f"system_allowlist must be a tuple[str, ...] or None; "
                    f"got {type(self.system_allowlist).__name__!r}."
                )
            for i, item in enumerate(self.system_allowlist):
                if not isinstance(item, str):
                    raise TypeError(
                        f"system_allowlist[{i}] must be a str; "
                        f"got {type(item).__name__!r}."
                    )


# ---------------------------------------------------------------------------
# Top-level pipeline config
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    """
    Canonical, frozen pipeline configuration contract.

    ``PipelineConfig`` is the single identity object for a pipeline run.  It
    aggregates all grouped configuration and produces a deterministic SHA256
    hash via ``config_hash()``.

    Every field is frozen after construction.  No runtime mutation is possible.
    No runtime objects (DataFrames, model handles, file descriptors) may be
    stored in this object.

    The ``config_hash()`` is stable across Python processes and machines.  It
    does not depend on Python's built-in ``hash()``, field insertion order, or
    object identity.

    Parameters
    ----------
    experimental :
        Experimental path enablement flags.
    lightgbm :
        Baseline LightGBM model hyperparameters.
    tabpfn :
        TabPFN experimental model settings.
    calibration :
        Calibration optimizer bounds and settings.
    router :
        Ensemble router settings.
    evaluation :
        Evaluation and bootstrap CI settings.
    massey :
        Massey ordinal feature extraction settings.
    random_seed :
        Global random seed for all stochastic operations in the pipeline.
    """

    experimental: ExperimentalFlags = field(default_factory=ExperimentalFlags)
    lightgbm: LightGBMConfig = field(default_factory=LightGBMConfig)
    tabpfn: TabPFNConfig = field(default_factory=TabPFNConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    router: RouterConfig = field(default_factory=RouterConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    massey: MasseyConfig = field(default_factory=MasseyConfig)
    random_seed: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.random_seed, int):
            raise TypeError(
                f"random_seed must be an int; "
                f"got {type(self.random_seed).__name__!r}."
            )

    # ------------------------------------------------------------------
    # Canonical serialization
    # ------------------------------------------------------------------

    def _to_canonical_dict(self) -> dict:
        """
        Convert config to a fully normalized, JSON-serializable dict.

        ``dataclasses.asdict()`` recursively converts all nested frozen
        dataclasses to plain dicts and all tuples to lists.  The result
        contains only ``str``, ``int``, ``float``, ``bool``, ``list``,
        ``dict``, and ``None`` — all JSON-serializable primitives with
        deterministic encoding.

        This output is fed directly to ``json.dumps(sort_keys=True)`` which
        recursively sorts all mapping keys, neutralizing any dict-insertion-
        order differences.
        """
        return dataclasses.asdict(self)

    def to_canonical_dict(self) -> dict:
        """
        Return the canonical dict representation of this config.

        Suitable for hashing, logging, and later manifest inclusion.
        All nested dataclasses are recursively expanded into plain dicts.
        All tuple fields are represented as lists.
        """
        return self._to_canonical_dict()

    # ------------------------------------------------------------------
    # Hashing
    # ------------------------------------------------------------------

    def config_hash(self) -> str:
        """
        Compute a deterministic SHA256 hex digest of the full config state.

        Returns a 64-character lowercase hexadecimal string that:

        - reflects the complete logical state of all config groups
        - is identical for logically equivalent ``PipelineConfig`` instances
        - changes when any meaningful field changes
        - is stable across Python processes and machines
        - does not depend on Python's built-in ``hash()``

        Hashing pipeline:

        1. Convert config to canonical dict via ``dataclasses.asdict()``.
           All nested dataclasses → dicts; all tuples → lists; ``None`` → null.
        2. ``json.dumps(sort_keys=True, separators=(",", ":"), ensure_ascii=True)``
           Compact, key-sorted JSON.  ``sort_keys=True`` is applied recursively
           to every mapping at every nesting depth, neutralizing insertion-order
           differences.
        3. Encode to UTF-8 bytes.
        4. SHA256 hex digest.
        """
        canonical = self._to_canonical_dict()
        serialized = json.dumps(
            canonical,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
