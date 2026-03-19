"""
aim_mlflow_tracker.py
=====================
Write-only experiment tracking interface.

This module implements the project's telemetry discipline contract:
- write-only by protocol: no read, query, or retrieval methods
- provenance-automatic: fold_id and config_hash are injected into every
  backend call at the tracker level — callers cannot omit them
- transport-agnostic: the concrete backend is injected at construction time
  via TrackingBackend; no Aim or MLflow API surface is hardwired here

Design constraints (PHASE_PLAN Step 2, MASTER_ARCHITECTURE_V2 §Reproducibility):
  - no imports from ML, calibration, routing, manifest, or runner modules
  - no global mutable session state
  - no environment-variable reads that silently affect behavior
  - no read/query paths exposed on AimMLflowTracker
  - provenance is constructed and forwarded by the tracker, not the caller
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


# ---------------------------------------------------------------------------
# TrackingBackend Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class TrackingBackend(Protocol):
    """
    Structural protocol for write-side tracking backends.

    Any object implementing these three methods satisfies this protocol.
    No explicit inheritance from TrackingBackend is required — the protocol
    uses structural (duck-type) matching via ``typing.Protocol``.

    ``@runtime_checkable`` enables ``isinstance(obj, TrackingBackend)``
    checks at construction time for early failure on mis-typed backends.

    All methods are write-only by contract.  Implementations must not
    add read or query behavior through this interface.

    Provenance is always passed as a plain ``dict[str, str]`` containing
    at minimum ``fold_id`` and ``config_hash``.  Backends decide how to
    persist or embed provenance (e.g., MLflow tags, Aim context, sidecar
    JSON).  That persistence strategy is backend-internal and does not
    concern the tracker layer.
    """

    def log_metrics(
        self,
        metrics: dict[str, float],
        provenance: dict[str, str],
        step: int | None,
    ) -> None: ...

    def log_params(
        self,
        params: dict[str, object],
        provenance: dict[str, str],
    ) -> None: ...

    def log_artifact(
        self,
        path: str,
        provenance: dict[str, str],
        artifact_name: str | None,
    ) -> None: ...


# ---------------------------------------------------------------------------
# Validation helpers (module-private)
# ---------------------------------------------------------------------------


def _require_nonempty_string(value: object, name: str) -> None:
    """Raise ValueError if value is not a non-empty, non-whitespace string."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"{name} must be a non-empty string; got {value!r}."
        )


def _require_config_hash(config_hash: object) -> None:
    """
    Raise TypeError or ValueError if config_hash is not a valid SHA256 hex digest.

    A valid config_hash is:
    - a str
    - exactly 64 characters
    - lowercase
    - valid hexadecimal (digits 0-9 and a-f only)
    """
    if not isinstance(config_hash, str):
        raise TypeError(
            f"config_hash must be a str; "
            f"got {type(config_hash).__name__!r}."
        )
    if len(config_hash) != 64:
        raise ValueError(
            f"config_hash must be exactly 64 characters; "
            f"got length {len(config_hash)}."
        )
    if config_hash != config_hash.lower():
        raise ValueError(
            f"config_hash must be lowercase; got {config_hash!r}."
        )
    try:
        int(config_hash, 16)
    except ValueError:
        raise ValueError(
            f"config_hash must be a valid hexadecimal string (0-9, a-f); "
            f"got {config_hash!r}."
        )


def _require_metrics(metrics: object) -> None:
    """
    Raise TypeError if metrics is not a dict[str, int | float].

    Integer values are accepted in addition to float since tracking
    backends handle both identically.
    """
    if not isinstance(metrics, dict):
        raise TypeError(
            f"metrics must be a dict[str, float]; "
            f"got {type(metrics).__name__!r}."
        )
    for key, value in metrics.items():
        if not isinstance(key, str):
            raise TypeError(
                f"metrics keys must be str; "
                f"got {type(key).__name__!r} for key {key!r}."
            )
        if not isinstance(value, (int, float)):
            raise TypeError(
                f"metrics values must be numeric (int or float); "
                f"got {type(value).__name__!r} for key {key!r}."
            )


def _require_params(params: object) -> None:
    """Raise TypeError if params is not a dict with string keys."""
    if not isinstance(params, dict):
        raise TypeError(
            f"params must be a dict; got {type(params).__name__!r}."
        )
    for key in params:
        if not isinstance(key, str):
            raise TypeError(
                f"params keys must be str; "
                f"got {type(key).__name__!r} for key {key!r}."
            )


# ---------------------------------------------------------------------------
# AimMLflowTracker
# ---------------------------------------------------------------------------


class AimMLflowTracker:
    """
    Write-only experiment tracking interface.

    ``AimMLflowTracker`` wraps a ``TrackingBackend`` and enforces:

    - **Provenance-automatic:** ``fold_id`` and ``config_hash`` are injected
      into every backend call.  Callers cannot forget or omit provenance.
    - **Write-only:** no ``read()``, ``query()``, ``list_runs()``,
      ``get_best_run()``, or any retrieval method exists.
    - **Identity-locked:** ``fold_id`` and ``config_hash`` are fixed at
      construction.  A new tracker must be constructed per fold.
    - **Transport-agnostic:** any object satisfying ``TrackingBackend``
      protocol is a valid backend — Aim, MLflow, or an in-process fake.

    Parameters
    ----------
    backend :
        Any object satisfying ``TrackingBackend`` protocol.  In production,
        this will be a thin Aim or MLflow adapter.  In tests, pass a
        ``CapturingBackend`` or equivalent in-process fake.
    fold_id :
        Non-empty fold identifier matching the active ``FoldContext.fold_id``.
        Locked at construction; cannot be changed without a new tracker.
    config_hash :
        64-character lowercase SHA256 hex string from
        ``PipelineConfig.config_hash()``.  Locked at construction.
    """

    __slots__ = ("_backend", "_fold_id", "_config_hash")

    def __init__(
        self,
        backend: TrackingBackend,
        fold_id: str,
        config_hash: str,
    ) -> None:
        if not isinstance(backend, TrackingBackend):
            raise TypeError(
                f"backend must implement TrackingBackend protocol; "
                f"got {type(backend).__name__!r}."
            )
        _require_nonempty_string(fold_id, "fold_id")
        _require_config_hash(config_hash)

        self._backend: TrackingBackend = backend
        self._fold_id: str = fold_id
        self._config_hash: str = config_hash

    # ------------------------------------------------------------------
    # Read-only identity properties
    # ------------------------------------------------------------------

    @property
    def fold_id(self) -> str:
        """The fold identifier locked at construction time."""
        return self._fold_id

    @property
    def config_hash(self) -> str:
        """The config hash locked at construction time."""
        return self._config_hash

    # ------------------------------------------------------------------
    # Write-only tracking interface
    # ------------------------------------------------------------------

    def log_metrics(
        self,
        metrics: dict[str, float],
        step: int | None = None,
    ) -> None:
        """
        Log scalar metrics to the backend.

        Provenance (``fold_id``, ``config_hash``) is automatically injected.

        Parameters
        ----------
        metrics :
            Mapping of metric name → numeric value.  Keys must be strings;
            values must be ``int`` or ``float``.
        step :
            Optional training step or iteration index for time-series metrics.
            Must be an ``int`` or ``None``.
        """
        _require_metrics(metrics)
        if step is not None and not isinstance(step, int):
            raise TypeError(
                f"step must be an int or None; got {type(step).__name__!r}."
            )
        self._backend.log_metrics(
            metrics=dict(metrics),
            provenance=self._provenance(),
            step=step,
        )

    def log_params(
        self,
        params: dict[str, object],
    ) -> None:
        """
        Log parameters or configuration snapshot to the backend.

        Provenance (``fold_id``, ``config_hash``) is automatically injected.

        Typical use: log ``PipelineConfig.to_canonical_dict()`` output or
        a subset of configuration fields for the current fold run.

        Parameters
        ----------
        params :
            Mapping of parameter name → value.  Keys must be strings.
            Values should be JSON-serializable primitives; the backend is
            responsible for serialization.
        """
        _require_params(params)
        self._backend.log_params(
            params=dict(params),
            provenance=self._provenance(),
        )

    def log_artifact(
        self,
        path: str,
        artifact_name: str | None = None,
    ) -> None:
        """
        Log an artifact path to the backend.

        Provenance (``fold_id``, ``config_hash``) is automatically embedded
        in the call.  Callers are never trusted to supply provenance
        manually — it is always injected from the locked construction-time
        identity.

        Parameters
        ----------
        path :
            Path to the artifact file or directory.  Must be a non-empty string.
        artifact_name :
            Optional human-readable name for the artifact.  If ``None``, the
            backend may derive a name from the path basename.
        """
        _require_nonempty_string(path, "path")
        if artifact_name is not None and not isinstance(artifact_name, str):
            raise TypeError(
                f"artifact_name must be a str or None; "
                f"got {type(artifact_name).__name__!r}."
            )
        self._backend.log_artifact(
            path=path,
            provenance=self._provenance(),
            artifact_name=artifact_name,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _provenance(self) -> dict[str, str]:
        """
        Construct the provenance dict for the current tracker identity.

        Returns a fresh dict on every call — never a shared mutable
        reference.  Callers on the backend side may mutate or extend the
        provenance dict without affecting the tracker's internal state.
        """
        return {
            "fold_id": self._fold_id,
            "config_hash": self._config_hash,
        }
