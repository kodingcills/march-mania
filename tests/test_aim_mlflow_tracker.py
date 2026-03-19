"""
test_aim_mlflow_tracker.py
==========================
Tests for AimMLflowTracker — the write-only experiment tracking interface.

Proves:
  A. Construction — valid and invalid arguments
  B. log_metrics — call forwarding, provenance injection, type validation
  C. log_params — call forwarding, provenance injection, type validation
  D. log_artifact — call forwarding, provenance injection, type validation
  E. Provenance correctness — fold_id and config_hash always attached
  F. No read interface — query/retrieval methods must not exist
  G. No architecture creep — no forbidden imports in the tracker module

All tests are:
  - deterministic
  - isolated (no shared mutable state across tests)
  - self-contained (no external files, no network calls, no real backends)
  - free of future-phase dependencies
"""

from __future__ import annotations

import ast
import pathlib

import pytest

from ncaa_pipeline.tracking.aim_mlflow_tracker import (
    AimMLflowTracker,
    TrackingBackend,
)


# ---------------------------------------------------------------------------
# Test constants
# ---------------------------------------------------------------------------

_VALID_FOLD_ID: str = "test-fold-2023"
_VALID_CONFIG_HASH: str = "a" * 64  # 64-char lowercase hex string


# ---------------------------------------------------------------------------
# CapturingBackend — in-process fake for assertion
# ---------------------------------------------------------------------------


class CapturingBackend:
    """
    In-process fake backend that records every call for assertion.

    Satisfies the TrackingBackend protocol structurally — no inheritance
    required.  Used exclusively in tests; never appears in production code.
    """

    def __init__(self) -> None:
        self.metrics_calls: list[dict] = []
        self.params_calls: list[dict] = []
        self.artifact_calls: list[dict] = []

    def log_metrics(
        self,
        metrics: dict[str, float],
        provenance: dict[str, str],
        step: int | None,
    ) -> None:
        self.metrics_calls.append(
            {"metrics": metrics, "provenance": dict(provenance), "step": step}
        )

    def log_params(
        self,
        params: dict[str, object],
        provenance: dict[str, str],
    ) -> None:
        self.params_calls.append(
            {"params": params, "provenance": dict(provenance)}
        )

    def log_artifact(
        self,
        path: str,
        provenance: dict[str, str],
        artifact_name: str | None,
    ) -> None:
        self.artifact_calls.append(
            {
                "path": path,
                "provenance": dict(provenance),
                "artifact_name": artifact_name,
            }
        )


# ---------------------------------------------------------------------------
# Shared factory helpers
# ---------------------------------------------------------------------------


def _make_backend() -> CapturingBackend:
    return CapturingBackend()


def _make_tracker(
    backend: CapturingBackend | None = None,
    fold_id: str = _VALID_FOLD_ID,
    config_hash: str = _VALID_CONFIG_HASH,
) -> AimMLflowTracker:
    if backend is None:
        backend = _make_backend()
    return AimMLflowTracker(backend=backend, fold_id=fold_id, config_hash=config_hash)


# ===========================================================================
# A. Construction
# ===========================================================================


class TestConstruction:
    """AimMLflowTracker must validate all constructor arguments at instantiation."""

    def test_valid_construction_succeeds(self) -> None:
        tracker = _make_tracker()
        assert tracker.fold_id == _VALID_FOLD_ID
        assert tracker.config_hash == _VALID_CONFIG_HASH

    def test_fold_id_accessible_as_property(self) -> None:
        tracker = _make_tracker(fold_id="fold-abc")
        assert tracker.fold_id == "fold-abc"

    def test_config_hash_accessible_as_property(self) -> None:
        h = "b" * 64
        tracker = _make_tracker(config_hash=h)
        assert tracker.config_hash == h

    def test_empty_fold_id_raises(self) -> None:
        with pytest.raises(ValueError, match="fold_id"):
            _make_tracker(fold_id="")

    def test_whitespace_fold_id_raises(self) -> None:
        with pytest.raises(ValueError, match="fold_id"):
            _make_tracker(fold_id="   ")

    def test_config_hash_wrong_length_raises(self) -> None:
        with pytest.raises(ValueError, match="64"):
            _make_tracker(config_hash="abc123")

    def test_config_hash_uppercase_raises(self) -> None:
        with pytest.raises(ValueError, match="lowercase"):
            _make_tracker(config_hash="A" * 64)

    def test_config_hash_non_hex_raises(self) -> None:
        with pytest.raises(ValueError, match="hexadecimal"):
            _make_tracker(config_hash="z" * 64)

    def test_config_hash_non_string_raises(self) -> None:
        with pytest.raises(TypeError, match="config_hash"):
            _make_tracker(config_hash=None)  # type: ignore[arg-type]

    def test_invalid_backend_raises(self) -> None:
        with pytest.raises(TypeError, match="TrackingBackend"):
            AimMLflowTracker(
                backend=object(),  # type: ignore[arg-type]
                fold_id=_VALID_FOLD_ID,
                config_hash=_VALID_CONFIG_HASH,
            )

    def test_capturing_backend_satisfies_protocol(self) -> None:
        """CapturingBackend must pass isinstance check via runtime_checkable."""
        assert isinstance(_make_backend(), TrackingBackend)

    def test_config_hash_mixed_hex_valid(self) -> None:
        """Mixed valid hex digits (0-9, a-f) must be accepted."""
        h = ("0123456789abcdef" * 4)[:64]
        tracker = _make_tracker(config_hash=h)
        assert tracker.config_hash == h


# ===========================================================================
# B. log_metrics
# ===========================================================================


class TestLogMetrics:
    """log_metrics must forward to backend and attach provenance."""

    def test_basic_metrics_forwarded(self) -> None:
        backend = _make_backend()
        tracker = _make_tracker(backend=backend)
        tracker.log_metrics({"brier": 0.15, "logloss": 0.42})
        assert len(backend.metrics_calls) == 1
        call = backend.metrics_calls[0]
        assert call["metrics"] == {"brier": 0.15, "logloss": 0.42}

    def test_step_forwarded_when_provided(self) -> None:
        backend = _make_backend()
        tracker = _make_tracker(backend=backend)
        tracker.log_metrics({"loss": 0.5}, step=10)
        assert backend.metrics_calls[0]["step"] == 10

    def test_step_none_by_default(self) -> None:
        backend = _make_backend()
        tracker = _make_tracker(backend=backend)
        tracker.log_metrics({"loss": 0.5})
        assert backend.metrics_calls[0]["step"] is None

    def test_step_zero_accepted(self) -> None:
        backend = _make_backend()
        tracker = _make_tracker(backend=backend)
        tracker.log_metrics({"loss": 0.5}, step=0)
        assert backend.metrics_calls[0]["step"] == 0

    def test_integer_metric_values_accepted(self) -> None:
        """int values are valid metrics (equivalent to float for tracking)."""
        backend = _make_backend()
        tracker = _make_tracker(backend=backend)
        tracker.log_metrics({"n_samples": 1000})
        assert backend.metrics_calls[0]["metrics"]["n_samples"] == 1000

    def test_empty_metrics_dict_accepted(self) -> None:
        backend = _make_backend()
        tracker = _make_tracker(backend=backend)
        tracker.log_metrics({})
        assert len(backend.metrics_calls) == 1

    def test_non_dict_metrics_raises(self) -> None:
        tracker = _make_tracker()
        with pytest.raises(TypeError, match="metrics"):
            tracker.log_metrics([("loss", 0.5)])  # type: ignore[arg-type]

    def test_non_str_metric_key_raises(self) -> None:
        tracker = _make_tracker()
        with pytest.raises(TypeError, match="metrics keys"):
            tracker.log_metrics({1: 0.5})  # type: ignore[arg-type]

    def test_non_numeric_metric_value_raises(self) -> None:
        tracker = _make_tracker()
        with pytest.raises(TypeError, match="metrics values"):
            tracker.log_metrics({"loss": "high"})  # type: ignore[arg-type]

    def test_non_int_step_raises(self) -> None:
        tracker = _make_tracker()
        with pytest.raises(TypeError, match="step"):
            tracker.log_metrics({"loss": 0.5}, step=1.0)  # type: ignore[arg-type]

    def test_multiple_calls_each_recorded(self) -> None:
        backend = _make_backend()
        tracker = _make_tracker(backend=backend)
        tracker.log_metrics({"a": 1.0}, step=0)
        tracker.log_metrics({"b": 2.0}, step=1)
        assert len(backend.metrics_calls) == 2

    def test_metrics_dict_is_copy(self) -> None:
        """Mutating the original dict after the call must not affect the recorded call."""
        backend = _make_backend()
        tracker = _make_tracker(backend=backend)
        metrics = {"loss": 0.5}
        tracker.log_metrics(metrics)
        metrics["loss"] = 99.0
        assert backend.metrics_calls[0]["metrics"]["loss"] == 0.5


# ===========================================================================
# C. log_params
# ===========================================================================


class TestLogParams:
    """log_params must forward to backend and attach provenance."""

    def test_basic_params_forwarded(self) -> None:
        backend = _make_backend()
        tracker = _make_tracker(backend=backend)
        tracker.log_params({"num_leaves": 31, "objective": "binary"})
        assert len(backend.params_calls) == 1
        call = backend.params_calls[0]
        assert call["params"] == {"num_leaves": 31, "objective": "binary"}

    def test_mixed_value_types_accepted(self) -> None:
        """Params values may be str, int, float, bool, None, or nested dicts."""
        backend = _make_backend()
        tracker = _make_tracker(backend=backend)
        tracker.log_params(
            {
                "lr": 0.05,
                "n_est": 500,
                "objective": "binary",
                "deterministic": True,
                "extra": None,
            }
        )
        assert len(backend.params_calls) == 1

    def test_empty_params_accepted(self) -> None:
        backend = _make_backend()
        tracker = _make_tracker(backend=backend)
        tracker.log_params({})
        assert len(backend.params_calls) == 1

    def test_non_dict_params_raises(self) -> None:
        tracker = _make_tracker()
        with pytest.raises(TypeError, match="params"):
            tracker.log_params([("a", 1)])  # type: ignore[arg-type]

    def test_non_str_params_key_raises(self) -> None:
        tracker = _make_tracker()
        with pytest.raises(TypeError, match="params keys"):
            tracker.log_params({1: "value"})  # type: ignore[arg-type]

    def test_params_dict_is_copy(self) -> None:
        backend = _make_backend()
        tracker = _make_tracker(backend=backend)
        params = {"lr": 0.05}
        tracker.log_params(params)
        params["lr"] = 99.0
        assert backend.params_calls[0]["params"]["lr"] == 0.05


# ===========================================================================
# D. log_artifact
# ===========================================================================


class TestLogArtifact:
    """log_artifact must forward to backend and attach provenance."""

    def test_basic_artifact_forwarded(self) -> None:
        backend = _make_backend()
        tracker = _make_tracker(backend=backend)
        tracker.log_artifact("/tmp/predictions.csv")
        assert len(backend.artifact_calls) == 1
        call = backend.artifact_calls[0]
        assert call["path"] == "/tmp/predictions.csv"

    def test_artifact_name_forwarded_when_provided(self) -> None:
        backend = _make_backend()
        tracker = _make_tracker(backend=backend)
        tracker.log_artifact("/tmp/model.pkl", artifact_name="lgbm_fold3")
        assert backend.artifact_calls[0]["artifact_name"] == "lgbm_fold3"

    def test_artifact_name_none_by_default(self) -> None:
        backend = _make_backend()
        tracker = _make_tracker(backend=backend)
        tracker.log_artifact("/tmp/model.pkl")
        assert backend.artifact_calls[0]["artifact_name"] is None

    def test_empty_path_raises(self) -> None:
        tracker = _make_tracker()
        with pytest.raises(ValueError, match="path"):
            tracker.log_artifact("")

    def test_whitespace_path_raises(self) -> None:
        tracker = _make_tracker()
        with pytest.raises(ValueError, match="path"):
            tracker.log_artifact("   ")

    def test_non_str_artifact_name_raises(self) -> None:
        tracker = _make_tracker()
        with pytest.raises(TypeError, match="artifact_name"):
            tracker.log_artifact("/tmp/out.csv", artifact_name=42)  # type: ignore[arg-type]

    def test_multiple_artifact_calls_recorded(self) -> None:
        backend = _make_backend()
        tracker = _make_tracker(backend=backend)
        tracker.log_artifact("/tmp/a.csv")
        tracker.log_artifact("/tmp/b.csv")
        assert len(backend.artifact_calls) == 2


# ===========================================================================
# E. Provenance correctness
# ===========================================================================


class TestProvenanceAttachment:
    """
    fold_id and config_hash must appear in the provenance dict of every
    backend call.  Provenance must be injected by the tracker, not by callers.
    """

    def _build_tracker_and_backend(
        self,
        fold_id: str = "fold-provenance-test",
        config_hash: str = _VALID_CONFIG_HASH,
    ) -> tuple[AimMLflowTracker, CapturingBackend]:
        backend = _make_backend()
        tracker = _make_tracker(
            backend=backend, fold_id=fold_id, config_hash=config_hash
        )
        return tracker, backend

    def test_metrics_provenance_contains_fold_id(self) -> None:
        tracker, backend = self._build_tracker_and_backend(fold_id="fold-001")
        tracker.log_metrics({"loss": 0.1})
        assert backend.metrics_calls[0]["provenance"]["fold_id"] == "fold-001"

    def test_metrics_provenance_contains_config_hash(self) -> None:
        h = "c" * 64
        tracker, backend = self._build_tracker_and_backend(config_hash=h)
        tracker.log_metrics({"loss": 0.1})
        assert backend.metrics_calls[0]["provenance"]["config_hash"] == h

    def test_params_provenance_contains_fold_id(self) -> None:
        tracker, backend = self._build_tracker_and_backend(fold_id="fold-002")
        tracker.log_params({"lr": 0.05})
        assert backend.params_calls[0]["provenance"]["fold_id"] == "fold-002"

    def test_params_provenance_contains_config_hash(self) -> None:
        h = "d" * 64
        tracker, backend = self._build_tracker_and_backend(config_hash=h)
        tracker.log_params({"lr": 0.05})
        assert backend.params_calls[0]["provenance"]["config_hash"] == h

    def test_artifact_provenance_contains_fold_id(self) -> None:
        tracker, backend = self._build_tracker_and_backend(fold_id="fold-003")
        tracker.log_artifact("/tmp/out.csv")
        assert backend.artifact_calls[0]["provenance"]["fold_id"] == "fold-003"

    def test_artifact_provenance_contains_config_hash(self) -> None:
        h = "e" * 64
        tracker, backend = self._build_tracker_and_backend(config_hash=h)
        tracker.log_artifact("/tmp/out.csv")
        assert backend.artifact_calls[0]["provenance"]["config_hash"] == h

    def test_provenance_is_independent_copy_per_call(self) -> None:
        """
        Each call to the backend must receive a fresh provenance dict.
        Mutating one call's provenance must not affect others.
        """
        backend = _make_backend()
        tracker = _make_tracker(backend=backend)
        tracker.log_metrics({"a": 1.0})
        tracker.log_metrics({"b": 2.0})
        # Mutate first call's captured provenance
        backend.metrics_calls[0]["provenance"]["fold_id"] = "mutated"
        # Second call's provenance must be unaffected
        assert backend.metrics_calls[1]["provenance"]["fold_id"] == _VALID_FOLD_ID

    def test_provenance_unchanged_across_different_log_types(self) -> None:
        """fold_id and config_hash must be identical across metrics/params/artifact calls."""
        fold_id = "fold-consistency"
        h = "f" * 64
        backend = _make_backend()
        tracker = _make_tracker(backend=backend, fold_id=fold_id, config_hash=h)
        tracker.log_metrics({"loss": 0.1})
        tracker.log_params({"lr": 0.05})
        tracker.log_artifact("/tmp/out.csv")
        for call in backend.metrics_calls + backend.params_calls + backend.artifact_calls:
            assert call["provenance"]["fold_id"] == fold_id
            assert call["provenance"]["config_hash"] == h

    def test_provenance_uses_construction_time_identity(self) -> None:
        """
        Provenance must reflect the fold_id and config_hash locked at
        construction, not values supplied at call time.
        """
        h1 = "1" * 64
        h2 = "2" * 64
        backend = _make_backend()
        tracker_a = _make_tracker(backend=backend, fold_id="fold-a", config_hash=h1)
        tracker_b = _make_tracker(backend=backend, fold_id="fold-b", config_hash=h2)

        tracker_a.log_metrics({"x": 1.0})
        tracker_b.log_metrics({"x": 2.0})

        assert backend.metrics_calls[0]["provenance"]["fold_id"] == "fold-a"
        assert backend.metrics_calls[0]["provenance"]["config_hash"] == h1
        assert backend.metrics_calls[1]["provenance"]["fold_id"] == "fold-b"
        assert backend.metrics_calls[1]["provenance"]["config_hash"] == h2


# ===========================================================================
# F. No read interface
# ===========================================================================


class TestNoReadInterface:
    """
    AimMLflowTracker must not expose any method that allows fold code to
    retrieve prior results, inspect previous runs, or access experiment state.

    These checks guard against the No Hidden Feedback Law violation.
    """

    _FORBIDDEN_METHODS: tuple[str, ...] = (
        "read",
        "query",
        "list_runs",
        "get_best_run",
        "get_metrics",
        "fetch_artifact",
        "search_runs",
        "load_artifact",
        "get_run",
        "list_artifacts",
        "get_artifact",
        "download_artifact",
    )

    def test_no_forbidden_method_exists(self) -> None:
        tracker = _make_tracker()
        violations = [
            m for m in self._FORBIDDEN_METHODS if hasattr(tracker, m)
        ]
        assert not violations, (
            f"AimMLflowTracker exposes forbidden retrieval methods: {violations}"
        )

    def test_tracker_has_exactly_the_approved_public_methods(self) -> None:
        """
        Enumerate all public methods on the tracker and assert they are only
        the three approved write methods plus the two identity properties.
        """
        tracker = _make_tracker()
        approved = {"log_metrics", "log_params", "log_artifact", "fold_id", "config_hash"}
        public_attrs = {
            name
            for name in dir(tracker)
            if not name.startswith("_")
        }
        unexpected = public_attrs - approved
        assert not unexpected, (
            f"AimMLflowTracker exposes unexpected public attributes: {sorted(unexpected)}"
        )


# ===========================================================================
# G. No architecture creep
# ===========================================================================


class TestNoArchitectureCreep:
    """
    aim_mlflow_tracker.py must not import ML, calibration, routing, tracking
    platform, or manifest libraries.  Checked via AST analysis of the source.
    """

    _MODULE_PATH: pathlib.Path = (
        pathlib.Path(__file__).parent.parent
        / "ncaa_pipeline"
        / "tracking"
        / "aim_mlflow_tracker.py"
    )

    _FORBIDDEN_ROOTS: frozenset[str] = frozenset(
        {
            "lightgbm",
            "sklearn",
            "mlflow",
            "aim",
            "torch",
            "tensorflow",
            "xgboost",
            "catboost",
            "optuna",
            "wandb",
            "boto3",
            "requests",
            "httpx",
            "numpy",
            "pandas",
        }
    )

    def _collect_imported_roots(self) -> set[str]:
        source = self._MODULE_PATH.read_text(encoding="utf-8")
        tree = ast.parse(source)
        roots: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    roots.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    roots.add(node.module.split(".")[0])
        return roots

    def test_no_forbidden_imports(self) -> None:
        imported = self._collect_imported_roots()
        violations = imported & self._FORBIDDEN_ROOTS
        assert not violations, (
            f"aim_mlflow_tracker.py imports forbidden libraries: {sorted(violations)}"
        )

    def test_only_stdlib_and_project_imports(self) -> None:
        """Only stdlib and ncaa_pipeline imports are allowed."""
        imported = self._collect_imported_roots()
        allowed_roots = {
            "typing",
            "__future__",
            "ncaa_pipeline",
        }
        unexpected = imported - allowed_roots
        assert not unexpected, (
            f"aim_mlflow_tracker.py has unexpected imports: {sorted(unexpected)}"
        )
