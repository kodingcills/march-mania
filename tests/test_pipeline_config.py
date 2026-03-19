"""
test_pipeline_config.py
=======================
Tests for PipelineConfig — the canonical frozen pipeline identity object.

Proves:
  A. Immutability — frozen at the top level and every nested config group
  B. Hash determinism — stable across calls and identical-config instances
  C. Hash sensitivity — any meaningful field change changes the hash
  D. Canonical serialization — stable, ordered, JSON-serializable output
  E. Hash format — 64-character lowercase SHA256 hex string
  F. Validation — illegal config values are rejected at construction
  G. No architecture creep — pipeline_config.py has no forbidden imports

All tests are:
  - deterministic
  - isolated (no shared mutable state)
  - self-contained (no external files, no network calls)
  - free of future-phase dependencies
"""

from __future__ import annotations

import ast
import dataclasses
import pathlib

import pytest

from ncaa_pipeline.context.pipeline_config import (
    CalibrationConfig,
    EvaluationConfig,
    ExperimentalFlags,
    LightGBMConfig,
    MasseyConfig,
    PipelineConfig,
    RouterConfig,
    TabPFNConfig,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _default() -> PipelineConfig:
    """Return a PipelineConfig with all defaults."""
    return PipelineConfig()


def _with_experimental(**kwargs) -> PipelineConfig:
    return PipelineConfig(experimental=ExperimentalFlags(**kwargs))


def _with_lgbm(**kwargs) -> PipelineConfig:
    return PipelineConfig(lightgbm=LightGBMConfig(**kwargs))


def _with_calibration(**kwargs) -> PipelineConfig:
    return PipelineConfig(calibration=CalibrationConfig(**kwargs))


def _with_router(**kwargs) -> PipelineConfig:
    return PipelineConfig(router=RouterConfig(**kwargs))


def _with_evaluation(**kwargs) -> PipelineConfig:
    return PipelineConfig(evaluation=EvaluationConfig(**kwargs))


def _with_massey(**kwargs) -> PipelineConfig:
    return PipelineConfig(massey=MasseyConfig(**kwargs))


# ===========================================================================
# A. Immutability
# ===========================================================================


class TestImmutability:
    """PipelineConfig and every nested config must be frozen after construction."""

    def test_top_level_random_seed_frozen(self) -> None:
        cfg = _default()
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            cfg.random_seed = 99  # type: ignore[misc]

    def test_top_level_experimental_field_frozen(self) -> None:
        cfg = _default()
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            cfg.experimental = ExperimentalFlags(enable_tabpfn=True)  # type: ignore[misc]

    def test_top_level_lightgbm_field_frozen(self) -> None:
        cfg = _default()
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            cfg.lightgbm = LightGBMConfig(num_leaves=64)  # type: ignore[misc]

    def test_nested_experimental_flags_frozen(self) -> None:
        cfg = _default()
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            cfg.experimental.enable_tabpfn = True  # type: ignore[misc]

    def test_nested_lgbm_frozen(self) -> None:
        cfg = _default()
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            cfg.lightgbm.num_leaves = 128  # type: ignore[misc]

    def test_nested_tabpfn_frozen(self) -> None:
        cfg = _default()
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            cfg.tabpfn.max_rows = 1  # type: ignore[misc]

    def test_nested_calibration_frozen(self) -> None:
        cfg = _default()
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            cfg.calibration.lambda_reg = 9.9  # type: ignore[misc]

    def test_nested_router_frozen(self) -> None:
        cfg = _default()
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            cfg.router.tau = 99.0  # type: ignore[misc]

    def test_nested_evaluation_frozen(self) -> None:
        cfg = _default()
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            cfg.evaluation.bootstrap_n = 1  # type: ignore[misc]

    def test_nested_massey_frozen(self) -> None:
        cfg = _default()
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            cfg.massey.stale_day_threshold = 0  # type: ignore[misc]


# ===========================================================================
# B. Hash determinism
# ===========================================================================


class TestHashDeterminism:
    """config_hash() must be stable and produce identical output for equal configs."""

    def test_hash_is_string(self) -> None:
        assert isinstance(_default().config_hash(), str)

    def test_hash_is_64_characters(self) -> None:
        assert len(_default().config_hash()) == 64

    def test_hash_is_lowercase_hex(self) -> None:
        h = _default().config_hash()
        assert all(c in "0123456789abcdef" for c in h), (
            f"Hash contains non-hex characters: {h!r}"
        )

    def test_hash_stable_on_same_object(self) -> None:
        cfg = _default()
        assert cfg.config_hash() == cfg.config_hash()

    def test_hash_stable_on_repeated_calls(self) -> None:
        cfg = _default()
        hashes = [cfg.config_hash() for _ in range(5)]
        assert len(set(hashes)) == 1

    def test_identical_default_configs_hash_equally(self) -> None:
        assert PipelineConfig().config_hash() == PipelineConfig().config_hash()

    def test_identical_custom_configs_hash_equally(self) -> None:
        cfg_a = PipelineConfig(
            lightgbm=LightGBMConfig(num_leaves=64, learning_rate=0.01),
            calibration=CalibrationConfig(bounds_a=(0.5, 2.5)),
            random_seed=42,
        )
        cfg_b = PipelineConfig(
            lightgbm=LightGBMConfig(num_leaves=64, learning_rate=0.01),
            calibration=CalibrationConfig(bounds_a=(0.5, 2.5)),
            random_seed=42,
        )
        assert cfg_a.config_hash() == cfg_b.config_hash()


# ===========================================================================
# C. Hash sensitivity — every meaningful field group changes the hash
# ===========================================================================


class TestHashSensitivity:
    """Changing any meaningful parameter must change the hash."""

    # --- random seed ---

    def test_random_seed_changes_hash(self) -> None:
        base = _default().config_hash()
        changed = PipelineConfig(random_seed=1).config_hash()
        assert base != changed

    # --- experimental flags ---

    def test_enable_tabpfn_changes_hash(self) -> None:
        base = _default().config_hash()
        changed = _with_experimental(enable_tabpfn=True).config_hash()
        assert base != changed

    def test_enable_graph_features_changes_hash(self) -> None:
        base = _default().config_hash()
        changed = _with_experimental(enable_graph_features=True).config_hash()
        assert base != changed

    def test_enable_weighted_router_changes_hash(self) -> None:
        base = _default().config_hash()
        changed = _with_experimental(enable_weighted_router=True).config_hash()
        assert base != changed

    # --- LightGBM ---

    def test_lgbm_num_leaves_changes_hash(self) -> None:
        base = _default().config_hash()
        changed = _with_lgbm(num_leaves=64).config_hash()
        assert base != changed

    def test_lgbm_learning_rate_changes_hash(self) -> None:
        base = _default().config_hash()
        changed = _with_lgbm(learning_rate=0.01).config_hash()
        assert base != changed

    def test_lgbm_n_estimators_changes_hash(self) -> None:
        base = _default().config_hash()
        changed = _with_lgbm(n_estimators=1000).config_hash()
        assert base != changed

    def test_lgbm_max_delta_step_changes_hash(self) -> None:
        base = _default().config_hash()
        changed = _with_lgbm(max_delta_step=0.5).config_hash()
        assert base != changed

    def test_lgbm_deterministic_changes_hash(self) -> None:
        base = _default().config_hash()
        changed = _with_lgbm(deterministic=False).config_hash()
        assert base != changed

    def test_lgbm_num_threads_changes_hash(self) -> None:
        base = _default().config_hash()
        changed = _with_lgbm(num_threads=4).config_hash()
        assert base != changed

    # --- calibration ---

    def test_calibration_bounds_a_changes_hash(self) -> None:
        base = _default().config_hash()
        changed = _with_calibration(bounds_a=(0.5, 2.5)).config_hash()
        assert base != changed

    def test_calibration_bounds_b_changes_hash(self) -> None:
        base = _default().config_hash()
        changed = _with_calibration(bounds_b=(-1.0, 1.0)).config_hash()
        assert base != changed

    def test_calibration_lambda_reg_changes_hash(self) -> None:
        base = _default().config_hash()
        changed = _with_calibration(lambda_reg=0.5).config_hash()
        assert base != changed

    def test_calibration_n_restarts_changes_hash(self) -> None:
        base = _default().config_hash()
        changed = _with_calibration(n_restarts=10).config_hash()
        assert base != changed

    def test_calibration_c_changes_hash(self) -> None:
        base = _default().config_hash()
        changed = _with_calibration(c=5.0).config_hash()
        assert base != changed

    # --- router ---

    def test_router_tau_changes_hash(self) -> None:
        base = _default().config_hash()
        changed = _with_router(tau=2.0).config_hash()
        assert base != changed

    # --- evaluation ---

    def test_evaluation_bootstrap_n_changes_hash(self) -> None:
        base = _default().config_hash()
        changed = _with_evaluation(bootstrap_n=5000).config_hash()
        assert base != changed

    def test_evaluation_bootstrap_ci_changes_hash(self) -> None:
        base = _default().config_hash()
        changed = _with_evaluation(bootstrap_ci=0.90).config_hash()
        assert base != changed

    # --- massey ---

    def test_massey_stale_threshold_changes_hash(self) -> None:
        base = _default().config_hash()
        changed = _with_massey(stale_day_threshold=120).config_hash()
        assert base != changed

    def test_massey_allowlist_none_vs_set_changes_hash(self) -> None:
        base = _default().config_hash()
        changed = _with_massey(system_allowlist=("SAG", "BPI")).config_hash()
        assert base != changed

    def test_massey_allowlist_content_changes_hash(self) -> None:
        cfg_a = _with_massey(system_allowlist=("SAG", "BPI")).config_hash()
        cfg_b = _with_massey(system_allowlist=("SAG", "NET")).config_hash()
        assert cfg_a != cfg_b

    # --- tabpfn ---

    def test_tabpfn_max_rows_changes_hash(self) -> None:
        base = _default().config_hash()
        changed = PipelineConfig(tabpfn=TabPFNConfig(max_rows=500)).config_hash()
        assert base != changed

    def test_tabpfn_n_ensemble_changes_hash(self) -> None:
        base = _default().config_hash()
        changed = PipelineConfig(tabpfn=TabPFNConfig(n_ensemble=8)).config_hash()
        assert base != changed


# ===========================================================================
# D. Canonical serialization
# ===========================================================================


class TestCanonicalSerialization:
    """to_canonical_dict() must be stable, complete, and JSON-serializable."""

    def test_to_canonical_dict_returns_dict(self) -> None:
        result = _default().to_canonical_dict()
        assert isinstance(result, dict)

    def test_canonical_dict_contains_all_top_level_groups(self) -> None:
        d = _default().to_canonical_dict()
        expected_keys = {
            "experimental",
            "lightgbm",
            "tabpfn",
            "calibration",
            "router",
            "evaluation",
            "massey",
            "random_seed",
        }
        assert expected_keys == set(d.keys())

    def test_canonical_dict_experimental_group_present(self) -> None:
        d = _default().to_canonical_dict()
        exp = d["experimental"]
        assert isinstance(exp, dict)
        assert "enable_tabpfn" in exp
        assert "enable_graph_features" in exp
        assert "enable_weighted_router" in exp

    def test_canonical_dict_tuple_fields_are_sequences(self) -> None:
        """
        dataclasses.asdict() preserves tuple types (type(obj)(...)  internally).
        Tuples remain tuples in the canonical dict.  json.dumps encodes tuples
        and lists identically as JSON arrays, so hashing is unaffected.
        """
        d = _default().to_canonical_dict()
        cal = d["calibration"]
        assert isinstance(cal["bounds_a"], (list, tuple)), (
            "bounds_a must be a sequence in canonical dict"
        )
        assert isinstance(cal["bounds_b"], (list, tuple)), (
            "bounds_b must be a sequence in canonical dict"
        )

    def test_canonical_dict_bounds_values_correct(self) -> None:
        d = _default().to_canonical_dict()
        assert tuple(d["calibration"]["bounds_a"]) == (0.3, 3.0)
        assert tuple(d["calibration"]["bounds_b"]) == (-2.0, 2.0)

    def test_canonical_dict_none_allowlist_is_null(self) -> None:
        d = _default().to_canonical_dict()
        assert d["massey"]["system_allowlist"] is None

    def test_canonical_dict_tuple_allowlist_is_sequence(self) -> None:
        """Tuple allowlist stays as a tuple in asdict() output; json.dumps handles both."""
        cfg = _with_massey(system_allowlist=("SAG", "BPI"))
        d = cfg.to_canonical_dict()
        assert tuple(d["massey"]["system_allowlist"]) == ("SAG", "BPI")

    def test_canonical_dict_is_json_serializable(self) -> None:
        """The canonical dict must serialize to JSON without errors."""
        import json

        d = _default().to_canonical_dict()
        # Should not raise
        serialized = json.dumps(d, sort_keys=True, separators=(",", ":"))
        assert isinstance(serialized, str)
        assert len(serialized) > 0

    def test_canonical_dict_lgbm_defaults(self) -> None:
        d = _default().to_canonical_dict()
        lgbm = d["lightgbm"]
        assert lgbm["objective"] == "binary"
        assert lgbm["metric"] == "binary_logloss"
        assert lgbm["num_leaves"] == 31
        assert lgbm["deterministic"] is True
        assert lgbm["num_threads"] == 1

    def test_hash_consistent_with_canonical_dict(self) -> None:
        """config_hash() must agree with the canonical dict → JSON → SHA256 pipeline."""
        import hashlib
        import json

        cfg = _default()
        d = cfg.to_canonical_dict()
        expected = hashlib.sha256(
            json.dumps(d, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
            .encode("utf-8")
        ).hexdigest()
        assert cfg.config_hash() == expected


# ===========================================================================
# E. Hash format
# ===========================================================================


class TestHashFormat:
    """The returned hash must be a valid 64-character lowercase SHA256 hex string."""

    def test_hash_length_is_64(self) -> None:
        assert len(_default().config_hash()) == 64

    def test_hash_is_lowercase(self) -> None:
        h = _default().config_hash()
        assert h == h.lower()

    def test_hash_is_hex(self) -> None:
        h = _default().config_hash()
        int(h, 16)  # raises ValueError if not valid hex

    def test_custom_config_hash_length_is_64(self) -> None:
        cfg = PipelineConfig(
            lightgbm=LightGBMConfig(num_leaves=255),
            experimental=ExperimentalFlags(enable_tabpfn=True),
            random_seed=12345,
        )
        assert len(cfg.config_hash()) == 64


# ===========================================================================
# F. Validation
# ===========================================================================


class TestValidation:
    """Illegal config values must be rejected at construction time."""

    # --- EvaluationConfig ---

    def test_eval_bootstrap_n_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="bootstrap_n"):
            EvaluationConfig(bootstrap_n=0)

    def test_eval_bootstrap_n_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="bootstrap_n"):
            EvaluationConfig(bootstrap_n=-1)

    def test_eval_bootstrap_ci_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="bootstrap_ci"):
            EvaluationConfig(bootstrap_ci=0.0)

    def test_eval_bootstrap_ci_one_raises(self) -> None:
        with pytest.raises(ValueError, match="bootstrap_ci"):
            EvaluationConfig(bootstrap_ci=1.0)

    def test_eval_bootstrap_ci_above_one_raises(self) -> None:
        with pytest.raises(ValueError, match="bootstrap_ci"):
            EvaluationConfig(bootstrap_ci=1.5)

    # --- CalibrationConfig ---

    def test_calibration_bounds_a_reversed_raises(self) -> None:
        with pytest.raises(ValueError, match="bounds_a"):
            CalibrationConfig(bounds_a=(3.0, 0.3))

    def test_calibration_bounds_a_equal_raises(self) -> None:
        with pytest.raises(ValueError, match="bounds_a"):
            CalibrationConfig(bounds_a=(1.0, 1.0))

    def test_calibration_bounds_b_reversed_raises(self) -> None:
        with pytest.raises(ValueError, match="bounds_b"):
            CalibrationConfig(bounds_b=(2.0, -2.0))

    def test_calibration_lambda_reg_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="lambda_reg"):
            CalibrationConfig(lambda_reg=-0.1)

    def test_calibration_n_restarts_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="n_restarts"):
            CalibrationConfig(n_restarts=0)

    def test_calibration_c_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="c must"):
            CalibrationConfig(c=0.0)

    def test_calibration_c_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="c must"):
            CalibrationConfig(c=-1.0)

    # --- LightGBMConfig ---

    def test_lgbm_num_leaves_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="num_leaves"):
            LightGBMConfig(num_leaves=0)

    def test_lgbm_n_estimators_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="n_estimators"):
            LightGBMConfig(n_estimators=0)

    def test_lgbm_num_threads_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="num_threads"):
            LightGBMConfig(num_threads=0)

    def test_lgbm_learning_rate_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="learning_rate"):
            LightGBMConfig(learning_rate=0.0)

    def test_lgbm_learning_rate_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="learning_rate"):
            LightGBMConfig(learning_rate=-0.01)

    def test_lgbm_max_delta_step_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="max_delta_step"):
            LightGBMConfig(max_delta_step=0.0)

    # --- TabPFNConfig ---

    def test_tabpfn_max_rows_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="max_rows"):
            TabPFNConfig(max_rows=0)

    def test_tabpfn_n_ensemble_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="n_ensemble"):
            TabPFNConfig(n_ensemble=0)

    def test_tabpfn_invalid_strategy_raises(self) -> None:
        with pytest.raises(ValueError, match="subsample_strategy"):
            TabPFNConfig(subsample_strategy="random")

    # --- RouterConfig ---

    def test_router_tau_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="tau"):
            RouterConfig(tau=0.0)

    def test_router_tau_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="tau"):
            RouterConfig(tau=-1.0)

    # --- MasseyConfig ---

    def test_massey_negative_threshold_raises(self) -> None:
        with pytest.raises(ValueError, match="stale_day_threshold"):
            MasseyConfig(stale_day_threshold=-1)

    def test_massey_allowlist_non_tuple_raises(self) -> None:
        with pytest.raises(TypeError, match="system_allowlist"):
            MasseyConfig(system_allowlist=["SAG", "BPI"])  # type: ignore[arg-type]

    def test_massey_allowlist_non_str_element_raises(self) -> None:
        with pytest.raises(TypeError, match="system_allowlist"):
            MasseyConfig(system_allowlist=(1, 2))  # type: ignore[arg-type]

    # --- PipelineConfig ---

    def test_pipeline_config_random_seed_non_int_raises(self) -> None:
        with pytest.raises(TypeError, match="random_seed"):
            PipelineConfig(random_seed=0.0)  # type: ignore[arg-type]

    # --- Valid edge cases pass ---

    def test_calibration_lambda_reg_zero_is_valid(self) -> None:
        cfg = CalibrationConfig(lambda_reg=0.0)
        assert cfg.lambda_reg == 0.0

    def test_massey_threshold_zero_is_valid(self) -> None:
        cfg = MasseyConfig(stale_day_threshold=0)
        assert cfg.stale_day_threshold == 0

    def test_eval_bootstrap_ci_near_boundaries_valid(self) -> None:
        low = EvaluationConfig(bootstrap_ci=0.01)
        high = EvaluationConfig(bootstrap_ci=0.99)
        assert low.bootstrap_ci == 0.01
        assert high.bootstrap_ci == 0.99

    def test_massey_empty_allowlist_is_valid(self) -> None:
        cfg = MasseyConfig(system_allowlist=())
        assert cfg.system_allowlist == ()


# ===========================================================================
# G. No architecture creep
# ===========================================================================


class TestNoArchitectureCreep:
    """
    pipeline_config.py must not import ML, tracking, or future-phase libraries.

    This test parses the module source via the AST to enumerate every import
    statement.  It does not rely on sys.modules (which can be polluted by
    other tests) and does not execute the module under inspection.
    """

    _MODULE_PATH: pathlib.Path = (
        pathlib.Path(__file__).parent.parent
        / "ncaa_pipeline"
        / "context"
        / "pipeline_config.py"
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
            f"pipeline_config.py imports forbidden libraries: {sorted(violations)}"
        )

    def test_only_stdlib_and_project_imports(self) -> None:
        """Only stdlib and ncaa_pipeline imports are allowed."""
        imported = self._collect_imported_roots()
        allowed_roots = {
            "dataclasses",
            "hashlib",
            "json",
            "typing",
            "__future__",
            "ncaa_pipeline",
        }
        unexpected = imported - allowed_roots
        assert not unexpected, (
            f"pipeline_config.py has unexpected imports: {sorted(unexpected)}"
        )
