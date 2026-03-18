"""
assembler.py
============
Matchup-level feature assembly with canonical ordering and symmetry guarantees.

Canonical Ordering Law (enforced here):
  ``team1_id < team2_id`` — always.  Any ``(A, B)`` input is reordered so
  that ``team1_id = min(A, B)`` before any feature is computed.

Symmetry guarantees:
  ``sum_*`` features: invariant under input team-order permutation.
      ``sum_f = f(team1) + f(team2)``; same value regardless of which team
      was passed as team_a vs team_b.
  ``diff_*`` features: deterministic relative to canonical order.
      ``diff_f = f(team1) - f(team2)`` where team1 is always the smaller ID.

No labels.  No dataset contract objects (TrainDataset, CalDataset, etc.).
No model logic.  No calibration logic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from ncaa_pipeline.context.fold_context import FoldContext
from ncaa_pipeline.features.rolling_store import RollingFeatureStore


# ---------------------------------------------------------------------------
# Output type
# ---------------------------------------------------------------------------

@dataclass
class AssembledFeatures:
    """
    Flat feature representation for a single canonical matchup.

    ``team1_id`` is always <= ``team2_id`` (enforced by ``FeatureAssembler``).

    Attributes
    ----------
    team1_id : int
        Canonical team 1 (smaller TeamID).
    team2_id : int
        Canonical team 2 (larger TeamID).
    season : int
    fold_id : str
        Provenance: ``fold_id`` from the active ``FoldContext``.
    features : dict[str, float]
        All numeric features.  Key naming convention:
          ``team1_<name>``  — raw feature for team1
          ``team2_<name>``  — raw feature for team2
          ``diff_<name>``   — team1 minus team2 (canonical)
          ``sum_<name>``    — team1 plus team2 (order-invariant)
        Optional key:
          ``seed_diff``     — seed1 minus seed2 (if seeds were provided)
    """

    team1_id: int
    team2_id: int
    season: int
    fold_id: str
    features: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Assembler
# ---------------------------------------------------------------------------

class FeatureAssembler:
    """
    Assembles per-matchup feature vectors from per-team season features.

    Enforces canonical ordering (``team1_id < team2_id``) and produces
    symmetry-preserving ``diff_*`` and ``sum_*`` transforms.

    Parameters
    ----------
    context : FoldContext
        Active fold context; ``fold_id`` is stamped into each
        ``AssembledFeatures`` for provenance.
    feature_store : RollingFeatureStore
        Materialized feature store.  Must have been materialized before
        ``assemble()`` is called.
    """

    def __init__(
        self,
        context: FoldContext,
        feature_store: RollingFeatureStore,
    ) -> None:
        self._context = context
        self._feature_store = feature_store

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def assemble(
        self,
        team_a_id: int,
        team_b_id: int,
        season: int,
        seed_a: int | None = None,
        seed_b: int | None = None,
    ) -> AssembledFeatures:
        """
        Assemble a canonical feature vector for a single matchup.

        Input team order is normalized: the team with the smaller ID becomes
        ``team1``.  All ``diff_*`` features are ``team1 - team2`` relative to
        this canonical order.

        Parameters
        ----------
        team_a_id : int
            First team (order does not matter; will be canonicalized).
        team_b_id : int
            Second team.
        season : int
            Season year.
        seed_a : int or None
            Tournament seed for ``team_a``, if available.
        seed_b : int or None
            Tournament seed for ``team_b``, if available.

        Returns
        -------
        AssembledFeatures
            Canonical (``team1_id < team2_id``) matchup representation with
            ``fold_id`` stamped for provenance.

        Raises
        ------
        ValueError
            If ``team_a_id == team_b_id``.
        RuntimeError
            If the feature store has not been materialized.
        """
        if team_a_id == team_b_id:
            raise ValueError(
                f"team_a_id and team_b_id must be distinct; both are {team_a_id}."
            )

        # --- Canonical ordering: team1_id < team2_id --------------------
        if team_a_id < team_b_id:
            team1_id, team2_id = team_a_id, team_b_id
            seed1, seed2 = seed_a, seed_b
        else:
            team1_id, team2_id = team_b_id, team_a_id
            seed1, seed2 = seed_b, seed_a

        # --- Per-team feature retrieval ---------------------------------
        feats1 = self._feature_store.get_team_features(season, team1_id)
        feats2 = self._feature_store.get_team_features(season, team2_id)

        # --- Build flat feature dict ------------------------------------
        feature_names = self._feature_store.FEATURE_NAMES
        features: dict[str, float] = {}

        for name in feature_names:
            v1 = feats1.get(name, float("nan"))
            v2 = feats2.get(name, float("nan"))

            features[f"team1_{name}"] = v1
            features[f"team2_{name}"] = v2

            # diff: team1 - team2 (canonical; deterministic under swap)
            features[f"diff_{name}"] = (
                float("nan") if (math.isnan(v1) or math.isnan(v2))
                else v1 - v2
            )

            # sum: invariant under team-order permutation
            features[f"sum_{name}"] = (
                float("nan") if (math.isnan(v1) or math.isnan(v2))
                else v1 + v2
            )

        # --- Optional seed differential ---------------------------------
        if seed1 is not None and seed2 is not None:
            features["seed_diff"] = float(seed1 - seed2)

        return AssembledFeatures(
            team1_id=team1_id,
            team2_id=team2_id,
            season=season,
            fold_id=self._context.fold_id,
            features=features,
        )
