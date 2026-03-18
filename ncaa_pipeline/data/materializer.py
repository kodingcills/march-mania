"""
materializer.py
===============
Fold-aware dataset materialization.

Coordinates RollingFeatureStore, FeatureAssembler, and (optionally)
MasseyOrdinalExtractor to produce typed, zone-separated dataset contracts.

Laws enforced:
1. Temporal Zoning Law  — Zone A/B/C outputs are physically distinct types.
2. Type Firewall Law    — EvalDataset has no .y; EvalLabels is a separate call.
3. Zone C Permutation Law — All C(K,2) pairs for the eval season's seeded teams.
4. Day 133 Cutoff Law   — Delegated to RollingFeatureStore (not re-implemented).
5. Provenance Law       — fold_id stamped on every output artifact.
6. No Architecture Creep Law — No model/calibration/routing/tracking imports.
7. Determinism Law      — Rows sorted by matchup_id ascending; identical inputs → identical outputs.
"""

from __future__ import annotations

import itertools
from typing import Callable

import numpy as np
import pandas as pd

from ncaa_pipeline.context.fold_context import FoldContext
from ncaa_pipeline.data.datasets import CalDataset, EvalDataset, EvalLabels, TrainDataset
from ncaa_pipeline.features.assembler import FeatureAssembler
from ncaa_pipeline.features.massey_extractor import MasseyOrdinalExtractor
from ncaa_pipeline.features.rolling_store import RollingFeatureStore


class DatasetMaterializer:
    """
    Stateless coordinator: raw DataFrames + FoldContext → typed dataset contracts.

    All feature computation is delegated to RollingFeatureStore and
    FeatureAssembler.  This class adds only:
      - Zone boundary enforcement
      - All-pairs Zone C generation (C(K,2) where K = seeded teams)
      - Fold provenance stamping
      - Deterministic row ordering (by matchup_id ascending)
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def materialize_fold(
        self,
        regular_season_df: pd.DataFrame,
        seeds_df: pd.DataFrame,
        labeled_games_df: pd.DataFrame,
        context: FoldContext,
        massey_df: pd.DataFrame | None = None,
        massey_systems: list[str] | None = None,
    ) -> tuple[TrainDataset, CalDataset, EvalDataset]:
        """
        Materialize Zone A (train), Zone B (cal), and Zone C (eval) datasets.

        EvalLabels is NEVER included in this return value.
        Obtain EvalLabels via ``materialize_eval_labels()``.

        Parameters
        ----------
        regular_season_df :
            Raw regular-season detailed results for feature computation.
        seeds_df :
            Tournament seed table (all seasons).
        labeled_games_df :
            Game outcomes used for Zone A and B labels.  The caller controls
            what game types to include (tournament results, regular-season
            games, or both) — the materializer imposes no restriction.
        context :
            Active fold context.
        massey_df :
            Massey ordinals table.  If None, Massey features are omitted.
        massey_systems :
            Which ranking systems to include.  If None or massey_df is None,
            Massey features are omitted entirely.

        Returns
        -------
        tuple[TrainDataset, CalDataset, EvalDataset]
            EvalLabels is NOT in this tuple.
        """
        store = self._build_feature_store(regular_season_df, context)
        assembler = FeatureAssembler(context, store)

        massey_extractor: MasseyOrdinalExtractor | None = None
        effective_massey_systems: list[str] | None = None
        if massey_df is not None and massey_systems:
            massey_extractor = MasseyOrdinalExtractor(massey_df)
            effective_massey_systems = massey_systems

        include_seeds = seeds_df is not None and len(seeds_df) > 0
        feature_names = self._build_feature_names(
            store,
            include_seeds=include_seeds,
            massey_systems=effective_massey_systems,
        )

        # Seed lookup cache: {season: {team_id: seed_num}}
        _seed_cache: dict[int, dict[int, int]] = {}

        def get_seed_lookup(season: int) -> dict[int, int]:
            if season not in _seed_cache:
                _seed_cache[season] = self._build_seed_lookup(seeds_df, season)
            return _seed_cache[season]

        # Zone A — train
        train_X, train_y, train_ids = self._build_zone_arrays(
            labeled_games_df=labeled_games_df,
            assembler=assembler,
            massey_extractor=massey_extractor,
            massey_systems=effective_massey_systems,
            get_seed_lookup=get_seed_lookup,
            feature_names=feature_names,
            seasons=set(context.train_seasons),
            context=context,
        )
        train_ds = TrainDataset(
            X=train_X,
            y=train_y,
            matchup_ids=train_ids,
            feature_names=feature_names,
            fold_id=context.fold_id,
        )

        # Zone B — cal
        cal_X, cal_y, cal_ids = self._build_zone_arrays(
            labeled_games_df=labeled_games_df,
            assembler=assembler,
            massey_extractor=massey_extractor,
            massey_systems=effective_massey_systems,
            get_seed_lookup=get_seed_lookup,
            feature_names=feature_names,
            seasons={context.cal_season},
            context=context,
        )
        cal_ds = CalDataset(
            X=cal_X,
            y=cal_y,
            matchup_ids=cal_ids,
            feature_names=feature_names,
            fold_id=context.fold_id,
        )

        # Zone C — eval (all C(K,2) pairs from eval season seeds)
        eval_seed_lookup = get_seed_lookup(context.eval_season)
        eval_X, eval_ids = self._build_eval_arrays(
            seeds_df=seeds_df,
            eval_season=context.eval_season,
            assembler=assembler,
            massey_extractor=massey_extractor,
            massey_systems=effective_massey_systems,
            seed_lookup=eval_seed_lookup,
            feature_names=feature_names,
            context=context,
        )
        eval_ds = EvalDataset(
            X=eval_X,
            matchup_ids=eval_ids,
            feature_names=feature_names,
            fold_id=context.fold_id,
        )

        return train_ds, cal_ds, eval_ds

    def materialize_eval_labels(
        self,
        labeled_games_df: pd.DataFrame,
        context: FoldContext,
    ) -> EvalLabels:
        """
        Materialize Zone C labels from realized game outcomes.

        This is the ONLY method that returns EvalLabels.

        Parameters
        ----------
        labeled_games_df :
            Game outcomes (must include Season, DayNum, WTeamID, LTeamID).
        context :
            Active fold context.

        Returns
        -------
        EvalLabels
        """
        eval_season = context.eval_season
        mask = labeled_games_df["Season"] == eval_season
        season_games = labeled_games_df.loc[mask]

        rows: list[tuple[str, int]] = []
        for _, row in season_games.iterrows():
            w_id = int(row["WTeamID"])
            l_id = int(row["LTeamID"])
            team1_id = min(w_id, l_id)
            team2_id = max(w_id, l_id)
            matchup_id = self._make_matchup_id(eval_season, team1_id, team2_id)
            label = self._derive_label(w_id, l_id)
            rows.append((matchup_id, label))

        if rows:
            rows_sorted = sorted(rows, key=lambda r: r[0])
            matchup_ids = np.array([r[0] for r in rows_sorted], dtype=object)
            y = np.array([r[1] for r in rows_sorted], dtype=np.int64)
        else:
            matchup_ids = np.empty(0, dtype=object)
            y = np.empty(0, dtype=np.int64)

        return EvalLabels(
            y=y,
            matchup_ids=matchup_ids,
            season=eval_season,
            fold_id=context.fold_id,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_feature_store(
        self,
        regular_season_df: pd.DataFrame,
        context: FoldContext,
    ) -> RollingFeatureStore:
        store = RollingFeatureStore(context)
        store.materialize(regular_season_df)
        store.freeze()
        return store

    def _build_seed_lookup(
        self,
        seeds_df: pd.DataFrame,
        season: int,
    ) -> dict[int, int]:
        """Return {team_id: seed_num} for a given season."""
        if seeds_df is None or len(seeds_df) == 0:
            return {}
        season_mask = seeds_df["Season"] == season
        result: dict[int, int] = {}
        for _, row in seeds_df.loc[season_mask].iterrows():
            team_id = int(row["TeamID"])
            seed_num = self._parse_seed_num(str(row["Seed"]))
            result[team_id] = seed_num
        return result

    def _build_feature_names(
        self,
        store: RollingFeatureStore,
        include_seeds: bool,
        massey_systems: list[str] | None,
    ) -> tuple[str, ...]:
        """Build canonical ordered feature name tuple.

        Order: for each base feature name, emit team1_, team2_, diff_, sum_.
        Then seed_diff (if include_seeds).
        Then massey_{sys}_team1/team2/diff for each system (if provided).
        """
        names: list[str] = []
        for base_name in store.FEATURE_NAMES:
            names.append(f"team1_{base_name}")
            names.append(f"team2_{base_name}")
            names.append(f"diff_{base_name}")
            names.append(f"sum_{base_name}")
        if include_seeds:
            names.append("seed_diff")
        if massey_systems:
            for sys in massey_systems:
                names.append(f"massey_{sys}_team1")
                names.append(f"massey_{sys}_team2")
                names.append(f"massey_{sys}_diff")
        return tuple(names)

    def _build_zone_arrays(
        self,
        labeled_games_df: pd.DataFrame,
        assembler: FeatureAssembler,
        massey_extractor: MasseyOrdinalExtractor | None,
        massey_systems: list[str] | None,
        get_seed_lookup: Callable[[int], dict[int, int]],
        feature_names: tuple[str, ...],
        seasons: set[int],
        context: FoldContext,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build (X, y, matchup_ids) for Zone A or Zone B."""
        mask = labeled_games_df["Season"].isin(seasons)
        filtered = labeled_games_df.loc[mask]

        n_features = len(feature_names)
        rows: list[tuple[str, np.ndarray, int]] = []

        for _, row in filtered.iterrows():
            season = int(row["Season"])
            w_id = int(row["WTeamID"])
            l_id = int(row["LTeamID"])
            team1_id = min(w_id, l_id)
            team2_id = max(w_id, l_id)

            seed_lookup = get_seed_lookup(season)
            seed_a = seed_lookup.get(w_id)
            seed_b = seed_lookup.get(l_id)

            assembled = assembler.assemble(w_id, l_id, season, seed_a, seed_b)
            feature_vec = self._extract_feature_vector(
                features_dict=assembled.features,
                feature_names=feature_names,
                n_features=n_features,
                massey_extractor=massey_extractor,
                massey_systems=massey_systems,
                season=season,
                team1_id=assembled.team1_id,
                team2_id=assembled.team2_id,
                context=context,
            )
            matchup_id = self._make_matchup_id(season, team1_id, team2_id)
            label = self._derive_label(w_id, l_id)
            rows.append((matchup_id, feature_vec, label))

        if rows:
            rows_sorted = sorted(rows, key=lambda r: r[0])
            matchup_ids = np.array([r[0] for r in rows_sorted], dtype=object)
            X = np.vstack([r[1] for r in rows_sorted]).astype(np.float64)
            y = np.array([r[2] for r in rows_sorted], dtype=np.int64)
        else:
            matchup_ids = np.empty(0, dtype=object)
            X = np.empty((0, n_features), dtype=np.float64)
            y = np.empty(0, dtype=np.int64)

        return X, y, matchup_ids

    def _build_eval_arrays(
        self,
        seeds_df: pd.DataFrame,
        eval_season: int,
        assembler: FeatureAssembler,
        massey_extractor: MasseyOrdinalExtractor | None,
        massey_systems: list[str] | None,
        seed_lookup: dict[int, int],
        feature_names: tuple[str, ...],
        context: FoldContext,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build (X, matchup_ids) for Zone C — all C(K,2) pairs."""
        season_mask = seeds_df["Season"] == eval_season
        team_ids = sorted(seeds_df.loc[season_mask]["TeamID"].tolist())
        pairs = list(itertools.combinations(team_ids, 2))  # sorted → (lo, hi)

        n_features = len(feature_names)
        rows: list[tuple[str, np.ndarray]] = []

        for team1_id, team2_id in pairs:
            seed1 = seed_lookup.get(team1_id)
            seed2 = seed_lookup.get(team2_id)
            assembled = assembler.assemble(team1_id, team2_id, eval_season, seed1, seed2)
            feature_vec = self._extract_feature_vector(
                features_dict=assembled.features,
                feature_names=feature_names,
                n_features=n_features,
                massey_extractor=massey_extractor,
                massey_systems=massey_systems,
                season=eval_season,
                team1_id=assembled.team1_id,
                team2_id=assembled.team2_id,
                context=context,
            )
            matchup_id = self._make_matchup_id(eval_season, team1_id, team2_id)
            rows.append((matchup_id, feature_vec))

        if rows:
            matchup_ids = np.array([r[0] for r in rows], dtype=object)
            X = np.vstack([r[1] for r in rows]).astype(np.float64)
        else:
            matchup_ids = np.empty(0, dtype=object)
            X = np.empty((0, n_features), dtype=np.float64)

        return X, matchup_ids

    def _extract_feature_vector(
        self,
        features_dict: dict[str, float],
        feature_names: tuple[str, ...],
        n_features: int,
        massey_extractor: MasseyOrdinalExtractor | None,
        massey_systems: list[str] | None,
        season: int,
        team1_id: int,
        team2_id: int,
        context: FoldContext,
    ) -> np.ndarray:
        """Extract ordered feature vector aligned to feature_names."""
        massey_vals: dict[str, float] = {}
        if massey_extractor is not None and massey_systems:
            for sys in massey_systems:
                snap1 = massey_extractor.safe_snapshot(
                    season, team1_id, context.day_cutoff, sys
                )
                snap2 = massey_extractor.safe_snapshot(
                    season, team2_id, context.day_cutoff, sys
                )
                massey_vals[f"massey_{sys}_team1"] = snap1.ordinal_rank
                massey_vals[f"massey_{sys}_team2"] = snap2.ordinal_rank
                if snap1.is_available and snap2.is_available:
                    massey_vals[f"massey_{sys}_diff"] = (
                        snap1.ordinal_rank - snap2.ordinal_rank
                    )
                else:
                    massey_vals[f"massey_{sys}_diff"] = float("nan")

        vec = np.empty(n_features, dtype=np.float64)
        for i, name in enumerate(feature_names):
            if name in massey_vals:
                vec[i] = massey_vals[name]
            else:
                vec[i] = float(features_dict.get(name, float("nan")))
        return vec

    @staticmethod
    def _parse_seed_num(seed_str: str) -> int:
        """Parse 'W01' → 1, 'X14' → 14."""
        return int(seed_str[1:3])

    @staticmethod
    def _make_matchup_id(season: int, team1_id: int, team2_id: int) -> str:
        return f"{season}_{team1_id}_{team2_id}"

    @staticmethod
    def _derive_label(w_team_id: int, l_team_id: int) -> int:
        """1 if the winning team has the smaller team_id (canonical team1), else 0."""
        return 1 if w_team_id == min(w_team_id, l_team_id) else 0
