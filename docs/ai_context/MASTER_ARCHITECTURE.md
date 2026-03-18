# NCAA March Madness 2026 — Master Architecture Specification

**Document Status:** DEFINITIVE — supersedes all prior drafts  
**Version:** 2.0  
**Date:** 2026-03-18  
**Inputs Synthesized:**
1. Original Architecture Specification (2026-03-16)
2. Claude Adversarial Review (2026-03-17)
3. Gemini Second-Pass Stress Test (2026-03-17)
4. ChatGPT Temporal Integrity Red-Team Audit (2026-03-17)

---

## 1. EXECUTIVE ARCHITECTURE VERDICT

### Final Default Architecture

The production pipeline is a **two-model calibrated ensemble** predicting all-pairs tournament matchup probabilities, optimizing the Kaggle Logistic Brier Score (c=7.0), validated via strict expanding-window temporal cross-validation with three disjoint zones.

**Default path:** LightGBM (binary log-loss) → per-model Platt calibration → simple average ensemble.

**Not default:** TabPFN (experimental until subsampling is validated), RSC-weighted routing (experimental until demonstrated lift), graph scalar features (experimental until validated on ≥3 folds with statistical acceptance criterion).

### Major Revisions From Original Design

| # | Original Design | Revision | Rationale |
|---|----------------|----------|-----------|
| R1 | LightGBM with Huber loss | LightGBM with `objective='binary'`, `max_delta_step` for gradient clipping | Huber produces unbounded margins incompatible with calibration pipeline. All three reviewers endorsed this fix. |
| R2 | Single calibrator class | Per-model calibrator instances, each 2-parameter Platt family | Different miscalibration profiles require separate correction. Unanimous agreement. |
| R3 | CFA Router as default | Simple average as default; performance-weighted router behind experimental flag | Router produces near-equal weights (±0.03) with τ=1.0 and M=2. Three reviewers independently identified this as overengineering. The "convex combination always helps" claim is mathematically false for out-of-sample evaluation. |
| R4 | RSC divergence as diversity metric | Mean absolute prediction difference as diagnostic; RSC removed from default path | RSC conflates distributional shape with per-sample disagreement. Both Claude and Gemini demonstrated concrete failure scenarios. |
| R5 | In-sample calibrated predictions feed router | LOO-CV calibrated predictions feed router (when router is enabled) | Calibrator-router in-sample feedback loop creates differential optimism bias of ~0.063 nats, sufficient to flip majority weight. All three reviewers flagged this. |
| R6 | `NelderMeadCalibrator.fit(raw_probs: np.ndarray, labels: np.ndarray)` | `NelderMeadCalibrator.fit(raw_probs: np.ndarray, cal_data: CalDataset)` | Typed signatures are the firewall. Raw array APIs contradict the anti-leakage claims. Claude identified the EvalLabels phantom type; ChatGPT confirmed the API contradiction. |
| R7 | Calibration bounds a ∈ [0.1, 5.0] | a ∈ [0.3, 3.0] with L2 regularization toward identity | a=5.0 produces catastrophic sharpening on 63-sample calibration sets. Claude quantified the failure mode; Gemini endorsed regularization. |
| R8 | Nelder-Mead with single start | Multi-start Nelder-Mead (5 restarts) OR L-BFGS-B (see §8 dispute) | Single-start risks local minima. Multi-start is trivially cheap. L-BFGS-B is valid for the convex core but disputed for c=7.0 clipping. |
| R9 | ExperimentRunner unspecified | Full class blueprint with aggregation, bootstrap CIs, decision criteria | Claude: "The most consequential decision in the entire pipeline is the one with the least engineering rigor around it." |
| R10 | Zone C labels lifecycle unspecified | Lazy-load at Stage 13 only; LeakageGuard enforces no eval labels in memory before Stage 13 | Both Claude and Gemini flagged this as a spec gap that must be filled regardless of empirical evidence. |

### What Remains Experimental

| Component | Status | Promotion Criterion |
|-----------|--------|-------------------|
| TabPFN v2.5 | Experimental | Subsampling strategy validated; deterministic reproduction demonstrated; performance ≥ LightGBM on ≥3 folds |
| Performance-weighted router | Experimental | LOO-CV router demonstrates >0.005 aggregate log-loss improvement over simple average across ≥5 folds, with bootstrap CI excluding zero |
| Graph scalar features (PageRank, CON, clustering) | Experimental | Statistically significant lift on ≥3 folds with pre-registered acceptance criterion; temporal safety verified per ChatGPT invariants |

### What Remains Unresolved

| Issue | Current Default | Evidence Needed |
|-------|----------------|-----------------|
| Nelder-Mead vs L-BFGS-B for calibration | Nelder-Mead (multi-start) | Proof that c=7.0 clipping introduces non-convexity at fitted parameter values, or empirical comparison on ≥5 folds |
| Calibration regularization strength λ | λ = 0.1 (L2 penalty toward identity) | Sensitivity analysis across historical folds |
| TabPFN subsampling strategy | Recency-weighted stratified (proposed) | Variance analysis: σ(log-loss) < 0.005 across 3 strategies × 3 folds |
| Softmax temperature τ for router | τ = 1.0 | If router is ever promoted: sensitivity analysis |

### Rejected — Must Not Silently Return

| Component | Why Rejected |
|-----------|-------------|
| Huber loss for LightGBM | Produces margins, not probabilities. Impedance mismatch with calibration pipeline. |
| Dense node2vec embeddings | Uninterpretable, leakage-prone, disproportionate to data regime |
| Per-sample adaptive routing | Stacking meta-learner in disguise; overfits on N=63 |
| Learned router of any kind | Data budget cannot support a third temporal fold |
| Full-season static graph features | Temporal graph leakage per ChatGPT Invariant 4 |
| Cross-season graph aggregation | Cross-season leakage per ChatGPT Invariant 5 |
| RankingDayNum ≥ 133 Massey ordinals | Post-cutoff ordinal leakage per ChatGPT Invariant 2 |

---

## 2. SYSTEM OBJECTIVE AND DESIGN PHILOSOPHY

### Objective

Minimize the Kaggle **Logistic Brier Score** with c=7.0 for every hypothetical matchup in the 2026 NCAA Men's Division I Basketball Tournament:

```
Loss = -(1/N) * Σᵢ [ yᵢ * log(clip(pᵢ)) + (1 - yᵢ) * log(1 - clip(pᵢ)) ]
where clip(p) = max(min(p, 1 - 10⁻⁷), 10⁻⁷)
```

This is standard log-loss with clipping at 10⁻⁷, bounding the maximum per-sample contribution to ~16.12 nats.

### Data Regime

This is a **small-data, high-variance, non-stationary** prediction problem:
- ~63 tournament games per season for evaluation
- ~5,000–6,000 regular-season games per season for training features
- ~30 NCAA seasons × ~5,500 games ≈ 165,000 total regular-season games
- Features are team-level aggregates; matchup-level samples are much smaller
- Year-to-year distribution shift from rule changes, transfer portal, conference realignment
- Irreducible upset rate of ~25% creates a hard Brier floor

### Anti-Leakage Philosophy

**Defense in depth, not defense by discipline.**

1. **Type-level firewalls:** `EvalDataset` has no `.y` attribute. `CalDataset` and `EvalLabels` are distinct types. Fitting methods accept typed objects, not raw arrays.
2. **Structural ordering:** Zone C predictions physically do not exist when the router fits. Zone C labels are not loaded until Stage 13.
3. **Runtime guards:** `LeakageGuard` validates temporal contracts at every stage boundary.
4. **Provenance embedding:** Every serialized artifact embeds its originating `fold_id`, season tuple, config hash, and pipeline version. Every `load()` validates provenance against the current `FoldContext`.

Per ChatGPT: "Safety by contract" is necessary but insufficient. Safety by construction (type-level + structural ordering) is the primary defense. Runtime guards and provenance are secondary.

### Simplicity-vs-Complexity Philosophy

**The default pipeline must be the simplest defensible system.** Complexity is earned through empirical evidence, not through theoretical appeal. Specifically:

- A component enters the default path only after demonstrating statistically significant improvement (bootstrap CI excluding zero) over the simpler alternative across ≥3 historical folds.
- A component enters the experimental path only after its temporal safety is verified and its interface is fully specified.
- A component is rejected if it cannot be made temporally safe, if its data-budget requirements exceed what is available, or if it introduces dependency loops.

---

## 3. CANONICAL TEMPORAL MODEL

### Zone Definitions

For a fold with training seasons {s₁, ..., sₜ}, calibration season sₜ₊₁, and evaluation season sₜ₊₂:

**Zone A (Train):** All regular-season games with DayNum ≤ 133 in seasons {s₁, ..., sₜ}. Used for: base model fitting, feature store computation for training matchups.

**Zone B (Calibrate):** Tournament games from season sₜ₊₁ (DayNum > 133). Features are computed using only DayNum ≤ 133 data from season sₜ₊₁. Labels are visible to calibrators and router. Used for: calibrator fitting, router weight computation.

**Zone C (Evaluate):** Tournament games from season sₜ₊₂ (DayNum > 133). Features are computed using only DayNum ≤ 133 data from season sₜ₊₂. Labels are visible only to MetricEngine at Stage 13. Used for: final metric computation.

### Day 133 Cutoff Law (Non-Negotiable)

**Law:** No feature for any prediction may depend on information from any game or ranking snapshot with DayNum ≥ 133 in the prediction season.

Rationale: Regular season ends at DayNum ≤ 132. Selection Sunday is DayNum 132. DayNum 133 (Selection Monday) is the latest safe information boundary. Tournament games begin at DayNum ≥ 134.

This cutoff is hardcoded (not configurable) per original spec: "the 133 is a constant, not configurable, to prevent accidental relaxation."

### Mathematical Invariants (From ChatGPT Audit)

These six invariants are architectural law. Violation of any invariant is a pipeline bug.

**Invariant 1 — Causal Feature Dependence:**
For every example g with season s and cutoff c*(g):
```
features(g) ∈ σ({RegSeason games with (Season < s) ∪ (Season = s, DayNum < c*(g))})
```

**Invariant 2 — Ordinal Time Constraint:**
For every ranking record r used for example g:
```
r.Season = g.Season AND r.RankingDayNum < c*(g)
```

**Invariant 3 — Tournament Freeze:**
For predicting any tournament matchup: c*(g) ≤ 133.

**Invariant 4 — Graph Causality:**
Graph features must be computed on A_{c*(g)}, never on A_season or any multi-season aggregated adjacency.

**Invariant 5 — Fold-Local Preprocessing:**
Any learned transform T (scaler, imputer, encoder) must be fit only on training data for that fold.

**Invariant 6 — No Winner/Loser-Coded Joins:**
Any feature computation referencing WTeamID or LTeamID must be proven purely historical and not conditioned on the outcome of the game being predicted.

### All-Pairs Evaluation Law

**Zone C matchups include all C(K, 2) possible pairs of tournament-field teams for season sₜ₊₂, not only the games that were played.** This is the Kaggle submission format. Evaluation metrics are computed on the subset that actually occurred.

This resolves Gemini Dispute 4. The model predicts probabilities for all possible matchups. The set of matchups does not encode bracket outcomes.

### Season-Level Expanding Window

Folds are defined as:
```
Fold k: train_seasons = [2003, ..., 2003+k-1], cal_season = 2003+k, eval_season = 2003+k+1
```

With data available through 2025, the maximum number of non-overlapping evaluation seasons is approximately 10 (2016–2025, excluding 2020 which had no tournament). Each fold adds one season to the training set.

---

## 4. DATA FOUNDATION AND AUDIT-CONSTRAINED RULES

### Trusted Raw Tables

| Table | Trust Level | Critical Fields | Audit Notes |
|-------|------------|----------------|-------------|
| MRegularSeasonDetailedResults | High | Season, DayNum, WTeamID, LTeamID, WScore, LScore, all box-score columns | Winner/loser-coded schema requires canonicalization before feature use (Invariant 6) |
| MNCAATourneyDetailedResults | High | Same as above | Used only for Zone B/C labels; never for features |
| MNCAATourneySeeds | High | Season, Seed, TeamID | Seed is a legitimate pre-tournament feature |
| MasseyOrdinals | Conditional | Season, RankingDayNum, SystemName, TeamID, OrdinalRank | **Must enforce RankingDayNum < 133.** Some systems are stale before Day 128. |
| MTeams | High | TeamID, TeamName | Reference only |
| MSeasons | High | Season, DayZero | Used for calendar calculations |

### Massey Ordinal Handling Rules

Per ChatGPT audit, Massey ordinals are the highest-risk leakage vector. Rules:

1. **Hard cutoff:** Only rows with `RankingDayNum < 133` are permissible for tournament prediction.
2. **Per-system time series:** Each SystemName is treated independently. Collapsing across systems before enforcing time constraints is a leakage vector.
3. **Last Available Day (LAD) logic:** For a given (Season, TeamID, SystemName), use the row with the maximum RankingDayNum that is still < 133. If no such row exists, the value is NULL/sentinel.
4. **Stale system policy:** Systems that have no update after DayNum 128 for a given season should be flagged. Whether to exclude or downweight them is a feature-engineering decision, not an architectural one, but the staleness flag must be computed and available.
5. **System presence as signal:** The set of systems that publish for a given team is itself information. However, system availability patterns should not be used as features unless they are computed within the temporal cutoff.

### Winner/Loser Schema Canonicalization

The raw data uses WTeamID/LTeamID columns. This is a label-reconstruction hazard:
- If features are joined separately for "W" and "L" sides, the model learns "W side wins" tautologically.
- All feature computation must first canonicalize to (Team1ID, Team2ID) where Team1ID < Team2ID (lower TeamID first).
- The outcome label is: "Did Team1 (lower TeamID) win?"
- Feature differences are computed as Team1_feature - Team2_feature.

### Data Limitations Elevated to Architectural Policy

1. **No post-DayNum-132 regular-season data exists** (by definition). But conference tournaments occur at DayNum ≤ 132, and their results are permissible if DayNum < 133.
2. **WNCAATourneyDetailedResults**: If the women's tournament data is used for auxiliary features, the LBlk column audit flag must be addressed (potential data quality issue in modern era).
3. **2026 seed-slot observations** from the audit are informational, not modeling blockers. Seed is a legitimate pre-tournament feature.

---

## 5. CANONICAL DATA CONTRACTS

### 5.1 FoldContext (Immutable)

```python
@dataclass(frozen=True)
class FoldContext:
    fold_id: str              # UUID, generated at creation
    train_seasons: tuple[int, ...]
    cal_season: int
    eval_season: int
    day_cutoff: int = 133     # NOT configurable
    random_seed: int          # Deterministic per fold
```

- **Created by:** ExperimentRunner (outer loop)
- **Read by:** All components
- **Invariant:** train_seasons, cal_season, eval_season are mutually disjoint
- **Serialization:** JSON, embedded in every artifact for provenance

### 5.2 TrainDataset (Immutable)

```python
@dataclass(frozen=True)
class TrainDataset:
    X: np.ndarray           # shape (n_train, n_features), writeable=False
    y: np.ndarray           # shape (n_train,), binary, writeable=False
    matchup_ids: np.ndarray # shape (n_train,), for provenance
    feature_names: tuple[str, ...]
    fold_id: str
    zone: str = "train"     # Always "train"
```

- **Created by:** DatasetMaterializer.build_train()
- **Read by:** Base model .fit() methods
- **Cannot be read by:** Calibrators, router, MetricEngine

### 5.3 CalDataset (Immutable)

```python
@dataclass(frozen=True)
class CalDataset:
    X: np.ndarray           # shape (n_cal, n_features), writeable=False
    y: np.ndarray           # shape (n_cal,), binary, writeable=False
    matchup_ids: np.ndarray
    feature_names: tuple[str, ...]
    fold_id: str
    zone: str = "cal"       # Always "cal"
```

- **Created by:** DatasetMaterializer.build_cal()
- **Read by:** Base model .predict_proba() (features only), Calibrator .fit(), Router .fit()
- **Cannot be read by:** MetricEngine (except for diagnostic; never for eval metrics)

### 5.4 EvalDataset (Immutable, NO LABELS)

```python
@dataclass(frozen=True)
class EvalDataset:
    X: np.ndarray           # shape (n_eval, n_features), writeable=False
    matchup_ids: np.ndarray
    feature_names: tuple[str, ...]
    fold_id: str
    zone: str = "eval"      # Always "eval"
    # NO .y attribute. This is the primary type-level firewall.
```

- **Created by:** DatasetMaterializer.build_eval()
- **Read by:** Base model .predict_proba() (features only), Calibrator .transform(), Router .combine()
- **Cannot be read by:** Any fitting method

### 5.5 EvalLabels (Immutable)

```python
@dataclass(frozen=True)
class EvalLabels:
    y: np.ndarray           # shape (n_eval,), binary, writeable=False
    matchup_ids: np.ndarray # Must match EvalDataset.matchup_ids
    season: int
    fold_id: str
```

- **Created by:** FoldOrchestrator at Stage 13 only (lazy-loaded)
- **Read by:** MetricEngine.evaluate() only
- **Cannot be read by:** Any model, calibrator, router, or feature component
- **Lifecycle:** Not loaded into memory until Stage 13. LeakageGuard asserts absence before Stage 13.

### 5.6 PredictionBundle (Immutable)

```python
@dataclass(frozen=True)
class PredictionBundle:
    matchup_ids: np.ndarray
    predictions: np.ndarray  # shape (n,), float64, in (0, 1)
    model_name: str
    zone: str               # "cal" or "eval"
    calibrated: bool        # False = raw, True = post-calibration
    fold_id: str
```

### 5.7 CalibrationBundle (Immutable)

```python
@dataclass(frozen=True)
class CalibrationBundle:
    params: dict[str, tuple[float, float]]  # model_name -> (a, b)
    fold_id: str
    fit_zone: str = "cal"
    diagnostics: dict       # optimization traces, pre/post loss
```

- Deep-copied from calibrators at construction. Source calibrator mutations do not propagate.

### 5.8 RunManifest (Immutable)

```python
@dataclass(frozen=True)
class RunManifest:
    fold_id: str
    context: FoldContext
    config_hash: str
    feature_config_hash: str
    schema_hash: str
    pipeline_version: str
    git_commit: str
    python_version: str
    dependency_versions: dict[str, str]  # {package: version}
    random_seed: int
    model_artifacts: dict[str, str]      # model_name -> path
    calibration_artifacts: dict[str, str]
    router_artifacts: dict[str, str]
    tabpfn_subsample_indices: np.ndarray | None  # If TabPFN used
    metric_report: dict
    timestamp: str
```

Per ChatGPT critique: the manifest alone is NOT sufficient for reproduction — it is sufficient only when combined with the tracked config, environment artifacts, and referenced model files. The document must not overclaim.

---

## 6. FEATURE SYSTEM SPECIFICATION

### Rolling Team Statistics

Features are per-team, per-season, as-of-DayNum-133 aggregates. Computed from RegularSeasonDetailedResults after canonicalization and cutoff filtering.

**Core features (all computed with DayNum < 133 filter):**
- Adjusted offensive/defensive efficiency (points per 100 possessions)
- Tempo (possessions per game)
- Effective field goal percentage (offensive and defensive)
- Turnover rate, offensive rebound rate, free throw rate (four factors)
- Win percentage overall and against tournament-quality opponents
- Scoring margin (mean and variance)
- Recent form (last 10 games rolling window, must be DayNum < 133)

**Derived features:**
- Strength of schedule: mean opponent efficiency (1st-order SOS)
- 2nd-order SOS: mean opponent's opponents' efficiency
- Consistency: standard deviation of game-level efficiency

### Massey Ordinal Features

Per system, per team, as-of-DayNum < 133:
- Rank value (lower is better)
- Rank difference between Team1 and Team2
- Missingness flag per system
- Number of systems with available rankings (coverage)
- Median rank across available systems
- Rank variance across available systems (agreement measure)

Implementation must use the `safe_massey_snapshot()` pattern from the ChatGPT audit.

### Matchup Feature Construction

For a matchup (Team1, Team2) with Team1ID < Team2ID:

```
diff_features = Team1_features - Team2_features  (signed difference)
sum_features  = Team1_features + Team2_features   (combined strength proxy)
seed_diff     = Team1_seed_numeric - Team2_seed_numeric
```

Symmetry invariant: `assemble(A, B)` with label 1 must produce identical features to `assemble(B, A)` with label 0 (after canonical ordering).

### Prohibited Features

1. Any feature computed from DayNum ≥ 133 data in the prediction season
2. Any feature derived from tournament game results
3. Any feature using winner/loser coding from the target game
4. Any cross-season aggregate that uses future seasons
5. Any global normalization fitted across all seasons (must be fold-local)
6. Dense embeddings (node2vec, GNN embeddings, etc.)

### Graph Scalar Features (Experimental Only)

**Status:** Behind `enable_graph_features` flag. Not in default path.

If enabled:
- Build directed win graph from DayNum < 133 games in the prediction season only
- Compute: PageRank, in-degree, out-degree, clustering coefficient
- CON score: computed from masked adjacency A_{133}, never A_season
- Must recompute per season and per fold (Invariant 4)
- Must not include cross-season edges (Invariant 5)

**Acceptance criterion (must be pre-registered before running):** Mean log-loss improvement > 0.003 over no-graph baseline across ≥3 folds, with bootstrap 90% CI excluding zero.

---

## 7. MODEL SYSTEM SPECIFICATION

### 7.1 LightGBMBinaryCore (Default)

**Single canonical name.** The lifecycle, blueprint, tests, and manifests must all use `LightGBMBinaryCore`. The name `LightGBMHuberCore` is retired and must not appear anywhere.

- **Objective:** `objective='binary'` (log-loss). Produces native log-odds; `.predict_proba()` applies sigmoid internally.
- **Gradient clipping:** `max_delta_step=1.0` (configurable). Achieves Huber-like robustness without destroying probability semantics.
- **Output:** 1-D array of pseudo-probabilities in (0, 1). These are σ(leaf_sum).
- **Miscalibration profile:** Tends overconfident near 0 and 1 due to boosting.
- **Training data:** Full Zone A (no row limit).
- **Determinism:** Set `seed`, `deterministic=True`, `num_threads=1` for reproducible training. Log effective hyperparameters.
- **Persistence:** `lgbm_model_{fold_id}.txt`

```python
class LightGBMBinaryCore:
    def __init__(self, params: dict, max_delta_step: float = 1.0):
        assert params.get('objective') == 'binary'
    
    def fit(self, dataset: TrainDataset) -> None: ...
    def predict_proba(self, X: np.ndarray) -> np.ndarray: ...  # returns (0, 1)
    def save(self, path: str) -> None: ...
    def load(self, path: str, context: FoldContext) -> None: ...  # validates provenance
```

### 7.2 TabPFNCore (Experimental)

**Status:** Experimental. Behind `enable_tabpfn` flag.

- **Architecture:** Transformer-based foundation model. Approximate Bayesian posterior via single forward pass.
- **Output:** 1-D array of posterior probabilities in (0, 1).
- **Miscalibration profile:** Well-calibrated in center, noisy in tails.
- **Row limit:** 10,000 rows maximum for open-source implementation. 50,000 for v2.5 with `ignore_pretraining_limits`.
- **Subsampling strategy (MUST BE SPECIFIED):**
  - Default: Recency-weighted stratified sampling. Each season contributes samples proportional to `1 / (current_season - season + 1)`, ensuring recent seasons are overrepresented.
  - Seed: `FoldContext.random_seed` determines subsample. Same seed → same subsample.
  - Logging: Actual subsample indices are persisted as `tabpfn_subsample_{fold_id}.npy`.
  - Reproducibility: `test_deterministic_fold` must verify identical predictions with identical seeds.
- **Memory:** O(N²) attention. Load-test required on target hardware before deployment.
- **Persistence:** `tabpfn_context_{fold_id}.pkl`

**Promotion criterion:** Demonstrate on ≥3 folds that TabPFN's Zone C log-loss is within 0.01 of LightGBM, with subsampling variance σ(log-loss) < 0.005 across 3 subsampling strategies.

```python
class TabPFNCore:
    def __init__(self, device: str = 'cpu', n_ensemble_configurations: int = 16,
                 max_train_rows: int = 10000, subsample_strategy: str = 'recency_weighted'):
        ...
    
    def fit(self, dataset: TrainDataset) -> None: ...  # stores training data; subsamples if needed
    def predict_proba(self, X: np.ndarray) -> np.ndarray: ...
    def get_subsample_indices(self) -> np.ndarray | None: ...
    def save(self, path: str) -> None: ...
    def load(self, path: str, context: FoldContext) -> None: ...
```

---

## 8. CALIBRATION SYSTEM SPECIFICATION

### Calibration Family

**2-parameter log-odds affine transform (Platt scaling with bounded parameters):**

```
logit(p) = log(p / (1 - p))
calibrated_logit = a * logit(p) + b
calibrated_p = σ(a * logit(p) + b)
```

Parameters: (a, b) ∈ ℝ²

### Parameter Bounds (Revised per Reviews)

```
a ∈ [0.3, 3.0]    # Revised from [0.1, 5.0] per Claude Dispute 8
b ∈ [-2.0, 2.0]   # Tightened from [-3.0, 3.0]
```

Rationale: a=5.0 transforms p=0.6 to 0.88 and p=0.7 to 0.99. On 63 calibration samples, this produces catastrophic sharpening on upsets. a=3.0 transforms p=0.6 to 0.77 and p=0.7 to 0.92, which is aggressive but bounded.

### Regularization (New, Per Reviews)

Add L2 penalty toward identity:

```
L_total(a, b) = L_logloss(a, b) + λ * [(a - 1)² + b²]
```

where λ = 0.1 (configurable, behind experiment flag).

This anchors the optimizer toward "do nothing" unless there is strong evidence for correction. On 63 samples, this is a meaningful prior.

### Objective Function

```
L_logloss(a, b) = -(1/N) * Σᵢ [ yᵢ * log(clip(pᵢ(a,b))) + (1 - yᵢ) * log(1 - clip(pᵢ(a,b))) ]

where:
  pᵢ(a, b) = σ(a * logit(raw_pᵢ) + b)
  clip(p) = max(min(p, 1 - 10⁻⁷), 10⁻⁷)
```

### Optimizer Choice (Unresolved Dispute)

**Current default:** Multi-start Nelder-Mead (5 restarts from random points within bounds, plus identity start).

**Dispute:** Gemini argues the Platt scaling objective is strictly convex and L-BFGS-B is mathematically superior. This is correct for the standard log-loss surface. However, the c=7.0 clipping at 10⁻⁷ introduces non-differentiable points where predictions hit the clip boundary. At a=3.0, predictions of 0.85+ are pushed near the boundary.

**Resolution path:** Run both optimizers on ≥5 historical folds. If they produce identical (a, b) to within 1e-3, use L-BFGS-B (faster, guaranteed convergence). If they disagree, investigate whether the disagreement is at clip boundaries.

**For now:** Nelder-Mead with multi-start is the safe default. It handles both the smooth interior and the clip-boundary non-differentiability.

### Per-Model Calibration

Each base model gets its own NelderMeadCalibrator instance with its own (a, b). They fit independently on the same Zone B data. Total parameter count: 4 across ~63 samples.

### Safety Valves

1. **Identity fallback:** If L_total(a_fitted, b_fitted) > L_total(1.0, 0.0), revert to (a=1, b=0) and log warning.
2. **Bound-hit diagnostic:** If fitted a or b is within 0.01 of a bound, log a warning that the calibrator is constrained. This suggests the raw model is pathologically miscalibrated.
3. **Monotonicity check:** Assert calibrated outputs preserve input rank ordering (guaranteed by a > 0, but verify numerically).

### Fit/Transform Lifecycle

```python
class NelderMeadCalibrator:
    def __init__(self, model_name: str, c: float = 7.0,
                 bounds_a: tuple = (0.3, 3.0), bounds_b: tuple = (-2.0, 2.0),
                 lambda_reg: float = 0.1, n_restarts: int = 5):
        ...
    
    def fit(self, raw_probs: np.ndarray, cal_data: CalDataset) -> CalibrationResult:
        """Fits on Zone B. cal_data carries zone provenance.
        raw_probs: model predictions on cal_data.X
        cal_data.y: Zone B labels
        Returns CalibrationResult with fitted params and diagnostics."""
        ...
    
    def transform(self, raw_probs: np.ndarray) -> np.ndarray:
        """Applies frozen (a, b). No labels. No fitting.
        Returns calibrated probabilities clipped to [10⁻⁷, 1 - 10⁻⁷]."""
        ...
    
    def get_params(self) -> tuple[float, float]: ...
    def save(self, path: str) -> None: ...
    def load(self, path: str, context: FoldContext) -> None: ...
```

**Critical:** `fit()` accepts `CalDataset` (typed), not raw `np.ndarray` for labels. This is the type-level firewall that prevents Zone C labels from entering calibration.

---

## 9. ENSEMBLING / ROUTING SYSTEM SPECIFICATION

### Default: SimpleAverageRouter

```
p_combined = 0.5 * p_model1 + 0.5 * p_model2
```

When only one model is enabled (Phase 1, LightGBM-only), the router is a pass-through.

This is the default because:
1. With M=2 models and typical log-loss differences of ~0.05, performance-weighted routing produces weights within ±0.03 of 0.5 (per Gemini Dispute 3).
2. The calibrator-router in-sample feedback loop biases weights (per all three reviewers).
3. The "convex combination always helps" claim is false for out-of-sample evaluation (per ChatGPT).
4. Simple averaging is deterministic, has zero hyperparameters, and cannot overfit.

### Experimental: PerformanceWeightedRouter

**Renamed from CFARouter.** The name "CFA" overstates the sophistication of what is a softmax-weighted average. Per Claude: "Reserve 'CFA' for actual crowd aggregation methods with multiple independent forecasters."

If enabled:

**Step 1:** Compute LOO-CV calibrated log-loss per model on Zone B.
For each game i in Zone B, fit calibrator on remaining 62 games, predict game i, compute loss. This eliminates the in-sample optimism bias identified by all three reviewers.

```
LL_m^(LOO) = -(1/N) * Σᵢ [ yᵢ * log(clip(p_m,i^(LOO))) + (1-yᵢ) * log(1-clip(p_m,i^(LOO))) ]
```

**Step 2:** Compute weights via softmax.
```
w_m = exp(-LL_m^(LOO) / τ) / Σ_{m'} exp(-LL_{m'}^(LOO) / τ)
```

**Step 3:** Freeze weights. Apply to Zone C:
```
p_combined = Σ_m w_m * p_m^(C)
```

**Diagnostic (not for weight computation):** Log mean absolute prediction difference:
```
D_direct = (1/N) * Σᵢ |p_model1,i - p_model2,i|
```

This replaces RSC divergence, which conflates distributional shape with per-sample disagreement.

**Promotion criterion:** Mean Δ = LL_simple_avg - LL_weighted > 0.005 across ≥5 folds, with bootstrap 95% CI excluding zero.

```python
class SimpleAverageRouter:
    """Default. No fitting required."""
    def combine(self, calibrated_preds: dict[str, np.ndarray]) -> np.ndarray:
        return np.mean(list(calibrated_preds.values()), axis=0)

class PerformanceWeightedRouter:
    """Experimental. Requires LOO-CV calibration losses."""
    def __init__(self, model_names: list[str], tau: float = 1.0): ...
    
    def fit(self, loo_cv_losses: dict[str, float]) -> None:
        """Accepts pre-computed LOO-CV losses, NOT raw predictions+labels.
        This prevents the router from accessing Zone B labels directly."""
        ...
    
    def combine(self, calibrated_preds: dict[str, np.ndarray]) -> np.ndarray: ...
    def get_weights(self) -> dict[str, float]: ...
```

**What is banned:**
- Per-sample routing (any form)
- Learned routing (any parameters beyond global weights)
- Router fitting on Zone C data
- Router accessing raw Zone B labels (receives only pre-computed LOO losses)

---

## 10. EVALUATION AND STATISTICAL INFERENCE SPECIFICATION

### Primary Metric

Kaggle Logistic Brier Score (c=7.0 log-loss with clip at 10⁻⁷).

### Per-Fold Metrics

For each fold, compute and log:
- `eval_logloss_c7`: Primary metric on Zone C
- `eval_brier_standard`: Standard Brier score (for comparison)
- `eval_calibration_error`: Expected Calibration Error (10-bin)
- `eval_logloss_model1_only`: Single-model baseline
- `eval_logloss_model2_only`: Single-model baseline (if enabled)
- `eval_logloss_simple_avg`: Simple average baseline
- `eval_logloss_weighted`: Weighted router (if enabled)

### Cross-Fold Aggregation (New, Per Reviews)

The ExperimentRunner computes:
- **Mean** per-fold log-loss across all folds
- **Bootstrap 95% CI** on the mean (10,000 resamples with replacement across folds)
- **Paired difference CI:** For each comparison (ensemble vs. baseline), compute Δ_k = LL_baseline,k - LL_ensemble,k per fold k, then bootstrap the mean Δ.
- **Decision criterion:** "Ensemble strategy X is adopted if the lower bound of the bootstrap 95% CI for mean(Δ) > 0."

### Effect-Size Thresholds

Given ~63 games per fold and ~10 folds, the standard error of per-fold log-loss is approximately 0.05–0.10. Differences < 0.01 are indistinguishable from noise. The pipeline must not make decisions based on improvements < 0.01 in aggregate log-loss.

### What Constitutes Real Improvement

Per Claude: "A difference of 0.02 in log-loss between the ensemble and a single model is well within noise. Without CIs, the team will see 'ensemble beats TabPFN by 0.018 on fold 7' and interpret it as signal when it's noise."

The MetricEngine must output uncertainty quantification on every metric. Point estimates without CIs are forbidden in decision-making contexts.

```python
class MetricEngine:
    def __init__(self, c: float = 7.0): ...
    
    def evaluate(self, predictions: np.ndarray, eval_labels: EvalLabels) -> MetricReport:
        """Accepts EvalLabels (typed), not raw arrays."""
        ...
    
    def bootstrap_ci(self, predictions: np.ndarray, eval_labels: EvalLabels,
                     n_bootstrap: int = 10000, ci: float = 0.95) -> ConfidenceInterval: ...
    
    def reliability_curve(self, predictions: np.ndarray, eval_labels: EvalLabels,
                         n_bins: int = 10) -> ReliabilityCurve: ...

class ExperimentRunner:
    def __init__(self, fold_definitions: list[tuple], config: PipelineConfig, 
                 tracker: AimMLflowTracker): ...
    
    def run_all_folds(self) -> ExperimentReport: ...
    
    def aggregate_results(self, fold_results: list[FoldResult]) -> AggregateReport:
        """Computes mean, bootstrap CIs, paired difference CIs.
        Returns is_ensemble_justified: bool based on CI criterion."""
        ...
```

---

## 11. ORCHESTRATION DAG AND LIFECYCLE

### Stage-by-Stage Lifecycle for One Fold

```
Stage 0:  FoldContext Creation
          Input: fold definition (train_seasons, cal_season, eval_season, seed)
          Output: frozen FoldContext
          Mutable: nothing after this stage
          Forbidden: accessing any data

Stage 1:  Cutoff Policy + LeakageGuard Initialization
          Input: FoldContext
          Output: Day133CutoffPolicy (stateless), LeakageGuard (bound to context)
          Frozen: FoldContext

Stage 2:  Feature Materialization
          Input: raw games DataFrame (read-only), cutoff policy
          Output: TeamFeatureTable (immutable snapshot, deep-copied)
          Guard: cutoff_policy.filter_dataframe() called FIRST
          Guard: LeakageGuard.assert_no_future_games()
          Frozen: TeamFeatureTable after this stage

Stage 3:  Matchup Assembly
          Input: TeamFeatureTable, matchup pairs for all three zones
          Output: MatchupFeatureMatrix per zone (immutable)
          Guard: Zone C matchups = all C(K,2) pairs, not actual games
          Guard: Features use only DayNum < 133 data regardless of zone
          Frozen: All feature matrices after this stage

Stage 4:  Dataset Materialization
          Input: MatchupFeatureMatrix per zone, labels for Zone A and B ONLY
          Output: TrainDataset, CalDataset, EvalDataset (NO labels)
          Guard: build_eval() has no labels parameter (type firewall)
          Frozen: All datasets after this stage

Stage 5:  Base Model Fitting (FIT-TIME ONLY)
          Input: TrainDataset
          Output: fitted LightGBMBinaryCore (and TabPFNCore if enabled)
          Guard: models see Zone A data only
          Frozen: model weights after this stage

Stage 6:  Zone B Prediction
          Input: CalDataset.X, fitted models
          Output: raw predictions per model on Zone B
          Stored: PredictionRepository (zone="cal", calibrated=False)

Stage 7:  Calibrator Fitting (FIT-TIME ONLY)
          Input: raw Zone B predictions, CalDataset (typed, carries Zone B labels)
          Output: fitted NelderMeadCalibrator per model
          Guard: fit() accepts CalDataset, not raw arrays
          Frozen: calibrator params after this stage

Stage 8:  Zone B Calibrated Prediction
          Input: raw Zone B predictions, frozen calibrators
          Output: calibrated Zone B predictions
          Stored: PredictionRepository (zone="cal", calibrated=True)

Stage 9:  Router Fitting (FIT-TIME ONLY)
          Input: calibrated Zone B predictions + CalDataset (for LOO-CV if router enabled)
          Output: frozen router weights
          DEFAULT: SimpleAverageRouter (no fitting needed)
          EXPERIMENTAL: PerformanceWeightedRouter (LOO-CV losses)
          Guard: router receives LOO losses, not raw labels (if weighted)
          Frozen: router weights after this stage

═══════════════ FREEZE BOUNDARY ═══════════════
After this point, NO fitting occurs. All parameters are frozen.
Zone C predictions do not exist yet.

Stage 10: Zone C Prediction (INFERENCE ONLY)
          Input: EvalDataset.X, frozen models
          Output: raw predictions per model on Zone C
          Guard: models are frozen; no .fit() calls permitted
          Stored: PredictionRepository (zone="eval", calibrated=False)

Stage 11: Zone C Calibrated Prediction (INFERENCE ONLY)
          Input: raw Zone C predictions, frozen calibrators
          Output: calibrated Zone C predictions
          Guard: calibrators are frozen; no .fit() calls permitted
          Stored: PredictionRepository (zone="eval", calibrated=True)

Stage 12: Router Combination (INFERENCE ONLY)
          Input: calibrated Zone C predictions, frozen router
          Output: final combined predictions
          Guard: router is frozen; no .fit() calls permitted

Stage 13: Evaluation (LABELS LOADED HERE)
          Input: combined Zone C predictions
          LABELS LOADED NOW: EvalLabels loaded from separate data source
          Guard: LeakageGuard.assert_eval_labels_not_in_memory() passed at stages 0-12
          Output: MetricReport with all metrics and bootstrap CIs

Stage 14: Tracking and Persistence
          Input: all artifacts, metrics, configs from stages 0-13
          Output: logged run in aim-mlflow, persisted manifest
          Guard: tracker is write-only, no read methods exposed to fold code
```

### Information Access Matrix

| Component | Zone A Labels | Zone B Labels | Zone C Labels | Calibrated Probs | Future Folds |
|-----------|:---:|:---:|:---:|:---:|:---:|
| FoldContext | — | — | — | — | Never |
| FeatureStore | — | — | — | — | Never |
| Base Models | ✓ (fit) | — | — | — | Never |
| Calibrators | — | ✓ (fit) | — | Own model, Zone B | Never |
| Router | — | LOO losses only | — | All models, Zone B | Never |
| MetricEngine | — | — | ✓ (Stage 13 only) | Zone C combined | Never |
| Tracker | — | — | — | — | Never |

---

## 12. CLASS AND MODULE BLUEPRINT

### 12.1 Day133CutoffPolicy (Default, Stateless)

- **Responsibility:** Answers whether a datum is permissible given the DayNum 133 cutoff.
- **Statefulness:** None. The constant 133 is hardcoded.
- **Constructor:** `Day133CutoffPolicy()` — no parameters.
- **Methods:**
  - `filter_dataframe(df, context) -> pd.DataFrame`: drops rows with DayNum ≥ 133 in the prediction season
  - `assert_permitted(season, day_num, context) -> None`: raises `CutoffViolationError`
- **Testing:** Feed DayNum 134 row; assert removed. Feed DayNum 132 row; assert kept.

### 12.2 LeakageGuard (Default, Stateless)

- **Responsibility:** Runtime assertion checker at every stage boundary.
- **Methods:**
  - `assert_no_future_games(df, max_season, max_daynum)`
  - `assert_no_eval_labels_in_memory(orchestrator_state, context)` — NEW: checks no Zone C labels exist before Stage 13
  - `assert_predictions_match_zone(matchup_ids, zone, context)`
  - `assert_artifact_provenance(artifact_path, context)` — NEW: validates embedded fold_id/config_hash match current context
- **Testing:** This class IS the test infrastructure. Inject violations; verify exceptions.

### 12.3 RollingFeatureStore (Default, Stateful per fold)

- **Responsibility:** Computes per-team rolling statistics from games with DayNum < 133.
- **Constructor:** `RollingFeatureStore(raw_games, cutoff_policy)`
- **Methods:**
  - `materialize(context) -> TeamFeatureTable`: calls cutoff_policy.filter_dataframe FIRST
  - `freeze() -> None`: prevents further materialization (NEW per Claude review)
- **Failure modes:** Post-cutoff games in rolling windows. Tested by injecting DayNum 140 game.
- **Must be instantiated fresh per fold.** Never shared, never cached across folds.

### 12.4 MasseyOrdinalExtractor (Default, Stateless)

- **Responsibility:** Computes temporally safe Massey ordinal features per the ChatGPT audit pattern.
- **Constructor:** `MasseyOrdinalExtractor(massey_df, cutoff_policy, system_allowlist=None)`
- **Methods:**
  - `safe_snapshot(season, team_id, cutoff_day) -> dict[str, float|None]`
  - `extract_features(matchups, context) -> pd.DataFrame`
- **Implements:** ChatGPT's `safe_massey_snapshot()` exactly. Per-system time series. LAD logic.
- **Failure modes:** Using RankingDayNum ≥ 133. Tested by injecting post-cutoff ranking.

### 12.5 FeatureAssembler (Default, Stateless)

- **Responsibility:** Joins team features into matchup features with canonical ordering.
- **Constructor:** `FeatureAssembler(feature_columns, symmetry='diff_and_sum')`
- **Methods:** `assemble_matchups(team_features, matchups) -> MatchupFeatureMatrix`
- **Invariant:** Team1ID < Team2ID always. Tested by verifying symmetry.

### 12.6 DatasetMaterializer (Default, Stateless)

- **Methods:**
  - `build_train(features, labels, context) -> TrainDataset`
  - `build_cal(features, labels, context) -> CalDataset`
  - `build_eval(features, context) -> EvalDataset` — NO labels parameter
- **Type firewall:** `build_eval` signature rejects labels at API level.

### 12.7 PredictionRepository (Default, Stateful per fold)

- **Responsibility:** Stores raw and calibrated predictions, keyed by (model, zone).
- **Write-once-freeze pattern:** Each (model, zone, calibrated) tuple can be written once. `freeze()` prevents all further writes.
- **Methods:** `store()`, `retrieve()`, `store_calibrated()`, `retrieve_calibrated()`, `freeze()`, `export()`
- **No `retrieve_labels()` method.** Labels are not stored here.

### 12.8 FoldOrchestrator (Default, Stateful)

- **Responsibility:** Executes stages 0–14 in strict order for one fold.
- **Constructor:** `FoldOrchestrator(context, config)`
- **Methods:** `run() -> FoldResult`, `run_to_stage(stage: int)`
- **Stage tracking:** Prevents re-execution and out-of-order execution.
- **Zone C labels lifecycle (NEW):** Labels are NOT loaded until Stage 13. The orchestrator calls `MetricEngine.load_eval_labels(season, matchup_ids)` at Stage 13 only. Before Stage 13, no attribute of the orchestrator contains Zone C outcomes. LeakageGuard.assert_no_eval_labels_in_memory() is called at each stage boundary.

### 12.9 ExperimentRunner (Default, NEW per reviews)

- **Responsibility:** Outer loop over folds. Aggregation. Decision logic.
- **Constructor:** `ExperimentRunner(fold_definitions, config, tracker)`
- **Methods:**
  - `run_all_folds() -> ExperimentReport`
  - `aggregate_results(fold_results) -> AggregateReport` with mean, bootstrap CIs, paired Δ CIs
  - `is_ensemble_justified() -> bool` — True iff bootstrap 95% CI for Δ excludes zero
- **Fold selection rules:** Non-overlapping eval seasons. Exclude 2020 (no tournament).
- **Output:** `ExperimentReport` with aggregate metrics, CIs, per-fold breakdown, `is_ensemble_justified`.

### 12.10 PipelineConfig (Default, NEW per reviews)

```python
@dataclass(frozen=True)
class PipelineConfig:
    # Feature flags
    enable_tabpfn: bool = False
    enable_graph_features: bool = False
    enable_weighted_router: bool = False
    
    # LightGBM
    lgbm_params: dict = field(default_factory=lambda: {
        'objective': 'binary', 'metric': 'binary_logloss',
        'num_leaves': 31, 'learning_rate': 0.05, 'n_estimators': 500,
        'max_delta_step': 1.0, 'deterministic': True, 'num_threads': 1
    })
    
    # TabPFN
    tabpfn_max_rows: int = 10000
    tabpfn_subsample_strategy: str = 'recency_weighted'
    tabpfn_n_ensemble: int = 16
    
    # Calibration
    cal_bounds_a: tuple = (0.3, 3.0)
    cal_bounds_b: tuple = (-2.0, 2.0)
    cal_lambda_reg: float = 0.1
    cal_n_restarts: int = 5
    cal_c: float = 7.0
    
    # Router
    router_tau: float = 1.0
    
    # Evaluation
    bootstrap_n: int = 10000
    bootstrap_ci: float = 0.95
    
    # Massey
    massey_system_allowlist: list[str] | None = None
    massey_stale_day_threshold: int = 128
    
    def config_hash(self) -> str: ...
```

### 12.11 AimMLflowTracker (Default)

- **Responsibility:** Logs metrics, params, artifacts. Write-only.
- **Critical constraint (NEW):** Exposes NO read/query methods to fold-level code. Only ExperimentRunner (outer loop) may aggregate. Per Claude: "AimMLflowTracker should expose no read() or query() methods to fold-level code."
- **Artifact provenance:** Every logged artifact includes embedded fold_id and config_hash. Per ChatGPT: "filename uniqueness is not the same as provenance enforcement."

---

## 13. REPRODUCIBILITY AND TRACKING SPEC

### Run Hierarchy

```
PARENT: "ncaa_2026_v{version}"
├── CHILD: "fold_{id}_lgbm"        [base model]
├── CHILD: "fold_{id}_tabpfn"      [base model, if enabled]
├── CHILD: "fold_{id}_calibration"  [calibrators]
├── CHILD: "fold_{id}_router"       [router]
├── CHILD: "fold_{id}_evaluation"   [metrics + predictions]
└── (repeat per fold)
AGGREGATE: "experiment_aggregate"    [cross-fold summary]
```

### Seed Policy

- Each fold has a deterministic seed derived from: `hash(pipeline_version + fold_id + str(train_seasons))`
- All stochastic components (LightGBM, TabPFN subsampling, bootstrap) use this seed.
- Seed is logged in manifest.

### Data Hash Policy

- `schema_hash`: SHA256 of raw input file checksums. Detects data drift.
- `feature_config_hash`: SHA256 of feature column list + window sizes + flags.
- Both hashes are logged per fold and validated on rerun.

### Environment Capture

Per ChatGPT: manifest alone is NOT sufficient for reproduction. The manifest is an index into tracked artifacts. Full reproduction requires:
- Manifest JSON
- All referenced model/calibrator/router artifacts
- `pip freeze` output (logged as artifact)
- Python version, OS, hardware spec
- Git commit hash

### Secure Handling of Eval Labels

- Eval labels are never logged as a standalone artifact accessible to fold code.
- The `predictions_eval_{fold_id}.parquet` logged at Stage 14 contains (matchup_id, p_model1, p_model2, p_combined, label) — but this artifact is written AFTER Stage 13 metric computation. It is a post-hoc record, not an input to any pipeline stage.

---

## 14. TEST PLAN AND SAFETY HARNESS

### Tier 1: Blocks Release (Must Pass)

| Test | Target | Failure |
|------|--------|---------|
| `test_cutoff_filters_day134` | Day133CutoffPolicy | DayNum 134 row survives |
| `test_massey_cutoff_strict` | MasseyOrdinalExtractor | RankingDayNum ≥ 133 value used |
| `test_eval_dataset_no_labels` | DatasetMaterializer | `hasattr(eval_ds, 'y')` is True |
| `test_build_eval_rejects_labels` | DatasetMaterializer | No TypeError when labels passed |
| `test_calibrator_rejects_eval_type` | NelderMeadCalibrator | fit() accepts EvalLabels without error |
| `test_no_eval_labels_before_stage13` | FoldOrchestrator | Zone C labels in memory before Stage 13 |
| `test_fold_context_frozen` | FoldContext | Attribute modification succeeds |
| `test_prediction_repo_freeze` | PredictionRepository | Write succeeds after freeze() |
| `test_artifact_provenance_validated` | All loaders | Artifact with wrong fold_id loads successfully |
| `test_feature_symmetry` | FeatureAssembler | assemble(A,B) ≠ assemble(B,A) with flipped label |
| `test_no_post133_games_in_features` | RollingFeatureStore | Features change when DayNum 140 game added |
| `test_no_tournament_edges_in_graph` | GraphScalarExtractor | Graph changes when tournament games added |
| `test_no_winner_loser_coded_features` | FeatureAssembler | WTeamID or LTeamID appears as feature |
| `test_deterministic_fold` | FoldOrchestrator | Same seed produces different metrics |

### Tier 2: Blocks Experimental Promotion

| Test | Target | Failure |
|------|--------|---------|
| `test_tabpfn_subsample_deterministic` | TabPFNCore | Different seeds produce same subsample |
| `test_tabpfn_subsample_variance` | TabPFNCore | σ(log-loss) > 0.005 across strategies |
| `test_router_weights_from_loo` | PerformanceWeightedRouter | Weights derived from in-sample (non-LOO) losses |
| `test_graph_features_temporal_safety` | GraphScalarExtractor | Feature for Day t game changes when Day t+1 data added |
| `test_router_improvement_significant` | ExperimentRunner | Router improvement CI includes zero |

### Tier 3: Diagnostic (Warnings, Not Blockers)

| Test | Target | Warning |
|------|--------|---------|
| `test_calibrator_bound_hit` | NelderMeadCalibrator | Fitted a or b within 0.01 of bound |
| `test_router_weight_deviation` | PerformanceWeightedRouter | |w - 0.5| < 0.02 on all folds |
| `test_bootstrap_ci_width` | MetricEngine | 95% CI width > 0.10 (too uncertain for decisions) |

---

## 15. PHASED IMPLEMENTATION ROADMAP

### Phase 1: Foundation + Single Model (Weeks 1–2)

**Goal:** End-to-end LightGBM-only pipeline with temporal safety, producing metrics with CIs.

Build:
1. `context/fold_context.py`, `policies/cutoff_policy.py`, `policies/leakage_guard.py`
2. `data/datasets.py` (all typed contracts including EvalLabels)
3. `features/rolling_store.py`, `features/massey_extractor.py`, `features/assembler.py`
4. `data/materializer.py`, `data/prediction_repo.py`
5. `models/lgbm_core.py`
6. `calibration/nelder_mead_calibrator.py`, `calibration/objectives.py`
7. `routing/simple_average_router.py` (pass-through for single model)
8. `metrics/engine.py` with bootstrap CIs
9. `orchestration/fold_orchestrator.py`
10. Full Tier 1 test suite

**Phase gate:** All Tier 1 tests pass. LightGBM-only baseline established with CIs on ≥5 folds.

### Phase 2: Tracking + Reproducibility (Week 3)

Build:
1. `tracking/aim_mlflow_tracker.py`
2. `context/run_manifest.py`, `context/pipeline_config.py`
3. `orchestration/experiment_runner.py` with aggregation and decision logic
4. Manifest round-trip test (reproduce fold from manifest + artifacts)

**Phase gate:** Clean-room replay succeeds (manifest → rerun → identical metrics).

### Phase 3: Second Model + Calibration Validation (Week 4)

Build:
1. `models/tabpfn_core.py` with subsampling
2. Calibration stability analysis (bootstrap/jackknife on (a,b))
3. Optimizer comparison (Nelder-Mead vs L-BFGS-B)
4. Two-model pipeline with simple average

**Phase gate:** TabPFN subsampling variance < 0.005. Calibration parameters stable across bootstrap resamples. Two-model simple average ≥ single-model baseline.

### Phase 4: Experimental Router + Graph Features (Week 5+)

Build (behind flags):
1. `routing/performance_weighted_router.py` with LOO-CV
2. `features/graph_extractor.py`
3. Router ablation study
4. Graph feature lift study

**Phase gate:** Pre-registered acceptance criteria met. Bootstrap CIs for improvement exclude zero.

### What Not to Build

- Dense embeddings of any kind
- Learned meta-models / stacking
- Per-sample routing logic
- Multi-round tournament simulation
- Any component not fully specified in this document

---

## 16. OPEN QUESTIONS AND DECISION TABLE

| # | Issue | Current Default | Alternatives | Status | Evidence to Change |
|---|-------|----------------|-------------|--------|-------------------|
| Q1 | Nelder-Mead vs L-BFGS-B | Nelder-Mead (multi-start) | L-BFGS-B | Unresolved | Compare on ≥5 folds; if identical (a,b) to 1e-3, use L-BFGS-B |
| Q2 | Calibration λ value | 0.1 | {0.01, 0.05, 0.1, 0.5} | Unresolved | Sensitivity analysis on historical folds |
| Q3 | TabPFN subsample strategy | Recency-weighted | Uniform, stratified-by-outcome | Unresolved | Variance < 0.005 across strategies |
| Q4 | Softmax τ for weighted router | 1.0 | {0.5, 1.0, 2.0} | Unresolved (experimental only) | Sensitivity analysis if router is promoted |
| Q5 | Number of evaluation folds | Max available (~10) | 5, 7 | Settled | Use all available non-overlapping eval seasons |
| Q6 | Massey system allowlist | All systems with data at DayNum < 133 | Curated list of non-stale systems | Unresolved | Compare performance with/without stale systems |
| Q7 | Graph feature window | Full season up to Day 133 | Rolling 30-day window | Unresolved (experimental only) | If graph features are promoted, test window sensitivity |
| Q8 | LightGBM hyperparameters | Default config in PipelineConfig | Bayesian optimization on Zone A | Settled (use defaults for now) | Hyperparameter tuning is a Phase 3+ activity |

---

## 17. ENGINEER / AGENT MISUSE WARNINGS

These are the most likely ways future engineers or coding agents will break this architecture:

1. **Passing Zone C labels to calibrator or router.** The type firewall prevents this IF signatures accept CalDataset/EvalLabels. If someone refactors to accept raw arrays "for flexibility," the firewall collapses.

2. **Using `groupby('TeamID').agg(...)` without Season filter.** This creates cross-season leakage. Every aggregation must include Season in the groupby or be explicitly designed as a cross-season feature with temporal guards.

3. **Taking "latest" Massey ordinal without RankingDayNum filter.** The most common Kaggle leakage pattern. Use `safe_massey_snapshot()` exclusively.

4. **Caching features across folds for performance.** Natural optimization, catastrophic leakage. Each fold must instantiate a fresh RollingFeatureStore.

5. **Loading all data at pipeline start.** If Zone C labels are in memory during fitting stages, a refactor could pass them to a fitting method. Zone C labels must be lazy-loaded at Stage 13.

6. **Building graph features on full-season adjacency.** Must use A_{133} (DayNum < 133 edges only). Must recompute per fold. Must not include tournament edges.

7. **Treating model naming as cosmetic.** The class is `LightGBMBinaryCore`, not `LightGBMHuberCore`. The lifecycle diagram, tests, manifests, and code must all use the canonical name. Inconsistency here caused confusion in the original spec (per ChatGPT review).

8. **Assuming the router helps.** The default is simple average. The weighted router is experimental. Do not promote it based on per-fold point estimates without CIs.

9. **Trusting manifest-only reproduction.** The manifest is an index. Full reproduction requires manifest + all referenced artifacts + environment spec.

10. **Fitting calibrators and evaluating them on the same Zone B data, then using that evaluation to set router weights.** This is the in-sample feedback loop all three reviewers flagged. Router weights must come from LOO-CV losses.

---

## 18. FINAL AUTHORITATIVE PIPELINE SUMMARY

The NCAA March Madness 2026 prediction pipeline is a **temporally safe, calibrated, simple-average ensemble** of one or two base models, validated via strict expanding-window cross-validation.

**Default path:**
1. Compute rolling team statistics and Massey ordinals from regular-season games with DayNum < 133.
2. Assemble matchup features with canonical Team1ID < Team2ID ordering.
3. Train LightGBM (binary log-loss, max_delta_step gradient clipping) on Zone A.
4. Predict Zone B. Fit per-model Platt calibrator on Zone B with multi-start Nelder-Mead, L2 regularization toward identity, bounds a ∈ [0.3, 3.0], b ∈ [-2.0, 2.0].
5. Predict Zone C with frozen model and frozen calibrator.
6. Output: calibrated probability for each of the C(K,2) possible tournament matchups.
7. Evaluate with Logistic Brier Score (c=7.0) on actual tournament games, with bootstrap 95% CIs.

**When two models are enabled:** Simple average of calibrated predictions from both models. The performance-weighted router is experimental and requires demonstrated lift with bootstrap CI excluding zero across ≥5 folds before promotion.

**What makes this pipeline trustworthy:**
- Type-level firewalls prevent Zone C labels from reaching fitting methods.
- Zone C labels are not loaded until Stage 13.
- All features respect the DayNum < 133 cutoff, enforced by policy, guard, and test.
- All artifacts embed provenance and validate on load.
- All evaluation metrics include uncertainty quantification.
- Complexity is earned through evidence, not assumed through theory.

**What this pipeline does NOT do:**
- Learn a router
- Use dense embeddings
- Use post-cutoff information
- Make decisions from point estimates without confidence intervals
- Preserve a component because it sounds advanced

---

*End of Master Architecture Specification. This document supersedes all prior drafts and is the authoritative reference for implementation.*