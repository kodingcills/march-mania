# Current Phase Plan: NCAA March Mania 2026 Quant Pipeline

## Document Purpose
This file is the execution-control document for the **current implementation frontier**.

It defines:
- the **current phase**
- the **current active step**
- the **exact build target**
- the **architectural laws in force**
- what is **in scope**
- what is **out of scope**
- the **acceptance criteria** required to advance
- the **common failure modes** most likely to cause leakage, symmetry breakage, or stale-feature contamination

This document is subordinate only to:
1. `MASTER_ARCHITECTURE_V2.md`
2. the current canonical decision ledger / memory capsule

If a coding task conflicts with this phase plan, the task must be narrowed, deferred, or escalated.

---

## Current Program State

**Current Project Status:** Phase 1 — Foundation + Single Model  
**Current Active Step:** Step 3 — Feature Engineering & Assembly

**Primary Objective of Phase 1:**  
Build the leak-proof, deterministic, pregame-only pipeline foundation required before dataset materialization, modeling, calibration, or routing are allowed.

**Success Condition for Step 3:**  
We have a deterministic, temporally safe feature layer that:
- computes rolling team-level features without future leakage
- extracts safe Massey snapshots using explicit Last Available Day logic
- assembles matchup features with canonical ordering and symmetry-preserving transformations
- does not yet materialize final train/cal/eval typed datasets
- does not yet touch modeling or prediction logic

---

## Source-of-Truth Assumptions in Force

The following decisions are active architectural law for this step:

1. **Day 133 Cutoff Law**
   - All pre-tournament feature eligibility is bounded by `DayNum < 133`
   - No feature engine may aggregate or snapshot data at or beyond Day 133
   - Cutoff enforcement must happen **before** aggregation, not after

2. **Temporal Safety Law**
   - No feature for a game may depend on information from that game or any future game
   - Rolling aggregations must only use the historical prefix visible at the query point

3. **Type Firewall Law**
   - Feature engines may produce feature tables / feature payloads only
   - They must not collapse into dataset materialization responsibilities
   - They must not mix eval labels into feature assembly

4. **Freeze / Immutability Law**
   - Once a feature store is frozen, it must reject any further mutation
   - Freeze exists to prevent accidental post-hoc contamination of precomputed state

5. **Canonical Ordering Law**
   - Matchup assembly must enforce `Team1ID < Team2ID`
   - Feature construction must preserve symmetry under team-order permutations
   - Differential and additive transforms must be used deliberately and consistently

6. **No Architecture Creep Law**
   - Step 3 builds feature engines only
   - It must not drift into dataset materialization, models, routing, calibration, or tracking

---

## Recently Completed

### Phase 1, Step 1 — Temporal Skeleton
Completed:
- `fold_context.py`
- `cutoff_policy.py`
- `leakage_guard.py`

Outcome:
- temporal fold context exists
- day cutoff policy exists
- first-line leakage guardrails exist

### Phase 1, Step 2 — Data Contracts & Type Firewalls
Completed:
- `TrainDataset`
- `CalDataset`
- `EvalDataset`
- `EvalLabels`

Outcome:
- strict no-label boundary for `EvalDataset`
- immutable / read-only contract enforcement
- **140/140 tests passing**

---

## Current Objective

## Phase 1, Step 3 — Feature Engineering & Assembly

### Target Files
1. `ncaa_pipeline/features/rolling_store.py`
2. `ncaa_pipeline/features/massey_extractor.py`
3. `ncaa_pipeline/features/assembler.py`

### Primary Goal
Build the feature engines that transform raw historical, pregame-eligible information into deterministic, symmetry-safe matchup feature representations.

This step is about **feature generation discipline**, not model training.

---

## Step 3 Detailed Requirements

### A. `rolling_store.py`
This module is responsible for rolling / historical team-level feature computation.

#### Required behavior
- `RollingFeatureStore.materialize()` must call `cutoff_policy.filter_dataframe(...)` **before** performing any pandas aggregation
- no aggregation may be computed on unfiltered post-cutoff data
- all features must be historical-only relative to the query point
- the store must support a `freeze()` mechanism
- once frozen, the store must reject mutation or rebuild attempts

#### Required design intent
- deterministic
- auditable
- explicit about which columns it consumes
- explicit about which aggregations it computes
- no hidden caching that can span incompatible folds unless explicitly keyed and safe

#### Likely feature families
Examples may include:
- rolling win/loss indicators
- rolling scoring margin
- rolling pace/efficiency summaries
- rolling home/away-neutral summaries if legally derivable
- rolling opponent-strength summaries only if built from pregame-safe inputs

#### Must not do
- no tournament data usage
- no post-cutoff aggregation
- no label-aware feature synthesis
- no cross-fold global cache reuse
- no implicit full-season aggregation without query-time restriction

---

### B. `massey_extractor.py`
This module is responsible for safe pregame ordinal extraction from `MMasseyOrdinals`-style data.

#### Required behavior
- implement `safe_snapshot(season, team_id, cutoff_day)`
- enforce `RankingDayNum < 133` for tournament-pregame use
- use explicit **Last Available Day (LAD)** logic
- if Day 133 is unavailable for a system, choose the latest available ranking day strictly below the cutoff
- systems known to be stale from the audit must be handled explicitly by policy

#### Required design intent
- no casual “latest available anywhere” behavior
- no future-looking fallback
- no silent acceptance of stale systems without explicit status
- explicit treatment of missingness

#### Must account for audit reality
- some systems do not publish on Day 133
- some systems stop before Day 128 and may be unusable as tournament priors
- stale-system policy must be explicit, not accidental

#### Must not do
- no ranking lookup at or beyond cutoff
- no interpolation across future days
- no system blending unless explicitly requested later
- no feature assembly responsibilities

---

### C. `assembler.py`
This module is responsible for matchup-level feature assembly.

#### Required behavior
- enforce canonical ordering: `Team1ID < Team2ID`
- generate symmetry-preserving features
- compute `diff_features`
- compute `sum_features`
- preserve enough provenance that downstream code knows what was derived from which base feature family

#### Required design intent
- if team order flips, the representation should remain mathematically consistent
- features must be deterministic and schema-stable
- no labels, no model logic, no calibration logic

#### Canonicalization rule
For any unordered matchup `(A, B)`:
- reorder into `(min(A,B), max(A,B))`
- then build features in a way that remains interpretable after reordering

#### Symmetry expectation
- additive features should remain invariant
- differential features should be consistent with canonical order
- no winner-side schema leakage from legacy Kaggle winner/loser-coded tables

#### Must not do
- no dataset typing
- no label attachment
- no eval materialization
- no probabilistic logic
- no routing / ensembling behavior

---

## In Scope Right Now

Only the following are in scope for Step 3:

- rolling feature computation
- safe ordinal snapshot extraction
- canonical matchup feature assembly
- freeze behavior for feature store state
- tests for leakage prevention, LAD logic, symmetry, and freeze semantics

---

## Explicitly Out of Scope Right Now

The following must **not** be built yet:

### Data Materialization / Prediction State
- `DatasetMaterializer`
- `PredictionRepository`

### Modeling
- `LightGBMBinaryCore`
- `TabPFNCore`

### Calibration / Ensemble Logic
- any calibrator
- any router
- any ensemble weight logic

### Tracking / Experiment Infrastructure
- `AimMLflowTracker`
- manifests
- run lineage logic

### Outer-loop Experiment Orchestration
- fold runners beyond what is minimally required for local feature tests
- bootstrap CI logic
- experiment aggregation logic

If any of these seem “helpful” to complete Step 3, treat that as drift unless explicitly approved.

---

## Deliverables for Step 3

To complete Step 3, the following must exist:

1. `ncaa_pipeline/features/rolling_store.py`
2. `ncaa_pipeline/features/massey_extractor.py`
3. `ncaa_pipeline/features/assembler.py`
4. one or more dedicated test modules for feature-layer behavior

At minimum, tests must prove:
- cutoff policy is applied before aggregation
- post-cutoff rows do not influence features
- LAD logic behaves correctly
- stale / missing ordinal behavior is explicit
- canonical ordering is enforced
- diff/sum features behave consistently under team-order reversal
- freeze blocks mutation/rebuild where intended

---

## Acceptance Criteria (Must Pass Before Advancing)

Before Step 3 is considered complete, all of the following must be proven via `pytest`:

1. `RollingFeatureStore.materialize()` uses cutoff filtering before any aggregation logic
2. Injecting Day 133+ rows does not change pre-cutoff rolling features
3. `freeze()` prevents illegal post-freeze mutation or recomputation
4. `MasseyOrdinalExtractor.safe_snapshot(...)` never returns a ranking from `RankingDayNum >= 133`
5. LAD logic returns the correct last available ranking below cutoff
6. systems that are too stale are explicitly surfaced according to policy
7. `FeatureAssembler` enforces `Team1ID < Team2ID`
8. differential features are consistent with canonical ordering
9. additive features are invariant under team-order permutation
10. no Step 3 code attaches labels or creates train/cal/eval dataset objects
11. tests run without external network calls or future-phase dependencies

---

## Phase Gate to Advance to Step 4

### Gate Name
**Phase 1 / Step 3 Feature Integrity Gate**

### Required Evidence
- all feature-layer tests pass
- no cutoff bypass exists
- no temporal leakage found in rolling or ordinal logic
- canonicalization logic is deterministic and stable
- freeze semantics are enforced

### Promotion Rule
Step 4 may begin only after Step 3 passes with:
- all tests green
- no unresolved architectural contradictions
- no provisional shortcuts left in feature logic
- no hidden coupling to modeling or dataset materialization

---

## Anticipated Next Step

## Phase 1, Step 4 — Dataset Materialization Boundary
Planned focus:
- convert assembled feature payloads into typed dataset contracts
- preserve type firewall
- preserve eval label separation
- prepare clean inputs for the first baseline model

This step must only begin after the feature layer is leak-safe and schema-stable.

---

## Common Failure Modes to Watch For

The following are the most likely implementation mistakes during Step 3:

1. Applying cutoff filtering **after** aggregation instead of before
2. Computing full-season aggregates and pretending they are historical
3. Using Day 133 or later ordinals “because they are available”
4. Allowing stale Massey systems without explicit policy
5. Breaking symmetry by failing to canonicalize matchup order
6. Building features from winner/loser-coded tables in a way that leaks outcome semantics
7. Letting the feature store mutate after freeze
8. Quietly starting dataset materialization or model-prep work inside assembler logic
9. Using broad caches that are not keyed safely by fold / cutoff / source data
10. making schema assumptions that are not asserted in tests

---

## Change-Control Rule for This Step

Any proposal to:
- relax the Day 133 rule
- allow `RankingDayNum >= 133`
- skip LAD logic
- weaken canonical ordering
- mix feature assembly with dataset construction
- bypass freeze semantics
- silently include stale systems without policy

must be treated as an architectural change request, not a local implementation choice.

---

## Recommended Verification Commands

Example commands for this step:

```bash
pytest ncaa_pipeline/tests/test_tier1_foundation.py -q
pytest ncaa_pipeline/tests/test_datasets.py -q
pytest ncaa_pipeline/tests/test_features.py -q
pytest -q