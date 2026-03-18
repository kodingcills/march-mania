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
- the **common failure modes** most likely to cause temporal leakage, zone contamination, provenance drift, or label-boundary violations

This document is subordinate only to:
1. `MASTER_ARCHITECTURE_V2.md`
2. the canonical decision ledger / memory capsule

If a coding task conflicts with this phase plan, the task must be narrowed, deferred, or escalated.

---

## Current Program State

**Current Project Status:** Phase 1 — Foundation + Single Model  
**Current Active Step:** Step 4 — Dataset Materialization & Orchestration

**Primary Objective of Phase 1:**  
Build the leak-proof, deterministic path from raw competition tables to typed, zone-safe dataset contracts, without yet introducing any model, calibrator, router, or experiment-tracking logic.

**Success Condition for Step 4:**  
We have a deterministic, provenance-safe materialization layer that:
- loads the required raw tables cleanly
- respects `FoldContext`
- produces **three disjoint zone outputs**
- preserves the **EvalDataset / EvalLabels firewall**
- generates the **full tournament all-pairs evaluation matrix**
- stamps all materialized artifacts with **fold-local provenance**
- does not yet invoke any modeling or tracking code

---

## Source-of-Truth Assumptions in Force

The following decisions are active architectural law for this step:

1. **Temporal Zoning Law**
   - Zone A = Train
   - Zone B = Calibration
   - Zone C = Evaluation
   - Materialization must preserve these as physically distinct outputs

2. **Type Firewall Law**
   - `TrainDataset` and `CalDataset` may carry labels
   - `EvalDataset` must not carry labels
   - `EvalLabels` must remain physically separate and only be materialized explicitly
   - No convenience API may recombine them

3. **Zone C Permutation Law**
   - `EvalDataset` must represent the full tournament all-pairs candidate space
   - The materializer must not restrict Zone C to only realized tournament games
   - This prevents bracket-topology leakage

4. **Day 133 Cutoff Law**
   - Feature eligibility remains bounded by `DayNum < 133`
   - Step 4 must respect the already-built feature layer and may not introduce post-cutoff contamination through joins or loader shortcuts

5. **Provenance Law**
   - Every materialized dataset must be stamped with the active `fold_id`
   - Fold provenance must remain explicit, not inferred

6. **No Architecture Creep Law**
   - Step 4 builds loading + materialization only
   - It must not drift into model training, calibration, routing, tracking, or experiment aggregation

7. **Determinism Law**
   - Materialization must be deterministic for a given `FoldContext`
   - No random subsampling, shuffling, or non-deterministic ordering is allowed in this step

---

## Recently Completed

### Phase 1, Step 1 — Foundation
Completed:
- temporal skeleton
- cutoff policy
- leakage guards

Outcome:
- fold context exists
- cutoff enforcement exists
- base runtime leakage assertions exist

### Phase 1, Step 2 — Contracts
Completed:
- `TrainDataset`
- `CalDataset`
- `EvalDataset`
- `EvalLabels`

Outcome:
- strict type firewall exists
- immutable/read-only dataset boundaries exist

### Phase 1, Step 3 — Features
Completed:
- `RollingFeatureStore`
- `MasseyOrdinalExtractor`
- `FeatureAssembler`

Outcome:
- temporally safe feature extraction exists
- LAD logic exists
- canonical matchup assembly exists
- provenance stamping exists in assembled outputs
- **185 total tests green**

---

## Current Objective

## Phase 1, Step 4 — Dataset Materialization & Orchestration

### Target Files
1. `ncaa_pipeline/data/materializer.py`
2. `ncaa_pipeline/data/loader.py`

### Primary Goal
Build the “glue layer” that converts raw competition tables plus a `FoldContext` into the correct typed dataset contracts for:
- training
- calibration
- evaluation inputs
- evaluation labels

This step is about **zone-safe construction**, not model consumption.

---

## Step 4 Detailed Requirements

### A. `loader.py`
This module is responsible for deterministic raw-table loading and schema validation.

#### Required behavior
- load required competition tables from disk
- validate required columns exist
- return pandas DataFrames, not dataset objects
- remain deterministic and side-effect-light
- make no modeling assumptions

#### Required design intent
- simple
- explicit
- testable
- easy to mock in unit tests
- no hidden caching unless clearly scoped and safe

#### Loader scope
The loader may load only the tables needed for Step 4, such as:
- regular-season detailed results
- Massey ordinals
- tournament seeds
- tournament compact results if needed for calibration/eval label derivation
- tournament slot/pair support only if strictly needed

#### Must not do
- no feature engineering
- no dataset typing
- no fold splitting
- no tracking
- no silent schema coercion
- no hidden filesystem magic

---

### B. `materializer.py`
This module is responsible for converting raw DataFrames plus `FoldContext` into the typed zone datasets.

#### Required behavior
- accept raw DataFrames and `FoldContext`
- invoke the already-built feature layer rather than reimplementing feature logic
- build:
  - `TrainDataset`
  - `CalDataset`
  - `EvalDataset`
  - `EvalLabels` (when explicitly requested / appropriate)
- keep train/cal/eval physically distinct
- stamp all materialized outputs with the active `fold_id`
- preserve deterministic ordering

#### Required design intent
- the materializer is a coordinator, not a feature engine
- it must reuse:
  - `FoldContext`
  - `Day133CutoffPolicy`
  - `RollingFeatureStore`
  - `MasseyOrdinalExtractor`
  - `FeatureAssembler`
  - dataset contracts from Step 2
- it must not take on model, calibration, or tracking responsibilities

#### Zone semantics
- **Zone A / TrainDataset:** historical regular-season/tournament-derived training examples appropriate for the current fold
- **Zone B / CalDataset:** calibration-period examples with labels
- **Zone C / EvalDataset:** full all-pairs tournament candidate matrix, no labels
- **Zone C / EvalLabels:** realized evaluation labels kept separate from `EvalDataset`

#### Critical boundary rule
The materializer must never attach `.y` or equivalent label-bearing fields to `EvalDataset`.

---

## Zone-Specific Materialization Laws

### Zone A — Train
Must produce:
- feature matrix `X`
- labels `y`
- explicit provenance (`fold_id`)
- deterministic row ordering

Must not:
- include calibration or evaluation data
- include post-cutoff feature contamination
- depend on tournament future structure beyond what is valid for that fold

### Zone B — Calibration
Must produce:
- feature matrix `X`
- labels `y`
- explicit provenance (`fold_id`)

Must not:
- include evaluation labels
- leak Zone C structure into calibration examples

### Zone C — Evaluation Inputs
Must produce:
- full all-pairs matchup feature matrix
- no labels
- explicit provenance (`fold_id`)

Must not:
- filter down to realized games only
- expose a `.y`
- infer later-round matchup existence from actual outcomes

### Zone C — Evaluation Labels
Must produce:
- labels only for the realized evaluation targets
- explicit provenance (`fold_id`)

Must remain:
- physically separate from `EvalDataset`
- inaccessible to any fit/calibration path in this step

---

## All-Pairs Evaluation Rule

For tournament evaluation, the materializer must generate the full combinatorial candidate space.

### Required behavior
- generate all tournament-eligible team pairs for the fold’s evaluation season
- produce the complete unordered matchup space
- canonicalize team order before assembly
- ensure deterministic ordering of rows

### Why this exists
If the materializer only constructs the realized tournament games, the mere existence of later-round matchups can leak earlier outcomes.
The all-pairs matrix prevents this structural leakage.

---

## In Scope Right Now

Only the following are in scope for Step 4:

- raw table loading
- schema validation for required tables
- fold-aware dataset materialization
- train/cal/eval zoning
- all-pairs evaluation matrix generation
- provenance stamping (`fold_id`)
- tests for disjointness, label separation, determinism, and permutation completeness

---

## Explicitly Out of Scope Right Now

The following must **not** be built yet:

### Modeling
- `LightGBMBinaryCore`
- `TabPFNCore`

### Calibration / Ensemble Logic
- any calibrator
- any router
- any ensemble weighting
- any prediction fusion

### Tracking / Experiment Infrastructure
- `AimMLflowTracker`
- manifests
- run lineage persistence

### Outer-loop Experiment Logic
- bootstrap confidence intervals
- full experiment runner
- cross-fold aggregation
- model comparison logic

If any of these seem “helpful” to complete Step 4, treat that as drift unless explicitly approved.

---

## Deliverables for Step 4

To complete Step 4, the following must exist:

1. `ncaa_pipeline/data/loader.py`
2. `ncaa_pipeline/data/materializer.py`
3. one or more dedicated test modules for Step 4 behavior

At minimum, tests must prove:
- correct schema enforcement in the loader
- materializer respects `FoldContext`
- Zone A / B / C are disjoint by construction
- `EvalDataset` has no labels
- `EvalLabels` remain separate
- Zone C all-pairs generation is complete and deterministic
- all outputs are stamped with `fold_id`
- no Step 4 code invokes model/calibrator/router/tracker code

---

## Acceptance Criteria (Must Pass Before Advancing)

Before Step 4 is considered complete, all of the following must be proven via `pytest`:

1. `loader.py` fails loudly when required columns are missing
2. materializer produces `TrainDataset`, `CalDataset`, `EvalDataset`, and `EvalLabels` with the correct contract types
3. `EvalDataset` does not expose labels
4. `EvalLabels` are constructed separately and cannot be confused with `EvalDataset`
5. all materialized outputs carry the active `fold_id`
6. Zone A / B / C are disjoint according to the fold definition
7. Zone C materialization generates the full all-pairs matchup set, not only realized games
8. team ordering in Zone C is canonical and deterministic
9. repeated materialization with identical inputs and `FoldContext` produces identical outputs
10. no Step 4 code imports or calls model, calibration, routing, or tracking components
11. tests run without network calls and without depending on future-phase modules

---

## Phase Gate to Advance to Step 5

### Gate Name
**Phase 1 / Step 4 Materialization Integrity Gate**

### Required Evidence
- all Step 4 tests pass
- all zone boundaries are physically preserved
- eval labels remain separate from eval inputs
- all-pairs evaluation generation is correct and deterministic
- provenance stamping is present on all materialized datasets

### Promotion Rule
Step 5 may begin only after Step 4 passes with:
- all tests green
- no unresolved architectural contradictions
- no temporary shortcuts in zoning or provenance
- no hidden coupling to models or experiment infrastructure

---

## Anticipated Next Step

## Phase 1, Step 5 — Baseline Model Integration
Planned focus:
- first baseline model interface
- safe consumption of `TrainDataset` / `CalDataset` / `EvalDataset`
- probability semantics at the model boundary
- no router / no ensemble / no advanced tracking yet

This step must only begin after the dataset materialization boundary is fully stable.

---

## Common Failure Modes to Watch For

The following are the most likely implementation mistakes during Step 4:

1. restricting Zone C to only realized tournament games
2. attaching labels to `EvalDataset` “for convenience”
3. recombining `EvalDataset` and `EvalLabels` in the materializer
4. bypassing `FoldContext` and materializing from ad hoc season logic
5. reimplementing feature logic inside the materializer instead of calling Step 3 components
6. losing deterministic ordering in the all-pairs matrix
7. failing to stamp `fold_id` onto outputs
8. introducing raw file loading assumptions into tests
9. coupling loader logic to feature logic too tightly
10. importing future-phase components into Step 4

---

## Change-Control Rule for This Step

Any proposal to:
- weaken the eval label firewall
- restrict Zone C to realized games
- materialize without provenance
- merge loading and feature engineering into one layer
- skip typed dataset contracts
- bypass `FoldContext`
- expose labels through convenience APIs

must be treated as an architectural change request, not a local implementation choice.

---

## Recommended Verification Commands

Example commands for this step:

```bash
pytest tests/test_tier1_foundation.py -q
pytest tests/test_datasets.py -q
pytest tests/test_features.py -q
pytest tests/test_materializer.py -q
pytest -q