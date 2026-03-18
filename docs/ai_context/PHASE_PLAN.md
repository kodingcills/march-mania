# Current Phase Plan: NCAA March Mania 2026 Quant Pipeline

## Document Purpose
This file is the execution control document for the current implementation phase.

It defines:
- the **current phase**
- the **current active step**
- what is **in scope**
- what is **out of scope**
- the **acceptance criteria** required to advance
- the **architectural laws** that must be preserved
- the **next immediate build target**

This file is subordinate only to:
1. `MASTER_ARCHITECTURE_V2.md`
2. the current adjudicated canonical memory / decision ledger

If a coding task conflicts with this phase plan, the task must be narrowed, deferred, or escalated.

---

## Current Program State

**Current Project Status:** Phase 1 — Foundation + Single Model  
**Current Active Step:** Step 2 — Data Contracts & Type Firewalls

**Primary Objective of Phase 1:**  
Build the leak-proof, deterministic foundation required before feature engineering, modeling, calibration, or experiment tracking are allowed.

**Success Condition for Phase 1:**  
A strict, typed, immutable data boundary exists between:
- training data
- calibration data
- evaluation inputs
- evaluation labels

No code in later stages should be able to “accidentally” bypass these boundaries.

---

## Source-of-Truth Assumptions in Force

The following decisions are currently treated as active architectural law for this phase:

1. **Day 133 Cutoff Law**
   - All pre-tournament feature eligibility is bounded by DayNum `< 133`
   - This cutoff is global and must not be relaxed in implementation

2. **Type Firewall Law**
   - Training/calibration/evaluation objects must be distinct typed contracts
   - Evaluation inputs and evaluation labels must be physically separated

3. **Temporal Zoning Law**
   - Zone A = Train
   - Zone B = Calibration
   - Zone C = Evaluation
   - No future information may cross backward across these zones

4. **Immutability Law**
   - Context and dataset contracts must be immutable
   - Array mutation must be blocked wherever practical

5. **No Hidden Convenience Law**
   - No broad generic blobs
   - No untyped side channels
   - No “temporary” backdoors for labels or mutable state

---

## Recently Completed

### Sprint 0 — Data Audit
Completed hostile audit of raw competition data.

Key outcomes:
- Verified the **Day 133 cutoff is globally safe**
- Identified **Massey ordinal sparsity / stale systems**
- Flagged **2020 / 2021 anomaly considerations**
- Confirmed raw data is usable, but feature logic must still obey strict pregame and temporal guards

### Phase 1, Step 1 — Temporal Skeleton
Completed:
- `march_mania/context/fold_context.py`
- `march_mania/policies/cutoff_policy.py`
- `march_mania/policies/leakage_guard.py`

Validation status:
- **53 Tier-1 foundation tests passed**
- Environment: **macOS MPS**
- Temporal context, cutoff policy, and leakage assertions are now implemented and verified

---

## Current Objective

## Phase 1, Step 2 — Data Contracts & Type Firewalls

**Target file:**  
- `march_mania/data/datasets.py`

**Primary goal:**  
Implement the strict typed dataset contracts that make leakage materially harder, not just “discouraged by convention.”

### Required Contracts
The following objects must be defined as immutable, explicit data contracts:

- `TrainDataset`
- `CalDataset`
- `EvalDataset`
- `EvalLabels`

### Core design intent
These classes must:
- enforce physical separation between evaluation inputs and labels
- prevent accidental mutation of arrays
- make it difficult for future code to misuse data from the wrong zone
- serve as the canonical boundary objects for later phases

---

## Step 2 Detailed Requirements

### Functional Requirements
1. `TrainDataset` must hold training inputs and targets
2. `CalDataset` must hold calibration inputs and targets
3. `EvalDataset` must hold evaluation inputs only
4. `EvalLabels` must hold evaluation labels only
5. `EvalDataset` must **not** expose a `.y` attribute
6. All underlying NumPy arrays must be made read-only in `__post_init__`
7. Constructors must validate shape consistency
8. Public fields must be typed explicitly
9. Contracts must be narrow and phase-appropriate

### Structural Requirements
- Use frozen dataclasses where appropriate
- Prefer `slots=True` if consistent with the codebase style
- No inheritance hierarchy unless clearly necessary
- No generic “BaseDataset” abstraction unless it materially improves correctness
- No optional fields that blur zone semantics
- No fallback behavior that silently accepts malformed arrays

### Validation Requirements
At minimum, dataset construction should validate:
- dimensional consistency between `X` and `y`
- row counts align
- arrays are NumPy arrays
- labels only exist in the correct dataset types
- arrays are set to non-writeable

---

## Critical Non-Negotiable Requirement

### EvalDataset must not contain labels
This is the single most important rule of Step 2.

`EvalDataset` must be physically incapable of carrying a `.y` attribute.

This is not a documentation preference.
This is a hard anti-leakage boundary.

Future orchestration, models, and evaluators must only receive `EvalLabels` when explicitly authorized at the metric-computation stage.

---

## In Scope Right Now

Only the following are in scope for Step 2:

- typed dataset contracts
- immutability enforcement
- shape / schema validation
- anti-label-boundary enforcement
- unit tests for contract integrity

---

## Explicitly Out of Scope Right Now

The following must **not** be built yet:

### Modeling
- `LightGBMBinaryCore`
- `TabPFNCore`

### Feature Engineering
- `RollingFeatureStore`
- `MasseyOrdinalExtractor`
- graph feature logic
- matchup feature assembly

### Calibration / Ensemble Logic
- `PlattScalingCalibrator`
- `SimpleAverageRouter`
- `PerformanceWeightedRouter`

### Tracking / Experiment Infrastructure
- `AimMLflowTracker`
- manifests
- run lineage logic

### Data Loading / Materialization
- CSV ingestion
- fold materialization
- feature matrix generation

If work on any of these seems “helpful” to complete Step 2, it should be treated as drift unless explicitly approved.

---

## Deliverables for Step 2

To complete Step 2, the following must exist:

1. `march_mania/data/datasets.py`
2. a dedicated test module for dataset contracts
3. passing tests proving:
   - immutability
   - no eval labels in eval inputs
   - correct shape validation
   - read-only array enforcement
   - no accidental contract overlap

---

## Acceptance Criteria (Must Pass Before Advancing)

Before Step 2 is considered complete, all of the following must be proven via `pytest`:

1. Attempting to access `eval_data.y` raises `AttributeError`
2. Attempting to mutate any dataset array raises a NumPy/ValueError-style write-protection failure
3. Constructing a dataset with mismatched `X` and `y` row counts fails loudly
4. `EvalDataset` cannot be instantiated with label-bearing fields
5. `EvalLabels` cannot be confused with `EvalDataset`
6. All contract tests pass without external files
7. No test relies on future pipeline modules that do not yet exist

---

## Phase Gate to Advance to Step 3

### Gate Name
**Phase 1 / Step 2 Contract Integrity Gate**

### Required Evidence
- full dataset contract test suite passes
- no mutable-array loophole exists
- evaluation labels are physically separated from evaluation inputs
- no broad generic dataset container has been introduced

### Promotion Rule
Step 3 may begin only after Step 2 passes with:
- all tests green
- no architectural contradictions
- no temporary shortcuts left behind

---

## Anticipated Next Step

## Phase 1, Step 3 — Feature Engineering Substrate

Planned focus:
- safe feature-layer scaffolding
- pregame-only feature boundaries
- matchup construction policy
- feature provenance discipline

This step must only begin after the typed dataset boundaries are locked.

---

## Common Failure Modes to Watch For

The following are the most likely implementation mistakes during Step 2:

1. Adding `.y` to `EvalDataset` “for symmetry”
2. Using mutable arrays and relying on convention instead of enforcement
3. Creating a generic dataset superclass that weakens type distinctions
4. Allowing optional labels on all dataset types
5. Writing tests that only check happy paths
6. Building dataset loaders or feature code prematurely
7. Using loose typing (`Any`, raw dicts, generic blobs) instead of explicit contracts

---

## Change-Control Rule for This Step

Any proposal to:
- relax Day 133
- merge dataset types
- add labels to evaluation inputs
- remove immutability protections
- broaden the contract interfaces

must be treated as an architectural change request, not a local implementation choice.

---

## Recommended Verification Commands

Example commands for this step:

```bash
pytest march_mania/tests/test_tier1_foundation.py -q
pytest march_mania/tests/test_datasets.py -q
pytest -q