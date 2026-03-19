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
- the **common failure modes** most likely to cause configuration drift, irreproducibility, stale experiment state, or accidental architecture creep

This document is subordinate only to:
1. `MASTER_ARCHITECTURE_V2.md`
2. the canonical decision ledger / memory capsule

If a coding task conflicts with this phase plan, the task must be narrowed, deferred, or escalated.

---

## Current Program State

**Current Project Status:** Phase 2 — Tracking & Reproducibility  
**Current Active Step:** Step 1 — Pipeline Configuration & Hashing

**Primary Objective of Phase 2:**  
Build the reproducibility and lineage substrate required before model training, calibration fitting, router experimentation, or cross-fold experiment management are allowed.

**Success Condition for Phase 2:**  
Every model run, calibration run, and evaluation artifact can eventually be tied back to:
- a frozen pipeline configuration
- a deterministic configuration hash
- explicit feature flags
- explicit experimental-path enablement
- stable provenance and reproducibility policy

---

## Source-of-Truth Assumptions in Force

The following decisions are active architectural law for this step:

1. **Immutability Law**
   - Core configuration must be immutable
   - No runtime mutation of pipeline settings is allowed once a config is constructed

2. **Determinism Law**
   - Configuration hashing must be deterministic
   - Equivalent configurations must yield identical hashes
   - Distinct configurations must yield distinct hashes

3. **No Hidden State Law**
   - Configuration must not depend on ambient environment variables, implicit defaults, mutable globals, or hidden runtime state

4. **Explicit Experimental-Path Law**
   - Experimental components must be opt-in through explicit feature flags
   - Experimental flags must be part of the config state and part of the hash

5. **No Architecture Creep Law**
   - Step 1 builds configuration and hashing only
   - It must not drift into model implementation, calibration logic, tracking transport, manifests, or experiment running

6. **Reproducibility Law**
   - Configuration must be serializable into a stable representation suitable for hashing and later manifest inclusion
   - Hashing must reflect the true logical config state, not incidental object identity

---

## Recently Completed

## Phase 1 — Foundation
Completed and green:

### Step 1 — Temporal Skeleton
Built:
- `fold_context.py`
- `cutoff_policy.py`
- `leakage_guard.py`

### Step 2 — Data Contracts
Built:
- `TrainDataset`
- `CalDataset`
- `EvalDataset`
- `EvalLabels`

### Step 3 — Feature Engineering
Built:
- `RollingFeatureStore`
- `MasseyOrdinalExtractor`
- `FeatureAssembler`

### Step 4 — Materialization
Built:
- `RawTableLoader`
- `DatasetMaterializer`

Outcome:
- stateless, leak-safe data factory is complete
- all-pairs Zone C generation exists
- type firewall is intact
- provenance stamping is present at the dataset layer

**Current test status:** **221/221 tests green**

---

## Current Objective

## Phase 2, Step 1 — Pipeline Configuration & Hashing

### Target File
1. `ncaa_pipeline/context/pipeline_config.py`

### Primary Goal
Create the single canonical, immutable configuration object that will govern:
- baseline model hyperparameters
- experimental model toggles
- calibration bounds
- feature-path toggles
- deterministic run identity through a stable `config_hash()`

This step is about **configuration discipline**, not model execution.

---

## Step 1 Detailed Requirements

### A. `PipelineConfig`
Create a single immutable configuration contract that centralizes the pipeline’s tunable and switchable behavior.

#### Required design intent
- frozen
- explicit
- hashable through a stable logical representation
- easy to serialize
- easy to inspect
- safe to include in manifests later
- hostile to silent default drift

#### Required structural properties
- must use `@dataclass(frozen=True)`
- should use `slots=True` if consistent with current codebase style
- must contain only explicit fields
- must not rely on nested mutable dicts/lists unless converted to immutable equivalents
- must not contain file handles, model objects, DataFrames, or runtime-only state

---

### B. Required Configuration Domains

At minimum, `PipelineConfig` must centralize the following:

#### 1. Baseline model config
For the baseline LightGBM path, include explicit defaults such as:
- objective = `"binary"`
- max_delta_step = `1.0`

If additional baseline-safe parameters are included, they must be:
- explicit
- phase-appropriate
- stable under hashing

Do **not** overbuild the full model grid-search universe here.

#### 2. Calibration config
Include the bounds and settings needed for later calibration phases, at minimum:
- `a` bounds: `0.3` to `3.0`
- `b` bounds: `-2.0` to `2.0`

These are config values only.
This step must **not** implement calibration fitting.

#### 3. Feature-path flags
Include explicit flags for experimental feature/model paths, including:
- `enable_tabpfn`
- `enable_graph_features`

Optional additional flags are acceptable if they clearly correspond to already-known architecture decisions, but do not bloat this object with speculative future flags.

#### 4. Reproducibility metadata fields
If this step includes fields like:
- global random seed
- deterministic mode
- feature family toggles

they must be:
- explicit
- stable
- included in hashing

---

## Required Hashing Behavior

### `config_hash()`
`PipelineConfig` must implement a method:

- `config_hash() -> str`

#### Required semantics
- returns a deterministic SHA256 hex string
- reflects the full logical state of the configuration
- does not depend on object memory address, field insertion accidents, or runtime ordering quirks
- must be stable across repeated calls
- must change if any meaningful configuration value changes

#### Approved logical strategy
The hash should be computed from a deterministic serialized representation of the config state, such as:
- sorted-key JSON built from immutable primitive values
- or an equivalent stable canonical encoding

#### Must not do
- no use of Python’s built-in `hash()`
- no unstable stringification of objects
- no omission of experimental flags
- no omission of nested config state if nested sub-config dataclasses are used

---

## In Scope Right Now

Only the following are in scope for Step 1:

- immutable pipeline configuration design
- deterministic config hashing
- configuration serialization helpers if needed
- tests proving hashing and immutability behavior
- explicit defaults for baseline model path, calibration bounds, and experimental flags

---

## Explicitly Out of Scope Right Now

The following must **not** be built yet:

### Model Code
- `LightGBMBinaryCore`
- `TabPFNCore`

### Calibration Fitting
- `NelderMeadCalibrator`
- any Platt / affine-logit fitting implementation
- any optimizer wrappers

### Routing / Ensembling
- `SimpleAverageRouter`
- `PerformanceWeightedRouter`
- any ensemble weighting logic

### Tracking Transport
- `AimMLflowTracker`

### Manifest / Lineage Persistence
- `RunManifest`

### Experiment Lifecycle
- `ExperimentRunner`
- bootstrap CI logic
- cross-fold aggregation

If any of these seem “helpful” to complete Step 1, treat that as drift unless explicitly approved.

---

## Deliverables for Step 1

To complete Step 1, the following must exist:

1. `ncaa_pipeline/context/pipeline_config.py`
2. one or more dedicated tests for configuration immutability and hashing behavior

At minimum, tests must prove:
- config is frozen
- config hash is stable
- changing a meaningful field changes the hash
- equivalent configs produce the same hash
- experimental flags participate in the hash
- serialization is deterministic

---

## Acceptance Criteria (Must Pass Before Advancing)

Before Step 1 is considered complete, all of the following must be proven via `pytest`:

1. `PipelineConfig` is immutable after instantiation
2. `config_hash()` returns a deterministic SHA256 string
3. repeated calls to `config_hash()` on the same object return the same value
4. two logically identical configs produce the same hash
5. changing any meaningful hyperparameter changes the hash
6. changing `enable_tabpfn` changes the hash
7. changing `enable_graph_features` changes the hash
8. calibration bounds are part of the hash
9. config serialization does not depend on field ordering accidents
10. no Step 1 code imports model, calibration, routing, tracking, or manifest modules
11. tests run without network calls and without depending on future-phase code

---

## Phase 2 Milestones

### Step 1 — Pipeline Configuration & Hashing
Build:
- `PipelineConfig`
- deterministic `config_hash()`
- config tests

### Step 2 — Experiment Tracking
Build:
- `AimMLflowTracker`
- write-only tracking interface
- no hidden reads of eval labels or future fold state

### Step 3 — Run Lineage
Build:
- `RunManifest`
- ability to anchor outputs to config hash, environment, code version, and data provenance

### Step 4 — Cross-Fold Aggregation
Build:
- `ExperimentRunner`
- fold lifecycle management
- aggregate metric computation
- 95% bootstrap confidence intervals

These are future milestones only.
They must not be started in Step 1.

---

## Phase Gate to Advance to Step 2

### Gate Name
**Phase 2 / Step 1 Configuration Integrity Gate**

### Required Evidence
- all config tests pass
- hash behavior is deterministic
- no mutable config loopholes exist
- experimental flags are explicit and hashed
- no hidden architecture creep into model/tracking code

### Promotion Rule
Step 2 may begin only after Step 1 passes with:
- all tests green
- no unresolved architectural contradictions
- no ambiguous hashing behavior
- no temporary shortcuts in config serialization

---

## Common Failure Modes to Watch For

The following are the most likely implementation mistakes during Step 1:

1. using Python’s built-in `hash()` instead of deterministic SHA256
2. including mutable containers directly in config state
3. forgetting to include feature flags in the hash
4. forgetting to include calibration bounds in the hash
5. creating a config object that depends on ambient environment or hidden defaults
6. building a giant speculative mega-config for future phases
7. allowing semantically identical configs to serialize differently
8. mixing runtime state into configuration state
9. importing model or tracking code into the config layer
10. treating “config exists” as equivalent to “reproducibility is solved”

---

## Change-Control Rule for This Step

Any proposal to:
- make config mutable
- allow hidden defaults
- exclude experimental flags from the hash
- exclude calibration bounds from the hash
- place runtime objects inside the config
- let config hashing depend on nondeterministic serialization
- broaden Step 1 into model/tracker/manifest work

must be treated as an architectural change request, not a local implementation choice.

---

## Recommended Verification Commands

Example commands for this step:

```bash
pytest tests/test_pipeline_config.py -q
pytest -q