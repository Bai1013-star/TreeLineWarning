# Tree Instance ML: Reviewable Session Tasks

## Status and guardrails

This is a proposed session sequence, not standing authorization.  No model
training, dependency installation, Conda environment creation, dataset
conversion, or ONNX inference is scheduled by this plan.  Git stage, commit,
and push are independently authorized actions and are not included in any
implementation session below.

Verified starting points:

- Public loader: `algorithms/vegetation/tree_inputs.py`,
  `load_tree_segmentation_input_set(...)`.
- Input contract: `core/contracts/tree_inputs.py`, `TreeSegmentationInputSet`.
- Standard result: `core/contracts/vegetation_trees.py`,
  `TreeInstanceRecord`, `TreeSegmentationStatistics`, and
  `TreeSegmentationResult`.

## Session ML0-0 — requirements review

- **Objective:** Review and approve the ML0 requirements freeze.
- **Allowed files:** `docs/tree-instance-ml-requirements.md`, this task file,
  and `hardware_and_environment_report.md`.
- **Prohibited actions:** Source-code changes; dependency/environment changes;
  training; inference; dataset/model/cache creation; Git staging, commit, or
  push.
- **Expected tests:** Documentation cross-reference review only; no test run
  is required.
- **Acceptance conditions:** Scope, non-goals, index preservation, confidence
  rules, readiness safety rule, ONNX constraints, and open decisions are
  explicitly approved.
- **Stop conditions:** Any request to infer confirmation/readiness from ML
  confidence, consume wire/risk fields, or begin implementation without
  requirement approval.

## Session ML0-1 — ML sidecar contract creation

- **Objective:** Add the immutable ML-only prediction contract and public
  exports, without model runtime code.
- **Allowed files:** Proposed only after approval:
  `core/contracts/tree_instance_ml.py`, `core/contracts/__init__.py`, and
  focused contract tests under `tests/contracts/`.
- **Prohibited actions:** Changes to `TreeSegmentationResult`; adapters;
  ONNX/Torch dependencies; model files; datasets; training; wire/clearance/
  growth/risk/warning/prediction-computer code; Git staging, commit, or push.
- **Expected tests:** Constructor acceptance/rejection, exact dtype/shape,
  confidence bounds, read-only owned C-contiguous arrays, enum validity,
  SHA-256 validation, and immutable metadata.
- **Acceptance conditions:** The contract contains all fields frozen in
  `docs/tree-instance-ml-requirements.md`, and requires no ML package.
- **Stop conditions:** A proposed field has unclear provenance, an object
  dtype is needed, or semantic/instance taxonomy remains unapproved.

## Session ML0-2 — alignment and CSR validators

- **Objective:** Add pure validators for source identity, exact row alignment,
  original/sample index preservation, segment IDs, and CSR membership.
- **Allowed files:** The approved ML sidecar contract and focused tests under
  `tests/contracts/` and/or `tests/vegetation/`.
- **Prohibited actions:** File-path loader changes; preprocessing code;
  model/runtime code; source-array mutation; any readiness calculation; Git
  stage, commit, or push.
- **Expected tests:** Identity mismatch, reordered rows, duplicate/missing
  original index, non-continuous segment IDs, invalid offsets, duplicate CSR
  rows, and exact membership reconciliation.
- **Acceptance conditions:** Validators reject coordinate-only remapping and
  all invalid alignments without copying or mutating the source input.
- **Stop conditions:** Validation requires guessed coordinate matching or
  implicitly changes source row order.

## Session ML0-3 — sidecar serialization and artifact tests

- **Objective:** Specify and test deterministic sidecar serialization, if
  artifact persistence is explicitly approved.
- **Allowed files:** Proposed only after a serialization-format review:
  ML-sidecar artifact contract/I/O files plus focused artifact tests.
- **Prohibited actions:** Standard-result artifact changes; external dataset
  readers; model download; ONNX runtime; training; Git stage, commit, or push.
- **Expected tests:** Canonical metadata, SHA verification, exact array
  inventory/dtype/shape, read-only detached loads, tamper rejection, and
  round-trip equality.
- **Acceptance conditions:** The sidecar is reproducibly loadable and carries
  source/model/preprocessing provenance without any model binary.
- **Stop conditions:** The format would store a model, dataset, cache, or log
  inside the repository, or versioning policy is undecided.

## Session ML1-0 — inference-interface review

- **Objective:** Approve the boundary between deterministic preprocessing,
  future ONNX invocation, and immutable sidecar construction.
- **Allowed files:** Documentation and proposed interface/test files only;
  exact list requires separate approval.
- **Prohibited actions:** Dependency installation; Conda creation; model
  export/download; actual inference; training; standard-result conversion;
  Git stage, commit, or push.
- **Expected tests:** Interface-level fake-runtime tests only, including input
  tensor metadata and explicit source-row mapping propagation.
- **Acceptance conditions:** ONNX opset/tensor names/dtypes, preprocessing
  signature, provider/fallback policy, and golden-test policy are approved.
- **Stop conditions:** Runtime choice, model licensing, GPU policy, or
  preprocessing mapping remains unspecified.

## Session ML1-1 — ONNX inference adapter

- **Objective:** Implement a narrow ONNX adapter that emits only the ML
  sidecar contract.
- **Allowed files:** Separately authorized adapter/config/test files and any
  explicitly approved dependency metadata.
- **Prohibited actions:** Training; checkpoint generation; changes to wire,
  clearance, growth, risk, warning, or prediction-computer modules; direct
  tree readiness assignment; Git stage, commit, or push.
- **Expected tests:** Fake-runtime unit tests, preprocessing determinism,
  tensor shape/dtype checks, index round trip, confidence validation,
  CPU fallback, and golden outputs once approved model fixtures exist.
- **Acceptance conditions:** The adapter records model SHA/version and
  preprocessing signature, has deterministic post-processing, and cannot
  produce standard-result readiness.
- **Stop conditions:** A model asset would be committed, reproducibility data
  is missing, or behavior differs by provider without an approved policy.

## Session ML1-2 — standard-result adapter design review

- **Objective:** Decide the versioned conversion strategy from ML sidecar to
  the existing standard result.
- **Allowed files:** Documentation, contracts, and tests explicitly approved
  for schema/version review.
- **Prohibited actions:** Reusing `v3-single-tree-segmentation-v1` for ML
  provenance; bypassing standard result invariants; readiness promotion from
  confidence; changes to excluded subsystems; Git stage, commit, or push.
- **Expected tests:** Existing `TreeSegmentationResult` compatibility suite,
  proposed version-policy tests, geometry/crown completeness tests, source
  cluster provenance tests, and negative confirmation/readiness tests.
- **Acceptance conditions:** A reviewed schema/algorithm-version policy
  exists and all required point confidence data has a declared home (sidecar
  or versioned standard extension).
- **Stop conditions:** The team cannot state how ML provenance is represented
  truthfully or how current artifact readers remain compatible.

## Session ML1-3 — standard-result adapter implementation

- **Objective:** Implement the approved adapter, deriving formal record
  geometry/crown/provenance while preserving standard safety invariants.
- **Allowed files:** Only the specifically approved standard-contract,
  adapter, export, and test files.
- **Prohibited actions:** Changes to wire/reconstruction, clearance, growth,
  risk/warning, prediction computers, model training, or unapproved package
  changes; Git stage, commit, or push.
- **Expected tests:** Exact row/member reconciliation; status statistics;
  no formal tree ID/readiness for non-confirmed records; review-only source
  behavior; ML confidence cannot promote readiness; provenance/version tests.
- **Acceptance conditions:** The adapter passes all existing and new contract
  tests and is isolated from excluded systems.
- **Stop conditions:** Any test demonstrates an ML-only confidence path to
  `tree_distance_ready` or a source-index mismatch.

## Separately authorized Git actions

The following are not implied by any session above.  Each requires explicit
user authorization after its reviewable change set is accepted:

1. `git stage` selected files.
2. Create a commit with an agreed message.
3. Push the reviewed branch.
4. Open or update a pull request.
