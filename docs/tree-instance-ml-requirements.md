# Tree Instance ML: ML0 Requirement Freeze

## Document status

This document freezes requirements for ML0 planning only.  It does **not**
authorize model, training, inference, adapter, artifact, dataset-conversion,
dependency, or environment implementation.

**Verified repository facts** are labelled as such.  Everything labelled
**Proposed design** requires implementation review before it becomes code.

## 1. Scope and non-goals

### In scope

The future individual-tree ML module is limited to:

- point semantic prediction: `TREE`, `SHRUB`, or `NON_TREE`;
- individual-tree instance segmentation;
- point-level semantic and instance confidence;
- instance-level confidence;
- a future, validated conversion boundary to the project's standard tree
  segmentation result; and
- reproducible future ONNX export and inference.

### Explicit non-goals

The module must not modify, call as a decision source, or redefine:

- wire detection, separation, or reconstruction;
- final tree-to-wire clearance calculation;
- growth prediction;
- risk grading or warning logic; or
- any existing prediction-computer module.

In particular, ML confidence is not a clearance or safety decision.

## 2. Verified public loader and input contract

**Verified fact.** The public loader is
[`algorithms/vegetation/tree_inputs.py`](../algorithms/vegetation/tree_inputs.py),
symbol `load_tree_segmentation_input_set` (line 720):

```python
load_tree_segmentation_input_set(
    v2_npz_path: Path | str,
    v2_json_path: Path | str,
    a2a_npz_path: Path | str,
    a2a_json_path: Path | str,
    a2b_npz_path: Path | str,
    a2b_json_path: Path | str,
) -> TreeSegmentationInputSet
```

It validates and aligns V2 vegetation instances, A2A geometry features, and
A2B height-above-ground artifacts before returning the immutable runtime view.
The immutable input contract is
[`core/contracts/tree_inputs.py`](../core/contracts/tree_inputs.py),
symbol `TreeSegmentationInputSet` (line 215).

### Fields available to a future ML module

**Verified fact.** `TreeSegmentationInputSet` exposes:

- `vegetation_points`: V2 point-aligned data, including `original_index`,
  `sample_index`, `xyz_local`, vegetation scores/confidence, existing cluster
  membership/status arrays, and provenance/quality flags.  See
  [`core/contracts/vegetation_instances.py`](../core/contracts/vegetation_instances.py),
  `VegetationInstancePointSet`.
- `geometry_features`: A2A point-aligned geometry: `xyz_local`,
  `origin_xyz`, `original_index`, `sample_index`, `linearity`, `planarity`,
  `scattering`, `verticality`, `local_density`, `height_m`,
  `neighbor_count`, `eigenvalues`, `principal_direction`, `normal`, and
  `quality_flags`.  See
  [`core/contracts/features.py`](../core/contracts/features.py),
  `PointFeatureSet`.
- `heights_above_ground`: A2B point-aligned fields `original_index`,
  `sample_index`, `ground_z_m`, `raw_height_agl_m`, `height_agl_m`,
  `ground_cell_id`, `ground_distance_m`, and `quality_flags`.  See
  [`core/contracts/ground.py`](../core/contracts/ground.py),
  `HeightAboveGroundSet`.
- cluster-level membership/provenance: `cluster_id`, `cluster_status`,
  `cluster_vegetation_id`, `cluster_member_offsets`,
  `ordered_cluster_member_row_index`, cluster partitions/dispositions,
  `ground_usable_by_point`, and `usable_ground_member_count_by_cluster`.
- `identity`: `TreeSegmentationInputIdentity`, containing source/manifest
  identity, origin, point count, upstream algorithm versions, and input
  SHA-256 provenance.

### Prohibited or leakage-prone features

The following are prohibited as ML predictive features in ML0/ML1 unless an
explicit design review changes this document:

- `wire_id_by_point`, wire geometry, wire reconstruction outputs, clearance,
  or any risk/warning output; this prevents coupling the tree model to wire
  logic.
- `vegetation_id_by_point`, `distance_ready_by_point`,
  `tree_id_by_point`, `tree_distance_ready_by_point`, final status labels, or
  any future manual disposition used as a target or feature; these are formal
  decisions and create direct leakage.
- `original_index`, `sample_index`, source IDs, manifest IDs, spatial tile
  names, output paths, or annotation split identifiers as predictive features.
  They are traceability fields only.
- data from a train/validation/test split outside the active split, including
  duplicate crops or overlapping point clouds.

Existing V2 vegetation scores, confidence, automatic/manual levels,
assignment confidence, and assignment flags are also leakage-prone.  They may
be retained solely for provenance, diagnostics, or an explicitly reviewed
gating policy; they are not default model features or labels.

## 3. Proposed ML-only immutable prediction sidecar

**Proposed design.** Introduce a new ML-only sidecar contract rather than
changing the existing standard result during ML0.  A suggested name is
`TreeInstanceMlPrediction`.  It must be point-row aligned with exactly one
`TreeSegmentationInputSet` and must not imply clearance readiness.

### Required sidecar fields

| Field | Proposed type/shape | Requirement |
| --- | --- | --- |
| `semantic_class_by_point` | `uint8[N]` | Enum values: `NON_TREE=0`, `SHRUB=1`, `TREE=2`. |
| `semantic_confidence_by_point` | `float32[N]` | Confidence of the selected semantic class. |
| `segment_id_by_point` | `int32[N]` | `-1` for no retained instance; otherwise continuous segment IDs. |
| `instance_confidence_by_point` | `float32[N]` | Confidence of the assigned instance; `0.0` when segment ID is `-1`. |
| `instance_confidence_by_segment` | `float32[S]` | One confidence per continuous segment ID. |
| `instance_member_offsets` | `int64[S + 1]` | CSR offsets, beginning at zero and ending at the member-row count. |
| `ordered_instance_member_row_index` | `int64[A]` | Strictly increasing rows within each CSR segment; exact members of `segment_id_by_point`. |
| `source_identity` | `TreeSegmentationInputIdentity` | Exact input identity/provenance. |
| `original_index_by_point` | `int64[N]` | Byte-for-byte equal to the loaded V2 input field. |
| `sample_index_by_point` | `int64[N]` | Byte-for-byte equal to the loaded V2 input field. |
| `model_version` | non-empty string | Human/version-control identifier, not a path. |
| `model_sha256` | lowercase SHA-256 string | Digest of the exact deployed model file. |
| `preprocessing_signature` | immutable JSON-safe mapping | Canonical, versioned description of preprocessing. |

`N` is `source_identity.point_count`; `S` is the number of retained instance
segments; and `A` is the total assigned point count.  A future implementation
may add explicit model input/output tensor metadata, but must not add
unreviewed object-dtype arrays or paths to the contract.

## 4. Alignment and index preservation

1. The sidecar point arrays have exactly `N` rows in the original input row
   order; no implicit reordering, sorting, compacting, or deduplication is
   permitted.
2. `original_index_by_point` and `sample_index_by_point` must exactly equal
   `tree_input.vegetation_points.original_index` and `.sample_index` in both
   values and order.
3. Every point must receive exactly one semantic enum value: `TREE`, `SHRUB`,
   or `NON_TREE`, and one bounded semantic-confidence value.
4. There is no `REVIEW_REQUIRED`, `PROVISIONAL`, or other "reviewed"
   point-semantic class.
5. `segment_id_by_point == -1` means only that a point belongs to no retained
   tree instance; it does not define or replace that point's semantic class.
6. Review or provisional status belongs to an instance or to the later
   standard-result adapter, not to the point semantic enum.
7. Every crop, voxelization, downsampling, batching, padding, augmentation,
   tiling, overlap merge, and inverse transform must carry an explicit
   source-row mapping and restore output for every original input row.
8. Coordinate equality is never an acceptable substitute for a source-row
   mapping; duplicate XYZ coordinates can occur.
9. If inference fails to produce a valid semantic class and confidence for all
   `N` rows, validation must reject the ML prediction artifact rather than
   invent an undefined semantic state or silently omit rows.
10. The CSR arrays and `segment_id_by_point` must describe the same membership
    relation exactly.  No row may occur in two instance segments.

中文说明：所有预处理步骤都必须保留“模型行 -> 原始输入行”的显式映射；禁止仅凭坐标回填索引。

## 5. Array, confidence, ownership, and mutability rules

**Proposed design.** All sidecar arrays must be NumPy ndarrays with owned,
C-contiguous storage and `writeable=False`.  Views, object dtype, ragged
arrays, silently coerced dtypes, and mutable shared buffers are invalid.

- Point arrays use exact shapes above; no `(N, 1)` substitutes.
- Confidence arrays are finite `float32` values in the closed interval
  `[0.0, 1.0]`.
- Semantic class values are exactly the three proposed `uint8` enum values.
- Segment IDs are `int32`, are either `-1` or in `[0, S)`, and are continuous
  over retained segments.
- CSR offsets and member rows are `int64`; offsets are monotonic and member
  rows are valid input-row positions.
- SHA-256 values are lowercase 64-character hexadecimal strings.
- The sidecar must retain immutable source identity and canonicalized,
  JSON-safe preprocessing metadata.

## 6. Safety status rules

**Verified fact.** The existing standard result types are
`TreeInstanceRecord` (line 400), `TreeSegmentationStatistics` (line 540), and
`TreeSegmentationResult` (line 613) in
[`core/contracts/vegetation_trees.py`](../core/contracts/vegetation_trees.py).
They provide instance confidence but no point semantic or instance confidence
arrays.

**Verified fact.** Input V2 `REVIEW_REQUIRED` clusters are mapped to
`REVIEW_ONLY` in
[`algorithms/vegetation/tree_inputs.py`](../algorithms/vegetation/tree_inputs.py),
symbol `_STATUS_DISPOSITIONS`.  Existing standard contracts require confirmed
and distance-ready source clusters for confirmed trees.

Therefore:

- An ML-only prediction never creates `CONFIRMED_TREE` or a formal `tree_id`.
- Under the current standard result contract, a source V2 cluster whose status
  is `REVIEW_REQUIRED` or `PROVISIONAL` cannot be promoted by an ML adapter to
  a `CONFIRMED_TREE` or `tree_distance_ready` tree.
- Under the current contract, confirmed records require the existing formal
  source requirements, including V2 `CONFIRMED` source status and source
  readiness.
- Model predictions, semantic confidence, point confidence, instance
  confidence, clustering quality, and adapter calculations are not substitutes
  for source confirmation.
- Low confidence, incomplete mapping, unsupported preprocessing, missing
  geometry/HAG, cross-cluster ambiguity, or conflicting evidence must remain
  `REVIEW_REQUIRED`, `PROVISIONAL`, `REJECTED`, or `UNRESOLVED` as appropriate;
  they must not be promoted by confidence.
- ML confidence, whether point or instance level, cannot independently set
  `tree_distance_ready` or `tree_distance_ready_by_point`.
- Any future policy permitting different promotion behavior requires an
  explicitly versioned contract change, separate technical review, tests, and
  authorization.  It must not be introduced silently by the ML adapter.

## 7. Standard-result adapter boundary

**Verified fact.** `TreeSegmentationResult` accepts only the current schema
and algorithm versions defined in
[`core/contracts/vegetation_trees.py`](../core/contracts/vegetation_trees.py):
`tree-segmentation-v1` and `v3-single-tree-segmentation-v1`.

**Requirement.** ML0 must not construct or publish a standard result under
the existing V3 algorithm version.  That would misrepresent ML provenance.
A later, separately reviewed standard-result adapter must first define a
versioned compatibility strategy, preserve all existing invariants, derive
required geometry/crown summaries, populate source-cluster provenance, and
have an approved algorithm-version/schema policy.  It must remain outside the
wire, clearance, growth, risk, warning, and prediction-computer paths.

## 8. ONNX deployment and reproducibility

**Proposed design.** ONNX is an ML1 concern, not an ML0 implementation task.
The future deployment specification must pin and record:

- model SHA-256, `model_version`, ONNX opset, model input/output names,
  shapes, dtypes, and semantic-class order;
- preprocessing signature, coordinate frame, units, normalization, crop/voxel
  policy, padding, post-processing, and confidence thresholds;
- runtime/provider/version, requested execution provider, CPU/GPU policy, and
  deterministic fallback behaviour;
- fixed golden input artifacts and expected immutable sidecar digests; and
- no-network, no-training inference behaviour with bounded resource limits.

ONNX export must be reproducible from a separately controlled training
environment.  Dynamic axes may be used only where the input/output mapping is
fully specified and covered by golden tests.

## 9. Files that remain outside Git

**Verified fact.** Current [`.gitignore`](../.gitignore) already excludes
`data/raw/`, `data/samples/`, `data/processed/`, `outputs/`, `cache/`,
`projects/`, `models/weights/`, point-cloud files, and `.pt`, `.pth`, and
`.ckpt` files.

**Requirement for future authorization.** Keep outside Git:

- raw/licensed scans, labels, annotation exports, split manifests containing
  sensitive paths, and derived training tiles/tensors;
- training checkpoints, optimizer state, exported `.onnx` models, quantized
  variants, TensorBoard/event files, profiling traces, and experiment logs;
- voxel/crop/inference caches, benchmark outputs, temporary visualization
  artifacts, and model registry downloads.

Before ML assets are introduced, a separately authorized `.gitignore` review
must cover `datasets/`, checkpoint/export folders, `.onnx`, `.npy`, `.npz`,
and logs.  ML0 does not modify ignore rules.

## 10. Acceptance criteria

### ML0 acceptance criteria

- This requirements document has explicit review approval.
- A future sidecar contract implements every required field and invariant in
  sections 3–5, with no ML dependency required.
- Tests prove exact row/index preservation, immutable ownership, bounds,
  CSR reconciliation, serialization round trip, invalid-input rejection, and
  no mutation of source arrays.
- No model, training, inference adapter, dataset conversion, environment, or
  package installation is introduced.

### ML1 acceptance criteria

- ML0 contract and tests are accepted first.
- ONNX preprocessing/inference is reproducible by golden tests and records
  all deployment metadata in section 8.
- A separately reviewed standard-result adapter passes all existing result
  invariants and a new ML provenance/version test suite.
- No ML output independently creates distance readiness or changes excluded
  wire/clearance/growth/risk/warning/prediction-computer behavior.

## 11. Open decisions requiring explicit review

1. Semantic taxonomy: whether `SHRUB` may have instances, or only `TREE` may
   be assigned segment IDs.
2. The exact semantic/instance confidence calibration method and approved
   thresholds; no default threshold is approved here.
3. Training labels, licensing, split policy, and treatment of mixed,
   occluded, or unassigned points.
4. Crop/voxel/downsampling policy, resource limits, and required CPU fallback.
5. Whether point confidence belongs in a future version of the standard result
   or remains a signed sidecar artifact.
6. The schema/algorithm-version evolution and backward-compatibility policy
   for ML-to-standard-result conversion.
7. Conditions, if any, under which an ML-derived instance can be confirmed;
   ML confidence alone is explicitly insufficient.
8. ONNX Runtime package/version, hardware provider policy, and golden-scene
   distribution policy.
