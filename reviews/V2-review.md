# V2 schema-v2 implementation, safety, sensitivity, and profile review

## Scope and published contract

- Baseline verified: branch `feature/algorithm`, HEAD `e63e011e3d9b3938077228ecf908e6a410bbc0cc`.
- Schema: `vegetation-instances-v2`.
- Algorithm: `v2-generalized-morphology-safe-vegetation-instances-v2`.
- Modified scope is limited to the seven approved V2 files: `core/contracts/vegetation_instances.py`, `core/contracts/__init__.py`, `algorithms/vegetation/instances.py`, `algorithms/vegetation/__init__.py`, `tests/contracts/test_vegetation_instance_contracts.py`, `tests/vegetation/test_instances.py`, and this review. No staging or commit was performed.

V2 accepts only verified V1/A2A/A2B NPZ/JSON sibling pairs. It does not read LAS/LAZ, A3, or raw point records. Schema-v1 is rejected without migration. Publication is NPZ followed by JSON as the completion marker, and an existing ignored V2 result is replaceable only with `overwrite=True`.

## Implemented grouping and safe-split evidence

Local-scale support accepts VALID, CLAMPED_LOW, and CLAMPED_HIGH; `q_scale` weights them `1.00`, `0.85`, and `0.70`. The atomic support check now requires core count/level/voxel/extent support, median core degree >= 2, median shared-primary support >= 2, `q_scale >= 0.70`, and grouping score >= 0.55. Automatic confirmation separately requires grouping score >= 0.65.

Core neighbours use deterministic normalized-distance/original-index ordering, strict mutual primary edges first, and reciprocal secondary edges subject to scale, HAG, voxel, shared-neighbour, and two-per-endpoint limits. On the regenerated development scene the core graph had 174,618 edges: 174,618 strict and 0 selected secondary edges; degree median/p95/max was 9/12/12 and shared-primary median was 5.

Safe splitting is fixed-precedence and stale-candidate-free: quotient bridge, then weak/empty six-slice vertical gap, then qualified mixed morphology. Quotient candidates use the frozen priority tuple `(-category_count, -evidence, -max_normalized_cross_edge, cross_edge_count, min_original_index, max_original_index)`. Evidence is bounded `[0,1]` from edge length, weak cross support, shared-neighbour deficit, score/confidence valley, HAG discontinuity, and geometry incompatibility. Only the highest candidate is attempted; failure marks the branch unresolved. Accepted children are independently recomputed, depth is capped at 3 and automatic children at 8. Gap cuts remove below-above edges and only weak below-gap/gap-above transitions; retained children must support both sides. Missing crown/surface regions set both supports, separation, and morphology-change fields to zero, so they cannot define a mixed split. Unsupported fragments remain unassigned. LOW points are discarded from old assignments and expanded once, from final retained cores only. No automatic merge exists.

All final clusters have six fixed equal-Z slice slots and PCA-aligned voxel fields. Slice floats are NaN when invalid; counts are zero and validity false. Planar, isolated-slender, compact-layered, mixed-morphology, grouping-degraded, morphology-degraded, and unresolved-topology evidence is carried to status. Only final `CONFIRMED` points have a vegetation ID and `distance_ready=true`; automatic rejection remains disabled.

## Deterministic synthetic and safe-split validation

Focused V2 validation contains 74 node IDs: 54 explicit parameterized crown cases (`3 scales x 3 rotations x 2 densities x 3 HAG modes`), 3 parameterized artificial-family cases (wall, pole, compact), plus 17 contract/grouping/publication/manual/safe-invariant cases. Fixtures are seeded and position-dependent rather than constant perfect A2A fields.

| safety family or mechanism | asserted result |
| --- | --- |
| dense clear crown | confirmed when reliable HAG is available |
| compact crown | accepted, never silently unassigned by CLAMPED_HIGH alone |
| scale/rotation/HAG crown matrix | final readiness always equals final confirmation |
| wall, pole, compact-layered object | `REVIEW_REQUIRED`; never automatic distance-ready |
| local-scale ties, secondary caps, LOW-only bridge prevention | deterministic and conservative |
| schema/reload/manual precedence | v1 rejection, standalone reload, and override provenance retained |
| bridge/gap/mixed eligibility | deterministic quotient priority tuple and bounded score, two-side vertical weak-gap cut, and exact missing-region zero values passed |

The parameterized crown matrix passed all 54 cases; the artificial matrix passed all 3 cases. `pytest tests/vegetation/test_instances.py tests/contracts/test_vegetation_instance_contracts.py -q` passed 74 tests.

## Full sensitivity matrix

The current implementation was exercised across 17 key threshold groups, five central/outer settings each, and three independent fixture outcomes per setting: 85 settings and 255 fixture evaluations. Continuous groups used `-20%, -10%, default, +10%, +20%`; discrete groups used the central value and two bounded neighbours on each side.

| groups | result over all five settings |
| --- | --- |
| core/vertical radius, secondary normalized distance, grouping support, bridge normalized length | compact remained assigned; wall never distance-ready |
| promotion/review morphology and confidence | clear crown confirmed at default and +/-10%; wall never distance-ready |
| primary/expanded caps, secondary shared/cap, bridge shared, weak-gap degree/voxel support, slice minimum | clear crown confirmed at default and +/-10%; compact remained assigned; wall never distance-ready |

There were zero central (`default +/-10%`) crown-confirmation failures, zero compact-unassigned failures, and zero artificial-distance-ready failures. The only outer change was `core_radius_multiplier=1.12` (-20%), where the clear crown changed conservatively from `CONFIRMED` to `REVIEW_REQUIRED`; it was not made distance-ready by an unsafe rule. No sensitivity setting is based on a vehicle, real coordinate, original index, object dimensions, or cluster ID.

## Regenerated `scene_simple` schema-v2 result

Verified input SHA-256: V1 NPZ `473acca89891a2ddd2ed5896515990ab4db5d9e8b984ff92821f2d28d33cbb65`; V1 JSON `8b08431834c71af9bfc42c904fa10d98da66644ea3392bbc8738410689057fdd`; A2A NPZ `8c13802e8aafa83a5e0c0f04318c40d1ac3cff8cb4e7d98713c63654d11a74d4`; A2A JSON `d21062adcb92abc0a15bce56e409f42c83369373f3661798f4e46a19eb76ebe0`; A2B NPZ `1dde7cba2eb0d5040a6bbc3844fcc1be735b2cd09bdbd8c50176065d687072ec`; A2B JSON `8bce9f6c9f9404ac47b25d66cfda2dface181944d925fb03120ba19f5986e8cf`.

| artifact | bytes | SHA-256 |
| --- | ---: | --- |
| `outputs/vegetation_instances/scene_simple_vegetation_instances.npz` | 3,491,262 | `95d40327417bcdb6f760e97890a71bfabe3b49d85949d2104fc75518a48a9772` |
| `outputs/vegetation_instances/scene_simple_vegetation_instances.json` | 1,856,553 | `c600bf8b672a0edcf9efcbdd84c28c9109ccffd4788ce8daef88f18764f26e06` |
| `outputs/previews/scene_simple_vegetation_instances_preview.html` | 5,618,734 | `3b2d4c982aadc0515aa8f0cf2cfab58242544a61fb7727274ffbecc4957f635e` |

| metric | result |
| --- | ---: |
| input / candidates / core | 146,126 / 64,475 / 41,872 |
| final clusters | 243 |
| confirmed / review / provisional / rejected | 15 / 213 / 15 / 0 |
| assigned / unassigned candidate points | 50,313 / 14,162 |
| local scale unavailable / valid / low-clamped / high-clamped | 82,054 / 61,137 / 66 / 2,869 |
| safe bridge / gap / mixed / unresolved branches | 15 / 1 / 0 / 111 |
| status points unassigned / confirmed / review / provisional | 95,813 / 2,124 / 47,644 / 545 |
| all-six-valid-slice clusters | 63 of 243 |

Cluster score min/median/max: grouping `0.548810/0.891667/1.000000`; vegetation morphology `0.224013/0.503422/0.914420`; planar ambiguity `0.018067/0.233971/0.391944`; isolated-slender `0.000000/0.006714/0.966054`; compact-layered `0.180920/0.346774/0.646396`; mixed conflict `0/0/0`. Thresholded hard ambiguity counts were planar 0, isolated-slender 9, compact-layered 2, and mixed 0. Valid-slice histogram for 0..6 valid slices was `110, 4, 6, 17, 18, 25, 63`.

The generated standalone preview was regenerated from the final sibling pair and its artifact hash was verified. Publication/reload validation uses the V2 loader and therefore verifies the JSON completion marker, NPZ sibling hash, arrays, schema, and readiness invariant; preview visuals are not used as evidence to promote a cluster.

## Profile, regression observations, and verification

Final end-to-end elapsed time was **12.2053 s**; RSS start/peak/end was `65,994,752 / 184,471,552 / 184,471,552` bytes. Measured phase timings were input validation 0.1636 s, local scale 1.0269 s, core graph plus safe splits 8.1570 s, one-pass LOW reassignment 2.1014 s, and descriptors/status 0.6765 s (grouping total 12.0416 s). The bounded nearest-neighbour core candidate set removes the former repeated radius-query profile hotspot; no dense N-by-N matrix is used.

Post-run-only regression observations: original index `302562` is cluster 3, `REVIEW_REQUIRED`, vegetation ID -1, and not distance-ready. Original index `1324416` is cluster 75, `REVIEW_REQUIRED`, vegetation ID -1, and not distance-ready. Neither index, coordinate, object size, nor a real cluster identifier occurs in implementation, configuration, or fixture rules.

Final checks: full pytest suite `264 passed`; focused V2 suite `74 passed`; changed-file `ruff check` passed; `compileall core algorithms tests` passed. Repository-wide formatter checking is reported separately because it has pre-existing unrelated findings outside the approved V2 scope. Generated artifacts remain ignored; no tracked runtime output, LAS/LAZ, or A3 artifact was added.

V2 is designed for general, scale-aware vegetation-instance grouping and artificial-structure ambiguity assessment. It is validated using diverse deterministic synthetic object families and one real development scene. Multi-scene real-world generalization is not proven and requires additional labelled scenes.
