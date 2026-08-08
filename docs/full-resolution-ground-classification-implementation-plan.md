# Full-Resolution Ground Classification Producer Implementation Plan

**Status:** PLAN-FROZEN
**Companion design:** docs/full-resolution-ground-classification-design.md
**Implementation authority:** frozen plan authority for the separate A+B+C implementation handoff; no implementation is performed in this documentation closure.

## 1. Objective and hard scope boundary

Implement FGC-PMF-RLP-v1 exactly as specified by the companion design: a
formal-source-reference-bound, full-resolution, CPU-first producer of an
immutable classified LAS derivative. It must not use A2B, P-HAG output,
W1/P-WIRE, vegetation output, a crop, all-file laspy.read, manual labels, or a
plain source path as authority. P-SOURCE and P-HAG semantic code remain
unchanged.

The run boundary is:

    classify_ground(
        source_reference,
        source_result,
        frv_configuration,
        p_hag_compatibility_config,
        output_root,
        policy_config,
        *,
        progress=None,
        cancellation=None,
    )

The only discoverable output root is:

    <output_root>/ground-classification/<raw-source-id>/<raw-sha-prefix>/
      decisions/
      fgc_decision_manifest.json
      fgc_publication_proof.json
      classified/<source-stem>.fgc-v1.las
      fgc_completion_manifest.json
      diagnostics/

Raw files and sample files are never overwritten. A missing or invalid
fgc_completion_manifest.json means INCOMPLETE_NONAUTHORITATIVE.

## 2. Dependency and production surface

Runtime dependencies are NumPy and laspy plus standard-library hashing,
pathlib, struct, heapq, and canonical JSON helpers. SciPy, GPU, deep learning,
KDTree, ndimage, Delaunay, pickle, object arrays, and global dense rasters are
prohibited. PMF offsets, local hashes, external sort/merge, run-union coverage,
and least squares are internal bounded NumPy/Python implementations.

| Production path | Change | Responsibility |
|---|---|---|
| core/contracts/ground_classification.py | new | exact policy, states, spool/decision/coverage/publication schemas and digest validators |
| core/contracts/__init__.py | modify | narrow FGC contract exports |
| algorithms/ground/__init__.py | new | narrow public producer exports |
| algorithms/ground/decision_store.py | new | uint8 routing, chunk validation, FGCCOV1 sort/merge helpers |
| algorithms/ground/classification.py | new | formal ingress, raw-tick preflight, spool, tiles, PMF/RLP, coverage, orchestration |
| algorithms/ground/visualization.py | new | bounded diagnostics from validated artifacts only |
| algorithms/io/classified_las.py | new | class-only writer, metadata/flag verifier, output identity construction |
| algorithms/io/__init__.py | modify | controlled classified-writer export |
| algorithms/io/samples.py | modify | classified-parent SourceIdentity and parent-index-map binding only; no crop numeric change |
| scripts/run_ground_classification.py | new | explicit formal-input operator command |

Expected production surface: **7 new files plus 3 modified files (10 paths)**.
algorithms/io/source_provider.py, algorithms/vegetation/hag_provider.py,
algorithms/features/ground.py, and raw inputs are out of scope.

## 3. Bundle A — contracts, routing store, and formal authority

**Production:** core/contracts/ground_classification.py,
algorithms/ground/__init__.py, algorithms/ground/decision_store.py, and
core/contracts/__init__.py.

**New tests:** tests/contracts/test_ground_classification_contracts.py,
tests/contracts/test_ground_classification_serialization.py, and
tests/ground/test_decision_store.py.

Implement immutable exact-schema FGCPOL1 policy validation, including all
numerical routing values, raw-grid commensurability rules, resource caps,
FGCSP1, FGCCOV1, decision artifact, publication proof, and
fgc-completion-manifest-v1. P-HAG digest must be excluded from policy and
decision digests. Implement sentinel routing, duplicate-write rejection,
contiguous decision chunks, hash/count validation, fixed-size external coverage
sort runs, deterministic merge/deduplication, and run-union component summaries.

**Test gate:** exact schema/digest/finite JSON; P-HAG digest decoupling; chunk
range overlap/gap/duplicate write; completion-manifest missing/corrupt/path
traversal; orphan LAS non-authority; FGCCOV1 ordering/deduplication/connectivity
and component-width boundaries; resource-preflight formulas.

## 4. Bundle B — formal source, spool, tiling, and classifier

**Production:** algorithms/ground/classification.py.

**New tests:** tests/ground/test_ground_spool_tiling.py,
tests/ground/test_ground_classification.py, and
tests/ground/test_synthetic_ground_scenes.py.

Validate the formal P-SOURCE reference/result/configuration before work. Apply
canonical-rational scale preflight and raw-integer tile/fine/coverage authority.
Reconstruct the source-min raw anchor from the finalized P-SOURCE bounds and
verify it against the actual raw minima during the one-pass FGCSP1 owner spool.
Verify every nonempty owner-tile digest/order/count, enumerate only those tile
IDs in deterministic order, and stream only the 3 x 3 possible neighboring
owner spools for each 20 m halo. Missing neighbor spools mean empty context;
no source replay per tile or world-size empty-tile enumeration is allowed.

Implement the exact 3/5/9/17/33 PMF footprints, validity-mask erosion and
dilation, independent-from-G0 scale recurrence, connected seed support,
lower-envelope median, ordinary least-squares planes, two synchronous clips,
RMS/competing-support predicates, local vertical hash, frozen routing precedence,
and bounded global coverage collection. Batch planes at the configured cap and
fail rather than allocate beyond any bound.

Seed-support tests must prove that D3 excludes the centre while D3_plus includes
it; that the 8-connected component C contains candidate cell c; and that c is
excluded from both the support count and lower-envelope calculation. They must
also prove that exactly five connected support-neighbour cells pass, four fail,
and cells inside D3_plus but disconnected from c do not count.

**Test gate:** PMF footprint/nodata erosion/nodata dilation/recurrence/equality;
halo maximum dependency; raw negative/internal/max boundaries; spool
original-index order, corruption, cap, and cancellation cleanup; seed
support/median/isolated-low; exact plane/tie/clipping/post-clip/RMS/slope/
discontinuity/vertical/routing cases; flat, slope, ditch, embankment, sparse
ground, low shrub/root, low outlier, tower/pole, and no-reliable-ground scenes.

## 5. Bundle C — classified LAS, lineage, and atomic publication

**Production:** algorithms/io/classified_las.py, algorithms/io/__init__.py,
algorithms/ground/classification.py, and algorithms/io/samples.py.

**New tests:** tests/io/test_classified_las.py,
tests/ground/test_ground_publication.py, and
tests/integration/test_ground_p_source_hag.py. Extend the existing
tests/io/test_scene_sample.py for classified-parent lineage; do not create a
second crop implementation.

Stream-copy complete raw records in original order into a same-volume temporary
LAS. Change only laspy's logical base classification field: confirmed to 2,
non-2 inputs preserved, and stale unconfirmed 2 to 1. Lockstep verify raw
integer XYZ, all unrelated point fields, Extra Bytes, every classification flag,
point format, scales/offsets, accepted VLR semantic sequence, expected base
classification, and output SHA. Reject unsupported EVLR/metadata or any
non-round-trip VLR rather than dropping it.

Construct and verify the new classified SourceIdentity, replay through unchanged
P-SOURCE, run the configured P-HAG compatibility replay, and bind all results in
the proof. Publish immutable components first and
fgc_completion_manifest.json last. Implement collision rejection and rollback
tests. Update scene-sample manifest/index-map binding to carry the classified
parent identity, parent SHA/record digest, index-map SHA and parent-index digest,
sample SHA, count, and crop bounds without changing selection geometry.

**Test gate:** all Bundle A/B gates plus LAS point formats, RGB/GPS/intensity/
returns/normals/Extra Bytes, synthetic/key-point/withheld/overlap and scan flags,
VLR preservation, unsupported metadata failure, stale class 2, source replay
mismatch, P-HAG compatibility, atomic failure, corrupt completion manifest, and
sample-parent lineage.

## 6. Bundle D — bounded diagnostics and operator CLI

**Production:** algorithms/ground/visualization.py and
scripts/run_ground_classification.py.

**New tests:** tests/ground/test_ground_visualization.py and
tests/scripts/test_run_ground_classification.py.

Diagnostics consume only validated artifacts and use deterministic
original-index sampling. They may show decisions, support/problem cells, tile
counts, residual/slope/coverage summaries, and resource statistics, but cannot
recompute authority or create a Human Gate result. The CLI requires formal
source-reference/configuration inputs and explicit output/policy; it must not
invoke downstream A1/A2A/A2B/A3/A4/W1/V1/V2 or claim production certification.

## 7. Test surface, execution order, and closure

The plan has **11 new focused FGC test files**: three contract/store, three
spool/classifier/synthetic, three writer/publication/integration, and two
diagnostic/CLI. It also modifies the existing tests/io/test_scene_sample.py for
the narrow lineage adapter. Tests are grouped by authority surface, not one file
per predicate.

Use E:\Miniforge\envs\treeline\python.exe -m pytest -q for focused gates then
the full suite, followed by ruff check core algorithms tests scripts and
git diff --check. Synthetic fixtures never require scan_small. Real-scene
evaluation occurs only after implementation closure and remains Human Gate
evidence, not calibration proof.

Bundle A, then Bundle B, then Bundle C may be implemented in **one** Claude
Code + DeepSeek session, with their listed internal gates run in that order.
Git closure boundaries are minimum validity conditions, not required commits
between bundles. One independent implementation review follows A+B+C. Bundle D
and real-scene preparation remain a second pass.

A_B_C_ONE_PASS_IMPLEMENTATION: SAFE

## 8. Correction closure matrix

| Finding | Plan coverage |
|---|---|
| FGC-IR-001 | Bundle B PMF footprints, raw grid, and 20 m halo tests |
| FGC-IR-002 | Bundle A/C completion schema, discovery, publication-last tests |
| FGC-IR-003 | Bundle B exact sparse PMF tests |
| FGC-IR-004 | Bundle B seed/plane/vertical/routing tests |
| FGC-IR-005 | Bundle A/B native coverage and no-review-fraction-gate tests |
| FGC-IR-006 | Bundle B FGCSP1 spool, caps, cleanup, and preflight |
| FGC-IR-007 | Bundle B rational/raw integer boundary tests |
| FGC-IR-008 | Bundle A/C P-HAG digest proof-only tests |
| FGC-IR-009 | Bundle C logical classification, flags, VLR tests |
| FGC-IR-010 | Bundle A/C formal ingress and samples.py lineage adapter |
| FGC-IR-011 | Sections 3–7 enumerate the formerly omitted authority tests |
| FGC-RR-001 | Bundle B D3/D3_plus, centre-excluded component-support, and lower-envelope tests |

**Review result:** PASS — FGC-RR-001 corrected; all review findings closed.
