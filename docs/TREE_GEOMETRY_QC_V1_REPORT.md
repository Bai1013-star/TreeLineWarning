# TREE_GEOMETRY_QC_V1_REPORT

`GEOMETRY_QC_NOT_GOOD_ENOUGH`. This bounded feasibility experiment does not justify
shipping Geometry QC. No product, scientific algorithm, editor, UI or package code
was changed. This is a failure of the current evidence/validation route, not proof
that geometry QC can never work.

The important result is not zero false PASS: the conservative experimental
diagnostic abstained on all eight bad references and therefore detected none.
A finer-grid signal finds five, but disappears when the grid is coarsened. Existing
stem hypotheses are promising supporting evidence but not independent ground truth.
There are also no defensible *explicit-good* labels in the audited reference set:
the prior eight good claims are preserved as provisional controls, not discarded or
silently counted as true negatives. The usefulness gate fails A and cannot establish C/E.

## Baseline and isolation

The requested status, current branch, HEAD, log, unstaged and staged diff checks were
performed. The original `E:/TreeLineWarning-fast` checkout was at `4d330ed` with ten
modified tracked files and many untracked artifacts. None was modified by this task.
The required frozen commit exists. A new clean worktree was created at
`E:/TreeLineWarning-tree-geometry-qc-v1`, branch `codex/tree-geometry-qc-v1`, starting
at `5f6f4d1fd5d79a3a922bf5924ae1b4e3d2b8f4d9`. No AGENTS.md was found in the
repository file search. No VLM code, runtime or model assets were copied or loaded.
Only the prior reference manifest, evidence descriptions and rendered images were read.

## Actual algorithm and available evidence

| Source | What it computes / persists | QC interpretation |
|---|---|---|
| `algorithms/spans/tree.py` | HAG CHM, smoothing, height-adaptive peak suppression, marker watershed, HAG-consistent 3D crown membership | Crown generation evidence, not independent segmentation validation |
| `algorithms/spans/tree_runner.py:511` | Legacy crown generation on fine representatives; hybrid refinement; exact original-index raw back-map; span manifest and NPZ publication | Authoritative candidate memberships and source indices; reuse caches, do not rescan LAS |
| `algorithms/spans/stem_guided_hybrid.py:239` | Multi-slice lower/mid-height column tracking; vertical extent, gaps, PCA verticality, radius, drift, support, heuristic stem score | Existing physical evidence; score is not calibrated probability |
| Same module, `_deduplicate_crown_stems` / `segment_stem_guided_hybrid` | Crown-conditioned nearby-column deduplication; separation, support and score checks; split only when every child is structurally confirmed; nearby fragment attachment | Multiple raw columns may be branches, duplicated columns or vegetation; do not call them multiple trees |
| Same module, `_candidate_from_crown` | Crown occupancy area/extent, apex at highest HAG member, stem/crown/vertical/connection evidence | Peak location and crown size do not prove a complete biological tree |
| `span_tree_manifest.json` | Per-candidate structural metrics, apex XY/HAG, raw/fine counts and reasons; production tree records include CONFIRMED_TREE candidates | Survives restart; cached original evidence needs lineage mapping and membership validity after edits |
| `tree_fine.npz`, `fine_representatives.npz`, `chm_review.npz` | Fine coordinates, representative original indices, HAG, labels, raw weights and CHM context | Cheap potential features; sampled representatives/CHM are not raw membership authority |
| `core/stage_b/dbh_recovery.py` V1 | Reconstructs accepted stem evidence, local raw supplementation, editor eligibility, continuity and DBH fit; optional explicitly marked allometry | DBH fit failure is not bad segmentation; allometry is not stem geometry proof |
| `core/stage_b/dbh_raw_recovery.py` V2 | Current membership seeds, bounded raw ROI, plausible distinct axes, bridge/continuity, owner conflict, candidate summaries | Useful competing-axis evidence only when revision/source binding matches |
| `core/stage_b/dbh_identity_recovery.py` V3 | Distinct axis claims, current crown-to-stem identity support, global ownership resolution | Detects ownership ambiguity, not automatically a segmentation defect |
| `core/stage_b/dbh_review.py` V4 | Revision-bound human stem identity confirmation and point-fitted measurement | Confirms a stem for DBH; does not certify the entire crown membership |
| `core/stage_b/dbh_evidence.py` and DBH loader | Strong source identity, hashes, membership/lineage, editor revision, persisted evidence/pointer validation | Existing safety pattern; DBH binding is largely batch-wide and should not be copied blindly for local QC invalidation |
| `core/stage_b/editor.py`, `algorithms/folder_pipeline/tree_editor.py` | Exact delta-encoded original-index removals, delete/merge operations, replay/undo, immutable originals and current finals | Proves edits and membership changes, not their biological intent; no explicit per-tree KEEP/segmentation label in these journals |
| Batch reconciliation / span core+halo lineage | Global/persistent IDs, merged span-tree keys, primary span, apex and full-resolution members | Authoritative identities; scoped neighbor context can be built locally; no validated boundary-contamination label is already present |

Important implementation findings:

- `CHM_peak_count` defaults to **1** when no `crown_peak_count_by_id` is supplied;
  the production caller does not supply that map. Treating it as measured multi-crown
  evidence would be a bug. `saddle_evidence` is a split indicator, not measured saddle depth.
- `validated_independent_stem_count` is **1 for all 15 original field trees**,
  including all eight bad references. It describes retained candidates after split
  validation/fallback, not the full competing-axis population.
- Raw columns range 7–69 in bad cases and 2–8 in the seven field provisional controls;
  these overlap. Deduplicated hypotheses range 1–4 in bad cases and equal 1 in those
  controls. Six bad cases have multiple hypotheses, but T009/T014 do not.
- Detailed validated stem trajectories are persisted by the separate candidate route;
  the production route saves summary fields rather than that complete hypothesis bank.
- The field DBH artifacts have 12 **post-edit** rows and historical V1/V2/V3/V4 evidence.
  They cannot be used as pre-edit bad-tree features: deleted trees are absent and
  edited crowns have changed. This would leak the human correction into the benchmark.
  `evidence.json` inventories the versions and bindings without treating them as valid
  pre-edit axes. No expensive DBH recovery was rerun.
- Journal `source_sha256` is null in the audited batches. Artifact and membership
  hashes provide benchmark provenance; they are not a complete production source
  identity binding. A future product implementation must reuse the strong preparation
  identity and verify it, rather than accept the null journal field.

## Reference audit

The complete manifest is `research/geometry_qc_v1/reference_manifest.json`; it retains
each original label claim, tree ID, project/run, before/after revision, defect claim,
source and justification, with a separate revised label and reason.

| Audited label | Count | Justification |
|---|---:|---|
| EXPLICIT_BAD | 8 | Pre-edit multi-view sheets show substantial separated geometry or extended non-tree-like structure; actual point correction corroborates it. Exact biological subtype remains provisional. |
| EXPLICIT_GOOD | 0 | No explicit per-tree segmentation acceptance was found. |
| UNCERTAIN_REFERENCE | 9 | Seven unchanged field memberships, one unchanged acceptance membership, one 315-point archived trim with missing geometry. |

All eight bad contact sheets were inspected in this audit and their hashes and
pre-edit bindings rechecked. A batch REGION_LASSO removed 10,000,089 points across
the eight field trees. It was one selection operation, not eight independently
recorded defect diagnoses. T012/T013/T014 were emptied by that operation; they were
not separate DELETE commands. The prior report's biological multi-tree subtypes
remain useful hypotheses, not adjudicated species/ownership truth.

For all 16 renderable references, both before/after membership hashes were checked
and exact index removal replay matched the persisted final members. The remaining
archived case lacks membership arrays. Other trees in the referenced acceptance
batches have no stronger labels and were not promoted into ground truth.

The manifest's 17 entries are **not a 17-tree current scene**: the field artifact has
15 original trees and 12 final trees; the separate acceptance final artifact has 16.
The benchmark extracts 16 reference memberships across three cached artifacts.

## Feature analysis and experimental decision

Full min/median/max distributions, original stem metrics, sensitivity checks and
all 17 case results are in `research/geometry_qc_v1/ANALYSIS.md`; machine-readable
measurements are in `results.json`. Explicit-good distributions are marked unavailable.
The eight prior good claims are shown transparently as provisional controls.

The bounded extractor samples at most 16,384 points deterministically using a hash
of original_index. Features are counts, sampled Z extents, 2–98% PCA horizontal
widths, width aspect and width/Z-extent ratio, XY variance ratio, 20-bin vertical
occupancy, normalized maximum vertical gap, and sampled connected-component fractions
at diagonal/16, /32 and /64 voxel sizes. Z extent is explicitly **not HAG tree height**.
The grids use 26-neighbor connectivity. These are exploratory measurement resolutions,
not scene-specific physical distances or production thresholds.

| Feature / evidence | Observed separation | Confounder / decision |
|---|---|---|
| Point count and size | Bad counts span 13,997–4,197,324, controls 106,330–704,065 | Acquisition density, species, size; reject standalone rule |
| Width/Z-extent ratio | Bad 0.837–4.894, controls 0.372–0.724 | Tempting scene separation; broad crowns, slope, crown-only capture and species make a fitted cutoff indefensible |
| Width aspect | Bad 1.030–4.512, controls 1.061–1.371 | Strong overlap; reject standalone rule |
| Vertical gaps | Bad 0.003–0.084, controls 0.002–0.059 | Occlusion, point density and sampling; reject standalone rule |
| Second component at /64 | >=10% for 5/8 bad, 0/8 provisional controls | Voxel orientation, sampling, bridges, foliage gaps; retain for research only |
| Same signal at /32 or /16 | 0/8 bad, 0/8 provisional controls | Scale instability; no validated automatic rule |
| Multiple deduplicated hypotheses plus /64 component | 4/8 bad, 0/7 field provisional controls | Promising combined evidence, still unvalidated axes and scale-sensitive; no reliable PASS complement |
| Independent stem count / peak count | Both 1 across original field references | Post-selection count/default, not independent validation; reject |
| DBH ambiguity | Available after editing, not revision-matched for pre-edit bad references | Leakage risk; inventory only, exclude from classifier |
| Crown modes, neighbor contamination, missing crown | No trusted independent mode/ownership/completeness measurements and subtype-balanced labels | Defer; do not fabricate features from CHM defaults or proximity alone |

The predeclared diagnostic flags a second component of at least 10% at **both /32
and /64**, as material disconnected sampled geometry (`GEOMETRY_AMBIGUOUS`).
10% is an explicitly unvalidated materiality assumption, not a learned probability.
It otherwise returns UNCERTAIN; there is no validated PASS criterion. It detected
0/8 bad cases. The finer-resolution sensitivity result is disclosed rather than
quietly replacing the failed rule with the best-looking cutoff.

This experiment is not an exhaustive geometric model search. A simple rule can
separate parts of this scene, but the evidence does not establish robust generalization
or acceptable false flags on known-good trees. Do not turn abstentions into successful
detections or infer a zero false-positive rate from zero explicit-good cases.

## Runtime and scientific safety

Measured feature extraction total: **0.6225445 s / 16 memberships**, mean **0.0389090 s**.
Decision total: **0.0001848 s**, mean **0.00001155 s**. Cached artifact load/group total:
**1.1158555 s**. Full benchmark wall time including input hashing and verification:
**3.7334882 s**, excluding Python startup and the separate detailed journal/DBH audit.
Peak RSS delta sampled every 10 ms: **947,519,488 bytes (~904 MiB)**. This includes
NPZ decoding and scene membership arrays, not just feature workspace. No isolated
per-feature memory claim is made. The offline audit is not yet a lightweight product
cache implementation. No full 105M-point source scan occurred.

Input artifacts/journals were hashed before and after extraction. Product/science
source trees have an empty diff against the frozen baseline. Thus this change cannot
alter Ground/HAG/Wire/Vegetation, membership/original_index/IDs, DBH, any clearance or
witness, CONTACT/risk, Survey or Prediction_Input. No full scientific rerun on the
105M scene was performed or claimed.

The research suite checks source-order invariance, exact sample membership digest,
nonmutation, scaling/translation, disconnected synthetic geometry, missing DBH,
invalid inputs, honest labels and real benchmark misses. Synthetic clouds are not
reference good/bad biological trees. Existing tests exercise tree segmentation,
DBH V1–V4, editor mutations/clearance, restart, multi-span, risk and exports.
Combined outcome: **119 passed, 1 failed in 14.24 s**, including seven new research
tests (112 existing passed). The failure is the existing export test expecting
`display_tree_id` at column 33 while baseline now puts `dbh_cm` there. It was
independently reproduced in the unchanged frozen-source worktree (1 failed in 2.47 s).
The frozen export contract was not modified to silence that assertion.

## Usefulness gate and next route

A FAIL: zero detections for the stable two-scale diagnostic.
B: no bad PASS, but only because it abstains; this alone is not usefulness.
C UNESTABLISHED: no explicit-good control set.
D PASS for experimental interpretability, with sampling caveats.
E UNESTABLISHED: single-field dominance, scale sensitivity and missing defect coverage.
F PARTIAL: sub-second extraction, but nearly 1 GB offline decoding overhead; no scene
product-cache integration measured. G PASS: no frozen science changes required or made.

Stop before product integration and packaging. The next justified route is **1:
collect adjudicated, revision-bound reviewed data**. First adjudicate the existing
eight provisional good controls, record explicit reasons/subtypes for the eight bad
memberships, and preserve before/after arrays, strong source identity and reviewer
uncertainty. Add independent scenes with different densities, tree forms, terrain,
occlusion and defect types, including natural multi-stem good trees. Hold out entire
scenes for evaluation. Preserve the competing stem trajectories and real crown-mode
evidence with their original revisions, rather than only retained-candidate summaries.

In parallel, a future **route 4 investigation** should trace the extended linear/
flat structures visible in bad memberships through vegetation admission, source-index
role exclusions and crown back-mapping. The visuals justify investigation, not an
unverified claim that a particular frozen stage is wrong. No segmentation repair is
made in this task. A lightweight supervised feature model (route 2) becomes justified
only after labels and held-out scenes exist. A native 3D model (route 3) is not justified
by these eight bad examples or by the failure of this simple diagnostic.

The research records preserve features, experimental result, unknown human outcome,
action evidence and pre/post membership identities for later adjudication. No KEEP,
DELETE, MERGE or LASSO action has been automatically converted to a perfect class.
Production training-log persistence, affected-tree QC invalidation, restart restore,
filters and tree jump are intentionally not implemented after the failed gate.

## Required report fields

```text
BASELINE_HEAD = 5f6f4d1fd5d79a3a922bf5924ae1b4e3d2b8f4d9
FINAL_HEAD = reported in final handoff; report belongs to that commit
REFERENCE_GOOD = 0 (8 prior good claims retained as provisional controls)
REFERENCE_BAD = 8
REFERENCE_UNCERTAIN = 9
EXISTING_GEOMETRIC_EVIDENCE_REUSED = cached memberships, original indices, lineage, span structural metrics, neutral renders, journal, DBH provenance inventory
FEATURES_EVALUATED = see ANALYSIS.md and feature-method section
FEATURES_SELECTED = NONE for product; bounded descriptors retained for research
FEATURES_REJECTED = standalone size/aspect/gaps; default peak field; retained-stem count; stale/post-edit DBH as pre-edit features; unvalidated component rule
GOOD_BAD_FEATURE_ANALYSIS = research/geometry_qc_v1/ANALYSIS.md; explicit-good unavailable, provisional controls disclosed
BAD_TOTAL = 8
BAD_DETECTED = 0
BAD_UNCERTAIN = 8
BAD_FALSE_PASS = 0
GOOD_TOTAL = 0
GOOD_PASS = 0
GOOD_UNCERTAIN = 0
GOOD_FALSE_FLAG = 0 (no denominator; not evidence of specificity)
CASE_BY_CASE_RESULTS = ANALYSIS.md and results.json; all 8 bad cases are disclosed misses/abstentions
USEFULNESS_GATE = FAIL / DO NOT INTEGRATE
FEATURE_EXTRACTION_PER_TREE = mean 0.0389090 s over 16 reference memberships
QC_DECISION_SECONDS_PER_TREE = mean 0.00001155 s
TOTAL_SCENE_QC_SECONDS = NOT_MEASURED as product; 3.7334882 s reference audit, 0.6227293 s extraction+decision
PEAK_MEMORY_DELTA = 947519488 bytes sampled process RSS including cached NPZ decoding
GENERALIZATION_GUARDS = no ID/location rules; source-index-stable sample; relative geometry; uncertain labels preserved; no scene-fitted width/height cutoff; no stale DBH leakage; scale sensitivity disclosed
PROJECT_SPECIFIC_MAGIC_CONSTANTS = NONE
DBH_EVIDENCE_REUSED = provenance/version/candidate inventory; not used as pre-edit classification evidence
NEW_FULL_SOURCE_SCAN = NO
QC_MUTATES_SCIENCE = NO
SCIENTIFIC_EQUIVALENCE = exact empty product/science diff + benchmark input hash equality; no full-scene rerun claimed
ORDER_INDEPENDENCE = research feature/sample synthetic permutation test PASS; product QC N/A
RESTART = existing regression passed; new QC restore NOT_IMPLEMENTED_AFTER_GATE
EDITOR_REPLAY = 16/16 reference pre/post membership hashes and exact index replay PASS; archived geometry unavailable
STALE_INVALIDATION = NOT_IMPLEMENTED_AFTER_GATE; research provenance only is not product stale protection
UI_QC_SUMMARY = NOT_IMPLEMENTED_AFTER_GATE
UI_FLAGGED_FILTER = NOT_IMPLEMENTED_AFTER_GATE
UI_TREE_JUMP = NOT_IMPLEMENTED_AFTER_GATE
FUTURE_TRAINING_LOG = compact offline research features/results/uncertain labels/pre-post identities/action evidence; product logging deferred
SCOPED_RUFF = PASS
NODE_SYNTAX = NOT_RUN; no JavaScript changes
FOCUSED_TESTS = 7 passed
SCIENCE_REGRESSION = 112 existing passed, 1 baseline export-column assertion failed and independently reproduced
PACKAGE_REBUILT = NO
ZIP_PATH = N/A (no Preview)
ZIP_SHA256 = N/A (no Preview)
FROZEN_V0_1_UNCHANGED = YES; before/after SHA-256 e90730aad77f3dc2f25109a9cca296cdcf7303ac13888ed7552f489c7ceb1726
PUSH = NO
IF_FAILED_NEXT_ROUTE = collect explicitly adjudicated controls and independent scenes first; investigate segmentation contamination provenance; supervised features only after data, no native 3D model yet
FINAL_VERDICT = GEOMETRY_QC_NOT_GOOD_ENOUGH
```

## Reproduction

From the geometry worktree, using the existing Python 3.11 environment:

```powershell
E:/TreeLineWarning-fast/.venv311/Scripts/python.exe research/geometry_qc_v1/audit.py --manifest research/geometry_qc_v1/reference_manifest.json --output research/geometry_qc_v1/results.json
E:/TreeLineWarning-fast/.venv311/Scripts/python.exe research/geometry_qc_v1/evidence_audit.py
E:/TreeLineWarning-fast/.venv311/Scripts/python.exe research/geometry_qc_v1/summarize.py
E:/TreeLineWarning-fast/.venv311/Scripts/python.exe -m ruff check research/geometry_qc_v1
E:/TreeLineWarning-fast/.venv311/Scripts/python.exe -m pytest -q research/geometry_qc_v1/test_audit.py
```

External source paths and expected hashes are explicit in the manifest. Only output
paths inside the research directory are written. No model/runtime download is needed.
