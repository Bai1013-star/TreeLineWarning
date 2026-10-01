# CLEARANCE_ORDER_INDEPENDENCE_FIX_V1

Date: 2026-08-30. This report concerns the local implementation gate only; it is not a REAL acceptance rerun.

## Baseline and scope

- Branch: `feature/fast-research-pipeline`.
- Required and verified baseline: `c32cc489344d2bfe16005ba6951dbf221d522e4a`.
- Implementation/testing: GPT-5.6 Luna Max. Independent review: GPT-5.6 Sol High.
- Tracked worktree was clean; unrelated untracked files and prior REAL evidence are preserved.
- Permitted production files: `algorithms/spans/clearance.py`, `core/stage_b/batch.py`.
- No changes to reconciliation, Wire V2 reconstruction, Ground, vegetation/FRV, CROWN_STEM_HYBRID, exact distance mathematics/BVH/candidate search, thresholds, or CONTACT policy.
- No reset, push, or 105,055,350-point REAL run was performed. The final review below approves a subsequent REAL rerun, but does not execute it or claim that the REAL gate has passed.

## Independently confirmed defects

Sol reproduced the baseline concatenator behavior using small in-memory W1 arrays, without reading or rerunning the REAL scene:

| Input | Sample prefix | Wire-to-segment prefix | Built edges |
| --- | --- | --- | ---: |
| empty then 73-sample segment | `[0,0,73]` | `[0,0,1]` | 0 |
| 73-sample segment then empty | `[0,73,73]` | `[0,1,1]` | 72 |

A separate 2-wire/3-segment plus 1-wire/2-segment example produced `[0,2,3,2,4]`, whereas its correct wire-to-segment prefix is `[0,2,3,5]`. This confirms both the duplicated boundary and incorrect `wire_cursor` rebase, independently of distance calculations.

The existing W1 contract (`core/contracts/wire_models.py`) requires:

- wire prefix length = wire count + 1; starts at 0; ends at segment count; monotonic;
- sample prefix length = segment count + 1; starts at 0; ends at sample count; monotonic.

`ProjectScienceBridge` derives its internal numeric Span ordinal from line/tower topology. `bind_spans(selected)` itself preserves requested order, so its returned sequence is not a canonical ordering authority. The numeric internal Span ordinal is the ordering authority, not product-ID lexical order or display names.

Sol also traced the publication consumer: editor finalize iterates `BatchAnalysisResult.span_results`, while JSON serialization sorts mapping keys. Consequently, sorting only the temporary automatic-clearance wire list would not preserve canonical wire IDs/tie evidence through finalize and hydration. The approved bounded design keeps the solution in `batch.py`: publish explicit canonical order metadata and restore result mapping order from that metadata; legacy results without metadata remain readable. Malformed authoritative metadata must fail closed, not invent a lexical fallback.

## Validation evidence

Before implementation, Sol ran:

```powershell
.\.venv311\Scripts\python.exe -m pytest -q -p no:cacheprovider tests/vegetation/test_fast_baseline_clearance.py tests/vegetation/test_span_clearance_fast_path.py tests/vegetation/test_span_clearance_candidate_backend.py tests/project/test_multispan_fast_track.py tests/stage_b/test_stage_b_integration.py --basetemp E:\TreeLineWarning-fast\.test-tmp-clearance-sol-baseline-20260830-a
```

Result: **68 passed in 4.50s**. These existing tests did not expose the independently reproduced multi-Span prefix defect.

Sol's additional independent post-patch checks:

- 40 seeded three-Span W1 inputs, each tested in all 6 permutations: **240 prefix/edge-count/parent-wire checks passed**. Inputs include zero wires, wires with zero segments, zero-sample segments and non-ready segments.
- For the same 40 seeds, single-Span concatenated arrays and dtypes match the original `c32cc489` module loaded directly from Git: **40 passed**.
- Single-Span records, statuses, minima, edge counts and danger evidence match that baseline for all three backends (`PYTHON_BVH`, `NUMPY_DIRECT`, `SCIPY_CANDIDATE_EXACT`): **120 passed**.
- JSON key sorting followed by result hydration restores canonical topology order; legacy metadata absence, pending selected Spans without a result, five malformed metadata cases, and numeric `S999`/`S1000` ordering were independently checked: **PASS**.
- AST comparison against `c32cc489` confirms the only changed existing function in `clearance.py` is `_concatenate_span_wire_edges`; the only new function is the local prefix validator. All exact mathematics functions are unchanged.
- Git comparison confirms reconciliation, Wire reconstruction, canonical executor, exact Tree-Wire engine, configuration and risk policy files are unchanged.
- An order-only mutation was tested in a fresh process: with the corrected prefixes retained, temporarily return request order instead of canonical order. Both directions still had 4 edges and minimum 1 m, but nearest wire XYZ changed from `(1,-1,10)` to `(1,1,10)`. This isolates the wire-ID/tie-order defect from the offset defect. No production file was modified by the mutation.

The original LAS was independently hashed again after the production edits. SHA-256 remains `6d7ae65ff91d22e62ba97321b975949adf9e361596b733d7c799574877aae596` (6,303,322,159 bytes); raw source content is unchanged.

Sol also reran the preceding five-file suite plus `tests/stage_b/test_editor_clearance_invalidation.py` against the implementation using `--basetemp E:\TreeLineWarning-fast\.test-tmp-clearance-sol-existing-20260830-a`: **71 passed in 4.56s**.

## Implementation and required A-H gates

The source fix is restricted to the two permitted production files:

- `algorithms/spans/clearance.py`: validate local prefix vectors, keep one global initial zero, append local suffixes, use `segment_cursor` for wire-to-segment offsets and `sample_cursor` for segment-to-sample offsets. Empty Spans add no spurious boundary. Missing prefixes are accepted only for empty collections. Monotonicity checks use adjacent comparisons, avoiding subtraction overflow on malformed input.
- `core/stage_b/batch.py`: derive canonical Span order from numeric `SpanScienceBinding.internal_span_id`; order wire publication and the published result map; persist `batch_diagnostics.canonical_span_order`; normalize result maps from this authority during hydration. Reject malformed order metadata rather than infer a lexical fallback. Execution scheduling and global reconciliation inputs are not reordered.

No additional production files were required. Existing `editor.py` and `desktop/workflow.py` consume the normalized `BatchAnalysisResult` without modification. Historical results without order metadata remain compatible; this patch does not retroactively repair or rewrite historical runs.

| Gate | Evidence | Result |
| --- | --- | --- |
| A: empty then valid | `[0,73]`, 1 ready segment, 72 edges | PASS |
| B: valid then empty | Same correct prefixes and 72 edges | PASS |
| C: two nonempty Spans | 3 wires, 5 segments, 20 samples; correct lengths/endpoints/parent wires | PASS |
| D: segment cursor | 2 wires/3 segments + 1 wire/2 segments gives `[0,2,3,5]` | PASS |
| E: forward/reverse Stage-B | Same status, minima, complete records, ready/edge counts and zero duplicate membership | PASS |
| F: both Spans have wires | Distinct equidistant geometries retain canonical wire ID and nearest XYZ in both orders | PASS |
| G: single Span | Arrays and edge bundle unchanged; independent baseline comparisons also passed | PASS |
| H: exact distance oracle | Two distinct trees, one nearest each Span; local-bundle exact oracle checked under all three backends | PASS |

Additional tests cover zero inputs, two empty Spans, malformed/missing prefixes, actual JSON serialization/hydration, pending selected Spans, malformed metadata, legacy compatibility, dataclass replacement and hydrated editor finalize.

New test files:

- `tests/vegetation/test_span_wire_concatenation.py`: 9 cases.
- `tests/stage_b/test_clearance_order_publication.py`: 7 cases.

With the baseline concatenator loaded directly from `git show c32cc489:algorithms/spans/clearance.py` in memory, the final 9 unit cases produce **8 failed, 1 passed**. Failures include extra boundaries, wrong wire cursor, missing validation and 3 edges instead of 7. Baseline production files were never checked out over the working tree.

## Final focused and bounded regression commands

Luna ran:

```powershell
.\.venv311\Scripts\ruff.exe check algorithms/spans/clearance.py core/stage_b/batch.py tests/vegetation/test_span_wire_concatenation.py tests/stage_b/test_clearance_order_publication.py

.\.venv311\Scripts\python.exe -m pytest -q tests/vegetation/test_span_wire_concatenation.py tests/stage_b/test_clearance_order_publication.py tests/vegetation/test_fast_baseline_clearance.py tests/vegetation/test_manual_clearance_equivalence.py tests/vegetation/test_span_clearance_fast_path.py tests/vegetation/test_span_clearance_candidate_backend.py tests/stage_b/test_stage_b_integration.py tests/stage_b/test_editor_clearance_invalidation.py tests/stage_b/test_danger_points.py --basetemp E:\TreeLineWarning-fast\.test-tmp-clearance-combined-01

.\.venv311\Scripts\python.exe -m pytest tests/stage_b tests/desktop tests/project/test_multispan_fast_track.py tests/project/test_identity_terminal_transaction.py tests/vegetation/test_tree_identity.py tests/folder_pipeline/test_tree_editor.py tests/vegetation/test_span_wire_concatenation.py tests/stage_b/test_clearance_order_publication.py -q -p no:cacheprovider --basetemp E:\TreeLineWarning-fast\.test-tmp-clearance-broad-02 --tb=short
```

Results: **Ruff PASS; focused 88 passed in 6.12s; bounded regression 193 passed, 1 skipped in 16.58s.** The focused invocation emitted one pytest cache warning. The bounded skip is `tests/desktop/test_pdal_runtime.py:8` because the default Python environment does not expose PDAL. The broader command explicitly names the Stage-B publication test as well as its parent directory; the 193 count is the command's reported test count, not a claim of 193 distinct node IDs.

Sol independently reran Ruff and the two new test files plus `tests/project/test_multispan_fast_track.py` and `tests/stage_b/test_stage_b_integration.py` with `-q -p no:cacheprovider --basetemp E:\TreeLineWarning-fast\.test-tmp-clearance-sol-final-focused-20260830-a`: **39 passed in 2.33s**. The existing lineage tests remain unchanged and pass, including nonzero/subset IDs, ownership and single-Span compatibility.

## Additional environment checks

Sol's extra all-directory regression includes synthetic full-pipeline tests that require PDAL. Two such tests initially failed at Ground preparation with `FAST_GROUND_BACKEND_UNAVAILABLE: Python PDAL binding missing`, before clearance is reached.

To diagnose without changing dependencies or source, a fresh process loaded only the existing ABI-compatible native runtime from `dist/TreeLineWarning-rc-20260823f/TreeLineWarning/_internal` and its `pdal_runtime` DLL directory, with `PROJ_NETWORK=OFF`. PDAL 3.5.4's real minimal operation passed. Absolute module-path assertions confirmed `algorithms.spans.clearance` and `algorithms.vegetation.fast_baseline_pipeline` still load from this checkout, not bundled science code.

In that process, the unchanged `tests/vegetation/test_fast_baseline_e2e.py` plus `tests/desktop/test_pdal_runtime.py` passed: **3 passed in 2.28s**, using a separate `.test-tmp-clearance-sol-native-e2e-20260830-a` directory. These are tiny synthetic scenes, not the 105,055,350-point REAL acceptance source. No production code was edited to bypass the environment failure.

Sol also reran the bounded broader set with the same native bootstrap: **194 passed in 15.09s, no skips** (`.test-tmp-clearance-sol-native-bounded-20260830-b`). Paths were `tests/stage_b`, `tests/desktop`, `tests/project/test_multispan_fast_track.py`, `tests/project/test_identity_terminal_transaction.py`, `tests/vegetation/test_tree_identity.py`, `tests/folder_pipeline/test_tree_editor.py`, and `tests/vegetation/test_span_wire_concatenation.py`, with `-q -p no:cacheprovider --tb=short`.

The first native-bootstrap bounded attempt used `python -` and reported 193 passed plus one failure in the multiprocessing identity test: Windows spawn cannot reload a `<stdin>` main module. This harness-only issue was corrected by using `python -c` with the same bootstrap and test selection; no production or test code changed. The successful 194-test result includes that concurrency test and the real PDAL minimal-operation test.

## Extra all-directory regression: not an all-green suite

Command:

```powershell
.\.venv311\Scripts\python.exe -m pytest -q -p no:cacheprovider tests/project tests/stage_b tests/vegetation tests/folder_pipeline/test_tree_editor.py tests/contracts/test_tree_wire_clearance_contracts.py tests/desktop/test_tree_review_ux.py tests/desktop/test_tree_display_numbering_surface.py --basetemp E:\TreeLineWarning-fast\.test-tmp-clearance-sol-broad-20260830-a
```

Result: **1562 passed, 11 failed, 2 skipped, 1 warning in 713.38s**. This run started before the final H expansion; the final 9-case oracle suite was separately validated afterward as described above.

- Two failures are the PDAL-dependent synthetic pipeline tests, resolved under the existing native runtime as described above.
- Eight failures are `test_v4_performance_equivalence.py::test_optimized_result_matches_frozen_baseline` (single_tree, two_trees, tile_2x2 through tile_7x7).
- One failure is `test_v4_tree_segmentation.py::test_fragment_ceiling_is_local_and_preserves_baseline_semantics`.
- Skips: `test_associations.py:597` lacks verified V2 artifacts; `test_tree_artifact_io.py:523` cannot create a filesystem symlink.
- The warning is the intentional duplicate ZIP member fixture in `test_tree_artifact_io.py`.

The nine V4 failures compare result digests with older golden values. Sol confirmed there is no diff against `c32cc489` in the V4 algorithm, its FRV/V4 contracts, either failing test file, or `golden_v4_optimization_baseline.json`. The V4 segmentation entry point does not import either changed production module.

Sol independently compiled the actual V4 algorithm source from `git show c32cc489:algorithms/vegetation/v4_tree_segmentation.py` in a fresh module and compared representative outputs:

| Scene | c32cc489 and current, identical | Golden expected |
| --- | --- | --- |
| single_tree | `bc3b6a5658b640e95acb3b6982a7727914ee63b92128bb1ae98d4200851496e6` | `d591575c4576bf95b7b051f13c5af5b80af6b08d709e5f3015ee7e0313f2aa20` |
| two_trees | `df2d66706a0f14f5f90209383d037559ce0547555cc3d474141f53087fd947aa` | `7c0744bed5b6539bafa14115a4f16e434431600538d06493b2fc8b5f28d31b36` |

Luna subsequently ran the exact nine failing node IDs in two fresh processes:

```powershell
.\.venv311\Scripts\python.exe -m pytest tests/vegetation/test_v4_performance_equivalence.py::test_optimized_result_matches_frozen_baseline tests/vegetation/test_v4_tree_segmentation.py::test_fragment_ceiling_is_local_and_preserves_baseline_semantics -q -vv -p no:cacheprovider --basetemp E:\TreeLineWarning-fast\.test-tmp-v4-current-20260830-a --tb=long
```

Current result: **9 failed in 6.62s**. In the baseline process, both changed production modules were replaced in memory with their complete `c32cc489` source from Git, then the same node IDs ran with `.test-tmp-v4-baseline-20260830-a`: **9 failed in 5.42s**. Every actual/expected digest pair and failing node ID was identical between baseline and candidate. No worktree checkout, reset, index modification, V4 change or golden update was used.

**Disposition:** these nine V4 golden mismatches are pre-existing baseline failures, not introduced by this clearance fix. Their underlying V4/golden compatibility cause is outside this task and was not repaired. The all-directory suite is explicitly **not all green**. This does not invalidate the passing scoped publication fix or justify changing frozen science.

## Final review and Git closure

Sol reviewed every production diff line and both new test files, verified `segment_cursor`/`sample_cursor` and prefix invariants, verified the bridge-derived ordering across JSON/editor hydration, and confirmed no exact distance or reconciliation changes. There are no remaining actionable findings in the scoped fix.

- Local implementation commit: `27f42e19b5206a5912613f199165347ba59bdd4c`.
- Parent: `c32cc489344d2bfe16005ba6951dbf221d522e4a`; the previous lineage repair remains intact.
- The commit contains only the two production files and two test files listed below. This report is a separate local acceptance artifact, not included in that source commit.
- No push. Remote branch was observed at `610c0f3178aa2632d2f7638216638b21b28436aa`.
- **Sol explicitly approves a subsequent REAL acceptance rerun on this local candidate. No REAL rerun was executed in this task.** The REAL Gate remains unverified after this second fix and must use fresh run IDs, preserve historical failures, and use the validated native runtime.
- CONTACT remains `PENDING_REVIEWED_PHYSICAL_DETECTOR`; no distance-zero/contact inference or policy change was made.

```text
CLEARANCE_ORDER_INDEPENDENCE_FIX_V1

BASELINE_HEAD = c32cc489344d2bfe16005ba6951dbf221d522e4a
IMPLEMENTATION_MODEL = GPT-5.6 Luna Max
REVIEW_MODEL = GPT-5.6 Sol High

FILES_CHANGED = algorithms/spans/clearance.py; core/stage_b/batch.py; tests/vegetation/test_span_wire_concatenation.py; tests/stage_b/test_clearance_order_publication.py

ROOT_CAUSE_CONFIRMED = YES
DUPLICATE_PREFIX_BOUNDARY_CONFIRMED = YES
WIRE_SEGMENT_OFFSET_CURSOR_CONFIRMED = YES
RESULTS_INSERTION_ORDER_DEPENDENCE_CONFIRMED = YES

SAMPLED_CURVE_OFFSETS_GLOBAL_CONTRACT = PASS
WIRE_SEGMENT_OFFSETS_GLOBAL_CONTRACT = PASS
STABLE_WIRE_RESULT_ORDER = PASS

EMPTY_THEN_VALID = PASS
VALID_THEN_EMPTY = PASS
TWO_NONEMPTY_SPANS = PASS
WIRE_SEGMENT_CURSOR_TEST = PASS
FORWARD_REVERSE_CLEARANCE = PASS
BOTH_SPANS_WIRE_ORDER_INDEPENDENCE = PASS
SINGLE_SPAN_CLEARANCE = PASS
EXACT_DISTANCE_ORACLE = PASS

LINEAGE_REGRESSION = PASS
DUPLICATE_FINAL_ORIGINAL_INDEX = 0

RUFF = PASS
FOCUSED_TESTS = 88 passed (Luna); 39 passed (Sol independent)
REGRESSION = 194 passed (Sol native bounded); 193 passed/1 skipped (Luna bounded); extra all-directory 1562 passed/11 failed/2 skipped (2 PDAL failures resolved by native rerun; 9 V4 failures identical on baseline)

SOL_INDEPENDENT_DIFF_REVIEW = PASS

FROZEN_SCIENCE_CHANGED = NO
RAW_DATA_MUTATED = NO

COMMIT = 27f42e19b5206a5912613f199165347ba59bdd4c
PUSH = NO

REAL_RERUN_APPROVED = YES_BY_SOL; NOT_EXECUTED
REMAINING_BLOCKERS = REAL_GATE_RERUN_PENDING; PREEXISTING_V4_GOLDEN_MISMATCH_9; CONTACT_PENDING_REVIEWED_PHYSICAL_DETECTOR
FINAL_VERDICT = PASS
```

`FINAL_VERDICT = PASS` is the scoped clearance implementation/review verdict, not a declaration that the full repository suite or the REAL end-to-end Gate has passed.
