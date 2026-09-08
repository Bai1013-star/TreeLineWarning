# TREE_SEGMENTATION_AI_QC_V1_REPORT

## Verdict

`MODEL_NOT_GOOD_ENOUGH`. Product integration and Preview packaging intentionally stopped at
the mandatory model-acceptance gate.

The pinned Q8_0 SmolVLM2-500M model was tested locally through pinned llama.cpp CPU binaries
against eight real pre-edit TreeLineWarning tree memberships. All eight trees were affected by
an existing human `REMOVE_POINTS` review operation in a finalized real project. Five were
classified `PASS` with confidence effectively equal to 1.0, three degraded to `UNCERTAIN`, and
none was classified `REVIEW_REQUIRED`. The false-PASS result violates the explicit acceptance
rule and makes UI/product integration unsafe.

## Baseline and isolation

- Baseline and new branch start: `5f6f4d1fd5d79a3a922bf5924ae1b4e3d2b8f4d9`
- Branch: `codex/tree-ai-qc-v1`
- Worktree: `E:\TreeLineWarning-ai-qc-v1`
- Frozen V0.1 package was found at
  `E:\TreeLineWarning-Releases\TreeLineWarning-0.1.0-Windows-x64-20260908-5f6f4d1.zip`
  (1,576,627,672 bytes) and was not modified.

## Pinned runtime inputs

- Model repository: `ggml-org/SmolVLM2-500M-Video-Instruct-GGUF`
- Repository revision: `ccd7aae53bcb1997355c2f094959e72b3642ce17`
- Model: `SmolVLM2-500M-Video-Instruct-Q8_0.gguf`
- Model bytes: 436,808,704
- Model SHA-256: `6f67b8036b2469fcd71728702720c6b51aebd759b78137a8120733b4d66438bc`
- Projector: `mmproj-SmolVLM2-500M-Video-Instruct-Q8_0.gguf`
- Projector bytes: 108,785,184
- Projector SHA-256: `921dc7e259f308e5b027111fa185efcbf33db13f6e35749ddf7f5cdb60ef520b`
- Model license metadata: Apache-2.0
- Runtime: llama.cpp `b10855`, official Windows x64 CPU release asset
- Runtime ZIP SHA-256: `71492912f2b36956bc6a9995ec6338ccf0d69b6ebc31bb3bd5791b372e92e874`
- Runtime license: MIT; bundled LLVM OpenMP component license file was present
- Extracted runtime bytes: 46,758,685
- Model plus projector bytes: 545,593,888
- Total candidate ONEDIR increase before notices: 592,352,573 bytes

Both licenses permit redistribution subject to their notice/license obligations. Because the
model failed acceptance, no shipping notice set or release payload was produced.

## Real evidence and expectation table

Source project batch:
`analysis-d434dfe375814b2d9b8a1d765a3e4d13`, status `COMPLETE`, span status `FINALIZED`.

The persisted human operation removed 10,000,089 points across tree IDs
`0, 1, 5, 9, 10, 12, 13, 14`. This is the only reference claim used: each pre-edit membership
requires human review. No species, defect subtype, or automatic ground-truth label was invented.

| Case | Original points | Human-removed share | Expected | Model result |
|---|---:|---:|---|---|
| PRE_EDIT_T000 | 2,285,682 | 85.39% | REVIEW_REQUIRED | PASS, 1.0 |
| PRE_EDIT_T001 | 4,197,324 | 99.88% | REVIEW_REQUIRED | UNCERTAIN, invalid/truncated JSON |
| PRE_EDIT_T005 | 1,934,135 | 70.84% | REVIEW_REQUIRED | PASS, 1.0 |
| PRE_EDIT_T009 | 33,031 | 99.17% | REVIEW_REQUIRED | UNCERTAIN, invalid/truncated JSON |
| PRE_EDIT_T010 | 1,957,287 | 76.08% | REVIEW_REQUIRED | PASS, 1.0 |
| PRE_EDIT_T012 | 468,245 | 100.00% | REVIEW_REQUIRED | UNCERTAIN, inconsistent PASS |
| PRE_EDIT_T013 | 481,685 | 100.00% | REVIEW_REQUIRED | PASS, 1.0 |
| PRE_EDIT_T014 | 13,997 | 100.00% | REVIEW_REQUIRED | PASS, 1.0 |

## Benchmark measurements

- Real trees tested: 8
- Known bad: 8
- Known good: 0 (no explicit KEEP labels were asserted)
- PASS: 5
- REVIEW_REQUIRED: 0
- UNCERTAIN: 3
- Known-bad false PASS: 5
- Obvious-error recall: 0/8
- Views per tree: 6 plus one contact sheet
- Total view render time: 3.995 s
- Mean view render time per tree: 0.499 s
- Model load time: 1.116 s
- Mean CPU inference time per tree: 7.602 s
- Total eight-tree QC time: 66.177 s
- Highest observed runtime working set: 1,206,407,168 bytes
- GPU/VRAM: not used; CPU mode worked
- Offline operation: yes after assets were staged
- Runtime lifecycle: started for the batch and terminated after it; no server remained running

Two repeated fixed-prompt classifications of `PRE_EDIT_T000` returned the same invalidly
confident PASS payload, demonstrating deterministic failure rather than useful discrimination.

## Implemented benchmark boundary

The branch contains a deterministic, visualization-only six-view renderer, strict result parser,
on-demand llama-server adapter, and reproducible real-evidence benchmark command. The renderer
records membership hash, render revision, editor revision, individual view hashes, and contact
sheet hash. It never writes authoritative coordinates or memberships.

No product workflow, science algorithm, editor behavior, web UI, release spec, or frozen package
was changed. Failure-safety, stale invalidation, UI, package smoke, and full scientific regression
work were correctly not pursued after the mandatory stop gate failed.

## Final report fields

```text
TREE_SEGMENTATION_AI_QC_V1_REPORT

BASELINE_HEAD = 5f6f4d1fd5d79a3a922bf5924ae1b4e3d2b8f4d9
MODEL = ggml-org/SmolVLM2-500M-Video-Instruct-GGUF@ccd7aae
MODEL_QUANTIZATION = Q8_0
MODEL_LICENSE = Apache-2.0
RUNTIME = llama.cpp b10855 Windows x64 CPU
RUNTIME_LICENSE = MIT (+ LLVM OpenMP license in binary distribution)

REAL_TREES_TESTED = 8
KNOWN_GOOD_CASES = 0 (no explicit KEEP labels asserted)
KNOWN_BAD_CASES = 8
KNOWN_BAD_FALSE_PASS = 5
KNOWN_GOOD_FLAGGED = N/A
UNCERTAIN_COUNT = 3
MODEL_ACCEPTED = NO

VIEW_COUNT_PER_TREE = 6
VIEW_RENDER_TIME = 0.499 seconds/tree mean
MODEL_LOAD_TIME = 1.116 seconds
CPU_INFERENCE_PER_TREE = 7.602 seconds mean
GPU_INFERENCE_PER_TREE = NOT_RUN
TOTAL_REAL_SCENE_QC_TIME = 66.177 seconds for 8 cases
PEAK_AI_RSS = 1,206,407,168 bytes observed
PEAK_AI_VRAM = 0 (CPU mode)

PASS_COUNT = 5
REVIEW_REQUIRED_COUNT = 0
UNCERTAIN_COUNT = 3
MERGED_MULTI_TREE_DETECTION = NOT_DEMONSTRATED
OVER_SEGMENTATION_DETECTION = 0 accepted detections
NEIGHBOR_CONTAMINATION_DETECTION = NOT_DEMONSTRATED
STEM_MISSING_DETECTION = NOT_DEMONSTRATED

AI_MUTATES_SCIENCE = NO
SCIENTIFIC_EQUIVALENCE = NOT_RUN; integration stopped before science changes
EDITOR_REPLAY = SOURCE EVIDENCE READ ONLY
RESTART = NOT_RUN; integration stopped
STALE_QC_INVALIDATION = RENDER PROVENANCE IMPLEMENTED; PRODUCT PATH NOT INTEGRATED

CPU_AI_QC = RUNTIME WORKED; MODEL FAILED ACCEPTANCE
GPU_AI_QC = NOT_RUN
LOW_MEMORY_BEHAVIOR = ON-DEMAND PROCESS RELEASE VERIFIED; PRODUCT PATH NOT INTEGRATED
MISSING_MODEL_FALLBACK = PARSER/RUNTIME BOUNDARY IMPLEMENTED; PRODUCT PATH NOT INTEGRATED
RUNTIME_FAILURE_FALLBACK = PRODUCT PATH NOT INTEGRATED

MODEL_BUNDLED = NO
FIRST_RUN_DOWNLOAD_REQUIRED = NO (candidate assets staged offline, package not built)
OFFLINE_OPERATION = MODEL/RUNTIME BENCHMARK YES; PRODUCT PACKAGE NOT BUILT
PACKAGE_SIZE_BEFORE = 1,576,627,672-byte V0.1 ZIP
PACKAGE_SIZE_AFTER = NOT_BUILT
PACKAGE_SIZE_INCREASE = candidate ONEDIR assets 592,352,573 bytes; ZIP not built

UI_AI_QC_SUMMARY = NOT_INTEGRATED
UI_FLAGGED_FILTER = NOT_INTEGRATED
UI_TREE_JUMP = NOT_INTEGRATED

PACKAGE_REBUILT = NO
ZIP_PATH = NONE
ZIP_SHA256 = NONE
CLEAN_PC_AI_QC = PENDING
PUSH = NO
FINAL_VERDICT = MODEL_NOT_GOOD_ENOUGH
```
