# TREE_SEGMENTATION_AI_QC_V2_MODEL_SELECTION_REPORT

## Verdict

`VLM_ROUTE_NOT_SUFFICIENT`.

No model is accepted for product integration. SmolVLM2-2.2B ran locally and repeatably, but
confidently passed every explicit-bad reference tree in both contact-sheet and multi-image modes.
Qwen2.5-VL-3B was rejected before download or inference because the exact upstream model revision
uses the Qwen Research License, whose grant is non-commercial only. The GGUF repository's
Apache-2.0 metadata cannot broaden the upstream weight license.

The next recommended route is deterministic geometry QC combined with a dedicated lightweight
supervised classifier trained on explicitly reviewed TreeLineWarning memberships. Do not
automatically escalate to general-purpose 7B/12B VLMs.

## Baseline and scope

- Required V1 baseline: `b3dd6573487fef2a8683ce88064aa4cd6da83147`
- Branch: `codex/tree-ai-qc-v1`
- The initial status, branch, HEAD, unstaged diff, and staged diff checks passed; the worktree was
  clean at the exact required commit.
- The frozen V0.1 package was not modified.
- Changes are limited to AI-QC benchmark/render tooling, its reference manifest/tests, and this
  report. There are no product UI, editor, segmentation, scientific-output, or packaging changes.

## Trustworthy reference set

The committed manifest is `tests/ai_qc/fixtures/reference_manifest_v2.json`. Each entry records
tree ID, project/run, pre- and post-edit membership revision, category, issue, evidence source,
and the reason the label is defensible. Artifact SHA-256 checks and per-tree membership digests
are enforced before rendering.

| Category | Count | Definitive metric use |
|---|---:|---|
| `EXPLICIT_GOOD` | 8 | Yes |
| `EXPLICIT_BAD_VISUAL` | 8 | Yes |
| `EDITED_BUT_VISUAL_LABEL_UNCERTAIN` | 1 | No |

Seven good cases are exact unchanged memberships in a finalized, human-edited field batch. The
eighth is an exact unchanged membership in a COMPLETE two-span acceptance batch whose only
persisted delete targeted a different tree. The eight bad cases come from the field batch's
pre-edit membership: every defect was visually audited as meaningful and independently
corroborated by a human operation that removed 70.84% to 100% of its points.

The uncertain case is a persisted 315-point lasso edit. That event proves a membership change but
not an obvious visual defect. Its archived point chunks are absent, so it is documented but not
rendered or counted as false-PASS evidence.

Reference limitations:

- The set is tiny, drawn mainly from one field batch, and is not evidence of statistical or
  production accuracy.
- Bad examples are dominated by merged-tree and severe noise contamination. There is no balanced
  coverage of crown-incomplete, stem-missing, or over-segmented defects.
- One good case is acceptance evidence rather than an independent field-review label.
- Visual defensibility was audited once, not adjudicated by multiple independent reviewers.
- Editor acceptance is the best available evidence, not absolute biological ground truth.

## V2 representation and contract

Rendering remains deterministic and revision-bound (`tree-ai-qc-render-v2`). Every render has
eight 512x512 views:

1. `TOP_XY`
2. `FRONT_XZ`
3. `SIDE_YZ`
4. `OBLIQUE_A`
5. `OBLIQUE_B`
6. `CONTEXT`
7. `STEM_DETAIL` (lower 42% of candidate height)
8. `HEIGHT_CODED` (blue low, yellow middle, red high)

The candidate is consistently gold except for the explicitly labelled height-gradient view.
Neighbors are muted blue in context/stem views. Camera, projection, padding, background, point
size, labels, and legends use fixed rules. The 2048x1024 contact sheet contains the same eight
panels. Rendering is visualization-only and has no authority over memberships or science.

Both input modes used the same conservative task semantics and low-temperature/seeded inference.
The strict authoritative payload has exactly `status`, `issue`, `confidence`, and `reason_codes`.
Any invalid payload fails closed to `UNCERTAIN`. A fixed confidence threshold of `0.65` defines a
confident PASS; it was not tuned per model or tree.

## Artifact and license gate

### Model A: SmolVLM2-2.2B-Instruct

- GGUF repository revision: `1bc3c9f74ceafd4c8d4411cc9cf188bba3798f91`
- Model: `SmolVLM2-2.2B-Instruct-Q4_K_M.gguf`
- Model bytes: 1,112,602,656 (1,112.603 MB; 1,061.061 MiB)
- Model SHA-256: `0cf76814555b8665149075b74ab6b5c1d428ea1d3d01c1918c12012e8d7c9f58`
- Projector: `mmproj-SmolVLM2-2.2B-Instruct-Q8_0.gguf`
- Projector bytes: 592,523,200 (592.523 MB; 565.074 MiB)
- Projector SHA-256: `ae07ea1facd07dd3230c4483b63e8cda96c6944ad2481f33d531f79e892dd024`
- Model license: Apache-2.0, verified from upstream and GGUF repository metadata.

### Model B: Qwen2.5-VL-3B-Instruct

- Upstream revision inspected: `66285546d2b821cf421d4f5eb2576359d3770cd3`
- Candidate GGUF metadata: Q4_K_M model 1,929,901,056 bytes (1,929.901 MB) and Q8_0
  projector 844,757,728 bytes (844.758 MB).
- The exact upstream `LICENSE` is the Qwen Research License Agreement. Its grant permits use,
  modification, and distribution **for non-commercial purposes only**, and directs commercial
  users to request a separate license.
- Outcome: `REJECTED_AT_LICENSE_GATE`. No Qwen model/projector was downloaded, loaded, or tested.
  All runtime, load, accuracy, memory, and speed fields are therefore `NOT_RUN`.

### Runtime

- llama.cpp `b10855`, official Windows x64 release assets.
- Runtime license: MIT; the distributed CPU archive also contains the LLVM OpenMP license.
- CPU extracted runtime: 46,758,685 bytes.
- CUDA 13.3 extracted runtime plus redistributables: 704,382,573 bytes.
- CPU archive SHA-256 (from V1):
  `71492912f2b36956bc6a9995ec6338ccf0d69b6ebc31bb3bd5791b372e92e874`.
- CUDA runtime archive SHA-256:
  `e3fc18370502ff3144bfc25f82f4a9fb5c726611359609a676a16201ddd01edc`.
- CUDA redistributable archive SHA-256:
  `1462a050eb4c684921ba51dcc4cc488a036674c3e73e9945ee705b854808d03e`.

## Model A benchmark

Every trustworthy case was run twice in both input modes with GPU offload. CPU inference was
measured on a bounded pair consisting of one explicit-good and one explicit-bad contact sheet;
an attempted full CPU matrix was stopped because low initial free RAM caused severe paging and
projected multi-hour runtime. The successful bounded sample ran after memory release.

### Resource measurements

| Measurement | CPU | GPU offload |
|---|---:|---:|
| Model load | 2.720 s | 2.433 s |
| Mean inference | 38.434 s/tree (2 cases, contact) | 1.348 s/tree (32 contact runs) |
| Multi-image inference | Not repeated on CPU | 1.700 s/tree (32 runs) |
| Peak process-tree RSS | 4,534,857,728 bytes (4.223 GiB) | 6,288,842,752 bytes (5.856 GiB) |
| Peak model-attributable dedicated VRAM | 0 | 132,120,576 bytes (126 MiB) |

Windows WDDM does not expose per-process VRAM for this process. The GPU value is the maximum
`nvidia-smi` total-dedicated-memory increase over the immediately measured desktop baseline;
the desktop was already using roughly 4.8 GiB. It is therefore an incremental estimate, not a
precise allocation audit. The server was confirmed as a CUDA compute process. Host RSS is the
more conservative resource-safety number.

Both CPU and GPU server processes terminated after their bounded run. No `llama-server` remained,
and available memory recovered. Only one model was ever resident.

### Accuracy and repeatability

Counts below are distinct cases, with run counts in parentheses.

| Metric | Contact sheet | Eight separate images |
|---|---:|---:|
| Bad false PASS | 8/8 (16/16 runs) | 8/8 (16/16 runs) |
| Bad detected (`REVIEW_REQUIRED`) | 0/8 (0/16) | 0/8 (0/16) |
| Bad uncertain | 0/8 (0/16) | 0/8 (0/16) |
| Good PASS | 8/8 (16/16) | 8/8 (15/16 runs) |
| Good false flag | 0/8 (0/16) | 0/8 (0/16) |
| Good uncertain | 0/8 (0/16) | 1/8 (1/16 runs) |

All valid PASS outputs had confidence 1.0. Therefore all 16 bad contact runs and all 16 bad
multi-image runs exceeded the fixed 0.65 confident-PASS threshold.

Contact sheets were status/issue/confidence stable for 16/16 cases across repeats, but this is
repeatable failure: every bad case was confidently passed. Multi-image mode was stable for 15/16
cases. The second `FIELD_GOOD_T011` multi-image response overflowed the reason-code constraint and
correctly failed closed to `UNCERTAIN`; it did not flip PASS to REVIEW_REQUIRED.

Contact-sheet input is marginally more stable and faster than multi-image input for this model,
but neither mode has useful bad-case discrimination.

## Decision

SmolVLM2-2.2B fails model-acceptance conditions 1 through 3: it demonstrates no meaningful
good/bad discrimination, confidently passes every obvious bad case, and emits no useful review
detections. Its stable output is not a safety advantage because it is stably wrong. Qwen cannot
enter the packaged Preview/product path under the inspected upstream non-commercial license.

No model is selected. No larger general VLM is tested automatically.

## Final report fields

```text
TREE_SEGMENTATION_AI_QC_V2_MODEL_SELECTION_REPORT

BASELINE_HEAD = b3dd6573487fef2a8683ce88064aa4cd6da83147
FINAL_HEAD = see final handoff (a commit cannot embed its own resulting hash)

REFERENCE_SET = tests/ai_qc/fixtures/reference_manifest_v2.json; 17 documented cases, 16 renderable/trustworthy metric cases

EXPLICIT_GOOD_COUNT = 8
EXPLICIT_BAD_VISUAL_COUNT = 8
EDITED_VISUAL_UNCERTAIN_COUNT = 1

REFERENCE_LIMITATIONS = tiny and mostly single-field-batch; defect-class imbalance; one acceptance-derived good; no multi-reviewer adjudication; uncertain archive lacks point chunks

VIEW_CONFIGURATION = 8 deterministic 512x512 views: TOP_XY, FRONT_XZ, SIDE_YZ, OBLIQUE_A, OBLIQUE_B, CONTEXT, STEM_DETAIL lower 42%, HEIGHT_CODED; 2048x1024 contact sheet
CONTACT_SHEET_TESTED = YES; compared with eight-image input

MODEL_A = SmolVLM2-2.2B-Instruct GGUF Q4_K_M + Q8_0 projector
MODEL_A_LICENSE = Apache-2.0; runtime llama.cpp MIT (+ LLVM OpenMP notice in CPU distribution)
MODEL_A_MODEL_MB = 1,112.603 MB (1,061.061 MiB)
MODEL_A_PROJECTOR_MB = 592.523 MB (565.074 MiB)
MODEL_A_LOAD_SECONDS = CPU 2.720; GPU 2.433
MODEL_A_CPU_SECONDS_PER_TREE = 38.434 mean; bounded 2-case contact-sheet sample
MODEL_A_GPU_SECONDS_PER_TREE = contact 1.348 mean; multi-image 1.700 mean
MODEL_A_PEAK_RSS = CPU 4,534,857,728 bytes; GPU run 6,288,842,752 bytes
MODEL_A_PEAK_VRAM = 132,120,576 bytes (126 MiB) incremental dedicated-memory estimate under WDDM

MODEL_A_BAD_FALSE_PASS = contact 8/8 cases (16/16 runs); multi 8/8 (16/16)
MODEL_A_BAD_DETECTED = contact 0/8; multi 0/8
MODEL_A_BAD_UNCERTAIN = contact 0/8; multi 0/8
MODEL_A_GOOD_PASS = contact 8/8 (16/16 runs); multi 8/8 (15/16 runs)
MODEL_A_GOOD_FALSE_FLAG = contact 0/8; multi 0/8
MODEL_A_GOOD_UNCERTAIN = contact 0/8; multi 1/8 (1/16 runs)
MODEL_A_REPEATABILITY = contact 16/16 status+issue stable with zero confidence range; multi 15/16 stable, no PASS-REVIEW flip

MODEL_B = Qwen2.5-VL-3B-Instruct GGUF Q4_K_M candidate; REJECTED_AT_LICENSE_GATE
MODEL_B_LICENSE = Qwen Research License Agreement; non-commercial only; commercial use requires separate license
MODEL_B_MODEL_MB = 1,929.901 MB advertised candidate size; not downloaded
MODEL_B_PROJECTOR_MB = 844.758 MB advertised Q8_0 candidate size; not downloaded
MODEL_B_LOAD_SECONDS = NOT_RUN
MODEL_B_CPU_SECONDS_PER_TREE = NOT_RUN
MODEL_B_GPU_SECONDS_PER_TREE = NOT_RUN
MODEL_B_PEAK_RSS = NOT_RUN
MODEL_B_PEAK_VRAM = NOT_RUN

MODEL_B_BAD_FALSE_PASS = NOT_RUN
MODEL_B_BAD_DETECTED = NOT_RUN
MODEL_B_BAD_UNCERTAIN = NOT_RUN
MODEL_B_GOOD_PASS = NOT_RUN
MODEL_B_GOOD_FALSE_FLAG = NOT_RUN
MODEL_B_GOOD_UNCERTAIN = NOT_RUN
MODEL_B_REPEATABILITY = NOT_RUN

MULTI_IMAGE_VS_CONTACT_SHEET = contact marginally faster/more stable; both confidently passed all explicit-bad cases and are unsafe

SELECTED_MODEL = NONE
MODEL_ACCEPTED_FOR_PRODUCT_INTEGRATION = NO

WHY_SELECTED = none: Model A has zero bad-case discrimination; Model B fails the commercial-license gate

IF_NONE_ACCEPTED_NEXT_ROUTE = geometry QC + dedicated lightweight supervised model trained on explicitly adjudicated memberships; alternatively evaluate a native 3D point-cloud QC model

SCOPED_RUFF = PASS; ruff check core/ai_qc scripts/benchmark_tree_ai_qc.py scripts/benchmark_tree_ai_qc_v2.py tests/ai_qc
FOCUSED_TESTS = PASS; 9 passed in 0.95 s (pytest -q tests/ai_qc with isolated basetemp)

PRODUCT_UI_MODIFIED = NO
SCIENCE_MODIFIED = NO
PACKAGE_REBUILT = NO
MODEL_BINARIES_COMMITTED = NO
PUSH = NO

FINAL_VERDICT = VLM_ROUTE_NOT_SUFFICIENT
```
