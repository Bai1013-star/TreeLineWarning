# V1 explainable vegetation-candidate review

## Baseline and scope

- Branch and baseline: `feature/algorithm`, `48533cc feat: add confirmed wire centerline reconstruction`.
- Official interpreter: `E:\Miniforge\envs\treeline\python.exe` (Python 3.11.15; NumPy 2.3.5; SciPy 1.17.1; pytest 9.1.1; Ruff 0.15.22).
- V1 schema/algorithm versions: `vegetation-candidates-v1` / `v1-explainable-woody-candidates-v1`.
- V1 reads A1/A2A/A2B/A4 NPZ/JSON only. It neither reads LAS/LAZ nor consumes A3. W1 is accepted solely by the preview API as an optional overlay and does not enter scoring.
- V1 is a full-M candidate stage for above-ground tree-related/woody vegetation relevant to later V2 grouping. It is not species recognition, individual-tree segmentation, universal botanical classification, or wire-clearance analysis. Reliable near-ground rejection is a corridor-product boundary, not a biological classification claim.

Changed tracked files are `core/contracts/vegetation_candidates.py`, `core/contracts/__init__.py`, `algorithms/vegetation/__init__.py`, `algorithms/vegetation/candidates.py`, `tests/contracts/test_vegetation_candidate_contracts.py`, `tests/vegetation/__init__.py`, `tests/vegetation/test_candidates.py`, and this review. A1, A2A, A2B, A3, A4, and W1 implementations and approved parameters were not changed.

## Contract and validation design

V1 requires full-M A1/A2A/A2B/A4 mapping equality before row-wise scoring: non-negative unique `original_index`/`sample_index`, M, origins, A1/A2A/A4 exact `xyz_local`, source/manifest identifiers, NPZ/JSON hashes, required dtypes/shapes, finite geometry, and A4 result/hash consistency. A mismatch fails rather than silently reordering a stage. Object arrays and pickle loading are rejected.

The public NPZ keeps `xyz_local`, mappings, A4 `wire_id_by_point`, full-M HAG state, rule score, confidence, automatic/final levels, final mask, evidence/reason/quality flags, manual provenance/action, copied A2A/A2B flags, and component arrays. The JSON stores portable sibling NPZ filename, hash, configuration, logical provenance, statistics, array descriptors, counts, and formulas. Reload accepts explicit NPZ/JSON paths or a JSON sibling pair after relocation; it does not require the original `E:` project root.

Publication uses temporary files, validates the temporary NPZ and JSON, publishes NPZ first and JSON last, and removes temporary/partial output after failure. Default overwrite is false. Output is restricted to `outputs/vegetation_candidates/`; previews are restricted to `outputs/previews/`.

## Formula and defaults

For `clip01(x)=min(1,max(0,x))`, V1 calculates:

- `E_H = clip01((height_agl_m - 0.30) / (3.00 - 0.30))` for reliable/uncertain HAG; strongly negative raw HAG has zero height evidence. Invalid HAG uses `0.5 * clip01((height_m-Q10)/(Q90-Q10))` only when valid fallback percentiles exist.
- `E_V = clip01((scattering-0.02)/(0.20-0.02))`.
- `P_P = clip01((planarity-0.35)/(0.70-0.35))`.
- `P_L = clip01((linearity-0.75)/(0.95-0.75)) * (1-E_V)`.
- Density support is robust `log1p(local_density)` normalized by valid scene Q10/Q90. Missing/undefined density contributes zero and reduces confidence.
- RGB/intensity are validated when present but contribute exactly zero by default.

The positive weights are height .35, volumetric .45, density .20, auxiliary .00. Planarity and linearity penalty weights are .25 and .20 with a .45 cap. The final likelihood is `clip01(weighted_positive_evidence - capped_penalty)`.

Confidence is independent: `.35*C_geometry + .35*C_HAG + .15*C_ground_support + .15*C_density`, clipped to `[0,1]`. Reliable/uncertain/strong-negative/invalid-with-fallback/invalid-without-fallback HAG confidences are 1.00/.55/.40/.30/.15. Confidence caps HIGH at .75 and MEDIUM at .50; lower confidence remains LOW/reviewable when likelihood clears the low threshold. Default score levels are .35/.55/.75 for LOW/MEDIUM/HIGH.

## HAG, hard exclusion, and manual behavior

The public `VegetationHagState` contains `RELIABLE`, `UNCERTAIN`, `INVALID_OR_UNAVAILABLE`, and `STRONGLY_NEGATIVE`; `HEIGHT_FALLBACK_USED` is an evidence bit. Actual A2B enum semantics are used, including `HIGH_LOCAL_GROUND_VARIATION`, `SPARSE_GROUND_CELL`, `INTERPOLATED_GROUND`, `INVALID_GROUND_ESTIMATE`, and `EXTRAPOLATION_TOO_FAR`.

Only reliable near-ground HAG at or below .30 m is automatically hard-excluded. Uncertain HAG is used as likelihood evidence with reduced confidence and is never hard-excluded merely for ground uncertainty. Strongly negative raw HAG records a reason and has zero height contribution, but is not a blanket hard exclusion.

The only formal wire rule is `wire_id_by_point >= 0`. Every confirmed wire is final `REJECTED`, mask false, and records `CONFIRMED_WIRE_HARD_EXCLUSION`. Non-confirmed A4 tracks only record an evidence bit; default score/confidence penalty is zero.

Manual selection uses original indices and inclusive local XYZ boxes. Separate include/exclude/conflict/wire-blocked provenance bits are retained. Precedence is structural invalidity, confirmed wire, manual exclusion, manual inclusion, then automatic output. Include/exclude conflicts end excluded; an include on a confirmed wire is recorded as `WIRE_BLOCKED`; no expert wire override exists.

## Synthetic and sensitivity validation

The focused V1 suite covers contract/enum/configuration validation; score-component clipping boundaries; positive-evidence and penalty monotonicity; likelihood/confidence separation; reliable near-ground and confirmed-wire exclusions; uncertain, invalid/fallback, and strongly negative HAG; manual conflict and wire-blocking; optional RGB/intensity neutrality; deterministic cancellation handling; relocated standalone reload; invalid geometry; and 0.5x/1x/2x spatially scaled equivalent HAG configurations.

The scale test scales local coordinates, physical HAG/ground-distance quantities, and corresponding physical configuration values together; likelihood and confidence remain equal. This demonstrates formula consistency under physically equivalent unit-scale transformations. It is not a claim of multi-scene real-world generalization. Synthetic tests deliberately do not select thresholds from scene_simple coordinates or tree locations.

## Real `scene_simple` run

Inputs were the current structured artifacts only:

| stage | NPZ SHA-256 |
| --- | --- |
| A1 | `7b12e8c950672a61e3a916a51820bf793a7301b12ace82593a9a660bf94d5d02` |
| A2A | `8c13802e8aafa83a5e0c0f04318c40d1ac3cff8cb4e7d98713c63654d11a74d4` |
| A2B | `1dde7cba2eb0d5040a6bbc3844fcc1be735b2cd09bdbd8c50176065d687072ec` |
| A4 | `86bbd912ee281ab938963c153467e75e986afec1c37dc0d7d92cbbb26a922583` |

Cross-stage validation passed with M=146,126 and exact mandatory mapping/coordinate alignment. No manual selection was supplied.

| metric | count |
| --- | ---: |
| reliable HAG | 102,574 |
| uncertain HAG | 42,878 |
| invalid/unavailable HAG | 0 |
| strongly negative raw HAG | 674 |
| height fallback used | 0 |
| structural invalid | 0 |
| reliable near-ground exclusions | 35,156 |
| confirmed-wire exclusions | 7,396 |
| rejected | 81,651 |
| LOW | 22,494 |
| MEDIUM | 25,773 |
| HIGH | 16,208 |
| final candidates | 64,475 (44.12%) |

Every confirmed-wire point was verified final rejected. Automatic and final levels are identical because manual counts are all zero. A4 non-confirmed-track evidence count is 7,544; it did not cause a formal exclusion. Geometry degradation count is 6, HAG-uncertain count is 42,878, and density-unusable count is 0.

Score percentiles min/Q25/Q50/Q75/max are `0/.10356/.33281/.60019/1`; confidence percentiles are `.61820/.84250/1/1/1`. Component medians are HAG `.26306`, volumetric `.42319`, planarity penalty `.40821`, linearity penalty `0`, and density `.69126`. Runtime was 0.301 s; RSS start/peak/end was 39,788,544 / 96,907,264 / 96,907,264 bytes.

Formal V1 NPZ: `outputs/vegetation_candidates/scene_simple_vegetation_candidates.npz`, 4,194,362 bytes, SHA-256 `473acca89891a2ddd2ed5896515990ab4db5d9e8b984ff92821f2d28d33cbb65`. JSON: 8,721 bytes, SHA-256 `8b08431834c71af9bfc42c904fa10d98da66644ea3392bbc8738410689057fdd`. Standalone reload passed with all 146,126 points and mask/level consistency.

Preview: `outputs/previews/scene_simple_vegetation_candidates_preview.html`, 10,999,169 bytes, SHA-256 `01ae46815d7dc0fd438e6ce5859f3dc14ea9941a23a40a4eaf6f8802e8468d25`. It uses deterministic bounded base/candidate/warning samples, distinct confirmed-wire styling, LOW legend-only by default, and optional W1 overlay support. The local browser security policy blocked loading a local `file:` preview, so no visual quality claim is made here; the generated preview requires human visual review.

## Checks and limitations

- Focused V1 tests: 12 passed.
- Full test suite: 190 passed.
- `ruff check .`: passed.
- `compileall core algorithms tests`: passed.
- `git diff --check`: passed.
- `ruff format --check .` reports only pre-existing unrelated formatting differences in `algorithms/io/samples.py`, `algorithms/preprocessing/preview.py`, and `core/contracts/pointcloud.py`; V1 files and the modified contract export were formatted.
- `git ls-files data outputs` and the prohibited artifact pattern check found no tracked runtime data/output files.

No precision, recall, F1, or proven real-world generalization is claimed: there are no reliable full vegetation labels and only one primary real development scene. Expected ambiguous cases include branches/trunks versus poles/fences/wires and building/roof edges versus sparse canopy. RGB/intensity are deliberately neutral by default. The 44.12% diagnostic candidate rate is not an accuracy measurement and must be visually reviewed before threshold changes; no post-preview retuning was performed.

## Human visual review and vehicle-versus-tree diagnosis

Human visual review confirmed that the major tree crowns and branches are broadly retained as vegetation candidates and that confirmed wire points remain visually distinct and excluded. It also identified a vehicle-like object that is substantially included as LOW/MEDIUM/HIGH vegetation candidates.

The inspected vehicle-like point, `original_index=302562`, has score `.5533`, confidence `1.0000`, final level `MEDIUM`, reliable HAG, `height_agl_m=.4461` m, linearity `.2410`, planarity `.5119`, scattering `.2471`, and local density `6458.41`. The inspected canopy point, `original_index=1324416`, has score `.6930`, confidence `.8425`, final level `MEDIUM`, uncertain HAG, `height_agl_m=2.5202` m, linearity `.8011`, planarity `.0652`, scattering `.1337`, and local density `2770.82`.

Existing point-level features overlap too strongly to justify a safe general V1 vehicle-removal rule. Raising near-ground exclusion risks low shrubs; increasing planarity penalties risks planar leaves and mixed canopy surfaces; and increasing linearity penalties risks branches and trunks. No car-specific rule will be added.

A bounded candidate-component comparison is more informative. The vehicle-like component has an approximately `2.892 × 3.701 × 1.284 m` bounding box, height-to-horizontal ratios of about `.35–.44`, median HAG about `1.221 m`, and is compact, shallow, and horizontally extended. The tree-like component has an approximately `2.126 × 2.799 × 3.542 m` bounding box, height-to-horizontal ratios of about `1.27–1.67`, median HAG about `2.390 m`, and substantial vertical/crown-like structure. This distinction remained useful at `.10–.20 m` graph radii. A `.30 m` radius bridged broad candidate regions and is unsuitable as a fixed global grouping radius.

V1 therefore remains a permissive point-level candidate stage. Vehicle/artificial-structure suppression is assigned primarily to V2 instance-level analysis. V2 should initially report, not immediately hard-filter, general descriptors: vertical extent; horizontal footprint and aspect ratios; PCA axis dimensions; HAG distribution; planar/scattered proportions; LOW/MEDIUM/HIGH composition; and top-to-base/crown-like vertical profile. These descriptors must be validated on more than one vehicle/tree example before rejection thresholds are frozen.

This vehicle false positive is an expected known limitation of the unlabeled first-version V1 candidate baseline. It does not constitute a precision, recall, F1, or other labelled accuracy measurement.
