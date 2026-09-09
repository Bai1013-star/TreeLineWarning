# Gate D minimal remediation worker report — all blockers

Worker: Claude Code (DeepSeek). **Worker report only; GPT-6 Supervisor decides.**
`START_HEAD = 17f6075730c18c0d49bddfc9991693556984e019`

Gate C2 remains PASS and was not rerun. No data was redownloaded, no 163 M-point scan
was repeated, no model was trained, no held-out or local-challenge tree was read, no
product or frozen V0.1 file was touched, and nothing was pushed.

`GENERATOR_VERSION = GATE_D_SYNTHETIC_GENERATOR_V2`
`GENERATOR_DIGEST = 363ca30e5d6ca83f80df4f9d893c602062f909c6c980405df7f17459ba47471b`
`PROTOCOL_DIGEST = 36e4cfca043190d4d12c2ab9ee635d92bb479cb9b56d482ae1121fd797939729`

This round closed the six blockers from `TREE_QC_GATE_D_MINIMAL_REMEDIATION_ALL_BLOCKERS_V1`
by minimal remediation of the existing implementation, not a rewrite.

## Recovery note

The session was interrupted by a forced reboot. All remediation edits survived in the
dirty worktree and were recovered and completed rather than restarted. Nothing was
reset, restored, checked out over, stashed or cleaned. Every modified file was re-parsed,
all JSON/JSONL/PNG artifacts were re-validated, and the diagnostic was regenerated from
a clean process.

## 1. Blocker 1 — current candidate membership

The channel was `publisher GT source membership`, which is the answer. It is now
`current_candidate_membership`: whether the *current* single-tree segmentation selected
the point.

Every example is built from one shared `BaseScene` (`gate_d_scene.py`): approved
candidate + approved surviving context clipped to the frozen 3 m query. `GOOD` keeps the
publisher GT membership; defects flip only the flag.

| Class | Mechanism | Flag |
|---|---|---|
| `MERGED_MULTI_TREE` | whole neighbouring tree inside the query claimed | 0 → 1 |
| `NEIGHBOR_CONTAMINATION` | contiguous neighbouring region claimed | 0 → 1 |
| `NOISE` | scattered non-structural context points claimed | 0 → 1 |
| `OVER_SEGMENTED` | contiguous candidate region dropped | 1 → 0 |
| `CROWN_INCOMPLETE` | contiguous crown region dropped | 1 → 0 |
| `STEM_MISSING` | contiguous stem region dropped | 1 → 0 |

Claimed neighbour points are membership 1 even though their GT source is a neighbour;
dropped candidate points are membership 0 while remaining in the scene. `gt_source_id`
is audit-only and never a model channel. The tensor exposes exactly four channels.

**`GOOD` now always contains real context and membership-0 points**, so the mere
existence of a zero flag cannot identify the class. Measured membership-only shortcut
(balanced accuracy, threshold fitted on one partition and applied to the other): **0.508**.

`GOOD_HAS_REAL_CONTEXT = YES` (30/30 GOOD examples, `good_without_real_context = 0`)
`CURRENT_MEMBERSHIP_GT_LEAKAGE = NONE`
`MEMBERSHIP_ONLY_SHORTCUT_ACCURACY = 0.508` (chance 0.5)

## 2. Blocker 2 — every point source obeys the frozen allowlist

The unauthorized background path is **deleted**. `gate_d_geometry` no longer has
`allowed_background_trees`, and `CandidateGeometry` no longer has a `background` field;
it loads only the candidate and its frozen surviving context. `gate_d_scene` clips
context to the frozen 3 m query (candidate XY bbox + 3 m, native Z extent) verbatim.

`assert_scene_sources` is the single unified check over candidate, context, donor,
background, refill and alias sources. The diagnostic independently rebuilds the source
union from the scene and requires `reported == reconstructed`, and rejects any alias in
the high-containment alias set.

`UNAUTHORIZED_BACKGROUND_REFERENCES = 0` (was 72)
`CONTEXT_3M_QUERY_COMPLIANCE = ENFORCED` (context points asserted inside the box)
`SOURCE_UNION_COMPLETE = YES` (`actual_point_source_ids` == counts keys, counts sum to
the scene size, verified per example)
`CROSS_PARTITION_COMPLETE_SOURCE_LEAKAGE = 0`
`HELDOUT_SOURCE_USAGE = 0`
`HIGH_CONTAINMENT_USAGE = 0`

`NOISE` needed a data-supported definition: the public data cannot supply a genuine
"background" separate from neighbouring canonical trees, so `NOISE` is limited to
**scattered** context points (at most one per spatial cell) and is structurally distinct
from the **contiguous** contamination region. No background source was invented.

## 3. Blocker 3 — class-specific refill removed; STEM applicability

`_background_in_region` and both class-specific refills are **deleted**. `CROWN_INCOMPLETE`
and `STEM_MISSING` no longer delete points or inject other trees: the real crown/stem
points stay in the scene with membership 0.

`STEM_MISSING` now requires observable lower evidence — at least 24 points in the bottom
45 % of the height range, within 0.75 m of the trunk axis, spanning at least 1.5 m
vertically. Otherwise it raises `STEM_MISSING_NOT_APPLICABLE` and the example is not
generated.

`CROWN_CLASS_SPECIFIC_REFILL = NONE`
`STEM_CLASS_SPECIFIC_REFILL = NONE`
`STEM_APPLICABLE = 30`
`STEM_NOT_APPLICABLE = 0` (diagnostic scope; the rejection path is exercised by test on a
flattened fixture)

## 4. Blocker 4 — metric scale and structured sampler

Per-example RMS scaling is **removed**. `local_translate` performs translation plus
division by one fixed research-wide constant (`METRIC_SCALE_CONSTANT = 10.0`), so metric
size and cross-example comparability survive. `per_example_scaling = False` is recorded.

Group budgets are now **proportional to each scene's own publisher GT split**, which is
identical for every class of a candidate. A fixed 60 % candidate quota was tried and
rejected because it shifted the sampled membership fraction by up to +0.43. Within each
group, spatial cells and height bands receive a quota floor. Changed-membership evidence
is reserved inside those fixed budgets and **never duplicated**; a shortfall is recorded.

Measured on the diagnostic: candidate/context quotas identical across every class of a
candidate (e.g. 602/3494 for every class of the first candidate), height bands 4/4
sampled, spatial coverage 1.0, changed evidence ≥ 32 wherever ≥ 32 changed points exist.

`PER_EXAMPLE_RMS_SCALING = NO`
`METRIC_SCALE_PRESERVED = YES`
`CANDIDATE_CONTEXT_COVERAGE = CLASS-INDEPENDENT`
`HEIGHT_COVERAGE = 1.0`
`BOUNDARY_COVERAGE = PRESENT` (both groups always sampled)
`CHANGED_MEMBERSHIP_EVIDENCE = RESERVED, NEVER DUPLICATED`

## 5. Blocker 5 — fingerprint detector replaced

`loo_nearest_centroid` now builds class references **and standardisation statistics**
from the other samples only, predicts, then compares. Global standardisation, which
leaked the held-out sample's own statistics, is gone.

Two mandatory negative controls verify the detector itself:

| Control | Result |
|---|---|
| random features (same shape as real) | separation 0.048, limit 0.25 → pass |
| implanted perfectly class-coded scalar | separation 1.000, minimum 0.90 → pass |
| detector verified | **YES** |

The membership-only shortcut is reported with **balanced** accuracy, because the
diagnostic is deliberately imbalanced (6 GOOD vs 36 defect per candidate); raw accuracy
would report the majority-class rate. Cross-partition balanced accuracy 0.508.

Eight checks, all pass, `CRITICAL_FINGERPRINTS = []`. Artifact:
`GATE_D_FINGERPRINT_REMEDIATION.json`.

`FINGERPRINT_RANDOM_CONTROL = PASS (0.048)`
`FINGERPRINT_IMPLANTED_SHORTCUT = PASS (1.000)`
`FINGERPRINT_REAL_DIAGNOSTIC = NONE_DETECTED`

## 6. Blocker 6 — training lock binds the current digests

`gate_d_lock` now requires all five fields and verifies three against the **current**
implementation: `generator_version`, `generator_digest` and `protocol_digest`. Changing
the generator or the frozen protocol after acceptance invalidates the marker.

Required: `gate_d = ACCEPTED`, `training_allowed = true`, `generator_version`,
`generator_digest`, `protocol_digest`. Any missing marker, missing field, stale version,
stale digest or stale protocol digest exits 3 with `GATE_D_NOT_ACCEPTED`. Tests cover
missing marker, missing field (each), explicit HOLD, `training_allowed = false`, stale
generator digest, wrong generator version, wrong protocol digest, a correct mock marker
(allowed only at test level) and a code change after the marker.

`TRAINING_LOCK_CURRENT_DIGEST_BOUND = YES`
`STALE_DIGEST_REJECTED = YES`
`MISSING_MARKER_REJECTED = YES`
`MISMATCH_REJECTED = YES`

## 7. Bounded diagnostic

210 examples (5 train + 5 validation representative trees × 7 classes × 3 severities),
21 representative PNGs, 57 s.

| Field | Value |
|---|---|
| examples | 210 |
| violations | `[]` |
| `GENERATOR_FINGERPRINT` | `NONE_DETECTED` |
| `CRITICAL_FINGERPRINTS` | `[]` |
| `QUARANTINE_USAGE` | 0 |
| `HELDOUT_USAGE` | 0 |
| `CROSS_PARTITION_LEAKAGE` | 0 |
| unauthorized point sources | 0 |
| source-union mismatches | 0 |
| stem applicable / rejected | 30 / 0 |
| determinism | two clean runs produce identical `deterministic_digest` |

Artifacts: `GATE_D_SYNTHETIC_DIAGNOSTIC.json`, `GATE_D_DIAGNOSTIC_MANIFEST.jsonl`,
`GATE_D_FINGERPRINT_REMEDIATION.json`, `GATE_D_GENERATOR_PROTOCOL.md`, 21 PNGs.

## 8. Verification

| Check | Result |
|---|---|
| `pytest test_gate_d.py` | **101 passed** |
| `pytest test_canonical_identity.py` | **30 passed** (Gate C2 regression) |
| combined | **131 passed** |
| ruff on new files | all checks passed |
| Python parse check on every module | clean |
| JSON / JSONL / PNG integrity after reboot | clean (210 JSONL lines, 21 valid PNGs) |
| diagnostic regenerated from a clean process | `violations = []`, digests stable |
| launchers 02–05, EFG | exit 3 `GATE_D_NOT_ACCEPTED` |

`TRAINING_STARTED = NO`, `POINTNET_TRAINED = NO`, `POINTNET2_TRAINED = NO`.

## 9. Verdict

`FINAL_WORKER_VERDICT = GATE_D_WORKER_READY_FOR_SUPERVISOR_REVIEW`

All six blockers are closed with deterministic, documented, fail-closed behaviour. Gate D
acceptance remains the Supervisor's decision; `TRAINING_ALLOWED` stays `FALSE` and no
acceptance marker exists.
