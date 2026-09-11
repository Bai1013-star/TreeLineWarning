# Gate D minimal remediation worker report — all blockers

Worker: Claude Code (DeepSeek). **Worker report only; GPT-6 Supervisor decides.**
`START_HEAD = 17f6075730c18c0d49bddfc9991693556984e019`

Gate C2 remains PASS and was not rerun. No data was redownloaded, no 163 M-point scan
was repeated, no model was trained, no held-out or local-challenge tree was read, no
product or frozen V0.1 file was touched, and nothing was pushed.

`GENERATOR_VERSION = GATE_D_SYNTHETIC_GENERATOR_V2`
`GENERATOR_DIGEST = f3c1bd98f74a9ff954e5455c306fb1db47a45a6eb4525dba99c1d587196d644a`
`PROTOCOL_DIGEST = 36e4cfca043190d4d12c2ab9ee635d92bb479cb9b56d482ae1121fd797939729`

This round closed the six blockers from `TREE_QC_GATE_D_MINIMAL_REMEDIATION_ALL_BLOCKERS_V1`
by minimal remediation of the existing implementation, not a rewrite.

## Round 3 addendum — B5 final minimal remediation (`TREE_QC_GATE_D_V2_B5_FINAL_MINIMAL_REMEDIATION_V2`)

The review confirmed blockers 1, 2, 3, 4 and 6 closed and left **Blocker 5** on HOLD: the
artifact/defect split was internally inconsistent.

`corruption` re-labels points and never moves them, so no XYZ statistic may vary with the
synthetic class. The previous revision nevertheless admitted only X and Y to the artifact
gate and grouped **Z with membership**, which was inconsistent with the frozen semantics.

### What changed

```
ARTIFACT_COLUMNS      = (0, 1, 2, 4, 5, 6)     # mean/std of X, Y, Z
ARTIFACT_COLUMN_NAMES = feature_mean[0..2], feature_std[0..2]
DEFECT_COLUMNS        = (3, 7)                 # membership only
DEFECT_COLUMN_NAMES   = feature_mean[3], feature_std[3]
```

`FINGERPRINT_VERSION = GATE_D_FINGERPRINT_V4`. The artifact slice is applied to the real
detector (`feature_stats[:, list(ARTIFACT_COLUMNS)]`), and the detector now reports
`artifact_columns`, `artifact_column_names` and `artifact_per_column_separation`
explicitly, so the reported slice and the decided slice cannot drift apart. The defect
signature is computed on its own membership slice.

Z belongs in the artifact gate because the model input is local X/Y/Z plus membership. If
the marginal Z distribution could predict the class, the model could learn "CROWN trees
have a different Z distribution" instead of the spatial relation between XYZ and the
current membership.

No threshold was relaxed: `PERMUTATIONS = 2000`, `ALPHA = 0.01` and
`MIN_REAL_SEPARATION = 0.25` are unchanged, and the generator was not modified.

### Real diagnostic result

| Quantity | Value |
|---|---|
| XYZ artifact observed separation | **0.190446** |
| XYZ artifact critical value | 0.250 |
| XYZ artifact p-value | 0.0005 |
| XYZ artifact worst column | 2 (`feature_mean[2]`, mean Z) |
| XYZ artifact per-column | `[0.000194, 0.000064, 0.190446, 0.0, 0.0, 0.000021]` |
| XYZ artifact gate | **PASS** |
| membership defect signature | 0.453673 |
| REAL_DIAGNOSTIC_FINGERPRINT | `NONE_DETECTED` |

### Recorded property of the 0.190446 value

The value is above the permutation null but below the frozen 0.250 floor, so the gate
passes. For completeness, its origin was investigated and is reported rather than
explained away:

* the sampler is **provably class-agnostic** — for a fixed seed it selects *identical
  point indices* for every class, so `mean_z` cannot be driven by the membership channel;
* 120 independent draws of one identical scene give `mean_z` standard deviation 0.000476
  and range 0.0026, the same order as the between-class differences (~1e-3);
* `enumerate_specs` assigns each class a distinct deterministic seed range
  (`10007 * class_index`), so each class receives a different fixed set of RNG draws from
  the same scene.

The residual is therefore RNG-draw coupling, not geometry and not membership. It is far
below the artifact floor, it does not affect the membership-only gate, and it is recorded
here so the Supervisor can decide whether the class-indexed seed scheme should be
decoupled in a future round.

### New regression coverage

Fourteen end-to-end tests drive the real `fingerprint.analyse(...)` path — not the helper:

* A–F: a class-coded scalar implanted in **each** of the six XYZ columns
  (`feature_mean[0..2]`, `feature_std[0..2]`) makes `GENERATOR_FINGERPRINT = CRITICAL`
  and names the offending column;
* the artifact slice is asserted to be exactly `(0, 1, 2, 4, 5, 6)` and the running
  detector to report six artifact columns;
* a class signal in `feature_mean[3]` and in `feature_std[3]` is reported as the defect
  signature and does **not** fail the artifact gate;
* the membership-only shortcut gate is asserted to still be present with all of its
  sub-checks (`cross_partition`, `leave_one_candidate_out`, `zero/one count`,
  `any_membership_zero`);
* random label-independent features do not trigger the gate.

### Round 3 verification

| Check | Result |
|---|---|
| `pytest test_gate_d.py` | **123 passed** (+ 8 subtests) |
| `pytest test_canonical_identity.py` | **30 passed** |
| combined | **153 passed** (+ 8 subtests) |
| ruff on touched files | all checks passed |
| diagnostic regenerated from a clean process | 210 examples, `violations = []` |
| determinism (two clean runs) | summary, manifest **and fingerprint artifact** byte-identical |
| training lock | `TRAINING_ALLOWED = False`, no marker, trainers exit 3 |
| old digest rejected | `fcadab3e…` marker refused with `GATE_D_NOT_ACCEPTED` |
| new generator digest bound | `f3c1bd98…` read by `current_binding()` |

## Round 2 addendum — B4/B5 review gaps (`TREE_QC_GATE_D_V2_SUPERVISOR_REMEDIATION_B4_B5`)

An independent Supervisor review of `af4eb3a` re-opened **Blocker 4** and **Blocker 5**.
Both are now closed; blockers 1, 2, 3 and 6 were not touched.

### Blocker 4 — sampler extreme-strata boundary

`_select_within` funded each occupied stratum with `floor` points, and when the budget
could not cover every stratum it cleared `quotas` to zero **without resetting
`allocated`**. On the next line the funding loop tested `allocated >= budget`, which was
still the pre-clear total, so it broke immediately: every quota stayed zero and the group
contributed **no** structured points. The generic whole-scene refill in `sample()` then
silently replaced that group's budget, bypassing both the candidate/context split and the
within-group spatial/height structure.

Reproduced before the fix: 100 occupied strata with budget 32 returned **0** points
instead of 32; 500 strata with budget 64 likewise returned 0.

The fix resets `allocated` with the quotas and rebuilds it from the grants actually made.
`_select_within` now asserts its own contract — exactly `min(budget, pool.size)` distinct
indices from the pool, never exceeding a stratum's real size — and raises rather than
returning a short selection. The proportional path is unchanged: 6 strata × 1000 with
budget 4096 still yields 4096 points in the expected 683/682 split.

`SAMPLER_EXTREME_STRATA_TEST = PASS`
`CANDIDATE_CONTEXT_BUDGET_TEST = PASS`
`NO_DUPLICATED_SAMPLE_INDICES = YES`

**The branch is latent at the current diagnostic scale.** The minimum group budget across
the 210 examples is 489 while the maximum possible strata count (spatial cells × height
bands) is 344, so no diagnostic example reaches it. It is covered by direct
`_select_within` regression tests over four (strata, budget) pairs and an uneven-strata
layout, plus a `sample()`-level test that forces both groups past the boundary and
verifies the budgets survive. Reverting the one-line fix makes 8 of those tests fail.

### Blocker 5 — real-gate wiring

The controls verified `max_feature_class_separation`, but the real gate computed a
p-value on `feature_stats[:, 0]` only and never let the LOO accuracy decide anything. A
shortcut in any other column would have passed, and `detector_control PASS` did not imply
the real detector path ran.

The real gate now runs the **same** detector the controls verify: permutation-calibrated
max-feature separation over the exposed statistics. The critical value is the **99th
percentile of the permutation null of the same max statistic** — which already absorbs the
max-over-columns selection and the sample size — floored at `MIN_REAL_SEPARATION = 0.25`
so a statistically detectable but practically useless separation cannot fail the gate.
Controls are calibrated identically and now decide on `exceeds_calibrated_null`.

**Finding during implementation, reported rather than tuned away.** With the strict
all-column gate the real diagnostic returned `CRITICAL` (separation 0.454). I did not
lower the detector. Investigation showed the signal is the *injected defect itself*:

* per-column separations are `[0.0002, 0.0001, 0.190, 0.454, 0.0000, 0.0000, 0.0000,
  0.411]` — the position columns (local X/Y mean and std) are **clean**, and all class
  information sits in the membership and vertical channels;
* `feature_mean[3]` is the fraction of points the current segmentation claims and
  `feature_std[3]` is exactly its Bernoulli function (`corr = 1.000`, max deviation
  2.6e-05), so a mis-assignment defect necessarily moves both;
* a threshold fitted on other candidates transfers at chance (0.500), confirming the
  descriptive separation is not a deployable shortcut.

The gate therefore decides on the **artifact columns** — the position statistics the
corruption cannot legitimately move — while the membership and vertical statistics are
reported as the **defect signature**. This is not a sensitivity reduction: an implanted
class-coded scalar in a position column still fails the whole gate end-to-end (verified),
and a construction shortcut landing in a defect column remains visible in the reported
signature.

`FINGERPRINT_RANDOM_CONTROL = PASS`
`FINGERPRINT_IMPLANTED_CONTROL = PASS`
`FINGERPRINT_END_TO_END_IMPLANTED_REAL_STAT = PASS`
`REAL_DIAGNOSTIC_FINGERPRINT = NONE_DETECTED`
`CRITICAL_FINGERPRINTS = []`

### Round 2 verification

| Check | Result |
|---|---|
| `pytest test_gate_d.py` | **116 passed** (+ 8 subtests) |
| `pytest test_canonical_identity.py` | **30 passed** |
| combined | **146 passed** (+ 8 subtests) |
| ruff on touched files | all checks passed |
| diagnostic regenerated from a clean process | 210 examples, `violations = []` |
| determinism | two clean runs byte-identical |
| training lock | `TRAINING_ALLOWED = False`, no marker, trainers exit 3 |
| new generator digest bound | `fcadab3e…` read by `current_binding()` |

`SAMPLER_VERSION = GATE_D_STRUCTURED_SAMPLER_V3`, `FINGERPRINT_VERSION =
GATE_D_FINGERPRINT_V3`. The generator digest changed by design, as the task requires.

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
