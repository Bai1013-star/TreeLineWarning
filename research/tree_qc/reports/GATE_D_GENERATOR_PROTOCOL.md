# Gate D synthetic generator protocol

Status: **worker protocol, pending GPT-6 Supervisor acceptance. Training remains locked.**
`GATE_D_GENERATOR_VERSION = GATE_D_SYNTHETIC_GENERATOR_V2`
`GATE_D_GENERATOR_DIGEST = f3c1bd98f74a9ff954e5455c306fb1db47a45a6eb4525dba99c1d587196d644a`

The digest is the SHA-256 over the generator implementation files listed in
`gate_d_diagnostic.GENERATOR_FILES`. The Supervisor's acceptance marker must carry this
exact digest; a different digest refuses to unlock training.

`GATE_D_PROTOCOL_DIGEST = 36e4cfca043190d4d12c2ab9ee635d92bb479cb9b56d482ae1121fd797939729`
`GATE_D_DIAGNOSTIC_MANIFEST_DIGEST = 626a6fc149733e9a1e58baf3febbd604bd7b0429649ca56ee3fe4412758deb7a`

Both artifacts are fully deterministic: re-running the diagnostic on the frozen inputs
reproduces the same file, so a rerun never leaves the worktree dirty. Wall-clock timing
is volatile run telemetry and is written to the gitignored
`GATE_D_DIAGNOSTIC_RUNTIME.json` instead of the frozen artifact.

## 1. Scope and non-goals

This protocol defines how synthetic tree-QC examples are built for the binary task
`PASS` vs `REVIEW_REQUIRED`. It does **not** define a model, a threshold, or a result.
Gate D was not accepted when this was written, so `TRAINING_ALLOWED = False` and no
PointNet or PointNet++ training run has been started.

## 2. Frozen inputs

Gate D consumes the frozen Gate C2 artifacts read-only:

| Input | Value |
|---|---|
| `CANONICAL_IDENTITY_PROTOCOL` | `GATE_C_CANONICAL_IDENTITY_PROTOCOL_V2` |
| identity manifest digest | `eb515773b5235f2cdb0206231493195f73f269c761d9da2d2186ba68ab3ed5aa` |
| provenance digest | `feab1423e6c9dc81ea05c4d02c56d5df987bb42c24d219f8089228ad06279688` |
| `TRAIN_UNIQUE` | 440 |
| `VALIDATION_UNIQUE` | 175 |
| `HELDOUT_FROZEN_COUNT` | 749 (`STRICT_HIGH_CONTAINMENT_QUARANTINE`) |
| high-containment quarantine | 25 trees, 28 alias files |
| complete-source leakage | 0 |

The generator re-derives the frozen invariants on load rather than trusting the files:
source-union disjointness, quarantine absence, non-empty surviving context, and the
three frozen counts. Any mismatch raises.

## 3. Shared scene construction

Every example — `GOOD` and every defect — is the **same object**: a `BaseScene` of real
publisher geometry whose `current_candidate_membership` channel is corrupted.

```
Gate C2 approved candidate tree
+ Gate C2 approved surviving context, clipped to the frozen 3 m query
= BaseScene (identical for every class of a candidate)
```

The frozen 3 m query is the Gate C2 rule verbatim: the candidate XY bbox expanded by
3 m with the candidate's native Z extent. A context tree contributes only the points
that actually fall inside that box, never its whole cloud. A scene whose context is
empty fails closed, because a `GOOD` example without real context would itself reveal
the class.

Consequences that are asserted by test:

* the scene point count is identical across all classes of one candidate;
* no coordinate is ever deleted or invented;
* `GOOD` always contains membership-0 context points, so the mere existence of a zero
  flag cannot identify the class.

## 4. Current candidate membership semantics

`current_candidate_membership` means **"the current single-tree segmentation selected
this point"**, never "this point is the answer".

| Class | Mechanism | Membership change |
|---|---|---|
| `GOOD` | publisher GT membership untouched | none |
| `MERGED_MULTI_TREE` | a whole neighbouring tree inside the query is claimed | 0 → 1 |
| `NEIGHBOR_CONTAMINATION` | a contiguous neighbouring region near the boundary is claimed | 0 → 1 |
| `NOISE` | scattered, non-structural context points are claimed | 0 → 1 |
| `OVER_SEGMENTED` | a contiguous part of the candidate is no longer selected | 1 → 0 |
| `CROWN_INCOMPLETE` | the contiguous crown region is no longer selected | 1 → 0 |
| `STEM_MISSING` | the contiguous stem/lower region is no longer selected | 1 → 0 |

Claimed neighbour points carry membership 1 even though their publisher GT source is a
neighbour, because the defect *is* the misassignment. Dropped candidate points carry
membership 0 while remaining in the scene, because the real points are still there; the
segmentation simply did not pick them. There is no class-specific refill of any kind.

`NOISE` and `NEIGHBOR_CONTAMINATION` share the same legal source (the frozen context)
but differ in structure: contamination claims a contiguous local region near the
candidate boundary, while noise claims at most one point per spatial cell, so it is
never contiguous. The public data cannot support a separate "background" source, so no
such source is invented.

`gt_source_id` records the publisher origin of each point and is **audit-only**; it is
never a model input. The model sees exactly four channels: local metric X, Y, Z and
`current_candidate_membership`. No class name, partition, tree id, filename, site,
corruption mask or donor identity reaches the tensor.

## 5. Source allowlist

Every point source — candidate, context, donor, background, refill, alias — passes one
unified authorization, and the candidate's legal external source set is exactly its
frozen surviving context.

An earlier revision invented a "background" source by searching for nearby trees outside
the frozen surviving context, which produced 72 unauthorized candidate→background
references. That path is deleted. `gate_d_geometry` now loads only the candidate and its
frozen context, and `gate_d_scene` clips context to the frozen query.

`assert_scene_sources` refuses: an unknown source, a high-containment source, a held-out
source, a cross-partition source, and any source outside the frozen allowlist. The
diagnostic independently rebuilds the source union from the scene and requires
`reported == reconstructed`; a mismatch is a hard failure.

## 6. Donor contract

The contract is an **allowlist**, not a denylist. `authorize_donor` checks, in order:

1. a tree may not donate to itself;
2. the donor must exist in the identity manifest;
3. high-containment quarantine is forbidden in every role;
4. held-out geometry is unreachable from a model-selection example;
5. a donor from the other model-selection partition is a leak;
6. the donor must appear in that candidate's own surviving `context`.

A **boundary-quarantine** tree (84 Litchfield trees whose bbox crosses the 3 m
train/validation boundary) passes 3–5 and is legal context. This distinction is
load-bearing: 70 of those trees are explicitly retained as context in the frozen
provenance, and the high-containment set is disjoint from them.

## 7. STEM_MISSING applicability

A missing-stem defect is only generated when the candidate shows observable, contiguous
lower evidence: at least 24 points in the bottom 45 % of its height range, within 0.75 m
of the trunk axis, spanning at least 1.5 m vertically. Otherwise the generator raises
`STEM_MISSING_NOT_APPLICABLE` and the example is **not** produced. The diagnostic
reports `stem_applicable`, `stem_rejected` and the rejection reasons.

## 8. Metric scale and the structured sampler

`sample` has no class or label parameter. The only normalisation is a translation to a
local origin plus division by one fixed research-wide constant
(`METRIC_SCALE_CONSTANT = 10.0`). There is **no** per-example RMS or unit-sphere scaling,
so a 12 m tree stays 12 m and examples remain comparable in metres.

Coverage guarantees, all class-independent:

* group budgets are **proportional** to each scene's own publisher GT split, which is
  identical for every class of a candidate. A fixed 60 % candidate quota was tried and
  rejected: it shifted the sampled membership fraction by up to +0.43;
* within each group, spatial cells and height bands receive a quota floor, so stem,
  crown and the candidate/context boundary are represented;
* changed-membership evidence is reserved inside those fixed budgets, never by
  duplicating points. If fewer changed points exist than the reservation, the shortfall
  is recorded and reported (`changed_evidence_shortfall`).

### 8.1 Extreme-strata boundary (`GATE_D_STRUCTURED_SAMPLER_V3`)

`_select_within` funds each occupied stratum with `floor` points, and when the budget
cannot cover every stratum it funds the largest strata first and leaves the rest at zero.
That fallback branch previously cleared `quotas` without resetting `allocated`, so the
funding loop broke on its first iteration, every quota stayed zero and the group
contributed **no** structured points at all. The generic whole-scene refill in `sample()`
then silently replaced that group's budget, bypassing both the candidate/context split and
the within-group spatial/height structure.

The fix resets `allocated` with the quotas and re-derives it from the grants actually
made. `_select_within` now asserts its own contract: it returns exactly
`min(budget, pool.size)` distinct indices drawn from the pool, never exceeding a
stratum's real size, and raises rather than returning a short selection.

The branch is **latent at the current diagnostic scale** — the minimum group budget is
489 while the maximum possible strata count (spatial cells × height bands) is 344, so no
of the 210 examples reaches it. It is covered by direct `_select_within` tests and by a
`sample()`-level test that verifies the group budgets survive.

## 9. Fingerprint diagnostics

`GATE_D_FINGERPRINT_V3` runs permutation tests (2000 permutations, α = 0.01) plus two
mandatory negative controls.

Checks: scene size constant per candidate, sampled size constant, drawn fraction
class-independent, corruption axis not class-constant, sampler shift class-independent,
tensor statistics not predictive, membership-only shortcut, identifier independence.

**The leak detector is honest and verified.** `loo_nearest_centroid` builds class
references *and standardisation statistics* from the other samples only, then predicts,
then compares. The random control (features independent of the label) must land near
chance, and the implanted control (a perfectly class-coded scalar) must be detected; if
either control misbehaves, the gate itself fails. Measured: random control separation
0.048, implanted shortcut separation 1.000.

### 9.1 Real-gate wiring

The controls and the real gate now run the **same** detector: the permutation-calibrated
max-feature class separation. The critical value is the **99th percentile of the
permutation null of the same max statistic**, which already accounts for the
max-over-columns selection and the sample size, floored at `MIN_REAL_SEPARATION = 0.25`
so that a statistically detectable but practically useless separation cannot fail the
gate on its own. Both the observed value and the calibrated critical value are reported.

An earlier revision tested only `feature_stats[:, 0]`, so a shortcut in any other column
would have passed, and it computed the LOO accuracy without letting it decide anything.

### 9.2 Artifact columns versus defect columns

The critical decision is taken on the **artifact columns**: all six marginal XYZ mean/std
statistics.

```
ARTIFACT_COLUMNS      = (0, 1, 2, 4, 5, 6)
ARTIFACT_COLUMN_NAMES = feature_mean[0] mean X
                        feature_mean[1] mean Y
                        feature_mean[2] mean Z
                        feature_std[0]  std X
                        feature_std[1]  std Y
                        feature_std[2]  std Z
```

The corruption re-labels points and never moves them, so **no** XYZ statistic — Z
included — may vary with the synthetic class. Z matters specifically because the model
input is local X/Y/Z plus membership: if the marginal Z distribution could predict the
class, the model could learn "this class has a different Z distribution" instead of
learning the spatial relation between XYZ and current membership. An earlier revision
admitted only X and Y here and grouped Z with membership, which was inconsistent with the
frozen corruption semantics.

The **defect columns** are the membership statistics alone:

```
DEFECT_COLUMNS      = (3, 7)
DEFECT_COLUMN_NAMES = feature_mean[3], feature_std[3]
```

The essence of every defect is a change to the current segmentation membership, so these
are *expected* to vary with the class and are reported as the defect signature rather than
failed: `feature_mean[3]` is the fraction of points the current segmentation claims and
`feature_std[3]` is exactly its Bernoulli function (measured correlation 1.000, max
deviation 2.6e-05).

Membership is not treated as unconditionally safe. The independent **membership-only
shortcut** gate still has to pass: it fits a threshold on one partition and applies it to
the other and reports **balanced** accuracy, because the diagnostic is deliberately
imbalanced (6 GOOD vs 36 defect examples per candidate); raw accuracy would report the
majority-class rate. Measured cross-partition balanced accuracy: 0.508 (chance 0.5).

Measured on the 210-example diagnostic:

| Quantity | Value |
|---|---|
| XYZ artifact observed separation | 0.190446 |
| XYZ artifact critical value | 0.250 |
| XYZ artifact p-value | 0.0005 |
| XYZ artifact worst column | 2 (`feature_mean[2]`, mean Z) |
| XYZ artifact per-column | `[0.000194, 0.000064, 0.190446, 0.0, 0.0, 0.000021]` |
| XYZ artifact gate | **PASS** (observed below the frozen 0.250 floor) |
| membership defect signature | 0.453673 (`[0.453673, 0.410516]`) |

`MIN_REAL_SEPARATION`, `ALPHA` and `PERMUTATIONS` were unchanged by this revision: the
gate passes on its own merits, not by relaxing the detector.

**Known property of the 0.190446 value, recorded for the Supervisor.** The sampler is
provably class-agnostic: for a fixed seed it selects *identical point indices* for every
class, so `mean_z` cannot be driven by the membership channel. The residual separation is
a consequence of `enumerate_specs` assigning each class a distinct, deterministic seed
range (`10007 * class_index`), so each class receives a different fixed set of RNG draws
from the same scene. 120 independent draws of one scene give `mean_z` standard deviation
0.000476, which is the same order as the between-class differences (~1e-3). The effect is
therefore RNG-draw coupling rather than geometry or membership, it is far below the
artifact floor, and it does not affect the membership-only gate.

## 10. Training lock

`gate_d_lock.TRAINING_ALLOWED = False`. Every trainer calls `require_training_allowed`
first and exits 3 with `GATE_D_NOT_ACCEPTED` unless `reports/GATE_D_ACCEPTANCE.json`
exists and **matches the current implementation**:

```json
{"gate_d": "ACCEPTED", "training_allowed": true,
 "generator_version": "GATE_D_SYNTHETIC_GENERATOR_V2",
 "generator_digest": "700bdca243...",
 "protocol_digest": "36e4cfca...",
 "accepted_by": "<supervisor>", "accepted_at": "<iso-8601>"}
```

All five fields are required. A missing marker, a missing field, `gate_d != ACCEPTED`,
`training_allowed != true`, a stale generator version, a stale generator digest or a
stale protocol digest all refuse with `GATE_D_NOT_ACCEPTED`. Changing the generator or
the frozen protocol after acceptance invalidates the marker and closes the lock again.
The marker is deliberately absent from this commit.

## 11. Held-out protection

Held-out is unreachable by construction: `DatasetConfig` refuses the partition,
`partition_of_candidate` raises, `authorize_donor` refuses, the scene allowlist excludes,
and `select_threshold` refuses any non-model-selection partition. The diagnostic reports
`held_out_examples = 0` and never loads a held-out tree.
