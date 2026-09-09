# Gate D synthetic generator protocol

Status: **worker protocol, pending GPT-6 Supervisor acceptance. Training remains locked.**
`GATE_D_GENERATOR_VERSION = GATE_D_SYNTHETIC_GENERATOR_V2`
`GATE_D_GENERATOR_DIGEST = 363ca30e5d6ca83f80df4f9d893c602062f909c6c980405df7f17459ba47471b`

The digest is the SHA-256 over the generator implementation files listed in
`gate_d_diagnostic.GENERATOR_FILES`. The Supervisor's acceptance marker must carry this
exact digest; a different digest refuses to unlock training.

`GATE_D_PROTOCOL_DIGEST = 36e4cfca043190d4d12c2ab9ee635d92bb479cb9b56d482ae1121fd797939729`
`GATE_D_DIAGNOSTIC_DETERMINISTIC_DIGEST = 17d966950a657ed1476c6e73eaeda063f18fdc002d255d44c18505177ed0e24a`
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

## 9. Fingerprint diagnostics

`GATE_D_FINGERPRINT_V2` runs permutation tests (2000 permutations, α = 0.01) plus two
mandatory negative controls.

Checks: scene size constant per candidate, sampled size constant, drawn fraction
class-independent, corruption axis not class-constant, sampler shift class-independent,
tensor statistics not predictive, membership-only shortcut, identifier independence.

**The leak detector is honest and verified.** `loo_nearest_centroid` builds class
references *and standardisation statistics* from the other samples only, then predicts,
then compares. The random control (features independent of the label) must land near
chance, and the implanted control (a perfectly class-coded scalar) must be detected; if
either control misbehaves, the gate itself fails. Measured: random control separation
0.048 (limit 0.25), implanted shortcut separation 1.000 (minimum 0.90).

The **membership-only shortcut** is the metric the Supervisor asked for. It fits a
threshold on one partition and applies it to the other, and reports **balanced**
accuracy, because the diagnostic is deliberately imbalanced (6 GOOD vs 36 defect
examples per candidate). Raw accuracy would report the majority-class rate and look like
a shortcut when nothing was learned. Measured cross-partition balanced accuracy: 0.508.

## 10. Training lock

`gate_d_lock.TRAINING_ALLOWED = False`. Every trainer calls `require_training_allowed`
first and exits 3 with `GATE_D_NOT_ACCEPTED` unless `reports/GATE_D_ACCEPTANCE.json`
exists and **matches the current implementation**:

```json
{"gate_d": "ACCEPTED", "training_allowed": true,
 "generator_version": "GATE_D_SYNTHETIC_GENERATOR_V2",
 "generator_digest": "363ca30e...",
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
