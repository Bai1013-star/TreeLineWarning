# Gate D synthetic generator protocol

Status: **worker protocol, pending GPT-6 Supervisor acceptance. Training remains locked.**
`GATE_D_GENERATOR_VERSION = GATE_D_SYNTHETIC_GENERATOR_V1`
`GATE_D_GENERATOR_DIGEST = ab62f3b9f3a0977d49b0ec988357e3663e641c2dd0b0fdfacb7e4622278c8741`

The digest is the SHA-256 over the generator implementation files listed in
`gate_d_diagnostic.GENERATOR_FILES`. The Supervisor's acceptance marker must carry this
exact digest; a different digest refuses to unlock training.

`GATE_D_DIAGNOSTIC_DETERMINISTIC_DIGEST = 9d62d0cc12454e73006b8e7312fe3c128aedc1855c7bce8bcb08be6aa797cdf2`
`GATE_D_DIAGNOSTIC_MANIFEST_DIGEST = 035bb077c1cb2160b6b3ef0fe89750f9557bfceb2b63efa9e9aaf02493e5dbd3`

Both artifacts are fully deterministic: re-running the diagnostic on the frozen inputs
reproduces the same file, so a rerun never leaves the worktree dirty. Wall-clock timing
is volatile run telemetry and is written to the gitignored
`GATE_D_DIAGNOSTIC_RUNTIME.json` instead of the frozen artifact.

## 1. Scope and non-goals

This protocol defines how synthetic tree-QC examples are built for the binary task
`PASS` vs `REVIEW_REQUIRED`. It does **not** define a model, a threshold, or a result.
Gate D was not accepted when this was written, so `TRAINING_ALLOWED = FALSE` and no
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

## 3. Donor contract

The contract is an **allowlist**, not a denylist. `authorize_donor` checks, in order:

1. a tree may not donate to itself;
2. the donor must exist in the identity manifest;
3. high-containment quarantine is forbidden in every role;
4. held-out geometry is unreachable from a model-selection example;
5. a donor from the other model-selection partition is a leak;
6. the donor must appear in that candidate's own surviving `context` in
   `GATE_C2_CONTEXT_PROVENANCE.json`.

A **boundary-quarantine** tree (84 Litchfield trees whose bbox crosses the 3 m
train/validation boundary) passes 3–5 and is legal context. This distinction is
load-bearing: 70 of those trees are explicitly retained as context in the frozen
provenance, and the high-containment set is disjoint from them. Conflating the two
would either wrongly reject legal context or wrongly admit a quarantined source.

Every generated example ends with `assert_example_sources(candidate, donors)`, so a
violation raises instead of producing a mislabelled example.

## 4. Geometry

The publisher ships per-tree files and merged plot clouds. Exact-cell cross-matching
between them is mostly empty for this benchmark (median match fraction 0), so
per-point membership cannot be derived from the plot clouds.

Gate D therefore uses the publisher's own per-tree annotation: candidate-tree points
carry membership 1, every point contributed by context or background geometry carries
membership 0. This is the official GT label for the tree instance, needs no cross-file
inference, and makes the fourth channel meaningful for exactly the defects that move
foreign geometry into a candidate.

Background geometry (used by `NOISE` and to refill vacated crown/stem volumes) is real
publisher geometry from **allowed** trees of the same site: not the candidate, not its
context, not high-containment, not held-out. The search radius escalates
deterministically (12 → 20 → 35 → 50 m) so an isolated tree still receives real
geometry instead of failing. All train and validation candidates have a non-empty
background allowlist, which is asserted by test.

Geometry is cached per candidate under a key that includes both frozen digests, the
context list and the background allowlist. Cache hits cost no plot scan; cache misses
read only the authorised tree files. No per-sample scan of the 105 M-point source is
possible.

## 5. Corruption classes

| Class | Mechanism | Membership | Label |
|---|---|---|---|
| `GOOD` | official observation untouched | 1 for all | `PASS` |
| `MERGED_MULTI_TREE` | real neighbouring trees moved in | mixed | `REVIEW_REQUIRED` |
| `NEIGHBOR_CONTAMINATION` | real neighbouring boundary geometry added | mixed | `REVIEW_REQUIRED` |
| `OVER_SEGMENTED` | spatially contiguous subset of the same tree | 1 for all | `REVIEW_REQUIRED` |
| `CROWN_INCOMPLETE` | contiguous crown region removed, region refilled with real non-member geometry | mixed | `REVIEW_REQUIRED` |
| `STEM_MISSING` | contiguous stem/lower column removed, column refilled with real floor geometry | mixed | `REVIEW_REQUIRED` |
| `NOISE` | real non-member geometry dominates the cloud | mostly 0 | `REVIEW_REQUIRED` |

`CROWN_INCOMPLETE` and `STEM_MISSING` refill the vacated volume on purpose: a
point-only model cannot see a hole, and a hole is not what a segmentation defect looks
like. What the tree loses, the plot actually contains.

No class invents a coordinate. `test_no_class_uses_invented_gaussian_geometry` asserts
that every generated point exists in the real geometry bundle.

Severity `LOW`/`MEDIUM`/`HIGH` multiplies the drawn corruption fraction by 0.55/1.0/1.65
and is capped at 0.72. The fraction itself is drawn from one shared uniform range
(0.06–0.34) for every class and severity; only the geometry mapping differs.

## 6. Point-count normalisation

Cloud size is normalised to a size drawn uniformly from 30 000–70 000 points before
sampling. Without this, cloud size alone reveals the class: `MERGED` appends whole
neighbouring trees, `NOISE` prepends background geometry, and `OVER_SEGMENTED` drops a
region. A cloud below the target is upsampled by repeating real points from the same
uniform draw, which introduces no synthetic coordinates.

## 7. Structured sampler

One label-agnostic code path serves `GOOD` and every defect class; `sample` has no
label parameter and a test asserts its signature.

* features: `local_x`, `local_y`, `local_z`, `candidate_membership` (exactly 4);
* local normalisation is a similarity transform (centroid centre, one isotropic scale);
* `POINTS_PRIMARY = 4096`, `POINTS_ROBUSTNESS = 2048`;
* selection is spatially stratified over the local XY footprint with **proportional**
  largest-remainder quotas.

Both a per-cell cap and a per-cell floor were tried and rejected: a cap
under-samples a dense cluster (measured membership fraction 0.11 → 0.007), and a floor
over-samples whichever class occupies more cells (0.11 → 0.134). Proportional
allocation reproduces the source membership fraction to within ~1 % for clustered,
spread and even two-component fixtures, while keeping spatial coverage at 0.79–0.96.
A class-dependent sampler shift would be a construction artefact; the measured
per-class mean shift is now ≤ 5 × 10⁻⁴.

## 8. Fingerprint diagnostics

Eight permutation tests (`GATE_D_FINGERPRINT_V1`, 2000 permutations, α = 0.01) run over
the generated examples:

1. `point_count_fingerprint_raw` — pre-normalisation size (reported, not critical;
   `MERGED`/`NOISE` legitimately append geometry);
2. `point_count_fingerprint_source` — post-normalisation size must be
   class-independent;
3. `point_count_fingerprint_sampled` — every example must reach the model at exactly
   the fixed size;
4. `fixed_fraction_fingerprint` — the **drawn** fraction must be class-independent
   (`GOOD` excluded: it has no corruption by construction);
5. `fixed_axis_fingerprint` — the corruption direction must not be class-constant;
6. `class_specific_sampling_fingerprint` — the sampler-induced shift of the membership
   fraction must be class-independent (the absolute fraction is the signal, not an
   artefact);
7. `metadata_leakage_tensor_stats` — a nearest-centroid probe on the 8 per-channel
   summary statistics of the exposed tensor must not beat chance beyond permutation;
8. `metadata_leakage_identifiers` — every candidate and severity must carry the same
   class set, so an identifier cannot stand in for the label.

Any critical fingerprint sets `GENERATOR_FINGERPRINT = CRITICAL`, adds
`critical_generator_fingerprint` to `violations`, and forces a HOLD.

## 9. Training lock

`gate_d_lock.TRAINING_ALLOWED = False`. Every trainer calls
`require_training_allowed` first and exits 3 with `GATE_D_NOT_ACCEPTED` unless
`reports/GATE_D_ACCEPTANCE.json` exists with:

```json
{"gate_d": "ACCEPTED", "training_allowed": true,
 "generator_digest": "b36cd620d7...", "accepted_by": "<supervisor>",
 "accepted_at": "<iso-8601>"}
```

The marker is deliberately absent from this commit. Tests assert that the lock is
active, that a forged marker with a mismatched digest is refused, and that both trainer
entry points return 3 without it.

## 10. Held-out protection

Held-out is unreachable by construction, not by convention:

* `DatasetConfig` refuses the `held_out` partition;
* `partition_of_candidate` raises `HeldOutAccessError` for any held-out tree;
* `authorize_donor` refuses held-out donors;
* the background allowlist excludes held-out trees;
* the diagnostic reports `held_out_examples = 0` and never loads a held-out tree.

Threshold selection additionally refuses any partition that is not `train` or
`validation`, so held-out can never influence the operating point, a hyper-parameter or
a model choice.
