# Gate D/E combined engineering worker report

Worker: Claude Code (DeepSeek). **Worker report only; GPT-6 Supervisor decides.**
`START_HEAD = 2c02f849863b2139ac014989a6f7cf1f957f2cab`

This round built the merged Gate D + Gate E engineering layer and ran **only** the
bounded Gate D diagnostic. No model was trained, no held-out tree was read, no frozen
V0.1 or product file was touched, and nothing was pushed.

## 1. What was delivered

| Item | Module |
|---|---|
| A. synthetic generator | `gate_d_generator.py` |
| B. generator science + fingerprints | `gate_d_fingerprint.py` |
| C. structured sampler | `gate_d_sampler.py` |
| D. model-ready Dataset/DataLoader | `gate_d_dataset.py` |
| E. PointNet baseline trainer | `train_pointnet.py` + `gate_d_train.py` |
| F. PointNet++ SSG trainer | `train_pointnet2.py` + `gate_d_train.py` |
| G. validation/threshold infrastructure | `gate_d_metrics.py` |
| H. E/F/G orchestrator + launchers | `run_tree_qc_efg.py` + 6 `.cmd` files |
| — provenance gate, geometry cache, training lock | `gate_d_provenance.py`, `gate_d_geometry.py`, `gate_d_lock.py` |
| — tests | `test_gate_d.py` (65 tests) |

`GATE_D_GENERATOR_VERSION = GATE_D_SYNTHETIC_GENERATOR_V1`
`GATE_D_GENERATOR_DIGEST = 345953a6a1d541b1f632ecba93090026c4a4b9dce273387f0b06f42e138854b6`

## 2. Frozen inputs re-derived, not trusted

The gate reloads the Gate C2 artifacts and re-derives the invariants: protocol
`GATE_C_CANONICAL_IDENTITY_PROTOCOL_V2`, manifest digest
`eb515773b5235f2c…3ed5aa`, `TRAIN_UNIQUE = 440`, `VALIDATION_UNIQUE = 175`,
`HELDOUT_FROZEN_COUNT = 749`, source unions 482/207/851 pairwise disjoint,
high-containment quarantine 25 trees / 28 aliases. Any mismatch raises.

**One finding during implementation, corrected in my code, not in the artifacts.**
`partition == 'quarantine'` in the identity manifest is *not* the high-containment set.
It is 84 Litchfield trees whose bbox crosses the 3 m train/validation boundary. They may
not be candidates, but the frozen provenance explicitly retains 70 of them as context
for model-selection candidates. My first gate conflated the two and rejected legal
context; it now distinguishes `high_containment_quarantine` (25, forbidden in every
role) from `boundary_quarantine` (84, legal context, never a candidate). The frozen
artifacts were correct.

## 3. Donor contract

Allowlist, checked in order: self-donation; donor exists; high-containment forbidden in
every role; held-out unreachable; cross-partition leak; finally the donor must be in the
candidate's own surviving `context`. Every generated example ends with
`assert_example_sources(candidate, donors)`, so a violation raises rather than producing
a mislabelled example.

`DONOR_CONTRACT = ENFORCED_ALLOWLIST_FAIL_CLOSED`
`CROSS_PARTITION_LEAKAGE = 0` (210 examples, 58 distinct donors used)
`QUARANTINE_USAGE = 0`
`HELDOUT_USAGE = 0`

## 4. Generator science

Seven classes, three severities, real publisher geometry only.

| Class | Mechanism | Membership |
|---|---|---|
| `GOOD` | official observation untouched | 1 |
| `MERGED_MULTI_TREE` | real neighbouring trees moved in | mixed |
| `NEIGHBOR_CONTAMINATION` | real neighbouring boundary geometry | mixed |
| `OVER_SEGMENTED` | contiguous spatial subset of the same tree | 1 |
| `CROWN_INCOMPLETE` | contiguous crown removed, refilled with real non-member geometry | mixed |
| `STEM_MISSING` | contiguous stem column removed, refilled with real floor geometry | mixed |
| `NOISE` | real non-member geometry dominates | mostly 0 |

No class invents a coordinate; `test_no_class_uses_invented_gaussian_geometry` asserts
every generated point exists in the real geometry bundle.

**Two design decisions worth the Supervisor's attention.**

1. *Membership source.* The publisher's per-tree files and merged plot clouds are
   largely non-overlapping by exact cell (median match fraction 0, only 14.5 % of trees
   with `dominant_share == 1.0`). Plot-derived per-point membership is therefore not
   sound for tree examples. Gate D uses the publisher's own per-tree annotation:
   candidate points carry membership 1, context/background points carry 0.
2. *Missing regions are refilled.* `CROWN_INCOMPLETE` and `STEM_MISSING` do not leave a
   hole — a point-only model cannot see a hole, and a hole is not what a segmentation
   defect looks like. The vacated volume is refilled with real non-member geometry
   (floor/understorey/neighbour) carrying membership 0.

## 5. Structured sampler

One label-agnostic path for `GOOD` and every defect class; `sample` has no label
parameter (asserted by test). Features are exactly `local_x`, `local_y`, `local_z`,
`candidate_membership`. `POINTS_PRIMARY = 4096`, `POINTS_ROBUSTNESS = 2048`.

**A real defect was found and fixed here.** The first sampler used a hard per-cell cap,
which under-sampled a dense cluster: on a two-component fixture with an 11 % member
fraction, the sampled fraction collapsed to 0.7 %. Replacing the cap with a per-cell
floor over-corrected in the other direction (0.11 → 0.134). The shipped sampler uses
proportional largest-remainder quotas with neither cap nor floor, and reproduces the
source membership fraction to within ~1 % on clustered, spread and even fixtures, with
spatial coverage 0.79–0.96. Measured per-class sampler shift on the diagnostic:
≤ 5 × 10⁻⁴.

## 6. Fingerprint diagnostics

Eight permutation tests, 2000 permutations, α = 0.01.

| Check | p | Result |
|---|---:|---|
| `point_count_fingerprint_raw` | 0.343 | pass (reported, not critical) |
| `point_count_fingerprint_source` | 0.017 | pass |
| `point_count_fingerprint_sampled` | 1.000 | pass |
| `fixed_fraction_fingerprint` | 0.953 | pass |
| `fixed_axis_fingerprint` | 1.000 | pass |
| `class_specific_sampling_fingerprint` | 0.099 | pass |
| `metadata_leakage_tensor_stats` | 0.995 | pass |
| `metadata_leakage_identifiers` | 1.000 | pass |

`GENERATOR_FINGERPRINT = NONE_DETECTED`, `CRITICAL_FINGERPRINTS = []`.

**Two critical fingerprints were found during development and fixed, not excused.**

1. *Point-count leak.* `MERGED` appends whole neighbouring trees and `NOISE` prepends
   background geometry, so cloud size revealed the class. The generator now normalises
   every cloud to a size drawn uniformly from 30 000–70 000 points before sampling.
   Diagnostic normalised range 30 649–69 995; raw range 2 908–632 869.
2. *Sampler shift.* See §5 — a genuine sampler bias, fixed in the sampler.

One test was also corrected for measuring the wrong thing: `fixed_fraction_fingerprint`
now tests the **drawn** fraction (one shared distribution across classes and severities,
`GOOD` excluded as it has no corruption), because the realised magnitude necessarily
differs when a fraction maps onto a crown cut versus a boundary injection. The realised
magnitude is reported alongside.

## 7. Diagnostic scope and result

Bounded diagnostic: 5 train + 5 validation representative trees (evenly spaced by point
count), 7 classes × 3 severities = **210 examples**, 21 representative PNGs.

| Field | Value |
|---|---|
| `GATE_D_DIAGNOSTICS_CMD` | `01_RUN_GATE_D_DIAGNOSTICS.cmd` |
| examples | 210 (10 per class×severity) |
| violations | `[]` |
| `GENERATOR_FINGERPRINT` | `NONE_DETECTED` |
| `CRITICAL_FINGERPRINTS` | `[]` |
| `QUARANTINE_USAGE` | 0 |
| `HELDOUT_USAGE` | 0 |
| `CROSS_PARTITION_LEAKAGE` | 0 |
| `DETERMINISTIC` | yes (same spec → identical bytes) |
| runtime | 7.4 s |

Artifacts: `GATE_D_SYNTHETIC_DIAGNOSTIC.json`, `GATE_D_DIAGNOSTIC_MANIFEST.jsonl`
(210 rows, full per-example provenance), `GATE_D_GENERATOR_PROTOCOL.md`,
`GATE_D_WORKER_REPORT.md`, `gate_d_visuals/` (21 PNGs).

## 8. Training lock

`TRAINING_ALLOWED = FALSE`. Every trainer calls `require_training_allowed` first.

| Launcher | Behaviour now |
|---|---|
| `02_RUN_POINTNET_BASELINE.cmd` | exit 3, `GATE_D_NOT_ACCEPTED`, trains nothing |
| `03_RUN_POINTNET2_PILOT.cmd` | exit 3, `GATE_D_NOT_ACCEPTED`, trains nothing |
| `04_RUN_CROSS_SITE_EVAL.cmd` | dry-run plan, exit 3 |
| `05_RUN_MULTI_SEED_ROBUSTNESS.cmd` | dry-run plan, exit 3 |
| `RUN_TREE_QC_EFG_FULL_VALIDATION.cmd` | exit 3 without acceptance, `--dry-run` exits 0 |

Unlocking requires `reports/GATE_D_ACCEPTANCE.json` with `gate_d = ACCEPTED`,
`training_allowed = true` and the matching `generator_digest`. The marker is **not**
committed. Tests assert the lock is active, that a forged marker with a mismatched
digest is refused, and that both trainer entry points return 3.

`TRAINING_STARTED = NO`, `POINTNET_TRAINED = NO`, `POINTNET2_TRAINED = NO`.

## 9. Held-out protection

Held-out is unreachable by construction: `DatasetConfig` refuses the partition,
`partition_of_candidate` raises, `authorize_donor` refuses, the background allowlist
excludes, and `select_threshold` refuses any non-model-selection partition. The
diagnostic reports `held_out_examples = 0`.

## 10. Verification

| Check | Result |
|---|---|
| `python -m pytest test_gate_d.py` | **65 passed** |
| `python -m pytest test_canonical_identity.py` | **30 passed** (Gate C2 regression) |
| `python -m ruff check` on new files | all checks passed |
| `python -m compileall` on new files | clean |
| `git status` | clean after commit |

PointNet and PointNet++ were **not** trained, and `engine.train()` was never called.
The shared engine was verified without touching a dataset or a real model:

| Verification | Result |
|---|---|
| both models accept `(B, 4096, 4)` and `(B, 2048, 4)`, finite logits | pass |
| parameter counts | PointNet 76 770, PointNet++ SSG 333 506 |
| checkpoint save/load round trip, config + dataset digests preserved | pass |
| `_run_epoch` forward + backward + AMP scaler + optimizer step, train and eval, on synthetic tensors | pass, finite loss, scores in [0, 1] |
| `train_pointnet.py --dry-run`, `train_pointnet2.py --dry-run` | exit 0 |
| real training attempt | exit 3 `GATE_D_NOT_ACCEPTED` |

The Gate D/E layer needs NumPy in the training interpreter because the generator,
sampler and geometry cache are numpy-only by design; `gate_d_requirements.txt` records
that and NumPy 2.4.6 + pytest 9.1.1 were installed into the existing
`D:\TreeQCTraining\venv` (torch 2.14.0+cu130 untouched, CUDA available, RTX 5070 Laptop).
The test suite passes in both the analysis interpreter and the training interpreter
(95 tests each).

## 12. Verdict

`FINAL_WORKER_VERDICT = GATE_D_WORKER_READY_FOR_SUPERVISOR_REVIEW`

No leakage, no quarantine use, no held-out use, no critical fingerprint, deterministic
generation, full provenance per example, and the training lock is verified active.
Gate D acceptance remains the Supervisor's decision.
