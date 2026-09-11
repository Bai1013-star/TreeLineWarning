# TREE_QC_PUBLIC_PILOT_PROTOCOL_V1

Gate D is accepted. This document records, **before the public pilot is launched**, what
the E/F/G round executes, what it may not do, and how it fails. It is the human-readable
companion to `PUBLIC_EXPERIMENT_FREEZE.json`; the freeze is data, this is the contract.

```
BASE_ACCEPTED_HEAD   = 1c4ed5b96e4ef1053bf9d0e47d597911ffbca727
GENERATOR_VERSION    = GATE_D_SYNTHETIC_GENERATOR_V2
GENERATOR_DIGEST     = a0550c17ff75cdf7d380fc677840c883f2262f2148a40edbbee0df35410b6b9d
PROTOCOL_DIGEST      = 36e4cfca043190d4d12c2ab9ee635d92bb479cb9b56d482ae1121fd797939729
GATE_D_ACCEPTANCE    = PASS   (research/tree_qc/reports/GATE_D_ACCEPTANCE.json)
GATE_E_ALLOWED       = YES
```

## 1. Entry point

```
RUN_TREE_QC_PUBLIC_PILOT.cmd        (double-click; Windows)
    -> python run_tree_qc_efg.py --auto
```

The launcher only locates a verified interpreter with a working PyTorch, sets UTF-8, runs
the orchestrator, prints the final status and pauses. It installs nothing, upgrades
nothing, deletes nothing and changes no system setting. The orchestrator is a resumable
state machine; the Python entry point is authoritative.

Other modes, all read-only apart from `--auto`:

```
--dry-run        resolved plan, no GPU work
--status         progress from CURRENT_STATUS.json
--resume-test    proves the resume, staleness and freeze guards
--smoke-test     bounded end-to-end rehearsal (never a scientific result)
```

## 2. Question this round answers

> Can PointNet / PointNet++ reliably identify a single-tree segmentation result that needs
> human review?

Not "can deep learning redo single-tree segmentation". The model input is
`local_x, local_y, local_z, current_candidate_membership` over real publisher geometry,
and the output is `PASS` vs `REVIEW_REQUIRED`.

## 3. Frozen artifacts

Nothing participating in the accepted generator digest is modified. The frozen files are:

```
gate_d_provenance.py  gate_d_scene.py     gate_d_geometry.py  gate_d_generator.py
gate_d_sampler.py     gate_d_dataset.py   gate_d_metrics.py   gate_d_train.py
gate_d_fingerprint.py gate_d_diagnostic.py   reports/GATE_C_CANONICAL_IDENTITY_PROTOCOL_V2.md
```

`S02_GATE_D_ACCEPTANCE` re-computes the runtime generator and protocol digests and compares
them with the accepted values. Any difference is `STOP_GATE_D_BINDING_MISMATCH`; the
acceptance is never updated to a new digest automatically.

The EFG round adds new modules only:

```
efg_gates.py      frozen gate constants and decision rules
efg_state.py      atomic, digest-verified run state
efg_resources.py  environment fingerprint, disk guard, batch calibration
efg_specs.py      partition-agnostic spec arithmetic (parity-proven against the frozen one)
efg_cache.py      byte-verified per-example feature cache
efg_dataset.py    model-selection source with cache and preload
efg_heldout.py    held-out adapter, authorization and the parity gate
efg_freeze.py     architecture/seed selection and PUBLIC_EXPERIMENT_FREEZE
efg_evaluate.py   scoring, per-defect breakdown, robustness
efg_leakage.py    held-out protocol audit and the local-challenge boundary
efg_trainer.py    guarded training loop (non-finite and OOM stops, resume rules)
efg_reports.py    TREE_QC_EFG_FINAL_REPORT and stop reports
run_tree_qc_efg.py  the state machine
test_efg_orchestrator.py  the guards' tests
```

### A frozen-path defect and how it is handled

`gate_d_dataset.collate` expects `(features, label, index)` triples while the frozen
`TorchGateDDataset.__getitem__` yields `(features, label)` pairs, so `make_loader` — and
therefore the frozen epoch loop — cannot complete a batch. Rebuilding the frozen file would
change the accepted generator digest, so `efg_trainer._loader` supplies a collate that
mirrors the frozen one and converts the index with the same `torch.stack` semantics. Only
the data-transport adapter is replaced; the model, the loss, the optimizer, the sampler and
the dataset are the frozen ones.

## 4. Stage sequence

| # | Stage | What it does |
|---|-------|--------------|
| 1 | `S01_ENVIRONMENT` | CPU/RAM/GPU/VRAM/torch/CUDA/driver/disk fingerprint; disk guard 35 GiB |
| 2 | `S02_GATE_D_ACCEPTANCE` | re-verify the accepted generator and protocol digests; load frozen provenance |
| 3 | `S03_RESOURCE_CALIBRATION` | batch probe `16, 8, 4, 2, 1`; safe = half the largest stable; `SAFE_COMMON_BATCH = min(pointnet, pointnet2)` |
| 4 | `S04_DATASET_FINGERPRINT` | provenance and identity digests, spec-list digests, declared class balance, held-out integrity re-derivation |
| 5 | `S05_ADAPTER_PARITY` | frozen pipeline vs held-out adapter on real validation candidates, all 7 classes × 3 severities, at 4096 and 2048 |
| 6 | `S06_POINTNET_SEED101` | PointNet baseline |
| 7 | `S07_POINTNET2_SEED101` | PointNet++ SSG |
| 8 | `S08_LEARNABILITY` | validation-only floor check; failure stops before held-out |
| 9 | `S09_ARCHITECTURE_SELECTION` | pre-declared validation-only rule |
| 10 | `S10_MODEL_SEEDS` | seeds 101, 202, 303; seed 101 is re-used if digests match |
| 11 | `S11_THRESHOLD_FREEZE` | one validation threshold per seed under the frozen rule |
| 12 | `S12_DEPLOYMENT_CANDIDATE` | validation-only deployment seed choice |
| 13 | `S13_SAMPLING_PROBE` | validation-only 2048 probe |
| 14 | `S14_PUBLIC_EXPERIMENT_FREEZE` | the scientific lock, written once with its own SHA-256 |
| 15 | `S15_HELDOUT_BLIND_EVALUATION` | first and only opening of held-out, per seed, at the frozen thresholds |
| 16 | `S16_MULTI_SEED_AGGREGATION` | mean/min/max/std across seeds; stability verdict |
| 17 | `S17_SAMPLING_ROBUSTNESS` | 4096 primary plus 2048 × three pre-declared sampling seeds |
| 18 | `S18_FINAL_REPORT` | `TREE_QC_EFG_FINAL_REPORT.json` / `.md` |

Stages 6–7 run under identical train candidates, validation candidates, examples, sampler,
points (4096), corruption, budget, effective batch, threshold rule and early-stopping rule.
The only difference is the network.

## 5. Declared constants

```
epochs 30              early-stop patience 8, min-delta 1e-4 on validation BALANCED_ACCURACY
learning rate 1e-3     weight decay 1e-4      AMP float16 when CUDA
points 4096 primary    2048 robustness        grad clip 1.0        workers 0
model seeds 101, 202, 303                     comparison seed 101
good-false-review budget 0.25                 uncertain band ±0.10
robustness sampling seeds 11, 23, 37          disk guard 35 GiB
max_train_candidates = None                   max_validation_candidates = None
```

`max_train_candidates` and `max_validation_candidates` are `None` for the formal run; the
CLI refuses to start a bounded run outside `--smoke-test`.

Loader workers are `0` because it was measured: with per-candidate geometry preloaded, the
loader reaches 12.4 examples/s with `workers=0` against 6.8 with 2 and 4.1 with 4, since
every spawned worker reloads the whole geometry set. The measurement is recorded rather than
assumed.

## 6. Threshold and selection rules

**Validation threshold, per seed** — `gate_d_metrics.select_threshold`, frozen:
maximise `BAD_RECALL` subject to `GOOD_FALSE_REVIEW <= 0.25`, on validation only.
`select_threshold` raises for any partition outside `train`/`validation`, so a held-out
threshold is impossible by construction, not by discipline.

**Architecture** — validation only, deterministic:

1. eligible = `GOOD_FALSE_REVIEW <= 0.25`;
2. higher `BAD_RECALL`;
3. then higher `BALANCED_ACCURACY`;
4. within 0.01 on both, the simpler `pointnet` (pre-declared non-inferiority);
5. lowest seed as the final tie-break.

If no architecture satisfies the budget: `STOP_ARCHITECTURE_BUDGET_UNSATISFIED`.

**Deployment seed** — validation only: budget satisfied → max `BAD_RECALL` → max
`BALANCED_ACCURACY` → lowest seed. Recorded before unblinding; never revised afterwards.

## 7. Held-out discipline

Held-out is unreachable by construction in the frozen pipeline:
`gate_d_provenance.authorize_donor` refuses a held-out donor, `assert_scene_sources` raises
the moment a held-out source reaches a model-selection example, `ExampleSpec` refuses a
non-model-selection partition and `DatasetConfig` does the same. None of those is weakened.

The held-out route is a **separate, explicitly labelled adapter**:

* model-selection examples are still produced only by the frozen `generate_example` through
  the frozen `GateDDataset`;
* held-out examples are produced by `efg_heldout.generate_heldout_example`, whose
  corruption body is a faithful port that differs in exactly one respect — which authorized
  context a candidate may draw real geometry from. It reuses the frozen RNG material, the
  frozen helper functions and the frozen sampler;
* the port is **proven**, not trusted. `S05` runs genuinely frozen validation scenes through
  the shared corruption body and requires identical spec key, corruption seed, sampling
  seed, label, provenance label, corruption class, severity, mechanism, changed-membership
  count and fraction, pre-reservation and final sampler index digests, and bit-identical
  feature tensors. Drift is `STOP_HELDOUT_ADAPTER_DRIFT` and nothing is unblinded.

A held-out scene may use only the candidate's own points and the trees in its own frozen
held-out context. Held-out source unions are disjoint from the train/validation source
unions (verified: 0 overlap), quarantine trees stay forbidden in every role, and the
runtime candidate count must equal the count recorded in the freeze
(`STOP_HELDOUT_MANIFEST_CHANGED` otherwise).

`S15` refuses to start unless a valid freeze exists, all three seeds have a live
digest-verified checkpoint and a frozen threshold, the parity gate passed, and the manifest
count matches. Unblinding writes an irreversible audit with the first-access timestamp and
the freeze digest. From that point every pre-unblind stage refuses to run and any change to
the freeze is `STOP_FREEZE_MUTATED_AFTER_UNBLIND`.

## 8. Metrics

Binary `PASS` vs `REVIEW_REQUIRED` is the primary task. The five reported metrics are
`BAD_RECALL`, `BAD_FALSE_PASS`, `GOOD_FALSE_REVIEW`, `BALANCED_ACCURACY` and `F1_REVIEW`,
with `F1_REVIEW` computed in the reporting layer so the frozen metric core is untouched.
`UNCERTAIN` is reported alongside and never counted as a detected defect.

The engineering gate on held-out moderate/high defects:

```
BAD_RECALL >= 0.85   BAD_FALSE_PASS <= 0.15
GOOD_FALSE_REVIEW <= 0.25   BALANCED_ACCURACY >= 0.80
```

Per-defect reporting covers `MERGED_MULTI_TREE`, `NEIGHBOR_CONTAMINATION`,
`OVER_SEGMENTED`, `CROWN_INCOMPLETE`, `STEM_MISSING` and `NOISE`, all severities and the
moderate/high subset separately. `LOW` is reported but never substitutes for the main gate.

Multi-seed: mean/min/max/std across the three frozen seeds; a `BAD_RECALL` spread above
0.10 is `MODEL_UNSTABLE`. Robustness: a `BAD_RECALL` drop above 10 percentage points
against the 4096 primary is `SAMPLING_TOO_SENSITIVE`.

## 9. Failure behaviour

Every stop writes the state, a stop report and a clear message; none of them continues by
changing the experiment.

```
STOP_GATE_D_BINDING_MISMATCH      STOP_ACCEPTANCE_INVALID
STOP_ENVIRONMENT_NOT_READY        STOP_DISK_GUARD
STOP_BATCH_CALIBRATION            STOP_DATASET_FINGERPRINT_CHANGED
STOP_RUNTIME_OOM_AFTER_FREEZE     STOP_NONFINITE_TRAINING
STOP_CHECKPOINT_CORRUPTION        STOP_CONFIG_MISMATCH
STOP_RESUME_MISMATCH              STOP_HELDOUT_BEFORE_FREEZE
STOP_HELDOUT_ADAPTER_DRIFT        STOP_HELDOUT_MANIFEST_CHANGED
STOP_FREEZE_MUTATED_AFTER_UNBLIND STOP_CRITICAL_LEAKAGE
```

A non-finite loss or logit stops immediately; the last valid checkpoint is kept and
training is not resumed into a polluted state. A real OOM at the frozen safe batch is a
stop, not a smaller batch.

The pilot never adds epochs, enlarges a model, re-picks a threshold, re-selects a
checkpoint, reopens training, downloads data, or introduces another architecture. It
automates execution, not scientific decision-making.

## 10. Resumability

State lives in `D:\TreeQCTraining\public_pilot_v1\state\AUTOPILOT_STATE.json`, written
atomically (temp → replace + fsync) and carrying its own SHA-256. A stage is `COMPLETE`
only while every digest that produced it still matches; otherwise it is `STALE` and is
recomputed. A stale or corrupt checkpoint, a changed config, dataset, or provenance digest
all refuse re-use. A second concurrent writer is refused by a pid lock.

Directory layout under the run root:

```
state/  logs/  runs/  freeze/  evaluation/  robustness/  reports/  cache/
```

`CURRENT_STATUS.json` mirrors the progress so a user can watch or restart at any time.

## 11. Boundaries

```
LOCAL_8_BAD_CHALLENGE_ACCESSED = NO
ONNX_STARTED                   = NO
PRODUCT_INTEGRATION            = NO
```

No local challenge geometry, label, point cloud or model output is read this round; the
scan is enforced in code (`efg_leakage.assert_local_challenge_untouched`) and run in `S18`.
No ONNX export, no `TreeRepairEngine`, no local resegmentation, no UI integration, no
PyInstaller bundle. The product direction is frozen as
`single-tree segmentation → AI QC → (future) one local resegmentation → second QC → human
review`, with `AUTO_RESEGMENT_MAX_ATTEMPTS = 1`, and only the output interface is reserved.

## 12. Verdicts

This phase may report exactly one of:

```
PUBLIC_PILOT_NOT_LEARNABLE
MODEL_UNSTABLE
SAMPLING_TOO_SENSITIVE
READY_FOR_GATE_H_LOCAL_BLIND
STOP_<CONCRETE_BLOCKER>
```

`READY_FOR_GATE_H_LOCAL_BLIND` requires all of: the four primary held-out targets met,
acceptable multi-seed stability, sampling recall drop within 10 percentage points, no
critical leakage, a clean held-out protocol, and meaningful `MERGED_MULTI_TREE` /
`NEIGHBOR_CONTAMINATION` / `OVER_SEGMENTED` performance. `PRODUCT_READY` is not in the
vocabulary of this phase.
