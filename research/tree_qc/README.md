# Tree QC Gate A implementation

This directory implements **Gate A** (runtime/model diagnostic), the **Gate C2** frozen
canonical identity and provenance, and the **Gate D/E engineering layer**. No formal
model training has been run and no held-out or local-challenge result exists.

`SETUP_TREE_QC_TRAINING.cmd` creates the independent Python 3.11 environment on D:.
The supervisor selected the official torch 2.14.0+cu130 Windows wheel for this driver.
`RUN_TREE_QC_MINI_GATE.cmd` runs model tests and the frozen diagnostic, retaining raw
logs, JSON and checkpoints in a timestamped D:\TreeQCTraining directory.

## Gate D/E engineering layer

`01_RUN_GATE_D_DIAGNOSTICS.cmd` runs the bounded synthetic-generator diagnostic. The
launchers `02_RUN_POINTNET_BASELINE.cmd`, `03_RUN_POINTNET2_PILOT.cmd`,
`04_RUN_CROSS_SITE_EVAL.cmd`, `05_RUN_MULTI_SEED_ROBUSTNESS.cmd` and
`RUN_TREE_QC_EFG_FULL_VALIDATION.cmd` are implemented but **refuse to train**: every one
checks `research/tree_qc/reports/GATE_D_ACCEPTANCE.json` and exits 3 with
`GATE_D_NOT_ACCEPTED` while Gate D is unaccepted. The acceptance marker is deliberately
not committed.

The layer is split so the generator never imports a model:

| Module | Role |
|---|---|
| `gate_d_provenance.py` | frozen-split loader, donor allowlist, fail-closed authorization |
| `gate_d_geometry.py` | bounded geometry cache; authorised tree reads only |
| `gate_d_generator.py` | 7 classes x 3 severities, real geometry only |
| `gate_d_sampler.py` | label-agnostic 4096/2048 structured sampler |
| `gate_d_dataset.py` | `ExampleSpec -> Generator -> Sampler -> Tensor`, numpy only |
| `gate_d_metrics.py` | confusion matrix, five named metrics, frozen thresholds |
| `gate_d_train.py` | shared engine: AMP, checkpoint, resume, best, device |
| `train_pointnet.py`, `train_pointnet2.py` | the two locked training entry points |
| `gate_d_fingerprint.py` | construction-artefact permutation tests |
| `gate_d_lock.py` | `TRAINING_ALLOWED = False` |
| `run_tree_qc_efg.py` | the future 8-step E/F/G run, dry-run only for now |

`GATE_D_GENERATOR_PROTOCOL.md` records the contract; `GATE_D_WORKER_REPORT.md` records
the diagnostic result.

The PointNet baseline is shared pointwise MLP + symmetric max pooling, **without
T-Net**; it is not an exact full paper reproduction. PointNet++ SSG has true FPS,
metric radius neighborhoods, local relative XYZ and learned local shared MLPs at
two abstraction scales. Membership is the fourth input channel. Neighborhood
padding repeats the closest valid point; FPS centers always guarantee one valid
point. This compact pilot uses 128 and 32 centers; no custom CUDA or C++ extension.

The toy consists of 32 balanced narrow/broad ellipsoids with the same membership
fraction. It deliberately checks only execution/optimization; it cannot establish
that tree segmentation defects are learnable. Frozen limits are in
`mini_gate_config.json`. No epoch extension or toy replacement is automatic.

AMP overflow causes GradScaler's standard skipped update and is recorded. Resume
requires exact float32 reload logits, changed parameters and optimizer step +1.
Every proposed CUDA batch (16/8/4/2/1) requires two successful optimizer updates in at most eight AMP attempts;
OOM is caught and logged, never invented. Safe batch is half the maximum stable.

RAM is the working-set sum of the parent and spawn descendants, sampled at log
boundaries; it may miss between-sample spikes. The parent OS high-water working set
is also recorded. GPU peaks use PyTorch allocator stats; they exclude driver and
other processes. Neither value is represented as a system-wide continuous peak.

Only the supervisor may grant scientific gate approval. A worker diagnostic pass
keeps `public_download_allowed=false`.
