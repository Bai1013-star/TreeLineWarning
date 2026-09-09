# Tree QC Gate A implementation

This directory currently implements **Gate A only**. No public-data, training-site,
held-out, local-challenge or deployment results exist. The four later-stage
launchers deliberately return exit code 3 until implementation and supervisor approval.

`SETUP_TREE_QC_TRAINING.cmd` creates the independent Python 3.11 environment on D:.
The supervisor selected the official torch 2.14.0+cu130 Windows wheel for this driver.
`RUN_TREE_QC_MINI_GATE.cmd` runs model tests and the frozen diagnostic, retaining raw
logs, JSON and checkpoints in a timestamped D:\TreeQCTraining directory.

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
