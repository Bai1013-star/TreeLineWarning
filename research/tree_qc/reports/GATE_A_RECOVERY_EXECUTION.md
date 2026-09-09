# Gate A recovery execution report (2026-09-09)

Worker result: WORKER_PASS_PENDING_SUPERVISOR. Public dataset download remains NO until independent review and supervisor approval. This is an execution diagnostic, not a tree defect model validation.

## Official wheel and resumable recovery

Current official GitHub release API reports v2.14.0 stable. Selected official index https://download.pytorch.org/whl/cu130/torch/ lists torch-2.14.0+cu130-cp311-cp311-win_amd64.whl, SHA256 00fb5440e7560b3b21e5e24567c6096fec4ef65b50a7ef1a105ebed5755eef7e. Primary index URL is https://download-r2.pytorch.org/whl/cu130/torch-2.14.0%2Bcu130-cp311-cp311-win_amd64.whl ; downloaded ranges used the equivalent official https://download.pytorch.org/whl/cu130/torch-2.14.0%2Bcu130-cp311-cp311-win_amd64.whl . Both returned HTTP 206 and total length 1,990,587,822.

The old tmp13s1vldg path no longer exists. Actual retained file was D:\TreeQCTraining\tmp\pip-unpack-97lmycbe\torch-2.14.0+cu130-cp311-cp311-win_amd64.whl at exactly 43,735,040 bytes, consistent with old pip log identity. ZIP signature is PK. Its first 1 MiB and final 64 KiB were compared against fresh official ranges, both equal. Its retained SHA256 is c8f0e139d9d40f9ef8aa541d213847a1e7fe0e4cffeeacea043d8709e855d463. No partial wheel was installed.

Four official range streams, 16 MiB chunks, preserved this prefix and independently persisted every incomplete chunk. 1,946,852,782 new bytes completed in 686.37 seconds including assembly/hash (2,836,449 B/s); 50 incomplete response events resumed at exact stored offsets, without restarting valid data. All completed chunk ETags matched. Full assembled SHA256 matches the official index, all ZIP CRCs passed, cp311-cp311-win_amd64 tag and Requires-Python were checked against Python 3.11.9 before local no-index installation. Dependencies were installed from the official PyTorch index. Network blockage is resolved.

Initial execution script at D:\TreeQCTraining\gate_a_recovery_20260909\download_resume.py enforced status/range/length, logged ETags, and validated final SHA/ZIP. The delivered resume_official_torch_20260909.py adds retained full-partial SHA verification, explicit per-response ETag matching, existing chunk length guards, reserved disk accounting and tag/Python assertions. Historical remote prefix/tail evidence is explicitly labeled; it is not claimed to be freshly recomputed on each invocation. Setup prefers an already matching installed torch or the verified local wheel, preventing a repeat full download.

## Actual precision failure and approved repair

The first original runtime stopped at MODEL_TEST_FAILED on batch independence; its failure logs are retained. Fixed seed 3 reproduced a PointNet 4096-point batch/single discrepancy: max absolute 1.4975667e-5 and relative 3.493807e-4. The first discrepancy occurs at encoder's second Conv2d (5.94258e-4), before the head. PointNet has no FPS or cdist. With identical saved parameters and input, disabling cuDNN and matmul TF32 reduced final discrepancy to 7.45058e-9 under the unchanged atol=1e-5/rtol=1e-4. PointNet++ seeds 0,3,101 had identical FPS/neighbor sets for batch vs single and default vs direct cdist.

Supervisor approved explicit IEEE FP32 policy, consistently applied in test and mini training/evaluation entry points via fp32_policy.py. AMP remains explicit float16 autocast. No model architecture, mini configuration, toy, epoch limit, targets or tolerances changed. Fixed seed 3 saved-case regression, all original model tests and the original Gate A passed. NumPy is not installed and torch emits its optional interoperability warning; this diagnostic does not use NumPy.

## Runtime result

Actual runtime: Python 3.11.9, torch 2.14.0+cu130, CUDA runtime 13.0, RTX 5070 Laptop, capability 12.0, arch list includes sm_120, driver 591.86. cuda:0 allocation/matmul/backward passed. Original model tests include both PointNet and PointNet++ at 2048/4096 with finite backward gradients.

PointNet mini loss 0.6822715104 -> 0.0287656975, accuracy 1.0, early stopped at epoch 4. PointNet++ loss 0.6909888536 -> 0.1491527297, accuracy 1.0, early stopped at epoch 3. Both checkpoint reload errors are exactly 0, resumed parameters changed, and optimizer steps increased exactly 1. Windows spawn executed with two workers, process count 3. AMP recorded 50 successful optimizer updates and 2 standard overflow skips; all 16/8/4/2/1 probes had two successful updates, safe batch 8, no OOM observed. Maximum stable means maximum tested, not a demonstrated hardware limit.

Gate duration 12.74 seconds. Maximum recorded allocator peak across calibration/mini reported here is 284,314,112 allocated / 434,110,464 reserved bytes from the calibration/mini records, not an external GPU device-wide peak. Sampled process-tree RAM high water was 3,425,959,936 bytes; parent OS high-water was 2,395,250,688 bytes. RAM sampling may miss between-log spikes. D free at completion 134,811,779,072 bytes.

Exact raw source/config hashes and model checkpoint paths/hashes are in runtime_ieee/mini_gate.json. Frozen config SHA256 remains b119fd3f242f94377e7c81c6ebaeff4411c127b1be7c381be698be92e1bc3251. All wheels, chunks, diagnostic saved-case and checkpoints remain on D; small review evidence is copied under gate_a_recovery_evidence. No product edits, public training data downloads or push.
