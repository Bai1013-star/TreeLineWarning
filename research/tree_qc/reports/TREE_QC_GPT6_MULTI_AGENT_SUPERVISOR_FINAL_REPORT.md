# TREE_QC_GPT6_MULTI_AGENT_SUPERVISOR_FINAL_REPORT

Current recovery checkpoint: **Gate A PASS; Gate B PASS; public archive transfer blocked.** This supersedes the previous pre-download-stop report, preserved in SUPERVISOR_INITIAL_NETWORK_STOP_ARCHIVE.md. The official PyTorch network problem is resolved. No final model/scientific verdict is issued: the public pilot has not run, so none of the original scientific success/failure labels is supported yet.

## Supervisor acceptance

SUPERVISOR_MODEL = GPT-6 family, host configured
SUBAGENTS_USED = 4 this recovery (maximum2 active concurrently)
SUBAGENT_TASKS = gate_a_recovery: official wheel recovery/install/real CUDA and mini execution; dataset_metadata_recovery: official metadata/latest/license; public_data_pipeline: authorized two-site downloader and bounded inventory; gate_a_runtime_reviewer: new independent read-only runtime and data-stage reviewer
SUBAGENT_FAILURES = real initial batch-independence test failed under implicit TF32; diagnosed and repaired IEEE FP32 policy with original tolerance. Public archive HTTP504/timeouts remain external obstruction.
BASELINE_HEAD = 5f6f4d1fd5d79a3a922bf5924ae1b4e3d2b8f4d9
RECOVERY_START_HEAD = 39a46b8c93950f6190705d4f491b31331815898d
FINAL_HEAD = exact committed hash in D:\TreeQCTraining\FINAL_GIT_STATE.json, written after commit
BRANCH = codex/tree-qc-pointnet2-final-pilot-v1
WORKTREE = E:\TreeLineWarning-tree-qc-pointnet2-final-pilot-v1

GATE_A_CUDA_MINI = PASS (actual engineering gate)
GATE_B_DATA_LICENSE = PASS (pinned isolated research use)
GATE_C_DEDUP_SPLIT = NOT_RUN; needs downloaded verified archives
GATE_D_GENERATOR = NOT_RUN
GATE_E_TRAINING = NOT_RUN (public training; mini diagnostic passed)
GATE_F_HELDOUT = NOT_RUN
GATE_G_REPEATABILITY = NOT_RUN
GATE_H_LOCAL_CHALLENGE = NOT_RUN
GATE_I_DEPLOYMENT = NOT_RUN
PUBLIC_DATA_DOWNLOAD_ALLOWED = YES, exactly pinned Litchfield/Wytham and inventory; extraction still unapproved
EXECUTION_STATUS = PUBLIC_DATA_DOWNLOAD_NETWORK_BLOCKED

## Gate A evidence

DATA_ROOT = D:\TreeQCData
TRAINING_ROOT = D:\TreeQCTraining
D_FREE_BEFORE = approximately132.24 GiB at recovery start
D_FREE_AFTER = approximately125.55 GiB; exact final bytes in FINAL_GIT_STATE.json
GPU = NVIDIA GeForce RTX5070 Laptop,8151MiB,capability12.0
DRIVER = 591.86
PYTORCH = 2.14.0+cu130, actually installed in existing Python3.11.9 environment
PYTORCH_CUDA = 13.0, arch_list includes sm_120
AUTO_BATCH = both models16/8/4/2/1 tested, all stable; safety batch8.16 is maximum tested, not proven GPU limit; no OOM observed.
AMP = float16 forward/backward and GradScaler,50 effective updates plus2 initial overflow skips; every probe had2 real updates
PEAK_VRAM = largest recorded allocator allocated284314112 bytes/reserved434110464 bytes across probes/mini, not whole-card peak
PEAK_RAM = sampled process-tree working-set3425959936 bytes; parent OS high-water2395250688 bytes, not continuous process-tree peak

Official wheel1990587822 bytes: torch-2.14.0+cu130-cp311-cp311-win_amd64.whl. SHA25600fb5440e7560b3b21e5e24567c6096fec4ef65b50a7ef1a105ebed5755eef7e independently rehashed by supervisor and reviewer; ZIP CRC, CPython3.11 Windows tag and Requires-Python verified. Reused43735040 verified partial bytes, fetched1946852782 new bytes in686.37seconds including assembly/hash, about2.84MB/s useful throughput. Four official bounded range streams retained partial segments and recovered interrupted responses. No third-party mirror, driver upgrade, toolkit or project rebuild. SETUP reuse branch was actually rerun and exited0 without download.

POINTNET_PARAMETERS = 76770 (shared-MLP/max baseline without T-Net)
POINTNET2_PARAMETERS = 333506 (compact true PointNet++ SSG)
MODEL_SIZE_MB = float32 parameter payload0.30708 /1.334024 decimal MB; checkpoint files additionally hold optimizer/scaler state
POINTNET_RESULTS = diagnostic loss0.6822715104→0.0287656975,4epochs,32samples,final accuracy1.0
POINTNET2_RESULTS = diagnostic loss0.6909888536→0.1491527297,3epochs,32samples,final accuracy1.0

These toy accuracies do not demonstrate tree-QC scientific learnability. Both models used4096 XYZ+membership, real GPU updates, Windows2 spawned workers. CUDA allocation/matmul/backward and both models2048/4096 FP32 forward/backward passed. Gate mini runtime12.737seconds includes calibration and diagnostic, not deployment latency. Checkpoint reload logits error0; resumed parameters changed and optimizer steps16→17/12→13. Independent reviewer loaded checkpoints on CUDA, reevaluated all32 samples and reproduced final metrics exactly; saved optimizer steps16/12 equal epochs×4 batches.

Initial real test failure was not suppressed. Same saved parameters/input showed batch-single logit error1.49757e-5 with TF32,7.45e-9 under IEEE FP32. Supervisor approved explicit cuDNN/matmul FP32 policy in both evaluation and training entry points. FP16 AMP remains unchanged. All7 final source hashes matched actual runtime evidence. Frozen mini config SHA256 remains b119fd3f242f94377e7c81c6ebaeff4411c127b1be7c381be698be92e1bc3251: no model/toy/epoch/acceptance-threshold relaxation.

## Public data and remaining scientific work

PUBLIC_DATASET = TLS Forest Instance Segmentation Benchmark
DATASET_VERSION = official latest record16875688 at verification, DOI10.5281/zenodo.16875688; official latest-chain JSON retained
LICENSE = metadata.id cc-by-4.0, publisher citation requirement and official attribution terms verified; does not grant automatic product/weight redistribution approval or license upstream evaluation code
SITES = Litchfield/Wytham only
DOWNLOAD_SIZE = public archive0 bytes received; pinned archive total5498034689 bytes
UNIQUE_TREES = NOT_MEASURED
DUPLICATES_REMOVED = NOT_MEASURED

The official metadata endpoint recovered after earlier failures, so metadata/license are no longer the blocker. Archive endpoints remain a separate transfer problem. See PUBLIC_DATA_PIPELINE_WORKER_20260909.md and public_data_network_attempts.log for exact per-site routes, attempts and status; HTTP000 denotes curl received no HTTP response before timeout. No archive or point cloud was extracted. Source provenance JSON is metadata, not downloaded data.

Central-directory inspection requires verified size/MD5 and records local SHA256. Approved budget: exact two archives + up to40GiB expanded +10GiB training reserve, retaining at least35GiB actual D free. Extraction remains allow_extract=false until actual combined ZIP expanded size is reviewed. Bounds and offline safety tests do not substitute for real archive/schema/coordinate verification.

CROSS_SITE_A_TO_B = NOT_RUN
CROSS_SITE_B_TO_A = NOT_RUN
BAD_RECALL = NOT_MEASURED
BAD_FALSE_PASS_RATE = NOT_MEASURED
GOOD_FALSE_REVIEW_RATE = NOT_MEASURED
BALANCED_ACCURACY = NOT_MEASURED
F1_REVIEW = NOT_MEASURED
UNCERTAIN_RATE = NOT_MEASURED
MERGED = NOT_RUN
CONTAMINATION = NOT_RUN
OVERSEG = NOT_RUN
CROWN_INCOMPLETE = NOT_RUN
STEM_MISSING = NOT_RUN
MULTI_SEED = NOT_RUN
SAMPLING_ROBUSTNESS = NOT_RUN
LOCAL_BAD_TOTAL = 8 expected,0 evaluated
LOCAL_BAD_REVIEW = NOT_MEASURED
LOCAL_BAD_UNCERTAIN = NOT_MEASURED
LOCAL_BAD_FALSE_PASS = NOT_MEASURED
DOMAIN_GAP = UNKNOWN

Prior historical local-statistics exposure is retained in INITIAL_SUPERVISOR_AUDIT.md. No local images/point clouds/predictions were used in this recovery. Future challenge cannot be described as fully information-blind to this supervisor. No reliable local GOOD distribution exists. Formal public pipeline/dedup/context/sampler/generator, deterministic cross-process pilot resume, quantitative per-core-class support rules, and data-stack dependencies remain future gated work. NumPy is absent at this pure-Torch checkpoint and is not a Gate A failure.

## Product isolation, review and recovery

CPU_MS_PER_TREE = NOT_MEASURED
GPU_MS_PER_TREE = NOT_MEASURED
TOTAL_17_TREE_CPU_SEC = NOT_MEASURED
TOTAL_17_TREE_GPU_SEC = NOT_MEASURED
NO_FULL_SOURCE_RESCAN = YES
ASYNC_ARCHITECTURE = future contract only
INCREMENTAL_RECOMPUTE = future contract only
EXISTING_PERFORMANCE_OPTIMIZATION_PRESERVED = YES, no product-path changes
ONNX = NOT_RUN
ONNX_EQUIVALENCE = NOT_RUN
PRODUCT_INTEGRATION = NO
FROZEN_PRODUCT_CHANGED = NO
SCIENCE_CHANGED = NO
INDEPENDENT_REVIEWER_VERDICT = PASS_WITH_LIMITATIONS for Gate A runtime and stage evidence; data-module follow-up findings resolved/reviewed in the accompanying review record
PUSH = NO

NEXT_STEP = Resume the pinned official data download when archive service responds correctly. DOWNLOAD_TREE_QC_PILOT_DATA.cmd defaults to the saved two-site approval and preserves trustworthy .part progress. SETUP_TREE_QC_TRAINING.cmd now detects the installed matching torch and avoids download; Gate A need not be repeated unless its implementation/environment changes. After complete archive verification review expanded totals, then proceed Gate C onward. Do not relabel the current archive network problem as OFFICIAL_PYTORCH_DOWNLOAD_NETWORK_BLOCKED, PRE_DOWNLOAD_GATE_FAILED or model failure; PyTorch and Gate A succeeded.

Detailed evidence: GATE_A_RECOVERY_EXECUTION.md; GATE_A_INDEPENDENT_RUNTIME_REVIEW.md; GATE_A_RECOVERY_SUPERVISOR.md; DATA_LICENSE_AUDIT.md; pinned_pilot_manifest.json; PUBLIC_DATA_PIPELINE_WORKER_20260909.md; gate_a_recovery_evidence/.
