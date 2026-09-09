# RESEARCH_PLAN_V1 — frozen before implementation and public download

Baseline: 5f6f4d1fd5d79a3a922bf5924ae1b4e3d2b8f4d9
Branch: codex/tree-qc-pointnet2-final-pilot-v1
Worktree: E:\TreeLineWarning-tree-qc-pointnet2-final-pilot-v1
Supervisor owns architecture, scientific gates and final acceptance. Requirements in SUPERVISOR_REQUIREMENTS_V1.md are authoritative.

## Isolation and resource contract
Only research/tree_qc and research launcher files may change, plus research ignores. Product source, UI, packaging, frozen release and scientific algorithms remain untouched. No push and no disk cleanup. D:\TreeQCData stores public data; D:\TreeQCTraining stores environment, downloads, logs and checkpoints. Projected D free space below 35 GiB stops an operation. Initial D free 132.41 GiB, E free 50.28 GiB, RAM 15.73 GiB; NVIDIA 5070 laptop class, 8 GiB, driver 591.86. Exact device identity must be logged by runtime.

## Frozen sequence
A: official-supported independent Python 3.11/3.12 CUDA wheel; pure PyTorch PointNet and PointNet++ SSG with actual FPS, radius grouping, local shared MLP and hierarchical abstraction. Real CUDA allocation/matmul/forward/backward/AMP; 32–64 sample diagnostic tiny overfit; checkpoint reload and continued optimizer update; Windows spawn DataLoader; batch probes 16,8,4,2,1 at 4096 points. Log actual failures and safe batch. Supervisor independently reads code and raw outputs. Failure stops public download.
B: online official TLS Forest Instance Segmentation Benchmark version/license/files audit; supervisor pins exactly two sites, preferably Litchfield and Wytham, and estimated download/extraction budget. No commercial conclusion with unresolved license issues.
C: verified resumable streaming download and bounded extraction; world-coordinate tree identity dedup including cross-plot spatial overlap; spatially grouped training-site train/validation with disjoint neighborhood identity pools; entirely held-out second site. No context tree can bridge train/validation. Freeze identity manifest before examples.
D: real spatial local context, 4096 XYZ+membership input and 2048 support. Local translation retains metric scale (no absolute world position, RGB or intensity). Stratify height/XY/membership boundary. Online multiple continuous spatial corruption variants; retain removed membership as context; real neighbors for merges/contamination/noise. Natural difficult GT trees remain GOOD. Freeze train/validation/test generator seeds, variants and severity ranges; audit label validity and shortcut features before training.
E: compare only PointNet and PointNet++ under identical examples, split, sampler and budget. Bounded first training run, validation-only selection and early stopping. No automatic model/epoch expansion on failure.
F: freeze config/checkpoint hashes and validation thresholds before accessing held-out scores. Primary binary BAD metrics and per-defect performance; ternary uncertain reported separately and never counted as bad recall. Moderate/high held-out targets: recall >=.85, false pass <=.15, good false review <=.25, balanced accuracy >=.80. MERGED/CONTAMINATION/OVERSEG must be meaningful.
G: three seeds when measured training cost allows; all runs/config choices specified before held-out evaluation; multiple sampling seeds at 4096/2048 and >10pp recall drop fails robustness. Held-out feedback cannot tune later seeds.
H: only after public experiment freeze open eight local EXPLICIT_BAD case geometries. Locate metadata only beforehand; never use local examples or labels to design features, generator, thresholds or architecture. No local GOOD means no local false-review metric. Report review/uncertain/false-pass counts, continuation >=6/8 nonpass, ideally >=5/8 review.
I: CPU/GPU bounded preprocessing and inference for 1/17/100 trees, RAM/VRAM/time. No full-source rescan; asynchronous advisory and revision-bound incremental future contract only. ONNX only if public pass and local promising; supported dynamo exporter then trunk fallback and >=100-sample parity.

## Measurement discipline
Mini gate demonstrates implementation only, not scientific learnability. Fixed threshold binary metrics use REVIEW as positive. Uncertain never inflates binary recall. Class-balanced evaluation with counts, per-defect denominators, and uncertainty rates. Public held-out evaluation is final, never tuning input. If a stage is unavailable, report NOT_RUN with cause rather than invent metrics or a success verdict. No product integration in this phase. Final independent read-only reviewer must not have implemented the code.

## Agent policy
Default supervisor + one worker; at most two concurrent children and second only for independent read-only audit. Worker halts at every gate pending supervisor review. Supervisor records decisions in reports. Final report follows supplied field list and permits only the supplied evidence-supported verdicts.
