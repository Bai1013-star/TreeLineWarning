# Gate A recovery supervisor record

Continues clean research HEAD39a46b8c93950f6190705d4f491b31331815898d; no rebuild or repeated architecture audit. User explicitly supersedes network-only final verdict with OFFICIAL_PYTORCH_DOWNLOAD_NETWORK_BLOCKED. Public data stays closed until actual Gate A pass.

Supervisor checked current official GitHub latest API: v2.14.0 published2026-09-02. Current tagged RELEASE.md lists Windows CUDA13.0.2 Blackwell12.0 support; current cu130 official index publishes CPython3.11 Windows wheel. Selected current stable torch2.14.0+cu130. This is a newly checked selection, not a stale hard-coded assumption. Driver591.86, device capability12.0, Python3.11.9 retained; real runtime validation pending.

Official URL: https://download-r2.pytorch.org/whl/cu130/torch-2.14.0%2Bcu130-cp311-cp311-win_amd64.whl
Official alternative host: https://download.pytorch.org/whl/cu130/torch-2.14.0%2Bcu130-cp311-cp311-win_amd64.whl
Expected SHA256: 00fb5440e7560b3b21e5e24567c6096fec4ef65b50a7ef1a105ebed5755eef7e
Sources: https://api.github.com/repos/pytorch/pytorch/releases/latest ; https://raw.githubusercontent.com/pytorch/pytorch/v2.14.0/RELEASE.md ; https://download.pytorch.org/whl/cu130/torch/

Old recorded tmp13s1vldg path is absent. Worker located same-sized pip-unpack wheel candidate at D:\TreeQCTraining\tmp\pip-unpack-97lmycbe\torch-2.14.0+cu130-cp311-cp311-win_amd64.whl (43,735,040 bytes). It is an incomplete wheel despite its filename suffix. Provenance corroboration, remote range comparison and complete final hash are required; filename or ZIP header alone is insufficient. Worker owns download/install execution and logs; supervisor never launches a competing transfer.

All prior failed attempts, including local historical-statistics exposure disclosure, remain part of the record. No change to model/toy/epoch/acceptance target is authorized as a response to network failure. Final reviewer must be newly assigned and read-only after actual Gate A execution.

## Recovery Gate A final supervisor decision

GATE_A_CUDA_MINI = PASS (engineering implementation gate only).
Independent new read-only reviewer = gate_a_runtime_reviewer; verdict PASS_WITH_LIMITATIONS. It independently hashed official wheel and both checkpoints, loaded both on CUDA and reproduced all32 sample evaluation losses exactly; optimizer steps16/12 equal epochs4/3 times4 batches. Supervisor independently checked all7 source hashes/config hash and actual raw logs. PointNet loss0.6822715104→0.0287656975; PointNet++0.6909888536→0.1491527297; both acc1.0, reload error0, resume steps+1, parameter updates verified. Two Windows spawn workers, float16 AMP, all10 batch probes with2 actual updates each verified. Safe batch8;16 is largest tested, no OOM observed and no hardware maximum claim. End-to-end mini runtime12.737s; not public training performance.

Tensor/cudnn IEEE FP32 policy was explicitly fixed after actual batch-single numeric test failure. Original saved input/parameter pair under TF32 showed1.49757e-5 max logit error; IEEE yielded7.45e-9 with unchanged tolerance. Model/toy/epoch/threshold and original mini config SHA b119fd3f242f94377e7c81c6ebaeff4411c127b1be7c381be698be92e1bc3251 unchanged. This is precision-policy repair, not changing scientific gate or weakening tests. Raw failed and successful results retained.

Wheel1990587822 bytes SHA independently confirmed. Transfer reused43735040 bytes and fetched1946852782 new bytes in686.37 seconds including assembly/hash. Actual runtime torch2.14.0+cu130/CUDA13.0 recognizes sm120. Dataset dependencies beyond puretorch remain a separate future check. VRAM is allocator peak; RAM sampled process-tree working-set, not continuous total peak.

Runtime validation is complete; earlier pending statements in this dated log are historical. Gate A clears the computational prerequisite to public download, but actual archives still require Gate B license/identity and storage approval.

## Automatic Gate B continuation
Supervisor followed the official record API after Gate A acceptance and obtained13895-byte JSON in3.414seconds. Earlier Zenodo timeouts are historical, not current proof of persistent block. Saved zenodo_16875688_verified_metadata.json. License object cc-by-4.0. Exactly two candidate archives: Litchfield2331915728 bytes, Wytham3166118961 bytes; total5498034689 bytes. Latest chain and capacity approval in progress. No archive downloaded at this checkpoint.

## Gate B approved
Supervisor independently inspected latest API JSON id16875688, license.id cc-by-4.0, description citation/attribution, exact two archive identities and official checksums/URLs. Latest-chain response saved separately confirms same record. GATE_B_DATA_LICENSE=PASS for this isolated research use; not a commercial product/weight redistribution conclusion. PUBLIC_DATA_DOWNLOAD_APPROVAL.json grants exactly two pinned archives, inspect only. Combined bytes5498034689. Expanded cap40GiB, training reserve10GiB, actual free remaining35GiB mandatory; actual central-directory totals must be reviewed before extraction. No additional sites/weights allowed. Original Gate A requirement PUBLIC_DATA_DOWNLOAD_ALLOWED is now YES for these approved archives only.
