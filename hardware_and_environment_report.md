# Hardware and Environment Report — Tree Instance ML Baseline

## Status

This report records the verified/planned ML baseline only.  No ML packages
were installed, no Conda environment was created, and no training or inference
was run for this documentation-only session.

## Repository

- **Repository:** `D:\TreeLineWarning-TreeML`
- **Branch:** `feature/tree-instance-ml`
- **Baseline commit:** `a8f255b8f74e08b6da578aa9e2b52098c52a5455`
- **Baseline subject:** `feat: add reviewed V3 tree segmentation visualization`

Verified repository integration references:

- `algorithms/vegetation/tree_inputs.py` —
  `load_tree_segmentation_input_set(...)`
- `core/contracts/tree_inputs.py` — `TreeSegmentationInputSet`
- `core/contracts/vegetation_trees.py` — `TreeInstanceRecord`,
  `TreeSegmentationStatistics`, and `TreeSegmentationResult`

## Host execution context

- **Operating-system context:** Windows host.
- **Shell context:** PowerShell.
- **Workspace drive:** `D:`.
- **Observed free space on D drive:** approximately **109.93 GB**.

Exact Windows build and PowerShell version were not collected in this
documentation-only session and are intentionally not inferred here.

## Conda environment inventory

The currently present environments reported for planning are:

- `base`
- `treeline-prediction`

**Planned isolated environment (not created):** `treeline-tree-ml`.

中文说明：`treeline-tree-ml` 仅为后续隔离环境的计划名称；当前尚未创建，也未安装任何 ML 依赖。

## GPU information

Previously verified GPU information:

- **GPU family reported:** NVIDIA GeForce RTX 5060 family
- **GPU memory reported:** approximately **8151 MiB**
- **NVIDIA driver reported by `nvidia-smi`:** `610.74`
- **CUDA UMD reported by `nvidia-smi`:** `13.3`

The exact GPU subtype, CUDA toolkit installation state, ONNX Runtime version,
Python version, and all ML software versions were not verified for this report;
none are invented here.

## ML readiness statement

- No ML packages were installed or updated.
- No Conda environment was created or modified.
- No model, checkpoint, ONNX file, dataset, cache, training log, or inference
  output was created.
- No source code, dependency metadata, or Git configuration was changed.

The next permitted work is review of the ML0 requirement freeze.  Dependency
installation, environment creation, GPU-provider selection, model export, and
training each require separate explicit authorization.
