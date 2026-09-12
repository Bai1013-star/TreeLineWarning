# LICENSE_AUDIT — Tree QC V2 feasibility round

Audit date: **2026-09-12**. Repo root: `E:\TreeLineWarning-tree-qc-pointnet2-final-pilot-v1`.
Scope: license clearance for every dataset, repository and library named in the Tree QC V2 feasibility brief,
with an explicit commercial-shipping verdict for the TreeLineWarning Windows product.

## How this audit was produced

* Every license below was read from a **primary source**: the publisher's own API (Zenodo, Dataverse, GitHub,
  Hugging Face), the repository's own `LICENSE` file at raw.githubusercontent.com, or the repository's own README.
* `web_fetch` was **blocked** in this session (`URL hostname "github.com" resolves to a non-public IP address`).
  All retrieval was done with `Invoke-WebRequest`/`Invoke-RestMethod` over HTTPS instead. That substitution is
  recorded here because it changes nothing about the sources — the same primary endpoints were read.
* Nothing was guessed. Where a license could not be established from a primary source, the entry says
  `UNVERIFIED` and the failed attempts are listed in the [UNVERIFIED log](#unverified-log).
* No existing file in this repository was modified by this audit. The two `LICENSE_AUDIT.*` files are new.

## Summary table

| # | Component | Identifier | Commercial use | Redistribution | Weights separately licensed | Used in this round |
|---|---|---|---|---|---|---|
| 1 | TLS Forest Instance Segmentation Benchmark (Zenodo 16875688) | `CC-BY-4.0` | **Yes** | **Yes**, with attribution | No — record-level CC BY 4.0 covers all files | **USED** |
| 2 | TreeInstSegEval | **none — UNVERIFIED (absent)** | **No grant** | **No grant** | n/a | **REFERENCED_ONLY** |
| 3 | TreeLearn | `MIT` (code) + `CC-BY-4.0` (weights) | **Yes** | **Yes** | **Yes** — weights on Göttingen Dataverse | **REFERENCED_ONLY** |
| 4 | OpenPoints | `MIT` | **Yes** | **Yes** | No separate statement | **NOT_USED** |
| 5 | PointNeXt | `MIT` | **Yes** | **Yes** | No separate statement | **REFERENCED_ONLY** |
| 6 | SoftGroup | `MIT` | **Yes** | **Yes** | No separate statement | **NOT_USED** |
| 7 | Point-SAM | `MIT` (code) + `Apache-2.0` (weights) | **Yes** | **Yes** | **Yes** — HF checkpoint is Apache-2.0 | **NOT_USED** |
| 8 | Pointcept | `MIT` | **Yes** | **Yes** | **Yes** — per-checkpoint (some NC) | **NOT_USED** |
| 9 | Sonata | `Apache-2.0` (code) + **`CC-BY-NC-4.0` (weights)** | Code **yes**, weights **NO** | Code **yes**, weights **NO** | **Yes** — weights are NC | **NOT_USED** |
| 10 | Point Transformer V3 | `MIT` | **Yes** | **Yes** | Checkpoint is MIT-tagged | **NOT_USED** |
| 10 | Point Transformer V1 | **UNVERIFIED** (official repo unreachable) | **Not established** | **Not established** | Unknown | **REFERENCED_ONLY** (name only) |
| 11 | Point-M2AE | `MIT` | **Yes** | **Yes** | No separate statement | **NOT_USED** |
| 12 | PyTorch | `BSD-3-Clause` | **Yes** | **Yes** | n/a | **USED** |
| 12 | NumPy | `BSD-3-Clause` | **Yes** | **Yes** | n/a | **USED** |
| 12 | SciPy | `BSD-3-Clause` | **Yes** | **Yes** | n/a | **REFERENCED_ONLY** |
| 12 | scikit-learn | `BSD-3-Clause` | **Yes** | **Yes** | n/a | **USED** |

---

## 1. TLS Forest Instance Segmentation Benchmark — Zenodo record 16875688

| Field | Value |
|---|---|
| Name | *TLS forest instance segmentation benchmark: 2983 manually segmented trees from four plots* |
| Source | <https://zenodo.org/records/16875688> |
| DOI | [`10.5281/zenodo.16875688`](https://doi.org/10.5281/zenodo.16875688) (concept DOI `10.5281/zenodo.14615492`) |
| License identifier | `CC-BY-4.0` — read from the publisher API as `metadata.license.id = "cc-by-4.0"`, `metadata.access_right = "open"` |
| License URL | <https://creativecommons.org/licenses/by/4.0/> |
| Commercial use | **Permitted.** CC BY 4.0 grants use "for any purpose, even commercially" ([CC BY 4.0 deed](https://creativecommons.org/licenses/by/4.0/), [legal code](https://creativecommons.org/licenses/by/4.0/legalcode.en)) |
| Redistribution | **Permitted**, with attribution, a link to the license, and an indication of changes. No NonCommercial and no NoDerivatives term is present. |
| Weights separately licensed | **No.** The record-level CC BY 4.0 declaration covers every file in the record, including `TreeLearn_finetuned_weights_all.pth` (361,611,057 bytes) and `ForAINet_finetuned_weights_all.pt` (617,565,194 bytes). Upstream provenance caveat below. |
| Used in this round | **USED.** `research/tree_qc_v2/v2_config.py` reads the published per-tree and per-plot PLY files for `litchfield` and `wytham` from the verified extraction root. |

**Plots confirmed against the publisher description** (not assumed): the four plots are Litchfield (Australia,
tropical savanna), Wytham Woods (United Kingdom, deciduous temperate forest), Ofental (Germany, coniferous
temperate forest) and Robson Creek (Australia, tropical rainforest). All were collected with RIEGL VZ-series
terrestrial laser scanners, processed in Riscan Pro, and manually extracted by nine operators with a second
quality-control pass. Semantic labels are `0: ground, 1: tree`; instance labels are sequential from `1`, with
`-1` for ground points. Source: [Zenodo record 16875688](https://zenodo.org/records/16875688).

**Required citation** (the record states *"Any use of this dataset should cite the paper above"*):

> Cherlet, W., Dayal, K., Chen, S., Cooper, Z., Disney, M., Hanzl, A., Levick, S., Nightingale, J., Origo, N.,
> Senf, C., Soenens, L., Terryn, L., Van den Broeck, W. A. J., & Calders, K. (2026). *Benchmarking tree instance
> segmentation of terrestrial laser scanning point clouds.* **ISPRS Journal of Photogrammetry and Remote
> Sensing, 231**, 230–247. ISSN 0924-2716. DOI: [10.1016/j.isprsjprs.2025.10.033](https://doi.org/10.1016/j.isprsjprs.2025.10.033)

The bibliographic record is independently corroborated by the [UGent institutional bibliography](https://biblio.ugent.be/publication/01KAVB41FW55319QCJV1MH1VXB)
and the [ScienceDirect article page](https://www.sciencedirect.com/science/article/abs/pii/S092427162500423X).

**Version chain re-verified in this audit.** `https://zenodo.org/api/records/16875688/versions/latest` returns
record id `16875688` itself with `license = cc-by-4.0` and `access_right = open`, so the pinned DOI is the latest
version at this audit time. The record carries 10 files.

**Upstream provenance caveat (not a defect in this record).** The fine-tuned weights published *inside* this
CC BY 4.0 record are derivatives of third-party projects with their own terms. The record's CC BY 4.0 declaration
is the publisher's grant over the files it distributes, but a diligent shipper should trace the upstream terms
too — see item 3 (TreeLearn) and the [adjacent observations](#adjacent-observations). This round does not
download or use those weights.

## 2. TreeInstSegEval — official evaluation code

| Field | Value |
|---|---|
| Name | TreeInstSegEval — *"Evaluation code and documentation for the paper 'Benchmarking tree instance segmentation of terrestrial laser scanning point clouds'"* |
| Source | <https://github.com/qforestlab/TreeInstSegEval> (default branch `master`) |
| License identifier | **None present. `UNVERIFIED` — no license is published, which is not the same as an unknown license.** |
| License URL | **None exists to cite.** |
| Commercial use | **No grant.** Absent any license, default copyright reserves all rights; no permission to use, copy, modify or distribute commercially has been granted. |
| Redistribution | **No grant.** |
| Weights separately licensed | Not applicable — this repository contains evaluation code only. |
| Used in this round | **REFERENCED_ONLY.** Cited as the authoritative source for the evaluator's coordinate convention, with the pinned commit SHA, in `research/tree_qc/reports/GATE_C_MANUAL_ARCHIVE_SCHEMA.md`. **No code was copied.** |

**Evidence of absence** (all executed in this audit, all negative):

* `GET https://api.github.com/repos/qforestlab/TreeInstSegEval/license` → **HTTP 404** (GitHub's licensee
  classifier found no license file).
* `GET https://api.github.com/repos/qforestlab/TreeInstSegEval` → the `license` field is **empty**.
* Raw probes for `LICENSE`, `LICENSE.md`, `LICENSE.txt`, `COPYING`, `license`, `LICENCE` on `master` → **all 404**.
* The repository root listing is exactly: `.gitignore`, `README.md`, `__init__.py`, `__pycache__/`,
  `base_evaluation.py`, `panopticseg_evaluation.py`, `raycloudtools_evaluation.py`, `sssc_evaluation.py`,
  `treeiso_evaluation.py`, `treelearn_ALL_evaluation.py`, `treelearn_evaluation.py` — **no license file**.
* The `README.md` contains no license section, and its data-download URL and its "Cite:" block are both still
  literal `TODO` placeholders, i.e. the repository was published in an unfinished state
  ([README at master](https://github.com/qforestlab/TreeInstSegEval/blob/master/README.md)).

**Consequence.** The dataset is CC BY 4.0, but **the CC BY 4.0 grant on the data does not extend to this code**.
A CC BY license on a dataset and the license of a separate code repository on a different host are independent
grants. The Zenodo record merely points at the GitHub organisation
(*"the code available at https://github.com/qforestlab"*), which is a pointer, not a license.

## 3. TreeLearn

| Field | Value |
|---|---|
| Name | TreeLearn — deep learning tree instance segmentation for ground-based LiDAR forest point clouds |
| Source | <https://github.com/ecker-lab/TreeLearn> |
| License identifier (code) | `MIT` — GitHub licensee: `spdx_id = MIT`, `path = LICENSE` |
| License URL (code) | <https://github.com/ecker-lab/TreeLearn/blob/main/LICENSE> |
| Commercial use (code) | **Permitted.** |
| Redistribution (code) | **Permitted**, retaining the copyright notice and permission notice. |
| Weights separately licensed | **Yes — and permissively.** The `.pth` weights are **not** in the git repository; `tree_learn/util/download.py` fetches them from the Göttingen research-online Dataverse at base path `https://data.goettingen-research-online.de/api/access/datafile/:persistentId?persistentId=doi:10.25625/VPMPID/`. That dataset ("TreeLearn Dataset", version 7.1) declares dataset license **`CC BY 4.0`** (`license.name = "CC BY 4.0"`, `uri = http://creativecommons.org/licenses/by/4.0`). |
| License URL (weights) | <https://creativecommons.org/licenses/by/4.0/> · record: <https://doi.org/10.25625/VPMPID> |
| Commercial use (weights) | **Permitted**, with attribution. |
| Weights present in that record | `model_weights_dirty.pth`, `model_weights_finetuned.pth`, `model_weights_diverse_training_data.pth`, `model_weights_20241213.pth`, `model_weights_with_small_20241213.pth` — all listed `restrict=False`. |
| Used in this round | **REFERENCED_ONLY.** No TreeLearn code or weights were downloaded or executed. It appears (a) as a named exclusion in `research/tree_qc/SUPERVISOR_REQUIREMENTS_V1.md` (*"不得无理由升级到 … TreeLearn"*) and (b) as the provenance of the fine-tuned weights shipped inside Zenodo 16875688, which were not downloaded (`research/tree_qc/reports/DATA_LICENSE_AUDIT.md`). |

**Weights-license nuance, stated precisely.** The Dataverse record's `ReadMe.txt` documents the five weight
files and their training lineage but states **no additional or alternative license** for them. The dataset-level
`CC BY 4.0` declaration is therefore the governing grant for the weight files. This audit read that `ReadMe.txt`
directly from the Dataverse access API and confirmed it contains no license carve-out.

## 4. OpenPoints

| Field | Value |
|---|---|
| Name | OpenPoints — a library for reproducing point cloud models |
| Source | <https://github.com/guochengqian/openpoints> (default branch `master`) |
| License identifier | `MIT` — GitHub licensee: `spdx_id = MIT`, `path = LICENSE`. `LICENSE` begins *"MIT License — Copyright (c) Guocheng Qian."* |
| License URL | <https://github.com/guochengqian/openpoints/blob/master/LICENSE> |
| Commercial use | **Permitted.** |
| Redistribution | **Permitted**, retaining the copyright and permission notice. |
| Weights separately licensed | **No separate statement found.** The repository is a training/evaluation codebase. (Hugging Face mirrors such as `torch-pointcloud/pointnet2.*.openpoints` carry MIT tags, but those are third-party re-uploads and are **not** an authoritative OpenPoints weights license — do not rely on them.) |
| Used in this round | **NOT_USED.** Zero references anywhere in the repository (case-sensitive searches for `openpoints` and `OpenPoints` both returned 0 content hits). |

## 5. PointNeXt

| Field | Value |
|---|---|
| Name | PointNeXt — revisiting PointNet++ with improved training and scaling strategies |
| Source | <https://github.com/guochengqian/PointNeXt> (default branch `master`) |
| License identifier | `MIT` — GitHub licensee: `spdx_id = MIT`, `path = LICENSE` |
| License URL | <https://github.com/guochengqian/PointNeXt/blob/master/LICENSE> |
| Commercial use | **Permitted.** |
| Redistribution | **Permitted**, retaining the copyright and permission notice. |
| Weights separately licensed | **No separate statement found.** The README's *Model Zoo (pretrained weights)* section points to <https://guochengqian.github.io/PointNeXt/modelzoo/> and `docs/checkpoints.md`; neither states a weights-specific license, and `docs/checkpoints.md` contains no license or terms text. Absent a carve-out, the checkpoints fall under the repository's MIT grant — recorded here as a reading of the repository, not as an explicit publisher statement. |
| Used in this round | **REFERENCED_ONLY.** Named only as a *future* comparison/upgrade path in planning documents — `docs/TreeLineWarning_requirements_v2.3.md` (*"PointNeXt-S 保留为后续对照或升级方案"*, and *"PointNeXt-S 仅作为后续对照，不在八月底前并行建设第二套正式链路"* in `docs/TreeLineWarning_codex_tasks_v2.3.md` and `docs/codex_tasks.md`). No PointNeXt code, config or checkpoint is present or imported. |

## 6. SoftGroup

| Field | Value |
|---|---|
| Name | SoftGroup — soft grouping for 3D instance segmentation |
| Source | <https://github.com/thangvubk/SoftGroup> (default branch `main`) |
| License identifier | `MIT` — GitHub licensee: `spdx_id = MIT`, `path = LICENSE`. `LICENSE` begins *"MIT License — Copyright (c) 2022"*. |
| License URL | <https://github.com/thangvubk/SoftGroup/blob/main/LICENSE> |
| Commercial use | **Permitted.** |
| Redistribution | **Permitted**, retaining the copyright and permission notice. |
| Weights separately licensed | **No separate statement found.** |
| Used in this round | **NOT_USED.** Zero references in the repository. Note for completeness: the TreeLearn README states *"The code is built based on SoftGroup and spconv"*, so SoftGroup is an **upstream architectural ancestor of TreeLearn**, and its MIT terms are compatible with TreeLearn's MIT grant. This matters only if TreeLearn is ever adopted. |

## 7. Point-SAM

| Field | Value |
|---|---|
| Name | Point-SAM — promptable 3D segmentation model for point clouds |
| Source | <https://github.com/zyc00/Point-SAM> |
| License identifier (code) | `MIT` — GitHub licensee: `spdx_id = MIT`, `path = LICENSE`. `LICENSE` begins *"MIT License — Copyright (c) 2023 BAAI-Vision"*. |
| License URL (code) | <https://github.com/zyc00/Point-SAM/blob/main/LICENSE> |
| Commercial use (code) | **Permitted.** |
| Redistribution (code) | **Permitted**, retaining the copyright and permission notice. |
| Weights separately licensed | **Yes — under a *different* license from the code.** The README directs users to the ViT-L checkpoint at <https://huggingface.co/yuchen0187/Point-SAM>. The Hugging Face model card declares `license: apache-2.0`. So this component carries **two licenses on two artifacts**: MIT for code, Apache-2.0 for weights. Both are permissive and commercially usable. |
| License URL (weights) | <https://huggingface.co/yuchen0187/Point-SAM> · <https://www.apache.org/licenses/LICENSE-2.0> |
| Submodule caveat | The repository declares git submodules for `torkit3d` (<https://github.com/Jiayuan-Gu/torkit3d>) and NVIDIA `apex` (<https://github.com/NVIDIA/apex>). These are not covered by the Point-SAM MIT grant and would each need their own review if the code were ever adopted. |
| Used in this round | **NOT_USED.** A naive substring search reports 7 hits for `point_sam`, but **every one is a false positive**: they are all occurrences of `farthest_point_sample`, `radius_group` and similar identifiers in `research/tree_qc/models.py`, `mini_gate.py` and `test_models.py`. There is **no** Point-SAM code, checkpoint or import in this repository. |

## 8. Pointcept

| Field | Value |
|---|---|
| Name | Pointcept — point cloud perception codebase |
| Source | <https://github.com/Pointcept/Pointcept> (default branch `main`) |
| License identifier | `MIT` — GitHub licensee: `spdx_id = MIT`, `path = LICENSE`. `LICENSE` begins *"MIT License — Copyright (c) 2023 Pointcept"*. |
| License URL | <https://github.com/Pointcept/Pointcept/blob/main/LICENSE> |
| Commercial use | **Permitted.** |
| Redistribution | **Permitted**, retaining the copyright and permission notice. |
| Weights separately licensed | **Yes — per checkpoint, and this is a real trap.** Pointcept publishes checkpoints under *different* licenses depending on the release: `Pointcept/PointTransformerV3` is MIT-tagged, whereas `Pointcept/Concerto` and `Pointcept/Utonia` are tagged **`cc-by-nc-4.0`**. The Pointcept *code* license therefore does **not** determine the license of any given checkpoint. Each adopted checkpoint must be cleared individually. |
| Used in this round | **NOT_USED.** Zero references in the repository. |

## 9. Sonata

| Field | Value |
|---|---|
| Name | Sonata — a self-supervised pre-training framework for 3D point clouds (CVPR 2025) |
| Source | <https://github.com/facebookresearch/Sonata> |
| License identifier (code) | `Apache-2.0` — GitHub licensee: `spdx_id = Apache-2.0`, `path = LICENSE`; the file is the standard Apache License, Version 2.0. |
| License URL (code) | <https://github.com/facebookresearch/Sonata/blob/main/LICENSE> · <https://www.apache.org/licenses/LICENSE-2.0> |
| Commercial use (code) | **Permitted.** |
| Redistribution (code) | **Permitted**, subject to Apache-2.0 conditions (retain notices, state changes, include the license and any `NOTICE`). |
| License identifier (weights) | **`CC-BY-NC-4.0` — NonCommercial. This is a hard commercial blocker for the weights.** |
| License URL (weights) | <https://creativecommons.org/licenses/by-nc/4.0/deed.en> |
| Weights separately licensed | **Yes — and restrictively.** The repository's own `## License` section states verbatim: *"Sonata code is released by Meta under the Apache 2.0 license; Sonata weight is released under the CC-BY-NC 4.0 license (restricted by NC of datasets like HM3D, ArkitScenes)."* |
| Commercial use (weights) | **Prohibited.** CC BY-NC 4.0 excludes commercial use. |
| Redistribution (weights) | Permitted only for NonCommercial purposes. |
| Used in this round | **NOT_USED.** Zero references in the repository. |

**Metadata conflict — flagged deliberately.** The official `facebook/sonata` Hugging Face model card carries
**no license tag at all** (`cardData` has no `license` key; the front matter declares only `model_hub_mixin` and
`pytorch_model_hub_mixin`). Meanwhile a third-party mirror `Pointcept/Sonata` **is** tagged `license:mit`. Neither
of those two is the authoritative grant: Meta's own repository README is, and it says the weights are
**CC-BY-NC-4.0**. Anyone who clears Sonata by reading the mirror's MIT tag would be clearing it **incorrectly**.

## 10. Point Transformer V3 / V1

### Point Transformer V3

| Field | Value |
|---|---|
| Name | Point Transformer V3 — simpler, faster, stronger |
| Source | <https://github.com/Pointcept/PointTransformerV3> |
| License identifier | `MIT` — GitHub licensee: `spdx_id = MIT`, `path = LICENSE` |
| License URL | <https://github.com/Pointcept/PointTransformerV3/blob/main/LICENSE> |
| Commercial use | **Permitted.** |
| Redistribution | **Permitted**, retaining the copyright and permission notice. |
| Weights separately licensed | The released checkpoint `Pointcept/PointTransformerV3` on Hugging Face carries `cardData.license = "mit"` — permissive and **consistent** with the code grant. |
| License URL (weights) | <https://huggingface.co/Pointcept/PointTransformerV3> |
| Used in this round | **NOT_USED.** No reference to PTv3 anywhere in the repository. |

### Point Transformer V1

| Field | Value |
|---|---|
| Name | Point Transformer (Zhao et al., ICCV 2021) |
| Source | **`UNVERIFIED` — the official code release could not be located.** See the [UNVERIFIED log](#unverified-log) for every retrieval attempted. |
| License identifier | **`UNVERIFIED`** |
| License URL | **None can be cited.** |
| Commercial use | **Not established.** Treat as unclear, not as permitted. |
| Redistribution | **Not established.** |
| Weights separately licensed | Unknown — no official weights artifact was reachable to inspect. |
| Used in this round | **REFERENCED_ONLY (by name only).** `research/tree_qc/SUPERVISOR_REQUIREMENTS_V1.md` lists `PointTransformer` in a *"must not upgrade to without justification"* list. That is a naming reference in a policy document, not a dependency. |

**What *is* verifiable for V1, and the safe route.** Pointcept ships a Point Transformer (V1) implementation in
its own MIT-licensed tree at `pointcept/models/point_transformer/` (files `point_transformer_cls.py`,
`point_transformer_partseg.py`, `point_transformer_seg.py`, `utils.py`), and the Pointcept README lists
*"**Point Transformer**"* among its supported backbones alongside PTv2 and PTv3. So if V1 architecture is wanted
for a commercial product, **the Pointcept MIT implementation is a cleared path**, whereas the original standalone
release is not currently auditable.

## 11. Point-M2AE

| Field | Value |
|---|---|
| Name | Point-M2AE — multi-scale masked autoencoders for hierarchical point cloud pre-training (NeurIPS 2022) |
| Source | <https://github.com/ZrrSkywalker/Point-M2AE> (default branch `main`) |
| License identifier | `MIT` — GitHub licensee: `spdx_id = MIT`, `path = LICENSE`. `LICENSE` begins *"MIT License — Copyright (c) 2022 Renrui Zhang"*. |
| License URL | <https://github.com/ZrrSkywalker/Point-M2AE/blob/main/LICENSE> |
| Commercial use | **Permitted.** |
| Redistribution | **Permitted**, retaining the copyright and permission notice. |
| Weights separately licensed | **No separate statement found.** |
| Used in this round | **NOT_USED.** Zero references in the repository (searches for `M2AE` returned 0 content hits). |

## 12. Framework and library licenses

| Library | Identifier | License URL | Commercial use | Redistribution | Used in this round |
|---|---|---|---|---|---|
| **PyTorch** | `BSD-3-Clause` | <https://github.com/pytorch/pytorch/blob/main/LICENSE> | **Yes** | **Yes**, retaining the copyright notice, the list of conditions, and the disclaimer | **USED** — `research/tree_qc_v2/v2_site_fingerprint.py` imports `torch` and `torch.nn`. Gate A installed `torch 2.14.0+cu130` (`research/tree_qc/reports/gate_a_recovery_evidence/requirements-installed.txt`). |
| **NumPy** | `BSD-3-Clause` | <https://github.com/numpy/numpy/blob/main/LICENSE.txt> | **Yes** | **Yes**, retaining the copyright notice, the list of conditions, and the disclaimer | **USED** — imported by `v2_data.py`, `v2_ply.py`, `recon.py`, `repr_check.py`, `v2_site_fingerprint.py`. |
| **SciPy** | `BSD-3-Clause` | <https://github.com/scipy/scipy/blob/main/LICENSE.txt> | **Yes** | **Yes**, retaining the copyright notice, the list of conditions, and the disclaimer | **REFERENCED_ONLY in this round** — zero imports under `research/tree_qc_v2/`. It *is* a declared product dependency (`pyproject.toml`: `scipy>=1.12,<2`) and is imported widely elsewhere in the repo (24 files, mostly `from scipy.spatial import cKDTree`). |
| **scikit-learn** | `BSD-3-Clause` | <https://github.com/scikit-learn/scikit-learn/blob/main/LICENSE> | **Yes** | **Yes**, retaining the copyright notice, the list of conditions, and the disclaimer | **USED** — `research/tree_qc_v2/repr_check.py` imports `sklearn.linear_model.LogisticRegression`. |

**Note on classifier output for PyTorch and NumPy.** GitHub's licensee classifier reports `NOASSERTION`
("Other") for both, because each project's `LICENSE`/`LICENSE.txt` bundles a BSD-3-Clause grant **together with**
additional third-party notices in the same file. This audit read the file text directly rather than trusting the
classifier label: PyTorch's `LICENSE` contains the canonical three-clause BSD text (*"Redistribution and use in
source and binary forms… 1. … 2. … 3. Neither the names of Facebook, Deepmind Technologies, NYU, NEC Laboratories
America and IDIAP Research Institute…"*), and NumPy's `LICENSE.txt` contains the three-clause BSD text
(*"Neither the name of the NumPy Developers…"*). Both are **BSD-3-Clause** for commercial-clearance purposes.
SciPy and scikit-learn are classified `BSD-3-Clause` normally.

---

## Research-only / NonCommercial statement

**Question asked: is anything in this list research-only, non-commercial, or CC-BY-NC?**

**Answer: yes — two distinct problems, and they are not the same kind of problem.**

### (a) Explicitly NonCommercial / research-only — `CC-BY-NC-4.0`

1. **Sonata weights — `CC-BY-NC-4.0`.** This is the clearest NC restriction in the list and it is stated by the
   publisher in plain language. The Sonata *code* is Apache-2.0 and commercially fine; the Sonata *weights* are
   NonCommercial. Source: [facebookresearch/Sonata README, `## License`](https://github.com/facebookresearch/Sonata#license).
2. **Pointcept's `Concerto` and `Utonia` checkpoints — `cc-by-nc-4.0`.** Surfaced here because the brief asks
   about Pointcept's weights. The Pointcept *code* is MIT. Source: <https://huggingface.co/Pointcept/Concerto>,
   <https://huggingface.co/Pointcept/Utonia>.

### (b) No license granted at all — legally *more* restrictive than NC in practice

3. **TreeInstSegEval — no license file, no license statement.** Because no rights are granted, every act that
   copyright controls (use, copying, modification, redistribution) is reserved to the author. A component with
   no license **cannot** be cleared for commercial shipping by reading permissive neighbors — the CC BY 4.0 grant
   on the Zenodo *dataset* does not reach the separate *code* repository. Source:
   <https://github.com/qforestlab/TreeInstSegEval> (verified absent — see item 2).

### (c) Not verifiable, therefore not cleared

4. **Point Transformer V1 official code.** License could not be established from any primary source, so it is
   **not** cleared for commercial use. Use the MIT-licensed Pointcept implementation instead, or re-audit if the
   original release reappears.

### (d) Everything else in the list is permissive

No other entry carries a research-only or NonCommercial term. The Zenodo benchmark data (item 1), TreeLearn code
and weights (item 3), OpenPoints, PointNeXt, SoftGroup, Point-SAM code and weights, Pointcept code, Point
Transformer V3 code and checkpoint, Point-M2AE, and PyTorch/NumPy/SciPy/scikit-learn are all under permissive
terms (`CC-BY-4.0`, `MIT`, `Apache-2.0`, `BSD-3-Clause`) that permit commercial use and redistribution subject
only to attribution and notice-retention conditions.

---

## BLOCKING_FINDINGS

A blocking finding is anything that would prevent commercial use of a component **if that component ended up in
the shipped TreeLineWarning Windows product**.

### BLOCK-1 — TreeInstSegEval has no license (severity: blocking, currently latent)

* **What:** `qforestlab/TreeInstSegEval` publishes no license of any kind (GitHub licensee 404; six probed
  license paths all 404; repo root listing contains no license file; README has no license section).
* **Why it blocks:** with no grant, there is no permission to copy, modify, or redistribute the code — or a
  derivative translation of its evaluation logic — in a commercial product. An unlicensed component is not a
  "clean" component.
* **Current exposure in this round: none.** The code is `REFERENCED_ONLY`. The round cites the evaluator's
  coordinate convention with a pinned commit SHA in `research/tree_qc/reports/GATE_C_MANUAL_ARCHIVE_SCHEMA.md`
  and explicitly notes *"No official implementation copied."*
* **What would make it block in fact:** copying any part of `base_evaluation.py` or its sibling evaluators into
  `research/tree_qc_v2/` or into product code; vendoring the repo; or shipping a rewrite that is a line-by-line
  translation of it.
* **Required action:** keep it reference-only and implement the IoU/matching metric independently from the
  paper's own definition; **or** obtain written permission from Q-ForestLab (contact `wout.cherlet@ugent.be`);
  **or** confirm the repository later gains a permissive license. Do not assume the dataset's CC BY 4.0 covers it.

### BLOCK-2 — Sonata weights are CC-BY-NC-4.0 (severity: blocking for weights, clear for code)

* **What:** *"Sonata weight is released under the CC-BY-NC 4.0 license (restricted by NC of datasets like HM3D,
  ArkitScenes)."* — publisher's own README.
* **Why it blocks:** CC BY-NC 4.0 excludes commercial use. Shipping Sonata weights, or a model fine-tuned from
  them, inside a product sold or commercially deployed as TreeLineWarning is outside the grant.
* **Current exposure in this round: none.** Sonata is `NOT_USED` (zero references).
* **Trap to avoid:** the `Pointcept/Sonata` Hugging Face mirror is tagged `license:mit` and the official
  `facebook/sonata` card carries no license tag at all. **Neither overrides Meta's stated NC term.** Clear Sonata
  only from the upstream README.
* **Required action:** if Sonata is ever adopted, use the **code only** (Apache-2.0) and train from scratch, or
  license the weights from Meta; do not ship the published checkpoint.

### BLOCK-3 — Point Transformer V1 official code license UNVERIFIED (severity: blocking until resolved)

* **What:** the official V1 code release could not be located, so its license is `UNVERIFIED`. Every retrieval
  attempt failed (documented in the UNVERIFIED log).
* **Why it blocks:** an unverifiable license cannot be cleared. "Probably MIT" is not a clearance.
* **Current exposure in this round: none.** `NOT_USED` / referenced by name only in a policy document.
* **Required action:** if V1 architecture is required, use the **MIT-licensed Pointcept implementation** at
  `pointcept/models/point_transformer/`, which is verifiable and permissive.

### BLOCK-4 — Per-checkpoint NC licensing inside permissive codebases (severity: blocking if assumed away)

* **What:** Pointcept's *code* is MIT, but its *checkpoints* are licensed individually —
  `Pointcept/PointTransformerV3` is MIT while `Pointcept/Concerto` and `Pointcept/Utonia` are `cc-by-nc-4.0`.
* **Why it blocks:** a permissive code license does not launder a NonCommercial weights license. A future
  decision to "use a Pointcept checkpoint" is not one decision, it is a decision per checkpoint.
* **Current exposure in this round: none.** Pointcept is `NOT_USED`.
* **Required action:** add a checkpoint-level license gate to any future model-adoption checklist. Never infer a
  checkpoint's license from its repository's `LICENSE` file.

### BLOCK-5 — Weights/artifacts carry licenses different from their code (severity: process risk)

* **What:** this pattern occurs at least three times in this list and is the single most likely way a future
  round ships a restricted artifact by accident:
  * Sonata — code Apache-2.0, weights **CC-BY-NC-4.0**.
  * Point-SAM — code MIT, weights **Apache-2.0** (both permissive, but still two licenses to honor).
  * Pointcept — code MIT, weights per-checkpoint (some **CC-BY-NC-4.0**).
  * TreeLearn — code MIT in git, weights **CC BY 4.0** on a separate Dataverse host.
* **Why it matters:** a reviewer who clears "the repo" has not cleared "the model".
* **Required action:** for every adopted component, record the code license and the weights license as **two
  separate rows** with their own URLs. This audit's JSON companion does exactly that.

### Non-blocking observations

* **scikit-learn is used but not declared** — `research/tree_qc_v2/repr_check.py` imports
  `sklearn.linear_model.LogisticRegression`, but `pyproject.toml` does not list `scikit-learn`. The license
  (BSD-3-Clause) is fine; this is a reproducibility/supply-chain hygiene issue, not a legal block. Either add it
  to the declared dependencies or move the check behind an optional extra.
* **`ForAINet_finetuned_weights_all.pt`** ships in the same CC BY 4.0 Zenodo record as the TreeLearn weights. It
  is outside the 12 audited items and was **not** audited here. If it is ever adopted, audit ForAINet's own
  upstream license separately — do not assume the record's CC BY 4.0 resolves a third-party model's terms.
* **The Tree QC V2 round is clean on all blocking findings as of this audit.** Every NC or unlicensed item is
  currently `NOT_USED` or `REFERENCED_ONLY`. The blockers above are **prospective**, and they exist so that a
  later adoption decision does not silently cross them.

---

## UNVERIFIED log

Exactly one item is `UNVERIFIED`. Everything attempted, in order:

**Point Transformer V1 — official code release and its license**

| Attempt | Result |
|---|---|
| `GET https://github.com/hengshuangzhao/pointtransformer` | HTTP **404** (and a connection reset on a retry) |
| `GET https://github.com/hengshuangzhao/point-transformer` | HTTP **404** |
| `GET https://api.github.com/users/hengshuangzhao` | HTTP **404** — **the GitHub account itself does not exist** |
| `GET https://api.github.com/repos/hengshuangzhao/point-transformer` | HTTP **404** |
| `GET https://api.github.com/repos/hengshuangzhao/PointTransformer` | HTTP **404** |
| GitHub repo search `PointTransformer in:name`, sorted by stars | Only PTv3 (MIT), PTv2 (no license detected), and third-party/unofficial forks. **No official V1 release.** |
| GitHub repo search `point transformer zhao` | Only informal reimplementations: `souheib1/Point-Transformer` (MIT, self-described *"Unofficial"*), `tuananh1007/point-transformer-pytorch` (MIT, *"Minimal"*), `pierrefdz/point-transformer` (no license). **None is the official release.** |
| GitHub search for repos named `PointTransformerV1` | Only `matfcp/PointTransformerV1_Tensoflow_Keras` (0 stars, no license) — not official. |
| `GET https://archive.org/wayback/available?url=github.com/hengshuangzhao/pointtransformer` | HTTP **429 Too Many Requests** (twice, including after a delay) |
| `GET https://web.archive.org/web/2023id_/https://github.com/hengshuangzhao/pointtransformer` | HTTP **404** — no snapshot at that URL |
| Wayback CDX prefix query `github.com/hengshuangzhao*` | HTTP **200 with zero rows** — no archived snapshots exist for this account |
| Wayback CDX query for the exact repo | Connection timed out |

**Conclusion:** the official Point Transformer V1 code release is not retrievable from GitHub or from the Wayback
Machine, so **its license cannot be verified**. `UNVERIFIED` is recorded rather than a guess. The verifiable,
permissive substitute is the Point Transformer implementation inside Pointcept (MIT).

**Verification limitation to carry forward.** `web_fetch` is unavailable in this session (DNS policy returns a
non-public IP for `github.com` and `zenodo.org`), so all retrieval used PowerShell HTTPS calls. The primary
endpoints reached were the same ones `web_fetch` would have reached. Two secondary sources — the Wayback
availability API (429) and the Wayback CDX endpoint for the exact V1 URL (timeout) — were rate-limited or
unreachable, and that is recorded rather than worked around.

---

## Adjacent observations (outside the 12 audited items)

* **`spconv`** — TreeLearn's README names it alongside SoftGroup as a code dependency, and its setup requires a
  CUDA-matched build. Not audited here; `spconv` would need its own review (Apache-2.0 upstream) if TreeLearn is
  ever adopted.
* **`open3d` and `python-fire`** — the TreeInstSegEval README lists these as its "main requirements". Not audited
  here because the Evaluator itself is reference-only.
* **`scipy`'s role** — although the V2 round does not import SciPy directly, it is already a hard product
  dependency in `pyproject.toml` and is used across `algorithms/`. Its BSD-3-Clause terms raise no commercial
  concern.
* **The 20-tree local corpus and the local 8-bad blind challenge** were explicitly out of scope and untouched by
  this audit, consistent with the boundary stated in `v2_data.py`.

## Reproduction recipe

Every fact above can be re-derived with:

```powershell
Invoke-RestMethod "https://zenodo.org/api/records/16875688"                       # data license, files, creators
Invoke-RestMethod "https://zenodo.org/api/records/16875688/versions/latest"      # latest-version chain
Invoke-RestMethod "https://api.github.com/repos/<owner>/<repo>/license"          # code license (+ 404 = none)
Invoke-RestMethod "https://data.goettingen-research-online.de/api/datasets/:persistentId/?persistentId=doi:10.25625/VPMPID"  # TreeLearn weights license
Invoke-RestMethod "https://huggingface.co/api/models/<owner>/<name>"             # weights license
```

Raw `LICENSE` files were read at `https://raw.githubusercontent.com/<owner>/<repo>/<branch>/LICENSE` (note:
`openpoints` uses `master`, not `main`).
