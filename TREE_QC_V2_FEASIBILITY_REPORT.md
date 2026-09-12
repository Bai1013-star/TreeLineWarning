# TREE_QC_V2_FEASIBILITY_AUDIT

TreeLineWarning Tree QC V2 feasibility audit - multi-site generalisation and
the local proposal refinement architecture.

This round does not train a production Tree QC V2 model. It answers one
question: does the new architecture have enough evidence to justify formal V2
training? Every number below is read from an artefact a run wrote.

## Supervisor summary

```
TREE_QC_V2_FEASIBILITY_AUDIT

GIT_HEAD = 877f1c48762319de01bd864997aa323719208739
BRANCH = research/tree-qc-v2-feasibility
WORKTREE_CLEAN = NO

STAGE_STATUS
  SITE_FINGERPRINT_TEST      = COMPLETE
  PUBLIC_GT_PROPOSAL_TEST    = COMPLETE
  ROI_LIBRARY                = COMPLETE
  MINI_REFINER_BAKEOFF       = COMPLETE_WITH_LIMITATION
  FINAL_REPORT               = COMPLETE
  MINI_REFINER_BAKEOFF cross-site folds run   = cross_site_test_wytham
  MINI_REFINER_BAKEOFF cross-site folds missing = cross_site_test_litchfield

V1_HISTORY_UNTOUCHED = YES
LOCAL_20_TREES_ACCESSED = NO
LOCAL_8_BAD_ACCESSED = NO

PUBLIC_SITES_USED = litchfield, wytham

SITE_FINGERPRINT_V1_RAW = 0.9417
SITE_FINGERPRINT_V2_NORMALIZED = 0.8667

PUBLIC_GT_PAIR_CONSTRUCTION = PASS

REAL_PROPOSAL_PAIRS = 25
SIMULATED_PROPOSAL_PAIRS = 375

MODEL_A_POINTNET2 [cross wytham] = delta_iou -0.1486, bad-proposal delta -0.0475, change_f1 0.2296, identity_copy PASS, good_preservation FAIL
MODEL_A_POINTNET2 [in-site litchfield] = delta_iou +0.1073, bad-proposal delta 0.1933, change_f1 0.4729, identity_copy PASS, good_preservation FAIL
MODEL_B_POINTNEXT_S [cross wytham] = delta_iou -0.0603, bad-proposal delta 0.0074, change_f1 0.2624, identity_copy PASS, good_preservation FAIL
MODEL_B_POINTNEXT_S [in-site litchfield] = delta_iou +0.0938, bad-proposal delta 0.1813, change_f1 0.4594, identity_copy PASS, good_preservation FAIL
MODEL_C_POINTNEXT_S_MULTITASK [cross wytham] = delta_iou -0.0540, bad-proposal delta 0.0134, change_f1 0.2885, identity_copy PASS, good_preservation FAIL
MODEL_C_POINTNEXT_S_MULTITASK [in-site litchfield] = delta_iou +0.0853, bad-proposal delta 0.1711, change_f1 0.4592, identity_copy PASS, good_preservation FAIL

BEST_CROSS_SITE_DELTA_IOU = -0.0540
BEST_IN_SITE_DELTA_IOU = +0.1073

GOOD_PRESERVATION = A:0/2 PASS; B:0/2 PASS; C:0/2 PASS
IDENTITY_COPY_GATE = A:2/2 PASS; B:2/2 PASS; C:2/2 PASS

BEST_ARCHITECTURE = MODEL_C_POINTNEXT_S_MULTITASK

FINAL_VERDICT = V2_FEASIBILITY_PARTIAL

FORMAL_V2_TRAINING_JUSTIFIED = NO

BLOCKERS =
  - The product CROWN_STEM_HYBRID backend cannot consume a public TLS plot directly: it requires a SpanRecord, a vegetation region, a coarse ground cloud and scene metadata that the benchmark does not publish. Only its stem-guided refinement function was reachable, which is why the real pair count is a diagnostic and the training pairs are simulated.
```

## 1. State of the repository and of V1

- HEAD `877f1c48762319de01bd864997aa323719208739` on `research/tree-qc-v2-feasibility`, branched from the V1 pilot tip `ff9604b48748de42d0d4422a243bcfe03c63dd00`.
- Working tree: `M TREE_QC_V2_FEASIBILITY_REPORT.json
 M TREE_QC_V2_FEASIBILITY_REPORT.md
 M research/tree_qc_v2/test_v2_feasibility.py
 M research/tree_qc_v2/v2_report.py
?? research/tree_qc_v2/verify_report.py`.
- Files changed under `research/tree_qc/`: 0. Under `algorithms/`: 0. Under `core/`: 0.
- this round added research/tree_qc_v2/ only; no file under research/tree_qc/, algorithms/ or core/ was modified.

The V1 verdict `PUBLIC_PILOT_NOT_LEARNABLE` is not revisited, reinterpreted or
softened anywhere in this round. No V1 threshold, checkpoint, generator, seed
or held-out result was touched, and the held-out partition is not read by any
module in `research/tree_qc_v2/`.

## 2. Public data actually used

| dataset | record | license | sites used | GT | annotated |
|---|---|---|---|---|---|
| TLS forest instance segmentation benchmark (Litchfield and Wytham plots) | Zenodo 16875688 | CC-BY-4.0 | litchfield, wytham | manual per-tree instance segmentation | yes |
| TreeInstSegEval (official benchmark evaluation code) | - | UNVERIFIED - NO LICENSE PUBLISHED | - | - | - |

1693 publisher per-tree PLY files are present, mean 174,207 points per tree.

| site | tree files | points | manifest digest |
|---|---|---|---|
| litchfield | 737 | 112,751,388 | `64f2d0112fe70072` |
| wytham | 956 | 182,180,502 | `3df3faa08d746811` |

Not present locally and therefore not covered: ofental, robsoncreek.

The two archive files were verified against the publisher MD5 before use;
the SHA-256 of each is recorded in the JSON report.

## 3. License audit

16 components audited, full detail in `research/tree_qc_v2/LICENSE_AUDIT.md`. every component that is actually used is CC-BY-4.0, MIT, Apache-2.0 or BSD-3-Clause; every non-commercial or unlicensed component is NOT_USED or REFERENCED_ONLY.

Used in this round:
- TLS forest instance segmentation benchmark: 2983 manually segmented trees from four plots
- PyTorch
- NumPy
- scikit-learn

Research-only or non-commercial components are `NOT_USED` or
`REFERENCED_ONLY`, so nothing restricted reaches this round. The findings
that would block a shipping product are listed in the audit itself.

## 4. Task A - SITE_FINGERPRINT_TEST

600 real good trees at 1024 points each, split by tree into train / validation / test. The same classifier architecture, the same
epoch budget and the same per-channel standardisation (fitted on the train
split) are used for every representation, so the comparison is about the
representation and not about capacity or conditioning.

| representation | channels | test site accuracy | macro F1 | litchfield recall | wytham recall |
|---|---|---|---|---|---|
| `V1_RAW` | 4 | 0.9417 | 0.9415 | 0.8833 | 1.0000 |
| `V2_NORMALIZED` | 7 | 0.8667 | 0.8653 | 0.7667 | 0.9667 |
| `V2_NORMALIZED_NO_META` | 4 | 0.7333 | 0.7333 | 0.7500 | 0.7167 |
| `V2_ANISO_NORMALIZED` | 7 | 0.8083 | 0.8082 | 0.7833 | 0.8333 |
| `V2_ANISO_NORMALIZED_NO_META` | 4 | 0.6583 | 0.6572 | 0.7167 | 0.6000 |

Random baseline for two sites: 0.50.

`V1_RAW` reproduces the frozen V1 input contract exactly: mean-centred
local metric XYZ divided by the research-wide constant 10, plus a
constant membership channel. It identifies the forest at
0.9417 on trees the classifier never saw,
which is direct evidence that the V1 representation carried a strong
site fingerprint.

`V2_NORMALIZED` (candidate-centred, per-cloud robust scaling, random XY
rotation, density standardisation, explicit global metric channel)
scores 0.8667.

`V2_NORMALIZED_NO_META` withholds the explicit metric channel and
scores 0.7333, so most of the remaining
site signal lives in that one named, auditable channel rather than
hidden in the coordinates.

`V2_ANISO_NORMALIZED` additionally removes aspect ratio and scores
0.8083; that is the declared ablation, not
a proposal, because aspect ratio is real tree geometry a refiner needs.

Read correctly: a lower number is not automatically better. The point is
that V1 carried a forest fingerprint, that the V2 representation moves the
scale information into one explicit channel, and that the refinement task
still has to work - which is what Task C tests.

## 5. Task B - PUBLIC_GT_PROPOSAL_TEST

**PASS**

Local windows of 12 m were cut from the real plot,
each containing the target tree and its neighbours. ROIs are the candidate
box grown in XY, so neighbour contamination and missing points are both
inside the region a refiner would see.

GT_COVERAGE_IN_ROI: mean 1.0, minimum 1.0, fraction below 0.95 0.0.

ROI size: median 223,428 points, p90 1,466,778.

- `REAL_ALGORITHM_PROPOSAL` pairs above IoU 0.1: **25** ({'litchfield': 13, 'wytham': 12}).
- `SIMULATED_PROPOSAL` rows: **375**.
- `GT_CORRECT` control rows: 25.

The real route runs the product's own `segment_stem_guided_hybrid` on real
public forest. It is a diagnostic of the crown/stem refinement stage, not a
run of the whole product pipeline, because public data forces two
substitutions that are recorded on every row:

- watershed crown labels -> GT_INSTANCE_LABELS
- height above ground -> local low-percentile grid proxy
- raw point weight -> uniform 1 per representative point

What the real route then shows is important and negative: it does not
repair. Handed clean ground-truth crowns it reproduces them exactly, and
handed perturbed crowns it returns what it was given for most rows.

| crown source | rows | median IoU | min IoU | fraction < 0.90 | median delta vs input | improved | unchanged |
|---|---|---|---|---|---|---|---|
| `GT_INSTANCE_LABELS` | 25 | 1.0000 | 1.0000 | 0.000 | +0.0000 | 0.000 | 1.000 |
| `CROWN_MERGE` | 21 | 1.0000 | 0.6449 | 0.143 | +0.0000 | 0.000 | 0.762 |
| `CROWN_SPLIT` | 28 | 1.0000 | 0.5000 | 0.214 | +0.0000 | 0.000 | 0.786 |

So the honest statement is: public data can form the task, but the real
algorithm route contributes almost no defective proposals. The pairs the
refiner is trained and tested on are therefore simulated corruptions,
marked `SIMULATED_PROPOSAL` on every row and never mixed with the real ones.

Simulated proposals: 375 rows, median IoU 0.75, merge fraction 0.261, incomplete fraction 0.352.

| corruption | rows | median IoU | min IoU |
|---|---|---|---|
| `ADD_CONTAMINATION` | 75 | 0.8000 | 0.6897 |
| `CROWN_INCOMPLETE` | 75 | 0.7500 | 0.5500 |
| `MERGED_NEIGHBOR` | 75 | 0.5774 | 0.3784 |
| `REMOVE_MISSING` | 75 | 0.7500 | 0.5500 |
| `STEM_MISSING` | 75 | 0.8416 | 0.5500 |

| severity | rows | median IoU |
|---|---|---|
| HIGH | 125 | 0.5500 |
| LOW | 125 | 0.9000 |
| MEDIUM | 125 | 0.7500 |

`MISSING_RECOVERABLE_FRACTION`: mean 0.5625, minimum 0.0. A missing point outside the ROI can never be
recovered by any refiner, so this bounds what repair could achieve.

## 6. Task C - MINI_REFINER_BAKEOFF

308 ROI targets ({'litchfield': 190, 'wytham': 118}) from a library whose index digest is `ac61a8c3f515b675`.

**MINI_REFINER_BAKEOFF status: COMPLETE_WITH_LIMITATION**

- Cross-site folds run: cross_site_test_wytham
- Cross-site folds missing: cross_site_test_litchfield (of 2 needed for a full rotation)
- In-site diagnostic folds: in_site_test_litchfield
- Why: The three architectures share one epoch budget, which is what makes them comparable. At that budget a second full cross-site fold did not fit the round's time window, so only Litchfield -> Wytham was trained. The protocol requires at least one unseen-site test; that requirement is met, and the missing reverse fold is recorded rather than replaced by an in-site number.
- What is not affected: Within the configurations that did run, every architecture was trained to the same budget, scored on the same full ROIs, and put through the same gates, so the comparison between the three architectures and the cross-site versus in-site contrast both stand as reported.

### Protocol shared by all three models

- **optimiser**: Adam
- **learning_rate**: 0.001
- **scheduler**: cosine annealing over the epoch budget
- **batch_size**: 8
- **early_stop_patience**: 8
- **early_stop_min_delta**: 0.0001
- **decision_threshold**: 0.5
- **selection_metric**: validation loss on held-in trees only
- **augmentation**: random Z rotation + 1 cm jitter, identical for all models
- **sampler**: stratified proposal / target-only / other-tree / background
- **evaluation**: full ROI, not the training sample
- **loss**: balanced BCE; MODEL_C adds offset L1 on target points and mask-quality MSE
- **evaluation cap**: 200,000 points per ROI

### The three candidates

| model | reference | parameters | auxiliary heads | implementation |
|---|---|---|---|---|
| `A` PointNet++ (Qi et al. 2017), SSG variant | - | 485,475 | no | LOCAL_IMPLEMENTATION (plain torch, no external code) |
| `B` PointNeXt-S (Qian et al. 2022), inverted residual blocks | - | 1,535,139 | no | LOCAL_IMPLEMENTATION (plain torch, no external code) |
| `C` PointNeXt-S + TreeLearn-style stem offset + mask quality head | - | 1,566,957 | yes | LOCAL_IMPLEMENTATION (plain torch, no external code) |

`MODEL_C` is a strict superset of `MODEL_B`: the same backbone and the same
membership head plus an offset head and a mask-quality head, verified by a
test that compares the shared tensors. So any difference between B and C
comes from the auxiliary supervision, not from extra main-path capacity.

### Cross-site results

The verdict is decided on the cross-site folds. The in-site configuration
is a diagnostic that separates "these architectures cannot refine at all"
from "they cannot transfer", and it never substitutes for a cross-site
number.

Only one cross-site fold was run: the model is trained on Litchfield
trees and tested on Wytham, a different forest, a different species mix
and a different acquisition. That is the unseen-site evidence. The
reverse fold, `cross_site_test_litchfield`, was not run: with three
architectures at a shared epoch budget a second fold exceeded the time
this round had. One unseen-site test is what the protocol requires, so
the requirement is met, and the missing fold is recorded here and on the
stage rather than being replaced by an in-site number or backfilled
after the fact.

#### Cross-site: `cross_site_test_wytham` - test site `wytham` (trained on litchfield)


Train 156 / validation 34 / test 118 targets. The test site was never used to early-stop, normalise,
augment or pick a threshold.

| model | current IoU | refined IoU | delta IoU | median delta | change-region F1 | identity copy | good preservation |
|---|---|---|---|---|---|---|---|
| `A` | 0.7686 | 0.6201 | -0.1486 | -0.0938 | 0.2296 | PASS | FAIL |
| `B` | 0.7686 | 0.7083 | -0.0603 | -0.0116 | 0.2624 | PASS | FAIL |
| `C` | 0.7686 | 0.7147 | -0.0540 | -0.0064 | 0.2885 | PASS | FAIL |

Best on this site: `C`. On bad proposals (current IoU below 0.80): mean delta 0.0134, improved by more than 0.02 in 0.3900 of rows. On good proposals: mean delta -0.1272, degraded by more than 0.05 in 0.4387 of rows, 303 severe cases.

#### In-site: `in_site_test_litchfield` - test site `litchfield` (trained on litchfield)

Both the training trees and the test trees come from this one
forest, and the split is by tree, so no tree appears in two
splits. The site is *named* on both sides; what is held out is the
trees, not the forest. This configuration answers "can the
architectures refine at all", not "does the refinement transfer".


Train 133 / validation 28 / test 29 targets. The test site was never used to early-stop, normalise,
augment or pick a threshold.

| model | current IoU | refined IoU | delta IoU | median delta | change-region F1 | identity copy | good preservation |
|---|---|---|---|---|---|---|---|
| `A` | 0.7727 | 0.8801 | +0.1073 | +0.0893 | 0.4729 | PASS | FAIL |
| `B` | 0.7727 | 0.8666 | +0.0938 | +0.0870 | 0.4594 | PASS | FAIL |
| `C` | 0.7727 | 0.8580 | +0.0853 | +0.0790 | 0.4592 | PASS | FAIL |

Best on this site: `A`. On bad proposals (current IoU below 0.80): mean delta 0.1933, improved by more than 0.02 in 0.8229 of rows. On good proposals: mean delta 0.0137, degraded by more than 0.05 in 0.1124 of rows, 14 severe cases.

### Identity-copy and GOOD-preservation gates

| model | configuration | site | change-region F1 | identity-copy fraction | gate | good mean delta | good degraded > 0.05 | severe | gate |
|---|---|---|---|---|---|---|---|---|---|
| `A` | `cross_site_test_wytham` | wytham | 0.2296 | 0.0000 | PASS | -0.2584 | 0.7332 | 511 | FAIL |
| `B` | `cross_site_test_wytham` | wytham | 0.2624 | 0.0312 | PASS | -0.1340 | 0.4990 | 318 | FAIL |
| `C` | `cross_site_test_wytham` | wytham | 0.2885 | 0.0142 | PASS | -0.1272 | 0.4387 | 303 | FAIL |
| `A` | `in_site_test_litchfield` | litchfield | 0.4729 | 0.0885 | PASS | 0.0137 | 0.1124 | 14 | FAIL |
| `B` | `in_site_test_litchfield` | litchfield | 0.4594 | 0.0654 | PASS | -0.0013 | 0.1727 | 20 | FAIL |
| `C` | `in_site_test_litchfield` | litchfield | 0.4592 | 0.0827 | PASS | -0.0081 | 0.1647 | 20 | FAIL |

A refiner that returns `current_mask` unchanged scores a perfect current
IoU and zero on every change-region metric. The gates exist because the
whole point of V2 is to change the mask, in the right direction, without
breaking the proposals that were already right.

### Change-region and added/removed point metrics

| model | site | change F1 | change recall | added F1 | removed F1 |
|---|---|---|---|---|---|
| `A` | wytham | 0.2296 | 0.4895 | 0.2066 | 0.1235 |
| `B` | wytham | 0.2624 | 0.4188 | 0.1670 | 0.1459 |
| `C` | wytham | 0.2885 | 0.4506 | 0.1930 | 0.1385 |
| `A` | litchfield | 0.4729 | 0.4928 | 0.3970 | 0.2609 |
| `B` | litchfield | 0.4594 | 0.4888 | 0.4195 | 0.2478 |
| `C` | litchfield | 0.4592 | 0.4799 | 0.3684 | 0.2506 |

### Per-defect breakdown

| model | site | corruption | rows | current IoU | refined IoU | delta | improved | identity copy |
|---|---|---|---|---|---|---|---|---|
| `A` | wytham | `ADD_CONTAMINATION` | 354 | 0.7967 | 0.7162 | -0.0805 | 0.444 | 0.000 |
| `A` | wytham | `ALGORITHM_OUTPUT` | 224 | 0.9057 | 0.6565 | -0.2493 | 0.107 | 0.000 |
| `A` | wytham | `CROWN_INCOMPLETE` | 354 | 0.7348 | 0.5891 | -0.1457 | 0.331 | 0.000 |
| `A` | wytham | `MERGED_NEIGHBOR` | 354 | 0.5721 | 0.4806 | -0.0916 | 0.110 | 0.000 |
| `A` | wytham | `REMOVE_MISSING` | 354 | 0.7332 | 0.6559 | -0.0774 | 0.435 | 0.000 |
| `A` | wytham | `STEM_MISSING` | 354 | 0.8424 | 0.6127 | -0.2297 | 0.186 | 0.000 |
| `B` | wytham | `ADD_CONTAMINATION` | 354 | 0.7967 | 0.7980 | +0.0013 | 0.571 | 0.000 |
| `B` | wytham | `ALGORITHM_OUTPUT` | 224 | 0.9057 | 0.7476 | -0.1581 | 0.107 | 0.040 |
| `B` | wytham | `CROWN_INCOMPLETE` | 354 | 0.7348 | 0.6598 | -0.0750 | 0.407 | 0.023 |
| `B` | wytham | `MERGED_NEIGHBOR` | 354 | 0.5721 | 0.5419 | -0.0303 | 0.246 | 0.119 |
| `B` | wytham | `REMOVE_MISSING` | 354 | 0.7332 | 0.7609 | +0.0277 | 0.582 | 0.000 |
| `B` | wytham | `STEM_MISSING` | 354 | 0.8424 | 0.7212 | -0.1212 | 0.311 | 0.003 |
| `C` | wytham | `ADD_CONTAMINATION` | 354 | 0.7967 | 0.8118 | +0.0151 | 0.641 | 0.000 |
| `C` | wytham | `ALGORITHM_OUTPUT` | 224 | 0.9057 | 0.7630 | -0.1428 | 0.121 | 0.031 |
| `C` | wytham | `CROWN_INCOMPLETE` | 354 | 0.7348 | 0.6687 | -0.0661 | 0.466 | 0.006 |
| `C` | wytham | `MERGED_NEIGHBOR` | 354 | 0.5721 | 0.5402 | -0.0319 | 0.102 | 0.042 |
| `C` | wytham | `REMOVE_MISSING` | 354 | 0.7332 | 0.7498 | +0.0166 | 0.562 | 0.000 |
| `C` | wytham | `STEM_MISSING` | 354 | 0.8424 | 0.7364 | -0.1061 | 0.432 | 0.000 |
| `A` | litchfield | `ADD_CONTAMINATION` | 87 | 0.8321 | 0.9519 | +0.1198 | 0.885 | 0.034 |
| `A` | litchfield | `ALGORITHM_OUTPUT` | 56 | 0.9054 | 0.9012 | -0.0042 | 0.179 | 0.429 |
| `A` | litchfield | `CROWN_INCOMPLETE` | 87 | 0.7334 | 0.9156 | +0.1821 | 0.943 | 0.011 |
| `A` | litchfield | `MERGED_NEIGHBOR` | 87 | 0.6508 | 0.6685 | +0.0177 | 0.552 | 0.034 |
| `A` | litchfield | `REMOVE_MISSING` | 87 | 0.7332 | 0.9655 | +0.2322 | 0.943 | 0.000 |
| `A` | litchfield | `STEM_MISSING` | 87 | 0.7529 | 0.8596 | +0.1067 | 0.920 | 0.000 |
| `B` | litchfield | `ADD_CONTAMINATION` | 87 | 0.8321 | 0.9586 | +0.1264 | 0.874 | 0.034 |
| `B` | litchfield | `ALGORITHM_OUTPUT` | 56 | 0.9054 | 0.8946 | -0.0108 | 0.196 | 0.286 |
| `B` | litchfield | `CROWN_INCOMPLETE` | 87 | 0.7334 | 0.9242 | +0.1907 | 0.920 | 0.000 |
| `B` | litchfield | `MERGED_NEIGHBOR` | 87 | 0.6508 | 0.5718 | -0.0790 | 0.483 | 0.034 |
| `B` | litchfield | `REMOVE_MISSING` | 87 | 0.7332 | 0.9622 | +0.2290 | 0.920 | 0.000 |
| `B` | litchfield | `STEM_MISSING` | 87 | 0.7529 | 0.8764 | +0.1235 | 0.839 | 0.000 |
| `C` | litchfield | `ADD_CONTAMINATION` | 87 | 0.8321 | 0.9436 | +0.1114 | 0.862 | 0.034 |
| `C` | litchfield | `ALGORITHM_OUTPUT` | 56 | 0.9054 | 0.8773 | -0.0281 | 0.179 | 0.339 |
| `C` | litchfield | `CROWN_INCOMPLETE` | 87 | 0.7334 | 0.8890 | +0.1556 | 0.897 | 0.023 |
| `C` | litchfield | `MERGED_NEIGHBOR` | 87 | 0.6508 | 0.6132 | -0.0376 | 0.483 | 0.034 |
| `C` | litchfield | `REMOVE_MISSING` | 87 | 0.7332 | 0.9568 | +0.2236 | 0.943 | 0.000 |
| `C` | litchfield | `STEM_MISSING` | 87 | 0.7529 | 0.8502 | +0.0973 | 0.862 | 0.034 |

### Resource measurements

| model | parameters | GPU peak MB | single-ROI GPU s | single-ROI CPU s |
|---|---|---|---|---|
| `A` | 485,475 | 42.8 | 0.0889 | 0.1846 |
| `B` | 1,535,139 | 55.2 | 0.0759 | 0.1795 |
| `C` | 1,566,957 | 86.5 | 0.0494 | 0.1861 |

None of these is a product gate in this round. They are recorded because
the eventual target is a Windows desktop build where a 24 GB accelerator
cannot be assumed, so an architecture that only works at one scale is a
worse candidate even at equal IoU. Both a CUDA and a CPU forward pass were
run for every model.

## 7. Verdict

**V2_FEASIBILITY_PARTIAL**

- V1 site/domain problem diagnosed: yes (`V1_RAW` site accuracy 0.9417 against a
  0.50 random baseline).
- Public GT can form the task: PASS, with 25 real and 375 simulated proposal pairs.
- Best cross-site delta IoU: -0.0540.
- Best in-site delta IoU (diagnostic): 0.1073.
- Best architecture: MODEL_C_POINTNEXT_S_MULTITASK.

Some of the conditions are met and at least one is not. The specific gaps
are listed under BLOCKERS and in the per-model tables above. No model was
added, no epoch budget was raised and no test site was changed to close
them.

Read the two configurations together. Cross-site, every architecture
degrades the mask on average and the GOOD-preservation gate fails badly:
the learned transformation does not transfer to a forest the model never
saw. In-site, the same architectures and the same budget improve bad
proposals substantially, which means the task is learnable and the
pipeline is sound; what fails is generalisation, and separately the
handling of proposals that were already good.

Per the protocol this round stops here. It does not proceed to a formal V2
training run, does not add a fourth or fifth model, does not raise the epoch
budget, and does not touch the TreeLineWarning local corpus.

## 8. Known limitations

- Only two of the four named public sites are present locally. Ofental and Robson Creek were not downloaded, so no statement here covers them.
- The real-algorithm route substitutes the publisher ground-truth instance labels for the watershed crown labels the product coarse stage would produce, and a local low-percentile grid proxy for the product ground classifier. It therefore measures the crown/stem refinement stage, not the whole product pipeline end to end.
- The real-algorithm route repairs almost nothing: run with clean crowns it reproduces the ground truth exactly (median delta IoU 0.0000, 100 percent unchanged), and run with perturbed crowns it returns what it was given for 76 to 79 percent of rows. The pairs the refiner is trained and tested on are therefore simulated corruptions, marked SIMULATED_PROPOSAL everywhere.
- The ROI library stores at most 120 000 points per target by a deterministic spatial stride; the stride is recorded per row.
- Region-level offsets use a stem/base proxy - the median XYZ of the lowest 10 percent of the instance by height above the local ground proxy - because the benchmark publishes no stem axis. It is a proxy and is never called ground truth.
- Offset supervision is applied only where the ground truth says a point belongs to the target, because for any other point the true offset is unknown. Modelling it as a keep-out offset would be an assumption, not a label.
- The refiner decides membership at a fixed 0.5 probability threshold. No threshold was tuned, on the test site or anywhere else.
- The mini bakeoff is small by design: it is an architecture elimination round, not a trained production model.
- Only one cross-site fold was trained. The reverse fold, and a full leave-one-site-out rotation, were not run: with three architectures at a shared 20-epoch budget a second fold exceeded the round's time. One unseen-site test is what the protocol asks for, but two would be stronger evidence and the gap is recorded here rather than implied away.
- The in-site configuration trains and tests on the same forest (different trees). It is a diagnostic of whether refinement is learnable at all and must not be read as generalisation evidence.

## 9. Blockers

- The product CROWN_STEM_HYBRID backend cannot consume a public TLS plot directly: it requires a SpanRecord, a vegetation region, a coarse ground cloud and scene metadata that the benchmark does not publish. Only its stem-guided refinement function was reachable, which is why the real pair count is a diagnostic and the training pairs are simulated.

## 10. Reproducing this round

```
set PYTHONPATH=research/tree_qc_v2
python research/tree_qc_v2/recon.py
python research/tree_qc_v2/v2_site_fingerprint.py --points 1024 --trees-per-site 300
python research/tree_qc_v2/v2_proposal_test.py --window-m 12 --minimum-points 8000
python research/tree_qc_v2/v2_roi_library.py
python research/tree_qc_v2/v2_bakeoff.py --points 2048 --epochs 30
python research/tree_qc_v2/v2_report.py
pytest research/tree_qc_v2/test_v2_feasibility.py
```

Config digest for the whole round: `44ac652b7a6018ba86cdd400798b0af53e4628c6d8c92de91a1331d9d8dd9794`.
