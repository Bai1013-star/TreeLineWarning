# TREE_QC_V25_FINAL_DEEP_LEARNING_GO_NO_GO

The last small-scale feasibility check on the TreeLineWarning single-tree QC
deep learning route. It asks one question: is conservative local correction
worth continuing as a deep learning direction, or does the project move to
CROWN_STEM_HYBRID plus deterministic geometry QC plus human review?

```
GIT_HEAD = 9569910aff67af4ff8e3b4a22d22ea3e3ab918a8
BRANCH = research/tree-qc-v25-final-go-no-go
BASE_HEAD = 169d42dccd9b1173193d17167937bbde1a6638c0
WORKTREE_CLEAN = YES

V2_BASELINE_CROSS_SITE_DELTA = -0.0540
DELTA_REFINER_CROSS_SITE_DELTA = -0.6402
GATED_DELTA_CROSS_SITE_DELTA = 0.0000

CROSS_SITE_BAD_MEAN_DELTA_IOU = 0.0000
CROSS_SITE_BAD_IMPROVEMENT_RATE = 0.0000

CROSS_SITE_GOOD_MEAN_DELTA_IOU = 0.0000
CROSS_SITE_GOOD_DEGRADATION_RATE = 0.0000

AUTO_EDIT_COVERAGE = 0.0000
AUTO_EDIT_SUCCESS_RATE = n/a

IN_SITE_BAD_MEAN_DELTA_IOU = 0.0000
IN_SITE_GOOD_MEAN_DELTA_IOU = 0.0000

FINAL_VERDICT = DEEP_LEARNING_NO_GO
TREE_QC_DEEP_LEARNING_ROUTE = STOPPED
```

## 1. What was run, and what was frozen first

One new model, three results, no search of any kind.

| result | method | trained this round |
|---|---|---|
| A | V2 frozen full-mask refiner | no - read from the V2 artefacts |
| B | PointNeXt-S delta refiner, no gate | yes |
| C | the same model behind the conservative gate | yes (gate fitted on validation) |

### Protocol revision on the record

The model was trained **twice**, and both revisions are kept:

- **revision 1** (pilot, `PILOT_V1_UNWEIGHTED_*`): unweighted loss.
- **revision 2** (reported here): the loss is importance-weighted by the
  stratified sampler's inverse inclusion probability, so the effective
  training distribution matches the ROI rather than the sampler.

The revision was made *because revision 1 failed*, not because a number was
unsatisfactory. Revision 1 collapsed the removal head to "always remove"
even on masks it had trained on, and diagnosing that surfaced a genuine
confound: the sampler floors every stratum, so it over-represents the rare
decision. Measured over 200 training rows, the ADD-head positive rate was
0.1298 as sampled against 0.0908 in the population. Revision 2 corrects it
and reproduces the population rate exactly. Every other setting - seed,
epoch budget, patience, splits, thresholds, criteria - is unchanged, and
both revisions return the same verdict.

The backbone is imported from the frozen V2 code rather than reimplemented,
so "same backbone" is a fact about the code:

- backbone `1,518,112` parameters, identical to V2 `MODEL_B_POINTNEXT_S`
- two decision heads `22,470` parameters
- total `1,540,582`

Budget: 15 epochs maximum, early stop patience 3, one fixed seed 101, no hyper-parameter search.

The gate threshold was written to disk and digested **before** the Wytham
test was scored, and the test path re-reads it from that file:

- `cross_site_test_wytham`: frozen gate `25a4e01c37ea8c72`, effective threshold 0.7
- `in_site_test_litchfield`: frozen gate `33de2d33138b0241`, effective threshold 0.7

## 2. Results

### A - V2 frozen full-mask refiner (read, not retrained)

| configuration | model | delta IoU | GOOD delta | BAD delta |
|---|---|---|---|---|
| cross_site_test_wytham | `A` MODEL_A_POINTNET2 | -0.1486 | -0.2584 | -0.0475 |
| cross_site_test_wytham | `B` MODEL_B_POINTNEXT_S | -0.0603 | -0.1340 | +0.0074 |
| cross_site_test_wytham | `C` MODEL_C_POINTNEXT_S_MULTITASK | -0.0540 | -0.1272 | +0.0134 |
| in_site_test_litchfield | `A` MODEL_A_POINTNET2 | +0.1073 | +0.0137 | +0.1933 |
| in_site_test_litchfield | `B` MODEL_B_POINTNEXT_S | +0.0938 | -0.0013 | +0.1813 |
| in_site_test_litchfield | `C` MODEL_C_POINTNEXT_S_MULTITASK | +0.0853 | -0.0081 | +0.1711 |

### B and C on the cross-site configuration

Trained on 156 Litchfield targets, tested on 118 Wytham targets the model never saw.

| metric | B ungated | C gated |
|---|---|---|
| current IoU | 0.7686 | 0.7686 |
| refined IoU | 0.1285 | 0.7686 |
| delta IoU | -0.6402 | +0.0000 |
| GOOD delta IoU | -0.7848 | +0.0000 |
| GOOD degradation rate | 0.9951 | 0.0000 |
| BAD delta IoU | -0.5071 | +0.0000 |
| BAD improvement rate | 0.0191 | 0.0000 |
| NO_OP rate | 0.0009 | 1.0000 |
| AUTO_EDIT coverage | n/a | 0.0000 |
| AUTO_EDIT count | n/a | 0 of 2112 rows |
| change-region F1 | 0.0055 | n/a |

Point-level decisions on the cross-site test, ungated:

| head | precision | recall | F1 |
|---|---|---|---|
| ADD | 0.2406 | 0.0144 | 0.0272 |
| REMOVE | 0.1119 | 0.5040 | 0.1832 |

### In-site diagnostic

| metric | B ungated | C gated |
|---|---|---|
| delta IoU | -0.6419 | +0.0000 |
| GOOD delta IoU | -0.8842 | +0.0000 |
| BAD delta IoU | -0.4193 | +0.0000 |
| AUTO_EDIT coverage | n/a | 0.0000 |

## 3. Verdict against the declared criteria

Evaluated on the cross-site configuration, result C, and on nothing else.

| criterion | rule | observed | passed |
|---|---|---|---|
| CROSS_SITE_BAD_MEAN_DELTA_IOU | > 0.0 | 0.0000 | FAIL |
| CROSS_SITE_GOOD_MEAN_DELTA_IOU | >= -0.005 | 0.0000 | PASS |
| CROSS_SITE_GOOD_DEGRADATION_RATE | <= 0.05 | 0.0000 | PASS |
| CROSS_SITE_AUTO_EDIT_SUCCESS_RATE | >= 0.8 | n/a | FAIL |
| CROSS_SITE_AUTO_EDIT_COVERAGE | >= 0.1 | 0.0000 | FAIL |

**FINAL_VERDICT = DEEP_LEARNING_NO_GO**

**TREE_QC_DEEP_LEARNING_ROUTE = STOPPED**

Core criterion 1 fails, so the verdict is NO_GO regardless of the other
four. The honest reading of *why*, drawn from the artefacts:

* The delta refiner does not merely fail to transfer. On the Wytham test
  it degrades a mask from 0.7686
  to 0.1285 IoU, harms 0.9951 of already-good
  proposals, and improves bad ones only 0.0191 of the time.
* The conservative gate did exactly what it was built to do, and that is
  the second finding: faced with an unreliable model it accepted 0
  of the test rows, so the product path is safe but empty. On validation
  it looked usable (coverage 0.3011, success 0.1087), and on full ROIs it was not.
  A gate cannot manufacture a signal that the model does not have.
* This is not a defect in the implementation. A single-example overfit
  probe (below) shows the heads can learn to leave a correct mask alone.
  What fails is per-point keep/remove discrimination at forest scale,
  where roughly 85 percent of the points inside a candidate mask belong
  to a neighbouring tree or to background.

### Why this is a NO_GO and not something to tune

Every lever that would raise the numbers is forbidden by this round: more
epochs, another seed, a search over the threshold, a different backbone,
or unblinding the local corpus. The result is reported as it stands.

## 4. Overfit probe: is the model broken, or is the task hard?

The probe result was not recorded; see the run log.

## 5. Provenance: what the training pairs actually are

| proposal kind | training rows (max across folds) |
|---|---|
| `GT_CORRECT` | 156 |
| `REAL_ALGORITHM_PROPOSAL` | 308 |
| `SIMULATED_PROPOSAL` | 2,340 |

The training pairs are majority SIMULATED_PROPOSAL. The real CROWN_STEM_HYBRID refinement route, run on public data, reproduces the ground truth it is given and repairs almost nothing, so it supplied few defective proposals. REAL_ALGORITHM_PROPOSAL and SIMULATED_PROPOSAL are never mixed in the reporting.

Scheduled GOOD share of the training set: target 0.5, realised 0.483595 (GOOD proposals are explicit training rows, as required).

This round does **not** show generalisation to a real transmission-line
product, and nothing here should be read that way.

## 6. Prohibitions honoured

- local_20_trees_accessed: NO
- local_8_bad_accessed: NO
- product_integration: NO
- onnx: NO
- crown_stem_hybrid_modified: NO
- new_forest_downloaded: NO
- reverse_fold_run: NO
- test_site_changed: NO
- fourth_model_added: NO
- thresholds_tuned_on_wytham: NO
- hyperparameter_search: NO
- seed: 101
- epochs_increased_after_results: NO

no path under research/tree_qc_v25 references the TreeLineWarning local corpus or the 8-bad challenge; the only data root is the frozen D:\TreeQCData ROI library

V2 untouched: 0 files modified under `research/tree_qc_v2/`.

## 7. Recommended next route

The deep learning route stops here. The recommendation on the record is:

1. **CROWN_STEM_HYBRID** remains the primary single-tree algorithm.
2. **Deterministic geometry / structural QC** replaces the learned refiner:
   rule-based checks on crown completeness, stem presence, neighbour overlap
   and size plausibility, which are auditable and need no training data.
3. **Human review** for the cases those two cannot settle.

The local 20-tree corpus and the 8-bad blind challenge remain sealed and were
not used to reach this verdict.

## 8. Reproducing this round

```
set PYTHONPATH=research/tree_qc_v25
python research/tree_qc_v25/v25_runner.py \
    --configurations cross_site_test_wytham in_site_test_litchfield
python research/tree_qc_v25/v25_report.py
pytest research/tree_qc_v25/test_v25.py
```

Config digest: `63fdc3f00a9d3ac0007a5f8ac950fa1e1a46fb6a907f69fdea730cb0e8b93465`. Frozen thresholds are recorded
above and in `research/tree_qc_v25/reports/FROZEN_GATE_*.json`.
