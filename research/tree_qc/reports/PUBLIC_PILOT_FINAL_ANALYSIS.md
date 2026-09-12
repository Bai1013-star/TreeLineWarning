PUBLIC_PILOT_FINAL_ANALYSIS
===========================

Round:   TREE_QC_EFG_ONE_CLICK_PUBLIC_PILOT_V1
Run:     public_pilot_v1_20260912T012641_fc75c8df
Base:    1c4ed5b96e4ef1053bf9d0e47d597911ffbca727
Head:    e4558a92c4f40764e2fbe23a579f8d593352a79d  (research/tree-qc-efg-public-pilot-v1)

VERDICT: PUBLIC_PILOT_NOT_LEARNABLE

This is a completed experiment with a negative result. Nothing in it was tuned after
unblinding, no gate was relaxed, and no reported number is estimated. Every value below
is read from an artifact the run wrote, or recomputed from the raw per-example scores the
run stored.


1. EXECUTION SUMMARY
--------------------

All nineteen stages completed:

  Environment            PASS      environment fingerprint, disk guard 35 GiB
  Gate D acceptance      PASS      binding re-verified, training_allowed true
  Resource calibration   PASS      batch probe 16/8/4/2/1 -> safe 8 for both networks
  Dataset fingerprint    PASS      440 train / 175 validation / 749 held-out (locked)
  Training data prewarm  PASS      9222/9240 train, 3672/3675 validation generated
  Held-out adapter parity PASS     63 + 105 examples, zero mismatches
  PointNet seed101       PASS      validation BA 0.8551, BAD_RECALL 0.9596
  PointNet++ seed101     PASS      validation BA 0.8951, BAD_RECALL 0.9978
  Initial learnability   PASS      both above the floors; held-out stayed locked
  Architecture selection PASS      pointnet2_ssg (validation only)
  Three model seeds      PASS      101 reused from comparison, 202 and 303 trained
  Threshold freeze       PASS      0.0050 / 0.0150 / 0.1000, validation only
  Deployment candidate   PASS      seed 101, chosen before unblinding
  Sampling probe         PASS      validation-only 2048 probe
  Experiment freeze      PASS      717fe17036f90eaf3fb2d0ea7aecbc4929dbf1d8c3a3c8b6c6b9277d2ad3e37a
  Held-out evaluation    PASS      3 seeds x 15611 scored examples
  Multi-seed aggregation PASS      MODEL_STABILITY PASS
  Sampling robustness    PASS      max BAD_RECALL drop 0.0 pp
  Final report           PASS


2. THE HELD-OUT RESULT
----------------------

Primary gate, moderate/high defects, deployment candidate seed 101:

  BAD_RECALL          0.996749   target >= 0.85   PASS
  BAD_FALSE_PASS      0.003251   target <= 0.15   PASS
  GOOD_FALSE_REVIEW   0.841121   target <= 0.25   FAIL
  BALANCED_ACCURACY   0.577814   target >= 0.80   FAIL
  F1_REVIEW           0.932410

Multi-seed (each seed at its own frozen threshold, same held-out set):

  seed 101   BR 0.996749   BFP 0.003251   GFR 0.841121   BA 0.577814
  seed 202   BR 0.983296   BFP 0.016704   GFR 0.832443   BA 0.575426
  seed 303   BR 0.992152   BFP 0.007848   GFR 0.878505   BA 0.556824

  BAD_RECALL mean 0.990732, min 0.983296, max 0.996749, std 0.005583  -> MODEL_STABILITY PASS

The failure is not a weak model. It is an over-eager one: on held-out trees it flags
84 percent of *good* segmentations for human review while catching essentially every
defective one. F1_REVIEW of 0.932 looks strong only because the defect base rate is
85.7 percent; balanced accuracy, which is what the gate uses, exposes it.

Per defect on the primary severities (seed 101), all six classes are detected:

  MERGED_MULTI_TREE        n=1493   BAD_RECALL 0.993972
  NEIGHBOR_CONTAMINATION   n=1497   BAD_RECALL 1.000000
  OVER_SEGMENTED           n=1498   BAD_RECALL 0.989987
  CROWN_INCOMPLETE         n=1498   BAD_RECALL 1.000000
  STEM_MISSING             n=1436   BAD_RECALL 0.996518
  NOISE                    n=1498   BAD_RECALL 1.000000

So the "critical defects must be meaningful" condition is met. What fails is the
specificity that makes the QC signal usable at all.

There is no severity effect and no class effect: GOOD_FALSE_REVIEW is 0.8451 / 0.8371 /
0.8398 on HIGH / MEDIUM / LOW, and BAD_RECALL is 0.99+ in every class. The failure is
uniform, which points at the operating point rather than at any particular defect type.


3. WHY IT FAILED: THE SCORE DISTRIBUTIONS
-----------------------------------------

Recomputed from the stored per-example scores (seed 101, threshold 0.0050):

  HELD-OUT     GOOD  mean 0.655  median 0.983  p90 1.000   fraction above threshold 0.841
               BAD   mean 0.974  median 1.000  p10 0.994   fraction above threshold 0.996

  VALIDATION   GOOD  mean 0.124  median 0.001  p90 0.877   fraction above threshold 0.208
               BAD   mean 0.987  median 1.000  p10 0.996   fraction above threshold 0.998

The two populations barely move for defective trees (median ~1.0 in both) and invert for
good trees: the typical good validation tree scores 0.001, the typical good held-out tree
scores 0.983. The model assigns high defect probability to a good single-tree segmentation
on the held-out site, so any threshold that keeps good false reviews inside the budget
destroys bad recall, and any threshold that keeps bad recall high flags almost every good
tree. At the frozen operating point the model is effectively saying "review this" for
everything: 14758 of 15611 examples score REVIEW_REQUIRED, 853 fall in the pre-declared
plus/minus 0.10 uncertain band, and none score PASS.

This is the failure the held-out protocol exists to catch, and it was caught. Validation
looked acceptable (GOOD_FALSE_REVIEW 0.194 to 0.221 across the three seeds, BALANCED_ACCURACY
0.88 to 0.90), the freeze was written and hashed, and the held-out site then showed the
learned decision boundary does not transfer. The train/validation site and the held-out
site are different forests, and the model fit the first one.

The uncertain band does not rescue it: UNCERTAIN is 5.5 percent of held-out examples and
is never counted as a detected defect, so it changes neither binary metric.


4. WHAT THIS DOES AND DOES NOT SAY
----------------------------------

Does say: PointNet and PointNet++ SSG, trained under this generator with this budget,
learn to detect every synthetic defect class reliably on the held-out site (BAD_RECALL
0.98 to 1.00, BAD_FALSE_PASS 0.003 to 0.017) and do not learn a good/defect boundary that
transfers to that site (BALANCED_ACCURACY 0.56 to 0.58). A QC model that flags 84 percent
of good trees is not deployable at any threshold.

Does not say: that Tree QC is impossible. Two specific things were not tested and would
change the picture if fixed:

  (a) The good/defect asymmetry is partly a corpus property. The frozen task is 85.7
      percent defective by construction: for each candidate and severity the generator
      emits one GOOD example and six defect examples. The operating point therefore sits
      where a single good-tree score excursion costs little on validation, and the frozen
      selection rule (max BAD_RECALL subject to GOOD_FALSE_REVIEW <= 0.25) picks the
      lowest threshold that satisfies the budget on validation. That rule is honest and
      was declared in advance; it is also the reason the chosen thresholds are 0.005,
      0.015 and 0.100 rather than near 0.5.

  (b) Nothing here separates "the model cannot represent this boundary" from "the
      threshold transfer is wrong". The held-out good scores are bimodal (median 0.983
      but mean 0.655), so a fraction of good trees are scored correctly and confidently.
      No threshold satisfies both constraints on held-out, so the gap is in the scores,
      not only in the cut point.

Neither (a) nor (b) was acted on. Acting on either after unblinding would be exactly the
tuning the protocol forbids, and the freeze digest is still
717fe17036f90eaf3fb2d0ea7aecbc4929dbf1d8c3a3c8b6c6b9277d2ad3e37a.


5. PROTOCOL INTEGRITY
---------------------

  GATE_D_ACCEPTANCE                PASS
  GATE_D_GENERATOR_DIGEST          a0550c17ff75cdf7d380fc677840c883f2262f2148a40edbbee0df35410b6b9d
  GATE_D_PROTOCOL_DIGEST          36e4cfca043190d4d12c2ab9ee635d92bb479cb9b56d482ae1121fd797939729
  GATE_D_CORE_CHANGED              NO
  HELDOUT_ADAPTER_PARITY           PASS   (63 examples @4096 and 105 @2048, zero mismatches)
  HELDOUT_ACCESSED_BEFORE_FREEZE   NO
  HELDOUT_ACCESSED                 YES, first access 2026-09-12T04:43:31+08:00
  FREEZE_UNCHANGED_SINCE_UNBLIND   YES (same digest recorded at unblinding and at audit)
  HELDOUT_CANDIDATES               749 (expected 749, actual 749)
  HELDOUT_COMPLETE                 true (all 749 candidates, no subset)
  HELDOUT_EXCLUSIONS               118 of 15729 declared examples (0.75%):
                                     STEM_MISSING_NOT_APPLICABLE 72,
                                     stem_missing_below_one_point 36,
                                     merged_below_one_point 9,
                                     claim_below_one_point 1
  CRITICAL_LEAKAGE                 none  (851 held-out sources checked, 0 overlap with the
                                         train/validation source unions)
  HELDOUT_PROTOCOL                 CLEAN
  MODEL_STABILITY                  PASS   (BAD_RECALL spread 0.0135)
  SAMPLING_ROBUSTNESS              PASS   (max BAD_RECALL drop 0.0 pp, 2048 x 3 seeds)
  ALL_MODEL_SEEDS_FROZEN_BEFORE_HELDOUT  YES
  DEPLOYMENT_CANDIDATE_SEED        101
  SAFE_BATCH_POINTNET/2/COMMON     8 / 8 / 8
  LOCAL_8_BAD_CHALLENGE_ACCESSED   NO
  ONNX_STARTED                     NO
  PRODUCT_INTEGRATION              NO

The pilot stopped at the verdict instead of continuing. It did not add epochs, enlarge a
model, re-pick a threshold, re-select a checkpoint, reopen training, change the generator,
or look at held-out again for any decision.


6. RESUME TEST
--------------

RESUME_TEST = PASS, proven three times on the live run and in unit tests:

  * an interrupted prewarm was recovered, set back to PENDING with the interrupt recorded,
    and re-entered, reusing every example it had already cached (9222 of 9240 were already
    generated when it restarted);
  * the first held-out attempt died on an untyped error; the next start found S15 RUNNING,
    recorded the interrupt, and re-entered it, preserving the original first-access
    timestamp of 04:43:31 and refusing to change any frozen value;
  * after the freeze, a restart skipped all fifteen completed stages as
    "COMPLETE (verified)", reused the calibrated batch without re-probing, and reused the
    three trained checkpoints without retraining them.

The final report's RESUME_TEST field reads NOT_RUN because the completed run's own state
never recorded the `--resume-test` output; the field is populated from the state file and
the state file is authoritative. The evidence above is from the run log and state, and
`python run_tree_qc_efg.py --resume-test` passes on demand. This is a reporting gap, not a
capability gap, and it is fixed by running the standard test before the report stage.


7. ENGINEERING FINDINGS
-----------------------

Seven defects were found by running the pipeline, and every one is now covered by a test.
Five are listed in PUBLIC_PILOT_STATUS_REPORT.md; the two found after it was written are:

  (f) The held-out record stored metrics but not the per-example scores, so the operating
      point could not be audited after the fact. The evaluator now writes
      <label>_scores.npz beside every held-out record, with its own SHA-256, and the
      analysis in section 3 was only possible because of it.

  (g) A restart left the previous attempt's stop code in the state, so CURRENT_STATUS.json
      reported a failure the new attempt was moving past. A stop is now moved to
      stop_history when a run resumes, and stale progress entries are cleared.

Three frozen-path findings (the loader cannot collate a batch, some declared specs are not
constructible, held-out identifiers are refused by the frozen geometry loader) are
documented in CURRENT_EFG_IMPLEMENTATION_AUDIT.md. All three are worked around outside the
accepted digest; repairing them in place would change the generator digest and void the
Gate D acceptance, which is a Supervisor decision, not this round's.

One reproducibility hazard was fixed: core.autocrlf silently rewrote LF to CRLF in three
frozen files, changing the digests without git status reporting anything. .gitattributes now
pins the frozen core to -text, the commits store the exact accepted bytes, and a clean
`git worktree add` of HEAD reproduces both digests with training_allowed true.


8. STAGE VERDICT
----------------

  PUBLIC_PILOT_NOT_LEARNABLE

Gate H (local 8 bad blind challenge) is not entered: it requires
READY_FOR_GATE_H_LOCAL_BLIND, which requires all four primary held-out targets. Two were
met (BAD_RECALL, BAD_FALSE_PASS) and two failed (GOOD_FALSE_REVIEW, BALANCED_ACCURACY).
No ONNX export, no product integration and no local resegmentation follow from this result.

The natural next question, for the Supervisor to decide rather than this round to assume,
is whether the good/defect balance of the frozen corpus - one GOOD example per six defects
per candidate and severity - is the right basis for an operating point a product would use,
or whether a QC model should be selected on balanced accuracy directly rather than on
maximum recall inside a false-review budget.
