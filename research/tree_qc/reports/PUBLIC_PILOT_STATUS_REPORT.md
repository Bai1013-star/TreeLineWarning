PUBLIC_PILOT_STATUS_REPORT
==========================

Round:  TREE_QC_EFG_ONE_CLICK_PUBLIC_PILOT_V1
Base:   1c4ed5b96e4ef1053bf9d0e47d597911ffbca727
Branch: research/tree-qc-efg-public-pilot-v1
Head:   1afb2b9f6f1d40632538e7d238442fb4bafccca3

This report is written while the public pilot is still running. No training, held-out
or robustness result below is estimated, interpolated or invented: every number is read
from an artifact the run itself wrote.


1. VERDICTS THAT ARE ALREADY EARNED
-----------------------------------

AUTOPILOT_IMPLEMENTATION = PASS

  Gate D core untouched. All ten files that participate in the accepted generator digest,
  plus gate_d_lock.py and the frozen protocol document, are byte-identical to the accepted
  baseline; `git log 1c4ed5b..HEAD` over those paths is empty. The runtime digests still
  equal the accepted values, and the acceptance marker is valid:

    generator a0550c17ff75cdf7d380fc677840c883f2262f2148a40edbbee0df35410b6b9d  MATCH
    protocol  36e4cfca043190d4d12c2ab9ee635d92bb479cb9b56d482ae1121fd797939729  MATCH
    training_allowed = true

  Tests: 163 Gate D + 30 Gate C2 + 141 EFG = 334 in the suites this round touches, all
  passing (plus the model, public-data and archive suites, 354 total, 42 subtests).
  Ruff clean on every file this round wrote and on the frozen core.

RESUME_TEST = PASS

  Exercised three times on the live run, not only in unit tests:

    * an interrupted prewarm was recovered and re-entered, reusing everything it had
      already durably produced;
    * a SIGTERM'd run left S15 RUNNING; the next start reset it to PENDING with the
      interrupt recorded and re-entered it;
    * a restart after the freeze skipped all fifteen completed stages as
      "COMPLETE (verified)", reused the calibrated batch without re-probing, and reused
      the three trained checkpoints without retraining them.

HELDOUT_ADAPTER_PARITY = PASS

  63 examples at 4096 points and 105 at 2048 points on real validation candidates, all
  7 classes x 3 severities, zero mismatches, on identical spec key, corruption seed,
  sampling seed, label, class, severity, mechanism, changed-membership count and
  fraction, pre-reservation and final sampler index digests, and bit-identical feature
  tensors. Held-out geometry was not read during the parity check.

DRY RUN = PASS   Points at the frame: 440 train candidates / 9240 examples, 175 validation
candidates / 3675 examples, 749 held-out candidates (metadata only), GPU RTX 5070 Laptop
8 GiB, RAM 15.7 GiB, D free 102.6 GiB, batch probe order 16/8/4/2/1, held-out locked YES,
local challenge locked YES.

BOUNDED SMOKE TEST = PASS   All nineteen stages ran end to end on a bounded rehearsal and
the run correctly refused to present itself as a scientific result
(STOP_PARTIAL_HELDOUT_EVALUATION).


2. FROZEN ARTIFACTS PRODUCED BY THE LIVE RUN
--------------------------------------------

SAFE_BATCH_POINTNET = 8      (measured, max stable 16, safe = half)
SAFE_BATCH_POINTNET2 = 8
COMMON_TRAIN_BATCH = 8

Dataset fingerprint: train digest 5ed443082427f18a, validation digest 7584b441d30b8e39,
provenance and identity-manifest digests from the frozen Gate C2 artifacts.

Training-data prewarm (one generation pass; every later epoch is a verified cache read):
  9240 declared train examples -> 9222 generated, 18 not applicable
  3675 declared validation examples -> 3672 generated, 3 not applicable
  Exclusion rates 0.19% and 0.08%, well inside the declared 5% limit.

Model comparison (Phase E1), identical train candidates, validation candidates, examples,
sampler, points=4096, corruption, budget, effective batch, threshold rule and early-stopping
rule; the only difference is the network:

  POINTNET_SEED101     validation BALANCED_ACCURACY 0.8551   BAD_RECALL 0.9596
                       best epoch 29, frozen validation threshold 0.4750
  POINTNET2_SEED101    validation BALANCED_ACCURACY 0.8951   BAD_RECALL 0.9978
                       best epoch 21, frozen validation threshold 0.0050

  Both cleared the learnability floors (BA >= 0.70 and BAD_RECALL >= 0.60), so the
  pilot continued instead of stopping and did not open held-out early.

SELECTED_ARCHITECTURE = pointnet2_ssg
ARCHITECTURE_SELECTION_BASIS = validation only, first criterion GOOD_FALSE_REVIEW <= 0.25
  satisfied by both, then higher BAD_RECALL (0.9978 vs 0.9596). The pre-declared 0.01
  non-inferiority preference for pointnet did not apply: the recall gap is 0.038.

Three frozen seeds (S10), each with its own validation-only threshold under the frozen
rule max BAD_RECALL subject to GOOD_FALSE_REVIEW <= 0.25:

  seed 101   threshold 0.0050   best epoch 21
  seed 202   threshold 0.0150   best epoch 17
  seed 303   threshold 0.1000   best epoch 7
  seed 101 was reused from the comparison run rather than retrained (digest-verified).

DEPLOYMENT_CANDIDATE_SEED = 101   (validation only: budget satisfied, then max BAD_RECALL,
  then max BALANCED_ACCURACY, then lowest seed; chosen before unblinding)

PUBLIC_EXPERIMENT_FREEZE_SHA256 =
  717fe17036f90eaf3fb2d0ea7aecbc4929dbf1d8c3a3c8b6c6b9277d2ad3e37a

HELDOUT_ACCESSED_BEFORE_FREEZE = NO
HELDOUT_ACCESSED = YES    first access 2026-09-12T04:43:31+08:00, freeze digest recorded
                          at unblinding and unchanged since
HELDOUT_CANDIDATES = 749, manifest count verified at unblinding (expected 749, actual 749)

LOCAL_8_BAD_CHALLENGE_ACCESSED = NO     ONNX_STARTED = NO     PRODUCT_INTEGRATION = NO


3. WHAT IS STILL RUNNING
------------------------

PUBLIC_PILOT_STATUS = RUNNING
CURRENT_STAGE = S15_HELDOUT_BLIND_EVALUATION
RESUME_READY = YES

Remaining: three per-seed held-out passes at the frozen thresholds, multi-seed aggregation,
the 2048 x three pre-declared sampling seeds robustness sweep, and the final report. The
held-out pass is CPU-bound: a real held-out scene holds up to 5.9 M points across 70
context trees, and the frozen structured sampler does O(N) work per example. Measured
throughput 3.84 examples/s over 15 729 examples, so roughly one hour per pass.

D free 81.14 GiB against a 35 GiB guard; the held-out geometry cache is 16.3 GiB
(749 candidates, built once); run root total 17.3 GiB.


4. FIVE REAL DEFECTS THE LIVE RUN EXPOSED, AND THEIR FIXES
----------------------------------------------------------

Every one of these was found by running the pipeline, and each is now covered by a test.
None of them changed an experimental decision.

(a) gate_d_dataset.make_loader cannot complete a batch. collate expects
    (features, label, index) triples while TorchGateDDataset.__getitem__ yields
    (features, label) pairs, so the frozen gate_d_train epoch loop could not run at all.
    Repaired outside the digest: the EFG loader builds over the EFG source wrapper with a
    collate that mirrors the frozen one and converts the index with the same torch.stack
    semantics. Model, loss, optimizer, sampler and dataset stay frozen.

(b) Some declared specs are not constructible from the real geometry
    (gate_d_generator.NotApplicableError, e.g. merged_below_one_point with a donor pool of
    3 points). The frozen dataset propagates the refusal, which is correct, but it made a
    full partition impossible to iterate. They are now counted exclusions with their
    reason, removed from the training batches and from the metric population, bounded at
    5% per partition, and never substituted.

(c) The held-out route propagated the same refusal as an untyped error, which crashed the
    first held-out attempt. HeldOutDataset now raises the same typed exclusion as the
    model-selection side and persists its exclusions so a restart does not rediscover them.

(d) An unexpected exception during a stage produced a bare traceback and exited 1, leaving
    the run with no stop record. Every stage now converts any exception into a recorded
    STOP_UNEXPECTED_ERROR with its traceback, a stop report and a clean state.

(e) A restart after the freeze failed its own precondition: stage_heldout restored the runs
    from state but never re-marked their thresholds frozen. Fixed by load_frozen_runs,
    which rebuilds the complete precondition from the persisted artifacts.

Two of the five (.a and .b) are defects in frozen files. Both are worked around outside the
accepted digest rather than repaired inside it, because repairing them would change the
generator digest and invalidate the Supervisor's acceptance. If the Supervisor prefers them
repaired in place, that is a new Gate D signature and needs a new acceptance.


5. A REPRODUCIBILITY HAZARD THAT WAS FIXED
------------------------------------------

The accepted digests are computed over the bytes on disk, and Git's `core.autocrlf=true`
silently rewrites LF to CRLF on checkout. `git status` reports nothing because the content
is "the same" after normalisation, but the digest changes and the training lock closes.
This actually happened twice during this round:

  * research/tree_qc/gate_d_sampler.py   — blob CRLF, working tree LF
  * research/tree_qc/gate_d_fingerprint.py — blob CRLF, working tree CRLF (4 bytes of CR)
  * research/tree_qc/reports/GATE_C_CANONICAL_IDENTITY_PROTOCOL_V2.md — blob CRLF, tree LF

`.gitattributes` now marks the whole frozen core and the protocol document `-text`, the
commits store the exact accepted bytes, and a clean `git worktree add` of the current HEAD
reproduces both accepted digests and `training_allowed = true`. Verified, not assumed.


6. BOUNDARIES HELD
------------------

No change to the frozen V0.1 product, the product UI, PyInstaller packaging, the
CROWN_STEM_HYBRID algorithm chain, or any other research launcher. No TreeRepairEngine, no
local resegmentation, no ONNX export, no product integration. The local Gate H challenge was
never read; the boundary is enforced in code (efg_leakage.assert_local_challenge_untouched)
and runs during the final-report stage. The output-interface concept is reserved in
efg_reports.interface_contract() with AUTO_RESEGMENT_MAX_ATTEMPTS = 1, and nothing is
implemented.
