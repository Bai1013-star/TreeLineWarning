CURRENT_EFG_IMPLEMENTATION_AUDIT
=================================

Round: TREE_QC_EFG_ONE_CLICK_PUBLIC_PILOT_V1
Base:  1c4ed5b96e4ef1053bf9d0e47d597911ffbca727
Branch: research/tree-qc-efg-public-pilot-v1

Gate D binding re-verified before the acceptance marker was written:

  GENERATOR_VERSION  GATE_D_SYNTHETIC_GENERATOR_V2            (runtime match)
  GENERATOR_DIGEST   a0550c17ff75cdf7d380fc677840c883f2262f2148a40edbbee0df35410b6b9d
  PROTOCOL_DIGEST    36e4cfca043190d4d12c2ab9ee635d92bb479cb9b56d482ae1121fd797939729
  gate_d_lock.acceptance_state().training_allowed == true
  train_pointnet.py --dry-run / train_pointnet2.py --dry-run -> no GATE_D_NOT_ACCEPTED


1. CAPABILITIES THAT ALREADY EXISTED AND WERE REUSED UNCHANGED
--------------------------------------------------------------

Frozen Gate D core (NOT modified; verified by per-file SHA-256 at every run):

  gate_d_provenance.py   donor allowlist, three quarantine concepts, fail-closed
                         authorization, frozen Gate C2 artifacts loader
  gate_d_scene.py        BaseScene, frozen 3 m query box, clipping, minimum-point rules
  gate_d_geometry.py     bounded per-tree cache, stride cap, publisher membership authority
  gate_d_generator.py    the seven corruption classes, ExampleSpec, enumerate_specs,
                         NotApplicableError semantics, generator version/digest source
  gate_d_sampler.py      structured sampler, class-independent sampling_seed, four features,
                         metric scale preservation
  gate_d_dataset.py      GateDDataset, DatasetConfig partition refusal of held-out,
                         restart_state and dataset digest
  gate_d_metrics.py      confusion, five metrics, select_threshold (refuses held-out),
                         freeze_threshold, three_way_decisions
  gate_d_train.py        TrainConfig, detect_device, set_determinism, _run_epoch,
                         _checkpoint_payload, save/load_checkpoint, build_model
  gate_d_fingerprint.py  generator fingerprint diagnostics (unchanged)
  gate_d_diagnostic.py   generator_digest() (the acceptance binding source)
  gate_d_lock.py         fail-closed acceptance lock
  models.py              PointNet and PointNet++ SSG
  train_pointnet.py / train_pointnet2.py   formal entry points with --dry-run

Directly reusable without adaptation: the whole generator/sampler/dataset/metric core,
the two networks, the frozen protocol document, the Gate C2 provenance artifacts, the
already-measured Gate A safe-batch policy and the run layout conventions.


2. MISSING CAPABILITY THAT THIS ROUND ADDED
-------------------------------------------

There was no executable E/F/G pipeline, no persistence, no held-out route and no
one-click entry point. run_tree_qc_efg.py was a plan printer that refused to run. Added
(thirteen new modules plus a rewritten orchestrator, all inside research/tree_qc):

  efg_gates.py       single declaration of every gate, budget, seed and verdict name
  efg_state.py       atomic state store, self-digested state files, pid run lock,
                     stage completeness bound to evidence digests
  efg_resources.py   environment fingerprint, 35 GiB disk guard, batch calibration
                     16/8/4/2/1 with a common safe batch for both architectures
  efg_specs.py       partition-agnostic spec arithmetic plus NOT_APPLICABLE handling
  efg_cache.py       per-example byte-verified feature cache
  efg_dataset.py     model-selection source with prewarm, cache and typed exclusions
  efg_heldout.py     held-out view, held-out authorization, held-out geometry loader,
                     the ported corruption body, the parity gate and HeldOutDataset
  efg_freeze.py      architecture selection, deployment-seed selection,
                     PUBLIC_EXPERIMENT_FREEZE, immutability and all-frozen checks
  efg_evaluate.py    scoring, per-defect/per-severity breakdown, three-way reporting,
                     multi-seed aggregation, robustness summary
  efg_leakage.py     held-out protocol audit and the local-challenge boundary scan
  efg_trainer.py     guarded training loop (non-finite, OOM, resume, exclusions)
  efg_reports.py     TREE_QC_EFG_FINAL_REPORT (frozen field list), stop reports,
                     reserved product interface
  run_tree_qc_efg.py 19-stage resumable state machine, --auto/--dry-run/--status/
                     --resume-test/--smoke-test
  RUN_TREE_QC_PUBLIC_PILOT.cmd   locator-only one-click launcher (no business logic)
  TREE_QC_PUBLIC_PILOT_PROTOCOL_V1.md   the protocol of record
  test_efg_orchestrator.py  137 tests covering cases A-Q of the round specification

Verified numbers: 352 tests pass (Gate D 163, Gate C2 30, EFG 137, plus the model,
public-data and archive suites) with 42 subtests; ruff clean on every file this round
touched and on the frozen Gate D core.


3. FILES THAT DELIBERATELY WERE NOT MODIFIED
--------------------------------------------

Every file that participates in the generator digest, listed in section 1. Confirmed by
`git status` on the research branch (no changes) and by re-computing the runtime digests,
which still equal the accepted values. Also untouched: the frozen V0.1 product, the
product UI, PyInstaller packaging, the CROWN_STEM_HYBRID algorithm chain, the local
Gate H challenge, and research launchers other than the new cmd file.


4. THREE FROZEN-PATH FINDINGS AND HOW THEY WERE HANDLED
-------------------------------------------------------

These were discovered by running the pipeline, not by reading it. All three are worked
around in the EFG layer because repairing a frozen file would change the accepted
generator digest and invalidate the Supervisor's acceptance. Each is reported, not
silently absorbed.

(a) gate_d_dataset.make_loader cannot complete a batch.
    collate expects (features, label, index) triples; TorchGateDDataset.__getitem__ yields
    (features, label) pairs. The frozen epoch loop therefore could not run its first batch
    — it had never been executed. EFG supplies its own loader over the EFG source wrapper
    with a local collate that mirrors the frozen one and converts the index with the same
    torch.stack semantics. Model, loss, optimizer, sampler and dataset are the frozen ones.

(b) Some declared specs cannot be constructed from the real geometry.
    gate_d_generator raises NotApplicableError (for example merged_below_one_point with a
    donor pool of 3 points) and the frozen GateDDataset propagates it. Over 12 915 declared
    examples this is fatal to any full run. EFG records the exclusion with its reason,
    removes those examples from the training batches and from the metric population, reports
    the count and the per-class breakdown, and stops if more than 5% of a partition is
    excluded. No example is substituted and no label is invented.

(c) Held-out geometry is unreachable through the frozen geometry loader.
    gate_d_geometry.load_cached_geometry authorises via partition_of_candidate, which
    refuses a held-out tree by design. EFG therefore has its own held-out geometry loader
    that reuses the frozen reader, stride cap and cache writer and replaces only the
    authorisation step. This is the one behavioural difference between the frozen route and
    the held-out route, and it is what the adapter parity gate exists to prove harmless:
    the ported corruption body reproduces the frozen pipeline on real validation candidates
    with identical spec key, corruption seed, sampling seed, label, class, severity,
    mechanism, changed-membership count and fraction, sampler pre-reservation and final
    index digests, and bit-identical feature tensors (63 examples at 4096 points and 105 at
    2048 points, zero mismatches).

If the Supervisor judges that (a) or (b) must be repaired inside the frozen files rather
than worked around outside them, that is a new Gate D signature and needs a new
acceptance; this round did not take that step on its own authority.
