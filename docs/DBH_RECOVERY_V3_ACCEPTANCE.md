# DBH Recovery V3 acceptance

Baseline: b849c3feb56c26ac1406989836e975e2035b48d9.
Project: E:/2026.8.31-2/2026.08.31-2/2026.08.31-2.
Evidence: acceptance_evidence/dbh_recovery_v3_20260831.

Result: **7 RECOVERED_STEM, 5 UNAVAILABLE, 0 estimates**. Existing six
numeric rows and their exact XYZ/HAG/original_index arrays are unchanged.
No source analysis, package rebuild, dependency change or push.

## New recovery: T-1-003

DBH: **16.561364126389037 cm**, RECOVERED_STEM / HIGH. HIGH describes
internal point-cloud geometry, not field-measured accuracy. The fitted evidence
contains 40,475 exact source rows, with RMSE **0.0036702446131132386 m**.
Four sections pass the unchanged V1 guards: centers at temporary axis-normal
heights 0.925, 1.075, 1.225 and 1.375 m, bracketing true breast height.
Their inlier fractions are 0.772705, 0.681396, 0.610107 and 0.616699. The two
higher sections remain rejected; no threshold was lowered to accept them.

Current upper geometry proposes an axis; accepted measured centers are followed
downward, then interpolated at true HAG 1.30. The resulting local XYZ anchor is
(-21.4585081973, -11.0099085741, 1.1519450707), not world-Z = 1.30. Axis tilt
is 2.187244 degrees; maximum center-track residual is 0.019596979 m, below the
existing 0.03 m bound. Original HAG/XYZ are retained; normal coordinates exist
only for fitting. The selected identity score is 0.813784. Its competitor scores
0.564316 and fails the absolute track gate. All twelve current crowns were
evaluated; no competing qualified crown or reserved stem claims this evidence.

The first bounded pass exposed an implementation defect: valid circles were
compared to a fixed extrapolated upper line, rather than followed downward.
The new synthetic regression failed under that behavior. Propagating each
measured center to the next layer fixed it, without changing the 0.03 m
prediction/residual bound, 20-degree tilt limit, score weights or DBH guards.
First-pass evidence is preserved; final_acceptance.json is authoritative.

## Competing axes and limits

Scores are deterministic evidence rankings, not probabilities. Selection requires
the structural gate, winner score >= 0.75, and margin >= 0.10 over a structurally
qualified runner, even if the runner's score is just below 0.75. A high score
cannot compensate for an incomplete crown-to-breast-height track.

| Tree | Candidate ID | Score | Absolute structural gate | Outcome |
|---|---|---:|---|---|
| T-1-006 | g0-r0-a0 | 0.679264 | FAIL | UNAVAILABLE |
| T-1-006 | g0-r0-a1 | 0.489691 | FAIL | UNAVAILABLE |
| T-1-006 | g0-r0-a2 | 0.384795 | FAIL | UNAVAILABLE |
| T-1-007 | g10-r0-a0 | 0.584839 | FAIL | UNAVAILABLE |
| T-1-007 | g10-r0-a1 | 0.471263 | FAIL | UNAVAILABLE |
| T-1-009 | g5-r0-a0 | 0.758881 | FAIL | UNAVAILABLE |
| T-1-009 | g5-r0-a1 | 0.504019 | FAIL | UNAVAILABLE |

T-1-006/007/009 have incomplete upper tracks under this conservative identity
policy. T-1-009's leading score exceeds the numeric floor, but upper sections
still fail the unchanged circle-quality guards, so it is not assigned. T-1-001
and T-1-011 produce no valid upper-axis proposal; their small current fragments
cannot inherit pre-edit stem ownership. T-1-003's two upper candidates and all
per-layer circles, distances, scores, twelve-crown comparisons and decisions
are also retained in final_recovery.json.

This bounded policy requires a continuous measured track from at least HAG
2.80 m through breast height to 1.05 m. It can reject branching, cluttered,
occluded or weakly sampled upper stems. UNAVAILABLE does not prove that the
raw cloud has no trunk or that another scientifically justified method could
never recover it. The preferred ten-tree target is not met. With seven reliable
observations, no allometric model is evaluated or used.

## Ownership, editor and persistence

Each candidate is compared to all current crowns. A symmetric conflict graph
is built before selection; reserved accepted stems cannot be reassigned.
Equivalent tracks from overlapping ROIs are deduplicated while every candidate
score and foreign claim remains auditable. No greedy fallback follows a
conflicted top candidate. Reversed claim/tree presentation produces the same
resolution. Final DBH indices are globally unique: duplicate ownership = 0.

Current DELETE, REMOVE_POINTS/lasso, MERGE and UNDO journal semantics reuse
V2's exact source-index exclusion. Deleted points are removed before tracking
and checked again before publication. Final selected points cannot already
belong to a neighboring current tree. Final memberships are never expanded.
The source, preparation, Ground, editor revision, final memberships and identity
bindings are checked before and after recovery. Invalid/stale sidecars fail closed.

The accepted fits used ten hash-validated V2 raw ROI captures, no new source
queries. No accepted six-tree fit was repeated. The HAG cache was reused only
after its accepted hash and current project bindings matched.

## Validation

Final related suite: **138 passed**, including 26 V3-focused cases and the
existing V1/V2 DBH, editor, tree geometry, clearance, UI label, API and export
tests. A final six-test writer check also passes, adding one old-estimate
compatibility case (27 V3 cases covered in total). Existing estimates remain
unchanged and do not reserve a fabricated physical stem or count as reliable
point-cloud training observations. Ruff and in-memory compilation pass.
Exact-project checks preserve:

- Twelve trees and 4,765,121 exact final points.
- Persistent/global/display identities and all non-DBH Excel fields.
- Height, crown dimensions/area, clearance and risk.
- All fifteen original critical hashes and exact Wire artifact proof.
- Every previous project file except the authorized DBH pointer; old exports
  and prior DBH generations remain preserved.

restart.json records actual fresh DesktopWorkflow/editor hydration and current
report publication. Every non-DBH API row, W1, KPI, display field and gate matches
V2 exactly. Only logging was redirected to audit evidence. It also validates
the new Survey.DBH_RECOVERY_V3.xlsx. Prediction_Input.DBH_RECOVERY_V3_FINAL.xlsx
is the accepted prediction-input export. This is not a GUI/HTTP or packaged-PC test.

Explicit Finalize and the CLI default use V3; --version v1 and --version v2
remain available. Read-only reports never trigger recovery. A missing/stale
finalize base is reconstructed through the existing V1/V2 path before V3.
DBH failure never rolls back successful Survey finalization or clearance.

The unrelated existing core/stage_b/batch.py diff is preserved and excluded
from the V3 commit. No frozen-science implementation was changed.
