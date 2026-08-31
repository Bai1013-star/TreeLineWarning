# DBH Recovery V2 acceptance

Baseline: `05336844c8bd2b7003fdadeda06f78b1545f818a`.
Project: `E:/2026.8.31-2/2026.08.31-2/2026.08.31-2`.
Evidence directory: `acceptance_evidence/dbh_recovery_v2_20260831`.

V2 recovers **6 of 12** finalized trees, up from V1's five. T-1-005 is
newly recovered at **16.890963166606355 cm**. The five accepted V1 rows
and their XYZ/HAG/original_index evidence remain exact. Six trees remain
UNAVAILABLE. No allometric model is used: six reliable observations are below
the unchanged ten-tree minimum. HIGH describes internal geometric quality,
not field measurement accuracy. These results do not establish a release or
independent-PC acceptance; no package was rebuilt.

## Bounded investigation

Nineteen 0.60 m local ROIs were captured from the existing exact-index raw
role cache. Fitting replayed those hash-validated arrays without another source
query. Neither the source LAS nor the 105M-point pipeline was rerun. Current
member HAG was computed with the existing canonical helper and cached separately;
Ground and HAG science were not changed.

T-1-005's strongest current low-member mode led to a stem near
XY (-29.866699, -16.415919), about a metre from the V1 location. Its winning
ROI contains 272,511 raw points. Across its three overlapping ROIs, there are
516,753 distinct raw source points and 81,004 retained raw breast-height points.
The winning ROI has 190,664 points in the low-stem HAG range. Axis-normal
continuity retains 135 supported XY columns and 36,804 fitting points. Five
sections pass the unchanged V1 guards, with overall RMSE 0.003482769 m.
The accepted sections bracket 1.30 m; the rejected section remains recorded.
The raw bridge has 10,561 / 11,871 / 9,105 points in its three height bins.
Current-member connection spans five consecutive half-metre bins. No neighboring
tree passes that attachment test, and selected source indices do not belong to
another current tree or a deleted point. This supports classification A: a valid
current-associated stem was recovered, rather than a fabricated DBH from clutter.

T-1-003 has 81,732 raw breast-height points across the new local ROIs, but only
two preliminary sections support an axis (near HAG 0.923 and 1.074 m). Above
breast height, the proposed circles have approximately 69 / 79 / 89 degrees
of angular coverage, below the unchanged 180-degree guard. The two additional
current-mode ROIs have no breast-height rows. V2 cannot establish a bracketed
axis and does not extrapolate from the lower sections. This is a bounded-method
failure, not proof that the source contains no trunk.

T-1-006, T-1-007 and T-1-009 retain respectively three, two and two distinct
plausible axes. Their scores and all primary/secondary proposals are recorded;
ambiguity is not resolved by choosing a convenient diameter. T-1-001 has only
a thin surviving fragment at HAG 2.263–2.448 m and no valid axis. T-1-011 has
274 surviving points and no valid axis. Historical labels are not reinstated.

## Per-tree results

Point count means retained fitting evidence. Raw and BH counts are unique source
rows per tree, with BH counts after editor exclusion. V1 counts are preserved
from V1; `—` in the candidate column means V2 did not search that accepted tree.
RMSE is in metres. Abbreviations: R = RECOVERED_STEM; U = UNAVAILABLE;
N = NO_VALID_AXIS_PROPOSAL; A = AMBIGUOUS_COMPETING_AXES.

| Tree | DBH cm | Method | Quality | Points | RMSE m | Raw ROI | Raw BH band | Candidates | Excluded | Failure |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---|
| T-1-001 | — | U | U | 0 | — | 194698 | 58415 | 0 | 36523 | N |
| T-1-002 | 16.021230 | R | HIGH | 39356 | .003911 | 317323 | 72865 | — | 0 | — |
| T-1-003 | — | U | U | 0 | — | 364141 | 81732 | 0 | 0 | N |
| T-1-004 | 17.587960 | R | HIGH | 31595 | .003895 | 266745 | 64678 | — | 0 | — |
| T-1-005 | 16.890963 | R | HIGH | 36804 | .003483 | 516753 | 81004 | 1 | 0 | — |
| T-1-006 | — | U | U | 0 | — | 451245 | 105024 | 3 | 0 | A |
| T-1-007 | — | U | U | 0 | — | 479356 | 99706 | 2 | 0 | A |
| T-1-008 | 18.061801 | R | HIGH | 37196 | .003552 | 270311 | 58021 | — | 0 | — |
| T-1-009 | — | U | U | 0 | — | 599152 | 70200 | 2 | 0 | A |
| T-1-010 | 16.818740 | R | HIGH | 27596 | .003767 | 283350 | 54101 | — | 0 | — |
| T-1-011 | — | U | U | 0 | — | 16901 | 3152 | 0 | 614 | N |
| T-1-012 | 16.110308 | R | HIGH | 38547 | .003778 | 297020 | 67582 | — | 0 | — |

## Non-regression and publication

`baseline.json`, `acceptance.json` and `final_acceptance.json` bind the exact
12 identities, 4,765,121 final points, all 15 baseline critical hashes, the
existing Wire proof, and every preexisting project file's size/mtime apart from
the authorized DBH pointer. Height, crown dimensions/area, clearance, risk,
identity/display fields and every non-DBH Excel field are unchanged. The final
diagnostic refresh counts distinct axes rather than repeated pair proposals;
it reuses all accepted fits/arrays and does not refit geometry.

`restart.json` records the fresh-process DesktopWorkflow/current-report and
Survey export check. The audit redirects logging only, to preserve the original
project log; it is not a GUI or HTTP acceptance. Existing UI rendering tests
exercise the recovered/estimated/unavailable labels.

Focused and related tests cover V1/V2 geometry, current ownership and neighbor
ambiguity, exact lasso and DELETE-after-MERGE exclusion, UNDO journal state,
stale evidence, source proof tampering, deterministic fitting, finalize failure
isolation, restart, API, UI labels, Excel, clearance and tree editing. Recorded
related regression: 111 passed; an additional finalize-entry test passed.
Ruff and compilation results are recorded in `validation.json`.

The CLI defaults to `--version v2` for an existing valid finalized DBH base;
`--version v1` is retained for older projects. Normal explicit Finalize creates
a fresh V1 base when needed before current raw recovery. Reports only read
sidecars and never initiate recovery. Existing invalid/stale proof remains
unavailable rather than being displayed as a valid diameter.

No Ground, HAG, Wire, CROWN_STEM_HYBRID, final crown membership, clearance,
risk policy or CONTACT implementation changed. The unrelated existing
`core/stage_b/batch.py` diff is preserved and excluded from the V2 commit.
No dependency, GPL source, package rebuild or push was introduced.
