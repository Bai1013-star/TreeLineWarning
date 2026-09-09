# TREE_QC_GATE_C2_WORKER_REPORT — canonical tree identity

Worker: Claude Code (DeepSeek). This is a worker report, not a Gate decision.
GPT-6 Supervisor independently reviews and accepts or rejects.

Baseline `START_HEAD = 9b1279d7672dd51659da5ab69525118900deda65`, branch
`codex/tree-qc-pointnet2-final-pilot-v1`, worktree
`E:\TreeLineWarning-tree-qc-pointnet2-final-pilot-v1`. Working tree clean at start.

## 1. What the task was

Gate C was on `HOLD_DATA_IDENTITY_CONTEXT_UNRESOLVED`. The old protocol
`GATE_C_SPATIAL_PROTOCOL_V1` treated publisher `instance` IDs and the tree↔plot
correspondence as unproven, so under its quarantine cascade only 3 of 876 Wytham
source groups survived as `preliminary_clear`. This turn builds a canonical identity
layer, re-derives the pools on that layer, and preserves every V1 scientific standard.

V1 and its reports are **unchanged**. V1 is not described as wrong: it was written
under the conservative assumption that cross-folder IDs and frames might not be
comparable. The archives show the official semantics are stronger than that
assumption, so `GATE_C_CANONICAL_IDENTITY_PROTOCOL_V2` is proposed as a separate,
prospective protocol.

## 2. Official identity semantics

### DOCUMENTED (publisher, Zenodo record 16875688 description)

- Each plot is split into separate labelled test/validation/training area clouds.
- Scalar fields are `semantic` (0 ground, 1 tree) and `instance` (sequential from 1,
  `-1` for ground); the area clouds may be merged into one labelled plot cloud.
- Individual tree clouds are split across test/validation/training folders, and
  "all trees overlapping with each area are in the corresponding folders, so some
  edge trees are duplicated in multiple folders".

### OBSERVED (this audit, over the verified extraction)

- 1,693 tree file occurrences; 1,579 distinct `(site, basename)` pairs.
- 113 groups of occurrences share an identical float32 coordinate multiset; all 113
  cross publisher folders (112 pairs, 1 triple) and explain exactly the 114 extra
  occurrences (1,693 − 114 = 1,579).
- Tree files in `train/`/`val/` carry native `instance`/`semantic`; tree files under
  `test/` carry XYZ only.
- The `instance` namespace is **site-global, not renumbered per area file**. The
  merged Wytham namespace spans 1–876 with 783 IDs actually present in the area
  clouds; Litchfield's merged namespace contains exactly 703 IDs (2–704), equal to
  its 703 canonical trees.
- Every tree file's points map onto one dominant plot instance ID. Of the 1,215
  canonical trees that carry a single native instance ID, the 1,154 with plot
  support all have native ID == dominant plot instance (**zero disagreements**); the
  other 61 have no plot support at all.
- 1,486 of 1,579 canonical trees have exact plot support; the 93 without are all
  Wytham out-of-plot trees and correspond exactly to the 93 IDs absent from Wytham's
  merged instance namespace.
- Exact coordinate cells carrying two instance IDs exist inside a plot (Wytham test
  2.76 %, train 0.98 %, val 0.52 % of point rows). These are duplicate raw scan
  points assigned to two neighbouring trees during manual extraction — shared
  *points*, not shared *trees*. In the membership audit 1,226 of 2,276 matching
  (tree, plot) records have 100 % of their matched points on the dominant instance
  and only 219 fall below 0.99; the lowest shares belong to the high-overlap pairs
  of §6.
- No `(site, basename)` maps to two coordinate groups, and no two distinct basenames
  share a coordinate multiset: **zero basename identity conflicts**.
- Independent coincidence worth recording: the canonical counts (703 Litchfield /
  876 Wytham) and the `test/in_plot_th0.90` counts (128 / 181) equal the per-site
  tree and evaluation-tree counts recorded from the associated publication in
  `DATASET_AUDIT.md`. That is a coincidence of counts, not a source confirmation.

### INFERRED

- A tree PLY is the complete extracted tree, not a per-area crop; the area cloud is
  the cloud clipped to that area.
- Metric metres and a shared local frame per site (pinned evaluator: 1 cm overlap
  threshold, 2-decimal rounding of unscaled XYZ).

### UNRESOLVED

- Absolute georeferencing (EPSG/global CRS) — not used anywhere.
- The physical meaning of a bounded set of high partial-overlap pairs (§6).
- Publisher intent for the two Wytham naming series.

## 3. Canonical identity layer

`canonical_tree_id = "<site>:<basename>"`. A canonical tree is the set of file
occurrences that are exact aliases of one another.

| Level | Requirement | Action |
|---|---|---|
| A `LEVEL_A_EXACT_ALIAS` | identical float32 coordinate multiset (equal point count + equal order-independent fingerprint) and agreeing basename | union |
| B `LEVEL_B_STRONG_ALIAS_CANDIDATE` | containment ≥ 0.95, centroid gap ≤ 0.25 m, overlapping bbox, not exact | review only, never auto-union |
| C `LEVEL_C_PARTIAL_OVERLAP_ONLY` | any shared XYZ cell or bbox intersection short of A/B | keep independent |

Result: 1,579 canonical trees, 113 level-A groups, **0 level-B**, **0 unresolved**,
0 basename conflicts. Every canonical tree carries `identity_status = EXACT`.

## 4. Plot membership

Exact float32 cell equality between each tree file and each labelled area cloud,
streamed in bounded chunks. Per `(tree, plot)`: matched cells, matched rows, match
fraction, dominant instance, dominant share, full instance histogram. No
nearest-neighbour matching, no ID-based inference, no union on shared cells.

## 5. Leakage-safe pools

- A canonical tree belongs to exactly one partition; alias occurrences never count
  twice and never span partitions.
- Litchfield keeps the V1 fixed geometric split: 70th-percentile index of canonical
  centroids, computed before eligibility filtering. Reproduced boundary
  `x = 67.92401220579217 m`, index 491 — identical to V1's documented value.
- Wytham is the held-out site in full.

**CONTEXT_METHOD.** Provenance-bound neighbour index over canonical trees. For a
candidate the query is the candidate XY bbox expanded 3 m with the candidate's native
Z extent. Neighbours are distinct canonical trees whose bbox intersects the query;
context membership is the actual neighbouring publisher points inside the query,
counted from the neighbour's tree file. No plot background, no invented geometry, no
random donor tree.

**DUPLICATE_OBSERVATION_POLICY.** Exact-alias occurrences of one canonical tree are
collapsed to a single entry before any pool, neighbour or support count, so an alias
can never be its own neighbour and never becomes an independent sample. Level-C
partial-overlap pairs stay independent because the publisher labels them as distinct
instances; the high-containment subset is listed for review and an alternate held-out
pool excluding every affected tree is reported alongside the primary pool.

## 6. Residual ambiguity (disclosed, not hidden)

17 partial-overlap pairs reach containment ≥ 0.30 (max 0.810881); the remaining 696
pairs are below 0.30 and 676 below 0.10. The high-containment pairs are **not** exact
aliases, have centroid gaps of 0.88–5.96 m, and are labelled as distinct instances by
the publisher. V2 therefore keeps them independent, and reports an alternate held-out
pool that excludes every affected tree so the Supervisor can choose either reading.
The two naming series in Wytham (`wytham_winter_<3-digit>` and `<4-digit>`) are the
main structural pattern behind those pairs.

## 7. Statistics

| Field | Value |
|---|---|
| SOURCE_FILE_OCCURRENCES | 1693 |
| OLD_SOURCE_GROUPS (V1) | 1579 |
| CANONICAL_TREE_COUNT | 1579 (703 Litchfield / 876 Wytham) |
| EXACT_ALIAS_GROUPS | 113 (114 extra occurrences) |
| STRONG_VERIFIED_ALIAS_GROUPS | 0 |
| UNRESOLVED_GROUPS | 0 |
| BOUNDARY_DUPLICATE_GROUPS | 113 |
| PARTIAL_OVERLAP_NONALIASES | 713 |
| TRAIN_UNIQUE | 442 |
| VALIDATION_UNIQUE | 177 |
| LITCHFIELD_QUARANTINE | 84 |
| HELDOUT_UNIQUE | 876 |
| TRAIN_VAL_CANONICAL_LEAKAGE | 0 |
| CROSS_SITE_LEAKAGE | 0 |
| OLD_PRELIMINARY_CLEAR | 3 |
| NEW_DEFENSIBLE_HELDOUT_POOL | **772** (863 without the plot-support requirement; 749 / 838 excluding high-containment-affected trees) |
| MINIMUM_SUPPORT_30_PRESERVED | YES |

Held-out blockers: 93 trees have no plot support (all Wytham out-of-plot) and 13 have
fewer than 4,096 points; every one of the 876 Wytham canonical trees has at least one
real neighbouring tree with actual points inside its 3 m query, so no tree is excluded
for missing context. Instance-ID collisions: **0**. Native instance ID equals the
dominant plot instance for all 1,154 trees that have both (61 trees with a native ID
have no plot support; 364 trees have no native ID at all).

**Remediation (see `GATE_C2_REMEDIATION_REPORT.md`).** After Supervisor review this
table's pool figures are superseded by the provenance-clean remediation: train **440**
eligible (2 excluded for base eligibility), validation **175** (2 excluded), strict
held-out pool **749** re-derived with the 25 high-containment trees removed from
candidate, context, donor and alias roles. Five canonical trees that had been context
on both model-selection sides were removed from both; all assertions are zero.

## 8. Representative case review

`GATE_C2_CASE_REVIEW.json` records four families with coordinate statistics, bbox,
centroid, point count, unique cells, duplicate-cell counts, match fractions,
dominant instance, neighbour support and full provenance:

- **A — clear exact alias**: a boundary-duplicated tree whose two occurrences hold an
  identical coordinate multiset and map to the same plot instance.
- **B — partial overlap, different trees**: two distinct basenames sharing exact cells
  well below the alias threshold; kept independent.
- **C — train/validation boundary duplicate**: an occurrence pair spanning native
  folders; assigned to exactly one partition.
- **D — complex unresolved case**: the highest-containment non-alias pairs plus the
  weakest plot-support trees.

## 9. Independent verification

`GATE_C2_VERIFICATION.json` — all checks pass:

| Check | Result |
|---|---|
| sampled fingerprint recompute (41 trees re-read from disk) | PASS, 0 mismatches |
| every multi-occurrence group is level A | PASS |
| canonical ids unique | PASS, 0 duplicates |
| every source occurrence accounted for | PASS, 1,693 distinct paths |
| no canonical leakage (train/val, train/held-out, val/held-out, cross-site) | PASS, all 0 |
| minimum support preserved | PASS, minimum 30, eligible 772 |
| no instance-id collisions | PASS, 0 |
| digest matches summary | PASS |

## 10. Determinism

Two independent full identity runs over the verified archives produced the identical
identity-manifest digest `eb515773b5235f2cdb0206231493195f73f269c761d9da2d2186ba68ab3ed5aa`.
The digest covers identity fields only, so later membership/context enrichment cannot
change it. Focused tests additionally cover rerun determinism, digest stability under
row reordering and order-independent pair keys.

## 11. Tests and lint

`python -m pytest test_canonical_identity.py` — **19 passed**. Coverage: exact
duplicate aliases; different instance IDs on one canonical tree; one instance ID
naming two trees; partial overlap is not an alias; containment above threshold stays
a review candidate and does not union; boundary duplicate gets exactly one partition;
held-out and unresolved trees never enter train; pool disjointness; alias occurrence
is never its own neighbour; real neighbours retained; distant trees excluded; rerun
determinism; digest stability under row reordering; order-independent pair keys;
signed-zero cell equality; PLY list-property rejection; exact-cell-only membership.

`python -m ruff check` on the three new files — **all checks passed**. Pre-existing
research files outside this turn's scope retain their own lint status and were not
touched.

## 12. Evidence artifacts

- `GATE_C2_IDENTITY_MANIFEST.json` — canonical manifest (all required fields)
- `GATE_C2_SOURCE_ROWS.json` — per-file fingerprints for all 1,693 occurrences
- `GATE_C2_IDENTITY_SUMMARY.json` — identity counts + manifest digest
- `GATE_C2_ALIAS_EVIDENCE.json` — every pair sharing an exact cell
- `GATE_C2_PLOT_MEMBERSHIP.json` — per (tree, plot) membership evidence
- `GATE_C2_INSTANCE_NAMESPACE.json` — per-plot and merged instance namespaces
- `GATE_C2_PARTITION_POOLS.json` — pools, leakage checks, neighbour support
- `DEDUP_REPORT.json` — dedup/split summary in the required name
- `GATE_C2_CASE_REVIEW.json` — representative cases A–D
- `GATE_C2_VERIFICATION.json` — independent re-computation checks
- `GATE_C2_REMEDIATION.json` — source-set isolation policy, quarantine set, assertions
- `GATE_C2_CONTEXT_PROVENANCE.json` — per-candidate context/donor provenance
- `GATE_C2_REMEDIATION_REPORT.md` — remediation report

## 13. Publisher clarification

Not required for the identity layer: the alias structure is exactly explained by the
publisher's own documented boundary duplication, the instance namespace is consistent
(1,154/1,154 agreement where testable), and the canonical counts coincide with the
published per-site tree counts. A short draft is nevertheless retained at
`GATE_C2_PUBLISHER_CLARIFICATION_DRAFT.md` for the three residual questions (instance
scope, completeness of tree files, and the Wytham naming series). It has not been
sent; sending it is the Supervisor's decision.

## 14. Verdict and what was NOT done

**FINAL_WORKER_VERDICT = GATE_C2_WORKER_READY_FOR_SUPERVISOR_REVIEW.** Canonical
identity is defensible, the held-out pool (772 primary, 863 core) exceeds the frozen
30 unique-root minimum by a wide margin, and no leakage was found. Gate D/E training
was **not** started, and the protocol V2 is a worker proposal awaiting independent
Supervisor review — it is not a self-granted PASS.

No Gate A/B rerun, no dependency installation, no ZIP download, no PointNet/PointNet++
training, no local TreeLineWarning challenge access, no product UI or scientific
algorithm change, no frozen V0.1 change, no push. Only `research/tree_qc` files were
added.
