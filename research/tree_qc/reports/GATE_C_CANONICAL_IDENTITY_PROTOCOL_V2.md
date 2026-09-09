# Gate C canonical identity protocol V2 — prospective, before examples/training

Status: **worker proposal for Supervisor review. Not a Gate C PASS.** V1 is retained
unchanged and is **not** described as wrong. V1 was written under a conservative
assumption ("local instance IDs and coordinate frames may not be comparable across
publisher folders"). Direct inspection of the verified archives shows the official
data semantics are stronger than that assumption, so a second protocol is proposed
that keeps every V1 standard — whole-source assignment, disjoint identity pools,
real neighbour geometry, the predeclared 30 unique-root support minimum — while
replacing the unresolved-ID layer with an evidence-graded canonical identity layer.

Nothing in V1, Gate A, Gate B or the frozen V0.1 is modified by this document.

## 1. Official identity semantics

### 1.1 DOCUMENTED (publisher, Zenodo record 16875688 description)

1. "Each plot is divided in labeled plot point clouds for the test, validation and
   training areas."
2. "All files are in ply format, with following scalar fields: semantic (0: ground,
   1: tree) and instance (sequential, starting at 1, -1 for ground points). They may
   be merged to form a single, labeled plot pointcloud."
3. "Additionally, individual tree point clouds are attached in a zip container,
   divided between test, validation and training folders. Please note that all trees
   overlapping with each area are in the corresponding folders, so some edge trees
   are duplicated in multiple folders."

Statement 3 is an explicit publisher statement that a single physical tree is stored
in more than one folder. It is the documented cause of the "boundary duplicate"
phenomenon; it is not an archive defect.

### 1.2 OBSERVED (this audit, over the verified extraction)

- 1,693 individual tree PLY occurrences; 1,579 distinct `(site, basename)` pairs.
- 113 groups of occurrences share an identical float32 coordinate multiset; every one
  of those groups crosses publisher folders. 112 groups are pairs, one is a triple.
  They account for exactly the 114 extra occurrences (1,693 - 114 = 1,579).
- Tree files in `train/` and `val/` carry native `instance`/`semantic` fields; tree
  files under `test/` carry XYZ only.
- The `instance` field is a **site-global namespace**, not a per-plot renumbering.
  Litchfield's three area clouds together contain exactly 703 distinct tree IDs
  (2–704), equal to its 703 canonical trees; Wytham's span 1–876 with 783 IDs
  actually present in the area clouds. A tree therefore keeps the same ID in every
  area file it appears in.
- Every tree file's points map onto one dominant plot instance ID. Of the 1,215
  canonical trees carrying a single native instance ID, the 1,154 with plot support
  all satisfy native ID == dominant plot instance, with zero disagreements; the other
  61 have no plot support. 1,486 of 1,579 canonical trees have exact plot support,
  and the 93 without are all Wytham out-of-plot trees, matching the 93 IDs absent
  from Wytham's merged namespace.
- In 1,226 of 2,276 matching (tree, plot) records 100 % of the matched points carry
  the dominant instance; only 219 fall below 0.99, and the lowest shares belong to
  the high-overlap pairs of §6.
- Exact coordinate cells carrying two different instance IDs exist inside a plot
  (Wytham test 2.76 %, train 0.98 %, val 0.52 % of point rows). These are duplicate
  raw scan points assigned to two neighbouring trees during manual extraction; they
  are shared *points*, not shared *trees*.
- No `(site, basename)` maps to two different coordinate groups, and no two distinct
  basenames share an identical coordinate multiset. There is no observed basename
  identity conflict.

### 1.3 INFERRED

- A tree PLY is the **complete** extracted tree, not a per-area crop; the area plot
  file is the cloud clipped to that area. This follows from the publisher statement,
  from the exact alias groups (identical full multisets across folders) and from the
  accounting that folder tree-file totals exceed the corresponding plot tree-point
  totals.
- Metric metres and a shared local frame per site follow from the pinned evaluator's
  documented 1 cm overlap threshold and 2-decimal rounding of unscaled XYZ.

### 1.4 UNRESOLVED

- Absolute georeferencing (EPSG/global CRS) remains unproven and is not used.
- The physical meaning of a few high partial-overlap pairs (up to 0.81 containment
  between two distinct basenames) is not resolved by the archive alone; V2 keeps them
  independent, which is the conservative reading and matches the publisher's own
  distinct instance IDs. All 17 pairs above 0.30 containment were checked against the
  labelled area clouds: in every case the two trees carry **different** publisher
  instance IDs (e.g. 1610 → 306 vs 8361 → 485; 8330 → 43 vs 8332 → 643), so the
  publisher's ground truth treats them as two trees. An alternate held-out pool that
  excludes every affected tree is reported alongside the primary pool so the
  Supervisor can choose the stricter reading without any re-derivation.

## 2. Identity evidence hierarchy

| Level | Name | Requirement | Action |
|---|---|---|---|
| A | `LEVEL_A_EXACT_ALIAS` | identical float32 coordinate multiset (equal point count and equal order-independent fingerprint) **and** agreeing basename | union into one canonical tree |
| B | `LEVEL_B_STRONG_ALIAS_CANDIDATE` | containment of the smaller set >= 0.95, centroid gap <= 0.25 m, overlapping bbox, but not an exact multiset match | report for review; **never** auto-union |
| C | `LEVEL_C_PARTIAL_OVERLAP_ONLY` | any shared XYZ cell or bbox intersection short of A/B | keep independent and unresolved |

Thresholds are declared here, before any pool statistic is computed. A level-A match
is coordinate identity of the complete stored observation; a level-C match is
compatible with two genuinely distinct trees and is therefore not evidence of
identity.

## 3. Canonical tree identity

`canonical_tree_id = "<site>:<basename>"`. A canonical tree is the set of all file
occurrences that are exact aliases of one another. If a basename were ever to map to
two different coordinate groups the id would carry a deterministic fingerprint suffix
and the group would be `UNRESOLVED`; this does not occur in the verified data.

The manifest records, per canonical tree: `canonical_tree_id`, `site`,
`source_aliases`, `source_file`, `folder_partition`, `folder_partitions`,
`local_instance_id`, `point_count`, `unique_cells`, `bbox`, `centroid`,
`coordinate_fingerprint`, `alias_evidence`, `identity_status`, `occurrence_count`,
plus the derived `dominant_plot_instance` and `plot_support_cells`.

`identity_status` is `EXACT` when the group is a verified exact-alias group or a
single occurrence with no conflicting alias, `STRONG_VERIFIED` only if a level-B
candidate were independently verified, and `UNRESOLVED` when basenames disagree
within one exact group.

## 4. Plot membership

Membership is decided by **exact float32 cell equality** between the tree file and the
labelled plot cloud, streaming the plot in bounded chunks. For each
`(canonical tree, plot)` the audit records `matched_cells`, `matched_rows`,
`match_fraction`, `dominant_instance`, `dominant_share` and the instance histogram.
No nearest-neighbour matching and no ID-based inference is used. A shared cell is a
shared observation, never a licence to union two trees.

## 5. Leakage prevention

- A canonical tree is assigned to **exactly one** partition. Its alias occurrences
  never count as independent samples and never span partitions.
- Litchfield (model-selection site) keeps the V1 fixed geometric split: the x boundary
  is the 70th-percentile index of canonical centroids, computed before eligibility
  filtering and never moved to improve yield. A tree goes to train only if
  `bbox_xmax + 3 m < boundary`, to validation only if `bbox_xmin - 3 m > boundary`;
  everything else is quarantined.
- Wytham is the held-out site; every Wytham canonical tree is held out.
- `TRAIN_VAL_CANONICAL_LEAKAGE` and `CROSS_SITE_LEAKAGE` are reported as set
  intersections of canonical ids and must be empty.
- The same local instance ID must never denote two canonical trees; collisions are
  reported, not silently resolved.

## 6. Real context

The neighbour index is provenance-bound and built over canonical trees, so an alias
file of the candidate can never appear as its own neighbour. For a candidate, the
query is the candidate XY bbox expanded by 3 m with the candidate's native Z extent.
Neighbours are canonical trees whose bbox intersects the query; context membership is
the actual neighbour points inside the query. Duplicate observations of the same
physical tree are suppressed by construction (one entry per canonical tree). No
invented context, no random donor trees, no plot background in this first
implementation.

## 7. Resulting pools

| Pool | Count |
|---|---:|
| Litchfield train | 442 |
| Litchfield validation | 177 |
| Litchfield boundary quarantine | 84 |
| Wytham held out (all canonical trees) | 876 |
| Held-out pool, primary (identity + plot support + real neighbour points + >= 4,096 points) | **772** |
| Held-out pool, core (without the plot-support requirement) | 863 |
| Primary pool excluding high-containment-affected trees | 749 |
| Core pool excluding high-containment-affected trees | 838 |
| `TRAIN_VAL_CANONICAL_LEAKAGE` | 0 |
| `CROSS_SITE_LEAKAGE` | 0 |

The boundary and the split sizes reproduce V1 exactly (`x = 67.92401220579217 m`,
442 / 177 / 84), so V2 changes only the identity/eligibility layer, not the geometry.

## 8. Preserved standards

- Minimum independent support stays at **30 unique roots**; it is not lowered.
- Context is not shrunk to manufacture support, and no partial match is promoted to
  identity.
- V1 and its reports remain unchanged as the record of what the conservative
  assumption produced.

## 9. Source-set isolation and high-containment quarantine (remediation R1/R2)

Added after independent Supervisor review found that partition checks covered candidate
identity but not every tree participating in an example.

**EXAMPLE_SOURCE_SET** = candidate canonical tree ∪ candidate aliases ∪ context
canonical trees ∪ context aliases ∪ corruption-donor canonical trees ∪ donor aliases.
For the two model-selection partitions, `TRAIN_SOURCE_UNION ∩ VALIDATION_SOURCE_UNION`
must be empty. The requirement applies to whole canonical source trees, not sampled
points: a canonical tree may not contribute even one context or donor point to both
partitions.

Deterministic policy, fixed before any model result exists (implemented as
`isolate_partition_sources`):

1. A quarantined source contributes to neither partition.
2. A candidate whose surviving context is empty is excluded — fail closed.
3. A source still present in both unions is removed from the side where it is not a
   candidate; if it is a candidate on neither side it is removed from both. A source
   is never removed from the side where it is a candidate.
4. Steps 1–3 iterate to a fixpoint; non-convergence raises rather than passes.
5. Donors are drawn only from the candidate's own surviving context, so donor leakage
   is structural. No generator exists yet, so donor sets are empty and the donor
   assertions hold vacuously; the rule binds the Gate D generator.

Geometry is unchanged: the 3 m context radius and the bbox/centroid rules are
untouched. The measured effect is 5 canonical trees that were context on both sides;
all five are Litchfield boundary-quarantine trees (candidates in neither partition), so
rule 3 removed them from both sides.

**HIGH_CONTAINMENT_QUARANTINE_SET** = the 25 canonical trees in the 17 pairs with
containment ≥ 0.30 (all Wytham). They may not contribute as candidate, context, donor
or alias source to the strict held-out pool. Publisher labels are unchanged and the 17
pairs are not unioned. Each held-out candidate's context is rebuilt without them and
re-evaluated; a candidate left with no point-bearing context is excluded with an
explicit reason. The 3 m radius is unchanged.

The previous `STRICT_HELDOUT_POOL = 749` is withdrawn as a frozen number. The
re-derived strict pool happens to contain the same 749 candidates — the candidate set
is unchanged because all 200 candidates that referenced the quarantine set retain
between 5 and 64 surviving point-bearing context sources — but its context provenance
is different for exactly those 200 candidates. The count is an outcome, not an
assumption.

## 10. What V2 does not claim

- It does not claim the archives were corrupt.
- It does not claim a level-C overlap pair is one tree, nor that it is two; it keeps
  them independent.
- It does not authorise training, generation, threshold selection or held-out
  scoring. Those remain blocked until an independent Supervisor accepts this protocol
  and the frozen manifest.
