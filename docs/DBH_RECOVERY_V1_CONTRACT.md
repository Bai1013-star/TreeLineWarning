# DBH Recovery V1 contract

Supervisor design recorded before implementation, 2026-08-31.
Baseline: e37371718c64d4efee9472b1b99a933510875043.
Existing dirty core/stage_b/batch.py is unrelated and excluded.

## Scope and authority

DBH is an additive finalized-geometry sidecar, never crown membership. Ground,
Wire, CROWN_STEM_HYBRID, reconciliation, clearance and risk remain unchanged.
Keep the existing breast-height interval [0.85, 1.75] m, radius of stem search
0.60 m, measured minimum 12 points, plausible diameter [0.05, 1.50] m.
No target count of successful measurements is an acceptance threshold.

Each generation binds source identity (including SHA256 and point count), run,
project, source preparation/ground artifact, final original_index/XYZ/tree-ID
arrays, persistent/global identity and span lineage, the entire editor journal
digest (not merely operation count), finalized state, algorithm/config version,
and exact DBH evidence original_index/XYZ/HAG arrays. References are confined
relative project paths. Files are content-validated on restart. Missing, stale,
malformed or corrupt proof never authorizes recovered DBH.

The geometry sidecar contains all final tree rows, including explicit unavailable
reasons. No analysis tree record, crown member array or clearance file is rewritten
by DBH recovery. Current report remains read-only. Recovery is explicit at finalize
or through a bounded project recovery command; opening old projects does not run
recovery. Legacy projects without a sidecar retain their existing behavior.

## Editor semantics

- DELETE: only surviving final identities are eligible; deleted lineages cannot
  donate evidence. Exact removed original_index values are always excluded.
- LASSO / REMOVE_POINTS: the existing historical journal stores removed crown
  indices, not the polygon/camera for previously unassigned points. Therefore V1
  invalidates ALL recovered/raw DBH evidence for an affected source tree lineage.
  It does not guess which excluded stem points were outside that polygon. Direct
  evidence still present in final membership may be fitted. This conservative
  rule also applies to component removals and does not alter existing edits.
- MERGE: union eligible source lineage with stable ordering and exact index dedup;
  ownership must describe one unambiguous physical stem. Distinct stems are not
  averaged or assigned a fabricated composite diameter. Edited parents remain
  invalid after merge. Cross-tree ambiguous raw candidates are rejected.
- UNDO: derive eligibility from the current complete operation list. Removing an
  operation restores the previous evidence state deterministically. A changed
  journal invalidates prior DBH results even when the operation count is equal.
- FINALIZE: only exact finalized members plus the matching journal may publish.
  A subsequent edit immediately invalidates the sidecar through its binding.
  DBH failure cannot invalidate or change existing clearance or final membership.

For the authoritative copied project, five surviving trees have legacy lasso
provenance. Those may legitimately remain unavailable even though pre-edit lower
stem representatives exist. Do not claim all ten F trees are automatically usable.

## Recovery and bounded association

First inspect direct final-member evidence. Next reproduce existing stem/crown
association from persisted fine representatives using the unchanged hybrid routine;
require exact agreement with persisted labels. Within the primary hypothesis's
existing deduplicated stem group, prefer a validated column with breast-height
support (support count, confidence, stable coordinates/ID tie-break). Keep the
primary axis when no such column exists. This review correction recovers evidence
discarded by the earlier representative choice; it changes neither hybrid labels
nor any fit threshold. Do not cross into another stem group. Retain exact
original_index and validate prepared source identity. Multiple plausible axes or
neighbor ownership conflicts fail closed.

Use the existing source-bound raw spatial tile cache for conservative local ROIs;
never scan the full source to build a new index during recovery. Filter only a
0.60 m cylinder around the validated axis, [0.5, 5.5] m HAG for continuity and the
unchanged DBH interval for fitting. Record intersecting tiles, opened points,
candidate counts, bounds and whether raw cache recovery was used. Missing cache
or absent valid axis reports unavailable, not absence of a stem in the raw LAS.
Points excluded by current editor state are not eligible for local recovery.
Engineering bounds are nine tiles, 12 million opened rows, 250,000-row read
chunks and two million returned ROI rows. The opened-row cap was raised after
the first bounded probe rejected dense tiles before testing any geometry;
the 0.60 m ROI and all scientific fit thresholds remained unchanged.

## New fitting-quality policy (V1, fixed before the 12-tree fit)

Independent NumPy/SciPy implementation; no external forestry source copied.
Six nonoverlapping 0.15 m horizontal sections in [0.85, 1.75]. Per section:
at least 12 unique points and 12 retained inliers (review clarification);
deterministic bounded 256 three-point circle hypotheses;
at most 4096 deterministically selected points for fitting; robust refinement;
radial inlier tolerance max(0.008 m, 5% radius), capped at 0.020 m; inlier fraction
>=0.60; angular span >=180 degrees and >=6/12 occupied sectors; plausible diameter
as above. Require >=3 accepted sections spanning breast height, relative diameter
spread <=20%, and a consistent stem center track. A maximum 20-degree tilt and
0.03 m center-track residual constrain V1 horizontal-section bias; report failure
rather than enabling an unvalidated ellipse fit. Retain every section diagnostic.
All thresholds are quality guards, not tuned against this project's success count.

### Review correction: separate stem candidates from ROI clutter

The initial exact-project probe passed every point in a search cylinder to the
circle fitter. Repeated circular candidates were rejected because nearby
vegetation dominated that cylinder. A search ROI is not itself stem membership.
Before further fitting, evaluate an independent vertical-continuity filter:
0.02 m XY columns, occupied in at least four of the six fixed DBH sections,
with support below and above 1.30 m. Occupancy counts distinct sections, never
dense points in one section. Use a local translated voxel origin, retain exact
input-row/original-index masks, and record filtered counts. This filter does not
use any fitted circle or its residuals to select points. It therefore cannot
make its own RANSAC winner pass by discarding everything outside that circle.
The unchanged circle quality checks are applied afterward. Strongly leaning or
sparse stems may remain unavailable under this conservative V1 filter. Validate
synthetic clutter/partial arcs and saved exact local evidence before acceptance.

## Estimates

Use no species assumptions or literature coefficients. Only reliable point-cloud
fits may train a project-local log(DBH)=a+b*log(height) model: >=10 independent tree
identities, positive slope, leave-one-out relative RMSE <=25%, maximum relative
error <=50%, and no extrapolation outside training height range. No multivariate
feature search on this tiny sample. Report training IDs, coefficients, version,
LOO errors and an approximate residual-based uncertainty (not a calibrated CI).
If unavailable/invalid, retain UNAVAILABLE with a reason. Never use an estimate as
training input or label it as directly measured. Unsafe edited associations do
not become measured through a model; all estimates must explicitly say ESTIMATE.

## Publication

Methods: DIRECT_STEM, RECOVERED_STEM, ALLOMETRIC_ESTIMATE, UNAVAILABLE.
Keep diameter_cm / diameter_source compatibility aliases and add dbh_cm,
dbh_method, dbh_quality, point_count, fit_rmse, slice_count, fit_status, source,
model_version and uncertainty. Expose source beside the number in the table.
Append Excel columns; preserve leading columns. A valid matching sidecar overrides
only DBH fields. No changes to tree count/IDs, height/crown geometry, clearance/risk.

## Public references and license handling

Reviewed public descriptions only, not implementation source:

- 3DFin tutorial: https://github.com/3DFin/3DFin_Tutorial — vertical axes,
  repeated cross sections and adjacent-section validation.
- dendromatics API: https://dendromatics.readthedocs.io/en/latest/dendromatics.html
  — section fitting and angular occupancy checks.
- https://github.com/3DFin/3DFin and https://github.com/3DFin/dendromatics:
  repositories identify GPL-3.0; dendromatics PyPI identifies GPL-3.0-or-later.
- https://github.com/tls-tools-ucl/TLS2trees — sliced stem/graph continuity;
  GPL-3.0. It is a segmentation reference, not proof of DBH accuracy here.
- https://antguz.github.io/rTLS/reference/circleRANSAC.html — bounded robust circle
  estimates, inlier proportion and RMSE; https://github.com/Antguz/rTLS identifies
  GPL-3.0. No code, snippets, binaries or dependencies from these projects added.

This is a copying/dependency scope assessment, not a general legal clearance.

## Gates

Focused fitting/provenance/editor/persistence/API/export tests, exact twelve-tree
acceptance and before/after non-DBH invariants. No 105M pipeline, package rebuild,
push, or local commit until implementation, tests and independent parent review
pass. Original audit and release artifacts remain intact.
