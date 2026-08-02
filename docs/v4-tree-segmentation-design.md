# V4 Full-Resolution Individual-Tree Segmentation Design

**Canonical path:** `docs/v4-tree-segmentation-design.md`
**Status:** `DESIGN-FROZEN`
**Scope:** proposed implementation design only. This document authorizes no V4 implementation, schema, dependency, or change to V2/V3 public contracts.

## 1. Authority, review, and freezing

This is the sole candidate design at the canonical path. It supersedes the untracked draft at `docs/V4_tree_segmentation_design_frozen.md`; that draft is deliberately removed, not preserved as a frozen authority.

**Normative upstream design:** `docs/full-resolution-vegetation-membership-design.md`
**Upstream status:** `DESIGN-FROZEN`
**Freeze date:** `2026-08-01`

FRV is the design authority for full-resolution vegetation membership, membership scope and inventory, membership/HAG evidence and readiness, and wire-provider/override/provenance semantics. V4 is the design authority for consuming validated FRV evidence, exact measured-point recovery, tree segmentation, its safety hand-offs, and its clearance request boundary. A conflict between this document and FRV on membership, scope, inventory, HAG-readiness, or wire-provider semantics must stop the affected work and reopen the applicable design; V4 must not silently override the frozen FRV rule.

The final independent V4 design review passed, and separate design-freeze approval was completed on 2026-08-01. This document is `DESIGN-FROZEN`. The frozen FRV design resolves the upstream structural boundary, but FRV implementation and runtime artifacts do not yet exist. Therefore V4 execution is blocked until valid FRV artifacts exist; no implementation is authorized by this document.

Post-freeze semantic changes require the controlled reopening process in Section 19.4. V4-C may establish `IMPLEMENTATION-ACCEPTED`, `BASELINE-ACCEPTED`, or `RELEASE-READY` (or equivalent implementation-acceptance states); it cannot replace this design-freeze approval.

## 2. Objective and method

V4 separates already identified vegetation candidates into individual, above-ground tree instances at original measured-point resolution. It must preserve the identity of every measured point used for safety decisions.

The required method is:

1. The frozen FRV upstream design defines and, after separate implementation authorization, produces full-resolution vegetation-membership evidence.
2. V4-A validates FRV source identity, scope coverage, membership-artifact conservation, and exact original-index replay.
3. V4-A recovers only real measured source rows admitted by valid FRV states, while membership-review points and coverage-blocked regions remain visible in separate safety hand-offs.
4. V4 constructs a robust, support-audited CHM and continues with the frozen V4 segmentation method.
5. All seeds synchronously compete in XY to establish initial identities, horizontal responsibility, three-dimensional candidate search ranges, and strong priors.
6. XY responsibility is revisable: overlapping claims can remain valid at different heights.
7. Adaptive overlapping vertical slices recover measured support and synchronously compare every candidate tree for each recovered point.
8. Only that final three-dimensional comparison creates formal original-point **tree** membership.
9. Only original measured points support formal tree-to-wire clearance; unresolved points and recovery failures remain in the conservative safety path.

V4 estimates above-ground tree extent and a trunk-base region. It does not model roots or underground geometry.

## 3. Exact module boundary

### 3.1 FRV consumer and coexistence boundary

V4 consumes the separately frozen FRV producer boundary. It does not own, implement, redefine, or infer the full-resolution vegetation-membership rule. V4 may define a read-only `V4ValidatedFrvMembershipView` only as a proposed V4 consumer boundary, not a replacement FRV schema. V4 is a new, side-by-side pipeline with its own segmentation contracts and must neither alter nor reinterpret V2/V3 public contracts. V3 remains the current V2-row-aligned result path.

The V2-row compatibility summary in Section 10 is a one-way, explicitly lossy adapter. It is optional, may support existing displays or review correlation, and must never be used as V4 recovery, formal membership, or formal-clearance input.

### 3.2 Inputs

The formal upstream inputs are a validated
`FullResolutionVegetationMembershipArtifact`, a validated
`FRVScopeCoverageManifest`, frozen-FRV-compatible `SourceIdentity`, exact
source-file stream access, FRV membership/HAG/readiness evidence, and FRV
provider/override/provenance bindings. V4 validates them before it derives a
segmentation-ready `V4RecoveredVegetationSet` or a
`V4RecoveryCoverageReport`.

V1 point labels, V2 cluster labels, V3 `TreeSegmentationInputSet` membership,
and A1 representative geometry are not authoritative V4 inputs. They may occur
only in diagnostic, compatibility-summary, or audit references and never prove
FRV membership, reconstruct an original point, or enter normal segmentation.

CHM, seeds, density field, XY claims/regions, slice results, final assignments, and envelopes are V4 products. Existing V3 CHM/candidate products may inform evaluation but are not authoritative V4 inputs because they are based on V2-aligned rows.

### 3.3 Outputs

The proposed primary output is `V4TreeSegmentationResult`: a full-resolution
segmentation-ready result keyed by unique `original_index`, final point states,
ragged measured-point membership, immutable provenance, and diagnostic evidence
references. Separate proposed `V4MeasuredClearanceRequest` and
`MeasuredClearanceResult` records preserve the request/result boundary. A
separate safety inventory contains known unresolved measured points and
unrecovered regions. A scenario may not claim complete measured clearance
unless the independent completeness axes in D05 permit it.

### 3.4 Non-goals and dependencies

V4 does not detect wires, replace the vegetation-candidate stage, recalculate a ground model, predict growth, assign warning thresholds, or build an editor. It does not create synthetic points. It does not make a CHM watershed, Voronoi partition, DBSCAN cluster, crown circle, or 3-D envelope into final clearance geometry.

No dependency is assumed by this design. The first implementation must use the installed runtime and repository-supported numerical/streaming facilities unless a dependency review approves a necessary capability. Polygon repair or 3-D mesh processing that would add a dependency needs separate approval. `algorithms.io.chunks.iter_las_chunks` is the existing measured-point streaming mechanism and is the required traversal primitive unless an equivalent replacement is separately reviewed.

## 4. Verified original-point recovery boundary (V4-A)

### 4.1 Gate and permitted evidence

V4 density segmentation must not start until valid FRV artifacts exist and the
independent V4-A gates for FRV compatibility/source identity, scope coverage and
membership conservation, measured-XYZ recovery, and segmentation HAG readiness
are satisfied. This is a runtime/implementation dependency, not a structural
design-freeze blocker. V2/V3 row identity is insufficient:
`TreeSegmentationInputSet` is aligned to V2/downsampled rows, and
`vegetation_points.xyz_local` is not the full original XYZ population.
Preserved `original_index` and `sample_index` do not, in the current V3 input,
expose enough evidence to recover every full-resolution vegetation member.

FRV, not V4, proves and publishes full-resolution vegetation membership over
original source rows. V4 consumes only the validated FRV membership artifact,
coverage manifest, `SourceIdentity`, configuration, provider, override, and
provenance bindings. V4 replays the named source through `iter_las_chunks`,
validates identity, and collects exactly the recorded `original_index` rows; it
does not evaluate a substitute membership rule.

Nearest-neighbour propagation from downsampled labels, fixed-radius expansion, arbitrary voxel bounds, admitting every point in an XY prism, and blind propagation from a representative cluster are forbidden recovery substitutes. Geometry may limit work only after membership is proven; it cannot prove membership.

### 4.2 Recovery procedure and HAG tolerance

For each FRV-compatible source identity, V4-A validates the source, artifact,
manifest, provider/override bindings, and conservation before it streams stable
ascending `original_index` and recovers real XYZ. Exact recovery applies to
`VERIFIED_VEGETATION_CANDIDATE`, `MANUAL_INCLUDE`, and, when present,
`REVIEW_REQUIRED_MEMBERSHIP` records. XYZ recovery is independent of FRV
membership and HAG states. Only points with membership state in
`{VERIFIED_VEGETATION_CANDIDATE, MANUAL_INCLUDE}` and
`segmentation_hag_readiness=READY` enter CHM, treetop, slice, and normal 3-D
assignment. Membership-review and HAG-not-ready real points remain in the D06
measured-point safety hand-off.

V4 validates and preserves FRV's versioned HAG tolerance, policy, raw/clamped
values, reason/quality, and readiness decision. It neither clips nor recomputes
HAG, nor changes an FRV HAG or membership state. Any invalid FRV HAG binding or
non-ready status is a V4 review/routing condition, never a local replacement
rule.

No output point may be copied from a downsampled XYZ array. Local coordinates are a deterministic translation of streamed measured XYZ by the recorded origin; they are not newly sampled geometry.

### 4.3 Source content identity and path relocation

Every recovery input, result, diagnostic bundle, and clearance hand-off carries a proposed immutable `SourceIdentity`.

| Field | Required meaning |
|---|---|
| `source_id` | Non-empty stable source identifier. |
| `recorded_source_path` | Original recorded path for provenance; it is not content identity. |
| `resolved_access_path` | Path resolved for this run; it is access information, not content identity. |
| `source_file_sha256` | SHA-256 over raw source-file bytes. |
| `canonical_point_content_digest` | Optional digest over frozen canonical decoded original rows; it is distinct from the file digest. |
| source summary | Point count, bounds, coordinate frame and units, origin/context, plus relocation and source-equivalence audit where applicable. |
| FRV bindings | Artifact, manifest, block, configuration, provider, override, and provenance digests. |

`source_file_sha256` validates raw-file identity. `canonical_point_content_digest`,
when present, validates canonical decoded point content under FRV's frozen
definition and must never be conflated with the file digest. A relocation audit
must record recorded and resolved paths, resolution method, timestamp/run
identity, and identity-check result. Exact first-version replay requires matching
file bytes; a re-encapsulated source requires FRV-compatible source-equivalence
audit. A digest, point count, bounds, origin/context, coordinate-frame, or unit
mismatch is a hard identity error.

### 4.4 Recovery coverage and failure closure

`V4RecoveryCoverageReport` is a proposed required V4 companion result derived
from `FRVScopeCoverageManifest`, FRV membership-artifact validation,
source replay/recovery outcomes, and V4 HAG/readiness partition outcomes. It
does not invent a competing upstream candidate inventory. It references the FRV
`frv_scope_coverage_manifest_digest`, `frv_scope_union_id`,
`frv_upstream_inventory_digest`, `frv_membership_artifact_digest`,
recovered-index digest, and recovered/review/blocked
counts. It records D05’s separate source identity, XYZ, membership, HAG,
segmentation-input, routing, and measured-clearance completeness axes. A
coverage-blocked region without recovered points remains distinct from a region
with recovered XYZ but unavailable HAG.

V4 validates and inherits, rather than redefines, FRV's completeness facts. In
the V4 consumer view/report, the validated FRV fields are exposed as
`frv_canonical_inventory_complete`, `frv_required_inventory_complete`,
`frv_approved_hard_rejection_complete`, `frv_scope_coverage_complete`,
`frv_routing_complete`, and `frv_membership_decision_complete`. Their
corresponding FRV counts are exposed as `frv_canonical_inventory_count`,
`frv_required_inventory_count`, `frv_approved_hard_rejection_count`,
`frv_scope_created_or_merged_count`, `frv_coverage_blocked_count`,
`frv_unique_in_scope_point_count`, `frv_evaluated_point_count`, and
`frv_unasserted_in_scope_count`. These are validated inherited FRV facts, not
V4-recomputed completeness fields. In particular, V4 validates:

```text
all canonical upstream inventory records
= required FRV records + approved upstream hard-rejection records

required FRV records
= scope-created or scope-merged records + coverage-blocked records

unique_in_scope_point_count
= all five terminal FRV membership-state counts

unasserted_in_scope_count = 0
evaluated_point_count = unique_in_scope_point_count
```

`frv_membership_decision_complete` is true only as the verified inherited FRV
meaning: FRV field `membership_decision_complete` is true; artifact, manifest, provider, stream, and
integrity validation passed; the required conservation equations pass; and no
affected required region is coverage-blocked. Mandatory-provider, stream,
artifact-integrity, coverage, or identity failure makes the affected scope
false. Independent healthy scopes may continue, but source-level completeness
aggregates every required scope and scene-level completeness aggregates every
relevant source/scope. Point-review counts never mask a coverage failure, and
V4 may only derive downstream completeness axes; it never overwrites FRV
completeness. V4 may validate, retain, reference, and aggregate this inherited
fact; it must not recompute, override, relax, or turn a false FRV value true.

The point recovery states are `POINT_RECOVERY_FAILED`,
`POINT_RECOVERED_HAG_READY`, `POINT_RECOVERED_HAG_REVIEW_REQUIRED`,
`POINT_RECOVERED_HAG_MISSING`, and `POINT_RECOVERED_HAG_INVALID`. A failed
region must not invent points, treat all prism/boundary rows as vegetation,
produce a formal tree, or produce a complete measured-clearance claim. A
recovered point with missing/invalid HAG retains `original_index` and measured
XYZ, is excluded from normal segmentation, and is routed under D06.

## 5. Full-resolution recovery structure

`V4RecoveredVegetationSet` represents exactly the measured original,
FRV-admitted, segmentation-HAG-ready vegetation candidates for one validated
FRV source/artifact/manifest set.
`N` is segmentation-ready candidate-point count. It is not the complete
inventory of known original points: recovered membership-review and
HAG-not-ready points are retained in the D06 measured-point safety hand-off.

| Field | Proposed dtype and shape | Meaning |
|---|---|---|
| `original_index` | `int64[N]` | Strictly increasing, unique source-row identity. |
| `xyz_local_m` | `float64[N,3]` | Streamed measured XYZ minus origin; finite. |
| `hag_m` | `float32[N]` | Finite FRV-provided full-resolution HAG for rows with readiness `READY`. |
| `membership_state` | `uint8[N]` | FRV state, only `VERIFIED_VEGETATION_CANDIDATE` or `MANUAL_INCLUDE`. |
| `hag_evidence_state` / `segmentation_hag_readiness` | enum arrays | FRV evidence state and validated V4 segmentation readiness; both are retained. |
| `candidate_reason_flags` | `uint32[N]` | Evidence/manual-override reasons, not a geometry label. |
| `candidate_confidence` | optional `float32[N]` | Optional upstream FRV confidence retained only when present, version-compatible, finite, and provenance-bound. It is not required for membership validity and is not recomputed by V4. |
| `source_identity` | immutable record | Section 4.3 identity. |

Cross-field invariants: all present point-array lengths equal `N`; indices are
unique and in range; the index set equals exactly the FRV-admitted,
segmentation-HAG-ready, successfully recovered point set for the validated
source/artifact/manifest binding; measured coordinates reconstruct within
declared quantization tolerance; and every exposed array/structure is
non-modifiable. `candidate_confidence`, when absent, neither invalidates a
legal FRV membership state nor permits V4 to synthesize or infer one; it cannot
replace `membership_state`, reason, quality, or readiness. A V4-created
recovery measure, if ever needed, must use a distinct name such as
`v4_recovery_quality` or `v4_recovery_confidence`, and is not FRV membership
confidence.

`N=0` is legal when (1) authoritative FRV evidence proves no admitted
candidate; (2) admitted candidates exist but none is segmentation-HAG-ready;
(3) `REVIEW_REQUIRED_MEMBERSHIP` points exist but no normal segmentation-ready
point exists; or (4) an independently healthy scope has no normal input while
all known points and failed regions are conserved through D05/D06. `N=0`
produces no normal tree, fictional member, or fictional HAG. Recovered real
review/HAG-not-ready points remain in the measured-point hand-off and
coverage-blocked or recovery-failed regions remain in the unrecovered-region
hand-off. It does not make `v4_routing_complete` or
`measured_clearance_complete` true automatically.

The recovered set excludes `REVIEW_REQUIRED_MEMBERSHIP`,
`EXCLUDED_NON_CANDIDATE`, `MANUAL_EXCLUDE`, admitted candidates that are
HAG-not-ready or HAG-missing/invalid/out-of-domain, coverage-blocked regions,
and point-recovery-failed records. Those records remain conserved in the audit
or the applicable safety hand-off; none may disappear from safety output.

The contract does not require unconditional duplicate full-resolution arrays or `OWNDATA=True`. It may use an approved read-only memory map, artifact-backed array, controlled zero-copy read-only view, chunked read-only store, or shared backing store with explicit lifecycle. Constructors validate dtype, shape, contiguity or chunking rules, non-writeability, and backing-store digest. They make a controlled copy only when needed. If storage is shared, its lifetime, non-writeability, and content identity must be guaranteed so callers cannot mutate data through an alias.

## 6. CHM and seed rules

The CHM's data identity derives from verified full-resolution original measured vegetation points. It may use a versioned robust cell-height estimator, outlier rules, support filtering, empty-cell policy, and limited smoothing; it must not use severely downsampled representative points as full-resolution density/CHM data, nor mechanically use every cell's absolute highest point.

For every CHM production, retain the cell-height estimator; outlier rule; support count; empty-cell handling; selected height quantile or support-filtered maximum; values before/after smoothing; smoothing radius; small-tree-top protection rule; and configuration summary/digest. Cell size, minimum HAG, local-maximum neighbourhood, plateau policy, and seed suppression are general versioned configuration, never scene-specific branches.

A seed record proposes `seed_id`, measured/supporting location and height, confidence, quality flags, and CHM-cell evidence. Watershed/local maxima are proposals only. A seed is accepted only with configured CHM support, measured-point support, valid HAG, and non-duplicate spatial/height evidence. Rejected and suppressed proposals remain diagnostics.

## 7. Original-point XY density field and robust trend

The density field projects recovered `xyz_local_m[:, :2]`. It may use deterministic adaptive grids or support-aware kernels, but stores measured count, area-normalized density, robust local spacing, height quantiles/occupancy, and acquisition-support normalization where available. Smoothing and aggregation use only restored candidate points.

For a seed/direction, configured overlapping radial bins produce a support-normalized density sequence and a documented robust smoother. A decision evaluates a window, not a single difference: persistent decline, a supported/prominent valley, sustained recovery, and concurrent competing-seed support. A stop/transition requires the configured conjunction; isolated fluctuation or weak scan coverage is not a boundary. Missing evidence reduces confidence and can require review, but does not manufacture an empty valley.

## 8. Simultaneous competition, revisable XY claims, and irregular regions

All accepted seeds advance in common deterministic iterations over common support cells/directional bins. No seed finalizes a complete region before rivals are considered. A claim score combines versioned density-trend support, seed support, height continuity, path cost, and neighbourhood consistency. Distance/Voronoi may only initialize ties or break exact ties, never define a final boundary. Determinism uses stable ordering by `(iteration, seed_id, cell_id)` and an explicit tie rule independent of scheduling.

Each cell/XY support position records one or more claim records using equivalent semantics to `PRIMARY_CLAIM`, `SECONDARY_CLAIM`, `MULTI_TREE_OVERLAP`, `CONTESTED`, and `UNSUPPORTED`: a best current claim; retained lower-ranked plausible claims; normal multi-tree support at distinct heights; unresolved competition; or insufficient support. Claims include candidate tree IDs, scores, support evidence, score margins, and iteration/version. `MULTI_TREE_OVERLAP` is normal real 3-D structure when evidence supports multiple trees at different heights; it is not automatically an error or manual-review state.

The XY stage establishes initial tree identity, horizontal responsibility, 3-D candidate search range, conflict evidence, and strong final-assignment priors. Its primary claim is never an irrevocable point label and must be corrected by slice evidence. A tall crown over a smaller tree, different-height structures with overlapping projections, leaning branches entering a neighbour's region, upper crown/lower trunk overlap, and measured support for multiple trees at different heights all permit the same XY support position to remain a candidate responsibility range for multiple trees. Synchronous competition, deterministic unique-winner rules where justified, and conflict records remain required; the XY plane is not permanently partitioned into exclusive tree labels.

An `IrregularXYRegion` is a tree-associated topology-checked support-cell set and optional polygonal boundary. It retains tree ID, coordinate frame, resolution/representation, rings, area, confidence, claim/overlap evidence, and repair evidence. It is a candidate responsibility region and 3-D recovery prior, not a final member contract. Any repair must retain validation/repair evidence; no new geometry dependency is assumed.

## 9. Candidate prisms, slices, and reconstruction

For each tree, a `CandidatePrism` extrudes the irregular XY candidate responsibility region over a supported vertical range with versioned uncertainty margins. It retrieves plausible points and schedules slices only. It may overlap other prisms; inclusion is not membership and no first-tree ownership is permitted.

Within each prism, form overlapping slices over HAG/local vertical coordinate. Local spacing is a robust statistic of measured nearest-neighbour distances in the relevant support set (for example a validated median after excluding isolated/invalid distances), not a fictional point diameter. Slice thickness is a bounded, versioned function of spacing. Each slice records bounds, centre, spacing estimate, min/max thickness, overlap, support count, extent, quality flags, and its relationship to concurrent XY claim evidence. Weak or empty slices are gaps/review evidence unless an explicit continuity rule finds sustained measured support; they cannot create formal members.

Adjacent slices form a height-ordered graph that compares extent, centroid, boundary displacement, support, and overlap. Robust continuity preserves sustained structural change (branching or taper) and flags isolated unsupported jumps. A smooth topology-checked envelope may join accepted slice boundaries for visualization, candidate search, occlusion uncertainty evidence, growth inputs, or review. It is derived `non_measured` geometry, stored/named/reported separately from measured clearance. It cannot create points, member indices, or current formal clearance. A trunk-base output is likewise a measured-point-supported low-HAG region, never an inferred root.

## 10. Final full-resolution point assignment

This proposed stage is the only creator of final tree membership. Each recovered original vegetation point may enter one or more overlapping candidate prisms and therefore receives a candidate-tree set. XY results contribute initial identity, prior score, candidate-search restriction, and overlap/conflict evidence only; a primary XY claim is revisable and can be outweighed by three-dimensional evidence.

For every point, compare all candidate trees synchronously, never by tree-order point capture. The evidence model must admit: measured support in its height slice; relation to that tree's slice support area; spatial continuity in adjacent slices; crown/branch/trunk direction continuity; XY responsibility score; tree-top support; local height and density; score difference from competitors; and data quality, occlusion, gaps, and scan-support evidence. Configuration may calibrate their computation/weighting but records rule and configuration versions. Reliable slice evidence may correct or overturn XY primary responsibility.

A point is `ASSIGNED` only when all configured/versioned rules establish
sufficient measured support, a unique or sufficiently clear advantage, and any
required quality conditions. `UNASSIGNED` means that the point entered a
complete final comparison, candidate generation completed without contract or
integrity error, and no tree had sufficient positive evidence or mandatory
conflict review. `REVIEW_REQUIRED` means that the point entered or attempted
final comparison but conflict, anomaly, invalid evidence, incomplete
comparison, override, or explicit review requirement prevented a normal
decision. These states have different reasons, priority, and user presentation,
but both remain on the safety path. Envelope, interpolation, smoothing
boundary, or continuity prior alone can never create an assignment.

`NOT_EVALUATED` is not a state in `V4TreeSegmentationResult`: it means a real
measured point never entered normal final comparison because FRV membership or
segmentation-HAG readiness, or FRV/provider/recovery validation, prevented
normal segmentation. It is recorded in `V4UnresolvedMeasuredPointHandOff`, not
in this result. `NOT_EVALUATED` is neither `UNASSIGNED` nor a substitute for
`REVIEW_REQUIRED`, and it has no normal tree ID.

The resulting proposed `V4TreeSegmentationResult` has `N` segmentation-ready,
successfully recovered candidates, `K` formal trees, and `A` assignments. It
does not contain FRV membership-review or HAG-not-ready points:

| Field | Required invariant |
|---|---|
| `original_index:int64[N]` | Exact strictly increasing recovery keys. Formal members are only these measured indices. |
| `assignment_state_by_point:uint8[N]` | Exactly `UNASSIGNED`, `ASSIGNED`, or `REVIEW_REQUIRED`; one state per recovered point. |
| `tree_id_by_point:int32[N]` | Non-negative only for `ASSIGNED`; `-1` otherwise. |
| `assignment_confidence_by_point:float32[N]` | Finite `[0,1]`, linked to recorded scoring/version rules. |
| `assignment_reason_flags_by_point:uint32[N]` | Explains support, ambiguity, quality, override, and safety routing. |
| `tree_member_offsets:int64[K+1]` and `ordered_tree_member_original_index:int64[A]` | Immutable ragged membership; starts at zero, ends at `A`, indices unique and sorted within tree. |
| `tree_records` | Only normal formal trees with at least one member; seed/region/slice/envelope/quality evidence references. |
| provenance/config/diagnostic manifest | Immutable and identity-consistent with recovery and coverage. |

Ragged membership is authoritative and exactly equals points whose state is `ASSIGNED` and matching `tree_id_by_point`; non-assigned points occur in no member slice. Tree IDs are contiguous. A seed ending with zero formal members enters diagnostic, suppression, or review handling; it is not automatically emitted as a normal tree record. The same protected read-only ownership/lifecycle rules in Section 5 apply to all arrays.

### 10.1 Lossy V2-row compatibility summary

If needed after V4 completion, `V4V2RowCompatibilitySummary` may map only matching V2-row indices to a summary state/tree/reason. It omits full-resolution population, ragged membership, slice evidence, and unresolved coverage. It cannot reconstruct membership and must never be V4 recovery input, `TreeSegmentationResult`, formal-clearance input, or a claim of complete tree extent.

## 11. Clearance and unresolved-vegetation safety behavior

### 11.1 Formal measured-clearance request and result

V4 produces a `V4MeasuredClearanceRequest` for each real point requiring a
formal measured distance. It contains verified vegetation source identity,
`original_index`, measured XYZ or exact retrieval reference, tree ID or
unresolved/membership/HAG-review state, vegetation reason flags, wire artifact
identity, wire source identity, wire instance ID, wire geometry version, and
request status. It does not contain a nearest primitive, curve parameter,
distance, or final numerical bound.

The clearance consumer validates authoritative wire geometry and returns a
separate `MeasuredClearanceResult`: request identity, evaluation status,
nearest wire primitive ID, curve parameter/segment position, distance method
version, `distance_m`, numerical tolerance/error bound, and failure/review
reason. A request has exactly one result or an explicit `PENDING` or
`NOT_EVALUATED` outcome. Formal distance is only original measured point ->
approved wire centreline segment/curve/primitive -> exact or approved-bounded
nearest distance. Discrete samples are not approved by default.

Envelopes, slice polygons, CHM cells, prism boundaries, interpolated geometry,
synthetic points, and V2 summaries are forbidden formal-clearance inputs.
Estimated/predicted/conservative envelope evidence uses a separate result type
and metric and must not be merged numerically with measured clearance.

### 11.2 Recovered unresolved vegetation

All known real points that are `UNASSIGNED`, `REVIEW_REQUIRED`, FRV
`REVIEW_REQUIRED_MEMBERSHIP`, or FRV-admitted candidates with HAG
review/missing/invalid/not-ready route to the D06 measured-point safety hand-off.
It reports exact point identity/XYZ, membership state, HAG evidence/readiness
and value-or-null, `tree_assignment_state`, reason, priority, wire evidence,
FRV references, and a clearance request where required. Its assignment-state
rules are: `NOT_EVALUATED` for a point that never entered B4 final comparison;
`UNASSIGNED` or `REVIEW_REQUIRED` only for a point that actually entered B4;
and `ASSIGNED` only for a successfully assigned normal segmentation point.
Thus FRV membership-review, HAG-not-ready, `HAG_MISSING`, `HAG_INVALID`,
`HAG_OUT_OF_DOMAIN`, and validation-prevented points use `NOT_EVALUATED`; they
have no normal tree ID, but their recovered measured XYZ remains available for
safety review and a required D08 request. A hand-off point does not silently
disappear because it has no normal tree record or valid HAG.

### 11.3 Unrecovered candidate regions and scenario completeness

Every FRV coverage-blocked required region, and every region with
`POINT_RECOVERY_FAILED`, FRV artifact-validation failure, incomplete membership
decision, or otherwise unavailable original points routes to the separate
unrecovered-region hand-off using retained FRV evidence, without fabricating
points, `original_index`, XYZ, ordinary measured-point request, or clearance
values. It may retain only region-level safety review, unknown relevance, and
not-evaluated status.

The scenario aggregator applies two independent conservations.
`v4_routing_complete=true` means every known record has exactly one mutually
exclusive primary safety route: normal segmentation path,
`V4UnresolvedMeasuredPointHandOff`, `V4UnrecoveredRegionHandOff`, or explicit
excluded/audit-only route. `measured_clearance_complete` is separate: each real
measured point that approved downstream policy requires for measured clearance
has exactly one `V4MeasuredClearanceRequest` and exactly one
`MeasuredClearanceResult`, or an explicit `PENDING`/`NOT_EVALUATED` outcome.
A real point may therefore be in the D06 unresolved measured-point hand-off and
also have a D08 request; D06 safety state and D08 calculation request are not
alternatives. A region never creates a fictitious point request.

## 12. Required diagnostic evidence and retention boundary

The diagnostic manifest retains/references immutable source/membership/HAG validation; decision-scope and recovery-index digest/counts; coverage report and all completeness axes; configuration digest; CHM support and seed decisions; density/trend windows; concurrent claims/ties/overlaps; XY repairs; prisms; slice/continuity evidence; non-measured envelope marker; final point states/reasons/confidence; clearance request/result identities and method evidence; and resource/determinism audit.

Retention distinguishes (a) long-lived evidence needed to reproduce/review a final point assignment, including source identity, membership/HAG bindings, final scoring inputs/rules, and state/reason outputs; (b) intermediate evidence reproducible from immutable source/configuration, such as replayable grids and temporary computations; and (c) evidence that may be compressed, chunked, or retained by digest/reference. Compaction must not discard the minimum evidence needed to reproduce a final assignment, unresolved safety hand-off, or recovery-completeness decision.

## 13. Manual review boundary

The first V4 implementation preserves review evidence and stable original-index keyed correction targets; it does not implement an editor. A future editor may correct boundaries, delete/merge/split trees, or reassign points only through versioned override artifacts naming source identity, affected indices/control data, editor/reviewer, reason, timestamp, and predecessor digest. No UI or ad hoc memory edit may silently overwrite measured membership.

## 14. Evaluation and acceptance metrics

Evaluate against independently labelled full-resolution scenes when available, retaining matching policy and threshold version. Report tree-count error; instance precision/recall/F1; matched IoU; over/under-segmentation; `UNASSIGNED`/`REVIEW_REQUIRED` rates and clearance relevance; recovery coverage/scene completeness; runtime/memory/throughput; deterministic normalized-output digest equality; original-index consistency; and no synthetic clearance input. Stratify by density, crown overlap, tree size, terrain/HAG quality, and scan support when evidence permits. No scene-specific exception may improve metrics.

## 15. Controlled decision and implementation stages

| Stage | Deliverable and gate |
|---|---|
| Current | `DESIGN-FROZEN` after final independent design review and separate freeze approval. No implementation authorization. |
| V4-A0 | Design freeze and interface approval: record the frozen FRV consumer boundary and V4-owned recovery, coverage, result, safety, and segmentation interfaces. |
| Independent design re-review | Completed: independently reviewed the V4-A0 resolutions and final candidate design. |
| Separate approval | Completed on 2026-08-01: explicit approval changed the status to `DESIGN-FROZEN`. |
| V4-A1 | V4-owned enums/config/status, FRV consumer references, SourceIdentity/recovery/coverage/result manifests, and pure validators. It does not implement an FRV producer. |
| V4-A2 | Validate and consume frozen FRV artifact, coverage manifest, provider/override bindings, and three-layer conservation. It does not run or modify the FRV membership rule. |
| V4-A3 | Stream the source file to recover exact `original_index` and measured XYZ; validate source replay, range, order, uniqueness, and digests. |
| V4-A4 | Consume FRV HAG evidence/readiness and partition normal segmentation input from measured review hand-off. |
| V4-A5 | Produce and aggregate `V4RecoveryCoverageReport`, unrecovered-region hand-off, routing, and safety completeness. |
| V4-B1 | Implement robust CHM and treetop evidence. |
| V4-B2 | Implement synchronous two-dimensional density competition and overlapping XY claims. |
| V4-B3 | Implement candidate prisms, adaptive overlapping slices, and continuity evidence. |
| V4-B4 | Implement synchronous three-dimensional point assignment, three states, and ragged membership. |
| V4-B5 | Implement formal clearance requests, V2 lossy compatibility summary, and downstream-interface validation. |
| V4-C | Labelled-scene, workflow, performance, and determinism acceptance; may create implementation/baseline/release acceptance statuses, not the first design freeze. |

Every stage requires its own tests and independent review; a later stage may not
cross an earlier gate. Tests may use explicit test configuration. Before D10’s
formal calibration approval, no production-default numeric configuration
exists. Calibration may wait for implementation and labelled evaluation: CHM
cell size and estimator parameters; density representation; smoother and
trend-window length; decline/recovery thresholds; claim-score weights; conflict
margin; slice thickness/overlap defaults; and continuity parameters. The design
must freeze how each is calibrated, versioned, recorded, protected from
scene-specific exceptions, and approved after labelled-scene evaluation; it
need not guess their numerical values today.

## 16. Hard prohibitions

The first V4 implementation must not use DBSCAN-only segmentation, fixed-radius crowns, plain Voronoi final boundaries, direct watershed final output, deep learning, scene-specific special cases, or artificial clearance points. It must not use downsampled points as authoritative density/CHM data, sequentially grow trees or capture points by tree order, or silently drop unresolved recovered vegetation or unrecovered candidate regions. V2-row compatibility remains a lossy adapter, never formal recovery, membership, or clearance input.

## 17. Repository evidence used and open decisions

Repository evidence: `core/contracts/tree_inputs.py` defines a V3 input view aligned to `vegetation_points`; `core/contracts/vegetation_trees.py` and `core/contracts/tree_artifacts.py` define V3 result/artifact boundaries; `algorithms/vegetation/tree_inputs.py` constructs the V3 V2/A2-derived path; and `algorithms/io/chunks.py` provides `iter_las_chunks` with stable streamed `original_index`. Existing V3 CHM/treetop/canopy modules evidence segmentation concepts, not full-resolution FRV membership. The normative FRV design defines the upstream membership boundary, but no runtime FRV producer or artifact is claimed to exist.

**Resolved upstream boundary:** the separately frozen FRV design resolves the membership producer/rule, scope/inventory, and membership-artifact authority. V4 consumes it and retains its own source replay, recovery, HAG-readiness partition, coverage, safety hand-off, final tree assignment, and diagnostic boundaries.

**Calibration allowed during implementation/evaluation:** choose the general configuration values listed in Section 15 through labelled-scene evidence and recorded approval. The exact storage format/retention implementation for large diagnostics and the formal unresolved-clearance consumer/threshold policy also require appropriate artifact and safety-interface decisions, without inventing repository facts.

**Current gate decision:** the upstream membership design boundary is resolved
by the separately frozen FRV design. The final independent V4 design review
passed, and separate design-freeze approval was completed on 2026-08-01.
Design boundary resolved: **YES**. FRV implementation available: **NO**. V4
runtime input available: **NO**. V4 implementation authorized: **NO**. The
document is `DESIGN-FROZEN`; its only next-step authorization is the V4-A1
implementation plan, not V4-A1 code implementation.

**Final status:** `DESIGN-FROZEN`

## 18. V4-A0 Decision Records

### Evidence-classification convention

`REPOSITORY-PROVEN` means that the named repository files and tests establish
only the stated existing behaviour. `PROPOSED-NEW-BOUNDARY` means a decision
made by this design for V4; it is **not** an existing repository type,
producer, or consumer. `NOT-YET-PROVEN` means the repository does not provide
the required production or validation evidence. `REJECTED` records an option
that must not be used as a substitute for the selected boundary.

All proposed types in this section are **Proposed contract; not yet
implemented.** They neither amend V2/V3 contracts nor authorize V4-A1 code
implementation or any later implementation.

### V4-A0-D01 — Full-resolution vegetation-membership producer

- **Authority.** `docs/full-resolution-vegetation-membership-design.md`,
  `DESIGN-FROZEN`, freeze date `2026-08-01`, is the normative upstream design.
- **Selected decision.** Full-resolution vegetation membership is produced by
  the separately frozen FRV upstream module. V4 does not own or implement the
  membership rule. V4 consumes only a validated
  `FullResolutionVegetationMembershipArtifact` together with its
  `FRVScopeCoverageManifest`, `SourceIdentity`, configuration, provider,
  override, and provenance bindings.
- **Rejected alternatives.** V2/V3 label propagation, nearest-neighbour
  expansion, fixed-radius expansion, voxel inheritance, cluster propagation,
  prism/box admission, a V4-local temporary classifier, and an ambiguous
  “existing rule or successor” boundary are `REJECTED`. None supplies FRV's
  full-resolution decision record or may be used to create membership.
- **Consumer invariants.** V4 validates the frozen-FRV design/version
  compatibility and the complete binding set before source replay. It recovers
  only exact recorded `original_index` rows, never synthetic points; invalid or
  unavailable evidence produces a coverage/safety route, never a guessed set.
- **Structural design status.** `PASS` — resolved by frozen FRV design.
- **Runtime availability.** `NOT IMPLEMENTED` — FRV artifacts do not yet exist
  in the repository.
- **V4 execution gate.** `BLOCKED UNTIL VALID FRV ARTIFACTS EXIST`.
- **Review status.** `PASS` — `RESOLVED_BY_FROZEN_FRV_DESIGN`.

### V4-A0-D02 — Membership-evidence artifact schema

- **Selected decision.** This is the FRV membership-artifact consumer and
  validation boundary. The authoritative schemas are
  `FullResolutionVegetationMembershipArtifact` and
  `FRVScopeCoverageManifest` from frozen FRV. `V4ValidatedFrvMembershipView`
  is permitted only as a proposed read-only consumer view, not a replacement
  FRV schema.
- **Minimum validation.** Validate FRV design/version compatibility;
  `SourceIdentity`; `source_file_sha256`; optional
  `canonical_point_content_digest`; artifact/manifest/block digests;
  `configuration_digest`; `frv_scope_coverage_manifest_digest`;
  `frv_scope_union_id`; `scope_construction_version`; `scope_union_geometry_digest`;
  `frv_upstream_inventory_digest`; `frv_canonical_inventory_count`,
  `frv_required_inventory_count`, `frv_approved_hard_rejection_count`,
  `frv_scope_created_or_merged_count`, `frv_coverage_blocked_count`,
  membership-state counts, `frv_unique_in_scope_point_count`,
  `frv_evaluated_point_count`, and `frv_unasserted_in_scope_count`; provider
  and override digests; and block range/order/uniqueness/read-only invariants.
  Validate FRV fields `canonical_inventory_complete`,
  `required_inventory_complete`, `approved_hard_rejection_complete`,
  `scope_coverage_complete`, `routing_complete`, and
  `membership_decision_complete`, exposing them in the V4 consumer view only as
  `frv_canonical_inventory_complete`, `frv_required_inventory_complete`,
  `frv_approved_hard_rejection_complete`, `frv_scope_coverage_complete`,
  `frv_routing_complete`, and `frv_membership_decision_complete`.
  These are validated inherited FRV facts, not V4-recomputed completeness
  fields. Validate the FRV inventory/scope and five-terminal-state conservation
  equations, with `frv_unasserted_in_scope_count=0` and
  `frv_evaluated_point_count=frv_unique_in_scope_point_count`. Any optional
  upstream `candidate_confidence` is retained only when present,
  version-compatible, finite, and provenance-bound; its absence is not a
  membership failure and V4 neither requires nor recomputes it.
- **Required routing.** `VERIFIED_VEGETATION_CANDIDATE` and `MANUAL_INCLUDE`
  are eligible for exact measured-point recovery and then the HAG-readiness
  partition. `REVIEW_REQUIRED_MEMBERSHIP` is recovered when possible but never
  enters normal segmentation; it enters the unresolved measured-point safety
  hand-off. `EXCLUDED_NON_CANDIDATE` and `MANUAL_EXCLUDE` retain audit and
  provenance but do not enter segmentation or receive V4 reinterpretation. A
  coverage-blocked required region enters the unrecovered-region safety route;
  it creates no point, tree ID, HAG, or distance. Missing membership assertion
  never becomes an explicit exclusion.
- **Failure behavior and review status.** Any FRV validation failure blocks
  V4 recovery for the affected scope and is retained in D05. V4 may not turn
  an inherited false `frv_membership_decision_complete` true merely because
  visible scope rows have terminal states. `PASS` —
  `FRV_CONSUMER_BOUNDARY_ALIGNED`.

### V4-A0-D03 — Source content identity and path relocation

- **Issue.** Separate location from source content identity so artifacts remain
  auditable after a file move or copy.
- **Why it blocks `DESIGN-FROZEN`.** Indexes are meaningless if their source
  can be silently substituted.
- **Repository evidence inspected.** `PointCloudSource` and LAS header
  contracts retain `source_id`, path, counts, bounds, scale/offset; V3/A2A/A2B
  contracts retain source IDs, origins, and artifact SHA-256 values. I1 tests
  demonstrate relocation of a verified bundle. No existing contract proves a
  raw-source SHA-256-plus-relocation audit for V4.
- **Evidence classification.** Existing identity fields and artifact hashes are
  `REPOSITORY-PROVEN`; V4 `SourceIdentity` and its comparison rule are
  `PROPOSED-NEW-BOUNDARY`.
- **Considered alternatives.** Path-only identity; source-ID-only identity;
  content digest alone; and a digest plus independently checked source summary
  and coordinate context.
- **Selected decision.** `SourceIdentity` aligns with frozen FRV: it requires
  `source_id`, `source_file_sha256`, optional
  `canonical_point_content_digest`, point count, bounds, origin/context,
  coordinate frame and units, `recorded_source_path`, `resolved_access_path`,
  `relocation_audit`, and source-equivalence audit when applicable. The raw-file
  digest and canonical decoded-point digest have distinct semantics; paths are
  audit/access fields only.
- **Rejected alternatives and reasons.** Path-only and source-ID-only matching
  are `REJECTED` because neither establishes bytes or coordinate meaning.
  Digest-only matching is `REJECTED` because it does not expose a practical
  coordinate-context incompatibility.
- **Proposed contract or interface.** `relocation_audit` records recorded and
  resolved paths, Windows-normalized lexical forms, resolved final path after
  symlink/junction resolution where supported, resolution method, timestamp or
  run ID, host-independent identity result, and any path-warning flags.
  Comparison is case-insensitive only for Windows access-path equivalence; it
  never changes content comparison. Drive-letter changes and copying identical
  content to another computer are permitted after a successful relocation
  audit. The access path is never included in the content digest.
- **Required fields and invariants.** Both digests, when present, are lowercase
  hexadecimal; counts/bounds/origin are finite and coordinate context is
  explicit; every source in a multi-source run has its own identity; and
  source-ID/content disagreement is a hard error. Exact replay requires matching
  `source_file_sha256`; any re-encapsulation requires FRV-compatible
  source-equivalence audit. A changed path with every content field equal is a
  relocation, not a new source.
- **Failure behavior.** Digest, count, bounds, origin, frame, unit, or
  source-ID mismatch is `SOURCE_IDENTITY_FAILED`; no membership, HAG, or
  clearance artifact may be consumed for that source.
- **Safety consequence.** Prevents membership indexes and distance results
  from being replayed against a different point cloud.
- **Relationship to V2/V3.** V3 provenance can supply supporting evidence but
  cannot satisfy the new raw-source identity by itself.
- **Downstream implementation gate.** Every producer and consumer validates
  identical `SourceIdentity` before accepting indexes or coordinates.
- **Remaining validation evidence.** Test same-content copy, case-only path
  change, symlink/junction resolution, altered bytes, altered bounds, and
  multi-source aggregation.
- **Review status.** `PASS` — `FRV_SOURCE_IDENTITY_ALIGNED`.

### V4-A0-D04 — Full-resolution HAG authority

- **Selected decision.** V4 consumes and validates frozen-FRV HAG evidence; it
  does not define an alternative HAG provider or alter FRV HAG semantics. For
  every recovered row it retains `membership_state`, `hag_evidence_state`,
  `segmentation_hag_readiness`, optional `raw_hag`, optional `clamped_hag`, HAG
  quality/reason, and FRV provider/provenance references.
- **Partition.** Normal V4 segmentation contains only FRV states
  `VERIFIED_VEGETATION_CANDIDATE` or `MANUAL_INCLUDE` with
  `segmentation_hag_readiness=READY`. Membership and HAG readiness are
  independent: no HAG condition may change an FRV membership state.
- **Safety behavior.** `REVIEW_REQUIRED_MEMBERSHIP`, an admitted candidate with
  readiness other than `READY`, `HAG_MISSING`, `HAG_INVALID`,
  `HAG_OUT_OF_DOMAIN`, or any other HAG-review state remains a real measured
  point when exact recovery succeeds. It enters D06 with `original_index`,
  measured XYZ, membership state, HAG evidence/readiness, reason/quality, wire
  evidence, and FRV references; it never enters normal segmentation or receives
  a normal tree ID solely because a HAG substitute was invented.
- **Review status.** `PASS` — FRV membership and V4 segmentation-HAG readiness
  are explicitly separated.

### V4-A0-D05 — Recovery coverage report

- **Issue.** Freeze independent completeness axes so an FRV scope/coverage unit, a
  membership-review point, a HAG-not-ready recovered point, and a clearance
  request awaiting a result cannot be collapsed into one ambiguous “complete”
  flag.
- **Why it blocks `DESIGN-FROZEN`.** A failed recovery otherwise disappears
  between vegetation candidate detection and clearance review.
- **Repository evidence inspected.** V3 exposes row-level unresolved/review
  statuses and source-cluster readiness, but no full-resolution coverage report
  or source/region completeness aggregator exists.
- **Evidence classification.** V3 status concepts are `REPOSITORY-PROVEN`;
  `V4RecoveryCoverageReport` is a `PROPOSED-NEW-BOUNDARY`.
- **Considered alternatives.** Drop failures; represent them as synthetic
  points; overload point-level unresolved records; or use a distinct FRV-scope
  report.
- **Selected decision.** Use a distinct `V4RecoveryCoverageReport` derived from
  `FRVScopeCoverageManifest` plus FRV artifact validation, source
  replay/recovery outcomes, and V4 HAG/readiness partition outcomes. It has one
  record per FRV scope/coverage unit and a deterministic source/scenario
  aggregate; it does not recreate an independent candidate inventory. Its
  inherited FRV completeness fields are
  `frv_canonical_inventory_complete`, `frv_required_inventory_complete`,
  `frv_approved_hard_rejection_complete`, `frv_scope_coverage_complete`,
  `frv_routing_complete`, and `frv_membership_decision_complete`. Its V4
  downstream completeness fields are `source_identity_complete`,
  `xyz_recovery_complete`, `hag_ready_complete`,
  `segmentation_input_complete`, `v4_routing_complete`,
  `measured_clearance_complete`, and `review_required`.
- **Namespace and authority.**

  | Field | Authority | Meaning |
  |---|---|---|
  | `frv_routing_complete` | FRV | Whether FRV has completed frozen upstream routing for every canonical/required inventory record, scope, coverage failure, and membership evidence. |
  | `frv_membership_decision_complete` | FRV | Whether upstream membership decisions satisfy frozen FRV completeness. |
  | `source_identity_complete` | V4 | Whether every V4 source identity validates. |
  | `xyz_recovery_complete` | V4 | Whether required real points have exact XYZ recovery. |
  | `segmentation_input_complete` | V4 | Whether normal segmentation input is available. |
  | `v4_routing_complete` | V4 | Whether every known point or region has one safe V4 primary route. |
  | `measured_clearance_complete` | V4/clearance aggregate | Whether every required measured request has a valid result with no uncovered safety risk. |

  `frv_routing_complete != v4_routing_complete`: the former is an upstream
  prerequisite and evidence field, while the latter is a downstream
  routing-conservation result. Both combinations are legal. FRV evidence may be
  complete while a V4 recovery point is not safely routed
  (`frv_routing_complete=true`, `v4_routing_complete=false`). Conversely, V4
  may route an FRV blocked region to `V4UnrecoveredRegionHandOff`
  (`frv_routing_complete=false`, `v4_routing_complete=true`); in that case
  `segmentation_input_complete=false`, `measured_clearance_complete=false`, and
  V4 must not claim `frv_membership_decision_complete=true`.
- **Rejected alternatives and reasons.** Dropping and synthetic-point options
  are `REJECTED` because they respectively hide risk or invent measurements.
  Overloading a point record is `REJECTED` because no original point exists for
  an unrecovered region.
- **Proposed contract or interface.** Each record contains `source_identity`,
  `frv_scope_coverage_manifest_digest`, `frv_scope_union_id`,
  `frv_upstream_inventory_digest`, `frv_membership_artifact_digest`,
  recovered-index digest, recovered/review/blocked counts, point-recovery and
  HAG-state counts, request/result counts, `failure_reason_flags`, FRV evidence
  references, `safety_relevance`, `limitations`, and `diagnostic_reference`.
  It contains the six inherited validated FRV completeness fields
  `frv_canonical_inventory_complete`, `frv_required_inventory_complete`,
  `frv_approved_hard_rejection_complete`, `frv_scope_coverage_complete`,
  `frv_routing_complete`, and `frv_membership_decision_complete`, together with
  their `frv_`-prefixed supporting counts and conservation validation. It also
  contains the seven V4 downstream fields `source_identity_complete`,
  `xyz_recovery_complete`, `hag_ready_complete`,
  `segmentation_input_complete`, `v4_routing_complete`,
  `measured_clearance_complete`, and `review_required`.
  Failure states include `RECOVERY_BLOCKED`, `SOURCE_IDENTITY_FAILED`,
  `MEMBERSHIP_EVIDENCE_MISSING`, `MEMBERSHIP_ARTIFACT_INVALID`,
  `SOURCE_STREAM_INCOMPLETE`, `INDEX_COMPLETENESS_FAILED`,
  `HAG_EVIDENCE_MISSING`, `HAG_ARTIFACT_INVALID`, `NOT_EVALUATED`,
  `COVERAGE_BLOCKED`, `INVALID_WIRE_GEOMETRY`, `DISTANCE_FAILED`, and
  `UNKNOWN_SAFETY_RELEVANCE` as applicable to the relevant axis.
- **Required fields and invariants.** `source_identity_complete` means every
  relevant source identity validates. `xyz_recovery_complete` means every
  original point with candidate or membership-review evidence has exact index
  and XYZ recovered; HAG failure alone does not change it.
  `frv_membership_decision_complete` is the verified inherited FRV field, not a
  V4 replacement rule: it requires FRV field `membership_decision_complete`
  to be true, explicit scope, legal terminal state for every scope row, zero
  unasserted-in-scope count, passed required FRV conservation, valid
  artifact/manifest/provider/stream/integrity bindings, and no affected required
  coverage-blocked region. `hag_ready_complete` means every
  segmentation-planned candidate has valid HAG.
  `segmentation_input_complete` requires validated FRV compatibility,
  `frv_membership_decision_complete`, `source_identity_complete`,
  `xyz_recovery_complete`, successful HAG/readiness partition, and no affected
  blocking failure for the segmentation population.
  `v4_routing_complete` means every known record has exactly one mutually
  exclusive primary safety route: normal segmentation, measured-point hand-off,
  unrecovered-region hand-off, or explicit excluded/audit-only route. It is not
  `measured_clearance_complete`.
  Separately, every policy-required real measured point has exactly one D08
  request and exactly one result or explicit `PENDING`/`NOT_EVALUATED` outcome;
  a point may have both a D06 hand-off and D08 request. A region generates no
  fictional request. `measured_clearance_complete` additionally requires a
  valid measured result for every required real-point request and no
  safety-relevant uncovered region, invalid wire geometry, distance failure,
  source identity failure, or unknown safety relevance.
- **Failure behavior.** Source-identity failure blocks that source and makes
  measured clearance incomplete. A region-local membership/HAG failure may
  allow unrelated segmentation to continue. A HAG-missing but XYZ-recovered
  point leaves `xyz_recovery_complete` true but makes HAG/segmentation input
  incomplete and requires routing. A pending clearance request may make
  `v4_routing_complete` true but always keeps measured clearance incomplete.
- **Safety consequence.** An unrecovered region may not be declared harmless
  merely because it is not near a wire or lacks a tree ID; without authoritative
  geometry its safety relevance remains `UNKNOWN`.
- **Relationship to V2/V3.** It is an additive V4 companion report; it does
  not reinterpret V3 unresolved status.
- **Downstream implementation gate.** Segmentation processes only its
  segmentation-input-complete, HAG-ready population; final reporting aggregates
  every region and every known review point across all axes.
- **Remaining validation evidence.** Verify single/multi-region and
  single/multi-source aggregation, count mismatch, missing stream chunks, and
  a recovery failure that remains visible in final output.
- **Review status.** `PASS` —
  `FRV_COMPLETENESS_AND_ROUTING_CONSERVATION_ALIGNED`.

### V4-A0-D06 — Unresolved-vegetation safety hand-off

- **Issue.** Keep every known recovered real point that cannot receive a normal
  tree assignment, and every unrecovered region, separate, visible, and safe
  for future consumers.
- **Why it blocks `DESIGN-FROZEN`.** Neither kind may vanish because it has no
  formal tree assignment.
- **Repository evidence inspected.** V3 has `UNRESOLVED`/review semantics and
  readiness flags, but no consumer contract for full-resolution unresolved
  vegetation or failed recovery regions.
- **Evidence classification.** Existing V3 concepts are `REPOSITORY-PROVEN`;
  both proposed V4 hand-offs are `PROPOSED-NEW-BOUNDARY`.
- **Considered alternatives.** One union type for both conditions; discard
  non-tree data; or two semantically disjoint hand-offs.
- **Selected decision.** Use two disjoint structures:
  `V4UnresolvedMeasuredPointHandOff` for real recovered points and
  `V4UnrecoveredRegionHandOff` for FRV/D05 coverage records without a
  recoverable original point. The measured-point structure covers (A) FRV
  `REVIEW_REQUIRED_MEMBERSHIP`; (B) FRV-admitted candidates with HAG/readiness
  review; (C) confirmed vegetation that is `UNASSIGNED`; and (D) confirmed
  vegetation that is `REVIEW_REQUIRED` after final V4 assignment. Cases (A)
  and (B), and any recovery/provider/FRV validation-prevented final comparison,
  have `tree_assignment_state=NOT_EVALUATED`.
- **Rejected alternatives and reasons.** A union that erases whether a real
  point exists is `REJECTED`; so are invented original points and discarded
  no-tree data.
- **Proposed contract or interface.** A measured-point record requires
  `source_identity`, `original_index`, verified measured XYZ or a verified
  retrieval reference, FRV `membership_state`, `hag_evidence_state`,
  `segmentation_hag_readiness`, optional raw/clamped HAG, HAG quality/reason,
  wire evidence, FRV artifact/manifest/provider/override references,
  `tree_assignment_state` (`NOT_EVALUATED`, `UNASSIGNED`, or
  `REVIEW_REQUIRED` for hand-off records), reason flags, priority,
  `candidate_tree_ids` where
  applicable, `assignment_confidence` where applicable, and
  `diagnostic_reference`. It may produce a D08 measured-clearance request using
  XYZ even if HAG is missing. A region record requires
  `coverage_report_reference`,
  `frv_scope_union_id`/coverage reference, `frv_upstream_inventory_digest`,
  verified bounds or scope-union geometry digest, failure reason,
  `safety_relevance` (`KNOWN` or `UNKNOWN`),
  `review_required=true`, and `measured_clearance_complete=false`.
- **Required fields and invariants.** Measured records name real unique source
  rows and may use measured-point distance only; missing HAG never prevents
  their safety hand-off. `NOT_EVALUATED` means that final B4 comparison was
  never entered; it is not `UNASSIGNED` and not a replacement for
  `REVIEW_REQUIRED`. `UNASSIGNED` and `REVIEW_REQUIRED` apply only after actual
  B4 entry; no `NOT_EVALUATED` record has a normal tree ID. A membership-review
  point is explicitly reported as
  membership review, not as confirmed non-vegetation or a tree member. Region
  records contain no fictitious point, HAG, tree ID, or clearance distance. The
  two sets are disjoint and are retained even while no downstream consumer
  exists. Every recovered FRV-review point and every FRV-admitted point that is
  not normal `ASSIGNED` enters the appropriate measured review hand-off exactly
  once; every unrecovered region enters the region hand-off exactly once. This
  is primary-route conservation, separate from D08 request/result conservation:
  each policy-required real measured point has one request and one result or
  explicit pending/not-evaluated outcome, while a region creates none.
- **Failure behavior.** A consumer must return an explicit scenario outcome:
  `FORMAL_MEASURED_CLEARANCE_AVAILABLE`, `UNRESOLVED_MEASURED_REVIEW`,
  `MEMBERSHIP_REVIEW`, `HAG_REVIEW`, `COVERAGE_BLOCKED_REVIEW`, or
  `NOT_EVALUATED`; it may not silently suppress a record. Threshold policy is
  deferred to a later safety-interface review.
- **Safety consequence.** Formal distances are never made up for uncovered
  regions, while real unresolved points remain eligible for measured review.
- **Relationship to V2/V3.** New V4-only outputs; no V2/V3 status or public
  contract changes.
- **Downstream implementation gate.** Final V4 output validation reconciles
  every non-normal-assigned known recovered point and every non-recovered
  coverage record to exactly one disjoint hand-off, then reconciles each
  required measured-point hand-off to one D08 request.
- **Remaining validation evidence.** Demonstrate conservation of records,
  exact original-index retrieval, no false distances, and consumer status
  propagation.
- **Review status.** `PASS` —
  `PRESEGMENTATION_STATE_AND_HANDOFF_ALIGNED`.

### V4-A0-D07 — Final three-dimensional point assignment contract

- **Issue.** Freeze the state machine and deterministic evidence interface for
  final tree membership without inventing uncalibrated weights.
- **Why it blocks `DESIGN-FROZEN`.** V4 must know which stage creates formal
  tree membership and how ambiguity remains visible.
- **Repository evidence inspected.** V3 has immutable point arrays and
  ragged, strictly ordered member-row offsets, but its construction maps V2
  rows through canopy-grid labels and produces provisional records. It does not
  provide simultaneous full-resolution 3-D claims.
- **Evidence classification.** Array/ragged validation is
  `REPOSITORY-PROVEN`; V4 final claims and states are
  `PROPOSED-NEW-BOUNDARY`.
- **Considered alternatives.** First-tree capture; retain only an XY winner;
  confidence-only assignment; or simultaneous all-candidate comparison with a
  three-state result.
- **Selected decision.** Use simultaneous all-candidate comparison. For every
  recovered point, retain every valid candidate-tree claim, including a
  measured `SECONDARY_CLAIM`; calculate claims in a deterministic order
  independent of scheduling; then emit exactly one `ASSIGNED`, `UNASSIGNED`,
  or `REVIEW_REQUIRED` state.
- **Rejected alternatives and reasons.** First-tree capture, XY-only winner,
  confidence-only state, envelope-created membership, and continuity-prior-only
  membership are `REJECTED` because none establishes a measured 3-D assignment.
- **Proposed contract or interface.** A point-tree claim contains `xy_prior`,
  `slice_support`, `adjacent_slice_continuity`, `seed_support`,
  `height_support`, `density_support`, `structure_continuity`,
  `scan_support_quality`, `occlusion_gap_evidence`, `claim_score`,
  `claim_rule_version`, and `reason_flags`. Claim construction evaluates all
  trees whose valid candidate prism contains the point; no prism creates a
  member. Stable ordering is `(original_index, candidate_tree_id)` for
  evaluation and `(claim_score descending, candidate_tree_id ascending)` for
  reporting. Versioned rules use the best/second-best relation but defer numeric
  margins to calibration.
- **Required fields and invariants.** `ASSIGNED` requires enough measured
  evidence plus a versioned clear advantage. `UNASSIGNED` means the point
  entered B4, candidate generation and full comparison completed without
  contract/integrity error, and the candidate-tree set is empty or contains no
  tree with sufficient positive evidence and no mandatory conflict review.
  `REVIEW_REQUIRED` covers strong conflict, quality/input anomaly, invalid
  evidence, override/review request, or a candidate-tree set made empty by
  incomplete/failed prism generation, corrupt or missing slice evidence,
  incomplete claim construction, non-finite evidence, or comparison that did
  not cover all valid trees. Such a failure may instead route to D05 validation
  or coverage failure according to its scope. `NOT_EVALUATED` is outside this
  three-state result and belongs to D06 when the point never entered B4.
  Confidence cannot
  replace state evidence. Ragged member indexes exactly equal `ASSIGNED`
  original indexes; no index occurs twice; zero-member seeds produce only
  diagnostic/suppression/review records; manual changes require a versioned
  override artifact.
- **Failure behavior.** A legally empty candidate-tree set after complete,
  valid candidate generation and comparison yields `UNASSIGNED`. An empty set
  caused by required prism-generation failure, missing/corrupt slice evidence,
  incomplete claim construction, input anomaly, non-finite evidence,
  incomplete valid-tree comparison, or contract/integrity failure yields
  `REVIEW_REQUIRED` or D05 validation/coverage failure according to scope; it
  cannot become `ASSIGNED`.
- **Safety consequence.** Ambiguous vegetation remains in the unresolved
  safety path instead of being consumed by the first plausible tree.
- **Relationship to V2/V3.** V3 remains its existing row-aligned/provisional
  system; its membership offsets are a design precedent only.
- **Downstream implementation gate.** Formal tree records use only `ASSIGNED`
  original points. Every other known real point requires D06 and may still
  create a D08 measured-clearance request without receiving a tree ID.
- **Remaining validation evidence.** Test schedule invariance, retained
  secondary claims, ties, empty seeds, overlapping crowns, and exact equality
  between state arrays and ragged membership.
- **Review status.** `PASS` —
  `ZERO_CANDIDATE_SET_STATE_MACHINE_ALIGNED`.

### V4-A0-D08 — Formal clearance request/result boundary

- **Issue.** Separate V4’s verified measured-vegetation clearance input from
  the clearance consumer’s nearest-primitive calculation result, without
  conflating either with derived geometry.
- **Why it blocks `DESIGN-FROZEN`.** A final distance is safety relevant and
  cannot rely on samples, envelopes, or candidate volumes by default.
- **Repository evidence inspected.** Existing wire contracts/I1 artifacts have
  wire IDs, segment identities, method/status/readiness fields, and sampled
  curve arrays. V3 tree contracts expose readiness but no V4 full-resolution
  clearance result or point-to-authoritative-primitive interface.
- **Evidence classification.** Existing wire artifact identity is
  `REPOSITORY-PROVEN`; the V4 clearance hand-off is a
  `PROPOSED-NEW-BOUNDARY`.
- **Considered alternatives.** Distance to sampled wire points; distance to
  envelope/CHM/prism geometry; exact/approved-bound distance to an authoritative
  wire primitive.
- **Selected decision.** V4 produces `V4MeasuredClearanceRequest`; the
  clearance module produces `MeasuredClearanceResult`; the scenario aggregator
  conserves requests/results to compute routing and measured-clearance
  completeness. Formal measured clearance is only `original measured vegetation
  point -> approved wire centreline segment, curve, or other authoritative
  primitive -> exact nearest distance`, or a method with an explicitly approved
  numerical error bound.
- **Rejected alternatives and reasons.** Unqualified discrete samples,
  envelopes, CHM/slice cells, candidate prisms, synthetic points, and
  conservative/predicted distances are `REJECTED` as formal measured distance.
  A conservative estimate may be separately reported under a different type
  and metric.
- **Proposed contract or interface.** `V4MeasuredClearanceRequest` requires
  vegetation `source_identity`, `original_index`, verified measured XYZ or an
  exact retrieval reference, tree ID or unresolved/membership/HAG-review state,
  vegetation reason flags, wire artifact identity, wire source identity, wire
  instance ID, wire geometry version, and request status. It must not contain
  a nearest primitive ID, curve parameter, `distance_m`, or final error bound.
  `MeasuredClearanceResult` requires request identity, evaluation status,
  nearest wire primitive ID, curve parameter/segment position, distance method
  version, `distance_m`, numerical tolerance/error bound, and failure/review
  reason.
- **Required fields and invariants.** Both source identities validate before
  geometry use; measured XYZ are retrieved by exact source index; primitive is
  approved and valid for distance; finite non-negative result distance and
  bound are required; and each request has exactly one result or an explicit
  `PENDING`/`NOT_EVALUATED` status. Formal measured, conservative, and
  predicted values occupy separate named fields and reports.
- **Failure behavior.** No approved primitive, invalid geometry identity,
  unavailable exact/bounded method, numerical validation failure,
  `NOT_EVALUATED`, or a pending request keeps
  `measured_clearance_complete=false`. It does not erase routing completeness
  when the request itself has been safely routed.
- **Safety consequence.** Prevents derived geometry or a lower of incomparable
  distances from being presented as an actual measured clearance.
- **Relationship to V2/V3.** No existing tree/wire contract is changed; V4 is
  a new producer-side hand-off only.
- **Downstream implementation gate.** V4 validates request identity and source
  XYZ; the clearance consumer validates wire geometry, selects the primitive,
  and returns a result; the aggregator validates one-request/one-result-or-
  explicit-pending conservation before a formal-distance claim.
- **Remaining validation evidence.** Test piecewise-linear/curve primitives,
  rejected sample-only geometry, error-bound approval, and identity mismatch.
- **Review status.** `PASS` — request and result remain separate consumer boundaries.

### V4-A0-D09 — Diagnostic evidence retention

- **Issue.** Preserve enough evidence to audit a final state without retaining
  every intermediate array permanently in memory.
- **Why it blocks `DESIGN-FROZEN`.** Safety decisions must be reproducible and
  recovery failures must remain inspectable on large point clouds.
- **Repository evidence inspected.** Existing artifact loaders use manifests,
  SHA-256 verification, strict inventories, compressed NPZ, immutable arrays,
  and corruption tests. They do not define V4’s long-term diagnostic classes.
- **Evidence classification.** Artifact integrity patterns are
  `REPOSITORY-PROVEN`; V4 retention classes are
  `PROPOSED-NEW-BOUNDARY`.
- **Considered alternatives.** Retain everything resident; retain only a final
  digest; or use explicit retention classes with immutable replay inputs.
- **Selected decision.** Use three retention classes: (A) long-lived audit
  evidence, (B) reproducible-from-immutable-source/configuration evidence, and
  (C) chunk-compressed or digest-referenced evidence. All artefacts have a
  manifest, artifact digest, and per-chunk digest where chunked; compression
  may not alter numeric semantics.
- **Rejected alternatives and reasons.** Permanent in-memory retention is
  `REJECTED` under the 16 GB constraint. A final digest alone is `REJECTED`
  because it cannot explain a point state or recovery failure.
- **Proposed contract or interface.** Class A retains source identity, FRV
  membership-artifact/coverage-manifest/provider/override/HAG/configuration
  digests, recovery index digest/count, coverage reports, final point
  state/tree/reason, final or replayable claim evidence,
  ragged-membership digest, unresolved hand-offs, and formal-clearance method
  identity. Class B may include complete CHM grids, density sequences,
  temporary candidate graphs, and slice work arrays only when their immutable
  inputs, algorithm/configuration versions, and deterministic replay procedure
  are retained. Class C includes chunked blocks, compressed diagnostics, and
  summary-referenced values needed only after manifest verification.
- **Required fields and invariants.** Every retained item names retention class,
  schema/version, content digest, source/configuration binding, and deletion
  eligibility. A cleanup policy may delete only Class B/C content after a
  verified replacement/reference exists; it must never invalidate Class A,
  final-state audit, unresolved hand-off, or coverage decision.
- **Failure behavior.** Missing Class A or a mismatched manifest/digest makes
  the affected final output non-auditable and routes it to review; absent Class
  B data is replayed only from verified immutable dependencies, otherwise it is
  treated as unavailable.
- **Safety consequence.** Retention pressure cannot erase evidence that a
  region was unrecovered or a point was unresolved.
- **Relationship to V2/V3.** Reuses no V2/V3 retention contract; it follows
  existing repository integrity practices without changing them.
- **Downstream implementation gate.** Implementations must publish a manifest
  and pass a replay/audit test before diagnostic compaction is enabled.
- **Remaining validation evidence.** Test large chunked artifacts, hash
  corruption, lossless compression semantics, replay from Class B, and cleanup
  protection of Class A.
- **Review status.** `PASS` — retention references FRV and V4 audit evidence.

### V4-A0-D10 — Calibration governance and prohibition of scene special cases

- **Issue.** Freeze how numerical values are calibrated without guessing them
  now or introducing file/scenario-specific branches.
- **Why it blocks `DESIGN-FROZEN`.** Uncontrolled calibration would make
  membership and safety output irreproducible.
- **Repository evidence inspected.** Existing preprocessing, ground, canopy,
  and tree modules use typed configuration/version fields and tests; repository
  evidence does not establish a V4 labelled-scene calibration protocol.
- **Evidence classification.** Versioned configuration practice is
  `REPOSITORY-PROVEN`; V4 governance is a `PROPOSED-NEW-BOUNDARY`.
- **Considered alternatives.** Fix values now; tune ad hoc per scene; or use a
  labelled calibration protocol with versioned general configuration.
- **Selected decision.** All CHM cell/estimator settings, density/kernel,
  smoothing window, decline/recovery thresholds, claim weights/conflict margin,
  slice thickness/overlap, and continuity thresholds belong to an immutable
  versioned configuration with a configuration digest. Values are selected only
  through a labelled calibration set and an independently held acceptance set,
  followed by recorded approval.
- **Rejected alternatives and reasons.** Hard-coding unvalidated values now and
  branching on filename, scene name, source ID, or test fixture are `REJECTED`.
  No configuration may contain a hidden scene selector.
- **Proposed contract or interface.** A proposed calibration record contains
  configuration version/digest, code/algorithm version, calibration dataset
  identities, held-out acceptance identities, annotation/matching policy,
  stratified metrics, approver, date, and superseded configuration digest. A
  configuration change creates a new digest and requires regression assessment
  against the same acceptance protocol.
- **Required fields and invariants.** Calibration and acceptance scenes are
  disjoint; results are stratified by density, crown overlap, tree size,
  terrain/HAG quality, and scan support; configuration lookup is independent of
  source name; and each final artifact records its exact configuration digest.
- **Failure behavior.** Missing/unknown configuration, overlap between
  calibration and acceptance sets, unapproved change, or scene-special-case
  detection rejects the run/configuration rather than falling back to defaults.
- **Safety consequence.** Prevents a superficially good single-scene result
  from masking uncontrolled membership or clearance behaviour.
- **Relationship to V2/V3.** This adds V4 governance only and does not alter
  existing configuration contracts.
- **Downstream implementation gate.** After separate V4-A1 code implementation
  authorization, V4-A1 may implement configurable mechanics, but no
  production/default numeric configuration is approved until the calibration
  record is accepted.
- **Remaining validation evidence.** Supply labelled-scene splits, stratified
  metrics, deterministic digest checks, special-case detection, and
  re-evaluation evidence for each changed configuration.
- **Review status.** `PASS` — versioned calibration and no scene special cases.

### V4-A0 gate checklist

| Structural decision | Decision | Gate result |
|---|---|---|
| Membership producer/rule boundary | D01 | `PASS` — `RESOLVED_BY_FROZEN_FRV_DESIGN`; runtime FRV artifacts remain unavailable. |
| FRV artifact consumer and conservation boundary | D02 | `PASS` — `FRV_CONSUMER_BOUNDARY_ALIGNED`; validated inherited FRV consumer fields are explicitly `frv_`-namespaced without a V4 replacement schema. |
| Source identity/path migration | D03 | `PASS` — FRV-compatible file/content digest distinction and relocation audit are frozen. |
| XYZ recovery versus HAG readiness | D04 | `PASS` — FRV membership, XYZ recovery, and V4 segmentation-HAG readiness are distinct. |
| Independent completeness axes | D05 | `PASS` — `FRV_COMPLETENESS_AND_ROUTING_CONSERVATION_ALIGNED`; inherited `frv_` completeness and V4 downstream fields, including `v4_routing_complete`, have separate semantics and conservation. |
| All known measured-point safety hand-off | D06 | `PASS` — `PRESEGMENTATION_STATE_AND_HANDOFF_ALIGNED`; `NOT_EVALUATED` is distinct from B4 states and real-point/region hand-offs are disjoint. |
| Final 3-D assignment | D07 | `PASS` — `ZERO_CANDIDATE_SET_STATE_MACHINE_ALIGNED`; legal empty candidate sets are deterministic and failed generation/comparison remains review/validation. |
| Clearance request/result split | D08 | `PASS` — V4 request, consumer result, and aggregate conservation are separated. |
| Diagnostic retention | D09 | `PASS` — proposed audit classes remain frozen. |
| Calibration governance | D10 | `PASS` — versioned no-special-case protocol remains frozen. |

All ten structural gates passed final independent design review. The separate
design-freeze approval is complete, and the V4 design is `DESIGN-FROZEN`. The
frozen FRV design resolves D01/D02 without claiming that the repository
contains a producer, artifact, consumer, or runtime HAG validation. FRV
implementation and V4 runtime artifacts remain unavailable. No implementation
is authorized. No proposed boundary in this section authorizes V4-A1 code
implementation.

## 19. Design freeze approval and change control

### 19.1 Freeze decision

```text
Freeze decision: APPROVED
Frozen status: DESIGN-FROZEN
Freeze date: 2026-08-01
D01-D10 structural gates: PASS
Structural design blockers: NONE
Implementation authorized: NO
```

The approval is based on multiple targeted V4 design revisions; the independent
freeze of the normative upstream FRV design; D01/D02 being resolved by the
frozen FRV boundary; structural review passing D03-D10; the final independent
V4 design review result `PASS_READY_FOR_FREEZE_APPROVAL`; separated FRV and V4
completeness namespaces; no structural `DESIGN-FREEZE-BLOCKED` item; and the
project lead's approval to perform this independent freeze operation. This
records a design decision only: it does not claim implemented code, contracts,
producers, artifacts, or runtime flows.

### 19.2 Frozen design scope

This freeze includes the following design decisions:

- V4 and FRV responsibility boundary; frozen FRV artifact/manifest consumer boundary.
- `SourceIdentity`, source replay, and exact `original_index` and real measured XYZ recovery.
- Membership versus segmentation-HAG readiness; `V4RecoveredVegetationSet` `N`, `N=0`, and index-set semantics.
- FRV `frv_*` completeness namespace; V4 `v4_routing_complete` and downstream completeness semantics; and `V4RecoveryCoverageReport`.
- Measured-point and unrecovered-region safety hand-offs; `NOT_EVALUATED`, `UNASSIGNED`, `REVIEW_REQUIRED`, and `ASSIGNED` state boundaries.
- Robust CHM and treetop evidence; synchronous XY density competition; revisable, overlapping XY claims; `IrregularXYRegion`; and `CandidatePrism`.
- Adaptive overlapping slices; synchronous three-dimensional original-point assignment; and ragged measured-point tree membership.
- The separation of `V4MeasuredClearanceRequest` from `MeasuredClearanceResult`; formal clearance based only on real measured points and approved wire geometry; and diagnostic evidence retention.
- Parameter calibration governance, prohibition of scenario special cases, and the V4-A1 through V4-A5, V4-B1 through V4-B5, and V4-C stage boundaries.

### 19.3 Not frozen as implemented

The following are not implemented: V4 Python enums, config, and contracts;
`V4ValidatedFrvMembershipView`; FRV artifact consumer/validator;
`SourceIdentity` validator; source replay and exact-recovery producer;
`V4RecoveredVegetationSet` producer; `V4RecoveryCoverageReport` producer;
safety hand-off contracts and producers; CHM producer; seed detector; XY
density/competition implementation; `IrregularXYRegion` producer;
`CandidatePrism` producer; slice/continuity implementation; three-dimensional
point assignment; `V4TreeSegmentationResult` producer; clearance-request
producer; clearance-result consumer; V2 lossy compatibility adapter; diagnostic
artifact publisher; labelled data; parameter calibration; performance testing;
and 16 GB memory acceptance.

### 19.4 Post-freeze change control

Any semantic change to this frozen design must reopen it through:

```text
DESIGN-FROZEN
→ DESIGN-REVIEW-CANDIDATE
→ targeted revision
→ independent design review
→ new freeze approval
```

Reopening is required for changes to the FRV/V4 responsibility boundary; V4
re-evaluation of vegetation membership; V1/V2/V3 label-propagation recovery;
the FRV artifact-consumer, `SourceIdentity`, digest, `N`, `N=0`, recovered-index,
membership/HAG-readiness, FRV/V4 completeness, `frv_routing_complete`, or
`v4_routing_complete` semantics; safety primary-route conservation; independent
D06 or D08 conservation; or removal of `NOT_EVALUATED` or changes to
`UNASSIGNED`, `REVIEW_REQUIRED`, or `ASSIGNED`.

Reopening is also required if XY claims, prisms, slices, or envelopes can
directly create formal tree members; final three-dimensional synchronous
comparison becomes sequential capture; DBSCAN-only, fixed-radius crown, plain
Voronoi, or watershed becomes the final segmentation; downsampled points become
formal full-resolution results; synthetic points become formal clearance input;
measured, predicted, and conservative clearance are merged; scenario name,
file name, source ID, device, or test-example special cases are introduced;
deep learning replaces the first frozen route; or the V4-A1 through V4-C stage
responsibility boundaries change.

### 19.5 Controlled calibration without reopening design

V4-C may calibrate, without reopening the design, CHM cell size; CHM height
estimator, outlier/support, seed-neighbourhood, and plateau-suppression
parameters; density-grid or kernel parameters; robust smoother; trend-window
length; decline/recovery thresholds; claim-score weights and conflict margin;
slice thickness and overlap; continuity thresholds; tile/chunk size; memory and
performance parameters; and diagnostic block size.

Each such calibration must be versioned, produce a new configuration digest,
use a calibration set and independent acceptance set, and retain its calibration
record. It must not use scenario special cases, change the state machine, lower
safety routing, or change the formal measured-clearance boundary. A calibration
change that alters algorithm structure, state semantics, or safety behaviour
must reopen the design.

### 19.6 Post-freeze implementation gate

`DESIGN-FROZEN` authorizes only the preparation of the `V4-A1 implementation
plan`.

```text
V4-A1 implementation plan authorized: YES
V4-A1 code implementation authorized: NO
```

Before any V4 code begins, the project must write the detailed V4-A1
implementation plan, independently review it, explicitly approve V4-A1 code
implementation, assign the work to an implementation tool, complete testing,
and perform independent code review. FRV runtime artifacts are still
unavailable, and V4 runtime execution remains blocked until valid FRV artifacts
exist. That runtime dependency does not prevent writing the V4-A1 plan or
designing V4-owned contracts and validators.

### 19.7 Relationship to FRV implementation

The FRV design is frozen, but FRV code is not implemented. This V4 design freeze
does not authorize FRV implementation or establish that an FRV artifact exists.
V4-A2 and later runtime stages depend on valid FRV artifacts. FRV and V4
implementation plans require separate review and authorization, and V4 must not
temporarily create a substitute FRV producer.
