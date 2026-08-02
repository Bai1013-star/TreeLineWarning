# Full-Resolution Vegetation Membership Evidence Design

**Canonical path:** `docs/full-resolution-vegetation-membership-design.md`
**Status:** `DESIGN-FROZEN`
**Authority:** Proposed upstream vegetation-candidate design only. **Implementation authorized: NO.** This document creates no code, contract, producer, dependency, configuration, or V4 change.

## 1. Objective, scope, and preserved boundaries

FRV (Full-Resolution Vegetation Membership Evidence) is the proposed upstream vegetation-candidate module between existing candidate evidence and V4 individual-tree segmentation. It determines, for every complete-resolution original point inside a declared decision scope, whether evidence supports vegetation-candidate membership, explicitly excludes it, or requires review.

FRV processes complete-resolution, real raw-source points directly. It does not perform single-tree segmentation, assign tree IDs, calculate clearance, predict growth, calculate warnings, or create synthetic points. It does not propagate V1/V2 labels. In particular, it forbids representative-voxel inheritance, nearest-neighbour expansion, fixed-radius expansion, prisms, bounding volumes, V2-cluster propagation, and interpolation. HAG, geometry, and wire evidence are re-established at full resolution.

V1 and V2 may provide upstream candidate evidence and scope context only. An A1 representative, its voxel, a V1 disposition, an A2A/A2B value, an A4 relation, or a V2 instance is never original-point vegetation membership proof. A1 voxel geometry only defines processing scope; it does not transfer vegetation membership.

The first production scope is `EXHAUSTIVE_REGION`. It makes a complete terminal decision for every original row in the final decision union, while rows outside that union are `OUT_OF_SCOPE`, not excluded. `EXHAUSTIVE_SOURCE` is a later, separately approved extension. `SPARSE_ASSERTIONS_ONLY` is supplemental evidence and never a production-completeness artifact.

All named artifacts, providers, graphs, resolvers, and schemas below are **Proposed contract; not yet implemented.** Nothing in this design claims that the repository already contains an exact FRV scope producer, `FRVScopeCoverageManifest`, a full-resolution wire provider, a structural-support graph, an override-chain resolver, or a membership-artifact producer.

## 2. Source identity and full-resolution input authority

`SourceIdentity` is a proposed, immutable binding containing `source_id`, raw LAS/LAZ file SHA-256, point count, bounds, coordinate frame and units, origin/context, recorded path, resolved path, and relocation audit. Paths are provenance only. A source-ID, digest, count, bounds, frame, unit, origin, or context mismatch is a coverage failure. File-byte identity and decoded point-content identity are distinct:

- `source_file_sha256` is SHA-256 over the raw source bytes.
- `canonical_point_content_digest`, when retained, is calculated over ordered decoded original rows with a frozen field inventory, dtype, byte order, quantization, and `original_index` ordering.

Exact first-version replay requires matching file bytes. A recompressed or re-encapsulated source requires a separately approved source-equivalence audit; equal point count alone is not equivalence.

Raw processing uses exact `original_index` and original XYZ. Nonfinite XYZ, unreplayable indexes, incomplete stream coverage, missing mandatory identity data, or duplicate formal output is a source/region coverage failure, never an ordinary exclusion.

### 2.1 HAG, geometry, scan, and wire inputs

The proposed `FullResolutionHagProvider` evaluates a frozen, provenance-digested ground-surface model at every original XYZ. It emits ground Z, `raw_hag`, optional `clamped_hag`, `hag_evidence_state`, and quality/reason fields. HAG states include `HAG_VALID_POSITIVE`, `HAG_VALID_NEAR_ZERO`, `HAG_WITHIN_NEGATIVE_TOLERANCE`, `HAG_MISSING`, `HAG_INVALID`, and `HAG_OUT_OF_DOMAIN`. Negative values may clamp only within a versioned tolerance and only while retaining raw value and reason. Missing, invalid, or out-of-domain HAG normally requires review; HAG readiness and membership are separate.

Full-resolution geometry and scan evidence includes deterministic multi-scale local PCA, local spacing, occupancy/density, neighbour sufficiency, column/branch structure, feature quality, and available scan/return attributes. It is built from complete-resolution points with frozen physical/adaptive scales and distance-then-`original_index` ties. Raw neighbour count, centreline distance, a single linearity threshold, or a single scattering threshold is never sufficient positive proof. Positive geometry must cover either canopy/branching-compatible structure or trunk/woody-column-compatible structure, with corroborating evidence.

Wire relation is computed only against identity-bound, authoritative wire geometry, when required by the policy in section 7. Existing A4 rows and discrete samples are not a full-resolution provider. In the first version, valid point-level relation may distinguish only `WIRE_BODY_POSSIBLE`, `WIRE_PROXIMITY_ONLY`, and `WIRE_VEGETATION_CONFLICT`; it cannot produce `WIRE_BODY_CONFIRMED`. Provider/output state separately distinguishes `WIRE_RELATION_VALID`, `WIRE_NOT_APPLICABLE_PROVEN`, `WIRE_PROVIDER_MISSING`, `WIRE_PROVIDER_INVALID`, `WIRE_IDENTITY_MISMATCH`, and `WIRE_GEOMETRY_UNAVAILABLE`.

## 3. Frozen first-version scope construction and coverage closure

### 3.1 Canonical and required upstream inventories

For each source, FRV constructs a **canonical upstream inventory** containing every discovered upstream candidate-related record, including:

- V2 `CONFIRMED`, `PROVISIONAL`, and `REVIEW_REQUIRED` records;
- V2 `REJECTED_AS_INSTANCE` records when their V1 candidate evidence remains;
- V1 candidate rows without a valid V2 disposition;
- candidate-region construction failures;
- source-identity or region-mapping failures; and
- records excluded by a trusted upstream hard non-vegetation rule.

The **required FRV inventory** is the subset of canonical inventory that is not an approved upstream hard-rejection record. Thus every record that retains vegetation-candidate evidence and has not been safely excluded by an approved, trusted upstream hard non-vegetation rule is required, including ordinary V2 instance rejections, absent V2 clusters, absent mappings, and absent output. Every required record enters `scope-created`/`scope-merged` or `coverage-blocked`; it cannot use `OUT_OF_FRV_POLICY` to bypass scope construction or safety routing.

An **approved upstream hard-rejection record** is the complementary canonical-inventory subset and is valid only with all of: `source_identity`, `upstream_candidate_reference`, `approved_rule_id`, `approved_rule_version`, `hard_non_vegetation_evidence_reference`, `decision_digest`, `approver`, `approval_timestamp`, `approval_digest`, and `safety_completeness_effect`. An ordinary V2 instance rejection is not hard non-vegetation proof. No V2 cluster, missing mapping, missing output, or missing approval evidence is a safe exclusion; if approval evidence is missing, the record returns to the required FRV inventory. `OUT_OF_FRV_POLICY` cannot substitute for an approved hard rejection.

`FRVScopeCoverageManifest` is the proposed coverage record. Each canonical row records source identity, canonical and required inventory digests, candidate reference/type/status, upstream-evidence reference, approved-hard-rejection evidence or null, `scope_id` or null, construction and coverage state, failure reasons, review requirement, safety-completeness effect, and diagnostic reference. It records `canonical_inventory_count`, `required_inventory_count`, `approved_hard_rejection_count`, `scope_created_or_merged_count`, `coverage_blocked_count`, `canonical_inventory_complete`, `required_inventory_complete`, `approved_hard_rejection_complete`, `scope_coverage_complete`, and `routing_complete`. `canonical_inventory_complete` requires every canonical row to be exactly required or approved-hard-rejection; `approved_hard_rejection_complete` requires the complete approval evidence listed above; `required_inventory_complete` requires every required row to be accounted for; and `scope_coverage_complete` requires every required row to have a successful scope route with no blocked required region. A mandatory in-domain wire-provider failure records the affected scope as `coverage-blocked`, sets its membership routing and `routing_complete` false, and makes the affected scope's safety completeness false. First-version legal routing states are `SCOPE_CREATED`, `SCOPE_MERGED`, `SCOPE_BLOCKED`, `SOURCE_IDENTITY_FAILED`, `UPSTREAM_EVIDENCE_INVALID`, `REGION_GEOMETRY_INVALID`, and `REGION_MAPPING_INCOMPLETE`; the `OUT_OF_FRV_POLICY` enum, if retained for schema compatibility, is prohibited from first-version production output.

The two inventory conservation laws are:

```text
all canonical upstream inventory records
= required FRV records
+ approved upstream hard-rejection records

required FRV records
= scope-created or scope-merged records
+ coverage-blocked records
```

Each term in both equations is mutually exclusive and complete. Every canonical record occurs exactly once on one side of the first partition; every required record occurs exactly once on one side of the second. Independent healthy regions can proceed when another required region is blocked, but a blocked region prevents its own complete-coverage and membership-complete claim.

### 3.2 Exact `EXHAUSTIVE_REGION` algorithm

The first version uses source-local XYZ, three-dimensional voxel geometry, not an XY footprint plus an implementation-chosen height range. For every validated inventory record:

1. Resolve its A1 representative or V2-member reference to an exact A1 `voxel_key`; validate source identity, the frozen A1 voxel configuration digest, origin, coordinate frame, units, and key mapping.
2. Reconstruct that key's precise source-local, half-open 3D voxel cell from the frozen A1 voxel size, origin, coordinate context, and boundary convention.
3. Reject any record whose identity, key, geometry, or mapping cannot be validated to `coverage-blocked`; it creates no normal scope.
4. Build `scope_core` as the deterministic union of all validated reconstructed voxel cells for the record. Overlap is permitted and is retained as provenance.
5. Derive `scope_context_margin` by expanding `scope_core` only by the frozen, versioned XY and Z margin policy. Its dimensions, units, maximum permitted values, and per-axis half-open expansion are fixed by the policy digest. It expands evaluation scope, not membership support.
6. Deterministically union all valid context-margin geometries for the same source. Sort inventory keys canonically, partition the resulting union into non-overlapping half-open core tiles, and use original XYZ to test membership in that union. A point at a boundary belongs to exactly one tile under the frozen half-open rule.

`scope_core`, `scope_context_margin`, and `feature_halo` are intentionally different:

| Term | Meaning | Formal-output effect |
|---|---|---|
| `scope_core` | Union of reconstructed A1 voxel cells directly evidenced by upstream candidate records. | Provenance and baseline processing area. |
| `scope_context_margin` | Versioned XY/Z expansion of the core used to evaluate nearby real points in the formal decision scope. | Points inside the final union must receive one terminal state. |
| `feature_halo` | Additional tile-neighbour rows needed only to calculate features or structural relations correctly. | Never independently forms scope and never produces formal output. |

The final first-version decision scope is the source-local 3D union of `scope_context_margin` geometries. No HAG coordinate may serve as the sole scope coordinate. Floating-point comparisons use a frozen numeric specification: coordinates are transformed once into the frozen source-local frame; voxel/tile boundaries are derived from integer keys and stored as finite IEEE-754 binary64 values; containment is `min <= xyz < max` on every axis after a frozen, documented coordinate canonicalization. No epsilon may be selected at runtime.

Every scope union stores `scope_construction_version`, A1 voxel configuration digest, coordinate context, boundary convention, margin-policy digest, upstream-inventory digest, union geometry digest, and `scope_union_id`. Changing the A1 configuration, coordinate context, boundary convention, or margin-policy dimensions creates a new construction version.

### 3.3 Union and point provenance

All valid scopes for one `SourceIdentity` are unioned and de-duplicated before point decisions. Every unique `original_index` inside the final union is evaluated once and output once. A record can contribute zero, one, or many raw points; a zero-point scope is not automatically successful and must retain its reason. Halos never duplicate output.

Each output point carries `scope_union_id`, `primary_scope_id`, and `supporting_upstream_region_reference`. The supporting reference is a typed offset/count, an externally digested block reference, or a digest-bound ragged table that retains all contributing upstream regions; retaining only a primary ID is forbidden. `primary_scope_id` is the lowest canonical inventory key and is not ownership. The coverage manifest and membership artifact bind each other by digest.

## 4. Deterministic evidence and low-HAG trunk-base decision

Base evidence is computed for all relevant original points before a membership state is assigned. It includes HAG evidence, multi-scale geometry, local column/branch structure, scan quality, ground conflict, wire conflict, source attributes, and scope/halo quality. Base evidence must not depend on a final membership state, a V1/V2 label, or point-processing order.

The proposed structural-support graph is versioned and read-only. In the first version, each vertex is exactly one complete-resolution real point: `vertex_id = original_index`. Each vertex binds `SourceIdentity`, original XYZ, tile/core/halo provenance, and a base-evidence reference. Derived support units, virtual nodes, voxel nodes, cluster nodes, and synthetic centre points are forbidden as vertices. The graph need not be resident in full: tile/halo implicit edge construction, chunked adjacency, deterministic edge-generation configuration, read-only memory maps or block files, and stable-order local reconstruction are allowed only if every structural-continuity evidence path remains traceable to real `original_index` values.

The stored base adjacency is an undirected edge. An edge exists only when both endpoints have the same `SourceIdentity`, are full-resolution real points, satisfy the versioned support-distance range, satisfy direction or column/branch structural compatibility, have complete required tile/halo support, have no prohibited spatial gap between them, and are selected by deterministic distance then `original_index` tie-breaks. Edge construction does not depend on final membership state, V1/V2 labels, tile order, thread order, or source-row read order. The graph binding includes `structural_graph_version`, `vertex_model = ORIGINAL_POINT`, `edge_rule_version`, `tile_halo_configuration_digest`, `support_radius_policy_digest`, `direction_compatibility_digest`, `seed_rule_digest`, and `synchronous_solver_version`. Changing the vertex identity model, using derived support units, edge-direction semantics, seed definition, synchronous solver structure, or the conflict types that propagation may cross requires an algorithm or rule-version change.

Upper vegetation-support seeds are derived solely from strong positive, non-low-HAG propagation-independent evidence: reliable positive HAG, multi-scale canopy/branching or woody-structure evidence, adequate scan quality, and no hard ground or valid wire conflict. Seeds are structural evidence, not tree IDs or final membership states.

For low/near-zero HAG, seed support flows directionally from higher-HAG/compatible structural seeds through the undirected base adjacency in synchronous rounds. In every round all vertices read the previous immutable state and write one next state simultaneously; a low-HAG point cannot become an upper seed or reverse-propagate support. A point may gain only the frozen structural-continuity evidence implied by a permitted path to a seed; no V1/V2 or final membership state propagates. Propagation cannot cross reliable ground conflict or valid wire conflict. The rule uses a fixed maximum round count and an explicit stable-state convergence test. Results are invariant to thread scheduling, tile completion order, and source-row order. Nonconvergence, support-path breakage, unstable evidence, a prohibited gap, or a decisive conflict routes the point to `REVIEW_REQUIRED_MEMBERSHIP`.

The low-HAG trunk-base path is therefore:

```text
low or near-zero HAG
+ reliable non-ground local structure
+ deterministic structural continuity to an upper support seed
+ trunk/woody-column-compatible geometry
+ adequate scan support
+ no reliable ground hard-exclusion or wire conflict
-> VERIFIED_VEGETATION_CANDIDATE
```

Structural continuity alone cannot override a reliable ground or wire conflict. Conversely, near-zero HAG alone is never a ground exclusion. Reliable ground exclusion needs trusted ground authority/class semantics, local ground morphology, sufficient quality, no upward structural support, and no credible vegetation counter-evidence. Missing/invalid/out-of-domain HAG is normally review. A canopy/branch route and a trunk-base route both require multi-evidence support; neither relies on one linearity or scattering threshold.

## 5. Formal membership states and order

The terminal states are mutually exclusive:

- `VERIFIED_VEGETATION_CANDIDATE`: sufficient automatic full-resolution evidence.
- `EXCLUDED_NON_CANDIDATE`: reliable explicit hard non-vegetation or ground evidence with no credible vegetation counter-evidence. First-version wire evidence cannot produce this state.
- `REVIEW_REQUIRED_MEMBERSHIP`: a real point has conflict, insufficient evidence, uncertain scope edge, inadequate scan support, provider uncertainty, or unstable structural evidence.
- `MANUAL_INCLUDE` and `MANUAL_EXCLUDE`: an approved, valid, conflict-resolved effective manual action, retaining automatic evidence and approval provenance.

Lack of positive evidence is not proven exclusion. The decision order is:

| Priority | Condition | Result |
|---:|---|---|
| 1 | SourceIdentity, scope/index, mandatory provider, stream, or artifact-integrity failure | Source/region coverage failure; no ordinary terminal point state. |
| 2 | Override chain has no uniquely resolvable effective action | Elevated review; no manual terminal state. |
| 3 | One approved, valid, conflict-resolved effective manual exclude | `MANUAL_EXCLUDE`. |
| 4 | One approved, valid, conflict-resolved effective manual include | `MANUAL_INCLUDE`. |
| 5 | Reliable hard non-vegetation or ground evidence, never first-version wire-body evidence, and no credible vegetation counter-evidence | `EXCLUDED_NON_CANDIDATE`. |
| 6 | Positive ordinary-HAG vegetation rule or deterministic low-HAG trunk-base rule | `VERIFIED_VEGETATION_CANDIDATE`. |
| 7 | Wire, ground, HAG, geometry, scope-edge, scan, or structural conflict | `REVIEW_REQUIRED_MEMBERSHIP`. |
| 8 | Evidence insufficient either to verify or to hard-exclude | `REVIEW_REQUIRED_MEMBERSHIP`. |

Only an approved, valid, conflict-resolved manual exclusion may produce `MANUAL_EXCLUDE`. A rejected individual request produces `MANUAL_OVERRIDE_REJECTED` and preserves the automatic state; it does not automatically fail the source or region.

## 6. Override-chain invariant and failure scope

`FRVManualOverrideArtifact` is a proposed immutable artifact holding the source identity, override ID, requested action, exact indexes or versioned geometry resolved deterministically to indexes, requester/time/reason/authority, approval record, automatic pre-override evidence, predecessor digest, conflict flags, resulting state, and artifact digest.

The formal invariant is:

```text
For one source identity, original_index, and predecessor chain,
at most one approved effective override may exist.
```

Every action names an explicit predecessor digest. Chains must be acyclic and deterministically resolvable. A newly approved successor replaces, rather than coexists with, the previous effective action; replaced actions remain historical but are not effective. Same-predecessor approved actions that conflict, an unresolved multiple-effective condition, or an unparseable chain creates no effective terminal manual state and routes the point to `REVIEW_REQUIRED_MEMBERSHIP` or coverage-level elevated review according to configured failure extent. `MANUAL_EXCLUDE` has no blanket precedence over `MANUAL_INCLUDE`.

Individual-request faults — out-of-range index, insufficient authority, missing reason, unapproved request, or out-of-scope target without a newly constructed scope — are `MANUAL_OVERRIDE_REJECTED` only. They preserve the automatic state. Artifact-integrity faults — required override artifact digest mismatch, unreadable/corrupt structure, missing approval record for a claimed approval, absent predecessor artifact, or impossible chain resolution — block override application and become artifact/coverage failure or elevated review. No override defeats identity, provider, stream, scope, or artifact-integrity failure.

## 7. Wire provider availability policy

A `wire influence domain` is defined only by authoritative confirmed-wire geometry, its approved spatial influence range, coordinate frame and uncertainty model, and provider version. A scope/region that authoritative wire inventory determines is within such a domain requires a complete full-resolution wire-relation provider. The first-version routing boundary is deterministic:

| Condition | Required routing |
|---|---|
| Mandatory provider is missing or invalid; `WIRE_PROVIDER_MISSING` or `WIRE_PROVIDER_INVALID` | Affected in-domain scope/region is `coverage-blocked`. |
| Provider `SourceIdentity` or authoritative geometry coordinate frame mismatches; `WIRE_IDENTITY_MISMATCH` | Affected in-domain scope/region is `coverage-blocked`. |
| Authoritative geometry is unavailable/unreadable, its required artifact digest mismatches, or `WIRE_GEOMETRY_UNAVAILABLE` applies | Affected in-domain scope/region is `coverage-blocked`. |
| Complete wire influence-domain query cannot be performed | Affected in-domain scope/region is `coverage-blocked`. |
| Provider is valid, identity-bound, geometry-readable, and its influence-domain query and point relation finish; point relation is `WIRE_BODY_POSSIBLE`, `WIRE_PROXIMITY_ONLY`, or `WIRE_VEGETATION_CONFLICT` | The formal in-scope point is `REVIEW_REQUIRED_MEMBERSHIP`. |
| Authoritative inventory, successful spatial query, and complete influence-domain geometry prove the whole relevant scope is outside all approved domains | `WIRE_NOT_APPLICABLE_PROVEN`. |
| A first-version configuration attempts a wire-body hard exclusion or `WIRE_BODY_CONFIRMED` | Illegal configuration/rule error. |

Point-level wire review is therefore permitted only after the provider is valid, `SourceIdentity` matches, authoritative geometry is readable, the influence-domain query succeeds, and the point-level relation calculation completes. It is not a substitute for a provider coverage failure. `WIRE_NOT_APPLICABLE_PROVEN` requires the authoritative inventory, successful spatial query, and complete influence-domain geometry together; no A4 match, no detected wire, no provider output, no wire ID, an unloaded file, query failure, damaged geometry, or identity mismatch proves non-applicability.

The repository currently has no approved full-resolution wire-body provider. Consequently, this first version cannot emit `WIRE_BODY_CONFIRMED`, cannot hard-exclude a point for wire body, and routes all valid in-domain `WIRE_BODY_POSSIBLE`, `WIRE_PROXIMITY_ONLY`, and `WIRE_VEGETATION_CONFLICT` results to `REVIEW_REQUIRED_MEMBERSHIP`.

## 8. Proposed membership artifact, provenance, and conservation

`FullResolutionVegetationMembershipArtifact` has one semantic artifact per `SourceIdentity + scope_union_id + configuration_digest`. Physical blocks are multiple ordered storage blocks beneath one manifest, never independent semantic artifacts.

Required artifact-level fields are:

```text
schema_version
algorithm_version
decision_rule_version
source_identity / source_file_sha256 / optional canonical_point_content_digest
artifact_id / artifact_sha256 / manifest_digest
configuration_digest
scope_coverage_manifest_digest
scope_union_id
scope_construction_version
scope_union_geometry_digest
upstream_inventory_digest
source_point_count
unique_in_scope_point_count
evaluated_point_count
state counts and unasserted_in_scope_count
canonical_inventory_count / required_inventory_count / approved_hard_rejection_count
scope_created_or_merged_count / coverage_blocked_count
canonical_inventory_complete / required_inventory_complete
approved_hard_rejection_complete / scope_coverage_complete
routing_complete
membership_decision_complete
provider, override, block, warning, limitation, and provenance digests
```

In a complete artifact, `evaluated_point_count == unique_in_scope_point_count`. Non-overlapping half-open index blocks contain strictly increasing unique `original_index`, `membership_state`, `reason_flags`, `input_quality_flags`, `hag_evidence_state`, optional `raw_hag`, optional `clamped_hag`, HAG quality/reason, `segmentation_hag_readiness`, `scope_union_id`, `primary_scope_id`, `supporting_upstream_region_reference`, and `manual_override_reference`. Raw XYZ remains exactly retrievable through validated source replay even when not duplicated in each block. Numeric payloads are finite; blocks and manifests are immutable/read-only; any digest, range, count, type, ordering, duplicate, source, provider, or stream-completeness failure invalidates the affected artifact/coverage claim.

The inventory, scope, and membership conservation laws are:

```text
all canonical upstream inventory records
= required FRV records + approved upstream hard-rejection records

required FRV records
= scope-created or scope-merged records + coverage-blocked records

all unique original rows inside the final decision scope
= unique_in_scope_point_count

unique_in_scope_point_count
= VERIFIED_VEGETATION_CANDIDATE + EXCLUDED_NON_CANDIDATE
+ REVIEW_REQUIRED_MEMBERSHIP + MANUAL_INCLUDE + MANUAL_EXCLUDE
```

The state equation uses mutually exclusive state counts. `membership_decision_complete` is true only when every required record has entered scope-created/scope-merged or coverage-blocked, the current artifact has a terminal state for every unique original row in its formal scope, `unasserted_in_scope_count = 0`, and no provider, stream, or integrity failure affects that artifact. A required region made `coverage-blocked` by a mandatory wire-provider failure has `routing_complete = false`, is not membership complete, has safety completeness false, and prevents source-level or scene-level completeness, although independent healthy scopes may publish their own artifacts. Point-review counts cannot mask provider coverage failure. Coverage manifest and membership artifact digests bind in both directions; publishing an artifact may not drop union or upstream provenance.

## 9. Streaming, V4 hand-off, and stage boundaries

Processing is deterministic and bounded: identity/scope validation; immutable provider preparation; core-plus-feature-halo tile evidence calculation; synchronous structural solving; ordered core output; and manifest replay/reconciliation. Tiles use a frozen support radius/validated equivalent, and only core rows produce output. Parallel work may merge only by deterministic block range and `original_index`. Completion must demonstrate schedule-independent output and remain within the approximately 16 GB acceptance budget with documented margin.

FRV proposes, but does not implement, V4 hand-off. Automatic candidates and manual includes may later enter a recovered set only with original-index/XYZ recovery, evidence, source identity, and independent segmentation HAG readiness. Review points preserve original XYZ, membership state, `hag_evidence_state`, optional HAG values, quality/reason, readiness, and wire evidence for unresolved measured-point hand-off. Coverage failures produce coverage routing, never fictional points or labels.

| Stage | Frozen boundary |
|---|---|
| FRV-A0 | Freeze design, authority, rule graph, scope semantics, states, artifact direction, and safety boundary. |
| FRV-A1 | State/configuration, SourceIdentity and file/point digest semantics, scope/artifact manifests, and pure validators. |
| FRV-A2 | Full-resolution HAG, geometry, scan-quality, and wire-provider input/identity/quality validation only. |
| FRV-A3 | Canonical/required inventory, approved-hard-rejection accounting, exact scope construction, coverage manifest, union, and deterministic partitioning; no membership rule. |
| FRV-A4 | Original-index-vertex structural graph, tile/halo base evidence, deterministic edge construction, seed evidence, synchronous-solver infrastructure, boundary consistency, resources, and replay; no final membership state. |
| FRV-A5 | Formal membership rules, low-HAG synchronous-solver decisions, valid-provider wire point uncertainty, override chain, and state machine; mandatory wire-provider failure may not be downgraded to ordinary point review. |
| FRV-A6 | Artifact publication, inventory/scope/membership conservation, provider-failure coverage routing, manifest replay, coverage/safety hand-off, and V4 adapter; no frozen-rule change. |
| FRV-B | Labels, calibration, independent acceptance, 16 GB performance, determinism, and corruption detection. |

Every stage requires independent testing and review after approval. FRV-A3 does not implement membership; FRV-A4 does not output a final membership state; FRV-A5 does not publish an artifact; FRV-A6 does not modify frozen rules. This document authorizes none of these stages.

## 10. Review gates and revision record

Independent review has confirmed the exact scope algorithm, separation of core/context/halo, no propagated V1/V2 membership, deterministic low-HAG evidence, complete union provenance, inventory/scope/membership conservation, one effective override action, conflict escalation, mandatory/non-applicable wire policy, conservative no-wire-body behavior, safety routing, and the stage boundaries above.

This revision integrates the previous targeted freeze resolutions into the governing sections. It does not rely on a later section to override earlier text. The final independent review result is `PASS_READY_FOR_FREEZE_APPROVAL`, and the approved design is now `DESIGN-FROZEN`. This remains not authorization to begin implementation.

## 11. Design freeze approval and change control

### 11.1 Freeze decision

```text
Freeze decision: APPROVED
Frozen status: DESIGN-FROZEN
Freeze date: 2026-08-01
Structural design blockers: NONE
Implementation authorized: NO
```

This approval follows multiple targeted FRV design reviews and the final independent result `PASS_READY_FOR_FREEZE_APPROVAL`. The first-version scope construction, canonical/required inventory conservation, original-index vertex model for the structural graph, wire-provider failure routing, artifacts, state machine, safety routing, and FRV phase boundaries have all been frozen. The project owner has approved this separate design-freeze operation. This decision records a frozen design only; it does not claim implementation is complete or authorize implementation.

### 11.2 Frozen scope

This freeze covers the following design semantics and boundaries:

- FRV module responsibility boundary, `SourceIdentity`, and source-summary semantics.
- First-version `EXHAUSTIVE_REGION` scope construction.
- Canonical/required inventory, approved hard-rejection rules, scope union, de-duplication, and provenance.
- Full-resolution HAG, geometry, scan, and wire input boundaries.
- The `original_index` structural-graph vertex model and synchronous low-HAG trunk-support method.
- Membership states, decision priority, and override-chain rules.
- Mandatory/not-applicable wire-provider policy.
- Membership-artifact direction, conservation, coverage, and safety hand-off.
- FRV-A1 through FRV-A6 and FRV-B stage boundaries.

### 11.3 Not frozen as implemented

The following are not implemented by this design freeze and remain subject to later controlled work or calibration: Python contracts and enums; producers and loaders; full-resolution HAG and geometry providers; the wire-relation provider; scope producer; structural-support graph; membership-rule implementation; artifact publisher; V4 adapter; manual annotation data; threshold and parameter calibration; and performance plus 16 GB memory acceptance. This document does not assert that any of those capabilities already exists.

### 11.4 Post-freeze change control

Any change that affects the frozen semantics must reopen the design through:

```text
DESIGN-FROZEN
→ DESIGN-REVIEW-CANDIDATE
→ targeted revision
→ independent review
→ new freeze approval
```

Reopening is mandatory for a change to FRV module responsibility; the first-version decision scope or scope construction; canonical/required inventory semantics; state machine or decision priority; hard-rejection semantics; structural-graph vertex model; use of a derived support unit, virtual point, voxel node, or cluster node; low-HAG synchronous support structure; mandatory wire-provider policy; first-version wire-body hard exclusion; override chain; artifact conservation or safety routing; or the FRV/V4 responsibility boundary. It is also mandatory to introduce a scenario, file, source-ID, or device special case; V1/V2 label propagation; or a deep-learning replacement for the first-version rules.

### 11.5 Controlled calibration without reopening overall design

Provided the frozen rule semantics remain unchanged, a later separately authorized FRV-B calibration or implementation stage may set or tune neighbourhood radius, `k`, physical scales, XY/Z margins, feature halo, tile and chunk sizes, synchronous maximum rounds, convergence tolerance, evidence weights, thresholds, review score differences, performance parameters, and storage block size. Such work must pass FRV-B calibration and acceptance, version configuration and parameters, produce a new configuration digest, add no scenario-specific exception, and never weaken safety routing. A parameter change that changes rule semantics must reopen the design.

### 11.6 Post-freeze implementation gate

`DESIGN-FROZEN` authorizes only the next document: an FRV-A1 implementation plan. It does not authorize FRV-A1 code implementation. Before any code begins, the detailed FRV-A1 implementation plan must be written, independently reviewed, explicitly approved for implementation, assigned to an implementation tool, and followed by separate testing and review for every stage.

### 11.7 Relationship to V4

After this FRV freeze, V4 D01 may be reconsidered, but it is not automatically unblocked. This task does not modify the V4 document; it requires a separate, targeted revision. The V4 document remains `DESIGN-REVIEW-CANDIDATE`, and V4 implementation remains unauthorized.
