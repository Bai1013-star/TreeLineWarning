# FRV-A1 Detailed Implementation Plan

**Status:** `IMPLEMENTATION-PLAN-REVIEW-CANDIDATE`
**Normative FRV design:** `docs/full-resolution-vegetation-membership-design.md` (`DESIGN-FROZEN`, 2026-08-01)
**Downstream V4 design:** `docs/v4-tree-segmentation-design.md` (`DESIGN-FROZEN`, 2026-08-01)
**Plan authorizes implementation:** `NO`

## 1. Authority, baseline, and frozen A1 boundary

This document is subordinate to the two frozen designs and is the sole FRV-A1 implementation-plan candidate. It freezes public contract semantics for later implementation; it does not alter the frozen designs. The verified revision baseline is `feature/algorithm` at `81b5a204cda29847c20178d12666cd70b77c58c6`, with `0 0` divergence from `origin/feature/algorithm...HEAD` and only the three expected untracked design/plan documents.

FRV-A1 defines immutable contracts, configuration rules, source identity, canonical serialization, manifest and summary schemas, and pure validators. A1 never accesses a filesystem, reads point or payload bytes, calculates a source/payload/geometry hash from data, discovers inventory, constructs scope/union/partition, validates provider truth, executes membership or overrides, publishes/replays an artifact, makes a safety decision, or supplies V4 runtime data.

| Stage | Frozen responsibility outside A1 |
|---|---|
| A2 | Provider input, source identity, HAG, geometry, scan, wire identity and quality validation. |
| A3 | Canonical inventory, approved hard-rejection evidence, required inventory, scopes, union, partition, coverage and real geometry production. |
| A4 | Tile/halo evidence, original-index graph and solver infrastructure. |
| A5 | Membership state machine, wire-conflict and manual-override execution. |
| A6 | Exact payload publication/replay, conservation execution, safety hand-off and V4 adapter. |

The five terminal membership states remain `VERIFIED_VEGETATION_CANDIDATE`, `EXCLUDED_NON_CANDIDATE`, `REVIEW_REQUIRED_MEMBERSHIP`, `MANUAL_INCLUDE`, and `MANUAL_EXCLUDE`. The first version prohibits label propagation, nearest-neighbour membership, voxel/cluster/prism membership, scenario/file/source-ID/device/test-sample special cases, forced exclusion for an unasserted point, and wire-body hard exclusion. A1 must reject unknown fields under `STRICT_EXACT_SCHEMA`; it never creates an exclusion from absent evidence.

## 2. Contract conventions

All contracts are frozen dataclasses with deep-immutable mappings/tuples and read-only arrays where arrays are materialized. `to_dict`, `from_dict`, `to_digest_dict`, and `computed_digest` are exact-schema operations. `from_dict` rejects missing and unknown fields, invalid versions, nonfinite numbers, bool values where integers/enums are required, unknown enum values, and unknown flag bits. Optional fields are always serialized as `null`, never omitted.

`canonical_frv_json_bytes(value)` is UTF-8 JSON with `ensure_ascii=False`, lexicographically sorted keys, compact separators `(',', ':')`, `allow_nan=False`, tuples represented as lists, and enum values represented by their frozen integer/string values. `frv_digest(value)` is SHA-256 of those bytes. Every SHA-256 field is exactly 64 lower-case hexadecimal characters.

All public records include their declared schema version in their digest payload. A self-digest excludes only itself unless a section below explicitly defines a different byte-domain digest. No producer or implementation step may choose an alternative public field, digest domain, array representation, or configuration rule.

## 3. Complete public contract inventory

| Public contract | Exact A1 responsibility | Later producer → consumer |
|---|---|---|
| `FrvMembershipState`, `FrvScopeMode`, `FrvHagEvidenceState`, `FrvSegmentationHagReadiness`, `FrvWireRelationState`, `FrvWireProviderState` | Closed enum representation and structural validation. | A2/A5 → A6/V4 |
| `FrvInventoryDisposition`, `FrvScopeConstructionState`, `FrvScopeCoverageState`, `FrvSafetyCompletenessEffect` | Closed inventory routing, coverage-result, and safety-completeness representation. | A3 → A6/V4 |
| `FrvOverrideRequestedAction`, `FrvOverrideApprovalState`, `FrvOverrideConflictFlag`, `FrvReasonFlag`, `FrvQualityFlag`, `FrvErrorCode` | Frozen B3 enum/flag/error library. | A2–A6 → A6/V4 |
| `FrvConfigurationPurpose`, `FrvConfiguration` | Exact configuration and production wire-provider binding. | configuration → all stages |
| `FrvCanonicalPointFieldSpec`, `FrvCanonicalPointContentSpec` | B1 canonical decoded-point content framing schema. | A2/A6 calculator → all consumers |
| `FrvRelocationAudit`, `FrvSourceEquivalenceAudit`, `SourceIdentity` | Acyclic content identity and full-record audit identity. | A2 → A3/A6/V4 |
| `FrvScopeGeometryEncodingSpec`, `FRVScopeDefinition` | Geometry encoding specification and content-digest references. | A3 → A6/V4 |
| `FRVApprovedHardRejectionEvidence`, `FRVScopeCoverageRecord`, `FRVScopeCoverageManifest` | Complete canonical-inventory disposition and coverage accounting contracts. | A3 → A6/V4 |
| `FRVManualOverrideArtifact` | Immutable override request/approval/chain descriptor only. | A5 → A6/V4 |
| `FRVArtifactArrayDescriptor`, `FRVArtifactPayloadSchema`, `FRVArtifactBlockDescriptor` | Exact point-array and block descriptor schema. | A6 → V4 |
| `FullResolutionVegetationMembershipArtifact`, `FRVArtifactBindingRecord` | Semantic artifact metadata, exact artifact digest domain, and binding. | A6 → V4 |

No A1 record is business-data production authority. Future implementation may add these names to `core/contracts/frv.py` and export them only after explicit code authorization; no code, exports, or tests are authorized by this plan.

## 4. Closed enums, flags, and state constraints

All integer enums use the frozen unsigned integer values and all flags serialize as `uint32`; `bool` and `numpy.bool_` are rejected before conversion. The B3 inventory is closed: `FrvMembershipState` is `VERIFIED_VEGETATION_CANDIDATE=1`, `EXCLUDED_NON_CANDIDATE=2`, `REVIEW_REQUIRED_MEMBERSHIP=3`, `MANUAL_INCLUDE=4`, `MANUAL_EXCLUDE=5`; `FrvScopeMode` is `EXHAUSTIVE_REGION=1`, `EXHAUSTIVE_SOURCE=2`, `SPARSE_ASSERTIONS_ONLY=3`; `FrvHagEvidenceState` is `HAG_VALID_POSITIVE=1`, `HAG_VALID_NEAR_ZERO=2`, `HAG_WITHIN_NEGATIVE_TOLERANCE=3`, `HAG_MISSING=4`, `HAG_INVALID=5`, `HAG_OUT_OF_DOMAIN=6`; and `FrvSegmentationHagReadiness` is `READY=1`, `REVIEW_REQUIRED=2`.

`FrvInventoryDisposition` is a public integer enum with exactly two stable values:

```text
REQUIRED_FRV = 1
APPROVED_HARD_REJECTION = 2
```

There is no third ordinary route. Each canonical inventory record has exactly one disposition. `FrvScopeConstructionState` retains `SCOPE_CREATED=1`, `SCOPE_MERGED=2`, and `NOT_CREATED=3`; `FrvScopeCoverageState` retains `COVERED=1`, `SCOPE_BLOCKED=2`, `SOURCE_IDENTITY_FAILED=3`, `UPSTREAM_EVIDENCE_INVALID=4`, `REGION_GEOMETRY_INVALID=5`, `REGION_MAPPING_INCOMPLETE=6`, and reserved `OUT_OF_FRV_POLICY=7`. `OUT_OF_FRV_POLICY` is prohibited from first-version production output.

`FrvWireProviderState.WIRE_NOT_APPLICABLE_PROVEN` is a runtime result only. It is not expressible by an absent configuration binding. Missing provider, missing provider digest, failed query, unavailable inventory, identity mismatch, geometry unavailable, and unverified empty output are provider/coverage failures, never non-applicability.

`FrvWireRelationState` is `WIRE_BODY_CONFIRMED=1`, `WIRE_BODY_POSSIBLE=2`, `WIRE_PROXIMITY_ONLY=3`, `WIRE_VEGETATION_CONFLICT=4`; the confirmed-body value has no first-version producer path. `FrvWireProviderState` is `WIRE_RELATION_VALID=1`, `WIRE_NOT_APPLICABLE_PROVEN=2`, `WIRE_PROVIDER_MISSING=3`, `WIRE_PROVIDER_INVALID=4`, `WIRE_IDENTITY_MISMATCH=5`, `WIRE_GEOMETRY_UNAVAILABLE=6`. `FrvOverrideRequestedAction` is `MANUAL_INCLUDE_REQUESTED=1`, `MANUAL_EXCLUDE_REQUESTED=2`; `FrvOverrideApprovalState` is `PENDING_REVIEW=1`, `MANUAL_INCLUDE_APPROVED=2`, `MANUAL_EXCLUDE_APPROVED=3`, `MANUAL_OVERRIDE_REJECTED=4`; `FrvOverrideConflictFlag` is `NONE=0`, `SAME_PREDECESSOR_CONFLICT=1`, `MULTIPLE_EFFECTIVE_OVERRIDE=2`, `CHAIN_CYCLE=4`, `UNRESOLVABLE_PREDECESSOR=8`, `ARTIFACT_INTEGRITY_FAILURE=16`.

`FrvReasonFlag` is `NONE=0`, `MANUAL_INCLUDE_PROVENANCE=1`, `MANUAL_EXCLUDE_PROVENANCE=2`, `INSUFFICIENT_EVIDENCE=4`, `GROUND_CONFLICT=8`, `VEGETATION_COUNTER_EVIDENCE=16`, `WIRE_CONFLICT=32`, `SCOPE_EDGE_CONFLICT=64`, `HAG_MISSING=128`, `HAG_INVALID=256`, `HAG_OUT_OF_DOMAIN=512`, `PROVIDER_UNCERTAINTY=1024`, `OVERRIDE_CHAIN_CONFLICT=2048`, `REVIEW_REQUIRED=4096`. `FrvQualityFlag` is `NONE=0`, `SOURCE_IDENTITY_FAILURE=1`, `ORIGINAL_INDEX_INVALID=2`, `ORIGINAL_INDEX_DUPLICATE=4`, `STREAM_INCOMPLETE=8`, `PROVIDER_MISSING=16`, `PROVIDER_INVALID=32`, `WIRE_IDENTITY_MISMATCH=64`, `WIRE_GEOMETRY_UNAVAILABLE=128`, `SCOPE_MAPPING_CONFLICT=256`, `SCOPE_GEOMETRY_CONFLICT=512`, `ARTIFACT_DIGEST_MISMATCH=1024`, `MANIFEST_DIGEST_MISMATCH=2048`, `BLOCK_DIGEST_MISMATCH=4096`, `OVERRIDE_ARTIFACT_FAILURE=8192`.

`FrvErrorCode` is exactly `FRV_SOURCE_IDENTITY_FAILED`, `FRV_ORIGINAL_INDEX_INVALID`, `FRV_ORIGINAL_INDEX_DUPLICATE`, `FRV_STREAM_INCOMPLETE`, `FRV_PROVIDER_MISSING`, `FRV_PROVIDER_INVALID`, `FRV_WIRE_IDENTITY_MISMATCH`, `FRV_WIRE_GEOMETRY_UNAVAILABLE`, `FRV_HAG_MISSING`, `FRV_HAG_INVALID`, `FRV_HAG_OUT_OF_DOMAIN`, `FRV_SCOPE_EDGE_CONFLICT`, `FRV_SCOPE_MAPPING_INCOMPLETE`, `FRV_SCOPE_GEOMETRY_INVALID`, `FRV_ARTIFACT_DIGEST_MISMATCH`, `FRV_MANIFEST_DIGEST_MISMATCH`, `FRV_BLOCK_DIGEST_MISMATCH`, `FRV_OVERRIDE_CHAIN_CONFLICT`, `FRV_INSUFFICIENT_EVIDENCE`, `FRV_GROUND_CONFLICT`, `FRV_VEGETATION_CONFLICT`, `FRV_WIRE_CONFLICT`, and `FRV_REVIEW_REQUIRED`.

`FrvSafetyCompletenessEffect` is the closed string enum `PRESERVES_SAFETY_COMPLETENESS` or `BLOCKS_SAFETY_COMPLETENESS`. An approved hard rejection and a covered required row require `PRESERVES_SAFETY_COMPLETENESS`; every coverage-blocked required row requires `BLOCKS_SAFETY_COMPLETENESS`.

## 5. Exact `FrvConfiguration` public API

`FrvConfigurationPurpose` has exactly `TEST_ONLY`, `CALIBRATION_CANDIDATE`, and `PRODUCTION_APPROVED`. `FrvConfiguration` has no paths, source/scene/device/test selectors, provider object, callable, runtime arrays, or file handles. `configuration_digest` is derived from every other field and excludes only itself.

| Fields | Type / required rule | Digest / validation / owner |
|---|---|---|
| `schema_version`, `configuration_version`, `algorithm_version`, `decision_rule_version`, `canonicalization_version` | nonempty strings, required | included; exact supported version; all stages consume |
| `source_identity_schema_version`, `canonical_point_content_spec_version`, `scope_schema_version`, `scope_construction_version`, `scope_geometry_encoding_version`, `coverage_manifest_schema_version`, `artifact_schema_version`, `block_schema_version`, `payload_schema_version`, `override_schema_version` | nonempty strings, required | included; must match referenced contracts |
| `scope_mode` | `FrvScopeMode`, required | included; production-v1 requires `EXHAUSTIVE_REGION` |
| `scope_policy_digest`, `override_policy_digest`, `feature_policy_digest`, `resource_policy_digest` | strict SHA, required | included; binding only |
| `hag_provider_interface_version`, `geometry_provider_interface_version`, `scan_provider_interface_version` | nonempty strings, required | included; A2 binding |
| `hag_provider_config_digest`, `geometry_provider_config_digest`, `scan_provider_config_digest` | strict SHA, required | included; A2 binding |
| `wire_provider_interface_version`, `wire_provider_config_digest` | see purpose rules below | included; pure binding validation |
| `feature_policy_version`, `membership_parameter_schema_version` | nonempty strings, required | included; A4/A5 binding |
| `calibration_record_digest` | SHA or `null` | included; production requires SHA |
| `allow_wire_body_hard_exclusion`, `allow_wire_body_confirmed`, `allow_review_routing_reduction`, `allow_source_specific_selector` | required bool, each exactly `false` | included; all stages |
| `configuration_purpose`, `configuration_is_runtime_complete`, `formal_completeness_allowed`, `membership_artifact_publication_allowed` | required purpose / bools | included; purpose-derived closure below |
| `configuration_digest` | derived strict SHA | excludes only itself |

Purpose is enforced as follows:

| Purpose | Wire binding | Runtime/completeness/publication rule |
|---|---|---|
| `PRODUCTION_APPROVED` | `wire_provider_interface_version` is a nonempty string and `wire_provider_config_digest` is a nonnull strict SHA. | `configuration_is_runtime_complete=true`; formal completeness and official membership artifact publication are permitted only if all later gates also pass. |
| `CALIBRATION_CANDIDATE` | Interface version is a nonempty string; `wire_provider_config_digest` may be `null`. | `configuration_is_runtime_complete=false`, `formal_completeness_allowed=false`, and `membership_artifact_publication_allowed=false`. |
| `TEST_ONLY` | Explicit test binding or `null` is allowed. | `configuration_is_runtime_complete=false`, `formal_completeness_allowed=false`, and `membership_artifact_publication_allowed=false`; it never produces a V4-usable formal artifact. |

`WIRE_NOT_APPLICABLE_PROVEN` can be emitted only by A2/A3 after the authoritative wire inventory, complete influence-domain geometry, and successful spatial query prove it. Configuration cannot create it. No production numeric default is created; reviewed policy/configuration digests carry later calibrated parameters.

## 6. B1 canonical decoded-point content specification

`FrvCanonicalPointContentSpec` fields are `spec_schema_version`, `spec_id`, `framing_version`, `row_order`, `ordered_field_specs`, `coordinate_frame`, `units`, `quantization_policy`, `missing_value_policy`, and `spec_digest`. `FrvCanonicalPointFieldSpec` fields are `field_name`, `semantic_role`, `dtype_code`, `byte_order`, `scale`, `offset`, `required`, and `null_policy`.

The first-version policy vocabulary is closed and exact:

```text
row_order:
ORIGINAL_INDEX_ASCENDING

byte_order:
LITTLE_ENDIAN

quantization_policy:
EXACT_TYPED_VALUE
SCALED_INTEGER

null_policy:
FORBID_NULL
VALIDITY_BYTE

missing_value_policy:
EXPLICIT_VALIDITY_BYTE
```

The first-version permitted fixed-width `dtype_code` values are exactly `<i8`, `<u8`, `<i4`, `<u4`, `<f4`, `<f8`, and `|u1`. `semantic_role` is a nonempty, schema-bound string drawn from the approved immutable point-content specification; it is not inferred from a field name. `ORIGINAL_INDEX_ASCENDING` requires `original_index` to be the mandatory first field with `semantic_role=ORIGINAL_INDEX`, `dtype_code=<i8`, `byte_order=LITTLE_ENDIAN`, `required=true`, and `null_policy=FORBID_NULL`. `EXACT_TYPED_VALUE` permits only the declared fixed-width representation with no additional quantization. `SCALED_INTEGER` is legal only for an integer `dtype_code` with finite nonzero `scale` and finite `offset`. `VALIDITY_BYTE` is legal only when `required=false`; `FORBID_NULL` is mandatory when `required=true`. No implementation may add a policy value or dtype without a new reviewed schema version.

The frozen point-content bytes are:

```text
b"FRVPCD1\0"
+ uint64_le(spec_header_length)
+ canonical_frv_json_bytes(spec.to_digest_dict())
+ ordered canonical row bytes
```

Rows are ascending `original_index`; `original_index` is the required first `<i8` field. Field dtypes are fixed-width little-endian without padding; finite floats are declared IEEE binary32/binary64, `-0.0` becomes `+0.0`, and nonfinite values fail. A nullable field prepends `uint8` validity (`0`/`1`) and has zero value bytes when null. A1 validates the specification but never reads points or calculates this content hash.

## 7. Source identity and the acyclic equivalence audit

`SourceIdentity` exact fields are:

```text
source_identity_schema_version
source_id
recorded_source_path
resolved_access_path
source_file_sha256
canonical_point_content_digest or null
canonical_point_content_spec_digest or null
canonical_point_content_spec_version or null
source_point_count
source_bounds_min_xyz
source_bounds_max_xyz
coordinate_frame
units
origin_context
relocation_audit
source_equivalence_audit or null
source_content_identity_digest
source_identity_record_digest
```

`source_file_sha256` is SHA-256 of original source-file bytes and remains distinct from canonical point-content identity. Source ID, coordinate frame, units, and all IDs/references are nonempty strings; count is a nonnegative non-bool integer; bounds/origin are fixed-length finite vectors. Canonical content digest, specification digest, and specification version are either all nonnull or all null.

`source_content_identity_digest` denotes source content and coordinate semantics independent of access paths and audit records. Its exact canonical JSON payload includes only:

```text
source_identity_schema_version
source_id
source_file_sha256
canonical_point_content_digest or null
canonical_point_content_spec_digest or null
canonical_point_content_spec_version or null
source_point_count
source_bounds_min_xyz
source_bounds_max_xyz
coordinate_frame
units
origin_context
```

It excludes `recorded_source_path`, `resolved_access_path`, `relocation_audit`, `source_equivalence_audit`, `source_content_identity_digest`, and `source_identity_record_digest`.

`source_identity_record_digest` denotes the complete audited `SourceIdentity` record. Its payload includes every content-identity field above, `recorded_source_path`, `resolved_access_path`, `relocation_audit.audit_digest`, `source_equivalence_audit.audit_digest or null`, and `source_content_identity_digest`. It excludes only `source_identity_record_digest`.

`FrvRelocationAudit` is a self-contained immutable public contract with exactly:

```text
audit_schema_version
path_style
recorded_path_raw
resolved_path_raw
recorded_path_lexical_normalized
resolved_path_lexical_normalized
resolved_final_path or null
resolution_method
case_comparison_mode
platform_name
run_id
audit_timestamp
content_identity_result
warning_flags
audit_digest
```

All required strings are nonempty. `resolved_final_path` is either a nonempty string or `null`. `warning_flags` is a known-bit `uint32` flag mask. `audit_digest` includes every listed field except itself. A1 validates the supplied lexical/audit record only: it never resolves a path, accesses the filesystem, follows a symlink/junction, or decides content identity. Path and audit fields never enter `source_content_identity_digest`; only `relocation_audit.audit_digest` enters `source_identity_record_digest`.

`FrvSourceEquivalenceAudit` exact fields are:

```text
audit_schema_version
prior_source_content_identity_digest
current_source_content_identity_digest
prior_source_file_sha256
current_source_file_sha256
canonical_point_content_spec_digest
canonical_point_content_digest
source_summary_comparison_digest
approval_reference
approver
approval_timestamp
decision_digest
approval_digest
audit_digest
```

All approval identifiers/references/timestamps are nonempty strings and every digest is strict SHA. Its digest payload includes all fields except `audit_digest`; it never contains the current source record digest. Therefore the dependency direction is content digest → equivalence-audit digest → source record digest, never source record digest → equivalence audit → same source record digest. A1 validates structure, formats, supplied cross-field bindings, and absence of a record-digest reference; it does not decide equivalence or hash content.

Semantic `artifact_id`, scope semantic identity, canonical inventory binding, membership semantic binding, cross-computer equivalence, and V4 original-point identity use `source_content_identity_digest`. Audit manifests, provenance, relocation/equivalence binding, and full-record round-trip may additionally use `source_identity_record_digest`.

## 8. Geometry encoding specification and real geometry contents

`FrvScopeGeometryEncodingSpec` replaces the prior geometry-spec name. Its exact fields are:

```text
encoding_schema_version
encoding_spec_id
coordinate_frame
units
origin_context_digest
boundary_rule_version
box_order_rule
float_encoding
byte_order
negative_zero_policy
box_framing_version
encoding_spec_digest
```

`encoding_spec_digest` is the self-digest of these specification fields and excludes only itself. It proves encoding rules, not real geometry.

The three real content digests are `scope_core_geometry_digest`, `scope_context_margin_geometry_digest`, and `scope_union_geometry_digest`. Each is SHA-256 of exactly:

```text
b"FRVGEO1\0"
+ uint64_le(canonical_header_length)
+ canonical_frv_json_bytes(header)
+ uint64_le(box_count)
+ ordered_box_bytes
```

The canonical header contains exactly `encoding_schema_version`, `encoding_spec_digest`, `coordinate_frame`, `units`, `origin_context_digest`, `boundary_rule_version`, and `geometry_role`. `geometry_role` is exactly one of `SCOPE_CORE`, `SCOPE_CONTEXT_MARGIN`, and `SCOPE_UNION` and is not interchangeable. Each ordered box contributes exactly `min_x`, `min_y`, `min_z`, `max_x`, `max_y`, `max_z` as IEEE-754 binary64 little-endian bytes, no padding; `-0.0` is normalized to `+0.0`; NaN/Inf are forbidden; and each axis requires `min < max`. Boxes are a nonoverlapping canonical union decomposition sorted lexicographically by the six-value tuple.

`FRVScopeDefinition` fields are `scope_schema_version`, `source_content_identity_digest`, `scope_id`, `region_id`, `scope_mode`, `geometry_encoding_spec_digest`, `scope_core_geometry_digest`, `scope_context_margin_geometry_digest`, `feature_halo_policy_digest`, `coordinate_frame`, `units`, `origin_context_digest`, `scope_construction_version`, `upstream_reference_digests`, `boundary_margin_policy_digest`, `raw_xyz_inclusion_rule_version`, `half_open_boundary_rule`, `scope_definition_digest`, `warnings`, and `limitations`. Core is upstream-evidence-supported area, context margin is formal decision expansion, and feature halo is feature-only: it never creates formal output geometry. A1 validates all references/roles/versions and the encoding specification; A3 canonicalizes real boxes and computes their content digests.

## 9. Approved hard rejection and coverage accounting

`FRVApprovedHardRejectionEvidence` is a separate immutable record with exactly:

```text
evidence_schema_version
source_content_identity_digest
canonical_inventory_key
canonical_inventory_record_digest
approved_rule_id
approved_rule_version
hard_non_vegetation_evidence_reference
hard_non_vegetation_evidence_digest
decision_digest
approver
approval_timestamp
approval_digest
safety_completeness_effect
evidence_digest
```

All IDs, versions, person fields, timestamps, and references are nonempty strings. Every digest is strict SHA. `approval_digest` binds the external approval record; `evidence_digest` is this contract's self-digest, so the two must differ. A1 validates structure, SHA format, `source_content_identity_digest`/canonical-key/canonical-record binding, and self-digest; it does not determine whether hard evidence is true. A3 produces the evidence, and A6/V4 validate and consume it. A missing approval field means the row is not an approved hard rejection.

`FRVScopeCoverageRecord` contains exactly:

```text
record_schema_version
source_content_identity_digest
canonical_inventory_key
canonical_inventory_record_digest
canonical_inventory_digest
inventory_disposition
required_inventory_record_digest or null
required_inventory_digest or null
approved_hard_rejection_evidence or null
upstream_candidate_reference
upstream_candidate_type
upstream_status
upstream_evidence_reference
scope_id or null
scope_construction_state
scope_coverage_state
failure_reason_flags
review_required
safety_completeness_effect
diagnostic_reference
out_of_policy_safety_approval_digest or null
record_digest
```

The record is a canonical-inventory row, not an inference from optional fields. Its disposition rules are exact:

| Disposition | Required values | Prohibited / route rule |
|---|---|---|
| `REQUIRED_FRV` | nonnull `required_inventory_record_digest` and `required_inventory_digest`; null approved-rejection evidence | route is `SCOPE_CREATED` or `SCOPE_MERGED` with `COVERED`, or `NOT_CREATED` with a coverage-blocked state and nonzero failure flags. |
| `APPROVED_HARD_REJECTION` | both required-inventory digests are null; nonnull valid `approved_hard_rejection_evidence` bound to this source/key/canonical record; `scope_coverage_state=COVERED`, `failure_reason_flags=NONE`, `review_required=false`, `safety_completeness_effect=PRESERVES_SAFETY_COMPLETENESS` | `scope_id=null`, `scope_construction_state=NOT_CREATED`; it is never in required inventory. |

`FRVScopeCoverageManifest` fields are `manifest_schema_version`, `source_content_identity_digest`, `configuration_digest`, `scope_construction_version`, `canonical_inventory_digest`, `required_inventory_digest or null`, `scope_union_id or null`, `scope_union_geometry_digest or null`, `records`, `canonical_inventory_count`, `required_inventory_count`, `approved_hard_rejection_count`, `scope_created_or_merged_count`, `coverage_blocked_count`, `canonical_inventory_complete`, `required_inventory_complete`, `approved_hard_rejection_complete`, `scope_coverage_complete`, `routing_complete`, `expected_semantic_artifact_id or null`, `provider_failure_reference_digests`, `stream_failure_reference_digests`, `identity_failure_reference_digests`, `warnings`, `limitations`, `provenance_digest`, and `manifest_digest`.

Records are immutable, canonical-key-lexicographically sorted, and each canonical key appears exactly once. Row and aggregate validation both enforce:

```text
canonical records = required-disposition records + approved-hard-rejection-disposition records
required-disposition records = scope-created-or-merged records + coverage-blocked records
```

The manifest derives each count from rows and rejects a supplied count mismatch. Missing, duplicate, unclassified, or disposition-inconsistent rows make completeness false and validation fail. Coverage failure never creates `EXCLUDED_NON_CANDIDATE`. The first-version manifest embeds rows; an external row descriptor requires a new reviewed schema version.

## 10. Manual override descriptor boundary

`FRVManualOverrideArtifact` is a self-contained immutable public contract with exactly:

```text
override_schema_version
source_content_identity_digest
override_id
requested_action
requested_original_indexes
requested_geometry_reference or null
requested_geometry_resolution_version or null
requester
request_timestamp
request_reason
authority_level
predecessor_artifact_digest or null
automatic_state_before_override
automatic_reason_flags
conflict_flags
reviewer or null
review_timestamp or null
approval_state
approval_reason or null
resulting_state or null
superseded_override_digest or null
override_digest
```

All IDs, people, timestamps, reasons, authority values, and references that are required in the active branch are nonempty strings; all digest fields are strict SHA-256 values. `requested_original_indexes` is an immutable sorted-unique tuple of nonnegative non-bool integers. An index request has a nonempty index tuple and both geometry fields `null`. A geometry request has an empty index tuple and nonnull geometry reference and resolution version. A root override has `predecessor_artifact_digest=null`; every successor has a nonnull predecessor digest.

Only `MANUAL_INCLUDE_APPROVED` may set `resulting_state=MANUAL_INCLUDE`, and only `MANUAL_EXCLUDE_APPROVED` may set `resulting_state=MANUAL_EXCLUDE`. `PENDING_REVIEW` and `MANUAL_OVERRIDE_REJECTED` require `resulting_state=null`. Approved states require nonnull `reviewer`, `review_timestamp`, and `approval_reason`; pending review requires those three fields to be `null`; a rejected request requires reviewer, review timestamp, and approval reason but no resulting state. Same-predecessor conflict, multiple-effective override, cycle, unresolvable predecessor, or artifact-integrity failure requires the corresponding conflict flag and prohibits a manual terminal result. `MANUAL_EXCLUDE` has no blanket priority over `MANUAL_INCLUDE`.

`override_digest` includes every listed field except itself. A1 validates supplied structure, exact-schema serialization, digest binding, and supplied chain-descriptor topology only; A5 executes effectiveness and conflict routing, and A6 publishes the result. No override bypasses source identity, scope, provider, stream, or artifact failure.

## 11. Exact payload array API and block descriptors

`FRVArtifactArrayDescriptor` fields are exactly `field_name`, `dtype_code`, `shape_rule`, `required`, `nullable`, `finite_required`, `enum_or_flag_schema_version or null`, and `storage_key`. It deliberately has no `array_digest`: this artifact-level schema describes representation, while array content is covered by each block's `block_digest` and the semantic artifact's `artifact_sha256`. A future per-array content digest requires a new block-level inventory and a new reviewed schema version. `FRVArtifactPayloadSchema` fields are exactly `payload_schema_version`, `block_schema_version`, `membership_state_schema_version`, `reason_flag_schema_version`, `quality_flag_schema_version`, `hag_evidence_schema_version`, `segmentation_hag_readiness_schema_version`, `payload_encoding_id`, `payload_encoder_version`, `payload_compression_algorithm`, `payload_compression_version`, `arrays`, `scope_union_reference_table`, `scope_union_reference_table_digest`, `primary_scope_reference_table`, `primary_scope_reference_table_digest`, `supporting_region_reference_table`, `supporting_region_reference_table_digest`, `manual_override_reference_table or null`, `manual_override_reference_table_digest or null`, and `payload_schema_digest`. The first-version values are exactly `payload_encoding_id="FRV-BLOCK-PAYLOAD-1"`, `payload_encoder_version="1"`, `payload_compression_algorithm="NONE"`, and `payload_compression_version="1"`; no physical compression is permitted.

The required point-aligned arrays are exactly:

| Array | Exact descriptor |
|---|---|
| `original_index` | `<i8[N]` |
| `membership_state` | `|u1[N]` |
| `reason_flags` | `<u4[N]` |
| `input_quality_flags` | `<u4[N]` |
| `hag_evidence_state` | `|u1[N]` |
| `segmentation_hag_readiness` | `|u1[N]` |
| `scope_union_id_reference` | `<u4[N]` |
| `primary_scope_id_reference` | `<u4[N]` |
| `supporting_region_offset` | `<u8[N]` |
| `supporting_region_count` | `<u4[N]` |
| `manual_override_reference` | `<u4[N]` |

`supporting_region_reference_table` is an artifact-level immutable `tuple[str, ...]`. Every entry is a nonempty normalized upstream-region reference; entries are lexicographically sorted and unique; its digest is stored in the payload schema; and it is never an object-dtype point array. For each point, `supporting_region_offset` and `supporting_region_count` select the consecutive half-open table range `[offset, offset + count)`. Both arrays have length `N`, no negative values, and read-only storage. `count` may be zero only with `offset=0`; otherwise `offset <= table_length` and `offset + count <= table_length`. The payload schema declares these exact dtype/shape requirements. A1 validates descriptors, tables, schema versions, required-array inventory, and rejects unknown/old array fields; A6 validates actual array bounds, contents, mutability, and point-level references.

`scope_union_id_reference` indexes the required scope-union table, `primary_scope_id_reference` indexes the lexicographically ordered primary-scope table, and `manual_override_reference` uses index zero as explicit null with positive indices into its digest-bound table. Optional arrays remain `raw_hag`, `clamped_hag`, HAG quality/reason flags, and `candidate_confidence` with their frozen finite/no-NaN rules; none changes membership validity.

`FRVArtifactBlockDescriptor` fields are `block_schema_version`, `payload_schema_digest`, `block_id`, `block_digest`, `original_index_start`, `original_index_stop`, `record_count`, `first_original_index or null`, `last_original_index or null`, `storage_reference`, `payload_schema_version`, `array_descriptor_digest`, and `block_descriptor_digest`. Descriptors are unique, ordered, nonoverlapping half-open ranges. Canonical block order is `original_index_start` ascending, then `original_index_stop` ascending, then `block_id` lexicographically. A1 validates descriptor metadata only; it does not open a block.

The exact encoded `block_payload_bytes` are uncompressed `FRV-BLOCK-PAYLOAD-1` bytes:

```text
b"FRVBLK1\0"
+ uint64_le(header_length)
+ canonical_frv_json_bytes({
    "block_schema_version": block_schema_version,
    "payload_schema_digest": payload_schema_digest,
    "block_id": block_id,
    "record_count": record_count,
    "present_field_names": sorted present descriptor field_name values
  })
+ for each present descriptor in ascending field_name order:
    uint64_le(field_name_utf8_length)
    field_name_utf8
    uint64_le(array_byte_length)
    exact_array_bytes
```

Every present array is a C-contiguous one-dimensional length-`N` array, encoded in its descriptor's exact declared dtype and little-endian byte order, without padding. The fixed-width one-byte unsigned dtype has no byte-order marker; all other required dtypes are exactly the ones in the required-array table. Optional absence is represented only by omission from `present_field_names` and its subsequent entry. No archive/container metadata, filename, directory, storage reference, mtime, host value, alternate codec, or object dtype is part of `block_payload_bytes`.

## 12. Artifact identity, exact payload byte domain, and manifests

`artifact_id` is `frv-membership-` plus SHA-256 of canonical JSON:

```text
{
  "artifact_schema_version": artifact_schema_version,
  "algorithm_version": algorithm_version,
  "decision_rule_version": decision_rule_version,
  "source_content_identity_digest": source_content_identity_digest,
  "scope_union_id": scope_union_id,
  "configuration_digest": configuration_digest
}
```

It never uses paths, relocation audit, equivalence audit, or `source_identity_record_digest`; a path-only relocation cannot change it.

`artifact_sha256` is SHA-256 of the normalized ordered membership payload, not a checksum of a manifest containing its own checksum. Its exact byte domain is:

```text
b"FRVART1\0"
+ uint64_le(header_length)
+ canonical_frv_json_bytes(header)
+ for each block in canonical block order:
    uint64_le(block_id_utf8_length)
    block_id_utf8
    uint64_le(block_payload_byte_length)
    exact_block_payload_bytes
```

The header contains exactly `artifact_schema_version`, `algorithm_version`, `decision_rule_version`, `artifact_id`, `source_content_identity_digest`, `scope_union_id`, `configuration_digest`, `payload_schema_digest`, `block_schema_version`, `block_count`, and `ordered_block_ids`. It excludes `artifact_sha256`, `manifest_digest`, `binding_digest`, storage paths, recorded/resolved source paths, file mtimes, and host-specific information. `block_payload_bytes` are the exact encoded bytes fixed by the payload/block schema. The first version uses that exact encoded byte domain: encoder, compression algorithm, and compression version are fixed by block schema. Thus different physical encoding versions change `artifact_sha256` but not semantic `artifact_id`. `block_digest` uses the same exact encoded block-payload bytes. Detached checksum files are non-authoritative copies only.

`FullResolutionVegetationMembershipArtifact` exact fields are:

```text
artifact_schema_version
algorithm_version
decision_rule_version
artifact_id
source_content_identity_digest
source_identity_record_digest
source_file_sha256
canonical_point_content_digest or null
canonical_point_content_spec_digest or null
scope_coverage_manifest_digest
scope_union_id
scope_construction_version
scope_union_geometry_digest
canonical_inventory_digest
required_inventory_digest or null
configuration_digest
payload_schema_digest
block_schema_version
artifact_sha256
block_descriptors
source_point_count
unique_in_scope_point_count
evaluated_point_count
verified_vegetation_candidate_count
excluded_non_candidate_count
review_required_membership_count
manual_include_count
manual_exclude_count
unasserted_in_scope_count
canonical_inventory_count
required_inventory_count
approved_hard_rejection_count
scope_created_or_merged_count
coverage_blocked_count
canonical_inventory_complete
required_inventory_complete
approved_hard_rejection_complete
scope_coverage_complete
routing_complete
membership_decision_complete
provider_digest_references
override_digest_references
block_digest_references
warning_digest
limitation_digest
provenance_digest
manifest_digest
```

All digest-reference tuples are immutable and canonical-lexicographically sorted. It never embeds payload arrays or provider objects. Five terminal counts sum exactly to `evaluated_point_count`; successful artifacts require positive `unique_in_scope_point_count == evaluated_point_count`, zero unasserted in-scope points, and at least one nonempty block.

`manifest_digest` is a metadata self-digest. Its payload includes `artifact_sha256`, all block descriptors, all semantic bindings, all counts/flags/references, and every other artifact field; it excludes only `manifest_digest`. It is distinct from payload-domain `artifact_sha256`.

Publication order is mandatory:

```text
A6 publishes exact block payload bytes
→ compute block_digest
→ compute artifact_sha256
→ construct artifact manifest
→ compute manifest_digest
→ construct binding record
→ compute binding_digest
```

`FRVArtifactBindingRecord` fields are exactly `binding_schema_version`, `source_content_identity_digest`, `source_identity_record_digest`, `scope_union_id`, `configuration_digest`, `semantic_artifact_id`, `scope_coverage_manifest_digest`, `membership_artifact_manifest_digest`, `membership_artifact_sha256`, `binding_created_by_stage`, and `binding_digest`. Its validator checks content semantic identity, record audit binding, artifact-ID derivation, source/scope/config equality, artifact SHA equality with the artifact manifest, and a self-digest excluding only `binding_digest`. A3's scope manifest never embeds A6's later artifact manifest/digests, so this sequence has no cycle.

## 13. Digest inclusion table

| Contract/content | Digest | Exact include/exclude rule |
|---|---|---|
| `FrvConfiguration` | `configuration_digest` | all fields except itself |
| `FrvCanonicalPointContentSpec` | `spec_digest` | all specification fields except itself |
| `FrvScopeGeometryEncodingSpec` | `encoding_spec_digest` | all encoding fields except itself |
| Source content identity | `source_content_identity_digest` | content/count/bounds/frame/units/origin/canonical spec; excludes paths, audits, and both identity digest fields |
| Full SourceIdentity record | `source_identity_record_digest` | content fields, paths, relocation audit digest, equivalence audit digest/null, content digest; excludes only itself |
| `FrvSourceEquivalenceAudit` | `audit_digest` | all listed audit fields; excludes only itself and contains no current record digest |
| Real core/context/union geometry | corresponding geometry content digest | exact `FRVGEO1` framed bytes in §8 |
| Hard-rejection evidence | `evidence_digest` | all evidence fields except itself; `approval_digest` is included |
| Coverage row/manifest, scope definition, override, payload schema, block descriptor | respective self-digest | all declared fields except that self-digest |
| Membership payload | `artifact_sha256` | exact `FRVART1` framed bytes in §12 |
| Membership manifest | `manifest_digest` | all metadata including `artifact_sha256`; excludes only itself |
| Binding record | `binding_digest` | all fields except itself |

Raw-file, canonical point-content, geometry-content, block-payload, artifact-payload, and metadata self-digest domains are never substitutable.

## 14. B5 empty-case truth table and B7 schema policy

| Case | Legal result | Completeness/publication rule |
|---|---|---|
| Canonical inventory is zero | required/approved counts are zero, no records, union is null | inventory/routing may be structurally complete; no membership artifact and no V4 runtime input claim |
| All rows approved hard rejection | canonical is positive, required is zero, approved equals canonical, each row has complete evidence, union is null | coverage manifest may be complete; no admitted candidate or membership artifact |
| All required rows blocked | required positive, created/merged zero, blocked equals required | scope/routing/membership completeness are false; no formal artifact |
| Legal union with zero original rows | not a successful scope outcome | A3 blocks it as geometry/mapping failure; no zero-point artifact |
| Legal membership artifact | positive in-scope/evaluated counts, five-state conservation, nonempty block | all binding and completeness gates must pass |

First version is `STRICT_EXACT_SCHEMA`: exact version match, no unknown/missing fields, no silent migration, and no ignored forward fields. A migration requires a new explicit schema/version, dedicated loader, digest rule, and migration tests.

## 15. Pure validator inventory

Every validator raises the frozen `ValueError` failure type with stable English failure text. “Cannot validate” is a hard A1 boundary.

| Validator | Input; can validate | Cannot validate | Producer → consumer; failure tests |
|---|---|---|---|
| `validate_exact_schema_fields`, `validate_frv_enum_value`, `validate_frv_flag_mask` | mappings/scalars; exact fields, versions, enum/flag values and bool rejection | business meaning or migration | all → all; missing/unknown/version/bit tests |
| `validate_configuration`, `validate_production_wire_provider_binding` | `FrvConfiguration`; purpose booleans, SHA/strings and wire binding | provider inventory, geometry, query success, non-applicability truth | config → all; production interface/digest/null, calibration/test closure, config-created-nonapplicability tests |
| `validate_canonical_point_content_spec` | B1 field/framing specification | source read/row encoding/hash | A2/A6 → all; field/order/endian/null/negative-zero tests |
| `validate_source_content_identity_digest` | `SourceIdentity` content payload/recomputed supplied digest | source bytes/content equivalence | A2 → A3/A6/V4; path/audit independence and changed-content tests |
| `validate_source_identity_record_digest` | complete identity record/recomputed supplied record digest | filesystem/audit truth | A2 → A6/V4; audit-change/round-trip tests |
| `validate_source_equivalence_audit_without_cycle` | equivalence audit fields, content bindings, no record-digest reference | approve equivalence or compare sources | A2 → A6/V4; absent-current-record-digest and no-cycle tests |
| `validate_scope_geometry_encoding_spec`, `validate_scope_geometry_content_digest_reference` | encoding spec and scope references/role values | construct boxes, canonicalize union, calculate geometry bytes | A3 → A6/V4; spec-v-content, role, NaN/Inf/decomposition reference tests |
| `validate_inventory_disposition`, `validate_approved_hard_rejection_evidence` | coverage row/evidence; exact exclusive disposition and approval/canonical binding | inventory discovery or hard-evidence truth | A3 → A6/V4; missing fields, mismatches, mixed disposition tests |
| `validate_scope_coverage_record`, `validate_scope_inventory_conservation` | rows/manifest; order, uniqueness, state routing, both equations and derived counts | produce scopes/routes | A3 → A6/V4; count/row/blocked/aggregate mismatch tests |
| `validate_manual_override_artifact`, `validate_override_chain_descriptors` | descriptor/chain topology and approval separation | execute effective override | A5 → A6/V4; request/approval/cycle/conflict tests |
| `validate_supporting_region_ragged_reference_schema`, `validate_payload_schema` | schema descriptors/tables; exact arrays, dtypes/shapes, sorted unique table, no object descriptor | open arrays or point bounds/content | A6 → V4; all supporting-array schema tests |
| `validate_block_descriptor_inventory` | descriptor IDs/ranges/order/schema | payload rows/bytes | A6 → V4; order/overlap/range tests |
| `validate_artifact_payload_digest_domain` | artifact/block headers, canonical block order and declared byte-domain configuration | encode/read block payload bytes | A6 → V4; header/order/path/encoding-domain tests |
| `validate_membership_state_count_conservation`, `validate_empty_inventory_scope_artifact_case` | manifest counts, B5 relations and five-state sum | create membership/source scan | A3/A6 → V4; truth-table/conservation tests |
| `validate_artifact_binding_record` | binding/manifest values, semantic ID and all digest equality relations | publish/replay artifact bytes | A6 → V4; SHA/manifest/binding mismatch tests |

## 16. Required future tests and implementation sequence

Future contract tests must cover the existing B1/B3/B5/B7 and frozen prohibitions plus the following exact matrix:

| Area | Required tests |
|---|---|
| Hard rejection | disposition required; mutual exclusion; missing rule ID/version, decision digest, approver, timestamp, approval digest; canonical-record mismatch; required digest in rejection row; aggregate/row count mismatch |
| Source identity | content digest unchanged by path/relocation/equivalence audit changes; record digest changes with audits; altered content changes content digest; audit lacks current record digest; relocation-only artifact ID stability; exact round-trip without a cycle |
| Relocation audit | exact field inventory; required/nonempty values; nullable final path; known warning bits; digest changes with audit evidence; content digest remains unchanged; filesystem/path resolution is never performed by A1 |
| Canonical point policy vocabulary | exact closed row-order, byte-order, quantization, null, missing-value, and dtype values; illegal policy/dtype rejected; original-index first-field invariant; scaled-integer type/scale rules |
| Manual override contract | exact field inventory; index-versus-geometry exclusivity; root/successor predecessor rules; pending/approved/rejected reviewer fields; resulting-state compatibility; conflict prohibits terminal result; self-digest and unknown-field rejection |
| Safety completeness enum | included in public inventory/exports tests; exact two string values; invalid value rejected; covered/rejected/blocked row compatibility |
| Geometry | spec digest differs from content digest; same spec/different boxes differs; reordered same boxes agrees; changed box differs; negative-zero canonicalization; NaN/Inf rejected; overlapping/noncanonical decomposition rejected or blocked by A3; roles are non-interchangeable |
| Artifact payload digest | no self-reference; canonical block order; byte mutation changes block/artifact hashes; storage-path mutation does not change artifact hash; metadata mutation changes manifest only; exact encoded bytes included; manifest includes artifact SHA; binding matches both |
| Wire configuration | production missing interface/digest/null fails; calibration null digest blocks completeness/publication; test null binding blocks official artifact; configuration cannot create `WIRE_NOT_APPLICABLE_PROVEN` |
| Supporting-region arrays and descriptor ownership | all required arrays; offset/count dtypes; table sorted/unique; zero count requires zero offset; offset and addition overflow; length mismatch; object dtype descriptor rejection; old logical field rejected; `array_digest` is rejected as an unknown `FRVArtifactArrayDescriptor` field; block/artifact digests remain the only v1 content-digest owners |

After authorization, only the following scoped future files would be changed: `core/contracts/frv.py` for contracts/validators, `core/contracts/__init__.py` for reviewed exports, and focused contract/serialization tests. The implementation sequence is enums/schema helpers; configuration/B1; source identity/audits; encoding spec/scope/coverage; override descriptor; payload/block/artifact; binding/B5; then only authorized exports/tests/review. Each step stops before A2–A6 production behavior.

## 17. Final plan review closure

The final micro-revision is self-contained and closes the last public-API ambiguities: `FrvRelocationAudit` and `FRVManualOverrideArtifact` now have exact field inventories; B1 policy values and permitted dtypes are closed; `FRVArtifactArrayDescriptor` has no ambiguous artifact-level `array_digest`; and `FrvSafetyCompletenessEffect` is part of the public inventory and test plan. No implementation choice remains for these contracts.

```text
FRV-A1 scope alignment: PASS
Contract inventory completeness: PASS
Digest acyclicity: PASS
Public contract self-containment: PASS
Canonical point policy closure: PASS
Array digest ownership: PASS
Frozen-design semantic conflicts: NONE
Remaining plan blockers: NONE
Ready for explicit FRV-A1 code authorization: YES
FRV-A1 code implementation authorized by this document: NO
```

## 18. Acceptance and authorization

The plan passes only when every listed contract, field, required/optional rule, digest inclusion rule, validator boundary, producer/consumer hand-off, and test is implemented exactly; all focused/full regressions pass; no dependency or V2/V3 semantic changes occur; and independent review approves code. Passing A1 does not itself produce an artifact, authorize later stages, or make V4 runtime usable.

```text
FRV design status:
DESIGN-FROZEN

FRV-A1 implementation plan status:
IMPLEMENTATION-PLAN-REVIEW-CANDIDATE

FRV-A1 code implementation authorized:
NO

FRV-A2 or later planning/implementation authorized:
NO
```
