# FRV-A2 Detailed Implementation Plan

**Status:** `IMPLEMENTATION-PLAN-FROZEN`  
**Normative FRV design:** `docs/full-resolution-vegetation-membership-design.md` (`DESIGN-FROZEN`, 2026-08-01)  
**Downstream V4 design:** `docs/v4-tree-segmentation-design.md` (`DESIGN-FROZEN`, 2026-08-01)  
**FRV-A1 baseline:** `baae028e73e5ff48e28174501997ab50089b7849`  
**Plan authorizes implementation:** `NO`

**Plan freeze review:** `PASS_READY_FOR_PLAN_FREEZE`

**Plan frozen date:** `2026-08-02`

**Plan authorizes implementation:** `NO`

## 1. Authority, baseline, and FRV-A2 boundary

This document is an implementation plan only. It records the repository evidence, the
approved FRV-A2 contract shape, the validation model, prerequisite provider boundaries,
test requirements, and acceptance gates. It does not authorize production code, tests,
provider implementation, FRV-A3 work, V4 work, or Git writes.

The normative authorities are:

- `docs/full-resolution-vegetation-membership-design.md`;
- `docs/v4-tree-segmentation-design.md`;
- the reviewed FRV-A1 public contracts at commit
  `baae028e73e5ff48e28174501997ab50089b7849`.

### 1.1 Required baseline

A future implementation session must start from:

| Check | Required value |
|---|---|
| Branch | `feature/algorithm` |
| HEAD | `baae028e73e5ff48e28174501997ab50089b7849` |
| Remote divergence | `0 0` |
| Latest commit | `baae028 feat: add reviewed FRV-A1 contracts` |
| Project worktree | Clean except the explicitly excluded local `.claude/` directory and this plan before it is committed |

`.claude/` is an excluded local tool directory. It must not be read, modified, deleted,
staged, committed, or treated as a project baseline failure.

### 1.2 Meaning of FRV-A2

FRV-A2 is the second stage of the full-resolution vegetation-membership pipeline. It is
not the existing preprocessing stage named A2A, and it is not the existing preprocessing
stage named A2B.

FRV-A2 has two separable responsibilities:

1. **A2 framework:** public readiness/failure/domain contracts, three A2 digest domains,
   phased fail-closed validation, strict serialization, and deterministic test-provider
   validation.
2. **A2 production integration:** validate and bind the concrete full-resolution
   source/scan, HAG, geometry, and wire providers, then return the exact immutable runtime
   views that produced the serializable readiness record.

FRV-A2 validates inputs and establishes whether downstream FRV-A3 may begin consuming
them. It does not produce FRV membership.

### 1.3 Frozen stage map

| Stage | Responsibility | FRV-A2 relationship |
|---|---|---|
| FRV-A1 | Immutable contracts, source identity, configuration, canonical serialization, artifact schemas, and pure validators | Reuse without weakening or duplicating |
| FRV-A2 | Provider identity, integrity, domain, readiness, cross-provider alignment, serializable readiness, and validated runtime handoff | This plan |
| FRV-A3 | Candidate inventory, hard rejection, scope, coverage, union, partition, and wire influence inventory | Downstream; no A3 output is created here |
| FRV-A4 | Graph, tile, halo, and supporting-region evidence | Out of scope |
| FRV-A5 | Point membership, review transitions, and manual override | Out of scope |
| FRV-A6 | Artifact publication, conservation, safety handoff, and V4 adapter | Out of scope |
| V4 | Exact replay and single-tree segmentation | Out of scope |

FRV-A2 must stop before candidate inventory, hard rejection, scope geometry, coverage,
wire influence inventory, membership, publication, or V4 transformation.

## 2. Repository evidence and public-provider inventory

Repository facts are classified as:

- `REPOSITORY_PROVEN`: a public type, loader, field, behavior, or test exists at the
  FRV-A1 baseline.
- `A2_SUBTASK`: required work that belongs inside the FRV-A2 framework.
- `SEPARATE_PREREQUISITE`: a provider contract/producer boundary that must be planned,
  reviewed, and implemented separately before production A2 integration.
- `PRODUCTION_GATE_BLOCKER_ONLY`: a missing fact that blocks production readiness but
  does not block independent A2 framework work.
- `OUT_OF_SCOPE`: work owned by FRV-A3 or later.

### 2.1 Inspected repository areas

| Area | Files |
|---|---|
| Frozen designs and A1 plan | `docs/full-resolution-vegetation-membership-design.md`, `docs/v4-tree-segmentation-design.md`, `docs/frv-a1-implementation-plan.md` |
| FRV-A1 contracts | `core/contracts/frv.py`, `core/contracts/__init__.py` |
| FRV-A1 tests | `tests/contracts/test_frv_contracts.py`, `tests/contracts/test_frv_serialization.py` |
| Source/header/chunk I/O | `core/contracts/pointcloud.py`, `algorithms/io/discovery.py`, `algorithms/io/headers.py`, `algorithms/io/chunks.py`, `algorithms/io/__init__.py` |
| Existing geometry | `core/contracts/features.py`, `algorithms/features/geometry.py`, `algorithms/features/__init__.py` |
| Existing HAG | `core/contracts/ground.py`, `algorithms/features/ground.py`, `algorithms/features/__init__.py` |
| Vegetation evidence | `algorithms/vegetation/candidates.py`, `instances.py`, `associations.py`, `tree_inputs.py`, `algorithms/vegetation/__init__.py` |
| Wire evidence | `core/contracts/wire_models.py`, `algorithms/wires/reconstruct.py`, `algorithms/wires/separate.py`, `algorithms/wires/__init__.py` |
| Integration provenance | `core/contracts/i1.py`, `algorithms/integration/i1.py` |
| Loader and integration tests | `tests/io`, `tests/features`, `tests/vegetation`, `tests/wires`, `tests/integration` |

No private helper is an approved FRV-A2 input.

### 2.2 Corrected public-provider inventory

| Category | Public boundary | Repository-proven facts | FRV-A2 disposition |
|---|---|---|---|
| Source/header metadata | `algorithms.io.discover_las_sources`, `read_las_header_summary`, `read_las_header_summary_for_source`, `PointCloudSource`, `LasHeaderSummary` | Source/path metadata, byte size, LAS version, point format, point count, bounds, scale/offset, dimensions, and availability flags. `PointCloudSource.sha256` is optional and is not produced as a complete FRV source-identity boundary. | Partial supporting metadata. The identity-bound source/scan provider is `SEPARATE_PREREQUISITE`. |
| Existing A2A geometry | `algorithms.features.load_geometry_features`, `GeometryFeatureResult`, `PointFeatureSet` | `scene_simple_geometry_features` is an output basename/artifact naming fact, not a schema version. `a2a-local-pca-v1` is the algorithm version. `GeometryFeatureResult` has no independent `schema_version`. Source/manifest IDs, origin, original/sample indices, artifact SHA/size, exact arrays, dtype/finite checks, and repository-proven frozen/read-only array behavior exist. | Supporting evidence only. The full-resolution geometry provider is `SEPARATE_PREREQUISITE`. |
| Existing A2B HAG | `algorithms.features.load_ground_estimation`, `GroundEstimationResult`, `HeightAboveGroundSet` | `a2b-grid-quantile-idw-v1` is the algorithm version. There is no independent FRV `schema_version` field. Source/manifest IDs, original/sample indices, ground/HAG arrays, artifact SHA/size, exact arrays, finite checks, quality flags, and repository-proven frozen/read-only array behavior exist. | Supporting evidence only. The full-resolution HAG provider is `SEPARATE_PREREQUISITE`. |
| V1 candidates | `algorithms.vegetation.load_vegetation_candidates` | Candidate evidence, source/manifest IDs, origin, original/sample indices, artifact integrity, counts, formulas, and review metadata exist. | `SUPPORTING_EVIDENCE`; never a formal FRV provider or membership authority. |
| V2 instances | `algorithms.vegetation.load_vegetation_instances` | Instance/cluster evidence, source/manifest IDs, origin, original/sample indices, artifact integrity, statistics, and readiness metadata exist. | `SUPPORTING_EVIDENCE`; never a formal FRV provider or FRV override authority. |
| V2A associations | `algorithms.vegetation.load_vegetation_associations` | Pair-level cluster evidence, source/manifest IDs, artifact integrity, review status, confidence, and reasons exist. | `SUPPORTING_EVIDENCE` with `PAIR_SUBSET`; not a point-domain substitute. |
| W1 wire models | `algorithms.wires.load_wire_models`, `WireModelSet` | Centreline/segment/sample evidence, wire IDs, status/method/quality, source/manifest IDs, artifact integrity, parameter domains, and offsets exist. | `SUPPORTING_EVIDENCE`; never `WIRE_BODY_CONFIRMED`. |
| Wire separation | `algorithms.wires.run_wire_separation`, `WireSeparationResult` | A public producer result exists, but no public identity-bound artifact loader was found. | Authoritative wire provider/public loader is `SEPARATE_PREREQUISITE`. |
| I1 bundle | `algorithms.integration.load_i1_bundle`, `ValidatedI1Bundle` | Artifact references, JSON/NPZ SHA and size checks, source/manifest IDs, origin, frame/units, and exact referenced-artifact provenance exist. | `SUPPORTING_EVIDENCE` with `ARTIFACT_REFERENCE`; not `SourceIdentity` and not full-resolution coverage. |
| FRV-A1 | `core.contracts.frv` and package exports | `SourceIdentity`, `FrvConfiguration`, canonical JSON/digest utilities, strict validators, errors, reasons, quality flags, HAG evidence states, and production wire-binding rules exist. | Required contract foundation. |

### 2.3 Loader mutability facts

The plan must not assume a uniform array immutability guarantee across every legacy
loader.

- A2A and A2B loaders have repository-proven frozen/read-only array behavior.
- V1, V2, V2A, W1, and I1 behavior must be recorded individually by their adapters.
- A runtime adapter may create a no-copy read-only view:

```python
readonly_view = array.view()
readonly_view.setflags(write=False)
```

It must not modify the provider array and must not create a second full-size payload
array.

## 3. FRV-A2 objectives and non-goals

### 3.1 Objectives

FRV-A2 must:

1. validate only approved public provider boundaries;
2. validate `SourceIdentity` and `FrvConfiguration` using FRV-A1 semantics;
3. validate artifact structure and integrity before semantic array use;
4. bind providers to source-content identity and source-identity-record identity;
5. validate frame, units, origin, transform, domain kind, domain count, and index-domain
   digests;
6. distinguish formal providers from supporting evidence;
7. validate HAG, geometry, scan, and wire provider readiness without manufacturing
   absent facts;
8. use deterministic phased failure collection;
9. produce a serializable `FrvA2ReadinessRecord`;
10. return the exact validated runtime provider views in `ValidatedFrvA2Inputs`;
11. support cross-process `load_frv_a2_inputs` with reload-and-revalidate semantics;
12. express `READY_FOR_A3`, `READY_FOR_A3_WITH_LIMITATIONS`, or `BLOCKED` without
    producing A3 inventory or scope.

### 3.2 Non-goals

FRV-A2 must not:

- create `SourceIdentity`;
- scan LAS/LAZ point payload bytes to backfill missing identity;
- compute source raw-file or canonical point-content identity;
- produce HAG, geometry, source/scan, or wire provider payloads;
- reinterpret A2A, A2B, V1, V2, V2A, W1, or I1 as formal FRV providers;
- reorder, truncate, silently drop, interpolate, nearest-neighbor match, or repair
  provider rows;
- create candidate inventory, hard rejection, scope, union, partition, or coverage;
- determine whether a region is inside or outside a wire influence domain;
- emit `WIRE_NOT_APPLICABLE_PROVEN` or a global `NOT_APPLICABLE` readiness;
- emit `WIRE_BODY_CONFIRMED`;
- build tiles, halos, graphs, membership, manual overrides, FRV artifact payloads,
  publication records, V4 adapters, CHM, or tree segmentation.

## 4. Provider-gap classification and implementation tracks

### 4.1 Fixed classification

| Gap | Classification | Meaning |
|---|---|---|
| Identity-bound source/scan provider | `SEPARATE_PREREQUISITE` | Blocks production integration, not A2 framework work |
| Full-resolution HAG provider | `SEPARATE_PREREQUISITE` | Blocks production integration, not A2 framework work |
| Full-resolution geometry provider | `SEPARATE_PREREQUISITE` | Blocks production integration, not A2 framework work |
| Authoritative influence-domain wire provider/public loader | `SEPARATE_PREREQUISITE` | Blocks the applicable production wire gate, not A2 framework work |
| A2 readiness/failure/domain/result contract | `A2_SUBTASK` | Must be implemented inside FRV-A2 |
| Missing production provider required by a selected production configuration | `PRODUCTION_GATE_BLOCKER_ONLY` | The framework may exist, but the selected run is `BLOCKED` |

A separate prerequisite is not permission to hide provider production inside
`algorithms/vegetation/frv_inputs.py`.

### 4.2 A2 framework track

The A2 framework can be implemented independently after separate authorization:

- A2-C1: public readiness/failure/domain contracts;
- A2-C2: three approved A2 digest domains;
- A2-C3: phased validation framework;
- A2-C4: deterministic synthetic/test-provider validation;
- A2-C5: serialization and strict negative tests.

### 4.3 Provider prerequisite track

Separate plans are required for:

- P-SOURCE: identity-bound source/scan provider;
- P-HAG: full-resolution HAG provider;
- P-GEOMETRY: full-resolution geometry provider;
- P-WIRE: authoritative influence-domain wire provider and public loader.

### 4.4 Production integration track

Production integration waits for the applicable prerequisite contracts:

- A2-I1: concrete public provider adapters;
- A2-I2: `ValidatedFrvA2Inputs`;
- A2-I3: `load_frv_a2_inputs` reload-and-revalidate;
- A2-I4: FRV-A3 runtime handoff.

## 5. Public contract proposal

The A2 public contract belongs in `core/contracts/frv_a2.py`. Every public enum below is
a `str, Enum`; canonical JSON and every A2 digest use its exact lower-case `.value`.
Declaration order is fixed and is used for deterministic sorting.

### 5.1 Exact enums

```text
FrvA2ProviderRole(str, Enum):
  SOURCE_SCAN = "source_scan"
  HAG = "hag"
  GEOMETRY = "geometry"
  WIRE = "wire"
  PROVENANCE = "provenance"

FrvA2EvidenceClass(str, Enum):
  FORMAL_PROVIDER = "formal_provider"
  SUPPORTING_EVIDENCE = "supporting_evidence"

FrvA2ProviderReadiness(str, Enum):
  VALIDATED = "validated"
  REVIEW_REQUIRED = "review_required"
  BLOCKED = "blocked"
  SUPPORTING_ONLY = "supporting_only"

FrvA2AggregateReadiness(str, Enum):
  READY_FOR_A3 = "ready_for_a3"
  READY_FOR_A3_WITH_LIMITATIONS = "ready_for_a3_with_limitations"
  BLOCKED = "blocked"

FrvA2DomainKind(str, Enum):
  FULL_SOURCE = "full_source"
  DECLARED_SUBSET = "declared_subset"
  SAMPLED_SUBSET = "sampled_subset"
  CANDIDATE_SUBSET = "candidate_subset"
  CLUSTER_SUBSET = "cluster_subset"
  PAIR_SUBSET = "pair_subset"
  ARTIFACT_REFERENCE = "artifact_reference"
  INFLUENCE_DOMAIN = "influence_domain"

FrvA2FailurePhase(str, Enum):
  STRUCTURE_INTEGRITY = "structure_integrity"
  IDENTITY_CONTEXT = "identity_context"
  DOMAIN_ALIGNMENT = "domain_alignment"
  SEMANTIC_READINESS = "semantic_readiness"
  CROSS_PROVIDER = "cross_provider"
  PRODUCTION_POLICY = "production_policy"

FrvA2ArtifactComponentKind(str, Enum):
  BUNDLE_MANIFEST_JSON = "bundle_manifest_json"
  MANIFEST_JSON = "manifest_json"
  PAYLOAD_NPZ = "payload_npz"
  SOURCE_FILE = "source_file"
  REFERENCED_MANIFEST_JSON = "referenced_manifest_json"
  REFERENCED_PAYLOAD_NPZ = "referenced_payload_npz"

FrvA2ReviewCode(str, Enum):
  FORMAL_PROVIDER_REVIEW_REQUIRED = "formal_provider_review_required"
  CALIBRATION_WIRE_CONFIG_UNBOUND = "calibration_wire_config_unbound"

FrvA2LimitationCode(str, Enum):
  NONPRODUCTION_CONFIGURATION = "nonproduction_configuration"
  FORMAL_COMPLETENESS_FORBIDDEN = "formal_completeness_forbidden"
  MEMBERSHIP_PUBLICATION_FORBIDDEN = "membership_publication_forbidden"
  OPTIONAL_TEST_WIRE_PROVIDER_ABSENT = "optional_test_wire_provider_absent"
  FORMAL_PROVIDER_REVIEW_REQUIRED = "formal_provider_review_required"
  CALIBRATION_WIRE_CONFIG_UNBOUND = "calibration_wire_config_unbound"
```

A2 does not define or emit a global `NOT_APPLICABLE` readiness.

### 5.2 Fixed semantic-role mappings

`FrvA2ProviderRole` is always the semantic role; it is never replaced by a generic
“supporting” role.

The following combinations are mandatory and validators reject every alternative:

```text
A2B:
  provider_role = HAG
  evidence_class = SUPPORTING_EVIDENCE
  provider_readiness = SUPPORTING_ONLY

A2A:
  provider_role = GEOMETRY
  evidence_class = SUPPORTING_EVIDENCE
  provider_readiness = SUPPORTING_ONLY

W1:
  provider_role = WIRE
  evidence_class = SUPPORTING_EVIDENCE
  provider_readiness = SUPPORTING_ONLY

I1:
  provider_role = PROVENANCE
  evidence_class = SUPPORTING_EVIDENCE
  provider_readiness = SUPPORTING_ONLY
```

General invariants:

- every `SUPPORTING_EVIDENCE` record must use `SUPPORTING_ONLY`;
- every `FORMAL_PROVIDER` record must not use `SUPPORTING_ONLY`;
- supporting evidence never satisfies a formal-provider role gate;
- `PROVENANCE` is reserved for cross-cutting reference-only evidence such as I1.

### 5.3 `FrvA2FailureRecord`

Exact fields:

```text
phase: FrvA2FailurePhase
provider_role: FrvA2ProviderRole
provider_key: str
field_name: str
error_code: FrvErrorCode
reason_flags: int
detail: str
```

Rules:

- frozen and immutable;
- exact field inventory;
- `provider_key` and `field_name` match
  `^[a-z][a-z0-9_]{0,63}$`;
- `reason_flags` is a nonnegative non-bool integer and serializes as a canonical JSON
  integer;
- `error_code` reuses `FrvErrorCode`;
- this plan authorizes no new `FrvErrorCode` value;
- `detail` is not free text. It must equal:

```text
"{phase.value}:{provider_role.value}:{provider_key}:{field_name}:{error_code.value}:{reason_flags}"
```

Raw exception messages, paths, tracebacks, host-specific text, and runtime values are
forbidden.

### 5.4 `FrvA2ArtifactComponentBinding`

Exact fields:

```text
component_key: str
component_kind: FrvA2ArtifactComponentKind
sha256: str
size_bytes: int
```

Rules:

- frozen, immutable, exact schema;
- `component_key` matches `^[a-z][a-z0-9_]{0,63}$`;
- `sha256` is a strict lower-case SHA-256 hex string of the exact raw bytes of that
  component;
- `size_bytes` is the exact raw byte length, a nonnegative non-bool integer;
- components are unique by `component_key`;
- the tuple is sorted by
  `(FrvA2ArtifactComponentKind declaration order, component_key)`;
- paths, mtimes, hosts, filenames, memory addresses, and runtime object IDs are excluded.

`artifact_reference_id` is derived, never accepted as a free string:

```text
"frv-a2-ref-v1|"
+ provider_role.value
+ "|"
+ provider_key
+ "|"
+ ";".join(
    component_kind.value
    + ","
    + component_key
    + ","
    + sha256
    + ","
    + canonical_decimal_size_bytes
    for each canonical artifact component
  )
```

`canonical_decimal_size_bytes` is the base-10 representation with no sign, no leading
zero except the value `0`, and no whitespace.

### 5.5 Exact artifact layouts

The A2 framework accepts no ambiguous “single artifact SHA” model.

#### Legacy JSON/NPZ pair providers

A2A, A2B, V1, V2, V2A, and W1 each use exactly:

```text
component_key = "manifest_json"
component_kind = MANIFEST_JSON
sha256 = SHA-256 of exact JSON file bytes
size_bytes = exact JSON file byte length

component_key = "payload_npz"
component_kind = PAYLOAD_NPZ
sha256 = SHA-256 of exact NPZ file bytes
size_bytes = exact NPZ file byte length
```

Their canonical component tuple contains exactly those two records and no others. Their
typed runtime reference must carry access paths separately from identity and must carry
the expected two hashes and sizes before loader dispatch. Existing public loaders remain
the semantic/integrity authority; an adapter must not select only one sibling file.

#### I1 bundle supporting evidence

I1 is fixed as one bundle manifest plus referenced W1/V2 and optional V2A pairs:

```text
bundle_manifest_json:
  BUNDLE_MANIFEST_JSON

w1_manifest_json:
  REFERENCED_MANIFEST_JSON
w1_payload_npz:
  REFERENCED_PAYLOAD_NPZ

v2_manifest_json:
  REFERENCED_MANIFEST_JSON
v2_payload_npz:
  REFERENCED_PAYLOAD_NPZ

v2a_manifest_json:
  REFERENCED_MANIFEST_JSON
v2a_payload_npz:
  REFERENCED_PAYLOAD_NPZ
```

The V2A two-component group is either wholly present or wholly absent. The mandatory
five-component group is always present. Every SHA/size covers the exact raw bytes of its
named file.

An I1 supporting adapter cannot be implemented until a public typed I1 reference exposes
the bundle-manifest expected SHA/size in addition to the existing referenced JSON/NPZ
facts. The adapter may not read an unbound path and then invent its expected digest.

#### Formal prerequisite providers

P-SOURCE, P-HAG, P-GEOMETRY, and P-WIRE must each freeze an exact component-key set,
component kinds, raw-byte SHA/size fields, and public typed reference before production
A2 integration. Production integration is blocked until those layouts are frozen. A2
does not let an adapter choose between a single file, pair, or bundle at runtime.

### 5.6 `FrvA2ProviderBindingEvidence`

Exact fields:

```text
provider_key: str
provider_role: FrvA2ProviderRole
evidence_class: FrvA2EvidenceClass
public_module: str
public_type_name: str
public_loader_name: str
interface_version: str
schema_version: str | None
algorithm_version: str
configuration_digest: str
provider_config_digest: str | None
artifact_reference_id: str
artifact_components: tuple[FrvA2ArtifactComponentBinding, ...]
source_content_identity_digest: str
source_identity_record_digest: str
coordinate_frame: str
units: str
origin_xyz: tuple[float, float, float]
transform_declaration: str
domain_kind: FrvA2DomainKind
domain_count: int
original_index_domain_digest: str | None
sample_index_domain_digest: str | None
provider_readiness: FrvA2ProviderReadiness
quality_flags: int
reason_flags: int
error_codes: tuple[FrvErrorCode, ...]
upstream_binding_digests: tuple[str, ...]
review_codes: tuple[FrvA2ReviewCode, ...]
limitation_codes: tuple[FrvA2LimitationCode, ...]
binding_digest: str
```

Rules:

- frozen, immutable, exact schema, canonicalizable;
- `provider_key` matches `^[a-z][a-z0-9_]{0,63}$`;
- public module/type/loader, versions, frame, units, and transform are exact constants or
  values validated by the owning public provider contract; arbitrary runtime text is
  forbidden;
- `schema_version=None` only where the public provider exposes no independent schema;
- `provider_config_digest` is a strict SHA or explicit null according to the owning
  configuration rule;
- `artifact_reference_id` must exactly equal the derived value from Section 5.4;
- component layout must equal the exact layout in Section 5.5 or the frozen formal
  prerequisite layout;
- `upstream_binding_digests` contains only strict SHA-256 binding digests, never paths or
  free reference text;
- `review_codes` and `limitation_codes` are typed enums in declaration order, unique;
- `provider_key` and `artifact_reference_id` are globally unique in one readiness record;
- `FORMAL_PROVIDER` may satisfy a formal role gate;
- the mappings and invariants in Section 5.2 are mandatory;
- the digest validator excludes `binding_digest`.

### 5.7 `FrvA2ReadinessRecord`

Schema version:

```text
frv-a2-readiness-v1
```

Exact fields:

```text
schema_version: str
source_content_identity_digest: str
source_identity_record_digest: str
configuration_digest: str
provider_evidence: tuple[FrvA2ProviderBindingEvidence, ...]
aggregate_readiness: FrvA2AggregateReadiness
blocking_failures: tuple[FrvA2FailureRecord, ...]
review_codes: tuple[FrvA2ReviewCode, ...]
limitation_codes: tuple[FrvA2LimitationCode, ...]
result_digest: str
```

Rules:

- frozen, immutable, exact schema, canonicalizable;
- `provider_evidence` is sorted by
  `(FrvA2ProviderRole declaration order, provider_key)`;
- `provider_key` and `artifact_reference_id` are globally unique;
- expected reload references are the exact set of
  `(provider_role.value, provider_key, artifact_reference_id)`;
- each provider evidence object exposes a deterministic readiness-digest projection
  containing exactly the Section 7.2 `binding_digest_payload` fields and excluding
  `binding_digest`;
- `blocking_failures` uses Section 10.7 canonical order;
- `review_codes` and `limitation_codes` are enum tuples in declaration order, unique;
- `READY_FOR_A3` requires no blocking failures, review codes, or limitation codes;
- `READY_FOR_A3_WITH_LIMITATIONS` requires no blocking failures and at least one review
  or limitation code;
- `BLOCKED` requires at least one blocking failure;
- no free review/limitation strings, arrays, paths, runtime objects, or A3 output fields;
- the readiness digest excludes `result_digest` and excludes every nested provider
  evidence `binding_digest`; validators reject any alternative projection.

### 5.8 Public API names

Framework public API:

```text
frv_a2_index_domain_digest
frv_a2_provider_binding_digest
frv_a2_readiness_record_digest
validate_frv_a2_failure_record
validate_frv_a2_artifact_component_binding
validate_frv_a2_provider_binding_evidence
validate_frv_a2_readiness_record
```

Production integration public API:

```text
validate_frv_a2_inputs
load_frv_a2_inputs
```

The production signatures remain blocked until P-SOURCE, P-HAG, P-GEOMETRY, and P-WIRE
freeze their concrete public result/reference types.

## 6. Domain model

### 6.1 Formal providers and supporting evidence

`FrvA2EvidenceClass.FORMAL_PROVIDER` and
`FrvA2EvidenceClass.SUPPORTING_EVIDENCE` are not interchangeable.

- A formal provider must satisfy the configured provider role and its declared domain.
- Supporting evidence proves only its own artifact integrity, provenance, and declared
  subset/reference domain.
- Supporting evidence cannot satisfy a formal provider gate.
- A supporting subset that differs from the full source is not automatically a source
  mismatch.
- A formal provider that falsely claims its required domain is a blocking failure.

### 6.2 Domain kinds

| Domain kind | Required meaning |
|---|---|
| `FULL_SOURCE` | Complete ordered source `original_index` domain declared by `SourceIdentity` |
| `DECLARED_SUBSET` | Exact ordered subset with an explicit parent-domain/source binding |
| `SAMPLED_SUBSET` | Exact ordered sampled points; normally carries applicable `sample_index` |
| `CANDIDATE_SUBSET` | Candidate/instance member point domain from V1/V2 evidence |
| `CLUSTER_SUBSET` | Cluster- or instance-level evidence; not a complete point-domain substitute |
| `PAIR_SUBSET` | V2A pair evidence; not a point-domain substitute |
| `ARTIFACT_REFERENCE` | I1-style artifact/provenance reference without point-domain authority |
| `INFLUENCE_DOMAIN` | Authoritative wire provider's explicitly declared influence domain |

### 6.3 Domain requirements

- `FULL_SOURCE` must exactly cover the authoritative source domain.
- `DECLARED_SUBSET`, `SAMPLED_SUBSET`, and `CANDIDATE_SUBSET` validate only the declared
  ordered subset.
- `PAIR_SUBSET` and `ARTIFACT_REFERENCE` must not masquerade as a point domain.
- `INFLUENCE_DOMAIN` does not need to cover the whole source, but it must carry an exact
  source binding, domain reference, coverage status in its prerequisite provider contract,
  and an index-domain digest when it is point-index based.
- Equal array length is never proof of domain equality.
- No sorting, truncation, nearest-neighbor repair, coordinate approximation, or silent
  dropping is allowed.

## 7. Three approved A2 digest domains

FRV-A2 adds exactly three digest domains. It does not add a separate context/origin
digest.

### 7.1 `frv_a2_index_domain_digest`

Domain header:

```text
FRVA2IDX1
```

Digest byte stream, in exact order:

1. ASCII bytes `FRVA2IDX1\0`;
2. UTF-8 index role, exactly `ORIGINAL_INDEX` or `SAMPLE_INDEX`;
3. one NUL byte;
4. UTF-8 `source_content_identity_digest`;
5. one NUL byte;
6. UTF-8 `FrvA2DomainKind.value`;
7. one NUL byte;
8. element count as unsigned little-endian 64-bit;
9. each index in declared order as unsigned little-endian 64-bit.

Validation before hashing:

- exact one-dimensional integer array or tuple;
- bool and NumPy bool rejected;
- float, object dtype, signed negative value, and value above `2**64 - 1` rejected;
- no implicit conversion;
- uniqueness required when the domain semantics require unique point identities;
- streamed/chunked hashing is allowed;
- a full-size canonical copy is forbidden.

The digest is independent of filesystem path, provider name, host, mtime, and memory
address.

### 7.2 `frv_a2_provider_binding_digest`

Domain header:

```text
FRVA2BND1
```

The digest is:

```text
frv_digest(binding_digest_payload)
```

A1 `frv_digest(value)` performs canonical serialization internally. The exact mapping is:

```text
digest_domain = FRVA2BND1
provider_key
provider_role.value
evidence_class.value
public_module
public_type_name
public_loader_name
interface_version
schema_version
algorithm_version
configuration_digest
provider_config_digest
artifact_reference_id
artifact_components as ordered mappings:
  component_key
  component_kind.value
  sha256
  size_bytes
source_content_identity_digest
source_identity_record_digest
coordinate_frame
units
origin_xyz
transform_declaration
domain_kind.value
domain_count
original_index_domain_digest
sample_index_domain_digest
provider_readiness.value
quality_flags
reason_flags
error_codes using exact FrvErrorCode values
upstream_binding_digests
review_codes using exact enum values
limitation_codes using exact enum values
```

`binding_digest` itself is excluded. Every included string is an enum value, strict SHA,
provider-contract constant, version token, or grammar-constrained stable key. Paths,
hostnames, exception text, mtimes, filenames, runtime objects, and memory addresses are
forbidden.

### 7.3 `frv_a2_readiness_record_digest`

Domain header:

```text
FRVA2RDY1
```

The digest is:

```text
frv_digest(readiness_digest_payload)
```

The exact mapping is:

```text
digest_domain = FRVA2RDY1
schema_version
source_content_identity_digest
source_identity_record_digest
configuration_digest
provider_evidence_projections
aggregate_readiness.value
blocking_failures in Section 10.7 canonical order
review_codes in declaration order
limitation_codes in declaration order
```

`provider_evidence_projections` is sorted by
`(provider_role declaration order, provider_key)`. Each element is exactly the
Section 7.2 `binding_digest_payload` mapping:

```text
digest_domain = FRVA2BND1
provider_key
provider_role.value
evidence_class.value
public_module
public_type_name
public_loader_name
interface_version
schema_version
algorithm_version
configuration_digest
provider_config_digest
artifact_reference_id
artifact_components as ordered mappings:
  component_key
  component_kind.value
  sha256
  size_bytes
source_content_identity_digest
source_identity_record_digest
coordinate_frame
units
origin_xyz
transform_declaration
domain_kind.value
domain_count
original_index_domain_digest
sample_index_domain_digest
provider_readiness.value
quality_flags
reason_flags
error_codes using exact FrvErrorCode values
upstream_binding_digests
review_codes using exact enum values
limitation_codes using exact enum values
```

The projection explicitly excludes each provider evidence `binding_digest`. It does not
replace the projection with the binding-digest string and does not serialize the complete
dataclass through a generic serializer. `validate_frv_a2_readiness_record` must recompute
the projection field by field and reject a record whose `result_digest` was produced from
a different nested projection.

`result_digest` itself is also excluded. Runtime provider views, typed access paths,
filenames, hostnames, exception text, filesystem paths, and every nested `binding_digest`
are excluded.

### 7.4 Existing/provider-owned digests

- FRV configuration digest: reuse FRV-A1.
- Source-content and source-identity-record digests: reuse FRV-A1.
- Artifact SHA/size: reuse each approved provider artifact boundary.
- Geometry content digest: defined by P-GEOMETRY; A2 validates it and does not redefine it.
- Wire geometry/content digest: defined by P-WIRE; A2 validates it and does not redefine it.

Any additional A2 digest domain requires a separate reviewed plan change.

## 8. Source identity and provider binding

FRV-A2 must reuse:

- `validate_source_content_identity_digest`;
- `validate_source_identity_record_digest`;
- `validate_source_equivalence_audit_without_cycle`;
- `validate_configuration`;
- `validate_production_wire_provider_binding`;
- `canonical_frv_json_bytes`;
- `frv_digest`.

A path, source ID, manifest ID, file size, mtime, or equal row count is never a substitute
for source identity.

Each formal provider must bind to:

- the same validated `source_content_identity_digest`;
- the same validated `source_identity_record_digest`, or an explicitly approved FRV-A1
  relocation/equivalence relationship;
- exact coordinate frame, units, origin, and transform declaration;
- its exact declared domain and index-domain digests;
- the selected configuration digest;
- its public provider identity, interface version, optional schema version, algorithm
  version, exact artifact component tuple, and derived artifact reference ID.

A legacy provider without a separate schema version must record `schema_version=None`;
the adapter must not invent a schema string.

## 9. HAG, geometry, scan, and wire validation

### 9.1 HAG

The frozen public HAG evidence states include the exact name:

```text
HAG_WITHIN_NEGATIVE_TOLERANCE
```

No alternative negative-tolerance HAG enum spelling is permitted.

The formal P-HAG contract must define:

- public interface version;
- source and record identity binding;
- declared domain and index-domain digest;
- raw/clamped HAG and ground evidence;
- HAG evidence/readiness state using approved FRV vocabulary;
- finite/nonfinite rules;
- configuration digest;
- artifact integrity and provenance;
- read-only runtime arrays.

Existing A2B can be recorded only as supporting evidence. Its algorithm version is
`a2b-grid-quantile-idw-v1`; it has no independent FRV schema version.

### 9.2 Geometry

The formal P-GEOMETRY contract must define:

- full-resolution or explicitly declared formal domain;
- source/record identity;
- frame/units/origin/transform;
- interface version, optional schema version, algorithm version, configuration digest;
- provider-owned geometry content digest and encoding;
- finite-value, shape, quality, and readiness rules;
- artifact integrity and provenance;
- read-only runtime arrays.

Existing A2A is supporting evidence. `scene_simple_geometry_features` is an output
basename, `a2a-local-pca-v1` is the algorithm version, and there is no independent
`GeometryFeatureResult.schema_version`.

### 9.3 Source/scan

P-SOURCE must provide an identity-bound public source/scan result containing:

- `SourceIdentity` binding;
- point count and authoritative source domain;
- frame, units, origin, transform;
- bounds, scale/offset, dimensions, return/classification/intensity availability;
- completion/truncation/corruption/quality state;
- interface/schema/algorithm/configuration facts as applicable;
- artifact integrity and provenance.

FRV-A2 must not invoke LAS/LAZ payload streaming to invent missing source identity.

### 9.4 Wire

A2 validates the wire provider boundary only. It may express provider validity, provider
review state, supporting-only evidence, or blocking failure. It must not perform influence
inventory, produce `WIRE_NOT_APPLICABLE_PROVEN`, or produce `WIRE_BODY_CONFIRMED`.

### 9.5 Configuration-to-required-role algorithm

The required formal-role set is derived only from validated `FrvConfiguration`; there is
no caller-supplied role list.

Base roles for every purpose:

```text
SOURCE_SCAN
HAG
GEOMETRY
```

These are always required because their interface-version and provider-config-digest
fields are mandatory in A1.

Wire role:

```text
PRODUCTION_APPROVED:
  WIRE is required.
  wire_provider_interface_version must be nonempty.
  wire_provider_config_digest must be a strict SHA.
  missing/invalid wire provider -> BLOCKED.

CALIBRATION_CANDIDATE:
  WIRE is required because the interface version is mandatory.
  wire_provider_config_digest may be null.
  missing/invalid wire provider -> BLOCKED.
  if the provider is otherwise identity/domain valid but the config digest is null:
    provider_readiness = REVIEW_REQUIRED
    review_codes includes CALIBRATION_WIRE_CONFIG_UNBOUND
    limitation_codes includes CALIBRATION_WIRE_CONFIG_UNBOUND.

TEST_ONLY:
  if both wire binding fields are null:
    WIRE is not a required formal role.
    a supplied wire result may be supporting evidence only.
    limitation_codes includes OPTIONAL_TEST_WIRE_PROVIDER_ABSENT.

  if both wire binding fields are non-null:
    WIRE is a required formal role.
    missing/invalid wire provider -> BLOCKED.

  if exactly one wire binding field is null:
    A1 `validate_configuration` may accept this TEST_ONLY combination.
    FRV-A2 therefore applies an additional stage-specific fail-closed rule in
    PRODUCTION_POLICY before required-role aggregation:
      aggregate_readiness = BLOCKED
      phase = PRODUCTION_POLICY
      provider_role = WIRE
      provider_key = "wire"
      field_name = "wire_provider_binding"
      error_code = FrvErrorCode.FRV_PROVIDER_INVALID
      reason_flags = int(FrvReasonFlag.PROVIDER_UNCERTAINTY)
      detail = the canonical FrvA2FailureRecord template
    no provider loading or semantic wire validation is attempted for the half-bound
    configuration.
```

`validate_configuration` is called first for A1 invariants. FRV-A2 must not claim that A1
rejects TEST_ONLY half-binding; the exact additional A2 rule above closes that gap.

Purpose-derived aggregate codes:

```text
CALIBRATION_CANDIDATE or TEST_ONLY:
  limitation_codes includes NONPRODUCTION_CONFIGURATION
  limitation_codes includes FORMAL_COMPLETENESS_FORBIDDEN
  limitation_codes includes MEMBERSHIP_PUBLICATION_FORBIDDEN
```

Aggregate readiness:

```text
BLOCKED:
  any required formal role is missing, invalid, identity/domain mismatched, or BLOCKED.

READY_FOR_A3_WITH_LIMITATIONS:
  no blocking failure;
  every required formal role is present and identity/domain valid;
  at least one required formal provider is REVIEW_REQUIRED,
  or purpose-derived limitation codes are present,
  or TEST_ONLY legally omits WIRE.

READY_FOR_A3:
  no blocking failure;
  all required formal roles are VALIDATED;
  review_codes and limitation_codes are empty.
```

Missing optional TEST_ONLY wire evidence never proves wire not-applicable.

## 10. Phased failure collection and validation order

The fixed model is:

```text
PHASED_FAILURE_COLLECTION
```

### 10.1 Phase 1 — `STRUCTURE_INTEGRITY`

Checks allowed:

- public type;
- exact schema/field inventory;
- manifest;
- artifact SHA and byte size;
- required/unknown fields;
- array inventory;
- dtype/shape metadata;
- `allow_pickle=False` and object-dtype rejection where applicable.

Independent providers may all be checked in this phase.

A provider that fails Phase 1:

- must not have semantic arrays read or interpreted;
- does not enter later phases;
- contributes deterministic failure records.

### 10.2 Phase 2 — `IDENTITY_CONTEXT`

Only Phase-1-passing providers are checked for:

- source-content identity;
- source-identity-record identity;
- provider/configuration identity;
- interface/schema/algorithm compatibility;
- frame;
- units;
- origin;
- transform.

### 10.3 Phase 3 — `DOMAIN_ALIGNMENT`

Only providers that pass Phases 1 and 2 are checked for:

- evidence class;
- domain kind;
- domain count;
- original-index domain digest;
- sample-index domain digest;
- full-source/subset/influence-domain claims.

### 10.4 Phase 4 — `SEMANTIC_READINESS`

Only providers that pass the first three phases are checked for:

- HAG readiness and finite-value semantics;
- geometry readiness and content binding;
- scan/source quality;
- wire provider quality and relation evidence;
- provider quality/reason flags.

### 10.5 Phase 5 — `CROSS_PROVIDER`

Only validated single-provider facts are used for exact cross-provider checks:

- source-content and record identity;
- configuration digest;
- frame/units/origin/transform;
- compatible formal domains;
- provenance references;
- required role uniqueness.

Supporting evidence is compared only within its declared subset/reference semantics.

### 10.6 Phase 6 — `PRODUCTION_POLICY`

Final checks:

- configuration purpose;
- required provider roles;
- runtime/exhaustive facts;
- calibration digest;
- production wire binding;
- aggregate readiness and limitations.

### 10.7 Deterministic failure ordering

Failures are sorted by the total-order key:

```text
(
  FrvA2FailurePhase declaration index,
  FrvA2ProviderRole declaration index,
  provider_key,
  field_name,
  FrvErrorCode.value,
  reason_flags,
)
```

`reason_flags` is a nonnegative non-bool integer and is compared numerically. In
`FRVA2RDY1` canonical JSON it serializes as a JSON integer.

The deduplication key is the same six-field tuple. `detail` is excluded because it is the
exact deterministic template derived from those fields.

A displayed `first_fatal` is derived from the first item in the canonical tuple and never
from runtime exception order.

## 11. Serializable readiness and validated runtime handoff

### 11.1 Serializable layer

`FrvA2ReadinessRecord` is the persistent proof of:

- which provider artifacts were validated;
- their identity, version, artifact, domain, readiness, and limitation facts;
- deterministic failures;
- whether A3 input consumption is allowed.

It contains no NumPy arrays and no runtime provider objects.

### 11.2 Runtime layer

`ValidatedFrvA2Inputs` is a non-serializable immutable runtime bundle in
`algorithms/vegetation/frv_inputs.py`.

Its field names are frozen as:

```text
source_identity
configuration
readiness_record
source_scan
hag
geometry
wire
supporting_evidence
```

The concrete public types of `source_scan`, `hag`, `geometry`, and `wire` are supplied by
P-SOURCE, P-HAG, P-GEOMETRY, and P-WIRE. Production `ValidatedFrvA2Inputs` must not be
implemented until those prerequisite contracts freeze.

`supporting_evidence` contains only explicitly approved loaded public result objects or
no-copy read-only adapter views. It cannot satisfy a formal provider role.

A3 in the same process must consume the exact `ValidatedFrvA2Inputs` returned by A2. It
must not reselect an artifact by path.

### 11.3 `validate_frv_a2_inputs`

The final production signature is structurally fixed:

```text
validate_frv_a2_inputs(
    *,
    source_identity,
    configuration,
    source_scan,
    hag,
    geometry,
    wire,
    supporting_evidence,
) -> ValidatedFrvA2Inputs
```

Concrete parameter types are inserted only after the prerequisite provider contracts
freeze. The function must:

1. run the six validation phases;
2. build provider-binding evidence;
3. build the serializable readiness record;
4. return the exact validated provider views in the runtime bundle;
5. return no partial bundle when aggregate readiness is `BLOCKED`.

The A2 framework may be tested with explicitly labelled synthetic/test provider contracts;
those fixtures cannot be reported as production providers.

### 11.4 `load_frv_a2_inputs`

The structurally fixed signature is:

```text
load_frv_a2_inputs(
    *,
    readiness_record,
    source_identity,
    configuration,
    provider_references,
) -> ValidatedFrvA2Inputs
```

Each typed reference exposes:

```text
provider_role
provider_key
artifact_reference_id
artifact_components
runtime_access_locations
```

`runtime_access_locations` is never serialized or digested. The owning public provider
contract freezes its type and loader consumption.

Exact reload algorithm:

1. validate the readiness record;
2. derive expected keys from provider evidence:
   `(provider_role.value, provider_key, artifact_reference_id)`;
3. derive supplied keys from typed references;
4. reject duplicate key, duplicate provider key, or duplicate artifact reference ID;
5. reject omitted or extra key;
6. require exact key-set equality;
7. require exact equality of every supplied component key/kind/SHA/size tuple with
   provider evidence;
8. only then call the public loader named by matching provider evidence;
9. recompute raw-byte SHA and size for every component through that approved loader or
   typed-reference boundary;
10. reject any changed JSON, NPZ, bundle manifest, source file, or referenced pair;
11. rerun interface/schema/algorithm/configuration/source/domain validation;
12. rerun the six validation phases;
13. return `ValidatedFrvA2Inputs`.

For I1, the bundle manifest and every mandatory/optional referenced component are checked
individually. A V2A half-pair is rejected.

Directory scanning, newest-file selection, filename inference, source-ID path guessing,
and readiness-record-only trust are forbidden.

A missing configured required wire reference is `BLOCKED`.

A readiness record alone never authorizes data use after restart.

## 12. Immutability, memory, and no-copy rules

A2 is a validation and handoff layer, not a payload transformation layer.

Requirements:

- inputs are never mutated;
- arrays are never reordered, truncated, expanded, or silently converted;
- object dtype is rejected;
- semantic nonfinite values follow only the approved provider contract;
- every runtime ndarray view exposed by `ValidatedFrvA2Inputs` must satisfy
  `np.shares_memory(runtime_view, provider_array)`;
- every runtime ndarray view must have `writeable == False`;
- runtime bundle newly owned full-size ndarray count must be `0`;
- a full-size canonical index copy is forbidden;
- index hashing may be streamed/chunked;
- small scalar and hashing buffers are allowed;
- no provider path, host, mtime, or memory address participates in identity digests;
- V1/V2/V2A/W1/I1 adapters must document whether they return an existing immutable object
  or create a no-copy read-only view.

## 13. Proposed implementation files and public exports

### 13.1 A2 framework files

| File | Change | Responsibility |
|---|---|---|
| `core/contracts/frv_a2.py` | New | Enums, artifact-component/failure/evidence/readiness records, three digest functions, pure validators |
| `core/contracts/__init__.py` | Modify | Export only reviewed A2 public contract names |
| `tests/contracts/test_frv_a2_contracts.py` | New | Strict type, enum, exact-schema, digest, failure ordering, and invariant tests |
| `tests/contracts/test_frv_a2_serialization.py` | New | Canonical JSON, round-trip, unknown/missing field, and self-digest tests |

### 13.2 Production integration files

| File | Change | Responsibility |
|---|---|---|
| `algorithms/vegetation/frv_inputs.py` | New | `ValidatedFrvA2Inputs`, approved adapters, six-phase orchestration, runtime handoff, reload-and-revalidate |
| `algorithms/vegetation/__init__.py` | Modify | Export `ValidatedFrvA2Inputs`, `validate_frv_a2_inputs`, and `load_frv_a2_inputs` |
| `tests/vegetation/test_frv_inputs.py` | New | Provider integration, handoff, TOCTOU, subset domains, no-copy, and stage-boundary tests |

Provider producers do not belong in `frv_inputs.py`. Each prerequisite has its own plan,
contract module, producer module, public loader, tests, and review.

## 14. Detailed implementation sequence

### A2-C1 — Public readiness/failure/domain contracts

Preconditions:

- FRV-A1 baseline unchanged;
- this plan frozen;
- separate implementation authorization granted.

Products:

- exact enums and dataclasses in `core/contracts/frv_a2.py`;
- artifact component/layout and derived reference-ID validation;
- deterministic configuration-to-required-role derivation;
- package exports;
- strict constructor/from-dict validators.

Pass conditions:

- exact field inventory;
- no bool-as-int;
- built-in string strictness;
- immutable tuple containers;
- invariants in Section 5.

Stop conditions:

- a proposed field introduces A3 inventory, scope, membership, artifact publication, or
  runtime objects into the serializable record.

### A2-C2 — Three digest domains

Products:

- `frv_a2_index_domain_digest`;
- `frv_a2_provider_binding_digest`;
- `frv_a2_readiness_record_digest`.

Pass conditions:

- exact domain headers;
- exact field domains;
- self-digest exclusion;
- no path/host/mtime/runtime object;
- chunked index hashing produces the same digest as scalar reference encoding;
- no fourth A2 digest.

### A2-C3 — Phased validation framework

Products:

- pure failure aggregation;
- phase gating;
- deterministic sorting/deduplication;
- aggregate readiness invariants.

Pass conditions:

- semantic arrays are never interpreted after structural failure;
- supporting evidence cannot satisfy formal roles;
- wire N/A is not emitted;
- no partial validated runtime bundle on `BLOCKED`.

### A2-C4 — Synthetic/test-provider framework

Products:

- deterministic public test-provider fixtures or stubs explicitly labelled non-production;
- positive and negative phase tests.

Pass conditions:

- test/calibration fixtures cannot satisfy production policy;
- no existing A2A/A2B/W1 result is silently promoted.

### A2-C5 — Serialization and strict negative tests

Products:

- contract and serialization suites;
- canonical digest round trips;
- unknown/missing field and mutation rejection.

### P-SOURCE / P-HAG / P-GEOMETRY / P-WIRE

Each is a separate prerequisite plan and review. Each must freeze:

- public result type;
- public loader;
- interface/schema/algorithm/configuration facts;
- source and record binding;
- domain and index digest;
- artifact integrity;
- quality/readiness;
- no-copy runtime behavior.

### A2-I1 — Concrete provider adapters

Precondition: applicable prerequisite contracts frozen.

Products:

- role-specific adapters;
- supporting evidence adapters;
- exact public provider identity registry.

### A2-I2 — `ValidatedFrvA2Inputs`

Precondition: all required concrete provider types exist.

Products:

- immutable runtime bundle;
- exact role-specific fields;
- no-copy read-only views.

### A2-I3 — Reload-and-revalidate

Products:

- `load_frv_a2_inputs`;
- artifact reference resolution through approved public loaders;
- TOCTOU rejection.

### A2-I4 — A3 runtime handoff

Products:

- public A3-facing consumption of `ValidatedFrvA2Inputs`;
- no path-based provider reselection;
- no A3 scope or inventory implementation inside A2.

## 15. Test matrix

All tests use public entry points.

### 15.1 Contract and digest tests

| Scenario | Public entry | Expected result |
|---|---|---|
| Exact enum bases, declaration order, `.value` strings, and strict fields | A2 dataclasses/from-dict | Only the frozen `str, Enum` values pass; unknown/missing fields fail |
| bool-as-int, NumPy scalar, string subclass, object dtype | Contract/digest validators | Rejected without conversion |
| `FRVA2IDX1` reference vectors | `frv_a2_index_domain_digest` | Stable digest; chunked and scalar encodings agree |
| Invalid index role/domain/value/order | `frv_a2_index_domain_digest` | Rejected |
| Multi-component provider binding and self-digest exclusion | `frv_a2_provider_binding_digest` | JSON/NPZ/I1 component tuples are exact, stable, path-free, and acyclic |
| Readiness self-digest and nested binding-digest exclusion | `frv_a2_readiness_record_digest` | `result_digest` and every provider `binding_digest` are excluded; the exact Section 7.2 projection is used |
| Provider evidence sorting/uniqueness | `validate_frv_a2_readiness_record` | Canonical order required |
| Failure ordering/deduplication/canonical detail | `validate_frv_a2_readiness_record` | `reason_flags` participates in total order; exact derived detail required |

### 15.2 Phase and provider tests

| Scenario | Expected result |
|---|---|
| JSON/NPZ manifest changed independently | `STRUCTURE_INTEGRITY` failure |
| I1 bundle manifest changed | Failure even when referenced pairs are unchanged |
| I1 mandatory W1/V2 component missing | Failure |
| I1 V2A half-pair present | Failure |
| Source-content or source-record mismatch | `BLOCKED`; no semantic phase |
| Frame/units/origin/transform mismatch | `BLOCKED` |
| Full-source index reorder/duplicate/missing | `BLOCKED` |
| Supporting subset/reference domain | Valid supporting-only evidence; never formal |
| A2B mapping differs from `HAG + SUPPORTING_EVIDENCE + SUPPORTING_ONLY` | Rejected |
| A2A mapping differs from `GEOMETRY + SUPPORTING_EVIDENCE + SUPPORTING_ONLY` | Rejected |
| W1 mapping differs from `WIRE + SUPPORTING_EVIDENCE + SUPPORTING_ONLY` | Rejected |
| I1 mapping differs from `PROVENANCE + SUPPORTING_EVIDENCE + SUPPORTING_ONLY` | Rejected |
| Supporting evidence uses readiness other than `SUPPORTING_ONLY` | Rejected |
| `PRODUCTION_APPROVED` missing wire | `BLOCKED` |
| `CALIBRATION_CANDIDATE` missing wire | `BLOCKED` |
| Calibration wire present but config digest null | `READY_FOR_A3_WITH_LIMITATIONS` with exact review/limitation codes |
| `TEST_ONLY` both wire fields null and no wire | Required roles exclude WIRE; limitations are exact |
| `TEST_ONLY` both wire fields non-null and missing wire | `BLOCKED` |
| `TEST_ONLY` interface non-null/config digest null | A1 may pass; A2 emits the exact PRODUCTION_POLICY `FRV_PROVIDER_INVALID` failure and `BLOCKED` |
| `TEST_ONLY` interface null/config digest non-null | A1 may pass; A2 emits the same exact fail-closed record and `BLOCKED` |
| HAG NaN/Inf outside provider rule | `BLOCKED` |
| Geometry content/config/version mismatch | `BLOCKED` |
| Test/calibration provider used as production | `BLOCKED` |
| Interface/schema/algorithm/configuration mismatch | Phase-appropriate failure |
| Review/limitation field contains path/host/free text | Impossible by enum type; strict deserialization rejects it |
| Two failures differ only by `reason_flags` | Canonical numeric order is stable |

### 15.3 Handoff tests

- `validate_frv_a2_inputs` returns `ValidatedFrvA2Inputs` containing the exact provider
  objects/views that produced `readiness_record`.
- A3-facing test code consumes that runtime bundle and does not resolve new artifacts.
- A runtime bundle is not JSON serializable through the readiness-record serializer.
- A blocked validation returns no partial runtime bundle.

### 15.4 Reload and TOCTOU tests

- change only JSON bytes: reload fails;
- change only NPZ bytes: reload fails;
- substitute same-schema pair: reload fails;
- change I1 bundle manifest but not child artifacts: reload fails;
- replace one I1 child pair: reload fails;
- omit/extra/duplicate typed reference: reload fails before dispatch;
- omit/extra/duplicate artifact component: reload fails;
- duplicate `artifact_reference_id`: readiness validation fails;
- correct artifact with wrong semantic role/provider key: reload fails;
- mix provider components from another source: reload fails;
- relocate unchanged files: succeeds only under FRV-A1 relocation/equivalence rules;
- change index-domain content without row-count change: `FRVA2IDX1` fails;
- alter only a stored provider `binding_digest` while leaving its Section 7.2 fields
  unchanged: provider binding validation fails, while the FRVA2RDY1 projection remains
  defined exclusively by the Section 7.2 fields;
- compute FRVA2RDY1 by serializing nested `binding_digest`: the fixed reference-vector
  test rejects the digest.

### 15.5 Supporting-domain tests

Explicitly cover:

- A2A/A2B `SAMPLED_SUBSET`;
- V1/V2 `CANDIDATE_SUBSET`;
- cluster evidence `CLUSTER_SUBSET`;
- V2A `PAIR_SUBSET`;
- W1 sampled/centreline evidence as supporting subset;
- I1 `ARTIFACT_REFERENCE`.

Each can validate as supporting evidence without satisfying a formal provider role.
The fixtures must use these exact combinations:

```text
A2B: HAG + SUPPORTING_EVIDENCE + SUPPORTING_ONLY
A2A: GEOMETRY + SUPPORTING_EVIDENCE + SUPPORTING_ONLY
W1: WIRE + SUPPORTING_EVIDENCE + SUPPORTING_ONLY
I1: PROVENANCE + SUPPORTING_EVIDENCE + SUPPORTING_ONLY
```

Any `SUPPORTING_EVIDENCE` record whose readiness is not `SUPPORTING_ONLY` is rejected.

### 15.6 No-copy tests

For every ndarray exposed by a runtime provider view:

```text
np.shares_memory(runtime_view, provider_array) == True
runtime_view.flags.writeable == False
```

The runtime bundle newly owned full-size ndarray count must be `0`.

The test must fail if a full provider payload is copied.

### 15.7 Stage-boundary tests

Tests must assert that A2 outputs contain none of:

- candidate inventory;
- hard-rejection result;
- scope/union/partition/coverage geometry;
- wire not-applicable proof;
- point membership;
- manual override;
- FRV artifact payload;
- publication record;
- V4 adapter or tree-segmentation result.

## 16. Required verification commands

All commands use the project interpreter and are separated by authorization scope.

### 16.1 A2 framework focused verification

This is the only mandatory command set for an A2-C1 through A2-C5 implementation:

```powershell
E:\Miniforge\envs\treeline\python.exe -m pytest `
  tests/contracts/test_frv_a2_contracts.py `
  tests/contracts/test_frv_a2_serialization.py -q

E:\Miniforge\envs\treeline\python.exe -m ruff check `
  core/contracts/frv_a2.py `
  tests/contracts/test_frv_a2_contracts.py `
  tests/contracts/test_frv_a2_serialization.py
```

No other test path is part of the framework-focused gate.

### 16.2 Separately authorized framework regression

Only after the owner explicitly authorizes broad read-only regression validation:

```powershell
E:\Miniforge\envs\treeline\python.exe -m pytest tests/contracts -q
E:\Miniforge\envs\treeline\python.exe -m pytest -q
```

These commands do not authorize file modification and are not prerequisites for the
focused framework gate unless the owner grants this separate validation authorization.

### 16.3 Production-integration verification

Only after provider prerequisites and A2-I1 through A2-I4 are authorized and files exist:

```powershell
E:\Miniforge\envs\treeline\python.exe -m pytest `
  tests/contracts/test_frv_a2_contracts.py `
  tests/contracts/test_frv_a2_serialization.py `
  tests/vegetation/test_frv_inputs.py -q

E:\Miniforge\envs\treeline\python.exe -m ruff check `
  core/contracts/frv_a2.py `
  algorithms/vegetation/frv_inputs.py `
  tests/contracts/test_frv_a2_contracts.py `
  tests/contracts/test_frv_a2_serialization.py `
  tests/vegetation/test_frv_inputs.py
```

Contract/full regression for production integration remains a separate owner-authorized
read-only validation step.

Bare `pytest` and bare `ruff` are never approved acceptance commands.

## 17. Acceptance gates

Every gate is binary.

| Gate | PASS condition | FAIL condition |
|---|---|---|
| Baseline | Branch, HEAD, divergence, and project status match the authorized stage | Any unauthorized project change |
| Public API | Every public name, `str, Enum` base/value/order, field, export, and function matches this plan | Implementer must invent a public name, enum encoding, or field |
| Contract | Strict immutable records, typed codes, exact artifact components, and exact invariants pass | Free review text, ambiguous artifact fields, or mutable state |
| Digest | Only `FRVA2IDX1`, `FRVA2BND1`, and `FRVA2RDY1` are added; FRVA2RDY1 uses the exact nested Section 7.2 projection and excludes every `binding_digest` | Undefined/fourth digest, generic nested serialization, or cyclic self-digest |
| Failure model | Six phases, gating, sorting, and deduplication are exact | Semantic use after structural failure or unstable ordering |
| Evidence class | Formal and supporting evidence remain distinct | Supporting evidence satisfies a formal gate |
| Domain | Full-source/subset/reference/influence semantics are exact | Equal-length, reorder, truncate, or fuzzy repair |
| Source identity | FRV-A1 source content and record identities validate | Path/source ID substituted for identity |
| HAG | Formal P-HAG contract and readiness rules pass | A2B is promoted without formal binding |
| Geometry | Formal P-GEOMETRY content/domain/readiness passes | A2A is promoted without formal binding |
| Source/scan | Formal P-SOURCE contract passes | Header metadata alone is treated as source identity |
| Wire | A2 validates boundary only and emits no N/A/body confirmation | A2 performs influence inventory or body confirmation |
| Aggregate readiness | Section 9.5 derives roles deterministically and A2 explicitly blocks both TEST_ONLY half-bound wire combinations | Caller-chosen role set, false claim that A1 rejects half-binding, or missing required provider becomes ready |
| Runtime handoff | A3 consumes exact `ValidatedFrvA2Inputs` views | Provider is reselected by path |
| Reload/TOCTOU | Typed references and every component key/kind/SHA/size match exactly and are revalidated | One JSON/NPZ/bundle component is omitted, changed, duplicated, or mixed |
| No-copy | Shares memory, read-only views, zero owned full-size arrays | Full payload copy or input mutation |
| Serialization | Exact schema, canonical JSON, stable digests, round trip | Runtime object/array enters readiness JSON |
| Stage boundary | No A3/A4/A5/A6/V4 output appears | Downstream responsibility implemented |
| Focused tests | New A2 suites pass | Any focused failure |
| Contract regression | Existing contracts pass | FRV-A1 regression |
| Full regression | Entire repository passes | Any repository regression |
| Ruff | Explicit project-interpreter Ruff command passes | Lint failure |
| Independent review | No unresolved blocker/major finding | Unresolved finding |
| Git authorization | Separate owner approval exists | Git write without approval |

Framework gates may pass before provider prerequisites. Production-integration gates cannot
pass until the applicable provider prerequisites are frozen and implemented.

## 18. Stage-boundary audit

| Concern | FRV-A2 may validate/return | FRV-A2 must not produce |
|---|---|---|
| Source | Identity-bound provider evidence and exact runtime view | New source identity or payload scan |
| HAG | Provider readiness/evidence and exact runtime view | Ground model or replacement HAG |
| Geometry | Provider content/readiness and exact runtime view | New geometry features or scope geometry |
| Scan | Bound scan metadata/runtime view | Point inventory or reconstruction |
| Wire | Provider binding/quality/relation evidence | Influence inventory, N/A proof, body confirmation |
| Domain | Exact full/subset/reference/influence declarations | Repair or implicit realignment |
| Handoff | `FrvA2ReadinessRecord` and `ValidatedFrvA2Inputs` | A3 inventory/scope |
| Persistence | Artifact references plus reload-and-revalidate | FRV-A6 publication |
| V4 | No transformation | CHM or tree segmentation |

## 19. Known gaps and fixed dispositions

| Question | Disposition | Plan decision |
|---|---|---|
| Is FRV-A1 sufficient as identity/config/digest foundation? | `REPOSITORY_PROVEN` | Reuse it; do not modify `core/contracts/frv.py` by default |
| Is an A2 readiness/failure/result contract available? | `A2_SUBTASK` | Add exact types in `core/contracts/frv_a2.py` |
| Is a formal source/scan provider available? | `SEPARATE_PREREQUISITE` | P-SOURCE plan required |
| Is a formal full-resolution HAG provider available? | `SEPARATE_PREREQUISITE` | P-HAG plan required |
| Is a formal full-resolution geometry provider available? | `SEPARATE_PREREQUISITE` | P-GEOMETRY plan required |
| Is an authoritative influence-domain wire provider/public loader available? | `SEPARATE_PREREQUISITE` | P-WIRE plan required |
| Do legacy public loaders provide useful integrity/provenance? | `REPOSITORY_PROVEN` | JSON/NPZ pair layouts are fixed; I1 additionally needs a public typed reference exposing bundle-manifest SHA/size before its A2 adapter is implemented |
| Can A2 avoid direct LAS/LAZ point-payload access? | `REPOSITORY_PROVEN` safety rule | Reject missing identity/provider facts; do not backfill |
| Can production `ValidatedFrvA2Inputs` be finalized now? | `PRODUCTION_GATE_BLOCKER_ONLY` | Wait for prerequisite concrete provider types |
| Does A2 decide wire not-applicable? | `OUT_OF_SCOPE` | FRV-A3 or later only |
| Does A2 implement inventory/scope/membership/publication/V4? | `OUT_OF_SCOPE` | Explicitly forbidden |

Separate provider gaps block production integration, not independent A2 framework work.

## 20. Final implementation authorization boundary

This corrected plan freezes the intended FRV-A2 design only after the next independent
plan review passes.

The plan does not authorize:

- A2 framework code;
- source/scan, HAG, geometry, or wire prerequisite providers;
- production A2 integration;
- FRV-A3 through FRV-A6;
- V4;
- Git add, commit, or push.

The implementation order after plan freeze is:

1. owner separately authorizes an A2 framework implementation or a provider prerequisite;
2. the authorized subtask is implemented within its exact file boundary;
3. focused tests, contract regression, full regression, Ruff, and independent review pass;
4. the owner separately authorizes Git operations.

**Plan authorizes implementation:** `NO`
