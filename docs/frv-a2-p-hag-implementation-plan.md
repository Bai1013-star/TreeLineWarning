# FRV-A2 P-HAG Independent Implementation Plan

Status: PLAN-REVIEW-CANDIDATE

Repository: E:\TreeLineWarning
Branch: feature/algorithm
Baseline HEAD: 73d512442c1a2b5c341168bbdcb1e65f84a9a7b3
Baseline commit: 73d5124 feat: add reviewed FRV P-SOURCE runtime
Remote divergence: 0 0

This is an independent P-HAG plan. It closes P-HAG-BLK-001 through
P-HAG-BLK-004, but does not authorize implementation, DESIGN-FROZEN status,
P-GEOMETRY, P-WIRE, V4, membership publication, or Git operations.

Authorities:

- docs/full-resolution-vegetation-membership-design.md
- docs/frv-a2-implementation-plan.md
- docs/frv-a2-p-source-implementation-plan.md
- docs/v4-tree-segmentation-design.md
- existing core/contracts/frv.py, frv_a2.py, frv_source.py
- existing algorithms/io/source_provider.py

P-SOURCE is treated as closed authority. This plan does not revise or re-review
P-SOURCE.

## 1. Blocker closure and scope

| Finding | Closure |
|---|---|
| P-HAG-BLK-001 | Exact public API, result/proof, loader, modules, exports, implementation files, and tests are frozen below. |
| P-HAG-BLK-002 | Ground authority, sparse grid, interpolation, defaults, finite rules, negative tolerance, and determinism are frozen below. |
| P-HAG-BLK-003 | Input is a source reference plus finalized source result. Exactly two independent P-SOURCE sessions are created only through load_frv_source. |
| P-HAG-BLK-004 | P-HAG config, chunk, proof, result, error, session, digests, and FRVA2BND1 adapter are frozen below. |

P-HAG owns:

- full-resolution HAG;
- classification-2 ground support evidence;
- HAG state, reason, and quality;
- complete FULL_SOURCE original_index conservation;
- SourceIdentity and P-SOURCE binding;
- P-HAG chunks, proof, finalized result, and BND1 evidence;
- downstream readiness evidence.

P-HAG does not own:

- new ground classification;
- vegetation membership;
- geometry, density, canopy, PCA, structural graph, or P-GEOMETRY;
- P-WIRE;
- FRV final artifact publication;
- V4 recovery, CHM, tree segmentation, clearance, or risk distance.

Existing FrvHagEvidenceState is sufficient. No new public enum or flag is added.



## 2. Exact input and lifecycle

### 2.1 Public inputs and upstream API

P-HAG requires all four inputs:

1. FrvSourceProviderReference;
2. finalized FrvSourceProviderResult;
3. FrvConfiguration;
4. FrvHagProviderConfig.

P-HAG has no LAS path, laspy handle, output-path, or source-provider-config
public parameter. It never directly opens LAS. Every source row comes through:

    load_frv_source(
        reference: FrvSourceProviderReference,
        *,
        frv_configuration: FrvConfiguration,
        provider_config: FrvSourceProviderConfig,
        chunk_size: int = FRV_SOURCE_DEFAULT_CHUNK_SIZE,
    ) -> FrvSourceLoadSession

The P-SOURCE session remains single-use:

    iter_chunks(self) -> Iterator[FrvSourceChunk]
    finalize(self) -> FrvSourceProviderResult
    close(self) -> None
    __enter__(self) -> FrvSourceLoadSession
    __exit__(self, exc_type, exc, traceback) -> Literal[False]

FrvSourceProviderResult exposes only proof and source_identity and has no public
constructor. FrvSourceChunk is the only source-row payload boundary and carries
XYZ, original_index, pass-through attributes, and is_last.

### 2.2 Pre-pass validation

Before Pass 1, the following ordered validation sequence is frozen. The first
failing item wins; no later item may read, normalize, or replace a failed
field:

1. Require type(source_result) is FrvSourceProviderResult and the exact
   module-private finalized capability.
2. Run validate_finalized_frv_source_provider_result(source_result).
3. Run the reference validator's structural/type gate on reference.
4. Run validate_configuration(frv_configuration).
5. Run validate_frv_hag_provider_config(provider_config).
6. Reconstruct the P-SOURCE config below and compare its recomputed digest to
   source_result.proof.provider_config_digest; then run the full existing
   validate_frv_source_provider_reference cross-binding gate with that
   reconstructed config and frv_configuration.configuration_digest.
7. Validate source proof role/key/domain/frame/units/transform and the positive
   source count.
8. Validate the classification declaration in both
   source_proof.pass_through_dimensions and source_proof.header.dimensions.
9. Validate exact reference.source_identity to
   source_result.source_identity content and record identity.
10. Only after all nine gates pass may P-HAG create Pass 1.

The current full validate_frv_source_provider_reference signature requires a
FrvSourceProviderConfig; therefore step 3 is its no-I/O structural/type gate
and step 6 is the single full reference cross-binding call. This preserves the
required first-failure order without deriving any field from a path or header.

Private helper, not public API:

    _reconstruct_frv_source_provider_config(
        source_proof: FrvSourceProviderProof,
    ) -> FrvSourceProviderConfig

The helper uses this exact P-SOURCE dataclass field order and source:

| Order | FrvSourceProviderConfig field | Exact source |
|---:|---|---|
| 1 | schema_version | FRV_SOURCE_PROVIDER_CONFIG_SCHEMA_VERSION |
| 2 | source_model | frozen FrvSourceModel.SINGLE_FILE_ONLY |
| 3 | coordinate_conversion_policy | frozen FrvSourceCoordinateConversionPolicy.FORBID |
| 4 | invalid_point_policy | frozen FrvSourceInvalidPointPolicy.FAIL_CLOSED |
| 5 | canonical_content_policy | frozen FrvSourceCanonicalContentPolicy.VERIFY_IF_DECLARED |
| 6 | pass_through_dimensions | exact tuple copy of source_proof.pass_through_dimensions, preserving order and enum values |
| 7 | provider_config_digest | recomputed by derive_frv_source_provider_config_digest from fields 1–6 |

Fields 1–5 are never copied from the path, header, reference, or runtime
reader. Field 6 is the only proof-derived config field: it is copied without
sorting, deduplication, or recasting. The classification member must remain
the original FrvSourcePassThroughDimension.CLASSIFICATION position; its formal
source-chunk dtype remains the existing |u1 authority. The helper calls
derive_frv_source_provider_config_digest with the exact FRVSRCFG1 payload,
compares the result byte-for-byte as a lowercase SHA-256 to
source_proof.provider_config_digest, and only then constructs
FrvSourceProviderConfig; its constructor must independently recompute the same
digest. Any mismatch is a provider failure. No field is inferred from a path
or header.

### 2.3 Pass 1

Create one new independent P-SOURCE session through load_frv_source.

Pass 1 must:

- naturally consume every source chunk once;
- validate chunk continuity and the 0..N-1 source domain;
- read only XYZ and classification;
- add classification-2 finite points to the sparse ground grid;
- retain only per-cell minimum finite Z and point count;
- never retain source point lists, full XYZ, or all chunks;
- finalize the source session before any P-HAG result exists;
- compare the new result to the supplied source result by exact proof
  serialization, SourceIdentity content/record digests, source artifact
  reference, component tuple, count, FULL_SOURCE domain digest, and source BND1;
- close the finalized session before Pass 2.

### 2.4 Pass 2

Create a second independent P-SOURCE session through load_frv_source.

Pass 2 must:

- naturally consume every source chunk once;
- repeat source identity, proof, domain, and count validation;
- evaluate every row exactly once;
- emit only the current HAG chunk in source original_index order;
- update the HAG point-content digest in that order;
- finalize and close the P-SOURCE session before P-HAG finalize;
- compare its finalized result to the supplied result by the same exact
  binding rules.

P-HAG finalize is legal only after Pass 2 natural completion, Pass 2
finalization, count reconciliation, digest completion, and cleanup. No partial
P-HAG Result or BND1 evidence is created.

A third pass, direct laspy access, consumed-session reuse, path-only trust,
full source materialization, full chunk materialization, N-length index-list
construction, or result-before-finalize is forbidden.

source_chunk_size is operational only. It is excluded from every P-HAG config,
grid, point-content, proof, source-binding, and result digest. Any legal chunk
size produces identical formal proof/result/digest fields.

## 3. Exact ground and HAG algorithm

### 3.1 Ground authority

Only the LAS standard classification dimension is used, with classification == 2.

- classification must be in P-SOURCE pass-through dimensions;
- dtype must be the existing standard classification dtype, uint8;
- bool, float, object, or other classification dtype is rejected;
- missing classification is a provider failure;
- zero classification-2 points is a provider failure;
- all other classification values remain ordinary source points;
- P-HAG does not create or repair labels;
- PDAL, lidR, CSF, ML, and network services are forbidden.

### 3.2 Sparse grid

The anchor is source proof source_bounds_min_xyz[0:2], never the first chunk or
first ground point:

    min_x = source_bounds_min_xyz[0]
    min_y = source_bounds_min_xyz[1]

For every classification-2 point:

    ix = floor((x - min_x) / cell_size_m)
    iy = floor((y - min_y) / cell_size_m)

Cells are half-open:

    [min_x + ix*cell_size_m, min_x + (ix+1)*cell_size_m)
    [min_y + iy*cell_size_m, min_y + (iy+1)*cell_size_m)

Each cell stores only:

    (ix, iy, minimum_finite_ground_z, ground_point_count)

The minimum finite Z is the ground authority. The full point list is forbidden.
The dictionary is keyed by (ix, iy), and formal digest iteration is sorted by
(ix, iy). Identical input produces identical inventory. No TIN, Delaunay,
rasterio, PDAL, SciPy KDTree, Shapely, or tile/halo rule is used.
Before a cell enters the inventory, its finite minimum ground Z is zero
normalized: value == 0.0 is stored as positive +0.0. The cell inventory never
retains or hashes an IEEE negative-zero ground value.

### 3.3 Support candidates

For each Pass-2 point:

    ix_lo = ceil((x - support_radius_m - min_x) / cell_size_m - 0.5)
    ix_hi = floor((x + support_radius_m - min_x) / cell_size_m - 0.5)
    iy_lo = ceil((y - support_radius_m - min_y) / cell_size_m - 0.5)
    iy_hi = floor((y + support_radius_m - min_y) / cell_size_m - 0.5)

For each occupied cell in this integer range:

    center_x = min_x + (ix + 0.5) * cell_size_m
    center_y = min_y + (iy + 0.5) * cell_size_m
    squared_distance = (x-center_x)^2 + (y-center_y)^2

Retain only finite candidates with squared_distance <= support_radius_m^2.
Sort exactly by:

    (squared_distance, ix, iy)

Use at most max_support_cells candidates. Fewer than min_support_cells means
per-point unavailable support, not provider failure.

### 3.4 Ground interpolation

If the first candidate has squared_distance == 0.0, ground_z is its cell
minimum Z. Otherwise, using the fixed candidate order:

    weight_i = 1.0 / squared_distance_i
    ground_z = sum(weight_i * cell_min_ground_z_i) / sum(weight_i)

All arithmetic is float64 and sequential in the fixed order. Nonfinite distance,
weight, numerator, denominator, or ground_z is a provider failure.

Formal zero normalization is global and occurs before any value enters a
FrvHagChunk, ground-cell inventory, digest input, Proof comparison, or count/
state evidence: every finite float64 value equal to 0.0 is stored as positive
+0.0. This covers cell minimum ground Z, interpolated ground_z_m, raw_hag_m,
clamped_hag_m, and explicit unsupported values. State evaluation order is
frozen: (1) validate finite; (2) evaluate the state from the unnormalized raw
value and tolerance; (3) normalize every value equal to 0.0 to +0.0; and
(4) write the normalized values to chunks and digest input. Thus -0.0 follows
the raw_hag_m == 0 state branch but is never serialized or hashed as negative
zero.

### 3.5 HAG state mapping

    raw_hag_m = point_z - ground_z
    clamped_hag_m = max(raw_hag_m, 0.0)

| Condition | State | Reason flags | Quality flags |
|---|---|---|---|
| finite raw_hag_m > 0 | HAG_VALID_POSITIVE | NONE | NONE |
| finite raw_hag_m == 0 | HAG_VALID_NEAR_ZERO | NONE | NONE |
| -0.25 <= raw_hag_m < 0 | HAG_WITHIN_NEGATIVE_TOLERANCE | NONE | NONE |
| fewer than min_support_cells | HAG_MISSING | INSUFFICIENT_EVIDENCE \| HAG_MISSING | NONE |
| raw_hag_m < -negative_hag_tolerance_m | HAG_INVALID | HAG_INVALID | PROVIDER_INVALID |
| defensive validator sees point outside declared domain | HAG_OUT_OF_DOMAIN | HAG_OUT_OF_DOMAIN | PROVIDER_INVALID |

The default tolerance is 0.25 m. For a supported point, raw and clamped values
are retained and finite even when state is HAG_INVALID. Such a point is
review-only evidence and is never silently treated as normal positive HAG.

For unsupported points, ground_z_m, raw_hag_m, and clamped_hag_m are exactly
positive 0.0; ground_support is false and support_cell_count is zero. Missingness
is explicit and never NaN or Infinity.

HAG_OUT_OF_DOMAIN is not emitted by a valid FULL_SOURCE stream. A point outside
the validated source domain is a provider-level identity/domain failure.

## 4. Exact public contracts

Default contract module: core/contracts/frv_hag.py

### 4.1 Constants

    FRV_HAG_PROVIDER_KEY = "hag_provider"
    FRV_HAG_PROVIDER_INTERFACE_VERSION = "frv-hag-provider-v1"
    FRV_HAG_PROVIDER_SCHEMA_VERSION = "frv-hag-provider-result-v1"
    FRV_HAG_CONFIG_SCHEMA_VERSION = "frv-hag-provider-config-v1"
    FRV_HAG_PROOF_SCHEMA_VERSION = "frv-hag-provider-proof-v1"
    FRV_HAG_ALGORITHM_VERSION = "frv-hag-sparse-grid-idw2-v1"
    FRV_HAG_CONFIG_DIGEST_DOMAIN = "FRVHAGCFG1"
    FRV_HAG_SOURCE_PROOF_DIGEST_DOMAIN = "FRVHAGSRC1"
    FRV_HAG_GRID_DIGEST_DOMAIN = "FRVHAGGRID1"
    FRV_HAG_POINT_CONTENT_DIGEST_DOMAIN = "FRVHAGPCD1"
    FRV_HAG_PROOF_DIGEST_DOMAIN = "FRVHAGPRF1"

### 4.2 FrvHagProviderConfig

    @dataclass(frozen=True, slots=True)
    class FrvHagProviderConfig:
        schema_version: str
        provider_key: str
        provider_role: FrvA2ProviderRole
        interface_version: str
        algorithm_version: str
        cell_size_m: float
        support_radius_m: float
        min_support_cells: int
        max_support_cells: int
        negative_hag_tolerance_m: float
        classification_dimension: str
        provider_config_digest: str

Defaults:

    schema_version = "frv-hag-provider-config-v1"
    provider_key = "hag_provider"
    provider_role = FrvA2ProviderRole.HAG
    interface_version = "frv-hag-provider-v1"
    algorithm_version = "frv-hag-sparse-grid-idw2-v1"
    cell_size_m = 1.0
    support_radius_m = 5.0
    min_support_cells = 1
    max_support_cells = 8
    negative_hag_tolerance_m = 0.25
    classification_dimension = "classification"

All strings are exact built-in nonempty strings. Float fields are finite
built-in floats, never bool or integer. Integer fields are positive built-in
ints, never bool or NumPy scalars. support_radius_m >= cell_size_m,
max_support_cells >= min_support_cells, and classification_dimension is fixed
to classification.

Public methods:

    to_dict(self) -> dict[str, Any]
    @classmethod
    from_dict(cls, data: Mapping[str, Any]) -> FrvHagProviderConfig

Missing or extra serialization keys are rejected.

Config digest function:

    derive_frv_hag_provider_config_digest(
        *,
        schema_version: str,
        provider_key: str,
        provider_role: FrvA2ProviderRole,
        interface_version: str,
        algorithm_version: str,
        cell_size_m: float,
        support_radius_m: float,
        min_support_cells: int,
        max_support_cells: int,
        negative_hag_tolerance_m: float,
        classification_dimension: str,
    ) -> str

It calls existing frv_digest on the exact canonical payload with
digest_domain FRVHAGCFG1 and all fields above except provider_config_digest.
Path, host, timestamp, source chunk size, runtime objects, and memory address
are excluded.

### 4.3 FrvHagChunk

Runtime-only frozen dataclass with slots; no to_dict/from_dict:

    @dataclass(frozen=True, slots=True)
    class FrvHagChunk:
        source_id: str
        chunk_id: int
        start_index: int
        point_count: int
        original_index: np.ndarray
        ground_support: np.ndarray
        support_cell_count: np.ndarray
        ground_z_m: np.ndarray
        raw_hag_m: np.ndarray
        clamped_hag_m: np.ndarray
        hag_evidence_state: np.ndarray
        reason_flags: np.ndarray
        quality_flags: np.ndarray
        is_last: bool

Array contract:

| Field | dtype | shape | rule |
|---|---|---|---|
| original_index | <i8 | (point_count,) | nonnegative, strictly increasing within chunk |
| ground_support | bool | (point_count,) | explicit presence marker |
| support_cell_count | <u2 | (point_count,) | zero unsupported, 1..max supported |
| ground_z_m | <f8 | (point_count,) | finite, +0.0 unsupported |
| raw_hag_m | <f8 | (point_count,) | finite, +0.0 unsupported |
| clamped_hag_m | <f8 | (point_count,) | finite, +0.0 unsupported |
| hag_evidence_state | <u1 | (point_count,) | existing FrvHagEvidenceState values |
| reason_flags | <u4 | (point_count,) | existing FrvReasonFlag mask |
| quality_flags | <u4 | (point_count,) | existing FrvQualityFlag mask |

FrvHagChunk construction must receive already zero-normalized float64 arrays.
The runtime normalizes owned copies before constructing the chunk; the chunk
validator rejects any formal float64 array that still contains an IEEE negative
zero. This applies to ground_z_m, raw_hag_m, clamped_hag_m, and explicit
unsupported +0.0 markers. Validator comparisons and canonical digest inputs use
the same positive-zero rule.

Pass-2 chunk mapping is one-to-one and positional. Each FrvSourceChunk produces
exactly one FrvHagChunk: no source chunk is merged, split, reordered, or
silently dropped. The scalar mapping is:

| FrvHagChunk scalar | Required value |
|---|---|
| source_id | exact copy of the corresponding FrvSourceChunk.source_id |
| chunk_id | exact copy of the corresponding FrvSourceChunk.chunk_id |
| start_index | exact copy of the corresponding FrvSourceChunk.start_index |
| point_count | exact equality to the corresponding FrvSourceChunk.point_count |
| original_index | independent read-only canonical array with exactly the source values and order |
| is_last | exact copy of the corresponding FrvSourceChunk.is_last |

The source_id must be identical in both passes and must equal the formal
P-SOURCE source identity binding. A scalar or array mismatch between a source
chunk and its HAG chunk is a provider failure, fail closed.

The HAG stream additionally requires chunk_id values 0, 1, ... with no gaps;
start_index equals the first original_index in each non-empty chunk; and
adjacent chunks satisfy

    next.start_index == current.start_index + current.point_count

point_count is a positive built-in int. P-HAG never creates an empty chunk. Only
the final chunk has is_last=True, and a one-point source produces exactly one
chunk with point_count=1 and is_last=True. The full stream still conserves the
source 0..N-1 original_index domain.

All arrays are C-contiguous and read-only. Object dtype, implicit conversion,
NaN, Infinity, writable aliases, and input mutation are rejected. Unsupported
rows have HAG_MISSING, support false, zero support count, and three positive
zero numeric values. A negative-tolerance row has HAG_INVALID, HAG_INVALID
reason, and PROVIDER_INVALID quality.
### 4.4 FrvHagProviderProof

Serializable frozen dataclass with slots. Exact field order:

    schema_version: str
    provider_key: str
    provider_role: FrvA2ProviderRole
    interface_version: str
    algorithm_version: str
    configuration_digest: str
    provider_config_digest: str
    source_artifact_reference_id: str
    artifact_reference_id: str
    artifact_components: tuple[FrvA2ArtifactComponentBinding, ...]
    source_content_identity_digest: str
    source_identity_record_digest: str
    source_proof_digest: str
    source_binding_digest: str
    source_point_count: int
    domain_kind: FrvA2DomainKind
    domain_count: int
    original_index_domain_digest: str
    hag_point_content_digest: str
    ground_cell_inventory_digest: str
    ground_point_count: int
    supported_point_count: int
    unsupported_point_count: int
    negative_tolerance_exceeded_point_count: int
    hag_valid_positive_count: int
    hag_valid_near_zero_count: int
    hag_within_negative_tolerance_count: int
    hag_missing_count: int
    hag_invalid_count: int
    hag_out_of_domain_count: int
    coordinate_frame: str
    units: str
    origin_xyz: tuple[float, float, float]
    transform_declaration: str
    pass_through_dimensions: tuple[FrvSourcePassThroughDimension, ...]
    quality_flags: int
    reason_flags: int
    error_codes: tuple[FrvErrorCode, ...]
    proof_digest: str

The proof is the P-HAG result identity; there is no second result digest.
proof_digest is computed from every field except proof_digest. The proof requires
schema/version/key/role constants, strict SHA values, exact source artifact
component tuple, FULL_SOURCE domain, source count equality, existing
FRVA2IDX1 over 0..N-1, classification pass-through, exact
LOCAL_CARTESIAN_XYZ/m/identity context, count conservation, zero out-of-domain
count, existing flag masks, and empty error_codes for success.

The proof validator freezes the following conservation formulas:

    supported_point_count = count(ground_support is True)
    unsupported_point_count = count(ground_support is False)
    supported_point_count + unsupported_point_count == source_point_count

    hag_valid_positive_count
    + hag_valid_near_zero_count
    + hag_within_negative_tolerance_count
    + hag_missing_count
    + hag_invalid_count
    + hag_out_of_domain_count
    == source_point_count

    hag_missing_count == unsupported_point_count
    negative_tolerance_exceeded_point_count == hag_invalid_count
    hag_out_of_domain_count == 0

Negative-tolerance-exceeded points have real local ground support and therefore
belong to supported_point_count. Local support insufficiency belongs to
unsupported_point_count. The success Proof aggregate fields are exactly:

    quality_flags = FrvQualityFlag.NONE
    reason_flags = FrvReasonFlag.NONE
    error_codes = ()

Per-point HAG_MISSING, HAG_INVALID, PROVIDER_INVALID, and
INSUFFICIENT_EVIDENCE values remain only in chunks, the HAG point-content digest,
and the state/count evidence. They are never bitwise-ORed into Proof-level
quality_flags or reason_flags. validate_frv_hag_provider_proof and
validate_finalized_frv_hag_provider_result enforce the same formulas before a
result or binding is exposed. Any provider-level failure creates no Proof,
Result, or BND1.

Public methods:

    to_digest_dict(self) -> dict[str, Any]
    computed_digest(self) -> str
    to_dict(self) -> dict[str, Any]
    @classmethod
    from_dict(cls, data: Mapping[str, Any]) -> FrvHagProviderProof

Serialization uses exact fields, enum values, JSON arrays, existing component
serialization, and rejects all missing/extra keys. Runtime objects, paths,
handles, capabilities, exception text, and source chunk size are excluded.

### 4.5 FrvHagProviderResult

    @dataclass(frozen=True, slots=True, init=False)
    class FrvHagProviderResult:
        proof: FrvHagProviderProof
        source_identity: SourceIdentity
        _finalized_capability: Any

The public constructor always raises TypeError. Only the module-private factory
may construct it:

    _create_finalized_frv_hag_provider_result(
        proof: FrvHagProviderProof,
        source_identity: SourceIdentity,
        *,
        capability: Any,
    ) -> FrvHagProviderResult

The factory checks exact private capability identity and full proof/source
binding. Capability is excluded from repr, equality, serialization, and all
digests. The private factory and capability are not public contract.

### 4.6 FrvHagLoadError

    class FrvHagLoadError(Exception):
        failure_record: FrvA2FailureRecord
        quality_flags: int

    FrvHagLoadError(
        *,
        failure_record: FrvA2FailureRecord,
        quality_flags: int,
    ) -> None

failure_record uses validate_frv_a2_failure_record. quality_flags uses the
existing FrvQualityFlag validator. str(error) equals failure_record.detail
exactly. No path, host, username, platform text, class name, or raw exception
text is public. Raw exceptions are attached only by raise ... from exc.

### 4.7 FrvHagLoadSession and loader

    class FrvHagLoadSession(Protocol):
        def iter_chunks(self) -> Iterator[FrvHagChunk]: ...
        def finalize(self) -> FrvHagProviderResult: ...
        def close(self) -> None: ...
        def __enter__(self) -> FrvHagLoadSession: ...
        def __exit__(
            self,
            exc_type: type[BaseException] | None,
            exc: BaseException | None,
            traceback: TracebackType | None,
        ) -> Literal[False]: ...

Runtime module: algorithms.vegetation.hag_provider

    def load_frv_hag(
        reference: FrvSourceProviderReference,
        source_result: FrvSourceProviderResult,
        *,
        frv_configuration: FrvConfiguration,
        provider_config: FrvHagProviderConfig,
        source_chunk_size: int = FRV_SOURCE_DEFAULT_CHUNK_SIZE,
    ) -> FrvHagLoadSession

Only load_frv_hag is exported from algorithms.vegetation. The private
implementation class is _FrvHagLoadSession and is explicitly
PRIVATE_NAME_NOT_CONTRACTUAL.

Session rules: single-use iterator; iterator call locks; exactly two source
sessions; P-HAG finalize only after Pass 2 natural completion and source
finalization; repeated finalize fails; close is idempotent; __exit__ returns
False; first-failure-wins; provider failure produces no Result or BND1.

### 4.8 Validators and adapter

    validate_frv_hag_provider_config(
        config: FrvHagProviderConfig,
    ) -> None

    validate_frv_hag_chunk(
        chunk: FrvHagChunk,
        *,
        provider_config: FrvHagProviderConfig,
    ) -> None

    validate_frv_hag_provider_proof(
        proof: FrvHagProviderProof,
    ) -> None

    validate_finalized_frv_hag_provider_result(
        result: FrvHagProviderResult,
    ) -> None

    to_frv_a2_hag_provider_binding_evidence(
        result: FrvHagProviderResult,
    ) -> FrvA2ProviderBindingEvidence

The result validator checks capability, proof, source identity, source proof
digest, source binding digest, artifact tuple, domain digest, configuration,
counts, and all HAG digests before success is exposed.

## 5. Digest and BND1 authority

P-HAG reuses canonical_frv_json_bytes, frv_digest, SourceIdentity validators,
FrvConfiguration, FRVA2IDX1, FRVA2BND1, A2 artifact component/reference
authority, existing HAG states, reasons, quality flags, and error codes. It does
not redefine or replace any A1/A2 formula.

### 5.1 Source proof digest

    source_proof_digest = frv_digest({
        "digest_domain": "FRVHAGSRC1",
        "source_proof": source_result.proof.to_dict(),
    })

source_binding_digest is the existing
to_frv_a2_provider_binding_evidence(source_result).binding_digest.

### 5.2 Ground inventory digest

    frv_hag_ground_cell_inventory_digest(
        *,
        source_content_identity_digest: str,
        source_identity_record_digest: str,
        provider_config_digest: str,
        ground_point_count: int,
        cells: tuple[tuple[int, int, float, int], ...],
    ) -> str

Byte framing:

    ASCII FRVHAGGRID1\0
    UTF-8 source_content_identity_digest, NUL
    UTF-8 source_identity_record_digest, NUL
    UTF-8 provider_config_digest, NUL
    uint64 little-endian ground_point_count
    uint64 little-endian cell_count
    per sorted (ix, iy) cell:
        int64 little-endian ix
        int64 little-endian iy
        float64 little-endian minimum_finite_ground_z
        uint64 little-endian cell ground count

Cells are unique and sorted. Negative zero is normalized to positive zero.
The digest helper applies the same normalization to every exact-zero cell
minimum before packing its float64 bytes; digest oracle inputs must therefore
use explicit +0.0 and no negative-zero byte sequence is permitted. Dictionary
order and source chunk size are excluded.

### 5.3 HAG point-content digest

    frv_hag_point_content_digest(
        *,
        source_content_identity_digest: str,
        source_identity_record_digest: str,
        provider_config_digest: str,
        source_point_count: int,
        chunks: Iterable[FrvHagChunk],
    ) -> str

Byte framing:

    ASCII FRVHAGPCD1\0
    UTF-8 source_content_identity_digest, NUL
    UTF-8 source_identity_record_digest, NUL
    UTF-8 provider_config_digest, NUL
    uint64 little-endian source_point_count
    per ascending original_index:
        uint64 little-endian original_index
        uint8 ground_support
        uint16 little-endian support_cell_count
        float64 little-endian ground_z_m
        float64 little-endian raw_hag_m
        float64 little-endian clamped_hag_m
        uint8 hag_evidence_state
        uint32 little-endian reason_flags
        uint32 little-endian quality_flags

Unsupported rows encode support false, count zero, and positive-zero numeric
values. Every ground_z_m, raw_hag_m, and clamped_hag_m digest input is first
zero-normalized, so an explicit +0.0 oracle is byte-identical to runtime output
and IEEE negative zero is forbidden. The digest requires exactly the complete
0..N-1 domain without building an N-length index collection.

### 5.4 Proof digest

computed_digest calls existing frv_digest on a canonical payload with
digest_domain FRVHAGPRF1 and every proof field except proof_digest. No runtime
object, path, host, timestamp, capability, exception, or source chunk size is
included. Before proof construction or comparison, all formal float64 values
are zero-normalized to +0.0; proof equality and digest-oracle tests must reject
negative-zero encoding.

### 5.5 FRVA2BND1 mapping

The adapter sets:

    provider_key = "hag_provider"
    provider_role = FrvA2ProviderRole.HAG
    evidence_class = FrvA2EvidenceClass.FORMAL_PROVIDER
    public_module = "algorithms.vegetation.hag_provider"
    public_type_name = "FrvHagProviderResult"
    public_loader_name = "load_frv_hag"
    interface_version = "frv-hag-provider-v1"
    schema_version = "frv-hag-provider-result-v1"
    algorithm_version = "frv-hag-sparse-grid-idw2-v1"
    domain_kind = FrvA2DomainKind.FULL_SOURCE
    provider_readiness = FrvA2ProviderReadiness.VALIDATED
    sample_index_domain_digest = None
    error_codes = ()
    review_codes = ()
    limitation_codes = ()

provider_readiness = VALIDATED means the formal provider output, its complete
FULL_SOURCE domain, all conservation formulas, identity bindings, digests, and
finalization lifecycle have been validated. It does not mean every point has
usable HAG. The BND1 quality_flags, reason_flags, and error_codes are copied
from the success Proof as NONE, NONE, and empty tuple; per-point flags are not
aggregated into BND1. A provider-level failure emits no BND1.

It copies all proof binding fields and computes binding_digest only with the
existing frv_a2_provider_binding_digest. upstream_binding_digests is the
lexicographically sorted unique tuple:

    (
        ground_cell_inventory_digest,
        hag_point_content_digest,
        source_binding_digest,
        source_proof_digest,
    )

The exact P-HAG artifact_components tuple is the source proof's one
source_file component. artifact_reference_id is re-derived for role HAG and
key hag_provider using the existing A2 reference authority. P-HAG publishes no
FRV-A6 artifact.


## 6. Failure mapping and public error boundary

All failures use existing FrvA2 failure phases, FrvErrorCode values,
FrvReasonFlag values, and FrvQualityFlag values. No new public enum is added.
The public detail is exactly:

    <phase>:<provider_role>:<provider_key>:<field_name>:<error_code>:<reason_flags>

The first failing condition wins. Detail contains no path, host, username,
platform, traceback, raw exception text, or source chunk values. Raw exceptions
cross the boundary only through raise ... from exc.

| Condition | Phase | Field | Error | Reason | Quality |
|---|---|---|---|---|---|
| source result not exactly finalized | STRUCTURE_INTEGRITY | source_result | FRV_PROVIDER_INVALID | PROVIDER_UNCERTAINTY | PROVIDER_INVALID |
| source-result validator failure | STRUCTURE_INTEGRITY | source_result | FRV_PROVIDER_INVALID | PROVIDER_UNCERTAINTY | PROVIDER_INVALID |
| source-reference validator failure | STRUCTURE_INTEGRITY | reference | FRV_PROVIDER_INVALID | PROVIDER_UNCERTAINTY | PROVIDER_INVALID |
| source-pass binding mismatch | IDENTITY_CONTEXT | source_result | FRV_SOURCE_IDENTITY_FAILED | PROVIDER_UNCERTAINTY | SOURCE_IDENTITY_FAILURE |
| classification dimension missing | STRUCTURE_INTEGRITY | classification | FRV_PROVIDER_INVALID | PROVIDER_UNCERTAINTY | PROVIDER_INVALID |
| classification dtype invalid | STRUCTURE_INTEGRITY | classification | FRV_PROVIDER_INVALID | PROVIDER_UNCERTAINTY | PROVIDER_INVALID |
| no class-2 ground point | SEMANTIC_READINESS | ground_points | FRV_HAG_MISSING | PROVIDER_UNCERTAINTY \| HAG_MISSING | PROVIDER_INVALID |
| provider config invalid | STRUCTURE_INTEGRITY | provider_config | FRV_PROVIDER_INVALID | PROVIDER_UNCERTAINTY | PROVIDER_INVALID |
| coordinate frame invalid | IDENTITY_CONTEXT | coordinate_frame | FRV_SOURCE_IDENTITY_FAILED | PROVIDER_UNCERTAINTY | SOURCE_IDENTITY_FAILURE |
| units invalid | IDENTITY_CONTEXT | units | FRV_SOURCE_IDENTITY_FAILED | PROVIDER_UNCERTAINTY | SOURCE_IDENTITY_FAILURE |
| transform invalid | IDENTITY_CONTEXT | transform_declaration | FRV_SOURCE_IDENTITY_FAILED | PROVIDER_UNCERTAINTY | SOURCE_IDENTITY_FAILURE |
| source pass early termination | DOMAIN_ALIGNMENT | source_pass | FRV_STREAM_INCOMPLETE | PROVIDER_UNCERTAINTY | STREAM_INCOMPLETE |
| source pass read failure | DOMAIN_ALIGNMENT | source_pass | FRV_STREAM_INCOMPLETE | PROVIDER_UNCERTAINTY | STREAM_INCOMPLETE |
| ground-grid build failure | SEMANTIC_READINESS | ground_grid | FRV_HAG_INVALID | PROVIDER_UNCERTAINTY | PROVIDER_INVALID |
| nonfinite source XYZ | SEMANTIC_READINESS | xyz | FRV_PROVIDER_INVALID | PROVIDER_UNCERTAINTY | PROVIDER_INVALID |
| nonfinite ground cell | SEMANTIC_READINESS | ground_z | FRV_HAG_INVALID | PROVIDER_UNCERTAINTY | PROVIDER_INVALID |
| nonfinite interpolation | SEMANTIC_READINESS | interpolation | FRV_HAG_INVALID | PROVIDER_UNCERTAINTY | PROVIDER_INVALID |
| original index lost | DOMAIN_ALIGNMENT | original_index | FRV_ORIGINAL_INDEX_INVALID | PROVIDER_UNCERTAINTY | ORIGINAL_INDEX_INVALID |
| original index duplicate | DOMAIN_ALIGNMENT | original_index | FRV_ORIGINAL_INDEX_DUPLICATE | PROVIDER_UNCERTAINTY | ORIGINAL_INDEX_DUPLICATE |
| original index out of range | DOMAIN_ALIGNMENT | original_index | FRV_ORIGINAL_INDEX_INVALID | PROVIDER_UNCERTAINTY | ORIGINAL_INDEX_INVALID |
| point count mismatch | DOMAIN_ALIGNMENT | point_count | FRV_STREAM_INCOMPLETE | PROVIDER_UNCERTAINTY | STREAM_INCOMPLETE |
| source/proof/content digest mismatch | CROSS_PROVIDER | digest | FRV_ARTIFACT_DIGEST_MISMATCH | PROVIDER_UNCERTAINTY | ARTIFACT_DIGEST_MISMATCH |
| repeated iterator call | STRUCTURE_INTEGRITY | session_state | FRV_PROVIDER_INVALID | PROVIDER_UNCERTAINTY | PROVIDER_INVALID |
| premature finalize | DOMAIN_ALIGNMENT | session_state | FRV_STREAM_INCOMPLETE | PROVIDER_UNCERTAINTY | STREAM_INCOMPLETE |
| repeated finalize | STRUCTURE_INTEGRITY | session_state | FRV_PROVIDER_INVALID | PROVIDER_UNCERTAINTY | PROVIDER_INVALID |
| close or cleanup failure | STRUCTURE_INTEGRITY | cleanup | FRV_PROVIDER_INVALID | PROVIDER_UNCERTAINTY | PROVIDER_INVALID |

The no-ground condition uses reason PROVIDER_UNCERTAINTY | HAG_MISSING and
quality PROVIDER_INVALID. For complete streams, local support
insufficiency is a successful per-point review result:
HAG_MISSING, reason INSUFFICIENT_EVIDENCE | HAG_MISSING, quality NONE.
A negative raw HAG below negative_hag_tolerance_m is a successful completed
point with HAG_INVALID, reason HAG_INVALID, and PROVIDER_INVALID quality; it
is counted as review-only and does not become NaN.

A failed session closes both source sessions best-effort, preserves the first
failure, and never returns a finalized result. Cleanup failure is reported only
when no earlier failure exists. All public exception construction and failure
records are validated before raising.

## 7. Dependency and file scope

The implementation has one dependency decision: NumPy is a direct runtime
dependency because the public P-HAG chunk contract and deterministic float64
algorithm use NumPy arrays directly. Modify pyproject.toml to add:

    "numpy>=1.26,<3",

while retaining Python >=3.10 and all existing dependencies. No lock file exists
in the baseline, so no lock-file edit is authorized. The current environment
NumPy 2.3.5 satisfies this range.

No new dependency is allowed for PDAL, rasterio, scipy, pyproj, shapely, lidR,
CSF, machine learning, network access, binary fixtures, or a second LAS reader.
The runtime uses only the existing source-provider boundary and NumPy for
contract arrays and scalar arithmetic.

Implementation files, in one task:

| Action | Path | Purpose |
|---|---|---|
| modify | core/contracts/__init__.py | export the public P-HAG contract names |
| add | core/contracts/frv_hag.py | config, chunk, proof, result, error, session protocol, validators, digests, BND1 adapter |
| modify | algorithms/vegetation/__init__.py | export load_frv_hag only |
| add | algorithms/vegetation/hag_provider.py | private two-pass runtime |
| modify | pyproject.toml | direct NumPy dependency only |
| add | tests/contracts/test_frv_hag_contracts.py | contract fields, enums, immutability, validators, private-result capability |
| add | tests/contracts/test_frv_hag_serialization.py | exact serialization and all digest vectors |
| add | tests/vegetation/test_hag_provider.py | lifecycle, algorithm, binding, failures, cleanup, resources |

No other repository file is in scope. In particular, do not modify P-SOURCE,
P-GEOMETRY, P-WIRE, V4, membership, FRV final artifact, large fixtures, or
binary fixture directories. Tests generate minimal LAS data dynamically in
pytest temporary directories and use load_frv_source; they do not commit LAS
files.

## 8. Internal implementation order

This is one implementation task, not separate authorization layers:

1. Freeze and validate the module-level constants, config, chunk, proof, result,
   error, and session protocol.
2. Add the exact NumPy dependency decision and public exports.
3. Implement config/chunk/proof/result validators and private finalization
   capability before opening a source session.
4. Implement source-pass guards, source-result comparison, classification
   extraction, pass-through checks, and first-failure error boundary.
5. Implement the sparse grid and deterministic support/interpolation functions.
6. Implement Pass 1 and Pass 2 with two independent load_frv_source sessions,
   domain/index conservation, and finalized result comparison.
7. Implement point-content, grid, source, proof, and FRVA2BND1 binding digests.
8. Implement result readiness validation and adapter output.
9. Add the contract, serialization, runtime, failure, and resource tests.
10. Run the verification commands in section 11 and review the diff for scope
    containment.

A result is published only after all validators, both source finalizations,
count/digest comparisons, and cleanup obligations succeed.

## 9. Test matrix

### 9.1 Normal contract cases

- exact config defaults and explicit non-default values;
- config digest stability, key order independence, and rejection of extra keys;
- exact chunk dtypes, C-contiguity, read-only arrays, explicit unsupported
  markers, and rejection of point_count=0;
- point_count=0 FrvHagChunk is rejected by the validator; runtime never emits
  an empty chunk; zero-point source is fail closed by upstream/P-HAG;
- frozen proof field order, enum round trips, exact JSON serialization, and
  proof digest stability;
- public result constructor rejection and private-capability finalization;
- validator rejection of mutation, NaN, Infinity, writable aliases, bad masks,
  bad counts, bad artifact tuples, wrong frame, wrong units, and wrong transform;
- stable source/proof/config/grid/point/proof digest vectors;
- input and interpolation -0.0 cases normalize to +0.0 in chunks, ground
  inventory, Proof comparisons, and digest inputs; explicit +0.0 oracle bytes
  match exactly;
- FRVA2BND1 mapping, sorted upstream digests, artifact binding, and binding
  digest stability.

### 9.2 Algorithm and lifecycle cases

- two independent source sessions, exact Pass 1 and Pass 2 source-result
  equality, and no session reuse;
- classification dimension is present, integer-compatible, and pass-through;
- class 2 points build the expected sparse grid using source proof min bounds,
  half-open floor cells, finite minimum Z, and point counts;
- exact radius boundary, outside-radius exclusion, tie ordering by distance,
  ix, iy, min/max support-cell constraints, inverse-square IDW, and float64;
- positive, zero, negative-within-tolerance, negative-beyond-tolerance,
  local-missing-support, and all-finite outputs;
- original_index chunks in arbitrary chunk boundaries and complete 0..N-1
  reconstruction without materializing a second source artifact;
- natural iterator exhaustion, finalize-before-exhaustion, repeated iterator,
  repeated finalize, idempotent close, context manager behavior, and cleanup;
- source artifact/reference binding and readiness after result finalization.

### 9.3 Failure cases

- malformed or non-finalized source result;
- invalid source reference and source-pass identity mismatch;
- missing or wrong classification dimension/dtype;
- no class-2 ground; class-2 nonfinite Z; nonfinite source XYZ;
- invalid config, source frame, units, transform, or pass-through declaration;
- source pass read failure, early termination, count mismatch, lost/duplicate/
  out-of-range original_index;
- out-of-domain original_index is a provider-level domain failure; successful
  FULL_SOURCE has hag_out_of_domain_count=0, emits no HAG_OUT_OF_DOMAIN success
  chunk, and creates no partial Proof/Result/BND1. The vocabulary remains
  covered by defensive validator tests;
- grid build, cell, support, interpolation, or output nonfinite failure;
- negative tolerance exceeded and local support missing per-point mappings;
- source/proof/grid/point/proof/BND1 digest mismatch;
- repeated iterator/finalize and close/cleanup failure;
- public detail sanitization and first-failure-wins behavior.

### 9.4 Resource and regression cases

- one-point and single-ground-point sources;
- sparse grids with empty cells and large coordinate offsets;
- non-default source chunk sizes and chunk boundaries that split every support
  neighborhood;
- different legal source chunk_size values produce identical normalized chunks,
  Proof, Result, BND1, and all formal digests;
- empty source rejection and large synthetic streaming input;
- memory does not scale with a second full XYZ/attribute materialization;
- existing contract, IO, and vegetation tests remain green;
- dependency import works in the supported Python 3.10+ environment.

## 10. Acceptance criteria and non-goals

The implementation is acceptance-ready only when:

- every public name, signature, field order, dtype, digest framing, lifecycle
  state, and failure mapping in this plan matches code and tests;
- exactly two independent load_frv_source sessions are observable;
- no direct LAS opening or path-trust shortcut exists;
- the result proves FULL_SOURCE original_index conservation and exact source
  identity binding;
- every formal output numeric field is finite, with no NaN missing marker;
- BND1 is validated and downstream-ready only after finalization;
- no out-of-scope file, dependency, enum, artifact, or feature is introduced;
- directed and full verification commands pass.

Non-goals are membership classification, geometry, P-WIRE, V4, final artifact
publication, tree segmentation, clearance, risk distance, and any redesign of
P-SOURCE.

## 11. Verification commands after plan freeze

These commands are specified for the future implementation and are not run by
this planning task:

    E:\Miniforge\envs\treeline\python.exe -m pytest tests/contracts/test_frv_hag_contracts.py tests/contracts/test_frv_hag_serialization.py tests/vegetation/test_hag_provider.py -q

    E:\Miniforge\envs\treeline\python.exe -m pytest tests/contracts tests/io tests/vegetation -q

    E:\Miniforge\envs\treeline\python.exe -m pytest -q

    E:\Miniforge\envs\treeline\python.exe -m ruff check core/contracts/frv_hag.py algorithms/vegetation/hag_provider.py tests/contracts/test_frv_hag_contracts.py tests/contracts/test_frv_hag_serialization.py tests/vegetation/test_hag_provider.py

Verification must also include a read-only scope check:

    git diff --check
    git diff --cached --name-status
    git status --short

The first command is the acceptance gate for the P-HAG task. The directed and
full suites are regression gates. No command above authorizes implementation,
staging, commit, push, or any P-GEOMETRY/P-WIRE/V4 work.

## 12. Plan self-review

| Required closure item | Status |
|---|---|
| independent P-HAG plan | PASS |
| exact four public inputs | PASS |
| exact loader signature and session lifecycle | PASS |
| exact two-pass source lifecycle | PASS |
| classification-2 ground authority and sparse-grid algorithm | PASS |
| fixed defaults and config digest exclusion rule | PASS |
| finite outputs and negative-tolerance behavior | PASS |
| chunk/result/proof/error/session contracts | PASS |
| source identity, original_index, FULL_SOURCE, and readiness binding | PASS |
| point/grid/source/proof digest authority | PASS |
| FRVA2BND1 adapter mapping | PASS |
| failure/error/reason/quality mapping | PASS |
| dependency and file-boundary decision | PASS |
| normal/failure/resource test matrix | PASS |
| explicit non-goals and authorization boundary | PASS |
| unresolved TODO/TBD/placeholder in implementation contract | PASS |
| P-HAG-PLAN-MAJ-001 exact P-SOURCE config reconstruction | CLOSED |
| P-HAG-PLAN-MAJ-002 FrvHagChunk scalar semantics | CLOSED |
| P-HAG-PLAN-MAJ-003 count/flags/readiness separation | CLOSED |
| P-HAG-PLAN-MAJ-004 empty-chunk and out-of-domain consistency | CLOSED |
| P-HAG-PLAN-MAJ-005 formal positive-zero normalization | CLOSED |

PROPOSED_CONTRACT_ADDITION: NONE

## 13. Final declaration

P_HAG_PLAN_RESULT: REVIEW_CANDIDATE_READY

Blockers:

- P-HAG-BLK-001: CLOSED_BY_PLAN
- P-HAG-BLK-002: CLOSED_BY_PLAN
- P-HAG-BLK-003: CLOSED_BY_PLAN
- P-HAG-BLK-004: CLOSED_BY_PLAN

Implementation mode: ONE_PASS_AFTER_PLAN_FREEZE

This file is a review candidate only. P-HAG implementation is NOT AUTHORIZED
until this plan is frozen/reviewed. P-GEOMETRY, P-WIRE, V4, membership, final
artifact publication, and Git add/commit/push are NOT AUTHORIZED.

## 14. Final report

Baseline

    Branch: feature/algorithm
    HEAD: 73d512442c1a2b5c341168bbdcb1e65f84a9a7b3
    Remote divergence: 0 0
    Initial worktree: ?? .claude/ only; EXCLUDED_AND_IGNORED

Plan file

    Path: docs/frv-a2-p-hag-implementation-plan.md
    Status: PLAN-REVIEW-CANDIDATE
    Lines: 1155
    Result: REVIEW_CANDIDATE_READY

Original blockers

    P-HAG-BLK-001: CLOSED_BY_PLAN
    P-HAG-BLK-002: CLOSED_BY_PLAN
    P-HAG-BLK-003: CLOSED_BY_PLAN
    P-HAG-BLK-004: CLOSED_BY_PLAN

Scope

    Expected: docs/frv-a2-p-hag-implementation-plan.md
    Unexpected: NONE
    Excluded: .claude/ EXCLUDED_AND_IGNORED
    Cached: NONE
    Git writes: NO

Final declaration

    P_HAG_PLAN_RESULT: REVIEW_CANDIDATE_READY
    Original blockers: CLOSED_BY_PLAN
    Implementation mode: ONE_PASS_AFTER_PLAN_FREEZE
    P-HAG implementation: NOT AUTHORIZED
    P-GEOMETRY/P-WIRE/V4: NOT AUTHORIZED
    Git add/commit/push: NOT AUTHORIZED

完成后立即停止。本文件不触发任何实现、测试、依赖安装、暂存、提交、
推送或后续阶段授权。

P_HAG_PLAN_FINAL_FIX:
COMPLETE_PENDING_FREEZE_AUTHORIZATION

P-HAG-PLAN-MAJ-001:
CLOSED

P-HAG-PLAN-MAJ-002:
CLOSED

P-HAG-PLAN-MAJ-003:
CLOSED

P-HAG-PLAN-MAJ-004:
CLOSED

P-HAG-PLAN-MAJ-005:
CLOSED

No TODO/TBD:
PASS

Exact contracts:
PASS

Internal consistency:
PASS

ONE_PASS implementable:
YES

P-HAG implementation:
NOT AUTHORIZED

Git add/commit/push:
NOT AUTHORIZED
