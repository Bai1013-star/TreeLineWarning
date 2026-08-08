"""Formal operator command for the full-resolution ground classification producer.

Authority: docs/full-resolution-ground-classification-implementation-plan.md
section 6 (PLAN-FROZEN) and the companion frozen design.  The CLI requires an
explicit formal P-SOURCE reference built from a raw LAS; there is no plain-path
bypass around P-SOURCE authority.  It runs FGC-PMF-RLP-v1, publishes the
classified derivative with the completion manifest last, then generates bounded
Ground Human Gate review material from that exact validated publication.

Usage::

    E:\\Miniforge\\envs\\treeline\\python.exe scripts/run_ground_classification.py \\
        --source-las <raw.las> \\
        --output-root <output-root> \\
        [--policy-json <reviewed-policy.json>] \\
        [--p-hag-cell-size <m>] \\
        [--review-output <dir>] \\
        [--progress]

The CLI never invokes downstream A1/A2A/A2B/A3/A4/W1/V1/V2, never modifies the
raw LAS, and never claims production certification.  A successful run reports
``READY_FOR_OWNER_HUMAN_GATE_REVIEW``; the Owner Human Gate decision remains
PENDING.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import signal
import sys
from pathlib import Path
from typing import Any

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from algorithms.ground.classification import (  # noqa: E402
    FgcExecutionError,
    classify_ground,
)
from algorithms.ground.visualization import (  # noqa: E402
    FgcReviewSamplingConfig,
    FgcVisualizationError,
    generate_ground_human_gate_material,
)
from algorithms.io import load_frv_source  # noqa: E402
from algorithms.io.discovery import stable_source_id  # noqa: E402
from core.contracts.frv import (  # noqa: E402
    FRV_CONFIGURATION_SCHEMA_VERSION,
    FRV_SCOPE_DEFINITION_SCHEMA_VERSION,
    FRV_SOURCE_IDENTITY_SCHEMA_VERSION,
    FrvConfiguration,
    FrvConfigurationPurpose,
    frv_digest,
)
from core.contracts.frv_a2 import (  # noqa: E402
    FrvA2ArtifactComponentBinding,
    FrvA2ArtifactComponentKind,
    FrvA2ProviderRole,
)
from core.contracts.frv_hag import (  # noqa: E402
    FRV_HAG_ALGORITHM_VERSION,
    FRV_HAG_CONFIG_SCHEMA_VERSION,
    FRV_HAG_PROVIDER_INTERFACE_VERSION,
    FRV_HAG_PROVIDER_KEY,
    FrvHagProviderConfig,
    derive_frv_hag_provider_config_digest,
)
from core.contracts.frv_source import (  # noqa: E402
    FRV_SOURCE_DEFAULT_CHUNK_SIZE,
    FRV_SOURCE_PROVIDER_ALGORITHM_VERSION,
    FRV_SOURCE_PROVIDER_CONFIG_SCHEMA_VERSION,
    FRV_SOURCE_PROVIDER_INTERFACE_VERSION,
    FRV_SOURCE_PROVIDER_KEY,
    FRV_SOURCE_PROVIDER_SCHEMA_VERSION,
    FrvSourceCanonicalContentPolicy,
    FrvSourceCoordinateConversionPolicy,
    FrvSourceInvalidPointPolicy,
    FrvSourceLoadError,
    FrvSourceModel,
    FrvSourceProviderConfig,
    FrvSourceProviderReference,
    derive_frv_source_provider_config_digest,
)
from core.contracts.ground_classification import (  # noqa: E402
    FGC_DIAGNOSTICS_DIR,
    GroundClassificationPolicyConfig,
)

_EXIT_OK = 0
_EXIT_FAIL_CLOSED = 1
_EXIT_USAGE = 2

_RESULT_STATUS_OK = "READY_FOR_OWNER_HUMAN_GATE_REVIEW"


class _CliFailure(Exception):
    """Structured CLI fail-closed condition with a stable code."""

    def __init__(self, code: str, message: str, *, blocked: bool = True) -> None:
        super().__init__(message)
        self.code = code
        self.blocked = blocked


class _CancellationToken:
    """SIGINT-backed cancellation token consumed by the frozen FGC API."""

    def __init__(self) -> None:
        self._cancelled = False
        self._previous = None

    def is_cancelled(self) -> bool:
        return self._cancelled

    def _handle(self, signum: int, frame: Any) -> None:
        self._cancelled = True

    def __enter__(self) -> _CancellationToken:
        self._previous = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, self._handle)
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        signal.signal(signal.SIGINT, self._previous)
        return False


class _PrintingProgress:
    """Deduplicated console progress reporter for the frozen progress protocol."""

    def __init__(self) -> None:
        self._last: str | None = None

    def update(self, progress: Any) -> None:
        stage = getattr(progress, "stage", "?")
        processed = int(getattr(progress, "processed_points", 0))
        total = int(getattr(progress, "total_points", 0))
        message = str(getattr(progress, "message", ""))
        line = f"[{stage}] {processed}/{total} {message}".rstrip()
        if line != self._last:
            print(line)
            self._last = line


# ---------------------------------------------------------------------------
# Formal P-SOURCE ingress builders
# ---------------------------------------------------------------------------


def _build_raw_source_identity(raw_path: Path, file_sha: str, header: Any) -> Any:
    """Build the formal SourceIdentity for a raw LAS (write-time-independent)."""
    import core.contracts.frv as frv

    path = str(raw_path)
    source_id = stable_source_id(raw_path, raw_path.parent)
    relocation_payload = {
        "audit_schema_version": frv.FRV_RELOCATION_AUDIT_SCHEMA_VERSION,
        "path_style": "windows",
        "recorded_path_raw": path,
        "resolved_path_raw": path,
        "recorded_path_lexical_normalized": path,
        "resolved_path_lexical_normalized": path,
        "resolved_final_path": None,
        "resolution_method": "exact_match",
        "case_comparison_mode": "case_insensitive",
        "platform_name": "windows",
        "run_id": f"fgc-{file_sha[:16]}",
        "audit_timestamp": "2026-01-01T00:00:00Z",
        "content_identity_result": "matched",
        "warning_flags": 0,
    }
    relocation_payload["audit_digest"] = frv.frv_digest(relocation_payload)
    relocation = frv.FrvRelocationAudit.from_dict(relocation_payload)
    content_payload = {
        "source_identity_schema_version": frv.FRV_SOURCE_IDENTITY_SCHEMA_VERSION,
        "source_id": source_id,
        "source_file_sha256": file_sha,
        "canonical_point_content_digest": None,
        "canonical_point_content_spec_digest": None,
        "canonical_point_content_spec_version": None,
        "source_point_count": int(header.point_count),
        "source_bounds_min_xyz": [
            float(header.mins[0]),
            float(header.mins[1]),
            float(header.mins[2]),
        ],
        "source_bounds_max_xyz": [
            float(header.maxs[0]),
            float(header.maxs[1]),
            float(header.maxs[2]),
        ],
        "coordinate_frame": "LOCAL_CARTESIAN_XYZ",
        "units": "m",
        "origin_context": [0.0, 0.0, 0.0],
    }
    content_digest = frv.frv_digest(content_payload)
    record_payload = {
        **content_payload,
        "recorded_source_path": path,
        "resolved_access_path": path,
        "relocation_audit_audit_digest": relocation.audit_digest,
        "source_equivalence_audit_audit_digest": None,
        "source_content_identity_digest": content_digest,
    }
    return frv.SourceIdentity.from_dict(
        {
            **content_payload,
            "recorded_source_path": path,
            "resolved_access_path": path,
            "relocation_audit": relocation.to_dict(),
            "source_equivalence_audit": None,
            "source_content_identity_digest": content_digest,
            "source_identity_record_digest": frv.frv_digest(record_payload),
        }
    )


def _build_provider_config() -> FrvSourceProviderConfig:
    return FrvSourceProviderConfig(
        schema_version=FRV_SOURCE_PROVIDER_CONFIG_SCHEMA_VERSION,
        source_model=FrvSourceModel.SINGLE_FILE_ONLY,
        coordinate_conversion_policy=FrvSourceCoordinateConversionPolicy.FORBID,
        invalid_point_policy=FrvSourceInvalidPointPolicy.FAIL_CLOSED,
        canonical_content_policy=FrvSourceCanonicalContentPolicy.VERIFY_IF_DECLARED,
        pass_through_dimensions=(),
        provider_config_digest=derive_frv_source_provider_config_digest(
            schema_version=FRV_SOURCE_PROVIDER_CONFIG_SCHEMA_VERSION,
            source_model=FrvSourceModel.SINGLE_FILE_ONLY,
            coordinate_conversion_policy=FrvSourceCoordinateConversionPolicy.FORBID,
            invalid_point_policy=FrvSourceInvalidPointPolicy.FAIL_CLOSED,
            canonical_content_policy=FrvSourceCanonicalContentPolicy.VERIFY_IF_DECLARED,
            pass_through_dimensions=(),
        ),
    )


def _build_configuration() -> FrvConfiguration:
    """Reviewed/frozen-policy formal configuration (REVIEWED_FIXED_POLICY).

    Null calibration digest; runtime complete; EXHAUSTIVE_REGION scope; a
    non-null wire provider binding is required by the purpose closure.  P-HAG
    compatibility remains a proof-only replay and never enters FGC numerics.
    """
    import core.contracts.frv as frv

    wire_digest = frv_digest(
        {
            "digest_domain": "frv-config-wire-binding-v1",
            "interface_version": "frv-wire-provider-v1",
        }
    )
    payload = {
        "schema_version": FRV_CONFIGURATION_SCHEMA_VERSION,
        "configuration_version": "fgc-reviewed-fixed-v1",
        "algorithm_version": "alg-1",
        "decision_rule_version": "rule-1",
        "canonicalization_version": frv._FRV_CANONICALIZATION_VERSION,
        "source_identity_schema_version": FRV_SOURCE_IDENTITY_SCHEMA_VERSION,
        "canonical_point_content_spec_version": (
            frv.FRV_CANONICAL_POINT_CONTENT_SCHEMA_VERSION
        ),
        "scope_schema_version": FRV_SCOPE_DEFINITION_SCHEMA_VERSION,
        "scope_construction_version": "sc-1",
        "scope_geometry_encoding_version": frv.FRV_SCOPE_GEOMETRY_ENCODING_SCHEMA_VERSION,
        "coverage_manifest_schema_version": frv.FRV_SCOPE_COVERAGE_MANIFEST_SCHEMA_VERSION,
        "artifact_schema_version": frv.FRV_MEMBERSHIP_ARTIFACT_SCHEMA_VERSION,
        "block_schema_version": frv.FRV_ARTIFACT_BLOCK_SCHEMA_VERSION,
        "payload_schema_version": frv.FRV_ARTIFACT_PAYLOAD_SCHEMA_VERSION,
        "override_schema_version": frv.FRV_MANUAL_OVERRIDE_SCHEMA_VERSION,
        "scope_mode": frv.FrvScopeMode.EXHAUSTIVE_REGION,
        "scope_policy_digest": wire_digest,
        "override_policy_digest": wire_digest,
        "feature_policy_digest": wire_digest,
        "resource_policy_digest": wire_digest,
        "hag_provider_interface_version": "hag-1",
        "geometry_provider_interface_version": "geo-1",
        "scan_provider_interface_version": "scan-1",
        "hag_provider_config_digest": wire_digest,
        "geometry_provider_config_digest": wire_digest,
        "scan_provider_config_digest": wire_digest,
        "wire_provider_interface_version": "frv-wire-provider-v1",
        "wire_provider_config_digest": wire_digest,
        "feature_policy_version": "fp-1",
        "membership_parameter_schema_version": "mp-1",
        "calibration_record_digest": None,
        "allow_wire_body_hard_exclusion": False,
        "allow_wire_body_confirmed": False,
        "allow_review_routing_reduction": False,
        "allow_source_specific_selector": False,
        "configuration_purpose": FrvConfigurationPurpose.REVIEWED_FIXED_POLICY,
        "configuration_is_runtime_complete": True,
        "formal_completeness_allowed": True,
        "membership_artifact_publication_allowed": True,
    }
    payload["configuration_digest"] = frv.frv_digest(payload)
    return frv.FrvConfiguration.from_dict(payload)


def _build_hag_config(cell_size_m: float) -> FrvHagProviderConfig:
    return FrvHagProviderConfig(
        schema_version=FRV_HAG_CONFIG_SCHEMA_VERSION,
        provider_key=FRV_HAG_PROVIDER_KEY,
        provider_role=FrvA2ProviderRole.HAG,
        interface_version=FRV_HAG_PROVIDER_INTERFACE_VERSION,
        algorithm_version=FRV_HAG_ALGORITHM_VERSION,
        cell_size_m=cell_size_m,
        support_radius_m=3.0,
        min_support_cells=1,
        max_support_cells=16,
        negative_hag_tolerance_m=0.1,
        classification_dimension="classification",
        provider_config_digest=derive_frv_hag_provider_config_digest(
            schema_version=FRV_HAG_CONFIG_SCHEMA_VERSION,
            provider_key=FRV_HAG_PROVIDER_KEY,
            provider_role=FrvA2ProviderRole.HAG,
            interface_version=FRV_HAG_PROVIDER_INTERFACE_VERSION,
            algorithm_version=FRV_HAG_ALGORITHM_VERSION,
            cell_size_m=cell_size_m,
            support_radius_m=3.0,
            min_support_cells=1,
            max_support_cells=16,
            negative_hag_tolerance_m=0.1,
            classification_dimension="classification",
        ),
    )


def _build_reference(
    raw_path: Path,
    identity: Any,
    provider_config: FrvSourceProviderConfig,
    configuration: FrvConfiguration,
) -> FrvSourceProviderReference:
    sha = identity.source_file_sha256
    size = int(raw_path.stat().st_size)
    component = FrvA2ArtifactComponentBinding(
        component_key="source_file",
        component_kind=FrvA2ArtifactComponentKind.SOURCE_FILE,
        sha256=sha,
        size_bytes=size,
    )
    artifact_id = (
        f"frv-a2-ref-v1|{FrvA2ProviderRole.SOURCE_SCAN.value}|source_scan|"
        f"source_file,source_file,{sha},{size}"
    )
    return FrvSourceProviderReference(
        provider_role=FrvA2ProviderRole.SOURCE_SCAN,
        provider_key=FRV_SOURCE_PROVIDER_KEY,
        artifact_reference_id=artifact_id,
        artifact_components=(component,),
        runtime_access_path=raw_path,
        source_identity=identity,
        canonical_point_content_spec=None,
        configuration_digest=configuration.configuration_digest,
        provider_config_digest=provider_config.provider_config_digest,
        interface_version=FRV_SOURCE_PROVIDER_INTERFACE_VERSION,
        schema_version=FRV_SOURCE_PROVIDER_SCHEMA_VERSION,
        algorithm_version=FRV_SOURCE_PROVIDER_ALGORITHM_VERSION,
    )


def _load_source_result(
    reference: FrvSourceProviderReference,
    provider_config: FrvSourceProviderConfig,
    configuration: FrvConfiguration,
) -> Any:
    session = load_frv_source(
        reference,
        frv_configuration=configuration,
        provider_config=provider_config,
        chunk_size=FRV_SOURCE_DEFAULT_CHUNK_SIZE,
    )
    try:
        for _ in session.iter_chunks():
            pass
        return session.finalize()
    finally:
        try:
            session.close()
        except Exception:  # noqa: BLE001 - cleanup must not mask the result
            pass


def _file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1 << 20)
            if not block:
                break
            hasher.update(block)
    return hasher.hexdigest()


# ---------------------------------------------------------------------------
# Arguments and policy
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run_ground_classification",
        description=(
            "Run the full-resolution ground classification producer over a "
            "formal P-SOURCE reference and generate Ground Human Gate review "
            "material."
        ),
    )
    parser.add_argument(
        "--source-las",
        required=True,
        type=Path,
        help="Raw uncompressed LAS input (formal P-SOURCE ingress).",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        type=Path,
        help="Output root for the ground-classification/<source-id>/<sha-prefix>/ namespace.",
    )
    parser.add_argument(
        "--policy-json",
        type=Path,
        default=None,
        help="Optional reviewed FGC policy JSON (must match the frozen policy digest).",
    )
    parser.add_argument(
        "--p-hag-cell-size",
        type=float,
        default=1.0,
        help="P-HAG compatibility cell size in metres (proof-only, never a decision input).",
    )
    parser.add_argument(
        "--review-output",
        type=Path,
        default=None,
        help="Directory for the review HTML/summary (default: transaction/diagnostics).",
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Print FGC progress lines to stdout.",
    )
    parser.add_argument(
        "--sample-cap",
        type=int,
        default=20_000,
        help="Deterministic per-state review sample cap (confirmed/review).",
    )
    parser.add_argument(
        "--non-ground-sample-cap",
        type=int,
        default=15_000,
        help="Deterministic non-ground context sample cap.",
    )
    return parser.parse_args(argv)


def _load_policy(policy_json: Path | None) -> GroundClassificationPolicyConfig:
    frozen = GroundClassificationPolicyConfig()
    if policy_json is None:
        return frozen
    try:
        payload = json.loads(policy_json.read_text(encoding="utf-8"))
        policy = GroundClassificationPolicyConfig.from_dict(payload)
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise _CliFailure(
            "FGC_POLICY_INVALID",
            f"policy JSON is not a valid fgc-ground-policy-v1: {exc}",
        ) from exc
    if policy.policy_digest != frozen.policy_digest:
        raise _CliFailure(
            "FGC_POLICY_NOT_REVIEWED",
            "policy JSON does not match the frozen reviewed FGC policy digest",
        )
    return policy


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        raw_path = args.source_las.resolve()
        if not raw_path.is_file():
            raise _CliFailure(
                "SOURCE_MISSING",
                f"source LAS does not exist: {raw_path}",
            )
        if raw_path.suffix.lower() != ".las":
            raise _CliFailure(
                "SOURCE_FORMAT_REJECTED",
                "only uncompressed .las is accepted in V1",
            )
        if args.p_hag_cell_size <= 0.0:
            raise _CliFailure(
                "P_HAG_CONFIG_INVALID",
                "p-hag-cell-size must be positive",
            )

        policy = _load_policy(args.policy_json)
        provider_config = _build_provider_config()
        configuration = _build_configuration()
        hag_config = _build_hag_config(float(args.p_hag_cell_size))

        import laspy

        try:
            with laspy.open(raw_path, mode="r") as reader:
                header = reader.header
        except Exception as exc:  # noqa: BLE001 - normalized fail-closed
            raise _CliFailure(
                "SOURCE_UNREADABLE",
                f"failed to read the LAS header: {exc}",
            ) from exc

        file_sha = _file_sha256(raw_path)
        identity = _build_raw_source_identity(raw_path, file_sha, header)
        reference = _build_reference(raw_path, identity, provider_config, configuration)
        source_result = _load_source_result(reference, provider_config, configuration)

        sampling_config = FgcReviewSamplingConfig(
            sample_cap_per_state=args.sample_cap,
            non_ground_sample_cap=args.non_ground_sample_cap,
        )
        progress = _PrintingProgress() if args.progress else None
        with _CancellationToken() as cancellation:
            result = classify_ground(
                reference,
                source_result,
                configuration,
                hag_config,
                args.output_root,
                policy,
                progress=progress,
                cancellation=cancellation,
            )

        transaction = (
            args.output_root
            / "ground-classification"
            / identity.source_id
            / identity.source_file_sha256[:16]
        )
        review_dir = args.review_output.resolve() if args.review_output else (
            transaction / FGC_DIAGNOSTICS_DIR
        )
        review = generate_ground_human_gate_material(
            transaction,
            review_dir,
            sampling_config,
        )

        _print_result(result, review)
        print(f"FGC_CLI_RESULT: {_RESULT_STATUS_OK}")
        return _EXIT_OK
    except _CliFailure as exc:
        status = (
            f"FGC_RUNTIME_BLOCKED:{exc.code}"
            if exc.blocked
            else f"FGC_RUNTIME_FAILED:{exc.code}"
        )
        print(f"FGC_CLI_RESULT: {status}", file=sys.stderr)
        print(str(exc), file=sys.stderr)
        return _EXIT_FAIL_CLOSED
    except FgcExecutionError as exc:
        print(f"FGC_CLI_RESULT: FGC_RUNTIME_BLOCKED:{exc.code}", file=sys.stderr)
        print(str(exc), file=sys.stderr)
        return _EXIT_FAIL_CLOSED
    except FrvSourceLoadError as exc:
        print("FGC_CLI_RESULT: FGC_RUNTIME_BLOCKED:SOURCE_LOAD_FAILED", file=sys.stderr)
        print(str(exc), file=sys.stderr)
        return _EXIT_FAIL_CLOSED
    except FgcVisualizationError as exc:
        print(f"FGC_CLI_RESULT: FGC_RUNTIME_FAILED:{exc.code}", file=sys.stderr)
        print(str(exc), file=sys.stderr)
        return _EXIT_FAIL_CLOSED
    except (ValueError, TypeError, OSError) as exc:
        print(
            f"FGC_CLI_RESULT: FGC_RUNTIME_BLOCKED:{type(exc).__name__}",
            file=sys.stderr,
        )
        print(str(exc), file=sys.stderr)
        return _EXIT_FAIL_CLOSED


def _print_result(result: dict[str, Any], review: dict[str, Any]) -> None:
    completion = result["completion_manifest"]
    counts = review["counts"]
    coverage = review.get("coverage")
    coverage_line = "n/a"
    if coverage is not None:
        coverage_line = (
            f"occupancy={coverage['occupancy_fraction']} "
            f"unsupported={coverage['unsupported_component_count']}"
        )
    print(f"FGC published: {result['classified_las_path']}")
    print(
        f"points={counts['total_points']} "
        f"confirmed={counts['confirmed_ground_count']} "
        f"review={counts['review_ground_count']} "
        f"non_ground={counts['non_ground_count']} "
        f"class2={counts['class_2_published_count']}"
    )
    print(f"coverage: {coverage_line}")
    print(f"completion manifest: {result['completion_manifest'] is not None}")
    print(f"completion digest: {completion.completion_digest}")
    print(f"policy digest: {completion.policy_digest}")
    artifacts = review["review_artifacts"]
    print(f"review html: {artifacts.get('html_path', '')}")
    print(f"review summary: {artifacts.get('summary_path', '')}")
    print(f"sampled: {review['sampling']['displayed_total']} "
          f"of {review['sampling']['source_count']} points")


if __name__ == "__main__":
    sys.exit(main())
