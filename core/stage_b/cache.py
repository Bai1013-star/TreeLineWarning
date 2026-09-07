"""Strict per-Span artifact cache for Stage-B restart and retry.

The batch manifest contains summary fields for display, but those fields do not
carry the full tree membership or wire geometry required to continue science.
This module keeps the restart seam deliberately small: a cache entry is usable
only when its binding/provenance snapshot and every file consumed by the tree
and wire loaders still match the current run.

This is an engineering transport layer.  It does not implement or duplicate
tree, wire, reconciliation, or clearance mathematics.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from uuid import uuid4

from algorithms.spans.clearance import (
    SpanClearanceResult,
    SpanDangerPoint,
    SpanTreeWireRecord,
)
from algorithms.spans.project_adapter import (
    SourceSharedPreparation,
    SpanAnalysisInput,
    SpanScienceResult,
)
from algorithms.spans.reconcile import reconcile_trees
from algorithms.spans.tree_runner import load_span_tree_outcome
from algorithms.spans.wire import load_span_wire_result

from core.config.fast_baseline import FastBaselineConfig, canonical_digest
from core.integrity import sha256_file
from core.stage_b.bridge import SpanScienceBinding

SPAN_CACHE_SCHEMA = "treeline-stage-b-span-cache-v1"
SPAN_CACHE_INDEX_NAME = "span-cache-index.json"
RUN_CACHE_SCHEMA = "treeline-stage-b-run-cache-v1"
RUN_CACHE_INDEX_NAME = "run-cache-index.json"
_PREPARATION_KEYS = (
    "source_binding_digest",
    "configuration_digest",
    "manifest_reference",
    "authority",
    "tower_membership_used_for_vegetation",
)


class SpanCacheUnavailable(RuntimeError):
    """The runner did not produce the exact files needed for cache reuse."""


def span_cache_path(project_root: Path | str, run_id: str, span_id: str) -> Path:
    """Return the deterministic cache-index path for one product Span."""

    return (
        Path(project_root).expanduser().resolve()
        / "artifacts"
        / "project-runs"
        / "batches"
        / str(run_id)
        / str(span_id)
        / SPAN_CACHE_INDEX_NAME
    )


def run_cache_path(project_root: Path | str, run_id: str) -> Path:
    """Return the deterministic run-level proof path."""

    return (
        Path(project_root).expanduser().resolve()
        / "artifacts"
        / "project-runs"
        / "batches"
        / str(run_id)
        / RUN_CACHE_INDEX_NAME
    )


def _project_relative(project_root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError as exc:
        raise SpanCacheUnavailable("SPAN_CACHE_ARTIFACT_OUTSIDE_PROJECT") from exc


def _resolve_runner_artifact(
    project_root: Path,
    output_root: Path,
    value: Any,
    role: str,
) -> Path:
    if not isinstance(value, str) or not value:
        raise SpanCacheUnavailable(f"SPAN_CACHE_ARTIFACT_MISSING:{role}")
    path = Path(value)
    if not path.is_absolute():
        path = output_root / path
    path = path.resolve()
    _project_relative(project_root, path)
    if not path.is_file():
        raise SpanCacheUnavailable(f"SPAN_CACHE_ARTIFACT_MISSING:{role}")
    return path


def _sha256(path: Path) -> str:
    return sha256_file(path)


def _artifact_reference(project_root: Path, path: Path, role: str) -> dict[str, Any]:
    return {
        "role": role,
        "relative_path": _project_relative(project_root, path),
        "sha256": _sha256(path),
        "byte_size": int(path.stat().st_size),
    }


def _source_path(input_data: SpanAnalysisInput) -> Path | None:
    value = (
        input_data.source_identity.get("resolved_access_path")
        or input_data.source_identity.get("point_cloud_source")
        or input_data.source_identity.get("path")
    )
    if not isinstance(value, str) or not value:
        return None
    return Path(value).expanduser().resolve()


def source_fingerprint(input_data: SpanAnalysisInput) -> dict[str, Any] | None:
    """Read the current source stat and any configured content hash."""

    path = _source_path(input_data)
    if path is None or not path.is_file():
        return None
    try:
        stat = path.stat()
    except OSError:
        return None
    configured_hash = next(
        (
            value
            for key in ("sha256", "source_file_sha256", "source_sha256")
            if isinstance((value := input_data.source_identity.get(key)), str)
            and len(value) == 64
        ),
        None,
    )
    actual_hash = sha256_file(path, reusable=True) if configured_hash is not None else None
    if configured_hash is not None and actual_hash != configured_hash:
        return None
    return {
        "path": str(path),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": actual_hash,
    }


def _source_fingerprint_matches(
    input_data: SpanAnalysisInput, expected: Any
) -> bool:
    current = source_fingerprint(input_data)
    return isinstance(expected, Mapping) and current is not None and current == dict(expected)


def _loader_artifacts(
    project_root: Path,
    output_root: Path,
    result: SpanScienceResult,
) -> list[dict[str, Any]]:
    """Collect exactly the files read by the persisted tree/wire loaders."""

    artifact_files = result.artifact_files
    if not isinstance(artifact_files, Mapping):
        raise SpanCacheUnavailable("SPAN_CACHE_ARTIFACT_MAP_INVALID")
    tree_manifest = _resolve_runner_artifact(
        project_root, output_root, artifact_files.get("tree"), "tree"
    )
    wire_manifest = _resolve_runner_artifact(
        project_root, output_root, artifact_files.get("wire"), "wire"
    )
    if tree_manifest.name != "span_tree_manifest.json":
        raise SpanCacheUnavailable("SPAN_CACHE_TREE_MANIFEST_NAME_INVALID")
    if wire_manifest.name != "span_wire_result.json":
        raise SpanCacheUnavailable("SPAN_CACHE_WIRE_MANIFEST_NAME_INVALID")

    tree_members = tree_manifest.parent / "tree_members.npz"
    wire_arrays = wire_manifest.parent / "w1_arrays.npz"
    for path, role in (
        (tree_members, "tree_members"),
        (wire_arrays, "wire_arrays"),
    ):
        if not path.is_file():
            raise SpanCacheUnavailable(f"SPAN_CACHE_ARTIFACT_MISSING:{role}")
        _project_relative(project_root, path)

    references = [
        _artifact_reference(project_root, tree_manifest, "tree_manifest"),
        _artifact_reference(project_root, tree_members, "tree_members"),
        _artifact_reference(project_root, wire_manifest, "wire_manifest"),
        _artifact_reference(project_root, wire_arrays, "wire_arrays"),
    ]
    # The wire loader consumes this optional diagnostic archive when present.
    wire_diagnostics = wire_manifest.parent / "span_wire_diagnostics.npz"
    if wire_diagnostics.is_file():
        references.append(
            _artifact_reference(project_root, wire_diagnostics, "wire_diagnostics")
        )
    # The canonical executor advertises a clearance path even when its
    # identity-terminal hook was not used.  Preserve it when present, while
    # keeping tree/wire artifacts as the authoritative cache requirements.
    raw_clearance = artifact_files.get("clearance")
    if isinstance(raw_clearance, str) and raw_clearance:
        clearance = Path(raw_clearance)
        if not clearance.is_absolute():
            clearance = output_root / clearance
        clearance = clearance.resolve()
        if clearance.is_file():
            references.append(_artifact_reference(project_root, clearance, "clearance"))
    return references


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{uuid4().hex}")
    temporary.write_text(
        json.dumps(dict(payload), ensure_ascii=False, sort_keys=True, indent=2, default=str)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _run_artifacts(root: Path, run_id: str) -> list[tuple[str, Path, bool]]:
    batch_root = root / "artifacts" / "project-runs" / "batches" / str(run_id)
    return [
        ("batch_manifest", batch_root / "batch.json", True),
        ("global_reconciliation", batch_root / "batch-reconciliation.json", True),
        ("global_membership", batch_root / "tree_segmentation_original.npz", True),
        ("batch_clearance", batch_root / "batch-clearance.json", False),
        # The batch JSON keeps only bounded per-tree clearance records.  The
        # exact danger XYZ payload is persisted by the editor in this pair of
        # sidecars and must be included in the detached proof when present.
        ("danger_points", batch_root / "danger_points_final.npz", False),
        ("danger_points_metadata", batch_root / "danger_points_final.json", False),
    ]


def publish_run_cache_index(project_root: Path | str, run_id: str) -> Path:
    """Publish hashes for persisted run files after final telemetry is written."""

    root = Path(project_root).expanduser().resolve()
    references: list[dict[str, Any]] = []
    for role, path, required in _run_artifacts(root, run_id):
        if not path.is_file():
            if required:
                raise SpanCacheUnavailable(f"RUN_CACHE_ARTIFACT_MISSING:{role}")
            continue
        references.append(_artifact_reference(root, path, role))
    required_roles = {
        role for role, _path, required in _run_artifacts(root, run_id) if required
    }
    if required_roles - {item["role"] for item in references}:
        raise SpanCacheUnavailable("RUN_CACHE_ARTIFACTS_INCOMPLETE")
    path = run_cache_path(root, run_id)
    _write_json_atomic(
        path,
        {
            "schema": RUN_CACHE_SCHEMA,
            "status": "COMPLETE",
            "run_id": str(run_id),
            "artifacts": references,
        },
    )
    return path


def validate_run_cache_index(
    project_root: Path | str, run_id: str
) -> tuple[bool, str]:
    """Validate the detached run-level publication proof before clean resume."""

    root = Path(project_root).expanduser().resolve()
    payload = _load_json(run_cache_path(root, run_id))
    if payload is None:
        return False, "RUN_CACHE_INDEX_MISSING"
    if (
        payload.get("schema") != RUN_CACHE_SCHEMA
        or payload.get("status") != "COMPLETE"
        or payload.get("run_id") != str(run_id)
    ):
        return False, "RUN_CACHE_INDEX_INVALID"
    raw = payload.get("artifacts")
    if not isinstance(raw, list):
        return False, "RUN_CACHE_ARTIFACT_INDEX_INVALID"
    by_role: dict[str, Path] = {}
    for item in raw:
        if not isinstance(item, Mapping):
            return False, "RUN_CACHE_ARTIFACT_REFERENCE_INVALID"
        role = item.get("role")
        relative = item.get("relative_path")
        expected_hash = item.get("sha256")
        expected_size = item.get("byte_size")
        if (
            not isinstance(role, str)
            or role in by_role
            or not isinstance(relative, str)
            or Path(relative).is_absolute()
            or not isinstance(expected_hash, str)
            or type(expected_size) is not int
        ):
            return False, "RUN_CACHE_ARTIFACT_REFERENCE_INVALID"
        path = (root / relative).resolve()
        try:
            path.relative_to(root.resolve())
        except ValueError:
            return False, "RUN_CACHE_ARTIFACT_REFERENCE_INVALID"
        if not path.is_file():
            return False, f"RUN_CACHE_ARTIFACT_MISSING:{role}"
        if int(path.stat().st_size) != expected_size or _sha256(path) != expected_hash:
            return False, f"RUN_CACHE_ARTIFACT_HASH_MISMATCH:{role}"
        by_role[role] = path
    expected_paths = {
        role: path.resolve()
        for role, path, required in _run_artifacts(root, run_id)
        if required or path.is_file()
    }
    for role, expected in expected_paths.items():
        if role not in by_role:
            return False, f"RUN_CACHE_ARTIFACT_MISSING:{role}"
        if by_role[role] != expected:
            return False, f"RUN_CACHE_ARTIFACT_PATH_MISMATCH:{role}"
    return True, "RUN_CACHE_HIT"


def publish_span_cache_index(
    project_root: Path | str,
    run_id: str,
    binding: SpanScienceBinding,
    input_data: SpanAnalysisInput,
    preparation: SourceSharedPreparation,
    config: FastBaselineConfig,
    result: SpanScienceResult,
    output_root: Path | str,
) -> Path:
    """Publish a binding-bound cache index after an exact Span run succeeds."""

    root = Path(project_root).expanduser().resolve()
    output = Path(output_root).expanduser().resolve()
    expected_output = (
        root
        / "artifacts"
        / "project-runs"
        / "batches"
        / str(run_id)
        / binding.product_span.span_id
    )
    if output != expected_output:
        raise SpanCacheUnavailable("SPAN_CACHE_OUTPUT_ROOT_MISMATCH")
    references = _loader_artifacts(root, output, result)
    if input_data.analysis_run_id != run_id:
        raise SpanCacheUnavailable("SPAN_CACHE_INPUT_RUN_MISMATCH")
    if binding.identity.analysis_run_id != run_id:
        raise SpanCacheUnavailable("SPAN_CACHE_BINDING_RUN_MISMATCH")
    current_source_fingerprint = source_fingerprint(input_data)
    if current_source_fingerprint is None:
        raise SpanCacheUnavailable("SPAN_CACHE_SOURCE_PROOF_UNAVAILABLE")
    preparation_reason = _validate_preparation(
        root, input_data, preparation.to_dict(), preparation
    )
    if preparation_reason is not None:
        raise SpanCacheUnavailable(preparation_reason)
    payload = {
        "schema": SPAN_CACHE_SCHEMA,
        "status": "COMPLETE",
        "project_id": input_data.project_id,
        "scan_id": binding.identity.scan_id,
        "group_id": input_data.group_id,
        "product_span_id": binding.product_span.span_id,
        "input_span_id": input_data.span_id,
        "run_id": run_id,
        "output_root": _project_relative(root, output),
        "binding": binding.to_dict(),
        "input": input_data.to_dict(),
        "source_preparation": preparation.to_dict(),
        "source_fingerprint": current_source_fingerprint,
        "configuration": {
            "fast_baseline_digest": config.computed_digest(),
            "science_configuration_digest": input_data.science_configuration_digest,
            "risk_configuration_digest": input_data.risk_configuration_digest,
        },
        "summary": {
            "tree_count": int(result.tree_count),
            "minimum_clearance_m": result.minimum_clearance_m,
            "risk_threshold_m": float(result.risk_threshold_m),
            "risk_counts": dict(result.risk_counts),
        },
        "artifacts": references,
        "diagnostics": dict(result.diagnostics),
    }
    path = output / SPAN_CACHE_INDEX_NAME
    _write_json_atomic(path, payload)
    return path


def _miss(reason: str) -> tuple[None, str]:
    return None, str(reason)


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _authoritative_preparation(value: Mapping[str, Any]) -> dict[str, Any] | None:
    if any(key not in value for key in _PREPARATION_KEYS):
        return None
    result = {key: value.get(key) for key in _PREPARATION_KEYS}
    if not isinstance(result["source_binding_digest"], str):
        return None
    if not isinstance(result["configuration_digest"], str):
        return None
    if not isinstance(result["manifest_reference"], str):
        return None
    if result["authority"] != "FULL_RESOLUTION_RAW":
        return None
    if result["tower_membership_used_for_vegetation"] is not False:
        return None
    return result


def _validate_preparation(
    project_root: Path,
    input_data: SpanAnalysisInput,
    stored: Any,
    current: SourceSharedPreparation | None,
) -> str | None:
    if not isinstance(stored, Mapping):
        return "SPAN_CACHE_PREPARATION_INVALID"
    authoritative = _authoritative_preparation(stored)
    if authoritative is None:
        return "SPAN_CACHE_PREPARATION_INVALID"
    expected_source_digest = canonical_digest(dict(input_data.source_identity))
    if authoritative["source_binding_digest"] != expected_source_digest:
        return "SPAN_CACHE_SOURCE_DIGEST_MISMATCH"
    if authoritative["configuration_digest"] != input_data.science_configuration_digest:
        return "SPAN_CACHE_CONFIGURATION_DIGEST_MISMATCH"
    if current is not None:
        current_values = _authoritative_preparation(current.to_dict())
        if current_values != authoritative:
            return "SPAN_CACHE_CURRENT_PREPARATION_MISMATCH"
    manifest_reference = str(authoritative["manifest_reference"])
    manifest_path = Path(manifest_reference)
    if manifest_path.is_absolute():
        return "SPAN_CACHE_PREPARATION_REFERENCE_INVALID"
    manifest_path = (project_root / manifest_path).resolve()
    try:
        manifest_path.relative_to(project_root.resolve())
    except ValueError:
        return "SPAN_CACHE_PREPARATION_REFERENCE_INVALID"
    manifest = _load_json(manifest_path)
    if manifest is None:
        return "SPAN_CACHE_PREPARATION_MANIFEST_MISSING"
    for key, expected in authoritative.items():
        # The existing source-shared writer binds this manifest by its
        # persisted path and predates a redundant ``manifest_reference``
        # field.  Keep that established schema readable while still checking
        # the path itself above and every other authoritative binding here.
        if key == "manifest_reference" and key not in manifest:
            continue
        if manifest.get(key) != expected:
            return "SPAN_CACHE_PREPARATION_MANIFEST_MISMATCH"
    return None


def _validate_artifacts(
    project_root: Path,
    output_root: Path,
    payload: Mapping[str, Any],
) -> tuple[dict[str, Path] | None, str | None]:
    raw = payload.get("artifacts")
    if not isinstance(raw, list):
        return None, "SPAN_CACHE_ARTIFACT_INDEX_INVALID"
    by_role: dict[str, Path] = {}
    for item in raw:
        if not isinstance(item, Mapping):
            return None, "SPAN_CACHE_ARTIFACT_REFERENCE_INVALID"
        role = item.get("role")
        relative = item.get("relative_path")
        expected_hash = item.get("sha256")
        expected_size = item.get("byte_size")
        if (
            not isinstance(role, str)
            or role in by_role
            or not isinstance(relative, str)
            or Path(relative).is_absolute()
            or not isinstance(expected_hash, str)
            or type(expected_size) is not int
        ):
            return None, "SPAN_CACHE_ARTIFACT_REFERENCE_INVALID"
        path = (project_root / relative).resolve()
        try:
            path.relative_to(project_root.resolve())
        except ValueError:
            return None, "SPAN_CACHE_ARTIFACT_REFERENCE_INVALID"
        if not path.is_file():
            return None, f"SPAN_CACHE_ARTIFACT_MISSING:{role}"
        if int(path.stat().st_size) != expected_size or _sha256(path) != expected_hash:
            return None, f"SPAN_CACHE_ARTIFACT_HASH_MISMATCH:{role}"
        by_role[role] = path

    required = {"tree_manifest", "tree_members", "wire_manifest", "wire_arrays"}
    missing = sorted(required - set(by_role))
    if missing:
        return None, "SPAN_CACHE_ARTIFACT_MISSING:" + ",".join(missing)
    expected_paths = {
        "tree_manifest": output_root / "tree" / "span_tree_manifest.json",
        "tree_members": output_root / "tree" / "tree_members.npz",
        "wire_manifest": output_root / "wire" / "span_wire_result.json",
        "wire_arrays": output_root / "wire" / "w1_arrays.npz",
    }
    for role, expected in expected_paths.items():
        if by_role[role] != expected.resolve():
            return None, f"SPAN_CACHE_ARTIFACT_PATH_MISMATCH:{role}"
    diagnostics_path = output_root / "wire" / "span_wire_diagnostics.npz"
    if diagnostics_path.is_file() and by_role.get("wire_diagnostics") != diagnostics_path.resolve():
        return None, "SPAN_CACHE_ARTIFACT_UNHASHED:wire_diagnostics"
    return by_role, None


def _origin_xyz(input_data: SpanAnalysisInput) -> tuple[float, float, float]:
    value = input_data.source_identity.get(
        "origin_context",
        input_data.source_identity.get("source_bounds_min_xyz", (0.0, 0.0, 0.0)),
    )
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return 0.0, 0.0, 0.0
    return tuple(float(item) for item in value)


def _as_optional_xyz(value: Any) -> tuple[float, float, float] | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError("clearance XYZ must have three coordinates")
    return tuple(float(item) for item in value)


def _load_clearance(path: Path) -> SpanClearanceResult:
    payload = _load_json(path)
    if payload is None:
        raise ValueError("clearance artifact is not valid JSON")
    parsed = load_span_clearance_payload(payload)
    if parsed is None:
        raise ValueError("clearance artifact payload is invalid")
    return parsed


def load_span_clearance_payload(payload: Any) -> SpanClearanceResult | None:
    """Hydrate the persisted Stage-B clearance shape without recomputing it."""

    if not isinstance(payload, Mapping):
        return None
    try:
        records = tuple(
            SpanTreeWireRecord(
                global_tree_id=int(item["global_tree_id"]),
                external_name=str(item["external_name"]),
                persistent_tree_id=item.get("persistent_tree_id"),
                line_id=int(item["line_id"]),
                span_id=int(item["span_id"]),
                span_name=str(item["span_name"]),
                nearest_wire_id=(
                    None
                    if item.get("nearest_wire_id") is None
                    else int(item["nearest_wire_id"])
                ),
                minimum_clearance_m=(
                    None
                    if item.get("minimum_clearance_m") is None
                    else float(item["minimum_clearance_m"])
                ),
                nearest_tree_original_index=(
                    None
                    if item.get("nearest_tree_original_index") is None
                    else int(item["nearest_tree_original_index"])
                ),
                nearest_tree_xyz_local_m=_as_optional_xyz(
                    item.get("nearest_tree_xyz_local_m")
                ),
                nearest_wire_xyz_local_m=_as_optional_xyz(
                    item.get("nearest_wire_xyz_local_m")
                ),
                tree_point_count=int(item.get("tree_point_count", 0)),
                evaluated_edge_count=int(item.get("evaluated_edge_count", 0)),
                status=str(item.get("status", "UNKNOWN")),
                danger_point_count=int(item.get("danger_point_count", 0)),
            )
            for item in payload.get("records", [])
        )
        diagnostics = dict(payload.get("diagnostics") or {})
        return SpanClearanceResult(
            records=records,
            status=str(payload.get("status", "UNKNOWN")),
            global_minimum_clearance_m=(
                None
                if payload.get("global_minimum_clearance_m") is None
                else float(payload["global_minimum_clearance_m"])
            ),
            edge_count=int(
                diagnostics.get("edge_count", payload.get("edge_count", 0)) or 0
            ),
            diagnostics=diagnostics,
            danger_points=tuple(
                SpanDangerPoint(
                    global_tree_id=int(item["global_tree_id"]),
                    persistent_tree_id=item.get("persistent_tree_id"),
                    original_index=int(item["original_index"]),
                    xyz_local_m=tuple(float(value) for value in item["xyz_local_m"]),
                )
                for item in payload.get("danger_points", [])
            ),
        )
    except (KeyError, TypeError, ValueError):
        return None


def load_exact_span_cache(
    project_root: Path | str,
    run_id: str,
    binding: SpanScienceBinding,
    config: FastBaselineConfig,
    *,
    preparation: SourceSharedPreparation | None = None,
) -> tuple[SpanScienceResult | None, str]:
    """Load one exact cache entry, returning an explicit miss reason."""

    root = Path(project_root).expanduser().resolve()
    index_path = span_cache_path(root, run_id, binding.product_span.span_id)
    payload = _load_json(index_path)
    if payload is None:
        return _miss("SPAN_CACHE_INDEX_MISSING")
    if payload.get("schema") != SPAN_CACHE_SCHEMA or payload.get("status") != "COMPLETE":
        return _miss("SPAN_CACHE_INDEX_SCHEMA_INVALID")
    input_data = binding.science_input
    if (
        payload.get("project_id") != input_data.project_id
        or payload.get("scan_id") != binding.identity.scan_id
        or payload.get("group_id") != input_data.group_id
        or payload.get("product_span_id") != binding.product_span.span_id
        or payload.get("input_span_id") != input_data.span_id
        or payload.get("run_id") != run_id
    ):
        return _miss("SPAN_CACHE_BINDING_IDENTITY_MISMATCH")
    if payload.get("output_root") != index_path.parent.relative_to(root).as_posix():
        return _miss("SPAN_CACHE_OUTPUT_ROOT_MISMATCH")
    if not _source_fingerprint_matches(input_data, payload.get("source_fingerprint")):
        return _miss("SPAN_CACHE_SOURCE_FINGERPRINT_MISMATCH")
    if payload.get("binding") != binding.to_dict():
        return _miss("SPAN_CACHE_BINDING_MISMATCH")
    stored_input = payload.get("input")
    if not isinstance(stored_input, Mapping):
        return _miss("SPAN_CACHE_INPUT_INVALID")
    expected_input = input_data.to_dict()
    if preparation is not None:
        expected_input["validated_source_preparation_references"] = [
            preparation.manifest_reference
        ]
    if preparation is None:
        # A clean COMPLETE restart can validate the prior preparation proof
        # without executing source preparation again.  The reference itself
        # remains part of the exact input snapshot.
        references = stored_input.get("validated_source_preparation_references")
        if not isinstance(references, list):
            return _miss("SPAN_CACHE_INPUT_PREPARATION_REFERENCE_INVALID")
        expected_input["validated_source_preparation_references"] = list(references)
    if dict(stored_input) != expected_input:
        return _miss("SPAN_CACHE_INPUT_MISMATCH")
    configuration = payload.get("configuration")
    if not isinstance(configuration, Mapping):
        return _miss("SPAN_CACHE_CONFIGURATION_INVALID")
    if (
        configuration.get("fast_baseline_digest") != config.computed_digest()
        or configuration.get("science_configuration_digest")
        != input_data.science_configuration_digest
        or configuration.get("risk_configuration_digest")
        != input_data.risk_configuration_digest
    ):
        return _miss("SPAN_CACHE_CONFIGURATION_MISMATCH")
    preparation_reason = _validate_preparation(
        root, input_data, payload.get("source_preparation"), preparation
    )
    if preparation_reason is not None:
        return _miss(preparation_reason)
    artifacts, artifact_reason = _validate_artifacts(root, index_path.parent, payload)
    if artifacts is None:
        return _miss(artifact_reason or "SPAN_CACHE_ARTIFACT_INDEX_INVALID")
    try:
        from algorithms.spans.canonical_executor import _span_record

        span_record = _span_record(input_data, config)
        tree_outcome = load_span_tree_outcome(span_record, artifacts["tree_manifest"].parent)
        wire_result = load_span_wire_result(artifacts["wire_manifest"].parent)
        reconciled = reconcile_trees(
            [tree_outcome], config, origin_xyz=_origin_xyz(input_data)
        )
        raw_summary = payload.get("summary")
        if not isinstance(raw_summary, Mapping):
            return _miss("SPAN_CACHE_SUMMARY_INVALID")
        tree_count = int(raw_summary["tree_count"])
        if tree_count != reconciled.final_tree_count:
            return _miss("SPAN_CACHE_TREE_COUNT_MISMATCH")
        clearance = (
            _load_clearance(artifacts["clearance"])
            if "clearance" in artifacts
            else None
        )
        minimum = raw_summary.get("minimum_clearance_m")
        if clearance is not None and clearance.global_minimum_clearance_m != minimum:
            return _miss("SPAN_CACHE_CLEARANCE_SUMMARY_MISMATCH")
        diagnostics = dict(payload.get("diagnostics") or {})
        diagnostics["span_cache"] = {
            "status": "REUSED_EXACT_ARTIFACTS",
            "index": index_path.relative_to(root).as_posix(),
        }
        artifact_files = {
            "tree": str(artifacts["tree_manifest"]),
            "wire": str(artifacts["wire_manifest"]),
        }
        if "clearance" in artifacts:
            artifact_files["clearance"] = str(artifacts["clearance"])
        result = SpanScienceResult(
            tree_count=tree_count,
            minimum_clearance_m=minimum,
            risk_threshold_m=float(raw_summary["risk_threshold_m"]),
            risk_counts=dict(raw_summary.get("risk_counts") or {}),
            artifact_files=artifact_files,
            diagnostics=diagnostics,
            tree_outcome=tree_outcome,
            reconciled=reconciled,
            wire_result=wire_result,
            clearance_result=clearance,
        )
        return result, "SPAN_CACHE_HIT"
    except (OSError, RuntimeError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        return _miss(f"SPAN_CACHE_RELOAD_FAILED:{type(exc).__name__}")


__all__ = [
    "SPAN_CACHE_INDEX_NAME",
    "SPAN_CACHE_SCHEMA",
    "RUN_CACHE_INDEX_NAME",
    "RUN_CACHE_SCHEMA",
    "SpanCacheUnavailable",
    "load_exact_span_cache",
    "load_span_clearance_payload",
    "publish_span_cache_index",
    "publish_run_cache_index",
    "source_fingerprint",
    "span_cache_path",
    "run_cache_path",
    "validate_run_cache_index",
]
