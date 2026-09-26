"""I1 immutable cross-computer bundle export and standalone validation."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
from core.contracts import (
    I1_ELIGIBILITY_POLICY_VERSION,
    I1_INTERFACE_VERSION,
    I1_MANIFEST_FILENAME,
    I1_ORIGIN_SOURCE_KIND,
    I1_ORIGIN_VERIFICATION,
    I1_SCHEMA_VERSION,
    CancellationToken,
    I1ArtifactKind,
    I1ArtifactReference,
    I1BundleManifest,
    I1CoordinateContext,
    I1CoordinateFrame,
    I1CoordinateUnit,
    I1Error,
    I1ErrorCode,
    I1OptionalArtifactState,
    VegetationClusterStatus,
    WireModelStatus,
    WireSeparationResult,
    calculate_bundle_id,
)

from algorithms.vegetation import load_vegetation_associations, load_vegetation_instances
from algorithms.wires import load_wire_models

W1_SCHEMA_VERSION = "wire-models-v2"
W1_ALGORITHM_VERSION = "w1-confirmed-wire-centreline-v3"
V2_SCHEMA_VERSION = "vegetation-instances-v2"
V2_ALGORITHM_VERSION = "v2-generalized-morphology-safe-vegetation-instances-v2"
V2A_SCHEMA_VERSION = "vegetation-associations-v1"
V2A_ALGORITHM_VERSION = "v2a-review-level-vegetation-association-v1"


@dataclass(frozen=True)
class ValidatedI1Bundle:
    """Standalone-validated views; this is not a prediction-side adapter."""

    root: Path
    manifest: I1BundleManifest
    w1_result: Any
    w1_arrays: Mapping[str, np.ndarray]
    v2_result: Any
    v2_arrays: Mapping[str, np.ndarray]
    v2a_result: Any | None = None
    v2a_arrays: Mapping[str, np.ndarray] | None = None


def _error(code: I1ErrorCode, message: str, **details: Any) -> I1Error:
    return I1Error(code, message, **details)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_cancelled(token: CancellationToken | None, stage: str) -> None:
    if token is not None and token.is_cancelled():
        raise _error(I1ErrorCode.CANCELLED, "I1 export was cancelled", stage=stage)


def _safe_join(root: Path, relpath: str) -> Path:
    try:
        pure = PurePosixPath(relpath)
        if (
            pure.is_absolute()
            or "\\" in relpath
            or any(part in {"", ".", ".."} for part in pure.parts)
        ):
            raise ValueError("unsafe relative path")
        candidate = (root / Path(*pure.parts)).resolve()
        if not candidate.is_relative_to(root.resolve()):
            raise ValueError("path escapes bundle root")
        return candidate
    except (TypeError, ValueError) as exc:
        raise _error(I1ErrorCode.CONTRACT, "I1 artifact path is unsafe", path=relpath) from exc


def _load_json(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise _error(I1ErrorCode.INTEGRITY, "unable to read JSON", path=str(path)) from exc
    if not isinstance(raw, dict):
        raise _error(I1ErrorCode.CONTRACT, "JSON root must be an object", path=str(path))
    return raw


def _verify_file(path: Path, digest: str, size: int) -> None:
    if not path.is_file():
        raise _error(I1ErrorCode.INTEGRITY, "referenced artifact is missing", path=str(path))
    if path.stat().st_size != size or _sha(path) != digest:
        raise _error(
            I1ErrorCode.INTEGRITY, "referenced artifact hash or size differs", path=str(path)
        )


def _ensure_no_object_arrays(arrays: Mapping[str, np.ndarray], label: str) -> None:
    if any(np.asarray(value).dtype == object for value in arrays.values()):
        raise _error(I1ErrorCode.SOURCE, "object dtype is forbidden", source=label)


def _ensure_sibling_pair(npz_path: Path, json_path: Path, label: str) -> None:
    if npz_path.suffix != ".npz" or json_path.suffix != ".json" or npz_path.stem != json_path.stem:
        raise _error(I1ErrorCode.SOURCE, "source NPZ/JSON must be named siblings", source=label)


def _load_w1(npz_path: Path, json_path: Path) -> tuple[Any, dict[str, np.ndarray]]:
    _ensure_sibling_pair(npz_path, json_path, "W1")
    try:
        result, arrays = load_wire_models(npz_path, json_path)
    except Exception as exc:
        raise _error(I1ErrorCode.SOURCE, "W1 source validation failed", cause=str(exc)) from exc
    if (
        result.schema_version != W1_SCHEMA_VERSION
        or result.algorithm_version != W1_ALGORITHM_VERSION
    ):
        raise _error(I1ErrorCode.SOURCE, "unsupported W1 version")
    if (
        npz_path.name != PurePosixPath(result.output_npz_relpath).name
        or json_path.name != PurePosixPath(result.output_json_relpath).name
    ):
        raise _error(I1ErrorCode.SOURCE, "W1 sibling filenames do not match its manifest")
    _ensure_no_object_arrays(arrays, "W1")
    _validate_w1_arrays(arrays)
    return result, arrays


def _load_v2(npz_path: Path, json_path: Path) -> tuple[Any, dict[str, np.ndarray]]:
    _ensure_sibling_pair(npz_path, json_path, "V2")
    try:
        result, arrays = load_vegetation_instances(npz_path, json_path)
    except Exception as exc:
        raise _error(I1ErrorCode.SOURCE, "V2 source validation failed", cause=str(exc)) from exc
    if (
        result.schema_version != V2_SCHEMA_VERSION
        or result.algorithm_version != V2_ALGORITHM_VERSION
    ):
        raise _error(I1ErrorCode.SOURCE, "unsupported V2 version")
    if npz_path.name != result.output_npz_filename:
        raise _error(I1ErrorCode.SOURCE, "V2 NPZ filename does not match its manifest")
    _ensure_no_object_arrays(arrays, "V2")
    _validate_v2_arrays(arrays, result)
    return result, arrays


def _load_v2a(npz_path: Path, json_path: Path) -> tuple[Any, dict[str, np.ndarray]]:
    _ensure_sibling_pair(npz_path, json_path, "V2A")
    try:
        result, arrays = load_vegetation_associations(npz_path, json_path)
    except Exception as exc:
        raise _error(I1ErrorCode.SOURCE, "V2A source validation failed", cause=str(exc)) from exc
    if (
        result.schema_version != V2A_SCHEMA_VERSION
        or result.algorithm_version != V2A_ALGORITHM_VERSION
    ):
        raise _error(I1ErrorCode.SOURCE, "unsupported V2A version")
    if npz_path.name != result.output_npz_filename:
        raise _error(I1ErrorCode.SOURCE, "V2A NPZ filename does not match its manifest")
    _ensure_no_object_arrays(arrays, "V2A")
    return result, arrays


def _validate_w1_arrays(arrays: Mapping[str, np.ndarray]) -> None:
    required = {
        "wire_id",
        "wire_status",
        "wire_segment_offsets",
        "segment_wire_id",
        "segment_index",
        "segment_distance_ready",
        "sampled_curve_xyz",
        "sampled_curve_offsets",
        "gap_wire_id",
        "gap_left_segment_index",
        "gap_right_segment_index",
        "gap_start_xyz",
        "gap_end_xyz",
    }
    if missing := required - set(arrays):
        raise _error(I1ErrorCode.SOURCE, "W1 arrays are incomplete", arrays=sorted(missing))
    wire_ids = arrays["wire_id"]
    statuses = arrays["wire_status"]
    segment_wire = arrays["segment_wire_id"]
    segment_index = arrays["segment_index"]
    ready = arrays["segment_distance_ready"]
    offsets = arrays["sampled_curve_offsets"]
    xyz = arrays["sampled_curve_xyz"]
    if (
        wire_ids.dtype != np.int32
        or statuses.dtype != np.uint8
        or segment_wire.dtype != np.int32
        or segment_index.dtype != np.int32
        or ready.dtype != np.bool_
        or offsets.dtype != np.int64
        or xyz.dtype != np.float32
        or xyz.ndim != 2
        or xyz.shape[1] != 3
    ):
        raise _error(I1ErrorCode.SOURCE, "W1 distance arrays have invalid dtype or shape")
    if (
        len(offsets) != len(segment_wire) + 1
        or offsets[0] != 0
        or offsets[-1] != len(xyz)
        or np.any(np.diff(offsets) < 0)
    ):
        raise _error(I1ErrorCode.SOURCE, "W1 segment offsets are invalid")
    if (
        not np.isfinite(xyz).all()
        or not np.isin(statuses, [int(item) for item in WireModelStatus]).all()
    ):
        raise _error(I1ErrorCode.SOURCE, "W1 geometry or statuses are invalid")
    for name in ("gap_wire_id", "gap_left_segment_index", "gap_right_segment_index"):
        if arrays[name].dtype != np.int32 or len(arrays[name]) != len(arrays["gap_wire_id"]):
            raise _error(I1ErrorCode.SOURCE, "W1 gap arrays are invalid", array=name)
    for name in ("gap_start_xyz", "gap_end_xyz"):
        value = arrays[name]
        if (
            value.dtype != np.float32
            or value.shape != (len(arrays["gap_wire_id"]), 3)
            or not np.isfinite(value).all()
        ):
            raise _error(I1ErrorCode.SOURCE, "W1 gap geometry is invalid", array=name)


def _validate_v2_arrays(arrays: Mapping[str, np.ndarray], result: Any) -> None:
    required = {
        "xyz_local",
        "origin_xyz",
        "cluster_id_by_point",
        "vegetation_id_by_point",
        "cluster_status_by_point",
        "distance_ready_by_point",
        "cluster_id",
        "cluster_status",
        "cluster_vegetation_id",
        "cluster_member_offsets",
        "ordered_cluster_member_row_index",
    }
    if missing := required - set(arrays):
        raise _error(I1ErrorCode.SOURCE, "V2 arrays are incomplete", arrays=sorted(missing))
    xyz = arrays["xyz_local"]
    origin = arrays["origin_xyz"]
    if xyz.dtype != np.float32 or xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all():
        raise _error(I1ErrorCode.SOURCE, "V2 local geometry is invalid")
    if (
        origin.dtype != np.float64
        or origin.shape != (3,)
        or not np.isfinite(origin).all()
        or not np.array_equal(origin, np.asarray(result.origin_xyz, np.float64))
    ):
        raise _error(I1ErrorCode.SOURCE, "V2 origin differs from its result manifest")
    cluster_ids = arrays["cluster_id"]
    statuses = arrays["cluster_status"]
    vegetation_ids = arrays["cluster_vegetation_id"]
    offsets = arrays["cluster_member_offsets"]
    members = arrays["ordered_cluster_member_row_index"]
    if (
        cluster_ids.dtype != np.int32
        or statuses.dtype != np.uint8
        or vegetation_ids.dtype != np.int32
        or offsets.dtype != np.int64
        or members.dtype != np.int64
        or len(offsets) != len(cluster_ids) + 1
        or offsets[0] != 0
        or offsets[-1] != len(members)
        or np.any(np.diff(offsets) < 0)
    ):
        raise _error(I1ErrorCode.SOURCE, "V2 cluster arrays or offsets are invalid")
    if len(cluster_ids) and (
        np.any(cluster_ids < 0) or len(np.unique(cluster_ids)) != len(cluster_ids)
    ):
        raise _error(I1ErrorCode.SOURCE, "V2 cluster IDs are invalid")
    if (
        not np.isin(statuses, [int(item) for item in VegetationClusterStatus]).all()
        or np.any(members < 0)
        or np.any(members >= len(xyz))
    ):
        raise _error(I1ErrorCode.SOURCE, "V2 cluster statuses or member rows are invalid")
    point_fields = (
        "cluster_id_by_point",
        "vegetation_id_by_point",
        "cluster_status_by_point",
        "distance_ready_by_point",
    )
    if any(len(arrays[name]) != len(xyz) for name in point_fields):
        raise _error(I1ErrorCode.SOURCE, "V2 point arrays are not aligned")
    for position, cluster_id in enumerate(cluster_ids):
        start, end = offsets[position : position + 2]
        rows = members[start:end]
        if len(rows) and (
            not np.all(arrays["cluster_id_by_point"][rows] == cluster_id)
            or not np.all(arrays["cluster_status_by_point"][rows] == statuses[position])
            or not np.all(arrays["vegetation_id_by_point"][rows] == vegetation_ids[position])
        ):
            raise _error(
                I1ErrorCode.SOURCE,
                "V2 member rows disagree with cluster metadata",
                cluster_id=int(cluster_id),
            )


def _recover_coordinate_context(
    w1_result: Any, v2_result: Any, project_root: Path
) -> I1CoordinateContext:
    origin_json = _safe_join(project_root, w1_result.input_a4_json_relpath)
    if not origin_json.is_file() or _sha(origin_json) != w1_result.input_a4_json_sha256:
        raise _error(I1ErrorCode.COORDINATE, "W1 A4 origin provenance is absent or stale")
    try:
        raw = _load_json(origin_json)
        a4 = WireSeparationResult.from_dict(raw)
    except Exception as exc:
        raise _error(
            I1ErrorCode.COORDINATE, "W1 A4 origin provenance is invalid", cause=str(exc)
        ) from exc
    if a4.source_id != w1_result.source_id or a4.manifest_source_id != w1_result.manifest_source_id:
        raise _error(I1ErrorCode.COORDINATE, "W1 A4 origin source identity differs")
    w1_origin = np.asarray(a4.origin_xyz, dtype=np.float64)
    v2_origin = np.asarray(v2_result.origin_xyz, dtype=np.float64)
    if (
        w1_origin.shape != (3,)
        or v2_origin.shape != (3,)
        or not np.isfinite(w1_origin).all()
        or not np.isfinite(v2_origin).all()
        or not np.array_equal(w1_origin, v2_origin)
    ):
        raise _error(I1ErrorCode.COORDINATE, "W1 and V2 local origins do not exactly match")
    return I1CoordinateContext(
        origin_xyz=tuple(w1_origin.tolist()),
        frame=I1CoordinateFrame.LOCAL_CARTESIAN_XYZ,
        unit=I1CoordinateUnit.METRE,
        transform="identity",
        origin_source_kind=I1_ORIGIN_SOURCE_KIND,
        origin_source_schema_version=str(raw.get("schema_version", "wire-separation-result-v1")),
        origin_source_algorithm_version=a4.algorithm_version,
        origin_source_json_sha256=_sha(origin_json),
        verification_result=I1_ORIGIN_VERIFICATION,
    )


def _validate_source_identity(w1: Any, v2: Any) -> None:
    if w1.source_id != v2.source_id:
        raise _error(I1ErrorCode.SOURCE, "W1 and V2 source IDs differ")
    if w1.manifest_source_id != v2.manifest_source_id:
        raise _error(I1ErrorCode.SOURCE, "W1 and V2 manifest source IDs differ")


def _reference(
    kind: I1ArtifactKind, npz_path: Path, json_path: Path, result: Any
) -> I1ArtifactReference:
    return I1ArtifactReference(
        kind=kind,
        npz_relpath=f"artifacts/{kind.value}/{npz_path.name}",
        json_relpath=f"artifacts/{kind.value}/{json_path.name}",
        npz_sha256=_sha(npz_path),
        npz_bytes=npz_path.stat().st_size,
        json_sha256=_sha(json_path),
        json_bytes=json_path.stat().st_size,
        schema_version=result.schema_version,
        algorithm_version=result.algorithm_version,
    )


def _manifest(
    w1_result: Any,
    v2_result: Any,
    context: I1CoordinateContext,
    w1: I1ArtifactReference,
    v2: I1ArtifactReference,
    v2a: I1ArtifactReference | None,
) -> I1BundleManifest:
    state = I1OptionalArtifactState.PRESENT if v2a is not None else I1OptionalArtifactState.ABSENT
    bundle_id = calculate_bundle_id(
        source_id=w1_result.source_id,
        manifest_source_id=w1_result.manifest_source_id,
        coordinate_context=context,
        w1=w1,
        v2=v2,
        v2a_state=state,
        v2a=v2a,
    )
    return I1BundleManifest(
        schema_version=I1_SCHEMA_VERSION,
        interface_version=I1_INTERFACE_VERSION,
        bundle_id=bundle_id,
        source_id=w1_result.source_id,
        manifest_source_id=w1_result.manifest_source_id,
        eligibility_policy_version=I1_ELIGIBILITY_POLICY_VERSION,
        coordinate_context=context,
        w1=w1,
        v2=v2,
        v2a_state=state,
        v2a=v2a,
        warnings=(),
        limitations=(
            "V2A is optional audit/UI metadata and cannot affect formal C1 eligibility.",
            "I1 v1 supports only exact-match local Cartesian metre coordinates.",
        ),
    )


def _copy_verified(source: Path, target: Path, digest: str, size: int) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    _verify_file(target, digest, size)


def export_i1_bundle(
    w1_npz_path: Path | str,
    w1_json_path: Path | str,
    v2_npz_path: Path | str,
    v2_json_path: Path | str,
    bundle_parent: Path | str,
    *,
    project_root: Path | str | None = None,
    v2a_npz_path: Path | str | None = None,
    v2a_json_path: Path | str | None = None,
    cancellation_token: CancellationToken | None = None,
) -> ValidatedI1Bundle:
    """Copy validated sibling artifacts into an immutable portable I1 bundle."""

    if (v2a_npz_path is None) != (v2a_json_path is None):
        raise _error(I1ErrorCode.CONTRACT, "V2A NPZ and JSON must be supplied together")
    _check_cancelled(cancellation_token, "before source validation")
    w1_npz, w1_json = Path(w1_npz_path).resolve(), Path(w1_json_path).resolve()
    v2_npz, v2_json = Path(v2_npz_path).resolve(), Path(v2_json_path).resolve()
    w1_result, _ = _load_w1(w1_npz, w1_json)
    v2_result, _ = _load_v2(v2_npz, v2_json)
    _validate_source_identity(w1_result, v2_result)
    root = (Path(project_root) if project_root is not None else Path.cwd()).resolve()
    context = _recover_coordinate_context(w1_result, v2_result, root)
    v2a_result: Any | None = None
    v2a_paths: tuple[Path, Path] | None = None
    if v2a_npz_path is not None and v2a_json_path is not None:
        v2a_paths = (Path(v2a_npz_path).resolve(), Path(v2a_json_path).resolve())
        v2a_result, _ = _load_v2a(*v2a_paths)
        if (
            v2a_result.source_id != w1_result.source_id
            or v2a_result.manifest_source_id != w1_result.manifest_source_id
        ):
            raise _error(I1ErrorCode.SOURCE, "V2A source identity differs")
        provenance = v2a_result.input_provenance
        if provenance.get("v2_npz", {}).get("sha256") != _sha(v2_npz) or provenance.get(
            "v2_json", {}
        ).get("sha256") != _sha(v2_json):
            raise _error(I1ErrorCode.SOURCE, "V2A provenance does not identify supplied V2")
    w1_ref = _reference(I1ArtifactKind.W1, w1_npz, w1_json, w1_result)
    v2_ref = _reference(I1ArtifactKind.V2, v2_npz, v2_json, v2_result)
    v2a_ref = _reference(I1ArtifactKind.V2A, *v2a_paths, v2a_result) if v2a_paths else None
    manifest = _manifest(w1_result, v2_result, context, w1_ref, v2_ref, v2a_ref)
    parent = Path(bundle_parent).resolve()
    parent.mkdir(parents=True, exist_ok=True)
    final = parent / manifest.bundle_id
    if final.exists():
        try:
            existing = load_i1_bundle(final, expected_bundle_id=manifest.bundle_id)
        except I1Error as exc:
            raise _error(
                I1ErrorCode.CONFLICT,
                "existing bundle ID is incomplete or conflicts with this immutable export",
            ) from exc
        if existing.manifest.to_dict() != manifest.to_dict():
            raise _error(I1ErrorCode.CONFLICT, "existing bundle ID has conflicting manifest")
        return existing
    temporary = parent / f".{manifest.bundle_id}.tmp-{uuid.uuid4().hex}"
    try:
        temporary.mkdir()
        _check_cancelled(cancellation_token, "before artifact copy")
        for reference, source_npz, source_json in (
            (w1_ref, w1_npz, w1_json),
            (v2_ref, v2_npz, v2_json),
            *([(v2a_ref, *v2a_paths)] if v2a_ref is not None and v2a_paths is not None else []),
        ):
            _copy_verified(
                source_npz,
                _safe_join(temporary, reference.npz_relpath),
                reference.npz_sha256,
                reference.npz_bytes,
            )
            _check_cancelled(cancellation_token, "between artifact copies")
            _copy_verified(
                source_json,
                _safe_join(temporary, reference.json_relpath),
                reference.json_sha256,
                reference.json_bytes,
            )
        _check_cancelled(cancellation_token, "before manifest publication")
        manifest_path = temporary / I1_MANIFEST_FILENAME
        with manifest_path.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(
                json.dumps(manifest.to_dict(), ensure_ascii=False, sort_keys=True, indent=2)
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        I1BundleManifest.from_dict(_load_json(manifest_path))
        temporary.rename(final)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return load_i1_bundle(final, expected_bundle_id=manifest.bundle_id)


def _load_manifest(root: Path) -> I1BundleManifest:
    if ".tmp-" in root.name or root.name.startswith("."):
        raise _error(I1ErrorCode.INTEGRITY, "temporary/staging bundles are not loadable")
    manifest_path = root / I1_MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise _error(I1ErrorCode.INTEGRITY, "I1 completion manifest is missing")
    try:
        return I1BundleManifest.from_dict(_load_json(manifest_path))
    except (ValueError, KeyError) as exc:
        raise _error(I1ErrorCode.CONTRACT, "I1 manifest is invalid", cause=str(exc)) from exc


def _validate_reference(root: Path, reference: I1ArtifactReference) -> tuple[Path, Path]:
    npz_path = _safe_join(root, reference.npz_relpath)
    json_path = _safe_join(root, reference.json_relpath)
    _verify_file(npz_path, reference.npz_sha256, reference.npz_bytes)
    _verify_file(json_path, reference.json_sha256, reference.json_bytes)
    return npz_path, json_path


def load_i1_bundle(
    bundle_path: Path | str, expected_bundle_id: str | None = None
) -> ValidatedI1Bundle:
    """Reload a completed I1 bundle without any original-machine paths."""

    root = Path(bundle_path).resolve()
    if not root.is_dir():
        raise _error(I1ErrorCode.INTEGRITY, "I1 bundle directory is missing")
    manifest = _load_manifest(root)
    if expected_bundle_id is not None and expected_bundle_id != manifest.bundle_id:
        raise _error(I1ErrorCode.INTEGRITY, "expected bundle ID does not match")
    if root.name != manifest.bundle_id:
        raise _error(I1ErrorCode.INTEGRITY, "bundle directory name does not match manifest ID")
    w1_paths = _validate_reference(root, manifest.w1)
    v2_paths = _validate_reference(root, manifest.v2)
    w1_result, w1_arrays = _load_w1(*w1_paths)
    v2_result, v2_arrays = _load_v2(*v2_paths)
    _validate_source_identity(w1_result, v2_result)
    if (
        w1_result.source_id != manifest.source_id
        or w1_result.manifest_source_id != manifest.manifest_source_id
    ):
        raise _error(I1ErrorCode.SOURCE, "manifest source identity differs from bundled sources")
    if (
        manifest.w1.schema_version != w1_result.schema_version
        or manifest.w1.algorithm_version != w1_result.algorithm_version
    ):
        raise _error(I1ErrorCode.SOURCE, "W1 reference version differs from source")
    if (
        manifest.v2.schema_version != v2_result.schema_version
        or manifest.v2.algorithm_version != v2_result.algorithm_version
    ):
        raise _error(I1ErrorCode.SOURCE, "V2 reference version differs from source")
    context = manifest.coordinate_context
    if not np.array_equal(
        np.asarray(context.origin_xyz, np.float64), np.asarray(v2_result.origin_xyz, np.float64)
    ):
        raise _error(I1ErrorCode.COORDINATE, "I1 coordinate origin differs from V2")
    if context.origin_source_json_sha256 != w1_result.input_a4_json_sha256:
        raise _error(I1ErrorCode.COORDINATE, "I1 origin provenance differs from W1")
    v2a_result = None
    v2a_arrays = None
    if manifest.v2a_state is I1OptionalArtifactState.PRESENT:
        assert manifest.v2a is not None
        v2a_paths = _validate_reference(root, manifest.v2a)
        v2a_result, v2a_arrays = _load_v2a(*v2a_paths)
        if (
            v2a_result.source_id != manifest.source_id
            or v2a_result.manifest_source_id != manifest.manifest_source_id
        ):
            raise _error(I1ErrorCode.SOURCE, "V2A source identity differs from I1")
        provenance = v2a_result.input_provenance
        if (
            provenance.get("v2_npz", {}).get("sha256") != manifest.v2.npz_sha256
            or provenance.get("v2_json", {}).get("sha256") != manifest.v2.json_sha256
        ):
            raise _error(I1ErrorCode.SOURCE, "V2A provenance does not identify bundled V2")
    return ValidatedI1Bundle(
        root, manifest, w1_result, w1_arrays, v2_result, v2_arrays, v2a_result, v2a_arrays
    )


def eligible_wire_segment_keys(bundle: ValidatedI1Bundle) -> tuple[tuple[int, int], ...]:
    """Return ordered, ready W1 segment keys without creating prediction records."""

    arrays = bundle.w1_arrays
    statuses = {
        int(identifier): int(status)
        for identifier, status in zip(arrays["wire_id"], arrays["wire_status"], strict=True)
    }
    result = []
    for position, (wire_id, index, ready) in enumerate(
        zip(
            arrays["segment_wire_id"],
            arrays["segment_index"],
            arrays["segment_distance_ready"],
            strict=True,
        )
    ):
        start, end = arrays["sampled_curve_offsets"][position : position + 2]
        xyz = arrays["sampled_curve_xyz"][start:end]
        if (
            statuses.get(int(wire_id)) == int(WireModelStatus.CONFIRMED_MODEL)
            and bool(ready)
            and len(xyz) >= 2
            and np.isfinite(xyz).all()
        ):
            result.append((int(wire_id), int(index)))
    return tuple(result)


def eligible_vegetation_cluster_ids(bundle: ValidatedI1Bundle) -> tuple[int, ...]:
    """Return ordered confirmed V2 cluster IDs satisfying the frozen I1 policy."""

    arrays = bundle.v2_arrays
    result = []
    for position, cluster_id in enumerate(arrays["cluster_id"]):
        start, end = arrays["cluster_member_offsets"][position : position + 2]
        rows = arrays["ordered_cluster_member_row_index"][start:end]
        status = int(arrays["cluster_status"][position])
        vegetation_id = int(arrays["cluster_vegetation_id"][position])
        if status != int(VegetationClusterStatus.CONFIRMED) or vegetation_id < 0 or not len(rows):
            continue
        if (
            np.all(arrays["cluster_id_by_point"][rows] == cluster_id)
            and np.all(arrays["vegetation_id_by_point"][rows] == vegetation_id)
            and np.all(arrays["distance_ready_by_point"][rows])
            and np.isfinite(arrays["xyz_local"][rows]).all()
        ):
            result.append(int(cluster_id))
    return tuple(result)
