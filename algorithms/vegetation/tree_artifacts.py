"""Strict, read-only loading for persisted V3 tree-segmentation artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np
from core.contracts._json import json_safe_mapping
from core.contracts.tree_artifacts import (
    TREE_SEGMENTATION_ARTIFACT_SCHEMA_VERSION,
    TREE_SEGMENTATION_MANIFEST_FILENAME,
    TREE_SEGMENTATION_METADATA_FILENAME,
    TREE_SEGMENTATION_NPZ_FILENAME,
    TreeSegmentationArtifactManifest,
    TreeSegmentationArtifactMetadata,
)
from core.contracts.tree_inputs import TreeSegmentationInputIdentity
from core.contracts.vegetation_trees import (
    TREE_SEGMENTATION_ALGORITHM_VERSION,
    TREE_SEGMENTATION_SCHEMA_VERSION,
    TreeSegmentationErrorCode,
    TreeSegmentationResult,
)
from numpy.lib import format as npy_format

_ARRAY_NAMES = (
    "segment_id_by_point",
    "tree_id_by_point",
    "tree_distance_ready_by_point",
    "tree_member_offsets",
    "ordered_tree_member_row_index",
)
_EXPECTED_FILENAMES = {
    TREE_SEGMENTATION_NPZ_FILENAME,
    TREE_SEGMENTATION_METADATA_FILENAME,
    TREE_SEGMENTATION_MANIFEST_FILENAME,
}
_DISK_DTYPES = {
    "segment_id_by_point": np.dtype("<i4"),
    "tree_id_by_point": np.dtype("<i4"),
    "tree_distance_ready_by_point": np.dtype("|b1"),
    "tree_member_offsets": np.dtype("<i8"),
    "ordered_tree_member_row_index": np.dtype("<i8"),
}
_RUNTIME_DTYPES = {
    "segment_id_by_point": np.dtype(np.int32),
    "tree_id_by_point": np.dtype(np.int32),
    "tree_distance_ready_by_point": np.dtype(np.bool_),
    "tree_member_offsets": np.dtype(np.int64),
    "ordered_tree_member_row_index": np.dtype(np.int64),
}
_HASH_CHUNK_BYTES = 1024 * 1024
_MAX_MANIFEST_BYTES = 64 * 1024
_MAX_NPY_HEADER_BYTES = 10 * 1024
_MAX_MESSAGE_TEXT = 160
_MAX_MESSAGE_ITEMS = 8


def _bounded_repr(value: Any) -> str:
    if isinstance(value, str):
        text = value
    elif isinstance(value, Path):
        text = str(value)
    elif isinstance(value, Mapping):
        items = list(value.items())
        suffix = ", ..." if len(items) > _MAX_MESSAGE_ITEMS else ""
        text = (
            "{"
            + ", ".join(
                f"{_bounded_repr(key)}: {_bounded_repr(item)}"
                for key, item in items[:_MAX_MESSAGE_ITEMS]
            )
            + suffix
            + "}"
        )
    elif isinstance(value, list | tuple | set | frozenset):
        items = list(value)
        suffix = ", ..." if len(items) > _MAX_MESSAGE_ITEMS else ""
        body = ", ".join(_bounded_repr(item) for item in items[:_MAX_MESSAGE_ITEMS])
        text = "[" + body + suffix + "]"
    elif isinstance(value, np.ndarray):
        text = f"ndarray(dtype={value.dtype.str!r}, shape={value.shape!r})"
    else:
        text = repr(value)
    if len(text) > _MAX_MESSAGE_TEXT:
        return f"{text[: _MAX_MESSAGE_TEXT - 3]}..."
    return text


class TreeSegmentationArtifactError(RuntimeError):
    """A stable, compact failure from strict V3 artifact loading."""

    def __init__(
        self,
        code: TreeSegmentationErrorCode,
        artifact_role: str,
        field: str,
        expected: Any,
        actual: Any,
        *,
        bundle_path: Path | None = None,
    ) -> None:
        if not isinstance(code, TreeSegmentationErrorCode):
            raise ValueError("code must be a TreeSegmentationErrorCode")
        if not isinstance(artifact_role, str) or not artifact_role:
            raise ValueError("artifact_role must be a non-empty string")
        if not isinstance(field, str) or not field:
            raise ValueError("field must be a non-empty string")
        if bundle_path is not None and (
            not isinstance(bundle_path, Path) or not bundle_path.is_absolute()
        ):
            raise ValueError("bundle_path must be an absolute Path or None")
        self.code = code
        self.artifact_role = artifact_role
        self.field = field
        self.expected = expected
        self.actual = actual
        self.bundle_path = bundle_path
        message = (
            f"{artifact_role}.{field} is invalid: expected {_bounded_repr(expected)}, "
            f"got {_bounded_repr(actual)}"
        )
        if bundle_path is not None:
            message = f"{message} (bundle_path={str(bundle_path)!r})"
        super().__init__(message)


def _error(
    code: TreeSegmentationErrorCode,
    artifact_role: str,
    field: str,
    expected: Any,
    actual: Any,
    *,
    bundle_path: Path | None,
) -> TreeSegmentationArtifactError:
    return TreeSegmentationArtifactError(
        code,
        artifact_role,
        field,
        expected,
        actual,
        bundle_path=bundle_path,
    )


def _result_arrays(result: TreeSegmentationResult) -> tuple[np.ndarray, ...]:
    return tuple(getattr(result, name) for name in _ARRAY_NAMES)


@dataclass(frozen=True)
class LoadedTreeSegmentationArtifact:
    """A complete validated V3 result detached from its artifact files."""

    bundle_path: Path
    result: TreeSegmentationResult
    source_identity: TreeSegmentationInputIdentity

    def __post_init__(self) -> None:
        if not isinstance(self.bundle_path, Path):
            raise ValueError("bundle_path must be a pathlib.Path")
        if not self.bundle_path.is_absolute():
            raise ValueError("bundle_path must be absolute")
        if not isinstance(self.result, TreeSegmentationResult):
            raise ValueError("result must be a TreeSegmentationResult")
        if not isinstance(self.source_identity, TreeSegmentationInputIdentity):
            raise ValueError("source_identity must be a TreeSegmentationInputIdentity")
        if self.result.source_id != self.source_identity.source_id:
            raise ValueError("result.source_id must match source_identity.source_id")
        if self.result.manifest_source_id != self.source_identity.manifest_source_id:
            raise ValueError(
                "result.manifest_source_id must match source_identity.manifest_source_id"
            )
        if self.result.origin_xyz != self.source_identity.origin_xyz:
            raise ValueError("result.origin_xyz must match source_identity.origin_xyz")
        if self.result.statistics.input_point_count != self.source_identity.point_count:
            raise ValueError(
                "result.statistics.input_point_count must match source_identity.point_count"
            )
        for name, value in zip(_ARRAY_NAMES, _result_arrays(self.result), strict=True):
            if not isinstance(value, np.ndarray):
                raise ValueError(f"result.{name} must be a numpy ndarray")
            if value.flags.writeable:
                raise ValueError(f"result.{name} must be read-only")
            if not value.flags.c_contiguous:
                raise ValueError(f"result.{name} must be C-contiguous")
            if isinstance(value, np.memmap) or not value.flags.owndata:
                raise ValueError(f"result.{name} must own detached in-memory data")


def _is_reparse_point(value: os.stat_result) -> bool:
    attributes = getattr(value, "st_file_attributes", 0)
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return bool(reparse_flag and attributes & reparse_flag)


def _adapt_bundle_path(value: str | Path) -> Path:
    if not isinstance(value, str | Path):
        raise _error(
            TreeSegmentationErrorCode.CORRUPT_SIBLING_ARTIFACT,
            "bundle",
            "path_type",
            "str or Path",
            type(value).__name__,
            bundle_path=None,
        )
    try:
        path = Path(value)
    except (TypeError, ValueError, OSError) as exc:
        raise _error(
            TreeSegmentationErrorCode.CORRUPT_SIBLING_ARTIFACT,
            "bundle",
            "path_type",
            "str or Path",
            type(value).__name__,
            bundle_path=None,
        ) from exc
    if not path.is_absolute():
        raise _error(
            TreeSegmentationErrorCode.CORRUPT_SIBLING_ARTIFACT,
            "bundle",
            "path",
            "absolute bundle directory",
            str(path),
            bundle_path=None,
        )
    return path


def _validate_bundle_root(path: Path) -> None:
    try:
        root_stat = path.lstat()
    except (FileNotFoundError, OSError) as exc:
        raise _error(
            TreeSegmentationErrorCode.CORRUPT_SIBLING_ARTIFACT,
            "bundle",
            "path",
            "existing non-linked directory",
            type(exc).__name__,
            bundle_path=path,
        ) from exc
    if stat.S_ISLNK(root_stat.st_mode) or _is_reparse_point(root_stat):
        raise _error(
            TreeSegmentationErrorCode.CORRUPT_SIBLING_ARTIFACT,
            "bundle",
            "path",
            "non-linked directory",
            "symlink or reparse point",
            bundle_path=path,
        )
    if not stat.S_ISDIR(root_stat.st_mode):
        raise _error(
            TreeSegmentationErrorCode.CORRUPT_SIBLING_ARTIFACT,
            "bundle",
            "path",
            "directory",
            "non-directory filesystem entry",
            bundle_path=path,
        )


def _validate_directory_inventory(path: Path) -> dict[str, Path]:
    try:
        with os.scandir(path) as stream:
            entries = list(stream)
    except (OSError, MemoryError) as exc:
        raise _error(
            TreeSegmentationErrorCode.CORRUPT_SIBLING_ARTIFACT,
            "bundle",
            "inventory",
            sorted(_EXPECTED_FILENAMES),
            type(exc).__name__,
            bundle_path=path,
        ) from exc
    names = [entry.name for entry in entries]
    if len({name.casefold() for name in names}) != len(names):
        raise _error(
            TreeSegmentationErrorCode.CORRUPT_SIBLING_ARTIFACT,
            "bundle",
            "inventory_case",
            "case-distinct exact filenames",
            sorted(names),
            bundle_path=path,
        )
    if set(names) != _EXPECTED_FILENAMES or len(names) != len(_EXPECTED_FILENAMES):
        raise _error(
            TreeSegmentationErrorCode.CORRUPT_SIBLING_ARTIFACT,
            "bundle",
            "inventory",
            sorted(_EXPECTED_FILENAMES),
            sorted(names),
            bundle_path=path,
        )
    by_name = {entry.name: entry for entry in entries}
    for name in sorted(_EXPECTED_FILENAMES):
        entry = by_name[name]
        try:
            entry_stat = entry.stat(follow_symlinks=False)
        except OSError as exc:
            raise _error(
                TreeSegmentationErrorCode.CORRUPT_SIBLING_ARTIFACT,
                name,
                "file_type",
                "regular non-linked file",
                type(exc).__name__,
                bundle_path=path,
            ) from exc
        if (
            entry.is_symlink()
            or _is_reparse_point(entry_stat)
            or not stat.S_ISREG(entry_stat.st_mode)
        ):
            raise _error(
                TreeSegmentationErrorCode.CORRUPT_SIBLING_ARTIFACT,
                name,
                "file_type",
                "regular non-linked file",
                "linked, reparse, or non-regular entry",
                bundle_path=path,
            )
    return {name: path / name for name in _EXPECTED_FILENAMES}


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _parse_strict_json(data: bytes) -> dict[str, Any]:
    if data.startswith(b"\xef\xbb\xbf"):
        raise ValueError("UTF-8 BOM is forbidden")
    text = data.decode("utf-8")
    value = json.loads(
        text,
        object_pairs_hook=_unique_json_object,
        parse_constant=_reject_json_constant,
    )
    if not isinstance(value, dict):
        raise ValueError("JSON root must be an object")
    return value


def _read_manifest(path: Path, bundle_path: Path) -> TreeSegmentationArtifactManifest:
    try:
        size = path.stat().st_size
        if size > _MAX_MANIFEST_BYTES:
            raise _error(
                TreeSegmentationErrorCode.CORRUPT_SIBLING_ARTIFACT,
                "manifest",
                "bytes",
                f"at most {_MAX_MANIFEST_BYTES}",
                size,
                bundle_path=bundle_path,
            )
        with path.open("rb") as stream:
            data = stream.read(_MAX_MANIFEST_BYTES + 1)
    except TreeSegmentationArtifactError:
        raise
    except (OSError, MemoryError) as exc:
        raise _error(
            TreeSegmentationErrorCode.CORRUPT_SIBLING_ARTIFACT,
            "manifest",
            "read",
            "readable canonical JSON",
            type(exc).__name__,
            bundle_path=bundle_path,
        ) from exc
    if len(data) != size or len(data) > _MAX_MANIFEST_BYTES:
        raise _error(
            TreeSegmentationErrorCode.CORRUPT_SIBLING_ARTIFACT,
            "manifest",
            "bytes",
            size,
            len(data),
            bundle_path=bundle_path,
        )
    try:
        raw = _parse_strict_json(data)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError, MemoryError) as exc:
        raise _error(
            TreeSegmentationErrorCode.CORRUPT_SIBLING_ARTIFACT,
            "manifest",
            "json",
            "strict canonical JSON object",
            type(exc).__name__,
            bundle_path=bundle_path,
        ) from exc
    if (
        "artifact_schema_version" in raw
        and raw["artifact_schema_version"] != TREE_SEGMENTATION_ARTIFACT_SCHEMA_VERSION
    ):
        raise _error(
            TreeSegmentationErrorCode.SCHEMA_MISMATCH,
            "manifest",
            "artifact_schema_version",
            TREE_SEGMENTATION_ARTIFACT_SCHEMA_VERSION,
            raw.get("artifact_schema_version"),
            bundle_path=bundle_path,
        )
    try:
        manifest = TreeSegmentationArtifactManifest.from_dict(raw)
    except (TypeError, ValueError) as exc:
        raise _error(
            TreeSegmentationErrorCode.CORRUPT_SIBLING_ARTIFACT,
            "manifest",
            "contract",
            "valid TreeSegmentationArtifactManifest",
            type(exc).__name__,
            bundle_path=bundle_path,
        ) from exc
    try:
        canonical = _canonical_json_bytes(manifest.to_dict())
    except (TypeError, ValueError, MemoryError) as exc:
        raise _error(
            TreeSegmentationErrorCode.CORRUPT_SIBLING_ARTIFACT,
            "manifest",
            "canonical_json",
            "canonical JSON bytes",
            type(exc).__name__,
            bundle_path=bundle_path,
        ) from exc
    if data != canonical:
        raise _error(
            TreeSegmentationErrorCode.CORRUPT_SIBLING_ARTIFACT,
            "manifest",
            "canonical_json",
            "exact canonical JSON bytes ending in one LF",
            "noncanonical bytes",
            bundle_path=bundle_path,
        )
    if manifest.metadata_bytes <= 0:
        raise _error(
            TreeSegmentationErrorCode.CORRUPT_SIBLING_ARTIFACT,
            "manifest",
            "metadata_bytes",
            "positive integer",
            manifest.metadata_bytes,
            bundle_path=bundle_path,
        )
    if manifest.npz_bytes <= 0:
        raise _error(
            TreeSegmentationErrorCode.CORRUPT_SIBLING_ARTIFACT,
            "manifest",
            "npz_bytes",
            "positive integer",
            manifest.npz_bytes,
            bundle_path=bundle_path,
        )
    return manifest


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(_HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_file(
    path: Path,
    *,
    role: str,
    expected_size: int,
    expected_sha256: str,
    bundle_path: Path,
) -> None:
    try:
        actual_size = path.stat().st_size
    except OSError as exc:
        raise _error(
            TreeSegmentationErrorCode.CORRUPT_SIBLING_ARTIFACT,
            role,
            "bytes",
            expected_size,
            type(exc).__name__,
            bundle_path=bundle_path,
        ) from exc
    if actual_size != expected_size:
        raise _error(
            TreeSegmentationErrorCode.CORRUPT_SIBLING_ARTIFACT,
            role,
            "bytes",
            expected_size,
            actual_size,
            bundle_path=bundle_path,
        )
    try:
        actual_sha256 = _sha256_file(path)
    except OSError as exc:
        raise _error(
            TreeSegmentationErrorCode.CORRUPT_SIBLING_ARTIFACT,
            role,
            "sha256",
            expected_sha256,
            type(exc).__name__,
            bundle_path=bundle_path,
        ) from exc
    if actual_sha256 != expected_sha256:
        raise _error(
            TreeSegmentationErrorCode.CORRUPT_SIBLING_ARTIFACT,
            role,
            "sha256",
            expected_sha256,
            actual_sha256,
            bundle_path=bundle_path,
        )


def _read_metadata(
    path: Path,
    manifest: TreeSegmentationArtifactManifest,
    bundle_path: Path,
) -> TreeSegmentationArtifactMetadata:
    _verify_file(
        path,
        role="metadata",
        expected_size=manifest.metadata_bytes,
        expected_sha256=manifest.metadata_sha256,
        bundle_path=bundle_path,
    )
    try:
        data = path.read_bytes()
    except (OSError, MemoryError) as exc:
        raise _error(
            TreeSegmentationErrorCode.CORRUPT_SIBLING_ARTIFACT,
            "metadata",
            "read",
            manifest.metadata_bytes,
            type(exc).__name__,
            bundle_path=bundle_path,
        ) from exc
    if len(data) != manifest.metadata_bytes or hashlib.sha256(data).hexdigest() != (
        manifest.metadata_sha256
    ):
        raise _error(
            TreeSegmentationErrorCode.CORRUPT_SIBLING_ARTIFACT,
            "metadata",
            "stable_content",
            "bytes matching validated size and SHA-256",
            "file changed during loading",
            bundle_path=bundle_path,
        )
    try:
        raw = _parse_strict_json(data)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError, MemoryError) as exc:
        raise _error(
            TreeSegmentationErrorCode.CORRUPT_SIBLING_ARTIFACT,
            "metadata",
            "json",
            "strict canonical JSON object",
            type(exc).__name__,
            bundle_path=bundle_path,
        ) from exc
    for field, expected in (
        ("artifact_schema_version", TREE_SEGMENTATION_ARTIFACT_SCHEMA_VERSION),
        ("result_schema_version", TREE_SEGMENTATION_SCHEMA_VERSION),
    ):
        if field in raw and raw[field] != expected:
            raise _error(
                TreeSegmentationErrorCode.SCHEMA_MISMATCH,
                "metadata",
                field,
                expected,
                raw.get(field),
                bundle_path=bundle_path,
            )
    if (
        "algorithm_version" in raw
        and raw["algorithm_version"] != TREE_SEGMENTATION_ALGORITHM_VERSION
    ):
        raise _error(
            TreeSegmentationErrorCode.ALGORITHM_VERSION_MISMATCH,
            "metadata",
            "algorithm_version",
            TREE_SEGMENTATION_ALGORITHM_VERSION,
            raw.get("algorithm_version"),
            bundle_path=bundle_path,
        )
    try:
        metadata = TreeSegmentationArtifactMetadata.from_dict(raw)
    except (TypeError, ValueError, MemoryError) as exc:
        raise _error(
            TreeSegmentationErrorCode.CORRUPT_SIBLING_ARTIFACT,
            "metadata",
            "contract",
            "valid TreeSegmentationArtifactMetadata",
            type(exc).__name__,
            bundle_path=bundle_path,
        ) from exc
    try:
        canonical = _canonical_json_bytes(metadata.to_dict())
    except (TypeError, ValueError, MemoryError) as exc:
        raise _error(
            TreeSegmentationErrorCode.CORRUPT_SIBLING_ARTIFACT,
            "metadata",
            "canonical_json",
            "canonical JSON bytes",
            type(exc).__name__,
            bundle_path=bundle_path,
        ) from exc
    if data != canonical:
        raise _error(
            TreeSegmentationErrorCode.CORRUPT_SIBLING_ARTIFACT,
            "metadata",
            "canonical_json",
            "exact canonical JSON bytes ending in one LF",
            "noncanonical bytes",
            bundle_path=bundle_path,
        )
    return metadata


class _CountingReader:
    def __init__(self, stream: BinaryIO) -> None:
        self._stream = stream
        self.bytes_read = 0

    def read(self, size: int = -1) -> bytes:
        value = self._stream.read(size)
        self.bytes_read += len(value)
        return value


def _expected_shapes(
    metadata: TreeSegmentationArtifactMetadata,
) -> dict[str, tuple[int, ...]]:
    point_count = metadata.source_identity.point_count
    record_count = len(metadata.records)
    member_count = metadata.arrays.ordered_tree_member_row_index.shape[0]
    if member_count > point_count:
        raise ValueError("member count must not exceed point count")
    return {
        "segment_id_by_point": (point_count,),
        "tree_id_by_point": (point_count,),
        "tree_distance_ready_by_point": (point_count,),
        "tree_member_offsets": (record_count + 1,),
        "ordered_tree_member_row_index": (member_count,),
    }


def _validate_npy_header(
    stream: BinaryIO,
    info: zipfile.ZipInfo,
    *,
    array_name: str,
    expected_dtype: np.dtype[Any],
    expected_shape: tuple[int, ...],
) -> None:
    reader = _CountingReader(stream)
    version = npy_format.read_magic(reader)
    if version == (1, 0):
        shape, fortran_order, dtype = npy_format.read_array_header_1_0(
            reader, max_header_size=_MAX_NPY_HEADER_BYTES
        )
    elif version == (2, 0):
        shape, fortran_order, dtype = npy_format.read_array_header_2_0(
            reader, max_header_size=_MAX_NPY_HEADER_BYTES
        )
    else:
        raise ValueError(f"unsupported NPY header version {version!r}")
    dtype = np.dtype(dtype)
    if (
        dtype != expected_dtype
        or dtype.str != expected_dtype.str
        or dtype.hasobject
        or dtype.fields is not None
        or dtype.subdtype is not None
        or dtype.kind in {"O", "S", "U", "V"}
    ):
        raise ValueError(f"{array_name} dtype must equal {expected_dtype.str!r}, got {dtype.str!r}")
    if not isinstance(shape, tuple) or len(shape) != 1 or shape != expected_shape:
        raise ValueError(f"{array_name} shape must equal {expected_shape!r}, got {shape!r}")
    if any(not isinstance(value, int) or value < 0 for value in shape):
        raise ValueError(f"{array_name} shape contains an invalid dimension")
    if fortran_order is not False:
        raise ValueError(f"{array_name} must use C order")
    expected_data_bytes = int(dtype.itemsize)
    for dimension in shape:
        expected_data_bytes *= int(dimension)
    expected_file_size = reader.bytes_read + expected_data_bytes
    if info.file_size != expected_file_size:
        raise ValueError(
            f"{array_name} uncompressed size must equal {expected_file_size}, got {info.file_size}"
        )


def _preflight_npz(
    path: Path,
    metadata: TreeSegmentationArtifactMetadata,
    bundle_path: Path,
) -> dict[str, tuple[int, ...]]:
    expected_shapes = _expected_shapes(metadata)
    expected_members = {f"{name}.npy" for name in _ARRAY_NAMES}
    try:
        with zipfile.ZipFile(path, "r") as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            if len(names) != len(set(names)):
                raise ValueError("duplicate ZIP member names are forbidden")
            if set(names) != expected_members or len(names) != len(expected_members):
                raise ValueError(
                    f"ZIP inventory differs: expected {sorted(expected_members)!r}, "
                    f"got {sorted(names)[:_MAX_MESSAGE_ITEMS]!r}"
                )
            for info in infos:
                name = info.filename
                if (
                    info.is_dir()
                    or "/" in name
                    or "\\" in name
                    or name in {".", ".."}
                    or not name.endswith(".npy")
                ):
                    raise ValueError(f"unsafe ZIP member name {name!r}")
                if info.flag_bits & 0x1:
                    raise ValueError(f"encrypted ZIP member is forbidden: {name!r}")
                if info.compress_type != zipfile.ZIP_DEFLATED:
                    raise ValueError(f"unsupported ZIP compression for {name!r}")
                array_name = name[:-4]
                with archive.open(info, "r") as stream:
                    _validate_npy_header(
                        stream,
                        info,
                        array_name=array_name,
                        expected_dtype=_DISK_DTYPES[array_name],
                        expected_shape=expected_shapes[array_name],
                    )
    except (OSError, EOFError, ValueError, zipfile.BadZipFile, MemoryError) as exc:
        raise _error(
            TreeSegmentationErrorCode.CORRUPT_SIBLING_ARTIFACT,
            "npz",
            "archive",
            "exact safe five-member NPZ",
            type(exc).__name__,
            bundle_path=bundle_path,
        ) from exc
    return expected_shapes


def _load_npz_arrays(
    path: Path,
    expected_shapes: Mapping[str, tuple[int, ...]],
    bundle_path: Path,
) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    try:
        with np.load(path, allow_pickle=False) as archive:
            if len(archive.files) != len(_ARRAY_NAMES) or set(archive.files) != set(_ARRAY_NAMES):
                raise ValueError("NpzFile inventory differs from the preflight inventory")
            for name in _ARRAY_NAMES:
                raw = archive[name]
                expected_dtype = _DISK_DTYPES[name]
                if (
                    type(raw) is not np.ndarray
                    or raw.dtype != expected_dtype
                    or raw.dtype.str != expected_dtype.str
                    or raw.ndim != 1
                    or raw.shape != expected_shapes[name]
                    or raw.dtype.hasobject
                    or raw.dtype.fields is not None
                    or raw.dtype.subdtype is not None
                    or raw.dtype.kind in {"O", "S", "U", "V"}
                ):
                    raise ValueError(f"{name} differs from its validated NPY header")
                value = np.array(
                    raw,
                    dtype=_RUNTIME_DTYPES[name],
                    order="C",
                    copy=True,
                    subok=False,
                )
                value.setflags(write=False)
                arrays[name] = value
                del raw
    except (OSError, EOFError, ValueError, zipfile.BadZipFile, MemoryError) as exc:
        raise _error(
            TreeSegmentationErrorCode.CORRUPT_SIBLING_ARTIFACT,
            "npz",
            "load",
            "five detached validated arrays",
            type(exc).__name__,
            bundle_path=bundle_path,
        ) from exc
    return arrays


def _reconstruct_result(
    metadata: TreeSegmentationArtifactMetadata,
    arrays: Mapping[str, np.ndarray],
    bundle_path: Path,
) -> TreeSegmentationResult:
    try:
        result = TreeSegmentationResult(
            schema_version=metadata.result_schema_version,
            algorithm_version=metadata.algorithm_version,
            source_id=metadata.source_identity.source_id,
            manifest_source_id=metadata.source_identity.manifest_source_id,
            origin_xyz=metadata.source_identity.origin_xyz,
            config=metadata.config,
            statistics=metadata.statistics,
            records=metadata.records,
            segment_id_by_point=arrays["segment_id_by_point"],
            tree_id_by_point=arrays["tree_id_by_point"],
            tree_distance_ready_by_point=arrays["tree_distance_ready_by_point"],
            tree_member_offsets=arrays["tree_member_offsets"],
            ordered_tree_member_row_index=arrays["ordered_tree_member_row_index"],
            input_provenance=metadata.input_provenance,
            coordinate_context=metadata.coordinate_context,
            warnings=metadata.warnings,
            limitations=metadata.limitations,
        )
    except (TypeError, ValueError, MemoryError) as exc:
        raise _error(
            TreeSegmentationErrorCode.CORRUPT_SIBLING_ARTIFACT,
            "result",
            "contract",
            "valid TreeSegmentationResult",
            type(exc).__name__,
            bundle_path=bundle_path,
        ) from exc
    if result.origin_xyz != metadata.source_identity.origin_xyz:
        raise _error(
            TreeSegmentationErrorCode.COORDINATE_MISMATCH,
            "result",
            "origin_xyz",
            metadata.source_identity.origin_xyz,
            result.origin_xyz,
            bundle_path=bundle_path,
        )
    structured_matches = (
        result.schema_version == metadata.result_schema_version
        and result.algorithm_version == metadata.algorithm_version
        and result.source_id == metadata.source_identity.source_id
        and result.manifest_source_id == metadata.source_identity.manifest_source_id
        and result.statistics == metadata.statistics
        and result.config == metadata.config
        and result.records == metadata.records
        and json_safe_mapping(result.input_provenance, "input_provenance")
        == json_safe_mapping(metadata.input_provenance, "input_provenance")
        and json_safe_mapping(result.coordinate_context, "coordinate_context")
        == json_safe_mapping(metadata.coordinate_context, "coordinate_context")
        and result.warnings == metadata.warnings
        and result.limitations == metadata.limitations
    )
    arrays_match = all(getattr(result, name) is arrays[name] for name in _ARRAY_NAMES)
    if not structured_matches or not arrays_match:
        raise _error(
            TreeSegmentationErrorCode.CORRUPT_SIBLING_ARTIFACT,
            "result",
            "reconstruction",
            "exact metadata and array preservation",
            "post-construction disagreement",
            bundle_path=bundle_path,
        )
    for name in _ARRAY_NAMES:
        value = getattr(result, name)
        if value.flags.writeable or not value.flags.c_contiguous or not value.flags.owndata:
            raise _error(
                TreeSegmentationErrorCode.CORRUPT_SIBLING_ARTIFACT,
                "result",
                name,
                "owning read-only C-contiguous array",
                {
                    "writeable": value.flags.writeable,
                    "c_contiguous": value.flags.c_contiguous,
                    "owndata": value.flags.owndata,
                },
                bundle_path=bundle_path,
            )
    return result


def load_tree_segmentation_artifact(
    bundle_path: str | Path,
) -> LoadedTreeSegmentationArtifact:
    """Load one complete, canonical V3 tree-segmentation artifact directory."""

    path = _adapt_bundle_path(bundle_path)
    _validate_bundle_root(path)
    files = _validate_directory_inventory(path)
    manifest = _read_manifest(files[TREE_SEGMENTATION_MANIFEST_FILENAME], path)
    metadata = _read_metadata(
        files[TREE_SEGMENTATION_METADATA_FILENAME],
        manifest,
        path,
    )
    npz_path = files[TREE_SEGMENTATION_NPZ_FILENAME]
    _verify_file(
        npz_path,
        role="npz",
        expected_size=manifest.npz_bytes,
        expected_sha256=manifest.npz_sha256,
        bundle_path=path,
    )
    expected_shapes = _preflight_npz(npz_path, metadata, path)
    arrays = _load_npz_arrays(npz_path, expected_shapes, path)
    _verify_file(
        npz_path,
        role="npz",
        expected_size=manifest.npz_bytes,
        expected_sha256=manifest.npz_sha256,
        bundle_path=path,
    )
    result = _reconstruct_result(metadata, arrays, path)
    try:
        return LoadedTreeSegmentationArtifact(path, result, metadata.source_identity)
    except ValueError as exc:
        raise _error(
            TreeSegmentationErrorCode.CORRUPT_SIBLING_ARTIFACT,
            "result",
            "loaded_artifact",
            "detached immutable runtime artifact",
            type(exc).__name__,
            bundle_path=path,
        ) from exc
