"""Portable I1 cross-computer bundle contracts.

I1 deliberately records references to verified W1/V2(/V2A) sibling artifacts
instead of repacking their point arrays.  The manifest is the completion marker
for an immutable directory bundle.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import PurePosixPath
from typing import Any

from core.contracts._json import reject_unknown_fields, require_fields

I1_SCHEMA_VERSION = "i1-bundle-v1"
I1_INTERFACE_VERSION = "1.0.0"
I1_ELIGIBILITY_POLICY_VERSION = "i1-eligibility-v1"
I1_MANIFEST_FILENAME = "i1_manifest.json"
I1_ORIGIN_VERIFICATION = "verified_exact_match"
I1_ORIGIN_SOURCE_KIND = "w1_a4_json"


class I1ArtifactKind(str, Enum):
    W1 = "w1"
    V2 = "v2"
    V2A = "v2a"


class I1CoordinateFrame(str, Enum):
    LOCAL_CARTESIAN_XYZ = "LOCAL_CARTESIAN_XYZ"


class I1CoordinateUnit(str, Enum):
    METRE = "m"


class I1OptionalArtifactState(str, Enum):
    ABSENT = "absent"
    PRESENT = "present"


class I1ErrorCode(str, Enum):
    CANCELLED = "I1_CANCELLED"
    CONTRACT = "I1_CONTRACT"
    SOURCE = "I1_SOURCE"
    COORDINATE = "I1_COORDINATE"
    INTEGRITY = "I1_INTEGRITY"
    PUBLICATION = "I1_PUBLICATION"
    CONFLICT = "I1_CONFLICT"


class I1Error(RuntimeError):
    """Typed public I1 failure with a stable error code."""

    def __init__(self, code: I1ErrorCode, message: str, **details: Any) -> None:
        super().__init__(message)
        self.code = code
        self.details = dict(details)


def _sha(value: str, name: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(ch not in "0123456789abcdef" for ch in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _safe_relpath(value: str, name: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"{name} must be a normalized relative POSIX path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
        or str(path) != value
    ):
        raise ValueError(f"{name} must be a normalized traversal-safe relative POSIX path")
    return value


def canonical_json(value: Mapping[str, Any]) -> bytes:
    """Return the fixed JSON representation used for deterministic bundle IDs."""

    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


@dataclass(frozen=True)
class I1ArtifactReference:
    kind: I1ArtifactKind
    npz_relpath: str
    json_relpath: str
    npz_sha256: str
    npz_bytes: int
    json_sha256: str
    json_bytes: int
    schema_version: str
    algorithm_version: str

    def __post_init__(self) -> None:
        if not isinstance(self.kind, I1ArtifactKind):
            raise ValueError("kind must be an I1ArtifactKind")
        for name in ("npz_relpath", "json_relpath"):
            _safe_relpath(getattr(self, name), name)
        expected_prefix = f"artifacts/{self.kind.value}/"
        if not self.npz_relpath.startswith(expected_prefix) or not self.json_relpath.startswith(
            expected_prefix
        ):
            raise ValueError("artifact paths must be contained in their kind directory")
        if not self.npz_relpath.endswith(".npz") or not self.json_relpath.endswith(".json"):
            raise ValueError("artifact references require NPZ and JSON siblings")
        for name in ("npz_sha256", "json_sha256"):
            _sha(getattr(self, name), name)
        for name in ("npz_bytes", "json_bytes"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if not self.schema_version or not self.algorithm_version:
            raise ValueError("source schema and algorithm versions are required")

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            **{name: getattr(self, name) for name in self.__dataclass_fields__ if name != "kind"},
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> I1ArtifactReference:
        require_fields(data, set(cls.__dataclass_fields__), cls.__name__)
        reject_unknown_fields(data, set(cls.__dataclass_fields__), cls.__name__)
        values = dict(data)
        values["kind"] = I1ArtifactKind(values["kind"])
        return cls(**values)


@dataclass(frozen=True)
class I1CoordinateContext:
    origin_xyz: tuple[float, float, float]
    frame: I1CoordinateFrame
    unit: I1CoordinateUnit
    transform: str
    origin_source_kind: str
    origin_source_schema_version: str
    origin_source_algorithm_version: str
    origin_source_json_sha256: str
    verification_result: str

    def __post_init__(self) -> None:
        origin = tuple(float(value) for value in self.origin_xyz)
        if len(origin) != 3 or not all(math.isfinite(value) for value in origin):
            raise ValueError("origin_xyz must contain three finite values")
        object.__setattr__(self, "origin_xyz", origin)
        if self.frame is not I1CoordinateFrame.LOCAL_CARTESIAN_XYZ:
            raise ValueError("I1 v1 supports only LOCAL_CARTESIAN_XYZ")
        if self.unit is not I1CoordinateUnit.METRE:
            raise ValueError("I1 v1 supports only metres")
        if self.transform != "identity":
            raise ValueError("I1 v1 supports only an identity transform")
        if self.origin_source_kind != I1_ORIGIN_SOURCE_KIND:
            raise ValueError("unsupported origin source kind")
        if not self.origin_source_schema_version or not self.origin_source_algorithm_version:
            raise ValueError("origin source versions are required")
        _sha(self.origin_source_json_sha256, "origin_source_json_sha256")
        if self.verification_result != I1_ORIGIN_VERIFICATION:
            raise ValueError("origin verification must be an exact verified match")

    def to_dict(self) -> dict[str, Any]:
        result = dict(self.__dict__)
        result["origin_xyz"] = list(self.origin_xyz)
        result["frame"] = self.frame.value
        result["unit"] = self.unit.value
        return result

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> I1CoordinateContext:
        require_fields(data, set(cls.__dataclass_fields__), cls.__name__)
        reject_unknown_fields(data, set(cls.__dataclass_fields__), cls.__name__)
        values = dict(data)
        values["frame"] = I1CoordinateFrame(values["frame"])
        values["unit"] = I1CoordinateUnit(values["unit"])
        return cls(**values)


@dataclass(frozen=True)
class I1BundleManifest:
    schema_version: str
    interface_version: str
    bundle_id: str
    source_id: str
    manifest_source_id: str
    eligibility_policy_version: str
    coordinate_context: I1CoordinateContext
    w1: I1ArtifactReference
    v2: I1ArtifactReference
    v2a_state: I1OptionalArtifactState
    v2a: I1ArtifactReference | None
    warnings: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if (
            self.schema_version != I1_SCHEMA_VERSION
            or self.interface_version != I1_INTERFACE_VERSION
        ):
            raise ValueError("unsupported I1 schema or interface version")
        _sha(self.bundle_id, "bundle_id")
        if not self.source_id or not self.manifest_source_id:
            raise ValueError("source identifiers are required")
        if self.eligibility_policy_version != I1_ELIGIBILITY_POLICY_VERSION:
            raise ValueError("unsupported I1 eligibility policy")
        if not isinstance(self.coordinate_context, I1CoordinateContext):
            raise ValueError("coordinate_context must use I1CoordinateContext")
        if not isinstance(self.w1, I1ArtifactReference) or self.w1.kind is not I1ArtifactKind.W1:
            raise ValueError("w1 reference is invalid")
        if not isinstance(self.v2, I1ArtifactReference) or self.v2.kind is not I1ArtifactKind.V2:
            raise ValueError("v2 reference is invalid")
        if self.v2a_state is I1OptionalArtifactState.PRESENT:
            if (
                not isinstance(self.v2a, I1ArtifactReference)
                or self.v2a.kind is not I1ArtifactKind.V2A
            ):
                raise ValueError("present V2A requires a V2A reference")
        elif self.v2a is not None:
            raise ValueError("absent V2A must not have a reference")
        object.__setattr__(self, "warnings", tuple(str(item) for item in self.warnings))
        object.__setattr__(self, "limitations", tuple(str(item) for item in self.limitations))
        if self.bundle_id != deterministic_bundle_id(self):
            raise ValueError("bundle_id does not match canonical I1 inputs")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "interface_version": self.interface_version,
            "bundle_id": self.bundle_id,
            "source_id": self.source_id,
            "manifest_source_id": self.manifest_source_id,
            "eligibility_policy_version": self.eligibility_policy_version,
            "coordinate_context": self.coordinate_context.to_dict(),
            "w1": self.w1.to_dict(),
            "v2": self.v2.to_dict(),
            "v2a_state": self.v2a_state.value,
            "v2a": self.v2a.to_dict() if self.v2a is not None else None,
            "warnings": list(self.warnings),
            "limitations": list(self.limitations),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> I1BundleManifest:
        require_fields(data, set(cls.__dataclass_fields__), cls.__name__)
        reject_unknown_fields(data, set(cls.__dataclass_fields__), cls.__name__)
        values = dict(data)
        values["coordinate_context"] = I1CoordinateContext.from_dict(values["coordinate_context"])
        values["w1"] = I1ArtifactReference.from_dict(values["w1"])
        values["v2"] = I1ArtifactReference.from_dict(values["v2"])
        values["v2a_state"] = I1OptionalArtifactState(values["v2a_state"])
        values["v2a"] = (
            I1ArtifactReference.from_dict(values["v2a"]) if values["v2a"] is not None else None
        )
        values["warnings"] = tuple(values["warnings"])
        values["limitations"] = tuple(values["limitations"])
        return cls(**values)


def _bundle_id_payload(manifest: I1BundleManifest) -> dict[str, Any]:
    return {
        "schema_version": manifest.schema_version,
        "interface_version": manifest.interface_version,
        "eligibility_policy_version": manifest.eligibility_policy_version,
        "source_id": manifest.source_id,
        "manifest_source_id": manifest.manifest_source_id,
        "coordinate_context": manifest.coordinate_context.to_dict(),
        "w1": manifest.w1.to_dict(),
        "v2": manifest.v2.to_dict(),
        "v2a_state": manifest.v2a_state.value,
        "v2a": manifest.v2a.to_dict() if manifest.v2a is not None else None,
    }


def deterministic_bundle_id(manifest: I1BundleManifest) -> str:
    """Calculate the destination-independent immutable I1 bundle identifier."""

    return hashlib.sha256(canonical_json(_bundle_id_payload(manifest))).hexdigest()


def calculate_bundle_id(
    *,
    source_id: str,
    manifest_source_id: str,
    coordinate_context: I1CoordinateContext,
    w1: I1ArtifactReference,
    v2: I1ArtifactReference,
    v2a_state: I1OptionalArtifactState,
    v2a: I1ArtifactReference | None,
) -> str:
    """Calculate an I1 ID before constructing the self-validating manifest."""

    payload = {
        "schema_version": I1_SCHEMA_VERSION,
        "interface_version": I1_INTERFACE_VERSION,
        "eligibility_policy_version": I1_ELIGIBILITY_POLICY_VERSION,
        "source_id": source_id,
        "manifest_source_id": manifest_source_id,
        "coordinate_context": coordinate_context.to_dict(),
        "w1": w1.to_dict(),
        "v2": v2.to_dict(),
        "v2a_state": v2a_state.value,
        "v2a": v2a.to_dict() if v2a is not None else None,
    }
    return hashlib.sha256(canonical_json(payload)).hexdigest()
