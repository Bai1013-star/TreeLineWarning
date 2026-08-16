"""Content-addressed exact RAW membership artifacts (``tower-memberships/*.npz``).

A confirmed ordered tower persists its exact full-resolution RAW membership as
an immutable, derived, content-addressed artifact under the project root.  The
artifact is a single ``.npz`` whose SHA-256 over the whole file is the content
address; it embeds the compact metadata needed to validate it in isolation:

- ``source_index``      int64    exact original source row identities
- ``xyz_absolute_m``    float64  exact SOURCE_LAS_ABSOLUTE XYZ (n, 3)
- ``meta_schema_version``                 artifact schema version
- ``meta_source_content_digest``          authoritative source content digest
- ``meta_source_record_digest``           authoritative source record digest
- ``meta_segmentation_config_digest``     sealed segmentation configuration
- ``meta_config_json``                    JSON of the full segmentation config
- ``meta_member_count``                   member count (must equal n)
- ``meta_member_bounds_min/max``          6 floats: inclusive member bounds
- ``meta_base_xyz`` / ``meta_top_xyz``    definitive refined shaft geometry

Reading validates schema, source binding, member count, bounds, and array
dtypes/contents; any corrupt, wrong-source, or non-finite artifact fails closed
with ``MembershipArtifactIntegrityError``.  The ``ArtifactReference`` returned
for a confirmed tower carries the digest + byte size + relative path so the
immutable tower generation records exactly which artifact produced it.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from core.contracts.project import ArtifactReference, SourceIdentityBinding

from algorithms.towers.segmentation import (
    TowerLocalMembership,
)

TOWER_MEMBERSHIP_ARTIFACT_SCHEMA_VERSION = "treeline-tower-membership-artifact-v1"
TOWER_MEMBERSHIP_ROLE = "tower-local-membership"
TOWER_MEMBERSHIPS_RELATIVE_ROOT = "tower-memberships"


class MembershipArtifactError(RuntimeError):
    """Closed failure raised by the membership artifact boundary."""


class MembershipArtifactIntegrityError(MembershipArtifactError):
    """Raised when an artifact is missing, corrupt, or source-mismatched."""


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def membership_artifact_relative_path(digest: str) -> str:
    if type(digest) is not str or len(digest) != 64:
        raise MembershipArtifactError("digest must be a SHA-256 digest")
    return f"{TOWER_MEMBERSHIPS_RELATIVE_ROOT}/{digest}.npz"


@dataclass(frozen=True, slots=True)
class MembershipArtifactRecord:
    """Metadata describing one persisted membership artifact."""

    relative_path: str
    sha256: str
    byte_size: int
    schema_version: str
    membership_digest: str
    source_content_identity_digest: str
    source_identity_record_digest: str
    segmentation_config_digest: str
    member_count: int
    member_bounds_absolute_m: tuple[tuple[float, float, float], tuple[float, float, float]]
    height_m: float
    algorithm_version: str

    def to_artifact_reference(self) -> ArtifactReference:
        return ArtifactReference(
            relative_path=self.relative_path,
            role=TOWER_MEMBERSHIP_ROLE,
            sha256=self.sha256,
            byte_size=self.byte_size,
            schema_version=self.schema_version,
        )

    def to_dict(self) -> dict[str, Any]:
        minimum, maximum = self.member_bounds_absolute_m
        return {
            "schema_version": self.schema_version,
            "relative_path": self.relative_path,
            "sha256": self.sha256,
            "byte_size": self.byte_size,
            "membership_digest": self.membership_digest,
            "source_content_identity_digest": self.source_content_identity_digest,
            "source_identity_record_digest": self.source_identity_record_digest,
            "segmentation_config_digest": self.segmentation_config_digest,
            "member_count": self.member_count,
            "member_bounds_min_xyz": list(minimum),
            "member_bounds_max_xyz": list(maximum),
            "height_m": self.height_m,
            "algorithm_version": self.algorithm_version,
        }


def write_membership_artifact(
    project_root: str | Path,
    membership: TowerLocalMembership,
) -> MembershipArtifactRecord:
    """Persist one exact membership as an immutable content-addressed artifact.

    Returns the metadata record (including the digest + byte size) so callers
    can build the ``ArtifactReference`` embedded in the confirmed tower's
    ``TowerRefinement.local_region_reference``.  If the artifact already exists
    for the same digest the existing file is reused (it is content-addressed and
    immutable).  The artifact is written before the confirming generation is
    published; an orphaned artifact is never authoritative because no immutable
    generation references it.
    """
    if type(membership) is not TowerLocalMembership:
        raise MembershipArtifactError("membership must be a TowerLocalMembership")
    root = Path(project_root).resolve()
    data = _encode_artifact(membership)
    file_digest = _sha256_bytes(data)
    relative = membership_artifact_relative_path(file_digest)
    path = root.joinpath(*relative.split("/"))
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + f".tmp-{_token_hex()}")
        try:
            tmp.write_bytes(data)
            tmp.replace(path)
        finally:
            try:
                tmp.unlink()
            except FileNotFoundError:
                pass
    byte_size = len(data)
    minimum, maximum = membership.member_bounds_absolute_m
    return MembershipArtifactRecord(
        relative_path=relative,
        sha256=file_digest,
        byte_size=byte_size,
        schema_version=TOWER_MEMBERSHIP_ARTIFACT_SCHEMA_VERSION,
        membership_digest=membership.membership_digest(),
        source_content_identity_digest=membership.source_binding.source_content_identity_digest,
        source_identity_record_digest=membership.source_binding.source_identity_record_digest,
        segmentation_config_digest=membership.config_digest,
        member_count=membership.member_count,
        member_bounds_absolute_m=(minimum, maximum),
        height_m=membership.height_m,
        algorithm_version=membership.algorithm_version,
    )


def _token_hex() -> str:
    import secrets

    return secrets.token_hex(8)


def _encode_artifact(membership: TowerLocalMembership) -> bytes:
    minimum, maximum = membership.member_bounds_absolute_m
    import io
    import json as _json

    buffer = io.BytesIO()
    np.savez(
        buffer,
        source_index=membership.source_index.astype(np.int64, copy=False),
        xyz_absolute_m=membership.xyz_absolute_m.astype(np.float64, copy=False),
        meta_schema_version=np.asarray([TOWER_MEMBERSHIP_ARTIFACT_SCHEMA_VERSION], dtype="U48"),
        meta_algorithm_version=np.asarray([membership.algorithm_version], dtype="U48"),
        meta_source_content_digest=np.asarray(
            [membership.source_binding.source_content_identity_digest], dtype="U64"
        ),
        meta_source_record_digest=np.asarray(
            [membership.source_binding.source_identity_record_digest], dtype="U64"
        ),
        meta_segmentation_config_digest=np.asarray([membership.config_digest], dtype="U64"),
        meta_membership_digest=np.asarray([membership.membership_digest()], dtype="U64"),
        meta_config_json=np.asarray(
            [_json.dumps(membership.segmentation_config.to_dict(), sort_keys=True)], dtype="U2048"
        ),
        meta_member_count=np.asarray([membership.member_count], dtype=np.int64),
        meta_member_bounds_min_xyz=np.asarray(list(minimum), dtype=np.float64),
        meta_member_bounds_max_xyz=np.asarray(list(maximum), dtype=np.float64),
        meta_base_xyz=np.asarray(list(membership.base_xyz_absolute_m), dtype=np.float64),
        meta_top_xyz=np.asarray(list(membership.top_xyz_absolute_m), dtype=np.float64),
        meta_height_m=np.asarray([membership.height_m], dtype=np.float64),
    )
    return buffer.getvalue()


def read_membership_artifact(
    project_root: str | Path,
    record: MembershipArtifactRecord | ArtifactReference | str,
    *,
    source_binding: SourceIdentityBinding,
    segmentation_config_digest: str | None = None,
    expected_digest: str | None = None,
) -> TowerLocalMembership:
    """Read + strictly validate a persisted membership artifact.

    Fails closed on a missing/corrupt file, a wrong SHA-256, a source-binding
    mismatch, a wrong member count/bounds, NaN/invalid points, or a
    segmentation-config mismatch.  Returns the exact ``TowerLocalMembership``
    reconstructed from the artifact (all authoritative member points preserved).
    """
    if type(source_binding) is not SourceIdentityBinding:
        raise MembershipArtifactError("source_binding must be a SourceIdentityBinding")
    if isinstance(record, str):
        relative_path = record
        expected_digest = expected_digest
    elif isinstance(record, ArtifactReference):
        relative_path = record.relative_path
        if expected_digest is None:
            expected_digest = record.sha256
        if record.role != TOWER_MEMBERSHIP_ROLE:
            raise MembershipArtifactIntegrityError(
                f"artifact role is not {TOWER_MEMBERSHIP_ROLE}: {record.role}"
            )
    elif isinstance(record, MembershipArtifactRecord):
        relative_path = record.relative_path
        if expected_digest is None:
            expected_digest = record.sha256
    else:
        raise MembershipArtifactError("record must be a path, ArtifactReference, or record")
    if expected_digest is not None and (
        type(expected_digest) is not str or len(expected_digest) != 64
    ):
        raise MembershipArtifactError("expected_digest must be a SHA-256 digest")
    root = Path(project_root).resolve()
    path = root.joinpath(*relative_path.split("/"))
    if not path.is_file():
        raise MembershipArtifactIntegrityError(f"membership artifact is missing: {relative_path}")
    data = path.read_bytes()
    actual = _sha256_bytes(data)
    if expected_digest is not None and actual != expected_digest:
        raise MembershipArtifactIntegrityError(
            f"membership artifact SHA-256 mismatch for {relative_path}"
        )
    config = _decode_artifact(data, relative_path)
    _validate_artifact_content(
        config,
        source_binding=source_binding,
        segmentation_config_digest=segmentation_config_digest,
    )
    return _rebuild_membership(config, source_binding, relative_path)


def _decode_artifact(data: bytes, relative_path: str) -> dict[str, Any]:
    import io

    try:
        payload = np.load(io.BytesIO(data), allow_pickle=False)
        return {
            key: payload[key] for key in payload.files
        }
    except Exception as exc:
        raise MembershipArtifactIntegrityError(
            f"membership artifact is corrupt: {relative_path} ({exc})"
        ) from exc


def _meta_string(payload: dict[str, Any], key: str, name: str) -> str:
    value = payload.get(key)
    if value is None or value.ndim != 1 or len(value) != 1 or value.dtype.kind != "U":
        raise MembershipArtifactIntegrityError(f"{name} metadata is malformed")
    text = str(value[0])
    if not text or text != text.strip():
        raise MembershipArtifactIntegrityError(f"{name} metadata is not a clean string")
    return text


def _validate_artifact_content(
    payload: dict[str, Any],
    *,
    source_binding: SourceIdentityBinding,
    segmentation_config_digest: str | None,
) -> None:
    schema = _meta_string(payload, "meta_schema_version", "schema_version")
    if schema != TOWER_MEMBERSHIP_ARTIFACT_SCHEMA_VERSION:
        raise MembershipArtifactIntegrityError("membership artifact schema version is unsupported")
    content = _meta_string(payload, "meta_source_content_digest", "source content digest")
    record = _meta_string(payload, "meta_source_record_digest", "source record digest")
    if (
        content != source_binding.source_content_identity_digest
        or record != source_binding.source_identity_record_digest
    ):
        raise MembershipArtifactIntegrityError(
            "membership artifact is bound to a different source identity"
        )
    if segmentation_config_digest is not None:
        config_digest = _meta_string(
            payload, "meta_segmentation_config_digest", "segmentation config digest"
        )
        if config_digest != segmentation_config_digest:
            raise MembershipArtifactIntegrityError(
                "membership artifact segmentation config does not match"
            )
    idx = payload.get("source_index")
    xyz = payload.get("xyz_absolute_m")
    if idx is None or xyz is None:
        raise MembershipArtifactIntegrityError("membership artifact is missing point arrays")
    if idx.dtype != np.int64 or idx.ndim != 1:
        raise MembershipArtifactIntegrityError("source_index must be a 1-D int64 array")
    if xyz.dtype != np.float64 or xyz.ndim != 2 or xyz.shape[1] != 3:
        raise MembershipArtifactIntegrityError("xyz_absolute_m must be an (n, 3) float64 array")
    if len(idx) != len(xyz):
        raise MembershipArtifactIntegrityError("source_index and xyz_absolute_m length mismatch")
    if not np.isfinite(idx).all() or not np.isfinite(xyz).all():
        raise MembershipArtifactIntegrityError("membership artifact contains non-finite points")
    member_count_meta = payload.get("meta_member_count")
    if (
        member_count_meta is None
        or member_count_meta.ndim != 1
        or len(member_count_meta) != 1
        or member_count_meta.dtype != np.int64
    ):
        raise MembershipArtifactIntegrityError("member_count metadata is malformed")
    if int(member_count_meta[0]) != len(idx):
        raise MembershipArtifactIntegrityError("member_count metadata does not match the arrays")


def _rebuild_membership(
    payload: dict[str, Any],
    source_binding: SourceIdentityBinding,
    relative_path: str,
) -> TowerLocalMembership:
    import json

    from algorithms.towers.segmentation import TowerLocalSegmentationConfig

    idx = payload["source_index"]
    xyz = payload["xyz_absolute_m"]
    member_count = int(len(idx))
    bounds_min = payload.get("meta_member_bounds_min_xyz")
    bounds_max = payload.get("meta_member_bounds_max_xyz")
    if (
        bounds_min is None
        or bounds_max is None
        or bounds_min.ndim != 1
        or bounds_min.dtype != np.float64
        or len(bounds_min) != 3
        or bounds_max.ndim != 1
        or bounds_max.dtype != np.float64
        or len(bounds_max) != 3
    ):
        raise MembershipArtifactIntegrityError("member bounds metadata is malformed")
    minimum = tuple(float(value) for value in bounds_min)
    maximum = tuple(float(value) for value in bounds_max)
    computed_min = tuple(float(value) for value in np.min(xyz, axis=0))
    computed_max = tuple(float(value) for value in np.max(xyz, axis=0))
    if minimum != computed_min or maximum != computed_max:
        raise MembershipArtifactIntegrityError("member bounds do not match the member points")
    height = payload.get("meta_height_m")
    if (
        height is None
        or height.ndim != 1
        or len(height) != 1
        or height.dtype != np.float64
        or not math_isfinite(float(height[0]))
        or float(height[0]) <= 0.0
    ):
        raise MembershipArtifactIntegrityError("height metadata is malformed")
    base_meta = payload.get("meta_base_xyz")
    top_meta = payload.get("meta_top_xyz")
    if (
        base_meta is None
        or top_meta is None
        or base_meta.ndim != 1
        or base_meta.dtype != np.float64
        or len(base_meta) != 3
        or top_meta.ndim != 1
        or top_meta.dtype != np.float64
        or len(top_meta) != 3
    ):
        raise MembershipArtifactIntegrityError("base/top geometry metadata is malformed")
    base_xyz = tuple(float(value) for value in base_meta)
    top_xyz = tuple(float(value) for value in top_meta)
    if not all(math_isfinite(value) for value in (*base_xyz, *top_xyz)):
        raise MembershipArtifactIntegrityError("base/top geometry metadata is non-finite")
    if top_xyz[2] - base_xyz[2] != float(height[0]):
        raise MembershipArtifactIntegrityError("base/top geometry does not match height metadata")
    config_digest = _meta_string(
        payload, "meta_segmentation_config_digest", "segmentation config digest"
    )
    config_json = payload.get("meta_config_json")
    if (
        config_json is None
        or config_json.ndim != 1
        or len(config_json) != 1
        or config_json.dtype.kind != "U"
    ):
        raise MembershipArtifactIntegrityError("segmentation config metadata is malformed")
    try:
        config_data = json.loads(str(config_json[0]))
    except (TypeError, ValueError) as exc:
        raise MembershipArtifactIntegrityError(
            f"segmentation config metadata is not valid JSON: {exc}"
        ) from exc
    if type(config_data) is not dict:
        raise MembershipArtifactIntegrityError("segmentation config metadata must be a JSON object")
    # ``schema_version`` is a derived field, not a dataclass constructor argument.
    config_fields = set(TowerLocalSegmentationConfig.__dataclass_fields__)
    allowed = config_fields | {"schema_version"}
    if not config_fields.issubset(set(config_data)):
        raise MembershipArtifactIntegrityError("segmentation config metadata is missing fields")
    unknown = set(config_data) - allowed
    if unknown:
        raise MembershipArtifactIntegrityError(
            f"segmentation config metadata has unknown fields: {sorted(unknown)}"
        )
    try:
        config = TowerLocalSegmentationConfig(
            **{key: config_data[key] for key in config_fields}
        )
    except (TypeError, ValueError) as exc:
        raise MembershipArtifactIntegrityError(
            f"segmentation config metadata is invalid: {exc}"
        ) from exc
    if config.configuration_digest() != config_digest:
        raise MembershipArtifactIntegrityError(
            "segmentation config digest does not match the embedded configuration"
        )
    membership = TowerLocalMembership(
        source_binding=source_binding,
        segmentation_config=config,
        config_digest=config_digest,
        source_index=idx.copy(),
        xyz_absolute_m=xyz.copy(),
        member_count=member_count,
        member_bounds_absolute_m=(minimum, maximum),
        base_xyz_absolute_m=base_xyz,
        top_xyz_absolute_m=top_xyz,
        height_m=float(height[0]),
        raw_tile_cache_identity={
            "artifact": relative_path,
            "reconstructed_from_artifact": True,
        },
        quality_diagnostics={
            "artifact": relative_path,
            "reconstructed_from_artifact": True,
        },
        algorithm_version=_meta_string(payload, "meta_algorithm_version", "algorithm version"),
    )
    embedded_membership_digest = _meta_string(
        payload, "meta_membership_digest", "membership digest"
    )
    if membership.membership_digest() != embedded_membership_digest:
        raise MembershipArtifactIntegrityError(
            "membership digest does not match the reconstructed membership content"
        )
    return membership


def math_isfinite(value: float) -> bool:
    import math

    return math.isfinite(value)


__all__ = [
    "TOWER_MEMBERSHIP_ARTIFACT_SCHEMA_VERSION",
    "TOWER_MEMBERSHIP_ROLE",
    "TOWER_MEMBERSHIPS_RELATIVE_ROOT",
    "MembershipArtifactError",
    "MembershipArtifactIntegrityError",
    "MembershipArtifactRecord",
    "membership_artifact_relative_path",
    "read_membership_artifact",
    "write_membership_artifact",
]
