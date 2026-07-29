"""Strict, non-I/O contracts for persisted V3 tree-segmentation artifacts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from core.contracts._json import json_safe_mapping, reject_unknown_fields, require_fields
from core.contracts.tree_inputs import TreeSegmentationInputIdentity
from core.contracts.vegetation_trees import (
    TREE_SEGMENTATION_ALGORITHM_VERSION,
    TREE_SEGMENTATION_SCHEMA_VERSION,
    TreeInstanceRecord,
    TreeSegmentationConfig,
    TreeSegmentationStatistics,
    TreeSegmentationStatus,
)

TREE_SEGMENTATION_ARTIFACT_SCHEMA_VERSION = "tree-segmentation-artifact-v1"
TREE_SEGMENTATION_NPZ_FILENAME = "tree_segmentation.npz"
TREE_SEGMENTATION_METADATA_FILENAME = "tree_segmentation.json"
TREE_SEGMENTATION_MANIFEST_FILENAME = "tree_segmentation_manifest.json"


def _strict_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _non_negative_int(value: Any, name: str) -> int:
    if (
        not isinstance(value, int | np.integer)
        or isinstance(value, bool | np.bool_)
        or int(value) < 0
    ):
        raise ValueError(f"{name} must be a non-negative integer")
    return int(value)


def _shape(value: Any) -> tuple[int]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes | bytearray):
        raise ValueError("shape must be a one-dimensional sequence")
    if len(value) != 1:
        raise ValueError("shape must contain exactly one dimension")
    return (_non_negative_int(value[0], "shape dimension"),)


def _string_sequence(value: Any, name: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes | bytearray):
        raise ValueError(f"{name} must be a non-string sequence")
    if any(not isinstance(item, str) for item in value):
        raise ValueError(f"{name} must contain only strings")
    return tuple(value)


def _records(value: Any) -> tuple[TreeInstanceRecord, ...]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes | bytearray):
        raise ValueError("records must be a non-string sequence")
    records = tuple(value)
    if any(not isinstance(record, TreeInstanceRecord) for record in records):
        raise ValueError("records must contain TreeInstanceRecord values")
    segment_ids = tuple(record.segment_id for record in records)
    if segment_ids != tuple(range(len(records))):
        raise ValueError("records must be in canonical contiguous segment_id order")
    return records


def _sha256(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


@dataclass(frozen=True)
class TreeSegmentationArtifactArraySpec:
    """Metadata descriptor for one one-dimensional array in the fixed NPZ."""

    dtype: str
    shape: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.dtype, str) or not self.dtype or self.dtype != self.dtype.strip():
            raise ValueError("dtype must be a non-empty string without surrounding whitespace")
        object.__setattr__(self, "shape", _shape(self.shape))

    def to_dict(self) -> dict[str, Any]:
        return {"dtype": self.dtype, "shape": list(self.shape)}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TreeSegmentationArtifactArraySpec:
        values = _strict_mapping(data, cls.__name__)
        fields = set(cls.__dataclass_fields__)
        require_fields(values, fields, cls.__name__)
        reject_unknown_fields(values, fields, cls.__name__)
        return cls(**dict(values))


@dataclass(frozen=True)
class TreeSegmentationArtifactArrayInventory:
    """Exact inventory of arrays persisted for one V3 segmentation result."""

    segment_id_by_point: TreeSegmentationArtifactArraySpec
    tree_id_by_point: TreeSegmentationArtifactArraySpec
    tree_distance_ready_by_point: TreeSegmentationArtifactArraySpec
    tree_member_offsets: TreeSegmentationArtifactArraySpec
    ordered_tree_member_row_index: TreeSegmentationArtifactArraySpec

    def __post_init__(self) -> None:
        expected_dtypes = {
            "segment_id_by_point": "<i4",
            "tree_id_by_point": "<i4",
            "tree_distance_ready_by_point": "|b1",
            "tree_member_offsets": "<i8",
            "ordered_tree_member_row_index": "<i8",
        }
        for name, expected_dtype in expected_dtypes.items():
            value = getattr(self, name)
            if not isinstance(value, TreeSegmentationArtifactArraySpec):
                raise ValueError(f"{name} must be a TreeSegmentationArtifactArraySpec")
            if value.dtype != expected_dtype:
                raise ValueError(f"{name}.dtype must equal {expected_dtype!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "segment_id_by_point": self.segment_id_by_point.to_dict(),
            "tree_id_by_point": self.tree_id_by_point.to_dict(),
            "tree_distance_ready_by_point": self.tree_distance_ready_by_point.to_dict(),
            "tree_member_offsets": self.tree_member_offsets.to_dict(),
            "ordered_tree_member_row_index": self.ordered_tree_member_row_index.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TreeSegmentationArtifactArrayInventory:
        values = _strict_mapping(data, cls.__name__)
        fields = set(cls.__dataclass_fields__)
        require_fields(values, fields, cls.__name__)
        reject_unknown_fields(values, fields, cls.__name__)
        parsed = {
            name: TreeSegmentationArtifactArraySpec.from_dict(values[name]) for name in fields
        }
        return cls(**parsed)


@dataclass(frozen=True)
class TreeSegmentationArtifactManifest:
    """Checksums and byte counts for the metadata and NPZ sibling files."""

    artifact_schema_version: str
    metadata_sha256: str
    metadata_bytes: int
    npz_sha256: str
    npz_bytes: int

    def __post_init__(self) -> None:
        if self.artifact_schema_version != TREE_SEGMENTATION_ARTIFACT_SCHEMA_VERSION:
            raise ValueError("unsupported V3 tree-segmentation artifact schema")
        _sha256(self.metadata_sha256, "metadata_sha256")
        _sha256(self.npz_sha256, "npz_sha256")
        object.__setattr__(
            self,
            "metadata_bytes",
            _non_negative_int(self.metadata_bytes, "metadata_bytes"),
        )
        object.__setattr__(self, "npz_bytes", _non_negative_int(self.npz_bytes, "npz_bytes"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_schema_version": self.artifact_schema_version,
            "metadata_sha256": self.metadata_sha256,
            "metadata_bytes": self.metadata_bytes,
            "npz_sha256": self.npz_sha256,
            "npz_bytes": self.npz_bytes,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TreeSegmentationArtifactManifest:
        values = _strict_mapping(data, cls.__name__)
        fields = set(cls.__dataclass_fields__)
        require_fields(values, fields, cls.__name__)
        reject_unknown_fields(values, fields, cls.__name__)
        return cls(**dict(values))


@dataclass(frozen=True)
class TreeSegmentationArtifactMetadata:
    """Structured V3 result metadata, excluding the five point/member arrays."""

    artifact_schema_version: str
    result_schema_version: str
    algorithm_version: str
    source_identity: TreeSegmentationInputIdentity
    config: TreeSegmentationConfig
    statistics: TreeSegmentationStatistics
    records: tuple[TreeInstanceRecord, ...]
    input_provenance: Mapping[str, Any]
    coordinate_context: Mapping[str, Any]
    warnings: tuple[str, ...]
    limitations: tuple[str, ...]
    arrays: TreeSegmentationArtifactArrayInventory

    def __post_init__(self) -> None:
        if self.artifact_schema_version != TREE_SEGMENTATION_ARTIFACT_SCHEMA_VERSION:
            raise ValueError("unsupported V3 tree-segmentation artifact schema")
        if self.result_schema_version != TREE_SEGMENTATION_SCHEMA_VERSION:
            raise ValueError("unsupported V3 tree segmentation schema")
        if self.algorithm_version != TREE_SEGMENTATION_ALGORITHM_VERSION:
            raise ValueError("unsupported V3 tree segmentation algorithm version")
        if not isinstance(self.source_identity, TreeSegmentationInputIdentity):
            raise ValueError("source_identity must be a TreeSegmentationInputIdentity")
        if not isinstance(self.config, TreeSegmentationConfig):
            raise ValueError("config must be a TreeSegmentationConfig")
        if not isinstance(self.statistics, TreeSegmentationStatistics):
            raise ValueError("statistics must be a TreeSegmentationStatistics")
        if not isinstance(self.arrays, TreeSegmentationArtifactArrayInventory):
            raise ValueError("arrays must be a TreeSegmentationArtifactArrayInventory")

        records = _records(self.records)
        provenance = _strict_mapping(self.input_provenance, "input_provenance")
        context = _strict_mapping(self.coordinate_context, "coordinate_context")
        warnings = _string_sequence(self.warnings, "warnings")
        limitations = _string_sequence(self.limitations, "limitations")

        point_count = self.source_identity.point_count
        record_count = len(records)
        member_count = self.arrays.ordered_tree_member_row_index.shape[0]
        if self.statistics.input_point_count != point_count:
            raise ValueError("statistics.input_point_count must match source_identity.point_count")
        if self.statistics.record_count != record_count:
            raise ValueError("statistics.record_count must match records")
        expected_status_counts = {
            "confirmed_tree_count": sum(
                record.status is TreeSegmentationStatus.CONFIRMED_TREE for record in records
            ),
            "review_required_count": sum(
                record.status is TreeSegmentationStatus.REVIEW_REQUIRED for record in records
            ),
            "provisional_count": sum(
                record.status is TreeSegmentationStatus.PROVISIONAL for record in records
            ),
            "rejected_count": sum(
                record.status is TreeSegmentationStatus.REJECTED for record in records
            ),
            "unresolved_count": sum(
                record.status is TreeSegmentationStatus.UNRESOLVED for record in records
            ),
        }
        status_count_errors = [
            f"statistics.{name} must match records"
            for name, expected in expected_status_counts.items()
            if getattr(self.statistics, name) != expected
        ]
        if status_count_errors:
            raise ValueError("; ".join(status_count_errors))
        for name in (
            "segment_id_by_point",
            "tree_id_by_point",
            "tree_distance_ready_by_point",
        ):
            if getattr(self.arrays, name).shape != (point_count,):
                raise ValueError(f"arrays.{name}.shape must match source_identity.point_count")
        if self.arrays.tree_member_offsets.shape != (record_count + 1,):
            raise ValueError("arrays.tree_member_offsets.shape must equal record_count plus one")
        if member_count > point_count:
            raise ValueError(
                "arrays.ordered_tree_member_row_index.shape must not exceed point_count"
            )

        object.__setattr__(self, "records", records)
        object.__setattr__(
            self,
            "input_provenance",
            json_safe_mapping(provenance, "input_provenance"),
        )
        object.__setattr__(
            self,
            "coordinate_context",
            json_safe_mapping(context, "coordinate_context"),
        )
        object.__setattr__(self, "warnings", warnings)
        object.__setattr__(self, "limitations", limitations)

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_schema_version": self.artifact_schema_version,
            "result_schema_version": self.result_schema_version,
            "algorithm_version": self.algorithm_version,
            "source_identity": self.source_identity.to_dict(),
            "config": self.config.to_dict(),
            "statistics": self.statistics.to_dict(),
            "records": [record.to_dict() for record in self.records],
            "input_provenance": dict(self.input_provenance),
            "coordinate_context": dict(self.coordinate_context),
            "warnings": list(self.warnings),
            "limitations": list(self.limitations),
            "arrays": self.arrays.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TreeSegmentationArtifactMetadata:
        values = _strict_mapping(data, cls.__name__)
        fields = set(cls.__dataclass_fields__)
        require_fields(values, fields, cls.__name__)
        reject_unknown_fields(values, fields, cls.__name__)
        parsed = dict(values)
        parsed["source_identity"] = TreeSegmentationInputIdentity.from_dict(
            _strict_mapping(parsed["source_identity"], "source_identity")
        )
        parsed["config"] = TreeSegmentationConfig.from_dict(
            _strict_mapping(parsed["config"], "config")
        )
        parsed["statistics"] = TreeSegmentationStatistics.from_dict(
            _strict_mapping(parsed["statistics"], "statistics")
        )
        raw_records = parsed["records"]
        if not isinstance(raw_records, Sequence) or isinstance(
            raw_records, str | bytes | bytearray
        ):
            raise ValueError("records must be a non-string sequence")
        parsed["records"] = tuple(
            TreeInstanceRecord.from_dict(_strict_mapping(value, "record")) for value in raw_records
        )
        parsed["arrays"] = TreeSegmentationArtifactArrayInventory.from_dict(
            _strict_mapping(parsed["arrays"], "arrays")
        )
        return cls(**parsed)
