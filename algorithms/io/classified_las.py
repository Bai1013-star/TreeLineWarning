"""Classification-only LAS derivative writer and lockstep verifier.

Authority: docs/full-resolution-ground-classification-design.md sections 12-13
and the frozen Bundle C plan.  The raw source is never opened for mutation; a
new derivative is created that preserves every point record field except the
logical base LAS classification, which is changed according to the frozen
mapping (confirmed -> 2, stale unconfirmed 2 -> 1, all others preserved).
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import laspy
import numpy as np
from core.contracts.frv import SourceIdentity
from core.contracts.ground_classification import (
    GroundDecision,
    map_derivative_class,
)

CLASSIFICATION_FLAG_BITS = 0xE0  # bits 5-7 hold synthetic/key_point/withheld


def _read_decision_bytes(decisions_dir: Path, source_point_count: int) -> np.ndarray:
    """Read the ordered decision chunk bytes into one uint8 array."""
    chunks = sorted(decisions_dir.glob("chunk_*.bin"))
    if not chunks:
        raise ValueError("no decision chunk files found")
    parts: list[np.ndarray] = []
    total = 0
    for path in chunks:
        data = np.frombuffer(path.read_bytes(), dtype=np.uint8)
        parts.append(data)
        total += len(data)
    if total != source_point_count:
        raise ValueError(
            f"decision chunks contain {total} rows but source has {source_point_count}"
        )
    return np.concatenate(parts)


def _vlr_semantic_key(vlr: Any) -> tuple[str, str, int, str, bytes]:
    """Canonical VLR preservation key (design section 13)."""
    if hasattr(vlr, "record_data"):
        payload = bytes(vlr.record_data)
    else:
        record_data_bytes = getattr(vlr, "record_data_bytes", None)
        if callable(record_data_bytes):
            payload = bytes(record_data_bytes())
        else:
            payload = bytes(record_data_bytes or b"")
    return (
        type(vlr).__name__,
        str(vlr.user_id),
        int(vlr.record_id),
        str(vlr.description or ""),
        payload,
    )


def verify_vlr_semantic_preservation(source_reader: Any, output_reader: Any) -> None:
    """Compare the ordered non-CRS VLR sequences by canonical semantic key."""
    source_vlrs = [_vlr_semantic_key(vlr) for vlr in source_reader.header.vlrs]
    output_vlrs = [_vlr_semantic_key(vlr) for vlr in output_reader.header.vlrs]
    if source_vlrs != output_vlrs:
        raise ValueError(
            "classified LAS VLR sequence does not round-trip under canonical comparison"
        )


def _assert_no_evlr(reader: Any) -> None:
    if getattr(reader.header, "evlrs", None):
        raise ValueError("EVLR is unsupported in V1 and must be rejected")


def write_classified_las(
    source_path: Path | str,
    decisions_dir: Path | str,
    output_path: Path | str,
    *,
    source_point_count: int,
    chunk_size: int = 250_000,
) -> dict[str, Any]:
    """Stream-copy raw records, mutating only the logical base classification.

    Returns header/flag/field verification metadata.
    """
    source = Path(source_path)
    output = Path(output_path)
    decisions = _read_decision_bytes(Path(decisions_dir), source_point_count)
    output.parent.mkdir(parents=True, exist_ok=True)

    mapping_counts = {"to_2": 0, "to_1": 0, "preserved": 0}
    with laspy.open(source, mode="r") as reader:
        _assert_no_evlr(reader)
        source_header = reader.header
        if int(source_header.point_count) != source_point_count:
            raise ValueError("source point count does not match the decision domain")
        with laspy.open(output, mode="w", header=source_header, do_compress=False) as writer:
            offset = 0
            for point_record in reader.chunk_iterator(chunk_size):
                count = len(point_record)
                base = np.asarray(point_record.classification, dtype=np.uint8)
                slice_decisions = decisions[offset : offset + count]
                mapped = np.empty(count, dtype=np.uint8)
                for index in range(count):
                    original = int(base[index])
                    decision = GroundDecision(int(slice_decisions[index]))
                    mapped[index] = map_derivative_class(original, decision)
                    if decision == GroundDecision.CONFIRMED_GROUND:
                        mapping_counts["to_2"] += 1
                    elif original == 2:
                        mapping_counts["to_1"] += 1
                    else:
                        mapping_counts["preserved"] += 1
                point_record.classification = mapped.astype(np.uint8)
                writer.write_points(point_record)
                offset += count
        output_reader = laspy.open(output, mode="r")
        verify_vlr_semantic_preservation(reader, output_reader)
        _assert_no_evlr(output_reader)
        output_reader.close()

    sha = hashlib.sha256(output.read_bytes()).hexdigest()
    return {
        "output_sha256": sha,
        "output_size_bytes": int(output.stat().st_size),
        "output_point_count": source_point_count,
        "classification_mapping_counts": mapping_counts,
    }


def _class_flag_arrays(point_format: Any, chunk: Any) -> dict[str, np.ndarray]:
    names = (
        "synthetic",
        "key_point",
        "withheld",
        "overlap",
        "scanner_channel",
        "scan_direction_flag",
        "edge_of_flight_line",
    )
    result: dict[str, np.ndarray] = {}
    dimensions = set(str(name) for name in point_format.dimension_names)
    for name in names:
        if name in dimensions:
            result[name] = np.asarray(getattr(chunk, name)).copy()
    return result


def verify_classified_las(
    source_path: Path | str,
    output_path: Path | str,
    decisions_dir: Path | str,
    *,
    source_point_count: int,
    chunk_size: int = 250_000,
) -> dict[str, Any]:
    """Lockstep verify source/derivative preservation and expected classification."""
    source = Path(source_path)
    output = Path(output_path)
    decisions = _read_decision_bytes(Path(decisions_dir), source_point_count)
    with laspy.open(source, mode="r") as source_reader:
        with laspy.open(output, mode="r") as output_reader:
            verify_vlr_semantic_preservation(source_reader, output_reader)
            source_format = source_reader.header.point_format
            output_format = output_reader.header.point_format
            if int(source_reader.header.point_count) != int(output_reader.header.point_count):
                raise ValueError("derivative point count differs from source")
            if tuple(source_reader.header.scales) != tuple(output_reader.header.scales):
                raise ValueError("derivative scales differ from source")
            if tuple(source_reader.header.offsets) != tuple(output_reader.header.offsets):
                raise ValueError("derivative offsets differ from source")
            if int(source_format.id) != int(output_format.id):
                raise ValueError("derivative point format differs from source")
            source_extra = tuple(str(name) for name in source_format.extra_dimension_names)
            output_extra = tuple(str(name) for name in output_format.extra_dimension_names)
            if source_extra != output_extra:
                raise ValueError("derivative Extra Bytes differ from source")

            offset = 0
            for source_chunk, output_chunk in zip(
                source_reader.chunk_iterator(chunk_size),
                output_reader.chunk_iterator(chunk_size),
                strict=False,
            ):
                count = len(source_chunk)
                if len(output_chunk) != count:
                    raise ValueError("derivative chunk size differs from source")
                src_xyz = np.stack(
                    [
                        np.asarray(source_chunk.x),
                        np.asarray(source_chunk.y),
                        np.asarray(source_chunk.z),
                    ],
                    axis=1,
                )
                out_xyz = np.stack(
                    [
                        np.asarray(output_chunk.x),
                        np.asarray(output_chunk.y),
                        np.asarray(output_chunk.z),
                    ],
                    axis=1,
                )
                if not np.array_equal(
                    np.rint(src_xyz).astype(np.int64), np.rint(out_xyz).astype(np.int64)
                ):
                    raise ValueError("derivative raw XYZ differs from source")
                src_base = np.asarray(source_chunk.classification, dtype=np.uint8)
                out_base = np.asarray(output_chunk.classification, dtype=np.uint8)
                slice_decisions = decisions[offset : offset + count]
                expected = np.empty(count, dtype=np.uint8)
                for index in range(count):
                    expected[index] = map_derivative_class(
                        int(src_base[index]), GroundDecision(int(slice_decisions[index]))
                    )
                if not np.array_equal(out_base, expected):
                    raise ValueError("derivative base classification does not match the mapping")
                # All non-classification standard fields must match.
                for dimension in source_format.dimension_names:
                    name = str(dimension)
                    if name in ("classification", "x", "y", "z"):
                        continue
                    src_values = np.asarray(getattr(source_chunk, name))
                    out_values = np.asarray(getattr(output_chunk, name))
                    if not np.array_equal(src_values, out_values):
                        raise ValueError(f"derivative field differs from source: {name}")
                for name in source_extra:
                    src_values = np.asarray(getattr(source_chunk, name))
                    out_values = np.asarray(getattr(output_chunk, name))
                    if not np.array_equal(src_values, out_values):
                        raise ValueError(f"derivative extra field differs from source: {name}")
                for name, src_values in _class_flag_arrays(source_format, source_chunk).items():
                    out_values = np.asarray(getattr(output_chunk, name))
                    if not np.array_equal(src_values, out_values):
                        raise ValueError(f"derivative classification flag differs: {name}")
                offset += count
    sha = hashlib.sha256(output.read_bytes()).hexdigest()
    return {"verified": True, "output_sha256": sha}


def build_classified_source_identity(
    *,
    classified_path: Path | str,
    raw_identity: SourceIdentity,
    decision_digest: str,
    point_count: int,
    bounds_min_xyz: tuple[float, float, float],
    bounds_max_xyz: tuple[float, float, float],
) -> SourceIdentity:
    """Build the new SourceIdentity for the classified derivative.

    The classified source ID is ``<raw-source-id>--fgc-v1--<first-16-hex-of-decision-digest>``
    (design section 3).  The identity carries a new SHA, new content/record
    digests, and never reuses the raw identity.
    """
    import core.contracts.frv as frv

    path = str(Path(classified_path))
    source_id = f"{raw_identity.source_id}--fgc-v1--{decision_digest[:16]}"
    file_sha = hashlib.sha256(Path(classified_path).read_bytes()).hexdigest()
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
        "run_id": f"fgc-{decision_digest[:16]}",
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
        "source_point_count": point_count,
        "source_bounds_min_xyz": list(bounds_min_xyz),
        "source_bounds_max_xyz": list(bounds_max_xyz),
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
