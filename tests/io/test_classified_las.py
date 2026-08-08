"""Classification-only LAS writer, flag/VLR preservation, and verifier tests.

Authority: docs/full-resolution-ground-classification-design.md section 13 and
the frozen Bundle C plan test gate (point formats, RGB/GPS/intensity/returns/
normals/Extra Bytes, synthetic/key-point/withheld/overlap and scan flags, VLR
preservation, unsupported metadata failure, stale class 2).
"""

from __future__ import annotations

from pathlib import Path

import laspy
import numpy as np
import pytest
from algorithms.io.classified_las import (
    build_classified_source_identity,
    verify_classified_las,
    write_classified_las,
)
from core.contracts.ground_classification import (
    CONFIRMED_GROUND,
    NON_GROUND,
    REVIEW_GROUND,
    GroundDecision,
    map_derivative_class,
)

from tests.ground._identity import make_source_identity


def _write_source(
    path: Path,
    *,
    point_format: int = 7,
    classification: np.ndarray,
    extra_names: tuple[str, ...] = (),
    add_vlr: bool = True,
) -> None:
    header = laspy.LasHeader(point_format=point_format, version="1.4")
    header.scales = np.array([0.01, 0.01, 0.01])
    header.offsets = np.array([100.0, 200.0, 0.0])
    for name in extra_names:
        header.add_extra_dim(laspy.ExtraBytesParams(name=name, type=np.float64))
    if add_vlr:
        vlr = laspy.VLR(
            user_id="TLW", record_id=7, description="test-vlr", record_data=b"\x01\x02\x03"
        )
        header.vlrs.append(vlr)
    las = laspy.LasData(header)
    count = len(classification)
    idx = np.arange(count)
    las.x = 100.0 + idx.astype(np.float64)
    las.y = 200.0 + idx.astype(np.float64)
    las.z = idx.astype(np.float64)
    las.classification = classification.astype(np.uint8)
    dims = set(las.point_format.dimension_names)
    if "intensity" in dims:
        las.intensity = (idx + 10).astype(np.uint16)
    if "gps_time" in dims:
        las.gps_time = (idx + 0.5).astype(np.float64)
    if "red" in dims:
        las.red = (idx + 1).astype(np.uint16)
    if "green" in dims:
        las.green = (idx + 2).astype(np.uint16)
    if "blue" in dims:
        las.blue = (idx + 3).astype(np.uint16)
    if "return_number" in dims:
        las.return_number = (idx % 3 + 1).astype(np.uint8)
    if "number_of_returns" in dims:
        las.number_of_returns = np.full(count, 3, dtype=np.uint8)
    if "point_source_id" in dims:
        las.point_source_id = (idx + 20).astype(np.uint16)
    if "NormalX" in dims:
        las.NormalX = (idx + 0.0).astype(np.float32)
    for name in extra_names:
        setattr(las, name, (idx + 1).astype(np.float64))
    las.write(path)


def _decisions(count: int) -> np.ndarray:
    values = np.empty(count, dtype=np.uint8)
    for i in range(count):
        if i % 3 == 0:
            values[i] = CONFIRMED_GROUND
        elif i % 3 == 1:
            values[i] = REVIEW_GROUND
        else:
            values[i] = NON_GROUND
    return values


def _write_decision_chunks(tmp_path: Path, decisions: np.ndarray) -> Path:
    directory = tmp_path / "decisions"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "chunk_000000.bin").write_bytes(decisions.tobytes())
    return directory


def _expected_map(source_class: np.ndarray, decisions: np.ndarray) -> np.ndarray:
    result = np.empty(len(source_class), dtype=np.uint8)
    for i in range(len(source_class)):
        result[i] = map_derivative_class(int(source_class[i]), GroundDecision(int(decisions[i])))
    return result


def test_write_classified_las_preserves_fields_and_flags(tmp_path: Path) -> None:
    count = 9
    source_class = np.arange(count, dtype=np.uint8) % 6
    decisions = _decisions(count)
    source = tmp_path / "src.las"
    _write_source(source, classification=source_class, extra_names=("NormalX", "MyExtra"))
    output = tmp_path / "out.fgc-v1.las"
    result = write_classified_las(
        source, _write_decision_chunks(tmp_path, decisions), output, source_point_count=count
    )
    assert output.exists()
    assert result["output_point_count"] == count
    verify_classified_las(
        source, output, _write_decision_chunks(tmp_path, decisions), source_point_count=count
    )

    with laspy.open(output) as reader:
        pts = reader.read()
        expected = _expected_map(source_class, decisions)
        assert np.array_equal(np.asarray(pts.classification), expected)
        # Stale class 2 -> 1, confirmed -> 2.
        for i in range(count):
            if decisions[i] == CONFIRMED_GROUND:
                assert int(pts.classification[i]) == 2
            elif int(source_class[i]) == 2:
                assert int(pts.classification[i]) == 1
        # Non-classification fields preserved.
        assert np.array_equal(np.asarray(pts.intensity), np.arange(count) + 10)
        assert np.array_equal(np.asarray(pts.NormalX), (np.arange(count) + 1).astype(np.float32))
        assert np.array_equal(np.asarray(pts.MyExtra), np.arange(count) + 1)


def test_write_classified_las_preserves_classification_flags(tmp_path: Path) -> None:
    count = 4
    source_class = np.array([2, 2, 5, 5], dtype=np.uint8)
    decisions = np.array(
        [CONFIRMED_GROUND, REVIEW_GROUND, NON_GROUND, CONFIRMED_GROUND], dtype=np.uint8
    )
    source = tmp_path / "src_flags.las"
    _write_source(source, classification=source_class)
    with laspy.open(source, mode="r") as reader:
        pts = reader.read()
    pts.synthetic = np.array([1, 0, 1, 0], dtype=bool)
    pts.key_point = np.array([0, 1, 1, 0], dtype=bool)
    pts.withheld = np.array([1, 1, 0, 0], dtype=bool)
    pts.overlap = np.array([1, 0, 0, 1], dtype=bool)
    pts.write(source)
    output = tmp_path / "out_flags.las"
    write_classified_las(
        source, _write_decision_chunks(tmp_path, decisions), output, source_point_count=count
    )
    verify_classified_las(
        source, output, _write_decision_chunks(tmp_path, decisions), source_point_count=count
    )
    with laspy.open(output) as reader:
        pts = reader.read()
        assert np.array_equal(np.asarray(pts.synthetic, dtype=np.uint8), np.array([1, 0, 1, 0]))
        assert np.array_equal(np.asarray(pts.key_point, dtype=np.uint8), np.array([0, 1, 1, 0]))
        assert np.array_equal(np.asarray(pts.withheld, dtype=np.uint8), np.array([1, 1, 0, 0]))
        assert np.array_equal(np.asarray(pts.overlap, dtype=np.uint8), np.array([1, 0, 0, 1]))
        assert int(pts.classification[0]) == 2  # confirmed
        assert int(pts.classification[1]) == 1  # stale 2 -> 1, flags preserved


def test_write_classified_las_preserves_vlr_sequence(tmp_path: Path) -> None:
    count = 4
    source = tmp_path / "src_vlr.las"
    _write_source(source, classification=np.full(count, 1, dtype=np.uint8), add_vlr=True)
    decisions = _decisions(count)
    output = tmp_path / "out_vlr.las"
    write_classified_las(
        source, _write_decision_chunks(tmp_path, decisions), output, source_point_count=count
    )
    verify_classified_las(
        source, output, _write_decision_chunks(tmp_path, decisions), source_point_count=count
    )
    from algorithms.io.classified_las import _vlr_semantic_key

    with laspy.open(source) as reader:
        src_vlrs = [_vlr_semantic_key(v) for v in reader.header.vlrs]
    with laspy.open(output) as reader:
        out_vlrs = [_vlr_semantic_key(v) for v in reader.header.vlrs]
    assert src_vlrs == out_vlrs


def _rewrite_classification(path: Path, new_classification: np.ndarray) -> None:
    with laspy.open(path, mode="r") as reader:
        las = reader.read()
    las.classification = new_classification.astype(np.uint8)
    las.write(path)


def test_verifier_detects_classification_mismatch(tmp_path: Path) -> None:
    count = 4
    source = tmp_path / "src.las"
    _write_source(source, classification=np.full(count, 2, dtype=np.uint8))
    decisions = np.array(
        [CONFIRMED_GROUND, CONFIRMED_GROUND, CONFIRMED_GROUND, CONFIRMED_GROUND], dtype=np.uint8
    )
    output = tmp_path / "out.las"
    write_classified_las(
        source, _write_decision_chunks(tmp_path, decisions), output, source_point_count=count
    )
    # Tamper with the output classification so the lockstep verifier must fail.
    _rewrite_classification(output, np.full(count, 5, dtype=np.uint8))
    with pytest.raises(ValueError, match="classification"):
        verify_classified_las(
            source, output, _write_decision_chunks(tmp_path, decisions), source_point_count=count
        )


def test_verifier_detects_scalar_drift(tmp_path: Path) -> None:
    count = 4
    source = tmp_path / "src.las"
    _write_source(source, classification=np.full(count, 1, dtype=np.uint8))
    decisions = _decisions(count)
    output = tmp_path / "out.las"
    write_classified_las(
        source, _write_decision_chunks(tmp_path, decisions), output, source_point_count=count
    )
    # Rewrite output with a different scale -> verifier must fail.
    with laspy.open(output, mode="r") as reader:
        header = reader.header
        las = reader.read()
    header.scales = np.array([0.02, 0.02, 0.02])
    las.header.scales = np.array([0.02, 0.02, 0.02])
    las.write(output)
    with pytest.raises(ValueError, match="scales"):
        verify_classified_las(
            source, output, _write_decision_chunks(tmp_path, decisions), source_point_count=count
        )


def test_mapping_counts_from_writer(tmp_path: Path) -> None:
    count = 6
    source_class = np.array([2, 2, 5, 5, 0, 1], dtype=np.uint8)
    decisions = np.array(
        [CONFIRMED_GROUND, REVIEW_GROUND, NON_GROUND, CONFIRMED_GROUND, REVIEW_GROUND, NON_GROUND],
        dtype=np.uint8,
    )
    source = tmp_path / "src.las"
    _write_source(source, classification=source_class)
    output = tmp_path / "out.las"
    result = write_classified_las(
        source, _write_decision_chunks(tmp_path, decisions), output, source_point_count=count
    )
    counts = result["classification_mapping_counts"]
    assert counts["to_2"] == 2  # confirmed rows
    assert counts["to_1"] == 1  # stale class 2 not confirmed
    assert counts["preserved"] == 3  # rows 2,3,5


def test_build_classified_source_identity_has_new_ids(tmp_path: Path) -> None:
    raw = make_source_identity(source_id="raw-scan")
    path = tmp_path / "classified.las"
    path.write_bytes(b"fake-classified-bytes")
    classified = build_classified_source_identity(
        classified_path=path,
        raw_identity=raw,
        decision_digest="0123456789abcdef" + "0" * 48,
        point_count=7,
        bounds_min_xyz=(0.0, 0.0, 0.0),
        bounds_max_xyz=(10.0, 10.0, 5.0),
    )
    assert classified.source_id.startswith("raw-scan--fgc-v1--0123456789abcdef")
    assert classified.source_id != raw.source_id
    assert classified.source_file_sha256 != raw.source_file_sha256
    assert classified.source_identity_record_digest != raw.source_identity_record_digest
