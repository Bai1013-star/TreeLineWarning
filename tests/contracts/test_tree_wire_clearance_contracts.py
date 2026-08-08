"""Contract tests for TREE_WIRE_DISTANCE_ALGORITHM_V1 clearance objects."""

from __future__ import annotations

from pathlib import Path

import pytest
from core.contracts.tree_wire_clearance import (
    TREE_WIRE_CLEARANCE_SCHEMA_VERSION,
    TREE_WIRE_DISTANCE_ALGORITHM_V1,
    TreeWireClearanceConfig,
    TreeWireClearanceQualityFlag,
    TreeWireClearanceRecord,
    TreeWireClearanceRecordStatus,
    TreeWireClearanceResult,
    TreeWireClearanceStatus,
    default_tree_wire_clearance_config,
)


def _record(tree_id: int = 0, **overrides) -> TreeWireClearanceRecord:
    fields = {
        "tree_id": tree_id,
        "status": TreeWireClearanceRecordStatus.VALID,
        "nearest_wire_id": 1,
        "minimum_clearance_m": 0.25,
        "nearest_tree_original_index": 101,
        "nearest_tree_xyz_local_m": (2.0, 3.0, 4.0),
        "nearest_wire_xyz_local_m": (2.0, 3.0, 4.2),
        "w1_segment_index": 0,
        "polyline_edge_index": 2,
        "edge_interpolation_t": 0.5,
        "tree_point_count": 191,
        "candidate_edge_count": 50,
        "evaluated_edge_count": 12,
        "quality_flags": 0,
        "nearest_wire_status": 0,
        "nearest_segment_quality_flags": 0,
    }
    fields.update(overrides)
    return TreeWireClearanceRecord(**fields)


def _with_result_digest(result: TreeWireClearanceResult) -> TreeWireClearanceResult:
    clone = TreeWireClearanceResult.__new__(TreeWireClearanceResult)
    for name in TreeWireClearanceResult.__dataclass_fields__:
        if name == "result_digest":
            object.__setattr__(clone, name, result.computed_result_digest())
        else:
            object.__setattr__(clone, name, getattr(result, name))
    return clone


def _result(**overrides) -> TreeWireClearanceResult:
    config = default_tree_wire_clearance_config()
    fields = {
        "schema_version": TREE_WIRE_CLEARANCE_SCHEMA_VERSION,
        "algorithm_version": TREE_WIRE_DISTANCE_ALGORITHM_V1,
        "status": TreeWireClearanceStatus.VALID,
        "v4_result_digest": "a" * 64,
        "v4_source_id": "scene-1",
        "v4_source_identity_record_digest": "b" * 64,
        "v4_origin_context": (0.0, 0.0, 0.0),
        "a4_source_id": "scene-1",
        "a4_manifest_source_id": "scene-1",
        "a4_origin_xyz": (0.0, 0.0, 0.0),
        "w1_npz_sha256": "c" * 64,
        "w1_json_sha256": "d" * 64,
        "a4_npz_sha256": "e" * 64,
        "a4_json_sha256": "f" * 64,
        "config_digest": config.config_digest,
        "records": (_record(),),
        "result_digest": "0" * 64,
    }
    fields.update(overrides)
    return _with_result_digest(TreeWireClearanceResult(**fields))


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def test_default_config_is_valid_and_round_trips() -> None:
    config = default_tree_wire_clearance_config()
    assert config.bvh_leaf_cap == 16
    assert config.tie_tolerance_m == 1e-9
    assert config.schema_version == "tree-wire-clearance-config-v1"
    assert config.algorithm_version == TREE_WIRE_DISTANCE_ALGORITHM_V1
    assert config.computed_digest() == config.config_digest
    assert TreeWireClearanceConfig.from_dict(config.to_dict()) == config


def test_config_overrides_are_bound() -> None:
    config = default_tree_wire_clearance_config(bvh_leaf_cap=8, tie_tolerance_m=1e-6)
    assert config.bvh_leaf_cap == 8
    assert config.tie_tolerance_m == 1e-6
    assert TreeWireClearanceConfig.from_dict(config.to_dict()) == config


@pytest.mark.parametrize(
    "overrides",
    [
        {"schema_version": "wrong"},
        {"algorithm_version": "wrong"},
        {"bvh_leaf_cap": 0},
        {"bvh_leaf_cap": -3},
        {"tie_tolerance_m": 0.0},
        {"tie_tolerance_m": -1.0},
        {"tie_tolerance_m": float("nan")},
        {"config_digest": "0" * 64},
    ],
)
def test_config_rejects_invalid_values(overrides) -> None:
    with pytest.raises((ValueError, TypeError)):
        default_tree_wire_clearance_config(**overrides)


def test_config_from_dict_rejects_unknown_fields() -> None:
    config = default_tree_wire_clearance_config()
    payload = dict(config.to_dict())
    payload["extra"] = True
    with pytest.raises(ValueError):
        TreeWireClearanceConfig.from_dict(payload)


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


def test_valid_record_round_trips() -> None:
    record = _record()
    clone = TreeWireClearanceRecord.from_dict(record.to_dict())
    assert clone == record


def test_invalid_no_points_record_round_trips() -> None:
    record = _record(
        status=TreeWireClearanceRecordStatus.INVALID_TREE_NO_VALID_POINT,
        nearest_wire_id=None,
        minimum_clearance_m=None,
        nearest_tree_original_index=None,
        nearest_tree_xyz_local_m=None,
        nearest_wire_xyz_local_m=None,
        w1_segment_index=None,
        polyline_edge_index=None,
        edge_interpolation_t=None,
        nearest_wire_status=None,
        nearest_segment_quality_flags=None,
    )
    clone = TreeWireClearanceRecord.from_dict(record.to_dict())
    assert clone == record
    assert clone.status is TreeWireClearanceRecordStatus.INVALID_TREE_NO_VALID_POINT


def test_no_distance_ready_edge_record_round_trips() -> None:
    record = _record(
        status=TreeWireClearanceRecordStatus.NO_DISTANCE_READY_EDGE,
        nearest_wire_id=None,
        minimum_clearance_m=None,
        nearest_tree_original_index=None,
        nearest_tree_xyz_local_m=None,
        nearest_wire_xyz_local_m=None,
        w1_segment_index=None,
        polyline_edge_index=None,
        edge_interpolation_t=None,
        nearest_wire_status=None,
        nearest_segment_quality_flags=None,
    )
    clone = TreeWireClearanceRecord.from_dict(record.to_dict())
    assert clone == record


@pytest.mark.parametrize(
    "overrides",
    [
        {"tree_id": -1},
        {"tree_point_count": -1},
        {"candidate_edge_count": -1},
        {"evaluated_edge_count": -1},
        {"minimum_clearance_m": -0.1},
        {"minimum_clearance_m": float("nan")},
        {"edge_interpolation_t": -0.01},
        {"edge_interpolation_t": 1.01},
        {"quality_flags": -1},
    ],
)
def test_valid_record_rejects_invalid_fields(overrides) -> None:
    with pytest.raises((ValueError, TypeError)):
        _record(**overrides)


def test_valid_record_requires_geometry() -> None:
    with pytest.raises(ValueError):
        _record(nearest_wire_id=None)
    with pytest.raises(ValueError):
        _record(minimum_clearance_m=None)
    with pytest.raises(ValueError):
        _record(nearest_tree_xyz_local_m=None)
    with pytest.raises(ValueError):
        _record(edge_interpolation_t=None)


def test_non_valid_record_rejects_geometry() -> None:
    with pytest.raises(ValueError):
        _record(
            status=TreeWireClearanceRecordStatus.INVALID_TREE_NO_VALID_POINT,
            nearest_wire_id=1,
            minimum_clearance_m=None,
            nearest_tree_original_index=None,
            nearest_tree_xyz_local_m=None,
            nearest_wire_xyz_local_m=None,
            w1_segment_index=None,
            polyline_edge_index=None,
            edge_interpolation_t=None,
            nearest_wire_status=None,
            nearest_segment_quality_flags=None,
        )


def test_record_accepts_quality_flags() -> None:
    record = _record(
        quality_flags=int(
            TreeWireClearanceQualityFlag.NEAREST_WIRE_NOT_CONFIRMED
            | TreeWireClearanceQualityFlag.NEAREST_SEGMENT_FALLBACK_OR_REVIEW
        )
    )
    assert record.quality_flags & int(TreeWireClearanceQualityFlag.NEAREST_WIRE_NOT_CONFIRMED)


def test_record_from_dict_rejects_unknown_fields() -> None:
    payload = dict(_record().to_dict())
    payload["extra"] = True
    with pytest.raises(ValueError):
        TreeWireClearanceRecord.from_dict(payload)


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


def test_result_round_trips_and_digest_is_deterministic() -> None:
    result = _result()
    clone = TreeWireClearanceResult.from_dict(result.to_dict())
    assert clone == result
    assert clone.computed_result_digest() == result.result_digest


def test_result_digest_changes_when_records_change() -> None:
    first = _result()
    second = _result(records=(_record(minimum_clearance_m=0.26),))
    assert first.result_digest != second.result_digest


def test_result_digest_changes_when_origin_or_source_change() -> None:
    first = _result()
    with pytest.raises(ValueError):
        _result(v4_source_id="scene-2")
    with pytest.raises(ValueError):
        _result(v4_origin_context=(1.0, 0.0, 0.0))
    different = _result(v4_source_id="scene-2", a4_source_id="scene-2")
    assert different.result_digest != first.result_digest


def test_result_rejects_source_and_origin_mismatch() -> None:
    with pytest.raises(ValueError):
        _result(v4_source_id="scene-2")
    with pytest.raises(ValueError):
        _result(a4_source_id="scene-2")
    with pytest.raises(ValueError):
        _result(v4_origin_context=(0.0, 0.0, 1.0))
    with pytest.raises(ValueError):
        _result(a4_origin_xyz=(1.0, 0.0, 0.0))


def test_result_rejects_bad_digest() -> None:
    result = _result()
    with pytest.raises(ValueError):
        TreeWireClearanceResult.from_dict(
            {**result.to_dict(), "result_digest": "0" * 64}
        )


def test_result_rejects_unknown_fields() -> None:
    payload = dict(_result().to_dict())
    payload["extra"] = True
    with pytest.raises(ValueError):
        TreeWireClearanceResult.from_dict(payload)


def test_result_no_distance_ready_edge_round_trips() -> None:
    result = _result(status=TreeWireClearanceStatus.NO_DISTANCE_READY_EDGE)
    clone = TreeWireClearanceResult.from_dict(result.to_dict())
    assert clone == result
    assert clone.status is TreeWireClearanceStatus.NO_DISTANCE_READY_EDGE


def test_result_rejects_unsupported_schema() -> None:
    with pytest.raises(ValueError):
        _result(schema_version="unsupported")


def test_result_digest_stable_across_json_round_trip(tmp_path: Path) -> None:
    import json

    result = _result(records=(_record(tree_id=0), _record(tree_id=1)))
    path = tmp_path / "clearance.json"
    path.write_text(json.dumps(result.to_dict()), encoding="utf-8")
    reloaded = TreeWireClearanceResult.from_dict(json.loads(path.read_text("utf-8")))
    assert reloaded.result_digest == result.result_digest
