from __future__ import annotations

import json
from pathlib import Path

from algorithms.spans.project_adapter import (
    SpanAnalysisInput,
    prepare_source_shared,
)
from core.config.fast_baseline import canonical_digest
from core.stage_b.role_preparation import RoleIndexPreparation, _representative_voxel_size


def _input(source: Path) -> SpanAnalysisInput:
    digest = "a" * 64
    return SpanAnalysisInput(
        project_id="project-1",
        source_identity={
            "source_id": "source-1",
            "path": str(source),
            "resolved_access_path": str(source),
        },
        group_id="G001",
        span_id="S001",
        from_tower_id="T001",
        to_tower_id="T002",
        from_tower_revision=1,
        to_tower_revision=1,
        from_endpoint_geometry={},
        to_endpoint_geometry={},
        span_geometry={},
        science_configuration_digest=digest,
        risk_configuration_digest="b" * 64,
        validated_source_preparation_references=(),
    )


def test_existing_ground_preparation_builds_and_binds_missing_role_indexes(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "source.las"
    source.write_bytes(b"source")
    input_data = _input(source)
    source_digest = canonical_digest(dict(input_data.source_identity))
    root = (
        tmp_path
        / "artifacts"
        / "project-runs"
        / "source-shared"
        / source_digest
    )
    root.mkdir(parents=True)
    manifest = root / "preparation.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "treeline-project-source-shared-preparation-v1",
                "source_binding_digest": source_digest,
                "configuration_digest": input_data.science_configuration_digest,
                "coarse_ground_reference": "ground.npy",
                "ground_preparation": {
                    "cache_hit": True,
                    "source_full_pass_count": 1,
                },
                "tower_membership_used_for_vegetation": False,
            }
        ),
        encoding="utf-8",
    )
    calls: list[Path] = []

    def fake_roles(project_root, preparation_root, source_identity, **kwargs):
        assert project_root == tmp_path
        assert dict(source_identity) == dict(input_data.source_identity)
        assert kwargs["span_inputs"] == (input_data,)
        calls.append(preparation_root)
        return RoleIndexPreparation(
            wire_reference=(
                f"artifacts/project-runs/source-shared/{source_digest}/"
                "wire-role-tile-index"
            ),
            vegetation_reference=(
                f"artifacts/project-runs/source-shared/{source_digest}/"
                "vegetation-role-tile-index"
            ),
            source_point_count=123,
            source_full_pass_count=5,
            cache_hit=False,
            details={"status": "PREPARED_AND_VALIDATED_ROLE_INDEXES"},
        )

    monkeypatch.setattr(
        "core.stage_b.role_preparation.ensure_role_tile_indexes", fake_roles
    )
    result = prepare_source_shared(tmp_path, input_data)
    payload = json.loads(manifest.read_text(encoding="utf-8"))

    assert calls == [root]
    assert payload["wire_role_tile_index_reference"].endswith(
        "/wire-role-tile-index"
    )
    assert payload["vegetation_role_tile_index_reference"].endswith(
        "/vegetation-role-tile-index"
    )
    assert payload["source_point_count"] == 123
    assert result.telemetry["source_prep_count"] == 1
    assert result.telemetry["ground_prep_count"] == 0
    assert result.telemetry["source_full_pass_count"] == 5


def test_role_preparation_keeps_vegetation_high_recall_and_wire_filtered() -> None:
    source = Path("core/stage_b/role_preparation.py").read_text(encoding="utf-8")

    assert "np.arange(working.representative_count" in source
    assert "wire_admission_mask |= rescued" in source
    assert "rescue_span_aligned_wire_evidence" in source
    assert "STRICT_PATH_A_UNION_SPAN_ALIGNED_PATH_B_RESCUE" in source
    assert "HIGH_RECALL_ALL_SOURCE_NO_BROAD_WIRE_EXCLUSION" in source
    assert "fast_baseline_pipeline" not in source
    assert _representative_voxel_size(105_055_350, 87_000.0) == 0.25
