from __future__ import annotations

import hashlib

import numpy as np
import pytest
from core.contracts import (
    VegetationAssociationConfig,
    VegetationAssociationResult,
    VegetationAssociationStatistics,
    VegetationAssociationStatus,
    validate_association_arrays,
)


def _arrays() -> dict[str, np.ndarray]:
    return {
        "pair_cluster_ids": np.array([[1, 4]], np.int32),
        "pair_status": np.array([VegetationAssociationStatus.ABSTAINED], np.uint8),
        "pair_confidence": np.array([0.0], np.float32),
        "pair_positive_evidence": np.zeros((1, 10), np.float32),
        "pair_blocking_evidence": np.zeros((1, 10), np.float32),
        "pair_reason_flags": np.zeros(1, np.uint32),
        "pair_distance_summary_m": np.zeros((1, 5), np.float32),
        "pair_scale_density_summary": np.zeros((1, 4), np.float32),
        "pair_hag_summary": np.zeros((1, 4), np.float32),
        "pair_low_support_counts": np.zeros((1, 6), np.int32),
        "pair_provenance_flags": np.zeros(1, np.uint32),
        "pair_decision_input_completeness": np.zeros(1, np.float32),
    }


def test_exact_status_mapping_and_config_roundtrip() -> None:
    assert [int(value) for value in VegetationAssociationStatus] == [0, 1, 2, 3]
    config = VegetationAssociationConfig()
    assert VegetationAssociationConfig.from_dict(config.to_dict()) == config
    with pytest.raises(ValueError, match="confidence"):
        VegetationAssociationConfig(possible_confidence_threshold=0.9)


def test_pair_arrays_require_exact_shapes_and_sorted_unique_pairs() -> None:
    arrays = _arrays()
    assert validate_association_arrays(arrays) == 1
    arrays["pair_cluster_ids"] = np.array([[4, 1]], np.int32)
    with pytest.raises(ValueError, match="ordered"):
        validate_association_arrays(arrays)
    arrays = _arrays()
    arrays["pair_evidence_completeness"] = arrays.pop("pair_decision_input_completeness")
    with pytest.raises(ValueError, match="exactly"):
        validate_association_arrays(arrays)


def test_result_rejects_wrong_schema_and_inconsistent_counts() -> None:
    digest = hashlib.sha256(b"v2a").hexdigest()
    stats = VegetationAssociationStatistics(2, 1, 1, 0, 0, 0, 0.01)
    result = VegetationAssociationResult(
        "vegetation-associations-v1",
        "v2a-review-level-vegetation-association-v1",
        "synthetic",
        "manifest",
        (0.0, 0.0, 0.0),
        "synthetic_vegetation_associations.npz",
        digest,
        1,
        VegetationAssociationConfig(source_id="synthetic"),
        {"v2_npz": {"sha256": digest}},
        {
            name: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for name, value in _arrays().items()
        },
        stats,
        created_at="2026-01-01T00:00:00+00:00",
    )
    payload = result.to_dict()
    assert VegetationAssociationResult.from_dict(payload) == result
    payload["schema_version"] = "vegetation-associations-v0"
    with pytest.raises(ValueError, match="unsupported"):
        VegetationAssociationResult.from_dict(payload)
