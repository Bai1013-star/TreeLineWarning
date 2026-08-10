"""Shared coarse-representative evidence arrays for pole and wire detection.

The pole-first baseline runs geometric evidence on the reviewed coarse
(0.25--0.75 m) routing representatives, never on the full 105M-point source.
This module loads the PDAL Ground/HAG output once and computes PDAL
Dimensionality covariance features (Linearity/Planarity/Scattering/
Verticality) for the non-ground representatives, scattering the per-point
feature values back onto the full representative array.  It is a FAST
NON_CERTIFIED research helper; it never claims Formal authority.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from core.config.fast_baseline import FastBaselineConfig

GROUND_EVIDENCE_SCHEMA = "fast-baseline-ground-evidence-v1"


class PoleEvidenceError(RuntimeError):
    """Closed failure raised by the coarse evidence loader."""


def local_pca_features(
    xyz_local_m: np.ndarray, *, neighbor_count: int, chunk_size: int = 100_000
) -> dict[str, np.ndarray]:
    """Per-point Dimensionality features plus the primary-direction z-component.

    Neighbours are queried in one batched KD-tree call and every point's 3x3
    covariance is eigen-decomposed with a batched ``numpy.linalg.eigh``.
    ``main_direction_z`` (the absolute z-component of the largest-eigenvalue
    eigenvector) is a robust "is this structure vertical/horizontal" indicator
    even for thin lines where the smallest-eigenvector normal is degenerate.
    """

    from scipy.spatial import cKDTree

    xyz = np.asarray(xyz_local_m, dtype=np.float64)
    count = len(xyz)
    empty = {
        name: np.empty(0, dtype=np.float32)
        for name in (
            "linearity",
            "planarity",
            "scattering",
            "verticality",
            "main_direction_z",
        )
    }
    if count == 0:
        return empty
    neighbor_count = min(max(3, int(neighbor_count)), count)
    if count == 1:
        return {
            "linearity": np.zeros(1, dtype=np.float32),
            "planarity": np.zeros(1, dtype=np.float32),
            "scattering": np.ones(1, dtype=np.float32),
            "verticality": np.zeros(1, dtype=np.float32),
            "main_direction_z": np.zeros(1, dtype=np.float32),
        }
    tree = cKDTree(xyz)
    linearity = np.zeros(count, dtype=np.float32)
    planarity = np.zeros(count, dtype=np.float32)
    scattering = np.zeros(count, dtype=np.float32)
    verticality = np.zeros(count, dtype=np.float32)
    main_direction_z = np.zeros(count, dtype=np.float32)
    for start in range(0, count, chunk_size):
        stop = min(count, start + chunk_size)
        neighbors = tree.query(xyz[start:stop], k=neighbor_count)[1]
        if neighbors.ndim == 1:
            neighbors = neighbors[:, None]
        centered = xyz[neighbors] - xyz[start:stop][:, None, :]
        covariance = np.einsum("nki,nkj->nij", centered, centered) / max(
            1, neighbor_count - 1
        )
        values, vectors = np.linalg.eigh(covariance)
        values = np.maximum(values, 0.0)
        totals = values.sum(axis=1)
        nonzero = totals > 1e-12
        block_linearity = np.zeros(stop - start, dtype=np.float32)
        block_planarity = np.zeros(stop - start, dtype=np.float32)
        block_scattering = np.zeros(stop - start, dtype=np.float32)
        block_verticality = np.zeros(stop - start, dtype=np.float32)
        block_main_direction_z = np.zeros(stop - start, dtype=np.float32)
        if np.any(nonzero):
            block_linearity[nonzero] = (values[nonzero, 2] - values[nonzero, 1]) / totals[nonzero]
            block_planarity[nonzero] = (values[nonzero, 1] - values[nonzero, 0]) / totals[nonzero]
            block_scattering[nonzero] = values[nonzero, 0] / totals[nonzero]
            normals = vectors[nonzero, :, 0]
            block_verticality[nonzero] = 1.0 - np.abs(normals[:, 2])
            block_main_direction_z[nonzero] = np.abs(vectors[nonzero, :, 2][:, 2])
        linearity[start:stop] = block_linearity
        planarity[start:stop] = block_planarity
        scattering[start:stop] = block_scattering
        verticality[start:stop] = block_verticality
        main_direction_z[start:stop] = block_main_direction_z
    return {
        "linearity": linearity,
        "planarity": planarity,
        "scattering": scattering,
        "verticality": verticality,
        "main_direction_z": main_direction_z,
    }


@dataclass(frozen=True, slots=True)
class GroundEvidence:
    """Aligned coarse-representative arrays plus per-point geometry features."""

    xyz_absolute_m: np.ndarray  # float64 (N,3)
    xyz_local_m: np.ndarray  # float64 (N,3)
    hag_m: np.ndarray  # float32 (N,)
    classification: np.ndarray  # uint8 (N,)
    representative_id: np.ndarray  # uint32 (N,)
    source_original_index: np.ndarray  # int64 (N,)
    raw_count: np.ndarray  # uint32 (N,)
    linearity: np.ndarray  # float32 (N,)
    planarity: np.ndarray  # float32 (N,)
    scattering: np.ndarray  # float32 (N,)
    verticality: np.ndarray  # float32 (N,)
    main_direction_z: np.ndarray  # float32 (N,)
    count: int
    origin_xyz: tuple[float, float, float]
    ground_class_2_count: int
    non_ground_count: int
    feature_runtime_seconds: float

    def non_ground_mask(self) -> np.ndarray:
        return self.classification != 2


def load_ground_evidence(
    ground_las_path: Path | str,
    config: FastBaselineConfig,
    *,
    origin_xyz: tuple[float, float, float],
) -> GroundEvidence:
    """Load the Ground/HAG representative LAS and compute covariance features.

    Features are computed only for non-ground representatives; ground class-2
    representatives receive zero-filled features (they are never pole or wire
    evidence).  Array alignment is strictly validated before returning.
    """

    import laspy

    path = Path(ground_las_path).resolve()
    if not path.is_file():
        raise PoleEvidenceError(f"POLE_EVIDENCE_GROUND_MISSING: {path}")
    las = laspy.read(path)
    xyz_absolute = np.column_stack((las.x, las.y, las.z)).astype(np.float64, copy=False)
    xyz_local = xyz_absolute - np.asarray(origin_xyz, dtype=np.float64)
    hag = np.asarray(las.HeightAboveGround, dtype=np.float32)
    classification = np.asarray(las.classification, dtype=np.uint8)
    representative_id = np.asarray(las.representative_id, dtype=np.uint32)
    source_original_index = np.asarray(las.source_original_index, dtype=np.int64)
    raw_count = np.asarray(las.raw_count, dtype=np.uint32)
    count = len(xyz_local)
    aligned = (hag, classification, representative_id, source_original_index, raw_count)
    if any(values.shape != (count,) for values in aligned):
        raise PoleEvidenceError("POLE_EVIDENCE_ALIGNMENT_INVALID")

    non_ground = classification != 2
    linearity = np.zeros(count, dtype=np.float32)
    planarity = np.zeros(count, dtype=np.float32)
    scattering = np.zeros(count, dtype=np.float32)
    verticality = np.zeros(count, dtype=np.float32)
    main_direction_z = np.zeros(count, dtype=np.float32)
    feature_runtime = 0.0
    eligible_rows = np.flatnonzero(non_ground)
    if len(eligible_rows) >= max(8, config.wire_pca_neighbor_count):
        started = time.perf_counter()
        features = local_pca_features(
            xyz_local[eligible_rows],
            neighbor_count=config.wire_pca_neighbor_count,
        )
        feature_runtime = time.perf_counter() - started
        linearity[eligible_rows] = features["linearity"]
        planarity[eligible_rows] = features["planarity"]
        scattering[eligible_rows] = features["scattering"]
        verticality[eligible_rows] = features["verticality"]
        main_direction_z[eligible_rows] = features["main_direction_z"]
    return GroundEvidence(
        xyz_absolute_m=xyz_absolute,
        xyz_local_m=xyz_local,
        hag_m=hag,
        classification=classification,
        representative_id=representative_id,
        source_original_index=source_original_index,
        raw_count=raw_count,
        linearity=linearity,
        planarity=planarity,
        scattering=scattering,
        verticality=verticality,
        main_direction_z=main_direction_z,
        count=count,
        origin_xyz=tuple(float(value) for value in origin_xyz),
        ground_class_2_count=int(np.count_nonzero(non_ground == False)),  # noqa: E712
        non_ground_count=int(np.count_nonzero(non_ground)),
        feature_runtime_seconds=feature_runtime,
    )


__all__ = [
    "GROUND_EVIDENCE_SCHEMA",
    "GroundEvidence",
    "PoleEvidenceError",
    "load_ground_evidence",
]
