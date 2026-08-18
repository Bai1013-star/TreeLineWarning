# ruff: noqa: E501
"""Deterministic stem-guided, hybrid individual-tree candidate path.

This is intentionally parallel to the current CHM/watershed span tree path.
It accepts its watershed labels as crown-context evidence, detects stems from
multi-slice HAG occupancy, then uses validated stems to split a shared crown.
It never creates persistent identities; reconciliation remains their owner.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

STEM_GUIDED_HYBRID_SCHEMA = "stem-guided-hybrid-tree-v1"


class TreeQualityState(str, Enum):
    CONFIRMED_TREE = "CONFIRMED_TREE"
    CROWN_ONLY_REVIEW = "CROWN_ONLY_REVIEW"
    REJECTED_SMALL_FRAGMENT = "REJECTED_SMALL_FRAGMENT"
    REJECTED_NO_CROWN = "REJECTED_NO_CROWN"
    STEM_CROWN_RECOVERY_REVIEW = "STEM_CROWN_RECOVERY_REVIEW"
    MULTI_STEM_SPLIT_CANDIDATE = "MULTI_STEM_SPLIT_CANDIDATE"
    ATTACHED_FRAGMENT = "ATTACHED_FRAGMENT"


@dataclass(frozen=True, slots=True)
class StemGuidedHybridConfig:
    """Physical, scene-independent controls for the candidate route.

    The values describe a narrow, persistent lower/mid-height column rather
    than an S005-specific count or coordinate.  Counts are supporting evidence
    only; no candidate is rejected merely because it is small.
    """

    slice_start_hag_m: float = 0.5
    slice_stop_hag_m: float = 5.5
    slice_height_m: float = 1.0
    stem_grid_m: float = 0.35
    track_xy_step_m: float = 0.55
    min_occupied_height_bins: int = 3
    min_stem_vertical_extent_m: float = 2.0
    max_vertical_gap_ratio: float = 0.45
    max_xy_drift_m: float = 0.65
    max_cross_section_radius_m: float = 0.85
    min_pca_verticality: float = 0.70
    min_weighted_stem_support: int = 8
    stem_crown_link_radius_m: float = 3.5
    min_crown_vertical_extent_m: float = 1.5
    min_crown_high_points: int = 4
    min_stem_crown_connection_points: int = 2
    attachment_gap_m: float = 0.55
    stem_dedup_radius_m: float = 1.50
    independent_stem_min_separation_m: float = 2.50
    independent_stem_min_confidence: float = 0.78
    min_independent_stem_side_support: int = 4
    crown_area_grid_m: float = 0.50
    min_crown_area_m2: float = 0.50

    def __post_init__(self) -> None:
        positive = (
            "slice_start_hag_m", "slice_stop_hag_m", "slice_height_m", "stem_grid_m",
            "track_xy_step_m", "min_stem_vertical_extent_m", "max_xy_drift_m",
            "max_cross_section_radius_m", "stem_crown_link_radius_m",
            "min_crown_vertical_extent_m", "attachment_gap_m",
            "stem_dedup_radius_m", "independent_stem_min_separation_m",
            "crown_area_grid_m", "min_crown_area_m2",
        )
        if any(getattr(self, name) <= 0.0 for name in positive):
            raise ValueError("stem-guided physical distances must be positive")
        if self.slice_stop_hag_m <= self.slice_start_hag_m:
            raise ValueError("slice_stop_hag_m must be greater than slice_start_hag_m")
        if (
            self.min_occupied_height_bins < 2
            or self.min_weighted_stem_support < 1
            or self.min_independent_stem_side_support < 1
        ):
            raise ValueError("stem support requirements are invalid")
        if not 0.0 <= self.max_vertical_gap_ratio <= 1.0:
            raise ValueError("max_vertical_gap_ratio must be in [0, 1]")
        if not 0.0 <= self.min_pca_verticality <= 1.0:
            raise ValueError("min_pca_verticality must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class StemEvidence:
    stem_seed_id: int
    xy_local_m: tuple[float, float]
    stem_vertical_extent_m: float
    occupied_height_bins: int
    vertical_gap_ratio: float
    pca_verticality: float
    cross_section_radius_m: float
    point_support: int
    xy_drift_m: float
    local_vegetation_connectivity: float
    stem_confidence: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "stem_seed_id": self.stem_seed_id,
            "xy_local_m": list(self.xy_local_m),
            "stem_vertical_extent_m": self.stem_vertical_extent_m,
            "occupied_height_bins": self.occupied_height_bins,
            "vertical_gap_ratio": self.vertical_gap_ratio,
            "pca_verticality": self.pca_verticality,
            "cross_section_radius_m": self.cross_section_radius_m,
            "point_support": self.point_support,
            "xy_drift_m": self.xy_drift_m,
            "local_vegetation_connectivity": self.local_vegetation_connectivity,
            "stem_confidence": self.stem_confidence,
        }


@dataclass(frozen=True, slots=True)
class HybridTreeCandidate:
    tree_id: int
    quality_state: TreeQualityState
    member_rows: np.ndarray
    stem_seed_ids: tuple[int, ...]
    source_crown_ids: tuple[int, ...]
    apex_xy_local_m: tuple[float, float]
    trunk_evidence: bool
    crown_evidence: bool
    vertical_structure: bool
    stem_crown_connectivity: bool
    membership_support: int
    multi_stem_count: int
    structural_metrics: dict[str, float | int] | None = None
    quality_reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "tree_id": self.tree_id,
            "quality_state": self.quality_state.value,
            "stem_seed_ids": list(self.stem_seed_ids),
            "source_crown_ids": list(self.source_crown_ids),
            "apex_xy_local_m": list(self.apex_xy_local_m),
            "TRUNK_EVIDENCE": self.trunk_evidence,
            "CROWN_EVIDENCE": self.crown_evidence,
            "VERTICAL_STRUCTURE": self.vertical_structure,
            "STEM_CROWN_CONNECTIVITY": self.stem_crown_connectivity,
            "MEMBERSHIP_SUPPORT": self.membership_support,
            "MULTI_STEM_COUNT": self.multi_stem_count,
            "structural_evidence": dict(self.structural_metrics or {}),
            "quality_reasons": list(self.quality_reasons),
        }


@dataclass(frozen=True, slots=True)
class StemGuidedHybridResult:
    fine_tree_id: np.ndarray
    candidates: tuple[HybridTreeCandidate, ...]
    validated_stems: tuple[StemEvidence, ...]
    diagnostics: dict[str, Any]


def _slice_components(
    xyz: np.ndarray, hag: np.ndarray, cfg: StemGuidedHybridConfig
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Build deterministic connected lower/mid-height XY tracks.

    Nodes are occupied XY cells in separate HAG slices.  Adjacent-slice nodes
    connect only within a physically bounded XY step, allowing one missing
    level without relying on a single fixed height slice.
    """

    valid = np.isfinite(hag) & (hag >= cfg.slice_start_hag_m) & (hag < cfg.slice_stop_hag_m)
    rows = np.flatnonzero(valid)
    if not len(rows):
        return []
    bins = np.floor((hag[rows] - cfg.slice_start_hag_m) / cfg.slice_height_m).astype(np.int32)
    cells = np.floor(xyz[rows, :2] / cfg.stem_grid_m).astype(np.int32)
    node_keys = np.column_stack((bins, cells))
    unique, inverse = np.unique(node_keys, axis=0, return_inverse=True)
    node_rows = [rows[np.flatnonzero(inverse == idx)] for idx in range(len(unique))]
    parent = np.arange(len(unique), dtype=np.int32)

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = int(parent[value])
        return value

    def union(left: int, right: int) -> None:
        left, right = find(left), find(right)
        if left != right:
            parent[max(left, right)] = min(left, right)

    # A dictionary makes neighbour lookup independent of input row order.
    lookup = {tuple(int(v) for v in key): idx for idx, key in enumerate(unique)}
    max_cell_step = int(np.ceil(cfg.track_xy_step_m / cfg.stem_grid_m))
    for idx, (height_bin, x_cell, y_cell) in enumerate(unique):
        for next_bin in (int(height_bin) + 1, int(height_bin) + 2):
            for dx in range(-max_cell_step, max_cell_step + 1):
                for dy in range(-max_cell_step, max_cell_step + 1):
                    if (dx * cfg.stem_grid_m) ** 2 + (dy * cfg.stem_grid_m) ** 2 > cfg.track_xy_step_m**2:
                        continue
                    other = lookup.get((next_bin, int(x_cell) + dx, int(y_cell) + dy))
                    if other is not None:
                        union(idx, other)
    groups: dict[int, list[int]] = {}
    for idx in range(len(unique)):
        groups.setdefault(find(idx), []).append(idx)
    return [(unique[np.asarray(nodes)], np.concatenate([node_rows[i] for i in nodes])) for nodes in groups.values()]


def _vertical_columns(
    xyz: np.ndarray, hag: np.ndarray, cfg: StemGuidedHybridConfig
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return narrow fixed-XY multi-slice columns before spatial NMS.

    A connected lower-canopy carpet may span many cells, so it must not itself
    become one enormous "stem" component.  Persistence is first evaluated at
    a fixed, narrow XY column; nearby accepted columns are collapsed later.
    """

    valid = np.isfinite(hag) & (hag >= cfg.slice_start_hag_m) & (hag < cfg.slice_stop_hag_m)
    rows = np.flatnonzero(valid)
    if not len(rows):
        return []
    bins = np.floor((hag[rows] - cfg.slice_start_hag_m) / cfg.slice_height_m).astype(np.int32)
    cells = np.floor(xyz[rows, :2] / cfg.stem_grid_m).astype(np.int32)
    unique_cells, cell_inverse = np.unique(cells, axis=0, return_inverse=True)
    columns: list[tuple[np.ndarray, np.ndarray]] = []
    for cell_id in range(len(unique_cells)):
        column_rows = rows[cell_inverse == cell_id]
        column_bins = bins[cell_inverse == cell_id]
        columns.append((np.unique(column_bins), column_rows))
    return columns


def detect_validated_stems(
    fine_xyz_local_m: np.ndarray,
    fine_hag_m: np.ndarray,
    *,
    raw_count: np.ndarray | None = None,
    config: StemGuidedHybridConfig | None = None,
) -> tuple[StemEvidence, ...]:
    """Detect and validate narrow multi-slice stem hypotheses from fine reps."""

    cfg = config or StemGuidedHybridConfig()
    xyz = np.asarray(fine_xyz_local_m, dtype=np.float64)
    hag = np.asarray(fine_hag_m, dtype=np.float64)
    weights = np.ones(len(xyz), dtype=np.int64) if raw_count is None else np.asarray(raw_count, dtype=np.int64)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or hag.shape != (len(xyz),) or weights.shape != (len(xyz),):
        raise ValueError("fine XYZ, HAG, and raw_count must be aligned")
    accepted: list[tuple[tuple[float, float, int], StemEvidence]] = []
    for occupied, rows in _vertical_columns(xyz, hag, cfg):
        if len(occupied) < cfg.min_occupied_height_bins:
            continue
        row_bins = np.floor((hag[rows] - cfg.slice_start_hag_m) / cfg.slice_height_m).astype(np.int32)
        centers = np.array([np.mean(xyz[rows][row_bins == b, :2], axis=0) for b in occupied])
        center = np.median(centers, axis=0)
        drift = float(np.max(np.linalg.norm(centers - center, axis=1)))
        vertical_extent = float(np.max(hag[rows]) - np.min(hag[rows]))
        expected = int(occupied[-1] - occupied[0] + 1)
        gap_ratio = 1.0 - len(occupied) / expected
        radius = float(np.quantile(np.linalg.norm(xyz[rows, :2] - center, axis=1), 0.90))
        points = xyz[rows]
        if len(points) >= 3:
            covariance = np.cov(points - points.mean(axis=0), rowvar=False)
            values, vectors = np.linalg.eigh(covariance)
            axis = vectors[:, int(np.argmax(values))]
            verticality = float(abs(axis[2]))
        else:
            verticality = 0.0
        support = int(np.sum(weights[rows], dtype=np.int64))
        # Requiring lower and upper support in the track is the local
        # vegetation connectivity evidence; it rejects an isolated vertical row.
        connectivity = float(len(occupied) / max(1, expected))
        valid = (
            vertical_extent >= cfg.min_stem_vertical_extent_m
            and gap_ratio <= cfg.max_vertical_gap_ratio
            and drift <= cfg.max_xy_drift_m
            and radius <= cfg.max_cross_section_radius_m
            and verticality >= cfg.min_pca_verticality
            and support >= cfg.min_weighted_stem_support
        )
        if not valid:
            continue
        confidence = float(np.clip((verticality + connectivity + (1.0 - radius / cfg.max_cross_section_radius_m) + (1.0 - drift / cfg.max_xy_drift_m)) / 4.0, 0.0, 1.0))
        evidence = StemEvidence(
            stem_seed_id=-1,
            xy_local_m=(float(center[0]), float(center[1])),
            stem_vertical_extent_m=vertical_extent,
            occupied_height_bins=int(len(occupied)),
            vertical_gap_ratio=float(gap_ratio),
            pca_verticality=verticality,
            cross_section_radius_m=radius,
            point_support=support,
            xy_drift_m=drift,
            local_vegetation_connectivity=connectivity,
            stem_confidence=confidence,
        )
        accepted.append(((round(center[0], 6), round(center[1], 6), int(rows.min())), evidence))
    # One physical shaft can occupy a few 0.35m columns.  Deterministic NMS
    # preserves its strongest evidence and gives stable seed ids.
    accepted.sort(key=lambda pair: (-pair[1].stem_confidence, -pair[1].point_support, pair[0]))
    selected: list[tuple[tuple[float, float, int], StemEvidence]] = []
    for item in accepted:
        if all(np.linalg.norm(np.asarray(item[1].xy_local_m) - np.asarray(other[1].xy_local_m)) > cfg.max_cross_section_radius_m for other in selected):
            selected.append(item)
    accepted = selected
    accepted.sort(key=lambda pair: pair[0])
    return tuple(
        StemEvidence(stem_seed_id=index, **{name: getattr(item, name) for name in StemEvidence.__dataclass_fields__ if name != "stem_seed_id"})
        for index, (_, item) in enumerate(accepted)
    )


def _crown_evidence(rows: np.ndarray, hag: np.ndarray, cfg: StemGuidedHybridConfig) -> tuple[bool, bool]:
    if not len(rows):
        return False, False
    finite = hag[rows][np.isfinite(hag[rows])]
    if not len(finite):
        return False, False
    vertical = float(np.max(finite) - np.min(finite)) >= cfg.min_crown_vertical_extent_m
    crown = vertical and int(np.count_nonzero(finite >= np.quantile(finite, 0.65))) >= cfg.min_crown_high_points
    return crown, vertical


def _candidate_for_rows(
    tree_id: int, rows: np.ndarray, xyz: np.ndarray, hag: np.ndarray, seeds: tuple[StemEvidence, ...], source_ids: tuple[int, ...], cfg: StemGuidedHybridConfig
) -> HybridTreeCandidate:
    crown, vertical = _crown_evidence(rows, hag, cfg)
    # A bare validated column is a stem-only object, not a crown merely
    # because its own vertical points span height.  Crown evidence originates
    # in an admitted CHM/watershed region.
    if not source_ids:
        crown = False
    stem = len(seeds) == 1
    if seeds and len(rows):
        distances = np.linalg.norm(xyz[rows, :2] - np.asarray(seeds[0].xy_local_m), axis=1)
        # A path from the measured stem envelope into the lower crown is
        # required; this is not an XY-nearest-stem-only test.
        connected = int(np.count_nonzero((distances <= cfg.stem_crown_link_radius_m) & (hag[rows] >= 0.75 * seeds[0].stem_vertical_extent_m))) >= cfg.min_stem_crown_connection_points
    else:
        connected = False
    if stem and crown and vertical and connected:
        state = TreeQualityState.CONFIRMED_TREE
    elif stem and not crown:
        state = TreeQualityState.REJECTED_NO_CROWN
    elif not stem and crown:
        state = TreeQualityState.CROWN_ONLY_REVIEW
    else:
        state = TreeQualityState.REJECTED_SMALL_FRAGMENT
    high = rows[int(np.nanargmax(hag[rows]))] if len(rows) else 0
    return HybridTreeCandidate(
        tree_id=tree_id, quality_state=state, member_rows=np.sort(rows),
        stem_seed_ids=tuple(seed.stem_seed_id for seed in seeds), source_crown_ids=source_ids,
        apex_xy_local_m=(float(xyz[high, 0]), float(xyz[high, 1])), trunk_evidence=stem,
        crown_evidence=crown, vertical_structure=vertical, stem_crown_connectivity=connected,
        membership_support=int(len(rows)), multi_stem_count=len(seeds),
    )


def _crown_metrics(
    rows: np.ndarray, xyz: np.ndarray, hag: np.ndarray, weights: np.ndarray, cfg: StemGuidedHybridConfig
) -> dict[str, float | int]:
    xy = xyz[rows, :2]
    finite_hag = hag[rows][np.isfinite(hag[rows])]
    cells = np.unique(np.floor(xy / cfg.crown_area_grid_m).astype(np.int32), axis=0)
    extent = np.ptp(xy, axis=0) if len(xy) else np.zeros(2)
    return {
        "crown_area": float(len(cells) * cfg.crown_area_grid_m**2),
        "crown_horizontal_extent": float(np.linalg.norm(extent)),
        "vertical_extent": float(np.ptp(finite_hag)) if len(finite_hag) else 0.0,
        "fine_point_count": int(len(rows)),
        "full_res_point_count": int(np.sum(weights[rows], dtype=np.int64)),
    }


def _deduplicate_crown_stems(
    raw_stems: tuple[StemEvidence, ...], cfg: StemGuidedHybridConfig
) -> tuple[StemEvidence, ...]:
    """Merge nearby stem columns into one crown-conditioned hypothesis."""

    if not raw_stems:
        return ()
    xy = np.asarray([stem.xy_local_m for stem in raw_stems], dtype=np.float64)
    neighbours = cKDTree(xy).query_ball_point(xy, r=cfg.stem_dedup_radius_m)
    seen: set[int] = set()
    output: list[StemEvidence] = []
    for start in range(len(raw_stems)):
        if start in seen:
            continue
        stack, group = [start], []
        seen.add(start)
        while stack:
            current = stack.pop()
            group.append(current)
            for other in neighbours[current]:
                if other not in seen:
                    seen.add(other)
                    stack.append(other)
        # A single representative carries the actual measured trajectory;
        # aggregation only removes duplicate columns from one local cylinder.
        best = min(
            group,
            key=lambda index: (
                -raw_stems[index].stem_confidence,
                -raw_stems[index].point_support,
                raw_stems[index].xy_local_m,
                raw_stems[index].stem_seed_id,
            ),
        )
        output.append(raw_stems[best])
    output.sort(key=lambda stem: (stem.xy_local_m, stem.stem_seed_id))
    return tuple(
        replace(stem, stem_seed_id=index) for index, stem in enumerate(output)
    )


def _candidate_from_crown(
    tree_id: int,
    rows: np.ndarray,
    xyz: np.ndarray,
    hag: np.ndarray,
    weights: np.ndarray,
    stems: tuple[StemEvidence, ...],
    crown_id: int,
    cfg: StemGuidedHybridConfig,
) -> HybridTreeCandidate:
    crown, vertical = _crown_evidence(rows, hag, cfg)
    metrics = _crown_metrics(rows, xyz, hag, weights, cfg)
    crown = crown and float(metrics["crown_area"]) >= cfg.min_crown_area_m2
    connected = False
    if stems:
        seed_xy = np.asarray(stems[0].xy_local_m)
        distance = np.linalg.norm(xyz[rows, :2] - seed_xy, axis=1)
        connected = int(
            np.count_nonzero(
                (distance <= cfg.stem_crown_link_radius_m)
                & (hag[rows] >= 0.75 * stems[0].stem_vertical_extent_m)
            )
        ) >= cfg.min_stem_crown_connection_points
    if stems and crown and vertical and connected:
        state = TreeQualityState.CONFIRMED_TREE
    elif crown and vertical:
        state = TreeQualityState.CROWN_ONLY_REVIEW
    else:
        state = TreeQualityState.REJECTED_SMALL_FRAGMENT
    high = rows[int(np.nanargmax(hag[rows]))]
    return HybridTreeCandidate(
        tree_id=tree_id,
        quality_state=state,
        member_rows=np.sort(rows),
        stem_seed_ids=tuple(stem.stem_seed_id for stem in stems),
        source_crown_ids=(crown_id,),
        apex_xy_local_m=(float(xyz[high, 0]), float(xyz[high, 1])),
        trunk_evidence=bool(stems),
        crown_evidence=crown,
        vertical_structure=vertical,
        stem_crown_connectivity=connected,
        membership_support=int(len(rows)),
        multi_stem_count=len(stems),
        structural_metrics={
            **metrics,
            "stem_candidate_count_raw": 0,
            "stem_hypothesis_count_deduplicated": len(stems),
            "validated_independent_stem_count": len(stems),
            "crown_evidence_score": float(crown),
            "stem_evidence_score": float(stems[0].stem_confidence) if stems else 0.0,
            "vertical_structure_score": float(vertical),
            "stem_crown_connectivity_score": float(connected),
            "CHM_peak_count": 1,
            "saddle_evidence": 0,
        },
        quality_reasons=(
            "crown_generated_candidate",
            "stem_validated" if stems else "stem_unavailable",
            state.value.lower(),
        ),
    )


def segment_stem_guided_hybrid(
    fine_xyz_local_m: np.ndarray,
    fine_hag_m: np.ndarray,
    watershed_tree_id: np.ndarray,
    *,
    raw_count: np.ndarray | None = None,
    config: StemGuidedHybridConfig | None = None,
    crown_peak_count_by_id: dict[int, int] | None = None,
) -> StemGuidedHybridResult:
    """Refine CHM/watershed crown candidates using stem evidence.

    Crowns are the sole normal source of tree candidates.  Stem columns may
    validate a crown, or cause a split only after crown-conditioned spatial
    deduplication and independent-support checks.  A stem never creates a
    normal tree by itself.
    """

    cfg = config or StemGuidedHybridConfig()
    started = time.perf_counter()
    xyz = np.asarray(fine_xyz_local_m, dtype=np.float64)
    hag = np.asarray(fine_hag_m, dtype=np.float64)
    baseline = np.asarray(watershed_tree_id, dtype=np.int32)
    weights = np.ones(len(xyz), dtype=np.int64) if raw_count is None else np.asarray(raw_count, dtype=np.int64)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or hag.shape != (len(xyz),) or baseline.shape != (len(xyz),):
        raise ValueError("fine XYZ, HAG, and watershed labels must be aligned")
    raw_stems = detect_validated_stems(xyz, hag, raw_count=weights, config=cfg)
    stem_detection_seconds = time.perf_counter() - started
    crown_ids = sorted(int(value) for value in np.unique(baseline) if value >= 0)
    crown_trees = {crown_id: cKDTree(xyz[baseline == crown_id, :2]) for crown_id in crown_ids}
    owned_raw: dict[int, list[StemEvidence]] = {crown_id: [] for crown_id in crown_ids}
    false_stem_count = 0
    for stem in raw_stems:
        choices = [(float(tree.query(stem.xy_local_m, k=1)[0]), crown_id) for crown_id, tree in crown_trees.items()]
        if not choices:
            false_stem_count += 1
            continue
        distance, crown_id = min(choices, key=lambda value: (value[0], value[1]))
        if distance <= cfg.stem_crown_link_radius_m:
            owned_raw[crown_id].append(stem)
        else:
            false_stem_count += 1
    hypotheses_by_crown = {
        crown_id: _deduplicate_crown_stems(tuple(stems), cfg)
        for crown_id, stems in owned_raw.items()
    }
    next_stem_id = 0
    for crown_id in crown_ids:
        hypotheses = hypotheses_by_crown[crown_id]
        hypotheses_by_crown[crown_id] = tuple(
            replace(stem, stem_seed_id=next_stem_id + index)
            for index, stem in enumerate(hypotheses)
        )
        next_stem_id += len(hypotheses)
    dedup_seconds = time.perf_counter() - started - stem_detection_seconds
    crown_validation_started = time.perf_counter()
    provisional: list[HybridTreeCandidate] = []
    split_count = 0
    split_crown_ids: set[int] = set()
    for crown_id in crown_ids:
        rows = np.flatnonzero(baseline == crown_id)
        hypotheses = hypotheses_by_crown[crown_id]
        strong = tuple(
            stem for stem in hypotheses if stem.stem_confidence >= cfg.independent_stem_min_confidence
        )
        independent: list[StemEvidence] = []
        for stem in strong:
            if all(
                np.linalg.norm(np.asarray(stem.xy_local_m) - np.asarray(other.xy_local_m))
                >= cfg.independent_stem_min_separation_m
                for other in independent
            ):
                side_support = int(
                    np.count_nonzero(
                        np.linalg.norm(xyz[rows, :2] - np.asarray(stem.xy_local_m), axis=1)
                        <= cfg.stem_crown_link_radius_m
                    )
                )
                if side_support >= cfg.min_independent_stem_side_support:
                    independent.append(stem)
        chm_peak_count = int((crown_peak_count_by_id or {}).get(crown_id, 1))
        # A split requires both independent stems and independent crown
        # structure. Missing CHM support always means keep the one crown.
        if len(independent) < 2 or chm_peak_count < 2:
            primary = independent[:1] or strong[:1]
            provisional.append(
                _candidate_from_crown(
                    len(provisional), rows, xyz, hag, weights, tuple(primary), crown_id, cfg
                )
            )
            continue
        # Strict multi-source split: known watershed membership is the crown
        # continuity constraint; height progression prevents low detached rows
        # from being claimed purely by nearest XY distance.
        seed_xy = np.asarray([stem.xy_local_m for stem in independent])
        xy_distance = np.linalg.norm(xyz[rows, None, :2] - seed_xy[None, :, :], axis=2)
        stem_top = np.asarray([stem.stem_vertical_extent_m + cfg.slice_start_hag_m for stem in independent])
        cost = xy_distance + 0.15 * np.maximum(0.0, stem_top[None, :] - hag[rows, None])
        assignment = np.argmin(cost, axis=1)
        children = [
            _candidate_from_crown(
                len(provisional) + index,
                rows[assignment == index],
                xyz,
                hag,
                weights,
                (stem,),
                crown_id,
                cfg,
            )
            for index, stem in enumerate(independent)
        ]
        # A structurally weak child invalidates the split and preserves the
        # original crown rather than publishing a tiny artificial tree.
        if all(child.quality_state == TreeQualityState.CONFIRMED_TREE for child in children):
            provisional.extend(children)
            split_count += 1
            split_crown_ids.add(crown_id)
        else:
            provisional.append(
                _candidate_from_crown(
                    len(provisional), rows, xyz, hag, weights, tuple(independent[:1]), crown_id, cfg
                )
            )
    owned_raw_ids = {
        stem.stem_seed_id for values in owned_raw.values() for stem in values
    }
    for stem in raw_stems:
        if stem.stem_seed_id in owned_raw_ids:
            continue
        nearby = np.flatnonzero(
            np.linalg.norm(xyz[:, :2] - np.asarray(stem.xy_local_m), axis=1)
            <= cfg.max_cross_section_radius_m
        )
        provisional.append(
            _candidate_for_rows(
                len(provisional), nearby, xyz, hag, (stem,), (), cfg
            )
        )
    for index, candidate in enumerate(provisional):
        if not candidate.source_crown_ids:
            continue
        crown_id = candidate.source_crown_ids[0]
        metrics = dict(candidate.structural_metrics or {})
        metrics.update(
            {
                "stem_candidate_count_raw": len(owned_raw[crown_id]),
                "stem_hypothesis_count_deduplicated": len(hypotheses_by_crown[crown_id]),
                "validated_independent_stem_count": candidate.multi_stem_count,
                "CHM_peak_count": int((crown_peak_count_by_id or {}).get(crown_id, 1)),
                "saddle_evidence": int(crown_id in split_crown_ids),
            }
        )
        reasons = candidate.quality_reasons + (
            ("multi_stem_split",) if crown_id in split_crown_ids else ()
        )
        provisional[index] = replace(candidate, structural_metrics=metrics, quality_reasons=reasons)
    attachment_target: dict[int, int] = {}
    confirmed = [item for item in provisional if item.quality_state == TreeQualityState.CONFIRMED_TREE]
    for index, item in enumerate(provisional):
        if item.quality_state != TreeQualityState.REJECTED_SMALL_FRAGMENT or not len(item.member_rows):
            continue
        best: tuple[float, int] | None = None
        for target in confirmed:
            distance = float(cKDTree(xyz[target.member_rows]).query(xyz[item.member_rows], k=1)[0].min())
            candidate_key = (distance, target.tree_id)
            if best is None or candidate_key < best:
                best = candidate_key
        if best is not None and best[0] <= cfg.attachment_gap_m:
            attachment_target[index] = best[1]
            provisional[index] = replace(item, quality_state=TreeQualityState.ATTACHED_FRAGMENT)
    labels = np.full(len(xyz), -1, dtype=np.int32)
    candidates: list[HybridTreeCandidate] = []
    for index, candidate in enumerate(provisional):
        if candidate.quality_state in {TreeQualityState.CONFIRMED_TREE, TreeQualityState.CROWN_ONLY_REVIEW}:
            labels[candidate.member_rows] = index
        elif candidate.quality_state == TreeQualityState.ATTACHED_FRAGMENT:
            labels[candidate.member_rows] = attachment_target[index]
        candidates.append(replace(candidate, tree_id=index))
    states = {state.value: sum(item.quality_state == state for item in candidates) for state in TreeQualityState}
    return StemGuidedHybridResult(
        fine_tree_id=labels,
        candidates=tuple(candidates),
        validated_stems=raw_stems,
        diagnostics={
            "schema": STEM_GUIDED_HYBRID_SCHEMA,
            "raw_stem_candidate_count": len(raw_stems),
            "deduplicated_stem_hypothesis_count": int(sum(len(value) for value in hypotheses_by_crown.values())),
            "validated_independent_stem_count": int(sum(candidate.multi_stem_count for candidate in candidates if candidate.trunk_evidence)),
            "false_stem_count": false_stem_count,
            "multi_stem_split_count": split_count,
            "state_counts": states,
            "membership_duplicates": 0,
            "runtime_seconds": {
                "stem_candidate_detection": stem_detection_seconds,
                "stem_deduplication": dedup_seconds,
                "crown_validation_and_split": time.perf_counter() - crown_validation_started,
                "full_hybrid": time.perf_counter() - started,
            },
        },
    )


__all__ = [
    "STEM_GUIDED_HYBRID_SCHEMA", "TreeQualityState", "StemGuidedHybridConfig", "StemEvidence",
    "HybridTreeCandidate", "StemGuidedHybridResult", "detect_validated_stems", "segment_stem_guided_hybrid",
]
