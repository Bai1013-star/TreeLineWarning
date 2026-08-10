"""V4 local-edge resource scalability tests (maximum_local_edges hotfix).

The ``maximum_local_edges`` guards are implementation safety/resource bounds,
not semantic validity bounds.  Under the 0.10 m coarse-to-fine input the local
voxel / fragment / pair counts legitimately grow while the computation stays
safe, so the guards are resource-aware (projected working memory / work) and
the voxel graph is streamed.  These tests prove:

* exact equivalence between the streamed implementation and a reference
  (full-adjacency) implementation,
* an old-limit-exceeded-but-safe case now completes,
* a genuinely unsafe projected-resource case still fails closed,
* edge determinism and the frozen enumeration order,
* multi-slice trees and highly fragmented adjacent slices complete.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest
from algorithms.vegetation import v4_tree_segmentation as v4s
from core.contracts.v4_tree_segmentation import (
    V4TreeSegmentationError,
    V4TreeSegmentationErrorCode,
)

from tests.vegetation.test_v4_tree_segmentation import make_config, make_handoff


def _voxel(cell_id: int, bin_id: int, xyz: tuple[float, float, float], index: int) -> v4s._Voxel:
    return v4s._Voxel(
        cell_id=cell_id,
        hag_bin_id=bin_id,
        positions=(index,),
        centroid_xyz=xyz,
        density=10.0,
        min_original_index=index,
    )


def _regular_voxel_grid(
    cols: int, rows: int, *, cell_m: float = 0.5
) -> dict[tuple[int, int], v4s._Voxel]:
    """Return a single-HAG-bin voxel grid over ``cols x rows`` cells."""
    voxels: dict[tuple[int, int], v4s._Voxel] = {}
    index = 0
    for row in range(rows):
        for col in range(cols):
            cell_id = row * cols + col
            voxels[(cell_id, 0)] = _voxel(
                cell_id, 0, (col * cell_m, row * cell_m, 1.0), index
            )
            index += 1
    return voxels


def _grid(cols: int, rows: int) -> SimpleNamespace:
    return SimpleNamespace(cols=cols, rows=rows)


def _scales() -> v4s.V4AlgorithmScales:
    return v4s.derive_algorithm_scales(make_config())


# Reference (OLD) full-adjacency behavior for exact-equivalence checks.
def _reference_voxel_adjacency(
    voxels: dict[tuple[int, int], v4s._Voxel],
    grid: SimpleNamespace,
    scales: v4s.V4AlgorithmScales,
) -> dict[tuple[int, int], list[tuple[int, int]]]:
    adjacency: dict[tuple[int, int], list[tuple[int, int]]] = {key: [] for key in voxels}
    for key, voxel in voxels.items():
        cell_id = voxel.cell_id
        col = cell_id % grid.cols
        row = cell_id // grid.cols
        bin_id = voxel.hag_bin_id
        for drow in (-1, 0, 1):
            nrow = row + drow
            if nrow < 0 or nrow >= grid.rows:
                continue
            for dcol in (-1, 0, 1):
                ncol = col + dcol
                if ncol < 0 or ncol >= grid.cols:
                    continue
                for dbin in (-1, 0, 1):
                    if drow == 0 and dcol == 0 and dbin == 0:
                        continue
                    neighbor_key = (nrow * grid.cols + ncol, bin_id + dbin)
                    neighbor = voxels.get(neighbor_key)
                    if neighbor is None:
                        continue
                    if v4s._voxel_adjacent(voxel, neighbor, grid, scales):
                        adjacency[key].append(neighbor_key)
    return adjacency


def _reference_connected_components(
    voxels: dict[tuple[int, int], v4s._Voxel],
    grid: SimpleNamespace,
    scales: v4s.V4AlgorithmScales,
    maximum_local_edges: int | None = None,
) -> list[list[v4s._Voxel]]:
    adjacency = _reference_voxel_adjacency(voxels, grid, scales)
    edge_count = sum(len(neighbors) for neighbors in adjacency.values()) // 2
    if maximum_local_edges is not None and edge_count > maximum_local_edges:
        raise V4TreeSegmentationError(
            code=V4TreeSegmentationErrorCode.RESOURCE_LIMIT,
            field_name="maximum_local_edges",
        )
    keys = sorted(voxels)
    parent = {key: key for key in keys}

    def find(key: tuple[int, int]) -> tuple[int, int]:
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    def union(a: tuple[int, int], b: tuple[int, int]) -> None:
        root_a = find(a)
        root_b = find(b)
        if root_a != root_b:
            if root_a < root_b:
                parent[root_b] = root_a
            else:
                parent[root_a] = root_b

    for key_a in keys:
        for key_b in adjacency[key_a]:
            union(key_a, key_b)
    groups: dict[tuple[int, int], list[v4s._Voxel]] = {}
    for key in keys:
        root = find(key)
        groups.setdefault(root, []).append(voxels[key])
    result = []
    for _root in sorted(groups):
        result.append(sorted(groups[_root], key=lambda voxel: voxel.min_original_index))
    return result


def _component_key(components: list[list[v4s._Voxel]]) -> list[list[int]]:
    return [
        [voxel.min_original_index for voxel in component] for component in components
    ]


def _fragment_pair(hypothesis_id: str, count: int) -> list[v4s._Fragment]:
    """Two adjacent slices, ``count`` fragments each, with unique fragment ids."""
    fragments: list[v4s._Fragment] = []
    for index in range(count):
        fragments.append(
            v4s._Fragment(
                f"frag-{hypothesis_id}-s0-{index:03d}",
                0,
                hypothesis_id,
                (index,),
                f"{index:064d}",
            )
        )
        fragments.append(
            v4s._Fragment(
                f"frag-{hypothesis_id}-s1-{index:03d}",
                1,
                hypothesis_id,
                (count + index,),
                f"{count + index:064d}",
            )
        )
    return fragments


class TestVoxelComponentEquivalence:
    def test_streamed_components_equal_reference(self) -> None:
        voxels = _regular_voxel_grid(20, 15)
        grid = _grid(20, 15)
        scales = _scales()
        reference = _reference_connected_components(voxels, grid, scales)
        streamed = v4s._connected_voxel_components(voxels, grid, scales, 50000)
        assert _component_key(streamed) == _component_key(reference)

    def test_disconnected_components_preserved(self) -> None:
        # Two separated 8 x 8 clusters (rows 0-7 and rows 20-27) must stay
        # separate under the streamed union-find.
        cols = 30
        voxels: dict[tuple[int, int], v4s._Voxel] = {}
        index = 0
        for base_row in (0, 20):
            for row in range(8):
                for col in range(8):
                    cell_id = (base_row + row) * cols + col
                    voxels[(cell_id, 0)] = _voxel(
                        cell_id, 0, (col * 0.5, (base_row + row) * 0.5, 1.0), index
                    )
                    index += 1
        grid = _grid(cols, 30)
        scales = _scales()
        components = v4s._connected_voxel_components(voxels, grid, scales, 50000)
        assert len(components) == 2

    def test_deterministic_repeat_runs(self) -> None:
        voxels = _regular_voxel_grid(17, 13)
        grid = _grid(17, 13)
        scales = _scales()
        first = _component_key(v4s._connected_voxel_components(voxels, grid, scales, 50000))
        second = _component_key(v4s._connected_voxel_components(voxels, grid, scales, 50000))
        assert first == second

    def test_neighbor_enumeration_matches_reference_order(self) -> None:
        voxels = _regular_voxel_grid(9, 7)
        grid = _grid(9, 7)
        scales = _scales()
        reference = _reference_voxel_adjacency(voxels, grid, scales)
        for key in sorted(voxels):
            streamed = v4s._voxel_adjacent_neighbor_keys(voxels, grid, scales, key, voxels[key])
            assert streamed == reference[key]


class TestOldLimitExceededButSafe:
    def test_voxel_component_exceeds_old_limit_but_completes(self) -> None:
        # A 130 x 130 single-bin grid has > 50,000 voxel-adjacency edges: the
        # OLD count guard fails it, the resource-aware implementation completes.
        cols = rows = 130
        voxels = _regular_voxel_grid(cols, rows)
        grid = _grid(cols, rows)
        scales = _scales()
        edge_count = (
            sum(
                len(neighbors)
                for neighbors in _reference_voxel_adjacency(voxels, grid, scales).values()
            )
            // 2
        )
        assert edge_count > 50_000
        with pytest.raises(V4TreeSegmentationError) as excinfo:
            _reference_connected_components(voxels, grid, scales, 50_000)
        assert excinfo.value.code == V4TreeSegmentationErrorCode.RESOURCE_LIMIT
        components = v4s._connected_voxel_components(voxels, grid, scales, 50_000)
        assert _component_key(components) == _component_key(
            _reference_connected_components(voxels, grid, scales)
        )

    def test_fragment_graph_exceeds_old_limit_but_completes(self) -> None:
        # A single hypothesis with 300 + 300 fragments in two adjacent slices is
        # 90,000 pair evaluations (> OLD 50,000) yet remains safe.
        fragments = _fragment_pair("hyp-a", 300)
        points = [(float(i) * 0.02, float(i) * 0.02, 3.0 + float(i) * 0.01) for i in range(600)]
        population = v4s._canonical_population(make_handoff(points))
        scales = _scales()
        adjacency, _continuity, available = v4s._build_adjacent_fragment_graph(
            fragments, population, scales, make_config(maximum_local_edges=50_000)
        )
        assert available == {"hyp-a": True}
        assert len(adjacency) == len(fragments)

    def test_structural_path_multi_slice_tree_completes(self) -> None:
        from tests.vegetation.test_v4_tree_segmentation import make_tree_points

        handoff = make_handoff(make_tree_points(3.0, 3.0))
        result = v4s.segment_v4_trees(handoff, make_config())
        assert len(result.tree_records) >= 1
        assert result.audit.total_points == int(result.original_index.shape[0])


class TestGenuinelyUnsafeFailClosed:
    def test_voxel_component_projected_unsafe_fails_closed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(v4s, "_MAXIMUM_LOCAL_WORKING_BYTES", 1000)
        voxels = _regular_voxel_grid(6, 4)  # 24 voxels * 160 bytes = 3840 > 1000
        grid = _grid(6, 4)
        scales = _scales()
        with pytest.raises(V4TreeSegmentationError) as excinfo:
            v4s._connected_voxel_components(voxels, grid, scales, 50_000)
        assert excinfo.value.code == V4TreeSegmentationErrorCode.RESOURCE_LIMIT

    def test_fragment_graph_projected_unsafe_fails_closed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(v4s, "_MAXIMUM_LOCAL_WORKING_BYTES", 1_000_000)
        # 300 x 300 pairs * 128 bytes = 11.5 MB > 1 MB budget.
        fragments = _fragment_pair("hyp-a", 300)
        points = [(float(i) * 0.02, float(i) * 0.02, 3.0 + float(i) * 0.01) for i in range(600)]
        population = v4s._canonical_population(make_handoff(points))
        scales = _scales()
        with pytest.raises(V4TreeSegmentationError) as excinfo:
            v4s._build_adjacent_fragment_graph(
                fragments, population, scales, make_config(maximum_local_edges=50_000)
            )
        assert excinfo.value.code == V4TreeSegmentationErrorCode.RESOURCE_LIMIT


class TestCrownEndpointConnectivity:
    def test_neighborhood_enumeration_matches_all_pairs(self) -> None:
        trunk = SimpleNamespace(core_hag_max_m=1.5)
        apex = SimpleNamespace(hag_m=2.0, xyz_local_m=np.asarray((5.0, 5.0, 2.0)))
        scales = _scales()
        grid = _grid(20, 20)
        # Build a corridor with upper voxels (bin >= trunk_upper) and apex voxels.
        corridor: dict[tuple[int, int], v4s._Voxel] = {}
        index = 0
        for row in range(6, 15):
            for col in range(6, 15):
                for bin_id in (2, 3, 4):
                    cell_id = row * grid.cols + col
                    corridor[(cell_id, bin_id)] = _voxel(
                        cell_id,
                        bin_id,
                        (float(col) * 0.5, float(row) * 0.5, float(bin_id) * 0.5),
                        index,
                    )
                    index += 1
        trunk_upper_bin = int(math.floor(trunk.core_hag_max_m / scales.btr))
        upper_voxels = [
            voxel for key, voxel in corridor.items() if voxel.hag_bin_id >= trunk_upper_bin
        ]
        apex_bin = int(math.floor(apex.hag_m / scales.btr))
        apex_xy = np.asarray(apex.xyz_local_m[:2])
        apex_voxels = [
            voxel for key, voxel in corridor.items()
            if abs(voxel.hag_bin_id - apex_bin) <= 1
            and math.hypot(voxel.centroid_xyz[0] - apex_xy[0], voxel.centroid_xyz[1] - apex_xy[1])
            <= scales.la
        ]
        assert upper_voxels and apex_voxels
        evaluated = len(upper_voxels) * len(apex_voxels)
        reference_linked = 0
        for upper in upper_voxels:
            for apex_voxel in apex_voxels:
                if v4s._voxel_adjacent(upper, apex_voxel, grid, scales):
                    reference_linked += 1
        linked, available = v4s._crown_endpoint_connectivity(
            trunk, apex, corridor, grid, scales, 50_000
        )
        assert available is True
        assert linked == pytest.approx(reference_linked / evaluated, abs=1e-12)
