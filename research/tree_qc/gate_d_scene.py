"""Gate D shared scene construction.

Every example — GOOD and every defect — starts from the **same** ``BaseScene``:

    Gate C2 approved candidate tree
  + Gate C2 approved surviving context, clipped to the frozen 3 m query

The scene carries two channels and nothing else:

* ``points`` — real publisher geometry in metres;
* ``current_candidate_membership`` — whether the *current* single-tree segmentation
  selected each point.

The frozen 3 m query is the Gate C2 rule verbatim: the candidate XY bbox expanded by
3 m with the candidate's native Z extent. A context tree may contribute only the points
that actually fall inside that box, never its whole cloud.

Corruption changes ``current_candidate_membership`` only. It never adds a point source
outside the candidate's frozen surviving context, never removes real geometry, and never
introduces synthetic coordinates. ``gt_source_id`` is audit-only metadata: it records
the publisher GT origin of each point and is never part of the model input.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

QUERY_PAD_M = 3.0
SCENE_VERSION = 'GATE_D_BASE_SCENE_V1'

# Minimum real points a scene needs before it is worth generating from.
MIN_SCENE_POINTS = 64
MIN_CANDIDATE_POINTS = 32
MIN_CONTEXT_POINTS = 8


class SceneError(RuntimeError):
    pass


@dataclass(frozen=True)
class ScenePoints:
    """Real geometry plus audit-only provenance for one source."""

    source_id: str
    points: np.ndarray          # (N, 3) float32 metres
    is_candidate: bool          # publisher GT: this source is the candidate tree


@dataclass
class BaseScene:
    """The shared scene every class starts from, before membership corruption."""

    candidate_id: str
    partition: str
    site: str
    points: np.ndarray                     # (N, 3) float32 metres, real geometry
    gt_source_id: np.ndarray               # (N,) int32 index into source_ids
    source_ids: tuple[str, ...]
    candidate_mask: np.ndarray             # (N,) bool publisher GT candidate points
    query_box: tuple[float, float, float, float, float, float]
    audit: dict = field(default_factory=dict)

    @property
    def point_count(self) -> int:
        return int(self.points.shape[0])

    @property
    def candidate_points(self) -> int:
        return int(self.candidate_mask.sum())

    @property
    def context_points(self) -> int:
        return int((~self.candidate_mask).sum())

    def source_counts(self) -> dict[str, int]:
        """Points contributed by each source, derived from the scene itself."""
        cached = getattr(self, '_source_counts', None)
        if cached is not None:
            return cached
        counts: dict[str, int] = {}
        for index, source_id in enumerate(self.source_ids):
            count = int((self.gt_source_id == index).sum())
            if count:
                counts[source_id] = count
        object.__setattr__(self, '_source_counts', counts)
        return counts

    def membership_gt(self) -> np.ndarray:
        """The GOOD membership: publisher GT candidate points are selected."""
        return self.candidate_mask.copy()


def frozen_query_box(bbox) -> tuple[float, float, float, float, float, float]:
    """The Gate C2 context query: candidate XY bbox + 3 m, native Z extent."""
    x0, y0, z0, x1, y1, z1 = (float(v) for v in bbox)
    return (x0 - QUERY_PAD_M, y0 - QUERY_PAD_M, z0,
            x1 + QUERY_PAD_M, y1 + QUERY_PAD_M, z1)


def clip_to_query(points: np.ndarray, box) -> np.ndarray:
    """Indices of points inside the frozen query box."""
    points = np.asarray(points)
    x0, y0, z0, x1, y1, z1 = box
    return np.flatnonzero(
        (points[:, 0] >= x0) & (points[:, 0] <= x1)
        & (points[:, 1] >= y0) & (points[:, 1] <= y1)
        & (points[:, 2] >= z0) & (points[:, 2] <= z1))


def build_base_scene(spec_candidate_id: str, partition: str, geometry,
                     provenance) -> BaseScene:
    """Build the shared scene from approved, clipped, real geometry.

    Fails closed if the candidate is not a model-selection tree, if any context source
    is not authorised, or if the clipped scene is too small to be meaningful.
    """
    if geometry.candidate_id != spec_candidate_id:
        raise SceneError(
            f'geometry is for {geometry.candidate_id}, requested {spec_candidate_id}')
    if provenance.partition_of_candidate(spec_candidate_id) != partition:
        raise SceneError(f'{spec_candidate_id} is not a {partition} candidate')
    provenance.assert_not_quarantined(spec_candidate_id, 'candidate')
    provenance.assert_not_boundary_candidate(spec_candidate_id)

    box = frozen_query_box(provenance.tree[spec_candidate_id]['bbox'])
    sources: list[ScenePoints] = []
    candidate_points = np.asarray(geometry.candidate.points, dtype=np.float32)
    if candidate_points.shape[0] == 0:
        raise SceneError(f'{spec_candidate_id} has no candidate geometry')
    sources.append(ScenePoints(source_id=spec_candidate_id, points=candidate_points,
                               is_candidate=True))

    authorised = provenance.authorized_context(spec_candidate_id)
    if tuple(authorised) != tuple(geometry.context_ids):
        raise SceneError(
            f'geometry context {geometry.context_ids} does not match the frozen '
            f'surviving context {authorised}')
    for tree in geometry.context:
        provenance.authorize_donor(spec_candidate_id, tree.canonical_tree_id)
        keep = clip_to_query(tree.points, box)
        if keep.size == 0:
            continue
        sources.append(ScenePoints(source_id=tree.canonical_tree_id,
                                   points=np.asarray(tree.points[keep], dtype=np.float32),
                                   is_candidate=False))

    blocks = [s.points for s in sources]
    points = np.concatenate(blocks, axis=0)
    gt_source_id = np.concatenate(
        [np.full(s.points.shape[0], index, dtype=np.int32)
         for index, s in enumerate(sources)])
    candidate_mask = np.concatenate(
        [np.full(s.points.shape[0], s.is_candidate, dtype=bool) for s in sources])
    source_ids = tuple(s.source_id for s in sources)

    scene = BaseScene(candidate_id=spec_candidate_id, partition=partition,
                      site=geometry.candidate.site, points=points,
                      gt_source_id=gt_source_id, source_ids=source_ids,
                      candidate_mask=candidate_mask, query_box=box)
    if scene.point_count < MIN_SCENE_POINTS:
        raise SceneError(f'{spec_candidate_id} scene has only {scene.point_count} points')
    if scene.candidate_points < MIN_CANDIDATE_POINTS:
        raise SceneError(
            f'{spec_candidate_id} has only {scene.candidate_points} candidate points')
    if scene.context_points < MIN_CONTEXT_POINTS:
        raise SceneError(
            f'{spec_candidate_id} has only {scene.context_points} clipped context '
            'points; a GOOD example without real context would leak the class')
    scene.audit = {
        'scene_version': SCENE_VERSION,
        'query_box': [round(float(v), 6) for v in box],
        'query_pad_m': QUERY_PAD_M,
        'candidate_id': spec_candidate_id,
        'source_ids': list(source_ids),
        'source_counts': scene.source_counts(),
        'source_aliases': {source_id: list(provenance.source_aliases_of.get(source_id, ()))
                           for source_id in source_ids},
        'candidate_points': scene.candidate_points,
        'context_points': scene.context_points,
        'context_sources_used': len(source_ids) - 1,
        'context_sources_authorised': len(authorised),
    }
    return scene


def scene_source_union(scene: BaseScene) -> tuple[str, ...]:
    """Every source that contributes at least one point, derived from the scene."""
    return tuple(sorted(scene.source_counts()))
