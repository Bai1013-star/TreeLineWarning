"""Local ROI construction and proposal generation for PUBLIC_GT_PROPOSAL_TEST.

A refiner cannot repair what is not in front of it. This module therefore builds
each region of interest from the *real* publisher per-tree clouds, so the ROI
contains the target tree, the neighbours that could contaminate it, and - when
the proposal has dropped them - the target's own missing points. Coverage of the
missing points inside the ROI is measured and reported, not assumed.

Proposal provenance is explicit and never blurred:

``REAL_ALGORITHM_PROPOSAL``
    Output of the product's own ``segment_stem_guided_hybrid`` - the crown/stem
    hybrid refinement the shipped ``CROWN_STEM_HYBRID`` backend calls.

``SIMULATED_PROPOSAL``
    A deterministic corruption of the publisher's ground truth, used only when
    the real algorithm cannot be run on the public data. Every simulated row
    records the corruption that made it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import v2_config as cfg
import v2_index as index_module
import v2_ply as ply
from scipy import ndimage

REAL = 'REAL_ALGORITHM_PROPOSAL'
SIMULATED = 'SIMULATED_PROPOSAL'

#: A proposal is treated as a usable training pair only above this IoU with the
#: ground-truth instance it is matched to.
MIN_PAIR_IOU = 0.10
#: The match must beat the runner-up by this factor to be called unique.
UNIQUE_MATCH_MARGIN = 1.5
#: Defect taxonomy used to stratify the bakeoff.
DEFECTS = ('ADD_CONTAMINATION', 'REMOVE_MISSING', 'MERGED_NEIGHBOR',
           'CROWN_INCOMPLETE', 'STEM_MISSING')
#: Neighbours closer than this to the target are the ones that can contaminate it.
NEIGHBOUR_RADIUS_M = 6.0


@dataclass
class Patch:
    """A local scene: all publisher instances inside a bounded XY window."""

    patch_id: str
    site: str
    xyz: np.ndarray                 # (N, 3) float32, native plot coordinates
    instance: np.ndarray            # (N,) int32, publisher instance id, -1 = none
    instance_ids: tuple[int, ...]   # publisher ids present, ascending
    point_counts: dict = field(default_factory=dict)
    ground_z: np.ndarray | None = None


# --- ground proxy -------------------------------------------------------------

def local_ground(xyz: np.ndarray, grid_m: float = 0.5,
                 floor_percentile: float = 1.0) -> np.ndarray:
    """Per-point ground Z from a low-percentile grid surface, then closing.

    The public benchmark publishes no ground class for these plots and this audit
    must not import the product ground classifier. The proxy is a bounded local
    percentile of the lowest points, smoothed by a grey closing so that a single
    stem base cannot pull a whole cell down. It is recorded as a proxy.
    """
    origin = xyz[:, :2].min(axis=0)
    cells = np.floor((xyz[:, :2] - origin) / grid_m).astype(np.int64)
    cols = int(cells[:, 0].max()) + 1
    rows = int(cells[:, 1].max()) + 1
    keys = cells[:, 1] * cols + cells[:, 0]
    surface = np.full(rows * cols, np.nan, dtype=np.float64)
    order = np.argsort(keys, kind='stable')
    sorted_keys = keys[order]
    sorted_z = xyz[order, 2].astype(np.float64)
    unique, starts = np.unique(sorted_keys, return_index=True)
    values = np.empty(len(unique), dtype=np.float64)
    for position, start in enumerate(starts):
        stop = starts[position + 1] if position + 1 < len(starts) else len(sorted_z)
        values[position] = np.percentile(sorted_z[start:stop], floor_percentile)
    surface[unique] = values
    filled = np.where(np.isnan(surface),
                      np.nanmin(surface) if np.isfinite(surface).any() else 0.0,
                      surface).reshape(rows, cols)
    closed = ndimage.grey_closing(filled, size=(3, 3))
    return closed[cells[:, 1], cells[:, 0]]


# --- neighbourhood assembly ---------------------------------------------------

def _xy_cells(rows, cell_m: float = 20.0) -> dict:
    grid: dict[tuple[int, int], list] = {}
    for row in rows:
        for cx in range(int(math.floor(row.xyz_min[0] / cell_m)),
                        int(math.floor(row.xyz_max[0] / cell_m)) + 1):
            for cy in range(int(math.floor(row.xyz_min[1] / cell_m)),
                            int(math.floor(row.xyz_max[1] / cell_m)) + 1):
                grid.setdefault((cx, cy), []).append(row)
    return grid


def window_rows(rows, bounds, grid: dict, cell_m: float = 20.0) -> list:
    """Every instance whose bounding box intersects ``bounds`` (x0, y0, x1, y1)."""
    found: dict[str, object] = {}
    for cx in range(int(math.floor(bounds[0] / cell_m)),
                    int(math.floor(bounds[2] / cell_m)) + 1):
        for cy in range(int(math.floor(bounds[1] / cell_m)),
                        int(math.floor(bounds[3] / cell_m)) + 1):
            for row in grid.get((cx, cy), ()):
                if (row.xyz_min[0] <= bounds[2] and row.xyz_max[0] >= bounds[0]
                        and row.xyz_min[1] <= bounds[3] and row.xyz_max[1] >= bounds[1]):
                    found[row.key] = row
    return sorted(found.values(), key=lambda r: r.key)


def densest_windows(index: dict, window_m: float, count: int,
                    cell_m: float = 20.0) -> list[tuple]:
    """The ``count`` densest non-overlapping XY windows of a site.

    A single window yields a handful of qualifying trees, which is far too few
    independent instances to compare architectures on. Placing several windows on
    the densest cells of the plot and de-duplicating targets across them gives a
    usable pool without ever re-using one tree as two examples.
    """
    from collections import Counter
    density: Counter = Counter()
    for entry in index['entries']:
        x0, y0, _, x1, y1, _ = entry['bbox']
        density[(int(np.floor((x0 + x1) / 2 / cell_m)),
                 int(np.floor((y0 + y1) / 2 / cell_m)))] += 1
    ranked = sorted(density.items(), key=lambda item: (-item[1], item[0]))
    span = max(1, int(np.ceil(window_m / cell_m)))
    chosen: list[tuple[int, int]] = []
    for (cx, cy), _ in ranked:
        if any(abs(cx - ox) < span and abs(cy - oy) < span for ox, oy in chosen):
            continue
        chosen.append((cx, cy))
        if len(chosen) == count:
            break
    windows = []
    for cx, cy in chosen:
        centre_x, centre_y = (cx + 0.5) * cell_m, (cy + 0.5) * cell_m
        windows.append((centre_x - window_m / 2, centre_y - window_m / 2,
                        centre_x + window_m / 2, centre_y + window_m / 2))
    return windows


def build_patch_at(site: str, bounds, index: dict, members=None,
                   patch_id: str | None = None, verbose: bool = False) -> Patch:
    """Assemble a patch from an explicit XY window.

    The same construction as :func:`build_patch`, but the window is supplied by
    the caller so several windows of one site can be used without duplicating
    logic.
    """
    if members is None:
        members = index_module.window_members(index, bounds)
    blocks, labels, next_id = [], [], 0
    counts: dict[int, int] = {}
    for entry in members:
        xyz = ply.read_xyz(Path(entry['path']), dtype=np.float32)
        inside = ((xyz[:, 0] >= bounds[0]) & (xyz[:, 0] <= bounds[2])
                  & (xyz[:, 1] >= bounds[1]) & (xyz[:, 1] <= bounds[3]))
        if not inside.any():
            continue
        kept = xyz[inside]
        blocks.append(kept)
        labels.append(np.full(len(kept), next_id, dtype=np.int32))
        counts[next_id] = {'tree_id': entry['tree_id'], 'key': entry['key'],
                           'points_in_window': int(inside.sum()),
                           'points_total': int(entry['point_count']),
                           'partition': entry['partition']}
        next_id += 1
    if not blocks:
        raise ValueError(f'empty patch for {site} in window {bounds}')
    xyz = np.concatenate(blocks, axis=0)
    instance = np.concatenate(labels, axis=0)
    if verbose:
        print(f'   {patch_id}: {len(xyz):,} points from {len(blocks)} trees', flush=True)
    return Patch(patch_id=patch_id or f'{site}_w{int(bounds[2] - bounds[0])}m',
                 site=site, xyz=xyz, instance=instance,
                 instance_ids=tuple(range(next_id)), point_counts=counts,
                 ground_z=local_ground(xyz))


def build_patch(site: str, window_m: float, index: dict,
                patch_id: str | None = None, verbose: bool = False) -> Patch:
    """Assemble one bounded local patch from publisher instances.

    A window is placed on the densest 20 m cell so the patch is a real piece of
    forest rather than an empty corner of the plot. Instances intersecting the
    window are clipped to it, so a tree crossing the window edge is present but
    truncated, exactly as a real local ROI would be. Tree membership comes from
    the cached bounding-box index, so only trees that intersect the window are
    ever read.

    There is deliberately no "keep the N biggest trees" cap. An earlier revision
    had one, and it silently deleted whole trees from the window, turning real
    instances into fragments and corrupting every ground-truth mask downstream.
    The window size is the only thing that bounds the patch.
    """
    bounds = index_module.densest_window(index, window_m)
    return build_patch_at(site, bounds, index, patch_id=patch_id, verbose=verbose)


def roi_mask(patch: Patch, target_id: int) -> np.ndarray:
    """The ROI the refiner will actually see, for one target instance.

    Built from the target instance's own bounding box, grown in XY by
    ``ROI_XY_PAD_M`` and extended below the target's floor. It is deliberately not
    the target cloud: neighbours inside the box are what the refiner must learn
    to exclude, and the target's own dropped points must still be inside the box
    for a correction to be possible at all.
    """
    target = patch.xyz[patch.instance == target_id]
    if len(target) == 0:
        raise ValueError(f'instance {target_id} is empty in {patch.patch_id}')
    lo, hi = target.min(axis=0), target.max(axis=0)
    height = max(float(hi[2] - lo[2]), 0.1)
    low_z = lo[2] - cfg.ROI_Z_FLOOR_FRACTION * height
    high_z = hi[2] + cfg.ROI_Z_CEILING_PAD_M
    return np.asarray(
        (patch.xyz[:, 0] >= lo[0] - cfg.ROI_XY_PAD_M)
        & (patch.xyz[:, 0] <= hi[0] + cfg.ROI_XY_PAD_M)
        & (patch.xyz[:, 1] >= lo[1] - cfg.ROI_XY_PAD_M)
        & (patch.xyz[:, 1] <= hi[1] + cfg.ROI_XY_PAD_M)
        & (patch.xyz[:, 2] >= low_z) & (patch.xyz[:, 2] <= high_z),
        dtype=bool)


# --- proposal from the real product algorithm ---------------------------------

def real_proposals(patch: Patch, config=None, labels: np.ndarray | None = None,
                   crown_source: str = 'GT_INSTANCE_LABELS',
                   raw_count: np.ndarray | None = None) -> dict:
    """Run the product's stem-guided crown/stem hybrid on a local public patch.

    What this is, precisely. The shipped ``CROWN_STEM_HYBRID`` route first builds
    crown candidates with the product's own coarse CHM/watershed stage and then
    calls ``segment_stem_guided_hybrid`` to refine crown membership with stem
    evidence. Only the second half is reachable here, so what is measured is the
    *crown/stem refinement stage* on real public geometry - not the whole product
    pipeline end to end.

    Two substitutions are unavoidable and are returned in the result rather than
    hidden:

    * ``labels`` is the watershed crown labelling the refinement consumes. The
      caller supplies it; ``GT_INSTANCE_LABELS`` means the publisher ground truth
      was used verbatim, which by construction yields proposals that reproduce
      that ground truth.
    * height above ground comes from the local ground proxy, because the public
      benchmark publishes no ground class and the product ground classifier is out
      of scope for this round.

    No product source is modified: the algorithm is imported and called.
    """
    import sys
    repo_root = str(cfg.REPO_ROOT)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from algorithms.spans.stem_guided_hybrid import segment_stem_guided_hybrid

    source_labels = patch.instance if labels is None else np.asarray(labels, dtype=np.int32)
    if source_labels.shape != (len(patch.xyz),):
        raise ValueError('crown labels must align with the patch')
    weights = (np.ones(len(patch.xyz), dtype=np.int64) if raw_count is None
               else np.asarray(raw_count, dtype=np.int64))
    hag = (patch.xyz[:, 2] - patch.ground_z).astype(np.float64)
    result = segment_stem_guided_hybrid(
        patch.xyz.astype(np.float64), hag, source_labels,
        raw_count=weights, config=config)
    candidates = result.candidates
    masks = []
    for candidate in candidates:
        rows = np.asarray(candidate.member_rows, dtype=np.int64)
        mask = np.zeros(len(patch.xyz), dtype=bool)
        if rows.size:
            mask[rows] = True
        masks.append({
            'mask': mask,
            'state': candidate.quality_state.value,
            'candidate_id': int(candidate.tree_id),
            'source_crown_ids': [int(v) for v in candidate.source_crown_ids],
            'stem_seed_count': len(candidate.stem_seed_ids),
            'membership_support': int(candidate.membership_support),
        })
    for entry in masks:
        matched = np.bincount(patch.instance[entry['mask']],
                              minlength=len(patch.instance_ids))
        matched = matched[1:] if len(matched) > 1 else matched
        entry['matched_instance'] = int(np.argmax(matched)) if matched.size else -1
        entry['matched_points'] = int(matched.max()) if matched.size else 0
    return {'proposals': masks,
            'algorithm': 'algorithms.spans.stem_guided_hybrid.segment_stem_guided_hybrid',
            'provenance': REAL,
            'crown_source': crown_source,
            'validated_stems': len(result.validated_stems),
            'diagnostics': {k: v for k, v in result.diagnostics.items()
                            if isinstance(v, (int, float, str, bool))},
            'substitutions': [
                f'watershed crown labels -> {crown_source}',
                'height above ground -> local low-percentile grid proxy',
                'raw point weight -> uniform 1 per representative point',
            ]}


# --- label-agnostic crown perturbations ---------------------------------------

CROWN_SOURCES = ('GT_INSTANCE_LABELS', 'CROWN_MERGE', 'CROWN_SPLIT')


def perturb_crown_labels(patch: Patch, kind: str, seed: int, fraction: float = 0.15,
                         k: int = 2) -> tuple[np.ndarray, dict]:
    """Degrade the crown labelling the refinement stage consumes.

    These perturbations are label-agnostic: they are computed from point
    coordinates and the labelling alone. A crown is merged with its nearest
    neighbouring crown, or cut into ``k`` pieces by k-means over XY. Both are the
    kind of error a coarse crown stage really makes, and neither consults the
    defect taxonomy being evaluated.

    Returns the perturbed labels plus a record of exactly what changed.
    """
    labels = patch.instance.copy()
    rng = np.random.default_rng(np.random.SeedSequence([seed, hash(kind) & 0xFFFF]))
    ids = np.array(sorted(int(v) for v in np.unique(labels) if v > 0), dtype=np.int64)
    if ids.size < 2:
        return labels, {'kind': kind, 'changed': 0, 'reason': 'fewer than two crowns'}
    centres = {int(i): patch.xyz[labels == i].mean(axis=0) for i in ids}
    changed = 0
    records = []

    if kind == 'CROWN_MERGE':
        count = max(1, int(round(fraction * ids.size)))
        picked = rng.choice(ids, size=min(count, ids.size - 1), replace=False)
        for source in picked:
            source = int(source)
            if labels[labels == source].size == 0:
                continue
            distances = sorted(
                (float(np.linalg.norm(centres[source] - centres[int(other)])), int(other))
                for other in ids if int(other) != source)
            if not distances:
                continue
            nearest = distances[0][1]
            labels[labels == source] = nearest
            changed += 1
            records.append({'merged': source, 'into': nearest,
                            'distance_m': distances[0][0]})
    elif kind == 'CROWN_SPLIT':
        count = max(1, int(round(fraction * ids.size)))
        picked = rng.choice(ids, size=min(count, ids.size), replace=False)
        next_label = int(ids.max()) + 1
        for source in picked:
            source = int(source)
            rows = np.flatnonzero(labels == source)
            if rows.size < 2 * k:
                continue
            xy = patch.xyz[rows, :2]
            centre = xy.mean(axis=0)
            spread = float(np.linalg.norm(xy - centre, axis=1).mean())
            if spread < 1e-6:
                continue
            angle = rng.random() * 2.0 * np.pi
            direction = np.array([np.cos(angle), np.sin(angle)])
            projection = (xy - centre) @ direction
            order = np.argsort(projection, kind='stable')
            groups = np.array_split(order, k)
            for group in groups[1:]:
                labels[rows[group]] = next_label
                next_label += 1
            changed += 1
            records.append({'split': source, 'pieces': k, 'rows': int(rows.size)})
    else:
        raise ValueError(f'unknown crown perturbation {kind!r}')
    return labels, {'kind': kind, 'changed': changed, 'records': records[:20],
                    'crown_count_before': int(ids.size),
                    'crown_count_after': int(np.unique(labels[labels > 0]).size)}


def crown_consistency(labels: np.ndarray, instance: np.ndarray) -> float:
    """Fraction of GT instances whose points lie in a single perturbed crown.

    The refinement stage can only undo a coarse label error if the damaged class
    actually appears in the input it receives. This measures that directly.
    """
    total = 0
    consistent = 0
    for instance_id in np.unique(instance):
        if instance_id <= 0:
            continue
        total += 1
        values = np.unique(labels[instance == instance_id])
        if values.size == 1:
            consistent += 1
    return float(consistent / total) if total else 0.0


# --- proposal metrics ---------------------------------------------------------

def proposal_metrics(mask: np.ndarray, gt_target: np.ndarray, gt_other: np.ndarray,
                     gt_background: np.ndarray) -> dict:
    """Every quantity the protocol asks a candidate row to record."""
    mask = np.asarray(mask, dtype=bool)
    intersection = int(np.count_nonzero(mask & gt_target))
    union = int(np.count_nonzero(mask | gt_target))
    extra = int(np.count_nonzero(mask & ~gt_target))
    missing = int(np.count_nonzero(gt_target & ~mask))
    neighbour = int(np.count_nonzero(mask & gt_other))
    background = int(np.count_nonzero(mask & gt_background))
    precision = intersection / max(int(mask.sum()), 1)
    recall = intersection / max(int(gt_target.sum()), 1)
    iou = intersection / union if union else 0.0
    return {
        'proposal_points': int(mask.sum()),
        'gt_points': int(gt_target.sum()),
        'intersection': intersection, 'union': union,
        'iou': float(iou), 'precision': float(precision), 'recall': float(recall),
        'extra_points': extra, 'missing_points': missing,
        'neighbour_contamination_points': neighbour,
        'background_points': background,
        'neighbour_contamination_fraction': float(neighbour / max(int(mask.sum()), 1)),
        'is_merge': bool(neighbour > 0.25 * max(int(gt_target.sum()), 1)),
        'is_split_or_incomplete': bool(recall < 0.80),
    }


def match_proposals(proposals, gt_instance: np.ndarray, instance_ids) -> dict:
    """Which ground-truth instance each proposal belongs to, and uniqueness.

    A proposal is uniquely matched when its best instance holds more than half of
    the proposal *and* at least ``UNIQUE_MATCH_MARGIN`` times the runner-up.

    The histogram is built by an explicit per-instance count rather than by a
    ``bincount`` offset. Patch instance ids are ``0..n-1``: an earlier revision
    dropped bin 0 in the belief that ids started at 1, which shifted every match
    by one and silently scored real proposals against the neighbouring tree.
    """
    ids = [int(v) for v in instance_ids]
    rows = []
    for position, entry in enumerate(proposals):
        mask = entry['mask']
        total = int(mask.sum())
        if total == 0:
            rows.append({'proposal': position, 'instance': -1, 'unique': False,
                         'share': 0.0, 'runner_up': 0, 'matched_points': 0})
            continue
        inside = gt_instance[mask]
        counts = np.array([int(np.count_nonzero(inside == i)) for i in ids],
                          dtype=np.int64)
        if counts.size == 0 or counts.max() == 0:
            rows.append({'proposal': position, 'instance': -1, 'unique': False,
                         'share': 0.0, 'runner_up': 0, 'matched_points': 0})
            continue
        order = np.argsort(-counts, kind='stable')
        best = int(order[0])
        second = int(counts[order[1]]) if len(order) > 1 else 0
        share = float(counts[best] / max(total, 1))
        rows.append({'proposal': position, 'instance': ids[best],
                     'instance_index': best,
                     'unique': bool(share > 0.5
                                    and counts[best] >= UNIQUE_MATCH_MARGIN * max(second, 1)),
                     'share': share, 'runner_up': second,
                     'matched_points': int(counts[best])})
    return {'rows': rows}
