"""Refiner dataset: turning the ROI library into fixed-size model inputs.

The input contract is the V2 contract this audit is testing:

``xyz``          candidate-centred, scaled by the candidate's own geometric
                 extent, so no absolute tree size is readable from the
                 coordinates
``membership``   ``current_candidate_membership`` - the proposal under review
``log_scale``    one explicit global channel: the candidate's own extent. Constant
                 per example, exactly like membership, so it adds no capacity.

Augmentation is applied on the fly and is identical for every architecture.

Two things this module refuses to do:

* it never fabricates an offset target. The reference is the target instance's own
  robust stem-base proxy, defined in :func:`stem_base_reference` and recorded with
  every row; offset supervision is applied only where the publisher's ground truth
  says a point belongs to the target, because for any other point the true offset
  is unknown.
* it never uses the held-out site for anything except the final test. Splits are
  by tree, and the cross-site split is by site.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import numpy as np
import torch

#: Input channels, in order. Declared once so the report can quote them.
FEATURE_CHANNELS = ('local_x', 'local_y', 'local_z', 'current_candidate_membership',
                    'log_candidate_extent')

#: Offset supervision weight applied only to ground-truth target points.
OFFSET_ACTIVE_FRACTION = 0.10
MIN_SCALE_M = 0.25
#: How many targets' geometries stay resident. Small, so a dataset over hundreds of
#: targets with 100 000-point ROIs cannot exhaust memory.
GEOMETRY_CACHE_TARGETS = 8


@dataclass
class Example:
    """One (proposal, target) training pair, materialised."""

    key: str
    site: str
    target_id: str
    proposal_id: str
    proposal_kind: str
    corruption: str
    severity: str
    split: str
    current_iou: float


def stem_base_reference(gt_xyz: np.ndarray, ground_z: np.ndarray,
                        fraction: float = OFFSET_ACTIVE_FRACTION) -> dict:
    """Robust stem/base proxy for a target instance.

    The public benchmark publishes tree instance membership, not a stem axis, so
    there is no stem ground truth to use. The proxy is the median XYZ of the
    lowest ``fraction`` of the instance by height above ground, which is stable
    against a single stray low point in a way that a bare minimum is not. It is
    recorded as a proxy on every row and is never presented as measured stem data.

    Returns ``{'reference': (3,), 'definition': str, 'points': int}``.
    """
    if gt_xyz.shape[0] == 0:
        raise ValueError('cannot define a stem reference for an empty instance')
    hag = gt_xyz[:, 2] - ground_z
    count = max(1, int(round(fraction * gt_xyz.shape[0])))
    order = np.argsort(hag, kind='stable')[:count]
    reference = np.median(gt_xyz[order], axis=0)
    return {'reference': reference.astype(np.float32),
            'definition': (f'median XYZ of the lowest {fraction:.0%} of the target '
                           'instance by height above the local ground proxy'),
            'points': int(count)}


def candidate_frame(current_xyz: np.ndarray) -> dict:
    """The frame the input is expressed in, derived from the *proposal* only.

    Using the proposal rather than the truth is the whole point: at inference the
    truth does not exist. The centre is the proposal's mean XY and its median Z,
    and the scale is the proposal's vertical extent, floored so a degenerate
    sliver cannot blow the coordinates up.
    """
    centre = np.array([current_xyz[:, 0].mean(), current_xyz[:, 1].mean(),
                       np.median(current_xyz[:, 2])], dtype=np.float32)
    height = float(current_xyz[:, 2].max() - current_xyz[:, 2].min())
    scale = max(height, MIN_SCALE_M)
    return {'centre': centre, 'scale': np.float32(scale)}


def proposal_mask(cluster: dict, proposal_id: str) -> np.ndarray:
    """The proposal's mask, with ``GT_CORRECT`` meaning the ground truth itself.

    The library stores one mask per defective proposal and one ground-truth mask.
    A correct proposal is not stored twice under a second name; it *is* the
    ground-truth mask, and resolving it here keeps that identity in one place.
    """
    if proposal_id == 'GT_CORRECT':
        return np.asarray(cluster['masks']['gt'], dtype=bool)
    return np.asarray(cluster['masks'][proposal_id], dtype=bool)


class RefinerDataset(torch.utils.data.Dataset):
    """One example per (target, proposal) row of the ROI library.

    Each ROI is loaded and its per-point quantities are derived once per target,
    not once per row. Eighteen proposals share one cloud; recomputing the ground
    proxy and the stem reference eighteen times is pure waste, and on a 300 000
    point ROI it is the dominant cost of an epoch.
    """

    def __init__(self, index: dict, examples: list[Example], points: int,
                 seed: int, augment: bool):
        self.index = index
        self.by_target = {row['target_id']: row for row in index['targets']}
        self.examples = examples
        self.points = int(points)
        self.seed = int(seed)
        self.augment = bool(augment)
        self._cache: dict[str, dict] = {}

    def __len__(self) -> int:
        return len(self.examples)

    def target_geometry(self, target_id: str) -> dict:
        """Per-point arrays for one target's ROI, derived once."""
        cached = self._cache.get(target_id)
        if cached is not None:
            return cached
        cluster = self.by_target[target_id]
        xyz = np.asarray(cluster['xyz'], dtype=np.float32)
        instance = np.asarray(cluster['instance'], dtype=np.int32)
        gt = np.asarray(cluster['masks']['gt'], dtype=bool)
        ground_z = z_from_ground(xyz)
        target_points = np.flatnonzero(gt)
        reference = stem_base_reference(xyz[target_points], ground_z[target_points])
        entry = {'xyz': xyz, 'instance': instance, 'gt': gt, 'ground_z': ground_z,
                 'reference': reference}
        if len(self._cache) >= GEOMETRY_CACHE_TARGETS:
            self._cache.pop(next(iter(self._cache)))
        self._cache[target_id] = entry
        return entry

    def __getitem__(self, item: int) -> dict:
        example = self.examples[item]
        cluster = self.by_target[example.target_id]
        geometry = self.target_geometry(example.target_id)
        rng = np.random.default_rng(np.random.SeedSequence(
            [self.seed, int(hashlib.sha256(example.key.encode()).hexdigest()[:8], 16)]))
        xyz = geometry['xyz']
        instance = geometry['instance']
        gt = geometry['gt']
        current = proposal_mask(cluster, example.proposal_id)
        reference = geometry['reference']
        frame = candidate_frame(xyz[current])

        selected = sample_points(current, gt, instance, self.points, rng)
        local = xyz[selected]
        local = (local - frame['centre']) / frame['scale']
        membership = current[selected].astype(np.float32)
        scale_channel = np.full((selected.size, 1), math.log(float(frame['scale'])),
                                dtype=np.float32)
        features = np.concatenate(
            [local.astype(np.float32), membership[:, None], scale_channel], axis=1)

        offset = ((reference['reference'][None, :] - xyz[selected])
                  / np.float32(frame['scale'])).astype(np.float32)
        offset_active = gt[selected].astype(np.float32)

        if self.augment:
            angle = rng.random() * 2.0 * math.pi
            cos, sin = math.cos(angle), math.sin(angle)
            rotation = np.array([[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]],
                                dtype=np.float32)
            features[:, :3] = features[:, :3] @ rotation.T
            offset = offset @ rotation.T
            features[:, :3] += rng.normal(0.0, 0.01, size=(1, 3)).astype(np.float32)

        return {
            'features': torch.from_numpy(features),
            'target': torch.from_numpy(gt[selected].astype(np.float32)),
            'current': torch.from_numpy(membership),
            'offset': torch.from_numpy(offset),
            'offset_active': torch.from_numpy(offset_active),
            'scale': torch.tensor(float(frame['scale'])),
            'filename': example.key,
            'site': example.site,
            'target_id': example.target_id,
            'proposal_kind': example.proposal_kind,
            'corruption': example.corruption,
            'severity': example.severity,
            'current_iou': float(example.current_iou),
        }


def z_from_ground(xyz: np.ndarray) -> np.ndarray:
    """Per-point ground Z from a 1th-percentile grid, for the stem reference only.

    The ROI library stores no ground channel, so the same bounded proxy the
    proposal stage uses is recomputed here. It is used only to order points by
    height when picking the stem-base proxy, never as a label. The percentile is
    ``np.minimum.reduceat`` over stable-sorted cells, which is the same bounded
    reduction the proposal stage performs.
    """
    grid = 0.5
    origin = xyz[:, :2].min(axis=0)
    cells = np.floor((xyz[:, :2] - origin) / grid).astype(np.int64)
    cols = int(cells[:, 0].max()) + 1
    keys = cells[:, 1] * cols + cells[:, 0]
    order = np.argsort(keys, kind='stable')
    sorted_keys = keys[order]
    sorted_z = xyz[order, 2]
    unique, starts = np.unique(sorted_keys, return_index=True)
    stops = np.concatenate([starts[1:], [len(sorted_z)]])
    surface = np.empty(int(keys.max()) + 1, dtype=np.float32)
    for key, start, stop in zip(unique, starts, stops, strict=True):
        surface[key] = np.percentile(sorted_z[start:stop], 1.0)
    return surface[keys]


def sample_points(current: np.ndarray, gt: np.ndarray, instance: np.ndarray,
                  points: int, rng: np.random.Generator) -> np.ndarray:
    """Stratified sample that guarantees changed evidence is represented.

    Three groups - proposal, target-only, other-tree - each get a floor, and the
    remainder is proportional. A free uniform draw can miss the changed region
    entirely on a large ROI, which would train the model on a task with no defect
    in it.
    """
    proposal_rows = np.flatnonzero(current)
    target_rows = np.flatnonzero(gt & ~current)
    other_rows = np.flatnonzero(~gt & ~current & (instance > 0))
    background_rows = np.flatnonzero(~gt & ~current & (instance <= 0))
    total = current.size
    if total <= points:
        base = np.arange(total)
        extra = rng.choice(total, size=points - total, replace=True)
        return np.concatenate([base, extra])

    floor = max(8, points // 16)
    groups = [proposal_rows, target_rows, other_rows, background_rows]
    sizes = np.array([g.size for g in groups], dtype=np.int64)
    quotas = np.minimum(sizes, floor)
    remaining = points - int(quotas.sum())
    if remaining > 0:
        headroom = sizes - quotas
        weights = headroom / max(headroom.sum(), 1)
        extra = np.floor(weights * remaining).astype(np.int64)
        extra = np.minimum(extra, headroom)
        left = remaining - int(extra.sum())
        if left > 0:
            order = np.argsort(-(weights * remaining - extra), kind='stable')[:left]
            extra[order] += 1
            extra = np.minimum(extra, headroom)
        quotas += extra
    picked = []
    for group, quota in zip(groups, quotas, strict=True):
        if quota <= 0 or group.size == 0:
            continue
        if quota >= group.size:
            picked.append(group)
        else:
            picked.append(np.sort(rng.choice(group, size=int(quota), replace=False)))
    selected = np.concatenate(picked) if picked else np.arange(0)
    if selected.size < points:
        pool = np.setdiff1d(np.arange(total), selected)
        add = rng.choice(pool, size=min(points - selected.size, pool.size), replace=False)
        selected = np.concatenate([selected, add])
    if selected.size > points:
        selected = np.sort(rng.choice(selected, size=points, replace=False))
    return np.sort(selected)


class FixedSubsetDataset(torch.utils.data.Dataset):
    """A deterministic, bounded subset of each ROI, for validation only.

    Early stopping must run every epoch, and a full-ROI pass over a 1.5 M-point
    region costs minutes. This class takes the same stratified sample the trainer
    consumes, with a fixed seed, so the validation signal is comparable across
    epochs and across architectures. The reported test metrics never use it; they
    are always computed over the whole ROI.
    """

    def __init__(self, index: dict, examples: list[Example], points: int, seed: int):
        self.by_target = {row['target_id']: row for row in index['targets']}
        self.examples = examples
        self.points = int(points)
        self.seed = int(seed)
        self._cache: dict[str, tuple] = {}

    def __len__(self) -> int:
        return len(self.examples)

    def _subset(self, target_id: str, proposal_id: str) -> tuple:
        token = f'{target_id}|{proposal_id}'
        cached = self._cache.get(token)
        if cached is not None:
            return cached
        cluster = self.by_target[target_id]
        xyz = np.asarray(cluster['xyz'], dtype=np.float32)
        instance = np.asarray(cluster['instance'], dtype=np.int32)
        gt = np.asarray(cluster['masks']['gt'], dtype=bool)
        current = proposal_mask(cluster, proposal_id)
        rng = np.random.default_rng(np.random.SeedSequence(
            [self.seed, int(hashlib.sha256(token.encode()).hexdigest()[:8], 16)]))
        selected = sample_points(current, gt, instance, self.points, rng)
        entry = (xyz[selected], gt[selected], current[selected])
        if len(self._cache) >= GEOMETRY_CACHE_TARGETS * 4:
            self._cache.pop(next(iter(self._cache)))
        self._cache[token] = entry
        return entry

    def __getitem__(self, item: int) -> dict:
        example = self.examples[item]
        xyz, gt, current = self._subset(example.target_id, example.proposal_id)
        frame = candidate_frame(xyz[current]) if current.any() else \
            {'centre': xyz.mean(axis=0), 'scale': np.float32(1.0)}
        local = (xyz - frame['centre']) / frame['scale']
        features = np.concatenate(
            [local.astype(np.float32), current.astype(np.float32)[:, None],
             np.full((len(xyz), 1), math.log(float(frame['scale'])), dtype=np.float32)],
            axis=1)
        return {
            'features': torch.from_numpy(features),
            'target': torch.from_numpy(gt.astype(np.float32)),
            'current': torch.from_numpy(current.astype(np.float32)),
            'filename': example.key,
            'site': example.site,
            'target_id': example.target_id,
            'proposal_kind': example.proposal_kind,
            'corruption': example.corruption,
            'severity': example.severity,
            'current_iou': float(example.current_iou),
        }


#: Tensor keys every training item carries.
TRAINING_TENSORS = ('features', 'target', 'current', 'offset', 'offset_active',
                    'scale')
#: Tensor keys a validation subset carries: enough to score the membership loss.
VALIDATION_TENSORS = ('features', 'target', 'current')


def collate(batch: list[dict]) -> dict:
    out = {}
    for key in TRAINING_TENSORS:
        if key in batch[0]:
            out[key] = torch.stack([item[key] for item in batch], dim=0)
    for key in ('filename', 'site', 'target_id', 'proposal_kind', 'corruption',
                'severity', 'current_iou'):
        out[key] = [item[key] for item in batch]
    return out


def collate_validation(batch: list[dict]) -> dict:
    out = {}
    for key in VALIDATION_TENSORS:
        out[key] = torch.stack([item[key] for item in batch], dim=0)
    for key in ('filename', 'site', 'proposal_kind', 'corruption', 'severity',
                'current_iou'):
        out[key] = [item[key] for item in batch]
    return out
