"""Dataset for the Conservative Delta Refiner.

The change this module makes relative to V2 is not the input; it is what the model
is asked to predict and, just as importantly, what it is trained on.

**Input** is the V2 contract, reused unchanged: candidate-centred normalised local
XYZ, ``current_candidate_membership``, and one explicit global metric channel. The
ROI library, the proposal pairs and the train/validation/test splits are read from
the frozen V2 artefacts; nothing here rebuilds them.

**Output** is not a mask. ``current_mask`` is the default answer, and the model only
predicts two per-point decisions:

* ``REMOVE`` - supervised **only** on points where ``current_mask == 1``
* ``ADD``    - supervised **only** on points where ``current_mask == 0``

so a point inside the mask that belongs there is a REMOVE-negative, and a point
outside it that belongs there is an ADD-positive. Neither head ever has to
re-derive the mask, and neither head can propose a change it was not asked about.

**GOOD proposals are explicit training rows.** The library carries 308 GT_CORRECT
rows against 5219 defective ones, so a naive pass would teach "always edit" and
reproduce the V2 failure. Sampling targets a declared GOOD share and the realised
ratio is recorded per split, never inferred.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import v25_config as cfg

#: Input channels, identical to V2 so the backbone sees the same tensor.
FEATURE_CHANNELS = ('local_x', 'local_y', 'local_z', 'current_candidate_membership',
                    'log_candidate_extent')

MIN_SCALE_M = 0.25
#: How many targets' geometry stay resident.
GEOMETRY_CACHE_TARGETS = 8


@dataclass
class Example:
    """One (proposal, target) training or evaluation row."""

    key: str
    site: str
    target_id: str
    proposal_id: str
    proposal_kind: str
    corruption: str
    severity: str
    current_iou: float

    @property
    def is_good(self) -> bool:
        return self.current_iou >= cfg.GOOD_IOU_THRESHOLD


@dataclass
class SamplingReport:
    """What was actually trained on, per split."""

    rows: int = 0
    good_rows: int = 0
    bad_rows: int = 0
    good_repeat: int = 1
    target_good_fraction: float = 0.0
    per_kind: dict = field(default_factory=dict)
    per_corruption: dict = field(default_factory=dict)

    @property
    def good_fraction(self) -> float:
        return float(self.good_rows / self.rows) if self.rows else 0.0

    def as_dict(self) -> dict:
        return {'rows': self.rows, 'good_rows': self.good_rows,
                'bad_rows': self.bad_rows,
                'realised_good_fraction': round(self.good_fraction, 6),
                'target_good_fraction': self.target_good_fraction,
                'good_repeat': self.good_repeat,
                'per_proposal_kind': self.per_kind,
                'per_corruption': self.per_corruption,
                'good_rows_are_explicit_training_samples': True}


def file_digest(path: Path) -> str:
    """SHA-256 of a file, streamed."""
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        while chunk := handle.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_library(directory: Path | None = None) -> dict:
    """Read the frozen V2 ROI library, verifying each file's recorded digest."""
    directory = Path(directory or cfg.ROI_DIR)
    index = json.loads((directory / 'index.json').read_text(encoding='utf-8'))
    for row in index['targets']:
        path = directory / row['file']
        if file_digest(path) != row['sha256']:
            raise RuntimeError(f'frozen ROI library entry changed: {path}')
        with np.load(path, allow_pickle=False) as data:
            row['xyz'] = np.asarray(data['xyz'])
            row['instance'] = np.asarray(data['instance'])
            masks = {key[len('mask_'):]: np.asarray(data[key])
                     for key in data.files if key.startswith('mask_')}
            masks['gt'] = (np.asarray(data['gt_mask'])
                           if 'gt_mask' in data.files
                           else np.asarray(data['instance'])
                           == int(str(row['target_id']).rsplit('#', 1)[1]))
            row['masks'] = masks
    return index


def proposal_mask(cluster: dict, proposal_id: str) -> np.ndarray:
    """The proposal's mask; ``GT_CORRECT`` is the ground-truth mask itself."""
    if proposal_id == 'GT_CORRECT':
        return np.asarray(cluster['masks']['gt'], dtype=bool)
    return np.asarray(cluster['masks'][proposal_id], dtype=bool)


def build_examples(index: dict, targets: list[str]) -> list[Example]:
    """Every proposal row of the named targets, in a deterministic order."""
    wanted = set(targets)
    out = []
    for row in index['targets']:
        if row['target_id'] not in wanted:
            continue
        for proposal in row['rows']:
            out.append(Example(
                key=f"{row['target_id']}|{proposal['proposal_id']}",
                site=row['site'], target_id=row['target_id'],
                proposal_id=proposal['proposal_id'],
                proposal_kind=proposal['proposal_kind'],
                corruption=proposal.get('corruption', 'NONE'),
                severity=proposal.get('severity', 'NONE'),
                current_iou=float(proposal['iou'])))
    return out


def sample_training_set(examples: list[Example], seed: int,
                        good_fraction: float = None,
                        max_good_repeat: int = None) -> tuple:
    """Repeat the GOOD rows so they carry a declared share of the training set.

    Returns the resampled list plus a report of what was actually achieved. The
    repeat factor is capped: a GOOD row repeated twenty times would memorise one
    tree rather than teach "leave a correct tree alone", and a capped factor that
    still falls short of the target is reported as short rather than hidden.
    """
    good_fraction = float(good_fraction if good_fraction is not None
                          else cfg.TARGET_GOOD_FRACTION)
    max_good_repeat = int(max_good_repeat if max_good_repeat is not None
                          else cfg.MAX_GOOD_REPEAT)
    good = [row for row in examples if row.is_good]
    bad = [row for row in examples if not row.is_good]
    report = SamplingReport()
    if not good or not bad:
        report.rows = len(examples)
        report.good_rows = len(good)
        report.bad_rows = len(bad)
        report.per_kind = _counts(examples, 'proposal_kind')
        report.per_corruption = _counts(examples, 'corruption')
        return list(examples), report

    # With GOOD rows repeated r times the GOOD share is (r*G)/(r*G + B), where G and
    # B are the original counts. The extra copies must not be counted as BAD, or the
    # realised share drifts from the target and the report would claim a balance
    # that was never trained: the first version of this function computed the repeat
    # from the resampled total and produced a 0.65 GOOD share while asking for 0.50.
    candidates = list(range(1, max_good_repeat + 1))
    shares = {r: (r * len(good)) / (r * len(good) + len(bad)) for r in candidates}
    repeat = min(candidates, key=lambda r: (abs(shares[r] - good_fraction), r))
    resampled = list(examples)
    for _ in range(repeat - 1):
        resampled.extend(good)
    report.rows = len(resampled)
    report.good_rows = len(good) * repeat
    report.bad_rows = len(bad)
    report.good_repeat = repeat
    report.target_good_fraction = good_fraction
    report.per_kind = _counts(resampled, 'proposal_kind')
    report.per_corruption = _counts(resampled, 'corruption')
    rng = np.random.default_rng(np.random.SeedSequence([seed, 31]))
    order = rng.permutation(len(resampled))
    return [resampled[int(i)] for i in order], report


def _counts(examples, attribute: str) -> dict:
    out: dict[str, int] = {}
    for row in examples:
        value = str(getattr(row, attribute))
        out[value] = out.get(value, 0) + 1
    return dict(sorted(out.items()))


def candidate_frame(current_xyz: np.ndarray) -> dict:
    """The frame the input is expressed in, derived from the proposal alone."""
    centre = np.array([current_xyz[:, 0].mean(), current_xyz[:, 1].mean(),
                       np.median(current_xyz[:, 2])], dtype=np.float32)
    height = float(current_xyz[:, 2].max() - current_xyz[:, 2].min())
    return {'centre': centre, 'scale': np.float32(max(height, MIN_SCALE_M))}


def sample_points(current: np.ndarray, gt: np.ndarray, instance: np.ndarray,
                  points: int, rng: np.random.Generator) -> tuple:
    """Stratified sample over the populations the delta heads care about.

    The sampler floors every stratum, so a stratum that is rare in the ROI is
    sampled at a far higher rate than its prevalence. That is deliberate - a free
    uniform draw can miss the sparse strata entirely, and a head cannot learn a
    population it never sees - but it distorts the decision base rate, so the
    inclusion probability of each point is returned with it.

    The first version of this round ignored that and trained the heads on the
    sampler's distribution instead of the forest's. Measured on 200 training rows,
    the removal-positive rate was 0.1506 in the population and 0.1717 as sampled,
    and the ADD-head rate 0.0908 against 0.1298. The correction belongs in the
    loss, and it is why this function returns weights at all.

    Returns ``(selected_indices, importance_weights)``.
    """
    groups = [
        np.flatnonzero(current & gt),                 # keep
        np.flatnonzero(current & ~gt),                # remove
        np.flatnonzero(~current & gt),                # add
        np.flatnonzero(~current & ~gt & (instance > 0)),   # neighbour, no add
        np.flatnonzero(~current & ~gt & (instance <= 0)),  # background, no add
    ]
    total = current.size
    if total <= points:
        base = np.arange(total)
        extra = rng.choice(total, size=points - total, replace=True)
        selected = np.concatenate([base, extra])
        return selected, np.ones(selected.size, dtype=np.float64)
    floor = max(8, points // 16)
    sizes = np.array([g.size for g in groups], dtype=np.int64)
    quotas = np.minimum(sizes, floor)
    remaining = points - int(quotas.sum())
    if remaining > 0:
        headroom = sizes - quotas
        weights = headroom / max(headroom.sum(), 1)
        extra = np.minimum(np.floor(weights * remaining).astype(np.int64), headroom)
        left = remaining - int(extra.sum())
        if left > 0:
            order = np.argsort(-(weights * remaining - extra), kind='stable')[:left]
            extra[order] += 1
            extra = np.minimum(extra, headroom)
        quotas += extra
    picked, importance = [], []
    for group, quota in zip(groups, quotas, strict=True):
        if quota <= 0 or group.size == 0:
            continue
        if quota >= group.size:
            chosen = group
            inclusion = 1.0
        else:
            chosen = np.sort(rng.choice(group, size=int(quota), replace=False))
            inclusion = float(quota) / float(group.size)
        picked.append(chosen)
        # Inverse inclusion probability: a point drawn from a stratum that was
        # over-sampled carries less weight, so the effective distribution matches
        # the ROI.
        importance.append(np.full(chosen.size, 1.0 / max(inclusion, 1e-9)))
    selected = np.concatenate(picked) if picked else np.arange(0)
    weight = np.concatenate(importance) if importance else np.arange(0) * 1.0
    if selected.size < points:
        pool = np.setdiff1d(np.arange(total), selected)
        take = min(points - selected.size, pool.size)
        if take:
            extra_rows = rng.choice(pool, size=take, replace=False)
            selected = np.concatenate([selected, extra_rows])
            # A point drawn by the top-up carries the population rate, not a floor.
            weight = np.concatenate([weight, np.full(take, float(total) / max(points, 1))])
    if selected.size > points:
        keep = rng.choice(selected.size, size=points, replace=False)
        selected, weight = selected[keep], weight[keep]
    order = np.argsort(selected, kind='stable')
    return selected[order], weight[order]


class DeltaDataset(torch.utils.data.Dataset):
    """One delta-training example: features plus the two head targets."""

    def __init__(self, index: dict, examples: list[Example], points: int,
                 seed: int, augment: bool):
        self.by_target = {row['target_id']: row for row in index['targets']}
        self.examples = examples
        self.points = int(points)
        self.seed = int(seed)
        self.augment = bool(augment)
        self._cache: dict[str, dict] = {}

    def __len__(self) -> int:
        return len(self.examples)

    def _geometry(self, target_id: str) -> dict:
        cached = self._cache.get(target_id)
        if cached is not None:
            return cached
        cluster = self.by_target[target_id]
        entry = {'xyz': np.asarray(cluster['xyz'], dtype=np.float32),
                 'instance': np.asarray(cluster['instance'], dtype=np.int32),
                 'gt': np.asarray(cluster['masks']['gt'], dtype=bool)}
        if len(self._cache) >= GEOMETRY_CACHE_TARGETS:
            self._cache.pop(next(iter(self._cache)))
        self._cache[target_id] = entry
        return entry

    def __getitem__(self, item: int) -> dict:
        example = self.examples[item]
        cluster = self.by_target[example.target_id]
        geometry = self._geometry(example.target_id)
        rng = np.random.default_rng(np.random.SeedSequence(
            [self.seed, int(hashlib.sha256(example.key.encode()).hexdigest()[:8], 16)]))
        xyz, gt, instance = geometry['xyz'], geometry['gt'], geometry['instance']
        current = proposal_mask(cluster, example.proposal_id)
        frame = candidate_frame(xyz[current]) if current.any() else \
            {'centre': xyz.mean(axis=0), 'scale': np.float32(1.0)}
        selected, importance = sample_points(current, gt, instance, self.points, rng)

        local = (xyz[selected] - frame['centre']) / frame['scale']
        membership = current[selected].astype(np.float32)
        scale_channel = np.full((selected.size, 1), math.log(float(frame['scale'])),
                                dtype=np.float32)
        features = np.concatenate([local.astype(np.float32), membership[:, None],
                                   scale_channel], axis=1)
        if self.augment:
            angle = rng.random() * 2.0 * math.pi
            cos, sin = math.cos(angle), math.sin(angle)
            rotation = np.array([[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]],
                                dtype=np.float32)
            features[:, :3] = features[:, :3] @ rotation.T
            features[:, :3] += rng.normal(0.0, 0.01, size=(1, 3)).astype(np.float32)

        truth = gt[selected]
        # REMOVE is defined and supervised only inside the mask; ADD only outside it.
        remove_target = np.where(membership > 0.5, truth.astype(np.float32), -1.0)
        add_target = np.where(membership > 0.5, -1.0, truth.astype(np.float32))
        return {
            'features': torch.from_numpy(features),
            'current': torch.from_numpy(membership),
            'truth': torch.from_numpy(truth.astype(np.float32)),
            'remove_target': torch.from_numpy(remove_target.astype(np.float32)),
            'add_target': torch.from_numpy(add_target.astype(np.float32)),
            'importance': torch.from_numpy(importance.astype(np.float32)),
            'filename': example.key,
            'site': example.site,
            'proposal_kind': example.proposal_kind,
            'corruption': example.corruption,
            'severity': example.severity,
            'current_iou': float(example.current_iou),
            'is_good': bool(example.is_good),
        }


class FullRoiDeltaDataset(torch.utils.data.Dataset):
    """Every point of one ROI, capped by the same stride V2 used for evaluation."""

    def __init__(self, index: dict, examples: list[Example],
                 max_points: int | None = None):
        self.by_target = {row['target_id']: row for row in index['targets']}
        self.examples = examples
        self.max_points = int(max_points or cfg.EVAL_MAX_POINTS)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, item: int) -> dict:
        example = self.examples[item]
        cluster = self.by_target[example.target_id]
        xyz = np.asarray(cluster['xyz'], dtype=np.float32)
        gt = np.asarray(cluster['masks']['gt'], dtype=bool)
        current = proposal_mask(cluster, example.proposal_id)
        stored, stride = int(len(xyz)), 1
        if stored > self.max_points:
            order = np.lexsort((xyz[:, 1], xyz[:, 0], xyz[:, 2]))
            stride = int(np.ceil(stored / self.max_points))
            keep = np.sort(order[::stride][:self.max_points])
            xyz, gt, current = xyz[keep], gt[keep], current[keep]
        frame = candidate_frame(xyz[current]) if current.any() else \
            {'centre': xyz.mean(axis=0), 'scale': np.float32(1.0)}
        local = (xyz - frame['centre']) / frame['scale']
        features = np.concatenate(
            [local.astype(np.float32), current.astype(np.float32)[:, None],
             np.full((len(xyz), 1), math.log(float(frame['scale'])), dtype=np.float32)],
            axis=1)
        return {
            'features': torch.from_numpy(features),
            'current': torch.from_numpy(current.astype(np.float32)),
            'truth': torch.from_numpy(gt.astype(np.float32)),
            'filename': example.key,
            'site': example.site,
            'proposal_kind': example.proposal_kind,
            'corruption': example.corruption,
            'severity': example.severity,
            'current_iou_library': float(example.current_iou),
            'is_good_library': bool(example.is_good),
            'roi_points': int(len(xyz)),
            'roi_points_stored': stored,
            'roi_stride': stride,
        }


def collate(batch: list[dict]) -> dict:
    out = {}
    for key in ('features', 'current', 'truth', 'remove_target', 'add_target',
                'importance'):
        out[key] = torch.stack([item[key] for item in batch], dim=0)
    for key in ('filename', 'site', 'proposal_kind', 'corruption', 'severity',
                'current_iou', 'is_good'):
        out[key] = [item[key] for item in batch]
    return out


def collate_full(batch: list[dict]) -> dict:
    out = {}
    for key in ('features', 'current', 'truth'):
        out[key] = [item[key] for item in batch]
    for key in ('filename', 'site', 'proposal_kind', 'corruption', 'severity',
                'current_iou_library', 'is_good_library', 'roi_points',
                'roi_points_stored', 'roi_stride'):
        out[key] = [item[key] for item in batch]
    return out
