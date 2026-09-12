"""Task A — SITE_FINGERPRINT_TEST.

Question: does the input a Tree QC model sees already identify *which forest* a
tree came from?

Three representations of the same real good trees are compared under one
otherwise identical pipeline:

``V1_RAW``
    The frozen V1 input contract, reproduced from ``gate_d_sampler``: mean-centred
    local metric XYZ divided by the research-wide constant 10, plus
    ``current_candidate_membership`` = 1. Metric scale is preserved on purpose
    ("a 12 m tree stays 12 m").

``V2_NORMALIZED``
    Candidate-centred, per-cloud robust scaling of geometry (median absolute
    radial extent), random XY rotation, density standardisation by voxel
    thinning plus random dropout. The point cloud itself carries no absolute
    scale, so the model cannot read size off the coordinates.

``V2_NORMALIZED_NO_META``
    ``V2_NORMALIZED`` with the explicit global metric channel withheld. This is
    the shortcut probe in the other direction: it measures how much site signal
    survives with *no* scale information anywhere.

The scientific claim under test is deliberately two-sided. A high ``V1_RAW``
accuracy is evidence that V1's input contained a forest fingerprint. Whether
``V2_NORMALIZED`` reduces it is only useful if the refinement task still works,
so this module reports the number and does not decide the verdict by itself.
"""
from __future__ import annotations

import json
import math
import time

import numpy as np
import torch
import v2_config as cfg
import v2_data as data
import v2_ply as ply
from torch import nn

REPRESENTATIONS = ('V1_RAW', 'V2_NORMALIZED', 'V2_NORMALIZED_NO_META')

#: Additional diagnostics, not the two primary arms.
DIAGNOSTIC_REPRESENTATIONS = ('V2_ANISO_NORMALIZED', 'V2_ANISO_NORMALIZED_NO_META')

ALL_REPRESENTATIONS = REPRESENTATIONS + DIAGNOSTIC_REPRESENTATIONS

#: The metric constant the frozen V1 sampler divides by. Reproduced, not re-chosen.
METRIC_SCALE_CONSTANT = 10.0

#: Explicit global metric metadata, in metres and log metres^-2.
META_CHANNELS = ('height_m', 'crown_width_m', 'log_point_density')
META_DIM = len(META_CHANNELS)


class SiteClassifier(nn.Module):
    """One shared, deliberately small backbone for every representation.

    Identical parameter count and identical training budget for all three arms:
    only the input representation differs, which is what makes the comparison a
    statement about the representation rather than about capacity.
    """

    def __init__(self, in_channels: int, hidden: int, classes: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, 32, 1), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 64, 1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, hidden, 1), nn.BatchNorm1d(hidden), nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(2 * hidden, 64), nn.ReLU(), nn.Linear(64, classes))

    def forward(self, points):
        # points: (B, N, C). Symmetric max + mean pooling, so the read-out cannot
        # depend on point order.
        hidden = self.encoder(points.transpose(1, 2))
        pooled = torch.cat([hidden.amax(dim=2), hidden.mean(dim=2)], dim=1)
        return self.head(pooled)


def _robust_extent(xyz: np.ndarray) -> float:
    """Per-cloud scale: median distance from the cloud centroid in 3-D.

    Chosen over the bounding-box diagonal because a handful of stray points move
    a box a lot and a median barely at all. The function cannot read the site, so
    a site cannot be encoded in the divisor.
    """
    centred = xyz - xyz.mean(axis=0, keepdims=True)
    distance = np.linalg.norm(centred, axis=1)
    scale = float(np.median(distance))
    return scale if scale > 1e-6 else 1.0


def _voxel_thin(xyz: np.ndarray, voxel: float, rng: np.random.Generator) -> np.ndarray:
    """Density standardisation: at most one point per voxel, chosen at random.

    Real acquisition density differs systematically between sites, so leaving it
    in the representation would hand the classifier a second, independent site
    cue that has nothing to do with tree geometry.
    """
    keys = np.floor((xyz - xyz.min(axis=0)) / voxel).astype(np.int64)
    linear = (keys[:, 0] * 73856093) ^ (keys[:, 1] * 19349663) ^ (keys[:, 2] * 83492791)
    order = np.lexsort((rng.random(len(xyz)), linear))
    sorted_keys = linear[order]
    first = np.ones(len(order), dtype=bool)
    first[1:] = sorted_keys[1:] != sorted_keys[:-1]
    return order[first]


def _resample(xyz: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    if len(xyz) == 0:
        raise ValueError('cannot resample an empty cloud')
    if len(xyz) >= count:
        return xyz[rng.choice(len(xyz), size=count, replace=False)]
    extra = rng.choice(len(xyz), size=count - len(xyz), replace=True)
    return np.concatenate([xyz, xyz[extra]], axis=0)


def _random_xy_rotation(xyz: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    angle = rng.random() * 2.0 * math.pi
    cos, sin = math.cos(angle), math.sin(angle)
    rotation = np.array([[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]],
                        dtype=np.float32)
    return xyz @ rotation.T


def build_feature_block(xyz: np.ndarray, representation: str,
                        rng: np.random.Generator, points: int) -> tuple[np.ndarray, np.ndarray]:
    """Features for one tree under one representation, plus its metric metadata.

    Returns ``(geometry_and_membership, metadata)``. Metadata is kept separate so
    a common training-pool standardisation can be fitted over every channel of
    every arm in exactly the same way; nothing about the arm changes the
    preprocessing, only the values that go into it.

    ``V1_RAW`` reproduces the frozen V1 input contract: mean-centred local metric
    XYZ divided by the research-wide constant 10, plus
    ``current_candidate_membership`` = 1 for a good tree. A 12 m tree stays 12 m.

    ``V2_*`` are candidate-centred and per-cloud scaled so the coordinates carry
    no absolute size. ``*_NO_META`` withholds the explicit metric channel.
    ``V2_ANISO_*`` additionally divides each axis by its own robust extent, which
    removes aspect ratio too; that is the declared ablation, not a proposal.
    """
    xyz = np.asarray(xyz, dtype=np.float32)
    meta = _raw_meta(xyz)
    if representation == 'V1_RAW':
        working = _resample(xyz, points, rng)
        local = (working - working.mean(axis=0, keepdims=True)) \
            / np.float32(METRIC_SCALE_CONSTANT)
        membership = np.ones((working.shape[0], 1), dtype=np.float32)
        return np.concatenate([local, membership], axis=1), meta

    if representation not in ALL_REPRESENTATIONS:
        raise ValueError(f'unknown representation {representation!r}')

    if len(xyz) > 8:
        steps = np.linalg.norm(np.diff(xyz[::7], axis=0), axis=1)
        voxel = float(np.median(steps)) if steps.size else 0.0
    else:
        voxel = 0.0
    thinned = _voxel_thin(xyz, max(voxel, 1e-3), rng) if voxel > 0 else np.arange(len(xyz))
    keep = rng.random(len(thinned)) >= 0.10
    if keep.sum() < 16:
        keep = np.ones(len(thinned), dtype=bool)
    working = xyz[thinned[keep]]

    working = _random_xy_rotation(working, rng)
    # Density standardisation changes the point count on purpose; the fixed-input
    # contract is restored here so every representation feeds the same N.
    working = _resample(working, points, rng)
    centre = working.mean(axis=0, keepdims=True)
    centred = working - centre
    if representation.startswith('V2_ANISO'):
        scale = np.median(np.abs(centred), axis=0)
        scale[scale < 1e-6] = 1.0
    else:
        scale = np.float32(_robust_extent(working))
    normalized = (centred / scale.astype(np.float32)).astype(np.float32)
    membership = np.ones((normalized.shape[0], 1), dtype=np.float32)
    channels = [normalized]
    if not representation.endswith('NO_META'):
        channels.append(np.broadcast_to(
            meta.reshape(1, -1).astype(np.float32),
            (normalized.shape[0], META_DIM)))
    channels.append(membership)
    return np.concatenate(channels, axis=1), meta


def _raw_meta(xyz: np.ndarray) -> np.ndarray:
    """Explicit global metric metadata: height, crown width, log point density.

    These are the quantities an operator would read off a report, so putting them
    in a named channel keeps them auditable. The alternative - letting the network
    infer them from the coordinates - is what makes them a hidden shortcut.
    """
    span = xyz.max(axis=0) - xyz.min(axis=0)
    volume = max(float(np.prod(span)), 1e-6)
    density = float(len(xyz)) / volume
    return np.array([span[2], min(span[0], span[1]), math.log(max(density, 1e-9))],
                    dtype=np.float32)


def _prepare_pool(rows, points: int, seed: int, progress=None) -> dict:
    """Read every sampled tree once and cache its resampled points and metadata."""
    pool = []
    for index, row in enumerate(rows):
        xyz = ply.read_xyz(row.path, dtype=np.float32)
        rng = np.random.default_rng(np.random.SeedSequence([seed, index]))
        resampled = _resample(xyz, points, rng)
        pool.append({'key': row.key, 'site': row.site, 'partition': row.partition,
                     'tree_id': row.tree_id, 'xyz': resampled,
                     'meta_raw': _raw_meta(xyz)})
        if progress is not None and (index + 1) % 100 == 0:
            progress(index + 1, len(rows))
    return {'trees': pool}


def _split_pool(pool: dict, seed: int) -> dict:
    """Site-stratified tree split: train / validation / test, no tree in two."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, 7]))
    assignment = {}
    for site in sorted({t['site'] for t in pool['trees']}):
        keys = sorted(t['key'] for t in pool['trees'] if t['site'] == site)
        order = rng.permutation(len(keys))
        n_train = int(round(cfg.SITE_CLASSIFIER_SPLIT[0] * len(keys)))
        n_val = int(round(cfg.SITE_CLASSIFIER_SPLIT[1] * len(keys)))
        for position, index in enumerate(order):
            part = ('train' if position < n_train
                    else 'validation' if position < n_train + n_val else 'test')
            assignment[keys[int(index)]] = part
    return assignment


def _materialise(pool: dict, assignment: dict, representation: str, seed: int,
                 points: int) -> dict:
    rng = np.random.default_rng(np.random.SeedSequence([seed, 11]))
    blocks, sites, keys, parts = [], [], [], []
    for tree in pool['trees']:
        block, _meta = build_feature_block(tree['xyz'], representation, rng, points)
        blocks.append(block.astype(np.float32))
        sites.append(tree['site'])
        keys.append(tree['key'])
        parts.append(assignment[tree['key']])
    return {'features': blocks, 'site': sites, 'keys': keys, 'partition': parts}


def _fit_standardisation(materialised: dict) -> dict:
    """Per-channel mean/std over the training trees only.

    This is the single common preprocessing step for every arm. It matters for
    fairness: without it the metric arm enters the network at millimetre scale
    while the normalised arm enters at unit scale, and the comparison would be
    about optimiser conditioning rather than about the representation. The
    statistics are fitted on the train split, never on validation or test.
    """
    index = [i for i, part in enumerate(materialised['partition']) if part == 'train']
    stacked = np.concatenate([materialised['features'][i] for i in index], axis=0)
    mean = stacked.mean(axis=0)
    std = stacked.std(axis=0)
    std[std < 1e-6] = 1.0
    return {'mean': mean.astype(np.float32), 'std': std.astype(np.float32),
            'fitted_on': 'train split only',
            'train_points': int(stacked.shape[0])}


def _apply_standardisation(materialised: dict, stats: dict) -> dict:
    features = [((block - stats['mean']) / stats['std']).astype(np.float32)
                for block in materialised['features']]
    return {**materialised, 'features': features}


def _collate(materialised: dict, part: str, site_to_index: dict):
    index = [i for i, p in enumerate(materialised['partition']) if p == part]
    features = np.stack([materialised['features'][i] for i in index], axis=0)
    labels = np.array([site_to_index[materialised['site'][i]] for i in index],
                      dtype=np.int64)
    return torch.from_numpy(features), torch.from_numpy(labels)


def train_one(representation: str, materialised: dict, site_to_index: dict,
              seed: int, device: str) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    train_x, train_y = _collate(materialised, 'train', site_to_index)
    val_x, val_y = _collate(materialised, 'validation', site_to_index)
    model = SiteClassifier(train_x.shape[-1], cfg.SITE_CLASSIFIER_HIDDEN,
                           len(site_to_index)).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=cfg.SITE_CLASSIFIER_LR)
    loss_fn = nn.CrossEntropyLoss()
    generator = torch.Generator().manual_seed(seed)
    best = {'epoch': -1, 'val_accuracy': -1.0, 'state': None}
    history = []
    for epoch in range(cfg.SITE_CLASSIFIER_EPOCHS):
        model.train()
        order = torch.randperm(train_x.shape[0], generator=generator)
        total_loss = 0.0
        for start in range(0, len(order), cfg.SITE_CLASSIFIER_BATCH):
            batch = order[start:start + cfg.SITE_CLASSIFIER_BATCH]
            optimiser.zero_grad()
            logits = model(train_x[batch].to(device))
            loss = loss_fn(logits, train_y[batch].to(device))
            loss.backward()
            optimiser.step()
            total_loss += float(loss.detach()) * len(batch)
        model.eval()
        with torch.no_grad():
            val_logits = model(val_x.to(device))
            val_accuracy = float((val_logits.argmax(1).cpu() == val_y).float().mean())
        history.append({'epoch': epoch, 'train_loss': total_loss / len(order),
                        'val_accuracy': val_accuracy})
        if val_accuracy > best['val_accuracy'] + 1e-9:
            best = {'epoch': epoch, 'val_accuracy': val_accuracy,
                    'state': {k: v.detach().clone() for k, v in model.state_dict().items()}}
    if best['state'] is not None:
        model.load_state_dict(best['state'])
    model.eval()
    return {'model': model, 'history': history, 'best': best,
            'parameters': int(sum(p.numel() for p in model.parameters()))}


def evaluate(model, materialised: dict, site_to_index: dict, part: str,
             device: str) -> dict:
    features, labels = _collate(materialised, part, site_to_index)
    with torch.no_grad():
        logits = model(features.to(device))
        predicted = logits.argmax(1).cpu()
    names = [name for name, _ in sorted(site_to_index.items(), key=lambda kv: kv[1])]
    classes = len(names)
    confusion = np.zeros((classes, classes), dtype=np.int64)
    for truth, guess in zip(labels.numpy(), predicted.numpy(), strict=True):
        confusion[int(truth), int(guess)] += 1
    total = int(confusion.sum())
    accuracy = float(np.trace(confusion) / total) if total else 0.0
    per_site_recall = {}
    f1s = []
    for index, name in enumerate(names):
        support = int(confusion[index].sum())
        recall = float(confusion[index, index] / support) if support else 0.0
        precision = (float(confusion[index, index] / confusion[:, index].sum())
                     if confusion[:, index].sum() else 0.0)
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        per_site_recall[name] = recall
        f1s.append(f1)
    return {'partition': part, 'examples': total, 'site_accuracy': accuracy,
            'macro_f1': float(np.mean(f1s)), 'per_site_recall': per_site_recall,
            'confusion_matrix': confusion.tolist(), 'class_order': names,
            'random_baseline_accuracy': 1.0 / classes,
            'majority_baseline_accuracy':
                float(np.bincount(labels.numpy(), minlength=classes).max() / total)
                if total else 0.0}


def run(points: int = None, trees_per_site: int = None, seed: int = None,
        device: str = None, verbose: bool = True,
        representations=None) -> dict:
    started = time.time()
    points = int(points or cfg.SITE_CLASSIFIER_POINTS)
    trees_per_site = int(trees_per_site or cfg.SITE_CLASSIFIER_TREES_PER_SITE)
    seed = int(seed if seed is not None else cfg.SEED)
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    representations = tuple(representations or ALL_REPRESENTATIONS)

    rows = data.discover_tree_instances(partitions=('train',))
    sample = data.sample_instances(rows, trees_per_site, seed)
    present = sorted({row.site for row in sample})
    if verbose:
        print(f'[A] {len(sample)} good trees over {present} '
              f'({device}, {points} points)', flush=True)

    pool = _prepare_pool(sample, points, seed,
                         progress=(lambda d, t: print(f'  read {d}/{t}', flush=True))
                         if verbose else None)
    assignment = _split_pool(pool, seed)
    site_to_index = {site: index for index, site in enumerate(present)}

    results = {}
    for representation in representations:
        materialised = _materialise(pool, assignment, representation, seed, points)
        stats = _fit_standardisation(materialised)
        materialised = _apply_standardisation(materialised, stats)
        trained = train_one(representation, materialised, site_to_index, seed, device)
        entry = {
            'representation': representation,
            'in_channels': int(materialised['features'][0].shape[1]),
            'parameters': trained['parameters'],
            'best_epoch': trained['best']['epoch'],
            'best_validation_accuracy': trained['best']['val_accuracy'],
            'standardisation': {'channels': int(len(stats['mean'])),
                                'fitted_on': stats['fitted_on'],
                                'train_points': stats['train_points']},
            'history': trained['history'],
        }
        for part in ('train', 'validation', 'test'):
            entry[part] = evaluate(trained['model'], materialised, site_to_index,
                                   part, device)
        entry['test_site_accuracy'] = entry['test']['site_accuracy']
        entry['test_macro_f1'] = entry['test']['macro_f1']
        results[representation] = entry
        if verbose:
            print(f'  {representation:26} test accuracy '
                  f'{entry["test_site_accuracy"]:.4f}  macro F1 '
                  f'{entry["test_macro_f1"]:.4f}  ({entry["in_channels"]} ch)', flush=True)

    counts = {}
    for key in assignment:
        site = key.split('|')[0]
        counts.setdefault(site, {'train': 0, 'validation': 0, 'test': 0})
        counts[site][assignment[key]] += 1

    return {
        'schema': cfg.SCHEMA,
        'task': 'SITE_FINGERPRINT_TEST',
        'config_digest': cfg.config_digest(),
        'seed': seed, 'points': points, 'trees_per_site_requested': trees_per_site,
        'trees_used': len(pool['trees']), 'sites': present,
        'absent_sites': list(cfg.ABSENT_SITES),
        'split_counts': counts,
        'meta_channels': list(META_CHANNELS),
        'representations': results,
        'seconds': round(time.time() - started, 2),
        'device': device,
    }


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--points', type=int, default=cfg.SITE_CLASSIFIER_POINTS)
    parser.add_argument('--trees-per-site', type=int,
                        default=cfg.SITE_CLASSIFIER_TREES_PER_SITE)
    parser.add_argument('--seed', type=int, default=cfg.SEED)
    parser.add_argument('--quick', action='store_true')
    args = parser.parse_args()
    points = 256 if args.quick else args.points
    trees = 24 if args.quick else args.trees_per_site
    payload = run(points=points, trees_per_site=trees, seed=args.seed)
    cfg.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = '_QUICK' if args.quick else ''
    out = cfg.REPORT_DIR / f'SITE_FINGERPRINT{suffix}.json'
    out.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    print(f'wrote {out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
