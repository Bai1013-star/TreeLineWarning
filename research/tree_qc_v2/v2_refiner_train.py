"""Task C - MINI_REFINER_BAKEOFF.

Three architectures, one identical protocol. Everything except the architecture
is shared: the same train, validation and test candidates; the same ROI points;
the same proposal pairs; the same sampler; the same augmentation; the same point
target; the same seed; the same epoch budget; the same early-stop policy; the same
optimiser family; and the same evaluation code.

Two gates decide the outcome, and neither can be satisfied by copying the input:

``IDENTITY_COPY_GATE``
    A refiner that returns ``current_mask`` unchanged scores a perfect current IoU
    and fails every change-region metric. Change-region F1, added-point F1 and
    removed-point F1 are therefore computed on the points that actually moved, and
    a model whose change-region F1 stays at zero is recorded as an identity
    copier no matter how good its overall IoU looks.

``GOOD_PRESERVATION_TEST``
    On proposals that were already good, the refiner must not degrade them. The
    report carries the mean delta IoU, the degraded fraction and the severe
    degradation count, because an architecture that repairs broken trees by
    breaking good ones is not a candidate.

The test site is never used to choose anything. Early stopping reads the held-in
validation split only.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np
import torch
import v2_config as cfg
import v2_models as models
import v2_refiner_data as refdata
from torch import nn

#: Loss weights for the multitask model. Declared, not tuned on test.
OFFSET_LOSS_WEIGHT = 0.5
QUALITY_LOSS_WEIGHT = 0.5
#: Decision threshold applied to the membership probability. Fixed for all models.
DECISION_THRESHOLD = 0.5
#: A proposal is "already good" above this IoU, per the declared gate.
GOOD_IOU_THRESHOLD = cfg.GOOD_IOU_THRESHOLD
#: Numerical cap on the per-example positive-class weight. See ``membership_loss``.
POSITIVE_WEIGHT_CAP = 50.0


@dataclass
class Split:
    """Which targets go where. Splits are by tree, and the cross-site fold is by site."""

    train: list[str] = field(default_factory=list)
    validation: list[str] = field(default_factory=list)
    test: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {'train': len(self.train), 'validation': len(self.validation),
                'test': len(self.test)}


def split_by_tree(index: dict, test_site: str | None, seed: int,
                  train_fraction: float = 0.7, validation_fraction: float = 0.15) -> Split:
    """Split targets, by tree, optionally holding one whole site out for test.

    When ``test_site`` is given, that site contributes only to the test split and
    the remaining sites are split into train and validation. When it is ``None``,
    every site is split internally and the per-site splits are pooled.
    """
    split = Split()
    sites = sorted({row['site'] for row in index['targets']})
    for site in sites:
        rows = sorted((row for row in index['targets'] if row['site'] == site),
                      key=lambda row: row['target_id'])
        if test_site is not None and site == test_site:
            split.test.extend(row['target_id'] for row in rows)
            continue
        order = np.random.default_rng(
            np.random.SeedSequence([seed, int.from_bytes(
                __import__('hashlib').sha256(site.encode()).digest()[:8], 'little')])
        ).permutation(len(rows))
        if test_site is None:
            n_train = int(round(train_fraction * len(rows)))
            n_validation = int(round(validation_fraction * len(rows)))
            for position, index_in_rows in enumerate(order):
                target = rows[int(index_in_rows)]['target_id']
                if position < n_train:
                    split.train.append(target)
                elif position < n_train + n_validation:
                    split.validation.append(target)
                else:
                    split.test.append(target)
        else:
            n_train = int(round(train_fraction / (train_fraction + validation_fraction)
                                * len(rows)))
            for position, index_in_rows in enumerate(order):
                target = rows[int(index_in_rows)]['target_id']
                if position < n_train:
                    split.train.append(target)
                else:
                    split.validation.append(target)
    return Split(sorted(split.train), sorted(split.validation), sorted(split.test))


def split_within_site(index: dict, site: str, seed: int,
                      train_fraction: float = 0.7,
                      validation_fraction: float = 0.15) -> Split:
    """A held-in split of one site, for the in-site diagnostic configuration.

    The cross-site folds answer "does this transfer to a forest the model never
    saw". This answers the different, prior question: "can any of these
    architectures refine within one forest at all". If the in-site numbers are also
    flat, the problem is the task or the budget, not the domain gap; if they are
    good, the gap is the domain and that is worth reporting separately. Only the
    named site contributes, so the other site is untouched here.
    """
    rows = sorted((row for row in index['targets'] if row['site'] == site),
                  key=lambda row: row['target_id'])
    order = np.random.default_rng(np.random.SeedSequence([seed, 99])).permutation(len(rows))
    n_train = int(round(train_fraction * len(rows)))
    n_validation = int(round(validation_fraction * len(rows)))
    split = Split()
    for position, index_in_rows in enumerate(order):
        target = rows[int(index_in_rows)]['target_id']
        if position < n_train:
            split.train.append(target)
        elif position < n_train + n_validation:
            split.validation.append(target)
        else:
            split.test.append(target)
    return Split(sorted(split.train), sorted(split.validation), sorted(split.test))


def build_examples(index: dict, targets: list[str], kinds=None) -> list[refdata.Example]:
    """One example per (target, proposal) row, filtered to the requested kinds."""
    wanted = set(targets)
    out = []
    for row in index['targets']:
        if row['target_id'] not in wanted:
            continue
        for proposal in row['rows']:
            if kinds is not None and proposal['proposal_kind'] not in kinds:
                continue
            out.append(refdata.Example(
                key=f"{row['target_id']}|{proposal['proposal_id']}",
                site=row['site'], target_id=row['target_id'],
                proposal_id=proposal['proposal_id'],
                proposal_kind=proposal['proposal_kind'],
                corruption=proposal.get('corruption', 'NONE'),
                severity=proposal.get('severity', 'NONE'),
                split='', current_iou=float(proposal['iou'])))
    return out


def membership_loss(logits: torch.Tensor, target: torch.Tensor,
                    current: torch.Tensor) -> torch.Tensor:
    """Balanced binary cross-entropy, with a bounded positive weight.

    A proposal that is 95 percent wrong should not be minimised by agreeing with
    itself, so the positive class is weighted by the negative-to-positive ratio.
    That ratio is computed **per example and clamped**, and the total weight mass is
    normalised, because an unclamped ratio on an example whose truth is a few
    hundred points inside a 300 000-point ROI reaches 10^3 and the loss diverges to
    NaN on the first epoch. The clamp is a numerical guard, not a tuning knob: it
    binds only on examples the balanced loss would otherwise let dominate the batch.
    """
    del current
    device = logits.device
    target = target.to(device)
    positive = target.sum(dim=1)
    total = target.shape[1]
    # Laplace-smoothed ratio: finite when an example has no target points at all.
    ratio = (total - positive + 1.0) / (positive + 1.0)
    weight = ratio.clamp(1.0, POSITIVE_WEIGHT_CAP)
    loss = nn.functional.binary_cross_entropy_with_logits(
        logits, target, reduction='none')
    weights = torch.where(target > 0.5, weight.unsqueeze(1), torch.ones_like(target))
    return (loss * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)


def total_loss(outputs: dict, batch: dict, model) -> tuple:
    device = outputs['logits'].device
    target = batch['target'].to(device)
    loss = membership_loss(outputs['logits'], target, batch['current'].to(device)).mean()
    parts = {'membership': float(loss.detach())}
    if model.auxiliary:
        active = batch['offset_active'].to(device)
        if float(active.sum()) > 0:
            difference = (outputs['offset'] - batch['offset'].to(device)).abs().sum(dim=2)
            offset_loss = (difference * active).sum() / active.sum().clamp(min=1.0)
        else:
            offset_loss = torch.zeros((), device=device)
        quality_loss = nn.functional.mse_loss(
            outputs['quality'], batch['quality'].to(device))
        parts['offset'] = float(offset_loss.detach())
        parts['quality'] = float(quality_loss.detach())
        loss = loss + OFFSET_LOSS_WEIGHT * offset_loss + QUALITY_LOSS_WEIGHT * quality_loss
    return loss, parts


def region_metrics(predicted: np.ndarray, target: np.ndarray,
                   current: np.ndarray) -> dict:
    """Full IoU plus the change-region metrics that defeat identity copying."""
    predicted = np.asarray(predicted, dtype=bool)
    target = np.asarray(target, dtype=bool)
    current = np.asarray(current, dtype=bool)

    def scores(left, right) -> tuple:
        intersection = int(np.count_nonzero(left & right))
        union = int(np.count_nonzero(left | right))
        precision = intersection / max(int(left.sum()), 1)
        recall = intersection / max(int(right.sum()), 1)
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        return intersection / union if union else 0.0, precision, recall, f1

    iou, precision, recall, f1 = scores(predicted, target)
    current_iou, current_precision, current_recall, _ = scores(current, target)

    changed = predicted != current
    added = predicted & ~current
    removed = current & ~predicted
    added_truth = target & ~current
    removed_truth = current & ~target

    change_iou, change_precision, change_recall, change_f1 = scores(
        predicted[changed], target[changed])
    add_iou, add_precision, add_recall, add_f1 = scores(added, added_truth)
    remove_iou, remove_precision, remove_recall, remove_f1 = scores(removed, removed_truth)

    return {
        'current_iou': float(current_iou),
        'current_precision': float(current_precision),
        'current_recall': float(current_recall),
        'refined_iou': float(iou),
        'refined_precision': float(precision),
        'refined_recall': float(recall),
        'refined_f1': float(f1),
        'delta_iou': float(iou - current_iou),
        'changed_points': int(changed.sum()),
        'change_region_iou': float(change_iou),
        'change_region_precision': float(change_precision),
        'change_region_recall': float(change_recall),
        'change_region_f1': float(change_f1),
        'added_points': int(added.sum()),
        'added_point_precision': float(add_precision),
        'added_point_recall': float(add_recall),
        'added_point_f1': float(add_f1),
        'removed_points': int(removed.sum()),
        'removed_point_precision': float(remove_precision),
        'removed_point_recall': float(remove_recall),
        'removed_point_f1': float(remove_f1),
        'unchanged': bool(not changed.any()),
    }


def train_one(model_key: str, index: dict, split: Split, points: int, epochs: int,
              seed: int, device: str, verbose: bool = True,
              kinds=None, width: int = 32) -> dict:
    started = time.time()
    train_examples = build_examples(index, split.train, kinds)
    validation_examples = build_examples(index, split.validation, kinds)
    test_examples = build_examples(index, split.test, kinds)
    if not train_examples or not validation_examples:
        raise ValueError('a split produced no examples')

    train_set = refdata.RefinerDataset(index, train_examples, points, seed, augment=True)
    validation_set = refdata.FixedSubsetDataset(index, validation_examples, points,
                                                seed + 1)
    validation_loader = torch.utils.data.DataLoader(
        validation_set, batch_size=cfg.REFINER_BATCH, shuffle=False,
        collate_fn=refdata.collate_validation)

    torch.manual_seed(seed)
    np.random.seed(seed)
    model = models.build_model(model_key, len(refdata.FEATURE_CHANNELS), width).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=cfg.REFINER_LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs)
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=cfg.REFINER_BATCH, shuffle=True, collate_fn=refdata.collate,
        drop_last=False)

    # The quality target is the IoU the refined mask actually achieves on the
    # training example, taken from the library rows. It is available at training
    # time only, which is the point: at inference the head predicts it.
    quality_by_key = {f"{row['target_id']}|{proposal['proposal_id']}":
                      float(proposal['iou'])
                      for row in index['targets'] for proposal in row['rows']}

    best = {'epoch': -1, 'validation_loss': float('inf'), 'state': None}
    history = []
    for epoch in range(epochs):
        epoch_started = time.time()
        model.train()
        running, parts_running = 0.0, {}
        for batch in train_loader:
            batch['quality'] = torch.tensor(
                [quality_by_key.get(name, 0.0) for name in batch['filename']],
                dtype=torch.float32)
            optimiser.zero_grad()
            outputs = model(batch['features'].to(device))
            loss, parts = total_loss(outputs, batch, model)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimiser.step()
            running += float(loss.detach()) * len(batch['filename'])
            for key, value in parts.items():
                parts_running[key] = parts_running.get(key, 0.0) + value * len(batch['filename'])
        scheduler.step()
        model.eval()
        validation_total = 0.0
        with torch.no_grad():
            for batch in validation_loader:
                outputs = model(batch['features'].to(device))
                # Early stopping reads the membership loss alone, for every model.
                # It is the primary objective and the only one all three share, so
                # the stopping rule cannot favour MODEL_C for having extra heads or
                # MODEL_A for lacking them. The auxiliary losses still train, they
                # just do not select.
                loss = membership_loss(outputs['logits'], batch['target'],
                                       batch['current']).mean()
                validation_total += float(loss.detach()) * len(batch['filename'])
        train_loss = running / max(len(train_set), 1)
        validation_loss = validation_total / max(len(validation_set), 1)
        history.append({'epoch': epoch, 'train_loss': train_loss,
                        'validation_loss': validation_loss,
                        'epoch_seconds': round(time.time() - epoch_started, 1),
                        'parts': {k: v / max(len(train_set), 1)
                                  for k, v in parts_running.items()}})
        if verbose:
            # Every epoch, with its wall clock. A bakeoff that has stalled and one
            # that is merely slow must be distinguishable from the log alone, and an
            # epoch count without a duration cannot tell them apart.
            print(f'    [{model.name}] epoch {epoch:3d} train {train_loss:.4f} '
                  f'val {validation_loss:.4f} '
                  f'({history[-1]["epoch_seconds"]:.1f}s)', flush=True)
        if validation_loss < best['validation_loss'] - cfg.REFINER_EARLY_STOP_MIN_DELTA:
            best = {'epoch': epoch, 'validation_loss': validation_loss,
                    'state': {k: v.detach().clone() for k, v in model.state_dict().items()}}
        elif epoch - best['epoch'] >= cfg.REFINER_EARLY_STOP_PATIENCE:
            if verbose:
                print(f'    [{model.name}] early stop at epoch {epoch} '
                      f'(best {best["epoch"]})', flush=True)
            break
    if best['state'] is not None:
        model.load_state_dict(best['state'])

    trained = {
        'model': model.name, 'model_key': model_key,
        'parameters': models.parameter_count(model),
        'in_channels': len(refdata.FEATURE_CHANNELS),
        'points': points, 'epochs_budget': epochs, 'epochs_run': len(history),
        'best_epoch': best['epoch'], 'best_validation_loss': best['validation_loss'],
        'history': history,
        'threshold': DECISION_THRESHOLD,
        'examples': {'train': len(train_examples),
                     'validation': len(validation_examples),
                     'test': len(test_examples)},
        'validation_used_for_selection': (
            'a deterministic bounded sample per ROI, scored on the membership loss '
            'alone so all three architectures share one stopping rule; all reported '
            'metrics use the full ROI'),
        'seconds': round(time.time() - started, 2),
        'device': device,
    }
    return trained, model


def measure_resources(model, device: str, points: int = None,
                      repeats: int = 5) -> dict:
    """Parameter count, GPU peak memory, CPU RAM and single-ROI inference time.

    These are not product gates in this round. They are recorded because the
    eventual target is a Windows desktop build, where a 24 GB accelerator cannot be
    assumed, so an architecture that only works at one scale is a worse candidate
    even if its IoU is equal.
    """
    import time as _time
    try:
        import psutil
    except ImportError:
        psutil = None
    points = int(points or cfg.REFINER_POINTS)
    model.eval()
    sample = torch.randn(1, points, len(refdata.FEATURE_CHANNELS), device=device)
    with torch.no_grad():
        model(sample)
    if device == 'cuda':
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    process = psutil.Process() if psutil is not None else None
    before = process.memory_info().rss if process else None
    started = _time.perf_counter()
    with torch.no_grad():
        for _ in range(repeats):
            model(sample)
    if device == 'cuda':
        torch.cuda.synchronize()
    single = (_time.perf_counter() - started) / repeats
    after = process.memory_info().rss if process else None
    return {
        'parameters': models.parameter_count(model),
        'device': device,
        'points': points,
        'single_roi_inference_seconds': round(single, 4),
        'cuda_peak_memory_mb': (round(torch.cuda.max_memory_allocated() / 1024 ** 2, 1)
                                if device == 'cuda' else None),
        'process_rss_delta_mb': (round((after - before) / 1024 ** 2, 1)
                                 if before is not None else None),
        'cpu_inference_supported': True,
    }


def cpu_inference_probe(model, points: int = None, repeats: int = 2) -> dict:
    """One forward on CPU, to prove the CPU-only path exists."""
    import copy
    import time as _time
    points = int(points or cfg.REFINER_POINTS)
    clone = copy.deepcopy(model).to('cpu').eval()
    sample = torch.randn(1, points, len(refdata.FEATURE_CHANNELS))
    started = _time.perf_counter()
    with torch.no_grad():
        outputs = clone(sample)
    elapsed = _time.perf_counter() - started
    del outputs
    return {'points': points, 'cpu_single_roi_inference_seconds': round(elapsed, 4),
            'supported': True}


def _iou(probability: np.ndarray, target: np.ndarray, current: np.ndarray) -> float:
    del current
    predicted = probability > DECISION_THRESHOLD
    target = target > 0.5
    intersection = int(np.count_nonzero(predicted & target))
    union = int(np.count_nonzero(predicted | target))
    return float(intersection / union) if union else 0.0


class FullRoiDataset(torch.utils.data.Dataset):
    """Every point of one ROI, with no sampling and no augmentation.

    The reported IoU has to be the IoU of the mask the model implies over the whole
    region. A 2048-point sample would decide the verdict for a 300 000-point ROI,
    and the change-region metrics would be computed on a sample of the change.
    """

    def __init__(self, index: dict, examples: list[refdata.Example],
                 max_points: int | None = None):
        self.by_target = {row['target_id']: row for row in index['targets']}
        self.examples = examples
        self.max_points = int(max_points or cfg.REFINER_EVAL_MAX_POINTS)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, item: int) -> dict:
        example = self.examples[item]
        cluster = self.by_target[example.target_id]
        xyz = np.asarray(cluster['xyz'], dtype=np.float32)
        gt = np.asarray(cluster['masks']['gt'], dtype=bool)
        current = refdata.proposal_mask(cluster, example.proposal_id)
        stored = int(len(xyz))
        stride = 1
        if stored > self.max_points:
            # One deterministic, spatially-sorted stride. Sorting by height first
            # keeps the vertical profile, which is the same rule the training
            # sampler and the library cap already use.
            order = np.lexsort((xyz[:, 1], xyz[:, 0], xyz[:, 2]))
            stride = int(np.ceil(stored / self.max_points))
            keep = np.sort(order[::stride][:self.max_points])
            xyz, gt, current = xyz[keep], gt[keep], current[keep]
        frame = refdata.candidate_frame(xyz[current]) if current.any() else \
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
            'current_iou_library': float(example.current_iou),
            'roi_points': int(len(xyz)),
            'roi_points_stored': stored,
            'roi_stride': stride,
        }


def full_collate(batch: list[dict]) -> dict:
    return {
        'features': [item['features'] for item in batch],
        'target': [item['target'] for item in batch],
        'current': [item['current'] for item in batch],
        'filename': [item['filename'] for item in batch],
        'site': [item['site'] for item in batch],
        'target_id': [item['target_id'] for item in batch],
        'proposal_kind': [item['proposal_kind'] for item in batch],
        'corruption': [item['corruption'] for item in batch],
        'severity': [item['severity'] for item in batch],
        'current_iou_library': [item['current_iou_library'] for item in batch],
        'roi_points': [item['roi_points'] for item in batch],
        'roi_points_stored': [item['roi_points_stored'] for item in batch],
        'roi_stride': [item['roi_stride'] for item in batch],
    }


def evaluate_full_roi(model, index: dict, targets: list[str], kinds, device: str,
                      seed: int = 0, batch_size: int = 8, max_points: int = None,
                      verbose: bool = False) -> list[dict]:
    """Score every example over its whole ROI (up to the declared cap), batched."""
    examples = build_examples(index, targets, kinds)
    if not examples:
        return []
    dataset = FullRoiDataset(index, examples, max_points=max_points)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                         collate_fn=full_collate)
    model.eval()
    rows = []
    done = 0
    del seed
    for batch in loader:
        # Manual right-padding to the longest ROI in the batch. Every operation in
        # the backbones is point-wise or neighbourhood-local, and the neighbourhood
        # searches clamp to the point count, so a padded tail cannot influence the
        # real points; the results are read back only up to each ROI's own length.
        lengths = batch['roi_points']
        width = max(lengths)
        stacked = torch.zeros(len(lengths), width,
                              batch['features'][0].shape[-1], dtype=torch.float32)
        for position, item in enumerate(batch['features']):
            stacked[position, :lengths[position]] = item
        with torch.no_grad():
            outputs = model(stacked.to(device))
            probability = torch.sigmoid(outputs['logits']).cpu().numpy()
        for position in range(len(batch['filename'])):
            n = batch['roi_points'][position]
            predicted = probability[position][:n] > DECISION_THRESHOLD
            gt = batch['target'][position].numpy()[:n] > 0.5
            current = batch['current'][position].numpy()[:n] > 0.5
            metrics = region_metrics(predicted, gt, current)
            metrics.update({
                'key': batch['filename'][position],
                'site': batch['site'][position],
                'target_id': batch['target_id'][position],
                'proposal_kind': batch['proposal_kind'][position],
                'corruption': batch['corruption'][position],
                'severity': batch['severity'][position],
                'roi_points': batch['roi_points'][position],
                'roi_points_stored': batch['roi_points_stored'][position],
                'roi_stride': batch['roi_stride'][position],
                'library_iou': batch['current_iou_library'][position],
            })
            if model.auxiliary:
                metrics['predicted_quality'] = float(outputs['quality'][position].cpu())
            rows.append(metrics)
        done += len(batch['filename'])
        if verbose and done % 200 < batch_size:
            print(f'      scored {done}/{len(examples)}', flush=True)
    return rows


def summarise(rows: list[dict], label: str) -> dict:
    if not rows:
        return {'label': label, 'rows': 0}
    delta = np.array([row['delta_iou'] for row in rows])
    current = np.array([row['current_iou'] for row in rows])
    refined = np.array([row['refined_iou'] for row in rows])
    good = current >= GOOD_IOU_THRESHOLD
    bad = ~good
    out = {
        'label': label, 'rows': len(rows),
        'current_iou_mean': float(current.mean()),
        'current_iou_median': float(np.median(current)),
        'refined_iou_mean': float(refined.mean()),
        'refined_iou_median': float(np.median(refined)),
        'delta_iou_mean': float(delta.mean()),
        'delta_iou_median': float(np.median(delta)),
        'improved_fraction': float((delta > 1e-9).mean()),
        'unchanged_fraction': float((np.abs(delta) <= 1e-9).mean()),
        'degraded_fraction': float((delta < -1e-9).mean()),
        'improvement_gt_0.02': float((delta > 0.02).mean()),
        'degradation_gt_0.02': float((delta < -0.02).mean()),
        'refined_precision_mean': float(np.mean([r['refined_precision'] for r in rows])),
        'refined_recall_mean': float(np.mean([r['refined_recall'] for r in rows])),
        'change_region_f1_mean': float(np.mean([r['change_region_f1'] for r in rows])),
        'change_region_recall_mean': float(np.mean([r['change_region_recall'] for r in rows])),
        'added_point_f1_mean': float(np.mean([r['added_point_f1'] for r in rows])),
        'removed_point_f1_mean': float(np.mean([r['removed_point_f1'] for r in rows])),
        'identity_copy_fraction': float(np.mean([r['unchanged'] for r in rows])),
        'good_proposals': {
            'rows': int(good.sum()),
            'mean_delta_iou': float(delta[good].mean()) if good.any() else None,
            'median_delta_iou': float(np.median(delta[good])) if good.any() else None,
            'degraded_fraction': float((delta[good] < -1e-9).mean()) if good.any() else None,
            'degraded_fraction_gt_0.05':
                float((delta[good] < -cfg.GOOD_DEGRADED_IOU_DROP).mean())
                if good.any() else None,
            'severe_degradation_count':
                int((delta[good] < -cfg.GOOD_SEVERE_IOU_DROP).sum()) if good.any() else 0,
            'mean_current_iou': float(current[good].mean()) if good.any() else None,
            'mean_refined_iou': float(refined[good].mean()) if good.any() else None,
        },
        'bad_proposals': {
            'rows': int(bad.sum()),
            'mean_delta_iou': float(delta[bad].mean()) if bad.any() else None,
            'median_delta_iou': float(np.median(delta[bad])) if bad.any() else None,
            'improved_fraction': float((delta[bad] > 1e-9).mean()) if bad.any() else None,
            'improvement_gt_0.02': float((delta[bad] > 0.02).mean()) if bad.any() else None,
            'mean_current_iou': float(current[bad].mean()) if bad.any() else None,
            'mean_refined_iou': float(refined[bad].mean()) if bad.any() else None,
        },
    }
    return out


def by_group(rows: list[dict], key: str) -> dict:
    out = {}
    for value in sorted({str(row.get(key, 'NONE')) for row in rows}):
        group = [row for row in rows if str(row.get(key, 'NONE')) == value]
        delta = np.array([row['delta_iou'] for row in group])
        out[value] = {
            'rows': len(group),
            'current_iou_mean': float(np.mean([r['current_iou'] for r in group])),
            'refined_iou_mean': float(np.mean([r['refined_iou'] for r in group])),
            'delta_iou_mean': float(delta.mean()),
            'delta_iou_median': float(np.median(delta)),
            'improved_fraction': float((delta > 1e-9).mean()),
            'identity_copy_fraction': float(np.mean([r['unchanged'] for r in group])),
            'change_region_f1_mean': float(np.mean([r['change_region_f1']
                                                    for r in group])),
        }
    return out


def identity_copy_gate(summary: dict) -> dict:
    """Did the model actually change anything, and did the change help?"""
    change_f1 = summary.get('change_region_f1_mean', 0.0) or 0.0
    bad = summary.get('bad_proposals', {}) or {}
    improvement = bad.get('improvement_gt_0.02')
    identity_fraction = summary.get('identity_copy_fraction', 1.0)
    passed = bool(change_f1 >= cfg.IDENTITY_COPY_CHANGE_F1_FLOOR
                  and identity_fraction < 0.9
                  and improvement is not None
                  and improvement >= cfg.IDENTITY_COPY_IMPROVEMENT_IOU_FLOOR)
    return {'verdict': 'PASS' if passed else 'FAIL',
            'change_region_f1_mean': change_f1,
            'change_region_f1_floor': cfg.IDENTITY_COPY_CHANGE_F1_FLOOR,
            'identity_copy_fraction': identity_fraction,
            'bad_improvement_fraction_gt_0.02': improvement,
            'bad_improvement_floor': cfg.IDENTITY_COPY_IMPROVEMENT_IOU_FLOOR}


def good_preservation_gate(summary: dict) -> dict:
    good = summary.get('good_proposals', {}) or {}
    mean_delta = good.get('mean_delta_iou')
    degraded = good.get('degraded_fraction_gt_0.05')
    severe = good.get('severe_degradation_count', 0) or 0
    passed = bool(good.get('rows', 0) > 0 and mean_delta is not None
                  and mean_delta >= -0.02 and degraded is not None
                  and degraded <= 0.25 and severe <= max(1, int(0.05 * good.get('rows', 1))))
    return {'verdict': 'PASS' if passed else 'FAIL',
            'good_rows': good.get('rows', 0),
            'mean_delta_iou': mean_delta,
            'degraded_fraction_gt_0.05': degraded,
            'severe_degradation_count': severe,
            'thresholds': {'mean_delta_iou_min': -0.02,
                           'degraded_fraction_gt_0.05_max': 0.25,
                           'severe_degradation_fraction_max': 0.05}}
