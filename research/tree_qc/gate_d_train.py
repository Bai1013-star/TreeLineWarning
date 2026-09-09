"""Shared training engine for the PointNet and PointNet++ QC classifiers.

Both models use this engine, so they necessarily share the same dataset, split,
corruption configuration, sampler, evaluation metric and threshold rule. Only the
network differs.

The engine refuses to run unless the Supervisor has accepted Gate D
(:mod:`gate_d_lock`). It is written and tested in this round but never executed
against a real training run.
"""
from __future__ import annotations

import hashlib
import json
import platform
import time
from dataclasses import dataclass
from pathlib import Path

import gate_d_dataset as dataset_module
import gate_d_lock as lock
import gate_d_metrics as metrics_module
import numpy as np
from gate_d_provenance import GateDProvenance

TRAINER_VERSION = 'GATE_D_TRAINER_V1'


class TrainerError(RuntimeError):
    pass


@dataclass(frozen=True)
class TrainConfig:
    """Fully serialisable training configuration."""

    model: str
    partition: str = 'train'
    validation_partition: str = 'validation'
    points: int = 4096
    epochs: int = 30
    batch_size: int = 8
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 101
    amp: bool = True
    amp_dtype: str = 'float16'
    workers: int = 0
    grad_clip: float = 1.0
    early_stop_patience: int = 8
    early_stop_min_delta: float = 1e-4
    good_false_review_budget: float = metrics_module.DEFAULT_GOOD_FALSE_REVIEW_BUDGET
    max_train_candidates: int | None = None
    max_validation_candidates: int | None = None
    repeats: int = 1
    seed_base: int = 0
    cache_root: str = str(dataset_module.DEFAULT_CACHE_ROOT)
    output_root: str = 'D:/TreeQCTraining/gate_d_runs'

    def as_row(self) -> dict:
        return dict(self.__dict__)

    @classmethod
    def from_row(cls, row: dict) -> TrainConfig:
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in row.items() if k in known})

    def digest(self) -> str:
        payload = json.dumps(self.as_row(), sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def detect_device(prefer_cuda: bool = True) -> dict:
    import torch
    cuda = bool(torch.cuda.is_available())
    device = 'cuda' if (prefer_cuda and cuda) else 'cpu'
    info = {'device': device, 'cuda_available': cuda,
            'torch': torch.__version__, 'python': platform.python_version()}
    if cuda:
        info['device_name'] = torch.cuda.get_device_name(0)
        info['capability'] = list(torch.cuda.get_device_capability(0))
    return info


def set_determinism(seed: int) -> dict:
    """Seed every generator we control and record what is *not* guaranteed."""
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    return {'seed': seed,
            'torch_deterministic_algorithms': False,
            'cudnn_benchmark': torch.backends.cudnn.benchmark,
            'note': 'CUDA kernel nondeterminism is not globally forced; the dataset '
                    'itself is bit-reproducible from the spec list.'}


def build_datasets(config: TrainConfig, provenance: GateDProvenance):
    def candidate_ids(partition: str, limit: int | None):
        ids = list(provenance.candidates(partition))
        return ids if limit is None else ids[:limit]

    train_dataset = dataset_module.GateDDataset(
        dataset_module.DatasetConfig(
            partition=config.partition, points=config.points,
            repeats=config.repeats, seed_base=config.seed_base),
        provenance,
        candidate_ids=candidate_ids(config.partition, config.max_train_candidates),
        cache_root=config.cache_root)
    validation_dataset = dataset_module.GateDDataset(
        dataset_module.DatasetConfig(
            partition=config.validation_partition, points=config.points,
            repeats=config.repeats, seed_base=config.seed_base),
        provenance,
        candidate_ids=candidate_ids(config.validation_partition,
                                    config.max_validation_candidates),
        cache_root=config.cache_root)
    return train_dataset, validation_dataset


def build_model(name: str):
    import models
    try:
        factory = models.MODELS[name]
    except KeyError as exc:
        raise TrainerError(f'unknown model {name!r}; have {sorted(models.MODELS)}') from exc
    return factory()


def _run_epoch(model, loader, criterion, device, optimizer=None, scaler=None,
               amp=False, amp_dtype=None, grad_clip=1.0):
    import torch
    training = optimizer is not None
    model.train(training)
    total_loss, seen, correct = 0.0, 0, 0
    scores, labels = [], []
    skipped = 0
    for features, batch_labels, _ in loader:
        features = features.to(device, non_blocking=True)
        batch_labels = batch_labels.to(device, non_blocking=True)
        with torch.set_grad_enabled(training):
            with torch.autocast(device_type=device.type, dtype=amp_dtype,
                                enabled=amp and device.type == 'cuda'):
                logits = model(features)
                loss = criterion(logits, batch_labels)
        if training:
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None and scaler.is_enabled():
                scaler.scale(loss).backward()
                if grad_clip:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                before = scaler._scale.item() if hasattr(scaler, '_scale') else None
                scaler.step(optimizer)
                scaler.update()
                after = scaler._scale.item() if hasattr(scaler, '_scale') else None
                if before is not None and after is not None and after < before:
                    skipped += 1
            else:
                loss.backward()
                if grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
        total_loss += float(loss.detach()) * batch_labels.shape[0]
        seen += int(batch_labels.shape[0])
        predictions = logits.detach().argmax(dim=1)
        correct += int((predictions == batch_labels).sum())
        probability = torch.softmax(logits.detach().float(), dim=1)[:, 1]
        scores.append(probability.cpu().numpy())
        labels.append(batch_labels.detach().cpu().numpy())
    if seen == 0:
        raise TrainerError('empty loader')
    return {'loss': total_loss / seen, 'accuracy': correct / seen,
            'examples': seen, 'amp_skipped_updates': skipped,
            'scores': np.concatenate(scores), 'labels': np.concatenate(labels)}


def _checkpoint_payload(model, optimizer, scaler, epoch, config, history,
                        dataset_digest, best):
    import torch
    return {
        'trainer_version': TRAINER_VERSION,
        'epoch': int(epoch),
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict() if optimizer is not None else None,
        'scaler_state': scaler.state_dict() if scaler is not None else None,
        'config': config.as_row(),
        'config_digest': config.digest(),
        'dataset_digest': dataset_digest,
        'history': history,
        'best': best,
        'torch': torch.__version__,
    }


def _digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        while chunk := handle.read(4 * 1024**2):
            digest.update(chunk)
    return digest.hexdigest()


def save_checkpoint(path: Path, payload: dict) -> str:
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.pt.tmp')
    torch.save(payload, tmp)
    tmp.replace(path)
    return _digest_file(path)


def load_checkpoint(path: Path, map_location='cpu') -> dict:
    import torch
    return torch.load(Path(path), map_location=map_location, weights_only=False)


def train(config: TrainConfig, provenance: GateDProvenance, reports: Path | None = None,
          resume_from: Path | str | None = None, device_preference: str = 'cuda') -> dict:
    """Run a full training job. Refuses to start while Gate D is unaccepted."""
    lock.require_training_allowed(reports)
    import fp32_policy
    import torch
    fp32 = fp32_policy.configure_fp32()

    device_info = detect_device(prefer_cuda=(device_preference == 'cuda'))
    device = torch.device(device_info['device'])
    determinism = set_determinism(config.seed)

    train_dataset, validation_dataset = build_datasets(config, provenance)
    dataset_digest = train_dataset.digest()
    train_loader = dataset_module.make_loader(
        train_dataset, config.batch_size, True, config.seed, config.workers,
        pin_memory=device.type == 'cuda')
    validation_loader = dataset_module.make_loader(
        validation_dataset, config.batch_size, False, config.seed + 1, config.workers,
        pin_memory=device.type == 'cuda')

    model = build_model(config.model).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                  weight_decay=config.weight_decay)
    amp_dtype = getattr(torch, config.amp_dtype)
    use_amp = bool(config.amp and device.type == 'cuda')
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    criterion = torch.nn.CrossEntropyLoss()

    output_root = Path(config.output_root) / f'{config.model}_seed{config.seed}'
    output_root.mkdir(parents=True, exist_ok=True)
    history: list[dict] = []
    best = {'epoch': -1, 'balanced_accuracy': -1.0, 'threshold': None, 'path': None}
    start_epoch = 0

    if resume_from is not None:
        payload = load_checkpoint(resume_from, map_location=device)
        if payload.get('config_digest') != config.digest():
            raise TrainerError('resume config does not match the checkpoint')
        if payload.get('dataset_digest') != dataset_digest:
            raise TrainerError('resume dataset digest does not match the checkpoint')
        model.load_state_dict(payload['model_state'])
        if payload.get('optimizer_state') is not None:
            optimizer.load_state_dict(payload['optimizer_state'])
        if payload.get('scaler_state') is not None and use_amp:
            scaler.load_state_dict(payload['scaler_state'])
        history = list(payload.get('history', []))
        best = dict(payload.get('best', best))
        start_epoch = int(payload['epoch']) + 1

    stale = 0
    started = time.time()
    for epoch in range(start_epoch, config.epochs):
        train_row = _run_epoch(model, train_loader, criterion, device, optimizer,
                               scaler, use_amp, amp_dtype, config.grad_clip)
        with torch.no_grad():
            validation_row = _run_epoch(model, validation_loader, criterion, device)
        selected = metrics_module.select_threshold(
            validation_row['labels'], validation_row['scores'],
            config.validation_partition, config.good_false_review_budget)
        row = {
            'epoch': epoch,
            'train_loss': round(train_row['loss'], 6),
            'train_accuracy': round(train_row['accuracy'], 6),
            'validation_loss': round(validation_row['loss'], 6),
            'validation_accuracy': round(validation_row['accuracy'], 6),
            'amp_skipped_updates': train_row['amp_skipped_updates'],
            'validation_threshold': selected['threshold'],
            'validation_BALANCED_ACCURACY': selected['BALANCED_ACCURACY'],
            'validation_BAD_RECALL': selected['BAD_RECALL'],
            'validation_GOOD_FALSE_REVIEW': selected['GOOD_FALSE_REVIEW'],
        }
        history.append(row)
        improved = (selected['BALANCED_ACCURACY']
                    > best['balanced_accuracy'] + config.early_stop_min_delta)
        if improved:
            best = {'epoch': epoch, 'balanced_accuracy': selected['BALANCED_ACCURACY'],
                    'threshold': selected['threshold'],
                    'path': str(output_root / 'best.pt')}
            save_checkpoint(Path(best['path']),
                            _checkpoint_payload(model, optimizer, scaler, epoch, config,
                                                history, dataset_digest, best))
            stale = 0
        else:
            stale += 1
        save_checkpoint(output_root / 'last.pt',
                        _checkpoint_payload(model, optimizer, scaler, epoch, config,
                                            history, dataset_digest, best))
        if stale >= config.early_stop_patience:
            row['early_stopped'] = True
            break

    final_threshold = best['threshold'] if best['threshold'] is not None else 0.5
    frozen = metrics_module.freeze_threshold(
        {'threshold': final_threshold, 'partition': config.validation_partition,
         'selection_rule': 'best_epoch_validation_selection',
         'good_false_review_budget': config.good_false_review_budget,
         'budget_satisfied': None},
        dataset_digest,
        _digest_file(Path(best['path'])) if best['path'] else '',
        config.seed)
    return {
        'trainer_version': TRAINER_VERSION,
        'model': config.model,
        'device': device_info,
        'determinism': determinism,
        'fp32_policy': fp32,
        'config': config.as_row(),
        'config_digest': config.digest(),
        'dataset': {'train': train_dataset.restart_state(),
                    'validation': validation_dataset.restart_state()},
        'epochs_run': len(history),
        'history': history,
        'best': best,
        'frozen_threshold': frozen,
        'seconds': round(time.time() - started, 3),
        'held_out_used': False,
    }
