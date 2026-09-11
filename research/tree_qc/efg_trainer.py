"""Guarded training loop for the Public Pilot.

The Public Pilot must stop on a scientific, safety, data-integrity or resource problem
instead of adapting the experiment. This module supplies the guards; the epoch body,
checkpoint payload and hyper-parameters stay the frozen ones in ``gate_d_train``:

* ``gate_d_train._run_epoch`` performs the actual forward/backward/step, so the training
  arithmetic is byte-for-byte the frozen implementation;
* ``gate_d_train._checkpoint_payload``, ``save_checkpoint``, ``load_checkpoint``,
  ``build_model``, ``set_determinism`` and ``detect_device`` are the frozen helpers, so
  ``config_digest`` and ``dataset_digest`` keep their frozen meaning and a checkpoint
  written here is readable by the frozen loader;
* ``gate_d_metrics.select_threshold`` and ``freeze_threshold`` remain the only threshold
  path, and they refuse any partition outside ``train``/``validation``.

What this layer adds is the fail-closed behaviour the task requires and the frozen engine
does not have:

* a non-finite monitor on every loss and every logit block (``STOP_NONFINITE_TRAINING``);
* a real OOM at a frozen safe batch is ``STOP_RUNTIME_OOM_AFTER_FREEZE``, never a silent
  smaller batch;
* resume refuses a config, dataset, model or points change (``STOP_RESUME_MISMATCH``);
* every completed run records the checkpoint SHA-256 and its frozen threshold so the
  experiment freeze can bind them.

The monitor is installed by rebinding the *model's own* ``forward`` for the duration of the
call. That is instrumentation of the local model object the trainer just constructed; it
does not patch, wrap or weaken any frozen module, and the rebinding is undone in ``finally``.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import efg_dataset as dataset_source
import efg_gates as gates
import gate_d_metrics as metrics_module
import gate_d_train as engine
import numpy as np

TRAINER_VERSION = 'EFG_TRAINER_V1'
DEFAULT_OUTPUT_ROOT = 'D:/TreeQCTraining/public_pilot_v1/runs'


class TrainingStop(RuntimeError):
    """A fail-closed stop with a stable code the orchestrator records."""

    def __init__(self, code: str, detail):
        super().__init__(f'{code}: {detail}')
        self.code = code
        self.detail = detail


class NonFiniteMonitor:
    """Records the most recent loss and logits finiteness seen during one epoch."""

    def __init__(self):
        self.last_loss = None
        self.last_logits_finite = True
        self.calls = 0

    def wrap(self, model):
        monitor = self
        original = model.forward

        def instrumented(*args, **kwargs):
            logits = original(*args, **kwargs)
            monitor.calls += 1
            import torch
            monitor.last_logits_finite = bool(torch.isfinite(logits).all())
            return logits

        model.forward = instrumented
        return original

    def check(self, loss, context: dict) -> None:
        import torch
        self.last_loss = float(loss.detach()) if torch.is_tensor(loss) else float(loss)
        if not np.isfinite(self.last_loss):
            raise TrainingStop('STOP_NONFINITE_TRAINING', dict(
                context, reason='non-finite loss', observed_loss=str(self.last_loss)))
        if not self.last_logits_finite:
            raise TrainingStop('STOP_NONFINITE_TRAINING', dict(
                context, reason='non-finite logits'))


def resume_decision(checkpoint_path, config, dataset_digest: str) -> dict:
    """Whether ``last.pt`` may be resumed, and why not when it may not.

    A resume must not change the learning rate, batch, points, seed, model, dataset or
    generator. All of those are inside the frozen ``config_digest`` and the
    ``dataset_digest``, so both are compared before a single weight is loaded.
    """
    path = Path(checkpoint_path)
    if not path.exists():
        return {'resume': False, 'reason': 'no checkpoint', 'path': str(path)}
    try:
        payload = engine.load_checkpoint(path)
    except Exception as exc:
        raise TrainingStop('STOP_CHECKPOINT_CORRUPTION', {
            'message': 'checkpoint is unreadable', 'path': str(path),
            'error': str(exc)[:400]}) from exc
    recorded_config = payload.get('config_digest')
    if recorded_config != config.digest():
        raise TrainingStop('STOP_RESUME_MISMATCH', {
            'message': 'checkpoint config digest does not match the frozen training '
                       'config; resuming would change the experiment',
            'path': str(path), 'recorded': recorded_config, 'current': config.digest(),
            'recorded_config': payload.get('config'), 'current_config': config.as_row()})
    if payload.get('dataset_digest') != dataset_digest:
        raise TrainingStop('STOP_RESUME_MISMATCH', {
            'message': 'checkpoint dataset digest does not match the current dataset',
            'path': str(path), 'recorded': payload.get('dataset_digest'),
            'current': dataset_digest})
    return {'resume': True, 'path': str(path), 'epoch': int(payload.get('epoch', -1)),
            'config_digest': recorded_config, 'dataset_digest': dataset_digest,
            'checkpoint_sha256': engine._digest_file(path)}


def train_run(config: engine.TrainConfig, provenance, cache_root=None,
              use_cache: bool = True, resume: bool = True,
              device_preference: str = 'cuda', audit_dir: Path | str | None = None,
              progress=None, prewarmed: bool = False) -> dict:
    """Train one (model, seed) run under the Public Pilot guards."""
    import fp32_policy
    import torch

    engine.lock.require_training_allowed()
    fp32 = fp32_policy.configure_fp32()
    device_info = engine.detect_device(prefer_cuda=(device_preference == 'cuda'))
    device = torch.device(device_info['device'])
    determinism = engine.set_determinism(config.seed)

    train_source = dataset_source.build_model_selection_source(
        config, provenance, cache_root=cache_root, use_cache=use_cache, preload=True,
        generator_digest=None)
    validation_source = dataset_source.build_model_selection_source(
        _validation_config(config), provenance, cache_root=cache_root,
        use_cache=use_cache, preload=True, generator_digest=None)
    dataset_digest = train_source.digest()
    validation_digest = validation_source.digest()
    train_tracker = NotApplicableTracker()
    validation_tracker = NotApplicableTracker()
    # A completed prewarm has already written an exclusion record for every spec it could
    # not generate, so the mask can be built from those records instead of regenerating
    # 12 915 examples before the first batch.
    probe = 'records' if (prewarmed and use_cache) else 'generation'

    train_loader = _loader(train_source, config.batch_size, True, config.seed,
                           config.workers, device, train_tracker, probe)
    validation_loader = _loader(validation_source, config.batch_size, False,
                                config.seed + 1, config.workers, device,
                                validation_tracker, probe)
    training_examples = len(train_loader.dataset)
    validation_examples = len(validation_loader.dataset)

    model = engine.build_model(config.model).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                  weight_decay=config.weight_decay)
    amp_dtype = getattr(torch, config.amp_dtype)
    use_amp = bool(config.amp and device.type == 'cuda')
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    criterion = torch.nn.CrossEntropyLoss()

    output_root = Path(config.output_root) / f'{config.model}_seed{config.seed}'
    output_root.mkdir(parents=True, exist_ok=True)
    best_path = output_root / 'best.pt'
    last_path = output_root / 'last.pt'
    history: list[dict] = []
    best = {'epoch': -1, 'balanced_accuracy': -1.0, 'threshold': None, 'path': None}
    start_epoch = 0
    resumed_from = None
    if resume:
        decision = resume_decision(last_path, config, dataset_digest)
        if decision['resume']:
            payload = engine.load_checkpoint(last_path, map_location=device)
            model.load_state_dict(payload['model_state'])
            if payload.get('optimizer_state') is not None:
                optimizer.load_state_dict(payload['optimizer_state'])
            if payload.get('scaler_state') is not None and use_amp:
                scaler.load_state_dict(payload['scaler_state'])
            history = list(payload.get('history', []))
            best = dict(payload.get('best', best))
            start_epoch = int(payload['epoch']) + 1
            resumed_from = decision

    monitor = NonFiniteMonitor()
    original_forward = monitor.wrap(model)
    stale = 0
    started = time.time()
    epochs_this_process = 0
    try:
        for epoch in range(start_epoch, config.epochs):
            context = {'model': config.model, 'seed': config.seed, 'epoch': epoch,
                       'device': str(device), 'batch_size': config.batch_size,
                       'points': config.points}
            train_row = engine._run_epoch(model, train_loader, criterion, device,
                                          optimizer, scaler, use_amp, amp_dtype,
                                          config.grad_clip)
            monitor.check(train_row['loss'], dict(context, phase='train'))
            with torch.no_grad():
                validation_row = engine._run_epoch(model, validation_loader, criterion,
                                                   device)
            monitor.check(validation_row['loss'], dict(context, phase='validation'))
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
                'validation_F1_REVIEW': gates.f1_review(selected),
                'seconds': None,
            }
            improved = (selected['BALANCED_ACCURACY']
                        > best['balanced_accuracy'] + config.early_stop_min_delta)
            if improved:
                best = {'epoch': epoch,
                        'balanced_accuracy': selected['BALANCED_ACCURACY'],
                        'threshold': selected['threshold'],
                        'selection_rule': selected.get('selection_rule'),
                        'budget_satisfied': selected.get('budget_satisfied'),
                        'path': str(best_path)}
                engine.save_checkpoint(best_path, engine._checkpoint_payload(
                    model, optimizer, scaler, epoch, config, history, dataset_digest,
                    best))
                stale = 0
            else:
                stale += 1
            history.append(row)
            engine.save_checkpoint(last_path, engine._checkpoint_payload(
                model, optimizer, scaler, epoch, config, history, dataset_digest, best))
            epochs_this_process += 1
            if progress is not None:
                progress(row)
            if stale >= config.early_stop_patience:
                row['early_stopped'] = True
                break
    except torch.cuda.OutOfMemoryError as exc:
        raise TrainingStop('STOP_RUNTIME_OOM_AFTER_FREEZE', {
            'message': 'CUDA OOM at the frozen safe batch; the batch is not reduced',
            'model': config.model, 'seed': config.seed, 'batch_size': config.batch_size,
            'points': config.points, 'config_digest': config.digest(),
            'error': str(exc)[:400],
            'last_valid_checkpoint': (str(last_path) if last_path.exists() else None),
            'note': 'no automatic batch change, no automatic re-run'}) from exc
    finally:
        model.forward = original_forward
        del train_loader, validation_loader

    if best['path'] is None or not Path(best['path']).exists():
        raise TrainingStop('STOP_CHECKPOINT_CORRUPTION', {
            'message': 'training finished without a best checkpoint',
            'model': config.model, 'seed': config.seed, 'epochs_run': len(history)})

    final_threshold = best['threshold'] if best['threshold'] is not None else 0.5
    frozen = metrics_module.freeze_threshold(
        {'threshold': final_threshold, 'partition': config.validation_partition,
         'selection_rule': 'best_epoch_validation_selection',
         'operating_point_rule': best.get('selection_rule',
                                          'max_bad_recall_within_good_false_review_'
                                          'budget'),
         'good_false_review_budget': config.good_false_review_budget,
         'budget_satisfied': best.get('budget_satisfied')},
        dataset_digest, engine._digest_file(Path(best['path'])), config.seed)
    best_epoch_row = next((row for row in history
                           if row['epoch'] == best['epoch']), history[-1])
    best_metrics = {
        'BAD_RECALL': best_epoch_row['validation_BAD_RECALL'],
        'GOOD_FALSE_REVIEW': best_epoch_row['validation_GOOD_FALSE_REVIEW'],
        'BALANCED_ACCURACY': best_epoch_row['validation_BALANCED_ACCURACY'],
        'F1_REVIEW': best_epoch_row['validation_F1_REVIEW'],
        'BAD_FALSE_PASS': None,
        'threshold': best_epoch_row['validation_threshold'],
        'epoch': best_epoch_row['epoch'],
    }
    run = {
        'trainer_version': TRAINER_VERSION,
        'model': config.model,
        'seed': int(config.seed),
        'device': device_info,
        'determinism': determinism,
        'fp32_policy': fp32,
        'config': config.as_row(),
        'config_digest': config.digest(),
        'config_sha256': config.digest(),
        'dataset': {'train': train_source.restart_state(),
                    'validation': validation_source.restart_state()},
        'dataset_digest': dataset_digest,
        'validation_dataset_digest': validation_digest,
        'epochs_run': len(history),
        'epochs_this_process': epochs_this_process,
        'resumed_from': resumed_from,
        'history': history,
        'best': best,
        'best_metrics': best_metrics,
        'frozen_threshold': frozen,
        'threshold': float(final_threshold),
        'checkpoint_path': best['path'],
        'checkpoint_sha256': engine._digest_file(Path(best['path'])),
        'last_checkpoint_path': str(last_path),
        'last_checkpoint_sha256': (engine._digest_file(last_path)
                                   if last_path.exists() else None),
        'seconds': round(time.time() - started, 3),
        'held_out_used': False,
        'cache': {'train': train_source.cache_stats(),
                  'validation': validation_source.cache_stats()},
        'training_examples': training_examples,
        'validation_examples': validation_examples,
        'not_applicable': {
            'train': train_tracker.enforce(len(train_source.dataset.specs)),
            'validation': validation_tracker.enforce(
                len(validation_source.dataset.specs)),
        },
    }
    if audit_dir is not None:
        import efg_state as state_module
        state_module.atomic_write_json(
            Path(audit_dir) / f'{config.model}_seed{config.seed}.json', run,
            with_digest=False)
    return run


def completed_run_is_valid(record: dict, config, dataset_digest: str,
                           model_seeds_ok: bool = True) -> tuple[bool, str]:
    """Whether a recorded run may be re-used instead of retrained."""
    if not record:
        return False, 'no recorded run'
    required = {
        'config_digest': config.digest(),
        'dataset_digest': dataset_digest,
        'model': config.model,
        'seed': int(config.seed),
    }
    for key, value in required.items():
        if record.get(key) != value:
            return False, (f'{key} mismatch: recorded {str(record.get(key))[:24]} != '
                           f'current {str(value)[:24]}')
    if int((record.get('config') or {}).get('points', -1)) != int(config.points):
        return False, 'points changed'
    if int((record.get('config') or {}).get('batch_size', -1)) != int(config.batch_size):
        return False, 'batch size changed'
    path = record.get('checkpoint_path')
    if not path or not Path(path).exists():
        return False, 'checkpoint missing'
    actual = engine._digest_file(path)
    if actual != record.get('checkpoint_sha256'):
        return False, (f'checkpoint digest mismatch: recorded '
                       f'{str(record.get("checkpoint_sha256"))[:24]} != current '
                       f'{actual[:24]}')
    if not record.get('frozen_threshold'):
        return False, 'threshold not frozen'
    return True, 'digest-bound evidence matches'


def _validation_config(config: engine.TrainConfig) -> engine.TrainConfig:
    """The validation counterpart of a training config.

    Only ``partition`` changes: the candidate caps, the points, the repeats and the seed
    base stay identical, so the validation population is exactly the one the training run
    was configured against. ``max_train_candidates`` becomes ``max_validation_candidates``
    rather than being dropped, because dropping it would silently widen the validation set
    of a bounded run.
    """
    payload = config.as_row()
    payload['partition'] = config.validation_partition
    payload['max_validation_candidates'] = config.max_validation_candidates
    payload['max_train_candidates'] = None
    return engine.TrainConfig.from_row(payload)


class NotApplicableTracker:
    """Counts and names the declared specs the frozen pipeline cannot construct.

    ``gate_d_generator.NotApplicableError`` is a property of the real geometry, not a bug:
    a defect class that the candidate cannot support must not be silently replaced by a
    different example. The loader drops such a sample from its batch, the epoch body masks
    any that slip through, and the run records how many and why. If more than
    ``MAX_EXCLUSION_RATE`` of a partition is excluded, the run stops rather than training on
    a quietly shrunken population.
    """

    def __init__(self, limit: float = 0.05):
        self.limit = float(limit)
        self.excluded: dict[str, str] = {}
        self.seen = 0

    def record(self, key: str, reason: str) -> None:
        self.excluded[key] = reason

    def observe(self, keys) -> None:
        self.seen = max(self.seen, len(keys))

    def summary(self, declared: int) -> dict:
        reasons: dict[str, int] = {}
        by_class: dict[str, int] = {}
        for key, reason in self.excluded.items():
            reasons[reason] = reasons.get(reason, 0) + 1
            parts = key.split('|')
            example_class = parts[2] if len(parts) > 2 else 'UNKNOWN'
            by_class[example_class] = by_class.get(example_class, 0) + 1
        rate = len(self.excluded) / max(1, declared)
        return {'declared': int(declared), 'excluded': len(self.excluded),
                'excluded_fraction': round(rate, 6), 'limit': self.limit,
                'within_limit': rate <= self.limit,
                'by_reason': dict(sorted(reasons.items())),
                'by_class': dict(sorted(by_class.items())),
                'examples': list(sorted(self.excluded))[:20]}

    def enforce(self, declared: int) -> dict:
        summary = self.summary(declared)
        if not summary['within_limit']:
            raise TrainingStop('STOP_DATASET_FINGERPRINT_CHANGED', {
                'message': 'too many declared specs are not applicable; refusing to train '
                           'on a shrunken population',
                'summary': summary})
        return summary


class _MaskedDataset:
    """Drops not-applicable samples so the loader never sees a short batch.

    A ``DataLoader`` batch is built from indices the sampler chose; returning a sentinel
    would either shift the batch size or inject a fake example. Excluding the index from
    the sampler's universe is the only construction that leaves the remaining batches
    exactly as the frozen sampler produced them.

    ``probe='generation'`` verifies applicability by generating every example, which is the
    safe default for a cold cache. ``probe='records'`` trusts the exclusion records a
    completed prewarm wrote; it is only valid after that prewarm, and an index with neither
    a record nor a cached example raises instead of being silently trained on.
    """

    def __init__(self, source, tracker: NotApplicableTracker, probe: str = 'generation'):
        self.source = source
        self.tracker = tracker
        if probe not in ('generation', 'records'):
            raise ValueError(f'unknown probe mode {probe!r}')
        self.probe = probe
        self._probe()

    def _probe(self) -> None:
        import efg_dataset as dataset_source
        self.valid: list[int] = []
        self.excluded: list[int] = []
        for index in range(len(self.source)):
            spec = self.source.dataset.specs[index]
            record = self.source.exclusion(index, spec.key)
            if record is not None:
                self.tracker.record(spec.key, str(record.get('reason', 'not_applicable')))
                self.excluded.append(index)
                continue
            if self.probe == 'generation':
                try:
                    self.source[index]
                except dataset_source.NotApplicableExample as exc:
                    self.tracker.record(spec.key, exc.reason)
                    self.excluded.append(index)
                    continue
            self.valid.append(index)
        self.tracker.observe(self.valid)

    def __len__(self) -> int:
        return len(self.valid)

    def __getitem__(self, position: int):
        return self.source[self.valid[position]]


def _collate(batch):
    """The frozen collate, with the sample index as a tensor.

    ``gate_d_dataset.collate`` expects ``(features, label, index)`` triples, but the
    frozen ``TorchGateDDataset.__getitem__`` yields ``(features, label)`` pairs, so the
    frozen ``make_loader`` — and therefore the frozen ``gate_d_train.train`` epoch loop —
    cannot complete a single batch. Rebuilding that frozen file would change the accepted
    generator digest, so the workaround lives here instead: the loader is built over the
    source wrapper with a collate that mirrors the frozen one and converts the index with
    the same ``torch.stack`` semantics.
    """
    import torch
    features = torch.stack([row[0] for row in batch], 0)
    labels = torch.stack([row[1] for row in batch], 0)
    indices = torch.stack([torch.as_tensor(row[2], dtype=torch.long) for row in batch], 0)
    return features, labels, indices


def _loader(source, batch_size, shuffle, seed, workers, device, tracker=None,
            probe: str = 'generation'):
    """A Windows-safe loader over the *source* wrapper, so the feature cache is used.

    ``gate_d_dataset.make_loader`` wraps ``dataset.dataset``, which would bypass the
    verified feature cache; the loader is therefore built here over the source itself
    using the frozen ``TorchGateDDataset`` adapter and :func:`_collate`. The collation
    semantics, the spawned order and the seeding are the frozen ones.
    """
    import gate_d_dataset as dataset_module
    import torch
    from torch.utils.data import DataLoader

    dataset = (source if tracker is None
               else _MaskedDataset(source, tracker, probe=probe))
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(dataset_module.TorchGateDDataset(dataset),
                      batch_size=batch_size, shuffle=shuffle, num_workers=workers,
                      collate_fn=_collate, generator=generator,
                      pin_memory=device.type == 'cuda',
                      persistent_workers=workers > 0, drop_last=False)


def threshold_record(run: dict, dataset_digest: str) -> dict:
    """The per-seed frozen threshold file the task requires.

    Two different rules are recorded and neither is inferred from the other:

    * ``epoch_selection_rule`` — how the best epoch was chosen (frozen
      ``best_epoch_validation_selection``: the epoch whose validation
      ``BALANCED_ACCURACY`` was highest, validation only);
    * ``operating_point_rule`` — how the threshold *at* that epoch was chosen (frozen
      ``max BAD_RECALL subject to GOOD_FALSE_REVIEW <= 0.25``, validation only).
    """
    frozen_metrics = run['frozen_threshold']['validation_metrics']
    return {
        'seed': int(run['seed']),
        'model': run['model'],
        'points': int(run['config']['points']),
        'threshold': float(run['threshold']),
        'epoch_selection_rule': frozen_metrics.get('selection_rule'),
        'operating_point_rule': frozen_metrics.get('operating_point_rule'),
        'selection_rule': frozen_metrics.get('selection_rule'),
        'good_false_review_budget': run['config']['good_false_review_budget'],
        'budget_satisfied': frozen_metrics.get('budget_satisfied'),
        'validation_metrics': run['best_metrics'],
        'checkpoint_path': run['checkpoint_path'],
        'checkpoint_sha256': run['checkpoint_sha256'],
        'config_sha256': run['config_sha256'],
        'dataset_digest': dataset_digest,
        'validation_dataset_digest': run['validation_dataset_digest'],
        'frozen': True,
        'held_out_used': False,
        'gate_d_generator_digest': run.get('gate_d_generator_digest'),
        'trainer_version': TRAINER_VERSION,
    }


def write_threshold_file(directory: Path | str, run: dict, dataset_digest: str) -> Path:
    import efg_state as state_module
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"seed{int(run['seed'])}_threshold.json"
    state_module.atomic_write_json(path, threshold_record(run, dataset_digest),
                                   with_digest=False)
    return path


def load_metrics_json(path: Path | str) -> dict:
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def score_run_checkpoint(run: dict, source, device_preference: str = 'cuda',
                         batch_size: int = 8) -> dict:
    """Score a recorded run's best checkpoint over a model-selection partition.

    Used to re-derive validation metrics for a run that was resumed from disk, so a
    resumed pipeline never has to trust a stored number it cannot recompute.
    """
    import efg_evaluate as evaluate_module
    import gate_d_train as engine_module
    import torch

    device_info = engine_module.detect_device(
        prefer_cuda=(device_preference == 'cuda'))
    device = torch.device(device_info['device'])
    payload = engine_module.load_checkpoint(run['checkpoint_path'], map_location=device)
    model = engine_module.build_model(run['model']).to(device)
    model.load_state_dict(payload['model_state'])
    result = evaluate_module.score_dataset(model, source, device, batch_size)
    del model
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return result
