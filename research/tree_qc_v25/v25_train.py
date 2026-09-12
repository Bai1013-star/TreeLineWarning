"""Train the delta refiner, fit the conservative gate, and score both.

The gate is the safety mechanism this round exists to test, so its discipline is
enforced in code rather than promised in prose:

* the gate is **fitted on validation rows** and its threshold is **chosen on
  validation rows**;
* the thresholds are written to disk, with a digest, **before** any test row is
  scored, and the test path re-reads them from that file;
* nothing in the test path can call the threshold search.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import v25_config as cfg
import v25_delta_data as data
import v25_delta_model as model_module
from torch import nn

# --- losses -------------------------------------------------------------------

def masked_bce(logits: torch.Tensor, target: torch.Tensor,
               importance: torch.Tensor | None = None) -> torch.Tensor:
    """Importance-weighted cross-entropy over the points where the head applies.

    ``target`` carries ``-1`` where the head does not apply; those points are
    excluded from the gradient entirely rather than pushed toward zero, so a head
    is never trained to answer a question it was not asked.

    ``importance`` carries the inverse inclusion probability of each point from the
    stratified sampler. Without it the loss fits the sampler's distorted class
    balance rather than the ROI's: a stratum that the floor over-samples would be
    treated as if it were that common in the forest, which biases the decision
    toward the over-sampled class. The round's first run did exactly that and the
    heads learned to delete masks they should have kept.
    """
    active = target >= 0.0
    if not bool(active.any()):
        return torch.zeros((), device=logits.device)
    labels = target[active]
    values = logits[active]
    if importance is None:
        weight = torch.ones_like(labels)
    else:
        weight = importance[active].to(logits.device).to(labels.dtype)
    loss = nn.functional.binary_cross_entropy_with_logits(values, labels,
                                                          reduction='none')
    weighted = loss * weight
    return weighted.sum() / weight.sum().clamp(min=1e-9)


def total_loss(outputs: dict, batch: dict) -> tuple:
    device = outputs['remove_logits'].device
    importance = batch['importance'].to(device)
    remove = masked_bce(outputs['remove_logits'], batch['remove_target'].to(device),
                        importance)
    add = masked_bce(outputs['add_logits'], batch['add_target'].to(device),
                     importance)
    return remove + add, {'remove': float(remove.detach()),
                          'add': float(add.detach())}


# --- training -----------------------------------------------------------------

def train_delta(index: dict, split, verbose: bool = True) -> dict:
    """Train one delta refiner. One seed, one budget, early stop on validation."""
    started = time.time()
    train_examples = data.build_examples(index, split.train)
    validation_examples = data.build_examples(index, split.validation)
    test_examples = data.build_examples(index, split.test)
    if not train_examples or not validation_examples:
        raise ValueError('a split produced no examples')

    train_order, sampling = data.sample_training_set(train_examples, cfg.SEED)
    if verbose:
        print(f'   sampling: {sampling.as_dict()}', flush=True)

    train_set = data.DeltaDataset(index, train_order, cfg.POINTS_PER_EXAMPLE,
                                  cfg.SEED, augment=True)
    validation_set = data.DeltaDataset(index, validation_examples,
                                       cfg.POINTS_PER_EXAMPLE, cfg.SEED + 1,
                                       augment=False)
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=cfg.BATCH_SIZE, shuffle=True, collate_fn=data.collate)
    validation_loader = torch.utils.data.DataLoader(
        validation_set, batch_size=cfg.BATCH_SIZE, shuffle=False,
        collate_fn=data.collate)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(cfg.SEED)
    np.random.seed(cfg.SEED)
    model = model_module.build_model(len(data.FEATURE_CHANNELS)).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=cfg.MAX_EPOCHS)

    best = {'epoch': -1, 'validation_loss': float('inf'), 'state': None}
    history = []
    for epoch in range(cfg.MAX_EPOCHS):
        epoch_started = time.time()
        model.train()
        running = 0.0
        for batch in train_loader:
            optimiser.zero_grad()
            outputs = model(batch['features'].to(device))
            loss, _ = total_loss(outputs, batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimiser.step()
            running += float(loss.detach()) * len(batch['filename'])
        scheduler.step()
        model.eval()
        validation_total = 0.0
        with torch.no_grad():
            for batch in validation_loader:
                outputs = model(batch['features'].to(device))
                loss, _ = total_loss(outputs, batch)
                validation_total += float(loss.detach()) * len(batch['filename'])
        train_loss = running / max(len(train_set), 1)
        validation_loss = validation_total / max(len(validation_set), 1)
        seconds = round(time.time() - epoch_started, 1)
        history.append({'epoch': epoch, 'train_loss': train_loss,
                        'validation_loss': validation_loss,
                        'epoch_seconds': seconds})
        if verbose:
            print(f'   epoch {epoch:3d} train {train_loss:.4f} val '
                  f'{validation_loss:.4f} ({seconds}s)', flush=True)
        if validation_loss < best['validation_loss'] - 1e-4:
            best = {'epoch': epoch, 'validation_loss': validation_loss,
                    'state': {k: v.detach().clone()
                              for k, v in model.state_dict().items()}}
        elif epoch - best['epoch'] >= cfg.PATIENCE:
            if verbose:
                print(f'   early stop at epoch {epoch} (best {best["epoch"]})',
                      flush=True)
            break
    if best['state'] is not None:
        model.load_state_dict(best['state'])
    return {'model': model, 'device': device,
            'training': {
                'epochs_budget': cfg.MAX_EPOCHS, 'epochs_run': len(history),
                'patience': cfg.PATIENCE, 'best_epoch': best['epoch'],
                'best_validation_loss': best['validation_loss'],
                'history': history,
                'seed': cfg.SEED,
                'seed_policy': 'one fixed seed; no search, no best-of-N',
                'sampling': sampling.as_dict(),
                'examples': {'train': len(train_order),
                             'train_before_resampling': len(train_examples),
                             'validation': len(validation_examples),
                             'test': len(test_examples)},
                'parameters': model_module.parameter_count(model),
                'seconds': round(time.time() - started, 2)}}


# --- full-ROI scoring ---------------------------------------------------------

def score_rows(model, index: dict, examples: list[data.Example],
               device: str, verbose: bool = False) -> list[dict]:
    """Score every example over its whole ROI, batched, capped as in V2."""
    if not examples:
        return []
    dataset = data.FullRoiDeltaDataset(index, examples)
    loader = torch.utils.data.DataLoader(dataset, batch_size=cfg.EVAL_BATCH_SIZE,
                                         shuffle=False,
                                         collate_fn=data.collate_full)
    model.eval()
    rows, done = [], 0
    for batch in loader:
        lengths = batch['roi_points']
        width = max(lengths)
        stacked = torch.zeros(len(lengths), width,
                              batch['features'][0].shape[-1], dtype=torch.float32)
        for position, item in enumerate(batch['features']):
            stacked[position, :lengths[position]] = item
        with torch.no_grad():
            outputs = model(stacked.to(device))
            remove = torch.sigmoid(outputs['remove_logits']).cpu()
            add = torch.sigmoid(outputs['add_logits']).cpu()
        for position in range(len(batch['filename'])):
            n = lengths[position]
            current = batch['current'][position].numpy()[:n] > 0.5
            truth = batch['truth'][position].numpy()[:n] > 0.5
            remove_probability = remove[position][:n].numpy()
            add_probability = add[position][:n].numpy()
            applied = model_module.apply_delta(
                torch.from_numpy(current.astype(np.float32)),
                torch.from_numpy(remove_probability),
                torch.from_numpy(add_probability))
            refined = applied['refined'].numpy()
            counts = model_module.delta_metrics(applied['refined'], applied['removed'],
                                                applied['added'],
                                                torch.from_numpy(truth.astype(np.float32)),
                                                torch.from_numpy(current.astype(np.float32)))
            row = {
                'key': batch['filename'][position],
                'site': batch['site'][position],
                'proposal_kind': batch['proposal_kind'][position],
                'corruption': batch['corruption'][position],
                'severity': batch['severity'][position],
                'library_iou': batch['current_iou_library'][position],
                'roi_points': n,
                'roi_points_stored': batch['roi_points_stored'][position],
                'roi_stride': batch['roi_stride'][position],
                **counts,
            }
            row.update(_iou_fields(refined, truth, current))
            row.update(gate_features(row, remove_probability, add_probability, current))
            rows.append(row)
        done += len(batch['filename'])
        if verbose and done % 200 < 4:
            print(f'      scored {done}/{len(examples)}', flush=True)
    return rows


def _iou_fields(refined: np.ndarray, truth: np.ndarray, current: np.ndarray) -> dict:
    def iou(left, right) -> float:
        intersection = int(np.count_nonzero(left & right))
        union = int(np.count_nonzero(left | right))
        return float(intersection / union) if union else 0.0

    current_iou = iou(current, truth)
    refined_iou = iou(refined, truth)
    changed = refined != current
    change_iou = iou(refined[changed], truth[changed]) if changed.any() else 0.0
    truth_changed = current != truth
    return {
        'current_iou': current_iou,
        'refined_iou': refined_iou,
        'delta_iou': refined_iou - current_iou,
        'change_region_iou': change_iou,
        'change_region_f1': _f1(refined[changed], truth[changed]) if changed.any() else 0.0,
        'truth_change_points': int(truth_changed.sum()),
        'unchanged': not bool(changed.any()),
    }


def _f1(predicted: np.ndarray, truth: np.ndarray) -> float:
    tp = int(np.count_nonzero(predicted & truth))
    fp = int(np.count_nonzero(predicted & ~truth))
    fn = int(np.count_nonzero(~predicted & truth))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return float(2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0


def gate_features(row: dict, remove_probability: np.ndarray,
                  add_probability: np.ndarray, current: np.ndarray) -> dict:
    """The evidence the gate is allowed to use. No ground truth is present.

    Every feature is computable at inference from the model's own outputs and the
    proposal, so the gate cannot be reading the answer.
    """
    inside = current
    outside = ~current
    remove_top = float(np.max(remove_probability[inside])) if inside.any() else 0.0
    remove_mean = float(np.mean(remove_probability[inside])) if inside.any() else 0.0
    add_top = float(np.max(add_probability[outside])) if outside.any() else 0.0
    add_mean = float(np.mean(add_probability[outside])) if outside.any() else 0.0
    region = max(row['region_points'], 1)
    change_fraction = row['changed_points'] / region
    remove_fraction = row['removed_points'] / region
    add_fraction = row['added_points'] / region
    return {
        'gate_remove_top': remove_top,
        'gate_remove_mean': remove_mean,
        'gate_add_top': add_top,
        'gate_add_mean': add_mean,
        'gate_change_fraction': change_fraction,
        'gate_remove_fraction': remove_fraction,
        'gate_add_fraction': add_fraction,
        'gate_change_score': remove_top + add_top - abs(remove_fraction - add_fraction),
    }


GATE_FEATURE_ORDER = ('gate_remove_top', 'gate_remove_mean', 'gate_add_top',
                      'gate_add_mean', 'gate_change_fraction', 'gate_remove_fraction',
                      'gate_add_fraction')


# --- the conservative gate ----------------------------------------------------

def fit_gate(rows: list[dict], verbose: bool = True) -> dict:
    """Fit the gate and freeze its threshold using validation rows only.

    The gate is a logistic regression on the model's own confidence features; it
    predicts whether applying the edit is worth it. The threshold is then chosen so
    that the edits it accepts are as reliable as possible at a coverage that is
    useful, and the chosen values are written to disk with a digest before any test
    row is scored.
    """
    from sklearn.linear_model import LogisticRegression

    features = np.array([[row[name] for name in GATE_FEATURE_ORDER] for row in rows],
                        dtype=np.float64)
    # The label is "this edit helped", which is only knowable on validation.
    labels = np.array([1 if row['delta_iou'] >= cfg.GATE_BENEFIT_IOU else 0
                       for row in rows], dtype=np.int64)
    edited = np.array([not row['unchanged'] for row in rows], dtype=bool)

    if labels.min() == labels.max():
        # A degenerate validation set cannot support a learned gate; fall back to
        # the frozen probability threshold on the raw confidence and say so.
        probabilities = features[:, 0]
        fitted = False
        coefficients = None
        intercept = None
    else:
        classifier = LogisticRegression(max_iter=2000, C=1.0)
        classifier.fit(features, labels)
        probabilities = classifier.predict_proba(features)[:, 1]
        fitted = True
        coefficients = [float(v) for v in classifier.coef_[0]]
        intercept = float(classifier.intercept_[0])

    choice = choose_threshold(rows, probabilities, edited)
    if verbose:
        print(f'   gate: fitted={fitted} threshold={choice["threshold"]:.4f} '
              f'coverage={choice["coverage"]:.4f} '
              f'success={choice["success_rate"]:.4f}', flush=True)
    return {
        'fitted_on': 'validation rows only',
        'fitted': fitted,
        'feature_order': list(GATE_FEATURE_ORDER),
        'coefficients': coefficients,
        'intercept': intercept,
        'rows': len(rows),
        'positive_rows': int(labels.sum()),
        'edited_rows': int(edited.sum()),
        **choice,
        'threshold_selection': (
            'smallest threshold reaching the declared coverage target that maximises '
            'observed success on validation; if none reaches it, the smallest '
            'threshold whose success rate clears the declared floor'),
        'frozen_before_test': True,
    }


def choose_threshold(rows: list[dict], probabilities: np.ndarray,
                     edited: np.ndarray) -> dict:
    """Pick the gate threshold on validation data, and record why."""
    candidates = sorted({float(value) for value in probabilities})
    best = None
    for threshold in candidates:
        accepted = (probabilities >= threshold) & edited
        coverage = float(accepted.mean())
        if not accepted.any():
            continue
        success = float(np.mean([row['delta_iou'] >= cfg.GATE_BENEFIT_IOU
                                 for row, take in zip(rows, accepted, strict=True)
                                 if take]))
        entry = {'threshold': threshold, 'coverage': coverage, 'success_rate': success}
        if best is None:
            best = entry
        elif (best['coverage'] < cfg.GATE_COVERAGE_TARGET
              and coverage >= cfg.GATE_COVERAGE_TARGET):
            # Reaching the declared coverage target takes priority over success.
            best = entry
        elif ((coverage >= cfg.GATE_COVERAGE_TARGET)
              == (best['coverage'] >= cfg.GATE_COVERAGE_TARGET)
              and success > best['success_rate']):
            best = entry
    if best is None:
        # Everything the model proposed was already a no-op, so there is nothing for
        # the gate to accept. It refuses everything and says so, rather than
        # inventing a threshold that was never validated.
        return {'threshold': 1.0, 'effective_threshold': 1.0, 'coverage': 0.0,
                'success_rate': 0.0, 'coverage_target': cfg.GATE_COVERAGE_TARGET,
                'coverage_target_met': False,
                'min_probability_floor': cfg.GATE_MIN_PROBABILITY,
                'note': 'no validation row carried an edit, so the gate refuses '
                        'everything'}
    best['coverage_target'] = cfg.GATE_COVERAGE_TARGET
    best['coverage_target_met'] = bool(best['coverage'] >= cfg.GATE_COVERAGE_TARGET)
    best['min_probability_floor'] = cfg.GATE_MIN_PROBABILITY
    best['effective_threshold'] = max(best['threshold'], cfg.GATE_MIN_PROBABILITY)
    return best


def freeze_gate(gate: dict, path: Path) -> str:
    """Persist the frozen thresholds and return their digest."""
    payload = {'gate': gate, 'frozen_thresholds': cfg.frozen_thresholds(),
               'config_digest': cfg.config_digest(),
               'frozen_before_wytham_test': True}
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True)
    path.write_text(text, encoding='utf-8')
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def apply_gate(rows: list[dict], gate: dict) -> list[dict]:
    """Apply the frozen gate. Reads thresholds; never chooses them."""
    threshold = float(gate.get('effective_threshold', cfg.GATE_MIN_PROBABILITY))
    coefficients = gate.get('coefficients')
    intercept = gate.get('intercept')
    out = []
    for row in rows:
        vector = np.array([row[name] for name in GATE_FEATURE_ORDER],
                          dtype=np.float64)
        if coefficients and intercept is not None:
            logit = float(np.dot(coefficients, vector) + intercept)
            probability = 1.0 / (1.0 + math.exp(-max(min(logit, 60.0), -60.0)))
        else:
            probability = float(vector[0])
        auto = bool(probability >= threshold) and not row['unchanged']
        entry = dict(row)
        entry['gate_probability'] = probability
        entry['auto_edit'] = auto
        if not auto:
            # NO_OP: the refined mask is discarded and the current one stands.
            entry['gated_refined_iou'] = row['current_iou']
            entry['gated_delta_iou'] = 0.0
        else:
            entry['gated_refined_iou'] = row['refined_iou']
            entry['gated_delta_iou'] = row['delta_iou']
        out.append(entry)
    return out
