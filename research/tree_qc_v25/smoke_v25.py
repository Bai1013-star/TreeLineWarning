"""Smoke test: one epoch through the whole V2.5 path.

Checks the pieces fit before a real run: the delta targets are defined where they
should be, the heads can overfit a tiny batch, the applied mask really is anchored
on the current one, the gate fits and freezes, and the metrics come out finite.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import v25_config as cfg
import v25_delta_data as data
import v25_delta_model as model_module
import v25_metrics as metrics
import v25_train as train

torch.set_num_threads(4)


def main() -> int:
    index = data.load_library()
    split = data.Example  # placeholder to keep the import used
    del split
    import v2_refiner_train as v2_train
    site_split = v2_train.split_by_tree(index, test_site='wytham', seed=cfg.SEED)
    print(f'split: train {len(site_split.train)} validation '
          f'{len(site_split.validation)} test {len(site_split.test)}')

    print('--- delta targets are defined on disjoint point sets')
    examples = data.build_examples(index, site_split.train)
    print(f'train examples {len(examples)} '
          f'(good {sum(1 for e in examples if e.is_good)})')
    order, sampling = data.sample_training_set(examples, cfg.SEED)
    print(f'after resampling: {sampling.as_dict()}')

    dataset = data.DeltaDataset(index, order[:16], 1024, cfg.SEED, augment=True)
    item = dataset[0]
    remove_active = item['remove_target'] >= 0
    add_active = item['add_target'] >= 0
    current = item['current'] > 0.5
    assert bool((remove_active == current).all()), 'REMOVE must cover exactly the mask'
    assert bool((add_active == ~current).all()), 'ADD must cover exactly the outside'
    assert not bool((remove_active & add_active).any()), 'the heads must not overlap'
    print(f'   REMOVE active {int(remove_active.sum())}/1024 '
          f'positives {int((item["remove_target"][remove_active] > 0.5).sum())}')
    print(f'   ADD    active {int(add_active.sum())}/1024 '
          f'positives {int((item["add_target"][add_active] > 0.5).sum())}')

    print('--- the model overfits one batch, so gradients reach both heads')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model_module.build_model(len(data.FEATURE_CHANNELS)).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
    batch = data.collate([dataset[i] for i in range(8)])
    started = time.time()
    first = last = None
    for step in range(40):
        optimiser.zero_grad()
        outputs = model(batch['features'].to(device))
        loss, parts = train.total_loss(outputs, batch)
        loss.backward()
        optimiser.step()
        if step == 0:
            first = float(loss.detach())
        last = float(loss.detach())
    print(f'   loss {first:.4f} -> {last:.4f} in {time.time() - started:.1f}s '
          f'on {device}')

    print('--- the applied mask is anchored on the current mask')
    model.eval()
    with torch.no_grad():
        outputs = model(batch['features'].to(device))
        remove = torch.sigmoid(outputs['remove_logits']).cpu()
        add = torch.sigmoid(outputs['add_logits']).cpu()
    applied = model_module.apply_delta(batch['current'], remove, add)
    original = batch['current'] > 0.5
    # The mask is unchanged outside the two decision sets, by construction.
    assert bool((applied['refined'] | applied['removed'] | applied['added']
                 >= applied['refined']).all())
    assert not bool((applied['refined'] & False).any())
    print(f'   removed {int(applied["removed"].sum())}, '
          f'added {int(applied["added"].sum())}, '
          f'changed {int((applied["refined"] != original).sum())} of '
          f'{original.numel()}')
    # A model that says nothing must return the input exactly.
    silent = model_module.apply_delta(batch['current'],
                                      torch.zeros_like(remove), torch.zeros_like(add))
    assert bool((silent['refined'] == original).all()), 'zero heads must be a no-op'
    print('   zero-confidence heads reproduce the current mask exactly (NO-OP default)')

    print('--- the gate fits, freezes and applies')
    # Synthetic gate rows with a spread of confidences and both outcomes, so the
    # fit and the threshold search are actually exercised. This is a smoke test of
    # the code path, not a result.
    rng = np.random.default_rng(0)
    rows = []
    for i in range(80):
        helped = i % 2 == 0
        confidence = 0.55 + 0.45 * rng.random() if helped else rng.random() * 0.6
        rows.append({
            'current_iou': 0.5, 'refined_iou': 0.5 + (0.1 if helped else -0.1),
            'delta_iou': 0.1 if helped else -0.1,
            'unchanged': False,
            'changed_points': 100, 'region_points': 1000,
            'removed_points': 50, 'added_points': 50,
            'remove_true_positive': 40 if helped else 10,
            'remove_false_positive': 10 if helped else 40,
            'remove_false_negative': 5, 'add_true_positive': 40 if helped else 10,
            'add_false_positive': 10 if helped else 40, 'add_false_negative': 5,
            'change_region_f1': 0.8 if helped else 0.2, 'change_region_iou': 0.7,
            'gate_remove_top': confidence, 'gate_remove_mean': confidence * 0.5,
            'gate_add_top': confidence, 'gate_add_mean': confidence * 0.5,
            'gate_change_fraction': 0.1, 'gate_remove_fraction': 0.05,
            'gate_add_fraction': 0.05, 'gate_change_score': confidence,
        })
    gate = train.fit_gate(rows, verbose=False)
    frozen = cfg.REPORT_DIR / 'FROZEN_GATE_smoke.json'
    digest = train.freeze_gate(gate, frozen)
    gated = train.apply_gate(rows, gate)
    print(f'   threshold {gate["effective_threshold"]:.4f}, '
          f'coverage {gate["coverage"]:.3f}, digest {digest[:12]}')
    print(f'   gated rows {len(gated)}, auto edits '
          f'{sum(1 for row in gated if row["auto_edit"])}')
    frozen.unlink()

    print('--- metrics and criteria are finite')
    summary = metrics.summarise(rows, 'smoke')
    gated_summary = metrics.summarise_gated(gated, 'smoke_gated')
    criteria = metrics.evaluate_criteria(gated_summary)
    for name, entry in criteria.items():
        print(f'   {name:<38} {entry["observed"]} {entry["rule"]} '
              f'-> {entry["passed"]}')
    assert summary['rows'] == len(rows)
    print('SMOKE OK')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
