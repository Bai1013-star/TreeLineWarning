"""Can the heads overfit ONE correct mask?

A conservative refiner that cannot reproduce a single perfect mask it trained on
has a defect, and any negative verdict built on it would be a statement about the
defect rather than about the research direction. This runs the smallest possible
test of that: one GT_CORRECT row, many steps, and a check of whether the refine
output converges on the input.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import v25_config as cfg
import v25_delta_data as data
import v25_delta_model as model_module
import v25_train as train

torch.set_num_threads(4)


def main() -> int:
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    points = int(sys.argv[2]) if len(sys.argv) > 2 else 2048
    index = data.load_library()
    sys.path.insert(0, str(cfg.V2_DIR))
    import v2_refiner_train as v2_train
    split = v2_train.split_by_tree(index, test_site='wytham', seed=cfg.SEED)
    examples = data.build_examples(index, split.train)
    good = [e for e in examples if e.proposal_id == 'GT_CORRECT']
    example = good[0]
    print(f'one example: {example.key} (current IoU {example.current_iou:.3f})')

    dataset = data.DeltaDataset(index, [example], points, cfg.SEED, augment=False)
    item = dataset[0]
    remove_active = item['remove_target'] >= 0
    add_active = item['add_target'] >= 0
    print(f'sampled {points} points: REMOVE active {int(remove_active.sum())} '
          f'positives {int((item["remove_target"][remove_active] > 0).sum())}; '
          f'ADD active {int(add_active.sum())} '
          f'positives {int((item["add_target"][add_active] > 0).sum())}')
    masked_points = int(data.proposal_mask(
        dataset.by_target[example.target_id], 'GT_CORRECT').sum())
    print(f'the whole ROI holds {masked_points:,} points and every one is a keep')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(cfg.SEED)
    model = model_module.build_model(len(data.FEATURE_CHANNELS)).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True,
                                         collate_fn=data.collate)
    started = 0
    for _ in range(steps):
        for batch in loader:
            optimiser.zero_grad()
            outputs = model(batch['features'].to(device))
            loss, parts = train.total_loss(outputs, batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimiser.step()
            started += 1
            if started in (1, 20, 100, 200, 300, steps):
                print(f'   step {started:4d} loss {float(loss.detach()):.5f} '
                      f'parts {parts}', flush=True)
            if started >= steps:
                break

    print('--- what the trained heads do to the FULL correct ROI')
    rows = train.score_rows(model, index, [example], device)
    row = rows[0]
    print(f"   current IoU {row['current_iou']:.4f} -> refined "
          f"{row['refined_iou']:.4f} (delta {row['delta_iou']:+.4f})")
    print(f"   removed {row['removed_points']:,} of "
          f"{data.proposal_mask(dataset.by_target[example.target_id], 'GT_CORRECT').sum():,} "
          f"masked points; added {row['added_points']:,}")
    print(f"   REMOVE tp {row['remove_true_positive']:,} "
          f"fp {row['remove_false_positive']:,}; "
          f"ADD tp {row['add_true_positive']:,} fp {row['add_false_positive']:,}")
    print()
    verdict_line = (
        'DEFECT: the head still deletes most of a mask it trained on'
        if row['removed_points'] > 0.2 * int(
            data.proposal_mask(dataset.by_target[example.target_id],
                               'GT_CORRECT').sum())
        else 'the head can honour a correct mask when trained on it')
    print(f'PROBE VERDICT: {verdict_line}')
    cfg.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = cfg.REPORT_DIR / 'V25_OVERFIT_PROBE.json'
    out.write_text(json.dumps({
        'probe': 'single GT_CORRECT row, many steps, then scored on the full ROI',
        'example': example.key,
        'steps': steps,
        'sampled_points': points,
        'current_iou': row['current_iou'],
        'refined_iou': row['refined_iou'],
        'delta_iou': row['delta_iou'],
        'removed_points': row['removed_points'],
        'added_points': row['added_points'],
        'masked_points': int(data.proposal_mask(
            dataset.by_target[example.target_id], 'GT_CORRECT').sum()),
        'verdict': verdict_line,
        'interpretation': (
            'If the heads can preserve a correct mask when that mask is all they '
            'train on, a negative result on the full corpus is about the difficulty '
            'of per-point keep/remove discrimination at forest scale, not about a '
            'coding defect.'),
    }, indent=2), encoding='utf-8')
    print(f'wrote {out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
