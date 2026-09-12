"""Why does the ungated delta refiner destroy the mask?

Measures the base rates of the two decisions, what the trained heads predict, and
what the heads would do on a perfectly correct proposal.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import v25_config as cfg
import v25_delta_data as data

torch.set_num_threads(4)


def main() -> int:
    index = data.load_library()
    import v2_refiner_train as v2_train
    split = v2_train.split_by_tree(index, test_site='wytham', seed=cfg.SEED)

    print('=== base rates of the two decisions on the training split')
    examples = data.build_examples(index, split.train)
    order, sampling = data.sample_training_set(examples, cfg.SEED)
    dataset = data.DeltaDataset(index, order[:64], 1024, cfg.SEED, augment=False)
    remove_pos = remove_act = add_pos = add_act = 0
    for i in range(64):
        item = dataset[i]
        remove_active = item['remove_target'] >= 0
        add_active = item['add_target'] >= 0
        remove_act += int(remove_active.sum())
        add_act += int(add_active.sum())
        remove_pos += int((item['remove_target'][remove_active] > 0.5).sum())
        add_pos += int((item['add_target'][add_active] > 0.5).sum())
    print(f'   REMOVE: {remove_pos:,} positives of {remove_act:,} active '
          f'({remove_pos / max(remove_act, 1):.4f})')
    print(f'   ADD:    {add_pos:,} positives of {add_act:,} active '
          f'({add_pos / max(add_act, 1):.4f})')

    print()
    print('=== per-corruption base rates (first 12 examples of each)')
    seen = {}
    for i in range(min(len(order), 400)):
        corruption = order[i].corruption
        bucket = seen.setdefault(corruption, [])
        if len(bucket) < 12:
            bucket.append(i)
    for corruption, positions in sorted(seen.items()):
        rp = ra = ap = aa = 0
        for i in positions:
            item = dataset[i % len(dataset)]
            r_active = item['remove_target'] >= 0
            a_active = item['add_target'] >= 0
            ra += int(r_active.sum())
            aa += int(a_active.sum())
            rp += int((item['remove_target'][r_active] > 0.5).sum())
            ap += int((item['add_target'][a_active] > 0.5).sum())
        print(f'   {corruption:<20} REMOVE {rp:>6}/{ra:>6} '
              f'({rp / max(ra, 1):.3f})   ADD {ap:>6}/{aa:>6} '
              f'({ap / max(aa, 1):.3f})')

    print()
    print('=== what the trained heads predict on validation')
    checkpoint = Path(r'E:\TreeLineWarning-tree-qc-pointnet2-final-pilot-v1'
                      r'\research\tree_qc_v25\reports\V25_ROWS_'
                      r'cross_site_test_wytham.json')
    if not checkpoint.exists():
        print('   no scored rows yet')
        return 0
    rows = json.loads(checkpoint.read_text(encoding='utf-8'))
    test_rows = rows['test_rows']
    print(f'   rows {len(test_rows)}')
    removed = np.array([r['removed_points'] for r in test_rows])
    added = np.array([r['added_points'] for r in test_rows])
    changed = np.array([r['changed_points'] for r in test_rows])
    region = np.array([r['region_points'] for r in test_rows])
    print(f'   removed points per row: median {np.median(removed):,.0f}')
    print(f'   added points per row:   median {np.median(added):,.0f}')
    print(f'   changed fraction:       median '
          f'{np.median(changed / np.maximum(region, 1)):.3f}')
    current = np.array([r['current_iou'] for r in test_rows])
    refined = np.array([r['refined_iou'] for r in test_rows])
    print(f'   current IoU median {np.median(current):.3f} -> '
          f'refined IoU median {np.median(refined):.3f}')
    print('   the mask collapses to almost nothing when most points are removed:')
    print(f'   rows where refined IoU is ~0: {int((refined < 0.01).sum())} of '
          f'{len(test_rows)}')
    print()
    print('=== gate features that were available on validation')
    validation = rows['validation_rows']
    for name in ('gate_remove_top', 'gate_add_top', 'gate_change_fraction'):
        values = np.array([row[name] for row in validation])
        print(f'   {name:<22} min {values.min():.4f} median '
              f'{np.median(values):.4f} max {values.max():.4f}')
    deltas = np.array([row['delta_iou'] for row in validation])
    print(f'   validation delta_iou: median {np.median(deltas):+.4f} '
          f'positive fraction {float((deltas > 0).mean()):.3f}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
