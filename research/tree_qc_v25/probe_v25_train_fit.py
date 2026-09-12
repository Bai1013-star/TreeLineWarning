"""Does the trained delta refiner work on its own training data?

This separates "the model is broken" from "the model does not transfer". If the
heads cannot even honour a mask they were trained on, the round's negative result
would be a statement about a defect rather than about the research direction, and
that distinction matters for a GO / NO-GO decision.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import v25_config as cfg
import v25_delta_data as data
import v25_train as train

torch.set_num_threads(4)


def main() -> int:
    index = data.load_library()
    sys.path.insert(0, str(cfg.V2_DIR))
    import v2_refiner_train as v2_train
    split = v2_train.split_by_tree(index, test_site='wytham', seed=cfg.SEED)

    cfg.MAX_EPOCHS = 6
    cfg.PATIENCE = 6
    print('training 6 epochs on the training split', flush=True)
    out = train.train_delta(index, split, verbose=True)
    model, device = out['model'], out['device']

    train_examples = data.build_examples(index, split.train)
    by_corruption = {}
    for example in train_examples:
        by_corruption.setdefault(example.corruption, []).append(example)

    print()
    print('--- scoring on TRAINING data, per corruption (12 rows each)')
    print(f"{'corruption':<20}{'current':>9}{'refined':>9}{'delta':>9}"
          f"{'noop':>7}{'rm rate':>9}{'add rate':>9}")
    for corruption, group in sorted(by_corruption.items()):
        rows = train.score_rows(model, index, group[:12], device)
        current = np.mean([r['current_iou'] for r in rows])
        refined = np.mean([r['refined_iou'] for r in rows])
        delta = np.mean([r['delta_iou'] for r in rows])
        noop = np.mean([r['unchanged'] for r in rows])
        rm_candidates = sum(r['removed_points'] for r in rows)
        rm_true = sum(r['remove_true_positive'] for r in rows)
        add_candidates = sum(r['added_points'] for r in rows)
        add_true = sum(r['add_true_positive'] for r in rows)
        print(f'{corruption:<20}{current:>9.3f}{refined:>9.3f}{delta:>+9.3f}'
              f'{noop:>7.2f}'
              f'{rm_true / max(rm_candidates, 1):>9.3f}'
              f'{add_true / max(add_candidates, 1):>9.3f}')

    print()
    print('--- the decisive question: what does the model do to a CORRECT mask?')
    good = [e for e in train_examples if e.proposal_id == 'GT_CORRECT'][:12]
    rows = train.score_rows(model, index, good, device)
    for row in rows[:6]:
        print(f"   {row['key'][:34]:<34} current {row['current_iou']:.3f} "
              f"refined {row['refined_iou']:.3f} delta {row['delta_iou']:+.3f} "
              f"removed {row['removed_points']:>6,} added {row['added_points']:>6,} "
              f"of {row['roi_points']:,}")
    deltas = np.array([row['delta_iou'] for row in rows])
    print(f'   mean delta on correct masks: {deltas.mean():+.4f} '
          f'(a conservative model should be near 0)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
