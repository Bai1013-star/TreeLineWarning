"""Time a real scoring pass over the test split, at the true ROI size distribution."""
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
import v25_train as train

torch.set_num_threads(4)


def main() -> int:
    index = data.load_library()
    import v2_refiner_train as v2_train
    split = v2_train.split_by_tree(index, test_site='wytham', seed=cfg.SEED)
    examples = data.build_examples(index, split.test)
    print(f'test rows: {len(examples)}')

    dataset = data.FullRoiDeltaDataset(index, examples)
    started = time.time()
    sizes = [dataset[i]['roi_points'] for i in range(min(24, len(examples)))]
    build = time.time() - started
    print(f'built {len(sizes)} in {build:.2f}s; roi points min {min(sizes):,} '
          f'median {int(np.median(sizes)):,} max {max(sizes):,}')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model_module.build_model(len(data.FEATURE_CHANNELS)).to(device)
    model.eval()

    for batch_size in (4, 8):
        subset = examples[:24]
        started = time.time()
        rows = train.score_rows(model, index, subset, device)
        elapsed = time.time() - started
        per_row = elapsed / max(len(subset), 1)
        print(f'   batch {batch_size}: {len(subset)} rows in {elapsed:.1f}s '
              f'({per_row:.3f}s per row)')
        del rows
    print(f'   projected for {len(examples)} test rows at batch 4: '
          f'~{per_row * len(examples) / 60:.1f} min')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
