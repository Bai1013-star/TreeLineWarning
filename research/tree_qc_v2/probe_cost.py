"""Per-model epoch cost, on the real training set.

The three architectures must share one epoch budget to be comparable, so the
budget has to fit the slowest of them. This measures what one epoch actually costs
before a multi-hour bakeoff is started on a guess.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import v2_config as cfg
import v2_models as models
import v2_refiner_data as refdata
import v2_refiner_train as trainer
import v2_roi_library as library


def main() -> int:
    points = int(sys.argv[1]) if len(sys.argv) > 1 else 2048
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    index = library.load_library()
    split = trainer.split_by_tree(index, test_site='wytham', seed=cfg.SEED)
    examples = trainer.build_examples(index, split.train)[:limit]
    print(f'{len(examples)} examples, points={points}, device={device}')

    for key in ('A', 'B', 'C'):
        dataset = refdata.RefinerDataset(index, examples, points, cfg.SEED, augment=True)
        loader = torch.utils.data.DataLoader(dataset, batch_size=cfg.REFINER_BATCH,
                                             shuffle=True, collate_fn=refdata.collate)
        model = models.build_model(key, len(refdata.FEATURE_CHANNELS)).to(device)
        optimiser = torch.optim.Adam(model.parameters(), lr=cfg.REFINER_LR)
        model.train()
        if device == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        first = None
        started = time.time()
        batches = 0
        for batch in loader:
            if first is None:
                first = time.time() - started
            batch['quality'] = torch.zeros(len(batch['filename']))
            optimiser.zero_grad()
            outputs = model(batch['features'].to(device))
            loss, _ = trainer.total_loss(outputs, batch, model)
            loss.backward()
            optimiser.step()
            batches += 1
        elapsed = time.time() - started
        per_batch = elapsed / max(batches, 1)
        peak = (torch.cuda.max_memory_allocated() / 1024 ** 2) if device == 'cuda' else 0
        rows_per_epoch = len(trainer.build_examples(index, split.train))
        projected = rows_per_epoch / cfg.REFINER_BATCH * per_batch / 60
        print(f'{key}: first batch {first:.2f}s, {batches} batches {elapsed:.1f}s, '
              f'{per_batch:.3f}s/batch, peak {peak:.0f} MB, '
              f'projected epoch over {rows_per_epoch} rows: {projected:.1f} min')
        del model, loader, dataset
        torch.cuda.empty_cache()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
