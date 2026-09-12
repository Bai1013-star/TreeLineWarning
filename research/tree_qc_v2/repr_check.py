"""Representation sanity check: does each feature block actually separate sites?

Prints per-representation, per-site feature statistics and a plain linear probe
accuracy, so a degenerate deep-training result can be told apart from a
degenerate representation.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import v2_config as cfg
import v2_data as data
import v2_site_fingerprint as sf


def main() -> int:
    rows = data.discover_tree_instances(partitions=('train',))
    sample = data.sample_instances(rows, 24, cfg.SEED)
    pool = sf._prepare_pool(sample, 256, cfg.SEED)
    meta_raw = np.asarray([t['meta_raw'] for t in pool['trees']], dtype=np.float64)
    print('meta_raw mean', meta_raw.mean(axis=0), 'std', meta_raw.std(axis=0))

    for representation in sf.ALL_REPRESENTATIONS:
        rng = np.random.default_rng(np.random.SeedSequence([cfg.SEED, 11]))
        per_site = {}
        for tree in pool['trees']:
            block, _meta = sf.build_feature_block(tree['xyz'], representation, rng, 256)
            per_site.setdefault(tree['site'], []).append(block)
        print(f'--- {representation}')
        flat = {}
        for site, blocks in sorted(per_site.items()):
            stack = np.stack(blocks)
            flat[site] = stack.reshape(len(blocks), -1)
            print(f'   {site:12} shape {stack.shape} '
                  f'mean|xyz| {np.abs(stack[:, :, :3]).mean():.4f} '
                  f'std {stack[:, :, :3].std():.4f} '
                  f'max {stack[:, :, :3].max():.4f}')
        # Plain linear probe on the flattened, permutation-variant features.
        x = np.concatenate([flat[s] for s in sorted(flat)])
        y = np.concatenate([[i] * len(flat[s]) for i, s in enumerate(sorted(flat))])
        from sklearn.linear_model import LogisticRegression
        probe = LogisticRegression(max_iter=2000).fit(x, y)
        print(f'   linear probe train accuracy {probe.score(x, y):.3f}')
        # And a permutation-invariant probe: per-channel mean + max + std.
        summary = []
        for s in sorted(flat):
            stack = np.stack(per_site[s])
            summary.append(np.concatenate([stack.mean(axis=1), stack.max(axis=1),
                                           stack.std(axis=1)], axis=1))
        summary = np.concatenate(summary)
        probe2 = LogisticRegression(max_iter=2000).fit(summary, y)
        print(f'   invariant-summary probe train accuracy {probe2.score(summary, y):.3f}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
