"""Separate two effects: the true base rate, and what the sampler does to it.

``sample_points`` stratifies over keep / remove / add / neighbour / background with
a per-stratum floor, so the sampled positive rate is not the population rate. If the
loss weights the sample as if the distribution were real, the model is trained on
the sampler's distortion rather than on geometry. This measures both numbers.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import v25_config as cfg
import v25_delta_data as data


def strata(current: np.ndarray, gt: np.ndarray, instance: np.ndarray) -> dict:
    return {
        'keep': current & gt,
        'remove': current & ~gt,
        'add': ~current & gt,
        'neighbour': ~current & ~gt & (instance > 0),
        'background': ~current & ~gt & (instance <= 0),
    }


def main() -> int:
    index = data.load_library()
    sys.path.insert(0, str(cfg.V2_DIR))
    import v2_refiner_train as v2_train
    split = v2_train.split_by_tree(index, test_site='wytham', seed=cfg.SEED)
    examples = data.build_examples(index, split.train)
    by_target = {row['target_id']: row for row in index['targets']}

    population = {key: 0 for key in ('keep', 'remove', 'add', 'neighbour', 'background')}
    sampled = dict(population)
    rows = 0
    for example in examples[:200]:
        cluster = by_target[example.target_id]
        gt = np.asarray(cluster['masks']['gt'], dtype=bool)
        instance = np.asarray(cluster['instance'], dtype=np.int32)
        current = data.proposal_mask(cluster, example.proposal_id)
        groups = strata(current, gt, instance)
        for key, mask in groups.items():
            population[key] += int(mask.sum())
        rng = np.random.default_rng(0)
        selected = data.sample_points(current, gt, instance, cfg.POINTS_PER_EXAMPLE, rng)
        for key, mask in groups.items():
            sampled[key] += int(mask[selected].sum())
        rows += 1

    pop_total = sum(population.values())
    sample_total = sum(sampled.values())
    print(f'{rows} training rows, {pop_total:,} points in the ROI population')
    print(f"{'stratum':<12}{'population':>12}{'share':>9}{'sampled':>10}"
          f"{'share':>9}{'inflation':>11}")
    for key in ('keep', 'remove', 'add', 'neighbour', 'background'):
        pop_share = population[key] / max(pop_total, 1)
        smp_share = sampled[key] / max(sample_total, 1)
        inflation = smp_share / pop_share if pop_share else float('inf')
        print(f'{key:<12}{population[key]:>12,}{pop_share:>9.4f}'
              f'{sampled[key]:>10,}{smp_share:>9.4f}{inflation:>11.3f}')

    inside = population['keep'] + population['remove']
    inside_sampled = sampled['keep'] + sampled['remove']
    print()
    print('REMOVE-head positive rate (points the mask should drop):')
    print(f'   true population : {population["remove"] / max(inside, 1):.4f}')
    print(f'   as sampled      : {sampled["remove"] / max(inside_sampled, 1):.4f}')
    outside = population['add'] + population['neighbour'] + population['background']
    outside_sampled = sampled['add'] + sampled['neighbour'] + sampled['background']
    print('ADD-head positive rate (points the mask should gain):')
    print(f'   true population : {population["add"] / max(outside, 1):.4f}')
    print(f'   as sampled      : {sampled["add"] / max(outside_sampled, 1):.4f}')
    print()
    print('inflation > 1 means the sampler over-represents that stratum, and a loss')
    print('that does not correct for it trains on the sampler, not on the forest')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
