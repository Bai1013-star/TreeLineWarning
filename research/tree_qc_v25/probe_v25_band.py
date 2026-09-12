"""Would a bounded candidate band fix the decision base rates?

The full-ROI candidate set is the V2 ROI: the candidate box grown in XY, which
contains whole neighbouring trees. Measured against the target's own truth, about
86 percent of the points inside the mask are not the target, so "remove" is the
majority strategy and a model that follows the majority destroys the mask.

Before changing anything, this measures what the base rates become when the
candidate set is restricted to a band around the current mask, which is what
"conservative local correction" actually means.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import v25_config as cfg
import v25_delta_data as data

#: Band width around the current mask, in metres. A point further than this from the
#: mask is neither a plausible removal nor a plausible addition for a *local*
#: correction, so it keeps its current label by default.
BAND_M = 1.5


def band_mask(xyz: np.ndarray, current: np.ndarray, band: float) -> np.ndarray:
    """Points within ``band`` metres of the current mask surface."""
    from scipy.spatial import cKDTree
    if not current.any():
        return np.zeros(len(xyz), dtype=bool)
    tree = cKDTree(xyz[current])
    distance, _ = tree.query(xyz, k=1)
    return distance <= band


def main() -> int:
    index = data.load_library()
    sys.path.insert(0, str(cfg.V2_DIR))
    import v2_refiner_train as v2_train
    split = v2_train.split_by_tree(index, test_site='wytham', seed=cfg.SEED)
    examples = data.build_examples(index, split.train)
    order, _sampling = data.sample_training_set(examples, cfg.SEED)
    by_target = {row['target_id']: row for row in index['targets']}

    # One example of each corruption, plus a good row, so the comparison is legible.
    wanted = {}
    for example in order:
        key = example.corruption if example.corruption != 'NONE' else 'NONE'
        if key not in wanted:
            wanted[key] = example
    print(f'band = {BAND_M} m')
    print(f"{'corruption':<20}{'in-mask':>10}{'out-mask':>10}{'band':>10}"
          f"{'rm rate full':>13}{'rm rate band':>13}{'add rate full':>14}"
          f"{'add rate band':>14}")
    for corruption, example in sorted(wanted.items()):
        cluster = by_target[example.target_id]
        xyz = np.asarray(cluster['xyz'], dtype=np.float32)
        truth = np.asarray(cluster['masks']['gt'], dtype=bool)
        current = data.proposal_mask(cluster, example.proposal_id)
        inside_band = band_mask(xyz, current, BAND_M)
        inside_band[current] = True   # every masked point is always a removal candidate

        # REMOVE candidates are masked points; ADD candidates are unmasked points,
        # and both are further restricted to the band.
        full_remove = current
        band_remove = current & inside_band
        full_add = ~current
        band_add = ~current & inside_band
        rm_full = float((~truth[full_remove]).mean()) if full_remove.any() else 0.0
        rm_band = float((~truth[band_remove]).mean()) if band_remove.any() else 0.0
        add_full = float(truth[full_add].mean()) if full_add.any() else 0.0
        add_band = float(truth[band_add].mean()) if band_add.any() else 0.0
        print(f'{corruption:<20}{int(current.sum()):>10,}'
              f'{int((~current).sum()):>10,}{int(band_add.sum()):>10,}'
              f'{rm_full:>13.3f}{rm_band:>13.3f}{add_full:>14.4f}{add_band:>14.4f}')
    print()
    print('rm rate = share of removal candidates that really should be removed')
    print('add rate = share of addition candidates that really should be added')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
