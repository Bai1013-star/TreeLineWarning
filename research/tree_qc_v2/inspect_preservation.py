"""Detail the GOOD-preservation gate and the cross-site bad-proposal behaviour."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

here = Path(__file__).resolve().parent
bakeoff = json.loads((here / 'reports' / 'MINI_REFINER_BAKEOFF.json').read_text(
    encoding='utf-8'))

for fold in bakeoff['folds']:
    configuration = fold['configuration']
    print(f'=== {configuration}')
    for key in sorted(fold['models']):
        rows_file = here / 'reports' / f'BAKEOFF_ROWS_{configuration}_{key}.json'
        if not rows_file.exists():
            print(f'  {key}: rows file missing')
            continue
        rows = json.loads(rows_file.read_text(encoding='utf-8'))['test_rows']
        current = np.array([r['current_iou'] for r in rows])
        delta = np.array([r['delta_iou'] for r in rows])
        good = current >= 0.80
        bad = ~good
        model = fold['models'][key]
        gate = model['good_preservation_gate']
        print(f'  {key}: rows {len(rows)} good {int(good.sum())} bad {int(bad.sum())}')
        print(f'     gate thresholds {gate["thresholds"]}')
        print(f'     GOOD mean {delta[good].mean():+.4f} median {np.median(delta[good]):+.4f} '
              f'p10 {np.percentile(delta[good], 10):+.4f} '
              f'p90 {np.percentile(delta[good], 90):+.4f}')
        print(f'     GOOD degraded>0.05 fraction '
              f'{float((delta[good] < -0.05).mean()):.4f} '
              f'severe>0.20 {int((delta[good] < -0.20).sum())} '
              f'({int((delta[good] < -0.20).sum()) / max(int(good.sum()), 1):.4f} of good)')
        print(f'     GOOD improved fraction {float((delta[good] > 1e-9).mean()):.4f} '
              f'unchanged {float((np.abs(delta[good]) <= 1e-9).mean()):.4f}')
        if bad.any():
            print(f'     BAD mean {delta[bad].mean():+.4f} '
                  f'improved {float((delta[bad] > 1e-9).mean()):.4f} '
                  f'>0.02 {float((delta[bad] > 0.02).mean()):.4f} '
                  f'degraded>0.02 {float((delta[bad] < -0.02).mean()):.4f}')
        # Where does the good-proposal damage come from?
        if bad.any() and good.any():
            by_corruption = {}
            for row, is_good in zip(rows, good, strict=True):
                by_corruption.setdefault(row['corruption'], [0, 0])
                by_corruption[row['corruption']][0 if is_good else 1] += 1
            print('     good/bad split by corruption:',
                  {k: tuple(v) for k, v in sorted(by_corruption.items())})
    print()
