"""Summarise a PUBLIC_GT_PROPOSAL report."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

path = Path(sys.argv[1] if len(sys.argv) > 1 else
            Path(__file__).resolve().parent / 'reports' / 'PUBLIC_GT_PROPOSAL.json')
payload = json.loads(path.read_text(encoding='utf-8'))
print('verdict', payload['PUBLIC_GT_PAIR_CONSTRUCTION'])
print('window', payload['window_m'], 'min points', payload['minimum_points'])
print('patches', [(p['patch_id'], p['points'], p['instances']) for p in payload['patches']])
print('ROI coverage', payload['roi_coverage'])
print('ROI points', payload['roi_points'])
print('missing recoverable', payload['missing_recoverable_fraction'])
print()
for route in payload['real_routes']:
    print(f"route {route['patch_id']} [{route['crown_source']}] "
          f"proposals={route['proposal_count']} pairs={route['pair_count']} "
          f"stems={route['validated_stem_count']}")
    print('   states', route['proposal_states'])
    if route.get('crown_perturbation'):
        pert = route['crown_perturbation']
        print('   crowns', pert.get('crown_count_before'), '->',
              pert.get('crown_count_after'), 'changed', pert.get('changed'))
    print('   diagnostics', route['algorithm_diagnostics'])
rows = payload['real_rows_all_crown_sources']
if rows:
    print()
    print('real rows by crown source:')
    for source in sorted({r['crown_source'] for r in rows}):
        group = [r for r in rows if r['crown_source'] == source]
        ious = np.array([r['iou'] for r in group])
        deltas = np.array([r['delta_iou_vs_input_crown'] for r in group])
        print(f'   {source:20} n={len(group):3d} iou med={np.median(ious):.4f} '
              f'min={ious.min():.4f} max={ious.max():.4f} '
              f'frac<0.9={float((ious < 0.9).mean()):.3f} '
              f'delta med={np.median(deltas):+.4f} improved={float((deltas > 1e-9).mean()):.3f} '
              f'unchanged={float((np.abs(deltas) <= 1e-9).mean()):.3f}')
print()
print('simulated profile:')
prof = payload['simulated_defect_profile']
for key in ('rows', 'iou_median', 'iou_mean', 'merge_fraction', 'incomplete_fraction'):
    print(f'   {key}: {prof.get(key)}')
for key, value in prof.get('by_corruption', {}).items():
    print(f'   {key:22} n={value["rows"]:4d} iou_med={value["iou_median"]:.4f} '
          f'iou_min={value["iou_min"]:.4f}')
for key, value in prof.get('by_severity', {}).items():
    print(f'   severity {key:8} n={value["rows"]:4d} iou_med={value["iou_median"]:.4f}')
print()
print('correct rows iou distribution:',
      np.unique([r['iou'] for r in payload['correct_rows']])[:5],
      'n =', len(payload['correct_rows']))
