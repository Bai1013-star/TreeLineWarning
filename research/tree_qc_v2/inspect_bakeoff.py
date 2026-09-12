"""Summarise the bakeoff: gates, per-defect, good vs bad, resources."""
from __future__ import annotations

import json
import sys
from pathlib import Path

path = Path(sys.argv[1] if len(sys.argv) > 1 else
            Path(__file__).resolve().parent / 'reports' / 'MINI_REFINER_BAKEOFF.json')
payload = json.loads(path.read_text(encoding='utf-8'))
print(f"points={payload['points']} epochs={payload['epochs']} "
      f"device={payload['device']} targets={payload['target_count']}")
print(f"cap {payload['evaluation_cap']['max_points_per_roi']:,} points/ROI")
print()
for fold in payload['folds']:
    print(f"=== {fold['configuration']}  test_site={fold['test_site']} "
          f"held_in={fold['held_in_sites']} split={fold['split']} "
          f"({fold['seconds']:.0f}s)")
    for key in sorted(fold['models']):
        model = fold['models'][key]
        s = model['summary']
        bad = s['bad_proposals']
        good = s['good_proposals']
        gate = model['identity_copy_gate']
        gp = model['good_preservation_gate']
        res = model['resources']
        print(f"  {key} {model.get('name') or model.get('model')}")
        print(f"     epochs_run={model['epochs_run']} best={model['best_epoch']} "
              f"params={model['parameters']:,}")
        print(f"     current IoU {s['current_iou_mean']:.4f} -> "
              f"refined {s['refined_iou_mean']:.4f}  delta {s['delta_iou_mean']:+.4f} "
              f"(median {s['delta_iou_median']:+.4f})")
        print(f"     change_f1 {s['change_region_f1_mean']:.4f} "
              f"change_recall {s['change_region_recall_mean']:.4f} "
              f"added_f1 {s['added_point_f1_mean']:.4f} "
              f"removed_f1 {s['removed_point_f1_mean']:.4f}")
        print(f"     identity_copy {gate['verdict']} "
              f"(unchanged frac {s['identity_copy_fraction']:.3f}, "
              f"change_f1 floor {gate['change_region_f1_floor']})")
        print(f"     GOOD n={good['rows']} mean_delta {good['mean_delta_iou']:+.4f} "
              f"degraded>0.05 {good['degraded_fraction_gt_0.05']:.3f} "
              f"severe {good['severe_degradation_count']} "
              f"({good['mean_current_iou']:.3f}->{good['mean_refined_iou']:.3f}) "
              f"-> {gp['verdict']}")
        print(f"     BAD  n={bad['rows']} mean_delta {bad['mean_delta_iou']:+.4f} "
              f"improved {bad['improved_fraction']:.3f} "
              f"improved>0.02 {bad['improvement_gt_0.02']:.3f} "
              f"({bad['mean_current_iou']:.3f}->{bad['mean_refined_iou']:.3f})")
        print(f"     GPU peak {res['cuda_peak_memory_mb']} MB, "
              f"single-ROI GPU {res['single_roi_inference_seconds']}s, "
              f"CPU {model['cpu_probe']['cpu_single_roi_inference_seconds']}s")
    print()
print('=== per-defect (bad-proposal delta by corruption)')
for fold in payload['folds']:
    print(f"  [{fold['configuration']}]")
    for key in sorted(fold['models']):
        by = fold['models'][key]['summary'].get('by_corruption', {})
        parts = [f"{name}:{v['delta_iou_mean']:+.3f}(n={v['rows']})"
                 for name, v in by.items() if name != 'NONE']
        print(f"    {key}: " + '  '.join(parts))
print()
print('=== verdict payload')
print(json.dumps(payload['verdict'], indent=2)[:2500])
