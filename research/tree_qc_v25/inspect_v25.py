"""Extract the final V2.5 numbers, separating what the gate did from what the model did."""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
payload = json.loads((HERE / 'reports' / 'V25_GO_NO_GO.json').read_text(encoding='utf-8'))

print('=== RESULT A: frozen V2 full-mask refiner (read, not retrained)')
baseline = payload['result_a_v2_frozen_full_mask']
for configuration, entry in baseline['configurations'].items():
    for key, model in sorted(entry['models'].items()):
        print(f"  {configuration:<26} {key} {model['name']:<34} "
              f"delta {model['delta_iou_mean']:+.4f}  good "
              f"{model['good_mean_delta_iou']:+.4f}  bad "
              f"{model['bad_mean_delta_iou']:+.4f}")

for fold in payload['folds']:
    print()
    print(f"=== {fold['configuration']}")
    print(f"  training: {fold['training']['epochs_run']} epochs run of "
          f"{fold['training']['epochs_budget']}, best epoch "
          f"{fold['training']['best_epoch']}, seed {fold['training']['seed']}")
    sampling = fold['training']['sampling']
    print(f"  sampling: {sampling['rows']} rows, GOOD {sampling['good_rows']} "
          f"({sampling['realised_good_fraction']:.3f} vs target "
          f"{sampling['target_good_fraction']}), BAD {sampling['bad_rows']}, "
          f"repeat {sampling['good_repeat']}")
    gate = fold['gate']
    print(f"  gate: fitted={gate['fitted']} threshold="
          f"{gate.get('effective_threshold')} validation coverage="
          f"{gate['coverage']:.4f} validation success={gate['success_rate']:.4f}")
    print(f"  gate frozen: sha256 {fold['gate_frozen_sha256'][:16]}")
    for which in ('ungated', 'gated'):
        summary = fold['test'][which]
        print(f"  --- test {which.upper()}")
        print(f"     ALL   current {summary['ALL']['current_iou_mean']:.4f} -> "
              f"refined {summary['ALL']['refined_iou_mean']:.4f}  "
              f"delta {summary['ALL']['delta_iou_mean']:+.4f}")
        print(f"     GOOD  delta {summary['GOOD_MEAN_DELTA_IOU']:+.4f}  "
              f"degradation rate {summary['GOOD_DEGRADATION_RATE']:.4f}")
        print(f"     BAD   delta {summary['BAD_MEAN_DELTA_IOU']:+.4f}  "
              f"improvement rate {summary['BAD_IMPROVEMENT_RATE']:.4f}")
        if which == 'ungated':
            print(f"     no_op {summary['NO_OP_RATE']:.4f}  change_f1 "
                  f"{summary['change_region_f1_mean']:.4f}")
        else:
            print(f"     no_op {summary['NO_OP_RATE']:.4f}  auto_edit_count "
                  f"{summary['AUTO_EDIT_COUNT']}")
        if summary.get('AUTO_EDIT_COVERAGE') is not None:
            print(f"     AUTO_EDIT coverage {summary['AUTO_EDIT_COVERAGE']:.4f} "
                  f"success {summary['AUTO_EDIT_SUCCESS_RATE']} "
                  f"count {summary['AUTO_EDIT_COUNT']}")
        for label in ('ADD', 'REMOVE'):
            precision = summary.get(f'{label}_PRECISION')
            if precision is None:
                print(f'     {label:<7} no auto-edited row, metrics undefined')
            else:
                print(f"     {label:<7} P {precision:.4f} R "
                      f"{summary[f'{label}_RECALL']:.4f} "
                      f"F1 {summary[f'{label}_F1']:.4f}")

print()
print('=== VERDICT')
print(json.dumps(payload['verdict'], indent=2)[:2200])
