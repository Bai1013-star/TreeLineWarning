"""Ad-hoc inspection helper for a SITE_FINGERPRINT report."""
from __future__ import annotations

import json
import sys
from pathlib import Path

path = Path(sys.argv[1] if len(sys.argv) > 1 else
            Path(__file__).resolve().parent / 'reports' / 'SITE_FINGERPRINT_QUICK.json')
payload = json.loads(path.read_text(encoding='utf-8'))
print('split', payload['split_counts'])
print('trees', payload['trees_used'], 'points', payload['points'])
for name, entry in payload['representations'].items():
    history = entry['history']
    picks = [0, 5, 20, 40, len(history) - 1]
    print(f'{name}  in_ch={entry["in_channels"]} params={entry["parameters"]}')
    print('   loss ', [round(history[i]['train_loss'], 4) for i in picks])
    print('   val  ', [round(history[i]['val_accuracy'], 3) for i in picks])
    print('   train_acc', round(entry['train']['site_accuracy'], 3),
          'val_acc', round(entry['validation']['site_accuracy'], 3),
          'test_acc', round(entry['test']['site_accuracy'], 3),
          'macroF1', round(entry['test_macro_f1'], 3))
    print('   confusion', entry['test']['confusion_matrix'], entry['test']['class_order'])
