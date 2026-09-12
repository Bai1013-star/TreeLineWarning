"""One-off lint fixer: report and fix specific, reviewable lines.

This exists because the project's ruff profile wants ``zip(..., strict=)`` and
several modules in this round iterate channel pairs where the two sequences are
always the same length by construction. Each edit is printed.
"""
from __future__ import annotations

import re
from pathlib import Path

REPLACEMENTS = [
    ('v2_models.py',
     'for first, second in zip(channels, channels[1:]):',
     'for first, second in zip(channels, channels[1:], strict=False):'),
    ('v2_refiner_data.py',
     'for key, start, stop in zip(unique, starts, stops):',
     'for key, start, stop in zip(unique, starts, stops, strict=True):'),
    ('v2_refiner_data.py',
     'for group, quota in zip(groups, quotas):',
     'for group, quota in zip(groups, quotas, strict=True):'),
    ('v2_site_fingerprint.py',
     'for truth, guess in zip(labels.numpy(), predicted.numpy()):',
     'for truth, guess in zip(labels.numpy(), predicted.numpy(), strict=True):'),
    ('test_v2_feasibility.py',
     'for delta, current in zip(delta_values, current_values):',
     'for delta, current in zip(delta_values, current_values, strict=True):'),
]


def main() -> int:
    root = Path(__file__).resolve().parent
    for name, old, new in REPLACEMENTS:
        path = root / name
        text = path.read_text(encoding='utf-8')
        if old not in text:
            print(f'SKIP {name}: pattern not present')
            continue
        text = text.replace(old, new)
        path.write_text(text, encoding='utf-8')
        print(f'FIXED {name}: {old}  ->  {new}')

    # Remove the two unused locals ruff flagged.
    path = root / 'v2_refiner_data.py'
    text = path.read_text(encoding='utf-8')
    text = re.sub(r'\n\s*cluster = self\.by_target\[example\.target_id\]\n(\s*)xyz, gt, current',
                  r'\n\1xyz, gt, current', text)
    path.write_text(text, encoding='utf-8')
    print('FIXED v2_refiner_data.py: dropped an unused cluster lookup')

    path = root / 'v2_roi_library.py'
    text = path.read_text(encoding='utf-8')
    text = text.replace(
        "    gt_other = (view.instance > 0) & (~view.target_mask)\n"
        "    gt_background = view.instance < 0\n\n", '')
    path.write_text(text, encoding='utf-8')
    print('FIXED v2_roi_library.py: dropped two unused mask locals')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
