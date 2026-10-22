"""Read-only prototype on the previous independently recovered original span."""
import json
import os
import time
from pathlib import Path
import numpy as np
from algorithms.vegetation.coarse_to_fine import RawRegion
from algorithms.spans.raw_stem_first import detect_raw_stem_axes, segment_from_raw_stems

root = Path('D:/TreeLineWarning/verification_real_20260930_weststation/20261001_从原始点云重新分析')
source_root = next((root/'artifacts/project-runs/source-shared').iterdir())
span_root = next((next((root/'artifacts/project-runs/batches').glob('analysis-*'))).glob('span-*'))
region_root = span_root/'regions/veg_final'
region = RawRegion(str(region_root/'xyz_local.bin'), str(region_root/'original_index.bin'),
    str(region_root/'classification.bin'), (region_root/'classification.bin').stat().st_size,
    (0., 0., 0.), (0., 0., 0.), (0., 0., 0.))
ground = np.load(source_root/'coarse_ground.npy')
start = time.monotonic()
out = Path('D:/TreeLineWarning/verification_real_20260930_weststation/stem_first_prototype_20261001.json')
if os.environ.get('AUDIT_REUSE_STEMS') == '1':
    from algorithms.spans.stem_guided_hybrid import StemEvidence
    cached = json.loads(out.read_text(encoding='utf-8'))
    stems = tuple(StemEvidence(**{**s, 'xy_local_m': tuple(s['xy_local_m'])}) for s in cached['stems'])
    diagnostic = cached['detection']
else:
    stems, diagnostic = detect_raw_stem_axes(region, ground, tuple(region.origin_xyz))
print(json.dumps(dict(raw_axis_count=len(stems), diagnostic=diagnostic, elapsed=time.monotonic()-start)),flush=True)
with np.load(span_root/'tree/legacy_reference/tree_fine.npz') as data:
    xyz, hag = data['fine_xyz_local_m'], data['fine_hag_m']
result = segment_from_raw_stems(xyz, hag, stems)
payload = dict(stems=[s.to_dict() for s in stems], detection=diagnostic, growth=result.diagnostics,
    trees=[c.to_dict() for c in result.candidates], elapsed=time.monotonic()-start)
out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(dict(growth=result.diagnostics, trees=[dict(id=c.tree_id, height=float(hag[c.member_rows].max()), metrics=c.structural_metrics) for c in result.candidates], elapsed=payload['elapsed'])), flush=True)
