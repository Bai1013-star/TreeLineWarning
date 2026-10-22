"""Real LAS acceptance: actual source rows, physical 1.30 m slab, export/reopen."""
from pathlib import Path
import json
import os
import numpy as np
import laspy
import openpyxl
from desktop.workflow import DesktopWorkflow
from core.stage_b import dbh_identity_recovery as recovery
from core.stage_b import dbh_recovery as baseline
from core.stage_b.dbh_evidence import read_json, confined
from desktop.tree_display_numbering_extensions import _current_display_identities

output = Path(os.environ.get('AUDIT_OUTPUT', 'D:/TreeLineWarning/verification_real_20260930_weststation'))
root = Path(os.environ.get('AUDIT_PROJECT', str(output / '北京西站南路_独立实测验收')))
workflow = DesktopWorkflow(root)
if os.environ.get('AUDIT_FINALIZE') == '1':
    # A preceding real-UI edit/undo changes the persisted review revision.
    # Exercise the actual finalize entrypoint before measuring/export/reopen.
    workflow.finalize()
result = workflow._current_result()
editor = workflow._ensure_editor()
members = editor.final_reconciled()
prior = baseline.load_finalized_dbh(root, result, members)
payload = recovery.finalize_dbh(root, result, members)
published = baseline.load_finalized_dbh(root, result, members)
if os.environ.get('AUDIT_ASSERT_STABLE') == '1':
    assert {gid: row['dbh_cm'] for gid, row in prior.items()} == {gid: row['dbh_cm'] for gid, row in published.items()}
assert not any(r['dbh_fit_status'] == 'DBH_PROOF_INVALID_OR_STALE' for r in published.values())
identities = _current_display_identities(workflow, editor)
_, _, _, origin = baseline._context(root, result)
las_path = Path(os.environ.get('AUDIT_LAS', 'E:/treeline/9.20北京西站南路-1_colorized.las'))
with laspy.open(las_path) as reader:
    header = reader.header
    source = np.memmap(las_path, mode='r', dtype=header.point_format.dtype(),
                       offset=header.offset_to_point_data, shape=(header.point_count,))
    checks = []
    with np.load(confined(root, payload['evidence_reference']), allow_pickle=False) as archive:
        for row in payload['trees']:
            gid = row['global_tree_id']
            detail = dict(id=identities[gid]['display_tree_id'], dbh_cm=row['dbh_cm'],
                          method=row['dbh_method'], status=row['dbh_fit_status'])
            if row['dbh_cm'] is not None:
                mask = archive['global_tree_id'] == gid
                idx = archive['original_index'][mask]
                exact = np.column_stack([source[name][idx].astype(float) * header.scales[i]
                                         + header.offsets[i] for i, name in enumerate(('X', 'Y', 'Z'))])
                saved = archive['xyz_local_m'][mask] + origin
                difference = float(np.max(np.abs(saved - exact)))
                assert difference < 1e-10
                ground_z = row['dbh_ground_z_absolute_m']
                target_z = row['dbh_measurement_z_absolute_m']
                assert abs(target_z - ground_z - 1.30) < 1e-8
                assert np.max(np.abs(exact[:, 2] - target_z)) <= .050001
                assert len(idx) == row['dbh_point_count'] >= 12
                detail.update(ground_z_m=ground_z, target_z_m=target_z,
                              actual_selected_hag_bounds_m=[float((exact[:, 2]-ground_z).min()),
                                                            float((exact[:, 2]-ground_z).max())],
                              source_points=len(idx), original_coordinate_max_error_m=difference,
                              first_original_point_indices=idx[:12].tolist())
            checks.append(detail)
    del source
measured_count = sum(c['dbh_cm'] is not None for c in checks)
assert measured_count > 0
export = workflow.export_prediction()
workbook = openpyxl.load_workbook(export, data_only=True)
headers = [cell.value for cell in workbook.active[1]]
rows = [dict(zip(headers, row)) for row in workbook.active.iter_rows(min_row=2, values_only=True)]
assert len(rows) == len(published)
assert sum(row.get('dbh_cm') is not None for row in rows) == measured_count
assert 'dbh_measurement_z_absolute_m' in headers
for row in rows:
    if row['dbh_cm'] is not None:
        assert abs(row['dbh_measurement_z_absolute_m']-row['dbh_ground_z_absolute_m']-1.30)<1e-8
        assert row['dbh_coordinate_source']=='DIRECT_ORIGINAL_LAS_INTEGER_COORDINATES'
reopened = DesktopWorkflow(root)
loaded = baseline.load_finalized_dbh(root, reopened._current_result(), reopened._ensure_editor().final_reconciled())
assert {gid: row['dbh_cm'] for gid, row in loaded.items()} == {gid: row['dbh_cm'] for gid, row in published.items()}
audit = dict(policy=payload['measurement_policy'], source=str(las_path),
             source_sha256=payload['binding']['source_identity']['sha256'],
             source_point_count=header.point_count, trees=checks,
             actual_export=str(export), exported_rows=len(rows),
             reopen_preserves_values=True,
             note=f'薄层厚度10cm，以原始地面以上1.30m为中心；{len(checks)-measured_count}棵未证明树干身份，不补零、不用树高公式代替胸径。')
(output / os.environ.get('AUDIT_DBH_REPORT', '原始LAS_1.30m胸径实测验证.json')).write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)
