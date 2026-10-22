"""Inspect actual DBH finalization in the isolated acceptance project only."""
from pathlib import Path
import json
import traceback
from desktop.workflow import DesktopWorkflow
from core.stage_b.dbh_identity_recovery import finalize_dbh
import sys

root = Path('D:/TreeLineWarning/verification_real_20260930_weststation/北京西站南路_独立实测验收')
workflow = DesktopWorkflow(root)
result = workflow._current_result()
editor = workflow._ensure_editor()
try:
    if '--editor-finalize' in sys.argv:
        editor.mark_reviewed()
        editor.finalize(progress=lambda stage: print(stage, flush=True))
        print(json.dumps(editor.dbh_recovery_status, ensure_ascii=False), flush=True)
        if editor.dbh_recovery_status['status'] != 'COMPLETE':
            raise RuntimeError(editor.dbh_recovery_status['reason'])
    payload = finalize_dbh(root, result, editor.final_reconciled())
    print(json.dumps([{'id': r['global_tree_id'], 'dbh': r['dbh_cm'], 'method': r['dbh_method'], 'reason': r['dbh_fit_status']} for r in payload['trees']], ensure_ascii=False), flush=True)
except Exception:
    traceback.print_exc()
    raise
