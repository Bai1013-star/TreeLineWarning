"""Real unchanged-members undo must not discard original-LAS seed proposals."""
from pathlib import Path
import json
import os
from desktop.workflow import DesktopWorkflow
from core.stage_b import dbh_recovery as v1, dbh_identity_recovery as v5

out = Path(os.environ.get('AUDIT_OUTPUT', 'D:/TreeLineWarning/verification_real_20260930_weststation'))
root = Path(os.environ.get('AUDIT_PROJECT', str(out / '北京西站南路_独立实测验收')))
w = DesktopWorkflow(root)
editor = w._ensure_editor()
members = editor.final_reconciled()
before = v1.load_finalized_dbh(root, w._current_result(), members)
assert sum(r['dbh_cm'] is not None for r in before.values()) == int(os.environ.get('AUDIT_EXPECTED_DBH', '3'))
signature = lambda p: {gid: (r['dbh_cm'], r['dbh_fit_status']) for gid,r in p.items()}
count = len(members.final_original_index)
span, ids = next(iter(editor.tree_ids_by_span.items()))
editor.delete(span, [ids[0]])
assert len(editor.final_reconciled().final_original_index) < count
editor.undo()
assert len(editor.final_reconciled().final_original_index) == count
w.finalize()
after = v1.load_finalized_dbh(root, w._current_result(), w._ensure_editor().final_reconciled())
assert signature(after) == signature(before), (signature(before), signature(after))
v5.finalize_dbh(root, w._current_result(), w._ensure_editor().final_reconciled())
again = v1.load_finalized_dbh(root, w._current_result(), w._ensure_editor().final_reconciled())
assert signature(again) == signature(before)
audit = dict(passed=True, restored_members=count, before=signature(before),
             after_undo_refinalize=signature(after), repeated=signature(again),
             note='No V2 reset; old values not copied; original LAS reread and verified')
(out/'undo_dbh_audit_20261001.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2),encoding='utf-8')
print(json.dumps(audit,ensure_ascii=False),flush=True)
