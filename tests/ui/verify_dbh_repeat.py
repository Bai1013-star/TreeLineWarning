from pathlib import Path
import json
from desktop.workflow import DesktopWorkflow
from core.stage_b import dbh_identity_recovery as v3, dbh_raw_recovery as v2

out=Path('D:/TreeLineWarning/verification_real_20260930_weststation')
root=out/'北京西站南路_独立实测验收'
w=DesktopWorkflow(root)
result=w._current_result()
members=w._ensure_editor().final_reconciled()
# Re-establish the pre-V5 proposal baseline once in this test-only project.
v2.finalize_dbh(root,result,members)
one=v3.finalize_dbh(root,result,members)
two=v3.finalize_dbh(root,result,members)
values=lambda p:{r['global_tree_id']:(r['dbh_cm'],r['dbh_fit_status']) for r in p['trees']}
assert values(one)==values(two),(values(one),values(two))
audit=dict(passed=True,first=values(one),second=values(two),policy=two['measurement_policy'])
(out/'dbh_repeat_audit_20261001.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(audit,ensure_ascii=False),flush=True)
