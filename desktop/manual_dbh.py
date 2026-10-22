"""Operator field measurements, separate from immutable LAS evidence."""
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
from threading import RLock
from uuid import uuid4

_lock = RLock()
SOURCE = 'MANUAL_FIELD_MEASUREMENT'
CORRECTION = 'MANUAL_FIELD_CORRECTION'


def diameter_or_none(value):
    if value is None or isinstance(value, str) and value.strip().lower() in ('', 'no', '未获取', '未测得'):
        return None
    if isinstance(value, bool):
        raise ValueError('胸径请填 No 或现场实测数值（cm）')
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError('胸径请填 No 或现场实测数值（cm）') from error
    if not math.isfinite(number) or not 0 < number <= 500:
        raise ValueError('人工胸径必须大于 0 且不超过 500 cm；未测量请填 No')
    return number


def _path(root):
    return Path(root) / 'runtime' / 'warning' / 'manual_dbh.json'


def read_measurements(root, run_id):
    path = _path(root)
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding='utf-8'))
    if data.get('schema') != 'treeline-manual-dbh-v1':
        raise ValueError('人工胸径档案格式不正确，未采用其中的数值')
    if not run_id or data.get('analysis_run_id') != run_id:
        return {}
    return data.get('trees', {})


def apply_measurements(rows, root, run_id):
    saved = read_measurements(root, run_id)
    for row in rows:
        original = diameter_or_none(row.get('diameter'))
        row['point_cloud_diameter'] = original
        row['diameter_source'] = 'ORIGINAL_LAS_1P30M' if original is not None else 'UNAVAILABLE'
        row['manual_diameter_cm'] = None
        if row['tree_key'] in saved:
            record = saved[row['tree_key']]
            value = diameter_or_none(record.get('diameter_cm'))
            if value is not None and ((original is None and record.get('source') == SOURCE) or record.get('source') == CORRECTION):
                row.update(diameter=value, manual_diameter_cm=value,
                           diameter_source=record['source'], diameter_measured_at=record.get('saved_at'))


def save_measurement(root, run_id, rows, tree_key, value, *, correct_existing=False):
    if not run_id:
        raise ValueError('当前分析任务尚未绑定，不能保存人工胸径')
    row = next((r for r in rows if r['tree_key'] == tree_key), None)
    if row is None:
        raise ValueError('该树号不在当前项目结果中')
    original = row.get('point_cloud_diameter', row.get('diameter'))
    if original is not None and not correct_existing:
        raise ValueError('点云已有有效胸径，不可用人工补填覆盖原始测量')
    number = diameter_or_none(value)
    with _lock:
        saved = read_measurements(root, run_id)
        if number is None:
            saved.pop(tree_key, None)
        else:
            saved[tree_key] = dict(diameter_cm=number, source=CORRECTION if original is not None else SOURCE,
                height_above_original_ground_m=1.30,
                saved_at=datetime.now(timezone.utc).isoformat(), tree_code=row['tree_code'])
        path = _path(root)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_name(path.name+'.'+uuid4().hex+'.tmp')
        temp.write_text(json.dumps(dict(schema='treeline-manual-dbh-v1',
            analysis_run_id=run_id, trees=saved), ensure_ascii=False, indent=2), encoding='utf-8')
        os.replace(temp, path)
    return dict(tree_key=tree_key, manual_diameter_cm=number,
                effective_diameter_cm=number if number is not None else original,
                diameter_source=(CORRECTION if original is not None else SOURCE) if number is not None
                    else 'ORIGINAL_LAS_1P30M' if original is not None else 'UNAVAILABLE')
