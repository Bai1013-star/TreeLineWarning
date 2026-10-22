"""Missing geometry supplied by an operator; original measurements stay intact."""
from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from uuid import uuid4

FIELDS = ('height', 'crown_width', 'horizontal_distance', 'vertical_distance', 'clearance_distance')
SOURCE = 'MANUAL_FIELD_MEASUREMENT'
CORRECTION = 'MANUAL_FIELD_CORRECTION'
_lock = RLock()


def measurement_or_none(field, value):
    if field not in FIELDS:
        raise ValueError('不支持的人工补填字段')
    if value is None or isinstance(value, str) and value.strip().lower() in ('', 'no', '未获取', '未测得'):
        return None
    if isinstance(value, bool):
        raise ValueError('人工测量必须是有效数值')
    try:
        number = float(value)
    except (ValueError, TypeError, OverflowError) as error:
        raise ValueError('人工测量必须是有效数值') from error
    if not math.isfinite(number) or number < 0 or field in ('height', 'crown_width') and number <= 0:
        raise ValueError('树高和冠幅须大于 0；距离须大于等于 0；未测量可留空')
    return number


def _path(root):
    return Path(root) / 'runtime/warning/manual_measurements.json'


def _read(root, run_id):
    path = _path(root)
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding='utf8'))
    if data.get('schema') != 'treeline-manual-geometry-v1':
        raise ValueError('人工测量档案格式不正确')
    return data.get('trees', {}) if run_id and data.get('analysis_run_id') == run_id else {}


def apply_measurements(rows, root, run_id):
    saved = _read(root, run_id)
    for row in rows:
        original = {}
        for field in FIELDS:
            try:
                original[field] = measurement_or_none(field, row.get(field))
            except ValueError:
                original[field] = None
            row[field] = original[field]
        row['point_cloud_measurements'] = original
        sources = row.setdefault('measurement_sources', {})
        for field, value in original.items():
            sources[field] = 'POINT_CLOUD' if value is not None else 'UNAVAILABLE'
            record = saved.get(row['tree_key'], {}).get(field, {})
            if (value is None and record.get('source') == SOURCE) or record.get('source') == CORRECTION:
                manual = measurement_or_none(field, record.get('value'))
                if manual is not None:
                    row[field] = manual
                    sources[field] = record['source']
        row['prediction_ready'] = row.get('height') is not None and row.get('clearance_distance') is not None


def save_measurements(root, run_id, rows, tree_key, fields, *, correct_existing=False):
    if not run_id:
        raise ValueError('当前分析任务尚未绑定，不能保存人工测量')
    row = next((r for r in rows if r['tree_key'] == tree_key), None)
    if row is None or not isinstance(fields, dict) or not fields:
        raise ValueError('请选择当前树号及需要补填的字段')
    original = row.get('point_cloud_measurements', row)
    parsed = {}
    for field, value in fields.items():
        number = measurement_or_none(field, value)
        if original.get(field) is not None and not correct_existing:
            raise ValueError('点云已有有效测量，不可覆盖原始数值')
        parsed[field] = number
    with _lock:
        saved = _read(root, run_id)
        record = saved.setdefault(tree_key, {})
        for field, number in parsed.items():
            if number is None:
                record.pop(field, None)
            else:
                source = CORRECTION if original.get(field) is not None else SOURCE
                record[field] = dict(value=number, source=source, saved_at=datetime.now(timezone.utc).isoformat())
        if not record:
            saved.pop(tree_key, None)
        path = _path(root)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + '.' + uuid4().hex + '.tmp')
        temporary.write_text(json.dumps(dict(schema='treeline-manual-geometry-v1', analysis_run_id=run_id,
                                            trees=saved), ensure_ascii=False, indent=2), encoding='utf8')
        os.replace(temporary, path)
    effective = {field: (original.get(field) if number is None else number) for field, number in parsed.items()}
    sources = {field: (('POINT_CLOUD' if original.get(field) is not None else 'UNAVAILABLE') if number is None
                      else CORRECTION if original.get(field) is not None else SOURCE) for field, number in parsed.items()}
    return dict(tree_key=tree_key, manual_fields=parsed, effective_fields=effective,
                measurement_sources=sources, measurement_source=SOURCE)
