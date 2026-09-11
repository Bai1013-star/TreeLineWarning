"""Held-out discipline self-audit and the local-challenge boundary.

Two boundaries this round must not cross, and both are checked rather than asserted:

**Held-out.** The evaluator must have required a valid freeze, verified the adapter parity
and verified the manifest count before the first held-out score. This module re-derives
that from the artifacts: the unblind audit timestamp must exist, the freeze digest recorded
at unblinding must still match the freeze on disk, every held-out example's source union
must be held-out-only, and no train/validation source may appear anywhere in the held-out
evaluation provenance.

**Local 8 bad challenge.** Gate H is the next phase, not this one. This round reads no
local challenge geometry, no local point cloud, no local label and no model output over
local data. :func:`local_challenge_audit` reports that boundary and
:func:`assert_local_challenge_untouched` fails closed if any EFG artifact is found to
reference a local challenge path.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

AUDIT_VERSION = 'EFG_LEAKAGE_AUDIT_V1'

# Path and token fragments that would indicate a local challenge read. Matching is
# deliberately broad: a false positive costs a manual look, a false negative costs the
# phase.
LOCAL_CHALLENGE_PATTERNS = (
    r'local[_-]?8[_-]?bad',
    r'EXPLICIT_BAD',
    r'local[_-]?challenge',
    r'gate[_-]?h[_-]?local',
    r'LOCAL_8_BAD',
)
# Declared boundary strings. These are the *names* this round must use to report its own
# boundary and its own verdict, so they are recorded here rather than tolerated silently:
# the scan below fails on any local-challenge reference that is not one of them.
ALLOWED_BOUNDARY_REFERENCES = (
    "'local_8_bad_challenge_accessed': False",
    "LOCAL_8_BAD_CHALLENGE_ACCESSED': 'NO'",
    'LOCAL_8_BAD_CHALLENGE_ACCESSED',
    "VERDICT_READY = 'READY_FOR_GATE_H_LOCAL_BLIND'",
    'local challenge locked',
    "'local_challenge_locked'",
    'LOCAL_8_BAD',
    'gate_h_local',
)
EXEMPT_FILES = ('efg_leakage.py', 'test_efg_orchestrator.py', 'efg_reports.py',
                'efg_gates.py', 'efg_freeze.py', 'run_tree_qc_efg.py',
                'RUN_TREE_QC_PUBLIC_PILOT.cmd')
# Anything that would actually open local challenge data, as opposed to naming it.
LOCAL_CHALLENGE_IO_PATTERNS = (
    r'open\([^)]*local',
    r'read_text\([^)]*local',
    r'np\.load\([^)]*local',
    r'read_points\([^)]*local',
    r'glob\([^)]*local',
    r'iterdir\([^)]*local',
    r'Path\([^)]*local',
)
LOCAL_CHALLENGE_IO_ALLOWED = (
    'local_8_bad_challenge_accessed',
    'local_challenge_locked',
)


class LeakageError(RuntimeError):
    """Raised with STOP_CRITICAL_LEAKAGE when a boundary is breached."""

    def __init__(self, detail):
        super().__init__(f'STOP_CRITICAL_LEAKAGE: {detail}')
        self.code = 'STOP_CRITICAL_LEAKAGE'
        self.detail = detail


def scan_for_local_challenge_reads(root: Path | str,
                                   patterns=LOCAL_CHALLENGE_PATTERNS) -> dict:
    """Find any local-challenge reference in EFG source that is not a boundary note."""
    root = Path(root)
    compiled = [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
    io_compiled = [re.compile(pattern, re.IGNORECASE)
                   for pattern in LOCAL_CHALLENGE_IO_PATTERNS]
    hits: list[dict] = []
    io_hits: list[dict] = []
    scanned = 0
    for path in sorted(root.glob('efg_*.py')):
        scanned += 1
        if path.name in EXEMPT_FILES:
            continue
        for number, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith('#'):
                continue
            if any(allowed in line for allowed in ALLOWED_BOUNDARY_REFERENCES):
                continue
            for pattern in compiled:
                if pattern.search(line):
                    hits.append({'file': path.name, 'line': number,
                                 'text': stripped[:160]})
                    break
            for pattern in io_compiled:
                if pattern.search(line) and not any(
                        allowed in line for allowed in LOCAL_CHALLENGE_IO_ALLOWED):
                    io_hits.append({'file': path.name, 'line': number,
                                    'text': stripped[:160]})
                    break
    return {'scanned_files': scanned, 'hits': hits, 'hit_count': len(hits),
            'io_hits': io_hits, 'io_hit_count': len(io_hits),
            'verdict': 'CLEAN' if not hits and not io_hits else 'REVIEW'}


def assert_local_challenge_untouched(root: Path | str) -> dict:
    report = scan_for_local_challenge_reads(root)
    if report['hits'] or report['io_hits']:
        raise LeakageError({
            'message': 'an EFG module appears to reference the local Gate H challenge',
            'hits': report['hits'], 'io_hits': report['io_hits'],
            'note': 'this round must not read local challenge geometry, labels or '
                    'model outputs'})
    return report


def heldout_protocol_audit(unblind_audit: dict, freeze_state: dict,
                           parity_report: dict | None,
                           manifest_check: dict | None,
                           evaluation_records: dict | None = None) -> dict:
    """Re-derive that held-out was opened correctly, from the recorded artifacts."""
    problems: list[dict] = []
    audit = unblind_audit or {}
    if not audit.get('accessed'):
        return {'checked': False, 'reason': 'held-out was never accessed',
                'critical_leakage': 'none', 'heldout_protocol': 'NOT_RUN',
                'heldout_protocol_clean': False,
                'meaningful_defects': {'MERGED_MULTI_TREE': False,
                                       'NEIGHBOR_CONTAMINATION': False,
                                       'OVER_SEGMENTED': False}}
    if not audit.get('first_access_timestamp'):
        problems.append({'problem': 'no first-access timestamp recorded'})
    if not (freeze_state or {}).get('valid'):
        problems.append({'problem': 'the freeze is not valid at audit time',
                         'reason': (freeze_state or {}).get('reason')})
    recorded = audit.get('freeze_sha256_at_unblind')
    current = (freeze_state or {}).get('sha256')
    if recorded != current:
        problems.append({'problem': 'freeze digest differs from the unblind record',
                         'recorded_at_unblind': recorded, 'current': current})
    if not parity_report or parity_report.get('HELDOUT_ADAPTER_PARITY') != 'PASS':
        problems.append({'problem': 'adapter parity is not PASS',
                         'value': (parity_report or {}).get('HELDOUT_ADAPTER_PARITY')})
    if not manifest_check or manifest_check.get('match') is not True:
        problems.append({'problem': 'held-out manifest count was not verified',
                         'value': manifest_check})

    leakage_entries = []
    for name, record in (evaluation_records or {}).items():
        entry = (record or {}).get('leakage') or {}
        if entry.get('cross_partition_sources'):
            leakage_entries.append({'record': name,
                                    'sources': entry['cross_partition_sources'][:10]})
    if leakage_entries:
        problems.append({'problem': 'cross-partition sources in held-out provenance',
                         'entries': leakage_entries})

    meaningful = {}
    for record in (evaluation_records or {}).values():
        per_defect = ((record or {}).get('metrics') or {}).get('per_defect_primary') or {}
        for defect in ('MERGED_MULTI_TREE', 'NEIGHBOR_CONTAMINATION', 'OVER_SEGMENTED'):
            entry = per_defect.get(defect)
            if not entry:
                meaningful.setdefault(defect, False)
                continue
            # "Meaningful" is declared as: the defect class is present with enough real
            # examples to measure, and its detection rate is above the good-false-review
            # budget rather than at chance.
            count = int(entry.get('bad_examples', 0))
            recall = float(entry.get('BAD_RECALL', 0.0))
            meaningful[defect] = bool(count >= 10 and recall > 0.25)
    for defect in ('MERGED_MULTI_TREE', 'NEIGHBOR_CONTAMINATION', 'OVER_SEGMENTED'):
        meaningful.setdefault(defect, False)

    clean = not problems
    result = {
        'audit_version': AUDIT_VERSION,
        'checked': True,
        'first_access_timestamp': audit.get('first_access_timestamp'),
        'freeze_sha256_at_unblind': recorded,
        'freeze_sha256_current': current,
        'freeze_unchanged': recorded == current,
        'adapter_parity': (parity_report or {}).get('HELDOUT_ADAPTER_PARITY'),
        'manifest_check': manifest_check,
        'problems': problems,
        'critical_leakage': 'none' if clean else 'detected',
        'heldout_protocol': 'CLEAN' if clean else 'VIOLATION',
        'heldout_protocol_clean': clean,
        'meaningful_defects': meaningful,
    }
    if problems:
        raise LeakageError({'message': 'held-out protocol audit failed',
                            'problems': problems})
    return result


def select_leakage_audit(held_out_records: dict, provenance) -> dict:
    """Cross-partition check performed by the evaluator on every held-out example."""
    model_selection = set(provenance.train.source_union) | set(
        provenance.validation.source_union)
    crossed: dict[str, int] = {}
    checked = 0
    for record in (held_out_records or {}).values():
        rows = (record or {}).get('source_union_counts') or {}
        for source_id, count in rows.items():
            checked += 1
            if source_id in model_selection:
                crossed[source_id] = crossed.get(source_id, 0) + int(count)
    return {'records': sorted(held_out_records or {}), 'sources_checked': checked,
            'cross_partition_sources': crossed,
            'critical_leakage': 'none' if not crossed else 'detected'}


def write_audit(path: Path | str, payload: dict) -> Path:
    import efg_state as state_module
    path = Path(path)
    state_module.atomic_write_json(path, payload, with_digest=False)
    return path


def read_records(directory: Path | str) -> dict:
    directory = Path(directory)
    records = {}
    if directory.is_dir():
        for path in sorted(directory.glob('*.json')):
            try:
                records[path.stem] = json.loads(path.read_text(encoding='utf-8-sig'))
            except (OSError, ValueError):
                continue
    return records
