"""Training lock.

Gate D has not been accepted by the Supervisor, so no PointNet or PointNet++ training
may start. Every trainer and every launcher goes through this module, and the lock is
fail-closed: an absent, malformed, stale or partially-filled acceptance marker refuses
to run.

Unlocking requires a real acceptance file created by the Supervisor:

    research/tree_qc/reports/GATE_D_ACCEPTANCE.json
    {
      "gate_d": "ACCEPTED",
      "training_allowed": true,
      "generator_digest": "<GATE_D_GENERATOR_DIGEST from the diagnostic>",
      "accepted_by": "<supervisor>",
      "accepted_at": "<iso-8601>"
    }

The marker is deliberately *not* shipped in this commit. Its absence is what keeps
``TRAINING_ALLOWED = FALSE`` true.
"""
from __future__ import annotations

import json
from pathlib import Path

REPORTS = Path(__file__).resolve().parent / 'reports'
ACCEPTANCE_FILE = 'GATE_D_ACCEPTANCE.json'

TRAINING_ALLOWED = False
LOCK_REASON = 'GATE_D_NOT_ACCEPTED'


class TrainingLockedError(RuntimeError):
    """Raised whenever training is attempted while Gate D is unaccepted."""


def acceptance_path(reports: Path | None = None) -> Path:
    return (Path(reports) if reports is not None else REPORTS) / ACCEPTANCE_FILE


def acceptance_state(reports: Path | None = None,
                     expected_generator_digest: str | None = None) -> dict:
    """Read the acceptance marker without raising; for reporting."""
    path = acceptance_path(reports)
    state = {'marker': str(path), 'exists': path.exists(),
             'training_allowed': False, 'reason': LOCK_REASON}
    if not path.exists():
        return state
    try:
        payload = json.loads(path.read_text(encoding='utf-8-sig'))
    except (OSError, ValueError) as exc:
        state['reason'] = f'{LOCK_REASON}: unreadable marker ({exc})'
        return state
    state['payload'] = payload
    if payload.get('gate_d') != 'ACCEPTED':
        state['reason'] = f'{LOCK_REASON}: gate_d={payload.get("gate_d")!r}'
        return state
    if payload.get('training_allowed') is not True:
        state['reason'] = f'{LOCK_REASON}: training_allowed is not true'
        return state
    digest = payload.get('generator_digest')
    if expected_generator_digest is not None and digest != expected_generator_digest:
        state['reason'] = (f'{LOCK_REASON}: accepted generator digest {digest!r} '
                           f'does not match current {expected_generator_digest!r}')
        return state
    state['training_allowed'] = True
    state['reason'] = 'GATE_D_ACCEPTED'
    return state


def require_training_allowed(reports: Path | None = None,
                             expected_generator_digest: str | None = None) -> dict:
    """Raise TrainingLockedError unless the Supervisor accepted Gate D."""
    state = acceptance_state(reports, expected_generator_digest)
    if not state['training_allowed']:
        raise TrainingLockedError(state['reason'])
    return state


def assert_not_training() -> None:
    """Guard for diagnostic code paths that must never train."""
    if TRAINING_ALLOWED:
        raise TrainingLockedError('TRAINING_ALLOWED is unexpectedly true')
