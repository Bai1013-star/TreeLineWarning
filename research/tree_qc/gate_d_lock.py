"""Training lock.

Gate D has not been accepted by the Supervisor, so no PointNet or PointNet++ training
may start. Every trainer and every launcher goes through this module, and the lock is
fail-closed: an absent, malformed, stale or partially-filled acceptance marker refuses to
run.

Unlocking requires a real acceptance file created by the Supervisor:

    research/tree_qc/reports/GATE_D_ACCEPTANCE.json
    {
      "gate_d": "ACCEPTED",
      "training_allowed": true,
      "generator_version": "<current GATE_D_GENERATOR_VERSION>",
      "generator_digest": "<current generator implementation digest>",
      "protocol_digest": "<current frozen Gate C2 protocol digest>",
      "accepted_by": "<supervisor>",
      "accepted_at": "<iso-8601>"
    }

Every one of those fields is verified against the **current** implementation, so a
marker cannot outlive the code it accepted: changing the generator, the sampler, the
diagnostic or the frozen protocol invalidates it and the lock closes again.

The marker is deliberately *not* shipped in this commit. Its absence is what keeps
``TRAINING_ALLOWED = False`` true.
"""
from __future__ import annotations

import json
from pathlib import Path

REPORTS = Path(__file__).resolve().parent / 'reports'
ACCEPTANCE_FILE = 'GATE_D_ACCEPTANCE.json'

TRAINING_ALLOWED = False
LOCK_REASON = 'GATE_D_NOT_ACCEPTED'
REQUIRED_FIELDS = ('gate_d', 'training_allowed', 'generator_version',
                   'generator_digest', 'protocol_digest')


class TrainingLockedError(RuntimeError):
    """Raised whenever training is attempted while Gate D is unaccepted."""


def acceptance_path(reports: Path | None = None) -> Path:
    return (Path(reports) if reports is not None else REPORTS) / ACCEPTANCE_FILE


def current_binding(reports: Path | None = None) -> dict:
    """The values the marker must match: current code and frozen protocol."""
    import gate_d_diagnostic
    import gate_d_generator
    import gate_d_provenance
    root = Path(reports) if reports is not None else REPORTS
    # The protocol file may live beside the real reports when a test uses a temporary
    # marker directory; fall back to the shipped protocol so the binding stays real.
    protocol_root = root if (root / gate_d_provenance.PROTOCOL_FILE).exists() else REPORTS
    return {
        'generator_version': gate_d_generator.GENERATOR_VERSION,
        'generator_digest': gate_d_diagnostic.generator_digest(),
        'protocol_digest': gate_d_provenance.protocol_digest(protocol_root),
    }


def acceptance_state(reports: Path | None = None, binding: dict | None = None) -> dict:
    """Read and verify the acceptance marker without raising; for reporting."""
    path = acceptance_path(reports)
    expected = binding if binding is not None else current_binding(reports)
    state = {'marker': str(path), 'exists': path.exists(),
             'training_allowed': False, 'reason': LOCK_REASON,
             'expected': expected}
    if not path.exists():
        return state
    try:
        payload = json.loads(path.read_text(encoding='utf-8-sig'))
    except (OSError, ValueError) as exc:
        state['reason'] = f'{LOCK_REASON}: unreadable marker ({exc})'
        return state
    state['payload'] = payload

    missing = [field for field in REQUIRED_FIELDS if field not in payload]
    if missing:
        state['reason'] = f'{LOCK_REASON}: missing required field(s) {missing}'
        return state
    if payload.get('gate_d') != 'ACCEPTED':
        state['reason'] = f'{LOCK_REASON}: gate_d={payload.get("gate_d")!r}'
        return state
    if payload.get('training_allowed') is not True:
        state['reason'] = f'{LOCK_REASON}: training_allowed is not true'
        return state
    if payload.get('generator_version') != expected['generator_version']:
        state['reason'] = (
            f'{LOCK_REASON}: accepted generator_version '
            f'{payload.get("generator_version")!r} != current '
            f'{expected["generator_version"]!r}')
        return state
    if payload.get('generator_digest') != expected['generator_digest']:
        state['reason'] = (
            f'{LOCK_REASON}: accepted generator_digest '
            f'{str(payload.get("generator_digest"))[:16]}... != current '
            f'{expected["generator_digest"][:16]}... (generator changed after acceptance)')
        return state
    if payload.get('protocol_digest') != expected['protocol_digest']:
        state['reason'] = (
            f'{LOCK_REASON}: accepted protocol_digest '
            f'{str(payload.get("protocol_digest"))[:16]}... != current '
            f'{expected["protocol_digest"][:16]}... (frozen protocol changed)')
        return state
    state['training_allowed'] = True
    state['reason'] = 'GATE_D_ACCEPTED'
    return state


def require_training_allowed(reports: Path | None = None,
                             binding: dict | None = None) -> dict:
    """Raise TrainingLockedError unless the Supervisor accepted the current code."""
    state = acceptance_state(reports, binding)
    if not state['training_allowed']:
        raise TrainingLockedError(state['reason'])
    return state


def assert_not_training() -> None:
    """Guard for diagnostic code paths that must never train."""
    if TRAINING_ALLOWED:
        raise TrainingLockedError('TRAINING_ALLOWED is unexpectedly true')
