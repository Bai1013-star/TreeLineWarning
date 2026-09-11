"""EFG persistent state: crash-safe, digest-verified, resume-authoritative.

The Public Pilot is one long unattended run that may be interrupted at any moment by a
power cut, a Windows restart, a closed terminal or a Python exception. Everything the
run has already decided therefore lives on disk, in one file per concern, written with
an atomic ``temp -> replace`` so a torn write can never leave a half-file behind.

Two invariants make resumption safe rather than merely convenient:

1. **Every state file carries its own SHA-256.** A state file that does not hash to its
   recorded digest is corrupt and is refused, never silently re-used.
2. **Every completeness claim is bound to the inputs that produced it.** A stage is
   ``COMPLETE`` only when the digests it was produced under — Gate D binding, dataset
   digest, config digest, checkpoint digest, model, seed, points, orchestrator version —
   still match. Anything else is ``STALE`` and is recomputed. Nothing is silently
   re-used, and nothing is silently re-run either.

This module never imports torch, the generator or the trainer: it is pure bookkeeping so
that it can be unit-tested without a GPU and cannot be affected by what it records.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

STATE_VERSION = 'EFG_STATE_V1'
AUTOPILOT_STATE = 'AUTOPILOT_STATE.json'

COMPLETE = 'COMPLETE'
RUNNING = 'RUNNING'
PENDING = 'PENDING'
FAILED = 'FAILED'
STALE = 'STALE'
SKIPPED = 'SKIPPED'

STAGE_STATES = (PENDING, RUNNING, COMPLETE, FAILED, STALE, SKIPPED)


class StateError(RuntimeError):
    """Raised when persisted state is corrupt, or a resume would be unsafe."""


def canonical(payload) -> str:
    return json.dumps(payload, sort_keys=True, separators=(',', ':'), default=str)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def digest_payload(payload: dict) -> str:
    """Digest of a payload, excluding its own digest field."""
    body = {k: v for k, v in payload.items() if k not in ('state_digest',)}
    return sha256_text(canonical(body))


def digest_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        while chunk := handle.read(4 * 1024**2):
            digest.update(chunk)
    return digest.hexdigest()


def utc_timestamp() -> str:
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def local_timestamp() -> str:
    """Local wall clock with the machine's real UTC offset (Asia/Shanghai here)."""
    return time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime()) + _offset()


def _offset() -> str:
    if time.daylight and time.localtime().tm_isdst > 0:
        delta = -time.altzone
    else:
        delta = -time.timezone
    sign = '+' if delta >= 0 else '-'
    delta = abs(int(delta))
    return f'{sign}{delta // 3600:02d}:{(delta % 3600) // 60:02d}'


def atomic_write_json(path: Path | str, payload: dict, with_digest: bool = True) -> str:
    """Write JSON atomically and return the digest that was recorded."""
    path = Path(path)
    body = dict(payload)
    if with_digest:
        body['state_digest'] = digest_payload(body)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.tmp{os.getpid()}')
    text = json.dumps(body, indent=2, sort_keys=False, default=str)
    with tmp.open('w', encoding='utf-8', newline='\n') as handle:
        handle.write(text + '\n')
        handle.flush()
        os.fsync(handle.fileno())
    tmp.replace(path)
    return body.get('state_digest', '')


def atomic_write_text(path: Path | str, text: str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.tmp{os.getpid()}')
    with tmp.open('w', encoding='utf-8', newline='\n') as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    tmp.replace(path)
    return path


def read_json(path: Path | str, verify: bool = True) -> dict | None:
    """Read a state file, refusing a corrupt or tampered one."""
    path = Path(path)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding='utf-8-sig'))
    except (OSError, ValueError) as exc:
        raise StateError(f'unreadable state file {path}: {exc}') from exc
    if verify and 'state_digest' in payload:
        expected = payload['state_digest']
        actual = digest_payload(payload)
        if expected != actual:
            raise StateError(
                f'state file {path} failed its own digest check '
                f'({expected[:16]}... != {actual[:16]}...)')
    return payload


def write_json_no_digest(path: Path | str, payload: dict) -> Path:
    return atomic_write_json(path, payload, with_digest=False)


class RunLayout:
    """The on-disk layout of one Public Pilot run."""

    SUBDIRS = ('state', 'logs', 'runs', 'freeze', 'evaluation', 'robustness',
               'reports', 'cache')

    def __init__(self, root: Path | str):
        self.root = Path(root)

    def path(self, *parts: str) -> Path:
        return self.root.joinpath(*parts)

    def ensure(self) -> RunLayout:
        self.root.mkdir(parents=True, exist_ok=True)
        for name in self.SUBDIRS:
            (self.root / name).mkdir(parents=True, exist_ok=True)
        return self

    @property
    def state_path(self) -> Path:
        return self.root / 'state' / AUTOPILOT_STATE

    @property
    def environment_fingerprint_path(self) -> Path:
        return self.root / 'state' / 'ENVIRONMENT_FINGERPRINT.json'

    @property
    def current_status_path(self) -> Path:
        return self.root / 'CURRENT_STATUS.json'

    @property
    def freeze_path(self) -> Path:
        return self.root / 'freeze' / 'PUBLIC_EXPERIMENT_FREEZE.json'

    @property
    def unblind_audit_path(self) -> Path:
        return self.root / 'freeze' / 'HELDOUT_UNBLIND_AUDIT.json'

    @property
    def lock_path(self) -> Path:
        return self.root / 'state' / 'AUTOPILOT.lock'

    @property
    def cache_root(self) -> Path:
        return self.root / 'cache' / 'features'

    @property
    def runs_root(self) -> Path:
        return self.root / 'runs'


class RunLock:
    """Single-writer lock so two double-clicks cannot race on the same run root."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.acquired = False

    def acquire(self) -> RunLock:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            try:
                holder = json.loads(self.path.read_text(encoding='utf-8-sig'))
                pid = int(holder.get('pid', -1))
            except (OSError, ValueError):
                pid = -1
            if pid > 0 and _process_alive(pid):
                raise StateError(
                    f'another Public Pilot process (pid {pid}) already owns '
                    f'{self.path}; refusing to run two writers on one run root')
        payload = {'pid': os.getpid(), 'started_at': local_timestamp(),
                   'executable': _executable()}
        atomic_write_json(self.path, payload, with_digest=False)
        self.acquired = True
        return self

    def release(self) -> None:
        if self.acquired:
            self.path.unlink(missing_ok=True)
            self.acquired = False

    def __enter__(self) -> RunLock:
        return self.acquire()

    def __exit__(self, *exc_info) -> None:
        self.release()


def _executable() -> str:
    import sys
    return sys.executable


def _process_alive(pid: int) -> bool:
    """Windows-safe liveness probe without extra dependencies."""
    if pid == os.getpid():
        return True
    try:
        import ctypes
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.OpenProcess.restype = ctypes.c_void_p
        handle = kernel.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        kernel.CloseHandle(ctypes.c_void_p(handle))
        return True
    except Exception:
        # Never block a legitimate resume because the probe itself failed.
        return False


class StageRecord:
    """One resumable unit of work plus the digests that authorise re-use."""

    def __init__(self, name: str, entry: dict | None = None):
        self.name = name
        self.entry = dict(entry or {})

    # --- construction ----------------------------------------------------
    @staticmethod
    def pending(name: str) -> dict:
        return {'name': name, 'status': PENDING, 'attempts': 0,
                'updated_at': utc_timestamp(), 'evidence': {}}

    def as_dict(self) -> dict:
        return dict(self.entry)


def stage_is_complete(stage: dict | None, required: dict) -> tuple[bool, str]:
    """Whether a recorded stage may be re-used given the current required digests.

    Returns ``(ok, reason)``. This is the single place that decides resume legality:
    every key in ``required`` must be present and equal in the stage's ``evidence``.
    A stage that passes an empty ``required`` is not meaningful, so callers always pass
    at least the dataset and config digests.
    """
    if not stage:
        return False, 'no recorded stage'
    if stage.get('status') != COMPLETE:
        return False, f"status is {stage.get('status')!r}, not COMPLETE"
    evidence = stage.get('evidence') or {}
    for key, value in required.items():
        if key not in evidence:
            return False, f'evidence is missing {key}'
        if evidence[key] != value:
            return False, (f'evidence {key} mismatch: recorded '
                           f'{str(evidence[key])[:24]} != current {str(value)[:24]}')
    return True, 'digest-bound evidence matches'


class AutopilotState:
    """The whole persisted picture of one Public Pilot run."""

    def __init__(self, path: Path | str, payload: dict | None = None):
        self.path = Path(path)
        self.payload = payload if payload is not None else self._blank()

    @staticmethod
    def _blank() -> dict:
        return {
            'state_version': STATE_VERSION,
            'run_id': None,
            'orchestrator_version': None,
            'created_at': None,
            'updated_at': None,
            'accepted_gate_d_binding': {},
            'dataset': {},
            'environment': {},
            'calibration': {},
            'stages': {},
            'models': {},
            'freeze': {'created': False, 'sha256': None, 'path': None},
            'heldout': {'unlocked': False, 'accessed': False,
                        'first_access_timestamp': None,
                        'freeze_sha256_at_unblind': None,
                        'adapter_parity': None, 'manifest_check': None},
            'robustness': {},
            'final': {'verdict': None, 'report_json': None, 'report_md': None},
            'stop': {'code': None, 'detail': None, 'timestamp': None},
            'resume': {'count': 0, 'events': []},
        }

    # --- lifecycle -------------------------------------------------------
    @classmethod
    def load(cls, path: Path | str) -> AutopilotState:
        payload = read_json(path)
        if payload is None:
            return cls(path)
        if payload.get('state_version') != STATE_VERSION:
            raise StateError(
                f"state version {payload.get('state_version')!r} is not {STATE_VERSION!r}; "
                'refusing to resume an incompatible run')
        state = cls(path, payload)
        for key, default in cls._blank().items():
            state.payload.setdefault(key, default)
        state.payload['resume']['count'] = int(state.payload['resume'].get('count', 0)) + 1
        state.payload['resume']['events'] = list(
            state.payload['resume'].get('events', []))[-49:]
        state.payload['resume']['events'].append(
            {'at': utc_timestamp(), 'pid': os.getpid()})
        return state

    def ensure_run_identity(self, run_id: str, orchestrator_version: str) -> bool:
        """Set the run identity once; returns True when this is a brand-new run."""
        fresh = self.payload.get('run_id') is None
        if fresh:
            self.payload['run_id'] = run_id
            self.payload['orchestrator_version'] = orchestrator_version
            self.payload['created_at'] = local_timestamp()
        else:
            if self.payload.get('orchestrator_version') != orchestrator_version:
                raise StateError(
                    'orchestrator version changed between runs '
                    f"({self.payload.get('orchestrator_version')!r} -> "
                    f'{orchestrator_version!r}); resume refused because the plan that '
                    'produced the recorded stages cannot be reproduced')
        return fresh

    def save(self) -> str:
        self.payload['updated_at'] = local_timestamp()
        return atomic_write_json(self.path, self.payload)

    # --- accessors -------------------------------------------------------
    def stage(self, name: str) -> dict:
        return self.payload['stages'].get(name) or StageRecord.pending(name)

    def set_stage(self, name: str, status: str, evidence: dict | None = None,
                  **extra) -> dict:
        if status not in STAGE_STATES:
            raise StateError(f'unknown stage status {status!r}')
        previous = self.payload['stages'].get(name) or StageRecord.pending(name)
        entry = dict(previous)
        entry.update(name=name, status=status, updated_at=utc_timestamp(), **extra)
        if evidence is not None:
            entry['evidence'] = dict(evidence)
        entry['attempts'] = int(previous.get('attempts', 0)) + (1 if status == RUNNING
                                                              else 0)
        self.payload['stages'][name] = entry
        return entry

    def record_stop(self, code: str, detail: dict) -> dict:
        self.payload['stop'] = {'code': code, 'detail': detail,
                                'timestamp': local_timestamp()}
        return self.payload['stop']

    def clear_stop(self) -> None:
        self.payload['stop'] = {'code': None, 'detail': None, 'timestamp': None}

    def model_run(self, key: str) -> dict:
        return self.payload['models'].get(key) or {}

    def set_model_run(self, key: str, record: dict) -> dict:
        merged = dict(self.payload['models'].get(key) or {})
        merged.update(record)
        merged['updated_at'] = utc_timestamp()
        self.payload['models'][key] = merged
        return merged

    def mark_heldout_accessed(self, freeze_sha256: str) -> dict:
        """Irreversible unblinding audit. The flag can only ever go to True."""
        audit = self.payload['heldout']
        if audit.get('accessed') is True:
            return audit
        audit['accessed'] = True
        audit['unlocked'] = True
        audit['first_access_timestamp'] = local_timestamp()
        audit['freeze_sha256_at_unblind'] = freeze_sha256
        return audit
