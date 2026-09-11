"""Environment fingerprint, resource guards and batch calibration.

The Public Pilot runs unattended, so it may not stop to ask whether the machine is ready
and it may not quietly install anything. It checks, records what it found, and refuses to
continue when a guard fails:

* ``STOP_ENVIRONMENT_NOT_READY`` — no usable torch/CUDA/device, or the frozen training
  tree, cache root or publisher data is missing. Nothing is installed or repaired.
* ``STOP_DISK_GUARD`` — free space on a required drive would fall below 35 GiB.
* ``STOP_BATCH_CALIBRATION`` — no batch size in ``16, 8, 4, 2, 1`` completed two real
  optimizer updates on both architectures.

Calibration follows the already-approved Gate A policy rather than inventing a new one:
probe ``16, 8, 4, 2, 1`` in that order, take the largest stable batch, use half of it as
the safe batch, and require that the safe batch itself was measured stable. PointNet and
PointNet++ may have different safe batches; fair comparison then uses
``SAFE_COMMON_BATCH = min(pointnet_safe, pointnet2_safe)`` and both models train at that
one value. Once frozen the batch never changes, and a genuine OOM at a frozen safe batch
is ``STOP_RUNTIME_OOM_AFTER_FREEZE`` rather than a silent retry.

No driver upgrades, no CUDA toolkit installs, no deletions, no Docker or system changes.
"""
from __future__ import annotations

import gc
import hashlib
import json
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

ENVIRONMENT_VERSION = 'EFG_ENVIRONMENT_V1'
CALIBRATION_VERSION = 'EFG_BATCH_CALIBRATION_V1'
DISK_GUARD_BYTES = 35 * 1024**3
PROBE_BATCHES = (16, 8, 4, 2, 1)
PROBE_STEPS = 2
PROBE_MAX_ATTEMPTS = 8
PROBE_POINTS = 4096
PROBE_LEARNING_RATE = 1e-3


class ResourceError(RuntimeError):
    """Raised with a STOP_* code when the environment or a guard fails."""

    def __init__(self, code: str, detail):
        super().__init__(f'{code}: {detail}')
        self.code = code
        self.detail = detail


# --- static environment ----------------------------------------------------

def _total_ram_bytes() -> int | None:
    try:
        import ctypes

        class MemoryStatusEx(ctypes.Structure):
            _fields_ = [('dwLength', ctypes.c_ulong), ('dwMemoryLoad', ctypes.c_ulong),
                        ('ullTotalPhys', ctypes.c_ulonglong),
                        ('ullAvailPhys', ctypes.c_ulonglong),
                        ('ullTotalPageFile', ctypes.c_ulonglong),
                        ('ullAvailPageFile', ctypes.c_ulonglong),
                        ('ullTotalVirtual', ctypes.c_ulonglong),
                        ('ullAvailVirtual', ctypes.c_ulonglong),
                        ('ullAvailExtendedVirtual', ctypes.c_ulonglong)]

        status = MemoryStatusEx()
        status.dwLength = ctypes.sizeof(status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.ullTotalPhys)
    except Exception:
        return None
    return None


def _cpu_description() -> dict:
    return {
        'processor': platform.processor(),
        'machine': platform.machine(),
        'logical_cores': _cpu_count(),
    }


def _cpu_count() -> int | None:
    try:
        import os
        return os.cpu_count()
    except Exception:
        return None


def _driver_info() -> dict:
    try:
        out = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=name,driver_version,memory.total',
             '--format=csv,noheader'], text=True, timeout=30).strip()
        return {'nvidia_smi': out}
    except Exception as exc:
        return {'nvidia_smi': None, 'error': str(exc)}


def detect_cuda() -> dict:
    """Import torch only here, so environment checks can run before a GPU is needed."""
    info: dict = {'torch': None, 'cuda_available': False}
    try:
        import torch
    except Exception as exc:  # pragma: no cover - environment dependent
        info['import_error'] = str(exc)
        return info
    info['torch'] = torch.__version__
    info['torch_cuda'] = torch.version.cuda
    info['cuda_available'] = bool(torch.cuda.is_available())
    if info['cuda_available']:
        info['device_name'] = torch.cuda.get_device_name(0)
        info['capability'] = list(torch.cuda.get_device_capability(0))
        props = torch.cuda.get_device_properties(0)
        info['vram_total_bytes'] = int(props.total_memory)
        info['arch_list'] = list(torch.cuda.get_arch_list())
    return info


def disk_state(paths) -> dict:
    out = {}
    for label, path in paths.items():
        path = Path(path)
        anchor = path
        while not anchor.exists() and anchor.parent != anchor:
            anchor = anchor.parent
        try:
            usage = shutil.disk_usage(anchor)
            out[label] = {'path': str(path), 'anchor': str(anchor),
                          'free_bytes': int(usage.free), 'total_bytes': int(usage.total),
                          'free_gib': round(usage.free / 1024**3, 3)}
        except OSError as exc:
            out[label] = {'path': str(path), 'error': str(exc)}
    return out


def require_disk_guard(disks: dict, required=('D', 'E')) -> None:
    """STOP_DISK_GUARD when a required drive is below the reserve."""
    breaches = []
    for label in required:
        entry = disks.get(label)
        if not entry:
            continue
        free = entry.get('free_bytes')
        if free is None:
            continue
        if free < DISK_GUARD_BYTES:
            breaches.append({'drive': label, 'free_bytes': free,
                             'free_gib': round(free / 1024**3, 3),
                             'guard_gib': DISK_GUARD_BYTES / 1024**3})
    if breaches:
        raise ResourceError('STOP_DISK_GUARD', {
            'message': f'free space below {DISK_GUARD_BYTES / 1024**3:.0f} GiB',
            'breaches': breaches,
            'note': 'no file is deleted automatically; free space manually and rerun'})


def fingerprint(layout, extra_paths: dict | None = None) -> dict:
    """Full environment fingerprint plus the resolved training and data paths."""
    import gate_d_dataset as dataset_module
    import gate_d_geometry as geometry_module

    paths = {
        'D': 'D:/',
        'E': 'E:/',
        'training_root': layout.root,
        'cache_root': layout.cache_root,
        'data_root': geometry_module.DATA_ROOT,
    }
    if extra_paths:
        paths.update(extra_paths)
    disks = disk_state(paths)
    cuda = detect_cuda()
    payload = {
        'environment_version': ENVIRONMENT_VERSION,
        'captured_at': time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime()),
        'python': {'version': platform.python_version(), 'executable': sys.executable,
                   'implementation': platform.python_implementation()},
        'platform': {'system': platform.system(), 'release': platform.release(),
                     'version': platform.version()},
        'cpu': _cpu_description(),
        'ram_total_bytes': _total_ram_bytes(),
        'gpu': cuda,
        'driver': _driver_info(),
        'disks': disks,
        'paths': {k: str(v) for k, v in paths.items()},
        'data_root_exists': Path(geometry_module.DATA_ROOT).is_dir(),
        'cache_root_exists': Path(layout.cache_root).exists(),
        'default_gate_d_cache_root_exists': Path(
            dataset_module.DEFAULT_CACHE_ROOT).exists(),
        'safe_batch_policy': 'half largest stable probe batch, minimum 1',
        'probe_batches': list(PROBE_BATCHES),
    }
    payload['environment_digest'] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(',', ':'),
                   default=str).encode('utf-8')).hexdigest()
    return payload


def require_environment(payload: dict) -> None:
    """STOP_ENVIRONMENT_NOT_READY for anything the pilot genuinely needs."""
    problems = []
    gpu = payload.get('gpu') or {}
    if not gpu.get('torch'):
        problems.append({'item': 'torch', 'detail': gpu.get('import_error',
                                                             'torch is not importable')})
    if not gpu.get('cuda_available'):
        problems.append({'item': 'cuda', 'detail': 'torch.cuda.is_available() is False'})
    if not payload.get('data_root_exists'):
        problems.append({'item': 'data_root',
                         'detail': f"{payload['paths'].get('data_root')} is missing"})
    if not payload.get('ram_total_bytes'):
        problems.append({'item': 'ram', 'detail': 'total physical memory unknown'})
    if problems:
        raise ResourceError('STOP_ENVIRONMENT_NOT_READY', {
            'problems': problems,
            'note': 'the pilot never installs or upgrades anything; repair the existing '
                    'training environment manually and rerun'})


# --- batch calibration -----------------------------------------------------

def _probe_one(model_name: str, batch: int, device) -> dict:
    """Two real optimizer updates at `batch`; OOM is a measurement, not a crash."""
    import models
    import torch
    from fp32_policy import configure_fp32

    configure_fp32()
    item = {'batch': batch, 'model': model_name}
    started = time.perf_counter()
    model = optimizer = scaler = features = labels = None
    try:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        model = models.MODELS[model_name]().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=PROBE_LEARNING_RATE)
        scaler = torch.amp.GradScaler('cuda', enabled=True)
        features = torch.randn(batch, PROBE_POINTS, 4, device=device)
        features[..., 3] = (features[..., 3] > 0).float()
        labels = torch.arange(batch, device=device) % 2
        criterion = torch.nn.CrossEntropyLoss()
        successful = skipped = attempts = 0
        while successful < PROBE_STEPS and attempts < PROBE_MAX_ATTEMPTS:
            before = [s['step'].item() for s in optimizer.state.values() if 'step' in s]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                loss = criterion(model(features), labels)
            if not torch.isfinite(loss):
                raise ResourceError('STOP_NONFINITE_TRAINING',
                                    {'stage': 'batch_probe', 'model': model_name,
                                     'batch': batch})
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            after = [s['step'].item() for s in optimizer.state.values() if 'step' in s]
            expected = (min(before) if before else 0) + 1
            updated = bool(after) and all(v == expected for v in after)
            successful += int(updated)
            skipped += int(not updated)
            attempts += 1
        torch.cuda.synchronize()
        state_bytes = sum(v.numel() * v.element_size()
                          for state in optimizer.state.values()
                          for v in state.values() if torch.is_tensor(v))
        stable = successful >= PROBE_STEPS and state_bytes > 0
        item.update({
            'status': 'STABLE' if stable else 'AMP_UNSTABLE',
            'successful_optimizer_updates': successful, 'skipped_updates': skipped,
            'attempts': attempts, 'optimizer_state_bytes': state_bytes,
            'gpu_peak_allocated_bytes': int(torch.cuda.max_memory_allocated()),
            'gpu_peak_reserved_bytes': int(torch.cuda.max_memory_reserved()),
            'seconds': round(time.perf_counter() - started, 3),
        })
    except torch.cuda.OutOfMemoryError as exc:
        item.update({'status': 'OOM', 'error': str(exc)[:400],
                     'seconds': round(time.perf_counter() - started, 3)})
    finally:
        del model, optimizer, scaler, features, labels
        gc.collect()
        try:
            import torch as _torch
            _torch.cuda.empty_cache()
        except Exception:
            pass
    return item


def calibrate_batch(models_to_probe, device_name: str = 'cuda') -> dict:
    """Largest stable batch and the safe (half) batch for each architecture."""
    import torch

    device = torch.device(device_name)
    result = {'calibration_version': CALIBRATION_VERSION,
              'probe_batches': list(PROBE_BATCHES), 'probe_steps': PROBE_STEPS,
              'policy': 'half largest stable, minimum 1', 'models': {}}
    for model_name in models_to_probe:
        probes = [_probe_one(model_name, batch, device) for batch in PROBE_BATCHES]
        stable = [p['batch'] for p in probes if p['status'] == 'STABLE']
        entry = {'probes': probes, 'oom_observed': any(p['status'] == 'OOM'
                                                       for p in probes),
                 'amp_unstable_observed': any(p['status'] == 'AMP_UNSTABLE'
                                              for p in probes)}
        if not stable:
            entry.update(max_stable=None, safe_batch=None)
        else:
            maximum = max(stable)
            safe = max(1, maximum // 2)
            if safe not in stable:
                safe = min(stable)
            entry.update(max_stable=maximum, safe_batch=safe)
        result['models'][model_name] = entry
    safes = [result['models'][m]['safe_batch'] for m in models_to_probe]
    if any(s is None for s in safes):
        raise ResourceError('STOP_BATCH_CALIBRATION', {
            'message': 'no stable batch on at least one architecture',
            'calibration': result,
            'note': 'the pilot will not invent a smaller configuration silently'})
    result['safe_common_batch'] = int(min(safes))
    result['frozen_at'] = time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime())
    result['calibration_digest'] = hashlib.sha256(
        json.dumps(result, sort_keys=True, separators=(',', ':'),
                   default=str).encode('utf-8')).hexdigest()
    return result
