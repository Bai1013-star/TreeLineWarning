"""Gate A only. Synthetic diagnostic results are NOT tree-QC evidence."""
import argparse
import ctypes
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import torch
from fp32_policy import configure_fp32
FP32_POLICY = configure_fp32()
from torch.utils.data import Dataset, DataLoader
from models import MODELS, farthest_point_sample, radius_group

ROOT = Path(r'D:\TreeQCTraining')


CONFIG_PATH = Path(__file__).with_name('mini_gate_config.json')
CONFIG = json.loads(CONFIG_PATH.read_text(encoding='utf-8-sig'))
AMP_EVENTS = {'updates': 0, 'overflow_skips': 0}


def resources():
    from windows_resources import process_memory
    return {**process_memory(), 'gpu_peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'gpu_peak_reserved_bytes': torch.cuda.max_memory_reserved(), 'process_cpu_seconds': time.process_time(), 'd_free_bytes': shutil.disk_usage('D:\\').free}

class TinyDiagnostic(Dataset):
    """Balanced shape toy, deliberately easy. Not public corruption generator."""
    def __len__(self):
        return CONFIG['samples']

    def __getitem__(self, index):
        gen = torch.Generator().manual_seed(CONFIG['dataset_seed_base'] + index)
        xyz = torch.randn(CONFIG['points'], 3, generator=gen).clamp(-2.5, 2.5)
        label = index % 2
        xyz *= torch.tensor([0.35 if label == 0 else 2.0, 0.35 if label == 0 else 2.0, 2.0])
        xyz -= xyz.mean(0)
        membership = (torch.arange(CONFIG['points']) % 2 == 0).float()[:, None]
        return torch.cat([xyz, membership], -1), label


def log(event, **values):
    print(json.dumps({'event': event, 'time': time.time(), **values}), flush=True)


def step(model, optimizer, scaler, x, y):
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast('cuda', dtype=torch.float16):
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(logits, y)
    assert torch.isfinite(loss), 'nonfinite loss'
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    finite = all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    scaler.step(optimizer)
    scaler.update()
    AMP_EVENTS['updates' if finite else 'overflow_skips'] += 1
    if not finite:
        log('amp_overflow_skipped', scale=scaler.get_scale())
    return loss.item()


@torch.no_grad()
def evaluate(model, dataset, batch):
    model.eval()
    losses, correct, n = 0.0, 0, 0
    for x, y in DataLoader(dataset, batch_size=batch):
        x, y = x.cuda(), y.cuda()
        logits = model(x)
        losses += torch.nn.functional.cross_entropy(logits, y, reduction='sum').item()
        correct += (logits.argmax(1) == y).sum().item()
        n += y.numel()
    model.train()
    return {'loss': losses/n, 'accuracy': correct/n, 'count': n}


def probe(name):
    results = []
    for batch in CONFIG['probe_batches']:
        model = optimizer = scaler = x = y = None
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        try:
            model = MODELS[name]().cuda()
            optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
            scaler = torch.amp.GradScaler('cuda')
            x = torch.randn(batch, 4096, 4, device='cuda')
            x[..., 3] = (x[..., 3] > 0).float()
            y = torch.arange(batch, device='cuda') % 2
            successful_updates, skipped_updates, attempts = 0, 0, 0
            while successful_updates < CONFIG['probe_steps'] and attempts < CONFIG['probe_max_attempts']:
                previous_steps = [s['step'].item() for s in optimizer.state.values() if 'step' in s]
                step(model, optimizer, scaler, x, y)
                current_steps = [s['step'].item() for s in optimizer.state.values() if 'step' in s]
                expected = (min(previous_steps) if previous_steps else 0) + 1
                updated = bool(current_steps) and all(value == expected for value in current_steps)
                successful_updates += int(updated)
                skipped_updates += int(not updated)
                attempts += 1
            torch.cuda.synchronize()
            state_bytes = sum(v.numel()*v.element_size() for state in optimizer.state.values() for v in state.values() if torch.is_tensor(v))
            stable = successful_updates >= CONFIG['probe_steps'] and state_bytes > 0
            item = {'batch': batch, 'status': 'STABLE' if stable else 'AMP_UNSTABLE', 'successful_optimizer_updates': successful_updates, 'skipped_updates': skipped_updates, 'attempts': attempts, 'optimizer_state_bytes': state_bytes, 'seconds': time.perf_counter()-start, **resources()}
        except torch.cuda.OutOfMemoryError as exc:
            item = {'batch': batch, 'status': 'OOM', 'error': str(exc), 'seconds': time.perf_counter()-start}
        results.append(item)
        log('batch_probe', model=name, **item)
        if item['status'] == 'AMP_UNSTABLE':
            raise RuntimeError('AMP_UNSTABLE: batch probe lacked two actual optimizer updates')
        del model, optimizer, scaler, x, y
    stable = [r['batch'] for r in results if r['status'] == 'STABLE']
    assert stable, 'no stable batch'
    maximum = max(stable)
    safe = max(1, maximum // 2)
    assert safe in stable
    return {'probes': results, 'max_stable': maximum, 'safe_batch': safe, 'oom_observed': any(r['status'] == 'OOM' for r in results)}


def train(name, calibration, output):
    torch.manual_seed(CONFIG['seed'])
    torch.cuda.manual_seed_all(CONFIG['seed'])
    torch.cuda.reset_peak_memory_stats()
    model = MODELS[name]().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    scaler = torch.amp.GradScaler('cuda')
    dataset = TinyDiagnostic()
    batch = calibration['safe_batch']
    loader = DataLoader(dataset, batch_size=batch, shuffle=True, num_workers=CONFIG['workers'], multiprocessing_context='spawn', persistent_workers=True, pin_memory=True, generator=torch.Generator().manual_seed(CONFIG['seed']))
    initial = evaluate(model, dataset, batch)
    history = []
    started = time.perf_counter()
    for epoch in range(CONFIG['max_epochs']):
        epoch_start = time.perf_counter()
        for x, y in loader:
            assert x.shape[1:] == (4096, 4)
            step(model, optimizer, scaler, x.cuda(non_blocking=True), y.cuda(non_blocking=True))
        torch.cuda.synchronize()
        metrics = evaluate(model, dataset, batch)
        metrics.update(epoch=epoch + 1, seconds=time.perf_counter()-epoch_start)
        history.append(metrics)
        log('epoch', model=name, **metrics, **resources())
        if epoch >= 2 and metrics['accuracy'] >= CONFIG['overfit_accuracy_min'] and metrics['loss'] < initial['loss'] * 0.25:
            break
    final = history[-1]
    passed = final['accuracy'] >= CONFIG['overfit_accuracy_min'] and final['loss'] < initial['loss'] * CONFIG['overfit_loss_ratio_max']
    checkpoint = output / (name + '.pt')
    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scaler': scaler.state_dict(), 'epoch': len(history), 'seed': 101, 'model_name': name, 'diagnostic_only': True}, checkpoint)
    state = torch.load(checkpoint, map_location='cuda', weights_only=True)
    restored = MODELS[name]().cuda()
    restored.load_state_dict(state['model'])
    restored_optimizer = torch.optim.Adam(restored.parameters(), lr=CONFIG['learning_rate'])
    restored_optimizer.load_state_dict(state['optimizer'])
    restored_scaler = torch.amp.GradScaler('cuda')
    restored_scaler.load_state_dict(state['scaler'])
    x, y = next(iter(DataLoader(dataset, batch_size=batch)))
    x, y = x.cuda(), y.cuda()
    model.eval(); restored.eval()
    with torch.no_grad():
        difference = (model(x) - restored(x)).abs().max().item()
    before = [p.detach().clone() for p in restored.parameters()]
    steps_before = [s['step'].item() for s in restored_optimizer.state.values()]
    restored.train()
    resumed_loss = step(restored, restored_optimizer, restored_scaler, x, y)
    steps_after = [s['step'].item() for s in restored_optimizer.state.values()]
    changed = any(not torch.equal(old, new) for old, new in zip(before, restored.parameters()))
    resume_ok = difference == 0 and changed and all(b == a + 1 for a, b in zip(steps_before, steps_after))
    torch.cuda.synchronize()
    result = {'initial': initial, 'final': final, 'history': history, 'overfit_pass': passed, 'resume_pass': resume_ok, 'reload_max_abs_error': difference, 'resume_parameter_changed': changed, 'optimizer_steps_before': steps_before, 'optimizer_steps_after': steps_after, 'resumed_loss': resumed_loss, 'checkpoint': str(checkpoint), 'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(), 'parameters': sum(p.numel() for p in model.parameters()), 'model_size_mb_float32': sum(p.numel()*4 for p in model.parameters())/1e6, 'seconds': time.perf_counter()-started, 'amp': True, 'workers': CONFIG['workers'], 'windows_spawn_executed': True, 'batch': batch, **resources()}
    del loader
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT/'gate_a')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    result = {'status': 'FAIL', 'scope': 'implementation diagnostic only; no public data', 'public_download_allowed': False}
    start = time.perf_counter()
    config_path = Path(__file__).with_name('mini_gate_config.json')
    result['frozen_config_sha256'] = hashlib.sha256(config_path.read_bytes()).hexdigest()
    result['frozen_config'] = json.loads(config_path.read_text(encoding='utf-8-sig'))
    try:
        assert shutil.disk_usage('D:\\').free > 35*1024**3, 'STOP_DISK_GUARD'
        torch.set_num_threads(4)
        assert torch.cuda.is_available(), 'CUDA unavailable'
        result['environment'] = {'fp32_policy': FP32_POLICY, 'python_executable': sys.executable, 'python_version': sys.version, 'torch': torch.__version__, 'cuda': torch.version.cuda, 'device': torch.cuda.get_device_name(), 'capability': torch.cuda.get_device_capability(), 'arch_list': torch.cuda.get_arch_list(), 'driver': subprocess.check_output(['nvidia-smi', '--query-gpu=driver_version,memory.total', '--format=csv,noheader'], text=True).strip()}
        a = torch.randn(256, 256, device='cuda', requires_grad=True)
        (a @ a).square().mean().backward()
        assert torch.isfinite(a.grad).all()
        result['cuda_allocation_matmul_backward'] = True
        xyz = torch.tensor([[[0.,0,0],[1.,0,0],[3.,0,0],[10.,0,0]]], device='cuda')
        fps = farthest_point_sample(xyz, 4)
        assert fps.unique().numel() == 4
        idx = radius_group(xyz, xyz[:, :1], 1.1, 4)
        assert set(idx.flatten().tolist()) == {0, 1}
        result['point_ops_checks'] = {'fps_unique': True, 'radius_excludes_outside': True}
        log('environment', **result['environment'])
        result['models'] = {}
        for name in MODELS:
            calibration = probe(name)
            trained = train(name, calibration, args.output)
            result['models'][name] = {'calibration': calibration, **trained}
            log('model_complete', model=name, overfit_pass=trained['overfit_pass'], resume_pass=trained['resume_pass'])
            assert trained['overfit_pass'] and trained['resume_pass'], name + ' mini gate failed; STOP'
        result['status'] = 'WORKER_PASS_PENDING_SUPERVISOR'
    except Exception:
        result['error'] = traceback.format_exc()
        log('failure', error=result['error'])
    finally:
        result['amp_events'] = dict(AMP_EVENTS)
        result['source_sha256'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}
        result['wall_seconds'] = time.perf_counter()-start
        (args.output/'mini_gate.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        log('gate_complete', status=result['status'], output=str(args.output))
    return 0 if result['status'] == 'WORKER_PASS_PENDING_SUPERVISOR' else 1


if __name__ == '__main__':
    raise SystemExit(main())
