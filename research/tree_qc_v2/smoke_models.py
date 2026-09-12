"""Shape, gradient and cost smoke test for the three refiner candidates."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import v2_config as cfg
import v2_models as models
import v2_refiner_data as refdata


def main() -> int:
    torch.manual_seed(cfg.SEED)
    channels = len(refdata.FEATURE_CHANNELS)
    points, batch_size = 512, 2
    points_tensor = torch.randn(batch_size, points, channels)
    # The network's radii are in normalised units, so the smoke input must be too.
    points_tensor[..., :3] = points_tensor[..., :3] * 0.5
    points_tensor[..., 3] = (torch.rand(batch_size, points) > 0.3).float()
    points_tensor[..., 4] = 0.7
    target = (torch.rand(batch_size, points) > 0.4).float()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    report = {'device': device, 'in_channels': channels,
              'smoke_points': points, 'smoke_batch': batch_size, 'models': {}}
    print(f'device {device}, channels {channels}')
    for key in ('A', 'B', 'C'):
        model = models.build_model(key, channels).to(device)
        parameter_count = models.parameter_count(model)
        optimiser = torch.optim.Adam(model.parameters(), lr=cfg.REFINER_LR)
        started = time.time()
        if device == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        outputs = model(points_tensor.to(device))
        assert outputs['logits'].shape == (batch_size, points), outputs['logits'].shape
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            outputs['logits'], target.to(device))
        if model.auxiliary:
            assert outputs['offset'].shape == (batch_size, points, 3)
            assert outputs['quality'].shape == (batch_size,)
            loss = (loss + outputs['offset'].abs().mean()
                    + outputs['quality'].mean())
        loss.backward()
        gradients = sum(1 for p in model.parameters()
                        if p.grad is not None and torch.isfinite(p.grad).all())
        total = sum(1 for _ in model.parameters())
        optimiser.step()
        elapsed = time.time() - started
        peak = (torch.cuda.max_memory_allocated() / 1024 ** 2) if device == 'cuda' else None
        report['models'][model.name] = {
            'parameters': parameter_count,
            'finite_gradients': f'{gradients}/{total}',
            'forward_backward_seconds': round(elapsed, 3),
            'gpu_peak_memory_mb': round(peak, 1) if peak else None,
            'auxiliary_heads': bool(model.auxiliary),
        }
        print(f'  {model.name:34} params {parameter_count:>8,} '
              f'grads {gradients}/{total} '
              f'step {elapsed:.3f}s peak {peak if peak is None else round(peak, 1)} MB')
    report['describe'] = models.describe_models(channels)
    out = cfg.REPORT_DIR / 'MODEL_SMOKE.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f'wrote {out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
