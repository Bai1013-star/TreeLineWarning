"""Can one model overfit one corrupted example? A pipeline sanity probe.

If a refiner cannot learn a single case it has seen two hundred times, the task
construction or the loss is wrong, and no architecture comparison on top of it
means anything. This is deliberately tiny and deliberately not a result.

This is a probe, not part of the audit's reported numbers.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

torch.set_num_threads(4)

sys.path.insert(0, str(Path(__file__).resolve().parent))

import v2_config as cfg  # noqa: E402
import v2_models as models  # noqa: E402
import v2_refiner_data as refdata  # noqa: E402
import v2_refiner_train as trainer  # noqa: E402
import v2_roi_library as library  # noqa: E402


def pick(index: dict, corruption: str, severity: str):
    for row in index['targets']:
        for proposal in row['rows']:
            if (proposal.get('corruption') == corruption
                    and proposal.get('severity') == severity
                    and not proposal.get('skipped')):
                return row, proposal
    return None, None


def main() -> int:
    key = sys.argv[1] if len(sys.argv) > 1 else 'A'
    corruption = sys.argv[2] if len(sys.argv) > 2 else 'MERGED_NEIGHBOR'
    severity = sys.argv[3] if len(sys.argv) > 3 else 'HIGH'
    steps = int(sys.argv[4]) if len(sys.argv) > 4 else 300
    points = 1024
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    index = library.load_library()
    row, proposal = pick(index, corruption, severity)
    if row is None:
        print(f'no {corruption}/{severity} row in the library')
        return 1
    example_key = f"{row['target_id']}|{proposal['proposal_id']}"
    print(f'{example_key} ({row["site"]}) library IoU {proposal["iou"]:.4f} '
          f'roi {row["roi_points"]} points', flush=True)

    example = refdata.Example(
        key=example_key, site=row['site'], target_id=row['target_id'],
        proposal_id=proposal['proposal_id'],
        proposal_kind=proposal['proposal_kind'], corruption=corruption,
        severity=severity, split='train', current_iou=float(proposal['iou']))

    def score(model) -> dict:
        rows = trainer.evaluate_full_roi(model, index, [row['target_id']],
                                         kinds=None, device=device)
        found = [r for r in rows if r['key'] == example_key]
        return found[0] if found else {}

    torch.manual_seed(cfg.SEED)
    model = models.build_model(key, len(refdata.FEATURE_CHANNELS)).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=cfg.REFINER_LR)
    dataset = refdata.RefinerDataset(index, [example], points, cfg.SEED, augment=True)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True,
                                         collate_fn=refdata.collate)

    before = score(model)
    print(f'before: refined IoU {before["refined_iou"]:.4f} '
          f'delta {before["delta_iou"]:+.4f} changed {before["changed_points"]}',
          flush=True)

    started = time.time()
    step = 0
    model.train()
    while step < steps:
        for batch in loader:
            batch['quality'] = torch.tensor([float(proposal['iou'])],
                                            dtype=torch.float32)
            if step == 0:
                features = batch['features']
                per_channel = [round(float(features[..., c].std()), 5)
                               for c in range(features.shape[-1])]
                print(f'  [debug] features {tuple(features.shape)} '
                      f'finite {bool(torch.isfinite(features).all())} '
                      f'std {per_channel}')
                print(f'  [debug] target sum {float(batch["target"].sum())} '
                      f'current sum {float(batch["current"].sum())} '
                      f'training {model.training}')
            optimiser.zero_grad()
            outputs = model(batch['features'].to(device))
            if step == 0:
                print(f'  [debug] logits abs mean '
                      f'{float(outputs["logits"].detach().abs().mean()):.6e} '
                      f'grad_fn {type(outputs["logits"].grad_fn).__name__}')
            loss, parts = trainer.total_loss(outputs, batch, model)
            loss.backward()
            grad_norm = sum(float(p.grad.norm()) for p in model.parameters()
                            if p.grad is not None)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimiser.step()
            step += 1
            if step in (1, 10, 50) or step % 100 == 0:
                print(f'  step {step:4d} loss {float(loss.detach()):.5f} '
                      f'grad {grad_norm:.3e} parts {parts}', flush=True)
            if step >= steps:
                break

    after = score(model)
    print(f'after {steps} steps ({time.time() - started:.1f}s): '
          f'refined IoU {after["refined_iou"]:.4f} delta {after["delta_iou"]:+.4f} '
          f'change_f1 {after["change_region_f1"]:.4f} '
          f'added_f1 {after["added_point_f1"]:.4f} '
          f'removed_f1 {after["removed_point_f1"]:.4f} '
          f'changed {after["changed_points"]}')
    out = cfg.REPORT_DIR / f'OVERFIT_PROBE_{key}_{corruption}_{severity}.json'
    cfg.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({'model': key, 'corruption': corruption,
                               'severity': severity, 'steps': steps,
                               'example': example_key,
                               'before': before, 'after': after}, indent=2),
                   encoding='utf-8')
    print(f'wrote {out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
