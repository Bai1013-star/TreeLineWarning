"""The Conservative Delta Refiner: PointNeXt-S with two decision heads.

The backbone is imported from the frozen V2 code rather than reimplemented, so
"same backbone" is a fact about the code rather than a claim in a report. Only the
task changes:

* V2 had one head predicting a full membership mask.
* This model has two heads. ``REMOVE`` scores points that are **currently in the
  mask** and answers "should this point leave it". ``ADD`` scores points that are
  **currently outside the mask** and answers "should this point enter it".

The final mask is therefore always anchored on the current one:

    refined = current - confident_remove + confident_add

A head is a no-op by default. A point outside the current mask cannot be removed
by the REMOVE head, because the head is never applied there, and vice versa; the
identity "return the current mask" is available to the model at zero cost, which is
what makes the conservative behaviour learnable instead of merely hoped for.
"""
from __future__ import annotations

import sys

import torch
import v25_config as cfg
from torch import nn

# The frozen V2 implementation, imported rather than copied. If V2's backbone ever
# changed, this import would pick up the change and the config digest would move.
if str(cfg.V2_DIR) not in sys.path:
    sys.path.insert(0, str(cfg.V2_DIR))

import v2_models as v2_models  # noqa: E402

#: Decision threshold on both heads. Frozen; see v25_config.frozen_thresholds.
HEAD_THRESHOLD = 0.5


class DecisionHead(nn.Module):
    """One per-point binary decision, in the ``(B, N, C)`` layout V2 uses."""

    def __init__(self, in_channels: int, hidden: int = 96):
        super().__init__()
        self.net = v2_models.pointwise([in_channels, hidden, hidden // 2, 1])

    def forward(self, features):
        return v2_models.conv1d(features, self.net).squeeze(-1)


class DeltaRefiner(nn.Module):
    """PointNeXt-S backbone plus a REMOVE head and an ADD head."""

    name = 'POINTNEXT_S_DELTA_REFINER'

    def __init__(self, in_channels: int, width: int = cfg.MODEL_WIDTH):
        super().__init__()
        self.in_channels = in_channels
        self.backbone = v2_models.PointNeXtBackbone(in_channels, width)
        self.remove_head = DecisionHead(self.backbone.out_channels)
        self.add_head = DecisionHead(self.backbone.out_channels)

    def forward(self, points):
        features = self.backbone(points)
        return {'remove_logits': self.remove_head(features),
                'add_logits': self.add_head(features)}


def build_model(in_channels: int, width: int = cfg.MODEL_WIDTH) -> DeltaRefiner:
    return DeltaRefiner(in_channels, width)


def parameter_count(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def parameter_breakdown(in_channels: int, width: int = cfg.MODEL_WIDTH) -> dict:
    """Where the parameters sit, and how this compares with the V2 candidates."""
    model = build_model(in_channels, width)
    total = parameter_count(model)
    backbone = parameter_count(model.backbone)
    v2 = {'MODEL_B_POINTNEXT_S': parameter_count(
        v2_models.build_model('B', in_channels, width))}
    return {
        'model': model.name,
        'in_channels': in_channels,
        'width': width,
        'parameters_total': total,
        'parameters_backbone': backbone,
        'parameters_heads': total - backbone,
        'v2_same_backbone_parameter_count': v2,
        'backbone_is_the_frozen_v2_module': (
            'research/tree_qc_v2/v2_models.PointNeXtBackbone, imported not copied'),
        'head_threshold': HEAD_THRESHOLD,
    }


def apply_delta(current: torch.Tensor, remove_probability: torch.Tensor,
                add_probability: torch.Tensor,
                threshold: float = HEAD_THRESHOLD) -> dict:
    """Build the refined mask from the two decisions, on ``(B, N)`` tensors."""
    current_mask = current > 0.5
    confident_remove = (remove_probability > threshold) & current_mask
    confident_add = (add_probability > threshold) & ~current_mask
    refined = (current_mask & ~confident_remove) | confident_add
    return {'refined': refined, 'removed': confident_remove, 'added': confident_add,
            'unchanged': refined == current_mask}


def delta_metrics(refined: torch.Tensor, removed: torch.Tensor, added: torch.Tensor,
                  truth: torch.Tensor, current: torch.Tensor) -> dict:
    """Counts for one example, from tensors, before any decision is applied."""
    truth = truth > 0.5
    current = current > 0.5
    should_remove = current & ~truth
    should_add = ~current & truth
    return {
        'removed_points': int(removed.sum()),
        'added_points': int(added.sum()),
        'remove_true_positive': int((removed & should_remove).sum()),
        'remove_false_positive': int((removed & ~should_remove).sum()),
        'remove_false_negative': int((should_remove & ~removed).sum()),
        'add_true_positive': int((added & should_add).sum()),
        'add_false_positive': int((added & ~should_add).sum()),
        'add_false_negative': int((should_add & ~added).sum()),
        'changed_points': int((refined != current).sum()),
        'region_points': int(refined.numel()),
    }


def summarise_delta_counts(rows: list[dict]) -> dict:
    """Aggregate the ADD and REMOVE precision/recall/F1 the round requires."""
    def scores(prefix: str) -> dict:
        tp = sum(row[f'{prefix}_true_positive'] for row in rows)
        fp = sum(row[f'{prefix}_false_positive'] for row in rows)
        fn = sum(row[f'{prefix}_false_negative'] for row in rows)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        return {f'{prefix}_true_positive': tp, f'{prefix}_false_positive': fp,
                f'{prefix}_false_negative': fn,
                f'{prefix}_precision': precision, f'{prefix}_recall': recall,
                f'{prefix}_f1': f1}
    out = {**scores('remove'), **scores('add')}
    changed = sum(row['changed_points'] for row in rows)
    region = sum(row['region_points'] for row in rows)
    out['changed_points'] = changed
    out['region_points'] = region
    out['change_fraction'] = float(changed / region) if region else 0.0
    return out
