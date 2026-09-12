"""The three candidate refiners, implemented locally in plain PyTorch.

``MODEL_A``  PointNet++ (SSG) with a feature-propagation decoder.
``MODEL_B``  PointNeXt-S style backbone: inverted-residual blocks with residual
             skip connections, strided set abstraction for downsampling, and the
             same decoder as ``MODEL_A``.
``MODEL_C``  ``MODEL_B`` plus two auxiliary heads: a target-tree stem/base
             offset and a predicted mask quality.

Everything is a plain PyTorch implementation in this repository. Nothing is
imported from OpenPoints, PointNeXt, SoftGroup or any other external codebase, so
there is no licence question and no dependency beyond torch. This is recorded as
``LOCAL_IMPLEMENTATION``; the published architectures are
``REFERENCE_IMPLEMENTATION`` only, and no pretrained weights are used anywhere.

**Layout convention.** Every tensor in this module is ``(B, N, C)``: batch, point,
channel. Convolutions need ``(B, C, N)``, so the transposes are pushed into two
helpers, :func:`conv1d` and :func:`pointwise`. Mixing the two layouts is the
easiest way to write a network that runs and is wrong, so there is exactly one
place where the transpose happens and no other tensor is stored transposed.

The three models share one interface. ``forward(points)`` takes ``(B, N, C)`` with
channels 0..2 the normalised local coordinates, and returns a dict with ``logits``
of shape ``(B, N)``; ``MODEL_C`` adds ``offset`` of shape ``(B, N, 3)`` and
``quality`` of shape ``(B,)``.

The comparison is fair on the two things that are easy to get wrong:

* the **decoder is identical** in all three, so a difference cannot come from the
  upsampling path;
* **``MODEL_C`` is a strict superset of ``MODEL_B``**: same backbone, same head,
  plus two auxiliary heads. Any gain it shows is attributable to the auxiliary
  supervision, not to extra capacity in the main path.
"""
from __future__ import annotations

import torch
from torch import nn

# --- layout helpers -----------------------------------------------------------

def _CALLER_HINT() -> str:
    """Name the calling function, so a layout error points at its source."""
    import inspect
    for frame in inspect.stack()[2:6]:
        if frame.function not in ('conv1d', '_CALLER_HINT'):
            return f' (called from {frame.function} at line {frame.lineno})'
    return ''


def conv1d(features: torch.Tensor, layers: nn.Module) -> torch.Tensor:
    """Apply a point-wise ``Conv1d`` stack: ``(B, N, C)`` in and out.

    ``Conv1d`` needs ``(B, C, N)``, so the transpose happens here and nowhere
    else. The leading-dimension guard exists because a tensor that is still in
    ``(B, K, N, C)`` reaches this function as a four-dimensional tensor and
    ``Conv1d`` then reads the neighbourhood dimension ``K`` as the batch. That
    produced a shape error three layers away from the mistake, which is expensive
    to debug.
    """
    if features.dim() != 3:
        raise ValueError(
            f'conv1d expects (B, N, C), got shape {tuple(features.shape)}')
    expected = None
    for module in layers:
        if hasattr(module, 'in_channels'):
            expected = module.in_channels
            break
    if expected is not None and features.shape[-1] != expected:
        raise ValueError(
            f'conv1d got shape {tuple(features.shape)} but the first layer wants '
            f'{expected} channels; the tensor is probably transposed'
            f'{_CALLER_HINT()}')
    return layers(features.transpose(1, 2)).transpose(1, 2)


def pointwise(channels) -> nn.Sequential:
    """A point-wise MLP expressed in the ``(B, N, C)`` layout."""
    layers = []
    for first, second in zip(channels, channels[1:], strict=False):
        layers.append(nn.Conv1d(first, second, 1))
        layers.append(nn.BatchNorm1d(second))
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


def grouped_mlp(channels) -> nn.Sequential:
    """A shared MLP over a ``(B, K, N, C)`` neighbourhood tensor.

    Inside a set-abstraction layer the tensor carries a neighbourhood axis, so the
    point-wise convolutions have to be two-dimensional. Routing this tensor
    through :func:`conv1d` reads ``K`` as the batch and fails three layers later,
    which is why the two layouts have separate helpers.
    """
    layers = []
    for first, second in zip(channels, channels[1:], strict=False):
        layers.append(nn.Conv2d(first, second, 1))
        layers.append(nn.BatchNorm2d(second))
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


def index_points(points: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Gather along the point dimension of a ``(B, N, C)`` tensor."""
    batch = torch.arange(points.shape[0], device=points.device)
    batch = batch.reshape((-1,) + (1,) * (indices.ndim - 1))
    return points[batch, indices]


# --- neighbourhood search -----------------------------------------------------

@torch.no_grad()
def farthest_point_sample(xyz: torch.Tensor, count: int) -> torch.Tensor:
    """Deterministic FPS: the first centre is the point farthest from the centroid."""
    xyz = xyz.float()
    count = min(count, xyz.shape[1])
    indices = torch.empty(xyz.shape[0], count, dtype=torch.long, device=xyz.device)
    distance = torch.full(xyz.shape[:2], float('inf'), device=xyz.device)
    farthest = ((xyz - xyz.mean(1, keepdim=True)) ** 2).sum(-1).argmax(1)
    for i in range(count):
        indices[:, i] = farthest
        centroid = index_points(xyz, farthest[:, None])
        distance = torch.minimum(distance, ((xyz - centroid) ** 2).sum(-1))
        farthest = distance.argmax(1)
    return indices


@torch.no_grad()
def ball_query(xyz: torch.Tensor, centers: torch.Tensor, radius: float,
               count: int) -> torch.Tensor:
    """Nearest points inside a physical radius; empty slots fall back to nearest.

    The radii are expressed in the units of the coordinates they are given, which
    for this project are normalised local coordinates. A caller that passes raw
    metres gets a neighbourhood that is either empty or everything; the fallback
    and the final clamp keep that mistake from producing an out-of-range gather.
    """
    distance = torch.cdist(centers.float(), xyz.float())
    values, indices = distance.topk(min(count, xyz.shape[1]), largest=False)
    nearest = indices[..., :1].expand_as(indices)
    selected = torch.where(values <= radius, indices, nearest)
    limit = xyz.shape[1] - 1
    return selected.clamp_(0, limit)


# --- encoder layers -----------------------------------------------------------

class SetAbstraction(nn.Module):
    """PointNet++ single-scale-grouping layer."""

    def __init__(self, count, radius, neighbors, channels, group_all: bool = False):
        super().__init__()
        self.count, self.radius, self.neighbors = count, radius, neighbors
        self.group_all = group_all
        self.mlp = grouped_mlp(channels)

    def forward(self, xyz, features):
        if self.group_all:
            # One centre, every point in its neighbourhood: (B, 1, N, C).
            new_xyz = xyz.mean(1, keepdim=True)
            relative = (xyz - new_xyz).unsqueeze(1)
            grouped = torch.cat([relative, features.unsqueeze(1)], dim=-1)
            # (B, 1, N, C) -> (B, C, 1, N) -> max over N -> (B, C, 1) -> (B, 1, C).
            output = self.mlp(grouped.permute(0, 3, 1, 2)).amax(dim=-1).transpose(1, 2)
            return new_xyz, output
        centers = index_points(xyz, farthest_point_sample(xyz, self.count))
        indices = ball_query(xyz, centers, self.radius, self.neighbors)
        grouped_xyz = index_points(xyz, indices)
        grouped_features = index_points(features, indices)
        relative = grouped_xyz - centers.unsqueeze(2)
        grouped = torch.cat([relative, grouped_features], dim=-1)
        output = self.mlp(grouped.permute(0, 3, 1, 2)).amax(dim=-1).transpose(1, 2)
        return centers, output


class InvResMLP(nn.Module):
    """PointNeXt inverted-residual block.

    Expand, aggregate over the local neighbourhood, mix with a grouped
    convolution, project back, add the residual. This is the structural idea of
    PointNeXt; the implementation here is local and far smaller than the published
    network. The residual is the only thing MODEL_B changes relative to MODEL_A's
    encoder.
    """

    def __init__(self, radius: float, neighbors: int, channels: int,
                 expansion: int = 4, groups: int = 4):
        super().__init__()
        self.radius, self.neighbors = radius, neighbors
        hidden = channels * expansion
        self.expand = grouped_mlp([channels + 3, hidden])
        self.mix = nn.Sequential(
            nn.Conv1d(hidden, hidden, 1, groups=groups),
            nn.BatchNorm1d(hidden), nn.ReLU(inplace=True))
        self.project = nn.Sequential(
            nn.Conv1d(hidden, channels, 1), nn.BatchNorm1d(channels))
        self.act = nn.ReLU(inplace=True)

    def forward(self, xyz, features):
        indices = ball_query(xyz, xyz, self.radius, self.neighbors)
        grouped = index_points(features, indices)
        relative = index_points(xyz, indices) - xyz.unsqueeze(2)
        # (B, N, K, C+3) -> (B, C+3, N, K), MLP over channels, max over neighbours.
        expanded = self.expand(
            torch.cat([relative, grouped], dim=-1).permute(0, 3, 1, 2))
        pooled = expanded.amax(dim=-1)
        # conv1d reads (B, N, C): (B, hidden, N) -> (B, N, hidden).
        mixed = conv1d(pooled.transpose(1, 2), self.mix)
        out = conv1d(mixed, self.project)
        return self.act(features + out)


class FeaturePropagation(nn.Module):
    """PointNet++ decoder layer: interpolate from the coarse level, concat, MLP.

    ``coarse`` and ``fine`` are both ``(B, C, N)`` here; the point-wise MLP wants
    ``(B, N, C)``, so the transpose is explicit and the concatenation is always on
    the channel axis.
    """

    def __init__(self, channels):
        super().__init__()
        self.mlp = pointwise(channels)

    def forward(self, xyz_coarse, xyz_fine, coarse, fine):
        if xyz_coarse is None or xyz_coarse.shape[1] == 1:
            # coarse is (B, C, 1); broadcast its single centre to every fine point.
            repeated = coarse.expand(-1, -1, xyz_fine.shape[1])
        else:
            distance = torch.cdist(xyz_fine.float(), xyz_coarse.float())
            nearest = distance.topk(min(3, xyz_coarse.shape[1]), largest=False).indices
            weight = 1.0 / (distance.gather(-1, nearest) + 1e-8)
            weight = weight / weight.sum(-1, keepdim=True)
            # (B, C, M) -> (B, M, C) so the gather and the weights share an order.
            gathered = index_points(coarse.transpose(1, 2), nearest)
            repeated = (gathered * weight.unsqueeze(-1)).sum(2).transpose(1, 2)
        fused = torch.cat([repeated, fine], dim=1) if fine is not None else repeated
        # fused is (B, C, N) and conv1d takes and returns (B, N, C), which is what
        # the next decoder level expects for both its operands.
        return conv1d(fused.transpose(1, 2), self.mlp)


# --- heads --------------------------------------------------------------------

class MembershipHead(nn.Module):
    def __init__(self, in_channels: int, hidden: int = 128):
        super().__init__()
        self.net = pointwise([in_channels, hidden, hidden // 2, 1])

    def forward(self, features):
        return conv1d(features, self.net).squeeze(-1)


class OffsetHead(nn.Module):
    """Per-point offset toward the target tree's stem/base reference."""

    def __init__(self, in_channels: int, hidden: int = 96):
        super().__init__()
        self.net = pointwise([in_channels, hidden, 3])

    def forward(self, features):
        return conv1d(features, self.net)


class QualityHead(nn.Module):
    """Predicted IoU of the refined mask, from the pooled latent plus the logits."""

    def __init__(self, in_channels: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_channels * 2 + 2, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden // 2), nn.ReLU(inplace=True),
            nn.Linear(hidden // 2, 1))

    def forward(self, features, logits):
        pooled = torch.cat([features.amax(dim=1), features.mean(dim=1)], dim=1)
        probability = torch.sigmoid(logits)
        summary = torch.stack([probability.mean(dim=1), probability.amax(dim=1)], dim=1)
        return torch.sigmoid(self.net(torch.cat([pooled, summary], dim=1))).squeeze(1)


# --- backbones ----------------------------------------------------------------

class PointNetPlusPlusBackbone(nn.Module):
    """PointNet++ SSG encoder and the shared feature-propagation decoder."""

    def __init__(self, in_channels: int = 5, width: int = 32):
        super().__init__()
        # The final decoder layer concatenates a skip from the input level, so the
        # raw channels are embedded once here and every level then carries `width`.
        # Feeding the raw (x, y, z, membership, scale) tensor straight into the
        # decoder instead makes the channel count of the last layer depend on how
        # many input channels happen to exist, which is not a property the
        # architecture should have.
        self.embed = pointwise([in_channels, width, width])
        self.sa1 = SetAbstraction(256, 0.4, 32, [width + 3, width, width * 2])
        self.sa2 = SetAbstraction(64, 0.8, 32, [width * 2 + 3, width * 2, width * 4])
        self.sa3 = SetAbstraction(16, 1.6, 32, [width * 4 + 3, width * 4, width * 8])
        self.sa4 = SetAbstraction(None, None, None,
                                  [width * 8 + 3, width * 8, width * 8, width * 8],
                                  group_all=True)
        # The decoder widths are fixed by the skip connections, not chosen:
        # fp4 gets the global latent (8w) plus the l3 skip (8w); fp3 gets 8w plus
        # the l2 skip (4w); fp2 gets 4w plus the l1 skip (2w); fp1 gets 2w plus w.
        self.fp4 = FeaturePropagation([width * 16, width * 8])
        self.fp3 = FeaturePropagation([width * 12, width * 4])
        self.fp2 = FeaturePropagation([width * 6, width * 2])
        self.fp1 = FeaturePropagation([width * 3, width * 2])
        self.out_channels = width * 2

    def forward(self, points):
        l0_xyz = points[..., :3]
        l0 = conv1d(points, self.embed)
        l1_xyz, l1 = self.sa1(l0_xyz, l0)
        l2_xyz, l2 = self.sa2(l1_xyz, l1)
        l3_xyz, l3 = self.sa3(l2_xyz, l2)
        l4_xyz, l4 = self.sa4(l3_xyz, l3)
        # Decoder operands are (B, C, N) and the encoder emits (B, N, C).
        l3 = self.fp4(l4_xyz, l3_xyz, l4.transpose(1, 2), l3.transpose(1, 2))
        l2 = self.fp3(l3_xyz, l2_xyz, l3.transpose(1, 2), l2.transpose(1, 2))
        l1 = self.fp2(l2_xyz, l1_xyz, l2.transpose(1, 2), l1.transpose(1, 2))
        return self.fp1(l1_xyz, l0_xyz, l1.transpose(1, 2), l0.transpose(1, 2))


class PointNeXtBackbone(PointNetPlusPlusBackbone):
    """PointNeXt-S style: the same skeleton, inverted-residual blocks at each level."""

    def __init__(self, in_channels: int = 5, width: int = 32):
        super().__init__(in_channels, width)
        self.ir1 = InvResMLP(0.4, 16, width * 2)
        self.ir2 = InvResMLP(0.8, 16, width * 4)
        self.ir3 = InvResMLP(1.6, 16, width * 8)

    def forward(self, points):
        l0_xyz = points[..., :3]
        l0 = conv1d(points, self.embed)
        l1_xyz, l1 = self.sa1(l0_xyz, l0)
        l1 = self.ir1(l1_xyz, l1)
        l2_xyz, l2 = self.sa2(l1_xyz, l1)
        l2 = self.ir2(l2_xyz, l2)
        l3_xyz, l3 = self.sa3(l2_xyz, l2)
        l3 = self.ir3(l3_xyz, l3)
        l4_xyz, l4 = self.sa4(l3_xyz, l3)
        # Decoder operands are (B, C, N) and the encoder emits (B, N, C).
        l3 = self.fp4(l4_xyz, l3_xyz, l4.transpose(1, 2), l3.transpose(1, 2))
        l2 = self.fp3(l3_xyz, l2_xyz, l3.transpose(1, 2), l2.transpose(1, 2))
        l1 = self.fp2(l2_xyz, l1_xyz, l2.transpose(1, 2), l1.transpose(1, 2))
        return self.fp1(l1_xyz, l0_xyz, l1.transpose(1, 2), l0.transpose(1, 2))


# --- the three candidates -----------------------------------------------------

class RefinerBase(nn.Module):
    name = 'BASE'
    auxiliary = False

    def __init__(self, in_channels: int, width: int = 32):
        super().__init__()
        self.in_channels = in_channels
        self.backbone = self.build_backbone(in_channels, width)
        self.membership = MembershipHead(self.backbone.out_channels)

    def build_backbone(self, in_channels: int, width: int) -> nn.Module:
        raise NotImplementedError

    def forward(self, points):
        features = self.backbone(points)
        return {'logits': self.membership(features)}


class ModelA(RefinerBase):
    """PointNet++ refiner."""

    name = 'MODEL_A_POINTNET2'

    def build_backbone(self, in_channels: int, width: int) -> nn.Module:
        return PointNetPlusPlusBackbone(in_channels, width)


class ModelB(RefinerBase):
    """PointNeXt-S refiner."""

    name = 'MODEL_B_POINTNEXT_S'

    def build_backbone(self, in_channels: int, width: int) -> nn.Module:
        return PointNeXtBackbone(in_channels, width)


class ModelC(ModelB):
    """PointNeXt-S refiner with an offset head and a mask-quality head."""

    name = 'MODEL_C_POINTNEXT_S_MULTITASK'
    auxiliary = True

    def __init__(self, in_channels: int, width: int = 32):
        super().__init__(in_channels, width)
        self.offset = OffsetHead(self.backbone.out_channels)
        self.quality = QualityHead(self.backbone.out_channels)

    def forward(self, points):
        features = self.backbone(points)
        logits = self.membership(features)
        return {'logits': logits,
                'offset': self.offset(features),
                'quality': self.quality(features, logits)}


MODELS = {'A': ModelA, 'B': ModelB, 'C': ModelC}


def parameter_count(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def build_model(key: str, in_channels: int, width: int = 32) -> nn.Module:
    if key not in MODELS:
        raise ValueError(f'unknown model key {key!r}')
    return MODELS[key](in_channels, width)


def describe_models(in_channels: int, width: int = 32) -> dict:
    """Parameter counts, computed rather than asserted."""
    out = {}
    for key in sorted(MODELS):
        model = build_model(key, in_channels, width)
        out[model.name] = {
            'key': key, 'parameters': parameter_count(model),
            'auxiliary_heads': bool(model.auxiliary),
            'in_channels': in_channels,
            'reference_implementation': {
                'A': 'PointNet++ (Qi et al. 2017), SSG variant',
                'B': 'PointNeXt-S (Qian et al. 2022), inverted residual blocks',
                'C': 'PointNeXt-S + TreeLearn-style stem offset + mask quality head',
            }[key],
            'implementation': 'LOCAL_IMPLEMENTATION (plain torch, no external code)',
            'pretrained_weights': None,
        }
    return out
