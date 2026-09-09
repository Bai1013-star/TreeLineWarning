"""Pure torch diagnostic models. XYZ stays in metres; channel 4 is membership."""
import torch
from torch import nn


def gather(points, indices):
    batch = torch.arange(points.shape[0], device=points.device)
    return points[batch.reshape((-1,) + (1,) * (indices.ndim - 1)), indices]


@torch.no_grad()
def farthest_point_sample(xyz, count):
    """Deterministic FPS seeded farthest from centroid, O(B*N*count)."""
    xyz = xyz.float()
    count = min(count, xyz.shape[1])
    indices = torch.empty(xyz.shape[0], count, dtype=torch.long, device=xyz.device)
    distance = torch.full(xyz.shape[:2], float('inf'), device=xyz.device)
    farthest = ((xyz - xyz.mean(1, keepdim=True)) ** 2).sum(-1).argmax(1)
    for i in range(count):
        indices[:, i] = farthest
        centroid = gather(xyz, farthest[:, None])
        distance = torch.minimum(distance, ((xyz - centroid) ** 2).sum(-1))
        farthest = distance.argmax(1)
    return indices


@torch.no_grad()
def radius_group(xyz, centers, radius, count):
    """Nearest within physical radius; fill empty slots with nearest point."""
    distance = torch.cdist(centers.float(), xyz.float())
    values, indices = distance.topk(min(count, xyz.shape[1]), largest=False)
    nearest = indices[..., :1].expand_as(indices)
    return torch.where(values <= radius, indices, nearest)


def mlp2(channels):
    layers = []
    for a, b in zip(channels, channels[1:]):
        layers.extend([nn.Conv2d(a, b, 1), nn.ReLU()])
    return nn.Sequential(*layers)


class SetAbstraction(nn.Module):
    def __init__(self, count, radius, neighbors, channels):
        super().__init__()
        self.count, self.radius, self.neighbors = count, radius, neighbors
        self.mlp = mlp2(channels)

    def forward(self, xyz, features):
        centers = gather(xyz, farthest_point_sample(xyz, self.count))
        indices = radius_group(xyz, centers, self.radius, self.neighbors)
        relative = gather(xyz, indices) - centers[:, :, None, :]
        grouped = torch.cat([relative, gather(features, indices)], -1)
        output = self.mlp(grouped.permute(0, 3, 1, 2)).amax(-1).transpose(1, 2)
        return centers, output


class PointNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = mlp2([4, 32, 64, 128, 256])
        self.head = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 2))

    def forward(self, points):
        return self.head(self.encoder(points.transpose(1, 2).unsqueeze(-1)).amax((2, 3)))


class PointNet2SSG(nn.Module):
    def __init__(self):
        super().__init__()
        self.sa1 = SetAbstraction(128, 1.5, 32, [4, 32, 32, 64])
        self.sa2 = SetAbstraction(32, 4.0, 32, [67, 64, 64, 128])
        self.global_mlp = mlp2([131, 128, 256, 512])
        self.head = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 2))

    def forward(self, points):
        xyz, feat = self.sa1(points[..., :3], points[..., 3:])
        xyz, feat = self.sa2(xyz, feat)
        local = xyz - xyz.mean(1, keepdim=True)
        pooled = self.global_mlp(torch.cat([local, feat], -1).transpose(1, 2).unsqueeze(-1)).amax((2, 3))
        return self.head(pooled)


MODELS = {'pointnet': PointNet, 'pointnet2_ssg': PointNet2SSG}
