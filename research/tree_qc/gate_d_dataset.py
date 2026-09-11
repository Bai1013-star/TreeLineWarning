"""Model-agnostic Gate D dataset.

Pipeline: ``ExampleSpec -> SyntheticGenerator -> StructuredSampler -> Tensor``.

The module imports numpy only. It never imports a model, so the generator and the
sampler stay independent of whatever consumes them. PointNet and PointNet++ share
this exact dataset, split, corruption configuration and sampler; only the network
differs.

Held-out trees are unreachable by construction: specs are built from a partition's
``eligible`` list, and every example is re-authorised by the provenance gate before
its geometry is loaded.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import gate_d_generator as gen
import gate_d_geometry as geometry
import gate_d_sampler as sampler
import gate_d_scene as scene_module
import numpy as np
from gate_d_provenance import MODEL_SELECTION_PARTITIONS, GateDProvenance

DEFAULT_CACHE_ROOT = Path('D:/TreeQCTraining/gate_d_cache')
DATASET_VERSION = 'GATE_D_DATASET_V2'
LABEL_INDEX = {gen.LABEL_PASS: 0, gen.LABEL_REVIEW: 1}


class DatasetError(RuntimeError):
    pass


@dataclass(frozen=True)
class DatasetConfig:
    """Everything that determines a dataset. Serialisable and restart-stable."""

    partition: str
    points: int = sampler.POINTS_PRIMARY
    classes: tuple[str, ...] = gen.CLASSES
    severities: tuple[str, ...] = gen.SEVERITIES
    repeats: int = 1
    seed_base: int = 0
    version: str = DATASET_VERSION

    def __post_init__(self) -> None:
        if self.partition not in MODEL_SELECTION_PARTITIONS:
            raise DatasetError(f'{self.partition!r} is not a model-selection partition')
        if self.points not in (sampler.POINTS_PRIMARY, sampler.POINTS_ROBUSTNESS):
            raise DatasetError(
                f'points must be {sampler.POINTS_PRIMARY} or '
                f'{sampler.POINTS_ROBUSTNESS}, got {self.points}')

    def as_row(self) -> dict:
        return {'partition': self.partition, 'points': self.points,
                'classes': list(self.classes), 'severities': list(self.severities),
                'repeats': self.repeats, 'seed_base': self.seed_base,
                'version': self.version}


@dataclass(frozen=True)
class DatasetItem:
    features: np.ndarray
    label: int
    key: str
    provenance: dict


class GateDDataset:
    """Deterministic, provenance-bound dataset over one model-selection partition."""

    def __init__(self, config: DatasetConfig, provenance: GateDProvenance,
                 candidate_ids=None, cache_root: Path | str | None = None,
                 data_root: Path | None = None):
        if config.partition not in MODEL_SELECTION_PARTITIONS:
            raise DatasetError(f'partition {config.partition!r} is not model-selection')
        self.config = config
        self.provenance = provenance
        self.cache_root = Path(cache_root) if cache_root is not None else DEFAULT_CACHE_ROOT
        self.data_root = data_root
        if candidate_ids is None:
            candidate_ids = provenance.candidates(config.partition)
        self.candidate_ids = tuple(candidate_ids)
        for candidate_id in self.candidate_ids:
            if provenance.partition_of_candidate(candidate_id) != config.partition:
                raise DatasetError(
                    f'{candidate_id} does not belong to partition {config.partition}')
        self.specs = tuple(gen.enumerate_specs(
            self.candidate_ids, config.partition, config.classes, config.severities,
            seed_base=config.seed_base, repeats=config.repeats))
        self._geometry: dict[str, geometry.CandidateGeometry] = {}

    # --- lazy, bounded geometry -----------------------------------------
    def _geometry_for(self, candidate_id: str) -> geometry.CandidateGeometry:
        cached = self._geometry.get(candidate_id)
        if cached is not None:
            return cached
        loaded = geometry.load_cached_geometry(
            self.cache_root, self.provenance, candidate_id, self.data_root)
        self._geometry[candidate_id] = loaded
        return loaded

    def preload(self, candidate_ids=None) -> None:
        """Warm the cache for a bounded candidate set in one plot pass."""
        ids = list(candidate_ids if candidate_ids is not None else
                   sorted({spec.candidate_id for spec in self.specs}))
        loaded = geometry.load_cached_geometry_batch(
            self.cache_root, self.provenance, ids, self.data_root)
        self._geometry.update(loaded)

    # --- dataset protocol ------------------------------------------------
    def __len__(self) -> int:
        return len(self.specs)

    def __getitem__(self, index: int) -> DatasetItem:
        spec = self.specs[index]
        geometry_for_candidate = self._geometry_for(spec.candidate_id)
        try:
            example = gen.generate_example(spec, geometry_for_candidate, self.provenance)
        except gen.NotApplicableError as exc:
            raise DatasetError(
                f'{spec.key} is not applicable: {exc.reason} {exc.detail}') from exc
        scene = scene_module.build_base_scene(spec.candidate_id, spec.partition,
                                              geometry_for_candidate, self.provenance)
        # Corruption RNG uses spec.seed (class-indexed by design); the sampler RNG uses
        # the class-independent sampling seed so the class cannot leak through the draw.
        sample_seed = spec.sampling_seed(self.config.points, self.config.seed_base)
        sample = sampler.sample(example.points, example.membership,
                                example.changed_membership, scene.candidate_mask,
                                sample_seed, self.config.points, spec.repeat)
        provenance = dict(example.provenance)
        provenance['sample'] = {
            'sampler_version': sampler.SAMPLER_VERSION,
            'sampling_seed': int(sample_seed),
            'corruption_seed': int(spec.seed),
            'repeat': int(spec.repeat),
            'points_target': int(sample.diagnostics['points_target']),
            'points': sample.points,
            'source_points': sample.source_count,
            'features': list(sampler.FEATURES),
            'diagnostics': sample.diagnostics,
        }
        provenance['features_digest'] = hashlib.sha256(
            np.ascontiguousarray(sample.features).tobytes()).hexdigest()
        return DatasetItem(features=sample.features,
                           label=LABEL_INDEX[example.label], key=spec.key,
                           provenance=provenance)

    # --- restart / audit --------------------------------------------------
    def digest(self) -> str:
        payload = json.dumps({
            'version': DATASET_VERSION,
            'config': self.config.as_row(),
            'candidate_ids': list(self.candidate_ids),
            'provenance_digest': self.provenance.provenance_digest,
            'identity_manifest_digest': self.provenance.identity_manifest_digest,
        }, sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(payload.encode('utf-8')).hexdigest()

    def manifest(self) -> list[dict]:
        """Full provenance for every example, without loading geometry."""
        return [spec.as_row() for spec in self.specs]

    def class_balance(self) -> dict:
        counts: dict[str, int] = {}
        for spec in self.specs:
            key = f'{spec.example_class}|{spec.severity}'
            counts[key] = counts.get(key, 0) + 1
        return dict(sorted(counts.items()))

    def restart_state(self) -> dict:
        """Minimal state that reproduces this dataset exactly."""
        return {'dataset_version': DATASET_VERSION, 'digest': self.digest(),
                'config': self.config.as_row(), 'examples': len(self.specs),
                'candidate_count': len(self.candidate_ids),
                'candidate_ids_digest': hashlib.sha256(
                    json.dumps(list(self.candidate_ids)).encode('utf-8')).hexdigest()}


class TorchGateDDataset:
    """Thin torch adapter. Imported lazily so the core stays model-agnostic."""

    def __init__(self, dataset: GateDDataset):
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        import torch
        item = self.dataset[index]
        return (torch.from_numpy(np.ascontiguousarray(item.features)),
                torch.tensor(item.label, dtype=torch.long), index)


def collate(batch):
    import torch
    features = torch.stack([b[0] for b in batch], 0)
    labels = torch.stack([b[1] for b in batch], 0)
    indices = torch.stack([b[2] for b in batch], 0)
    return features, labels, indices


def make_loader(dataset: GateDDataset, batch_size: int, shuffle: bool, seed: int,
                workers: int = 0, pin_memory: bool = False):
    """Windows-safe DataLoader: spawn start method, persistent workers, seeded order."""
    import torch
    from torch.utils.data import DataLoader

    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(TorchGateDDataset(dataset), batch_size=batch_size,
                      shuffle=shuffle, num_workers=workers, collate_fn=collate,
                      generator=generator, pin_memory=pin_memory,
                      persistent_workers=workers > 0, drop_last=False)
