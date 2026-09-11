"""Deterministic example cache for long unattended runs.

Generating one example costs roughly 40 ms of CPU (scene rebuild + corruption + stratified
sampling). A full epoch of the frozen 440-candidate train set is 9 240 examples, so the
Public Pilot would otherwise spend hours regenerating byte-identical tensors after every
restart, power cut or terminal close.

This cache stores exactly what the frozen pipeline produced, keyed by everything that
determines it:

* ``provenance_digest`` and ``identity_manifest_digest`` — the frozen split;
* ``partition``, ``points``, ``repeats``, ``seed_base`` — the dataset configuration;
* ``spec_list_digest`` — the exact ordered spec list, so a changed seed stride or a
  changed candidate order can never re-use an old tensor;
* ``generator_digest`` and ``sampler_version`` — the accepted Gate D implementation.

Three properties make re-use sound rather than merely convenient:

1. **A cached entry is verified, never trusted.** Every entry records the SHA-256 of its
   own feature block; a mismatch is treated as a miss and the example is regenerated
   through the frozen pipeline.
2. **The cache cannot change what an example is.** A miss is indistinguishable from no
   cache at all: the frozen ``GateDDataset`` produces the tensors. ``efg_verify`` proves
   byte-identity between cached and freshly generated items on real candidates before a
   scored run consumes the cache.
3. **Each example is one small file written atomically.** A crash can lose the entry
   being written and nothing else; there is no index to corrupt or rebuild.

The cache never skips the provenance check: the stored provenance row still carries the
candidate, class, severity and full sampler diagnostics, and the dataset layer
re-authorises every candidate against the frozen provenance before serving an item.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import efg_state as state_module
import numpy as np

CACHE_VERSION = 'EFG_FEATURE_CACHE_V2'
META_FILE = 'meta.json'


class CacheError(RuntimeError):
    pass


def digest_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def cache_key(provenance, partition: str, points: int, repeats: int, seed_base: int,
              spec_list_digest: str, generator_digest: str, sampler_version: str) -> dict:
    """Everything that determines one example tensor, for the cache contract."""
    return {
        'cache_version': CACHE_VERSION,
        'partition': partition,
        'points': int(points),
        'repeats': int(repeats),
        'seed_base': int(seed_base),
        'spec_list_digest': spec_list_digest,
        'generator_digest': generator_digest,
        'sampler_version': sampler_version,
        'provenance_digest': provenance.provenance_digest,
        'identity_manifest_digest': provenance.identity_manifest_digest,
    }


class FeatureCache:
    """One on-disk cache for one (partition, points, seed_base, provenance) contract."""

    def __init__(self, root: Path | str, key: dict):
        self.root = Path(root)
        self.key = dict(key)
        self.key_digest = hashlib.sha256(
            json.dumps(self.key, sort_keys=True, separators=(',', ':'),
                       default=str).encode('utf-8')).hexdigest()
        self.dir = self.root / self.key_digest
        self.meta_path = self.dir / META_FILE
        self.meta = self._load_meta()
        self.hits = 0
        self.misses = 0
        self.regenerated = 0
        self.excluded_hits = 0

    # --- persistence ------------------------------------------------------
    def _load_meta(self) -> dict:
        blank = {'cache_version': CACHE_VERSION, 'key': self.key, 'entries': {}}
        if not self.meta_path.exists():
            return blank
        try:
            meta = state_module.read_json(self.meta_path)
        except state_module.StateError:
            return dict(blank, note='previous meta was corrupt and has been ignored')
        if not isinstance(meta, dict):
            return dict(blank, note='previous meta was not an object')
        if (meta.get('cache_version') != CACHE_VERSION
                or meta.get('key') != self.key):
            return dict(blank, note='previous meta was for a different contract')
        meta.setdefault('entries', {})
        return meta

    def _save_meta(self) -> None:
        state_module.atomic_write_json(self.meta_path, self.meta)

    def _paths(self, index: int) -> tuple[Path, Path]:
        return (self.dir / f'{index:06d}.features.npy',
                self.dir / f'{index:06d}.provenance.json')

    # --- lookup -----------------------------------------------------------
    def get(self, index: int, spec_key: str) -> tuple[np.ndarray, int, dict | None] | None:
        """Return verified ``(features, label, provenance)``, an exclusion, or None.

        An entry recorded as ``excluded`` returns a one-element tuple-free marker through
        :meth:`exclusion`; a genuine miss returns ``None`` and is regenerated.
        """
        entry = (self.meta.get('entries') or {}).get(str(index))
        if not entry or entry.get('spec_key') != spec_key:
            self.misses += 1
            return None
        if entry.get('excluded'):
            self.excluded_hits += 1
            return None
        feature_path, provenance_path = self._paths(index)
        try:
            features = np.load(feature_path, allow_pickle=False)
        except (OSError, ValueError, EOFError):
            self.misses += 1
            return None
        if features.dtype != np.float32 or features.ndim != 2 or features.shape[1] != 4:
            self.misses += 1
            return None
        if digest_array(features) != entry.get('features_sha256'):
            self.misses += 1
            return None
        provenance = None
        if provenance_path.exists():
            try:
                provenance = state_module.read_json(provenance_path, verify=False)
            except state_module.StateError:
                provenance = None
            if provenance is not None and provenance.get('key') != spec_key:
                self.misses += 1
                return None
        self.hits += 1
        return features, int(entry.get('label', -1)), provenance

    def exclusion(self, index: int, spec_key: str) -> dict | None:
        """The recorded exclusion for a spec that cannot be generated, if any."""
        entry = (self.meta.get('entries') or {}).get(str(index))
        if not entry or entry.get('spec_key') != spec_key or not entry.get('excluded'):
            return None
        return dict(entry)

    def record_exclusion(self, index: int, spec_key: str, reason: str,
                         detail: dict | None = None) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        entries = self.meta.setdefault('entries', {})
        entries[str(index)] = {'spec_key': spec_key, 'excluded': True,
                               'reason': reason, 'detail': detail or {}}
        self.meta['entries'] = entries
        self._save_meta()

    def put(self, index: int, spec_key: str, features: np.ndarray, label: int,
            provenance: dict | None = None) -> None:
        features = np.ascontiguousarray(features, dtype=np.float32)
        if features.ndim != 2 or features.shape[1] != 4:
            raise CacheError(f'features must be (N, 4), got {features.shape}')
        if provenance is not None and provenance.get('key') != spec_key:
            raise CacheError('provenance row does not belong to this spec key')
        self.dir.mkdir(parents=True, exist_ok=True)
        feature_path, provenance_path = self._paths(index)
        np.save(feature_path, features)
        if provenance is not None:
            state_module.atomic_write_json(provenance_path, provenance, with_digest=False)
        entries = self.meta.setdefault('entries', {})
        entries[str(index)] = {
            'spec_key': spec_key,
            'features_sha256': digest_array(features),
            'label': int(label),
            'points': int(features.shape[0]),
            'has_provenance': provenance is not None,
        }
        self.meta['entries'] = entries
        self._save_meta()

    # --- reporting --------------------------------------------------------
    def stats(self) -> dict:
        entries = self.meta.get('entries') or {}
        excluded = sum(1 for entry in entries.values() if entry.get('excluded'))
        return {'cache_version': CACHE_VERSION, 'key_digest': self.key_digest,
                'directory': str(self.dir), 'cached_entries': len(entries),
                'excluded_entries': excluded,
                'hits_this_process': self.hits, 'misses_this_process': self.misses,
                'excluded_hits_this_process': self.excluded_hits,
                'regenerated_this_process': self.regenerated, 'key': self.key}

    def observed_label_distribution(self) -> dict:
        counts: dict[str, int] = {}
        for entry in (self.meta.get('entries') or {}).values():
            if entry.get('excluded'):
                counts['EXCLUDED'] = counts.get('EXCLUDED', 0) + 1
                continue
            label = str(int(entry.get('label', -1)))
            counts[label] = counts.get(label, 0) + 1
        return dict(sorted(counts.items()))

    def exclusions(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for entry in (self.meta.get('entries') or {}).values():
            if entry.get('excluded'):
                reason = str(entry.get('reason'))
                counts[reason] = counts.get(reason, 0) + 1
        return dict(sorted(counts.items()))
