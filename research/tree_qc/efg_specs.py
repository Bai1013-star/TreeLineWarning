"""Partition-agnostic example specs for the Public Pilot.

The frozen ``gate_d_generator.enumerate_specs`` is the authority for model-selection
examples and is re-used verbatim for train and validation. It cannot express a held-out
spec, because ``gate_d_generator.ExampleSpec`` deliberately refuses any partition that is
not a model-selection partition — and that refusal is exactly the Gate D safeguard this
phase must preserve, not weaken.

This module therefore contributes a *second route to the same arithmetic*:

* ``enumerate_specs`` here is byte-for-byte the frozen loop (same stride constants, same
  ordering, same ``(candidate, class, severity, repeat) -> seed`` mapping);
* ``SamplingSpec`` carries the same four fields ``generate_example`` reads and the same
  ``key``/``as_row``/``sampling_seed`` methods.

Equality with the frozen module is not asserted by inspection. ``efg_heldout`` proves it
example by example, on real validation candidates, before any held-out geometry is read:
same spec key, same seed, same sampling seed, same label and byte-identical feature
tensor against the frozen ``GateDDataset``. Drift is a stop, not a warning.
"""
from __future__ import annotations

import hashlib
import json

import gate_d_generator as gen
import gate_d_sampler as sampler
from gate_d_provenance import HELD_OUT, MODEL_SELECTION_PARTITIONS

SPEC_VERSION = 'EFG_SAMPLING_SPEC_V1'
# Marker for a declared spec whose defect is not supported by the candidate's real
# geometry (``gate_d_generator.NotApplicableError``). The frozen dataset raises for these,
# which is correct — a defect that cannot be constructed must not be silently turned into
# a different example — but a full run over 9 240 declared specs meets some of them, so the
# EFG layer records the exclusion and removes the example from training and metrics instead
# of inventing a substitute.
NOT_APPLICABLE = 'NOT_APPLICABLE'
# Fraction of a partition that may be excluded before the run stops rather than reporting
# a metric over a silently shrunken population.
MAX_EXCLUSION_RATE = 0.05
# Same stride constants as gate_d_generator.enumerate_specs. They are copied as literals
# on purpose: importing them would not make the arithmetic any more equal, and the parity
# gate is what actually proves equality.
CANDIDATE_STRIDE = 1_000_003
CLASS_STRIDE = 10_007
SEVERITY_STRIDE = 101

EFG_PARTITIONS = (MODEL_SELECTION_PARTITIONS[0], MODEL_SELECTION_PARTITIONS[1], HELD_OUT)


class SpecError(RuntimeError):
    pass


class SamplingSpec:
    """One example recipe: the frozen ``ExampleSpec`` contract, plus held-out support."""

    __slots__ = ('candidate_id', 'partition', 'example_class', 'severity', 'seed',
                 'repeat')

    def __init__(self, candidate_id: str, partition: str, example_class: str,
                 severity: str, seed: int, repeat: int = 0):
        if example_class not in gen.CLASSES:
            raise SpecError(f'unknown class {example_class!r}')
        if severity not in gen.SEVERITIES:
            raise SpecError(f'unknown severity {severity!r}')
        if partition not in EFG_PARTITIONS:
            raise SpecError(f'unknown partition {partition!r}')
        if int(repeat) < 0:
            raise SpecError(f'repeat must be non-negative, got {repeat!r}')
        self.candidate_id = candidate_id
        self.partition = partition
        self.example_class = example_class
        self.severity = severity
        self.seed = int(seed)
        self.repeat = int(repeat)

    @property
    def key(self) -> str:
        return (f'{self.partition}|{self.candidate_id}|{self.example_class}'
                f'|{self.severity}|{self.seed}')

    def as_row(self) -> dict:
        return {'candidate_id': self.candidate_id, 'partition': self.partition,
                'class': self.example_class, 'severity': self.severity,
                'seed': self.seed, 'repeat': self.repeat, 'key': self.key}

    @classmethod
    def from_row(cls, row: dict) -> SamplingSpec:
        return cls(candidate_id=row['candidate_id'], partition=row['partition'],
                   example_class=row['class'], severity=row['severity'],
                   seed=int(row['seed']), repeat=int(row.get('repeat', 0)))

    def sampling_seed(self, points_target: int = sampler.POINTS_PRIMARY,
                      seed_base: int = 0) -> int:
        """Class-independent sampler seed, identical to the frozen mapping."""
        return sampler.sampling_seed(self.partition, self.candidate_id, self.severity,
                                     points_target=points_target,
                                     repeat=self.repeat, seed_base=seed_base)


def enumerate_specs(candidate_ids, partition: str, classes=gen.CLASSES,
                    severities=gen.SEVERITIES, seed_base: int = 0,
                    repeats: int = 1) -> list[SamplingSpec]:
    """The frozen enumeration loop, extended to any declared partition."""
    if partition not in EFG_PARTITIONS:
        raise SpecError(f'unknown partition {partition!r}')
    specs: list[SamplingSpec] = []
    for index, candidate_id in enumerate(candidate_ids):
        for class_index, example_class in enumerate(classes):
            for severity_index, severity in enumerate(severities):
                for repeat in range(repeats):
                    seed = (seed_base + CANDIDATE_STRIDE * index + CLASS_STRIDE * class_index
                            + SEVERITY_STRIDE * severity_index + repeat)
                    specs.append(SamplingSpec(candidate_id=candidate_id,
                                              partition=partition,
                                              example_class=example_class,
                                              severity=severity, seed=seed,
                                              repeat=repeat))
    return specs


def spec_list_digest(specs) -> str:
    payload = json.dumps([spec.as_row() for spec in specs], sort_keys=True,
                         separators=(',', ':'))
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def class_balance(provenance, candidate_ids, partition: str, points: int = 4096,
                  seed_base: int = 0, repeats: int = 1) -> dict:
    """Declared example counts per class|severity, without loading any geometry.

    STEM_MISSING applicability is a geometry fact, so the realised counts can be lower
    than the declared ones. The declared numbers are what the freeze records; the
    realised numbers are recorded after generation.
    """
    specs = enumerate_specs(candidate_ids, partition, seed_base=seed_base,
                            repeats=repeats)
    counts: dict[str, int] = {}
    for spec in specs:
        key = f'{spec.example_class}|{spec.severity}'
        counts[key] = counts.get(key, 0) + 1
    return {'declared_examples': len(specs), 'declared_per_class_severity':
            dict(sorted(counts.items())),
            'points': int(points), 'repeats': int(repeats), 'seed_base': int(seed_base),
            'spec_list_digest': spec_list_digest(specs)}
