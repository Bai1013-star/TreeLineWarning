"""Gate D provenance gate: the only authority for which trees an example may use.

The donor contract frozen by the Supervisor is implemented here as an **allowlist**,
not a denylist:

* a donor is legal only if it is a member of that candidate's own surviving
  ``context`` in ``GATE_C2_CONTEXT_PROVENANCE.json``;
* no source may cross partitions;
* the 25 high-containment quarantine trees and every alias file of them may never
  appear as candidate, context, donor or source;
* held-out sources are never reachable from a train/validation example.

Every violation raises. Nothing is silently skipped, downgraded or repaired, because
a repaired example is a silently mislabelled example.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

REPORTS = Path(__file__).resolve().parent / 'reports'
PROVENANCE_FILE = 'GATE_C2_CONTEXT_PROVENANCE.json'
MANIFEST_FILE = 'GATE_C2_IDENTITY_MANIFEST.json'
REMEDIATION_FILE = 'GATE_C2_REMEDIATION.json'
PROTOCOL_FILE = 'GATE_C_CANONICAL_IDENTITY_PROTOCOL_V2.md'

TRAIN = 'train'
VALIDATION = 'validation'
HELD_OUT = 'held_out'
MODEL_SELECTION_PARTITIONS = (TRAIN, VALIDATION)

EXPECTED_PROTOCOL = 'GATE_C_CANONICAL_IDENTITY_PROTOCOL_V2'
EXPECTED_TRAIN_UNIQUE = 440
EXPECTED_VALIDATION_UNIQUE = 175
EXPECTED_HELDOUT_UNIQUE = 749
EXPECTED_QUARANTINE_TREES = 25


class ProvenanceViolation(RuntimeError):
    """Base class: every provenance problem is fatal by design."""


class UnknownTreeError(ProvenanceViolation):
    pass


class CrossPartitionError(ProvenanceViolation):
    pass


class QuarantineError(ProvenanceViolation):
    pass


class HeldOutAccessError(ProvenanceViolation):
    pass


class UnauthorizedDonorError(ProvenanceViolation):
    pass


class SelfDonorError(ProvenanceViolation):
    pass


class AliasViolation(ProvenanceViolation):
    pass


class ManifestMismatchError(ProvenanceViolation):
    pass


@dataclass(frozen=True)
class PartitionView:
    """Frozen, validated view of one model-selection partition."""

    name: str
    eligible: tuple[str, ...]
    context: dict[str, tuple[str, ...]]
    blocked: tuple[str, ...]
    source_union: tuple[str, ...]


@dataclass(frozen=True)
class GateDProvenance:
    """Everything the generator is allowed to know about the frozen split.

    Two quarantine concepts must never be conflated:

    * ``high_containment_quarantine`` — the 25 Wytham trees of the 17 high-containment
      pairs. Forbidden in **every** role (candidate, context, donor, alias source).
    * ``boundary_quarantine`` — the 84 Litchfield trees whose bbox crosses the 3 m
      train/validation boundary. They may not be candidates, but the frozen provenance
      legitimately keeps 70 of them as context for model-selection candidates.
      Context membership is therefore governed by the allowlist, not by this set.
    """

    protocol: str
    protocol_digest: str
    identity_manifest_digest: str
    provenance_digest: str
    train: PartitionView
    validation: PartitionView
    held_out_eligible: tuple[str, ...]
    high_containment_quarantine: frozenset[str]
    high_containment_aliases: frozenset[str]
    boundary_quarantine: frozenset[str]
    aliases_by_canonical_id: dict[str, tuple[str, ...]]
    tree: dict[str, dict]
    partition_of: dict[str, str]
    source_aliases_of: dict[str, tuple[str, ...]]
    site_of: dict[str, str]

    # --- authorization ---------------------------------------------------
    def partition_of_candidate(self, canonical_tree_id: str) -> str:
        if canonical_tree_id not in self.partition_of:
            raise UnknownTreeError(f'unknown canonical tree: {canonical_tree_id}')
        partition = self.partition_of[canonical_tree_id]
        if partition not in MODEL_SELECTION_PARTITIONS:
            raise HeldOutAccessError(
                f'{canonical_tree_id} is {partition}, not a model-selection candidate')
        return partition

    def view(self, partition: str) -> PartitionView:
        if partition == TRAIN:
            return self.train
        if partition == VALIDATION:
            return self.validation
        raise HeldOutAccessError(f'partition {partition} is not a model-selection partition')

    def authorized_context(self, candidate: str) -> tuple[str, ...]:
        """The candidate's own surviving context. This is the donor universe."""
        partition = self.partition_of_candidate(candidate)
        context = self.view(partition).context.get(candidate)
        if context is None:
            raise UnknownTreeError(f'{candidate} has no surviving context record')
        return context

    def authorize_donor(self, candidate: str, donor: str) -> None:
        """Raise unless `donor` may legally donate geometry to `candidate`.

        Checks run from the most absolute prohibition to the most specific:

        1. self-donation is impossible;
        2. the donor must exist;
        3. high-containment quarantine is forbidden in every role;
        4. held-out geometry is unreachable from a model-selection example;
        5. a donor from the other model-selection partition is a leak;
        6. finally, the allowlist: the donor must be in the candidate's own surviving
           context. A boundary-quarantine tree passes 3-5 and is legal context.
        """
        if donor == candidate:
            raise SelfDonorError(f'{candidate} may not donate to itself')
        candidate_partition = self.partition_of_candidate(candidate)
        if donor not in self.partition_of:
            raise UnknownTreeError(f'unknown donor: {donor}')
        self.assert_not_quarantined(donor, 'donor')
        donor_partition = self.partition_of[donor]
        if donor_partition == HELD_OUT:
            raise HeldOutAccessError(f'held-out donor {donor} refused for {candidate}')
        if (donor_partition in MODEL_SELECTION_PARTITIONS
                and donor_partition != candidate_partition):
            raise CrossPartitionError(
                f'donor {donor} ({donor_partition}) crosses into '
                f'{candidate_partition} for {candidate}')
        if donor not in self.authorized_context(candidate):
            raise UnauthorizedDonorError(
                f'{donor} is not in the surviving context of {candidate}')
        # A boundary-quarantine tree is a legal context donor: it is not a candidate
        # in either partition and the frozen provenance explicitly kept it as context.
        return

    def assert_not_quarantined(self, canonical_tree_id: str, role: str) -> None:
        if canonical_tree_id in self.high_containment_quarantine:
            raise QuarantineError(
                f'{role} {canonical_tree_id} is a high-containment quarantine tree')

    def assert_not_boundary_candidate(self, canonical_tree_id: str) -> None:
        if canonical_tree_id in self.boundary_quarantine:
            raise CrossPartitionError(
                f'candidate {canonical_tree_id} is a boundary-quarantine tree')

    def assert_alias_clean(self, canonical_tree_id: str, partition: str | None = None) -> None:
        """No alias file of a quarantined tree may be used, and aliases must not cross."""
        aliases = self.source_aliases_of.get(canonical_tree_id, ())
        for alias in aliases:
            if alias in self.high_containment_aliases:
                raise QuarantineError(
                    f'alias {alias} of {canonical_tree_id} belongs to a '
                    f'high-containment quarantined tree')
        if partition is not None:
            for alias in aliases:
                other = self.alias_owner_partition(alias, exclude=canonical_tree_id)
                if other is not None and other != partition:
                    raise AliasViolation(
                        f'alias {alias} of {canonical_tree_id} is also owned by a '
                        f'{other} tree; aliases may not cross partitions')

    def alias_owner_partition(self, alias: str, exclude: str | None = None) -> str | None:
        for cid, aliases in self.aliases_by_canonical_id.items():
            if cid == exclude:
                continue
            if alias in aliases:
                return self.partition_of.get(cid)
        return None

    def assert_example_sources(self, candidate: str, donors: tuple[str, ...] | list[str]) -> None:
        """Full fail-closed check for one generated example."""
        partition = self.partition_of_candidate(candidate)
        self.assert_not_quarantined(candidate, 'candidate')
        self.assert_not_boundary_candidate(candidate)
        self.assert_alias_clean(candidate, partition)
        for donor in donors:
            self.authorize_donor(candidate, donor)
            self.assert_alias_clean(donor, partition)
        union = {candidate, *donors}
        for cid in union:
            self.assert_not_quarantined(cid, 'source')
            if self.partition_of[cid] == HELD_OUT:
                raise HeldOutAccessError(f'held-out source {cid} reached an example')

    def assert_scene_sources(self, candidate: str,
                             contributing_sources) -> tuple[str, ...]:
        """Every source that contributed a point must be legal for this candidate.

        This is the unified check the task requires: candidate, context, donor,
        background and alias sources all pass through it, and the candidate's legal
        external source set is exactly its frozen surviving context.
        """
        partition = self.partition_of_candidate(candidate)
        self.assert_not_quarantined(candidate, 'candidate')
        self.assert_not_boundary_candidate(candidate)
        self.assert_alias_clean(candidate, partition)
        context = set(self.authorized_context(candidate))
        union = tuple(sorted(set(contributing_sources)))
        for source_id in union:
            if source_id not in self.partition_of:
                raise UnknownTreeError(f'unknown contributing source: {source_id}')
            self.assert_not_quarantined(source_id, 'source')
            self.assert_alias_clean(source_id, partition)
            source_partition = self.partition_of[source_id]
            if source_partition == HELD_OUT:
                raise HeldOutAccessError(
                    f'held-out source {source_id} reached an example')
            if source_id == candidate:
                continue
            if source_id not in context:
                raise UnauthorizedDonorError(
                    f'{source_id} is not in the surviving context of {candidate}; '
                    'external geometry may only come from the frozen allowlist')
            if (source_partition in MODEL_SELECTION_PARTITIONS
                    and source_partition != partition):
                raise CrossPartitionError(
                    f'source {source_id} ({source_partition}) crosses into '
                    f'{partition} for {candidate}')
        if candidate not in union:
            raise ManifestMismatchError(
                f'candidate {candidate} contributed no points to its own example')
        return union

    # --- geometry lookup -------------------------------------------------
    def source_file(self, canonical_tree_id: str) -> str:
        record = self.tree.get(canonical_tree_id)
        if record is None:
            raise UnknownTreeError(f'unknown canonical tree: {canonical_tree_id}')
        return record['source_file']

    def site(self, canonical_tree_id: str) -> str:
        try:
            return self.site_of[canonical_tree_id]
        except KeyError as exc:
            raise UnknownTreeError(f'unknown canonical tree: {canonical_tree_id}') from exc

    def candidates(self, partition: str) -> tuple[str, ...]:
        return self.view(partition).eligible


def _digest(payload) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(',', ':'), default=str)
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def protocol_digest(reports: Path | None = None) -> str:
    """Digest of the frozen Gate C2 protocol document.

    The training lock binds this value, so editing the frozen protocol invalidates any
    acceptance that was granted against the previous text.
    """
    root = Path(reports) if reports is not None else REPORTS
    return hashlib.sha256((root / PROTOCOL_FILE).read_bytes()).hexdigest()


def load_gate_d_provenance(reports: Path | None = None,
                           require_frozen_counts: bool = True) -> GateDProvenance:
    """Load and fully validate the frozen Gate C2 provenance."""
    root = Path(reports) if reports is not None else REPORTS
    provenance = json.loads((root / PROVENANCE_FILE).read_text(encoding='utf-8'))
    manifest = json.loads((root / MANIFEST_FILE).read_text(encoding='utf-8'))
    remediation = json.loads((root / REMEDIATION_FILE).read_text(encoding='utf-8'))

    protocol = remediation['protocol']
    if not protocol.startswith(EXPECTED_PROTOCOL):
        raise ManifestMismatchError(f'unexpected protocol: {protocol}')

    tree = {row['canonical_tree_id']: row for row in manifest}
    if len(tree) != len(manifest):
        raise ManifestMismatchError('duplicate canonical_tree_id in identity manifest')
    partition_of = {cid: row['partition'] for cid, row in tree.items()}
    site_of = {cid: row['site'] for cid, row in tree.items()}
    source_aliases_of = {cid: tuple(row['source_aliases']) for cid, row in tree.items()}

    aliases = {cid: tuple(v) for cid, v in provenance['aliases_by_canonical_id'].items()}
    for cid, owned in aliases.items():
        if cid not in tree:
            raise ManifestMismatchError(f'alias map names unknown tree {cid}')
        if tuple(owned) != source_aliases_of[cid]:
            raise ManifestMismatchError(f'alias map disagrees with manifest for {cid}')

    quarantine = frozenset(remediation['high_containment']['trees'])
    if require_frozen_counts and len(quarantine) != EXPECTED_QUARANTINE_TREES:
        raise ManifestMismatchError(
            f'expected {EXPECTED_QUARANTINE_TREES} quarantine trees, got {len(quarantine)}')
    quarantine_aliases = frozenset(
        alias for cid in quarantine for alias in source_aliases_of.get(cid, ()))
    for cid in quarantine:
        if partition_of.get(cid) != HELD_OUT:
            raise ManifestMismatchError(f'quarantine tree {cid} is not held-out')
    boundary_quarantine = frozenset(
        cid for cid, part in partition_of.items() if part == 'quarantine')
    if quarantine & boundary_quarantine:
        raise ManifestMismatchError('high-containment and boundary quarantine must be disjoint')

    def view(name: str, block_field: bool) -> PartitionView:
        payload = provenance[name]
        eligible = tuple(payload['eligible'])
        context = {cid: tuple(v) for cid, v in payload['context'].items()}
        blocked = tuple(payload.get('blocked', ()))
        return PartitionView(name=name, eligible=eligible, context=context,
                             blocked=blocked,
                             source_union=tuple(payload['source_union']))

    train = view(TRAIN, True)
    validation = view(VALIDATION, True)
    held_out = provenance[HELD_OUT]

    if require_frozen_counts:
        if len(train.eligible) != EXPECTED_TRAIN_UNIQUE:
            raise ManifestMismatchError(
                f'expected TRAIN_UNIQUE {EXPECTED_TRAIN_UNIQUE}, got {len(train.eligible)}')
        if len(validation.eligible) != EXPECTED_VALIDATION_UNIQUE:
            raise ManifestMismatchError(
                f'expected VALIDATION_UNIQUE {EXPECTED_VALIDATION_UNIQUE}, '
                f'got {len(validation.eligible)}')
        if len(held_out['eligible']) != EXPECTED_HELDOUT_UNIQUE:
            raise ManifestMismatchError(
                f'expected HELDOUT_FROZEN_COUNT {EXPECTED_HELDOUT_UNIQUE}, '
                f'got {len(held_out["eligible"])}')

    frozen = GateDProvenance(
        protocol=protocol,
        protocol_digest=protocol_digest(root),
        identity_manifest_digest=remediation['identity_manifest_digest'],
        provenance_digest=_digest(provenance),
        train=train, validation=validation,
        held_out_eligible=tuple(held_out['eligible']),
        high_containment_quarantine=quarantine,
        high_containment_aliases=quarantine_aliases,
        boundary_quarantine=boundary_quarantine,
        aliases_by_canonical_id=aliases, tree=tree, partition_of=partition_of,
        source_aliases_of=source_aliases_of, site_of=site_of)

    _self_check(frozen)
    return frozen


def _self_check(frozen: GateDProvenance) -> None:
    """Re-derive the frozen invariants instead of trusting the artifacts."""
    model_selection = set(frozen.train.eligible) | set(frozen.validation.eligible)
    for cid in model_selection:
        if frozen.partition_of[cid] not in MODEL_SELECTION_PARTITIONS:
            raise ManifestMismatchError(f'{cid} is eligible but {frozen.partition_of[cid]}')
        if frozen.partition_of[cid] != (TRAIN if cid in frozen.train.eligible else VALIDATION):
            raise ManifestMismatchError(f'{cid} partition label disagrees with eligibility')
        frozen.assert_not_quarantined(cid, 'candidate')
        frozen.assert_not_boundary_candidate(cid)
    for view in (frozen.train, frozen.validation):
        for cid in view.eligible:
            context = view.context[cid]
            if not context:
                raise ManifestMismatchError(f'{cid} has empty surviving context')
            for neighbour in context:
                if neighbour == cid:
                    raise SelfDonorError(f'{cid} lists itself as context')
                if neighbour not in frozen.partition_of:
                    raise UnknownTreeError(f'context {neighbour} of {cid} is unknown')
                # High containment and held-out are absolute; boundary quarantine is
                # legal context by the frozen allowlist.
                frozen.assert_not_quarantined(neighbour, 'context')
                if frozen.partition_of[neighbour] == HELD_OUT:
                    raise HeldOutAccessError(f'held-out context {neighbour} of {cid}')
                if frozen.partition_of[neighbour] == 'quarantine':
                    if neighbour not in frozen.boundary_quarantine:
                        raise CrossPartitionError(f'context {neighbour} of {cid} is quarantined')
    train_union = set(frozen.train.source_union)
    validation_union = set(frozen.validation.source_union)
    if train_union & validation_union:
        raise CrossPartitionError(
            f'train/validation source unions intersect: {sorted(train_union & validation_union)}')
    for union, label in ((train_union, 'train'), (validation_union, 'validation')):
        if union & frozen.high_containment_quarantine:
            raise QuarantineError(f'high-containment tree inside the {label} source union')
    for view in (frozen.train, frozen.validation):
        for cid in view.eligible:
            if set(view.context[cid]) - set(view.source_union):
                raise ManifestMismatchError(f'context of {cid} escapes its source union')
            if cid not in view.source_union:
                raise ManifestMismatchError(f'candidate {cid} missing from its source union')


def require_gate_d_accepted(marker: Path | str, expected_digest: str | None = None) -> dict:
    """Fail-closed training lock. Raises unless the Supervisor accepted Gate D."""
    path = Path(marker)
    if not path.exists():
        raise ManifestMismatchError('GATE_D_NOT_ACCEPTED: marker absent at ' + str(path))
    payload = json.loads(path.read_text(encoding='utf-8-sig'))
    if payload.get('gate_d') != 'ACCEPTED':
        raise ManifestMismatchError(
            f'GATE_D_NOT_ACCEPTED: marker says {payload.get("gate_d")!r}')
    if payload.get('training_allowed') is not True:
        raise ManifestMismatchError('GATE_D_NOT_ACCEPTED: training_allowed is not true')
    if expected_digest is not None and payload.get('generator_digest') != expected_digest:
        raise ManifestMismatchError(
            'GATE_D_NOT_ACCEPTED: accepted generator digest does not match current generator')
    return payload
