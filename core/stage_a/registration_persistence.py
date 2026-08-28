"""Persist corridor registration in the existing Stage-A Scan envelope.

The V0.1 corridor workflow deliberately does not add a second project schema.
Registration status stays on ``Scan.registration_status`` and detailed evidence
is stored under ``Scan.metadata['corridor_registration_v1']``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from core.stage_a.models import Project, Scan
from core.stage_a.persistence import load_project, save_project
from core.stage_a.registration import AnchorCorrespondence, RegistrationResult

REGISTRATION_METADATA_KEY = "corridor_registration_v1"
REGISTRATION_METADATA_SCHEMA = "treeline-corridor-registration-v1"


def _json_mapping(value: Mapping[str, Any] | None) -> dict[str, Any] | None:
    return None if value is None else {str(key): item for key, item in value.items()}


def save_scan_registration(
    root: Path | str,
    scan_id: str,
    registration: RegistrationResult,
    *,
    anchors: Sequence[AnchorCorrespondence] = (),
    checkpoint: AnchorCorrespondence | None = None,
    gis_source_identity: Mapping[str, Any] | None = None,
) -> Project:
    """Persist registration evidence without changing the Stage-A project schema."""

    if type(registration) is not RegistrationResult:
        raise ValueError("registration must be a RegistrationResult")
    project = load_project(root)
    target = next((scan for scan in project.scans if scan.scan_id == scan_id), None)
    if target is None:
        raise ValueError("SCAN_NOT_FOUND")
    if any(type(anchor) is not AnchorCorrespondence for anchor in anchors):
        raise ValueError("anchors must contain AnchorCorrespondence values")
    if checkpoint is not None and type(checkpoint) is not AnchorCorrespondence:
        raise ValueError("checkpoint must be an AnchorCorrespondence")

    record = {
        "schema_version": REGISTRATION_METADATA_SCHEMA,
        "result": registration.to_dict(),
        "anchors": [anchor.to_dict() for anchor in anchors],
        "checkpoint": checkpoint.to_dict() if checkpoint is not None else None,
        "gis_source_identity": _json_mapping(gis_source_identity),
    }
    metadata = dict(target.metadata)
    metadata[REGISTRATION_METADATA_KEY] = record
    updated_scan = replace(
        target,
        registration_status=registration.state.value,
        metadata=metadata,
    )
    scans = tuple(updated_scan if scan.scan_id == scan_id else scan for scan in project.scans)
    updated = replace(project, scans=scans)
    save_project(updated, root)
    return updated


def scan_registration_record(scan: Scan) -> dict[str, Any] | None:
    """Return persisted registration evidence when the V1 metadata is present."""

    if type(scan) is not Scan:
        raise ValueError("scan must be a Scan")
    raw = scan.metadata.get(REGISTRATION_METADATA_KEY)
    if not isinstance(raw, Mapping):
        return None
    if raw.get("schema_version") != REGISTRATION_METADATA_SCHEMA:
        return None
    return dict(raw)


__all__ = [
    "REGISTRATION_METADATA_KEY",
    "REGISTRATION_METADATA_SCHEMA",
    "save_scan_registration",
    "scan_registration_record",
]
