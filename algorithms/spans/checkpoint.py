"""Digest-keyed stage checkpoints for the pole-first span-based baseline.

A stage checkpoint is valid only when every provenance field matches the
current run: source SHA-256, source identity, config digest, schema version
and the digest of the upstream stage that produced its inputs.  No stale or
partial artifact is ever reused.  This lets a downstream failure resume without
re-running validated upstream stages on the 105M-point source.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from core.config.fast_baseline import canonical_digest

CHECKPOINT_SCHEMA = "fast-baseline-checkpoint-v1"
_NON_CERTIFIED_MARKER = "NON_CERTIFIED_RESEARCH_DEMONSTRATION"


def checkpoint_digest(payload: dict[str, Any]) -> str:
    """Canonical digest of a checkpoint manifest (excludes output_digest)."""

    fields = dict(payload)
    fields.pop("output_digest", None)
    fields.pop("completed", None)
    return canonical_digest(fields)


def checkpoint_path(output_root: Path | str, stage: str) -> Path:
    return Path(output_root) / f"{stage}.checkpoint.json"


def write_checkpoint(
    output_root: Path | str,
    stage: str,
    *,
    source_sha256: str,
    source_id: str,
    config_digest: str,
    upstream_digest: str,
    summary: dict[str, Any],
    output_digest: str,
) -> Path:
    """Persist a completed stage checkpoint with full provenance."""

    payload = {
        "schema_version": CHECKPOINT_SCHEMA,
        "non_certified_marker": _NON_CERTIFIED_MARKER,
        "stage": stage,
        "source_sha256": source_sha256,
        "source_id": source_id,
        "config_digest": config_digest,
        "upstream_digest": upstream_digest,
        "summary": summary,
        "output_digest": output_digest,
        "completed": True,
    }
    path = checkpoint_path(output_root, stage)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2), "utf-8"
    )
    temporary.replace(path)
    return path


def read_checkpoint(
    output_root: Path | str,
    stage: str,
    *,
    source_sha256: str,
    source_id: str,
    config_digest: str,
    upstream_digest: str,
) -> tuple[dict[str, Any], str] | None:
    """Return ``(summary, output_digest)`` when a valid checkpoint exists.

    The checkpoint is valid only when every provenance field matches the
    current run exactly; otherwise ``None`` is returned so the stage reruns.
    """

    path = checkpoint_path(output_root, stage)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text("utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    if (
        payload.get("schema_version") != CHECKPOINT_SCHEMA
        or payload.get("non_certified_marker") != _NON_CERTIFIED_MARKER
        or payload.get("stage") != stage
        or payload.get("completed") is not True
        or payload.get("source_sha256") != source_sha256
        or payload.get("source_id") != source_id
        or payload.get("config_digest") != config_digest
        or payload.get("upstream_digest") != upstream_digest
    ):
        return None
    summary = payload.get("summary")
    output_digest = payload.get("output_digest")
    if not isinstance(summary, dict) or not isinstance(output_digest, str):
        return None
    return summary, output_digest


def load_span_checkpoint(
    span_output_root: Path | str,
    stage: str,
    *,
    source_sha256: str,
    source_id: str,
    config_digest: str,
    upstream_digest: str,
) -> dict[str, Any] | None:
    """Per-span checkpoint variant keyed by span artifact sub-directory."""

    found = read_checkpoint(
        span_output_root,
        stage,
        source_sha256=source_sha256,
        source_id=source_id,
        config_digest=config_digest,
        upstream_digest=upstream_digest,
    )
    return None if found is None else found[0]


def span_checkpoint_path(span_output_root: Path | str, stage: str) -> Path:
    return checkpoint_path(span_output_root, stage)


__all__ = [
    "CHECKPOINT_SCHEMA",
    "checkpoint_digest",
    "checkpoint_path",
    "write_checkpoint",
    "read_checkpoint",
    "load_span_checkpoint",
    "span_checkpoint_path",
]
