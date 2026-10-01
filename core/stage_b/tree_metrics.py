"""Resolve canonical post-editor tree geometry for Stage-B exports.

The Survey product already has one reviewed full-resolution metric implementation
(``algorithms.manual_span.metrics``).  This adapter reuses it for Stage-B instead
of inventing a second DBH/crown formula.  Ground/HAG provenance is recovered only
from a validated source-shared preparation manifest; ambiguous legacy projects
fail closed and leave optional metric cells blank rather than guessing.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from algorithms.manual_span.config import default_manual_span_config
from algorithms.manual_span.metrics import TreeMetrics, compute_tree_metrics

from core.config.fast_baseline import canonical_digest

_SOURCE_PREPARATION_SCHEMA = "treeline-project-source-shared-preparation-v1"


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _resolve_artifact(project_root: Path, value: str) -> Path | None:
    root = project_root.resolve()
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = root / candidate
    try:
        candidate = candidate.resolve()
        candidate.relative_to(root)
    except (OSError, ValueError):
        return None
    return candidate if candidate.is_file() else None


def _valid_preparation_context(
    project_root: Path,
    preparation_path: Path,
    *,
    expected_point_count: int | None,
) -> tuple[np.ndarray, tuple[float, float, float]] | None:
    payload = _read_json(preparation_path)
    if payload is None or payload.get("schema") != _SOURCE_PREPARATION_SCHEMA:
        return None
    if payload.get("authority") != "FULL_RESOLUTION_RAW":
        return None
    if payload.get("tower_membership_used_for_vegetation") is not False:
        return None
    source_point_count = payload.get("source_point_count")
    if type(source_point_count) is not int or source_point_count <= 0:
        return None
    if expected_point_count is not None and source_point_count != expected_point_count:
        return None
    reference = payload.get("coarse_ground_reference")
    origin = payload.get("origin_xyz")
    if not isinstance(reference, str) or not reference:
        return None
    if not isinstance(origin, (list, tuple)) or len(origin) != 3:
        return None
    try:
        origin_xyz = tuple(float(value) for value in origin)
    except (TypeError, ValueError):
        return None
    if not np.all(np.isfinite(np.asarray(origin_xyz, dtype=np.float64))):
        return None
    ground_path = _resolve_artifact(project_root, reference)
    if ground_path is None:
        return None
    try:
        ground = np.asarray(
            np.load(ground_path, allow_pickle=False, mmap_mode="r"),
            dtype=np.float64,
        )
    except (OSError, ValueError):
        return None
    if ground.ndim != 2 or ground.shape[1] != 3 or len(ground) == 0:
        return None
    if not np.all(np.isfinite(ground)):
        return None
    return ground, origin_xyz


def _project_root_from_output(output_path: Path) -> Path | None:
    parent = output_path.resolve().parent
    if parent.name.lower() not in {"exports", "artifacts"}:
        return None
    root = parent.parent
    if (root / "stage-a-project.json").is_file() or (root / "project.json").is_file():
        return root
    return None


def _active_preparation_candidate(project_root: Path, result: Any) -> Path | None:
    """Resolve the exact source-shared preparation used by the active scan."""

    try:
        from core.stage_a import load_project
        from core.stage_b.bridge import ProjectScienceBridge

        project = load_project(project_root)
        bridge = ProjectScienceBridge(project, str(result.scan_id))
        source_digest = canonical_digest(dict(bridge.source_identity))
    except (OSError, ValueError, TypeError, KeyError):
        return None
    candidate = (
        project_root
        / "artifacts"
        / "project-runs"
        / "source-shared"
        / source_digest
        / "preparation.json"
    )
    return candidate if candidate.is_file() else None


def _preparation_candidates(
    project_root: Path,
    result: Any,
) -> list[Path]:
    diagnostics = getattr(result, "batch_diagnostics", {})
    reference = (
        diagnostics.get("source_preparation_reference")
        if isinstance(diagnostics, Mapping)
        else None
    )
    if isinstance(reference, str) and reference:
        path = _resolve_artifact(project_root, reference)
        return [] if path is None else [path]

    active = _active_preparation_candidate(project_root, result)
    if active is not None:
        return [active]

    # Legacy fallback: only an unambiguous validated preparation may be used.
    source_shared = project_root / "artifacts" / "project-runs" / "source-shared"
    if not source_shared.is_dir():
        return []
    return sorted(
        path for path in source_shared.glob("*/preparation.json") if path.is_file()
    )


def compute_stage_b_tree_metrics(
    result: Any,
    reconciled: Any,
    output_path: Path | str,
    *,
    project_root: Path | None = None,
) -> dict[int, TreeMetrics]:
    """Return canonical metrics keyed by final global tree id.

    The active source preparation is preferred and validated against the
    source-bound manifest.  Historical batches are supported only when exactly
    one candidate is valid for the recorded source point count.
    """

    output = Path(output_path)
    root = (
        Path(project_root).resolve()
        if project_root is not None
        else _project_root_from_output(output)
    )
    if root is None:
        return {}
    telemetry = getattr(result, "telemetry", None)
    raw_count = getattr(telemetry, "source_point_count", None)
    expected_point_count = raw_count if type(raw_count) is int and raw_count > 0 else None
    valid: list[tuple[np.ndarray, tuple[float, float, float]]] = []
    for candidate in _preparation_candidates(root, result):
        context = _valid_preparation_context(
            root,
            candidate,
            expected_point_count=expected_point_count,
        )
        if context is not None:
            valid.append(context)
    if len(valid) != 1:
        return {}
    ground, origin_xyz = valid[0]
    metrics = compute_tree_metrics(
        reconciled,
        ground,
        origin_xyz,
        default_manual_span_config(),
    )
    return {int(metric.global_tree_id): metric for metric in metrics}


__all__ = ["compute_stage_b_tree_metrics"]
