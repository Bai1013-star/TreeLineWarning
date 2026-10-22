"""Source/scan-level Ground/HAG preparation and provenance.

The producer is deliberately the existing PDAL SMRF + HAG-NN route.  This
module only makes its reusable output explicit and digest-bound so a fresh
Span cannot reach the canonical executor with a null ground reference.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from core.config.fast_baseline import canonical_digest

if TYPE_CHECKING:
    from algorithms.spans.project_adapter import SpanAnalysisInput

GROUND_PREPARATION_SCHEMA = "treeline-stage-b-ground-preparation-v1"
GROUND_BACKEND = "algorithms.vegetation.fast_research.run_fast_ground_hag"


class GroundPreparationError(RuntimeError):
    """Fail-closed Ground/HAG preparation or provenance error."""


@dataclass(frozen=True, slots=True)
class GroundPreparation:
    source_binding_digest: str
    algorithm_config_digest: str
    ground_hag_reference: str
    coarse_ground_reference: str
    origin_xyz: tuple[float, float, float]
    source_identity: Mapping[str, Any]
    cache_hit: bool = False
    source_full_pass_count: int = 0
    region_digest: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "source_binding_digest",
            "algorithm_config_digest",
        ):
            value = getattr(self, name)
            if type(value) is not str or len(value) != 64:
                raise ValueError(f"{name} must be a SHA-256 digest")
        for name in ("ground_hag_reference", "coarse_ground_reference"):
            value = getattr(self, name)
            if type(value) is not str or not value.strip():
                raise ValueError(f"{name} must be a non-empty artifact reference")
        if len(self.origin_xyz) != 3 or not all(np.isfinite(self.origin_xyz)):
            raise ValueError("origin_xyz must contain three finite values")
        if type(self.cache_hit) is not bool or type(self.source_full_pass_count) is not int:
            raise ValueError("invalid GroundPreparation telemetry")
        if self.source_full_pass_count < 0:
            raise ValueError("source_full_pass_count must be non-negative")
        if self.region_digest is not None and (
            type(self.region_digest) is not str or len(self.region_digest) != 64
        ):
            raise ValueError("region_digest must be a SHA-256 digest or None")
        object.__setattr__(self, "source_identity", dict(self.source_identity))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": GROUND_PREPARATION_SCHEMA,
            "source_binding_digest": self.source_binding_digest,
            "algorithm_config_digest": self.algorithm_config_digest,
            "ground_hag_reference": self.ground_hag_reference,
            "coarse_ground_reference": self.coarse_ground_reference,
            "origin_xyz": list(self.origin_xyz),
            "source_identity": dict(self.source_identity),
            "backend": GROUND_BACKEND,
            "cache_hit": self.cache_hit,
            "source_full_pass_count": self.source_full_pass_count,
            "region_digest": self.region_digest,
        }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), ensure_ascii=False, sort_keys=True, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def _source_path(source_identity: Mapping[str, Any]) -> Path:
    value = (
        source_identity.get("resolved_access_path")
        or source_identity.get("point_cloud_source")
        or source_identity.get("path")
        or source_identity.get("source_path")
    )
    if not isinstance(value, str) or not value:
        raise GroundPreparationError("GROUND_SOURCE_PATH_MISSING")
    return Path(value).expanduser().resolve()


def _validate_existing(
    project_root: Path,
    manifest_path: Path,
    expected_source_digest: str,
    expected_config_digest: str,
) -> GroundPreparation | None:
    if not manifest_path.is_file():
        return None
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return None
        if payload.get("schema") != GROUND_PREPARATION_SCHEMA:
            return None
        if payload.get("source_binding_digest") != expected_source_digest:
            return None
        if payload.get("algorithm_config_digest") != expected_config_digest:
            return None
        coarse = project_root / Path(str(payload["coarse_ground_reference"]))
        ground = project_root / Path(str(payload["ground_hag_reference"]))
        if not coarse.is_file() or not ground.is_file():
            return None
        array = np.load(coarse, allow_pickle=False, mmap_mode="r")
        if array.ndim != 2 or array.shape[1] != 3 or len(array) == 0:
            return None
        origin = tuple(float(item) for item in payload["origin_xyz"])
        return GroundPreparation(
            expected_source_digest,
            expected_config_digest,
            str(payload["ground_hag_reference"]),
            str(payload["coarse_ground_reference"]),
            origin,
            dict(payload.get("source_identity") or {}),
            cache_hit=True,
            source_full_pass_count=0,
            region_digest=payload.get("region_digest"),
        )
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return None


def ensure_ground_preparation(
    project_root: Path | str,
    source_identity: Mapping[str, Any],
    *,
    configuration: Mapping[str, Any] | None = None,
    producer: Callable[[Path, Path], Mapping[str, Any]] | None = None,
    span_inputs: Sequence[SpanAnalysisInput] | None = None,
) -> GroundPreparation:
    """Load or create one source-bound Ground/HAG preparation.

    ``producer`` is an explicit test/fixture seam.  Production defaults to
    ``run_fast_ground_hag`` and ``load_coarse_ground_representatives`` from the
    validated Ground/HAG route.
    """

    root = Path(project_root).expanduser().resolve()
    identity = dict(source_identity)
    source_digest = canonical_digest(identity)
    region_digest: str | None = None
    if span_inputs:
        from core.stage_b.ground_workset import ground_region_plan

        region_digest, _regions = ground_region_plan(span_inputs)
    config_payload = {
        "backend": GROUND_BACKEND,
        "configuration": dict(configuration or {}),
        "ground_region_digest": region_digest,
    }
    config_digest = canonical_digest(config_payload)
    shared_root = root / "artifacts" / "project-runs" / "source-shared" / source_digest
    manifest_path = shared_root / "ground-preparation.json"
    existing = _validate_existing(root, manifest_path, source_digest, config_digest)
    if existing is not None:
        return existing

    source = _source_path(identity)
    if not source.is_file():
        raise GroundPreparationError(f"GROUND_SOURCE_MISSING:{source}")
    shared_root.mkdir(parents=True, exist_ok=True)
    ground_root = shared_root / "ground"
    if producer is None:
        from algorithms.vegetation.coarse_to_fine import load_coarse_ground_representatives
        from algorithms.vegetation.fast_research import run_fast_ground_hag
        ground_input = source
        source_full_pass_count = 1
        if span_inputs:
            from core.stage_b.ground_workset import ensure_ground_workset

            workset_status, workset = ensure_ground_workset(
                source,
                # Keep internal cache paths short enough for Windows hosts
                # where long-path policy is not enabled.  Digests remain in
                # manifests; directory names are implementation details.
                shared_root / "gw",
                identity,
                span_inputs,
            )
            ground_input = Path(workset.workset_las_path)
            source_full_pass_count = (
                0 if workset_status == "SKIPPED_FROM_CACHE" else workset.source_full_pass_count
            )
            ground_root = shared_root / "gr" / workset.region_digest[:20]

        _status, result = run_fast_ground_hag(ground_input, ground_root)
        coarse_xyz = load_coarse_ground_representatives(result)
        ground_reference = Path(result.output_las_path)
        origin = tuple(
            float(item)
            for item in identity.get("origin_context")
            or identity.get("source_bounds_min_xyz")
            or (0.0, 0.0, 0.0)
        )
    else:
        produced = dict(producer(source, ground_root))
        coarse_value = produced.get("coarse_ground_xyz")
        if coarse_value is None:
            coarse_value = produced.get("coarse_ground_reference")
        if isinstance(coarse_value, (str, Path)):
            coarse_xyz = np.asarray(np.load(coarse_value, allow_pickle=False), dtype=np.float64)
        else:
            coarse_xyz = np.asarray(coarse_value, dtype=np.float64)
        ground_reference = Path(
            str(produced.get("ground_hag_reference", ground_root / "ground_hag_result.json"))
        )
        source_full_pass_count = int(produced.get("source_full_pass_count", 1))
        origin = tuple(float(item) for item in produced.get("origin_xyz", (0.0, 0.0, 0.0)))
    if coarse_xyz.ndim != 2 or coarse_xyz.shape[1] != 3 or len(coarse_xyz) == 0:
        raise GroundPreparationError("GROUND_REFERENCE_INVALID")
    coarse_path = shared_root / "coarse_ground.npy"
    temporary = coarse_path.with_name(coarse_path.name + ".tmp.npy")
    np.save(temporary, coarse_xyz.astype(np.float64, copy=False))
    temporary.replace(coarse_path)
    try:
        ground_relative = ground_reference.resolve().relative_to(root).as_posix()
    except ValueError as exc:
        raise GroundPreparationError("GROUND_ARTIFACT_OUTSIDE_PROJECT") from exc
    coarse_relative = coarse_path.relative_to(root).as_posix()
    preparation = GroundPreparation(
        source_digest,
        config_digest,
        ground_relative,
        coarse_relative,
        origin,
        identity,
        cache_hit=False,
        source_full_pass_count=source_full_pass_count,
        region_digest=region_digest,
    )
    _write_json(manifest_path, preparation.to_dict())
    return preparation


__all__ = [
    "GROUND_BACKEND",
    "GROUND_PREPARATION_SCHEMA",
    "GroundPreparation",
    "GroundPreparationError",
    "ensure_ground_preparation",
]
