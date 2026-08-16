"""Prepare the real-source V5A2/V5A3 tower-review project + RAW tile cache.

This is the deliberate one-time preparation for the Owner validation workflow:

1. builds a digest-consistent ``SourceIdentity`` for the protected LAS;
2. creates the dedicated tower-review project under ``cache/`` (gitignored,
   never inside the protected authoritative run);
3. prepares the derived RAW spatial tile cache with one bounded streaming scan
   of the protected LAS (never modifying it);
4. runs the automatic detector against the compatible coarse ground/HAG
   representatives and persists the candidate set into the project.

No candidate is accepted and no ``T###`` is created here — the Owner performs
detection/review/confirmation in the running Giro3D page.

Usage::

    python scripts/prepare_tower_review_project.py \\
        --source input_las/2026-08-07_16没54-10_PointCloud_colorized.las \\
        --ground artifacts/one_click_runs/<run>/02_ground/classified_hag.las \\
        --project-root cache/tower_review_project \\
        --tile-cache cache/tower_raw_tiles
"""

from __future__ import annotations

import argparse
import sys
import time
import uuid
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from algorithms.poles.evidence import load_ground_evidence  # noqa: E402
from algorithms.towers.automatic import (  # noqa: E402
    TowerAutomaticDetectorConfig,
    detect_tower_candidates,
)
from algorithms.towers.raw_access import RawTileCache, RawTileCacheConfig  # noqa: E402
from core.config.fast_baseline import FastBaselineConfig  # noqa: E402
from core.contracts.frv import (  # noqa: E402
    FRV_RELOCATION_AUDIT_SCHEMA_VERSION,
    FRV_SOURCE_IDENTITY_SCHEMA_VERSION,
    FrvRelocationAudit,
    SourceIdentity,
    frv_digest,
)
from core.contracts.project import (  # noqa: E402
    PROJECT_SCHEMA_VERSION,
    PROJECT_SOURCE_REFERENCE_SCHEMA_VERSION,
    ProjectContract,
    ProjectSourceReference,
    SourceIdentityBinding,
    ViewerReference,
)
from core.project import create_project_writable  # noqa: E402
from core.project.session import open_project_read_only  # noqa: E402

DEFAULT_PROJECT_ROOT = _REPOSITORY_ROOT / "cache" / "tower_review_project"
DEFAULT_TILE_CACHE = _REPOSITORY_ROOT / "cache" / "tower_raw_tiles"
PROJECT_NAME = "P3A tower review (V5A2/V5A3)"


def _timestamp() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def build_real_source_identity(source_path: Path) -> SourceIdentity:
    """Build a digest-consistent SourceIdentity for the protected LAS.

    The content SHA-256 and header facts are read directly from the protected
    file (never assumed), so the resulting binding is authoritative.
    """
    import hashlib

    import laspy

    sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
    with laspy.open(source_path) as las:
        header = las.header
    min_xyz = [float(value) for value in header.mins]
    max_xyz = [float(value) for value in header.maxs]
    point_count = int(header.point_count)
    source_id = f"p3a_source_{sha256[:16]}"
    relocation_fields = {
        "audit_schema_version": FRV_RELOCATION_AUDIT_SCHEMA_VERSION,
        "path_style": "native",
        "recorded_path_raw": str(source_path),
        "resolved_path_raw": str(source_path),
        "recorded_path_lexical_normalized": str(source_path),
        "resolved_path_lexical_normalized": str(source_path),
        "resolved_final_path": str(source_path),
        "resolution_method": "exact_match",
        "case_comparison_mode": "case_sensitive",
        "platform_name": "windows",
        "run_id": "tower-review-prep",
        "audit_timestamp": _timestamp(),
        "content_identity_result": "matched",
        "warning_flags": 0,
    }
    relocation = FrvRelocationAudit(
        **{
            **relocation_fields,
            "audit_digest": frv_digest(relocation_fields),
        }
    )
    content = {
        "source_identity_schema_version": FRV_SOURCE_IDENTITY_SCHEMA_VERSION,
        "source_id": source_id,
        "source_file_sha256": sha256,
        "canonical_point_content_digest": None,
        "canonical_point_content_spec_digest": None,
        "canonical_point_content_spec_version": None,
        "source_point_count": point_count,
        "source_bounds_min_xyz": min_xyz,
        "source_bounds_max_xyz": max_xyz,
        "coordinate_frame": "SOURCE_LAS_ABSOLUTE",
        "units": "m",
        "origin_context": [0.0, 0.0, 0.0],
    }
    content_digest = frv_digest(content)
    record = {
        **content,
        "recorded_source_path": str(source_path),
        "resolved_access_path": str(source_path),
        "relocation_audit_audit_digest": relocation.audit_digest,
        "source_equivalence_audit_audit_digest": None,
        "source_content_identity_digest": content_digest,
    }
    return SourceIdentity(
        **{
            **content,
            "recorded_source_path": str(source_path),
            "resolved_access_path": str(source_path),
            "relocation_audit": relocation,
            "source_equivalence_audit": None,
            "source_content_identity_digest": content_digest,
            "source_identity_record_digest": frv_digest(record),
        }
    )


def build_project(source_identity: SourceIdentity) -> tuple[ProjectContract, dict]:
    binding = SourceIdentityBinding.from_source_identity(source_identity)
    project = ProjectContract(
        schema_version=PROJECT_SCHEMA_VERSION,
        project_id=str(uuid.uuid4()),
        project_name=PROJECT_NAME,
        created_at=_timestamp(),
        updated_at=_timestamp(),
        source_reference=ProjectSourceReference(
            schema_version=PROJECT_SOURCE_REFERENCE_SCHEMA_VERSION,
            source_identity=source_identity,
            coordinate_context=binding.coordinate_context,
        ),
        viewer=ViewerReference.not_built(),
        tower_registry_reference="towers/towers.json",
        span_registry_reference="spans/spans.json",
        run_index_reference="runs/run-index.json",
        next_tower_sequence=1,
        next_span_sequence=1,
    )
    metadata = {
        "policy": "REFERENCE_ORIGINAL_SOURCE_READ_ONLY",
        "current_resolved_source_path": source_identity.resolved_access_path,
        "file_size_bytes": Path(source_identity.resolved_access_path).stat().st_size,
        "source_file_sha256": source_identity.source_file_sha256,
        "point_count": source_identity.source_point_count,
        "bounds_min_xyz": list(source_identity.source_bounds_min_xyz),
        "bounds_max_xyz": list(source_identity.source_bounds_max_xyz),
        "header": {"format": "real-colorized-las"},
        "source_available_at_save": True,
        "recorded_at": _timestamp(),
    }
    return project, metadata


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True,
                        help="protected RAW source LAS")
    parser.add_argument("--ground", type=Path, required=True,
                        help="compatible coarse ground/HAG evidence LAS for detection")
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--tile-cache", type=Path, default=DEFAULT_TILE_CACHE)
    parser.add_argument("--tile-cell-m", type=float, default=40.0)
    parser.add_argument("--force-tile-cache", action="store_true",
                        help="rebuild the tile cache even if one exists")
    args = parser.parse_args(argv)

    source_path = args.source.resolve()
    ground_path = args.ground.resolve()
    if not source_path.is_file():
        print(f"[prepare] ERROR: source LAS missing: {source_path}", file=sys.stderr)
        return 2
    if not ground_path.is_file():
        print(f"[prepare] ERROR: ground evidence LAS missing: {ground_path}", file=sys.stderr)
        return 2

    print(f"[prepare] protected source: {source_path}")
    print("[prepare] point count expected: 105055350")
    started = time.perf_counter()
    identity = build_real_source_identity(source_path)
    print(f"[prepare] source identity built in {time.perf_counter() - started:.1f}s")
    print(f"[prepare]   content digest: {identity.source_content_identity_digest}")

    project_root = args.project_root.resolve()
    if (project_root / "project.json").exists():
        print(f"[prepare] project already exists: {project_root} (reusing)")
    else:
        project, metadata = build_project(identity)
        with create_project_writable(project_root, project, source_access_metadata=metadata) as _:
            print(f"[prepare] project created: {project_root}")
            print(f"[prepare]   project_id: {project.project_id}")

    # ------------------------------------------------------------------ tiles
    tile_config = RawTileCacheConfig(cell_size_m=args.tile_cell_m)
    tile_cache = args.tile_cache.resolve()
    binding = SourceIdentityBinding.from_source_identity(identity)
    if tile_cache.exists() and not args.force_tile_cache:
        print(f"[prepare] tile cache exists: {tile_cache} (reusing; --force-tile-cache to rebuild)")
        cache = RawTileCache.load(tile_cache, source_binding=binding, config=tile_config)
    else:
        print(f"[prepare] preparing RAW tile cache (streaming {identity.source_point_count} pts)…")
        started = time.perf_counter()
        cache = RawTileCache.prepare(
            source_path, tile_cache, source_binding=binding, config=tile_config
        )
        print(f"[prepare]   cache ready in {cache.preparation_runtime_seconds:.1f}s")
        print(f"[prepare]   cache size: {cache.byte_size} bytes across {cache.tile_count} tiles")
        print(f"[prepare]   scanned points: {cache.point_count}")

    # --------------------------------------------------------------- detection
    print(f"[prepare] running automatic detector on {ground_path.name}…")
    started = time.perf_counter()
    evidence = load_ground_evidence(ground_path, FastBaselineConfig(), origin_xyz=(0.0, 0.0, 0.0))
    print(f"[prepare]   ground evidence: {evidence.count} representatives "
          f"({evidence.non_ground_count} non-ground) in {evidence.feature_runtime_seconds:.1f}s")
    with open_project_read_only(project_root) as session:
        existing = {cs.candidate_set_id for cs in session.candidate_store.candidate_sets}
    from core.project import open_project_writable

    if "auto-owner-review-001" in existing:
        print("[prepare] candidate set auto-owner-review-001 already persisted (skipping detect)")
    else:
        with open_project_writable(project_root) as session:
            binding = session.project.source_reference.binding
            candidate_set = detect_tower_candidates(
                evidence,
                FastBaselineConfig(),
                source_binding=binding,
                coordinate_context=binding.coordinate_context,
                project_id=session.project.project_id,
                candidate_set_id="auto-owner-review-001",
                generated_at=_timestamp(),
                detector_config=TowerAutomaticDetectorConfig(),
            )
            session.persist_tower_candidate_set(candidate_set)
            detector_runtime = time.perf_counter() - started
            print(f"[prepare] detector runtime: {detector_runtime:.1f}s")
            print(f"[prepare] candidate set: {candidate_set.candidate_set_id}")
            print(f"[prepare]   digest: {candidate_set.digest}")
            print(f"[prepare]   candidates: {candidate_set.candidate_count}")
            for candidate in candidate_set.candidates:
                x, y, z = candidate.candidate_xyz_absolute_m
                print(
                    f"[prepare]     {candidate.candidate_id} "
                    f"xyz=({x:.2f}, {y:.2f}, {z:.2f}) conf={candidate.confidence:.3f}"
                )
    return 0


if __name__ == "__main__":
    sys.exit(main())
