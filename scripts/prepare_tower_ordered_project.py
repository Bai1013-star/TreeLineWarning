"""Prepare the CLEAN V5A3 ordered-manual tower-selection project.

The ordered workflow is the PRIMARY tower-selection path: the Owner clicks real
towers in corridor order and confirms them one by one (T001, T002, T003, ...).
This one-time preparation builds a clean project under ``cache/tower_ordered_project``
bound to the SAME protected source LAS:

- 0 confirmed towers
- ``next_tower_sequence`` = 1
- no candidate / review / confirmation history (the ordered selections are the
  only history, added live by the Owner in the Giro3D page)

It deliberately does NOT run automatic detection (the automatic detector stays
available as OPTIONAL ASSISTANCE through the viewer's collapsible Automatic
Suggestions section).  It reuses the existing derived RAW tile cache
(``cache/tower_raw_tiles``) after strict source-identity validation and never
rebuilds it.

The historical experimental project ``cache/tower_review_project`` (V5A2/V5A3
review activity with T### confirmations) is preserved untouched.

The protected source path is resolved safely (never a hard-coded non-ASCII
literal): an explicit ``--source``, else the canonical source-bound identity
path from the historical review project, else a single-file scan of
``input_las/``.  The launcher therefore needs no Chinese filename argument.

Usage::

    python scripts/prepare_tower_ordered_project.py
    # (optional --source <path> / --project-root / --tile-cache overrides)
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from core.project import create_project_writable  # noqa: E402

from scripts.prepare_tower_review_project import (  # noqa: E402
    build_project,
    build_real_source_identity,
)

DEFAULT_PROJECT_ROOT = _REPOSITORY_ROOT / "cache" / "tower_ordered_project"
DEFAULT_TILE_CACHE = _REPOSITORY_ROOT / "cache" / "tower_raw_tiles"
DEFAULT_REVIEW_PROJECT = _REPOSITORY_ROOT / "cache" / "tower_review_project"
PROJECT_NAME = "P3A ordered tower selection (V5A3)"


def load_canonical_source_identity(review_project_root: Path):
    """Load the persisted SourceIdentity from the historical review project.

    The existing RAW tile cache is bound to the review project's exact
    ``SourceIdentity`` (including its record digest).  ``build_real_source_identity``
    embeds a wall-clock timestamp in the relocation audit, so a freshly rebuilt
    identity would never match that binding.  The ordered project MUST reuse the
    canonical persisted identity so it can reuse the existing tile cache.
    """
    from core.project import open_project_read_only

    try:
        with open_project_read_only(review_project_root) as session:
            return session.project.source_reference.source_identity
    except Exception as exc:  # pragma: no cover - historical project absent
        print(
            f"[ordered-prep] WARNING: cannot read canonical identity from {review_project_root}: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return None


def resolve_source_path(
    source_arg: Path | None, identity=None
) -> Path | None:
    """Resolve the protected source LAS path without embedding a non-ASCII literal.

    The protected source filename contains a non-ASCII character and must never
    be duplicated in a fragile cmd.exe batch file.  Resolution order:

    1. an explicit ``--source`` argument (Unicode-safe in Python);
    2. the canonical persisted ``SourceIdentity.resolved_access_path`` (already
       source-bound through the historical review project);
    3. a scan of ``input_las/`` (fail closed unless exactly one LAS/LAZ exists).
    """
    if source_arg is not None:
        path = Path(source_arg).resolve()
        if path.is_file():
            return path
        print(f"[ordered-prep] ERROR: --source not found: {path}", file=sys.stderr)
        return None
    if identity is not None and identity.resolved_access_path:
        path = Path(identity.resolved_access_path)
        if path.is_file():
            return path
        print(
            f"[ordered-prep] ERROR: canonical source path is missing: {path}",
            file=sys.stderr,
        )
        return None
    candidates = sorted(_REPOSITORY_ROOT.joinpath("input_las").glob("*.las")) + sorted(
        _REPOSITORY_ROOT.joinpath("input_las").glob("*.laz")
    )
    if len(candidates) == 1:
        return candidates[0].resolve()
    if len(candidates) > 1:
        print(
            "[ordered-prep] ERROR: multiple LAS sources in input_las/; specify --source",
            file=sys.stderr,
        )
        return None
    print("[ordered-prep] ERROR: no LAS source found in input_las/", file=sys.stderr)
    return None


def _default_group_registry(project_id: str):
    """A new ordered project starts with the default G001 / 区域 1."""
    from datetime import UTC, datetime

    from core.contracts.tower_groups import (
        TOWER_GROUP_REGISTRY_SCHEMA_VERSION,
        TOWER_GROUP_SCHEMA_VERSION,
        TowerGroup,
        TowerGroupRegistry,
    )

    timestamp = datetime.now(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")
    group = TowerGroup(
        schema_version=TOWER_GROUP_SCHEMA_VERSION,
        group_id="G001",
        display_name="区域 1",
        ordered_tower_ids=(),
        created_at=timestamp,
    )
    return TowerGroupRegistry(
        schema_version=TOWER_GROUP_REGISTRY_SCHEMA_VERSION,
        generation_id="initial",
        project_id=project_id,
        next_group_sequence=2,
        groups=(group,),
    )


def _bootstrap_legacy_groups(project_root: Path) -> int:
    """Resolve a legacy group-less ordered project to G001 via a successor generation."""
    from core.project import open_project_writable

    with open_project_writable(project_root) as session:
        if not session.tower_group_registry.bootstrap_pending:
            print("[ordered-prep]   group registry already present (no bootstrap needed)")
            return 0
        registry = session.ensure_ordered_group()
        for group in registry.groups:
            print(
                f"[ordered-prep]   bootstrapped {group.group_id} / {group.display_name}: "
                f"{group.ordered_tower_ids}"
            )
        return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=None,
                        help="protected RAW source LAS (optional; resolved safely if omitted)")
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--tile-cache", type=Path, default=DEFAULT_TILE_CACHE)
    parser.add_argument("--review-project", type=Path, default=DEFAULT_REVIEW_PROJECT,
                        help="historical review project whose SourceIdentity is reused")
    args = parser.parse_args(argv)

    identity = load_canonical_source_identity(args.review_project.resolve())
    source_path = resolve_source_path(args.source, identity)
    if source_path is None:
        return 2
    if identity is None:
        started = time.perf_counter()
        identity = build_real_source_identity(source_path)
        print(f"[ordered-prep] source identity built in {time.perf_counter() - started:.1f}s")
    else:
        print("[ordered-prep] reused canonical SourceIdentity from the historical review project")
    print(f"[ordered-prep] protected source: {source_path}")
    print(f"[ordered-prep]   content digest: {identity.source_content_identity_digest}")
    print(f"[ordered-prep]   record digest: {identity.source_identity_record_digest}")
    print(f"[ordered-prep]   point count: {identity.source_point_count}")

    project_root = args.project_root.resolve()
    if (project_root / "project.json").exists():
        print(f"[ordered-prep] project already exists: {project_root} (reusing)")
        if _bootstrap_legacy_groups(project_root) != 0:
            return 2
    else:
        project, metadata = build_project(identity)
        project = replace(project, project_name=PROJECT_NAME)
        with create_project_writable(
            project_root,
            project,
            source_access_metadata=metadata,
            tower_group_registry=_default_group_registry(project.project_id),
        ):
            print(f"[ordered-prep] CLEAN ordered project created: {project_root}")
            print(f"[ordered-prep]   project_id: {project.project_id}")
            print("[ordered-prep]   confirmed towers: 0")
            print(f"[ordered-prep]   next_tower_sequence: {project.next_tower_sequence}")
            print("[ordered-prep]   default group: G001 / 区域 1 (next_group_sequence=2)")

    from algorithms.towers.raw_access import RawTileCache, RawTileCacheConfig
    from core.contracts.project import SourceIdentityBinding

    binding = SourceIdentityBinding.from_source_identity(identity)
    tile_cache = args.tile_cache.resolve()
    if not tile_cache.exists():
        print(f"[ordered-prep] ERROR: RAW tile cache missing: {tile_cache}", file=sys.stderr)
        return 2
    try:
        cache = RawTileCache.load(
            tile_cache, source_binding=binding, config=RawTileCacheConfig()
        )
    except Exception as exc:  # closed failure: cache is bound to another identity
        print(
            f"[ordered-prep] ERROR: RAW tile cache is not bound to this source identity: {exc}",
            file=sys.stderr,
        )
        return 2
    print(f"[ordered-prep] RAW tile cache REUSED: {tile_cache}")
    print(f"[ordered-prep]   cache_id: {cache.cache_id}")
    print(f"[ordered-prep]   tiles: {cache.tile_count} | points: {cache.point_count}")
    print(f"[ordered-prep]   byte_size: {cache.byte_size}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
