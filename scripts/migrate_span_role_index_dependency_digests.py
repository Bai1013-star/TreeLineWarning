"""Validate and rebind source-shared role-index dependency digests in place.

The tile payloads are not rebuilt: they are first validated against the source
identity and their previous role contract, then only the role-index metadata
digest is atomically rebound to the narrower current contract.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from algorithms.spans.project_adapter import (  # noqa: E402
    PROJECT_SPAN_VEGETATION_ROLE_CONFIGURATION_DIGEST,
    wire_role_index_configuration_digest,
)
from algorithms.spans.role_tiles import (  # noqa: E402
    ROLE_TILE_CELL_SIZE_M,
    migrate_role_tile_configuration_digest,
)
from core.project import open_project_read_only  # noqa: E402


def _preparations(project_root: Path):
    root = project_root / "artifacts" / "project-runs" / "source-shared"
    yield from sorted(root.glob("*/preparation.json"))


def _cache_root(project_root: Path, reference: object) -> Path:
    if not isinstance(reference, str) or not reference:
        raise SystemExit("SOURCE_SHARED_ROLE_INDEX_REFERENCE_MISSING")
    root = (project_root / reference).resolve()
    try:
        root.relative_to(project_root.resolve())
    except ValueError as exc:
        raise SystemExit("SOURCE_SHARED_ROLE_INDEX_REFERENCE_ESCAPE") from exc
    return root


def _migrate(
    project_root: Path,
    prep: dict,
    *,
    role: str,
    source_identity: dict,
    expected_digest: str,
) -> str:
    root = _cache_root(project_root, prep.get(f"{role}_role_tile_index_reference"))
    metadata_path = root / "role-tile-cache.json"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        previous_digest = metadata["role_configuration_digest"]
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"ROLE_INDEX_METADATA_INVALID:{role}") from exc
    if role == "vegetation" and metadata.get("admission_semantics") != "HIGH_RECALL":
        raise SystemExit("ROLE_INDEX_VEGETATION_ADMISSION_NOT_HIGH_RECALL")
    if previous_digest == expected_digest:
        status = "ALREADY_CURRENT"
    else:
        migrate_role_tile_configuration_digest(
            root,
            role=role,
            source_identity=source_identity,
            previous_config_digest=previous_digest,
            configuration_digest=expected_digest,
            source_point_count=int(prep["source_point_count"]),
            expected_cell_size_m=ROLE_TILE_CELL_SIZE_M,
        )
        status = "MIGRATED_WITHOUT_CONTENT_REBUILD"
    print(f"{role}: {status}; root={root}; digest={expected_digest}")
    return status


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args(argv)
    project_root = args.project_root.resolve()
    with open_project_read_only(project_root) as session:
        source_identity = session.project.source_reference.binding.to_dict()
    preparations = tuple(_preparations(project_root))
    if not preparations:
        raise SystemExit("SOURCE_SHARED_PREPARATION_MISSING")
    for path in preparations:
        try:
            prep = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise SystemExit(f"SOURCE_SHARED_PREPARATION_INVALID:{path}") from exc
        print(f"preparation={path}")
        _migrate(
            project_root,
            prep,
            role="wire",
            source_identity=source_identity,
            expected_digest=wire_role_index_configuration_digest(),
        )
        _migrate(
            project_root,
            prep,
            role="vegetation",
            source_identity=source_identity,
            expected_digest=PROJECT_SPAN_VEGETATION_ROLE_CONFIGURATION_DIGEST,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
