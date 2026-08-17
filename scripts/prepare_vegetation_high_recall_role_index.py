"""Rebuild the source-shared vegetation role index with HIGH-RECALL admission.

The OLD vegetation role index excluded the ENTIRE broad wire-role search-corridor
source set (millions of points), which systematically dropped conductor-adjacent
vegetation before the tree pipeline.  This script builds the corrected HIGH-RECALL
vegetation role index (no broad wire exclusion) side-by-side under
``vegetation-role-tile-index-5m-hirecall`` with a NEW role-configuration digest
so stale old-semantics caches fail closed.  Only exact final confirmed conductor
member source indices are subtracted later, per Span, in the canonical executor.

Usage::

    python scripts/prepare_vegetation_high_recall_role_index.py \\
        --project-root cache/tower_ordered_project
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from algorithms.spans.project_adapter import (  # noqa: E402
    PROJECT_SPAN_VEGETATION_ROLE_CONFIGURATION_DIGEST,
)
from algorithms.spans.role_tiles import (  # noqa: E402
    RoleSpecificTileCache,
)
from algorithms.vegetation.coarse_to_fine import RawRegion  # noqa: E402

NEW_SUFFIX = "-hirecall"


def _find_preparation(project_root: Path):
    base = project_root / "artifacts" / "project-runs" / "source-shared"
    if not base.is_dir():
        raise SystemExit(f"no source-shared preparation under {base}")
    yield from base.glob("*/preparation.json")


def _open_vegetation_region(project_root: Path, prep_path: Path):
    prep = json.loads(prep_path.read_text(encoding="utf-8"))
    region_ref = prep.get("vegetation_region_reference")
    if region_ref:
        root = project_root / region_ref
    else:
        root = prep_path.parent / "vegetation-region"
    sidecar_path = root / "region_sidecar.json"
    if not sidecar_path.is_file():
        raise SystemExit(f"vegetation region sidecar missing: {sidecar_path}")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    region = RawRegion(
        str(root / "xyz_local.bin"),
        str(root / "original_index.bin"),
        str(root / "classification.bin"),
        int(sidecar["count"]),
        tuple(float(v) for v in sidecar["bounds_min_xyz"]),
        tuple(float(v) for v in sidecar["bounds_max_xyz"]),
        tuple(float(v) for v in sidecar["origin_xyz"]),
    )
    return prep, region


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument(
        "--source-point-count",
        type=int,
        help="optional authoritative source count; defaults to preparation metadata",
    )
    args = parser.parse_args(argv)
    project_root = args.project_root.resolve()
    from core.project import open_project_read_only

    with open_project_read_only(project_root) as session:
        source_identity = session.project.source_reference.binding.to_dict()
    print(f"[veg-hirecall] source binding digest: {len(str(source_identity))} fields")
    for prep_path in _find_preparation(project_root):
        prep, region = _open_vegetation_region(project_root, prep_path)
        source_point_count = args.source_point_count
        if source_point_count is None:
            value = prep.get("source_point_count")
            if type(value) is not int or value <= 0:
                raise SystemExit(
                    "source_point_count missing from preparation; pass --source-point-count"
                )
            source_point_count = value
        old_ref = prep.get("vegetation_role_tile_index_reference")
        new_dir_name = "vegetation-role-tile-index-5m" + NEW_SUFFIX
        new_dir = prep_path.parent / new_dir_name
        print(f"[veg-hirecall] preparation: {prep_path.parent.name}")
        print(f"[veg-hirecall] old vegetation role ref: {old_ref}")
        print(f"[veg-hirecall] vegetation region points: {region.count}")
        if new_dir.is_dir() and any(new_dir.iterdir()):
            print(f"[veg-hirecall] new cache exists: {new_dir} (reusing)")
        else:
            started = time.perf_counter()
            cache = RoleSpecificTileCache.prepare_from_raw_region(
                region,
                new_dir,
                role="vegetation",
                source_identity=source_identity,
                config_digest=PROJECT_SPAN_VEGETATION_ROLE_CONFIGURATION_DIGEST,
                source_point_count=source_point_count,
                cell_size_m=5.0,
                excluded_source_indices=None,
            )
            print(
                f"[veg-hirecall] new HIGH-RECALL cache built in "
                f"{time.perf_counter() - started:.1f}s"
            )
            print(
                f"[veg-hirecall]   points={cache.metadata['totals']['point_count']} "
                f"tiles={cache.metadata['totals']['tile_count']} "
                f"bytes={cache.metadata['totals']['byte_size']}"
            )
            print(
                f"[veg-hirecall]   admission_semantics="
                f"{cache.metadata.get('admission_semantics')}"
            )
        new_ref = new_dir.relative_to(project_root).as_posix()
        if prep.get("vegetation_role_tile_index_reference") != new_ref:
            prep["vegetation_role_tile_index_reference"] = new_ref
            prep_path.write_text(
                json.dumps(prep, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )
            print(f"[veg-hirecall] preparation updated -> {new_ref}")
        else:
            print(f"[veg-hirecall] preparation already references {new_ref}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
