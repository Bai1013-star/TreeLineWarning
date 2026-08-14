"""Convert the P3A ROI LAS to a Cloud-Optimized Point Cloud (COPC) for the
Giro3D viewer Candidate B POC.

Reads the same dense real-RGB ROI the Potree Candidate A uses
(``cache/p3a_potree/input/p3a_roi.las``, 982,504 points) and writes a valid
COPC file under the ignored viewer-cache root
(``cache/p3a_giro3d/dataset/p3a_roi.copc.laz``).

COPC writing is done with ``copclib`` (the Python bindings of the BSD-licensed
copc-lib reference library; LAZ chunk compression is provided by the bundled
lazperf).  The octree is built as a capacity-based spatial decimation:

- each node keeps a Morton-order (spatial z-order) decimated subset of its
  voxel so coarse levels show a well-spread sample of the whole cloud;
- the remaining points are distributed to the node's children by the COPC voxel
  formula ``Box(key, header)`` from copc-lib (axis-aligned cells of size
  ``header.Span() / 2^depth`` anchored at ``header.min``);
- every point lands in exactly one node, giving natural LOD.

This is a derived, reproducible viewer cache.  The protected source and the
protected authoritative run are never touched.

Usage (run with the giro3d tooling venv python)::

    cache/p3a_giro3d/python/giro3d-venv/Scripts/python.exe \\
        scripts/prepare_p3a_giro3d_copc.py [--validate]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import copclib
import laspy
import numpy as np

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = _REPOSITORY_ROOT / "cache" / "p3a_potree" / "input" / "p3a_roi.las"
DEFAULT_OUTPUT_DIR = _REPOSITORY_ROOT / "cache" / "p3a_giro3d" / "dataset"
DEFAULT_OUTPUT = DEFAULT_OUTPUT_DIR / "p3a_roi.copc.laz"
DEFAULT_PROVENANCE = _REPOSITORY_ROOT / "cache" / "p3a_giro3d" / "copc-provenance.json"

# Point format 7 = RGB (same as the source ROI LAS).
POINT_FORMAT_ID = 7

# Per-node point capacity.  Coarse nodes keep ~this many decimated points; the
# remainder flows to children, so the octree depth emerges from the data.
DEFAULT_MAX_POINTS_PER_NODE = 100000

# Hard ceiling so pathological dense regions cannot explode the octree.
MAX_DEPTH = 10

TOOL_NAME = "copclib (copc-lib Python bindings + bundled lazperf)"
TOOL_VERSION = getattr(copclib, "__version__", "2.6.3")
TOOL_LICENSE = "BSD-3-Clause (copc-lib); LASzip-compatible LAZ via lazperf"


def _morton_key(xyz: np.ndarray) -> np.ndarray:
    """Spatial z-order (Morton) codes for a Nx3 integer-coordinate array.

    Spreads each 21-bit coordinate and interleaves x/y/z bits so that points
    close in 3D space are close in sort order (used for a well-spread
    decimation within each voxel).
    """
    x = xyz[:, 0].astype(np.int64)
    y = xyz[:, 1].astype(np.int64)
    z = xyz[:, 2].astype(np.int64)

    # Spread each 21-bit coordinate into 63 bits (3 bits per input bit) so the
    # three axes interleave without overlapping.  All masks fit in int64.
    def spread(v: np.ndarray) -> np.ndarray:
        v = v & 0x1FFFFF
        v = (v | (v << 32)) & 0x1F00000000FFFF
        v = (v | (v << 16)) & 0x1F0000FF0000FF
        v = (v | (v << 8)) & 0x100F00F00F00F00F
        v = (v | (v << 4)) & 0x10C30C30C30C30C3
        v = (v | (v << 2)) & 0x1249249249249249
        return v

    return spread(x) | (spread(y) << 1) | (spread(z) << 2)


def _child_key(d: int, ix: int, iy: int, iz: int, mn: np.ndarray, span: float,
               coord: np.ndarray) -> tuple[int, int, int, int]:
    """COPC voxel formula: cell index at depth ``d`` of a coordinate.

    Mirrors copc-lib's ``Box(key, header)``: cell size is ``span / 2^d`` anchored
    at ``header.min``; indices are clamped into ``[0, 2^d - 1]``.
    """
    step = span / (1 << d)
    x = min(int((coord[0] - mn[0]) / step), (1 << d) - 1)
    y = min(int((coord[1] - mn[1]) / step), (1 << d) - 1)
    z = min(int((coord[2] - mn[2]) / step), (1 << d) - 1)
    return (d, x, y, z)


def build_octree(
    xyz: np.ndarray, mn: np.ndarray, span: float, max_points: int
) -> dict[tuple[int, int, int, int], np.ndarray]:
    """Return ``{voxel_key: index_array}`` with every point in exactly one node.

    Each node keeps a Morton-decimated subset (``max_points``) of its voxel and
    passes the remainder down to the child whose voxel contains it.
    """
    nodes: dict[tuple[int, int, int, int], np.ndarray] = {}

    # Morton codes need integer voxel coordinates in [0, 2^21).  Normalize the
    # real coords into the cube then scale to the Morton range.
    norm = (xyz - mn) / span  # [0, 1]^3 (span is the cube side)
    qi = np.floor(np.clip(norm, 0, 1) * ((1 << 21) - 1)).astype(np.int64)
    morton = _morton_key(qi)

    def split(key: tuple[int, int, int, int], idx: np.ndarray, depth: int) -> None:
        d, ix, iy, iz = key
        n = idx.size
        if n <= max_points or depth >= MAX_DEPTH:
            nodes[key] = idx
            return
        # Decimate: keep every ~ceil(n/max_points)-th point in Morton order so
        # the node holds a spatially well-spread subset of its voxel.
        stride = max(2, int(np.ceil(n / max_points)))
        order = np.argsort(morton[idx], kind="stable")
        sorted_idx = idx[order]
        keep = sorted_idx[::stride]
        rest = np.delete(sorted_idx, np.arange(0, sorted_idx.size, stride))
        nodes[key] = keep
        if rest.size == 0:
            return
        # Distribute the remainder to the 8 children by the COPC voxel formula.
        # rel = global cell index at depth d+1; the local octant is rel & 1, and
        # the child cell is (2*ix+local, 2*iy+local, 2*iz+local).
        child_d = d + 1
        step = span / (1 << child_d)
        rel = (xyz[rest] - mn) / step
        rel = np.clip(np.floor(rel), 0, (1 << child_d) - 1).astype(np.int32)
        for cx in range(2):
            for cy in range(2):
                for cz in range(2):
                    m = (
                        ((rel[:, 0] & 1) == cx)
                        & ((rel[:, 1] & 1) == cy)
                        & ((rel[:, 2] & 1) == cz)
                    )
                    if m.any():
                        child_key = (child_d, ix * 2 + cx, iy * 2 + cy, iz * 2 + cz)
                        split(child_key, rest[m], depth + 1)

    split((0, 0, 0, 0), np.arange(xyz.shape[0]), 0)
    return nodes


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--provenance", type=Path, default=DEFAULT_PROVENANCE)
    parser.add_argument("--max-points-per-node", type=int, default=DEFAULT_MAX_POINTS_PER_NODE)
    parser.add_argument(
        "--validate",
        action="store_true",
        help="round-trip read the written COPC and verify the point count",
    )
    args = parser.parse_args(argv)

    if not args.input.is_file():
        print(f"[p3a-giro3d] ERROR: input LAS not found: {args.input}", file=sys.stderr)
        return 2

    t0 = time.time()
    print(f"[p3a-giro3d] reading {args.input} ...")
    las = laspy.read(args.input)
    n = las.header.point_count
    print(f"[p3a-giro3d] source point count: {n}")

    xyz = np.column_stack([las.x, las.y, las.z]).astype(np.float64)
    rgb = np.column_stack([las.red, las.green, las.blue]).astype(np.int64)
    del las

    mn = xyz.min(axis=0)
    mx = xyz.max(axis=0)
    span = float(max(mx - mn))
    scale = tuple(float(s) for s in (0.001, 0.001, 0.001))
    offset = tuple(float(o) for o in (mn + mx) / 2)  # center offset preserves precision

    print(f"[p3a-giro3d] bounds min={mn.tolist()} max={mx.tolist()} span={span:.3f}")

    print(f"[p3a-giro3d] building octree (max {args.max_points_per_node}/node) ...")
    nodes = build_octree(xyz, mn, span, args.max_points_per_node)
    total_in_nodes = sum(int(v.size) for v in nodes.values())
    depths = [k[0] for k in nodes]
    print(
        f"[p3a-giro3d] octree: {len(nodes)} nodes, depth {min(depths)}..{max(depths)}, "
        f"points {total_in_nodes}"
    )
    assert total_in_nodes == n, f"octree lost points: {total_in_nodes} != {n}"

    args.output.parent.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------
    # COPC configuration (mirrors copc-lib CopcConfig + CreateVoxelStructure)
    # ---------------------------------------------------------------
    cfg = copclib.CopcConfigWriter(
        POINT_FORMAT_ID,
        scale=copclib.Vector3(*scale),
        offset=copclib.Vector3(*offset),
    )
    cfg.las_header.min = copclib.Vector3(*mn)
    cfg.las_header.max = copclib.Vector3(*mx)
    # Octree cube: side = header.Span(), anchored at header.min.
    cfg.copc_info.spacing = span
    cfg.copc_info.halfsize = span / 2
    cfg.copc_info.center_x = mn[0] + span / 2
    cfg.copc_info.center_y = mn[1] + span / 2
    cfg.copc_info.center_z = mn[2] + span / 2

    print(f"[p3a-giro3d] writing COPC to {args.output} ...")
    writer = copclib.FileWriter(str(args.output), cfg)
    try:
        for key, idx in nodes.items():
            pts = copclib.Points(POINT_FORMAT_ID)
            for i in idx:
                p = pts.CreatePoint()
                p.x = float(xyz[i, 0])
                p.y = float(xyz[i, 1])
                p.z = float(xyz[i, 2])
                p.red = int(rgb[i, 0])
                p.green = int(rgb[i, 1])
                p.blue = int(rgb[i, 2])
                pts.AddPoint(p)
            writer.AddNode(copclib.VoxelKey(*key), pts)
    finally:
        writer.Close()

    size = args.output.stat().st_size
    sha = sha256_file(args.output)
    elapsed = time.time() - t0
    print(f"[p3a-giro3d] wrote {size} bytes ({size / 1e6:.2f} MB) in {elapsed:.1f}s")
    print(f"[p3a-giro3d] sha256 {sha}")

    # ---------------------------------------------------------------
    # Round-trip validation
    # ---------------------------------------------------------------
    if args.validate:
        print("[p3a-giro3d] validating round-trip read ...")
        reader = copclib.FileReader(str(args.output))
        read_nodes = list(reader.GetAllNodes())
        read_count = 0
        for node in read_nodes:
            read_count += sum(1 for _ in reader.GetPoints(node))
        reader.Close()
        print(f"[p3a-giro3d] read back {read_nodes.__len__()} nodes, {read_count} points")
        if read_count != n:
            print(
                f"[p3a-giro3d] ERROR: round-trip count mismatch {read_count} != {n}",
                file=sys.stderr,
            )
            return 3

    args.provenance.parent.mkdir(parents=True, exist_ok=True)
    prov = {
        "schema": "p3a-giro3d-copc-provenance/v1",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tool": TOOL_NAME,
        "tool_version": TOOL_VERSION,
        "tool_license": TOOL_LICENSE,
        "point_format_id": POINT_FORMAT_ID,
        "input": {"path": str(args.input), "point_count": int(n)},
        "output": {
            "path": str(args.output),
            "size_bytes": int(size),
            "sha256": sha,
        },
        "octree": {
            "nodes": len(nodes),
            "depth_min": int(min(depths)),
            "depth_max": int(max(depths)),
            "max_points_per_node": args.max_points_per_node,
        },
        "bounds": {"min": mn.tolist(), "max": mx.tolist(), "span": span},
    }
    args.provenance.write_text(json.dumps(prov, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[p3a-giro3d] provenance written to {args.provenance}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
