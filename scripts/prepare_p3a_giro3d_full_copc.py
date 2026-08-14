"""Convert the COMPLETE protected source LAS to a COPC for the full-scale
Giro3D viewer gate (105,055,350 points).

This is a viewer-cache operation ONLY.  The protected source is streamed
read-only and never modified; a separate derived cache is written under the
ignored root ``cache/giro3d_full_source/``.

Memory strategy: the source is too large to hold in RAM (only ~2 GB available
on the dev machine), so the raw point data is streamed into numpy memmap files,
the octree is built with bounded-memory slices, and the COPC is written
node-by-node (per-point is fast enough: ~115 s for 105M).

Usage (run with the giro3d tooling venv python)::

    cache/p3a_giro3d/python/giro3d-venv/Scripts/python.exe \\
        scripts/prepare_p3a_giro3d_full_copc.py [--validate]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

import copclib
import laspy
import numpy as np

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE = _REPOSITORY_ROOT / "input_las" / "2026-08-07_16没54-10_PointCloud_colorized.las"
CACHE_ROOT = _REPOSITORY_ROOT / "cache" / "giro3d_full_source"
TMP_DIR = CACHE_ROOT / "tmp"
DATASET_DIR = CACHE_ROOT / "dataset"
OUTPUT = DATASET_DIR / "p3a_full.copc.laz"
PROVENANCE = CACHE_ROOT / "copc-provenance.json"

POINT_FORMAT_ID = 7
DEFAULT_MAX_POINTS_PER_NODE = 100000
MAX_DEPTH = 12
CHUNK = 1_000_000        # LAS streaming chunk
SLICE = 2_000_000        # octree-distribution slice (bounds transient memory)

TOOL_NAME = "copclib (copc-lib Python bindings + bundled lazperf)"
TOOL_VERSION = getattr(copclib, "__version__", "2.6.3")
TOOL_LICENSE = "BSD-3-Clause (copc-lib); LASzip-compatible LAZ via lazperf"


def sha256_stream(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def _morton_spread(q: np.ndarray) -> np.ndarray:
    q = q & 0x1FFFFF
    q = (q | (q << 32)) & 0x1F00000000FFFF
    q = (q | (q << 16)) & 0x1F0000FF0000FF
    q = (q | (q << 8)) & 0x100F00F00F00F00F
    q = (q | (q << 4)) & 0x10C30C30C30C30C3
    q = (q | (q << 2)) & 0x1249249249249249
    return q


def _morton_key_slice(xyz_raw: np.ndarray, mn: np.ndarray, span: float) -> np.ndarray:
    """Morton code for a slice of raw int32 xyz (normalized into [0, 2^21))."""
    q = ((xyz_raw.astype(np.int64) - mn) * ((1 << 21) / span)).astype(np.int64)
    q = np.clip(q, 0, (1 << 21) - 1)
    return _morton_spread(q[:, 0]) | (_morton_spread(q[:, 1]) << 1) | (_morton_spread(q[:, 2]) << 2)


def stream_to_memmap(src: Path, tmp_dir: Path, n: int, scale, offset):
    """Stream the LAS into raw memmap arrays (int32 xyz + uint16 rgb)."""
    xyz = np.memmap(tmp_dir / "xyz_raw.bin", dtype=np.int32, mode="w+", shape=(n, 3))
    rgb = np.memmap(tmp_dir / "rgb_raw.bin", dtype=np.uint16, mode="w+", shape=(n, 3))
    las = laspy.open(src)
    off = 0
    t0 = time.time()
    for chunk in las.chunk_iterator(CHUNK):
        m = len(chunk)
        xyz[off : off + m, 0] = chunk.X
        xyz[off : off + m, 1] = chunk.Y
        xyz[off : off + m, 2] = chunk.Z
        rgb[off : off + m, 0] = chunk.red
        rgb[off : off + m, 1] = chunk.green
        rgb[off : off + m, 2] = chunk.blue
        off += m
    las.close()
    xyz.flush()
    rgb.flush()
    print(f"[full-copc] streamed {off} points in {time.time() - t0:.1f}s")
    if off != n:
        raise RuntimeError(f"streamed {off} != expected {n}")
    return xyz, rgb


def build_octree(xyz_raw: np.ndarray, mn: np.ndarray, span: float,
                 max_points: int) -> dict[tuple[int, int, int, int], np.ndarray]:
    """Return {voxel_key: index_array} with every point in exactly one node.

    Same capacity-based Morton-decimation as the ROI converter, but the child
    distribution is processed in slices so the transient memory stays bounded
    (the raw point data itself lives in the memmap).
    """
    nodes: dict[tuple[int, int, int, int], np.ndarray] = {}

    def split(key, idx, depth):
        d, ix, iy, iz = key
        n = idx.size
        if n <= max_points or depth >= MAX_DEPTH:
            nodes[key] = idx
            return
        # Morton decimation: keep every ~ceil(n/max_points)-th point (spatial
        # order).  Computed in slices so the root's Morton transient stays small.
        m = np.empty(n, dtype=np.int64)
        for start in range(0, n, SLICE):
            m[start : start + SLICE] = _morton_key_slice(
                xyz_raw[idx[start : start + SLICE]], mn, span
            )
        order = np.argsort(m, kind="stable")
        del m
        sorted_idx = idx[order]
        stride = max(2, int(math.ceil(n / max_points)))
        keep = sorted_idx[::stride]
        rest = np.delete(sorted_idx, np.arange(0, n, stride))
        nodes[key] = keep
        if rest.size == 0:
            return
        # Distribute the remainder to children in slices (bounded memory).
        child_d = d + 1
        n_cells = 1 << child_d
        step = span / n_cells
        bins = [list() for _ in range(8)]
        for start in range(0, rest.size, SLICE):
            s = rest[start : start + SLICE]
            rel = ((xyz_raw[s].astype(np.int64) - mn) / step).astype(np.int64)
            rel = np.clip(rel, 0, n_cells - 1).astype(np.int32)
            for o in range(8):
                cx, cy, cz = o & 1, (o >> 1) & 1, (o >> 2) & 1
                m = (
                    ((rel[:, 0] & 1) == cx)
                    & ((rel[:, 1] & 1) == cy)
                    & ((rel[:, 2] & 1) == cz)
                )
                if m.any():
                    bins[o].append(s[m])
        for o in range(8):
            if bins[o]:
                cx, cy, cz = o & 1, (o >> 1) & 1, (o >> 2) & 1
                child_key = (child_d, ix * 2 + cx, iy * 2 + cy, iz * 2 + cz)
                split(child_key, np.concatenate(bins[o]), depth + 1)

    split((0, 0, 0, 0), np.arange(xyz_raw.shape[0]), 0)
    return nodes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--max-points-per-node", type=int, default=DEFAULT_MAX_POINTS_PER_NODE)
    parser.add_argument("--validate", action="store_true", help="round-trip read the COPC")
    args = parser.parse_args(argv)

    if not args.source.is_file():
        print(f"[full-copc] ERROR: source not found: {args.source}", file=sys.stderr)
        return 2

    free = shutil.disk_usage(CACHE_ROOT if CACHE_ROOT.exists() else _REPOSITORY_ROOT).free
    print(f"[full-copc] free disk on cache drive: {free / 1e9:.1f} GB")

    # ------------------------------------------------------------------ source
    stat = args.source.stat()
    src_size, src_mtime_ns = stat.st_size, stat.st_mtime_ns
    print(f"[full-copc] source size {src_size} bytes, mtime {src_mtime_ns}")
    print("[full-copc] computing source SHA-256 (streaming, read-only) ...")
    t0 = time.time()
    src_sha = sha256_stream(args.source)
    print(f"[full-copc] source sha256 {src_sha} in {time.time() - t0:.1f}s")

    header = laspy.open(args.source).header
    n = int(header.point_count)
    scale = tuple(float(s) for s in header.scales)
    offset = tuple(float(o) for o in header.offsets)
    print(f"[full-copc] point count {n}, format {header.point_format.id}, scale {scale}, offset {offset}")

    # ------------------------------------------------------------- stream to memmap
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    print("[full-copc] streaming LAS to memmap ...")
    xyz_raw, rgb_raw = stream_to_memmap(args.source, TMP_DIR, n, scale, offset)
    # bounds in raw units
    mn_raw = xyz_raw.min(axis=0)
    mx_raw = xyz_raw.max(axis=0)
    span = float(max(mx_raw - mn_raw))
    print(f"[full-copc] raw bounds min {mn_raw.tolist()} max {mx_raw.tolist()} span {span:.1f}")

    # ------------------------------------------------------------- build octree
    print(f"[full-copc] building octree (max {args.max_points_per_node}/node, memmap-backed) ...")
    t0 = time.time()
    nodes = build_octree(xyz_raw, mn_raw, span, args.max_points_per_node)
    depths = [k[0] for k in nodes]
    total_in_nodes = sum(int(v.size) for v in nodes.values())
    print(f"[full-copc] octree: {len(nodes)} nodes, depth {min(depths)}..{max(depths)}, "
          f"{total_in_nodes} points in {time.time() - t0:.1f}s")
    if total_in_nodes != n:
        raise RuntimeError(f"octree lost points: {total_in_nodes} != {n}")

    # ------------------------------------------------------------- write COPC
    cfg = copclib.CopcConfigWriter(
        POINT_FORMAT_ID,
        scale=copclib.Vector3(*scale),
        offset=copclib.Vector3(*offset),
    )
    cfg.las_header.min = copclib.Vector3(*(float(x) for x in mn_raw * np.array(scale) + np.array(offset)))
    cfg.las_header.max = copclib.Vector3(*(float(x) for x in mx_raw * np.array(scale) + np.array(offset)))
    real_span = max(
        (mx_raw - mn_raw) * np.array(scale)
    )
    cfg.copc_info.spacing = float(real_span)
    cfg.copc_info.halfsize = float(real_span) / 2
    real_center = (mn_raw + mx_raw) / 2 * np.array(scale) + np.array(offset)
    cfg.copc_info.center_x = float(real_center[0])
    cfg.copc_info.center_y = float(real_center[1])
    cfg.copc_info.center_z = float(real_center[2])

    print(f"[full-copc] writing COPC to {OUTPUT} ...")
    t0 = time.time()
    writer = copclib.FileWriter(str(OUTPUT), cfg)
    try:
        for key, idx in nodes.items():
            pts = copclib.Points(POINT_FORMAT_ID)
            for i in idx:
                p = pts.CreatePoint()
                p.x = float(xyz_raw[i, 0]) * scale[0] + offset[0]
                p.y = float(xyz_raw[i, 1]) * scale[1] + offset[1]
                p.z = float(xyz_raw[i, 2]) * scale[2] + offset[2]
                p.red = int(rgb_raw[i, 0])
                p.green = int(rgb_raw[i, 1])
                p.blue = int(rgb_raw[i, 2])
                pts.AddPoint(p)
            writer.AddNode(copclib.VoxelKey(*key), pts)
            del pts
    finally:
        writer.Close()
    write_s = time.time() - t0
    print(f"[full-copc] COPC written in {write_s:.1f}s")

    out_size = OUTPUT.stat().st_size
    out_sha = sha256_stream(OUTPUT)
    print(f"[full-copc] COPC size {out_size / 1e6:.1f} MB, sha256 {out_sha}")

    # ------------------------------------------------------------- verify source
    stat2 = args.source.stat()
    if stat2.st_size != src_size or stat2.st_mtime_ns != src_mtime_ns:
        print("[full-copc] ERROR: source changed during conversion!", file=sys.stderr)
        return 4
    print("[full-copc] source unchanged (size + mtime verified)")

    # ------------------------------------------------------------- validate
    if args.validate:
        print("[full-copc] validating round-trip read ...")
        reader = copclib.FileReader(str(OUTPUT))
        total = 0
        for node in reader.GetAllNodes():
            total += sum(1 for _ in reader.GetPoints(node))
        reader.Close()
        print(f"[full-copc] round-trip read {total} points")
        if total != n:
            print(f"[full-copc] ERROR: round-trip count mismatch {total} != {n}", file=sys.stderr)
            return 5

    # ------------------------------------------------------------- provenance
    prov = {
        "schema": "giro3d-full-source-copc-provenance/v1",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tool": TOOL_NAME,
        "tool_version": TOOL_VERSION,
        "tool_license": TOOL_LICENSE,
        "point_format_id": POINT_FORMAT_ID,
        "source": {
            "path": str(args.source),
            "size_bytes": int(src_size),
            "mtime_ns": int(src_mtime_ns),
            "point_count": int(n),
            "sha256": src_sha,
            "modified_during_conversion": False,
        },
        "output": {"path": str(OUTPUT), "size_bytes": int(out_size), "sha256": out_sha},
        "octree": {"nodes": len(nodes), "depth_min": int(min(depths)), "depth_max": int(max(depths)),
                   "max_points_per_node": args.max_points_per_node},
        "bounds_raw": {"min": mn_raw.tolist(), "max": mx_raw.tolist(), "span": span},
        "conversion_duration_seconds": None,  # filled by caller if desired
    }
    PROVENANCE.write_text(json.dumps(prov, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[full-copc] provenance written to {PROVENANCE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
