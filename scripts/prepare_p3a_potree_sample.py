"""Prepare bounded real-RGB LAS samples for the Potree POC (P3A-I3).

Two modes:

* **Whole-scene stride** (default): stream a source LAS read-only and keep
  every ``stride``-th point deterministically.  Good for a sparse technical /
  global / debug view of a large cloud.

* **ROI corridor** (``--pole-a ... --pole-b ...``): stream read-only and keep
  points inside an XY corridor around the given span (perpendicular distance
  to the segment <= ``--corridor-half-width``, projection within
  ``[ -extension, length + extension ]``).  This preserves real local RGB at
  high density for Owner tower/span visibility.  The ROI output is capped at
  ``target_points`` (<= ``MAX_P3A_SAMPLE_POINTS``); if the raw ROI exceeds the
  cap a deterministic in-ROI subsample is applied.

The source is never modified; its immutability (size and mtime) is re-verified
after sampling, failing closed if it changed.  This is a **bounded sample
generator**: ``--target-points`` is hard-capped at ``MAX_P3A_SAMPLE_POINTS``
(1,000,000) and validated before any file is read or written.

Usage::

    python scripts/prepare_p3a_potree_sample.py
    python scripts/prepare_p3a_potree_sample.py --target-points 500000
    python scripts/prepare_p3a_potree_sample.py --output /tmp/sample.las
    python scripts/prepare_p3a_potree_sample.py \\
        --pole-a "-66.3449 2.2295 0.8314" --pole-b "-25.3627 20.4183 -0.2397" \\
        --target-points 1000000 --output cache/p3a_potree/input/p3a_roi.las \\
        --provenance cache/p3a_potree/roi-provenance.json

Prints a JSON summary line prefixed with ``SUMMARY `` that the automated tests
parse to assert point counts / RGB presence / bounds / provenance.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_SOURCE = (
    _REPOSITORY_ROOT
    / "input_las"
    / "2026-08-07_16没54-10_PointCloud_colorized.las"
)
DEFAULT_OUTPUT = _REPOSITORY_ROOT / "cache" / "p3a_potree" / "input" / "p3a_sample.las"
DEFAULT_PROVENANCE = (
    _REPOSITORY_ROOT / "cache" / "p3a_potree" / "sample-provenance.json"
)

# Hard, non-overridable bound on the derived sample size.  This is the P3A
# contract: the sample generator never emits more than this many points.
MAX_P3A_SAMPLE_POINTS = 1_000_000

# Default corridor geometry for the ROI mode (meters).
DEFAULT_ROI_HALF_WIDTH = 25.0
DEFAULT_ROI_EXTENSION = 15.0

_SHA256_CHUNK = 1 << 20


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(_SHA256_CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def _utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


def _check_dependencies() -> None:
    """Give a clear error when run outside the P3A venv."""
    try:
        import laspy  # noqa: F401
    except ImportError as exc:  # pragma: no cover - environment error path
        raise SystemExit(
            "laspy is not importable. Activate the P3A venv, e.g.\n"
            "  cache\\p3a_potree\\python\\p3a-venv\\Scripts\\python.exe -m pip install laspy numpy"
        ) from exc


def _validate_target_points(value: str) -> int:
    """argparse ``type`` validator for ``--target-points``.

    Rejects zero, negatives and anything above the hard sample cap before any
    file is read or written.
    """
    try:
        parsed = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"--target-points must be an integer, got {value!r}"
        )
    if parsed <= 0:
        raise argparse.ArgumentTypeError(
            f"--target-points must be positive, got {parsed}"
        )
    if parsed > MAX_P3A_SAMPLE_POINTS:
        raise argparse.ArgumentTypeError(
            f"--target-points must be <= MAX_P3A_SAMPLE_POINTS="
            f"{MAX_P3A_SAMPLE_POINTS}, got {parsed}"
        )
    return parsed


def _validate_target_points_function(target_points: int) -> None:
    if target_points <= 0:
        raise ValueError(
            f"target_points must be positive, got {target_points}"
        )
    if target_points > MAX_P3A_SAMPLE_POINTS:
        raise ValueError(
            f"target_points {target_points} exceeds hard cap "
            f"MAX_P3A_SAMPLE_POINTS={MAX_P3A_SAMPLE_POINTS}"
        )


def _parse_xyz(value: str) -> tuple[float, float, float]:
    parts = [float(part) for part in value.replace(",", " ").split()]
    if len(parts) < 2:
        raise argparse.ArgumentTypeError("expected at least X Y (Z optional)")
    if len(parts) == 2:
        return (parts[0], parts[1], 0.0)
    return (parts[0], parts[1], parts[2])


def _roi_mask(
    xs: np.ndarray,
    ys: np.ndarray,
    pole_a: tuple[float, float],
    pole_b: tuple[float, float],
    half_width: float,
    extension: float,
) -> np.ndarray:
    """Boolean mask for points inside the A-B XY corridor."""
    ax, ay = pole_a
    bx, by = pole_b
    dx = bx - ax
    dy = by - ay
    seg_len = math.sqrt(dx * dx + dy * dy)
    seg_len2 = seg_len * seg_len
    cx = xs - ax
    cy = ys - ay
    proj = (cx * dx + cy * dy) / seg_len2
    perp = np.abs(cx * dy - cy * dx) / seg_len
    t_min = -extension / seg_len
    t_max = 1 + extension / seg_len
    return (perp <= half_width) & (proj >= t_min) & (proj <= t_max)


def prepare_sample(
    source: Path,
    output: Path,
    target_points: int,
    chunk_size: int = 1_000_000,
    provenance_path: Path | None = None,
    *,
    roi_pole_a: tuple[float, float] | None = None,
    roi_pole_b: tuple[float, float] | None = None,
    roi_half_width: float = DEFAULT_ROI_HALF_WIDTH,
    roi_extension: float = DEFAULT_ROI_EXTENSION,
) -> dict:
    """Stream ``source`` read-only and write a bounded RGB sample.

    Args:
        source: input LAS (never modified; re-statted for immutability).
        output: sample LAS to write.
        target_points: requested sample size / ROI cap; validated against the
            hard cap.
        chunk_size: streaming chunk size in points.
        provenance_path: where to write the derivation-provenance JSON; if
            ``None`` the provenance facts are still returned in the summary but
            no file is written.
        roi_pole_a / roi_pole_b: both given -> ROI-corridor mode (XY only).
        roi_half_width: corridor half width in meters.
        roi_extension: corridor extension beyond the poles in meters.

    Returns:
        Summary dict including source immutability facts and the sample SHA-256.

    Raises:
        ValueError: invalid parameters.
        RuntimeError: the source changed while sampled, or bounds violated.
    """
    import laspy
    from laspy.point import record as laspy_record

    source = Path(source)
    output = Path(output)

    # P3A-001: hard bounded-sample guard, checked before any I/O.
    _validate_target_points_function(target_points)
    if source.resolve() == output.resolve():
        raise ValueError("output must not equal the protected source path")

    if (roi_pole_a is None) != (roi_pole_b is None):
        raise ValueError("both pole A and pole B must be provided for ROI mode")
    roi_mode = roi_pole_a is not None and roi_pole_b is not None
    if roi_half_width <= 0:
        raise ValueError("roi_half_width must be positive")
    if roi_extension < 0:
        raise ValueError("roi_extension must be >= 0")
    if roi_mode:
        pole_a = (float(roi_pole_a[0]), float(roi_pole_a[1]))
        pole_b = (float(roi_pole_b[0]), float(roi_pole_b[1]))
        if pole_a == pole_b:
            raise ValueError("pole A and pole B must differ")

    source_stat_before = source.stat()
    source_size_before = source_stat_before.st_size
    source_mtime_ns_before = source_stat_before.st_mtime_ns

    kept_x: list[np.ndarray] = []
    kept_y: list[np.ndarray] = []
    kept_z: list[np.ndarray] = []
    kept_r: list[np.ndarray] = []
    kept_g: list[np.ndarray] = []
    kept_b: list[np.ndarray] = []
    step = 1
    roi_raw_count = 0
    roi_stride = 1

    with laspy.open(str(source)) as reader:
        header = reader.header
        total = int(header.point_count)
        if total <= 0:
            raise ValueError(f"source has no points: {source}")

        if roi_mode:
            # Pass 1: count raw ROI points (read-only).
            for chunk in reader.chunk_iterator(chunk_size):
                n = len(chunk)
                if n == 0:
                    continue
                mask = _roi_mask(
                    np.asarray(chunk.x),
                    np.asarray(chunk.y),
                    pole_a,
                    pole_b,
                    roi_half_width,
                    roi_extension,
                )
                roi_raw_count += int(mask.sum())
            if roi_raw_count == 0:
                raise RuntimeError("ROI corridor contains no points; tighten or widen geometry")
            # Deterministic in-ROI subsample stride to honour the cap.
            roi_stride = 1 if roi_raw_count <= target_points else math.ceil(
                roi_raw_count / target_points
            )
            # Pass 2: extract ROI points, keeping every roi_stride-th in ROI order.
            roi_seen = 0
            with laspy.open(str(source)) as reader2:
                for chunk in reader2.chunk_iterator(chunk_size):
                    n = len(chunk)
                    if n == 0:
                        continue
                    xs = np.asarray(chunk.x)
                    ys = np.asarray(chunk.y)
                    mask = _roi_mask(xs, ys, pole_a, pole_b, roi_half_width, roi_extension)
                    indices = np.nonzero(mask)[0]
                    if indices.size == 0:
                        continue
                    if roi_stride > 1:
                        global_pos = roi_seen + np.arange(indices.size)
                        keep = global_pos % roi_stride == 0
                        sel = indices[keep]
                    else:
                        sel = indices
                    roi_seen += indices.size
                    if sel.size:
                        kept_x.append(xs[sel])
                        kept_y.append(ys[sel])
                        kept_z.append(np.asarray(chunk.z)[sel])
                        kept_r.append(np.asarray(chunk.red[sel], dtype=np.uint16))
                        kept_g.append(np.asarray(chunk.green[sel], dtype=np.uint16))
                        kept_b.append(np.asarray(chunk.blue[sel], dtype=np.uint16))
            step = roi_stride
        else:
            # Whole-scene stride mode (existing behaviour).
            step = max(1, math.ceil(total / target_points))
            seen = 0
            for chunk in reader.chunk_iterator(chunk_size):
                n = len(chunk)
                if n == 0:
                    continue
                pos = np.arange(n, dtype=np.int64) + seen
                keep = pos % step == 0
                if keep.any():
                    kept_x.append(chunk.x[keep])
                    kept_y.append(chunk.y[keep])
                    kept_z.append(chunk.z[keep])
                    kept_r.append(np.asarray(chunk.red[keep], dtype=np.uint16))
                    kept_g.append(np.asarray(chunk.green[keep], dtype=np.uint16))
                    kept_b.append(np.asarray(chunk.blue[keep], dtype=np.uint16))
                seen += n

    # P3A-005: fail-closed source immutability re-stat.
    source_stat_after = source.stat()
    source_size_after = source_stat_after.st_size
    source_mtime_ns_after = source_stat_after.st_mtime_ns
    if (
        source_size_before != source_size_after
        or source_mtime_ns_before != source_mtime_ns_after
    ):
        raise RuntimeError(
            "protected source changed during sampling: "
            f"size {source_size_before}->{source_size_after}, "
            f"mtime_ns {source_mtime_ns_before}->{source_mtime_ns_after}"
        )

    if not kept_x:
        raise RuntimeError("no points kept; sampling step too large")

    xs = np.concatenate(kept_x)
    ys = np.concatenate(kept_y)
    zs = np.concatenate(kept_z)
    rs = np.concatenate(kept_r)
    gs = np.concatenate(kept_g)
    bs = np.concatenate(kept_b)
    count = int(xs.shape[0])

    # Defense in depth on the bounded-sample contract.
    if count > MAX_P3A_SAMPLE_POINTS:
        raise RuntimeError(
            f"sample output {count} exceeds MAX_P3A_SAMPLE_POINTS="
            f"{MAX_P3A_SAMPLE_POINTS}"
        )

    out_header = laspy.LasHeader(point_format=7, version="1.4")
    out_header.scales = header.scales
    out_header.offsets = header.offsets
    record = laspy_record.ScaleAwarePointRecord(
        np.zeros(count, dtype=out_header.point_format.dtype()),
        out_header.point_format,
        out_header.scales,
        out_header.offsets,
    )
    las = laspy.LasData(out_header)
    las.points = record
    las.x = xs
    las.y = ys
    las.z = zs
    las.red = rs
    las.green = gs
    las.blue = bs

    output.parent.mkdir(parents=True, exist_ok=True)
    las.write(str(output))

    sample_sha256 = _sha256(output)
    sample_size_bytes = int(output.stat().st_size)

    if roi_mode:
        provenance = {
            "schema": "p3a-roi-provenance/v1",
            "generated_utc": _utc_now(),
            "derivation_provenance_only": True,
            "note": (
                "P3A display-sample ROI derivation provenance; pole coordinates "
                "are read-only validated-run guidance, not project contracts; "
                "not a scientific SourceIdentity record."
            ),
            "protected_source": {
                "path": str(source),
                "size_bytes_before": source_size_before,
                "size_bytes_after": source_size_after,
                "mtime_ns_before": source_mtime_ns_before,
                "mtime_ns_after": source_mtime_ns_after,
                "immutability": "PASS",
                "point_count": total,
            },
            "roi": {
                "pole_a": list(pole_a),
                "pole_b": list(pole_b),
                "half_width": roi_half_width,
                "extension": roi_extension,
                "definition": (
                    "XY corridor around the A-B segment: perpendicular distance "
                    "<= half_width AND projection within [-extension, length+extension]; "
                    "Z unrestricted within the bounded XY region"
                ),
            },
            "sample": {
                "output_path": str(output),
                "point_count": count,
                "size_bytes": sample_size_bytes,
                "sha256": sample_sha256,
                "roi_raw_point_count": roi_raw_count,
                "roi_subsample_stride": roi_stride,
                "sampling_rule": (
                    "keep all ROI points when raw count <= cap; otherwise keep "
                    "every roi_subsample_stride-th ROI point (deterministic, no RNG)"
                ),
                "point_format": 7,
                "las_version": "1.4",
            },
        }
        summary_extra = {
            "roi_raw_point_count": roi_raw_count,
            "roi_subsample_stride": roi_stride,
            "roi_pole_a": list(pole_a),
            "roi_pole_b": list(pole_b),
            "roi_half_width": roi_half_width,
            "roi_extension": roi_extension,
        }
    else:
        provenance = {
            "schema": "p3a-sample-provenance/v2",
            "generated_utc": _utc_now(),
            "derivation_provenance_only": True,
            "note": "P3A derivation provenance; not a scientific SourceIdentity record.",
            "protected_source": {
                "path": str(source),
                "size_bytes_before": source_size_before,
                "size_bytes_after": source_size_after,
                "mtime_ns_before": source_mtime_ns_before,
                "mtime_ns_after": source_mtime_ns_after,
                "immutability": "PASS",
                "point_count": total,
            },
            "sample": {
                "output_path": str(output),
                "point_count": count,
                "size_bytes": sample_size_bytes,
                "sha256": sample_sha256,
                "sampling_stride": step,
                "sampling_rule": (
                    "keep every Nth source record (0-based index divisible by "
                    "stride), deterministic, no RNG"
                ),
                "selected_source_record_semantics": (
                    "source record k with (k % stride == 0) supplies position and "
                    "RGB verbatim, in source order"
                ),
                "point_format": 7,
                "las_version": "1.4",
            },
        }
        summary_extra = {
            "sampling_step": step,
        }

    if provenance_path is not None:
        provenance_path = Path(provenance_path)
        provenance_path.parent.mkdir(parents=True, exist_ok=True)
        provenance_path.write_text(
            json.dumps(provenance, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    summary = {
        "source_point_count": total,
        "output_point_count": count,
        "sample_sha256": sample_sha256,
        "sample_size_bytes": sample_size_bytes,
        "source_size_bytes_before": source_size_before,
        "source_size_bytes_after": source_size_after,
        "source_mtime_ns_before": source_mtime_ns_before,
        "source_mtime_ns_after": source_mtime_ns_after,
        "output_path": str(output),
        "output_mb": round(sample_size_bytes / (1024 * 1024), 3),
        "bounds_min": [float(xs.min()), float(ys.min()), float(zs.min())],
        "bounds_max": [float(xs.max()), float(ys.max()), float(zs.max())],
        "rgb_present": bool((rs.max() > 0) or (gs.max() > 0) or (bs.max() > 0)),
        "point_format": 7,
        "las_version": "1.4",
        "provenance_path": str(provenance_path) if provenance_path is not None else None,
    }
    summary.update(summary_extra)
    return summary


def main(argv: list[str] | None = None) -> int:
    _check_dependencies()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help="protected source LAS (read-only; default: input_las source)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="output sample LAS path (default: cache/p3a_potree/input/p3a_sample.las)",
    )
    parser.add_argument(
        "--target-points",
        type=_validate_target_points,
        default=300_000,
        help=(
            f"approximate number of sample points / ROI cap (default: 300000, "
            f"hard max {MAX_P3A_SAMPLE_POINTS})"
        ),
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1_000_000,
        help="streaming chunk size in points (default: 1000000)",
    )
    parser.add_argument(
        "--provenance",
        type=Path,
        default=DEFAULT_PROVENANCE,
        help="derivation-provenance JSON output (default: cache/p3a_potree/sample-provenance.json)",
    )
    parser.add_argument(
        "--pole-a",
        type=_parse_xyz,
        default=None,
        help='ROI mode: pole A as "X Y [Z]" (validated-span guidance)',
    )
    parser.add_argument(
        "--pole-b",
        type=_parse_xyz,
        default=None,
        help='ROI mode: pole B as "X Y [Z]" (validated-span guidance)',
    )
    parser.add_argument(
        "--corridor-half-width",
        type=float,
        default=DEFAULT_ROI_HALF_WIDTH,
        help=f"ROI corridor half width in meters (default: {DEFAULT_ROI_HALF_WIDTH})",
    )
    parser.add_argument(
        "--corridor-extension",
        type=float,
        default=DEFAULT_ROI_EXTENSION,
        help=f"ROI extension beyond the poles in meters (default: {DEFAULT_ROI_EXTENSION})",
    )
    args = parser.parse_args(argv)

    summary = prepare_sample(
        args.source,
        args.output,
        args.target_points,
        args.chunk_size,
        provenance_path=args.provenance,
        roi_pole_a=args.pole_a,
        roi_pole_b=args.pole_b,
        roi_half_width=args.corridor_half_width,
        roi_extension=args.corridor_extension,
    )
    print("SUMMARY " + json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
