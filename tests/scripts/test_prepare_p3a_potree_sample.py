"""P3A-I3 sample-preparation tests (incl. P3A-001 cap + P3A-005 provenance).

The bounded sample must be a deterministic, spatially representative real-RGB
subset of a source, written read-only-safe, refuse to overwrite the protected
source itself, be hard-capped at ``MAX_P3A_SAMPLE_POINTS``, and record source
immutability plus a sample SHA-256.  Synthetic fixtures are used; the real
105M-point protected source is never touched.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import laspy
import numpy as np
import pytest
from laspy.point import record as laspy_record

import scripts.prepare_p3a_potree_sample as prepare

_N = 10_000
_SRC_DIR_NAME = "点云_没"  # CJK (incl. 没) for the Unicode-path test


def _write_rgb_las(path: Path, n: int = _N) -> Path:
    header = laspy.LasHeader(point_format=7, version="1.4")
    header.scales = np.array([0.0001, 0.0001, 0.0001])
    header.offsets = np.array([-197.0, -147.0, 0.0])
    rng = np.random.default_rng(7)
    x = -197.0 + rng.uniform(0.0, 300.0, n)
    y = -147.0 + rng.uniform(0.0, 280.0, n)
    z = rng.uniform(0.0, 45.0, n)
    red = rng.integers(0, 65536, n, dtype=np.uint16)
    green = rng.integers(0, 65536, n, dtype=np.uint16)
    blue = rng.integers(0, 65536, n, dtype=np.uint16)
    red[0] = 65535  # guarantee a non-zero channel for the rgb_present check
    record = laspy_record.ScaleAwarePointRecord(
        np.zeros(n, dtype=header.point_format.dtype()),
        header.point_format,
        header.scales,
        header.offsets,
    )
    las = laspy.LasData(header)
    las.points = record
    las.x = x
    las.y = y
    las.z = z
    las.red = red
    las.green = green
    las.blue = blue
    las.write(str(path))
    return path


def _write_deterministic_las(path: Path, n: int) -> Path:
    """LAS whose per-record raw X/Y/Z/R/G/B are a known function of the index.

    Values round-trip exactly through scale/offset so the exact-selection test
    can compare raw integer fields.
    """
    header = laspy.LasHeader(point_format=7, version="1.4")
    header.scales = np.array([0.0001, 0.0001, 0.0001])
    header.offsets = np.array([100.0, 200.0, 1.0])
    idx = np.arange(n, dtype=np.int64)
    raw = np.zeros(n, dtype=header.point_format.dtype())
    raw["X"] = idx
    raw["Y"] = idx * 2
    raw["Z"] = idx * 3
    raw["red"] = (idx * 7) % 65536
    raw["green"] = (idx * 11) % 65536
    raw["blue"] = (idx * 13) % 65536
    record = laspy_record.ScaleAwarePointRecord(
        raw, header.point_format, header.scales, header.offsets
    )
    las = laspy.LasData(header)
    las.points = record
    las.write(str(path))
    return path


# ---------------------------------------------------------------------------
# P3A-001: hard bounded-sample guard
# ---------------------------------------------------------------------------


def test_cli_rejects_zero_target(tmp_path: Path) -> None:
    src = _write_rgb_las(tmp_path / "src.las")
    with pytest.raises(SystemExit) as exc:
        prepare.main(
            ["--source", str(src), "--output", str(tmp_path / "s.las"), "--target-points", "0"]
        )
    assert exc.value.code != 0


def test_cli_rejects_negative_target(tmp_path: Path) -> None:
    src = _write_rgb_las(tmp_path / "src.las")
    with pytest.raises(SystemExit) as exc:
        prepare.main(
            ["--source", str(src), "--output", str(tmp_path / "s.las"), "--target-points", "-1"]
        )
    assert exc.value.code != 0


def test_cli_rejects_over_cap_target(tmp_path: Path) -> None:
    src = _write_rgb_las(tmp_path / "src.las")
    too_big = prepare.MAX_P3A_SAMPLE_POINTS + 1
    with pytest.raises(SystemExit) as exc:
        prepare.main(
            ["--source", str(src), "--output", str(tmp_path / "s.las"), "--target-points", str(too_big)]
        )
    assert exc.value.code != 0


def test_cli_rejects_huge_target_before_output(tmp_path: Path) -> None:
    src = _write_rgb_las(tmp_path / "src.las")
    out = tmp_path / "s.las"
    with pytest.raises(SystemExit):
        prepare.main(
            ["--source", str(src), "--output", str(out), "--target-points", "999999999"]
        )
    assert not out.exists(), "huge target must be rejected before output generation"


def test_function_rejects_invalid_targets(tmp_path: Path) -> None:
    src = _write_rgb_las(tmp_path / "src.las")
    out = tmp_path / "s.las"
    for bad in (0, -5):
        with pytest.raises(ValueError):
            prepare.prepare_sample(src, out, bad)
    with pytest.raises(ValueError):
        prepare.prepare_sample(src, out, prepare.MAX_P3A_SAMPLE_POINTS + 1)


def test_max_target_is_allowed(tmp_path: Path) -> None:
    src = _write_rgb_las(tmp_path / "src.las", n=2_000)
    out = tmp_path / "s.las"
    summary = prepare.prepare_sample(src, out, prepare.MAX_P3A_SAMPLE_POINTS)
    # ceil(2000 / 1M) == 1 -> keeps all 2000 points of the small source.
    assert summary["output_point_count"] == 2_000
    assert summary["output_point_count"] <= prepare.MAX_P3A_SAMPLE_POINTS


def test_large_target_does_not_copy_whole_source(tmp_path: Path) -> None:
    # A target > total/2 previously produced step == 1 (a whole-source copy);
    # ceil-step keeps the output bounded by target_points instead.
    src = _write_rgb_las(tmp_path / "src.las", n=100)
    out = tmp_path / "s.las"
    summary = prepare.prepare_sample(src, out, 60)
    assert summary["sampling_step"] == 2
    assert summary["output_point_count"] == 50
    assert summary["output_point_count"] < 100


# ---------------------------------------------------------------------------
# P3A-005: provenance, immutability, unicode path, exact selection
# ---------------------------------------------------------------------------


def test_source_immutability(tmp_path: Path) -> None:
    src = _write_rgb_las(tmp_path / "src.las")
    out = tmp_path / "sample.las"
    before = src.stat()
    digest_before = hashlib.sha256(src.read_bytes()).hexdigest()
    prepare.prepare_sample(src, out, 1_000)
    after = src.stat()
    assert after.st_size == before.st_size
    assert after.st_mtime_ns == before.st_mtime_ns
    assert hashlib.sha256(src.read_bytes()).hexdigest() == digest_before


def test_provenance_records_source_stats_and_sample_sha(tmp_path: Path) -> None:
    src = _write_rgb_las(tmp_path / "src.las", n=2_000)
    out = tmp_path / "sample.las"
    prov = tmp_path / "prov.json"
    summary = prepare.prepare_sample(src, out, 500, provenance_path=prov)

    assert summary["source_size_bytes_before"] == summary["source_size_bytes_after"]
    assert summary["source_mtime_ns_before"] == summary["source_mtime_ns_after"]

    data = json.loads(prov.read_text(encoding="utf-8"))
    ps = data["protected_source"]
    assert ps["size_bytes_before"] == ps["size_bytes_after"]
    assert ps["mtime_ns_before"] == ps["mtime_ns_after"]
    assert ps["immutability"] == "PASS"
    assert ps["point_count"] == 2_000

    sample = data["sample"]
    assert sample["sha256"] == summary["sample_sha256"]
    assert len(sample["sha256"]) == 64
    assert sample["point_count"] == 500
    assert sample["size_bytes"] == out.stat().st_size
    assert sample["sampling_stride"] == 4
    assert sample["selected_source_record_semantics"]
    assert sample["sampling_rule"]


def test_unicode_path_round_trip(tmp_path: Path) -> None:
    sub = tmp_path / _SRC_DIR_NAME
    sub.mkdir()
    src = sub / "src.las"
    _write_rgb_las(src, n=500)
    out = sub / "sample.las"
    prov = sub / "prov.json"
    summary = prepare.prepare_sample(src, out, 100, provenance_path=prov)

    assert out.is_file()
    assert prov.is_file()
    assert len(summary["sample_sha256"]) == 64

    las = laspy.read(str(out))
    assert las.header.point_count == 100
    assert int(las.red.max()) > 0

    data = json.loads(prov.read_text(encoding="utf-8"))
    assert data["protected_source"]["immutability"] == "PASS"
    assert Path(data["protected_source"]["path"]) == src


def test_exact_selected_xyz_rgb(tmp_path: Path) -> None:
    n = 100
    src = _write_deterministic_las(tmp_path / "src.las", n)
    src_read = laspy.read(str(src))

    out = tmp_path / "sample.las"
    summary = prepare.prepare_sample(src, out, 25)
    assert summary["sampling_step"] == 4

    out_read = laspy.read(str(out))
    assert out_read.header.point_count == 25

    # first / middle / last selected output rows map to source rows 0 / 48 / 96.
    for j, src_idx in ((0, 0), (12, 48), (24, 96)):
        assert int(out_read.X[j]) == int(src_read.X[src_idx])
        assert int(out_read.Y[j]) == int(src_read.Y[src_idx])
        assert int(out_read.Z[j]) == int(src_read.Z[src_idx])
        assert int(out_read.red[j]) == int(src_read.red[src_idx])
        assert int(out_read.green[j]) == int(src_read.green[src_idx])
        assert int(out_read.blue[j]) == int(src_read.blue[src_idx])


# ---------------------------------------------------------------------------
# Baseline sample behaviour
# ---------------------------------------------------------------------------


def test_sample_is_stratified_rgb(tmp_path: Path) -> None:
    src = _write_rgb_las(tmp_path / "src.las")
    out = tmp_path / "sample.las"
    summary = prepare.prepare_sample(src, out, target_points=1_000)

    assert summary["source_point_count"] == _N
    assert summary["output_point_count"] == 1_000  # every 10th point
    assert summary["rgb_present"] is True
    assert summary["point_format"] == 7
    assert summary["las_version"] == "1.4"
    assert out.is_file()

    las = laspy.read(str(out))
    assert las.header.point_count == 1_000
    assert las.header.point_format.id == 7
    assert int(las.red.max()) > 0
    assert int(las.green.max()) > 0
    assert int(las.blue.max()) > 0


def test_sample_preserves_source_scale_and_offset(tmp_path: Path) -> None:
    src = _write_rgb_las(tmp_path / "src.las")
    out = tmp_path / "sample.las"
    prepare.prepare_sample(src, out, target_points=500)
    las = laspy.read(str(out))
    np.testing.assert_allclose(las.header.scales, [0.0001, 0.0001, 0.0001])
    np.testing.assert_allclose(las.header.offsets, [-197.0, -147.0, 0.0])
    assert las.header.mins[0] >= -197.0
    assert las.header.maxs[0] <= -197.0 + 300.0


def test_refuses_output_equal_to_source(tmp_path: Path) -> None:
    src = _write_rgb_las(tmp_path / "src.las")
    with pytest.raises(ValueError):
        prepare.prepare_sample(src, src, target_points=100)


def test_cli_prints_summary_json(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    src = _write_rgb_las(tmp_path / "src.las", n=2_000)
    out = tmp_path / "sample.las"
    prov = tmp_path / "prov.json"
    rc = prepare.main(
        [
            "--source", str(src),
            "--output", str(out),
            "--target-points", "500",
            "--provenance", str(prov),
        ]
    )
    assert rc == 0
    captured = capsys.readouterr().out.splitlines()
    summary_line = [ln for ln in captured if ln.startswith("SUMMARY ")]
    assert len(summary_line) == 1
    data = json.loads(summary_line[0][len("SUMMARY "):])
    assert data["output_point_count"] == 500
    assert Path(data["output_path"]) == out
    assert data["sample_sha256"]
    assert prov.is_file()


# ---------------------------------------------------------------------------
# VISUAL-003: ROI corridor mode
# ---------------------------------------------------------------------------

_POLE_A = (-66.3449, 2.2295)
_POLE_B = (-25.3627, 20.4183)


def _write_roi_corridor_las(path: Path, n_inside: int, n_outside: int = 3) -> Path:
    """Points lying exactly on the A-B segment (inside) plus far points (outside).

    Raw X/Y/Z/R/G/B derive deterministically from the point index.
    """
    header = laspy.LasHeader(point_format=7, version="1.4")
    header.scales = np.array([0.0001, 0.0001, 0.0001])
    header.offsets = np.array([-100.0, -50.0, 0.0])
    ax, ay = _POLE_A
    bx, by = _POLE_B
    outside = [(-160.0, 120.0), (60.0, -100.0), (5.0, 90.0)]
    total = n_inside + n_outside
    x = np.empty(total)
    y = np.empty(total)
    for i in range(n_inside):
        t = i / (n_inside - 1) if n_inside > 1 else 0.0
        x[i] = ax + t * (bx - ax)
        y[i] = ay + t * (by - ay)
    for j, (ox, oy) in enumerate(outside[:n_outside]):
        x[n_inside + j] = ox
        y[n_inside + j] = oy
    z = np.zeros(total)
    red = np.arange(total, dtype=np.uint16)
    green = np.arange(total, dtype=np.uint16) * 2
    blue = np.arange(total, dtype=np.uint16) * 3
    record = laspy_record.ScaleAwarePointRecord(
        np.zeros(total, dtype=header.point_format.dtype()),
        header.point_format,
        header.scales,
        header.offsets,
    )
    las = laspy.LasData(header)
    las.points = record
    las.x = x
    las.y = y
    las.z = z
    las.red = red
    las.green = green
    las.blue = blue
    las.write(str(path))
    return path


def test_roi_keeps_only_corridor_points(tmp_path: Path) -> None:
    src = _write_roi_corridor_las(tmp_path / "src.las", n_inside=10, n_outside=3)
    out = tmp_path / "roi.las"
    summary = prepare.prepare_sample(
        src, out, 1_000_000,
        roi_pole_a=_POLE_A, roi_pole_b=_POLE_B,
        roi_half_width=25.0, roi_extension=15.0,
    )
    assert summary["roi_raw_point_count"] == 10
    assert summary["output_point_count"] == 10
    src_read = laspy.read(str(src))
    out_read = laspy.read(str(out))
    assert out_read.header.point_count == 10
    # The 10 inside points (source indices 0..9) are preserved exactly.
    for j in range(10):
        assert int(out_read.X[j]) == int(src_read.X[j])
        assert int(out_read.Y[j]) == int(src_read.Y[j])
        assert int(out_read.Z[j]) == int(src_read.Z[j])
        assert int(out_read.red[j]) == int(src_read.red[j])
        assert int(out_read.green[j]) == int(src_read.green[j])
        assert int(out_read.blue[j]) == int(src_read.blue[j])
    # The 3 outside points (source red values 10,11,12) are excluded.
    out_reds = {int(v) for v in np.asarray(out_read.red)}
    assert {10, 11, 12}.isdisjoint(out_reds)


def test_roi_source_immutable(tmp_path: Path) -> None:
    src = _write_roi_corridor_las(tmp_path / "src.las", n_inside=5)
    before = src.stat()
    digest_before = hashlib.sha256(src.read_bytes()).hexdigest()
    prepare.prepare_sample(
        src, tmp_path / "roi.las", 1_000_000,
        roi_pole_a=_POLE_A, roi_pole_b=_POLE_B,
    )
    after = src.stat()
    assert after.st_size == before.st_size
    assert after.st_mtime_ns == before.st_mtime_ns
    assert hashlib.sha256(src.read_bytes()).hexdigest() == digest_before


def test_roi_unicode_path(tmp_path: Path) -> None:
    sub = tmp_path / _SRC_DIR_NAME
    sub.mkdir()
    src = _write_roi_corridor_las(sub / "src.las", n_inside=6)
    out = sub / "roi.las"
    prov = sub / "prov.json"
    summary = prepare.prepare_sample(
        src, out, 1_000_000,
        roi_pole_a=_POLE_A, roi_pole_b=_POLE_B,
        provenance_path=prov,
    )
    assert out.is_file()
    assert prov.is_file()
    assert summary["sample_sha256"]
    data = json.loads(prov.read_text(encoding="utf-8"))
    assert data["protected_source"]["immutability"] == "PASS"
    assert data["roi"]["half_width"] == 25.0
    assert data["sample"]["point_count"] == 6


def test_roi_subsamples_when_over_cap(tmp_path: Path) -> None:
    # 200 inside points with a 50-point cap -> deterministic stride-4 in-ROI
    # subsample keeping source-ROI order 0,4,8,...
    src = _write_roi_corridor_las(tmp_path / "src.las", n_inside=200, n_outside=0)
    out = tmp_path / "roi.las"
    summary = prepare.prepare_sample(
        src, out, 50,
        roi_pole_a=_POLE_A, roi_pole_b=_POLE_B,
        roi_half_width=25.0, roi_extension=15.0,
    )
    assert summary["roi_raw_point_count"] == 200
    assert summary["roi_subsample_stride"] == 4
    assert summary["output_point_count"] == 50
    assert summary["output_point_count"] <= prepare.MAX_P3A_SAMPLE_POINTS
    src_read = laspy.read(str(src))
    out_read = laspy.read(str(out))
    for j, src_idx in enumerate(range(0, 200, 4)):
        assert int(out_read.X[j]) == int(src_read.X[src_idx])
        assert int(out_read.red[j]) == int(src_read.red[src_idx])


def test_roi_requires_both_poles(tmp_path: Path) -> None:
    src = _write_roi_corridor_las(tmp_path / "src.las", n_inside=3)
    with pytest.raises(ValueError):
        prepare.prepare_sample(
            src, tmp_path / "roi.las", 100,
            roi_pole_a=_POLE_A, roi_pole_b=None,
        )


def test_roi_requires_distinct_poles(tmp_path: Path) -> None:
    src = _write_roi_corridor_las(tmp_path / "src.las", n_inside=3)
    with pytest.raises(ValueError):
        prepare.prepare_sample(
            src, tmp_path / "roi.las", 100,
            roi_pole_a=_POLE_A, roi_pole_b=_POLE_A,
        )
