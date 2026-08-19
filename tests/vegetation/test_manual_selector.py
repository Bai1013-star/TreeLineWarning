"""Focused tests for the manual pole selector (RGB conversion, click state, server).

Covers the Owner-required selector pieces without touching the expensive
downstream pipeline: LAS RGB display conversion (8-bit / 16-bit / fallback),
the top-down XY click state machine (Pole A -> Pole B -> confirm -> reset),
server persistence into the manual selection, source coordinate preservation
and stale-source rejection.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import laspy
import numpy as np
import pytest
from algorithms.manual_span.config import default_manual_span_config
from algorithms.manual_span.pipeline import (
    _normalize_source_rgb,
    _sample_preprocessed_cloud,
    _sample_source_cloud,
)
from algorithms.manual_span.server import (
    WorkflowState,
    _make_handler_factory,
    _selector_html,
)
from algorithms.manual_span.span import manual_span_selection_payload

# ---------------------------------------------------------------------------
# RGB display conversion
# ---------------------------------------------------------------------------


def test_rgb_8bit_passthrough() -> None:
    rgb = np.asarray([[10, 20, 30], [255, 0, 128]], dtype=np.uint16)
    out = _normalize_source_rgb(rgb)
    assert out.dtype == np.uint8
    assert out.tolist() == [[10, 20, 30], [255, 0, 128]]


def test_rgb_16bit_scaled_into_field() -> None:
    # 8-bit colours stored as value*256 in a uint16 field (the real LAS case).
    rgb = np.asarray([[11 * 256, 20 * 256, 30 * 256]], dtype=np.uint16)
    out = _normalize_source_rgb(rgb)
    assert out.tolist() == [[11, 20, 30]]


def test_rgb_true_16bit_scales() -> None:
    # Values that are not multiples of 256 -> true 16-bit, linear scale.
    rgb = np.asarray([[0, 65535, 32768]], dtype=np.uint16)
    out = _normalize_source_rgb(rgb)
    assert out[0, 1] == 255
    assert 0 <= out[0, 2] <= 255


def test_rgb_empty() -> None:
    out = _normalize_source_rgb(np.empty((0, 3), dtype=np.uint16))
    assert out.shape == (0, 3)


# ---------------------------------------------------------------------------
# Spatial sample preserves coordinates / RGB
# ---------------------------------------------------------------------------


def _write_synthetic_las(path: Path, xyz: np.ndarray, rgb: np.ndarray | None) -> Path:
    header = laspy.LasHeader(point_format=7, version="1.4")
    header.scales = np.asarray((0.001, 0.001, 0.001))
    header.offsets = np.asarray((0.0, 0.0, 0.0))
    las = laspy.LasData(header)
    las.x = xyz[:, 0]
    las.y = xyz[:, 1]
    las.z = xyz[:, 2]
    if rgb is not None:
        las.red = rgb[:, 0].astype(np.uint16)
        las.green = rgb[:, 1].astype(np.uint16)
        las.blue = rgb[:, 2].astype(np.uint16)
    las.write(path)
    return path


def test_spatial_sample_preserves_real_coords_and_color(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    xyz = rng.uniform([1000.0, 2000.0, 5.0], [1100.0, 2100.0, 40.0], (50_000, 3))
    # Dense pole column: many points at one XY cell.
    pole = np.column_stack(
        (
            np.full(5000, 1050.0) + rng.normal(0, 0.05, 5000),
            np.full(5000, 2050.0) + rng.normal(0, 0.05, 5000),
            np.linspace(6.0, 30.0, 5000),
        )
    )
    all_xyz = np.concatenate((xyz, pole), axis=0)
    rgb8 = rng.integers(0, 255, (len(all_xyz), 3))
    rgb16 = (rgb8.astype(np.uint16) * 256).astype(np.uint16)
    source = _write_synthetic_las(tmp_path / "synthetic.las", all_xyz, rgb16)
    sample = _sample_source_cloud(source, maximum_points=20_000)
    # top (+base for tall cells) keeps the sample bounded near the cell budget.
    assert sample["xyz_absolute"].shape[0] <= 40_000
    assert sample["xyz_absolute"].shape[0] > 0
    # Coordinates stay in the real LAS frame.
    assert np.all(sample["xyz_absolute"][:, 0] >= 999.0)
    assert np.all(sample["xyz_absolute"][:, 0] <= 1101.0)
    # The pole cell is retained (dense vertical structure).
    pole_rows = np.flatnonzero(
        (np.abs(sample["xyz_absolute"][:, 0] - 1050.0) < 0.5)
        & (np.abs(sample["xyz_absolute"][:, 1] - 2050.0) < 0.5)
    )
    assert len(pole_rows) >= 1
    # RGB normalized to 8-bit and not black.
    assert sample["rgb"].dtype == np.uint8
    assert np.mean(sample["rgb"]) > 10


def test_spatial_sample_no_rgb_falls_back_to_zero_palette(tmp_path: Path) -> None:
    rng = np.random.default_rng(2)
    xyz = rng.uniform([0.0, 0.0, 0.0], [100.0, 100.0, 30.0], (20_000, 3))
    source = _write_synthetic_las(tmp_path / "nocolor.las", xyz, None)
    sample = _sample_source_cloud(source, maximum_points=5_000)
    assert sample["rgb"].shape == (len(sample["xyz_absolute"]), 3)


# ---------------------------------------------------------------------------
# Server click state machine + persistence
# ---------------------------------------------------------------------------


class _FakeRun:
    """Minimal stand-in exposing what the selector handler touches."""

    def __init__(self, display_sample: dict, run_root: Path | None = None) -> None:
        self.display_sample = display_sample
        self.run_root = run_root
        self.manual = default_manual_span_config()
        from core.config.fast_baseline import default_fast_baseline_config

        self.cfg = default_fast_baseline_config()
        self.origin_xyz = (0.0, 0.0, 0.0)
        self.source_sha256 = "src-sha"
        self.source_id = "src-id"
        self.inspection = {
            "bounds_min_xyz": [0.0, 0.0, 0.0],
            "bounds_max_xyz": [400.0, 400.0, 30.0],
        }
        self.pole_a = None
        self.pole_b = None
        self.span = None
        self.pending_clicks: tuple | None = None
        self._local_raw = None

    def local_raw_for_xy(self, x, y, radius=None):  # noqa: ANN001
        if self._local_raw is None:
            rng = np.random.default_rng(5)
            z = np.linspace(1.0, 12.0, 200)
            self._local_raw = {
                "xyz_absolute": np.column_stack(
                    (np.full_like(z, x) + rng.normal(0, 0.1, len(z)),
                     np.full_like(z, y) + rng.normal(0, 0.1, len(z)),
                     z)
                ),
                "rgb": np.full((len(z), 3), 200, dtype=np.uint8),
                "original_index": np.arange(len(z), dtype=np.int64),
            }
        return self._local_raw

    def _refine_pole(self, label, click_xyz):  # noqa: ANN001
        from algorithms.manual_span.span import refine_pole_from_raw_cloud

        return refine_pole_from_raw_cloud(
            label, click_xyz, self.local_raw_for_xy(click_xyz[0], click_xyz[1])["xyz_absolute"],
            self.origin_xyz, self.cfg, self.manual,
        )

    def refine_and_build_span(self, click_a, click_b):  # noqa: ANN001
        self.pending_clicks = (tuple(click_a), tuple(click_b))
        from algorithms.manual_span.span import ManualRefinedPole

        self.pole_a = ManualRefinedPole(
            label="A", clicked_xyz_absolute_m=click_a,
            base_xyz_absolute_m=click_a,
            top_xyz_absolute_m=(click_a[0], click_a[1], click_a[2] + 8.0),
            base_xyz_local_m=(0.0, 0.0, 0.0), top_xyz_local_m=(0.0, 0.0, 8.0),
            height_m=8.0, xy_footprint_m=0.5, median_verticality=0.9, median_linearity=0.6,
            representative_count=10, ground_support_reps=3, defensible=True, reason=None,
        )
        self.pole_b = ManualRefinedPole(
            label="B", clicked_xyz_absolute_m=click_b,
            base_xyz_absolute_m=click_b,
            top_xyz_absolute_m=(click_b[0], click_b[1], click_b[2] + 8.0),
            base_xyz_local_m=(0.0, 0.0, 0.0), top_xyz_local_m=(0.0, 0.0, 8.0),
            height_m=8.0, xy_footprint_m=0.5, median_verticality=0.9, median_linearity=0.6,
            representative_count=10, ground_support_reps=3, defensible=True, reason=None,
        )
        from algorithms.poles.topology import SpanRecord

        self.span = SpanRecord(
            line_id=0, line_name="ManualLine", span_id=0, span_name="ManualLine-Span001",
            start_pole_candidate_id=-1, end_pole_candidate_id=-2,
            start_xyz_absolute_m=click_a, end_xyz_absolute_m=click_b,
            start_pole_top_z_m=click_a[2] + 8.0, end_pole_top_z_m=click_b[2] + 8.0,
            length_m=100.0, direction_xy=(1.0, 0.0, 0.0), half_width_m=9.0, halo_m=6.0,
            core_bounds_min_xyz=(0.0, 0.0, 0.0), core_bounds_max_xyz=(100.0, 9.0, 30.0),
            halo_bounds_min_xyz=(0.0, 0.0, 0.0), halo_bounds_max_xyz=(106.0, 9.0, 30.0),
            quality="MANUAL_CONFIRMED_SPAN",
        )
        from algorithms.manual_span.span import manual_span_selection_payload

        self.selection_payload = manual_span_selection_payload(
            self.pole_a, self.pole_b, self.span, "src-sha", "src-id", (0.0, 0.0, 0.0),
            selection_version="manual-single-span-pipeline-v1",
        )


def _serve(run: _FakeRun, state: WorkflowState):
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _make_handler_factory(run, state))  # type: ignore[arg-type]
    thread = threading.Thread(
        target=httpd.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True
    )
    thread.start()
    return httpd, f"http://127.0.0.1:{httpd.server_address[1]}/"


def _get(url: str, path: str) -> dict:
    with urllib.request.urlopen(url + path, timeout=5) as response:
        return json.loads(response.read().decode("utf-8"))


def _post(url: str, path: str, payload: dict) -> dict:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url + path, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=5) as response:
        return json.loads(response.read().decode("utf-8"))


def _post_error(url: str, path: str, payload: dict) -> str:
    try:
        _post(url, path, payload)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8")
        return json.loads(body).get("error", "")
    raise AssertionError(f"expected HTTP error for {path}")


def _sample_points() -> dict:
    rng = np.random.default_rng(3)
    xyz = rng.uniform([0.0, 0.0, 5.0], [200.0, 200.0, 30.0], (2000, 3))
    rgb = rng.integers(0, 255, (2000, 3)).astype(np.uint8)
    return {"xyz_absolute": xyz, "rgb": rgb}


def test_selector_click_state_machine() -> None:
    run = _FakeRun(_sample_points())
    state = WorkflowState()
    httpd, url = _serve(run, state)
    try:
        s = _get(url, "/state")
        assert s["state"] == "SELECTING"
        assert s["selection_mode"] == "IDLE"
        assert s["clicks"] == {"A": {}, "B": {}}
        assert "先按选择杆 A" in _selector_status_text()
        assert _post_error(url, "/click", {"pole": "A", "x": 50.0, "y": 60.0})
        # Explicitly enter Pole A mode, then click it.
        s = _post(url, "/mode", {"pole": "A"})
        assert s["selection_mode"] == "SELECT_A"
        s = _post(url, "/click", {"pole": "A", "x": 50.0, "y": 60.0})
        assert s["clicks"]["A"]["x"] == 50.0
        assert s["clicks"]["B"] == {}
        assert s["selection_mode"] == "IDLE"
        # Explicitly enter Pole B mode, then click it.
        s = _post(url, "/mode", {"pole": "B"})
        assert s["selection_mode"] == "SELECT_B"
        s = _post(url, "/click", {"pole": "B", "x": 150.0, "y": 160.0})
        assert s["clicks"]["B"]["x"] == 150.0
        # Confirm -> refine_and_build_span invoked, state CONFIRMED
        s = _post(url, "/confirm", {})
        assert s["state"] == "CONFIRMED"
        assert run.pending_clicks is not None
        assert run.pending_clicks[0][0] == 50.0
        assert run.pending_clicks[1][0] == 150.0
        assert "有效" in _post_error(url, "/confirm", {})
        assert "不能重新选择" in _post_error(url, "/mode", {"pole": "A"})
        # Reset clears both
        s = _post(url, "/reselect", {})
        assert s["clicks"]["A"] == {}
        assert s["clicks"]["B"] == {}
        assert s["state"] == "SELECTING"
        assert s["selection_mode"] == "IDLE"
        assert run.pole_a is None and run.pole_b is None and run.span is None
    finally:
        httpd.shutdown()


def test_selector_click_persists_real_coordinates() -> None:
    run = _FakeRun(_sample_points())
    state = WorkflowState()
    httpd, url = _serve(run, state)
    try:
        click_a = (104.5, 205.0, 18.2)
        click_b = (214.0, 305.5, 20.1)
        _post(url, "/mode", {"pole": "A"})
        _post(url, "/click", {"pole": "A", "x": click_a[0], "y": click_a[1], "z": click_a[2]})
        _post(url, "/mode", {"pole": "B"})
        _post(url, "/click", {"pole": "B", "x": click_b[0], "y": click_b[1], "z": click_b[2]})
        _post(url, "/confirm", {})
        # The refined span must be built from the SAME real-LAS coordinates.
        assert run.pending_clicks[0] == click_a
        assert run.pending_clicks[1] == click_b
    finally:
        httpd.shutdown()


def test_selection_binds_source_sha256() -> None:
    # The selection payload must record the exact source SHA for stale rejection.
    run = _FakeRun(_sample_points())
    run.refine_and_build_span((10.0, 10.0, 5.0), (110.0, 110.0, 6.0))
    payload = manual_span_selection_payload(
        run.pole_a, run.pole_b, run.span, "abcd1234", "src-id", (0.0, 0.0, 0.0),
        selection_version="manual-single-span-pipeline-v1",
    )
    assert payload["source_sha256"] == "abcd1234"
    assert payload["source_identity"] == "src-id"
    assert payload["coordinate_frame"] == "SOURCE_LAS_ABSOLUTE"
    # A stale selection (different source SHA) must not be accepted.
    stale = dict(payload)
    stale["source_sha256"] = "other-sha"
    assert stale["source_sha256"] != payload["source_sha256"]


def test_display_points_serve_real_frame() -> None:
    points = _sample_points()
    run = _FakeRun(points)
    state = WorkflowState()
    httpd, url = _serve(run, state)
    try:
        with urllib.request.urlopen(url + "/points", timeout=5) as response:
            data = json.loads(response.read().decode("utf-8"))
        assert data["count"] == len(points["xyz_absolute"])
        assert len(data["xyz"][0]) == 3
        assert len(data["rgb"][0]) == 3
    finally:
        httpd.shutdown()


def _selector_status_text() -> str:
    """Keep the server-side test readable without requiring a browser runtime."""

    return "请先按选择杆 A"


def test_reset_clears_pending_and_confirmed_persistence(tmp_path: Path) -> None:
    run = _FakeRun(_sample_points(), run_root=tmp_path / "run")
    (run.run_root / "03_selection").mkdir(parents=True)
    (
        run.run_root / "03_selection" / "manual_span_selection.json"
    ).write_text("old", encoding="utf-8")
    state = WorkflowState()
    httpd, url = _serve(run, state)
    try:
        _post(url, "/mode", {"pole": "A"})
        _post(url, "/click", {"pole": "A", "x": 20.0, "y": 30.0})
        pending = run.run_root / "03_selection" / "manual_selector_state.json"
        assert pending.is_file()
        s = _post(url, "/reselect", {})
        assert s["clicks"] == {"A": {}, "B": {}}
        assert not pending.exists()
        assert not (run.run_root / "03_selection" / "manual_span_selection.json").exists()
        assert run.selection_payload is None
        # A deliberate browser reload reads the same authoritative empty server state.
        assert _get(url, "/state")["clicks"] == {"A": {}, "B": {}}
    finally:
        httpd.shutdown()


def test_select_a_then_reset_and_select_a_b_then_reset(tmp_path: Path) -> None:
    run = _FakeRun(_sample_points(), run_root=tmp_path / "run")
    state = WorkflowState()
    httpd, url = _serve(run, state)
    try:
        _post(url, "/mode", {"pole": "A"})
        _post(url, "/click", {"pole": "A", "x": 10.0, "y": 10.0})
        _post(url, "/reselect", {})
        assert _get(url, "/state")["clicks"] == {"A": {}, "B": {}}
        _post(url, "/mode", {"pole": "A"})
        _post(url, "/click", {"pole": "A", "x": 10.0, "y": 10.0})
        _post(url, "/mode", {"pole": "B"})
        _post(url, "/click", {"pole": "B", "x": 110.0, "y": 10.0})
        s = _get(url, "/state")
        assert s["clicks"]["A"]["x"] == 10.0 and s["clicks"]["B"]["x"] == 110.0
        _post(url, "/reselect", {})
        assert _get(url, "/state")["clicks"] == {"A": {}, "B": {}}
    finally:
        httpd.shutdown()


def test_mode_and_source_bounds_fail_closed() -> None:
    run = _FakeRun(_sample_points())
    state = WorkflowState()
    httpd, url = _serve(run, state)
    try:
        # Pan/zoom do not call a click endpoint; an unarmed click is rejected.
        assert "先按选择杆" in _post_error(url, "/click", {"pole": "A", "x": 50.0, "y": 60.0})
        _post(url, "/mode", {"pole": "A"})
        assert "outside" in _post_error(url, "/click", {"pole": "A", "x": 999.0, "y": 60.0})
        assert "有效" in _post_error(url, "/confirm", {})
    finally:
        httpd.shutdown()


def test_selector_html_requires_explicit_modes_and_preserves_navigation() -> None:
    html = _selector_html()
    assert 'id="btn-select-a"' in html
    assert 'id="btn-select-b"' in html
    assert 'id="btn-reselect"' in html
    assert 'id="btn-full"' in html
    assert 'scrollZoom: true' in html
    # Primary RAW overview must be canvas ``scatter`` (NOT WebGL scattergl) so it
    # renders even when WebGL is unavailable/blocked in the browser.
    assert 'type: "scatter", mode: "markers"' in html
    assert 'scattergl' not in html
    assert 'selectionMode === "SELECT_A"' in html
    assert 'selectionMode === "SELECT_B"' in html
    assert 'let nextPole' not in html
    # Owner-required: native click anywhere on the plot converts to real-LAS XY
    # so a click NEAR a pole (not an exact point pixel) also selects.
    assert 'g2d.addEventListener("click"' in html
    # Zoom-safe: point clicks come through Plotly's own transformed coordinates.
    assert 'plotly_click' in html
    # Locked 1:1 XY aspect ratio (Y anchored to X).
    assert 'scaleanchor: "x", scaleratio: 1' in html


def test_selection_distance_and_clicked_xy_survive_navigation_events() -> None:
    run = _FakeRun(_sample_points())
    state = WorkflowState()
    httpd, url = _serve(run, state)
    try:
        _post(url, "/mode", {"pole": "A"})
        _post(url, "/click", {"pole": "A", "x": 12.25, "y": 42.75})
        # Simulated pan/zoom is a client-only Plotly operation; it must not
        # create a second click or alter the source-frame coordinates.
        s = _get(url, "/state")
        assert s["clicks"]["A"]["x"] == 12.25
        assert s["clicks"]["A"]["y"] == 42.75
        assert s["selection_distance_xy_m"] is None
        _post(url, "/mode", {"pole": "B"})
        _post(url, "/click", {"pole": "B", "x": 112.25, "y": 42.75})
        s = _get(url, "/state")
        assert s["selection_complete"] is True
        assert s["selection_distance_xy_m"] == 100.0
        assert _post_error(url, "/click", {"pole": "A", "x": 13.0, "y": 43.0})
    finally:
        httpd.shutdown()


def test_preprocessed_display_maps_source_rows_and_rgb(tmp_path: Path) -> None:
    xyz = np.asarray(
        [[10.0, 20.0, 2.0], [30.0, 40.0, 8.0], [50.0, 60.0, 12.0]],
        dtype=np.float64,
    )
    rgb8 = np.asarray([[11, 22, 33], [44, 55, 66], [77, 88, 99]], dtype=np.uint16)
    source = _write_synthetic_las(tmp_path / "display-source.las", xyz, rgb8)
    reps = tmp_path / "working_set_representatives.npz"
    np.savez(
        reps,
        xyz_absolute=xyz[[2, 0, 1]],
        original_index=np.asarray([2, 0, 1], dtype=np.int64),
    )
    working = SimpleNamespace(representatives_npz_path=str(reps))
    display = _sample_preprocessed_cloud(source, working, maximum_points=10)
    assert np.array_equal(display["xyz_absolute"], xyz[[2, 0, 1]])
    assert display["rgb"].tolist() == rgb8[[2, 0, 1]].tolist()
    assert display["display_source"].tolist() == ["PREPROCESSED_WEIGHTED_REPRESENTATIVES"]


# ---------------------------------------------------------------------------
# RAW-based refinement + local raw view + failure behavior
# ---------------------------------------------------------------------------


def _pole_cloud(center_x: float, center_y: float, height: float = 9.0) -> np.ndarray:
    rng = np.random.default_rng(7)
    z = np.linspace(0.5, height, 150)
    return np.column_stack(
        (
            np.full_like(z, center_x) + rng.normal(0, 0.08, len(z)),
            np.full_like(z, center_y) + rng.normal(0, 0.08, len(z)),
            z,
        )
    )


def test_refine_from_raw_cloud_success() -> None:
    from algorithms.manual_span.config import default_manual_span_config
    from algorithms.manual_span.span import refine_pole_from_raw_cloud
    from core.config.fast_baseline import default_fast_baseline_config

    raw = _pole_cloud(100.0, 200.0)
    pole = refine_pole_from_raw_cloud(
        "A", (100.0, 200.0, 5.0), raw, (0.0, 0.0, 0.0),
        default_fast_baseline_config(), default_manual_span_config(),
    )
    assert pole.defensible, pole.reason
    assert pole.refinement_diagnostics["vertical_extent_m"] > 7.0
    assert pole.refinement_diagnostics["refined_xy_absolute_m"][0] == pytest.approx(100.0, abs=0.5)
    assert pole.base_xyz_absolute_m[2] == pytest.approx(0.5, abs=0.2)


def test_refine_from_raw_cloud_insufficient_extent() -> None:
    from algorithms.manual_span.config import default_manual_span_config
    from algorithms.manual_span.span import refine_pole_from_raw_cloud
    from core.config.fast_baseline import default_fast_baseline_config

    rng = np.random.default_rng(3)
    # A DENSE low structure (e.g. a bush): many points but only ~1.5 m tall.
    flat = np.column_stack(
        (
            100.0 + rng.normal(0, 0.5, 4000),
            200.0 + rng.normal(0, 0.5, 4000),
            rng.uniform(0.0, 1.5, 4000),
        )
    )
    pole = refine_pole_from_raw_cloud(
        "B", (100.0, 200.0, 0.5), flat, (0.0, 0.0, 0.0),
        default_fast_baseline_config(), default_manual_span_config(),
    )
    assert not pole.defensible
    assert "INSUFFICIENT_VERTICAL_EXTENT" in (pole.reason or "")
    assert pole.refinement_diagnostics["vertical_extent_m"] < 5.0
    assert "vertical_extent_m" in pole.refinement_diagnostics


def test_refine_from_raw_cloud_too_sparse() -> None:
    from algorithms.manual_span.config import default_manual_span_config
    from algorithms.manual_span.span import refine_pole_from_raw_cloud
    from core.config.fast_baseline import default_fast_baseline_config

    sparse = np.asarray([[100.0, 200.0, 1.0], [100.1, 200.0, 1.2]], dtype=np.float64)
    pole = refine_pole_from_raw_cloud(
        "A", (100.0, 200.0, 1.0), sparse, (0.0, 0.0, 0.0),
        default_fast_baseline_config(), default_manual_span_config(),
    )
    assert not pole.defensible
    assert pole.reason == "LOCAL_POINT_CLOUD_TOO_SPARSE"


def test_local_points_endpoint_and_refine_endpoint() -> None:
    run = _FakeRun(_sample_points())
    state = WorkflowState()
    httpd, url = _serve(run, state)
    try:
        _post(url, "/mode", {"pole": "A"})
        _post(url, "/click", {"pole": "A", "x": 10.0, "y": 10.0})
        local = json.loads(
            urllib.request.urlopen(
                url + "/local_points?cx=10&cy=10&r=6", timeout=5
            ).read().decode("utf-8")
        )
        assert local["count"] > 0
        assert len(local["xyz"][0]) == 3
        # Refine endpoint reports a defensible pole with diagnostics.
        refined = _post(url, "/refine", {"pole": "A"})
        assert refined["defensible"] is True
        assert refined["refinement_diagnostics"]["vertical_extent_m"] > 7.0
    finally:
        httpd.shutdown()


def test_refine_failure_keeps_click_and_reports_reason() -> None:
    run = _FakeRun(_sample_points())
    run._local_raw = {
        "xyz_absolute": np.asarray(
            [[10.0, 10.0, 0.5], [11.0, 11.0, 1.0], [10.5, 10.5, 1.2]],
            dtype=np.float64,
        ),
        "rgb": np.full((3, 3), 200, dtype=np.uint8),
        "original_index": np.arange(3, dtype=np.int64),
    }
    state = WorkflowState()
    httpd, url = _serve(run, state)
    try:
        _post(url, "/mode", {"pole": "A"})
        _post(url, "/click", {"pole": "A", "x": 10.0, "y": 10.0})
        # Click remains visible in state.
        assert _get(url, "/state")["clicks"]["A"]["x"] == 10.0
        refined = _post(url, "/refine", {"pole": "A"})
        assert refined["defensible"] is False
        assert refined["reason"] == "LOCAL_POINT_CLOUD_TOO_SPARSE"
        assert "vertical_extent_m" in refined["refinement_diagnostics"]
        # Click is still present after failure -> immediate reselection possible.
        assert _get(url, "/state")["clicks"]["A"]["x"] == 10.0
    finally:
        httpd.shutdown()


def test_points_payload_reports_raw_display_source() -> None:
    run = _FakeRun({**_sample_points(), "display_source": np.asarray(["SOURCE_RAW_PRE_DENOISE"])})
    state = WorkflowState()
    httpd, url = _serve(run, state)
    try:
        data = json.loads(urllib.request.urlopen(url + "/points", timeout=5).read().decode("utf-8"))
        assert data["display_source"] == "SOURCE_RAW_PRE_DENOISE"
    finally:
        httpd.shutdown()
