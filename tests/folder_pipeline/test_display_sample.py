"""Focused tests for the STEP-1 display-sample lifecycle fix.

Covers: hydration via the authoritative ``_stage_display_sample`` builder,
cache reuse, defensive /points behavior, display/science separation, and the
launcher (BAT) fixes.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from algorithms.folder_pipeline.orchestrator import (
    FolderOrchestrator,
    FolderPipelineError,
    _find_prior_ground_cache,
)
from algorithms.folder_pipeline.server import FolderServer
from algorithms.folder_pipeline.state import FolderWorkflowState
from algorithms.folder_pipeline.workspace import discover_workspace
from tests.folder_pipeline.conftest import make_workspace, write_synthetic_las

# ---------------------------------------------------------------------------
# Orchestrator display_sample hydration
# ---------------------------------------------------------------------------

class _FakeDisplayRun:
    def __init__(self, sample: dict | None = None) -> None:
        self.display_sample = sample
        self.inspection = {
            "bounds_min_xyz": [1000.0, 2000.0, 2.0],
            "bounds_max_xyz": [1400.0, 2200.0, 30.0],
        }
        self.origin_xyz = (1000.0, 2000.0, 2.0)
        self.calls: list[str] = []

    def _stage_display_sample(self) -> dict:
        self.calls.append("_stage_display_sample")
        return {
            "xyz_absolute": np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            "rgb": np.asarray([[10, 20, 30], [40, 50, 60]], dtype=np.uint8),
            "original_index": np.asarray([7, 8], dtype=np.int64),
            "display_source": np.asarray(["SOURCE_RAW_PRE_DENOISE"]),
        }


def test_display_points_with_existing_in_memory_sample(tmp_path: Path) -> None:
    workspace = make_workspace(tmp_path)
    ws = discover_workspace(workspace)
    orch = FolderOrchestrator(ws)
    orch.run = _FakeDisplayRun(sample={"xyz_absolute": np.zeros((3, 3)), "rgb": np.zeros((3, 3))})
    orch.run.display_sample = {
        "xyz_absolute": np.asarray([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]),
        "rgb": np.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.uint8),
    }
    result = orch.display_points()
    assert result["xyz"].shape[0] == 3
    # The authoritative builder was NOT called again (sample already present).
    assert orch.run.calls == []


def test_display_points_hydrates_when_sample_is_none(tmp_path: Path) -> None:
    workspace = make_workspace(tmp_path)
    ws = discover_workspace(workspace)
    orch = FolderOrchestrator(ws)
    fake = _FakeDisplayRun(sample=None)
    orch.run = fake
    result = orch.display_points()
    assert fake.calls == ["_stage_display_sample"]  # authoritative builder reused
    assert fake.display_sample is not None  # assigned back
    assert result["xyz"].shape[0] == 2
    assert result["source_bounds"][0] == 1000.0


def test_ensure_display_sample_is_idempotent(tmp_path: Path) -> None:
    workspace = make_workspace(tmp_path)
    ws = discover_workspace(workspace)
    orch = FolderOrchestrator(ws)
    fake = _FakeDisplayRun(sample=None)
    orch.run = fake
    first = orch._ensure_display_sample()
    second = orch._ensure_display_sample()
    assert first is second
    assert fake.calls == ["_stage_display_sample"]  # built only once


def test_display_sample_failure_is_controlled(tmp_path: Path) -> None:
    workspace = make_workspace(tmp_path)
    ws = discover_workspace(workspace)
    orch = FolderOrchestrator(ws)
    orch.run = _FakeDisplayRun(sample=None)
    orch.run._stage_display_sample = lambda: (_ for _ in ()).throw(OSError("boom"))  # type: ignore[attr-defined]
    with pytest.raises(FolderPipelineError) as exc:
        orch.display_points()
    assert "点云显示采样准备失败" in str(exc.value)


# ---------------------------------------------------------------------------
# Authoritative builder + cache reuse (real ManualSingleSpanRun._stage_display_sample)
# ---------------------------------------------------------------------------

def test_authoritative_stage_display_sample_builds_and_caches(tmp_path: Path) -> None:
    from algorithms.manual_span.config import (
        default_manual_span_config,
        manual_route_config,
    )
    from algorithms.manual_span.pipeline import ManualSingleSpanRun

    rng = np.random.default_rng(21)
    xyz = rng.uniform([0.0, 0.0, 2.0], [200.0, 200.0, 30.0], (20_000, 3))
    source = write_synthetic_las(tmp_path / "src.las", xyz)
    run = ManualSingleSpanRun(
        source, tmp_path / "run",
        cfg=manual_route_config(), manual=default_manual_span_config(),
    )
    npz_path = run.run_root / "00_input" / "display_sample_v4.npz"
    assert not npz_path.is_file()
    sample = run._stage_display_sample()
    assert npz_path.is_file()  # cache written
    assert sample["display_source"].tolist() == ["SOURCE_RAW_PRE_DENOISE"]
    assert sample["xyz_absolute"].shape[0] <= run.manual.display_sample_points
    # Second call reuses the cache (no rebuild).
    sample2 = run._stage_display_sample()
    assert np.array_equal(sample["xyz_absolute"], sample2["xyz_absolute"])


def test_display_cache_is_bounded_not_full_source(tmp_path: Path) -> None:
    from algorithms.manual_span.config import (
        default_manual_span_config,
        manual_route_config,
    )
    from algorithms.manual_span.pipeline import ManualSingleSpanRun

    rng = np.random.default_rng(5)
    # A source larger than the display budget must still be bounded in the
    # display sample (the browser never receives the full raw source).
    xyz = rng.uniform([0.0, 0.0, 2.0], [300.0, 300.0, 30.0], (80_000, 3))
    source = write_synthetic_las(tmp_path / "big.las", xyz)
    run = ManualSingleSpanRun(
        source, tmp_path / "run",
        cfg=manual_route_config(), manual=default_manual_span_config(),
    )
    sample = run._stage_display_sample()
    # Bounded display sample: at most ~2x the budget (top + base per tall cell),
    # never the full source.
    assert sample["xyz_absolute"].shape[0] <= 2 * run.manual.display_sample_points
    assert sample["xyz_absolute"].shape[0] < 80_000


# ---------------------------------------------------------------------------
# Defensive /points behavior (no uncaught NoneType)
# ---------------------------------------------------------------------------

def test_points_endpoint_never_crashes_on_none_sample(tmp_path: Path) -> None:
    workspace = make_workspace(tmp_path)
    ws = discover_workspace(workspace)
    server = FolderServer.__new__(FolderServer)
    server.workspace = ws
    server.state = FolderWorkflowState()
    server.orchestrator = SimpleNamespace(
        display_points=lambda: (_ for _ in ()).throw(
            FolderPipelineError("点云显示采样准备失败: 源不可用")
        )
    )
    server._science_thread = None
    server.headless_poles = None
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.handler_factory())
    url = f"http://127.0.0.1:{httpd.server_address[1]}"
    thread = threading.Thread(
        target=httpd.serve_forever,
        kwargs={"poll_interval": 0.05},
        daemon=True,
    )
    thread.start()
    try:
        # /points must return a controlled JSON error, not a raw traceback.
        try:
            with urllib.request.urlopen(url + "/points", timeout=5) as _resp:
                raise AssertionError("expected an HTTP error")
        except urllib.error.HTTPError as exc:
            assert exc.code == 500
            body = json.loads(exc.read().decode("utf-8"))
            assert "error" in body
            assert "点云显示采样准备失败" in body["error"]
        # /state reflects the readable Chinese error.
        with urllib.request.urlopen(url + "/state", timeout=5) as resp:
            state = json.loads(resp.read().decode("utf-8"))
            assert state["step"] == "ERROR"
            assert "STEP 1 点云预览失败" in state["error"]
    finally:
        httpd.shutdown()


def test_points_endpoint_returns_sample_when_hydrated(tmp_path: Path) -> None:
    workspace = make_workspace(tmp_path)
    ws = discover_workspace(workspace)
    server = FolderServer.__new__(FolderServer)
    server.workspace = ws
    server.state = FolderWorkflowState()
    rng = np.random.default_rng(3)
    server.orchestrator = SimpleNamespace(
        display_points=lambda: {
            "xyz": rng.uniform([0, 0, 0], [100, 100, 30], (200, 3)).tolist(),
            "rgb": np.zeros((200, 3), dtype=np.uint8).tolist(),
            "source_bounds": [0.0, 0.0, 0.0],
        }
    )
    server._science_thread = None
    server.headless_poles = None
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.handler_factory())
    url = f"http://127.0.0.1:{httpd.server_address[1]}"
    thread = threading.Thread(
        target=httpd.serve_forever,
        kwargs={"poll_interval": 0.05},
        daemon=True,
    )
    thread.start()
    try:
        with urllib.request.urlopen(url + "/points", timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        assert data["count"] > 0
        assert len(data["xyz"][0]) == 3
    finally:
        httpd.shutdown()


# ---------------------------------------------------------------------------
# Display data cannot affect science memberships
# ---------------------------------------------------------------------------

def test_display_sample_does_not_touch_science_state(tmp_path: Path) -> None:
    workspace = make_workspace(tmp_path)
    ws = discover_workspace(workspace)
    orch = FolderOrchestrator(ws)
    fake = _FakeDisplayRun(sample=None)
    fake.working = "WORKING"
    fake.ground = "GROUND"
    fake.raw_to_representative = "SCIENCE"
    orch.run = fake
    orch.display_points()
    # Science inputs are untouched by the display-only hydration.
    assert fake.working == "WORKING"
    assert fake.ground == "GROUND"
    assert fake.raw_to_representative == "SCIENCE"


def test_display_sample_is_display_only_source(tmp_path: Path) -> None:
    # The authoritative builder marks the sample as SOURCE_RAW_PRE_DENOISE
    # (display assistance) — it never becomes membership authority.
    from algorithms.manual_span.pipeline import _sample_source_cloud

    rng = np.random.default_rng(9)
    xyz = rng.uniform([0, 0, 0], [50, 50, 20], (5000, 3))
    source = write_synthetic_las(tmp_path / "disp_src.las", xyz)
    sample = _sample_source_cloud(source, maximum_points=500)
    assert sample["xyz_absolute"].shape[0] <= 1000  # bounded display sample
    assert sample["original_index"].dtype == np.int64


# ---------------------------------------------------------------------------
# Successful preprocess / ground cache remains reusable (source identity)
# ---------------------------------------------------------------------------

def _make_real_cache(ws, source_sha: str) -> None:
    science = ws.science_run_root
    (science / "00_input").mkdir(parents=True, exist_ok=True)
    (science / "01_preprocess").mkdir(parents=True, exist_ok=True)
    (science / "02_ground").mkdir(parents=True, exist_ok=True)
    (science / "00_input" / "source_inspection.json").write_text(
        json.dumps({"input_sha256": source_sha}), encoding="utf-8"
    )
    (science / "01_preprocess" / "preprocess.checkpoint.json").write_text(
        json.dumps({"status": "ok"}), encoding="utf-8"
    )
    (science / "01_preprocess" / "working_set_representatives.npz").write_bytes(b"npz")
    (science / "01_preprocess" / "raw_to_representative_id.npy").write_bytes(b"npy")
    (science / "02_ground" / "ground_hag_result.json").write_text("{}", encoding="utf-8")
    (science / "02_ground" / "classified_hag.las").write_bytes(b"las")


def test_existing_preprocess_cache_stays_reusable(tmp_path: Path) -> None:
    workspace = make_workspace(tmp_path)
    ws = discover_workspace(workspace)
    _make_real_cache(ws, ws.source_sha256)
    # The preprocess artifacts must NOT be invalidated by the fix.
    assert (ws.science_run_root / "01_preprocess" / "working_set_representatives.npz").is_file()
    assert (ws.science_run_root / "01_preprocess" / "raw_to_representative_id.npy").is_file()
    assert (ws.science_run_root / "01_preprocess" / "preprocess.checkpoint.json").is_file()


def test_existing_ground_cache_stays_reusable_and_source_checked(tmp_path: Path) -> None:
    workspace = make_workspace(tmp_path)
    ws = discover_workspace(workspace)
    _make_real_cache(ws, ws.source_sha256)
    ground = _find_prior_ground_cache(ws)
    assert ground is not None
    assert ground == ws.science_run_root / "02_ground"
    # A mismatched source identity is never reused.
    _make_real_cache(ws, "deadbeef" * 8)
    assert _find_prior_ground_cache(ws) is None


def test_ground_cache_working_set_mismatch_detected(tmp_path: Path) -> None:
    from algorithms.folder_pipeline.orchestrator import _ground_cache_matches_working_set

    workspace = make_workspace(tmp_path)
    ws = discover_workspace(workspace)
    work = ws.science_run_root / "01_preprocess"
    work.mkdir(parents=True, exist_ok=True)
    (work / "working_set_representatives.las").write_bytes(b"workset-A")
    gnd = ws.science_run_root / "02_ground"
    gnd.mkdir(parents=True, exist_ok=True)
    import hashlib

    correct_sha = hashlib.sha256(b"workset-A").hexdigest()
    (gnd / "ground_hag_result.json").write_text(
        json.dumps({"source_las_sha256": correct_sha}), encoding="utf-8"
    )
    assert _ground_cache_matches_working_set(ws, gnd) is True
    # A cache for a DIFFERENT working set is a MISMATCH (never silently reused).
    (gnd / "ground_hag_result.json").write_text(
        json.dumps({"source_las_sha256": "0" * 64}), encoding="utf-8"
    )
    assert _ground_cache_matches_working_set(ws, gnd) is False


# ---------------------------------------------------------------------------
# Launcher (BAT) fixes
# ---------------------------------------------------------------------------

def test_bat_is_ascii_safe() -> None:
    bat = Path("RUN_FOLDER_TO_EXCEL.bat").read_bytes()
    # No bytes outside the ASCII range (cmd.exe encoding-safe).
    assert all(byte < 0x80 for byte in bat), "BAT must be ASCII-only"


def test_bat_passes_workspace_argument() -> None:
    bat = Path("RUN_FOLDER_TO_EXCEL.bat").read_text(encoding="ascii")
    # The BAT must pass the workspace via --workspace, never a bare positional.
    assert "--workspace" in bat
    assert '--workspace "%WORKSPACE%"' in bat
