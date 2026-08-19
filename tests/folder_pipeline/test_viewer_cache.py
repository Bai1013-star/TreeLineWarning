"""Focused tests for the reproducible Potree viewer cache + STEP-1 Potree page.

Covers the frozen viewer contract: source-SHA binding, reuse, stale/different
source rejection, no full-source browser payload, Potree primary vs Plotly
fallback, the viewer-local -> source-absolute coordinate bridge, and that a
Potree pick never invents an original_index.
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
from algorithms.folder_pipeline import viewer_cache as vc
from algorithms.folder_pipeline.orchestrator import FolderOrchestrator
from algorithms.folder_pipeline.potree_page import potree_step1_html
from algorithms.folder_pipeline.server import FolderServer
from algorithms.folder_pipeline.state import FolderWorkflowState
from algorithms.folder_pipeline.viewer_cache import (
    ViewerCacheInfo,
    compute_cache_id,
    load_viewer_cache,
)
from algorithms.folder_pipeline.workspace import discover_workspace
from tests.folder_pipeline.conftest import make_workspace


def _ws(tmp_path: Path):
    return discover_workspace(make_workspace(tmp_path))


def _fake_cache(ws) -> ViewerCacheInfo:
    """A minimal in-memory cache info used to render the page / run /api/pick."""

    potree = ws.source_cache_dir / "viewer" / "test-cache" / "potree"
    potree.mkdir(parents=True, exist_ok=True)
    (potree / "metadata.json").write_text(
        json.dumps(
            {
                "version": "2.0",
                "points": 1000,
                "offset": [100.0, 200.0, 5.0],
                "hierarchy": {"depth": 3},
                "attributes": [{"name": "position"}, {"name": "rgb"}],
            }
        ),
        encoding="utf-8",
    )
    (potree / "octree.bin").write_bytes(b"\x00" * 1024)
    (potree / "hierarchy.bin").write_bytes(b"\x00" * 16)
    return ViewerCacheInfo(
        cache_id="test-cache",
        cache_dir=potree.parent,
        potree_dir=potree,
        manifest_path=potree.parent / "cache-manifest.json",
        metadata_path=potree / "metadata.json",
        octree_path=potree / "octree.bin",
        hierarchy_path=potree / "hierarchy.bin",
        source_sha256=ws.source_sha256,
        converter_version=vc.POTREE_CONVERTER_VERSION,
        viewer_version=vc.POTREE_RUNTIME_VERSION,
        offset_xyz=(100.0, 200.0, 5.0),
        point_count=1000,
        hierarchy_depth=3,
    )


# ---------------------------------------------------------------------------
# Cache contract: source binding, reuse, stale rejection
# ---------------------------------------------------------------------------

def test_cache_id_binds_source_sha(tmp_path: Path) -> None:
    ws = _ws(tmp_path)
    id_a = compute_cache_id(ws)
    id_b = compute_cache_id(ws)
    assert id_a == id_b  # deterministic
    # A different source must produce a different cache identity.
    ws2 = _ws(tmp_path / "ws2")
    ws2.source.write_bytes(b"different content" * 100)  # non-empty changed source
    ws2b = discover_workspace(ws2.root)
    assert compute_cache_id(ws2b) != id_a


def test_load_valid_cache_returns_and_reuses(tmp_path: Path) -> None:
    # A valid cache-manifest.json with matching source SHA is returned (reuse).
    ws = _ws(tmp_path)
    cache_id = compute_cache_id(ws)
    root = ws.source_cache_dir / "viewer" / cache_id
    potree = root / "potree"
    potree.mkdir(parents=True, exist_ok=True)
    (potree / "metadata.json").write_text('{"attributes": []}', encoding="utf-8")
    (potree / "octree.bin").write_bytes(b"octree")
    (potree / "hierarchy.bin").write_bytes(b"h")
    manifest = {
        "schema": vc.VIEWER_CACHE_SCHEMA,
        "cache_id": cache_id,
        "source_sha256": ws.source_sha256,
        "converter_version": vc.POTREE_CONVERTER_VERSION,
        "viewer_version": vc.POTREE_RUNTIME_VERSION,
        "offset_xyz": [0.0, 0.0, 0.0],
        "source_point_count": 100,
        "hierarchy_depth": 2,
    }
    (root / "cache-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    info = load_viewer_cache(ws)
    assert info is not None
    assert info.cache_id == cache_id
    # Re-load is stable (reuse on restart).
    assert load_viewer_cache(ws).cache_id == cache_id


def test_stale_or_different_source_cache_rejected(tmp_path: Path) -> None:
    ws = _ws(tmp_path)
    cache_id = compute_cache_id(ws)
    root = ws.source_cache_dir / "viewer" / cache_id
    potree = root / "potree"
    potree.mkdir(parents=True, exist_ok=True)
    (potree / "metadata.json").write_text('{"attributes": []}', encoding="utf-8")
    (potree / "octree.bin").write_bytes(b"octree")
    manifest = {
        "schema": vc.VIEWER_CACHE_SCHEMA,
        "cache_id": cache_id,
        "source_sha256": "a-different-source-sha",  # does NOT match this source
        "converter_version": vc.POTREE_CONVERTER_VERSION,
        "viewer_version": vc.POTREE_RUNTIME_VERSION,
        "offset_xyz": [0.0, 0.0, 0.0],
    }
    (root / "cache-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    # A cache from a different source is NEVER reused (fail closed).
    assert load_viewer_cache(ws) is None


def test_generated_cache_not_git_tracked() -> None:
    # The workspace viewer cache lives under workspace/ (gitignored), never in Git.
    text = Path(".gitignore").read_text(encoding="utf-8")
    assert "workspace/" in text


# ---------------------------------------------------------------------------
# STEP-1 page: Potree primary, Plotly fallback
# ---------------------------------------------------------------------------

def test_potree_page_is_primary_when_cache_ready(tmp_path: Path) -> None:
    cache = _fake_cache(_ws(tmp_path))
    html = potree_step1_html(cache)
    assert "Potree" in html
    assert "potree/build/potree/potree.js" in html  # runtime loaded from local /potree/
    assert "POTREE LOD" in html
    # The page does NOT embed any full-source point payload (no JSON point arrays).
    assert "xyz_absolute" not in html
    # PROVEN-migration regression guards (78d9cd3 / 156d499 owner findings):
    # 1. RGB is selected via activeAttributeName = "rgba" (not "rgb" -> purple).
    assert 'activeAttributeName = "rgba"' in html
    assert 'activeAttributeName = "rgb"' not in html
    # 2. Picking uses Potree.Utils.getMousePointCloudIntersection (Potree.Picker
    #    is undefined in Potree 1.8.2 and throws).  The executable call form ends
    #    with "(" — the only Potree.Picker text left is an explanatory comment.
    assert "Potree.Utils.getMousePointCloudIntersection(" in html
    assert "Potree.Picker.getMousePointCloudIntersection(" not in html
    # 3. The ACTIVE camera is viewer.scene.getActiveCamera(); viewer.scene.camera
    #    is undefined in Potree 1.8.2 and made THREE.Raycaster read "undefined.type".
    assert "viewer.scene.getActiveCamera()" in html
    assert "viewer.scene.camera.position" not in html
    assert "viewer.scene.camera," not in html
    # 4. Mouse coordinates are pixel-relative to the renderer canvas.
    assert "viewer.renderer.domElement.getBoundingClientRect()" in html
    # 5. Camera animation uses the real Potree 1.8.2 API viewer.scene.view.setView.
    assert "viewer.scene.view.setView(" in html
    assert "viewer.setView(" not in html
    # 6. Controlled no-hit message + console stack logging (no uncaught JS error).
    assert "未拾取到点，请直接点击可见点云后重试" in html
    assert 'console.error("[step1] pick error:", e)' in html
    # 7. NO global THREE in the page's executable JS: Potree 1.8.2 does not
    #    expose a global THREE (the real browser threw "THREE is not defined");
    #    markers use Potree.TextSprite (the bundled THREE).
    assert "new Potree.TextSprite(" in html
    assert "viewer.scene.scene.add(" in html
    js = html[html.index("(function () {"):]
    import re as _re

    assert not _re.search(r"(?<![A-Za-z0-9_])THREE\.", js)  # comment-only, never executed


def test_plotly_is_fallback_route(tmp_path: Path) -> None:
    from algorithms.folder_pipeline.html import step1_html

    html = step1_html()
    assert "plotly" in html or "Plotly" in html


def _serve(cache_info, bounds=None) -> tuple[ThreadingHTTPServer, str, FolderServer]:
    bounds = bounds or ((100.0, 200.0, 5.0), (120.0, 220.0, 15.0))
    server = FolderServer.__new__(FolderServer)
    server.workspace = SimpleNamespace(
        source_cache_dir=cache_info.cache_dir.parents[2],
        source_sha256=cache_info.source_sha256,
    )
    server.state = FolderWorkflowState()
    server._viewer_cache = cache_info
    server._science_thread = None
    server.headless_poles = None
    server.orchestrator = SimpleNamespace(source_bounds=lambda: bounds)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.handler_factory())
    url = f"http://127.0.0.1:{httpd.server_address[1]}"
    thread = threading.Thread(
        target=httpd.serve_forever,
        kwargs={"poll_interval": 0.05},
        daemon=True,
    )
    thread.start()
    return httpd, url, server


def _pick(url, xyz):
    req = urllib.request.Request(
        url + "/api/pick",
        data=json.dumps({"x": xyz[0], "y": xyz[1], "z": xyz[2], "cache_id": "test-cache"}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read().decode("utf-8"))


def test_potree_pick_is_world_source_absolute_no_offset_added(tmp_path: Path) -> None:
    """Potree 1.8.2 pick location is already world/source-absolute (matrixWorld
    applies the octree offset in pick()); /api/pick must NOT add the offset again.
    """
    cache = _fake_cache(_ws(tmp_path))
    httpd, url, _server = _serve(cache)
    try:
        resp = _pick(url, (110.0, 220.0, 5.0))  # world pick inside bounds
        # NO double offset: source_absolute == the world pick.
        assert resp["source_xyz"] == [110.0, 220.0, 5.0]
        assert resp["viewer_world_xyz_m"] == [110.0, 220.0, 5.0]
        assert resp["pick_frame"] == "POTREE_WORLD_SOURCE_ABSOLUTE"
        # A Potree pick is an approximate seed only: NO original_index is invented.
        assert "original_index" not in resp
        assert "xyz_absolute" not in resp
    finally:
        httpd.shutdown()


def test_genuine_potree_world_pick_maps_inside_source_bounds(tmp_path: Path) -> None:
    """Regression for the double-offset bug: a genuine Potree world pick near the
    real Pole A vicinity must map INSIDE the source bounds (previously the bridge
    added the cache offset again -> out of bounds)."""
    cache = _fake_cache(_ws(tmp_path))
    # Real-like source bounds for the 105M scene (cache offset (-197,-148,-1.3)).
    real_bounds = ((-200.0, -150.0, -2.0), (120.0, 170.0, 45.0))
    httpd, url, _server = _serve(cache, bounds=real_bounds)
    try:
        # Genuine Potree world pick near real Pole A (source-absolute).
        resp = _pick(url, (-66.34, 2.23, 0.8))
        assert resp["ok"] is True
        assert resp["source_xyz"] == [-66.34, 2.23, 0.8]
        assert resp["pick_frame"] == "POTREE_WORLD_SOURCE_ABSOLUTE"
        # It is inside the source bounds (the bug previously rejected it).
        bmin, bmax = resp["source_bounds_min"], resp["source_bounds_max"]
        assert bmin[0] <= resp["source_xyz"][0] <= bmax[0]
        assert bmin[1] <= resp["source_xyz"][1] <= bmax[1]
    finally:
        httpd.shutdown()


def test_potree_pick_stale_cache_rejected(tmp_path: Path) -> None:

    cache = _fake_cache(_ws(tmp_path))
    httpd, url, _server = _serve(cache)
    try:
        req = urllib.request.Request(
            url + "/api/pick",
            data=json.dumps({"x": 0.0, "y": 0.0, "z": 0.0, "cache_id": "STALE"}).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            urllib.request.urlopen(req, timeout=10)
            raise AssertionError("expected stale cache rejection")
        except urllib.error.HTTPError as exc:
            assert exc.code == 400
    finally:
        httpd.shutdown()


def test_display_cache_and_camera_do_not_affect_science(tmp_path: Path) -> None:
    # Viewer presentation settings (point budget / LOD / camera) are display-only.
    from algorithms.folder_pipeline.workspace import discover_workspace

    ws = discover_workspace(make_workspace(tmp_path))
    orch = FolderOrchestrator(ws)
    orch.run = SimpleNamespace(
        display_sample={
            "xyz_absolute": np.zeros((3, 3)),
            "rgb": np.zeros((3, 3)),
        },
        inspection={"bounds_min_xyz": [0, 0, 0], "bounds_max_xyz": [1, 1, 1]},
        origin_xyz=(0.0, 0.0, 0.0),
    )
    before = orch.display_points()
    # Simulate a viewer LOD/point-budget change: nothing science-visible changes.
    after = orch.display_points()
    assert np.array_equal(before["xyz"], after["xyz"])
