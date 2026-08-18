"""Contracts for the main Viewer confirmed-wire and danger-point layers."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_main_viewer_never_routes_shadow_wire_evidence() -> None:
    source = (ROOT / "scripts" / "serve_p3a_giro3d.py").read_text(encoding="utf-8")
    assert '"wire-preview"' in source
    assert '"danger-points"' in source
    assert "spans_wire_preview" in source
    assert "spans_danger_points" in source
    assert "wire-v2" not in source
    assert "shadow" not in source.lower()


def test_confirmed_wire_and_danger_layers_guard_stale_responses() -> None:
    html = (ROOT / "viewer_poc" / "giro3d" / "index.html").read_text(encoding="utf-8")
    source = (ROOT / "viewer_poc" / "giro3d" / "viewer.js").read_text(encoding="utf-8")
    assert 'id="confirmed_wire_inspector"' in html
    assert "AUTHORITATIVE RUN ARTIFACTS" in html
    assert "spanAnalysisRequestGeneration" in source
    assert "fetchSpanAnalysisLayers(span.span_id);" in source
    assert "/wire/" in source and "/danger-points" in source
    assert "dangerPointsOverlayGroup" in source
    assert "new THREE.Points" in source


def test_run_resolution_requires_verified_tree_wire_and_clearance_artifacts() -> None:
    source = (ROOT / "scripts" / "serve_p3a_giro3d.py").read_text(encoding="utf-8")
    assert 'all(key in artifacts for key in ("tree", "wire", "clearance"))' in source
    assert '"SPAN_HAS_NO_COMPLETE_RUN"' in source
    assert "hashlib.sha256(artifact_data).hexdigest()" in source
