"""Focused offline acceptance checks for the 2026-10-21 release.

The script intentionally uses only the portable runtime dependencies so the
same checks can run from source and from the delivered program directory.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

# Match the portable bootstrap when this script is run with runtime/python.exe.
ROOT = Path(__file__).resolve().parents[1]
if (ROOT / "packages").is_dir():
    sys.path.insert(0, str(ROOT / "packages"))
    sys.path.insert(0, str(ROOT))
    runtime = ROOT / "runtime"
    sys.path.append(str(runtime / "Lib/site-packages"))
    os.environ["PATH"] = os.pathsep.join(
        (str(runtime / "Library/bin"), str(runtime / "DLLs"), os.environ.get("PATH", ""))
    )
    for directory in (runtime / "Library/bin", runtime / "DLLs"):
        if directory.is_dir() and hasattr(os, "add_dll_directory"):
            os.add_dll_directory(str(directory))

import numpy as np

from algorithms.spans.wire import _parallel_family_interior_rescue
from algorithms.spans.wire_v2 import CONFIRMED, reconstruct_slice_tracks
from core.config.fast_baseline import FastBaselineConfig


def require(condition: bool, label: str) -> None:
    if not condition:
        raise AssertionError(label)


def verify_ui_sources() -> dict[str, bool]:
    index = (ROOT / "web/templates/index.html").read_text(encoding="utf-8")
    dbh = (ROOT / "viewer_poc/giro3d/dbh_review.js").read_text(encoding="utf-8")
    future_html = (ROOT / "viewer_poc/giro3d/future_preview.html").read_text(encoding="utf-8")
    future_js = (ROOT / "viewer_poc/giro3d/future_preview.js").read_text(encoding="utf-8")

    require('id="integratedRecognitionPhotos"' in index, "integrated photo recognition missing")
    require('id="integratedRecognitionVideos"' in index, "integrated video recognition missing")
    require('id="integratedRecognitionPointCloud"' in index, "integrated point-cloud recognition missing")
    require('id="predictIntegratedSpecies"' in index, "integrated prediction action missing")
    require(".slice(0,3)" in index, "top-three candidate limit missing")
    training = index.split('id="trainingPage"', 1)[1]
    require('id="predictRecognition"' not in training, "prediction still duplicated in training page")

    require("state.review.read_only" not in dbh, "automatic DBH still disables review controls")
    require("保留原胸径" in dbh, "failed recheck preservation message missing")

    require('id="growthPlot"' in future_html, "growth plot missing")
    require('id="clearancePlot"' in future_html, "clearance plot missing")
    require("plot.call(window.Plotly, growthPlot" in future_js, "growth chart renderer missing")
    require("plot.call(window.Plotly, clearancePlot" in future_js, "clearance chart renderer missing")
    return {
        "dbh_recheck_controls": True,
        "split_growth_clearance_charts": True,
        "short_term_species_prediction": True,
    }


def verify_wire_reconstruction() -> dict[str, bool]:
    config = FastBaselineConfig()
    for count in (3, 4):
        arrays = []
        for wire in range(count):
            s = np.arange(0.1, 30, 0.2)
            if wire == count - 1:
                s = s[(s < 13) | (s > 15)]
            arrays.append(
                np.column_stack((s, np.full(len(s), wire * 1.2), 8 + 0.001 * (s - 15) ** 2))
            )
        xyz = np.concatenate(arrays)
        tracks = reconstruct_slice_tracks(
            np.arange(len(xyz)), xyz[:, 0], xyz[:, 1], xyz[:, 2],
            np.ones(len(xyz), dtype=np.int64), 30, config,
        )
        confirmed = [track for track in tracks if track.status == CONFIRMED]
        require(len(confirmed) == count, f"expected {count} conductors, got {len(confirmed)}")

    strong = SimpleNamespace(
        status=CONFIRMED,
        coverage_ratio=0.94,
        supported_slice_count=38,
        supported_slice_ratio=0.90,
        bounded_supported_slice_ratio=0.90,
        maximum_gap_m=2.94,
        inlier_ratio=0.92,
        median_residual_m=0.08,
        p95_residual_m=0.22,
        supported_s_min_m=0., supported_s_max_m=30., model_type='ROBUST_QUADRATIC',
        model_u_coefficients=np.array([15.,30.,0.,0.]),
        model_z_coefficients=np.array([15.,30.,0.,0.,11.]),
    )
    left=SimpleNamespace(**{**strong.__dict__,'model_u_coefficients':np.array([15.,30.,0.,-1.])})
    right=SimpleNamespace(**{**strong.__dict__,'model_u_coefficients':np.array([15.,30.,0.,1.])})
    require(
        _parallel_family_interior_rescue(strong, [left, strong, right], config),
        "strong occluded third conductor was not retained",
    )
    require(
        not _parallel_family_interior_rescue(strong, [left, strong], config),
        "two-line family incorrectly enabled the rescue",
    )
    clutter = SimpleNamespace(**{**strong.__dict__, "p95_residual_m": 0.9})
    require(
        not _parallel_family_interior_rescue(clutter, [left, right, clutter], config),
        "high-residual clutter incorrectly enabled the rescue",
    )
    return {"three_conductors": True, "four_conductors": True, "clutter_rejected": True}


def main() -> None:
    result = {"ui": verify_ui_sources(), "wire": verify_wire_reconstruction()}
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
