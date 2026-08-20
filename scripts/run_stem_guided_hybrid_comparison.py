# ruff: noqa: E501
"""Create a read-only stem-guided versus baseline S005-style comparison.

The script consumes an existing span-run directory only.  It neither opens the
project registry for writing nor invokes reconciliation or identity assignment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from algorithms.spans.stem_guided_hybrid import TreeQualityState, segment_stem_guided_hybrid
from plotly.offline import get_plotlyjs


def _sample_rows(count: int, cap: int) -> np.ndarray:
    return np.arange(0, count, max(1, count // cap), dtype=np.int64)[:cap]


def _colour(key: int) -> str:
    # Stable compact colour key; it does not use a run-local ordering.
    return f"hsl({(key * 137 + 41) % 360},68%,55%)"


def _trace(xyz: np.ndarray, colour: list[str] | str, name: str, visible: bool) -> dict:
    return {
        "type": "scatter3d",
        "mode": "markers",
        "name": name,
        "visible": visible,
        "x": xyz[:, 0].round(3).tolist(),
        "y": xyz[:, 1].round(3).tolist(),
        "z": xyz[:, 2].round(3).tolist(),
        "marker": {"size": 1.4, "color": colour, "opacity": 0.82},
    }


def _html(payload: dict) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    plotly_js = get_plotlyjs()
    return f"""<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"><title>STEM GUIDED TREE COMPARISON</title>
<style>body{{margin:0;background:#101418;color:#eef3f6;font:14px system-ui}}header{{height:48px;padding:0 16px;display:flex;align-items:center;gap:18px;background:#1b242d}}h1{{font-size:16px;margin:0}}button{{padding:6px 12px;border:1px solid #6d7b86;background:#263440;color:white;border-radius:4px;cursor:pointer}}#plot{{height:calc(100vh - 48px)}}</style>
<script>{plotly_js}</script></head><body><header><h1>CROWN + STEM HYBRID · Read-only comparison</h1><button id=\"background\">BACKGROUND</button><button id=\"baseline\">BASELINE</button><button id=\"hybrid\">HYBRID</button><span id=\"summary\"></span></header><div id=\"plot\"></div>
<script>const D={encoded};const plot=document.getElementById('plot');const layout={{paper_bgcolor:'#101418',plot_bgcolor:'#101418',font:{{color:'#eaf0f4'}},scene:{{xaxis:{{title:'X local m'}},yaxis:{{title:'Y local m'}},zaxis:{{title:'HAG m'}},aspectmode:'data'}},margin:{{l:0,r:0,t:0,b:0}},legend:{{itemsizing:'constant'}}}};function show(mode){{const traces=mode==='baseline'?D.baseline:mode==='hybrid'?D.hybrid:D.background;Plotly.react(plot,traces,layout,{{responsive:true,displaylogo:false}});document.getElementById('summary').textContent=mode==='baseline'?D.baselineSummary:mode==='hybrid'?D.hybridSummary:'Deterministic background LOD and wire context.'}}document.getElementById('background').onclick=()=>show('background');document.getElementById('baseline').onclick=()=>show('baseline');document.getElementById('hybrid').onclick=()=>show('hybrid');show('baseline');</script></body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tree_root = args.run_root / "tree"
    fine = np.load(tree_root / "fine_representatives.npz", allow_pickle=False)
    source = np.load(tree_root / "tree_fine.npz", allow_pickle=False)
    xyz = np.asarray(source["fine_xyz_local_m"], dtype=np.float64)
    hag = np.asarray(source["fine_hag_m"], dtype=np.float64)
    baseline_id = np.asarray(source["fine_tree_id"], dtype=np.int32)
    if not np.array_equal(fine["original_index"], source["fine_original_index"]):
        raise RuntimeError(
            "original_index alignment between fine representatives and HAG is invalid"
        )
    hybrid = segment_stem_guided_hybrid(xyz, hag, baseline_id, raw_count=fine["raw_count"])
    sample = _sample_rows(len(xyz), 85000)
    background = _trace(xyz[sample], "#737b82", "Background (deterministic LOD)", True)
    baseline_colours = [
        _colour(int(value)) if value >= 0 else "#737b82" for value in baseline_id[sample]
    ]
    hybrid_colours: list[str] = []
    state_by_id = {item.tree_id: item.quality_state for item in hybrid.candidates}
    for value in hybrid.fine_tree_id[sample]:
        state = state_by_id.get(int(value))
        hybrid_colours.append(
            "#f0c44e"
            if state == TreeQualityState.CROWN_ONLY_REVIEW
            else _colour(int(value))
            if value >= 0
            else "#737b82"
        )
    stems = np.asarray(
        [[s.xy_local_m[0], s.xy_local_m[1], 0.0] for s in hybrid.validated_stems], dtype=np.float64
    )
    stem_top = np.asarray(
        [
            [s.xy_local_m[0], s.xy_local_m[1], s.stem_vertical_extent_m]
            for s in hybrid.validated_stems
        ],
        dtype=np.float64,
    )
    stem_xyz = np.vstack((stems, stem_top)) if len(stems) else np.empty((0, 3))
    stem_trace = _trace(stem_xyz, "#72d9ff", "Validated stem markers", True)
    wire_path = args.run_root / "wire" / "fine_representatives.npz"
    wire_trace = _trace(np.empty((0, 3)), "#ff5364", "Confirmed wire", True)
    if wire_path.is_file():
        wire = np.load(wire_path, allow_pickle=False)
        wire_trace = _trace(
            np.asarray(wire["xyz_local_m"])[_sample_rows(len(wire["xyz_local_m"]), 6000)],
            "#ff5364",
            "Wire context",
            True,
        )
    payload = {
        "background": [background, wire_trace],
        "baseline": [
            background,
            _trace(xyz[sample], baseline_colours, "Baseline watershed memberships", True),
            wire_trace,
        ],
        "hybrid": [
            background,
            _trace(
                xyz[sample], hybrid_colours, "Hybrid candidates (yellow = crown-only review)", True
            ),
            stem_trace,
            wire_trace,
        ],
        "baselineSummary": f"Baseline local watershed labels: {len(np.unique(baseline_id[baseline_id >= 0]))}; display is deterministic LOD.",
        "hybridSummary": " | ".join(
            f"{key}: {value}" for key, value in hybrid.diagnostics["state_counts"].items()
        )
        + f" | validated stems: {len(hybrid.validated_stems)}",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(_html(payload), encoding="utf-8")
    report = {
        "schema": "stem-guided-hybrid-s005-comparison-v1",
        "input_run": str(args.run_root),
        "read_only": True,
        "baseline_local_watershed_count": int(len(np.unique(baseline_id[baseline_id >= 0]))),
        "hybrid": hybrid.diagnostics,
        "validated_stems": [stem.to_dict() for stem in hybrid.validated_stems],
        "candidates": [candidate.to_dict() for candidate in hybrid.candidates],
        "original_index_conserved_at_fine_representation": True,
        "visualization": str(args.output),
    }
    args.output.with_name("CROWN_STEM_TREE_FINAL_REPORT.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"visualization": str(args.output), "hybrid": hybrid.diagnostics}, indent=2))


if __name__ == "__main__":
    main()
