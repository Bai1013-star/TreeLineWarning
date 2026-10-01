"""Read-only AUTO_CORE / AUTO_HALO / UNSUPPORTED mask diagnostic for S005.

This deliberately consumes existing authoritative artifacts but writes only a
separate diagnostic directory.  It must never be imported by the Tree runner.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.stats import rankdata


RUN_ID = "s005-c96b79b5d7f04e46a4258c17bd3cdaa8"
RUN_ROOT = Path("cache/tower_ordered_project/artifacts/project-runs/groups/G002/S005") / RUN_ID
OUTPUT_ROOT = Path("cache/debug/auto_analysis_mask_s005")
CHM_MIN_HAG_M = 1.5  # Existing FastBaselineConfig value; not a new tree threshold.


def _norm_rank(values: np.ndarray, present: np.ndarray) -> np.ndarray:
    result = np.zeros(values.shape, dtype=np.float64)
    if int(np.count_nonzero(present)):
        result[present] = (rankdata(values[present], method="average") - 1.0) / max(
            1.0, float(np.count_nonzero(present) - 1)
        )
    return result


def _longest_run(bits: np.ndarray) -> int:
    longest = current = 0
    for item in bits:
        current = current + 1 if item else 0
        longest = max(longest, current)
    return longest


def build_grid(xyz: np.ndarray, hag: np.ndarray, raw_count: np.ndarray, scale: float) -> dict:
    """Aggregate existing 0.10m fine representatives without reopening source LAS."""
    origin = np.floor(np.min(xyz[:, :2], axis=0) / scale) * scale
    dims = np.floor((np.max(xyz[:, :2], axis=0) - origin) / scale).astype(int) + 1
    col = np.floor((xyz[:, 0] - origin[0]) / scale).astype(np.int32)
    row = np.floor((xyz[:, 1] - origin[1]) / scale).astype(np.int32)
    flat = row * dims[0] + col
    cells = int(dims[0] * dims[1])
    reps = np.bincount(flat, minlength=cells).reshape(dims[1], dims[0])
    raw = np.bincount(flat, weights=raw_count, minlength=cells).reshape(dims[1], dims[0])
    present = reps > 0

    # Vertical bins adapt to this span.  The first break intentionally reuses the
    # present CHM admission height; the other breaks are local distribution facts.
    elevated = hag[np.isfinite(hag) & (hag >= CHM_MIN_HAG_M)]
    q33, q66, q90 = np.quantile(elevated, [0.33, 0.66, 0.90])
    edges = np.array([CHM_MIN_HAG_M, q33, q66, q90, np.inf], dtype=np.float64)
    edges = np.maximum.accumulate(edges)
    level = np.searchsorted(edges, hag, side="right") - 1
    valid_level = (hag >= CHM_MIN_HAG_M) & (level >= 0) & (level < 4)
    occupied_levels = np.zeros((4, cells), dtype=bool)
    for k in range(4):
        occupied_levels[k, :] = np.bincount(flat[valid_level & (level == k)], minlength=cells) > 0
    occupied_levels = occupied_levels.reshape(4, dims[1], dims[0])
    vertical_bins = occupied_levels.sum(axis=0).astype(np.float64)
    longest = np.zeros_like(vertical_bins)
    for r, c in zip(*np.nonzero(present)):
        longest[r, c] = _longest_run(occupied_levels[:, r, c])
    continuity = longest / 4.0

    # Stem proxy: low and middle layers at a consistent XY cell, rewarded only
    # when the local vertical sequence is continuous.  It is diagnostic only.
    low_mid = occupied_levels[0] & occupied_levels[1]
    trunk = np.sqrt(low_mid.astype(float) * np.clip(continuity, 0.0, 1.0))

    kernel = np.ones((3, 3), dtype=np.float64)
    neighbor_occupied = ndimage.convolve(present.astype(float), kernel, mode="constant") / 9.0
    raw_per_rep = np.divide(raw, reps, out=np.zeros_like(raw), where=reps > 0)
    density = _norm_rank(np.log1p(raw / (scale * scale)), present)
    occupancy_score = vertical_bins / 4.0
    stability = 0.55 * neighbor_occupied + 0.45 * _norm_rank(raw_per_rep, present)
    support = (
        0.22 * density
        + 0.20 * occupancy_score
        + 0.22 * continuity
        + 0.24 * trunk
        + 0.12 * stability
    )
    support[~present] = 0.0

    # All classification cut points are quantiles of this span.  The component
    # area guard is expressed in square metres, so it remains scale-independent.
    values = support[present]
    q25, q40, q55, q67 = np.quantile(values, [0.25, 0.40, 0.55, 0.67])
    density_q55 = np.quantile(density[present], 0.55)
    cont_q50 = np.quantile(continuity[present], 0.50)
    trunk_q55 = np.quantile(trunk[present], 0.55)
    candidate = present & (support >= q67) & (density >= density_q55) & (continuity >= cont_q50) & (trunk >= trunk_q55)
    labels, count = ndimage.label(candidate, structure=np.ones((3, 3), dtype=np.uint8))
    component_size = np.bincount(labels.ravel(), minlength=count + 1)
    min_component_cells = max(3, int(math.ceil(2.0 / (scale * scale))))
    core = candidate & (component_size[labels] >= min_component_cells)
    # Halo is a support-bearing adjacent context, never an independent island.
    dilation_steps = max(1, int(math.ceil(1.5 / scale)))
    adjacent_core = ndimage.binary_dilation(core, structure=np.ones((3, 3), dtype=bool), iterations=dilation_steps)
    halo = present & ~core & adjacent_core & (support >= q40)
    mask = np.zeros(present.shape, dtype=np.uint8)  # 0 background, 1 halo, 2 core
    mask[halo] = 1
    mask[core] = 2

    reason = np.full(mask.shape, "UNSUPPORTED:no adequate connected multi-signal support", dtype=object)
    reason[halo] = "HALO:medium support adjacent to connected core"
    reason[core] = "CORE:high density + vertical continuity + trunk proxy + connected support"
    return {
        "scale": scale, "origin": origin, "dims": dims, "row": row, "col": col, "flat": flat,
        "mask": mask, "reason": reason, "reps": reps, "raw": raw, "density": density,
        "vertical_bins": vertical_bins, "continuity": continuity, "trunk": trunk, "stability": stability,
        "support": support, "thresholds": {"support_q25": float(q25), "support_q40": float(q40), "support_q55": float(q55), "support_q67": float(q67), "density_q55": float(density_q55), "continuity_q50": float(cont_q50), "trunk_q55": float(trunk_q55)},
        "vertical_edges": edges.tolist(), "min_component_cells": min_component_cells,
    }


def _component_rows(grid: dict) -> list[dict]:
    core = grid["mask"] == 2
    labels, count = ndimage.label(core, structure=np.ones((3, 3), dtype=np.uint8))
    rows = []
    for lab in range(1, count + 1):
        inside = labels == lab
        rows.append({"label": lab, "area_m2": float(np.count_nonzero(inside) * grid["scale"] ** 2), "raw_points": int(grid["raw"][inside].sum()), "cells": int(np.count_nonzero(inside)), "mean_support": float(grid["support"][inside].mean())})
    return sorted(rows, key=lambda item: item["area_m2"], reverse=True)


def _mask_at(grid: dict, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    col = np.floor((xy[:, 0] - grid["origin"][0]) / grid["scale"]).astype(int)
    row = np.floor((xy[:, 1] - grid["origin"][1]) / grid["scale"]).astype(int)
    valid = (row >= 0) & (row < grid["dims"][1]) & (col >= 0) & (col < grid["dims"][0])
    result = np.zeros(len(xy), dtype=np.uint8)
    result[valid] = grid["mask"][row[valid], col[valid]]
    return result, row, col


def _safe_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def write_html(path: Path, grid: dict, xyz: np.ndarray, hag: np.ndarray, wire_xyz: np.ndarray, trees: list[dict], stems: list[dict], towers: list[dict]) -> None:
    # Canvas points are a deterministic 1/12 sample, capped to keep the local
    # standalone diagnostic responsive and portable.
    step = max(1, len(xyz) // 65000)
    pts = np.column_stack((xyz[::step, :2], hag[::step])).round(3).tolist()
    wire = wire_xyz[::max(1, len(wire_xyz) // 3500), :2].round(3).tolist()
    cells = []
    for r, c in zip(*np.nonzero(grid["reps"] > 0)):
        cells.append([int(c), int(r), int(grid["mask"][r, c]), round(float(grid["support"][r, c]), 3), round(float(grid["density"][r, c]), 3), int(grid["vertical_bins"][r, c]), round(float(grid["continuity"][r, c]), 3), round(float(grid["trunk"][r, c]), 3), round(float(grid["stability"][r, c]), 3), grid["reason"][r, c]])
    payload = {"origin": grid["origin"].round(3).tolist(), "dims": grid["dims"].tolist(), "scale": grid["scale"], "points": pts, "wire": wire, "cells": cells, "trees": trees, "stems": stems, "towers": towers}
    html = f"""<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"><title>AUTO ANALYSIS MASK S005</title><style>
body{{margin:0;background:#101418;color:#e8eef2;font:14px system-ui,sans-serif}} header{{padding:12px 16px;background:#18212a;display:flex;gap:18px;align-items:center;flex-wrap:wrap}} h1{{font-size:18px;margin:0}} label{{cursor:pointer}} canvas{{display:block;width:100%;height:calc(100vh - 76px);min-height:620px;background:#0b1014}} #tip{{position:fixed;pointer-events:none;padding:7px 9px;background:#f5f7f8;color:#192129;border-radius:4px;display:none;font-size:12px;white-space:pre-line}} .green{{color:#62e58a}} .orange{{color:#ffbf5b}} .gray{{color:#a8b0b8}}</style></head><body>
<header><h1>AUTO ANALYSIS MASK · G002 / S005</h1><label><input id=\"raw\" type=\"checkbox\" checked> raw vegetation</label><label><input id=\"score\" type=\"checkbox\"> support score</label><label><input id=\"mask\" type=\"checkbox\" checked> mask overlay</label><span><span class=\"green\">■ CORE</span> <span class=\"orange\">■ HALO</span> <span class=\"gray\">■ UNSUPPORTED</span> · click a cell for evidence</span></header><canvas id=\"view\" aria-label=\"S005 support mask diagnostic\"></canvas><div id=\"tip\"></div>
<script>const D={_safe_json(payload)}, cv=document.getElementById('view'),ctx=cv.getContext('2d'),tip=document.getElementById('tip');let selected=null;const bounds={{x0:D.origin[0],y0:D.origin[1],x1:D.origin[0]+D.dims[0]*D.scale,y1:D.origin[1]+D.dims[1]*D.scale}};function size(){{cv.width=cv.clientWidth*devicePixelRatio;cv.height=cv.clientHeight*devicePixelRatio;ctx.setTransform(devicePixelRatio,0,0,devicePixelRatio,0,0);draw()}}function xy(x,y){{let p=20,w=cv.clientWidth-40,h=cv.clientHeight-40,s=Math.min(w/(bounds.x1-bounds.x0),h/(bounds.y1-bounds.y0));return [p+(x-bounds.x0)*s,cv.clientHeight-p-(y-bounds.y0)*s,s]}}function draw(){{let w=cv.clientWidth,h=cv.clientHeight;ctx.clearRect(0,0,w,h);let zmax=Math.max(...D.points.map(p=>p[2]));if(document.getElementById('raw').checked){{ctx.globalAlpha=.28;for(const p of D.points){{let q=xy(p[0],p[1]);let c=Math.floor(55+180*Math.max(0,p[2])/zmax);ctx.fillStyle=`rgb(${{45}},${{c}},${{105}})`;ctx.fillRect(q[0],q[1],1.2,1.2)}}ctx.globalAlpha=1}}let showScore=document.getElementById('score').checked,showMask=document.getElementById('mask').checked;for(const a of D.cells){{let q=xy(D.origin[0]+a[0]*D.scale,D.origin[1]+a[1]*D.scale),s=q[2]*D.scale;if(showScore){{ctx.globalAlpha=.56;ctx.fillStyle=`hsl(${{120*a[3]}},75%,48%)`;ctx.fillRect(q[0],q[1]-s,s,s)}}if(showMask){{ctx.globalAlpha=a[2]===2?.48:a[2]===1?.4:.17;ctx.fillStyle=a[2]===2?'#36d36f':a[2]===1?'#ffad34':'#8e98a0';ctx.fillRect(q[0],q[1]-s,s,s)}}}}ctx.globalAlpha=1;ctx.strokeStyle='#ff4e6a';ctx.lineWidth=1.4;ctx.beginPath();for(let i=0;i<D.wire.length;i++){{let q=xy(D.wire[i][0],D.wire[i][1]);i?ctx.lineTo(q[0],q[1]):ctx.moveTo(q[0],q[1])}}ctx.stroke();for(const q0 of D.towers){{let q=xy(q0.xy[0],q0.xy[1]);ctx.fillStyle='#67d8ff';ctx.beginPath();ctx.moveTo(q[0],q[1]-7);ctx.lineTo(q[0]+7,q[1]);ctx.lineTo(q[0],q[1]+7);ctx.lineTo(q[0]-7,q[1]);ctx.closePath();ctx.fill();ctx.fillStyle='#d8f6ff';ctx.fillText(q0.id,q[0]+8,q[1]+11)}}for(const t of D.trees){{let q=xy(t.apex[0],t.apex[1]);ctx.fillStyle=t.mask===2?'#39ff86':t.mask===1?'#ffd069':'#d4d9de';ctx.beginPath();ctx.arc(q[0],q[1],4,0,Math.PI*2);ctx.fill();ctx.fillStyle='#ffffff';ctx.fillText(t.persistent,q[0]+6,q[1]-5)}}ctx.strokeStyle='#71d8ff';for(const s of D.stems){{let q=xy(s.xy[0],s.xy[1]);ctx.strokeRect(q[0]-3,q[1]-3,6,6)}}if(selected){{let q=xy(D.origin[0]+selected[0]*D.scale,D.origin[1]+(selected[1]+1)*D.scale),ss=q[2]*D.scale;ctx.strokeStyle='#fff';ctx.lineWidth=2;ctx.strokeRect(q[0],q[1]-ss,ss,ss)}}}}function pick(ev){{let r=cv.getBoundingClientRect(),x=ev.clientX-r.left,y=ev.clientY-r.top;let p=xy(bounds.x0,bounds.y0),s=p[2],col=Math.floor((x-20)/s/D.scale),row=Math.floor((cv.clientHeight-20-y)/s/D.scale);let a=D.cells.find(v=>v[0]===col&&v[1]===row);if(a){{selected=[col,row];tip.style.display='block';tip.style.left=(ev.clientX+12)+'px';tip.style.top=(ev.clientY+12)+'px';tip.textContent=`${{a[9]}}\nscore=${{a[3]}} density=${{a[4]}} vert bins=${{a[5]}} continuity=${{a[6]}} trunk=${{a[7]}} stability=${{a[8]}}`;draw()}}}}cv.addEventListener('click',pick);document.querySelectorAll('input').forEach(x=>x.addEventListener('change',draw));addEventListener('resize',size);size();</script></body></html>"""
    path.write_text(html, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    start = time.perf_counter()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)

    fine = np.load(RUN_ROOT / "tree" / "fine_representatives.npz", allow_pickle=False)
    tree_fine = np.load(RUN_ROOT / "tree" / "tree_fine.npz", allow_pickle=False)
    xyz = fine["xyz_local_m"].astype(np.float64, copy=False)
    raw_count = fine["raw_count"].astype(np.float64, copy=False)
    hag = tree_fine["fine_hag_m"].astype(np.float64, copy=False)
    if not np.array_equal(fine["original_index"], tree_fine["fine_original_index"]):
        raise RuntimeError("fine representative / HAG original_index alignment is invalid")

    grids = {scale: build_grid(xyz, hag, raw_count, scale) for scale in (0.5, 1.0)}
    grid = grids[1.0]
    point_masks = {scale: _mask_at(item, xyz[:, :2])[0] for scale, item in grids.items()}
    raw_weight = raw_count
    a, b = point_masks[0.5] == 2, point_masks[1.0] == 2
    core_iou = float(raw_weight[a & b].sum() / max(1.0, raw_weight[a | b].sum()))
    stability = "PASS" if core_iou >= 0.60 else "FAIL"

    recon = json.loads((RUN_ROOT / "reconciliation.json").read_text(encoding="utf-8"))
    local_tree_id = tree_fine["fine_tree_id"].astype(np.int32, copy=False)
    centroid_by_local_tree = {}
    for local_id in np.unique(local_tree_id[local_tree_id >= 0]):
        rows = local_tree_id == local_id
        centroid_by_local_tree[int(local_id)] = np.average(xyz[rows, :2], axis=0, weights=raw_count[rows]).round(4).tolist()
    tree_rows = []
    for rec in recon["global_tree_records"]:
        xy = np.asarray(rec["apex_xy_local_m"], dtype=float)[None, :]
        mask, row, col = _mask_at(grid, xy)
        r, c = int(row[0]), int(col[0])
        local_tree_id_value = int(rec["primary_span_tree_id"])
        tree_rows.append({"persistent": rec["persistent_tree_id"], "global_tree_id": rec["global_tree_id"], "apex": rec["apex_xy_local_m"], "anchor": rec["apex_xy_local_m"], "anchor_source": "authoritative apex (no distinct anchor field persisted)", "centroid": centroid_by_local_tree.get(local_tree_id_value), "mask": int(mask[0]), "support": float(grid["support"][r, c]) if mask[0] or (0 <= r < grid["dims"][1] and 0 <= c < grid["dims"][0]) else 0.0, "full_resolution_point_count": rec["full_resolution_point_count"]})
    tree_classes = Counter(item["mask"] for item in tree_rows)

    # Diagnostic-only trunk seeds: core cells with high stem proxy; no growth,
    # entity creation, membership write, or registry access.
    trunk_gate = np.quantile(grid["trunk"][grid["reps"] > 0], 0.80)
    stem_rc = np.argwhere((grid["mask"] == 2) & (grid["trunk"] >= trunk_gate))
    stems = [{"xy": [float(grid["origin"][0] + (c + .5) * grid["scale"]), float(grid["origin"][1] + (r + .5) * grid["scale"])], "confidence": round(float(grid["trunk"][r, c]), 3)} for r, c in stem_rc]

    wire_fine = np.load(RUN_ROOT / "wire" / "fine_representatives.npz", allow_pickle=False)
    wire_xyz = wire_fine["xyz_local_m"].astype(np.float64, copy=False)
    source_sidecar = json.loads((Path("cache/tower_ordered_project/artifacts/project-runs/source-shared/56dca4df09a2e32363e70519d6782e88ef57703b88b261a7e0caa47077f4098f/vegetation-region/region_sidecar.json")).read_text(encoding="utf-8"))
    scene_origin = np.asarray(source_sidecar["origin_xyz"], dtype=float)
    tower_state = json.loads((Path("cache/tower_ordered_project/state/generations/g-0000000000000107-a9213696/towers/towers.json")).read_text(encoding="utf-8"))
    towers = []
    for record in tower_state["records"]:
        if record["tower_id"] in {"T006", "T007"}:
            absolute = np.asarray(record["revisions"][-1]["refined_geometry"]["base_xyz_absolute_m"], dtype=float)
            towers.append({"id": record["tower_id"], "xy": (absolute[:2] - scene_origin[:2]).round(3).tolist()})
    tower_xy = {item["id"]: np.asarray(item["xy"], dtype=float) for item in towers}
    component_rows = _component_rows(grid)
    largest = component_rows[0] if component_rows else {"area_m2": 0.0, "raw_points": 0, "cells": 0, "mean_support": 0.0}
    category_stats = {}
    for code, name in ((2, "AUTO_CORE"), (1, "AUTO_HALO"), (0, "UNSUPPORTED_BACKGROUND")):
        cells = grid["mask"] == code
        on_points = point_masks[1.0] == code
        category_stats[name] = {"area_m2": float(np.count_nonzero(cells) * grid["scale"] ** 2), "fine_representatives": int(np.count_nonzero(on_points)), "estimated_raw_points": int(raw_count[on_points].sum()), "cells": int(np.count_nonzero(cells))}
    fragmented = (grid["mask"] == 0) & (grid["reps"] > 0)
    fragmented_anchor_ids = [t["persistent"] for t in tree_rows if t["mask"] == 0]
    core_labels, core_count = ndimage.label(grid["mask"] == 2, structure=np.ones((3, 3), dtype=np.uint8))
    largest_label = int(max(range(1, core_count + 1), key=lambda label: int(np.count_nonzero(core_labels == label)))) if core_count else 0
    largest_cells = core_labels == largest_label
    point_largest = core_labels[grid["row"], grid["col"]] == largest_label
    largest_tree_ids = []
    for tree in tree_rows:
        _, rr, cc = _mask_at(grid, np.asarray(tree["apex"], dtype=float)[None, :])
        if 0 <= rr[0] < grid["dims"][1] and 0 <= cc[0] < grid["dims"][0] and core_labels[rr[0], cc[0]] == largest_label:
            largest_tree_ids.append(tree["persistent"])
    largest_region = {**largest, "vegetation_raw_point_count": int(raw_count[point_largest].sum()), "current_tree_anchors": largest_tree_ids, "stem_evidence_cell_count": int(np.count_nonzero(largest_cells & (grid["trunk"] >= trunk_gate))), "vertical_support_statistics": {"mean_occupied_bins": round(float(grid["vertical_bins"][largest_cells].mean()), 3), "median_continuity": round(float(np.median(grid["continuity"][largest_cells])), 3)}}
    direction = tower_xy["T007"] - tower_xy["T006"]
    direction /= np.linalg.norm(direction)
    center = (tower_xy["T006"] + tower_xy["T007"]) / 2.0
    yy, xx = np.indices(grid["mask"].shape)
    cell_xy = np.stack((grid["origin"][0] + (xx + .5) * grid["scale"], grid["origin"][1] + (yy + .5) * grid["scale"]), axis=-1)
    lateral = direction[0] * (cell_xy[..., 1] - center[1]) - direction[1] * (cell_xy[..., 0] - center[0])
    side_stats = {}
    for name, side in (("positive_lateral", lateral >= 0), ("negative_lateral", lateral < 0)):
        side_stats[name] = {"auto_core_area_m2": float(np.count_nonzero(side & (grid["mask"] == 2))), "unsupported_area_m2": float(np.count_nonzero(side & (grid["mask"] == 0))), "mean_support": round(float(grid["support"][side & (grid["reps"] > 0)].mean()), 3)}
    high_side = max(side_stats, key=lambda key: side_stats[key]["auto_core_area_m2"])
    low_side = min(side_stats, key=lambda key: side_stats[key]["auto_core_area_m2"])
    high_support_side = {"exists": bool(side_stats[high_side]["auto_core_area_m2"] >= 1.5 * max(1.0, side_stats[low_side]["auto_core_area_m2"])), "reference": "signed lateral coordinate relative to ordered T006 -> T007; no left/right rule", "high_support_side": high_side, "low_support_fragmented_side": low_side, "side_statistics": side_stats}
    # Conservative confidence: a dominant connected core, separation from the
    # unsupported field, stable scale comparison, and anchors retained.
    core_share = category_stats["AUTO_CORE"]["area_m2"] / max(1e-9, sum(v["area_m2"] for v in category_stats.values()))
    confidence = "HIGH" if stability == "PASS" and core_share >= 0.30 and tree_classes[0] <= 2 else "MEDIUM" if stability == "PASS" else "LOW"
    recommendation = "PROCEED_WITH_A" if confidence == "HIGH" else "USE_C_HYBRID"

    mapped_fine_count = int(np.count_nonzero(np.isin(point_masks[1.0], (0, 1, 2))))
    report = {
        "AUTO_MASK_INPUT_AUDIT": {"AVAILABLE_SIGNALS": ["S005 selected vegetation after confirmed-wire exclusion (52,477,380 raw points)", "0.10m fine representatives with original_index and raw_count (588,500)", "fine HAG aligned by original_index", "tree memberships / current 22 reconciled records", "confirmed wire fine representatives and 3 confirmed strands", "span geometry / endpoint identities T006 -> T007", "classification in selected vegetation region"], "MISSING_SIGNALS": ["Per-point scan angle / return-number quality metrics", "explicit scanner trajectory / incidence-angle metadata", "separate raw trunk-labelled ground truth"], "SUFFICIENT_FOR": ["local density", "vertical occupancy / continuity", "cell-local trunk proxy", "sampling regularity proxy", "connected support routing"], "SOURCE_READ": "NO full 105M source scan; reused authoritative fine representations"},
        "INPUT_SPAN": "G002 / S005 / T006 -> T007", "AUTHORITATIVE_RUN_CHANGED": "NO", "TREE_CORE_CHANGED": "NO", "WIRE_CHANGED": "NO", "CLEARANCE_CHANGED": "NO", "GRID_SCALE_TESTED_M": [0.5, 1.0], "RECOMMENDED_GRID_SCALE_M": 1.0, "RECOMMENDED_GRID_RATIONALE": "1.0m preserves the connected support field while reducing 0.5m micro-islands; comparison retained for stability only.", "SIGNALS_USED": ["raw point support per m2", "occupied 0.10m representatives", "span-adaptive HAG vertical occupancy", "vertical continuity", "lower/mid vertical trunk proxy", "neighbor occupancy and raw-per-representative stability", "connected-component routing"], "VERTICAL_BINS_M": grid["vertical_edges"], "AUTO_CORE": category_stats["AUTO_CORE"], "AUTO_HALO": category_stats["AUTO_HALO"], "UNSUPPORTED": category_stats["UNSUPPORTED_BACKGROUND"], "CURRENT_TREE_COUNT": len(tree_rows), "CURRENT_TREE_IN_CORE": tree_classes[2], "CURRENT_TREE_IN_HALO": tree_classes[1], "CURRENT_TREE_IN_UNSUPPORTED": tree_classes[0], "CURRENT_TREE_MASK_ASSIGNMENT": tree_rows, "POSSIBLE_TRUNK_SEEDS": len(stems), "LARGEST_CONNECTED_AUTO_CORE": largest_region, "HIGH_SUPPORT_SIDE": high_support_side, "FRAGMENTED_UNSUPPORTED_AREA_M2": float(np.count_nonzero(fragmented)), "FRAGMENTED_UNSUPPORTED_POINT_COUNT": int(raw_count[point_masks[1.0] == 0].sum()), "CURRENT_TREE_ANCHORS_IN_FRAGMENTED_AREA": fragmented_anchor_ids, "SOURCE_MAPPING_VALIDATION": {"fine_representatives_with_original_index": int(len(fine["original_index"])), "fine_representatives_routed_to_exactly_one_mask": mapped_fine_count, "raw_point_weight_sum": int(raw_count.sum()), "raw_point_weight_sum_routed": int(sum(item["estimated_raw_points"] for item in category_stats.values()))}, "SOURCE_POINTS_DELETED_BY_MASK": 0, "PARAMETER_STABILITY": stability, "CORE_WEIGHTED_IOU_0_5M_VS_1_0M": core_iou, "ANALYSIS_MASK_CONFIDENCE": confidence, "SCENE_HARDCODE": "NO", "AUTO_A_RECOMMENDATION": recommendation, "NEXT_INTEGRATION_IF_APPROVED": ["Add a versioned mask artifact produced before tree seeding", "Restrict new seeds to AUTO_CORE; admit AUTO_HALO only as crown context", "Preserve original_index routing and render unsupported raw background grey", "Keep an owner mask-review override if confidence is not HIGH"], "RUNTIME_SECONDS": round(time.perf_counter() - start, 3), "POINTS_PROCESSED": int(len(xyz)), "AUTHORITATIVE_PROJECT_CHANGED": "NO", "FINAL_STATUS": "READY_FOR_OWNER_MASK_REVIEW"}
    (out / "AUTO_ANALYSIS_MASK_S005_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    np.savez_compressed(out / "AUTO_ANALYSIS_MASK_S005_cells_1m.npz", mask=grid["mask"], support_score=grid["support"], density=grid["density"], vertical_bins=grid["vertical_bins"], vertical_continuity=grid["continuity"], trunk_support=grid["trunk"], sampling_stability=grid["stability"], origin_xy=grid["origin"], grid_scale_m=grid["scale"])
    write_html(out / "AUTO_ANALYSIS_MASK_S005.html", grid, xyz, hag, wire_xyz, tree_rows, stems, towers)
    print(json.dumps({key: report[key] for key in ("AUTO_CORE", "AUTO_HALO", "UNSUPPORTED", "CURRENT_TREE_IN_CORE", "CURRENT_TREE_IN_HALO", "CURRENT_TREE_IN_UNSUPPORTED", "POSSIBLE_TRUNK_SEEDS", "CORE_WEIGHTED_IOU_0_5M_VS_1_0M", "ANALYSIS_MASK_CONFIDENCE", "AUTO_A_RECOMMENDATION", "RUNTIME_SECONDS")}, indent=2))


if __name__ == "__main__":
    main()
