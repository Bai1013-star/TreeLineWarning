# ruff: noqa: E501  (long HTML/JS template content)
"""Potree LOD STEP-1 viewer page for the Folder pipeline (primary viewer).

The page is served by the Folder server.  It loads the pinned Potree 1.8.2
runtime and the reproducible workspace viewer cache, renders the full-scene
multi-resolution LOD point cloud, and lets the owner pick Pole A / Pole B.

A rendered pick is ONLY an approximate seed: the page sends the viewer-LOCAL
XYZ + cache id to the backend, which transforms to source-absolute coordinates
(explicit bridge), validates the source bounds, runs the existing bounded raw
recovery + pole refinement, and returns a refined-pole preview.  The page
never invents an ``original_index`` and never persists the raw display pick as
a confirmed tower.
"""

from __future__ import annotations

from typing import Any

_POTREE_PAGE_CSS = """
body{margin:0;font-family:"Microsoft YaHei","Segoe UI",sans-serif;background:#222;color:#eee}
#render_area{position:absolute;left:0;top:0;right:320px;bottom:0}
#panel{position:absolute;right:0;top:0;width:320px;height:100%;background:#1b1f27;border-left:1px solid #333;
  box-sizing:border-box;padding:12px;overflow-y:auto;z-index:1000}
#panel h2{font-size:15px;margin:0 0 10px}
button{background:#1f6feb;color:#fff;border:0;border-radius:4px;padding:8px 12px;font-size:13px;cursor:pointer;
  margin:4px 4px 4px 0}
button:disabled{background:#3a4354;cursor:not-allowed}
button.gray{background:#5f6b7a}button.red{background:#d64545}button.green{background:#2ea44f}
#status,#msg{font-size:12px;margin-top:8px;min-height:16px;color:#ffd24a}
#msg{color:#ff8a80}
table{font-size:12px;border-collapse:collapse;width:100%;margin-top:8px}
td,th{border:1px solid #333;padding:4px;text-align:left}
.kpi{display:flex;justify-content:space-between;font-size:12px;margin-top:6px}
select,input[type=range]{width:100%}
.badge{display:inline-block;padding:2px 8px;border-radius:10px;font-size:11px;margin-left:6px}
.badge.ready{background:#1b5e20}.badge.pending{background:#b26a00}.badge.error{background:#8b0000}
"""


def potree_step1_html(
    cache_info: Any,
    source_bounds: tuple[tuple[float, float, float], tuple[float, float, float]] | None = None,
) -> str:
    """Render the primary STEP-1 Potree LOD page.

    ``cache_info`` is a ``ViewerCacheInfo`` with the metadata URL, offset and
    hierarchy used to bootstrap the viewer; ``source_bounds`` is the source
    LAS absolute bounds (min, max) used by the backend for pick validation.
    """

    metadata_url = f"/pointclouds/{cache_info.cache_id}/metadata.json"
    offset = list(cache_info.offset_xyz)
    cache_id = cache_info.cache_id
    point_count = getattr(cache_info, "point_count", 0)
    depth = getattr(cache_info, "hierarchy_depth", 0)
    if source_bounds is None:
        source_min = [0.0, 0.0, 0.0]
        source_max = [0.0, 0.0, 0.0]
    else:
        source_min = list(source_bounds[0])
        source_max = list(source_bounds[1])
    cache_json = {
        "cache_id": cache_id,
        "metadata_url": metadata_url,
        "offset_xyz": offset,
        "source_bounds_min": source_min,
        "source_bounds_max": source_max,
        "point_count": point_count,
        "hierarchy_depth": depth,
        "converter_version": getattr(cache_info, "converter_version", ""),
        "viewer_version": getattr(cache_info, "viewer_version", ""),
    }
    import json as _json

    cache_payload = _json.dumps(cache_json, ensure_ascii=False)

    return f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, user-scalable=no">
<title>STEP 1 档距选择 — Potree LOD</title>
<link rel="stylesheet" type="text/css" href="/potree/build/potree/potree.css">
<link rel="stylesheet" type="text/css" href="/potree/libs/jquery-ui/jquery-ui.min.css">
<link rel="stylesheet" type="text/css" href="/potree/libs/openlayers3/ol.css">
<link rel="stylesheet" type="text/css" href="/potree/libs/spectrum/spectrum.css">
<link rel="stylesheet" type="text/css" href="/potree/libs/jstree/themes/mixed/style.css">
<style>{_POTREE_PAGE_CSS}</style>
</head>
<body>
<div id="render_area"></div>
<div id="panel">
  <h2>STEP 1 人工选择档距 <span id="viewerbadge" class="badge pending">POTREE LOD</span></h2>
  <p style="font-size:12px;color:#9aa;margin:4px 0 8px">
    在点云中点击真实杆塔：先按「选择杆塔 A」再按「选择杆塔 B」。点击只是近似坐标，
    后端会用全分辨率原始点云做杆塔精化。左键旋转 / 右键平移 / 滚轮缩放。
  </p>
  <div>
    <button id="btn-a">选择杆塔 A</button>
    <button id="btn-b">选择杆塔 B</button>
    <button id="btn-reset" class="gray">重新选择</button>
    <button id="btn-fit" class="gray">适配/重置</button>
  </div>
  <div>
    <button id="btn-top" class="gray">俯视</button>
    <button id="btn-side" class="gray">侧视</button>
  </div>
  <div id="status">Potree 加载中…</div>
  <div id="msg"></div>
  <div class="kpi"><span>点预算</span><span id="budget_val">750,000</span></div>
  <input type="range" id="budget" min="100000" max="2000000" step="50000" value="750000">
  <table id="seltable">
    <tr><th></th><th>X</th><th>Y</th><th>Z</th><th>状态</th></tr>
    <tr id="rowA"><td>A</td><td id="a_x">—</td><td id="a_y">—</td><td id="a_z">—</td><td id="a_s">—</td></tr>
    <tr id="rowB"><td>B</td><td id="b_x">—</td><td id="b_y">—</td><td id="b_z">—</td><td id="b_s">—</td></tr>
  </table>
  <div class="kpi"><span>可见点数</span><span id="visible">—</span></div>
  <div class="kpi"><span>可见节点(LOD)</span><span id="nodes">—</span></div>
  <div class="kpi"><span>缓存</span><span>{cache_id[:12]}…</span></div>
  <div style="margin-top:10px">
    <button id="btn-confirm" class="green" disabled>确认档距并继续</button>
  </div>
  <div id="fallback" style="margin-top:10px;font-size:11px;color:#889">
    若 WebGL 不可用，请用 <a href="/step1_plotly" style="color:#79c">Plotly 传统选择器</a>。
  </div>
</div>

<script src="/potree/libs/jquery/jquery-3.1.1.min.js"></script>
<script src="/potree/libs/spectrum/spectrum.js"></script>
<script src="/potree/libs/jquery-ui/jquery-ui.min.js"></script>
<script src="/potree/libs/other/BinaryHeap.js"></script>
<script src="/potree/libs/tween/tween.min.js"></script>
<script src="/potree/libs/d3/d3.js"></script>
<script src="/potree/libs/proj4/proj4.js"></script>
<script src="/potree/libs/openlayers3/ol.js"></script>
<script src="/potree/libs/i18next/i18next.js"></script>
<script src="/potree/libs/jstree/jstree.js"></script>
<script src="/potree/build/potree/potree.js"></script>
<script src="/potree/libs/plasio/js/laslaz.js"></script>
<script>
(function () {{
  "use strict";
  const CACHE = {cache_payload};
  let viewer = null;
  let pointcloud = null;
  let mode = "IDLE"; // IDLE | SELECT_A | SELECT_B
  let poles = {{A: null, B: null}};
  let markers = {{A: null, B: null}};
  const renderArea = document.getElementById("render_area");
  const $status = (t) => (document.getElementById("status").textContent = t);
  const $msg = (t) => (document.getElementById("msg").textContent = t);

  function badge(state) {{
    const b = document.getElementById("viewerbadge");
    b.className = "badge " + state;
    b.textContent = state === "ready" ? "POTREE LOD ✓" : "POTREE LOD";
  }}

  function showError(message) {{
    $msg("失败：" + message);
    $status("Potree 加载失败");
  }}

  // ---- coordinate bridge: Potree pick is ALREADY world/source-absolute -----
  // Pinned Potree 1.8.2 fact: PointCloudOctree constructor sets
  // `this.position.copy(geometry.offset)` and pick() applies
  // `position.applyMatrix4(pc.matrixWorld)`, so
  // Potree.Utils.getMousePointCloudIntersection(...).location is WORLD =
  // source-absolute.  The backend /api/pick treats it as source-absolute and
  // does NOT add the cache offset again (that would double-offset and push the
  // pick outside the source bounds).

  async function post(path, payload) {{
    const r = await fetch(path, {{method: "POST", headers: {{"Content-Type": "application/json"}}, body: JSON.stringify(payload || {{}})}});
    const body = await r.json().catch(() => ({{}}));
    if (!r.ok) {{ throw new Error(body.error || ("请求失败 " + path)); }}
    return body;
  }}

  // ---- markers: Potree's OWN TextSprite (internal THREE, no global THREE) --
  // Potree 1.8.2 does NOT expose a global THREE; its official examples create
  // markers with Potree.TextSprite (uses the bundled THREE).  Custom page code
  // must never reference the global THREE (that throws "THREE is not defined").
  function makeMarker(color, label) {{
    const sprite = new Potree.TextSprite(label);
    sprite.setTextColor({{r: 255, g: 255, b: 255, a: 1.0}});
    sprite.setBorderColor({{r: 0, g: 0, b: 0, a: 1.0}});
    sprite.setBackgroundColor(color);
    sprite.fontsize = 24;
    sprite.material.depthTest = false;
    sprite.scale.set(2.0, 2.0, 1.0);
    return sprite;
  }}

  function setMarker(pole, xyz) {{
    if (markers[pole]) {{ viewer.scene.scene.remove(markers[pole]); }}
    const m = makeMarker(pole === "A" ? {{r: 255, g: 82, b: 82, a: 1.0}} : {{r: 64, g: 196, b: 255, a: 1.0}}, pole);
    m.position.set(xyz[0], xyz[1], xyz[2]);
    viewer.scene.scene.add(m);
    markers[pole] = m;
    showSpanInfo();
  }}

  function showSpanInfo() {{
    if (poles.A && poles.B) {{
      const dx = poles.A[0] - poles.B[0], dy = poles.A[1] - poles.B[1], dz = poles.A[2] - poles.B[2];
      const len = Math.sqrt(dx * dx + dy * dy + dz * dz);
      $status("已选择杆塔 A 与 B，档距长度 " + len.toFixed(2) + " m");
    }}
  }}

  // ---- pick + refine ------------------------------------------------------
  async function handlePick(pole, worldXyz) {{
    try {{
      // Diagnostic (owner real-click audit): log every coordinate in the
      // Potree -> source bridge so a real click is fully inspectable.
      if (pointcloud && pointcloud.pcoGeometry) {{
        const mw = pointcloud.matrixWorld;
        console.log("[step1] pick world (I.location)      =", [worldXyz.x, worldXyz.y, worldXyz.z]);
        console.log("[step1] pointcloud.position          =", pointcloud.position.toArray());
        console.log("[step1] pointcloud.pcoGeometry.offset=", pointcloud.pcoGeometry.offset.toArray());
        console.log("[step1] matrixWorld.position         =", [mw.elements[12], mw.elements[13], mw.elements[14]]);
      }}
      console.log("[step1] cache offset                  =", CACHE.offset_xyz);
      console.log("[step1] cache source bounds min/max   =", CACHE.source_bounds_min, CACHE.source_bounds_max);
      const source = await post("/api/pick", {{x: worldXyz.x, y: worldXyz.y, z: worldXyz.z, cache_id: CACHE.cache_id}});
      const sx = source.source_xyz;
      console.log("[step1] backend source_absolute_xyz   =", sx);
      $msg("杆塔 " + pole + " 近似坐标: " + sx.map(v => v.toFixed(2)).join(", "));
      await post("/click", {{pole: pole, x: sx[0], y: sx[1], z: sx[2]}});
      const refined = await post("/refine", {{pole: pole}});
      poles[pole] = sx;
      setMarker(pole, sx);
      const row = document.getElementById("row" + pole);
      document.getElementById((pole === "A" ? "a" : "b") + "_x").textContent = sx[0].toFixed(2);
      document.getElementById((pole === "A" ? "a" : "b") + "_y").textContent = sx[1].toFixed(2);
      document.getElementById((pole === "A" ? "a" : "b") + "_z").textContent = sx[2].toFixed(2);
      if (refined.defensible) {{
        document.getElementById((pole === "A" ? "a" : "b") + "_s").textContent = "有效 高" + refined.height_m.toFixed(1) + "m";
        $status("杆塔 " + pole + " 精化有效");
      }} else {{
        document.getElementById((pole === "A" ? "a" : "b") + "_s").textContent = "失败:" + (refined.reason || "未知");
        $msg("杆塔 " + pole + " 精化失败：" + (refined.reason || "未知"));
      }}
    }} catch (e) {{
      $msg("杆塔 " + pole + " 处理失败：" + e.message);
    }}
    mode = "IDLE";
    document.getElementById("btn-a").style.outline = "";
    document.getElementById("btn-b").style.outline = "";
    if (poles.A && poles.B) document.getElementById("btn-confirm").disabled = false;
  }}

  function registerCanvasClick() {{
    renderArea.addEventListener("click", function (ev) {{
      if (mode === "IDLE") {{
        $msg("请先按「选择杆塔 A」或「选择杆塔 B」");
        return;
      }}
      try {{
        // Pixel coordinates relative to the Potree renderer canvas (pinned
        // 1.8.2 expects this, NOT container-relative / NDC / event objects).
        const rect = viewer.renderer.domElement.getBoundingClientRect();
        const mouse = {{x: ev.clientX - rect.left, y: ev.clientY - rect.top}};
        // Canonical 1.8.2 call (matches Potree's own drag tool at the pinned
        // bundle): the ACTIVE camera is viewer.scene.getActiveCamera().
        // viewer.scene.camera is undefined in Potree 1.8.2 and makes
        // (the internal THREE raycaster setFromCamera then reads "undefined.type").
        const camera = viewer.scene.getActiveCamera();
        const pointclouds = viewer.scene.pointclouds;
        const hit = Potree.Utils.getMousePointCloudIntersection(
          mouse, camera, viewer, pointclouds, {{pickClipped: false}}
        );
        if (hit) {{
          handlePick(mode === "SELECT_A" ? "A" : "B", hit.location);
        }} else {{
          $msg("未拾取到点，请直接点击可见点云后重试");
        }}
      }} catch (e) {{
        console.error("[step1] pick error:", e);
        $msg("拾取失败：" + (e && e.message ? e.message : String(e)));
      }}
    }});
  }}

  // ---- camera controls ----------------------------------------------------
  function fitView() {{
    if (pointcloud && pointcloud.pcoGeometry) viewer.fitToScreen();
  }}
  function topView() {{
    try {{
      // Potree 1.8.2 camera animation is on viewer.scene.view.setView(position,
      // target, duration) — the viewer-level yaw/pitch setView is NOT the API.
      const view = viewer.scene.view;
      const bmin = CACHE.source_bounds_min || [0, 0, 0];
      const bmax = CACHE.source_bounds_max || [0, 0, 0];
      const cx = (bmin[0] + bmax[0]) / 2, cy = (bmin[1] + bmax[1]) / 2;
      const zTop = bmax[2] + Math.max(bmax[0] - bmin[0], bmax[1] - bmin[1]);
      view.setView([cx, cy, zTop], [cx, cy, bmin[2]], 400);
    }} catch (e) {{
      console.error("[step1] top view error:", e);
      $msg("俯视失败：" + (e && e.message ? e.message : String(e)));
    }}
  }}
  function sideView() {{
    try {{
      const view = viewer.scene.view;
      const bmin = CACHE.source_bounds_min || [0, 0, 0];
      const bmax = CACHE.source_bounds_max || [0, 0, 0];
      const cx = (bmin[0] + bmax[0]) / 2, cy = (bmin[1] + bmax[1]) / 2;
      const cz = (bmin[2] + bmax[2]) / 2;
      const dx = Math.max(bmax[0] - bmin[0], bmax[1] - bmin[1]);
      view.setView([cx - dx, cy, cz], [cx, cy, cz], 400);
    }} catch (e) {{
      console.error("[step1] side view error:", e);
      $msg("侧视失败：" + (e && e.message ? e.message : String(e)));
    }}
  }}

  // ---- LOD diagnostics ----------------------------------------------------
  setInterval(function () {{
    if (!pointcloud) return;
    document.getElementById("visible").textContent =
      pointcloud.numVisiblePoints != null ? Number(pointcloud.numVisiblePoints).toLocaleString() : "—";
    document.getElementById("nodes").textContent =
      pointcloud.numVisibleNodes != null ? pointcloud.numVisibleNodes : "—";
  }}, 1000);

  // ---- boot ---------------------------------------------------------------
  async function start() {{
    try {{
      const metaResp = await fetch(CACHE.metadata_url, {{cache: "no-store"}});
      if (!metaResp.ok) throw new Error("viewer metadata HTTP " + metaResp.status);
      const metadata = await metaResp.json();
      if (!metadata || !Array.isArray(metadata.attributes)) throw new Error("viewer metadata invalid");

      viewer = new Potree.Viewer(renderArea);
      window.viewer = viewer;
      viewer.setControls(viewer.orbitControls);
      viewer.setEDLEnabled(false);
      viewer.setPointBudget(750000);
      viewer.setDescription("STEP 1 档距选择 — Potree LOD");
      viewer.loadGUI(function () {{ viewer.setLanguage("en"); }});

      const loaded = await Potree.loadPointCloud(CACHE.metadata_url, "source");
      pointcloud = loaded.pointcloud;
      // PROVEN migration (viewer_poc/p3a/viewer.js + poc_config.json):
      // Potree decodes the converter "rgb" attribute as "rgba"; setting
      // "rgba" selects real RGB color (the "rgb" name renders the default
      // purple).  The material is configured exactly as the proven P3A viewer
      // (which does NOT touch Potree.PointColorType — not exported in the
      // minified Potree 1.8.2 bundle).
      pointcloud.material.activeAttributeName = "rgba";
      pointcloud.material.pointSizeType = Potree.PointSizeType.ADAPTIVE;
      pointcloud.material.size = 1.0;
      pointcloud.material.minSize = 2.0;
      pointcloud.material.maxSize = 5.0;
      viewer.scene.addPointCloud(pointcloud);

      // Tight-bounds fit so the cloud is not a tiny dot.
      const posAttr = metadata.attributes.find(function (a) {{ return a.name === "position"; }});
      if (posAttr && posAttr.min && posAttr.max && pointcloud.pcoGeometry) {{
        const off = (metadata.offset || metadata.boundingBox.min || [0,0,0]);
        const box = pointcloud.pcoGeometry.boundingBox.clone();
        box.min.set(posAttr.min[0] - off[0], posAttr.min[1] - off[1], posAttr.min[2] - off[2]);
        box.max.set(posAttr.max[0] - off[0], posAttr.max[1] - off[1], posAttr.max[2] - off[2]);
        pointcloud.pcoGeometry.tightBoundingBox = box;
      }}
      requestAnimationFrame(function () {{ fitView(); }});

      document.getElementById("budget").oninput = function () {{
        const v = Number(this.value);
        document.getElementById("budget_val").textContent = Number(v).toLocaleString();
        viewer.setPointBudget(v);
      }};

      badge("ready");
      $status("Potree 加载完成：105M 点云 LOD 就绪。点击选择杆塔 A / B。");
      registerCanvasClick();
    }} catch (e) {{
      badge("error");
      showError((e && e.message) || String(e));
    }}
  }}

  document.getElementById("btn-a").onclick = function () {{ mode = "SELECT_A"; document.getElementById("btn-a").style.outline = "3px solid #ff5252"; document.getElementById("btn-b").style.outline = ""; $msg("请点击杆塔 A 的位置"); }};
  document.getElementById("btn-b").onclick = function () {{ mode = "SELECT_B"; document.getElementById("btn-b").style.outline = "3px solid #40c4ff"; document.getElementById("btn-a").style.outline = ""; $msg("请点击杆塔 B 的位置"); }};
  document.getElementById("btn-reset").onclick = async function () {{
    mode = "IDLE";
    poles = {{A: null, B: null}};
    document.getElementById("btn-confirm").disabled = true;
    document.getElementById("a_x").textContent = document.getElementById("a_y").textContent = document.getElementById("a_z").textContent = "—";
    document.getElementById("b_x").textContent = document.getElementById("b_y").textContent = document.getElementById("b_z").textContent = "—";
    document.getElementById("a_s").textContent = document.getElementById("b_s").textContent = "—";
    for (const p of ["A", "B"]) {{ if (markers[p]) {{ viewer.scene.scene.remove(markers[p]); markers[p] = null; }} }}
    await post("/reselect", {{}});
  }};
  document.getElementById("btn-fit").onclick = fitView;
  document.getElementById("btn-top").onclick = topView;
  document.getElementById("btn-side").onclick = sideView;
  document.getElementById("btn-confirm").onclick = async function () {{
    const btn = document.getElementById("btn-confirm");
    btn.disabled = true;
    $status("档距已确认，正在后台执行 导线/植被/单木 分析…");
    try {{
      await post("/confirm", {{}});
      setInterval(async function () {{
        const s = await (await fetch("/state")).json();
        if (s.step === "TREE_EDITING") window.location.href = "/step2";
        if (s.step === "DONE") window.location.href = "/step3";
        if (s.error) {{ $status("失败：" + s.error); btn.disabled = false; }}
      }}, 1200);
    }} catch (e) {{
      $msg("确认失败：" + e.message);
      btn.disabled = false;
    }}
  }};

  start();
}})();
</script>
</body>
</html>
"""


__all__ = ["potree_step1_html"]
