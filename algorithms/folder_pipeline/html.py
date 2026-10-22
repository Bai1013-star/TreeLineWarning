# ruff: noqa: E501  (long HTML/JS template content)
"""Three-step browser pages for the Folder pipeline.

The pages are served by ONE local server and share ONE workflow state object
(``/state``), so they are a single run/session — not three disconnected static
programs.  Plotly remains the bounded fallback for span selection and final
charts.  The tree editor uses the project's offline Giro3D / Three.js stack;
every colour is still assigned from a stable deterministic mapping.
"""

from __future__ import annotations

_PAGE_CSS = """
body{font-family:"Microsoft YaHei","Segoe UI",sans-serif;margin:0;background:#f5f6f8;color:#1a1a1a}
header{background:#1f2a44;color:#fff;padding:10px 18px;display:flex;align-items:center;gap:14px}
header h1{font-size:16px;margin:0}
header .step{font-size:12px;opacity:.85}
#status{font-size:12px;color:#ffd24a}
main{display:flex;gap:12px;padding:12px;flex-wrap:wrap}
.panel{background:#fff;border:1px solid #dde1e6;border-radius:6px;padding:10px;box-shadow:0 1px 2px rgba(0,0,0,.06)}
.plot-panel{flex:1 1 62%;min-width:420px;height:70vh}
.side{flex:1 1 28%;min-width:300px;display:flex;flex-direction:column;gap:12px}
button{background:#1f6feb;color:#fff;border:0;border-radius:4px;padding:8px 14px;font-size:13px;cursor:pointer}
button:disabled{background:#b8c1cc;cursor:not-allowed}
button.gray{background:#5f6b7a}
button.red{background:#d64545}
button.green{background:#2ea44f}
table{border-collapse:collapse;width:100%;font-size:12px}
th,td{border:1px solid #dde1e6;padding:4px 6px;text-align:left}
th{background:#eef1f5}
tr.sel td{background:#fff3bf}
.msg{font-size:12px;color:#b01212;min-height:18px}
.kpis{display:flex;gap:10px;flex-wrap:wrap}
.kpi{background:#fff;border:1px solid #dde1e6;border-radius:6px;padding:10px 14px;min-width:120px}
.kpi .v{font-size:20px;font-weight:700}
  .kpi .k{font-size:12px;color:#5f6b7a}
  a{color:#1f6feb}
  #tree-label-overlay{position:absolute;inset:0;pointer-events:none;z-index:10}
  .tree-label{position:absolute;transform:translate(-50%,-100%);pointer-events:auto;border:1px solid #64748b;border-radius:4px;background:rgba(255,255,255,.92);color:#172033;padding:2px 5px;font:700 12px/1.1 "Microsoft YaHei","Segoe UI",sans-serif;cursor:pointer;box-shadow:0 1px 3px rgba(0,0,0,.28)}
  .tree-label.selected{border:2px solid #d97706;background:#fff3a3;color:#111827;font-size:15px;padding:3px 7px;box-shadow:0 1px 5px rgba(146,64,14,.48)}
  .tree-review-main{flex-wrap:nowrap;height:calc(100vh - 48px);min-height:520px;min-width:1180px;overflow:visible}
  .tree-review-main .plot-panel{flex:1 0 780px;min-width:780px;height:auto;min-height:500px;padding:0;overflow:hidden;background:#071015}
  .tree-review-main .side{flex:0 0 360px;min-width:360px;max-width:360px;overflow:auto}
  #tree-review-stage{position:relative;width:100%;height:100%;min-height:500px;overflow:hidden}
  #plot3d{position:absolute;inset:0}
  .tree-review-canvas{display:block;width:100%;height:100%;touch-action:none;cursor:grab}
  #viewer-toolbar{position:absolute;z-index:12;top:8px;left:8px;display:flex;gap:5px;flex-wrap:wrap;pointer-events:auto}
  #viewer-toolbar button{padding:6px 9px;background:rgba(31,42,68,.9);border:1px solid #7f94a8}
  #viewer-renderer-status{position:absolute;z-index:11;left:9px;bottom:8px;padding:4px 7px;border-radius:4px;background:rgba(7,16,21,.84);color:#b7d5df;font:11px/1.3 Consolas,monospace;pointer-events:none}
  .lasso-overlay{position:absolute;inset:0;z-index:30;width:100%;height:100%;cursor:crosshair;touch-action:none;user-select:none}
  @media(max-width:720px){.tree-review-main{gap:6px;padding:6px}.tree-review-main .side{min-width:218px}.tree-review-main .plot-panel{min-width:270px}}
  """


def _page_shell(
    title: str,
    step: str,
    body: str,
    script: str,
    *,
    plotly: bool = True,
    external_scripts: tuple[str, ...] = (),
) -> str:
    script_tags = []
    if plotly:
        script_tags.append('<script src="/plotly.js"></script>')
    script_tags.extend(f'<script src="{source}"></script>' for source in external_scripts)
    dependencies = "\n".join(script_tags)
    return f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>{_PAGE_CSS}</style>
</head>
<body>
<header>
  <h1>TreeLineWarning — 文件夹到 Excel</h1>
  <span class="step">{step}</span>
  <span id="status"></span>
</header>
{body}
{dependencies}
<script>
{script}
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# STEP 1 — manual span selection
# ---------------------------------------------------------------------------


def step1_html() -> str:
    body = """
<div style="background:#8b0000;color:#fff;padding:6px 16px;font-size:13px">
  ⚠ LEGACY / DEBUG / FALLBACK 选择器（Plotly 固定采样）。若 WebGL 可用，请使用
  主选页 /step1 的 Potree LOD 查看器。
</div>
<main>
  <div class="panel plot-panel">
    <div id="plot2d" style="height:48%"></div>
    <div id="plot3d" style="height:48%"></div>
  </div>
  <div class="side">
    <div class="panel">
      <h3>第一步：选择档距</h3>
      <p style="font-size:12px;color:#5f6b7a">
        在俯视图中点击两根真实杆塔。先按“选择杆塔 A”，再按“选择杆塔 B”，
        确认候选后点击“确认档距并继续”。
      </p>
      <div style="display:flex;gap:8px;flex-wrap:wrap">
        <button id="btn-select-a">选择杆塔 A</button>
        <button id="btn-select-b">选择杆塔 B</button>
        <button id="btn-reselect" class="gray">重新选择</button>
      </div>
      <div id="clickinfo" style="font-size:12px;margin-top:8px"></div>
      <div id="refinemessage" class="msg" style="margin-top:6px"></div>
      <div style="margin-top:12px">
        <button id="btn-confirm" class="green">确认档距并继续</button>
      </div>
      <div id="science-progress" style="font-size:12px;margin-top:8px"></div>
    </div>
  </div>
</main>
"""
    script = """
let selectionMode = "IDLE";
let clicks = {"A": null, "B": null};
let refreshTimer = null;

function setMode(mode) {
  selectionMode = mode;
  const a = document.getElementById("btn-select-a");
  const b = document.getElementById("btn-select-b");
  a.style.outline = mode === "SELECT_A" ? "3px solid #1f6feb" : "";
  b.style.outline = mode === "SELECT_B" ? "3px solid #1f6feb" : "";
}
async function post(path, payload) {
  const r = await fetch(path, {method:"POST",
    headers:{"Content-Type":"application/json"},
    body: JSON.stringify(payload)});
  if (!r.ok) {
    const t = await r.json().catch(()=>({}));
    throw new Error(t.error || ("请求失败 " + path));
  }
  return r.json();
}
function setStatus(text) {
  document.getElementById("status").textContent = text;
}
async function refreshState() {
  try {
    const s = await (await fetch("/state")).json();
    setStatus(s.step_label + (s.current_message ? " — " + s.current_message : ""));
    if (s.error) {
      document.getElementById("refinemessage").textContent = "失败：" + s.error;
      setStatus("失败：" + s.error);
    }
    if (s.step === "TREE_EDITING") {
      clearInterval(refreshTimer);
      window.location.href = "/step2";
    }
    if (s.step === "DONE") {
      clearInterval(refreshTimer);
      window.location.href = "/step3";
    }
  } catch (e) { /* server starting */ }
}

async function loadPoints() {
  const data = await (await fetch("/points")).json();
  const xyz = data.xyz, rgb = data.rgb;
  const x = xyz.map(p=>p[0]), y = xyz.map(p=>p[1]), z = xyz.map(p=>p[2]);
  const color = rgb.map(c=>`rgb(${c[0]},${c[1]},${c[2]})`);
  const marker = {size:2, color:color, opacity:0.9};
  Plotly.newPlot("plot2d", [{x:x, y:y, type:"scatter", mode:"markers",
    marker:marker, name:"点云(俯视)"}],
    {title:{text:"俯视图：点击杆塔",font:{size:14}},
     xaxis:{title:"X (m)", scaleanchor:"x", scaleratio:1},
     yaxis:{title:"Y (m)"},
     margin:{l:40,r:20,t:36,b:30}, scrollZoom:true,
     hovermode:"closest"}, {responsive:true});
  const stride = Math.max(1, Math.floor(xyz.length / 30000));
  const x3=[],y3=[],z3=[],c3=[];
  for (let i=0;i<xyz.length;i+=stride) { x3.push(x[i]);y3.push(y[i]);z3.push(z[i]);c3.push(color[i]); }
  Plotly.newPlot("plot3d", [{x:x3,y:y3,z:z3,type:"scatter3d",mode:"markers",
    marker:{size:1.4,color:c3,opacity:0.85}, name:"3D预览"}],
    {title:{text:"3D 预览（显示采样）",font:{size:14}},
     scene:{aspectmode:"data"},
     margin:{l:0,r:0,t:36,b:0}}, {responsive:true});
  document.getElementById("plot2d").on("plotly_click", data => {
    if (!data || !data.points || !data.points.length) return;
    const p = data.points[0];
    registerClick(p.x, p.y, p.z);
  });
  const gd = document.getElementById("plot2d");
  gd.addEventListener("click", ev => {
    const rect = gd.getBoundingClientRect();
    const x = ev.clientX - rect.left, y = ev.clientY - rect.top;
    try {
      const xa = gd._fullLayout.xaxis, ya = gd._fullLayout.yaxis;
      registerClick(xa.p2c(x), ya.p2c(y), null);
    } catch(e) {}
  });
}
async function registerClick(x, y, z) {
  if (selectionMode === "IDLE") {
    document.getElementById("refinemessage").textContent = "请先按“选择杆塔 A”或“选择杆塔 B”";
    return;
  }
  const pole = selectionMode === "SELECT_A" ? "A" : "B";
  try {
    await post("/click", {pole:pole, x:x, y:y, z:z});
    document.getElementById("clickinfo").textContent =
      `杆塔 ${pole} 已记录：(${x.toFixed(2)}, ${y.toFixed(2)})`;
    const r = await post("/refine", {pole:pole});
    if (r.defensible) {
      document.getElementById("refinemessage").textContent =
        `杆塔 ${pole} 有效：高 ${r.height_m.toFixed(2)} m，底部 ${r.base_xyz_absolute_m[0].toFixed(1)}, ${r.base_xyz_absolute_m[1].toFixed(1)}`;
    } else {
      document.getElementById("refinemessage").textContent =
        `杆塔 ${pole} 识别失败：${r.reason}`;
    }
    setMode("IDLE");
    if (pole === "A") clicks.A = {x,y};
    if (pole === "B") clicks.B = {x,y};
  } catch(e) {
    document.getElementById("refinemessage").textContent = "失败：" + e.message;
  }
}
async function confirmSpan() {
  const btn = document.getElementById("btn-confirm");
  btn.disabled = true;
  document.getElementById("science-progress").textContent = "档距已确认，正在后台执行 导线/植被/单木 分析...";
  try {
    await post("/confirm", {});
  } catch(e) {
    document.getElementById("refinemessage").textContent = "确认失败：" + e.message;
    btn.disabled = false;
    return;
  }
  refreshTimer = setInterval(refreshState, 1200);
}

document.getElementById("btn-select-a").onclick = () => setMode("SELECT_A");
document.getElementById("btn-select-b").onclick = () => setMode("SELECT_B");
document.getElementById("btn-reselect").onclick = async () => {
  await post("/reselect", {});
  clicks = {"A":null,"B":null};
  document.getElementById("clickinfo").textContent = "";
  document.getElementById("refinemessage").textContent = "";
  setMode("IDLE");
  loadPoints();
};
document.getElementById("btn-confirm").onclick = confirmSpan;
loadPoints();
refreshTimer = setInterval(refreshState, 1500);
"""
    return _page_shell("STEP 1 人工选择档距", "STEP 1 人工选择档距", body, script)


# ---------------------------------------------------------------------------
# STEP 2 — tree editor (DELETE / MERGE / UNDO)
# ---------------------------------------------------------------------------


def _legacy_step2_plotly_html() -> str:
    body = """
<main class="tree-review-main">
  <button id="btn-review-back" type="button" class="gray" style="position:fixed;right:18px;top:6px;z-index:40">← 返回上一步 / 退出复核</button>
  <div class="panel plot-panel">
    <div id="tree-review-stage">
      <div id="plot3d"></div>
      <div id="tree-label-overlay" aria-label="树编号标签"></div>
      <div id="viewer-toolbar" aria-label="三维视图控制">
        <button id="btn-view-reset" type="button">重置</button>
        <button id="btn-view-top" type="button">俯视</button>
        <button id="btn-view-side" type="button">侧视</button>
      </div>
      <div id="viewer-renderer-status">正在初始化 Giro3D / Three.js…</div>
    </div>
  </div>
  <div class="side">
    <div class="panel">
      <h3>第二步：人工审核单木</h3>
      <p style="font-size:12px;color:#5f6b7a">
        点击选中单木。局部剔除会先红色预览，确认后才修改完整点成员。
      </p>
      <div id="review-status" style="font-size:13px;font-weight:600;margin-bottom:8px">正在加载单木复核数据...</div>
      <div id="current-tree" style="font-size:12px;color:#5f6b7a;margin-bottom:8px">当前：未选择</div>
      <div style="display:flex;gap:8px;flex-wrap:wrap">
        <button id="btn-prev" class="gray">上一棵</button>
        <button id="btn-next" class="gray">下一棵</button>
        <button id="btn-delete" class="red">删除单木</button>
        <button id="btn-merge">合并选中</button>
        <button id="btn-split" type="button">分开粘连树</button>
        <button id="btn-trim-component">剔除误分团块</button>
        <button id="btn-trim-lasso">套索剔除</button>
        <button id="btn-undo" class="gray">撤销</button>
      </div>
      <div id="selection-state" style="font-size:12px;color:#5f6b7a;margin-top:7px">未选择单木</div>
      <div id="lasso-mode-indicator" style="display:none;margin-top:8px;padding:8px;border-radius:4px;background:#fff3cd;color:#6b4f00;font-weight:600;white-space:pre-line"></div>
      <div id="editmsg" class="msg" style="margin-top:6px"></div>
      <details id="error-details" style="display:none;font-size:11px;color:#5f6b7a;margin-top:4px"><summary>查看详情</summary><pre id="error-detail-text" style="white-space:pre-wrap"></pre></details>
      <div id="trim-preview" style="display:none;margin-top:8px;padding:8px;border:2px solid #d64545;border-radius:4px;background:#fff5f5">
        <div id="trim-summary" style="font-size:13px;margin-bottom:7px"></div>
        <button id="btn-trim-confirm" class="red">确认剔除</button>
        <button id="btn-trim-cancel" class="gray">取消</button>
      </div>
      <label style="display:flex;align-items:center;gap:6px;margin-top:10px;font-size:12px">
        <input id="toggle-labels" type="checkbox" checked>
        显示树编号
      </label>
      <div style="margin-top:10px">
        <button id="btn-confirm" class="green">完成复核并生成 Excel</button>
      </div>
      <div id="progress" style="font-size:12px;margin-top:8px"></div>
    </div>
    <div class="panel">
      <h3>单木列表 <span id="treecount" style="font-size:12px;color:#5f6b7a"></span></h3>
      <div id="treelist" style="max-height:40vh;overflow:auto">
        <table id="treetable">
          <thead><tr><th>ID</th><th>点数</th><th>高度(m)</th><th>编辑状态</th></tr></thead>
          <tbody></tbody>
        </table>
      </div>
    </div>
  </div>
</main>
"""
    script = """
let treeMap = new Map();
let selected = new Set();
let activeTreeId = null;
let currentSnapshot = null;
let editorDisplay = null;
let plotClickBound = false;
let showLabels = true;
let refreshTimer = null;
let trimMode = "IDLE";
let activePreview = null;
let lassoOverlay = null;
let reviewed = new Set();
let editorLoadPromise = null;
// Trace indexes are never derived from a tree id or a trace position after a
// refresh.  The current renderer intentionally uses one sampled TREE trace;
// every visible tree id maps explicitly to that trace.
const viewerState = {
  currentTreeId: null,
  treeTraceById: new Map(),
  activePreviewType: null,
  activePreviewTreeId: null,
  previewPointKeys: new Set(),
  lassoMode: false,
  mergeSelection: selected,
};
let treeTraceIndex = -1;
let displayPreviewXYZ = [];
let previewRequestId = 0;
const reviewSpanId=typeof URLSearchParams==="function"?new URLSearchParams(window.location.search||"").get("span_id"):null;

// Capability-aware LOD is deliberately display-only.  Weak WebView/GPU
// combinations receive fewer representative points; exact memberships never
// leave the backend sampling path.
function displayBudget(){
  const device=typeof navigator==="undefined"?{}:navigator;
  const cores=Number(device.hardwareConcurrency||4), memory=Number(device.deviceMemory||4);
  if(cores<=4||memory<=4) return 3500;
  if(cores<=8||memory<=8) return 6500;
  return 10000;
}

function setStatus(t){document.getElementById("status").textContent=t;}
function setEditingBusy(busy){
  ["btn-prev","btn-next","btn-delete","btn-merge","btn-trim-component","btn-trim-lasso","btn-undo","btn-trim-confirm","btn-confirm"].forEach(id=>{
    const element=document.getElementById(id);if(element)element.disabled=Boolean(busy);
  });
}
function showFailure(error,prefix="失败："){
  document.getElementById("editmsg").textContent=prefix+(error?.message||String(error));
  const details=document.getElementById("error-details"),text=document.getElementById("error-detail-text");
  if(details&&text&&error?.details){text.textContent=error.details;details.style.display="block";}
  else if(details){details.style.display="none";}
}
function colorOf(treeId){
  const hue = ((treeId * 137.508) % 360) / 360;
  const c = hsvToRgb(hue, 0.68, 0.92);
  return `rgb(${Math.round(c[0]*255)},${Math.round(c[1]*255)},${Math.round(c[2]*255)})`;
}
function hsvToRgb(h,s,v){
  const i=Math.floor(h*6), f=h*6-i, p=v*(1-s), q=v*(1-f*s), t=v*(1-(1-f)*s);
  switch(i%6){case 0:return[v,t,p];case 1:return[q,v,p];case 2:return[p,v,t];case 3:return[p,q,v];case 4:return[t,p,v];default:return[v,p,q];}
}
async function post(path, payload){
  const r=await fetch(path,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(payload||{})});
  if(!r.ok){const t=await r.json().catch(()=>({}));const e=new Error(t.error?.message||t.error||("请求失败 "+path));e.details=t.error?.details||"";throw e;}
  return r.json();
}
function scrollToTreeRow(treeId){
  const row=document.querySelector(`#treetable tr[data-tid="${treeId}"]`);
  if(row) row.scrollIntoView({block:"nearest",inline:"nearest"});
}
function renderTrees(snap, scrollToActive=false){
  currentSnapshot = snap;
  treeMap = new Map();
  const tbody=document.querySelector("#treetable tbody");
  tbody.innerHTML="";
  const currentIds = new Set((snap.trees||[]).map(t=>Number(t.tree_id)));
  selected = new Set([...selected].filter(id=>currentIds.has(Number(id))));
  reviewed = new Set([...reviewed].filter(id=>currentIds.has(Number(id))));
  if(activeTreeId!=null && !currentIds.has(Number(activeTreeId))) activeTreeId=null;
  (snap.trees||[]).forEach(t=>{
    treeMap.set(Number(t.tree_id), t);
    const tr=document.createElement("tr");
    tr.dataset.tid=t.tree_id;
    if(selected.has(Number(t.tree_id)) || activeTreeId===Number(t.tree_id)) tr.className="sel";
    tr.innerHTML=`<td>${t.tree_id}</td><td>${t.point_count}</td><td>${
      t.apex_height_m!=null?t.apex_height_m.toFixed(1):"—"}</td><td>${t.merged?"合并":"—"}</td>`;
    tr.onclick=()=>selectTree(t.tree_id, true, true);
    tbody.appendChild(tr);
  });
  document.getElementById("treecount").textContent = `共 ${snap.current_tree_count} 棵`;
  updateReviewStatus();
  if(scrollToActive && activeTreeId!=null) setTimeout(()=>scrollToTreeRow(activeTreeId),0);
}
function updateReviewStatus(){
  const total=treeMap.size, reviewedCount=[...reviewed].filter(id=>treeMap.has(Number(id))).length;
  document.getElementById("review-status").textContent=`${total} 棵树 · 已复核 ${reviewedCount} · 待复核 ${Math.max(0,total-reviewedCount)}`;
  document.getElementById("current-tree").textContent=activeTreeId==null?"当前：未选择":`当前：树 ${activeTreeId}`;
  const ids=[...selected].sort((a,b)=>a-b);
  document.getElementById("selection-state").textContent=ids.length?`已选择 ${ids.length} 棵树：${ids.join("、")}`:"未选择单木";
  document.getElementById("btn-delete").textContent=ids.length===1?`删除树 ${ids[0]}`:"删除单木";
  document.getElementById("btn-merge").textContent=ids.length>1?`合并选中（${ids.length} 棵）`:"合并选中";
}
function treeIdFromPlotPoint(pt){
  const raw=pt.customdata!=null?pt.customdata:pt.data?.customdata?.[pt.pointNumber];
  const tid=Number(raw);
  return Number.isFinite(tid)?tid:null;
}
function selectTree(treeId, toggle=true, scroll=false){
  const tid=Number(treeId);
  if(!treeMap.has(tid)) return;
  if(activeTreeId!==tid) previewRequestId++;
  if(activePreview && viewerState.activePreviewTreeId!==tid) void cancelTrim("已切换单木，已取消待剔除预览。");
  if(activeTreeId!=null && activeTreeId!==tid) reviewed.add(Number(activeTreeId));
  activeTreeId=tid;viewerState.currentTreeId=tid;
  if(toggle){
    if(selected.has(tid)) selected.delete(tid); else selected.add(tid);
  }else{
    selected=new Set([tid]);
  }
  applyTreeSelectionStyles();
  updateReviewStatus();
  if(scroll) scrollToTreeRow(activeTreeId);
  // This must stay local: no backend call and no Plotly figure rebuild on a
  // tree switch.  `restyle(selectedpoints)` only updates the highlight buffer.
  updatePlotSelection();
  scheduleHtmlLabels();
}
function applyTreeSelectionStyles(){
  document.querySelectorAll("#treetable tr[data-tid]").forEach(row=>{
    const tid=Number(row.dataset.tid);
    row.className=(selected.has(tid)||activeTreeId===tid)?"sel":"";
  });
}
function selectionHighlightIds(){
  const ids=new Set(selected);
  if(activeTreeId!=null) ids.add(Number(activeTreeId));
  return ids;
}
function selectedPointRows(treeIds){
  const ids=selectionHighlightIds();
  if(!editorDisplay || !Array.isArray(editorDisplay.trees3d?.tree_id_by_index)) return [];
  return editorDisplay.trees3d.tree_id_by_index.reduce((rows,tid,index)=>{
    if(ids.has(Number(tid))) rows.push(index);
    return rows;
  },[]);
}
function updatePlotSelection(){
  const gd=document.getElementById("plot3d");
  const traceIndex=viewerState.treeTraceById.get(Number(activeTreeId));
  if(traceIndex==null||!gd||typeof Plotly==="undefined"||typeof Plotly.restyle!=="function") return;
  const rows=selectedPointRows();
  try{Plotly.restyle(gd,{selectedpoints:[rows]},[traceIndex]);}catch(_){/* next full data refresh recovers */}
}
function pointKey(point){return point.map(value=>Number(value).toPrecision(15)).join(",");}
function displayTreeColors(){
  const cloud=editorDisplay?.trees3d||{}, colors=Array.isArray(cloud.color)?cloud.color:[];
  const ids=Array.isArray(cloud.tree_id_by_index)?cloud.tree_id_by_index:[];
  return colors.map((color,index)=>viewerState.previewPointKeys.has(pointKey(cloud.xyz?.[index]||[])) &&
    Number(ids[index])===Number(viewerState.activePreviewTreeId)?"#ff0000":color);
}
function updatePreviewTrace(points, treeId=null){
  const previousTreeId=viewerState.activePreviewTreeId;
  displayPreviewXYZ=Array.isArray(points)?points:[];
  const gd=document.getElementById("plot3d");
  viewerState.previewPointKeys=new Set(displayPreviewXYZ.map(pointKey));
  const previewTreeId=displayPreviewXYZ.length?Number(treeId):previousTreeId;
  viewerState.activePreviewTreeId=displayPreviewXYZ.length?previewTreeId:null;
  const traceIndex=viewerState.treeTraceById.get(Number(previewTreeId));
  if(traceIndex==null||!gd||typeof Plotly==="undefined"||typeof Plotly.restyle!=="function") return;
  // Preview points are recoloured in the base sampled cloud, not rendered as a
  // second red scatter trace.  This prevents a red+original ghost point cloud.
  try{Plotly.restyle(gd,{"marker.color":[displayTreeColors()]},[traceIndex]);}catch(_){/* next full data refresh recovers */}
}
function plotLayout(){
  const gd=document.getElementById("plot3d");
  const previous=gd.layout?.scene?.camera || gd._fullLayout?.scene?.camera;
  const layout={title:{text:"单木审核（不同颜色=不同树）",font:{size:14}},
    scene:{aspectmode:"data"},margin:{l:0,r:0,t:36,b:0}};
  if(previous) layout.scene.camera=previous;
  return layout;
}
function renderHtmlLabels(){
  const overlay=document.getElementById("tree-label-overlay");
  if(!overlay) return;
  overlay.innerHTML="";
  const labels=editorDisplay?.trees3d?.labels;
  if(!Array.isArray(labels) || (!showLabels && !selectionHighlightIds().size)) return;
  const visible=showLabels?labels:labels.filter(label=>selectionHighlightIds().has(Number(label.tree_id)));
  let matrix;
  try{matrix=currentViewProjection();}catch(e){return;}
  const host=document.getElementById("plot3d"), rect=host.getBoundingClientRect();
  if(!rect.width || !rect.height) return;
  visible.forEach(label=>{
    const point=label.xyz, clip=[
      matrix[0][0]*point[0]+matrix[0][1]*point[1]+matrix[0][2]*point[2]+matrix[0][3],
      matrix[1][0]*point[0]+matrix[1][1]*point[1]+matrix[1][2]*point[2]+matrix[1][3],
      matrix[2][0]*point[0]+matrix[2][1]*point[1]+matrix[2][2]*point[2]+matrix[2][3],
      matrix[3][0]*point[0]+matrix[3][1]*point[1]+matrix[3][2]*point[2]+matrix[3][3]
    ];
    if(!clip.every(Number.isFinite) || clip[3]<=0) return;
    const ndcX=clip[0]/clip[3], ndcY=clip[1]/clip[3];
    if(!Number.isFinite(ndcX)||!Number.isFinite(ndcY)||ndcX<-1.2||ndcX>1.2||ndcY<-1.2||ndcY>1.2) return;
    const button=document.createElement("button");
    button.type="button";button.className="tree-label";
    if(selectionHighlightIds().has(Number(label.tree_id))) button.classList.add("selected");
    button.textContent=String(label.tree_id);
    button.title=`选择树 ${label.tree_id}`;
    button.setAttribute("aria-label",`选择树 ${label.tree_id}`);
    button.style.left=`${(ndcX+1)*50}%`;button.style.top=`${(1-ndcY)*50}%`;
    button.onclick=event=>{event.stopPropagation();selectTree(Number(label.tree_id),true,true);};
    overlay.appendChild(button);
  });
}
let labelRenderFrame=null;
function scheduleHtmlLabels(){
  if(labelRenderFrame!=null) return;
  const paint=()=>{labelRenderFrame=null;renderHtmlLabels();};
  labelRenderFrame=typeof requestAnimationFrame==="function"?requestAnimationFrame(paint):setTimeout(paint,0);
}
function renderEditorPlot(){
  if(!editorDisplay) return;
  const data=editorDisplay, bg=data.background, wire=data.wire, tower=data.tower, wires3d=data.wires3d;
  const trees3d=data.trees3d||{};
  const traces=[];treeTraceIndex=-1;viewerState.treeTraceById.clear();
  if(bg && Array.isArray(bg.xyz) && bg.xyz.length){
    const c=(Array.isArray(bg.rgb)?bg.rgb:[]).map(v=>`rgb(${v[0]},${v[1]},${v[2]})`);
    traces.push({x:bg.xyz.map(p=>p[0]),y:bg.xyz.map(p=>p[1]),z:bg.xyz.map(p=>p[2]),
      type:"scatter3d",mode:"markers",name:"背景",meta:{role:"BACKGROUND"},hoverinfo:"skip",marker:{size:1,color:c,opacity:0.4}});
  }
  if(wire && Array.isArray(wire.xyz) && wire.xyz.length){
    traces.push({x:wire.xyz.map(p=>p[0]),y:wire.xyz.map(p=>p[1]),z:wire.xyz.map(p=>p[2]),
      type:"scatter3d",mode:"markers",name:"导线",meta:{role:"WIRE"},hoverinfo:"skip",marker:{size:2,color:"#ff1744",opacity:1}});
  }
  if(wires3d && Array.isArray(wires3d.xyz) && wires3d.xyz.length){
    traces.push({x:wires3d.xyz.map(p=>p[0]),y:wires3d.xyz.map(p=>p[1]),z:wires3d.xyz.map(p=>p[2]),
      type:"scatter3d",mode:"lines",name:"导线曲线",meta:{role:"WIRE_CURVE"},hoverinfo:"skip",line:{color:"#ff1744",width:3}});
  }
  if(tower && Array.isArray(tower.xyz) && tower.xyz.length){
    traces.push({x:tower.xyz.map(p=>p[0]),y:tower.xyz.map(p=>p[1]),z:tower.xyz.map(p=>p[2]),
      type:"scatter3d",mode:"markers+text",name:"杆塔",meta:{role:"TOWER"},text:tower.labels||[],hoverinfo:"skip",
      marker:{size:7,color:"#ff9800",symbol:"diamond"}});
  }
  if(Array.isArray(trees3d.xyz) && trees3d.xyz.length){
    const highlightRows=selectedPointRows();
    const treeTrace={x:trees3d.xyz.map(p=>p[0]),y:trees3d.xyz.map(p=>p[1]),z:trees3d.xyz.map(p=>p[2]),
      type:"scatter3d",mode:"markers",name:"单木",meta:{role:"TREE",tree_ids:[...new Set(trees3d.tree_id_by_index||[])].map(Number)},customdata:trees3d.tree_id_by_index,
      hovertemplate:"树 %{customdata}<extra></extra>",marker:{size:1.6,color:displayTreeColors(),opacity:0.95},selectedpoints:highlightRows,
      selected:{marker:{size:3.8,opacity:1}},unselected:{marker:{opacity:0.18}}};
    treeTraceIndex=traces.length;traces.push(treeTrace);
    (trees3d.tree_id_by_index||[]).forEach(treeId=>viewerState.treeTraceById.set(Number(treeId),treeTraceIndex));
  }
  if(!traces.length){
    document.getElementById("plot3d").textContent="暂无可审核的 Tree display sample；完成 Batch 后刷新。";
    return;
  }
  const gd=document.getElementById("plot3d");
  let renderPromise=null;
  // Full construction is only allowed after new display data (load, edit,
  // merge, delete, undo). Selection and preview use restyle above.
  plotClickBound=false;
  renderPromise=Plotly.react(gd,traces,plotLayout(),{responsive:true,displaylogo:false});
  bindPlotClicks();
  if(renderPromise&&typeof renderPromise.then==="function") renderPromise.then(scheduleHtmlLabels);
  else scheduleHtmlLabels();
}
function bindPlotClicks(){
  const gd=document.getElementById("plot3d");
  if(!gd || plotClickBound || typeof gd.on!=="function") return;
  plotClickBound=true;
  gd.on("plotly_click", data=>{
    if(!data.points||!data.points.length) return;
    const pt=data.points[0], name=pt.data?.name||pt.fullData?.name||"";
    const tid=name==="树编号"||name==="选中树编号"?treeIdFromPlotPoint(pt):
      (name==="单木"?Number(editorDisplay?.trees3d?.tree_id_by_index?.[pt.pointNumber]):null);
    if(tid==null || !Number.isFinite(Number(tid))) return;
    if(name==="单木" && trimMode==="COMPONENT"){
      if(selected.size!==1 || !selected.has(Number(tid))){
        document.getElementById("editmsg").textContent="请点击当前选中单木中的误分点团。";
        return;
      }
      previewComponent(Number(tid),[Number(pt.x),Number(pt.y),Number(pt.z)]);
      return;
    }
    selectTree(Number(tid),true,true);
  });
  gd.on("plotly_relayout",scheduleHtmlLabels);
  gd.on("plotly_relayouting",scheduleHtmlLabels);
}
async function refreshTrees(){
  try{
    const s=await (await fetch("/state")).json();
    setStatus(s.step_label + (s.current_message?" — "+s.current_message:""));
    if(s.step==="FINALIZING") document.getElementById("progress").textContent=s.current_message||s.step_label;
    if(s.error){
      const error=new Error(s.error);error.details=s.error_details||"";showFailure(error);
      document.getElementById("progress").textContent=s.current_message||"复核结果生成失败";
      setEditingBusy(false);
    }
    if(s.step==="DONE"){clearInterval(refreshTimer);window.location.href="/step3";}
    if(s.tree_snapshot) renderTrees(s.tree_snapshot);
  }catch(e){}
}
async function loadEditorData(){
  if(editorLoadPromise) return editorLoadPromise;
  editorLoadPromise=(async()=>{
  if(!editorDisplay) document.getElementById("editmsg").textContent="正在加载单木列表与 3D 预览...";
  const spanQuery=reviewSpanId?`&span_id=${encodeURIComponent(reviewSpanId)}`:"";
  const data=await (await fetch(`/api/editor_data?display_budget=${displayBudget()}${spanQuery}`)).json();
  // Payload contract: each display layer is an optional dict `{xyz:[...], ...}`
  // (background/wire/wires3d/tower) and the tree 3D cloud lives under
  // `trees3d` `{xyz,color,tree_id_by_index}` — NOT under `trees`, which is the
  // right-panel tree LIST.  Guard every layer independently so a missing or
  // empty layer renders as an empty trace instead of throwing `.length` of
  // undefined and blanking the whole viewer.
  editorDisplay=data;
  const currentIds=new Set((data.trees||[]).map(t=>Number(t.tree_id)));
  selected=new Set([...selected].filter(id=>currentIds.has(Number(id))));
  if(activeTreeId!=null && !currentIds.has(Number(activeTreeId))) activeTreeId=null;
  renderTrees({trees:data.trees||[],current_tree_count:(data.trees||[]).length}, false);
  renderEditorPlot();
  if(document.getElementById("editmsg").textContent.startsWith("正在加载")) document.getElementById("editmsg").textContent="";
  return data;
  })();
  try{return await editorLoadPromise;}finally{editorLoadPromise=null;}
}
function requireOneTree(){
  if(selected.size!==1){document.getElementById("editmsg").textContent="请先选择一棵单木。";return null;}
  return [...selected][0];
}
function setTrimMode(mode,message){
  trimMode=mode;viewerState.lassoMode=mode==="LASSO";
  document.getElementById("btn-trim-component").style.outline=mode==="COMPONENT"?"3px solid #d64545":"";
  document.getElementById("btn-trim-lasso").style.outline=mode==="LASSO"?"3px solid #d64545":"";
  document.getElementById("editmsg").textContent=message||"";
  const indicator=document.getElementById("lasso-mode-indicator");
  indicator.style.display=mode==="LASSO"?"block":"none";
  indicator.textContent=mode==="LASSO"?"套索模式：拖动鼠标圈选要剔除的区域\\nEsc 取消":"";
}
async function showPreview(preview){
  activePreview=preview;viewerState.activePreviewType=trimMode;viewerState.activePreviewTreeId=Number(preview.tree_id);setTrimMode("PREVIEW","");removeLassoOverlay();
  const percent=(preview.fraction*100).toFixed(1);
  const warning=preview.large_removal_warning?" 本次将移除该树较大比例的点，请确认选择是否正确。":"";
  document.getElementById("trim-summary").textContent=`待剔除 ${preview.point_count.toLocaleString()} 个点，占当前单木 ${percent}%。${warning}`;
  document.getElementById("trim-preview").style.display="block";
  updatePreviewTrace(preview.preview_xyz||displayPreviewXYZ,preview.tree_id);
}
async function previewComponent(treeId,seed){
  try{document.getElementById("editmsg").textContent="正在分析当前单木的完整点成员...";
    const requestId=++previewRequestId,preview=await post("/api/trim/component-preview",{tree_id:treeId,seed_xyz:seed});
    if(requestId!==previewRequestId||Number(activeTreeId)!==Number(treeId)){await post("/api/trim/cancel",{preview_id:preview.preview_id});return;}
    await showPreview(preview);
  }catch(e){setTrimMode("IDLE",e.message);}
}
function matrixMultiply(a,b){
  const out=new Array(16).fill(0);
  for(let c=0;c<4;c++)for(let r=0;r<4;r++)for(let k=0;k<4;k++)out[c*4+r]+=a[k*4+r]*b[c*4+k];
  return out;
}
function currentViewProjection(){
  const gp=document.getElementById("plot3d")._fullLayout?.scene?._scene?.glplot;
  const cp=gp?.cameraParams;
  if(!cp?.view||!cp?.projection||!cp?.model) throw new Error("当前 3D 相机状态不可用，请稍后重试。");
  const columnMajor=matrixMultiply(cp.projection,matrixMultiply(cp.view,cp.model));
  const raw=Array.from({length:4},(_,r)=>Array.from({length:4},(_,c)=>columnMajor[c*4+r]));
  const scene=document.getElementById("plot3d")._fullLayout.scene;
  const internal=scene?._scene;
  const scale=internal?.dataScale;
  const ranges=[scene?.xaxis?.range,scene?.yaxis?.range,scene?.zaxis?.range];
  if(!scale || scale.length!==3 || ranges.some(range=>!range||range.length!==2)) return raw;
  const normalize=[
    [Number(scale[0]),0,0,0],
    [0,Number(scale[1]),0,0],
    [0,0,Number(scale[2]),0],
    [0,0,0,1]
  ];
  return Array.from({length:4},(_,r)=>Array.from({length:4},(_,c)=>
    raw[r].reduce((sum,value,k)=>sum+value*normalize[k][c],0)));
}
function removeLassoOverlay(){if(lassoOverlay){lassoOverlay.remove();lassoOverlay=null;}}
function pointInPolygon(point,polygon){
  let inside=false;
  for(let i=0,j=polygon.length-1;i<polygon.length;j=i++){
    const a=polygon[i],b=polygon[j];
    if(((a[1]>point[1])!==(b[1]>point[1])) && point[0]<(b[0]-a[0])*(point[1]-a[1])/(b[1]-a[1])+a[0]) inside=!inside;
  }
  return inside;
}
function displayLassoPreview(treeId,matrix,polygon){
  const cloud=editorDisplay?.trees3d||{}, xyz=cloud.xyz||[], ids=cloud.tree_id_by_index||[];
  const preview=[];
  for(let i=0;i<xyz.length;i++){
    if(Number(ids[i])!==Number(treeId)) continue;
    const p=xyz[i], clip=[
      matrix[0][0]*p[0]+matrix[0][1]*p[1]+matrix[0][2]*p[2]+matrix[0][3],
      matrix[1][0]*p[0]+matrix[1][1]*p[1]+matrix[1][2]*p[2]+matrix[1][3],
      matrix[3][0]*p[0]+matrix[3][1]*p[1]+matrix[3][2]*p[2]+matrix[3][3]
    ];
    if(Number.isFinite(clip[0])&&Number.isFinite(clip[1])&&clip[2]>Number.EPSILON&&pointInPolygon([clip[0]/clip[2],clip[1]/clip[2]],polygon)) preview.push(p);
  }
  updatePreviewTrace(preview,treeId);
  return preview.length;
}
function beginLasso(){
  const treeId=requireOneTree();if(treeId==null)return;
  setTrimMode("LASSO","套索模式已开启：拖动鼠标圈选；松开后恢复旋转。Esc 取消。");
  const host=document.getElementById("plot3d"),rect=host.getBoundingClientRect();
  const svg=document.createElementNS("http://www.w3.org/2000/svg","svg");lassoOverlay=svg;
  Object.assign(svg.style,{position:"fixed",left:`${rect.left}px`,top:`${rect.top}px`,width:`${rect.width}px`,height:`${rect.height}px`,zIndex:20,cursor:"crosshair",touchAction:"none",userSelect:"none"});
  const line=document.createElementNS(svg.namespaceURI,"polyline");line.setAttribute("fill","rgba(255,0,0,.12)");line.setAttribute("stroke","#ff0000");line.setAttribute("stroke-width","2");svg.appendChild(line);document.body.appendChild(svg);
  let points=[],drawing=false;
  svg.onpointerdown=e=>{drawing=true;points=[[e.offsetX,e.offsetY]];svg.setPointerCapture(e.pointerId);};
  svg.onpointermove=e=>{if(!drawing)return;const last=points[points.length-1];if(Math.hypot(e.offsetX-last[0],e.offsetY-last[1])>=3){points.push([e.offsetX,e.offsetY]);line.setAttribute("points",points.map(p=>p.join(",")).join(" "));}};
  const abortDraw=message=>{drawing=false;removeLassoOverlay();setTrimMode("IDLE",message||"已取消套索，3D 旋转已恢复。");};
  svg.onpointercancel=()=>abortDraw("套索已取消，3D 旋转已恢复。");
  svg.onpointerup=async e=>{if(!drawing)return;drawing=false;svg.releasePointerCapture(e.pointerId);
    if(points.length<3){removeLassoOverlay();setTrimMode("IDLE","套索至少需要三个顶点。");return;}
    const polygon=points.map(p=>[2*p[0]/rect.width-1,1-2*p[1]/rect.height]);
    try{const matrix=currentViewProjection();removeLassoOverlay();const displayCount=displayLassoPreview(treeId,matrix,polygon);setTrimMode("PREVIEW_LOADING",`预览中… 显示样本命中 ${displayCount.toLocaleString()} 个点；正在精确计算完整点成员，3D 旋转已恢复...`);
      const requestId=++previewRequestId,preview=await post("/api/trim/lasso-preview",{tree_id:treeId,view_projection:matrix,polygon_ndc:polygon});if(requestId!==previewRequestId||Number(activeTreeId)!==Number(treeId)){await post("/api/trim/cancel",{preview_id:preview.preview_id});return;}await showPreview(preview);}
    catch(err){removeLassoOverlay();updatePreviewTrace([]);setTrimMode("IDLE","");showFailure(err,"");}
  };
}
document.getElementById("btn-trim-component").onclick=()=>{if(requireOneTree()!=null)setTrimMode("COMPONENT","请点击需要从当前单木中剔除的误分点团。");};
document.getElementById("btn-trim-lasso").onclick=beginLasso;
document.getElementById("btn-trim-confirm").onclick=async()=>{
  if(!activePreview)return;
  setEditingBusy(true);
  try{document.getElementById("editmsg").textContent="正在写入完整分辨率点成员...";
    const snap=await post("/api/trim/commit",{preview_id:activePreview.preview_id});activePreview=null;viewerState.activePreviewType=null;updatePreviewTrace([]);selected.clear();document.getElementById("trim-preview").style.display="none";setTrimMode("IDLE","已剔除所选点，可使用撤销精确恢复。");renderTrees(snap);await loadEditorData();}
  catch(e){showFailure(e);}finally{setEditingBusy(false);}
};
async function cancelTrim(message="已取消，单木点成员未改变。"){
  previewRequestId++;
  if(activePreview){try{await post("/api/trim/cancel",{preview_id:activePreview.preview_id});}catch(e){}}
  activePreview=null;viewerState.activePreviewType=null;removeLassoOverlay();document.getElementById("trim-preview").style.display="none";setTrimMode("IDLE",message);
  updatePreviewTrace([]);
}
document.getElementById("btn-trim-cancel").onclick=cancelTrim;
document.getElementById("toggle-labels").onchange=e=>{showLabels=Boolean(e.target.checked);scheduleHtmlLabels();};
window.addEventListener?.("resize",scheduleHtmlLabels);
function stepTree(delta){
  if(!treeMap.size||trimMode!=="IDLE"||activePreview)return;
  const ids=[...treeMap.keys()].sort((a,b)=>a-b),index=activeTreeId==null?(delta>0?-1:0):ids.indexOf(Number(activeTreeId));
  const next=Math.max(0,Math.min(ids.length-1,index+delta));
  selectTree(ids[next],false,true);
}
document.getElementById("btn-prev").onclick=()=>stepTree(-1);
document.getElementById("btn-next").onclick=()=>stepTree(1);
document.addEventListener?.("keydown",e=>{
  if(e.key==="Escape"&&(trimMode!=="IDLE"||activePreview)){e.preventDefault();cancelTrim();return;}
  if((e.key==="ArrowLeft"||e.key==="ArrowRight")&&trimMode==="IDLE"&&!activePreview&&!/INPUT|TEXTAREA|SELECT/.test(e.target?.tagName||"")){
    e.preventDefault();stepTree(e.key==="ArrowLeft"?-1:1);
  }
});
document.getElementById("btn-delete").onclick=async()=>{
  setEditingBusy(true);try{if(activePreview)await cancelTrim();await post("/api/select",{tree_ids:[...selected]});const s=await post("/api/delete",{});selected.clear();renderTrees(s);
      document.getElementById("editmsg").textContent="已删除所选单木";await loadEditorData();}
  catch(e){showFailure(e);}finally{setEditingBusy(false);}
};
document.getElementById("btn-merge").onclick=async()=>{
  setEditingBusy(true);try{if(activePreview)await cancelTrim();await post("/api/select",{tree_ids:[...selected]});const s=await post("/api/merge",{});selected.clear();renderTrees(s);
      document.getElementById("editmsg").textContent="已合并所选单木";await loadEditorData();}
  catch(e){showFailure(e);}finally{setEditingBusy(false);}
};
document.getElementById("btn-undo").onclick=async()=>{
  setEditingBusy(true);try{if(activePreview)await cancelTrim();const s=await post("/api/undo",{});selected.clear();renderTrees(s);
      document.getElementById("editmsg").textContent="已撤销上一步";await loadEditorData();}
  catch(e){showFailure(e);}finally{setEditingBusy(false);}
};
document.getElementById("btn-confirm").onclick=async()=>{
  setEditingBusy(true);
  if(activeTreeId!=null) reviewed.add(Number(activeTreeId));updateReviewStatus();
  const pending=Math.max(0,treeMap.size-reviewed.size);
  document.getElementById("editmsg").textContent=pending?`仍有 ${pending} 棵未逐棵查看，将按现有批量确认规则完成复核。`:"";
  document.getElementById("progress").textContent="正在应用人工修改";
  try{await post("/api/confirm_trees",{});}catch(e){showFailure(e);setEditingBusy(false);return;}
  if(refreshTimer)clearInterval(refreshTimer);refreshTimer=setInterval(refreshTrees,700);refreshTrees();
};
loadEditorData().then(()=>{refreshTrees();refreshTimer=setInterval(refreshTrees,1500);}).catch(showFailure);
"""
    return _page_shell("STEP 2 Plotly 旧版调试页", "STEP 2 Plotly 旧版调试页", body, script)


def step2_html() -> str:
    """Return the production Step-5 tree editor using local Giro3D / Three.js."""

    body = """
<main class="tree-review-main">
  <div class="panel plot-panel">
    <div id="tree-review-stage">
      <div id="plot3d"></div>
      <div id="tree-label-overlay" aria-label="树编号标签"></div>
      <div id="viewer-toolbar" aria-label="三维视图控制">
        <button id="btn-review-back" type="button" title="退出复核视图，不撤销已提交修改；后台任务继续">← 返回 / 退出复核</button>
        <button id="btn-view-reset" type="button">重置</button>
        <button id="btn-view-top" type="button">俯视</button>
        <button id="btn-view-side" type="button">侧视</button>
      </div>
      <div id="viewer-renderer-status">正在初始化 Giro3D / Three.js…</div>
      <section id="viewer-init-failure" role="alert" style="display:none;position:absolute;
        z-index:20;inset:12px;padding:18px;overflow:auto;background:#fff;color:#182635;
        border:2px solid #b42318;border-radius:6px">
        <h3>3D Viewer could not start on this device.</h3>
        <p>此设备无法启动 3D 视图。项目和后台服务仍然可用；请复制诊断信息。</p>
        <pre id="viewer-init-summary" style="white-space:pre-wrap"></pre>
        <button id="btn-viewer-retry" type="button">Retry Viewer / 重试视图</button>
        <button id="btn-viewer-copy" type="button">Copy diagnostic information / 复制诊断</button>
        <textarea id="viewer-init-details" readonly aria-label="3D Viewer diagnostic information"
          style="display:block;width:100%;height:200px;margin-top:12px;box-sizing:border-box">
        </textarea>
      </section>
    </div>
  </div>
  <div class="side">
    <div class="panel">
      <h3>第二步：人工审核单木</h3>
      <p style="font-size:12px;color:#5f6b7a">单击单选；按住 Ctrl 或自定义按键再点击可多选。先选操作模式，再执行；剔除先红色预览，确认后才修改。</p>
      <label style="display:flex;gap:8px;align-items:center;font-size:12px">多选按键（按住）
        <select id="multi-select-key" aria-label="自定义多选按键"></select>
        <span>Ctrl 始终可用</span>
      </label>
      <div id="review-status" style="font-size:13px;font-weight:600;margin-bottom:8px">正在加载单木复核数据...</div>
      <div id="current-tree" style="font-size:12px;color:#5f6b7a;margin-bottom:8px">当前：未选择</div>
      <div style="display:flex;gap:8px;flex-wrap:wrap">
        <button id="btn-prev" class="gray">上一棵</button>
        <button id="btn-next" class="gray">下一棵</button>
        <button id="btn-delete" class="red">删除单木</button>
        <button id="btn-merge" type="button" aria-pressed="false">合并选中</button>
        <button id="btn-split" type="button" aria-pressed="false">分开粘连树</button>
        <button id="btn-trim-component" type="button" aria-pressed="false">剔除误分团块</button>
        <button id="btn-trim-lasso">套索剔除</button>
        <button id="btn-undo" class="gray">撤销</button>
        <button id="btn-dbh-review" class="gray" type="button">查看原始树干并复核最低点以上 1.30m 胸径</button>
      </div>
      <div id="edit-operation-hint" style="margin-top:10px;font-size:13px;line-height:1.6" role="status"></div>
      <div style="display:flex;gap:8px;margin-top:8px">
        <button id="btn-edit-execute" class="green" type="button" disabled>请选择操作</button>
        <button id="btn-selection-clear" class="gray" type="button">清空选择</button>
      </div>
      <div style="margin-top:8px;padding:8px;border-left:3px solid #2878c7;background:#eef6ff;color:#31475d;font-size:12px;line-height:1.55">
        当前彩色图显示的是处理后的单木成员，不代表胸径取样点。树高与胸径均回到该树未去地面的原始 LAS 成员计算；胸径断面固定为“该树原始最低点 + 1.30 m”。
      </div>
      <div id="selection-state" style="font-size:12px;color:#5f6b7a;margin-top:7px">未选择单木</div>
      <div id="lasso-mode-indicator" style="display:none;margin-top:8px;padding:8px;border-radius:4px;background:#fff3cd;color:#6b4f00;font-weight:600;white-space:pre-line"></div>
      <div id="editmsg" class="msg" style="margin-top:6px"></div>
      <details id="error-details" style="display:none;font-size:11px;color:#5f6b7a;margin-top:4px"><summary>查看详情</summary><pre id="error-detail-text" style="white-space:pre-wrap"></pre></details>
      <div id="trim-preview" style="display:none;margin-top:8px;padding:8px;border:2px solid #d64545;border-radius:4px;background:#fff5f5">
        <div id="trim-summary" style="font-size:13px;margin-bottom:7px"></div>
        <button id="btn-trim-confirm" class="red">确认剔除</button>
        <button id="btn-trim-cancel" class="gray">取消</button>
      </div>
      <label style="display:flex;align-items:center;gap:6px;margin-top:10px;font-size:12px"><input id="toggle-labels" type="checkbox" checked>显示树编号</label>
      <div style="margin-top:10px"><button id="btn-confirm" class="green">完成复核并生成 Excel</button></div>
      <div id="progress" style="font-size:12px;margin-top:8px"></div>
    </div>
    <div class="panel">
      <h3>单木列表 <span id="treecount" style="font-size:12px;color:#5f6b7a"></span></h3>
      <div id="treelist" style="max-height:40vh;overflow:auto">
        <p class="hint">此列表高度为分割阶段参考值；原始树木最高点减最低点的正式树高请查看“查看数据”或最终报告。</p>
        <table id="treetable"><thead><tr><th>ID</th><th>点数</th><th>分割参考高(m)</th><th>编辑状态</th></tr></thead><tbody></tbody></table>
      </div>
    </div>
  </div>
</main>
"""
    # Install this before external scripts: even a missing diagnostic module must
    # leave readable recovery controls rather than an unhandled black canvas.
    body += """
<script>
(function () {
  var failures = window.__treeViewerResourceFailures = [];
  window.addEventListener('error', function (event) {
    var target = event.target;
    if (!target || target.tagName !== 'SCRIPT' || !target.src) return;
    var url = new URL(target.src, window.location.href).pathname;
    var allowed = ['/viewer-runtime.js', '/giro3d.bundle.js',
      '/tree-review-3d.js', '/tree-review-app.js'];
    if (allowed.indexOf(url) < 0) return;
    failures.push({url: url, message: event.message || 'Script resource failed to load',
      stack: event.error && event.error.stack || ''});
    if (window.TreeLineViewerRuntime) return;
    var panel = document.getElementById('viewer-init-failure');
    if (!panel) return;
    panel.style.display = 'block';
    document.getElementById('viewer-init-summary').textContent =
      'VIEWER_RESOURCE_MISSING: ' + url + '\\nWebGL/WebGL2: NOT_TESTED' +
      '\\nWebView2: UNKNOWN; diagnostic module did not load.';
    var details = document.getElementById('viewer-init-details');
    details.value = JSON.stringify({failure_code: 'VIEWER_RESOURCE_MISSING',
      userAgent: navigator.userAgent, resources: failures}, null, 2);
    document.getElementById('btn-viewer-retry').onclick = function () {
      window.location.reload();
    };
    document.getElementById('btn-viewer-copy').onclick = function () {
      details.focus(); details.select();
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(details.value).catch(function () {});
      }
    };
  }, true);
})();
</script>
"""
    return _page_shell(
        "STEP 2 人工审核单木",
        "STEP 2 人工审核单木 · Giro3D / Three.js",
        body,
        "window.TreeLineTreeReviewApp.start();",
        plotly=False,
        external_scripts=(
            "/viewer-runtime.js",
            "/giro3d.bundle.js",
            "/tree-review-3d.js",
            "/tree-review-app.js",
        ),
    )


# ---------------------------------------------------------------------------
# STEP 3 — final result + Excel
# ---------------------------------------------------------------------------


def step3_html() -> str:
    body = """
<main>
  <div class="panel" style="flex:1 1 100%">
    <h3>第三步：最终结果</h3>
    <div id="kpis" class="kpis"></div>
    <div style="margin-top:10px;display:flex;gap:10px">
      <a id="excel-link" href="/download_excel"><button class="green">下载 Prediction_Input.xlsx</button></a>
      <button id="btn-restart" class="gray">重新开始</button>
    </div>
    <div id="metadata" style="font-size:12px;color:#5f6b7a;margin-top:8px"></div>
  </div>
  <div class="panel plot-panel">
    <div id="plot3d" style="height:100%"></div>
  </div>
  <div class="panel" style="flex:1 1 100%">
    <h3>最终单木表</h3>
    <div style="max-height:50vh;overflow:auto">
      <table id="finaltable">
        <thead><tr>
          <th>Tree ID</th><th>高度(m)</th><th>冠幅(m)</th><th>冠幅面积(m²)</th>
          <th>胸径(cm)</th><th>最小净空(m)</th><th title="三维最近点对的水平分量">水平净空分量(m)</th><th title="三维最近点对的垂直分量">垂直净空分量(m)</th><th>危险点数</th><th>风险</th>
          <th>质量</th><th>人工编辑</th>
        </tr></thead>
        <tbody></tbody>
      </table>
    </div>
  </div>
</main>
"""
    script = """
function setStatus(t){document.getElementById("status").textContent=t;}
async function refreshState(){
  try{
    const s=await (await fetch("/state")).json();
    setStatus(s.step_label + (s.current_message?" — "+s.current_message:""));
    if(s.error){setStatus("失败："+s.error);}
    if(s.final_summary) renderSummary(s.final_summary);
  }catch(e){}
}
function renderSummary(sum){
  const k=document.getElementById("kpis");k.innerHTML="";
  const items=[["源文件","见 Run_Metadata"],["最终单木数",sum.FINAL_TREE_COUNT],
    ["危险单木数",sum.DANGEROUS_TREE_COUNT],["危险点数",sum.DANGER_POINT_COUNT],
    ["最小净空(m)",sum.GLOBAL_MINIMUM_CLEARANCE_M!=null?sum.GLOBAL_MINIMUM_CLEARANCE_M.toFixed(3):"—"],
    ["导线数",sum.WIRE_STRAND_COUNT],["档距(m)",sum.SPAN_LENGTH_M!=null?sum.SPAN_LENGTH_M.toFixed(2):"—"]];
  items.forEach(([name,value])=>{
    const d=document.createElement("div");d.className="kpi";
    d.innerHTML=`<div class="v">${value}</div><div class="k">${name}</div>`;
    k.appendChild(d);
  });
}
async function loadFinal(){
  const data=await (await fetch("/api/final")).json();
  // table
  const tbody=document.querySelector("#finaltable tbody");tbody.innerHTML="";
  data.trees.forEach(t=>{
    const tr=document.createElement("tr");
    const risk=t.current_risk;
    tr.innerHTML=`<td>${t.persistent_tree_id}</td><td>${
      t.height_m!=null?t.height_m.toFixed(2):"—"}</td><td>${
      t.crown_width_m!=null?t.crown_width_m.toFixed(2):"—"}</td><td>${
      t.crown_area_m2!=null?t.crown_area_m2.toFixed(2):"—"}</td><td>${
      t.diameter_cm!=null?t.diameter_cm.toFixed(2):"No"}</td><td>${
      t.minimum_clearance_m!=null?t.minimum_clearance_m.toFixed(3):"—"}</td><td>${
      t.horizontal_clearance_m!=null?t.horizontal_clearance_m.toFixed(3):"—"}</td><td>${
      t.vertical_clearance_m!=null?t.vertical_clearance_m.toFixed(3):"—"}</td><td>${
      t.danger_point_count}</td><td>${risk}</td><td>${t.quality_state}</td><td>${
      t.human_edited?"是":"否"}</td>`;
    if(risk==="DANGEROUS") tr.style.background="#ffe9e9";
    tbody.appendChild(tr);
  });
  // 3d — same display payload contract as Step 2: `background`/`wire3d`/`towers`
  // are optional dicts `{xyz:[...]}` while `trees`/`danger` are plain point
  // arrays; guard each layer so an empty layer renders as nothing instead of
  // crashing (or silently skipping) the plot.
  const disp=data.display||{};
  const traces=[];
  if(disp.background && Array.isArray(disp.background.xyz) && disp.background.xyz.length){
    const c=disp.background.rgb.map(v=>`rgb(${v[0]},${v[1]},${v[2]})`);
    traces.push({x:disp.background.xyz.map(p=>p[0]),y:disp.background.xyz.map(p=>p[1]),
      z:disp.background.xyz.map(p=>p[2]),type:"scatter3d",mode:"markers",name:"背景",
      marker:{size:1,color:c,opacity:0.4}});
  }
  if(Array.isArray(disp.trees) && disp.trees.length){
    traces.push({x:disp.trees.map(p=>p[0]),y:disp.trees.map(p=>p[1]),
      z:disp.trees.map(p=>p[2]),type:"scatter3d",mode:"markers",name:"最终单木",
      marker:{size:1.6,color:disp.tree_color,opacity:0.95}});
  }
  if(Array.isArray(disp.danger) && disp.danger.length){
    traces.push({x:disp.danger.map(p=>p[0]),y:disp.danger.map(p=>p[1]),
      z:disp.danger.map(p=>p[2]),type:"scatter3d",mode:"markers",name:"危险点",
      marker:{size:2.4,color:"#ff0000",opacity:1}});
  }
  if(disp.wire3d && Array.isArray(disp.wire3d.xyz) && disp.wire3d.xyz.length){
    traces.push({x:disp.wire3d.xyz.map(p=>p[0]),y:disp.wire3d.xyz.map(p=>p[1]),
      z:disp.wire3d.xyz.map(p=>p[2]),type:"scatter3d",mode:"lines",name:"导线",
      line:{color:"#ff1744",width:3}});
  }
  if(disp.towers && Array.isArray(disp.towers.xyz) && disp.towers.xyz.length){
    traces.push({x:disp.towers.xyz.map(p=>p[0]),y:disp.towers.xyz.map(p=>p[1]),
      z:disp.towers.xyz.map(p=>p[2]),type:"scatter3d",mode:"markers",name:"杆塔",
      marker:{size:7,color:"#ff9800",symbol:"diamond"}});
  }
  Plotly.newPlot("plot3d", traces,
    {title:{text:"最终结果（危险点=红色）",font:{size:14}},
     scene:{aspectmode:"data"}, margin:{l:0,r:0,t:36,b:0}},
    {responsive:true, displaylogo:false});
  const meta=data.metadata||{};
  document.getElementById("metadata").textContent=
    `后端=${meta.tree_backend} · 导线剔除=${meta.wire_exclusion_policy} · 杆塔剔除=${meta.tower_exclusion_policy}`;
}
document.getElementById("btn-restart").onclick=async()=>{
  await fetch("/restart",{method:"POST"});window.location.href="/step1";
};
refreshState();
loadFinal();
setInterval(refreshState, 2000);
"""
    return _page_shell("STEP 3 最终结果", "STEP 3 最终结果", body, script)


__all__ = ["step1_html", "step2_html", "step3_html"]
