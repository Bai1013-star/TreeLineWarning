# ruff: noqa: E501  (long HTML/JS template content)
"""Three-step browser pages for the Folder pipeline.

The pages are served by ONE local server and share ONE workflow state object
(``/state``), so they are a single run/session — not three disconnected static
programs.  Plotly is used for the point rendering (proven pattern of the
manual-span selector); every colour is assigned server-side from a stable
deterministic mapping so a page refresh never changes a tree's colour.
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
  """


def _page_shell(title: str, step: str, body: str, script: str) -> str:
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
<script src="/plotly.js"></script>
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


def step2_html() -> str:
    body = """
<main>
  <div class="panel plot-panel">
    <div style="position:relative;height:100%">
      <div id="plot3d" style="height:100%"></div>
      <div id="tree-label-overlay" aria-label="树编号标签"></div>
    </div>
  </div>
  <div class="side">
    <div class="panel">
      <h3>第二步：人工审核单木</h3>
      <p style="font-size:12px;color:#5f6b7a">
        点击选中单木。局部剔除会先红色预览，确认后才修改完整点成员。
      </p>
      <div style="display:flex;gap:8px;flex-wrap:wrap">
        <button id="btn-delete" class="red">删除单木</button>
        <button id="btn-merge">合并选中</button>
        <button id="btn-trim-component">剔除误分团块</button>
        <button id="btn-trim-lasso">套索剔除</button>
        <button id="btn-undo" class="gray">撤销</button>
      </div>
      <div id="editmsg" class="msg" style="margin-top:6px"></div>
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
        <button id="btn-confirm" class="green">确认单木并继续</button>
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

function setStatus(t){document.getElementById("status").textContent=t;}
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
  if(!r.ok){const t=await r.json().catch(()=>({}));throw new Error(t.error?.message||t.error||("请求失败 "+path));}
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
  if(scrollToActive && activeTreeId!=null) setTimeout(()=>scrollToTreeRow(activeTreeId),0);
}
function treeIdFromPlotPoint(pt){
  const raw=pt.customdata!=null?pt.customdata:pt.data?.customdata?.[pt.pointNumber];
  const tid=Number(raw);
  return Number.isFinite(tid)?tid:null;
}
function selectTree(treeId, toggle=true, scroll=false){
  const tid=Number(treeId);
  if(!treeMap.has(tid)) return;
  activeTreeId=tid;
  if(toggle){
    if(selected.has(tid)) selected.delete(tid); else selected.add(tid);
  }else{
    selected=new Set([tid]);
  }
  renderTrees(currentSnapshot, scroll);
  renderEditorPlot();
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
  const traces=[];
  if(bg && Array.isArray(bg.xyz) && bg.xyz.length){
    const c=(Array.isArray(bg.rgb)?bg.rgb:[]).map(v=>`rgb(${v[0]},${v[1]},${v[2]})`);
    traces.push({x:bg.xyz.map(p=>p[0]),y:bg.xyz.map(p=>p[1]),z:bg.xyz.map(p=>p[2]),
      type:"scatter3d",mode:"markers",name:"背景",marker:{size:1,color:c,opacity:0.4}});
  }
  if(wire && Array.isArray(wire.xyz) && wire.xyz.length){
    traces.push({x:wire.xyz.map(p=>p[0]),y:wire.xyz.map(p=>p[1]),z:wire.xyz.map(p=>p[2]),
      type:"scatter3d",mode:"markers",name:"导线",marker:{size:2,color:"#ff1744",opacity:1}});
  }
  if(wires3d && Array.isArray(wires3d.xyz) && wires3d.xyz.length){
    traces.push({x:wires3d.xyz.map(p=>p[0]),y:wires3d.xyz.map(p=>p[1]),z:wires3d.xyz.map(p=>p[2]),
      type:"scatter3d",mode:"lines",name:"导线曲线",line:{color:"#ff1744",width:3}});
  }
  if(tower && Array.isArray(tower.xyz) && tower.xyz.length){
    traces.push({x:tower.xyz.map(p=>p[0]),y:tower.xyz.map(p=>p[1]),z:tower.xyz.map(p=>p[2]),
      type:"scatter3d",mode:"markers+text",name:"杆塔",text:tower.labels||[],
      marker:{size:7,color:"#ff9800",symbol:"diamond"}});
  }
  if(Array.isArray(trees3d.xyz) && trees3d.xyz.length){
    const highlightRows=selectedPointRows();
    const treeTrace={x:trees3d.xyz.map(p=>p[0]),y:trees3d.xyz.map(p=>p[1]),z:trees3d.xyz.map(p=>p[2]),
      type:"scatter3d",mode:"markers",name:"单木",customdata:trees3d.tree_id_by_index,
      marker:{size:1.6,color:trees3d.color,opacity:0.95}};
    if(highlightRows.length){
      treeTrace.selectedpoints=highlightRows;
      treeTrace.selected={marker:{size:3.8,opacity:1}};
      treeTrace.unselected={marker:{opacity:0.18}};
    }
    traces.push(treeTrace);
  }
  if(activePreview && Array.isArray(activePreview.preview_xyz)){
    const p=activePreview.preview_xyz;
    traces.push({x:p.map(v=>v[0]),y:p.map(v=>v[1]),z:p.map(v=>v[2]),type:"scatter3d",mode:"markers",
      name:"待剔除",marker:{size:3.2,color:"#ff0000",opacity:1}});
  }
  if(!traces.length){
    document.getElementById("plot3d").textContent="暂无可审核的 Tree display sample；完成 Batch 后刷新。";
    return;
  }
  const gd=document.getElementById("plot3d");
  let renderPromise=null;
  if(plotClickBound && typeof Plotly.react==="function"){
    renderPromise=Plotly.react(gd,traces,plotLayout(),{responsive:true,displaylogo:false});
  }else{
    renderPromise=Plotly.newPlot(gd,traces,plotLayout(),{responsive:true,displaylogo:false});
    if(!plotClickBound) bindPlotClicks();
  }
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
    if(s.error){document.getElementById("editmsg").textContent="失败："+s.error;}
    if(s.step==="DONE"){clearInterval(refreshTimer);window.location.href="/step3";}
    if(s.tree_snapshot) renderTrees(s.tree_snapshot);
  }catch(e){}
}
async function loadEditorData(){
  const data=await (await fetch("/api/editor_data")).json();
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
}
function requireOneTree(){
  if(selected.size!==1){document.getElementById("editmsg").textContent="请先选择一棵单木。";return null;}
  return [...selected][0];
}
function setTrimMode(mode,message){
  trimMode=mode;
  document.getElementById("btn-trim-component").style.outline=mode==="COMPONENT"?"3px solid #d64545":"";
  document.getElementById("btn-trim-lasso").style.outline=mode==="LASSO"?"3px solid #d64545":"";
  document.getElementById("editmsg").textContent=message||"";
}
async function showPreview(preview){
  activePreview=preview;setTrimMode("PREVIEW","");removeLassoOverlay();
  const percent=(preview.fraction*100).toFixed(1);
  const warning=preview.large_removal_warning?" 本次将移除该树较大比例的点，请确认选择是否正确。":"";
  document.getElementById("trim-summary").textContent=`待剔除 ${preview.point_count.toLocaleString()} 个点，占当前单木 ${percent}%。${warning}`;
  document.getElementById("trim-preview").style.display="block";
  await loadEditorData();
}
async function previewComponent(treeId,seed){
  try{document.getElementById("editmsg").textContent="正在分析当前单木的完整点成员...";
    await showPreview(await post("/api/trim/component-preview",{tree_id:treeId,seed_xyz:seed}));
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
function beginLasso(){
  const treeId=requireOneTree();if(treeId==null)return;
  setTrimMode("LASSO","请圈出需要剔除的区域，闭合后可旋转视图检查。");
  const host=document.getElementById("plot3d"),rect=host.getBoundingClientRect();
  const svg=document.createElementNS("http://www.w3.org/2000/svg","svg");lassoOverlay=svg;
  Object.assign(svg.style,{position:"fixed",left:`${rect.left}px`,top:`${rect.top}px`,width:`${rect.width}px`,height:`${rect.height}px`,zIndex:20,cursor:"crosshair"});
  const line=document.createElementNS(svg.namespaceURI,"polyline");line.setAttribute("fill","rgba(255,0,0,.12)");line.setAttribute("stroke","#ff0000");line.setAttribute("stroke-width","2");svg.appendChild(line);document.body.appendChild(svg);
  let points=[],drawing=false;
  svg.onpointerdown=e=>{drawing=true;points=[[e.offsetX,e.offsetY]];svg.setPointerCapture(e.pointerId);};
  svg.onpointermove=e=>{if(!drawing)return;const last=points[points.length-1];if(Math.hypot(e.offsetX-last[0],e.offsetY-last[1])>=3){points.push([e.offsetX,e.offsetY]);line.setAttribute("points",points.map(p=>p.join(",")).join(" "));}};
  svg.onpointerup=async e=>{if(!drawing)return;drawing=false;svg.releasePointerCapture(e.pointerId);
    if(points.length<3){removeLassoOverlay();setTrimMode("IDLE","套索至少需要三个顶点。");return;}
    const polygon=points.map(p=>[2*p[0]/rect.width-1,1-2*p[1]/rect.height]);
    try{const preview=await post("/api/trim/lasso-preview",{tree_id:treeId,view_projection:currentViewProjection(),polygon_ndc:polygon});await showPreview(preview);}
    catch(err){removeLassoOverlay();setTrimMode("IDLE",err.message);}
  };
}
document.getElementById("btn-trim-component").onclick=()=>{if(requireOneTree()!=null)setTrimMode("COMPONENT","请点击需要从当前单木中剔除的误分点团。");};
document.getElementById("btn-trim-lasso").onclick=beginLasso;
document.getElementById("btn-trim-confirm").onclick=async()=>{
  if(!activePreview)return;
  try{const snap=await post("/api/trim/commit",{preview_id:activePreview.preview_id});activePreview=null;selected.clear();document.getElementById("trim-preview").style.display="none";setTrimMode("IDLE","已剔除所选点，可使用撤销精确恢复。");renderTrees(snap);await loadEditorData();}
  catch(e){document.getElementById("editmsg").textContent="失败："+e.message;}
};
async function cancelTrim(){
  if(activePreview){try{await post("/api/trim/cancel",{preview_id:activePreview.preview_id});}catch(e){}}
  activePreview=null;removeLassoOverlay();document.getElementById("trim-preview").style.display="none";setTrimMode("IDLE","已取消，单木点成员未改变。");await loadEditorData();
}
document.getElementById("btn-trim-cancel").onclick=cancelTrim;
document.getElementById("toggle-labels").onchange=e=>{showLabels=Boolean(e.target.checked);renderEditorPlot();};
window.addEventListener?.("resize",scheduleHtmlLabels);
document.addEventListener?.("keydown",e=>{if(e.key==="Escape"&&(trimMode!=="IDLE"||activePreview)){e.preventDefault();cancelTrim();}});
document.getElementById("btn-delete").onclick=async()=>{
  try{await post("/api/select",{tree_ids:[...selected]});const s=await post("/api/delete",{});selected.clear();renderTrees(s);
      document.getElementById("editmsg").textContent="已删除所选单木";await loadEditorData();}
  catch(e){document.getElementById("editmsg").textContent="失败："+e.message;}
};
document.getElementById("btn-merge").onclick=async()=>{
  try{await post("/api/select",{tree_ids:[...selected]});const s=await post("/api/merge",{});selected.clear();renderTrees(s);
      document.getElementById("editmsg").textContent="已合并所选单木";await loadEditorData();}
  catch(e){document.getElementById("editmsg").textContent="失败："+e.message;}
};
document.getElementById("btn-undo").onclick=async()=>{
  try{const s=await post("/api/undo",{});selected.clear();renderTrees(s);
      document.getElementById("editmsg").textContent="已撤销上一步";await loadEditorData();}
  catch(e){document.getElementById("editmsg").textContent="失败："+e.message;}
};
document.getElementById("btn-confirm").onclick=async()=>{
  const btn=document.getElementById("btn-confirm");btn.disabled=true;
  document.getElementById("progress").textContent="正在计算最终净空/危险并生成 Excel...";
  try{await post("/api/confirm_trees",{});}catch(e){
    document.getElementById("editmsg").textContent="失败："+e.message;btn.disabled=false;return;}
  refreshTimer=setInterval(refreshTrees,1200);
};
loadEditorData();
refreshTimer=setInterval(refreshTrees,1500);
"""
    return _page_shell("STEP 2 人工审核单木", "STEP 2 人工审核单木", body, script)


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
          <th>DBH(cm)</th><th>最小净空(m)</th><th>危险点数</th><th>风险</th>
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
      t.diameter_cm!=null?t.diameter_cm:"—"}</td><td>${
      t.minimum_clearance_m!=null?t.minimum_clearance_m.toFixed(3):"—"}</td><td>${
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
