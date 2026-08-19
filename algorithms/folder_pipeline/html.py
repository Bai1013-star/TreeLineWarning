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
    <div id="plot3d" style="height:100%"></div>
  </div>
  <div class="side">
    <div class="panel">
      <h3>第二步：人工审核单木</h3>
      <p style="font-size:12px;color:#5f6b7a">
        点击选中一棵树（可多选）。可删除、合并或撤销。确认后计算最终净空与 Excel。
      </p>
      <div style="display:flex;gap:8px;flex-wrap:wrap">
        <button id="btn-delete" class="red">删除单木</button>
        <button id="btn-merge">合并选中</button>
        <button id="btn-undo" class="gray">撤销</button>
      </div>
      <div id="editmsg" class="msg" style="margin-top:6px"></div>
      <div style="margin-top:10px">
        <button id="btn-confirm" class="green">确认单木并继续</button>
      </div>
      <div id="progress" style="font-size:12px;margin-top:8px"></div>
    </div>
    <div class="panel">
      <h3>单木列表 <span id="treecount" style="font-size:12px;color:#5f6b7a"></span></h3>
      <div style="max-height:40vh;overflow:auto">
        <table id="treetable">
          <thead><tr><th>ID</th><th>点数</th><th>高度(m)</th><th>合并</th></tr></thead>
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
let refreshTimer = null;

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
  if(!r.ok){const t=await r.json().catch(()=>({}));throw new Error(t.error||("请求失败 "+path));}
  return r.json();
}
function renderTrees(snap){
  treeMap = new Map();
  const tbody=document.querySelector("#treetable tbody");
  tbody.innerHTML="";
  snap.trees.forEach(t=>{
    treeMap.set(t.tree_id, t);
    const tr=document.createElement("tr");
    tr.dataset.tid=t.tree_id;
    if(selected.has(t.tree_id)) tr.className="sel";
    tr.innerHTML=`<td>${t.tree_id}</td><td>${t.point_count}</td><td>${
      t.apex_height_m!=null?t.apex_height_m.toFixed(1):"—"}</td><td>${t.merged?"合并":""}</td>`;
    tr.onclick=()=>{
      if(selected.has(t.tree_id)) selected.delete(t.tree_id); else selected.add(t.tree_id);
      renderTrees(snap);
    };
    tbody.appendChild(tr);
  });
  document.getElementById("treecount").textContent = `共 ${snap.current_tree_count} 棵`;
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
  const bg=data.background, wire=data.wire, tower=data.tower, trees=data.trees, wires3d=data.wires3d;
  const traces=[];
  if(bg && bg.xyz.length){
    const c=bg.rgb.map(v=>`rgb(${v[0]},${v[1]},${v[2]})`);
    traces.push({x:bg.xyz.map(p=>p[0]),y:bg.xyz.map(p=>p[1]),z:bg.xyz.map(p=>p[2]),
      type:"scatter3d",mode:"markers",name:"背景",
      marker:{size:1,color:c,opacity:0.4}});
  }
  if(wire && wire.xyz.length){
    traces.push({x:wire.xyz.map(p=>p[0]),y:wire.xyz.map(p=>p[1]),z:wire.xyz.map(p=>p[2]),
      type:"scatter3d",mode:"markers",name:"导线",
      marker:{size:2,color:"#ff1744",opacity:1}});
  }
  if(wires3d && wires3d.xyz.length){
    traces.push({x:wires3d.xyz.map(p=>p[0]),y:wires3d.xyz.map(p=>p[1]),z:wires3d.xyz.map(p=>p[2]),
      type:"scatter3d",mode:"lines",name:"导线曲线",
      line:{color:"#ff1744",width:3}});
  }
  if(tower && tower.xyz.length){
    traces.push({x:tower.xyz.map(p=>p[0]),y:tower.xyz.map(p=>p[1]),z:tower.xyz.map(p=>p[2]),
      type:"scatter3d",mode:"markers+text",name:"杆塔",
      text:tower.labels||[],
      marker:{size:7,color:"#ff9800",symbol:"diamond"}});
  }
  if(trees && trees.xyz.length){
    traces.push({x:trees.xyz.map(p=>p[0]),y:trees.xyz.map(p=>p[1]),z:trees.xyz.map(p=>p[2]),
      type:"scatter3d",mode:"markers",name:"单木",
      marker:{size:1.6,color:trees.color,opacity:0.95}});
  }
  Plotly.newPlot("plot3d", traces,
    {title:{text:"单木审核（不同颜色=不同树）",font:{size:14}},
     scene:{aspectmode:"data"}, margin:{l:0,r:0,t:36,b:0}},
    {responsive:true, displaylogo:false});
  const gd=document.getElementById("plot3d");
  gd.on("plotly_click", data=>{
    if(!data.points||!data.points.length) return;
    const pt=data.points[0];
    if(pt.trace.name==="单木"){
      const tid=trees.tree_id_by_index[pt.pointNumber];
      if(tid!=null){
        if(selected.has(tid)) selected.delete(tid); else selected.add(tid);
        refreshTrees();
      }
    }
  });
}
document.getElementById("btn-delete").onclick=async()=>{
  try{const s=await post("/api/delete",{});renderTrees(s);
      document.getElementById("editmsg").textContent="已删除所选单木";await loadEditorData();}
  catch(e){document.getElementById("editmsg").textContent="失败："+e.message;}
};
document.getElementById("btn-merge").onclick=async()=>{
  try{const s=await post("/api/merge",{});renderTrees(s);
      document.getElementById("editmsg").textContent="已合并所选单木";await loadEditorData();}
  catch(e){document.getElementById("editmsg").textContent="失败："+e.message;}
};
document.getElementById("btn-undo").onclick=async()=>{
  try{const s=await post("/api/undo",{});renderTrees(s);
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
  // 3d
  const traces=[];
  if(data.display.background.length){
    const c=data.display.background.map(v=>`rgb(${v[0]},${v[1]},${v[2]})`);
    traces.push({x:data.display.background.map(p=>p[0]),y:data.display.background.map(p=>p[1]),
      z:data.display.background.map(p=>p[2]),type:"scatter3d",mode:"markers",name:"背景",
      marker:{size:1,color:c,opacity:0.4}});
  }
  if(data.display.trees.length){
    traces.push({x:data.display.trees.map(p=>p[0]),y:data.display.trees.map(p=>p[1]),
      z:data.display.trees.map(p=>p[2]),type:"scatter3d",mode:"markers",name:"最终单木",
      marker:{size:1.6,color:data.display.tree_color,opacity:0.95}});
  }
  if(data.display.danger.length){
    traces.push({x:data.display.danger.map(p=>p[0]),y:data.display.danger.map(p=>p[1]),
      z:data.display.danger.map(p=>p[2]),type:"scatter3d",mode:"markers",name:"危险点",
      marker:{size:2.4,color:"#ff0000",opacity:1}});
  }
  if(data.display.wire3d.length){
    traces.push({x:data.display.wire3d.map(p=>p[0]),y:data.display.wire3d.map(p=>p[1]),
      z:data.display.wire3d.map(p=>p[2]),type:"scatter3d",mode:"lines",name:"导线",
      line:{color:"#ff1744",width:3}});
  }
  if(data.display.towers.length){
    traces.push({x:data.display.towers.map(p=>p[0]),y:data.display.towers.map(p=>p[1]),
      z:data.display.towers.map(p=>p[2]),type:"scatter3d",mode:"markers",name:"杆塔",
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
