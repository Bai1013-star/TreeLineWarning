(() => {
  'use strict';

  const state = {
    project: null,
    capability: null,
    batch: null,
    uiState: 'NO_PROJECT',
    towerRows: [],
    towerViewer: null,
    localPreview: null,
    towerView: {scale: 1, offsetX: 0, offsetY: 0},
    treeReviewReady: false,
    finalized: false,
    testMode: false,
    viewerLoaded: false,
  };
  const browserAcceptance = new URLSearchParams(window.location.search).has('browser-test');
  const $ = id => document.getElementById(id);
  const esc = value => String(value ?? '—').replace(/[&<>"']/g, ch => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
  }[ch]));
  const api = (path, options = {}) => fetch(path, Object.assign({cache: 'no-store'}, options))
    .then(async response => {
      const payload = await response.json().catch(() => ({ok: false, error: {message: '本地服务返回了无法读取的数据'}}));
      if (!response.ok || payload.ok === false) throw new Error(payload.error?.message || '本地请求失败');
      return payload;
    });
  const jsonPost = (path, body = {}) => api(path, {
    method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body),
  });
  const nativeApi = () => window.pywebview?.api || null;
  const statusLabels = {
    NOT_STARTED: '未开始', RUNNING: '运行中', COMPLETE: '已完成', PARTIAL_FAILURE: '部分失败',
    FAILED: '失败', WAITING: '等待', PREPARING: '准备中', AUTO_ANALYZED: '分析完成',
    REVIEW_REQUIRED: '需要复核', USER_REVIEWED: '已复核', FINALIZED: '已定稿', AVAILABLE: '已有结果',
  };
  const statusLabel = value => statusLabels[value] || value || '未开始';
  const userError = (error, fallback) => {
    console.error(fallback, error);
    return /[\u3400-\u9fff]/.test(error?.message || '') ? error.message : fallback;
  };
  const setStatus = (message, kind = '') => {
    $('footerState').textContent = `应用状态：${message}`;
    $('footerState').className = kind ? `status-${kind}` : '';
  };

  async function getNativeBridge(requiredMethods) {
    let bridge = nativeApi();
    if (bridge && requiredMethods.every(name => typeof bridge[name] === 'function')) return bridge;
    if (browserAcceptance) return null;
    await new Promise(resolve => {
      let settled = false;
      const done = () => { if (!settled) { settled = true; resolve(); } };
      window.addEventListener('pywebviewready', done, {once: true});
      window.setTimeout(done, 1800);
    });
    bridge = nativeApi();
    if (!bridge || !requiredMethods.every(name => typeof bridge[name] === 'function')) {
      throw new Error(`DESKTOP_BRIDGE_UNAVAILABLE:${requiredMethods.join(',')}`);
    }
    return bridge;
  }

  function spanName(span) {
    if (span.display_name) return span.display_name;
    const towers = new Map((state.project?.tower_records || []).map(tower => [tower.tower_id, tower.display_name]));
    return `${towers.get(span.start_tower_id) || span.start_tower_id} → ${towers.get(span.end_tower_id) || span.end_tower_id}`;
  }

  function renderProject(project) {
    state.project = project;
    state.towerRows = (project.tower_records || []).map(tower => ({
      display_name: tower.display_name,
      x: tower.position?.[0] ?? tower.x,
      y: tower.position?.[1] ?? tower.y,
      z: tower.position?.[2] ?? tower.z,
      height_m: tower.attributes?.height_m ?? tower.height_m,
    }));
    $('projectName').textContent = project.is_desktop_home ? '未打开项目' : (project.project_name || '未命名项目');
    $('lineCount').textContent = project.is_desktop_home ? '—' : (project.lines ?? 0);
    $('towerCount').textContent = project.is_desktop_home ? '—' : (project.towers ?? 0);
    $('spanCount').textContent = project.is_desktop_home ? '—' : (project.spans ?? 0);
    $('analysisState').textContent = project.is_desktop_home ? '—' : statusLabel(project.analysis);
    $('riskThreshold').value = project.risk_policy?.warning_clearance_threshold_m ?? 0.5;
    $('riskPolicyNote').textContent = `当前阈值：${$('riskThreshold').value} m`;
    if (project.is_desktop_home) {
      $('projectTree').innerHTML = '<li class="muted">新建或打开项目后显示项目结构</li>';
      return;
    }
    const tree = [
      `<li class="node">项目 <span class="badge">${esc(statusLabel(project.analysis))}</span></li>`,
      `<li class="child">点云 · ${project.pointcloud === 'NOT_IMPORTED' ? '未导入' : '已导入'}</li>`,
      `<li class="node">线路 <span class="badge">${project.lines ?? 0}</span></li>`,
    ];
    (project.line_records || []).forEach(line => tree.push(`<li class="child">${esc(line.display_name)} · ${line.line_role === 'BRANCH' ? '支线路' : '主线路'}</li>`));
    tree.push(`<li class="node">杆塔 <span class="badge">${project.towers ?? 0}</span></li>`);
    (project.tower_records || []).forEach(tower => tree.push(`<li class="child">${esc(tower.display_name)}</li>`));
    tree.push(`<li class="node">逻辑档距 <span class="badge">${project.spans ?? 0}</span></li>`);
    (project.span_records || []).forEach(span => tree.push(`<li class="child span" data-span="${esc(span.span_id)}">${esc(spanName(span))}</li>`));
    $('projectTree').innerHTML = tree.join('');
    document.querySelectorAll('.span[data-span]').forEach(node => node.addEventListener('click', () => {
      const box = document.querySelector(`#spanChecks input[data-span="${CSS.escape(node.dataset.span)}"]`);
      if (box) { box.checked = true; box.scrollIntoView({block: 'nearest'}); }
    }));
  }

  function renderCapability(capability) {
    state.capability = capability;
    if (!capability?.available) {
      $('capabilityBox').className = 'notice warn';
      $('capabilityBox').textContent = '请先导入 LAS/LAZ 点云。';
      $('capabilityRows').innerHTML = '';
      return;
    }
    const coord = capability.coordinate_declaration || {};
    const confirmed = capability.unit_confirmed;
    const unitDisplay = {METRE: '米（m）', LOCAL_METRIC: '本地米制', FOOT: '英尺（ft）', REQUIRES_CONFIRMATION: '待确认'}[coord.unit_status] || coord.linear_unit || '未知';
    $('capabilityBox').className = `notice ${confirmed ? '' : 'warn'}`;
    $('capabilityBox').innerHTML = `${confirmed ? '点云信息已确认' : '需要确认点云单位'}<br>格式：${esc(capability.details?.format)} · 点数：${esc(capability.details?.point_count)} · 坐标参考系（CRS）：${esc(coord.horizontal_crs || '未知')} · 单位：${esc(unitDisplay)}`;
    const labels = {XYZ: '三维坐标', RGB: '颜色', INTENSITY: '强度', CLASSIFICATION: '分类', CRS: '坐标参考系（CRS）', LINEAR_UNIT: '线性单位'};
    $('capabilityRows').innerHTML = Object.entries(labels).map(([name, label]) => {
      const available = capability.capabilities?.[name] === 'AVAILABLE';
      return `<div class="metric"><span>${label}</span><b class="${available ? 'status-good' : 'status-warn'}">${available ? '可用' : '待确认'}</b></div>`;
    }).join('');
    $('unitSelect').value = coord.unit_status === 'FOOT' ? 'FOOT' : (coord.unit_status === 'LOCAL_METRIC' ? 'LOCAL_METRIC' : 'METRE');
  }

  function towerRow(row = {}, index) {
    return `<div class="tower-picked" data-row="${index}"><b>${esc(row.display_name || `T${String(index + 1).padStart(3, '0')}`)}</b><span>X ${Number(row.x).toFixed(2)} · Y ${Number(row.y).toFixed(2)}</span><small>Z ${Number(row.z).toFixed(2)} · 高度 ${Number(row.height_m).toFixed(2)} m · 真实局部点云</small></div>`;
  }
  function renderTowerRows() {
    $('towerRows').innerHTML = state.towerRows.map((row, index) => towerRow(row, index)).join('');
  }
  function ensureTowerRows() { renderTowerRows(); }

  function renderBatch(batch) {
    state.batch = batch;
    $('footerBatch').textContent = `分析任务：${statusLabel(batch.status)}`;
    const spans = batch.spans || [];
    $('spanChecks').innerHTML = spans.length ? spans.map(span => `<label class="check-item"><input type="checkbox" data-span="${esc(span.span_id)}" ${span.selected ? 'checked' : ''} ${span.status === 'FINALIZED' ? 'disabled' : ''}><span><b>${esc(spanName(span))}</b><small>${esc(span.span_id)} · ${esc(span.stage)}</small></span><span class="state ${span.status === 'FAILED' ? 'status-bad' : (span.status === 'FINALIZED' ? 'status-good' : '')}">${esc(statusLabel(span.status))}</span></label>`).join('') : '<div class="muted padded">暂无档距</div>';
    const failed = spans.filter(span => span.status === 'FAILED');
    const completed = spans.filter(span => ['AUTO_ANALYZED', 'REVIEW_REQUIRED', 'USER_REVIEWED', 'FINALIZED'].includes(span.status));
    $('batchBox').className = `notice ${failed.length ? 'bad' : ''}`;
    $('batchBox').innerHTML = `${batch.acceptance_mode === 'FIXTURE' ? '<b>测试模式</b> · ' : ''}任务状态：${esc(statusLabel(batch.status))}<br>已完成档距：${completed.length}/${spans.length}${failed.length ? `<br>失败：${failed.map(span => esc(spanName(span))).join('、')}` : ''}`;
    const treeCount = completed.reduce((sum, span) => sum + Number(span.summary?.tree_count || 0), 0);
    $('resultBox').innerHTML = completed.length ? `<b>已有 ${completed.length} 个档距结果</b><br>单木：${treeCount} · 预警净空阈值：${esc(batch.risk_policy?.warning_clearance_threshold_m)} m<br>导线或净空证据不足时保持“需复核”。` : '分析完成后显示结果。';
  }

  function setStepper(uiState) {
    const active = {NO_PROJECT: 1, PROJECT_CREATED: 2, SOURCE_IMPORTED_UNIT_PENDING: 2, TOWER_SELECTION: 3, TOPOLOGY_READY: 4, ANALYSIS_RUNNING: 4, TREE_REVIEW_READY: 5, FINALIZED: 6}[uiState] || 1;
    document.querySelectorAll('#stepper span').forEach(node => {
      const step = Number(node.dataset.step);
      node.classList.toggle('active', step === active);
      node.classList.toggle('done', step < active);
    });
  }
  function show(id, visible = true) { $(id).classList.toggle('hidden', !visible); }
  function stageCard(kicker, title, body, actions = '', extra = '') {
    return `<div class="stage-card"><div class="stage-kicker">${kicker}</div><h1>${title}</h1><p>${body}</p>${extra}${actions ? `<div class="stage-actions">${actions}</div>` : ''}</div>`;
  }
  function towerPicker() {
    return `<div class="picker-card"><div class="picker-head"><div><h3>真实点云俯视图（XY）</h3><p>请沿线路方向依次点击实际杆塔。点击将使用真实局部点云推导坐标与高度。</p></div><div class="picker-controls"><select id="displayMode"><option value="rgb">RGB</option><option value="height">高度着色</option><option value="intensity">强度</option></select><label>点大小 <input id="pointSize" type="range" min="1" max="5" value="2"></label><button id="resetTowerView" class="small">重置视角</button></div></div><div id="towerLoad" class="notice">正在准备点云显示…</div><div class="picker-grid"><canvas id="towerCanvas" aria-label="真实点云俯视图"></canvas><canvas id="tower3dCanvas" aria-label="3D确认视图"></canvas></div><div class="picker-caption"><span>左：俯视 XY（滚轮缩放、拖拽平移、点击选杆）</span><span>右：3D 确认（最近一次点击的真实局部点云）</span></div></div>`;
  }
  function pipelineMarkup(mode) {
    const names = ['地面处理', '导线识别', '植被提取', '单木分割', '跨档整理', '净空计算', '风险分析'];
    return names.map(name => `<div class="stage ${mode}">${name}<br><small>${mode === 'done' ? '完成' : (mode === 'fail' ? '失败' : '进行中')}</small></div>`).join('');
  }

  function renderWorkspace() {
    const uiState = state.uiState;
    setStepper(uiState);
    ['capabilitySection', 'towerSection', 'batchSection', 'resultSection', 'objectSection'].forEach(id => show(id, false));
    show('riskSection', uiState !== 'NO_PROJECT');
    show('analysisOverview', false);
    show('viewerWrap', false);
    $('center').classList.remove('review-mode');
    document.querySelectorAll('.secondary-action').forEach(button => button.classList.toggle('hidden', uiState === 'NO_PROJECT'));
    document.querySelectorAll('.test-only').forEach(node => node.classList.toggle('hidden', !state.testMode));

    if (uiState === 'NO_PROJECT') {
      $('stageHost').innerHTML = stageCard('欢迎使用', 'TreeLineWarning Survey', '树线智能巡检软件。请选择新建项目或打开已有项目，开始本地巡检工作。', '<button class="primary" data-action="new-project">新建项目</button><button data-action="open-project">打开项目</button>', '<div class="feature-list"><div class="feature"><b>全离线运行</b>无需互联网连接</div><div class="feature"><b>支持 LAS / LAZ</b>本地点云输入</div><div class="feature"><b>数据保存在本机</b>项目可关闭后继续</div></div>');
    } else if (uiState === 'PROJECT_CREATED') {
      $('stageHost').innerHTML = stageCard('步骤 1 / 6', '导入点云', '项目已创建。下一步请选择本机 LAS/LAZ 点云文件。', '<button class="primary" data-action="import-pointcloud">导入 LAS/LAZ</button>', `<div class="project-detail"><b>项目：</b>${esc(state.project.project_name)}<br><b>路径：</b>${esc(state.project.root)}<br><b>状态：</b>等待导入点云</div>`);
    } else if (uiState === 'SOURCE_IMPORTED_UNIT_PENDING') {
      show('capabilitySection');
      const coord = state.capability?.coordinate_declaration || {};
      $('stageHost').innerHTML = stageCard('步骤 2 / 6', '确认点云信息', '点云已导入。请核对点数、坐标参考系（CRS）和单位；确认单位后才能进入杆塔选取。', '<button class="primary" data-action="focus-unit">确认点云单位</button>', `<div class="source-summary"><div class="summary-tile"><span>点云文件</span><b>${esc(state.capability?.details?.format || 'LAS/LAZ')}</b></div><div class="summary-tile"><span>点数</span><b>${esc(state.capability?.details?.point_count || '待读取')}</b></div><div class="summary-tile"><span>坐标参考系（CRS）</span><b>${esc(coord.horizontal_crs || '未知')}</b></div></div>`);
    } else if (uiState === 'TOWER_SELECTION') {
      show('capabilitySection'); show('towerSection'); ensureTowerRows();
      $('stageHost').innerHTML = `<div class="stage-card picker-stage"><div class="stage-kicker">步骤 3 / 6</div><h1>杆塔选取</h1>${towerPicker()}</div>`;
      window.setTimeout(loadTowerViewer, 0);
    } else if (uiState === 'TOPOLOGY_READY') {
      show('towerSection'); show('batchSection');
      $('stageHost').innerHTML = stageCard('步骤 4 / 6', '准备开始分析', '线路拓扑已确认。请选择需要处理的档距，然后开始离线分析。', '<button class="primary" data-action="start-analysis">开始分析</button>', `<div class="topology-summary"><div class="summary-tile"><span>线路</span><b>${state.project.lines ?? 0}</b></div><div class="summary-tile"><span>杆塔</span><b>${state.project.towers ?? 0}</b></div><div class="summary-tile"><span>逻辑档距</span><b>${state.project.spans ?? 0}</b></div></div>`);
    } else if (uiState === 'ANALYSIS_RUNNING') {
      show('batchSection'); show('analysisOverview');
      $('pipelineStages').innerHTML = pipelineMarkup('current');
      $('stageHost').innerHTML = stageCard('步骤 4 / 6', '正在分析', '软件正在本机完成地面、导线、植被、单木、净空和风险分析。可继续查看右侧任务状态。', '', '<div class="stage-progress"><i></i></div><div class="notice">请保持软件运行。全部计算均在本机完成。</div>');
    } else if (uiState === 'TREE_REVIEW_READY') {
      show('resultSection'); show('objectSection'); show('analysisOverview'); show('viewerWrap');
      $('center').classList.add('review-mode');
      $('pipelineStages').innerHTML = pipelineMarkup('done');
      $('stageHost').innerHTML = '<div class="stage-card compact"><div class="stage-kicker">步骤 5 / 6</div><h2>单木结果复核</h2><p>分析结果已就绪。可在下方视图删除、合并或撤销；确认无误后完成复核。</p></div>';
      loadViewer();
    } else if (uiState === 'FINALIZED') {
      show('resultSection'); show('objectSection'); show('analysisOverview');
      $('pipelineStages').innerHTML = pipelineMarkup('done');
      const completed = (state.batch.spans || []).filter(span => span.status === 'FINALIZED').length;
      $('stageHost').innerHTML = stageCard('步骤 6 / 6', '结果已完成', '单木复核和档距结果已定稿。现在可以查看结果或导出预测输入。', '<button data-action="open-viewer">打开复核视图</button><button class="primary" data-action="export-prediction">导出预测输入</button>', `<div class="result-summary"><div class="summary-tile"><span>已定稿档距</span><b>${completed}</b></div><div class="summary-tile"><span>风险阈值</span><b>${esc(state.batch.risk_policy?.warning_clearance_threshold_m)} m</b></div><div class="summary-tile"><span>结果状态</span><b class="status-good">已完成</b></div></div>`);
    }
  }

  async function loadWorkspace() {
    try {
      const payload = await api('/api/desktop/workspace');
      state.uiState = payload.ui_state;
      state.treeReviewReady = Boolean(payload.tree_review_ready);
      state.finalized = Boolean(payload.finalized);
      state.testMode = Boolean(payload.test_mode);
      renderProject(payload.project);
      renderCapability(payload.capability);
      renderBatch(payload.batch);
      renderWorkspace();
      setStatus('离线本地服务已就绪', 'good');
    } catch (error) {
      setStatus(userError(error, '本地服务暂时不可用，请重新启动软件。'), 'bad');
      $('stageHost').innerHTML = stageCard('连接失败', '无法连接本地服务', '请关闭软件后重新打开；详细信息已写入项目日志。');
    }
  }

  async function createProject() {
    try {
      const bridge = await getNativeBridge(['choose_project_location', 'create_project']);
      if (!bridge) {
        await jsonPost('/api/desktop/create', {project_name: `界面验收项目-${Date.now()}`});
      } else {
        const parent = await bridge.choose_project_location();
        if (!parent) { setStatus('已取消新建项目'); return; }
        const name = window.prompt('请输入项目名称：');
        if (!name?.trim()) { setStatus('已取消新建项目'); return; }
        await bridge.create_project(parent, name.trim());
      }
      state.viewerLoaded = false;
      state.towerViewer = null; state.localPreview = null; state.towerView = {scale: 1, offsetX: 0, offsetY: 0};
      await loadWorkspace();
      setStatus('项目已创建，可以导入点云', 'good');
    } catch (error) {
      setStatus(userError(error, '新建项目失败，请检查保存位置或项目名称后重试。'), 'bad');
    }
  }
  async function openProject() {
    try {
      const bridge = await getNativeBridge(['choose_existing_project']);
      if (!bridge) { setStatus('浏览器验收模式使用当前本地项目', 'warn'); return; }
      const result = await bridge.choose_existing_project();
      if (!result) { setStatus('已取消打开项目'); return; }
      state.viewerLoaded = false;
      state.towerViewer = null; state.localPreview = null; state.towerView = {scale: 1, offsetX: 0, offsetY: 0};
      await loadWorkspace();
      setStatus('项目已打开', 'good');
    } catch (error) {
      setStatus(userError(error, '打开项目失败，请确认所选文件夹是有效项目。'), 'bad');
    }
  }
  async function importPointCloud() {
    try {
      const bridge = await getNativeBridge(['import_point_cloud']);
      if (!bridge) await jsonPost('/api/desktop/fixture-source');
      else if (!await bridge.import_point_cloud()) { setStatus('已取消导入点云'); return; }
      state.towerViewer = null; state.localPreview = null; state.towerView = {scale: 1, offsetX: 0, offsetY: 0};
      await loadWorkspace();
      setStatus('点云已导入，请确认坐标系和单位', 'good');
    } catch (error) {
      setStatus(userError(error, '点云导入失败，请选择可读取的 LAS/LAZ 文件。'), 'bad');
    }
  }
  async function importGIS() {
    try {
      const bridge = await getNativeBridge(['import_gis']);
      if (!bridge) { setStatus('GIS 导入仅在桌面软件中提供', 'warn'); return; }
      if (!await bridge.import_gis()) { setStatus('已取消导入 GIS'); return; }
      await loadWorkspace();
      setStatus('GIS 数据已导入', 'good');
    } catch (error) { setStatus(userError(error, 'GIS 导入失败，请检查文件格式。'), 'bad'); }
  }
  async function confirmUnit() {
    try {
      await jsonPost('/api/desktop/capability/confirm', {unit: $('unitSelect').value});
      await loadWorkspace();
      setStatus('点云单位已确认，可以选取杆塔', 'good');
    } catch (error) { setStatus(userError(error, '单位确认失败，请重新选择。'), 'bad'); }
  }
  function colorFor(point, mode) {
    if (mode === 'height') { const z = point[2]; return `hsl(${220 - Math.max(0, Math.min(220, z % 220))},75%,58%)`; }
    if (mode === 'intensity') { const c = Math.max(30, Math.min(245, point[3] / 256)); return `rgb(${c},${c},${c})`; }
    return `rgb(${point[4]},${point[5]},${point[6]})`;
  }
  function drawTowerPicker() {
    const canvas = $('towerCanvas'); if (!canvas || !state.towerViewer) return;
    const ctx = canvas.getContext('2d'); const rect = canvas.getBoundingClientRect();
    canvas.width = Math.max(1, Math.floor(rect.width * devicePixelRatio)); canvas.height = Math.max(1, Math.floor(rect.height * devicePixelRatio));
    ctx.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0); ctx.fillStyle = '#071015'; ctx.fillRect(0, 0, rect.width, rect.height);
    const b = state.towerViewer.source_bounds, view = state.towerView, spanX = Math.max(1, b[3] - b[0]), spanY = Math.max(1, b[4] - b[1]);
    const base = Math.min(rect.width / spanX, rect.height / spanY) * .92 * view.scale;
    const project = p => [rect.width / 2 + (p[0] - (b[0] + b[3]) / 2) * base + view.offsetX, rect.height / 2 - (p[1] - (b[1] + b[4]) / 2) * base + view.offsetY];
    canvas._towerProject = project; canvas._towerBase = base;
    const size = Number($('pointSize')?.value || 2), mode = $('displayMode')?.value || 'rgb';
    ctx.globalAlpha = .68;
    for (let i = 0; i < state.towerViewer.xyz.length; i++) { const p = state.towerViewer.xyz[i], q = project(p); ctx.fillStyle = colorFor([p[0], p[1], p[2], state.towerViewer.intensity[i], ...state.towerViewer.rgb[i]], mode); ctx.fillRect(q[0], q[1], size, size); }
    ctx.globalAlpha = 1;
    state.towerRows.forEach((tower, i) => { const q = project([tower.x, tower.y]); ctx.strokeStyle = '#fff'; ctx.lineWidth = 3; ctx.fillStyle = '#ff9f1c'; ctx.beginPath(); ctx.arc(q[0], q[1], 9, 0, Math.PI * 2); ctx.fill(); ctx.stroke(); ctx.fillStyle = '#fff'; ctx.font = 'bold 13px Microsoft YaHei'; ctx.fillText(tower.display_name || `T${String(i + 1).padStart(3, '0')}`, q[0] + 11, q[1] - 10); });
  }
  function drawLocal3d() {
    const canvas = $('tower3dCanvas'); if (!canvas) return; const ctx = canvas.getContext('2d'), rect = canvas.getBoundingClientRect();
    canvas.width = Math.max(1, Math.floor(rect.width * devicePixelRatio)); canvas.height = Math.max(1, Math.floor(rect.height * devicePixelRatio)); ctx.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0); ctx.fillStyle = '#071015'; ctx.fillRect(0, 0, rect.width, rect.height);
    const local = state.localPreview; if (!local) { ctx.fillStyle = '#a8bdc8'; ctx.font = '14px Microsoft YaHei'; ctx.fillText('点击左侧真实点云中的杆塔后显示局部 3D 确认', 20, 40); return; }
    const xyz = local.xyz, min = [Math.min(...xyz.map(p => p[0])), Math.min(...xyz.map(p => p[1])), Math.min(...xyz.map(p => p[2]))], max = [Math.max(...xyz.map(p => p[0])), Math.max(...xyz.map(p => p[1])), Math.max(...xyz.map(p => p[2]))];
    const scale = Math.min(rect.width / Math.max(1, max[0]-min[0]+max[1]-min[1]), rect.height / Math.max(1, max[2]-min[2])) * .82;
    for (let i = 0; i < xyz.length; i++) { const p=xyz[i], qx=rect.width/2+(p[0]-min[0]-(p[1]-min[1]))*scale, qy=rect.height-25-(p[2]-min[2])*scale+(p[0]-min[0]+p[1]-min[1])*scale*.22; const c=local.rgb[i]; ctx.fillStyle=`rgb(${c[0]},${c[1]},${c[2]})`; ctx.fillRect(qx,qy,2,2); }
  }
  async function loadTowerViewer() {
    if (!$('towerCanvas') || state.towerViewer) { drawTowerPicker(); drawLocal3d(); return; }
    try { $('towerLoad').textContent = '正在加载俯视点云…'; const data = await api('/api/desktop/tower-viewer/overview'); state.towerViewer = data; $('towerLoad').textContent = `真实点云已就绪：显示 ${data.display_point_count.toLocaleString()} 个浏览样本点（源点 ${data.source_point_count.toLocaleString()} 个）`; drawTowerPicker(); drawLocal3d(); bindTowerCanvas(); }
    catch (error) { $('towerLoad').className = 'notice bad'; $('towerLoad').textContent = userError(error, '点云显示加载失败，请重试。'); }
  }
  function bindTowerCanvas() {
    const canvas = $('towerCanvas'); if (!canvas || canvas._bound) return; canvas._bound = true; let drag = null;
    canvas.addEventListener('wheel', e => { e.preventDefault(); state.towerView.scale = Math.max(.25, Math.min(30, state.towerView.scale * (e.deltaY < 0 ? 1.2 : .83))); drawTowerPicker(); }, {passive:false});
    canvas.addEventListener('pointerdown', e => { drag = [e.clientX, e.clientY]; canvas.setPointerCapture(e.pointerId); });
    canvas.addEventListener('pointermove', e => { if (!drag) return; state.towerView.offsetX += e.clientX-drag[0]; state.towerView.offsetY += e.clientY-drag[1]; drag=[e.clientX,e.clientY]; drawTowerPicker(); });
    canvas.addEventListener('pointerup', async e => { if (!drag) return; const moved = Math.hypot(e.clientX-drag[0], e.clientY-drag[1]); drag=null; if (moved > 4) return; const r=canvas.getBoundingClientRect(), b=state.towerViewer.source_bounds, base=canvas._towerBase, v=state.towerView; const x=(e.clientX-r.left-r.width/2-v.offsetX)/base+(b[0]+b[3])/2, y=(r.height/2-(e.clientY-r.top)-v.offsetY)/base+(b[1]+b[4])/2; try { $('towerLoad').textContent='正在加载局部高密度点云…'; const result=await jsonPost('/api/desktop/tower-viewer/refine',{x,y}); const tower={...result.tower,display_name:`T${String(state.towerRows.length+1).padStart(3,'0')}`}; state.towerRows.push(tower); state.localPreview=result.local; renderTowerRows(); $('towerLoad').textContent=`已选择 ${tower.display_name}：真实坐标已由局部点云确认`; drawTowerPicker(); drawLocal3d(); } catch(error) { $('towerLoad').className='notice bad'; $('towerLoad').textContent=userError(error,'杆塔局部点云加载失败，请缩放后重试。'); }});
    $('displayMode').addEventListener('change', drawTowerPicker); $('pointSize').addEventListener('input', drawTowerPicker); $('resetTowerView').addEventListener('click', () => { state.towerView={scale:1,offsetX:0,offsetY:0}; drawTowerPicker(); }); window.addEventListener('resize', () => { drawTowerPicker(); drawLocal3d(); });
  }
  function selectedSpanIds() {
    return [...document.querySelectorAll('#spanChecks input[data-span]:checked')].map(input => input.dataset.span);
  }
  async function saveTowers() {
    try {
      const towers = state.towerRows;
      await jsonPost('/api/desktop/towers', {line_name: $('lineName').value, line_role: $('lineRole').value, parent_line_id: $('parentLine').value || null, towers});
      await loadWorkspace();
      setStatus('杆塔顺序已保存，逻辑档距已生成', 'good');
    } catch (error) {
      $('towerMessage').textContent = userError(error, '无法保存杆塔，请检查名称、坐标和顺序。');
      setStatus($('towerMessage').textContent, 'bad');
    }
  }
  async function startBatch(mode) {
    try {
      const spanIds = selectedSpanIds();
      if (!spanIds.length) { setStatus('请先勾选至少一个逻辑档距', 'warn'); return; }
      const started = await jsonPost('/api/desktop/batch/start', {span_ids: spanIds, mode});
      renderBatch(started);
      state.uiState = started.status === 'RUNNING' ? 'ANALYSIS_RUNNING' : state.uiState;
      renderWorkspace();
      setStatus(mode === 'FIXTURE' ? '测试分析正在运行' : '离线分析正在运行', 'good');
      window.setTimeout(pollBatch, 900);
    } catch (error) { setStatus(userError(error, '无法开始分析，请检查点云、单位和档距状态。'), 'bad'); }
  }
  async function pollBatch() {
    try {
      await loadWorkspace();
      if (state.batch.status === 'RUNNING' || state.batch.spans?.some(span => ['PREPARING', 'RUNNING'].includes(span.status))) window.setTimeout(pollBatch, 600);
      else setStatus(state.batch.status === 'FAILED' ? '分析失败，请查看本地日志' : '分析已完成，可以复核单木结果', state.batch.status === 'FAILED' ? 'bad' : 'good');
    } catch (error) { setStatus(userError(error, '暂时无法读取分析进度。'), 'bad'); }
  }
  async function retryBatch() {
    try { await jsonPost('/api/desktop/batch/retry'); await loadWorkspace(); pollBatch(); }
    catch (error) { setStatus(userError(error, '重试失败任务未能启动。'), 'bad'); }
  }
  function loadViewer(force = false) {
    if (!state.treeReviewReady && !state.finalized) { setStatus('单木结果尚未就绪', 'warn'); return; }
    show('viewerWrap'); $('center').classList.add('review-mode');
    if (!state.viewerLoaded || force) {
      $('viewer').src = `/step2?run=${Date.now()}`;
      state.viewerLoaded = true;
    }
    setStatus('单木结果复核视图已打开', 'good');
  }
  function localizeViewerFrame() {
    try {
      const documentRef = $('viewer').contentDocument;
      if (!documentRef?.body) return;
      const replacements = new Map([
        ['TreeLineWarning — 文件夹到 Excel', 'TreeLineWarning — 单木结果复核'],
        ['STEP 2 人工审核单木', '单木结果复核'],
        ['STEP 2 · Tree Editor', '单木结果复核'],
      ]);
      const translate = () => {
        const walker = documentRef.createTreeWalker(documentRef.body, NodeFilter.SHOW_TEXT);
        let node;
        while ((node = walker.nextNode())) {
          const trimmed = node.nodeValue.trim();
          if (replacements.has(trimmed)) node.nodeValue = node.nodeValue.replace(trimmed, replacements.get(trimmed));
        }
      };
      translate();
      if (!documentRef._treelineChineseObserver) {
        documentRef._treelineChineseObserver = new MutationObserver(translate);
        documentRef._treelineChineseObserver.observe(documentRef.body, {childList: true, subtree: true, characterData: true});
      }
      documentRef.title = 'TreeLineWarning — 单木结果复核';
    } catch (error) {
      console.warn('单木复核视图中文化未完成', error);
    }
  }
  async function finalizeBatch() {
    try { await jsonPost('/api/desktop/batch/finalize'); await loadWorkspace(); setStatus('单木复核已完成', 'good'); }
    catch (error) { setStatus(userError(error, '无法完成复核，请先检查单木结果。'), 'bad'); }
  }
  async function exportPrediction() {
    try {
      if (!state.finalized) await jsonPost('/api/desktop/batch/finalize');
      window.open('/download_prediction', '_blank');
      await loadWorkspace();
      setStatus('已请求导出 Prediction_Input.xlsx', 'good');
    } catch (error) { setStatus(userError(error, '导出失败，请先完成单木复核。'), 'bad'); }
  }
  async function saveRisk() {
    const value = Number($('riskThreshold').value);
    if (!(value > 0)) { setStatus('预警净空阈值必须大于 0', 'warn'); return; }
    try { await jsonPost('/api/desktop/risk-policy', {warning_clearance_threshold_m: value}); await loadWorkspace(); setStatus('风险阈值已保存', 'good'); }
    catch (error) { setStatus(userError(error, '风险阈值保存失败。'), 'bad'); }
  }

  $('newProject').addEventListener('click', createProject);
  $('openProject').addEventListener('click', openProject);
  $('importPointCloud').addEventListener('click', importPointCloud);
  $('importGIS').addEventListener('click', importGIS);
  $('confirmUnit').addEventListener('click', confirmUnit);
  $('saveTowers').addEventListener('click', saveTowers);
  $('removeTower').addEventListener('click', () => { if (state.towerRows.length) { state.towerRows.pop(); renderTowerRows(); drawTowerPicker(); } });
  $('startFixture').addEventListener('click', () => startBatch('FIXTURE'));
  $('startReal').addEventListener('click', () => startBatch('REAL'));
  $('retryBatch').addEventListener('click', retryBatch);
  $('openViewer').addEventListener('click', () => loadViewer());
  $('refreshViewer').addEventListener('click', () => loadViewer(true));
  $('viewer').addEventListener('load', localizeViewerFrame);
  $('finalizeBatch').addEventListener('click', finalizeBatch);
  $('exportPrediction').addEventListener('click', exportPrediction);
  $('exportWorkOrder').addEventListener('click', exportPrediction);
  $('saveRisk').addEventListener('click', saveRisk);
  $('stageHost').addEventListener('click', event => {
    const action = event.target.closest('[data-action]')?.dataset.action;
    if (action === 'new-project') createProject();
    else if (action === 'open-project') openProject();
    else if (action === 'import-pointcloud') importPointCloud();
    else if (action === 'focus-unit') $('unitSelect').focus();
    else if (action === 'focus-towers') $('lineName').focus();
    else if (action === 'start-analysis') startBatch('REAL');
    else if (action === 'open-viewer') loadViewer();
    else if (action === 'export-prediction') exportPrediction();
  });
  window.addEventListener('error', event => {
    console.error('Unhandled UI error', event.error || event.message);
    setStatus('界面发生错误，详细信息已写入本地日志', 'bad');
  });
  window.addEventListener('unhandledrejection', event => {
    console.error('Unhandled promise rejection', event.reason);
    setStatus('操作未能完成，详细信息已写入本地日志', 'bad');
  });

  loadWorkspace();
})();
