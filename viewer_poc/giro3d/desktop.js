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
    towerMain3d: null,
    towerLocal3d: null,
    towerTimings: null,
    treeReviewReady: false,
    finalized: false,
    canPredict: false,
    testMode: false,
    viewerLoaded: false,
    activeSpanId: null,
    executionMode: 'AUTO',
    warningFrameLoaded: false,
  };
  const browserAcceptance = new URLSearchParams(window.location.search).has('browser-test');
  const warningModuleUrl = new URLSearchParams(window.location.search).get('warning_url');
  const $ = id => document.getElementById(id);
  let currentRoute = null, restoringNavigation = false, pinnedProcessingView = false;
  const previousRoutes = [];
  const processingLabels = {NO_PROJECT:'项目起始页',PROJECT_CREATED:'导入点云',SOURCE_IMPORTED_UNIT_PENDING:'确认点云信息',TOWER_SELECTION:'杆塔选取',TOPOLOGY_READY:'分析准备',ANALYSIS_RUNNING:'分析进度',TREE_REVIEW_READY:'单木复核',FINALIZED:'当前项目结果'};
  function routeKey(route) {
    if (route.kind === 'warning') return `warning:${route.token}`;
    if (route.url) { const url=new URL(route.url,location.href);url.searchParams.delete('run');return `${route.kind}:${url.pathname}${url.search}`; }
    return `${route.kind}:${route.uiState}:${route.tool||''}`;
  }
  function updateBackButton() {
    const previous=previousRoutes.at(-1),button=$('navigateBack');
    button.disabled=!previous;button.title=previous?`返回：${previous.label}；保留项目数据和输入`:'当前是起始页面';
  }
  function rememberRoute(route) {
    if(restoringNavigation)return;
    if(currentRoute&&routeKey(currentRoute)===routeKey(route)){currentRoute={...currentRoute,...route};updateBackButton();return;}
    if(currentRoute)previousRoutes.push(currentRoute);
    currentRoute=route;updateBackButton();
  }
  function showModuleFrame(frameId) {
    document.querySelector('.left')?.classList.add('hidden');$('center').classList.add('hidden');document.querySelector('.right')?.classList.add('hidden');
    show('leftSplitter',false);show('rightSplitter',false);show('moduleFrameWrap',true);
    show('moduleFrame',frameId==='moduleFrame');show('auxiliaryFrame',frameId==='auxiliaryFrame');
  }
  function navigateBack() {
    const route=previousRoutes.pop();if(!route)return;
    currentRoute=route;restoringNavigation=true;pinnedProcessingView=route.kind==='processing';
    try {
      if(route.kind==='warning'){
        showModuleFrame('moduleFrame');setActiveTask(route.task);
        $('moduleFrame').contentWindow.postMessage({type:'treeline-restore-view',token:route.token},new URL(warningModuleUrl).origin);
      } else if(route.kind==='auxiliary'){
        showModuleFrame('auxiliaryFrame');setActiveTask(route.task);if($('auxiliaryFrame').contentWindow.location.href!==route.url)$('auxiliaryFrame').src=route.url;
      } else if(route.kind==='viewer'){
        showProcessingWorkspace('ribbonReview',false);show('viewerWrap');$('center').classList.add('review-mode');
        if($('viewer').contentWindow.location.href!==route.url)$('viewer').src=route.url;
      } else {
        showProcessingWorkspace(route.task,false);
        if(route.tool)focusProcessingTool(route.tool,route.uiState);else renderWorkspace(route.uiState,false);
      }
      setStatus(`已返回：${route.label}。可修改后继续；返回不会删除结果或撤销点云编辑。`,'good');
    } finally {restoringNavigation=false;updateBackButton();}
  }
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
  const executionModes = new Set(['AUTO', 'SERIAL', 'PARALLEL']);
  const normalizeExecutionMode = value => {
    const mode = String(value || '').toUpperCase();
    return executionModes.has(mode) ? mode : 'AUTO';
  };
  function selectedExecutionMode() {
    const select = $('executionMode');
    const mode = normalizeExecutionMode(select?.value || state.executionMode);
    state.executionMode = mode;
    return mode;
  }
  function executionTelemetry(batch) {
    return batch?.telemetry?.EXECUTION ?? batch?.EXECUTION ?? batch?.execution ?? batch?.execution_status ?? null;
  }
  function executionTelemetryText(value) {
    if (value === null || value === undefined || value === '') return '';
    if (typeof value === 'string' || typeof value === 'number' || typeof value === 'boolean') return String(value);
    if (Array.isArray(value)) return value.map(item => executionTelemetryText(item)).filter(Boolean).join(' · ');
    if (typeof value === 'object') {
      const fields = [
        ['mode', value.effective_mode || value.actual_mode || value.mode || value.execution_mode],
        ['状态', value.status],
        ['worker', value.worker_count ?? value.workers ?? value.max_workers],
        ['原因', value.reason || value.fallback_reason || value.message],
      ].filter(([, item]) => item !== undefined && item !== null && item !== '');
      if (fields.length) return fields.map(([label, item]) => `${label}=${item}`).join(' · ');
      try {
        const serialized = JSON.stringify(value);
        return serialized && serialized !== '{}' ? serialized : '';
      } catch (_) { return ''; }
    }
    return String(value);
  }
  function renderFuturePreviewLink() {
    const button = $('openFuturePreview');
    const note = $('futurePreviewNote');
    if (!button) return;
    const allowed = Boolean(state.canPredict);
    button.disabled = !allowed;
    button.textContent = allowed ? '预测与模拟' : '预测与模拟 🔒';
    button.classList.toggle('future-preview-locked', !allowed);
    button.title = allowed ? '打开当前树木点云的本地场景预测' : '请先完成点云分析并确认至少一棵树。';
    if (note) {
      note.textContent = allowed
        ? (state.finalized
          ? '✓ 本地基础预测仅用于实验性场景预览，不替代当前正式勘察结果。'
          : '✓ 可按当前处理后的树木点云预览；未定稿结果会明确标记为实验性。')
        : '请先完成点云分析并确认至少一棵树。';
      note.classList.toggle('status-good', allowed);
      note.classList.toggle('status-warn', !allowed);
    }
  }
  function openFuturePreview() {
    if (state.canPredict) openTwinWorkspace();
    else setStatus('请先完成点云分析并确认至少一棵树', 'warn');
  }
  const userError = (error, fallback) => {
    console.error(fallback, error);
    return /[\u3400-\u9fff]/.test(error?.message || '') ? error.message : fallback;
  };
  const setStatus = (message, kind = '') => {
    $('footerState').textContent = `应用状态：${message}`;
    $('footerState').className = kind ? `status-${kind}` : '';
  };
  function setActiveTask(buttonId) {
    document.querySelectorAll('.ribbon-buttons button').forEach(button => button.classList.toggle('active-task', button.id === buttonId));
  }
  function showProcessingWorkspace(buttonId = '', record = true) {
    show('moduleFrameWrap', false);
    show('leftSplitter', true); show('rightSplitter', true);
    document.querySelector('.left')?.classList.remove('hidden');
    $('center')?.classList.remove('hidden');
    document.querySelector('.right')?.classList.remove('hidden');
    setActiveTask(buttonId);
    if(record){pinnedProcessingView=false;rememberRoute({kind:'processing',uiState:state.uiState,task:buttonId,label:processingLabels[state.uiState]||'点云处理'});}
  }
  function openWarningWorkspace(workspace = 'single') {
    if (workspace === 'project-batch') workspace = 'single';
    if (!warningModuleUrl) {
      setStatus('风险预测预警模块未启动', 'bad');
      return;
    }
    const target = new URL(warningModuleUrl);
    target.searchParams.set('processing_url', window.location.href);
    target.searchParams.set('workspace', workspace);
    target.searchParams.set('embedded', '1');
    document.querySelector('.left')?.classList.add('hidden');
    $('center')?.classList.add('hidden');
    document.querySelector('.right')?.classList.add('hidden');
    show('leftSplitter', false); show('rightSplitter', false);
    show('moduleFrameWrap', true);
    show('moduleFrame',true);show('auxiliaryFrame',false);pinnedProcessingView=false;
    const taskButtons = {data: 'ribbonData', single: 'openWarningModule', batch: 'ribbonExternalBatch', training: 'ribbonTraining', report: 'exportWorkOrder', history: 'ribbonHistory'};
    setActiveTask(taskButtons[workspace] || 'openWarningModule');
    const frame = $('moduleFrame');
    const warningOrigin = new URL(warningModuleUrl).origin;
    let reuseFrame = false;
    try { reuseFrame = state.warningFrameLoaded && new URL(frame.src).origin === warningOrigin; } catch (_) { reuseFrame = false; }
    state.warningWorkspace = workspace;
    if (reuseFrame) {
      frame.contentWindow.postMessage({type: 'treeline-workspace', workspace}, warningOrigin);
    } else {
      // Give every task switch a distinct URL.  In particular, the external
      // batch page must not be allowed to keep showing the previously loaded
      // current-project data page while that page is still fetching results.
      target.searchParams.set('_workspace_seq', String(Date.now()));
      state.warningFrameLoaded = false;
      frame.src = target.toString();
    }
    setStatus(`已在当前主界面打开：${{data:'查看数据',single:'树木预测（1～多棵）','project-batch':'当前项目多木预测',batch:'外部表格批量预测',training:'模型训练',report:'预测报告与工单',history:'历史记录'}[workspace] || '预测模块'}`, 'good');
  }
  function exitReview() {
    // A project can open directly into review with no navigation history.
    // Returning must still leave the embedded viewer, without deleting edits.
    if (previousRoutes.length) { navigateBack(); return; }
    showProcessingWorkspace('ribbonPreprocess', false);
    focusProcessingTool('点云预处理');
    pinnedProcessingView = true;
    setStatus('已退出复核，项目及已提交编辑保留。', 'good');
  }
  function openWarningModule() { openWarningWorkspace('single'); }
  function openTwinWorkspace() {
    if (!state.canPredict) { setStatus('请先完成点云分析并确认至少一棵树', 'warn'); return; }
    document.querySelector('.left')?.classList.add('hidden');
    $('center')?.classList.add('hidden');
    document.querySelector('.right')?.classList.add('hidden');
    show('leftSplitter', false); show('rightSplitter', false);
    show('moduleFrameWrap', true);
    show('moduleFrame',false);show('auxiliaryFrame',true);pinnedProcessingView=false;
    setActiveTask('ribbonTwin');
    const target = new URL('/future-preview.html', window.location.href);
    target.searchParams.set('embedded', '1');
    const frame=$('auxiliaryFrame');
    if(frame.src!==target.toString())frame.src=target.toString();
    rememberRoute({kind:'auxiliary',url:target.toString(),task:'ribbonTwin',label:'点云数字孪生'});
    setStatus('已在当前主界面打开：点云数字孪生', 'good');
  }
  function focusProcessingTool(label, viewState = state.uiState) {
    const taskButtons = {'点云预处理':'ribbonPreprocess','地面点去除':'ribbonGround','导线重建':'ribbonWire','单木分割':'ribbonTrees','点云分析':'ribbonPreprocess'};
    showProcessingWorkspace(taskButtons[label] || 'ribbonPreprocess',false);
    if(!restoringNavigation)pinnedProcessingView=false;
    rememberRoute({kind:'processing',uiState:viewState,tool:label,task:taskButtons[label]||'ribbonPreprocess',label});
    const details = {
      '点云预处理': '检查点云格式、坐标系、线性单位和基础质量，为后续自动分析准备统一数据。',
      '地面点去除': '打开地面点去除任务界面，用于分离地面与非地面点。',
      '导线重建': '打开导线识别与几何重建任务界面，用于形成树线距离计算基准。',
      '单木分割': '打开单木分割任务界面，用于生成每棵树的高度、胸径、冠幅和空间距离数据。',
      '点云分析': '打开当前项目的点云分析任务界面。',
    };
    const ready = ['TOPOLOGY_READY', 'ANALYSIS_RUNNING', 'TREE_REVIEW_READY', 'FINALIZED'].includes(viewState);
    if (!ready) {
      renderWorkspace(viewState,false);
      setStatus(`请先完成当前准备步骤，然后进入${label}`, 'warn');
      return;
    }
    $('center').classList.remove('review-mode');
    show('viewerWrap', false); show('batchSection', true); show('analysisOverview', true);
    $('pipelineStages').innerHTML = pipelineMarkup(viewState === 'ANALYSIS_RUNNING' ? 'current' : 'done');
    const action = viewState === 'ANALYSIS_RUNNING' ? '' : '<button class="primary" data-action="start-analysis">运行当前项目处理流程</button>';
    $('stageHost').innerHTML = stageCard('点云处理任务', label, details[label] || details['点云分析'], action, '<div class="notice">这一页只显示您点选的任务；运行时依旧保持完整处理链的数据一致性。</div>');
    setStatus(`已打开：${label}`, 'good');
  }
  function exportPredictionExcel() {
    const link = document.createElement('a');
    link.href = '/download_excel';
    link.download = 'Prediction_Input.xlsx';
    document.body.appendChild(link);
    link.click();
    link.remove();
    setStatus('已按预测板块格式生成 Excel', 'good');
  }

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
    if (state.towerMain3d) state.towerMain3d.setMarkers(state.towerRows);
  }
  function ensureTowerRows() { renderTowerRows(); }

  function renderBatch(batch) {
    state.batch = batch;
    $('footerBatch').textContent = `分析任务：${statusLabel(batch.status)}`;
    const requestedExecutionMode = batch.execution_mode || batch.execution_mode_requested;
    const executionSelect = $('executionMode');
    if (requestedExecutionMode && executionSelect && !executionSelect.dataset.userSelected) {
      state.executionMode = normalizeExecutionMode(requestedExecutionMode);
      executionSelect.value = state.executionMode;
    }
    const spans = batch.spans || [];
    if (!state.activeSpanId || !spans.some(span => span.span_id === state.activeSpanId)) state.activeSpanId = spans[0]?.span_id || null;
    $('spanChecks').innerHTML = spans.length ? spans.map(span => {
      const failure = span.failure;
      const failureMarkup = span.status === 'FAILED' && failure ? `<div class="failure-reason"><b>分析失败</b><span>阶段：${esc(failure.failure_stage_label || span.stage)}</span><span>原因：${esc(failure.user_message)}</span><span>建议：${esc(failure.recommended_action)}</span><details><summary>查看详情</summary><span>故障代码：${esc(failure.failure_code)}</span><span>${esc(failure.technical_message)}</span></details></div>` : '';
      return `<label class="check-item ${span.status === 'FAILED' ? 'failed' : ''}"><input type="checkbox" data-span="${esc(span.span_id)}" ${span.selected ? 'checked' : ''} ${span.status === 'FINALIZED' || (span.status === 'FAILED' && !span.retryable) ? 'disabled' : ''}><span><b>${esc(spanName(span))}</b><small>${esc(span.span_id)} · ${esc(span.stage)}</small>${failureMarkup}</span><span class="state ${span.status === 'FAILED' ? 'status-bad' : (span.status === 'FINALIZED' ? 'status-good' : '')}">${esc(statusLabel(span.status))}</span></label>`;
    }).join('') : '<div class="muted padded">暂无档距</div>';
    const failed = spans.filter(span => span.status === 'FAILED');
    const completed = spans.filter(span => ['AUTO_ANALYZED', 'REVIEW_REQUIRED', 'USER_REVIEWED', 'FINALIZED'].includes(span.status));
    $('batchBox').className = `notice ${failed.length ? 'bad' : ''}`;
    const executionRequested = normalizeExecutionMode(batch.execution_mode || batch.execution_mode_requested || state.executionMode);
    const executionActual = executionTelemetryText(executionTelemetry(batch));
    const executionMarkup = `<br>执行模式：${esc(executionRequested)}${executionActual ? `<br>EXECUTION：${esc(executionActual)}` : ''}`;
    $('batchBox').innerHTML = `${batch.acceptance_mode === 'FIXTURE' ? '<b>测试模式</b> · ' : ''}任务状态：${esc(statusLabel(batch.status))}<br>已完成档距：${completed.length}/${spans.length}${failed.length ? `<br>失败：${failed.map(span => esc(spanName(span))).join('、')}` : ''}${executionMarkup}`;
    const treeCount = Number.isInteger(batch.final_tree_count) ? batch.final_tree_count : '—';
    const riskCounts = batch.risk_counts || {};
    $('resultBox').innerHTML = completed.length ? `<b>已有 ${completed.length} 个档距结果</b><br>全局归并单木：${esc(treeCount)} · 预警净空阈值：${esc(batch.risk_policy?.warning_clearance_threshold_m)} m · WARNING：${esc(riskCounts.WARNING ?? 0)} · DANGER：${esc(riskCounts.DANGER ?? 0)} · SAFE：${esc(riskCounts.SAFE ?? 0)}<br>计数来自已保存的最终净空即时重分类；导线或净空证据不足时保持“需复核”。` : '分析完成后显示结果。';
    renderFuturePreviewLink();
    $('lineOverview').innerHTML = spans.map(span => {
      const cls = ['FAILED'].includes(span.status) ? 'failed' : (['REVIEW_REQUIRED'].includes(span.status) ? 'review' : (['AUTO_ANALYZED', 'USER_REVIEWED', 'FINALIZED'].includes(span.status) ? 'done' : ''));
      return `<button class="${cls} ${span.span_id === state.activeSpanId ? 'active' : ''}" data-line-span="${esc(span.span_id)}"><b>${esc(spanName(span))}</b><br>${esc(statusLabel(span.status))}</button>`;
    }).join('') || '<span class="muted">暂无档距</span>';
    document.querySelectorAll('[data-line-span]').forEach(button => button.addEventListener('click', () => selectActiveSpan(button.dataset.lineSpan)));
  }

  function selectActiveSpan(spanId) {
    state.activeSpanId = spanId;
    const box = document.querySelector(`#spanChecks input[data-span="${CSS.escape(spanId)}"]`);
    if (box) { box.checked = true; box.scrollIntoView({block: 'nearest'}); }
    renderBatch(state.batch);
    if (state.treeReviewReady || state.finalized) loadViewer(true);
  }

  function setStepper(uiState) {
    const active = {NO_PROJECT: 1, PROJECT_CREATED: 2, SOURCE_IMPORTED_UNIT_PENDING: 2, TOWER_SELECTION: 3, TOPOLOGY_READY: 4, ANALYSIS_RUNNING: 4, TREE_REVIEW_READY: 5, FINALIZED: 6}[uiState] || 1;
    document.querySelectorAll('#stepper span').forEach(node => {
      const step = Number(node.dataset.step);
      node.classList.toggle('active', step === active);
      node.classList.toggle('done', step < active);
    });
  }
  function show(id, visible = true) {
    if ((id === 'viewerWrap' && !visible) || (id === 'moduleFrameWrap' && visible)) {
      // Leaving a review is navigation, not cancellation of its server task.
      // A late finalize response must not navigate the hidden frame to /step3.
      // renderWorkspace temporarily hides then shows panes in the same tick.
      // Notify only when the resulting layout actually leaves review hidden.
      setTimeout(() => {
        if ($('viewerWrap').classList.contains('hidden') || $('center').classList.contains('hidden')
            || !$('moduleFrameWrap').classList.contains('hidden')) {
          $('viewer')?.contentWindow?.postMessage({type:'treeline-review-hidden'},location.origin);
        }
      }, 0);
    }
    $(id).classList.toggle('hidden', !visible);
  }
  function stageCard(kicker, title, body, actions = '', extra = '') {
    return `<div class="stage-card"><div class="stage-kicker">${kicker}</div><h1>${title}</h1><p>${body}</p>${extra}${actions ? `<div class="stage-actions">${actions}</div>` : ''}</div>`;
  }
  function towerPicker() {
    return `<div class="picker-card"><div class="picker-head"><div><h3>真实三维点云杆塔选取器</h3><p>斜视、侧视或自由旋转看清杆塔后单击；点击将使用真实局部点云确认坐标与高度。</p></div><div class="picker-controls"><select id="displayMode"><option value="height" selected>高度着色</option><option value="rgb">RGB</option><option value="intensity">强度</option></select><label>点大小 <input id="pointSize" type="range" min="1" max="8" value="2"></label><button id="perspectiveTowerView" class="small">透视视图</button><button id="topTowerView" class="small">俯视</button><button id="sideTowerView" class="small">侧视</button><button id="resetTowerView" class="small">重置视角</button></div></div><div id="towerLoad" class="notice">正在准备点云显示…</div><div id="towerDiagnostics" class="picker-diagnostics"></div><div class="picker-grid"><canvas id="towerCanvas" aria-label="可旋转真实三维点云主视图"></canvas><canvas id="tower3dCanvas" aria-label="局部高密度三维确认视图"></canvas></div><div class="picker-caption"><span>左：主 3D（左键旋转 · 右/中键平移 · 滚轮缩放 · 单击选杆）</span><span>右：局部高密度 3D 确认（同样支持旋转、平移、缩放）</span></div></div>`;
  }
  function pipelineMarkup(mode) {
    const names = ['地面处理', '导线识别', '植被提取', '单木分割', '跨档整理', '净空计算', '风险分析'];
    return names.map(name => `<div class="stage ${mode}">${name}<br><small>${mode === 'done' ? '完成' : (mode === 'fail' ? '失败' : '进行中')}</small></div>`).join('');
  }

  function renderWorkspace(uiState = state.uiState, record = true) {
    if(record&&!$('moduleFrameWrap').classList.contains('hidden'))return;
    if(record&&uiState!=='TREE_REVIEW_READY')rememberRoute({kind:'processing',uiState,task:document.querySelector('.active-task')?.id||'',label:processingLabels[uiState]||'点云处理'});
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
      const cacheMessage = state.batch?.source_cache_message || '首次处理大型点云，正在建立可复用缓存。';
      $('stageHost').innerHTML = stageCard('步骤 4 / 6', '正在分析', '软件正在本机完成地面、导线、植被、单木、净空和风险分析。可继续查看右侧任务状态。', '', `<div class="stage-progress"><i></i></div><div class="notice">${esc(cacheMessage)}<br>请保持软件运行。全部计算均在本机完成。</div>`);
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
      const futureAction = state.canPredict
        ? '<button data-action="open-future-preview">预测与模拟</button>'
        : '<button data-action="open-future-preview" disabled>预测与模拟 🔒</button>';
      $('stageHost').innerHTML = stageCard('步骤 6 / 6', '结果已完成', '单木复核和档距结果已定稿。现在可以直接生成预警报告与工单，或查看点云数字孪生趋势。', `<button data-action="open-viewer">打开复核视图</button><button data-action="open-current-report">打开当前最终报告</button>${futureAction}<button class="primary" data-action="open-warning">生成预警报告与工单</button>`, `<div class="result-summary"><div class="summary-tile"><span>已定稿档距</span><b>${completed}</b></div><div class="summary-tile"><span>风险阈值</span><b>${esc(state.batch.risk_policy?.warning_clearance_threshold_m)} m</b></div><div class="summary-tile"><span>结果状态</span><b class="status-good">已完成</b></div></div>`);
    }
  }

  async function loadWorkspace(advanceWorkflow = false) {
    try {
      if(advanceWorkflow)pinnedProcessingView=false;
      const payload = await api('/api/desktop/workspace');
      if(state.project?.root&&payload.project?.root&&state.project.root!==payload.project.root){
        previousRoutes.length=0;currentRoute=null;pinnedProcessingView=false;state.warningFrameLoaded=false;state.warningWorkspace=null;
        $('moduleFrame').src='about:blank';$('auxiliaryFrame').src='about:blank';showProcessingWorkspace('',false);updateBackButton();
      }
      state.uiState = payload.ui_state;
      state.treeReviewReady = Boolean(payload.tree_review_ready);
      state.finalized = Boolean(payload.finalized);
      // The twin is a read-only experimental view.  It may use the current
      // processed tree cloud before formal publication; reports and exports
      // remain protected by their independent finalization gates.
      state.canPredict = state.treeReviewReady || state.finalized || Boolean(payload.can_predict);
      state.testMode = Boolean(payload.test_mode);
      renderProject(payload.project);
      renderCapability(payload.capability);
      renderBatch(payload.batch);
      if(!pinnedProcessingView)renderWorkspace();
      setStatus('离线本地服务已就绪', 'good');
    } catch (error) {
      setStatus(userError(error, '本地服务暂时不可用，请重新启动软件。'), 'bad');
      $('stageHost').innerHTML = stageCard('连接失败', '无法连接本地服务', '请关闭软件后重新打开；详细信息已写入项目日志。');
    }
  }

  async function createProject() {
    try {
      const bridge = await getNativeBridge(['choose_project_location', 'create_project_in_folder']);
      if (!bridge) {
        await jsonPost('/api/desktop/create', {project_name: `界面验收项目-${Date.now()}`});
      } else {
        const folder = await bridge.choose_project_location();
        if (!folder) { setStatus('已取消新建项目'); return; }
        await bridge.create_project_in_folder(folder);
      }
      state.viewerLoaded = false;
      state.towerViewer = null; state.localPreview = null; state.towerMain3d = null; state.towerLocal3d = null;
      await loadWorkspace(true);
      setStatus('项目已直接创建在所选文件夹中，可以导入点云', 'good');
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
      state.towerViewer = null; state.localPreview = null; state.towerMain3d = null; state.towerLocal3d = null;
      await loadWorkspace(true);
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
      state.towerViewer = null; state.localPreview = null; state.towerMain3d = null; state.towerLocal3d = null;
      await loadWorkspace(true);
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
      await loadWorkspace(true);
      setStatus('GIS 数据已导入', 'good');
    } catch (error) { setStatus(userError(error, 'GIS 导入失败，请检查文件格式。'), 'bad'); }
  }
  async function confirmUnit() {
    try {
      await jsonPost('/api/desktop/capability/confirm', {unit: $('unitSelect').value});
      await loadWorkspace(true);
      setStatus('点云单位已确认，可以选取杆塔', 'good');
    } catch (error) { setStatus(userError(error, '单位确认失败，请重新选择。'), 'bad'); }
  }
  async function timedJson(path, options = {}) {
    const requestStart = performance.now();
    const response = await fetch(path, Object.assign({cache: 'no-store'}, options));
    const text = await response.text();
    const transportMs = performance.now() - requestStart;
    const parseStart = performance.now();
    let payload;
    try { payload = JSON.parse(text); }
    catch (_) { payload = {ok: false, error: {message: '本地服务返回了无法读取的数据'}}; }
    const parseMs = performance.now() - parseStart;
    if (!response.ok || payload.ok === false) throw new Error(payload.error?.message || '本地请求失败');
    return {payload, transport_ms: transportMs, js_parse_ms: parseMs, request_start: requestStart};
  }

  function showTowerTimings(timings) {
    state.towerTimings = timings;
    const node = $('towerDiagnostics'); if (!node) return;
    const ms = value => Number(value || 0).toFixed(0);
    node.textContent = `诊断：缓存 ${timings.overview_cache_hit ? '命中' : '新建'} · 源打开 ${ms(timings.source_open_ms)} ms · 缓存查询 ${ms(timings.overview_cache_lookup_ms)} ms · overview 构建 ${ms(timings.overview_build_ms)} ms · Python payload ${ms(timings.python_payload_build_ms)} ms · 传输 ${ms(timings.transport_ms)} ms · JS 解析 ${ms(timings.js_parse_ms)} ms · 3D 初始化 ${ms(timings.gpu_viewer_init_ms)} ms · 首帧 ${ms(timings.first_visible_ms)} ms`;
    console.info('TOWER_VIEWER_TIMINGS', timings);
  }

  async function acceptTowerPick(world) {
    if ($('towerLoad').dataset.loading === 'yes') return;
    $('towerLoad').dataset.loading = 'yes';
    try {
      $('towerLoad').className = 'notice';
      $('towerLoad').textContent = '正在加载局部高密度点云…';
      const timed = await timedJson('/api/desktop/tower-viewer/refine', {
        method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(world),
      });
      const result = timed.payload;
      const tower = {...result.tower, display_name: `T${String(state.towerRows.length + 1).padStart(3, '0')}`};
      state.towerRows.push(tower);
      state.localPreview = result.local;
      if (!state.towerLocal3d) state.towerLocal3d = new window.TreeLineTowerViewer3D($('tower3dCanvas'), {mode: $('displayMode').value, pointSize: 2});
      const localRenderStart = performance.now();
      state.towerLocal3d.setData(result.local);
      state.towerLocal3d.setMode($('displayMode').value);
      await state.towerLocal3d.firstVisible;
      result.diagnostic_timings.transport_ms = timed.transport_ms;
      result.diagnostic_timings.js_parse_ms = timed.js_parse_ms;
      result.diagnostic_timings.local_3d_render_ms = performance.now() - localRenderStart;
      console.info('TOWER_LOCAL_TIMINGS', result.diagnostic_timings);
      renderTowerRows();
      $('towerLoad').textContent = `已选择 ${tower.display_name}：3D 可见点 ${world.x.toFixed(2)}, ${world.y.toFixed(2)}, ${world.z.toFixed(2)}；真实局部点云已确认`;
    } catch (error) {
      $('towerLoad').className = 'notice bad';
      $('towerLoad').textContent = userError(error, '杆塔局部点云加载失败，请缩放后重试。');
    } finally { $('towerLoad').dataset.loading = 'no'; }
  }

  async function loadTowerViewer() {
    if (!$('towerCanvas')) return;
    if (state.towerViewer && state.towerMain3d) { state.towerMain3d.setMarkers(state.towerRows); return; }
    try {
      $('towerLoad').textContent = '正在加载三维点云…';
      const timed = await timedJson('/api/desktop/tower-viewer/overview');
      const data = timed.payload;
      state.towerViewer = data;
      if (data.viewer_cache !== 'HIT') $('towerLoad').textContent = '正在生成显示缓存…';
      const initStart = performance.now();
      state.towerMain3d = new window.TreeLineTowerViewer3D($('towerCanvas'), {mode: 'height', pointSize: 2, onPick: acceptTowerPick});
      const init = state.towerMain3d.setData(data);
      state.towerMain3d.setMarkers(state.towerRows);
      await state.towerMain3d.firstVisible;
      const timings = Object.assign({}, data.diagnostic_timings, init, {
        transport_ms: timed.transport_ms,
        js_parse_ms: timed.js_parse_ms,
        first_visible_ms: performance.now() - timed.request_start,
        gpu_viewer_init_ms: performance.now() - initStart,
      });
      showTowerTimings(timings);
      $('towerLoad').textContent = `真实 3D 点云已就绪：显示 ${data.display_point_count.toLocaleString()} 个浏览点（源点 ${data.source_point_count.toLocaleString()} 个，${data.viewer_cache === 'HIT' ? '显示缓存命中' : '显示缓存已建立'}）`;
      bindTowerControls();
    } catch (error) {
      $('towerLoad').className = 'notice bad';
      $('towerLoad').textContent = userError(error, '点云显示加载失败，请重试。');
    }
  }

  function bindTowerControls() {
    if ($('displayMode').dataset.bound) return;
    $('displayMode').dataset.bound = 'yes';
    $('displayMode').addEventListener('change', event => {
      state.towerMain3d?.setMode(event.target.value); state.towerLocal3d?.setMode(event.target.value);
    });
    $('pointSize').addEventListener('input', event => state.towerMain3d?.setPointSize(event.target.value));
    $('perspectiveTowerView').addEventListener('click', () => state.towerMain3d?.perspectiveView());
    $('topTowerView').addEventListener('click', () => state.towerMain3d?.topView());
    $('sideTowerView').addEventListener('click', () => state.towerMain3d?.sideView());
    $('resetTowerView').addEventListener('click', () => state.towerMain3d?.resetView());
  }
  function selectedSpanIds() {
    return [...document.querySelectorAll('#spanChecks input[data-span]:checked')].map(input => input.dataset.span);
  }
  function allSpanIds() { return (state.batch?.spans || []).map(span => span.span_id).filter(Boolean); }
  async function saveTowers() {
    try {
      const towers = state.towerRows;
      await jsonPost('/api/desktop/towers', {line_name: $('lineName').value, line_role: $('lineRole').value, parent_line_id: $('parentLine').value || null, towers});
      await loadWorkspace(true);
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
      const execution_mode = selectedExecutionMode();
      const started = await jsonPost('/api/desktop/batch/start', {span_ids: spanIds, mode, execution_mode});
      renderBatch(started);
      state.uiState = started.status === 'RUNNING' ? 'ANALYSIS_RUNNING' : state.uiState;
      pinnedProcessingView=false;
      renderWorkspace();
      setStatus(mode === 'FIXTURE' ? '测试分析正在运行' : (started.source_cache_message || '离线分析正在运行'), 'good');
      window.setTimeout(pollBatch, 900);
    } catch (error) { setStatus(userError(error, '无法开始分析，请检查点云、单位和档距状态。'), 'bad'); }
  }
  async function startCurrentSpan(mode) {
    if (!state.activeSpanId) { setStatus('请先在档距总览中选择当前档距', 'warn'); return; }
    const boxes = document.querySelectorAll('#spanChecks input[data-span]');
    boxes.forEach(box => { box.checked = box.dataset.span === state.activeSpanId; });
    await startBatch(mode);
  }
  async function startAllSpans(mode) {
    document.querySelectorAll('#spanChecks input[data-span]:not(:disabled)').forEach(box => { box.checked = true; });
    await startBatch(mode);
  }
  async function pollBatch() {
    try {
      await loadWorkspace();
      if (state.batch.status === 'RUNNING' || state.batch.spans?.some(span => ['PREPARING', 'RUNNING'].includes(span.status))) window.setTimeout(pollBatch, 600);
      else setStatus(state.batch.status === 'FAILED' ? '分析未完成，请查看档距卡片中的失败原因' : '分析已完成，可以复核单木结果', state.batch.status === 'FAILED' ? 'bad' : 'good');
    } catch (error) { setStatus(userError(error, '暂时无法读取分析进度。'), 'bad'); }
  }
  async function retryBatch() {
    try { await jsonPost('/api/desktop/batch/retry', {execution_mode: selectedExecutionMode()}); await loadWorkspace(true); pollBatch(); }
    catch (error) { setStatus(userError(error, '重试失败任务未能启动。'), 'bad'); }
  }
  function loadViewer(force = false) {
    if (!state.treeReviewReady && !state.finalized) { setStatus('单木结果尚未就绪', 'warn'); return; }
    showProcessingWorkspace('ribbonReview',false);pinnedProcessingView=false;
    show('viewerWrap'); $('center').classList.add('review-mode');
    let reviewPageLoaded = false;
    try { reviewPageLoaded = new URL($('viewer').contentWindow.location.href, window.location.href).pathname === '/step2'; } catch (_) { reviewPageLoaded = false; }
    if (!state.viewerLoaded || force || !reviewPageLoaded) {
      const span = state.activeSpanId ? `&span_id=${encodeURIComponent(state.activeSpanId)}` : '';
      $('viewer').src = `/step2?run=${Date.now()}${span}`;
      state.viewerLoaded = true;
    }
    rememberRoute({kind:'viewer',url:new URL($('viewer').src,location.href).toString(),task:'ribbonReview',label:'单木人工复核'});
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
    try { await jsonPost('/api/desktop/batch/finalize'); await loadWorkspace(true); setStatus('单木复核已完成', 'good'); }
    catch (error) { setStatus(userError(error, '无法完成复核，请先检查单木结果。'), 'bad'); }
  }
  async function saveRisk() {
    const value = Number($('riskThreshold').value);
    if (!(value > 0)) { setStatus('预警净空阈值必须大于 0', 'warn'); return; }
    const reviewState = $('viewer').contentWindow?.TreeLineTreeReviewApp?.state;
    if (reviewState && (reviewState.editingBusy || reviewState.mutation || reviewState.trimMode !== 'IDLE' || reviewState.operationMode !== 'SELECT')) {
      setStatus('请先执行或取消当前树木编辑，再保存风险阈值。', 'warn'); return;
    }
    try {
      await jsonPost('/api/desktop/risk-policy', {warning_clearance_threshold_m: value});
      const review = $('viewer').contentWindow?.TreeLineTreeReviewApp;
      const active = !$('viewerWrap').classList.contains('hidden') && review?.refreshThreshold;
      await loadWorkspace();
      if (active) { loadViewer(); await review.refreshThreshold(); }
      else state.viewerLoaded = false;
      setStatus('风险阈值已保存，危险点与风险分类已按当前阈值更新', 'good');
    }
    catch (error) { setStatus(userError(error, '风险阈值保存失败。'), 'bad'); }
  }

  function installPaneResizers() {
    const workspace=$('workspace'),left=$('leftSplitter'),right=$('rightSplitter');
    const savedLeft=Number(localStorage.getItem('treeline-left-width')),savedRight=Number(localStorage.getItem('treeline-right-width'));
    if(Number.isFinite(savedLeft)&&savedLeft>=160)workspace.style.setProperty('--left-width',`${savedLeft}px`);
    if(Number.isFinite(savedRight)&&savedRight>=260)workspace.style.setProperty('--right-width',`${savedRight}px`);
    const drag=(splitter,side)=>splitter.addEventListener('pointerdown',event=>{
      if(event.target.closest('button'))return;event.preventDefault();splitter.setPointerCapture(event.pointerId);splitter.classList.add('dragging');
      const move=moveEvent=>{const rect=workspace.getBoundingClientRect(),width=side==='left'?moveEvent.clientX-rect.left:rect.right-moveEvent.clientX;const bounded=Math.round(Math.max(side==='left'?160:260,Math.min(side==='left'?520:620,width)));workspace.classList.remove(`${side}-collapsed`);workspace.style.setProperty(`--${side}-width`,`${bounded}px`);localStorage.setItem(`treeline-${side}-width`,String(bounded));};
      const stop=()=>{splitter.classList.remove('dragging');splitter.removeEventListener('pointermove',move);splitter.removeEventListener('pointerup',stop);splitter.removeEventListener('pointercancel',stop);};
      splitter.addEventListener('pointermove',move);splitter.addEventListener('pointerup',stop);splitter.addEventListener('pointercancel',stop);
    });
    drag(left,'left');drag(right,'right');
    $('collapseLeft').addEventListener('click',()=>workspace.classList.toggle('left-collapsed'));
    $('collapseRight').addEventListener('click',()=>workspace.classList.toggle('right-collapsed'));
  }
  installPaneResizers();
  $('navigateBack').addEventListener('click',navigateBack);
  window.addEventListener('message', event => {
    if (event.origin === location.origin && event.source === $('viewer').contentWindow
        && event.data?.type === 'treeline-exit-review') exitReview();
  });
  // The prediction frame uses a separate local origin; pywebview installs
  // its native API on this top-level window, not on the embedded frame.
  window.addEventListener('message',async event=>{
    if(!warningModuleUrl||event.source!==$('moduleFrame').contentWindow||event.origin!==new URL(warningModuleUrl).origin)return;
    const message=event.data,allowed=new Set(['save_export_url','save_export_file','save_table_export']);
    if(message?.type!=='treeline-native-export'||typeof message.id!=='string'||!allowed.has(message.method)||!Array.isArray(message.args))return;
    const reply={type:'treeline-native-export-result',id:message.id};
    try{const api=nativeApi();reply.available=typeof api?.[message.method]==='function';if(reply.available)reply.result=await api[message.method](...message.args);}
    catch(error){reply.error=String(error?.message||error);}
    event.source.postMessage(reply,event.origin);
  });
  window.addEventListener('message',event=>{
    if(!warningModuleUrl||event.source!==$('moduleFrame').contentWindow||event.origin!==new URL(warningModuleUrl).origin)return;
    if(event.data?.type==='treeline-open-processing'){
      const labels={preprocess:'点云预处理',ground:'地面点去除',wire:'导线重建',trees:'单木分割',processing:'点云分析'};
      if(event.data.workspace==='twin')openTwinWorkspace();else if(event.data.workspace==='review')loadViewer();else focusProcessingTool(labels[event.data.workspace]||'点云分析');
    }else if(event.data?.type==='treeline-navigation-state'&&event.data.token&&!$('moduleFrameWrap').classList.contains('hidden')&&!$('moduleFrame').classList.contains('hidden')){
      const tasks={data:'ribbonData',input:event.data.entry==='batch'?'ribbonExternalBatch':'openWarningModule',results:'exportWorkOrder',history:'ribbonHistory',training:'ribbonTraining'};
      const task=tasks[event.data.page]||'openWarningModule';setActiveTask(task);
      rememberRoute({kind:'warning',token:event.data.token,task,label:event.data.label||'预测任务'});
    }
  });

  $('newProject').addEventListener('click', () => { showProcessingWorkspace('newProject'); createProject(); });
  $('openWarningModule').addEventListener('click', openWarningModule);
  $('ribbonPreprocess').addEventListener('click', () => focusProcessingTool('点云预处理'));
  $('ribbonGround').addEventListener('click', () => focusProcessingTool('地面点去除'));
  $('ribbonWire').addEventListener('click', () => focusProcessingTool('导线重建'));
  $('ribbonTrees').addEventListener('click', () => focusProcessingTool('单木分割'));
  $('ribbonReview').addEventListener('click', () => loadViewer());
  $('ribbonData').addEventListener('click', () => openWarningWorkspace('data'));
  $('ribbonExternalBatch').addEventListener('click', () => openWarningWorkspace('batch'));
  $('ribbonTwin').addEventListener('click', openTwinWorkspace);
  $('ribbonTraining').addEventListener('click', () => openWarningWorkspace('training'));
  $('ribbonHistory').addEventListener('click', () => openWarningWorkspace('history'));
  $('moduleFrame').addEventListener('load', () => {
    state.warningFrameLoaded = $('moduleFrame').src !== 'about:blank';
    if (!state.warningFrameLoaded || !state.warningWorkspace || !warningModuleUrl) return;
    try {
      $('moduleFrame').contentWindow.postMessage(
        {type: 'treeline-workspace', workspace: state.warningWorkspace},
        new URL(warningModuleUrl).origin,
      );
    } catch (error) {
      console.warn('Unable to confirm warning workspace after frame load', error);
    }
  });
  $('openProject').addEventListener('click', () => { showProcessingWorkspace('openProject'); openProject(); });
  $('importPointCloud').addEventListener('click', () => { showProcessingWorkspace('importPointCloud'); importPointCloud(); });
  $('importGIS').addEventListener('click', () => { showProcessingWorkspace('importGIS'); importGIS(); });
  $('confirmUnit').addEventListener('click', confirmUnit);
  $('saveTowers').addEventListener('click', saveTowers);
  $('removeTower').addEventListener('click', () => { if (state.towerRows.length) { state.towerRows.pop(); renderTowerRows(); } });
  $('startFixture').addEventListener('click', () => startBatch('FIXTURE'));
  $('executionMode').addEventListener('change', event => {
    state.executionMode = normalizeExecutionMode(event.target.value);
    event.target.dataset.userSelected = 'yes';
  });
  $('startCurrent').addEventListener('click', () => startCurrentSpan('REAL'));
  $('startSelected').addEventListener('click', () => startBatch('REAL'));
  $('startAll').addEventListener('click', () => startAllSpans('REAL'));
  $('retryBatch').addEventListener('click', retryBatch);
  $('openViewer').addEventListener('click', () => loadViewer());
  $('openFuturePreview').addEventListener('click', openFuturePreview);
  $('refreshViewer').addEventListener('click', () => loadViewer(true));
  $('viewer').addEventListener('load',()=>{
    localizeViewerFrame();
    try{const url=$('viewer').contentWindow.location.href,path=new URL(url).pathname;if(!['/step2','/step3','/dbh-review'].includes(path))return;
      if(currentRoute?.kind==='viewer'&& !$('center').classList.contains('hidden'))rememberRoute({kind:'viewer',url,task:'ribbonReview',label:{'/step2':'单木人工复核','/step3':'当前最终报告','/dbh-review':'胸径复核'}[path]});
    }catch(_){/* Only same-origin processing pages participate in history. */}
  });
  $('auxiliaryFrame').addEventListener('load',()=>{
    try{
      const frame=$('auxiliaryFrame'),url=frame.contentWindow.location.href,path=new URL(url).pathname;
      if(currentRoute?.kind==='auxiliary'&&['/step3','/future-preview.html'].includes(path))rememberRoute({kind:'auxiliary',url,task:path==='/step3'?'exportWorkOrder':'ribbonTwin',label:path==='/step3'?'当前最终报告':'点云数字孪生'});
      frame.contentDocument?.querySelectorAll('a[href="/"]').forEach(link=>link.addEventListener('click',event=>{event.preventDefault();showProcessingWorkspace('',false);pinnedProcessingView=false;renderWorkspace();}));
    }catch(_){/* Keep external links outside the local history. */}
  });
  $('finalizeBatch').addEventListener('click', finalizeBatch);
  $('exportPrediction').addEventListener('click', openWarningModule);
  $('exportWorkOrder').addEventListener('click', () => openWarningWorkspace('report'));
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
    else if (action === 'open-current-report') {loadViewer();const url=new URL('/step3',location.href).toString();$('viewer').src=url;rememberRoute({kind:'viewer',url,task:'ribbonReview',label:'当前最终报告'});}
    else if (action === 'open-future-preview') openFuturePreview();
    else if (action === 'open-warning') openWarningModule();
  });
  window.addEventListener('error', event => {
    console.error('Unhandled UI error', event.error || event.message);
    setStatus('界面发生错误，详细信息已写入本地日志', 'bad');
  });
  window.addEventListener('unhandledrejection', event => {
    console.error('Unhandled promise rejection', event.reason);
    setStatus('操作未能完成，详细信息已写入本地日志', 'bad');
  });

  loadWorkspace().then(() => {
    const requested = new URLSearchParams(window.location.search).get('workspace');
    if (requested === 'review') loadViewer();
    else if (requested === 'preprocess') focusProcessingTool('点云预处理');
    else if (requested === 'ground') focusProcessingTool('地面点去除');
    else if (requested === 'wire') focusProcessingTool('导线重建');
    else if (requested === 'trees') focusProcessingTool('单木分割');
    else if (requested === 'analysis' || requested === 'processing') focusProcessingTool('点云分析');
  });
})();
