(function () {
  'use strict';

  const state = {
    treeMap: new Map(),
    selected: new Set(),
    operationMode: 'SELECT',
    editPicks: [],
    additiveKey: 'N',
    additiveHeld: false,
    reviewed: new Set(),
    activeTreeId: null,
    editorDisplay: null,
    viewer: null,
    refreshTimer: null,
    trimMode: 'IDLE',
    activePreview: null,
    lassoOverlay: null,
    editingBusy: false,
    refreshPaused: false,
    refreshInFlight: false,
    refreshGeneration: 0,
    mutation: null,
    displayRefreshUncertain: false,
    mutationRequestSequence: 0,
    committedJournalCount: 0,
    previewRequestId: 0,
    editorLoadPromise: null,
    redirectOnDone: false,
    viewExitRequested: false,
    workflowDone: false,
    showLabels: true,
    localPreviewXYZ: [],
    latestViewerData: null,
    viewerInitFailure: null,
    viewerRetryPromise: null,
    viewerRuntimeUnsubscribe: null,
    viewerState: {
      treeIdToMetadata: new Map(),
      activePreviewType: null,
      activePreviewTreeId: null,
      lassoMode: false,
    },
  };

  const metrics = window.__treeReviewMetrics = {
    renderer: 'THREE_JS_GIRO3D',
    backend_calls_during_camera: 0,
    tree_click_ms: [],
    lasso_display_preview_ms: [],
    lasso_exact_preview_ms: [],
    quality_events: [],
  };

  const reviewSpanId = new URLSearchParams(window.location.search || '').get('span_id');
  const byId = id => document.getElementById(id);

  function displayBudget() {
    const cores = Number(navigator.hardwareConcurrency || 4);
    const memory = Number(navigator.deviceMemory || 4);
    const runtime = window.TreeLineViewerRuntime;
    const capabilities = runtime && typeof runtime.getCapabilities === 'function'
      ? runtime.getCapabilities() : null;
    const webgl2 = Boolean(capabilities && capabilities.webgl2 && capabilities.webgl2.contextAvailable);
    if (!webgl2 || cores <= 4 || memory <= 4) return 12000;
    if (cores >= 12 && memory >= 12) return 50000;
    return 30000;
  }

  function setStatus(text) { byId('status').textContent = text; }

  function journalCountFrom(source) {
    if (!source || typeof source !== 'object') return null;
    if (Array.isArray(source.operations)) return source.operations.length;
    const count = Number(source.tree_edit_count);
    return Number.isFinite(count) && count >= 0 ? Math.floor(count) : null;
  }

  function updateUndoControl() {
    const button = byId('btn-undo');
    if (button) button.disabled = state.editingBusy || !(state.committedJournalCount > 0);
  }

  function updateJournalCount(source) {
    const count = journalCountFrom(source);
    if (count != null) state.committedJournalCount = count;
    updateUndoControl();
    return count;
  }

  function setEditingBusy(busy) {
    state.refreshGeneration += 1;
    state.editingBusy = Boolean(busy);
    if (!state.editingBusy) state.refreshPaused = false;
    ['btn-prev', 'btn-next', 'btn-delete', 'btn-merge', 'btn-split', 'btn-trim-component', 'btn-trim-lasso', 'btn-trim-cancel', 'btn-undo', 'btn-trim-confirm', 'btn-confirm', 'btn-dbh-review', 'btn-edit-execute', 'btn-selection-clear', 'multi-select-key'].forEach(id => {
      const element = byId(id);
      if (element) element.disabled = state.editingBusy;
    });
    updateUndoControl();
    updateOperationControls();
  }

  function showFailure(error, prefix = '失败：') {
    restoreCameraControls();
    if (error && error.viewerInitFailure) {
      showViewerDiagnostic(error);
      const editMessage = byId('editmsg');
      if (editMessage) editMessage.textContent = prefix + (error.message || '3D Viewer 初始化失败');
      return;
    }
    const message = error && error.message ? error.message : String(error);
    const editMessage = byId('editmsg');
    if (editMessage) editMessage.textContent = prefix + message;
    if (!state.viewer || !state.editorDisplay) {
      const rendererStatus = byId('viewer-renderer-status');
      if (rendererStatus) rendererStatus.textContent = error?.displayLoad
        ? '复核点云数据暂未加载，请查看具体原因或刷新视图。'
        : '3D复核视图初始化失败，请查看详情。';
    }
    const details = byId('error-details');
    const text = byId('error-detail-text');
    const technicalDetail = error && (error.details || error.stack || error.message);
    if (details && text && technicalDetail) {
      text.textContent = technicalDetail;
      details.style.display = 'block';
    } else if (details) details.style.display = 'none';
  }

  function viewerRuntime() {
    return window.TreeLineViewerRuntime || null;
  }

  function fallbackViewerDiagnostic(error) {
    const message = error && error.message ? String(error.message) : String(error || '');
    return {
      schema: 'treeline-viewer-diagnostic-v1',
      code: error && error.code ? String(error.code) : 'UNKNOWN_VIEWER_INIT_FAILURE',
      phase: 'initialization',
      message,
      original: {
        name: error && error.name ? String(error.name) : 'Error',
        message,
        stack: error && error.stack ? String(error.stack) : '',
      },
      capabilities: {},
      nativeRuntime: null,
      resources: [],
      attempts: [],
    };
  }

  function showViewerDiagnostic(error) {
    const runtime = viewerRuntime();
    const diagnostic = error && error.viewerDiagnostic
      ? error.viewerDiagnostic : (error && error.code ? error : fallbackViewerDiagnostic(error));
    state.viewerInitFailure = diagnostic;
    if (runtime && typeof runtime.showFailure === 'function') runtime.showFailure(diagnostic);
    const panel = byId('viewer-init-failure');
    const summary = byId('viewer-init-summary');
    const details = byId('viewer-init-details');
    if (panel) {
      panel.hidden = false;
      if (panel.style) panel.style.display = 'block';
    }
    if (summary) summary.textContent = '3D Viewer could not start on this device.';
    if (details) {
      details.readOnly = true;
      details.value = runtime && typeof runtime.diagnosticText === 'function'
        ? runtime.diagnosticText(diagnostic)
        : `3D Viewer could not start on this device.\nFailure code: ${diagnostic.code}\nOriginal exception: ${diagnostic.message}\n${diagnostic.original && diagnostic.original.stack || ''}`;
    }
    const status = byId('viewer-renderer-status');
    if (status) status.textContent = `3D Viewer 初始化失败 · ${diagnostic.code}`;
    bindViewerDiagnosticControls();
  }

  function clearViewerDiagnostic() {
    state.viewerInitFailure = null;
    const runtime = viewerRuntime();
    if (runtime && typeof runtime.clearFailure === 'function') runtime.clearFailure();
    const panel = byId('viewer-init-failure');
    if (panel) {
      panel.hidden = true;
      if (panel.style) panel.style.display = 'none';
    }
  }

  async function copyViewerDiagnostic() {
    const runtime = viewerRuntime();
    if (runtime && typeof runtime.copyDiagnostic === 'function') {
      await runtime.copyDiagnostic(state.viewerInitFailure);
      return;
    }
    const details = byId('viewer-init-details');
    if (!details) return;
    try {
      details.readOnly = true;
      if (typeof details.select === 'function') details.select();
      if (document.execCommand) document.execCommand('copy');
    } catch (_) {}
  }

  function bindViewerDiagnosticControls() {
    const retry = byId('btn-viewer-retry');
    if (retry && !retry.dataset) retry.dataset = {};
    if (retry && !retry.dataset.viewerDiagnosticBound) {
      retry.dataset.viewerDiagnosticBound = 'true';
      retry.onclick = () => { void retryViewer(); };
    }
    const copy = byId('btn-viewer-copy');
    if (copy && !copy.dataset) copy.dataset = {};
    if (copy && !copy.dataset.viewerDiagnosticBound) {
      copy.dataset.viewerDiagnosticBound = 'true';
      copy.onclick = () => { void copyViewerDiagnostic(); };
    }
  }

  const ASYNC_MUTATION_PATHS = new Set([
    '/api/trim/lasso-preview',
    '/api/trim/component-preview',
    '/api/trim/commit',
    '/api/delete',
    '/api/merge',
    '/api/split',
    '/api/undo',
  ]);
  const MUTATION_POLL_MS = 500;
  const MUTATION_RECOVERY_MS = 60000;
  const MUTATION_REQUEST_TIMEOUT_MS = 15000;

  function newRequestId() {
    const sequence = (++state.mutationRequestSequence).toString(36);
    if (window.crypto && typeof window.crypto.randomUUID === 'function') return `${window.crypto.randomUUID()}-${sequence}`;
    return `tree-edit-${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}-${sequence}`;
  }

  function wait(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  function mutationLabel(path) {
    if (path === '/api/trim/lasso-preview') return '正在按完整分辨率精确计算套索区域';
    if (path === '/api/trim/component-preview') return '正在分析当前单木的完整点成员';
    if (path === '/api/trim/commit') return '正在写入完整分辨率点成员';
    if (path === '/api/delete') return '正在应用删除操作';
    if (path === '/api/merge') return '正在应用合并操作';
    if (path === '/api/undo') return '正在应用撤销操作';
    return '正在处理编辑操作';
  }

  function setMutationProgress(mutation) {
    if (!mutation || state.mutation !== mutation) return;
    const elapsed = Math.max(0, (performance.now() - mutation.startedAt) / 1000);
    const elapsedText = elapsed < 1 ? `${Math.round(elapsed * 1000)} 毫秒` : `${elapsed.toFixed(1)} 秒`;
    const element = byId('editmsg');
    if (element) element.textContent = `${mutation.label}（已等待 ${elapsedText}，仍在处理，请稍候）`;
  }

  function networkError(path, cause, mutationPath = null) {
    const unknownCommit = mutationPath === '/api/trim/commit';
    const message = unknownCommit
      ? '网络连接中断，编辑提交状态暂时无法确认；正在查询状态，未自动重复提交。'
      : `网络连接中断，${mutationPath ? '正在查询编辑操作状态' : '本次请求'}；未自动重复提交，请稍后重试。`;
    const error = new Error(message);
    error.recoverable = true;
    error.network = true;
    error.details = cause && cause.message ? cause.message : `${path} 网络请求失败`;
    return error;
  }

  function mutationResponseError(path, response, result) {
    const source = result && Object.prototype.hasOwnProperty.call(result, 'error') ? result.error : result;
    const message = source && typeof source === 'object' ? source.message : source;
    const error = new Error(
      message || (result && result.status === 'EDIT_RUNNING' ? '已有编辑操作正在运行，请等待完成后再试。' : `请求失败 ${path}`)
    );
    error.statusCode = response && response.status;
    error.code = result && (result.status || result.code) || (source && source.code);
    error.details = source && typeof source === 'object' && source.details
      ? source.details
      : (result && result.details) || '';
    return error;
  }

  function responseParseError(path, cause, statusCode) {
    const error = new Error('服务器响应无法解析，操作结果未确认，请稍后重试。');
    error.recoverable = true;
    error.responseParse = true;
    error.statusCode = statusCode;
    error.details = cause && cause.message ? cause.message : `${path} 返回了无效响应`;
    return error;
  }

  function mutationRequestTimeout(path) {
    const error = new Error('网络请求超时，操作结果未确认，正在查询编辑状态。');
    error.recoverable = true;
    error.network = true;
    error.timeout = true;
    error.details = `${path} 请求超过 ${MUTATION_REQUEST_TIMEOUT_MS} 毫秒未完成`;
    return error;
  }

  async function fetchJsonWithDeadline(path, init = {}, timeoutMs = MUTATION_REQUEST_TIMEOUT_MS) {
    const controller = typeof AbortController === 'function' ? new AbortController() : null;
    const requestInit = {...init};
    if (controller) requestInit.signal = controller.signal;
    let timer = null;
    const request = (async () => {
      const response = await fetch(path, requestInit);
      let result;
      try {
        result = await response.json();
      } catch (cause) {
        throw responseParseError(path, cause, response && response.status);
      }
      return {response, result};
    })();
    const deadline = new Promise((resolve, reject) => {
      timer = setTimeout(() => {
        if (controller) {
          try { controller.abort(); } catch (_) {}
        }
        reject(mutationRequestTimeout(path));
      }, timeoutMs);
    });
    try {
      return await Promise.race([request, deadline]);
    } finally {
      if (timer != null) clearTimeout(timer);
    }
  }

  async function fetchMutationStatus(requestId) {
    const path = `/api/editor/mutation?request_id=${encodeURIComponent(requestId)}`;
    let outcome;
    try {
      outcome = await fetchJsonWithDeadline(path, {headers: {Accept: 'application/json'}, cache: 'no-store'});
    } catch (cause) {
      throw networkError(path, cause, '/api/editor/mutation');
    }
    const {response, result} = outcome;
    if (!response.ok) throw mutationResponseError(path, response, result);
    return result;
  }

  function belongsToMutation(result, requestId) {
    return Boolean(result && result.request_id != null && String(result.request_id) === String(requestId));
  }

  function mutationIsRunning(result) {
    return Boolean(result && (result.state === 'RUNNING' || result.status === 'EDIT_RUNNING'));
  }

  function mutationIsTerminal(result) {
    if (!result) return false;
    if (result.state === 'IDLE') return result.status === 'EDIT_PREVIEW_READY';
    return ['COMMITTED', 'FAILED', 'CANCELLED'].includes(result.state);
  }

  function terminalMutationResult(path, result) {
    if (result && (result.state === 'FAILED' || result.status === 'EDIT_FAILED' || result.error)) {
      throw mutationResponseError(path, {status: 500}, result);
    }
    if (result && (result.state === 'CANCELLED' || result.status === 'EDIT_CANCELLED')) {
      const error = new Error('编辑操作已取消，未应用本次修改。');
      error.code = result.status || result.state;
      error.details = result.details || '';
      throw error;
    }
    if (result && Object.prototype.hasOwnProperty.call(result, 'result')) return result.result;
    return result;
  }

  function unknownMutationError(path, lastError) {
    const unknownCommit = path === '/api/trim/commit';
    const error = new Error(
      unknownCommit
        ? '网络连接中断，编辑提交状态暂时无法确认；请刷新或重新打开复核页核对结果，核对前不要重复操作。'
        : '网络连接中断，编辑操作状态暂时无法确认；请刷新或重新打开复核页核对结果，核对前不要重复操作。'
    );
    error.recoverable = true;
    error.unknownMutation = true;
    error.details = lastError && (lastError.details || lastError.message) ? (lastError.details || lastError.message) : '';
    return error;
  }

  async function pollMutation(requestId, path, mutation) {
    let outageDeadline = Date.now() + MUTATION_RECOVERY_MS;
    let lastError = null;
    await wait(MUTATION_POLL_MS);
    while (true) {
      setMutationProgress(mutation);
      if (Date.now() >= outageDeadline) throw unknownMutationError(path, lastError);
      let result;
      try {
        result = await fetchMutationStatus(requestId);
      } catch (error) {
        lastError = error;
        await wait(MUTATION_POLL_MS);
        continue;
      }
      if (!belongsToMutation(result, requestId)) {
        lastError = new Error('编辑操作状态与当前请求不匹配。');
        await wait(MUTATION_POLL_MS);
        continue;
      }
      outageDeadline = Date.now() + MUTATION_RECOVERY_MS;
      lastError = null;
      if (mutationIsRunning(result)) {
        mutation.jobId = result.job_id || mutation.jobId || null;
        mutation.kind = result.kind || mutation.kind || null;
        await wait(MUTATION_POLL_MS);
        continue;
      }
      if (mutationIsTerminal(result)) return terminalMutationResult(path, result);
      if (result && (Object.prototype.hasOwnProperty.call(result, 'result') || result.preview_id || result.trees)) return result.result || result;
      await wait(MUTATION_POLL_MS);
    }
  }

  function clearMutation(requestId) {
    if (state.mutation && state.mutation.requestId === requestId) state.mutation = null;
  }

  function retainUnknownMutation(mutation, error) {
    mutation.uncertain = true;
    mutation.status = 'UNKNOWN';
    mutation.lastError = error;
  }

  async function post(path, payload) {
    const asyncMutation = ASYNC_MUTATION_PATHS.has(path);
    if (asyncMutation && state.mutation) {
      const error = new Error('已有编辑操作正在运行，请等待完成后再试。');
      error.code = 'EDIT_RUNNING';
      throw error;
    }
    const requestId = asyncMutation ? newRequestId() : null;
    const body = asyncMutation
      ? {...(payload || {}), async: true, request_id: requestId}
      : (payload || {});
    const mutation = asyncMutation ? {
      requestId,
      path,
      startedAt: performance.now(),
      label: mutationLabel(path),
      jobId: null,
      kind: null,
    } : null;
    if (mutation) {
      state.mutation = mutation;
      setMutationProgress(mutation);
    }
    let response;
    let result;
    try {
      const outcome = await fetchJsonWithDeadline(path, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(body),
      });
      response = outcome.response;
      result = outcome.result;
    } catch (cause) {
      if (!asyncMutation) {
        if (cause && cause.responseParse) throw cause;
        throw networkError(path, cause);
      }
      try {
        const recovered = await pollMutation(requestId, path, mutation);
        clearMutation(requestId);
        return recovered;
      } catch (error) {
        if (!error || !error.unknownMutation) clearMutation(requestId);
        else retainUnknownMutation(mutation, error);
        throw error;
      }
    }
    if (!response.ok) {
      clearMutation(requestId);
      throw mutationResponseError(path, response, result);
    }
    if (!asyncMutation) return result;
    try {
      if (mutationIsRunning(result) || response.status === 202) {
        mutation.jobId = result.job_id || null;
        mutation.kind = result.kind || null;
        const terminal = await pollMutation(requestId, path, mutation);
        clearMutation(requestId);
        return terminal;
      }
      const terminal = mutationIsTerminal(result) ? terminalMutationResult(path, result) : result;
      clearMutation(requestId);
      return terminal;
    } catch (error) {
      if (!error || !error.unknownMutation) clearMutation(requestId);
      else retainUnknownMutation(mutation, error);
      throw error;
    }
  }

  function scrollToTreeRow(treeId) {
    const row = document.querySelector(`#treetable tr[data-tid="${treeId}"]`);
    const list = byId('treelist');
    if (!row || !list) return;
    const target = Math.max(0, row.offsetTop - Math.max(0, (list.clientHeight - row.offsetHeight) / 2));
    if (typeof list.scrollTo === 'function') list.scrollTo({top: target, behavior: 'smooth'});
    else list.scrollTop = target;
  }

  function renderTrees(snapshot, scrollToActive) {
    updateJournalCount(snapshot);
    state.treeMap = new Map();
    state.viewerState.treeIdToMetadata.clear();
    const tbody = document.querySelector('#treetable tbody');
    tbody.innerHTML = '';
    const trees = Array.isArray(snapshot && snapshot.trees) ? snapshot.trees : [];
    const currentIds = new Set(trees.map(tree => Number(tree.tree_id)));
    state.selected = new Set([...state.selected].filter(id => currentIds.has(Number(id))));
    state.reviewed = new Set([...state.reviewed].filter(id => currentIds.has(Number(id))));
    if (state.activeTreeId != null && !currentIds.has(Number(state.activeTreeId))) state.activeTreeId = null;
    trees.forEach(tree => {
      const treeId = Number(tree.tree_id);
      state.treeMap.set(treeId, tree);
      state.viewerState.treeIdToMetadata.set(treeId, tree);
      const row = document.createElement('tr');
      row.dataset.tid = String(treeId);
      row.innerHTML = `<td>${treeId}</td><td>${tree.point_count}</td><td>${tree.apex_height_m != null ? Number(tree.apex_height_m).toFixed(1) : '—'}</td><td>${tree.merged ? '合并' : '—'}</td>`;
      row.onclick = event => selectTree(treeId, additiveSelection(event), true);
      tbody.appendChild(row);
    });
    byId('treecount').textContent = `共 ${trees.length} 棵`;
    applyTreeSelectionStyles();
    updateReviewStatus();
    if (scrollToActive && state.activeTreeId != null) setTimeout(() => scrollToTreeRow(state.activeTreeId), 0);
  }

  function updateReviewStatus() {
    const total = state.treeMap.size;
    const reviewedCount = [...state.reviewed].filter(id => state.treeMap.has(Number(id))).length;
    byId('review-status').textContent = `${total} 棵树 · 本次已查看 ${reviewedCount} · 本次待查看 ${Math.max(0, total - reviewedCount)}（项目定稿状态见主界面）`;
    byId('current-tree').textContent = state.activeTreeId == null ? '当前：未选择' : `当前：树 ${state.activeTreeId}`;
    const ids = [...state.selected].sort((a, b) => a - b);
    byId('selection-state').textContent = ids.length ? `已选择 ${ids.length} 棵树：${ids.join('、')}` : '未选择单木';
    byId('btn-delete').textContent = ids.length === 1 ? `删除树 ${ids[0]}` : '删除单木';
    byId('btn-merge').textContent = ids.length > 1 ? `合并选中（${ids.length} 棵）` : '合并选中';
    updateOperationControls();
  }

  function applyTreeSelectionStyles() {
    document.querySelectorAll('#treetable tr[data-tid]').forEach(row => {
      const treeId = Number(row.dataset.tid);
      row.className = state.selected.has(treeId) ? 'sel' : '';
    });
  }

  function selectionHighlightIds() {
    const ids = new Set(state.selected);
    return ids;
  }

  function updatePlotSelection() {
    if (state.viewer) state.viewer.setSelection(selectionHighlightIds());
  }

  function selectTree(treeId, toggle = false, scroll = false) {
    if (state.editingBusy) return;
    const started = performance.now();
    const id = Number(treeId);
    if (!state.treeMap.has(id)) return;
    if (state.activeTreeId !== id) state.previewRequestId += 1;
    if (state.activePreview && state.viewerState.activePreviewTreeId !== id) void cancelTrim('已切换单木，已取消待剔除预览。');
    if (state.activeTreeId != null && state.activeTreeId !== id) state.reviewed.add(Number(state.activeTreeId));
    state.activeTreeId = id;
    if (toggle) {
      if (state.selected.has(id)) state.selected.delete(id);
      else state.selected.add(id);
    } else state.selected = new Set([id]);
    if (!state.selected.has(id)) state.activeTreeId = [...state.selected].at(-1) ?? null;
    // Labels/list rows select entities; point picks below select split/component positions.
    if (state.editPicks.length && !state.editPicks.every(pick => state.selected.has(pick.tree_id))) clearEditPicks();
    applyTreeSelectionStyles();
    updateReviewStatus();
    updatePlotSelection();
    if (scroll) scrollToTreeRow(id);
    const elapsed = performance.now() - started;
    metrics.tree_click_ms.push(elapsed);
    byId('viewer-renderer-status').dataset.lastTreeClickMs = elapsed.toFixed(3);
  }

  function updatePreviewTrace(points, treeId) {
    state.localPreviewXYZ = Array.isArray(points) ? points : [];
    state.viewerState.activePreviewTreeId = state.localPreviewXYZ.length ? Number(treeId) : null;
    if (state.viewer) state.viewer.setPreview(state.localPreviewXYZ, treeId);
  }

  function pointInPolygon(point, polygon) {
    let inside = false;
    for (let index = 0, previous = polygon.length - 1; index < polygon.length; previous = index, index += 1) {
      const a = polygon[index];
      const b = polygon[previous];
      if (((a[1] > point[1]) !== (b[1] > point[1])) && point[0] < (b[0] - a[0]) * (point[1] - a[1]) / (b[1] - a[1]) + a[0]) inside = !inside;
    }
    return inside;
  }

  function polygonBounds(polygon) {
    let minX = Infinity;
    let maxX = -Infinity;
    let minY = Infinity;
    let maxY = -Infinity;
    polygon.forEach(point => {
      const x = Number(point[0]);
      const y = Number(point[1]);
      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
    });
    return {minX, maxX, minY, maxY};
  }

  function displayLassoPreview(treeId, matrix, polygon, scopeIds) {
    const started = performance.now();
    const cloud = (state.editorDisplay && state.editorDisplay.trees3d) || {};
    const xyz = Array.isArray(cloud.xyz) ? cloud.xyz : [];
    const ids = Array.isArray(cloud.tree_id_by_index) ? cloud.tree_id_by_index : [];
    const visible = Array.isArray(cloud.visible_by_index) ? cloud.visible_by_index : null;
    const region = Number(treeId) < 0;
    const scope = new Set(
      (scopeIds || (region ? [...state.treeMap.keys()] : [])).map(Number)
    );
    const bounds = polygonBounds(polygon);
    const preview = [];
    for (let index = 0; index < xyz.length; index += 1) {
      if (visible && !visible[index]) continue;
      const pointTreeId = Number(ids[index]);
      if (region ? !scope.has(pointTreeId) : pointTreeId !== Number(treeId)) continue;
      const point = xyz[index];
      const clip = [
        matrix[0][0] * point[0] + matrix[0][1] * point[1] + matrix[0][2] * point[2] + matrix[0][3],
        matrix[1][0] * point[0] + matrix[1][1] * point[1] + matrix[1][2] * point[2] + matrix[1][3],
        matrix[3][0] * point[0] + matrix[3][1] * point[1] + matrix[3][2] * point[2] + matrix[3][3],
      ];
      if (!Number.isFinite(clip[0]) || !Number.isFinite(clip[1]) || clip[2] <= Number.EPSILON) continue;
      const ndc = [clip[0] / clip[2], clip[1] / clip[2]];
      if (ndc[0] < bounds.minX || ndc[0] > bounds.maxX || ndc[1] < bounds.minY || ndc[1] > bounds.maxY) continue;
      if (pointInPolygon(ndc, polygon)) preview.push(point);
    }
    updatePreviewTrace(preview, treeId);
    const elapsed = performance.now() - started;
    metrics.lasso_display_preview_ms.push(elapsed);
    byId('viewer-renderer-status').dataset.lastLassoDisplayMs = elapsed.toFixed(3);
    return preview.length;
  }

  function setTrimMode(mode, message) {
    state.trimMode = mode;
    state.viewerState.lassoMode = mode === 'LASSO';
    byId('btn-trim-lasso').style.outline = mode === 'LASSO' ? '3px solid #d64545' : '';
    byId('editmsg').textContent = message || '';
    const indicator = byId('lasso-mode-indicator');
    indicator.style.display = mode === 'LASSO' ? 'block' : 'none';
    indicator.textContent = mode === 'LASSO' ? '区域套索：拖动鼠标圈选要删除的点云\n不限树木分类 · 跨对象生效 · Esc 取消' : '';
    updateOperationControls();
  }

  function additiveSelection(event) {
    return Boolean((event && (event.ctrlKey || event.metaKey)) || state.additiveHeld);
  }

  function clearEditPicks() {
    state.editPicks = [];
    if (state.viewer && state.viewer.setEditSeeds) state.viewer.setEditSeeds([]);
  }

  function updateOperationControls() {
    const modes = {'btn-merge': 'MERGE', 'btn-split': 'SPLIT', 'btn-trim-component': 'COMPONENT'};
    Object.entries(modes).forEach(([id, mode]) => {
      const button = byId(id);
      if (!button) return;
      const active = state.operationMode === mode;
      button.setAttribute('aria-pressed', String(active));
      button.style.outline = active ? '3px solid #25c6da' : '';
    });
    const labels = {SELECT: '请选择操作', MERGE: '执行合并', SPLIT: '执行分开', COMPONENT: '预览所选团块'};
    const button = byId('btn-edit-execute');
    if (button) {
      button.textContent = labels[state.operationMode];
      const ready = state.operationMode === 'MERGE' ? state.selected.size >= 2
        : state.operationMode === 'SPLIT' ? state.selected.size === 1 && [0, 2].includes(state.editPicks.length)
        : state.operationMode === 'COMPONENT' && state.editPicks.length > 0;
      button.disabled = !ready || state.editingBusy || Boolean(state.mutation) || Boolean(state.activePreview) || state.trimMode === 'LASSO';
    }
    const hint = byId('edit-operation-hint');
    const modifier = `Ctrl 或按住 ${state.additiveKey}`;
    if (hint) hint.textContent = {
      SELECT: `单击单选；${modifier}再点击可加选/取消。选好两个或更多编号，点击“合并选中”并确认即可合并。`,
      MERGE: `合并模式：已选 ${state.selected.size} 个编号；${modifier}加选。选好两个或更多编号后，再点“合并选中”、点击“执行合并”或按 Enter。`,
      SPLIT: `分开模式：在同一编号的粘连树中点第一个中心，再${modifier}点第二个中心（已选 ${state.editPicks.length}/2 个位置）；或只选择编号后自动分开。`,
      COMPONENT: `团块模式：直接点误分点团，再${modifier}加选其他点团（已选 ${state.editPicks.length} 个位置）；点击预览，检查红色区域后确认剔除。`,
    }[state.operationMode];
  }

  async function chooseOperationMode(mode, toggle = true) {
    if (state.editingBusy || state.mutation) return;
    const nextMode = toggle && state.operationMode === mode ? 'SELECT' : mode;
    if (state.activePreview || state.trimMode !== 'IDLE') await cancelTrim(undefined, true);
    clearEditPicks();
    state.operationMode = nextMode;
    updateOperationControls();
    byId('editmsg').textContent = '已切换操作模式，点云尚未改变。';
  }

  async function mergeSelection() {
    if (state.editingBusy || state.mutation) return;
    await chooseOperationMode('MERGE', false);
    if (state.selected.size < 2) {
      byId('editmsg').textContent = `请选两个或更多编号：单击第一个，再按 Ctrl 或按住 ${state.additiveKey} 点击其他编号，最后再点“合并选中”。`;
      return;
    }
    await executeOperation();
  }

  function handleEditPointPick(hit, event) {
    if (state.editingBusy || state.mutation || state.activePreview) return true;
    if (!['SPLIT', 'COMPONENT'].includes(state.operationMode)) return false;
    const id = Number(hit.treeId), xyz = hit.xyz.map(Number);
    if (!state.treeMap.has(id)) return true;
    const additive = additiveSelection(event);
    if (state.operationMode === 'SPLIT' && additive && state.editPicks.length && state.editPicks[0].tree_id !== id) {
      byId('editmsg').textContent = '分开需要点击同一编号里粘连的两个部分；不同编号请使用合并。';
      return true;
    }
    if (!additive) clearEditPicks();
    const existing = state.editPicks.findIndex(pick => pick.tree_id === id && Math.hypot(...pick.seed_xyz.map((v, i) => v - xyz[i])) < 0.05);
    if (existing >= 0) state.editPicks.splice(existing, 1);
    else {
      const limit = state.operationMode === 'SPLIT' ? 2 : 32;
      if (state.editPicks.length >= limit) {
        byId('editmsg').textContent = `最多选择 ${limit} 个位置，请清空选择后重选。`;
        return true;
      }
      state.editPicks.push({tree_id: id, seed_xyz: xyz});
    }
    state.selected = new Set(state.editPicks.map(pick => pick.tree_id));
    state.activeTreeId = state.selected.has(id) ? id : [...state.selected].at(-1) ?? null;
    state.viewer.setEditSeeds(state.editPicks.map(pick => pick.seed_xyz));
    applyTreeSelectionStyles(); updatePlotSelection(); updateReviewStatus();
    byId('editmsg').textContent = `已选 ${state.editPicks.length} 个位置，尚未修改点云。`;
    return true;
  }

  async function executeOperation() {
    const button = byId('btn-edit-execute');
    if (!button || button.disabled || state.editingBusy || state.mutation) return;
    if (state.operationMode === 'MERGE') {
      if (!window.confirm(`将 ${state.selected.size} 个选中编号合并为一棵。所有点保留，可撤销。继续吗？`)) return;
      await editAndReload('/api/merge', '已合并所选单木');
    } else if (state.operationMode === 'SPLIT') {
      if (!window.confirm(state.editPicks.length === 2
        ? '按所选两个中心分开当前粘连树。所有点保留，请检查结果，不正确可撤销。继续吗？'
        : '自动寻找两个中心分开当前粘连树。所有点保留，请检查结果，不正确可撤销。继续吗？')) return;
      await editAndReload('/api/split', '已分开粘连树，请检查两个新编号', state.editPicks.length === 2 ? {seed_xyz: state.editPicks.map(pick => pick.seed_xyz)} : {});
    } else if (state.operationMode === 'COMPONENT') {
      await previewComponent(null, null, state.editPicks.map(pick => ({...pick, seed_xyz: [...pick.seed_xyz]})));
    }
  }

  function requireOneTree() {
    if (state.editingBusy) return null;
    if (state.selected.size !== 1) {
      byId('editmsg').textContent = '请先选择一棵单木。';
      return null;
    }
    return [...state.selected][0];
  }

  function restoreCameraControls() {
    if (state.viewer) state.viewer.setControlsEnabled(true);
  }

  function removeLassoOverlay() {
    if (state.lassoOverlay) state.lassoOverlay.remove();
    state.lassoOverlay = null;
    restoreCameraControls();
  }

  async function showPreview(preview, keepLocalDisplayPreview) {
    state.activePreview = preview;
    state.viewerState.activePreviewType = state.trimMode;
    state.viewerState.activePreviewTreeId = Number(preview.tree_id);
    setTrimMode('PREVIEW', '');
    removeLassoOverlay();
    const percent = (Number(preview.fraction) * 100).toFixed(1);
    if (Number(preview.tree_id) < 0 || preview.method === 'REGION_LASSO') {
      const affected = Number(preview.affected_tree_count || (preview.tree_ids || []).length || 0);
      const fullyRemoved = Array.isArray(preview.fully_removed_tree_ids) ? preview.fully_removed_tree_ids.length : 0;
      const warning = preview.large_removal_warning ? ' 本次区域剔除点数较多，请确认红色区域是否正确。' : '';
      const fullText = fullyRemoved ? `，其中 ${fullyRemoved} 个对象将被完全移除` : '';
      byId('trim-summary').textContent = `待剔除 ${Number(preview.point_count).toLocaleString()} 个完整分辨率点，影响 ${affected} 个对象${fullText}；占当前可编辑点云 ${percent}%。不论树木还是误识别物体，圈中点都可删除；即使仅剩少量点，圈外点也保留。前后重叠点按屏幕投影一起选取，请检查红色预览。${warning}`;
    } else {
      const warning = preview.large_removal_warning ? ' 本次将移除该树较大比例的点，请确认选择是否正确。' : '';
      byId('trim-summary').textContent = `待剔除 ${Number(preview.point_count).toLocaleString()} 个点，占当前单木 ${percent}%。${warning}`;
    }
    byId('trim-preview').dataset.previewId = String(preview.preview_id || '');
    byId('trim-preview').style.display = 'block';
    if (!keepLocalDisplayPreview) updatePreviewTrace(preview.preview_xyz || [], preview.tree_id);
    byId('editmsg').textContent = '圈选已完成，但尚未删除。点击红色“确认删除圈内点”才会应用；取消不改变点云。';
    // In the desktop iframe the editor panel has its own scroll container.
    // Bring the commit button into that container, not below its visible edge.
    const panel = byId('trim-preview').closest('.side');
    if (panel) {
      const container = panel.getBoundingClientRect();
      const button = byId('btn-trim-confirm').getBoundingClientRect();
      if (button.bottom > container.bottom - 12) panel.scrollTop += button.bottom - container.bottom + 12;
      else if (button.top < container.top + 12) panel.scrollTop += button.top - container.top - 12;
    }
    requestAnimationFrame(() => {
      if (state.activePreview === preview) byId('btn-trim-confirm').scrollIntoView({block: 'nearest', inline: 'nearest'});
    });
  }

  async function previewComponent(treeId, seed, picks = null) {
    if (state.editingBusy || state.mutation) return;
    setEditingBusy(true);
    state.refreshPaused = true;
    try {
      byId('editmsg').textContent = '正在分析当前单木的完整点成员...';
      const requestId = ++state.previewRequestId;
      const preview = await post('/api/trim/component-preview', picks ? {picks} : {tree_id: treeId, seed_xyz: seed});
      if (requestId !== state.previewRequestId || (!picks && Number(state.activeTreeId) !== Number(treeId))) {
        await post('/api/trim/cancel', {preview_id: preview.preview_id});
        return;
      }
      await showPreview(preview, false);
    } catch (error) {
      setTrimMode('IDLE', '');
      showFailure(error, '');
    } finally {
      if (!state.mutation) setEditingBusy(false);
      restoreCameraControls();
    }
  }

  function beginLasso() {
    if (state.editingBusy || state.mutation) return;
    const visibleTreeIds = [...state.treeMap.keys()].map(Number);
    if (!visibleTreeIds.length || !state.viewer) {
      byId('editmsg').textContent = '当前视图没有可进行区域剔除的可编辑点云。';
      return;
    }
    const treeId = -1;
    state.operationMode = 'SELECT';
    clearEditPicks();
    removeLassoOverlay();
    setTrimMode('LASSO', '区域套索已开启：圈住不需要的点云，不限树木分类和已选编号。检查红色预览后确认删除；前后重叠点也会圈中，原始 LAS 不变，可撤销。');
    state.viewer.setControlsEnabled(false);
    const stage = byId('tree-review-stage');
    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.classList.add('lasso-overlay');
    svg.setAttribute('aria-label', '区域套索绘制');
    state.lassoOverlay = svg;
    const line = document.createElementNS(svg.namespaceURI, 'polyline');
    line.setAttribute('fill', 'rgba(255,0,0,.12)');
    line.setAttribute('stroke', '#ff3030');
    line.setAttribute('stroke-width', '2');
    line.setAttribute('pointer-events', 'none');
    svg.appendChild(line);
    stage.appendChild(svg);
    let points = [];
    let drawing = false;
    // offsetX/Y are relative to the event target. SVG painted children can
    // become that target, shifting the polygon as it crosses its own outline.
    const localPoint = event => {
      const rect = svg.getBoundingClientRect();
      return [event.clientX - rect.left, event.clientY - rect.top];
    };
    svg.onpointerdown = event => {
      if (event.button !== 0) return;
      event.preventDefault();
      drawing = true;
      points = [localPoint(event)];
      svg.setPointerCapture(event.pointerId);
    };
    svg.onpointermove = event => {
      if (!drawing) return;
      const next = localPoint(event);
      const last = points[points.length - 1];
      // Preserve every pointer vertex; only an exact consecutive duplicate is redundant.
      if (!last || next[0] !== last[0] || next[1] !== last[1]) {
        points.push(next);
        line.setAttribute('points', points.map(point => point.join(',')).join(' '));
      }
    };
    const abort = message => {
      drawing = false;
      removeLassoOverlay();
      setTrimMode('IDLE', message || '已取消区域套索，3D 旋转已恢复。');
    };
    svg.onpointercancel = () => abort('区域套索已取消，3D 旋转已恢复。');
    svg.onpointerup = async event => {
      if (!drawing) return;
      drawing = false;
      const finalPoint = localPoint(event);
      const last = points[points.length - 1];
      if (!last || finalPoint[0] !== last[0] || finalPoint[1] !== last[1]) {
        points.push(finalPoint);
        line.setAttribute('points', points.map(point => point.join(',')).join(' '));
      }
      try { svg.releasePointerCapture(event.pointerId); } catch (_) {}
      const width = Math.max(1, svg.clientWidth);
      const height = Math.max(1, svg.clientHeight);
      removeLassoOverlay();
      if (points.length < 3) {
        setTrimMode('IDLE', '套索至少需要三个顶点。');
        return;
      }
      const polygon = points.map(point => [2 * point[0] / width - 1, 1 - 2 * point[1] / height]);
      if (state.editingBusy || state.mutation) return;
      setEditingBusy(true);
      state.refreshPaused = true;
      const requestId = ++state.previewRequestId;
      try {
        const matrix = state.viewer.currentViewProjection();
        const displayCount = displayLassoPreview(treeId, matrix, polygon);
        setTrimMode('PREVIEW_LOADING', `显示样本已命中 ${displayCount.toLocaleString()} 个点并标红；正在按完整分辨率精确计算圈内可编辑点云...`);
        await post('/api/select', {tree_ids: visibleTreeIds});
        const exactStarted = performance.now();
        const preview = await post('/api/trim/lasso-preview', {tree_id: treeId, view_projection: matrix, polygon_ndc: polygon});
        const exactElapsed = performance.now() - exactStarted;
        metrics.lasso_exact_preview_ms.push(exactElapsed);
        byId('viewer-renderer-status').dataset.lastLassoExactMs = exactElapsed.toFixed(3);
        if (requestId !== state.previewRequestId) {
          await post('/api/trim/cancel', {preview_id: preview.preview_id});
          return;
        }
        await showPreview(preview, true);
      } catch (error) {
        updatePreviewTrace([], null);
        setTrimMode('IDLE', '');
        showFailure(error, '');
      } finally {
        restoreCameraControls();
        if (!state.mutation) setEditingBusy(false);
      }
    };
  }

  async function cancelTrim(message = '已取消，树木点成员未改变。', internal = false) {
    if (state.mutation) return;
    if (state.editingBusy && !internal) return;
    state.previewRequestId += 1;
    if (state.activePreview) {
      try { await post('/api/trim/cancel', {preview_id: state.activePreview.preview_id}); } catch (_) {}
    }
    state.activePreview = null;
    state.viewerState.activePreviewType = null;
    removeLassoOverlay();
    byId('trim-preview').style.display = 'none';
    delete byId('trim-preview').dataset.previewId;
    setTrimMode('IDLE', message);
    updatePreviewTrace([], null);
    if (!internal) {
      state.operationMode = 'SELECT';
      clearEditPicks();
      updateOperationControls();
    }
  }

  function initializeViewer() {
    if (state.viewer) return;
    const viewer = new window.TreeLineTreeReview3D(byId('plot3d'), byId('tree-label-overlay'), {
      onTreePick: (treeId, event) => selectTree(treeId, additiveSelection(event), true),
      onPointPick: (hit, event) => {
        if (handleEditPointPick(hit, event)) return true;
        if (state.editingBusy || state.mutation) return true;
        if (state.trimMode !== 'COMPONENT') return false;
        if (state.selected.size !== 1 || !state.selected.has(Number(hit.treeId))) {
          byId('editmsg').textContent = '请点击当前选中单木中的误分点团。';
          return true;
        }
        void previewComponent(Number(hit.treeId), hit.xyz.map(Number));
        return true;
      },
      onQualityChange: event => {
        metrics.quality_events.push(event);
        byId('viewer-renderer-status').textContent = `Giro3D / Three.js · 自适应 ${event.visible_points.toLocaleString()} 点 · P90 ${event.p90_ms.toFixed(1)} ms`;
      },
      onInitFailure: error => showViewerDiagnostic(error),
    });
    state.viewer = viewer;
    viewer.setLabelsVisible(state.showLabels);
    byId('btn-view-reset').onclick = () => { if (state.viewer) state.viewer.resetView(); };
    byId('btn-view-top').onclick = () => { if (state.viewer) state.viewer.topView(); };
    byId('btn-view-side').onclick = () => { if (state.viewer) state.viewer.sideView(); };
  }

  async function loadEditorData() {
    if (state.editorLoadPromise) return state.editorLoadPromise;
    const wasEditingBusy = state.editingBusy;
    let waitingForBackend = false;
    state.editorLoadPromise = (async () => {
      if (!state.editorDisplay) byId('editmsg').textContent = '正在加载单木列表与 Giro3D 预览...';
      const spanQuery = reviewSpanId ? `&span_id=${encodeURIComponent(reviewSpanId)}` : '';
      const path = `/api/editor_data?display_budget=${displayBudget()}${spanQuery}`;
      let outcome;
      const deadline = performance.now() + 120000;
      // Returning from a cold DBH evidence read can overlap its backend task.
      // Only retry this read-only GET on EDIT_RUNNING, never replay an edit,
      // finalize, or DBH preparation request. Other failures remain explicit.
      while (true) {
        if (waitingForBackend && performance.now() >= deadline) {
          const error = mutationResponseError(path, outcome.response, outcome.result);
          error.message = '后台任务仍未完成，暂时不能读取复核点云。请稍后点击“刷新视图”，无需重新分析。';
          error.displayLoad = true;
          throw error;
        }
        try {
          outcome = await fetchJsonWithDeadline(path, {headers: {Accept: 'application/json'}, cache: 'no-store'}, Math.max(1, deadline - performance.now()));
        } catch (cause) {
          if (cause && cause.responseParse) throw cause;
          throw networkError(path, cause);
        }
        const {response, result} = outcome;
        const busy = response.status === 409 && (result?.status === 'EDIT_RUNNING' || result?.error?.code === 'EDIT_RUNNING');
        if (!busy) break;
        if (!waitingForBackend) {
          waitingForBackend = true;
          setEditingBusy(true);
          state.refreshPaused = true;
        }
        const message = '正在等待后台完成胸径证据读取或编辑任务，完成后自动加载复核点云，请稍候…';
        byId('editmsg').textContent = message;
        byId('review-status').textContent = message;
        byId('viewer-renderer-status').textContent = '后台数据准备中，3D 点云将在完成后自动显示。';
        if (performance.now() >= deadline) {
          const error = mutationResponseError(path, response, result);
          error.message = '后台任务仍未完成，暂时不能读取复核点云。请稍后点击“刷新视图”，无需重新分析。';
          error.displayLoad = true;
          throw error;
        }
        await wait(Math.min(750, Math.max(1, deadline - performance.now())));
      }
      const {response, result: data} = outcome;
      if (!response.ok || data?.ok === false) {
        const error = mutationResponseError(path, response, data);
        error.displayLoad = true;
        throw error;
      }
      // Keep the latest display payload available for a Viewer-only retry. It
      // is visualization data already returned by the display endpoint; retry
      // never starts or repeats an editor/science operation.
      state.latestViewerData = data;
      const previousDisplay = state.editorDisplay;
      const restorePreviousViewer = () => {
        if (!previousDisplay || !state.viewer) return;
        try {
          state.viewer.setData(previousDisplay);
          state.viewer.setLabelsVisible(state.showLabels);
          updatePlotSelection();
        } catch (_) {}
      };
      try {
        initializeViewer();
        const timing = state.viewer.setData(data);
        state.viewer.setLabelsVisible(state.showLabels);
        const info = state.viewer.rendererInfo();
        updateJournalCount(data);
        renderTrees({
          trees: Array.isArray(data.trees) ? data.trees : [],
          operations: data.operations,
          tree_edit_count: data.tree_edit_count,
        }, false);
        updatePlotSelection();
        metrics.renderer_info = info;
        const dangerCount = Number(data.danger_points && data.danger_points.full_point_count || 0);
        const dangerText = dangerCount ? ` · 危险点 ${dangerCount.toLocaleString()}` : '';
        const locked = data.infrastructure;
        const fittedWireCount=Array.isArray(data.wires3d?.segments)?data.wires3d.segments.length:0;
        const lockedText = locked ? ` · 杆线锁定 ${Number(locked.protected_point_count || 0).toLocaleString()} 原始点 / ${Number(locked.pole_count || 0)} 根已拟合杆 / ${fittedWireCount} 根已拟合导线` : '';
        byId('viewer-renderer-status').textContent = `Giro3D / Three.js r${info.three_version} · WebGL${info.webgl_version} · ${timing.visible_tree_points.toLocaleString()} 树点${dangerText}${lockedText}`;
        state.editorDisplay = data;
        clearViewerDiagnostic();
        if (byId('error-details')) byId('error-details').style.display = 'none';
        if (byId('error-detail-text')) byId('error-detail-text').textContent = '';
        if (byId('editmsg').textContent.startsWith('正在加载')) byId('editmsg').textContent = '';
        if (waitingForBackend) byId('editmsg').textContent = '后台任务已完成，复核点云已自动加载。';
        if (locked && locked.status !== 'AVAILABLE') byId('editmsg').textContent = `杆线原始证据已锁定，但拟合未完整通过，暂不能定稿：${(locked.diagnostics || []).join('；')}`;
        return data;
      } catch (error) {
        restorePreviousViewer();
        throw error;
      }
    })();
    try { return await state.editorLoadPromise; }
    finally {
      state.editorLoadPromise = null;
      if (waitingForBackend && !wasEditingBusy && !state.mutation) setEditingBusy(false);
    }
  }

  async function retryViewer() {
    if (state.viewerRetryPromise) return state.viewerRetryPromise;
    if (state.mutation || state.editingBusy || state.displayRefreshUncertain) {
      const message = 'Viewer retry is unavailable while an edit is running or its result is uncertain. Finish or resolve the edit before retrying.';
      const editMessage = byId('editmsg');
      if (editMessage) editMessage.textContent = message;
      return state.viewer;
    }
    const retryButton = byId('btn-viewer-retry');
    if (retryButton) retryButton.disabled = true;
    state.viewerRetryPromise = (async () => {
      const runtime = viewerRuntime();
      if (runtime && typeof runtime.clearFailure === 'function') runtime.clearFailure();
      // A context-loss viewer may still be referenced by the editor state.
      // Dispose it before creating a fresh canvas/context; this leaves all
      // selection, lasso, mutation, and journal locks untouched.
      if (state.viewer) {
        try { if (typeof state.viewer.dispose === 'function') state.viewer.dispose(); } catch (_) {}
        state.viewer = null;
      }
      try {
        await loadEditorData();
        await refreshTrees();
        if (!state.workflowDone && !state.refreshTimer) state.refreshTimer = setInterval(refreshTrees, 1500);
        return state.viewer;
      } catch (error) {
        if (error && error.viewerInitFailure) showViewerDiagnostic(error);
        else showFailure(error);
        return null;
      } finally {
        if (retryButton) retryButton.disabled = false;
      }
    })();
    try { return await state.viewerRetryPromise; }
    finally { state.viewerRetryPromise = null; }
  }

  function committedDisplayRefreshError(error) {
    state.displayRefreshUncertain = true;
    state.refreshPaused = true;
    const wrapped = new Error('编辑已提交，但3D复核显示刷新失败；请刷新或重新打开复核页核对结果。');
    wrapped.recoverable = true;
    wrapped.committed = true;
    wrapped.details = error && (error.details || error.message) ? (error.details || error.message) : '';
    return wrapped;
  }

  async function refreshCommittedDisplay(snapshot) {
    // Retain the server-confirmed journal count even when the display fetch fails.
    updateJournalCount(snapshot);
    byId('editmsg').textContent = '编辑已提交，正在刷新3D复核显示...';
    try {
      const delta = snapshot && snapshot.display_delta;
      if (state.viewer && delta && delta.available === true
        && typeof state.viewer.applyDelta === 'function') {
        try {
          const timing = state.viewer.applyDelta(delta);
          const trees = Array.isArray(delta.trees) ? delta.trees : [];
          if (state.editorDisplay) {
            state.editorDisplay.trees = trees;
            state.editorDisplay.operations = Array.isArray(delta.operations) ? delta.operations : [];
            state.editorDisplay.editor_revision = Number(delta.revision);
          }
          renderTrees({trees, operations: delta.operations, tree_edit_count: delta.revision}, false);
          const info = state.viewer.rendererInfo();
          byId('viewer-renderer-status').textContent = `Giro3D / Three.js r${info.three_version} · 增量刷新 ${timing.delta_apply_ms.toFixed(1)} ms · ${timing.visible_tree_points.toLocaleString()} 树点`;
        } catch (_) {
          await loadEditorData();
        }
      } else {
        await loadEditorData();
      }
      updatePlotSelection();
    } catch (error) {
      throw committedDisplayRefreshError(error);
    }
  }

  async function refreshTrees() {
    if (state.refreshInFlight || state.refreshPaused || state.mutation) return;
    const refreshGeneration = state.refreshGeneration;
    state.refreshInFlight = true;
    try {
      const outcome = await fetchJsonWithDeadline('/state', {headers: {Accept: 'application/json'}, cache: 'no-store'});
      const {response, result: current} = outcome;
      if (!response.ok || refreshGeneration !== state.refreshGeneration || state.refreshPaused || state.mutation) return;
      if (refreshGeneration !== state.refreshGeneration || state.refreshPaused || state.mutation) return;
      updateJournalCount(current);
      setStatus(current.step_label + (current.current_message ? ` — ${current.current_message}` : ''));
      if (current.step === 'FINALIZING') byId('progress').textContent = current.current_message || current.step_label;
      if (current.error) {
        const error = new Error(current.error);
        error.details = current.error_details || '';
        showFailure(error);
        byId('progress').textContent = current.current_message || '复核结果生成失败';
        if (!state.mutation) setEditingBusy(false);
      }
      if (current.step === 'DONE') {
        state.workflowDone = true;
        clearInterval(state.refreshTimer);
        state.refreshTimer = null;
        if (state.redirectOnDone) {
          window.location.href = '/step3';
          return;
        }
        byId('progress').textContent = '复核结果已生成，可继续查看或编辑单木。';
        if (!state.mutation) setEditingBusy(false);
      }
      if (current.tree_snapshot && !reviewSpanId) renderTrees(current.tree_snapshot, false);
    } catch (_) {}
    finally { state.refreshInFlight = false; }
  }

  function stepTree(delta) {
    if (state.editingBusy || state.mutation || !state.treeMap.size || state.trimMode !== 'IDLE' || state.activePreview) return;
    const ids = [...state.treeMap.keys()].sort((a, b) => a - b);
    const index = state.activeTreeId == null ? (delta > 0 ? -1 : 0) : ids.indexOf(Number(state.activeTreeId));
    selectTree(ids[Math.max(0, Math.min(ids.length - 1, index + delta))], false, true);
  }

  async function editAndReload(path, successMessage, payload = {}) {
    if (state.editingBusy || state.mutation) return;
    setEditingBusy(true);
    state.refreshPaused = true;
    try {
      if (state.activePreview) await cancelTrim(undefined, true);
      await post('/api/select', {tree_ids: [...state.selected]});
      const snapshot = await post(path, payload);
      await refreshCommittedDisplay(snapshot);
      state.selected.clear();
      state.activeTreeId = null;
      clearEditPicks();
      applyTreeSelectionStyles(); updateReviewStatus();
      updatePlotSelection();
      byId('editmsg').textContent = successMessage;
    } catch (error) { showFailure(error); }
    finally {
      if (!state.mutation && !state.displayRefreshUncertain) setEditingBusy(false);
    }
  }

  function bindControls() {
    const reviewBack = byId('btn-review-back');
    if (reviewBack) reviewBack.onclick = async () => {
      // Leaving a view is not an undo/cancel operation. The server continues
      // a submitted edit even if loading or the WebGL display has failed.
      state.viewExitRequested = true;
      state.redirectOnDone = false;
      if (!state.editingBusy && !state.mutation && (state.activePreview || state.trimMode !== 'IDLE')) {
        try { await cancelTrim(undefined, true); } catch (_) { /* An ephemeral preview is not a committed edit. */ }
      }
      if (window.parent !== window) window.parent.postMessage({type:'treeline-exit-review'}, location.origin);
      else window.location.href = '/';
    };
    window.addEventListener('message', event => {
      if (event.origin === location.origin && event.source === window.parent
          && event.data?.type === 'treeline-review-hidden') {
        state.viewExitRequested = true;
        state.redirectOnDone = false;
      }
    });
    const required = ['btn-prev', 'btn-next', 'btn-delete', 'btn-merge', 'btn-split',
      'btn-trim-component', 'btn-trim-lasso', 'btn-trim-cancel', 'btn-trim-confirm',
      'btn-undo', 'btn-confirm', 'toggle-labels', 'btn-edit-execute', 'btn-selection-clear', 'multi-select-key'];
    const missing = required.filter(id => !byId(id));
    if (missing.length) throw new Error(`复核页面控件不完整：${missing.join('、')}。请关闭软件并重新打开最新版本。`);
    const dbhReview = byId('btn-dbh-review');
    if (dbhReview) dbhReview.onclick = () => {
      if (state.editingBusy || state.mutation || state.displayRefreshUncertain) return;
      window.location.href = '/dbh-review';
    };
    byId('btn-trim-component').onclick = () => void chooseOperationMode('COMPONENT');
    byId('btn-edit-execute').onclick = () => void executeOperation();
    byId('btn-selection-clear').onclick = async () => {
      if (state.editingBusy || state.mutation) return;
      if (state.activePreview || state.trimMode !== 'IDLE') await cancelTrim(undefined, true);
      clearEditPicks(); state.selected.clear(); state.activeTreeId = null;
      applyTreeSelectionStyles(); updatePlotSelection(); updateReviewStatus();
      byId('editmsg').textContent = '已清空选择，点云没有改变。';
    };
    const keyControl = byId('multi-select-key');
    for (let code = 65; code <= 90; code += 1) {
      const option = document.createElement('option');
      option.value = option.textContent = String.fromCharCode(code);
      keyControl.appendChild(option);
    }
    try { const saved = localStorage.getItem('treeline-review-additive-key'); if (/^[A-Z]$/.test(saved || '')) state.additiveKey = saved; } catch (_) {}
    keyControl.value = state.additiveKey;
    keyControl.onchange = () => {
      state.additiveKey = keyControl.value;
      state.additiveHeld = false;
      try { localStorage.setItem('treeline-review-additive-key', state.additiveKey); } catch (_) {}
      updateOperationControls();
      keyControl.blur();
    };
    byId('btn-trim-lasso').textContent = '区域套索剔除';
    byId('btn-trim-lasso').title = '圈内可编辑点均可删除（含误识别物体，不限已选编号），先预览再确认，支持撤销；原始 LAS 不变';
    byId('btn-trim-lasso').onclick = beginLasso;
    byId('btn-trim-confirm').textContent = '确认删除圈内点';
    byId('btn-trim-cancel').onclick = () => void cancelTrim();
    byId('btn-trim-confirm').onclick = async () => {
      if (state.editingBusy || state.mutation || !state.activePreview) return;
      setEditingBusy(true);
      state.refreshPaused = true;
      try {
        byId('editmsg').textContent = '正在写入完整分辨率点成员...';
        const snapshot = await post('/api/trim/commit', {preview_id: state.activePreview.preview_id});
        state.activePreview = null;
        state.viewerState.activePreviewType = null;
        updatePreviewTrace([], null);
        byId('trim-preview').style.display = 'none';
        delete byId('trim-preview').dataset.previewId;
        setTrimMode('IDLE', '已删除圈内点云，圈外点保留；可用撤销精确恢复整个区域操作，原始 LAS 未修改。');
        await refreshCommittedDisplay(snapshot);
        state.selected.clear();
        state.activeTreeId = null;
        clearEditPicks();
        updateReviewStatus(); applyTreeSelectionStyles();
        updatePlotSelection();
        byId('editmsg').textContent = '已删除圈内点云，圈外点保留；可用撤销精确恢复整个区域操作，原始 LAS 未修改。';
      } catch (error) { showFailure(error); }
      finally {
        if (!state.mutation && !state.displayRefreshUncertain) setEditingBusy(false);
        restoreCameraControls();
      }
    };
    byId('toggle-labels').onchange = event => {
      state.showLabels = Boolean(event.target.checked);
      if (state.viewer) state.viewer.setLabelsVisible(state.showLabels);
    };
    byId('btn-prev').onclick = () => stepTree(-1);
    byId('btn-next').onclick = () => stepTree(1);
    byId('btn-delete').onclick = () => void editAndReload('/api/delete', '已删除所选单木');
    byId('btn-merge').onclick = () => void mergeSelection();
    byId('btn-split').onclick = () => void chooseOperationMode('SPLIT');
    byId('btn-undo').onclick = async () => {
      if (state.editingBusy || state.mutation) return;
      if (!(state.committedJournalCount > 0)) {
        byId('editmsg').textContent = '当前没有可撤销的已提交编辑。';
        updateUndoControl();
        return;
      }
      const journalCountBefore = state.committedJournalCount;
      setEditingBusy(true);
      state.refreshPaused = true;
      try {
        if (state.activePreview) await cancelTrim(undefined, true);
        const snapshot = await post('/api/undo', {});
        await refreshCommittedDisplay(snapshot);
        state.selected.clear();
        state.activeTreeId = null;
        clearEditPicks();
        updateReviewStatus(); applyTreeSelectionStyles();
        updatePlotSelection();
        const journalCountAfter = journalCountFrom(snapshot);
        byId('editmsg').textContent = journalCountAfter != null && journalCountAfter >= journalCountBefore
          ? '未发现可撤销的已提交编辑。'
          : '已撤销上一步';
      } catch (error) { showFailure(error); }
      finally {
        if (!state.mutation && !state.displayRefreshUncertain) setEditingBusy(false);
      }
    };
    byId('btn-confirm').onclick = async () => {
      if (state.editingBusy || state.mutation) return;
      setEditingBusy(true);
      state.viewExitRequested = false;
      if (state.activeTreeId != null) state.reviewed.add(Number(state.activeTreeId));
      updateReviewStatus();
      const pending = Math.max(0, state.treeMap.size - state.reviewed.size);
      byId('editmsg').textContent = pending ? `仍有 ${pending} 棵未逐棵查看，将按现有批量确认规则完成复核。` : '';
      byId('progress').textContent = '正在应用人工修改';
      try {
        await post('/api/confirm_trees', {});
        state.redirectOnDone = !state.viewExitRequested;
      }
      catch (error) {
        state.redirectOnDone = false;
        showFailure(error);
        if (!state.mutation) setEditingBusy(false);
        return;
      }
      clearInterval(state.refreshTimer);
      state.refreshTimer = setInterval(refreshTrees, 700);
      void refreshTrees();
    };
    document.addEventListener('keydown', event => {
      const editable = /INPUT|TEXTAREA|SELECT/.test((event.target && event.target.tagName) || '') || event.target?.isContentEditable;
      if (editable) return;
      if (event.code === `Key${state.additiveKey}` && !event.altKey && !event.ctrlKey && !event.metaKey) {
        state.additiveHeld = true; event.preventDefault(); return;
      }
      if (!state.editingBusy && !state.mutation && event.key === 'Escape' && (state.operationMode !== 'SELECT' || state.trimMode !== 'IDLE' || state.activePreview || state.editPicks.length)) {
        event.preventDefault();
        void cancelTrim();
        return;
      }
      if (event.key === 'Enter' && !event.repeat && (!/BUTTON|A/.test(event.target?.tagName || '') || ['btn-merge', 'btn-split', 'btn-trim-component'].includes(event.target?.id))) {
        event.preventDefault(); void executeOperation(); return;
      }
      if ((event.key === 'ArrowLeft' || event.key === 'ArrowRight') && state.trimMode === 'IDLE' && !state.activePreview && !/INPUT|TEXTAREA|SELECT/.test((event.target && event.target.tagName) || '')) {
        event.preventDefault();
        stepTree(event.key === 'ArrowLeft' ? -1 : 1);
      }
    });
    document.addEventListener('keyup', event => { if (event.code === `Key${state.additiveKey}`) state.additiveHeld = false; });
    window.addEventListener('blur', () => { state.additiveHeld = false; });
    document.addEventListener('visibilitychange', () => { if (document.hidden) state.additiveHeld = false; });
    window.addEventListener('error', restoreCameraControls);
    window.addEventListener('unhandledrejection', restoreCameraControls);
    updateUndoControl();
    updateOperationControls();
  }

  async function start() {
    try {
      bindControls();
      await loadEditorData();
      await refreshTrees();
      if (!state.workflowDone) state.refreshTimer = setInterval(refreshTrees, 1500);
    } catch (error) {
      showFailure(error);
      const reviewStatus = byId('review-status');
      if (reviewStatus) reviewStatus.textContent = '复核加载失败，请查看下方详情或刷新视图。';
    }
  }

  window.TreeLineTreeReviewApp = {start, state, selectTree, beginLasso, cancelTrim, metrics};
  window.TreeLineTreeReviewApp.retryViewer = retryViewer;
  window.TreeLineTreeReviewApp.refreshThreshold = async function () {
    if (state.editingBusy || state.mutation || state.trimMode !== 'IDLE') throw new Error('请先执行或取消当前编辑，再刷新危险点阈值。');
    const camera = state.viewer?.camera;
    const position = camera?.position.clone(), target = state.viewer?.controls.target.clone();
    const quaternion = camera?.quaternion.clone();
    await loadEditorData();
    await refreshTrees();
    if (position && target && state.viewer?.camera === camera) {
      camera.position.copy(position); camera.quaternion.copy(quaternion);
      state.viewer.controls.target.copy(target); state.viewer.controls.update();
      state.viewer.requestRender(); state.viewer.renderLabels();
    }
  };
  window.TreeLineTreeReviewApp.showViewerDiagnostic = showViewerDiagnostic;
  window.TreeLineTreeReviewApp.clearViewerDiagnostic = clearViewerDiagnostic;
  installViewerRuntimeDiagnostics();

  function installViewerRuntimeDiagnostics() {
    bindViewerDiagnosticControls();
    const runtime = viewerRuntime();
    if (!runtime || typeof runtime.subscribe !== 'function') return;
    if (state.viewerRuntimeUnsubscribe) state.viewerRuntimeUnsubscribe();
    state.viewerRuntimeUnsubscribe = runtime.subscribe(error => showViewerDiagnostic(error));
    if (typeof runtime.getLastFailure === 'function') {
      const diagnostic = runtime.getLastFailure();
      if (diagnostic) showViewerDiagnostic(diagnostic);
    }
  }
})();
