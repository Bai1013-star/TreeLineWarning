(function () {
  'use strict';

  const state = {
    treeMap: new Map(),
    selected: new Set(),
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
    ['btn-prev', 'btn-next', 'btn-delete', 'btn-merge', 'btn-trim-component', 'btn-trim-lasso', 'btn-trim-cancel', 'btn-undo', 'btn-trim-confirm', 'btn-confirm', 'btn-dbh-review'].forEach(id => {
      const element = byId(id);
      if (element) element.disabled = state.editingBusy;
    });
    updateUndoControl();
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
    if (error && error.dataLoad) {
      const rendererStatus = byId('viewer-renderer-status');
      if (rendererStatus) rendererStatus.textContent = '单木复核数据加载失败，请刷新视图重试。';
    } else if (!state.viewer || !state.editorDisplay) {
      const rendererStatus = byId('viewer-renderer-status');
      if (rendererStatus) rendererStatus.textContent = '3D复核视图初始化失败，请查看详情。';
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
    '/api/undo',
  ]);
  const MUTATION_POLL_MS = 500;
  const MUTATION_RECOVERY_MS = 60000;
  const MUTATION_REQUEST_TIMEOUT_MS = 15000;
  // A first read after reopening a real project may need to hydrate the
  // persisted final membership and build the bounded 3D display cache.  That
  // is a read-only operation and has legitimately taken about 30 seconds on a
  // 105-million-point survey.  Do not abort it with the edit-mutation deadline.
  const DISPLAY_REQUEST_TIMEOUT_MS = 180000;
  const DISPLAY_SLOW_NOTICE_MS = 15000;

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
    error.code = result && (result.status || result.code);
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

  function requestTimeout(path, timeoutMs) {
    const displayRead = path.startsWith('/api/editor_data') || path.startsWith('/api/desktop/review');
    const error = new Error(displayRead
      ? '单木复核数据准备超时，请刷新复核视图重试。'
      : '网络请求超时，操作结果未确认，正在查询编辑状态。');
    error.recoverable = true;
    error.network = true;
    error.timeout = true;
    error.dataLoad = displayRead;
    error.details = `${path} 请求超过 ${timeoutMs} 毫秒未完成`;
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
        reject(requestTimeout(path, timeoutMs));
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
      row.onclick = () => selectTree(treeId, true, true);
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
    byId('review-status').textContent = `${total} 棵树 · 已复核 ${reviewedCount} · 待复核 ${Math.max(0, total - reviewedCount)}`;
    byId('current-tree').textContent = state.activeTreeId == null ? '当前：未选择' : `当前：树 ${state.activeTreeId}`;
    const ids = [...state.selected].sort((a, b) => a - b);
    byId('selection-state').textContent = ids.length ? `已选择 ${ids.length} 棵树：${ids.join('、')}` : '未选择单木';
    byId('btn-delete').textContent = ids.length === 1 ? `删除树 ${ids[0]}` : '删除单木';
    byId('btn-merge').textContent = ids.length > 1 ? `合并选中（${ids.length} 棵）` : '合并选中';
  }

  function applyTreeSelectionStyles() {
    document.querySelectorAll('#treetable tr[data-tid]').forEach(row => {
      const treeId = Number(row.dataset.tid);
      row.className = state.selected.has(treeId) || state.activeTreeId === treeId ? 'sel' : '';
    });
  }

  function selectionHighlightIds() {
    const ids = new Set(state.selected);
    if (state.activeTreeId != null) ids.add(Number(state.activeTreeId));
    return ids;
  }

  function updatePlotSelection() {
    if (state.viewer) state.viewer.setSelection(selectionHighlightIds());
  }

  function selectTree(treeId, toggle = true, scroll = false) {
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
    const region = Number(treeId) < 0;
    const scope = new Set(
      (scopeIds || (region ? [...state.treeMap.keys()] : [])).map(Number)
    );
    const bounds = polygonBounds(polygon);
    const preview = [];
    for (let index = 0; index < xyz.length; index += 1) {
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
    byId('btn-trim-component').style.outline = mode === 'COMPONENT' ? '3px solid #d64545' : '';
    byId('btn-trim-lasso').style.outline = mode === 'LASSO' ? '3px solid #d64545' : '';
    byId('editmsg').textContent = message || '';
    const indicator = byId('lasso-mode-indicator');
    indicator.style.display = mode === 'LASSO' ? 'block' : 'none';
    indicator.textContent = mode === 'LASSO' ? '区域套索：拖动鼠标圈选要剔除的树木点\n框内跨多棵树生效 · Esc 取消' : '';
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
      const fullText = fullyRemoved ? `，其中 ${fullyRemoved} 棵树将被完全移除` : '';
      byId('trim-summary').textContent = `待剔除 ${Number(preview.point_count).toLocaleString()} 个完整分辨率树木点，影响 ${affected} 棵树${fullText}；占当前可见范围树点 ${percent}%。${warning}`;
    } else {
      const warning = preview.large_removal_warning ? ' 本次将移除该树较大比例的点，请确认选择是否正确。' : '';
      byId('trim-summary').textContent = `待剔除 ${Number(preview.point_count).toLocaleString()} 个点，占当前单木 ${percent}%。${warning}`;
    }
    byId('trim-preview').dataset.previewId = String(preview.preview_id || '');
    byId('trim-preview').style.display = 'block';
    if (!keepLocalDisplayPreview) updatePreviewTrace(preview.preview_xyz || [], preview.tree_id);
  }

  async function previewComponent(treeId, seed) {
    if (state.editingBusy || state.mutation) return;
    setEditingBusy(true);
    state.refreshPaused = true;
    try {
      byId('editmsg').textContent = '正在分析当前单木的完整点成员...';
      const requestId = ++state.previewRequestId;
      const preview = await post('/api/trim/component-preview', {tree_id: treeId, seed_xyz: seed});
      if (requestId !== state.previewRequestId || Number(state.activeTreeId) !== Number(treeId)) {
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
      byId('editmsg').textContent = '当前视图没有可进行区域剔除的树木。';
      return;
    }
    const treeId = -1;
    removeLassoOverlay();
    setTrimMode('LASSO', '区域套索已开启：直接框住不需要的树木区域；框内所有树木点都会被剔除。');
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
    svg.appendChild(line);
    stage.appendChild(svg);
    let points = [];
    let drawing = false;
    svg.onpointerdown = event => {
      drawing = true;
      points = [[event.offsetX, event.offsetY]];
      svg.setPointerCapture(event.pointerId);
    };
    svg.onpointermove = event => {
      if (!drawing) return;
      const next = [event.offsetX, event.offsetY];
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
      const finalPoint = [event.offsetX, event.offsetY];
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
        setTrimMode('PREVIEW_LOADING', `显示样本已命中 ${displayCount.toLocaleString()} 个树木点并标红；正在按完整分辨率精确计算框内所有树木点...`);
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
  }

  function initializeViewer() {
    if (state.viewer) return;
    const viewer = new window.TreeLineTreeReview3D(byId('plot3d'), byId('tree-label-overlay'), {
      onTreePick: treeId => selectTree(treeId, true, true),
      onPointPick: hit => {
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
    state.editorLoadPromise = (async () => {
      if (!state.editorDisplay) byId('editmsg').textContent = '正在加载单木列表与 Giro3D 预览...';
      const spanQuery = reviewSpanId ? `&span_id=${encodeURIComponent(reviewSpanId)}` : '';
      const path = `/api/editor_data?display_budget=${displayBudget()}${spanQuery}`;
      let outcome;
      const slowNotice = setTimeout(() => {
        if (state.editorDisplay) return;
        const message = byId('editmsg');
        if (message) message.textContent = '正在准备真实点云复核数据，首次加载可能需要几十秒，请稍候…';
      }, DISPLAY_SLOW_NOTICE_MS);
      try {
        outcome = await fetchJsonWithDeadline(
          path,
          {headers: {Accept: 'application/json'}, cache: 'no-store'},
          DISPLAY_REQUEST_TIMEOUT_MS,
        );
      } catch (cause) {
        if (cause && cause.responseParse) throw cause;
        const error = cause && cause.timeout ? cause : networkError(path, cause);
        error.dataLoad = true;
        throw error;
      } finally { clearTimeout(slowNotice); }
      const {response, result: data} = outcome;
      if (!response.ok) {
        const error = new Error(
          data && data.error && data.error.message
            ? data.error.message : '单木显示数据加载失败'
        );
        error.dataLoad = true;
        error.statusCode = response.status;
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
        byId('viewer-renderer-status').textContent = `Giro3D / Three.js r${info.three_version} · WebGL${info.webgl_version} · ${timing.visible_tree_points.toLocaleString()} 树点${dangerText}`;
        state.editorDisplay = data;
        clearViewerDiagnostic();
        if (byId('editmsg').textContent.startsWith('正在加载')) byId('editmsg').textContent = '';
        return data;
      } catch (error) {
        restorePreviousViewer();
        throw error;
      }
    })();
    try { return await state.editorLoadPromise; }
    finally { state.editorLoadPromise = null; }
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
    state.workflowDone = false;
    state.redirectOnDone = false;
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

  async function editAndReload(path, successMessage) {
    if (state.editingBusy || state.mutation) return;
    setEditingBusy(true);
    state.refreshPaused = true;
    try {
      if (state.activePreview) await cancelTrim(undefined, true);
      await post('/api/select', {tree_ids: [...state.selected]});
      const snapshot = await post(path, {});
      await refreshCommittedDisplay(snapshot);
      state.selected.clear();
      updatePlotSelection();
      byId('editmsg').textContent = successMessage;
    } catch (error) { showFailure(error); }
    finally {
      if (!state.mutation && !state.displayRefreshUncertain) setEditingBusy(false);
    }
  }

  function bindControls() {
    const dbhReview = byId('btn-dbh-review');
    if (dbhReview) dbhReview.onclick = () => {
      if (state.editingBusy || state.mutation || state.displayRefreshUncertain) return;
      window.location.href = '/dbh-review';
    };
    byId('btn-trim-component').onclick = () => { if (requireOneTree() != null) setTrimMode('COMPONENT', '请点击需要从当前单木中剔除的误分点团。'); };
    byId('btn-trim-lasso').textContent = '区域套索剔除';
    byId('btn-trim-lasso').title = '框选区域后，剔除框内所有树木点（跨多棵树）';
    byId('btn-trim-lasso').onclick = beginLasso;
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
        setTrimMode('IDLE', '已剔除框内树木点，可使用撤销精确恢复整个区域操作。');
        await refreshCommittedDisplay(snapshot);
        state.selected.clear();
        updatePlotSelection();
        byId('editmsg').textContent = '已剔除框内树木点，可使用撤销精确恢复整个区域操作。';
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
    byId('btn-merge').onclick = () => void editAndReload('/api/merge', '已合并所选单木');
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
      if (state.activeTreeId != null) state.reviewed.add(Number(state.activeTreeId));
      updateReviewStatus();
      const pending = Math.max(0, state.treeMap.size - state.reviewed.size);
      byId('editmsg').textContent = pending ? `仍有 ${pending} 棵未逐棵查看，将按现有批量确认规则完成复核。` : '';
      byId('progress').textContent = '正在应用人工修改';
      try {
        await post('/api/confirm_trees', {});
        state.redirectOnDone = true;
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
      if (!state.editingBusy && !state.mutation && event.key === 'Escape' && (state.trimMode !== 'IDLE' || state.activePreview)) {
        event.preventDefault();
        void cancelTrim();
        return;
      }
      if ((event.key === 'ArrowLeft' || event.key === 'ArrowRight') && state.trimMode === 'IDLE' && !state.activePreview && !/INPUT|TEXTAREA|SELECT/.test((event.target && event.target.tagName) || '')) {
        event.preventDefault();
        stepTree(event.key === 'ArrowLeft' ? -1 : 1);
      }
    });
    window.addEventListener('error', restoreCameraControls);
    window.addEventListener('unhandledrejection', restoreCameraControls);
    updateUndoControl();
  }

  async function start() {
    bindControls();
    try {
      await loadEditorData();
      await refreshTrees();
      if (!state.workflowDone) state.refreshTimer = setInterval(refreshTrees, 1500);
    } catch (error) { showFailure(error); }
  }

  window.TreeLineTreeReviewApp = {start, state, selectTree, beginLasso, cancelTrim, metrics};
  window.TreeLineTreeReviewApp.retryViewer = retryViewer;
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
