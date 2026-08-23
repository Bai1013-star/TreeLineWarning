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
    previewRequestId: 0,
    editorLoadPromise: null,
    redirectOnDone: false,
    workflowDone: false,
    showLabels: true,
    localPreviewXYZ: [],
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
    const probe = document.createElement('canvas');
    const webgl2 = Boolean(probe.getContext('webgl2'));
    if (!webgl2 || cores <= 4 || memory <= 4) return 12000;
    if (cores >= 12 && memory >= 12) return 50000;
    return 30000;
  }

  function setStatus(text) { byId('status').textContent = text; }

  function setEditingBusy(busy) {
    ['btn-prev', 'btn-next', 'btn-delete', 'btn-merge', 'btn-trim-component', 'btn-trim-lasso', 'btn-undo', 'btn-trim-confirm', 'btn-confirm'].forEach(id => {
      const element = byId(id);
      if (element) element.disabled = Boolean(busy);
    });
  }

  function showFailure(error, prefix = '失败：') {
    restoreCameraControls();
    const message = error && error.message ? error.message : String(error);
    byId('editmsg').textContent = prefix + message;
    if (!state.viewer || !state.editorDisplay) {
      byId('viewer-renderer-status').textContent = '3D复核视图初始化失败，请查看详情。';
    }
    const details = byId('error-details');
    const text = byId('error-detail-text');
    const technicalDetail = error && (error.details || error.stack || error.message);
    if (details && text && technicalDetail) {
      text.textContent = technicalDetail;
      details.style.display = 'block';
    } else if (details) details.style.display = 'none';
  }

  async function post(path, payload) {
    const response = await fetch(path, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(payload || {}),
    });
    if (!response.ok) {
      const result = await response.json().catch(() => ({}));
      const error = new Error((result.error && result.error.message) || result.error || `请求失败 ${path}`);
      error.details = result.error && result.error.details ? result.error.details : '';
      throw error;
    }
    return response.json();
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

  function pointKey(point) { return point.map(value => Number(value).toPrecision(15)).join(','); }

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

  function displayLassoPreview(treeId, matrix, polygon) {
    const started = performance.now();
    const cloud = (state.editorDisplay && state.editorDisplay.trees3d) || {};
    const xyz = Array.isArray(cloud.xyz) ? cloud.xyz : [];
    const ids = Array.isArray(cloud.tree_id_by_index) ? cloud.tree_id_by_index : [];
    const preview = [];
    for (let index = 0; index < xyz.length; index += 1) {
      if (Number(ids[index]) !== Number(treeId)) continue;
      const point = xyz[index];
      const clip = [
        matrix[0][0] * point[0] + matrix[0][1] * point[1] + matrix[0][2] * point[2] + matrix[0][3],
        matrix[1][0] * point[0] + matrix[1][1] * point[1] + matrix[1][2] * point[2] + matrix[1][3],
        matrix[3][0] * point[0] + matrix[3][1] * point[1] + matrix[3][2] * point[2] + matrix[3][3],
      ];
      if (Number.isFinite(clip[0]) && Number.isFinite(clip[1]) && clip[2] > Number.EPSILON && pointInPolygon([clip[0] / clip[2], clip[1] / clip[2]], polygon)) preview.push(point);
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
    indicator.textContent = mode === 'LASSO' ? '套索模式：拖动鼠标圈选要剔除的区域\nEsc 取消' : '';
  }

  function requireOneTree() {
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
    const warning = preview.large_removal_warning ? ' 本次将移除该树较大比例的点，请确认选择是否正确。' : '';
    byId('trim-summary').textContent = `待剔除 ${Number(preview.point_count).toLocaleString()} 个点，占当前单木 ${percent}%。${warning}`;
    byId('trim-preview').dataset.previewId = String(preview.preview_id || '');
    byId('trim-preview').style.display = 'block';
    if (!keepLocalDisplayPreview) updatePreviewTrace(preview.preview_xyz || [], preview.tree_id);
  }

  async function previewComponent(treeId, seed) {
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
      setTrimMode('IDLE', error.message);
      restoreCameraControls();
    }
  }

  function beginLasso() {
    const treeId = requireOneTree();
    if (treeId == null || !state.viewer) return;
    removeLassoOverlay();
    setTrimMode('LASSO', '套索模式已开启：拖动鼠标圈选；松开后立即恢复旋转。Esc 取消。');
    state.viewer.setControlsEnabled(false);
    const stage = byId('tree-review-stage');
    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.classList.add('lasso-overlay');
    svg.setAttribute('aria-label', '套索绘制区域');
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
      const last = points[points.length - 1];
      if (Math.hypot(event.offsetX - last[0], event.offsetY - last[1]) >= 3) {
        points.push([event.offsetX, event.offsetY]);
        line.setAttribute('points', points.map(point => point.join(',')).join(' '));
      }
    };
    const abort = message => {
      drawing = false;
      removeLassoOverlay();
      setTrimMode('IDLE', message || '已取消套索，3D 旋转已恢复。');
    };
    svg.onpointercancel = () => abort('套索已取消，3D 旋转已恢复。');
    svg.onpointerup = async event => {
      if (!drawing) return;
      drawing = false;
      try { svg.releasePointerCapture(event.pointerId); } catch (_) {}
      const width = Math.max(1, svg.clientWidth);
      const height = Math.max(1, svg.clientHeight);
      removeLassoOverlay();
      if (points.length < 3) {
        setTrimMode('IDLE', '套索至少需要三个顶点。');
        return;
      }
      const polygon = points.map(point => [2 * point[0] / width - 1, 1 - 2 * point[1] / height]);
      try {
        const matrix = state.viewer.currentViewProjection();
        const displayCount = displayLassoPreview(treeId, matrix, polygon);
        setTrimMode('PREVIEW_LOADING', `显示样本已命中 ${displayCount.toLocaleString()} 个点并标红；正在精确计算完整点成员，3D 旋转已恢复...`);
        const requestId = ++state.previewRequestId;
        const exactStarted = performance.now();
        const preview = await post('/api/trim/lasso-preview', {tree_id: treeId, view_projection: matrix, polygon_ndc: polygon});
        const exactElapsed = performance.now() - exactStarted;
        metrics.lasso_exact_preview_ms.push(exactElapsed);
        byId('viewer-renderer-status').dataset.lastLassoExactMs = exactElapsed.toFixed(3);
        if (requestId !== state.previewRequestId || Number(state.activeTreeId) !== Number(treeId)) {
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
      }
    };
  }

  async function cancelTrim(message = '已取消，单木点成员未改变。') {
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
    state.viewer = new window.TreeLineTreeReview3D(byId('plot3d'), byId('tree-label-overlay'), {
      onTreePick: treeId => selectTree(treeId, true, true),
      onPointPick: hit => {
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
    });
    state.viewer.setLabelsVisible(state.showLabels);
    byId('btn-view-reset').onclick = () => state.viewer.resetView();
    byId('btn-view-top').onclick = () => state.viewer.topView();
    byId('btn-view-side').onclick = () => state.viewer.sideView();
  }

  async function loadEditorData() {
    if (state.editorLoadPromise) return state.editorLoadPromise;
    state.editorLoadPromise = (async () => {
      if (!state.editorDisplay) byId('editmsg').textContent = '正在加载单木列表与 Giro3D 预览...';
      const spanQuery = reviewSpanId ? `&span_id=${encodeURIComponent(reviewSpanId)}` : '';
      const response = await fetch(`/api/editor_data?display_budget=${displayBudget()}${spanQuery}`);
      if (!response.ok) throw new Error('单木显示数据加载失败');
      const data = await response.json();
      state.editorDisplay = data;
      renderTrees({trees: Array.isArray(data.trees) ? data.trees : []}, false);
      initializeViewer();
      const timing = state.viewer.setData(data);
      state.viewer.setLabelsVisible(state.showLabels);
      updatePlotSelection();
      const info = state.viewer.rendererInfo();
      metrics.renderer_info = info;
      byId('viewer-renderer-status').textContent = `Giro3D / Three.js r${info.three_version} · WebGL${info.webgl_version} · ${timing.visible_tree_points.toLocaleString()} 树点`;
      if (byId('editmsg').textContent.startsWith('正在加载')) byId('editmsg').textContent = '';
      return data;
    })();
    try { return await state.editorLoadPromise; }
    finally { state.editorLoadPromise = null; }
  }

  async function refreshTrees() {
    try {
      const current = await (await fetch('/state')).json();
      setStatus(current.step_label + (current.current_message ? ` — ${current.current_message}` : ''));
      if (current.step === 'FINALIZING') byId('progress').textContent = current.current_message || current.step_label;
      if (current.error) {
        const error = new Error(current.error);
        error.details = current.error_details || '';
        showFailure(error);
        byId('progress').textContent = current.current_message || '复核结果生成失败';
        setEditingBusy(false);
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
        setEditingBusy(false);
      }
      // A global workflow snapshot contains every batch tree.  In scoped review,
      // the authoritative filtered list comes from /api/editor_data?span_id=…;
      // do not let the background status poll replace it with all spans.
      if (current.tree_snapshot && !reviewSpanId) renderTrees(current.tree_snapshot, false);
    } catch (_) {}
  }

  function stepTree(delta) {
    if (!state.treeMap.size || state.trimMode !== 'IDLE' || state.activePreview) return;
    const ids = [...state.treeMap.keys()].sort((a, b) => a - b);
    const index = state.activeTreeId == null ? (delta > 0 ? -1 : 0) : ids.indexOf(Number(state.activeTreeId));
    selectTree(ids[Math.max(0, Math.min(ids.length - 1, index + delta))], false, true);
  }

  async function editAndReload(path, successMessage) {
    setEditingBusy(true);
    try {
      if (state.activePreview) await cancelTrim();
      await post('/api/select', {tree_ids: [...state.selected]});
      const snapshot = await post(path, {});
      state.selected.clear();
      renderTrees(snapshot, false);
      byId('editmsg').textContent = successMessage;
      await loadEditorData();
    } catch (error) { showFailure(error); }
    finally { setEditingBusy(false); }
  }

  function bindControls() {
    byId('btn-trim-component').onclick = () => { if (requireOneTree() != null) setTrimMode('COMPONENT', '请点击需要从当前单木中剔除的误分点团。'); };
    byId('btn-trim-lasso').onclick = beginLasso;
    byId('btn-trim-cancel').onclick = () => void cancelTrim();
    byId('btn-trim-confirm').onclick = async () => {
      if (!state.activePreview) return;
      setEditingBusy(true);
      try {
        byId('editmsg').textContent = '正在写入完整分辨率点成员...';
        const snapshot = await post('/api/trim/commit', {preview_id: state.activePreview.preview_id});
        state.activePreview = null;
        state.viewerState.activePreviewType = null;
        updatePreviewTrace([], null);
        state.selected.clear();
        byId('trim-preview').style.display = 'none';
        delete byId('trim-preview').dataset.previewId;
        setTrimMode('IDLE', '已剔除所选点，可使用撤销精确恢复。');
        renderTrees(snapshot, false);
        await loadEditorData();
      } catch (error) { showFailure(error); }
      finally { setEditingBusy(false); restoreCameraControls(); }
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
      setEditingBusy(true);
      try {
        if (state.activePreview) await cancelTrim();
        const snapshot = await post('/api/undo', {});
        state.selected.clear();
        renderTrees(snapshot, false);
        byId('editmsg').textContent = '已撤销上一步';
        await loadEditorData();
      } catch (error) { showFailure(error); }
      finally { setEditingBusy(false); }
    };
    byId('btn-confirm').onclick = async () => {
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
        setEditingBusy(false);
        return;
      }
      clearInterval(state.refreshTimer);
      state.refreshTimer = setInterval(refreshTrees, 700);
      void refreshTrees();
    };
    document.addEventListener('keydown', event => {
      if (event.key === 'Escape' && (state.trimMode !== 'IDLE' || state.activePreview)) {
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
})();
