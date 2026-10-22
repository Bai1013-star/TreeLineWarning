(() => {
  'use strict';

  // This page is a human identity gate.  It never accepts a typed diameter
  // and never sends an automatic candidate or seed selection to the server.
  const POLL_INTERVAL_MS = 700;
  const MUTATION_TIMEOUT_MS = 600000;
  const REQUEST_TIMEOUT_MS = 300000;
  const AXIS_COLORS = ['#55d4e4', '#ffcc47', '#c18cff', '#ff8a65', '#82d394', '#7aa7ff'];
  let requestSequence = 0;

  const state = {
    trees: [],
    treeMap: new Map(),
    activeTreeId: null,
    review: null,
    selectedCandidateId: null,
    selectedSeedOriginalIndex: null,
    selectionMode: 'NONE',
    viewer: null,
    viewerIndexById: new Map(),
    rendererReady: false,
    pending: null,
    pendingPromise: null,
    uncertain: false,
    completed: false,
    started: false,
  };

  const byId = id => document.getElementById(id);

  function text(id, value) {
    const node = byId(id);
    if (node) node.textContent = value == null ? '' : String(value);
  }

  function hidden(id, value) {
    const node = byId(id);
    if (node) node.classList.toggle('hidden', Boolean(value));
  }

  function finite(value, fallback = null) {
    if (value == null || value === '') return fallback;
    const number = Number(value);
    return Number.isFinite(number) ? number : fallback;
  }

  function displayNumber(value, digits = 2) {
    const number = finite(value);
    return number == null ? '—' : number.toFixed(digits);
  }

  function esc(value) {
    return String(value ?? '').replace(/[&<>"']/g, character => ({
      '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
    }[character]));
  }

  function displayId(tree) {
    return tree.display_tree_id || tree.persistent_tree_id || tree.global_tree_id || '未命名单木';
  }

  function isUnavailable(tree) {
    return Boolean(tree) && tree.raw_original_las_review_available !== false;
  }

  function methodLabel(method) {
    const labels = {
      REVIEW_CONFIRMED_STEM: '人工确认点云',
      RECOVERED_STEM: '恢复点云',
      DIRECT_STEM: '点云',
      ALLOMETRIC_ESTIMATE: '估算',
      RAW_GROUND_BH_SECTION: '原始树木最低点+1.30 m断面拟合',
      UNAVAILABLE: '不可用',
    };
    return labels[String(method || '')] || String(method || '—');
  }

  function requestId() {
    requestSequence += 1;
    if (window.crypto && typeof window.crypto.randomUUID === 'function') {
      return `${window.crypto.randomUUID()}-${requestSequence.toString(36)}`;
    }
    return `dbh-review-${Date.now().toString(36)}-${requestSequence.toString(36)}`;
  }

  function setReviewStatus(message, kind = '') {
    const node = byId('review-status');
    if (!node) return;
    node.textContent = message;
    node.className = `notice ${kind}`.trim();
  }

  function setConfirmMessage(message, kind = '') {
    const node = byId('confirm-message');
    if (!node) return;
    node.textContent = message || '';
    node.dataset.kind = kind;
  }

  function renderTreeList() {
    const list = byId('tree-list');
    if (!list) return;
    list.innerHTML = '';
    const unavailable = state.trees.filter(isUnavailable);
    const selectableCount = unavailable.length;
    text('tree-count', `${selectableCount} 棵可查看原始1.30 m断面`);
    if (!state.trees.length) {
      text('tree-list-note', '当前没有可读取的单木复核记录。');
      return;
    }
    if (!selectableCount) text('tree-list-note', '所有胸径已经定稿；请在“查看数据”中查看已发布数值。');
    state.trees.forEach(tree => {
      const button = document.createElement('button');
      const selectable = isUnavailable(tree);
      const treeId = String(tree.persistent_tree_id ?? tree.global_tree_id ?? '');
      button.type = 'button';
      button.className = `tree-entry${state.activeTreeId === treeId ? ' active' : ''}${selectable ? '' : ' locked'}`;
      button.dataset.treeId = treeId;
      button.disabled = !selectable || Boolean(state.pending) || state.uncertain;
      button.innerHTML = `<span class="tree-entry-title">${esc(displayId(tree))}</span><span class="tree-entry-meta">当前胸径：${tree.dbh_cm == null ? 'No（未测量，可在预测信息中补填实测值）' : `${displayNumber(tree.dbh_cm)} cm（点云估计）`} · ${esc(methodLabel(tree.dbh_method))}</span>`;
      if (tree.dbh_cm == null) {
        const status = String(tree.dbh_fit_status || '');
        const reason = status.includes('ORIGINAL_LAS_MISSING') ? '原始LAS缺失，请恢复原文件路径'
          : status.includes('ORIGINAL_GROUND') ? '未取得可信原始树木最低点'
          : status.includes('RAW_1P30_SECTION') ? '1.30 m断面证据不足'
          : '树干几何/归属证据不足，需原始点云复核';
        button.insertAdjacentHTML('beforeend', `<span class="tree-entry-meta">${esc(reason)}</span>`);
      }
      if (selectable) button.addEventListener('click', () => selectTree(treeId));
      list.appendChild(button);
    });
  }

  function renderActiveTree() {
    const tree = state.treeMap.get(String(state.activeTreeId));
    if (!tree) {
      hidden('review-empty', false);
      hidden('review-content', true);
      return;
    }
    hidden('review-empty', true);
    hidden('review-content', false);
    text('review-tree-title', `${displayId(tree)} · 胸径复核`);
    text('review-tree-meta', `persistent_tree_id=${tree.persistent_tree_id ?? '—'} · global_tree_id=${tree.global_tree_id ?? '—'} · 当前方法=${methodLabel(tree.dbh_method)}`);
    setReviewStatus('请选择“准备点云复核”。');
    setConfirmMessage('');
    updateControls();
  }

  function disposeViewer() {
    if (state.viewer && typeof state.viewer.dispose === 'function') {
      try { state.viewer.dispose(); } catch (_) { /* cleanup is best effort */ }
    }
    state.viewer = null;
    state.viewerIndexById.clear();
    state.rendererReady = false;
    const host = byId('dbh-plot3d');
    if (host) host.textContent = '';
    const axisOverlay = byId('dbh-axis-overlay');
    if (axisOverlay) axisOverlay.innerHTML = '';
  }

  function resetSelection() {
    state.selectedCandidateId = null;
    state.selectedSeedOriginalIndex = null;
    state.selectionMode = 'NONE';
    const checkbox = byId('identity-confirmed');
    if (checkbox) checkbox.checked = false;
    text('seed-mode-label', '未选择');
    text('seed-selection', '尚未选择原始种子点。');
    text('point-detail', '');
  }

  function resetReview(message = '') {
    if (state.pending) return false;
    disposeViewer();
    state.review = null;
    state.completed = false;
    resetSelection();
    hidden('review-workspace', true);
    text('diagnostics-text', '');
    text('dbh-context-label', '当前树冠上下文：—');
    if (message) setReviewStatus(message);
    updateControls();
    return true;
  }

  function selectedTree() {
    return state.treeMap.get(String(state.activeTreeId)) || null;
  }

  function normalizeDisplay(review) {
    const display = review && review.display;
    const xyz = display && Array.isArray(display.xyz) ? display.xyz : null;
    const originalIndex = display && Array.isArray(display.original_index) ? display.original_index : null;
    const hag = display && Array.isArray(display.hag_m) ? display.hag_m : null;
    const selectable = display && Array.isArray(display.selectable) ? display.selectable : null;
    if (!xyz || !originalIndex || !hag || !selectable || new Set([xyz.length, originalIndex.length, hag.length, selectable.length]).size !== 1) {
      throw new Error('复核显示数据的 xyz、original_index、hag_m、selectable 未对齐');
    }
    const indices = originalIndex.map(value => Number(value));
    if (indices.some(value => !Number.isSafeInteger(value) || value < 0) || new Set(indices).size !== indices.length) {
      throw new Error('复核显示数据包含无效或重复 original_index');
    }
    if (xyz.some(point => !Array.isArray(point) || point.length < 3 || point.slice(0, 3).some(value => !Number.isFinite(Number(value))))) {
      throw new Error('复核显示数据包含无效 XYZ');
    }
    if (hag.some(value => !Number.isFinite(Number(value)))) throw new Error('复核显示数据包含无效 HAG');
    if (selectable.some(value => typeof value !== 'boolean')) throw new Error('复核显示数据包含无效 selectable 标志');
    return {xyz, originalIndex: indices, hag: hag.map(Number), selectable: selectable.map(Boolean)};
  }

  function axisEndpoints(axis, extent = 2.5) {
    const origin = Array.isArray(axis && axis.bh_reference_xyz) ? axis.bh_reference_xyz.map(Number) : null;
    const direction = Array.isArray(axis && axis.direction_xyz) ? axis.direction_xyz.map(Number) : null;
    if (!origin || !direction || origin.length < 3 || direction.length < 3) return [];
    if (![...origin, ...direction].every(Number.isFinite)) return [];
    return [
      origin.map((value, index) => value - direction[index] * extent),
      origin.map((value, index) => value + direction[index] * extent),
    ];
  }

  function viewerPayload(review, normalized) {
    const crown = review.current_crown && Array.isArray(review.current_crown.xyz)
      ? review.current_crown.xyz : [];
    const viewerIds = normalized.originalIndex.map(index => Number(index));
    state.viewerIndexById = new Map(viewerIds.map((viewerId, row) => [String(viewerId), normalized.originalIndex[row]]));
    const colors = normalized.hag.map((height, index) => Math.abs(height - 1.30) <= 0.05
      ? '#ffcc47'
      : (normalized.selectable[index] ? '#63d471' : '#6f8192'));
    return {
      background: {xyz: crown, rgb: crown.map(() => [94, 119, 234])},
      trees3d: {
        xyz: normalized.xyz,
        color: colors,
        // The viewer's sourceIndex is the display row.  Its tree id is set to
        // the exact original_index carried by that row, so a pick never
        // reconstructs identity from XYZ or from a crown overlay.
        tree_id_by_index: viewerIds,
        labels: [],
      },
      // TreeReviewViewer3D treats wires3d as one connected polyline.  Candidate
      // axes are therefore drawn by the independent SVG overlay below so two
      // plausible stems can never be visually joined into a false axis.
      wires3d: {xyz: []},
      wire: {xyz: []},
      tower: {xyz: []},
      danger_points: {xyz: []},
    };
  }

  function initViewer() {
    if (state.viewer) return state.viewer;
    const constructor = window.TreeLineTreeReview3D;
    if (typeof constructor !== 'function') throw new Error('本地 Giro3D / Three.js 资源不可用');
    state.viewer = new constructor(byId('dbh-plot3d'), byId('dbh-tree-label-overlay'), {
      onTreePick: () => true,
      onPointPick: hit => {
        if (state.pending || state.uncertain || !state.rendererReady || state.selectionMode !== 'SEED') return true;
        const originalIndex = state.viewerIndexById.get(String(Number(hit && hit.treeId)));
        if (originalIndex == null) return true;
        selectSeedIndex(originalIndex);
        return true;
      },
      onQualityChange: event => {
        text('dbh-renderer-status', `Giro3D / Three.js · 显示 ${Number(event.visible_points || 0).toLocaleString()} 点`);
      },
      onInitFailure: error => handleViewerFailure(error),
      onRendered: () => renderAxisOverlay(),
    });
    if (byId('btn-view-reset')) byId('btn-view-reset').onclick = () => state.viewer && state.viewer.resetView();
    if (byId('btn-view-top')) byId('btn-view-top').onclick = () => state.viewer && state.viewer.topView();
    if (byId('btn-view-side')) byId('btn-view-side').onclick = () => state.viewer && state.viewer.sideView();
    if (typeof state.viewer.setLabelsVisible === 'function') state.viewer.setLabelsVisible(false);
    return state.viewer;
  }

  function projectAxisPoint(point, matrix, width, height) {
    if (!Array.isArray(point) || point.length < 3 || !Array.isArray(matrix) || matrix.length < 4) return null;
    const vector = [Number(point[0]), Number(point[1]), Number(point[2]), 1];
    if (!vector.slice(0, 3).every(Number.isFinite)) return null;
    const clip = matrix.slice(0, 4).map(row => {
      if (!Array.isArray(row) || row.length < 4) return NaN;
      return row.slice(0, 4).reduce((sum, value, index) => sum + Number(value) * vector[index], 0);
    });
    const w = clip[3];
    if (!Number.isFinite(w) || Math.abs(w) < 1e-12 || !clip.slice(0, 3).every(Number.isFinite)) return null;
    const x = clip[0] / w;
    const y = clip[1] / w;
    const z = clip[2] / w;
    // SVG clips the projected line to the stage.  Keep endpoints outside the
    // frustum so a long axis remains visible when only part of it is in view.
    if (![x, y, z].every(Number.isFinite)) return null;
    return {x: (x + 1) * 0.5 * width, y: (1 - y) * 0.5 * height};
  }

  function renderAxisOverlay() {
    const svg = byId('dbh-axis-overlay');
    if (!svg) return;
    svg.innerHTML = '';
    if (!state.viewer || !state.review || typeof state.viewer.currentViewProjection !== 'function') return;
    if (typeof document.createElementNS !== 'function') return;
    let matrix;
    try { matrix = state.viewer.currentViewProjection(); } catch (_) { return; }
    const rect = typeof svg.getBoundingClientRect === 'function' ? svg.getBoundingClientRect() : null;
    const width = Math.max(1, Number(rect && rect.width) || svg.clientWidth || 1);
    const height = Math.max(1, Number(rect && rect.height) || svg.clientHeight || 1);
    svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
    const candidates = Array.isArray(state.review.candidates) ? state.review.candidates : [];
    candidates.forEach((candidate, index) => {
      const endpoints = axisEndpoints(candidate.candidate_axis);
      if (endpoints.length !== 2) return;
      const first = projectAxisPoint(endpoints[0], matrix, width, height);
      const second = projectAxisPoint(endpoints[1], matrix, width, height);
      if (!first || !second) return;
      const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
      const color = AXIS_COLORS[index % AXIS_COLORS.length];
      const selected = state.selectedCandidateId != null
        && String(state.selectedCandidateId) === String(candidate.candidate_id);
      line.setAttribute('x1', String(first.x));
      line.setAttribute('y1', String(first.y));
      line.setAttribute('x2', String(second.x));
      line.setAttribute('y2', String(second.y));
      line.setAttribute('stroke', color);
      line.setAttribute('stroke-width', selected ? '5' : '2.5');
      line.setAttribute('stroke-linecap', 'round');
      line.setAttribute('class', selected ? 'dbh-axis-line selected' : 'dbh-axis-line');
      line.setAttribute('data-candidate-id', String(candidate.candidate_id ?? ''));
      line.setAttribute('vector-effect', 'non-scaling-stroke');
      svg.appendChild(line);
    });
  }

  function renderCandidateCards(candidates) {
    const container = byId('axis-candidates');
    if (!container) return;
    container.innerHTML = '';
    if (!candidates.length) {
      container.textContent = '后台未提供可供确认的候选轴；可使用种子点模式。';
      return;
    }
    candidates.forEach((candidate, index) => {
      const card = document.createElement('article');
      const button = document.createElement('button');
      const color = AXIS_COLORS[index % AXIS_COLORS.length];
      const axis = candidate.candidate_axis || {};
      const support = candidate.candidate_support == null ? '—' : candidate.candidate_support;
      const continuity = candidate.continuity_status == null ? '—' : candidate.continuity_status;
      card.className = `axis-candidate${state.selectedCandidateId === candidate.candidate_id ? ' selected' : ''}`;
      card.style.setProperty('--axis-color', color);
      card.dataset.candidateId = String(candidate.candidate_id || '');
      button.type = 'button';
      button.disabled = Boolean(state.pending) || state.uncertain || !state.rendererReady || state.completed;
      button.innerHTML = `<i class="axis-swatch" style="--axis-color:${color}"></i><span><span class="axis-candidate-title">候选 ${esc(candidate.candidate_id || '—')}</span><span class="axis-candidate-meta">score=${esc(finite(candidate.candidate_score, '—'))} · support=${esc(support)} · continuity=${esc(continuity)}</span></span>`;
      button.addEventListener('click', () => selectCandidate(candidate.candidate_id));
      card.appendChild(button);
      const detail = document.createElement('div');
      detail.className = 'axis-candidate-detail';
      detail.textContent = `BH ${JSON.stringify(axis.bh_reference_xyz || [])} · direction ${JSON.stringify(axis.direction_xyz || [])} · slope ${JSON.stringify(axis.slope_xyz_per_hag || [])} · radius ${finite(axis.radius_m, '—')} m`;
      card.appendChild(detail);
      container.appendChild(card);
    });
  }

  function renderReview(review) {
    const normalized = normalizeDisplay(review);
    state.review = review;
    state.selectedCandidateId = null;
    state.selectedSeedOriginalIndex = null;
    state.selectionMode = 'NONE';
    state.completed = false;
    state.rendererReady = false;
    const candidates = Array.isArray(review.candidates) ? review.candidates : [];
    renderCandidateCards(candidates);
    hidden('review-workspace', false);
    const crownLabel = review.current_crown && review.current_crown.label
      ? review.current_crown.label : '未提供标签';
    const references = Array.isArray(review.ground_references) ? review.ground_references : [];
    const ground = references.length ? displayNumber(references[0].ground_z_absolute_m, 4) : '—';
    const section = references.length ? displayNumber(references[0].breast_height_z_absolute_m, 4) : '—';
    text('dbh-context-label', `当前树冠：${crownLabel} · 原始树最低点 Z=${ground} m · 胸径截面 Z=${section} m（最低点+1.30 m）`);
    if (review.current_dbh_cm != null) {
      setConfirmMessage(`当前已有胸径 ${displayNumber(review.current_dbh_cm)} cm。您可重新选择候选轴或树干种子点；只有新计算通过质量检查才会替换原值，失败会保留原值。`, 'warn');
    }
    const sample = values => values.length <= 12 ? values : [...values.slice(0, 6), '…', ...values.slice(-3)];
    text('diagnostics-text', JSON.stringify({
      review_id: review.review_id,
      editor_revision: review.editor_revision,
      mode: review.mode,
      candidate_count: candidates.length,
      candidates: candidates.map(candidate => ({
        candidate_id: candidate.candidate_id,
        candidate_score: candidate.candidate_score,
        candidate_support: candidate.candidate_support,
        continuity_status: candidate.continuity_status,
        candidate_axis: candidate.candidate_axis,
      })),
      display_point_count: normalized.xyz.length,
      selectable_point_count: normalized.selectable.filter(Boolean).length,
      display_original_index_sample: sample(normalized.originalIndex),
      display_hag_m_sample: sample(normalized.hag),
      height_reference: review.height_reference,
      ground_references: references,
    }, null, 2));
    try {
      const viewer = initViewer();
      if (viewer.disposed || viewer.runtimeFailure) {
        throw viewer.runtimeFailure || new Error('Giro3D / Three.js 渲染器初始化失败');
      }
      const timing = viewer.setData(viewerPayload(review, normalized));
      if (viewer.disposed || viewer.runtimeFailure) {
        throw viewer.runtimeFailure || new Error('Giro3D / Three.js 渲染器初始化失败');
      }
      state.rendererReady = true;
      if (typeof viewer.setSelection === 'function') viewer.setSelection([]);
      if (typeof viewer.setPreview === 'function') viewer.setPreview([], null);
      renderAxisOverlay();
      text('dbh-renderer-status', `Giro3D / Three.js · 原始显示 ${normalized.xyz.length.toLocaleString()} 点 · ${timing && timing.visible_tree_points != null ? Number(timing.visible_tree_points).toLocaleString() : ''}`);
      hidden('dbh-viewer-error', true);
    } catch (error) {
      handleViewerFailure(error);
    }
    if (state.rendererReady) {
      setReviewStatus(review.current_dbh_cm != null
        ? `原始点云已准备；当前胸径 ${displayNumber(review.current_dbh_cm)} cm。可选候选轴或点选树干重新复核。`
        : '点云已准备。请选择候选轴，或进入种子点模式后点击可选原始点。');
    }
    resetSelection();
    updateControls();
  }

  function showViewerError(error) {
    const panel = byId('dbh-viewer-error');
    if (!panel) return;
    panel.textContent = `3D 视图不可用：${error && error.message ? error.message : String(error)}。原始点索引与诊断仍可查看。`;
    panel.classList.remove('hidden');
  }

  function handleViewerFailure(error) {
    state.rendererReady = false;
    showViewerError(error);
    setReviewStatus('3D 视图不可用；请修复本地渲染资源后再确认点云身份。', 'error');
    updateControls();
  }

  function rowForIndex(index) {
    if (!state.review) return -1;
    const display = normalizeDisplay(state.review);
    return display.originalIndex.indexOf(Number(index));
  }

  function updateControls() {
    const pending = Boolean(state.pending) || state.uncertain;
    const hasSelection = state.selectedCandidateId != null || state.selectedSeedOriginalIndex != null;
    const checked = Boolean(byId('identity-confirmed') && byId('identity-confirmed').checked);
    const prepare = byId('btn-prepare');
    if (prepare) prepare.disabled = pending || !selectedTree() || !isUnavailable(selectedTree()) || Boolean(state.review);
    const reset = byId('btn-reset-review');
    if (reset) reset.disabled = pending || !state.review;
    const seed = byId('btn-pick-seed');
    if (seed) seed.disabled = pending || !state.review || !state.rendererReady || state.completed;
    const identity = byId('identity-confirmed');
    if (identity) identity.disabled = pending;
    const cancelSeed = byId('btn-cancel-seed');
    if (cancelSeed) cancelSeed.disabled = pending || state.selectionMode !== 'SEED';
    const confirm = byId('btn-confirm');
    if (confirm) confirm.disabled = pending || state.completed || !state.review || !state.rendererReady || !hasSelection || !checked;
    const best = byId('btn-best-candidate');
    if (best) best.disabled = pending || state.completed || !state.review || !state.rendererReady || !(state.review.candidates || []).length;
    document.querySelectorAll('#tree-list .tree-entry').forEach(node => {
      const tree = state.treeMap.get(String(node.dataset.treeId));
      node.disabled = pending || !isUnavailable(tree);
    });
    document.querySelectorAll('#axis-candidates button').forEach(node => {
      node.disabled = pending || state.uncertain || !state.rendererReady || state.completed;
    });
    const seedLabel = state.selectionMode === 'SEED' ? '点选中' : state.selectedCandidateId ? '候选轴已选' : '未选择';
    text('seed-mode-label', seedLabel);
  }

  function selectTree(treeId) {
    if (state.pending || state.uncertain) return;
    const tree = state.treeMap.get(String(treeId));
    if (!tree || !isUnavailable(tree)) return;
    state.activeTreeId = String(treeId);
    resetReview();
    renderActiveTree();
    renderTreeList();
  }

  function selectCandidate(candidateId) {
    if (!state.review || state.pending || state.uncertain || !state.rendererReady || state.completed) return;
    const candidate = (state.review.candidates || []).find(item => String(item.candidate_id) === String(candidateId));
    if (!candidate) return;
    state.selectedCandidateId = candidate.candidate_id;
    state.selectedSeedOriginalIndex = null;
    state.selectionMode = 'CANDIDATE';
    const checkbox = byId('identity-confirmed');
    if (checkbox) checkbox.checked = false;
    if (state.viewer && typeof state.viewer.setSelection === 'function') state.viewer.setSelection([]);
    renderAxisOverlay();
    renderCandidateCards(state.review.candidates || []);
    text('point-detail', '');
    text('seed-selection', '已选择候选轴；未选择种子点。');
    setReviewStatus(`已选择候选轴 ${candidate.candidate_id}。请阅读诊断并勾选身份确认。`);
    updateControls();
  }

  function selectBestCandidate() {
    if (!state.review) return;
    const candidates = [...(state.review.candidates || [])].filter(item => Number.isFinite(Number(item.candidate_score)));
    candidates.sort((left, right) => Number(right.candidate_score) - Number(left.candidate_score));
    if (!candidates.length) {
      setReviewStatus('没有可用候选轴，请使用“点击树干”选择精确原始种子点。', 'warn');
      return;
    }
    selectCandidate(candidates[0].candidate_id);
    byId('identity-confirm-label')?.scrollIntoView({block: 'center', behavior: 'smooth'});
    byId('identity-confirmed')?.focus();
  }

  function enterSeedMode() {
    if (!state.review || state.pending || state.uncertain || !state.rendererReady || state.completed) return;
    state.selectionMode = 'SEED';
    state.selectedCandidateId = null;
    state.selectedSeedOriginalIndex = null;
    const checkbox = byId('identity-confirmed');
    if (checkbox) checkbox.checked = false;
    renderCandidateCards(state.review.candidates || []);
    renderAxisOverlay();
    setReviewStatus('种子点模式已开启。只点击当前显示点云中的可选原始点；树冠覆盖层不可选。');
    text('seed-selection', '等待点击可选原始点…');
    updateControls();
  }

  function selectSeedIndex(index) {
    if (!state.review || state.pending || state.uncertain || !state.rendererReady || state.completed || state.selectionMode !== 'SEED') return;
    const row = rowForIndex(index);
    if (row < 0) return;
    const display = normalizeDisplay(state.review);
    if (!display.selectable[row]) {
      setReviewStatus('该原始点未被后台标记为可选点，请选择绿色当前树干点。', 'warn');
      return;
    }
    state.selectedCandidateId = null;
    state.selectedSeedOriginalIndex = display.originalIndex[row];
    const point = display.xyz[row];
    const hag = display.hag[row];
    if (state.viewer && typeof state.viewer.setSelection === 'function') state.viewer.setSelection([Number(state.selectedSeedOriginalIndex)]);
    if (state.viewer && typeof state.viewer.setPreview === 'function') state.viewer.setPreview([point], Number(state.selectedSeedOriginalIndex));
    text('seed-selection', `已选择原始种子点 original_index=${state.selectedSeedOriginalIndex}`);
    text('point-detail', `XYZ=${point.slice(0, 3).map(value => finite(value, '—')).join(', ')} · HAG=${finite(hag, '—')} m`);
    setReviewStatus('已选择精确原始种子点。请阅读诊断并勾选身份确认。');
    updateControls();
  }

  function staleError(error) {
    const textValue = String(error && (error.code || error.message || error) || '').toUpperCase();
    return textValue.includes('STALE') || textValue.includes('REVISION') || textValue.includes('VERSION');
  }

  function unknownError(message) {
    const error = new Error(message);
    error.uncertainMutation = true;
    return error;
  }

  async function jsonResponse(response) {
    let result;
    try { result = await response.json(); } catch (_) {
      const error = new Error('后台返回了无法读取的 JSON');
      error.responseParse = true;
      throw error;
    }
    return result;
  }

  async function fetchJson(path, init = {}, timeoutMs = REQUEST_TIMEOUT_MS) {
    let response;
    let result;
    const controller = typeof AbortController === 'function' && !init.signal ? new AbortController() : null;
    const requestInit = controller ? {...init, signal: controller.signal} : init;
    const timeout = controller ? setTimeout(() => controller.abort(), timeoutMs) : null;
    try {
      response = await fetch(path, requestInit);
      // Keep the same abort deadline through response.json().  A stalled or
      // malformed mutation body must remain fail-closed and never be retried.
      result = await jsonResponse(response);
    } catch (cause) {
      if (cause && cause.responseParse) {
        cause.network = true;
        throw cause;
      }
      const timedOut = Boolean(controller && controller.signal.aborted);
      const error = new Error(timedOut
        ? `请求超时（${Math.ceil(timeoutMs / 1000)} 秒）`
        : `请求失败：${cause && cause.message ? cause.message : cause}`);
      error.network = true;
      error.timeout = timedOut;
      throw error;
    } finally {
      if (timeout != null) clearTimeout(timeout);
    }
    if (!response.ok) {
      const error = new Error(result && result.error && (result.error.message || result.error) || result && result.message || `请求失败 ${response.status}`);
      error.code = result && result.error && result.error.code || result && result.code || '';
      error.details = result && result.error && result.error.details || '';
      error.httpStatus = response.status;
      throw error;
    }
    return {response, result};
  }

  function running(result, status) {
    const stateValue = String(result && (result.state || result.status) || '').toUpperCase();
    return status === 202 || ['RUNNING', 'PENDING', 'PROCESSING', 'QUEUED', 'STARTED'].some(value => stateValue.includes(value));
  }

  function terminalPayload(result) {
    if (result && result.result && typeof result.result === 'object' && !Array.isArray(result.result)) return result.result;
    return result;
  }

  function requestMatches(result, requestIdValue) {
    return Boolean(result && typeof result.request_id === 'string'
      && result.request_id === requestIdValue);
  }

  function delay(milliseconds) {
    return new Promise(resolve => setTimeout(resolve, milliseconds));
  }

  async function pollMutation(requestIdValue, pending) {
    const deadline = Date.now() + MUTATION_TIMEOUT_MS;
    while (Date.now() <= deadline) {
      if (!state.pending || state.pending.requestId !== requestIdValue) throw unknownError('复核请求状态已改变');
      const elapsed = Math.max(0, Math.floor((Date.now() - pending.startedAt) / 1000));
      setReviewStatus(`${pending.phase === 'prepare' ? '正在准备点云证据' : '正在提交人工确认'}（已等待 ${elapsed} 秒）`);
      await delay(POLL_INTERVAL_MS);
      const remaining = deadline - Date.now();
      if (remaining <= 0) throw unknownError('复核请求等待超过 600 秒，结果不确定；请重新打开复核页面核对，勿重复提交。');
      const path = `/api/editor/mutation?request_id=${encodeURIComponent(requestIdValue)}`;
      let outcome;
      try {
        outcome = await fetchJson(path, {headers: {Accept: 'application/json'}, cache: 'no-store'}, Math.min(REQUEST_TIMEOUT_MS, remaining));
      } catch (error) {
        error.uncertainMutation = true;
        throw error;
      }
      if (!requestMatches(outcome.result, requestIdValue)) {
        throw unknownError('复核请求状态的 request_id 不匹配，结果不确定；请重新打开页面核对。');
      }
      if (running(outcome.result, outcome.response.status)) continue;
      return terminalPayload(outcome.result);
    }
    throw unknownError('复核请求等待超过 600 秒，结果不确定；请重新打开复核页面核对，勿重复提交。');
  }

  async function submitMutation(path, payload, phase) {
    if (state.pending || state.uncertain) return state.pendingPromise;
    const requestIdValue = requestId();
    const pending = {requestId: requestIdValue, phase, startedAt: Date.now(), path};
    state.pending = pending;
    updateControls();
    const body = {...payload, async: true, request_id: requestIdValue};
    const promise = (async () => {
      try {
        const outcome = await fetchJson(path, {
          method: 'POST',
          headers: {'Content-Type': 'application/json', Accept: 'application/json'},
          body: JSON.stringify(body),
        });
        if (!requestMatches(outcome.result, requestIdValue)) {
          throw unknownError('复核提交响应的 request_id 不匹配，结果不确定；请重新打开页面核对。');
        }
        const terminal = running(outcome.result, outcome.response.status)
          ? await pollMutation(requestIdValue, pending) : terminalPayload(outcome.result);
        if (!terminal || terminal.ok !== true) {
          const error = new Error(terminal && (terminal.message || terminal.error && terminal.error.message) || '后台未确认复核结果');
          error.code = terminal && terminal.error && terminal.error.code || terminal && terminal.code || '';
          throw error;
        }
        return terminal;
      } catch (error) {
        if (staleError(error)) {
          state.pending = null;
          state.pendingPromise = null;
          // A stale confirm belongs to an old editor revision.  Discard the
          // displayed review so the next action must prepare fresh evidence.
          if (state.review) resetReview();
          setReviewStatus('编辑版本已变化，请重新准备当前版本的点云复核。', 'warn');
          updateControls();
          return null;
        }
        if (error && (error.uncertainMutation || error.network)) {
          state.uncertain = true;
          pending.uncertain = true;
          pending.status = 'UNKNOWN';
          setReviewStatus('结果不确定，请重新打开复核页面核对编辑版本；未自动重复提交。', 'warn');
          updateControls();
          return null;
        }
        state.pending = null;
        state.pendingPromise = null;
        updateControls();
        throw error;
      }
    })();
    state.pendingPromise = promise;
    promise.finally(() => {
      if (state.pending === pending && !state.uncertain) {
        state.pending = null;
        state.pendingPromise = null;
        updateControls();
      }
    }).catch(() => {});
    return promise;
  }

  async function loadTrees() {
    try {
      const outcome = await fetchJson('/api/dbh-review/trees', {headers: {Accept: 'application/json'}, cache: 'no-store'}, MUTATION_TIMEOUT_MS);
      const payload = outcome.result || {};
      if (payload.human_can_type_dbh !== false) throw new Error('后台未声明禁止手工输入胸径');
      state.trees = Array.isArray(payload.trees) ? payload.trees : [];
      state.treeMap = new Map(state.trees.map(tree => [String(tree.persistent_tree_id ?? tree.global_tree_id ?? ''), tree]));
      renderTreeList();
      if (!state.trees.some(isUnavailable)) text('tree-list-note', '所有胸径已经定稿；请在“查看数据”中查看已发布数值。');
      return payload;
    } catch (error) {
      text('tree-count', '读取失败');
      text('tree-list-note', `读取待复核单木失败：${error.message || error}`);
      setReviewStatus('无法读取待复核单木。', 'error');
      return null;
    }
  }

  async function prepareTree(treeId = state.activeTreeId) {
    if (state.pending || state.uncertain) return state.pendingPromise;
    const tree = state.treeMap.get(String(treeId));
    if (!tree || !isUnavailable(tree)) return null;
    state.activeTreeId = String(treeId);
    resetReview();
    renderActiveTree();
    setReviewStatus('正在准备当前编辑版本的原始点云证据…');
    let terminal;
    try {
      terminal = await submitMutation('/api/dbh-review/prepare', {
        persistent_tree_id: tree.persistent_tree_id,
      }, 'prepare');
    } catch (error) {
      setReviewStatus(`准备点云复核失败：${error.message || error}`, 'error');
      setConfirmMessage('未提交任何手工胸径；可在确认编辑版本后重试。', 'warn');
      return null;
    }
    if (!terminal) return null;
    try {
      if (!terminal.review) throw new Error('后台复核结果缺少 review');
      renderReview(terminal.review);
      return terminal;
    } catch (error) {
      setReviewStatus(`复核显示数据无效：${error.message || error}`, 'error');
      hidden('review-workspace', true);
      return null;
    }
  }

  function selectionPayload() {
    if (state.selectedCandidateId != null && state.selectedSeedOriginalIndex == null) {
      return {candidate_id: state.selectedCandidateId};
    }
    if (state.selectedCandidateId == null && state.selectedSeedOriginalIndex != null) {
      return {seed_original_index: state.selectedSeedOriginalIndex};
    }
    return null;
  }

  async function confirmReview() {
    if (state.pending || state.uncertain || state.completed || !state.review) return null;
    if (!state.rendererReady) {
      setConfirmMessage('3D 视图未就绪，暂不能确认点云身份。', 'warn');
      return null;
    }
    const checkbox = byId('identity-confirmed');
    if (!checkbox || !checkbox.checked) {
      setConfirmMessage('请先确认点云身份。', 'warn');
      return null;
    }
    const selection = selectionPayload();
    if (!selection) {
      setConfirmMessage('请选择一个候选轴或一个精确原始种子点。', 'warn');
      return null;
    }
    setConfirmMessage('正在提交人工身份确认；后台将独立计算胸径…');
    let terminal;
    try {
      terminal = await submitMutation('/api/dbh-review/confirm', {
        review_id: state.review.review_id,
        identity_confirmed: true,
        ...selection,
      }, 'confirm');
    } catch (error) {
      setConfirmMessage(`人工确认提交失败：${error.message || error}`, 'error');
      setReviewStatus('人工确认未完成；未写入手工胸径。', 'error');
      return null;
    }
    if (!terminal) return null;
    const status = String(terminal.status || '').toUpperCase();
    if (status === 'RECOVERED') {
      state.completed = true;
      const tree = selectedTree();
      if (tree) {
        tree.dbh_cm = terminal.dbh_cm;
        tree.dbh_method = terminal.dbh_method || 'REVIEW_CONFIRMED_STEM';
        tree.dbh_quality = terminal.dbh_quality;
      }
      renderTreeList();
      setReviewStatus(`已恢复胸径 ${finite(terminal.dbh_cm, '—')} cm · ${methodLabel(terminal.dbh_method || 'REVIEW_CONFIRMED_STEM')}`, '');
      setConfirmMessage(`人工身份已确认，几何计算状态：${terminal.dbh_fit_status || 'PASS'}。`, 'ok');
    } else {
      state.completed = true;
      const previous = finite(state.review.current_dbh_cm);
      setReviewStatus(previous == null
        ? '该树干点云不足，无法可靠计算胸径'
        : `重新复核未通过，已保留原胸径 ${displayNumber(previous)} cm`, 'warn');
      setConfirmMessage(`结果：UNAVAILABLE · ${terminal.dbh_fit_status || terminal.message || '质量门未通过'}。${previous == null ? '未写入胸径。' : '原胸径未被覆盖。'}`, 'warn');
    }
    updateControls();
    return terminal;
  }

  function bindControls() {
    const prepare = byId('btn-prepare');
    if (prepare) prepare.addEventListener('click', () => void prepareTree());
    const reset = byId('btn-reset-review');
    if (reset) reset.addEventListener('click', () => {
      if (resetReview('复核状态已清空，请准备当前编辑版本。')) updateControls();
    });
    const seed = byId('btn-pick-seed');
    if (seed) seed.addEventListener('click', enterSeedMode);
    const best = byId('btn-best-candidate');
    if (best) best.addEventListener('click', selectBestCandidate);
    const cancelSeed = byId('btn-cancel-seed');
    if (cancelSeed) cancelSeed.addEventListener('click', () => {
      if (state.pending || state.uncertain) return;
      state.selectionMode = 'NONE';
      state.selectedSeedOriginalIndex = null;
      text('seed-mode-label', '未选择');
      text('seed-selection', '尚未选择原始种子点。');
      if (state.viewer && typeof state.viewer.setPreview === 'function') state.viewer.setPreview([], null);
      updateControls();
    });
    const confirm = byId('btn-confirm');
    if (confirm) confirm.addEventListener('click', () => void confirmReview());
    const identity = byId('identity-confirmed');
    if (identity) identity.addEventListener('change', updateControls);
    window.addEventListener('beforeunload', disposeViewer);
  }

  async function start() {
    if (state.started) return state;
    state.started = true;
    bindControls();
    await loadTrees();
    return state;
  }

  // Tests and the page use this explicit export; there is no implicit submit.
  window.TreeLineDbhReview = {
    start,
    state,
    loadTrees,
    selectTree,
    prepareTree,
    selectCandidate,
    selectBestCandidate,
    enterSeedMode,
    selectSeedIndex,
    confirmReview,
    resetReview,
    disposeViewer,
    methodLabel,
    normalizeDisplay,
    submitMutation,
  };
})();
