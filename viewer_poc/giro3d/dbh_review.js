(() => {
  'use strict';

  // Diameter is derived from each tree's current point-cloud evidence.
  const POLL_INTERVAL_MS = 700;
  const MUTATION_TIMEOUT_MS = 600000;
  const REQUEST_TIMEOUT_MS = 15000;
  const AXIS_COLORS = ['#55d4e4', '#ffcc47', '#c18cff', '#ff8a65', '#82d394', '#7aa7ff'];
  let requestSequence = 0;

  const state = {
    trees: [],
    treeMap: new Map(),
    activeTreeId: null,
    review: null,
    selectedCandidateId: null,
    selectedSeedOriginalIndex: null,
    selectedGroundOriginalIndex: null,
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
    const number = Number(value);
    return Number.isFinite(number) ? number : fallback;
  }

  function esc(value) {
    return String(value ?? '').replace(/[&<>"']/g, character => ({
      '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
    }[character]));
  }

  function displayId(tree) {
    return tree.display_tree_id || tree.persistent_tree_id || tree.global_tree_id || '未命名单木';
  }

  function isReviewable(tree) {
    const method = String(tree && tree.dbh_method || '').toUpperCase();
    return method === 'UNAVAILABLE' || method === 'REFERENCE_ESTIMATE';
  }

  function methodLabel(method) {
    const labels = {
      REVIEW_CONFIRMED_STEM: '人工确认点云',
      RECOVERED_STEM: '恢复点云',
      DIRECT_STEM: '点云',
      ALLOMETRIC_ESTIMATE: '估算',
      REFERENCE_ESTIMATE: '旧参考估算（已失效）',
      UNAVAILABLE: '不可用',
    };
    return labels[String(method || '')] || String(method || '—');
  }

  function fitReason(status) {
    const reasons = {
      NO_CROWN_TO_STEM_IDENTITY: '未找到与当前树冠可靠对应的树干',
      NO_ABSOLUTE_CANDIDATE: '树干证据不足，未通过自动确认',
      TOP_CANDIDATE_CONFLICT: '候选树干同时匹配多棵树，需人工复核归属',
      EDITOR_REMOVED_BH_EVIDENCE: '胸高处点已由人工删点操作剔除',
      INCOMPLETE_WIRE_RESULTS: '缺少完整证据',
      REFERENCE_REQUIRED: '本树点云证据不足，无法可靠计算胸径',
      FOREIGN_STEM_CLAIM_NOT_ASSIGNABLE: '树干已被其他树占用，无法自动归属',
      SELECTED_SOURCE_OWNERSHIP_CONFLICT: '树干来源存在归属冲突',
      LOCAL_BASE_HEIGHT_AMBIGUOUS: '本树根部有多个可能地面高度，需人工确认树根地面',
      LOCAL_BASE_HEIGHT_UNVERIFIED: '本树根部地面尚未确认，需指定本树地面点',
      MULTIPLE_STEM_AXES_REVIEW_REQUIRED: '本树有多根候选树干轴，需选择对应的一根',
      PER_TREE_STEM_EVIDENCE_REQUIRED: '缺少本树可核验的树干和胸高点证据，需逐树复核',
      SEARCH_BUDGET_EXHAUSTED: '树干原始点搜索达到预算上限，需缩小范围或指定树干证据',
      PASS: '几何质量检查通过',
    };
    const selected = String(status || '');
    return reasons[selected] || (selected.startsWith('SEARCH_BUDGET') ? reasons.SEARCH_BUDGET_EXHAUSTED : (status ? '证据不足，需人工复核' : ''));
  }

  function treeDbhLabel(tree) {
    if (tree.dbh_method === 'REFERENCE_ESTIMATE') {
      const reason = tree.dbh_fit_status || tree.dbh_pre_reference_failure_reason;
      return `未测得 · 旧参考估算已失效${reason ? ` · ${fitReason(reason)}` : ''}，请逐树复核`;
    }
    if (tree.dbh_cm == null) return `未测得 · ${fitReason(tree.dbh_fit_status)}`;
    const estimated = tree.dbh_quality === 'ESTIMATE' || tree.dbh_method === 'ALLOMETRIC_ESTIMATE';
    const model = String(tree.dbh_model_version || '');
    const source = model.includes('UPPER_PARTIAL_ARC') ? '本树胸高点云拟合（单侧估算）'
      : model.includes('UPPER_CIRCULAR') ? '本树上部树干外推估算'
      : estimated ? '本树点云估算' : methodLabel(tree.dbh_method);
    const uncertainty = tree.dbh_uncertainty_cm == null ? '' : ` · ±${Number(tree.dbh_uncertainty_cm).toFixed(1)} cm`;
    return `${Number(tree.dbh_cm).toFixed(1)} cm · ${source}${uncertainty}`;
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
    const selectableCount = state.trees.filter(isReviewable).length;
    text('tree-count', `${selectableCount} 可复核`);
    if (!state.trees.length) {
      text('tree-list-note', '当前没有可读取的单木复核记录。');
      return;
    }
    if (!selectableCount) text('tree-list-note', '当前没有胸径缺失的单木。');
    state.trees.forEach(tree => {
      const button = document.createElement('button');
      const selectable = true;
      const reviewable = isReviewable(tree);
      const treeId = String(tree.persistent_tree_id ?? tree.global_tree_id ?? '');
      button.type = 'button';
      button.className = `tree-entry${state.activeTreeId === treeId ? ' active' : ''}${reviewable ? '' : ' measured'}`;
      button.dataset.treeId = treeId;
      button.disabled = !selectable || Boolean(state.pending) || state.uncertain;
      button.innerHTML = `<span class="tree-entry-title">${esc(displayId(tree))}</span><span class="tree-entry-meta">${esc(treeDbhLabel(tree))}</span>`;
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
    text('review-tree-meta', `树木 ${displayId(tree)} · ${treeDbhLabel(tree)}`);
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
    state.selectedGroundOriginalIndex = null;
    state.selectionMode = 'NONE';
    const checkbox = byId('identity-confirmed');
    if (checkbox) checkbox.checked = false;
    text('seed-mode-label', '未选择');
    text('seed-selection', '尚未选择原始种子点。');
    text('ground-mode-label', '未选择');
    text('ground-selection', '未指定地面点；使用原始地面高程。');
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
    const groundSelectable = display && Array.isArray(display.ground_selectable)
      ? display.ground_selectable : (xyz ? xyz.map(() => false) : null);
    if (!xyz || !originalIndex || !hag || !selectable || !groundSelectable || new Set([xyz.length, originalIndex.length, hag.length, selectable.length, groundSelectable.length]).size !== 1) {
      throw new Error('复核显示数据的 xyz、original_index、hag_m、selectable、ground_selectable 未对齐');
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
    if (groundSelectable.some(value => typeof value !== 'boolean')) throw new Error('复核显示数据包含无效 ground_selectable 标志');
    return {xyz, originalIndex: indices, hag: hag.map(Number), selectable: selectable.map(Boolean), groundSelectable};
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
    const colors = normalized.selectable.map((value, row) => value ? '#63d471'
      : normalized.groundSelectable[row] ? '#42d9e8' : '#6f8192');
    const published = review.published_measurement || {};
    const evidence = Array.isArray(published.evidence_xyz) ? published.evidence_xyz.filter(point =>
      Array.isArray(point) && point.length >= 3 && point.slice(0, 3).every(value => Number.isFinite(Number(value)))) : [];
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
      danger_points: {xyz: evidence},
    };
  }

  function initViewer() {
    if (state.viewer) return state.viewer;
    const constructor = window.TreeLineTreeReview3D;
    if (typeof constructor !== 'function') throw new Error('本地 Giro3D / Three.js 资源不可用');
    state.viewer = new constructor(byId('dbh-plot3d'), byId('dbh-tree-label-overlay'), {
      onTreePick: () => true,
      onPointPick: hit => {
        if (state.pending || state.uncertain || !state.rendererReady || !['SEED', 'GROUND'].includes(state.selectionMode)) return true;
        const originalIndex = state.viewerIndexById.get(String(Number(hit && hit.treeId)));
        if (originalIndex == null) return true;
        if (state.selectionMode === 'SEED') selectSeedIndex(originalIndex);
        else selectGroundIndex(originalIndex);
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
    state.selectedGroundOriginalIndex = null;
    state.selectionMode = 'NONE';
    state.completed = false;
    state.rendererReady = false;
    const candidates = Array.isArray(review.candidates) ? review.candidates : [];
    const published = review.published_measurement || null;
    const wideSearch = review.wide_raw_stem_search || null;
    const viewOnly = review.confirmation_allowed === false;
    hidden('identity-confirm-label', viewOnly);
    hidden('btn-confirm', viewOnly);
    renderCandidateCards(candidates);
    hidden('review-workspace', false);
    const crownLabel = review.current_crown && review.current_crown.label
      ? review.current_crown.label : '未提供标签';
    const evidenceHag = Array.isArray(published?.evidence_hag_m) ? published.evidence_hag_m : [];
    const breastHeightCount = evidenceHag.filter(value => Number.isFinite(Number(value)) && Math.abs(Number(value) - 1.3) <= 0.2).length;
    text('dbh-context-label', viewOnly
      ? `当前树冠上下文：${crownLabel} · 已发布 ${finite(published?.dbh_cm, '—')} cm（${methodLabel(published?.dbh_method)}）；红色点为测量证据，胸高 1.3±0.2 m 切片 ${breastHeightCount} 点`
      : `当前树冠上下文：${crownLabel}`);
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
      ground_selectable_point_count: normalized.groundSelectable.filter(Boolean).length,
      wide_raw_stem_search: wideSearch,
      display_original_index_sample: sample(normalized.originalIndex),
      display_hag_m_sample: sample(normalized.hag),
      published_measurement: published ? {
        dbh_cm: published.dbh_cm,
        dbh_method: published.dbh_method,
        dbh_quality: published.dbh_quality,
        dbh_fit_status: published.dbh_fit_status,
        evidence_point_count: Array.isArray(published.evidence_xyz) ? published.evidence_xyz.length : 0,
        breast_height_slice_point_count: breastHeightCount,
      } : null,
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
      const searchLimited = wideSearch && wideSearch.status && String(wideSearch.status) !== 'PASS';
      const baseStatus = viewOnly
        ? '已加载当前树的点云与已发布胸径证据；此页面仅供查看。'
        : '点云已准备。请选择候选轴，或进入种子点模式后点击可选原始点。';
      setReviewStatus(searchLimited
        ? `${baseStatus} ${wideSearch.message || `宽范围树干搜索未完成（${wideSearch.status}）；请在当前原始点中人工选择树干。`}`
        : baseStatus, searchLimited ? 'warn' : '');
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

  function refreshPickedPointColors() {
    if (!state.viewer || !state.review) return;
    const display = normalizeDisplay(state.review);
    const seed = state.selectedSeedOriginalIndex;
    const ground = state.selectedGroundOriginalIndex;
    if (typeof state.viewer.setSelection === 'function') state.viewer.setSelection(seed == null ? [] : [Number(seed)]);
    const groundRow = ground == null ? -1 : display.originalIndex.indexOf(Number(ground));
    if (typeof state.viewer.setPreview === 'function') state.viewer.setPreview(
      groundRow < 0 ? [] : [display.xyz[groundRow]], groundRow < 0 ? null : Number(ground)
    );
  }

  function updateControls() {
    const pending = Boolean(state.pending) || state.uncertain;
    const hasSelection = state.selectedCandidateId != null || state.selectedSeedOriginalIndex != null;
    const checked = Boolean(byId('identity-confirmed') && byId('identity-confirmed').checked);
    const prepare = byId('btn-prepare');
    if (prepare) prepare.disabled = pending || !selectedTree() || Boolean(state.review);
    const recompute = byId('btn-recompute');
    if (recompute) recompute.disabled = pending;
    const reset = byId('btn-reset-review');
    if (reset) reset.disabled = pending || !state.review;
    const seed = byId('btn-pick-seed');
    if (seed) seed.disabled = pending || !state.review || !state.rendererReady || state.completed || state.review.confirmation_allowed === false;
    const ground = byId('btn-pick-ground');
    if (ground) ground.disabled = pending || !state.review || !state.rendererReady || state.completed || state.review.confirmation_allowed === false
      || !normalizeDisplay(state.review).groundSelectable.some(Boolean);
    const cancelGround = byId('btn-cancel-ground');
    if (cancelGround) cancelGround.disabled = pending || state.selectedGroundOriginalIndex == null;
    const cancelSeed = byId('btn-cancel-seed');
    if (cancelSeed) cancelSeed.disabled = pending || state.selectionMode !== 'SEED';
    const confirm = byId('btn-confirm');
    if (confirm) confirm.disabled = pending || state.completed || !state.review || !state.rendererReady || !hasSelection || !checked || !isReviewable(selectedTree()) || state.review.confirmation_allowed === false;
    document.querySelectorAll('#tree-list .tree-entry').forEach(node => {
      const tree = state.treeMap.get(String(node.dataset.treeId));
      node.disabled = pending;
    });
    document.querySelectorAll('#axis-candidates button').forEach(node => {
      node.disabled = pending || state.uncertain || !state.rendererReady || state.completed;
    });
    const seedLabel = state.selectionMode === 'SEED' ? '点选中' : state.selectedCandidateId ? '候选轴已选' : '未选择';
    text('seed-mode-label', seedLabel);
    text('ground-mode-label', state.selectionMode === 'GROUND' ? '点选中'
      : state.selectedGroundOriginalIndex != null ? '地面点已选' : '未选择');
  }

  function selectTree(treeId) {
    if (state.pending || state.uncertain) return;
    const tree = state.treeMap.get(String(treeId));
    if (!tree) return;
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
    refreshPickedPointColors();
    renderAxisOverlay();
    renderCandidateCards(state.review.candidates || []);
    text('point-detail', '');
    text('seed-selection', '已选择候选轴；未选择种子点。');
    setReviewStatus(`已选择候选轴 ${candidate.candidate_id}。请阅读诊断并勾选身份确认。`);
    updateControls();
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

  function enterGroundMode() {
    if (!state.review || state.pending || state.uncertain || !state.rendererReady || state.completed) return;
    state.selectionMode = 'GROUND';
    setReviewStatus('树根地面点模式已开启。请点击青色当前原始点；后台会检验地面支持和树干距离。');
    text('ground-selection', '等待点击可选树根地面点…');
    updateControls();
  }

  function selectGroundIndex(index) {
    if (!state.review || state.pending || state.uncertain || !state.rendererReady || state.completed || state.selectionMode !== 'GROUND') return;
    const display = normalizeDisplay(state.review);
    const row = display.originalIndex.indexOf(Number(index));
    if (row < 0 || !display.groundSelectable[row]) {
      setReviewStatus('该原始点不是可选树根地面点，请选择青色点。', 'warn');
      return;
    }
    state.selectedGroundOriginalIndex = display.originalIndex[row];
    const point = display.xyz[row];
    refreshPickedPointColors();
    text('ground-selection', `已选择树根地面点 original_index=${state.selectedGroundOriginalIndex} · XYZ=${point.slice(0, 3).join(', ')} · HAG=${display.hag[row]} m`);
    setReviewStatus('已选择精确原始树根地面点；提交后后台会校验其几何支持。');
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
    refreshPickedPointColors();
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
      const phaseLabel = pending.phase === 'prepare' ? '正在准备点云证据'
        : pending.phase === 'recompute' ? '正在重新计算胸径' : '正在提交人工确认';
      setReviewStatus(`${phaseLabel}（已等待 ${elapsed} 秒）`);
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
      if (!state.trees.some(isReviewable)) text('tree-list-note', '当前没有胸径缺失的单木。');
      return payload;
    } catch (error) {
      text('tree-count', '读取失败');
      text('tree-list-note', `读取待复核单木失败：${error.message || error}`);
      setReviewStatus('无法读取待复核单木。', 'error');
      return null;
    }
  }

  async function recomputeDbh() {
    if (state.pending || state.uncertain) return null;
    text('recompute-status', '正在按当前定稿结果重新计算胸径，请稍候…');
    let outcome;
    try {
      outcome = await submitMutation('/api/dbh-review/recompute', {}, 'recompute');
    } catch (error) {
      text('recompute-status', `胸径重算失败：${error.message || error}`);
      return null;
    }
    if (!outcome) return null;
    resetReview();
    await loadTrees();
    text('recompute-status', `已重算 ${outcome.tree_count} 棵树，${outcome.dbh_available_count ?? 0} 棵有逐树胸径；其余保留空值和具体原因。`);
    if (window.parent && window.parent !== window) {
      window.parent.postMessage({type: 'treeline:tree-review-changed', kind: 'DBH_RECOMPUTED'}, window.location.origin);
    }
    return outcome;
  }

  async function prepareTree(treeId = state.activeTreeId) {
    if (state.pending || state.uncertain) return state.pendingPromise;
    const tree = state.treeMap.get(String(treeId));
    if (!tree) return null;
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
      return {candidate_id: state.selectedCandidateId,
        ...(state.selectedGroundOriginalIndex == null ? {} : {ground_original_index: state.selectedGroundOriginalIndex})};
    }
    if (state.selectedCandidateId == null && state.selectedSeedOriginalIndex != null) {
      return {seed_original_index: state.selectedSeedOriginalIndex,
        ...(state.selectedGroundOriginalIndex == null ? {} : {ground_original_index: state.selectedGroundOriginalIndex})};
    }
    return null;
  }

  async function confirmReview() {
    if (state.pending || state.uncertain || state.completed || !state.review || !isReviewable(selectedTree()) || state.review.confirmation_allowed === false) return null;
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
        tree.dbh_fit_status = terminal.dbh_fit_status;
        tree.dbh_model_version = terminal.dbh_model_version || tree.dbh_model_version;
      }
      renderTreeList();
      setReviewStatus(`已恢复胸径 ${finite(terminal.dbh_cm, '—')} cm · ${methodLabel(terminal.dbh_method || 'REVIEW_CONFIRMED_STEM')}`, '');
      setConfirmMessage(`人工身份已确认，${fitReason(terminal.dbh_fit_status || 'PASS')}。`, 'ok');
    } else {
      state.completed = true;
      setReviewStatus('该树干点云不足，无法可靠计算胸径', 'warn');
      setConfirmMessage(`${fitReason(terminal.dbh_fit_status)}。未写入手工胸径。`, 'warn');
    }
    if (window.parent && window.parent !== window) {
      window.parent.postMessage({type: 'treeline:tree-review-changed', kind: 'DBH_CONFIRMED'}, window.location.origin);
    }
    updateControls();
    return terminal;
  }

  function bindControls() {
    const recompute = byId('btn-recompute');
    if (recompute) recompute.addEventListener('click', () => void recomputeDbh());
    const prepare = byId('btn-prepare');
    if (prepare) prepare.addEventListener('click', () => void prepareTree());
    const reset = byId('btn-reset-review');
    if (reset) reset.addEventListener('click', () => {
      if (resetReview('复核状态已清空，请准备当前编辑版本。')) updateControls();
    });
    const seed = byId('btn-pick-seed');
    if (seed) seed.addEventListener('click', enterSeedMode);
    const ground = byId('btn-pick-ground');
    if (ground) ground.addEventListener('click', enterGroundMode);
    const cancelGround = byId('btn-cancel-ground');
    if (cancelGround) cancelGround.addEventListener('click', () => {
      if (state.pending || state.uncertain) return;
      state.selectedGroundOriginalIndex = null;
      if (state.selectionMode === 'GROUND') state.selectionMode = 'NONE';
      text('ground-selection', '未指定地面点；使用原始地面高程。');
      refreshPickedPointColors();
      updateControls();
    });
    const cancelSeed = byId('btn-cancel-seed');
    if (cancelSeed) cancelSeed.addEventListener('click', () => {
      if (state.pending || state.uncertain) return;
      state.selectionMode = 'NONE';
      state.selectedSeedOriginalIndex = null;
      text('seed-mode-label', '未选择');
      text('seed-selection', '尚未选择原始种子点。');
      refreshPickedPointColors();
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
    enterSeedMode,
    selectSeedIndex,
    enterGroundMode,
    selectGroundIndex,
    confirmReview,
    recomputeDbh,
    resetReview,
    disposeViewer,
    methodLabel,
    fitReason,
    normalizeDisplay,
    submitMutation,
  };
})();
