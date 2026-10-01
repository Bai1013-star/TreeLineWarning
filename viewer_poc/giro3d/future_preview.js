(() => {
  'use strict';

  const $ = id => document.getElementById(id);
  const state = {
    controller: null,
    requestId: 0,
    data: null,
    horizon: 1,
    view: 'future',
    gateChecking: false,
    loading: false,
  };

  const esc = value => String(value ?? '—').replace(/[&<>"']/g, char => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
  }[char]));
  const value = (item, digits = 2) => {
    if (item === null || item === undefined || item === '') return '—';
    const number = Number(item);
    return Number.isFinite(number) ? number.toFixed(digits) : esc(item);
  };
  const number = item => {
    const result = Number(item);
    return Number.isFinite(result) ? result : null;
  };

  function setStatus(message, kind = '') {
    const node = $('futureStatus');
    node.className = `notice ${kind}`.trim();
    node.textContent = message;
  }

  function clearPlot() {
    const plot = $('futurePlot');
    if (!plot) return;
    if (window.Plotly?.purge) {
      try { window.Plotly.purge(plot); } catch (error) { console.warn('预测图清理失败', error); }
    }
    plot.replaceChildren();
  }

  function clearPreview() {
    state.data = null;
    $('futureControls').classList.add('hidden');
    $('futureBody').classList.add('hidden');
    $('treeRows').replaceChildren();
    $('rowCount').textContent = '';
    $('geometrySource').textContent = '';
    $('geometryNote').textContent = '';
    $('assumptions').textContent = '实验性预测、不替代当前正式勘察结果。此页面不作生物学准确性声明。';
    clearPlot();
  }

  function payloadFrom(response) {
    if (!response || typeof response !== 'object') return null;
    if (response.schema) return response;
    if (response.data && typeof response.data === 'object') return response.data;
    return response;
  }

  function wireSegments(data) {
    return Array.isArray(data?.wires?.segments) ? data.wires.segments : [];
  }

  function safePoints(points) {
    if (!Array.isArray(points)) return [];
    return points.filter(point => Array.isArray(point) && point.length >= 3 && point.slice(0, 3).every(item => number(item) !== null));
  }

  function renderTable(data, view) {
    const future = view === 'future';
    const trees = Array.isArray(data.trees) ? data.trees : [];
    const horizonLabel = future ? `+${Number(data.forecast_horizon_years || state.horizon)} 年` : '当前';
    $('rowCount').textContent = `${trees.length} 棵 · ${horizonLabel}`;
    $('treeRows').innerHTML = trees.map(tree => {
      const displayId = tree.display_tree_id || tree.persistent_tree_id || tree.tree_id || '—';
      const persistentId = tree.persistent_tree_id || '—';
      const height = future ? tree.predicted_height_m : tree.current_height_m;
      const major = future ? tree.predicted_crown_major_m : tree.current_crown_major_m;
      const minor = future ? tree.predicted_crown_minor_m : tree.current_crown_minor_m;
      const dbh = future ? tree.predicted_dbh_cm : tree.current_dbh_cm;
      return `<tr><td title="持久 ID：${esc(persistentId)}"><b>${esc(displayId)}</b><small> · ${esc(persistentId)}</small></td><td>${esc(tree.span_id || '—')}</td><td>${value(height)}</td><td>${value(major)}</td><td>${value(minor)}</td><td>${value(dbh)}</td><td>${value(tree.parent_point_count, 0)}</td><td>${value(tree.display_point_count, 0)}</td></tr>`;
    }).join('') || '<tr><td colspan="8">没有可显示的单木场景数据</td></tr>';
  }

  function treeTrace(data, view, future) {
    const x = [], y = [], z = [], text = [];
    const trees = Array.isArray(data.trees) ? data.trees : [];
    trees.forEach(tree => {
      const points = safePoints(tree[future ? 'future_xyz' : 'current_xyz']);
      const label = tree.display_tree_id || tree.persistent_tree_id || tree.tree_id || '—';
      points.forEach(point => {
        x.push(point[0]); y.push(point[1]); z.push(point[2]); text.push(label);
      });
    });
    if (!x.length) return null;
    return {
      type: 'scatter3d', mode: 'markers',
      name: future ? `+${Number(data.forecast_horizon_years || state.horizon)} 年场景树木` : '当前树木',
      x, y, z, text,
      marker: {size: future ? 3.2 : 2.5, color: future ? '#ffd166' : '#4dd0e1', opacity: future ? .82 : (view === 'future' ? .24 : .76)},
      hovertemplate: `${future ? '场景' : '当前'}树 %{text}<extra></extra>`,
    };
  }

  function renderPlot(data, view) {
    const traces = [];
    const current = treeTrace(data, view, false);
    const future = view === 'future' ? treeTrace(data, view, true) : null;
    if (current) traces.push(current);
    if (future) traces.push(future);
    wireSegments(data).forEach((segment, index) => {
      const points = safePoints(segment.xyz);
      if (points.length < 2) return;
      traces.push({
        type: 'scatter3d', mode: 'lines',
        name: `当前 W1 ${segment.span_id || '—'} · wire ${segment.wire_id ?? '—'} · seg ${segment.segment_index ?? index}`,
        x: points.map(point => point[0]), y: points.map(point => point[1]), z: points.map(point => point[2]),
        line: {width: 5, dash: segment.distance_ready ? 'solid' : 'dot', color: `hsl(${(index * 83) % 360} 78% 68%)`},
        hovertemplate: '当前 W1 持久化曲线<extra></extra>',
      });
    });
    const plot = $('futurePlot');
    if (!window.Plotly?.newPlot) {
      plot.textContent = traces.length ? '本地 Plotly 资源不可用，指标表仍已加载。' : '没有可显示的当前或场景几何。';
      return;
    }
    window.Plotly.newPlot(plot, traces, {
      paper_bgcolor: '#0b121b', plot_bgcolor: '#0b121b', margin: {l: 0, r: 0, t: 20, b: 0},
      scene: {xaxis: {title: 'X'}, yaxis: {title: 'Y'}, zaxis: {title: 'Z'}, aspectmode: 'data', bgcolor: '#0b121b'},
      legend: {font: {color: '#edf7ff'}}, font: {color: '#edf7ff'},
    }, {responsive: true, displaylogo: false});
  }

  function renderAssumptions(data) {
    const assumptions = data.assumptions;
    const text = Array.isArray(assumptions)
      ? assumptions.join('；')
      : (assumptions && typeof assumptions === 'object' ? Object.entries(assumptions).map(([key, item]) => `${key}：${item}`).join('；') : String(assumptions || '使用当前最终几何进行本地场景缩放。'));
    $('assumptions').textContent = `实验性预测、不替代当前正式勘察结果。${text} 此页面不作生物学准确性声明。`;
  }

  function render(data, horizon) {
    const status = String(data.status || '').toUpperCase();
    if (status === 'LOCKED') {
      clearPreview();
      $('futureTitle').textContent = '预测与模拟 🔒';
      setStatus(data.message || '请先完成当前项目的单木复核。', 'locked');
      return;
    }
    if (status !== 'PASS') {
      clearPreview();
      $('futureTitle').textContent = '预测与模拟';
      setStatus(data.message || '本地预测暂时不可用。', 'error');
      return;
    }
    if (!Array.isArray(data.trees) || !data.wires || typeof data.wires !== 'object') {
      clearPreview();
      setStatus('预测接口返回的数据不完整。', 'error');
      return;
    }
    state.data = data;
    state.horizon = Number(data.forecast_horizon_years || horizon);
    state.view = 'future';
    $('futureTitle').textContent = '预测与模拟 ✓';
    $('futureControls').classList.remove('hidden');
    $('futureBody').classList.remove('hidden');
    $('predictionMode').textContent = '✓ 本地基础预测';
    $('modelInfo').textContent = `${data.model_name || 'LocalScenarioPredictor'} · ${data.model_version || '本地版本'} · 不调用外部模型`;
    setStatus(data.message || `本地基础预测已加载：+${state.horizon} 年场景。`, '');
    renderView('future');
  }

  function renderView(view) {
    if (!state.data) return;
    state.view = view;
    const future = view === 'future';
    document.querySelectorAll('[data-horizon], [data-view="current"]').forEach(button => {
      const active = button.dataset.view === 'current' ? !future : future && Number(button.dataset.horizon) === state.horizon;
      button.classList.toggle('active', active);
      button.setAttribute('aria-pressed', active ? 'true' : 'false');
    });
    const label = future ? `+${state.horizon} 年场景` : '当前几何';
    $('scenarioStatus').textContent = label;
    renderTable(state.data, view);
    renderPlot(state.data, view);
    renderAssumptions(state.data);
    const wireCount = wireSegments(state.data).length;
    const treePoints = (state.data.trees || []).reduce((total, tree) => total + safePoints(tree[future ? 'future_xyz' : 'current_xyz']).length, 0);
    $('geometrySource').textContent = `${treePoints.toLocaleString()} ${future ? '场景' : '当前'}点 · ${wireCount} 条当前 W1`;
    $('geometryNote').textContent = future
      ? '当前树木以浅色叠加，黄色点为本地场景预测；导线保持当前最终 W1。未来净空和未来导线尚未计算。'
      : '当前视图显示当前最终树木几何与当前持久化 W1 导线；没有使用未来预测替代当前结果。';
  }

  async function loadPreview(horizon) {
    const requested = Number(horizon);
    if (![1, 3, 5].includes(requested)) return;
    const requestId = ++state.requestId;
    state.controller?.abort();
    state.controller = typeof AbortController === 'function' ? new AbortController() : null;
    const controller = state.controller;
    const timeout = setTimeout(() => controller?.abort('timeout'), 180000);
    state.loading = true;
    clearPreview();
    setStatus(`正在读取 +${requested} 年本地场景…`, 'loading');
    try {
      const response = await fetch(`/api/future-preview?horizon=${requested}`, {
        cache: 'no-store', signal: state.controller?.signal,
      });
      const payload = payloadFrom(await response.json());
      if (requestId !== state.requestId) return;
      if (!response.ok) throw new Error(payload?.message || payload?.error?.message || '预测接口请求失败');
      render(payload, requested);
    } catch (error) {
      if (requestId !== state.requestId) return;
      clearPreview();
      setStatus(controller?.signal.reason === 'timeout' ? '预测读取超时，请返回项目检查后重试。' : `预测读取失败：${error?.message || error}`, 'error');
    } finally {
      clearTimeout(timeout);
      if (requestId === state.requestId) {
        state.controller = null;
        state.loading = false;
      }
    }
  }

  function sameGate(left, right) {
    const fields = ['run_id', 'batch_mtime_ns', 'journal_mtime_ns', 'prediction_mtime_ns',
      'editor_operation_count', 'published_editor_operation_count', 'final_generation_id',
      'result_generation_id', 'risk_generation_id'];
    return right?.status === 'PASS' && fields.every(field => left?.[field] === right[field]);
  }

  async function checkGate() {
    if (state.gateChecking || state.loading) return false;
    state.gateChecking = true;
    const previous = state.data;
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 15000);
    try {
      const response = await fetch('/api/future-preview/gate', {cache: 'no-store', signal: controller.signal});
      const payload = await response.json();
      if (state.data !== previous) return false;
      if (!response.ok || payload.gate?.status !== 'PASS') {
        if (state.data) clearPreview();
        $('futureTitle').textContent = '预测与模拟 🔒';
        setStatus(`当前结果尚未定稿，预测暂时隐藏。${(payload.gate?.reasons || []).join('；')}`, 'locked');
        return false;
      }
      if (!previous || !sameGate(previous.gate, payload.gate)) {
        await loadPreview(state.horizon);
        return false;
      }
      return true;
    } catch (_) {
      if (state.data === previous && state.data) {
        clearPreview();
      }
      setStatus('无法确认当前定稿状态，已清除预览；连接恢复后自动刷新。', 'error');
      return false;
    } finally {
      clearTimeout(timeout);
      state.gateChecking = false;
    }
  }

  $('currentViewButton').addEventListener('click', async () => { if (await checkGate()) renderView('current'); });
  document.querySelectorAll('[data-horizon]').forEach(button => button.addEventListener('click', () => loadPreview(button.dataset.horizon)));
  window.addEventListener('resize', () => {
    if (state.data && window.Plotly?.Plots?.resize) window.Plotly.Plots.resize($('futurePlot'));
  });
  setInterval(() => { void checkGate(); }, 3000);
  loadPreview(1);
})();
