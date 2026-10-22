(() => {
  'use strict';

  const $ = id => document.getElementById(id);
  const state = {
    controller: null,
    requestId: 0,
    data: null,
    horizonValue: 1,
    horizonUnit: 'days',
    view: 'future',
    gateChecking: false,
    sliderTimer: null,
    maxData: null,
    busyRetryCount: 0,
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
    if (item === null || item === undefined || item === '') return null;
    const result = Number(item);
    return Number.isFinite(result) ? result : null;
  };
  const unitText = unit => ({days: '天', months: '月', years: '年'}[unit] || unit);
  const horizonText = data => `+${Number(data.forecast_horizon_value || state.horizonValue)} ${unitText(data.forecast_horizon_unit || state.horizonUnit)}`;
  const dayCount = data => Number(data?.trees?.[0]?.forecast_horizon_days || (data?.forecast_horizon_unit === 'days' ? data.forecast_horizon_value : 1));
  const dateText = iso => iso ? new Date(`${iso}T00:00:00`).toLocaleDateString('zh-CN') : '5 年内未达到';

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
    const horizonLabel = future ? horizonText(data) : '当前';
    $('rowCount').textContent = `${trees.length} 棵 · ${horizonLabel}`;
    $('treeRows').innerHTML = trees.map(tree => {
      const displayId = tree.display_tree_id || tree.persistent_tree_id || tree.tree_id || '—';
      const persistentId = tree.persistent_tree_id || '—';
      const height = future ? tree.predicted_height_m : tree.current_height_m;
      const major = future ? tree.predicted_crown_major_m : tree.current_crown_major_m;
      const minor = future ? tree.predicted_crown_minor_m : tree.current_crown_minor_m;
      const dbh = future ? tree.predicted_dbh_cm : tree.current_dbh_cm;
      const clearance = future ? tree.predicted_clearance_envelope_m : tree.current_minimum_clearance_m;
      return `<tr><td title="持久 ID：${esc(persistentId)}"><b>${esc(displayId)}</b><small> · ${esc(persistentId)}</small></td><td>${esc(tree.species || '其他')} / ${value(tree.tree_age,1)} 年</td><td>${esc(tree.target_date || '—')}</td><td>${value(height)}</td><td>${value(major)}</td><td>${value(minor)}</td><td>${value(dbh)}</td><td class="${future&&tree.risk_state==='STANDARD_EXCEEDED'?'exceeded':''}">${value(clearance)}</td><td>${value(tree.display_point_count, 0)}</td></tr>`;
    }).join('') || '<tr><td colspan="9">没有可显示的单木场景数据</td></tr>';
  }

  function treeTrace(data, view, future) {
    const x = [], y = [], z = [], text = [], colors = [];
    const trees = Array.isArray(data.trees) ? data.trees : [];
    trees.forEach(tree => {
      const points = safePoints(tree[future ? 'future_xyz' : 'current_xyz']);
      const label = tree.display_tree_id || tree.persistent_tree_id || tree.tree_id || '—';
      points.forEach(point => {
        x.push(point[0]); y.push(point[1]); z.push(point[2]); text.push(label);
        colors.push(future && tree.risk_state === 'STANDARD_EXCEEDED' ? '#ff5f65' : (future ? '#ffd166' : '#4dd0e1'));
      });
    });
    if (!x.length) return null;
    return {
      type: 'scatter3d', mode: 'markers',
      name: future ? `${horizonText(data)} 预测孪生` : '当前实测树木',
      x, y, z, text,
      marker: {size: future ? 3.2 : 2.5, color: colors, opacity: future ? .82 : (view === 'future' ? .24 : .76)},
      hovertemplate: `${future ? '场景' : '当前'}树 %{text}<extra></extra>`,
    };
  }

  function trajectoryTree(data) {
    const trees = Array.isArray(data?.trees) ? data.trees : [];
    return [...trees].sort((left, right) => {
      const ld = left.trajectory?.standard_crossing_day ?? Number.MAX_SAFE_INTEGER;
      const rd = right.trajectory?.standard_crossing_day ?? Number.MAX_SAFE_INTEGER;
      if (ld !== rd) return ld - rd;
      return (left.predicted_clearance_envelope_m ?? Number.MAX_VALUE) - (right.predicted_clearance_envelope_m ?? Number.MAX_VALUE);
    })[0] || null;
  }

  function renderTrend(data) {
    const tree = trajectoryTree(data), growthPlot = $('growthPlot'), clearancePlot = $('clearancePlot');
    if (!tree || !Array.isArray(tree.trajectory?.points)) {
      growthPlot.textContent = '当前没有可绘制的树木生长轨迹。';
      clearancePlot.textContent = '当前没有可绘制的树线净空轨迹。';
      return;
    }
    const points = tree.trajectory.points, threshold = Number(tree.trajectory.warning_threshold_m ?? data.warning_threshold_m ?? .5);
    const x = points.map(item => item.date), clearance = points.map(item => item.clearance_m);
    const safe = clearance.map(value => value === null || value <= threshold ? null : value);
    const exceeded = clearance.map(value => value === null || value > threshold ? null : value);
    const selected = tree.target_date;
    $('trendTree').textContent = `${tree.display_tree_id || tree.persistent_tree_id} · 5 年趋势`;
    if (!window.Plotly?.newPlot) {
      growthPlot.textContent = '本地 Plotly 资源不可用。';
      clearancePlot.textContent = '本地 Plotly 资源不可用。';
      return;
    }
    const plot = window.Plotly.react || window.Plotly.newPlot;
    plot.call(window.Plotly, growthPlot, [
      {type:'scatter',mode:'lines',name:'树高',x,y:points.map(item=>item.height_m),line:{color:'#81c784',width:2}},
      {type:'scatter',mode:'lines',name:'冠幅',x,y:points.map(item=>item.crown_major_m),line:{color:'#ffd166',width:2}},
      {type:'scatter',mode:'lines',name:'胸径(cm)',x,y:points.map(item=>item.dbh_cm),yaxis:'y2',line:{color:'#b78cff',width:2}},
    ], {
      paper_bgcolor:'#0b121b',plot_bgcolor:'#0b121b',font:{color:'#edf7ff'},margin:{l:62,r:70,t:22,b:48},
      xaxis:{title:'日期',gridcolor:'rgba(179,229,252,.08)'},
      yaxis:{title:'树高 / 冠幅 (m)',gridcolor:'rgba(179,229,252,.08)'},
      yaxis2:{title:'胸径 (cm)',overlaying:'y',side:'right',showgrid:false},
      legend:{orientation:'h'},
      shapes:[{type:'line',xref:'x',x0:selected,x1:selected,yref:'paper',y0:0,y1:1,line:{color:'#fff',width:1,dash:'dot'}}],
    }, {responsive:true,displaylogo:false});
    plot.call(window.Plotly, clearancePlot, [
      {type:'scatter',mode:'lines',name:'净空（未越标）',x,y:safe,line:{color:'#4dd0e1',width:3},connectgaps:false},
      {type:'scatter',mode:'lines',name:'净空（越过标准）',x,y:exceeded,line:{color:'#ff5f65',width:4},connectgaps:false},
    ], {
      paper_bgcolor:'#0b121b',plot_bgcolor:'#0b121b',font:{color:'#edf7ff'},margin:{l:62,r:36,t:22,b:48},
      xaxis:{title:'日期',gridcolor:'rgba(179,229,252,.08)'},
      yaxis:{title:'净空 (m)',gridcolor:'rgba(179,229,252,.08)'},
      legend:{orientation:'h'},
      shapes:[
        {type:'line',xref:'paper',x0:0,x1:1,yref:'y',y0:threshold,y1:threshold,line:{color:'#ff9d5c',width:2,dash:'dash'}},
        {type:'line',xref:'x',x0:selected,x1:selected,yref:'paper',y0:0,y1:1,line:{color:'#fff',width:1,dash:'dot'}},
      ],
      annotations:[{xref:'paper',x:1,xanchor:'right',yref:'y',y:threshold,text:`项目净空标准 ${threshold.toFixed(2)} m`,showarrow:false,font:{color:'#ff9d5c'}}],
    }, {responsive:true,displaylogo:false});
  }

  function renderKpis(data) {
    const trees = Array.isArray(data.trees) ? data.trees : [];
    const currentDay = dayCount(data);
    const selectedClearances = trees.map(tree => tree.predicted_clearance_envelope_m).filter(value => Number.isFinite(Number(value))).map(Number);
    const earliestStandard = [...trees].filter(tree => tree.trajectory?.standard_crossing_day !== null && tree.trajectory?.standard_crossing_day !== undefined).sort((a,b)=>a.trajectory.standard_crossing_day-b.trajectory.standard_crossing_day)[0];
    const earliestContact = [...trees].filter(tree => tree.trajectory?.contact_day !== null && tree.trajectory?.contact_day !== undefined).sort((a,b)=>a.trajectory.contact_day-b.trajectory.contact_day)[0];
    $('selectedDate').textContent = data.trees?.[0]?.target_date || '—';
    $('selectedClearance').textContent = selectedClearances.length ? `${Math.min(...selectedClearances).toFixed(3)} m` : '无法估计';
    $('selectedClearance').classList.toggle('exceeded', selectedClearances.length && Math.min(...selectedClearances) <= Number(data.warning_threshold_m ?? .5));
    $('standardCrossing').textContent = earliestStandard ? `${dateText(earliestStandard.trajectory.standard_crossing_date)} · ${earliestStandard.display_tree_id || earliestStandard.persistent_tree_id}` : '5 年内未越过';
    $('contactTime').textContent = earliestContact ? `${dateText(earliestContact.trajectory.contact_date)} · ${earliestContact.display_tree_id || earliestContact.persistent_tree_id}` : '5 年内未触线';
    $('timelineSlider').value = String(Math.max(1, Math.min(1826, currentDay)));
    $('timelineValue').textContent = `第 ${currentDay} 天 · ${data.trees?.[0]?.target_date || ''}`;
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
    (window.Plotly.react || window.Plotly.newPlot).call(window.Plotly, plot, traces, {
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

  function render(data, requestedValue, requestedUnit) {
    const status = String(data.status || '').toUpperCase();
    if (status === 'LOCKED') {
      clearPreview();
      $('futureTitle').textContent = '树木数字孪生趋势 🔒';
      setStatus(data.message || '请先完成点云分析并确认至少一棵树。', 'locked');
      return;
    }
    if (status !== 'PASS') {
      clearPreview();
      $('futureTitle').textContent = '树木数字孪生趋势';
      setStatus(data.message || '本地预测暂时不可用。', 'error');
      return;
    }
    if (!Array.isArray(data.trees) || !data.wires || typeof data.wires !== 'object') {
      clearPreview();
      setStatus('预测接口返回的数据不完整。', 'error');
      return;
    }
    state.data = data;
    if (Number(data.forecast_horizon_days || data?.trees?.[0]?.forecast_horizon_days) >= 1825) state.maxData = data;
    state.horizonValue = Number(data.forecast_horizon_value || requestedValue);
    state.horizonUnit = data.forecast_horizon_unit || requestedUnit;
    state.view = 'future';
    $('futureTitle').textContent = '树木数字孪生趋势 ✓';
    $('futureControls').classList.remove('hidden');
    $('futureBody').classList.remove('hidden');
    $('predictionMode').textContent = '✓ 季节非刚性点云孪生';
    $('modelInfo').textContent = `${data.model_name || 'seasonal-nonrigid-tree-twin'} · ${data.model_version || '本地版本'} · 天/月/年逐日季节积分`;
    $('futureClearanceStatus').textContent = data.future_clearance_status === 'BOUNDED_DISPLAY_POINT_ESTIMATE' ? '未来净空：显示点趋势估计' : '未来净空：当前无法估计';
    setStatus(data.message || `预测孪生已加载：${horizonText(data)}。`, '');
    renderView('future');
  }

  function localScenarioForDay(day) {
    const source = state.maxData;
    if (!source) return null;
    const targetDay = Math.max(0, Math.min(1826, Math.round(Number(day) || 0)));
    const trees = (source.trees || []).map(tree => {
      const points = tree.trajectory?.points || [];
      let lower = points[0] || null, upper = points.at(-1) || lower;
      for (const item of points) {
        if (Number(item.day) <= targetDay) lower = item;
        if (Number(item.day) >= targetDay) { upper = item; break; }
      }
      const lowerDay = Number(lower?.day ?? 0), upperDay = Number(upper?.day ?? lowerDay);
      const alpha = upperDay > lowerDay ? (targetDay - lowerDay) / (upperDay - lowerDay) : 0;
      const lerp = key => {
        const left = number(lower?.[key]), right = number(upper?.[key]);
        if (!Number.isFinite(left)) return Number.isFinite(right) ? right : null;
        return Number.isFinite(right) ? left + (right - left) * alpha : left;
      };
      const startDate = points[0]?.date ? new Date(`${points[0].date}T00:00:00Z`) : null;
      if (startDate && !Number.isNaN(startDate.getTime())) startDate.setUTCDate(startDate.getUTCDate() + targetDay);
      const metric = {
        day: targetDay,
        date: startDate && !Number.isNaN(startDate.getTime()) ? startDate.toISOString().slice(0, 10) : (lower?.date || tree.target_date),
        seasonal_growth_years: lerp('seasonal_growth_years'),
        height_m: lerp('height_m'),
        crown_major_m: lerp('crown_major_m'),
        crown_minor_m: lerp('crown_minor_m'),
        dbh_cm: lerp('dbh_cm'),
        clearance_m: lerp('clearance_m'),
      };
      const last = points.at(-1) || metric;
      const seasonal = Number(metric?.seasonal_growth_years || 0), maximum = Number(last?.seasonal_growth_years || 0);
      const ratio = maximum > 0 ? Math.max(0, Math.min(1, seasonal / maximum)) : 0;
      const current = safePoints(tree.current_xyz), maximumFuture = safePoints(tree.future_xyz);
      const future = current.length === maximumFuture.length ? current.map((point,index)=>point.map((value,axis)=>value+(maximumFuture[index][axis]-value)*ratio)) : current;
      return {...tree,future_xyz:future,target_date:metric?.date||tree.target_date,forecast_horizon_days:targetDay,forecast_horizon_value:targetDay,forecast_horizon_unit:'days',predicted_height_m:metric?.height_m,predicted_crown_major_m:metric?.crown_major_m,predicted_crown_minor_m:metric?.crown_minor_m,predicted_dbh_cm:metric?.dbh_cm,predicted_clearance_envelope_m:metric?.clearance_m,risk_state:metric?.clearance_m==null?'UNKNOWN':Number(metric.clearance_m)<=Number(tree.warning_threshold_m??source.warning_threshold_m??.5)?'STANDARD_EXCEEDED':'ABOVE_STANDARD'};
    });
    return {...source,trees,forecast_horizon_days:targetDay,forecast_horizon_value:targetDay,forecast_horizon_unit:'days',message:`已在本机即时显示第 ${targetDay} 天的季节非刚性孪生（无需重新计算）。`};
  }

  function applyLocalDay(day) {
    const local = localScenarioForDay(day);
    if (!local) return false;
    state.data = local; state.horizonValue = Number(day); state.horizonUnit = 'days'; state.view = 'future';
    setStatus(local.message, ''); renderView('future');
    return true;
  }

  function renderView(view) {
    if (!state.data) return;
    state.view = view;
    const future = view === 'future';
    document.querySelectorAll('[data-value], [data-view="current"]').forEach(button => {
      const active = button.dataset.view === 'current' ? !future : future && Number(button.dataset.value) === state.horizonValue && button.dataset.unit === state.horizonUnit;
      button.classList.toggle('active', active);
      button.setAttribute('aria-pressed', active ? 'true' : 'false');
    });
    const label = future ? horizonText(state.data) : '当前实测几何';
    $('scenarioStatus').textContent = label;
    renderTable(state.data, view);
    renderPlot(state.data, view);
    renderTrend(state.data);
    renderKpis(state.data);
    renderAssumptions(state.data);
    const wireCount = wireSegments(state.data).length;
    const treePoints = (state.data.trees || []).reduce((total, tree) => total + safePoints(tree[future ? 'future_xyz' : 'current_xyz']).length, 0);
    $('geometrySource').textContent = `${treePoints.toLocaleString()} ${future ? '场景' : '当前'}点 · ${wireCount} 条当前 W1`;
    $('geometryNote').textContent = future
      ? '当前树木以浅色叠加，黄色点为季节非刚性预测孪生；树干基部中心与地面高程固定，顶梢、冠层和胸径分部位变化，且每个时段只推进部分方向的枝条。导线保持当前最终 W1。'
      : '当前视图显示当前最终树木几何与当前持久化 W1 导线；没有使用未来预测替代当前结果。';
  }

  async function loadPreview(value, unit) {
    const requested = Number(value), requestedUnit = String(unit);
    if (!Number.isInteger(requested) || !['days','months','years'].includes(requestedUnit)) return;
    const requestId = ++state.requestId;
    state.controller?.abort();
    state.controller = typeof AbortController === 'function' ? new AbortController() : null;
    const controller = state.controller;
    const timeout = setTimeout(() => controller?.abort('timeout'), 180000);
    clearPreview();
    setStatus(`正在计算 +${requested} ${unitText(requestedUnit)}季节非刚性孪生…`, 'loading');
    try {
      const response = await fetch(`/api/future-preview?value=${requested}&unit=${requestedUnit}`, {
        cache: 'no-store', signal: state.controller?.signal,
      });
      const payload = payloadFrom(await response.json());
      if (requestId !== state.requestId) return;
      if (!response.ok) throw new Error(payload?.message || payload?.error?.message || '预测接口请求失败');
      if (payload.status === 'LOCKED' && payload.retryable === true) {
        if (state.busyRetryCount++ < 15) {
          setStatus('后台正在准备当前项目数据，稍后自动打开孪生…', 'loading');
          setTimeout(() => { if (requestId === state.requestId) void loadPreview(requested, requestedUnit); }, 2000);
          return;
        }
        setStatus('后台操作尚未结束，请返回项目检查进度，再重新打开孪生。', 'error');
        return;
      }
      state.busyRetryCount = 0;
      render(payload, requested, requestedUnit);
    } catch (error) {
      if (requestId !== state.requestId) return;
      clearPreview();
      setStatus(controller?.signal.reason === 'timeout' ? '预测读取超时，请返回项目检查后重试。' : `预测读取失败：${error?.message || error}`, 'error');
    } finally {
      clearTimeout(timeout);
      if (requestId === state.requestId) state.controller = null;
    }
  }

  function sameGate(left, right) {
    const fields = ['run_id', 'batch_mtime_ns', 'journal_mtime_ns', 'editor_operation_count', 'published_editor_operation_count'];
    return right?.status === 'PASS' && fields.every(field => left?.[field] === right[field]);
  }

  async function checkGate() {
    if (!state.data || state.gateChecking) return false;
    state.gateChecking = true;
    const previous = state.data;
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 15000);
    try {
      const response = await fetch('/api/future-preview/gate', {cache: 'no-store', signal: controller.signal});
      const payload = await response.json();
      if (state.data !== previous) return false;
      if (!response.ok || !sameGate(previous.gate, payload.gate)) {
        clearPreview();
        $('futureTitle').textContent = '树木数字孪生趋势 🔒';
        setStatus('当前树木点云已变化。请刷新并重新打开本地预测。', 'locked');
        return false;
      }
      return true;
    } catch (_) {
      if (state.data === previous) {
        clearPreview();
        setStatus('无法确认当前树木点云状态，已清除预览。请返回项目检查后重试。', 'error');
      }
      return false;
    } finally {
      clearTimeout(timeout);
      state.gateChecking = false;
    }
  }

  $('currentViewButton').addEventListener('click', async () => { if (await checkGate()) renderView('current'); });
  document.querySelectorAll('[data-value]').forEach(button => button.addEventListener('click', () => {
    const unit=button.dataset.unit,value=Number(button.dataset.value),days=unit==='days'?value:unit==='months'?Math.round(value*30.4375):Math.round(value*365.25);
    if (!applyLocalDay(days)) loadPreview(button.dataset.value, unit);
  }));
  $('timelineSlider').addEventListener('input', event => {
    const days = Number(event.target.value);
    $('timelineValue').textContent = `第 ${days} 天 · 本机即时变化`;
    applyLocalDay(days);
  });
  $('timelineSlider').addEventListener('change', event => {
    applyLocalDay(Number(event.target.value));
  });
  window.addEventListener('resize', () => {
    if (state.data && window.Plotly?.Plots?.resize) window.Plotly.Plots.resize($('futurePlot'));
  });
  setInterval(() => { void checkGate(); }, 3000);
  if (new URLSearchParams(window.location.search).get('embedded') === '1') document.body.classList.add('embedded');
  loadPreview(5, 'years').then(() => applyLocalDay(1));
})();
