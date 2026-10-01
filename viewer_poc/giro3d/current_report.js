(() => {
  const $ = (id) => document.getElementById(id);
  let displayedGate = null;
  let gateChecking = false;
  let reportLoading = false;
  const esc = (value) => String(value ?? '').replace(/[&<>"']/g, (char) => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
  }[char]));
  const value = (item, digits = 2) => {
    if (item === null || item === undefined || item === '') return '—';
    const number = Number(item);
    return Number.isFinite(number) ? number.toFixed(digits) : esc(item);
  };

  function dbhReason(status) {
    const reasons = {
      NO_CROWN_TO_STEM_IDENTITY: '未找到与树冠可靠对应的树干',
      NO_ABSOLUTE_CANDIDATE: '树干证据不足，未通过自动确认',
      TOP_CANDIDATE_CONFLICT: '候选树干归属冲突，需人工复核',
      EDITOR_REMOVED_BH_EVIDENCE: '胸高处点已由人工剔除',
      REFERENCE_REQUIRED: '本树点云证据不足，请在胸径复核页检查',
      STALE_REFERENCE_ESTIMATE: '旧参考估算已失效，请逐树复核',
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
    return reasons[selected] || (selected.startsWith('SEARCH_BUDGET') ? reasons.SEARCH_BUDGET_EXHAUSTED : '证据不足，需人工复核');
  }

  function dbhCell(row) {
    if (row.dbh_method === 'REFERENCE_ESTIMATE') {
      const reason = row.dbh_fit_status || row.dbh_pre_reference_failure_reason;
      const detail = reason ? ` · ${esc(dbhReason(reason))}` : '';
      return `<span>未测得 · 旧参考估算已失效${detail} · <a href="/dbh-review">胸径复核</a></span>`;
    }
    const number = Object.prototype.hasOwnProperty.call(row, 'dbh_cm') ? row.dbh_cm : row.diameter_cm;
    if (number === null || number === undefined || number === '') {
      return row.dbh_fit_status
        ? `<span title="${esc(dbhReason(row.dbh_fit_status))}">未测得 · ${esc(dbhReason(row.dbh_fit_status))} · <a href="/dbh-review">胸径复核</a></span>`
        : '—';
    }
    const labels = { DIRECT_STEM: '点云', RECOVERED_STEM: '恢复', REVIEW_CONFIRMED_STEM: '人工确认点云', ALLOMETRIC_ESTIMATE: '估算' };
    const estimated = row.dbh_quality === 'ESTIMATE' || row.dbh_method === 'ALLOMETRIC_ESTIMATE';
    const model = String(row.dbh_model_version || '');
    const source = model.includes('UPPER_PARTIAL_ARC') ? '本树胸高点云拟合（单侧估算）'
      : model.includes('UPPER_CIRCULAR') ? '本树上部树干外推估算'
      : estimated ? '本树点云估算' : (labels[row.dbh_method] || '');
    const label = [source,
      row.dbh_quality === 'REVIEW' ? '待复核' : ''].filter(Boolean).join(' · ');
    const uncertainty = row.dbh_uncertainty_cm == null ? '' : ` · ±${value(row.dbh_uncertainty_cm, 1)} cm`;
    const detail = [row.dbh_method, row.dbh_quality,
      row.dbh_point_count == null ? '' : `${row.dbh_point_count} 点`,
      row.dbh_fit_rmse == null ? '' : `RMSE ${value(row.dbh_fit_rmse, 4)} m`,
      row.dbh_fit_status ? dbhReason(row.dbh_fit_status) : ''].filter(Boolean).join(' · ');
    return `<span title="${esc(detail)}">${value(number)}${label ? ` · ${label}` : ''}${uncertainty}</span>`;
  }

  function showStatus(message, kind = '') {
    const node = $('reportStatus');
    node.className = `notice ${kind}`;
    node.textContent = message;
  }

  async function browserDownload(url, filename) {
    const response = await fetch(url, {cache: 'no-store'});
    if (!response.ok) {
      const payload = await response.json().catch(() => ({}));
      throw new Error(payload.error?.message || `下载失败 (${response.status})`);
    }
    const objectUrl = URL.createObjectURL(await response.blob());
    const anchor = document.createElement('a');
    anchor.href = objectUrl;
    anchor.download = filename;
    document.body.appendChild(anchor);
    anchor.click();
    anchor.remove();
    setTimeout(() => URL.revokeObjectURL(objectUrl), 1000);
    return {ok: true};
  }

  async function saveReport(kind) {
    try {
      const bridge = window.pywebview?.api;
      const excel = kind === 'excel';
      const result = bridge?.[excel ? 'export_excel' : 'export_html']
        ? await bridge[excel ? 'export_excel' : 'export_html']()
        : await browserDownload(
          excel ? '/download_prediction' : '/download_report_html',
          excel ? '树木风险结果.xlsx' : 'TreeLineWarning_Current_Report.html'
        );
      if (result?.cancelled) { showStatus('已取消导出。', 'locked'); return; }
      showStatus(`${excel ? 'Excel' : 'HTML 报告'}已保存${result?.path ? `：${result.path}` : ''}`);
    } catch (error) {
      showStatus(`导出失败：${error.message || error}`, 'error');
    }
  }

  function renderKpis(kpis = {}) {
    const cards = [
      ['最终全局单木', value(kpis.tree_count, 0)],
      ['全局最小净空 (m)', value(kpis.minimum_clearance_m, 3)],
      ['当前预警阈值 (m)', value(kpis.warning_threshold_m, 2)],
      ['最终成员点数', value(kpis.final_member_point_count, 0)],
      ['风险分类', Object.entries(kpis.risk_counts || {}).map(([key, count]) => `${key} ${count}`).join(' · ') || '—'],
    ];
    $('kpiGrid').innerHTML = cards.map(([label, content]) => `<div class="kpi"><span>${esc(label)}</span><strong>${esc(content)}</strong></div>`).join('');
  }

  function renderRows(rows = []) {
    $('rowCount').textContent = `${rows.length} 行 · FINALIZED`;
    $('treeRows').innerHTML = rows.map((row) => {
      const risk = String(row.current_risk || 'UNKNOWN');
      const displayId = row.display_tree_id || row.persistent_tree_id || '—';
      const persistentId = row.persistent_tree_id || '—';
      return `<tr><td title="持久 ID：${esc(persistentId)}"><b>${esc(displayId)}</b><small> · ${esc(persistentId)}</small></td><td>${value(row.height_m)}</td><td>${value(row.crown_width_m)}</td><td>${value(row.crown_area_m2)}</td><td>${dbhCell(row)}</td><td>${value(row.minimum_clearance_m, 3)}</td><td>${value(row.horizontal_clearance_m, 3)}</td><td>${value(row.vertical_clearance_m, 3)}</td><td>${value(row.danger_point_count, 0)}</td><td data-risk="${esc(risk)}">${esc(risk)}</td><td>${esc(row.quality_state || '—')}</td><td>${row.human_edited ? '是' : '否'}</td></tr>`;
    }).join('') || '<tr><td colspan="12">没有当前最终单木记录</td></tr>';
  }

  function renderPlot(display = {}, wires = {}) {
    const tree = display.trees3d || {};
    const xyz = tree.xyz || [];
    const traces = [];
    if (xyz.length) {
      traces.push({
        type: 'scatter3d', mode: 'markers', name: '当前树木',
        x: xyz.map((point) => point[0]), y: xyz.map((point) => point[1]), z: xyz.map((point) => point[2]),
        text: tree.tree_id_by_index || [],
        marker: {size: 2.4, color: tree.color?.length ? tree.color : '#55c7d6', opacity: .72},
        hovertemplate: 'tree %{text}<extra>当前最终点</extra>'
      });
    }
    (wires.segments || []).forEach((segment, index) => {
      const points = segment.xyz || [];
      if (points.length < 2) return;
      traces.push({
        type: 'scatter3d', mode: 'lines', name: `W1 ${segment.span_id} · wire ${segment.wire_id} · seg ${segment.segment_index}${segment.distance_ready ? '' : ' · 非净空'}`,
        x: points.map((point) => point[0]), y: points.map((point) => point[1]), z: points.map((point) => point[2]),
        line: {width: 5, dash: segment.distance_ready ? 'solid' : 'dot', color: `hsl(${(index * 83) % 360} 78% 62%)`},
        hovertemplate: 'W1 persisted curve<extra></extra>'
      });
    });
    $('geometrySource').textContent = `${tree.point_count || 0} tree points · ${wires.segment_count || 0} W1 segments`;
    const unavailableReady = (wires.segments || []).filter((segment) => !segment.distance_ready).length;
    $('geometryNote').textContent = wires.status === 'AVAILABLE'
      ? `树木点为最终编辑器显示采样；导线为持久化 SpanWireResult W1 的当前导线拟合。${unavailableReady ? ` ${unavailableReady} 条曲线不具备净空距离资格，以点线显示。` : ''}${wires.truncated ? ' 导线显示已限额采样。' : ''}`
      : wires.status === 'PARTIAL'
      ? `部分当前导线拟合可用；其余档距未提供有效 W1 曲线。${(wires.diagnostics || []).join('；')}`
      : '当前最终结果没有可用的持久化 W1 导线拟合，页面不会用杆塔示意线代替。';
    if (!window.Plotly?.newPlot) {
      $('currentPlot').textContent = traces.length ? '本地 Plotly 资源不可用，点数据仍已加载。' : '没有可显示的当前最终点数据。';
      return;
    }
    window.Plotly.newPlot('currentPlot', traces, {
      paper_bgcolor: '#0b121b', plot_bgcolor: '#0b121b', margin: {l: 0, r: 0, t: 20, b: 0},
      scene: {xaxis: {title: 'X'}, yaxis: {title: 'Y'}, zaxis: {title: 'Z'}, aspectmode: 'data', bgcolor: '#0b121b'},
      legend: {font: {color: '#edf7ff'}}, font: {color: '#edf7ff'}
    }, {responsive: true, displaylogo: false});
  }

  function clearDisplayedReport(message, kind) {
    if (displayedGate && window.Plotly?.purge) window.Plotly.purge('currentPlot');
    displayedGate = null;
    $('reportBody').classList.add('hidden');
    showStatus(message, kind);
  }

  async function load() {
    if (reportLoading) return;
    reportLoading = true;
    try {
      const response = await fetch('/api/current-report', {cache: 'no-store'});
      const data = await response.json();
      if (!response.ok || !data.ok) throw new Error(data.error?.message || '当前报告接口不可用');
      if (data.status !== 'FINALIZED') {
        clearDisplayedReport(`${data.message || '当前结果尚未定稿，报告已锁定。'}${(data.gate?.reasons || []).length ? ` 原因：${data.gate.reasons.join('；')}` : ''}`, 'locked');
        return;
      }
      displayedGate = data.gate;
      showStatus(data.dbh_delivery_message || data.message || '当前最终结果（只读）');
      const towerWarnings = Array.isArray(data.tower_exclusion_warnings) ? data.tower_exclusion_warnings : [];
      $('towerWarning').classList.toggle('hidden', towerWarnings.length === 0);
      $('towerWarning').textContent = towerWarnings.map(item => String(item.message || '')).join('；');
      $('reportBody').classList.remove('hidden');
      renderKpis(data.kpis);
      renderRows(data.trees);
      renderPlot(data.display, data.wires);
    } catch (error) {
      clearDisplayedReport(`当前报告读取失败：${error.message || error}`, 'error');
    } finally {
      reportLoading = false;
    }
  }

  function sameGate(left, right) {
    const fields = ['run_id', 'batch_mtime_ns', 'journal_mtime_ns', 'prediction_mtime_ns',
      'editor_operation_count', 'published_editor_operation_count', 'final_generation_id',
      'result_generation_id', 'risk_generation_id'];
    return right?.status === 'PASS' && fields.every(field => left?.[field] === right[field]);
  }

  async function checkGate() {
    if (gateChecking || reportLoading) return;
    gateChecking = true;
    try {
      const response = await fetch('/api/future-preview/gate', {cache: 'no-store'});
      const payload = await response.json();
      if (!response.ok || payload.gate?.status !== 'PASS') {
        const reasons = (payload.gate?.reasons || []).join('；');
        clearDisplayedReport(`当前结果尚未定稿，报告暂时隐藏。${reasons ? ` 原因：${reasons}` : ''}`, 'locked');
      } else if (!displayedGate || !sameGate(displayedGate, payload.gate)) {
        clearDisplayedReport('检测到新的当前定稿，正在刷新报告…', '');
        await load();
      }
    } catch (_) {
      clearDisplayedReport('无法确认当前定稿状态，已清除报告图；连接恢复后自动刷新。', 'error');
    } finally {
      gateChecking = false;
    }
  }

  $('printReport').addEventListener('click', () => window.print());
  $('downloadExcel').addEventListener('click', () => saveReport('excel'));
  $('downloadHtml').addEventListener('click', () => saveReport('html'));
  setInterval(() => { void checkGate(); }, 3000);
  document.addEventListener('visibilitychange', () => { if (!document.hidden) void checkGate(); });
  load();
})();
