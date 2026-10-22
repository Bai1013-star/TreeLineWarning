(() => {
  const $ = (id) => document.getElementById(id);
  const esc = (value) => String(value ?? '').replace(/[&<>"']/g, (char) => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
  }[char]));
  const value = (item, digits = 2) => {
    if (item === null || item === undefined || item === '') return '—';
    const number = Number(item);
    return Number.isFinite(number) ? number.toFixed(digits) : esc(item);
  };

  function dbhCell(row) {
    const number = row.dbh_cm ?? row.diameter_cm;
    if (number === null || number === undefined || number === '') {
      const reasons = {NO_CROWN_TO_STEM_IDENTITY:'树冠与原始树干无法可靠对应，请人工点选树干',NO_ABSOLUTE_CANDIDATE:'原始点云中没有足够的绝对坐标候选，请人工复核',NO_SUPPORTED_STEM:'缺少可信树干证据'};
      return row.dbh_fit_status
        ? `<span title="${esc(row.dbh_fit_status)}">No · ${esc(reasons[row.dbh_fit_status] || row.dbh_fit_status)}（可在预测信息中补填现场实测值）</span>`
        : '—';
    }
    const labels = { DIRECT_STEM: '点云', RECOVERED_STEM: '恢复', REVIEW_CONFIRMED_STEM: '人工确认点云', RAW_GROUND_BH_SECTION: '原始树木最低点以上1.30米断面', ALLOMETRIC_ESTIMATE: '估算' };
    const label = [labels[row.dbh_method] || '',
      row.dbh_quality === 'ESTIMATED' ? '估计' : '',
      row.dbh_quality === 'REVIEW' ? '待复核' : ''].filter(Boolean).join(' · ');
    const heightReference = row.dbh_measurement_z_absolute_m == null ? ''
      : `原始树最低点 ${value(row.dbh_ground_z_absolute_m, 3)} m + 1.30 m = ${value(row.dbh_measurement_z_absolute_m, 3)} m`;
    const detail = [row.dbh_method, row.dbh_quality, heightReference,
      row.dbh_point_count == null ? '' : `${row.dbh_point_count} 点`,
      row.dbh_fit_rmse == null ? '' : `RMSE ${value(row.dbh_fit_rmse, 4)} m`].filter(Boolean).join(' · ');
    return `<span title="${esc(detail)}">${value(number)}${label ? ` · ${label}` : ''}</span>`;
  }

  function showStatus(message, kind = '') {
    const node = $('reportStatus');
    node.className = `notice ${kind}`;
    node.textContent = message;
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
        line: {width: 6, dash: segment.distance_ready ? 'solid' : 'dot', color: '#ff4d5a'},
        hovertemplate: 'W1 persisted curve<extra></extra>'
      });
    });
    $('geometrySource').textContent = `${tree.point_count || 0} 个处理后单木点 · ${wires.segment_count || 0} 条拟合导线`;
    const unavailableReady = (wires.segments || []).filter((segment) => !segment.distance_ready).length;
    $('geometryNote').textContent = wires.status === 'AVAILABLE'
      ? `本图只显示处理后的最终单木点云和红色拟合导线，不混入原始地面、杆塔示意或未分类背景点。${unavailableReady ? ` ${unavailableReady} 条曲线不具备净空距离资格，以点线显示。` : ''}${wires.truncated ? ' 导线显示已限额采样。' : ''}`
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

  async function load() {
    try {
      const response = await fetch('/api/current-report', {cache: 'no-store'});
      const data = await response.json();
      if (!response.ok || !data.ok) throw new Error(data.error?.message || '当前报告接口不可用');
      if (data.status !== 'FINALIZED') {
        showStatus(`${data.message || '当前结果尚未定稿，报告已锁定。'}${(data.gate?.reasons || []).length ? ` 原因：${data.gate.reasons.join('；')}` : ''}`, 'locked');
        return;
      }
      showStatus(data.message || '当前最终结果（只读）', '');
      $('reportBody').classList.remove('hidden');
      renderKpis(data.kpis);
      renderRows(data.trees);
      renderPlot(data.display, data.wires);
    } catch (error) {
      showStatus(`当前报告读取失败：${error.message || error}`, 'error');
    }
  }

  $('printReport').addEventListener('click', () => window.print());
  load();
})();
