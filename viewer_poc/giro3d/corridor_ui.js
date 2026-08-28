(() => {
  'use strict';

  const $ = id => document.getElementById(id);
  const esc = value => String(value ?? '—').replace(/[&<>"']/g, ch => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
  }[ch]));
  const api = (path, options = {}) => fetch(path, Object.assign({cache: 'no-store'}, options))
    .then(async response => {
      const payload = await response.json().catch(() => ({ok: false, error: {message: '本地服务返回了无法读取的数据'}}));
      if (!response.ok || payload.ok === false) throw new Error(payload.error?.message || '本地请求失败');
      return payload;
    });
  const post = (path, body = {}) => api(path, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(body),
  });

  const state = {
    project: null,
    activeTowerId: null,
    anchorViewer: null,
    anchorOverview: null,
    candidateViewer: null,
    candidatePayload: null,
    anchorBusy: false,
  };

  const style = document.createElement('style');
  style.textContent = `
    #corridorSection .corridor-actions{display:flex;gap:6px;flex-wrap:wrap;margin:8px 0}
    #corridorTowerList{display:flex;flex-direction:column;gap:6px;max-height:260px;overflow:auto}
    .corridor-tower{border:1px solid var(--line,#d9dee7);border-radius:8px;padding:8px;background:rgba(255,255,255,.55)}
    .corridor-tower .head{display:flex;justify-content:space-between;gap:8px;align-items:center}
    .corridor-tower .meta{font-size:12px;opacity:.72;margin:4px 0}
    .corridor-tower .actions{display:flex;gap:5px;flex-wrap:wrap}
    .corridor-pill{font-size:11px;border-radius:999px;padding:2px 7px;background:#eef2f7}
    .corridor-pill.good{background:#e7f6ec;color:#176b34}.corridor-pill.warn{background:#fff3d8;color:#8a5b00}
    #corridorAnchorWrap,#corridorCandidateWrap{margin-top:10px;border-top:1px solid var(--line,#d9dee7);padding-top:9px}
    #corridorAnchorCanvas,#corridorCandidateCanvas{width:100%;height:250px;display:block;border-radius:8px;background:#111827}
    #corridorCandidateCards{display:flex;flex-direction:column;gap:6px;margin-top:7px}
    .corridor-candidate{display:grid;grid-template-columns:1fr auto;gap:6px;border:1px solid var(--line,#d9dee7);border-radius:7px;padding:7px}
    .corridor-candidate small{display:block;opacity:.72}
  `;
  document.head.appendChild(style);

  function installSection() {
    if ($('corridorSection')) return;
    const towerSection = $('towerSection');
    if (!towerSection?.parentElement) return;
    const section = document.createElement('section');
    section.className = 'section contextual hidden';
    section.id = 'corridorSection';
    section.innerHTML = `
      <h3>GIS 线路注册与杆塔定位</h3>
      <div id="corridorStatus" class="notice warn">等待 GIS 数据</div>
      <div class="corridor-actions">
        <button id="corridorPrepare" class="small primary">自动注册坐标</button>
        <button id="corridorRebuild" class="small">生成 / 刷新档距</button>
        <button id="corridorRefresh" class="small ghost">刷新</button>
      </div>
      <div id="corridorTowerList"></div>
      <div id="corridorAnchorWrap" class="hidden">
        <div id="corridorAnchorMessage" class="notice">请选择一根 GIS 杆塔，再在全局点云中点击它的真实位置。</div>
        <canvas id="corridorAnchorCanvas" aria-label="GIS 锚点真实三维点云"></canvas>
      </div>
      <div id="corridorCandidateWrap" class="hidden">
        <div id="corridorCandidateMessage" class="notice">局部候选仅用于辅助定位，必须人工确认。</div>
        <canvas id="corridorCandidateCanvas" aria-label="GIS 杆塔局部候选三维点云"></canvas>
        <div id="corridorCandidateCards"></div>
      </div>`;
    towerSection.parentElement.insertBefore(section, towerSection);
    $('corridorPrepare').addEventListener('click', prepareRegistration);
    $('corridorRebuild').addEventListener('click', rebuildTopology);
    $('corridorRefresh').addEventListener('click', refresh);
    $('corridorTowerList').addEventListener('click', onTowerAction);
    $('corridorCandidateCards').addEventListener('click', onCandidateConfirm);
  }

  function registrationLabel(value) {
    return ({
      UNREGISTERED: '未注册',
      GEOREFERENCED: '已按 CRS 注册',
      ASSISTED_ONE_ANCHOR: '已有 1 个锚点',
      REGISTERED_TWO_ANCHOR: '已有 2 个锚点，待检查点验证',
      VERIFIED: '锚点注册已验证',
      REQUIRES_CONFIRMATION: '需要人工确认',
      INVALID: '注册无效',
    })[value] || value || '未注册';
  }

  function render() {
    installSection();
    const project = state.project;
    if (!project || project.is_desktop_home) {
      $('corridorSection')?.classList.add('hidden');
      return;
    }
    const gisTowers = (project.tower_records || []).filter(tower => tower.attributes?.gis_source);
    const registration = project.corridor_registration || {};
    const hasGIS = registration.available || gisTowers.length > 0;
    $('corridorSection').classList.toggle('hidden', !hasGIS);
    if (!hasGIS) return;

    const summary = project.span_readiness_summary || {ready: 0, blocked: 0, total: project.spans || 0};
    const topology = project.corridor_topology;
    const statusKind = registration.ready_for_science ? '' : 'warn';
    $('corridorStatus').className = `notice ${statusKind}`;
    $('corridorStatus').innerHTML = [
      `注册：<b>${esc(registrationLabel(registration.state))}</b>`,
      `点云 CRS：${esc(registration.point_cloud_crs || '本地 / 未声明')}`,
      `GIS CRS：${esc(registration.gis_crs || '待确认')}`,
      `档距：${summary.ready || 0} READY / ${summary.blocked || 0} BLOCKED`,
      topology?.status ? `拓扑：${esc(topology.status)}` : '拓扑：待生成',
    ].join('<br>');

    $('corridorTowerList').innerHTML = gisTowers.length ? gisTowers.map(tower => {
      const confirmed = tower.confirmation_state === 'CONFIRMED';
      const canCandidate = Boolean(registration.transform_available);
      return `<div class="corridor-tower" data-tower-id="${esc(tower.tower_id)}">
        <div class="head"><b>${esc(tower.display_name)}</b><span class="corridor-pill ${confirmed ? 'good' : 'warn'}">${confirmed ? '已确认' : 'GIS seed'}</span></div>
        <div class="meta">${esc(tower.external_asset_id || tower.tower_id)} · ${esc(tower.position_source)}</div>
        <div class="actions">
          <button class="small" data-corridor-action="anchor" data-tower="${esc(tower.tower_id)}">设为锚点</button>
          <button class="small primary" data-corridor-action="candidate" data-tower="${esc(tower.tower_id)}" ${canCandidate ? '' : 'disabled'}>${confirmed ? '重新定位候选' : '查看候选 A/B/C'}</button>
        </div>
      </div>`;
    }).join('') : '<div class="muted padded">GIS 中没有杆塔点。</div>';
  }

  async function refresh() {
    try {
      const payload = await api('/api/desktop/workspace');
      state.project = payload.project;
      render();
    } catch (error) {
      console.warn('Corridor UI refresh failed', error);
    }
  }

  async function prepareRegistration() {
    try {
      $('corridorStatus').textContent = '正在检查 CRS / 注册关系…';
      await post('/api/desktop/corridor/prepare');
      await refresh();
      const registration = state.project?.corridor_registration;
      if (registration?.ready_for_science) {
        $('corridorStatus').className = 'notice';
      }
    } catch (error) {
      $('corridorStatus').className = 'notice bad';
      $('corridorStatus').textContent = error.message || '坐标注册失败';
    }
  }

  async function rebuildTopology() {
    try {
      const result = await post('/api/desktop/corridor/topology/rebuild');
      const topology = result.topology || {};
      if (topology.status === 'REQUIRES_CONFIRMATION') {
        $('corridorStatus').className = 'notice warn';
        $('corridorStatus').textContent = `路线归属存在歧义：${(topology.ambiguous_tower_ids || []).length} 基杆塔需要确认。`;
        await refresh();
        return;
      }
      window.setTimeout(() => window.location.reload(), 120);
    } catch (error) {
      $('corridorStatus').className = 'notice bad';
      $('corridorStatus').textContent = error.message || '档距生成失败';
    }
  }

  async function ensureAnchorViewer() {
    $('corridorAnchorWrap').classList.remove('hidden');
    if (state.anchorViewer) return;
    $('corridorAnchorMessage').textContent = '正在加载全局真实点云…';
    const overview = await api('/api/desktop/tower-viewer/overview');
    state.anchorOverview = overview;
    state.anchorViewer = new window.TreeLineTowerViewer3D(
      $('corridorAnchorCanvas'),
      {mode: 'height', pointSize: 2, onPick: saveAnchorPick},
    );
    state.anchorViewer.setData(overview);
    await state.anchorViewer.firstVisible;
    $('corridorAnchorMessage').textContent = '点云已就绪：请选择 GIS 杆塔后，在这里点击它的真实位置。';
  }

  async function beginAnchor(towerId) {
    state.activeTowerId = towerId;
    const tower = (state.project?.tower_records || []).find(item => item.tower_id === towerId);
    try {
      await ensureAnchorViewer();
      $('corridorAnchorMessage').innerHTML = `当前锚点：<b>${esc(tower?.display_name || towerId)}</b>。请在真实点云中点击该杆塔。`;
      $('corridorAnchorCanvas').scrollIntoView({block: 'nearest'});
    } catch (error) {
      $('corridorAnchorMessage').className = 'notice bad';
      $('corridorAnchorMessage').textContent = error.message || '锚点点云加载失败';
    }
  }

  async function saveAnchorPick(world) {
    if (!state.activeTowerId || state.anchorBusy) return;
    state.anchorBusy = true;
    try {
      $('corridorAnchorMessage').textContent = '正在保存锚点并验证注册…';
      await post('/api/desktop/corridor/anchor', {
        tower_id: state.activeTowerId,
        point_cloud_xyz: [world.x, world.y, world.z],
      });
      state.activeTowerId = null;
      await refresh();
      const registration = state.project?.corridor_registration;
      $('corridorAnchorMessage').textContent = registration?.ready_for_science
        ? '注册已验证，正在刷新杆塔/档距状态…'
        : `锚点已保存：${registrationLabel(registration?.state)}。可继续选择下一根 GIS 杆塔。`;
      if (registration?.ready_for_science) {
        window.setTimeout(() => window.location.reload(), 180);
      }
    } catch (error) {
      $('corridorAnchorMessage').className = 'notice bad';
      $('corridorAnchorMessage').textContent = error.message || '锚点保存失败';
    } finally {
      state.anchorBusy = false;
    }
  }

  async function showCandidates(towerId) {
    state.activeTowerId = towerId;
    $('corridorCandidateWrap').classList.remove('hidden');
    $('corridorCandidateMessage').className = 'notice';
    $('corridorCandidateMessage').textContent = '正在读取 GIS seed 附近真实局部点云并生成候选…';
    $('corridorCandidateCards').innerHTML = '';
    try {
      const result = await api(`/api/desktop/corridor/tower-candidates?tower_id=${encodeURIComponent(towerId)}`);
      state.candidatePayload = result;
      if (!state.candidateViewer) {
        state.candidateViewer = new window.TreeLineTowerViewer3D(
          $('corridorCandidateCanvas'),
          {mode: 'height', pointSize: 2},
        );
      }
      state.candidateViewer.setData(result.local || {xyz: [], rgb: [], intensity: []});
      const markers = (result.candidates || []).map(item => ({
        display_name: item.candidate_id.replace('candidate-', '候选 '),
        x: item.xyz[0], y: item.xyz[1], z: item.xyz[2], height_m: item.height_m,
      }));
      state.candidateViewer.setMarkers(markers);
      await state.candidateViewer.firstVisible;
      $('corridorCandidateMessage').textContent = result.candidates?.length
        ? '候选是辅助提示；点击“确认”后软件还会再次读取 RAW 局部点云精修。'
        : '未找到可靠候选，请改用锚点/人工点选。';
      $('corridorCandidateCards').innerHTML = (result.candidates || []).map(item => `
        <div class="corridor-candidate">
          <div><b>${esc(item.candidate_id.replace('candidate-', '候选 '))}</b>
          <small>高度 ${Number(item.height_m).toFixed(1)} m · 距 seed ${Number(item.distance_to_seed_m).toFixed(1)} m · 得分 ${Number(item.score).toFixed(2)}</small></div>
          <button class="small primary" data-confirm-candidate="${esc(item.candidate_id)}">人工确认</button>
        </div>`).join('');
      $('corridorCandidateWrap').scrollIntoView({block: 'nearest'});
    } catch (error) {
      $('corridorCandidateMessage').className = 'notice bad';
      $('corridorCandidateMessage').textContent = error.message || '杆塔候选生成失败';
    }
  }

  async function onCandidateConfirm(event) {
    const button = event.target.closest('[data-confirm-candidate]');
    if (!button || !state.activeTowerId || !state.candidatePayload) return;
    const candidate = (state.candidatePayload.candidates || []).find(
      item => item.candidate_id === button.dataset.confirmCandidate,
    );
    if (!candidate) return;
    button.disabled = true;
    try {
      $('corridorCandidateMessage').textContent = '正在用真实 RAW 局部点云再次精修并保存…';
      await post('/api/desktop/corridor/tower-confirm', {
        tower_id: state.activeTowerId,
        candidate_id: candidate.candidate_id,
        candidate_xyz: candidate.xyz,
      });
      $('corridorCandidateMessage').textContent = '杆塔已人工确认，正在刷新档距可分析状态…';
      window.setTimeout(() => window.location.reload(), 160);
    } catch (error) {
      button.disabled = false;
      $('corridorCandidateMessage').className = 'notice bad';
      $('corridorCandidateMessage').textContent = error.message || '杆塔确认失败';
    }
  }

  function onTowerAction(event) {
    const button = event.target.closest('[data-corridor-action]');
    if (!button) return;
    const towerId = button.dataset.tower;
    if (button.dataset.corridorAction === 'anchor') beginAnchor(towerId);
    if (button.dataset.corridorAction === 'candidate') showCandidates(towerId);
  }

  function interceptAnalyzeAll() {
    const button = $('startAll');
    if (!button || button.dataset.corridorBound) return;
    button.dataset.corridorBound = 'yes';
    button.addEventListener('click', async event => {
      const project = state.project;
      if (!project?.corridor_registration?.available) return;
      const summary = project.span_readiness_summary || {};
      if (!(Number(summary.blocked || 0) > 0)) return;
      event.preventDefault();
      event.stopImmediatePropagation();
      try {
        const result = await post('/api/desktop/corridor/batch-ready', {mode: 'REAL'});
        const skipped = result.blocked_span_ids || [];
        const message = skipped.length
          ? `已启动 ${result.selected_ready_span_ids.length} 个 READY 档距；${skipped.length} 个未确认档距未运行。`
          : `已启动 ${result.selected_ready_span_ids.length} 个 READY 档距。`;
        const footer = $('footerState');
        if (footer) footer.textContent = `应用状态：${message}`;
        window.setTimeout(() => window.location.reload(), 500);
      } catch (error) {
        const footer = $('footerState');
        if (footer) footer.textContent = `应用状态：${error.message || '没有可运行的 READY 档距'}`;
      }
    }, true);
  }

  installSection();
  interceptAnalyzeAll();
  refresh();
  window.addEventListener('focus', () => window.setTimeout(refresh, 250));
})();
