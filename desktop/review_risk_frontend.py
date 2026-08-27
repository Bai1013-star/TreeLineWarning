"""Inline offline Giro3D clearance-review extension for the desktop Step-2 page."""

TREE_REVIEW_RISK_EXTENSION_JS = r'''
(function () {
  'use strict';

  const Viewer = window.TreeLineTreeReview3D;
  const App = window.TreeLineTreeReviewApp;
  if (!Viewer || !App) return;

  const FILTERS = new Set(['ALL', 'WARNING', 'DANGER', 'REVIEW_REQUIRED']);
  const prototype = Viewer.prototype;
  const originalSetData = prototype.setData;
  const originalRenderLabels = prototype.renderLabels;
  const originalHitAt = prototype.hitAt;

  function classColor(viewer, displayClass) {
    const values = {
      DANGER: 0xff1744,
      WARNING: 0xffa000,
      REVIEW_REQUIRED: 0xab47bc,
      SAFE: 0x90a4ae,
    };
    return new viewer.THREE.Color(values[displayClass] || 0x90a4ae);
  }

  function validXYZ(value) {
    return Array.isArray(value) && value.length >= 3 && value.slice(0, 3).every(item => Number.isFinite(Number(item)));
  }

  function matchesFilter(record, filter) {
    return filter === 'ALL' || String(record.display_class || '') === filter;
  }

  function evidenceRecords(viewer) {
    const payload = viewer.source && viewer.source.clearance_evidence || {};
    return Array.isArray(payload.records) ? payload.records : [];
  }

  function addClearanceEvidence(viewer) {
    viewer._clearanceEvidenceGroups = [];
    evidenceRecords(viewer).forEach(record => {
      if (!validXYZ(record.tree_xyz) || !validXYZ(record.wire_xyz)) return;
      const points = [record.tree_xyz, record.wire_xyz];
      const color = classColor(viewer, String(record.display_class || ''));
      const group = new viewer.THREE.Group();
      group.userData.role = 'CLEARANCE_EVIDENCE';
      group.userData.record = record;

      const lineGeometry = new viewer.THREE.BufferGeometry();
      lineGeometry.setAttribute('position', new viewer.THREE.BufferAttribute(viewer.positions(points), 3));
      const line = new viewer.THREE.Line(
        lineGeometry,
        new viewer.THREE.LineBasicMaterial({color: color.clone(), transparent: true, opacity: 0.92, depthTest: true})
      );
      line.userData.role = 'CLEARANCE_SEGMENT';
      line.userData.record = record;
      group.add(line);

      const endpointGeometry = new viewer.THREE.BufferGeometry();
      endpointGeometry.setAttribute('position', new viewer.THREE.BufferAttribute(viewer.positions(points), 3));
      const endpoints = new viewer.THREE.Points(
        endpointGeometry,
        new viewer.THREE.PointsMaterial({color: color.clone(), size: 5.8, sizeAttenuation: false, depthTest: true, depthWrite: true})
      );
      endpoints.userData.role = 'CLEARANCE_ENDPOINT';
      endpoints.userData.record = record;
      group.add(endpoints);

      viewer.content.add(group);
      viewer._clearanceEvidenceGroups.push(group);
    });
  }

  function findDangerObject(viewer) {
    let found = null;
    viewer.content.traverse(object => {
      if (!found && object.userData && object.userData.role === 'DANGER_POINT') found = object;
    });
    return found;
  }

  function filterDangerGeometry(viewer, filter) {
    const object = viewer._dangerPointsObject;
    if (!object || !object.geometry) {
      viewer._dangerFilteredCount = 0;
      return;
    }
    const danger = viewer.source && viewer.source.danger_points || {};
    const xyz = Array.isArray(danger.xyz) ? danger.xyz : [];
    const treeIds = Array.isArray(danger.tree_id_by_index) ? danger.tree_id_by_index : [];
    const classByTree = new Map(evidenceRecords(viewer).map(record => [Number(record.global_tree_id), String(record.display_class || '')]));
    const count = Math.min(xyz.length, treeIds.length);
    const rows = [];
    for (let index = 0; index < count; index += 1) {
      const treeId = Number(treeIds[index]);
      if (filter === 'ALL' || classByTree.get(treeId) === filter) rows.push(index);
    }
    object.geometry.setAttribute('position', new viewer.THREE.BufferAttribute(viewer.positions(xyz, rows), 3));
    object.geometry.setAttribute('sourceIndex', new viewer.THREE.Uint32BufferAttribute(rows, 1));
    if (rows.length) object.geometry.computeBoundingSphere();
    object.geometry.attributes.position.needsUpdate = true;
    object.geometry.attributes.sourceIndex.needsUpdate = true;
    viewer._dangerFilteredCount = rows.length;
  }

  function clearancePickAt(viewer, clientX, clientY) {
    if (!viewer._dangerVisible) return null;
    const rect = viewer.canvasRect();
    if (!rect.width || !rect.height) return null;
    const ndc = new viewer.THREE.Vector2(
      ((clientX - rect.left) / rect.width) * 2 - 1,
      -((clientY - rect.top) / rect.height) * 2 + 1
    );
    viewer.raycaster.setFromCamera(ndc, viewer.camera);
    const distance = viewer.camera.position.distanceTo(viewer.controls.target);
    const worldPixel = distance * Math.tan(viewer.camera.fov * Math.PI / 360) / Math.max(1, rect.height);
    viewer.raycaster.params.Points.threshold = Math.max(viewer.maxDim * 0.0008, worldPixel * 9);
    viewer.raycaster.params.Line.threshold = Math.max(viewer.maxDim * 0.0005, worldPixel * 6);

    const pickables = [];
    if (viewer._dangerPointsObject && viewer._dangerPointsObject.visible) pickables.push(viewer._dangerPointsObject);
    (viewer._clearanceEvidenceGroups || []).forEach(group => {
      if (!group.visible) return;
      group.children.forEach(child => {
        if (child.visible !== false) pickables.push(child);
      });
    });
    if (!pickables.length) return null;
    const hits = viewer.raycaster.intersectObjects(pickables, false);
    if (!hits.length) return null;
    const hit = hits[0];
    let record = hit.object && hit.object.userData ? hit.object.userData.record : null;
    let treeId = record ? Number(record.global_tree_id) : NaN;
    let sourceIndex = null;
    if (!record && hit.object === viewer._dangerPointsObject) {
      const attr = hit.object.geometry.getAttribute('sourceIndex');
      if (attr && Number.isInteger(hit.index) && hit.index >= 0 && hit.index < attr.count) {
        sourceIndex = Number(attr.getX(hit.index));
        const danger = viewer.source && viewer.source.danger_points || {};
        const ids = Array.isArray(danger.tree_id_by_index) ? danger.tree_id_by_index : [];
        treeId = Number(ids[sourceIndex]);
        record = evidenceRecords(viewer).find(item => Number(item.global_tree_id) === treeId) || null;
      }
    }
    if (!record || !Number.isFinite(treeId)) return null;
    const xyz = sourceIndex != null
      ? ((viewer.source.danger_points && viewer.source.danger_points.xyz || [])[sourceIndex] || record.tree_xyz)
      : record.tree_xyz;
    return {
      treeId,
      xyz: validXYZ(xyz) ? xyz : record.tree_xyz,
      renderedPointIndex: Number.isInteger(hit.index) ? hit.index : null,
      clearanceRecord: record,
      riskRole: String(hit.object && hit.object.userData && hit.object.userData.role || 'DANGER_POINT'),
    };
  }

  function applyVisibility(viewer) {
    const visible = Boolean(viewer._dangerVisible);
    const filter = FILTERS.has(viewer._clearanceFilter) ? viewer._clearanceFilter : 'ALL';
    filterDangerGeometry(viewer, filter);
    if (viewer._dangerPointsObject) {
      viewer._dangerPointsObject.visible = visible && Number(viewer._dangerFilteredCount || 0) > 0;
    }
    (viewer._clearanceEvidenceGroups || []).forEach(group => {
      group.visible = visible && matchesFilter(group.userData.record || {}, filter);
    });
    viewer.requestRender();
    viewer.renderLabels();
  }

  prototype.setData = function (data) {
    if (typeof this._dangerVisible !== 'boolean') this._dangerVisible = false;
    if (!FILTERS.has(this._clearanceFilter)) this._clearanceFilter = 'ALL';
    const result = originalSetData.call(this, data);
    this._dangerPointsObject = findDangerObject(this);
    addClearanceEvidence(this);
    applyVisibility(this);
    const danger = this.source && this.source.danger_points || {};
    const evidence = this.source && this.source.clearance_evidence || {};
    window.dispatchEvent(new CustomEvent('treeline-clearance-data', {detail: {
      dangerAuthority: String(danger.authority || 'UNAVAILABLE'),
      dangerPointCount: Number(danger.full_point_count || 0),
      evidenceAuthority: String(evidence.authority || 'UNAVAILABLE'),
      evidenceCount: Number(evidence.record_count || 0),
    }}));
    return result;
  };

  prototype.hitAt = function (clientX, clientY) {
    return clearancePickAt(this, clientX, clientY) || originalHitAt.call(this, clientX, clientY);
  };

  prototype.setDangerVisible = function (visible) {
    this._dangerVisible = Boolean(visible);
    applyVisibility(this);
  };

  prototype.setClearanceFilter = function (filter) {
    const normalized = String(filter || 'ALL').toUpperCase();
    this._clearanceFilter = FILTERS.has(normalized) ? normalized : 'ALL';
    applyVisibility(this);
  };

  prototype.dangerLayerInfo = function () {
    const danger = this.source && this.source.danger_points || {};
    const evidence = this.source && this.source.clearance_evidence || {};
    return {
      visible: Boolean(this._dangerVisible),
      filter: this._clearanceFilter || 'ALL',
      danger_authority: String(danger.authority || 'UNAVAILABLE'),
      danger_point_count: Number(danger.full_point_count || 0),
      danger_filtered_display_count: Number(this._dangerFilteredCount || 0),
      evidence_authority: String(evidence.authority || 'UNAVAILABLE'),
      evidence_count: Number(evidence.record_count || 0),
    };
  };

  prototype.renderLabels = function () {
    originalRenderLabels.call(this);
    if (!this._dangerVisible || !this.labelOverlay || !this.source) return;
    const records = evidenceRecords(this);
    const filter = FILTERS.has(this._clearanceFilter) ? this._clearanceFilter : 'ALL';
    records.filter(record => matchesFilter(record, filter)).forEach(record => {
      const midpoint = record.midpoint_xyz;
      if (!validXYZ(midpoint)) return;
      const projected = new this.THREE.Vector3(
        Number(midpoint[0]) - this.origin[0],
        Number(midpoint[1]) - this.origin[1],
        Number(midpoint[2]) - this.origin[2]
      ).project(this.camera);
      if (![projected.x, projected.y, projected.z].every(Number.isFinite) || projected.z < -1 || projected.z > 1 || Math.abs(projected.x) > 1.2 || Math.abs(projected.y) > 1.2) return;
      const button = document.createElement('button');
      button.type = 'button';
      button.className = 'tree-label clearance-label';
      button.textContent = `${Number(record.minimum_clearance_m).toFixed(2)} m`;
      button.title = `树 ${record.global_tree_id} · ${record.display_class} · 档距 ${record.span_name || '—'} · 导线 ${record.nearest_wire_id == null ? '—' : record.nearest_wire_id}`;
      button.style.left = `${(projected.x + 1) * 50}%`;
      button.style.top = `${(1 - projected.y) * 50}%`;
      button.style.borderColor = `#${classColor(this, String(record.display_class || '')).getHexString()}`;
      button.style.background = 'rgba(7,16,21,.92)';
      button.style.color = '#ffffff';
      button.onclick = event => {
        event.stopPropagation();
        if (typeof this.options.onClearancePick === 'function') this.options.onClearancePick(record);
      };
      this.labelOverlay.appendChild(button);
    });
  };

  const originalStart = App.start;
  App.start = async function () {
    await originalStart();
    const state = App.state;
    const viewer = state && state.viewer;
    const toolbar = document.getElementById('viewer-toolbar');
    if (!viewer || !toolbar || document.getElementById('btn-toggle-danger')) return;

    const toggle = document.createElement('button');
    toggle.id = 'btn-toggle-danger';
    toggle.type = 'button';
    toggle.textContent = '显示危险点';
    toggle.title = '显示 exact clearance 危险点和最近树点—导线点';
    toolbar.appendChild(toggle);

    const filter = document.createElement('select');
    filter.id = 'clearance-risk-filter';
    filter.setAttribute('aria-label', '危险点风险筛选');
    filter.style.padding = '6px 8px';
    filter.style.borderRadius = '4px';
    filter.style.background = 'rgba(31,42,68,.94)';
    filter.style.color = '#fff';
    filter.style.border = '1px solid #7f94a8';
    [
      ['ALL', '全部净空'],
      ['WARNING', 'WARNING'],
      ['DANGER', 'DANGER'],
      ['REVIEW_REQUIRED', '需复核'],
    ].forEach(([value, text]) => {
      const option = document.createElement('option');
      option.value = value;
      option.textContent = text;
      filter.appendChild(option);
    });
    toolbar.appendChild(filter);

    let shown = false;
    function setShown(next) {
      shown = Boolean(next);
      viewer.setDangerVisible(shown);
      toggle.textContent = shown ? '隐藏危险点' : '显示危险点';
      toggle.style.outline = shown ? '2px solid #ff5252' : '';
    }
    toggle.onclick = () => setShown(!shown);
    filter.onchange = () => viewer.setClearanceFilter(filter.value);
    viewer.setClearanceFilter('ALL');
    setShown(false);

    viewer.options.onClearancePick = record => {
      if (Number.isFinite(Number(record.global_tree_id))) App.selectTree(Number(record.global_tree_id), false, true);
      const edit = document.getElementById('editmsg');
      if (edit) edit.textContent = `树 ${record.global_tree_id} · 档距 ${record.span_name || '—'} · 最小净空 ${Number(record.minimum_clearance_m).toFixed(3)} m · ${record.display_class || record.risk || '—'} · 导线 ${record.nearest_wire_id == null ? '—' : record.nearest_wire_id} · ${record.contact_state || '—'}`;
    };
    const originalPointPick = viewer.options.onPointPick;
    viewer.options.onPointPick = hit => {
      if (hit && hit.clearanceRecord) {
        if (typeof viewer.options.onClearancePick === 'function') viewer.options.onClearancePick(hit.clearanceRecord);
        return true;
      }
      return typeof originalPointPick === 'function' ? originalPointPick(hit) : false;
    };

    function applyAvailability(detail) {
      const authority = String(detail && detail.evidenceAuthority || 'UNAVAILABLE');
      const available = authority === 'EXACT_FINAL_CLEARANCE_NEAREST_PAIR' && Number(detail && detail.evidenceCount || 0) > 0;
      toggle.disabled = !available;
      filter.disabled = !available;
      if (!available) {
        setShown(false);
        toggle.title = authority === 'STALE_AFTER_TREE_EDIT'
          ? '树木已修改；请完成复核并重新计算净空后再显示危险点。'
          : '当前没有可显示的 exact clearance 证据。';
      } else {
        toggle.title = '显示 exact clearance 危险点和最近树点—导线点';
      }
    }

    window.addEventListener('treeline-clearance-data', event => applyAvailability(event.detail || {}));
    const info = viewer.dangerLayerInfo();
    applyAvailability({
      evidenceAuthority: info.evidence_authority,
      evidenceCount: info.evidence_count,
      dangerAuthority: info.danger_authority,
      dangerPointCount: info.danger_point_count,
    });
  };
})();
'''

__all__ = ["TREE_REVIEW_RISK_EXTENSION_JS"]
