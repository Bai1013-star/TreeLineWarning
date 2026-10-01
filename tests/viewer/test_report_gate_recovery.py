"""Browser gate polling must recover after editing and preserve stable views."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
CURRENT = ROOT / "viewer_poc" / "giro3d" / "current_report.js"
FUTURE = ROOT / "viewer_poc" / "giro3d" / "future_preview.js"


_NODE = r"""
const fs = require('fs');
const vm = require('vm');
const assert = require('assert');

class Element {
  constructor() {
    this.textContent = '';
    this.innerHTML = '';
    this.dataset = {};
    this.classNames = new Set();
    this.classList = {
      add: name => this.classNames.add(name),
      remove: name => this.classNames.delete(name),
      toggle: (name, enabled) => enabled ? this.classNames.add(name) : this.classNames.delete(name),
    };
  }
  addEventListener() {}
  replaceChildren() { this.innerHTML = ''; this.textContent = ''; }
  setAttribute() {}
}

function gate(revision) {
  return {status: 'PASS', run_id: 'run', batch_mtime_ns: revision,
    journal_mtime_ns: revision, prediction_mtime_ns: revision,
    editor_operation_count: revision, published_editor_operation_count: revision,
    final_generation_id: `final-${revision}`, result_generation_id: `result-${revision}`,
    risk_generation_id: `risk-${revision}`};
}
const locked = {status: 'LOCKED', reasons: ['编辑结果仍在生成']};
const response = value => ({ok: true, json: async () => value});
const flush = async () => { for (let i = 0; i < 12; i += 1) await new Promise(setImmediate); };

async function runCurrent(source) {
  const nodes = new Map();
  const document = {
    getElementById: id => {
      if (!nodes.has(id)) nodes.set(id, new Element());
      return nodes.get(id);
    },
    addEventListener() {},
  };
  let plots = 0, purges = 0, reads = 0;
  const callbacks = [];
  const dbhGate = {...gate(2), prediction_mtime_ns: 99};
  const gates = [gate(1), locked, gate(2), gate(2), dbhGate, dbhGate, null, gate(3)];
  const reports = [gate(1), gate(2), dbhGate, gate(3)];
  const window = {Plotly: {newPlot: () => { plots += 1; }, purge: () => { purges += 1; }}};
  const fetch = async path => {
    if (path === '/api/current-report') {
      reads += 1;
      const current = reports.shift();
      assert.ok(current, 'unexpected current report reload');
      return response({ok: true, status: 'FINALIZED', gate: current,
        kpis: {tree_count: 1}, trees: [], display: {trees3d: {xyz: []}},
        wires: {status: 'AVAILABLE', segments: [], segment_count: 0}});
    }
    if (path === '/api/future-preview/gate') {
      const next = gates.shift();
      if (next === null) throw new Error('network unavailable');
      assert.ok(next, 'unexpected current gate poll');
      return response({ok: true, gate: next});
    }
    throw new Error(`unexpected current URL ${path}`);
  };
  vm.runInNewContext(source, {window, document, fetch,
    setInterval: callback => callbacks.push(callback), setTimeout, URL, console});
  await flush();
  assert.strictEqual(plots, 1);
  callbacks[0](); await flush();
  assert.strictEqual(plots, 1, 'stable gate must preserve current plot rotation');
  callbacks[0](); await flush();
  assert.strictEqual(purges, 1, 'editing purges stale current points');
  callbacks[0](); await flush();
  assert.strictEqual(plots, 2, 'new finalized revision reloads automatically');
  callbacks[0](); await flush();
  assert.strictEqual(plots, 2, 'stable new revision does not redraw');
  callbacks[0](); await flush();
  assert.strictEqual(plots, 3, 'DBH workbook update reloads without editor change');
  callbacks[0](); await flush();
  assert.strictEqual(plots, 3, 'stable DBH publication does not redraw');
  callbacks[0](); await flush();
  assert.strictEqual(purges, 3, 'network loss clears unverified current points');
  callbacks[0](); await flush();
  assert.strictEqual(plots, 4, 'network recovery reloads finalized report');
  assert.strictEqual(reads, 4);
}

async function runFuture(source) {
  const nodes = new Map();
  const document = {
    getElementById: id => {
      if (!nodes.has(id)) nodes.set(id, new Element());
      return nodes.get(id);
    },
    querySelectorAll: () => [],
    addEventListener() {},
  };
  let plots = 0, purges = 0, reads = 0;
  const callbacks = [];
  const dbhGate = {...gate(2), prediction_mtime_ns: 99};
  const gates = [gate(1), locked, gate(2), gate(2), dbhGate, dbhGate, null, gate(3)];
  const previews = [gate(1), gate(2), dbhGate, gate(3)];
  const window = {Plotly: {newPlot: () => { plots += 1; }, purge: () => { purges += 1; },
    Plots: {resize() {}}}, addEventListener() {}};
  const fetch = async path => {
    if (path.startsWith('/api/future-preview?horizon=')) {
      reads += 1;
      const current = previews.shift();
      assert.ok(current, 'unexpected future preview reload');
      return response({schema: 'future', status: 'PASS', gate: current,
        forecast_horizon_years: 1, trees: [], wires: {segments: []}});
    }
    if (path === '/api/future-preview/gate') {
      const next = gates.shift();
      if (next === null) throw new Error('network unavailable');
      assert.ok(next, 'unexpected future gate poll');
      return response({ok: true, gate: next});
    }
    throw new Error(`unexpected future URL ${path}`);
  };
  vm.runInNewContext(source, {window, document, fetch, AbortController,
    setInterval: callback => callbacks.push(callback),
    setTimeout: () => 1, clearTimeout() {}, console});
  await flush();
  assert.strictEqual(plots, 1);
  callbacks[0](); await flush();
  assert.strictEqual(plots, 1, 'stable gate must preserve future plot rotation');
  callbacks[0](); await flush();
  assert.ok(purges >= 1, 'editing purges stale predicted points');
  callbacks[0](); await flush();
  assert.strictEqual(plots, 2, 'new finalized revision reloads future scenario');
  callbacks[0](); await flush();
  assert.strictEqual(plots, 2, 'stable new revision does not redraw');
  callbacks[0](); await flush();
  assert.strictEqual(plots, 3, 'DBH workbook update reloads future scenario');
  callbacks[0](); await flush();
  assert.strictEqual(plots, 3, 'stable DBH publication does not redraw future scene');
  callbacks[0](); await flush();
  callbacks[0](); await flush();
  assert.strictEqual(plots, 4, 'network recovery reloads future scenario');
  assert.strictEqual(reads, 4);
}

(async () => {
  await runCurrent(fs.readFileSync(process.argv[1], 'utf8'));
  await runFuture(fs.readFileSync(process.argv[2], 'utf8'));
})().catch(error => { console.error(error); process.exitCode = 1; });
"""


def test_report_and_future_gate_recover_after_new_finalized_revision() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is unavailable")
    result = subprocess.run(
        [node, "--input-type=commonjs", "-e", _NODE, str(CURRENT), str(FUTURE)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
