"""Mocked fetch checks for the Giro3D editor mutation client protocol."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
APP_JS = ROOT / "viewer_poc" / "giro3d" / "tree_review_app.js"


_NODE_CHECK = r"""
const fs = require('fs');
const vm = require('vm');
const assert = require('assert');
const realSetTimeout = setTimeout;
const realClearTimeout = clearTimeout;

const appPath = process.argv[1];
const original = fs.readFileSync(appPath, 'utf8');
const exposed = original.replace(
  'window.TreeLineTreeReviewApp = {start, state, selectTree, beginLasso, cancelTrim, metrics};',
  'window.TreeLineTreeReviewApp = {start, state, selectTree, beginLasso, cancelTrim, '
    + 'metrics, post, setEditingBusy, updateJournalCount, bindControls};'
);
if (exposed === original) throw new Error('test hook replacement did not match the app export');

function response(status, value, broken = false) {
  return {
    status,
    ok: status >= 200 && status < 300,
    json: async () => {
      if (broken) throw new Error('malformed response');
      return value;
    },
  };
}

function load(fetchImpl, options = {}) {
  const elements = new Map();
  const tbody = {innerHTML: '', appendChild: () => {}};
  const document = {
    getElementById(id) {
      if (!elements.has(id)) {
        elements.set(id, {textContent: '', disabled: false, style: {}, dataset: {}});
      }
      return elements.get(id);
    },
    querySelector: selector => selector === '#treetable tbody' ? tbody : null,
    querySelectorAll: () => [],
    createElement: () => ({getContext: () => null, dataset: {}, style: {}}),
    addEventListener: () => {},
  };
  const window = {
    location: {search: options.search || ''},
    crypto: options.crypto,
    addEventListener: () => {},
  };
  const context = {
    window,
    document,
    navigator: {},
    URLSearchParams,
    performance,
    fetch: fetchImpl,
    AbortController,
    setTimeout: (callback, delay) => {
      if (delay === 500) { callback(); return 0; }
      return realSetTimeout(callback, 0);
    },
    clearTimeout: realClearTimeout,
    Math, Number, String, Boolean, Set, Map, Array, Object, Promise, Error, TypeError,
  };
  if (options.source) context.__source = options.source;
  vm.runInNewContext(options.source || exposed, context, {filename: appPath});
  return Object.assign(window.TreeLineTreeReviewApp, {elements});
}

async function main() {
  {
    const scoped = {trees: [{tree_id: 7, point_count: 3, display_tree_id: 'T-S001-007'}]};
    const globalSnapshot = {operations: [], trees: [
      {tree_id: 7, point_count: 3}, {tree_id: 8, point_count: 4}]};
    let displayed;
    const app = load(async path => {
      if (path.startsWith('/api/editor_data')) {
        assert.ok(path.includes('span_id=S001'));
        return response(200, scoped);
      }
      if (path === '/api/undo') return response(202, {
        state: 'RUNNING', request_id: 'request-1'});
      return response(200, {state: 'COMMITTED', request_id: 'request-1',
        result: globalSnapshot});
    }, {crypto: {randomUUID: () => 'request'}, search: '?span_id=S001'});
    app.state.viewer = {
      setData: data => { displayed = data; return {visible_tree_points: 3}; },
      setLabelsVisible: () => {}, setSelection: () => {},
      rendererInfo: () => ({three_version: 'test', webgl_version: 2}),
    };
    app.updateJournalCount({tree_edit_count: 1});
    app.bindControls();
    await app.elements.get('btn-undo').onclick();
    assert.strictEqual(displayed, scoped);
    assert.strictEqual(app.state.treeMap.size, 1, 'global journal must not replace scoped rows');
    assert.strictEqual(app.state.treeMap.get(7).display_tree_id, 'T-S001-007');
    assert.strictEqual(app.state.editingBusy, false);
    assert.strictEqual(app.elements.get('btn-undo').disabled, true);
  }

  {
    const calls = [];
    let index = 0;
    const app = load(async (path, init) => {
      calls.push({path, init});
      return [
        response(202, {
          state: 'RUNNING', status: 'EDIT_RUNNING', request_id: 'request-1', job_id: 'job'}),
        response(200, {
          state: 'RUNNING', status: 'EDIT_RUNNING', request_id: 'request-1', job_id: 'job'}),
        response(200, {state: 'IDLE', status: 'EDIT_PREVIEW_READY',
          request_id: 'request-1', result: {preview_id: 'p1'}}),
      ][index++];
    }, {crypto: {randomUUID: () => 'request'}});
    const result = await app.post('/api/trim/lasso-preview', {tree_id: -1});
    assert.deepStrictEqual(result, {preview_id: 'p1'});
    assert.strictEqual(calls.length, 3);
    assert.strictEqual(JSON.parse(calls[0].init.body).async, true);
    assert.strictEqual(JSON.parse(calls[0].init.body).request_id, 'request-1');
    assert.ok(calls[1].path.startsWith('/api/editor/mutation?request_id='));
    assert.strictEqual(app.state.mutation, null);
  }

  {
    const calls = [];
    let index = 0;
    const app = load(async (path, init) => {
      calls.push({path, init});
      if (index++ === 0) throw new Error('connection reset');
      return response(200, index === 2
        ? {state: 'RUNNING', status: 'EDIT_RUNNING', request_id: 'request-1', job_id: 'job'}
        : {state: 'COMMITTED', status: 'EDIT_COMMITTED',
          request_id: 'request-1', result: {trees: []}});
    }, {crypto: {randomUUID: () => 'request'}});
    const result = await app.post('/api/delete', {});
    assert.deepStrictEqual(result, {trees: []});
    assert.strictEqual(calls.filter(call => call.init && call.init.method === 'POST').length, 1);
    assert.strictEqual(app.state.mutation, null);
  }

  {
    let index = 0;
    const app = load(async () => [
      response(202, null, true),
      response(200, {state: 'COMMITTED', status: 'EDIT_COMMITTED',
        request_id: 'request-1', result: {trees: []}}),
    ][index++], {crypto: {randomUUID: () => 'request'}});
    assert.deepStrictEqual(await app.post('/api/merge', {}), {trees: []});
    assert.strictEqual(app.state.mutation, null);
  }

  {
    const app = load(async () => response(200, null, true), {
      crypto: {randomUUID: () => 'request'}});
    await assert.rejects(
      app.post('/api/select', {tree_ids: []}),
      error => error.recoverable === true && error.responseParse === true,
    );
  }

  {
    const calls = [];
    const app = load(async (path, init) => {
      calls.push({path, init});
      return response(200, {trees: []});
    });
    app.state.previewRequestId = 17;
    await app.post('/api/undo', {});
    await app.post('/api/undo', {});
    const requestIds = calls.map(call => JSON.parse(call.init.body).request_id);
    assert.notStrictEqual(requestIds[0], requestIds[1]);
    assert.strictEqual(app.state.previewRequestId, 17);
  }

  {
    const timeoutSource = exposed.replace(
      'const MUTATION_RECOVERY_MS = 60000;', 'const MUTATION_RECOVERY_MS = 0;');
    const app = load(async () => response(202, {
      state: 'RUNNING', status: 'EDIT_RUNNING', request_id: 'request-1'}), {
      crypto: {randomUUID: () => 'request'},
      source: timeoutSource,
    });
    app.setEditingBusy(true);
    await assert.rejects(
      app.post('/api/trim/commit', {preview_id: 'p1'}),
      error => error.unknownMutation === true && /刷新或重新打开/.test(error.message),
    );
    assert.ok(app.state.mutation && app.state.mutation.uncertain === true);
    assert.strictEqual(app.state.editingBusy, true);
  }

  {
    let aborted = 0;
    let posts = 0;
    const hungStatusSource = exposed
      .replace('const MUTATION_RECOVERY_MS = 60000;', 'const MUTATION_RECOVERY_MS = 10;')
      .replace('const MUTATION_REQUEST_TIMEOUT_MS = 15000;',
        'const MUTATION_REQUEST_TIMEOUT_MS = 1;');
    const app = load(async (path, init) => {
      if (init && init.method === 'POST') {
        posts += 1;
        return response(202, {state: 'RUNNING', status: 'EDIT_RUNNING', request_id: 'request-1'});
      }
      return new Promise((resolve, reject) => {
        init.signal.addEventListener('abort', () => {
          aborted += 1;
          reject(new Error('status request aborted'));
        });
      });
    }, {crypto: {randomUUID: () => 'request'}, source: hungStatusSource});
    await assert.rejects(
      app.post('/api/delete', {}),
      error => error.unknownMutation === true,
    );
    assert.strictEqual(posts, 1);
    assert.ok(aborted >= 1);
    assert.ok(app.state.mutation && app.state.mutation.uncertain === true);
  }

  {
    const app = load(async () => response(200, {trees: []}));
    app.updateJournalCount({operations: []});
    assert.strictEqual(app.elements.get('btn-undo').disabled, true);
    app.updateJournalCount({tree_edit_count: 2});
    assert.strictEqual(app.elements.get('btn-undo').disabled, false);
  }
}

main().catch(error => {
  console.error(error);
  process.exitCode = 1;
});
"""


def test_tree_review_app_async_protocol_with_mocked_fetch() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not available")
    result = subprocess.run(
        [node, "--input-type=commonjs", "-e", _NODE_CHECK, str(APP_JS)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
