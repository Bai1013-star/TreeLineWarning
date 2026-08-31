"""Focused browser-less checks for the human DBH review page."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
HTML = ROOT / "viewer_poc" / "giro3d" / "dbh_review.html"
CSS = ROOT / "viewer_poc" / "giro3d" / "dbh_review.css"
JS = ROOT / "viewer_poc" / "giro3d" / "dbh_review.js"
STEP2 = ROOT / "algorithms" / "folder_pipeline" / "html.py"
TREE_REVIEW_APP = ROOT / "viewer_poc" / "giro3d" / "tree_review_app.js"
CURRENT_REPORT = ROOT / "viewer_poc" / "giro3d" / "current_report.js"


_NODE_BEHAVIOR = r"""
const fs = require('fs');
const vm = require('vm');
const assert = require('assert');
const realSetTimeout = setTimeout;
const realClearTimeout = clearTimeout;

const source = fs.readFileSync(process.argv[1], 'utf8');

function response(status, value) {
  return {
    status,
    ok: status >= 200 && status < 300,
    json: async () => value,
  };
}

class ClassList {
  constructor() { this.values = new Set(); }
  add(...names) { names.forEach(name => this.values.add(name)); }
  remove(...names) { names.forEach(name => this.values.delete(name)); }
  toggle(name, force) {
    const next = force === undefined ? !this.values.has(name) : Boolean(force);
    if (next) this.values.add(name); else this.values.delete(name);
    return next;
  }
  contains(name) { return this.values.has(name); }
}

class Element {
  constructor(tag = 'div') {
    this.tagName = tag.toUpperCase();
    this.children = [];
    this.dataset = {};
    this.attributes = {};
    this.listeners = {};
    this.classList = new ClassList();
    this.style = {setProperty: (name, value) => { this.style[name] = value; }};
    this.disabled = false;
    this.checked = false;
    this.clientWidth = 640;
    this.clientHeight = 480;
    this._innerHTML = '';
    this.textContent = '';
  }
  get innerHTML() { return this._innerHTML; }
  set innerHTML(value) {
    this._innerHTML = String(value ?? '');
    if (!this._innerHTML) this.children = [];
  }
  setAttribute(name, value) { this.attributes[name] = String(value); }
  addEventListener(name, callback) { this.listeners[name] = callback; }
  appendChild(child) { this.children.push(child); return child; }
  getBoundingClientRect() {
    return {left: 0, top: 0, width: this.clientWidth, height: this.clientHeight};
  }
  click() {
    if (!this.disabled && this.listeners.click) return this.listeners.click({target: this});
  }
}

function makeDocument() {
  const elements = new Map();
  const document = {
    getElementById(id) {
      if (!elements.has(id)) elements.set(id, new Element());
      return elements.get(id);
    },
    querySelectorAll(selector) {
      if (selector === '#tree-list .tree-entry') {
        return document.getElementById('tree-list').children;
      }
      return [];
    },
    createElement: tag => new Element(tag),
    createElementNS: (_namespace, tag) => new Element(tag),
    addEventListener: () => {},
  };
  return {document, elements};
}

class MockViewer {
  constructor(host, labels, options) {
    this.host = host;
    this.labels = labels;
    this.options = options;
    this.data = null;
    this.selection = [];
    this.preview = [];
    this.disposed = false;
  }
  setData(data) { this.data = data; return {visible_tree_points: data.trees3d.xyz.length}; }
  setSelection(ids) { this.selection = [...ids]; }
  setPreview(points, id) { this.preview = [points, id]; }
  setLabelsVisible() {}
  currentViewProjection() { return [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]; }
  resetView() {}
  topView() {}
  sideView() {}
  dispose() { this.disposed = true; }
}

function load(fetchImpl, options = {}) {
  const {document, elements} = makeDocument();
  const window = {
    crypto: {randomUUID: () => 'dbh-test-request'},
    addEventListener: () => {},
    TreeLineTreeReview3D: MockViewer,
  };
  const context = {
    window,
    document,
    navigator: {hardwareConcurrency: 8, deviceMemory: 8},
    fetch: fetchImpl,
    setTimeout: options.timed
      ? (callback, _delay) => realSetTimeout(callback, 5)
      : (callback, _delay) => { callback(); return 0; },
    clearTimeout: options.timed ? realClearTimeout : () => {},
    AbortController,
    performance: {now: () => 0},
    URLSearchParams,
    Math, Number, String, Boolean, Set, Map, Array, Object, Promise, Error, TypeError,
  };
  vm.runInNewContext(source, context, {filename: process.argv[1]});
  return {app: window.TreeLineDbhReview, elements, window};
}

function reviewPayload(reviewId = 'review-1') {
  return {
    ok: true,
    review: {
      review_id: reviewId,
      persistent_tree_id: 'p1',
      global_tree_id: 11,
      display_tree_id: 'T-11',
      editor_revision: 'rev-1',
      mode: 'CANDIDATE_OR_SEED',
      candidates: [
        {
          candidate_id: 'axis-a', candidate_score: 0.91,
          candidate_axis: {
            bh_reference_xyz: [0, 0, 1.3], direction_xyz: [0, 0, 1],
            slope_xyz_per_hag: [0, 0, 0], radius_m: 0.09,
          }, candidate_support: 42, continuity_status: 'PASS',
        },
        {
          candidate_id: 'axis-b', candidate_score: 0.78,
          candidate_axis: {
            bh_reference_xyz: [0.4, 0, 1.3], direction_xyz: [0, 0, 1],
            slope_xyz_per_hag: [0, 0, 0], radius_m: 0.08,
          }, candidate_support: 31, continuity_status: 'REVIEW',
        },
      ],
      display: {
        xyz: [[0, 0, 1.2], [0.01, 0, 1.35], [0.02, 0, 1.5]],
        original_index: [100, 101, 102],
        hag_m: [1.2, 1.35, 1.5],
        selectable: [true, true, false],
      },
      current_crown: {xyz: [[0, 0, 2.2]], label: 'T-11 crown'},
      status: 'REVIEW_READY',
    },
  };
}

async function main() {
  const calls = [];
  let activeMutation = null;
  let prepareCount = 0;
  let confirmCount = 0;
  const fetchImpl = async (path, init = {}) => {
    calls.push({path, init});
    if (path === '/api/dbh-review/trees') {
      return response(200, {
        ok: true,
        human_can_type_dbh: false,
        trees: [
          {persistent_tree_id: 'p1', global_tree_id: 11,
            display_tree_id: 'T-11', dbh_cm: null, dbh_method: 'UNAVAILABLE'},
          {persistent_tree_id: 'p2', global_tree_id: 12,
            display_tree_id: 'T-12', dbh_cm: 18, dbh_method: 'DIRECT_STEM'},
        ],
      });
    }
    if (path === '/api/dbh-review/prepare') {
      prepareCount += 1;
      activeMutation = {kind: 'prepare', id: JSON.parse(init.body).request_id};
      return response(202, {state: 'RUNNING', request_id: activeMutation.id});
    }
    if (path === '/api/dbh-review/confirm') {
      confirmCount += 1;
      activeMutation = {kind: 'confirm', id: JSON.parse(init.body).request_id};
      return response(202, {state: 'RUNNING', request_id: activeMutation.id});
    }
    if (path.startsWith('/api/editor/mutation?request_id=')) {
      if (activeMutation.kind === 'prepare') {
        return response(200, {
          ...reviewPayload(`review-${prepareCount}`), request_id: activeMutation.id,
        });
      }
      return response(200, {
        ok: true, request_id: activeMutation.id, status: 'RECOVERED', dbh_cm: 18.2,
        dbh_method: 'REVIEW_CONFIRMED_STEM', dbh_quality: 'HIGH',
        dbh_fit_status: 'PASS', message: 'geometry pass',
      });
    }
    throw new Error(`unexpected request ${path}`);
  };

  const {app, elements} = load(fetchImpl);
  await app.loadTrees();
  app.selectTree('p1');
  assert.strictEqual(elements.get('btn-prepare').disabled, false);
  app.selectTree('p2');
  assert.strictEqual(app.state.activeTreeId, 'p1', 'direct trees stay locked');

  const firstPrepare = app.prepareTree('p1');
  const duplicatePrepare = app.prepareTree('p1');
  assert.strictEqual(prepareCount, 1, 'pending prepare calls share one mutation');
  await Promise.all([firstPrepare, duplicatePrepare]);
  assert.strictEqual(app.state.selectedCandidateId, null, 'review never auto-selects an axis');
  assert.strictEqual(app.state.selectedSeedOriginalIndex, null, 'review never auto-selects a seed');
  assert.strictEqual(
    app.state.viewer.data.wires3d.xyz.length, 0,
    'candidate axes remain independent overlay lines');
  assert.strictEqual(
    elements.get('dbh-axis-overlay').children.length, 2,
    'all candidate axes are rendered distinctly');

  app.selectCandidate('axis-a');
  assert.strictEqual(app.state.selectedCandidateId, 'axis-a');
  assert.strictEqual(app.state.selectedSeedOriginalIndex, null);
  assert.strictEqual(elements.get('dbh-axis-overlay').children[0].attributes['stroke-width'], '5');
  elements.get('identity-confirmed').checked = true;
  await app.confirmReview();
  const candidateBody = JSON.parse(
    calls.find(call => call.path === '/api/dbh-review/confirm').init.body);
  assert.strictEqual(candidateBody.identity_confirmed, true);
  assert.strictEqual(candidateBody.candidate_id, 'axis-a');
  assert.ok(!Object.prototype.hasOwnProperty.call(candidateBody, 'seed_original_index'));
  assert.strictEqual(confirmCount, 1);
  assert.strictEqual(app.state.completed, true);
  assert.strictEqual(app.state.treeMap.get('p1').dbh_method, 'REVIEW_CONFIRMED_STEM');

  const {app: seedApp, elements: seedElements} = load(fetchImpl);
  await seedApp.loadTrees();
  seedApp.selectTree('p1');
  await seedApp.prepareTree('p1');
  seedApp.enterSeedMode();
  seedApp.selectSeedIndex(101);
  assert.strictEqual(
    seedApp.state.selectedSeedOriginalIndex, 101,
    'seed selection uses exact original_index');
  assert.strictEqual(seedApp.state.selectedCandidateId, null);
  seedApp.selectSeedIndex(102);
  assert.strictEqual(
    seedApp.state.selectedSeedOriginalIndex, 101,
    'nonselectable rows cannot replace selection');
  seedElements.get('identity-confirmed').checked = true;
  await seedApp.confirmReview();
  const confirmBodies = calls.filter(call => call.path === '/api/dbh-review/confirm');
  const seedBody = JSON.parse(confirmBodies[confirmBodies.length - 1].init.body);
  assert.strictEqual(seedBody.seed_original_index, 101);
  assert.ok(!Object.prototype.hasOwnProperty.call(seedBody, 'candidate_id'));

  let uncertainPosts = 0;
  const uncertainFetch = async (path, init = {}) => {
    if (path === '/api/dbh-review/trees') {
      return response(200, {ok: true, human_can_type_dbh: false,
        trees: [{persistent_tree_id: 'p1', dbh_method: 'UNAVAILABLE'}]});
    }
    if (path === '/api/dbh-review/prepare') {
      uncertainPosts += 1;
      return response(202, {state: 'RUNNING', request_id: JSON.parse(init.body).request_id});
    }
    throw new Error('connection reset while polling');
  };
  const uncertain = load(uncertainFetch).app;
  await uncertain.loadTrees();
  uncertain.selectTree('p1');
  assert.strictEqual(await uncertain.prepareTree(), null);
  assert.strictEqual(uncertain.state.uncertain, true);
  assert.strictEqual(await uncertain.prepareTree(), null);
  assert.strictEqual(uncertainPosts, 1, 'uncertain mutation is never automatically repeated');

  let mismatchPosts = 0;
  const mismatchFetch = async (path, init = {}) => {
    if (path === '/api/dbh-review/trees') {
      return response(200, {ok: true, human_can_type_dbh: false,
        trees: [{persistent_tree_id: 'p1', dbh_method: 'UNAVAILABLE'}]});
    }
    if (path === '/api/dbh-review/prepare') {
      mismatchPosts += 1;
      return response(202, {state: 'RUNNING', request_id: JSON.parse(init.body).request_id});
    }
    return response(200, {
      ok: true, request_id: 'different-request', status: 'COMMITTED',
      review: reviewPayload('wrong-request').review,
    });
  };
  const mismatch = load(mismatchFetch).app;
  await mismatch.loadTrees();
  mismatch.selectTree('p1');
  assert.strictEqual(await mismatch.prepareTree(), null);
  assert.strictEqual(mismatch.state.uncertain, true, 'mismatched task records fail closed');
  assert.strictEqual(await mismatch.prepareTree(), null);
  assert.strictEqual(mismatchPosts, 1, 'mismatched records are never retried');

  let bodyPolls = 0;
  const stalledBodyFetch = async (path, init = {}) => {
    if (path === '/api/dbh-review/trees') {
      return response(200, {ok: true, human_can_type_dbh: false,
        trees: [{persistent_tree_id: 'p1', dbh_method: 'UNAVAILABLE'}]});
    }
    if (path === '/api/dbh-review/prepare') {
      return response(202, {state: 'RUNNING', request_id: JSON.parse(init.body).request_id});
    }
    bodyPolls += 1;
    return {
      status: 200,
      ok: true,
      json: () => new Promise((resolve, reject) => {
        if (init.signal && init.signal.aborted) {
          reject(new Error('body aborted'));
          return;
        }
        init.signal.addEventListener('abort', () => reject(new Error('body aborted')));
      }),
    };
  };
  const stalled = load(stalledBodyFetch, {timed: true}).app;
  await stalled.loadTrees();
  stalled.selectTree('p1');
  assert.strictEqual(await stalled.prepareTree(), null);
  assert.strictEqual(bodyPolls, 1, 'stalled response bodies are bounded');
  assert.strictEqual(stalled.state.uncertain, true, 'stalled mutation bodies fail closed');
}

main().catch(error => {
  console.error(error);
  process.exitCode = 1;
});
"""


def test_dbh_review_assets_and_behavior() -> None:
    html = HTML.read_text(encoding="utf-8")
    css = CSS.read_text(encoding="utf-8")
    js = JS.read_text(encoding="utf-8")
    step2 = STEP2.read_text(encoding="utf-8")
    app = TREE_REVIEW_APP.read_text(encoding="utf-8")
    report = CURRENT_REPORT.read_text(encoding="utf-8")

    assert 'href="/dbh_review.css"' in html
    assert 'src="/dbh_review.js"' in html
    assert 'src="/tree-review-3d.js"' in html
    assert 'id="dbh-axis-overlay"' in html
    assert 'id="identity-confirmed"' in html
    assert "该树干点云不足，无法可靠计算胸径" in js
    assert "human_can_type_dbh !== false" in js
    assert "/api/dbh-review/trees" in js
    assert "/api/dbh-review/prepare" in js
    assert "/api/dbh-review/confirm" in js
    assert "/api/editor/mutation?request_id=" in js
    assert "async: true" in js
    assert "request_id" in js
    assert "600000" in js
    assert "original_index" in js
    assert "selectable" in js
    assert 'type="number"' not in html
    assert "手工输入胸径" not in html
    assert ".dbh-axis-line.selected" in css
    assert "#dbh-plot3d canvas" in css and "width: 100%" in css and "height: 100%" in css
    assert 'id="btn-dbh-review"' in step2
    assert "window.location.href = '/dbh-review'" in app
    assert "btn-dbh-review" in app
    assert "REVIEW_CONFIRMED_STEM: '人工确认点云'" in js
    assert "REVIEW_CONFIRMED_STEM: '人工确认点云'" in report

    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not available")
    result = subprocess.run(
        [node, "--input-type=commonjs", "-e", _NODE_BEHAVIOR, str(JS)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("path", [JS, TREE_REVIEW_APP, CURRENT_REPORT])
def test_dbh_review_related_javascript_syntax(path: Path) -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not available")
    result = subprocess.run(
        [node, "--check", str(path)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
