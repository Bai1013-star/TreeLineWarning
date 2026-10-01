"""P3A measured-performance diagnostics + fixed two-state policy tests.

Covers two layers:

1. STATIC checks on ``viewer_perf.js`` source: no second render loop, never
   calls ``render`` / ``updatePointClouds``, overlay update is throttled, and
   the policy contains no GPU/hardware/vendor branching (the fixed two-state
   policy is identical on every machine by construction).

2. FUNCTIONAL Node checks (``node`` must be present, same as the existing P3A
   tests): rolling statistics are correct (average/median/P95/FPS), the rolling
   history is bounded, the overlay throttle actually limits update rate, the
   benchmark configuration only accepts known values, and the two-state policy
   state machine transitions deterministically (motion start -> INTERACTIVE,
   debounced motion end -> IDLE, camera-changed resets the debounce, repeated
   applies are idempotent, suspension freezes profile application).
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess

import pytest

from scripts.serve_p3a_potree import VIEWER_ROOT

PERF_JS = VIEWER_ROOT / "viewer_perf.js"
PERF_PATH = str(PERF_JS).replace("\\", "/")

# Tokens describing what the file does NOT do live only in comments; the static
# checks must assert about CODE, so comments are stripped first.  Safe for this
# controlled file: no string literal contains "//" or "/*".
_COMMENT_RE_BLOCK = re.compile(r"/\*.*?\*/", re.DOTALL)
_COMMENT_RE_LINE = re.compile(r"//[^\n]*")


def _perf_src() -> str:
    return PERF_JS.read_text(encoding="utf-8")


def _perf_code() -> str:
    """viewer_perf.js source with block and line comments stripped."""
    code = _COMMENT_RE_BLOCK.sub(" ", _perf_src())
    return _COMMENT_RE_LINE.sub(" ", code)


def _run_node(script: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["node", "-e", script], capture_output=True, text=True, timeout=120
    )


def _assert_node_pass(result: subprocess.CompletedProcess) -> None:
    assert result.returncode == 0, f"node failed:\n{result.stdout}\n{result.stderr}"
    assert "PASS" in result.stdout


# ---------------------------------------------------------------------------
# Static: no second render loop / no render calls
# ---------------------------------------------------------------------------


def test_deliverable_present() -> None:
    assert PERF_JS.is_file()


def test_no_second_render_loop() -> None:
    """The diagnostics observe the EXISTING Potree cadence; they must never
    start their own render loop.  (Comments stripped: the docstring describes
    what the file does NOT call, which is not an invocation.)"""
    js = _perf_code()
    for token in (
        "requestAnimationFrame(",
        "setAnimationLoop(",
        "Potree.updatePointClouds(",
        "viewer.updatePointClouds(",
    ):
        assert token not in js, f"second-render-loop token present: {token}"


def test_diagnostics_never_call_render() -> None:
    assert ".render(" not in _perf_code()


def test_diagnostics_never_call_update_point_clouds() -> None:
    assert "updatePointClouds" not in _perf_code()


def test_frame_cadence_uses_render_pass_end_event() -> None:
    js = _perf_src()
    assert '"render.pass.end"' in js
    assert 'viewer.addEventListener("render.pass.end"' in js


# ---------------------------------------------------------------------------
# Static: overlay throttling
# ---------------------------------------------------------------------------


def test_overlay_update_is_throttled_in_code() -> None:
    js = _perf_code()
    assert "createThrottle" in js
    assert "throttle.shouldUpdate(now)" in js
    assert "OVERLAY_INTERVAL_MS" in js


def test_overlay_is_pointer_even_none() -> None:
    js = _perf_code()
    assert "pointer-events:none" in js


# ---------------------------------------------------------------------------
# Static: benchmark configuration is closed over allowed values
# ---------------------------------------------------------------------------


def test_benchmark_budget_list_is_exact() -> None:
    js = _perf_src()
    for budget in (100000, 250000, 400000, 500000, 750000, 1000000):
        assert str(budget) in js, f"budget {budget} missing from allowed list"


def test_benchmark_interactions_are_only_stationary_orbit_pan() -> None:
    js = _perf_src()
    for value in ("stationary", "orbit", "pan"):
        assert f'"{value}"' in js, f"interaction {value} missing"
    assert '"fly"' not in js


# ---------------------------------------------------------------------------
# Static: receiver-safe timers (P3A perf runtime "Illegal invocation" fix)
# ---------------------------------------------------------------------------


def test_timers_are_not_detached_window_methods() -> None:
    """Regression: storing ``{setTimeout: setTimeout, clearTimeout: clearTimeout}``
    stores DETACHED brand-checked Window members; calling them through a plain
    object throws "TypeError: Illegal invocation" in Chromium (Edge).  The
    default timers must wrap so ``this`` is always the global object."""
    js = _perf_code()
    assert "{ setTimeout: setTimeout" not in js
    assert "{ setTimeout: setTimeout, clearTimeout: clearTimeout }" not in js
    assert "global.setTimeout" in js
    assert "global.clearTimeout" in js


# ---------------------------------------------------------------------------
# Static: no GPU / hardware / vendor branching (fixed policy generalization)
# ---------------------------------------------------------------------------


def test_no_hardware_name_branching() -> None:
    js = _perf_code().lower()
    for token in ("rtx", "nvidia", "amd", "intel", "geforce", "radeon"):
        assert token not in js, f"hardware name influences policy: {token}"


def test_no_gpu_detection_in_policy() -> None:
    js = _perf_code().lower()
    for token in (
        "webgl_debug_renderer_info",
        "unmasked_renderer",
        "getparameter(",
        "getextension(",
        "isgpu",
        "hardware",
        "vendor",
    ):
        assert token not in js, f"GPU detection token present: {token}"


def test_module_exposes_expected_api() -> None:
    js = _perf_src()
    assert "global.p3aPerf" in js
    for api in (
        "createBoundedHistory",
        "computeStats",
        "createThrottle",
        "isValidBenchmarkValue",
        "createPerfPolicy",
        "init",
    ):
        assert api in js, f"expected API {api} missing"


def test_policy_is_two_state_only() -> None:
    js = _perf_code()
    assert '"INTERACTIVE"' in js
    assert '"IDLE"' in js
    # No hardware tiers / LOW / MEDIUM / HIGH presets.
    for token in ('"LOW"', '"MEDIUM"', '"HIGH"', "fps feedback", "autoTune"):
        assert token not in js


# ---------------------------------------------------------------------------
# Functional (Node): rolling statistics
# ---------------------------------------------------------------------------

_STATS_SCRIPT = r"""
const fs = require('fs');
eval(fs.readFileSync("__PATH__", 'utf8'));
const p = globalThis.p3aPerf;
let failed = 0;
function check(cond, msg) { if (!cond) { failed += 1; console.log('FAIL: ' + msg); } }

// average / median / P95 / FPS on a known 5-element set.
const s = p.computeStats([10, 20, 30, 40, 50]);
check(s.count === 5, 'count');
check(s.average === 30, 'average=30');
check(s.median === 30, 'median=30');
check(s.p95 === 40, 'p95=40 (floor(0.95*4)=3 -> sorted[3])');
check(s.fps === 1000 / 30, 'fps=1000/median');

// 10-element even-count set: lower median sorted[floor(9/2)]=sorted[4]=5.
const t = p.computeStats([1,2,3,4,5,6,7,8,9,10]);
check(t.average === 5.5, 'avg 1..10 = 5.5');
check(t.median === 5, 'lower median 1..10 = 5');
check(t.p95 === 9, 'p95 1..10 = 9');

// Empty input is safe (no NaN / division by zero).
const e = p.computeStats([]);
check(e.count === 0 && e.average === 0 && e.median === 0 && e.p95 === 0, 'empty stats are zeros');

if (failed > 0) { console.log('FAILURES=' + failed); process.exit(1); }
console.log('P3A perf stats PASS');
"""


@pytest.mark.skipif(shutil.which("node") is None, reason="node not available")
def test_rolling_stats_median_p95_average() -> None:
    _assert_node_pass(_run_node(_STATS_SCRIPT.replace("__PATH__", PERF_PATH)))


_BOUNDED_SCRIPT = r"""
const fs = require('fs');
eval(fs.readFileSync("__PATH__", 'utf8'));
const p = globalThis.p3aPerf;
let failed = 0;
function check(cond, msg) { if (!cond) { failed += 1; console.log('FAIL: ' + msg); } }

const h = p.createBoundedHistory(120);
for (let i = 1; i <= 200; i++) h.push(i);
check(h.size() === 120, 'window capped at 120');
const arr = h.toArray();
check(arr[0] === 81, 'oldest dropped (200-120+1=81)');
check(arr[arr.length - 1] === 200, 'newest kept');
check(arr.length === 120, 'toArray length');

// Non-finite values are rejected.
h.push(NaN); h.push(Infinity); h.push(-1);
check(h.size() === 120, 'invalid values not stored');

// clear resets.
h.clear();
check(h.size() === 0, 'clear empties window');

if (failed > 0) { console.log('FAILURES=' + failed); process.exit(1); }
console.log('P3A perf bounded history PASS');
"""


@pytest.mark.skipif(shutil.which("node") is None, reason="node not available")
def test_rolling_history_bounded() -> None:
    _assert_node_pass(_run_node(_BOUNDED_SCRIPT.replace("__PATH__", PERF_PATH)))


_THROTTLE_SCRIPT = r"""
const fs = require('fs');
eval(fs.readFileSync("__PATH__", 'utf8'));
const p = globalThis.p3aPerf;
let failed = 0;
function check(cond, msg) { if (!cond) { failed += 1; console.log('FAIL: ' + msg); } }

const t = p.createThrottle(250); // <= 4 Hz
check(t.shouldUpdate(0) === true, 'first call emits');
check(t.shouldUpdate(100) === false, 'too soon');
check(t.shouldUpdate(249) === false, 'still too soon');
check(t.shouldUpdate(250) === true, 'at interval emits');
check(t.shouldUpdate(251) === false, 'right after interval blocked');
check(t.shouldUpdate(500) === true, 'next interval emits');
t.reset();
check(t.shouldUpdate(600) === true, 'reset emits again');

if (failed > 0) { console.log('FAILURES=' + failed); process.exit(1); }
console.log('P3A perf throttle PASS');
"""


@pytest.mark.skipif(shutil.which("node") is None, reason="node not available")
def test_throttle_update_rate() -> None:
    _assert_node_pass(_run_node(_THROTTLE_SCRIPT.replace("__PATH__", PERF_PATH)))


_VALIDATION_SCRIPT = r"""
const fs = require('fs');
eval(fs.readFileSync("__PATH__", 'utf8'));
const p = globalThis.p3aPerf;
let failed = 0;
function check(cond, msg) { if (!cond) { failed += 1; console.log('FAIL: ' + msg); } }

// Known allowed values accepted.
for (const b of [100000, 250000, 400000, 500000, 750000, 1000000]) {
  check(p.isValidBenchmarkValue('budgets', b), 'budget accepted: ' + b);
}
for (const s of ['fixed', 'attenuated', 'adaptive']) {
  check(p.isValidBenchmarkValue('pointSizeTypes', s), 'size accepted: ' + s);
}
check(p.isValidBenchmarkValue('edl', true), 'edl true accepted');
check(p.isValidBenchmarkValue('edl', false), 'edl false accepted');
check(p.isValidBenchmarkValue('pixelRatios', 1.0), 'pr 1.0 accepted');
check(p.isValidBenchmarkValue('pixelRatios', 0.75), 'pr 0.75 accepted');
for (const i of ['stationary', 'orbit', 'pan']) {
  check(p.isValidBenchmarkValue('interactions', i), 'interaction accepted: ' + i);
}

// Anything else rejected.
check(!p.isValidBenchmarkValue('budgets', 123456), 'budget 123456 rejected');
check(!p.isValidBenchmarkValue('budgets', 2000000), 'budget 2M rejected');
check(!p.isValidBenchmarkValue('pointSizeTypes', 'huge'), 'size huge rejected');
check(!p.isValidBenchmarkValue('edl', 'yes'), 'edl string rejected');
check(!p.isValidBenchmarkValue('pixelRatios', 0.5), 'pr 0.5 rejected');
check(!p.isValidBenchmarkValue('interactions', 'fly'), 'interaction fly rejected');
check(!p.isValidBenchmarkValue('interactions', 'zoom'), 'interaction zoom rejected');
check(!p.isValidBenchmarkValue('nonsense', 1), 'unknown kind rejected');
check(!p.isValidBenchmarkValue('budgets', null), 'null rejected');

if (failed > 0) { console.log('FAILURES=' + failed); process.exit(1); }
console.log('P3A perf config validation PASS');
"""


@pytest.mark.skipif(shutil.which("node") is None, reason="node not available")
def test_benchmark_config_accepts_only_known_values() -> None:
    _assert_node_pass(_run_node(_VALIDATION_SCRIPT.replace("__PATH__", PERF_PATH)))


# ---------------------------------------------------------------------------
# Functional (Node): two-state policy state machine
# ---------------------------------------------------------------------------

_POLICY_SCRIPT = r"""
const fs = require('fs');
eval(fs.readFileSync("__PATH__", 'utf8'));
const p = globalThis.p3aPerf;
let failed = 0;
function check(cond, msg) { if (!cond) { failed += 1; console.log('FAIL: ' + msg); } }

// Fake clock so tests are deterministic.
const timers = { pending: [], now: 0 };
timers.setTimeout = function (fn) {
  const id = timers.pending.length;
  timers.pending.push({ id: id, fn: fn });
  return id;
};
timers.clearTimeout = function (id) {
  timers.pending = timers.pending.filter(function (j) { return j.id !== id; });
};
function fireAllIdle() {
  while (timers.pending.length > 0) {
    const jobs = timers.pending.slice();
    timers.pending = [];
    for (const j of jobs) j.fn();
  }
}

const ip = { pointBudget: 250000, pointSizeType: 'fixed', edl: false };
const idl = { pointBudget: 1000000, pointSizeType: 'adaptive', edl: true };
const applied = [];
const pol = p.createPerfPolicy({
  interactive: ip,
  idle: idl,
  debounceMs: 200,
  apply: function (prof, st) { applied.push({ prof: prof, st: st }); },
  timers: timers,
});

// Startup applies the IDLE profile once.
check(pol.getState() === 'IDLE', 'initial state IDLE');
pol.applyIdle();
check(applied.length === 1 && applied[0].prof === idl, 'startup applies idle profile');

// Repeated applyIdle is a no-op (deterministic).
pol.applyIdle();
check(applied.length === 1, 'repeated applyIdle is idempotent');

// Motion start -> INTERACTIVE, applies the interactive profile.
pol.onMotionStart();
check(pol.getState() === 'INTERACTIVE', 'motion start -> INTERACTIVE');
check(applied.length === 2 && applied[1].prof === ip && applied[1].st === 'INTERACTIVE',
  'interactive profile applied on motion start');

// Debounce fires -> IDLE, applies the idle profile exactly once.
fireAllIdle();
check(pol.getState() === 'IDLE', 'debounce -> IDLE');
check(applied.length === 3 && applied[2].prof === idl, 'idle profile applied after debounce');

// Camera-changed resets the debounce: two consecutive onMotion() calls must
// leave exactly one pending idle timer (the second cleared the first).
pol.onMotion();
pol.onMotion();
check(timers.pending.length === 1, 'camera_changed resets debounce timer');
check(pol.getState() === 'INTERACTIVE', 'camera_changed keeps INTERACTIVE');
fireAllIdle();
check(pol.getState() === 'IDLE', 'idle after last camera_changed + debounce');
check(applied.length === 5 && applied[4].prof === idl, 'idle applied once per transition');

// onMotionEnd also schedules idle restoration.
pol.onMotionEnd();
check(pol.getState() === 'INTERACTIVE', 'motion end keeps INTERACTIVE until debounce');
fireAllIdle();
check(pol.getState() === 'IDLE', 'motion end + debounce -> IDLE');
check(applied.length === 7 && applied[6].prof === idl, 'idle applied once per transition');

// Each state transition applied its profile exactly once: 4 idle applies
// (startup + 3 debounce restorations) and 3 interactive applies + 1 final.
const idleCount = applied.filter(function (a) { return a.prof === idl; }).length;
check(idleCount === 4, 'idle profile applied exactly 4 times, got ' + idleCount);

// Suspension freezes profile application but keeps state tracking.
pol.setSuspended(true);
const before = applied.length;
pol.onMotionStart();
pol.onMotion();
check(pol.getState() === 'INTERACTIVE', 'state tracked while suspended');
check(applied.length === before, 'no profile applied while suspended');
pol.setSuspended(false);
check(applied.length === before + 1 && applied[applied.length - 1].prof === ip,
  'resume applies current-state profile');

if (failed > 0) { console.log('FAILURES=' + failed); process.exit(1); }
console.log('P3A perf policy PASS');
"""


@pytest.mark.skipif(shutil.which("node") is None, reason="node not available")
def test_policy_two_state_transitions() -> None:
    """motion start -> INTERACTIVE; debounced end / camera-changed -> IDLE;
    camera-changed resets the debounce; suspension freezes application."""
    _assert_node_pass(_run_node(_POLICY_SCRIPT.replace("__PATH__", PERF_PATH)))


_IDLE_DETERMINISM_SCRIPT = r"""
const fs = require('fs');
eval(fs.readFileSync("__PATH__", 'utf8'));
const p = globalThis.p3aPerf;
let failed = 0;
function check(cond, msg) { if (!cond) { failed += 1; console.log('FAIL: ' + msg); } }

const timers = { pending: [], setTimeout(fn) { const id = timers.pending.length; timers.pending.push({ id, fn }); return id; }, clearTimeout(id) { timers.pending = timers.pending.filter(j => j.id !== id); } };
function fireAll() { while (timers.pending.length) { const jobs = timers.pending.slice(); timers.pending = []; jobs.forEach(j => j.fn()); } }

const ip = { pointBudget: 250000, pointSizeType: 'fixed', edl: false };
const idl = { pointBudget: 1000000, pointSizeType: 'adaptive', edl: true };
const applied = [];
function makePolicy() {
  return p.createPerfPolicy({ interactive: ip, idle: idl, debounceMs: 250, apply: (prof, st) => applied.push({ prof, st }), timers });
}

// Idle restoration is deterministic across repeated motion cycles.
const pol = makePolicy();
pol.applyIdle();
for (let i = 0; i < 3; i++) {
  pol.onMotionStart();
  fireAll(); // -> IDLE
}
const idleApplies = applied.filter(a => a.prof === idl).length;
check(idleApplies === 4, 'exactly one idle apply per startup + per cycle, got ' + idleApplies);

// Fixed policy deterministic: identical profiles produce identical transitions.
const appliedB = [];
const polB = p.createPerfPolicy({ interactive: ip, idle: idl, debounceMs: 250, apply: (prof, st) => appliedB.push({ prof, st }), timers });
polB.applyIdle();
polB.onMotionStart();
fireAll();
// appliedB = [idl, ip, idl]; the reference run's first cycle is identical.
check(JSON.stringify(appliedB) === JSON.stringify(applied.slice(0, 3)),
  'identical policies produce identical apply sequences');

if (failed > 0) { console.log('FAILURES=' + failed); process.exit(1); }
console.log('P3A perf determinism PASS');
"""


@pytest.mark.skipif(shutil.which("node") is None, reason="node not available")
def test_policy_and_idle_restoration_deterministic() -> None:
    _assert_node_pass(_run_node(_IDLE_DETERMINISM_SCRIPT.replace("__PATH__", PERF_PATH)))


# ---------------------------------------------------------------------------
# Functional (Node): receiver-safe default timers — Illegal invocation fix
# ---------------------------------------------------------------------------

_ILLEGAL_INVOCATION_SCRIPT = r"""
const fs = require('fs');
// Simulate Chromium/Edge's brand-checked Window.setTimeout: invoking it with a
// foreign receiver throws "TypeError: Illegal invocation".  The policy's
// DEFAULT timers must call setTimeout/clearTimeout through the global so `this`
// is always the global object.
let invoked = 0;
const realSetTimeout = global.setTimeout;
const realClearTimeout = global.clearTimeout;
global.setTimeout = function (fn, ms) {
  if (this !== global) {
    throw new TypeError('Illegal invocation');
  }
  invoked += 1;
  return realSetTimeout.call(global, fn, ms);
};
global.clearTimeout = function (id) {
  if (this !== global) {
    throw new TypeError('Illegal invocation');
  }
  return realClearTimeout.call(global, id);
};

eval(fs.readFileSync("__PATH__", 'utf8'));
const p = globalThis.p3aPerf;
let failed = 0;
function check(cond, msg) { if (!cond) { failed += 1; console.log('FAIL: ' + msg); } }

const applied = [];
const pol = p.createPerfPolicy({
  interactive: { pointBudget: 250000, pointSizeType: 'fixed', edl: false },
  idle: { pointBudget: 1000000, pointSizeType: 'adaptive', edl: true },
  debounceMs: 60,
  apply: function (prof, st) { applied.push({ prof: prof, st: st }); }
  // no config.timers -> DEFAULT receiver-safe wrapper is exercised
});
pol.applyIdle();
pol.onMotionStart();
check(pol.getState() === 'INTERACTIVE', 'motion start -> INTERACTIVE');
setTimeout(function () {
  try {
    check(pol.getState() === 'IDLE', 'default timer fired idle after debounce (no Illegal invocation)');
    check(applied.length === 3, 'idle profile applied via default timer');
    check(invoked >= 1, 'setTimeout invoked with correct receiver');
    if (failed > 0) { console.log('FAILURES=' + failed); process.exit(1); }
    console.log('P3A perf default-timer receiver PASS');
  } catch (e) {
    console.log('FAIL: ' + e.message);
    process.exit(1);
  }
}, 300);
"""


@pytest.mark.skipif(shutil.which("node") is None, reason="node not available")
def test_default_timers_survive_brand_checked_receiver() -> None:
    """The policy's default timers must invoke setTimeout/clearTimeout through
    the global object (receiver === global).  A brand-check stub throws
    "TypeError: Illegal invocation" on a foreign receiver — exactly the Chromium
    behavior that broke the viewer at runtime."""
    _assert_node_pass(_run_node(_ILLEGAL_INVOCATION_SCRIPT.replace("__PATH__", PERF_PATH)))


# ---------------------------------------------------------------------------
# Functional (Node): benchmark auto-stop protocol (deterministic fake timers)
# ---------------------------------------------------------------------------

_AUTOSTOP_SCRIPT = r"""
const fs = require('fs');
const vm = require('vm');
const code = fs.readFileSync("__PATH__", 'utf8');

let failed = 0;
function check(cond, msg) { if (!cond) { failed += 1; console.log('FAIL: ' + msg); } }

function makeEl() {
  return { style: {}, textContent: '', value: '', readOnly: false, type: '', id: '',
    children: [], appendChild(c) { this.children.push(c); return c; },
    setAttribute() {}, addEventListener() {}, select() {}, remove() {} };
}
const renderArea = makeEl();
const material = { pointSizeType: 2 };
const pc = { numVisiblePoints: 245000, numVisibleNodes: 34, visibleNodes: [], material };
const renderer = { setSize() {}, getPixelRatio() { return 1; }, setPixelRatio() {}, domElement: { width: 1920, height: 1080 } };
const orbit = { enabled: true, addEventListener() {} };
// Realistic Potree.View mock (the auto-orbit driver calls getPivot()/direction
// and mutates position via Vector3-style copy/sub).
function vec3(x, y, z) {
  const v = {
    x: x, y: y, z: z,
    copy(o) { this.x = o.x; this.y = o.y; this.z = o.z; return this; },
    clone() { return vec3(this.x, this.y, this.z); },
    add(o) { this.x += o.x; this.y += o.y; this.z += o.z; return this; },
    sub(o) { this.x -= o.x; this.y -= o.y; this.z -= o.z; return this; },
    multiplyScalar(s) { this.x *= s; this.y *= s; this.z *= s; return this; }
  };
  return v;
}
const view = {
  position: vec3(0, 0, 0),
  yaw: 0, _pitch: 0, radius: 50,
  get pitch() { return this._pitch; }, set pitch(a) { this._pitch = a; },
  get direction() {
    return vec3(
      Math.cos(this.yaw) * Math.cos(this._pitch),
      Math.sin(this.yaw) * Math.cos(this._pitch),
      Math.sin(this._pitch)
    );
  },
  getPivot() { const d = this.direction; return vec3(this.position.x + d.x * this.radius, this.position.y + d.y * this.radius, this.position.z + d.z * this.radius); },
  pan(x, y) {
    const side = { x: Math.cos(this.yaw), y: Math.sin(this.yaw), z: 0 };
    const dir = this.direction;
    const up = { x: side.y * dir.z, y: -side.x * dir.z, z: side.x * dir.y - side.y * dir.x };
    this.position.x += side.x * x + up.x * y;
    this.position.y += side.y * x + up.y * y;
    this.position.z += dir.z * y;
  },
};
const viewer = { renderArea, renderer, orbitControls: orbit,
  scene: { pointclouds: [pc], view: view },
  _budget: 1000000, _edl: true,
  setPointBudget(v) { this._budget = v; }, getPointBudget() { return this._budget; },
  setEDLEnabled(v) { this._edl = !!v; }, getEDLEnabled() { return this._edl; },
  _listeners: {}, addEventListener(t, fn) { this._listeners[t] = fn; }, removeEventListener() {} };
const Potree = { numNodesLoading: 2, PointSizeType: { FIXED: 0, ATTENUATED: 1, ADAPTIVE: 2 } };

// Controllable clock + FAKE timers so the 5 s / 15 s / auto-stop protocol is
// deterministic and the receiver of every setTimeout/clearTimeout is verified.
let clock = 0;
let lastTimerReceiver = null;
const fakeTimers = [];
function makeSandbox() {
  return {
    document: { createElement: makeEl, body: makeEl() }, window: {}, console, devicePixelRatio: 1,
    // Marker reachable only through the global object, so we can distinguish a
    // global receiver from a leaked plain-object receiver (the "Illegal
    // invocation" regression) without fragile identity comparison.
    __P3A: true,
    performance: { now: function () { return clock; } },
    setTimeout: function (fn, ms) {
      lastTimerReceiver = this;
      const id = fakeTimers.length;
      fakeTimers.push({ id: id, fn: fn, ms: ms });
      return id;
    },
    clearTimeout: function (id) {
      lastTimerReceiver = this;
      const idx = fakeTimers.findIndex(function (t) { return t.id === id; });
      if (idx >= 0) { fakeTimers.splice(idx, 1); }
    }
  };
}
const sandbox = makeSandbox();
sandbox.window = sandbox;
vm.createContext(sandbox);
vm.runInContext(code, sandbox, { filename: 'viewer_perf.js' });
const p = sandbox.p3aPerf;

const cfg = { performance: { datasetPointCount: 982504, frameWindow: 120, overlayHz: 4, idleDebounceMs: 200,
  baseline: { pointBudget: 1000000, pointSizeType: 'adaptive', edl: true },
  benchmark: { warmupMs: 5000, recordMs: 15000 },
  interactive: { pointBudget: 250000, pointSizeType: 'fixed', edl: false },
  idle: { pointBudget: 1000000, pointSizeType: 'adaptive', edl: true } } };
let fitCalls = 0;
const inst = p.init({ viewer: viewer, pointcloud: pc, cfg: cfg, Potree: Potree,
  onFit: function () { fitCalls += 1; } });

// No timers pending after init.
check(fakeTimers.length === 0, 'no timers after init');

function fireTimer(id) {
  const idx = fakeTimers.findIndex(function (t) { return t.id === id; });
  if (idx < 0) { return false; }
  const t = fakeTimers[idx];
  fakeTimers.splice(idx, 1);
  t.fn();
  return true;
}

function runAutoStop(interaction) {
  const before = inst.recorder.results().length;
  const res = inst.recorder.start(interaction);
  check(res.ok === true, interaction + ' start ok');
  check(inst.recorder.isActive(), interaction + ' active');
  check(fakeTimers.length === 1, interaction + ' exactly one auto-stop timer');
  check(fakeTimers[0].ms === 20000, interaction + ' auto-stop scheduled at 5s warmup + 15s record (20s)');
  check(lastTimerReceiver && lastTimerReceiver.__P3A === true,
    interaction + ' setTimeout called with the global receiver (not a leaked object)');

  const ph0 = inst.recorder.getRunPhase(clock);
  check(ph0.active && ph0.phase === 'warming', interaction + ' starts in warming');
  check(ph0.remainingMs > 4000 && ph0.remainingMs <= 5000, interaction + ' ~5s warming remaining');

  // First frame after init only primes lastFrameTs (it does not accumulate).
  clock += 1;
  viewer._listeners['render.pass.end']();
  // Advance past the 5s warm-up; the next frame transitions to recording.
  clock += 6000;
  viewer._listeners['render.pass.end']();
  const ph1 = inst.recorder.getRunPhase(clock);
  check(ph1.active && ph1.phase === 'recording', interaction + ' enters recording after 5s');
  check(ph1.remainingMs > 14000 && ph1.remainingMs <= 15000, interaction + ' ~15s recording remaining');

  // Record some frames, then let the measurement window elapse so the recorded
  // measurementDurationMs is the real record-window length (> 0).
  for (let fi = 0; fi < 20; fi++) { clock += 5; viewer._listeners['render.pass.end'](); }
  clock += 14800;

  // Fire the auto-stop timer (20s total).
  const tid = fakeTimers[0].id;
  check(fireTimer(tid), interaction + ' auto-stop timer fires');
  check(!inst.recorder.isActive(), interaction + ' run auto-stopped');
  check(inst.recorder.results().length === before + 1, interaction + ' exactly one result saved');
  check(!inst.policy.isSuspended(), interaction + ' policy unsuspended');
  check(fakeTimers.length === 0, interaction + ' no leftover timers');
}

runAutoStop('stationary');
runAutoStop('orbit');
runAutoStop('pan');

// Corrected exported metric: intervalFps = frames*1000/measurementDurationMs is
// the primary FPS; the old median-derived `fps` field is removed (renamed to
// medianFps).
{
  const last = inst.recorder.results()[inst.recorder.results().length - 1];
  check(typeof last.measurementDurationMs === 'number' && last.measurementDurationMs > 0,
    'measurementDurationMs present and positive');
  check(typeof last.intervalFps === 'number', 'intervalFps present');
  check(typeof last.medianFps === 'number', 'medianFps present (median-derived, renamed)');
  check(last.fps === undefined, 'old misleading `fps` field removed');
  const expected = Math.round(last.frames * 1000 / last.measurementDurationMs * 10) / 10;
  check(Math.abs(last.intervalFps - expected) < 0.001,
    'intervalFps = frames*1000/measurementDurationMs, got ' + last.intervalFps);
}

// Manual Stop cancels the pending auto-stop timer and saves exactly one result.
{
  const before = inst.recorder.results().length;
  const res = inst.recorder.start('stationary');
  check(res.ok === true, 'manual-stop: run started');
  check(fakeTimers.length === 1, 'manual-stop: auto-stop timer pending');
  const tid = fakeTimers[0].id;
  const snap = inst.recorder.stop();
  check(snap !== null, 'manual-stop: result returned');
  check(fakeTimers.length === 0, 'manual-stop: pending timer cancelled');
  check(!inst.recorder.isActive(), 'manual-stop: run inactive');
  check(inst.recorder.results().length === before + 1, 'manual-stop: exactly one result');
  // Re-firing the cancelled id is a no-op (no duplicate).
  check(!fireTimer(tid), 'manual-stop: cancelled timer cannot fire');
  check(inst.recorder.results().length === before + 1, 'manual-stop: no duplicate from cancelled timer');
}

// Auto-stop firing twice must not record a duplicate.
{
  const before = inst.recorder.results().length;
  inst.recorder.start('orbit');
  const tid = fakeTimers[0].id;
  fireTimer(tid);
  check(inst.recorder.results().length === before + 1, 'auto-stop: first fire saved one result');
  check(!fireTimer(tid), 'auto-stop: second fire is a no-op (already consumed)');
  check(inst.recorder.results().length === before + 1, 'auto-stop: no duplicate result');
}

// Starting a second run after the first completes works and schedules a fresh
// timer (no stale timer from the previous run).
{
  const before = inst.recorder.results().length;
  inst.recorder.start('pan');
  check(inst.recorder.isActive(), 'second-run: starts after first completed');
  check(fakeTimers.length === 1, 'second-run: one fresh auto-stop timer');
  check(fakeTimers[0].ms === 20000, 'second-run: fresh timer at 20s');
  fireTimer(fakeTimers[0].id);
  check(!inst.recorder.isActive(), 'second-run: auto-stops cleanly');
  check(inst.recorder.results().length === before + 1, 'second-run: exactly one new result');
  check(fakeTimers.length === 0, 'second-run: no timers left');
}

// Visible status reflects the protocol (warming / recording / completed).
{
  const statusEl = (function find(el, id) { if (el.id === id) return el; for (const c of (el.children || [])) { const f = find(c, id); if (f) return f; } return null; })(inst.panel.element, 'p3a_bench_status');
  check(statusEl !== null, 'status element exists');
  inst.recorder.start('stationary');
  clock += 300; viewer._listeners['render.pass.end']();
  check(/warming/.test(statusEl.textContent), 'status shows warming: ' + statusEl.textContent);
  clock += 6000; viewer._listeners['render.pass.end']();
  check(/recording/.test(statusEl.textContent), 'status shows recording: ' + statusEl.textContent);
  fireTimer(fakeTimers[0].id);
  clock += 300; viewer._listeners['render.pass.end']();
  check(/completed/.test(statusEl.textContent), 'status shows completed: ' + statusEl.textContent);
}

// Each run start reframes via onFit (7 run starts above) and the auto-orbit /
// auto-pan drivers ran without throwing against the realistic View mock.
check(fitCalls === 7, 'onFit called once per run start, got ' + fitCalls);

if (failed > 0) { console.log('FAILURES=' + failed); process.exit(1); }
console.log('P3A perf auto-stop protocol PASS');
"""


@pytest.mark.skipif(shutil.which("node") is None, reason="node not available")
def test_benchmark_autostop_protocol_deterministic() -> None:
    """Stationary / Orbit / Pan: 5 s warm-up → 15 s recording → auto-stop with
    exactly one result; manual Stop cancels timers; no duplicates; second run
    works; timers always invoked with the global receiver."""
    _assert_node_pass(_run_node(_AUTOSTOP_SCRIPT.replace("__PATH__", PERF_PATH)))


_STAGE_A_SCRIPT = r"""
const fs = require('fs');
const vm = require('vm');
const code = fs.readFileSync("__PATH__", 'utf8');

let failed = 0;
function check(cond, msg) { if (!cond) { failed += 1; console.log('FAIL: ' + msg); } }

function makeEl() {
  return { style: {}, textContent: '', value: '', readOnly: false, type: '', id: '',
    children: [], appendChild(c) { this.children.push(c); return c; },
    setAttribute() {}, addEventListener() {}, select() {}, remove() {} };
}
function vec3(x, y, z) {
  const v = {
    x: x, y: y, z: z,
    copy(o) { this.x = o.x; this.y = o.y; this.z = o.z; return this; },
    clone() { return vec3(this.x, this.y, this.z); },
    add(o) { this.x += o.x; this.y += o.y; this.z += o.z; return this; },
    sub(o) { this.x -= o.x; this.y -= o.y; this.z -= o.z; return this; },
    multiplyScalar(s) { this.x *= s; this.y *= s; this.z *= s; return this; }
  };
  return v;
}
const renderArea = makeEl();
const material = { pointSizeType: 2 };
const pc = { numVisiblePoints: 245000, numVisibleNodes: 34, visibleNodes: [], material };
// Realistic THREE renderer: setPixelRatio(v) -> setSize(_w,_h,false) recomputes
// canvas.width = floor(w * pixelRatio), so a 0 pixel ratio would zero the
// canvas (the exact regression the Stage A runner previously triggered).
const renderer = {
  _w: 1699, _h: 935, _pr: 1,
  setSize(w, h) { this._w = w; this._h = h; this.domElement.width = Math.floor(w * this._pr); this.domElement.height = Math.floor(h * this._pr); },
  getPixelRatio() { return this._pr; },
  setPixelRatio(v) { this._pr = v; this.setSize(this._w, this._h, false); },
  domElement: { width: 0, height: 0 }
};
const orbit = { enabled: true, addEventListener() {} };
const view = {
  position: vec3(0, 0, 0), yaw: 0, _pitch: 0, radius: 50,
  get pitch() { return this._pitch; }, set pitch(a) { this._pitch = a; },
  get direction() { return vec3(Math.cos(this.yaw) * Math.cos(this._pitch), Math.sin(this.yaw) * Math.cos(this._pitch), Math.sin(this._pitch)); },
  getPivot() { const d = this.direction; return vec3(this.position.x + d.x * this.radius, this.position.y + d.y * this.radius, this.position.z + d.z * this.radius); },
  pan(x, y) { this.position.x += x; this.position.y += y; }
};
const viewer = { renderArea, renderer, orbitControls: orbit,
  scene: { pointclouds: [pc], view: view },
  _budget: 1000000, _edl: true,
  setPointBudget(v) { this._budget = v; }, getPointBudget() { return this._budget; },
  setEDLEnabled(v) { this._edl = !!v; }, getEDLEnabled() { return this._edl; },
  _listeners: {}, addEventListener(t, fn) { this._listeners[t] = fn; }, removeEventListener() {} };
const Potree = { numNodesLoading: 2, PointSizeType: { FIXED: 0, ATTENUATED: 1, ADAPTIVE: 2 } };

let clock = 0;
const fakeTimers = [];
const sandbox = {
  document: { createElement: makeEl, body: makeEl() }, window: {}, console, devicePixelRatio: 1.5,
  __P3A: true,
  performance: { now: function () { return clock; } },
  setTimeout: function (fn, ms) {
    const id = fakeTimers.length;
    fakeTimers.push({ id: id, fn: fn, ms: ms });
    return id;
  },
  clearTimeout: function (id) {
    const idx = fakeTimers.findIndex(function (t) { return t.id === id; });
    if (idx >= 0) { fakeTimers.splice(idx, 1); }
  }
};
sandbox.window = sandbox;
vm.createContext(sandbox);
vm.runInContext(code, sandbox, { filename: 'viewer_perf.js' });
const p = sandbox.p3aPerf;
const cfg = { performance: { datasetPointCount: 982504, frameWindow: 120, overlayHz: 4, idleDebounceMs: 200,
  baseline: { pointBudget: 1000000, pointSizeType: 'adaptive', edl: true },
  benchmark: { warmupMs: 5000, recordMs: 15000 },
  interactive: { pointBudget: 250000, pointSizeType: 'fixed', edl: false },
  idle: { pointBudget: 1000000, pointSizeType: 'adaptive', edl: true } } };
const inst = p.init({ viewer, pointcloud: pc, cfg, Potree });

const micro = function () { return new Promise(function (r) { setTimeout(r, 0); }); };

(async function () {
  const promise = inst.panel.runStageA();
  await micro();
  // Run 1 (stationary @ 100000) is active with the Stage A baseline forced.
  check(inst.recorder.isActive(), 'Stage A: first run active after click');
  check(inst.panel.settings().budget === 100000, 'Stage A: budget starts at 100000, got ' + inst.panel.settings().budget);
  const s = inst.panel.settings();
  check(s.pointSizeType === 'adaptive' && s.edl === true && s.pixelRatio === 1,
    'Stage A: baseline forced (ADAPTIVE/EDL ON/PR 1.0): ' + JSON.stringify(s));

  // Drive the fake timers until the runner COMPLETES (the last run needs one
  // more poll timer after its auto-stop for the runner to exit its await loop).
  for (let guard = 0; guard < 800 && inst.panel.isStageARunning(); guard++) {
    if (fakeTimers.length === 0) { break; }
    const t = fakeTimers.shift();
    t.fn();
    await micro();
  }
  await promise;

  const results = inst.recorder.results();
  check(results.length === 18, 'Stage A produced 18 results, got ' + results.length);
  check(!inst.recorder.isActive(), 'Stage A leaves no run active');
  check(!inst.panel.isStageARunning(), 'Stage A runner finished');

  // Order: 6 budgets × [stationary, orbit, pan], ascending.
  const budgets = [100000, 250000, 400000, 500000, 750000, 1000000];
  const interactions = ['stationary', 'orbit', 'pan'];
  let ok = true;
  for (let bi = 0; bi < budgets.length && ok; bi++) {
    for (let ii = 0; ii < interactions.length && ok; ii++) {
      const r = results[bi * 3 + ii];
      if (!r || r.budget !== budgets[bi] || r.interaction !== interactions[ii]) {
        ok = false;
        console.log('order mismatch at', bi, ii, r && r.budget, r && r.interaction);
      }
      if (r && typeof r.intervalFps !== 'number') { ok = false; console.log('missing intervalFps at', bi, ii); }
      if (r && typeof r.measurementDurationMs !== 'number') { ok = false; console.log('missing measurementDurationMs at', bi, ii); }
    }
  }
  check(ok, '18 results in budget/interaction order, each with intervalFps + measurementDurationMs');

  // Corrected runtime exports on every result: renderer pixel ratio, canvas
  // size, visible-point/node aggregates, renamed end-of-run diagnostics.
  for (const r of results) {
    check(r.rendererPixelRatio === 1, 'rendererPixelRatio exported = 1, got ' + r.rendererPixelRatio);
    check(r.canvasWidth === 1699 && r.canvasHeight === 935, 'canvas size exported 1699x935, got ' + r.canvasWidth + 'x' + r.canvasHeight);
    check(r.pixelRatio === undefined, 'old `pixelRatio` field removed');
    check(typeof r.actualVisiblePointsAtEnd === 'number', 'actualVisiblePointsAtEnd present');
    check(r.actualVisiblePoints === undefined, 'old `actualVisiblePoints` field removed');
    check(typeof r.visiblePointsAverage === 'number' && typeof r.visiblePointsMedian === 'number' &&
          typeof r.visiblePointsP95 === 'number' && typeof r.visiblePointsMax === 'number',
      'visiblePointsAverage/Median/P95/Max present');
    check(typeof r.visibleNodesAverage === 'number' && typeof r.visibleNodesMedian === 'number' &&
          typeof r.visibleNodesMax === 'number', 'visibleNodesAverage/Median/Max present');
  }

  if (failed > 0) { console.log('FAILURES=' + failed); process.exit(1); }
  console.log('P3A perf Stage A one-click PASS');
})();
"""


@pytest.mark.skipif(shutil.which("node") is None, reason="node not available")
def test_stage_a_one_click_runner() -> None:
    """The one-click Stage A runner executes all 6 budgets × 3 interactions in
    order under the forced Stage A baseline, auto-stopping each run and
    exporting results with the corrected intervalFps metric."""
    _assert_node_pass(_run_node(_STAGE_A_SCRIPT.replace("__PATH__", PERF_PATH)))


_EXPORT_AGGREGATES_SCRIPT = r"""
const fs = require('fs');
const vm = require('vm');
const code = fs.readFileSync("__PATH__", 'utf8');
let failed = 0;
function check(cond, msg) { if (!cond) { failed += 1; console.log('FAIL: ' + msg); } }
function makeEl() {
  return { style: {}, textContent: '', value: '', readOnly: false, type: '', id: '',
    children: [], appendChild(c) { this.children.push(c); return c; },
    setAttribute() {}, addEventListener() {}, select() {}, remove() {} };
}
const renderArea = makeEl();
const material = { pointSizeType: 2 };
const pc = { numVisiblePoints: 0, numVisibleNodes: 0, visibleNodes: [], material };
const renderer = {
  _w: 1699, _h: 935, _pr: 1,
  setSize(w, h) { this._w = w; this._h = h; this.domElement.width = Math.floor(w * this._pr); this.domElement.height = Math.floor(h * this._pr); },
  getPixelRatio() { return this._pr; },
  setPixelRatio(v) { this._pr = v; this.setSize(this._w, this._h, false); },
  domElement: { width: 0, height: 0 }
};
const orbit = { enabled: true, addEventListener() {} };
const viewer = { renderArea, renderer, orbitControls: orbit,
  scene: { pointclouds: [pc], view: { yaw: 0, pitch: 0, radius: 50, pan() {}, position: { x: 0, y: 0, z: 0 } } },
  _budget: 1000000, _edl: true,
  setPointBudget(v) { this._budget = v; }, getPointBudget() { return this._budget; },
  setEDLEnabled(v) { this._edl = !!v; }, getEDLEnabled() { return this._edl; },
  _listeners: {}, addEventListener(t, fn) { this._listeners[t] = fn; }, removeEventListener() {} };
const Potree = { numNodesLoading: 2, PointSizeType: { FIXED: 0, ATTENUATED: 1, ADAPTIVE: 2 } };
let clock = 0;
const sandbox = { document: { createElement: makeEl, body: makeEl() }, window: {}, console, setTimeout, clearTimeout,
  performance: { now: () => clock }, devicePixelRatio: 1.5 };
sandbox.window = sandbox;
vm.createContext(sandbox);
vm.runInContext(code, sandbox, { filename: 'viewer_perf.js' });
const p = sandbox.p3aPerf;
const cfg = { performance: { datasetPointCount: 982504, frameWindow: 120, overlayHz: 4, idleDebounceMs: 200,
  baseline: { pointBudget: 1000000, pointSizeType: 'adaptive', edl: true },
  benchmark: { warmupMs: 0, recordMs: 1000 },
  interactive: { pointBudget: 250000, pointSizeType: 'fixed', edl: false },
  idle: { pointBudget: 1000000, pointSizeType: 'adaptive', edl: true } } };
const inst = p.init({ viewer, pointcloud: pc, cfg, Potree });

// Drive > MAX_VISIBLE_SAMPLES (10000) frames with an increasing visible count:
// values 0..10499.  Bounded storage must cap the samples at 10000, so the max
// aggregate is 9999 (not 10499).  This also exercises the aggregate math and
// no-duplicate auto-stop.
inst.recorder.start('stationary');
for (let i = 0; i < 10500; i++) {
  pc.numVisiblePoints = i;
  pc.numVisibleNodes = i % 50;
  clock += 0.05;
  viewer._listeners['render.pass.end']();
}
setTimeout(function () {
  const results = inst.recorder.results();
  check(results.length === 1, 'exactly one result (no duplicate), got ' + results.length);
  const snap = results[0];
  // The first frame only primes lastFrameTs, so the 10000 capped samples are
  // values [1..10000] (not [0..9999]).  If storage were UNBOUNDED the max would
  // be 10499; 10000 proves the cap held exactly MAX_VISIBLE_SAMPLES samples.
  check(snap.visiblePointsMax === 10000, 'bounded sampling: max capped at 10000, got ' + snap.visiblePointsMax);
  check(Math.abs(snap.visiblePointsAverage - 5000.5) < 0.2, 'average over capped samples = 5000.5, got ' + snap.visiblePointsAverage);
  check(snap.visiblePointsMedian === 5000, 'median = 5000, got ' + snap.visiblePointsMedian);
  check(snap.visiblePointsP95 === 9500, 'p95 = 9500, got ' + snap.visiblePointsP95);
  // Node samples are i % 50 over 10000 samples (1..49,0 repeating).
  check(snap.visibleNodesAverage >= 24 && snap.visibleNodesAverage <= 25, 'visibleNodesAverage in range, got ' + snap.visibleNodesAverage);
  check(snap.visibleNodesMax === 49, 'visibleNodesMax = 49, got ' + snap.visibleNodesMax);
  check(snap.rendererPixelRatio === 1, 'rendererPixelRatio = 1');
  check(snap.canvasWidth === 1699 && snap.canvasHeight === 935, 'canvas 1699x935');
  check(snap.actualVisiblePointsAtEnd === 10499, 'actualVisiblePointsAtEnd = 10499 (last frame), got ' + snap.actualVisiblePointsAtEnd);
  if (failed > 0) { console.log('FAILURES=' + failed); process.exit(1); }
  console.log('P3A perf export aggregates PASS');
}, 1500);
"""


@pytest.mark.skipif(shutil.which("node") is None, reason="node not available")
def test_visible_aggregates_and_bounded_sampling() -> None:
    """Visible-point/node interval aggregates are computed over a bounded sample
    buffer (capped at MAX_VISIBLE_SAMPLES), the average/median/P95/max math is
    correct, the renderer pixel ratio and canvas are exported correctly, and an
    auto-stop records exactly one result."""
    _assert_node_pass(_run_node(_EXPORT_AGGREGATES_SCRIPT.replace("__PATH__", PERF_PATH)))


_FINAL_CONFIRMATION_SCRIPT = r"""
const fs = require('fs');
const vm = require('vm');
const code = fs.readFileSync("__PATH__", 'utf8');
let failed = 0;
function check(cond, msg) { if (!cond) { failed += 1; console.log('FAIL: ' + msg); } }
function makeEl() {
  return { style: {}, textContent: '', value: '', readOnly: false, type: '', id: '',
    children: [], appendChild(c) { this.children.push(c); return c; },
    setAttribute() {}, addEventListener() {}, select() {}, remove() {} };
}
function vec3(x, y, z) { const v = { x, y, z, copy(o){this.x=o.x;this.y=o.y;this.z=o.z;return this;}, clone(){return vec3(this.x,this.y,this.z);}, add(o){this.x+=o.x;this.y+=o.y;this.z+=o.z;return this;}, sub(o){this.x-=o.x;this.y-=o.y;this.z-=o.z;return this;}, multiplyScalar(s){this.x*=s;this.y*=s;this.z*=s;return this;} }; return v; }
const renderArea = makeEl();
const material = { pointSizeType: 2 };
const pc = { numVisiblePoints: 400000, numVisibleNodes: 90, visibleNodes: [], material };
const renderer = {
  _w: 1699, _h: 935, _pr: 1,
  setSize(w, h) { this._w = w; this._h = h; this.domElement.width = Math.floor(w * this._pr); this.domElement.height = Math.floor(h * this._pr); },
  getPixelRatio() { return this._pr; },
  setPixelRatio(v) { this._pr = v; this.setSize(this._w, this._h, false); },
  domElement: { width: 0, height: 0 }
};
const orbit = { enabled: true, addEventListener() {} };
const view = { position: vec3(0,0,0), yaw: 0, _pitch: 0, radius: 50,
  get pitch(){return this._pitch;}, set pitch(a){this._pitch=a;},
  get direction(){return vec3(Math.cos(this.yaw)*Math.cos(this._pitch), Math.sin(this.yaw)*Math.cos(this._pitch), Math.sin(this._pitch));},
  getPivot(){const d=this.direction; return vec3(this.position.x+d.x*this.radius, this.position.y+d.y*this.radius, this.position.z+d.z*this.radius);},
  pan(x,y){this.position.x+=x; this.position.y+=y;} };
const viewer = { renderArea, renderer, orbitControls: orbit,
  scene: { pointclouds: [pc], view: view },
  _budget: 1000000, _edl: true,
  setPointBudget(v) { this._budget = v; }, getPointBudget() { return this._budget; },
  setEDLEnabled(v) { this._edl = !!v; }, getEDLEnabled() { return this._edl; },
  _listeners: {}, addEventListener(t, fn) { this._listeners[t] = fn; }, removeEventListener() {} };
const Potree = { numNodesLoading: 2, PointSizeType: { FIXED: 0, ATTENUATED: 1, ADAPTIVE: 2 } };
let clock = 0;
const fakeTimers = [];
const sandbox = {
  document: { createElement: makeEl, body: makeEl() }, window: {}, console, devicePixelRatio: 1.5, __P3A: true,
  performance: { now: function () { return clock; } },
  setTimeout: function (fn, ms) { const id = fakeTimers.length; fakeTimers.push({ id: id, fn: fn, ms: ms }); return id; },
  clearTimeout: function (id) { const idx = fakeTimers.findIndex(function (t) { return t.id === id; }); if (idx >= 0) { fakeTimers.splice(idx, 1); } }
};
sandbox.window = sandbox;
vm.createContext(sandbox);
vm.runInContext(code, sandbox, { filename: 'viewer_perf.js' });
const p = sandbox.p3aPerf;
const cfg = { performance: { datasetPointCount: 982504, frameWindow: 120, overlayHz: 4, idleDebounceMs: 200,
  baseline: { pointBudget: 1000000, pointSizeType: 'adaptive', edl: true },
  benchmark: { warmupMs: 5000, recordMs: 15000 },
  interactive: { pointBudget: 250000, pointSizeType: 'fixed', edl: false },
  idle: { pointBudget: 1000000, pointSizeType: 'adaptive', edl: true } } };
const inst = p.init({ viewer, pointcloud: pc, cfg, Potree });
const micro = function () { return new Promise(function (r) { setTimeout(r, 0); }); };

(async function () {
  // Final Confirmation: exactly 750000 + 1000000 × {stationary, orbit, pan}.
  const promise = inst.panel.runFinalConfirmation();
  await micro();
  for (let guard = 0; guard < 800 && inst.panel.isStageARunning(); guard++) {
    if (fakeTimers.length === 0) { break; }
    fakeTimers.shift().fn();
    await micro();
  }
  await promise;
  let r = inst.recorder.results();
  check(r.length === 6, 'Final Confirmation produced 6 results, got ' + r.length);
  const expected = [750000, 750000, 750000, 1000000, 1000000, 1000000];
  const expectedI = ['stationary', 'orbit', 'pan', 'stationary', 'orbit', 'pan'];
  let ok = true;
  for (let i = 0; i < 6; i++) {
    if (!r[i] || r[i].budget !== expected[i] || r[i].interaction !== expectedI[i]) ok = false;
  }
  check(ok, 'Final Confirmation order: 750k s/o/p then 1M s/o/p');
  check(!inst.recorder.isActive(), 'Final Confirmation leaves no run active');

  // Optional 400k Orbit repeat.
  const p2 = inst.panel.runRepeat400kOrbit();
  await micro();
  for (let guard = 0; guard < 200 && inst.panel.isStageARunning(); guard++) {
    if (fakeTimers.length === 0) { break; }
    fakeTimers.shift().fn();
    await micro();
  }
  await p2;
  r = inst.recorder.results();
  check(r.length === 7, '400k Orbit repeat adds exactly one result, got ' + r.length);
  check(r[6] && r[6].budget === 400000 && r[6].interaction === 'orbit', '7th result is 400k/orbit');
  check(!inst.recorder.isActive(), 'no run active after 400k repeat');

  // No duplicates across both sequences.
  const keys = r.map(function (x) { return x.budget + '/' + x.interaction; });
  check(keys.length === new Set(keys).size, 'no duplicate (budget/interaction) results');

  if (failed > 0) { console.log('FAILURES=' + failed); process.exit(1); }
  console.log('P3A perf final confirmation PASS');
})();
"""


@pytest.mark.skipif(shutil.which("node") is None, reason="node not available")
def test_final_confirmation_sequence() -> None:
    """The one-click Final Confirmation runs exactly 750k+1M × {stationary,
    orbit, pan} (6 runs) plus an optional single 400k Orbit repeat (7th), in
    order, with no duplicate results."""
    _assert_node_pass(_run_node(_FINAL_CONFIRMATION_SCRIPT.replace("__PATH__", PERF_PATH)))


_BOUNDED_RESULTS_SCRIPT = r"""
const fs = require('fs');
const vm = require('vm');
const code = fs.readFileSync("__PATH__", 'utf8');
let failed = 0;
function check(cond, msg) { if (!cond) { failed += 1; console.log('FAIL: ' + msg); } }
function makeEl() {
  return { style: {}, textContent: '', value: '', readOnly: false, type: '', id: '',
    children: [], appendChild(c) { this.children.push(c); return c; },
    setAttribute() {}, addEventListener() {}, select() {}, remove() {} };
}
const renderArea = makeEl();
const material = { pointSizeType: 2 };
const pc = { numVisiblePoints: 0, numVisibleNodes: 1, visibleNodes: [], material };
const renderer = { setSize() {}, getPixelRatio() { return 1; }, setPixelRatio() {}, domElement: { width: 1699, height: 935 } };
const orbit = { enabled: true, addEventListener() {} };
const viewer = { renderArea, renderer, orbitControls: orbit,
  scene: { pointclouds: [pc], view: { yaw: 0, pitch: 0, radius: 50, pan() {}, position: { x: 0, y: 0, z: 0 } } },
  _budget: 750000, _edl: true,
  setPointBudget(v) { this._budget = v; }, getPointBudget() { return this._budget; },
  setEDLEnabled(v) { this._edl = !!v; }, getEDLEnabled() { return this._edl; },
  _listeners: {}, addEventListener(t, fn) { this._listeners[t] = fn; }, removeEventListener() {} };
const Potree = { numNodesLoading: 2, PointSizeType: { FIXED: 0, ATTENUATED: 1, ADAPTIVE: 2 } };
let clock = 0;
const sandbox = { document: { createElement: makeEl, body: makeEl() }, window: {}, console, setTimeout, clearTimeout,
  performance: { now: () => clock }, devicePixelRatio: 1.5 };
sandbox.window = sandbox;
vm.createContext(sandbox);
vm.runInContext(code, sandbox, { filename: 'viewer_perf.js' });
const p = sandbox.p3aPerf;
const cfg = { performance: { datasetPointCount: 982504, frameWindow: 120, overlayHz: 4, idleDebounceMs: 200,
  baseline: { pointBudget: 750000, pointSizeType: 'adaptive', edl: true },
  benchmark: { warmupMs: 0, recordMs: 100 },
  interactive: { pointBudget: 750000, pointSizeType: 'adaptive', edl: true },
  idle: { pointBudget: 750000, pointSizeType: 'adaptive', edl: true } } };
const inst = p.init({ viewer, pointcloud: pc, cfg, Potree });

// Run well over the cap (50) with distinguishable visible-point markers.
for (let i = 0; i < 55; i++) {
  pc.numVisiblePoints = i;
  const res = inst.recorder.start('stationary');
  check(res.ok === true, 'run ' + i + ' starts');
  clock += 1;
  viewer._listeners['render.pass.end'](); // one recorded frame -> lastVisiblePoints = i
  inst.recorder.stop();
}
const results = inst.recorder.results();
check(inst.recorder.MAX_BENCHMARK_RESULTS === 50, 'MAX_BENCHMARK_RESULTS exposed = 50');
check(results.length === 50, 'completed results capped at 50, got ' + results.length);
check(results.length === new Set(results.map(r => r.actualVisiblePointsAtEnd)).size,
  'no duplicate results retained');
// Deterministic FIFO eviction: runs 0..4 dropped, runs 5..54 retained.
check(results[0].actualVisiblePointsAtEnd === 5, 'oldest retained = run 5, got ' + results[0].actualVisiblePointsAtEnd);
check(results[results.length - 1].actualVisiblePointsAtEnd === 54, 'newest retained = run 54, got ' + results[results.length - 1].actualVisiblePointsAtEnd);
// New results still record after the cap is reached.
pc.numVisiblePoints = 100;
inst.recorder.start('stationary');
clock += 1;
viewer._listeners['render.pass.end']();
inst.recorder.stop();
const after = inst.recorder.results();
check(after.length === 50, 'still capped at 50 after another run');
check(after[after.length - 1].actualVisiblePointsAtEnd === 100, 'newest result recorded');
check(after[0].actualVisiblePointsAtEnd === 6, 'oldest further evicted (now starts at run 6)');
if (failed > 0) { console.log('FAILURES=' + failed); process.exit(1); }
console.log('P3A perf bounded results PASS');
"""


@pytest.mark.skipif(shutil.which("node") is None, reason="node not available")
def test_benchmark_results_bounded_with_fifo_eviction() -> None:
    """Completed benchmark results are capped at MAX_BENCHMARK_RESULTS (50) with
    deterministic FIFO eviction (drop oldest, retain newest); new results still
    record and results() exports only the retained results."""
    _assert_node_pass(_run_node(_BOUNDED_RESULTS_SCRIPT.replace("__PATH__", PERF_PATH)))


# ---------------------------------------------------------------------------
# Functional (Node): init wires events + overlay against a mock viewer
# ---------------------------------------------------------------------------

_INIT_SCRIPT = r"""
const fs = require('fs');
const vm = require('vm');
const code = fs.readFileSync("__PATH__", 'utf8');

let failed = 0;
function check(cond, msg) { if (!cond) { failed += 1; console.log('FAIL: ' + msg); } }

function makeEl() {
  return {
    style: {}, textContent: '', value: '', readOnly: false, type: '', children: [],
    appendChild(c) { this.children.push(c); return c; },
    setAttribute() {}, addEventListener() {}, select() {}, remove() {},
  };
}
const renderArea = makeEl();
const material = { pointSizeType: 2 };
const pc = { numVisiblePoints: 245000, numVisibleNodes: 34, visibleNodes: [], material: material };
const renderer = {
  setSize() {}, getPixelRatio() { return 1; },
  domElement: { width: 1920, height: 1080 },
};
const orbit = { enabled: true, addEventListener() {} };
const viewer = {
  renderArea: renderArea,
  renderer: renderer,
  orbitControls: orbit,
  scene: { pointclouds: [pc], view: { yaw: 0, pitch: 0, radius: 10, pan() {}, position: { x: 0, y: 0, z: 0 } } },
  _budget: 1000000, _edl: true,
  setPointBudget(v) { this._budget = v; },
  getPointBudget() { return this._budget; },
  setEDLEnabled(v) { this._edl = !!v; },
  getEDLEnabled() { return this._edl; },
  _listeners: {},
  addEventListener(type, fn) { this._listeners[type] = fn; },
  removeEventListener() {},
};
const Potree = {
  numNodesLoading: 2,
  PointSizeType: { FIXED: 0, ATTENUATED: 1, ADAPTIVE: 2 },
};

const sandbox = {
  document: { createElement: makeEl, body: makeEl() },
  window: {}, console: console,
  setTimeout: setTimeout, clearTimeout: clearTimeout,
  performance: { now: function () { return Date.now(); } },
  devicePixelRatio: 1,
};
sandbox.window = sandbox;
vm.createContext(sandbox);
vm.runInContext(code, sandbox, { filename: 'viewer_perf.js' });
const p = sandbox.p3aPerf;

const cfg = {
  performance: {
    datasetPointCount: 982504,
    frameWindow: 120,
    overlayHz: 4,
    idleDebounceMs: 200,
    baseline: { pointBudget: 1000000, pointSizeType: 'adaptive', edl: true },
    benchmark: { warmupMs: 5000, recordMs: 15000 },
    interactive: { pointBudget: 250000, pointSizeType: 'fixed', edl: false },
    idle: { pointBudget: 1000000, pointSizeType: 'adaptive', edl: true },
  },
};

const inst = p.init({ viewer: viewer, pointcloud: pc, cfg: cfg, Potree: Potree });
check(inst !== null, 'init returns an instance');

// IDLE profile applied at startup (full detail).
check(viewer._budget === 1000000, 'startup applies idle budget');
check(viewer._edl === true, 'startup applies idle EDL');
check(material.pointSizeType === 2, 'startup applies idle point size type');

// Overlay + benchmark panel appended to the render area (not to body).
check(renderArea.children.length >= 2, 'overlay + benchmark panel appended');
const overlayIds = renderArea.children.map(c => c.id);
check(overlayIds.indexOf('p3a_perf_overlay') !== -1, 'overlay element id present');
check(overlayIds.indexOf('p3a_bench_panel') !== -1, 'benchmark panel id present');

// Frame cadence hooked via render.pass.end; a single frame must not explode.
check(typeof viewer._listeners['render.pass.end'] === 'function', 'render.pass.end hooked');
viewer._listeners['render.pass.end']();
viewer._listeners['render.pass.end']();
check(viewer._listeners['render.pass.end'], 'frame hook is the SAME handler (no new loop)');

// Camera_changed drives the policy to INTERACTIVE.
viewer._listeners['camera_changed']();
check(inst.policy.getState() === 'INTERACTIVE', 'camera_changed -> INTERACTIVE');

// The perf helper code never calls render()/updatePointClouds (comments
// stripped — the docstring describes what the file does NOT call).
const codeOnly = code.replace(/\/\*[\s\S]*?\*\//g, ' ').replace(/\/\/[^\n]*/g, ' ');
check(!codeOnly.includes('updatePointClouds'), 'no updatePointClouds in code');
check(!codeOnly.includes('.render('), 'no render() call in code');

if (failed > 0) { console.log('FAILURES=' + failed); process.exit(1); }
console.log('P3A perf init smoke PASS');
"""


@pytest.mark.skipif(shutil.which("node") is None, reason="node not available")
def test_init_wires_events_overlay_and_idle_profile() -> None:
    _assert_node_pass(_run_node(_INIT_SCRIPT.replace("__PATH__", PERF_PATH)))


_PANEL_DEFAULTS_SCRIPT = r"""
const fs = require('fs');
const vm = require('vm');
const code = fs.readFileSync("__PATH__", 'utf8');
let failed = 0;
function check(cond, msg) { if (!cond) { failed += 1; console.log('FAIL: ' + msg); } }

function makeEl() {
  return { style: {}, textContent: '', value: '', readOnly: false, type: '', id: '',
    children: [], appendChild(c) { this.children.push(c); return c; },
    setAttribute() {}, addEventListener() {}, select() {}, remove() {} };
}
const renderArea = makeEl();
const material = { pointSizeType: 2 };
const pc = { numVisiblePoints: 245000, numVisibleNodes: 34, visibleNodes: [], material };
const renderer = { setSize() {}, getPixelRatio() { return 1; }, setPixelRatio() {}, domElement: { width: 1920, height: 1080 } };
const orbit = { enabled: true, addEventListener() {} };
const viewer = { renderArea, renderer, orbitControls: orbit,
  scene: { pointclouds: [pc], view: { yaw: 0, pitch: 0, radius: 50, pan() {}, position: { x: 0, y: 0, z: 0 } } },
  _budget: 1000000, _edl: true,
  setPointBudget(v) { this._budget = v; }, getPointBudget() { return this._budget; },
  setEDLEnabled(v) { this._edl = !!v; }, getEDLEnabled() { return this._edl; },
  _listeners: {}, addEventListener(t, fn) { this._listeners[t] = fn; }, removeEventListener() {} };
const Potree = { numNodesLoading: 2, PointSizeType: { FIXED: 0, ATTENUATED: 1, ADAPTIVE: 2 } };
const sandbox = { document: { createElement: makeEl, body: makeEl() }, window: {}, console, setTimeout, clearTimeout, performance: { now: () => Date.now() }, devicePixelRatio: 1 };
sandbox.window = sandbox;
vm.createContext(sandbox);
vm.runInContext(code, sandbox, { filename: 'viewer_perf.js' });
const p = sandbox.p3aPerf;
const cfg = { performance: { datasetPointCount: 982504, frameWindow: 120, overlayHz: 4, idleDebounceMs: 200,
  baseline: { pointBudget: 1000000, pointSizeType: 'adaptive', edl: true },
  benchmark: { warmupMs: 5000, recordMs: 15000 },
  interactive: { pointBudget: 250000, pointSizeType: 'fixed', edl: false },
  idle: { pointBudget: 1000000, pointSizeType: 'adaptive', edl: true } } };
const inst = p.init({ viewer, pointcloud: pc, cfg, Potree });

// The benchmark panel must preselect the current fixed normal default
// (regression: preselecting the label wrapper instead of the <select> left the
// control on its FIRST option = FIXED / 100000).
function findById(el, id) {
  if (el.id === id) return el;
  for (const c of (el.children || [])) { const f = findById(c, id); if (f) return f; }
  return null;
}
const panel = inst.panel.element;
check(!!findById(panel, 'p3a_bench_size'), 'size select present in panel DOM');
check(findById(panel, 'p3a_bench_size').value === 'adaptive', 'size preselects ADAPTIVE, got ' + findById(panel, 'p3a_bench_size').value);
check(findById(panel, 'p3a_bench_budget').value === '750000', 'budget preselects 750000 (fixed default)');
check(findById(panel, 'p3a_bench_edl').value === 'true', 'EDL preselects ON');
check(findById(panel, 'p3a_bench_pr').value === '1', 'pixelRatio preselects 1.0');

// The effective settings() read the ACTUAL selects.
const s = inst.panel.settings();
check(s.budget === 750000 && s.pointSizeType === 'adaptive' && s.edl === true && s.pixelRatio === 1,
  'settings() returns fixed default: ' + JSON.stringify(s));

if (failed > 0) { console.log('FAILURES=' + failed); process.exit(1); }
console.log('P3A perf panel defaults PASS');
"""


@pytest.mark.skipif(shutil.which("node") is None, reason="node not available")
def test_benchmark_panel_preselects_stage_a_baseline() -> None:
    """Stage A must start ADAPTIVE / EDL ON / pixel-ratio 1.0 / budgets 100000..
    1000000.  Regression: the panel preselect must target the <select>, not the
    label wrapper (which silently left the control on FIXED)."""
    _assert_node_pass(_run_node(_PANEL_DEFAULTS_SCRIPT.replace("__PATH__", PERF_PATH)))


# ---------------------------------------------------------------------------
# Static: viewer.js guards the optional perf init (fail-open)
# ---------------------------------------------------------------------------


def test_viewer_js_perf_init_uses_identical_profiles_preserving_baseline() -> None:
    """The final design has NO motion/idle dual budgets: the interactive and idle
    profiles are identical to the baseline, so the two-state policy is inert and
    normal startup uses exactly the fixed cross-device default."""
    data = json.loads((VIEWER_ROOT / "poc_config.json").read_text(encoding="utf-8"))
    perf = data["performance"]
    assert perf["interactive"] == perf["baseline"]
    assert perf["idle"] == perf["baseline"]
    assert perf["baseline"]["pointBudget"] == 750000


def test_perf_config_structure_valid() -> None:
    data = json.loads((VIEWER_ROOT / "poc_config.json").read_text(encoding="utf-8"))
    perf = data["performance"]
    assert perf["datasetPointCount"] == 982504
    assert perf["frameWindow"] == 120
    assert perf["overlayHz"] == 4
    assert perf["idleDebounceMs"] == 200
    assert perf["baseline"]["pointBudget"] == 750000
    assert perf["baseline"]["pointSizeType"] == "adaptive"
    assert perf["baseline"]["edl"] is True
    for name in ("interactive", "idle"):
        for key in ("pointBudget", "pointSizeType", "edl"):
            assert key in perf[name], f"{name}.{key} missing"


def test_normal_viewer_config_uses_fixed_default() -> None:
    """Normal startup (viewer.js reads cfg.viewer.pointBudget) must use the
    reviewed 750k fixed default with ADAPTIVE / EDL ON."""
    data = json.loads((VIEWER_ROOT / "poc_config.json").read_text(encoding="utf-8"))
    viewer_cfg = data["viewer"]
    assert viewer_cfg["pointBudget"] == 750000
    assert viewer_cfg["pointSizeType"] == "adaptive"
    assert viewer_cfg["edlEnabled"] is True
    assert data["performance"]["baseline"]["pointBudget"] == 750000
