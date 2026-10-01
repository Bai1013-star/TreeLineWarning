/**
 * TreeLineWarning P3A — measured performance diagnostics + fixed two-state
 * rendering policy.
 *
 * Two distinct responsibilities, kept separate so the pure/decision logic is
 * testable in Node while the DOM/viewer glue only ever runs in the browser:
 *
 *  1. MEASUREMENT: observe the EXISTING Potree render cadence (via the
 *     per-frame ``render.pass.end`` event Potree 1.8.2 dispatches at the end of
 *     every render pass), keep a bounded rolling window of frame times, compute
 *     average / median / P95 / FPS, read the ACTUAL visible-point and
 *     visible-node counts each frame, count ``renderer.setSize`` calls per frame
 *     (no vendor patch — the renderer method is wrapped at runtime and
 *     delegated), and drive a small debug overlay that updates no faster than
 *     ~4 Hz.
 *
 *  2. FIXED TWO-STATE POLICY: when the camera is moving use ONE fixed
 *     INTERACTIVE profile; when the camera has been idle for a debounce window
 *     use ONE fixed IDLE profile.  No GPU detection, no FPS feedback, no
 *     autotuning, no hardware tiers — every device uses the same rules.
 *
 * CONTRACT (mirrors the P3A-004/P3A-010 startup-guard style):
 *  - This file NEVER creates a second render loop: no ``requestAnimationFrame``,
 *    no ``setAnimationLoop``, no ``viewer.render()``, no
 *    ``Potree.updatePointClouds(...)``.  It only subscribes to events Potree
 *    already emits and reads values Potree already publishes.
 *  - The module is a pure IIFE attached to the global object (window in the
 *    browser, globalThis in Node) so the Node test harness can ``eval`` it
 *    directly, exactly like ``viewer_util.js``.
 */
(function (global) {
	"use strict";

	// -----------------------------------------------------------------------
	// Constants / defaults
	// -----------------------------------------------------------------------

	/** Default rolling frame-history length (frames). */
	var FRAME_WINDOW = 120;

	/** Default overlay refresh interval (ms) → ≤4 Hz. */
	var OVERLAY_INTERVAL_MS = 250;

	/** Default idle debounce before INTERACTIVE → IDLE (ms). */
	var IDLE_DEBOUNCE_MS = 200;

	/** Benchmark warm-up / record durations (ms). */
	var BENCHMARK_DEFAULT_WARMUP_MS = 5000;
	var BENCHMARK_DEFAULT_RECORD_MS = 15000;

	/** The single fixed budget list the benchmark panel offers. */
	var BENCHMARK_BUDGETS = [100000, 250000, 400000, 500000, 750000, 1000000];

	/** Allowed benchmark knob values (single source of truth for validation). */
	var ALLOWED = {
		budgets: BENCHMARK_BUDGETS,
		pointSizeTypes: ["fixed", "attenuated", "adaptive"],
		edl: [true, false],
		pixelRatios: [1.0, 0.75],
		interactions: ["stationary", "orbit", "pan"]
	};

	var POINT_SIZE_TYPE_NAMES = { 0: "FIXED", 1: "ATTENUATED", 2: "ADAPTIVE" };

	// -----------------------------------------------------------------------
	// Pure helpers (no DOM) — Node-testable
	// -----------------------------------------------------------------------

	function nowMillis() {
		if (typeof performance !== "undefined" && typeof performance.now === "function") {
			return performance.now();
		}
		return Date.now();
	}

	function isFiniteNumber(v) {
		return typeof v === "number" && isFinite(v);
	}

	function clampInt(v, lo, hi) {
		v = Math.floor(v);
		if (v < lo) return lo;
		if (v > hi) return hi;
		return v;
	}

	/**
	 * Bounded rolling history of the most recent ``maxFrames`` values.
	 * When full, the oldest value is dropped (FIFO).  Cheap: a plain array.
	 */
	function createBoundedHistory(maxFrames) {
		var max = clampInt(maxFrames, 1, 1000000);
		var items = [];

		return {
			push: function (value) {
				if (!isFiniteNumber(value)) {
					return;
				}
				if (items.length >= max) {
					items.shift();
				}
				items.push(value);
			},
			toArray: function () {
				return items.slice();
			},
			size: function () {
				return items.length;
			},
			clear: function () {
				items = [];
			}
		};
	}

	/**
	 * Summary statistics over an array of frame times (ms).
	 *
	 * Conventions (documented so the unit tests match the implementation):
	 *  - median = lower median: ``sorted[floor((n - 1) / 2)]``.
	 *  - P95 = ``sorted[min(n - 1, floor(0.95 * (n - 1)))]``.
	 *  - FPS = 1000 / median (robust frame-rate proxy).
	 */
	function computeStats(values) {
		var n = values.length;
		if (n === 0) {
			return { count: 0, average: 0, median: 0, p95: 0, fps: 0 };
		}
		var sum = 0;
		var i;
		for (i = 0; i < n; i++) {
			sum += values[i];
		}
		var sorted = values.slice().sort(function (a, b) {
			return a - b;
		});
		var median = sorted[Math.floor((n - 1) / 2)];
		var p95 = sorted[Math.min(n - 1, Math.floor(0.95 * (n - 1)))];
		return {
			count: n,
			average: sum / n,
			median: median,
			p95: p95,
			fps: median > 0 ? 1000 / median : 0
		};
	}

	/**
	 * Aggregate stats (average / median / P95 / max) over a run's lightweight
	 * visible-point / visible-node samples.  Same median/P95 conventions as
	 * ``computeStats``; ``max`` is the largest sampled value.
	 */
	function aggregateSamples(values) {
		var n = values.length;
		if (n === 0) {
			return { average: 0, median: 0, p95: 0, max: 0 };
		}
		var sum = 0;
		var i;
		for (i = 0; i < n; i++) {
			sum += values[i];
		}
		var sorted = values.slice().sort(function (a, b) {
			return a - b;
		});
		return {
			average: sum / n,
			median: sorted[Math.floor((n - 1) / 2)],
			p95: sorted[Math.min(n - 1, Math.floor(0.95 * (n - 1)))],
			max: sorted[n - 1]
		};
	}

	/**
	 * Throttle: ``shouldUpdate(now)`` returns true at most once per
	 * ``minIntervalMs``, so the DOM overlay is not rewritten every frame.
	 */
	function createThrottle(minIntervalMs) {
		var interval = minIntervalMs > 0 ? minIntervalMs : OVERLAY_INTERVAL_MS;
		// Start far in the past so the very first shouldUpdate() emits.
		var lastEmit = -Infinity;
		return {
			shouldUpdate: function (now) {
				if (now - lastEmit >= interval) {
					lastEmit = now;
					return true;
				}
				return false;
			},
			reset: function () {
				lastEmit = -Infinity;
			}
		};
	}

	/** Validate a benchmark knob against the ALLOWED value list. */
	function isValidBenchmarkValue(kind, value) {
		var allowed = ALLOWED[kind];
		if (!allowed || !Array.isArray(allowed)) {
			return false;
		}
		return allowed.indexOf(value) !== -1;
	}

	function pointSizeTypeName(value) {
		return POINT_SIZE_TYPE_NAMES[value] || String(value);
	}

	function addCommas(n) {
		var s = String(Math.round(n));
		return s.replace(/\B(?=(\d{3})+(?!\d))/g, ",");
	}

	function fmtNum(v, digits) {
		if (!isFiniteNumber(v)) return "?";
		return v.toFixed(digits === undefined ? 1 : digits);
	}

	/**
	 * FIXED TWO-STATE POLICY state machine.
	 *
	 * ``config = {
	 *   interactive: profile,      // {pointBudget, pointSizeType, edl}
	 *   idle:         profile,
	 *   debounceMs:   number,
	 *   apply:        function(profile, state) -> void,
	 *   timers:       {setTimeout, clearTimeout}   // injectable for tests
	 * }``
	 *
	 * States: "INTERACTIVE" (camera moving) and "IDLE" (camera still).  The
	 * state transitions are driven ONLY by Potree control/camera events; there
	 * is no FPS feedback and no GPU/hardware input.
	 *
	 * ``setSuspended(true)`` freezes profile application (used while the
	 * benchmark recorder drives explicit settings); state tracking continues so
	 * the overlay still shows MOVING/IDLE.  Releasing suspension re-applies the
	 * profile matching the current state.
	 */
	function createPerfPolicy(config) {
		var interactive = config.interactive;
		var idle = config.idle;
		var debounceMs = isFiniteNumber(config.debounceMs)
			? clampInt(config.debounceMs, 0, 60000)
			: IDLE_DEBOUNCE_MS;
		var apply = typeof config.apply === "function" ? config.apply : function () {};
		// Receiver-safe default timers.  Storing ``window.setTimeout`` /
		// ``window.clearTimeout`` detached into a plain object and invoking them
		// via that object would call the brand-checked Window members with
		// ``this`` = plain object, which throws "TypeError: Illegal invocation"
		// in Chromium (Edge).  Wrapping so the call always resolves through the
		// global object keeps ``this`` === the Window / globalThis.
		var timers = config.timers || {
			setTimeout: function (fn, ms) {
				return global.setTimeout(fn, ms);
			},
			clearTimeout: function (id) {
				return global.clearTimeout(id);
			}
		};

		var state = "IDLE";
		var idleTimer = null;
		var applied = null;
		var suspended = false;

		function clearIdleTimer() {
			if (idleTimer !== null) {
				timers.clearTimeout(idleTimer);
				idleTimer = null;
			}
		}

		function applyProfile(profile, nextState) {
			// State tracking always advances (the overlay needs it even while
			// the policy is suspended during a benchmark run).
			state = nextState;
			if (suspended) {
				return;
			}
			if (applied === profile) {
				return;
			}
			applied = profile;
			apply(profile, nextState);
		}

		function scheduleIdle() {
			clearIdleTimer();
			idleTimer = timers.setTimeout(function () {
				idleTimer = null;
				applyProfile(idle, "IDLE");
			}, debounceMs);
		}

		return {
			getState: function () {
				return state;
			},
			isSuspended: function () {
				return suspended;
			},
			/** Freeze profile application while the benchmark recorder runs. */
			setSuspended: function (value) {
				suspended = !!value;
				if (!suspended) {
					// Restore whatever profile matches the current camera state.
					applied = null;
					applyProfile(state === "INTERACTIVE" ? interactive : idle, state);
				}
			},
			/** Camera starts moving (Potree orbit-controls "start"). */
			onMotionStart: function () {
				clearIdleTimer();
				applyProfile(interactive, "INTERACTIVE");
				scheduleIdle();
			},
			/** Camera moved (Potree viewer "camera_changed", fires per frame). */
			onMotion: function () {
				applyProfile(interactive, "INTERACTIVE");
				scheduleIdle();
			},
			/** Camera stops moving (Potree orbit-controls "end"). */
			onMotionEnd: function () {
				applyProfile(interactive, "INTERACTIVE");
				scheduleIdle();
			},
			/** Apply the idle profile now (startup: show full detail). */
			applyIdle: function () {
				clearIdleTimer();
				applyProfile(idle, "IDLE");
			},
			dispose: function () {
				clearIdleTimer();
			}
		};
	}

	// -----------------------------------------------------------------------
	// Viewer/DOM glue (browser only) — exercised by VM-based tests
	// -----------------------------------------------------------------------

	function getPointcloud(viewer, deps) {
		if (deps.pointcloud) {
			return deps.pointcloud;
		}
		if (viewer && viewer.scene && Array.isArray(viewer.scene.pointclouds)) {
			return viewer.scene.pointclouds[0] || null;
		}
		return null;
	}

	/**
	 * Wrap ``viewer.renderer.setSize`` to count calls per frame and whether the
	 * width/height actually changed.  This is instrumentation ONLY — the
	 * original method is always delegated to, so Potree behaviour is unchanged
	 * and the vendor source is never patched.
	 */
	function instrumentSetSize(viewer) {
		var counter = {
			calls: 0, changes: 0, prevCalls: 0, prevChanges: 0,
			lastW: null, lastH: null, first: true, wrapped: false
		};
		var renderer = viewer && viewer.renderer;
		if (renderer && typeof renderer.setSize === "function" && !renderer.__p3aSetSizeWrapped) {
			renderer.__p3aSetSizeWrapped = true;
			counter.wrapped = true;
			var original = renderer.setSize;
			renderer.setSize = function (w, h, updateStyle) {
				counter.calls += 1;
				if (counter.first) {
					// First call only establishes the baseline size; it is not a
					// change (avoids counting the wrap-instant call as a resize).
					counter.first = false;
					counter.lastW = w;
					counter.lastH = h;
				} else if (w !== counter.lastW || h !== counter.lastH) {
					counter.changes += 1;
					counter.lastW = w;
					counter.lastH = h;
				}
				return original.call(renderer, w, h, updateStyle);
			};
		}
		return counter;
	}

	/** Apply a fixed profile to the live viewer (budget / point size / EDL). */
	function applyProfileToViewer(viewer, pointcloud, Potree, profile) {
		if (!viewer || !profile) {
			return;
		}
		if (typeof viewer.setPointBudget === "function") {
			viewer.setPointBudget(profile.pointBudget);
		}
		if (typeof viewer.setEDLEnabled === "function") {
			viewer.setEDLEnabled(profile.edl);
		}
		if (Potree && Potree.PointSizeType && pointcloud && pointcloud.material) {
			var name = String(profile.pointSizeType).toUpperCase();
			if (typeof Potree.PointSizeType[name] === "number") {
				pointcloud.material.pointSizeType = Potree.PointSizeType[name];
			}
		}
	}

	function createOverlayElement(viewer) {
		var el = document.createElement("div");
		el.id = "p3a_perf_overlay";
		el.setAttribute("aria-hidden", "true");
		el.style.cssText =
			"position:fixed;left:8px;top:8px;z-index:9990;" +
			"padding:6px 8px;font:11px/1.45 Consolas,Menlo,monospace;" +
			"color:#d7ffd7;background:rgba(0,0,0,0.55);border:1px solid rgba(255,255,255,0.25);" +
			"border-radius:3px;white-space:pre;pointer-events:none;user-select:none;";
		el.textContent = "P3A PERF loading…";
		appendToRenderArea(viewer, el);
		return el;
	}

	function appendToRenderArea(viewer, el) {
		var host = viewer && viewer.renderArea;
		if (host && typeof host.appendChild === "function") {
			host.appendChild(el);
		} else if (document.body) {
			document.body.appendChild(el);
		}
	}

	/**
	 * Benchmark recorder: start a labeled run (stationary/orbit/pan), apply the
	 * panel's explicit settings, suspend the two-state policy, accumulate frame
	 * deltas after a warm-up window, then snapshot on stop.  The camera can be
	 * auto-driven (no second render loop — the same ``render.pass.end`` hook
	 * mutates ``scene.view`` once per rendered frame) so orbit/pan runs are
	 * reproducible without the Owner holding the mouse.
	 */
	function createBenchmarkRecorder(viewer, pointcloud, Potree, api) {
		var policy = api.policy;
		var autoDrive = api.autoDrive;
		var readFrameState = api.readFrameState;
		var getPanelSettings = api.getPanelSettings;
		var onFit = typeof api.onFit === "function" ? api.onFit : null;
		var warmupMs = api.warmupMs;
		var recordMs = api.recordMs;
		var datasetPointCount = api.datasetPointCount;

		var run = null;
		var results = [];
		var autoStopTimer = null;
		var lastCompleted = null;
		// Bounded per-frame visible-point/node sample storage (the 15 s record
		// window at 240 Hz ≈ 3600 samples; the cap guards against any runaway
		// frame rate without doing any octree traversal — these values are
		// already computed by Potree's own visibility update each frame).
		var MAX_VISIBLE_SAMPLES = 10000;
		// Bounded completed-result storage.  Enough for Stage A (18) + Final
		// Confirmation (6) + a handful of repeat runs, while preventing unlimited
		// in-memory growth.  When full, the OLDEST result is dropped and the
		// newest is retained (deterministic FIFO eviction).
		var MAX_BENCHMARK_RESULTS = 50;

		function applyPanelSettings() {
			var s = getPanelSettings ? getPanelSettings() : null;
			if (!s) {
				return;
			}
			viewer.setPointBudget(s.budget);
			viewer.setEDLEnabled(s.edl);
			if (viewer.renderer && typeof viewer.renderer.setPixelRatio === "function") {
				viewer.renderer.setPixelRatio(s.pixelRatio);
			}
			if (Potree && Potree.PointSizeType && pointcloud && pointcloud.material) {
				var name = String(s.pointSizeType).toUpperCase();
				if (typeof Potree.PointSizeType[name] === "number") {
					pointcloud.material.pointSizeType = Potree.PointSizeType[name];
				}
			}
		}

		function start(interaction) {
			if (run) {
				return { ok: false, error: "a benchmark run is already active" };
			}
			if (!isValidBenchmarkValue("interactions", interaction)) {
				return { ok: false, error: "invalid interaction: " + String(interaction) };
			}
			policy.setSuspended(true);
			if (onFit) {
				// Reframe to the tight-bounds fit FIRST (policy is suspended, so
				// the fit's camera_changed cannot switch profiles mid-run); the
				// 5 s warm-up then lets the octree re-select for the fit view.
				onFit();
			}
			applyPanelSettings();
			autoDrive.setMode(interaction === "orbit" ? "orbit" : interaction === "pan" ? "pan" : null);
			run = {
				interaction: interaction,
				startWarmupTs: nowMillis(),
				startRecordTs: null,
				frames: 0,
				deltas: [],
				setSizeCalls: 0,
				setSizeChanges: 0,
				lastVisiblePoints: 0,
				lastVisibleNodes: 0,
				lastNodesLoading: 0,
				visiblePointSamples: [],
				visibleNodeSamples: []
			};
			if (recordMs > 0) {
				autoStopTimer = global.setTimeout(function () {
					autoStopTimer = null;
					stop();
				}, warmupMs + recordMs);
			}
			return { ok: true };
		}

		/** Called from the frame hook; accumulates once the warm-up has passed. */
		function accumulate(delta) {
			if (!run) {
				return;
			}
			var st = readFrameState();
			if (run.startRecordTs === null) {
				if (nowMillis() - run.startWarmupTs >= warmupMs) {
					run.startRecordTs = nowMillis();
				} else {
					return;
				}
			}
			run.frames += 1;
			run.deltas.push(delta);
			run.setSizeCalls += st.setSizeCalls;
			run.setSizeChanges += st.setSizeChanges;
			run.lastVisiblePoints = st.visiblePoints;
			run.lastVisibleNodes = st.visibleNodes;
			run.lastNodesLoading = st.nodesLoading;
			if (run.visiblePointSamples.length < MAX_VISIBLE_SAMPLES) {
				run.visiblePointSamples.push(st.visiblePoints);
				run.visibleNodeSamples.push(st.visibleNodes);
			}
		}

		function buildSnapshot() {
			var stats = computeStats(run.deltas);
			var canvas = viewer.renderer && viewer.renderer.domElement;
			var setSizePerFrame = run.frames > 0 ? run.setSizeCalls / run.frames : 0;
			// The ACTUAL measurement window runs from when the 5 s warm-up ended
			// (startRecordTs) until the run stopped (stopTs).  intervalFps is the
			// true achieved frame rate over that window (frames / duration),
			// which correctly reflects slower runs — unlike a median-derived FPS
			// that reports the display rate even when many frames were dropped.
			var measurementDurationMs =
				run.startRecordTs !== null ? run.stopTs - run.startRecordTs : 0;
			var intervalFps = measurementDurationMs > 0
				? (run.frames * 1000) / measurementDurationMs
				: 0;
			// Visible-point/node interval aggregates over the sampled frames.
			var pts = aggregateSamples(run.visiblePointSamples || []);
			var nodes = aggregateSamples(run.visibleNodeSamples || []);
			return {
				timestamp: new Date().toISOString(),
				datasetPointCount: datasetPointCount,
				budget: typeof viewer.getPointBudget === "function" ? viewer.getPointBudget() : null,
				actualVisiblePointsAtEnd: run.lastVisiblePoints,
				visiblePointsAverage: round1(pts.average),
				visiblePointsMedian: round1(pts.median),
				visiblePointsP95: round1(pts.p95),
				visiblePointsMax: round1(pts.max),
				visibleNodesAverage: round1(nodes.average),
				visibleNodesMedian: round1(nodes.median),
				visibleNodesMax: round1(nodes.max),
				visibleNodesAtEnd: run.lastVisibleNodes,
				nodesLoading: run.lastNodesLoading,
				pointSizeMode: pointSizeTypeName(pointcloud && pointcloud.material ? pointcloud.material.pointSizeType : -1),
				edl: typeof viewer.getEDLEnabled === "function" ? viewer.getEDLEnabled() : null,
				rendererPixelRatio: viewer.renderer && typeof viewer.renderer.getPixelRatio === "function" ? viewer.renderer.getPixelRatio() : null,
				devicePixelRatio: typeof global.devicePixelRatio === "number" ? global.devicePixelRatio : 1,
				canvasWidth: canvas ? canvas.width : null,
				canvasHeight: canvas ? canvas.height : null,
				interaction: run.interaction,
				frames: run.frames,
				measurementDurationMs: Math.round(measurementDurationMs),
				intervalFps: round1(intervalFps),
				averageFrameMs: round2(stats.average),
				medianFrameMs: round2(stats.median),
				p95FrameMs: round2(stats.p95),
				// Median-derived FPS is preserved under an explicit name only.
				medianFps: round1(stats.fps),
				setSizeCallsPerFrame: round2(setSizePerFrame),
				setSizeWidthHeightChanges: run.setSizeChanges
			};
		}

		function stop() {
			if (autoStopTimer !== null) {
				global.clearTimeout(autoStopTimer);
				autoStopTimer = null;
			}
			if (!run) {
				return null;
			}
			run.stopTs = nowMillis();
			var snap = buildSnapshot();
			results.push(snap);
			// Deterministic FIFO eviction: when the result cap is reached, drop
			// the OLDEST result and retain the newest, so memory stays bounded.
			if (results.length > MAX_BENCHMARK_RESULTS) {
				results.shift();
			}
			if (typeof console !== "undefined" && console.log) {
				console.log("[p3a-benchmark]", JSON.stringify(snap));
			}
			lastCompleted =
				"completed: " + snap.interaction + " — intervalFps " + snap.intervalFps +
				", " + snap.frames + " frames";
			run = null;
			autoDrive.setMode(null);
			policy.setSuspended(false);
			return snap;
		}

		/**
		 * Deterministic run-phase view used by the throttled status line.
		 * Returns ``{active, phase, remainingMs, lastCompleted}`` where phase is
		 * "warming" (before 5 s) or "recording" (the 15 s measurement window).
		 */
		function getRunPhase(now) {
			if (!run) {
				return { active: false, phase: "idle", remainingMs: 0, lastCompleted: lastCompleted };
			}
			if (run.startRecordTs === null) {
				return {
					active: true,
					phase: "warming",
					remainingMs: Math.max(0, warmupMs - (now - run.startWarmupTs)),
					lastCompleted: lastCompleted
				};
			}
			return {
				active: true,
				phase: "recording",
				remainingMs: Math.max(0, recordMs - (now - run.startRecordTs)),
				lastCompleted: lastCompleted
			};
		}

		return {
			start: start,
			stop: stop,
			accumulate: accumulate,
			isActive: function () {
				return run !== null;
			},
			getRunPhase: getRunPhase,
			results: function () {
				return results.slice();
			},
			clearResults: function () {
				results = [];
			},
			MAX_BENCHMARK_RESULTS: MAX_BENCHMARK_RESULTS
		};
	}

	function round2(v) {
		return Math.round(v * 100) / 100;
	}
	function round1(v) {
		return Math.round(v * 10) / 10;
	}

	/**
	 * Continuous camera motion driver.  Runs inside the ``render.pass.end``
	 * frame hook (NOT a second loop): it only mutates ``scene.view`` once per
	 * rendered frame, and Potree's own next-frame update re-derives the camera,
	 * which fires the normal ``camera_changed`` event.
	 *
	 * ORBIT must be a real orbit, not a yaw-in-place: Potree's ``controls.update``
	 * re-derives ``view.position = pivot + direction * -radius`` every frame, so
	 * mutating only ``view.yaw`` makes the camera spin about its own position
	 * while the cloud slides out of the frustum (which made earlier "orbit"
	 * benchmark runs measure sparse near-empty views).  A true orbit captures the
	 * look-at pivot BEFORE the yaw step, then repositions the camera to the same
	 * pivot at the same radius with the NEW direction — the cloud stays framed.
	 */
	function createAutoDrive(viewer) {
		var mode = null;
		var lastTs = null;
		// Moderate, sustained camera motion: ~0.5 rad/s orbit and a slow lateral
		// pan (fraction of the view radius per second) so the cloud stays in
		// view for the whole 20 s benchmark run.
		var ORBIT_RAD_PER_S = 0.5;
		var PAN_RADIUS_PER_S = 0.04;

		return {
			setMode: function (m) {
				mode = m;
				lastTs = null;
			},
			getMode: function () {
				return mode;
			},
			step: function (now) {
				if (!mode) {
					lastTs = null;
					return;
				}
				if (lastTs === null) {
					lastTs = now;
					return;
				}
				var dt = Math.min(0.1, (now - lastTs) / 1000);
				lastTs = now;
				var view = viewer && viewer.scene && viewer.scene.view;
				if (!view) {
					return;
				}
				if (mode === "orbit") {
					// Capture the CURRENT look-at pivot, then orbit around it.
					var pivot = view.getPivot();
					view.yaw += ORBIT_RAD_PER_S * dt;
					view.position.copy(pivot).sub(
						view.direction.multiplyScalar(view.radius)
					);
				} else if (mode === "pan") {
					view.pan(view.radius * PAN_RADIUS_PER_S * dt, 0);
				}
			}
		};
	}

	// -----------------------------------------------------------------------
	// Benchmark control panel (small, bottom-left, interactive)
	// -----------------------------------------------------------------------

	/**
	 * Build a labelled ``<select>``.  Returns ``{label, select}`` so the panel
	 * can BOTH append the visual label AND read/preset the actual select —
	 * preselecting ``<select>.value`` on the wrapper label would silently do
	 * nothing, leaving the control on its first option.
	 */
	function createSelect(id, options, labelText, defaultValue) {
		var label = document.createElement("label");
		label.textContent = labelText + " ";
		var sel = document.createElement("select");
		sel.id = id;
		for (var i = 0; i < options.length; i++) {
			var opt = document.createElement("option");
			opt.value = String(options[i]);
			opt.textContent = String(options[i]);
			sel.appendChild(opt);
		}
		if (defaultValue !== undefined) {
			sel.value = String(defaultValue);
		}
		label.appendChild(sel);
		return { label: label, select: sel };
	}

	function createBenchmarkPanel(viewer, recorder) {
		var panel = document.createElement("div");
		panel.id = "p3a_bench_panel";
		panel.style.cssText =
			"position:fixed;left:8px;bottom:8px;z-index:9991;" +
			"padding:6px 8px;font:11px/1.5 Consolas,Menlo,monospace;" +
			"color:#e0e0e0;background:rgba(0,0,0,0.6);border:1px solid rgba(255,255,255,0.25);" +
			"border-radius:3px;max-width:360px;";

		// Defaults match the current fixed normal default (750k budget, ADAPTIVE,
		// EDL ON, pixel ratio 1.0) so a first manual benchmark run starts from
		// the same profile as normal startup.
		var budgetSel = createSelect("p3a_bench_budget", ALLOWED.budgets, "budget", 750000);
		var sizeSel = createSelect("p3a_bench_size", ALLOWED.pointSizeTypes, "size", "adaptive");
		var prSel = createSelect("p3a_bench_pr", ALLOWED.pixelRatios, "pixelRatio", 1.0);
		var edlLabel = document.createElement("label");
		edlLabel.textContent = "EDL ";
		var edlSel = document.createElement("select");
		edlSel.id = "p3a_bench_edl";
		var on = document.createElement("option");
		on.value = "true";
		on.textContent = "ON";
		var off = document.createElement("option");
		off.value = "false";
		off.textContent = "OFF";
		edlSel.appendChild(on);
		edlSel.appendChild(off);
		edlSel.value = "true";
		edlLabel.appendChild(edlSel);

		function btn(text, interaction) {
			var b = document.createElement("button");
			b.type = "button";
			b.textContent = text;
			b.style.cssText =
				"margin:2px 3px 0 0;padding:2px 6px;font:11px sans-serif;" +
				"border:1px solid rgba(255,255,255,0.35);border-radius:3px;" +
				"background:rgba(255,255,255,0.15);color:#fff;cursor:pointer;";
			b.addEventListener("click", function () {
				var res = recorder.start(interaction);
				if (!res.ok) {
					status.textContent = "cannot start: " + (res.error || "unknown");
					return;
				}
				status.textContent = interaction + ": warming…";
			});
			return b;
		}

		// ONE-CLICK sequence runners: run a fixed list of (budget, interaction)
		// conditions sequentially under the fixed baseline (ADAPTIVE / EDL ON /
		// pixel ratio 1.0), letting each run warm up 5 s + measure 15 s and
		// auto-stop before the next.  Keep the tab in the foreground.  Note:
		// pixel-ratio selects use the OPTION VALUE string (String(1.0) === "1");
		// setting the literal "1.0" leaves the select empty -> pixelRatio 0.
		var stageARunning = false;
		var stageAbort = false;
		function sleepMs(ms) {
			return new Promise(function (resolve) {
				global.setTimeout(resolve, ms);
			});
		}
		async function runConditions(conditions, label) {
			if (stageARunning) {
				return;
			}
			stageARunning = true;
			stageAbort = false;
			status.textContent = label + ": forcing baseline (ADAPTIVE / EDL ON / PR 1.0)…";
			sizeSel.select.value = "adaptive";
			edlSel.value = "true";
			prSel.select.value = String(1.0); // "1", matches the <option> value
			for (var ci = 0; ci < conditions.length && !stageAbort; ci++) {
				var condition = conditions[ci];
				budgetSel.select.value = String(condition.budget);
				var interaction = condition.interaction;
				status.textContent = label + " " + condition.budget + " " + interaction + ": warming…";
				var res = recorder.start(interaction);
				if (!res.ok) {
					status.textContent = label + " error @ " + condition.budget + "/" + interaction + ": " + (res.error || "?");
					stageARunning = false;
					return;
				}
				while (recorder.isActive() && !stageAbort) {
					await sleepMs(400);
				}
			}
			status.textContent = (stageAbort ? label + " aborted" : label + " complete") +
				" — " + recorder.results().length + " results (Show / Copy JSON)";
			stageARunning = false;
		}
		async function runStageA() {
			var conditions = [];
			for (var b = 0; b < ALLOWED.budgets.length; b++) {
				for (var i = 0; i < 3; i++) {
					conditions.push({
						budget: ALLOWED.budgets[b],
						interaction: ["stationary", "orbit", "pan"][i]
					});
				}
			}
			return runConditions(conditions, "Stage A");
		}
		async function runFinalConfirmation() {
			var conditions = [];
			[750000, 1000000].forEach(function (budget) {
				["stationary", "orbit", "pan"].forEach(function (interaction) {
					conditions.push({ budget: budget, interaction: interaction });
				});
			});
			return runConditions(conditions, "Final Confirmation");
		}
		async function runRepeat400kOrbit() {
			return runConditions([{ budget: 400000, interaction: "orbit" }], "400k Orbit");
		}
		function actionBtn(text, onClick, accent) {
			var b = document.createElement("button");
			b.type = "button";
			b.textContent = text;
			b.style.cssText =
				"margin:2px 3px 0 0;padding:2px 6px;font:11px sans-serif;" +
				"border:1px solid " + (accent || "rgba(255,255,255,0.35)") + ";border-radius:3px;" +
				"background:rgba(255,255,255,0.15);color:#fff;cursor:pointer;";
			b.addEventListener("click", onClick);
			return b;
		}

		var stopBtn = document.createElement("button");
		stopBtn.type = "button";
		stopBtn.textContent = "Stop";
		stopBtn.style.cssText =
			"margin:2px 3px 0 0;padding:2px 6px;font:11px sans-serif;" +
			"border:1px solid rgba(255,180,180,0.5);border-radius:3px;" +
			"background:rgba(180,60,60,0.4);color:#fff;cursor:pointer;";
		stopBtn.addEventListener("click", function () {
			stageAbort = true; // also aborts an in-flight Stage A sequence
			var snap = recorder.stop();
			status.textContent = snap
				? "stopped: " + snap.interaction + " intervalFps=" + snap.intervalFps
				: "no active run";
		});

		var status = document.createElement("div");
		status.id = "p3a_bench_status";
		status.style.cssText = "margin-top:4px;color:#ffd;white-space:nowrap;";
		status.textContent = "idle — press Stationary / Orbit / Pan";

		var resultTextarea = document.createElement("textarea");
		resultTextarea.id = "p3a_bench_results";
		resultTextarea.readOnly = true;
		resultTextarea.style.cssText =
			"display:none;width:100%;height:120px;margin-top:4px;font:10px/1.3 monospace;" +
			"background:rgba(0,0,0,0.4);color:#cfc;border:1px solid rgba(255,255,255,0.2);";

		var copyBtn = document.createElement("button");
		copyBtn.type = "button";
		copyBtn.textContent = "Show / Copy JSON";
		copyBtn.style.cssText =
			"margin:2px 3px 0 0;padding:2px 6px;font:11px sans-serif;" +
			"border:1px solid rgba(255,255,255,0.35);border-radius:3px;" +
			"background:rgba(255,255,255,0.15);color:#fff;cursor:pointer;";
		copyBtn.addEventListener("click", function () {
			var list = recorder.results();
			if (resultTextarea.style.display === "none") {
				resultTextarea.style.display = "block";
			}
			resultTextarea.value = JSON.stringify(list, null, 2);
			resultTextarea.select();
			if (global.navigator && navigator.clipboard && navigator.clipboard.writeText) {
				navigator.clipboard.writeText(resultTextarea.value).then(null, function () {});
			}
		});

		panel.appendChild(budgetSel.label);
		panel.appendChild(sizeSel.label);
		panel.appendChild(edlLabel);
		panel.appendChild(prSel.label);
		panel.appendChild(btn("Stationary", "stationary"));
		panel.appendChild(btn("Orbit", "orbit"));
		panel.appendChild(btn("Pan", "pan"));
		panel.appendChild(actionBtn("Run Stage A (18)", function () { runStageA(); }, "rgba(120,220,120,0.5)"));
		panel.appendChild(actionBtn("Run Final Confirmation (6)", function () { runFinalConfirmation(); }, "rgba(120,180,220,0.5)"));
		panel.appendChild(actionBtn("Repeat 400k Orbit", function () { runRepeat400kOrbit(); }, "rgba(220,180,120,0.5)"));
		panel.appendChild(stopBtn);
		panel.appendChild(copyBtn);
		panel.appendChild(status);
		panel.appendChild(resultTextarea);

		appendToRenderArea(viewer, panel);

		return {
			element: panel,
			settings: function () {
				// A 0 / NaN pixel ratio is never valid (it would zero the
				// renderer canvas via THREE setPixelRatio -> setSize); fall back
				// to 1.0 if the select is empty or malformed.
				var pr = Number(prSel.select.value);
				return {
					budget: Number(budgetSel.select.value),
					pointSizeType: String(sizeSel.select.value),
					edl: edlSel.value === "true",
					pixelRatio: isFiniteNumber(pr) && pr > 0 ? pr : 1
				};
			},
			setStatus: function (text) {
				status.textContent = text;
			},
			runStageA: runStageA,
			runFinalConfirmation: runFinalConfirmation,
			runRepeat400kOrbit: runRepeat400kOrbit,
			isStageARunning: function () {
				return stageARunning;
			}
		};
	}

	// -----------------------------------------------------------------------
	// init — wire measurement + overlay + policy + benchmark into a live viewer
	// -----------------------------------------------------------------------

	/**
	 * Main entry, called by viewer.js after the point cloud has loaded and the
	 * initial fit has been scheduled.  Returns the created instance, or null if
	 * the viewer is not usable.
	 *
	 * deps = { viewer, pointcloud, cfg, Potree }
	 */
	function init(deps) {
		if (!deps || !deps.viewer) {
			return null;
		}
		var viewer = deps.viewer;
		var Potree = deps.Potree || global.Potree;
		var cfg = deps.cfg || {};
		var perfCfg = cfg.performance || {};
		var pointcloud = getPointcloud(viewer, deps);

		var frameWindow = isFiniteNumber(perfCfg.frameWindow)
			? clampInt(perfCfg.frameWindow, 16, 2000)
			: FRAME_WINDOW;
		var overlayHz = isFiniteNumber(perfCfg.overlayHz) && perfCfg.overlayHz > 0
			? perfCfg.overlayHz
			: 4;
		var overlayIntervalMs = 1000 / overlayHz;
		var idleDebounceMs = isFiniteNumber(perfCfg.idleDebounceMs)
			? clampInt(perfCfg.idleDebounceMs, 50, 5000)
			: IDLE_DEBOUNCE_MS;
		var bm = perfCfg.benchmark || {};
		var warmupMs = isFiniteNumber(bm.warmupMs) ? clampInt(bm.warmupMs, 0, 120000) : BENCHMARK_DEFAULT_WARMUP_MS;
		var recordMs = isFiniteNumber(bm.recordMs) ? clampInt(bm.recordMs, 1000, 600000) : BENCHMARK_DEFAULT_RECORD_MS;

		var baseline = perfCfg.baseline || {};
		var interactiveProfile = {
			pointBudget: isFiniteNumber((perfCfg.interactive || {}).pointBudget)
				? (perfCfg.interactive.pointBudget)
				: isFiniteNumber(baseline.pointBudget) ? baseline.pointBudget : 1000000,
			pointSizeType: ((perfCfg.interactive || {}).pointSizeType || baseline.pointSizeType || "adaptive").toLowerCase(),
			edl: (perfCfg.interactive || {}).edl !== undefined ? !!perfCfg.interactive.edl : !!baseline.edl
		};
		var idleProfile = {
			pointBudget: isFiniteNumber((perfCfg.idle || {}).pointBudget)
				? (perfCfg.idle.pointBudget)
				: isFiniteNumber(baseline.pointBudget) ? baseline.pointBudget : 1000000,
			pointSizeType: ((perfCfg.idle || {}).pointSizeType || baseline.pointSizeType || "adaptive").toLowerCase(),
			edl: (perfCfg.idle || {}).edl !== undefined ? !!perfCfg.idle.edl : !!baseline.edl
		};

		var history = createBoundedHistory(frameWindow);
		var throttle = createThrottle(overlayIntervalMs);
		var setSizeCounter = instrumentSetSize(viewer);

		var policy = createPerfPolicy({
			interactive: interactiveProfile,
			idle: idleProfile,
			debounceMs: idleDebounceMs,
			apply: function (profile, state) {
				applyProfileToViewer(viewer, pointcloud, Potree, profile);
			}
		});

		var autoDrive = createAutoDrive(viewer);

		var overlay = createOverlayElement(viewer);

		var lastFrameTs = null;
		var lastSetSizeCalls = 0;
		var lastSetSizeChanges = 0;
		var datasetPointCount = isFiniteNumber(perfCfg.datasetPointCount) ? perfCfg.datasetPointCount : 0;

		var instance = {
			viewer: viewer,
			pointcloud: pointcloud,
			policy: policy,
			history: history,
			throttle: throttle,
			autoDrive: autoDrive,
			setSizeCounter: setSizeCounter,
			getSetSizePerFrame: function () {
				return { calls: lastSetSizeCalls, changes: lastSetSizeChanges };
			}
		};

		var readFrameState = function () {
			var pc = instance.pointcloud;
			var nodes = pc && pc.numVisibleNodes != null
				? pc.numVisibleNodes
				: pc && pc.visibleNodes ? pc.visibleNodes.length : 0;
			var loading = Potree && Potree.numNodesLoading != null ? Potree.numNodesLoading : 0;
			return {
				visiblePoints: pc && pc.numVisiblePoints != null ? pc.numVisiblePoints : 0,
				visibleNodes: nodes,
				nodesLoading: loading,
				setSizeCalls: lastSetSizeCalls,
				setSizeChanges: lastSetSizeChanges
			};
		};

		var panelSettingsGetter = null;
		var recorder = createBenchmarkRecorder(viewer, pointcloud, Potree, {
			policy: policy,
			autoDrive: autoDrive,
			readFrameState: readFrameState,
			getPanelSettings: function () {
				return panelSettingsGetter ? panelSettingsGetter() : null;
			},
			// Reframe to the tight-bounds view at the start of every run so each
			// measurement starts from the same framing (falls back to the current
			// camera when viewer.js does not supply a fit callback).
			onFit: typeof deps.onFit === "function" ? deps.onFit : null,
			warmupMs: warmupMs,
			recordMs: recordMs,
			datasetPointCount: datasetPointCount
		});
		var panel = createBenchmarkPanel(viewer, recorder);
		panelSettingsGetter = panel.settings;
		instance.panel = panel;
		instance.recorder = recorder;

		// Lightweight run-phase status line: warming → recording countdown →
		// completed.  Runs on the same 4 Hz throttle as the overlay.
		function refreshBenchStatus(now) {
			if (typeof panel.setStatus !== "function") {
				return;
			}
			var ph = recorder.getRunPhase(now);
			if (!ph.active) {
				panel.setStatus(ph.lastCompleted ? ph.lastCompleted : "idle — press Stationary / Orbit / Pan");
				return;
			}
			var label = ph.phase === "warming" ? "warming " : "recording ";
			panel.setStatus(label + (ph.remainingMs / 1000).toFixed(1) + "s…");
		}
		instance.refreshBenchStatus = refreshBenchStatus;

		function onFrameEnd() {
			var now = nowMillis();

			// Per-frame setSize attribution FIRST so the recorder's accumulate
			// reads the CURRENT frame's values.  The counter is cumulative and
			// never reset; each frame consumes the delta since the last frame, so
			// a resize change is attributed to the frame that observed it even if
			// that frame is the final one before a benchmark stop.
			lastSetSizeCalls = setSizeCounter.calls - setSizeCounter.prevCalls;
			lastSetSizeChanges = setSizeCounter.changes - setSizeCounter.prevChanges;
			setSizeCounter.prevCalls = setSizeCounter.calls;
			setSizeCounter.prevChanges = setSizeCounter.changes;

			if (lastFrameTs !== null) {
				var delta = now - lastFrameTs;
				if (delta >= 0) {
					history.push(delta);
					recorder.accumulate(delta);
				}
			}
			lastFrameTs = now;

			autoDrive.step(now);

			if (throttle.shouldUpdate(now)) {
				updateOverlay(now);
				refreshBenchStatus(now);
			}
		}

		function updateOverlay(now) {
			var stats = computeStats(history.toArray());
			var pc = instance.pointcloud;
			var canvas = viewer.renderer && viewer.renderer.domElement;
			var state = policy.getState() === "INTERACTIVE" ? "MOVING" : "IDLE";
			var sizeName = pc && pc.material ? pointSizeTypeName(pc.material.pointSizeType) : "?";
			var lines = [
				"P3A PERF  " + state,
				"FPS " + fmtNum(stats.fps, 1),
				"median " + fmtNum(stats.median, 1) + " ms",
				"avg " + fmtNum(stats.average, 1) + " ms",
				"P95 " + fmtNum(stats.p95, 1) + " ms",
				"budget " + addCommas(typeof viewer.getPointBudget === "function" ? viewer.getPointBudget() : 0),
				"visible " + addCommas(pc && pc.numVisiblePoints != null ? pc.numVisiblePoints : 0) + " pts",
				"nodes " + addCommas(pc && pc.numVisibleNodes != null ? pc.numVisibleNodes : (pc && pc.visibleNodes ? pc.visibleNodes.length : 0)) +
					" (load " + addCommas(Potree && Potree.numNodesLoading != null ? Potree.numNodesLoading : 0) + ")",
				"size " + sizeName,
				"EDL " + (typeof viewer.getEDLEnabled === "function" && viewer.getEDLEnabled() ? "ON" : "OFF"),
				"pixelRatio " + fmtNum(viewer.renderer && viewer.renderer.getPixelRatio ? viewer.renderer.getPixelRatio() : 1, 2) +
					" (dpr " + fmtNum(typeof global.devicePixelRatio === "number" ? global.devicePixelRatio : 1, 2) + ")",
				"canvas " + (canvas ? canvas.width : "?") + " x " + (canvas ? canvas.height : "?"),
				"setSize " + lastSetSizeCalls + "/frame (chg " + lastSetSizeChanges + ")"
			];
			overlay.textContent = lines.join("\n");
		}

		instance.onFrameEnd = onFrameEnd;
		instance.updateOverlay = updateOverlay;

		// Wire into the EXISTING Potree render cadence — no second loop.
		if (typeof viewer.addEventListener === "function") {
			viewer.addEventListener("render.pass.end", onFrameEnd);
			viewer.addEventListener("camera_changed", function () {
				policy.onMotion();
			});
		}
		if (viewer.orbitControls && typeof viewer.orbitControls.addEventListener === "function") {
			viewer.orbitControls.addEventListener("start", function () {
				policy.onMotionStart();
			});
			viewer.orbitControls.addEventListener("end", function () {
				policy.onMotionEnd();
			});
		}

		// Startup: apply the IDLE (full-detail) profile once so the initial
		// post-fit view shows the maximum detail the fixed idle profile allows.
		policy.applyIdle();

		instance.dispose = function () {
			if (typeof viewer.removeEventListener === "function") {
				viewer.removeEventListener("render.pass.end", onFrameEnd);
			}
			policy.dispose();
		};

		return instance;
	}

	// -----------------------------------------------------------------------
	// Exports
	// -----------------------------------------------------------------------

	global.p3aPerf = {
		FRAME_WINDOW: FRAME_WINDOW,
		OVERLAY_INTERVAL_MS: OVERLAY_INTERVAL_MS,
		IDLE_DEBOUNCE_MS: IDLE_DEBOUNCE_MS,
		BENCHMARK_DEFAULT_WARMUP_MS: BENCHMARK_DEFAULT_WARMUP_MS,
		BENCHMARK_DEFAULT_RECORD_MS: BENCHMARK_DEFAULT_RECORD_MS,
		BENCHMARK_BUDGETS: BENCHMARK_BUDGETS,
		ALLOWED: ALLOWED,
		createBoundedHistory: createBoundedHistory,
		computeStats: computeStats,
		createThrottle: createThrottle,
		isValidBenchmarkValue: isValidBenchmarkValue,
		createPerfPolicy: createPerfPolicy,
		init: init,
		pointSizeTypeName: pointSizeTypeName
	};
})(typeof window !== "undefined" ? window : globalThis);
