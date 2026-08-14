/**
 * TreeLineWarning P3A — Giro3D + COPC viewer (Candidate B).
 *
 * Renders the SAME dense real-RGB ROI (982,504 pts) that Potree Candidate A
 * renders, but from a derived COPC file and with the Giro3D engine, so the
 * Owner can compare CloudCompare / Potree / Giro3D on identical data.
 *
 * Implemented: RGB point cloud, orbit (left drag), pan (right drag), zoom
 * (wheel), useful initial framing from the OCCUPIED point bounds, Fit / Reset,
 * click -> world XYZ (point picking), loading / visible error state, and a
 * small diagnostics section.  No production UI, no tower review UI, no
 * prediction, no scientific pipeline.
 *
 * The pure decision helpers (LAS-header bounds parsing, framing math,
 * click-coordinate plumbing) are exposed on `window.p3aGiro3dUtil` and use no
 * browser APIs, so the Node test harness can exercise them directly.
 */
(function () {
	"use strict";

	// -----------------------------------------------------------------------
	// Pure helpers (no browser APIs — Node-testable).  Defined FIRST so they are
	// available even when the Giro3D bundle is absent (fail-open diagnostics).
	// -----------------------------------------------------------------------

	/**
	 * Parse the occupied XYZ bounds from a LAS/COPC header (the first 375 bytes).
	 * LAS 1.x header: max X=179, min X=187, max Y=195, min Y=203, max Z=211,
	 * min Z=219 (little-endian doubles).
	 */
	function parseLasHeaderBounds(bytes) {
		const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
		function read(offset) {
			return view.getFloat64(offset, true);
		}
		return {
			min: { x: read(187), y: read(203), z: read(219) },
			max: { x: read(179), y: read(195), z: read(211) }
		};
	}

	/**
	 * Compute an angled camera position + look-at target that frames the
	 * occupied box (plain {x,y,z} min/max) with margin.  ``direction`` is the
	 * normalized offset from the target (default: from up-and-to-the-side so the
	 * corridor/towers are recognizable immediately).
	 */
	function computeFitCamera(min, max, fovDeg, direction) {
		const center = {
			x: (min.x + max.x) / 2,
			y: (min.y + max.y) / 2,
			z: (min.z + max.z) / 2
		};
		const size = { x: max.x - min.x, y: max.y - min.y, z: max.z - min.z };
		const maxDim = Math.max(size.x, size.y, size.z) || 1;
		const fovRad = ((fovDeg || 60) * Math.PI) / 180;
		const dist = maxDim / 2 / Math.tan(fovRad / 2);
		const dir = direction || { x: -0.6, y: -0.6, z: 0.5 };
		const len = Math.sqrt(dir.x * dir.x + dir.y * dir.y + dir.z * dir.z) || 1;
		const nx = dir.x / len;
		const ny = dir.y / len;
		const nz = dir.z / len;
		const margin = 1.15;
		return {
			center: center,
			size: size,
			maxDim: maxDim,
			position: {
				x: center.x + nx * dist * margin,
				y: center.y + ny * dist * margin,
				z: center.z + nz * dist * margin
			},
			target: { x: center.x, y: center.y, z: center.z }
		};
	}

	/** Convert a Giro3D pick result array into {x,y,z} (world/source frame) or null. */
	function pickResultToXyz(results) {
		if (!results || results.length === 0) {
			return null;
		}
		const first = results.find(function (r) {
			return r && r.point;
		}) || results[0];
		if (!first || !first.point) {
			return null;
		}
		const p = first.point;
		return { x: p.x, y: p.y, z: p.z };
	}

	window.p3aGiro3dUtil = {
		parseLasHeaderBounds: parseLasHeaderBounds,
		computeFitCamera: computeFitCamera,
		pickResultToXyz: pickResultToXyz
	};

	// -----------------------------------------------------------------------
	// Browser glue (requires the Giro3D bundle; without it the viewer fails
	// visibly but the pure diagnostics above remain available)
	// -----------------------------------------------------------------------

	const G = window.P3AGiro3D;
	if (!G) {
		// showError is defined below, but at this point the DOM elements exist;
		// route through a minimal direct banner so the failure is visible.
		const el = document.getElementById("p3a_giro3d_error");
		if (el) {
			el.style.display = "block";
			el.textContent = "P3A Giro3D error: Giro3D bundle failed to load (window.P3AGiro3D missing).";
		}
		return;
	}
	const { Instance, PointCloud, COPCSource, CoordinateSystem, setLazPerfPath, THREE, MapControls } = G;

	const errorEl = document.getElementById("p3a_giro3d_error");
	const loadingEl = document.getElementById("p3a_giro3d_loading");
	const statusEl = document.getElementById("p3a_giro3d_status");
	const helpEl = document.getElementById("p3a_giro3d_help");
	const fitBtn = document.getElementById("p3a_giro3d_fit");

	function showError(message) {
		const text = String(message == null ? "unknown error" : message);
		console.error("[p3a-giro3d] " + text);
		errorEl.style.display = "block";
		errorEl.textContent = "P3A Giro3D error: " + text.replace(/[<>&]/g, "");
	}

	window.addEventListener("error", function (event) {
		showError("uncaught error: " + (event.message || "unknown"));
	});
	window.addEventListener("unhandledrejection", function (event) {
		const reason = event.reason;
		showError(
			"unhandled rejection: " + (reason && reason.message ? reason.message : String(reason))
		);
	});

	function showLoading(visible, text) {
		loadingEl.style.display = visible ? "block" : "none";
		if (text) {
			loadingEl.textContent = text;
		}
	}

	function fmtVec(v) {
		if (!v) {
			return "—";
		}
		return v.x.toFixed(2) + ", " + v.y.toFixed(2) + ", " + v.z.toFixed(2);
	}

	async function loadConfig() {
		const response = await fetch("poc_config.json", { cache: "no-store" });
		if (!response.ok) {
			throw new Error("failed to load poc_config.json: HTTP " + response.status);
		}
		return response.json();
	}

	function updateStatus(cfg, viewer) {
		const pc = viewer.pointCloud;
		const lines = [
			"engine: Giro3D",
			"dataset: " + cfg.dataset,
			"total pts: " + (pc && pc.pointCount != null ? Number(pc.pointCount).toLocaleString() : "—"),
			"displayed: " + (pc && pc.displayedPointCount != null ? Number(pc.displayedPointCount).toLocaleString() : "—"),
			"budget: " + (pc && pc.pointBudget != null ? Number(pc.pointBudget).toLocaleString() : "null"),
			"decimation: " + (pc && pc.decimation != null ? pc.decimation : "—"),
			"subdivThresh: " + (pc && pc.subdivisionThreshold != null ? pc.subdivisionThreshold : "—"),
			"pointSize: " + (pc && pc.pointSize != null ? pc.pointSize : "—"),
			"loading: " + (viewer.instance.loading
				? "yes (" + Math.round(viewer.instance.progress * 100) + "%)"
				: "no"),
			"cam: " + fmtVec(viewer.lastCameraPos),
			"target: " + fmtVec(viewer.lastTarget),
			"last XYZ: " + fmtVec(viewer.lastPick)
		];
		statusEl.textContent = lines.join("\n");
	}

	async function start() {
		const cfg = await loadConfig();
		document.title = cfg.title;
		if (cfg.controlHelp) {
			helpEl.textContent = Object.values(cfg.controlHelp).join("　|　");
		}

		// LAZ decompression WASM is served locally (never the default CDN).
		setLazPerfPath(cfg.lazPerfPath || "/laz-perf/");

		const viewer = {
			instance: null,
			pointCloud: null,
			controls: null,
			bounds: null,
			lastCameraPos: null,
			lastTarget: null,
			lastPick: null
		};

		// Occupied bounds come straight from the COPC LAS header (range fetch).
		const headerResp = await fetch(cfg.copcUrl, { headers: { Range: "bytes=0-374" } });
		if (!headerResp.ok) {
			throw new Error("cannot fetch COPC header: HTTP " + headerResp.status);
		}
		const headerBytes = new Uint8Array(await headerResp.arrayBuffer());
		viewer.bounds = parseLasHeaderBounds(headerBytes);

		const instance = new Instance({
			target: "view",
			crs: CoordinateSystem.unknown,
			backgroundColor: cfg.viewer.backgroundColor,
			renderer: { logarithmicDepthBuffer: true }
		});
		viewer.instance = instance;

		if (cfg.viewer.edlEnabled) {
			instance.renderingOptions.enableEDL = true;
		}

		const source = new COPCSource({
			url: cfg.copcUrl,
			decimate: cfg.viewer.decimate || 1
		});

		const loadPromise = Promise.resolve().then(async function () {
			await source.initialize();
			const entity = new PointCloud({
				source: source,
				crs: CoordinateSystem.unknown,
				pointSize: cfg.viewer.pointSize,
				pointBudget: cfg.viewer.pointBudget
			});
			await instance.add(entity);
			return entity;
		});

		const timeoutMs = cfg.viewer.loadTimeoutMs || 60000;
		const entity = await withTimeout(
			loadPromise,
			timeoutMs,
			"point cloud load timed out after " + timeoutMs + " ms"
		);
		viewer.pointCloud = entity;
		showLoading(false);

		// -------------------------------------------------------------------
		// Controls: THREE MapControls -> left-drag rotate, right-drag pan,
		// wheel zoom (orbit style).
		// -------------------------------------------------------------------
		const camera = instance.view.camera;
		const controls = new MapControls(camera, instance.domElement);
		controls.enableDamping = true;
		controls.dampingFactor = 0.25;
		controls.update();
		instance.view.setControls(controls);
		viewer.controls = controls;

		// -------------------------------------------------------------------
		// Initial framing + Fit / Reset from the occupied bounds.
		// -------------------------------------------------------------------
		const fit = function () {
			const frame = computeFitCamera(viewer.bounds.min, viewer.bounds.max, cfg.viewer.fov);
			viewer.instance.view.camera.position.set(frame.position.x, frame.position.y, frame.position.z);
			viewer.instance.view.camera.lookAt(frame.target.x, frame.target.y, frame.target.z);
			viewer.controls.target.set(frame.target.x, frame.target.y, frame.target.z);
			viewer.controls.minDistance = frame.maxDim * 0.001;
			viewer.controls.maxDistance = frame.maxDim * 1000;
			viewer.controls.update();
			viewer.instance.notifyChange(viewer.instance.view.camera);
		};
		fitBtn.addEventListener("click", fit);
		fit();

		// -------------------------------------------------------------------
		// Click -> world XYZ (point picking)
		// -------------------------------------------------------------------
		instance.domElement.addEventListener("click", function (event) {
			const results = viewer.pointCloud.pick(new THREE.Vector2(event.clientX, event.clientY));
			viewer.lastPick = pickResultToXyz(results);
			updateStatus(cfg, viewer);
		});

		// -------------------------------------------------------------------
		// Diagnostics / camera tracking
		// -------------------------------------------------------------------
		instance.addEventListener("after-camera-update", function () {
			const cam = viewer.instance.view.camera;
			viewer.lastCameraPos = { x: cam.position.x, y: cam.position.y, z: cam.position.z };
			viewer.lastTarget = {
				x: viewer.controls.target.x,
				y: viewer.controls.target.y,
				z: viewer.controls.target.z
			};
		});
		setInterval(function () {
			updateStatus(cfg, viewer);
		}, 500);

		// -------------------------------------------------------------------
		// Manual LOD diagnostics (subdivision threshold / point budget) so the
		// Owner can test detail refinement without any code change.  An empty
		// point budget = null (Giro3D stops automatic decimation).
		// -------------------------------------------------------------------
		const threshInput = document.getElementById("p3a_giro3d_thresh");
		const budgetInput = document.getElementById("p3a_giro3d_budget");
		if (cfg.viewer.pointBudget != null) {
			budgetInput.value = cfg.viewer.pointBudget;
		}
		threshInput.addEventListener("change", function () {
			const v = Number(threshInput.value);
			if (isFinite(v) && v > 0 && viewer.pointCloud) {
				viewer.pointCloud.subdivisionThreshold = v;
			}
		});
		budgetInput.addEventListener("change", function () {
			if (!viewer.pointCloud) {
				return;
			}
			const text = budgetInput.value.trim();
			viewer.pointCloud.pointBudget = text === "" ? null : Number(text);
			updateStatus(cfg, viewer);
		});

		window.p3aGiro3dViewer = viewer;
	}

	function withTimeout(promise, ms, message) {
		return new Promise(function (resolve, reject) {
			const timer = setTimeout(function () {
				reject(new Error(message));
			}, ms);
			promise.then(
				function (value) {
					clearTimeout(timer);
					resolve(value);
				},
				function (reason) {
					clearTimeout(timer);
					reject(reason);
				}
			);
		});
	}

	start().catch(function (err) {
		showError(err && err.message ? err.message : String(err));
	});
})();
