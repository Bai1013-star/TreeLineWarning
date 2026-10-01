/**
 * TreeLineWarning P3A Potree POC viewer bootstrap.
 *
 * Loads `poc_config.json`, preflights the point-cloud metadata, constructs a
 * Potree.Viewer with explicit orbit controls, and loads the converted RGB
 * octree.  All startup/load failures (missing metadata, JS errors, unhandled
 * rejections, load timeouts) are routed to a visible in-page error banner
 * instead of a silent blank viewer.
 *
 * P3A-004 contract:
 *  - metadata.json is fetched and validated BEFORE Potree startup; on failure
 *    the visible banner is shown and Potree.loadPointCloud is never invoked.
 *  - window "error" and "unhandledrejection" handlers route to the banner.
 *  - the point-cloud load is bounded by a load timeout (Potree 1.8.2's
 *    loadPointCloud promise never rejects on a failed load, it simply never
 *    settles, so the timeout is what turns a hang into a visible error).
 *
 * P3A-010 contract:
 *  - visible-error capability and the global JS error/unhandled-rejection
 *    handlers are installed FIRST.
 *  - only then is the shared viewer utility (`p3aViewerUtil`, loaded from
 *    viewer_util.js) validated via `p3aDecideViewerStartup`.
 *  - if the helper is missing or corrupt the startup fails visibly and stops,
 *    never dereferencing the helper and never starting metadata/Potree load.
 *
 * Visual acceptance contract (P3A-010 owner findings):
 *  - VISUAL-001: after a successful load the view is framed to the TIGHT point
 *    bounds (from the metadata "position" attribute), not the octree's inflated
 *    cubic bounding box, so the cloud is not a tiny center dot.  A visible
 *    "Fit / Reset View" control restores this framing.
 *  - VISUAL-002: the active navigation control is explicitly
 *    `viewer.setControls(viewer.orbitControls)` (Potree 1.8.2), giving
 *    Left-drag rotate, Right-drag pan, Wheel zoom.  A small Chinese control
 *    hint is shown.
 */
(function () {
	"use strict";

	const renderArea = document.getElementById("potree_render_area");
	let errorShown = false;

	/**
	 * P3A-010: normalize a caught value into {name, message, stack} so the
	 * runtime error banner and console log are informative (error NAME + message,
	 * and the stack when an actual Error is available).
	 */
	function describeError(err) {
		if (err && typeof err === "object") {
			return {
				name: typeof err.name === "string" && err.name ? err.name : "Error",
				message:
					typeof err.message === "string" && err.message ? err.message : String(err),
				stack: typeof err.stack === "string" ? err.stack : ""
			};
		}
		if (err !== null && err !== undefined) {
			return { name: "Error", message: String(err), stack: "" };
		}
		return { name: "Error", message: "unknown error", stack: "" };
	}

	function sanitizeForHtml(text) {
		return String(text).replace(/[<>&]/g, "");
	}

	/**
	 * Show the visible fail-closed error banner.  The Owner UI shows the error
	 * name + message (and a short stack hint when available); the FULL stack is
	 * logged to the console for P3A debug (filesystem paths stay out of the
	 * banner).
	 */
	function showError(message, error) {
		const detail = describeError(error);
		let head = String(message == null ? "unknown error" : message);
		let bannerText = head;
		if (detail.stack) {
			const full = detail.name + ": " + detail.message;
			console.error("[p3a] " + full + "\n" + detail.stack);
			bannerText = full;
			const stackLine = detail.stack.split("\n").find(function (line) {
				return line.indexOf("viewer_") !== -1 || line.indexOf("viewer.js") !== -1;
			});
			if (stackLine) {
				bannerText += "\n" + stackLine.trim();
			}
		} else {
			console.error("[p3a] " + head);
		}
		if (errorShown) {
			return;
		}
		errorShown = true;
		renderArea.insertAdjacentHTML(
			"beforeend",
			'<div id="p3a_error_banner" style="position:absolute;z-index:9999;' +
				"left:16px;top:16px;background:#8b0000;color:#fff;" +
				"padding:12px 16px;font:14px sans-serif;max-width:70%;" +
				'border-radius:4px;white-space:pre-wrap">P3A POC error: ' +
				sanitizeForHtml(bannerText) +
				"</div>"
		);
	}

	function showLoading(label) {
		if (document.getElementById("p3a_loading_banner")) {
			return;
		}
		renderArea.insertAdjacentHTML(
			"beforeend",
			'<div id="p3a_loading_banner" style="position:absolute;z-index:9998;' +
				"left:16px;bottom:16px;background:rgba(0,0,0,0.6);color:#fff;" +
				"padding:10px 14px;font:13px sans-serif;border-radius:4px;" +
				'">Loading point cloud…</div>'
		);
	}

	function hideLoading() {
		const el = document.getElementById("p3a_loading_banner");
		if (el) {
			el.remove();
		}
	}

	// P3A-004/P3A-010: global error handlers installed before any startup work,
	// so even an early failure is visible.
	window.addEventListener("error", function (event) {
		showError("uncaught error: " + (event.message || "unknown"), event.error);
	});
	window.addEventListener("unhandledrejection", function (event) {
		const reason = event.reason;
		showError(
			"unhandled rejection: " +
				(reason && reason.message ? reason.message : String(reason)),
			reason
		);
	});

	// -----------------------------------------------------------------------
	// VISUAL-001: tight-bounds framing helpers
	// -----------------------------------------------------------------------

	/**
	 * Set the octree's tight bounding box from the metadata "position"
	 * attribute (converted into octree-local coordinates) so the official
	 * `viewer.fitToScreen()` frames the ACTUAL cloud instead of the octree's
	 * inflated cubic bounding box.
	 */
	function applyTightBounds(pointcloud, metadata) {
		if (
			!pointcloud ||
			!pointcloud.pcoGeometry ||
			!pointcloud.pcoGeometry.boundingBox ||
			!metadata ||
			!metadata.boundingBox ||
			!Array.isArray(metadata.attributes)
		) {
			return;
		}
		const positionAttr = metadata.attributes.find(function (a) {
			return a.name === "position";
		});
		if (!positionAttr || !positionAttr.min || !positionAttr.max) {
			return;
		}
		const off = metadata.boundingBox.min;
		const box = pointcloud.pcoGeometry.boundingBox.clone();
		box.min.set(
			positionAttr.min[0] - off[0],
			positionAttr.min[1] - off[1],
			positionAttr.min[2] - off[2]
		);
		box.max.set(
			positionAttr.max[0] - off[0],
			positionAttr.max[1] - off[1],
			positionAttr.max[2] - off[2]
		);
		pointcloud.pcoGeometry.tightBoundingBox = box;
	}

	function frameToPointcloud(viewer, pointcloud, metadata) {
		applyTightBounds(pointcloud, metadata);
		viewer.fitToScreen(); // Potree 1.8.2 official fit path
	}

	// -----------------------------------------------------------------------
	// VISUAL-001: Fit / Reset View control
	// -----------------------------------------------------------------------

	function addFitResetButton(onFit) {
		if (document.getElementById("p3a_fit_reset")) {
			return;
		}
		const el = document.createElement("button");
		el.id = "p3a_fit_reset";
		el.textContent = "Fit / Reset View（适配 / 重置）";
		el.style.cssText =
			"position:absolute;top:12px;right:12px;z-index:10000;" +
			"padding:8px 12px;font:13px sans-serif;border:1px solid rgba(255,255,255,0.4);" +
			"border-radius:4px;background:rgba(0,0,0,0.55);color:#fff;cursor:pointer;";
		el.addEventListener("click", onFit);
		renderArea.appendChild(el);
	}

	// -----------------------------------------------------------------------
	// VISUAL-002: control help hint (Chinese, unobtrusive)
	// -----------------------------------------------------------------------

	function addControlHelp(cfg) {
		if (document.getElementById("p3a_control_help")) {
			return;
		}
		const help = cfg.controlHelp || {};
		const parts = [help.left, help.right, help.wheel].filter(Boolean);
		if (parts.length === 0) {
			return;
		}
		const el = document.createElement("div");
		el.id = "p3a_control_help";
		el.textContent = parts.join("　|　");
		el.style.cssText =
			"position:absolute;left:50%;bottom:16px;transform:translateX(-50%);" +
			"z-index:9997;padding:6px 12px;font:12px sans-serif;border-radius:4px;" +
			"background:rgba(0,0,0,0.5);color:#fff;pointer-events:none;white-space:nowrap;";
		renderArea.appendChild(el);
	}

	async function loadConfig() {
		const response = await fetch("poc_config.json", { cache: "no-store" });
		if (!response.ok) {
			throw new Error(
				"failed to load poc_config.json: HTTP " + response.status
			);
		}
		return response.json();
	}

	// P3A-004: preflight the configured point-cloud metadata before Potree
	// startup.  Requires HTTP 200 and a Potree-octree-shaped JSON document.
	async function preflightMetadata(pointcloudUrl) {
		const response = await fetch(pointcloudUrl, { cache: "no-store" });
		if (!response.ok) {
			throw new Error(
				"point cloud metadata not reachable: HTTP " + response.status
			);
		}
		const metadata = await response.json();
		if (
			!metadata ||
			typeof metadata !== "object" ||
			!Array.isArray(metadata.attributes)
		) {
			throw new Error(
				"point cloud metadata is not valid Potree octree metadata"
			);
		}
		return metadata;
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

	async function start() {
		try {
			// P3A-010: validate the shared viewer utility BEFORE any
			// p3aViewerUtil dereference.  A missing/corrupt helper fails
			// visibly and stops; Potree load is never attempted.
			const decide = window.p3aDecideViewerStartup;
			if (typeof decide !== "function") {
				showError(
					"Viewer startup failed: required viewer utility failed to load."
				);
				return;
			}
			const decision = decide(window.p3aViewerUtil);
			if (!decision || !decision.ok) {
				showError(
					"Viewer startup failed: required viewer utility failed to load."
				);
				return;
			}
			const util = decision.util;

			const cfg = await loadConfig();
			document.title = cfg.title;

			// Preflight FIRST: a failed metadata fetch shows the banner and
			// never reaches Potree.loadPointCloud.
			const metadata = await preflightMetadata(cfg.pointcloudUrl);

			const viewer = new Potree.Viewer(renderArea);
			window.viewer = viewer;

			// VISUAL-002: explicitly select Potree 1.8.2 orbit controls
			// (Left-drag rotate, Right-drag pan, Wheel zoom).
			viewer.setControls(viewer.orbitControls);

			viewer.setFOV(cfg.viewer.fov);
			viewer.setEDLEnabled(cfg.viewer.edlEnabled);
			viewer.setPointBudget(cfg.viewer.pointBudget);
			viewer.setDescription(cfg.description);
			viewer.loadSettingsFromURL();

			viewer.loadGUI(function () {
				viewer.setLanguage("en");
			});

			showLoading(cfg.pointcloudName);
			addControlHelp(cfg);

			const timeoutMs = util.resolveLoadTimeoutMs(
				cfg.viewer && cfg.viewer.loadTimeoutMs
			);
			const loadPromise = Potree.loadPointCloud(
				cfg.pointcloudUrl,
				cfg.pointcloudName
			);
			const loaded = await withTimeout(
				loadPromise,
				timeoutMs,
				"point cloud load timed out after " + timeoutMs + " ms"
			);

			hideLoading();
			const pointcloud = loaded.pointcloud;
			const material = pointcloud.material;

			material.activeAttributeName = cfg.viewer.activeAttribute;
			material.pointSizeType = Potree.PointSizeType.ADAPTIVE;
			material.size = cfg.viewer.size;
			material.minSize = cfg.viewer.minSize;
			material.maxSize = cfg.viewer.maxSize;

			viewer.scene.addPointCloud(pointcloud);

			// VISUAL-001: frame the tight bounds ONLY after the cloud is
			// successfully attached.  One animation frame is the smallest
			// deterministic deferral so the scene has settled.
			const fit = function () {
				frameToPointcloud(viewer, pointcloud, metadata);
			};
			requestAnimationFrame(fit);
			addFitResetButton(fit);

			// P3A performance diagnostics + fixed two-state rendering policy.
			// Deliberately FAIL-OPEN: if the optional helper did not load the
			// viewer still works exactly as before.  Only the REQUIRED startup
			// helpers above (viewer_util) are fail-closed.
			if (
				window.p3aPerf &&
				typeof window.p3aPerf === "object" &&
				typeof window.p3aPerf.init === "function"
			) {
				window.p3aPerf.init({
					viewer: viewer,
					pointcloud: pointcloud,
					cfg: cfg,
					Potree: Potree,
					// The benchmark recorder reframes to the tight-bounds view at
					// the start of every run so each measurement starts from the
					// same framing without requiring a manual Fit/Reset.
					onFit: fit
				});
			}
		} catch (err) {
			showError(err && err.message ? err.message : String(err));
		}
	}

	start();
})();
