/**
 * TreeLineWarning P3A — Giro3D + COPC viewer (Candidate B) + V5A2/V5A3 review.
 *
 * Renders the SAME dense real-RGB ROI (982,504 pts) that Potree Candidate A
 * renders, but from a derived COPC file and with the Giro3D engine, so the
 * Owner can compare CloudCompare / Potree / Giro3D on identical data.
 *
 * Implemented: RGB point cloud, orbit (left drag), pan (right drag), zoom
 * (wheel), useful initial framing from the OCCUPIED point bounds, Fit / Reset,
 * click -> world XYZ (point picking), loading / visible error state, small
 * diagnostics, and — when the tower-review API is enabled on the same server —
 * the V5A2/V5A3 tower-review UI: candidate (P###) and confirmed-tower (T###)
 * overlay markers, a review panel with Accept / Reject / Adjust / Manual Add /
 * Refresh, and explicit interaction modes (NAVIGATE / ADJUST_CANDIDATE /
 * MANUAL_ADD).  The V5A3 PRIMARY workflow is ORDERED MANUAL TOWER SELECTION
 * (顺序选杆): the Owner clicks a real tower, the server runs bounded local RAW
 * membership segmentation, and the ACTUAL segmented RAW member points are
 * highlighted as a native Three.js points overlay (Tower Preview / 待确认杆塔)
 * before the Owner explicitly confirms the next canonical T###.
 *
 * Tower groups (区域 / Ordered Tower Groups, G001..G999): confirmed towers stay
 * persistently visible as lightweight T### markers; the panel shows the
 * grouped confirmed-tower list with per-tower [定位 / Locate], an active-group
 * selector (localStorage keyed by project), and a separate on-demand
 * "已确认杆塔成员 / Confirmed Tower Membership" Points overlay for the ONE
 * selected confirmed tower (replaced/cleared on selection change).  The
 * TowerGroupRegistry is authoritative; the UI group is session convenience only.
 *
 * The viewer is INTERACTION ONLY: markers are drawn from the project state
 * served by the local JSON API; detection/refinement geometry never comes from
 * the Giro3D-displayed points or COPC LOD nodes.
 *
 * The pure decision helpers (LAS-header bounds parsing, framing math,
 * click-coordinate plumbing, marker styling) are exposed on
 * `window.p3aGiro3dUtil` and use no browser APIs, so the Node test harness can
 * exercise them directly.
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

	/**
	 * Marker styling for a candidate review state.  Returns a plain object with
	 * background / foreground hex colors so the label stays legible while the
	 * owner orbits/zooms.  Explicit label/text is always shown in addition to the
	 * marker color so states are never only color-coded.
	 */
	function reviewMarkerColors(reviewState) {
		switch (reviewState) {
			case "ACCEPTED":
				return { bg: "#1b5e20", fg: "#ffffff" };
			case "REJECTED":
				return { bg: "#4a148c", fg: "#e0e0e0" };
			case "SUPERSEDED":
				return { bg: "#4a148c", fg: "#e0e0e0" };
			case "REFINEMENT_FAILED":
			case "RAW_REFINEMENT_FAILED":
				return { bg: "#b71c1c", fg: "#ffffff" };
			default:
				return { bg: "#ff6f00", fg: "#ffffff" };
		}
	}

	/** Confirmed-tower marker styling. */
	function towerMarkerColors() {
		return { bg: "#00695c", fg: "#ffffff" };
	}

	/**
	 * World-space scale (per axis) for a sprite label that should render at
	 * roughly ``desiredPx`` pixels on screen at the given camera distance.
	 */
	function spriteScaleForDistance(distance, desiredPx, fovDeg, canvasHeight) {
		const safe = Math.max(distance, 1e-3);
		const fovRad = ((fovDeg || 60) * Math.PI) / 180;
		const worldPerPixel = (2 * safe * Math.tan(fovRad / 2)) / (canvasHeight || 800);
		return desiredPx * worldPerPixel;
	}

	/** Short human label for one candidate projection row. */
	function candidateDisplayLabel(c) {
		const state = c.review_state || "PENDING_REVIEW";
		return c.candidate_id + " · " + c.origin + " · " + (c.confidence != null ? c.confidence.toFixed(2) : "?") + " · " + state;
	}

	/** Next tower label for the ordered panel (e.g. T003), or null. */
	function nextTowerLabel(state) {
		if (state && state.ordered && state.ordered.next_tower_id) {
			return state.ordered.next_tower_id;
		}
		return null;
	}

	/** Ordered confirmed-tower list line (T001 T002 …), or an empty marker. */
	function confirmedTowerLine(state) {
		if (!state || !state.ordered || !state.ordered.confirmed_tower_ids) {
			return "";
		}
		const ids = state.ordered.confirmed_tower_ids;
		return ids.length ? ids.join(" ") : "（无 / none）";
	}

	/** Resolve a group by id from the state, defaulting to the first group. */
	function resolveActiveGroup(state, groupId) {
		const groups = state && state.ordered && state.ordered.tower_groups
			? state.ordered.tower_groups : [];
		if (!groups.length) {
			return null;
		}
		if (groupId) {
			const found = groups.filter(function (g) { return g.group_id === groupId; });
			if (found.length) {
				return found[0];
			}
		}
		return groups[0];
	}

	/**
	 * Ordered group status panel lines for the active group.  Distinguishes the
	 * current group's next position (第 N 根) from the project-global next Tower
	 * ID (T###), so switching groups never confuses the two.
	 */
	function groupStatusLines(state, groupId) {
		if (!state || !state.ordered) {
			return null;
		}
		const group = resolveActiveGroup(state, groupId);
		if (!group) {
			return null;
		}
		const globalNext = state.ordered.next_tower_id || "?";
		return [
			"当前区域: " + group.group_id + " " + (group.display_name || ""),
			"本区域已确认: " + (group.tower_count != null ? group.tower_count : "?"),
			"本区域下一位置: 第 " + (group.next_group_position != null ? group.next_group_position : "?") + " 根",
			"项目下一 Tower ID: " + globalNext
		].join("\n");
	}

	/**
	 * Ordered preview panel text lines from a preview payload.  Returns null
	 * when there is no preview.  The member points are VIEWER-ONLY display
	 * samples; authoritative membership stays server-side.
	 */
	function orderedPreviewLines(preview) {
		if (!preview) {
			return null;
		}
		const b = preview.member_bounds_absolute_m || {};
		const mn = b.min_xyz_absolute_m || [];
		const mx = b.max_xyz_absolute_m || [];
		const q = preview.quality || {};
		const vert = q.median_verticality != null ? q.median_verticality.toFixed(2) : "?";
		const line = q.median_linearity != null ? q.median_linearity.toFixed(2) : "?";
		const support = q.ground_support_points != null ? q.ground_support_points : "?";
		const digest = q.membership_digest ? q.membership_digest.slice(0, 12) + "…" : "?";
		const fmt = function (v) { return v != null ? v.toFixed(2) : "?"; };
		return [
			"待确认杆塔 / Tower Preview",
			"member count: " + (preview.member_count != null ? preview.member_count : "?") +
				"（highlighted RAW member points）",
			"height: " + (preview.height_m != null ? preview.height_m.toFixed(2) + " m" : "?"),
			"bounds: min [" + mn.map(fmt).join(", ") + "]",
			"        max [" + mx.map(fmt).join(", ") + "]",
			"quality: vert=" + vert + " line=" + line + " support=" + support,
			"RAW status: " + (preview.refinement_ready || "?") + " / preview-ready",
			"member digest: " + digest
		].join("\n");
	}

	/**
	 * Compute the Viewer-only inspection region for one candidate.
	 *
	 * The region is a visualization envelope only — it never alters scientific
	 * geometry.  It uses the ACTUAL candidate bounds (XY centre + Z extent) when
	 * available so different candidates keep their real footprints instead of
	 * being exaggerated into one uniform 2.5 m x 25 m volume.  When no bounds
	 * exist it falls back to a compact inspection box that is explicitly labeled
	 * as a Viewer fallback.  The candidate is UNREFINED until RAW refinement;
	 * this is a "候选区域 / Candidate Region", not a segmented tower.
	 */
	function candidateRegion(candidate) {
		if (!candidate) {
			return null;
		}
		const p = candidate.proposal_xyz_absolute_m || candidate.candidate_xyz_absolute_m;
		if (!p || p.length !== 3) {
			return null;
		}
		const b = candidate.candidate_bounds_absolute_m;
		let cx, cy, baseZ, topZ, radius, source;
		if (b && b.min_xyz_absolute_m && b.max_xyz_absolute_m) {
			const mn = b.min_xyz_absolute_m;
			const mx = b.max_xyz_absolute_m;
			cx = (mn[0] + mx[0]) / 2;
			cy = (mn[1] + mx[1]) / 2;
			baseZ = mn[2];
			topZ = mx[2];
			// Honour the real horizontal footprint.  Only a tiny 0.5 m floor keeps
			// a near-point bounds visually readable; it never inflates a candidate
			// into a large uniform cylinder.
			radius = Math.max((mx[0] - mn[0]) / 2, (mx[1] - mn[1]) / 2, 0.5);
			source = "CANDIDATE_BOUNDS";
		} else {
			// Compact Viewer fallback only (no candidate bounds available).
			cx = p[0];
			cy = p[1];
			baseZ = p[2];
			topZ = p[2] + 18;
			radius = 1.2;
			source = "VIEWER_FALLBACK";
		}
		const height = Math.max(topZ - baseZ, 0.5);
		topZ = baseZ + height;
		return { cx: cx, cy: cy, baseZ: baseZ, topZ: topZ, radius: radius, height: height, source: source };
	}

	/**
	 * Z-up inspection-marker spec for one candidate region.
	 *
	 * The marker is a thin wireframe BOX (no large filled translucent surface).
	 * X and Y are the horizontal footprint and Z is the vertical extent, so a
	 * ``BoxGeometry(width, depth, height)`` maps directly onto the Z-up source
	 * frame with no rotation ambiguity (a raw ``CylinderGeometry`` would be
	 * Y-axis-aligned by default — the misleading horizontal tube seen by the
	 * Owner).  ``axis`` is always ``"Z"``.
	 */
	function selectionMarkerSpec(region) {
		return {
			kind: "wireframe-box",
			axis: "Z",
			centerX: region.cx,
			centerY: region.cy,
			centerZ: (region.baseZ + region.topZ) / 2,
			width: region.radius * 2,
			depth: region.radius * 2,
			height: region.height,
			baseZ: region.baseZ,
			topZ: region.topZ
		};
	}

	/**
	 * Attach an overlay group to the Giro3D instance that is actually served by
	 * the pinned bundle.  The pinned Instance exposes ``threeObjects`` (the group
	 * reserved for native Three.js objects) and ``scene``; ``instance.view`` does
	 * NOT expose ``scene``/``threeObjects`` in this bundle.  Returns the name of
	 * the target that accepted the group, and throws a clear error when no add
	 * target exists so the panel can fail visibly instead of hanging.
	 */
	function attachOverlayGroup(host, group) {
		if (host && host.threeObjects && typeof host.threeObjects.add === "function") {
			host.threeObjects.add(group);
			return "threeObjects";
		}
		if (host && host.scene && typeof host.scene.add === "function") {
			host.scene.add(group);
			return "scene";
		}
		throw new Error(
			"Giro3D instance exposes no scene/threeObjects overlay .add target " +
			"(host threeObjects=" + (host && host.threeObjects ? "present" : "absent") +
			", scene=" + (host && host.scene ? "present" : "absent") + ")"
		);
	}

	window.p3aGiro3dUtil = {
		parseLasHeaderBounds: parseLasHeaderBounds,
		computeFitCamera: computeFitCamera,
		pickResultToXyz: pickResultToXyz,
		reviewMarkerColors: reviewMarkerColors,
		towerMarkerColors: towerMarkerColors,
		spriteScaleForDistance: spriteScaleForDistance,
		candidateDisplayLabel: candidateDisplayLabel,
		candidateRegion: candidateRegion,
		selectionMarkerSpec: selectionMarkerSpec,
		attachOverlayGroup: attachOverlayGroup,
		nextTowerLabel: nextTowerLabel,
		confirmedTowerLine: confirmedTowerLine,
		orderedPreviewLines: orderedPreviewLines,
		resolveActiveGroup: resolveActiveGroup,
		groupStatusLines: groupStatusLines
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
			"last XYZ: " + fmtVec(viewer.lastPick),
			"mode: " + (viewer.review ? viewer.review.mode() : "navigate")
		];
		statusEl.textContent = lines.join("\n");
	}

	// -----------------------------------------------------------------------
	// V5A2/V5A3 tower-review controller (interaction-only; API same-origin)
	// -----------------------------------------------------------------------

	function makeTowerReview(viewer) {
		const apiBase = "";
		let mode = "navigate"; // navigate | adjust | manual
		let selected = null; // { set, candidate } refs
		let adjustedSeed = null; // {x,y,z} from Adjust pick
		let lastState = null; // most recent ordered review state projection
		// Ordered manual tower selection (V5A3) — the PRIMARY workflow.
		let orderedMode = "navigate"; // navigate | select
		let previewToken = null;
		let preview = null; // preview metadata payload
		// Tower groups (V5A3): active UI group is a session convenience stored in
		// localStorage keyed by project; the TowerGroupRegistry is authoritative.
		let activeGroupId = null;
		let confirmedMembership = null; // { tower_id, preview } for reinspection
		const labelGroup = new THREE.Group();
		labelGroup.name = "tower-review-overlays";
		labelGroup.renderOrder = 999;
		// Selected-candidate beacon / ring / large label live in their own group so
		// they stay visually distinct and scale independently of the list markers.
		const selectionGroup = new THREE.Group();
		selectionGroup.name = "tower-review-selection";
		selectionGroup.renderOrder = 1000;
		// The ACTUAL segmented RAW member-point preview (V5A3) is a native
		// Three.js Points overlay — the primary visual cue for the Owner.  It
		// renders the VIEWER-ONLY deterministic display sample from the server;
		// the authoritative exact membership stays server-side.
		const previewPointsGroup = new THREE.Group();
		previewPointsGroup.name = "tower-review-membership-preview";
		previewPointsGroup.renderOrder = 1001;
		// Confirmed-tower membership reinspection overlay (V5A3): separate from the
		// pending "待确认杆塔" preview; only ONE selected confirmed tower loads at a
		// time and it is replaced/cleared when another tower is selected.
		const confirmedPointsGroup = new THREE.Group();
		confirmedPointsGroup.name = "tower-review-confirmed-membership";
		confirmedPointsGroup.renderOrder = 1001;
		// The pinned Giro3D exposes native-object insertion on the Instance
		// (threeObjects / scene), not on instance.view.
		attachOverlayGroup(viewer.instance, labelGroup);
		attachOverlayGroup(viewer.instance, selectionGroup);
		attachOverlayGroup(viewer.instance, previewPointsGroup);
		attachOverlayGroup(viewer.instance, confirmedPointsGroup);

		function api(path, options) {
			return fetch(apiBase + path, options).then(function (resp) {
				return resp.json().catch(function () {
					return { ok: false, error: { code: "BAD_RESPONSE", message: "non-JSON response" } };
				}).then(function (payload) {
					if (resp.status === 404 && payload && payload.error && payload.error.code === "API_DISABLED") {
						throw new Error("tower-review API is disabled on this server");
					}
					if (!resp.ok || !payload.ok) {
						const err = payload && payload.error ? payload.error : { code: "HTTP_" + resp.status, message: "request failed" };
						throw new Error((err.code || "ERROR") + ": " + (err.message || ""));
					}
					return payload;
				});
			});
		}

		function postJson(path, body) {
			return api(path, {
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify(body || {})
			});
		}

		function makeLabelSprite(text, bg, fg, shape, opts) {
			const options = opts || {};
			const canvas = document.createElement("canvas");
			canvas.width = 256;
			canvas.height = 128;
			const ctx = canvas.getContext("2d");
			ctx.font = "bold 44px sans-serif";
			const w = Math.max(72, ctx.measureText(text).width + 96);
			canvas.width = Math.ceil(w);
			const cx = canvas.getContext("2d");
			cx.fillStyle = bg;
			if (shape === "tower") {
				cx.fillRect(8, 24, canvas.width - 16, 80);
			} else {
				cx.beginPath();
				cx.roundRect ? cx.roundRect(8, 24, canvas.width - 16, 80, 16) : cx.fillRect(8, 24, canvas.width - 16, 80);
				cx.fill();
			}
			cx.font = "bold 44px sans-serif";
			cx.fillStyle = fg;
			cx.textAlign = "left";
			cx.textBaseline = "middle";
			cx.fillText(text, 24, 64);
			const texture = new THREE.CanvasTexture(canvas);
			texture.minFilter = THREE.LinearFilter;
			const material = new THREE.SpriteMaterial({
				map: texture,
				transparent: true,
				depthTest: false,
				sizeAttenuation: true
			});
			const sprite = new THREE.Sprite(material);
			sprite.scale.set(canvas.width / 32, canvas.height / 32, 1);
			sprite.userData.isLabel = true;
			sprite.userData.labelPx = options.labelPx || 26;
			return sprite;
		}

		/** Keep a label at a roughly constant on-screen size (never microscopic). */
		function scaleLabelSprite(sprite, camera) {
			if (!camera || !sprite) {
				return;
			}
			const distance = camera.position.distanceTo(sprite.position);
			const fov = camera.fov || 60;
			const height = window.innerHeight || 800;
			const desiredPx = sprite.userData && sprite.userData.labelPx ? sprite.userData.labelPx : 26;
			const dimFactor = sprite.userData && sprite.userData.dimmed ? 0.55 : 1.0;
			const scale = spriteScaleForDistance(distance, desiredPx * dimFactor, fov, height);
			const aspect = sprite.scale.x / sprite.scale.y;
			sprite.scale.set(scale * aspect, scale, 1);
		}

		function drawOverlays(state) {
			while (labelGroup.children.length) {
				labelGroup.remove(labelGroup.children[0]);
			}
			if (!state || !state.candidate_sets) {
				return;
			}
			const camera = viewer.instance.view.camera;
			state.candidate_sets.forEach(function (cs) {
				(cs.candidates || []).forEach(function (c) {
					const p = c.proposal_xyz_absolute_m || c.candidate_xyz_absolute_m;
					if (!p) {
						return;
					}
					const cols = reviewMarkerColors(
						c.refinement_state === "RAW_REFINEMENT_FAILED" ? "RAW_REFINEMENT_FAILED" : c.review_state
					);
					const sprite = makeLabelSprite(c.candidate_id, cols.bg, cols.fg, "candidate");
					sprite.position.set(p[0], p[1], p[2]);
					sprite.userData = Object.assign({}, sprite.userData, {
						kind: "candidate", set_id: cs.candidate_set_id, candidate_id: c.candidate_id
					});
					labelGroup.add(sprite);
					scaleLabelSprite(sprite, camera);
				});
			});
			(state.confirmed_towers || []).forEach(function (t) {
				const p = t.display_xyz_absolute_m || t.top_xyz_absolute_m;
				if (!p) {
					return;
				}
				const cols = towerMarkerColors();
				const sprite = makeLabelSprite(t.tower_id, cols.bg, cols.fg, "tower");
				sprite.position.set(p[0], p[1], p[2]);
				sprite.userData = Object.assign({}, sprite.userData, {
					kind: "tower", tower_id: t.tower_id
				});
				labelGroup.add(sprite);
				scaleLabelSprite(sprite, camera);
			});
			applySelectionVisuals();
		}

		/** A flat circle LineLoop (crosshair) at (x, y, z). */
		function makeCircleLoop(x, y, z, radius, color, opacity) {
			const points = [];
			const segments = 40;
			for (let i = 0; i < segments; i++) {
				const a = (i / segments) * Math.PI * 2;
				points.push(Math.cos(a) * radius, Math.sin(a) * radius, 0);
			}
			const geo = new THREE.BufferGeometry();
			geo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(points), 3));
			const loop = new THREE.LineLoop(geo, new THREE.LineBasicMaterial({
				color: color,
				transparent: true,
				opacity: opacity
			}));
			loop.position.set(x, y, z);
			return loop;
		}

		/**
		 * Rebuild the selected-candidate overlay: a thin Z-up wireframe box
		 * (X/Y = horizontal footprint, Z = vertical extent), a base-point
		 * crosshair ring as the primary location cue, a secondary top cap ring,
		 * a vertical leader line, and a large screen-readable label.  There is no
		 * large filled translucent surface, so the REAL point cloud stays visible.
		 * All added to ``selectionGroup`` on the instance.
		 */
		function rebuildSelection() {
			while (selectionGroup.children.length) {
				selectionGroup.remove(selectionGroup.children[0]);
			}
			if (!selected) {
				return;
			}
			const region = candidateRegion(selected.candidate);
			if (!region) {
				return;
			}
			const spec = selectionMarkerSpec(region);
			const cx = region.cx;
			const cy = region.cy;
			const baseZ = region.baseZ;
			const topZ = region.topZ;

			// Thin Z-up wireframe inspection box.  BoxGeometry(width, depth, height)
			// maps X=width, Y=depth, Z=height directly onto the Z-up source frame.
			const boxEdges = new THREE.EdgesGeometry(
				new THREE.BoxGeometry(spec.width, spec.depth, spec.height)
			);
			const box = new THREE.LineSegments(
				boxEdges,
				new THREE.LineBasicMaterial({ color: 0x4fc3f7, transparent: true, opacity: 0.9 })
			);
			box.position.set(spec.centerX, spec.centerY, spec.centerZ);
			box.userData.candidateRegion = true;
			selectionGroup.add(box);

			// Base-point crosshair ring — the primary location cue at the candidate
			// XYZ/base.  Sized from the real footprint (never a huge default ring).
			const baseRing = makeCircleLoop(cx, cy, baseZ + 0.2, Math.max(region.radius * 1.4, 1.0), 0xffd54f, 1.0);
			baseRing.userData.baseMarker = true;
			selectionGroup.add(baseRing);

			// Secondary top cap crosshair at the top of the inspection extent.
			selectionGroup.add(makeCircleLoop(cx, cy, topZ, Math.max(region.radius * 0.9, 0.5), 0xffd54f, 0.6));

			// Vertical Z-up leader line from the marker top up to the big label.
			const leaderGeo = new THREE.BufferGeometry();
			leaderGeo.setAttribute("position", new THREE.BufferAttribute(new Float32Array([
				cx, cy, topZ + 0.5,
				cx, cy, topZ + 2.5
			]), 3));
			const leader = new THREE.Line(
				leaderGeo,
				new THREE.LineBasicMaterial({ color: 0xffd54f, transparent: true, opacity: 0.9, depthTest: false })
			);
			leader.userData.leader = true;
			selectionGroup.add(leader);

			const label = makeLabelSprite(selected.candidate.candidate_id, "#ff3d81", "#ffffff", "candidate", { labelPx: 46 });
			label.position.set(cx, cy, topZ + 3);
			label.userData.selectedLabel = true;
			selectionGroup.add(label);
			scaleLabelSprite(label, viewer.instance.view.camera);
		}

		/**
		 * Make the selected candidate visually primary and the others secondary
		 * (reduced opacity + smaller scale) without removing them.  T### markers
		 * always stay at full visibility so they remain distinguishable.
		 */
		function applySelectionVisuals() {
			labelGroup.children.forEach(function (sprite) {
				const kind = sprite.userData && sprite.userData.kind;
				const key = sprite.userData && sprite.userData.set_id != null
					? sprite.userData.set_id + "/" + sprite.userData.candidate_id
					: null;
				const isSelected = selected && key !== null && key === selected.key;
				const dim = !isSelected && !(kind === "tower") && key !== null && selected != null;
				sprite.userData.dimmed = dim;
				if (sprite.material) {
					sprite.material.opacity = dim ? 0.30 : 1.0;
				}
				scaleLabelSprite(sprite, viewer.instance.view.camera);
			});
			rebuildSelection();
		}

		function updateDetails() {
			const el = document.getElementById("trp_details");
			if (!el) {
				return;
			}
			if (!selected) {
				el.textContent = "Selected: —";
				return;
			}
			const c = selected.candidate;
			const p = c.proposal_xyz_absolute_m || c.candidate_xyz_absolute_m || [];
			const xyz = p.length === 3
				? "Candidate XYZ (X, Y, Z):\n  " + p[0].toFixed(2) + ", " + p[1].toFixed(2) + ", " + p[2].toFixed(2)
				: "Candidate XYZ: unavailable";
			const b = c.candidate_bounds_absolute_m;
			const boundsLine = (b && b.min_xyz_absolute_m && b.max_xyz_absolute_m)
				? "\nCandidate bounds:\n  min [" + b.min_xyz_absolute_m.map(function (v) { return v.toFixed(2); }).join(", ") +
				  "]\n  max [" + b.max_xyz_absolute_m.map(function (v) { return v.toFixed(2); }).join(", ") + "]"
				: "\nCandidate bounds: — (no bounds)";
			const region = candidateRegion(c);
			const regionLine = region
				? "\n候选区域 / Candidate Region (localization only):\n  radius=" + region.radius.toFixed(2) +
				  "m height=" + region.height.toFixed(1) + "m (" +
				  (region.source === "CANDIDATE_BOUNDS"
					? "candidate bounds"
					: "Viewer fallback inspection volume") +
				  ")"
				: "\n候选区域 / Candidate Region: —";
			el.textContent = "Selected: " + c.candidate_id +
				"\nOrigin: " + (c.origin || "?") +
				"\nConfidence: " + (c.confidence != null ? c.confidence.toFixed(3) : "?") +
				"\nState: " + (c.review_state || "?") +
				"\n" + xyz + boundsLine + regionLine +
				"\n\n候选标记仅用于定位，未进行杆塔分割\nCandidate marker is localization only; tower is not yet segmented/refined.";
		}

		/** Move the camera to a closer oblique view framing the candidate region. */
		function focusOnSelected() {
			if (!selected) {
				return;
			}
			const region = candidateRegion(selected.candidate);
			if (!region) {
				return;
			}
			// Frame the ACTUAL candidate region tightly (it now tracks the real
			// candidate bounds) so the Owner can judge the local RAW structure —
			// tower / pole / tree / streetlight / vegetation.  Never frame around a
			// large artificial inspection volume.
			const boxHeight = Math.max(region.height / 0.45, 18);
			const boxHalf = Math.max(region.radius * 1.7, 5);
			const box = {
				min: { x: region.cx - boxHalf, y: region.cy - boxHalf, z: region.baseZ - 3 },
				max: { x: region.cx + boxHalf, y: region.cy + boxHalf, z: region.baseZ + boxHeight }
			};
			const frame = computeFitCamera(box.min, box.max, 60);
			const cam = viewer.instance.view.camera;
			cam.position.set(frame.position.x, frame.position.y, frame.position.z);
			cam.lookAt(frame.target.x, frame.target.y, frame.target.z);
			viewer.controls.target.set(frame.target.x, frame.target.y, frame.target.z);
			viewer.controls.update();
			viewer.instance.notifyChange(cam);
		}

		/** Re-resolve the selected candidate against a fresh state projection. */
		function refreshSelected(state) {
			if (!selected || !state) {
				return;
			}
			const resolved = selectCandidate(state, selected.key);
			if (resolved) {
				selected = resolved;
			}
		}

		/** Keep every overlay label at a stable on-screen size for the camera. */
		function scaleLabels() {
			const cam = viewer.instance.view.camera;
			labelGroup.children.forEach(function (sprite) {
				if (sprite.userData && sprite.userData.isLabel) {
					scaleLabelSprite(sprite, cam);
				}
			});
			selectionGroup.children.forEach(function (child) {
				if (child.userData && child.userData.isLabel) {
					scaleLabelSprite(child, cam);
				}
			});
		}

		function setMode(next, hint) {
			mode = next;
			const el = document.getElementById("trp_mode");
			if (el) {
				el.textContent = "mode: " + next.toUpperCase() + (hint ? " — " + hint : "");
			}
			const status = document.getElementById("trp_status");
			if (hint && status) {
				status.textContent = hint;
			}
		}

		function panelStatus(text, ok) {
			const el = document.getElementById("trp_status");
			if (el) {
				el.textContent = text || "";
				el.style.color = ok === false ? "#ff8a80" : "#a5d6a7";
			}
		}

		function renderPanel(state) {
			const listEl = document.getElementById("trp_list");
			if (!listEl) {
				return;
			}
			const accepted = document.getElementById("trp_accept");
			const rejected = document.getElementById("trp_reject");
			if (accepted) {
				accepted.disabled = !selected;
			}
			if (rejected) {
				rejected.disabled = !selected;
			}
			let html = "";
			(state.candidate_sets || []).forEach(function (cs) {
				(cs.candidates || []).forEach(function (c) {
					const key = cs.candidate_set_id + "/" + c.candidate_id;
					const sel = selected && selected.key === key ? " class='sel'" : "";
					const label = candidateDisplayLabel(c);
					const fail = c.refinement_state === "RAW_REFINEMENT_FAILED" ? " ⚠ fail" : "";
					html += "<div data-key='" + key + "'" + sel + ">" +
						"<span class='pill'>" + c.candidate_id + "</span>" +
						"<span class='info'>" + label + fail + "</span></div>";
				});
			});
			if (!html) {
				html = "<div class='empty'>no candidates</div>";
			}
			listEl.innerHTML = html;
			Array.prototype.forEach.call(listEl.querySelectorAll("[data-key]"), function (node) {
				node.addEventListener("click", function () {
					const key = node.getAttribute("data-key");
					selected = selectCandidate(state, key);
					adjustedSeed = null;
					renderPanel(state);
					applySelectionVisuals();
					updateDetails();
					focusOnSelected();
					setMode("navigate", "located " + (selected ? selected.candidate.candidate_id : ""));
				});
			});
		}

		function selectCandidate(state, key) {
			const parts = key.split("/");
			const set = (state.candidate_sets || []).find(function (s) {
				return s.candidate_set_id === parts[0];
			});
			if (!set) {
				return null;
			}
			const cand = (set.candidates || []).find(function (c) {
				return c.candidate_id === parts[1];
			});
			if (!cand) {
				return null;
			}
			return { key: key, set: set, candidate: cand };
		}

		function refreshState() {
			return api("/api/tower-review/state", { cache: "no-store" }).then(function (payload) {
				const state = payload.state;
				lastState = state;
				if (!activeGroupId) {
					const saved = loadActiveGroup(state.project_id);
					const current = resolveActiveGroup(state, saved);
					if (current) {
						activeGroupId = current.group_id;
					}
				}
				drawOverlays(state);
				refreshSelected(state);
				renderPanel(state);
				renderOrderedPanel(state);
				renderGroupBar(state);
				applySelectionVisuals();
				updateDetails();
				panelStatus("project " + state.project_id.slice(0, 8) + " · " + state.confirmed_towers.length + " T###");
				return state;
			}).catch(function (err) {
				panelStatus("review API: " + err.message, false);
				return null;
			});
		}

		function refreshAfter(label, promise) {
			panelStatus("working: " + label);
			return promise.then(function () {
				return refreshState();
			}).catch(function (err) {
				panelStatus(label + " failed: " + err.message, false);
				return null;
			});
		}

		// ---------------------------------------------------------------- ordered
		function clearPreviewPoints() {
			while (previewPointsGroup.children.length) {
				const child = previewPointsGroup.children[0];
				if (child.geometry) {
					child.geometry.dispose();
				}
				if (child.material) {
					child.material.dispose();
				}
				previewPointsGroup.remove(child);
			}
		}

		/** Render the ACTUAL segmented RAW member-point display sample. */
		function renderPreviewPoints(sample) {
			clearPreviewPoints();
			if (!sample || !sample.length) {
				return;
			}
			const positions = new Float32Array(sample.length * 3);
			for (let i = 0; i < sample.length; i++) {
				positions[i * 3] = sample[i][0];
				positions[i * 3 + 1] = sample[i][1];
				positions[i * 3 + 2] = sample[i][2];
			}
			const geo = new THREE.BufferGeometry();
			geo.setAttribute("position", new THREE.BufferAttribute(positions, 3));
			const material = new THREE.PointsMaterial({
				color: 0xffd54f,
				size: 0.6,
				sizeAttenuation: true,
				transparent: true,
				opacity: 1.0,
				depthTest: true
			});
			const points = new THREE.Points(geo, material);
			points.userData.membershipPreview = true;
			previewPointsGroup.add(points);
			viewer.instance.notifyChange(viewer.instance.view.camera);
		}

		/** Frame the camera tightly around the preview member points. */
		function focusPreview() {
			if (!preview || !preview.preview_sample || !preview.preview_sample.length) {
				return;
			}
			let minX = Infinity, minY = Infinity, minZ = Infinity;
			let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
			preview.preview_sample.forEach(function (p) {
				if (p[0] < minX) { minX = p[0]; }
				if (p[0] > maxX) { maxX = p[0]; }
				if (p[1] < minY) { minY = p[1]; }
				if (p[1] > maxY) { maxY = p[1]; }
				if (p[2] < minZ) { minZ = p[2]; }
				if (p[2] > maxZ) { maxZ = p[2]; }
			});
			const frame = computeFitCamera(
				{ x: minX, y: minY, z: minZ }, { x: maxX, y: maxY, z: maxZ }, 60
			);
			const cam = viewer.instance.view.camera;
			cam.position.set(frame.position.x, frame.position.y, frame.position.z);
			cam.lookAt(frame.target.x, frame.target.y, frame.target.z);
			viewer.controls.target.set(frame.target.x, frame.target.y, frame.target.z);
			viewer.controls.update();
			viewer.instance.notifyChange(cam);
		}

		function clearPreview() {
			clearPreviewPoints();
			previewToken = null;
			preview = null;
			renderOrderedPanel(lastState);
			renderGroupBar(lastState);
		}

		/** Panel: next-tower label, preview details, and ordered button states. */
		function renderOrderedPanel(state) {
			const nextEl = document.getElementById("trp_ordered_next");
			const previewEl = document.getElementById("trp_ordered_preview");
			const confirmBtn = document.getElementById("trp_order_confirm");
			const reselectBtn = document.getElementById("trp_order_reselect");
			const cancelBtn = document.getElementById("trp_order_cancel");
			const selectBtn = document.getElementById("trp_order_select");
			const ordered = state && state.ordered ? state.ordered : null;
			const next = nextTowerLabel(state);
			const confirmed = ordered ? confirmedTowerLine(state) : "";
			if (nextEl) {
				nextEl.textContent = "已确认: " + confirmed + "\n下一根: " + (next || "?") +
					"（确认顺序即线路顺序 / confirmation order = corridor order）";
			}
			const hasPreview = previewToken !== null && preview !== null;
			if (previewEl) {
				previewEl.textContent = hasPreview ? orderedPreviewLines(preview) : "待确认杆塔 / Tower Preview: —";
			}
			if (confirmBtn) {
				confirmBtn.disabled = !hasPreview;
				confirmBtn.textContent = "确认 " + (next || "T###");
			}
			if (reselectBtn) {
				reselectBtn.disabled = !hasPreview;
			}
			if (cancelBtn) {
				cancelBtn.disabled = !hasPreview;
			}
			if (selectBtn) {
				selectBtn.textContent = orderedMode === "select" ? "点击杆塔位置…（click tower）" : "选择下一根杆塔";
			}
		}

		// ---------------------------------------------------------------- groups
		function _pointsGroupStorage() {
			try { return window.localStorage; } catch (e) { return null; }
		}

		function persistActiveGroup(projectId, groupId) {
			const store = _pointsGroupStorage();
			if (store && projectId && groupId) {
				store.setItem("p3a_ordered_active_group_" + projectId, groupId);
			}
		}

		function loadActiveGroup(projectId) {
			const store = _pointsGroupStorage();
			if (store && projectId) {
				return store.getItem("p3a_ordered_active_group_" + projectId) || null;
			}
			return null;
		}

		/** Render points from a sample into a target overlay group. */
		function _renderPointsInto(group, sample, color, size) {
			while (group.children.length) {
				const child = group.children[0];
				if (child.geometry) { child.geometry.dispose(); }
				if (child.material) { child.material.dispose(); }
				group.remove(child);
			}
			if (!sample || !sample.length) {
				return;
			}
			const positions = new Float32Array(sample.length * 3);
			for (let i = 0; i < sample.length; i++) {
				positions[i * 3] = sample[i][0];
				positions[i * 3 + 1] = sample[i][1];
				positions[i * 3 + 2] = sample[i][2];
			}
			const geo = new THREE.BufferGeometry();
			geo.setAttribute("position", new THREE.BufferAttribute(positions, 3));
			const material = new THREE.PointsMaterial({
				color: color || 0x81d4fa,
				size: size || 0.55,
				sizeAttenuation: true,
				transparent: true,
				opacity: 1.0,
				depthTest: true
			});
			const points = new THREE.Points(geo, material);
			points.userData.confirmedMembership = true;
			group.add(points);
			viewer.instance.notifyChange(viewer.instance.view.camera);
		}

		function clearConfirmedMembership() {
			confirmedMembership = null;
			_renderPointsInto(confirmedPointsGroup, null, 0x81d4fa, 0.55);
			const el = document.getElementById("trp_confirmed_membership");
			if (el) {
				el.textContent = "已确认杆塔成员 / Confirmed Tower Membership: —";
			}
		}

		/** Focus the camera on a confirmed tower + load its membership reinspection. */
		function locateConfirmedTower(towerRow) {
			if (!towerRow) {
				return;
			}
			const p = towerRow.display_xyz_absolute_m || [];
			if (p.length === 3) {
				const height = Math.max(towerRow.height_m || 10, 8);
				const box = {
					min: { x: p[0] - 4, y: p[1] - 4, z: p[2] - 3 },
					max: { x: p[0] + 4, y: p[1] + 4, z: p[2] + height }
				};
				const frame = computeFitCamera(box.min, box.max, 60);
				const cam = viewer.instance.view.camera;
				cam.position.set(frame.position.x, frame.position.y, frame.position.z);
				cam.lookAt(frame.target.x, frame.target.y, frame.target.z);
				viewer.controls.target.set(frame.target.x, frame.target.y, frame.target.z);
				viewer.controls.update();
				viewer.instance.notifyChange(cam);
			}
			panelStatus("定位 " + towerRow.tower_id + " — loading confirmed membership…");
			return api("/api/tower-review/towers/" + encodeURIComponent(towerRow.tower_id) + "/membership-preview", {
				cache: "no-store"
			}).then(function (payload) {
				confirmedMembership = { tower_id: towerRow.tower_id, preview: payload };
				_renderPointsInto(confirmedPointsGroup, payload.preview_sample || [], 0x81d4fa, 0.55);
				const groups = (payload.group_ids || []).join(", ") || "—";
				const el = document.getElementById("trp_confirmed_membership");
				if (el) {
					const b = payload.member_bounds_absolute_m || {};
					const mn = (b.min_xyz_absolute_m || []).map(function (v) { return v.toFixed(2); }).join(", ");
					const mx = (b.max_xyz_absolute_m || []).map(function (v) { return v.toFixed(2); }).join(", ");
					el.textContent = "已确认杆塔成员 / Confirmed Tower Membership: " + towerRow.tower_id +
						"\nGroup: " + groups +
						"\nmember count: " + (payload.member_count != null ? payload.member_count : "?") +
						"（VIEWER-ONLY capped sample）" +
						"\nheight: " + (payload.height_m != null ? payload.height_m.toFixed(2) + " m" : "?") +
						"\nbounds: [" + mn + "] … [" + mx + "]";
				}
				panelStatus("已确认杆塔成员 / confirmed membership: " + towerRow.tower_id);
			}).catch(function (err) {
				// The camera Locate already happened above, so locating a confirmed
				// tower always works even when its membership preview cannot load.
				// A legitimately-unavailable historical tower (no membership artifact)
				// and a transient network failure are both reported calmly — never a
				// scary generic "Failed to fetch".
				const message = String(err && err.message ? err.message : "unknown");
				const unavailable = message.indexOf("MEMBERSHIP_UNAVAILABLE") !== -1 ||
					message.indexOf("MEMBERSHIP_ARTIFACT_INVALID") !== -1;
				const el = document.getElementById("trp_confirmed_membership");
				if (el) {
					el.textContent = unavailable
						? "已确认杆塔成员 / Confirmed Tower Membership: " + towerRow.tower_id +
							"\n（该杆塔暂无成员数据 / membership preview unavailable for this historical tower）"
						: "已确认杆塔成员 / Confirmed Tower Membership: " + towerRow.tower_id +
							"\n（无法加载成员预览 / cannot load membership preview — 定位仍可用 / Locate still works）";
				}
				// Clear any stale confirmed-membership overlay.
				_renderPointsInto(confirmedPointsGroup, null, 0x81d4fa, 0.55);
				panelStatus(
					unavailable
						? "该杆塔暂无成员数据 / membership preview unavailable"
						: "无法加载成员预览 / membership preview could not load — Locate still works"
				);
			});
		}

		/** Render the group bar: current-group select, create button, grouped list. */
		function renderGroupBar(state) {
			const groups = state && state.ordered && state.ordered.tower_groups
				? state.ordered.tower_groups : [];
			const hasPreview = previewToken !== null && preview !== null;
			const selectEl = document.getElementById("trp_group_select");
			const createBtn = document.getElementById("trp_group_create");
			const statusEl = document.getElementById("trp_group_status");
			if (selectEl) {
				const current = resolveActiveGroup(state, activeGroupId);
				activeGroupId = current ? current.group_id : null;
				selectEl.innerHTML = "";
				groups.forEach(function (g) {
					const opt = document.createElement("option");
					opt.value = g.group_id;
					opt.textContent = g.group_id + " " + (g.display_name || "");
					if (current && g.group_id === current.group_id) {
						opt.selected = true;
					}
					selectEl.appendChild(opt);
				});
				selectEl.disabled = hasPreview;
				if (selectEl.onchange === null) {
					selectEl.onchange = function () {
						switchGroup(selectEl.value);
					};
				}
			}
			if (createBtn) {
				createBtn.disabled = hasPreview;
			}
			if (statusEl) {
				statusEl.textContent = hasPreview && preview
					? "待确认杆塔属于: " + preview.group_id + " " + (preview.group_display_name || "") +
						"\n请先确认、重新选择或取消当前待确认杆塔"
					: (groupStatusLines(state, activeGroupId) || "");
			}
			renderGroupList(state);
		}

		/** Grouped confirmed-tower list with per-tower [定位] actions. */
		function renderGroupList(state) {
			const listEl = document.getElementById("trp_group_list");
			if (!listEl) {
				return;
			}
			const groups = state && state.ordered && state.ordered.tower_groups
				? state.ordered.tower_groups : [];
			if (!groups.length) {
				listEl.innerHTML = "<div class='empty'>no groups yet — create one</div>";
				return;
			}
			let html = "";
			groups.forEach(function (g) {
				const open = (activeGroupId === g.group_id) || (groups.length === 1);
				html += "<div class='ghead' data-group='" + g.group_id + "'>" +
					(g.group_id + " " + (g.display_name || "")) +
					" <span style='opacity:0.7'>" + (g.tower_count != null ? g.tower_count : 0) + " 根</span></div>";
				if (open) {
					(g.towers || []).forEach(function (t) {
						html += "<div class='gtower'>" +
							"<span class='tid'>" + t.tower_id + "</span>" +
							"<span style='opacity:0.6'>h=" + (t.height_m != null ? t.height_m.toFixed(1) : "?") + "m</span>" +
							"<button data-locate='" + t.tower_id + "' type='button'>定位 / Locate</button>" +
							"</div>";
					});
					if (!(g.towers || []).length) {
						html += "<div class='gtower empty'>（无确认杆塔 / none）</div>";
					}
				}
			});
			listEl.innerHTML = html;
			Array.prototype.forEach.call(listEl.querySelectorAll("[data-locate]"), function (btn) {
				btn.addEventListener("click", function () {
					const tid = btn.getAttribute("data-locate");
					const row = (groups || []).reduce(function (acc, g) {
						return acc || (g.towers || []).filter(function (t) { return t.tower_id === tid; })[0] || null;
					}, null);
					locateConfirmedTower(row);
				});
			});
			Array.prototype.forEach.call(listEl.querySelectorAll(".ghead"), function (head) {
				head.addEventListener("click", function () {
					if (previewToken !== null) {
						panelStatus("请先确认、重新选择或取消当前待确认杆塔", false);
						return;
					}
					switchGroup(head.getAttribute("data-group"));
				});
			});
		}

		function switchGroup(groupId) {
			if (!groupId) {
				return;
			}
			if (previewToken !== null) {
				panelStatus("请先确认、重新选择或取消当前待确认杆塔", false);
				return;
			}
			activeGroupId = groupId;
			persistActiveGroup(lastState ? lastState.project_id : null, groupId);
			clearConfirmedMembership();
			renderGroupBar(lastState);
			renderOrderedPanel(lastState);
			panelStatus("当前区域: " + groupId);
		}

		function beginCreateGroup() {
			if (previewToken !== null) {
				panelStatus("请先确认、重新选择或取消当前待确认杆塔", false);
				return;
			}
			const row = document.getElementById("trp_group_create_row");
			const nameEl = document.getElementById("trp_group_name");
			if (row) { row.style.display = "flex"; }
			if (nameEl) { nameEl.value = ""; nameEl.focus(); }
		}

		function submitCreateGroup() {
			const nameEl = document.getElementById("trp_group_name");
			const name = nameEl ? nameEl.value.trim() : "";
			if (!name) {
				panelStatus("请输入区域名称", false);
				return;
			}
			const row = document.getElementById("trp_group_create_row");
			if (row) { row.style.display = "none"; }
			panelStatus("创建区域…");
			return postJson("/api/tower-review/groups", {
				display_name: name,
				operator_identity: "owner"
			}).then(function (payload) {
				const created = payload.group && payload.group.group_id;
				lastState = payload.state || lastState;
				if (created) {
					activeGroupId = created;
					persistActiveGroup(lastState ? lastState.project_id : null, created);
				}
				renderGroupBar(lastState);
				renderOrderedPanel(lastState);
				panelStatus("已创建 " + created);
				return created;
			}).catch(function (err) {
				panelStatus("create group failed: " + err.message, false);
				return null;
			});
		}

		function cancelCreateGroup() {
			const row = document.getElementById("trp_group_create_row");
			if (row) { row.style.display = "none"; }
			panelStatus("cancelled");
		}

		function setOrderedMode(next, hint) {
			orderedMode = next;
			setMode(orderedMode, hint || "");
			renderOrderedPanel(lastState);
		}

		function beginOrderSelect() {
			clearPreview();
			setOrderedMode("select", "单击杆塔位置（click the real tower position）");
		}

		function doOrderSelect(xyz) {
			if (!activeGroupId) {
				panelStatus("请先选择区域 / select a group first", false);
				setOrderedMode("navigate", "");
				return Promise.resolve(null);
			}
			panelStatus("正在局部分割…");
			return postJson("/api/tower-review/ordered/select", {
				xyz: [xyz.x, xyz.y, xyz.z],
				group_id: activeGroupId,
				operator_identity: "owner"
			}).then(function (payload) {
				if (!payload.success) {
					panelStatus("local segmentation failed: " + (payload.failure_reason || "?"), false);
					lastState = payload.state || lastState;
					setOrderedMode("navigate", "");
					return null;
				}
				previewToken = payload.preview.token;
				preview = payload.preview;
				lastState = payload.state || lastState;
				renderPreviewPoints(payload.preview.preview_sample || []);
				setOrderedMode("navigate", "preview ready — inspect RAW members");
				renderOrderedPanel(lastState);
				focusPreview();
				panelStatus("待确认杆塔 / preview ready");
				return preview;
			}).catch(function (err) {
				panelStatus("select failed: " + err.message, false);
				setOrderedMode("navigate", "");
				return null;
			});
		}

		function doOrderConfirm() {
			if (!previewToken) {
				panelStatus("no pending preview", false);
				return Promise.resolve(null);
			}
			const tok = previewToken;
			panelStatus("confirming…");
			return postJson("/api/tower-review/ordered/confirm", {
				token: tok,
				operator_identity: "owner"
			}).then(function (payload) {
				const towerId = payload.result && payload.result.tower_id;
				clearPreview();
				panelStatus("confirmed " + (towerId || "?"));
				return refreshState().then(function () {
					return towerId || null;
				});
			}).catch(function (err) {
				panelStatus("confirm failed: " + err.message, false);
				return null;
			});
		}

		function doOrderReselect() {
			if (!previewToken) {
				panelStatus("no pending preview", false);
				return Promise.resolve();
			}
			const tok = previewToken;
			return postJson("/api/tower-review/ordered/reselect", {
				token: tok,
				operator_identity: "owner"
			}).then(function () {
				clearPreview();
				panelStatus("reselect — click the next tower");
				return refreshState();
			}).catch(function (err) {
				panelStatus("reselect failed: " + err.message, false);
			});
		}

		function doOrderCancel() {
			// CANCEL also exits an active "select" mode (no pending preview yet),
			// so the Owner can always return to navigate without picking a point.
			if (!previewToken) {
				if (orderedMode === "select") {
					setOrderedMode("navigate", "selection cancelled");
				}
				panelStatus("cancelled");
				return Promise.resolve();
			}
			const tok = previewToken;
			return postJson("/api/tower-review/ordered/cancel", {
				token: tok,
				operator_identity: "owner"
			}).then(function () {
				clearPreview();
				setOrderedMode("navigate", "");
				panelStatus("cancelled");
				return refreshState();
			}).catch(function (err) {
				panelStatus("cancel failed: " + err.message, false);
			});
		}

		function doAccept() {
			if (!selected) {
				panelStatus("select a candidate first", false);
				return Promise.resolve();
			}
			const set = selected.set;
			const cand = selected.candidate;
			const body = {
				candidate_set_digest: set.candidate_set_digest,
				operator_identity: "owner"
			};
			if (adjustedSeed) {
				body.adjusted_seed_xyz = [adjustedSeed.x, adjustedSeed.y, adjustedSeed.z];
			}
			return refreshAfter("accept " + cand.candidate_id, postJson(
				"/api/tower-review/candidates/" + encodeURIComponent(set.candidate_set_id) + "/" + cand.candidate_id + "/accept",
				body
			)).then(function () {
				adjustedSeed = null;
			});
		}

		function doReject() {
			if (!selected) {
				panelStatus("select a candidate first", false);
				return Promise.resolve();
			}
			const set = selected.set;
			const cand = selected.candidate;
			return refreshAfter("reject " + cand.candidate_id, postJson(
				"/api/tower-review/candidates/" + encodeURIComponent(set.candidate_set_id) + "/" + cand.candidate_id + "/reject",
				{ candidate_set_digest: set.candidate_set_digest, operator_identity: "owner", reason: "rejected by owner" }
			));
		}

		function doManualAdd(xyz) {
			return refreshAfter("manual add", postJson("/api/tower-review/manual-candidates", {
				xyz: [xyz.x, xyz.y, xyz.z],
				operator_identity: "owner",
				reason: "owner manual add"
			}));
		}

		function onPick(xyz) {
			if (orderedMode === "select") {
				doOrderSelect(xyz);
				return true;
			}
			if (mode === "manual") {
				doManualAdd(xyz).then(function () {
					setMode("navigate", "manual add sent");
				});
				return true;
			}
			if (mode === "adjust") {
				if (!selected) {
					panelStatus("select a candidate before Adjust", false);
					setMode("navigate", "");
					return true;
				}
				adjustedSeed = xyz;
				panelStatus("adjusted seed set for " + selected.candidate.candidate_id);
				setMode("navigate", "adjusted seed set — press Accept");
				return true;
			}
			return false;
		}

		function wirePanel() {
			const orderSelectBtn = document.getElementById("trp_order_select");
			const orderConfirmBtn = document.getElementById("trp_order_confirm");
			const orderReselectBtn = document.getElementById("trp_order_reselect");
			const orderCancelBtn = document.getElementById("trp_order_cancel");
			const autoToggle = document.getElementById("trp_auto_toggle");
			const autoBody = document.getElementById("trp_auto_body");
			const acceptBtn = document.getElementById("trp_accept");
			const rejectBtn = document.getElementById("trp_reject");
			const adjustBtn = document.getElementById("trp_adjust");
			const manualBtn = document.getElementById("trp_manual");
			const refreshBtn = document.getElementById("trp_refresh");
			const locateBtn = document.getElementById("trp_locate");
			const groupCreateBtn = document.getElementById("trp_group_create");
			const groupNameOk = document.getElementById("trp_group_name_ok");
			const groupNameCancel = document.getElementById("trp_group_name_cancel");
			if (groupCreateBtn) {
				groupCreateBtn.addEventListener("click", beginCreateGroup);
			}
			if (groupNameOk) {
				groupNameOk.addEventListener("click", submitCreateGroup);
			}
			if (groupNameCancel) {
				groupNameCancel.addEventListener("click", cancelCreateGroup);
			}
			if (orderSelectBtn) {
				orderSelectBtn.addEventListener("click", beginOrderSelect);
			}
			if (orderConfirmBtn) {
				orderConfirmBtn.addEventListener("click", function () { doOrderConfirm(); });
			}
			if (orderReselectBtn) {
				orderReselectBtn.addEventListener("click", function () { doOrderReselect(); });
			}
			if (orderCancelBtn) {
				orderCancelBtn.addEventListener("click", function () { doOrderCancel(); });
			}
			if (autoToggle && autoBody) {
				autoToggle.addEventListener("click", function () {
					const open = autoBody.classList.toggle("open");
					autoToggle.textContent = "自动候选辅助 / Automatic Suggestions " + (open ? "▾" : "▸");
				});
			}
			if (acceptBtn) {
				acceptBtn.addEventListener("click", function () { doAccept(); });
			}
			if (rejectBtn) {
				rejectBtn.addEventListener("click", function () { doReject(); });
			}
			if (adjustBtn) {
				adjustBtn.addEventListener("click", function () {
					if (!selected) {
						panelStatus("select a candidate before Adjust", false);
						return;
					}
					setMode("adjust", "click a point near " + selected.candidate.candidate_id);
				});
			}
			if (manualBtn) {
				manualBtn.addEventListener("click", function () {
					setMode("manual", "click the missed tower position");
				});
			}
			if (refreshBtn) {
				refreshBtn.addEventListener("click", function () {
					refreshState();
				});
			}
			if (locateBtn) {
				locateBtn.addEventListener("click", function () {
					if (!selected) {
						panelStatus("select a candidate before Locate", false);
						return;
					}
					focusOnSelected();
					panelStatus("located " + selected.candidate.candidate_id);
				});
			}
		}

		return {
			apiBase: apiBase,
			labelGroup: labelGroup,
			selectionGroup: selectionGroup,
			previewPointsGroup: previewPointsGroup,
			confirmedPointsGroup: confirmedPointsGroup,
			mode: function () { return mode; },
			orderedMode: function () { return orderedMode; },
			previewToken: function () { return previewToken; },
			activeGroup: function () { return activeGroupId; },
			onPick: onPick,
			refresh: refreshState,
			scaleLabels: scaleLabels,
			focusSelected: focusOnSelected,
			selectedKey: function () { return selected ? selected.key : null; },
			beginOrderSelect: beginOrderSelect,
			confirmPreview: doOrderConfirm,
			reselectPreview: doOrderReselect,
			cancelPreview: doOrderCancel,
			switchGroup: switchGroup,
			createGroup: submitCreateGroup,
			locateConfirmedTower: locateConfirmedTower,
			init: function () {
				wirePanel();
				setMode("navigate", "");
				renderOrderedPanel(null);
				return refreshState().catch(function () { return null; });
			}
		};
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
			lastPick: null,
			review: null
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
		// Click -> world XYZ (point picking), routed by review interaction mode.
		// -------------------------------------------------------------------
		instance.domElement.addEventListener("click", function (event) {
			const results = viewer.pointCloud.pick(new THREE.Vector2(event.clientX, event.clientY));
			viewer.lastPick = pickResultToXyz(results);
			const handled = viewer.review && viewer.lastPick ? viewer.review.onPick(viewer.lastPick) : false;
			updateStatus(cfg, viewer);
			if (handled) {
				return;
			}
			// NAVIGATE mode: keep the plain diagnostics behaviour.
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
			if (viewer.review && viewer.review.scaleLabels) {
				viewer.review.scaleLabels();
			}
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

		// -------------------------------------------------------------------
		// V5A2/V5A3 tower review (interaction-only; no scientific geometry).
		// -------------------------------------------------------------------
		try {
			viewer.review = makeTowerReview(viewer);
		} catch (err) {
			// Fail-safe: never leave the panel stuck at "loading".  Show a clear
			// error in the panel AND re-throw so the red banner still surfaces the
			// underlying error.
			const status = document.getElementById("trp_status");
			if (status) {
				status.textContent = "review init error: " + (err && err.message ? err.message : err);
				status.style.color = "#ff8a80";
			}
			const list = document.getElementById("trp_list");
			if (list) {
				list.innerHTML = "<div class='empty'>review init failed</div>";
			}
			throw err;
		}

		window.p3aGiro3dViewer = viewer;
		viewer.review.init();
		window.dispatchEvent(new Event("p3a-giro3d-ready"));
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
