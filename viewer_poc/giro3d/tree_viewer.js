/*
 * Tree Viewer D: identity-first, read-only inspection for final reconciled
 * trees.  It intentionally consumes only the server projection and never
 * creates or persists tree identities.
 */
(function () {
	"use strict";

	const PALETTE = [
		"#00bcd4", "#ff7043", "#ab47bc", "#66bb6a", "#ffa726", "#42a5f5",
		"#ec407a", "#8d6e63", "#d4e157", "#26a69a", "#7e57c2", "#ef5350",
		"#29b6f6", "#9ccc65", "#ffca28", "#26c6da", "#5c6bc0", "#ec407a",
		"#8bc34a", "#ff8a65", "#7cb342", "#26a69a", "#ffb300", "#7986cb"
	];

	function stableHash(value) {
		let hash = 2166136261;
		const text = String(value == null ? "" : value);
		for (let i = 0; i < text.length; i += 1) {
			hash ^= text.charCodeAt(i);
			hash = Math.imul(hash, 16777619);
		}
		return hash >>> 0;
	}

	function candidatesFor(tree) {
		return Array.from(new Set((tree.candidate_persistent_tree_ids || [])
			.filter(function (id) { return typeof id === "string" && id.trim(); })
			.map(function (id) { return id.trim(); }))).sort();
	}

	function treePresentation(tree) {
		const persistentId = typeof tree.persistent_tree_id === "string"
			&& tree.persistent_tree_id.trim() ? tree.persistent_tree_id.trim() : null;
		const candidates = candidatesFor(tree);
		if (persistentId) {
			return {
				kind: "persistent", primary: persistentId, state: tree.identity_state || "RESOLVED",
				candidates: candidates, colorKey: "persistent:" + persistentId,
				longTermPredictionEntity: true
			};
		}
		if (tree.identity_state === "MERGE_AMBIGUOUS_REVIEW") {
			return {
				kind: "merge-review", primary: "未绑定实体", state: "MERGE REVIEW",
				candidates: candidates, colorKey: "merge-review:" + candidates.join("|"),
				longTermPredictionEntity: false
			};
		}
		return {
			kind: "legacy", primary: "Legacy Tree " + String(tree.global_tree_id), state: "LEGACY",
			candidates: candidates, colorKey: "legacy:" + String(tree.global_tree_id),
			longTermPredictionEntity: false
		};
	}

	function treeColor(tree) {
		const presentation = treePresentation(tree);
		return tree.instance_color || PALETTE[stableHash(presentation.colorKey) % PALETTE.length];
	}

	function sortTrees(trees) {
		return (trees || []).slice().sort(function (left, right) {
			const leftOrder = Number.isFinite(Number(left.span_order_index))
				? Number(left.span_order_index) : Number.MAX_SAFE_INTEGER;
			const rightOrder = Number.isFinite(Number(right.span_order_index))
				? Number(right.span_order_index) : Number.MAX_SAFE_INTEGER;
			return leftOrder - rightOrder || Number(left.global_tree_id) - Number(right.global_tree_id);
		});
	}

	window.p3aTreeViewerUtil = {
		palette: PALETTE.slice(), stableHash: stableHash, treePresentation: treePresentation,
		treeColor: treeColor, sortTrees: sortTrees
	};

	function api(path) {
		return fetch(path, { cache: "no-store" }).then(function (response) {
			return response.json().then(function (payload) {
				if (!response.ok || !payload.ok) {
					const error = payload && payload.error ? payload.error : {};
					throw new Error((error.code || "HTTP_" + response.status) + ": " + (error.message || "request failed"));
				}
				return payload;
			});
		});
	}

	function start(host) {
		const spanSelect = document.getElementById("tree_span_select");
		const header = document.getElementById("tree_header");
		const list = document.getElementById("tree_list");
		const details = document.getElementById("tree_details");
		if (!spanSelect || !header || !list || !details || !host || !host.instance) return;

		const G = window.P3AGiro3D;
		const THREE = G && G.THREE;
		if (!THREE) return;
		const markerGroup = new THREE.Group();
		const previewGroup = new THREE.Group();
		markerGroup.name = "persistent-tree-markers";
		previewGroup.name = "persistent-tree-selection-preview";
		const attach = window.p3aGiro3dUtil && window.p3aGiro3dUtil.attachOverlayGroup;
		if (typeof attach !== "function") return;
		attach(host.instance, markerGroup);
		attach(host.instance, previewGroup);
		let activeSpanId = null;
		let activeRunId = null;
		let payload = null;
		let activeTree = null;

		function disposeGroup(group) {
			while (group.children.length) {
				const child = group.children.pop();
				if (child.geometry) child.geometry.dispose();
				if (child.material) child.material.dispose();
			}
		}

		function redrawMarkers() {
			disposeGroup(markerGroup);
			sortTrees(payload && payload.trees).forEach(function (tree) {
				const position = tree.apex_xyz_absolute_m;
				if (!Array.isArray(position) || position.length !== 3) return;
				const selected = activeTree && activeTree.global_tree_id === tree.global_tree_id;
				const marker = new THREE.Mesh(
					new THREE.SphereGeometry(selected ? 1.05 : 0.48, 14, 10),
					new THREE.MeshBasicMaterial({ color: treeColor(tree), transparent: true, opacity: selected ? 1 : 0.78, depthTest: false })
				);
				marker.position.set(position[0], position[1], position[2]);
				marker.userData.persistentTreeGlobalId = tree.global_tree_id;
				markerGroup.add(marker);
			});
			host.instance.notifyChange(host.instance.view.camera);
		}

		function renderPreview(points, tree) {
			disposeGroup(previewGroup);
			if (!points || !points.length) return;
			const geometry = new THREE.BufferGeometry();
			geometry.setAttribute("position", new THREE.Float32BufferAttribute(points.flat(), 3));
			const material = new THREE.PointsMaterial({ color: treeColor(tree), size: 0.7, sizeAttenuation: true, transparent: true, opacity: 0.98 });
			previewGroup.add(new THREE.Points(geometry, material));
			host.instance.notifyChange(host.instance.view.camera);
		}

		function renderDetails(tree) {
			const view = treePresentation(tree);
			const clearance = tree.minimum_clearance_m == null ? "N/A" : Number(tree.minimum_clearance_m).toFixed(3) + " m";
			details.textContent = [
				"Tree: " + view.primary,
				"Span: " + (tree.span_id || activeSpanId) + " · 顺序 " + (tree.span_order_index == null ? "—" : tree.span_order_index),
				"Identity state: " + view.state,
				view.candidates.length ? "Candidates: " + view.candidates.join(" / ") : "",
				"global_tree_id: " + tree.global_tree_id,
				"Clearance: " + clearance
			].filter(Boolean).join("\n");
		}

		function renderList() {
			const trees = sortTrees(payload && payload.trees);
			header.textContent = "Trees: " + (payload ? payload.tree_count : "—") + " · Core: " + (payload ? payload.core_count : "—") + " · Review: " + (payload ? payload.halo_review_count : "—");
			list.innerHTML = trees.map(function (tree) {
				const view = treePresentation(tree);
				const selected = activeTree && activeTree.global_tree_id === tree.global_tree_id ? " sel" : "";
				const clearance = tree.minimum_clearance_m == null ? "N/A" : Number(tree.minimum_clearance_m).toFixed(2) + " m";
				const order = tree.span_order_index == null ? "—" : String(tree.span_order_index).padStart(2, "0");
				const category = tree.category === "CORE" ? "CORE" : "REVIEW";
				const candidates = view.candidates.length ? " · " + view.candidates.join(" / ") : "";
				return "<div class='tree-row" + selected + "' data-tree='" + tree.global_tree_id + "'>" +
					"<span class='tswatch' style='background:" + treeColor(tree) + "'></span>" +
					"<span class='tname'>" + view.primary + "</span>" +
					"<span class='tmeta'>" + (tree.span_id || activeSpanId) + " · 顺序" + order + " · " + category + candidates + "</span>" +
					"<span class='tclear'>" + clearance + "</span></div>";
			}).join("") || "<div id='tree_empty'>无树 / no trees</div>";
			Array.prototype.forEach.call(list.querySelectorAll("[data-tree]"), function (row) {
				row.addEventListener("click", function () {
					const id = Number(row.getAttribute("data-tree"));
					selectTree((payload.trees || []).find(function (tree) { return tree.global_tree_id === id; }));
				});
			});
			redrawMarkers();
		}

		function focus(preview, tree) {
			const bounds = preview && preview.bounds_absolute_m;
			const min = bounds && bounds.min_xyz_absolute_m;
			const max = bounds && bounds.max_xyz_absolute_m;
			if (!min || !max || !window.p3aGiro3dUtil) return;
			const frame = window.p3aGiro3dUtil.computeFitCamera(
				{ x: min[0] - 4, y: min[1] - 4, z: min[2] - 3 },
				{ x: max[0] + 4, y: max[1] + 4, z: max[2] + 6 }, 60
			);
			const camera = host.instance.view.camera;
			camera.position.set(frame.position.x, frame.position.y, frame.position.z);
			camera.lookAt(frame.target.x, frame.target.y, frame.target.z);
			host.controls.target.set(frame.target.x, frame.target.y, frame.target.z);
			host.controls.update();
			host.instance.notifyChange(camera);
		}

		function selectTree(tree) {
			if (!tree) return;
			activeTree = tree;
			renderList();
			renderDetails(tree);
			api("/api/spans/" + encodeURIComponent(activeSpanId) + "/trees/" + tree.global_tree_id + "/preview")
				.then(function (preview) { if (activeTree === tree && preview.span_id === activeSpanId && preview.run_id === activeRunId) { renderPreview(preview.sampled_xyz_absolute_m || [], tree); focus(preview, tree); } })
				.catch(function () {});
		}

		function loadTrees(spanId, runId) {
			activeSpanId = spanId;
			activeRunId = runId || activeRunId;
			activeTree = null;
			disposeGroup(previewGroup);
			api("/api/spans/" + encodeURIComponent(spanId) + "/trees").then(function (next) {
				if (activeSpanId !== spanId || !next.run_id || (activeRunId && next.run_id !== activeRunId)) return;
				activeRunId = next.run_id;
				payload = next;
				renderList();
				details.textContent = "—";
			}).catch(function (error) {
				if (activeSpanId !== spanId) return;
				payload = null;
				list.innerHTML = "<div id='tree_empty'>" + error.message + "</div>";
				details.textContent = "—";
				disposeGroup(markerGroup);
			});
		}

		function requestActiveSpan(spanId) {
			if (host.review && typeof host.review.setActiveSpan === "function") {
				host.review.setActiveSpan(spanId);
				return;
			}
			loadTrees(spanId, null);
		}

		spanSelect.addEventListener("change", function () { requestActiveSpan(spanSelect.value); });
		window.addEventListener("p3a-active-span-changed", function (event) {
			const detail = event && event.detail;
			if (!detail || !detail.span_id) return;
			spanSelect.value = detail.span_id;
			if (activeSpanId !== detail.span_id || activeRunId !== detail.run_id) loadTrees(detail.span_id, detail.run_id);
		});
		api("/api/spans").then(function (spansPayload) {
			const spans = spansPayload.spans || [];
			spanSelect.innerHTML = spans.map(function (span) {
				return "<option value='" + span.span_id + "'>" + (span.display_name || span.span_id) + "</option>";
			}).join("");
			if (spans.length) requestActiveSpan(spans[0].span_id);
		}).catch(function (error) {
			spanSelect.innerHTML = "<option>Tree API unavailable</option>";
			details.textContent = error.message;
		});
	}

	function begin() {
		const host = window.p3aGiro3dViewer;
		if (host) start(host);
	}
	window.addEventListener("p3a-giro3d-ready", begin, { once: true });
	begin();
})();
