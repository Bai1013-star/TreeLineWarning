"""Additive finalized DBH geometry: recovery never changes tree membership."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
from algorithms.manual_span.metrics import _hag_chunked
from algorithms.spans.stem_guided_hybrid import (
    StemGuidedHybridConfig,
    _deduplicate_crown_stems,
    segment_stem_guided_hybrid,
)
from scipy.spatial import cKDTree

from core.config.fast_baseline import canonical_digest
from core.stage_b.dbh_evidence import (
    SCHEMA,
    array_digest,
    confined,
    editor_eligibility,
    local_raw_query,
    member_binding,
    read_json,
    sha256,
    strong_source,
)

VERSION = "dbh-recovery-v1.0"
DBH_FIELDS = (
    "dbh_cm",
    "dbh_method",
    "dbh_quality",
    "dbh_point_count",
    "dbh_fit_rmse",
    "dbh_slice_count",
    "dbh_fit_status",
    "dbh_source",
    "dbh_model_version",
    "dbh_uncertainty_cm",
)


def unavailable(reason: str) -> dict[str, Any]:
    return {
        "dbh_cm": None,
        "dbh_method": "UNAVAILABLE",
        "dbh_quality": "UNAVAILABLE",
        "dbh_point_count": 0,
        "dbh_fit_rmse": None,
        "dbh_slice_count": 0,
        "dbh_fit_status": reason,
        "dbh_source": VERSION,
        "dbh_model_version": None,
        "dbh_uncertainty_cm": None,
    }


def publication_fields(row: dict[str, Any]) -> dict[str, Any]:
    fields = {key: row.get(key) for key in DBH_FIELDS}
    fields.update(diameter_cm=row.get("dbh_cm"), diameter_source=row.get("dbh_method"))
    return fields


def _run_root(root: Path, run_id: str) -> Path:
    if not run_id or any(
        c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in run_id
    ):
        raise ValueError("DBH_INVALID_RUN_ID")
    return confined(root, f"artifacts/project-runs/batches/{run_id}")


def _context(root: Path, result: Any) -> tuple[Path, dict[str, Any], np.ndarray, tuple]:
    from core.stage_b.tree_metrics import _preparation_candidates, _valid_preparation_context

    contexts = []
    for path in _preparation_candidates(root, result):
        value = _valid_preparation_context(
            root, path, expected_point_count=result.telemetry.source_point_count
        )
        if value is not None:
            manifest = read_json(path)
            strong_source(manifest.get("ground_preparation", {}).get("source_identity", {}))
            contexts.append((path, manifest, *value))
    if len(contexts) != 1:
        raise ValueError("DBH_SOURCE_PREPARATION_AMBIGUOUS_OR_MISSING")
    return contexts[0]


def _binding(root: Path, result: Any, reconciled: Any) -> dict[str, Any]:
    br = _run_root(root, str(result.analysis_run_id))
    batch = read_json(br / "batch.json")
    workflow_path = root / "desktop-workflow.json"
    workflow = read_json(workflow_path) if workflow_path.is_file() else {}
    states = set(batch.get("span_statuses", {}).values())
    finalized = workflow.get("finalize_state") == {
        "run_id": result.analysis_run_id,
        "status": "DONE",
    } or states == {"FINALIZED"}
    if (
        not finalized
        or batch.get("analysis_run_id") != result.analysis_run_id
        or batch.get("project_id") != result.project_id
    ):
        raise ValueError("DBH_REQUIRES_MATCHING_FINALIZED_RUN")
    journal = read_json(br / "tree_edits.json")
    if journal.get("analysis_run_id") != result.analysis_run_id:
        raise ValueError("DBH_EDITOR_RUN_MISMATCH")
    path, preparation, _ground, _origin = _context(root, result)
    source = strong_source(preparation["ground_preparation"]["source_identity"])
    project_path = root / "stage-a-project.json"
    if project_path.is_file():
        project = read_json(project_path)["project"]
        scans = [s for s in project["scans"] if s["scan_id"] == result.scan_id]
        if len(scans) != 1 or project["project_id"] != result.project_id:
            raise ValueError("DBH_CURRENT_PROJECT_SCAN_MISMATCH")
        identity = dict(scans[0]["source_identity"])
        identity["source_point_count"] = scans[0]["capability_report"]["details"]["point_count"]
        if strong_source(identity) != source:
            raise ValueError("DBH_CURRENT_SOURCE_IDENTITY_MISMATCH")
    return {
        "schema": SCHEMA,
        "version": VERSION,
        "project_id": result.project_id,
        "analysis_run_id": result.analysis_run_id,
        "finalized_state": "FINALIZED",
        "source_identity": strong_source(preparation["ground_preparation"]["source_identity"]),
        "preparation_reference": path.relative_to(root).as_posix(),
        "preparation_sha256": sha256(path),
        "ground_sha256": sha256(confined(root, preparation["coarse_ground_reference"])),
        "editor_revision": canonical_digest(journal),
        "final_membership_artifact_sha256": sha256(br / "tree_segmentation_final.npz"),
        "original_lineage_sha256": sha256(br / "batch-reconciliation.json"),
        "membership": member_binding(reconciled),
    }


def load_finalized_dbh(root: Path, result: Any, reconciled: Any) -> dict[int, dict[str, Any]]:
    """Read-only publication. Existing invalid proof overrides with explicit unavailable."""
    root = Path(root).resolve()
    pointer = _run_root(root, str(result.analysis_run_id)) / "dbh_final.json"
    if not pointer.is_file():
        return {}
    try:
        ref = read_json(pointer)
        path = confined(root, ref["geometry_reference"])
        if sha256(path) != ref["geometry_sha256"]:
            raise ValueError("DBH_GEOMETRY_HASH_MISMATCH")
        payload = read_json(path)
        if payload["binding"] != _binding(root, result, reconciled):
            raise ValueError("DBH_STALE_PROVENANCE")
        evidence = confined(root, payload["evidence_reference"])
        if sha256(evidence) != payload["evidence_sha256"]:
            raise ValueError("DBH_EVIDENCE_HASH_MISMATCH")
        for proof in payload.get("additional_input_hashes", []):
            if sha256(confined(root, proof["reference"])) != proof["sha256"]:
                raise ValueError("DBH_ADDITIONAL_INPUT_HASH_MISMATCH")
        rows = payload["trees"]
        expected = {
            int(r.global_tree_id): r.persistent_tree_id for r in reconciled.global_tree_records
        }
        if (
            len(rows) != len(expected)
            or {r["global_tree_id"]: r["persistent_tree_id"] for r in rows} != expected
        ):
            raise ValueError("DBH_FINAL_IDENTITY_MISMATCH")
        for row in rows:
            value, method = row.get("dbh_cm"), row.get("dbh_method")
            if method not in {
                "DIRECT_STEM",
                "RECOVERED_STEM",
                "REVIEW_CONFIRMED_STEM",
                "ALLOMETRIC_ESTIMATE",
                "UNAVAILABLE",
            }:
                raise ValueError("DBH_INVALID_METHOD")
            if value is not None and (
                not isinstance(value, (float, int))
                or not np.isfinite(value)
                or not 5 <= value <= 150
            ):
                raise ValueError("DBH_INVALID_DIAMETER")
            if (value is None) != (method == "UNAVAILABLE"):
                raise ValueError("DBH_VALUE_METHOD_MISMATCH")
            if method == "REVIEW_CONFIRMED_STEM":
                review = row.get("dbh_review", {})
                decision = confined(root, row["dbh_review_decision_reference"])
                if (
                    sha256(decision) != row["dbh_review_decision_sha256"]
                    or read_json(decision) != review
                ):
                    raise ValueError("DBH_REVIEW_DECISION_PROOF_MISMATCH")
                if (
                    review.get("identity_confirmed") is not True
                    or review.get("persistent_tree_id") != row["persistent_tree_id"]
                    or review.get("editor_revision") != payload["binding"]["editor_revision"]
                ):
                    raise ValueError("DBH_INVALID_REVIEW_CONFIRMATION")
        return {int(row["global_tree_id"]): row for row in rows}
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return {
            int(r.global_tree_id): unavailable("DBH_PROOF_INVALID_OR_STALE")
            for r in reconciled.global_tree_records
        }


def _breast_height_anchor(primary: Any, raw: list, support: dict, cfg: Any) -> dict:
    """Prefer low-stem support only within the already deduplicated stem group."""
    xy = np.asarray([s.xy_local_m for s in raw])
    start = int(np.argmin(np.linalg.norm(xy - primary.xy_local_m, axis=1)))
    neighbors = cKDTree(xy).query_ball_point(xy, r=cfg.stem_dedup_radius_m)
    seen, stack = {start}, [start]
    while stack:
        for other in neighbors[stack.pop()]:
            if other not in seen:
                seen.add(other)
                stack.append(other)

    def count(stem):
        cell = tuple(np.floor(np.asarray(stem.xy_local_m) / cfg.stem_grid_m).astype(int))
        return support.get(cell, 0)

    eligible = [raw[i] for i in sorted(seen) if count(raw[i]) > 0]
    selected = (
        min(
            eligible,
            key=lambda s: (-count(s), -s.stem_confidence, tuple(s.xy_local_m), s.stem_seed_id),
        )
        if eligible
        else primary
    )
    return {
        "xy": np.asarray(selected.xy_local_m),
        "confidence": selected.stem_confidence,
        "primary_xy": list(primary.xy_local_m),
        "selection": "BH_SUPPORTED_COLUMN_IN_EXISTING_STEM_GROUP"
        if eligible
        else "EXISTING_PRIMARY_STEM_AXIS",
        "group_raw_stem_ids": sorted(raw[i].stem_seed_id for i in seen),
    }


def _span_evidence(tree_root: Path) -> dict[str, Any]:
    """Reconstruct existing stem hypotheses; exact saved label equality is mandatory."""
    with np.load(tree_root / "tree_fine.npz", allow_pickle=False) as archive:
        fine = {k: archive[k] for k in archive.files}
    with np.load(tree_root / "legacy_reference/tree_fine.npz", allow_pickle=False) as archive:
        baseline = archive["fine_tree_id"]
        for key in ("fine_original_index", "fine_xyz_local_m", "fine_hag_m"):
            if not np.array_equal(fine[key], archive[key]):
                raise ValueError("DBH_FINE_LINEAGE_MISMATCH")
    with np.load(
        tree_root / "legacy_reference/fine_representatives.npz", allow_pickle=False
    ) as archive:
        weights = archive["raw_count"]
    xyz, hag = fine["fine_xyz_local_m"], fine["fine_hag_m"]
    if len(np.unique(fine["fine_original_index"])) != len(xyz):
        raise ValueError("DBH_FINE_DUPLICATE_ORIGINAL_INDEX")
    hybrid = segment_stem_guided_hybrid(xyz, hag, baseline, raw_count=weights)
    if not np.array_equal(hybrid.fine_tree_id, fine["fine_tree_id"]):
        raise ValueError("DBH_HYBRID_REPLAY_MISMATCH")
    cfg = StemGuidedHybridConfig()
    crown_ids = sorted(int(v) for v in np.unique(baseline) if v >= 0)
    crowns = {cid: cKDTree(xyz[baseline == cid, :2]) for cid in crown_ids}
    owned = {cid: [] for cid in crown_ids}
    for stem in hybrid.validated_stems:
        choices = [(float(tree.query(stem.xy_local_m)[0]), cid) for cid, tree in crowns.items()]
        if choices:
            distance, cid = min(choices)
            if distance <= cfg.stem_crown_link_radius_m:
                owned[cid].append(stem)
    hypotheses = {}
    for cid in crown_ids:
        for stem in _deduplicate_crown_stems(tuple(owned[cid]), cfg):
            number = len(hypotheses)
            hypotheses[number] = replace(stem, stem_seed_id=number)
    band = (hag >= 0.85) & (hag <= 1.75)
    cells, counts = np.unique(
        np.floor(xyz[band, :2] / cfg.stem_grid_m).astype(np.int64), axis=0, return_counts=True
    )
    support = {tuple(cell): int(count) for cell, count in zip(cells, counts, strict=True)}
    anchors = {}
    for candidate in hybrid.candidates:
        if candidate.quality_state.value == "CONFIRMED_TREE" and len(candidate.stem_seed_ids) == 1:
            stem = hypotheses[candidate.stem_seed_ids[0]]
            anchors[candidate.tree_id] = _breast_height_anchor(
                stem, owned[candidate.source_crown_ids[0]], support, cfg
            )
    return {
        "fine": fine,
        "anchors": anchors,
        "proof": {
            name: sha256(tree_root / name)
            for name in (
                "tree_fine.npz",
                "span_tree_manifest.json",
                "legacy_reference/tree_fine.npz",
                "legacy_reference/fine_representatives.npz",
            )
        },
    }


def _persist(root: Path, result: Any, payload: dict, arrays: dict) -> Path:
    br = _run_root(root, str(result.analysis_run_id))
    generation = br / "dbh_v1" / uuid4().hex
    generation.mkdir(parents=True, exist_ok=False)
    evidence = generation / "evidence.npz"
    np.savez_compressed(evidence, **arrays)
    payload.update(
        evidence_reference=evidence.relative_to(root).as_posix(), evidence_sha256=sha256(evidence)
    )
    path = generation / "finalized_geometry.json"
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8"
    )
    pointer = br / "dbh_final.json"
    temp = pointer.with_name("dbh_final." + uuid4().hex + ".tmp")
    temp.write_text(
        json.dumps(
            {
                "schema": SCHEMA,
                "geometry_reference": path.relative_to(root).as_posix(),
                "geometry_sha256": sha256(path),
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    temp.replace(pointer)
    return path


def _fit_evidence(xyz, idx, hag, row, *, method, source):
    from algorithms.manual_span.dbh_recovery import (
        fit_stem_dbh,
        select_continuous_stem_points,
    )

    row.pop("dbh_continuity", None)
    fit = fit_stem_dbh(xyz, hag, method=method)
    row["fit_attempts"].append({"source": source, **fit})
    if fit["dbh_cm"] is None:
        keep, continuity = select_continuous_stem_points(xyz, hag)
        continuity["selected_original_index_sha256"] = array_digest(idx[keep])
        row["dbh_continuity"] = continuity
        xyz, idx, hag = xyz[keep], idx[keep], hag[keep]
        fit = fit_stem_dbh(xyz, hag, method=method)
        if not len(idx):
            fit["dbh_fit_status"] = continuity["status"]
        row["fit_attempts"].append(
            {"source": source + "_VERTICAL_CONTINUITY", "continuity": continuity, **fit}
        )
    return fit, xyz, idx, hag


def recover_finalized_dbh(
    root: Path,
    result: Any,
    reconciled: Any,
    *,
    persist: bool = True,
    progress: Any = None,
) -> dict[str, Any]:
    """Explicit bounded recovery. Never invoked by a read-only report request."""
    from algorithms.manual_span.dbh_recovery import estimate_local_dbh

    from core.stage_b.tree_metrics import compute_stage_b_tree_metrics

    root = Path(root).resolve()
    binding = _binding(root, result, reconciled)
    br = _run_root(root, str(result.analysis_run_id))
    original = read_json(br / "batch-reconciliation.json")["trees"]
    operations = read_json(br / "tree_edits.json")["operations"]
    eligibility, removed = editor_eligibility(original, operations)
    _prep_path, prep, ground, origin = _context(root, result)
    origin = np.asarray(origin)
    ground_tree = cKDTree(ground[:, :2])
    cache_root = confined(root, prep["vegetation_role_tile_index_reference"])
    source = prep["ground_preparation"]["source_identity"]
    metrics = compute_stage_b_tree_metrics(
        result, reconciled, root / "exports/Prediction_Input.xlsx", project_root=root
    )
    if len(metrics) != len(reconciled.global_tree_records):
        raise ValueError("DBH_CANONICAL_METRICS_MISSING")
    # Bind internal Span identity from the canonical per-span manifest, not folder order.
    spans = {}
    for tree_root in sorted(br.glob("*/tree")):
        manifest_path = tree_root / "span_tree_manifest.json"
        if manifest_path.is_file():
            manifest = read_json(manifest_path)
            span_id = int(manifest["span_id"])
            if span_id in spans:
                raise ValueError("DBH_DUPLICATE_SPAN_IDENTITY")
            spans[span_id] = tree_root
    span_cache = {}
    evidence_xyz, evidence_idx, evidence_hag, evidence_gid = [], [], [], []
    rows = []
    for record in sorted(reconciled.global_tree_records, key=lambda r: r.global_tree_id):
        gid = int(record.global_tree_id)
        if progress is not None:
            progress(f"DBH_TREE:{gid}")
        metric = metrics[gid]
        row = {
            **unavailable("NO_SUPPORTED_STEM"),
            "global_tree_id": gid,
            "persistent_tree_id": record.persistent_tree_id,
            "height_m": metric.tree_height_m,
            "dbh_evidence_point_count": 0,
            "dbh_band_point_count": 0,
            "editor_filtered_point_count": 0,
            "ambiguous_point_count": 0,
            "raw_local_query_used": False,
            "raw_local_query_attempted": False,
            "raw_local_point_count": 0,
            "raw_dbh_band_point_count": 0,
            "stem_association_status": "UNAVAILABLE",
            "stem_association_confidence": None,
            "lineage": [list(k) for k in record.merged_span_tree_keys],
            "fit_attempts": [],
        }
        xyz, idx, hag = np.empty((0, 3)), np.empty(0, dtype=np.int64), np.empty(0)
        if metric.dbh.band_point_count:
            mask = np.asarray(reconciled.final_tree_id) == gid
            xyz = np.asarray(reconciled.final_xyz_local_m)[mask]
            idx = np.asarray(reconciled.final_original_index)[mask]
            hag = _hag_chunked(xyz, ground_tree, ground, origin, 250_000)
            band = (hag >= 0.85) & (hag <= 1.75)
            xyz, idx, hag = xyz[band], idx[band], hag[band]
            fit, xyz, idx, hag = _fit_evidence(
                xyz, idx, hag, row, method="DIRECT_STEM", source="FINAL_MEMBERS"
            )
            row.update(fit)
        lineage = [tuple(int(v) for v in k) for k in record.merged_span_tree_keys]
        try:
            anchors, blocks = [], []
            for span_id, source_tree in lineage:
                if span_id not in span_cache:
                    span_cache[span_id] = _span_evidence(spans[span_id])
                span = span_cache[span_id]
                anchor = span["anchors"].get(source_tree)
                if anchor is None:
                    raise ValueError("NO_VALIDATED_STEM_AXIS")
                anchors.append(anchor)
                fine = span["fine"]
                selected = (
                    np.linalg.norm(fine["fine_xyz_local_m"][:, :2] - anchor["xy"], axis=1) <= 0.60
                )
                selected &= (fine["fine_hag_m"] >= 0.5) & (fine["fine_hag_m"] <= 5.5)
                blocks.append(
                    (
                        fine["fine_xyz_local_m"][selected],
                        fine["fine_original_index"][selected],
                        fine["fine_hag_m"][selected],
                    )
                )
            if not anchors:
                raise ValueError("NO_VALIDATED_STEM_AXIS")
            anchor_xy = np.asarray([a["xy"] for a in anchors])
            if np.max(np.linalg.norm(anchor_xy - anchor_xy[0], axis=1)) > 0.15:
                raise ValueError("MERGED_MULTIPLE_STEMS_AMBIGUOUS")
            candidate_xyz = np.concatenate([b[0] for b in blocks])
            candidate_idx = np.concatenate([b[1] for b in blocks])
            candidate_hag = np.concatenate([b[2] for b in blocks])
            _, unique = np.unique(candidate_idx, return_index=True)
            candidate_xyz, candidate_idx, candidate_hag = (
                candidate_xyz[unique],
                candidate_idx[unique],
                candidate_hag[unique],
            )
            row["pre_editor_evidence_point_count"] = len(candidate_idx)
            row["pre_editor_dbh_band_point_count"] = int(
                np.count_nonzero((candidate_hag >= 0.85) & (candidate_hag <= 1.75))
            )
            if row["dbh_cm"] is None:
                states = [eligibility.get(k, "UNKNOWN_EDITOR_LINEAGE") for k in lineage]
                if any(s != "ELIGIBLE" for s in states):
                    row["editor_filtered_point_count"] = len(candidate_idx)
                    raise ValueError(next(s for s in states if s != "ELIGIBLE"))
                # All other validated tree axes compete for ownership, including deleted trees.
                for span_id, _ in lineage:
                    for tid, other in span_cache[span_id]["anchors"].items():
                        if (span_id, tid) not in lineage and np.linalg.norm(
                            other["xy"] - anchor_xy[0]
                        ) < 1.20:
                            row["ambiguous_point_count"] = len(candidate_idx)
                            raise ValueError("NEIGHBOR_STEM_ASSOCIATION_AMBIGUOUS")
                keep = ~np.isin(candidate_idx, removed)
                row["editor_filtered_point_count"] = int(np.count_nonzero(~keep))
                xyz, idx, hag = candidate_xyz[keep], candidate_idx[keep], candidate_hag[keep]
                fit, xyz, idx, hag = _fit_evidence(
                    xyz, idx, hag, row, method="RECOVERED_STEM", source="PERSISTED_FINE"
                )
                row.update(fit)
                row["stem_association_status"] = "VALIDATED_PRE_EDIT_AXIS_CURRENT_LINEAGE"
                row["stem_association_confidence"] = float(min(a["confidence"] for a in anchors))
                row["stem_axis_xy_local_m"] = anchor_xy[0].tolist()
                row["stem_axis_selection"] = [
                    {k: v for k, v in a.items() if k != "xy"} for a in anchors
                ]
                if row["dbh_cm"] is None:
                    row["raw_local_query_attempted"] = True
                    seed_xy = anchor_xy[0] + origin[:2]
                    neighbor_ids = ground_tree.query(seed_xy, k=8)[1]
                    ground_z = float(np.median(ground[neighbor_ids, 2]))
                    raw_xyz, raw_idx, query = local_raw_query(
                        cache_root, source, seed_xy, (ground_z - 1, ground_z + 7)
                    )
                    row.update(query)
                    raw_xyz = raw_xyz - origin
                    raw_hag = _hag_chunked(raw_xyz, ground_tree, ground, origin, 250_000)
                    keep = (raw_hag >= 0.5) & (raw_hag <= 5.5) & ~np.isin(raw_idx, removed)
                    xyz, idx, hag = raw_xyz[keep], raw_idx[keep], raw_hag[keep]
                    row["raw_dbh_band_point_count"] = int(
                        np.count_nonzero((hag >= 0.85) & (hag <= 1.75))
                    )
                    fit, xyz, idx, hag = _fit_evidence(
                        xyz, idx, hag, row, method="RECOVERED_STEM", source="EXACT_RAW_LOCAL"
                    )
                    row.update(fit)
        except (OSError, ValueError, KeyError, RuntimeError) as exc:
            if row["dbh_cm"] is None:
                row.update(unavailable(str(exc)))
                row["stem_association_status"] = str(exc)
                if not row["raw_local_query_attempted"]:
                    xyz, idx, hag = np.empty((0, 3)), np.empty(0, dtype=np.int64), np.empty(0)
        row["dbh_evidence_point_count"] = len(idx)
        row["dbh_band_point_count"] = int(np.count_nonzero((hag >= 0.85) & (hag <= 1.75)))
        row["dbh_source"] = VERSION
        evidence_xyz.append(xyz)
        evidence_idx.append(idx)
        evidence_hag.append(hag)
        evidence_gid.append(np.full(len(idx), gid, dtype=np.int32))
        rows.append(row)
    all_indices = np.concatenate(evidence_idx)
    if len(np.unique(all_indices)) != len(all_indices):
        # A halo may expose the same source stem to different spans. Never
        # publish it twice or use duplicated ownership as allometric training.
        raise ValueError("DBH_CROSS_TREE_EVIDENCE_OWNERSHIP_AMBIGUOUS")
    estimates = estimate_local_dbh(rows, [r for r in rows if r["dbh_cm"] is None])
    for row in rows:
        estimate = estimates.get(row["persistent_tree_id"])
        if row["dbh_cm"] is None and estimate:
            row["estimation_diagnostics"] = estimate
            if estimate.get("dbh_method") == "ALLOMETRIC_ESTIMATE":
                row.update(estimate)
    if binding != _binding(root, result, reconciled):
        raise ValueError("DBH_PROJECT_CHANGED_DURING_RECOVERY")
    payload = {
        "schema": SCHEMA,
        "binding": binding,
        "trees": rows,
        "span_proofs": {str(k): v["proof"] for k, v in span_cache.items()},
    }
    arrays = {
        "xyz_local_m": np.concatenate(evidence_xyz),
        "original_index": np.concatenate(evidence_idx),
        "hag_m": np.concatenate(evidence_hag),
        "global_tree_id": np.concatenate(evidence_gid),
    }
    if persist:
        path = _persist(root, result, payload, arrays)
        payload["published_reference"] = path.relative_to(root).as_posix()
    return payload
