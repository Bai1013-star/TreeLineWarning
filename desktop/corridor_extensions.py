"""Desktop corridor-registration and GIS tower assistance extensions.

This module stays at the product adapter boundary.  It does not alter Ground,
Wire, vegetation, tree segmentation, clearance, or risk science.  Candidate
ranking is display assistance; only RAW local refinement plus explicit operator
confirmation can update a persistent Tower position.
"""

from __future__ import annotations

from functools import wraps
from typing import Any

from core.stage_a import (
    AnchorCorrespondence,
    RegistrationResult,
    RegistrationState,
    apply_registration_xy,
    confirm_gis_tower,
    georeferenced_registration,
    load_project,
    register_one_anchor_geospatial,
    register_two_anchors_geospatial,
    registration_result_from_dict,
    save_project,
    save_scan_registration,
    scan_registration_record,
    validate_registration_checkpoint,
)
from core.stage_a.tower_registration import GIS_CONFIRMED_POSITION_SOURCE

from desktop.tower_candidates import rank_tower_candidates

_INSTALLED = False


def _gis_source_identity(project: Any) -> dict[str, Any] | None:
    latest = project.metadata.get("gis_import")
    if not isinstance(latest, dict):
        return None
    source = latest.get("source")
    return dict(source) if isinstance(source, dict) else None


def _tower_gis_position(tower: Any) -> tuple[float, float, float]:
    original = tower.attributes.get("gis_original_position")
    if isinstance(original, (list, tuple)) and len(original) == 3:
        return tuple(float(item) for item in original)
    if str(tower.position_source).startswith("GIS:"):
        return tuple(float(item) for item in tower.position)
    raise ValueError("TOWER_HAS_NO_GIS_SEED_POSITION")


def _tower_gis_crs(tower: Any) -> str | None:
    raw = tower.attributes.get("gis_original_position_crs")
    if isinstance(raw, dict):
        value = raw.get("horizontal_crs")
        if isinstance(value, str) and value.strip():
            return value.strip()
    if str(tower.position_source).startswith("GIS:"):
        value = tower.position_crs.horizontal_crs
        return value.strip() if isinstance(value, str) and value.strip() else None
    return None


def _project_gis_crs(project: Any) -> str | None:
    values: set[str] = set()
    for tower in project.towers:
        value = _tower_gis_crs(tower)
        if value:
            values.add(value)
    for route in project.routes:
        raw = route.source_provenance.get("route_position_crs")
        if isinstance(raw, dict):
            value = raw.get("horizontal_crs")
            if isinstance(value, str) and value.strip():
                values.add(value.strip())
    if len(values) > 1:
        raise ValueError("GIS_CRS_CONFLICT_REQUIRES_CONFIRMATION")
    return next(iter(values), None)


def _registration_from_scan(scan: Any) -> RegistrationResult | None:
    record = scan_registration_record(scan)
    if record is None:
        return None
    return registration_result_from_dict(record.get("result"))


def _anchors_from_scan(scan: Any) -> tuple[AnchorCorrespondence, ...]:
    record = scan_registration_record(scan)
    if record is None:
        return ()
    raw_anchors = record.get("anchors")
    result: list[AnchorCorrespondence] = []
    if isinstance(raw_anchors, list):
        for item in raw_anchors:
            if not isinstance(item, dict):
                continue
            result.append(
                AnchorCorrespondence(
                    tuple(float(value) for value in item["point_cloud_xyz"]),
                    tuple(float(value) for value in item["gis_xyz"]),
                    item.get("gis_asset_id"),
                )
            )
    checkpoint = record.get("checkpoint")
    if isinstance(checkpoint, dict):
        result.append(
            AnchorCorrespondence(
                tuple(float(value) for value in checkpoint["point_cloud_xyz"]),
                tuple(float(value) for value in checkpoint["gis_xyz"]),
                checkpoint.get("gis_asset_id"),
            )
        )
    return tuple(result)


def _tower_ready(tower: Any) -> bool:
    source = str(tower.position_source)
    if source.startswith("GIS:"):
        return False
    if tower.attributes.get("gis_source") is not None:
        return source == GIS_CONFIRMED_POSITION_SOURCE
    # Preserve legacy/manual tower workflows that predate GIS seed states.
    return True


def _span_readiness(project: Any) -> list[dict[str, Any]]:
    tower_by_id = {tower.tower_id: tower for tower in project.towers}
    rows: list[dict[str, Any]] = []
    for span in project.spans:
        reasons: list[str] = []
        start = tower_by_id.get(span.start_tower_id)
        end = tower_by_id.get(span.end_tower_id)
        if start is None or not _tower_ready(start):
            reasons.append("START_TOWER_NOT_CONFIRMED")
        if end is None or not _tower_ready(end):
            reasons.append("END_TOWER_NOT_CONFIRMED")
        rows.append(
            {
                "span_id": span.span_id,
                "status": "READY" if not reasons else "BLOCKED",
                "reasons": reasons,
            }
        )
    return rows


def install_corridor_extensions() -> None:
    global _INSTALLED
    if _INSTALLED:
        return

    from desktop.workflow import DesktopWorkflow, _coordinate_from_scan

    original_project_projection = DesktopWorkflow.project_projection
    original_start_batch = DesktopWorkflow.start_batch

    def corridor_registration_projection(self: Any) -> dict[str, Any]:
        project = self._stage_project()
        scan = self._scan(project)
        if scan is None:
            return {
                "available": False,
                "state": RegistrationState.UNREGISTERED.value,
                "message": "POINT_CLOUD_NOT_IMPORTED",
            }
        source = _gis_source_identity(project)
        gis_crs = _project_gis_crs(project)
        registration = _registration_from_scan(scan)
        return {
            "available": source is not None,
            "state": (
                registration.state.value
                if registration is not None
                else RegistrationState.UNREGISTERED.value
            ),
            "ready_for_science": bool(registration and registration.ready_for_science),
            "transform_available": bool(
                registration
                and registration.transform_2d is not None
                and registration.unique_rigid_transform
            ),
            "anchor_count": registration.anchor_count if registration else 0,
            "checkpoint_residual_m": (
                registration.checkpoint_residual_m if registration else None
            ),
            "diagnostic": registration.diagnostic if registration else None,
            "point_cloud_crs": _coordinate_from_scan(scan).horizontal_crs,
            "gis_crs": gis_crs,
            "gis_source_identity": source,
            "manual_fallback_available": True,
        }

    def prepare_corridor_registration(self: Any) -> dict[str, Any]:
        project = self._stage_project()
        scan = self._scan(project)
        if scan is None:
            raise ValueError("POINT_CLOUD_NOT_IMPORTED")
        source = _gis_source_identity(project)
        if source is None:
            raise ValueError("GIS_NOT_IMPORTED")
        gis_crs = _project_gis_crs(project)
        point_cloud_crs = _coordinate_from_scan(scan).horizontal_crs
        if not point_cloud_crs or not gis_crs:
            return corridor_registration_projection(self)
        result = georeferenced_registration(
            point_cloud_crs=point_cloud_crs,
            gis_crs=gis_crs,
        )
        save_scan_registration(
            self._root,
            scan.scan_id,
            result,
            gis_source_identity=source,
        )
        self._log(
            "corridor_registration_prepared",
            state=result.state.value,
            point_cloud_crs=point_cloud_crs,
            gis_crs=gis_crs,
        )
        return corridor_registration_projection(self)

    def set_corridor_anchor(self: Any, payload: Any) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise ValueError("ANCHOR_PAYLOAD_REQUIRED")
        project = self._stage_project()
        scan = self._scan(project)
        if scan is None:
            raise ValueError("POINT_CLOUD_NOT_IMPORTED")
        tower_id = str(payload.get("tower_id") or "")
        tower = next((item for item in project.towers if item.tower_id == tower_id), None)
        if tower is None:
            raise ValueError("TOWER_NOT_FOUND")
        raw_xyz = payload.get("point_cloud_xyz")
        if not isinstance(raw_xyz, (list, tuple)) or len(raw_xyz) != 3:
            raise ValueError("ANCHOR_POINT_CLOUD_XYZ_REQUIRED")
        point_cloud_xyz = tuple(float(item) for item in raw_xyz)
        gis_xyz = _tower_gis_position(tower)
        gis_crs = _tower_gis_crs(tower) or _project_gis_crs(project)
        if not gis_crs:
            raise ValueError("GIS_CRS_REQUIRED_FOR_ANCHOR")
        anchor = AnchorCorrespondence(
            point_cloud_xyz,
            gis_xyz,
            tower.external_asset_id,
        )
        anchors = list(_anchors_from_scan(scan))
        anchors = [item for item in anchors if item.gis_asset_id != anchor.gis_asset_id]
        anchors.append(anchor)
        if len(anchors) > 3:
            raise ValueError("ANCHOR_CHECKPOINT_ALREADY_DEFINED")
        if len(anchors) == 1:
            result = register_one_anchor_geospatial(anchors[0], gis_crs=gis_crs)
            saved_anchors = tuple(anchors)
            checkpoint = None
        else:
            result = register_two_anchors_geospatial(
                anchors[0],
                anchors[1],
                gis_crs=gis_crs,
            )
            saved_anchors = tuple(anchors[:2])
            checkpoint = anchors[2] if len(anchors) == 3 else None
            if checkpoint is not None and result.state is RegistrationState.REGISTERED_TWO_ANCHOR:
                result = validate_registration_checkpoint(result, checkpoint)
        save_scan_registration(
            self._root,
            scan.scan_id,
            result,
            anchors=saved_anchors,
            checkpoint=checkpoint,
            gis_source_identity=_gis_source_identity(project),
        )
        refreshed = load_project(self._root)
        point_cloud_coordinate = _coordinate_from_scan(
            next(item for item in refreshed.scans if item.scan_id == scan.scan_id)
        )
        confirmed = confirm_gis_tower(
            refreshed,
            tower_id,
            point_cloud_xyz,
            point_cloud_coordinate,
            evidence={
                "registration_role": (
                    "CHECKPOINT" if checkpoint is not None else f"ANCHOR_{len(saved_anchors)}"
                ),
                "registration_state": result.state.value,
            },
        )
        save_project(confirmed, self._root)
        self._log(
            "corridor_anchor_saved",
            tower_id=tower_id,
            state=result.state.value,
            anchor_count=result.anchor_count,
            checkpoint_residual_m=result.checkpoint_residual_m,
        )
        return corridor_registration_projection(self)

    def tower_candidate_projection(self: Any, tower_id: str) -> dict[str, Any]:
        project = self._stage_project()
        scan = self._scan(project)
        if scan is None:
            raise ValueError("POINT_CLOUD_NOT_IMPORTED")
        tower = next((item for item in project.towers if item.tower_id == tower_id), None)
        if tower is None:
            raise ValueError("TOWER_NOT_FOUND")
        registration = _registration_from_scan(scan)
        if registration is None:
            prepare_corridor_registration(self)
            project = self._stage_project()
            scan = self._scan(project)
            registration = _registration_from_scan(scan) if scan is not None else None
        if registration is None or registration.transform_2d is None:
            raise ValueError("REGISTRATION_TRANSFORM_REQUIRED")
        if not registration.unique_rigid_transform:
            raise ValueError("REGISTRATION_UNIQUE_TRANSFORM_REQUIRED")
        gis_xyz = _tower_gis_position(tower)
        seed_xy = apply_registration_xy(
            registration,
            gis_xyz,
            direction="GIS_TO_POINT_CLOUD",
        )
        local = self.refine_tower_pick({"x": seed_xy[0], "y": seed_xy[1]})
        local_xyz = local.get("local", {}).get("xyz") or []
        candidates = rank_tower_candidates(local_xyz, seed_xy)
        return {
            "tower_id": tower.tower_id,
            "external_asset_id": tower.external_asset_id,
            "display_name": tower.display_name,
            "seed_xy_point_cloud": list(seed_xy),
            "registration_state": registration.state.value,
            "registration_ready_for_science": registration.ready_for_science,
            "candidates": [item.to_dict() for item in candidates],
            "local": local.get("local"),
            "diagnostic_timings": local.get("diagnostic_timings"),
            "authority": "PRESENTATION_ASSISTANCE_ONLY",
            "manual_fallback_available": True,
        }

    def confirm_gis_tower_candidate(self: Any, payload: Any) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise ValueError("TOWER_CONFIRMATION_PAYLOAD_REQUIRED")
        tower_id = str(payload.get("tower_id") or "")
        candidate = payload.get("candidate_xyz")
        if not isinstance(candidate, (list, tuple)) or len(candidate) != 3:
            raise ValueError("CANDIDATE_XYZ_REQUIRED")
        refined = self.refine_tower_pick(
            {"x": candidate[0], "y": candidate[1], "z": candidate[2]}
        )
        project = self._stage_project()
        scan = self._scan(project)
        if scan is None:
            raise ValueError("POINT_CLOUD_NOT_IMPORTED")
        tower_payload = refined["tower"]
        confirmed = confirm_gis_tower(
            project,
            tower_id,
            (
                float(tower_payload["x"]),
                float(tower_payload["y"]),
                float(tower_payload["z"]),
            ),
            _coordinate_from_scan(scan),
            evidence={
                "candidate_id": payload.get("candidate_id"),
                "candidate_authority": "PRESENTATION_ASSISTANCE_ONLY",
                "raw_local_refinement": tower_payload.get("refinement"),
                "picked_source_xyz": tower_payload.get("picked_source_xyz"),
                "local_point_count": tower_payload.get("local_point_count"),
            },
        )
        save_project(confirmed, self._root)
        self._log(
            "gis_tower_confirmed",
            tower_id=tower_id,
            candidate_id=payload.get("candidate_id"),
            x=tower_payload["x"],
            y=tower_payload["y"],
            z=tower_payload["z"],
        )
        return project_projection_with_corridor(self)

    def span_readiness_projection(self: Any) -> list[dict[str, Any]]:
        return _span_readiness(self._stage_project())

    @wraps(original_project_projection)
    def project_projection_with_corridor(self: Any) -> dict[str, Any]:
        projection = original_project_projection(self)
        try:
            project = self._stage_project()
        except (FileNotFoundError, ValueError):
            return projection
        readiness = {item["span_id"]: item for item in _span_readiness(project)}
        tower_records = []
        for tower in project.towers:
            row = tower.to_dict()
            row["confirmation_state"] = "CONFIRMED" if _tower_ready(tower) else "GIS_SEED_ONLY"
            tower_records.append(row)
        span_records = []
        for span in project.spans:
            row = span.to_dict()
            row["readiness"] = readiness.get(span.span_id, {"status": "BLOCKED"})
            span_records.append(row)
        projection["tower_records"] = tower_records
        projection["span_records"] = span_records
        projection["corridor_registration"] = corridor_registration_projection(self)
        projection["gis_supported_formats"] = ["KML", "KMZ", "CSV", "XLSX", "GEOJSON"]
        return projection

    @wraps(original_start_batch)
    def start_batch_with_readiness(
        self: Any,
        span_ids: list[str],
        *,
        mode: str = "REAL",
        execution_mode: str = "AUTO",
        acceleration_mode: str = "AUTO",
    ) -> Any:
        execution_mode = self._validate_execution_mode(execution_mode)
        project = self._stage_project()
        readiness = {item["span_id"]: item for item in _span_readiness(project)}
        blocked = [
            str(span_id)
            for span_id in span_ids
            if readiness.get(str(span_id), {}).get("status") != "READY"
        ]
        if blocked:
            raise ValueError(
                "SPAN_ENDPOINT_CONFIRMATION_REQUIRED:" + ",".join(blocked)
            )
        return original_start_batch(
            self,
            span_ids,
            mode=mode,
            execution_mode=execution_mode,
            acceleration_mode=acceleration_mode,
        )

    DesktopWorkflow.corridor_registration_projection = corridor_registration_projection
    DesktopWorkflow.prepare_corridor_registration = prepare_corridor_registration
    DesktopWorkflow.set_corridor_anchor = set_corridor_anchor
    DesktopWorkflow.tower_candidate_projection = tower_candidate_projection
    DesktopWorkflow.confirm_gis_tower_candidate = confirm_gis_tower_candidate
    DesktopWorkflow.span_readiness_projection = span_readiness_projection
    DesktopWorkflow.project_projection = project_projection_with_corridor
    DesktopWorkflow.start_batch = start_batch_with_readiness
    _INSTALLED = True


__all__ = ["install_corridor_extensions"]
