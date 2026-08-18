"""Serve the P3A Giro3D + COPC viewer (Candidate B) over loopback only.

Mirrors the P3A Potree server's safety posture: strictly loopback
(``127.0.0.1``, no ``--host`` override), Windows-first exclusive bind
(``SO_EXCLUSIVEADDRUSE``) so an occupied port fails deterministically, and only
virtual roots that cannot escape to the protected source.

Virtual routes::

    /vendor/        -> viewer_poc/giro3d/vendor/            (Giro3D bundle)
    /laz-perf/      -> viewer_poc/giro3d/vendor/laz-perf/   (lazperf WASM)
    /pointclouds/   -> cache/p3a_giro3d/dataset/            (the derived COPC)
    /               -> viewer_poc/giro3d/                   (viewer page)
    /p3a_giro3d_healthz -> JSON status (active COPC presence)

When launched with ``--tower-project-root`` the server additionally enables the
local JSON tower-review API (V5A2/V5A3 + V5A3 ordered).  The API is opt-in so
the existing comparison viewers are unchanged.  The PRIMARY ordered workflow is
``/api/tower-review/ordered/{select,preview,confirm,reselect,cancel}``: the
Owner clicks a real tower, bounded local FULL_RESOLUTION_RAW membership
segmentation runs, the exact member points are previewed (VIEWER-ONLY capped
sample), and only an explicit CONFIRM allocates the next canonical ``T###`` in
confirmation (corridor) order.  Safety posture for every mutation:

- open/resolve the actual authoritative project through the writable session;
- validate the current generation, source identity, and candidate-set digest;
- use only the existing project APIs (never allocate a ``T###`` directly);
- accept no arbitrary filesystem paths from the browser;
- return structured JSON errors.

Usage::

    python scripts/serve_p3a_giro3d.py
    python scripts/serve_p3a_giro3d.py --port 8091 --no-open
    python scripts/serve_p3a_giro3d.py --config poc_config_full.json \\
        --dataset-root cache/giro3d_full_source/dataset \\
        --tower-project-root state/p3a_tower_review \\
        --tower-tile-cache cache/tower_raw_tiles \\
        --tower-detect-ground <classified_hag.las>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import mimetypes
import re
import secrets
import socket
import sys
import urllib.parse
from datetime import datetime
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
# The launcher may be executed directly (sys.path[0] = scripts/) rather than as
# a module, so make the repository root importable for the lazy API imports.
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from scripts.tree_viewer_api import TreeViewerApiMixin  # noqa: E402

VIEWER_ROOT = _REPOSITORY_ROOT / "viewer_poc" / "giro3d"
VENDOR_ROOT = VIEWER_ROOT / "vendor"
LAZPERF_ROOT = VENDOR_ROOT / "laz-perf"
DATASET_ROOT = _REPOSITORY_ROOT / "cache" / "p3a_giro3d" / "dataset"
# Viewer config served at /poc_config.json (ROI POC or FULL-SOURCE gate).
_CONFIG_FILE = "poc_config.json"

# The single canonical P3A binding.  Loopback-only; never broaden.
P3A_BIND_HOST = "127.0.0.1"

# Bounded Viewer-only preview of authoritative tree-member points.  Science is
# always based on the complete FULL_RESOLUTION_RAW membership; the Viewer never
# sends/renders every member (S005 spans hold tens of millions).
TREE_PREVIEW_SAMPLE_CAP = 40_000

# Confirmed-tower membership reinspection uses a lighter deterministic display
# cap than the pending-preview cap, keeping the response small and responsive.
CONFIRMED_MEMBERSHIP_PREVIEW_SAMPLE_MAX = 8000

def _routes() -> tuple[tuple[str, Path], ...]:
    """Virtual roots, computed at request time so the dataset root (ROI POC or
    FULL-SOURCE gate) can be switched per launcher via --dataset-root."""
    return (
        ("/vendor/", VENDOR_ROOT),
        ("/laz-perf/", LAZPERF_ROOT),
        ("/pointclouds/", DATASET_ROOT),
        ("/", VIEWER_ROOT),
    )

_EXTRA_MIME = {
    ".js": "text/javascript",
    ".mjs": "text/javascript",
    ".json": "application/json",
    ".wasm": "application/wasm",
    ".laz": "application/octet-stream",
    ".glsl": "text/plain",
}


def _guess_type(path: Path) -> str:
    mime = mimetypes.guess_type(str(path))[0]
    if mime:
        return mime
    return _EXTRA_MIME.get(path.suffix.lower(), "application/octet-stream")


def active_copc_name() -> str | None:
    """Derive the active COPC filename from the served viewer config's copcUrl."""
    try:
        cfg = json.loads((VIEWER_ROOT / _CONFIG_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    url = cfg.get("copcUrl") if isinstance(cfg, dict) else None
    if isinstance(url, str) and url.endswith(".laz"):
        return url.rsplit("/", 1)[-1]
    return None


class _ExclusiveBindServer(ThreadingHTTPServer):
    allow_reuse_address = False

    def server_bind(self) -> None:
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()


# ---------------------------------------------------------------------------
# V5A2/V5A3 tower-review local API (opt-in; JSON-only; loopback only)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TowerReviewConfig:
    """Explicit project/source configuration for the tower-review API."""

    project_root: Path
    tile_cache: Path
    detect_ground_las: Path | None = None
    detect_origin_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    operator_identity: str = "owner"


class _ApiError(Exception):
    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message


class TowerReviewApi(TreeViewerApiMixin):
    """Thin HTTP-facing boundary; all project work goes through the session."""

    def __init__(self, config: TowerReviewConfig) -> None:
        self.config = config
        from algorithms.towers.ordered import OrderedPreviewStore

        self.ordered_store = OrderedPreviewStore()
        from algorithms.spans.project_orchestrator import ProjectSpanOrchestrator

        # One shared orchestrator is used by synchronize/current/pending/retry
        # actions; HTTP handlers never contain science logic.
        self.span_orchestrator = ProjectSpanOrchestrator()

    def _opened(self):
        from core.project.session import open_project_read_only

        return open_project_read_only(self.config.project_root)

    def _opened_writable(self):
        from core.project.session import open_project_writable

        return open_project_writable(self.config.project_root)

    def _tree_error(self, status: int, code: str, message: str) -> _ApiError:
        return _ApiError(status, code, message)

    def _tile_cache(self, session) -> Any:
        from algorithms.towers.raw_access import RawTileCache, RawTileCacheConfig
        from core.contracts.project import SourceIdentityBinding

        binding = SourceIdentityBinding.from_source_identity(
            session.project.source_reference.source_identity
        )
        return RawTileCache.load(
            self.config.tile_cache,
            source_binding=binding,
            config=RawTileCacheConfig(),
        )

    # ------------------------------------------------------------- actions
    def state(self) -> dict[str, Any]:
        from algorithms.towers.ordered import ordered_review_state

        with self._opened() as session:
            return ordered_review_state(session)

    def desktop_project(self) -> dict[str, Any]:
        """Desktop lifecycle projection, including deliberately empty projects."""
        from desktop.project_management import describe_project

        return describe_project(self.config.project_root)

    # ------------------------------------------------------------- project Spans
    def _span_projection(self, session, span) -> dict[str, Any]:
        from core.contracts.project import AnalysisStatus

        state = "READY" if span.analysis.state is AnalysisStatus.NOT_STARTED else span.analysis.state.value
        towers = {entry.tower_id: entry.current for entry in session.tower_registry.records}
        projection = {
            "span_id": span.span_id,
            "display_name": span.display_name,
            "group_id": span.group_id,
            "within_group_order": span.within_group_order,
            "from_tower_id": span.from_tower_id,
            "to_tower_id": span.to_tower_id,
            "from_tower_revision": span.from_tower_revision,
            "to_tower_revision": span.to_tower_revision,
            "from_tower_geometry": towers[span.from_tower_id].refined_geometry.to_dict(),
            "to_tower_geometry": towers[span.to_tower_id].refined_geometry.to_dict(),
            "status": state,
            "active_run_id": span.analysis.active_run_id,
            "historical_run_ids": list(span.analysis.historical_run_ids),
            "geometry": span.geometry.to_dict(),
            "artifact_index_reference": (
                span.artifact_index_reference.to_dict()
                if span.artifact_index_reference is not None else None
            ),
        }
        try:
            artifact, run_dir = self._read_span_run_index(session, span)
            projection["active_run_id"] = run_dir.name
            projection["registry_active_run_id"] = span.analysis.active_run_id
            projection["summary"] = dict(artifact.get("summary") or {})
            diagnostics = artifact.get("diagnostics") or {}
            clearance = diagnostics.get("clearance") if isinstance(diagnostics, dict) else None
            if isinstance(clearance, dict):
                for key in ("total_wires", "danger_threshold_m", "danger_point_count"):
                    if key in clearance:
                        projection["summary"][{"total_wires": "wire_count"}.get(key, key)] = clearance[key]
        except _ApiError:
            projection["summary"] = {}
        return projection

    def _spans_payload(self, session) -> dict[str, Any]:
        spans = sorted(session.span_registry.records, key=lambda item: int(item.span_id[1:]))
        return {
            "project_id": session.project.project_id,
            "generation_id": session.generation_id,
            "spans": [self._span_projection(session, span) for span in spans],
        }

    def spans_list(self) -> dict[str, Any]:
        with self._opened() as session:
            return self._spans_payload(session)

    def spans_get(self, span_id: str) -> dict[str, Any]:
        with self._opened() as session:
            try:
                span = next(item for item in session.span_registry.records if item.span_id == span_id)
            except StopIteration as exc:
                raise _ApiError(HTTPStatus.NOT_FOUND, "SPAN_NOT_FOUND", f"unknown Span {span_id}") from exc
            return {"span": self._span_projection(session, span)}

    # ------------------------------------------------------------------ trees
    def _resolve_span_record(self, session, span_id: str):
        try:
            return next(
                item for item in session.span_registry.records if item.span_id == span_id
            )
        except StopIteration as exc:
            raise _ApiError(
                HTTPStatus.NOT_FOUND, "SPAN_NOT_FOUND", f"unknown Span {span_id}"
            ) from exc

    def _read_span_run_index(self, session, span):
        """Resolve the latest complete, digest-checked run for one span.

        The registry can legitimately point at a newer in-progress/stale run while
        a prior completed run remains the last renderable result.  The Viewer must
        never mix those artifacts: it selects one terminal index only when all
        three scientific artifacts validate, and returns its single run directory
        to every Tree/Wire/Clearance/Danger endpoint.
        """
        reference = span.artifact_index_reference
        candidates: list[Path] = []
        if reference is not None:
            candidates.append(session.project_root / Path(reference.relative_path))
        runs_root = session.project_root / "artifacts" / "project-runs" / "groups" / span.group_id / span.span_id
        if runs_root.is_dir():
            candidates.extend(runs_root.glob("*/terminal-artifact-index.json"))
        valid: list[tuple[float, dict[str, Any], Path]] = []
        for path in {item.resolve() for item in candidates if item.is_file()}:
            try:
                data = path.read_bytes()
                index = json.loads(data.decode("utf-8"))
                artifacts = index.get("artifacts") if isinstance(index, dict) else None
                if not isinstance(artifacts, dict) or not all(key in artifacts for key in ("tree", "wire", "clearance")):
                    continue
                for key in ("tree", "wire", "clearance"):
                    artifact_ref = artifacts[key]
                    artifact_path = session.project_root / Path(artifact_ref["relative_path"])
                    artifact_data = artifact_path.read_bytes()
                    if len(artifact_data) != artifact_ref["byte_size"] or hashlib.sha256(artifact_data).hexdigest() != artifact_ref["sha256"]:
                        raise ValueError("artifact digest mismatch")
                summary = index.get("summary") or {}
                if not isinstance(summary.get("tree_count"), int):
                    continue
                valid.append((path.stat().st_mtime, index, path.parent))
            except (OSError, KeyError, TypeError, ValueError, UnicodeDecodeError, json.JSONDecodeError):
                continue
        if not valid:
            raise _ApiError(HTTPStatus.CONFLICT, "SPAN_HAS_NO_COMPLETE_RUN", f"{span.span_id} has no complete verified run")
        _mtime, index, run_dir = max(valid, key=lambda item: item[0])
        return index, run_dir

    @staticmethod
    def _read_index_artifact(session, index: dict[str, Any], key: str) -> dict[str, Any]:
        reference = (index.get("artifacts") or {}).get(key)
        if not isinstance(reference, dict):
            raise _ApiError(HTTPStatus.CONFLICT, "SPAN_ARTIFACT_MISSING", f"{key} artifact is unavailable")
        try:
            data = (session.project_root / Path(reference["relative_path"])).read_bytes()
            if len(data) != reference["byte_size"] or hashlib.sha256(data).hexdigest() != reference["sha256"]:
                raise ValueError("digest mismatch")
            payload = json.loads(data.decode("utf-8"))
        except (OSError, KeyError, ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise _ApiError(HTTPStatus.CONFLICT, "SPAN_ARTIFACT_INVALID", f"{key} artifact is invalid") from exc
        if not isinstance(payload, dict):
            raise _ApiError(HTTPStatus.CONFLICT, "SPAN_ARTIFACT_INVALID", f"{key} artifact is invalid")
        return payload

    def spans_wire(self, span_id: str) -> dict[str, Any]:
        with self._opened() as session:
            span = self._resolve_span_record(session, span_id)
            index, run_dir = self._read_span_run_index(session, span)
            wire = self._read_index_artifact(session, index, "wire")
            if wire.get("schema") != "fast-baseline-span-wire-v1":
                raise _ApiError(HTTPStatus.CONFLICT, "WIRE_ARTIFACT_INVALID", "unsupported confirmed wire artifact")
            strands = [item for item in wire.get("strands") or [] if item.get("status") == "CONFIRMED"]
            return {
                "span_id": span_id,
                "run_id": run_dir.name,
                "authoritative": True,
                "strand_count": len(strands),
                "strands": strands,
            }

    def spans_wire_preview(self, span_id: str, wire_id: str) -> dict[str, Any]:
        import numpy as np

        target_id = int(wire_id)
        with self._opened() as session:
            span = self._resolve_span_record(session, span_id)
            index, run_dir = self._read_span_run_index(session, span)
            origin = self._span_tree_origin(session, index)
            wire = self._read_index_artifact(session, index, "wire")
        strand = next((item for item in wire.get("strands") or [] if int(item.get("wire_id", -1)) == target_id and item.get("status") == "CONFIRMED"), None)
        if strand is None:
            raise _ApiError(HTTPStatus.NOT_FOUND, "CONFIRMED_WIRE_NOT_FOUND", "unknown confirmed wire")
        arrays_path = run_dir / "wire" / "w1_arrays.npz"
        try:
            with np.load(arrays_path, allow_pickle=False) as arrays:
                ids = np.asarray(arrays["wire_id"], dtype=np.int32)
                position = int(np.flatnonzero(ids == target_id)[0])
                offsets = np.asarray(arrays["sampled_curve_offsets"], dtype=np.int64)
                points = np.asarray(arrays["sampled_curve_xyz"], dtype=np.float32)[offsets[position]:offsets[position + 1]]
        except (OSError, KeyError, IndexError, ValueError) as exc:
            raise _ApiError(HTTPStatus.CONFLICT, "CONFIRMED_WIRE_GEOMETRY_MISSING", "confirmed wire geometry is unavailable") from exc
        points += np.asarray(origin, dtype=np.float32)
        return {"span_id": span_id, "run_id": run_dir.name, "wire_id": target_id, "authoritative": True, "strand": strand, "sampled_xyz_absolute_m": points.tolist()}

    def spans_danger_points(self, span_id: str) -> dict[str, Any]:
        with self._opened() as session:
            span = self._resolve_span_record(session, span_id)
            index, run_dir = self._read_span_run_index(session, span)
            origin = self._span_tree_origin(session, index)
            clearance = self._read_index_artifact(session, index, "clearance")
        if clearance.get("schema") != "treeline-span-clearance-result-v1":
            raise _ApiError(HTTPStatus.CONFLICT, "DANGER_ARTIFACT_INVALID", "unsupported clearance artifact")
        points = clearance.get("danger_points") or []
        absolute = []
        for point in points:
            xyz = self._local_to_absolute(point.get("xyz_local_m"), origin)
            if xyz is not None:
                absolute.append(xyz)
        diagnostics = index.get("diagnostics") or {}
        details = diagnostics.get("clearance") if isinstance(diagnostics, dict) else {}
        return {
            "span_id": span_id,
            "run_id": run_dir.name,
            "authoritative": True,
            "risk_threshold_m": details.get("danger_threshold_m"),
            "danger_point_count": len(absolute),
            "danger_tree_count": (index.get("summary") or {}).get("risk_counts", {}).get("below_threshold"),
            "points_xyz_absolute_m": absolute,
        }

    def _span_tree_origin(self, session, index: dict):
        """The tree-local frame origin (local -> absolute offset) from source prep."""
        provenance = index.get("provenance") or {}
        prep_ref = provenance.get("source_preparation", {}).get("manifest_reference")
        if isinstance(prep_ref, str):
            prep_path = session.project_root / Path(prep_ref)
            try:
                prep = json.loads(prep_path.read_bytes())
                origin = prep.get("origin_xyz")
                if isinstance(origin, (list, tuple)) and len(origin) == 3:
                    return tuple(float(value) for value in origin)
            except (OSError, ValueError, TypeError):
                pass
        # A deterministic safe fallback: the source absolute frame (local == absolute).
        return (0.0, 0.0, 0.0)

    @staticmethod
    def _tree_category(record: dict) -> str:
        core_fraction = record.get("core_fraction", 0.0) or 0.0
        review = bool(record.get("review", False))
        if core_fraction > 0.0 and not review:
            return "CORE"
        return "HALO / REVIEW"

    def _local_to_absolute(self, local, origin):
        if not isinstance(local, (list, tuple)) or len(local) != 3:
            return None
        return [
            float(local[0]) + origin[0],
            float(local[1]) + origin[1],
            float(local[2]) + origin[2],
        ]

    @staticmethod
    def _tree_instance_color(color_key: str) -> str:
        """Stable high-contrast color derived from an identity presentation key."""

        palette = (
            "#00bcd4", "#ff7043", "#ab47bc", "#66bb6a", "#ffa726", "#42a5f5",
            "#ec407a", "#8d6e63", "#d4e157", "#26a69a", "#7e57c2", "#ef5350",
            "#29b6f6", "#9ccc65", "#ffca28", "#26c6da", "#5c6bc0", "#f06292",
            "#8bc34a", "#ff8a65", "#7cb342", "#00897b", "#ffb300", "#7986cb",
        )
        digest = hashlib.sha256(color_key.encode("utf-8")).digest()
        return palette[int.from_bytes(digest[:4], "big") % len(palette)]

    @staticmethod
    def _tree_identity_presentation(record: dict[str, Any]) -> dict[str, Any]:
        """Read-only display/prediction policy for resolved, review, and legacy rows."""

        persistent_id = record.get("persistent_tree_id")
        if isinstance(persistent_id, str) and persistent_id.strip():
            persistent_id = persistent_id.strip()
            return {
                "display_id": persistent_id,
                "presentation": "PERSISTENT",
                "color_key": f"persistent:{persistent_id}",
                "persistent_tree_id": persistent_id,
                "candidate_persistent_tree_ids": [],
                "prediction_eligible": True,
                "needs_review": False,
            }
        candidates = sorted({
            item.strip()
            for item in record.get("candidate_persistent_tree_ids") or []
            if isinstance(item, str) and item.strip()
        })
        identity_state = record.get("identity_state", "LEGACY_RUN_LOCAL")
        if identity_state == "MERGE_AMBIGUOUS_REVIEW":
            return {
                "display_id": "未绑定实体",
                "presentation": "MERGE REVIEW",
                "color_key": "merge-review:" + "|".join(candidates),
                "persistent_tree_id": None,
                "candidate_persistent_tree_ids": candidates,
                "prediction_eligible": False,
                "needs_review": True,
            }
        return {
            "display_id": f"Legacy Tree {record.get('global_tree_id')}",
            "presentation": "LEGACY",
            "color_key": f"legacy:{record.get('global_tree_id')}",
            "persistent_tree_id": None,
            "candidate_persistent_tree_ids": candidates,
            "prediction_eligible": False,
            "needs_review": False,
        }

    def tree_spans_list(self) -> dict[str, Any]:
        """Minimal read-only Span selector projection for the Tree Viewer."""

        with self._opened() as session:
            spans = sorted(session.span_registry.records, key=lambda item: item.span_id)
            return {
                "spans": [
                    {"span_id": span.span_id, "display_name": span.display_name}
                    for span in spans
                ]
            }

    def spans_trees(self, span_id: str) -> dict[str, Any]:
        """Read-only final reconciled Tree list for the current Span run."""
        with self._opened() as session:
            span = self._resolve_span_record(session, span_id)
            index, run_dir = self._read_span_run_index(session, span)
            origin = self._span_tree_origin(session, index)
            reconc_path = run_dir / "reconciliation.json"
            try:
                reconciliation = json.loads(reconc_path.read_bytes())
            except OSError as exc:
                raise _ApiError(
                    HTTPStatus.CONFLICT, "SPAN_RECONCILIATION_MISSING",
                    "reconciliation artifact missing",
                ) from exc
            final_trees = reconciliation.get("global_tree_records") or []
            clearance_by_tree = {}
            clearance_path = run_dir / "clearance.json"
            if clearance_path.is_file():
                try:
                    clearance = json.loads(clearance_path.read_bytes())
                    for rec in clearance.get("records") or []:
                        clearance_by_tree[rec.get("global_tree_id")] = rec
                except (OSError, ValueError):
                    clearance_by_tree = {}
            tree_rows = []
            for record in sorted(final_trees, key=lambda item: (
                int(item["span_order_index"])
                if item.get("span_order_index") is not None else 2**31 - 1,
                int(item.get("global_tree_id", -1)),
            )):
                gid = record.get("global_tree_id")
                clearance_rec = clearance_by_tree.get(gid)
                apex_local = record.get("apex_xy_local_m")
                identity = self._tree_identity_presentation(record)
                row = {
                    "global_tree_id": gid,
                    "persistent_tree_id": identity["persistent_tree_id"],
                    "display_id": identity["display_id"],
                    "external_name": record.get("external_name"),
                    "identity_state": record.get("identity_state", "LEGACY_RUN_LOCAL"),
                    "identity_presentation": identity["presentation"],
                    "candidate_persistent_tree_ids": identity["candidate_persistent_tree_ids"],
                    "instance_color": self._tree_instance_color(identity["color_key"]),
                    "prediction_identity": {
                        "persistent_tree_id": identity["persistent_tree_id"],
                        "global_tree_id": gid,
                        "needs_review": identity["needs_review"],
                        "eligible_for_long_term_history": identity["prediction_eligible"],
                    },
                    "span_id": span_id,
                    "span_name": record.get("span_name"),
                    "primary_span_tree_id": record.get("primary_span_tree_id"),
                    "span_order_index": record.get("span_order_index"),
                    "longitudinal_s_m": record.get("longitudinal_s_m"),
                    "lateral_n_m": record.get("lateral_n_m"),
                    "merged_span_tree_keys": record.get("merged_span_tree_keys", []),
                    "full_resolution_point_count": record.get("full_resolution_point_count"),
                    "apex_height_m": record.get("apex_height_m"),
                    "apex_xyz_local_m": list(apex_local) if apex_local else None,
                    "apex_xyz_absolute_m": self._local_to_absolute(apex_local, origin),
                    "core_fraction": record.get("core_fraction", 0.0),
                    "core_point_count": record.get("core_point_count", 0),
                    "halo_point_count": record.get("halo_point_count", 0),
                    "review": bool(record.get("review", False)),
                    "category": self._tree_category(record),
                    "minimum_clearance_m": (
                        clearance_rec.get("minimum_clearance_m") if clearance_rec else None
                    ),
                    "danger_point_count": (
                        clearance_rec.get("danger_point_count", 0) if clearance_rec else None
                    ),
                    "dangerous": bool(clearance_rec and clearance_rec.get("danger_point_count", 0)),
                    "clearance": (
                        {
                            "minimum_clearance_m": clearance_rec.get("minimum_clearance_m"),
                            "status": clearance_rec.get("status"),
                            "nearest_tree_original_index": (
                            clearance_rec.get("nearest_tree_original_index")
                        ),
                            "nearest_wire_id": clearance_rec.get("nearest_wire_id"),
                            "tree_xyz_local_m": clearance_rec.get("nearest_tree_xyz_local_m"),
                            "wire_xyz_local_m": clearance_rec.get("nearest_wire_xyz_local_m"),
                            "tree_xyz_absolute_m": self._local_to_absolute(
                                clearance_rec.get("nearest_tree_xyz_local_m"), origin
                            ),
                            "wire_xyz_absolute_m": self._local_to_absolute(
                                clearance_rec.get("nearest_wire_xyz_local_m"), origin
                            ),
                        }
                        if clearance_rec else None
                    ),
                }
                tree_rows.append(row)
            core_count = sum(1 for row in tree_rows if row["category"] == "CORE")
            halo_review_count = len(tree_rows) - core_count
            return {
                "span_id": span_id,
                "run_id": run_dir.name,
                "origin_xyz": list(origin),
                "tree_count": len(tree_rows),
                "core_count": core_count,
                "halo_review_count": halo_review_count,
                "trees": tree_rows,
            }

    def spans_tree_preview(self, span_id: str, tree_id: str) -> dict[str, Any]:
        """Read-only bounded authoritative member preview for one final Tree."""
        import numpy as np

        with self._opened() as session:
            span = self._resolve_span_record(session, span_id)
            index, run_dir = self._read_span_run_index(session, span)
            origin = self._span_tree_origin(session, index)
            reconc_path = run_dir / "reconciliation.json"
            try:
                reconciliation = json.loads(reconc_path.read_bytes())
            except OSError as exc:
                raise _ApiError(
                    HTTPStatus.CONFLICT, "SPAN_RECONCILIATION_MISSING",
                    "reconciliation artifact missing",
                ) from exc
            records = reconciliation.get("global_tree_records") or []
            target = next(
                (record for record in records
                if int(record.get("global_tree_id", -1)) == int(tree_id)),
                None,
            )
            if target is None:
                raise _ApiError(
                    HTTPStatus.NOT_FOUND, "TREE_NOT_FOUND",
                    f"Tree {tree_id} not in {span_id} reconciliation",
                )
            identity = self._tree_identity_presentation(target)
            span_tree_ids = [
                int(key[1])
                for key in target.get("merged_span_tree_keys") or []
                if isinstance(key, (list, tuple)) and len(key) >= 2
            ]
            if not span_tree_ids:
                primary = target.get("primary_span_tree_id")
                span_tree_ids = [int(primary)] if primary is not None else []
            members_path = run_dir / "tree" / "tree_members.npz"
            if not members_path.is_file():
                raise _ApiError(
                    HTTPStatus.CONFLICT, "TREE_MEMBERS_MISSING", "tree member artifact missing",
                )
            members = np.load(str(members_path), mmap_mode="r", allow_pickle=True)
            tree_ids = np.asarray(members["tree_id"], dtype=np.int32)
            mask = np.isin(tree_ids, span_tree_ids)
            rows = np.flatnonzero(mask)
            full_count = int(len(rows))
            if full_count:
                sample = np.asarray(members["xyz_local_m"], dtype=np.float64)[rows]
            else:
                sample = np.empty((0, 3), dtype=np.float64)
            cap = TREE_PREVIEW_SAMPLE_CAP
            if full_count > cap:
                chosen = np.linspace(0, full_count - 1, cap, dtype=np.int64)
                sample = sample[chosen]
            sample_absolute = sample + np.asarray(origin, dtype=np.float64)
            bounds = None
            centroid = None
            if full_count:
                local_min = np.min(sample, axis=0)
                local_max = np.max(sample, axis=0)
                bounds = {
                    "min_xyz_absolute_m": list(local_min + np.asarray(origin)),
                    "max_xyz_absolute_m": list(local_max + np.asarray(origin)),
                }
                centroid = list(np.mean(sample_absolute, axis=0))
            return {
                "span_id": span_id,
                "run_id": run_dir.name,
                "global_tree_id": target.get("global_tree_id"),
                "persistent_tree_id": identity["persistent_tree_id"],
                "external_name": target.get("external_name"),
                "identity_state": target.get("identity_state", "LEGACY_RUN_LOCAL"),
                "candidate_persistent_tree_ids": identity["candidate_persistent_tree_ids"],
                "prediction_identity": {
                    "persistent_tree_id": identity["persistent_tree_id"],
                    "global_tree_id": target.get("global_tree_id"),
                    "needs_review": identity["needs_review"],
                    "eligible_for_long_term_history": identity["prediction_eligible"],
                },
                "origin_xyz": list(origin),
                "full_member_count": full_count,
                "sample_count": int(len(sample)),
                "sampled_xyz_absolute_m": sample_absolute.tolist(),
                "bounds_absolute_m": bounds,
                "centroid_xyz_absolute_m": centroid,
            }

    def spans_synchronize(self) -> dict[str, Any]:
        with self._opened_writable() as session:
            result = self.span_orchestrator.synchronize(session)
            return {
                "created_span_ids": list(result.created_span_ids),
                "preserved_span_ids": list(result.preserved_span_ids),
                "migrated_span_ids": list(result.migrated_span_ids),
                **self._spans_payload(session),
            }

    def spans_analyze(self, span_id: str) -> dict[str, Any]:
        with self._opened_writable() as session:
            outcome = self.span_orchestrator.analyze_span(session, span_id)
            return {"outcome": {
                "span_id": outcome.span_id,
                "status": outcome.status,
                "run_id": outcome.run_id,
                "reused": outcome.reused,
                "error": outcome.error,
                "summary": outcome.summary,
            }, **self._spans_payload(session)}

    def spans_prediction(self, span_id: str, request: dict[str, Any] | None = None) -> dict[str, Any]:
        """Build a strict downstream package and invoke the shipped local core.

        This endpoint intentionally reads only terminal science projections.  It
        does not change a Span's analysis state or synthesize unavailable model
        inputs.
        """
        from algorithms.prediction.local_provider import (
            LocalPredictionProvider,
            build_prediction_input,
        )

        request = request or {}
        mode = str(request.get("prediction_mode", "baseline")).strip().lower()
        if mode not in {"baseline", "formula", "ai"}:
            raise _ApiError(
                HTTPStatus.BAD_REQUEST,
                "PREDICTION_MODE_INVALID",
                "prediction_mode must be baseline, formula, or ai",
            )
        trees = self.spans_trees(span_id)
        with self._opened() as session:
            span = self._resolve_span_record(session, span_id)
            geometry = span.geometry.to_dict()
            package = build_prediction_input(
                project_id=session.project.project_id,
                group_id=span.group_id,
                span_id=span_id,
                run_id=trees["run_id"],
                trees=trees["trees"],
                span_context={
                    "tower_pair": f"{span.from_tower_id}-{span.to_tower_id}",
                    "span_length_m": geometry.get("length_m"),
                },
            )
        result = LocalPredictionProvider().run(package, prediction_mode=mode)
        artifact = {
            "artifact_type": "treeline_prediction_result",
            "source_science_run_id": package.source_science_run_id,
            "input": package.to_dict(),
            "result": result.to_dict(),
        }
        artifact_path = (
            self.config.project_root
            / "artifacts"
            / "prediction-runs"
            / "groups"
            / package.group_id
            / package.span_id
            / package.source_science_run_id
            / {
                "baseline": "baseline-rule-based.json",
                "formula": "formula-long-horizon.json",
                "ai": "temporal-ai-long-horizon.json",
            }[mode]
        )
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = artifact_path.with_suffix(".tmp")
        temporary_path.write_text(
            json.dumps(artifact, ensure_ascii=False, indent=2, allow_nan=False),
            encoding="utf-8",
        )
        temporary_path.replace(artifact_path)
        return {
            "input": package.to_dict(),
            "result": result.to_dict(),
            "input_tree_count": len(package.observations),
            "output_tree_count": len(result.entries),
            "failed_tree_count": len(result.failed_tree_ids),
            "prediction_mode": mode,
            "prediction_artifact": str(artifact_path.relative_to(self.config.project_root)),
        }

    def spans_work_order(self, span_id: str, request: dict[str, Any]) -> dict[str, Any]:
        """Render one selected, already-predicted Tree as printable A4 HTML."""
        from desktop.project_management import describe_project
        from desktop.work_order import render_work_order

        persistent_id = request.get("persistent_tree_id")
        mode = str(request.get("prediction_mode", "")).strip().lower()
        if not isinstance(persistent_id, str) or not persistent_id.strip():
            raise _ApiError(HTTPStatus.BAD_REQUEST, "TREE_SELECTION_REQUIRED", "请先选择一棵树。")
        if mode not in {"baseline", "formula", "ai"}:
            raise _ApiError(HTTPStatus.BAD_REQUEST, "PREDICTION_REQUIRED", "请先完成该树风险预测。")
        trees = self.spans_trees(span_id)
        tree = next((item for item in trees["trees"] if item.get("persistent_tree_id") == persistent_id), None)
        if tree is None:
            raise _ApiError(HTTPStatus.NOT_FOUND, "TREE_NOT_FOUND", "所选树不属于当前档距。")
        names = {"baseline": "baseline-rule-based.json", "formula": "formula-long-horizon.json", "ai": "temporal-ai-long-horizon.json"}
        artifact_path = (
            self.config.project_root / "artifacts" / "prediction-runs" / "groups"
            / next(item["group_id"] for item in self.spans_list()["spans"] if item["span_id"] == span_id)
            / span_id / trees["run_id"] / names[mode]
        )
        try:
            artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
            result = artifact["result"]
            entry = next(item for item in result["entries"] if item.get("persistent_tree_id") == persistent_id)
        except (OSError, KeyError, StopIteration, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise _ApiError(HTTPStatus.CONFLICT, "PREDICTION_REQUIRED", "请先完成该树风险预测。") from exc
        with self._opened() as session:
            span = self._resolve_span_record(session, span_id)
            span_projection = self._span_projection(session, span)
        today = datetime.now().strftime("%Y%m%d")
        safe_id = "".join(char if char.isalnum() or char in "-_" else "_" for char in persistent_id)
        filename = f"TreeLineWarning_风险工单_{safe_id}_{today}.pdf"
        document = {
            "project": describe_project(self.config.project_root), "span": span_projection,
            "tree": tree, "entry": entry,
            "work_order_id": f"TLW-{today}-{safe_id}",
        }
        return {"html": render_work_order(document), "suggested_filename": filename, "work_order": document}

    def spans_analyze_pending(self) -> dict[str, Any]:
        with self._opened_writable() as session:
            outcomes = self.span_orchestrator.analyze_pending(session)
            return {"outcomes": [_outcome_dict(item) for item in outcomes], **self._spans_payload(session)}

    def spans_retry_failed(self) -> dict[str, Any]:
        with self._opened_writable() as session:
            outcomes = self.span_orchestrator.retry_failed(session)
            return {"outcomes": [_outcome_dict(item) for item in outcomes], **self._spans_payload(session)}

    def detect(self, body: dict[str, Any]) -> dict[str, Any]:
        import time as _time

        from algorithms.poles.evidence import load_ground_evidence
        from algorithms.towers.automatic import (
            TowerAutomaticDetectorConfig,
            detect_tower_candidates,
        )
        from core.config.fast_baseline import FastBaselineConfig

        ground_path = self.config.detect_ground_las
        if ground_path is None or not ground_path.is_file():
            raise _ApiError(
                HTTPStatus.SERVICE_UNAVAILABLE,
                "DETECT_INPUT_UNAVAILABLE",
                "no compatible ground/HAG evidence LAS configured for detection",
            )
        requested_set_id = body.get("candidate_set_id")
        if requested_set_id is not None and type(requested_set_id) is not str:
            raise _ApiError(
                HTTPStatus.BAD_REQUEST, "BAD_CANDIDATE_SET_ID", "candidate_set_id must be a string"
            )
        started = _time.perf_counter()
        with self._opened_writable() as session:
            binding = session.project.source_reference.binding
            evidence = load_ground_evidence(
                ground_path,
                FastBaselineConfig(),
                origin_xyz=self.config.detect_origin_xyz,
            )
            candidate_set = detect_tower_candidates(
                evidence,
                FastBaselineConfig(),
                source_binding=binding,
                coordinate_context=binding.coordinate_context,
                project_id=session.project.project_id,
                candidate_set_id=requested_set_id or f"auto-{secrets.token_hex(8)}",
                generated_at=_utc_timestamp(),
                detector_config=TowerAutomaticDetectorConfig(),
            )
            session.persist_tower_candidate_set(candidate_set)
            return {
                "candidate_set_id": candidate_set.candidate_set_id,
                "candidate_set_digest": candidate_set.digest,
                "candidate_count": candidate_set.candidate_count,
                "detector_runtime_seconds": _time.perf_counter() - started,
                "detector_provenance": candidate_set.detector_provenance.to_dict(),
                "candidates": [
                    {
                        "candidate_id": candidate.candidate_id,
                        "candidate_xyz_absolute_m": list(
                            candidate.candidate_xyz_absolute_m
                        ),
                        "confidence": candidate.confidence,
                    }
                    for candidate in candidate_set.candidates
                ],
            }

    def _resolve_set(self, session, candidate_set_id: str, body: dict[str, Any]):
        from core.contracts.tower_candidates import CandidateSetReference

        actual = None
        for candidate_set in session.candidate_store.candidate_sets:
            if candidate_set.candidate_set_id == candidate_set_id:
                actual = candidate_set
                break
        if actual is None:
            raise _ApiError(
                HTTPStatus.NOT_FOUND,
                "CANDIDATE_SET_NOT_FOUND",
                f"candidate set {candidate_set_id} is absent from the project",
            )
        digest = body.get("candidate_set_digest")
        if type(digest) is not str:
            raise _ApiError(
                HTTPStatus.BAD_REQUEST,
                "CANDIDATE_SET_DIGEST_REQUIRED",
                "candidate_set_digest must be provided in the request body",
            )
        reference = CandidateSetReference(
            candidate_set_id=actual.candidate_set_id, candidate_set_digest=digest
        )
        if actual.reference != reference:
            raise _ApiError(
                HTTPStatus.CONFLICT,
                "CANDIDATE_SET_DIGEST_MISMATCH",
                "candidate_set_digest does not match the persisted candidate set",
            )
        return actual

    def _seed_from_body(self, body: dict[str, Any], name: str) -> tuple[float, float, float] | None:
        raw = body.get(name)
        if raw is None:
            return None
        if (
            type(raw) is not list
            or len(raw) != 3
            or any(type(value) is not int and type(value) is not float for value in raw)
        ):
            raise _ApiError(
                HTTPStatus.BAD_REQUEST,
                f"BAD_{name.upper()}",
                f"{name} must be a length-3 numeric array [x, y, z]",
            )
        return tuple(float(value) for value in raw)

    def reject(
        self, candidate_set_id: str, candidate_id: str, body: dict[str, Any]
    ) -> dict[str, Any]:
        from algorithms.towers.review_workflow import project_tower_review_state, reject_candidate

        reason = body.get("reason")
        if reason is not None and type(reason) is not str:
            raise _ApiError(HTTPStatus.BAD_REQUEST, "BAD_REASON", "reason must be a string")
        operator = body.get("operator_identity") or self.config.operator_identity
        with self._opened_writable() as session:
            candidate_set = self._resolve_set(session, candidate_set_id, body)
            event = reject_candidate(
                session,
                candidate_set,
                candidate_id,
                reason=reason or "rejected by reviewer",
                operator_identity=operator,
            )
            state = project_tower_review_state(session)
            return {
                "event": event.to_dict(),
                "state": state,
            }

    def accept(
        self, candidate_set_id: str, candidate_id: str, body: dict[str, Any]
    ) -> dict[str, Any]:
        from algorithms.towers.review_workflow import accept_candidate, project_tower_review_state

        operator = body.get("operator_identity") or self.config.operator_identity
        notes = body.get("notes")
        if notes is not None and type(notes) is not str:
            raise _ApiError(HTTPStatus.BAD_REQUEST, "BAD_NOTES", "notes must be a string")
        adjusted_seed = self._seed_from_body(body, "adjusted_seed_xyz")
        with self._opened_writable() as session:
            candidate_set = self._resolve_set(session, candidate_set_id, body)
            cache = self._tile_cache(session)
            result = accept_candidate(
                session,
                candidate_set,
                candidate_id,
                cache,
                review_adjusted_seed_xyz_absolute_m=adjusted_seed,
                operator_identity=operator,
                notes=notes,
            )
            state = project_tower_review_state(session)
            return {
                "result": result.to_dict(),
                "state": state,
            }

    def manual_add(self, body: dict[str, Any]) -> dict[str, Any]:
        from algorithms.towers.review_workflow import (
            manual_add_candidate_set,
            project_tower_review_state,
        )

        seed = self._seed_from_body(body, "xyz")
        if seed is None:
            raise _ApiError(
                HTTPStatus.BAD_REQUEST, "XYZ_REQUIRED", "xyz must be a length-3 numeric array"
            )
        reason = body.get("reason")
        if reason is not None and type(reason) is not str:
            raise _ApiError(HTTPStatus.BAD_REQUEST, "BAD_REASON", "reason must be a string")
        note = body.get("note")
        if note is not None and type(note) is not str:
            raise _ApiError(HTTPStatus.BAD_REQUEST, "BAD_NOTE", "note must be a string")
        operator = body.get("operator_identity") or self.config.operator_identity
        with self._opened_writable() as session:
            candidate_set = manual_add_candidate_set(
                session,
                project_id=session.project.project_id,
                source_binding=session.project.source_reference.binding,
                coordinate_context=session.project.source_reference.coordinate_context,
                candidate_xyz_absolute_m=seed,
                candidate_set_id=body.get("candidate_set_id") or f"manual-{secrets.token_hex(8)}",
                generated_at=_utc_timestamp(),
                reason=reason or "owner manual add",
                operator_identity=operator,
                note=note,
            )
            state = project_tower_review_state(session)
            return {
                "candidate_set_id": candidate_set.candidate_set_id,
                "candidate_set_digest": candidate_set.digest,
                "candidate_id": candidate_set.candidates[0].candidate_id,
                "state": state,
            }

    # ------------------------------------------------- ordered (V5A3)
    def _ordered_seed(self, body: dict[str, Any]) -> tuple[float, float, float]:
        raw = body.get("xyz")
        if (
            type(raw) is not list
            or len(raw) != 3
            or any(type(value) is not int and type(value) is not float for value in raw)
        ):
            raise _ApiError(
                HTTPStatus.BAD_REQUEST,
                "BAD_XYZ",
                "xyz must be a length-3 numeric array [x, y, z]",
            )
        if not all(value == value for value in raw):
            raise _ApiError(HTTPStatus.BAD_REQUEST, "BAD_XYZ", "xyz must be finite")
        return tuple(float(value) for value in raw)

    def _ordered_token(self, body: dict[str, Any]) -> str:
        token = body.get("token")
        if type(token) is not str or not token:
            raise _ApiError(HTTPStatus.BAD_REQUEST, "BAD_TOKEN", "token must be a string")
        return token

    def _ordered_group_id(self, body: dict[str, Any]) -> str:
        group_id = body.get("group_id")
        if type(group_id) is not str or not group_id:
            raise _ApiError(
                HTTPStatus.BAD_REQUEST, "BAD_GROUP_ID", "group_id must be a string"
            )
        return group_id

    def ordered_select(self, body: dict[str, Any]) -> dict[str, Any]:
        from algorithms.towers.ordered import ordered_review_state, ordered_select

        seed = self._ordered_seed(body)
        group_id = self._ordered_group_id(body)
        operator = body.get("operator_identity") or self.config.operator_identity
        notes = body.get("notes")
        if notes is not None and type(notes) is not str:
            raise _ApiError(HTTPStatus.BAD_REQUEST, "BAD_NOTES", "notes must be a string")
        with self._opened_writable() as session:
            cache = self._tile_cache(session)
            result = ordered_select(
                session,
                cache,
                self.ordered_store,
                candidate_xyz_absolute_m=seed,
                group_id=group_id,
                operator_identity=operator,
                notes=notes,
            )
            state = ordered_review_state(session)
            if not result.success:
                return {
                    "success": False,
                    "failure_reason": result.failure_reason,
                    "candidate_set_id": result.candidate_set_id,
                    "candidate_id": result.candidate_id,
                    "diagnostics": result.diagnostics,
                    "state": state,
                }
            return {
                "success": True,
                "preview": result.preview,
                "candidate_set_id": result.candidate_set_id,
                "candidate_id": result.candidate_id,
                "state": state,
            }

    def ordered_preview(self, token: str) -> dict[str, Any]:
        from algorithms.towers.ordered import ordered_preview

        preview = ordered_preview(self.ordered_store, token)
        return {"preview": preview}

    def ordered_confirm(self, body: dict[str, Any]) -> dict[str, Any]:
        from algorithms.towers.ordered import ordered_confirm, ordered_review_state

        token = self._ordered_token(body)
        operator = body.get("operator_identity") or self.config.operator_identity
        notes = body.get("notes")
        if notes is not None and type(notes) is not str:
            raise _ApiError(HTTPStatus.BAD_REQUEST, "BAD_NOTES", "notes must be a string")
        with self._opened_writable() as session:
            result = ordered_confirm(
                session,
                self.ordered_store,
                token,
                operator_identity=operator,
                notes=notes,
            )
            state = ordered_review_state(session)
            return {"result": result, "state": state}

    def ordered_reselect(self, body: dict[str, Any]) -> dict[str, Any]:
        from algorithms.towers.ordered import ordered_reselect

        token = self._ordered_token(body)
        operator = body.get("operator_identity") or self.config.operator_identity
        with self._opened_writable() as session:
            return ordered_reselect(
                session, self.ordered_store, token, operator_identity=operator
            )

    def ordered_cancel(self, body: dict[str, Any]) -> dict[str, Any]:
        from algorithms.towers.ordered import ordered_cancel

        token = self._ordered_token(body)
        operator = body.get("operator_identity") or self.config.operator_identity
        with self._opened_writable() as session:
            return ordered_cancel(
                session, self.ordered_store, token, operator_identity=operator
            )

    def groups_create(self, body: dict[str, Any]) -> dict[str, Any]:
        """POST /api/tower-review/groups — allocate the next G### with a display_name."""
        from algorithms.towers.ordered import ordered_review_state

        display_name = body.get("display_name")
        if type(display_name) is not str or not display_name.strip():
            raise _ApiError(
                HTTPStatus.BAD_REQUEST, "BAD_DISPLAY_NAME", "display_name must be a string"
            )
        with self._opened_writable() as session:
            group = session.create_tower_group(display_name=display_name.strip())
            state = ordered_review_state(session)
            return {"group": group.to_dict(), "state": state}

    def tower_membership_preview(self, tower_id: str) -> dict[str, Any]:
        """GET /api/tower-review/towers/{tower_id}/membership-preview

        Strictly reads the immutable membership artifact linked by the confirmed
        tower's ``TowerRefinement.local_region_reference`` and returns a
        deterministic capped VIEWER-ONLY display sample.  Never falls back to
        Giro3D/COPC points and never returns the full raw membership.
        """
        from algorithms.towers.membership_artifact import (
            MembershipArtifactError,
            read_membership_artifact,
        )
        from algorithms.towers.ordered import preview_sample_points
        from core.contracts.project import ArtifactReference

        with self._opened() as session:
            try:
                entry = session.tower_registry.get(tower_id)
            except KeyError as exc:
                raise _ApiError(
                    HTTPStatus.NOT_FOUND, "TOWER_NOT_FOUND", f"tower {tower_id} is unknown"
                ) from exc
            revision = entry.current
            reference = revision.refinement.local_region_reference
            if reference is None or type(reference) is not ArtifactReference:
                raise _ApiError(
                    HTTPStatus.NOT_FOUND,
                    "MEMBERSHIP_UNAVAILABLE",
                    f"tower {tower_id} has no membership artifact reference",
                )
            try:
                membership = read_membership_artifact(
                    session.project_root,
                    reference,
                    source_binding=revision.source_binding,
                )
            except MembershipArtifactError as exc:
                raise _ApiError(
                    HTTPStatus.CONFLICT,
                    "MEMBERSHIP_ARTIFACT_INVALID",
                    f"tower {tower_id} membership artifact failed validation: {exc}",
                ) from exc
            # The confirmed-membership reinspection uses a lighter deterministic
            # display cap so the response stays small and responsive even for a
            # tower with a large membership (the exact membership stays
            # server-side; the sample is VIEWER-ONLY).
            sample_max = CONFIRMED_MEMBERSHIP_PREVIEW_SAMPLE_MAX
            points = preview_sample_points(membership, max_points=sample_max)
            minimum, maximum = membership.member_bounds_absolute_m
            return {
                "tower_id": tower_id,
                "group_ids": [
                    group.group_id
                    for group in session.tower_group_registry.groups
                    if tower_id in group.ordered_tower_ids
                ],
                "member_count": membership.member_count,
                "height_m": membership.height_m,
                "member_bounds_absolute_m": {
                    "min_xyz_absolute_m": list(minimum),
                    "max_xyz_absolute_m": list(maximum),
                },
                "membership_digest": membership.membership_digest(),
                "artifact_relative_path": reference.relative_path,
                "preview_sample_authority": "VIEWER_ONLY",
                "preview_sample_cap": sample_max,
                "preview_sample_size": len(points),
                "preview_sample": points,
            }


def _utc_timestamp() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------


def _outcome_dict(outcome: Any) -> dict[str, Any]:
    return {
        "span_id": outcome.span_id,
        "status": outcome.status,
        "run_id": outcome.run_id,
        "reused": outcome.reused,
        "error": outcome.error,
        "summary": outcome.summary,
    }


class _P3aHandler(BaseHTTPRequestHandler):
    server_version = "P3aGiro3dPOC/0.1"
    tower_review_api: TowerReviewApi | None = None

    def _resolve(self, path: str) -> Path | None:
        path = urllib.parse.unquote(path.split("?")[0].split("#")[0])
        # Collapse duplicate slashes (e.g. the "/laz-perf//laz-perf.wasm" URL that
        # Giro3D produces when a trailing-slash lazPerfPath is combined with its
        # own "/laz-perf.wasm").  The path still stays within the virtual roots.
        path = re.sub(r"//+", "/", path)
        # The viewer always fetches poc_config.json; serve whichever config this
        # launcher was started with (ROI POC or FULL-SOURCE gate).
        if path == "/poc_config.json":
            target = (VIEWER_ROOT / _CONFIG_FILE).resolve()
            if target.is_file():
                return target
            return None
        for prefix, root in _routes():
            if path == prefix or path.startswith(prefix):
                relative = path[len(prefix):]
                if not relative:
                    relative = "index.html"
                target = (root / relative).resolve()
                try:
                    target.relative_to(root.resolve())
                except ValueError:
                    return None
                return target if target.is_file() else None
        return None

    def _serve_file(self, target: Path) -> None:
        """Serve a file, honouring HTTP Range requests (copc.js reads the COPC
        via byte-range fetches for the header, hierarchy pages and chunks)."""
        data = target.read_bytes()
        total = len(data)
        start, end = 0, total - 1

        range_header = self.headers.get("Range")
        if range_header:
            m = range_header.strip().lower()
            if m.startswith("bytes="):
                spec = m[len("bytes="):].split(",", 1)[0].strip()
                if spec:
                    parts = spec.split("-", 1)
                    try:
                        if parts[0] == "" and parts[1] != "":
                            # suffix range: last N bytes
                            n = int(parts[1])
                            start = max(0, total - n)
                        elif parts[0] != "":
                            start = int(parts[0])
                            if parts[1] != "":
                                end = min(int(parts[1]), total - 1)
                        if 0 <= start <= end < total:
                            self.send_response(HTTPStatus.PARTIAL_CONTENT)
                            self.send_header("Content-Range", f"bytes {start}-{end}/{total}")
                            self.send_header("Accept-Ranges", "bytes")
                            self.send_header("Content-Type", _guess_type(target))
                            self.send_header("Content-Length", str(end - start + 1))
                            self.send_header("Cache-Control", "no-store")
                            self.end_headers()
                            self.wfile.write(data[start : end + 1])
                            return
                    except ValueError:
                        pass
        self.send_response(HTTPStatus.OK)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Type", _guess_type(target))
        self.send_header("Content-Length", str(total))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _serve_healthz(self) -> None:
        # The active COPC is derived from the served viewer config's copcUrl so
        # the health check reflects whatever dataset this launcher is serving
        # (ROI POC or FULL-SOURCE gate) — never a stale hard-coded name.
        active_copc = active_copc_name()
        copc = DATASET_ROOT / active_copc if active_copc else None
        body = json.dumps(
            {
                "status": "ok",
                "viewer_root": str(VIEWER_ROOT),
                "active_copc": active_copc,
                "dataset_ready": bool(copc and copc.is_file()),
                "bundle_ready": (VENDOR_ROOT / "giro3d.bundle.js").is_file(),
                "lazperf_ready": (LAZPERF_ROOT / "laz-perf.wasm").is_file(),
            },
            ensure_ascii=False,
        ).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    # ----------------------------------------------------------- JSON API
    def _json_response(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _handle_tree_viewer_api(self) -> None:
        """Serve Tree Viewer D's read-only span/list/preview routes."""

        api = self.tower_review_api
        if api is None:
            self._json_response(
                HTTPStatus.NOT_FOUND,
                {"ok": False, "error": {"code": "API_DISABLED", "message": "Tree API is not enabled"}},
            )
            return
        path = self.path.split("?")[0].rstrip("/")
        preview = re.fullmatch(r"/api/spans/(S[0-9]{3,})/trees/([0-9]+)/preview", path)
        trees = re.fullmatch(r"/api/spans/(S[0-9]{3,})/trees", path)
        try:
            if path == "/api/spans":
                result = api.tree_spans_list()
            elif preview:
                result = api.spans_tree_preview(preview.group(1), preview.group(2))
            elif trees:
                result = api.spans_trees(trees.group(1))
            else:
                self._json_response(
                    HTTPStatus.NOT_FOUND,
                    {"ok": False, "error": {"code": "NOT_FOUND", "message": "unknown Tree API route"}},
                )
                return
        except _ApiError as exc:
            self._json_response(exc.status, {"ok": False, "error": {"code": exc.code, "message": exc.message}})
            return
        self._json_response(HTTPStatus.OK, {"ok": True, **result})

    def _query_param(self, name: str) -> str:
        parsed = urllib.parse.urlparse(self.path)
        values = urllib.parse.parse_qs(parsed.query)
        raw = values.get(name)
        if not raw or not raw[0]:
            raise _ApiError(
                HTTPStatus.BAD_REQUEST,
                f"BAD_{name.upper()}",
                f"query parameter {name} is required",
            )
        return raw[0]

    def _read_json_body(self) -> dict[str, Any]:
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError as exc:
            raise _ApiError(
                HTTPStatus.BAD_REQUEST, "BAD_CONTENT_LENGTH", "invalid Content-Length"
            ) from exc
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise _ApiError(
                HTTPStatus.BAD_REQUEST, "BAD_JSON", "request body is not valid JSON"
            ) from exc
        if type(payload) is not dict:
            raise _ApiError(
                HTTPStatus.BAD_REQUEST, "BAD_JSON", "request body must be a JSON object"
            )
        return payload

    def _api_route(self) -> tuple[str, str, str] | None:
        """Split an /api/tower-review/... path into (action, set_id, candidate_id)."""
        path = self.path.split("?")[0].split("#")[0].rstrip("/")
        if not path.startswith("/api/tower-review/"):
            return None
        parts = [item for item in path[len("/api/tower-review/"):].split("/") if item]
        if len(parts) == 1 and parts[0] in {"state", "detect", "manual-candidates", "groups"}:
            return (parts[0], "", "")
        if (
            len(parts) == 4
            and parts[0] == "candidates"
            and parts[3] in {"accept", "reject"}
        ):
            return (parts[3], parts[1], parts[2])
        if len(parts) == 2 and parts[0] == "ordered" and parts[1] in {
            "select",
            "preview",
            "confirm",
            "reselect",
            "cancel",
        }:
            return (f"ordered-{parts[1]}", "", "")
        if (
            len(parts) == 3
            and parts[0] == "towers"
            and parts[2] == "membership-preview"
        ):
            return ("tower-membership-preview", parts[1], "")
        return None

    def _span_api_route(self) -> tuple[str, str, str] | None:
        path = self.path.split("?")[0].split("#")[0].rstrip("/")
        if path == "/api/spans":
            return ("list", "", "")
        if path == "/api/spans/synchronize":
            return ("synchronize", "", "")
        if path == "/api/spans/analyze-pending":
            return ("analyze-pending", "", "")
        if path == "/api/spans/retry-failed":
            return ("retry-failed", "", "")
        prediction = re.fullmatch(r"/api/spans/(S[0-9]{3,})/prediction", path)
        if prediction:
            return ("prediction", prediction.group(1), "")
        work_order = re.fullmatch(r"/api/spans/(S[0-9]{3,})/work-order", path)
        if work_order:
            return ("work-order", work_order.group(1), "")
        wire_preview = re.fullmatch(r"/api/spans/(S[0-9]{3,})/wire/([0-9]+)/preview", path)
        if wire_preview:
            return ("wire-preview", wire_preview.group(1), wire_preview.group(2))
        wire = re.fullmatch(r"/api/spans/(S[0-9]{3,})/wire", path)
        if wire:
            return ("wire", wire.group(1), "")
        danger = re.fullmatch(r"/api/spans/(S[0-9]{3,})/danger-points", path)
        if danger:
            return ("danger-points", danger.group(1), "")
        tree_preview = re.fullmatch(r"/api/spans/(S[0-9]{3,})/trees/([0-9]+)/preview", path)
        if tree_preview:
            return ("tree-preview", tree_preview.group(1), tree_preview.group(2))
        trees = re.fullmatch(r"/api/spans/(S[0-9]{3,})/trees", path)
        if trees:
            return ("trees", trees.group(1), "")
        match = re.fullmatch(r"/api/spans/(S[0-9]{3,})(?:/analyze)?", path)
        if match:
            return ("analyze" if path.endswith("/analyze") else "get", match.group(1), "")
        return None

    def _handle_span_api(self) -> None:
        api = self.tower_review_api
        route = self._span_api_route()
        if api is None or route is None:
            self._json_response(
                HTTPStatus.NOT_FOUND,
                {"ok": False, "error": {"code": "API_DISABLED", "message": "Span API is not enabled"}},
            )
            return
        action, span_id, tree_id = route
        try:
            if self.command == "GET" and action == "list":
                result = api.spans_list()
            elif self.command == "GET" and action == "get":
                result = api.spans_get(span_id)
            elif self.command == "GET" and action == "trees":
                result = api.spans_trees(span_id)
            elif self.command == "GET" and action == "tree-preview":
                result = api.spans_tree_preview(span_id, tree_id)
            elif self.command == "GET" and action == "wire":
                result = api.spans_wire(span_id)
            elif self.command == "GET" and action == "wire-preview":
                result = api.spans_wire_preview(span_id, tree_id)
            elif self.command == "GET" and action == "danger-points":
                result = api.spans_danger_points(span_id)
            elif self.command == "POST" and action == "synchronize":
                result = api.spans_synchronize()
            elif self.command == "POST" and action == "analyze":
                result = api.spans_analyze(span_id)
            elif self.command == "POST" and action == "analyze-pending":
                result = api.spans_analyze_pending()
            elif self.command == "POST" and action == "retry-failed":
                result = api.spans_retry_failed()
            elif self.command == "POST" and action == "prediction":
                result = api.spans_prediction(span_id, self._read_json_body())
            elif self.command == "POST" and action == "work-order":
                result = api.spans_work_order(span_id, self._read_json_body())
            else:
                raise _ApiError(HTTPStatus.NOT_FOUND, "NOT_FOUND", "unsupported Span route")
        except _ApiError as exc:
            self._json_response(exc.status, {"ok": False, "error": {"code": exc.code, "message": exc.message}})
            return
        except Exception as exc:
            code = getattr(exc, "code", "INTERNAL_ERROR")
            status = HTTPStatus.CONFLICT if code in {"TOPOLOGY_DRIFT", "MISSING_TOWER_ENDPOINT"} else HTTPStatus.INTERNAL_SERVER_ERROR
            self._json_response(status, {"ok": False, "error": {"code": code, "message": f"{type(exc).__name__}: {exc}"}})
            return
        self._json_response(HTTPStatus.OK, {"ok": True, **result})

    def _handle_tower_api(self) -> None:
        api = self.tower_review_api
        if api is None:
            self._json_response(
                HTTPStatus.NOT_FOUND,
                {
                    "ok": False,
                    "error": {
                        "code": "API_DISABLED",
                        "message": "tower-review API is not enabled on this launcher",
                    },
                },
            )
            return
        route = self._api_route()
        if route is None:
            self._json_response(
                HTTPStatus.NOT_FOUND,
                {"ok": False, "error": {"code": "NOT_FOUND", "message": "unknown API route"}},
            )
            return
        action, set_id, candidate_id = route
        try:
            if self.command == "GET" and action == "state":
                payload = {"ok": True, "state": api.state()}
            elif self.command == "POST" and action == "detect":
                payload = {"ok": True, "result": api.detect(self._read_json_body())}
            elif self.command == "POST" and action == "manual-candidates":
                payload = {"ok": True, **api.manual_add(self._read_json_body())}
            elif self.command == "POST" and action == "reject" and set_id and candidate_id:
                payload = {"ok": True, **api.reject(set_id, candidate_id, self._read_json_body())}
            elif self.command == "POST" and action == "accept" and set_id and candidate_id:
                payload = {"ok": True, **api.accept(set_id, candidate_id, self._read_json_body())}
            elif self.command == "GET" and action == "ordered-preview":
                payload = {"ok": True, **api.ordered_preview(self._query_param("token"))}
            elif self.command == "POST" and action == "ordered-select":
                payload = {"ok": True, **api.ordered_select(self._read_json_body())}
            elif self.command == "POST" and action == "ordered-confirm":
                payload = {"ok": True, **api.ordered_confirm(self._read_json_body())}
            elif self.command == "POST" and action == "ordered-reselect":
                payload = {"ok": True, **api.ordered_reselect(self._read_json_body())}
            elif self.command == "POST" and action == "ordered-cancel":
                payload = {"ok": True, **api.ordered_cancel(self._read_json_body())}
            elif self.command == "POST" and action == "groups":
                payload = {"ok": True, **api.groups_create(self._read_json_body())}
            elif self.command == "GET" and action == "tower-membership-preview" and set_id:
                payload = {"ok": True, **api.tower_membership_preview(set_id)}
            else:
                raise _ApiError(
                    HTTPStatus.NOT_FOUND, "NOT_FOUND", "unsupported tower-review route"
                )
        except _ApiError as exc:
            payload = {
                "ok": False,
                "error": {"code": exc.code, "message": exc.message},
            }
            self._json_response(exc.status, payload)
            return
        except Exception as exc:  # structured ordered-workflow failure
            from algorithms.towers.ordered import (
                OrderedPreviewInvalidError,
                OrderedWorkflowError,
            )

            if isinstance(exc, OrderedPreviewInvalidError):
                payload = {
                    "ok": False,
                    "error": {
                        "code": "PREVIEW_TOKEN_INVALID",
                        "message": str(exc),
                    },
                }
                self._json_response(HTTPStatus.CONFLICT, payload)
                return
            if isinstance(exc, OrderedWorkflowError):
                payload = {
                    "ok": False,
                    "error": {
                        "code": "ORDERED_WORKFLOW_ERROR",
                        "message": str(exc),
                    },
                }
                self._json_response(HTTPStatus.BAD_REQUEST, payload)
                return
            payload = {
                "ok": False,
                "error": {"code": "INTERNAL_ERROR", "message": f"{type(exc).__name__}: {exc}"},
            }
            self._json_response(HTTPStatus.INTERNAL_SERVER_ERROR, payload)
            return
        self._json_response(HTTPStatus.OK, payload)

    def do_GET(self) -> None:  # noqa: N802 - http.server API
        path = self.path.split("?")[0]
        if path == "/p3a_giro3d_healthz":
            self._serve_healthz()
            return
        if re.fullmatch(r"/api/spans/S[0-9]{3,}/trees(?:/[0-9]+/preview)?", path):
            self._handle_tree_viewer_api()
            return
        if path == "/api/desktop/project":
            api = self.tower_review_api
            if api is None:
                self._json_response(HTTPStatus.NOT_FOUND, {"ok": False, "error": {"code": "API_DISABLED", "message": "Desktop API is disabled"}})
                return
            try:
                self._json_response(HTTPStatus.OK, {"ok": True, "project": api.desktop_project()})
            except ValueError as exc:
                self._json_response(HTTPStatus.CONFLICT, {"ok": False, "error": {"code": "PROJECT_INVALID", "message": str(exc)}})
            return
        if path.startswith("/api/"):
            if path.startswith("/api/spans"):
                self._handle_span_api()
                return
            self._handle_tower_api()
            return
        target = self._resolve(self.path)
        if target is None:
            self.send_error(HTTPStatus.NOT_FOUND, "not found")
            return
        self._serve_file(target)

    def do_POST(self) -> None:  # noqa: N802 - http.server API
        if self.path.split("?")[0].startswith("/api/"):
            if self.path.split("?")[0].startswith("/api/spans"):
                self._handle_span_api()
                return
            self._handle_tower_api()
            return
        self.send_error(HTTPStatus.NOT_FOUND, "not found")

    do_HEAD = do_GET  # noqa: N815 - http.server API

    def log_message(self, fmt: str, *args) -> None:  # noqa: N802
        sys.stdout.write(f"[p3a-giro3d] {self.address_string()} {fmt % args}\n")
        sys.stdout.flush()


def _parse_origin(value: str) -> tuple[float, float, float]:
    try:
        parts = [float(item) for item in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("origin must be three comma-separated floats") from exc
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("origin must be three comma-separated floats")
    return tuple(parts)


def main(argv: list[str] | None = None) -> int:
    global DATASET_ROOT, _CONFIG_FILE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8091,
                        help="loopback port to bind (default: 8091, 0 = auto-assign)")
    parser.add_argument("--no-open", action="store_true", help="do not open the browser")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="dataset cache root to serve at /pointclouds/ (default: ROI POC cache)",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="viewer config served at /poc_config.json (default: poc_config.json = ROI POC)",
    )
    parser.add_argument(
        "--tower-project-root",
        type=Path,
        default=None,
        help="enable the tower-review API against this project root",
    )
    parser.add_argument(
        "--tower-tile-cache",
        type=Path,
        default=None,
        help="derived RAW spatial tile cache used for one-click ACCEPT refinement",
    )
    parser.add_argument(
        "--tower-detect-ground",
        type=Path,
        default=None,
        help="compatible coarse ground/HAG evidence LAS used by POST /detect",
    )
    parser.add_argument(
        "--tower-detect-origin",
        type=_parse_origin,
        default=(0.0, 0.0, 0.0),
        help="origin_xyz for the ground evidence (default: 0,0,0)",
    )
    parser.add_argument(
        "--tower-operator",
        default="owner",
        help="default operator identity recorded on review events",
    )
    args = parser.parse_args(argv)

    if args.dataset_root is not None:
        DATASET_ROOT = args.dataset_root
    if args.config is not None:
        _CONFIG_FILE = args.config

    for label, root in (
        ("viewer", VIEWER_ROOT),
        ("vendor", VENDOR_ROOT),
        ("dataset", DATASET_ROOT),
    ):
        if not root.is_dir():
            print(f"[p3a-giro3d] ERROR: {label} root missing: {root}", file=sys.stderr)
            return 2

    tower_api: TowerReviewApi | None = None
    if args.tower_project_root is not None:
        if args.tower_tile_cache is None:
            print(
                "[p3a-giro3d] ERROR: --tower-tile-cache is required with --tower-project-root",
                file=sys.stderr,
            )
            return 2
        tower_api = TowerReviewApi(
            TowerReviewConfig(
                project_root=args.tower_project_root,
                tile_cache=args.tower_tile_cache,
                detect_ground_las=args.tower_detect_ground,
                detect_origin_xyz=args.tower_detect_origin,
                operator_identity=args.tower_operator,
            )
        )
        _P3aHandler.tower_review_api = tower_api
        print(f"[p3a-giro3d] tower-review API enabled for project: {args.tower_project_root}")
        if args.tower_detect_ground is None:
            print("[p3a-giro3d]   (POST /detect disabled: no --tower-detect-ground given)")

    try:
        httpd = _ExclusiveBindServer((P3A_BIND_HOST, args.port), _P3aHandler)
    except OSError as exc:
        print(
            f"[p3a-giro3d] ERROR: cannot bind loopback {P3A_BIND_HOST}:{args.port} "
            f"(port in use or address unavailable): {exc}",
            file=sys.stderr,
        )
        return 3

    url = f"http://127.0.0.1:{httpd.server_address[1]}/"
    print(f"[p3a-giro3d] serving P3A Giro3D viewer at {url}")
    print(f"[p3a-giro3d] dataset root: {DATASET_ROOT}")
    print(f"[p3a-giro3d] viewer config: {_CONFIG_FILE}")
    copc = active_copc_name()
    if copc:
        copc_path = DATASET_ROOT / copc
        print(f"[p3a-giro3d] COPC: {url}pointclouds/{copc}"
              + (" (ready)" if copc_path.is_file() else " (MISSING — run the converter)"))
    if not args.no_open:
        import webbrowser

        webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[p3a-giro3d] shutting down")
    return 0


if __name__ == "__main__":
    sys.exit(main())
