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
import json
import mimetypes
import re
import secrets
import socket
import sys
import urllib.parse
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from scripts.tree_viewer_api import TreeViewerApiMixin

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
# The launcher may be executed directly (sys.path[0] = scripts/) rather than as
# a module, so make the repository root importable for the lazy API imports.
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

VIEWER_ROOT = _REPOSITORY_ROOT / "viewer_poc" / "giro3d"
VENDOR_ROOT = VIEWER_ROOT / "vendor"
LAZPERF_ROOT = VENDOR_ROOT / "laz-perf"
DATASET_ROOT = _REPOSITORY_ROOT / "cache" / "p3a_giro3d" / "dataset"
# Viewer config served at /poc_config.json (ROI POC or FULL-SOURCE gate).
_CONFIG_FILE = "poc_config.json"

# The single canonical P3A binding.  Loopback-only; never broaden.
P3A_BIND_HOST = "127.0.0.1"

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
                {
                    "ok": False,
                    "error": {"code": "API_DISABLED", "message": "Tree API is not enabled"},
                },
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
                raise _ApiError(HTTPStatus.NOT_FOUND, "NOT_FOUND", "unknown Tree API route")
        except _ApiError as exc:
            self._json_response(
                exc.status, {"ok": False, "error": {"code": exc.code, "message": exc.message}}
            )
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
        if path == "/api/spans" or re.fullmatch(
            r"/api/spans/S[0-9]{3,}/trees(?:/[0-9]+/preview)?", path
        ):
            self._handle_tree_viewer_api()
            return
        if path.startswith("/api/"):
            self._handle_tower_api()
            return
        target = self._resolve(self.path)
        if target is None:
            self.send_error(HTTPStatus.NOT_FOUND, "not found")
            return
        self._serve_file(target)

    def do_POST(self) -> None:  # noqa: N802 - http.server API
        if self.path.split("?")[0].startswith("/api/"):
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
