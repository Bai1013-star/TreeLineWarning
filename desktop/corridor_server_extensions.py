"""Loopback HTTP endpoints for corridor registration and tower assistance."""

from __future__ import annotations

import urllib.parse
from typing import Any

_INSTALLED = False


def install_corridor_server_extensions() -> None:
    global _INSTALLED
    if _INSTALLED:
        return

    from desktop.server import _DesktopHandler, _json_body

    original_get = _DesktopHandler._handle_get
    original_post = _DesktopHandler._handle_post

    def handle_get(self: Any, path: str) -> None:
        try:
            if path == "/corridor-ui.js":
                self._static("viewer_poc/giro3d/corridor_ui.js")
                return
            if path == "/api/desktop/corridor/registration":
                self._send_json(
                    {
                        "ok": True,
                        "registration": self.workflow.corridor_registration_projection(),
                    }
                )
                return
            if path == "/api/desktop/corridor/topology":
                project = self.workflow.project_projection()
                self._send_json(
                    {
                        "ok": True,
                        "topology": project.get("corridor_topology"),
                        "readiness": self.workflow.span_readiness_projection(),
                        "summary": project.get("span_readiness_summary"),
                    }
                )
                return
            if path == "/api/desktop/corridor/readiness":
                self._send_json(
                    {
                        "ok": True,
                        "spans": self.workflow.span_readiness_projection(),
                    }
                )
                return
            if path == "/api/desktop/corridor/tower-candidates":
                query = urllib.parse.parse_qs(urllib.parse.urlsplit(self.path).query)
                tower_id = str(query.get("tower_id", [""])[0])
                if not tower_id:
                    raise ValueError("TOWER_ID_REQUIRED")
                self._send_json(
                    {
                        "ok": True,
                        **self.workflow.tower_candidate_projection(tower_id),
                    }
                )
                return
        except (ValueError, KeyError, TypeError, OSError, RuntimeError) as exc:
            self.workflow._log(
                "api_failure",
                path=path,
                error=f"{type(exc).__name__}: {exc}",
            )
            self._exception_error(exc)
            return
        original_get(self, path)

    def handle_post(self: Any, path: str) -> None:
        handled = {
            "/api/desktop/corridor/prepare",
            "/api/desktop/corridor/anchor",
            "/api/desktop/corridor/tower-confirm",
            "/api/desktop/corridor/topology/rebuild",
            "/api/desktop/corridor/batch-ready",
        }
        if path not in handled:
            original_post(self, path)
            return
        try:
            payload = _json_body(self)
            if path == "/api/desktop/corridor/prepare":
                self._send_json(
                    {
                        "ok": True,
                        "registration": self.workflow.prepare_corridor_registration(),
                    }
                )
                return
            if path == "/api/desktop/corridor/anchor":
                self._send_json(
                    {
                        "ok": True,
                        "registration": self.workflow.set_corridor_anchor(payload),
                    }
                )
                return
            if path == "/api/desktop/corridor/tower-confirm":
                self._send_json(
                    {
                        "ok": True,
                        "project": self.workflow.confirm_gis_tower_candidate(payload),
                    }
                )
                return
            if path == "/api/desktop/corridor/topology/rebuild":
                self._send_json(
                    {
                        "ok": True,
                        "topology": self.workflow.rebuild_corridor_topology(payload),
                        "project": self.workflow.project_projection(),
                    }
                )
                return
            if path == "/api/desktop/corridor/batch-ready":
                span_ids = payload.get("span_ids")
                selected = list(span_ids) if isinstance(span_ids, list) else None
                self._send_json(
                    {
                        "ok": True,
                        **self.workflow.start_ready_batch(
                            selected,
                            mode=str(payload.get("mode", "REAL")),
                        ),
                    }
                )
                return
        except (ValueError, KeyError, TypeError, OSError, RuntimeError) as exc:
            self.workflow._log(
                "api_failure",
                path=path,
                error=f"{type(exc).__name__}: {exc}",
            )
            self._exception_error(exc)

    _DesktopHandler._handle_get = handle_get
    _DesktopHandler._handle_post = handle_post
    _INSTALLED = True


__all__ = ["install_corridor_server_extensions"]
