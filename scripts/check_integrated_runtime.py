"""Finite smoke check for the integrated processing and warning backends."""

from __future__ import annotations

import json
import tempfile
import urllib.request
from pathlib import Path

from desktop.app import BackendProcess, WarningBackend
from desktop.project_management import create_empty_project


def _read(url: str) -> str:
    with urllib.request.urlopen(url, timeout=20) as response:
        return response.read().decode("utf-8")


def main() -> int:
    repository_root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="integrated-smoke-", dir=repository_root) as temporary:
        parent = Path(temporary)
        descriptor = create_empty_project(parent, "smoke-project")
        project_root = Path(descriptor["root"])
        processing = BackendProcess(project_root, repository_root)
        warning = WarningBackend(project_root, repository_root)
        try:
            processing_url = processing.start()
            warning_url = warning.start()
            processing_page = _read(processing_url)
            warning_page = _read(warning_url)
            health = json.loads(_read(warning_url + "health"))
            if "openWarningModule" not in processing_page:
                raise RuntimeError("processing module navigation is missing")
            if "openProcessingModule" not in warning_page:
                raise RuntimeError("warning module navigation is missing")
            if health.get("status") != "ok" and health.get("message") != "TreeLineWarning API running":
                raise RuntimeError(f"warning health failed: {health}")
            print(
                json.dumps(
                    {
                        "status": "PASS",
                        "processing_backend": processing_url,
                        "warning_backend": warning_url,
                        "navigation": "BIDIRECTIONAL",
                    },
                    ensure_ascii=False,
                )
            )
            return 0
        finally:
            warning.stop()
            processing.stop()


if __name__ == "__main__":
    raise SystemExit(main())
