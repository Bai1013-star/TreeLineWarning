"""Stage-A runtime boundary for the fully offline product."""

from __future__ import annotations

RUNTIME_NETWORK_REQUIRED = False
OFFLINE_RUNTIME_SCHEMA = "full-offline-runtime-v1"
ALLOWED_BASEMAP_POLICIES = frozenset({"NO_BASEMAP", "OFFLINE_BASEMAP"})


__all__ = [
    "ALLOWED_BASEMAP_POLICIES",
    "OFFLINE_RUNTIME_SCHEMA",
    "RUNTIME_NETWORK_REQUIRED",
]
