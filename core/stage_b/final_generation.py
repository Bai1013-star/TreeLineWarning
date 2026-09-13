"""Deterministic generation identity for post-editor publications.

The analysis run identifies frozen upstream science.  A final generation
identifies one exact replay of that run's edit journal and surviving tree
membership.  Downstream consumers use it to reject mixed pre/post-editor
state without changing any scientific calculation.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from core.config.fast_baseline import canonical_digest

FINAL_GENERATION_SCHEMA = "treeline-final-generation-v1"


def final_generation_id(
    analysis_run_id: str,
    records: Iterable[Any],
    operation_count: int,
) -> str:
    """Return a stable fingerprint for one authoritative final tree state."""

    trees = []
    for record in records:
        getter = record.get if isinstance(record, Mapping) else lambda key, default=None: getattr(record, key, default)
        trees.append(
            {
                "global_tree_id": int(getter("global_tree_id")),
                "persistent_tree_id": getter("persistent_tree_id"),
                "full_resolution_point_count": int(
                    getter("full_resolution_point_count", 0) or 0
                ),
                "merged_span_tree_keys": [
                    [int(pair[0]), int(pair[1])]
                    for pair in (getter("merged_span_tree_keys", ()) or ())
                ],
            }
        )
    payload = {
        "schema": FINAL_GENERATION_SCHEMA,
        "analysis_run_id": str(analysis_run_id),
        "editor_operation_count": int(operation_count),
        "trees": sorted(trees, key=lambda item: item["global_tree_id"]),
    }
    return "final-" + canonical_digest(payload)[:32]


def derived_generation_map(value: Any) -> dict[str, str]:
    diagnostics = getattr(value, "batch_diagnostics", {}) or {}
    raw = diagnostics.get("derived_generations") if isinstance(diagnostics, Mapping) else None
    return {
        str(key): str(item)
        for key, item in (raw.items() if isinstance(raw, Mapping) else ())
        if isinstance(item, str) and item
    }


__all__ = [
    "FINAL_GENERATION_SCHEMA",
    "derived_generation_map",
    "final_generation_id",
]
