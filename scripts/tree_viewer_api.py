"""Read-only Tree Viewer D API projection.

This module deliberately contains presentation and artifact-reading code only;
tree segmentation and persistent-identity allocation remain in the frozen core.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


class TreeViewerApiMixin:
    """Mixin for ``TowerReviewApi`` with the identity-first Tree endpoints."""

    _palette = (
        "#00bcd4",
        "#ff7043",
        "#ab47bc",
        "#66bb6a",
        "#ffa726",
        "#42a5f5",
        "#ec407a",
        "#8d6e63",
        "#d4e157",
        "#26a69a",
        "#7e57c2",
        "#ef5350",
        "#29b6f6",
        "#9ccc65",
        "#ffca28",
        "#26c6da",
        "#5c6bc0",
        "#f06292",
        "#8bc34a",
        "#ff8a65",
        "#7cb342",
        "#00897b",
        "#ffb300",
        "#7986cb",
    )

    def _tree_error(self, status: int, code: str, message: str) -> Exception:
        raise NotImplementedError

    @classmethod
    def _tree_color(cls, color_key: str) -> str:
        digest = hashlib.sha256(color_key.encode("utf-8")).digest()
        return cls._palette[int.from_bytes(digest[:4], "big") % len(cls._palette)]

    @staticmethod
    def _tree_identity(record: dict[str, Any]) -> dict[str, Any]:
        persistent = record.get("persistent_tree_id")
        if isinstance(persistent, str) and persistent.strip():
            persistent = persistent.strip()
            return {
                "display_id": persistent,
                "persistent_tree_id": persistent,
                "presentation": "PERSISTENT",
                "color_key": f"persistent:{persistent}",
                "candidate_persistent_tree_ids": [],
                "needs_review": False,
                "prediction_eligible": True,
            }
        candidates = sorted(
            {
                item.strip()
                for item in record.get("candidate_persistent_tree_ids") or []
                if isinstance(item, str) and item.strip()
            }
        )
        if record.get("identity_state") == "MERGE_AMBIGUOUS_REVIEW":
            return {
                "display_id": "未绑定实体",
                "persistent_tree_id": None,
                "presentation": "MERGE REVIEW",
                "color_key": "merge-review:" + "|".join(candidates),
                "candidate_persistent_tree_ids": candidates,
                "needs_review": True,
                "prediction_eligible": False,
            }
        return {
            "display_id": f"Legacy Tree {record.get('global_tree_id')}",
            "persistent_tree_id": None,
            "presentation": "LEGACY",
            "color_key": f"legacy:{record.get('global_tree_id')}",
            "candidate_persistent_tree_ids": candidates,
            "needs_review": False,
            "prediction_eligible": False,
        }

    def _tree_span(self, session: Any, span_id: str) -> Any:
        try:
            return next(item for item in session.span_registry.records if item.span_id == span_id)
        except StopIteration as exc:
            raise self._tree_error(404, "SPAN_NOT_FOUND", f"unknown Span {span_id}") from exc

    def _tree_run(self, session: Any, span: Any) -> tuple[dict[str, Any], Path]:
        reference = span.artifact_index_reference
        if reference is None:
            raise self._tree_error(409, "SPAN_HAS_NO_RUN", f"{span.span_id} has no current run")
        path = session.project_root / Path(reference.relative_path)
        try:
            data = path.read_bytes()
        except OSError as exc:
            raise self._tree_error(
                409, "SPAN_RUN_ARTIFACT_MISSING", "run artifact is missing"
            ) from exc
        if len(data) != reference.byte_size or hashlib.sha256(data).hexdigest() != reference.sha256:
            raise self._tree_error(409, "SPAN_RUN_ARTIFACT_INVALID", "run artifact digest mismatch")
        return json.loads(data.decode("utf-8")), path.parent

    @staticmethod
    def _tree_origin(session: Any, index: dict[str, Any]) -> tuple[float, float, float]:
        prep_ref = (
            (index.get("provenance") or {}).get("source_preparation", {}).get("manifest_reference")
        )
        if isinstance(prep_ref, str):
            try:
                origin = json.loads((session.project_root / prep_ref).read_bytes()).get(
                    "origin_xyz"
                )
                if isinstance(origin, (list, tuple)) and len(origin) == 3:
                    return tuple(float(value) for value in origin)
            except (OSError, TypeError, ValueError):
                pass
        return (0.0, 0.0, 0.0)

    @staticmethod
    def _absolute(local: Any, origin: tuple[float, float, float]) -> list[float] | None:
        if not isinstance(local, (list, tuple)) or len(local) != 3:
            return None
        return [float(local[index]) + origin[index] for index in range(3)]

    @staticmethod
    def _category(record: dict[str, Any]) -> str:
        return (
            "CORE"
            if (record.get("core_fraction", 0.0) or 0.0) > 0.0 and not record.get("review")
            else "HALO / REVIEW"
        )

    def tree_spans_list(self) -> dict[str, Any]:
        with self._opened() as session:
            return {
                "spans": [
                    {"span_id": span.span_id, "display_name": span.display_name}
                    for span in sorted(session.span_registry.records, key=lambda item: item.span_id)
                ]
            }

    def spans_trees(self, span_id: str) -> dict[str, Any]:
        with self._opened() as session:
            span = self._tree_span(session, span_id)
            index, run_dir = self._tree_run(session, span)
            origin = self._tree_origin(session, index)
            try:
                records = (
                    json.loads((run_dir / "reconciliation.json").read_bytes()).get(
                        "global_tree_records"
                    )
                    or []
                )
            except OSError as exc:
                raise self._tree_error(
                    409, "SPAN_RECONCILIATION_MISSING", "reconciliation artifact missing"
                ) from exc
            clearance_by_id: dict[Any, dict[str, Any]] = {}
            try:
                for clearance in (
                    json.loads((run_dir / "clearance.json").read_bytes()).get("records") or []
                ):
                    clearance_by_id[clearance.get("global_tree_id")] = clearance
            except (OSError, ValueError):
                pass
            rows = []
            for record in sorted(
                records,
                key=lambda item: (
                    int(item["span_order_index"])
                    if item.get("span_order_index") is not None
                    else 2**31 - 1,
                    int(item.get("global_tree_id", -1)),
                ),
            ):
                identity = self._tree_identity(record)
                clearance = clearance_by_id.get(record.get("global_tree_id"))
                apex = record.get("apex_xy_local_m")
                rows.append(
                    {
                        "global_tree_id": record.get("global_tree_id"),
                        "persistent_tree_id": identity["persistent_tree_id"],
                        "display_id": identity["display_id"],
                        "identity_state": record.get("identity_state", "LEGACY_RUN_LOCAL"),
                        "identity_presentation": identity["presentation"],
                        "candidate_persistent_tree_ids": identity["candidate_persistent_tree_ids"],
                        "instance_color": self._tree_color(identity["color_key"]),
                        "prediction_identity": {
                            "persistent_tree_id": identity["persistent_tree_id"],
                            "global_tree_id": record.get("global_tree_id"),
                            "needs_review": identity["needs_review"],
                            "eligible_for_long_term_history": identity["prediction_eligible"],
                        },
                        "span_id": span_id,
                        "span_name": record.get("span_name"),
                        "span_order_index": record.get("span_order_index"),
                        "longitudinal_s_m": record.get("longitudinal_s_m"),
                        "lateral_n_m": record.get("lateral_n_m"),
                        "apex_height_m": record.get("apex_height_m"),
                        "apex_xyz_absolute_m": self._absolute(apex, origin),
                        "full_resolution_point_count": record.get("full_resolution_point_count"),
                        "core_fraction": record.get("core_fraction", 0.0),
                        "review": bool(record.get("review", False)),
                        "category": self._category(record),
                        "minimum_clearance_m": clearance.get("minimum_clearance_m")
                        if clearance
                        else None,
                        "clearance": self._clearance_projection(clearance, origin),
                    }
                )
            core_count = sum(row["category"] == "CORE" for row in rows)
            return {
                "span_id": span_id,
                "tree_count": len(rows),
                "core_count": core_count,
                "halo_review_count": len(rows) - core_count,
                "trees": rows,
            }

    def _clearance_projection(
        self, clearance: dict[str, Any] | None, origin: tuple[float, float, float]
    ) -> dict[str, Any] | None:
        if clearance is None:
            return None
        tree = clearance.get("nearest_tree_xyz_local_m")
        wire = clearance.get("nearest_wire_xyz_local_m")
        return {
            "minimum_clearance_m": clearance.get("minimum_clearance_m"),
            "status": clearance.get("status"),
            "nearest_wire_id": clearance.get("nearest_wire_id"),
            "tree_xyz_absolute_m": self._absolute(tree, origin),
            "wire_xyz_absolute_m": self._absolute(wire, origin),
        }

    def spans_tree_preview(self, span_id: str, tree_id: str) -> dict[str, Any]:
        import numpy as np

        with self._opened() as session:
            span = self._tree_span(session, span_id)
            index, run_dir = self._tree_run(session, span)
            origin = self._tree_origin(session, index)
            records = (
                json.loads((run_dir / "reconciliation.json").read_bytes()).get(
                    "global_tree_records"
                )
                or []
            )
            target = next(
                (
                    record
                    for record in records
                    if int(record.get("global_tree_id", -1)) == int(tree_id)
                ),
                None,
            )
            if target is None:
                raise self._tree_error(404, "TREE_NOT_FOUND", f"Tree {tree_id} not in {span_id}")
            identities = [
                int(item[1]) for item in target.get("merged_span_tree_keys") or [] if len(item) >= 2
            ]
            if not identities and target.get("primary_span_tree_id") is not None:
                identities = [int(target["primary_span_tree_id"])]
            try:
                members = np.load(
                    str(run_dir / "tree" / "tree_members.npz"), mmap_mode="r", allow_pickle=True
                )
            except OSError as exc:
                raise self._tree_error(
                    409, "TREE_MEMBERS_MISSING", "tree member artifact missing"
                ) from exc
            rows = np.flatnonzero(np.isin(np.asarray(members["tree_id"]), identities))
            full_count = int(len(rows))
            sample = np.asarray(members["xyz_local_m"], dtype=np.float64)[rows]
            cap = 40_000
            if full_count > cap:
                sample = sample[np.linspace(0, full_count - 1, cap, dtype=np.int64)]
            absolute = sample + np.asarray(origin, dtype=np.float64)
            bounds = (
                None
                if not full_count
                else {
                    "min_xyz_absolute_m": list(np.min(sample, axis=0) + np.asarray(origin)),
                    "max_xyz_absolute_m": list(np.max(sample, axis=0) + np.asarray(origin)),
                }
            )
            identity = self._tree_identity(target)
            return {
                "span_id": span_id,
                "global_tree_id": target.get("global_tree_id"),
                "persistent_tree_id": identity["persistent_tree_id"],
                "identity_state": target.get("identity_state", "LEGACY_RUN_LOCAL"),
                "candidate_persistent_tree_ids": identity["candidate_persistent_tree_ids"],
                "prediction_identity": {
                    "persistent_tree_id": identity["persistent_tree_id"],
                    "global_tree_id": target.get("global_tree_id"),
                    "needs_review": identity["needs_review"],
                    "eligible_for_long_term_history": identity["prediction_eligible"],
                },
                "full_member_count": full_count,
                "sample_count": int(len(sample)),
                "sampled_xyz_absolute_m": absolute.tolist(),
                "bounds_absolute_m": bounds,
                "centroid_xyz_absolute_m": list(np.mean(absolute, axis=0)) if full_count else None,
            }
