"""HTTP acceptance of an already running, copied packaged Survey project.

The script never starts the application. DBH recomputation writes only to the
project served by --base-url, so point the packaged application at a test copy.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import time
import urllib.parse
import urllib.request
from pathlib import Path
from uuid import uuid4

from openpyxl import load_workbook

COLUMNS = (
    "tree_id", "species", "observation_date", "observation_year",
    "observation_month", "observation_day", "height", "age", "diameter",
    "vertical_distance", "horizontal_distance", "clearance_distance",
    "wind_offset_distance", "voltage_level_kv", "maximum_sag_winter",
    "maximum_sag_summer", "wind_speed", "rainfall", "humidity",
    "temperature", "sunlight_hours", "risk_score", "risk_level",
)


class Client:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")

    def get(self, path: str) -> bytes:
        request = urllib.request.Request(self.base_url + path, headers={"Accept": "*/*"})
        with urllib.request.urlopen(request, timeout=90) as response:
            return response.read()

    def json(self, path: str) -> dict:
        return json.loads(self.get(path))

    def post(self, path: str, payload: dict) -> dict:
        request = urllib.request.Request(
            self.base_url + path,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Accept": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=90) as response:
            return json.load(response)


def workbook_rows(payload: bytes) -> list[tuple]:
    workbook = load_workbook(io.BytesIO(payload), read_only=True, data_only=True)
    try:
        if workbook.sheetnames != ["树木风险结果"]:
            raise AssertionError(f"unexpected delivery sheets: {workbook.sheetnames}")
        rows = [tuple(row) for row in workbook.active.iter_rows(values_only=True)]
    finally:
        workbook.close()
    if not rows or rows[0] != COLUMNS:
        raise AssertionError(
            f"delivery columns differ from 23-column CSV: {rows[0] if rows else None}"
        )
    if any(len(row) != len(COLUMNS) for row in rows):
        raise AssertionError("delivery row width differs from CSV")
    return rows


def csv_value(value: object) -> str:
    return "" if value is None else str(value)


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_csv(path: Path, rows: list[tuple]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        for row in rows:
            writer.writerow([csv_value(value) for value in row])
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        parsed = list(csv.reader(stream))
    expected = [[csv_value(value) for value in row] for row in rows]
    if parsed != expected:
        raise AssertionError("generated CSV differs cell by cell from delivery workbook")


def collect(
    client: Client, output: Path, phase: str, expected_count: int,
    project_root: Path | None,
) -> dict:
    workspace = client.json("/api/desktop/workspace")
    review = client.json("/api/desktop/review?display_budget=1500")
    report = client.json("/api/current-report")
    batch = workspace["batch"]
    if report.get("status") != "FINALIZED":
        raise AssertionError(
            f"{phase}: current report is {report.get('status')}: {report.get('message')}"
        )
    if workspace.get("ui_state") != "FINALIZED":
        raise AssertionError(f"{phase}: workspace is {workspace.get('ui_state')}")
    review_trees = review.get("trees", [])
    report_trees = report.get("trees", [])
    counts = {
        "workspace": batch.get("final_tree_count"),
        "review": len(review_trees),
        "report": len(report_trees),
        "report_kpi": report.get("kpis", {}).get("tree_count"),
    }
    if set(counts.values()) != {expected_count}:
        raise AssertionError(f"{phase}: inconsistent current tree counts: {counts}")
    review_ids = {str(row["display_tree_id"]) for row in review_trees}
    report_ids = {str(row["display_tree_id"]) for row in report_trees}
    if len(review_ids) != expected_count or review_ids != report_ids:
        raise AssertionError(f"{phase}: review/report current IDs differ")
    member_counts = {
        str(row["display_tree_id"]): int(row["point_count"])
        for row in review_trees
    }
    display = review.get("trees3d") or {}
    displayed_membership = {
        key: display.get(key, [])
        for key in (
            "source_tree_id_by_index", "original_index_by_index",
            "visible_by_index", "tree_id_by_index",
        )
    }
    sample_digest = hashlib.sha256(
        json.dumps(displayed_membership, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    membership_files = {}
    if project_root is not None:
        batch_id = str(batch["batch_id"])
        artifact = project_root / "artifacts" / "project-runs" / "batches" / batch_id
        for name in ("tree_segmentation_final.npz", "tree_edits.json"):
            path = artifact / name
            if not path.is_file():
                raise AssertionError(f"{phase}: missing exact membership artifact {path}")
            membership_files[name] = file_hash(path)
    download_rows = []
    downloads = {}
    for name, route in (("prediction", "/download_prediction"), ("excel", "/download_excel")):
        payload = client.get(route)
        destination = output / f"{phase}_{name}.xlsx"
        destination.write_bytes(payload)
        rows = workbook_rows(payload)
        save_csv(output / f"{phase}_{name}.csv", rows)
        ids = [str(row[0]) for row in rows[1:]]
        if len(ids) != expected_count or set(ids) != report_ids or len(set(ids)) != len(ids):
            raise AssertionError(
                f"{phase}: {route} includes a deleted tree or misses a current tree"
            )
        download_rows.append(rows)
        downloads[name] = {
            "sha256": hashlib.sha256(payload).hexdigest(),
            "tree_count": len(ids),
            "columns": len(rows[0]),
        }
    if download_rows[0] != download_rows[1]:
        raise AssertionError(f"{phase}: the two Excel download routes differ cell by cell")
    # Legacy V3 projects may upgrade their DBH sidecar on the first export.
    # Read the report again so both artifacts describe the same published state.
    report = client.json("/api/current-report")
    if report.get("status") != "FINALIZED":
        raise AssertionError(f"{phase}: report changed state during export")
    report_trees = report.get("trees", [])
    if {str(row["display_tree_id"]) for row in report_trees} != report_ids:
        raise AssertionError(f"{phase}: report tree identities changed during export")
    final = client.json("/api/final")
    final_by_persistent = {
        str(row["persistent_tree_id"]): row for row in final.get("trees", [])
    }
    report_by_persistent = {
        str(row["persistent_tree_id"]): row for row in report_trees
    }
    if set(final_by_persistent) != set(report_by_persistent):
        raise AssertionError(f"{phase}: final projection/report tree identities differ")
    metric_values = {}
    for persistent_id, report_row in report_by_persistent.items():
        final_row = final_by_persistent[persistent_id]
        metric_values[persistent_id] = {}
        for field in ("height_m", "crown_width_m", "crown_area_m2"):
            actual = report_row.get(field)
            expected = final_row.get(field)
            if (
                not isinstance(actual, (int, float))
                or not math.isfinite(actual)
                or actual <= 0
                or not isinstance(expected, (int, float))
                or not math.isfinite(expected)
                or expected <= 0
            ):
                raise AssertionError(
                    f"{phase}: missing or invalid {field} for {persistent_id}: "
                    f"report={actual}, /api/final={expected}"
                )
            if not math.isclose(float(actual), float(expected), rel_tol=0, abs_tol=1e-6):
                raise AssertionError(
                    f"{phase}: {field} differs for {persistent_id}: "
                    f"report={actual}, /api/final={expected}"
                )
            metric_values[persistent_id][field] = float(actual)
    excel_diameter = {str(row[0]): row[8] for row in download_rows[0][1:]}
    excel_height = {str(row[0]): row[6] for row in download_rows[0][1:]}
    dbh_groups = {"measured": [], "geometry": [], "reference": []}
    for row in report_trees:
        tree_id = str(row["display_tree_id"])
        value = row.get("dbh_cm")
        delivered = excel_diameter[tree_id]
        delivered_height = excel_height[tree_id]
        report_height = row.get("height_m")
        if (
            not isinstance(delivered_height, (int, float))
            or not math.isfinite(delivered_height)
            or not math.isclose(
                float(delivered_height), float(report_height), rel_tol=0, abs_tol=1e-6
            )
        ):
            raise AssertionError(
                f"{phase}: Excel/report height differs for {tree_id}: "
                f"{delivered_height} != {report_height}"
            )
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise AssertionError(f"{phase}: report chest diameter invalid for {tree_id}: {value}")
        if (
            not isinstance(delivered, (int, float))
            or not math.isfinite(delivered)
            or delivered <= 0
        ):
            raise AssertionError(
                f"{phase}: Excel chest diameter invalid for {tree_id}: {delivered}"
            )
        if not math.isclose(float(value), float(delivered), rel_tol=0, abs_tol=1e-6):
            raise AssertionError(
                f"{phase}: report/Excel chest diameter differs for {tree_id}: "
                f"{value} != {delivered}"
            )
        method = row.get("dbh_method")
        quality = row.get("dbh_quality")
        if method == "REFERENCE_ESTIMATE":
            group = "reference"
            if row.get("dbh_reference_basis") not in {
                "PROJECT_TRUSTED_MEASUREMENTS", "USER_PROVIDED_REFERENCE"
            }:
                raise AssertionError(f"{phase}: reference provenance missing for {tree_id}")
        elif method in {"DIRECT_STEM", "RECOVERED_STEM", "REVIEW_CONFIRMED_STEM"}:
            group = "geometry" if quality in {"ESTIMATE", "ESTIMATED"} else "measured"
        elif method == "ALLOMETRIC_ESTIMATE":
            group = "geometry"
        else:
            raise AssertionError(f"{phase}: unexpected DBH method for {tree_id}: {method}")
        dbh_groups[group].append(tree_id)
    if expected_count == 10 and {key: len(value) for key, value in dbh_groups.items()} != {
        "measured": 2, "geometry": 2, "reference": 6
    }:
        raise AssertionError(
            f"{phase}: expected 2 measured, 2 geometric, 6 reference: {dbh_groups}"
        )
    return {
        "counts": counts,
        "current_display_ids": sorted(report_ids),
        "current_persistent_ids": sorted(str(row["persistent_tree_id"]) for row in report_trees),
        "current_member_point_counts": member_counts,
        "display_membership_sample_sha256": sample_digest,
        "membership_file_sha256": membership_files,
        "dbh_available_count": sum(row.get("dbh_cm") is not None for row in report_trees),
        "dbh_source_groups": dbh_groups,
        "validated_tree_metrics": metric_values,
        "dbh_measured_count": sum(
            row.get("dbh_cm") is not None
            and row.get("dbh_quality") not in {"ESTIMATE", "ESTIMATED"}
            and row.get("dbh_method") != "ALLOMETRIC_ESTIMATE"
            for row in report_trees
        ),
        "dbh_estimated_count": sum(
            row.get("dbh_cm") is not None
            and (row.get("dbh_quality") in {"ESTIMATE", "ESTIMATED"}
                 or row.get("dbh_method") == "ALLOMETRIC_ESTIMATE")
            for row in report_trees
        ),
        "dbh_unavailable_count": sum(row.get("dbh_cm") is None for row in report_trees),
        "dbh_human_confirmed": {
            str(row["persistent_tree_id"]): row.get("dbh_cm")
            for row in report_trees if row.get("dbh_method") == "REVIEW_CONFIRMED_STEM"
        },
        "downloads": downloads,
    }


def recompute(client: Client, timeout_seconds: int) -> dict:
    request_id = "packaged-acceptance-" + uuid4().hex
    started = client.post(
        "/api/dbh-review/recompute", {"async": True, "request_id": request_id}
    )
    if started.get("request_id") != request_id:
        raise AssertionError(f"DBH task did not acknowledge request {request_id}")
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        status = client.json(
            "/api/editor/mutation?request_id=" + urllib.parse.quote(request_id)
        )
        if status.get("request_id") != request_id:
            raise AssertionError("DBH task status belongs to another request")
        if status.get("state") == "COMMITTED":
            result = status.get("result") or {}
            if result.get("status") != "RECOMPUTED" or not result.get("workbook_updated"):
                raise AssertionError(f"DBH result was not published: {result}")
            return result
        if status.get("state") == "FAILED":
            raise AssertionError(f"DBH recomputation failed: {status.get('error')}")
        time.sleep(1)
    raise TimeoutError(f"DBH recomputation exceeded {timeout_seconds} seconds")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-tree-count", type=int, default=10)
    parser.add_argument("--project-root", type=Path,
                        help="Copied project directory for exact final NPZ and edit journal hashes")
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--skip-recompute", action="store_true")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    project_root = args.project_root.resolve() if args.project_root else None
    before = collect(Client(args.base_url), output, "before", args.expected_tree_count,
                     project_root)
    result = None
    after = None
    if not args.skip_recompute:
        result = recompute(Client(args.base_url), args.timeout_seconds)
        after = collect(Client(args.base_url), output, "after", args.expected_tree_count,
                        project_root)
        if before["current_persistent_ids"] != after["current_persistent_ids"]:
            raise AssertionError("DBH recomputation changed the reviewed tree membership")
        if before["current_member_point_counts"] != after["current_member_point_counts"]:
            raise AssertionError("DBH recomputation changed exact per-tree member counts")
        if before["display_membership_sample_sha256"] != after["display_membership_sample_sha256"]:
            raise AssertionError("DBH recomputation changed the visible editor membership")
        if (
            project_root is not None
            and before["membership_file_sha256"] != after["membership_file_sha256"]
        ):
            raise AssertionError("DBH recomputation changed final NPZ or the edit journal")
        if before["dbh_human_confirmed"] != after["dbh_human_confirmed"]:
            raise AssertionError("DBH recomputation changed a human-confirmed diameter")
    evidence = {"status": "PASS", "base_url": args.base_url, "before": before,
                "recompute": result, "after": after}
    (output / "acceptance.json").write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(evidence, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
