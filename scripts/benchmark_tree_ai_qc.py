"""Render and benchmark SmolVLM2 against existing real human-review evidence."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
from core.ai_qc.renderer import render_tree_views
from core.ai_qc.runtime import LlamaServerSession


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-root", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def _cases(batch_root: Path) -> list[dict[str, object]]:
    original = np.load(batch_root / "tree_segmentation_original.npz", allow_pickle=False)
    final = np.load(batch_root / "tree_segmentation_final.npz", allow_pickle=False)
    try:
        original_ids = set(int(value) for value in np.unique(original["tree_id"]))
        final_ids = set(int(value) for value in np.unique(final["tree_id"]))
    finally:
        original.close()
        final.close()
    journal = json.loads((batch_root / "tree_edits.json").read_text(encoding="utf-8"))
    affected: set[int] = set()
    for operation in journal.get("operations", []):
        affected.update(int(value) for value in operation.get("tree_ids", []))
    cases: list[dict[str, object]] = []
    for tree_id in sorted(original_ids):
        if tree_id in affected:
            cases.append(
                {
                    "case_id": f"PRE_EDIT_T{tree_id:03d}",
                    "source": "original",
                    "tree_id": tree_id,
                    "reference": "REVIEW_REQUIRED",
                    "reference_basis": "existing human REMOVE_POINTS operation affected this tree",
                }
            )
    for tree_id in sorted(final_ids):
        cases.append(
            {
                "case_id": f"FINAL_T{tree_id:03d}",
                "source": "final",
                "tree_id": tree_id,
                "reference": "PASS" if tree_id not in affected else "NO_REFERENCE",
                "reference_basis": (
                    "finalized tree untouched by the only persisted human point-edit operation"
                    if tree_id not in affected
                    else "human-edited final membership; no explicit KEEP label"
                ),
            }
        )
    return cases


def main() -> int:
    args = _args()
    args.output.mkdir(parents=True, exist_ok=True)
    cases = _cases(args.batch_root)
    if args.limit is not None:
        cases = cases[: args.limit]
    arrays: dict[str, dict[str, np.ndarray]] = {}
    for source in {str(case["source"]) for case in cases}:
        archive = np.load(args.batch_root / f"tree_segmentation_{source}.npz", allow_pickle=False)
        try:
            arrays[source] = {key: np.asarray(archive[key]) for key in archive.files}
        finally:
            archive.close()
    render_total = 0.0
    for case in cases:
        source = str(case["source"])
        tree_id = int(case["tree_id"])
        selected = arrays[source]["tree_id"] == tree_id
        rendered = render_tree_views(
            tree_id=str(case["case_id"]),
            tree_xyz=arrays[source]["xyz_local_m"][selected],
            tree_original_index=arrays[source]["original_index"][selected],
            scene_xyz=arrays[source]["xyz_local_m"],
            scene_tree_id=arrays[source]["tree_id"],
            selected_tree_id=tree_id,
            output_dir=args.output / str(case["case_id"]),
            source_editor_revision="PRE_EDIT" if source == "original" else "HUMAN_EDIT_SEQUENCE_1",
        )
        render_total += rendered.elapsed_seconds
        case["render"] = {
            "contact_sheet": str(rendered.contact_sheet),
            "contact_sheet_sha256": rendered.contact_sheet_sha256,
            "membership_sha256": rendered.membership_sha256,
            "seconds": rendered.elapsed_seconds,
        }
    del arrays
    results: list[dict[str, object]] = []
    runtime = args.asset_root / "runtimes" / "ai_qc" / "llama-server.exe"
    model = args.asset_root / "models" / "tree_qc" / "model.gguf"
    mmproj = args.asset_root / "models" / "tree_qc" / "mmproj.gguf"
    scene_started = time.perf_counter()
    with LlamaServerSession(server_exe=runtime, model_path=model, mmproj_path=mmproj) as session:
        for case in cases:
            prediction, metrics = session.classify(
                tree_id=str(case["case_id"]),
                contact_sheet=Path(str(case["render"]["contact_sheet"])),
            )
            results.append(
                {
                    **case,
                    "prediction": prediction.to_dict(),
                    "model_load_seconds": metrics.model_load_seconds,
                    "inference_seconds": metrics.inference_seconds,
                    "peak_rss_bytes": metrics.peak_rss_bytes,
                    "raw_model_text": metrics.raw_text,
                }
            )
    total_seconds = time.perf_counter() - scene_started
    known_bad = [item for item in results if item["reference"] == "REVIEW_REQUIRED"]
    known_good = [item for item in results if item["reference"] == "PASS"]
    false_pass = [item["case_id"] for item in known_bad if item["prediction"]["status"] == "PASS"]
    false_review = [
        item["case_id"] for item in known_good if item["prediction"]["status"] == "REVIEW_REQUIRED"
    ]
    summary = {
        "schema": "tree-ai-qc-real-benchmark-v1",
        "tree_count": len(results),
        "known_bad_cases": len(known_bad),
        "known_good_cases": len(known_good),
        "known_bad_false_pass": false_pass,
        "known_good_flagged": false_review,
        "pass_count": sum(item["prediction"]["status"] == "PASS" for item in results),
        "review_required_count": sum(
            item["prediction"]["status"] == "REVIEW_REQUIRED" for item in results
        ),
        "uncertain_count": sum(item["prediction"]["status"] == "UNCERTAIN" for item in results),
        "obvious_error_recall": (
            sum(item["prediction"]["status"] == "REVIEW_REQUIRED" for item in known_bad)
            / len(known_bad)
            if known_bad
            else None
        ),
        "render_seconds_total": render_total,
        "render_seconds_per_tree": render_total / len(results),
        "model_load_seconds": results[0]["model_load_seconds"] if results else None,
        "inference_seconds_per_tree": statistics.mean(
            float(item["inference_seconds"]) for item in results
        ),
        "total_real_scene_qc_seconds": total_seconds + render_total,
        "peak_rss_bytes": max((int(item["peak_rss_bytes"]) for item in results), default=0),
        "cases": results,
    }
    report = args.output / "benchmark.json"
    report.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(report)
    print(json.dumps({key: value for key, value in summary.items() if key != "cases"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
