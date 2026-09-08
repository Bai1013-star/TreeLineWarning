"""Render and benchmark local VLMs against the V2 tree AI-QC reference set."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from core.ai_qc.contracts import AiQcResult
from core.ai_qc.renderer import RENDER_REVISION, render_tree_views
from core.ai_qc.runtime import LlamaServerSession, RuntimeMetrics


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runtime", type=Path)
    parser.add_argument("--runtime-root", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--mmproj", type=Path)
    parser.add_argument("--model-name", default="render-only")
    parser.add_argument("--model-license", default="not-applicable")
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--gpu-layers", type=int, default=99)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--input-modes", default="contact,multi")
    parser.add_argument(
        "--case-ids",
        help="Optional comma-separated bounded subset (for resource/performance sampling).",
    )
    parser.add_argument("--timeout-seconds", type=float, default=600.0)
    parser.add_argument("--render-only", action="store_true")
    parser.add_argument("--reuse-renders", action="store_true")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _load_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "tree-ai-qc-reference-set-v2":
        raise ValueError("AI_QC_REFERENCE_SCHEMA_INVALID")
    return payload


def _validate_artifact(dataset: dict[str, Any], source: str) -> Path:
    artifact = Path(dataset["batch_root"]) / f"tree_segmentation_{source}.npz"
    expected = str(dataset[f"{source}_artifact_sha256"]).casefold()
    if _sha256(artifact).casefold() != expected:
        raise ValueError(f"AI_QC_REFERENCE_ARTIFACT_HASH_MISMATCH:{artifact}")
    return artifact


def _render_cases(manifest: dict[str, Any], output: Path, *, reuse: bool) -> list[dict[str, Any]]:
    trustworthy = [case for case in manifest["cases"] if case.get("renderable", True)]
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for case in trustworthy:
        grouped[(str(case["dataset"]), str(case["source"]))].append(case)
    rendered_cases: list[dict[str, Any]] = []
    for (dataset_name, source), cases in grouped.items():
        dataset = manifest["datasets"][dataset_name]
        artifact = _validate_artifact(dataset, source)
        with np.load(artifact, allow_pickle=False) as archive:
            arrays = {key: np.asarray(archive[key]) for key in archive.files}
        for case in cases:
            case_id = str(case["case_id"])
            tree_id = int(case["tree_id"])
            case_output = output / "renders" / case_id
            metadata_path = case_output / "render.json"
            if reuse and metadata_path.is_file():
                render_payload = json.loads(metadata_path.read_text(encoding="utf-8"))
                if render_payload.get("render_revision") != RENDER_REVISION:
                    raise ValueError(f"AI_QC_STALE_RENDER:{case_id}")
            else:
                selected = arrays["tree_id"] == tree_id
                if not np.any(selected):
                    raise ValueError(f"AI_QC_REFERENCE_TREE_MISSING:{case_id}")
                raw_revision = hashlib.sha256(
                    np.sort(np.asarray(arrays["original_index"][selected], dtype="<i8")).tobytes()
                ).hexdigest()
                revision_key = "pre_edit_revision" if source == "original" else "post_edit_revision"
                expected_revision = str(case[revision_key])
                if raw_revision.casefold() != expected_revision.casefold():
                    raise ValueError(f"AI_QC_REFERENCE_MEMBERSHIP_MISMATCH:{case_id}")
                rendered = render_tree_views(
                    tree_id=case_id,
                    tree_xyz=arrays["xyz_local_m"][selected],
                    tree_original_index=arrays["original_index"][selected],
                    scene_xyz=arrays["xyz_local_m"],
                    scene_tree_id=arrays["tree_id"],
                    selected_tree_id=tree_id,
                    output_dir=case_output,
                    source_editor_revision=expected_revision,
                )
                render_payload = json.loads(rendered.metadata_path.read_text(encoding="utf-8"))
            rendered_cases.append(
                {
                    **case,
                    "project_run": dataset["project_run"],
                    "render": {
                        "metadata": str(metadata_path),
                        "membership_sha256": render_payload["membership_sha256"],
                        "view_files": [
                            str(case_output / name) for name in render_payload["view_files"]
                        ],
                        "view_sha256": render_payload["view_sha256"],
                        "contact_sheet": str(case_output / render_payload["contact_sheet"]),
                        "contact_sheet_sha256": render_payload["contact_sheet_sha256"],
                        "elapsed_seconds": render_payload["elapsed_seconds"],
                    },
                }
            )
        del arrays
    return sorted(rendered_cases, key=lambda item: str(item["case_id"]))


def _fallback(tree_id: str, reason: str) -> AiQcResult:
    return AiQcResult(
        tree_id=tree_id,
        status="UNCERTAIN",
        issue="OTHER",
        confidence=0.0,
        reason_codes=(reason,),
        parser_fallback=True,
    )


def _metric_counts(results: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    bad = [item for item in results if item["reference_category"] == "EXPLICIT_BAD_VISUAL"]
    good = [item for item in results if item["reference_category"] == "EXPLICIT_GOOD"]
    return {
        "bad_case_runs": len(bad),
        "bad_false_pass": sum(item["prediction"]["status"] == "PASS" for item in bad),
        "bad_confident_false_pass": sum(
            item["prediction"]["status"] == "PASS"
            and float(item["prediction"]["confidence"]) >= threshold
            for item in bad
        ),
        "bad_detected": sum(item["prediction"]["status"] == "REVIEW_REQUIRED" for item in bad),
        "bad_uncertain": sum(item["prediction"]["status"] == "UNCERTAIN" for item in bad),
        "good_case_runs": len(good),
        "good_pass": sum(item["prediction"]["status"] == "PASS" for item in good),
        "good_false_flag": sum(
            item["prediction"]["status"] == "REVIEW_REQUIRED" for item in good
        ),
        "good_uncertain": sum(item["prediction"]["status"] == "UNCERTAIN" for item in good),
    }


def _repeatability(results: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in results:
        grouped[str(item["case_id"])].append(item)
    stable_status = 0
    stable_issue = 0
    pass_review_flip = 0
    confidence_ranges: list[float] = []
    unstable_cases: list[str] = []
    for case_id, case_runs in grouped.items():
        statuses = {str(item["prediction"]["status"]) for item in case_runs}
        issues = {str(item["prediction"]["issue"]) for item in case_runs}
        if len(statuses) == 1:
            stable_status += 1
        else:
            unstable_cases.append(case_id)
        if len(issues) == 1:
            stable_issue += 1
        if {"PASS", "REVIEW_REQUIRED"}.issubset(statuses):
            pass_review_flip += 1
        confidence = [float(item["prediction"]["confidence"]) for item in case_runs]
        confidence_ranges.append(max(confidence) - min(confidence))
    return {
        "case_count": len(grouped),
        "stable_status_cases": stable_status,
        "stable_issue_cases": stable_issue,
        "pass_review_flip_cases": pass_review_flip,
        "mean_confidence_range": statistics.mean(confidence_ranges) if confidence_ranges else None,
        "max_confidence_range": max(confidence_ranges, default=None),
        "unstable_cases": unstable_cases,
    }


def _mode_summary(results: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    inference = [float(item["inference_seconds"]) for item in results]
    return {
        **_metric_counts(results, threshold),
        "inference_seconds_per_tree": statistics.mean(inference) if inference else None,
        "inference_seconds_p95": (
            float(np.quantile(np.asarray(inference), 0.95)) if inference else None
        ),
        "peak_rss_bytes": max((int(item["peak_rss_bytes"]) for item in results), default=0),
        "peak_vram_bytes": max((int(item["peak_vram_bytes"]) for item in results), default=0),
        "parser_fallback_count": sum(item["prediction"]["parser_fallback"] for item in results),
        "runtime_error_count": sum(bool(item.get("runtime_error")) for item in results),
        "repeatability": _repeatability(results),
    }


def _preferred_mode(summaries: dict[str, dict[str, Any]]) -> str | None:
    if not summaries:
        return None

    def score(item: tuple[str, dict[str, Any]]) -> tuple[int, ...]:
        summary = item[1]
        return (
            int(summary["bad_confident_false_pass"]),
            int(summary["bad_false_pass"]),
            -int(summary["bad_detected"]),
            int(summary["good_false_flag"]),
            -int(summary["good_pass"]),
            int(summary["bad_uncertain"]) + int(summary["good_uncertain"]),
        )

    return min(summaries.items(), key=score)[0]


def _available_memory() -> int:
    try:
        import psutil

        return int(psutil.virtual_memory().available)
    except ImportError:
        return 0


def main() -> int:
    args = _args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = _load_manifest(args.manifest)
    cases = _render_cases(manifest, args.output, reuse=args.reuse_renders)
    reference_summary = {
        category: sum(case["reference_category"] == category for case in manifest["cases"])
        for category in (
            "EXPLICIT_GOOD",
            "EXPLICIT_BAD_VISUAL",
            "EDITED_BUT_VISUAL_LABEL_UNCERTAIN",
        )
    }
    render_index = {
        "schema": "tree-ai-qc-render-index-v2",
        "render_revision": RENDER_REVISION,
        "reference_counts": reference_summary,
        "cases": cases,
    }
    (args.output / "render-index.json").write_text(
        json.dumps(render_index, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if args.render_only:
        print(args.output / "render-index.json")
        return 0
    if args.case_ids:
        requested = {value.strip() for value in args.case_ids.split(",") if value.strip()}
        cases = [case for case in cases if case["case_id"] in requested]
        missing = requested - {str(case["case_id"]) for case in cases}
        if missing:
            raise ValueError(f"AI_QC_REFERENCE_CASE_UNKNOWN:{','.join(sorted(missing))}")
    if not all((args.runtime, args.runtime_root, args.model, args.mmproj)):
        raise ValueError("AI_QC_RUNTIME_ARGUMENTS_REQUIRED")
    modes = [mode.strip() for mode in args.input_modes.split(",") if mode.strip()]
    if not modes or any(mode not in {"contact", "multi"} for mode in modes):
        raise ValueError("AI_QC_INPUT_MODE_INVALID")
    threshold = float(manifest["confident_pass_threshold"])
    available_before = _available_memory()
    results: list[dict[str, Any]] = []
    gpu_layers = args.gpu_layers if args.device == "gpu" else 0
    started = time.perf_counter()
    with LlamaServerSession(
        server_exe=args.runtime,
        model_path=args.model,
        mmproj_path=args.mmproj,
        timeout_seconds=args.timeout_seconds,
        gpu_layers=gpu_layers,
    ) as session:
        model_load_seconds = session.model_load_seconds
        for mode in modes:
            for repeat in range(1, args.repeats + 1):
                for case in cases:
                    print(
                        f"AI_QC_CASE_START device={args.device} mode={mode} "
                        f"repeat={repeat} case={case['case_id']}",
                        flush=True,
                    )
                    error: str | None = None
                    try:
                        if mode == "contact":
                            prediction, metrics = session.classify(
                                tree_id=str(case["case_id"]),
                                contact_sheet=Path(case["render"]["contact_sheet"]),
                            )
                        else:
                            prediction, metrics = session.classify(
                                tree_id=str(case["case_id"]),
                                image_paths=[Path(path) for path in case["render"]["view_files"]],
                            )
                    except Exception as exc:  # noqa: BLE001 - record one case and continue the gate
                        prediction = _fallback(str(case["case_id"]), "RUNTIME_ERROR")
                        error = f"{type(exc).__name__}:{exc}"
                        metrics = RuntimeMetrics(
                            session.model_load_seconds,
                            0.0,
                            session.peak_rss_bytes,
                            session.peak_vram_bytes,
                            "",
                        )
                    results.append(
                        {
                            **case,
                            "input_mode": mode,
                            "repeat": repeat,
                            "prediction": prediction.to_dict(),
                            "inference_seconds": metrics.inference_seconds,
                            "peak_rss_bytes": metrics.peak_rss_bytes,
                            "peak_vram_bytes": metrics.peak_vram_bytes,
                            "raw_model_text": metrics.raw_text,
                            "runtime_error": error,
                        }
                    )
                    print(
                        f"AI_QC_CASE_DONE case={case['case_id']} "
                        f"status={prediction.status} seconds={metrics.inference_seconds:.3f}",
                        flush=True,
                    )
        peak_rss_bytes = session.peak_rss_bytes
        peak_vram_bytes = session.peak_vram_bytes
    process_released = session.process is None
    time.sleep(2.0)
    available_after = _available_memory()
    summaries = {
        mode: _mode_summary([item for item in results if item["input_mode"] == mode], threshold)
        for mode in modes
    }
    payload = {
        "schema": "tree-ai-qc-model-benchmark-v2",
        "model": {
            "name": args.model_name,
            "license": args.model_license,
            "model_path": str(args.model),
            "model_sha256": _sha256(args.model),
            "model_file_size": args.model.stat().st_size,
            "projector_path": str(args.mmproj),
            "projector_sha256": _sha256(args.mmproj),
            "projector_file_size": args.mmproj.stat().st_size,
            "runtime_path": str(args.runtime),
            "runtime_size": _directory_size(args.runtime_root),
            "device": args.device,
            "gpu_layers": gpu_layers,
        },
        "reference_counts": reference_summary,
        "tested_reference_counts": {
            category: sum(case["reference_category"] == category for case in cases)
            for category in ("EXPLICIT_GOOD", "EXPLICIT_BAD_VISUAL")
        },
        "confident_pass_threshold": threshold,
        "render_revision": RENDER_REVISION,
        "input_modes": modes,
        "repeats": args.repeats,
        "model_load_seconds": model_load_seconds,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_rss_bytes": peak_rss_bytes,
        "peak_vram_bytes": peak_vram_bytes,
        "available_memory_before": available_before,
        "available_memory_after_release": available_after,
        "runtime_process_released": process_released,
        "mode_summaries": summaries,
        "preferred_input_mode": _preferred_mode(summaries),
        "results": results,
    }
    report = args.output / f"benchmark-{args.device}.json"
    report.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(report)
    print(json.dumps({key: value for key, value in payload.items() if key != "results"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
