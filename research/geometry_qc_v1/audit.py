"""Read-only, offline geometry feasibility experiment. Never imported by product.

The diagnostic rule is deliberately not an accepted QC model. No PASS is issued
without validated negative controls. Sampled occupancy is not exact connectivity.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import threading
import time
from pathlib import Path

import numpy as np
from scipy import ndimage

SAMPLE_LIMIT = 16384


def rss_bytes():
    """Windows working set, without an additional runtime dependency."""
    from ctypes import wintypes

    class Counters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("faults", wintypes.DWORD)] + [
            (name, ctypes.c_size_t)
            for name in (
                "peak",
                "working",
                "peak_pool",
                "pool",
                "peak_nonpaged",
                "nonpaged",
                "pagefile",
                "peak_pagefile",
            )
        ]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    psapi.GetProcessMemoryInfo.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(Counters),
        wintypes.DWORD,
    ]
    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    if not psapi.GetProcessMemoryInfo(
        kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb
    ):
        raise ctypes.WinError(ctypes.get_last_error())
    return counters.working


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def membership_digest(indices):
    return hashlib.sha256(np.sort(np.asarray(indices, dtype="<i8")).tobytes()).hexdigest()


def sample(xyz, indices):
    """Stable source-index hash sampling; input row order never selects the sample."""
    indices = np.asarray(indices, dtype=np.uint64)
    hashed = indices.copy()
    hashed ^= hashed >> np.uint64(30)
    hashed *= np.uint64(0xBF58476D1CE4E5B9)
    hashed ^= hashed >> np.uint64(27)
    hashed *= np.uint64(0x94D049BB133111EB)
    hashed ^= hashed >> np.uint64(31)
    rows = np.argpartition(hashed, min(len(hashed), SAMPLE_LIMIT) - 1)[:SAMPLE_LIMIT]
    rows = rows[np.argsort(indices[rows])]
    return np.asarray(xyz[rows], dtype=np.float64)


def components(xyz, divisions):
    span = np.ptp(xyz, axis=0)
    scale = float(np.linalg.norm(span))
    if scale == 0:
        return {"largest": 1.0, "second": 0.0, "singleton": 0.0}
    cells = np.floor((xyz - xyz.min(axis=0)) / (scale / divisions)).astype(int)
    grid = np.zeros(tuple(cells.max(axis=0) + 1), dtype=bool)
    grid[tuple(cells.T)] = True
    labels, _ = ndimage.label(grid, structure=np.ones((3, 3, 3)))
    counts = np.bincount(labels[tuple(cells.T)])[1:]
    fractions = np.sort(counts)[::-1] / len(xyz)
    return {
        "largest": float(fractions[0]),
        "second": float(fractions[1]) if len(fractions) > 1 else 0.0,
        "singleton": float(np.count_nonzero(counts == 1) / len(xyz)),
    }


def features(xyz, indices):
    if len(xyz) < 3 or not np.isfinite(xyz).all():
        raise ValueError("INVALID_GEOMETRY")
    cloud = sample(xyz, indices)
    centered = cloud - cloud.mean(axis=0)
    eigenvalues, vectors = np.linalg.eigh(np.cov(centered[:, :2].T))
    projected = centered[:, :2] @ vectors
    widths = np.diff(np.quantile(projected, [0.02, 0.98], axis=0), axis=0)[0]
    minor, major = sorted(widths)
    z = cloud[:, 2]
    height = float(np.ptp(z))
    robust_height = float(np.quantile(z, 0.98) - np.quantile(z, 0.02))
    hist = np.histogram(z, bins=20)[0]
    result = {
        "point_count": len(xyz),
        "sample_count": len(cloud),
        "sample_z_extent_m": height,
        "robust_z_extent_m": robust_height,
        "width_major_m": float(major),
        "width_minor_m": float(minor),
        "width_aspect": float(major / max(minor, 1e-12)),
        "width_height_ratio": float(major / max(robust_height, 1e-12)),
        "xy_minor_major_variance": float(eigenvalues[0] / max(eigenvalues[1], 1e-12)),
        "empty_vertical_bin_fraction": float(np.count_nonzero(hist == 0) / 20),
        "max_vertical_gap_fraction": float(np.diff(np.sort(z)).max() / max(height, 1e-12)),
    }
    for divisions in (16, 32, 64):
        for key, value in components(cloud, divisions).items():
            result[f"component_{divisions}_{key}"] = value
    return result


def diagnostic(features):
    """Predeclared research hypothesis, not a release decision.

    Two material sampled components (each >=10%) surviving two grid resolutions
    justify inspecting disconnected geometry. 10% is an unvalidated materiality
    assumption, never fitted to a case; natural multi-stem/occluded trees confound it.
    Absence of this signal cannot establish PASS or rule out contamination.
    """
    flag = all(features[f"component_{n}_second"] >= 0.10 for n in (32, 64))
    return {
        "status": "REVIEW_REQUIRED" if flag else "UNCERTAIN",
        "issue": "GEOMETRY_AMBIGUOUS",
        "reason_codes": [
            "MATERIAL_DISCONNECTED_SAMPLED_GEOMETRY_AT_TWO_SCALES"
            if flag
            else "NO_VALIDATED_PASS_CRITERION"
        ],
    }


def run(manifest_path, output):
    manifest = read(manifest_path)
    baseline = rss_bytes()
    peak = [baseline]
    stop = threading.Event()

    def monitor():
        while not stop.wait(0.01):
            peak[0] = max(peak[0], rss_bytes())

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    started = time.perf_counter()
    cases, inputs, artifact_loads, scene_counts = [], [], [], {}
    try:
        for name, dataset in manifest["datasets"].items():
            root = Path(dataset["batch_root"])
            selected = [c for c in manifest["cases"] if c["dataset"] == name]
            journal_path = root / "tree_edits.json"
            journal_hash = sha256(journal_path)
            if dataset.get("editor_journal_sha256"):
                assert journal_hash == dataset["editor_journal_sha256"], "JOURNAL_CHANGED"
            journal = read(journal_path)
            inputs.append({"path": str(journal_path), "sha256": journal_hash})
            for source in ("original", "final"):
                subset = [c for c in selected if c["source"] == source and c["renderable"]]
                if not subset:
                    continue
                path = root / f"tree_segmentation_{source}.npz"
                digest = sha256(path)
                assert digest == dataset[f"{source}_artifact_sha256"], "ARTIFACT_CHANGED"
                inputs.append({"path": str(path), "sha256": digest})
                t = time.perf_counter()
                with np.load(path, allow_pickle=False) as archive:
                    xyz, indices, labels = (
                        archive[k] for k in ("xyz_local_m", "original_index", "tree_id")
                    )
                # One cached membership grouping per artifact, no source-LAS reads.
                order = np.argsort(labels, kind="stable")
                sorted_labels = labels[order]
                scene_counts[f"{name}/{source}"] = len(np.unique(labels[labels >= 0]))
                artifact_loads.append(
                    {"path": str(path), "load_and_group_seconds": time.perf_counter() - t}
                )
                for case in subset:
                    tree = case["tree_id"]
                    lo, hi = np.searchsorted(sorted_labels, [tree, tree], side="left")
                    hi = np.searchsorted(sorted_labels, tree, side="right")
                    rows = order[lo:hi]
                    assert len(rows), "TREE_MISSING"
                    member = indices[rows]
                    revision = membership_digest(member)
                    key = "pre_edit_revision" if source == "original" else "post_edit_revision"
                    assert revision == case[key], "MEMBERSHIP_CHANGED"
                    t = time.perf_counter()
                    f = features(xyz[rows], member)
                    elapsed = time.perf_counter() - t
                    d = time.perf_counter()
                    prediction = diagnostic(f)
                    decision_seconds = time.perf_counter() - d
                    cases.append(
                        {
                            **case,
                            "features": f,
                            "prediction": prediction,
                            "feature_seconds": elapsed,
                            "decision_seconds": decision_seconds,
                            "binding": {
                                "artifact_sha256": digest,
                                "membership_sha256": revision,
                                "editor_journal_sha256": journal_hash,
                                "source_sha256": journal.get("source_sha256"),
                            },
                        }
                    )
                del xyz, indices, labels, order, sorted_labels
            for case in selected:
                if not case["renderable"]:
                    cases.append(
                        {
                            **case,
                            "features": None,
                            "prediction": {
                                "status": "UNCERTAIN",
                                "issue": "GEOMETRY_AMBIGUOUS",
                                "reason_codes": ["REFERENCE_GEOMETRY_UNAVAILABLE"],
                            },
                        }
                    )
        # Exact post-read file hashes establish that this experiment did not write inputs.
        for entry in inputs:
            assert sha256(entry["path"]) == entry["sha256"], "INPUT_MUTATED"
    finally:
        stop.set()
        thread.join()
    totals = {}
    for prefix, label in (("BAD", "EXPLICIT_BAD"), ("GOOD", "EXPLICIT_GOOD")):
        group = [c for c in cases if c["reference_label"] == label]
        totals[f"{prefix}_TOTAL"] = len(group)
        for status, key in (
            ("PASS", "FALSE_PASS" if prefix == "BAD" else "PASS"),
            ("UNCERTAIN", "UNCERTAIN"),
            ("REVIEW_REQUIRED", "DETECTED" if prefix == "BAD" else "FALSE_FLAG"),
        ):
            totals[f"{prefix}_{key}"] = sum(c["prediction"]["status"] == status for c in group)
    payload = {
        "schema": "tree-geometry-qc-feasibility-v1",
        "cases": cases,
        "metrics": totals,
        "scene_tree_counts": scene_counts,
        "input_hashes": inputs,
        "cached_artifact_loads": artifact_loads,
        "audit_wall_seconds": time.perf_counter() - started,
        "feature_total_seconds": sum(c.get("feature_seconds", 0) for c in cases),
        "decision_total_seconds": sum(c.get("decision_seconds", 0) for c in cases),
        "peak_rss_delta_bytes_sampled_10ms": peak[0] - baseline,
        "product_accepted": False,
        "verdict": "GEOMETRY_QC_NOT_GOOD_ENOUGH",
    }
    Path(output).write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.manifest, args.output)
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in {"cases", "input_hashes"}}, indent=2
        )
    )
