"""Importable synthetic science seam for real spawn tests; no real LAS input."""

from __future__ import annotations

import json
import os
import time
from dataclasses import replace

import numpy as np
from algorithms.spans.clearance import run_span_tree_wire_clearance
from algorithms.spans.project_adapter import (
    PROJECT_SPAN_VEGETATION_ROLE_CONFIGURATION_DIGEST,
    wire_role_index_configuration_digest,
)
from algorithms.spans.role_tiles import RoleSpecificTileCache
from algorithms.vegetation.coarse_to_fine import _RawRegionWriter

from tests.stage_b.test_batch_exact_cache import _exact_result, _preparation, _write_json


def prepared_fixture(root, input_data):
    preparation = _preparation(root, input_data)
    np.save(root / "ground.npy", np.asarray([[0.0, 0.0, 0.0], [30.0, 0.0, 0.0]]))
    writer = _RawRegionWriter(root / "prepared-region", origin_xyz=(0.0, 0.0, 0.0))
    writer.append(
        np.asarray([[5.0, 0.0, 10.0], [15.0, 0.0, 10.0], [25.0, 0.0, 10.0]]),
        np.asarray([1, 2, 3], dtype=np.int64),
        np.asarray([5, 5, 5], dtype=np.uint8),
    )
    region = writer.close()
    references = {}
    for role, digest in (
        ("wire", wire_role_index_configuration_digest()),
        ("vegetation", PROJECT_SPAN_VEGETATION_ROLE_CONFIGURATION_DIGEST),
    ):
        cache_root = root / "artifacts" / "project-runs" / "source-shared" / role
        if (cache_root / "role-tile-cache.json").is_file():
            RoleSpecificTileCache.load(
                cache_root,
                role=role,
                source_identity=input_data.source_identity,
                source_point_count=input_data.source_point_count,
                config_digest=digest,
            )
        else:
            RoleSpecificTileCache.prepare_from_raw_region(
                region,
                cache_root,
                role=role,
                source_identity=input_data.source_identity,
                source_point_count=input_data.source_point_count,
                config_digest=digest,
            )
        references[f"{role}_role_tile_index_reference"] = cache_root.relative_to(root).as_posix()
    manifest_path = root / preparation.manifest_reference
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload.update(
        source_point_count=input_data.source_point_count,
        coarse_ground_reference="ground.npy",
        origin_xyz=[0.0, 0.0, 0.0],
        **references,
    )
    _write_json(manifest_path, payload)
    return preparation


def _attempt(output_root):
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "attempts.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"pid": os.getpid(), "started": time.monotonic()}) + "\n")


def fixture_executor(input_data, _preparation, output_root, *, config):
    _attempt(output_root)
    result = _exact_result(input_data, output_root, config)
    # A real exact clearance operation on tiny geometry includes danger points.
    result.wire_result.w1_arrays["sampled_curve_xyz"][:, 2] = 10.1
    np.savez(output_root / "wire" / "w1_arrays.npz", **result.wire_result.w1_arrays)
    clearance = run_span_tree_wire_clearance(
        result.reconciled,
        [result.wire_result],
        config,
        danger_threshold_m=result.risk_threshold_m,
    )
    clearance_path = output_root / "clearance.json"
    _write_json(
        clearance_path,
        {
            "schema": "treeline-span-clearance-result-v1",
            "status": clearance.status,
            "global_minimum_clearance_m": clearance.global_minimum_clearance_m,
            "records": [record.to_dict() for record in clearance.records],
            "danger_points": [point.to_dict() for point in clearance.danger_points],
            "diagnostics": dict(clearance.diagnostics),
        },
    )
    return replace(
        result,
        minimum_clearance_m=clearance.global_minimum_clearance_m,
        clearance_result=clearance,
        artifact_files={**result.artifact_files, "clearance": str(clearance_path)},
        identity_terminal_preparation=object(),
        identity_terminal_publish=lambda _value: result,
    )


def _reverse(input_data, preparation, output_root, config, span_count):
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "worker-started").write_text(str(os.getpid()), encoding="ascii")
    deadline = time.monotonic() + 20
    while len(list(output_root.parent.glob("*/worker-started"))) < span_count:
        if time.monotonic() > deadline:
            raise RuntimeError("FIXTURE_SPAWN_BARRIER_TIMEOUT")
        time.sleep(0.01)
    time.sleep(0.3 * (span_count - int(input_data.span_id[1:])))
    return fixture_executor(input_data, preparation, output_root, config=config)


def reverse_two_executor(input_data, preparation, output_root, *, config):
    return _reverse(input_data, preparation, output_root, config, 2)


def reverse_three_executor(input_data, preparation, output_root, *, config):
    return _reverse(input_data, preparation, output_root, config, 3)


def mixed_failure_executor(input_data, preparation, output_root, *, config):
    if input_data.span_id == "S002":
        _attempt(output_root)
        raise RuntimeError("FIXTURE_TREE_FAILURE: original science error")
    return fixture_executor(input_data, preparation, output_root, config=config)


def crashing_executor(input_data, preparation, output_root, *, config):
    if input_data.span_id == "S002":
        _attempt(output_root)
        # Let the first completed Span reach parent cache publication before the
        # second process exits abruptly; never invoke this outside a subprocess.
        deadline = time.monotonic() + 20
        while not (output_root.parent / "span-0" / "span-cache-index.json").is_file():
            if time.monotonic() > deadline:
                raise RuntimeError("FIXTURE_CACHE_BARRIER_TIMEOUT")
            time.sleep(0.01)
        os._exit(7)
    return fixture_executor(input_data, preparation, output_root, config=config)


def invalid_result_executor(_input_data, _preparation, _output_root, *, config):
    return None
