"""Resource-bounded Stage-B execution; no source preparation or batch publication."""

from __future__ import annotations

import ctypes
import importlib
import inspect
import math
import multiprocessing
import os
import sys
import time
import traceback
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict, dataclass, fields, is_dataclass, replace
from functools import partial
from multiprocessing.reduction import ForkingPickler
from pathlib import Path
from typing import Any

import numpy as np
from algorithms.spans.project_adapter import (
    SourceSharedPreparation,
    SpanAnalysisInput,
    SpanScienceResult,
    run_canonical_span_science,
)

from core.config.fast_baseline import FastBaselineConfig
from core.stage_b.failures import structured_failure


@dataclass(frozen=True, slots=True)
class HostResources:
    cpu_count: int | None
    available_bytes: int | None
    process_bytes: int | None


def host_resources() -> HostResources:
    """Read actual capability; unavailable measurements are never guessed."""
    cpu = os.cpu_count()
    try:
        import psutil

        available = psutil.virtual_memory().available
        resident = psutil.Process().memory_info().rss
        if any(type(value) is not int or value <= 0 for value in (available, resident)):
            raise ValueError("invalid optional memory measurement")
        return HostResources(cpu, available, resident)
    except Exception:  # optional psutil can also raise AccessDenied/other Error subclasses
        pass
    if os.name != "nt":
        return HostResources(cpu, None, None)
    try:
        from ctypes import wintypes

        class MemoryStatus(ctypes.Structure):
            _fields_ = [("length", wintypes.DWORD), ("load", wintypes.DWORD)] + [
                (name, ctypes.c_ulonglong)
                for name in (
                    "total_physical",
                    "available_physical",
                    "total_page_file",
                    "available_page_file",
                    "total_virtual",
                    "available_virtual",
                    "available_extended_virtual",
                )
            ]

        class ProcessMemory(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("page_faults", wintypes.DWORD)] + [
                (name, ctypes.c_size_t)
                for name in (
                    "peak_working_set",
                    "working_set",
                    "peak_paged_pool",
                    "paged_pool",
                    "peak_nonpaged_pool",
                    "nonpaged_pool",
                    "pagefile",
                    "peak_pagefile",
                )
            ]

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        memory_status = kernel.GlobalMemoryStatusEx
        memory_status.argtypes = [ctypes.POINTER(MemoryStatus)]
        memory_status.restype = wintypes.BOOL
        status = MemoryStatus()
        status.length = ctypes.sizeof(status)
        current_process = kernel.GetCurrentProcess
        current_process.argtypes = []
        current_process.restype = wintypes.HANDLE
        process_info = ctypes.WinDLL("psapi", use_last_error=True).GetProcessMemoryInfo
        process_info.argtypes = [wintypes.HANDLE, ctypes.POINTER(ProcessMemory), wintypes.DWORD]
        process_info.restype = wintypes.BOOL
        process = ProcessMemory()
        process.cb = ctypes.sizeof(process)
        if not memory_status(ctypes.byref(status)) or not process_info(
            current_process(), ctypes.byref(process), ctypes.sizeof(process)
        ):
            return HostResources(cpu, None, None)
        return HostResources(cpu, int(status.available_physical), int(process.working_set))
    except (AttributeError, OSError, ValueError):
        return HostResources(cpu, None, None)


@dataclass(frozen=True, slots=True)
class SpanJob:
    span_id: str
    input_data: SpanAnalysisInput
    preparation: SourceSharedPreparation
    output_root: Path
    config: FastBaselineConfig

    def arguments(self) -> tuple[Any, ...]:
        return self.input_data, self.preparation, self.output_root, self.config


@dataclass(frozen=True, slots=True)
class SpanWorkset:
    vegetation_points: int
    wire_points: int
    chm_cells: int
    working_bytes: int
    result_bytes: int
    transfer_bytes: int


def estimate_worksets(project_root: Path, jobs: Sequence[SpanJob]) -> dict[str, SpanWorkset]:
    """Use verified role metadata and canonical regions, never read the RAW source.

    The canonical loaders verify the prepared cache (including tile integrity).
    Query planning itself reads metadata only, and counts entire intersecting
    tiles as an upper bound rather than guessing exact oriented point counts.
    """
    from algorithms.spans.canonical_executor import (
        _read_preparation,
        _resolve_reference,
        _span_record,
        _tile_cache_from_preparation,
    )
    from algorithms.spans.region import SpanLocalFrame, SpanRegionSpec

    if not jobs:
        return {}
    first = jobs[0]
    payload = _read_preparation(project_root, first.preparation, first.input_data)
    caches = {
        role: _tile_cache_from_preparation(project_root, payload, first.input_data, role)
        for role in ("wire", "vegetation")
    }
    if any(cache is None for cache in caches.values()):
        raise ValueError("ROLE_WORKSET_METADATA_UNAVAILABLE")
    ground_bytes = (
        _resolve_reference(project_root, payload["coarse_ground_reference"]).stat().st_size
    )
    source_points = payload.get("source_point_count")
    if type(source_points) is not int or source_points < 0:
        raise ValueError("SOURCE_WORKSET_METADATA_UNAVAILABLE")
    estimates = {}
    for job in jobs:
        if (
            job.input_data.source_identity != first.input_data.source_identity
            or job.preparation != first.preparation
        ):
            raise ValueError("SOURCE_WORKSET_BINDING_MISMATCH")
        span = _span_record(job.input_data, job.config)
        frame = SpanLocalFrame.from_endpoints(
            span.start_xyz_absolute_m, span.end_xyz_absolute_m, tuple(payload["origin_xyz"])
        )
        counts = {}
        vegetation_tiles = []
        for role, cache in caches.items():
            spec = SpanRegionSpec(
                role="vegetation-search" if role == "vegetation" else role,
                half_width_m=job.config.span_half_width_m,
                longitudinal_before_m=job.config.span_halo_m,
                longitudinal_after_m=job.config.span_halo_m,
                z_min_m=float(span.halo_bounds_min_xyz[2]),
                z_max_m=float(span.halo_bounds_max_xyz[2]),
            )
            _, tile_ids = cache.plan_oriented_query(frame, spec)
            items = [cache.index[tile_id] for tile_id in tile_ids]
            if any(
                type(item["point_count"]) is not int or item["point_count"] < 0 for item in items
            ):
                raise ValueError("ROLE_WORKSET_POINT_COUNT_INVALID")
            counts[role] = sum(item["point_count"] for item in items)
            if role == "vegetation":
                vegetation_tiles = items
        cells = 0
        if vegetation_tiles:
            lower = np.min([item["min_xyz"][:2] for item in vegetation_tiles], axis=0)
            upper = np.max([item["max_xyz"][:2] for item in vegetation_tiles], axis=0)
            resolution = job.config.chm_resolution_m
            if (
                not np.isfinite([*lower, *upper, resolution]).all()
                or resolution <= 0
                or np.any(upper < lower)
            ):
                raise ValueError("ROLE_WORKSET_BOUNDS_INVALID")
            # Floor-aligned CHM origins can add a boundary cell on each axis.
            cells = math.prod(math.ceil(float(width) / resolution) + 2 for width in upper - lower)
            # Canonical CHM fails above this limit before allocating the grid.
            cells = min(cells, 12_000_000)
        vegetation, wire = counts["vegetation"], counts["wire"]
        transport_input = len(ForkingPickler.dumps(job.arguments()))
        # Deliberately broad planning allowances, not a claim of exact RSS:
        # fine/raw/segmentation work arrays, full result graphs, and pickle copies.
        result_bytes = 512 * (vegetation + wire) + 8 * transport_input
        working_bytes = (
            1024 * vegetation
            + 2048 * wire
            + 4 * 19 * cells
            + 4 * ground_bytes
            + 2 * source_points
            + 8 * transport_input
        )
        estimates[job.span_id] = SpanWorkset(
            vegetation,
            wire,
            cells,
            working_bytes,
            result_bytes,
            2 * result_bytes + 2 * transport_input,
        )
    return estimates


def retained_size(value: Any) -> int:
    """Conservative resident/transport size without serializing large arrays."""
    seen = set()

    def size(item: Any) -> int:
        if id(item) in seen:
            return 0
        seen.add(id(item))
        own = sys.getsizeof(item)
        if isinstance(item, np.ndarray):
            return own + item.nbytes
        if is_dataclass(item) and not isinstance(item, type):
            return own + sum(
                size(getattr(item, field.name))
                for field in fields(item)
                if not field.name.startswith("identity_terminal_")
            )
        if isinstance(item, Mapping):
            return own + sum(size(key) + size(child) for key, child in item.items())
        if isinstance(item, (tuple, list, set, frozenset)):
            return own + sum(size(child) for child in item)
        return own

    return size(value)


def validate_test_executor(executor: Callable[..., Any]) -> None:
    """An explicit private test seam; closures/bound objects are never admitted."""
    if (
        not inspect.isfunction(executor)
        or "<locals>" in executor.__qualname__
        or executor.__module__ == "__main__"
    ):
        raise ValueError("process test executor must be an importable top-level function")
    module = importlib.import_module(executor.__module__)
    if getattr(module, executor.__name__, None) is not executor:
        raise ValueError("process test executor must be an importable top-level function")
    ForkingPickler.dumps(executor)


def canonical_result(
    input_data: SpanAnalysisInput,
    preparation: SourceSharedPreparation,
    output_root: Path,
    config: FastBaselineConfig,
    *,
    executor_override=None,
) -> SpanScienceResult:
    from algorithms.spans.canonical_executor import run_tracked_canonical_span_science

    executor = executor_override or run_tracked_canonical_span_science
    return run_canonical_span_science(
        input_data, preparation, output_root, executor=partial(executor, config=config)
    )


@dataclass(frozen=True, slots=True)
class WorkerOutcome:
    result: SpanScienceResult | None
    failure: Mapping[str, Any] | None
    elapsed_seconds: float
    process_id: int


def run_span_worker(
    input_data: SpanAnalysisInput,
    preparation: SourceSharedPreparation,
    output_root: Path,
    config: FastBaselineConfig,
    *,
    executor_override=None,
) -> WorkerOutcome:
    """Windows-spawn entry point: one exclusive Span, no parent-owned publication."""
    started = time.perf_counter()
    try:
        result = canonical_result(
            input_data, preparation, output_root, config, executor_override=executor_override
        )
        result = replace(result, identity_terminal_preparation=None, identity_terminal_publish=None)
        return WorkerOutcome(result, None, time.perf_counter() - started, os.getpid())
    except Exception as exc:
        # Exception objects themselves can be non-picklable. Preserve the actual
        # science type/message/code/traceback in a simple, explicit failure payload.
        failure = structured_failure(exc, phase="SPAN_ANALYSIS")
        failure.update(failure_phase="WORKER_SCIENCE", traceback=traceback.format_exc())
        return WorkerOutcome(None, failure, time.perf_counter() - started, os.getpid())


def normalize_mode(mode: str) -> str:
    if not isinstance(mode, str) or mode.upper() not in {"AUTO", "SERIAL", "PARALLEL"}:
        raise ValueError("execution_mode must be AUTO, SERIAL or PARALLEL")
    return mode.upper()


def execution_decision(
    mode: str,
    jobs: Sequence[SpanJob],
    *,
    project_root: Path,
    retained_results: Mapping[str, SpanScienceResult],
    injected_runner: bool,
    executor_override=None,
) -> tuple[dict[str, Any], dict[str, SpanWorkset]]:
    mode = normalize_mode(mode)
    decision: dict[str, Any] = {
        "requested_mode": mode,
        "effective_mode": "SERIAL",
        "worker_count": 1,
        "reason": "EXPLICIT_SERIAL",
        "submitted_span_ids": [],
        "completion_order": [],
        "worker_process_ids": [],
        "max_in_flight": 0,
    }
    if mode == "SERIAL":
        return decision, {}
    if injected_runner:
        decision["reason"] = "INJECTED_RUNNER_SERIAL"
        return decision, {}
    if len(jobs) < 2:
        decision["reason"] = "FEWER_THAN_TWO_PENDING_SPANS"
        return decision, {}
    resources = host_resources()
    decision["host_resources"] = asdict(resources)
    if any(
        type(value) is not int or value <= 0
        for value in (resources.cpu_count, resources.available_bytes, resources.process_bytes)
    ):
        decision["reason"] = "HOST_CAPABILITY_UNKNOWN"
        return decision, {}
    try:
        if executor_override is not None:
            validate_test_executor(executor_override)
        for job in jobs:
            ForkingPickler.dumps(job.arguments())
        worksets = estimate_worksets(project_root, jobs)
        if set(worksets) != {job.span_id for job in jobs}:
            raise ValueError("INCOMPLETE_WORKSET_ESTIMATE")
    except Exception as exc:
        decision.update(
            reason="WORKSET_OR_TRANSPORT_UNKNOWN", detail=f"{type(exc).__name__}: {exc}"
        )
        return decision, {}
    reserve = max(resources.available_bytes // 4, resources.process_bytes)
    retained = retained_size(retained_results) + sum(
        item.result_bytes for item in worksets.values()
    )
    worker_base = 2 * resources.process_bytes
    fixed = reserve + 3 * retained  # all retained returns plus global reconcile/clearance scratch
    cpu_limit = min(61, max(1, resources.cpu_count // 2), len(jobs))
    costs = sorted(
        (item.working_bytes + item.transfer_bytes for item in worksets.values()), reverse=True
    )
    worker_count = 1
    for count in range(2, cpu_limit + 1):
        if fixed + count * worker_base + sum(costs[:count]) <= resources.available_bytes:
            worker_count = count
    decision.update(
        reserve_bytes=reserve,
        retained_result_cap_bytes=retained,
        reconciliation_reserve_bytes=2 * retained,
        fixed_bytes=fixed,
        worker_base_bytes=worker_base,
        cpu_limit=cpu_limit,
        available_bytes=resources.available_bytes,
        estimates={span_id: asdict(item) for span_id, item in worksets.items()},
    )
    if worker_count < 2:
        decision["reason"] = "CPU_LIMIT" if cpu_limit < 2 else "MEMORY_LIMIT"
    else:
        decision.update(
            effective_mode="PARALLEL",
            worker_count=worker_count,
            reason="CONSERVATIVE_RESOURCE_BUDGET",
        )
    return decision, worksets


class PoolStartupError(RuntimeError):
    """No scientific task has been submitted; serial fallback cannot overwrite it."""


class _OwnedSpawnContext:
    """Track only Process objects created by this pool, using public process APIs."""

    def __init__(self):
        self.context = multiprocessing.get_context("spawn")
        self.processes = []

    def __getattr__(self, name):
        return getattr(self.context, name)

    def Process(self, *args, **kwargs):  # noqa: N802 - multiprocessing context protocol
        process = self.context.Process(*args, **kwargs)
        self.processes.append(process)
        return process

    def stop_startup(self) -> None:
        # Called only before a scientific task is submitted. A hung import must
        # not turn the read-only startup timeout into an unbounded shutdown wait.
        for process in self.processes:
            if process.is_alive():
                process.terminate()
        deadline = time.monotonic() + 5
        for process in self.processes:
            if process.pid is not None:
                process.join(max(0, deadline - time.monotonic()))
                if process.is_alive():
                    process.kill()


def _worker_ready() -> int:
    return os.getpid()


def _infrastructure_failure(code: str, exc: Exception | str) -> WorkerOutcome:
    failure = structured_failure(RuntimeError(f"{code}:{exc}"), phase="SPAN_ANALYSIS")
    failure["failure_phase"] = "PROCESS_INFRASTRUCTURE"
    return WorkerOutcome(None, failure, 0.0, 0)


def execute_parallel(
    jobs: Sequence[SpanJob],
    decision: dict[str, Any],
    worksets: Mapping[str, SpanWorkset],
    *,
    executor_override=None,
) -> Iterator[tuple[SpanJob, WorkerOutcome]]:
    """Bound admission and preserve every completion; never rerun submitted work."""
    pool = None
    context = _OwnedSpawnContext()
    try:
        pool = ProcessPoolExecutor(max_workers=decision["worker_count"], mp_context=context)
        # The pool is lazy. This read-only handshake can fail safely before any
        # science submission; a later child failure is never a startup fallback.
        pool.submit(_worker_ready).result(timeout=30)
    except Exception as exc:
        context.stop_startup()
        if pool is not None:
            pool.shutdown(wait=False, cancel_futures=True)
        raise PoolStartupError(f"{type(exc).__name__}: {exc}") from exc

    waiting = list(jobs)
    running = {}
    completed = set()
    worker = partial(run_span_worker, executor_override=executor_override)
    fixed = decision["fixed_bytes"] + decision["worker_count"] * decision["worker_base_bytes"]
    try:
        while waiting or running:
            resources = host_resources()
            available = resources.available_bytes
            valid_memory = type(available) is int and available > 0
            available = min(available, decision["available_bytes"]) if valid_memory else 0
            active_bytes = sum(
                worksets[job.span_id].working_bytes + worksets[job.span_id].transfer_bytes
                for job in running.values()
            )
            while waiting and len(running) < decision["worker_count"]:
                job = waiting[0]
                estimate = worksets[job.span_id]
                cost = estimate.working_bytes + estimate.transfer_bytes
                if fixed + active_bytes + cost > available:
                    break
                # Treat even submit() failure as uncertain after this boundary.
                # Never overwrite that Span by automatically running it again.
                decision["submitted_span_ids"].append(job.span_id)
                future = pool.submit(worker, *job.arguments())
                waiting.pop(0)
                running[future] = job
                active_bytes += cost
                decision["max_in_flight"] = max(decision["max_in_flight"], len(running))
            if not running:
                reason = "MEMORY_CAPABILITY_LOST" if not valid_memory else "MEMORY_BUDGET_CHANGED"
                for job in waiting:
                    completed.add(job.span_id)
                    yield job, _infrastructure_failure(f"SPAN_PROCESS_{reason}", "not submitted")
                waiting.clear()
                break
            # No arbitrary science deadline: a long valid Span retains the same
            # semantics as SERIAL. Polling keeps admission/capability checks finite.
            done, _ = wait(running, timeout=1.0, return_when=FIRST_COMPLETED)
            for future in done:
                job = running.pop(future)
                outcome = (
                    future.result()
                )  # pool/transport errors handled below, never science fallback
                if type(outcome) is not WorkerOutcome:
                    raise RuntimeError("SPAN_PROCESS_RETURNED_INVALID_ENVELOPE")
                if outcome.result is not None:
                    fixed += 3 * max(
                        0, retained_size(outcome.result) - worksets[job.span_id].result_bytes
                    )
                completed.add(job.span_id)
                decision["completion_order"].append(job.span_id)
                if outcome.process_id not in decision["worker_process_ids"]:
                    decision["worker_process_ids"].append(outcome.process_id)
                yield job, outcome
    except Exception as exc:
        decision["process_failure"] = f"{type(exc).__name__}: {exc}"
        # Collect already delivered successes even if another child broke the pool.
        for future, job in list(running.items()):
            if job.span_id in completed or not future.done() or future.cancelled():
                continue
            try:
                outcome = future.result()
                if type(outcome) is WorkerOutcome:
                    completed.add(job.span_id)
                    yield job, outcome
            except Exception:
                pass
        for job in jobs:
            if job.span_id not in completed:
                completed.add(job.span_id)
                yield job, _infrastructure_failure("SPAN_PROCESS_POOL_OR_TRANSPORT_FAILED", exc)
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
