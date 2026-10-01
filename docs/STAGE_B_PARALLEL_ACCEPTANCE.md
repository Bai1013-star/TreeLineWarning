# Stage-B adaptive process execution

Scope: engineering orchestration and exact process transport only. Canonical
science, `FastBaselineConfig`, cache formats, global ownership, clearance and
export rules are unchanged. No REAL project was used for these tests.

## Public contract

`BatchAnalysisScheduler(..., execution_mode="AUTO")` accepts `AUTO`, `SERIAL`
and `PARALLEL`, case-insensitively. Execution policy is independent of the
scientific configuration/digests. `SERIAL` always retains the existing runner
path. Ordinary injected `science_runner` callables remain serial, including
closures. `PARALLEL` requests processes but does not bypass resource limits.

The decision is exposed as `result.telemetry.execution`, serialized under
`telemetry.EXECUTION`, and copied into `batch_diagnostics.execution`:

- `requested_mode`, `effective_mode`, `worker_count`, `reason`.
- When measured: `host_resources`, `cpu_limit`, `available_bytes`,
  `reserve_bytes`, `retained_result_cap_bytes`, `reconciliation_reserve_bytes`,
  `fixed_bytes`, `worker_base_bytes`, and per-Span `estimates`.
- Process audit: `submitted_span_ids`, `completion_order`,
  `worker_process_ids`, `max_in_flight`; errors can add `detail` or
  `process_failure`.

Principal reasons are `EXPLICIT_SERIAL`, `INJECTED_RUNNER_SERIAL`,
`FEWER_THAN_TWO_PENDING_SPANS`, `HOST_CAPABILITY_UNKNOWN`,
`WORKSET_OR_TRANSPORT_UNKNOWN`, `CPU_LIMIT`, `MEMORY_LIMIT`,
`CONSERVATIVE_RESOURCE_BUDGET`, and
`POOL_STARTUP_FAILED_BEFORE_SCIENCE_SUBMISSION`.

Clean COMPLETE restart still runs before source preparation, execution planning,
pool creation and persistence. It returns the persisted telemetry; requesting a
different mode does not rewrite or reanalyze an exact complete run. Edited,
finalized or malformed existing runs remain guarded before scheduler writes.

## Worker and parent ownership

`core.stage_b.parallel.run_span_worker` is a top-level Windows-spawn target
receiving `SpanAnalysisInput`, `SourceSharedPreparation`, `Path output_root`
and `FastBaselineConfig`. It invokes `run_canonical_span_science` with the
same `run_tracked_canonical_span_science` executor/config. Authority and result
validation therefore remain active. Only the unused
`identity_terminal_preparation` and `identity_terminal_publish` return fields
are cleared. Every scientific array, nested result, record and diagnostic stays.

The return envelope is `WorkerOutcome(result, failure, elapsed_seconds,
process_id)`. A science exception is represented by its existing structured
failure fields plus `failure_phase="WORKER_SCIENCE"` and traceback text; the
exception object itself need not be picklable. This is a failed Span, never a
fabricated successful result.

Source/Ground/role preparation runs once in the parent. A worker only writes the
existing science artifacts in its exclusive `batches/<run>/<span>` directory.
The parent publishes cache indices/status/telemetry and performs topology-ordered
global reconciliation, full clearance, editor persistence and export. Completion
order does not define scientific order. Exact cache hits are never submitted.

The private `_process_science_executor` constructor seam accepts only a
pickleable importable top-level function and cannot coexist with an injected
`science_runner`. It is for synthetic process acceptance. Both its serial and
process execution still go through the canonical validation wrapper. With no
override, the executor is always the production canonical implementation.

## Resource planning and failure handling

Actual CPU/free RAM/current-process RSS measurements are used. Optional psutil
errors do not abort analysis; Windows falls back to `GlobalMemoryStatusEx` and
`GetProcessMemoryInfo`. Missing measurements select SERIAL. There is no guessed
4 GiB capacity or machine-specific worker setting.

The estimator uses canonical preparation/role-cache validation, canonical Span
frames/specs and `plan_oriented_query`. It sums complete intersecting tile counts
as conservative vegetation/wire bounds. Cache integrity validation reads prepared
artifacts; capacity planning never opens the original LAS/LAZ or scans an oriented
point cloud. CHM dimensions use tile-bound extents and the existing 12-million-cell
allocation limit. These are planning allowances, not a certified RSS maximum:

- Per-return allowance: 512 bytes per vegetation/wire point plus input metadata.
- Work allowance: 1024 bytes per vegetation point, 2048 per wire point,
  four copies of the 19-byte CHM cell budget, ground/source-validation/input space.
- Transfer allowance: two full return copies plus input transport.
- Reserve: the larger of one quarter of measured available RAM and parent RSS.
- Every worker reserves twice parent RSS for interpreter/import overhead.
- All retained cached results and all predicted future returns remain budgeted;
  twice that total is additionally reserved for parent reconciliation/clearance.

The selected worker count fits the largest simultaneous jobs, at most half the
reported CPUs, at most the Windows ProcessPool limit and at most pending Spans.
Admission is a bounded window, not submission of the whole batch. Free RAM is
rechecked, and unexpectedly large returned graphs increase the retained budget.
After capacity is lost, unresolved unsubmitted work is explicitly failed rather
than reported COMPLETE; submitted work is never automatically rerun.

A read-only pool handshake precedes scientific submission. If it fails, serial
fallback is safe and separately recorded. Startup has a 30-second timeout;
tracked public Process objects belonging only to that pool are terminated on
startup failure, and shutdown is nonblocking. Running science retains the prior
serial behavior of no arbitrary total-duration cutoff; waits poll every second.
A process/transport failure after submission preserves delivered successes and
their caches, marks every unresolved Span FAILED, and does not overwrite partial
artifacts by silently retrying. User-triggered retry retains the existing exact
cache-plus-rerun union.

## Source acceptance

Windows, Python 3.11, synthetic fixtures in a uniquely named system TEMP directory:

```powershell
$parallelTemp = Join-Path $env:TEMP ('treeline-batch-parallel-core-' + [guid]::NewGuid().ToString('N'))
.venv311/Scripts/python.exe -m pytest tests/stage_b/test_batch_parallel.py tests/stage_b/test_batch_exact_cache.py tests/stage_b/test_stage_b_integration.py tests/stage_b/test_clearance_order_publication.py --basetemp $parallelTemp -p no:cacheprovider -q
.venv311/Scripts/python.exe -m ruff check core/stage_b/batch.py core/stage_b/parallel.py tests/stage_b/test_batch_parallel.py tests/stage_b/spawn_batch_fixtures.py
```

Result: **62 passed in 32.77 seconds**; scoped Ruff and `git diff --check` passed.
Coverage includes actual 2/3-process reversed completion and distinct child PIDs,
full membership/XYZ/IDs/lineage/W1/clearance/danger-point equivalence, exact
COMPLETE restart with no file/mtime changes, finalized export cell equivalence,
mixed science failure/retry/cache union, abrupt process death preserving a
completed cache, startup fallback/timeout behavior, changing memory admission,
all-retained-result budgeting, corrupted role metadata, non-picklable inputs,
injected closures and existing immutable-run guards. Export comparison validates
then normalizes only the deliberately different science-run IDs of the two
control runs; every other workbook cell is compared exactly.

No package build, EXE, server or REAL processing was run for this acceptance.
Source spawn is not proof of frozen execution. `desktop/app.py` already invokes
`multiprocessing.freeze_support`, and frozen children run the existing native
runtime hook themselves. The post-preparation canonical Span chain does not
import PDAL; it consumes prepared role and ground artifacts. A separate frozen
Stage-B transport diagnostic remains required before final package acceptance.
