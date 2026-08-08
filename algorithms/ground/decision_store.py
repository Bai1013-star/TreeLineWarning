"""FGC uint8 decision routing, chunk publication, and FGCCOV1 coverage helpers.

Authority: docs/full-resolution-ground-classification-design.md section 10
(coverage) and section 12 (decision chunks), and the frozen Bundle A plan.

The decision routing array is a same-volume uint8 memmap over ``[0, N)``
initialised to UNROUTED; a row may be routed exactly once.  Published chunks
are raw uint8 files over ordered inclusive-exclusive contiguous ranges.  The
coverage helpers implement the deterministic external sort / k-way merge /
OR-dedup and the row-streaming run-union of unsupported 4-connected components
without a dense raster.
"""

from __future__ import annotations

import hashlib
import heapq
from collections.abc import Callable, Iterable, Iterator, Sequence
from pathlib import Path

import numpy as np
from core.contracts.ground_classification import (
    FGC_COVERAGE_FLAG_CONFIRMED,
    FGC_COVERAGE_FLAG_OBSERVED,
    FGC_COVERAGE_RECORD_SIZE,
    UNROUTED,
    FgcCoverageComponent,
    FgcCoverageResult,
    FgcDecisionChunkRecord,
    GroundClassificationPolicyConfig,
    GroundDecision,
    pack_coverage_record,
    unpack_coverage_record,
)


class DecisionConservationError(Exception):
    """Raised when routing conservation or duplicate-write rules are violated."""


# ---------------------------------------------------------------------------
# Decision routing store
# ---------------------------------------------------------------------------


class DecisionStore:
    """Same-volume uint8 decision authority over the full source domain."""

    def __init__(
        self,
        source_point_count: int,
        *,
        scratch_dir: Path | str,
        policy: GroundClassificationPolicyConfig,
    ) -> None:
        if source_point_count <= 0:
            raise ValueError("source_point_count must be positive")
        if policy.decision_memmap_bytes < source_point_count:
            raise ValueError("decision memmap is too small for the source domain")
        self.source_point_count = int(source_point_count)
        self.policy = policy
        self._path = Path(scratch_dir) / "fgc_decision_routing.dat"
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._array = np.memmap(
            self._path,
            mode="w+",
            dtype=np.uint8,
            shape=(self.source_point_count,),
        )
        self._array[:] = UNROUTED
        self._array.flush()

    def route(self, original_index: int, decision: GroundDecision) -> None:
        """Write one decision; a second write to the same row is rejected."""
        if isinstance(decision, int) and not isinstance(decision, GroundDecision):
            decision = GroundDecision(decision)
        if decision not in (
            GroundDecision.NON_GROUND,
            GroundDecision.CONFIRMED_GROUND,
            GroundDecision.REVIEW_GROUND,
        ):
            raise DecisionConservationError(f"cannot route non-terminal decision {decision!r}")
        index = int(original_index)
        if not 0 <= index < self.source_point_count:
            raise DecisionConservationError(
                f"original_index {index} is outside [0, {self.source_point_count})"
            )
        if self._array[index] != UNROUTED:
            raise DecisionConservationError(f"duplicate decision write at original_index {index}")
        self._array[index] = int(decision)

    def peek(self, original_index: int) -> GroundDecision:
        index = int(original_index)
        if not 0 <= index < self.source_point_count:
            raise ValueError(f"original_index {index} is outside [0, {self.source_point_count})")
        return GroundDecision(int(self._array[index]))

    def assert_no_unrouted(self) -> None:
        if np.count_nonzero(self._array == UNROUTED):
            raise DecisionConservationError("UNROUTED rows remain; conservation failed")

    def total_state_counts(self) -> dict[str, int]:
        non_ground = int(np.count_nonzero(self._array == GroundDecision.NON_GROUND))
        confirmed = int(np.count_nonzero(self._array == GroundDecision.CONFIRMED_GROUND))
        review = int(np.count_nonzero(self._array == GroundDecision.REVIEW_GROUND))
        unrouted = int(np.count_nonzero(self._array == UNROUTED))
        if non_ground + confirmed + review + unrouted != self.source_point_count:
            raise DecisionConservationError(
                "routing state counts do not conserve the source domain"
            )
        return {
            "non_ground": non_ground,
            "confirmed_ground": confirmed,
            "review_ground": review,
            "unrouted": unrouted,
        }

    # ------------------------------------------------------------------
    # Chunk publication
    # ------------------------------------------------------------------

    def write_chunks(self, decisions_dir: Path | str) -> tuple[FgcDecisionChunkRecord, ...]:
        """Write raw uint8 chunk files over ordered contiguous ranges."""
        self.assert_no_unrouted()
        directory = Path(decisions_dir)
        directory.mkdir(parents=True, exist_ok=True)
        chunk_rows = int(self.policy.decision_chunk_rows)
        records: list[FgcDecisionChunkRecord] = []
        start = 0
        chunk_index = 0
        while start < self.source_point_count:
            end = min(start + chunk_rows, self.source_point_count)
            data = np.ascontiguousarray(self._array[start:end])
            sha = hashlib.sha256(data.tobytes()).hexdigest()
            relative = f"chunk_{chunk_index:06d}.bin"
            chunk_path = directory / relative
            chunk_path.write_bytes(data.tobytes())
            state_counts = {
                "non_ground": int(np.count_nonzero(data == GroundDecision.NON_GROUND)),
                "confirmed_ground": int(np.count_nonzero(data == GroundDecision.CONFIRMED_GROUND)),
                "review_ground": int(np.count_nonzero(data == GroundDecision.REVIEW_GROUND)),
            }
            records.append(
                FgcDecisionChunkRecord(
                    start_index=start,
                    end_index=end,
                    relative_path=relative,
                    sha256=sha,
                    size_bytes=end - start,
                    state_counts=state_counts,
                )
            )
            start = end
            chunk_index += 1
        return tuple(records)

    def stream_verify(
        self, decisions_dir: Path | str, chunks: Sequence[FgcDecisionChunkRecord]
    ) -> None:
        """Stream-re-read every published chunk and compare digest and bytes."""
        directory = Path(decisions_dir)
        for chunk in chunks:
            path = directory / chunk.relative_path
            if not path.exists():
                raise DecisionConservationError(f"missing decision chunk {chunk.relative_path}")
            hasher = hashlib.sha256()
            with path.open("rb") as handle:
                while True:
                    block = handle.read(1 << 20)
                    if not block:
                        break
                    hasher.update(block)
            if hasher.hexdigest() != chunk.sha256:
                raise DecisionConservationError(
                    f"decision chunk digest mismatch: {chunk.relative_path}"
                )
            if path.stat().st_size != chunk.size_bytes:
                raise DecisionConservationError(
                    f"decision chunk size mismatch: {chunk.relative_path}"
                )
            expected = self._array[chunk.start_index : chunk.end_index]
            actual = np.frombuffer(path.read_bytes(), dtype=np.uint8)
            if not np.array_equal(actual, expected):
                raise DecisionConservationError(
                    f"decision chunk byte mismatch: {chunk.relative_path}"
                )

    def close(self) -> None:
        if self._array is None:
            return
        self._array.flush()
        self._array._mmap.close()
        self._array = None

    def cleanup(self) -> None:
        self.close()
        if self._path.exists():
            try:
                self._path.unlink()
            except OSError:
                pass


# ---------------------------------------------------------------------------
# FGCCOV1 coverage external sort / merge / run-union
# ---------------------------------------------------------------------------


def _sort_key(record: tuple[int, int, int]) -> tuple[int, int, int]:
    return record


class CoverageRecordAccumulator:
    """Accumulate FGCCOV1 records and write deterministic external-sort runs."""

    def __init__(
        self, *, run_dir: Path | str, run_bytes: int, callback: Callable[[int], None] | None = None
    ) -> None:
        self._run_dir = Path(run_dir)
        self._run_dir.mkdir(parents=True, exist_ok=True)
        self._run_bytes = int(run_bytes)
        self._callback = callback
        self._records: list[tuple[int, int, int]] = []
        self._run_paths: list[Path] = []
        self._run_index = 0

    def append(self, ix: int, iy: int, flags: int) -> None:
        self._records.append((int(iy), int(ix), int(flags)))
        if len(self._records) * FGC_COVERAGE_RECORD_SIZE >= self._run_bytes:
            self._flush_run()

    def _flush_run(self) -> None:
        if not self._records:
            return
        self._records.sort(key=_sort_key)
        path = self._run_dir / f"coverage_run_{self._run_index:06d}.bin"
        with path.open("wb") as handle:
            for iy, ix, flags in self._records:
                handle.write(pack_coverage_record(ix, iy, flags))
        self._run_paths.append(path)
        self._run_index += 1
        self._records = []
        if self._callback is not None:
            self._callback(len(self._run_paths))

    def finish(self) -> list[Path]:
        self._flush_run()
        return list(self._run_paths)


def _iter_run_records(path: Path) -> Iterator[tuple[int, int, int]]:
    with path.open("rb") as handle:
        while True:
            record = handle.read(FGC_COVERAGE_RECORD_SIZE)
            if len(record) < FGC_COVERAGE_RECORD_SIZE:
                if len(record) != 0:
                    raise ValueError(f"truncated coverage run record in {path}")
                break
            ix, iy, flags = unpack_coverage_record(record)
            yield (iy, ix, flags)


def merge_coverage_cells(run_paths: Sequence[Path]) -> Iterator[tuple[int, int, int]]:
    """K-way merge runs, OR-deduplicate by (iy, ix), emit sorted unique cells."""
    iterators = [_iter_run_records(path) for path in run_paths]
    if not iterators:
        return
    heap: list[tuple[tuple[int, int, int], int, Iterator[tuple[int, int, int]]]] = []
    for index, iterator in enumerate(iterators):
        try:
            first = next(iterator)
        except StopIteration:
            continue
        heap.append((first, index, iterator))
    heapq.heapify(heap)
    while heap:
        (iy, ix, flags), index, iterator = heap[0]
        heapq.heappop(heap)
        # OR-deduplicate every record sharing (iy, ix) across all runs.
        while True:
            if heap and heap[0][0][0] == iy and heap[0][0][1] == ix:
                other_record, other_index, other_iterator = heap[0]
                heapq.heappop(heap)
                flags |= other_record[2]
                try:
                    replacement = next(other_iterator)
                except StopIteration:
                    continue
                heapq.heappush(heap, (replacement, other_index, other_iterator))
                continue
            try:
                replacement = next(iterator)
            except StopIteration:
                break
            if replacement[0] == iy and replacement[1] == ix:
                flags |= replacement[2]
                continue
            heapq.heappush(heap, (replacement, index, iterator))
            break
        yield (iy, ix, flags)


def run_union_components(
    cells: Iterable[tuple[int, int, int]],
    *,
    max_component_count: int = 10_000_000,
) -> list[FgcCoverageComponent]:
    """Build 4-connected unsupported-component summaries via row run-union.

    ``cells`` yields sorted unique ``(iy, ix, flags)`` coverage cells.  A cell
    is observed when bit 0 is set and confirmed when bit 1 is set.  Unsupported
    cells (observed and not confirmed) form horizontal runs; a run joins only
    prior-row runs with overlapping x intervals.  A component is an edge
    component when any of its cells has an unobserved north, south, east, or
    west neighbour.  Observed previous/current/next rows decide exterior
    adjacency; the row immediately below a run decides its south edge.
    """

    class _Label:
        __slots__ = ("min_ix", "max_ix", "min_iy", "max_iy", "edge", "parent")

        def __init__(self, run_ix0: int, run_ix1: int, iy: int, edge: bool) -> None:
            self.min_ix = run_ix0
            self.max_ix = run_ix1
            self.min_iy = iy
            self.max_iy = iy
            self.edge = edge
            self.parent: _Label | None = None

        def find(self) -> _Label:
            root = self
            while root.parent is not None:
                root = root.parent
            return root

        def union(self, other: _Label) -> None:
            a = self.find()
            b = other.find()
            if a is b:
                return
            b.parent = a
            a.min_ix = min(a.min_ix, b.min_ix)
            a.max_ix = max(a.max_ix, b.max_ix)
            a.min_iy = min(a.min_iy, b.min_iy)
            a.max_iy = max(a.max_iy, b.max_iy)
            a.edge = a.edge or b.edge

    def _finalize(root: _Label, *, south_edge: bool) -> FgcCoverageComponent:
        root.edge = root.edge or south_edge
        return FgcCoverageComponent(
            min_ix=root.min_ix,
            max_ix=root.max_ix,
            min_iy=root.min_iy,
            max_iy=root.max_iy,
            is_edge=root.edge,
        )

    def _rows() -> Iterator[tuple[int, np.ndarray, list[tuple[int, int]]]]:
        """Group unique cells by ascending iy with sorted observed ix arrays."""
        current_iy: int | None = None
        current_observed: list[int] = []
        current_unsupported: list[tuple[int, int]] = []
        for iy, ix, flags in cells:
            if current_iy is not None and iy != current_iy:
                yield current_iy, np.asarray(current_observed, dtype=np.int64), current_unsupported
                current_observed = []
                current_unsupported = []
            current_iy = iy
            if flags & FGC_COVERAGE_FLAG_OBSERVED:
                current_observed.append(ix)
            if (flags & FGC_COVERAGE_FLAG_OBSERVED) and not (flags & FGC_COVERAGE_FLAG_CONFIRMED):
                if current_unsupported and ix == current_unsupported[-1][1] + 1:
                    start, _ = current_unsupported[-1]
                    current_unsupported[-1] = (start, ix)
                else:
                    current_unsupported.append((ix, ix))
        if current_iy is not None:
            yield current_iy, np.asarray(current_observed, dtype=np.int64), current_unsupported

    def _run_edge(
        run_start: int,
        run_end: int,
        prev_observed: np.ndarray | None,
        current_observed: np.ndarray,
    ) -> bool:
        """West/east/north exterior edges for one horizontal run."""
        edge = not _is_observed(current_observed, run_start - 1)
        edge = edge or not _is_observed(current_observed, run_end + 1)
        edge = edge or not _all_observed(prev_observed, run_start, run_end)
        return edge

    prev_iy: int | None = None
    prev_observed: np.ndarray | None = None
    prev_runs: list[tuple[int, int, _Label]] = []
    finalized: list[FgcCoverageComponent] = []
    count = 0

    for iy, observed, unsupported in _rows():
        count += 1
        if count > max_component_count:
            raise ValueError("run-union cell stream exceeds the bounded component cap")

        # A gap of at least one empty row means previous runs can never join a
        # later run; their south neighbours are unobserved so they are edge.
        if prev_iy is not None and iy > prev_iy + 1:
            for prev_run in prev_runs:
                finalized.append(_finalize(prev_run[2].find(), south_edge=True))
            prev_runs = []
            prev_observed = None

        # Decide each previous run's south edge against the current observed row,
        # then carry its exterior flag into its label.
        current_runs: list[tuple[int, int, _Label]] = []
        matched_prev: list[bool] = [False] * len(prev_runs)
        for cur_start, cur_end in unsupported:
            label = _Label(
                cur_start, cur_end, iy, _run_edge(cur_start, cur_end, prev_observed, observed)
            )
            for index, (prev_start, prev_end, prev_label) in enumerate(prev_runs):
                if prev_start <= cur_end and cur_start <= prev_end:
                    matched_prev[index] = True
                    label.union(prev_label)
            current_runs.append((cur_start, cur_end, label))

        for index, (prev_start, prev_end, prev_label) in enumerate(prev_runs):
            south_edge = not _all_observed(observed, prev_start, prev_end)
            if not matched_prev[index]:
                finalized.append(_finalize(prev_label.find(), south_edge=south_edge))
            elif south_edge:
                prev_label.find().edge = True

        prev_runs = current_runs
        prev_observed = observed
        prev_iy = iy

    for prev_run in prev_runs:
        finalized.append(_finalize(prev_run[2].find(), south_edge=True))

    return finalized


def _is_observed(observed: np.ndarray | None, ix: int) -> bool:
    if observed is None or len(observed) == 0:
        return False
    position = int(np.searchsorted(observed, ix, side="left"))
    return position < len(observed) and int(observed[position]) == ix


def _all_observed(observed: np.ndarray | None, start_ix: int, end_ix: int) -> bool:
    """Return True when every integer ix in [start_ix, end_ix] is observed."""
    if observed is None or len(observed) == 0:
        return False
    left = int(np.searchsorted(observed, start_ix, side="left"))
    right = int(np.searchsorted(observed, end_ix, side="right"))
    return right - left == end_ix - start_ix + 1


def compute_coverage_result(
    run_paths: Sequence[Path],
    *,
    policy: GroundClassificationPolicyConfig,
    max_cells: int | None = None,
) -> FgcCoverageResult:
    """Merge coverage runs, OR-dedup, and compute metrics in one streaming pass."""
    observed_count = 0
    confirmed_count = 0
    seen = 0

    def _counted_cells() -> Iterator[tuple[int, int, int]]:
        nonlocal observed_count, confirmed_count, seen
        for iy, ix, flags in merge_coverage_cells(run_paths):
            seen += 1
            if max_cells is not None and seen > max_cells:
                raise ValueError("coverage cell stream exceeds max_coverage_cells")
            if flags & FGC_COVERAGE_FLAG_OBSERVED:
                observed_count += 1
            if flags & FGC_COVERAGE_FLAG_CONFIRMED:
                confirmed_count += 1
            yield iy, ix, flags

    components = run_union_components(_counted_cells())
    if confirmed_count > observed_count:
        raise ValueError("confirmed occupied cells cannot exceed observed cells")
    occupancy = 0.0 if observed_count == 0 else confirmed_count / observed_count
    return FgcCoverageResult(
        observed_cell_count=observed_count,
        confirmed_occupied_cell_count=confirmed_count,
        occupancy_fraction=occupancy,
        unsupported_components=tuple(components),
    )
