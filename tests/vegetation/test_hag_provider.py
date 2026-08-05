"""FRV-A2 P-HAG runtime provider tests (PLAN-FROZEN)."""

from __future__ import annotations

import dataclasses
import hashlib
from pathlib import Path
from unittest import mock

import core.contracts as contracts
import core.contracts.frv as frv
import core.contracts.frv_hag as hag
import laspy
import numpy as np
import pytest
from algorithms.io.source_provider import load_frv_source
from algorithms.vegetation import load_frv_hag
from core.contracts.frv_a2 import (
    FrvA2ArtifactComponentBinding,
    FrvA2ArtifactComponentKind,
    FrvA2ProviderRole,
)
from core.contracts.frv_hag import (
    FRV_HAG_PROVIDER_KEY,
    FrvHagLoadError,
    FrvHagProviderConfig,
    FrvHagProviderResult,
    frv_hag_point_content_digest,
    to_frv_a2_hag_provider_binding_evidence,
    validate_finalized_frv_hag_provider_result,
)
from core.contracts.frv_source import (
    FRV_SOURCE_PROVIDER_ALGORITHM_VERSION,
    FRV_SOURCE_PROVIDER_INTERFACE_VERSION,
    FRV_SOURCE_PROVIDER_SCHEMA_VERSION,
    FrvSourcePassThroughDimension,
    FrvSourceProviderReference,
    FrvSourceProviderResult,
)

from tests.contracts.test_frv_hag_contracts import _config as _hag_config
from tests.io.test_source_provider import (
    _configuration,
    _provider_config,
    _source_identity,
)


def write_hag_las(path: Path, *, point_count: int = 8) -> None:
    header = laspy.LasHeader(point_format=7, version="1.4")
    header.scales = np.array([0.01, 0.01, 0.01])
    header.offsets = np.array([100.0, 200.0, 0.0])
    las = laspy.LasData(header)
    idx = np.arange(point_count)
    las.x = 100.0 + idx.astype(np.float64)
    las.y = 200.0 + idx.astype(np.float64)
    las.z = np.where(idx % 2 == 0, 0.0, 10.0).astype(np.float64)
    las.classification = np.where(idx % 2 == 0, 2, 5).astype(np.uint8)
    las.write(path)


def _source_reference(
    path: Path, *, count: int = 8
) -> tuple[FrvSourceProviderReference, object, object]:
    write_hag_las(path, point_count=count)
    file_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    file_size = path.stat().st_size
    config = _provider_config(
        pass_through_dimensions=(FrvSourcePassThroughDimension.CLASSIFICATION,)
    )
    configuration = _configuration()
    bounds = ([100.0, 200.0, 0.0], [100.0 + count - 1, 200.0 + count - 1, 10.0])
    identity = _source_identity(str(path), file_sha, count, bounds)
    component = FrvA2ArtifactComponentBinding(
        component_key="source_file",
        component_kind=FrvA2ArtifactComponentKind.SOURCE_FILE,
        sha256=file_sha,
        size_bytes=file_size,
    )
    artifact_id = (
        f"frv-a2-ref-v1|source_scan|source_scan|"
        f"source_file,source_file,{file_sha},{file_size}"
    )
    reference = FrvSourceProviderReference(
        provider_role=FrvA2ProviderRole.SOURCE_SCAN,
        provider_key="source_scan",
        artifact_reference_id=artifact_id,
        artifact_components=(component,),
        runtime_access_path=path,
        source_identity=identity,
        canonical_point_content_spec=None,
        configuration_digest=configuration.configuration_digest,
        provider_config_digest=config.provider_config_digest,
        interface_version=FRV_SOURCE_PROVIDER_INTERFACE_VERSION,
        schema_version=FRV_SOURCE_PROVIDER_SCHEMA_VERSION,
        algorithm_version=FRV_SOURCE_PROVIDER_ALGORITHM_VERSION,
    )
    return reference, config, configuration


def _source_result(path: Path, *, count: int = 8, chunk_size: int = 3):
    reference, config, configuration = _source_reference(path, count=count)
    session = load_frv_source(
        reference, frv_configuration=configuration, provider_config=config, chunk_size=chunk_size
    )
    with session:
        list(session.iter_chunks())
        result = session.finalize()
    return reference, configuration, result


def _run_hag(path: Path, *, count: int = 8, source_chunk_size: int = 3, **hag_overrides):
    reference, configuration, source_result = _source_result(
        path, count=count, chunk_size=source_chunk_size
    )
    provider_config = _hag_config(**hag_overrides)
    session = load_frv_hag(
        reference,
        source_result,
        frv_configuration=configuration,
        provider_config=provider_config,
        source_chunk_size=source_chunk_size,
    )
    return session, source_result


def test_exactly_two_source_sessions(tmp_path: Path) -> None:
    reference, configuration, source_result = _source_result(tmp_path / "s.las", count=8)
    provider_config = _hag_config()
    with mock.patch(
        "algorithms.vegetation.hag_provider.load_frv_source",
        wraps=load_frv_source,
    ) as mocked:
        session = load_frv_hag(
            reference,
            source_result,
            frv_configuration=configuration,
            provider_config=provider_config,
            source_chunk_size=3,
        )
        with session:
            list(session.iter_chunks())
            session.finalize()
        assert mocked.call_count == 2  # Pass 1 and Pass 2


def test_normal_flow_chunks_and_result(tmp_path: Path) -> None:
    session, source_result = _run_hag(tmp_path / "s.las", count=8)
    with session:
        chunks = list(session.iter_chunks())
        result = session.finalize()
    assert [c.point_count for c in chunks] == [3, 3, 2]
    assert [c.is_last for c in chunks] == [False, False, True]
    all_index = np.concatenate([c.original_index for c in chunks])
    assert all_index.tolist() == list(range(8))
    validate_finalized_frv_hag_provider_result(result)
    evidence = to_frv_a2_hag_provider_binding_evidence(result)
    contracts.validate_frv_a2_provider_binding_evidence(evidence)
    assert evidence.provider_key == FRV_HAG_PROVIDER_KEY
    assert evidence.provider_role is FrvA2ProviderRole.HAG
    assert evidence.provider_readiness.value == "validated"
    assert result.proof.source_point_count == 8
    assert result.proof.hag_out_of_domain_count == 0


def test_chunk_one_to_one_and_hag_values(tmp_path: Path) -> None:
    session, _source_result = _run_hag(tmp_path / "s.las", count=8)
    with session:
        chunks = list(session.iter_chunks())
        result = session.finalize()
    # ground points (even) at z=0; canopy points (odd) at z=10 -> positive HAG
    assert result.proof.hag_valid_positive_count == 4
    assert result.proof.hag_valid_near_zero_count == 4
    assert result.proof.hag_missing_count == 0
    assert result.proof.hag_invalid_count == 0
    assert result.proof.supported_point_count == 8
    assert result.proof.unsupported_point_count == 0
    # conservation formulas
    assert result.proof.supported_point_count + result.proof.unsupported_point_count == 8
    for chunk in chunks:
        assert np.isfinite(chunk.raw_hag_m).all()
        assert np.isfinite(chunk.clamped_hag_m).all()
        assert (chunk.clamped_hag_m >= 0.0).all()


def test_point_content_digest_matches_oracle(tmp_path: Path) -> None:
    session, source_result = _run_hag(tmp_path / "s.las", count=8)
    with session:
        chunks = list(session.iter_chunks())
        result = session.finalize()
    oracle = frv_hag_point_content_digest(
        source_content_identity_digest=result.proof.source_content_identity_digest,
        source_identity_record_digest=result.proof.source_identity_record_digest,
        provider_config_digest=result.proof.provider_config_digest,
        source_point_count=8,
        chunks=chunks,
    )
    assert oracle == result.proof.hag_point_content_digest


def test_chunk_size_invariance(tmp_path: Path) -> None:
    path = tmp_path / "s.las"
    results = []
    for chunk_size in (2, 5):
        session, _ = _run_hag(path, count=8, source_chunk_size=chunk_size)
        with session:
            list(session.iter_chunks())
            results.append(session.finalize())
    assert results[0].proof == results[1].proof


def test_lifecycle_premature_and_repeated(tmp_path: Path) -> None:
    session, _ = _run_hag(tmp_path / "s.las", count=8)
    with pytest.raises(FrvHagLoadError):
        session.finalize()  # premature (never iterated)
    session.close()
    # partial iteration then finalize is premature
    session2, _ = _run_hag(tmp_path / "s.las", count=8)
    iterator = session2.iter_chunks()
    next(iterator)  # partial consumption
    with pytest.raises(FrvHagLoadError):
        session2.finalize()
    session2.close()
    iterator.close()
    session3, _ = _run_hag(tmp_path / "s.las", count=8)
    with session3:
        list(session3.iter_chunks())
        session3.finalize()
        with pytest.raises(FrvHagLoadError):
            session3.finalize()  # repeated finalize


def test_no_ground_failure(tmp_path: Path) -> None:
    reference, configuration, source_result = _source_result(tmp_path / "s.las", count=8)
    provider_config = _hag_config()
    session = load_frv_hag(
        reference,
        source_result,
        frv_configuration=configuration,
        provider_config=provider_config,
        source_chunk_size=3,
    )
    # patch Pass 1 grid to be empty -> NO_GROUND
    with pytest.raises(FrvHagLoadError) as excinfo:
        def reject_ground(*args, **kwargs):
            raise _build_hag_no_ground()

        session._accumulate_grid = reject_ground
        list(session.iter_chunks())
    assert "FRV_HAG_MISSING" in str(excinfo.value)
    session.close()


def _build_hag_no_ground():
    from algorithms.vegetation.hag_provider import _build_hag_error

    return _build_hag_error("NO_GROUND")


def test_missing_classification_failure(tmp_path: Path) -> None:
    # build a source without classification in pass-through -> gate 8 fails
    path = tmp_path / "s.las"
    write_hag_las(path, point_count=4)
    file_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    file_size = path.stat().st_size
    config = _provider_config()  # no pass-through dims -> no classification
    configuration = _configuration()
    identity = _source_identity(
        str(path), file_sha, 4, ([100.0, 200.0, 0.0], [103.0, 203.0, 10.0])
    )
    component = FrvA2ArtifactComponentBinding(
        component_key="source_file",
        component_kind=FrvA2ArtifactComponentKind.SOURCE_FILE,
        sha256=file_sha,
        size_bytes=file_size,
    )
    artifact_id = (
        f"frv-a2-ref-v1|source_scan|source_scan|"
        f"source_file,source_file,{file_sha},{file_size}"
    )
    reference = FrvSourceProviderReference(
        provider_role=FrvA2ProviderRole.SOURCE_SCAN,
        provider_key="source_scan",
        artifact_reference_id=artifact_id,
        artifact_components=(component,),
        runtime_access_path=path,
        source_identity=identity,
        canonical_point_content_spec=None,
        configuration_digest=configuration.configuration_digest,
        provider_config_digest=config.provider_config_digest,
        interface_version=FRV_SOURCE_PROVIDER_INTERFACE_VERSION,
        schema_version=FRV_SOURCE_PROVIDER_SCHEMA_VERSION,
        algorithm_version=FRV_SOURCE_PROVIDER_ALGORITHM_VERSION,
    )
    session = load_frv_source(
        reference, frv_configuration=configuration, provider_config=config, chunk_size=2
    )
    with session:
        list(session.iter_chunks())
        source_result = session.finalize()
    with pytest.raises(FrvHagLoadError) as excinfo:
        load_frv_hag(
            reference,
            source_result,
            frv_configuration=configuration,
            provider_config=_hag_config(),
            source_chunk_size=2,
        )
    assert "classification" in str(excinfo.value)


def test_active_exception_not_overridden_by_cleanup(tmp_path: Path) -> None:
    session, _ = _run_hag(tmp_path / "s.las", count=8)
    with pytest.raises(RuntimeError, match="boom"):
        with session:
            raise RuntimeError("boom")


# ---------------------------------------------------------------------------
# MAJ-007 strengthened tests: helpers
# ---------------------------------------------------------------------------


def _write_points_las(path: Path, points) -> None:
    """Write a LAS whose xyz/classification round-trip exactly (0.01 scale)."""
    header = laspy.LasHeader(point_format=7, version="1.4")
    header.scales = np.array([0.01, 0.01, 0.01])
    header.offsets = np.array([100.0, 200.0, 0.0])
    las = laspy.LasData(header)
    arr = np.asarray(points, dtype=np.float64)
    las.x = arr[:, 0]
    las.y = arr[:, 1]
    las.z = arr[:, 2]
    las.classification = arr[:, 3].astype(np.uint8)
    las.write(path)


def _build_reference(
    path: Path, config, configuration
) -> FrvSourceProviderReference:
    """Build a reference whose identity reflects the actual file extents."""
    file_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    file_size = path.stat().st_size
    with laspy.open(path) as reader:
        header = reader.header
    count = int(header.point_count)
    mins = tuple(float(v) for v in header.mins)
    maxs = tuple(float(v) for v in header.maxs)
    bounds = ([mins[0], mins[1], mins[2]], [maxs[0], maxs[1], maxs[2]])
    identity = _source_identity(str(path), file_sha, count, bounds)
    component = FrvA2ArtifactComponentBinding(
        component_key="source_file",
        component_kind=FrvA2ArtifactComponentKind.SOURCE_FILE,
        sha256=file_sha,
        size_bytes=file_size,
    )
    artifact_id = (
        f"frv-a2-ref-v1|source_scan|source_scan|"
        f"source_file,source_file,{file_sha},{file_size}"
    )
    return FrvSourceProviderReference(
        provider_role=FrvA2ProviderRole.SOURCE_SCAN,
        provider_key="source_scan",
        artifact_reference_id=artifact_id,
        artifact_components=(component,),
        runtime_access_path=path,
        source_identity=identity,
        canonical_point_content_spec=None,
        configuration_digest=configuration.configuration_digest,
        provider_config_digest=config.provider_config_digest,
        interface_version=FRV_SOURCE_PROVIDER_INTERFACE_VERSION,
        schema_version=FRV_SOURCE_PROVIDER_SCHEMA_VERSION,
        algorithm_version=FRV_SOURCE_PROVIDER_ALGORITHM_VERSION,
    )


def _run_source_custom(path: Path, config, configuration, *, chunk_size: int = 2):
    reference = _build_reference(path, config, configuration)
    session = load_frv_source(
        reference,
        frv_configuration=configuration,
        provider_config=config,
        chunk_size=chunk_size,
    )
    with session:
        list(session.iter_chunks())
        result = session.finalize()
    return reference, configuration, result


def _install_grid(
    session, grid, *, ground_point_count: int = 0, pass1_source_id: str | None = None
) -> None:
    """Replace Pass 1 with a hand-built sparse ground grid."""

    def fake_pass1():
        session._grid = grid
        session._ground_point_count = ground_point_count
        if pass1_source_id is not None:
            session._pass1_source_id = pass1_source_id
        else:
            session._pass1_source_id = session._expected_source_id

    session._run_pass1 = fake_pass1  # type: ignore[method-assign]


def _tamper_original_index(session, mutator) -> None:
    """Re-wrap Pass 2's chunk builder to mutate original_index per chunk."""
    original = session._build_hag_chunk

    def builder(source_chunk):
        chunk = original(source_chunk)
        idx = chunk.original_index.copy()
        mutator(idx)
        idx.setflags(write=False)
        return dataclasses.replace(chunk, original_index=idx)

    session._build_hag_chunk = builder  # type: ignore[method-assign]


class _FakeSourceSession:
    """Minimal source-session double for Pass 1 failure injection."""

    def __init__(
        self,
        *,
        iter_error: Exception | None = None,
        finalize_error: Exception | None = None,
        close_error: Exception | None = None,
        result: FrvSourceProviderResult | None = None,
    ) -> None:
        self._iter_error = iter_error
        self._finalize_error = finalize_error
        self._close_error = close_error
        self._result = result
        self.closed = False

    def iter_chunks(self):
        if self._iter_error is not None:
            raise self._iter_error
        return iter(())

    def finalize(self):
        if self._finalize_error is not None:
            raise self._finalize_error
        if self._result is None:
            raise RuntimeError("no result configured")
        return self._result

    def close(self) -> None:
        self.closed = True
        if self._close_error is not None:
            raise self._close_error

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        try:
            self.close()
        except Exception:
            if exc_type is None:
                raise
        return False


def _rows_by_index(chunks):
    """Map original_index -> (chunk, row) across the emitted HAG chunks."""
    return {
        int(chunk.original_index[r]): (chunk, r)
        for chunk in chunks
        for r in range(chunk.point_count)
    }


def _idw_oracle(px, py, pz, grid, min_x, min_y, config):
    """Independent sparse-grid IDW evaluation (mirrors the frozen algorithm)."""
    cell_size = config.cell_size_m
    radius = config.support_radius_m
    radius_sq = radius * radius
    min_support = config.min_support_cells
    max_support = config.max_support_cells
    tolerance = config.negative_hag_tolerance_m
    candidates = []
    for (ix, iy), (cell_min_z, _count) in grid.items():
        cx = min_x + (ix + 0.5) * cell_size
        cy = min_y + (iy + 0.5) * cell_size
        dx = px - cx
        dy = py - cy
        squared = dx * dx + dy * dy
        if squared <= radius_sq:
            candidates.append((squared, ix, iy, cell_min_z))
    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    if len(candidates) < min_support:
        return dict(
            supported=False,
            count=0,
            raw=0.0,
            clamped=0.0,
            ground_z=0.0,
            state=int(frv.FrvHagEvidenceState.HAG_MISSING),
        )
    candidates = candidates[:max_support]
    if candidates[0][0] == 0.0:
        ground_z = candidates[0][3]
    else:
        numerator = 0.0
        denominator = 0.0
        for squared, _ix, _iy, cell_min_z in candidates:
            weight = 1.0 / squared
            numerator += weight * cell_min_z
            denominator += weight
        ground_z = numerator / denominator
    raw_hag = pz - ground_z
    clamped = max(raw_hag, 0.0)
    if raw_hag > 0.0:
        state = int(frv.FrvHagEvidenceState.HAG_VALID_POSITIVE)
    elif raw_hag == 0.0:
        state = int(frv.FrvHagEvidenceState.HAG_VALID_NEAR_ZERO)
    elif raw_hag >= -tolerance:
        state = int(frv.FrvHagEvidenceState.HAG_WITHIN_NEGATIVE_TOLERANCE)
    else:
        state = int(frv.FrvHagEvidenceState.HAG_INVALID)
    return dict(
        supported=True,
        count=len(candidates),
        raw=raw_hag,
        clamped=clamped,
        ground_z=ground_z,
        state=state,
    )


# ---------------------------------------------------------------------------
# MAJ-007: pre-pass exact order and stepwise short-circuit
# ---------------------------------------------------------------------------


def test_pre_pass_gate_errors(tmp_path: Path) -> None:
    reference, configuration, source_result = _source_result(tmp_path / "s.las", count=4)
    kwargs = dict(
        frv_configuration=configuration,
        provider_config=_hag_config(),
        source_chunk_size=2,
    )

    # Gate 1/2: typed but forged source result (missing private capability)
    forged = object.__new__(FrvSourceProviderResult)
    object.__setattr__(forged, "proof", source_result.proof)
    object.__setattr__(forged, "source_identity", source_result.source_identity)
    object.__setattr__(forged, "_finalized_capability", object())
    with pytest.raises(FrvHagLoadError) as ei:
        load_frv_hag(reference, forged, **kwargs)
    assert "source_result" in str(ei.value)
    assert "FRV_PROVIDER_INVALID" in str(ei.value)

    # Gate 3 boundary: a wrong-type reference is a programming error at the
    # loader boundary (TypeError), not an invalid-P-HAG FrvHagLoadError.
    with pytest.raises(TypeError):
        load_frv_hag(object(), source_result, **kwargs)  # type: ignore[arg-type]

    # Gate 4: invalid frv_configuration
    bad_configuration = object.__new__(frv.FrvConfiguration)
    with pytest.raises(FrvHagLoadError) as ei:
        load_frv_hag(
            reference,
            source_result,
            **{**kwargs, "frv_configuration": bad_configuration},
        )
    assert "provider_config" in str(ei.value)

    # Gate 5: forged P-HAG config with a stale digest (typed, never constructed)
    forged_config = object.__new__(FrvHagProviderConfig)
    valid_config = _hag_config()
    for field_ in dataclasses.fields(FrvHagProviderConfig):
        object.__setattr__(forged_config, field_.name, getattr(valid_config, field_.name))
    object.__setattr__(forged_config, "provider_config_digest", "b" * 64)
    with pytest.raises(FrvHagLoadError) as ei:
        load_frv_hag(
            reference,
            source_result,
            **{**kwargs, "provider_config": forged_config},
        )
    assert "provider_config" in str(ei.value)

    # Gate 7: reference that fails cross-binding (wrong configuration_digest)
    tampered = dataclasses.replace(reference, configuration_digest="b" * 64)
    with pytest.raises(FrvHagLoadError) as ei:
        load_frv_hag(tampered, source_result, **kwargs)
    assert "reference" in str(ei.value)
    assert "FRV_PROVIDER_INVALID" in str(ei.value)


def test_pre_pass_source_config_reconstruction_failure(tmp_path: Path) -> None:
    reference, configuration, source_result = _source_result(tmp_path / "s.las", count=4)
    with mock.patch(
        "algorithms.vegetation.hag_provider.derive_frv_source_provider_config_digest",
        return_value="b" * 64,
    ):
        with pytest.raises(FrvHagLoadError) as ei:
            load_frv_hag(
                reference,
                source_result,
                frv_configuration=configuration,
                provider_config=_hag_config(),
                source_chunk_size=2,
            )
    assert "provider_config" in str(ei.value)
    assert "FRV_PROVIDER_INVALID" in str(ei.value)


def test_pre_pass_short_circuit_before_classification(tmp_path: Path) -> None:
    # Gate 7 (reference binding) must fire before gate 11 (classification),
    # even though gate 11 would also fail for this no-classification source.
    config = _provider_config()  # no pass-through dimensions
    configuration = _configuration()
    path = tmp_path / "s.las"
    write_hag_las(path, point_count=4)
    reference, _, result = _run_source_custom(path, config, configuration, chunk_size=2)
    tampered = dataclasses.replace(reference, configuration_digest="b" * 64)
    with pytest.raises(FrvHagLoadError) as ei:
        load_frv_hag(
            tampered,
            result,
            frv_configuration=configuration,
            provider_config=_hag_config(),
            source_chunk_size=2,
        )
    assert "reference" in str(ei.value)
    assert "classification" not in str(ei.value)


# ---------------------------------------------------------------------------
# MAJ-007: original_index lost / duplicate / out-of-range
# ---------------------------------------------------------------------------


def test_original_index_lost_duplicate_out_of_range(tmp_path: Path) -> None:
    # INDEX_LOST: first chunk jumps ahead of the expected 0..N-1 stream.
    # The error surfaces while consuming the Pass-2 generator.
    session, _ = _run_hag(tmp_path / "s.las", count=4)
    _tamper_original_index(session, lambda idx: idx.__setitem__(0, int(idx[0]) + 5))
    with pytest.raises(FrvHagLoadError) as ei:
        list(session.iter_chunks())
    assert "FRV_ORIGINAL_INDEX_INVALID" in str(ei.value)
    assert "FRV_STREAM_INCOMPLETE" not in str(ei.value)
    session.close()

    # INDEX_DUPLICATE: second row repeats the first index
    session2, _ = _run_hag(tmp_path / "s.las", count=4)
    _tamper_original_index(session2, lambda idx: idx.__setitem__(1, int(idx[0])))
    with pytest.raises(FrvHagLoadError) as ei:
        list(session2.iter_chunks())
    assert "FRV_ORIGINAL_INDEX_DUPLICATE" in str(ei.value)
    session2.close()

    # INDEX_OUT_OF_RANGE: negative index
    session3, _ = _run_hag(tmp_path / "s.las", count=4)
    _tamper_original_index(session3, lambda idx: idx.__setitem__(0, -1))
    with pytest.raises(FrvHagLoadError) as ei:
        list(session3.iter_chunks())
    assert "FRV_ORIGINAL_INDEX_INVALID" in str(ei.value)
    session3.close()


# ---------------------------------------------------------------------------
# MAJ-007: source_id cross-chunk and cross-pass binding
# ---------------------------------------------------------------------------


def test_source_id_cross_binding_mismatches(tmp_path: Path) -> None:
    # Pass 1: real chunk source_id does not match the source identity
    session, _ = _run_hag(tmp_path / "s.las", count=4)
    session._expected_source_id = "other-scene"
    with pytest.raises(FrvHagLoadError) as ei:
        list(session.iter_chunks())
    assert "FRV_SOURCE_IDENTITY_FAILED" in str(ei.value)
    session.close()

    # Pass 2: Pass 1 recorded a different source_id than Pass 2 chunks
    session2, _ = _run_hag(tmp_path / "s.las", count=4)
    _install_grid(session2, {(0, 0): (10.0, 1)}, pass1_source_id="other-scene")
    with pytest.raises(FrvHagLoadError) as ei:
        list(session2.iter_chunks())
    assert "FRV_SOURCE_IDENTITY_FAILED" in str(ei.value)
    session2.close()


# ---------------------------------------------------------------------------
# MAJ-007: IDW independent numerical oracle + HAG state classification
# ---------------------------------------------------------------------------


def test_hag_rows_match_independent_idw_oracle(tmp_path: Path) -> None:
    points = [
        (100.0, 200.0, 10.0, 2),  # 0: anchor, IDW -> raw 0 (near-zero)
        (100.5, 200.5, 15.0, 5),  # 1: exact center -> positive
        (100.5, 200.5, 10.0, 5),  # 2: exact center -> near-zero
        (101.0, 200.5, 16.0, 5),  # 3: IDW -> positive
        (101.0, 200.5, 9.9, 5),  # 4: within negative tolerance
        (101.0, 200.5, 9.5, 5),  # 5: invalid negative
        (105.0, 205.0, 5.0, 5),  # 6: far outside radius -> missing
        (100.5, 201.0, 12.0, 5),  # 7: near center -> positive
    ]
    path = tmp_path / "s.las"
    _write_points_las(path, points)
    source_config = _provider_config(
        pass_through_dimensions=(FrvSourcePassThroughDimension.CLASSIFICATION,)
    )
    configuration = _configuration()
    reference, _, source_result = _run_source_custom(path, source_config, configuration)
    grid = {(0, 0): (10.0, 1)}
    config = _hag_config(
        cell_size_m=1.0,
        support_radius_m=2.0,
        min_support_cells=1,
        max_support_cells=8,
        negative_hag_tolerance_m=0.25,
    )
    session = load_frv_hag(
        reference,
        source_result,
        frv_configuration=configuration,
        provider_config=config,
        source_chunk_size=8,
    )
    _install_grid(session, grid, ground_point_count=1)
    with session:
        chunks = list(session.iter_chunks())
        result = session.finalize()

    min_x = float(source_result.proof.source_bounds_min_xyz[0])
    min_y = float(source_result.proof.source_bounds_min_xyz[1])
    assert (min_x, min_y) == (100.0, 200.0)
    rows: dict[int, tuple] = {}
    for chunk in chunks:
        for r in range(chunk.point_count):
            rows[int(chunk.original_index[r])] = (chunk, r)

    state_counts: dict[int, int] = {}
    for i, (px, py, pz, _cls) in enumerate(points):
        chunk, r = rows[i]
        expected = _idw_oracle(px, py, pz, grid, min_x, min_y, config)
        assert bool(chunk.ground_support[r]) == expected["supported"]
        assert int(chunk.support_cell_count[r]) == expected["count"]
        assert float(chunk.ground_z_m[r]) == pytest.approx(expected["ground_z"], abs=1e-9)
        assert float(chunk.raw_hag_m[r]) == pytest.approx(expected["raw"], abs=1e-9)
        assert float(chunk.clamped_hag_m[r]) == pytest.approx(expected["clamped"], abs=1e-9)
        assert int(chunk.hag_evidence_state[r]) == expected["state"]
        state_counts[expected["state"]] = state_counts.get(expected["state"], 0) + 1

    # exact-center rows read the cell min-z directly (no IDW)
    for i in (1, 2):
        chunk, r = rows[i]
        assert float(chunk.ground_z_m[r]) == pytest.approx(10.0)

    # explicit state classification for the controlled rows
    chunk, r = rows[1]
    assert int(chunk.hag_evidence_state[r]) == int(frv.FrvHagEvidenceState.HAG_VALID_POSITIVE)
    chunk, r = rows[2]
    assert int(chunk.hag_evidence_state[r]) == int(frv.FrvHagEvidenceState.HAG_VALID_NEAR_ZERO)
    chunk, r = rows[4]
    assert int(chunk.hag_evidence_state[r]) == int(
        frv.FrvHagEvidenceState.HAG_WITHIN_NEGATIVE_TOLERANCE
    )
    chunk, r = rows[5]
    assert int(chunk.hag_evidence_state[r]) == int(frv.FrvHagEvidenceState.HAG_INVALID)
    chunk, r = rows[6]
    assert int(chunk.hag_evidence_state[r]) == int(frv.FrvHagEvidenceState.HAG_MISSING)
    assert not bool(chunk.ground_support[r])

    # proof counts must agree with the independent oracle tallies
    assert result.proof.hag_valid_positive_count == state_counts.get(
        int(frv.FrvHagEvidenceState.HAG_VALID_POSITIVE), 0
    )
    assert result.proof.hag_valid_near_zero_count == state_counts.get(
        int(frv.FrvHagEvidenceState.HAG_VALID_NEAR_ZERO), 0
    )
    assert result.proof.hag_within_negative_tolerance_count == state_counts.get(
        int(frv.FrvHagEvidenceState.HAG_WITHIN_NEGATIVE_TOLERANCE), 0
    )
    assert result.proof.hag_invalid_count == state_counts.get(
        int(frv.FrvHagEvidenceState.HAG_INVALID), 0
    )
    assert result.proof.hag_missing_count == state_counts.get(
        int(frv.FrvHagEvidenceState.HAG_MISSING), 0
    )
    supported = len(points) - state_counts.get(int(frv.FrvHagEvidenceState.HAG_MISSING), 0)
    assert result.proof.supported_point_count == supported
    assert result.proof.unsupported_point_count == len(points) - supported
    assert result.proof.negative_tolerance_exceeded_point_count == result.proof.hag_invalid_count
    validate_finalized_frv_hag_provider_result(result)


# ---------------------------------------------------------------------------
# MAJ-007: radius boundary, exclusion, tie ordering, min/max support cells
# ---------------------------------------------------------------------------


def test_radius_boundary_inclusion_and_exclusion(tmp_path: Path) -> None:
    points = [
        (100.0, 200.0, 10.0, 2),  # anchor
        (101.5, 200.5, 15.0, 5),  # exactly on the radius -> supported
        (101.6, 200.5, 5.0, 5),  # just beyond the radius -> missing
    ]
    path = tmp_path / "s.las"
    _write_points_las(path, points)
    source_config = _provider_config(
        pass_through_dimensions=(FrvSourcePassThroughDimension.CLASSIFICATION,)
    )
    configuration = _configuration()
    reference, _, source_result = _run_source_custom(path, source_config, configuration)
    grid = {(0, 0): (10.0, 1)}
    config = _hag_config(
        cell_size_m=1.0,
        support_radius_m=1.0,
        min_support_cells=1,
        max_support_cells=8,
        negative_hag_tolerance_m=0.25,
    )
    session = load_frv_hag(
        reference,
        source_result,
        frv_configuration=configuration,
        provider_config=config,
        source_chunk_size=8,
    )
    _install_grid(session, grid, ground_point_count=1)
    with session:
        chunks = list(session.iter_chunks())
        result = session.finalize()
    rows = _rows_by_index(chunks)

    chunk, r = rows[1]
    assert bool(chunk.ground_support[r]) is True
    assert int(chunk.support_cell_count[r]) == 1
    assert float(chunk.ground_z_m[r]) == pytest.approx(10.0)
    assert float(chunk.raw_hag_m[r]) == pytest.approx(5.0)
    assert int(chunk.hag_evidence_state[r]) == int(frv.FrvHagEvidenceState.HAG_VALID_POSITIVE)

    chunk, r = rows[2]
    assert bool(chunk.ground_support[r]) is False
    assert int(chunk.support_cell_count[r]) == 0
    assert int(chunk.hag_evidence_state[r]) == int(frv.FrvHagEvidenceState.HAG_MISSING)
    validate_finalized_frv_hag_provider_result(result)


def test_tie_ordering_and_max_support_cells(tmp_path: Path) -> None:
    points = [
        (100.0, 200.0, 10.0, 2),  # anchor
        (101.0, 200.5, 16.0, 5),  # ix tie between (0,0) and (1,0)
        (100.5, 201.0, 25.0, 5),  # iy tie between (0,0) and (0,1)
    ]
    path = tmp_path / "s.las"
    _write_points_las(path, points)
    source_config = _provider_config(
        pass_through_dimensions=(FrvSourcePassThroughDimension.CLASSIFICATION,)
    )
    configuration = _configuration()
    reference, _, source_result = _run_source_custom(path, source_config, configuration)
    grid = {(0, 0): (10.0, 1), (1, 0): (20.0, 1), (0, 1): (30.0, 1)}
    config = _hag_config(
        cell_size_m=1.0,
        support_radius_m=2.0,
        min_support_cells=1,
        max_support_cells=2,
        negative_hag_tolerance_m=0.25,
    )
    session = load_frv_hag(
        reference,
        source_result,
        frv_configuration=configuration,
        provider_config=config,
        source_chunk_size=8,
    )
    _install_grid(session, grid, ground_point_count=2)
    with session:
        chunks = list(session.iter_chunks())
        result = session.finalize()
    rows = _rows_by_index(chunks)

    # (101.0, 200.5): distances 0.25/0.25/1.25 -> truncated to top-2 (ix tie)
    chunk, r = rows[1]
    assert int(chunk.support_cell_count[r]) == 2
    assert float(chunk.ground_z_m[r]) == pytest.approx(15.0)
    assert float(chunk.raw_hag_m[r]) == pytest.approx(1.0)

    # (100.5, 201.0): distances 0.25/0.25/1.25 -> truncated to top-2 (iy tie)
    chunk, r = rows[2]
    assert int(chunk.support_cell_count[r]) == 2
    assert float(chunk.ground_z_m[r]) == pytest.approx(20.0)
    assert float(chunk.raw_hag_m[r]) == pytest.approx(5.0)
    assert int(chunk.hag_evidence_state[r]) == int(frv.FrvHagEvidenceState.HAG_VALID_POSITIVE)
    validate_finalized_frv_hag_provider_result(result)


def test_min_support_cells_missing(tmp_path: Path) -> None:
    points = [
        (100.0, 200.0, 10.0, 2),  # anchor
        (100.5, 200.5, 15.0, 5),  # only 1 support cell < min_support_cells=2
    ]
    path = tmp_path / "s.las"
    _write_points_las(path, points)
    source_config = _provider_config(
        pass_through_dimensions=(FrvSourcePassThroughDimension.CLASSIFICATION,)
    )
    configuration = _configuration()
    reference, _, source_result = _run_source_custom(path, source_config, configuration)
    grid = {(0, 0): (10.0, 1)}
    config = _hag_config(
        cell_size_m=1.0,
        support_radius_m=2.0,
        min_support_cells=2,
        max_support_cells=8,
        negative_hag_tolerance_m=0.25,
    )
    session = load_frv_hag(
        reference,
        source_result,
        frv_configuration=configuration,
        provider_config=config,
        source_chunk_size=8,
    )
    _install_grid(session, grid, ground_point_count=1)
    with session:
        chunks = list(session.iter_chunks())
        result = session.finalize()
    rows = _rows_by_index(chunks)
    chunk, r = rows[1]
    assert bool(chunk.ground_support[r]) is False
    assert int(chunk.support_cell_count[r]) == 0
    assert int(chunk.hag_evidence_state[r]) == int(frv.FrvHagEvidenceState.HAG_MISSING)
    validate_finalized_frv_hag_provider_result(result)


# ---------------------------------------------------------------------------
# MAJ-007: Pass 1 load / read / finalize / binding / close failures -> ABORTED
# ---------------------------------------------------------------------------


def test_pass1_load_failure_aborts(tmp_path: Path) -> None:
    reference, configuration, source_result = _source_result(tmp_path / "s.las", count=4)
    session = load_frv_hag(
        reference,
        source_result,
        frv_configuration=configuration,
        provider_config=_hag_config(),
        source_chunk_size=2,
    )
    with mock.patch(
        "algorithms.vegetation.hag_provider.load_frv_source",
        side_effect=RuntimeError("load-boom"),
    ):
        with pytest.raises(FrvHagLoadError) as ei:
            session.iter_chunks()
    assert "FRV_STREAM_INCOMPLETE" in str(ei.value)
    assert session._state == "ABORTED"
    session.close()


def test_pass1_read_failure_aborts(tmp_path: Path) -> None:
    reference, configuration, source_result = _source_result(tmp_path / "s.las", count=4)
    fake = _FakeSourceSession(iter_error=RuntimeError("read-boom"))
    session = load_frv_hag(
        reference,
        source_result,
        frv_configuration=configuration,
        provider_config=_hag_config(),
        source_chunk_size=2,
    )
    with mock.patch(
        "algorithms.vegetation.hag_provider.load_frv_source",
        side_effect=[fake],
    ):
        with pytest.raises(FrvHagLoadError) as ei:
            session.iter_chunks()
    assert "FRV_STREAM_INCOMPLETE" in str(ei.value)
    assert session._state == "ABORTED"
    assert fake.closed
    session.close()


def test_pass1_finalize_failure_aborts(tmp_path: Path) -> None:
    reference, configuration, source_result = _source_result(tmp_path / "s.las", count=4)
    fake = _FakeSourceSession(finalize_error=RuntimeError("finalize-boom"))
    session = load_frv_hag(
        reference,
        source_result,
        frv_configuration=configuration,
        provider_config=_hag_config(),
        source_chunk_size=2,
    )
    with mock.patch(
        "algorithms.vegetation.hag_provider.load_frv_source",
        side_effect=[fake],
    ):
        with pytest.raises(FrvHagLoadError) as ei:
            session.iter_chunks()
    assert "FRV_STREAM_INCOMPLETE" in str(ei.value)
    assert session._state == "ABORTED"
    session.close()


def test_pass1_binding_mismatch_aborts(tmp_path: Path) -> None:
    reference, configuration, source_result = _source_result(tmp_path / "s.las", count=4)
    _, _, other_result = _source_result(tmp_path / "other.las", count=6)
    fake = _FakeSourceSession(result=other_result)
    session = load_frv_hag(
        reference,
        source_result,
        frv_configuration=configuration,
        provider_config=_hag_config(),
        source_chunk_size=2,
    )
    with mock.patch(
        "algorithms.vegetation.hag_provider.load_frv_source",
        side_effect=[fake],
    ):
        with pytest.raises(FrvHagLoadError) as ei:
            session.iter_chunks()
    assert "FRV_SOURCE_IDENTITY_FAILED" in str(ei.value)
    assert session._state == "ABORTED"
    session.close()


def test_pass1_close_failure_maps_to_stream_incomplete(tmp_path: Path) -> None:
    reference, configuration, source_result = _source_result(tmp_path / "s.las", count=4)
    fake = _FakeSourceSession(
        result=source_result,
        close_error=RuntimeError("close-boom"),
    )
    session = load_frv_hag(
        reference,
        source_result,
        frv_configuration=configuration,
        provider_config=_hag_config(),
        source_chunk_size=2,
    )
    with mock.patch(
        "algorithms.vegetation.hag_provider.load_frv_source",
        side_effect=[fake],
    ):
        with pytest.raises(FrvHagLoadError) as ei:
            session.iter_chunks()
    assert "FRV_STREAM_INCOMPLETE" in str(ei.value)
    assert session._state == "ABORTED"
    session.close()


# ---------------------------------------------------------------------------
# MAJ-007: first-failure-wins (active exception / cleanup)
# ---------------------------------------------------------------------------


def test_active_exception_not_masked_by_failing_close(tmp_path: Path) -> None:
    session, _ = _run_hag(tmp_path / "s.las", count=4)
    failing = _FakeSourceSession(close_error=RuntimeError("cleanup-boom"))
    session._active_source_sessions.append(failing)
    with pytest.raises(RuntimeError, match="boom"):
        with session:
            raise RuntimeError("boom")
    assert session._state == "CLOSED"


def test_cleanup_failure_suppressed_after_prior_failure(tmp_path: Path) -> None:
    session, _ = _run_hag(tmp_path / "s.las", count=4)
    failing = _FakeSourceSession(close_error=RuntimeError("cleanup-boom"))
    session._active_source_sessions.append(failing)
    session._had_failure = True
    session._state = "ABORTED"
    session.close()  # must not raise: prior failure wins
    assert session._state == "CLOSED"
    assert failing.closed


def test_cleanup_failure_raises_without_prior_failure(tmp_path: Path) -> None:
    session, _ = _run_hag(tmp_path / "s.las", count=4)
    failing = _FakeSourceSession(close_error=RuntimeError("cleanup-boom"))
    session._active_source_sessions.append(failing)
    session._state = "STREAM_COMPLETE"
    with pytest.raises(FrvHagLoadError) as ei:
        session.close()
    assert "cleanup" in str(ei.value)
    assert session._state == "CLOSED"


# ---------------------------------------------------------------------------
# MAJ-007: forged + rehashed Proof/Result rejected by the BND1 adapter
# ---------------------------------------------------------------------------


def test_bnd1_rejects_forged_rehashed_result(tmp_path: Path) -> None:
    reference, configuration, source_result = _source_result(tmp_path / "s.las", count=4)
    session = load_frv_hag(
        reference,
        source_result,
        frv_configuration=configuration,
        provider_config=_hag_config(),
        source_chunk_size=2,
    )
    with session:
        list(session.iter_chunks())
        result = session.finalize()
    # forge a proof whose counts break conservation, then rehash the digest
    forged = object.__new__(hag.FrvHagProviderProof)
    for field_ in dataclasses.fields(hag.FrvHagProviderProof):
        object.__setattr__(forged, field_.name, getattr(result.proof, field_.name))
    object.__setattr__(forged, "supported_point_count", result.proof.supported_point_count - 1)
    object.__setattr__(
        forged, "hag_valid_positive_count", result.proof.hag_valid_positive_count - 1
    )
    object.__setattr__(forged, "proof_digest", forged.computed_digest())
    bad = object.__new__(FrvHagProviderResult)
    object.__setattr__(bad, "proof", forged)
    object.__setattr__(bad, "source_identity", result.source_identity)
    object.__setattr__(bad, "_finalized_capability", hag._FRV_HAG_FINALIZED_CAPABILITY)
    with pytest.raises(ValueError):
        to_frv_a2_hag_provider_binding_evidence(bad)
    with pytest.raises(ValueError):
        validate_finalized_frv_hag_provider_result(bad)


# ---------------------------------------------------------------------------
# MAJ-007: large-domain streaming (no complete index set materialization)
# ---------------------------------------------------------------------------


def _streaming_index_oracle(total: int, source_content_identity_digest: str) -> str:
    """Independent lazy index-domain digest (hashlib + struct, no collection)."""
    import hashlib
    import struct

    hasher = hashlib.sha256()
    hasher.update(b"FRVA2IDX1\0")
    hasher.update(b"ORIGINAL_INDEX")
    hasher.update(b"\0")
    hasher.update(source_content_identity_digest.encode("utf-8"))
    hasher.update(b"\0")
    hasher.update(b"full_source\0")  # FrvA2DomainKind.FULL_SOURCE.value
    hasher.update(struct.pack("<Q", total))
    for index in range(total):
        hasher.update(struct.pack("<Q", index))
    return hasher.hexdigest()


def test_hag_proof_validation_streams_large_domain(tmp_path: Path) -> None:
    # A 2000-point P-HAG proof validates end-to-end and its index-domain
    # digest chains to an independent streaming hash over a lazy range,
    # confirming the whole path never builds tuple(range(N)).
    reference, configuration, source_result = _source_result(tmp_path / "s.las", count=2000)
    session = load_frv_hag(
        reference,
        source_result,
        frv_configuration=configuration,
        provider_config=_hag_config(),
        source_chunk_size=500,
    )
    with session:
        list(session.iter_chunks())
        result = session.finalize()
    validate_finalized_frv_hag_provider_result(result)
    to_frv_a2_hag_provider_binding_evidence(result)
    expected = _streaming_index_oracle(
        2000, result.proof.source_content_identity_digest
    )
    assert result.proof.original_index_domain_digest == expected
    assert len(expected) == 64
