"""Focused P1 project/tower/span/prediction contract tests."""

from __future__ import annotations

from dataclasses import replace
from math import inf, nan

import pytest
from core.contracts.frv import (
    FRV_RELOCATION_AUDIT_SCHEMA_VERSION,
    FRV_SOURCE_IDENTITY_SCHEMA_VERSION,
    FrvRelocationAudit,
    SourceIdentity,
    frv_digest,
)
from core.contracts.project import (
    AnalysisStage,
    AnalysisStatus,
    AnalysisTreeObservationIdentity,
    ArtifactReference,
    COORDINATE_CONTEXT_SCHEMA_VERSION,
    CoordinateContext,
    CoordinateFrame,
    CoordinateUnit,
    PREDICTION_ARTIFACT_REFERENCE_SCHEMA_VERSION,
    PREDICTION_PROVIDER_SCHEMA_VERSION,
    PREDICTION_REQUEST_SCHEMA_VERSION,
    PREDICTION_RESULT_SCHEMA_VERSION,
    PROJECT_SCHEMA_VERSION,
    PROJECT_SOURCE_REFERENCE_SCHEMA_VERSION,
    SPAN_ANALYSIS_SCHEMA_VERSION,
    SPAN_GEOMETRY_SCHEMA_VERSION,
    SPAN_SCHEMA_VERSION,
    TOWER_REFINEMENT_SCHEMA_VERSION,
    TOWER_SCHEMA_VERSION,
    VIEWER_REFERENCE_SCHEMA_VERSION,
    PredictionArtifactReference,
    PredictionCapability,
    PredictionExecutionMode,
    PredictionProviderDescriptor,
    PredictionRequest,
    PredictionResult,
    PredictionResultEntry,
    ProjectContract,
    ProjectSourceReference,
    ProviderPayload,
    SourceIdentityBinding,
    SpanAnalysis,
    SpanContract,
    SpanGeometry,
    SpanTopologyState,
    TowerContract,
    TowerGeometry,
    TowerLifecycleState,
    TowerRefinement,
    TowerRefinementQuality,
    TowerRefinementStatus,
    TowerRevisionReference,
    ViewerCacheState,
    ViewerReference,
    derive_span_geometry_digest,
    span_display_id_for_sequence,
    tower_display_id_for_sequence,
    validate_prediction_result_for_request,
)


SHA = "a" * 64
PROJECT_ID = "123e4567-e89b-12d3-a456-426614174000"
TIMESTAMP = "2026-08-12T00:00:00Z"


def _source_identity() -> SourceIdentity:
    relocation_fields = {
        "audit_schema_version": FRV_RELOCATION_AUDIT_SCHEMA_VERSION,
        "path_style": "posix",
        "recorded_path_raw": "source/line.las",
        "resolved_path_raw": "source/line.las",
        "recorded_path_lexical_normalized": "source/line.las",
        "resolved_path_lexical_normalized": "source/line.las",
        "resolved_final_path": None,
        "resolution_method": "exact_match",
        "case_comparison_mode": "case_sensitive",
        "platform_name": "linux",
        "run_id": "run-1",
        "audit_timestamp": TIMESTAMP,
        "content_identity_result": "matched",
        "warning_flags": 0,
    }
    relocation = FrvRelocationAudit(
        **{
            **relocation_fields,
            "audit_digest": frv_digest(relocation_fields),
        }
    )
    content = {
        "source_identity_schema_version": FRV_SOURCE_IDENTITY_SCHEMA_VERSION,
        "source_id": "line-source",
        "source_file_sha256": SHA,
        "canonical_point_content_digest": None,
        "canonical_point_content_spec_digest": None,
        "canonical_point_content_spec_version": None,
        "source_point_count": 10,
        "source_bounds_min_xyz": [0.0, 0.0, 0.0],
        "source_bounds_max_xyz": [100.0, 100.0, 50.0],
        "coordinate_frame": "SOURCE_LAS_ABSOLUTE",
        "units": "m",
        "origin_context": [0.0, 0.0, 0.0],
    }
    content_digest = frv_digest(content)
    record = {
        **content,
        "recorded_source_path": "source/line.las",
        "resolved_access_path": "source/line.las",
        "relocation_audit_audit_digest": relocation.audit_digest,
        "source_equivalence_audit_audit_digest": None,
        "source_content_identity_digest": content_digest,
    }
    return SourceIdentity(
        **{
            **content,
            "recorded_source_path": "source/line.las",
            "resolved_access_path": "source/line.las",
            "relocation_audit": relocation,
            "source_equivalence_audit": None,
            "source_content_identity_digest": content_digest,
            "source_identity_record_digest": frv_digest(record),
        }
    )


@pytest.fixture
def source_binding() -> SourceIdentityBinding:
    return SourceIdentityBinding.from_source_identity(_source_identity())


def _tower(tower_id: str, binding: SourceIdentityBinding, x: float) -> TowerContract:
    evidence = {"support_count": 32, "diagnostics": {"verticality": 0.99}}
    return TowerContract(
        schema_version=TOWER_SCHEMA_VERSION,
        tower_id=tower_id,
        display_name=tower_id,
        state=TowerLifecycleState.CONFIRMED,
        source_binding=binding,
        coordinate_context=binding.coordinate_context,
        approximate_click_xyz_absolute_m=(x, 10.0, 2.0),
        refined_geometry=TowerGeometry(
            base_xyz_absolute_m=(x, 10.0, 1.0),
            top_xyz_absolute_m=(x, 10.0, 21.0),
            height_m=20.0,
        ),
        refinement=TowerRefinement(
            schema_version=TOWER_REFINEMENT_SCHEMA_VERSION,
            status=TowerRefinementStatus.DEFENSIBLE,
            quality=TowerRefinementQuality.MANUAL_CONFIRMED,
            evidence=evidence,
            algorithm_version="manual-refinement-v1",
            confirmed_at=TIMESTAMP,
        ),
        revision=1,
        notes=None,
        created_at=TIMESTAMP,
        updated_at=TIMESTAMP,
    )


def _geometry(
    binding: SourceIdentityBinding,
    from_tower_id: str,
    to_tower_id: str,
    from_revision: int = 1,
    to_revision: int = 1,
) -> SpanGeometry:
    provisional = SpanGeometry(
        schema_version=SPAN_GEOMETRY_SCHEMA_VERSION,
        definition_revision=1,
        coordinate_context=binding.coordinate_context,
        length_m=20.0,
        direction_xy=(1.0, 0.0),
        core_bounds_min_xyz=(0.0, 0.0, 0.0),
        core_bounds_max_xyz=(20.0, 20.0, 30.0),
        halo_bounds_min_xyz=(-5.0, -5.0, -5.0),
        halo_bounds_max_xyz=(25.0, 25.0, 35.0),
        configuration_digest=SHA,
        geometry_digest=SHA,
    )
    digest = derive_span_geometry_digest(
        source_binding=binding,
        from_tower=_tower_ref(from_tower_id, from_revision),
        to_tower=_tower_ref(to_tower_id, to_revision),
        geometry=provisional,
    )
    return replace(provisional, geometry_digest=digest)


def _tower_ref(tower_id: str, revision: int):
    return TowerRevisionReference(tower_id=tower_id, revision=revision)


def _span(
    binding: SourceIdentityBinding,
    span_id: str = "S001",
    from_tower_id: str = "T001",
    to_tower_id: str = "T002",
    from_revision: int = 1,
    to_revision: int = 1,
    analysis: SpanAnalysis | None = None,
) -> SpanContract:
    return SpanContract(
        schema_version=SPAN_SCHEMA_VERSION,
        span_id=span_id,
        display_name=f"{span_id}  {from_tower_id} -> {to_tower_id}",
        from_tower_id=from_tower_id,
        to_tower_id=to_tower_id,
        from_tower_revision=from_revision,
        to_tower_revision=to_revision,
        source_binding=binding,
        topology_state=SpanTopologyState.CONFIRMED,
        geometry=_geometry(
            binding,
            from_tower_id,
            to_tower_id,
            from_revision=from_revision,
            to_revision=to_revision,
        ),
        analysis=analysis or SpanAnalysis.not_started(),
        artifact_index_reference=ArtifactReference(
            relative_path="spans/S001/artifact-index.json",
            role="artifact-index",
            sha256=SHA,
            byte_size=1,
            schema_version="artifact-index-v1",
        ),
    )


def _provider() -> PredictionProviderDescriptor:
    return PredictionProviderDescriptor(
        schema_version=PREDICTION_PROVIDER_SCHEMA_VERSION,
        provider_id="org.example.growth",
        provider_version="1.0.0",
        supported_input_schema_versions=(PREDICTION_REQUEST_SCHEMA_VERSION,),
        supported_result_schema_versions=(PREDICTION_RESULT_SCHEMA_VERSION,),
        supported_execution_modes=(PredictionExecutionMode.FILE_EXCHANGE,),
    )


def _prediction_artifact() -> PredictionArtifactReference:
    return PredictionArtifactReference(
        schema_version=PREDICTION_ARTIFACT_REFERENCE_SCHEMA_VERSION,
        relative_path="spans/S001/prediction/input.json",
        role="static-input",
        sha256=SHA,
        byte_size=1,
        artifact_schema_version="static-input-v1",
    )


def _project(binding: SourceIdentityBinding) -> ProjectContract:
    return ProjectContract(
        schema_version=PROJECT_SCHEMA_VERSION,
        project_id=PROJECT_ID,
        project_name="Line A",
        created_at=TIMESTAMP,
        updated_at=TIMESTAMP,
        source_reference=ProjectSourceReference(
            schema_version=PROJECT_SOURCE_REFERENCE_SCHEMA_VERSION,
            source_identity=_source_identity(),
            coordinate_context=binding.coordinate_context,
        ),
        viewer=ViewerReference.not_built(),
        tower_registry_reference="towers/towers.json",
        span_registry_reference="spans/spans.json",
        run_index_reference="runs/run-index.json",
        next_tower_sequence=1,
        next_span_sequence=1,
    )


def _prediction_request(
    binding: SourceIdentityBinding,
    *,
    provider_payload: ProviderPayload | None = None,
    static_artifacts: tuple[PredictionArtifactReference, ...] | list[PredictionArtifactReference]
    | None = None,
) -> PredictionRequest:
    span = _span(binding)
    observation = AnalysisTreeObservationIdentity.create("run-001", 17)
    return PredictionRequest(
        schema_version=PREDICTION_REQUEST_SCHEMA_VERSION,
        request_id="request-001",
        project_id=PROJECT_ID,
        source_binding=binding,
        span_id=span.span_id,
        span_dependencies=span.current_dependencies,
        analysis_run_id="run-001",
        static_result_schema_version="static-result-v1",
        coordinate_context=binding.coordinate_context,
        provider=_provider(),
        tree_observations=(observation,),
        static_artifacts=static_artifacts or (_prediction_artifact(),),
        provider_payload=provider_payload
        or ProviderPayload("org.example.growth", "growth-request-v1", {"opaque": [1]}),
        requested_at=TIMESTAMP,
    )


def _prediction_result(
    request: PredictionRequest,
    *,
    entries: dict[str, PredictionResultEntry] | None = None,
) -> PredictionResult:
    observation = request.tree_observations[0]
    result_entries = entries or {
        observation.observation_id: PredictionResultEntry(
            observation=observation,
            provider_payload=ProviderPayload("org.example.growth", "growth-result-v1", {}),
        )
    }
    return PredictionResult(
        schema_version=PREDICTION_RESULT_SCHEMA_VERSION,
        result_id="result-001",
        request_id=request.request_id,
        request_schema_version=request.schema_version,
        project_id=request.project_id,
        source_binding=request.source_binding,
        span_id=request.span_id,
        span_dependencies=request.span_dependencies,
        analysis_run_id=request.analysis_run_id,
        static_result_schema_version=request.static_result_schema_version,
        coordinate_context=request.coordinate_context,
        provider_id=request.provider.provider_id,
        provider_version=request.provider.provider_version,
        model_version="model-v1",
        entries=result_entries,
        artifacts=(),
        provider_payload=ProviderPayload("org.example.growth", "result-v1", {}),
        completed_at=TIMESTAMP,
    )


def test_project_reuses_complete_source_identity_and_defaults_prediction_not_connected(
    source_binding: SourceIdentityBinding,
) -> None:
    source = _source_identity()
    source_reference = ProjectSourceReference(
        schema_version=PROJECT_SOURCE_REFERENCE_SCHEMA_VERSION,
        source_identity=source,
        coordinate_context=source_binding.coordinate_context,
    )
    project = ProjectContract(
        schema_version=PROJECT_SCHEMA_VERSION,
        project_id=PROJECT_ID,
        project_name="Line A",
        created_at=TIMESTAMP,
        updated_at=TIMESTAMP,
        source_reference=source_reference,
        viewer=ViewerReference(
            schema_version=VIEWER_REFERENCE_SCHEMA_VERSION,
            state=ViewerCacheState.NOT_BUILT,
            active_cache_id=None,
        ),
        tower_registry_reference="towers/towers.json",
        span_registry_reference="spans/spans.json",
        run_index_reference="runs/run-index.json",
        next_tower_sequence=1,
        next_span_sequence=1,
    )

    assert project.source_reference.binding == source_binding
    assert project.prediction_capability.status.value == "NOT_CONNECTED"
    assert project.to_dict()["source_reference"]["source_identity"]["source_id"] == "line-source"


def test_project_schema_sequences_and_display_id_derivation_fail_closed(
    source_binding: SourceIdentityBinding,
) -> None:
    source_reference = ProjectSourceReference(
        PROJECT_SOURCE_REFERENCE_SCHEMA_VERSION,
        _source_identity(),
        source_binding.coordinate_context,
    )
    common = {
        "project_id": PROJECT_ID,
        "project_name": "Line A",
        "created_at": TIMESTAMP,
        "updated_at": TIMESTAMP,
        "source_reference": source_reference,
        "viewer": ViewerReference.not_built(),
        "tower_registry_reference": "towers/towers.json",
        "span_registry_reference": "spans/spans.json",
        "run_index_reference": "runs/run-index.json",
        "next_tower_sequence": 1,
        "next_span_sequence": 1,
    }
    with pytest.raises(ValueError, match="schema"):
        ProjectContract(schema_version="treeline-project-v2", **common)
    with pytest.raises(ValueError, match="next_tower_sequence"):
        ProjectContract(
            schema_version=PROJECT_SCHEMA_VERSION,
            **{**common, "next_tower_sequence": 0},
        )
    assert tower_display_id_for_sequence(1) == "T001"
    assert tower_display_id_for_sequence(1000) == "T1000"
    assert span_display_id_for_sequence(1) == "S001"
    with pytest.raises(ValueError, match="sequence"):
        span_display_id_for_sequence(0)


def test_coordinate_context_rejects_unknown_enums_and_malformed_digests(
    source_binding: SourceIdentityBinding,
) -> None:
    common = {
        "schema_version": COORDINATE_CONTEXT_SCHEMA_VERSION,
        "frame": CoordinateFrame.SOURCE_LAS_ABSOLUTE,
        "units": CoordinateUnit.METRE,
        "origin_xyz": (0.0, 0.0, 0.0),
        "source_content_identity_digest": source_binding.source_content_identity_digest,
        "source_identity_record_digest": source_binding.source_identity_record_digest,
    }
    with pytest.raises(ValueError, match="frame"):
        CoordinateContext(**{**common, "frame": "SOURCE_LAS_ABSOLUTE"})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="SHA-256"):
        CoordinateContext(**{**common, "source_content_identity_digest": "A" * 64})


@pytest.mark.parametrize(
    "factory",
    [
        lambda binding: _project(binding),
        lambda binding: _project(binding).source_reference,
        lambda binding: binding.coordinate_context,
        lambda binding: ViewerReference.not_built(),
        lambda binding: _tower("T001", binding, 0.0),
        lambda binding: _tower("T001", binding, 0.0).refinement,
        lambda binding: _span(binding),
        lambda binding: _geometry(binding, "T001", "T002"),
        lambda binding: SpanAnalysis.not_started(),
        lambda binding: PredictionCapability.not_connected(),
        lambda binding: _provider(),
        lambda binding: _prediction_artifact(),
        lambda binding: _prediction_request(binding),
        lambda binding: _prediction_result(_prediction_request(binding)),
    ],
    ids=[
        "project",
        "project-source-reference",
        "coordinate-context",
        "viewer-reference",
        "tower",
        "tower-refinement",
        "span",
        "span-geometry",
        "span-analysis",
        "prediction-capability",
        "prediction-provider",
        "prediction-artifact-reference",
        "prediction-request",
        "prediction-result",
    ],
)
def test_exact_p1_schema_versions_reject_non_exact_variants(
    source_binding: SourceIdentityBinding,
    factory,
) -> None:
    contract = factory(source_binding)
    exact_schema_version = contract.schema_version

    for invalid_schema_version in (
        f"prefix-{exact_schema_version}",
        f"{exact_schema_version}-suffix",
        f"{exact_schema_version} ",
        exact_schema_version.upper(),
    ):
        with pytest.raises(ValueError, match="schema"):
            replace(contract, schema_version=invalid_schema_version)


@pytest.mark.parametrize("project_id", ["", "UPPERCASE", "123E4567-E89B-12D3-A456-426614174000"])
def test_project_id_is_canonical_uuid(
    source_binding: SourceIdentityBinding, project_id: str
) -> None:
    with pytest.raises(ValueError, match="project_id"):
        ProjectContract(
            schema_version=PROJECT_SCHEMA_VERSION,
            project_id=project_id,
            project_name="Line A",
            created_at=TIMESTAMP,
            updated_at=TIMESTAMP,
            source_reference=ProjectSourceReference(
                PROJECT_SOURCE_REFERENCE_SCHEMA_VERSION,
                _source_identity(),
                source_binding.coordinate_context,
            ),
            viewer=ViewerReference.not_built(),
            tower_registry_reference="towers/towers.json",
            span_registry_reference="spans/spans.json",
            run_index_reference="runs/run-index.json",
            next_tower_sequence=1,
            next_span_sequence=1,
        )


@pytest.mark.parametrize("tower_id", ["", "T0", "T1", "Tower001", "t001", "T01"])
def test_tower_ids_are_strict(source_binding: SourceIdentityBinding, tower_id: str) -> None:
    with pytest.raises(ValueError, match="tower_id"):
        _tower(tower_id, source_binding, 0.0)


@pytest.mark.parametrize("span_id", ["", "S0", "S1", "Span001", "s001", "S01"])
def test_span_ids_are_strict(source_binding: SourceIdentityBinding, span_id: str) -> None:
    with pytest.raises(ValueError, match="span_id"):
        _span(source_binding, span_id=span_id)


@pytest.mark.parametrize(
    ("coordinate", "message"),
    [((nan, 0.0, 0.0), "finite"), ((inf, 0.0, 0.0), "finite"), ((0.0, 0.0), "exactly 3")],
)
def test_coordinate_context_rejects_nonfinite_and_malformed_xyz(
    source_binding: SourceIdentityBinding, coordinate: tuple[float, ...], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        CoordinateContext(
            schema_version=COORDINATE_CONTEXT_SCHEMA_VERSION,
            frame=CoordinateFrame.SOURCE_LAS_ABSOLUTE,
            units=CoordinateUnit.METRE,
            origin_xyz=coordinate,  # type: ignore[arg-type]
            source_content_identity_digest=source_binding.source_content_identity_digest,
            source_identity_record_digest=source_binding.source_identity_record_digest,
        )


def test_span_rejects_self_reference_bad_revision_and_bad_direction(
    source_binding: SourceIdentityBinding,
) -> None:
    with pytest.raises(ValueError, match="same tower"):
        _span(source_binding, to_tower_id="T001")
    with pytest.raises(ValueError, match="positive"):
        _span(source_binding, from_revision=0)

    geometry = _geometry(source_binding, "T001", "T002")
    with pytest.raises(ValueError, match="unit vector"):
        replace(geometry, direction_xy=(2.0, 0.0))


@pytest.mark.parametrize(
    ("active_run_id", "run_dependencies"),
    [
        pytest.param("run-001", None, id="active-run-only"),
        pytest.param(None, "dependencies", id="dependencies-only"),
        pytest.param("run-001", "dependencies", id="active-run-and-dependencies"),
    ],
)
def test_not_started_analysis_rejects_any_run_provenance(
    source_binding: SourceIdentityBinding,
    active_run_id: str | None,
    run_dependencies: str | None,
) -> None:
    dependencies = _span(source_binding).current_dependencies
    with pytest.raises(ValueError, match="NOT_STARTED"):
        SpanAnalysis(
            schema_version=SPAN_ANALYSIS_SCHEMA_VERSION,
            state=AnalysisStatus.NOT_STARTED,
            active_run_id=active_run_id,
            stage_statuses={stage: AnalysisStatus.NOT_STARTED for stage in AnalysisStage},
            run_dependencies=dependencies if run_dependencies is not None else None,
        )


def test_not_started_analysis_without_run_provenance_is_valid() -> None:
    analysis = SpanAnalysis.not_started()

    assert analysis.active_run_id is None
    assert analysis.run_dependencies is None


def test_span_geometry_allows_halo_larger_than_or_equal_to_core(
    source_binding: SourceIdentityBinding,
) -> None:
    geometry = _geometry(source_binding, "T001", "T002")
    assert geometry.halo_bounds_min_xyz < geometry.core_bounds_min_xyz
    equal_halo = replace(
        geometry,
        halo_bounds_min_xyz=geometry.core_bounds_min_xyz,
        halo_bounds_max_xyz=geometry.core_bounds_max_xyz,
    )

    assert equal_halo.halo_bounds_min_xyz == equal_halo.core_bounds_min_xyz
    assert equal_halo.halo_bounds_max_xyz == equal_halo.core_bounds_max_xyz


@pytest.mark.parametrize(
    ("halo_minimum", "halo_maximum", "axis"),
    [
        ((1.0, -5.0, -5.0), (25.0, 25.0, 35.0), "X"),
        ((-5.0, 1.0, -5.0), (25.0, 25.0, 35.0), "Y"),
        ((-5.0, -5.0, 1.0), (25.0, 25.0, 35.0), "Z"),
        ((21.0, 21.0, 31.0), (41.0, 41.0, 41.0), "disjoint"),
    ],
)
def test_span_geometry_rejects_halo_that_does_not_contain_core(
    source_binding: SourceIdentityBinding,
    halo_minimum: tuple[float, float, float],
    halo_maximum: tuple[float, float, float],
    axis: str,
) -> None:
    del axis
    geometry = _geometry(source_binding, "T001", "T002")
    with pytest.raises(ValueError, match="halo_bounds"):
        replace(
            geometry,
            halo_bounds_min_xyz=halo_minimum,
            halo_bounds_max_xyz=halo_maximum,
        )


def test_shared_tower_is_referenced_by_two_spans_without_duplicate_tower_objects(
    source_binding: SourceIdentityBinding,
) -> None:
    towers = {
        tower.tower_id: tower
        for tower in (
            _tower("T001", source_binding, 0.0),
            _tower("T002", source_binding, 20.0),
            _tower("T003", source_binding, 40.0),
        )
    }
    spans = (
        _span(source_binding, "S001", "T001", "T002"),
        _span(source_binding, "S002", "T002", "T003"),
    )

    assert set(towers) == {"T001", "T002", "T003"}
    assert len(towers) == 3
    assert spans[0].to_tower_id == "T002"
    assert spans[1].from_tower_id == "T002"
    assert spans[0].to_tower_id == spans[1].from_tower_id
    assert "refined_geometry" not in spans[0].to_dict()


def test_archived_tower_retains_its_allocated_identity_for_future_tombstoning(
    source_binding: SourceIdentityBinding,
) -> None:
    archived = replace(_tower("T001", source_binding, 0.0), state=TowerLifecycleState.ARCHIVED)

    assert archived.tower_id == "T001"
    assert archived.display_name == "T001"
    assert archived.revision == 1


def test_shared_tower_revision_dependency_isolated_from_unrelated_spans(
    source_binding: SourceIdentityBinding,
) -> None:
    first = _span(source_binding, "S001", "T001", "T002")
    second = _span(source_binding, "S002", "T002", "T003")
    unrelated = _span(source_binding, "S003", "T004", "T005")
    first_after_t002_revision = _span(
        source_binding,
        "S001",
        "T001",
        "T002",
        to_revision=2,
    )
    second_after_t002_revision = _span(
        source_binding,
        "S002",
        "T002",
        "T003",
        from_revision=2,
    )

    assert first.current_dependencies != first_after_t002_revision.current_dependencies
    assert second.current_dependencies != second_after_t002_revision.current_dependencies
    assert unrelated.current_dependencies == _span(
        source_binding, "S003", "T004", "T005"
    ).current_dependencies
    assert "T002" not in {
        unrelated.current_dependencies.from_tower.tower_id,
        unrelated.current_dependencies.to_tower.tower_id,
    }


def test_tower_revision_change_requires_stale_span_analysis(
    source_binding: SourceIdentityBinding,
) -> None:
    initial = _span(source_binding)
    stale_analysis = SpanAnalysis(
        schema_version=SPAN_ANALYSIS_SCHEMA_VERSION,
        state=AnalysisStatus.STALE,
        active_run_id="run-001",
        stage_statuses={stage: AnalysisStatus.STALE for stage in AnalysisStage},
        run_dependencies=initial.current_dependencies,
        historical_run_ids=("run-001",),
    )
    stale_span = _span(source_binding, from_revision=2, analysis=stale_analysis)

    assert stale_span.is_stale
    assert stale_span.analysis.run_dependencies is not None
    assert stale_span.analysis.run_dependencies.from_tower.revision == 1
    assert stale_span.analysis.historical_run_ids == ("run-001",)

    complete_analysis = SpanAnalysis(
        schema_version=SPAN_ANALYSIS_SCHEMA_VERSION,
        state=AnalysisStatus.COMPLETE,
        active_run_id="run-001",
        stage_statuses={stage: AnalysisStatus.COMPLETE for stage in AnalysisStage},
        run_dependencies=initial.current_dependencies,
    )
    with pytest.raises(ValueError, match="must be STALE"):
        _span(source_binding, from_revision=2, analysis=complete_analysis)


def test_span_staleness_is_representable_for_source_and_configuration_changes(
    source_binding: SourceIdentityBinding,
) -> None:
    initial = _span(source_binding)
    stale_analysis = SpanAnalysis(
        schema_version=SPAN_ANALYSIS_SCHEMA_VERSION,
        state=AnalysisStatus.STALE,
        active_run_id="run-001",
        stage_statuses={stage: AnalysisStatus.STALE for stage in AnalysisStage},
        run_dependencies=initial.current_dependencies,
        historical_run_ids=("run-001",),
    )
    changed_configuration = replace(
        initial.geometry,
        configuration_digest="b" * 64,
        geometry_digest=SHA,
    )
    changed_configuration = replace(
        changed_configuration,
        geometry_digest=derive_span_geometry_digest(
            source_binding=source_binding,
            from_tower=initial.from_tower_reference,
            to_tower=initial.to_tower_reference,
            geometry=changed_configuration,
        ),
    )
    assert replace(initial, geometry=changed_configuration, analysis=stale_analysis).is_stale

    changed_context = CoordinateContext(
        schema_version=COORDINATE_CONTEXT_SCHEMA_VERSION,
        frame=CoordinateFrame.SOURCE_LAS_ABSOLUTE,
        units=CoordinateUnit.METRE,
        origin_xyz=(0.0, 0.0, 0.0),
        source_content_identity_digest="b" * 64,
        source_identity_record_digest="c" * 64,
    )
    changed_source = SourceIdentityBinding(
        source_content_identity_digest="b" * 64,
        source_identity_record_digest="c" * 64,
        coordinate_context=changed_context,
    )
    changed_geometry = replace(
        initial.geometry,
        coordinate_context=changed_context,
        geometry_digest=SHA,
    )
    changed_geometry = replace(
        changed_geometry,
        geometry_digest=derive_span_geometry_digest(
            source_binding=changed_source,
            from_tower=initial.from_tower_reference,
            to_tower=initial.to_tower_reference,
            geometry=changed_geometry,
        ),
    )
    assert replace(
        initial,
        source_binding=changed_source,
        geometry=changed_geometry,
        analysis=stale_analysis,
    ).is_stale


def test_analysis_state_requires_consistent_active_run_and_stage_statuses() -> None:
    with pytest.raises(ValueError, match="active_run_id"):
        SpanAnalysis(
            schema_version=SPAN_ANALYSIS_SCHEMA_VERSION,
            state=AnalysisStatus.COMPLETE,
            active_run_id=None,
            stage_statuses={stage: AnalysisStatus.COMPLETE for stage in AnalysisStage},
            run_dependencies=None,
        )
    with pytest.raises(ValueError, match="NOT_STARTED"):
        SpanAnalysis(
            schema_version=SPAN_ANALYSIS_SCHEMA_VERSION,
            state=AnalysisStatus.NOT_STARTED,
            active_run_id=None,
            stage_statuses={stage: AnalysisStatus.READY for stage in AnalysisStage},
            run_dependencies=None,
        )


def test_metadata_and_stage_mappings_are_alias_safe(source_binding: SourceIdentityBinding) -> None:
    evidence = {"diagnostics": {"support": [1, 2]}}
    refinement = TowerRefinement(
        schema_version=TOWER_REFINEMENT_SCHEMA_VERSION,
        status=TowerRefinementStatus.DEFENSIBLE,
        quality=TowerRefinementQuality.MANUAL_CONFIRMED,
        evidence=evidence,
        algorithm_version="manual-refinement-v1",
        confirmed_at=TIMESTAMP,
    )
    evidence["diagnostics"]["support"].append(3)
    assert refinement.to_dict()["evidence"]["diagnostics"]["support"] == [1, 2]
    with pytest.raises(TypeError):
        refinement.evidence["new"] = "value"  # type: ignore[index]

    stages = {stage: AnalysisStatus.NOT_STARTED for stage in AnalysisStage}
    analysis = SpanAnalysis(
        schema_version=SPAN_ANALYSIS_SCHEMA_VERSION,
        state=AnalysisStatus.NOT_STARTED,
        active_run_id=None,
        stage_statuses=stages,
        run_dependencies=None,
    )
    stages[AnalysisStage.GROUND] = AnalysisStatus.COMPLETE
    assert analysis.stage_statuses[AnalysisStage.GROUND] is AnalysisStatus.NOT_STARTED


@pytest.mark.parametrize(
    "relative_path",
    [
        "C:/project/artifact.json",
        "//server/share/artifact.json",
        "../artifact.json",
        "spans\\S001\\x.json",
        "NUL",
        "nul",
        "NUL.txt",
        "CON",
        "CON.json",
        "aux/data",
        "foo/COM1.txt",
        "foo/LPT9.bin",
        "foo/CON .txt",
        "foo/\x00name.json",
        "foo/\x1fname.json",
        "file.txt:stream",
        "foo?.json",
        "foo.",
        "foo ",
        "folder./file.json",
        "folder /file.json",
    ],
)
def test_prediction_artifact_path_rejects_unsafe_forms(relative_path: str) -> None:
    with pytest.raises(ValueError, match="path"):
        PredictionArtifactReference(
            schema_version=PREDICTION_ARTIFACT_REFERENCE_SCHEMA_VERSION,
            relative_path=relative_path,
            role="result",
            sha256=SHA,
            byte_size=1,
            artifact_schema_version="prediction-output-v1",
        )


@pytest.mark.parametrize(
    "relative_path",
    ["console.json", "auxiliary.json", "foo/com10.bin", "foo/lpt10.bin", "spans/S001/x.json"],
)
def test_prediction_artifact_path_allows_non_reserved_near_misses(relative_path: str) -> None:
    reference = PredictionArtifactReference(
        schema_version=PREDICTION_ARTIFACT_REFERENCE_SCHEMA_VERSION,
        relative_path=relative_path,
        role="result",
        sha256=SHA,
        byte_size=1,
        artifact_schema_version="prediction-output-v1",
    )

    assert reference.relative_path == relative_path


def test_prediction_uses_analysis_scoped_tree_identity_and_identity_keyed_entries(
    source_binding: SourceIdentityBinding,
) -> None:
    span = _span(source_binding)
    observation = AnalysisTreeObservationIdentity.create("run-001", 17)
    other_run_observation = AnalysisTreeObservationIdentity.create("run-002", 17)
    assert observation.observation_id != other_run_observation.observation_id

    request = PredictionRequest(
        schema_version=PREDICTION_REQUEST_SCHEMA_VERSION,
        request_id="request-001",
        project_id=PROJECT_ID,
        source_binding=source_binding,
        span_id=span.span_id,
        span_dependencies=span.current_dependencies,
        analysis_run_id="run-001",
        static_result_schema_version="static-result-v1",
        coordinate_context=source_binding.coordinate_context,
        provider=_provider(),
        tree_observations=(observation,),
        static_artifacts=(_prediction_artifact(),),
        provider_payload=ProviderPayload(
            "org.example.growth", "growth-request-v1", {"opaque": [1]}
        ),
        requested_at=TIMESTAMP,
    )
    result = PredictionResult(
        schema_version=PREDICTION_RESULT_SCHEMA_VERSION,
        result_id="result-001",
        request_id=request.request_id,
        request_schema_version=request.schema_version,
        project_id=request.project_id,
        source_binding=request.source_binding,
        span_id=request.span_id,
        span_dependencies=request.span_dependencies,
        analysis_run_id=request.analysis_run_id,
        static_result_schema_version=request.static_result_schema_version,
        coordinate_context=request.coordinate_context,
        provider_id=request.provider.provider_id,
        provider_version=request.provider.provider_version,
        model_version="model-v1",
        entries={
            observation.observation_id: PredictionResultEntry(
                observation=observation,
                provider_payload=ProviderPayload("org.example.growth", "growth-result-v1", {}),
            )
        },
        artifacts=(),
        provider_payload=ProviderPayload("org.example.growth", "result-v1", {}),
        completed_at=TIMESTAMP,
    )

    validate_prediction_result_for_request(request, result)
    assert tuple(result.entries) == ("run-001:tree:17",)
    with pytest.raises(ValueError, match="observation_id"):
        AnalysisTreeObservationIdentity("run-001", 17, "tree-17")


@pytest.mark.parametrize(
    ("field_name", "replacement"),
    [
        ("project_id", "123e4567-e89b-12d3-a456-426614174001"),
        ("span_id", "S002"),
        ("analysis_run_id", "run-002"),
        ("provider_id", "org.example.other"),
        ("provider_version", "2.0.0"),
    ],
)
def test_prediction_result_validation_rejects_stable_identity_mismatch(
    source_binding: SourceIdentityBinding,
    field_name: str,
    replacement: str,
) -> None:
    request = _prediction_request(source_binding)
    result = _prediction_result(request)

    with pytest.raises(ValueError, match=field_name):
        validate_prediction_result_for_request(
            request, replace(result, **{field_name: replacement})
        )


def test_prediction_payload_and_artifact_collections_are_alias_safe(
    source_binding: SourceIdentityBinding,
) -> None:
    payload_data = {"nested": {"scores": [1, 2]}}
    artifact_list = [_prediction_artifact()]
    request = _prediction_request(
        source_binding,
        provider_payload=ProviderPayload("org.example.growth", "growth-request-v1", payload_data),
        static_artifacts=artifact_list,
    )
    payload_data["nested"]["scores"].append(3)
    artifact_list.clear()

    assert request.provider_payload.to_dict()["data"]["nested"]["scores"] == [1, 2]
    assert request.static_artifacts == (_prediction_artifact(),)
    with pytest.raises(TypeError):
        request.provider_payload.data["replacement"] = "value"  # type: ignore[index]

    observation = request.tree_observations[0]
    result_entries = {
        observation.observation_id: PredictionResultEntry(
            observation=observation,
            provider_payload=ProviderPayload("org.example.growth", "growth-result-v1", {}),
        )
    }
    result = _prediction_result(request, entries=result_entries)
    result_entries.clear()
    assert len(result.entries) == 1
    with pytest.raises(TypeError):
        result.entries["replacement"] = next(iter(result.entries.values()))  # type: ignore[index]


def test_prediction_result_rejects_array_position_style_mismatched_entry_key(
    source_binding: SourceIdentityBinding,
) -> None:
    span = _span(source_binding)
    observation = AnalysisTreeObservationIdentity.create("run-001", 1)
    with pytest.raises(ValueError, match="keyed"):
        PredictionResult(
            schema_version=PREDICTION_RESULT_SCHEMA_VERSION,
            result_id="result-001",
            request_id="request-001",
            request_schema_version=PREDICTION_REQUEST_SCHEMA_VERSION,
            project_id=PROJECT_ID,
            source_binding=source_binding,
            span_id=span.span_id,
            span_dependencies=span.current_dependencies,
            analysis_run_id="run-001",
            static_result_schema_version="static-result-v1",
            coordinate_context=source_binding.coordinate_context,
            provider_id="org.example.growth",
            provider_version="1.0.0",
            model_version="model-v1",
            entries={
                "row-17": PredictionResultEntry(
                    observation, ProviderPayload("org.example.growth", "v1", {})
                )
            },
            artifacts=(),
            provider_payload=ProviderPayload("org.example.growth", "v1", {}),
            completed_at=TIMESTAMP,
        )
