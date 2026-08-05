"""V1 vegetation candidate helpers."""

from algorithms.vegetation.associations import (
    VegetationAssociationError,
    load_vegetation_associations,
    run_vegetation_associations,
)
from algorithms.vegetation.candidates import (
    VegetationCandidateError,
    generate_vegetation_candidate_preview,
    load_vegetation_candidates,
    run_vegetation_candidate_selection,
)
from algorithms.vegetation.canopy_height_model import (
    CanopyHeightModelConfig,
    CanopyHeightModelError,
    CanopyHeightModelErrorCode,
    CanopyHeightModelResult,
    CanopySupportFlag,
    build_canopy_height_model,
)
from algorithms.vegetation.canopy_segmentation import (
    CanopySegmentationConfig,
    CanopySegmentationError,
    CanopySegmentationErrorCode,
    CanopySegmentationResult,
    segment_canopy_regions,
)
from algorithms.vegetation.hag_provider import load_frv_hag
from algorithms.vegetation.instances import (
    VegetationInstanceError,
    generate_vegetation_instances_preview,
    load_vegetation_instances,
    run_vegetation_instance_grouping,
)
from algorithms.vegetation.tree_artifacts import (
    LoadedTreeSegmentationArtifact,
    TreeSegmentationArtifactError,
    load_tree_segmentation_artifact,
)
from algorithms.vegetation.tree_segmentation import (
    TreeResultConstructionConfig,
    TreeResultConstructionError,
    TreeResultConstructionErrorCode,
    build_tree_segmentation_result,
)
from algorithms.vegetation.treetop_candidates import (
    TreetopCandidateResult,
    TreetopDetectionConfig,
    TreetopDetectionError,
    TreetopDetectionErrorCode,
    detect_treetop_candidates,
)

__all__ = [
    "CanopyHeightModelConfig",
    "CanopyHeightModelError",
    "CanopyHeightModelErrorCode",
    "CanopyHeightModelResult",
    "CanopySegmentationConfig",
    "CanopySegmentationError",
    "CanopySegmentationErrorCode",
    "CanopySegmentationResult",
    "CanopySupportFlag",
    "LoadedTreeSegmentationArtifact",
    "TreeSegmentationArtifactError",
    "TreeResultConstructionConfig",
    "TreeResultConstructionError",
    "TreeResultConstructionErrorCode",
    "TreetopCandidateResult",
    "TreetopDetectionConfig",
    "TreetopDetectionError",
    "TreetopDetectionErrorCode",
    "VegetationCandidateError",
    "detect_treetop_candidates",
    "generate_vegetation_candidate_preview",
    "load_frv_hag",
    "load_vegetation_candidates",
    "load_tree_segmentation_artifact",
    "run_vegetation_candidate_selection",
    "VegetationInstanceError",
    "generate_vegetation_instances_preview",
    "load_vegetation_instances",
    "run_vegetation_instance_grouping",
    "VegetationAssociationError",
    "load_vegetation_associations",
    "run_vegetation_associations",
    "build_canopy_height_model",
    "build_tree_segmentation_result",
    "segment_canopy_regions",
]
