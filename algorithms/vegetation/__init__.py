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
from algorithms.vegetation.frv_artifacts import (
    FRV_BLOCK_MAGIC,
    FRV_EVIDENCE_BLOCK_MAGIC,
    build_frv_payload_schema,
    compute_art1_artifact_sha256,
    decode_frv_block,
    encode_frv_block,
)
from algorithms.vegetation.frv_membership import (
    build_frv_membership,
    load_frv_membership_result,
)
from algorithms.vegetation.geometry_provider import load_frv_geometry
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
from algorithms.vegetation.v4_inputs import (
    build_v4_input_handoff,
    load_v4_input_handoff,
)
from algorithms.vegetation.v4_tree_artifacts import (
    load_v4_tree_segmentation_artifact,
    publish_v4_tree_segmentation_artifact,
)
from algorithms.vegetation.v4_tree_segmentation import segment_v4_trees
from algorithms.vegetation.wire_provider import load_frv_wire

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
    "FRV_BLOCK_MAGIC",
    "FRV_EVIDENCE_BLOCK_MAGIC",
    "VegetationCandidateError",
    "build_frv_membership",
    "build_frv_payload_schema",
    "build_v4_input_handoff",
    "compute_art1_artifact_sha256",
    "load_v4_input_handoff",
    "load_v4_tree_segmentation_artifact",
    "publish_v4_tree_segmentation_artifact",
    "segment_v4_trees",
    "decode_frv_block",
    "detect_treetop_candidates",
    "encode_frv_block",
    "generate_vegetation_candidate_preview",
    "load_frv_geometry",
    "load_frv_hag",
    "load_frv_membership_result",
    "load_frv_wire",
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
