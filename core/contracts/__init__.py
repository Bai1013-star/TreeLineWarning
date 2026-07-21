"""Public contracts shared by TreeLineWarning pipeline modules."""

from core.contracts.candidates import (
    CandidateEvidenceSet,
    CandidateLevel,
    CandidatePointSet,
    CandidateReasonFlag,
    CandidateSource,
    ManualSelectionConfig,
    WireCandidateConfig,
    WireCandidateQualityFlag,
    WireCandidateResult,
    WireCandidateStatistics,
)
from core.contracts.errors import UnifiedError
from core.contracts.features import (
    FeatureQualityFlag,
    GeometryFeatureConfig,
    GeometryFeatureResult,
    GeometryFeatureStatistics,
    PointFeatureSet,
)
from core.contracts.ground import (
    GroundEstimationConfig,
    GroundEstimationResult,
    GroundEstimationStatistics,
    GroundQualityFlag,
    GroundSurfaceModel,
    HeightAboveGroundSet,
)
from core.contracts.pointcloud import (
    CropBounds,
    CropResult,
    LasHeaderSummary,
    PointCloudChunk,
    PointCloudSource,
)
from core.contracts.preprocessing import (
    DownsampledPointCloud,
    PreprocessingConfig,
    PreprocessingResult,
    ProcessingStatistics,
)
from core.contracts.progress import CancellationToken, PipelineProgress, ProgressReporter

__all__ = [
    "CancellationToken",
    "CandidateEvidenceSet",
    "CandidateLevel",
    "CandidatePointSet",
    "CandidateReasonFlag",
    "CandidateSource",
    "CropBounds",
    "CropResult",
    "DownsampledPointCloud",
    "FeatureQualityFlag",
    "GeometryFeatureConfig",
    "GeometryFeatureResult",
    "GeometryFeatureStatistics",
    "ManualSelectionConfig",
    "GroundEstimationConfig",
    "GroundEstimationResult",
    "GroundEstimationStatistics",
    "GroundQualityFlag",
    "GroundSurfaceModel",
    "HeightAboveGroundSet",
    "LasHeaderSummary",
    "PipelineProgress",
    "PointCloudChunk",
    "PointCloudSource",
    "PointFeatureSet",
    "PreprocessingConfig",
    "PreprocessingResult",
    "ProcessingStatistics",
    "ProgressReporter",
    "UnifiedError",
    "WireCandidateConfig",
    "WireCandidateQualityFlag",
    "WireCandidateResult",
    "WireCandidateStatistics",
]
