"""Public contracts shared by TreeLineWarning pipeline modules."""

from core.contracts.errors import UnifiedError
from core.contracts.features import (
    FeatureQualityFlag,
    GeometryFeatureConfig,
    GeometryFeatureResult,
    GeometryFeatureStatistics,
    PointFeatureSet,
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
    "CropBounds",
    "CropResult",
    "DownsampledPointCloud",
    "FeatureQualityFlag",
    "GeometryFeatureConfig",
    "GeometryFeatureResult",
    "GeometryFeatureStatistics",
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
]
