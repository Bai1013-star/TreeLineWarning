"""Stage-A product foundation for TreeLineWarning.

The modules in this package sit around the validated single-span science
pipeline.  They deliberately do not replace the existing P1/P2 project
contracts or implement a new science backend.
"""

from core.stage_a.capability import (
    CapabilityReport,
    CapabilityState,
    inspect_point_cloud,
)
from core.stage_a.crs_transform import (
    CRSTransformPlan,
    CRSTransformUnavailable,
    build_crs_transform,
    offline_proj_status,
    transform_xy,
    transform_xyz,
)
from core.stage_a.gis import (
    CanonicalGISImport,
    GISImportReport,
    GISSourceIdentity,
    RouteSeed,
    TowerSeed,
    import_gis,
    read_csv,
    read_kml,
)
from core.stage_a.models import (
    AnalysisRun,
    AnalysisRunStatus,
    CoordinateDeclaration,
    Line,
    LineRole,
    MetricDistance,
    Project,
    Route,
    Scan,
    SourceDescriptor,
    Span,
    Tower,
    UnitStatus,
    to_metric_distance,
)
from core.stage_a.offline import (
    ALLOWED_BASEMAP_POLICIES,
    OFFLINE_RUNTIME_SCHEMA,
    RUNTIME_NETWORK_REQUIRED,
)
from core.stage_a.persistence import (
    STAGE_A_FILENAME,
    LegacyProjectCompatibility,
    load_project,
    migrate_legacy_project,
    open_project,
    save_project,
)
from core.stage_a.registration import (
    AnchorCorrespondence,
    RegistrationResult,
    RegistrationState,
    apply_registration_xy,
    georeferenced_registration,
    no_gis_registration,
    register_one_anchor,
    register_two_anchors,
    validate_registration_checkpoint,
)
from core.stage_a.registration_persistence import (
    REGISTRATION_METADATA_KEY,
    REGISTRATION_METADATA_SCHEMA,
    save_scan_registration,
    scan_registration_record,
)
from core.stage_a.risk import (
    ReviewState,
    RiskAssessment,
    RiskClass,
    RiskPolicy,
)
from core.stage_a.shared_source import (
    CacheState,
    SharedSourceAccess,
    SourceAccessMetrics,
)
from core.stage_a.topology import (
    generate_spans_from_ordered_towers,
    generate_spans_from_route,
)

__all__ = [
    "ALLOWED_BASEMAP_POLICIES",
    "AnalysisRun",
    "AnalysisRunStatus",
    "AnchorCorrespondence",
    "CRSTransformPlan",
    "CRSTransformUnavailable",
    "CacheState",
    "CanonicalGISImport",
    "CapabilityReport",
    "CapabilityState",
    "CoordinateDeclaration",
    "GISImportReport",
    "GISSourceIdentity",
    "LegacyProjectCompatibility",
    "Line",
    "LineRole",
    "MetricDistance",
    "OFFLINE_RUNTIME_SCHEMA",
    "Project",
    "REGISTRATION_METADATA_KEY",
    "REGISTRATION_METADATA_SCHEMA",
    "RUNTIME_NETWORK_REQUIRED",
    "RegistrationResult",
    "RegistrationState",
    "ReviewState",
    "RiskAssessment",
    "RiskClass",
    "RiskPolicy",
    "Route",
    "RouteSeed",
    "STAGE_A_FILENAME",
    "Scan",
    "SharedSourceAccess",
    "SourceAccessMetrics",
    "SourceDescriptor",
    "Span",
    "Tower",
    "TowerSeed",
    "UnitStatus",
    "apply_registration_xy",
    "build_crs_transform",
    "generate_spans_from_ordered_towers",
    "generate_spans_from_route",
    "georeferenced_registration",
    "import_gis",
    "inspect_point_cloud",
    "load_project",
    "migrate_legacy_project",
    "no_gis_registration",
    "offline_proj_status",
    "open_project",
    "read_csv",
    "read_kml",
    "register_one_anchor",
    "register_two_anchors",
    "save_project",
    "save_scan_registration",
    "scan_registration_record",
    "to_metric_distance",
    "transform_xy",
    "transform_xyz",
    "validate_registration_checkpoint",
]
