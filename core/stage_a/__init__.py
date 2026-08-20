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
    georeferenced_registration,
    no_gis_registration,
    register_one_anchor,
    register_two_anchors,
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
    "AnalysisRun",
    "AnalysisRunStatus",
    "ALLOWED_BASEMAP_POLICIES",
    "AnchorCorrespondence",
    "CacheState",
    "CanonicalGISImport",
    "CapabilityReport",
    "CapabilityState",
    "CoordinateDeclaration",
    "GISImportReport",
    "GISSourceIdentity",
    "Line",
    "LineRole",
    "LegacyProjectCompatibility",
    "MetricDistance",
    "OFFLINE_RUNTIME_SCHEMA",
    "Project",
    "Route",
    "RegistrationResult",
    "RegistrationState",
    "RiskAssessment",
    "RiskClass",
    "RiskPolicy",
    "RUNTIME_NETWORK_REQUIRED",
    "ReviewState",
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
    "georeferenced_registration",
    "import_gis",
    "inspect_point_cloud",
    "load_project",
    "migrate_legacy_project",
    "no_gis_registration",
    "open_project",
    "read_csv",
    "read_kml",
    "register_one_anchor",
    "register_two_anchors",
    "save_project",
    "generate_spans_from_ordered_towers",
    "generate_spans_from_route",
    "to_metric_distance",
]
