"""Stage-B product integration around the validated science contracts.

Stage B intentionally keeps :mod:`core.stage_a` as the product aggregate and
the existing ``algorithms.spans`` contracts as the science boundary.  The
objects exported here are adapters, batch lifecycle records, and audit
helpers; they are not a second Project/Line/Tower/Span persistence model.
"""

from core.stage_b.batch import (
    BatchAnalysisResult,
    BatchAnalysisScheduler,
    BatchSpanStatus,
    BatchTelemetry,
)
from core.stage_b.bridge import (
    BridgeIdentity,
    ProjectScienceBridge,
    SpanScienceBinding,
    bridge_project_to_science,
)
from core.stage_b.contact import (
    ContactEvidence,
    ContactState,
    contact_evidence_from_clearance,
)
from core.stage_b.editor import BatchTreeEditor, BatchTreeEditorError
from core.stage_b.export import export_batch_prediction
from core.stage_b.ground import (
    GroundPreparation,
    ensure_ground_preparation,
)
from core.stage_b.ui import render_batch_dashboard

__all__ = [
    "BatchAnalysisResult",
    "BatchAnalysisScheduler",
    "BatchSpanStatus",
    "BatchTelemetry",
    "BatchTreeEditor",
    "BatchTreeEditorError",
    "ContactEvidence",
    "ContactState",
    "contact_evidence_from_clearance",
    "BridgeIdentity",
    "GroundPreparation",
    "ProjectScienceBridge",
    "SpanScienceBinding",
    "bridge_project_to_science",
    "ensure_ground_preparation",
    "export_batch_prediction",
    "render_batch_dashboard",
]
