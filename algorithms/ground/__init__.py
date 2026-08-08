"""Narrow public exports for the Full-Resolution Ground Classification producer.

Bundle A publishes the routing store and coverage helpers; the classifier and
publication orchestration land in Bundle B/C from this same namespace.
"""

from algorithms.ground.decision_store import (
    CoverageRecordAccumulator,
    DecisionConservationError,
    DecisionStore,
    compute_coverage_result,
    merge_coverage_cells,
    run_union_components,
)

__all__ = [
    "CoverageRecordAccumulator",
    "DecisionConservationError",
    "DecisionStore",
    "compute_coverage_result",
    "merge_coverage_cells",
    "run_union_components",
]
