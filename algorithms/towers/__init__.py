"""V5A2/V5A3 tower automatic detection, RAW access, and bounded refinement.

The tower-candidate review pipeline builds on the frozen V5A1-B project
services.  ``automatic`` adapts the existing pole detector into a deterministic
``TowerCandidateSet``; ``raw_access`` provides a derived, source-bound spatial
tile cache over the protected full-resolution LAS; ``refinement`` performs the
bounded local RAW tower refinement that produces the defensible
``TowerGeometry`` required by an accepted review; ``review_workflow``
orchestrates one-click ACCEPT / REJECT / ADJUST / MANUAL_ADD through the V5A1-B
project services.

Scientific authority is always FULL_RESOLUTION_RAW; the Giro3D viewer is
interaction-only and never contributes detection/refinement geometry.
"""

from __future__ import annotations

__all__: list[str] = []
