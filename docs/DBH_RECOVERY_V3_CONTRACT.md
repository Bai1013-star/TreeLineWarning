# V3 crown-to-stem identity policy

Baseline b849c3feb56c26ac1406989836e975e2035b48d9; exact accepted project
E:/2026.8.31-2/2026.08.31-2/2026.08.31-2. Preserve all six numeric rows and
their exact evidence. Only unavailable identities may gain a DBH sidecar.

This policy is fixed before evaluating project candidate scores. Scores are
engineering evidence rankings, not calibrated probabilities or field accuracy.
No Ground, HAG, Wire, crown membership, clearance, risk or CONTACT changes.

## Search and tracking

Reuse the bounded V2 ROIs and editor exclusion. No radius expansion. Start with
all distinct V2 axes, not just its best pair fit. For a no-axis ROI, propose an
axis independently from upper sections around HAG 2–3 m, using current crown
support, then test it downward. Circle proposals use existing robust section
fitting and V1 geometry bounds. Up to two proposals per section and bounded
section-pair hypotheses; no historical owner supplies identity.

Follow candidate structure downward in 0.25 m HAG steps, inherited from the V2
raw-to-current bridge, from 3.30 to 1.05 m. Each diagnostic section has the V1
0.15 m thickness and is normal to the candidate axis. Its search tube uses the
V2 radius allowance (candidate radius + 0.08 m). Track circle centers and
require existing minimum points, angular coverage/sectors, radius/residual
limits, and center prediction correction no greater than 0.03 m. After an
accepted section, propagate its measured normal-plane center to the next layer;
do not keep comparing to an unmeasured fixed lower extrapolation. The full
measured center track must still fit within 0.03 m and 20 degrees. This mask is identity evidence
only: it is never substituted for the final DBH quality test.

Absolute identity requires an uninterrupted track from at least 2.80 m through
1.30 m to 1.05 m, the V2 raw bridge and current-member connection (three
consecutive 0.50 m bins), and intersection with the current crown footprint.
This demands upper support and actual breast-height continuity. Tiny surviving
crown fragments cannot inherit historical stems. Direction continuity is
reported through consecutive center links and checked against the existing
20-degree tilt bound; residual limits remain 0.03 m.

Footprint comes from occupied current-member XY cells at V2's 0.35 m resolution;
centroid is the mean occupied-cell center, apex is the median XY of the top
5% HAG points. Record distances to both, upper support and all neighboring
crowns. Footprint intersection allows only the existing 0.60 m search tolerance;
centroid/apex distances alone cannot establish ownership.

## Deterministic score and ownership

Score in [0,1]: 0.40 tracked-layer coverage + 0.25 center-track consistency
+ 0.20 cylindrical inlier support + 0.10 footprint consistency + 0.05 joint
centroid/apex consistency. The absolute minimum is 0.75; the minimum margin
over the next structurally qualified candidate is 0.10, even when that runner
is just below the winner's 0.75 score floor. A ten-percentage-point
margin prevents a small auxiliary spatial feature alone from settling a tie.
Both gates apply before DBH fitting; do not fall through to a lower-ranked
candidate merely because the preferred stem has an unmeasurable DBH.

Evaluate each candidate against all twelve current crowns. Build a symmetric
cross-tree conflict graph before assignment. Shared exact source indices or
equivalent axes (existing 0.15 m / 5-degree equivalence) conflict. Every
absolute-qualified claim participates, including competitors not eventually
selected. Existing six accepted stems reserve their evidence and cannot be
reassigned. Any cross-tree ambiguity fails closed; no per-tree greedy claiming.

Only an identity selected by these gates reaches final DBH fitting. Rotate the
full original ROI to the selected axis and call the unchanged V1 fitter; its
independent V1 continuity fallback is allowed. A resulting fitted center must
remain on the selected axis within 0.03 m. Preserve all V1/V2 fit guards and
true-HAG 1.30 anchoring. Never use the winning circle's inlier mask to inflate
final quality. Final evidence again excludes deleted/other-owner indices and
must be globally unique before publication.

Below ten reliable point-cloud DBHs, estimates remain zero. No rebuild or push.
Focused tests, V1/V2 regressions, exact project/restart/API/export checks and
parent review precede a single scoped V3 commit.
