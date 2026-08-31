# DBH Recovery V4: human stem identity review

V4 adds an explicit review to the accepted V3 checkpoint
`09e07a56ef75274abf5a179080db2bfcf9f2a1dc`. It does not run another automatic
recovery pass, edit final crown members, or change Ground/HAG/Wire/clearance/risk.
The existing seven numeric DBHs are locked. No allometric estimate is introduced.

## Human and scientific responsibilities

Only an unavailable, current finalized tree can enter review. The operator can
choose a persisted candidate axis or click a displayed raw point. Confirmation
requires an explicit identity checkbox and submit action. Requests contain the
session ID and exactly one candidate ID or integer raw `original_index`; diameter,
XYZ, radius and arbitrary additional fields are rejected. `REVIEW_CONFIRMED_STEM`
means **human-confirmed identity, point-cloud-computed diameter**.

Preparation displays current crown context and low-stem raw points at HAG
0.5–5.5 m. It uses the existing maximum three current-tree ROIs, each with the
existing 0.60 m horizontal search bound, and caps the union at two million points.
It never expands the ROI to obtain a successful diameter. Display is at most
80,000 deterministic sampled raw rows and 5,000 current crown context rows;
indices/HAG/XYZ remain aligned. Context points cannot become seeds. The full
local raw arrays, not the visualization LOD, feed the fitter.

For candidate confirmation, the existing axis and its radius plus the established
0.08 m association allowance define the local tube. A seed defines the existing
0.60 m neighboring search. Both use unchanged V2 axis proposals, true-HAG 1.30
interpolation and normal-plane transformation, then unchanged V1 robust circles
and continuity fallback. Competing local axes remain unavailable; fitted axes
must match the confirmed candidate, or pass through the selected seed's stem
surface within the established association allowance.

Human confirmation replaces automatic **identity** inference, including V3's
upper-crown track/score selection. It cannot override fit quality. V4 additionally
requires the two adjacent normal-plane sections straddling the measured BH anchor
to pass: a gap at BH cannot be justified by distant accepted sections. This is a
review safety gate, not a change to the frozen V1/V2/V3 algorithms or their thresholds.
Original XYZ/HAG are never replaced by temporary fitting coordinates.

## Ownership, revision and persistence

Preparation stores an immutable `dbh_review_v4/<uuid>` session, raw NPZ and hashes
under the current run. The session binds project/source/preparation/Ground, current
final membership and identities, editor journal revision, and current DBH geometry
generation. It records every existing V3 candidate's score, axis, support and
continuity diagnostics, with no automatic human choice.

Existing V2 DELETE, lasso REMOVE_POINTS, MERGE and UNDO source-index exclusions
apply before display and again at confirmation. A displayed point owned by another
current tree or published DBH is not selectable. Final fitting evidence cannot
belong to another current tree, duplicate another DBH source index, or claim an
already accepted physical stem. Its BH anchor must lie within the current crown's
existing footprint allowance. Human identity cannot bypass those vetoes.

The same desktop editor task coordinator serializes preparation/confirmation with
editing and project changes. Fit and publication recheck current binding and DBH
generation. Any edit, merge, undo, new finalization or other DBH publication makes
an old session stale. An invalid/stale published DBH proof reads as unavailable.
There is no silent reuse of historical confirmation after editing.

Each submitted session has a separate immutable decision record, with persistent
tree ID, editor revision, timestamp, human choice, exact selected source indices
and complete geometry result. A failed fit records the attempt but does not
replace the current DBH pointer. Success publishes a new additive DBH generation
through the existing writer, preserving every other row and evidence array.
Review decisions and artifacts are hash-bound and checked on restart/publication.
Reports remain read-only. Both Survey and Prediction_Input carry the numeric
geometry result and `REVIEW_CONFIRMED_STEM` provenance.

## UI and boundary

Tree Review → **胸径复核 / DBH Review** opens an in-application local review page.
Only unavailable trees are selectable. All candidate axes are distinct, selection
is highlighted, and **点击树干** arms exact raw-point picking. No diameter input
exists. Rendering failure disables scientific confirmation; failed geometry shows
“该树干点云不足，无法可靠计算胸径”. The user can reopen a fresh review after an edit.
Long operations use existing asynchronous editor-task status and bounded polling;
an uncertain submission is never silently resubmitted.

No package rebuild or independent-PC claim is part of V4. Synthetic fixtures may
confirm stems. Real acceptance may only prepare and inspect the five review views;
the user must make the scientific identity choices before any new real DBH is fitted.
