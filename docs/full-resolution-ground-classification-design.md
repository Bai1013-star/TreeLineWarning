# Full-Resolution Ground Classification Producer Design

**Status:** DESIGN-FROZEN
**Design identifier:** FGC-D1
**Algorithm:** FGC-PMF-RLP-v1
**Authority:** reviewed/fixed-policy ground-classification authority only; not a statistically calibrated or production-certification claim.

## 1. Status, scope, and non-goals

This is the sole frozen design for the Full-Resolution Ground Classification
Producer (FGC). It is the design authority for the separate implementation
handoff only; no implementation is performed in this documentation closure.

FGC produces an immutable LAS derivative whose only permitted point-record
change is the base LAS classification. It is CPU-first, NumPy/laspy-only, and
does not consume A2B, P-HAG output, W1/P-WIRE, vegetation output, labels, or
scene-specific selectors. P-SOURCE and P-HAG semantic code remain unchanged.
The raw LAS is never opened for mutation.

The run authority is a formally validated source reference, not a plain path.
The V1 accepted source is the currently supported P-SOURCE LAS subset:
uncompressed LAS with finite XYZ, a uint8 logical classification dimension, no
EVLR, and no CRS authority rejected by P-SOURCE. LAZ and rejected metadata are
not silently converted; they fail closed in V1.

## 2. Authority model

Every source row in [0, N) receives exactly one uint8 terminal decision:

| Code | State | Publication effect |
|---:|---|---|
| 0 | NON_GROUND | preserve input base class except stale-class-2 normalization |
| 1 | CONFIRMED_GROUND | base classification becomes LAS class 2 |
| 2 | REVIEW_GROUND | preserve input base class except stale-class-2 normalization |
| 255 | UNROUTED | scratch-only sentinel; publication is forbidden |

Only CONFIRMED_GROUND may become class 2. Existing base class 2 is retained
only when its row is CONFIRMED_GROUND; every other existing base class 2 becomes
LAS class 1 (unclassified). All other base classes are preserved. This
normalization is necessary for the invariant that every output class 2 is
FGC-confirmed.

The decision artifact is one authoritative uint8 decision byte per original
source row. It is not coordinate-derived, crop-derived, or representative-only
authority.

## 3. Formal ingress and source lineage

The normative public boundary is:

    classify_ground(
        source_reference,
        source_result,
        frv_configuration,
        p_hag_compatibility_config,
        output_root,
        policy_config,
        *,
        progress=None,
        cancellation=None,
    )

source_reference is FrvSourceProviderReference, source_result is a finalized
FrvSourceProviderResult, frv_configuration is the matching formal
FrvConfiguration, and p_hag_compatibility_config is a validated
FrvHagProviderConfig. The function validates the existing P-SOURCE reference,
result, source identity, source count, original-index-domain digest,
classification pass-through dtype, and configuration bindings before any FGC
scratch is created. A path inside the reference is access information only,
never standalone authority.

FGC replays the classified derivative through unchanged P-SOURCE after writing.
It builds a new SourceIdentity for the classified LAS: its source file SHA,
canonical point-content digest, content identity, record identity, and source
ID are new. The classified source ID is:

    <raw-source-id>--fgc-v1--<first-16-hex-of-decision-digest>

The output provenance binds raw SourceIdentity, raw SHA, N, identity-preserving
row mapping, policy digest, decision digest, output SHA, and classified
SourceIdentity. Raw row i equals classified row i; the identities must never be
reused.

The P-HAG compatibility configuration is not an FGC numerical input. Its
digest, compatibility replay result, ground-cell inventory summary, and
P-SOURCE replay binding are recorded only in fgc_publication_proof.json.
Changing it may require a new proof but must not change the FGC policy digest,
decision bytes, decision digest, or classified LAS.

## 4. Exact coordinate-grid authority

Decoded floating-point floor division is forbidden for ownership, fine-grid
cells, coverage cells, and halo membership. FGC uses raw LAS integer X/Y/Z
coordinates and a validated rational header-scale policy.

For each horizontal header scale, form its canonical decimal rational by parsing
format(scale, ".17g") as a base-10 rational. For a physical policy length L,
L / scale_axis must be a positive integer. V1 preflight requires this for:

| Length | Required on X and Y |
|---|---|
| fine cell 0.5 m | yes |
| coverage cell 1.0 m | yes |
| tile side 256.0 m | yes |
| halo 20.0 m | yes |
| plane radius 3.0 m | yes |
| vertical radius 1.0 m | yes |

It also forms the greatest common rational horizontal metric tick g such that
scale_x/g and scale_y/g are integers, and requires 1.0 m/g to be an integer
whose squared value fits signed int64. Otherwise FGC fails with
UNSUPPORTED_COORDINATE_GRID. This makes every XY radius comparison an integer
comparison in metric ticks.

For raw rows p and q, define the exact metric-tick deltas:

    dx_g = (X_q - X_p) * (scale_x / g)
    dy_g = (Y_q - Y_p) * (scale_y / g)

All factors are validated integers. A radius R comparison is exactly
dx_g squared + dy_g squared <= (R/g) squared; equality is included. Fine-cell
centre comparisons use the same expression after multiplying both centre
coordinates and the radius by two, so half-cell centres never introduce a
floating-point tie.

Let X0 and Y0 be the reconstructed raw minima. Before the spool pass, obtain
the corresponding decoded minima from the finalized P-SOURCE proof, parse their
format(value, ".17g") strings under the same rational policy, subtract the
header offset, divide by scale, and require an exact int32 result. During the
one raw spool pass, calculate the actual minimum raw X/Y and require it to
equal the reconstructed X0/Y0; otherwise discard scratch and fail
SOURCE_GRID_ANCHOR_MISMATCH. This makes the source-min anchor available before
owner-spool routing without trusting a rounded floating floor. Let
Tx, Ty, Fx, Fy, Cx, Cy, Hx, Hy be the exact integer raw ticks for tile, fine
cell, coverage cell, and halo sizes. Tile counts are:

    nx = floor((Xmax - X0) / Tx) + 1
    ny = floor((Ymax - Y0) / Ty) + 1

For a row with raw coordinate X, its authoritative tile x index is:

    min((X - X0) // Tx, nx - 1)

and analogously for Y. Thus internal tiles own half-open raw intervals and the
source maximum is explicitly clamped into the final tile. This rule is valid
for negative decoded coordinates and arbitrary offsets because it operates on
raw-coordinate differences. Tile IDs are ordered (ty, tx).

Fine-grid and coverage indices are respectively:

    ix_f = (X - X0) // Fx;  iy_f = (Y - Y0) // Fy
    ix_c = (X - X0) // Cx;  iy_c = (Y - Y0) // Cy

All integer boundary comparisons are inclusive exactly where the formulas use
less-than-or-equal and otherwise half-open. No epsilon or decoded-coordinate
snapping is permitted.

## 5. Tiles, halo, and sufficiency proof

The core side is 256.0 m and the frozen halo is 20.0 m. A tile owns only its
core rows; halo rows are read-only context and never write decisions. The halo
rectangle is closed in raw ticks:

    [core_x_lo - Hx, core_x_hi + Hx] x
    [core_y_lo - Hy, core_y_hi + Hy]

clipped to source raw minima and maxima. A core point on an internal boundary
belongs only to the higher-index tile. A source-maximum point belongs only to
the final tile. Duplicate writes, ownership disagreement, out-of-range index,
or an UNROUTED byte causes TILE_OWNERSHIP_FAILED.

### HALO_SUFFICIENCY_PROOF

For PMF window w_m, define:

    k(w) = ceil(w_m / 0.5)
    n(w) = k(w) if k(w) is odd else k(w) + 1
    h(w) = (n(w) - 1) / 2

The square footprint offsets are all integer pairs (dx, dy) with
-h(w) <= dx <= h(w) and -h(w) <= dy <= h(w). The resulting footprints and
opening dependency radii are:

| w_m | n(w) cells | erosion half-width | opening dependency radius |
|---:|---:|---:|---:|
| 1 | 3 | 0.5 m | 1.0 m |
| 2 | 5 | 1.0 m | 2.0 m |
| 4 | 9 | 2.0 m | 4.0 m |
| 8 | 17 | 4.0 m | 8.0 m |
| 16 | 33 | 8.0 m | 16.0 m |

At cell q, dilation reads erosion values within 8 m; each of those erosion
values reads original-grid values within another 8 m. Therefore a 16 m opening
at q depends on original grid cells within 16 m of q in each axis. An evaluated
core point can use a seed representative within a 3.0 m Euclidean plane radius,
so its support may depend on raw-grid evidence up to 16 + 3 = 19.0 m away.
The source-point halo is point-based whereas fine cells have 0.5 m extent; a
full-cell allowance of 0.5 m gives 19.5 m. Vertical context is only 1.0 m and
is subsumed. The frozen 20.0 m halo is therefore sufficient without
operation truncation.

Only fine cells whose complete raw-tick cell rectangle is inside the halo
rectangle are eligible PMF work cells. Partially covered halo-edge cells are
nodata. This prevents a point just outside the loaded halo from silently
changing a boundary cell minimum.

## 6. Single-pass spatial owner-spool

Spatial access is a single-pass owner-tile spool, not one source replay per
tile. After formal P-SOURCE validation, FGC performs one sequential raw-LAS
spool pass, with SHA/size checked before and after against source_reference and
SourceIdentity. Each row's original_index is its contiguous source record
position and is written physically once to its owner-tile spool.

The scratch schema is FGCSP1. Each record is exactly little-endian:

    <qiii

meaning original_index int64 followed by raw X, Y, Z int32; record size is
20 bytes. Classification and Extra Bytes are not spooled because FGC numerics
do not use them and the writer later replays complete raw records. Records in
each spool are strictly ascending original_index because the source pass is
ascending. A spool manifest records schema, source identity, N, tile-layout
digest, every nonempty owner tile ID, row count, SHA-256, and record size. It
is scratch integrity evidence, not formal publication authority. Empty tiles
have no spool and are never enumerated merely because their coordinate range is
between two observed tiles.

For core tile T, FGC visits neighbor owner spools with offsets dx,dy in
{-1, 0, 1}. Halo is less than the tile side, so no further owner spool can
intersect it. Spools are visited in (dy, dx) order, records are streamed, and
only exact-halo rows are retained. Retained rows are then sorted by
original_index before grid construction. The core spool alone supplies rows
eligible for decision writes. If retained rows exceed max_halo_tile_points, or
the core count exceeds max_core_tile_points, FGC fails with RESOURCE_LIMIT_EXCEEDED.

Spool corruption, a missing declared nonempty spool, a record-size/schema mismatch,
non-ascending index, duplicate index, or spool SHA mismatch causes
SPATIAL_SPOOL_CORRUPT. Cancellation and every failure delete only the
same-volume transaction scratch, including all spools.

## 7. Exact sparse PMF and seed model

The source-min-anchored fine grid has 0.5 m cells. For every eligible work cell
c, G0(c) is the elevation of the lexicographically minimum representative
(raw Z, original_index) among halo rows in c; the representative source row is
retained. Because LAS Z scale is positive, this is exactly equivalent to
ordering by decoded z while avoiding a floating-point tie. Decoded elevation is
computed only as float64(raw_Z) * scale_z + offset_z after the ordering. Other
cells are explicit nodata.

For every window w, PMF operates on G0, never on a preceding scale's opened
surface. Let Fw be the footprint in Section 5 and let m = 5 be the frozen
minimum valid-cell support.

    E_w(c) is valid iff at least m cells q in c + Fw have valid G0(q).
    E_w(c) = min(G0(q)) over those valid q.

    O_w(c) is valid iff at least m cells q in c + Fw have valid E_w(q).
    O_w(c) = max(E_w(q)) over those valid q.

Cells outside the eligible work domain and all nodata cells are absent from
these sets; they are never assigned plus/minus infinity or filled terrain.
Boundary clipping is therefore the same valid-neighbor rule. Min and max ties
need no arbitrary row choice because only elevations are used; representative
ties remain (z, original_index).

The scale tolerance is:

    tau(w) = 0.10 + min(0.75 * w, 1.50) metres

A valid cell passes scale w iff:

    G0(c) - O_w(c) <= tau(w)

Equality passes. The progressive mask is:

    M_0(c) = valid G0(c)
    M_w(c) = M_previous(c) and valid O_w(c) and
             G0(c) - O_w(c) <= tau(w)

for w in ascending (1, 2, 4, 8, 16). A morphology seed candidate must pass all
five scales.

For seed local support, define D3 as the set of fine-cell offsets other than
(0,0) whose exact integer metric cell-centre distance is at most 3.0 m; thus
D3 excludes the centre. Define:

    D3_plus = D3 union {(0,0)}

For candidate cell c, construct the valid G0 cells in c + D3_plus and compute
the 8-connected component C containing c. Define the support-neighbour
population as:

    C_without_center = C \ {c}

The support requirement is |C_without_center| >= 5. From C_without_center
choose the five lowest keys (G0(q), qy, qx); their lower-envelope median is the
third elevation. c is excluded from the support count and all lower-envelope
statistics. c is an isolated-low candidate iff:

    lower_envelope_median - G0(c) > 0.30 m

Equality is retained. A seed is exactly a valid representative cell that passes
all PMF scales, has the required connected support, and is not isolated-low.
There is no additional undefined continuity or roughness predicate.

## 8. Local terrain plane, discontinuity, and vertical structure

For an evaluated core point p, gather seed representatives within exact
Euclidean XY radius 3.0 m. Order by (squared_distance, original_index), retain
at most 32, and require at least 12. Let R = 3.0 m and use:

    u_i = (x_i - x_p) / R
    v_i = (y_i - y_p) / R
    z_i = alpha*u_i + beta*v_i + c

Ordinary least squares is used; every retained support has weight 1. Let H have
rows (u_i, v_i, 1). The fit is valid only when all values are finite and the
eigenvalues of Q = H transpose H satisfy:

    lambda_max > 0 and lambda_min >= 1e-10 * lambda_max

No library-default rank decision is used. Solve Q theta = H transpose z.
Physical plane coefficients are a = alpha/R and b = beta/R:

    predicted_ground_z(p) = c
    residual(p) = z_p - predicted_ground_z(p)
    slope = sqrt(a*a + b*b)

Two synchronous clipping rounds are mandatory. Starting from the selected
supports S0, each round fits its current set, calculates every residual against
that one fit, and retains exactly supports with abs(residual) <= 0.20 m.
Equality is retained. If a fit is singular/nonfinite or a retained set has
fewer than 12 supports, the plane is unavailable and the evaluated row routes
to REVIEW_GROUND. After round two, fit the final retained population again.
Its roughness is the single defined statistic:

    RMS = sqrt(mean(final_support_residual squared))

RMS has no alternate interpretation.

Competing support is a bounded discontinuity predicate. Partition final retained
supports twice: X-negative versus X-nonnegative, and Y-negative versus
Y-nonnegative, relative to p. A partition is valid only when both halves have
at least six supports and each half passes the same finite/eigenvalue test using
ordinary least squares without further clipping. Compare the two predicted
intercepts at p. A valid partition is competing iff the absolute intercept
difference is at least 0.35 m. Any competing partition routes p to REVIEW_GROUND.
If neither partition is valid, competing support is false; it is not an
implicit failure or a second plane search.

Vertical structure is computed from a tile-local fine-cell hash of all
tile-plus-halo source rows. For p, inspect only cells whose raw-tick ranges can
intersect its 1.0 m circle, then retain other source rows q with exact integer
metric distance squared <= (1.0 m)^2. Self is excluded. If there are at least
eight other rows, form the span over p plus those retained rows:

    max(z) - min(z) >= 1.50 m

Equality triggers the predicate. Vertical structure routes p to REVIEW_GROUND.
The hash is created per loaded halo and discarded after the tile; no KDTree,
SciPy helper, or global neighbor structure is used.

## 9. Frozen routing precedence

Source integrity, coordinate-grid, spool, ownership, decision-store, and
publication failures are source-level hard failures; they leave no ordinary
per-point decision and block publication. For every otherwise valid core row,
apply this exact first-match order:

1. If vertical structure is true, write REVIEW_GROUND.
2. If no valid final plane exists, write REVIEW_GROUND.
3. If competing support is true, write REVIEW_GROUND.
4. If slope > 0.75 or RMS > 0.12 m, write REVIEW_GROUND.
5. If -0.15 m <= residual <= +0.20 m, write CONFIRMED_GROUND.
6. If residual > +0.60 m, write NON_GROUND.
7. Otherwise write REVIEW_GROUND.

The diagnostic review band is [-0.35 m, +0.60 m], inclusive. A residual below
-0.35 m is still REVIEW_GROUND, never inferred as non-ground; a residual in
the positive ambiguity interval (+0.20, +0.60] is REVIEW_GROUND. This ordering
has no optional positive-structure rule and no unqualified terms such as
adequate, likely, stable, or near-transition.

## 10. Global coverage and review publication gates

FGC-native coverage uses the 1.0 m source-min-anchored integer cells from
Section 4. An observed coverage cell contains at least one finite source row.
A confirmed occupied coverage cell is an observed cell containing at least one
CONFIRMED_GROUND row. The frozen occupancy fraction is:

    confirmed_occupied_cell_count / observed_cell_count

and must be at least 0.20. Equality passes.

V1 has no global review-fraction publication gate. Review fraction is recorded
as a diagnostic only and cannot optionally block publication.

An unsupported cell is an observed cell that is not confirmed occupied.
Unsupported cells form 4-connected components. A component is an edge component
when any of its cells has an unobserved north, south, east, or west neighbor; it
is interior otherwise. This treats the actual observed domain, including an
irregular boundary, as the exterior rather than using a bounding rectangle.
For an interior component, width is:

    max((max_ix - min_ix + 1), (max_iy - min_iy + 1)) * 1.0 m

Publication fails with GROUND_COVERAGE_GAP only when an interior component has
width strictly greater than 25.0 m. A 25.0 m component passes.

Coverage records use schema FGCCOV1 and fixed little-endian record <qqB:
coverage ix int64, iy int64, flags uint8; bit 0 means observed and bit 1 means
confirmed. Exactly one record is appended for each owned core source row, so
there is no tile duplication. Records are externally sorted by (iy, ix, flags)
in deterministic 64 MiB runs, then k-way merged and OR-deduplicated by (iy,ix).
The merge emits sorted unique sparse cells only. A row-streaming run-union
algorithm constructs 4-connected unsupported components: consecutive
unsupported cells become horizontal runs; a run joins only prior-row runs with
overlapping x intervals; active labels retain min/max ix/iy and exterior status.
Observed previous/current/next rows determine cardinal exterior adjacency.
Finalized component summaries are emitted when no active run remains. No dense
raster or coordinate-magnitude-sized allocation is permitted.

## 11. Reviewed fixed policy and resource bounds

GroundClassificationPolicyConfig is an immutable exact-schema
fgc-ground-policy-v1 object, hashed with canonical_frv_json_bytes under FGCPOL1.
It contains only FGC algorithm version, all values below, classification mapping,
decision layout, resource bounds, and schemas. It rejects unknown/missing
fields, nonfinite values, unordered sequences, paths, source identities,
timestamps, callables, and digest mismatch.

| Parameter | Frozen reviewed default |
|---|---:|
| tile / halo / fine cell / coverage cell | 256.0 / 20.0 / 0.5 / 1.0 m |
| PMF windows / minimum valid cells | (1,2,4,8,16) m / 5 |
| seed radius / connected support / lower envelope | 3.0 m / 5 cells / 5 cells |
| isolated-low / PMF tolerance | 0.30 m / 0.10 + min(0.75w,1.50) m |
| plane radius / min-max supports | 3.0 m / 12-32 |
| clipping / singular ratio / slope / RMS | 0.20 m x 2 / 1e-10 / 0.75 / 0.12 m |
| competing halves / disagreement | 6 supports per half / 0.35 m |
| vertical rule | 8 other neighbours, 1.0 m, 1.50 m span |
| residual bands | confirm [-0.15,+0.20], review [-0.35,+0.60] m |
| coverage occupancy / hole width | 20% / interior width >25.0 m fails |
| source / core / halo point caps | 50,000,000 / 8,000,000 / 8,000,000 |
| halo grid cells / plane batch | 500,000 / 65,536 |
| decision memmap / coverage cells | 64 MiB / 50,000,000 |

The 16 GB working-set bound is:

    256 * max_halo_tile_points
    + 128 * max_halo_grid_cells
    + 128 * max_plane_batch
    + 96 MiB source/chunk and routing workspace
    + 256 MiB fixed safety reserve

With the reviewed per-tile point caps (8,000,000 core and 8,000,000 halo rows)
this is about 2.4 GiB for the tile algorithm, before the process baseline. The
estimated single-tile peak for scan_small (6,905,520 rows, ~47 m x ~39 m, one
256 m tile) is well below 1.5 GiB, far inside the 16 GB ceiling; the real-scene
runtime records the observed peak. FGC uses plane batches no larger than
max_plane_batch and must not allocate a core-points by 32 candidate matrix. The
result is CONDITIONAL_WITHIN_POLICY_LIMITS: a pathological tile above a cap
fails closed rather than risking a 16 GB exhaustion. scan_small with 6,905,520
rows is feasible within these checked limits; runtime is an order-of-magnitude
expectation, not a benchmark promise.

Same-volume free-space preflight requires at least:

    ceil(1.15 * (
        2 * input_file_bytes
        + 20 * N                         # owner spools
        + 2 * N                          # routing memmap plus final chunks
        + 51 * N                         # FGCCOV1 external-sort worst case
        + 128 MiB                        # manifests, bounded diagnostics, margin
    ))

bytes. The formula covers temporary classified output/finalization, spool,
routing, decision chunks, sparse coverage sorting, proof/manifest transaction,
and a 15% safety margin. Insufficient space is SPATIAL_SPOOL_RESOURCE_FAILED.

## 12. Decision storage, artifacts, and completion authority

Decision chunks are raw uint8 files over ordered inclusive-exclusive contiguous
original-index ranges, default one million rows per chunk. Each chunk contains
only ascending decision bytes. The decision manifest records source identity,
raw SHA, N, policy digest, algorithm ID, ordered ranges, per-range SHA/size and
state counts, total counts, tile summaries, coverage metrics, and decision
digest. Its digest is over the source identity, policy digest, algorithm ID,
N, and the ordered complete chunk records. Moving a chunk, changing a boundary,
changing source/policy, reordering chunks, or changing bytes is detectable.

The transaction layout is:

    ground-classification/<raw-source-id>/<raw-sha-prefix>/
      decisions/
      fgc_decision_manifest.json
      fgc_publication_proof.json
      classified/<source-stem>.fgc-v1.las
      fgc_completion_manifest.json
      diagnostics/                 # non-authoritative, bounded

fgc_completion_manifest.json is the only discoverable formal root. Its exact
schema is fgc-completion-manifest-v1. It binds safe relative paths, SHA-256,
sizes, schema versions, policy identity/schema, raw and classified
SourceIdentity, algorithm ID, decision-manifest identity, publication-proof
identity, classified LAS, and its own completion digest. Optional Human Gate
HTML is never a formal component. The completion digest is SHA-256 of the
canonical completion payload with the completion_digest field omitted; the
stored field must equal that computed digest.

Consumers must: load the completion manifest; exact-schema validate; reject
unsafe paths; hash/size-check components; validate the decision manifest and
all chunks; validate the proof; validate classified SourceIdentity; then verify
all cross-digest bindings. Only then is the publication
FORMAL_FGC_PUBLICATION_VALID. Without a valid completion manifest the directory
is INCOMPLETE_NONAUTHORITATIVE, even if LAS, proof, and decision manifest exist.

All components are created and validated in one same-volume transaction
directory. Immutable chunks and classified LAS are published first, then proof
and decision manifest, and the completion manifest is atomically renamed last.
No placeholder completion manifest is written. If the target namespace already
contains a valid completion manifest, or any collision, FGC fails with
PUBLICATION_COLLISION rather than overwriting. A failure before the final rename
leaves only removable non-authoritative scratch.

## 13. Classification-only LAS preservation

The writer streams raw LAS records and aligned decision bytes in source order.
It copies each complete record then sets only laspy's logical base
classification accessor; it never assigns packed classification storage.
This preserves synthetic, key_point, withheld, overlap, scanner channel,
scan-direction, edge-of-flight-line, and other applicable flag fields.

The lockstep verifier checks source/derivative point count and record order,
raw integer XYZ, point format, scales/offsets, all non-classification standard
and Extra Bytes dimensions, every applicable classification flag, expected
base classification, and output SHA. It verifies semantic header metadata
rather than whole-file byte identity.

V1 preserves the ordered P-SOURCE-accepted non-CRS VLR sequence by canonical
comparison of VLR type, user ID, record ID, description, and raw payload bytes.
EVLR is unsupported because current P-SOURCE rejects it. Any VLR type,
waveform/external-reference feature, or backend behavior that cannot round-trip
under this comparison fails with UNSUPPORTED_LAS_METADATA or
PUBLICATION_PRESERVATION_FAILED; it is never dropped silently.

## 14. P-HAG, scene_simple, Human Gate, and deferred work

P-HAG code change required: NO. Its existing code consumes a P-SOURCE uint8
classification attribute and selects exactly class 2 to form a sparse ground
grid. The FGC proof records a compatibility replay with the supplied
p_hag_compatibility_config; it does not use P-HAG to decide ground.

After successful FGC publication, scene_simple is regenerated from the
classified parent. Its index map values are classified-parent original_index,
not crop-local row indices. The regenerated sample manifest binds classified
parent SourceIdentity, parent source SHA and identity-record digest, crop
bounds, index-map SHA and parent-index mapping digest, sample LAS SHA, and
sample point count. Existing sample artifacts are not relabelled.

The Ground Human Gate remains external. Machine diagnostics are deterministic,
bounded, and non-authoritative. Owner outcomes are
PASS_READY_FOR_FRV_RUNTIME, PASS_WITH_REVIEW_NOTES,
GROUND_POLICY_ADJUSTMENT_REQUIRED, and DATA_OR_ARTIFACT_BLOCKED. Runtime never
creates a human approval artifact.

Deferred work remains LAZ qualification, parallel execution, Algorithm-V2,
manual editing, P-WIRE publication, V2 provenance repair, and statistical
calibration/production certification.

## 15. Normative algorithm order

The following order is normative:

1. Validate formal source_reference, source_result, frv_configuration, policy,
   P-HAG compatibility config, source limits, raw-grid commensurability, and
   transaction free space.
2. Perform the one sequential owner-spool pass and validate its manifest.
3. Enumerate only nonempty source-min-anchored owner tile IDs from the validated
   spool manifest, in ascending (ty,tx) order.
4. For each tile, stream adjacent owner spools, filter exact halo rows, enforce
   caps, and build the local vertical hash and eligible sparse minimum grid.
5. Compute G0, every exact PMF opening from G0, the progressive mask, and seeds.
6. For each core row, apply the exact local-plane, competing-support,
   vertical-structure, and routing precedence; write once at original_index.
7. Validate decision conservation and publish ordered uint8 decision chunks.
8. Build sparse global coverage records, validate occupancy and interior holes,
   and fail before publication on coverage failure.
9. Create the classification-only derivative in the transaction, verify
   preservation, derive classified SourceIdentity, and replay it through
   P-SOURCE.
10. Run the P-HAG compatibility replay, write the decision manifest and proof,
    validate all cross-bindings, then atomically publish the completion manifest
    last.

## 16. Failure semantics and tests

The exact source-level fail-closed failures include
UNSUPPORTED_COORDINATE_GRID, NONFINITE_SOURCE, SPATIAL_SPOOL_RESOURCE_FAILED,
SPATIAL_SPOOL_CORRUPT, RESOURCE_LIMIT_EXCEEDED, GROUND_COVERAGE_GAP,
DECISION_CONSERVATION_FAILED, TILE_OWNERSHIP_FAILED,
PUBLICATION_PRESERVATION_FAILED, UNSUPPORTED_LAS_METADATA,
CLASSIFIED_REPLAY_MISMATCH, COMPLETION_MANIFEST_FINALIZATION_FAILED, and
PUBLICATION_COLLISION. PMF_UNRESOLVED_SUPPORT, INSUFFICIENT_SEED_SUPPORT, and
PLANE_DEGENERATE are exact per-row/tile diagnostic reasons; their affected rows
route REVIEW_GROUND under Section 9 rather than converting a local support hole
into a global failure. Source-level failure precedence is validation, spool,
tile/decision, coverage, writer, replay, then completion finalization. No
source-level failure is converted into ordinary class 2/1/0 output.

Required tests include exact PMF footprints/nodata/recurrence/equality; halo
proof; raw-coordinate boundaries including negative and maximum coordinates;
spool order/resource/cancellation; seed median and isolated-low behavior; plane
fits/ties/clipping/degeneracy/RMS/discontinuity; vertical local-grid routing;
coverage and component definitions; completion-manifest discovery; P-HAG digest
decoupling; flags/VLR preservation; formal ingress; and sample parent lineage.

## 17. Independent-review correction matrix

| Finding | Resolution location | Frozen correction |
|---|---|---|
| FGC-IR-001 | §§4–5 | Raw integer authority; 20 m proven halo |
| FGC-IR-002 | §12 | Explicit completion root, discovery, finalization |
| FGC-IR-003 | §7 | Exact sparse PMF recurrence and nodata semantics |
| FGC-IR-004 | §§7–9 | Exact seed, plane, clipping, roughness, vertical, routing rules |
| FGC-IR-005 | §10 | Native coverage, hole, and review-gate rules |
| FGC-IR-006 | §§6,11 | Single-pass owner spool and complete disk/resource model |
| FGC-IR-007 | §4 | Rational/raw integer ownership and cell authority |
| FGC-IR-008 | §§3,14 | P-HAG compatibility proof decoupled from policy |
| FGC-IR-009 | §13 | Logical class write and flag/VLR verification |
| FGC-IR-010 | §§3,14 | Formal ingress and bound scene_simple lineage |
| FGC-IR-011 | §16 | Explicit grouped authority-test requirements |
| FGC-RR-001 | §7 | D3-plus component domain; centre-excluded support and lower envelope |

**Review result:** PASS — FGC-RR-001 corrected; all review findings closed.
