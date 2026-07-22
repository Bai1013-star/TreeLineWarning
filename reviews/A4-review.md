# A4 Review: Separation, Confirmation, and Track/Wire Semantics

## Scope and baseline

- Branch: `feature/algorithm`
- Baseline commit: `1918c0f feat: add rule and manual wire candidate baseline`
- This review covers A4 and the approved A4-FIX only.
- No A1/A2A/A2B/A3 behavior or parameters were changed.
- A4 read only the approved A3 candidate NPZ and JSON. No LAS/LAZ input was read.
- No commit, push, merge, dependency change, or A5 work was performed.

## Visual issue and correction

The original preview rendered confirmed wires, provisional tracklets, and review-required
tracklets with equally prominent multi-color styling. This made short canopy, branch, and
near-ground tracklets appear to be formal wires.

The output now separates the concepts:

- `track_id_by_point`: every assigned track point, including provisional and review tracks;
  unassigned is `-1`.
- `wire_id_by_point`: confirmed wires only; every temporary or unassigned point is `-1`.
- Track IDs are continuous after deterministic all-track sorting.
- Wire IDs are continuous after deterministic confirmed-track sorting by `median_u`, descending
  `median_v`, `first_s`, then minimum `original_index`.

The preview now shows low-opacity background and one normal-visibility trace per confirmed
wire. Review tracklets, provisional tracklets, and unassigned candidates are legend-only by
default. Temporary tracks use neutral colors rather than confirmed-wire colors. Hover includes
track ID, wire ID, status, assignment score, span, and axis angle.

## Contract and algorithm changes

The public contract now stores both point-level ID arrays, all-track offsets
(`ordered_track_point_indices`, `track_offsets`) and confirmed-wire offsets
(`ordered_wire_point_indices`, `wire_offsets`). `WireTrackSummary` now includes track ID,
optional wire ID, high/medium/low fractions, actual PCA direction, axis angle, and confirmation
channel. JSON/NPZ metadata records these arrays and confirmation transitions.

Actual track direction is PCA-derived from each track's XYZ points, unit normalized, and sign
aligned to the global `s_axis`. Degenerate directions use a zero sentinel with
`NUMERIC_INVALID`, and cannot be confirmed. The prior incorrect behavior of writing the global
axis as every track's `main_direction` was removed.

Two explicit `WireSeparationConfig` confirmation channels are applied after association:

1. Obvious: span >= 10.0 m, coverage >= 0.80, direction consistency >= 0.99, residual <=
   0.04 m, curvature <= 0.45 rad, gap <= 0.20 m, zero branch conflicts, axis angle <= 2.0 deg,
   mean candidate score >= 0.90, and high fraction >= 0.95.
2. Fine: span >= 6.0 m, coverage >= 0.55, direction consistency >= 0.99, residual <= 0.04 m,
   curvature <= 1.05 rad, gap <= 0.20 m, zero branch conflicts, axis angle <= 2.0 deg, mean
   candidate score >= 0.63, and medium-plus-low fraction >= 0.40.

The fine channel deliberately has no high-fraction minimum. Fine conductors can be dominated by
medium/low evidence, but still require long, continuous, smooth, near-axis geometry. Tracks that
fail both channels remain available for review through `track_id_by_point`; no track points are
deleted.

## Full real-input validation

Input A3 artifact:

- Point count: 146,126
- Candidate count: 17,784; high/medium/low: 8,226 / 3,126 / 6,432
- A3 NPZ SHA-256: `7bb56ed4f40c2409ef8d25b72c4e59a0227480db95546975154930288987dfac`

Output validation:

- A4 NPZ SHA-256: `0ec78d6da68d4ee53da1256c57dcca83eebb081a38a1e1671c25058519a8ed04`
- A3 input hash and A4 output hash match their JSON metadata.
- A3/A4 `original_index`, `sample_index`, and `xyz_local` are aligned exactly.
- Total tracks: 344; confirmed wires: 8; provisional tracks: 191; review-required tracks: 145.
- Assigned track points: 14,940; assigned wire points: 7,396; temporary-only track points: 7,544.
- Unassigned candidates: 2,844; duplicate track assignments: 0; duplicate wire assignments: 0.
- Track IDs and confirmed wire IDs are continuous. Track/wire offsets span their respective
  flattened arrays exactly.
- Confirmation transition: pre-fix legacy confirmation 9, post-fix confirmed wires 8,
  demoted confirmed-to-temporary 1, promoted temporary-to-confirmed 0.

The formerly confirmed approximately 3.83 m track does not meet the approved 6.0 m fine-wire
span requirement and now remains temporary. The approximately 13.46 m medium/low-dominant track
passes the fine channel based on geometry and continuity, not a hard-coded track ID.

## Confirmed wire summaries

| wire | track | channel | points | span m | coverage | score | high/medium/low | axis angle deg |
|---:|---:|---|---:|---:|---:|---:|---|---:|
| 0 | 50 | obvious | 834 | 14.21 | 1.000 | .973 | .999/.000/.001 | .32 |
| 1 | 51 | obvious | 943 | 14.42 | 1.000 | .964 | .980/.020/.000 | .67 |
| 2 | 57 | obvious | 962 | 15.28 | 1.000 | .982 | .997/.003/.000 | .63 |
| 3 | 64 | obvious | 957 | 15.71 | 1.000 | .990 | 1.000/.000/.000 | .14 |
| 4 | 69 | fine | 690 | 13.46 | .822 | .684 | .206/.383/.412 | .57 |
| 5 | 78 | obvious | 1,035 | 16.04 | 1.000 | .990 | 1.000/.000/.000 | .50 |
| 6 | 92 | obvious | 997 | 15.83 | 1.000 | .955 | .983/.012/.005 | .68 |
| 7 | 94 | obvious | 978 | 15.75 | 1.000 | .960 | .983/.017/.000 | .19 |

All confirmed tracks have zero branch conflicts. The fine wire has direction consistency .995,
residual .029 m, curvature .829 rad, and maximum gap .20 m; it satisfies the fine channel.

## Performance and safeguards

- Runtime: 2.323 s
- RSS start / peak / end: 74,862,592 / 103,329,792 / 103,329,792 bytes
- No global NxN distance matrix is constructed; clustering uses local `cKDTree.query_pairs` and
  association uses per-slice Hungarian cost matrices.
- The A3 artifacts were unchanged. No LAS/LAZ was read.
- Generated outputs are ignored by Git. The preview exists and contains three legend-only
  temporary/unassigned layers.

## Tests and quality checks

Focused A4 tests cover contract validation, W=0 offsets, all-track versus confirmed-wire IDs,
obvious and fine threshold boundaries, short-track rejection, actual PCA directions and sign,
numeric-invalid direction handling, manual exclusion, preview default visibility, stable IDs after
input row reordering, cancellation, output protection, no LAS access, and no global distance
matrix API.

Final quality results:

```text
153 passed in 4.65s
ruff check .: passed
compileall algorithms core tests scripts: passed
ruff format --check [A4 files]: 6 files already formatted
git diff --check: passed
```

## Recommendation

The code and data invariants are ready for GPT review. Do not begin A5 until the updated preview
is visually approved. A4 should be committed only after that review confirms the default
confirmed-only rendering and the eight final wires are acceptable.
