# A2B Review: Grid Ground Estimation and HAG Baseline

## Scope and baseline

- Branch: `feature/algorithm`.
- Baseline HEAD: `9e3a304 feat: add local PCA geometry features`.
- Scope is A2B only: deterministic grid ground estimation and height-above-ground (HAG) from existing A1/A2A artifacts.
- No branch/worktree, dependency change, LAS/LAZ point-record read, A1/A2A rerun, data edit, preview HTML, commit, push, merge, A3, or other business algorithm was performed.

## Modified files

- `core/contracts/ground.py`
- `core/contracts/__init__.py`
- `algorithms/features/ground.py`
- `algorithms/features/__init__.py`
- `tests/contracts/test_ground_contracts.py`
- `tests/features/test_ground.py`
- `reviews/A2B-review.md`

## Public contracts

`GroundEstimationConfig` provides the reviewed defaults. `GroundSurfaceModel` stores grid origin, size, shape, per-cell point counts, ground heights, control mask, cell flags, and direct-control coordinates/heights. `HeightAboveGroundSet` preserves source id, manifest id, A1/A2A `original_index`/`sample_index`, ground elevation, raw and clamped HAG, deterministic grid id, support distance, and point flags. `GroundEstimationStatistics` records grid/point counts, quality counters, elapsed time, and RSS. `GroundEstimationResult` records all A1/A2A paths and hashes, output checksum, arrays, formulas, distributions, provenance, and classification-validation status.

`GroundQualityFlag` is stable `uint16`: `NONE=0`, `SPARSE_GROUND_CELL=1`, `EMPTY_GROUND_CELL=2`, `INTERPOLATED_GROUND=4`, `EXTRAPOLATION_TOO_FAR=8`, `HIGH_LOCAL_GROUND_VARIATION=16`, `BELOW_GROUND_TOLERANCE=32`, `EXTREME_HEIGHT_AGL=64`, `INVALID_GROUND_ESTIMATE=128`, and `SOURCE_CLASSIFICATION_DISAGREES=256`.

## Input verification

- A1/A2A point count: `146126`.
- `xyz_local` is `(146126, 3) float32`; `origin_xyz` is `(3,) float64`.
- A1/A2A `xyz_local`, origin, `original_index`, and `sample_index` are element-wise identical. Both index arrays are non-negative and unique.
- All required numeric inputs are finite. A2A `NUMERIC_INVALID` count is zero.
- A1/A2A source id and manifest source id are consistent.
- A1 input SHA-256 values: NPZ `7b12e8c950672a61e3a916a51820bf793a7301b12ace82593a9a660bf94d5d02`; JSON `95d0cbebcf4cd2a1791e0f2ecef7c08d4552ab99c214496934296a0781a63b74`.
- A2A NPZ SHA-256 is `8c13802e8aafa83a5e0c0f04318c40d1ac3cff8cb4e7d98713c63654d11a74d4`; its JSON declaration matches. The A2A JSON SHA-256 consumed by A2B is `d21062adcb92abc0a15bce56e409f42c83369373f3661798f4e46a19eb76ebe0`.
- A1/A2A input hashes were identical before and after the A2B run.
- A1 `classification` exists but all 146,126 values are `0`; no source point has ground code `2`. Classification validation is explicitly recorded as not applicable, not treated as ground truth.

## Grid, quantile, and IDW definition

- `grid_origin_xy = floor(min_xy / grid_size_m) * grid_size_m`.
- `ix=floor((x-origin_x)/grid_size_m)`, `iy=floor((y-origin_y)/grid_size_m)`, and `ground_cell_id=iy*grid_width+ix`.
- Default configuration: `grid_size_m=0.5`, `ground_quantile=0.05`, `quantile_method=linear`, `min_points_per_cell=32`, `idw_neighbors=4`, `idw_power=2.0`, `interpolation_max_distance_m=1.5`, `smoothing_radius_cells=0`, gradient threshold `2.0 m/m`, below-ground tolerance `0.15 m`, and extreme HAG threshold `20 m`.
- A nonempty cell with at least 32 points is a direct control and uses `np.quantile(cell_z, 0.05, method="linear")`.
- Sparse cells are not direct controls. Empty and sparse cell centers query a `cKDTree` built only from direct-control centers. Valid estimates use up to four local neighbors with inverse-distance-squared weights; distance zero uses the control height directly.
- The nearest support farther than `1.5 m` produces `NaN` ground/HAG and both `EXTRAPOLATION_TOO_FAR` and `INVALID_GROUND_ESTIMATE`. The run fails only if no point has a valid estimate.
- `HIGH_LOCAL_GROUND_VARIATION` compares final finite up/down/left/right grid heights. It is diagnostic only: it does not clamp, smooth, or modify ground elevation.

## HAG and classification definition

- `raw_height_agl_m = xyz_local[:,2] - ground_z_m`.
- `height_agl_m = maximum(raw_height_agl_m, 0)`.
- A raw negative between `0` and `-0.15 m` is preserved in `raw_height_agl_m`, clamped to zero only in `height_agl_m`, and not marked below-ground.
- A raw HAG below `-0.15 m` receives `BELOW_GROUND_TOLERANCE`; a raw HAG above `20 m` receives `EXTREME_HEIGHT_AGL`.
- Invalid ground leaves all three ground/HAG values as `NaN`; it is never silently written as zero.
- Only `classification == 2` is an auxiliary disagreement check. As no point has that code, no disagreement flag is set.

## Synthetic tests and quality checks

The 21 A2B-focused tests cover horizontal ground, a sloped plane, vegetation/wire heights, one extreme low point, sparse controls, empty-cell IDW, zero-distance IDW, `requested_k` above control count, limited extrapolation, zero-control failure, max boundary and deterministic cell ids, linear quantile use, small/large negative HAG, extreme HAG, gradient diagnostics without changing estimates, classification agreement/disagreement, index preservation, JSON/NPZ reload, no overwrite, cancellation cleanup, output-path rejection, input hashes, no LAS full read, and no global N x N distance matrix.

- Full test suite: `133 passed in 3.73s`.
- `ruff check .`: passed.
- `compileall algorithms core tests scripts`: passed, with cache directed outside the workspace.
- A2B file format check: `6 files already formatted`.
- `git diff --check`: passed before Review creation and will be checked again in final protection checks.

## Real scene_simple result

Output artifacts:

- `outputs/features/scene_simple_ground_hag.npz`, 1,933,459 bytes, SHA-256 `1dde7cba2eb0d5040a6bbc3844fcc1be735b2cd09bdbd8c50176065d687072ec`.
- `outputs/features/scene_simple_ground_hag.json`, 7,721 bytes.

All point arrays have first dimension 146,126. The output `original_index` and `sample_index` match both A1 and A2A element-by-element; `origin_xyz` also matches. JSON array metadata matches NPZ arrays, the JSON-declared NPZ checksum matches, all valid ground/HAG values are finite, and the raw/clamped HAG formulas were verified element-wise. There are no invalid output points.

Grid statistics:

| Measure | Count |
| --- | ---: |
| total cells | 520 (`20 x 26`) |
| nonempty cells | 455 |
| direct control cells | 423 |
| sparse cells | 32 |
| empty cells | 65 |
| interpolated cells | 82 |
| too-far cells | 15 |
| high-gradient cells | 139 |
| valid points | 146,126 |
| invalid points | 0 |

The 15 too-far cells are empty grid positions, so they do not create invalid point estimates. The 32 sparse observed cells contain 454 points; all are within the finite interpolation limit and are marked sparse/interpolated.

Distributions (`min, p01, p05, p25, median, p75, p95, p99, max, mean, std`):

| Value | Distribution (m) |
| --- | --- |
| `ground_z_m` | 0.074500, 0.107460, 0.312800, 0.397860, 0.446570, 0.881270, 4.293200, 5.196360, 6.670080, 1.048057, 1.249079 |
| `raw_height_agl_m` | -5.579480, -0.126914, -0.000100, 0.145029, 1.010250, 2.225290, 7.004487, 8.938054, 10.564000, 1.773226, 2.199424 |
| `height_agl_m` | 0.000000, 0.000000, 0.000000, 0.145029, 1.010250, 2.225290, 7.004487, 8.938054, 10.564000, 1.778577, 2.193748 |

HAG counts: raw `<0`: 7,460; raw `<-0.15 m`: 1,206; clamped HAG `>=2/4/6/8 m`: 40,971 / 21,579 / 12,003 / 3,974; HAG `>20 m`: 0.

Point quality counts: `NONE=102574`, `SPARSE_GROUND_CELL=454`, `EMPTY_GROUND_CELL=0`, `INTERPOLATED_GROUND=454`, `EXTRAPOLATION_TOO_FAR=0`, `HIGH_LOCAL_GROUND_VARIATION=42711`, `BELOW_GROUND_TOLERANCE=1206`, `EXTREME_HEIGHT_AGL=0`, `INVALID_GROUND_ESTIMATE=0`, and `SOURCE_CLASSIFICATION_DISAGREES=0`.

## Performance and memory

- Algorithm elapsed time before artifact publication: `0.2708404 s`.
- Observed end-to-end command wall time, including Python startup and result serialization: about `5.7 s`.
- RSS start / peak / end: 64,864,256 / 76,075,008 / 77,234,176 bytes.
- KDTree coordinate storage: 3,384 bytes for 423 control centers.
- No N x N distance matrix was built. Only a control-cell `cKDTree`, grid-sized arrays (520 cells), and point-sized arrays are retained.
- The run is within the targets of under 60 seconds and under 1 GB peak RSS.

## Git and artifact protection

The two formal A2B output files are under `outputs/features` and must remain ignored by Git. No data, A1, or A2A artifact was modified. No LAS point record was read: A2B loads only the existing A1/A2A NPZ/JSON artifacts and tests fail if `laspy.read` is invoked.

## Known limitations and recommendation

- This is a baseline constant-per-cell low-quantile surface with finite IDW support. It is not a validated terrain classifier, progressive ground filter, or slope-constrained surface reconstruction.
- Geometry features are validated but intentionally not used as default ground-selection thresholds. Their use requires separate calibration.
- The 139 high-gradient cells and 42,711 flagged points indicate discontinuities between low-quantile surface estimates. They are retained as diagnostics, not hidden by smoothing or threshold changes. A3 must be able to reject or downweight these points.
- Source classification is unusable for validation on this sample because all values are unclassified (`0`).
- No HAG preview HTML was generated, and no A3 candidate threshold is defined here.

Recommendation: approve A2B for commit after GPT review. Do not start A3 automatically; first review this baseline, especially the high-gradient quality distribution, then obtain separate A3 planning/implementation authorization.
