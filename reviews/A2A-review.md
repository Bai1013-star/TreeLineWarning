# A2A Review: Local PCA Geometry Features

## 1. Scope and baseline

- Branch: `feature/algorithm`.
- Baseline HEAD: `e10cccd feat: add basic preprocessing point cloud pipeline`.
- This review covers A2A only: local PCA geometry features from the existing A1 NPZ/JSON output. It does not implement A2B, A3, or any later algorithm stage.
- No commit, push, merge, environment change, A1 rerun, data-file edit, or LAS point-record read was performed.

## 2. Modified files

The uncommitted A2A draft consists of:

- `core/contracts/features.py`
- `core/contracts/__init__.py`
- `algorithms/features/geometry.py`
- `algorithms/features/__init__.py`
- `tests/contracts/test_feature_contracts.py`
- `tests/features/test_geometry.py`
- `reviews/A2A-review.md` (this review)

There are no staged changes. Existing A1 and I/O/preprocessing modules were not changed in this task.

## 3. Public contracts

The A2A draft adds `GeometryFeatureConfig`, `FeatureQualityFlag`, `PointFeatureSet`, `GeometryFeatureStatistics`, and `GeometryFeatureResult`. The point-feature arrays preserve A1 order and contain `source_id`, `original_index`, `sample_index`, `xyz_local`, `linearity`, `planarity`, `scattering`, `verticality`, `local_density`, `height_m`, `neighbor_count`, `eigenvalues`, `principal_direction`, `normal`, and `quality_flags`.

Large arrays are stored in NPZ. JSON contains provenance, configuration, formulas, array shape/dtype metadata, statistics, output checksum, and timing/memory records. `original_index` remains the raw LAS global index and `sample_index` remains the `scene_simple.las` row index.

## 4. A1 input validation

Read-only A1 input validation passed for `outputs/pointcloud/scene_simple_preprocessed.npz` and `.json`:

- Point count `M=146126`.
- `xyz_local`: `(146126, 3)`, `float32`, finite.
- `origin_xyz`: `(3,)`, `float64`.
- `original_index`: `(146126,)`, `int64`, unique.
- `sample_index`: `(146126,)`, `int64`, unique.

Before and after A2A, the A1 files were identical:

| File | SHA-256 | Bytes | Modified time (ns) |
| --- | --- | ---: | ---: |
| `scene_simple_preprocessed.npz` | `7b12e8c950672a61e3a916a51820bf793a7301b12ace82593a9a660bf94d5d02` | 5,612,534 | 1784609874954736000 |
| `scene_simple_preprocessed.json` | `95d0cbebcf4cd2a1791e0f2ecef7c08d4552ab99c214496934296a0781a63b74` | 3,772 | 1784609874955707200 |

## 5. KNN, PCA, and memory strategy

- Default KNN configuration: `k_neighbors=32`, `min_neighbors=8`, `query_batch_size=12000`, `workers=1`.
- The query point itself is excluded by requesting 33 entries and retaining up to 32 non-self neighbors.
- A `scipy.spatial.cKDTree` is built once over A1 local coordinates. Every batch holds only its bounded query results and compact neighbor coordinates; no all-point Python neighbor container is retained.
- Each valid neighborhood produces a 3x3 covariance matrix and ordered eigenpairs. The implementation has no N x N distance matrix and does not retain a full M x K neighbor-index matrix after a batch.

## 6. Final mathematical definitions

For `lambda_1 >= lambda_2 >= lambda_3 >= 0` and `epsilon=feature_epsilon_m2`:

- `linearity = (lambda_1 - lambda_2) / max(lambda_1, epsilon)`
- `planarity = (lambda_2 - lambda_3) / max(lambda_1, epsilon)`
- `scattering = lambda_3 / max(lambda_1, epsilon)`
- `verticality = 1 - abs(normal_z)`, where `normal` is the unit eigenvector of `lambda_3`.
- `local_density = k / ((4 / 3) * pi * r_k^3)` with explicit quality handling for undefined density.
- `height_m = xyz_local[:, 2]`.

`scattering` intentionally uses the approved raw ratio, not `sqrt(lambda_3 / lambda_1)`. The ratio is in `[0, 1]` and keeps future low-scattering thresholds directly interpretable for A3. The default `direction_gap_relative_tolerance` is `1e-3`.

## 7. Quality flags

The contract supports all required flags: `INSUFFICIENT_NEIGHBORS`, `DEGENERATE_COVARIANCE`, `DUPLICATE_NEIGHBOR_COORDINATES`, `NEGATIVE_EIGENVALUE_CLIPPED`, `NORMAL_DIRECTION_AMBIGUOUS`, `PRINCIPAL_DIRECTION_AMBIGUOUS`, `NUMERIC_INVALID`, and `DENSITY_UNDEFINED`.

Full-sample counts:

| Flag | Count | Ratio |
| --- | ---: | ---: |
| `INSUFFICIENT_NEIGHBORS` | 0 | 0 |
| `DEGENERATE_COVARIANCE` | 0 | 0 |
| `DUPLICATE_NEIGHBOR_COORDINATES` | 0 | 0 |
| `NEGATIVE_EIGENVALUE_CLIPPED` | 0 | 0 |
| `NORMAL_DIRECTION_AMBIGUOUS` | 2 | 0.00001369 |
| `PRINCIPAL_DIRECTION_AMBIGUOUS` | 4 | 0.00002737 |
| `NUMERIC_INVALID` | 0 | 0 |
| `DENSITY_UNDEFINED` | 0 | 0 |
| no flags | 146,120 | 0.99995894 |

No point had more than one flag. The six strict direction ambiguities are not abnormal at the `1e-3` gap threshold; zero vectors, when used, carry their matching ambiguity flag.

## 8. Synthetic and full test results

The synthetic tests cover lines, planes, volumetric isotropic points, horizontal/vertical plane verticality, principal/normal ambiguity sentinels, duplicate and degenerate neighborhoods, index order, serialization, cancellation, no overwrite, no full LAS read, and ratio scattering. The isotropic fixture is a deterministic 3D volume sample (seed 7), rather than a sphere surface that locally behaves as a plane. The plane ambiguity branch uses an explicit `direction_gap_relative_tolerance=0.01`; the contract test preserves the default `1e-3`.

- `E:\Miniforge\envs\treeline\python.exe -m pytest`: `112 passed in 7.38s`.
- `ruff check .`: passed.
- `compileall algorithms core tests scripts`: passed.
- A2A-only `ruff format --check`: `6 files already formatted`.
- Whole-repository `ruff format --check .`: fails only on the pre-existing, prohibited-to-format files `algorithms/io/samples.py`, `algorithms/preprocessing/preview.py`, and `core/contracts/pointcloud.py`; 38 other files are formatted. No change was made to those files.
- `git diff --check`: passed.

## 9. Full real-sample execution

The approved call ran once with the default configuration and atomic output publication:

```python
run_geometry_feature_extraction(
    "outputs/pointcloud/scene_simple_preprocessed.npz",
    "outputs/pointcloud/scene_simple_preprocessed.json",
    "outputs/features",
    project_root="E:\\TreeLineWarning",
)
```

- Input points: 146,126.
- Output points: 146,126.
- PCA computed: 146,126; no input point was reordered, removed, or added.
- Output NPZ: `outputs/features/scene_simple_geometry_features.npz`, 10,132,393 bytes, SHA-256 `8c13802e8aafa83a5e0c0f04318c40d1ac3cff8cb4e7d98713c63654d11a74d4`.
- Output JSON: `outputs/features/scene_simple_geometry_features.json`, 4,965 bytes; its declared NPZ checksum matches the actual file.
- All output array first dimensions are 146,126; JSON shapes/dtypes match NPZ arrays.
- `original_index`, `sample_index`, `xyz_local`, and `origin_xyz` are element-wise identical to A1. All output arrays are finite.
- Eigenvalues are descending and non-negative within configured tolerance; ratio scattering and `height_m == xyz_local[:, 2]` checks passed. Nonzero directions have unit length within floating-point tolerance, and no zero direction lacks the matching flag.

## 10. Feature distributions

Values are `min, p01, p05, p25, median, p75, p95, p99, max, mean, std`.

| Feature | Distribution |
| --- | --- |
| linearity | 0.000458, 0.022329, 0.066380, 0.184170, 0.323755, 0.558996, 0.904975, 0.985475, 0.996277, 0.385897, 0.252024 |
| planarity | 0.000470, 0.013006, 0.052657, 0.265835, 0.492874, 0.705672, 0.884881, 0.971706, 0.999472, 0.484388, 0.263112 |
| scattering | 0.000003, 0.000030, 0.000096, 0.022545, 0.096174, 0.195339, 0.390181, 0.547563, 0.883561, 0.129715, 0.129594 |
| verticality | 0.000000, 0.000028, 0.000222, 0.075563, 0.626326, 0.886033, 0.982385, 0.996401, 0.999999, 0.524764, 0.377252 |
| local_density (points/m3) | 2.274841, 97.788113, 302.922821, 1519.166443, 2721.671753, 4587.278442, 6365.750854, 7553.471924, 10412.893555, 3038.287979, 1939.582646 |
| height_m (m) | 0.000000, 0.312200, 0.369200, 0.875200, 1.888700, 4.227300, 8.585650, 9.859625, 10.882700, 2.821283, 2.540948 |
| neighbor_count | 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 0 |
| lambda_1 (m2) | 0.001900, 0.002500, 0.002886, 0.003816, 0.005211, 0.007670, 0.024048, 0.043942, 0.836304, 0.008187, 0.012578 |
| lambda_2 (m2) | 0.000168, 0.000400, 0.001142, 0.002318, 0.003066, 0.004283, 0.007196, 0.017312, 0.224915, 0.003826, 0.004140 |
| lambda_3 (m2) | 0.000000026, 0.000000234, 0.000000860, 0.000149, 0.000499, 0.000943, 0.002278, 0.005868, 0.067285, 0.000779, 0.001418 |

Diagnostic combinations only, not A3 rules:

| Condition | Count | Ratio |
| --- | ---: | ---: |
| `linearity >= 0.7` | 20,110 | 0.137621 |
| `linearity >= 0.8` | 12,137 | 0.083058 |
| `linearity >= 0.9` | 7,522 | 0.051476 |
| `scattering <= 0.01` | 29,377 | 0.201039 |
| `scattering <= 0.03` | 40,426 | 0.276652 |
| `scattering <= 0.05` | 50,445 | 0.345216 |
| `scattering <= 0.1` | 74,866 | 0.512339 |
| `linearity >= 0.8 and scattering <= 0.05` | 10,017 | 0.068550 |
| `linearity >= 0.9 and scattering <= 0.03` | 7,120 | 0.048725 |
| `height_m >= 2` | 68,774 | 0.470649 |
| `height_m >= 4` | 39,419 | 0.269760 |
| `height_m >= 6` | 20,556 | 0.140673 |
| `linearity >= 0.8 and scattering <= 0.05 and height_m >= 4` | 9,309 | 0.063705 |

## 11. Spatial diagnostic samples

For the top 100 points of each feature, every selected point occupied a distinct A1 voxel, no selected `original_index` repeated, no selected coordinate set was constant, and no high feature value came from a degenerate-quality point. Median companion values were plausible:

| Top-100 feature | xyz range (m), height range (m) | raw original-index range | Median companion values |
| --- | --- | --- | --- |
| linearity | x/y/z min `[1.1974, 5.0047, 0.2872]`, max `[12.5663, 9.3938, 9.4351]`; h 0.2872..9.4351 | 1,255,380..6,481,297 | planarity 0.005127, scattering 0.000635, verticality 0.043965, density 178.892 |
| planarity | min `[1.6645, 0.4785, 0.1148]`, max `[12.8199, 9.6802, 4.2664]`; h 0.1148..4.2664 | 1,558..4,689,594 | linearity 0.004405, scattering 0.000098, verticality 0.000180, density 1956.272 |
| scattering | min `[2.6973, 1.4335, 0.0768]`, max `[12.4383, 9.6691, 6.4052]`; h 0.0768..6.4052 | 28,289..5,015,079 | linearity 0.122409, planarity 0.100539, verticality 0.484395, density 6529.429 |
| verticality | min `[0.0205, 0.3786, 0.5321]`, max `[12.9581, 9.9701, 10.0147]`; h 0.5321..10.0147 | 54,179..6,482,147 | linearity 0.240591, planarity 0.598444, scattering 0.089676, density 4388.212 |

## 12. Performance and memory

| Measure | Result |
| --- | ---: |
| KDTree build | 0.038963 s |
| batch KNN/PCA | 5.677414 s |
| KDTree query accumulation | 0.754816 s |
| feature core elapsed (before write) | 5.832366 s |
| output write | 0.858067 s |
| total wall time | 6.695414 s |
| validation/other | 0.159932 s |
| RSS start | 65,236,992 bytes |
| peak RSS | 87,224,320 bytes |
| RSS end | 80,560,128 bytes |
| KDTree coordinate storage | 3,507,024 bytes |
| maximum query batch | 12,000 points |

The run is well within the targets of under 120 seconds and under 1.5 GB peak RSS. No N x N matrix was built, and no whole-neighbor matrix was retained across batches.

## 13. Output and Git protection

Both A2A outputs are ignored by `.gitignore` through `.gitignore:4:outputs/`. `git ls-files data outputs` is empty, so neither source/derived point-cloud data nor feature results are tracked. The A1 files are unchanged as shown in section 4. No LAS point record was read: the full run consumes only the existing A1 NPZ/JSON, and the A2A module does not import a LAS reader.

## 14. Environment note

The repaired numerical path executed successfully with OpenBLAS runtime libraries and no actual MKL runtime loaded in the verification process. A Conda metadata file-lock warning can still occur while listing package metadata; Python imports, NumPy/SciPy linear algebra, full tests, and this A2A run are healthy. This is a non-blocking environment-hygiene follow-up and was not altered here.

## 15. Known limitations and recommendation

- A2A provides local geometry only. It does not provide ground/HAG, candidate rules, conductor separation, tracking, spline reconstruction, vegetation, clearance, risk, prediction, models, or a UI.
- Direction signs are mathematically non-unique; ambiguous directions use the explicit flags/zero-vector convention.
- Results are validated on `scene_simple` only. Threshold calibration for later candidate work must not be inferred from this diagnostic report alone.
- No large HTML preview was generated.

Recommendation: approve the A2A draft for commit after GPT review. Do not begin A2B or A3 in this task; each requires its own reviewed plan and implementation authorization.
