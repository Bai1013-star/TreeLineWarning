# T5A Review: Formal scene_simple Crop and Validation

## Branch and Baseline

- Current branch: `feature/io`
- Baseline commit: `95bc8ce feat: add interactive point cloud crop selector`
- No commit made for T5A.
- No push performed.

Start checks:

```text
git status --short --branch
## feature/io

git log -7 --oneline
95bc8ce feat: add interactive point cloud crop selector
20f03bb feat: add point cloud crop range selector
8700e55 feat: add streamed point cloud preview generator
9763d3c feat: add chunked LAS reader and crop engine
80c444d feat: add readonly LAS discovery and header summary
81a6dbb feat: add core contracts and test scaffold
72f9935 chore: initialize repository and add project documentation
```

Worktree was clean before T5A changes.

## Environment Verification

Actual environment path used:

```text
E:\Miniforge\envs\treeline
```

Checks:

```text
Test-Path E:\Miniforge\envs\treeline\python.exe
True

E:\Miniforge\envs\treeline\python.exe --version
Python 3.11.15
```

No install, upgrade, uninstall, or environment modification command was run.

## Approved Crop Bounds

Selection file:

```text
outputs/selections/scene_simple_bounds.json
```

Validated values:

```json
{
  "source_id": "raw_scene_01_small_scan_small_las_685cb9a412",
  "source_path": "E:\\TreeLineWarning\\data\\raw\\scene_01_small\\scan_small.las",
  "min_xyz": [-11.0, -7.0, -3.1609000000000003],
  "max_xyz": [2.0, 3.0, 9.0],
  "coordinate_mode": "source",
  "label": "scene_simple"
}
```

Additional validation:

- `source_path` exactly matched the approved source LAS.
- `source_id` matched `outputs/previews/scan_small_preview.json`.
- `coordinate_mode` was `source`.
- Every axis satisfied `min < max`.
- Values matched the approved range.

## Actual Modified Files

Code and tests:

```text
algorithms/io/__init__.py
algorithms/io/samples.py
tests/io/test_scene_sample.py
reviews/T5A-review.md
```

Generated sample artifacts, intentionally ignored by Git:

```text
data/samples/scene_simple.las
data/samples/scene_simple_index_map.npz
data/samples/scene_simple_manifest.json
```

## Source LAS Header

Source:

```text
E:\TreeLineWarning\data\raw\scene_01_small\scan_small.las
```

Header-only summary:

```text
file_size_bytes: 414332205
las_version: 1.4
point_format_id: 7
point_record_length_bytes: 60
point_count: 6905520
min_xyz: [-25.051700592041016, -20.719900131225586, -3.1609000000000003]
max_xyz: [21.584299407958987, 18.31899986877442, 13.747100000000001]
scale: [0.0001, 0.0001, 0.0001]
offset: [-25.051700592041016, -20.719900131225586, 0.0]
extra_dimensions: [NormalX, NormalY, NormalZ]
has_rgb: true
has_intensity: true
has_classification: true
VLR summary: ExtraBytesVlr, user_id LASF_Spec, record_id 4
```

Standard / source dimensions:

```text
X, Y, Z, intensity, return_number, number_of_returns, synthetic, key_point,
withheld, overlap, scanner_channel, scan_direction_flag, edge_of_flight_line,
classification, user_data, scan_angle, point_source_id, gps_time, red, green,
blue, NormalX, NormalY, NormalZ
```

## Precheck Results

Precheck used the existing chunk reader and did not write output.

```text
chunk_size: 250000
input_point_count: 6905520
estimated_kept_point_count: 2008464
kept_ratio: 0.29084905988252874
actual_kept_min_xyz: [-10.999700592041014, -6.999900131225585, -1.8827]
actual_kept_max_xyz: [1.9999994079589847, 2.9999998687744167, 9.0]
```

Boundary counts within `0.10 m` among kept points:

```text
x_min: 1344
x_max: 19931
y_min: 5460
y_max: 27961
z_min: 0
z_max: 585
```

Z max boundary report:

```text
z_max_boundary_count: 585
z_max_boundary_ratio: 0.0002912673565470927
```

No automatic boundary adjustment was made.

## Formal Crop Outputs

Output files:

```text
data/samples/scene_simple.las
data/samples/scene_simple_index_map.npz
data/samples/scene_simple_manifest.json
```

Output file sizes:

```text
scene_simple.las: 120508845 bytes
scene_simple_index_map.npz: 3060946 bytes
scene_simple_manifest.json: 5735 bytes
```

Formal crop result:

```text
output_point_count: 2008464
kept_ratio: 0.29084905988252874
processing_seconds: 4.139892700004566
process_memory_before: 54902784 bytes
process_memory_after: 74133504 bytes
warnings: []
```

## Output LAS Validation

`scene_simple.las` was reopened with `laspy.open()` and validated with chunk iteration.

```text
output_las_opened: true
output_point_count: 2008464
output_points_within_bounds: true
output_min_xyz: [-10.999700592041014, -6.999900131225585, -1.8827]
output_max_xyz: [1.9999994079589847, 2.9999998687744167, 9.0]
header_preserved: true
```

Preserved header / dimensions:

```text
las_version: 1.4
point_format_id: 7
scale: [0.0001, 0.0001, 0.0001]
offset: [-25.051700592041016, -20.719900131225586, 0.0]
RGB: present
intensity: present
classification: present
gps_time: present
return fields: present
NormalX/NormalY/NormalZ: present
Extra Bytes VLR: preserved
```

Output dimensions:

```text
X, Y, Z, intensity, return_number, number_of_returns, synthetic, key_point,
withheld, overlap, scanner_channel, scan_direction_flag, edge_of_flight_line,
classification, user_data, scan_angle, point_source_id, gps_time, red, green,
blue, NormalX, NormalY, NormalZ
```

## Index Map Validation

Sidecar:

```text
data/samples/scene_simple_index_map.npz
```

Contents:

- `original_index`: int64
- `source_id`
- `crop_bounds`
- `output_point_count`

Validation:

```text
index_map_length: 2008464
output_las_point_count: 2008464
index_map_min: 0
index_map_max: 6888864
source_point_count: 6905520
index_map_unique: true
out_of_range: false
duplicate_original_index: false
```

The index map is stored as a sidecar and was not written into the LAS point format.

## Manifest Validation

Manifest:

```text
data/samples/scene_simple_manifest.json
```

Manifest was checked against actual generated files:

```text
manifest_matches_files: true
source_file_size matches source: true
output_file_size matches output: true
output_point_count matches LAS header: true
index_map length matches output point count: true
source_file_unchanged: true
```

The manifest records:

- source id and source path
- source file size and point count
- source header summary
- crop bounds and coordinate mode
- output path and point count
- kept ratio and output file size
- index map path
- creation time
- chunk size
- processing time
- process memory before / after
- source file before / after snapshots
- validation results
- algorithm name and version

## Source LAS Before / After

Before:

```text
size_bytes: 414332205
last_write_time: 2026-07-20T07:57:48+00:00
attributes: 32
path: E:\TreeLineWarning\data\raw\scene_01_small\scan_small.las
```

After:

```text
size_bytes: 414332205
last_write_time: 2026-07-20T07:57:48+00:00
attributes: 32
path: E:\TreeLineWarning\data\raw\scene_01_small\scan_small.las
```

Result:

```text
source_file_unchanged: true
```

## Tests

Command:

```powershell
E:\Miniforge\envs\treeline\python.exe -m pytest
```

Result:

```text
collected 70 items
tests\contracts\test_pointcloud_contracts.py ...........
tests\contracts\test_progress_contracts.py ......
tests\io\test_chunks_and_crop.py .............
tests\io\test_discovery.py ..
tests\io\test_headers.py ......
tests\io\test_scene_sample.py .......
tests\preprocessing\test_preview.py ...........
tests\scripts\test_crop_selector.py ..............

70 passed in 1.01s
```

Added T5A test coverage:

- formal crop and index map one-to-one correspondence
- manifest fields and validation
- all output points are within bounds
- Point Format, Scale, Offset, RGB, intensity, classification, and Extra Bytes preservation
- existing outputs rejected by default
- forbidden non-`data/samples` output rejected
- failure cleanup removes generated files and temp files
- source file unchanged
- full point read interfaces are not called

## Ruff

Command:

```powershell
E:\Miniforge\envs\treeline\python.exe -m ruff check .
```

Result:

```text
All checks passed!
```

## Compileall

Command:

```powershell
E:\Miniforge\envs\treeline\python.exe -m compileall algorithms core tests scripts
```

Result:

```text
Listing 'algorithms'...
Listing 'algorithms\io'...
Listing 'algorithms\preprocessing'...
Listing 'core'...
Listing 'core\config'...
Listing 'core\contracts'...
Listing 'core\pipeline'...
Listing 'tests'...
Listing 'tests\contracts'...
Listing 'tests\io'...
Listing 'tests\preprocessing'...
Listing 'tests\scripts'...
Listing 'scripts'...
```

## Git Ignore Confirmation

```text
git check-ignore -v data\samples\scene_simple.las
.gitignore:2:data/samples/	"data\\samples\\scene_simple.las"

git check-ignore -v data\samples\scene_simple_index_map.npz
.gitignore:2:data/samples/	"data\\samples\\scene_simple_index_map.npz"

git check-ignore -v data\samples\scene_simple_manifest.json
.gitignore:2:data/samples/	"data\\samples\\scene_simple_manifest.json"
```

## Git Status and Diff

Command:

```powershell
git status --short --branch
```

Current result after this review file was added:

```text
## feature/io
 M algorithms/io/__init__.py
?? algorithms/io/samples.py
?? reviews/T5A-review.md
?? tests/io/test_scene_sample.py
```

Command:

```powershell
git diff --stat
```

Current result:

```text
algorithms/io/__init__.py | 10 ++++++++++
1 file changed, 10 insertions(+)
```

Note: `git diff --stat` does not include untracked files. T5A also adds:

```text
algorithms/io/samples.py
tests/io/test_scene_sample.py
reviews/T5A-review.md
```

Command:

```powershell
git diff --check
```

Result:

```text
```

Exit code: 0. Git printed only LF-to-CRLF working-copy normalization warnings.

## Safety Confirmation

- Used `laspy.open()` and `chunk_iterator` for source and output point traversal.
- Did not call `laspy.read()`.
- Did not call `reader.read()`.
- Did not load the full original LAS into memory.
- Did not read `scan_large.las` point records.
- Did not modify, move, delete, or overwrite anything under `data/raw`.
- Did not use Open3D to load point clouds.
- Did not start preprocessing, wires, trees, clearance, risk, prediction, or model work.
- Did not install, upgrade, or uninstall dependencies.
- Did not create model environments or download weights.
- Did not commit or push.

## Known Limits and Risks

- The formal output is about 2.0 million points and 120 MB; downstream tasks should continue to
  use chunking or deliberate downsampling for display and algorithms.
- `z_max=9.0` is touched by 585 kept points within 0.10 m of the upper boundary, about 0.029% of
  kept points. This is not a large concentration, but the upper boundary should still be visually
  reviewed by the project group.
- The index map is a sidecar file, not embedded in the LAS. Downstream modules must load it when
  they need original point traceability.

## Recommendation

- Recommend accepting the generated `scene_simple` sample for the first formal small-scene
  workflow.
- Recommend GPT review T5A.
- If approved, recommend committing only code, tests, and this review package. Do not commit
  `data/samples/*`, LAS files, NPZ files, generated outputs, caches, or models.
