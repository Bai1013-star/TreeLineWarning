# T5B Review: scene_simple Integrity Check and Preview

## Branch and Baseline

- Current branch: `feature/io`
- Baseline commit: `723010e feat: add validated scene sample generation`
- No commit made for T5B.
- No push performed.

Start checks:

```text
git status --short --branch
## feature/io

git log -8 --oneline
723010e feat: add validated scene sample generation
95bc8ce feat: add interactive point cloud crop selector
20f03bb feat: add point cloud crop range selector
8700e55 feat: add streamed point cloud preview generator
9763d3c feat: add chunked LAS reader and crop engine
80c444d feat: add readonly LAS discovery and header summary
81a6dbb feat: add core contracts and test scaffold
72f9935 chore: initialize repository and add project documentation
```

Worktree was clean before T5B changes.

## Environment Verification

```text
Test-Path E:\Miniforge\envs\treeline\python.exe
True

E:\Miniforge\envs\treeline\python.exe --version
Python 3.11.15
```

No environment path fallback was used. No install, upgrade, or uninstall command was run.

## Actual Modified Files

```text
algorithms/io/__init__.py
algorithms/io/samples.py
algorithms/preprocessing/preview.py
tests/io/test_scene_sample.py
reviews/T5B-review.md
```

Generated preview artifacts, ignored by Git:

```text
outputs/previews/scene_simple_preview.npz
outputs/previews/scene_simple_preview.json
outputs/previews/scene_simple_preview.html
```

## Sample Inputs

```text
data/samples/scene_simple.las
data/samples/scene_simple_index_map.npz
data/samples/scene_simple_manifest.json
```

Approved bounds:

```text
X: -11.0 to 2.0
Y: -7.0 to 3.0
Z: -3.1609000000000003 to 9.0
coordinate_mode: source
```

## scene_simple Header and Dimensions

Header-only and chunked validation:

```text
file_size_bytes: 120508845
las_version: 1.4
point_format_id: 7
point_record_length_bytes: 60
point_count: 2008464
min_xyz: [-10.999700592041014, -6.999900131225585, -1.8827]
max_xyz: [1.9999994079589847, 2.9999998687744167, 9.0]
scale: [0.0001, 0.0001, 0.0001]
offset: [-25.051700592041016, -20.719900131225586, 0.0]
has_rgb: true
has_intensity: true
has_classification: true
```

Dimensions:

```text
X, Y, Z, intensity, return_number, number_of_returns, synthetic, key_point,
withheld, overlap, scanner_channel, scan_direction_flag, edge_of_flight_line,
classification, user_data, scan_angle, point_source_id, gps_time, red, green,
blue, NormalX, NormalY, NormalZ
```

Extra Bytes:

```text
NormalX, NormalY, NormalZ
```

VLR:

```text
ExtraBytesVlr, user_id LASF_Spec, record_id 4
```

## Index Map and Manifest Consistency

Validation result:

```text
points_within_bounds: true
manifest_matches_files: true
index_map_length: 2008464
index_map_min: 0
index_map_max: 6888864
index_map_unique: true
index_map_source_id: raw_scene_01_small_scan_small_las_685cb9a412
```

Manifest matched:

- `output_path`
- `index_map_path`
- `output_point_count`
- `output_file_size`
- `crop_bounds`
- `coordinate_mode`

## Boundary Proximity Counts

Counts and ratios use the full `scene_simple.las` point count: `2,008,464`.

| Boundary | <=0.05m | Ratio | <=0.10m | Ratio | <=0.20m | Ratio |
|---|---:|---:|---:|---:|---:|---:|
| X min | 748 | 0.000372 | 1344 | 0.000669 | 3067 | 0.001527 |
| X max | 8810 | 0.004386 | 19931 | 0.009924 | 35520 | 0.017685 |
| Y min | 2576 | 0.001283 | 5460 | 0.002718 | 12105 | 0.006027 |
| Y max | 13967 | 0.006954 | 27961 | 0.013922 | 56946 | 0.028353 |
| Z min | 0 | 0.000000 | 0 | 0.000000 | 0 | 0.000000 |
| Z max | 305 | 0.000152 | 585 | 0.000291 | 1190 | 0.000592 |

## Z Max Detail

Points within `0.10m` of `Z max = 9.0`:

```text
count: 585
ratio: 0.0002912673565470927
xyz_min: [-8.720800592041016, 0.0343998687744147, 8.9002]
xyz_max: [-5.456500592041014, 2.9982998687744136, 9.0]
classification_counts: {"0": 585}
intensity_min: 0
intensity_max: 213
intensity_mean: 35.723076923076924
rgb_min: [1024, 1024, 0]
rgb_max: [57600, 57344, 61440]
```

Automated judgment:

- Z max boundary contact exists, including points exactly at `9.0`.
- The count is very small relative to the full sample, about `0.029%`.
- All Z max near-boundary points have classification `0`, so current LAS labels do not identify
  them as a specific target class.
- The near-boundary points are spatially concentrated around X `-8.72` to `-5.46`, Y `0.03` to
  `3.00`.
- No large numeric concentration suggests broad tree crown or wire truncation at `Z max`.
- Automated statistics alone cannot prove no individual wire or crown segment is clipped; use the
  generated HTML preview for visual acceptance.

Parallel wire / crown truncation check:

- No obvious large-scale truncation pattern was detected from boundary counts.
- `Y max` has the largest boundary-adjacent population, especially within `0.20m`
  (`56,946`, about `2.84%`), so the project group should visually inspect the upper-Y side.
- Recommend accepting only after visual review of `scene_simple_preview.html`; no automatic
  boundary change was made.

## Preview Generation

Existing streamed preview code was reused.

Output paths:

```text
outputs/previews/scene_simple_preview.npz
outputs/previews/scene_simple_preview.json
outputs/previews/scene_simple_preview.html
```

Preview result:

```text
original_point_count: 2008464
preview_point_count: 143462
sampling_ratio: 0.07142871368369062
stride: 14
plotly_available: true
elapsed_seconds: 0.22648550000303658
rss_start_bytes: 57675776
rss_peak_bytes: 90636288
warnings: []
```

Preview HTML behavior:

- default display mode is height on a dark background
- includes enhanced RGB mode
- includes Classification mode
- includes Intensity mode
- marks the approved crop bounds with an orange 3D box
- embeds only the low-density preview, not the full 2,008,464 points

Preview artifact sizes:

```text
scene_simple_preview.npz: 6546477 bytes
scene_simple_preview.json: 1513 bytes
scene_simple_preview.html: 60015422 bytes
```

Git ignore confirmation:

```text
git check-ignore -v outputs\previews\scene_simple_preview.npz
.gitignore:4:outputs/	"outputs\\previews\\scene_simple_preview.npz"

git check-ignore -v outputs\previews\scene_simple_preview.json
.gitignore:4:outputs/	"outputs\\previews\\scene_simple_preview.json"

git check-ignore -v outputs\previews\scene_simple_preview.html
.gitignore:4:outputs/	"outputs\\previews\\scene_simple_preview.html"
```

## Sample Before / After

`scene_simple.las` before:

```text
size_bytes: 120508845
last_write_time: 2026-07-20T14:06:30.483563+00:00
attributes: 32
path: E:\TreeLineWarning\data\samples\scene_simple.las
```

`scene_simple.las` after:

```text
size_bytes: 120508845
last_write_time: 2026-07-20T14:06:30.483563+00:00
attributes: 32
path: E:\TreeLineWarning\data\samples\scene_simple.las
```

Result:

```text
sample_unchanged: true
```

Original `scan_small.las` was also checked by metadata snapshot and unchanged:

```text
raw_unchanged: true
```

## Tests

Command:

```powershell
E:\Miniforge\envs\treeline\python.exe -m pytest
```

Result:

```text
collected 73 items
tests\contracts\test_pointcloud_contracts.py ...........
tests\contracts\test_progress_contracts.py ......
tests\io\test_chunks_and_crop.py .............
tests\io\test_discovery.py ..
tests\io\test_headers.py ......
tests\io\test_scene_sample.py ..........
tests\preprocessing\test_preview.py ...........
tests\scripts\test_crop_selector.py ..............

73 passed in 1.19s
```

Added / updated coverage:

- sample, index map, and manifest consistency check
- boundary proximity counts for multiple tolerances
- no full read interfaces during scene sample preview
- preview HTML includes approved crop bounds
- source sample unchanged after preview generation

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
Compiling 'tests\io\test_scene_sample.py'...
Listing 'tests\preprocessing'...
Listing 'tests\scripts'...
Listing 'scripts'...
```

## Git Status and Diff

Command:

```powershell
git status --short --branch
```

Final result after this review file was added:

```text
## feature/io
 M algorithms/io/__init__.py
 M algorithms/io/samples.py
 M algorithms/preprocessing/preview.py
 M tests/io/test_scene_sample.py
?? reviews/T5B-review.md
```

Command:

```powershell
git diff --stat
```

Final result:

```text
algorithms/io/__init__.py           |   4 +
algorithms/io/samples.py            | 233 ++++++++++++++++++++++++++++++++++
algorithms/preprocessing/preview.py | 245 ++++++++++++++++++++++++++++++++----
tests/io/test_scene_sample.py       | 114 +++++++++++++++++
4 files changed, 573 insertions(+), 23 deletions(-)
```

Note: `git diff --stat` does not include untracked files. T5B also adds:

```text
reviews/T5B-review.md
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

- Used `laspy.open()` and `chunk_iterator`.
- Did not call `laspy.read()`.
- Did not call `reader.read()`.
- Did not load the full `scene_simple.las` point cloud at once.
- Did not modify `scene_simple.las`, index map, or manifest.
- Did not regenerate or overwrite the formal sample.
- Did not read `scan_large.las` point records.
- Did not modify `data/raw`.
- Did not start preprocessing, wires, trees, clearance, risk, prediction, or model work.
- Did not install, upgrade, or uninstall dependencies.
- Did not commit or push.

## Recommendation

- Recommend accepting the current `scene_simple` sample for the next module-level review, subject
  to project group visual inspection of the generated HTML preview.
- Recommend entering `feature/io` module total review after T5B review.
- If approved, commit only code, tests, and this review package. Do not commit `data/`, `outputs/`,
  LAS, NPZ, HTML, caches, models, or generated results.
