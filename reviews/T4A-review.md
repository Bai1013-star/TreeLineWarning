# T4A Review: Low-Density 3D Preview for scan_small.las

## Branch and Baseline

- Current branch: `feature/io`
- Baseline commit: `9763d3c feat: add chunked LAS reader and crop engine`
- No commit made for T4A.
- No push performed.

Start checks:

```text
git status --short --branch
## feature/io

git log -4 --oneline
9763d3c feat: add chunked LAS reader and crop engine
80c444d feat: add readonly LAS discovery and header summary
81a6dbb feat: add core contracts and test scaffold
72f9935 chore: initialize repository and add project documentation
```

Worktree was clean before T4A changes.

Read before implementation:

- `docs/requirements.md`
- `docs/codex_tasks.md`
- `core/contracts/`
- `algorithms/io/chunks.py`
- `algorithms/io/headers.py`
- `algorithms/io/crop.py`

## Modified Files

Added:

```text
algorithms/preprocessing/__init__.py
algorithms/preprocessing/preview.py
scripts/__init__.py
tests/preprocessing/__init__.py
tests/preprocessing/test_preview.py
reviews/T4A-review.md
```

`scripts/__init__.py` is an empty package placeholder so the required
`compileall algorithms core tests scripts` command has an existing `scripts` target.

Ignored runtime outputs created:

```text
outputs/previews/scan_small_preview.npz
outputs/previews/scan_small_preview.json
outputs/previews/scan_small_preview.html
```

## Preview Algorithm

Public API:

```python
build_pointcloud_preview(
    source_path,
    output_dir,
    *,
    source_id,
    target_preview_points=200_000,
    max_preview_points=300_000,
    chunk_size=250_000,
    overwrite=False,
    reporter=None,
    cancellation_token=None,
    generate_html=True,
    project_root=None,
)
```

Algorithm:

- Reads the LAS header with the existing T2 header helper.
- Streams points through existing `iter_las_chunks()`.
- Does not call `laspy.read()`.
- Does not call `reader.read()`.
- Does not load the full 6,905,520-point file at once.
- Uses deterministic global stride sampling:
  - `stride = ceil(original_point_count / min(target_preview_points, max_preview_points))`
  - selects points where `original_index % stride == 0`
- Caps final arrays at `max_preview_points`.
- Keeps `original_index` for traceability.
- Saves source XYZ and local XYZ.
- Uses preview XYZ minimum as `origin_xyz`; empty files fall back to header min XYZ.

Determinism:

- Selection depends only on total point count, target/max point parameters, and global
  `original_index`.
- Same input and parameters produce the same preview rows.

## Parameters

```text
chunk_size: 250000
target_preview_points: 200000
max_preview_points: 300000
sampling method: deterministic_global_stride
actual stride for scan_small.las: 35
```

The chunk size reuses the conservative T3A default for 16GB memory.

## Preserved Fields

NPZ stores fields when available:

- `xyz`
- `local_xyz`
- `original_index`
- `rgb`
- `red`
- `green`
- `blue`
- `intensity`
- `classification`
- `NormalX`
- `NormalY`
- `NormalZ`
- `origin_xyz`
- `source_id`

Missing optional fields are ignored without failure.

## Output Safety

- Output directory is rejected if it is under `data/raw`.
- Output directory is allowed only under `outputs/previews` or pytest temp directories.
- Existing preview outputs are rejected by default when `overwrite=False`.
- Writes use temporary sibling files first.
- Failure or cancellation removes temporary files.
- Preview artifacts live under ignored `outputs/`.

## Real scan_small.las Run

Input:

```text
E:\TreeLineWarning\data\raw\scene_01_small\scan_small.las
```

Before:

```text
Name: scan_small.las
Size: 414332205
LastWriteTimeUtc: 2026/7/20 7:57:48
mtime_ns: 1784534268000000000
Attributes: Archive
```

After:

```text
Name: scan_small.las
Size: 414332205
LastWriteTimeUtc: 2026/7/20 7:57:48
mtime_ns: 1784534268000000000
Attributes: Archive
unchanged: true
```

The source LAS path, name, size, modification time, and attributes were unchanged.

Metadata:

```text
original_point_count: 6905520
preview_point_count: 197301
sampling_ratio: 0.028571490633580093
chunk_size: 250000
elapsed_seconds: 0.8643840000004275
rss_start_bytes: 55500800
rss_peak_bytes: 93097984
warnings: []
```

Original XYZ range:

```text
min: [-25.051700592041016, -20.719900131225586, -3.1609000000000003]
max: [21.584299407958987, 18.31899986877442, 13.747100000000001]
```

Preview XYZ range:

```text
min: [-24.958600592041016, -20.708700131225587, -2.0701]
max: [19.31489940795899, 17.491799868774415, 13.249600000000001]
```

Output files:

```text
E:\TreeLineWarning\outputs\previews\scan_small_preview.npz
  size: 9195097 bytes

E:\TreeLineWarning\outputs\previews\scan_small_preview.json
  size: 1352 bytes

E:\TreeLineWarning\outputs\previews\scan_small_preview.html
  size: 23398999 bytes
```

Plotly:

```text
available: true
HTML generated: true
```

The HTML uses a Plotly `Scatter3d` low-density preview, prioritizing RGB colors and
including X/Y/Z hover text. It embeds only the preview points, not the full source cloud.

## Large LAS Handling

For:

```text
E:\TreeLineWarning\data\raw\scene_02_large\scan_large.las
```

T4A did not read point records and did not run chunk iteration.

## Tests

Command:

```powershell
E:\Miniforge3\envs\treeline\python.exe -m pytest
```

Result:

```text
============================= test session starts =============================
platform win32 -- Python 3.11.15, pytest-9.1.1, pluggy-1.6.0
rootdir: E:\TreeLineWarning
configfile: pyproject.toml
testpaths: tests
plugins: anyio-4.14.2, dash-4.4.0, cov-7.1.0
collected 49 items

tests\contracts\test_pointcloud_contracts.py ...........                 [ 22%]
tests\contracts\test_progress_contracts.py ......                        [ 34%]
tests\io\test_chunks_and_crop.py .............                           [ 61%]
tests\io\test_discovery.py ..                                            [ 65%]
tests\io\test_headers.py ......                                          [ 77%]
tests\preprocessing\test_preview.py ...........                          [100%]

============================= 49 passed in 0.72s ==============================
```

Preview tests cover:

- Multi-chunk streaming.
- No full LAS read interface calls.
- Output count cap.
- Deterministic repeated output.
- Valid, in-range `original_index`.
- RGB preservation.
- intensity and classification preservation.
- Missing optional fields.
- Source/local coordinate recovery.
- JSON-serializable metadata.
- Rejecting `data/raw` output.
- Default no-overwrite behavior.
- Cancellation cleanup.
- Failure cleanup.
- Empty LAS behavior.
- NPZ reload and field recovery.

## Ruff

Command:

```powershell
E:\Miniforge3\envs\treeline\python.exe -m ruff check .
```

Result:

```text
All checks passed!
```

## Compileall

Command:

```powershell
E:\Miniforge3\envs\treeline\python.exe -m compileall algorithms core tests scripts
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
Compiling 'tests\preprocessing\test_preview.py'...
Listing 'scripts'...
Compiling 'scripts\__init__.py'...
```

## Git Status

Command:

```powershell
git status --short --branch
```

Result:

```text
## feature/io
?? algorithms/preprocessing/
?? scripts/
?? tests/preprocessing/
```

Note: `reviews/T4A-review.md` was added after this command and appears in the final status.

## Git Diff Stat

Command:

```powershell
git diff --stat
```

Result:

```text

```

All T4A code/review files are currently untracked, so `git diff --stat` has no tracked-file output.

## Git Diff Check

Command:

```powershell
git diff --check
```

Result:

```text

```

Exit code: 0.

## Git Ignore Verification

```text
git check-ignore -v outputs/previews/scan_small_preview.npz
.gitignore:4:outputs/  outputs/previews/scan_small_preview.npz

git check-ignore -v outputs/previews/scan_small_preview.json
.gitignore:4:outputs/  outputs/previews/scan_small_preview.json

git check-ignore -v outputs/previews/scan_small_preview.html
.gitignore:4:outputs/  outputs/previews/scan_small_preview.html
```

Preview artifacts are ignored and must not be committed.

## Known Limits

- The preview is a deterministic stride sample, not voxel downsampling.
- The generated HTML is a static Plotly artifact, not a Streamlit page.
- The RGB view is primary; height fallback is used only when RGB is absent.
- This does not select or crop `scene_simple.las`.
- This does not implement semantic classification, wires, trees, clearance, risk, or prediction.

## Safety Confirmation

- Did not read `scan_large.las` point records.
- Did not formally crop a real sample.
- Did not create `data/samples/scene_simple.las`.
- Did not modify `data/raw`.
- Did not modify any original LAS file.
- Did not install dependencies.
- Did not create model environments or download model weights.
- Did not commit or push.

## Commit Recommendation

Recommend GPT review T4A. If approved, recommend committing code and tests only.
Do not commit `outputs/previews/*` artifacts.
