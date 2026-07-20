# T3A Review: LAS Chunk Reader and XYZ Crop Engine

## Branch and Baseline

- Current branch: `feature/io`
- Baseline commit: `80c444d feat: add readonly LAS discovery and header summary`
- No commit made for T3A.
- No push performed.

Start checks:

```text
git status --short --branch
## feature/io

git log -3 --oneline
80c444d feat: add readonly LAS discovery and header summary
81a6dbb feat: add core contracts and test scaffold
72f9935 chore: initialize repository and add project documentation
```

Worktree was clean before T3A changes.

Read before implementation:

- `docs/requirements.md`
- `docs/codex_tasks.md`
- `core/contracts/`
- `algorithms/io/discovery.py`
- `algorithms/io/headers.py`
- `tests/io/`

## Modified Files

Modified:

```text
algorithms/io/__init__.py
```

Added:

```text
algorithms/io/chunks.py
algorithms/io/crop.py
tests/io/test_chunks_and_crop.py
reviews/T3A-review.md
```

## Chunk Reader Interface

Public API:

```python
DEFAULT_CHUNK_SIZE = 250_000

iter_las_chunks(
    path,
    *,
    source_id=None,
    chunk_size=DEFAULT_CHUNK_SIZE,
    reporter=None,
    cancellation_token=None,
    task_id=None,
)
```

Implementation:

- Uses `laspy.open(path, mode="r")`.
- Uses `reader.chunk_iterator(chunk_size)`.
- Does not call `laspy.read()`.
- Does not call `reader.read()`.
- Does not read full point arrays.
- Closes the reader with a context manager.
- Checks `CancellationToken` before each chunk and after each yielded chunk.
- Reports progress with `ProgressReporter.update(PipelineProgress)`.

Default chunk size:

```text
250000 points
```

Reason: conservative default for a 16GB machine. It keeps current chunk memory bounded while
still avoiding excessive iterator overhead for multi-million point LAS files.

Each yielded `PointCloudChunk` includes:

- `source_id`
- `chunk_id`
- `start_index`
- `point_count`
- `xyz`
- `original_index`
- `attributes`
- `bounds_xyz`
- `is_last`

## original_index

`original_index` is generated per chunk as:

```python
np.arange(start_index, start_index + point_count, dtype=np.int64)
```

It is the global source LAS point index range for that chunk. Tests verify continuity and no
duplicates across multiple chunks.

## Optional Attributes

Only approved attributes are extracted, and only when present in the point format:

```text
intensity
return_number
number_of_returns
classification
user_data
scan_angle
point_source_id
gps_time
red
green
blue
NormalX
NormalY
NormalZ
```

Missing optional fields are ignored without error. Attribute arrays are kept with their laspy /
NumPy dtype where practical; XYZ is `float64`, and `original_index` is `int64`.

## Crop Engine Interface

Public API:

```python
crop_las_xyz(
    source_path,
    output_path,
    bounds,
    *,
    source_id,
    chunk_size=DEFAULT_CHUNK_SIZE,
    overwrite=False,
    reporter=None,
    cancellation_token=None,
    project_root=None,
    task_id=None,
)
```

Crop rule:

```text
min_x <= x <= max_x
min_y <= y <= max_y
min_z <= z <= max_z
```

Coordinate modes:

- `source`: uses `CropBounds.min_xyz` and `max_xyz` directly.
- `local`: requires `origin_xyz`; effective bounds are `origin_xyz + min/max`.

Write behavior:

- Writes selected points directly to a new LAS/LAZ file with `LasWriter.write_points`.
- Uses a temporary sibling path such as `output.tmp.las`.
- Replaces target only after successful writer close.
- Removes temporary output on failure or cancellation.
- Allows zero selected points; creates an empty LAS and returns a warning.

Output safety rules:

- Output extension must be `.las` or `.laz`.
- Output cannot be the source file.
- Existing output is rejected unless `overwrite=True`.
- Output under project root must be in:
  - `data/samples`
  - `data/processed`
  - `outputs`
- Output under `data/raw` is forbidden.
- Pytest temp directories are allowed for tests.

Preservation:

- LAS version
- Point Format
- Scale
- Offset
- VLR / Extra Bytes within laspy writer support
- RGB
- intensity
- classification
- `NormalX`, `NormalY`, `NormalZ`

## Synthetic LAS Tests

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
collected 38 items

tests\contracts\test_pointcloud_contracts.py ...........                 [ 28%]
tests\contracts\test_progress_contracts.py ......                        [ 44%]
tests\io\test_chunks_and_crop.py .............                           [ 78%]
tests\io\test_discovery.py ..                                            [ 84%]
tests\io\test_headers.py ......                                          [100%]

============================= 38 passed in 1.04s ==============================
```

Covered:

- Correct chunk point counts.
- Continuous, non-duplicated `original_index`.
- Correct `is_last`.
- `chunk_size` smaller than total points.
- `chunk_size` larger than total points.
- Normal operation before cancellation.
- Stop after cancellation.
- Progress callbacks.
- XYZ crop bounds.
- Inclusive boundary points.
- Crop output point count.
- Point Format, Scale, Offset preservation.
- RGB, intensity, classification preservation.
- Extra Bytes / `NormalX/Y/Z` preservation.
- Missing optional fields do not crash.
- Reject output under `data/raw`.
- Reject source overwrite.
- Reject silent overwrite of existing output.
- Empty crop result.
- Failure cleanup removes temporary file.
- `CropResult` JSON serialization.
- Monkeypatch guard proving implementation does not call full read interfaces.

## Real Small LAS Chunk Validation

Allowed file:

```text
E:\TreeLineWarning\data\raw\scene_01_small\scan_small.las
```

Validation read only the first two chunks with `chunk_size=250000`.
It did not traverse the full file and did not create a crop output.

Before:

```text
name: scan_small.las
size: 414332205
mtime_ns: 1784534268000000000
attributes: Archive
```

Chunk 0:

```text
point_count: 250000
start_index: 0
end_index_inclusive: 249999
is_last: false
bounds_xyz:
  min: [-6.999700592041016, -7.999700131225586, -1.8057]
  max: [-5.920410153237299e-07, -0.5049001312255861, 0.0]
attribute_names:
  NormalX, NormalY, NormalZ, blue, classification, gps_time, green, intensity,
  number_of_returns, point_source_id, red, return_number, scan_angle, user_data
```

Chunk 1:

```text
point_count: 250000
start_index: 250000
end_index_inclusive: 499999
is_last: false
bounds_xyz:
  min: [-11.999500592041015, -14.991900131225584, -1.8658000000000001]
  max: [-5.920410153237299e-07, -1.3122558328859668e-07, 0.0]
attribute_names:
  NormalX, NormalY, NormalZ, blue, classification, gps_time, green, intensity,
  number_of_returns, point_source_id, red, return_number, scan_angle, user_data
```

After:

```text
name: scan_small.las
size: 414332205
mtime_ns: 1784534268000000000
attributes: Archive
unchanged: true
```

## Large LAS Handling

For:

```text
E:\TreeLineWarning\data\raw\scene_02_large\scan_large.las
```

T3A did not read point chunks and did not traverse point records. The only large-file data used
for review context is the existing T2 header summary.

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
E:\Miniforge3\envs\treeline\python.exe -m compileall algorithms core tests
```

Result:

```text
Listing 'algorithms'...
Listing 'algorithms\io'...
Listing 'core'...
Listing 'core\config'...
Listing 'core\contracts'...
Listing 'core\pipeline'...
Listing 'tests'...
Listing 'tests\contracts'...
Listing 'tests\io'...
```

## Git Status

Command:

```powershell
git status --short --branch
```

Result:

```text
## feature/io
 M algorithms/io/__init__.py
?? algorithms/io/chunks.py
?? algorithms/io/crop.py
?? reviews/T3A-review.md
?? tests/io/test_chunks_and_crop.py
```

## Git Diff Stat

Command:

```powershell
git diff --stat
```

Result:

```text
 algorithms/io/__init__.py | 5 +++++
 1 file changed, 5 insertions(+)
warning: in the working copy of 'algorithms/io/__init__.py', LF will be replaced by CRLF the next time Git touches it
```

Note: new T3A files are untracked and therefore not shown by `git diff --stat`.

## Git Diff Check

Command:

```powershell
git diff --check
```

Result:

```text
warning: in the working copy of 'algorithms/io/__init__.py', LF will be replaced by CRLF the next time Git touches it
```

Exit code: 0.

## Known Limits

- This is a crop engine only; no official real-scene crop sample was generated.
- No coordinate guessing for `scene_simple.las`.
- No downsampling, denoising, Open3D preview, Streamlit UI, preprocessing, classification,
  wire, tree, clearance, risk, or prediction logic.
- `.laz` writing depends on laspy environment support.
- Crop progress currently reports processed input points and selected output count in message;
  `CropResult` reports final counts.
- Output path validation uses project-root allowlist plus pytest temp directory support.

## Safety Confirmation

- Did not formally crop a real sample.
- Did not create `data/samples/scene_simple.las`.
- Did not read complete LAS files.
- Did not call full read APIs in implementation.
- Did not read `scan_large.las` point records.
- Did not modify, move, delete, rename, overwrite, or change attributes of any original LAS.
- Did not install dependencies.
- Did not commit or push.

## Commit Recommendation

Recommend GPT review T3A. If approved, recommend commit T3A because:

- pytest: 38 passed.
- ruff: All checks passed.
- compileall: passed.
- Small LAS chunk validation read only the first two chunks and left the source unchanged.
- Large LAS point records were not read.
