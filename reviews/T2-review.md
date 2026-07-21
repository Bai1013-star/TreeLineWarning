# T2 Review: LAS File Discovery and Read-Only Header Summary

## Branch and Baseline

- Current branch: `feature/io`
- Baseline commit: `81a6dbb feat: add core contracts and test scaffold`
- No commit made for T2.
- No push performed.

## Start Checks

```text
git status --short --branch
## feature/io
```

```text
git log -2 --oneline
81a6dbb feat: add core contracts and test scaffold
72f9935 chore: initialize repository and add project documentation
```

The worktree was clean before T2 changes.

Read before implementation:

- `docs/requirements.md`
- `docs/codex_tasks.md`
- `core/contracts/pointcloud.py`
- `core/contracts/errors.py`
- `core/contracts/__init__.py`
- `algorithms/io/__init__.py`

Confirmed dependency:

```text
laspy 2.7.0
```

## Actual Changes

Modified:

```text
algorithms/io/__init__.py
```

Added:

```text
algorithms/io/discovery.py
algorithms/io/headers.py
tests/io/test_discovery.py
tests/io/test_headers.py
reviews/T2-review.md
```

## LAS Discovery Interface

Public function:

```python
discover_las_sources(raw_root: Path | str) -> list[PointCloudSource]
```

Behavior:

- Recursively discovers `.las` and `.laz` files under the given root.
- Skips non-files, symlinks, and unrelated extensions.
- Does not open LAS files.
- Returns stable sorted results by relative path.
- Builds stable source ids from relative path plus SHA-1 prefix.
- Returns `PointCloudSource` with:
  - `role="raw"`
  - `readonly_required=True`
  - `path`
  - `size_bytes`

Real `data/raw` discovery result:

```text
raw_scene_01_small_scan_small_las_685cb9a412
  path: data\raw\scene_01_small\scan_small.las
  size_bytes: 414332205
  role: raw
  readonly_required: true

raw_scene_02_large_scan_large_las_f11f671723
  path: data\raw\scene_02_large\scan_large.las
  size_bytes: 2404477459
  role: raw
  readonly_required: true
```

## Header Reading Interface

Public functions:

```python
read_las_header_summary(path: Path | str, source_id: str | None = None) -> LasHeaderSummary
read_las_header_summary_for_source(source: PointCloudSource) -> LasHeaderSummary
```

Error type:

```python
LasIoError
```

`LasIoError.error` contains a `UnifiedError` with code, message, module, recoverability,
details, and cause when available.

Implementation notes:

- Uses `laspy.open(path, mode="r")` and `reader.header`.
- Does not call `laspy.read()`.
- Does not call `reader.read()`.
- Does not access `las.x`, `las.y`, `las.z`, or any complete point arrays.
- Does not import or use Open3D.
- Does not implement chunk reading or crop output.
- Stores header warnings and VLR summaries in `LasHeaderSummary.metadata`.

Input guards:

- Missing file -> `LAS_INPUT_NOT_FOUND`
- Directory input -> `LAS_INPUT_IS_DIRECTORY`
- Non `.las` / `.laz` extension -> `LAS_INPUT_EXTENSION_INVALID`
- Header parse failure -> `LAS_HEADER_READ_FAILED`
- Empty LAS -> warning in metadata, no crash

## Real LAS Read-Only Header Summaries

Both real files were read only through `laspy.open(...).header`.
No point arrays or chunks were read.

### scene_01_small / scan_small.las

Before:

```text
size: 414332205
mtime_ns: 1784534268000000000
```

Summary:

```text
source_id: raw_scene_01_small_scan_small_las_685cb9a412
las_version: 1.4
point_format_id: 7
point_record_length_bytes: 60
point_count: 6905520
min_xyz: [-25.051700592041016, -20.719900131225586, -3.1609000000000003]
max_xyz: [21.584299407958987, 18.31899986877442, 13.747100000000001]
scale: [0.0001, 0.0001, 0.0001]
offset: [-25.051700592041016, -20.719900131225586, 0.0]
has_rgb: true
has_intensity: true
has_classification: true
extra_dimensions: [NormalX, NormalY, NormalZ]
estimated_record_memory_bytes: 414331200
warnings: []
vlrs:
  - ExtraBytesVlr, user_id=LASF_Spec, record_id=4
```

After:

```text
size: 414332205
mtime_ns: 1784534268000000000
unchanged: true
```

### scene_02_large / scan_large.las

Before:

```text
size: 2404477459
mtime_ns: 1784534126000000000
```

Summary:

```text
source_id: raw_scene_02_large_scan_large_las_f11f671723
las_version: 1.4
point_format_id: 7
point_record_length_bytes: 60
point_count: 40074605
min_xyz: [-47.18149948120117, -45.37910079956055, -3.1393]
max_xyz: [23.466600518798828, 46.96219920043946, 13.114]
scale: [0.0001, 0.0001, 0.0001]
offset: [-47.18149948120117, -45.37910079956055, 0.0]
has_rgb: true
has_intensity: true
has_classification: true
extra_dimensions: [NormalX, NormalY, NormalZ]
estimated_record_memory_bytes: 2404476300
warnings: []
vlrs:
  - WktCoordinateSystemVlr, user_id=LASF_Projection, record_id=2112
  - ExtraBytesVlr, user_id=LASF_Spec, record_id=4
```

After:

```text
size: 2404477459
mtime_ns: 1784534126000000000
unchanged: true
```

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
collected 25 items

tests\contracts\test_pointcloud_contracts.py ...........                 [ 44%]
tests\contracts\test_progress_contracts.py ......                        [ 68%]
tests\io\test_discovery.py ..                                            [ 76%]
tests\io\test_headers.py ......                                          [100%]

============================= 25 passed in 0.49s ==============================
```

Coverage added:

- Recursive LAS discovery.
- Ignoring non-point-cloud files.
- Stable discovery order.
- LAS 1.4 header reading from synthetic temp files.
- Point format, point count, scale, offset.
- RGB, intensity, classification detection.
- Extra Bytes detection.
- Missing file, wrong extension, directory input.
- Empty LAS warning.
- Path and summary JSON serialization.
- Monkeypatch guard confirming summary does not call full point read interfaces.

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
?? algorithms/io/discovery.py
?? algorithms/io/headers.py
?? reviews/T2-review.md
?? tests/io/test_discovery.py
?? tests/io/test_headers.py
```

## Git Diff Stat

Command:

```powershell
git diff --stat
```

Result:

```text
 algorithms/io/__init__.py | 17 ++++++++++++++---
 1 file changed, 14 insertions(+), 3 deletions(-)
warning: in the working copy of 'algorithms/io/__init__.py', LF will be replaced by CRLF the next time Git touches it
```

Note: new files are untracked and therefore not included in `git diff --stat`.
Untracked T2 files:

```text
algorithms/io/discovery.py
algorithms/io/headers.py
reviews/T2-review.md
tests/io/test_discovery.py
tests/io/test_headers.py
```

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

- No chunk reading.
- No crop implementation.
- No `data/samples` output generation.
- Header VLR summary captures only safe metadata fields, not VLR payload bytes.
- `.laz` discovery is supported; `.laz` header reading depends on `laspy.open` support in the environment.
- `LasHeaderSummary.metadata` carries `warnings` and `vlrs` because T1A intentionally kept the public contract minimal.

## LAS Safety Confirmation

- Did not read complete LAS files.
- Did not read LAS point arrays.
- Did not perform chunk reading.
- Did not load Open3D.
- Did not create `data/samples`.
- Did not write to `data/raw`.
- Did not modify, move, delete, rename, overwrite, or change attributes of original LAS files.
- Real LAS size and modification timestamps were unchanged before and after header reads.

## Commit Recommendation

Recommend GPT review T2. If approved, recommend commit T2 because:

- pytest: 25 passed.
- ruff: All checks passed.
- compileall: passed.
- Real header-only reads succeeded on both approved LAS files.
- Original LAS files were unchanged.
