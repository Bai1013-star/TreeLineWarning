# feature/io Final Quality Review

Date: 2026-07-21

## Scope and current state

- Branch: `feature/io`
- HEAD: `ed610b3 feat: add scene sample integrity validation`
- Merge base branch: `main` at `72f9935 chore: initialize repository and add project documentation`
- Actual Python: `E:\Miniforge\envs\treeline\python.exe` (Python 3.11.15)
- Initial worktree status: clean (`## feature/io`).
- This review performed checks only. It did not generate or overwrite samples, previews, selections, or any raw LAS file. It did not read any `scan_large.las` point records.

## Branch history

`main..feature/io` contains exactly these eight commits:

1. `ed610b3 feat: add scene sample integrity validation`
2. `723010e feat: add validated scene sample generation`
3. `95bc8ce feat: add interactive point cloud crop selector`
4. `20f03bb feat: add point cloud crop range selector`
5. `8700e55 feat: add streamed point cloud preview generator`
6. `9763d3c feat: add chunked LAS reader and crop engine`
7. `80c444d feat: add readonly LAS discovery and header summary`
8. `81a6dbb feat: add core contracts and test scaffold`

`git diff --check main...feature/io` passed with no output.

## Complete change inventory

`main...feature/io` contains 42 changed files, 8,389 insertions and 3 deletions:

```text
algorithms/__init__.py
algorithms/io/__init__.py
algorithms/io/chunks.py
algorithms/io/crop.py
algorithms/io/discovery.py
algorithms/io/headers.py
algorithms/io/samples.py
algorithms/preprocessing/__init__.py
algorithms/preprocessing/preview.py
core/__init__.py
core/config/__init__.py
core/contracts/__init__.py
core/contracts/_json.py
core/contracts/errors.py
core/contracts/pointcloud.py
core/contracts/progress.py
core/pipeline/__init__.py
docs/codex_tasks.md
docs/requirements.md
pyproject.toml
reviews/T1A-review.md
reviews/T2-review.md
reviews/T3A-review.md
reviews/T4A-review.md
reviews/T4B-review.md
reviews/T5A-review.md
reviews/T5B-review.md
scripts/__init__.py
scripts/crop_selector.py
tests/__init__.py
tests/contracts/__init__.py
tests/contracts/test_pointcloud_contracts.py
tests/contracts/test_progress_contracts.py
tests/io/__init__.py
tests/io/test_chunks_and_crop.py
tests/io/test_discovery.py
tests/io/test_headers.py
tests/io/test_scene_sample.py
tests/preprocessing/__init__.py
tests/preprocessing/test_preview.py
tests/scripts/__init__.py
tests/scripts/test_crop_selector.py
```

The range is limited to project scaffolding, contracts, LAS IO, preview/selection tools, tests, documentation, and review records. No `data/`, `outputs/`, point-cloud, model, cache, or generated binary artifact is part of the diff.

## Quality checks

Commands were executed with `E:\Miniforge\envs\treeline\python.exe`.

```text
pytest:      73 passed in 2.33s
ruff:        All checks passed!
compileall:  Passed for algorithms, core, tests, and scripts
```

No pytest warnings, ruff failures, or compileall failures were reported.

## Git data isolation

- `git ls-files data outputs` produced no output.
- `git ls-files | Select-String -Pattern '\.(las|laz|ply|pcd|npz|html|pt|pth|ckpt)$'` produced no output.
- `.gitignore` correctly excludes:
  - `data/raw/` including both raw LAS inputs;
  - `data/samples/` including `scene_simple.las`, the NPZ index map, and manifest;
  - `outputs/` including preview NPZ/JSON/HTML and range selections.

All requested `git check-ignore -v` calls resolved to those three ignore rules. No data, output, point-cloud, preview, model, or cache artifact is tracked.

## LAS IO safety review

- Search for `laspy.read(`, `reader.read(`, and `open3d.io.read_point_cloud` found no executable use. Matches for the first two are prohibition text in `algorithms/io/headers.py` and historical review records only.
- `algorithms/io/headers.py` uses `laspy.open(...).header` for header summaries.
- `algorithms/io/chunks.py` opens files read-only and iterates with `reader.chunk_iterator(chunk_size)`; it creates `original_index` as `np.arange(start_index, next_start, dtype=np.int64)`.
- `algorithms/io/crop.py` reads and writes in chunks, has inclusive masks (`min <= xyz <= max`), rejects source-as-output, rejects existing outputs unless `overwrite=False` is explicitly overridden, restricts output roots, and removes temporary output after failure.
- `algorithms/io/samples.py` retains the selected global chunk indices in output order and writes them to the independent NPZ sidecar as `int64`; it validates count, range, and uniqueness.
- Preview generation has `overwrite=False` by default, rejects `data/raw` output, uses temporary files and atomic replacement, and cleans temporary files after exceptions.
- `overwrite=True` appears only in tests. Production API defaults are `False`.
- Search for `data/raw` in Python production code found only a preview rejection message, not a write path.
- No tracked file contains `git add .`.

## Public contract review

The approved contracts are present with the expected field names and JSON-safe conversion support:

| Contract | Reviewed fields / behavior |
| --- | --- |
| `PointCloudSource` | `source_id`, `path`, `role`, size/read-only/hash/CRS/metadata; raw source validation and Path serialization are covered by tests. |
| `LasHeaderSummary` | Source identity/path, file/header counts, XYZ range, scale/offset, dimensions, feature flags, estimated memory, JSON-safe metadata. |
| `PointCloudChunk` | `source_id`, chunk/start/count, N x 3 XYZ, global `original_index`, attributes, bounds, and final-chunk marker. |
| `CropBounds` | XYZ bounds, `source`/`local` mode and optional origin; processing masks use inclusive boundaries. |
| `CropResult` | Source/bounds/counts/output/header/dimensions/timestamps/metadata, with output count validation. |
| `UnifiedError` | Code, message, severity, module, recoverability, details, and optional cause. |
| `PipelineProgress` | Pending/running/cancelled/failed/**completed** states, processed/total/percent/message/error metadata validation. |
| `ProgressReporter` / `CancellationToken` | `update(progress)` and `is_cancelled()` / `cancel()` names are consistent across consumers. |

`core/contracts/_json.py` and contract tests cover Path JSON serialization and restoration. These contracts are suitable as the direct input boundary for a later `feature/algorithm` branch.

## Local scene_simple v1 read-only verification

The sample artifacts were checked without regeneration. `validate_scene_sample_artifacts` was run with `CropBounds.from_dict(manifest["crop_bounds"])`, reading the sample in chunks and the sidecar with `allow_pickle=False`.

- Sample: `data/samples/scene_simple.las`
- File size: 120,508,845 bytes
- LAS version / point format / record length: 1.4 / 7 / 60 bytes
- Point count: 2,008,464
- Actual XYZ range:
  - min: `[-10.999700592041014, -6.999900131225585, -1.8827]`
  - max: `[1.9999994079589847, 2.9999998687744167, 9.0]`
- Scale: `[0.0001, 0.0001, 0.0001]`; source offset preserved.
- RGB, intensity, classification, GPS time, return fields, and `NormalX/NormalY/NormalZ` are present. The normals remain declared as Extra Bytes.
- All checked output points are within the manifest crop bounds.
- Index map length: 2,008,464; min: 0; max: 6,888,864; unique: true; no out-of-range values.
- Manifest path, file size, point count, bounds, coordinate mode, and index-map path match the local artifacts when evaluated with the manifest's canonical bounds.
- `scan_small.las` still exists at 414,332,205 bytes, matching the manifest's recorded source size and timestamp. `scan_large.las` still exists at 2,404,477,459 bytes; only filesystem metadata was checked, not its point records.

## Findings and merge blockers

### Blocking: stable environment documentation is inconsistent

The actual verified environment is `E:\Miniforge\envs\treeline`, but these current documentation files still contain `E:\Miniforge3`:

- `docs/requirements.md:45` and `docs/requirements.md:61`
- `docs/codex_tasks.md:67`
- `docs/TreeLineWarning_requirements_v2.3.md:45` and `docs/TreeLineWarning_requirements_v2.3.md:61`
- `docs/TreeLineWarning_codex_tasks_v2.3.md:67`

There are also expected historical references in review records such as `reviews/T1A-review.md`, `reviews/T2-review.md`, `reviews/T3A-review.md`, `reviews/T4A-review.md`, and `reviews/T4B-review.md`; those should remain historical records. The stable-document references above must be corrected in a separately approved documentation-only fix before merging.

### Blocking: v1 artifact validation uses exact float-list equality for bounds

`validate_scene_sample_artifacts` compares manifest bounds with `list(bounds.min_xyz)` and `list(bounds.max_xyz)` using exact equality. The approved displayed lower Z value `-3.1609` becomes different from the canonical manifest serialization `-3.1609000000000003`; passing a newly constructed `CropBounds(..., -3.1609)` therefore raises `ValueError("manifest does not match scene_simple sample artifacts")` even though the numeric range is semantically the same.

The artifact itself is valid when the bounds are reconstructed from the manifest, and all integrity checks then pass. The public validation interface should compare finite coordinates with a documented tolerance or normalize its serialized bounds before `feature/io` is merged. Add a regression test for the approved displayed bound value.

## Known limitations

- `scene_simple` v1 is accepted; no v2 sample was created.
- Preview data is intentionally low density and is not a substitute for full-resolution spatial analysis.
- LAS processing intentionally remains limited to header-only or chunked access; no Open3D or complete LAS read path is implemented.
- No wire, vegetation, distance, risk, prediction, or UI workflow beyond range selection is in scope.

## Merge recommendation

Do **not** merge `feature/io` into `main` yet. Resolve the two blockers above through a narrowly scoped, approved fix and repeat this final quality review.

After approval, recommended merge commands are:

```powershell
git switch main
git merge --no-ff feature/io -m "merge: integrate feature/io"
E:\Miniforge\envs\treeline\python.exe -m pytest
E:\Miniforge\envs\treeline\python.exe -m ruff check .
E:\Miniforge\envs\treeline\python.exe -m compileall algorithms core tests scripts
git status --short --branch
git log --oneline --decorate --graph -12
git ls-files data outputs
```

Only after those checks pass and the merged worktree is clean should a separate `feature/algorithm` branch be created from `main`.
