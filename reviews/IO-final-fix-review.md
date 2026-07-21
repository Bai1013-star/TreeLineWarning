# IO-FINAL-FIX Review

Date: 2026-07-21

## Scope

- Branch: `feature/io`
- Baseline HEAD: `ed610b3 feat: add scene sample integrity validation`
- Python used for all checks: `E:\Miniforge\envs\treeline\python.exe` (Python 3.11.15)
- Scope was limited to the two final-review blockers: environment-path documentation and crop-bound manifest comparison.
- No LAS, NPZ, JSON, HTML, data sample, raw input, preview, or output artifact was generated, overwritten, or modified. No `scan_large.las` point records were opened.

## Modified files

```text
algorithms/io/samples.py
docs/requirements.md
docs/codex_tasks.md
docs/TreeLineWarning_requirements_v2.3.md
docs/TreeLineWarning_codex_tasks_v2.3.md
tests/io/test_scene_sample.py
reviews/IO-final-fix-review.md
```

`reviews/IO-final-review.md` remains unchanged as the historical record of the earlier failed final review.

## Environment path correction

All current-document references to `E:\Miniforge3\envs\treeline` were corrected to `E:\Miniforge\envs\treeline` in the four approved documents. The corresponding `treeline-ml` base path references were corrected consistently as part of the same exact prefix replacement.

Post-fix searches produced no matches:

```powershell
Select-String -Path docs\*.md -Pattern 'E:\Miniforge3'
Get-ChildItem algorithms, core, scripts -Recurse -Filter *.py | Select-String -Pattern 'E:\Miniforge3'
```

Historical review files were not modified. They may retain old environment findings as historical records.

## CropBounds comparison fix

`validate_scene_sample_artifacts` now compares manifest `min_xyz` and `max_xyz` with:

```python
numpy.allclose(..., rtol=0.0, atol=1e-9)
```

- Constant: `CROP_BOUNDS_COMPARISON_ATOL_M = 1e-9` meters.
- The comment documents that this only absorbs JSON floating-point representation noise and is far below the LAS scale of `0.0001` meters.
- Both manifest and expected vectors must have shape `(3,)` and contain finite values; NaN and infinity fail validation.
- `coordinate_mode` remains an exact string comparison.
- Existing output path, output size, point count, index map length/range/uniqueness, and manifest path checks remain strict and unchanged.

## New regression coverage

`tests/io/test_scene_sample.py` now verifies:

- `-3.1609000000000003` in a manifest accepts a caller-provided `-3.1609`.
- A max-coordinate difference of `5e-10` meters is accepted.
- A `1e-6` meter coordinate mismatch is rejected.
- NaN and infinity are rejected.
- `coordinate_mode="local"` is rejected when the expected bound mode is `source`.
- All fixture LAS files are synthetic files under pytest temporary directories.

The existing tests continue to cover chunked sample creation, original-index integrity, output preservation, failure cleanup, and full-read API avoidance. No test reads `scan_large.las`.

## Quality checks

```text
pytest:      77 passed in 1.37s
ruff:        All checks passed!
compileall:  Passed for algorithms, core, tests, and scripts
git diff --check: no whitespace errors (only Git LF-to-CRLF worktree notices)
```

## scene_simple v1 read-only verification

The existing local sample was validated with the approved display bounds directly:

```text
min_xyz = (-11.0, -7.0, -3.1609)
max_xyz = (2.0, 3.0, 9.0)
coordinate_mode = source
```

Results:

- LAS version: 1.4
- Point format: 7
- Point count: 2,008,464
- All sample points are within bounds: true
- Index map length: 2,008,464
- Index map unique: true
- Manifest matches files: true
- Sample size and modification timestamp were unchanged before and after validation: true
- `scan_small.las` and `scan_large.las` both exist; only the sample was read in chunks, and the large raw LAS was not opened for point data.

## Git data isolation

- `git ls-files data outputs` produced no output.
- No tracked file matches LAS, LAZ, PLY, PCD, NPZ, HTML, PT, PTH, or CKPT suffixes.
- `git check-ignore -v` confirms `data/raw/`, `data/samples/`, and `outputs/` still ignore all requested raw, sample, preview, and selection paths.

## Merge blockers and recommendation

The two identified final-review blockers are resolved. No new merge blocker was found in this limited fix.

Recommendation: approve and commit this IO-FINAL-FIX. After that commit, rerun the feature/io final merge review from a clean worktree before merging to `main`. The final review should repeat pytest, ruff, compileall, merge-range diff checks, and Git data-isolation checks.
