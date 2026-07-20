# T4B Review: Low-Density Preview Crop Range Selector

## Branch and Baseline

- Current branch: `feature/io`
- Baseline commit: `8700e55 feat: add streamed point cloud preview generator`
- No commit made for T4B.
- No push performed.

Start checks:

```text
git status --short --branch
## feature/io

git log -5 --oneline
8700e55 feat: add streamed point cloud preview generator
9763d3c feat: add chunked LAS reader and crop engine
80c444d feat: add readonly LAS discovery and header summary
81a6dbb feat: add core contracts and test scaffold
72f9935 chore: initialize repository and add project documentation
```

Worktree was clean before T4B changes.

Read before implementation:

- `docs/requirements.md`
- `docs/codex_tasks.md`
- `algorithms/preprocessing/preview.py`
- `core/contracts/`

## Page Launch Command

```powershell
E:\Miniforge3\envs\treeline\python.exe -m streamlit run scripts/crop_selector.py
```

Environment check:

```text
streamlit 1.59.2
plotly 6.9.0
```

## Actual Modified Files

Added:

```text
scripts/crop_selector.py
tests/scripts/__init__.py
tests/scripts/test_crop_selector.py
reviews/T4B-review.md
```

## Range Selection Method

The page automatically reads only:

```text
outputs/previews/scan_small_preview.npz
outputs/previews/scan_small_preview.json
```

It does not open or read the original LAS. The page displays source-coordinate XYZ range and
provides numeric controls for:

- X min / X max
- Y min / Y max
- Z min / Z max

Filtering rule is inclusive:

```text
min_x <= x <= max_x
min_y <= y <= max_y
min_z <= z <= max_z
```

Displayed values:

- selected preview point count
- selected preview ratio
- current X/Y/Z bounds
- copyable CropBounds JSON

3D view:

- RGB mode when RGB exists
- height fallback
- classification mode when classification exists
- X/Y/Z hover in source coordinates
- local-coordinate axes for display
- crop bounding box trace

The page explicitly warns that this only selects a range on preview data and has not cropped the
original LAS.

Selection guidance included in the page:

- choose a 20-35 m line section
- include multiple intact parallel wires
- include vegetation / tree areas
- avoid the densest left-side pole or mixed area
- do not select the whole scene
- do not select only wires without trees
- inspect the middle-right section but choose manually

## JSON Output Format

Copyable CropBounds JSON:

```json
{
  "min_xyz": [x_min, y_min, z_min],
  "max_xyz": [x_max, y_max, z_max],
  "coordinate_mode": "source",
  "label": "scene_simple"
}
```

Save button target:

```text
outputs/selections/scene_simple_bounds.json
```

Saved selection record fields:

- `source_id`
- `source_path`
- `created_at`
- `preview_point_count`
- `selected_preview_point_count`
- `estimated_keep_ratio`
- `min_xyz`
- `max_xyz`
- `coordinate_mode`
- `label`
- `note`

Safety:

- validates each `min < max`
- only writes under `outputs/selections`
- rejects `data/raw`
- rejects existing output by default

No selection file was created during T4B implementation:

```text
Test-Path outputs\selections\scene_simple_bounds.json
False
```

## Real Preview Load Verification

Command loaded only NPZ and JSON:

```text
preview_point_count: 197301
selected_preview_point_count using full original range: 197301
source_id: raw_scene_01_small_scan_small_las_685cb9a412
source_path metadata string: E:\TreeLineWarning\data\raw\scene_01_small\scan_small.las
```

Default full-range CropBounds:

```json
{
  "min_xyz": [-25.051700592041016, -20.719900131225586, -3.1609000000000003],
  "max_xyz": [21.584299407958987, 18.31899986877442, 13.747100000000001],
  "coordinate_mode": "source",
  "label": "scene_simple"
}
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
collected 57 items

tests\contracts\test_pointcloud_contracts.py ...........                 [ 19%]
tests\contracts\test_progress_contracts.py ......                        [ 29%]
tests\io\test_chunks_and_crop.py .............                           [ 52%]
tests\io\test_discovery.py ..                                            [ 56%]
tests\io\test_headers.py ......                                          [ 66%]
tests\preprocessing\test_preview.py ...........                          [ 85%]
tests\scripts\test_crop_selector.py ........                             [100%]

============================= 57 passed in 0.81s ==============================
```

Covered:

- NPZ and JSON loading.
- min/max range validation.
- range filtering point count.
- inclusive boundary behavior.
- empty selection mask.
- CropBounds JSON format.
- save path restricted to `outputs/selections`.
- no overwrite by default.
- no original LAS read required.
- Streamlit page / figure smoke test.

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
Listing 'tests\scripts'...
Compiling 'tests\scripts\test_crop_selector.py'...
Listing 'scripts'...
```

## Git Status

Command:

```powershell
git status --short --branch
```

Result before writing this review:

```text
## feature/io
?? scripts/crop_selector.py
?? tests/scripts/
```

This review file is added after that status and appears in the final status.

## Git Diff Stat

Command:

```powershell
git diff --stat
```

Result:

```text

```

All T4B files are currently untracked, so there is no tracked diff stat.

## Git Diff Check

Command:

```powershell
git diff --check
```

Result:

```text

```

Exit code: 0.

## Git Ignore Confirmation

Command:

```powershell
git check-ignore -v outputs/selections/scene_simple_bounds.json
```

Result:

```text
.gitignore:4:outputs/  outputs/selections/scene_simple_bounds.json
```

Selection files under `outputs/selections` are ignored and must not be committed.

## Safety Confirmation

- Did not read original LAS point records.
- Did not read `scan_large.las`.
- Did not formally crop `scene_simple.las`.
- Did not modify `data/raw`.
- Did not install, upgrade, or uninstall dependencies.
- Did not implement wires, trees, classification, clearance, risk, or prediction.
- Did not commit or push.

## Known Limits

- Numeric range controls are source-coordinate number inputs rather than drawing interaction.
- The page filters low-density preview points only.
- The saved bounds are intended for a later approved formal crop task.
- Bounding box visualization is a Plotly line trace over the filtered preview view.

## Commit Recommendation

Recommend GPT review T4B. If approved, recommend committing only the script, tests, and review
package. Do not commit any files under `outputs/`.
