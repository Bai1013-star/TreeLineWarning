# Validation record

Environment: existing `E:/TreeLineWarning-fast/.venv311/Scripts/python.exe`;
worktree `E:/TreeLineWarning-tree-geometry-qc-v1`. No dependency installation.

`python -m ruff check research/geometry_qc_v1`: PASS.
`python -m pytest -q research/geometry_qc_v1/test_audit.py`: 7 passed (final run 0.54 s).
`git diff 5f6f4d1 --exit-code -- algorithms core desktop viewer_poc tests desktop_build.spec`:
PASS, empty diff. `git diff --check`: PASS.

Combined regression command (119 passed, 1 failed, 14.24 s):

```powershell
E:/TreeLineWarning-fast/.venv311/Scripts/python.exe -m pytest -q research/geometry_qc_v1/test_audit.py tests/spans/test_stem_guided_hybrid.py tests/stage_b/test_dbh_recovery.py tests/stage_b/test_dbh_raw_recovery.py tests/stage_b/test_dbh_identity_recovery.py tests/stage_b/test_dbh_review.py tests/stage_b/test_editor_clearance_invalidation.py tests/stage_b/test_clearance_order_publication.py tests/stage_b/test_tree_metrics_export.py tests/stage_b/test_tree_display_export.py tests/folder_pipeline/test_tree_editor.py tests/folder_pipeline/test_restart_hydration.py tests/project/test_multispan_fast_track.py tests/desktop/test_risk_threshold_reclassification.py tests/desktop/test_prediction_export_contract.py --basetemp E:/TreeLineWarning-geometry-qc-regression
```

Only failure:
`tests/stage_b/test_tree_display_export.py::test_stage_b_export_adds_display_id_without_replacing_persistent_id`.
Assertion at line 88 expects `headers[33] == 'display_tree_id'`, actual `dbh_cm`.
Independent baseline check from `E:/TreeLineWarning-fast-performance-finalization-v1`
at frozen HEAD, after checking empty product/test diff against `5f6f4d1`:

```powershell
E:/TreeLineWarning-fast/.venv311/Scripts/python.exe -m pytest -q tests/stage_b/test_tree_display_export.py::test_stage_b_export_adds_display_id_without_replacing_persistent_id --basetemp E:/TreeLineWarning-geometry-qc-baseline-check
```

Result: same failure, 2.47 s. No frozen test/product code modified.

Real evidence audit: 16 pre/post hash pairs and exact index-removal replay PASS.
The 315-point archived reference has no membership arrays; not falsely counted PASS.
Eight bad image contact-sheet hashes and pre-edit revisions verified.
Before/after frozen package hashes match (see adjacent JSON files).

New product QC restart/invalidation/delete/merge/undo/multi-span/fallback, dedicated
stem/crown/neighbor classifiers and UI integration tests are NOT IMPLEMENTED: the
mandatory usefulness gate stopped that work. Existing regressions cover their current
product behaviors; they do not establish acceptance of a new QC feature.
