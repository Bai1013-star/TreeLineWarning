# TREE_LINE_WARNING_RELEASE_FINAL_REPORT

Generated: 2026-09-15 (local) · Project: `E:\TreeLineWarning-fast`
Acceptance scope: Windows x64 ONEDIR packaged release, Survey Core baseline.
Handover note: this session continued an in-progress Codex acceptance loop. No
architecture, UI, framework, data-format, or algorithm change was made.

---

## CURRENT_HEAD

```
64ff43820aa51877d800b387c2ca063c1be7b5e2
fix(review): keep the first review read alive across a cold display hydration
branch: codex/final-deliverable-20260830
parent: 606127c3f3a4a8ad355b3053ca7069d25148e1ee (Stabilize final-state publication and runtime evidence)
```

- Working tree: **clean** — zero tracked modifications remaining.
- Remaining `git status` entries are **untracked test scratch directories only**
  (`.test-tmp-*`, `.final-test-tmp-*`); none are release inputs.
- **Discrepancy vs the task brief:** the brief named branch
  `feature/fast-research-pipeline`. The release work was actually performed on
  `codex/final-deliverable-20260830`, and that is the branch the shipped EXE was
  built from. No branch switch was performed — switching would have orphaned the
  build. `feature/fast-research-pipeline` exists but is unrelated to this package.

**Commit made this session.** The release EXE had been built from an *uncommitted*
worktree, so the artifact was not reproducible from any commit. Three files were
committed (61 insertions, 19 deletions):

| File | Change |
|---|---|
| `desktop/server.py` | Serialize the process-local display cache fill behind the existing `workflow._lock` (`threading.RLock`) so a manual iframe refresh cannot race an in-flight read and turn a valid read into HTTP 409. |
| `viewer_poc/giro3d/tree_review_app.js` | Display reads get their own `DISPLAY_REQUEST_TIMEOUT_MS = 180000` instead of the 15 s edit-mutation deadline; 15 s slow-load notice; server error text surfaced instead of a generic message. |
| `tests/folder_pipeline/test_step2_viewer_contract.py` | New test pinning the display-deadline contract. |

Deadlock risk of the new `self.workflow._lock` usage was checked explicitly:
`workflow._lock` is a **reentrant** `RLock` (`desktop/workflow.py:388`), and
`editor_data()` re-enters it via `_stage_project()`. Same-thread re-entry is safe.

**Git closure gap (open):** the branch has **no upstream** and is **349 commits
ahead of `origin/main`**. The deliverable commit exists only on this machine.
Pushing is the remaining closure step and was not performed (no remote was
specified as authoritative).

---

## BUILD_STATUS

**BUILT** — PyInstaller `--noconfirm`, ONEDIR, `returncode 0`, no timeout,
`source_changed_during_run: false`, build elapsed 104.5 s.

Source of truth: `artifacts/review-load-fix-dist-20260915/TreeLineWarning-Engineering-Release-2026-Final/build_receipt.json`

- `package_status`: BUILT · `package_type`: Windows x64 ONEDIR
- `baseline`: Survey Core · `preview`: EXPERIMENTAL_NOT_ACCEPTED
- `gpu`: OPTIONAL_CUPY_CUDA13_CERTIFIED_SCREEN_CPU_EXACT_AUTHORITY
  (CUDA toolkit not required; CPU fallback authoritative)
- Native runtime: GDAL 3.13.1, PDAL 2.10.2, pdal-python 3.5.4, PROJ 9.8.1 (68 files)
- PROJ data: 16 files · WebView2 offline installer bundled, `executed: false`
- Python 3.11.9 from `.venv311`, `include_system_site_packages: false`

### Package integrity — independently verified this session

Full SHA-256 re-verification of every manifested file:

```
entries            : 2860
verified ok        : 2860
missing            : 0
size mismatches    : 0
sha256 mismatches  : 0
total payload      : 2,257,444,457 bytes = 2.102 GiB
VERDICT            : INTEGRITY_OK
```

### Build provenance — verified by three independent checks

1. Recomputing `tracked_source_digest()` over the **build checkout**
   (`E:\TreeLineWarning-review-fix-build-20260915`) reproduces the receipt value
   exactly: `74fe2bc24f3cd0fc2d7645635d69e632eae92dd123c6095b8ad9616eebbd6e47`,
   651 tracked files. This proves which tree produced the EXE.
2. The committed blobs of `desktop/server.py` and
   `viewer_poc/giro3d/tree_review_app.js` are **byte-identical to the built
   files after CRLF normalisation** — the commit captures exactly what shipped.
3. The packaged `_internal/viewer_poc/giro3d/tree_review_app.js` equals the
   committed source after CRLF normalisation, and contains
   `const DISPLAY_REQUEST_TIMEOUT_MS = 180000;`.

> **Caveat worth recording:** `tracked_source_sha256` is a digest of **worktree
> bytes** (`scripts/build_desktop_release.py:189`, "tracked worktree bytes"), so it
> is **line-ending sensitive and not reproducible from a git commit**. The build
> checkout used CRLF; `TreeLineWarning-fast` uses LF, so the same source yields
> `287a5a71…` instead of `74fe2bc2…`. This is a line-ending difference, **not**
> code drift — confirmed by normalised comparison. Treat this receipt field as a
> worktree fingerprint, not a commit fingerprint.

---

## EXE_PATH

```
E:\TreeLineWarning-fast\artifacts\review-load-fix-dist-20260915\
  TreeLineWarning-Engineering-Release-2026-Final\TreeLineWarning.exe
```

- Size 24,272,711 bytes (23.1 MB) · built 2026-09-15 20:51:26
- SHA-256 `D136C67CAE928F993044B7254898E8E96E9F960368BE8C8376928310B255FA76`
- Package root: same directory (2861 files, 2.15 GB incl. manifest)
- Standalone start: **verified** — launched from its own directory with no Python,
  Conda, PDAL or developer environment present, and served HTTP successfully.
- WebView2 offline prerequisite:
  `webview2_offline_prerequisite\offline_verified\MicrosoftEdgeWebView2RuntimeInstallerX64.exe`
  (246.5 MB, bundled, not auto-installed)

---

## SMOKE_TEST

**PASS** — real packaged EXE, fresh process, real project `E:\3\3`, `--backend-smoke`
mode. Three runs executed; run 3 is the authoritative full-surface run.

```
backend_publish          : PASS  (http://127.0.0.1:64070)
first_review_display     : PASS  HTTP 200, 10 trees, 50000 display points
editor_data_shape        : PASS  trees3d.xyz + tree_id_by_index present
no_undefined_in_display  : PASS  no literal "undefined" in payload
concurrent_review_reads  : PASS  statuses=[200, 200], trees=[10, 10]
no_409_concurrent        : PASS  statuses=[200, 200]
tree_list                : PASS  HTTP 200, 10 trees, edit_count=3
static assets (5/5)      : PASS  /, tree-review-app.js, giro3d.bundle.js, /step2, current_report.js
packaged_frontend_deadline: PASS 180 s display deadline present
probes                   : 32 total, 31 PASS
```

- **No 404, no undefined, no crash** on any exercised surface.
- The single non-pass (`editor_component_preview`, see EDITOR_TEST) is a
  **by-design guard triggered by a synthetic seed**, not a defect.

Cold first-load latency, measured across three fresh processes:

| Run | First review read | Display points |
|---|---|---|
| 1 | 43.9 s | (probe mis-derived) |
| 2 | 69.5 s | 49,998 |
| 3 | 130.4 s | **50,000** |

Variance is OS file-cache dependent. **The 15 s mutation deadline would have
aborted every one of these reads** — this is direct confirmation that the fix
addresses the real failure, and that a 180 s window is load-bearing, not cosmetic.

---

## API_TEST

**PASS** — all read surfaces returned HTTP 200:

```
/state                       200
/api/desktop/project         200
/api/desktop/workspace       200
/api/desktop/capability      200
/api/desktop/topology        200
/api/desktop/batch           200
/api/final                   200
/api/desktop/current-report  200
/api/desktop/dependency-health 200
/api/dbh-review/trees        200
/api/viewer/runtime          200
/api/editor_data             200 (cold + 2 concurrent)
```

Concurrency result reproduces the previously reported acceptance:

```
two simultaneous /api/editor_data?display_budget=50000
  → HTTP 200, HTTP 200
  → 10 trees each, 50000 display points, 50000 display points
  → NO 409, NO broken pipe, NO duplicate-load, NO crash
```

Dependency health confirms the packaged runtime resolves from `_internal`
(`PyiFrozenLoader`) and GPU certification passes with CPU-exact match
(`effective_mode: GPU_CERTIFIED_CPU_EXACT`, RTX 5070 Laptop, 0 fallback points).

---

## EDITOR_TEST

**PASS** — reversible mutation round-trip executed against the real project:

```
editor_select        : PASS  HTTP 200, tree 11 selected
editor_delete        : PASS  HTTP 200
editor_delete_effect : PASS  trees 10 -> 9
editor_undo          : PASS  HTTP 200
editor_round_trip    : PASS  trees restored 10 -> 9 -> 10
```

Tree list read back correctly (10 trees, `tree_edit_count=3`), and the
non-mutating DBH review surface `/api/dbh-review/trees` returns 200. The project
was returned to its original tree count; no data loss.

**One non-pass, triaged as expected behaviour, not a defect.**
`POST /api/trim/component-preview` returned **HTTP 409**. Investigation:

- `ValueError` → `HTTPStatus.CONFLICT` is the server's deliberate validation
  mapping (`desktop/server.py:155`, `_exception_error`).
- `preview_trim_component` raises user-facing guards including
  *"点击位置未命中当前单木，请重新点击"* (seed outside the tree) and
  *"该区域与树体主体连接较紧密…建议使用‘套索剔除’"* (component too large a
  fraction of the tree) — `algorithms/folder_pipeline/tree_editor.py:500-528`.
- The probe fed a **decimated 3D display point** as the click seed, not a real
  UI ray-pick. Landing in the trunk component is the expected outcome for such a
  seed, which is precisely what the second guard rejects with an actionable
  message.

So this is a **correct, actionable rejection of a synthetic click**, and it is
also *not* the duplicate-load 409 the release fix targets. A genuine UI click
seed was not reproduced in this headless harness; that remains the one editor
path not positively demonstrated end-to-end.

---

## EXPORT_TEST

**PASS**

```
/download_prediction   → HTTP 200, 12,066 bytes, ZIP/XLSX magic "PK" ✓
/download_report_html  → HTTP 200, 2,696,562 bytes, HTML ✓
```

Both downloads were served by the packaged EXE over HTTP and validated by
content, not merely by status code. Export depends on the editor's finalized
state and therefore also exercises the Stage-B finalization boundary.

---

## KNOWN_LIMITATIONS

1. **Git closure incomplete — branch not pushed.** `codex/final-deliverable-20260830`
   has no upstream and is 349 commits ahead of `origin/main`. Commit `64ff438`
   exists only locally. No remote was nominated as authoritative, so nothing was
   pushed.
2. **Four pre-existing unit-test failures, unrelated to this release.**
   `tests/desktop/test_future_preview.py`: 4 failed at the **pristine HEAD
   baseline** (`git archive HEAD` into a clean tree → 4 failed, 196 passed) and
   4 failed on the working tree (4 failed, 197 passed — the extra pass being the
   newly added contract test). They are **not regressions**. They are confined to
   `/api/future-preview`, which the build receipt itself marks
   `preview: EXPERIMENTAL_NOT_ACCEPTED`. Root cause: the route returns
   `status: LOCKED` ("当前项目正在处理其他操作") when it cannot take
   `workflow._lock` non-blockingly (`desktop/future_preview.py:156-158`), while
   the tests assert `PASS`. Reproduces in isolation; left untouched per the
   "do not delete or rewrite existing tests" constraint.
3. **Cold-start latency has a thin margin.** First review read reached **130.4 s**
   on a cold OS cache against the new 180 s frontend deadline — roughly a 50 s
   margin. On a slower disk or a larger survey this could still exceed 180 s and
   surface the (now clear) "数据准备超时" message. Worth watching; not a blocker.
4. **`tracked_source_sha256` is not commit-reproducible** (line-ending sensitive
   worktree digest). See BUILD_STATUS caveat. Recommend switching this field to a
   commit-blob digest in a future release.
5. **`clean_pc: NOT_TESTED`.** WebView2/.NET launch on a clean machine was not
   tested; the offline installer is bundled but never auto-executed. Unchanged
   from the prior contract.
6. **`unresolved_provenance_count: 31`** in third-party notices. Unchanged.
7. **Trim-component preview not demonstrated with a real UI click seed** (see
   EDITOR_TEST). The route responds correctly, but the happy path was not
   exercised headlessly.
8. **Stale processes left by the earlier session.** Two `TreeLineWarning.exe`
   backends from 20:34 were still running and holding state; one further orphan
   was created by a probe crash in this session. **All three were terminated**
   and their port released, so the next launch starts clean.

---

## FINAL_STATUS

```
RELEASE_READY — GO
```

The packaged artifact is built, complete, integrity-verified, provenance-verified,
and functionally accepted against the real project with the real EXE:

- package integrity 2860/2860 files, 2.102 GiB, zero mismatches
- EXE starts standalone and serves HTTP with no developer environment
- cold review read survives (43.9 s / 69.5 s / 130.4 s) where the old 15 s
  deadline always failed
- concurrent reads: 200/200, **no 409**, 10 trees, 50,000 display points each
- tree list, reversible edit round-trip (10→9→10), and both exports pass
- no 404, no undefined, no crash on any exercised surface

Remaining closure items are **not** code blockers: push the branch to the
authoritative remote (limitation 1), and optionally address the pre-existing
`future-preview` test failures and the 180 s latency margin.

### Evidence

| Artifact | Path |
|---|---|
| Full smoke report (v3) | `artifacts/review-load-fix-final-smoke2-20260915/smoke_report.json` |
| Smoke run 3 backend | `artifacts/review-load-fix-final-smoke2-20260915/backend.json` |
| Smoke run 2 (partial) | `artifacts/review-load-fix-final-smoke-20260915/` |
| Earlier session smoke | `artifacts/review-load-fix-acceptance-20260915/` |
| Build receipt | `artifacts/review-load-fix-dist-20260915/TreeLineWarning-Engineering-Release-2026-Final/build_receipt.json` |
| Release manifest | `artifacts/review-load-fix-dist-20260915/TreeLineWarning-Engineering-Release-2026-Final/release_file_manifest.json` |
| Dependency health | `artifacts/review-load-fix-acceptance-20260915/dependency-health.json` |
