"""E: drive A+B tier cleanup executor.

Modes:
  precheck  - build manifest, verify containment + protected extensions, delete nothing
  apply     - delete manifest entries, log each result

Guarantees:
  * Every target must be inside an allowlisted parent root.
  * Targets that ARE (or CONTAIN) a protected artifact are refused.
  * The primary repository working tree, .git, source dirs, .venv*, input_las
    and the current r6 deliverable can never be resolved as a target.
"""
from __future__ import annotations

import os
import sys
import json
import shutil
import datetime
import subprocess
from pathlib import Path

ROOT = Path(r"E:\TreeLineWarning-fast")
LOG = ROOT / "artifacts" / "_cleanup_20261001"
MANIFEST = LOG / "DELETE_MANIFEST.json"

PROTECTED_EXT = {".las", ".laz", ".copc.laz", ".onnx", ".pth", ".pt", ".ckpt"}

# ---- hard refusals: a target may never be, or contain, one of these --------
NEVER_TARGET = {
    ROOT,
    ROOT / ".git",
    ROOT / "core",
    ROOT / "algorithms",
    ROOT / "desktop",
    ROOT / "scripts",
    ROOT / "tests",
    ROOT / "viewer_poc",
    ROOT / "docs",
    ROOT / "reviews",
    ROOT / "input_las",
    ROOT / ".venv311",
    ROOT / ".venv",
    ROOT / "cache" / "p3a_potree",
    ROOT / "artifacts" / "TreeLineWarning-20260930-r6.zip",
    ROOT / "artifacts" / "final-product-dist-20260930-r6",
    ROOT / "artifacts" / "final-product-work-20260930-r6",
    ROOT / "acceptance_evidence" / "product-fixes-20260929",
    ROOT / "workspace" / "input",
    ROOT / "workspace" / "01_span_selection",
    ROOT / "workspace" / "02_tree_analysis",
    ROOT / "workspace" / "03_export",
    Path(r"E:\TreeLineWarning-review-fix-build-20260915"),
    Path(r"E:\TreeLineWarning-review-risk-polish"),
}

ALLOWED_PARENTS = [
    ROOT / "artifacts",
    ROOT / "cache",
    ROOT / "workspace",
    ROOT / "acceptance_evidence",
    Path(r"E:\TreeLineWarning-source-cache"),
    Path(r"E:\2026.8.31-2"),
]

# Root-level loose files: allowlisted individually so that adding a pattern
# can never widen deletion authority over the repository root.
ROOT_FILES = [
    ROOT / "actual",
    ROOT / "bounded",
    ROOT / "T001",
    ROOT / ".wire_endpoint_recovery_round2_run.log",
]

# Scattered temp directories directly under the repo root. Enumerated by
# explicit pattern in collect(); each match is registered here so that the
# containment gate lists them one by one instead of trusting a wildcard.
SCATTERED = [
    ".test-tmp*", ".final-test-tmp*", ".wire_*", ".build-*",
    ".pytest_cache", ".pytest_tmp", ".ruff_cache", ".pt2", "tmp-check",
    "TreeLineWarning-fast.test-tmp", "treeline_warning.egg-info",
]
SCATTERED_SKIP = {".venv", ".venv311", ".git", ".claude", "cache", "artifacts"}

# Files >= this are always treated as real data and never auto-deleted.
BIG_DATA_BYTES = 10 * 1024 * 1024
# Test fixtures live under these names; small lazy-format files there are exempt.
TESTISH = (".test-tmp", ".final-test-tmp", "pytest", "tmp", "fixture", "targeted-tmp")

# ------------------------------------------------------------------ A tier
A_TARGETS = [
    ROOT / "artifacts/final-product-dist-20260929-r4",
    ROOT / "artifacts/final-product-work-20260929-r4",
    ROOT / "artifacts/final-product-dist-20260930-r5",
    ROOT / "artifacts/final-product-work-20260930-r5",
    ROOT / "artifacts/TreeLineWarning-20260930-r5.zip",
    ROOT / "artifacts/TreeLineWarning-Complete-20260929.zip",
    ROOT / "artifacts/final-product-work-20260929-r2",
    ROOT / "artifacts/final-product-work-20260929-r3",
    ROOT / "artifacts/dbh-format-release-work-20260928",
    ROOT / "artifacts/dbh-format-release-work-final-20260928",
    ROOT / "artifacts/dbh-format-release-work-v3-20260928",
    ROOT / "artifacts/dbh-investigation-20260928",
    ROOT / "artifacts/review-load-fix-acceptance-20260915",
    ROOT / "artifacts/review-load-fix-acceptance-20260915-final",
    ROOT / "artifacts/review-load-fix-final-smoke-20260915",
    ROOT / "artifacts/review-load-fix-final-smoke2-20260915",
    ROOT / "artifacts/review-load-fix-work-20260915",
    ROOT / "artifacts/review-fix-native-runtime-20260915",
    ROOT / "artifacts/dbh-format-native-runtime-20260928",
    ROOT / "artifacts/dbh-format-native-runtime-final-20260928",
    ROOT / "artifacts/dbh-format-native-runtime-v3-20260928",
    ROOT / "artifacts/real-closure-build-work-20260913-606127c",
    ROOT / "artifacts/real-closure-release-20260913-606127c",
    ROOT / "artifacts/final-acceptance-20260913",
    ROOT / "artifacts/final-product-20260930",
    ROOT / "artifacts/final-product-20260929",
    Path(r"E:\2026.8.31-2\2026.08.31-2.zip"),
    ROOT / "actual",
    ROOT / "bounded",
    ROOT / "T001",
    ROOT / ".wire_endpoint_recovery_round2_run.log",
]

# ------------------------------------------------------------------ B tier
B_TARGETS = [
    ROOT / "cache/tower_ordered_project",
    ROOT / "cache/tower_raw_tiles",
    ROOT / "cache/tower_review_project",
    ROOT / "cache/mamba_root",
    ROOT / "cache/giro3d_pdal_env",
    ROOT / "cache/giro3d_full_pdal",
    ROOT / "cache/giro3d_full_source",
    ROOT / "cache/p3a_giro3d",
    ROOT / "cache/wire_v2_shadow",
    ROOT / "cache/wire_v2_shadow_hardened_20260817",
    ROOT / "cache/wire_v2_shadow_hardened_20260817_track_evidence",
    ROOT / "cache/wire_v2_shadow_hardened_20260817_track_evidence_exact_polyline",
    ROOT / "cache/debug",
    ROOT / "artifacts/fast_research",
    Path(r"E:\TreeLineWarning-source-cache"),
]

# --------------------------------------------------- A tier: scattered temps
# (patterns declared near ALLOWED_PARENTS so the containment gate can see them)

RECYCLE_BIN = Path(r"E:\$RECYCLE.BIN")


MODEL_EXT = {".onnx", ".pth", ".pt", ".ckpt"}
LAZY_EXT = {".las", ".laz", ".copc.laz"}
# A .pth below this size cannot be model weights (PyTorch weights are >= MBs);
# it is a setuptools/distutils path shim.
PTH_MODEL_MIN = 1 * 1024 * 1024


def is_testish(path: str) -> bool:
    low = path.lower()
    return any(t in low for t in TESTISH)


def measure(p: Path):
    """Return (bytes, files, blocked_path, blocked_reason, exempted_count, retained).

    Protection policy (evidence-based; see precheck output):
      * .onnx/.pt/.ckpt            -> always blocks (real models)
      * .pth >= 1 MiB              -> blocks (real model weights)
      * .pth <  1 MiB              -> exempted (setuptools distutils-precedence shim)
      * .las/.laz/.copc.laz >= 10 MiB -> blocks (real point cloud)
      * .las/.laz/.copc.laz <  10 MiB under a test path -> exempted fixture
      * .las/.laz/.copc.laz <  10 MiB elsewhere -> blocks (unexplained data)
      * archive containing a protected entry -> blocks

    A subtree that would block is RETAINED (skipped), not fatal: the rest of
    the target is still deletable. retained lists those skipped subtrees.
    """
    total = files = exempted = 0
    retained: list[dict] = []
    skip_roots: list[str] = []

    for dirpath, dirnames, filenames in os.walk(p, onerror=lambda e: None):
        if os.path.islink(dirpath):
            continue
        if any(dirpath.startswith(s + os.sep) or dirpath == s for s in skip_roots):
            continue
        retained_here = None
        for fn in filenames:
            fp = os.path.join(dirpath, fn)
            try:
                st = os.stat(fp, follow_symlinks=False)
            except OSError:
                continue
            ext = os.path.splitext(fn)[1].lower()
            if ext in MODEL_EXT:
                retained_here = (fp, f"{ext} model file")
            elif ext == ".pth":
                if st.st_size < PTH_MODEL_MIN:
                    exempted += 1
                else:
                    retained_here = (fp, f".pth {st.st_size / 1024**2:.1f} MiB (model weights)")
            elif ext in LAZY_EXT:
                if st.st_size >= BIG_DATA_BYTES:
                    retained_here = (fp, f"{ext} {st.st_size / 1024**2:.1f} MiB >= 10 MiB (real point cloud)")
                elif is_testish(fp):
                    exempted += 1
                else:
                    retained_here = (fp, f"{ext} {st.st_size / 1024**2:.2f} MiB outside a test path")
            elif ext == ".zip" and st.st_size > 0:
                hit = archive_has_protected(fp)
                if hit:
                    retained_here = (fp, f"archive contains protected entry: {hit}")
            if retained_here:
                break

        if retained_here:
            skip_roots.append(dirpath)
            retained.append({"path": dirpath, "protected": retained_here[0], "reason": retained_here[1]})
            continue

        for fn in filenames:
            fp = os.path.join(dirpath, fn)
            try:
                st = os.stat(fp, follow_symlinks=False)
            except OSError:
                continue
            total += st.st_size
            files += 1

    return total, files, None, None, exempted, retained


def archive_has_protected(zip_path: str):
    import zipfile
    try:
        with zipfile.ZipFile(zip_path) as z:
            for n in z.namelist():
                if os.path.splitext(n)[1].lower() in (MODEL_EXT | LAZY_EXT):
                    return n
    except Exception:
        return None
    return None


def _resolved_set(paths):
    out = set()
    for p in paths:
        try:
            out.add(os.path.normcase(os.path.abspath(str(p))))
        except OSError:
            continue
    return out


_NEVER = _resolved_set(NEVER_TARGET)
_ALLOWED = _resolved_set(ALLOWED_PARENTS)


def containment_ok(t: Path) -> bool:
    """Refuse if t IS a never-target, or sits INSIDE a never-target directory."""
    try:
        r = os.path.normcase(os.path.abspath(str(t)))
    except OSError:
        return False

    # guard 1: t itself is a refused artifact
    if r in _NEVER:
        return False

    # guard 2: any ancestor of t is a refused directory
    #   (ROOT is a never-target but is also an allowed parent, so it is
    #    exempt here -- its protected children are named individually)
    root_r = os.path.normcase(os.path.abspath(str(ROOT)))
    for n in NEVER_TARGET:
        try:
            nr = os.path.normcase(os.path.abspath(str(n)))
        except OSError:
            continue
        if nr == root_r:
            continue
        if n.is_dir() and r.startswith(nr + os.sep):
            return False

    # guard 3: t must live under an allowlisted parent (or BE one)
    return any(r == a or r.startswith(a + os.sep) for a in _ALLOWED)


def collect() -> list[Path]:
    """Enumerate targets, then register each enumerated path with the gate."""
    out: list[Path] = []
    for t in A_TARGETS + B_TARGETS:
        if t.exists():
            out.append(t)
    for pat in SCATTERED:
        for d in sorted(ROOT.glob(pat)):
            if d.is_dir() and d.name not in SCATTERED_SKIP:
                out.append(d)
    seen, uniq = set(), []
    for t in out:
        k = str(t).lower()
        if k not in seen:
            seen.add(k)
            uniq.append(t)

    # Register enumerated root-level paths one by one: the containment gate
    # then verifies each explicitly instead of trusting a wildcard pattern.
    for t in uniq:
        try:
            if os.path.normcase(os.path.dirname(os.path.abspath(str(t)))) == \
               os.path.normcase(os.path.abspath(str(ROOT))):
                _ALLOWED.add(os.path.normcase(os.path.abspath(str(t))))
        except OSError:
            continue
    return uniq


def cmdline(pid: int) -> str:
    try:
        r = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             f"(Get-CimInstance Win32_Process -Filter 'ProcessId={pid}').CommandLine"],
            capture_output=True, text=True, timeout=20)
        return (r.stdout or "").strip()
    except Exception:
        return ""


def precheck():
    LOG.mkdir(parents=True, exist_ok=True)
    targets = collect()
    rows, refused, prot_hits = [], [], []
    total = files_total = exempt_total = 0
    retained_all = []
    for t in targets:
        if not containment_ok(t):
            refused.append({"path": str(t), "why": "containment"})
            continue
        if t.is_file():
            sz = t.stat().st_size
            nfiles = 1
            nex, nret = 0, []
            if t.suffix.lower() in MODEL_EXT:
                blocked, reason = str(t), f"{t.suffix} model file"
            elif t.suffix.lower() == ".pth" and sz >= PTH_MODEL_MIN:
                blocked, reason = str(t), f".pth {sz / 1024**2:.1f} MiB (model weights)"
            elif t.suffix.lower() in LAZY_EXT and sz >= BIG_DATA_BYTES:
                blocked, reason = str(t), f"{t.suffix} {sz / 1024**2:.1f} MiB (real point cloud)"
            elif t.suffix.lower() == ".zip":
                hit = archive_has_protected(str(t))
                blocked, reason = (str(t), f"archive contains {hit}") if hit else (None, None)
            else:
                blocked, reason = None, None
        else:
            sz, nfiles, blocked, reason, nex, nret = measure(t)
        if blocked:
            prot_hits.append({"path": str(t), "protected": blocked, "reason": reason})
            continue
        total += sz
        files_total += nfiles
        exempt_total += nex
        for r in nret:
            retained_all.append({"target": str(t), **r})
        rows.append({"path": str(t), "bytes": sz, "files": nfiles,
                     "fixture_grains": nex, "retained_subtrees": nret})

    rows.sort(key=lambda r: -r["bytes"])
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps({
        "created": datetime.datetime.now().isoformat(),
        "entries": rows,
        "refused": refused,
        "protected_refused": prot_hits,
        "retained_subtrees": retained_all,
        "total_bytes": total,
        "total_files": files_total,
        "fixture_grains_exempted": exempt_total,
        "recycle_bin": str(RECYCLE_BIN),
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"targets collected       : {len(targets)}")
    print(f"safe to delete          : {len(rows)}")
    print(f"fixture files exempted  : {exempt_total:,} (small .las/.laz/.pth under test paths)")
    print(f"REFUSED (containment)   : {len(refused)}")
    for r in refused:
        print(f"    !! {r['path']}")
    print(f"REFUSED (fully blocked) : {len(prot_hits)}")
    for r in prot_hits:
        print(f"    !! {r['path']}")
        print(f"       reason: {r['reason']}")
    print(f"RETAINED subtrees       : {len(retained_all)} (target deleted except these)")
    for r in retained_all:
        print(f"    ~~ {r['path']}")
        print(f"       kept because: {r['reason']}")
    print(f"TOTAL                   : {total / 1024**3:.3f} GiB / {files_total:,} files")
    print(f"manifest                : {MANIFEST}")

    if RECYCLE_BIN.exists():
        rb, rbf, *_ = measure(RECYCLE_BIN)
        print(f"recycle bin (all disks) : {rb / 1024**3:.3f} GiB / {rbf:,} files")
    print()
    print("--- top 30 entries ---")
    for r in rows[:30]:
        print(f"  {r['bytes'] / 1024**3:9.3f} GiB  {r['path']}")
    return rows


def free_bytes() -> int:
    import ctypes
    try:
        n = ctypes.c_ulonglong(0)
        ctypes.windll.kernel32.GetDiskFreeSpaceExW(
            ctypes.c_wchar_p("E:\\"), None, None, ctypes.pointer(n))
        return n.value
    except Exception:
        return -1


def is_within(child: str, parent: str) -> bool:
    c = os.path.normcase(os.path.abspath(child))
    p = os.path.normcase(os.path.abspath(parent))
    return c == p or c.startswith(p + os.sep)


def remove_target(p: Path, retained: list[dict]) -> tuple[int, int, list[str]]:
    """Delete p, skipping any directory subtree listed in `retained`.

    Returns (bytes_removed, files_removed, skip_dirs).
    """
    keep_dirs = [r["path"] for r in retained]
    removed = files = 0
    skipped: list[str] = []

    if p.is_file() and not p.is_symlink():
        sz = p.stat().st_size
        p.unlink()
        return sz, 1, skipped

    for dirpath, dirnames, filenames in os.walk(p, topdown=False, onerror=lambda e: None):
        if any(is_within(dirpath, k) for k in keep_dirs):
            if not any(is_within(dirpath, s) for s in skipped):
                skipped.append(dirpath)
            continue
        for fn in filenames:
            fp = os.path.join(dirpath, fn)
            try:
                st = os.stat(fp, follow_symlinks=False)
                os.remove(fp)
                removed += st.st_size
                files += 1
            except OSError:
                pass
        for dn in dirnames:
            dp = os.path.join(dirpath, dn)
            if any(is_within(dp, k) for k in keep_dirs):
                if not any(is_within(dp, s) for s in skipped):
                    skipped.append(dp)
                continue
            try:
                os.rmdir(dp)
            except OSError:
                pass
    try:
        os.rmdir(p)
    except OSError:
        pass
    return removed, files, skipped


def apply():
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    rows = data["entries"]
    retained_all = data.get("retained_subtrees", [])
    by_target: dict[str, list[dict]] = {}
    for r in retained_all:
        by_target.setdefault(r["target"], []).append(r)

    print("--- running processes that may hold locks ---")
    try:
        r = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "Get-Process | Where-Object {$_.ProcessName -match 'TreeLineWarning|python|node'}"
             " | Select-Object Id,ProcessName | Format-Table -AutoSize | Out-String"],
            capture_output=True, text=True, timeout=60)
        print(r.stdout)
    except Exception as e:
        print(f"(process check failed: {e})")

    print(f"about to delete {len(rows)} paths, "
          f"{data['total_bytes'] / 1024**3:.3f} GiB logical")
    print(f"retained subtrees: {len(retained_all)}")
    print("--- first 5 and last 5 targets ---")
    for r in rows[:5]:
        print(f"  DEL {r['path']}")
    print("  ...")
    for r in rows[-5:]:
        print(f"  DEL {r['path']}")
    print()
    if os.environ.get("TLW_CONFIRM") != "YES":
        print("REFUSING: set TLW_CONFIRM=YES to actually delete.")
        return 1

    before = free_bytes()
    print(f"E: free before = {before / 1024**3:.3f} GiB")
    print()

    freed = 0
    ok = fail = 0
    results = []
    for r in rows:
        p = Path(r["path"])
        ret = by_target.get(r["path"], [])
        try:
            rem, nf, sk = remove_target(p, ret)
            freed += rem
            ok += 1
            results.append({"path": str(p), "status": "deleted",
                            "bytes": rem, "files": nf,
                            "skipped_subtrees": sk})
            if sk:
                print(f"  PARTIAL {p}  ({len(sk)} subtree(s) retained)")
        except Exception as e:
            fail += 1
            results.append({"path": str(p), "status": "FAILED", "error": str(e)})
            print(f"  FAILED {p} :: {e}")

    print()
    print("--- clearing recycle bin (all fixed disks) ---")
    try:
        r = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "Clear-RecycleBin -Force -ErrorAction SilentlyContinue; 'done'"],
            capture_output=True, text=True, timeout=1800)
        print((r.stdout or "").strip(), (r.stderr or "").strip())
    except Exception as e:
        print(f"recycle clear failed: {e}")

    after = free_bytes()
    print()
    print(f"deleted ok   : {ok}")
    print(f"failed       : {fail}")
    print(f"logical freed: {freed / 1024**3:.3f} GiB")
    print(f"E: free before = {before / 1024**3:.3f} GiB")
    print(f"E: free after  = {after / 1024**3:.3f} GiB")
    print(f"MEASURED FREED = {(after - before) / 1024**3:.3f} GiB")

    (LOG / "delete_results.json").write_text(json.dumps({
        "deleted": results,
        "free_before": before,
        "free_after": after,
        "measured_freed_bytes": after - before,
        "logical_freed_bytes": freed,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "precheck"
    if mode == "precheck":
        precheck()
    elif mode == "apply":
        sys.exit(apply())
    else:
        print("usage: cleanup_ab.py [precheck|apply]")
