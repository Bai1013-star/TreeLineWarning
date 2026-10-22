"""Verify a portable release ZIP without extracting another multi-gigabyte copy."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import zipfile


def stream_digest(member, chunk_size: int = 2**20) -> str:
    digest = hashlib.sha256()
    for chunk in iter(lambda: member.read(chunk_size), b""):
        digest.update(chunk)
    return digest.hexdigest()


def verify(archive: Path, report: Path) -> dict:
    archive = archive.resolve()
    archive.relative_to(Path("D:/TreeLineWarning").resolve())
    with zipfile.ZipFile(archive) as zipped:
        files = {item.filename: item for item in zipped.infolist() if not item.is_dir()}
        roots = {PurePosixPath(name).parts[0] for name in files}
        if len(roots) != 1:
            raise AssertionError(f"ZIP must have one root directory, got {sorted(roots)}")
        root = roots.pop()
        for name in files:
            parts = PurePosixPath(name).parts
            if name.startswith("/") or ".." in parts:
                raise AssertionError(f"unsafe ZIP member: {name}")

        manifest_name = f"{root}/SHA256文件清单.json"
        if manifest_name not in files:
            raise AssertionError("manifest is missing")
        manifest = json.loads(zipped.read(manifest_name).decode("utf-8"))
        for index, (relative_name, expected) in enumerate(manifest.items(), start=1):
            member_name = f"{root}/{relative_name}"
            info = files.get(member_name)
            if info is None:
                raise AssertionError(f"manifest member missing: {relative_name}")
            if info.file_size != expected["bytes"]:
                raise AssertionError(f"size mismatch: {relative_name}")
            with zipped.open(info, "r") as stream:
                actual_hash = stream_digest(stream)
            if actual_hash != expected["sha256"]:
                raise AssertionError(f"hash mismatch: {relative_name}")
            if index % 2000 == 0:
                print(f"Verified {index}/{len(manifest)}", flush=True)

        match = re.fullmatch(r"TreeLineWarning_一体化_(\d{8})", root)
        if match is None:
            raise AssertionError(f"unexpected release root: {root}")
        version_tag = match.group(1)
        required = {
            f"{root}/TreeLineWarning_{version_tag}.exe",
            f"{root}/程序文件/desktop/app.py",
            f"{root}/程序文件/web/templates/index.html",
            f"{root}/程序文件/viewer_poc/giro3d/desktop.html",
            f"{root}/使用说明/先读我.txt",
            f"{root}/使用说明/验收范围与限制.txt",
        }
        missing = sorted(required - files.keys())
        if missing:
            raise AssertionError(f"required files missing: {missing}")

        directory_names = {item.filename for item in zipped.infolist() if item.is_dir()}
        for directory in ("项目数据", "运行日志", "导出结果"):
            if f"{root}/{directory}/" not in directory_names:
                raise AssertionError(f"empty writable directory missing: {directory}")

    archive_hash = hashlib.sha256()
    with archive.open("rb") as stream:
        for chunk in iter(lambda: stream.read(2**20), b""):
            archive_hash.update(chunk)
    result = {
        "passed": True,
        "archive": str(archive),
        "root": root,
        "verifiedFiles": len(manifest),
        "archiveBytes": archive.stat().st_size,
        "archiveSha256": archive_hash.hexdigest(),
    }
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    verify(args.archive, args.report)


if __name__ == "__main__":
    main()
