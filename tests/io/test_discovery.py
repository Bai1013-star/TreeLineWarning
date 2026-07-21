from __future__ import annotations

from pathlib import Path

from algorithms.io import discover_las_sources


def test_discover_las_sources_recurses_and_ignores_non_point_cloud_files(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    nested = raw / "scene_b"
    nested.mkdir(parents=True)
    (raw / "scene_a").mkdir()
    (raw / "scene_a" / "scan_02.LAZ").write_bytes(b"not opened")
    (nested / "scan_01.las").write_bytes(b"not opened")
    (nested / "notes.txt").write_text("ignore", encoding="utf-8")

    sources = discover_las_sources(raw)

    assert [source.path.name for source in sources] == ["scan_02.LAZ", "scan_01.las"]
    assert all(source.role == "raw" for source in sources)
    assert all(source.readonly_required is True for source in sources)
    assert [source.size_bytes for source in sources] == [10, 10]
    assert len({source.source_id for source in sources}) == 2


def test_discover_las_sources_order_is_stable(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    (raw / "z").mkdir(parents=True)
    (raw / "a").mkdir()
    (raw / "z" / "later.las").write_bytes(b"las")
    (raw / "a" / "first.las").write_bytes(b"las")

    first = discover_las_sources(raw)
    second = discover_las_sources(raw)

    assert [source.path for source in first] == [source.path for source in second]
    assert [source.path.relative_to(raw).as_posix() for source in first] == [
        "a/first.las",
        "z/later.las",
    ]
