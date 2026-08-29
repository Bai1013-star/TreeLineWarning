from __future__ import annotations

from pathlib import Path


def test_desktop_installs_tree_display_numbering_extension() -> None:
    root = Path(__file__).resolve().parents[2]
    package = (root / "desktop" / "__init__.py").read_text(encoding="utf-8")
    extension = (root / "desktop" / "tree_display_numbering_extensions.py").read_text(
        encoding="utf-8"
    )

    assert "install_tree_display_numbering_extensions()" in package
    assert 'row["display_tree_id"]' in extension
    assert "${t.display_tree_id||t.persistent_tree_id}" in extension
    assert 'page.replace("<th>Tree ID</th>", "<th>树编号</th>")' in extension
    assert "persistent_tree_id" in extension


def test_tree_display_numbering_is_presentation_only() -> None:
    root = Path(__file__).resolve().parents[2]
    implementation = (root / "core" / "stage_b" / "tree_display_id.py").read_text(
        encoding="utf-8"
    )

    assert 'f"T-{span_number}-{tree_order:03d}"' in implementation
    assert "LOWER_TOWER_NUMBER" in implementation
    assert "persistent_tree_id" in implementation
    assert "replace(" not in implementation
