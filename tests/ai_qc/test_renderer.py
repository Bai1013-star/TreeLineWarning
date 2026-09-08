from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from core.ai_qc.renderer import RENDER_REVISION, VIEW_NAMES, render_tree_views


def _fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    z = np.linspace(0.0, 8.0, 240)
    trunk = np.column_stack((np.sin(z) * 0.08, np.cos(z) * 0.08, z))
    theta = np.linspace(0.0, np.pi * 12.0, 480)
    crown = np.column_stack(
        (
            np.cos(theta) * (1.2 + 0.2 * np.sin(theta * 0.5)),
            np.sin(theta) * (1.0 + 0.1 * np.cos(theta)),
            6.5 + np.sin(theta * 0.7),
        )
    )
    tree = np.vstack((trunk, crown)).astype(np.float64)
    context = tree + np.asarray((4.0, 1.0, 0.0))
    return (
        tree,
        np.arange(len(tree), dtype=np.int64),
        np.vstack((tree, context)),
        np.concatenate(
            (np.full(len(tree), 7, dtype=np.int32), np.full(len(context), 8, dtype=np.int32))
        ),
    )


def _render(output: Path, revision: str):
    tree, original_index, scene, owners = _fixture()
    return render_tree_views(
        tree_id="T-1-007",
        tree_xyz=tree,
        tree_original_index=original_index,
        scene_xyz=scene,
        scene_tree_id=owners,
        selected_tree_id=7,
        output_dir=output,
        source_editor_revision=revision,
    )


def test_six_views_are_byte_deterministic(tmp_path: Path) -> None:
    first = _render(tmp_path / "first", "rev-1")
    second = _render(tmp_path / "second", "rev-1")

    assert len(first.view_files) == len(VIEW_NAMES) == 6
    assert first.view_sha256 == second.view_sha256
    assert first.contact_sheet_sha256 == second.contact_sheet_sha256
    assert first.membership_sha256 == second.membership_sha256


def test_render_manifest_binds_editor_revision(tmp_path: Path) -> None:
    rendered = _render(tmp_path / "render", "edit-sequence-4")
    manifest = json.loads(rendered.metadata_path.read_text(encoding="utf-8"))

    assert manifest["render_revision"] == RENDER_REVISION
    assert manifest["source_editor_revision"] == "edit-sequence-4"
    assert manifest["membership_sha256"] == rendered.membership_sha256
    assert manifest["view_sha256"] == list(rendered.view_sha256)
