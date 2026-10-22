from pathlib import Path

import pytest

from desktop.project_management import create_empty_project_in_folder


def test_selected_empty_folder_is_project_root_without_extra_child(tmp_path: Path):
    selected = tmp_path / "用户选择的项目文件夹"
    selected.mkdir()

    descriptor = create_empty_project_in_folder(selected)

    assert Path(descriptor["root"]) == selected.resolve()
    assert descriptor["project_name"] == selected.name
    assert (selected / "desktop-project.json").is_file()
    assert (selected / "stage-a-project.json").is_file()
    assert not (selected / selected.name).exists()


def test_selected_nonempty_folder_is_rejected(tmp_path: Path):
    selected = tmp_path / "not-empty"
    selected.mkdir()
    (selected / "unrelated.txt").write_text("keep", encoding="utf-8")

    with pytest.raises(ValueError, match="不是空文件夹"):
        create_empty_project_in_folder(selected)
