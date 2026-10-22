from __future__ import annotations

import json

from desktop.account import DEFAULT_PASSWORD, DEFAULT_USERNAME, DesktopAccount


AVATAR = "data:image/png;base64,iVBORw0KGgo="


def test_first_login_forces_password_change_and_persists_hash(tmp_path):
    account = DesktopAccount(tmp_path)
    failed = account.login(DEFAULT_USERNAME, "bad")
    assert failed == {"ok": False, "error": "用户名或密码错误"}

    first = account.login(DEFAULT_USERNAME, DEFAULT_PASSWORD)
    assert first["authenticated"] is True
    assert first["must_change_password"] is True

    mismatch = account.change_password(DEFAULT_PASSWORD, "new-secret", "wrong")
    assert mismatch["ok"] is False
    changed = account.change_password(
        DEFAULT_PASSWORD, "new-secret", "new-secret", "巡检员01",
        "张三", "13800000000", AVATAR,
    )
    assert changed["authenticated"] is True
    assert changed["must_change_password"] is False
    assert changed["username"] == "巡检员01"

    stored = json.loads(account.path.read_text("utf-8"))
    assert "new-secret" not in account.path.read_text("utf-8")
    assert stored["password_hash"]
    assert stored["salt"]

    account.logout()
    assert account.login(DEFAULT_USERNAME, DEFAULT_PASSWORD)["ok"] is False
    assert account.login("巡检员01", "new-secret")["authenticated"] is True


def test_default_password_cannot_be_reused(tmp_path):
    account = DesktopAccount(tmp_path)
    assert account.login(DEFAULT_USERNAME, DEFAULT_PASSWORD)["ok"] is True
    result = account.change_password(DEFAULT_PASSWORD, DEFAULT_PASSWORD, DEFAULT_PASSWORD)
    assert result["ok"] is False
    assert "默认密码" in result["error"]


def test_forgot_password_verifies_local_profile_and_resets(tmp_path):
    account = DesktopAccount(tmp_path)
    assert account.login(DEFAULT_USERNAME, DEFAULT_PASSWORD)["ok"] is True
    assert account.change_password(
        DEFAULT_PASSWORD, "first-secret", "first-secret", "巡检员01",
        "张三", "13800000000", AVATAR,
    )["ok"] is True
    account.logout()

    denied = account.reset_password(
        "巡检员01", "张三", "13900000000", AVATAR, "second-secret", "second-secret"
    )
    assert denied["ok"] is False
    reset = account.reset_password(
        "巡检员01", "张三", "13800000000", AVATAR, "second-secret", "second-secret"
    )
    assert reset["ok"] is True
    assert account.login("巡检员01", "first-secret")["ok"] is False
    assert account.login("巡检员01", "second-secret")["ok"] is True
