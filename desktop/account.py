"""Local desktop-account storage for the offline application.

The desktop account is deliberately independent from the project directory so
opening another project cannot reset the password.  Only a salted PBKDF2 hash
is persisted; the default credential exists virtually until the first change.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import base64
import re
from pathlib import Path
from typing import Any


DEFAULT_USERNAME = "000000"
DEFAULT_PASSWORD = "000000"
PBKDF2_ITERATIONS = 310_000
ACCOUNT_SCHEMA = "treeline-desktop-account-v2"
_PHONE_RE = re.compile(r"^\+?\d{6,20}$")
_AVATAR_RE = re.compile(r"^data:image/(?:png|jpeg|jpg|webp);base64,([A-Za-z0-9+/=]+)$")


class DesktopAccount:
    def __init__(self, repository_root: Path) -> None:
        repository = Path(repository_root).resolve()
        self.config_dir = (
            repository.parent / "用户配置"
            if repository.name == "程序文件"
            else repository / "用户配置"
        )
        self.path = self.config_dir / "账户.json"
        self._authenticated = False
        self._session_username: str | None = None

    @staticmethod
    def _hash(password: str, salt: bytes, iterations: int) -> str:
        value = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), salt, int(iterations)
        )
        return value.hex()

    def _read(self) -> dict[str, Any]:
        if not self.path.is_file():
            return {
                "schema": "treeline-desktop-account-v1",
                "username": DEFAULT_USERNAME,
                "must_change_password": True,
                "virtual_default": True,
                "profile": None,
            }
        try:
            value = json.loads(self.path.read_text("utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("ACCOUNT_CONFIG_UNREADABLE") from exc
        required = {"username", "salt", "password_hash", "iterations"}
        if not isinstance(value, dict) or not required.issubset(value):
            raise ValueError("ACCOUNT_CONFIG_INVALID")
        return value

    @staticmethod
    def _normal_phone(value: str) -> str:
        phone = re.sub(r"[\s-]", "", str(value).strip())
        if not _PHONE_RE.fullmatch(phone):
            raise ValueError("手机号应为 6–20 位数字，可使用国际区号 +")
        return phone

    @staticmethod
    def _avatar(value: str) -> tuple[str, str]:
        avatar = str(value or "").strip()
        match = _AVATAR_RE.fullmatch(avatar)
        if match is None:
            raise ValueError("请选择 PNG、JPEG 或 WebP 头像")
        try:
            raw = base64.b64decode(match.group(1), validate=True)
        except (ValueError, base64.binascii.Error) as exc:
            raise ValueError("头像文件无效") from exc
        if not raw or len(raw) > 2 * 1024 * 1024:
            raise ValueError("头像大小须在 1 字节至 2 MB 之间")
        return avatar, hashlib.sha256(raw).hexdigest()

    @classmethod
    def _profile(cls, real_name: str, phone: str, avatar_data_url: str) -> dict[str, str]:
        name = str(real_name).strip()
        if not 1 <= len(name) <= 64:
            raise ValueError("姓名长度应为 1–64 个字符")
        normalized_phone = cls._normal_phone(phone)
        avatar, digest = cls._avatar(avatar_data_url)
        return {
            "real_name": name,
            "phone": normalized_phone,
            "avatar_data_url": avatar,
            "avatar_sha256": digest,
        }

    def _write(self, payload: dict[str, Any]) -> None:
        self.config_dir.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), "utf-8")
        os.replace(temporary, self.path)

    def status(self) -> dict[str, Any]:
        data = self._read()
        profile = data.get("profile") if isinstance(data.get("profile"), dict) else None
        return {
            "ok": True,
            "authenticated": self._authenticated,
            "username": self._session_username if self._authenticated else None,
            "must_change_password": bool(data.get("must_change_password", False))
            if self._authenticated
            else False,
            "first_login": bool(data.get("virtual_default", False)),
            "profile_setup_required": self._authenticated and profile is None,
            "real_name": profile.get("real_name") if self._authenticated and profile else None,
            "phone_masked": self._masked_phone(profile.get("phone")) if self._authenticated and profile else None,
            "avatar_data_url": profile.get("avatar_data_url") if self._authenticated and profile else None,
        }

    @staticmethod
    def _masked_phone(value: Any) -> str | None:
        phone = str(value or "")
        if not phone:
            return None
        if len(phone) <= 7:
            return phone[:2] + "***" + phone[-2:]
        return phone[:3] + "****" + phone[-4:]

    def login(self, username: str, password: str) -> dict[str, Any]:
        data = self._read()
        supplied_user = str(username).strip()
        supplied_password = str(password)
        if data.get("virtual_default"):
            valid = hmac.compare_digest(supplied_user, DEFAULT_USERNAME) and hmac.compare_digest(
                supplied_password, DEFAULT_PASSWORD
            )
        else:
            try:
                salt = bytes.fromhex(str(data["salt"]))
                expected = str(data["password_hash"])
                actual = self._hash(supplied_password, salt, int(data["iterations"]))
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("ACCOUNT_CONFIG_INVALID") from exc
            valid = hmac.compare_digest(
                supplied_user.encode("utf-8"), str(data["username"]).encode("utf-8")
            ) and hmac.compare_digest(
                actual, expected
            )
        if not valid:
            self._authenticated = False
            self._session_username = None
            return {"ok": False, "error": "用户名或密码错误"}
        self._authenticated = True
        self._session_username = str(data["username"])
        return self.status()

    def change_password(
        self,
        current_password: str,
        new_password: str,
        confirm_password: str,
        new_username: str | None = None,
        real_name: str | None = None,
        phone: str | None = None,
        avatar_data_url: str | None = None,
    ) -> dict[str, Any]:
        if not self._authenticated or self._session_username is None:
            return {"ok": False, "error": "请先登录"}
        current = self.login(self._session_username, str(current_password))
        if not current.get("ok"):
            return {"ok": False, "error": "当前密码不正确"}
        username = str(new_username or self._session_username).strip()
        password = str(new_password)
        if not username or len(username) > 64:
            return {"ok": False, "error": "用户名长度应为 1–64 个字符"}
        if password != str(confirm_password):
            return {"ok": False, "error": "两次输入的新密码不一致"}
        if len(password) < 6 or len(password) > 128:
            return {"ok": False, "error": "新密码长度应为 6–128 个字符"}
        if password == DEFAULT_PASSWORD:
            return {"ok": False, "error": "新密码不能继续使用默认密码 000000"}
        previous = self._read()
        stored_profile = previous.get("profile") if isinstance(previous.get("profile"), dict) else None
        if stored_profile is None:
            try:
                stored_profile = self._profile(real_name or "", phone or "", avatar_data_url or "")
            except ValueError as exc:
                return {"ok": False, "error": str(exc)}
        salt = secrets.token_bytes(16)
        payload = {
            "schema": ACCOUNT_SCHEMA,
            "username": username,
            "salt": salt.hex(),
            "password_hash": self._hash(password, salt, PBKDF2_ITERATIONS),
            "iterations": PBKDF2_ITERATIONS,
            "must_change_password": False,
            "profile": stored_profile,
        }
        self._write(payload)
        self._authenticated = True
        self._session_username = username
        return self.status()

    def setup_profile(self, real_name: str, phone: str, avatar_data_url: str) -> dict[str, Any]:
        if not self._authenticated or self._session_username is None:
            return {"ok": False, "error": "请先登录"}
        data = self._read()
        if data.get("virtual_default"):
            return {"ok": False, "error": "首次登录请同时设置新密码和身份资料"}
        try:
            data["profile"] = self._profile(real_name, phone, avatar_data_url)
        except ValueError as exc:
            return {"ok": False, "error": str(exc)}
        data["schema"] = ACCOUNT_SCHEMA
        self._write(data)
        return self.status()

    def reset_password(
        self,
        username: str,
        real_name: str,
        phone: str,
        avatar_data_url: str,
        new_password: str,
        confirm_password: str,
    ) -> dict[str, Any]:
        data = self._read()
        profile = data.get("profile") if isinstance(data.get("profile"), dict) else None
        if data.get("virtual_default") or profile is None:
            return {"ok": False, "error": "尚未建立找回资料，请先使用默认账户完成首次设置"}
        try:
            candidate = self._profile(real_name, phone, avatar_data_url)
        except ValueError as exc:
            return {"ok": False, "error": str(exc)}
        identity_ok = all(
            hmac.compare_digest(str(left).encode("utf-8"), str(right).encode("utf-8"))
            for left, right in (
                (str(username).strip(), data.get("username")),
                (candidate["real_name"], profile.get("real_name")),
                (candidate["phone"], profile.get("phone")),
                (candidate["avatar_sha256"], profile.get("avatar_sha256")),
            )
        )
        if not identity_ok:
            return {"ok": False, "error": "用户名、姓名、手机号或头像与首次登记资料不一致"}
        password = str(new_password)
        if password != str(confirm_password):
            return {"ok": False, "error": "两次输入的新密码不一致"}
        if len(password) < 6 or len(password) > 128:
            return {"ok": False, "error": "新密码长度应为 6–128 个字符"}
        if password == DEFAULT_PASSWORD:
            return {"ok": False, "error": "新密码不能使用默认密码 000000"}
        salt = secrets.token_bytes(16)
        data.update(
            schema=ACCOUNT_SCHEMA,
            salt=salt.hex(),
            password_hash=self._hash(password, salt, PBKDF2_ITERATIONS),
            iterations=PBKDF2_ITERATIONS,
            must_change_password=False,
        )
        data.pop("virtual_default", None)
        self._write(data)
        self._authenticated = False
        self._session_username = None
        return {"ok": True, "authenticated": False, "message": "密码已重置，请使用新密码登录"}

    def logout(self) -> dict[str, Any]:
        self._authenticated = False
        self._session_username = None
        return {"ok": True, "authenticated": False}

