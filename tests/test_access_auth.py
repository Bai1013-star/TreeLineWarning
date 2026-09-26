import base64

from api.prediction_api import create_app
from fastapi.testclient import TestClient


def _basic_auth(username: str, password: str) -> dict[str, str]:
    encoded = base64.b64encode(f"{username}:{password}".encode()).decode()
    return {"Authorization": f"Basic {encoded}"}


def test_access_password_protects_entire_application():
    app = create_app(access_password="strong-test-password")
    client = TestClient(app)

    for path in ("/", "/health", "/history", "/docs", "/openapi.json"):
        response = client.get(path)
        assert response.status_code == 401
        assert response.headers["www-authenticate"].startswith('Basic realm="TreeLineWarning"')
        assert response.headers["cache-control"] == "no-store"


def test_access_password_accepts_only_matching_credentials():
    app = create_app(access_username="operator", access_password="strong-test-password")
    client = TestClient(app)

    assert client.get(
        "/health", headers=_basic_auth("operator", "wrong-password")
    ).status_code == 401
    response = client.get(
        "/health", headers=_basic_auth("operator", "strong-test-password")
    )
    assert response.status_code == 200
    assert response.json() == {"message": "TreeLineWarning API running"}
    assert response.headers["cache-control"] == "no-store"


def test_access_password_can_be_loaded_from_environment(monkeypatch):
    monkeypatch.setenv("TREELINE_WARNING_ACCESS_USERNAME", "environment-user")
    monkeypatch.setenv("TREELINE_WARNING_ACCESS_PASSWORD", "environment-password")
    client = TestClient(create_app())

    assert client.get("/health").status_code == 401
    assert client.get(
        "/health", headers=_basic_auth("environment-user", "environment-password")
    ).status_code == 200


def test_application_remains_unprotected_when_password_is_not_configured(monkeypatch):
    monkeypatch.delenv("TREELINE_WARNING_ACCESS_PASSWORD", raising=False)
    client = TestClient(create_app())

    assert client.get("/health").status_code == 200
