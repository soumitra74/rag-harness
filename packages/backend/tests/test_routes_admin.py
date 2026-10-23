from fastapi.testclient import TestClient

from app.config import Settings
from app.main import app
import app.api.routes_admin as routes_admin
from app.services.reset_db import BackupFailedError, ResetBlockedError


class FakeDatabase:
    def __init__(self):
        self.backup_calls = []
        self.execute_calls = []

    async def list_chunk_tables(self):
        return []


async def fake_preview_reset(db, categories):
    return {"graph_entities": 1}


async def fake_backup_before_reset(categories, backup_dir, run_subprocess=None,
                                    database_url=None, chunk_tables=None):
    return "/backups/reset-fake.sql"


async def fake_execute_reset(db, categories):
    db.execute_calls.append(categories)


def _client(monkeypatch, db=None):
    monkeypatch.setattr(routes_admin, "preview_reset", fake_preview_reset)
    monkeypatch.setattr(routes_admin, "backup_before_reset", fake_backup_before_reset)
    monkeypatch.setattr(routes_admin, "execute_reset", fake_execute_reset)
    app.state.db = db or FakeDatabase()
    app.state.settings = Settings()
    return TestClient(app)


def test_preview_reset_route_returns_counts(monkeypatch):
    client = _client(monkeypatch)

    response = client.post("/api/admin/reset/preview", json={"categories": ["graph_data"]})

    assert response.status_code == 200
    assert response.json() == {"graph_entities": 1}


def test_execute_reset_route_rejects_wrong_confirm_phrase(monkeypatch):
    db = FakeDatabase()
    client = _client(monkeypatch, db)

    response = client.post(
        "/api/admin/reset/execute",
        json={"categories": ["graph_data"], "confirm_phrase": "not it"},
    )

    assert response.status_code == 400
    assert db.execute_calls == []


def test_execute_reset_route_succeeds_with_correct_phrase(monkeypatch):
    db = FakeDatabase()
    client = _client(monkeypatch, db)

    response = client.post(
        "/api/admin/reset/execute",
        json={"categories": ["graph_data"], "confirm_phrase": "RESET"},
    )

    assert response.status_code == 200
    assert db.execute_calls == [{"graph_data"}]


def test_execute_reset_route_returns_409_when_backup_fails(monkeypatch):
    async def failing_backup(categories, backup_dir, run_subprocess=None,
                              database_url=None, chunk_tables=None):
        raise BackupFailedError("pg_dump exited with code 1; reset aborted")

    db = FakeDatabase()
    monkeypatch.setattr(routes_admin, "preview_reset", fake_preview_reset)
    monkeypatch.setattr(routes_admin, "backup_before_reset", failing_backup)
    monkeypatch.setattr(routes_admin, "execute_reset", fake_execute_reset)
    app.state.db = db
    app.state.settings = Settings()
    client = TestClient(app)

    response = client.post(
        "/api/admin/reset/execute",
        json={"categories": ["graph_data"], "confirm_phrase": "RESET"},
    )

    assert response.status_code == 409
    assert db.execute_calls == []


def test_execute_reset_route_returns_409_when_reset_is_blocked(monkeypatch):
    async def blocked_execute(db, categories):
        raise ResetBlockedError("rca_sessions references eval_cycles")

    db = FakeDatabase()
    monkeypatch.setattr(routes_admin, "preview_reset", fake_preview_reset)
    monkeypatch.setattr(routes_admin, "backup_before_reset", fake_backup_before_reset)
    monkeypatch.setattr(routes_admin, "execute_reset", blocked_execute)
    app.state.db = db
    app.state.settings = Settings()
    client = TestClient(app)

    response = client.post(
        "/api/admin/reset/execute",
        json={"categories": ["eval_history"], "confirm_phrase": "RESET"},
    )

    assert response.status_code == 409
