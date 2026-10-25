from datetime import datetime, timezone

from types import SimpleNamespace
import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import app
import app.api.routes_sweep as routes_sweep
from tests.codegen_fakes import CodegenFakeMethods


class FakeDatabase(CodegenFakeMethods):
    def __init__(self):
        self._sweeps = {}
        self._sweep_configs = {}
        self._next_id = 1

    async def create_sweep(self, ingestion_budget, eval_budget, codegen_budget=None,
                           bootstrap_options=None):
        sweep_id = self._next_id
        self._next_id += 1
        self._sweeps[sweep_id] = {
            "id": sweep_id, "status": "bootstrapping", "ingestion_budget": ingestion_budget,
            "eval_budget": eval_budget, "best_config_id": None, "best_score": None,
            "summary_markdown": None, "error_message": None, "corpus_snapshot": ["doc1.pdf"],
            "codegen_budget": codegen_budget, "bootstrap_options": bootstrap_options,
            "bootstrap_complete": True,
            "created_at": datetime.now(timezone.utc), "completed_at": None,
        }
        return sweep_id

    async def get_sweep(self, sweep_id):
        return self._sweeps.get(sweep_id)

    async def update_sweep_status(self, sweep_id, status, error_message=None):
        self._sweeps[sweep_id]["status"] = status
        self._sweeps[sweep_id]["error_message"] = error_message

    async def list_sweep_configs(self, sweep_id):
        return self._sweep_configs.get(sweep_id, [])

    async def sum_ingestion_spend(self, sweep_id):
        return {"cost_usd": 0.0, "minutes": 0.0, "iterations": 0}

    async def sum_eval_spend(self, sweep_id):
        return {"cost_usd": 0.0, "minutes": 0.0, "iterations": 0}


async def _noop(*args, **kwargs):
    return None


def _patch_orchestrator(monkeypatch):
    monkeypatch.setattr(routes_sweep, "run_bootstrap", _noop)
    monkeypatch.setattr(routes_sweep, "run_sweep_loop", _noop)
    monkeypatch.setattr(routes_sweep, "resume_sweep", _noop)
    monkeypatch.setattr(routes_sweep, "apply_user_suggestion", _noop)
    monkeypatch.setattr(routes_sweep, "finalize_sweep_run", _noop)
    monkeypatch.setattr(routes_sweep, "list_pdf_filenames", lambda corpus_dir: ["doc1.pdf"])


def _client(monkeypatch, db=None):
    _patch_orchestrator(monkeypatch)
    app.state.db = db or FakeDatabase()
    app.state.settings = Settings()
    app.state.ner = SimpleNamespace(labels=[])
    app.state.chat_provider = object()
    app.state.judge = object()
    app.state.sweep_in_progress = False
    app.state.sweep_provider = object()
    return TestClient(app)


VALID_START_BODY = {
    "ingestion_budget": {"cost_usd": 10, "minutes": 60, "iterations": 10},
    "eval_budget": {"cost_usd": 10, "minutes": 30, "iterations": 50},
    "generate_golden_qa": False,
}


def test_start_sweep_rejects_zero_budget_axis(monkeypatch):
    client = _client(monkeypatch)
    body = {**VALID_START_BODY, "ingestion_budget": {**VALID_START_BODY["ingestion_budget"], "cost_usd": 0}}

    response = client.post("/api/sweep/start", json=body)

    assert response.status_code == 400


def test_start_sweep_succeeds_with_bootstrapping_status(monkeypatch):
    client = _client(monkeypatch)

    response = client.post("/api/sweep/start", json=VALID_START_BODY)

    assert response.status_code == 200
    assert response.json()["status"] == "bootstrapping"


def test_approve_sweep_rejects_wrong_status(monkeypatch):
    db = FakeDatabase()
    client = _client(monkeypatch, db)
    sweep_id = client.post("/api/sweep/start", json=VALID_START_BODY).json()["sweep_id"]
    # status is still "bootstrapping", not "awaiting_approval"

    response = client.post(f"/api/sweep/{sweep_id}/approve")

    assert response.status_code == 409


def test_full_happy_path_status_progression(monkeypatch):
    db = FakeDatabase()
    client = _client(monkeypatch, db)

    start_resp = client.post("/api/sweep/start", json=VALID_START_BODY)
    sweep_id = start_resp.json()["sweep_id"]
    assert client.get(f"/api/sweep/{sweep_id}").json()["status"] == "bootstrapping"

    await_approval = db._sweeps[sweep_id]
    await_approval["status"] = "awaiting_approval"
    approve_resp = client.post(f"/api/sweep/{sweep_id}/approve")
    assert approve_resp.status_code == 200
    assert client.get(f"/api/sweep/{sweep_id}").json()["status"] == "running"

    db._sweeps[sweep_id]["status"] = "paused"
    resume_resp = client.post(f"/api/sweep/{sweep_id}/resume")
    assert resume_resp.status_code == 200

    db._sweeps[sweep_id]["status"] = "awaiting_suggestion"
    suggest_resp = client.post(f"/api/sweep/{sweep_id}/suggest", json={"accept": True})
    assert suggest_resp.status_code == 200


def test_resume_sets_status_synchronously_before_background_task_runs(monkeypatch):
    # apply_user_suggestion/run_sweep_loop/resume_sweep/finalize_sweep_run
    # are all faked as no-ops by _patch_orchestrator, so if status ever
    # becomes something other than "paused", only the route itself (not
    # the background task) could have set it -- proving it happens
    # synchronously, before the HTTP response is returned, not racing a
    # poll that might land before the background task has run.
    db = FakeDatabase()
    client = _client(monkeypatch, db)
    sweep_id = client.post("/api/sweep/start", json=VALID_START_BODY).json()["sweep_id"]
    db._sweeps[sweep_id]["status"] = "paused"

    response = client.post(f"/api/sweep/{sweep_id}/resume")

    assert response.status_code == 200
    assert db._sweeps[sweep_id]["status"] == "running"


def test_suggest_continue_sets_status_synchronously_before_background_task_runs(monkeypatch):
    db = FakeDatabase()
    client = _client(monkeypatch, db)
    sweep_id = client.post("/api/sweep/start", json=VALID_START_BODY).json()["sweep_id"]
    db._sweeps[sweep_id]["status"] = "awaiting_suggestion"

    response = client.post(f"/api/sweep/{sweep_id}/suggest", json={"hint": "try smaller chunks"})

    assert response.status_code == 200
    assert db._sweeps[sweep_id]["status"] == "running"


def test_suggest_accept_sets_status_synchronously_before_background_task_runs(monkeypatch):
    db = FakeDatabase()
    client = _client(monkeypatch, db)
    sweep_id = client.post("/api/sweep/start", json=VALID_START_BODY).json()["sweep_id"]
    db._sweeps[sweep_id]["status"] = "awaiting_suggestion"

    response = client.post(f"/api/sweep/{sweep_id}/suggest", json={"accept": True})

    assert response.status_code == 200
    # Must not still read "awaiting_suggestion" -- that's exactly what
    # bounces the wizard back to the suggestion page (review finding I-5).
    assert db._sweeps[sweep_id]["status"] != "awaiting_suggestion"


def test_resume_rejects_changed_corpus(monkeypatch):
    db = FakeDatabase()
    client = _client(monkeypatch, db)
    sweep_id = client.post("/api/sweep/start", json=VALID_START_BODY).json()["sweep_id"]
    db._sweeps[sweep_id]["status"] = "paused"
    db._sweeps[sweep_id]["corpus_snapshot"] = ["doc1.pdf", "doc2.pdf"]  # differs from fake listing

    response = client.post(f"/api/sweep/{sweep_id}/resume")

    assert response.status_code == 409


def test_cancel_sets_status(monkeypatch):
    db = FakeDatabase()
    client = _client(monkeypatch, db)
    sweep_id = client.post("/api/sweep/start", json=VALID_START_BODY).json()["sweep_id"]

    response = client.post(f"/api/sweep/{sweep_id}/cancel")

    assert response.status_code == 200
    assert response.json()["status"] == "cancelled"


def test_get_unknown_sweep_404(monkeypatch):
    client = _client(monkeypatch)

    response = client.get("/api/sweep/999")

    assert response.status_code == 404


@pytest.mark.asyncio
async def test_run_sweep_task_marks_sweep_failed_on_unexpected_exception():
    db = FakeDatabase()
    sweep_id = await db.create_sweep({}, {})

    async def boom():
        raise RuntimeError("something broke")

    await routes_sweep._run_sweep_task(db, sweep_id, boom())

    sweep = await db.get_sweep(sweep_id)
    assert sweep["status"] == "failed"
    assert "something broke" in sweep["error_message"]


@pytest.mark.asyncio
async def test_run_sweep_task_leaves_status_alone_on_success():
    db = FakeDatabase()
    sweep_id = await db.create_sweep({}, {})
    await db.update_sweep_status(sweep_id, "running")

    async def fine():
        return None

    await routes_sweep._run_sweep_task(db, sweep_id, fine())

    sweep = await db.get_sweep(sweep_id)
    assert sweep["status"] == "running"


@pytest.mark.parametrize(
    "status, resumable",
    [("paused", True), ("failed", True), ("running", False), ("cancelled", False)],
)
def test_latest_sweep_resumable_when_paused_or_failed(monkeypatch, status, resumable):
    class LatestDb(FakeDatabase):
        async def get_latest_sweep_summary(self):
            return {"id": 7, "status": status, "bootstrap_complete": False, "config_count": 0}

    client = _client(monkeypatch, LatestDb())

    response = client.get("/api/sweep/latest")

    assert response.json() == {"sweep_id": 7, "status": status, "resumable": resumable}


def test_resume_failed_bootstrap_reruns_bootstrap_with_stored_options(monkeypatch):
    calls = []

    async def recording_bootstrap(db, provider, ner, settings, sweep_id, generate_golden_qa_flag,
                                  codegen_tasks=None, codegen_budget=None, codegen_provider=None):
        calls.append((sweep_id, generate_golden_qa_flag, codegen_tasks))

    db = FakeDatabase()
    client = _client(monkeypatch, db)
    monkeypatch.setattr(routes_sweep, "run_bootstrap", recording_bootstrap)
    monkeypatch.setattr(routes_sweep, "get_codegen_provider", lambda settings: object())
    sweep_id = _insert_bootstrapping_sweep(db, bootstrap_options={"generate_golden_qa": True,
                                                                  "codegen_tasks": ["ocr"]})
    db._sweeps[sweep_id].update(status="failed", bootstrap_complete=False, error_message="boom")

    response = client.post(f"/api/sweep/{sweep_id}/resume")

    assert response.status_code == 200
    assert response.json()["status"] == "bootstrapping"
    assert db._sweeps[sweep_id]["status"] == "bootstrapping"
    assert calls == [(sweep_id, True, ["ocr"])]


def _insert_bootstrapping_sweep(db, bootstrap_options):
    sweep_id = db._next_id
    db._next_id += 1
    db._sweeps[sweep_id] = {
        "id": sweep_id, "status": "bootstrapping", "ingestion_budget": {}, "eval_budget": {},
        "best_config_id": None, "best_score": None, "summary_markdown": None,
        "error_message": None, "corpus_snapshot": None, "codegen_budget": None,
        "bootstrap_options": bootstrap_options, "bootstrap_complete": False,
        "created_at": datetime.now(timezone.utc), "completed_at": None,
    }
    return sweep_id


def test_sweep_judge_is_built_from_app_settings_once(monkeypatch):
    sentinel = object()
    calls = []

    def fake_create_judge(settings, router):
        calls.append((settings, router))
        return sentinel

    monkeypatch.setattr(routes_sweep, "create_judge", fake_create_judge)
    state = SimpleNamespace(judge=None, settings="settings", router="router")
    request = SimpleNamespace(app=SimpleNamespace(state=state))

    assert routes_sweep._get_judge(request) is sentinel
    assert routes_sweep._get_judge(request) is sentinel
    assert calls == [("settings", "router")]


def test_latest_sweep_returns_null_when_no_sweeps(monkeypatch):
    class EmptyDb(FakeDatabase):
        async def get_latest_sweep_summary(self):
            return None

    client = _client(monkeypatch, EmptyDb())

    assert client.get("/api/sweep/latest").json() is None
