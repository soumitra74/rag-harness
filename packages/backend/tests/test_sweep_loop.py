import pytest

from app.config import Settings
from app.db import ExperimentConfig
from app.services.corpus_inspector import CorpusConfigCandidate
from app.services.sweep_next_config import NextConfigResult
import app.services.sweep as sweep
from tests.codegen_fakes import CodegenFakeMethods

CONFIG = ExperimentConfig(
    id=1, chunking_strategy="sliding_window", chunk_size=1000, chunk_overlap=200,
    embedding_provider="local", embedding_model="BAAI/bge-base-en-v1.5", embedding_dim=768,
)

NEW_CANDIDATE = CorpusConfigCandidate(
    chunking_strategy="sliding_window", chunk_size=1200, chunk_overlap=200,
    embedding_provider="local", embedding_model="BAAI/bge-base-en-v1.5", rationale="try bigger",
)


class FakeDatabase(CodegenFakeMethods):
    def __init__(self, pending_sweep_config):
        self._pending = [pending_sweep_config] if pending_sweep_config else []
        self._next_id = 100
        self.status_updates = []
        self.created_sweep_configs = []
        self.evaluated_combos_updates = []
        self.status_set_updates = []
        self.finalize_calls = []

    async def get_sweep(self, sweep_id):
        return {"id": sweep_id, "ingestion_budget": {"cost_usd": 10.0}, "eval_budget": {"cost_usd": 10.0}}

    async def get_experiment_config(self, config_id):
        return CONFIG

    async def get_next_pending_sweep_config(self, sweep_id):
        return self._pending[0] if self._pending else None

    async def update_sweep_config_status(self, sweep_config_id, status):
        self.status_set_updates.append((sweep_config_id, status))
        if self._pending and self._pending[0]["id"] == sweep_config_id and status == "evaluated":
            self._pending.pop(0)

    async def update_sweep_config_evaluated_combos(self, sweep_config_id, combos):
        self.evaluated_combos_updates.append((sweep_config_id, list(combos)))
        if self._pending and self._pending[0]["id"] == sweep_config_id:
            self._pending[0]["evaluated_combos"] = list(combos)

    async def update_sweep_status(self, sweep_id, status, error_message=None):
        self.status_updates.append((sweep_id, status, error_message))

    async def finalize_sweep(self, sweep_id, status, summary_markdown, best_config_id, best_score):
        self.finalize_calls.append((sweep_id, status, summary_markdown, best_config_id, best_score))

    async def create_experiment_config(self, chunking_strategy, chunk_size, chunk_overlap,
                                        embedding_provider, embedding_model, embedding_dim):
        config = ExperimentConfig(
            id=self._next_id, chunking_strategy=chunking_strategy, chunk_size=chunk_size,
            chunk_overlap=chunk_overlap, embedding_provider=embedding_provider,
            embedding_model=embedding_model, embedding_dim=embedding_dim,
        )
        self._next_id += 1
        return config

    async def create_sweep_config(self, sweep_id, config_id, rationale, status, source):
        self.created_sweep_configs.append(
            {"sweep_id": sweep_id, "config_id": config_id, "rationale": rationale,
             "status": status, "source": source}
        )
        return len(self.created_sweep_configs)

    async def list_sweep_configs(self, sweep_id):
        return []

    async def list_eval_cycles_for_sweep(self, sweep_id):
        return []


def _pending_row(status="ingested"):
    return {"id": 1, "sweep_id": 1, "config_id": CONFIG.id, "status": status,
            "evaluated_combos": []}


def _settings():
    return Settings()


def _patch_common(monkeypatch, run_eval_combo_calls, budget_exhausted_after=None,
                   propose_result=None, summary_text="summary"):
    async def fake_run_eval_combo(db, provider, chat_provider, judge, ner, settings,
                                   config, combo, sweep_id):
        run_eval_combo_calls.append(combo)
        return len(run_eval_combo_calls)

    async def fake_check_budget(db, sweep_id, budget):
        if budget_exhausted_after is None:
            return False
        return len(run_eval_combo_calls) >= budget_exhausted_after

    async def fake_propose_next_config(provider, corpus_summary, tried, remaining_budget,
                                        user_hint=None):
        return propose_result or NextConfigResult(converged=True, candidate=None)

    async def fake_generate_sweep_summary(provider, corpus_summary, tried, stop_reason, best):
        return summary_text

    monkeypatch.setattr(sweep, "_run_eval_combo", fake_run_eval_combo)
    monkeypatch.setattr(sweep, "check_ingestion_budget", fake_check_budget)
    monkeypatch.setattr(sweep, "check_eval_budget", fake_check_budget)
    monkeypatch.setattr(sweep, "propose_next_config", fake_propose_next_config)
    monkeypatch.setattr(sweep, "generate_sweep_summary", fake_generate_sweep_summary)
    monkeypatch.setattr(sweep, "load_quality_gate", lambda path: {})
    monkeypatch.setattr(sweep, "load_scoring_weights", lambda path: {})


@pytest.mark.asyncio
async def test_run_sweep_loop_stops_on_budget_exhaustion_mid_grid(monkeypatch):
    run_eval_combo_calls = []
    _patch_common(monkeypatch, run_eval_combo_calls, budget_exhausted_after=2)
    db = FakeDatabase(_pending_row())

    await sweep.run_sweep_loop(db, object(), object(), object(), object(), _settings(), sweep_id=1)

    assert len(run_eval_combo_calls) == 2
    assert db.finalize_calls[-1][1] == "stopped_budget"


@pytest.mark.asyncio
async def test_run_sweep_loop_sets_awaiting_suggestion_on_convergence(monkeypatch):
    run_eval_combo_calls = []
    _patch_common(
        monkeypatch, run_eval_combo_calls,
        propose_result=NextConfigResult(converged=True, candidate=None),
    )
    db = FakeDatabase(_pending_row())

    await sweep.run_sweep_loop(db, object(), object(), object(), object(), _settings(), sweep_id=1)

    assert len(run_eval_combo_calls) == 8
    assert db.status_updates[-1] == (1, "awaiting_suggestion", None)


@pytest.mark.asyncio
async def test_run_sweep_loop_queues_new_candidate_from_next_config_subagent(monkeypatch):
    run_eval_combo_calls = []
    _patch_common(
        monkeypatch, run_eval_combo_calls,
        propose_result=NextConfigResult(converged=False, candidate=NEW_CANDIDATE),
    )
    db = FakeDatabase(_pending_row())

    # After the first config's grid, the subagent proposes NEW_CANDIDATE; the loop then
    # finds no more pending configs (FakeDatabase only ever seeds one) and stops there --
    # this test only checks the candidate got queued correctly.
    await sweep.run_sweep_loop(db, object(), object(), object(), object(), _settings(), sweep_id=1)

    assert len(db.created_sweep_configs) == 1
    created = db.created_sweep_configs[0]
    assert created["source"] == "loop"
    assert created["status"] == "proposed"


@pytest.mark.asyncio
async def test_run_sweep_loop_generates_final_summary_on_stopped_budget(monkeypatch):
    run_eval_combo_calls = []
    _patch_common(
        monkeypatch, run_eval_combo_calls, budget_exhausted_after=1, summary_text="the write-up",
    )
    db = FakeDatabase(_pending_row())

    await sweep.run_sweep_loop(db, object(), object(), object(), object(), _settings(), sweep_id=1)

    assert len(db.finalize_calls) == 1
    sweep_id, status, summary_markdown, best_config_id, best_score = db.finalize_calls[0]
    assert status == "stopped_budget"
    assert summary_markdown == "the write-up"


@pytest.mark.asyncio
async def test_apply_user_suggestion_with_explicit_config_creates_row_and_resumes(monkeypatch):
    resumed = []

    async def fake_run_sweep_loop(db, provider, chat_provider, judge, ner, settings, sweep_id):
        resumed.append(sweep_id)

    monkeypatch.setattr(sweep, "run_sweep_loop", fake_run_sweep_loop)
    db = FakeDatabase(pending_sweep_config=None)
    explicit_config = {
        "chunking_strategy": "sliding_window", "chunk_size": 1500, "chunk_overlap": 300,
        "embedding_provider": "local", "embedding_model": "BAAI/bge-base-en-v1.5",
        "embedding_dim": 768,
    }

    await sweep.apply_user_suggestion(
        db, object(), object(), object(), object(), _settings(), sweep_id=1,
        hint=None, explicit_config=explicit_config,
    )

    assert len(db.created_sweep_configs) == 1
    assert db.created_sweep_configs[0]["source"] == "user_suggested"
    assert ("running" in [u[1] for u in db.status_updates])
    assert resumed == [1]
