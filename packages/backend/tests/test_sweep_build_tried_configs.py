import pytest

from app.config import Settings
from app.db import ExperimentConfig
from app.services.sweep_next_config import NextConfigResult, TriedConfig
import app.services.sweep as sweep
from tests.codegen_fakes import CodegenFakeMethods

CONFIG_A = ExperimentConfig(
    id=1, chunking_strategy="sliding_window", chunk_size=1000, chunk_overlap=200,
    embedding_provider="local", embedding_model="BAAI/bge-base-en-v1.5", embedding_dim=768,
)
# Deliberately shares chunking_strategy and chunk_size with CONFIG_A (differs
# only in embedding_model) to exercise the "match by field, not id" bug.
CONFIG_B = ExperimentConfig(
    id=2, chunking_strategy="sliding_window", chunk_size=1000, chunk_overlap=200,
    embedding_provider="local", embedding_model="a-different-model", embedding_dim=768,
)


class FakeDatabase(CodegenFakeMethods):
    def __init__(self, sweep_configs, cycles):
        self._sweep_configs = sweep_configs
        self._cycles = cycles
        self._configs = {1: CONFIG_A, 2: CONFIG_B}
        self.finalize_calls = []

    async def get_experiment_config(self, config_id):
        return self._configs[config_id]

    async def list_sweep_configs(self, sweep_id):
        return self._sweep_configs

    async def list_eval_cycles_for_sweep(self, sweep_id):
        return self._cycles

    async def finalize_sweep(self, sweep_id, status, summary_markdown, best_config_id, best_score):
        self.finalize_calls.append((sweep_id, status, summary_markdown, best_config_id, best_score))

    async def update_sweep_status(self, sweep_id, status, error_message=None):
        self.status_updates = getattr(self, "status_updates", [])
        self.status_updates.append((sweep_id, status, error_message))


def _settings(monkeypatch):
    monkeypatch.setattr(sweep, "load_quality_gate", lambda path: {})
    monkeypatch.setattr(sweep, "load_scoring_weights", lambda path: {"hit_rate": 1.0})
    return Settings()


@pytest.mark.asyncio
async def test_build_tried_configs_picks_best_not_last_score_per_config(monkeypatch):
    settings = _settings(monkeypatch)
    sweep_configs = [{"config_id": 1}]
    # Two cycles for the same config: a low-scoring one and a high-scoring
    # one. The bug (dict() over a descending-sorted list keeps the LAST,
    # i.e. lowest, entry per key) would report 0.2 as config 1's score.
    cycles = [
        {"config_id": 1, "aggregates": {"hit_rate": 0.9}},
        {"config_id": 1, "aggregates": {"hit_rate": 0.2}},
    ]

    tried = await sweep._build_tried_configs(FakeDatabase(sweep_configs, cycles), sweep_id=1, settings=settings)

    assert tried[0].score == pytest.approx(0.9)


@pytest.mark.asyncio
async def test_finalize_sweep_run_matches_best_config_by_id_not_by_shared_fields(monkeypatch):
    settings = _settings(monkeypatch)
    sweep_configs = [{"config_id": 1}, {"config_id": 2}]
    # Config 2 scores higher but shares chunking_strategy/chunk_size with
    # config 1 -- the old field-matching code would pick config 1 (the
    # first list match) regardless of which one actually won.
    cycles = [
        {"config_id": 1, "aggregates": {"hit_rate": 0.5}},
        {"config_id": 2, "aggregates": {"hit_rate": 0.9}},
    ]
    db = FakeDatabase(sweep_configs, cycles)

    async def fake_generate_sweep_summary(provider, corpus_summary, tried, stop_reason, best):
        return "summary"

    monkeypatch.setattr(sweep, "generate_sweep_summary", fake_generate_sweep_summary)

    await sweep.finalize_sweep_run(db, object(), settings, sweep_id=1, stop_reason="stopped_optimal")

    assert len(db.finalize_calls) == 1
    _, _, _, best_config_id, best_score = db.finalize_calls[0]
    assert best_config_id == 2
    assert best_score == pytest.approx(0.9)


def test_consecutive_non_improving_streak_resets_on_improvement():
    tried = [
        TriedConfig(config={}, score=0.5, config_id=1),
        TriedConfig(config={}, score=0.9, config_id=2),  # improves -> resets streak
        TriedConfig(config={}, score=0.6, config_id=3),  # below 0.9 -> streak 1
    ]
    assert sweep._consecutive_non_improving_streak(tried) == 1


def test_consecutive_non_improving_streak_ignores_gate_failures():
    tried = [
        TriedConfig(config={}, score=0.9, config_id=1),
        TriedConfig(config={}, score=None, config_id=2),  # gate-failed, doesn't count
        TriedConfig(config={}, score=0.5, config_id=3),
    ]
    assert sweep._consecutive_non_improving_streak(tried) == 1


@pytest.mark.asyncio
async def test_propose_and_record_next_forces_convergence_after_threshold(monkeypatch):
    """Review finding I-4: the plan's Global Constraint ("3 consecutive
    non-improving candidates triggers awaiting_suggestion") was left
    entirely to the LLM's own self-report, with nothing actually enforcing
    it. This proves the loop forces convergence on the real persisted
    scores even when propose_next_config itself returns a fresh (non-
    converged) candidate."""
    settings = _settings(monkeypatch)
    sweep_configs = [{"config_id": 1}, {"config_id": 2}, {"config_id": 3}, {"config_id": 4}]
    cycles = [
        {"config_id": 1, "aggregates": {"hit_rate": 0.9}},
        {"config_id": 2, "aggregates": {"hit_rate": 0.5}},
        {"config_id": 3, "aggregates": {"hit_rate": 0.5}},
        {"config_id": 4, "aggregates": {"hit_rate": 0.5}},
    ]
    db = FakeDatabase(sweep_configs, cycles)
    db._configs[3] = CONFIG_A
    db._configs[4] = CONFIG_A

    async def fake_propose_next_config(provider, corpus_summary, tried, remaining_budget, user_hint=None):
        return NextConfigResult(converged=False, candidate=None)  # LLM itself wants to keep going

    monkeypatch.setattr(sweep, "propose_next_config", fake_propose_next_config)

    converged = await sweep._propose_and_record_next(
        db, object(), settings, sweep_id=1, sweep_row={"ingestion_budget": {}},
    )

    assert converged is True
