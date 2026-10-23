import time

import pytest

from app.api.routes_corpus import _record_run
from app.config import Settings
from app.db import ExperimentConfig

LOCAL_CONFIG = ExperimentConfig(
    id=1, chunking_strategy="sliding_window", chunk_size=1000, chunk_overlap=200,
    embedding_provider="local", embedding_model="BAAI/bge-base-en-v1.5", embedding_dim=768,
)

OPENAI_CONFIG = ExperimentConfig(
    id=2, chunking_strategy="sliding_window", chunk_size=1000, chunk_overlap=200,
    embedding_provider="openai", embedding_model="text-embedding-3-small", embedding_dim=1536,
)


class FakeDatabase:
    def __init__(self):
        self.record_ingestion_run_calls = []

    async def get_config_vector_storage_bytes(self, config_id, dim):
        return 0

    async def record_ingestion_run(self, *args, **kwargs):
        self.record_ingestion_run_calls.append((args, kwargs))


@pytest.mark.asyncio
async def test_record_run_sets_started_and_completed_at():
    db = FakeDatabase()
    settings = Settings(local_compute_cost_per_hour=0.0)
    start = time.monotonic()

    await _record_run(db, LOCAL_CONFIG, [], 0, start, settings)

    (args, kwargs) = db.record_ingestion_run_calls[0]
    started_at = kwargs["started_at"]
    completed_at = kwargs["completed_at"]
    assert started_at is not None
    assert completed_at is not None
    assert completed_at >= started_at


@pytest.mark.asyncio
async def test_record_run_approximates_cost_by_time_for_local_provider():
    db = FakeDatabase()
    settings = Settings(local_compute_cost_per_hour=3600.0)
    start = time.monotonic() - 2.0  # pretend 2 seconds elapsed

    await _record_run(db, LOCAL_CONFIG, [], 0, start, settings)

    (args, kwargs) = db.record_ingestion_run_calls[0]
    assert kwargs["cost_usd"] == pytest.approx(2.0, rel=0.05)


@pytest.mark.asyncio
async def test_record_run_cost_is_zero_for_openai_embedding_provider():
    db = FakeDatabase()
    settings = Settings(local_compute_cost_per_hour=3600.0)
    start = time.monotonic() - 2.0

    await _record_run(db, OPENAI_CONFIG, [], 0, start, settings)

    (args, kwargs) = db.record_ingestion_run_calls[0]
    assert kwargs["cost_usd"] == 0.0


@pytest.mark.asyncio
async def test_record_run_passes_sweep_id_through_when_given():
    db = FakeDatabase()
    settings = Settings(local_compute_cost_per_hour=0.0)
    start = time.monotonic()

    await _record_run(db, LOCAL_CONFIG, [], 0, start, settings, sweep_id=42)

    (args, kwargs) = db.record_ingestion_run_calls[0]
    assert kwargs["sweep_id"] == 42


@pytest.mark.asyncio
async def test_record_run_sweep_id_defaults_to_none():
    db = FakeDatabase()
    settings = Settings(local_compute_cost_per_hour=0.0)
    start = time.monotonic()

    await _record_run(db, LOCAL_CONFIG, [], 0, start, settings)

    (args, kwargs) = db.record_ingestion_run_calls[0]
    assert kwargs["sweep_id"] is None
