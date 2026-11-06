import pytest

from app.config import Settings
from app.db import Database


@pytest.fixture
async def db():
    settings = Settings(database_url="postgresql://raguser:ragpass@localhost:15443/ragdb")
    database = Database(settings.database_url)
    await database.connect()
    await database.migrate(settings)
    yield database
    await database.close()


@pytest.mark.asyncio
async def test_golden_qa_llm_cost_counts_toward_eval_not_ingestion(db):
    sweep_id = await db.create_sweep(
        {"cost_usd": 1, "minutes": 1, "iterations": 1},
        {"cost_usd": 1, "minutes": 1, "iterations": 1},
    )
    try:
        await db.insert_llm_usage(sweep_id, "golden_qa", "openai", "gpt-6-luna", 1000, 100, 0.25)
        await db.insert_llm_usage(sweep_id, "inspect_corpus", "openai", "gpt-6-luna", 1000, 100, 0.5)

        ingestion = await db.sum_ingestion_spend(sweep_id)
        eval_spend = await db.sum_eval_spend(sweep_id)

        assert ingestion["cost_usd"] == pytest.approx(0.5)
        assert eval_spend["cost_usd"] == pytest.approx(0.25)
    finally:
        async with db.pool.acquire() as conn:
            await conn.execute("DELETE FROM llm_usage WHERE sweep_id = $1", sweep_id)
            await conn.execute("DELETE FROM sweeps WHERE id = $1", sweep_id)
