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
async def test_codegen_spend_and_artifacts_round_trip(db):
    sweep_id = await db.create_sweep(
        {"cost_usd": 1, "minutes": 1, "iterations": 1},
        {"cost_usd": 1, "minutes": 1, "iterations": 1},
        codegen_budget={"cost_usd": 5, "minutes": 10, "attempts": 4},
    )
    try:
        sweep = await db.get_sweep(sweep_id)
        assert sweep["codegen_budget"] == {"cost_usd": 5, "minutes": 10, "attempts": 4}

        await db.insert_codegen_run(sweep_id, "ocr", 1, False, "no text", 0.25, 30_000)
        await db.insert_codegen_run(sweep_id, "ocr", 2, True, None, 0.5, 90_000)
        spent = await db.sum_codegen_spend(sweep_id)
        assert spent["attempts"] == 2
        assert spent["cost_usd"] == pytest.approx(0.75)
        assert spent["minutes"] == pytest.approx(2.0)

        await db.upsert_generated_artifact(sweep_id, "ocr", "def ocr_page(p, n): return ''")
        await db.upsert_generated_artifact(sweep_id, "ocr", "def ocr_page(p, n): return 'v2'")
        artifacts = await db.get_generated_artifacts(sweep_id)
        assert artifacts == {"ocr": "def ocr_page(p, n): return 'v2'"}
    finally:
        async with db.pool.acquire() as conn:
            await conn.execute("DELETE FROM generated_artifacts WHERE sweep_id = $1", sweep_id)
            await conn.execute("DELETE FROM codegen_runs WHERE sweep_id = $1", sweep_id)
            await conn.execute("DELETE FROM sweeps WHERE id = $1", sweep_id)


@pytest.mark.asyncio
async def test_sweep_without_codegen_budget_has_no_codegen_budget(db):
    sweep_id = await db.create_sweep(
        {"cost_usd": 1, "minutes": 1, "iterations": 1},
        {"cost_usd": 1, "minutes": 1, "iterations": 1},
    )
    try:
        assert (await db.get_sweep(sweep_id))["codegen_budget"] is None
    finally:
        async with db.pool.acquire() as conn:
            await conn.execute("DELETE FROM sweeps WHERE id = $1", sweep_id)


@pytest.mark.asyncio
async def test_llm_usage_counts_toward_ingestion_spend_except_codegen(db):
    sweep_id = await db.create_sweep(
        {"cost_usd": 1, "minutes": 1, "iterations": 1},
        {"cost_usd": 1, "minutes": 1, "iterations": 1},
    )
    try:
        await db.insert_llm_usage(sweep_id, "inspect_corpus", "openai", "gpt-x", 1000, 500, 0.25)
        await db.insert_llm_usage(sweep_id, "summary", "openai", "gpt-x", 1000, 500, 0.5)
        await db.insert_llm_usage(sweep_id, "codegen", "openai", "gpt-x", 1000, 500, 9.0)

        spent = await db.sum_ingestion_spend(sweep_id)

        assert spent["cost_usd"] == pytest.approx(0.75)
    finally:
        async with db.pool.acquire() as conn:
            await conn.execute("DELETE FROM llm_usage WHERE sweep_id = $1", sweep_id)
            await conn.execute("DELETE FROM sweeps WHERE id = $1", sweep_id)
