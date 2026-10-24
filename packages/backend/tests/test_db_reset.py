import pytest

from app.config import Settings
from app.db import Database
from app.services.reset_db import execute_reset, resolve_categories


@pytest.fixture
async def db():
    settings = Settings(database_url="postgresql://raguser:ragpass@localhost:15443/ragdb")
    database = Database(settings.database_url)
    await database.connect()
    await database.migrate(settings)
    yield database
    await database.close()


@pytest.mark.asyncio
async def test_reset_ingestion_data_succeeds_with_sweep_and_eval_rows_present(db):
    async with db.pool.acquire() as conn:
        config_id = await conn.fetchval(
            "INSERT INTO experiment_configs (chunking_strategy, chunk_size, chunk_overlap, "
            "embedding_provider, embedding_model, embedding_dim) "
            "VALUES ('sliding_window', 1000, 200, 'local', 'reset-test', 8) RETURNING id"
        )
        sweep_id = await conn.fetchval(
            "INSERT INTO sweeps (ingestion_budget, eval_budget, best_config_id) "
            "VALUES ('{}'::jsonb, '{}'::jsonb, $1) RETURNING id",
            config_id,
        )
        await conn.execute(
            "INSERT INTO sweep_configs (sweep_id, config_id, status, source) "
            "VALUES ($1, $2, 'evaluated', 'bootstrap')",
            sweep_id, config_id,
        )
        await conn.execute(
            "INSERT INTO eval_cycles (config_id, status, total, sweep_id) "
            "VALUES ($1, 'completed', 1, $2)",
            config_id, sweep_id,
        )

    await execute_reset(db, resolve_categories(["ingestion_data"]))

    async with db.pool.acquire() as conn:
        for table in ("experiment_configs", "sweeps", "sweep_configs", "eval_cycles"):
            assert await conn.fetchval(f"SELECT count(*) FROM {table}") == 0
