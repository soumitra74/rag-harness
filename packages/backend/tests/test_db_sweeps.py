import time

import pytest

from app.config import Settings
from app.db import Database
from app.services.ingest import record_ingestion_metrics
from app.services.sweep_budget import check_ingestion_budget


@pytest.fixture
async def db():
    # Settings.database_url's hardcoded default still points at port 15432
    # (pre-dates the .env.example port reassignment to 15443 -- see
    # .env.example and app/mcp-server/harness_mcp/config.py, both already on
    # 15443). Port 15432 on this host is a different, unrelated project's
    # Postgres (tagore), so this test must not rely on the Settings default.
    settings = Settings(database_url="postgresql://raguser:ragpass@localhost:15443/ragdb")
    database = Database(settings.database_url)
    await database.connect()
    await database.migrate(settings)
    yield database
    await database.close()


@pytest.mark.asyncio
async def test_sweeps_tables_exist(db):
    async with db.pool.acquire() as conn:
        assert await conn.fetchval("SELECT to_regclass('sweeps')") is not None
        assert await conn.fetchval("SELECT to_regclass('sweep_configs')") is not None
        assert await conn.fetchval(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name='eval_cycles' AND column_name='sweep_id'"
        ) == "sweep_id"
        assert await conn.fetchval(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name='ingestion_runs' AND column_name='sweep_id'"
        ) == "sweep_id"
        assert await conn.fetchval(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name='ingestion_runs' AND column_name='started_at'"
        ) == "started_at"
        assert await conn.fetchval(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name='ingestion_runs' AND column_name='completed_at'"
        ) == "completed_at"
        assert await conn.fetchval(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name='sweeps' AND column_name='corpus_snapshot'"
        ) == "corpus_snapshot"


@pytest.mark.asyncio
async def test_sum_ingestion_and_eval_spend(db):
    async with db.pool.acquire() as conn:
        sweep_id = await conn.fetchval(
            "INSERT INTO sweeps (ingestion_budget, eval_budget) "
            "VALUES ('{}'::jsonb, '{}'::jsonb) RETURNING id"
        )
        config_id = await conn.fetchval(
            "INSERT INTO experiment_configs "
            "(chunking_strategy, chunk_size, chunk_overlap, embedding_provider, "
            " embedding_model, embedding_dim) "
            "VALUES ('sliding_window', 1000, 200, 'local', 'test-model', 8) RETURNING id"
        )
        try:
            await conn.execute(
                "INSERT INTO ingestion_runs "
                "(config_id, sweep_id, results, total_chunks, total_time_ms, error_count, "
                " input_size_bytes, vector_db_size_bytes, cost_usd) "
                "VALUES ($1, $2, '[]'::jsonb, 0, 60000, 0, 0, 0, 2.5)",
                config_id, sweep_id,
            )
            await conn.execute(
                "INSERT INTO eval_cycles "
                "(config_id, sweep_id, status, total, aggregates, started_at, completed_at) "
                "VALUES ($1, $2, 'completed', 1, '{\"total_cost_usd\": 1.5}'::jsonb, "
                " now() - interval '30 seconds', now())",
                config_id, sweep_id,
            )

            ingestion_spend = await db.sum_ingestion_spend(sweep_id)
            eval_spend = await db.sum_eval_spend(sweep_id)

            assert ingestion_spend == {"cost_usd": 2.5, "minutes": 1.0, "iterations": 1}
            assert eval_spend["cost_usd"] == 1.5
            assert eval_spend["minutes"] == pytest.approx(0.5, rel=0.05)
            assert eval_spend["iterations"] == 1
        finally:
            await conn.execute("DELETE FROM eval_cycles WHERE sweep_id = $1", sweep_id)
            await conn.execute("DELETE FROM ingestion_runs WHERE sweep_id = $1", sweep_id)
            await conn.execute("DELETE FROM experiment_configs WHERE id = $1", config_id)
            await conn.execute("DELETE FROM sweeps WHERE id = $1", sweep_id)


@pytest.mark.asyncio
async def test_create_sweep_and_sweep_config_round_trip(db):
    sweep_id = await db.create_sweep({"cost_usd": 10.0}, {"cost_usd": 5.0})
    try:
        async with db.pool.acquire() as conn:
            config_id = await conn.fetchval(
                "INSERT INTO experiment_configs "
                "(chunking_strategy, chunk_size, chunk_overlap, embedding_provider, "
                " embedding_model, embedding_dim) "
                "VALUES ('sliding_window', 1000, 200, 'local', 'test-model', 8) RETURNING id"
            )
        try:
            sweep = await db.get_sweep(sweep_id)
            assert sweep["status"] == "bootstrapping"
            assert sweep["ingestion_budget"] == {"cost_usd": 10.0}

            await db.create_sweep_config(sweep_id, config_id, "a rationale", "proposed", "bootstrap")
            await db.update_sweep_status(sweep_id, "failed", error_message="boom")

            sweep = await db.get_sweep(sweep_id)
            assert sweep["status"] == "failed"
            assert sweep["error_message"] == "boom"
        finally:
            async with db.pool.acquire() as conn:
                await conn.execute("DELETE FROM sweep_configs WHERE sweep_id = $1", sweep_id)
                await conn.execute("DELETE FROM experiment_configs WHERE id = $1", config_id)
    finally:
        async with db.pool.acquire() as conn:
            await conn.execute("DELETE FROM sweeps WHERE id = $1", sweep_id)


@pytest.mark.asyncio
async def test_pause_interrupted_sweeps_updates_running_and_bootstrapping_only(db):
    # pause_interrupted_sweeps() is a blanket startup-recovery operation by
    # design (same pattern as the existing abort_running_eval_cycles()), so
    # it necessarily affects every running/bootstrapping sweep in the
    # database, not just this test's own rows -- including a real one in
    # progress on this shared dev DB (review finding I-7). Snapshot and
    # restore any pre-existing running/bootstrapping sweeps so this test
    # exercises the real function without disturbing concurrent real work.
    async with db.pool.acquire() as conn:
        pre_existing = await conn.fetch(
            "SELECT id, status FROM sweeps WHERE status IN ('running', 'bootstrapping')"
        )

    running_id = await db.create_sweep({}, {})
    bootstrapping_id = await db.create_sweep({}, {})
    awaiting_id = await db.create_sweep({}, {})
    try:
        await db.update_sweep_status(running_id, "running")
        await db.update_sweep_status(awaiting_id, "awaiting_approval")
        # bootstrapping_id is left at its default status.

        paused_count = await db.pause_interrupted_sweeps()

        # >= rather than == : this runs against the shared dev DB, which may
        # have other leftover running/bootstrapping sweeps from elsewhere.
        assert paused_count >= 2
        assert (await db.get_sweep(running_id))["status"] == "paused"
        assert (await db.get_sweep(bootstrapping_id))["status"] == "paused"
        assert (await db.get_sweep(awaiting_id))["status"] == "awaiting_approval"
    finally:
        async with db.pool.acquire() as conn:
            for sweep_id in (running_id, bootstrapping_id, awaiting_id):
                await conn.execute("DELETE FROM sweeps WHERE id = $1", sweep_id)
            for row in pre_existing:
                await conn.execute(
                    "UPDATE sweeps SET status = $1 WHERE id = $2", row["status"], row["id"],
                )


@pytest.mark.asyncio
async def test_count_rows_matches_a_real_select_count(db):
    async with db.pool.acquire() as conn:
        expected = await conn.fetchval("SELECT count(*) FROM sweeps")
    assert await db.count_rows("sweeps") == expected


@pytest.mark.asyncio
async def test_list_chunk_tables_and_reset_tables_drops_them(db):
    async with db.pool.acquire() as conn:
        await conn.execute("CREATE TABLE IF NOT EXISTS chunks_dim_999999 (id INT)")
    try:
        tables = await db.list_chunk_tables()
        assert "chunks_dim_999999" in tables

        await db.reset_tables(["chunks_dim_999999"], [])

        async with db.pool.acquire() as conn:
            assert await conn.fetchval("SELECT to_regclass('chunks_dim_999999')") is None
    finally:
        async with db.pool.acquire() as conn:
            await conn.execute("DROP TABLE IF EXISTS chunks_dim_999999")


@pytest.mark.asyncio
async def test_reset_tables_truncates_and_restarts_identity(db):
    async with db.pool.acquire() as conn:
        await conn.execute("CREATE TABLE IF NOT EXISTS reset_db_test_table (id SERIAL PRIMARY KEY)")
        await conn.execute("INSERT INTO reset_db_test_table DEFAULT VALUES")
        await conn.execute("INSERT INTO reset_db_test_table DEFAULT VALUES")
    try:
        await db.reset_tables([], [["reset_db_test_table"]])

        async with db.pool.acquire() as conn:
            assert await conn.fetchval("SELECT count(*) FROM reset_db_test_table") == 0
            new_id = await conn.fetchval(
                "INSERT INTO reset_db_test_table DEFAULT VALUES RETURNING id"
            )
            assert new_id == 1
    finally:
        async with db.pool.acquire() as conn:
            await conn.execute("DROP TABLE IF EXISTS reset_db_test_table")


@pytest.mark.asyncio
async def test_reset_tables_refuses_and_rolls_back_when_rca_sessions_would_be_orphaned(db):
    """Real-DB regression test for review finding C-2: truncating
    eval_cycles/eval_queries without also truncating rca_sessions (which
    references both) must fail outright -- not cascade-delete the RCA
    data, and not leave anything partially truncated."""
    async with db.pool.acquire() as conn:
        config_id = await conn.fetchval(
            "INSERT INTO experiment_configs "
            "(chunking_strategy, chunk_size, chunk_overlap, embedding_provider, "
            " embedding_model, embedding_dim) "
            "VALUES ('sliding_window', 1000, 200, 'local', 'test-model', 8) RETURNING id"
        )
        cycle_id = await conn.fetchval(
            "INSERT INTO eval_cycles (config_id, status, total) "
            "VALUES ($1, 'completed', 1) RETURNING id",
            config_id,
        )
        query_id = await conn.fetchval(
            "INSERT INTO eval_queries (query_text, gold_answer) "
            "VALUES ('q', 'a') RETURNING id"
        )
        session_id = await conn.fetchval(
            "INSERT INTO rca_sessions (cycle_id, query_id) VALUES ($1, $2) RETURNING id",
            cycle_id, query_id,
        )
    try:
        with pytest.raises(Exception):
            await db.reset_tables([], [["eval_cycles", "eval_runs"]])

        async with db.pool.acquire() as conn:
            assert await conn.fetchval(
                "SELECT count(*) FROM eval_cycles WHERE id = $1", cycle_id
            ) == 1
            assert await conn.fetchval(
                "SELECT count(*) FROM rca_sessions WHERE id = $1", session_id
            ) == 1
    finally:
        async with db.pool.acquire() as conn:
            await conn.execute("DELETE FROM rca_sessions WHERE id = $1", session_id)
            await conn.execute("DELETE FROM eval_queries WHERE id = $1", query_id)
            await conn.execute("DELETE FROM eval_cycles WHERE id = $1", cycle_id)
            await conn.execute("DELETE FROM experiment_configs WHERE id = $1", config_id)


@pytest.mark.asyncio
async def test_record_ingestion_metrics_feeds_the_ingestion_budget_check(db):
    """End-to-end proof (C-1 from the final review): a real ingestion run
    recorded with a sweep_id is actually counted by sum_ingestion_spend and
    can trip check_ingestion_budget -- not just a fake-backed unit test."""
    sweep_id = await db.create_sweep({}, {})
    config_id = None
    try:
        await db.ensure_chunks_table(8)
        config = await db.create_experiment_config(
            chunking_strategy="sliding_window", chunk_size=1000, chunk_overlap=200,
            embedding_provider="local", embedding_model="BAAI/bge-base-en-v1.5", embedding_dim=8,
        )
        config_id = config.id

        settings = Settings(local_compute_cost_per_hour=3600.0)
        start = time.monotonic() - 2.0  # pretend 2 seconds elapsed -> ~$2 at this rate
        await record_ingestion_metrics(
            db, config, results=[], input_size_bytes=0, start=start, settings=settings,
            sweep_id=sweep_id,
        )

        spent = await db.sum_ingestion_spend(sweep_id)
        assert spent["cost_usd"] > 0
        assert spent["iterations"] == 1

        assert await check_ingestion_budget(db, sweep_id, {"cost_usd": 1.0}) is True
        assert await check_ingestion_budget(db, sweep_id, {"cost_usd": 100.0}) is False
    finally:
        async with db.pool.acquire() as conn:
            await conn.execute("DELETE FROM ingestion_runs WHERE sweep_id = $1", sweep_id)
            if config_id is not None:
                await conn.execute("DELETE FROM experiment_configs WHERE id = $1", config_id)
            await conn.execute("DELETE FROM sweeps WHERE id = $1", sweep_id)
