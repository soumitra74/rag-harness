import uuid

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
async def test_batched_entity_and_mention_inserts_are_idempotent(db):
    config = await db.create_experiment_config(
        chunking_strategy="sliding_window", chunk_size=500, chunk_overlap=50,
        embedding_provider="local", embedding_model=f"batch-test-{uuid.uuid4()}", embedding_dim=768,
    )
    async with db.pool.acquire() as conn:
        document_id = str(await conn.fetchval(
            "INSERT INTO documents (filename) VALUES ($1) RETURNING id", f"batch-{uuid.uuid4()}.pdf",
        ))
    chunk_id = str(uuid.uuid4())
    pairs = [("DATA_COUNTER_ID", "9017"), ("REPORT_NUMBER", "tm-104598")]

    entity_ids = await db.upsert_entities(pairs)
    assert await db.upsert_entities(pairs) == entity_ids
    assert set(entity_ids) == set(pairs)

    mentions = [
        (chunk_id, entity_ids[("DATA_COUNTER_ID", "9017")], 4, 8),
        (chunk_id, entity_ids[("REPORT_NUMBER", "tm-104598")], 20, 30),
    ]
    await db.insert_mentions(config.id, document_id, mentions)
    await db.insert_mentions(config.id, document_id, mentions)

    async with db.pool.acquire() as conn:
        count = await conn.fetchval(
            "SELECT COUNT(*) FROM graph_edges WHERE edge_type = 'MENTIONS' AND config_id = $1 "
            "AND document_id = $2::uuid",
            config.id, document_id,
        )
        await conn.execute(
            "DELETE FROM graph_edges WHERE config_id = $1 AND document_id = $2::uuid",
            config.id, document_id,
        )
        await conn.execute("DELETE FROM documents WHERE id = $1::uuid", document_id)
    assert count == 2
