BACKFILL_PAGE_SIZE = 200


async def _iter_all_chunks(db, config_id: int, dim: int, document_id: str):
    page = 1
    while True:
        chunks, total = await db.list_chunks(
            config_id, dim, document_id, None, page, BACKFILL_PAGE_SIZE
        )
        for chunk in chunks:
            yield chunk
        if page * BACKFILL_PAGE_SIZE >= total:
            break
        page += 1


async def rebuild_cooccurrences_for_config(db, config) -> dict:
    await db.fill_mention_documents(config.id, config.embedding_dim)
    documents = await db.list_chunk_documents(config.id, config.embedding_dim)
    cooccurrences_created = 0
    for doc in documents:
        cooccurrences_created += await db.derive_cooccurrence_edges(
            config.id, config.embedding_dim, str(doc["document_id"])
        )
    return {"documents": len(documents), "cooccurrences_created": cooccurrences_created}


async def backfill_graph_for_config(db, ner, config) -> dict:
    documents = await db.list_chunk_documents(config.id, config.embedding_dim)
    chunks_processed = 0
    entities_touched: set[str] = set()
    mentions_created = 0
    cooccurrences_created = 0

    for doc in documents:
        document_id = str(doc["document_id"])
        async for chunk in _iter_all_chunks(db, config.id, config.embedding_dim, document_id):
            chunks_processed += 1
            extracted = await ner.extract_windowed(chunk["chunk_text"])
            for entity in extracted:
                entity_id = await db.insert_entity(entity.entity_type, entity.normalized_text)
                entities_touched.add(entity_id)
                await db.insert_mention(
                    config.id, document_id, str(chunk["id"]), entity_id,
                    entity.char_start, entity.char_end,
                )
                mentions_created += 1
        cooccurrences_created += await db.derive_cooccurrence_edges(
            config.id, config.embedding_dim, document_id
        )

    return {
        "chunks_processed": chunks_processed,
        "entities_touched": len(entities_touched),
        "mentions_created": mentions_created,
        "cooccurrences_created": cooccurrences_created,
    }
