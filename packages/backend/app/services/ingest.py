import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from app.db import ExperimentConfig
from app.providers.embedding_catalog import get_max_input_tokens
from app.services.chunking import chunk_page, extract_pages, normalize_page_text

LOCAL_EMBEDDING_PROVIDERS = {"ollama", "local"}


@dataclass
class IngestResult:
    document_id: str
    chunk_count: int
    page_count: int


async def ingest_pdf(
    filename: str, pdf_bytes: bytes, db, embedding_provider, config: ExperimentConfig,
    ocr_page=None,
) -> IngestResult:
    # Extracted unconditionally, even on a cache hit below, since page_count
    # is a property of the PDF itself, not of this config's chunking -- a
    # re-ingest into a different config still needs the real count, not a
    # count derived from chunks (which could undercount a page that produced
    # zero chunks).
    pages = extract_pages(pdf_bytes)
    page_count = len(pages)

    existing_document_id = await db.get_document_by_filename(filename)
    if existing_document_id is not None:
        existing_chunk_count = await db.count_chunks_for_document(
            config.id, existing_document_id, config.embedding_dim
        )
        if existing_chunk_count > 0:
            return IngestResult(
                document_id=existing_document_id, chunk_count=existing_chunk_count,
                page_count=page_count,
            )

    if ocr_page is not None:
        pages = [
            (n, normalize_page_text(await ocr_page(pdf_bytes, n)) if not text.strip() else text)
            for n, text in pages
        ]

    max_tokens = get_max_input_tokens(config.embedding_provider, config.embedding_model)
    chunks = []
    for page_number, text in pages:
        chunks.extend(
            chunk_page(
                config.chunking_strategy, page_number, text, config.chunk_size, config.chunk_overlap,
                max_tokens=max_tokens,
            )
        )

    document_id = existing_document_id or await db.insert_document(filename)

    if not chunks:
        return IngestResult(document_id=document_id, chunk_count=0, page_count=page_count)

    embeddings = await embedding_provider.embed_documents([c.text for c in chunks])
    await db.insert_chunks(document_id, config.id, config.embedding_dim, chunks, embeddings)
    return IngestResult(document_id=document_id, chunk_count=len(chunks), page_count=page_count)


async def record_ingestion_metrics(
    db, config: ExperimentConfig, results: list[dict], input_size_bytes: int, start: float,
    settings, sweep_id: int | None = None,
) -> dict:
    total_time_ms = int((time.monotonic() - start) * 1000)
    total_chunks = sum(r.get("chunk_count", 0) for r in results)
    total_pages = sum(r.get("page_count", 0) for r in results)
    error_count = sum(1 for r in results if "error" in r)
    vector_db_size_bytes = await db.get_config_vector_storage_bytes(
        config.id, config.embedding_dim
    )

    # No embedding provider returns token/usage counts (embed_documents ->
    # list[list[float]] only), so a real per-token cost can't be computed --
    # see the optimization-harness plan's Task 3 ruling. Approximated by
    # elapsed time for local/ollama providers only (genuinely compute-time
    # bound); left at 0.0 for the openai provider rather than fabricating a
    # time-based number for a cost that's actually token-priced.
    cost_usd = (
        total_time_ms / 1000 / 3600 * settings.local_compute_cost_per_hour
        if config.embedding_provider in LOCAL_EMBEDDING_PROVIDERS
        else 0.0
    )
    completed_at = datetime.now(timezone.utc)
    started_at = completed_at - timedelta(milliseconds=total_time_ms)

    await db.record_ingestion_run(
        config.id, results, total_chunks, total_time_ms, error_count,
        input_size_bytes, vector_db_size_bytes, total_pages,
        started_at=started_at, completed_at=completed_at, cost_usd=cost_usd,
        sweep_id=sweep_id,
    )

    return {
        "chunks": total_chunks,
        "total_pages": total_pages,
        "total_time_ms": total_time_ms,
        "error_count": error_count,
        "input_size_bytes": input_size_bytes,
        "vector_db_size_bytes": vector_db_size_bytes,
    }
