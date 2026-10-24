from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel

from app.services.graph_backfill import backfill_graph_for_config, rebuild_cooccurrences_for_config

router = APIRouter()

DEFAULT_PAGE_SIZE = 25


class BackfillRequest(BaseModel):
    config_id: int


@router.post("/api/graph/backfill")
async def backfill_graph(request: Request, body: BackfillRequest):
    db = request.app.state.db
    ner = request.app.state.ner

    config = await db.get_experiment_config(body.config_id)
    if config is None:
        raise HTTPException(status_code=400, detail=f"Unknown config_id: {body.config_id}")

    return await backfill_graph_for_config(db, ner, config)


@router.post("/api/graph/rebuild-cooccurrences")
async def rebuild_cooccurrences(request: Request, body: BackfillRequest):
    db = request.app.state.db

    config = await db.get_experiment_config(body.config_id)
    if config is None:
        raise HTTPException(status_code=400, detail=f"Unknown config_id: {body.config_id}")

    return await rebuild_cooccurrences_for_config(db, config)


@router.get("/api/graph/entities")
async def list_graph_entities(
    request: Request,
    config_id: int,
    document_id: str,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=DEFAULT_PAGE_SIZE, ge=1, le=200),
):
    db = request.app.state.db
    config = await db.get_experiment_config(config_id)
    if config is None:
        raise HTTPException(status_code=400, detail=f"Unknown config_id: {config_id}")
    entities, total = await db.list_graph_entities(
        config_id, config.embedding_dim, document_id, page, page_size
    )
    return {
        "entities": [
            {
                "entity_type": e["entity_type"],
                "entity_text": e["entity_text"],
                "chunk_id": str(e["chunk_id"]),
                "page_number": e["page_number"],
                "filename": e["filename"],
            }
            for e in entities
        ],
        "total": total,
        "page": page,
        "page_size": page_size,
    }


@router.get("/api/graph/cooccurrences")
async def list_cooccurrences(
    request: Request,
    document_id: str,
    entity_type: str | None = None,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=DEFAULT_PAGE_SIZE, ge=1, le=200),
):
    db = request.app.state.db
    pairs, total = await db.list_cooccurrences(document_id, entity_type, page, page_size)
    return {
        "cooccurrences": [dict(p) for p in pairs],
        "total": total,
        "page": page,
        "page_size": page_size,
    }
