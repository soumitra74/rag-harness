from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from app.services.reset_db import (
    BackupFailedError,
    ResetBlockedError,
    backup_before_reset,
    execute_reset,
    preview_reset,
    remove_generated_files,
    resolve_categories,
)

router = APIRouter()

REQUIRED_CONFIRM_PHRASE = "RESET"


class ResetPreviewRequest(BaseModel):
    categories: list[str]


class ResetExecuteRequest(BaseModel):
    categories: list[str]
    confirm_phrase: str


@router.post("/api/admin/reset/preview")
async def reset_preview(request: Request, body: ResetPreviewRequest):
    db = request.app.state.db
    categories = resolve_categories(body.categories)
    return await preview_reset(db, categories)


@router.post("/api/admin/reset/execute")
async def reset_execute(request: Request, body: ResetExecuteRequest):
    if body.confirm_phrase != REQUIRED_CONFIRM_PHRASE:
        raise HTTPException(
            status_code=400,
            detail=f"confirm_phrase must be {REQUIRED_CONFIRM_PHRASE!r}",
        )

    db = request.app.state.db
    settings = request.app.state.settings
    categories = resolve_categories(body.categories)
    chunk_tables = await db.list_chunk_tables() if "ingestion_data" in categories else []

    try:
        backup_path = await backup_before_reset(
            categories, settings.reset_backup_dir,
            database_url=settings.database_url, chunk_tables=chunk_tables,
        )
        await execute_reset(db, categories)
    except (BackupFailedError, ResetBlockedError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    removed_files = remove_generated_files(settings.generated_dir) if "generated_code" in categories else []
    return {
        "categories": sorted(categories),
        "backup_path": backup_path,
        "removed_generated_files": removed_files,
    }
