import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from app.services.codegen import ARTIFACT_FILES


@dataclass
class ResetCategory:
    tables: list[str]
    requires: list[str]


# Dynamic per-config chunk tables (chunks_dim_N) aren't listed here -- their
# names depend on which embedding dims are in use, so execute_reset
# enumerates them separately (via information_schema) rather than the
# preview counting rows in tables it can't name in advance.
RESET_CATEGORIES: dict[str, ResetCategory] = {
    "ingestion_data": ResetCategory(
        tables=[
            "sweep_configs", "codegen_runs", "generated_artifacts", "sweeps",
            "ingestion_runs", "experiment_configs", "llm_usage",
        ],
        requires=["graph_data", "eval_history"],
    ),
    "documents": ResetCategory(
        tables=["documents"], requires=["graph_data", "golden_qa"],
    ),
    "graph_data": ResetCategory(
        tables=["graph_entities", "graph_edges"], requires=[],
    ),
    "golden_qa": ResetCategory(
        tables=["eval_queries", "eval_query_gold_pages"], requires=["eval_history"],
    ),
    "eval_history": ResetCategory(
        tables=["eval_cycles", "eval_runs"], requires=["rca_sessions"],
    ),
    "rca_sessions": ResetCategory(
        tables=["rca_sessions", "rca_messages"], requires=[],
    ),
    "generated_code": ResetCategory(
        tables=["generated_artifacts"], requires=[],
    ),
}


class ResetBlockedError(Exception):
    """Raised when the resolved categories can't be safely reset -- e.g. a
    table outside the selection (such as rca_sessions, which is never
    touched unless explicitly selected) still references data inside it.
    Reset execution never uses CASCADE specifically so this surfaces as a
    clear, caught error instead of silently deleting unselected data."""


class BackupFailedError(Exception):
    """Raised when the pre-delete pg_dump backup didn't actually succeed --
    reset must never proceed past a backup that failed."""


def resolve_categories(selected: list[str]) -> set[str]:
    resolved: set[str] = set()
    stack = list(selected)
    while stack:
        name = stack.pop()
        if name in resolved:
            continue
        resolved.add(name)
        stack.extend(RESET_CATEGORIES[name].requires)
    return resolved


async def preview_reset(db, categories: set[str]) -> dict[str, int]:
    if not categories:
        return {}
    counts = {}
    for category in categories:
        for table in RESET_CATEGORIES[category].tables:
            counts[table] = await db.count_rows(table)
    return counts


async def _default_run_subprocess(args: list[str]) -> int:
    process = await asyncio.create_subprocess_exec(*args)
    await process.wait()
    return process.returncode


def _resolved_tables(categories: set[str]) -> set[str]:
    return {table for category in categories for table in RESET_CATEGORIES[category].tables}


async def backup_before_reset(
    categories: set[str], backup_dir: str, run_subprocess=None,
    database_url: str | None = None, chunk_tables: list[str] | None = None,
) -> str:
    run_subprocess = run_subprocess or _default_run_subprocess
    Path(backup_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_path = str(Path(backup_dir) / f"reset-{timestamp}.sql")

    args = ["pg_dump"]
    if database_url:
        args += ["-d", database_url]
    for table in sorted(_resolved_tables(categories)) + sorted(chunk_tables or []):
        args += ["-t", table]
    args += ["-f", backup_path]

    exit_code = await run_subprocess(args)
    if exit_code != 0:
        raise BackupFailedError(f"pg_dump exited with code {exit_code}; reset aborted")
    return backup_path


def remove_generated_files(generated_dir: str) -> list[str]:
    removed = []
    for filename in ARTIFACT_FILES.values():
        path = Path(generated_dir) / filename
        if path.is_file():
            path.unlink()
            removed.append(filename)
    return removed


async def execute_reset(db, categories: set[str]) -> None:
    chunk_tables = await db.list_chunk_tables() if "ingestion_data" in categories else []
    # One TRUNCATE for every resolved table: Postgres checks foreign keys at
    # TRUNCATE time, so a referencing table cleared in a later statement still
    # blocks an earlier one.
    tables = sorted(_resolved_tables(categories))

    try:
        await db.reset_tables(chunk_tables, [tables] if tables else [])
    except Exception as exc:
        raise ResetBlockedError(
            f"Reset blocked: {exc}. Data outside the selected categories (for example, "
            f"RCA sessions) may still reference what you're trying to clear -- select "
            f"that category too, or clear it first."
        ) from exc
