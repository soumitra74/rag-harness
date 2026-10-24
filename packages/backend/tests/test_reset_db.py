import pytest

from app.services.reset_db import (
    BackupFailedError,
    ResetBlockedError,
    backup_before_reset,
    execute_reset,
    preview_reset,
    resolve_categories,
)


def test_resolve_categories_expands_documents_to_its_dependencies():
    assert resolve_categories(["documents"]) == {
        "documents", "graph_data", "golden_qa", "eval_history", "rca_sessions",
    }


def test_resolve_categories_expands_ingestion_data_to_everything_referencing_configs():
    assert resolve_categories(["ingestion_data"]) == {
        "ingestion_data", "graph_data", "eval_history", "rca_sessions",
    }


def test_resolve_categories_empty_selection_resolves_to_empty_set():
    assert resolve_categories([]) == set()


class FakeDatabase:
    def __init__(self, counts):
        self._counts = counts
        self.count_rows_calls = []

    async def count_rows(self, table):
        self.count_rows_calls.append(table)
        return self._counts.get(table, 0)


@pytest.mark.asyncio
async def test_preview_reset_with_empty_categories_does_not_touch_db():
    db = FakeDatabase({})

    result = await preview_reset(db, set())

    assert result == {}
    assert db.count_rows_calls == []


@pytest.mark.asyncio
async def test_preview_reset_returns_row_counts_for_resolved_tables():
    db = FakeDatabase({"graph_entities": 3, "graph_edges": 7})

    result = await preview_reset(db, {"graph_data"})

    assert result == {"graph_entities": 3, "graph_edges": 7}


@pytest.mark.asyncio
async def test_backup_before_reset_dumps_every_resolved_table(tmp_path):
    calls = []

    async def fake_run_subprocess(args):
        calls.append(args)
        return 0

    path = await backup_before_reset({"graph_data"}, str(tmp_path), fake_run_subprocess)

    assert len(calls) == 1
    args = calls[0]
    assert "-t" in args and "graph_entities" in args and "graph_edges" in args
    assert path.startswith(str(tmp_path))


@pytest.mark.asyncio
async def test_backup_before_reset_includes_chunk_tables(tmp_path):
    calls = []

    async def fake_run_subprocess(args):
        calls.append(args)
        return 0

    await backup_before_reset(
        {"graph_data"}, str(tmp_path), fake_run_subprocess, chunk_tables=["chunks_dim_768"],
    )

    assert "chunks_dim_768" in calls[0]


@pytest.mark.asyncio
async def test_backup_before_reset_passes_database_url(tmp_path):
    calls = []

    async def fake_run_subprocess(args):
        calls.append(args)
        return 0

    await backup_before_reset(
        {"graph_data"}, str(tmp_path), fake_run_subprocess,
        database_url="postgresql://u:p@host/db",
    )

    args = calls[0]
    assert "-d" in args
    assert args[args.index("-d") + 1] == "postgresql://u:p@host/db"


@pytest.mark.asyncio
async def test_backup_before_reset_raises_on_nonzero_exit_code(tmp_path):
    async def failing_run_subprocess(args):
        return 1

    with pytest.raises(BackupFailedError):
        await backup_before_reset({"graph_data"}, str(tmp_path), failing_run_subprocess)


class FakeExecuteDatabase:
    def __init__(self, chunk_tables, fk_violation_on_groups_containing=None):
        self._chunk_tables = chunk_tables
        self._fk_violation_on_groups_containing = fk_violation_on_groups_containing or set()
        self.reset_tables_calls = []

    async def list_chunk_tables(self):
        return self._chunk_tables

    async def reset_tables(self, chunk_tables, table_groups):
        self.reset_tables_calls.append((list(chunk_tables), [list(g) for g in table_groups]))
        for group in table_groups:
            if self._fk_violation_on_groups_containing & set(group):
                raise RuntimeError("foreign key violation (simulated)")


@pytest.mark.asyncio
async def test_execute_reset_drops_chunk_tables_only_when_ingestion_data_selected():
    db = FakeExecuteDatabase(chunk_tables=["chunks_dim_768"])

    await execute_reset(db, {"graph_data"})
    assert db.reset_tables_calls[-1][0] == []

    await execute_reset(db, {"ingestion_data", "graph_data"})
    assert db.reset_tables_calls[-1][0] == ["chunks_dim_768"]


@pytest.mark.asyncio
async def test_execute_reset_truncates_all_resolved_tables_in_one_statement():
    db = FakeExecuteDatabase(chunk_tables=[])

    await execute_reset(db, {"graph_data", "eval_history", "rca_sessions"})

    _, groups = db.reset_tables_calls[-1]
    assert len(groups) == 1
    assert set(groups[0]) == {
        "graph_entities", "graph_edges", "eval_cycles", "eval_runs", "rca_sessions", "rca_messages",
    }


@pytest.mark.asyncio
async def test_execute_reset_ingestion_data_includes_sweep_and_config_dependents():
    db = FakeExecuteDatabase(chunk_tables=["chunks_dim_768"])

    await execute_reset(db, {"ingestion_data", "graph_data", "eval_history", "rca_sessions"})

    chunk_tables, groups = db.reset_tables_calls[-1]
    assert chunk_tables == ["chunks_dim_768"]
    assert {"experiment_configs", "sweeps", "sweep_configs", "codegen_runs",
            "generated_artifacts", "ingestion_runs"} <= set(groups[0])


@pytest.mark.asyncio
async def test_execute_reset_raises_reset_blocked_error_on_fk_violation():
    # Simulates rca_sessions referencing eval_cycles: truncating eval_history
    # without also selecting rca_sessions must be refused, not silently
    # cascade-delete the RCA data (review finding C-2).
    db = FakeExecuteDatabase(chunk_tables=[], fk_violation_on_groups_containing={"eval_cycles"})

    with pytest.raises(ResetBlockedError):
        await execute_reset(db, {"eval_history"})
