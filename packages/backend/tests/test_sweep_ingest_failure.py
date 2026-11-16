import pytest

from app.config import Settings
import app.services.sweep as sweep
from tests.test_sweep_bootstrap import CANDIDATES, FakeDatabase, FakeProvider, _patch_collaborators


def _settings(tmp_path, file_count):
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    for i in range(file_count):
        (corpus_dir / f"doc{i:02d}.pdf").write_bytes(b"x")
    return Settings(corpus_dir=str(corpus_dir), eval_dataset_path=str(tmp_path / "missing.json"),
                    generated_dir=str(tmp_path / "generated"))


def _inspect(corpus_dir):
    return CANDIDATES


def _failing_ingest(failing_names):
    async def ingest(filename, pdf_bytes, db, embedding_provider, config, **kwargs):
        if filename in failing_names:
            raise RuntimeError(f"embedding model unavailable for {filename}")
        return type("R", (), {"document_id": "d", "chunk_count": 1, "page_count": 1})()
    return ingest


@pytest.mark.asyncio
async def test_bootstrap_fails_when_a_quarter_of_files_fail(tmp_path, monkeypatch, caplog):
    settings = _settings(tmp_path, file_count=4)
    _patch_collaborators(monkeypatch, _inspect)
    monkeypatch.setattr(sweep, "ingest_pdf", _failing_ingest({"doc00.pdf"}))
    db = FakeDatabase()

    with caplog.at_level("ERROR"):
        await sweep.run_bootstrap(db, FakeProvider(), object(), settings, sweep_id=1,
                                   generate_golden_qa_flag=False)

    _, status, error_message = db.status_updates[-1]
    assert status == "failed"
    assert "1 of 4 file(s) failed" in error_message
    assert "doc00.pdf" in error_message
    assert "doc00.pdf failed: RuntimeError: embedding model unavailable" in caplog.text


@pytest.mark.asyncio
async def test_bootstrap_tolerates_failures_below_threshold(tmp_path, monkeypatch):
    settings = _settings(tmp_path, file_count=10)
    _patch_collaborators(monkeypatch, _inspect)
    monkeypatch.setattr(sweep, "ingest_pdf", _failing_ingest({"doc00.pdf"}))
    db = FakeDatabase()

    await sweep.run_bootstrap(db, FakeProvider(), object(), settings, sweep_id=1,
                               generate_golden_qa_flag=False)

    assert db.status_updates == [(1, "awaiting_approval", None)]
    assert [row["status"] for row in db.sweep_configs] == ["ingested"]
