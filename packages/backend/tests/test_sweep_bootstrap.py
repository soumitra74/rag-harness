import json

import pytest

from app.config import Settings
from app.db import ExperimentConfig
from app.providers.base import LLMResponse
from app.services.corpus_inspector import CorpusConfigCandidate, CorpusInspectionError
from app.services.golden_qa_generator import GoldenQAGenerationError
import app.services.sweep as sweep
from tests.codegen_fakes import CodegenFakeMethods

CANDIDATES = [
    CorpusConfigCandidate(
        chunking_strategy="sliding_window", chunk_size=1000, chunk_overlap=200,
        embedding_provider="local", embedding_model="BAAI/bge-base-en-v1.5",
        rationale="baseline",
    ),
]


class FakeProvider:
    async def complete(self, prompt: str) -> LLMResponse:
        return LLMResponse(text="{}", input_tokens=1, output_tokens=1)


class FakeDatabase(CodegenFakeMethods):
    def __init__(self):
        self.configs = {}
        self.sweep_configs = []
        self.status_updates = []
        self._next_config_id = 1

    async def get_or_create_experiment_config(self, chunking_strategy, chunk_size, chunk_overlap,
                                               embedding_provider, embedding_model, embedding_dim):
        key = (chunking_strategy, chunk_size, chunk_overlap, embedding_provider, embedding_model,
               embedding_dim)
        for config in self.configs.values():
            if (config.chunking_strategy, config.chunk_size, config.chunk_overlap,
                    config.embedding_provider, config.embedding_model,
                    config.embedding_dim) == key:
                return config
        config = ExperimentConfig(
            id=self._next_config_id, chunking_strategy=chunking_strategy, chunk_size=chunk_size,
            chunk_overlap=chunk_overlap, embedding_provider=embedding_provider,
            embedding_model=embedding_model, embedding_dim=embedding_dim,
        )
        self._next_config_id += 1
        self.configs[config.id] = config
        return config

    async def get_experiment_config(self, config_id):
        return self.configs[config_id]

    async def create_sweep_config(self, sweep_id, config_id, rationale, status, source):
        row = {"id": len(self.sweep_configs) + 1, "sweep_id": sweep_id, "config_id": config_id,
               "rationale": rationale, "status": status, "source": source}
        self.sweep_configs.append(row)
        return row["id"]

    async def list_sweep_configs(self, sweep_id):
        return [dict(row) for row in self.sweep_configs if row["sweep_id"] == sweep_id]

    async def update_sweep_config_status(self, sweep_config_id, status):
        for row in self.sweep_configs:
            if row["id"] == sweep_config_id:
                row["status"] = status

    async def finish_bootstrap(self, sweep_id):
        self.status_updates.append((sweep_id, "awaiting_approval", None))

    async def update_sweep_status(self, sweep_id, status, error_message=None):
        self.status_updates.append((sweep_id, status, error_message))

    async def set_corpus_snapshot(self, sweep_id, filenames):
        self.corpus_snapshot = filenames


class _FakeIngestResult:
    document_id = "doc-1"
    chunk_count = 0
    page_count = 0


async def _fake_ingest_pdf(filename, pdf_bytes, db, embedding_provider, config, ocr_page=None):
    return _FakeIngestResult()


async def _fake_backfill(db, ner, config):
    return {"chunks_processed": 0, "entities_touched": 0, "mentions_created": 0,
            "cooccurrences_created": 0}


async def _fake_record_ingestion_metrics(db, config, results, input_size_bytes, start, settings, sweep_id):
    return {}


def _patch_collaborators(monkeypatch, inspect_corpus_fn):
    monkeypatch.setattr(sweep, "inspect_corpus", inspect_corpus_fn)
    monkeypatch.setattr(sweep, "ingest_pdf", _fake_ingest_pdf)
    monkeypatch.setattr(sweep, "get_embedding_provider_for_config", lambda settings, config: object())
    monkeypatch.setattr(sweep, "backfill_graph_for_config", _fake_backfill)
    monkeypatch.setattr(sweep, "record_ingestion_metrics", _fake_record_ingestion_metrics)


@pytest.mark.asyncio
async def test_run_bootstrap_happy_path_creates_sweep_configs_and_approves(tmp_path, monkeypatch):
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    (corpus_dir / "doc1.pdf").write_bytes(b"not a real pdf")
    dataset_path = tmp_path / "golden_qa.json"
    monkeypatch.setattr(sweep, "load_corpus_pages", lambda corpus_dir: {"doc1.pdf": [(1, "x " * 150)]})
    dataset_path.write_text(json.dumps([
        {"id": "q1", "question": "q", "reference_answer": "a",
         "golden_pages": [{"document": "doc1.pdf", "page": 1}], "query_type": "extractive"}
    ]))
    settings = Settings(corpus_dir=str(corpus_dir), eval_dataset_path=str(dataset_path),
                        generated_dir=str(tmp_path / "generated"))

    async def fake_inspect_corpus(provider, corpus_dir_arg):
        return CANDIDATES

    _patch_collaborators(monkeypatch, fake_inspect_corpus)
    db = FakeDatabase()

    await sweep.run_bootstrap(db, FakeProvider(), object(), settings, sweep_id=1,
                               generate_golden_qa_flag=False)

    assert len(db.sweep_configs) == 1
    assert db.sweep_configs[0]["status"] == "ingested"
    assert db.sweep_configs[0]["source"] == "bootstrap"
    assert db.status_updates == [(1, "awaiting_approval", None)]


@pytest.mark.asyncio
async def test_run_bootstrap_fails_on_corpus_inspection_error_with_no_partial_rows(
    tmp_path, monkeypatch,
):
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    settings = Settings(corpus_dir=str(corpus_dir), eval_dataset_path=str(tmp_path / "missing.json"),
                        generated_dir=str(tmp_path / "generated"))

    async def failing_inspect_corpus(provider, corpus_dir_arg):
        raise CorpusInspectionError("no files")

    _patch_collaborators(monkeypatch, failing_inspect_corpus)
    db = FakeDatabase()

    await sweep.run_bootstrap(db, FakeProvider(), object(), settings, sweep_id=1,
                               generate_golden_qa_flag=False)

    assert db.sweep_configs == []
    assert len(db.status_updates) == 1
    sweep_id, status, error_message = db.status_updates[0]
    assert status == "failed"
    assert "no files" in error_message


@pytest.mark.asyncio
async def test_run_bootstrap_fails_on_malformed_existing_golden_qa_file(tmp_path, monkeypatch):
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    (corpus_dir / "doc1.pdf").write_bytes(b"not a real pdf")
    dataset_path = tmp_path / "golden_qa.json"
    dataset_path.write_text(json.dumps([{"id": "q1", "question": "q"}]))  # missing required keys
    settings = Settings(corpus_dir=str(corpus_dir), eval_dataset_path=str(dataset_path),
                        generated_dir=str(tmp_path / "generated"))

    async def fake_inspect_corpus(provider, corpus_dir_arg):
        return CANDIDATES

    _patch_collaborators(monkeypatch, fake_inspect_corpus)
    db = FakeDatabase()

    await sweep.run_bootstrap(db, FakeProvider(), object(), settings, sweep_id=1,
                               generate_golden_qa_flag=False)

    assert db.sweep_configs == []
    sweep_id, status, error_message = db.status_updates[0]
    assert status == "failed"
    assert "golden" in error_message.lower() or "required" in error_message.lower()


TWO_CANDIDATES = CANDIDATES + [
    CorpusConfigCandidate(
        chunking_strategy="sliding_window", chunk_size=500, chunk_overlap=50,
        embedding_provider="local", embedding_model="BAAI/bge-base-en-v1.5",
        rationale="smaller chunks",
    ),
]


@pytest.mark.asyncio
async def test_run_bootstrap_restart_ingests_only_unfinished_candidates(tmp_path, monkeypatch):
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    (corpus_dir / "doc1.pdf").write_bytes(b"not a real pdf")
    settings = Settings(corpus_dir=str(corpus_dir), eval_dataset_path=str(tmp_path / "missing.json"),
                        generated_dir=str(tmp_path / "generated"))
    backfilled = []

    async def first_run_inspect(provider, corpus_dir_arg):
        return TWO_CANDIDATES

    async def fail_on_second_backfill(db, ner, config):
        backfilled.append(config.id)
        if len(backfilled) == 2:
            raise RuntimeError("ner crashed")
        return {}

    _patch_collaborators(monkeypatch, first_run_inspect)
    monkeypatch.setattr(sweep, "backfill_graph_for_config", fail_on_second_backfill)
    db = FakeDatabase()

    with pytest.raises(RuntimeError, match="ner crashed"):
        await sweep.run_bootstrap(db, FakeProvider(), object(), settings, sweep_id=1,
                                   generate_golden_qa_flag=False)

    assert [row["status"] for row in db.sweep_configs] == ["ingested", "proposed"]

    backfilled.clear()

    async def backfill_ok(db_arg, ner, config):
        backfilled.append(config.id)
        return {}

    async def inspect_must_not_rerun(provider, corpus_dir_arg):
        raise AssertionError("restart must reuse saved candidates, not re-inspect")

    monkeypatch.setattr(sweep, "backfill_graph_for_config", backfill_ok)
    monkeypatch.setattr(sweep, "inspect_corpus", inspect_must_not_rerun)

    await sweep.run_bootstrap(db, FakeProvider(), object(), settings, sweep_id=1,
                               generate_golden_qa_flag=False)

    assert backfilled == [db.sweep_configs[1]["config_id"]]
    assert [row["status"] for row in db.sweep_configs] == ["ingested", "ingested"]
    assert db.status_updates[-1] == (1, "awaiting_approval", None)


@pytest.mark.asyncio
async def test_run_bootstrap_checks_empty_corpus_before_generating_golden_qa(
    tmp_path, monkeypatch,
):
    """Review finding I-2: golden-QA generation used to run before the
    empty-corpus check, so an empty corpus with the generation toggle on
    would ask the LLM to hallucinate a golden QA set from nothing, and
    persist it, before the clear empty-corpus error ever fired."""
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()  # no PDFs
    settings = Settings(corpus_dir=str(corpus_dir), eval_dataset_path=str(tmp_path / "missing.json"),
                        generated_dir=str(tmp_path / "generated"))

    generate_golden_qa_calls = []

    async def tracking_generate_golden_qa(provider, corpus_dir_arg, output_path, num_questions=20, verifier=None):
        generate_golden_qa_calls.append(output_path)
        return 0

    async def failing_inspect_corpus(provider, corpus_dir_arg):
        raise CorpusInspectionError("no files")

    _patch_collaborators(monkeypatch, failing_inspect_corpus)
    monkeypatch.setattr(sweep, "generate_golden_qa", tracking_generate_golden_qa)
    db = FakeDatabase()

    await sweep.run_bootstrap(db, FakeProvider(), object(), settings, sweep_id=1,
                               generate_golden_qa_flag=True)

    assert generate_golden_qa_calls == []
    sweep_id, status, error_message = db.status_updates[0]
    assert status == "failed"
    assert "no files" in error_message


@pytest.mark.asyncio
async def test_run_bootstrap_generates_golden_qa_with_dedicated_provider(tmp_path, monkeypatch):
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    (corpus_dir / "doc1.pdf").write_bytes(b"not a real pdf")
    settings = Settings(corpus_dir=str(corpus_dir), eval_dataset_path=str(tmp_path / "missing.json"),
                        generated_dir=str(tmp_path / "generated"))

    async def fake_inspect_corpus(provider, corpus_dir_arg):
        return CANDIDATES

    generator_providers = []

    async def recording_generate_golden_qa(provider, corpus_dir_arg, output_path, num_questions=20, verifier=None):
        generator_providers.append(provider)
        return 0

    _patch_collaborators(monkeypatch, fake_inspect_corpus)
    monkeypatch.setattr(sweep, "generate_golden_qa", recording_generate_golden_qa)
    golden_provider = FakeProvider()
    judge_provider = FakeProvider()

    await sweep.run_bootstrap(FakeDatabase(), judge_provider, object(), settings, sweep_id=1,
                               generate_golden_qa_flag=True, golden_qa_provider=golden_provider)

    assert generator_providers == [golden_provider]


@pytest.mark.asyncio
async def test_run_bootstrap_falls_back_to_judge_provider_for_golden_qa(tmp_path, monkeypatch):
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    (corpus_dir / "doc1.pdf").write_bytes(b"not a real pdf")
    settings = Settings(corpus_dir=str(corpus_dir), eval_dataset_path=str(tmp_path / "missing.json"),
                        generated_dir=str(tmp_path / "generated"))

    async def fake_inspect_corpus(provider, corpus_dir_arg):
        return CANDIDATES

    generator_providers = []

    async def recording_generate_golden_qa(provider, corpus_dir_arg, output_path, num_questions=20, verifier=None):
        generator_providers.append(provider)
        return 0

    _patch_collaborators(monkeypatch, fake_inspect_corpus)
    monkeypatch.setattr(sweep, "generate_golden_qa", recording_generate_golden_qa)
    judge_provider = FakeProvider()

    await sweep.run_bootstrap(FakeDatabase(), judge_provider, object(), settings, sweep_id=1,
                               generate_golden_qa_flag=True)

    assert generator_providers == [judge_provider]


@pytest.mark.asyncio
async def test_run_bootstrap_fails_on_existing_golden_qa_with_out_of_range_page(tmp_path, monkeypatch):
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    (corpus_dir / "doc1.pdf").write_bytes(b"not a real pdf")
    dataset_path = tmp_path / "golden_qa.json"
    dataset_path.write_text(json.dumps([
        {"id": "q1", "question": "q", "reference_answer": "a",
         "golden_pages": [{"document": "doc1.pdf", "page": 9}], "query_type": "extractive"}
    ]))
    monkeypatch.setattr(sweep, "load_corpus_pages", lambda corpus_dir: {"doc1.pdf": [(1, "x " * 150)]})
    settings = Settings(corpus_dir=str(corpus_dir), eval_dataset_path=str(dataset_path),
                        generated_dir=str(tmp_path / "generated"))

    async def fake_inspect_corpus(provider, corpus_dir_arg):
        return CANDIDATES

    _patch_collaborators(monkeypatch, fake_inspect_corpus)
    db = FakeDatabase()

    await sweep.run_bootstrap(db, FakeProvider(), object(), settings, sweep_id=1,
                               generate_golden_qa_flag=False)

    assert db.sweep_configs == []
    _, status, error_message = db.status_updates[0]
    assert status == "failed"
    assert "out of range" in error_message
