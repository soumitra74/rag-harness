import importlib.util
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.providers.base import LLMResponse
from app.services import codegen
from app.services.codegen import (
    GRAPH_LABELS_TASK,
    OCR_TASK,
    CodegenContext,
    CodegenFailed,
    apply_graph_labels,
    check_python_module,
    extract_block,
    run_codegen_task,
    validate_graph_labels,
)
from app.services.ner import ENTITY_LABELS
from tests.codegen_fakes import CodegenFakeMethods

VALID_OCR = "```python\ndef ocr_page(pdf_path, page_number):\n    return 'FRAUD REPORT TEXT'\n```"


class FakeDb(CodegenFakeMethods):
    pass


class ScriptedProvider:
    def __init__(self, replies):
        self._replies = list(replies)
        self.prompts = []

    async def complete(self, prompt):
        self.prompts.append(prompt)
        return LLMResponse(text=self._replies.pop(0), input_tokens=100, output_tokens=50)


class FakeEntity:
    def __init__(self, entity_type):
        self.entity_type = entity_type


class FakeNER:
    def __init__(self, labels_per_sample):
        self._per_sample = labels_per_sample

    async def extract(self, text, labels=None):
        return [FakeEntity(label) for label in self._per_sample.get(text, [])]


def _ctx(ner=None):
    return CodegenContext(
        settings=SimpleNamespace(
            corpus_dir="unused", judge_cost_per_1k_in_usd=0.0, judge_cost_per_1k_out_usd=0.0,
            codegen_debug=False, codegen_provider=None, codegen_model=None,
            generated_dir=str(Path(tempfile.gettempdir()) / "codegen-tests"),
            eval_judge_provider="openai", eval_judge_model="gpt-6-luna",
        ),
        ner=ner,
        sandbox_url="http://sandbox.test",
    )


def _budget(attempts=3):
    return {"cost_usd": 100.0, "minutes": 100.0, "attempts": attempts}


def test_extract_block_reads_fenced_python_and_json():
    assert extract_block(OCR_TASK, VALID_OCR).startswith("def ocr_page")
    assert extract_block(GRAPH_LABELS_TASK, "```json\n[\"A_B\"]\n```") == '["A_B"]\n'


def test_extract_block_rejects_missing_fence():
    with pytest.raises(ValueError):
        extract_block(OCR_TASK, "no code here")


def test_scanned_page_numbers_skips_truly_empty_pages():
    page_texts = [(1, ""), (2, "   "), (3, ""), (4, "has text")]
    image_counts = [1, 0, 0, 1]
    assert codegen.scanned_page_numbers(page_texts, image_counts) == [1]


def test_check_python_module_requires_entrypoint():
    with pytest.raises(ValueError, match="ocr_page"):
        check_python_module("def other():\n    pass\n", "ocr_page")


@pytest.mark.asyncio
async def test_ocr_task_passes_validation_and_stores_artifact(monkeypatch):
    monkeypatch.setattr(codegen, "blank_page_samples", lambda corpus_dir, n: [("a.pdf", b"%PDF", 3)])

    async def fake_sandbox(url, code, entrypoint, args, files=None, timeout_s=60, debug=False):
        assert entrypoint == "ocr_page" and args == ["input.pdf", 3]
        return "Fraud report text here"

    monkeypatch.setattr(codegen, "run_in_sandbox", fake_sandbox)
    db = FakeDb()
    provider = ScriptedProvider([VALID_OCR])

    content = await run_codegen_task(db, provider, _ctx(), 7, OCR_TASK, _budget())

    assert "def ocr_page" in content
    assert db.generated_artifacts["ocr"] == content
    assert len(db.codegen_runs) == 1 and db.codegen_runs[0]["passed"] is True


@pytest.mark.asyncio
async def test_failed_validation_is_fed_back_to_the_next_attempt(monkeypatch):
    monkeypatch.setattr(codegen, "blank_page_samples", lambda corpus_dir, n: [("a.pdf", b"%PDF", 1)])
    replies = iter(["", "Fraud report text here"])

    async def fake_sandbox(url, code, entrypoint, args, files=None, timeout_s=60, debug=False):
        return next(replies)

    monkeypatch.setattr(codegen, "run_in_sandbox", fake_sandbox)
    db = FakeDb()
    provider = ScriptedProvider([VALID_OCR, VALID_OCR])

    await run_codegen_task(db, provider, _ctx(), 7, OCR_TASK, _budget())

    assert [r["passed"] for r in db.codegen_runs] == [False, True]
    assert "produced no readable text" in provider.prompts[1]


@pytest.mark.asyncio
async def test_budget_exhaustion_raises_codegen_failed(monkeypatch):
    monkeypatch.setattr(codegen, "blank_page_samples", lambda corpus_dir, n: [("a.pdf", b"%PDF", 1)])

    async def always_empty(url, code, entrypoint, args, files=None, timeout_s=60, debug=False):
        return ""

    monkeypatch.setattr(codegen, "run_in_sandbox", always_empty)
    db = FakeDb()
    provider = ScriptedProvider([VALID_OCR] * 5)

    with pytest.raises(CodegenFailed, match="budget exhausted \\(attempts\\)"):
        await run_codegen_task(db, provider, _ctx(), 7, OCR_TASK, _budget(attempts=2))

    assert len(db.codegen_runs) == 2
    assert db.generated_artifacts == {}


@pytest.mark.asyncio
async def test_graph_labels_rejects_malformed_and_unmatched_labels(monkeypatch):
    monkeypatch.setattr(codegen, "text_samples", lambda corpus_dir, n: ["s1", "s2"])
    ner = FakeNER({"s1": ["WIRE_TRANSFER"], "s2": ["WIRE_TRANSFER"]})

    problems = await validate_graph_labels(_ctx(ner), ["wire transfer", "WIRE_TRANSFER"])
    assert any("match" in p for p in problems)

    problems = await validate_graph_labels(_ctx(ner), ["WIRE_TRANSFER", "ACCOUNT_TYPE", "MERCHANT_ID"])
    assert problems and "ACCOUNT_TYPE" in problems[0]


@pytest.mark.asyncio
async def test_graph_labels_accepts_labels_seen_in_enough_samples(monkeypatch):
    monkeypatch.setattr(codegen, "text_samples", lambda corpus_dir, n: ["s1", "s2"])
    ner = FakeNER({
        "s1": ["WIRE_TRANSFER", "MERCHANT_ID", "ACCOUNT_TYPE"],
        "s2": ["WIRE_TRANSFER", "MERCHANT_ID", "ACCOUNT_TYPE"],
    })

    assert await validate_graph_labels(_ctx(ner), ["WIRE_TRANSFER", "MERCHANT_ID", "ACCOUNT_TYPE"]) == []


def test_apply_graph_labels_sets_and_leaves_alone():
    ner = SimpleNamespace(labels=list(ENTITY_LABELS))
    apply_graph_labels(ner, None)
    assert ner.labels == ENTITY_LABELS
    apply_graph_labels(ner, '["WIRE_TRANSFER", "MERCHANT_ID", "ACCOUNT_TYPE"]')
    assert ner.labels == ["WIRE_TRANSFER", "MERCHANT_ID", "ACCOUNT_TYPE"]


def _load_executor():
    path = Path(__file__).resolve().parents[2] / "codegen-sandbox" / "executor.py"
    spec = importlib.util.spec_from_file_location("sandbox_executor", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_sandbox_executor_runs_generated_code_and_returns_result():
    client = TestClient(_load_executor().app)
    response = client.post("/run", json={
        "code": "def add(a, b):\n    return a + b\n",
        "entrypoint": "add",
        "args": [2, 3],
        "files": {},
        "timeout_s": 30,
    })
    assert response.json() == {"ok": True, "result": 5, "error": None}


def test_sandbox_executor_reports_generated_code_errors():
    client = TestClient(_load_executor().app)
    response = client.post("/run", json={
        "code": "def boom():\n    raise RuntimeError('nope')\n",
        "entrypoint": "boom",
        "args": [],
        "files": {},
        "timeout_s": 30,
    })
    body = response.json()
    assert body["ok"] is False and "nope" in body["error"]


def test_sandbox_executor_rejects_unsafe_file_names():
    client = TestClient(_load_executor().app)
    response = client.post("/run", json={
        "code": "def f():\n    return 1\n",
        "entrypoint": "f",
        "args": [],
        "files": {"../escape.pdf": "AAAA"},
        "timeout_s": 30,
    })
    assert response.json()["ok"] is False


@pytest.mark.asyncio
async def test_codegen_failure_stops_bootstrap_before_ingestion(monkeypatch):
    from app.services import sweep

    class BootstrapDb(FakeDb):
        def __init__(self):
            self.status_updates = []
            self.created_configs = []

        async def update_sweep_status(self, sweep_id, status, error_message=None):
            self.status_updates.append((status, error_message))

        async def set_corpus_snapshot(self, sweep_id, filenames):
            pass

        async def list_sweep_configs(self, sweep_id):
            return []

        async def get_or_create_experiment_config(self, **kwargs):
            self.created_configs.append(kwargs)

    async def fake_inspect(provider, corpus_dir):
        return [SimpleNamespace(rationale="r", chunking_strategy="sliding_window", chunk_size=500,
                                chunk_overlap=50, embedding_provider="openai",
                                embedding_model="text-embedding-3-small")]

    async def failing_codegen(*args, **kwargs):
        raise CodegenFailed("ocr: budget exhausted (attempts) after 2 attempt(s)")

    monkeypatch.setattr(sweep, "inspect_corpus", fake_inspect)
    monkeypatch.setattr(sweep, "run_codegen_tasks", failing_codegen)
    db = BootstrapDb()

    await sweep.run_bootstrap(
        db, object(), SimpleNamespace(labels=[]),
        SimpleNamespace(corpus_dir="unused", codegen_sandbox_url="http://sandbox.test"), 1, False,
        codegen_tasks=["ocr"], codegen_budget=_budget(),
    )

    assert db.status_updates == [("failed", "ocr: budget exhausted (attempts) after 2 attempt(s)")]
    assert db.created_configs == []
