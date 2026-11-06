import json

import pytest

from app.providers.base import LLMResponse
from app.services import golden_qa_generator
from app.services.golden_qa_generator import GoldenQAGenerationError, generate_golden_qa

LONG_TEXT = "Synthetic data supports fraud detection research. " * 10


class FakeProvider:
    def __init__(self, responses):
        self._responses = list(responses)
        self.prompts = []

    async def complete(self, prompt: str) -> LLMResponse:
        self.prompts.append(prompt)
        text = self._responses.pop(0) if len(self._responses) > 1 else self._responses[0]
        return LLMResponse(text=text, input_tokens=10, output_tokens=10)


def _answer(question="What is X?", answer="X is Y.", query_type="extractive", **extra):
    return json.dumps({"question": question, "reference_answer": answer,
                       "query_type": query_type, **extra})


def _patch_pages(monkeypatch, pages_by_file):
    monkeypatch.setattr(golden_qa_generator, "extract_pages",
                        lambda pdf_bytes: pages_by_file[pdf_bytes.decode()])


def _corpus(tmp_path, files):
    for name in files:
        (tmp_path / name).write_bytes(name.encode())


@pytest.mark.asyncio
async def test_generate_golden_qa_writes_one_entry_per_sampled_page(tmp_path, monkeypatch):
    _corpus(tmp_path, ["doc1.pdf"])
    _patch_pages(monkeypatch, {"doc1.pdf": [(1, LONG_TEXT), (2, LONG_TEXT)]})
    output_path = tmp_path / "golden_qa.json"
    provider = FakeProvider([_answer(question="Q?")])

    count = await generate_golden_qa(provider, str(tmp_path), str(output_path), num_questions=2)

    assert count == 2
    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert sorted(e["golden_pages"][0]["page"] for e in written) == [1, 2]
    assert all(e["golden_pages"][0]["document"] == "doc1.pdf" for e in written)
    assert all(e["question"] == "Q?" and e["query_type"] == "extractive" for e in written)


@pytest.mark.asyncio
async def test_generate_golden_qa_ignores_model_supplied_golden_pages(tmp_path, monkeypatch):
    _corpus(tmp_path, ["doc1.pdf"])
    _patch_pages(monkeypatch, {"doc1.pdf": [(1, LONG_TEXT)]})
    output_path = tmp_path / "golden_qa.json"
    provider = FakeProvider([_answer(golden_pages=[{"document": "other.pdf", "page": 99}])])

    await generate_golden_qa(provider, str(tmp_path), str(output_path), num_questions=1)

    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert written[0]["golden_pages"] == [{"document": "doc1.pdf", "page": 1}]


@pytest.mark.asyncio
async def test_generate_golden_qa_skips_malformed_responses(tmp_path, monkeypatch):
    _corpus(tmp_path, ["doc1.pdf"])
    _patch_pages(monkeypatch, {"doc1.pdf": [(1, LONG_TEXT), (2, LONG_TEXT)]})
    output_path = tmp_path / "golden_qa.json"
    provider = FakeProvider(["not json", _answer(question="Good?")])

    count = await generate_golden_qa(provider, str(tmp_path), str(output_path), num_questions=2)

    assert count == 1
    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert written[0]["question"] == "Good?"


@pytest.mark.asyncio
async def test_generate_golden_qa_raises_and_writes_nothing_when_no_response_is_valid(tmp_path, monkeypatch):
    _corpus(tmp_path, ["doc1.pdf"])
    _patch_pages(monkeypatch, {"doc1.pdf": [(1, LONG_TEXT)]})
    output_path = tmp_path / "golden_qa.json"
    provider = FakeProvider([json.dumps({"question": "No answer key"})])

    with pytest.raises(GoldenQAGenerationError):
        await generate_golden_qa(provider, str(tmp_path), str(output_path), num_questions=1)

    assert not output_path.exists()


@pytest.mark.asyncio
async def test_sample_spreads_across_pages_instead_of_title_page(tmp_path, monkeypatch):
    _corpus(tmp_path, ["doc1.pdf"])
    _patch_pages(monkeypatch, {"doc1.pdf": [(n, LONG_TEXT) for n in range(1, 11)]})
    output_path = tmp_path / "golden_qa.json"
    provider = FakeProvider([_answer()])

    await generate_golden_qa(provider, str(tmp_path), str(output_path), num_questions=2)

    pages = [e["golden_pages"][0]["page"] for e in json.loads(output_path.read_text(encoding="utf-8"))]
    assert len(set(pages)) == 2 and 1 not in pages


@pytest.mark.asyncio
async def test_sample_skips_pages_without_enough_text(tmp_path, monkeypatch):
    _corpus(tmp_path, ["doc1.pdf"])
    _patch_pages(monkeypatch, {"doc1.pdf": [(1, "short"), (2, LONG_TEXT)]})
    output_path = tmp_path / "golden_qa.json"
    provider = FakeProvider([_answer()])

    await generate_golden_qa(provider, str(tmp_path), str(output_path), num_questions=5)

    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert [e["golden_pages"][0]["page"] for e in written] == [2]
    assert len(provider.prompts) == 1
    assert "Document: doc1.pdf, page 2" in provider.prompts[0]


@pytest.mark.asyncio
async def test_generate_golden_qa_rejects_questions_naming_authors(tmp_path, monkeypatch):
    _corpus(tmp_path, ["Jensen et al. - 2023 - A Synthetic Data Set.pdf"])
    _patch_pages(monkeypatch, {"Jensen et al. - 2023 - A Synthetic Data Set.pdf": [(1, LONG_TEXT), (2, LONG_TEXT)]})
    output_path = tmp_path / "golden_qa.json"
    provider = FakeProvider([_answer(question="What did Jensen et al. propose?"),
                             _answer(question="What is the stated purpose of the dataset?")])

    count = await generate_golden_qa(provider, str(tmp_path), str(output_path), num_questions=2)

    assert count == 1
    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert written[0]["question"] == "What is the stated purpose of the dataset?"
