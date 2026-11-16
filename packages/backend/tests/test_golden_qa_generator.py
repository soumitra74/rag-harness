import json

import pytest

from app.providers.base import LLMResponse
from app.services import golden_qa_generator
from app.services.golden_qa_generator import GoldenQAGenerationError, generate_golden_qa

LONG_TEXT = "Synthetic data supports transaction fraud detection research. " * 10


VERIFY_MARKER = '"supported"'
CLOSED_BOOK_MARKER = '"answerable_without_document"'


def _next(queue: list[str]) -> str:
    return queue.pop(0) if len(queue) > 1 else queue[0]


class FakeProvider:
    def __init__(self, responses, verdicts=None, closed_book=None):
        self._responses = list(responses)
        self._verdicts = list(verdicts) if verdicts is not None else [_verdict(True)]
        self._closed_book = list(closed_book) if closed_book is not None else [_closed_book(False)]
        self.prompts = []

    async def complete(self, prompt: str) -> LLMResponse:
        self.prompts.append(prompt)
        if VERIFY_MARKER in prompt:
            queue = self._verdicts
        elif CLOSED_BOOK_MARKER in prompt:
            queue = self._closed_book
        else:
            queue = self._responses
        return LLMResponse(text=_next(queue), input_tokens=10, output_tokens=10)


def _answer(question="What is X?", answer="X is Y.", query_type="extractive", **extra):
    return json.dumps({"question": question, "reference_answer": answer,
                       "query_type": query_type, **extra})


def _verdict(supported, quote="Synthetic data supports transaction fraud detection research."):
    return json.dumps({"supported": supported, "quote": quote})


def _closed_book(answerable):
    return json.dumps({"answerable_without_document": answerable})


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
    provider = FakeProvider([_answer(question="Q1?"), _answer(question="Q2?")])

    count = await generate_golden_qa(provider, str(tmp_path), str(output_path), num_questions=2)

    assert count == 2
    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert sorted(e["golden_pages"][0]["page"] for e in written) == [1, 2]
    assert all(e["golden_pages"][0]["document"] == "doc1.pdf" for e in written)
    assert sorted(e["question"] for e in written) == ["Q1?", "Q2?"]
    assert all(e["query_type"] == "extractive" for e in written)


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
async def test_generate_golden_qa_fills_from_next_page_when_one_is_rejected(tmp_path, monkeypatch):
    _corpus(tmp_path, ["doc1.pdf"])
    _patch_pages(monkeypatch, {"doc1.pdf": [(n, LONG_TEXT) for n in range(1, 4)]})
    output_path = tmp_path / "golden_qa.json"
    provider = FakeProvider(["not json", _answer(question="Q1?"), _answer(question="Q2?")])

    count = await generate_golden_qa(provider, str(tmp_path), str(output_path), num_questions=2)

    assert count == 2
    generation_prompts = [p for p in provider.prompts
                          if VERIFY_MARKER not in p and CLOSED_BOOK_MARKER not in p]
    assert len(generation_prompts) == 3


@pytest.mark.asyncio
async def test_generate_golden_qa_rejects_questions_with_broken_encoding(tmp_path, monkeypatch):
    _corpus(tmp_path, ["doc1.pdf"])
    _patch_pages(monkeypatch, {"doc1.pdf": [(n, LONG_TEXT) for n in range(1, 3)]})
    output_path = tmp_path / "golden_qa.json"
    provider = FakeProvider([_answer(question="What is TransXion�s score?"), _answer(question="Q2?")])

    count = await generate_golden_qa(provider, str(tmp_path), str(output_path), num_questions=1)

    assert count == 1
    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert written[0]["question"] == "Q2?"


@pytest.mark.asyncio
async def test_generate_golden_qa_stops_generating_once_size_is_reached(tmp_path, monkeypatch):
    _corpus(tmp_path, ["doc1.pdf"])
    _patch_pages(monkeypatch, {"doc1.pdf": [(n, LONG_TEXT) for n in range(1, 6)]})
    output_path = tmp_path / "golden_qa.json"
    provider = FakeProvider([_answer(question="Q1?"), _answer(question="Q2?")])

    count = await generate_golden_qa(provider, str(tmp_path), str(output_path), num_questions=1)

    assert count == 1
    generation_prompts = [p for p in provider.prompts
                          if VERIFY_MARKER not in p and CLOSED_BOOK_MARKER not in p]
    assert len(generation_prompts) == 1


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
    provider = FakeProvider([_answer(question="Q1?"), _answer(question="Q2?")])

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
    assert len(provider.prompts) == 3
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


@pytest.mark.asyncio
async def test_verification_drops_questions_the_page_does_not_support(tmp_path, monkeypatch):
    _corpus(tmp_path, ["doc1.pdf"])
    _patch_pages(monkeypatch, {"doc1.pdf": [(1, LONG_TEXT), (2, LONG_TEXT)]})
    output_path = tmp_path / "golden_qa.json"
    provider = FakeProvider([_answer(question="First?"), _answer(question="Second?")],
                            verdicts=[_verdict(False), _verdict(True)])

    count = await generate_golden_qa(provider, str(tmp_path), str(output_path), num_questions=2)

    assert count == 1
    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert [e["question"] for e in written] == ["Second?"]


@pytest.mark.asyncio
async def test_verification_requires_the_quote_to_appear_on_the_page(tmp_path, monkeypatch):
    _corpus(tmp_path, ["doc1.pdf"])
    _patch_pages(monkeypatch, {"doc1.pdf": [(1, LONG_TEXT)]})
    output_path = tmp_path / "golden_qa.json"
    provider = FakeProvider([_answer()], verdicts=[_verdict(True, quote="A sentence the page never says.")])

    with pytest.raises(GoldenQAGenerationError, match="0 of 1 candidate"):
        await generate_golden_qa(provider, str(tmp_path), str(output_path), num_questions=1)

    assert not output_path.exists()


@pytest.mark.asyncio
async def test_verification_quote_match_ignores_case_and_whitespace(tmp_path, monkeypatch):
    _corpus(tmp_path, ["doc1.pdf"])
    _patch_pages(monkeypatch, {"doc1.pdf": [(1, LONG_TEXT)]})
    output_path = tmp_path / "golden_qa.json"
    provider = FakeProvider([_answer()],
                            verdicts=[_verdict(True, quote="SYNTHETIC   data\nsupports transaction fraud detection research.")])

    count = await generate_golden_qa(provider, str(tmp_path), str(output_path), num_questions=1)

    assert count == 1


@pytest.mark.asyncio
async def test_verification_prompt_carries_the_generated_question_and_answer(tmp_path, monkeypatch):
    _corpus(tmp_path, ["doc1.pdf"])
    _patch_pages(monkeypatch, {"doc1.pdf": [(1, LONG_TEXT)]})
    output_path = tmp_path / "golden_qa.json"
    provider = FakeProvider([_answer(question="What is Z?", answer="Z is W.")])

    await generate_golden_qa(provider, str(tmp_path), str(output_path), num_questions=1)

    verify_prompts = [p for p in provider.prompts if VERIFY_MARKER in p]
    assert len(verify_prompts) == 1
    assert "What is Z?" in verify_prompts[0] and "Z is W." in verify_prompts[0]


@pytest.mark.asyncio
async def test_verification_runs_on_the_verifier_not_the_generator(tmp_path, monkeypatch):
    _corpus(tmp_path, ["doc1.pdf"])
    _patch_pages(monkeypatch, {"doc1.pdf": [(1, LONG_TEXT)]})
    output_path = tmp_path / "golden_qa.json"
    generator = FakeProvider([_answer()])
    verifier = FakeProvider([], verdicts=[_verdict(True)])

    count = await generate_golden_qa(generator, str(tmp_path), str(output_path),
                                     num_questions=1, verifier=verifier)

    assert count == 1
    assert len(generator.prompts) == 1 and VERIFY_MARKER not in generator.prompts[0]
    assert len(verifier.prompts) == 2 and VERIFY_MARKER in verifier.prompts[0]
    assert CLOSED_BOOK_MARKER in verifier.prompts[1]


@pytest.mark.asyncio
async def test_closed_book_answerable_questions_are_rejected(tmp_path, monkeypatch):
    _corpus(tmp_path, ["doc1.pdf"])
    _patch_pages(monkeypatch, {"doc1.pdf": [(1, LONG_TEXT)]})
    output_path = tmp_path / "golden_qa.json"
    provider = FakeProvider([_answer()], closed_book=[_closed_book(True)])

    with pytest.raises(GoldenQAGenerationError, match="0 of 1 candidate"):
        await generate_golden_qa(provider, str(tmp_path), str(output_path), num_questions=1)

    assert not output_path.exists()


@pytest.mark.asyncio
async def test_closed_book_check_drops_unparseable_answers(tmp_path, monkeypatch):
    _corpus(tmp_path, ["doc1.pdf"])
    _patch_pages(monkeypatch, {"doc1.pdf": [(1, LONG_TEXT)]})
    output_path = tmp_path / "golden_qa.json"
    provider = FakeProvider([_answer()], closed_book=["I am not sure."])

    with pytest.raises(GoldenQAGenerationError):
        await generate_golden_qa(provider, str(tmp_path), str(output_path), num_questions=1)


@pytest.mark.asyncio
async def test_repeated_questions_are_generated_and_verified_once(tmp_path, monkeypatch):
    _corpus(tmp_path, ["doc1.pdf"])
    _patch_pages(monkeypatch, {"doc1.pdf": [(1, LONG_TEXT), (2, LONG_TEXT)]})
    output_path = tmp_path / "golden_qa.json"
    provider = FakeProvider([_answer(question="Same question?"), _answer(question="Same question??")])

    count = await generate_golden_qa(provider, str(tmp_path), str(output_path), num_questions=2)

    assert count == 1
    verify_prompts = [p for p in provider.prompts if VERIFY_MARKER in p]
    assert len(verify_prompts) == 1


def _entry(entry_id="q1", question="What is X?", pages=None):
    return {"id": entry_id, "question": question, "reference_answer": "X is Y.", "query_type": "extractive",
            "golden_pages": pages if pages is not None else [{"document": "doc1.pdf", "page": 1}]}


@pytest.mark.parametrize("entries, expected", [
    ([_entry()], []),
    ([_entry(pages=[{"document": "missing.pdf", "page": 1}])], ["unknown document 'missing.pdf'"]),
    ([_entry(pages=[{"document": "doc1.pdf", "page": 9}])], ["page 9 out of range"]),
    ([_entry(pages=[{"document": "doc1.pdf", "page": 2}])], ["page 2 of 'doc1.pdf' has no usable text"]),
    ([_entry(pages=[])], ["no golden pages"]),
    ([_entry("q1"), _entry("q2", question="what is x")], ["duplicate question"]),
])
def test_check_golden_entries_reports_structural_problems(entries, expected):
    documents = {"doc1.pdf": [(1, LONG_TEXT), (2, "short")]}

    problems = golden_qa_generator.check_golden_entries(entries, documents)

    assert len(problems) == len(expected)
    for needle, problem in zip(expected, problems):
        assert needle in problem


def _split_entries(documents: int, per_document: int) -> list[dict]:
    return [
        {"id": f"q{d:02d}{i:02d}", "question": f"Q {d} {i}?",
         "golden_pages": [{"document": f"doc{d:02d}.pdf", "page": 1}]}
        for d in range(documents) for i in range(per_document)
    ]


def test_split_golden_dataset_is_disjoint_sized_and_spans_documents():
    entries = _split_entries(documents=10, per_document=5)

    dev, holdout = golden_qa_generator.split_golden_dataset(entries, dev_size=20)

    assert len(dev) == 20 and len(holdout) == 30
    assert {e["id"] for e in dev}.isdisjoint({e["id"] for e in holdout})
    assert len({e["golden_pages"][0]["document"] for e in dev}) == 10


def test_split_golden_dataset_is_deterministic():
    entries = _split_entries(documents=10, per_document=5)

    first = golden_qa_generator.split_golden_dataset(entries)
    second = golden_qa_generator.split_golden_dataset(list(reversed(entries)))

    assert {e["id"] for e in first[0]} == {e["id"] for e in second[0]}
