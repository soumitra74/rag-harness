import json

import pytest

from app.providers.base import LLMResponse
from app.services.golden_qa_generator import GoldenQAGenerationError, generate_golden_qa

VALID_ENTRIES = [
    {
        "id": "q001", "question": "What is X?", "reference_answer": "X is Y.",
        "golden_pages": [{"document": "doc1.pdf", "page": 1}], "query_type": "extractive",
    },
]


class FakeProvider:
    def __init__(self, text):
        self._text = text

    async def complete(self, prompt: str) -> LLMResponse:
        return LLMResponse(text=self._text, input_tokens=10, output_tokens=10)


@pytest.mark.asyncio
async def test_generate_golden_qa_writes_valid_file(tmp_path):
    (tmp_path / "doc1.pdf").write_bytes(b"not a real pdf")
    output_path = tmp_path / "golden_qa.json"
    provider = FakeProvider(json.dumps(VALID_ENTRIES))

    count = await generate_golden_qa(provider, str(tmp_path), str(output_path))

    assert count == 1
    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert written == VALID_ENTRIES


@pytest.mark.asyncio
async def test_generate_golden_qa_raises_and_writes_nothing_on_malformed_output(tmp_path):
    (tmp_path / "doc1.pdf").write_bytes(b"not a real pdf")
    output_path = tmp_path / "golden_qa.json"
    missing_key_entries = [{"id": "q001", "question": "What is X?"}]
    provider = FakeProvider(json.dumps(missing_key_entries))

    with pytest.raises(GoldenQAGenerationError):
        await generate_golden_qa(provider, str(tmp_path), str(output_path))

    assert not output_path.exists()
