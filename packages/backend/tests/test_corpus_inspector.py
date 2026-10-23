import json

import pytest

from app.providers.base import LLMResponse
from app.services.corpus_inspector import (
    CorpusConfigCandidate,
    CorpusInspectionError,
    inspect_corpus,
)

CANNED_CANDIDATES = [
    {
        "chunking_strategy": "sliding_window", "chunk_size": 1000, "chunk_overlap": 200,
        "embedding_provider": "local", "embedding_model": "BAAI/bge-base-en-v1.5",
        "rationale": "generic baseline",
    },
    {
        "chunking_strategy": "document_structure_aware", "chunk_size": 800, "chunk_overlap": 100,
        "embedding_provider": "local", "embedding_model": "BAAI/bge-base-en-v1.5",
        "rationale": "structured long docs",
    },
]


class FakeProvider:
    def __init__(self, text):
        self._text = text
        self.last_prompt = None

    async def complete(self, prompt: str) -> LLMResponse:
        self.last_prompt = prompt
        return LLMResponse(text=self._text, input_tokens=10, output_tokens=10)


@pytest.mark.asyncio
async def test_inspect_corpus_raises_on_empty_corpus(tmp_path):
    with pytest.raises(CorpusInspectionError):
        await inspect_corpus(FakeProvider(""), str(tmp_path))


@pytest.mark.asyncio
async def test_inspect_corpus_returns_candidates_from_provider(tmp_path):
    (tmp_path / "doc1.pdf").write_bytes(b"not a real pdf")
    provider = FakeProvider(json.dumps(CANNED_CANDIDATES))

    candidates = await inspect_corpus(provider, str(tmp_path))

    assert len(candidates) == 2
    assert all(isinstance(c, CorpusConfigCandidate) for c in candidates)
    assert candidates[0].chunking_strategy == "sliding_window"
    assert candidates[1].rationale == "structured long docs"


@pytest.mark.asyncio
async def test_inspect_corpus_drops_candidates_with_unknown_chunking_strategy(tmp_path):
    (tmp_path / "doc1.pdf").write_bytes(b"not a real pdf")
    bad = [dict(CANNED_CANDIDATES[0], chunking_strategy="not_a_real_strategy")]
    provider = FakeProvider(json.dumps(bad))

    candidates = await inspect_corpus(provider, str(tmp_path))

    assert candidates == []
