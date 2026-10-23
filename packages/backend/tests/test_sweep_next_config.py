import json

import pytest

from app.providers.base import LLMResponse
from app.services.sweep_next_config import TriedConfig, propose_next_config

CANDIDATE = {
    "chunking_strategy": "sliding_window", "chunk_size": 1200, "chunk_overlap": 200,
    "embedding_provider": "local", "embedding_model": "BAAI/bge-base-en-v1.5",
    "rationale": "larger chunks might help",
}


class FakeProvider:
    def __init__(self, text):
        self._text = text
        self.last_prompt = None

    async def complete(self, prompt: str) -> LLMResponse:
        self.last_prompt = prompt
        return LLMResponse(text=self._text, input_tokens=5, output_tokens=5)


TRIED = [TriedConfig(config={"chunking_strategy": "sliding_window", "chunk_size": 800}, score=0.7)]
BUDGET = {"cost_usd": 5.0, "minutes": 30.0, "iterations": 5}


@pytest.mark.asyncio
async def test_propose_next_config_returns_converged():
    provider = FakeProvider(json.dumps({"converged": True}))

    result = await propose_next_config(provider, "corpus summary", TRIED, BUDGET)

    assert result.converged is True
    assert result.candidate is None


@pytest.mark.asyncio
async def test_propose_next_config_returns_candidate():
    provider = FakeProvider(json.dumps({"converged": False, "candidate": CANDIDATE}))

    result = await propose_next_config(provider, "corpus summary", TRIED, BUDGET)

    assert result.converged is False
    assert result.candidate.chunk_size == 1200
    assert result.candidate.rationale == "larger chunks might help"


@pytest.mark.asyncio
async def test_propose_next_config_includes_user_hint_in_prompt():
    provider = FakeProvider(json.dumps({"converged": False, "candidate": CANDIDATE}))

    await propose_next_config(
        provider, "corpus summary", TRIED, BUDGET, user_hint="try a smaller chunk size",
    )

    assert "try a smaller chunk size" in provider.last_prompt
