import pytest

from app.providers.base import LLMResponse
from app.services.sweep_next_config import TriedConfig
from app.services.sweep_summary import generate_sweep_summary


class FakeProvider:
    def __init__(self, text):
        self._text = text
        self.last_prompt = None

    async def complete(self, prompt: str) -> LLMResponse:
        self.last_prompt = prompt
        return LLMResponse(text=self._text, input_tokens=5, output_tokens=5)


@pytest.mark.asyncio
async def test_generate_sweep_summary_returns_provider_text_passthrough():
    provider = FakeProvider("## Recommendation\nUse config 3.")
    tried = [TriedConfig(config={"chunk_size": 800}, score=0.7)]
    best = tried[0]

    result = await generate_sweep_summary(
        provider, "corpus summary", tried, stop_reason="stopped_budget", best=best,
    )

    assert result == "## Recommendation\nUse config 3."


@pytest.mark.asyncio
async def test_generate_sweep_summary_prompt_includes_stop_reason_and_scores():
    provider = FakeProvider("summary text")
    tried = [
        TriedConfig(config={"chunk_size": 800}, score=0.7),
        TriedConfig(config={"chunk_size": 1200}, score=0.9),
    ]

    await generate_sweep_summary(
        provider, "corpus summary", tried, stop_reason="stopped_optimal", best=tried[1],
    )

    assert "stopped_optimal" in provider.last_prompt
    assert "0.7" in provider.last_prompt
    assert "0.9" in provider.last_prompt
