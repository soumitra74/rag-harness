import logging

import pytest

from app.providers.base import LLMResponse
from app.services import pricing
from app.services.sweep_log import MeteredProvider, progress_log, stage


class FakeProvider:
    async def complete(self, prompt):
        return LLMResponse(text="ok", input_tokens=120, output_tokens=30)


class FakeUsageDb:
    def __init__(self):
        self.usage = []

    async def insert_llm_usage(self, sweep_id, stage_name, provider, model, input_tokens, output_tokens, cost):
        self.usage.append((sweep_id, stage_name, provider, model, input_tokens, output_tokens, cost))


def _meter(db, provider_name="openai", model="gpt-x", in_rate=0.0, out_rate=0.0):
    return MeteredProvider(FakeProvider(), db, provider_name, model, in_rate, out_rate)


@pytest.fixture
def priced(monkeypatch):
    monkeypatch.setattr(pricing, "MODEL_PRICES_PER_1M", {("openai", "gpt-x"): (2.0, 8.0)})


@pytest.mark.asyncio
async def test_stage_logs_start_and_end_with_llm_tokens_and_cost(caplog, priced):
    db = FakeUsageDb()
    provider = _meter(db)
    caplog.set_level(logging.INFO, logger=progress_log.name)

    with stage(7, "inspect_corpus"):
        await provider.complete("a")
        await provider.complete("b")

    messages = [r.getMessage() for r in caplog.records if r.name == progress_log.name]
    assert messages[0] == "sweep 7: START inspect_corpus"
    assert "llm_calls=2 input_tokens=240 output_tokens=60 total_tokens=300" in messages[1]
    assert "cost=$0.000960" in messages[1]
    assert [u[1] for u in db.usage] == ["inspect_corpus", "inspect_corpus"]
    assert db.usage[0][:6] == (7, "inspect_corpus", "openai", "gpt-x", 120, 30)


@pytest.mark.asyncio
async def test_unpriced_model_is_recorded_as_zero_cost_and_flagged(caplog):
    db = FakeUsageDb()
    provider = _meter(db, model="unlisted")
    caplog.set_level(logging.INFO, logger=progress_log.name)

    with stage(7, "summary"):
        await provider.complete("a")

    end = [r.getMessage() for r in caplog.records if "END summary" in r.getMessage()][0]
    assert "unpriced_calls=1" in end
    assert db.usage[0][6] == 0.0


@pytest.mark.asyncio
async def test_stage_logs_failure_and_reraises(caplog):
    caplog.set_level(logging.INFO, logger=progress_log.name)

    with pytest.raises(RuntimeError):
        with stage(7, "codegen", tasks="ocr"):
            raise RuntimeError("boom")

    failed = [r for r in caplog.records if "FAILED codegen" in r.getMessage()]
    assert len(failed) == 1 and failed[0].levelno == logging.ERROR
    assert "boom" in failed[0].getMessage()


@pytest.mark.asyncio
async def test_calls_outside_a_stage_are_not_recorded():
    db = FakeUsageDb()
    provider = _meter(db)

    await provider.complete("before any stage")

    assert db.usage == []
