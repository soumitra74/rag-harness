import logging

import pytest

from app.providers.base import LLMResponse
from app.services.sweep_log import MeteredProvider, progress_log, stage


class FakeProvider:
    async def complete(self, prompt):
        return LLMResponse(text="ok", input_tokens=120, output_tokens=30)


@pytest.mark.asyncio
async def test_stage_logs_start_and_end_with_llm_tokens(caplog):
    provider = MeteredProvider(FakeProvider())
    caplog.set_level(logging.INFO, logger=progress_log.name)

    with stage(7, "inspect_corpus"):
        await provider.complete("a")
        await provider.complete("b")

    messages = [r.getMessage() for r in caplog.records if r.name == progress_log.name]
    assert messages[0] == "sweep 7: START inspect_corpus"
    assert messages[1].startswith("sweep 7: END inspect_corpus in ")
    assert "llm_calls=2 input_tokens=240 output_tokens=60 total_tokens=300" in messages[1]


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
async def test_tokens_outside_a_stage_are_not_counted_into_later_stages(caplog):
    provider = MeteredProvider(FakeProvider())
    caplog.set_level(logging.INFO, logger=progress_log.name)

    await provider.complete("before any stage")
    with stage(7, "summary"):
        pass

    end = [r.getMessage() for r in caplog.records if "END summary" in r.getMessage()][0]
    assert "llm_calls=0 input_tokens=0 output_tokens=0 total_tokens=0" in end
