import contextvars
import logging
import time
from contextlib import contextmanager
from typing import Iterator

from app.services.pricing import token_cost_usd

# Child of uvicorn.error so records reach the console handler uvicorn installs
# (root has none) and inherit main.py's timestamped formatter.
progress_log = logging.getLogger("uvicorn.error.sweep")

_stage: contextvars.ContextVar[dict | None] = contextvars.ContextVar("sweep_stage", default=None)


class MeteredProvider:
    """Records each LLM call's tokens and cost against the active sweep stage."""

    def __init__(
        self, inner, db, provider_name: str, model: str,
        local_in_per_1k: float = 0.0, local_out_per_1k: float = 0.0,
    ):
        self._inner = inner
        self._db = db
        self._provider_name = provider_name
        self._model = model
        self._local_rates = (local_in_per_1k, local_out_per_1k)

    async def complete(self, prompt: str):
        response = await self._inner.complete(prompt)
        counter = _stage.get()
        if counter is None:
            return response
        input_tokens = response.input_tokens or 0
        output_tokens = response.output_tokens or 0
        cost = token_cost_usd(
            self._provider_name, self._model, input_tokens, output_tokens, *self._local_rates,
        )
        counter["calls"] += 1
        counter["input_tokens"] += input_tokens
        counter["output_tokens"] += output_tokens
        counter["cost_usd"] += cost or 0.0
        if cost is None:
            counter["unpriced_calls"] += 1
        if counter["sweep_id"] is not None:
            await self._db.insert_llm_usage(
                counter["sweep_id"], counter["name"], self._provider_name, self._model,
                input_tokens, output_tokens, cost or 0.0,
            )
        return response

    def __getattr__(self, name):
        return getattr(self._inner, name)


def _fields(fields: dict) -> str:
    return "".join(f" {key}={value}" for key, value in fields.items())


@contextmanager
def stage(sweep_id: int | None, name: str, **fields) -> Iterator[None]:
    counter = {
        "sweep_id": sweep_id, "name": name, "calls": 0, "input_tokens": 0,
        "output_tokens": 0, "cost_usd": 0.0, "unpriced_calls": 0,
    }
    token = _stage.set(counter)
    started = time.monotonic()
    progress_log.info("sweep %s: START %s%s", sweep_id, name, _fields(fields))
    try:
        yield
    except BaseException as exc:
        progress_log.error(
            "sweep %s: FAILED %s after %.1fs%s: %s",
            sweep_id, name, time.monotonic() - started, _fields(fields), exc,
        )
        raise
    else:
        total = counter["input_tokens"] + counter["output_tokens"]
        unpriced = f" unpriced_calls={counter['unpriced_calls']}" if counter["unpriced_calls"] else ""
        progress_log.info(
            "sweep %s: END %s in %.1fs%s llm_calls=%d input_tokens=%d output_tokens=%d "
            "total_tokens=%d cost=$%.6f%s",
            sweep_id, name, time.monotonic() - started, _fields(fields), counter["calls"],
            counter["input_tokens"], counter["output_tokens"], total, counter["cost_usd"], unpriced,
        )
    finally:
        _stage.reset(token)
