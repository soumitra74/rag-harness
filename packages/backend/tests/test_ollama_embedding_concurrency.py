import asyncio
import json

import httpx
import pytest

from app.config import Settings
from app.providers.ollama_provider import OllamaEmbeddingProvider

_RealAsyncClient = httpx.AsyncClient


def _patch_transport(monkeypatch, handler):
    monkeypatch.setattr(
        httpx, "AsyncClient",
        lambda *a, **kw: _RealAsyncClient(*a, transport=httpx.MockTransport(handler), **kw),
    )


def _settings(concurrency: int) -> Settings:
    return Settings(ollama_embedding_concurrency=concurrency)


@pytest.mark.asyncio
async def test_embed_documents_preserves_input_order(monkeypatch):
    async def handler(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(0.01 * (5 - int(json.loads(request.content)["prompt"])))
        prompt = json.loads(request.content)["prompt"]
        return httpx.Response(200, json={"embedding": [float(prompt)]})

    _patch_transport(monkeypatch, handler)
    provider = OllamaEmbeddingProvider(_settings(concurrency=4))

    result = await provider.embed_documents([str(i) for i in range(5)])

    assert result == [[float(i)] for i in range(5)]


@pytest.mark.asyncio
async def test_embed_documents_caps_in_flight_requests(monkeypatch):
    in_flight = 0
    max_in_flight = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal in_flight, max_in_flight
        in_flight += 1
        max_in_flight = max(max_in_flight, in_flight)
        await asyncio.sleep(0.01)
        in_flight -= 1
        return httpx.Response(200, json={"embedding": [0.0]})

    _patch_transport(monkeypatch, handler)
    provider = OllamaEmbeddingProvider(_settings(concurrency=3))

    await provider.embed_documents([f"t{i}" for i in range(12)])

    assert max_in_flight == 3


@pytest.mark.asyncio
async def test_embed_documents_failure_raises_and_cancels_siblings(monkeypatch):
    completed = []

    async def handler(request: httpx.Request) -> httpx.Response:
        prompt = json.loads(request.content)["prompt"]
        if prompt == "bad":
            return httpx.Response(500, json={"error": "boom"})
        await asyncio.sleep(0.05)
        completed.append(prompt)
        return httpx.Response(200, json={"embedding": [0.0]})

    _patch_transport(monkeypatch, handler)
    provider = OllamaEmbeddingProvider(_settings(concurrency=2))

    with pytest.raises(httpx.HTTPStatusError):
        await provider.embed_documents(["bad", "a", "b", "c", "d"])
    await asyncio.sleep(0.1)

    assert len(completed) < 4


@pytest.mark.asyncio
async def test_embed_documents_empty_input_returns_empty():
    provider = OllamaEmbeddingProvider(_settings(concurrency=4))

    assert await provider.embed_documents([]) == []
