import asyncio

import httpx

from app.config import Settings
from app.providers.base import LLMResponse, build_prompt


class OllamaChatProvider:
    def __init__(self, settings: Settings):
        self._base_url = settings.ollama_base_url
        self._model = settings.ollama_chat_model
        self._num_ctx = settings.ollama_num_ctx

    async def generate(self, question: str, context_chunks: list[str]) -> LLMResponse:
        prompt = build_prompt(question, context_chunks)
        async with httpx.AsyncClient(base_url=self._base_url, timeout=300) as client:
            response = await client.post(
                "/api/generate",
                json={
                    "model": self._model, "prompt": prompt, "stream": False,
                    "options": {"num_ctx": self._num_ctx},
                },
            )
            response.raise_for_status()
            data = response.json()
            return LLMResponse(
                text=data["response"],
                input_tokens=data.get("prompt_eval_count"),
                output_tokens=data.get("eval_count"),
            )


class OllamaEmbeddingProvider:
    def __init__(self, settings: Settings):
        self._base_url = settings.ollama_base_url
        self._model = settings.ollama_embedding_model
        self._concurrency = settings.ollama_embedding_concurrency

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        semaphore = asyncio.Semaphore(self._concurrency)
        async with httpx.AsyncClient(
            base_url=self._base_url, timeout=60,
            limits=httpx.Limits(max_connections=self._concurrency),
        ) as client:

            async def embed_one(text: str) -> list[float]:
                async with semaphore:
                    response = await client.post(
                        "/api/embeddings",
                        json={"model": self._model, "prompt": text},
                    )
                    response.raise_for_status()
                    return response.json()["embedding"]

            tasks = [asyncio.create_task(embed_one(text)) for text in texts]
            try:
                return await asyncio.gather(*tasks)
            except BaseException:
                for task in tasks:
                    task.cancel()
                raise

    async def embed_query(self, text: str) -> list[float]:
        [embedding] = await self.embed_documents([text])
        return embedding
