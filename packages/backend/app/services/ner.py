import asyncio
from dataclasses import dataclass, replace

from gliner import GLiNER

from app.services.ner_windows import ner_windows

MODEL_NAME = "urchade/gliner_base"
ENTITY_LABELS = ["REPORT_NUMBER", "DATA_COUNTER_ID", "REPORT_DATE", "AIRCRAFT_TYPE"]
SCORE_THRESHOLD = 0.5


def normalize_entity_text(text: str) -> str:
    return " ".join(text.strip().lower().split())


@dataclass
class ExtractedEntity:
    entity_type: str
    entity_text: str
    normalized_text: str
    char_start: int
    char_end: int


class NERService:
    def __init__(self):
        self._model: GLiNER | None = None
        # Replaced per sweep by the generated graph-label set (see codegen.py).
        self.labels: list[str] = list(ENTITY_LABELS)

    def _get_model(self) -> GLiNER:
        # Lazy load, mirroring SentenceTransformersEmbeddingProvider: constructing
        # this service must stay cheap so unit tests don't trigger a model
        # download just by instantiating the class.
        if self._model is None:
            self._model = GLiNER.from_pretrained(MODEL_NAME)
        return self._model

    def _predict(self, text: str, labels: list[str]) -> list[ExtractedEntity]:
        model = self._get_model()
        raw = model.predict_entities(text, labels, threshold=SCORE_THRESHOLD)
        return [
            ExtractedEntity(
                entity_type=r["label"],
                entity_text=r["text"],
                normalized_text=normalize_entity_text(r["text"]),
                char_start=r["start"],
                char_end=r["end"],
            )
            for r in raw
        ]

    async def extract(self, text: str, labels: list[str] | None = None) -> list[ExtractedEntity]:
        return await asyncio.to_thread(self._predict, text, labels or self.labels)

    def _count_tokens(self, text: str) -> int:
        tokenizer = self._get_model().data_processor.transformer_tokenizer
        return len(tokenizer.encode(text, add_special_tokens=False))

    async def extract_windowed(self, text: str, labels: list[str] | None = None) -> list[ExtractedEntity]:
        # Long chunks exceed GLiNER's 384-token input and would lose their tail,
        # so NER runs on overlapping windows and offsets map back to the chunk.
        seen: set[tuple[str, int, int]] = set()
        merged: list[ExtractedEntity] = []
        for window_start, window_end in ner_windows(text, self._count_tokens):
            for entity in await self.extract(text[window_start:window_end], labels):
                start = entity.char_start + window_start
                end = entity.char_end + window_start
                key = (entity.entity_type, start, end)
                if key in seen:
                    continue
                seen.add(key)
                merged.append(replace(entity, char_start=start, char_end=end))
        return merged

    async def warm_up(self) -> None:
        """Force model load at a known point (container startup) instead of
        on the first real request -- see docs/pending-tasks.md's first-load
        freeze finding for local models."""
        await self.extract("warm up")
