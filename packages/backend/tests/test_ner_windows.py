import pytest

from app.services.ner import ExtractedEntity, NERService
from app.services.ner_windows import NER_OVERLAP_TOKENS, ner_windows


def words(text: str) -> int:
    return len(text.split())


def _long_text(sentences: int) -> str:
    return " ".join(f"Sentence number {i} says the value is {i} units." for i in range(sentences))


def test_short_text_is_one_window_covering_the_text():
    text = "A short paragraph. It has two sentences."
    assert ner_windows(text, words) == [(0, len(text))]


def test_windows_fit_the_token_budget_and_cover_the_text():
    text = _long_text(200)
    windows = ner_windows(text, words, max_tokens=60, overlap_tokens=12)
    assert all(words(text[s:e]) <= 60 for s, e in windows)
    assert windows[0][0] == 0
    assert windows[-1][1] == len(text.rstrip())


def test_consecutive_windows_overlap_by_about_the_overlap_budget():
    text = _long_text(200)
    windows = ner_windows(text, words, max_tokens=60, overlap_tokens=12)
    for (_, prev_end), (next_start, _) in zip(windows, windows[1:]):
        assert next_start < prev_end
        assert words(text[next_start:prev_end]) <= 12 + 12


def test_windows_start_and_end_on_sentence_boundaries():
    text = _long_text(50)
    for start, end in ner_windows(text, words, max_tokens=60, overlap_tokens=12):
        assert text[start] != " " and text[end - 1] == "."


def test_oversized_sentence_is_split_by_words():
    text = " ".join(f"w{i}" for i in range(500))
    windows = ner_windows(text, words, max_tokens=100, overlap_tokens=10)
    assert all(words(text[s:e]) <= 100 for s, e in windows)
    assert windows[-1][1] == len(text)


def test_default_overlap_constant_is_the_advised_64():
    assert NER_OVERLAP_TOKENS == 64


class FakeGlinerModel:
    def __init__(self, per_call_results):
        self._results = list(per_call_results)
        self.received = []

    def predict_entities(self, text, labels, threshold=0.5):
        self.received.append(text)
        return self._results.pop(0)


class WordCountNER(NERService):
    def __init__(self, fake_model):
        super().__init__()
        self._fake_model = fake_model

    def _get_model(self):
        return self._fake_model

    def _count_tokens(self, text):
        return words(text)


@pytest.mark.asyncio
async def test_extract_windowed_maps_offsets_to_chunk_and_dedupes_overlaps(monkeypatch):
    import app.services.ner as ner_module

    text = _long_text(40)
    monkeypatch.setattr(ner_module, "ner_windows", lambda t, count: ner_windows(
        t, words, max_tokens=60, overlap_tokens=12,
    ))
    marker = "says the value is"
    per_window = []
    for start, end in ner_windows(text, words, max_tokens=60, overlap_tokens=12):
        window_text = text[start:end]
        per_window.append([
            {"start": at, "end": at + len(marker), "text": marker, "label": "METRIC", "score": 0.9}
            for at in _all_positions(window_text, marker)
        ])
    service = WordCountNER(FakeGlinerModel(per_window))

    entities = await service.extract_windowed(text)

    assert all(text[e.char_start:e.char_end] == marker for e in entities)
    assert len({e.char_start for e in entities}) == len(entities) == 40
    assert len(entities) < sum(len(w) for w in per_window)


def _all_positions(haystack: str, needle: str) -> list[int]:
    positions, at = [], haystack.find(needle)
    while at != -1:
        positions.append(at)
        at = haystack.find(needle, at + 1)
    return positions
