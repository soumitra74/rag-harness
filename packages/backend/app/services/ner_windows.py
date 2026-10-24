import re
from collections.abc import Callable

NER_WINDOW_TOKENS = 320
NER_OVERLAP_TOKENS = 64

_BOUNDARY = re.compile(r"(?<=[.!?])\s+|\n\s*\n")


def _trimmed(text: str, start: int, end: int) -> tuple[int, int] | None:
    piece = text[start:end]
    stripped = piece.strip()
    if not stripped:
        return None
    lead = len(piece) - len(piece.lstrip())
    return start + lead, start + lead + len(stripped)


def _sentence_spans(text: str) -> list[tuple[int, int]]:
    spans, start = [], 0
    for match in _BOUNDARY.finditer(text):
        spans.append((start, match.start()))
        start = match.end()
    spans.append((start, len(text)))
    return [s for s in (_trimmed(text, a, b) for a, b in spans) if s is not None]


def _split_oversized(
    text: str, start: int, end: int, count_tokens: Callable[[str], int], max_tokens: int,
) -> list[tuple[int, int]]:
    words = [m for m in re.finditer(r"\S+", text[start:end])]
    pieces, piece_start, piece_end = [], None, None
    for word in words:
        ws, we = start + word.start(), start + word.end()
        if piece_start is None:
            piece_start, piece_end = ws, we
        elif count_tokens(text[piece_start:we]) <= max_tokens:
            piece_end = we
        else:
            pieces.append((piece_start, piece_end))
            piece_start, piece_end = ws, we
    if piece_start is not None:
        pieces.append((piece_start, piece_end))
    return pieces


def ner_windows(
    text: str, count_tokens: Callable[[str], int],
    max_tokens: int = NER_WINDOW_TOKENS, overlap_tokens: int = NER_OVERLAP_TOKENS,
) -> list[tuple[int, int]]:
    """Character spans of overlapping windows that each fit max_tokens, cut at
    sentence or paragraph boundaries. Consecutive windows share roughly
    overlap_tokens of trailing sentences so an entity near a cut is seen whole
    in at least one window."""
    if not text.strip():
        return []

    segments: list[tuple[int, int]] = []
    for start, end in _sentence_spans(text):
        if count_tokens(text[start:end]) <= max_tokens:
            segments.append((start, end))
        else:
            segments.extend(_split_oversized(text, start, end, count_tokens, max_tokens))
    counts = [count_tokens(text[s:e]) for s, e in segments]

    windows: list[tuple[int, int]] = []
    i, n = 0, len(segments)
    while i < n:
        j, total = i, 0
        while j < n and total + counts[j] <= max_tokens:
            total += counts[j]
            j += 1
        j = max(j, i + 1)
        windows.append((segments[i][0], segments[j - 1][1]))
        if j >= n:
            break
        k, back = j, 0
        while k - 1 > i and back + counts[k - 1] <= overlap_tokens:
            k -= 1
            back += counts[k]
        i = max(k, i + 1)
    return windows
