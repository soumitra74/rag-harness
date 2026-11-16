import pytest

from app.api.routes_configs import ALL_CHUNKING_STRATEGIES
from app.providers.embedding_catalog import get_embedding_dim
from app.services.corpus_inspector import (
    BOOTSTRAP_CANDIDATES,
    CorpusConfigCandidate,
    CorpusInspectionError,
    inspect_corpus,
)


def test_inspect_corpus_raises_on_empty_corpus(tmp_path):
    with pytest.raises(CorpusInspectionError):
        inspect_corpus(str(tmp_path))


def test_inspect_corpus_returns_bootstrap_candidates(tmp_path):
    (tmp_path / "doc1.pdf").write_bytes(b"not a real pdf")

    candidates = inspect_corpus(str(tmp_path))

    assert candidates == BOOTSTRAP_CANDIDATES
    assert all(isinstance(c, CorpusConfigCandidate) for c in candidates)


def test_bootstrap_candidates_cover_several_chunkers_and_embedders():
    chunkers = {c.chunking_strategy for c in BOOTSTRAP_CANDIDATES}
    embedders = {(c.embedding_provider, c.embedding_model) for c in BOOTSTRAP_CANDIDATES}

    assert len(chunkers) >= 3
    assert len(embedders) >= 3


def test_bootstrap_candidates_are_valid_configs():
    for candidate in BOOTSTRAP_CANDIDATES:
        assert candidate.chunking_strategy in ALL_CHUNKING_STRATEGIES
        assert get_embedding_dim(candidate.embedding_provider, candidate.embedding_model) is not None
        assert candidate.chunk_overlap < candidate.chunk_size
