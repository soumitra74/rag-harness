from dataclasses import dataclass
from pathlib import Path


class CorpusInspectionError(Exception):
    pass


@dataclass
class CorpusConfigCandidate:
    chunking_strategy: str
    chunk_size: int
    chunk_overlap: int
    embedding_provider: str
    embedding_model: str
    rationale: str


# One config per chunker and embedder family, so the search starts from
# distinct points instead of near-duplicates of a single chunker.
BOOTSTRAP_CANDIDATES = [
    CorpusConfigCandidate(
        chunking_strategy="recursive_semantic", chunk_size=800, chunk_overlap=120,
        embedding_provider="openai", embedding_model="text-embedding-3-small",
        rationale="Recursive chunking keeps sentences and numbers together; OpenAI embeddings as the reference embedder.",
    ),
    CorpusConfigCandidate(
        chunking_strategy="recursive_semantic", chunk_size=800, chunk_overlap=120,
        embedding_provider="ollama", embedding_model="nomic-embed-text",
        rationale="Same chunks as the reference, embedded locally, to isolate the embedder effect.",
    ),
    CorpusConfigCandidate(
        chunking_strategy="sentence_window", chunk_size=2, chunk_overlap=0,
        embedding_provider="ollama", embedding_model="nomic-embed-text",
        rationale="Sentence-level matching with two sentences of context, for precise retrieval.",
    ),
    CorpusConfigCandidate(
        chunking_strategy="fixed_token", chunk_size=512, chunk_overlap=64,
        embedding_provider="ollama", embedding_model="nomic-embed-text",
        rationale="Fixed-size token windows as a simple baseline chunker.",
    ),
    CorpusConfigCandidate(
        chunking_strategy="recursive_semantic", chunk_size=150, chunk_overlap=25,
        embedding_provider="ollama", embedding_model="mxbai-embed-large",
        rationale="Small chunks that fit mxbai-embed-large's input limit, as a third embedder family.",
    ),
]


def _list_pdf_filenames(corpus_dir: str) -> list[str]:
    path = Path(corpus_dir)
    if not path.is_dir():
        return []
    return sorted(p.name for p in path.glob("*.pdf"))


def inspect_corpus(corpus_dir: str) -> list[CorpusConfigCandidate]:
    if not _list_pdf_filenames(corpus_dir):
        raise CorpusInspectionError(f"No PDF files found in corpus directory {corpus_dir!r}")
    return list(BOOTSTRAP_CANDIDATES)
