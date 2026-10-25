from dataclasses import dataclass
from pathlib import Path

from app.api.routes_configs import ALL_CHUNKING_STRATEGIES
from app.providers.embedding_catalog import get_embedding_dim
from app.services.chunking import extract_pages
from app.services.llm_json import extract_json
from app.services.sweep_log import progress_log

SYSTEM_PROMPT = """You are helping configure a RAG (retrieval-augmented generation) pipeline for
a new corpus. Given a summary of the corpus's documents, propose 2-3 candidate ingestion
configs to try first in an automated search. Each candidate must have: chunking_strategy (one
of sliding_window, fixed_token, recursive_semantic, sentence_window, document_structure_aware),
chunk_size (int), chunk_overlap (int), embedding_provider (one of openai, ollama, local),
embedding_model (a model name valid for that provider), and a one-sentence rationale.

Respond with only a JSON array of candidate objects, no other text."""


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


def _list_pdf_filenames(corpus_dir: str) -> list[str]:
    path = Path(corpus_dir)
    if not path.is_dir():
        return []
    return sorted(p.name for p in path.glob("*.pdf"))


def _corpus_summary(corpus_dir: str, filenames: list[str]) -> str:
    path = Path(corpus_dir)
    page_counts = []
    for filename in filenames:
        try:
            page_counts.append(len(extract_pages((path / filename).read_bytes())))
        except Exception:
            # A corrupt/unreadable file shouldn't crash the inspector --
            # it's just excluded from the page-count signal.
            continue
    avg_pages = sum(page_counts) / len(page_counts) if page_counts else 0
    return (
        f"Corpus has {len(filenames)} PDF file(s): {', '.join(filenames)}. "
        f"Average extractable page count per file: {avg_pages:.1f}."
    )


async def inspect_corpus(provider, corpus_dir: str) -> list[CorpusConfigCandidate]:
    filenames = _list_pdf_filenames(corpus_dir)
    if not filenames:
        raise CorpusInspectionError(f"No PDF files found in corpus directory {corpus_dir!r}")

    prompt = f"{SYSTEM_PROMPT}\n\n{_corpus_summary(corpus_dir, filenames)}"
    response = await provider.complete(prompt)
    progress_log.info("corpus inspector raw reply: %s", response.text)
    raw_candidates = extract_json(response.text)

    candidates = []
    for item in raw_candidates:
        description = (
            f"{item.get('chunking_strategy')} size={item.get('chunk_size')} "
            f"overlap={item.get('chunk_overlap')} "
            f"{item.get('embedding_provider')}/{item.get('embedding_model')}"
        )
        if item.get("chunking_strategy") not in ALL_CHUNKING_STRATEGIES:
            progress_log.info("corpus inspector dropped candidate (unknown chunking strategy): %s",
                              description)
            continue
        if get_embedding_dim(item.get("embedding_provider"), item.get("embedding_model")) is None:
            progress_log.info("corpus inspector dropped candidate (unknown embedding provider/model): %s",
                              description)
            continue
        progress_log.info("corpus inspector accepted candidate: %s", description)
        candidates.append(CorpusConfigCandidate(
            chunking_strategy=item["chunking_strategy"],
            chunk_size=item["chunk_size"],
            chunk_overlap=item["chunk_overlap"],
            embedding_provider=item["embedding_provider"],
            embedding_model=item["embedding_model"],
            rationale=item["rationale"],
        ))
    if not candidates:
        raise CorpusInspectionError(
            "Corpus inspector proposed no candidate with a known embedding provider and model"
        )
    return candidates
