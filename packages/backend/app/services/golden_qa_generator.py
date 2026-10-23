import json
from pathlib import Path

from app.services.chunking import extract_pages
from app.services.llm_json import extract_json

REQUIRED_KEYS = {"id", "question", "reference_answer", "golden_pages", "query_type"}

SYSTEM_PROMPT = """You are generating a golden question-answer evaluation set for a RAG
pipeline, grounded in the corpus below. Produce {num_questions} questions. Each entry must be a
JSON object with exactly these keys: id (string, e.g. "q001"), question (string),
reference_answer (string), golden_pages (array of {{"document": filename, "page": page_number}}
pairs identifying where the answer is found), and query_type (one of "extractive",
"multi-hop", "abstractive").

Respond with only a JSON array of entries, no other text."""


class GoldenQAGenerationError(Exception):
    pass


def _list_pdf_filenames(corpus_dir: str) -> list[str]:
    path = Path(corpus_dir)
    if not path.is_dir():
        return []
    return sorted(p.name for p in path.glob("*.pdf"))


def _corpus_text_summary(corpus_dir: str, filenames: list[str]) -> str:
    path = Path(corpus_dir)
    parts = []
    for filename in filenames:
        try:
            pages = extract_pages((path / filename).read_bytes())
        except Exception:
            continue
        parts.append(f"{filename} ({len(pages)} pages)")
    return "Corpus files: " + ", ".join(parts) if parts else "Corpus files: (none readable)"


def _validate_entries(entries) -> None:
    if not isinstance(entries, list) or not entries:
        raise GoldenQAGenerationError("Expected a non-empty JSON array of golden QA entries")
    for entry in entries:
        if not isinstance(entry, dict) or not REQUIRED_KEYS.issubset(entry.keys()):
            raise GoldenQAGenerationError(f"Entry missing required keys: {entry!r}")


async def generate_golden_qa(
    provider, corpus_dir: str, output_path: str, num_questions: int = 20,
) -> int:
    filenames = _list_pdf_filenames(corpus_dir)
    prompt = (
        SYSTEM_PROMPT.format(num_questions=num_questions)
        + "\n\n" + _corpus_text_summary(corpus_dir, filenames)
    )
    response = await provider.complete(prompt)

    try:
        entries = extract_json(response.text)
        _validate_entries(entries)
    except (json.JSONDecodeError, GoldenQAGenerationError) as exc:
        raise GoldenQAGenerationError(f"Failed to generate golden QA set: {exc}") from exc

    destination = Path(output_path)
    tmp_path = destination.with_suffix(destination.suffix + ".tmp")
    tmp_path.write_text(json.dumps(entries, indent=2), encoding="utf-8")
    tmp_path.replace(destination)
    return len(entries)
