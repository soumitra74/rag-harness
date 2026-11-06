import json
import random
import re
from pathlib import Path

from app.services.chunking import extract_pages
from app.services.llm_json import extract_json

MIN_PAGE_CHARS = 200
QUERY_TYPES = {"extractive", "abstractive"}
AUTHOR_STOPWORDS = {"and", "the", "for"}

PAGE_PROMPT = """You write one question for a retrieval evaluation set. Use ONLY the page below.

Return a single JSON object with exactly these keys:
- "question": a question that can be answered from this page alone. Do not mention the
  paper's authors or title.
- "reference_answer": the answer, taken from the page.
- "query_type": "extractive" if the answer is stated directly, "abstractive" if it must be
  inferred or summarized.

Respond with only the JSON object, no other text.

Document: {filename}, page {page_number}
Page text:
{page_text}"""


class GoldenQAGenerationError(Exception):
    pass


def _list_pdf_filenames(corpus_dir: str) -> list[str]:
    path = Path(corpus_dir)
    if not path.is_dir():
        return []
    return sorted(p.name for p in path.glob("*.pdf"))


def _eligible_pages(pages: list[tuple[int, str]]) -> list[tuple[int, str]]:
    return [(number, text) for number, text in pages if len(text.strip()) >= MIN_PAGE_CHARS]


def _sample_pages(
    documents: list[tuple[str, list[tuple[int, str]]]], num_questions: int, seed: int = 0,
) -> list[tuple[str, int, str]]:
    """Seeded, reproducible sample: papers are visited in a shuffled order and
    each contributes a random page, so the set spans papers and page positions
    instead of always taking each paper's first page."""
    rng = random.Random(seed)
    eligible = [(filename, _eligible_pages(pages)) for filename, pages in documents]
    eligible = [(filename, pages) for filename, pages in eligible if pages]
    rng.shuffle(eligible)
    order = {filename: rng.sample(pages, len(pages)) for filename, pages in eligible}
    sample: list[tuple[str, int, str]] = []
    for rank in range(max((len(pages) for _, pages in eligible), default=0)):
        for filename, _ in eligible:
            if rank < len(order[filename]):
                number, text = order[filename][rank]
                sample.append((filename, number, text))
                if len(sample) == num_questions:
                    return sample
    return sample


def _load_documents(corpus_dir: str, filenames: list[str]) -> list[tuple[str, list[tuple[int, str]]]]:
    path = Path(corpus_dir)
    documents = []
    for filename in filenames:
        try:
            documents.append((filename, extract_pages((path / filename).read_bytes())))
        except Exception:
            continue
    return documents


def _author_terms(filename: str) -> set[str]:
    authors = filename.split(" - ")[0].replace(" et al.", "")
    return {w.lower() for w in re.findall(r"[A-Za-z]+", authors) if w.lower() not in AUTHOR_STOPWORDS and len(w) > 2}


def _entry_from_response(text: str, filename: str, page_number: int, entry_id: str) -> dict | None:
    try:
        parsed = extract_json(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    question = parsed.get("question")
    answer = parsed.get("reference_answer")
    query_type = parsed.get("query_type")
    if not (isinstance(question, str) and question.strip()):
        return None
    if not (isinstance(answer, str) and answer.strip()):
        return None
    if query_type not in QUERY_TYPES:
        return None
    question_words = set(re.findall(r"[A-Za-z]+", question.lower()))
    if _author_terms(filename) & question_words:
        return None
    return {
        "id": entry_id,
        "question": question.strip(),
        "reference_answer": answer.strip(),
        "golden_pages": [{"document": filename, "page": page_number}],
        "query_type": query_type,
    }


async def generate_golden_qa(
    provider, corpus_dir: str, output_path: str, num_questions: int = 20,
) -> int:
    filenames = _list_pdf_filenames(corpus_dir)
    documents = _load_documents(corpus_dir, filenames)
    sample = _sample_pages(documents, num_questions)
    if not sample:
        raise GoldenQAGenerationError("No pages with enough text to generate questions from")

    entries = []
    for filename, page_number, page_text in sample:
        response = await provider.complete(PAGE_PROMPT.format(
            filename=filename, page_number=page_number, page_text=page_text[:6000],
        ))
        entry = _entry_from_response(response.text, filename, page_number, f"q{len(entries) + 1:03d}")
        if entry is not None:
            entries.append(entry)

    if not entries:
        raise GoldenQAGenerationError("Failed to generate golden QA set: no valid entries")

    destination = Path(output_path)
    tmp_path = destination.with_suffix(destination.suffix + ".tmp")
    tmp_path.write_text(json.dumps(entries, indent=2), encoding="utf-8")
    tmp_path.replace(destination)
    return len(entries)
