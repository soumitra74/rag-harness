import json
import random
import re
from pathlib import Path

from app.services.chunking import extract_pages
from app.services.llm_json import extract_json

MIN_GOLDEN_QA_SIZE = 25
MIN_PAGE_CHARS = 200
MAX_PAGE_CHARS = 6000
QUERY_TYPES = {"extractive", "abstractive"}
AUTHOR_STOPWORDS = {"and", "the", "for"}

PAGE_PROMPT = """You write one question for a retrieval evaluation set. Use ONLY the page below.

Return a single JSON object with exactly these keys:
- "question": a question that can be answered from this page alone. Name the subject
  specifically (the model, dataset, report, or method the page is about) so the question
  points to this document and not to any other paper. Do not mention the paper's authors.
  Do not refer to figures or tables by number.
- "reference_answer": the answer, taken from the page.
- "query_type": "extractive" if the answer is stated directly, "abstractive" if it must be
  inferred or summarized.

Respond with only the JSON object, no other text.

Document: {filename}, page {page_number}
Page text:
{page_text}"""

VERIFY_PROMPT = """Decide whether the page below supports the reference answer to the question.

Question: {question}
Reference answer: {answer}
Page text:
{page_text}

Respond with only a JSON object: {{"supported": true or false, "quote": "a sentence copied word-for-word from the page that supports the answer, or empty if none"}}"""

CLOSED_BOOK_PROMPT = """Without any document, could you answer the question below correctly from general knowledge alone? Answer honestly; do not guess.

Question: {question}

Respond with only a JSON object: {{"answerable_without_document": true or false}}"""


class GoldenQAGenerationError(Exception):
    pass


def _list_pdf_filenames(corpus_dir: str) -> list[str]:
    path = Path(corpus_dir)
    if not path.is_dir():
        return []
    return sorted(p.name for p in path.glob("*.pdf"))


def _eligible_pages(pages: list[tuple[int, str]]) -> list[tuple[int, str]]:
    return [(number, text) for number, text in pages if len(text.strip()) >= MIN_PAGE_CHARS]


def _page_queue(
    documents: list[tuple[str, list[tuple[int, str]]]], seed: int = 0,
) -> list[tuple[str, int, str]]:
    """Seeded, reproducible order over every eligible page: papers are visited in
    a shuffled order and each round takes one page from every paper, so any
    prefix of the queue spans papers and page positions."""
    rng = random.Random(seed)
    eligible = [(filename, _eligible_pages(pages)) for filename, pages in documents]
    eligible = [(filename, pages) for filename, pages in eligible if pages]
    rng.shuffle(eligible)
    order = {filename: rng.sample(pages, len(pages)) for filename, pages in eligible}
    queue: list[tuple[str, int, str]] = []
    for rank in range(max((len(pages) for _, pages in eligible), default=0)):
        for filename, _ in eligible:
            if rank < len(order[filename]):
                number, text = order[filename][rank]
                queue.append((filename, number, text))
    return queue


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


def _normalize_question(question: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", question.lower()))


def _is_not_answerable_without_document(text: str) -> bool:
    try:
        verdict = extract_json(text)
    except json.JSONDecodeError:
        return False
    return isinstance(verdict, dict) and verdict.get("answerable_without_document") is False


def load_corpus_pages(corpus_dir: str) -> dict[str, list[tuple[int, str]]]:
    return dict(_load_documents(corpus_dir, _list_pdf_filenames(corpus_dir)))


def check_golden_entries(entries: list[dict], documents: dict[str, list[tuple[int, str]]]) -> list[str]:
    problems = []
    seen_questions: set[str] = set()
    for entry in entries:
        entry_id = entry.get("id")
        normalized = _normalize_question(str(entry.get("question", "")))
        if normalized in seen_questions:
            problems.append(f"{entry_id}: duplicate question")
        seen_questions.add(normalized)
        references = entry.get("golden_pages") or []
        if not references:
            problems.append(f"{entry_id}: no golden pages")
        for reference in references:
            document = reference.get("document") if isinstance(reference, dict) else None
            page = reference.get("page") if isinstance(reference, dict) else None
            if document not in documents:
                problems.append(f"{entry_id}: unknown document {document!r}")
                continue
            pages = dict(documents[document])
            if page not in pages:
                problems.append(f"{entry_id}: page {page!r} out of range for {document!r}")
            elif len(pages[page].strip()) < MIN_PAGE_CHARS:
                problems.append(f"{entry_id}: page {page!r} of {document!r} has no usable text")
    return problems


def _squash(text: str) -> str:
    return " ".join(text.split()).lower()


def _is_verified(verdict_text: str, page_text: str) -> bool:
    try:
        verdict = extract_json(verdict_text)
    except json.JSONDecodeError:
        return False
    if not isinstance(verdict, dict) or verdict.get("supported") is not True:
        return False
    quote = verdict.get("quote")
    return isinstance(quote, str) and bool(quote.strip()) and _squash(quote) in _squash(page_text)


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
    if "�" in question or "�" in answer:
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
    provider, corpus_dir: str, output_path: str, num_questions: int = MIN_GOLDEN_QA_SIZE, verifier=None,
) -> int:
    verifier = verifier if verifier is not None else provider
    filenames = _list_pdf_filenames(corpus_dir)
    documents = _load_documents(corpus_dir, filenames)
    queue = _page_queue(documents)
    if not queue:
        raise GoldenQAGenerationError("No pages with enough text to generate questions from")

    entries = []
    candidates = 0
    seen_questions: set[str] = set()
    for filename, page_number, page_text in queue:
        if len(entries) == num_questions:
            break
        shown = page_text[:MAX_PAGE_CHARS]
        response = await provider.complete(PAGE_PROMPT.format(
            filename=filename, page_number=page_number, page_text=shown,
        ))
        entry = _entry_from_response(response.text, filename, page_number, f"q{len(entries) + 1:03d}")
        if entry is None:
            continue
        normalized = _normalize_question(entry["question"])
        if normalized in seen_questions:
            continue
        seen_questions.add(normalized)
        candidates += 1
        verdict = await verifier.complete(VERIFY_PROMPT.format(
            question=entry["question"], answer=entry["reference_answer"], page_text=shown,
        ))
        if not _is_verified(verdict.text, shown):
            continue
        closed_book = await verifier.complete(CLOSED_BOOK_PROMPT.format(question=entry["question"]))
        if _is_not_answerable_without_document(closed_book.text):
            entries.append(entry)

    if not entries:
        raise GoldenQAGenerationError(
            f"Failed to generate golden QA set: 0 of {candidates} candidate questions passed verification"
        )

    problems = check_golden_entries(entries, dict(documents))
    if problems:
        raise GoldenQAGenerationError("Generated golden QA set is invalid: " + "; ".join(problems[:5]))

    destination = Path(output_path)
    tmp_path = destination.with_suffix(destination.suffix + ".tmp")
    tmp_path.write_text(json.dumps(entries, indent=2), encoding="utf-8")
    tmp_path.replace(destination)
    return len(entries)
