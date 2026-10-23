import ast
import asyncio
import base64
import json
import re
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import httpx

from app.services.chunking import extract_pages, normalize_page_text
from app.services.ner import ENTITY_LABELS
from app.services.sweep_budget import budget_exhausted

CODEGEN_AXES = ("cost_usd", "minutes", "attempts")
LABEL_PATTERN = re.compile(r"^[A-Z][A-Z_]{2,31}$")
MIN_LABEL_SAMPLE_HITS = 2


class CodegenFailed(Exception):
    pass


class SandboxRunError(Exception):
    pass


@dataclass(frozen=True)
class CodegenTask:
    name: str
    kind: str  # "python" (module with an entrypoint) or "json" (a value)
    brief: str
    contract: str
    entrypoint: str | None = None


OCR_TASK = CodegenTask(
    name="ocr",
    kind="python",
    entrypoint="ocr_page",
    brief=(
        "Write code that extracts the text of one PDF page that has no embedded text layer "
        "(a scanned or image-only page)."
    ),
    contract=(
        "def ocr_page(pdf_path: str, page_number: int) -> str\n"
        "- pdf_path: a PDF in the current working directory\n"
        "- page_number: 1-based page number\n"
        "- returns the page's plain UTF-8 text\n"
        "- render the page with pypdfium2 and OCR it with pytesseract (English). "
        "Only use pypdfium2, pytesseract, PIL and the standard library."
    ),
)

GRAPH_LABELS_TASK = CodegenTask(
    name="graph_labels",
    kind="json",
    brief=(
        "Propose entity labels for a zero-shot NER model (GLiNER). The labels will tag entities "
        "in this corpus so that chunks mentioning the same entities can be linked in a graph."
    ),
    contract=(
        "A JSON array of 3 to 10 distinct label names, each matching ^[A-Z][A-Z_]{2,31}$, "
        "naming entity types that actually occur in the corpus excerpts."
    ),
)

TASKS = {OCR_TASK.name: OCR_TASK, GRAPH_LABELS_TASK.name: GRAPH_LABELS_TASK}


@dataclass
class CodegenContext:
    settings: object
    ner: object
    sandbox_url: str


async def run_in_sandbox(
    sandbox_url: str, code: str, entrypoint: str, args: list,
    files: dict[str, bytes] | None = None, timeout_s: int = 60,
):
    body = {
        "code": code,
        "entrypoint": entrypoint,
        "args": args,
        "timeout_s": timeout_s,
        "files": {name: base64.b64encode(data).decode() for name, data in (files or {}).items()},
    }
    async with httpx.AsyncClient(timeout=timeout_s + 15) as client:
        response = await client.post(f"{sandbox_url}/run", json=body)
    response.raise_for_status()
    data = response.json()
    if not data["ok"]:
        raise SandboxRunError(data["error"] or "sandbox run failed")
    return data["result"]


def build_prompt(task: CodegenTask, corpus_context: str | None, feedback: str | None) -> str:
    parts = [f"Task: {task.brief}", f"Contract:\n{task.contract}"]
    if corpus_context:
        parts.append(f"Corpus excerpts:\n{corpus_context}")
    if feedback:
        parts.append(f"The previous attempt failed validation:\n{feedback}\nFix these problems.")
    if task.kind == "python":
        parts.append("Reply with exactly one ```python fenced block holding the full module, nothing else.")
    else:
        parts.append("Reply with exactly one ```json fenced block holding the value, nothing else.")
    return "\n\n".join(parts)


def extract_block(task: CodegenTask, text: str) -> str:
    match = re.search(rf"```{task.kind}\s*\n(.*?)```", text, re.S)
    if match is None:
        raise ValueError(f"response has no ```{task.kind} block")
    return match.group(1)


def check_python_module(code: str, entrypoint: str) -> None:
    tree = ast.parse(code)
    defined = {n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    if entrypoint not in defined:
        raise ValueError(f"module must define def {entrypoint}(...)")


def blank_page_samples(corpus_dir: str, max_files: int) -> list[tuple[str, bytes, int]]:
    samples = []
    for path in sorted(Path(corpus_dir).glob("*.pdf")):
        pdf_bytes = path.read_bytes()
        blank = [n for n, text in extract_pages(pdf_bytes) if not text.strip()]
        if blank:
            samples.append((path.name, pdf_bytes, blank[0]))
            if len(samples) >= max_files:
                break
    return samples


def text_samples(corpus_dir: str, max_samples: int) -> list[str]:
    samples: list[str] = []
    for path in sorted(Path(corpus_dir).glob("*.pdf")):
        for _, text in extract_pages(path.read_bytes()):
            body = text.strip()
            if len(body) >= 400:
                samples.append(body[:1500])
                if len(samples) >= max_samples:
                    return samples
    return samples


async def validate_ocr(ctx: CodegenContext, code: str) -> list[str]:
    samples = await asyncio.to_thread(blank_page_samples, ctx.settings.corpus_dir, 2)
    if not samples:
        return []
    problems = []
    for _, pdf_bytes, page in samples:
        try:
            text = await run_in_sandbox(
                ctx.sandbox_url, code, OCR_TASK.entrypoint, ["input.pdf", page],
                {"input.pdf": pdf_bytes}, timeout_s=120,
            )
        except SandboxRunError as exc:
            problems.append(f"ocr_page raised on page {page}: {exc}")
            continue
        if not isinstance(text, str):
            problems.append("ocr_page must return a str")
            continue
        if len(re.findall(r"[A-Za-z]{3,}", text)) >= 3:
            return []
        problems.append(f"page {page} produced no readable text")
    return problems


async def validate_graph_labels(ctx: CodegenContext, labels) -> list[str]:
    if not isinstance(labels, list) or not all(isinstance(label, str) for label in labels):
        return ["the value must be a JSON array of strings"]
    problems = []
    if not 3 <= len(labels) <= 10:
        problems.append(f"need 3 to 10 labels, got {len(labels)}")
    if len(set(labels)) != len(labels):
        problems.append("labels must be distinct")
    invalid = [label for label in labels if not LABEL_PATTERN.match(label)]
    if invalid:
        problems.append(f"labels must match ^[A-Z][A-Z_]{{2,31}}$: {invalid}")
    if problems:
        return problems

    samples = await asyncio.to_thread(text_samples, ctx.settings.corpus_dir, 15)
    hits: Counter = Counter()
    for sample in samples:
        hits.update({entity.entity_type for entity in await ctx.ner.extract(sample, labels)})
    unmatched = [label for label in labels if hits[label] < MIN_LABEL_SAMPLE_HITS]
    if unmatched:
        problems.append(
            f"these labels matched fewer than {MIN_LABEL_SAMPLE_HITS} corpus excerpts, "
            f"replace them with entity types the corpus actually contains: {unmatched}"
        )
    return problems


VALIDATORS = {"ocr": validate_ocr, "graph_labels": validate_graph_labels}


def _llm_cost(settings, response) -> float:
    return (
        (response.input_tokens or 0) / 1000 * settings.judge_cost_per_1k_in_usd
        + (response.output_tokens or 0) / 1000 * settings.judge_cost_per_1k_out_usd
    )


async def _graph_corpus_context(ctx: CodegenContext) -> str:
    samples = await asyncio.to_thread(text_samples, ctx.settings.corpus_dir, 5)
    return "\n---\n".join(sample[:800] for sample in samples)


async def run_codegen_task(
    db, provider, ctx: CodegenContext, sweep_id: int, task: CodegenTask, budget: dict,
) -> str:
    corpus_context = await _graph_corpus_context(ctx) if task.name == "graph_labels" else None
    feedback: str | None = None
    attempt = 0
    while True:
        spent = await db.sum_codegen_spend(sweep_id)
        if budget_exhausted(spent, budget):
            exhausted = [axis for axis, cap in budget.items() if spent.get(axis, 0) >= cap]
            raise CodegenFailed(
                f"{task.name}: budget exhausted ({', '.join(exhausted)}) after {attempt} "
                f"attempt(s). Last problem: {feedback or 'none'}"
            )
        attempt += 1
        started = time.monotonic()
        content = None
        cost = 0.0
        try:
            response = await provider.complete(build_prompt(task, corpus_context, feedback))
            cost = _llm_cost(ctx.settings, response)
            content = extract_block(task, response.text)
            if task.kind == "python":
                check_python_module(content, task.entrypoint)
                problems = await VALIDATORS[task.name](ctx, content)
            else:
                problems = await VALIDATORS[task.name](ctx, json.loads(content))
        except Exception as exc:
            problems = [f"{type(exc).__name__}: {exc}"]
        duration_ms = int((time.monotonic() - started) * 1000)
        await db.insert_codegen_run(
            sweep_id, task.name, attempt, not problems, "; ".join(problems) or None,
            cost, duration_ms,
        )
        if not problems:
            await db.upsert_generated_artifact(sweep_id, task.name, content)
            return content
        feedback = "\n".join(problems)


async def run_codegen_tasks(
    db, provider, ctx: CodegenContext, sweep_id: int, task_names: list[str], budget: dict,
) -> None:
    for name in task_names:
        task = TASKS[name]
        if task.name == "ocr" and not await asyncio.to_thread(
            blank_page_samples, ctx.settings.corpus_dir, 1,
        ):
            continue
        await run_codegen_task(db, provider, ctx, sweep_id, task, budget)


def make_ocr_page(sandbox_url: str, code: str):
    async def ocr_page(pdf_bytes: bytes, page_number: int) -> str:
        return await run_in_sandbox(
            sandbox_url, code, OCR_TASK.entrypoint, ["input.pdf", page_number],
            {"input.pdf": pdf_bytes}, timeout_s=120,
        )

    return ocr_page


def apply_graph_labels(ner, content: str | None) -> None:
    if content is not None:
        ner.labels = json.loads(content)


def reset_graph_labels(ner) -> None:
    ner.labels = list(ENTITY_LABELS)
