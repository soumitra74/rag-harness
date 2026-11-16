from dataclasses import dataclass

from app.services.corpus_inspector import CorpusConfigCandidate
from app.services.llm_json import extract_json

SYSTEM_PROMPT = """You are driving a greedy coordinate search over RAG ingestion configs for a
corpus, within a budget. Given the corpus summary, every config tried so far with its score (a
weighted composite of retrieval/judge metrics -- higher is better), and the remaining ingestion
budget, propose ONE new candidate config with a one-sentence rationale for why it might improve
the score.

The candidate must change the chunking_strategy or the embedding_model compared with the config it
is based on. Do not propose changes to chunk_size or chunk_overlap alone: size and overlap were
found to matter little compared with the chunker and embedder. Do not repeat a config already tried.

If you believe no further candidate is likely to improve on the best score tried so far, respond
with {"converged": true} instead.

Respond with only JSON: either {"converged": true} or {"converged": false, "candidate": {...}}
where candidate has chunking_strategy, chunk_size, chunk_overlap, embedding_provider,
embedding_model, and rationale."""


@dataclass
class TriedConfig:
    config: dict
    score: float | None
    config_id: int | None = None


@dataclass
class NextConfigResult:
    converged: bool
    candidate: CorpusConfigCandidate | None


def _build_prompt(
    corpus_summary: str, tried: list[TriedConfig], remaining_ingestion_budget: dict,
    user_hint: str | None,
) -> str:
    tried_lines = "\n".join(f"- {t.config} -> score {t.score}" for t in tried) or "(none yet)"
    prompt = (
        f"{SYSTEM_PROMPT}\n\nCorpus summary: {corpus_summary}\n\n"
        f"Configs tried so far:\n{tried_lines}\n\n"
        f"Remaining ingestion budget: {remaining_ingestion_budget}"
    )
    if user_hint:
        prompt += f"\n\nHuman suggestion for the next candidate: {user_hint}"
    return prompt


async def propose_next_config(
    provider, corpus_summary: str, tried: list[TriedConfig], remaining_ingestion_budget: dict,
    user_hint: str | None = None,
) -> NextConfigResult:
    prompt = _build_prompt(corpus_summary, tried, remaining_ingestion_budget, user_hint)
    response = await provider.complete(prompt)
    data = extract_json(response.text)

    if data.get("converged"):
        return NextConfigResult(converged=True, candidate=None)

    c = data["candidate"]
    candidate = CorpusConfigCandidate(
        chunking_strategy=c["chunking_strategy"],
        chunk_size=c["chunk_size"],
        chunk_overlap=c["chunk_overlap"],
        embedding_provider=c["embedding_provider"],
        embedding_model=c["embedding_model"],
        rationale=c["rationale"],
    )
    return NextConfigResult(converged=False, candidate=candidate)
