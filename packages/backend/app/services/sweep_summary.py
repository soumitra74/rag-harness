from app.services.sweep_next_config import TriedConfig

SYSTEM_PROMPT = """You are writing a retrospective summary for a completed RAG config
optimization sweep. Given the corpus summary, every config tried with its score, why the sweep
stopped, and the best config found, write a short markdown write-up covering: the recommendation
and why, suggestions for further tuning, and -- only where the results actually point that way
-- advisory suggestions about infrastructure (vector DB, relational DB, graph DB choice). Label
infrastructure suggestions as advisory, since they're outside what this automated search can
change itself."""


def _build_prompt(
    corpus_summary: str, tried: list[TriedConfig], stop_reason: str, best: TriedConfig | None,
) -> str:
    tried_lines = "\n".join(f"- {t.config} -> score {t.score}" for t in tried) or "(none)"
    best_line = f"{best.config} -> score {best.score}" if best else "(no config passed the gate)"
    return (
        f"{SYSTEM_PROMPT}\n\nCorpus summary: {corpus_summary}\n\n"
        f"Configs tried:\n{tried_lines}\n\n"
        f"Stop reason: {stop_reason}\n\nBest config: {best_line}"
    )


async def generate_sweep_summary(
    provider, corpus_summary: str, tried: list[TriedConfig], stop_reason: str,
    best: TriedConfig | None,
) -> str:
    prompt = _build_prompt(corpus_summary, tried, stop_reason, best)
    response = await provider.complete(prompt)
    return response.text
