import json
import time
from pathlib import Path

from app.providers.embedding_catalog import get_embedding_dim
from app.providers.factory import get_embedding_provider_for_config
from app.providers.reranker import DEFAULT_RERANK_CANDIDATES, DEFAULT_RERANK_MODEL, get_reranker
from app.routing.from_settings import judge_identity
from app.services.corpus_inspector import CorpusInspectionError, inspect_corpus
from app.services.eval_run import load_golden_dataset, new_eval_job, run_eval_job
from app.services.golden_qa_generator import GoldenQAGenerationError, generate_golden_qa
from app.services.graph_backfill import backfill_graph_for_config
from app.services.ingest import ingest_pdf, record_ingestion_metrics
from app.services.quality_gate import load_quality_gate
from app.services.sweep_budget import check_eval_budget, check_ingestion_budget
from app.services.sweep_next_config import TriedConfig, propose_next_config
from app.services.sweep_scoring import load_scoring_weights, rank_configs
from app.services.sweep_summary import generate_sweep_summary

GOLDEN_QA_REQUIRED_KEYS = {"id", "question", "reference_answer", "golden_pages", "query_type"}

# Global Constraint: "3 consecutive non-improving candidates (configurable)
# triggers awaiting_suggestion". Enforced here against the real persisted
# scores, not left to the next-config subagent's own self-report (review
# finding I-4) -- the LLM can still converge sooner on its own judgment,
# but it was never actually forced to stop even after many non-improving
# tries in a row.
CONVERGENCE_THRESHOLD = 3

INNER_GRID = [
    {"top_k": top_k, "rerank_model": rerank_model, "graph_augmented": graph_augmented}
    for top_k in (5, 10)
    for rerank_model in (None, DEFAULT_RERANK_MODEL)
    for graph_augmented in (False, True)
]


def list_pdf_filenames(corpus_dir: str) -> list[str]:
    path = Path(corpus_dir)
    if not path.is_dir():
        return []
    return sorted(p.name for p in path.glob("*.pdf"))


async def _ensure_golden_qa(provider, settings, generate_golden_qa_flag: bool) -> None:
    path = Path(settings.eval_dataset_path)
    if path.exists():
        entries = json.loads(path.read_text(encoding="utf-8"))
        for entry in entries:
            if not GOLDEN_QA_REQUIRED_KEYS.issubset(entry.keys()):
                raise GoldenQAGenerationError(
                    f"Existing golden QA file {settings.eval_dataset_path!r} has an entry "
                    f"missing required keys: {entry!r}"
                )
        return
    if generate_golden_qa_flag:
        await generate_golden_qa(provider, settings.corpus_dir, settings.eval_dataset_path)


async def _ingest_and_backfill_config(db, ner, settings, config, sweep_id: int | None) -> None:
    embedding_provider = get_embedding_provider_for_config(settings, config)
    corpus_path = Path(settings.corpus_dir)
    start = time.monotonic()
    input_size_bytes = 0
    results = []
    for filename in list_pdf_filenames(settings.corpus_dir):
        pdf_bytes = (corpus_path / filename).read_bytes()
        input_size_bytes += len(pdf_bytes)
        try:
            result = await ingest_pdf(filename, pdf_bytes, db, embedding_provider, config)
            results.append({
                "filename": filename, "document_id": result.document_id,
                "chunk_count": result.chunk_count, "page_count": result.page_count,
            })
        except Exception as exc:
            results.append({"filename": filename, "error": str(exc)})
    await record_ingestion_metrics(db, config, results, input_size_bytes, start, settings, sweep_id)
    await backfill_graph_for_config(db, ner, config)


async def _create_and_ingest_candidate(db, ner, settings, candidate, sweep_id: int | None):
    embedding_dim = get_embedding_dim(candidate.embedding_provider, candidate.embedding_model)
    config = await db.create_experiment_config(
        chunking_strategy=candidate.chunking_strategy,
        chunk_size=candidate.chunk_size,
        chunk_overlap=candidate.chunk_overlap,
        embedding_provider=candidate.embedding_provider,
        embedding_model=candidate.embedding_model,
        embedding_dim=embedding_dim,
    )
    await _ingest_and_backfill_config(db, ner, settings, config, sweep_id)
    return config


async def run_bootstrap(
    db, provider, ner, settings, sweep_id: int, generate_golden_qa_flag: bool,
) -> None:
    try:
        # Corpus-emptiness must be checked before golden-QA generation: with
        # the flag on, generating first would ask the LLM to hallucinate a
        # golden QA set from an empty corpus and persist it, before the
        # clear empty-corpus error from inspect_corpus ever fires (review
        # finding I-2). inspect_corpus itself raises on empty, so simply
        # calling it first is enough to gate both checks on the same signal.
        candidates = await inspect_corpus(provider, settings.corpus_dir)
        await _ensure_golden_qa(provider, settings, generate_golden_qa_flag)
        await db.set_corpus_snapshot(sweep_id, list_pdf_filenames(settings.corpus_dir))

        ingested = []
        for candidate in candidates:
            config = await _create_and_ingest_candidate(db, ner, settings, candidate, sweep_id)
            ingested.append((config, candidate.rationale))

        for config, rationale in ingested:
            # "ingested", not "proposed" -- bootstrap already ingested and
            # graph-backfilled this config above, so the loop (Task 13) must
            # not redo backfill for it (ingest_pdf tolerates a repeat call,
            # backfill_graph_for_config does not -- it would insert duplicate
            # mentions rows).
            await db.create_sweep_config(sweep_id, config.id, rationale, "ingested", "bootstrap")

        await db.update_sweep_status(sweep_id, "awaiting_approval")
    except (CorpusInspectionError, GoldenQAGenerationError) as exc:
        await db.update_sweep_status(sweep_id, "failed", error_message=str(exc))


def _corpus_summary_text(settings) -> str:
    filenames = list_pdf_filenames(settings.corpus_dir)
    return f"Corpus has {len(filenames)} PDF file(s): {', '.join(filenames)}."


async def _ensure_ingested_for_loop(db, ner, settings, sweep_config, sweep_id: int):
    config = await db.get_experiment_config(sweep_config["config_id"])
    if sweep_config["status"] not in ("ingested", "evaluated"):
        await _ingest_and_backfill_config(db, ner, settings, config, sweep_id)
        await db.update_sweep_config_status(sweep_config["id"], "ingested")
    return config


async def _run_eval_combo(
    db, provider, chat_provider, judge, ner, settings, config, combo, sweep_id,
) -> int:
    dataset = load_golden_dataset(settings.eval_dataset_path)
    embedding_provider = get_embedding_provider_for_config(settings, config)
    rerank_candidates = DEFAULT_RERANK_CANDIDATES if combo["rerank_model"] else None
    reranker = get_reranker({}, combo["rerank_model"]) if combo["rerank_model"] else None

    cycle_id = await db.create_eval_cycle(
        config.id, len(dataset), *judge_identity(settings), combo["graph_augmented"],
        combo["rerank_model"], rerank_candidates, sweep_id=sweep_id,
    )
    job = new_eval_job(config.id, len(dataset))
    await run_eval_job(
        job, db, config, embedding_provider, chat_provider, judge, dataset, combo["top_k"],
        cycle_id, settings.eval_judge_concurrency, settings.eval_judge_combine_calls,
        combo["graph_augmented"], ner, settings.eval_recall_k_values,
        reranker=reranker, rerank_candidates=rerank_candidates,
        generation_cost_per_1k_in_usd=settings.generation_cost_per_1k_in_usd,
        generation_cost_per_1k_out_usd=settings.generation_cost_per_1k_out_usd,
        judge_cost_per_1k_in_usd=settings.judge_cost_per_1k_in_usd,
        judge_cost_per_1k_out_usd=settings.judge_cost_per_1k_out_usd,
        local_compute_cost_per_hour=settings.local_compute_cost_per_hour,
        chat_provider_name=settings.chat_provider, eval_judge_provider_name=settings.eval_judge_provider,
    )
    return cycle_id


async def _build_tried_configs(db, sweep_id: int, settings) -> list[TriedConfig]:
    sweep_configs = await db.list_sweep_configs(sweep_id)
    cycles = await db.list_eval_cycles_for_sweep(sweep_id)
    gate_thresholds = load_quality_gate(settings.eval_quality_gate_path)
    weights = load_scoring_weights(settings.config_scoring_weights_path)
    # rank_configs returns one (config_id, score) pair per *cycle* (a config
    # can have several, one per inner-grid combo), sorted descending by
    # score. Keep the first (highest) score seen per config_id rather than
    # `dict(...)`, which keeps the *last* -- i.e. the lowest -- score for a
    # repeated key.
    best_score_by_config: dict[int, float] = {}
    for config_id, score in rank_configs(cycles, gate_thresholds, weights):
        best_score_by_config.setdefault(config_id, score)

    tried = []
    for sweep_config in sweep_configs:
        config = await db.get_experiment_config(sweep_config["config_id"])
        config_dict = {
            "chunking_strategy": config.chunking_strategy, "chunk_size": config.chunk_size,
            "chunk_overlap": config.chunk_overlap, "embedding_provider": config.embedding_provider,
            "embedding_model": config.embedding_model,
        }
        tried.append(TriedConfig(
            config=config_dict, score=best_score_by_config.get(config.id), config_id=config.id,
        ))
    return tried


def _consecutive_non_improving_streak(tried: list[TriedConfig]) -> int:
    """How many of the most recently tried configs, in a row, failed to beat
    the best score seen before them. Configs with no score (gate-failed,
    or not yet evaluated) are skipped rather than counted either way, since
    they're not a meaningful comparison point."""
    best_so_far = None
    streak = 0
    for t in tried:
        if t.score is None:
            continue
        if best_so_far is None or t.score > best_so_far:
            best_so_far = t.score
            streak = 0
        else:
            streak += 1
    return streak


async def _propose_and_record_next(
    db, provider, settings, sweep_id: int, sweep_row: dict, user_hint: str | None = None,
) -> bool:
    """Returns True if the sweep converged (status set to awaiting_suggestion)."""
    tried = await _build_tried_configs(db, sweep_id, settings)

    if _consecutive_non_improving_streak(tried) >= CONVERGENCE_THRESHOLD:
        await db.update_sweep_status(sweep_id, "awaiting_suggestion")
        return True

    result = await propose_next_config(
        provider, _corpus_summary_text(settings), tried, sweep_row["ingestion_budget"], user_hint,
    )
    if result.converged:
        await db.update_sweep_status(sweep_id, "awaiting_suggestion")
        return True

    candidate = result.candidate
    embedding_dim = get_embedding_dim(candidate.embedding_provider, candidate.embedding_model)
    config = await db.create_experiment_config(
        chunking_strategy=candidate.chunking_strategy, chunk_size=candidate.chunk_size,
        chunk_overlap=candidate.chunk_overlap, embedding_provider=candidate.embedding_provider,
        embedding_model=candidate.embedding_model, embedding_dim=embedding_dim,
    )
    await db.create_sweep_config(sweep_id, config.id, candidate.rationale, "proposed", "loop")
    return False


async def finalize_sweep_run(db, provider, settings, sweep_id: int, stop_reason: str) -> None:
    tried = await _build_tried_configs(db, sweep_id, settings)
    passing = [t for t in tried if t.score is not None]
    best = max(passing, key=lambda t: t.score, default=None)
    summary = await generate_sweep_summary(
        provider, _corpus_summary_text(settings), tried, stop_reason, best,
    )
    best_config_id = best.config_id if best is not None else None
    await db.finalize_sweep(sweep_id, stop_reason, summary, best_config_id,
                             best.score if best else None)


async def run_sweep_loop(db, provider, chat_provider, judge, ner, settings, sweep_id: int) -> None:
    sweep_row = await db.get_sweep(sweep_id)

    while True:
        pending = await db.get_next_pending_sweep_config(sweep_id)
        if pending is None:
            return

        config = await _ensure_ingested_for_loop(db, ner, settings, pending, sweep_id)
        evaluated = set(pending["evaluated_combos"])

        for idx, combo in enumerate(INNER_GRID):
            if idx in evaluated:
                continue
            await _run_eval_combo(db, provider, chat_provider, judge, ner, settings, config,
                                   combo, sweep_id)
            evaluated.add(idx)
            await db.update_sweep_config_evaluated_combos(pending["id"], sorted(evaluated))

            if await check_eval_budget(db, sweep_id, sweep_row["eval_budget"]) or \
               await check_ingestion_budget(db, sweep_id, sweep_row["ingestion_budget"]):
                await finalize_sweep_run(db, provider, settings, sweep_id, "stopped_budget")
                return

        await db.update_sweep_config_status(pending["id"], "evaluated")

        converged = await _propose_and_record_next(db, provider, settings, sweep_id, sweep_row)
        if converged:
            return


async def apply_user_suggestion(
    db, provider, chat_provider, judge, ner, settings, sweep_id: int,
    hint: str | None, explicit_config: dict | None,
) -> None:
    if explicit_config is not None:
        config = await db.create_experiment_config(**explicit_config)
        await db.create_sweep_config(
            sweep_id, config.id, hint or "user-suggested config", "proposed", "user_suggested",
        )
    else:
        sweep_row = await db.get_sweep(sweep_id)
        converged = await _propose_and_record_next(
            db, provider, settings, sweep_id, sweep_row, user_hint=hint,
        )
        if converged:
            return

    await db.update_sweep_status(sweep_id, "running")
    await run_sweep_loop(db, provider, chat_provider, judge, ner, settings, sweep_id)


class CorpusChangedError(Exception):
    pass


async def resume_sweep(db, provider, chat_provider, judge, ner, settings, sweep_id: int) -> None:
    sweep_row = await db.get_sweep(sweep_id)
    current_files = list_pdf_filenames(settings.corpus_dir)
    if sorted(current_files) != sorted(sweep_row["corpus_snapshot"] or []):
        raise CorpusChangedError(
            f"Corpus files changed since this sweep's bootstrap: had "
            f"{sweep_row['corpus_snapshot']!r}, now {current_files!r}"
        )
    await db.update_sweep_status(sweep_id, "running")
    await run_sweep_loop(db, provider, chat_provider, judge, ner, settings, sweep_id)
