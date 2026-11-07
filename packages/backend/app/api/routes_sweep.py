import asyncio
import logging
from pathlib import Path

from typing import Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from app.services.codegen import CODEGEN_AXES, reset_graph_labels
from app.services.golden_qa_generator import MIN_GOLDEN_QA_SIZE, generate_golden_qa
from app.services.judge import (
    create_judge, get_codegen_provider, get_completion_provider, get_golden_qa_provider,
    get_golden_qa_verifier_provider,
)
from app.services.sweep_log import MeteredProvider
from app.services.sweep import (
    INNER_GRID,
    apply_user_suggestion,
    finalize_sweep_run,
    list_pdf_filenames,
    resume_sweep,
    run_bootstrap,
    run_sweep_loop,
)

router = APIRouter()
logger = logging.getLogger(__name__)

BUDGET_AXES = ("cost_usd", "minutes", "iterations")


class BudgetInput(BaseModel):
    cost_usd: float
    minutes: float
    iterations: float


class BudgetPatch(BaseModel):
    cost_usd: float | None = None
    minutes: float | None = None
    iterations: float | None = None


class BudgetUpdateRequest(BaseModel):
    ingestion_budget: BudgetPatch | None = None
    eval_budget: BudgetPatch | None = None


class CodegenInput(BaseModel):
    cost_usd: float
    minutes: float
    attempts: int
    tasks: list[Literal["ocr", "graph_labels"]] = ["ocr", "graph_labels"]


class StartSweepRequest(BaseModel):
    ingestion_budget: BudgetInput
    eval_budget: BudgetInput
    generate_golden_qa: bool = False
    golden_qa_size: int = Field(default=MIN_GOLDEN_QA_SIZE, ge=MIN_GOLDEN_QA_SIZE)
    codegen: CodegenInput | None = None


class GoldenQAGenerateRequest(BaseModel):
    num_questions: int = Field(default=MIN_GOLDEN_QA_SIZE, ge=1)
    overwrite: bool = False


class SuggestRequest(BaseModel):
    hint: str | None = None
    explicit_config: dict | None = None
    accept: bool = False


def _validate_budget(budget, label: str, axes: tuple[str, ...] = BUDGET_AXES) -> None:
    for axis in axes:
        if getattr(budget, axis) <= 0:
            raise HTTPException(
                status_code=400,
                detail=f"{label}.{axis} must be greater than 0",
            )


async def _run_sweep_task(db, sweep_id: int, coro) -> None:
    """Every orchestrator entry point is scheduled via asyncio.create_task
    (fire-and-forget from the route's perspective), so an unhandled
    exception there would otherwise just vanish into the event loop's
    default exception handler, leaving the sweep stuck in whatever status
    it was in (bootstrapping/running) forever -- the only recovery being a
    backend restart, which merely flips it to paused, still with no error
    shown. Catches anything the orchestrator itself doesn't already handle
    and records it as a clear failure instead."""
    try:
        await coro
    except Exception as exc:
        await db.update_sweep_status(sweep_id, "failed", error_message=str(exc))


def _get_provider(request: Request):
    if request.app.state.sweep_provider is None:
        settings = request.app.state.settings
        request.app.state.sweep_provider = MeteredProvider(
            get_completion_provider(settings), request.app.state.db,
            settings.eval_judge_provider, settings.eval_judge_model,
            settings.judge_cost_per_1m_in_usd, settings.judge_cost_per_1m_out_usd,
        )
    return request.app.state.sweep_provider


def _get_judge(request: Request):
    if request.app.state.judge is None:
        request.app.state.judge = create_judge(request.app.state.settings, request.app.state.router)
    return request.app.state.judge


def _sweep_to_dict(sweep: dict) -> dict:
    return {
        "id": sweep["id"],
        "status": sweep["status"],
        "ingestion_budget": sweep["ingestion_budget"],
        "eval_budget": sweep["eval_budget"],
        "best_config_id": sweep["best_config_id"],
        "best_score": sweep["best_score"],
        "summary_markdown": sweep["summary_markdown"],
        "error_message": sweep["error_message"],
        "codegen_budget": sweep.get("codegen_budget"),
        "created_at": sweep["created_at"].isoformat(),
        "completed_at": sweep["completed_at"].isoformat() if sweep["completed_at"] else None,
    }


def _get_codegen_provider(request: Request):
    settings = request.app.state.settings
    return MeteredProvider(
        get_codegen_provider(settings), request.app.state.db,
        settings.codegen_provider or settings.eval_judge_provider,
        settings.codegen_model or settings.eval_judge_model,
        settings.codegen_cost_per_1m_in_usd, settings.codegen_cost_per_1m_out_usd,
    )


def _get_golden_qa_provider(request: Request):
    settings = request.app.state.settings
    return MeteredProvider(
        get_golden_qa_provider(settings), request.app.state.db,
        settings.golden_qa_provider or settings.eval_judge_provider,
        settings.golden_qa_model or settings.eval_judge_model,
        settings.judge_cost_per_1m_in_usd, settings.judge_cost_per_1m_out_usd,
    )


def _get_golden_qa_verifier_provider(request: Request):
    settings = request.app.state.settings
    return MeteredProvider(
        get_golden_qa_verifier_provider(settings), request.app.state.db,
        settings.golden_qa_verifier_provider or settings.eval_judge_provider,
        settings.golden_qa_verifier_model or settings.eval_judge_model,
        settings.golden_qa_verifier_cost_per_1m_in_usd, settings.golden_qa_verifier_cost_per_1m_out_usd,
    )


def _bootstrap_coroutine(request: Request, sweep: dict):
    options = sweep.get("bootstrap_options") or {}
    codegen_tasks = options.get("codegen_tasks")
    generate_golden_qa = options.get("generate_golden_qa", False)
    reset_graph_labels(request.app.state.ner)
    return run_bootstrap(
        request.app.state.db, _get_provider(request), request.app.state.ner,
        request.app.state.settings, sweep["id"], generate_golden_qa,
        codegen_tasks=codegen_tasks,
        codegen_budget=sweep.get("codegen_budget"),
        codegen_provider=_get_codegen_provider(request) if codegen_tasks else None,
        golden_qa_provider=_get_golden_qa_provider(request) if generate_golden_qa else None,
        golden_qa_verifier_provider=_get_golden_qa_verifier_provider(request) if generate_golden_qa else None,
        golden_qa_size=options.get("golden_qa_size", MIN_GOLDEN_QA_SIZE),
    )


async def _run_golden_qa_generation(provider, verifier, settings, num_questions: int) -> None:
    try:
        count = await generate_golden_qa(
            provider, settings.corpus_dir, settings.eval_dataset_path,
            num_questions=num_questions, verifier=verifier,
        )
        logger.info("Generated %d golden QA entries at %s", count, settings.eval_dataset_path)
    except Exception:
        logger.exception("Golden QA generation failed")


@router.post("/api/golden-qa/generate", status_code=202)
async def generate_golden_qa_route(request: Request, body: GoldenQAGenerateRequest):
    settings = request.app.state.settings
    if Path(settings.eval_dataset_path).exists() and not body.overwrite:
        raise HTTPException(
            status_code=409,
            detail=f"{settings.eval_dataset_path} already exists; pass overwrite=true to replace it",
        )
    asyncio.create_task(_run_golden_qa_generation(
        _get_golden_qa_provider(request), _get_golden_qa_verifier_provider(request),
        settings, body.num_questions,
    ))
    return {
        "status": "generating",
        "num_questions": body.num_questions,
        "path": settings.eval_dataset_path,
    }


@router.post("/api/sweep/start")
async def start_sweep(request: Request, body: StartSweepRequest):
    _validate_budget(body.ingestion_budget, "ingestion_budget")
    _validate_budget(body.eval_budget, "eval_budget")
    codegen_budget = None
    if body.codegen is not None:
        _validate_budget(body.codegen, "codegen", CODEGEN_AXES)
        codegen_budget = {axis: getattr(body.codegen, axis) for axis in CODEGEN_AXES}

    if request.app.state.sweep_in_progress:
        raise HTTPException(status_code=409, detail="A sweep is already in progress")
    request.app.state.sweep_in_progress = True

    db = request.app.state.db
    bootstrap_options = {
        "generate_golden_qa": body.generate_golden_qa,
        "golden_qa_size": body.golden_qa_size,
        "codegen_tasks": body.codegen.tasks if body.codegen else None,
    }
    sweep_id = await db.create_sweep(
        body.ingestion_budget.model_dump(), body.eval_budget.model_dump(),
        codegen_budget=codegen_budget, bootstrap_options=bootstrap_options,
    )
    bootstrap = _bootstrap_coroutine(request, {
        "id": sweep_id, "codegen_budget": codegen_budget, "bootstrap_options": bootstrap_options,
    })

    async def _run_and_release():
        try:
            await _run_sweep_task(db, sweep_id, bootstrap)
        finally:
            request.app.state.sweep_in_progress = False

    asyncio.create_task(_run_and_release())

    return {"sweep_id": sweep_id, "status": "bootstrapping"}


@router.get("/api/sweeps")
async def list_sweeps(request: Request):
    sweeps = await request.app.state.db.list_sweep_summaries()
    return [
        {"sweep_id": s["id"], "status": s["status"], "created_at": s["created_at"].isoformat()}
        for s in sweeps
    ]


@router.get("/api/sweep/latest")
async def get_latest_sweep(request: Request):
    latest = await request.app.state.db.get_latest_sweep_summary()
    if latest is None:
        return None
    return {
        "sweep_id": latest["id"],
        "status": latest["status"],
        "resumable": latest["status"] in ("paused", "failed"),
    }


@router.get("/api/sweep/{sweep_id}")
async def get_sweep(request: Request, sweep_id: int):
    db = request.app.state.db
    sweep = await db.get_sweep(sweep_id)
    if sweep is None:
        raise HTTPException(status_code=404, detail=f"Unknown sweep: {sweep_id}")

    sweep_configs = await db.list_sweep_configs(sweep_id)
    result = _sweep_to_dict(sweep)
    result["ingestion_spent"] = await db.sum_ingestion_spend(sweep_id)
    result["eval_spent"] = await db.sum_eval_spend(sweep_id)
    result["codegen_spent"] = await db.sum_codegen_spend(sweep_id)
    result["generated_tasks"] = sorted(await db.get_generated_artifacts(sweep_id))
    result["sweep_configs"] = [
        {
            "id": sc["id"], "config_id": sc["config_id"], "rationale": sc["rationale"],
            "status": sc["status"], "source": sc["source"],
        }
        for sc in sweep_configs
    ]
    return result


@router.post("/api/sweep/{sweep_id}/budget")
async def update_sweep_budget(request: Request, sweep_id: int, body: BudgetUpdateRequest):
    db = request.app.state.db
    sweep = await db.get_sweep(sweep_id)
    if sweep is None:
        raise HTTPException(status_code=404, detail=f"Unknown sweep: {sweep_id}")
    if body.ingestion_budget is None and body.eval_budget is None:
        raise HTTPException(status_code=400, detail="Provide ingestion_budget and/or eval_budget")

    patches = {
        "ingestion_budget": body.ingestion_budget.model_dump(exclude_none=True) if body.ingestion_budget else None,
        "eval_budget": body.eval_budget.model_dump(exclude_none=True) if body.eval_budget else None,
    }
    for label, patch in patches.items():
        for axis, value in (patch or {}).items():
            if value <= 0:
                raise HTTPException(status_code=400, detail=f"{label}.{axis} must be greater than 0")

    await db.update_sweep_budgets(sweep_id, patches["ingestion_budget"], patches["eval_budget"])
    updated = await db.get_sweep(sweep_id)
    return {
        "sweep_id": sweep_id,
        "ingestion_budget": updated["ingestion_budget"],
        "eval_budget": updated["eval_budget"],
    }


def _budget_usage(spent: dict, budget: dict) -> dict:
    return {
        "cost_used_usd": round(spent.get("cost_usd", 0.0), 4),
        "cost_cap_usd": budget.get("cost_usd"),
        "minutes_used": round(spent.get("minutes", 0.0), 1),
        "minutes_cap": budget.get("minutes"),
        "iterations_used": spent.get("iterations", 0),
        "iterations_cap": budget.get("iterations"),
    }


@router.get("/api/sweep/{sweep_id}/progress")
async def get_sweep_progress(request: Request, sweep_id: int):
    db = request.app.state.db
    sweep = await db.get_sweep(sweep_id)
    if sweep is None:
        raise HTTPException(status_code=404, detail=f"Unknown sweep: {sweep_id}")

    configs = await db.list_sweep_configs(sweep_id)
    in_progress = [c for c in configs if c["status"] != "evaluated"]
    current = in_progress[-1] if in_progress else None
    latest_cycle = await db.latest_sweep_eval_cycle(sweep_id)

    return {
        "sweep_id": sweep_id,
        "status": sweep["status"],
        "resumable": sweep["status"] in ("paused", "failed"),
        "configs_evaluated": sum(1 for c in configs if c["status"] == "evaluated"),
        "current_config": None if current is None else {
            "config_id": current["config_id"],
            "status": current["status"],
            "combos_done": len(current["evaluated_combos"]),
            "combos_total": len(INNER_GRID),
        },
        "latest_eval_cycle": None if latest_cycle is None else {
            "id": latest_cycle["id"],
            "config_id": latest_cycle["config_id"],
            "questions_written": latest_cycle["questions_written"],
            "questions_total": latest_cycle["total"],
            "completed": latest_cycle["completed_at"] is not None,
        },
        "budgets": {
            "ingestion": _budget_usage(
                await db.sum_ingestion_spend(sweep_id), sweep["ingestion_budget"],
            ),
            "eval": _budget_usage(await db.sum_eval_spend(sweep_id), sweep["eval_budget"]),
        },
    }


@router.post("/api/sweep/{sweep_id}/approve")
async def approve_sweep(request: Request, sweep_id: int):
    db = request.app.state.db
    settings = request.app.state.settings
    sweep = await db.get_sweep(sweep_id)
    if sweep is None:
        raise HTTPException(status_code=404, detail=f"Unknown sweep: {sweep_id}")
    if sweep["status"] != "awaiting_approval":
        raise HTTPException(
            status_code=409,
            detail=f"Sweep {sweep_id} is not awaiting approval (status: {sweep['status']!r})",
        )

    await db.update_sweep_status(sweep_id, "running")
    provider = _get_provider(request)
    asyncio.create_task(
        _run_sweep_task(
            db, sweep_id,
            run_sweep_loop(
                db, provider, request.app.state.chat_provider, _get_judge(request),
                request.app.state.ner, settings, sweep_id,
            ),
        )
    )
    return {"sweep_id": sweep_id, "status": "running"}


@router.post("/api/sweep/{sweep_id}/resume")
async def resume_sweep_route(request: Request, sweep_id: int):
    db = request.app.state.db
    settings = request.app.state.settings
    sweep = await db.get_sweep(sweep_id)
    if sweep is None:
        raise HTTPException(status_code=404, detail=f"Unknown sweep: {sweep_id}")
    if sweep["status"] not in ("paused", "failed"):
        raise HTTPException(
            status_code=409,
            detail=f"Sweep {sweep_id} is not paused or failed (status: {sweep['status']!r})",
        )

    if not sweep["bootstrap_complete"]:
        await db.update_sweep_status(sweep_id, "bootstrapping")
        asyncio.create_task(_run_sweep_task(db, sweep_id, _bootstrap_coroutine(request, sweep)))
        return {"sweep_id": sweep_id, "status": "bootstrapping"}

    current_files = list_pdf_filenames(settings.corpus_dir)
    if sorted(current_files) != sorted(sweep["corpus_snapshot"] or []):
        raise HTTPException(
            status_code=409,
            detail="Corpus files have changed since this sweep's bootstrap; cannot resume.",
        )

    # Set synchronously, before dispatching the background task: resume_sweep
    # itself only sets this after re-validating the corpus (redundant but
    # harmless), and a client polling GET /api/sweep/{id} right after this
    # response must never still see "paused" (review finding I-5).
    await db.update_sweep_status(sweep_id, "running")
    provider = _get_provider(request)
    asyncio.create_task(
        _run_sweep_task(
            db, sweep_id,
            resume_sweep(
                db, provider, request.app.state.chat_provider, _get_judge(request),
                request.app.state.ner, settings, sweep_id,
            ),
        )
    )
    return {"sweep_id": sweep_id, "status": "running"}


@router.post("/api/sweep/{sweep_id}/suggest")
async def suggest_sweep(request: Request, sweep_id: int, body: SuggestRequest):
    db = request.app.state.db
    settings = request.app.state.settings
    sweep = await db.get_sweep(sweep_id)
    if sweep is None:
        raise HTTPException(status_code=404, detail=f"Unknown sweep: {sweep_id}")
    if sweep["status"] != "awaiting_suggestion":
        raise HTTPException(
            status_code=409,
            detail=f"Sweep {sweep_id} is not awaiting a suggestion (status: {sweep['status']!r})",
        )

    provider = _get_provider(request)

    # Set synchronously, before dispatching either path's background task:
    # finalize_sweep_run's real terminal status (stopped_optimal) and
    # apply_user_suggestion's own "running" write both only happen deep
    # inside an LLM round-trip, so a client polling GET /api/sweep/{id}
    # right after this response must never still see "awaiting_suggestion"
    # -- that's exactly what bounces the wizard back to this page (review
    # finding I-5). The response always reports "running" since that's
    # what's true the instant this returns; the real terminal status lands
    # once the background task finishes, same as every other sweep action.
    await db.update_sweep_status(sweep_id, "running")

    if body.accept and not body.hint and not body.explicit_config:
        asyncio.create_task(
            _run_sweep_task(
                db, sweep_id,
                finalize_sweep_run(db, provider, settings, sweep_id, "stopped_optimal"),
            )
        )
        return {"sweep_id": sweep_id, "status": "running"}

    asyncio.create_task(
        _run_sweep_task(
            db, sweep_id,
            apply_user_suggestion(
                db, provider, request.app.state.chat_provider, _get_judge(request),
                request.app.state.ner, settings, sweep_id, body.hint, body.explicit_config,
            ),
        )
    )
    return {"sweep_id": sweep_id, "status": "running"}


@router.post("/api/sweep/{sweep_id}/cancel")
async def cancel_sweep(request: Request, sweep_id: int):
    db = request.app.state.db
    sweep = await db.get_sweep(sweep_id)
    if sweep is None:
        raise HTTPException(status_code=404, detail=f"Unknown sweep: {sweep_id}")
    # Cooperative cancellation of an in-flight background loop isn't wired
    # up (run_sweep_loop doesn't check for this status mid-iteration) --
    # this stops the sweep from being resumed/approved further, but a
    # currently-running loop will finish its current step before anything
    # would notice. Acceptable for now: not one of this task's tested
    # scenarios, and cancel is rare compared to budget/convergence stops.
    await db.update_sweep_status(sweep_id, "cancelled")
    return {"sweep_id": sweep_id, "status": "cancelled"}
