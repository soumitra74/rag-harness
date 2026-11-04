# Sweep budgets

A sweep has three budgets. Each is a set of caps, and a budget is exhausted when any one axis reaches its cap (`budget_exhausted` in `packages/backend/app/services/sweep_budget.py`).

| Budget | Axes | Checked | Spend comes from |
|---|---|---|---|
| Ingestion (`ingestion_budget`) | `cost_usd`, `minutes`, `iterations` | After each eval combo (`sweep.py`, `run_sweep_loop`) | `ingestion_runs` plus planning-stage `llm_usage` |
| Eval (`eval_budget`) | `cost_usd`, `minutes`, `iterations` | After each eval combo (same check) | `eval_cycles.aggregates.total_cost_usd` |
| Codegen (`codegen`) | `cost_usd`, `minutes`, `attempts` | Before each codegen attempt (`codegen.py`, `run_codegen_task`) | `codegen_runs` |

## Ingestion budget

Spend (`db.py`, `sum_ingestion_spend`):
- `cost_usd`: `ingestion_runs.cost_usd`, which is local embedding compute time times `LOCAL_COMPUTE_COST_PER_HOUR` (0 for non-local embedding providers), plus `llm_usage.cost_usd` for stages `inspect_corpus`, `golden_qa`, `propose_next_config`, `summary`.
- `minutes`: sum of `ingestion_runs.total_time_ms`.
- `iterations`: count of `ingestion_runs`.

LLM calls covered. These use the sweep provider, which is the eval judge provider and model, priced at `JUDGE_COST_PER_1M_*`:
- `inspect_corpus`: `corpus_inspector.py:64`
- `golden_qa`: `golden_qa_generator.py:58`
- `propose_next_config`: `sweep_next_config.py:54`
- `summary`: `sweep_summary.py:29`

The remaining budget is also passed into the `propose_next_config` prompt (`sweep_next_config.py:42`).

## Eval budget

Spend (`db.py`, `sum_eval_spend`):
- `cost_usd`: sum of `aggregates.total_cost_usd` across the sweep's eval cycles.
- `minutes`: sum of cycle wall time (`completed_at - started_at`).
- `iterations`: count of eval cycles.

Each cycle's `total_cost_usd` (`eval_run.py`, `_summarize`) is the sum of:
- Generation: the RAG answer call per question, priced at `GENERATION_COST_PER_1M_*`.
- Judge: the per-question judge calls (correctness, faithfulness, answer relevance, info recall, citation completeness), priced at `JUDGE_COST_PER_1M_*`.
- Local providers are priced by `LOCAL_COMPUTE_COST_PER_HOUR` times latency instead of tokens.

## Codegen budget

Spend (`db.py`, `sum_codegen_spend`):
- `cost_usd`: sum of `codegen_runs.cost_usd`.
- `minutes`: sum of attempt `duration_ms` / 60000.
- `attempts`: count of `codegen_runs`.

LLM calls covered: each generation attempt for the `ocr` and `graph_labels` tasks (`codegen.py:295`). Retries count against `attempts`. Codegen runs on `CODEGEN_PROVIDER` / `CODEGEN_MODEL`, falling back to the eval judge when unset, and is priced at `CODEGEN_COST_PER_1M_*` (`codegen.py`, `_codegen_cost`).

Codegen calls are not counted in the ingestion budget, even though they are logged to `llm_usage` under stage `codegen`.

## Codegen pricing

Both `codegen_runs.cost_usd` and the `llm_usage` row for stage `codegen` use `CODEGEN_COST_PER_1M_IN_USD` / `CODEGEN_COST_PER_1M_OUT_USD`, so the two records agree.
