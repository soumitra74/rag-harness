# Transaction Fraud RAG

RAG over a transaction-fraud corpus, used to find the best retrieval/generation
config. Separate from the FLAIR project: own Postgres (pgvector), own ports, own
volumes.

## Setup

1. `cp .env.example .env` and fill in provider keys.
2. Put the corpus files in `corpus/` (read-only mount into the backend).
3. `docker compose up --build`
4. Ingest with `./scripts/seed.sh` once the seed script is adapted to this corpus.

## Ports

| Service  | Host port |
|----------|-----------|
| Postgres | 25432     |
| Backend  | 8089      |

## Sweep budgets

Each sweep has two budgets, `ingestion_budget` and `eval_budget`, each with `minutes`, `cost_usd` and `iterations` caps. A sweep stops with `stopped_budget` when any cap is reached.

Change a sweep's budget without the frontend:

```
curl -X POST http://localhost:8089/api/sweep/1/budget \
  -H "Content-Type: application/json" \
  -d '{"eval_budget": {"minutes": 600}}'
```

Send only the axes you want to change; the others keep their current values. Values must be greater than 0. Both `ingestion_budget` and `eval_budget` are optional, and the response returns both budgets after the update.

A running sweep keeps the budget it loaded when its current config started, so a change takes effect at the next config boundary only after the sweep is restarted or resumed. To apply a new budget to a `stopped_budget` sweep, update it and then call `POST /api/sweep/{id}/resume`.
