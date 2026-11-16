# RAG Harness

A harness for choosing the retrieval and generation setup that works best for a
document corpus. You point it at a folder of PDFs. It generates a golden
question-and-answer set from the documents, then sweeps chunking, embedding and
retrieval configurations, scoring each one against that set under a time and
cost budget. The result is a ranked comparison of configurations for your data.

## What it does

- **Golden set generation.** Writes grounded questions with reference answers and source pages, and verifies each one against the page text before keeping it.
- **Config sweeps.** Starts from a set of bootstrap configurations, then proposes new ones (chunker and embedder changes) and keeps the best scoring result.
- **Budgets.** Each sweep has ingestion and evaluation caps on minutes, cost and iterations. A sweep stops with `stopped_budget` when a cap is reached.
- **Codegen.** Can generate small OCR and graph-label helpers for the corpus, run them in an isolated sandbox service, and store accepted output in `packages/backend/generated/`.
- **Local or hosted models.** Chat, embedding and judge models can come from Anthropic, OpenAI, or Ollama (through the local load balancer below).

## Architecture

The system is described with Mermaid diagrams in [architecture-diagrams.md](architecture-diagrams.md):

- System architecture and Docker Compose services
- Ingestion, chat retrieval, and eval cycle
- Sweep loop and sweep orchestration
- RCA agent, model routing, and the Ollama load balancer

## Requirements

- Docker and Docker Compose
- Python 3.12 (the backend image) if you run the backend outside Docker; the Ollama load balancer needs Python with Flask and requests
- Node.js 20+ if you run the frontend outside Docker
- API keys for any hosted providers you enable, or a running Ollama instance for local models

## Quick start

1. Copy the environment template and fill in provider keys:

   ```
   cp .env.example .env
   ```

2. Put your PDF corpus in `corpus/`. The backend mounts it read-only.

3. Start the stack:

   ```
   docker compose up --build
   ```

4. Open the frontend at http://localhost:5183 and start a sweep from the setup wizard.

5. To upload the corpus from the command line instead, run `./scripts/seed.sh`. Set `API_BASE_URL=http://localhost:8089` first, because the script's default points at a different port.

## Non-standard ports

This project deliberately avoids the default ports of common services so it can run next to other local stacks. If you already run a service on the default port, nothing collides here.

| Service                     | Default port | This project | Notes                                              |
|-----------------------------|--------------|--------------|----------------------------------------------------|
| Postgres (pgvector)         | 5432         | **25432**    | `POSTGRES_HOST_PORT` in `.env`                     |
| Backend API                 | 8000         | **8089**     | `BACKEND_HOST_PORT`; Swagger at `/docs`            |
| Frontend (Vite dev server)  | 5173         | **5183**     | `FRONTEND_HOST_PORT`                               |
| Ollama load balancer        | 11434        | **11444**    | Runs on the host; the backend calls this port      |
| Ollama nodes (behind LB)    | 11434        | 11434        | Each Ollama server keeps its default port          |
| Codegen sandbox             | —            | internal     | Not published to the host                          |

If you change a host port, update the matching URL in `.env` (for example `OLLAMA_BASE_URL`) or in the frontend (`VITE_API_BASE_URL`).

## Local Ollama load balancer

`ollama_load_balancer.py` is a small Flask service that sits in front of several Ollama servers and presents them as one endpoint on port 11444. Use it when you have more than one machine with a GPU, or want to spread embedding work across them.

- **Routing.** Each request goes to a free node. Nodes where the model fits in VRAM are preferred, then nodes where it fits VRAM plus RAM, then the rest. Within a tier the least-loaded node wins.
- **Queueing.** When every node is at its concurrency limit, requests wait in an in-memory queue. A request that waits too long gets a 503.
- **Retries.** Upstream failures are requeued, up to a fixed limit.
- **Model pins.** `MODEL_ROUTES` can restrict a model to specific nodes.
- **Dashboard.** Served at http://localhost:11444/dashboard.

Configure your nodes in the `NODES` table at the top of the file. Replace the placeholder hostnames and GPU details with your own. Then:

```
pip install flask requests
python ollama_load_balancer.py
```

Set `OLLAMA_BASE_URL` in `.env` to `http://localhost:11444` for a backend running on the host, or `http://host.docker.internal:11444` for the backend in Docker (already set in `docker-compose.yml`). Set `OLLAMA_EMBEDDING_CONCURRENCY` to the number of nodes behind the balancer.

Model sizes come from each node's `/api/tags` listing and are on-disk sizes, so VRAM fit is an estimate.

## Sweep budgets

Each sweep has two budgets, `ingestion_budget` and `eval_budget`, each with `minutes`, `cost_usd` and `iterations` caps. A sweep stops with `stopped_budget` when any cap is reached.

Change a sweep's budget through the API:

```
curl -X POST http://localhost:8089/api/sweep/1/budget \
  -H "Content-Type: application/json" \
  -d '{"eval_budget": {"minutes": 600}}'
```

Send only the axes you want to change; the others keep their current values. Values must be greater than 0. Both `ingestion_budget` and `eval_budget` are optional, and the response returns both budgets after the update.

A running sweep keeps the budget it loaded when its current config started, so a change takes effect at the next config boundary only after the sweep is restarted or resumed. To apply a new budget to a `stopped_budget` sweep, update it and then call `POST /api/sweep/{id}/resume`.

## Configuration

All settings live in `.env` (see `.env.example`). The main groups are:

- **Providers**: `CHAT_PROVIDER`, `EMBEDDING_PROVIDER`, the Anthropic, OpenAI and Ollama settings, and `LOCAL_EMBEDDING_MODEL`.
- **Judge and codegen models**: `EVAL_JUDGE_PROVIDER`, `EVAL_JUDGE_MODEL`, `CODEGEN_PROVIDER`, `CODEGEN_MODEL`.
- **Cost tracking**: per-million-token prices used for budget caps, and an hourly rate for local compute.
- **Database**: `POSTGRES_*` and `DATABASE_URL`.

The database credentials in `.env.example` are local development defaults. Change them if the port is reachable from a network you don't control.

## Development

- Backend tests: `pytest` from `packages/backend` (install from `requirements-dev.txt`).
- Frontend: `npm install`, then `npm run dev`. See `packages/frontend/README.md`.
- Eval judge tests: `pytest` from `eval/`.

## Status

Current version: **0.8**. Early stage. Interfaces and config names may change between commits.

Version history is tagged in git (`0.1` to `0.8`); see the milestone list in [timeline_and_learning.md](timeline_and_learning.md).

## License

Apache License 2.0. See [LICENSE](LICENSE).
