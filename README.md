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
