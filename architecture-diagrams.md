# RAG Test Harness: Technical Diagrams

Mermaid diagrams for the talk and the repo. Source of truth: `docker-compose.yml`, `packages/backend/app/main.py`, `services/chat.py`, `services/eval_run.py`, `services/sweep.py`, and `docs/features.md`.

## 1. System architecture

```mermaid
flowchart LR
    user(["Researcher"]) --> fe["React / TypeScript frontend<br/>Vite, port 5183"]
    fe -->|"REST /api"| be["FastAPI backend<br/>port 8088"]
    cc(["Claude Code"]) -->|"stdio MCP"| mcp["MCP server<br/>packages/mcp-server"]

    be --> pg[("Postgres 16 + pgvector<br/>port 15432")]
    mcp -->|"rca_readonly role"| pg
    be -->|"rca_readonly role<br/>RCA SQL only"| pg

    be --> lb["Ollama load balancer<br/>port 11444"]
    lb --> ollama["Ollama nodes"]

    be -.->|"generator / judge"| llm["Anthropic / OpenAI APIs"]
    be -.->|"RCA supervisor"| llm
    be --> local["Local models on CPU<br/>sentence-transformers embedders,<br/>bge rerankers, GLiNER NER"]

    corpus[("corpus/ PDFs<br/>read-only mount")] --> be
    golden[("eval/golden_qa.json<br/>+ gate and scoring config")] --> be
```

## 2. Docker Compose services

```mermaid
flowchart TB
    subgraph host["Host machine"]
        p1["localhost:5183"]
        p2["localhost:8088"]
        p3["localhost:15432"]
        p4["localhost:11444"]
        ollama_host["Ollama on host"]
    end

    subgraph compose["docker compose network"]
        fe["frontend<br/>VITE_API_BASE_URL=localhost:8088"]
        be["backend<br/>DATABASE_URL, RCA_READONLY_DATABASE_URL<br/>OLLAMA_BASE_URL=host.docker.internal:11444"]
        db[("postgres<br/>pgvector/pgvector:pg16<br/>volume: pgdata")]
        vols["volumes: rca_logs, backups,<br/>corpus (ro), eval config (ro)"]
    end

    p1 --> fe
    p2 --> be
    p3 --> db
    fe -->|"depends_on"| be
    be -->|"depends_on, healthy"| db
    be --> vols
    be -->|"host.docker.internal"| p4
    p4 --> ollama_host
```

## 3. Ingestion

```mermaid
flowchart LR
    pdf["PDF in corpus/"] --> parse["Parse pages"]
    parse --> chunk{"Chunking strategy<br/>from experiment config"}
    chunk -->|"sliding_window"| c1["Fixed chars + overlap"]
    chunk -->|"fixed_token"| c2["tiktoken token count"]
    chunk -->|"recursive_semantic"| c3["Coarsest boundary that fits"]
    chunk -->|"document_structure_aware"| c4["Section and paragraph boundaries"]
    chunk -->|"sentence_window"| c5["Embed sentence,<br/>store context_text window"]
    c1 --> emb
    c2 --> emb
    c3 --> emb
    c4 --> emb
    c5 --> emb
    emb["Embedder<br/>local / openai / ollama<br/>dim derived from model"] --> store[("chunks_dim_N table<br/>row carries config_id")]
    store --> run["ingestion_run row<br/>chunks, pages, time, errors, size"]
    store -.->|"optional, phase 1"| ner["GLiNER entities<br/>+ CO_OCCURS_WITH edges"]
```

## 4. Chat retrieval pipeline

```mermaid
flowchart TD
    q["Question"] --> qe["Embed question<br/>(config's embedder)"]
    qe --> vs["Vector search<br/>over-fetch 3x top_k, cap 30"]
    vs --> gq{"Graph augmentation<br/>enabled?"}
    gq -->|"yes"| ge["Entities in question<br/>pull linked chunks"]
    gq -->|"no"| merge
    ge --> merge["Merge candidates"]
    merge --> rr{"Reranker set?"}
    rr -->|"yes: bge-reranker-v2-m3 or base<br/>score top rerank_candidates"| dd
    rr -->|"no"| dd["Drop near-duplicates<br/>text similarity >= 0.9"]
    dd --> topk["Keep top_k"]
    topk --> ctx["Label sources<br/>[Source N -- doc, page P]"]
    ctx --> gen["Generator<br/>anthropic / openai / ollama<br/>must cite [Source N]"]
    gen --> ans["Answer + sources<br/>(doc, page, snippet, chunk id)"]
```

## 5. Eval cycle

```mermaid
sequenceDiagram
    autonumber
    participant UI as Eval tab
    participant API as routes_eval
    participant Job as run_eval_job
    participant DB as Postgres
    participant R as Retriever + reranker
    participant G as Generator LLM
    participant J as Judge LLM

    UI->>API: POST /api/eval/run (config, top_k, graph, reranker)
    API->>API: 409 if running >= eval_cycle_concurrency
    API->>DB: create eval cycle
    API-->>UI: job_id
    loop every golden question (10)
        Job->>R: embed, vector search, graph, rerank
        R-->>Job: candidates and top_k pages
        Job->>DB: deterministic metrics<br/>hit, mrr, citation P/R, recall@k, nDCG
        Job->>G: generate answer from top_k context
        G-->>Job: answer with citations
        Job->>J: correctness, faithfulness,<br/>relevance, info recall, citation completeness
        J-->>Job: scores + reasoning
        Job->>DB: save per-question row (or error)
    end
    Job->>DB: cycle summary = mean per metric
    UI->>API: GET /api/eval/run/{job_id} (poll every 2s)
    API-->>UI: status, progress, result
```

## 6. Metrics map

```mermaid
flowchart LR
    subgraph det["Deterministic, per question"]
        hit["hit"]
        mrr["mrr"]
        cp["citation_accuracy<br/>(precision)"]
        cr["citation_recall"]
        rk["recall@k<br/>k = 5, 10, 20, 50"]
        nd["ndcg@10"]
        rrk["rerank_recall@k,<br/>rerank_ndcg@10"]
    end
    subgraph llm["LLM judge, per question"]
        cor["correctness"]
        fai["faithfulness"]
        rel["answer_relevance"]
        ir["info_recall"]
        cc["citation_completeness"]
    end
    subgraph cost["Cost and latency"]
        lat["gen, judge, rerank latency"]
        tok["tokens, LLM call count"]
    end
    det --> mean["Cycle aggregates<br/>mean over questions"]
    llm --> mean
    cost --> mean
    mean --> cmp["Compare Evals<br/>side-by-side table, CSV"]
    rk --> json[("retrieval_metrics JSON column<br/>no schema change to add metrics")]
    rrk --> json
    nd --> json
```

## 7. Sweep loop

```mermaid
stateDiagram-v2
    [*] --> bootstrap: start sweep
    bootstrap --> pending: first config proposed
    pending --> ingesting: next pending config
    ingesting --> evaluating: ingest + graph backfill
    evaluating --> evaluating: next inner-grid combo<br/>(top_k, graph, reranker)
    evaluating --> evaluated: grid done
    evaluated --> pending: propose next config
    evaluated --> awaiting_suggestion: converged (LLM or<br/>non-improving streak)
    evaluating --> stopped_budget: eval or ingestion budget hit
    awaiting_suggestion --> pending: user hint or explicit config
    awaiting_suggestion --> finalized: finalize
    stopped_budget --> finalized: summary + best config
    finalized --> [*]
```

## 8. Sweep orchestration

```mermaid
flowchart TD
    wiz["Sweep Wizard UI"] -->|"POST /api/sweeps"| sr["routes_sweep"]
    sr --> boot["run_bootstrap<br/>corpus summary, golden QA check"]
    boot --> loop["run_sweep_loop"]
    loop --> ing["_ensure_ingested_for_loop"]
    ing --> combo["_run_eval_combo<br/>over INNER_GRID"]
    combo --> budget{"check_eval_budget /<br/>check_ingestion_budget"}
    budget -->|"over"| fin["finalize_sweep_run"]
    budget -->|"ok"| tried["_build_tried_configs<br/>scores via quality gate + scoring weights"]
    tried --> conv{"Streak >= CONVERGENCE_THRESHOLD<br/>or LLM converged?"}
    conv -->|"no"| prop["propose_next_config<br/>LLM proposes next config"]
    prop --> loop
    conv -->|"yes"| await_["status: awaiting_suggestion"]
    fin --> summ["generate_sweep_summary<br/>best config, stop reason"]
    summ --> done["status: finalized"]
    await_ --> human["User accepts, hints, or stops"]
    human --> loop
    restart["Backend restart"] -.->|"pause_interrupted_sweeps<br/>abort_running_eval_cycles"| loop
```

## 9. RCA agent

```mermaid
sequenceDiagram
    autonumber
    actor U as Researcher
    participant FE as RCA tab
    participant API as routes_rca
    participant A as rca_agent.run_turn
    participant S as Supervisor LLM<br/>(Anthropic tool use)
    participant T as Deterministic tools
    participant DB as Postgres<br/>(rca_readonly role)

    U->>FE: pick cycle + question, ask
    FE->>API: POST /api/rca/sessions/{id}/messages
    API->>A: history + new message
    loop up to 8 tool calls per turn
        A->>S: messages + tool schemas
        S-->>A: tool_use request
        A->>T: dispatch
        T->>DB: get_eval_run, resolve_chunks,<br/>get_gold_pages, run_readonly_sql (SELECT only)
        T->>T: check_text_contains,<br/>get_backend_logs (log file)
        T-->>A: summarized result
    end
    S-->>A: final answer (Markdown)
    A-->>API: reply + token usage
    API-->>FE: render Markdown, store in thread
```

## 10. Model routing (opt-in)

```mermaid
flowchart LR
    call["Generation or judge call"] --> router{"DynamicRouter<br/>ROUTER_TIERS set?"}
    router -->|"no"| direct["Configured provider<br/>CHAT_PROVIDER / EVAL_JUDGE_PROVIDER"]
    router -->|"yes"| task["Task to tier map<br/>ROUTER_TASK_TIERS"]
    task --> tier["Selected tier<br/>model, base URL, temperature, cost"]
    tier --> call2["Call model"]
    call2 -->|"cheap check fails<br/>or tier errors"| fb["Fallback tier<br/>ROUTER_FALLBACK_TIER"]
    call2 -->|"ok"| log["INFO log line:<br/>task, tier, model, reason,<br/>latency, tokens, cost"]
    fb --> log
```

## 11. Ollama load balancer

```mermaid
flowchart LR
    client["Backend or any Ollama client"] --> lb["Ollama load balancer<br/>port 11444"]
    lb --> q{"Free slot on a node<br/>for this model?"}
    q -->|"yes"| node["Ollama node"]
    q -->|"no"| wait["Wait queue<br/>600 s timeout"]
    wait --> q
    node -->|"5xx or network error"| requeue["Mark unhealthy,<br/>requeue, up to 10 attempts"]
    requeue --> q
    node -->|"4xx"| back["Return to caller<br/>node stays healthy"]
    node -->|"ok"| back2["Return to caller"]
    health["Health check every 30 s"] -.-> node
    routes["MODEL_ROUTES pins"] -.-> q
    dash["/dashboard"] -.-> lb
```
