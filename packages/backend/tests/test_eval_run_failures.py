import pytest

import app.services.eval_run as eval_run
from app.db import ExperimentConfig
from app.services.eval_run import new_eval_job

CONFIG = ExperimentConfig(
    id=1, chunking_strategy="sliding_window", chunk_size=1000, chunk_overlap=200,
    embedding_provider="local", embedding_model="BAAI/bge-base-en-v1.5", embedding_dim=768,
)
ITEM = {
    "id": "q1", "question": "what happened?", "reference_answer": "reference",
    "golden_pages": [{"document": "doc.pdf", "page": 3}], "query_type": "extractive",
}


class FailureDatabase:
    def __init__(self):
        self.recorded = []
        self.completed = None

    async def find_or_create_eval_query(self, question, reference_answer, query_type):
        return 7

    async def ensure_gold_pages(self, query_id, pages):
        self.gold_pages = pages

    async def record_eval_run(self, config_id, query_id, generated_answer, hit_rate, mrr,
                               citation_accuracy, correctness, generation_latency_ms,
                               eval_latency_ms, **kwargs):
        self.recorded.append({"query_id": query_id, "error": kwargs.get("error"),
                              "cycle_id": kwargs.get("cycle_id")})

    async def complete_eval_cycle(self, cycle_id, status, aggregates):
        self.completed = (cycle_id, status, aggregates)


@pytest.mark.asyncio
async def test_failed_question_is_stored_with_its_error_and_cycle_still_completes(monkeypatch):
    async def failing_answer_question(*args, **kwargs):
        raise TimeoutError()

    monkeypatch.setattr(eval_run, "answer_question", failing_answer_question)
    db = FailureDatabase()
    job = new_eval_job(config_id=1, total=1)

    await eval_run.run_eval_job(
        job, db, CONFIG, None, None, object(), [ITEM], 5, 100, 1,
    )

    [result] = job.per_query
    assert result.error == "TimeoutError: "
    assert db.recorded == [{"query_id": 7, "error": "TimeoutError: ", "cycle_id": 100}]
    assert db.gold_pages == [("doc.pdf", 3)]
    assert db.completed[:2] == (100, "completed")
    assert job.aggregates["num_queries"] == 1
