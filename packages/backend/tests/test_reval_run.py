import asyncio

import pytest

import app.services.reval_run as reval_run
from app.db import ExperimentConfig
from app.services.eval_run import EvalQueryResult, new_eval_job
from app.services.judge import CombinedJudgeResult, JudgeResult
from app.services.reval_run import (
    LLM_CALLS_PER_QUESTION_COMBINED,
    LLM_CALLS_PER_QUESTION_SEPARATE,
    run_reval_job,
)

TEST_CONFIG = ExperimentConfig(
    id=1,
    chunking_strategy="sliding_window",
    chunk_size=1000,
    chunk_overlap=200,
    embedding_provider="local",
    embedding_model="BAAI/bge-base-en-v1.5",
    embedding_dim=768,
)

CHUNK_ID = "11111111-1111-1111-1111-111111111111"
MISSING_CHUNK_ID = "22222222-2222-2222-2222-222222222222"
CHUNK = {"chunk_text": "chunk one", "context_text": None, "page_number": 3, "filename": "doc.pdf"}


def _item(item_id, question):
    return {"id": item_id, "question": question, "reference_answer": "reference",
            "golden_pages": [], "query_type": "extractive"}


def _row(query_id, question, chunk_ids=(CHUNK_ID,), error=None):
    return {
        "query_id": query_id, "question": question, "reference_answer": "reference",
        "generated_answer": f"stored answer for {question}", "hit_rate": True, "mrr": 1.0,
        "citation_accuracy": 1.0, "citation_recall": 1.0,
        "retrieval_metrics": {"recall@5": 1.0, "ndcg@10": 1.0},
        "retrieved_chunk_ids": list(chunk_ids), "generation_latency_ms": 5000,
        "generation_input_tokens": 100, "generation_output_tokens": 20, "error": error,
    }


class FakeJudge:
    """Exposes both the combined and separate-call interfaces so the same
    fake can back either combine_calls value."""

    async def score_reference_based(self, question, reference_answer, generated_answer,
                                     source_labels):
        return CombinedJudgeResult(
            correctness=0.9, correctness_reasoning="correct",
            answer_relevance=0.7, answer_relevance_reasoning="relevant",
            citation_completeness=0.5, citation_completeness_reasoning="one claim uncited",
            input_tokens=10, output_tokens=5,
        )

    async def score_correctness(self, question, reference_answer, generated_answer):
        return JudgeResult(score=0.9, reasoning="correct", input_tokens=3, output_tokens=2)

    async def score_answer_relevance(self, question, generated_answer):
        return JudgeResult(score=0.7, reasoning="relevant", input_tokens=3, output_tokens=2)

    async def score_citation_completeness(self, question, source_labels, generated_answer):
        return JudgeResult(
            score=0.5, reasoning="one claim uncited", input_tokens=4, output_tokens=1
        )

    async def score_faithfulness(self, question, context_chunks, generated_answer):
        return JudgeResult(score=0.8, reasoning="grounded", input_tokens=11, output_tokens=6)

    async def score_info_recall(self, reference_answer, context_chunks):
        return JudgeResult(score=0.6, reasoning="covers most", input_tokens=13, output_tokens=8)


class FakeDatabase:
    def __init__(self, source_rows, chunk_map=None):
        self._source_rows = source_rows
        self._chunk_map = {CHUNK_ID: CHUNK} if chunk_map is None else chunk_map
        self._query_ids = {row["question"]: row["query_id"] for row in source_rows}
        self.recorded_calls: list[dict] = []
        self.completed_cycle: tuple | None = None

    async def get_cycle_reval_source(self, cycle_id):
        return self._source_rows

    async def find_or_create_eval_query(self, question, reference_answer, query_type):
        return self._query_ids.get(question, 1000 + len(self._query_ids))

    async def resolve_chunks(self, chunk_ids, dim):
        return {cid: self._chunk_map[cid] for cid in chunk_ids if cid in self._chunk_map}

    async def record_eval_run(self, config_id, query_id, generated_answer, hit_rate, mrr,
                               citation_accuracy, correctness, generation_latency_ms,
                               eval_latency_ms, **kwargs):
        self.recorded_calls.append({
            "config_id": config_id, "query_id": query_id, "generated_answer": generated_answer,
            "correctness": correctness, **kwargs,
        })

    async def complete_eval_cycle(self, cycle_id, status, aggregates):
        self.completed_cycle = (cycle_id, status, aggregates)


@pytest.fixture
def regeneration(monkeypatch):
    """Stands in for run_eval_job on the regeneration path and records what it was asked to answer."""
    calls = []

    async def fake_run_eval_job(job, db, config, embedding_provider, chat_provider, judge,
                                dataset, top_k, cycle_id, judge_concurrency, *args, **kwargs):
        calls.append({"items": [item["id"] for item in dataset], "cycle_id": cycle_id,
                      "finalize": kwargs.get("finalize", True)})
        for item in dataset:
            job.per_query.append(EvalQueryResult(
                id=item["id"], question=item["question"], hit=False, mrr=0.0,
                citation_accuracy=0.0, citation_recall=0.0, retrieval_metrics={},
                correctness=0.8, correctness_reasoning="regenerated",
                generated_answer="fresh answer", generation_latency_ms=10, eval_latency_ms=5,
            ))

    monkeypatch.setattr(reval_run, "run_eval_job", fake_run_eval_job)
    return calls


async def _run(db, dataset, judge=None, **kwargs):
    job = new_eval_job(config_id=1, total=len(dataset))
    await run_reval_job(
        job, db, TEST_CONFIG, judge or FakeJudge(), source_cycle_id=99, cycle_id=100,
        judge_concurrency=2, dataset=dataset, top_k=5, embedding_provider=None,
        chat_provider=None, ner=None, reranker=None, rerank_candidates=None,
        graph_augmented=False, recall_k_values="5,10", **kwargs,
    )
    return job


@pytest.mark.asyncio
async def test_run_reval_job_rejudges_stored_answer_with_reconstructed_context(regeneration):
    db = FakeDatabase([_row(42, "what happened?")])

    job = await _run(db, [_item("q1", "what happened?")])

    assert job.status == "completed"
    assert regeneration == []
    [result] = job.per_query
    assert result.error is None
    assert result.hit is True
    assert result.mrr == 1.0
    assert result.citation_recall == 1.0
    assert result.retrieval_metrics == {"recall@5": 1.0, "ndcg@10": 1.0}
    assert result.generated_answer == "stored answer for what happened?"
    assert result.generation_latency_ms == 5000
    assert result.generation_input_tokens == 100
    assert result.correctness == 0.9
    assert result.faithfulness == 0.8
    assert result.answer_relevance == 0.7
    assert result.info_recall == 0.6
    assert result.citation_completeness == 0.5
    assert result.llm_call_count == LLM_CALLS_PER_QUESTION_COMBINED == 3

    [recorded] = db.recorded_calls
    assert recorded["config_id"] == 1
    assert recorded["query_id"] == 42
    assert recorded["cycle_id"] == 100
    assert recorded["faithfulness"] == 0.8
    assert recorded["retrieved_chunk_ids"] == [CHUNK_ID]

    assert job.aggregates["num_queries"] == 1
    assert job.aggregates["mean_correctness"] == 0.9
    assert db.completed_cycle == (100, "completed", job.aggregates)


@pytest.mark.asyncio
async def test_run_reval_job_with_combine_calls_false_scores_separately(regeneration):
    db = FakeDatabase([_row(42, "what happened?")])

    job = await _run(db, [_item("q1", "what happened?")], combine_calls=False)

    [result] = job.per_query
    assert result.correctness == 0.9
    assert result.answer_relevance == 0.7
    assert result.citation_completeness == 0.5
    assert result.faithfulness == 0.8
    assert result.info_recall == 0.6
    assert result.llm_call_count == LLM_CALLS_PER_QUESTION_SEPARATE == 5

    [recorded] = db.recorded_calls
    assert recorded["answer_relevance"] == 0.7
    assert recorded["llm_call_count"] == 5


@pytest.mark.asyncio
async def test_run_reval_job_regenerates_questions_whose_chunks_were_reingested(regeneration):
    db = FakeDatabase([_row(43, "what happened?", chunk_ids=(CHUNK_ID, MISSING_CHUNK_ID))])

    job = await _run(db, [_item("q1", "what happened?")])

    assert regeneration == [{"items": ["q1"], "cycle_id": 100, "finalize": False}]
    [result] = job.per_query
    assert result.error is None
    assert result.generated_answer == "fresh answer"
    assert db.recorded_calls == []  # regeneration writes its own rows via run_eval_job


@pytest.mark.asyncio
async def test_run_reval_job_regenerates_questions_that_failed_in_source_run(regeneration):
    db = FakeDatabase([
        _row(1, "first question"),
        _row(2, "second question", error="TimeoutError: boom"),
    ], )
    dataset = [_item("q1", "first question"), _item("q2", "second question")]

    job = await _run(db, dataset)

    assert regeneration == [{"items": ["q2"], "cycle_id": 100, "finalize": False}]
    assert [r.id for r in job.per_query] == ["1", "q2"]
    assert job.per_query[0].generated_answer == "stored answer for first question"
    assert job.per_query[1].generated_answer == "fresh answer"
    assert job.aggregates["num_queries"] == 2
    assert [c["query_id"] for c in db.recorded_calls] == [1]


@pytest.mark.asyncio
async def test_run_reval_job_covers_every_dataset_question_when_no_source_rows(regeneration):
    db = FakeDatabase([])
    dataset = [_item("q1", "one"), _item("q2", "two"), _item("q3", "three")]

    job = await _run(db, dataset)

    assert regeneration == [{"items": ["q1", "q2", "q3"], "cycle_id": 100, "finalize": False}]
    assert len(job.per_query) == 3
    assert job.aggregates["num_queries"] == 3
    assert db.completed_cycle[1] == "completed"


@pytest.mark.asyncio
async def test_run_reval_job_bounds_judge_concurrency_to_worker_count(regeneration):
    class ConcurrencyTrackingJudge:
        CALL_DELAY = 0.05

        def __init__(self):
            self.in_flight = 0
            self.peak_in_flight = 0

        async def _tick(self, score, reasoning):
            self.in_flight += 1
            self.peak_in_flight = max(self.peak_in_flight, self.in_flight)
            await asyncio.sleep(self.CALL_DELAY)
            self.in_flight -= 1
            return JudgeResult(score=score, reasoning=reasoning)

        async def score_reference_based(self, question, reference_answer, generated_answer,
                                         source_labels):
            self.in_flight += 1
            self.peak_in_flight = max(self.peak_in_flight, self.in_flight)
            await asyncio.sleep(self.CALL_DELAY)
            self.in_flight -= 1
            return CombinedJudgeResult(
                correctness=0.9, correctness_reasoning="correct",
                answer_relevance=0.7, answer_relevance_reasoning="relevant",
                citation_completeness=0.5, citation_completeness_reasoning="one claim uncited",
            )

        async def score_faithfulness(self, question, context_chunks, generated_answer):
            return await self._tick(0.8, "grounded")

        async def score_info_recall(self, reference_answer, context_chunks):
            return await self._tick(0.6, "covers most")

    db = FakeDatabase([_row(44, "what happened?")])
    judge = ConcurrencyTrackingJudge()

    await _run(db, [_item("q1", "what happened?")], judge=judge)

    assert judge.peak_in_flight == 2
