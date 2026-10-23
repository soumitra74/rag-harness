import pytest

from app.services.sweep_budget import budget_exhausted, check_eval_budget, check_ingestion_budget


def test_budget_exhausted_false_when_under_cap_on_all_axes():
    spent = {"cost_usd": 1.0, "minutes": 5.0, "iterations": 1}
    budget = {"cost_usd": 10.0, "minutes": 60.0, "iterations": 10}
    assert budget_exhausted(spent, budget) is False


def test_budget_exhausted_true_when_at_cap_on_one_axis():
    spent = {"cost_usd": 10.0, "minutes": 5.0, "iterations": 1}
    budget = {"cost_usd": 10.0, "minutes": 60.0, "iterations": 10}
    assert budget_exhausted(spent, budget) is True


def test_budget_exhausted_true_when_over_cap_on_all_axes():
    spent = {"cost_usd": 20.0, "minutes": 120.0, "iterations": 20}
    budget = {"cost_usd": 10.0, "minutes": 60.0, "iterations": 10}
    assert budget_exhausted(spent, budget) is True


class FakeDatabase:
    def __init__(self, ingestion_spent, eval_spent):
        self._ingestion_spent = ingestion_spent
        self._eval_spent = eval_spent

    async def sum_ingestion_spend(self, sweep_id):
        return self._ingestion_spent

    async def sum_eval_spend(self, sweep_id):
        return self._eval_spent


@pytest.mark.asyncio
async def test_check_ingestion_budget_fetches_and_checks():
    db = FakeDatabase(
        ingestion_spent={"cost_usd": 10.0, "minutes": 5.0, "iterations": 1}, eval_spent={},
    )
    budget = {"cost_usd": 10.0, "minutes": 60.0, "iterations": 10}

    assert await check_ingestion_budget(db, sweep_id=1, budget=budget) is True


@pytest.mark.asyncio
async def test_check_eval_budget_fetches_and_checks():
    db = FakeDatabase(
        ingestion_spent={}, eval_spent={"cost_usd": 1.0, "minutes": 5.0, "iterations": 1},
    )
    budget = {"cost_usd": 10.0, "minutes": 60.0, "iterations": 10}

    assert await check_eval_budget(db, sweep_id=1, budget=budget) is False
