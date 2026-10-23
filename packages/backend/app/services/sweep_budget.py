def budget_exhausted(spent: dict, budget: dict) -> bool:
    return any(spent.get(axis, 0) >= cap for axis, cap in budget.items())


async def check_ingestion_budget(db, sweep_id: int, budget: dict) -> bool:
    spent = await db.sum_ingestion_spend(sweep_id)
    return budget_exhausted(spent, budget)


async def check_eval_budget(db, sweep_id: int, budget: dict) -> bool:
    spent = await db.sum_eval_spend(sweep_id)
    return budget_exhausted(spent, budget)
