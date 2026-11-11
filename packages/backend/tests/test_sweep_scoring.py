from app.services.sweep_scoring import pick_best_cycle, rank_configs, score_config

WEIGHTS = {"hit_rate": 0.5, "mean_mrr": 0.5}


def test_score_config_computes_weighted_sum():
    aggregates = {"hit_rate": 0.8, "mean_mrr": 0.6}
    assert score_config(aggregates, WEIGHTS) == 0.8 * 0.5 + 0.6 * 0.5


def test_score_config_skips_missing_metrics():
    aggregates = {"hit_rate": 0.8}
    assert score_config(aggregates, WEIGHTS) == 0.8 * 0.5


def test_rank_configs_sorts_by_weighted_score():
    cycles = [
        {"config_id": 1, "aggregates": {"hit_rate": 0.9, "mean_mrr": 0.5}},  # score 0.7
        {"config_id": 2, "aggregates": {"hit_rate": 0.5, "mean_mrr": 0.9}},  # score 0.7
        {"config_id": 3, "aggregates": {"hit_rate": 1.0, "mean_mrr": 1.0}},  # score 1.0
    ]

    ranked = rank_configs(cycles, WEIGHTS)

    assert ranked[0] == (3, 1.0)
    assert sorted(ranked[1:]) == [(1, 0.7), (2, 0.7)]


def test_pick_best_cycle_breaks_near_ties_by_hit_rate():
    cycles = [
        {"config_id": 1, "aggregates": {"hit_rate": 0.9, "mean_mrr": 0.5}},  # score 0.70
        {"config_id": 2, "aggregates": {"hit_rate": 0.95, "mean_mrr": 0.5}},  # score 0.725
    ]

    assert pick_best_cycle(cycles, WEIGHTS) == (2, 0.725)

    cycles[1]["aggregates"] = {"hit_rate": 0.85, "mean_mrr": 0.5}  # score 0.675, outside tolerance
    assert pick_best_cycle(cycles, WEIGHTS) == (1, 0.7)


def test_pick_best_cycle_breaks_ties_by_recall_then_correctness():
    base = {"hit_rate": 0.9, "mean_mrr": 0.5}
    cycles = [
        {"config_id": 1, "aggregates": {**base, "mean_retrieval_metrics": {"recall@10": 0.3}, "mean_correctness": 0.9}},
        {"config_id": 2, "aggregates": {**base, "mean_retrieval_metrics": {"recall@10": 0.4}, "mean_correctness": 0.5}},
    ]

    assert pick_best_cycle(cycles, WEIGHTS)[0] == 2


def test_pick_best_cycle_returns_none_without_cycles():
    assert pick_best_cycle([], WEIGHTS) is None
