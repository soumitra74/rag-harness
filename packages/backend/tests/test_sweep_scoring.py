from app.services.sweep_scoring import rank_configs, score_config

WEIGHTS = {"hit_rate": 0.5, "mean_mrr": 0.5}
GATE = {"hit_rate": 0.8}


def test_score_config_computes_weighted_sum():
    aggregates = {"hit_rate": 0.8, "mean_mrr": 0.6}
    assert score_config(aggregates, WEIGHTS) == 0.8 * 0.5 + 0.6 * 0.5


def test_score_config_skips_missing_metrics():
    aggregates = {"hit_rate": 0.8}
    assert score_config(aggregates, WEIGHTS) == 0.8 * 0.5


def test_rank_configs_excludes_gate_failures_and_sorts_by_score():
    cycles = [
        {"config_id": 1, "aggregates": {"hit_rate": 0.9, "mean_mrr": 0.5}},  # passes, score 0.7
        {"config_id": 2, "aggregates": {"hit_rate": 0.5, "mean_mrr": 0.9}},  # fails gate
        {"config_id": 3, "aggregates": {"hit_rate": 1.0, "mean_mrr": 1.0}},  # passes, score 1.0
    ]

    ranked = rank_configs(cycles, GATE, WEIGHTS)

    assert ranked == [(3, 1.0), (1, 0.7)]
