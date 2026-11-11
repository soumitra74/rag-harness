import json
from pathlib import Path

ScoringWeights = dict[str, float]


def load_scoring_weights(path: str) -> ScoringWeights:
    """Re-read on every call -- same reasoning as quality_gate.load_quality_gate:
    a handful of floats read from disk, cheap next to the caller's DB query."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def score_config(aggregates: dict, weights: ScoringWeights) -> float:
    return sum(
        aggregates[metric] * weight
        for metric, weight in weights.items()
        if aggregates.get(metric) is not None
    )


def rank_configs(cycles: list[dict], weights: ScoringWeights) -> list[tuple[int, float]]:
    scored = [(cycle["config_id"], score_config(cycle["aggregates"], weights)) for cycle in cycles]
    return sorted(scored, key=lambda pair: pair[1], reverse=True)


BEST_PICK_TOLERANCE = 0.02


def _tiebreak_key(aggregates: dict) -> tuple[float, float, float]:
    retrieval = aggregates.get("mean_retrieval_metrics") or {}
    return (
        aggregates.get("hit_rate") or 0.0,
        retrieval.get("recall@10") or 0.0,
        aggregates.get("mean_correctness") or 0.0,
    )


def pick_best_cycle(
    cycles: list[dict], weights: ScoringWeights, tolerance: float = BEST_PICK_TOLERANCE,
) -> tuple[int, float] | None:
    """Highest weighted score wins, except that cycles within `tolerance` of
    the top score are treated as tied and separated by hit rate, then
    recall@10, then correctness. Returns (config_id, weighted score)."""
    scored = [
        (cycle["config_id"], score_config(cycle["aggregates"], weights), cycle["aggregates"])
        for cycle in cycles
    ]
    if not scored:
        return None
    top = max(score for _, score, _ in scored)
    contenders = [entry for entry in scored if entry[1] >= top - tolerance]
    config_id, score, _ = max(contenders, key=lambda entry: _tiebreak_key(entry[2]))
    return config_id, score
