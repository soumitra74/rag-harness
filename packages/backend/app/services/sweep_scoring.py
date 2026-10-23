import json
from pathlib import Path

from app.services.quality_gate import QualityGateThresholds, evaluate_quality_gate

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


def rank_configs(
    cycles: list[dict], gate_thresholds: QualityGateThresholds, weights: ScoringWeights,
) -> list[tuple[int, float]]:
    scored = [
        (cycle["config_id"], score_config(cycle["aggregates"], weights))
        for cycle in cycles
        if evaluate_quality_gate(cycle["aggregates"], gate_thresholds)["status"] == "pass"
    ]
    return sorted(scored, key=lambda pair: pair[1], reverse=True)
