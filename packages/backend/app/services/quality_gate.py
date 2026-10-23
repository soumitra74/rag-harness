import json
from pathlib import Path

# Metric names here must match the keys _summarize() in eval_run.py puts in
# EvalJob.aggregates / the DB's stored cycle aggregates. A threshold for a
# metric that isn't present in a given cycle's aggregates is silently
# skipped (older cycles may predate a metric), rather than counted as a
# failure.
QualityGateThresholds = dict[str, float]


def load_quality_gate(path: str) -> QualityGateThresholds:
    """Re-read on every call (not cached) so editing quality_gate.json takes
    effect on the next /api/eval/history request without a backend restart.
    It's a handful of floats read from disk; the cost is negligible next to
    the DB query the caller already does."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def evaluate_quality_gate(aggregates: dict, thresholds: QualityGateThresholds) -> dict:
    """Pass/fail against a flat {metric_name: min_value} threshold map.

    Judge metrics (correctness, faithfulness, ...) swing run-to-run on pure
    LLM-judge sampling noise -- see docs/eval-run-comparisons.md -- so their
    thresholds in quality_gate.json are deliberately set as a floor well
    below the typical score, not as a tight regression band."""
    failures = [
        {"metric": metric, "value": aggregates[metric], "min": min_value}
        for metric, min_value in thresholds.items()
        if aggregates.get(metric) is not None and aggregates[metric] < min_value
    ]
    return {
        "status": "fail" if failures else "pass",
        "failures": failures,
        "summary": summarize_failures(failures),
    }


def summarize_failures(failures: list[dict]) -> str | None:
    if not failures:
        return None
    return "; ".join(
        f"{f['metric']} {f['value']:.2f} < {f['min']:.2f}" for f in failures
    )
