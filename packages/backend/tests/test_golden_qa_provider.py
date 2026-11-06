from app.config import Settings
from app.services import judge


def _capture_settings(monkeypatch):
    seen = []
    monkeypatch.setattr(judge, "get_completion_provider", lambda settings: seen.append(settings) or object())
    return seen


def test_golden_qa_provider_uses_configured_model_without_changing_judge(monkeypatch):
    seen = _capture_settings(monkeypatch)
    settings = Settings(eval_judge_provider="openai", eval_judge_model="gpt-6-luna",
                        golden_qa_provider="anthropic", golden_qa_model="claude-haiku-5-5")

    judge.get_golden_qa_provider(settings)

    assert seen[0].eval_judge_provider == "anthropic"
    assert seen[0].eval_judge_model == "claude-haiku-5-5"
    assert settings.eval_judge_provider == "openai"
    assert settings.eval_judge_model == "gpt-6-luna"


def test_golden_qa_provider_falls_back_to_judge_when_unset(monkeypatch):
    seen = _capture_settings(monkeypatch)
    settings = Settings(eval_judge_provider="openai", eval_judge_model="gpt-6-luna")

    judge.get_golden_qa_provider(settings)

    assert seen[0] is settings


def test_golden_qa_verifier_uses_configured_model_without_changing_judge(monkeypatch):
    seen = _capture_settings(monkeypatch)
    settings = Settings(eval_judge_provider="openai", eval_judge_model="gpt-6-luna",
                        golden_qa_verifier_provider="ollama", golden_qa_verifier_model="qwen2.5:14b")

    judge.get_golden_qa_verifier_provider(settings)

    assert seen[0].eval_judge_provider == "ollama"
    assert seen[0].eval_judge_model == "qwen2.5:14b"
    assert settings.eval_judge_provider == "openai"
    assert settings.eval_judge_model == "gpt-6-luna"


def test_golden_qa_verifier_falls_back_to_judge_when_unset(monkeypatch):
    seen = _capture_settings(monkeypatch)
    settings = Settings(eval_judge_provider="openai", eval_judge_model="gpt-6-luna")

    judge.get_golden_qa_verifier_provider(settings)

    assert seen[0] is settings
