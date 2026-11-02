import pytest

from app.config import Settings
from app.services.judge import OllamaCompletionProvider, OpenAICompletionProvider, get_codegen_provider


def test_codegen_falls_back_to_judge_when_unset():
    settings = Settings(eval_judge_provider="ollama", eval_judge_model="gpt-oss:20b")
    provider = get_codegen_provider(settings)
    assert isinstance(provider, OllamaCompletionProvider)
    assert provider._model == "gpt-oss:20b"


def test_codegen_override_selects_its_own_provider_and_model():
    settings = Settings(
        eval_judge_provider="ollama", eval_judge_model="gpt-oss:20b",
        codegen_provider="openai", codegen_model="gpt-6-luna", openai_api_key="test",
    )
    provider = get_codegen_provider(settings)
    assert isinstance(provider, OpenAICompletionProvider)
    assert provider._model == "gpt-6-luna"
    assert settings.eval_judge_model == "gpt-oss:20b"


def test_anthropic_text_skips_thinking_blocks():
    from types import SimpleNamespace

    from app.providers.base import anthropic_text

    blocks = [
        SimpleNamespace(type="thinking", thinking="reasoning...", signature="sig"),
        SimpleNamespace(type="text", text="def ocr_page(p, n):\n    return 'x'"),
    ]
    assert anthropic_text(blocks) == "def ocr_page(p, n):\n    return 'x'"


def test_token_cost_uses_env_rates_only_for_ollama():
    from app.services.pricing import token_cost_usd

    assert token_cost_usd("ollama", "gpt-oss:20b", 2000, 1000, 500.0, 1000.0) == pytest.approx(2.0)


def test_token_cost_for_cloud_model_uses_price_table_and_unknown_is_none(monkeypatch):
    from app.services import pricing

    monkeypatch.setattr(pricing, "MODEL_PRICES_PER_1M", {("openai", "gpt-x"): (2.0, 8.0)})
    assert pricing.token_cost_usd("openai", "gpt-x", 1_000_000, 500_000) == 6.0
    assert pricing.token_cost_usd("openai", "unlisted", 100, 100) is None


def test_gpt_6_luna_is_priced():
    from app.services.pricing import token_cost_usd

    assert token_cost_usd("openai", "gpt-6-luna", 1_000_000, 1_000_000) == 0.60


def test_anthropic_models_are_priced():
    from app.services.pricing import token_cost_usd

    assert token_cost_usd("anthropic", "claude-haiku-5-5", 1_000_000, 1_000_000) == 0.60
    assert token_cost_usd("anthropic", "claude-sonnet-5", 1_000_000, 1_000_000) == 12.00


def test_artifact_files_round_trip(tmp_path):
    from app.services.codegen import load_artifact_files, save_artifact_file

    save_artifact_file(str(tmp_path), "ocr", "def ocr_page(p, n):\n    return ''\n")
    save_artifact_file(str(tmp_path), "graph_labels", '["PERSON", "ORGANIZATION", "METRIC"]')

    assert load_artifact_files(str(tmp_path)) == {
        "ocr": "def ocr_page(p, n):\n    return ''\n",
        "graph_labels": '["PERSON", "ORGANIZATION", "METRIC"]',
    }
    assert load_artifact_files(str(tmp_path / "missing")) == {}
