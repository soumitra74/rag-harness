from app.providers.embedding_catalog import get_max_input_tokens


def test_ollama_nomic_limit_is_its_trained_context():
    assert get_max_input_tokens("ollama", "nomic-embed-text") == 2048


def test_openai_limit_is_the_documented_input_size():
    assert get_max_input_tokens("openai", "text-embedding-3-small") == 8191


def test_unknown_model_has_no_limit():
    assert get_max_input_tokens("ollama", "some-unknown-model") is None
    assert get_max_input_tokens("nonexistent", "x") is None
