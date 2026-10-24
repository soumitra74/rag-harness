# USD per 1M tokens (input, output) for cloud models, keyed by (provider, model).
# Fill from the provider's current price sheet. A model with no entry is priced
# as unknown rather than free.
MODEL_PRICES_PER_1M: dict[tuple[str, str], tuple[float, float]] = {
    ("openai", "gpt-6-luna"): (0.10, 0.50),
    ("anthropic", "claude-haiku-5-5"): (0.10, 0.50),
    ("anthropic", "claude-sonnet-5"): (2.00, 10.00),
}


def token_cost_usd(
    provider: str, model: str, input_tokens: int, output_tokens: int,
    local_in_per_1k: float = 0.0, local_out_per_1k: float = 0.0,
) -> float | None:
    if provider == "ollama":
        return input_tokens / 1000 * local_in_per_1k + output_tokens / 1000 * local_out_per_1k
    price = MODEL_PRICES_PER_1M.get((provider, model))
    if price is None:
        return None
    return input_tokens / 1_000_000 * price[0] + output_tokens / 1_000_000 * price[1]
