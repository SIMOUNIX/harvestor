"""
LLM Provider implementations.

Supported providers:
- Anthropic (Claude models)
- OpenAI (GPT models)
- Ollama (local models)
"""

from typing import Optional, Type

from .anthropic import ANTHROPIC_MODELS, AnthropicProvider
from .base import BaseLLMProvider, CompletionResult, ModelInfo
from .ollama import OLLAMA_MODELS, OllamaProvider
from .openai import OPENAI_MODELS, OpenAIProvider

# Combine all models into a single registry
MODELS = {
    **{k: {**v, "provider": "anthropic"} for k, v in ANTHROPIC_MODELS.items()},
    **{k: {**v, "provider": "openai"} for k, v in OPENAI_MODELS.items()},
    **{k: {**v, "provider": "ollama"} for k, v in OLLAMA_MODELS.items()},
}

PROVIDERS: dict[str, Type[BaseLLMProvider]] = {
    "anthropic": AnthropicProvider,
    "openai": OpenAIProvider,
    "ollama": OllamaProvider,
}

DEFAULT_MODEL = "claude-haiku"


def get_provider(
    model: str,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> BaseLLMProvider:
    """
    Get the appropriate provider for a model.

    Args:
        model: Model name (e.g., 'claude-haiku', 'gpt-4o-mini', 'llama3')
        api_key: API key (not needed for Ollama)
        base_url: Optional base URL override

    Returns:
        Initialized provider instance

    Raises:
        ValueError: If model is not recognized
    """
    if model not in MODELS:
        # Check if it might be an Ollama model (allows custom local models)
        if ":" in model or model.startswith("llama") or model.startswith("mistral"):
            return OllamaProvider(model=model, base_url=base_url)
        raise ValueError(
            f"Unknown model: {model}. Available models: {list(MODELS.keys())}"
        )

    provider_name = MODELS[model]["provider"]
    provider_class = PROVIDERS[provider_name]

    return provider_class(model=model, api_key=api_key, base_url=base_url)


def list_models() -> dict[str, dict]:
    """List all available models with their info."""
    return MODELS.copy()


def list_providers() -> list[str]:
    """List all available providers."""
    return list(PROVIDERS.keys())


__all__ = [
    "BaseLLMProvider",
    "CompletionResult",
    "ModelInfo",
    "AnthropicProvider",
    "OpenAIProvider",
    "OllamaProvider",
    "MODELS",
    "PROVIDERS",
    "DEFAULT_MODEL",
    "get_provider",
    "list_models",
    "list_providers",
]
