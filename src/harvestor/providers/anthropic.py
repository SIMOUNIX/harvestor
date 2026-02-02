"""
Anthropic Claude provider implementation.
"""

import base64
import os
from typing import Optional

from anthropic import Anthropic

from .base import BaseLLMProvider, CompletionResult, ModelInfo

ANTHROPIC_MODELS = {
    "claude-haiku": {
        "id": "claude-3-haiku-20240307",
        "input_cost": 0.25,
        "output_cost": 1.25,
        "supports_vision": True,
        "context_window": 200000,
    },
    "claude-haiku-4": {
        "id": "claude-haiku-4-5-20251001",
        "input_cost": 1.0,
        "output_cost": 5.0,
        "supports_vision": True,
        "context_window": 200000,
    },
    "claude-sonnet": {
        "id": "claude-sonnet-4-5-20250929",
        "input_cost": 3.0,
        "output_cost": 15.0,
        "supports_vision": True,
        "context_window": 200000,
    },
    "claude-sonnet-3.7": {
        "id": "claude-3-7-sonnet-20250219",
        "input_cost": 3.0,
        "output_cost": 15.0,
        "supports_vision": True,
        "context_window": 200000,
    },
    "claude-opus": {
        "id": "claude-opus-4-5-20251101",
        "input_cost": 15.0,
        "output_cost": 75.0,
        "supports_vision": True,
        "context_window": 200000,
    },
}


class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude provider."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-haiku",
        base_url: Optional[str] = None,
    ):
        api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "Anthropic API key required. Set ANTHROPIC_API_KEY env var or pass api_key."
            )

        super().__init__(api_key=api_key, model=model, base_url=base_url)

        if model not in ANTHROPIC_MODELS:
            raise ValueError(
                f"Unknown Anthropic model: {model}. "
                f"Available: {list(ANTHROPIC_MODELS.keys())}"
            )

        self.model_config = ANTHROPIC_MODELS[model]
        self.model_id = self.model_config["id"]
        self.client = Anthropic(api_key=api_key, base_url=base_url)

    def complete(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.0,
    ) -> CompletionResult:
        try:
            response = self.client.messages.create(
                model=self.model_id,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
            )

            return CompletionResult(
                success=True,
                content=response.content[0].text,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                model=self.model_id,
                metadata={"stop_reason": response.stop_reason},
            )

        except Exception as e:
            return CompletionResult(
                success=False,
                content="",
                model=self.model_id,
                error=str(e),
            )

    def complete_vision(
        self,
        prompt: str,
        image_data: bytes,
        media_type: str = "image/jpeg",
        max_tokens: int = 2048,
        temperature: float = 0.0,
    ) -> CompletionResult:
        if not self.supports_vision():
            return CompletionResult(
                success=False,
                content="",
                model=self.model_id,
                error=f"Model {self.model} does not support vision",
            )

        try:
            image_b64 = base64.standard_b64encode(image_data).decode("utf-8")

            response = self.client.messages.create(
                model=self.model_id,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": image_b64,
                                },
                            },
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
            )

            return CompletionResult(
                success=True,
                content=response.content[0].text,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                model=self.model_id,
                metadata={"stop_reason": response.stop_reason, "vision": True},
            )

        except Exception as e:
            return CompletionResult(
                success=False,
                content="",
                model=self.model_id,
                error=str(e),
            )

    def supports_vision(self) -> bool:
        return self.model_config.get("supports_vision", False)

    def get_model_info(self) -> ModelInfo:
        return ModelInfo(
            name=self.model,
            provider="anthropic",
            model_id=self.model_id,
            input_cost_per_million=self.model_config["input_cost"],
            output_cost_per_million=self.model_config["output_cost"],
            supports_vision=self.model_config.get("supports_vision", False),
            context_window=self.model_config.get("context_window", 200000),
        )

    @classmethod
    def get_provider_name(cls) -> str:
        return "anthropic"
