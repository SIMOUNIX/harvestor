"""
OpenAI provider implementation.
"""

import base64
import os
from typing import Optional

from openai import OpenAI

from .base import BaseLLMProvider, CompletionResult, ModelInfo

OPENAI_MODELS = {
    "gpt-4o": {
        "id": "gpt-4o",
        "input_cost": 2.50,
        "output_cost": 10.0,
        "supports_vision": True,
        "context_window": 128000,
    },
    "gpt-4o-mini": {
        "id": "gpt-4o-mini",
        "input_cost": 0.15,
        "output_cost": 0.60,
        "supports_vision": True,
        "context_window": 128000,
    },
    "gpt-4-turbo": {
        "id": "gpt-4-turbo",
        "input_cost": 10.0,
        "output_cost": 30.0,
        "supports_vision": True,
        "context_window": 128000,
    },
    "gpt-4": {
        "id": "gpt-4",
        "input_cost": 30.0,
        "output_cost": 60.0,
        "supports_vision": False,
        "context_window": 8192,
    },
}


class OpenAIProvider(BaseLLMProvider):
    """OpenAI GPT provider."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        base_url: Optional[str] = None,
    ):
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY env var or pass api_key."
            )

        super().__init__(api_key=api_key, model=model, base_url=base_url)

        if model not in OPENAI_MODELS:
            raise ValueError(
                f"Unknown OpenAI model: {model}. "
                f"Available: {list(OPENAI_MODELS.keys())}"
            )

        self.model_config = OPENAI_MODELS[model]
        self.model_id = self.model_config["id"]
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def complete(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.0,
    ) -> CompletionResult:
        try:
            response = self.client.chat.completions.create(
                model=self.model_id,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
            )

            choice = response.choices[0]
            usage = response.usage

            return CompletionResult(
                success=True,
                content=choice.message.content or "",
                input_tokens=usage.prompt_tokens if usage else 0,
                output_tokens=usage.completion_tokens if usage else 0,
                model=self.model_id,
                metadata={"finish_reason": choice.finish_reason},
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
            data_url = f"data:{media_type};base64,{image_b64}"

            response = self.client.chat.completions.create(
                model=self.model_id,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": data_url}},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
            )

            choice = response.choices[0]
            usage = response.usage

            return CompletionResult(
                success=True,
                content=choice.message.content or "",
                input_tokens=usage.prompt_tokens if usage else 0,
                output_tokens=usage.completion_tokens if usage else 0,
                model=self.model_id,
                metadata={"finish_reason": choice.finish_reason, "vision": True},
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
            provider="openai",
            model_id=self.model_id,
            input_cost_per_million=self.model_config["input_cost"],
            output_cost_per_million=self.model_config["output_cost"],
            supports_vision=self.model_config.get("supports_vision", False),
            context_window=self.model_config.get("context_window", 128000),
        )

    @classmethod
    def get_provider_name(cls) -> str:
        return "openai"
