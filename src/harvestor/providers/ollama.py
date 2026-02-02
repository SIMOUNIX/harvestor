"""
Ollama provider implementation for local LLM models.
"""

import base64
import os
from typing import Optional

import httpx

from .base import BaseLLMProvider, CompletionResult, ModelInfo

OLLAMA_MODELS = {
    "llama3": {
        "id": "llama3:latest",
        "input_cost": 0.0,
        "output_cost": 0.0,
        "supports_vision": False,
        "context_window": 8192,
    },
    "llama3.2": {
        "id": "llama3.2:latest",
        "input_cost": 0.0,
        "output_cost": 0.0,
        "supports_vision": False,
        "context_window": 128000,
    },
    "mistral": {
        "id": "mistral:latest",
        "input_cost": 0.0,
        "output_cost": 0.0,
        "supports_vision": False,
        "context_window": 32000,
    },
    "llava": {
        "id": "llava:latest",
        "input_cost": 0.0,
        "output_cost": 0.0,
        "supports_vision": True,
        "context_window": 4096,
    },
    "llava-llama3": {
        "id": "llava-llama3:latest",
        "input_cost": 0.0,
        "output_cost": 0.0,
        "supports_vision": True,
        "context_window": 8192,
    },
}

DEFAULT_OLLAMA_URL = "http://localhost:11434"


class OllamaProvider(BaseLLMProvider):
    """Ollama provider for local models."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "llama3",
        base_url: Optional[str] = None,
    ):
        base_url = base_url or os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_URL)
        super().__init__(api_key=api_key, model=model, base_url=base_url)

        if model not in OLLAMA_MODELS:
            # Allow custom models not in the predefined list
            self.model_config = {
                "id": f"{model}:latest" if ":" not in model else model,
                "input_cost": 0.0,
                "output_cost": 0.0,
                "supports_vision": False,
                "context_window": 8192,
            }
        else:
            self.model_config = OLLAMA_MODELS[model]

        self.model_id = self.model_config["id"]
        self.client = httpx.Client(base_url=base_url, timeout=120.0)

    def complete(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.0,
    ) -> CompletionResult:
        try:
            response = self.client.post(
                "/api/generate",
                json={
                    "model": self.model_id,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens,
                    },
                },
            )
            response.raise_for_status()
            data = response.json()

            return CompletionResult(
                success=True,
                content=data.get("response", ""),
                input_tokens=data.get("prompt_eval_count", 0),
                output_tokens=data.get("eval_count", 0),
                model=self.model_id,
                metadata={
                    "total_duration": data.get("total_duration"),
                    "load_duration": data.get("load_duration"),
                },
            )

        except httpx.ConnectError:
            return CompletionResult(
                success=False,
                content="",
                model=self.model_id,
                error=f"Cannot connect to Ollama at {self.base_url}. Is Ollama running?",
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
                error=f"Model {self.model} does not support vision. Use 'llava' or 'llava-llama3'.",
            )

        try:
            image_b64 = base64.standard_b64encode(image_data).decode("utf-8")

            response = self.client.post(
                "/api/generate",
                json={
                    "model": self.model_id,
                    "prompt": prompt,
                    "images": [image_b64],
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens,
                    },
                },
            )
            response.raise_for_status()
            data = response.json()

            return CompletionResult(
                success=True,
                content=data.get("response", ""),
                input_tokens=data.get("prompt_eval_count", 0),
                output_tokens=data.get("eval_count", 0),
                model=self.model_id,
                metadata={
                    "total_duration": data.get("total_duration"),
                    "vision": True,
                },
            )

        except httpx.ConnectError:
            return CompletionResult(
                success=False,
                content="",
                model=self.model_id,
                error=f"Cannot connect to Ollama at {self.base_url}. Is Ollama running?",
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
            provider="ollama",
            model_id=self.model_id,
            input_cost_per_million=0.0,
            output_cost_per_million=0.0,
            supports_vision=self.model_config.get("supports_vision", False),
            context_window=self.model_config.get("context_window", 8192),
        )

    @classmethod
    def get_provider_name(cls) -> str:
        return "ollama"

    def list_local_models(self) -> list[str]:
        """List models available in the local Ollama installation."""
        try:
            response = self.client.get("/api/tags")
            response.raise_for_status()
            data = response.json()
            return [m["name"] for m in data.get("models", [])]
        except Exception:
            return []
