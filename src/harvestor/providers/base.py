"""
Base provider abstraction for LLM providers.

Defines the interface that all LLM providers must implement.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class ModelInfo:
    """Information about an LLM model."""

    name: str
    provider: str
    model_id: str
    input_cost_per_million: float
    output_cost_per_million: float
    supports_vision: bool = False
    max_tokens: int = 4096
    context_window: int = 128000


@dataclass
class CompletionResult:
    """Unified result from an LLM completion."""

    success: bool
    content: str
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "",
        base_url: Optional[str] = None,
    ):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url

    @abstractmethod
    def complete(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.0,
    ) -> CompletionResult:
        """
        Generate a completion for the given prompt.

        Args:
            prompt: The input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 for deterministic)

        Returns:
            CompletionResult with the generated content
        """
        pass

    @abstractmethod
    def complete_vision(
        self,
        prompt: str,
        image_data: bytes,
        media_type: str = "image/jpeg",
        max_tokens: int = 2048,
        temperature: float = 0.0,
    ) -> CompletionResult:
        """
        Generate a completion for an image + prompt.

        Args:
            prompt: The input prompt
            image_data: Raw image bytes
            media_type: Image MIME type
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            CompletionResult with the generated content
        """
        pass

    @abstractmethod
    def supports_vision(self) -> bool:
        """Check if this provider/model supports vision."""
        pass

    @abstractmethod
    def get_model_info(self) -> ModelInfo:
        """Get information about the current model."""
        pass

    @classmethod
    @abstractmethod
    def get_provider_name(cls) -> str:
        """Get the provider name (e.g., 'anthropic', 'openai')."""
        pass
