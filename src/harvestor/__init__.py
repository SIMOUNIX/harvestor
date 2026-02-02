"""
Harvestor - Harvest intelligence from any document

Extract structured data from any document with AI-powered extraction.
Supports multiple LLM providers: Anthropic, OpenAI, and Ollama.
"""

__version__ = "0.1.0"

import sys

if sys.version_info < (3, 10):
    raise RuntimeError("Harvestor requires Python 3.10 or higher")

from .config import SUPPORTED_MODELS
from .core.cost_tracker import cost_tracker
from .core.harvestor import Harvestor, harvest
from .providers import (
    DEFAULT_MODEL,
    MODELS,
    PROVIDERS,
    AnthropicProvider,
    BaseLLMProvider,
    CompletionResult,
    ModelInfo,
    OllamaProvider,
    OpenAIProvider,
    get_provider,
    list_models,
    list_providers,
)
from .schemas.base import (
    ExtractionResult,
    ExtractionStrategy,
    HarvestResult,
    ValidationResult,
)
from .schemas.defaults import InvoiceData, LineItem, ReceiptData

__all__ = [
    "__version__",
    # Main API
    "harvest",
    "Harvestor",
    "cost_tracker",
    # Result types
    "ExtractionResult",
    "ExtractionStrategy",
    "HarvestResult",
    "ValidationResult",
    # Output schemas
    "InvoiceData",
    "ReceiptData",
    "LineItem",
    # Config
    "SUPPORTED_MODELS",
    "MODELS",
    "DEFAULT_MODEL",
    # Providers
    "PROVIDERS",
    "BaseLLMProvider",
    "CompletionResult",
    "ModelInfo",
    "AnthropicProvider",
    "OpenAIProvider",
    "OllamaProvider",
    "get_provider",
    "list_models",
    "list_providers",
]
