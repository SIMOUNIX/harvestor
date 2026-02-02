"""
Configuration for Harvestor.

Model definitions are now managed in the providers module.
This file re-exports them for backwards compatibility.
"""

from .providers import DEFAULT_MODEL, MODELS, list_models, list_providers

# Backwards compatibility alias
SUPPORTED_MODELS = MODELS

__all__ = [
    "MODELS",
    "SUPPORTED_MODELS",
    "DEFAULT_MODEL",
    "list_models",
    "list_providers",
]
