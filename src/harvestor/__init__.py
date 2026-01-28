"""
🌾 Harvestor - Harvest intelligence from any document

Extract structured data from any document with AI-powered extraction.
"""

__version__ = "0.1.0"

import sys

if sys.version_info < (3, 10):
    raise RuntimeError("Harvestor requires Python 3.10 or higher")

from .core.harvester import Harvester, harvest
from .core.cost_tracker import cost_tracker
from .schemas.base import (
    ExtractionResult,
    ExtractionStrategy,
    HarvestResult,
    ValidationResult,
)
from .config import SUPPORTED_MODELS

__all__ = [
    "__version__",
    "harvest",
    "Harvester",
    "cost_tracker",
    "ExtractionResult",
    "ExtractionStrategy",
    "HarvestResult",
    "ValidationResult",
    "SUPPORTED_MODELS",
]
