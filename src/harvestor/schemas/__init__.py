"""Data models for Harvestor."""

from .base import (
    CostReport,
    ExtractionResult,
    ExtractionStrategy,
    HarvestResult,
    ValidationResult,
)
from .defaults import InvoiceData, LineItem, ReceiptData
from .prompt_builder import PromptBuilder

__all__ = [
    # Core result types
    "ExtractionStrategy",
    "ExtractionResult",
    "ValidationResult",
    "HarvestResult",
    "CostReport",
    # Output schema support
    "PromptBuilder",
    # Default schemas
    "InvoiceData",
    "ReceiptData",
    "LineItem",
]
