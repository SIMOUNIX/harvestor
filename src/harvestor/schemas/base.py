"""
Base data models for Harvestor extraction results.

These models define the structure for extraction, validation, and final harvest results.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class ExtractionStrategy(str, Enum):
    """Strategies for extracting data from documents."""

    LLM_HAIKU = "llm_haiku"  # Claude Haiku
    LLM_SONNET = "llm_sonnet"  # Claude Sonnet
    LLM_GPT35 = "llm_gpt35"  # GPT-3.5-turbo


@dataclass
class ExtractionResult:
    """Result from a single extraction attempt."""

    # Core data (required fields first)
    success: bool
    data: Dict[str, Any]  # Extracted fields
    strategy: ExtractionStrategy  # How the data was retrieved

    # Optional fields
    raw_text: Optional[str] = None
    confidence: float = 0.0  # 0.0 to 1.0
    processing_time: float = 0.0  # seconds

    # Cost tracking
    cost: float = 0.0  # USD -> for llm calls
    tokens_used: int = 0

    # Error handling
    error: Optional[str] = None
    warnings: List[str] = field(default_factory=list)

    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        """Validate confidence is between 0 and 1."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"Confidence must be between 0 and 1, got {self.confidence}"
            )

    def is_high_confidence(self, threshold: float = 0.8) -> bool:
        """Check if extraction confidence is above threshold."""
        return self.success and self.confidence >= threshold

    def is_free_method(self) -> bool:
        """Check if this strategy incurred no API costs."""
        # All current strategies are LLM-based and have costs
        return False


@dataclass
class ValidationResult:
    """Result from validation checks. Determine is the document is legit.

    Will be implemented later on.
    """

    # Core validation
    is_valid: bool
    confidence: float = 1.0  # 0.0 to 1.0 (lower = more suspicious)

    # Issues found
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    # Fraud detection (optional)
    fraud_checked: bool = False
    fraud_risk: Optional[str] = None  # "clean", "low", "medium", "high", "critical"
    fraud_reasons: List[str] = field(default_factory=list)

    # Cost tracking
    cost: float = 0.0  # Cost of validation (fraud detection)

    # Metadata
    rules_checked: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def risk_level(self) -> str:
        """Overall risk level based on validation and fraud detection."""
        if not self.is_valid:
            return "high"
        elif self.fraud_risk in ["high", "critical"]:
            return "high"
        elif self.fraud_risk == "medium" or self.confidence < 0.7:
            return "medium"
        elif len(self.warnings) > 0 or self.fraud_risk == "low":
            return "low"
        return "clean"

    def needs_manual_review(self) -> bool:
        """Check if document needs human review."""
        return self.risk_level in ["high", "medium"]


@dataclass
class HarvestResult:
    """Final result from the complete harvest pipeline."""

    # Core result
    success: bool
    document_id: str
    document_type: str

    # Extracted data
    data: Dict[str, Any]  # Final structured data

    # Extraction details
    extraction_results: List[ExtractionResult] = field(default_factory=list)
    final_strategy: Optional[ExtractionStrategy] = None
    final_confidence: float = 0.0

    # Validation
    validation: Optional[ValidationResult] = None

    # Cost analysis
    total_cost: float = 0.0
    cost_breakdown: Dict[str, float] = field(default_factory=dict)

    # Performance
    total_time: float = 0.0  # seconds

    # Error handling
    error: Optional[str] = None
    partial_result: bool = False  # True if some fields missing

    # Metadata
    file_path: Optional[str] = None
    file_size_bytes: Optional[int] = None
    language: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)

    def get_free_success_rate(self) -> float:
        """Calculate percentage of successful free extractions."""
        if not self.extraction_results:
            return 0.0

        free_successes = sum(
            1 for r in self.extraction_results if r.is_free_method() and r.success
        )
        return free_successes / len(self.extraction_results)

    def get_cost_efficiency(self) -> str:
        """Human-readable cost efficiency assessment."""
        if self.total_cost == 0.0:
            return "FREE"
        elif self.total_cost < 0.01:
            return "EXCELLENT"
        elif self.total_cost < 0.03:
            return "GOOD"
        elif self.total_cost < 0.05:
            return "ACCEPTABLE"
        else:
            return "HIGH"

    def needs_review(self) -> bool:
        """Check if result needs human review."""
        if not self.success:
            return True
        if self.partial_result:
            return True
        if self.validation and self.validation.needs_manual_review():
            return True
        if self.final_confidence < 0.7:
            return True
        return False

    def to_summary(self) -> str:
        """Generate a human-readable summary."""
        status = "SUCCESS" if self.success else "FAILED"
        strategy = self.final_strategy.value if self.final_strategy else "N/A"

        return f"""
            Harvest Result: {status}
            Document: {self.document_id} ({self.document_type})
            Strategy: {strategy}
            Confidence: {self.final_confidence:.2%}
            Cost: ${self.total_cost:.4f} ({self.get_cost_efficiency()})
            Time: {self.total_time:.2f}s
            Needs Review: {self.needs_review()}
            Data: {self.data}
        """.strip()


@dataclass
class CostReport:
    """Cost tracking report for multiple documents."""

    total_documents: int
    successful_documents: int
    failed_documents: int

    # Cost breakdown
    total_cost: float
    free_method_successes: int
    llm_calls: int

    cost_by_strategy: Dict[str, float] = field(default_factory=dict)

    # Performance
    avg_cost_per_doc: float = 0.0
    avg_time_per_doc: float = 0.0
    free_success_rate: float = 0.0

    # Comparison
    baseline_cost: float = 0.0  # What it would cost with GPT-4
    savings_percent: float = 0.0

    def __post_init__(self):
        """Calculate derived metrics."""
        if self.total_documents > 0:
            self.avg_cost_per_doc = self.total_cost / self.total_documents
            self.free_success_rate = self.free_method_successes / self.total_documents

            # Assume GPT-4 baseline at $0.50/doc
            self.baseline_cost = self.total_documents * 0.50
            if self.baseline_cost > 0:
                self.savings_percent = (
                    (self.baseline_cost - self.total_cost) / self.baseline_cost * 100
                )
