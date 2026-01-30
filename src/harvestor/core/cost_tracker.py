"""
Cost tracking system for API calls.

Tracks all LLM API usage, enforces limits, and generates cost reports.
"""

import json
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

from ..schemas.base import CostReport, ExtractionStrategy
from ..config import SUPPORTED_MODELS


@dataclass
class APICall:
    """Record of a single API call."""

    timestamp: datetime
    model: str
    strategy: ExtractionStrategy

    input_tokens: int
    output_tokens: int
    total_tokens: int

    cost: float

    document_id: Optional[str] = None
    success: bool = True
    error: Optional[str] = None


@dataclass
class CostTrackerStats:
    """Current statistics from the cost tracker."""

    total_calls: int
    total_cost: float
    total_tokens: int

    calls_by_model: Dict[str, int]
    cost_by_model: Dict[str, float]

    daily_cost: float
    daily_calls: int

    documents_processed: int
    avg_cost_per_doc: float


class CostTracker:
    """
    Singleton cost tracker for monitoring API usage.

    Features:
    - Track all API calls with detailed metrics
    - Enforce daily and per-document cost limits
    - Generate cost reports
    - Thread-safe for concurrent processing
    - Optional persistence to disk
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        """Ensure singleton pattern."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize cost tracker."""
        if hasattr(self, "_initialized"):
            return

        self._initialized = True
        self._calls: List[APICall] = []
        self._document_costs: Dict[str, float] = {}

        # Limits
        self.daily_limit: Optional[float] = None  # USD
        self.per_document_limit: float = 0.10  # USD

        # Persistence
        self.log_file: Optional[Path] = None

    def set_limits(
        self, daily_limit: Optional[float] = None, per_document_limit: float = 0.10
    ):
        """Set cost limits."""
        self.daily_limit = daily_limit
        self.per_document_limit = per_document_limit

    def enable_logging(self, log_file: Path):
        """Enable logging to file."""
        self.log_file = log_file
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

    def calculate_cost(
        self, model: str, input_tokens: int, output_tokens: int
    ) -> float:
        """Calculate cost for a given API call."""
        if model not in SUPPORTED_MODELS:
            # Unknown model, use conservative estimate (GPT-4 pricing)
            raise ModelNotSupported(f"Model {model} is not supported.")
        else:
            pricing = SUPPORTED_MODELS[model]

        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]

        return input_cost + output_cost

    def track_call(
        self,
        model: str,
        strategy: ExtractionStrategy,
        input_tokens: int,
        output_tokens: int,
        document_id: Optional[str] = None,
        success: bool = True,
        error: Optional[str] = None,
    ) -> float:
        """
        Track an API call and return the cost.

        Raises:
            CostLimitExceeded: If daily or per-document limit would be exceeded
        """
        cost = self.calculate_cost(model, input_tokens, output_tokens)

        # Check limits
        if self.daily_limit and self.get_daily_cost() + cost > self.daily_limit:
            raise CostLimitExceeded(
                f"Daily limit of ${self.daily_limit:.2f} would be exceeded"
            )

        if document_id:
            doc_cost = self._document_costs.get(document_id, 0.0) + cost
            if doc_cost > self.per_document_limit:
                raise CostLimitExceeded(
                    f"Per-document limit of ${self.per_document_limit:.2f} would be exceeded"
                )
            self._document_costs[document_id] = doc_cost

        # Record call
        call = APICall(
            timestamp=datetime.now(),
            model=model,
            strategy=strategy,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            cost=cost,
            document_id=document_id,
            success=success,
            error=error,
        )

        with self._lock:
            self._calls.append(call)

        # Log to file if enabled
        if self.log_file:
            self._log_call(call)

        return cost

    def _log_call(self, call: APICall):
        """Append call to log file."""
        with open(self.log_file, "a") as f:
            log_entry = {
                "timestamp": call.timestamp.isoformat(),
                "model": call.model,
                "strategy": call.strategy.value,
                "tokens": {
                    "input": call.input_tokens,
                    "output": call.output_tokens,
                    "total": call.total_tokens,
                },
                "cost": call.cost,
                "document_id": call.document_id,
                "success": call.success,
                "error": call.error,
            }
            f.write(json.dumps(log_entry) + "\n")

    def get_stats(self) -> CostTrackerStats:
        """Get current tracker statistics."""
        total_calls = len(self._calls)
        total_cost = sum(c.cost for c in self._calls)
        total_tokens = sum(c.total_tokens for c in self._calls)

        calls_by_model: Dict[str, int] = {}
        cost_by_model: Dict[str, float] = {}

        for call in self._calls:
            calls_by_model[call.model] = calls_by_model.get(call.model, 0) + 1
            cost_by_model[call.model] = cost_by_model.get(call.model, 0.0) + call.cost

        daily_cost = self.get_daily_cost()
        daily_calls = self.get_daily_calls()

        documents_processed = len(self._document_costs)
        avg_cost_per_doc = (
            total_cost / documents_processed if documents_processed > 0 else 0.0
        )

        return CostTrackerStats(
            total_calls=total_calls,
            total_cost=total_cost,
            total_tokens=total_tokens,
            calls_by_model=calls_by_model,
            cost_by_model=cost_by_model,
            daily_cost=daily_cost,
            daily_calls=daily_calls,
            documents_processed=documents_processed,
            avg_cost_per_doc=avg_cost_per_doc,
        )

    def get_daily_cost(self) -> float:
        """Get cost for the current day."""
        today = datetime.now().date()
        return sum(c.cost for c in self._calls if c.timestamp.date() == today)

    def get_daily_calls(self) -> int:
        """Get number of calls for the current day."""
        today = datetime.now().date()
        return sum(1 for c in self._calls if c.timestamp.date() == today)

    def get_document_cost(self, document_id: str) -> float:
        """Get total cost for a specific document."""
        return self._document_costs.get(document_id, 0.0)

    def generate_report(self, days: int = 7) -> CostReport:
        """
        Generate cost report for the last N days.

        Args:
            days: Number of days to include in report
        """
        cutoff = datetime.now() - timedelta(days=days)
        recent_calls = [c for c in self._calls if c.timestamp >= cutoff]

        # Get unique documents
        doc_ids = set(c.document_id for c in recent_calls if c.document_id)
        total_docs = len(doc_ids)
        successful_docs = len(
            [
                doc_id
                for doc_id in doc_ids
                if all(c.success for c in recent_calls if c.document_id == doc_id)
            ]
        )
        failed_docs = total_docs - successful_docs

        # Calculate costs
        total_cost = sum(c.cost for c in recent_calls)
        free_successes = 0
        llm_calls = len(
            [
                c
                for c in recent_calls
                if c.strategy
                in {
                    ExtractionStrategy.LLM_HAIKU,
                    ExtractionStrategy.LLM_SONNET,
                    ExtractionStrategy.LLM_GPT35,
                }
            ]
        )

        # Cost by strategy
        cost_by_strategy: Dict[str, float] = {}
        for call in recent_calls:
            strategy = call.strategy.value
            cost_by_strategy[strategy] = cost_by_strategy.get(strategy, 0.0) + call.cost

        return CostReport(
            total_documents=total_docs,
            successful_documents=successful_docs,
            failed_documents=failed_docs,
            total_cost=total_cost,
            free_method_successes=free_successes,
            llm_calls=llm_calls,
            cost_by_strategy=cost_by_strategy,
        )

    def reset(self):
        """Reset tracker (useful for testing)."""
        with self._lock:
            self._calls.clear()
            self._document_costs.clear()

    def print_summary(self):
        """Print a formatted summary of costs."""
        stats = self.get_stats()

        print("\n" + "=" * 60)
        print("📊 Harvestor Cost Summary")
        print("=" * 60)
        print(f"\n💰 Total Cost: ${stats.total_cost:.4f}")
        print(f"📞 Total Calls: {stats.total_calls}")
        print(f"🔢 Total Tokens: {stats.total_tokens:,}")
        print(f"\n📄 Documents: {stats.documents_processed}")
        print(f"💵 Avg Cost/Doc: ${stats.avg_cost_per_doc:.4f}")
        print(f"\n📅 Today: ${stats.daily_cost:.4f} ({stats.daily_calls} calls)")

        if stats.calls_by_model:
            print("\n🤖 By Model:")
            for model, count in sorted(stats.calls_by_model.items()):
                cost = stats.cost_by_model.get(model, 0.0)
                print(f"   {model}: {count} calls (${cost:.4f})")

        if self.daily_limit:
            remaining = self.daily_limit - stats.daily_cost
            pct = (stats.daily_cost / self.daily_limit) * 100
            print(f"\n⚠️  Daily Limit: ${remaining:.4f} remaining ({pct:.1f}% used)")

        print("=" * 60 + "\n")


class CostLimitExceeded(Exception):
    """Raised when a cost limit would be exceeded."""

    pass


class ModelNotSupported(Exception):
    """Raised when the used model is not supported."""

    pass


# Global singleton instance
cost_tracker = CostTracker()
