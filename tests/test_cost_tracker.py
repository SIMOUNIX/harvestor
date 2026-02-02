"""Test cost tracking functionality."""

import pytest

from harvestor.core.cost_tracker import (
    CostLimitExceeded,
    CostTracker,
    cost_tracker,
)
from harvestor.schemas.base import ExtractionStrategy


class TestCostCalculation:
    """Test cost calculation for different models."""

    def setup_method(self):
        """Reset cost tracker before each test."""
        cost_tracker.reset()

    def test_calculate_haiku_cost(self):
        """Test cost calculation for Claude Haiku."""
        cost = cost_tracker.calculate_cost(
            model="claude-haiku", input_tokens=1000, output_tokens=500
        )

        # Haiku: $0.25/MTok input, $1.25/MTok output
        expected = (1000 / 1_000_000 * 0.25) + (500 / 1_000_000 * 1.25)
        assert cost == pytest.approx(expected)

    def test_calculate_sonnet_cost(self):
        """Test cost calculation for Claude Sonnet."""
        cost = cost_tracker.calculate_cost(
            model="claude-sonnet", input_tokens=1000, output_tokens=500
        )

        # Sonnet: $3/MTok input, $15/MTok output
        expected = (1000 / 1_000_000 * 3.0) + (500 / 1_000_000 * 15.0)
        assert cost == pytest.approx(expected)

    def test_unknown_model_returns_zero_cost(self):
        """Test that unknown models return zero cost (e.g., custom Ollama models)."""
        cost = cost_tracker.calculate_cost(
            model="unknown-model", input_tokens=1000, output_tokens=500
        )
        assert cost == 0.0


class TestCostTracking:
    """Test cost tracking and limits."""

    def setup_method(self):
        """Reset cost tracker before each test."""
        cost_tracker.reset()
        # Reset limits to None
        cost_tracker.set_limits(daily_limit=None, per_document_limit=10.0)

    def test_track_single_call(self):
        """Test tracking a single API call."""
        cost = cost_tracker.track_call(
            model="claude-haiku",
            strategy=ExtractionStrategy.LLM_ANTHROPIC,
            input_tokens=1000,
            output_tokens=500,
            document_id="doc1",
        )

        assert cost > 0
        stats = cost_tracker.get_stats()
        assert stats.total_calls == 1
        assert stats.total_cost == pytest.approx(cost)
        assert stats.documents_processed == 1

    def test_track_multiple_calls(self):
        """Test tracking multiple API calls."""
        for i in range(3):
            cost_tracker.track_call(
                model="claude-haiku",
                strategy=ExtractionStrategy.LLM_ANTHROPIC,
                input_tokens=1000,
                output_tokens=500,
                document_id=f"doc{i}",
            )

        stats = cost_tracker.get_stats()
        assert stats.total_calls == 3
        assert stats.documents_processed == 3

    def test_per_document_limit_enforcement(self):
        """Test that per-document cost limit is enforced."""
        cost_tracker.set_limits(per_document_limit=0.001)

        # First call should succeed
        cost_tracker.track_call(
            model="claude-haiku",
            strategy=ExtractionStrategy.LLM_ANTHROPIC,
            input_tokens=1000,
            output_tokens=500,
            document_id="doc1",
        )

        # Second call for same document would exceed limit
        with pytest.raises(CostLimitExceeded):
            cost_tracker.track_call(
                model="claude-sonnet",
                strategy=ExtractionStrategy.LLM_ANTHROPIC,
                input_tokens=10000,
                output_tokens=5000,
                document_id="doc1",
            )

    def test_daily_limit_enforcement(self):
        """Test that daily cost limit is enforced."""
        cost_tracker.set_limits(daily_limit=0.001)

        # First small call should succeed
        cost_tracker.track_call(
            model="claude-haiku",
            strategy=ExtractionStrategy.LLM_ANTHROPIC,
            input_tokens=100,
            output_tokens=50,
            document_id="doc1",
        )

        # Second call that would exceed daily limit
        with pytest.raises(CostLimitExceeded):
            cost_tracker.track_call(
                model="claude-sonnet",
                strategy=ExtractionStrategy.LLM_ANTHROPIC,
                input_tokens=100000,
                output_tokens=50000,
                document_id="doc2",
            )

    def test_get_document_cost(self):
        """Test getting cost for specific document."""
        doc_id = "test_doc"

        cost1 = cost_tracker.track_call(
            model="claude-haiku",
            strategy=ExtractionStrategy.LLM_ANTHROPIC,
            input_tokens=1000,
            output_tokens=500,
            document_id=doc_id,
        )

        cost2 = cost_tracker.track_call(
            model="claude-haiku",
            strategy=ExtractionStrategy.LLM_ANTHROPIC,
            input_tokens=500,
            output_tokens=250,
            document_id=doc_id,
        )

        total_doc_cost = cost_tracker.get_document_cost(doc_id)
        assert total_doc_cost == pytest.approx(cost1 + cost2)


class TestCostStatistics:
    """Test cost statistics and reporting."""

    def setup_method(self):
        """Reset cost tracker before each test."""
        cost_tracker.reset()
        # Reset limits to None to avoid interference between tests
        cost_tracker.set_limits(daily_limit=None, per_document_limit=10.0)

    def test_stats_by_model(self):
        """Test statistics grouped by model."""
        cost_tracker.track_call(
            model="claude-haiku",
            strategy=ExtractionStrategy.LLM_ANTHROPIC,
            input_tokens=1000,
            output_tokens=500,
            document_id="doc1",
        )

        cost_tracker.track_call(
            model="claude-sonnet",
            strategy=ExtractionStrategy.LLM_ANTHROPIC,
            input_tokens=1000,
            output_tokens=500,
            document_id="doc2",
        )

        stats = cost_tracker.get_stats()
        assert len(stats.calls_by_model) == 2
        assert "claude-haiku" in stats.calls_by_model
        assert "claude-sonnet" in stats.calls_by_model

    def test_average_cost_per_document(self):
        """Test average cost per document calculation."""
        for i in range(3):
            cost_tracker.track_call(
                model="claude-haiku",
                strategy=ExtractionStrategy.LLM_ANTHROPIC,
                input_tokens=1000,
                output_tokens=500,
                document_id=f"doc{i}",
            )

        stats = cost_tracker.get_stats()
        assert stats.avg_cost_per_doc > 0
        assert stats.avg_cost_per_doc == pytest.approx(stats.total_cost / 3)

    def test_cost_report_generation(self):
        """Test cost report generation."""
        # Track some successful calls
        cost_tracker.track_call(
            model="claude-haiku",
            strategy=ExtractionStrategy.LLM_ANTHROPIC,
            input_tokens=1000,
            output_tokens=500,
            document_id="doc1",
            success=True,
        )

        report = cost_tracker.generate_report(days=7)

        assert report.total_documents == 1
        assert report.successful_documents == 1
        assert report.failed_documents == 0
        assert report.total_cost > 0
        assert report.llm_calls == 1


class TestCostTrackerReset:
    """Test cost tracker reset functionality."""

    def test_reset_clears_all_data(self):
        """Test that reset clears all tracked data."""
        cost_tracker.track_call(
            model="claude-haiku",
            strategy=ExtractionStrategy.LLM_ANTHROPIC,
            input_tokens=1000,
            output_tokens=500,
            document_id="doc1",
        )

        stats_before = cost_tracker.get_stats()
        assert stats_before.total_calls > 0

        cost_tracker.reset()

        stats_after = cost_tracker.get_stats()
        assert stats_after.total_calls == 0
        assert stats_after.total_cost == 0
        assert stats_after.documents_processed == 0


class TestCostTrackerSingleton:
    """Test that CostTracker is a singleton."""

    def test_singleton_pattern(self):
        """Test that multiple CostTracker instances are the same object."""
        tracker1 = CostTracker()
        tracker2 = CostTracker()

        assert tracker1 is tracker2

    def test_global_instance(self):
        """Test that the global cost_tracker instance works correctly."""
        from harvestor.core.cost_tracker import cost_tracker as ct1
        from harvestor.core.cost_tracker import cost_tracker as ct2

        assert ct1 is ct2
