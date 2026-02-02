"""Test Harvestor class core functionality."""

from unittest.mock import MagicMock, patch

import pytest

from harvestor import Harvestor, InvoiceData
from harvestor.schemas.base import HarvestResult


class TestHarvestorInitialization:
    """Test Harvestor initialization."""

    def test_init_with_api_key(self):
        """Test initialization with explicit API key."""
        harvestor = Harvestor(api_key="sk-test-key")
        assert harvestor.llm_parser.provider.api_key == "sk-test-key"

    def test_init_with_env_api_key(self, monkeypatch):
        """Test initialization with API key from environment."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-env-key")
        harvestor = Harvestor()
        assert harvestor.llm_parser.provider.api_key == "sk-env-key"

    def test_init_without_api_key_raises_error(self, monkeypatch):
        """Test that initialization without API key raises error."""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        with pytest.raises(ValueError, match="API key required"):
            Harvestor()

    def test_init_with_custom_model(self):
        """Test initialization with custom model."""
        harvestor = Harvestor(api_key="sk-test-key", model="claude-sonnet")
        assert harvestor.model_name == "claude-sonnet"

    def test_init_sets_cost_limits(self):
        """Test that initialization sets cost limits."""
        from harvestor.core.cost_tracker import cost_tracker

        cost_tracker.reset()

        Harvestor(api_key="sk-test-key", cost_limit_per_doc=0.20, daily_cost_limit=50.0)

        assert cost_tracker.per_document_limit == 0.20
        assert cost_tracker.daily_limit == 50.0


class TestTextExtraction:
    """Test text extraction from different file types."""

    @patch("harvestor.providers.anthropic.Anthropic")
    def test_harvest_text_basic(
        self, mock_anthropic, sample_invoice_text, mock_anthropic_response, api_key
    ):
        """Test basic text harvesting."""
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_anthropic_response
        mock_anthropic.return_value = mock_client

        harvestor = Harvestor(api_key=api_key)
        result = harvestor.harvest_text(
            sample_invoice_text, schema=InvoiceData, doc_type="invoice"
        )

        assert isinstance(result, HarvestResult)
        assert result.success is True
        assert result.document_type == "invoice"
        assert result.total_cost >= 0

    def test_extract_text_from_bytes_txt(self, api_key):
        """Test text extraction from bytes (.txt)."""
        harvestor = Harvestor(api_key=api_key)
        text_bytes = "Hello, world!".encode("utf-8")

        text = harvestor._extract_text_from_bytes(text_bytes, ".txt")

        assert text == "Hello, world!"

    def test_unsupported_file_extension_raises_error(self, api_key):
        """Test that unsupported file extensions raise ValueError."""
        harvestor = Harvestor(api_key=api_key)

        with pytest.raises(ValueError, match="Unsupported file type"):
            harvestor._extract_text_from_bytes(b"content", ".xyz")


class TestBatchProcessing:
    """Test batch processing functionality."""

    @patch("harvestor.providers.anthropic.Anthropic")
    def test_harvest_batch(
        self, mock_anthropic, tmp_path, mock_anthropic_response, api_key
    ):
        """Test batch processing of multiple files."""
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_anthropic_response
        mock_anthropic.return_value = mock_client

        # Create test files
        files = []
        for i in range(3):
            file = tmp_path / f"test_{i}.jpg"
            file.write_bytes(b"fake_image_data")
            files.append(file)

        harvestor = Harvestor(api_key=api_key)
        results = harvestor.harvest_batch(
            files, schema=InvoiceData, doc_type="invoice", show_progress=False
        )

        assert len(results) == 3
        assert all(isinstance(r, HarvestResult) for r in results)
        assert all(r.success for r in results)

    @patch("harvestor.providers.anthropic.Anthropic")
    def test_harvest_batch_with_failures(self, mock_anthropic, tmp_path, api_key):
        """Test batch processing handles failures gracefully."""
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = [
            Exception("API Error"),  # First call fails
            MagicMock(
                usage=MagicMock(input_tokens=100, output_tokens=50),
                content=[MagicMock(text='{"invoice_number": "123"}')],
                stop_reason="end_turn",
            ),  # Second succeeds
        ]
        mock_anthropic.return_value = mock_client

        # Create test files
        file1 = tmp_path / "test_1.jpg"
        file1.write_bytes(b"fake_image_data")
        file2 = tmp_path / "test_2.jpg"
        file2.write_bytes(b"fake_image_data")

        harvestor = Harvestor(api_key=api_key)
        results = harvestor.harvest_batch(
            [file1, file2], schema=InvoiceData, show_progress=False
        )

        assert len(results) == 2
        assert results[0].success is False  # First failed
        assert results[1].success is True  # Second succeeded


class TestDocumentIDGeneration:
    """Test document ID generation."""

    @patch("harvestor.providers.anthropic.Anthropic")
    def test_document_id_from_filename(
        self, mock_anthropic, tmp_path, mock_anthropic_response, api_key
    ):
        """Test that document ID is generated from filename."""
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_anthropic_response
        mock_anthropic.return_value = mock_client

        file = tmp_path / "invoice_12345.jpg"
        file.write_bytes(b"fake_data")

        harvestor = Harvestor(api_key=api_key)
        result = harvestor.harvest_file(file, schema=InvoiceData)

        assert result.document_id == "invoice_12345"

    @patch("harvestor.providers.anthropic.Anthropic")
    def test_custom_document_id(
        self, mock_anthropic, tmp_path, mock_anthropic_response, api_key
    ):
        """Test providing custom document ID."""
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_anthropic_response
        mock_anthropic.return_value = mock_client

        file = tmp_path / "test.jpg"
        file.write_bytes(b"fake_data")

        harvestor = Harvestor(api_key=api_key)
        result = harvestor.harvest_file(
            file, schema=InvoiceData, document_id="custom_id_123"
        )

        assert result.document_id == "custom_id_123"


class TestHarvestResult:
    """Test HarvestResult properties and methods."""

    @patch("harvestor.providers.anthropic.Anthropic")
    def test_harvest_result_structure(
        self, mock_anthropic, tmp_path, mock_anthropic_response, api_key
    ):
        """Test that HarvestResult has all expected fields."""
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_anthropic_response
        mock_anthropic.return_value = mock_client

        file = tmp_path / "test.jpg"
        file.write_bytes(b"fake_data")

        harvestor = Harvestor(api_key=api_key)
        result = harvestor.harvest_file(file, schema=InvoiceData)

        # Check required fields
        assert hasattr(result, "success")
        assert hasattr(result, "document_id")
        assert hasattr(result, "document_type")
        assert hasattr(result, "data")
        assert hasattr(result, "total_cost")
        assert hasattr(result, "total_time")
        assert hasattr(result, "file_path")
        assert hasattr(result, "file_size_bytes")

    @patch("harvestor.providers.anthropic.Anthropic")
    def test_harvest_result_cost_efficiency(
        self, mock_anthropic, tmp_path, mock_anthropic_response, api_key
    ):
        """Test cost efficiency rating."""
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_anthropic_response
        mock_anthropic.return_value = mock_client

        file = tmp_path / "test.jpg"
        file.write_bytes(b"fake_data")

        harvestor = Harvestor(api_key=api_key)
        result = harvestor.harvest_file(file, schema=InvoiceData)

        efficiency = result.get_cost_efficiency()
        assert efficiency in ["FREE", "EXCELLENT", "GOOD", "ACCEPTABLE", "HIGH"]


class TestErrorHandling:
    """Test error handling in various scenarios."""

    def test_missing_api_key_error_message(self, monkeypatch):
        """Test that missing API key has helpful error message."""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        with pytest.raises(ValueError) as exc_info:
            Harvestor()

        assert "api key" in str(exc_info.value).lower()

    @patch("harvestor.providers.anthropic.Anthropic")
    def test_api_error_returns_failed_result(self, mock_anthropic, tmp_path, api_key):
        """Test that API errors return failed HarvestResult."""
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = Exception("API Error")
        mock_anthropic.return_value = mock_client

        file = tmp_path / "test.jpg"
        file.write_bytes(b"fake_data")

        harvestor = Harvestor(api_key=api_key)
        result = harvestor.harvest_file(file, schema=InvoiceData)

        assert result.success is False
        assert result.error is not None

    def test_nonexistent_file_returns_error(self, api_key):
        """Test that non-existent file returns error result."""
        harvestor = Harvestor(api_key=api_key)
        result = harvestor.harvest_file("nonexistent_file.jpg", schema=InvoiceData)

        assert result.success is False
        assert "not found" in result.error.lower()
