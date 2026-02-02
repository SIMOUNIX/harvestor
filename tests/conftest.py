"""Pytest configuration and fixtures."""

import io
import json
from pathlib import Path
from typing import Dict
from unittest.mock import MagicMock

import pytest

from harvestor import InvoiceData


@pytest.fixture
def sample_invoice_image_path() -> Path:
    """Provide path to sample invoice image."""
    return Path("data/uploads/keep_for_test.jpg")


@pytest.fixture
def sample_invoice_bytes(sample_invoice_image_path) -> bytes:
    """Provide sample invoice as bytes."""
    return sample_invoice_image_path.read_bytes()


@pytest.fixture
def sample_invoice_fileobj(sample_invoice_bytes) -> io.BytesIO:
    """Provide sample invoice as file-like object."""
    return io.BytesIO(sample_invoice_bytes)


@pytest.fixture
def sample_invoice_text() -> str:
    """Provide sample invoice text."""
    return """
INVOICE

Invoice Number: INV-2024-001
Date: January 15, 2024
Due Date: February 15, 2024

BILL TO:
Acme Corporation
123 Business Street
New York, NY 10001

FROM:
Tech Solutions Inc.
456 Tech Avenue
San Francisco, CA 94102

DESCRIPTION                    QUANTITY    UNIT PRICE    AMOUNT
Web Development Services           40 hrs      $150.00    $6,000.00
Cloud Hosting (Monthly)             1          $500.00      $500.00
SSL Certificate                     1          $100.00      $100.00

                                            SUBTOTAL:    $6,600.00
                                            TAX (8%):      $528.00
                                            TOTAL:       $7,128.00

Payment Terms: Net 30
"""


@pytest.fixture
def sample_invoice_data() -> Dict:
    """Provide expected invoice data."""
    return {
        "invoice_number": "INV-2024-001",
        "date": "January 15, 2024",
        "due_date": "February 15, 2024",
        "vendor_name": "Tech Solutions Inc.",
        "vendor_address": "456 Tech Avenue\nSan Francisco, CA 94102",
        "customer_name": "Acme Corporation",
        "customer_address": "123 Business Street\nNew York, NY 10001",
        "subtotal": 6600.00,
        "tax_amount": 528.00,
        "total_amount": 7128.00,
        "currency": "USD",
        "line_items": [
            {
                "name": "Web Development Services",
                "quantity": 40,
                "amount": 6000.00,
            },
            {"name": "Cloud Hosting (Monthly)", "quantity": 1, "amount": 500.00},
            {"name": "SSL Certificate", "quantity": 1, "amount": 100.00},
        ],
    }


@pytest.fixture
def mock_anthropic_response(sample_invoice_data):
    """Mock Anthropic API response."""

    class MockUsage:
        input_tokens = 1000
        output_tokens = 500

    class MockContent:
        text = json.dumps(sample_invoice_data)

    class MockResponse:
        usage = MockUsage()
        content = [MockContent()]
        stop_reason = "end_turn"

    return MockResponse()


@pytest.fixture
def mock_anthropic_client(mock_anthropic_response, monkeypatch):
    """Mock Anthropic client."""
    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_anthropic_response

    def mock_init(self, *args, **kwargs):
        return mock_client

    monkeypatch.setattr(
        "anthropic.Anthropic.__new__", lambda cls, *args, **kwargs: mock_client
    )

    return mock_client


@pytest.fixture
def api_key() -> str:
    """Provide test API key."""
    return "sk-ant-test-key-12345"


@pytest.fixture
def invoice_schema():
    """Provide the default InvoiceData schema for testing."""
    return InvoiceData


@pytest.fixture
def temp_output_dir(tmp_path) -> Path:
    """Provide temporary output directory."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return output_dir


@pytest.fixture(autouse=True)
def reset_cost_tracker():
    """Automatically reset cost tracker before each test."""
    from harvestor.core.cost_tracker import cost_tracker

    cost_tracker.reset()
    # Set reasonable default limits
    cost_tracker.set_limits(daily_limit=None, per_document_limit=10.0)
    yield
    # Clean up after test
    cost_tracker.reset()
