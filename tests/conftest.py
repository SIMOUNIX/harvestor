"""Pytest configuration and fixtures."""

import io
import json
from pathlib import Path
from typing import Dict
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def sample_invoice_image_path(tmp_path) -> Path:
    """Provide path to sample invoice image."""
    # Use the actual test image if it exists
    actual_path = Path("data/uploads/Template1_Instance0.jpg")
    if actual_path.exists():
        return actual_path

    # Otherwise create a minimal test image
    test_image = tmp_path / "test_invoice.jpg"
    # Create a minimal valid JPEG (1x1 pixel)
    jpeg_data = bytes.fromhex(
        "ffd8ffe000104a46494600010100000100010000"
        "ffdb004300080606070605080707070909080a0c"
        "140d0c0b0b0c1912130f141d1a1f1e1d1a1c1c20"
        "2428342c202433251c1c28372c27313c3a3e3e3e"
        "252d44494438433d3e3bffdb004301090909090c"
        "0b0c180d0d1832211c213232323232323232323232"
        "32323232323232323232323232323232323232323232"
        "32323232323232323232323232ffc00011080001"
        "000103012200021101031101ffc4001500010100"
        "00000000000000000000000000000009ffc4001401"
        "0100000000000000000000000000000000ffda000c"
        "03010002110311003f00bf800ffd9"
    )
    test_image.write_bytes(jpeg_data)
    return test_image


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
                "description": "Web Development Services",
                "quantity": 40,
                "price": 150.00,
            },
            {"description": "Cloud Hosting (Monthly)", "quantity": 1, "price": 500.00},
            {"description": "SSL Certificate", "quantity": 1, "price": 100.00},
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
