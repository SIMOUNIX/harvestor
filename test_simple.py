"""
Simple test script to verify Harvestor works end-to-end.

Usage:
    uv run python test_simple.py
"""

from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from .env
load_dotenv()

from harvestor import cost_tracker, harvest  # noqa: E402

# Sample invoice text
SAMPLE_INVOICE = """
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
Payment Methods: Wire Transfer, Check, Credit Card

Thank you for your business!
"""


def test_text_extraction():
    """Test extraction from text."""
    print("\n" + "=" * 60)
    print("🧪 Testing Harvestor - Text Extraction")
    print("=" * 60 + "\n")

    # Create simple text file
    test_file = Path("data/uploads/test_invoice.txt")
    test_file.parent.mkdir(parents=True, exist_ok=True)
    test_file.write_text(SAMPLE_INVOICE)

    print(f"✅ Created test invoice: {test_file}\n")

    # Extract data
    print("🔄 Extracting data with Claude Haiku...\n")

    result = harvest(file_path=test_file, doc_type="invoice")

    # Print results
    print("\n" + "=" * 60)
    print("📊 EXTRACTION RESULTS")
    print("=" * 60 + "\n")

    print(f"Success: {result.success}")
    print(f"Document ID: {result.document_id}")
    print(
        f"Strategy: {result.final_strategy.value if result.final_strategy else 'N/A'}"
    )
    print(f"Confidence: {result.final_confidence:.2%}")
    print(f"Processing Time: {result.total_time:.2f}s")
    print(f"Cost: ${result.total_cost:.4f}")
    print(f"Cost Efficiency: {result.get_cost_efficiency()}")

    if result.success:
        print("\n📄 EXTRACTED DATA:")
        print("-" * 60)
        for field, value in result.data.items():
            if value is not None:
                print(f"  {field}: {value}")

    if result.error:
        print(f"\n❌ Error: {result.error}")

    # Print cost summary
    cost_tracker.print_summary()

    print("\n✅ Test completed!\n")

    return result


if __name__ == "__main__":
    test_text_extraction()
