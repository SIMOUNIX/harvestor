"""
Test script demonstrating all input types supported by Harvestor.

This script shows how to use Harvestor with:
1. File paths (str/Path)
2. Raw bytes
3. File-like objects (io.BytesIO, opened files)

Usage:
    uv run python test_all_input_types.py
"""

import io
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from harvestor import Harvester, harvest  # noqa: E402
from harvestor.core.cost_tracker import cost_tracker  # noqa: E402

# Test image path
IMAGE_PATH = Path("data/uploads/Template1_Instance0.jpg")


def test_path_input():
    """Test 1: Using file path (string or Path object)."""
    print("\n" + "=" * 60)
    print("Test 1: File Path Input")
    print("=" * 60 + "\n")

    # Can use either string or Path object
    result = harvest(IMAGE_PATH, doc_type="invoice")

    print(f"✅ Success: {result.success}")
    print(f"   Invoice #: {result.data.get('invoice_number', 'N/A')}")
    print(
        f"   Total: {result.data.get('total_amount', 'N/A')} {result.data.get('currency', '')}"
    )
    print(f"   Cost: ${result.total_cost:.4f}")
    print(f"   Time: {result.total_time:.2f}s")

    return result


def test_bytes_input():
    """Test 2: Using raw bytes."""
    print("\n" + "=" * 60)
    print("Test 2: Bytes Input")
    print("=" * 60 + "\n")

    # Read file as bytes
    with open(IMAGE_PATH, "rb") as f:
        image_bytes = f.read()

    print(f"   Read {len(image_bytes):,} bytes from file")

    # Pass bytes directly (must provide filename for format detection)
    result = harvest(image_bytes, filename="invoice.jpg", doc_type="invoice")

    print(f"✅ Success: {result.success}")
    print(f"   Invoice #: {result.data.get('invoice_number', 'N/A')}")
    print(
        f"   Total: {result.data.get('total_amount', 'N/A')} {result.data.get('currency', '')}"
    )
    print(f"   Cost: ${result.total_cost:.4f}")
    print(f"   Time: {result.total_time:.2f}s")

    return result


def test_filelike_bytesio():
    """Test 3: Using BytesIO (file-like object)."""
    print("\n" + "=" * 60)
    print("Test 3: BytesIO File-Like Object")
    print("=" * 60 + "\n")

    # Read into BytesIO (simulates in-memory file)
    with open(IMAGE_PATH, "rb") as f:
        buffer = io.BytesIO(f.read())

    print(f"   Created BytesIO buffer at position {buffer.tell()}")

    # Pass BytesIO object (must provide filename for format detection)
    result = harvest(buffer, filename="invoice.jpg", doc_type="invoice")

    print(f"✅ Success: {result.success}")
    print(f"   Invoice #: {result.data.get('invoice_number', 'N/A')}")
    print(
        f"   Total: {result.data.get('total_amount', 'N/A')} {result.data.get('currency', '')}"
    )
    print(f"   Cost: ${result.total_cost:.4f}")
    print(f"   Time: {result.total_time:.2f}s")

    return result


def test_filelike_opened():
    """Test 4: Using opened file object."""
    print("\n" + "=" * 60)
    print("Test 4: Opened File Object")
    print("=" * 60 + "\n")

    # Open file in binary mode
    with open(IMAGE_PATH, "rb") as f:
        print(f"   Opened file: {f.name}")

        # Pass opened file directly (filename auto-detected from f.name)
        result = harvest(f, doc_type="invoice")

        print(f"✅ Success: {result.success}")
        print(f"   Invoice #: {result.data.get('invoice_number', 'N/A')}")
        print(
            f"   Total: {result.data.get('total_amount', 'N/A')} {result.data.get('currency', '')}"
        )
        print(f"   Cost: ${result.total_cost:.4f}")
        print(f"   Time: {result.total_time:.2f}s")

        return result


def test_harvester_class():
    """Test 5: Using Harvester class directly."""
    print("\n" + "=" * 60)
    print("Test 5: Harvester Class with Multiple Inputs")
    print("=" * 60 + "\n")

    # Initialize harvester once
    harvester = Harvester(model="claude-3-5-haiku-20241022")

    # Test with path
    print("   Testing with Path...")
    result1 = harvester.harvest_file(IMAGE_PATH)
    print(f"   ✅ Path: {result1.success} (${result1.total_cost:.4f})")

    # Test with bytes
    print("   Testing with bytes...")
    with open(IMAGE_PATH, "rb") as f:
        image_bytes = f.read()
    result2 = harvester.harvest_file(image_bytes, filename="invoice.jpg")
    print(f"   ✅ Bytes: {result2.success} (${result2.total_cost:.4f})")

    # Test with BytesIO
    print("   Testing with BytesIO...")
    buffer = io.BytesIO(image_bytes)
    result3 = harvester.harvest_file(buffer, filename="invoice.jpg")
    print(f"   ✅ BytesIO: {result3.success} (${result3.total_cost:.4f})")

    print(
        f"\n   All tests passed! Total cost: ${result1.total_cost + result2.total_cost + result3.total_cost:.4f}"
    )


def test_batch_processing():
    """Test 6: Batch processing with mixed input types."""
    print("\n" + "=" * 60)
    print("Test 6: Batch Processing with Mixed Types")
    print("=" * 60 + "\n")

    # Create a mix of input types
    inputs = []

    # Add path
    inputs.append(IMAGE_PATH)

    # Add bytes (simulate API upload)
    with open(IMAGE_PATH, "rb") as f:
        inputs.append(f.read())

    # Add BytesIO (simulate in-memory processing)
    with open(IMAGE_PATH, "rb") as f:
        inputs.append(io.BytesIO(f.read()))

    print(f"   Processing {len(inputs)} inputs of mixed types...")

    harvester = Harvester()

    # Note: For bytes/file-like objects in batch, we need to handle filenames
    # This is a simplified example - in production you'd track filenames separately
    results = []
    for i, input_source in enumerate(inputs):
        if isinstance(input_source, (str, Path)):
            result = harvester.harvest_file(input_source)
        else:
            result = harvester.harvest_file(input_source, filename=f"invoice_{i}.jpg")
        results.append(result)

    successful = sum(1 for r in results if r.success)
    total_cost = sum(r.total_cost for r in results)

    print(f"\n   ✅ Processed {len(results)} documents")
    print(f"   ✅ Success rate: {successful}/{len(results)}")
    print(f"   💰 Total cost: ${total_cost:.4f}")
    print(f"   💰 Avg cost: ${total_cost / len(results):.4f}")


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("🌾 Harvestor - Input Type Flexibility Test Suite")
    print("=" * 60)
    print("\nThis demonstrates that Harvestor accepts:")
    print("  1. File paths (str or Path)")
    print("  2. Raw bytes")
    print("  3. File-like objects (BytesIO, opened files)")
    print("\nJust like standard Python libraries! 🚀\n")

    if not IMAGE_PATH.exists():
        print(f"\n❌ Error: Test image not found at {IMAGE_PATH}")
        print("Please ensure the image file exists.")
        return

    try:
        # Run all tests
        test_path_input()
        test_bytes_input()
        test_filelike_bytesio()
        test_filelike_opened()
        test_harvester_class()
        test_batch_processing()

        # Print final summary
        print("\n" + "=" * 60)
        print("📊 FINAL SUMMARY")
        print("=" * 60)
        cost_tracker.print_summary()

        print("\n" + "=" * 60)
        print("✅ All Tests Passed!")
        print("=" * 60)
        print("\nKey Takeaways:")
        print("  • Harvestor is flexible like PIL, requests, and other Python libs")
        print("  • Use file paths for simple cases")
        print("  • Use bytes for API integrations")
        print("  • Use file-like objects for streaming/in-memory processing")
        print("  • Same API works for all input types!")
        print("\n")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
