"""
Test script for extracting data from the invoice image using Claude's vision capabilities.

This script demonstrates:
1. Using Claude's vision API to process the invoice image
2. Extracting structured data from the invoice
3. Cost tracking and reporting

Usage:
    # First, add your Anthropic API key to .env:
    # ANTHROPIC_API_KEY=sk-ant-api03-...

    # Then run:
    uv run python test_image_invoice.py

Note: Uses Claude's vision capabilities which support image input directly.
"""

import base64
import json
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from harvestor.core.cost_tracker import cost_tracker  # noqa: E402
from harvestor.schemas.base import ExtractionStrategy  # noqa: E402

# Path to the invoice image
IMAGE_PATH = Path("data/uploads/Template1_Instance0.jpg")


def encode_image_to_base64(image_path: Path) -> str:
    """Encode image file to base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.standard_b64encode(image_file.read()).decode("utf-8")


def extract_invoice_data_from_image(image_path: Path, api_key: str = None) -> dict:
    """
    Extract structured invoice data from an image using Claude's vision API.

    Args:
        image_path: Path to the invoice image
        api_key: Anthropic API key (optional, uses env var if not provided)

    Returns:
        Dictionary containing extracted data and metadata
    """
    print(f"\n{'=' * 60}")
    print("📸 Processing Invoice Image")
    print(f"{'=' * 60}\n")
    print(f"Image: {image_path}")
    print(f"File size: {image_path.stat().st_size / 1024:.2f} KB\n")

    # Initialize Anthropic client
    client = Anthropic(api_key=api_key)

    # Read and encode the image
    print("🔄 Encoding image...")
    image_data = encode_image_to_base64(image_path)

    # Determine image type (handle .jpg -> image/jpeg mapping)
    suffix = image_path.suffix.lower().replace(".", "")
    if suffix == "jpg":
        image_type = "image/jpeg"
    else:
        image_type = f"image/{suffix}"

    # Create the prompt for structured extraction
    prompt = """Extract all structured data from this invoice image.

Return a JSON object with the following fields:
- invoice_number: The invoice number
- date: Invoice date
- due_date: Due date
- po_number: Purchase order number
- vendor_name: Vendor/seller name
- vendor_address: Vendor address
- vendor_email: Vendor email
- vendor_gstin: Vendor GSTIN (tax ID)
- customer_name: Customer/buyer name (Bill to)
- customer_address: Customer address
- customer_email: Customer email
- customer_phone: Customer phone
- gstin: Customer GSTIN
- line_items: Array of items with description, quantity, and price
- subtotal: Subtotal amount
- tax_amount: Tax amount with percentage
- discount: Discount amount if any
- total_amount: Total amount
- currency: Currency code (EUR, USD, etc.)
- bank_name: Bank name for payment
- bank_account: Bank account number
- bank_swift: Bank SWIFT/BIC code
- notes: Any special notes or instructions

Extract all available information. Return only the JSON object, no other text."""

    print("🤖 Calling Claude Vision API...")

    # Make API call with vision
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",  # Sonnet for better accuracy with vision
        max_tokens=2048,
        temperature=0.0,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": image_type,
                            "data": image_data,
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    )

    # Extract tokens and calculate cost
    input_tokens = response.usage.input_tokens
    output_tokens = response.usage.output_tokens

    # Track cost (using Sonnet pricing for vision)
    cost = cost_tracker.track_call(
        model="claude-3-5-sonnet-20241022",
        strategy=ExtractionStrategy.LLM_SONNET,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        document_id=image_path.stem,
        success=True,
    )

    # Parse the response
    response_text = response.content[0].text

    print("✅ API call completed\n")
    print("📊 API Usage:")
    print(f"   Input tokens: {input_tokens:,}")
    print(f"   Output tokens: {output_tokens:,}")
    print(f"   Total tokens: {input_tokens + output_tokens:,}")
    print(f"   Cost: ${cost:.4f}\n")

    # Extract JSON from response
    try:
        json_start = response_text.find("{")
        json_end = response_text.rfind("}") + 1

        if json_start >= 0 and json_end > json_start:
            json_str = response_text[json_start:json_end]
            data = json.loads(json_str)
        else:
            data = json.loads(response_text)

        return {
            "success": True,
            "data": data,
            "cost": cost,
            "tokens": input_tokens + output_tokens,
            "model": "claude-3-5-sonnet-20241022",
            "raw_response": response_text,
        }

    except json.JSONDecodeError as e:
        print(f"❌ Failed to parse JSON: {e}")
        print(f"\nRaw response:\n{response_text}\n")
        return {
            "success": False,
            "error": f"JSON parsing failed: {str(e)}",
            "raw_response": response_text,
        }


def print_extracted_data(result: dict):
    """Pretty print the extracted invoice data."""
    print(f"\n{'=' * 60}")
    print("📄 EXTRACTED INVOICE DATA")
    print(f"{'=' * 60}\n")

    if not result["success"]:
        print(f"❌ Extraction failed: {result.get('error', 'Unknown error')}")
        return

    data = result["data"]

    # Invoice header
    print("📋 Invoice Information:")
    print(f"   Invoice Number: {data.get('invoice_number', 'N/A')}")
    print(f"   Date: {data.get('date', 'N/A')}")
    print(f"   Due Date: {data.get('due_date', 'N/A')}")
    print(f"   PO Number: {data.get('po_number', 'N/A')}")

    # Vendor info
    print("\n🏢 Vendor:")
    print(f"   Name: {data.get('vendor_name', 'N/A')}")
    print(f"   Address: {data.get('vendor_address', 'N/A')}")
    print(f"   Email: {data.get('vendor_email', 'N/A')}")
    print(f"   GSTIN: {data.get('vendor_gstin', 'N/A')}")

    # Customer info
    print("\n👤 Customer:")
    print(f"   Name: {data.get('customer_name', 'N/A')}")
    print(f"   Address: {data.get('customer_address', 'N/A')}")
    print(f"   Email: {data.get('customer_email', 'N/A')}")
    print(f"   Phone: {data.get('customer_phone', 'N/A')}")
    print(f"   GSTIN: {data.get('gstin', 'N/A')}")

    # Line items
    line_items = data.get("line_items", [])
    if line_items:
        print(f"\n📦 Line Items ({len(line_items)}):")
        for i, item in enumerate(line_items, 1):
            print(f"   {i}. {item.get('description', 'N/A')}")
            print(f"      Quantity: {item.get('quantity', 'N/A')}")
            print(f"      Price: {item.get('price', 'N/A')}")

    # Financial summary
    print("\n💰 Financial Summary:")
    print(f"   Subtotal: {data.get('subtotal', 'N/A')}")
    print(f"   Tax: {data.get('tax_amount', 'N/A')}")
    if data.get("discount"):
        print(f"   Discount: {data.get('discount', 'N/A')}")
    print(f"   Total: {data.get('total_amount', 'N/A')} {data.get('currency', '')}")

    # Banking details
    if data.get("bank_name"):
        print("\n🏦 Banking Details:")
        print(f"   Bank: {data.get('bank_name', 'N/A')}")
        print(f"   Account: {data.get('bank_account', 'N/A')}")
        print(f"   SWIFT: {data.get('bank_swift', 'N/A')}")

    # Notes
    if data.get("notes"):
        print("\n📝 Notes:")
        print(f"   {data.get('notes', 'N/A')}")


def main():
    """Main test function."""
    print("\n" + "=" * 60)
    print("🌾 Harvestor - Invoice Image Extraction Test")
    print("=" * 60)
    print("\nCredits:")
    print("- Powered by Anthropic Claude Vision API")
    print("- Dataset: FATURA (Limam et al., 2023)")
    print("  https://doi.org/10.5281/zenodo.10371464")

    # Check if image exists
    if not IMAGE_PATH.exists():
        print(f"\n❌ Error: Image not found at {IMAGE_PATH}")
        print("Please ensure the image file exists.")
        return

    try:
        # Extract data from image
        result = extract_invoice_data_from_image(IMAGE_PATH)

        # Print results
        print_extracted_data(result)

        # Print cost summary
        cost_tracker.print_summary()

        # Save results to file
        output_file = Path("data/output/invoice_extraction_result.json")
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w") as f:
            json.dump(result, f, indent=2)

        print(f"\n💾 Results saved to: {output_file}")
        print("\n✅ Test completed successfully!\n")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
