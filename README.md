# 🌾 Harvestor

**AI-powered document data extraction toolkit**

Extract structured data from documents (invoices, receipts, forms) using Claude's vision API. Easily integrate into your Python applications with flexible input options and built-in cost tracking.

> ⚠️ **Early Development**: This project is in active development. Core functionality is working, but many features are still being built.

## What Works Now

- ✅ **Vision API Integration**: Extract data from images (.jpg, .png, .gif, .webp)
- ✅ **Flexible Input**: Accepts file paths, bytes, or file-like objects (like PIL, requests)
- ✅ **Cost Tracking**: Built-in monitoring and limits for API usage
- ✅ **Structured Output**: Returns Pydantic-validated data models that you can define
- 🚧 **Multi-strategy Extraction**: Cost-optimized cascade to reduce api calls (planned)

## Quick Start

```bash
# Install dependencies
uv sync

# Setup environment
cp .env.template .env
# Add your Anthropic API key to .env

# Run a test
uv run python example.py
```

## Basic Usage

```python
from harvestor import harvest

# From file path
result = harvest("invoice.jpg")
print(f"Invoice #: {result.data.get('invoice_number')}")
print(f"Total: ${result.data.get('total_amount')}")
print(f"Cost: ${result.total_cost:.4f}")

# From bytes (e.g., API upload)
with open("invoice.jpg", "rb") as f:
    data = f.read()
result = harvest(data, filename="invoice.jpg")

# From file-like object
from io import BytesIO
buffer = BytesIO(image_data)
result = harvest(buffer, filename="invoice.jpg")

# Display summary output
print(result.to_summary())
```

## Testing

```bash
# Install test dependencies
uv sync --extra dev

# Run tests
make test

# Run with coverage
make test-cov
```

## Requirements

- Python 3.13
- Anthropic API key (for Claude vision API)

## Citation

For testing and evaluation, we are using the following dataset:

> Limam, M., et al. FATURA Dataset. Zenodo, 13 Dec. 2023, https://doi.org/10.5281/zenodo.10371464.

## License

MIT
