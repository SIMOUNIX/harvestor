# 🌾 Harvestor

**Harvest intelligence from any document**

AI-powered document data extraction toolkit.

## Features

- 🌍 Multi-language support (any language)
- 📄 Template-independent extraction
- 💰 Cost-optimized ($0.01-0.05 per document)
- ⚡ Fast processing (1-2s average)
- 🎯 High accuracy (95%+)
- 🐍 Python 3.10-3.13 support

## Quick Start
```bash
# Install dependencies
uv sync

# Install Tesseract
# Ubuntu: sudo apt-get install tesseract-ocr
# macOS: brew install tesseract

# Setup environment
cp .env.template .env
# Edit .env with your API keys

# Use it
python -c "from harvestor import __version__; print(__version__)"
```

## Requirements

- Python 3.10+ (3.12 recommended)
- Tesseract OCR
- Anthropic or OpenAI API key

## Citation

For testing and evaluation, we are using the following dataset:

Limam, M., et al. FATURA Dataset. Zenodo, 13 Dec. 2023, https://doi.org/10.5281/zenodo.10371464.

## License

MIT
