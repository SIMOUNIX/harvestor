# Harvestor Test Suite

Comprehensive pytest test suite for the Harvestor document extraction toolkit.

## Running Tests

### Install Test Dependencies

```bash
# Install with dev dependencies
uv sync --extra dev
```

### Run All Tests

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run with coverage report
pytest --cov=src/harvestor --cov-report=html
```

### Run Specific Test Files

```bash
# Test input types
pytest tests/test_input_types.py

# Test cost tracking
pytest tests/test_cost_tracker.py

# Test harvestor core
pytest tests/test_harvestor.py
```

### Run Specific Test Classes or Methods

```bash
# Run a specific test class
pytest tests/test_input_types.py::TestFilePathInput

# Run a specific test method
pytest tests/test_input_types.py::TestFilePathInput::test_harvest_with_string_path
```

### Run Tests by Marker

```bash
# Run only unit tests
pytest -m unit

# Run only integration tests
pytest -m integration

# Skip slow tests
pytest -m "not slow"
```

## Test Organization

### `test_input_types.py`
Tests for flexible input handling (paths, bytes, file-like objects):
- **TestFilePathInput**: Tests with string and Path objects
- **TestBytesInput**: Tests with raw bytes
- **TestFileLikeInput**: Tests with BytesIO and opened files
- **TestInputTypeEquivalence**: Ensures all input types produce same results
- **TestConvenienceFunction**: Tests the `harvest()` convenience function
- **TestImageFormatDetection**: Tests media type mapping (.jpg → image/jpeg, etc.)
- **TestErrorHandling**: Tests error scenarios

### `test_cost_tracker.py`
Tests for cost tracking and limits:
- **TestCostCalculation**: Cost calculation for different models
- **TestCostTracking**: Tracking API calls and enforcing limits
- **TestCostStatistics**: Cost reporting and statistics
- **TestCostTrackerReset**: Reset functionality
- **TestCostTrackerSingleton**: Singleton pattern verification

### `test_harvestor.py`
Tests for core Harvestor functionality:
- **TestHarvestorInitialization**: API key handling and configuration
- **TestTextExtraction**: Text extraction from various formats
- **TestBatchProcessing**: Batch document processing
- **TestDocumentIDGeneration**: Document ID handling
- **TestHarvestResult**: Result structure and properties
- **TestErrorHandling**: Error handling and recovery

## Test Fixtures

Defined in `conftest.py`:

- `sample_invoice_image_path`: Path to test invoice image
- `sample_invoice_bytes`: Invoice as raw bytes
- `sample_invoice_fileobj`: Invoice as BytesIO object
- `sample_invoice_text`: Sample invoice text
- `sample_invoice_data`: Expected extraction results
- `mock_anthropic_response`: Mocked API response
- `mock_anthropic_client`: Mocked Anthropic client
- `api_key`: Test API key
- `temp_output_dir`: Temporary directory for test outputs

## Mocking Strategy

Tests use mocking to avoid actual API calls:

```python
@patch("harvestor.core.harvestor.Anthropic")
def test_example(mock_anthropic):
    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_response
    mock_anthropic.return_value = mock_client

    # Test code here
```

This ensures:
- Fast test execution (no network calls)
- No API costs during testing
- Deterministic test results
- Tests can run offline

## Coverage Goals

Target: **>80% code coverage**

View coverage report:
```bash
pytest --cov=src/harvestor --cov-report=html
open htmlcov/index.html
```

## Writing New Tests

### Test Naming Convention

```python
class TestFeatureName:
    """Test feature description."""

    def test_specific_behavior(self):
        """Test that specific behavior works correctly."""
        # Arrange
        ...

        # Act
        ...

        # Assert
        ...
```

### Use Fixtures

```python
def test_with_fixtures(sample_invoice_bytes, api_key):
    """Use fixtures for common test data."""
    harvestor = Harvestor(api_key=api_key)
    result = harvestor.harvest_file(sample_invoice_bytes, filename="test.jpg")
    assert result.success
```

### Mock External Dependencies

```python
@patch("harvestor.core.harvestor.Anthropic")
def test_with_mock(mock_anthropic, mock_anthropic_response):
    """Mock external API calls."""
    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_anthropic_response
    mock_anthropic.return_value = mock_client

    # Test with mocked API
    ...
```

### Parametrize for Multiple Inputs

```python
@pytest.mark.parametrize("filename,expected_type", [
    ("test.jpeg", "image/jpeg"),
    ("test.png", "image/png"),
    ("test.gif", "image/gif"),
])
def test_formats(filename, expected_type):
    """Test multiple formats efficiently."""
    ...
```

## Continuous Integration

Tests should be run on:
- Every commit (pre-commit hook)
- Every pull request
- Before deployment

Example CI configuration:
```yaml
# .github/workflows/test.yml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.12'
      - run: pip install uv
      - run: uv sync --extra dev
      - run: pytest --cov
```

## Best Practices

1. **Fast Tests**: Use mocking to avoid slow operations
2. **Isolated Tests**: Each test should be independent
3. **Clear Names**: Test names should describe what they test
4. **Arrange-Act-Assert**: Follow AAA pattern
5. **One Concept**: Test one thing per test method
6. **Fixtures**: Use fixtures for common setup
7. **Parametrize**: Test multiple inputs efficiently
8. **Coverage**: Aim for high coverage, but quality over quantity

## Troubleshooting

### Import Errors

```bash
# Ensure package is installed in development mode
uv pip install -e .
```

### Fixture Not Found

Make sure fixtures are defined in `conftest.py` or the test file.

### Mock Not Working

Check that you're patching the correct import path:
```python
# Patch where it's used, not where it's defined
@patch("harvestor.core.harvestor.Anthropic")  # ✓ Correct
@patch("anthropic.Anthropic")                  # ✗ Wrong
```

### Tests Pass Locally But Fail in CI

- Check for hardcoded paths
- Ensure test data is committed
- Verify environment variables
- Check Python version compatibility

## Future Improvements

- [ ] Add integration tests with real API (optional, manual)
- [ ] Add performance benchmarks
- [ ] Add property-based tests with Hypothesis
- [ ] Add mutation testing
- [ ] Add E2E tests with real documents
