"""Test different input types for Harvestor (path, bytes, file-like objects)."""

import io
from unittest.mock import MagicMock, patch

import pytest

from harvestor import Harvester, harvest
from harvestor.schemas.base import HarvestResult


class TestFilePathInput:
    """Test file path inputs (str and Path objects)."""

    @patch("anthropic.Anthropic")
    def test_harvest_with_string_path(
        self,
        mock_anthropic,
        sample_invoice_image_path,
        mock_anthropic_response,
        api_key,
    ):
        """Test harvesting with string path."""
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_anthropic_response
        mock_anthropic.return_value = mock_client

        harvester = Harvester(api_key=api_key)
        result = harvester.harvest_file(str(sample_invoice_image_path))

        assert isinstance(result, HarvestResult)
        assert result.success is True
        assert result.file_path == str(sample_invoice_image_path)
        assert result.file_size_bytes > 0

    @patch("anthropic.Anthropic")
    def test_harvest_with_path_object(
        self,
        mock_anthropic,
        sample_invoice_image_path,
        mock_anthropic_response,
        api_key,
    ):
        """Test harvesting with Path object."""
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_anthropic_response
        mock_anthropic.return_value = mock_client

        harvester = Harvester(api_key=api_key)
        result = harvester.harvest_file(sample_invoice_image_path)

        assert isinstance(result, HarvestResult)
        assert result.success is True
        assert result.file_path == str(sample_invoice_image_path)

    def test_harvest_with_nonexistent_path(self, api_key):
        """Test harvesting with non-existent file path."""
        harvester = Harvester(api_key=api_key)
        result = harvester.harvest_file("nonexistent_file.jpg")

        assert result.success is False
        assert "not found" in result.error.lower()


class TestBytesInput:
    """Test raw bytes input."""

    @patch("anthropic.Anthropic")
    def test_harvest_with_bytes(
        self, mock_anthropic, sample_invoice_bytes, mock_anthropic_response, api_key
    ):
        """Test harvesting with bytes input."""
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_anthropic_response
        mock_anthropic.return_value = mock_client

        harvester = Harvester(api_key=api_key)
        result = harvester.harvest_file(sample_invoice_bytes, filename="invoice.jpg")

        assert isinstance(result, HarvestResult)
        assert result.success is True
        assert result.file_size_bytes == len(sample_invoice_bytes)
        mock_client.messages.create.assert_called_once()

    @patch("anthropic.Anthropic")
    def test_harvest_with_bytes_without_filename(
        self, mock_anthropic, sample_invoice_bytes, mock_anthropic_response, api_key
    ):
        """Test that bytes without filename generates auto filename."""
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_anthropic_response
        mock_anthropic.return_value = mock_client

        harvester = Harvester(api_key=api_key)
        result = harvester.harvest_file(sample_invoice_bytes)

        # Should fail because we can't determine file type without extension
        assert result.success is False
        assert "unsupported file type" in result.error.lower()

    @patch("anthropic.Anthropic")
    def test_harvest_with_bytes_different_formats(
        self, mock_anthropic, mock_anthropic_response, api_key
    ):
        """Test bytes input with different image formats."""
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_anthropic_response
        mock_anthropic.return_value = mock_client

        harvester = Harvester(api_key=api_key)

        # Test different image formats
        formats = [".jpg", ".jpeg", ".png", ".gif", ".webp"]
        for fmt in formats:
            result = harvester.harvest_file(b"fake_image_data", filename=f"test{fmt}")
            assert result.success is True, f"Failed for format {fmt}"


class TestFileLikeInput:
    """Test file-like object inputs (BytesIO, opened files)."""

    @patch("anthropic.Anthropic")
    def test_harvest_with_bytesio(
        self, mock_anthropic, sample_invoice_bytes, mock_anthropic_response, api_key
    ):
        """Test harvesting with BytesIO object."""
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_anthropic_response
        mock_anthropic.return_value = mock_client

        buffer = io.BytesIO(sample_invoice_bytes)
        harvester = Harvester(api_key=api_key)
        result = harvester.harvest_file(buffer, filename="invoice.jpg")

        assert isinstance(result, HarvestResult)
        assert result.success is True
        assert result.file_size_bytes == len(sample_invoice_bytes)

    @patch("anthropic.Anthropic")
    def test_harvest_with_opened_file(
        self,
        mock_anthropic,
        sample_invoice_image_path,
        mock_anthropic_response,
        api_key,
    ):
        """Test harvesting with opened file object."""
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_anthropic_response
        mock_anthropic.return_value = mock_client

        harvester = Harvester(api_key=api_key)

        with open(sample_invoice_image_path, "rb") as f:
            result = harvester.harvest_file(f)

        assert isinstance(result, HarvestResult)
        assert result.success is True
        # Should auto-detect filename from f.name
        assert result.document_id is not None

    @patch("anthropic.Anthropic")
    def test_harvest_with_fileobj_without_name_attribute(
        self, mock_anthropic, sample_invoice_bytes, mock_anthropic_response, api_key
    ):
        """Test file-like object without name attribute needs filename."""
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_anthropic_response
        mock_anthropic.return_value = mock_client

        # BytesIO doesn't have .name attribute
        buffer = io.BytesIO(sample_invoice_bytes)
        harvester = Harvester(api_key=api_key)

        # Should work with explicit filename
        result = harvester.harvest_file(buffer, filename="invoice.jpg")
        assert result.success is True


class TestInputTypeEquivalence:
    """Test that all input types produce equivalent results."""

    @patch("anthropic.Anthropic")
    def test_all_input_types_produce_same_result(
        self,
        mock_anthropic,
        sample_invoice_image_path,
        mock_anthropic_response,
        api_key,
    ):
        """Test that path, bytes, and file-like inputs produce equivalent results."""
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_anthropic_response
        mock_anthropic.return_value = mock_client

        harvester = Harvester(api_key=api_key)

        # Test with path
        result_path = harvester.harvest_file(sample_invoice_image_path)

        # Test with bytes
        with open(sample_invoice_image_path, "rb") as f:
            image_bytes = f.read()
        result_bytes = harvester.harvest_file(
            image_bytes, filename=sample_invoice_image_path.name
        )

        # Test with BytesIO
        buffer = io.BytesIO(image_bytes)
        result_fileobj = harvester.harvest_file(
            buffer, filename=sample_invoice_image_path.name
        )

        # All should succeed
        assert result_path.success is True
        assert result_bytes.success is True
        assert result_fileobj.success is True

        # All should have same data
        assert result_path.data == result_bytes.data == result_fileobj.data

        # All should have same cost
        assert (
            result_path.total_cost
            == result_bytes.total_cost
            == result_fileobj.total_cost
        )


class TestConvenienceFunction:
    """Test the harvest() convenience function."""

    @patch("anthropic.Anthropic")
    def test_harvest_function_with_path(
        self,
        mock_anthropic,
        sample_invoice_image_path,
        mock_anthropic_response,
        api_key,
    ):
        """Test harvest() convenience function with path."""
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_anthropic_response
        mock_anthropic.return_value = mock_client

        result = harvest(sample_invoice_image_path, api_key=api_key)

        assert isinstance(result, HarvestResult)
        assert result.success is True

    @patch("anthropic.Anthropic")
    def test_harvest_function_with_bytes(
        self, mock_anthropic, sample_invoice_bytes, mock_anthropic_response, api_key
    ):
        """Test harvest() convenience function with bytes."""
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_anthropic_response
        mock_anthropic.return_value = mock_client

        result = harvest(sample_invoice_bytes, filename="invoice.jpg", api_key=api_key)

        assert isinstance(result, HarvestResult)
        assert result.success is True

    @patch("anthropic.Anthropic")
    def test_harvest_function_with_fileobj(
        self, mock_anthropic, sample_invoice_fileobj, mock_anthropic_response, api_key
    ):
        """Test harvest() convenience function with file-like object."""
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_anthropic_response
        mock_anthropic.return_value = mock_client

        result = harvest(
            sample_invoice_fileobj, filename="invoice.jpg", api_key=api_key
        )

        assert isinstance(result, HarvestResult)
        assert result.success is True


class TestImageFormatDetection:
    """Test image format detection and media type mapping."""

    @patch("anthropic.Anthropic")
    def test_jpg_maps_to_jpeg_mime(
        self, mock_anthropic, sample_invoice_bytes, mock_anthropic_response, api_key
    ):
        """Test that .jpg files map to image/jpeg media type."""
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_anthropic_response
        mock_anthropic.return_value = mock_client

        harvester = Harvester(api_key=api_key)
        result = harvester.harvest_file(sample_invoice_bytes, filename="test.jpg")

        assert result.success is True

        # Check that the API was called with image/jpeg media type
        call_args = mock_client.messages.create.call_args
        messages = call_args.kwargs["messages"]
        image_source = messages[0]["content"][0]["source"]
        assert image_source["media_type"] == "image/jpeg"

    @patch("anthropic.Anthropic")
    @pytest.mark.parametrize(
        "filename,expected_type",
        [
            ("test.jpeg", "image/jpeg"),
            ("test.png", "image/png"),
            ("test.gif", "image/gif"),
            ("test.webp", "image/webp"),
        ],
    )
    def test_image_format_media_types(
        self,
        mock_anthropic,
        sample_invoice_bytes,
        mock_anthropic_response,
        api_key,
        filename,
        expected_type,
    ):
        """Test that different image formats map to correct media types."""
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_anthropic_response
        mock_anthropic.return_value = mock_client

        harvester = Harvester(api_key=api_key)
        result = harvester.harvest_file(sample_invoice_bytes, filename=filename)

        assert result.success is True

        call_args = mock_client.messages.create.call_args
        messages = call_args.kwargs["messages"]
        image_source = messages[0]["content"][0]["source"]
        assert image_source["media_type"] == expected_type


class TestErrorHandling:
    """Test error handling for various input scenarios."""

    def test_unsupported_source_type(self, api_key):
        """Test error handling for unsupported source type."""
        harvester = Harvester(api_key=api_key)
        result = harvester.harvest_file(12345)  # Invalid type

        assert result.success is False
        assert "unsupported source type" in result.error.lower()

    def test_unsupported_file_format(self, api_key):
        """Test error handling for unsupported file format."""
        harvester = Harvester(api_key=api_key)
        result = harvester.harvest_file(b"data", filename="test.exe")

        assert result.success is False
        assert "unsupported file type" in result.error.lower()

    @patch("anthropic.Anthropic")
    def test_api_error_handling(self, mock_anthropic, sample_invoice_bytes, api_key):
        """Test error handling when API call fails."""
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = Exception("API Error")
        mock_anthropic.return_value = mock_client

        harvester = Harvester(api_key=api_key)
        result = harvester.harvest_file(sample_invoice_bytes, filename="test.jpg")

        assert result.success is False
        assert "extraction failed" in result.error.lower()
