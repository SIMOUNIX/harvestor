"""
Main Harvestor class for document data extraction.

This is the primary public API for Harvestor.
"""

import base64
import io
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import BinaryIO, List, Optional, Type, Union

from anthropic import Anthropic
from pydantic import BaseModel

from ..config import SUPPORTED_MODELS
from ..core.cost_tracker import cost_tracker
from ..parsers.llm_parser import LLMParser
from ..schemas.base import ExtractionResult, ExtractionStrategy, HarvestResult
from ..schemas.prompt_builder import PromptBuilder


class Harvestor:
    """
    Main document extraction class.

    Features:
    - Extract structured data from documents
    - Multiple extraction strategies (LLM)
    - Cost optimization (LLM fallback for now)
    - Batch processing support
    - Progress tracking and reporting
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "Claude Haiku 3",
        cost_limit_per_doc: float = 0.10,
        daily_cost_limit: Optional[float] = None,
    ):
        """
        Initialize Harvestor.

        Args:
            api_key: Anthropic API key (uses ANTHROPIC_API_KEY env var if not provided)
            model: LLM model to use (default: Claude Haiku for cost optimization)
            cost_limit_per_doc: Maximum cost per document (default: $0.10)
            daily_cost_limit: Optional daily cost limit
        """
        # Get API key
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Anthropic API key required. Set ANTHROPIC_API_KEY env var or pass api_key parameter."
            )

        # Resolve model name to API model ID
        if model not in SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported model: {model}. Supported models: {list(SUPPORTED_MODELS.keys())}"
            )
        self.model_name = model  # Friendly name for cost tracking
        self.model = SUPPORTED_MODELS[model]["id"]  # API model ID

        # Set cost limits
        cost_tracker.set_limits(
            daily_limit=daily_cost_limit, per_document_limit=cost_limit_per_doc
        )

        # Initialize LLM parser
        self.llm_parser = LLMParser(model=model, api_key=self.api_key)

    @staticmethod
    def get_doc_type_from_schema(schema: Type[BaseModel]) -> str:
        """
        Extract doc_type from schema class name.

        InvoiceSchema -> invoice
        IDDocumentOutput -> id_document
        CustomerReceiptData -> customer_receipt
        """
        name = schema.__name__

        # Remove common suffixes
        for suffix in ("Schema", "Output", "Data", "Model"):
            if name.endswith(suffix):
                name = name[: -len(suffix)]
                break

        # Convert CamelCase to snake_case
        # "IDDocument" -> "id_document"
        name = re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()

        return name

    def harvest_text(
        self,
        text: str,
        schema: Type[BaseModel],
        doc_type: Optional[str] = None,
        document_id: Optional[str] = None,
        language: str = "en",
    ) -> HarvestResult:
        """
        Extract structured data from text.

        This is the simplest method - just LLM extraction on provided text.

        Args:
            text: Document text to extract from
            schema: Pydantic model defining the output structure
            doc_type: Type of document (derived from schema name if not provided)
            document_id: Unique identifier for this document
            language: Document language (for future use)

        Returns:
            HarvestResult with extracted data and metadata
        """
        start_time = time.time()

        # Use provided doc_type or derive from schema
        doc_type = doc_type or self.get_doc_type_from_schema(schema)

        # Generate document ID if not provided
        if not document_id:
            document_id = f"doc_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Extract using LLM with schema
        extraction_result = self.llm_parser.extract(
            text=text, schema=schema, doc_type=doc_type, document_id=document_id
        )

        total_time = time.time() - start_time

        # Build harvest result
        return HarvestResult(
            success=extraction_result.success,
            document_id=document_id,
            document_type=doc_type,
            data=extraction_result.data,
            extraction_results=[extraction_result],
            final_strategy=extraction_result.strategy,
            final_confidence=extraction_result.confidence,
            total_cost=extraction_result.cost,
            cost_breakdown={extraction_result.strategy.value: extraction_result.cost},
            total_time=total_time,
            error=extraction_result.error,
            language=language,
        )

    def harvest_file(
        self,
        source: Union[str, Path, bytes, BinaryIO],
        schema: Type[BaseModel],
        doc_type: Optional[str] = None,
        document_id: Optional[str] = None,
        language: str = "en",
        filename: Optional[str] = None,
    ) -> HarvestResult:
        """
        Extract structured data from a file, bytes, or file-like object.

        This method accepts multiple input types for maximum flexibility:
        - str/Path: File path to read from disk
        - bytes: Raw file content as bytes
        - BinaryIO: File-like object (e.g., io.BytesIO, opened file)

        Supported formats:
        - Images (.jpg, .jpeg, .png, .gif, .webp) - uses vision API
        - Text files (.txt)
        - PDF files (.pdf) - extracts text first

        Args:
            source: File path, bytes, or file-like object
            schema: The output Pydantic BaseModel schema wanted
            doc_type: Type of document (invoice, receipt, etc.) will default the schema name in lower case
            document_id: Unique identifier (auto-generated if not provided)
            language: Document language
            filename: Original filename (used when source is bytes/file-like)

        Returns:
            HarvestResult with extracted data

        Examples:
            >>> # From file path (str or Path)
            >>> result = harvestor.harvest_file("invoice.jpg", schema)

            >>> # From bytes
            >>> with open("invoice.jpg", "rb") as f:
            ...     data = f.read()
            >>> result = harvestor.harvest_file(data, schema, filename="invoice.jpg")

            >>> # From file-like object
            >>> from io import BytesIO
            >>> buffer = BytesIO(image_data)
            >>> result = harvestor.harvest_file(buffer, schema, filename="invoice.jpg")
        """
        start_time = time.time()

        # Detect input type and normalize to bytes + metadata
        file_bytes: Optional[bytes] = None
        file_path_str: Optional[str] = None
        file_size: Optional[int] = None
        inferred_filename: Optional[str] = None

        # Use provided doc_type or resolved it from gave schema
        doc_type = doc_type or self.get_doc_type_from_schema(schema)

        if isinstance(source, (str, Path)):
            # Path-based input
            file_path = Path(source)
            file_path_str = str(file_path)
            inferred_filename = file_path.name

            if not file_path.exists():
                return HarvestResult(
                    success=False,
                    document_id=document_id or file_path.stem,
                    document_type=doc_type,
                    data={},
                    error=f"File not found: {file_path}",
                    file_path=file_path_str,
                    total_time=time.time() - start_time,
                )

            file_size = file_path.stat().st_size
            document_id = document_id or file_path.stem

            # Read file content for image processing
            with open(file_path, "rb") as f:
                file_bytes = f.read()

        elif isinstance(source, bytes):
            # Bytes input
            file_bytes = source
            file_size = len(source)
            inferred_filename = (
                filename or f"doc_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            document_id = document_id or Path(inferred_filename).stem

        elif hasattr(source, "read"):
            # File-like object (BinaryIO)
            file_bytes = source.read()
            file_size = len(file_bytes)

            # Try to get filename from file object
            if hasattr(source, "name"):
                inferred_filename = Path(source.name).name
            else:
                inferred_filename = (
                    filename or f"doc_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                )

            document_id = document_id or Path(inferred_filename).stem

        else:
            return HarvestResult(
                success=False,
                document_id=document_id or "unknown",
                document_type=doc_type,
                data={},
                error=f"Unsupported source type: {type(source)}. Use str, Path, bytes, or file-like object.",
                total_time=time.time() - start_time,
            )

        # Use provided filename or inferred one
        final_filename = filename or inferred_filename

        # Determine file type from filename
        file_extension = Path(final_filename).suffix.lower()

        try:
            # Route to appropriate extraction method based on file type
            if file_extension in [".jpg", ".jpeg", ".png", ".gif", ".webp"]:
                # Image file - use vision API
                result = self._harvest_image(
                    image_bytes=file_bytes,
                    schema=schema,
                    doc_type=doc_type,
                    document_id=document_id,
                    language=language,
                    filename=final_filename,
                )
            elif file_extension in [".txt", ".pdf"]:
                # Text-based file - extract text first
                text = self._extract_text_from_bytes(file_bytes, file_extension)
                result = self.harvest_text(
                    text=text,
                    schema=schema,
                    doc_type=doc_type,
                    document_id=document_id,
                    language=language,
                )
            else:
                return HarvestResult(
                    success=False,
                    document_id=document_id,
                    document_type=doc_type,
                    data={},
                    error=f"Unsupported file type: {file_extension}. Supported: .jpg, .jpeg, .png, .gif, .webp, .txt, .pdf",
                    file_path=file_path_str,
                    file_size_bytes=file_size,
                    total_time=time.time() - start_time,
                )

            # Add file metadata to result
            if file_path_str:
                result.file_path = file_path_str
            result.file_size_bytes = file_size

            return result

        except Exception as e:
            return HarvestResult(
                success=False,
                document_id=document_id,
                document_type=doc_type,
                data={},
                error=f"Extraction failed: {str(e)}",
                file_path=file_path_str,
                file_size_bytes=file_size,
                total_time=time.time() - start_time,
            )

    def _extract_text_from_file(self, file_path: Path) -> str:
        """
        Extract text from file based on type.

        Args:
            file_path: Path to file

        Returns:
            Extracted text

        Raises:
            ValueError: If file type is not supported
        """
        suffix = file_path.suffix.lower()

        if suffix == ".txt":
            # Plain text file
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()

        elif suffix == ".pdf":
            # PDF file - use pdfplumber for native text extraction
            try:
                import pdfplumber

                text_parts = []
                with pdfplumber.open(file_path) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text_parts.append(page_text)

                if not text_parts:
                    raise ValueError("No text found in PDF (might need OCR)")

                return "\n\n".join(text_parts)

            except ImportError:
                raise ValueError(
                    "pdfplumber not installed. Install with: pip install pdfplumber"
                )

        else:
            raise ValueError(f"Unsupported file type: {suffix}. Supported: .txt, .pdf")

    def _extract_text_from_bytes(self, file_bytes: bytes, file_extension: str) -> str:
        """
        Extract text from bytes based on file type.

        Args:
            file_bytes: Raw file content
            file_extension: File extension (e.g., '.txt', '.pdf')

        Returns:
            Extracted text

        Raises:
            ValueError: If file type is not supported
        """
        if file_extension == ".txt":
            # Plain text file
            return file_bytes.decode("utf-8")

        elif file_extension == ".pdf":
            # PDF file - use pdfplumber
            try:
                import pdfplumber

                text_parts = []
                with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text_parts.append(page_text)

                if not text_parts:
                    raise ValueError("No text found in PDF (might need OCR)")

                return "\n\n".join(text_parts)

            except ImportError:
                raise ValueError(
                    "pdfplumber not installed. Install with: pip install pdfplumber"
                )

        else:
            raise ValueError(
                f"Unsupported file type: {file_extension}. Supported: .txt, .pdf"
            )

    def _harvest_image(
        self,
        image_bytes: bytes,
        schema: Type[BaseModel],
        doc_type: str,
        document_id: Optional[str] = None,
        language: str = "en",
        filename: Optional[str] = None,
    ) -> HarvestResult:
        """
        Extract structured data from an image using Claude's vision API.

        Args:
            image_bytes: Raw image data
            schema: Pydantic model defining the output structure
            doc_type: Document type
            document_id: Document identifier
            language: Document language
            filename: Original filename for determining image type

        Returns:
            HarvestResult with extracted data
        """
        start_time = time.time()

        # Determine image media type from filename
        if filename:
            extension = Path(filename).suffix.lower().replace(".", "")
            if extension == "jpg":
                media_type = "image/jpeg"
            else:
                media_type = f"image/{extension}"
        else:
            # Default to jpeg
            media_type = "image/jpeg"

        # Encode image to base64
        image_b64 = base64.standard_b64encode(image_bytes).decode("utf-8")

        # Create extraction prompt from schema
        builder = PromptBuilder(schema)
        prompt = builder.build_vision_prompt(doc_type)

        # Initialize Anthropic client
        client = Anthropic(api_key=self.api_key)

        # Call vision API
        response = client.messages.create(
            model=self.model,
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
                                "media_type": media_type,
                                "data": image_b64,
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        )

        # Extract token usage
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens

        # Determine strategy based on model
        if "haiku" in self.model.lower():
            strategy = ExtractionStrategy.LLM_HAIKU
        elif "sonnet" in self.model.lower():
            strategy = ExtractionStrategy.LLM_SONNET
        else:
            strategy = ExtractionStrategy.LLM_HAIKU

        # Track cost
        cost = cost_tracker.track_call(
            model=self.model_name,
            strategy=strategy,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            document_id=document_id,
            success=True,
        )

        # Parse response
        response_text = response.content[0].text
        processing_time = time.time() - start_time

        try:
            # Extract JSON from response
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1

            if json_start >= 0 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                data = json.loads(json_str)
            else:
                data = json.loads(response_text)

            # Validate against schema
            validated_data = schema(**data)
            data = validated_data.model_dump()

            # Create extraction result
            extraction_result = ExtractionResult(
                success=True,
                data=data,
                raw_text=response_text[:500],
                strategy=strategy,
                confidence=0.85,
                processing_time=processing_time,
                cost=cost,
                tokens_used=input_tokens + output_tokens,
                metadata={
                    "model": self.model,
                    "media_type": media_type,
                    "vision_api": True,
                },
            )

            # Build harvest result
            return HarvestResult(
                success=True,
                document_id=document_id,
                document_type=doc_type,
                data=data,
                extraction_results=[extraction_result],
                final_strategy=strategy,
                final_confidence=0.85,
                total_cost=cost,
                cost_breakdown={strategy.value: cost},
                total_time=processing_time,
                language=language,
            )

        except json.JSONDecodeError as e:
            return HarvestResult(
                success=False,
                document_id=document_id,
                document_type=doc_type,
                data={},
                error=f"Failed to parse JSON response: {str(e)}",
                total_cost=cost,
                total_time=processing_time,
                language=language,
            )
        except Exception as e:
            return HarvestResult(
                success=False,
                document_id=document_id,
                document_type=doc_type,
                data={},
                error=f"Vision API extraction failed: {str(e)}",
                total_cost=cost,
                total_time=processing_time,
                language=language,
            )

    def harvest_batch(
        self,
        files: List[Union[str, Path]],
        schema: Type[BaseModel],
        doc_type: Optional[str] = None,
        show_progress: bool = True,
    ) -> List[HarvestResult]:
        """
        Process multiple documents.

        Args:
            files: List of file paths to process
            schema: Pydantic model defining the output structure
            doc_type: Document type for all files (derived from schema if not provided)
            show_progress: Show progress bar

        Returns:
            List of HarvestResult objects
        """
        results = []

        if show_progress:
            try:
                from tqdm import tqdm

                iterator = tqdm(files, desc="Processing documents")
            except ImportError:
                iterator = files
        else:
            iterator = files

        for file_source in iterator:
            result = self.harvest_file(
                source=file_source, schema=schema, doc_type=doc_type
            )
            results.append(result)

        return results

    def print_summary(self):
        """Print cost summary."""
        cost_tracker.print_summary()


def harvest(
    source: Union[str, Path, bytes, BinaryIO],
    schema: Type[BaseModel],
    doc_type: Optional[str] = None,
    language: str = "en",
    model: str = "Claude Haiku 3",
    api_key: Optional[str] = None,
    filename: Optional[str] = None,
) -> HarvestResult:
    """
    One-liner function for quick extraction.

    Accepts file paths, bytes, or file-like objects for maximum flexibility.

    Examples:
        ```python
        from harvestor import harvest
        from harvestor.schemas import InvoiceData

        # From file path
        result = harvest("invoice.pdf", schema=InvoiceData)
        print(f"Invoice #: {result.data.get('invoice_number')}")
        print(f"Total: ${result.data.get('total_amount')}")
        print(f"Cost: ${result.total_cost:.4f}")

        # From bytes with custom schema
        from pydantic import BaseModel, Field

        class ContractData(BaseModel):
            parties: list[str] = Field(description="Contract parties")
            value: float | None = Field(None, description="Contract value")

        with open("contract.pdf", "rb") as f:
            data = f.read()
        result = harvest(data, schema=ContractData, filename="contract.pdf")
        ```

    Args:
        source: File path, bytes, or file-like object
        schema: Pydantic model defining the output structure
        doc_type: Document type (derived from schema name if not provided)
        language: Document language
        model: LLM model to use
        api_key: API key (uses env var if not provided)
        filename: Original filename (required when source is bytes/file-like)

    Returns:
        HarvestResult with extracted data
    """
    harvestor = Harvestor(api_key=api_key, model=model)
    return harvestor.harvest_file(
        source=source,
        schema=schema,
        doc_type=doc_type,
        language=language,
        filename=filename,
    )
