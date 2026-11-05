"""
Main Harvestor class for document data extraction.

This is the primary public API for Harvestor.
"""

import io
import re
import time
from datetime import datetime
from pathlib import Path
from typing import BinaryIO, List, Optional, Type, Union

from pydantic import BaseModel

from ..core.cost_tracker import cost_tracker
from ..parsers.llm_parser import LLMParser
from ..providers import DEFAULT_MODEL
from ..schemas.base import HarvestResult


class Harvestor:
    """
    Main document extraction class.

    Features:
    - Extract structured data from documents
    - Multi-provider support (Anthropic, OpenAI, Ollama)
    - Cost optimization
    - Batch processing support
    - Progress tracking and reporting
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        cost_limit_per_doc: float = 0.10,
        daily_cost_limit: Optional[float] = None,
        base_url: Optional[str] = None,
        validate: bool = False,
        validation_rules: Optional[List] = None,
    ):
        """
        Initialize Harvestor.

        Args:
            api_key: API key (uses env var if not provided, not needed for Ollama)
            model: Model to use (e.g., 'claude-haiku', 'gpt-4o-mini', 'llama3')
            cost_limit_per_doc: Maximum cost per document (default: $0.10)
            daily_cost_limit: Optional daily cost limit
            base_url: Optional base URL override for the provider
            validate: Run validation rules on extracted data (default: False)
            validation_rules: Custom validation rules (used with validate=True)
        """
        self.model_name = model
        self.api_key = api_key
        self.base_url = base_url

        # Set cost limits
        cost_tracker.set_limits(
            daily_limit=daily_cost_limit, per_document_limit=cost_limit_per_doc
        )

        # Initialize LLM parser (handles provider selection)
        self.llm_parser = LLMParser(model=model, api_key=api_key, base_url=base_url)

        # Initialize validation engine if enabled
        self._validate = validate
        self._validation_engine = None
        if validate:
            from ..validators import ValidationEngine

            self._validation_engine = ValidationEngine(rules=validation_rules)

    def _maybe_validate(
        self, result: HarvestResult, schema: Type[BaseModel]
    ) -> HarvestResult:
        """Run validation if enabled and extraction succeeded."""
        if self._validate and self._validation_engine and result.success:
            result.validation = self._validation_engine.validate(
                data=result.data, schema=schema
            )
        return result

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

        result = HarvestResult(
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
        return self._maybe_validate(result, schema)

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

        Accepts multiple input types:
        - str/Path: File path to read from disk
        - bytes: Raw file content as bytes
        - BinaryIO: File-like object (e.g., io.BytesIO, opened file)

        Supported formats:
        - Images (.jpg, .jpeg, .png, .gif, .webp) - uses vision API
        - Text files (.txt)
        - PDF files (.pdf) - extracts text first

        Args:
            source: File path, bytes, or file-like object
            schema: The output Pydantic BaseModel schema
            doc_type: Type of document (derived from schema name if not provided)
            document_id: Unique identifier (auto-generated if not provided)
            language: Document language
            filename: Original filename (used when source is bytes/file-like)

        Returns:
            HarvestResult with extracted data
        """
        start_time = time.time()

        # Detect input type and normalize to bytes + metadata
        file_bytes: Optional[bytes] = None
        file_path_str: Optional[str] = None
        file_size: Optional[int] = None
        inferred_filename: Optional[str] = None

        # Use provided doc_type or derive from schema
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

            with open(file_path, "rb") as f:
                file_bytes = f.read()

        elif isinstance(source, bytes):
            file_bytes = source
            file_size = len(source)
            inferred_filename = (
                filename or f"doc_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            document_id = document_id or Path(inferred_filename).stem

        elif hasattr(source, "read"):
            file_bytes = source.read()
            file_size = len(file_bytes)

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

        final_filename = filename or inferred_filename
        file_extension = Path(final_filename).suffix.lower()

        try:
            if file_extension in [".jpg", ".jpeg", ".png", ".gif", ".webp"]:
                result = self._harvest_image(
                    image_bytes=file_bytes,
                    schema=schema,
                    doc_type=doc_type,
                    document_id=document_id,
                    language=language,
                    filename=final_filename,
                )
            elif file_extension in [".txt", ".pdf"]:
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

    def _extract_text_from_bytes(self, file_bytes: bytes, file_extension: str) -> str:
        """Extract text from bytes based on file type."""
        if file_extension == ".txt":
            return file_bytes.decode("utf-8")

        elif file_extension == ".pdf":
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
        """Extract structured data from an image using vision API."""
        start_time = time.time()

        # Determine media type from filename
        if filename:
            extension = Path(filename).suffix.lower().replace(".", "")
            if extension == "jpg":
                media_type = "image/jpeg"
            else:
                media_type = f"image/{extension}"
        else:
            media_type = "image/jpeg"

        # Use LLMParser's vision extraction
        extraction_result = self.llm_parser.extract_vision(
            image_data=image_bytes,
            schema=schema,
            doc_type=doc_type,
            document_id=document_id,
            media_type=media_type,
        )

        processing_time = time.time() - start_time

        result = HarvestResult(
            success=extraction_result.success,
            document_id=document_id,
            document_type=doc_type,
            data=extraction_result.data,
            extraction_results=[extraction_result],
            final_strategy=extraction_result.strategy,
            final_confidence=extraction_result.confidence,
            total_cost=extraction_result.cost,
            cost_breakdown={extraction_result.strategy.value: extraction_result.cost},
            total_time=processing_time,
            error=extraction_result.error,
            language=language,
        )
        return self._maybe_validate(result, schema)

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
            doc_type: Document type for all files
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
    model: str = DEFAULT_MODEL,
    api_key: Optional[str] = None,
    filename: Optional[str] = None,
    base_url: Optional[str] = None,
    validate: bool = False,
    validation_rules: Optional[List] = None,
) -> HarvestResult:
    """
    One-liner function for quick extraction.

    Accepts file paths, bytes, or file-like objects.

    Examples:
        ```python
        from harvestor import harvest
        from harvestor.schemas import InvoiceData

        # From file path with default model (claude-haiku)
        result = harvest("invoice.pdf", schema=InvoiceData)
        print(f"Invoice #: {result.data.get('invoice_number')}")

        # With OpenAI
        result = harvest("invoice.jpg", schema=InvoiceData, model="gpt-4o-mini")

        # With validation
        result = harvest("invoice.pdf", schema=InvoiceData, validate=True)
        print(result.validation.fraud_risk)
        ```

    Args:
        source: File path, bytes, or file-like object
        schema: Pydantic model defining the output structure
        doc_type: Document type (derived from schema name if not provided)
        language: Document language
        model: Model to use (default: claude-haiku)
        api_key: API key (uses env var if not provided)
        filename: Original filename (required when source is bytes/file-like)
        base_url: Optional base URL override
        validate: Run validation rules on extracted data (default: False)
        validation_rules: Custom validation rules (used with validate=True)

    Returns:
        HarvestResult with extracted data
    """
    harvestor = Harvestor(
        api_key=api_key,
        model=model,
        base_url=base_url,
        validate=validate,
        validation_rules=validation_rules,
    )
    return harvestor.harvest_file(
        source=source,
        schema=schema,
        doc_type=doc_type,
        language=language,
        filename=filename,
    )
